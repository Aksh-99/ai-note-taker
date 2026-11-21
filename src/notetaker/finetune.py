"""LoRA fine-tuning of Whisper on the lecture clips from Step 6.1.

Why this exists:
    The baseline (reports/baseline.md) shows where off-the-shelf Whisper
    struggles: technical terms like "eigenvalue" or "Bellman" that it rarely
    heard in training. Phase 6 adapts Whisper small to lecture audio. This
    module holds all the training code; the free-GPU Colab/Kaggle notebook
    (Step 6.2b) only installs the package and calls `notetaker finetune`.

What LoRA is (the math, briefly):
    A full fine-tune updates every weight matrix W (d x k) in the model;
    for Whisper small that is 242M numbers, too much for a free GPU and very
    easy to overfit with a few hours of audio. LoRA ("Low-Rank Adaptation",
    Hu et al. 2021) freezes W and learns a correction of low rank instead:

        W_new = W + (alpha / r) * B @ A,   B is d x r, A is r x k, r << d, k

    Only A and B are trained. With r = 32 on the attention query and value
    projections that is about 1-2% of the parameters. B starts at zero, so at
    step 0 the model is exactly the pretrained Whisper and training can only
    move it gradually. The saved "adapter" is just A and B (a few MB), and it
    can later be merged back into W for fast inference (Step 6.4).

What the model is trained to do:
    Whisper is an encoder-decoder. The encoder reads an 80 x 3000 log-mel
    spectrogram (30 s of audio); the decoder writes text one token at a time.
    Training uses "teacher forcing": the decoder is shown the correct previous
    tokens and the loss is the cross-entropy of predicting each next token.
    The labels look like
        <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
        So the row picture is ... <|endoftext|>
    Padding positions get the label -100, the value PyTorch's cross-entropy
    skips (`ignore_index`), so padding never counts towards the loss.

Design notes:
    - Heavy libraries (torch, transformers, peft) are imported lazily, inside
      the functions that need them. The planning code (`plan_training`,
      `notetaker finetune --dry-run`) and the rest of the package work
      without the big install. Install it with: pip install -e ".[finetune]"
    - The training loop is written out by hand (about 60 lines) instead of
      using Hugging Face's `Seq2SeqTrainer`. That keeps every step visible
      (gradient accumulation, mixed precision, learning-rate schedule,
      gradient clipping) and lets the tests run it on a tiny random model.
    - `run_finetune` receives a `load_model` function (dependency injection,
      as in evalset.py). Real runs download Whisper from the Hugging Face Hub;
      tests pass a function that builds a 37k-parameter random Whisper, so the
      whole pipeline is tested offline on a CPU in seconds.
"""

from __future__ import annotations

import json
import math
import random
import time
import wave
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from notetaker.finetune_data import SPLITS, WHISPER_WINDOW_S, load_metadata

# Whisper's decoder has 448 positions, so a label sequence (including the
# special tokens) can be at most this long. Longer labels are dropped.
WHISPER_MAX_LABEL_TOKENS = 448

# Label value that PyTorch's cross-entropy ignores (its `ignore_index`).
IGNORE_INDEX = -100

# Whisper expects 16 kHz audio; the clips from Step 6.1 are written at this rate.
WHISPER_SAMPLE_RATE = 16_000

# finetune_data pads clips by up to 0.2 s per side, so allow a little slack
# over 30 s. Anything longer is a sign of a broken dataset, not padding.
_DURATION_SLACK_S = 0.5

# Files `run_finetune` writes, so `force=True` removes only its own output.
ADAPTER_DIRNAME = "adapter"
SUMMARY_FILENAME = "finetune.json"


class FinetuneError(Exception):
    """The fine-tuning data, settings or output folder are not usable."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class FinetuneConfig:
    """Every knob of a fine-tuning run, saved next to the adapter.

    The defaults follow the Hugging Face PEFT Whisper example and fit a free
    16 GB Colab/Kaggle T4 GPU with Whisper small in fp16.
    """

    base_model: str = "openai/whisper-small"
    language: str = "en"
    task: str = "transcribe"
    # LoRA: rank r of the correction B @ A, its scale alpha (the update is
    # multiplied by alpha / r), dropout on the LoRA input, and which layers
    # get an adapter. q_proj / v_proj are the attention query and value
    # projections in every encoder and decoder layer.
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    # Optimisation. LoRA tolerates a much larger learning rate than a full
    # fine-tune (1e-3 vs about 1e-5) because only the small A, B matrices move.
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 8
    # Gradients of this many batches are added up before one optimizer step,
    # so the "effective" batch is batch_size * grad_accum_steps without the
    # memory cost of a big batch.
    grad_accum_steps: int = 2
    max_steps: int = 1000  # optimizer steps, not batches
    warmup_steps: int = 50
    eval_every: int = 100
    max_grad_norm: float = 1.0
    # Half-precision training on a GPU (about 2x faster, half the memory).
    # Ignored on CPU, where fp16 is slow and poorly supported.
    fp16: bool = True
    seed: int = 0
    max_label_tokens: int = WHISPER_MAX_LABEL_TOKENS

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum_steps

    def validate(self) -> None:
        """Raise FinetuneError on settings that cannot train."""
        positive = ("lora_r", "lora_alpha", "batch_size", "grad_accum_steps")
        for name in positive + ("max_steps", "eval_every"):
            if getattr(self, name) < 1:
                raise FinetuneError(f"{name} must be at least 1")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise FinetuneError("lora_dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            raise FinetuneError("learning_rate must be positive")
        if self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise FinetuneError("weight_decay >= 0 and max_grad_norm > 0 required")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise FinetuneError("warmup_steps must be in [0, max_steps)")
        if not 3 <= self.max_label_tokens <= WHISPER_MAX_LABEL_TOKENS:
            raise FinetuneError(
                f"max_label_tokens must be in [3, {WHISPER_MAX_LABEL_TOKENS}]"
            )
        if not self.target_modules:
            raise FinetuneError("target_modules must name at least one layer")

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["target_modules"] = list(self.target_modules)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FinetuneConfig:
        """Build a config from a dict, rejecting unknown keys (likely typos)."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise FinetuneError(f"Unknown config keys: {', '.join(unknown)}")
        data = dict(data)
        if "target_modules" in data:
            data["target_modules"] = tuple(data["target_modules"])
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> FinetuneConfig:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FinetuneError(f"Cannot read config {path}: {error}") from error
        if not isinstance(data, dict):
            raise FinetuneError(f"{path}: expected a JSON object")
        return cls.from_json(data)


def lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    """Learning-rate factor at optimizer step `step` (0-based).

    Linear warmup from 0 to 1 over `warmup_steps`, then linear decay to 0 at
    `total_steps`; the same shape as transformers'
    `get_linear_schedule_with_warmup`. Warmup matters because Adam's running
    averages are noisy for the first steps, and a full-size step based on
    them can knock the pretrained model off course.
    """
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


# ---------------------------------------------------------------------------
# Data: reading the Step 6.1 dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Clip:
    """One training example: a WAV file and the words spoken in it."""

    path: Path
    text: str
    lecture_id: str
    duration: float


def load_clips(data_dir: str | Path, split: str) -> list[Clip]:
    """Read `<data_dir>/<split>/metadata.jsonl` and check every clip exists."""
    if split not in SPLITS:
        raise FinetuneError(f"Unknown split {split!r}; expected one of {SPLITS}")
    split_dir = Path(data_dir) / split
    if not (split_dir / "metadata.jsonl").is_file():
        raise FinetuneError(
            f"No {split_dir / 'metadata.jsonl'}. Build the dataset first with "
            "`notetaker finetune-data`."
        )
    clips: list[Clip] = []
    for number, row in enumerate(load_metadata(split_dir), start=1):
        try:
            clip = Clip(
                path=split_dir / row["file_name"],
                text=str(row["text"]).strip(),
                lecture_id=str(row["lecture_id"]),
                duration=float(row["duration"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise FinetuneError(
                f"{split_dir}/metadata.jsonl line {number}: bad row ({error})"
            ) from error
        if clip.duration > WHISPER_WINDOW_S + _DURATION_SLACK_S:
            raise FinetuneError(
                f"{clip.path.name} is {clip.duration:.1f} s, longer than "
                f"Whisper's {WHISPER_WINDOW_S:.0f} s window"
            )
        if not clip.path.is_file():
            raise FinetuneError(f"Missing audio file {clip.path}")
        clips.append(clip)
    return clips


def read_wav(path: str | Path) -> np.ndarray:
    """Read a 16 kHz mono 16-bit WAV (what Step 6.1 writes) as float32.

    Uses Python's built-in `wave` module, so no ffmpeg call per clip.
    """
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise FinetuneError(f"{path}: expected mono 16-bit audio")
        if wav_file.getframerate() != WHISPER_SAMPLE_RATE:
            raise FinetuneError(
                f"{path}: sample rate {wav_file.getframerate()} Hz, "
                f"expected {WHISPER_SAMPLE_RATE}"
            )
        raw = wav_file.readframes(wav_file.getnframes())
    # "<i2" = little-endian 16-bit integers; dividing by 32768 maps them
    # to floats in [-1, 1), the range Whisper's feature extractor expects.
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


@dataclass
class SplitStats:
    clips: int
    hours: float
    lectures: int


@dataclass
class TrainingPlan:
    """What a run will do, computed from the dataset without any model."""

    splits: dict[str, SplitStats]
    effective_batch: int
    steps_per_epoch: int
    max_steps: int

    @property
    def epochs(self) -> float:
        """How many times training will see each clip (can be fractional)."""
        return self.max_steps / self.steps_per_epoch

    def to_json(self) -> dict[str, Any]:
        return {
            "splits": {name: asdict(stats) for name, stats in self.splits.items()},
            "effective_batch": self.effective_batch,
            "steps_per_epoch": self.steps_per_epoch,
            "max_steps": self.max_steps,
            "epochs": round(self.epochs, 3),
        }


def plan_training(config: FinetuneConfig, data_dir: str | Path) -> TrainingPlan:
    """Count clips and hours per split and work out epochs. No model needed."""
    config.validate()
    splits: dict[str, SplitStats] = {}
    for split in ("train", "val"):
        clips = load_clips(data_dir, split)
        if not clips:
            raise FinetuneError(f"The {split} split has no clips")
        splits[split] = SplitStats(
            clips=len(clips),
            hours=round(sum(c.duration for c in clips) / 3600, 3),
            lectures=len({c.lecture_id for c in clips}),
        )
    # ceil: a final partial batch still counts as a step.
    steps_per_epoch = math.ceil(splits["train"].clips / config.effective_batch)
    return TrainingPlan(
        splits=splits,
        effective_batch=config.effective_batch,
        steps_per_epoch=steps_per_epoch,
        max_steps=config.max_steps,
    )


def plan_markdown(plan: TrainingPlan, config: FinetuneConfig) -> str:
    lines = [
        f"Base model: {config.base_model}  |  LoRA r={config.lora_r}, "
        f"alpha={config.lora_alpha} on {', '.join(config.target_modules)}",
        "",
        "| Split | Clips | Hours | Lectures |",
        "|---|---:|---:|---:|",
    ]
    for name, stats in plan.splits.items():
        lines.append(
            f"| {name} | {stats.clips} | {stats.hours:.2f} | {stats.lectures} |"
        )
    lines += [
        "",
        f"Effective batch {plan.effective_batch} ({config.batch_size} x "
        f"{config.grad_accum_steps} accumulation), {plan.steps_per_epoch} "
        f"steps per epoch, {plan.max_steps} steps = {plan.epochs:.1f} epochs.",
    ]
    if plan.epochs > 20:
        lines.append(
            "Warning: over 20 epochs on this little data will likely overfit; "
            "consider fewer --max-steps (watch the val loss)."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data: turning clips into model inputs
# ---------------------------------------------------------------------------


class ClipDataset:
    """Map-style dataset: `dataset[i]` gives model inputs for clip i.

    PyTorch's DataLoader only needs `__len__` and `__getitem__` (it does not
    require subclassing torch's Dataset), so this class stays torch-free.

    Labels are tokenized up front (cheap) so over-long ones can be dropped and
    counted before training starts. Audio is read lazily in `__getitem__`, so
    hours of clips never sit in memory at once.

    Args:
        clips: from `load_clips`.
        feature_extractor: turns a waveform into the 80 x 3000 log-mel input,
            e.g. transformers' WhisperFeatureExtractor. It pads or cuts every
            clip to exactly 30 s, so all inputs have the same shape.
        tokenizer: callable with `tokenizer(text).input_ids`, e.g. a
            WhisperTokenizer with language and task prefix tokens set.
        max_label_tokens: drop clips whose labels are longer than this.
        audio_loader: path -> float32 waveform (tests can swap it).
    """

    def __init__(
        self,
        clips: Sequence[Clip],
        feature_extractor: Any,
        tokenizer: Any,
        max_label_tokens: int = WHISPER_MAX_LABEL_TOKENS,
        audio_loader: Callable[[Path], np.ndarray] = read_wav,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.audio_loader = audio_loader
        self.items: list[tuple[Clip, list[int]]] = []
        self.dropped_too_long = 0
        for clip in clips:
            ids = list(tokenizer(clip.text).input_ids)
            if len(ids) > max_label_tokens:
                self.dropped_too_long += 1
                continue
            self.items.append((clip, ids))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip, ids = self.items[index]
        audio = self.audio_loader(clip.path)
        features = self.feature_extractor(
            audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="np"
        ).input_features[0]
        return {"input_features": features.astype(np.float32), "labels": ids}


@dataclass
class WhisperCollator:
    """Stack a list of examples into one padded batch of tensors.

    Args:
        decoder_start_token_id: the token the model itself puts in front of
            the decoder input (<|startoftranscript|> for Whisper).

    Why strip a token: the model builds the decoder input by shifting the
    labels one position right and inserting `decoder_start_token_id` in
    front. The tokenizer already starts the labels with that same token, so
    without stripping it the decoder would see it twice.
    """

    decoder_start_token_id: int

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        features = torch.from_numpy(np.stack([e["input_features"] for e in examples]))
        label_lists = [list(e["labels"]) for e in examples]
        if label_lists and all(
            ids and ids[0] == self.decoder_start_token_id for ids in label_lists
        ):
            label_lists = [ids[1:] for ids in label_lists]
        longest = max(len(ids) for ids in label_lists)
        labels = torch.full((len(label_lists), longest), IGNORE_INDEX, dtype=torch.long)
        for row, ids in enumerate(label_lists):
            labels[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return {"input_features": features, "labels": labels}


# ---------------------------------------------------------------------------
# Model: LoRA
# ---------------------------------------------------------------------------


def _import(name: str) -> Any:
    """Import an optional library with a helpful message if it is missing."""
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise FinetuneError(
            f"{name} is not installed. Install the fine-tuning extra with: "
            'pip install -e ".[finetune]"'
        ) from error


def apply_lora(model: Any, config: FinetuneConfig) -> Any:
    """Freeze `model` and wrap it with trainable LoRA adapters (via peft)."""
    peft = _import("peft")
    lora = peft.LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",  # keep all bias vectors frozen too
    )
    return peft.get_peft_model(model, lora)


def count_parameters(model: Any) -> tuple[int, int]:
    """(trainable, total) number of weights in a torch model."""
    trainable = total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class LogRow:
    step: int
    lr: float
    train_loss: float
    val_loss: float | None = None
    seconds: float = 0.0


@dataclass
class TrainResult:
    history: list[LogRow] = field(default_factory=list)
    best_step: int = 0
    best_val_loss: float = math.inf


def _cycle(loader: Any) -> Iterator[Any]:
    """Loop over a DataLoader forever (a new shuffle each pass)."""
    while True:
        yield from loader


def evaluate_loss(model: Any, loader: Any, device: str) -> float:
    """Mean cross-entropy per label token over a whole loader.

    The model returns the mean loss of each batch. Batches have different
    numbers of real (non-padding) tokens, so averaging batch means would give
    short batches too much weight; weighting by token count gives the true
    per-token mean over the split.
    """
    import torch

    model.eval()  # turns dropout off
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():  # no gradients needed: saves memory and time
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            tokens = int((batch["labels"] != IGNORE_INDEX).sum())
            total_loss += float(model(**batch).loss) * tokens
            total_tokens += tokens
    model.train()
    return total_loss / max(1, total_tokens)


def train_loop(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    config: FinetuneConfig,
    device: str,
    on_best: Callable[[int], None] | None = None,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """Train the trainable parameters of `model` for `config.max_steps` steps.

    One optimizer step = `grad_accum_steps` forward/backward passes whose
    gradients add up in each parameter's `.grad`. Dividing each loss by
    `grad_accum_steps` makes the sum equal the gradient of the mean loss over
    the effective batch. The validation loss is measured every `eval_every`
    steps and at the end; `on_best(step)` is called whenever it improves
    (used to save the best adapter, a form of early stopping).
    """
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # LambdaLR multiplies the base learning rate by our function of the step.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_multiplier(step, config.warmup_steps, config.max_steps),
    )
    use_amp = config.fp16 and device.startswith("cuda")
    # The GradScaler multiplies the loss before backward so small fp16
    # gradients do not round to zero, then divides them back before the step.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    result = TrainResult()
    batches = _cycle(train_loader)
    start = time.perf_counter()
    model.train()
    for step in range(1, config.max_steps + 1):
        lr = scheduler.get_last_lr()[0]
        step_loss = 0.0
        for _ in range(config.grad_accum_steps):
            batch = {k: v.to(device) for k, v in next(batches).items()}
            with torch.autocast(
                device_type="cuda" if use_amp else "cpu",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                loss = model(**batch).loss / config.grad_accum_steps
            scaler.scale(loss).backward()
            step_loss += float(loss.detach())
        scaler.unscale_(optimizer)  # so clipping sees the true gradient sizes
        # Gradient clipping: if the gradient vector is longer than
        # max_grad_norm, shrink it to that length. Guards against one bad
        # batch producing a huge update.
        torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        row = LogRow(step=step, lr=lr, train_loss=step_loss)
        if step % config.eval_every == 0 or step == config.max_steps:
            row.val_loss = evaluate_loss(model, val_loader, device)
            if row.val_loss < result.best_val_loss:
                result.best_val_loss = row.val_loss
                result.best_step = step
                if on_best is not None:
                    on_best(step)
            log(
                f"step {step}/{config.max_steps}  train loss {step_loss:.4f}  "
                f"val loss {row.val_loss:.4f}  lr {lr:.2e}"
            )
        row.seconds = round(time.perf_counter() - start, 2)
        result.history.append(row)
    return result


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------


def _load_pretrained(config: FinetuneConfig) -> tuple[Any, Any, Any]:
    """Download (first time) and load Whisper, its feature extractor and
    tokenizer from the Hugging Face Hub. Free, no account needed."""
    transformers = _import("transformers")
    processor = transformers.WhisperProcessor.from_pretrained(
        config.base_model, language=config.language, task=config.task
    )
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        config.base_model
    )
    # Make later generation (Step 6.3) transcribe English instead of
    # auto-detecting the language from the first seconds of audio.
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    return model, processor.feature_extractor, processor.tokenizer


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    torch = _import("torch")
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_finetune(
    config: FinetuneConfig,
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    device: str | None = None,
    force: bool = False,
    load_model: Callable[[FinetuneConfig], tuple[Any, Any, Any]] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fine-tune with LoRA and save the best adapter plus `finetune.json`.

    Output:
        <out_dir>/adapter/          LoRA weights (adapter_model.safetensors,
                                    adapter_config.json), a few MB
        <out_dir>/finetune.json     config, data plan, parameter counts and
                                    the loss history (for plots in Step 6.3)
    Returns the contents of finetune.json.
    """
    plan = plan_training(config, data_dir)
    out_dir = Path(out_dir)
    summary_path = out_dir / SUMMARY_FILENAME
    if summary_path.exists() and not force:
        raise FinetuneError(f"{summary_path} exists; pass force=True to replace it")
    torch = _import("torch")
    device = pick_device(device)
    if device == "cpu":
        log(
            "Warning: training on CPU. Fine for tiny tests, but Whisper small "
            "needs a GPU."
        )
    _seed_everything(config.seed)

    # Looked up at call time (not as a default argument value) so tests can
    # monkeypatch `_load_pretrained` to avoid a download.
    model, feature_extractor, tokenizer = (load_model or _load_pretrained)(config)
    model = apply_lora(model, config).to(device)
    trainable, total = count_parameters(model)
    log(f"Trainable parameters: {trainable:,} of {total:,} ({trainable / total:.2%})")

    datasets = {
        split: ClipDataset(
            load_clips(data_dir, split),
            feature_extractor,
            tokenizer,
            max_label_tokens=config.max_label_tokens,
        )
        for split in ("train", "val")
    }
    for split, dataset in datasets.items():
        if len(dataset) == 0:
            raise FinetuneError(f"Every {split} clip has labels over the token limit")
    collate = WhisperCollator(model.config.decoder_start_token_id)
    # A seeded generator makes the shuffle order reproducible.
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = torch.utils.data.DataLoader(
        datasets["train"],
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets["val"], batch_size=config.batch_size, collate_fn=collate
    )

    adapter_dir = out_dir / ADAPTER_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_best(step: int) -> None:
        # For a peft model, save_pretrained writes only the LoRA weights.
        model.save_pretrained(str(adapter_dir))

    result = train_loop(
        model, train_loader, val_loader, config, device, on_best=save_best, log=log
    )
    summary = {
        "config": config.to_json(),
        "plan": plan.to_json(),
        "device": device,
        "parameters": {"trainable": trainable, "total": total},
        "dropped_too_long": {s: d.dropped_too_long for s, d in datasets.items()},
        "best_step": result.best_step,
        "best_val_loss": round(result.best_val_loss, 5),
        "history": [asdict(row) for row in result.history],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(
        f"Best val loss {result.best_val_loss:.4f} at step {result.best_step}; "
        f"adapter saved to {adapter_dir}"
    )
    return summary
