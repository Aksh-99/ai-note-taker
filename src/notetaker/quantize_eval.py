"""Speed / accuracy / size tradeoff across model precisions (Step 6.4c).

The question this answers:
    `notetaker export-ct2` (Step 6.4a) can store the fine-tuned Whisper as
    float32, float16 or int8. Smaller numbers mean a smaller file and faster
    CPU inference, but every weight is rounded. How much accuracy does that
    rounding cost, and how much speed and disk does it buy?

What is measured, per model, on the same held-out clips:
    - accuracy: pooled WER / CER and technical-term recall, exactly as in the
      Phase 2 baseline and the Step 6.3 report (wer.py, term_recall.py)
    - speed: real-time factor (RTF) = seconds spent transcribing / seconds of
      audio. RTF 0.1 means one hour of lecture takes six minutes
    - size: megabytes on disk (model_source.model_size_mb)
    - drift: WER of this model's transcripts measured AGAINST the reference
      model's transcripts (not against the human captions). It isolates the
      effect of rounding: 0% means int8 wrote exactly what float32 wrote,
      even where both were wrong
    - "WER change vs reference" with a paired bootstrap 95% interval, reusing
      `finetune_eval.paired_bootstrap` (see that module for the statistics)

The first model given is the reference (normally the float32 export), so the
command reads naturally:
    notetaker quantize-eval -m models/...-float32 models/...-float16 models/...-int8

Fairness details:
    - Every model decodes the same clips with the same settings (beam size,
      language, no chunking since clips are under 30 s, no word timestamps,
      which would add a second alignment pass that has nothing to do with
      precision).
    - Each model gets one untimed "warm-up" transcription first. The first
      call into CTranslate2 allocates buffers and picks kernels, and counting
      that one-off cost would penalise whichever model ran first.
    - Models run one after another in this process, and each is released
      before the next loads, so they do not compete for memory.

Two ways to run, mirroring Step 6.3:
    1. Full run: `run_quantize_eval` loads each model, transcribes, saves
       everything (texts and timings) to one JSON file, then scores it.
    2. Re-score only: `evaluate_runs` on that saved JSON. No model, seconds.
       This is what most tests use.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from notetaker.finetune import FinetuneError, load_clips, read_wav
from notetaker.finetune_eval import (
    Bootstrap,
    ClipScore,
    SystemSummary,
    paired_bootstrap,
)
from notetaker.model_source import (
    display_name,
    model_size_mb,
    stored_quantization,
)
from notetaker.normalize import normalize_text
from notetaker.term_recall import Term, score_term_texts
from notetaker.wer import ErrorCounts, score_texts

# Default output names.
RUNS_FILENAME = "quantization_runs.json"
REPORT_JSON = "quantization.json"
REPORT_MD = "quantization.md"

# A loader takes (model spec, device, compute type) and returns an object
# with faster-whisper's `.transcribe()` method. Same shape as benchmark.Loader.
Loader = Callable[[str, str, str], Any]


class QuantizeEvalError(FinetuneError):
    """Bad input for the tradeoff evaluation (subclass so the CLI catches it)."""


# ---------------------------------------------------------------------------
# What one model produced
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipInfo:
    """One evaluation clip: where it is and what was really said."""

    clip: str  # path relative to the split folder, e.g. "lec/lec_00003.wav"
    lecture_id: str
    reference: str
    duration: float  # seconds


@dataclass
class ModelRun:
    """One model's transcripts of every clip, plus how long it took."""

    label: str  # short name shown in the report, e.g. "whisper-small-ct2-int8"
    spec: str  # what was passed to the loader (size name or folder path)
    quantization: str | None  # storage type from export.json, if known
    device: str
    compute_type: str
    size_mb: float | None
    load_seconds: float
    transcribe_seconds: float  # summed over clips, warm-up excluded
    audio_seconds: float  # summed clip durations
    beam_size: int
    texts: list[str] = field(default_factory=list)  # same order as the clips

    @property
    def rtf(self) -> float:
        """Real-time factor: compute seconds per second of audio."""
        if self.audio_seconds <= 0:
            return float("nan")
        return self.transcribe_seconds / self.audio_seconds

    @property
    def speedup(self) -> float:
        """Times faster than real time (1 / RTF)."""
        return 1.0 / self.rtf if self.rtf > 0 else float("inf")


def save_runs(
    clips: Sequence[ClipInfo],
    runs: Sequence[ModelRun],
    path: str | Path,
    *,
    split: str = "test",
) -> Path:
    """Write clips + every model's texts and timings to one JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "split": split,
        "clips": [asdict(c) for c in clips],
        "runs": [asdict(r) for r in runs],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return path


def load_runs(path: str | Path) -> tuple[str, list[ClipInfo], list[ModelRun]]:
    """Read a file from `save_runs`. Returns (split, clips, runs)."""
    path = Path(path)
    if not path.is_file():
        raise QuantizeEvalError(f"No runs file {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        clips = [ClipInfo(**c) for c in data["clips"]]
        runs = [ModelRun(**r) for r in data["runs"]]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise QuantizeEvalError(
            f"{path}: not a quantize-eval runs file ({error})"
        ) from error
    check_runs(clips, runs)
    return str(data.get("split", "test")), clips, runs


def check_runs(clips: Sequence[ClipInfo], runs: Sequence[ModelRun]) -> None:
    """Catch inconsistent input early, with messages that say what is wrong."""
    if not clips:
        raise QuantizeEvalError("No clips to evaluate")
    if not runs:
        raise QuantizeEvalError("No models to compare")
    labels = [r.label for r in runs]
    duplicates = sorted({x for x in labels if labels.count(x) > 1})
    if duplicates:
        raise QuantizeEvalError(
            f"Two models share the label {duplicates[0]!r}; give the export "
            "folders different names so the report can tell them apart"
        )
    for run in runs:
        if len(run.texts) != len(clips):
            raise QuantizeEvalError(
                f"{run.label} has {len(run.texts)} transcripts for {len(clips)} clips"
            )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class ModelScore:
    """Everything the report says about one model."""

    run: ModelRun
    summary: SystemSummary  # pooled WER / CER / term recall vs the captions
    vs_reference: Bootstrap | None  # None for the reference model itself
    drift: float | None  # WER against the reference model's transcripts

    def to_dict(self) -> dict[str, Any]:
        run = asdict(self.run)
        run.pop("texts")  # already in the runs file; keeps the report small
        return {
            **run,
            "rtf": round(self.run.rtf, 6),
            "speedup": round(self.run.speedup, 3),
            **self.summary.to_dict(),
            "wer_change_vs_reference": (
                asdict(self.vs_reference) if self.vs_reference else None
            ),
            "drift": None if self.drift is None else round(self.drift, 6),
        }


def drift_rate(reference_texts: Sequence[str], texts: Sequence[str]) -> float:
    """Pooled WER of `texts` measured against another model's `texts`.

    Clips where the reference model wrote nothing (after normalization) are
    left out, because WER is undefined with an empty reference. If that is
    every clip, the drift is 0 when the other model also wrote nothing, and
    1 (completely different) otherwise.
    """
    total = ErrorCounts()
    for ref, hyp in zip(reference_texts, texts, strict=True):
        if normalize_text(ref).split():
            total = total + score_texts(ref, hyp).words
    if total.reference_length == 0:
        return 0.0 if not any(normalize_text(t).split() for t in texts) else 1.0
    return total.rate


@dataclass
class QuantizeEvalReport:
    """The Step 6.4c tradeoff table plus enough detail to rebuild it."""

    split: str
    reference: str  # label of the reference model
    models: list[ModelScore]
    clips: int
    reference_words: int
    skipped: list[str]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "created_at": self.created_at,
            "reference": self.reference,
            "clips": self.clips,
            "reference_words": self.reference_words,
            "skipped": self.skipped,
            "models": [m.to_dict() for m in self.models],
            "recommended": recommend(self),
        }


def evaluate_runs(
    clips: Sequence[ClipInfo],
    runs: Sequence[ModelRun],
    terms: list[Term],
    *,
    split: str = "test",
    resamples: int = 2000,
    seed: int = 0,
) -> QuantizeEvalReport:
    """Score saved runs and build the tradeoff report (no model needed).

    Reuses the Step 6.3 building blocks: a `ClipScore` holds per-clip counts
    in dicts keyed by system name, and here the "systems" are simply the
    model labels, so `SystemSummary.of(scores, label)` pools them unchanged.
    """
    check_runs(clips, runs)
    labels = [r.label for r in runs]
    scores: list[ClipScore] = []
    skipped: list[str] = []
    kept: list[int] = []  # indices of clips that have a usable reference
    for i, clip in enumerate(clips):
        if not normalize_text(clip.reference).split():
            skipped.append(clip.clip)
            continue
        words, chars, term_counts = {}, {}, {}
        for run in runs:
            pair = score_texts(clip.reference, run.texts[i], name=clip.clip)
            words[run.label] = pair.words
            chars[run.label] = pair.chars
            term_counts[run.label] = score_term_texts(
                clip.reference, run.texts[i], terms, name=clip.clip
            ).counts
        kept.append(i)
        scores.append(
            ClipScore(
                clip=clip.clip,
                lecture_id=clip.lecture_id,
                reference_words=words[labels[0]].reference_length,
                words=words,
                chars=chars,
                terms=term_counts,
            )
        )
    if not scores:
        raise QuantizeEvalError("No clip has a non-empty reference; nothing to score")

    reference = runs[0]
    ref_errors = [s.errors(reference.label) for s in scores]
    ref_words = [s.reference_words for s in scores]
    models = []
    for run in runs:
        is_ref = run is reference
        models.append(
            ModelScore(
                run=run,
                summary=SystemSummary.of(scores, run.label),
                vs_reference=None
                if is_ref
                else paired_bootstrap(
                    ref_errors,
                    [s.errors(run.label) for s in scores],
                    ref_words,
                    resamples=resamples,
                    seed=seed,
                ),
                drift=None
                if is_ref
                else drift_rate(
                    [reference.texts[i] for i in kept], [run.texts[i] for i in kept]
                ),
            )
        )
    return QuantizeEvalReport(
        split=split,
        reference=reference.label,
        models=models,
        clips=len(scores),
        reference_words=sum(ref_words),
        skipped=skipped,
    )


def recommend(report: QuantizeEvalReport) -> str:
    """The fastest model that is not measurably less accurate than the reference.

    "Measurably less accurate" means the whole 95% interval for its WER
    change is above zero. Among the rest we take the lowest RTF, then the
    smallest file. This is a rule of thumb for the write-up, not a law: a
    model that is 0.4 pp worse but 3x faster might still be the right call.
    """

    def ok(m: ModelScore) -> bool:
        return m.vs_reference is None or m.vs_reference.low <= 0

    candidates = [m for m in report.models if ok(m)]
    best = min(
        candidates,
        key=lambda m: (m.run.rtf, m.run.size_mb if m.run.size_mb is not None else 1e18),
    )
    return best.run.label


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _pct(rate: float) -> str:
    return f"{100 * rate:.1f}%"


def _pp(delta: float) -> str:
    return f"{100 * delta:+.1f} pp"


def _mb(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.1f}"


def to_markdown(report: QuantizeEvalReport) -> str:
    """Render the report as GitHub-flavoured markdown."""
    first = report.models[0].run
    lines = [
        "# Precision tradeoff: speed, accuracy and size",
        "",
        f"Split: `{report.split}` ({report.clips} clips, "
        f"{report.reference_words:,} reference words, "
        f"{first.audio_seconds / 60:.1f} min of audio). Device: {first.device}, "
        f"beam size {first.beam_size}. Reference model: `{report.reference}`. "
        f"Generated {report.created_at}.",
        "",
        "| Model | Stored | Compute | Size (MB) | RTF | x real time | WER | CER | "
        "Term recall | WER vs ref (95% CI) | Drift vs ref |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for m in report.models:
        r, s = m.run, m.summary
        change = (
            "reference"
            if m.vs_reference is None
            else f"{_pp(m.vs_reference.delta)} ({_pp(m.vs_reference.low)} to "
            f"{_pp(m.vs_reference.high)})"
        )
        drift = "" if m.drift is None else _pct(m.drift)
        lines.append(
            f"| {r.label} | {r.quantization or 'n/a'} | {r.compute_type} | "
            f"{_mb(r.size_mb)} | {r.rtf:.3f} | {r.speedup:.1f}x | {_pct(s.wer)} | "
            f"{_pct(s.cer)} | {_pct(s.term_recall)} | {change} | {drift} |"
        )
    ref = report.models[0]
    lines += ["", "## Compared with the reference", ""]
    for m in report.models[1:]:
        r = m.run
        faster = ref.run.rtf / r.rtf if r.rtf > 0 else float("inf")
        size = (
            f"{ref.run.size_mb / r.size_mb:.1f}x smaller"
            if r.size_mb and ref.run.size_mb
            else "size unknown"
        )
        lines.append(
            f"- `{r.label}`: {faster:.2f}x the speed, {size}, WER "
            f"{_pp(m.summary.wer - ref.summary.wer)}, term recall "
            f"{_pp(m.summary.term_recall - ref.summary.term_recall)}, and "
            f"{_pct(m.drift or 0.0)} of words differ from `{report.reference}`."
        )
    if len(report.models) == 1:
        lines.append("Only one model was given, so there is nothing to compare.")
    lines += [
        "",
        f"**Pick:** `{recommend(report)}`, the fastest model whose WER is not "
        "measurably worse than the reference (its 95% interval reaches zero or "
        "below).",
        "",
        "## How to read this",
        "",
        "- RTF (real-time factor) is transcription seconds per second of audio, "
        "summed over all clips after one untimed warm-up call. Lower is faster.",
        "- WER, CER and term recall are pooled over clips and measured against "
        "the human captions, as in `reports/baseline.md`.",
        "- WER vs ref is the paired bootstrap interval for (this WER minus the "
        "reference WER). Clips from one lecture are correlated, so the true "
        "interval is somewhat wider.",
        "- Drift vs ref is WER measured against the reference model's own "
        "transcripts. It shows how much rounding changed the output, whether "
        "or not the change happened to help.",
        "- Timings depend on the machine; compare rows of one table, not "
        "numbers across machines.",
    ]
    if report.skipped:
        lines.append(
            f"- {len(report.skipped)} clip(s) skipped because their reference "
            "was empty after normalization."
        )
    return "\n".join(lines) + "\n"


def write_report(report: QuantizeEvalReport, out_dir: str | Path) -> list[Path]:
    """Write `quantization.json` and `quantization.md` into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / REPORT_JSON
    md_path = out_dir / REPORT_MD
    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", "utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return [json_path, md_path]


# ---------------------------------------------------------------------------
# Full run: load each model, time it on every clip
# ---------------------------------------------------------------------------


def _default_loader(spec: str, device: str, compute_type: str) -> Any:
    """Load a faster-whisper model WITHOUT the lru_cache in transcribe.py.

    `load_model` caches the last two models so repeated CLI calls are fast,
    but here we want each model gone from memory before the next one loads.
    `functools.lru_cache` keeps the original function as `__wrapped__`, so
    calling that skips the cache.
    """
    from notetaker.transcribe import load_model

    return load_model.__wrapped__(spec, device=device, compute_type=compute_type)


def run_model(
    spec: str,
    clips: Sequence[ClipInfo],
    audio: Sequence[Any],
    *,
    beam_size: int = 5,
    language: str | None = "en",
    device: str | None = None,
    compute_type: str | None = None,
    warmup: bool = True,
    loader: Loader | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> ModelRun:
    """Load one model, warm it up, and time a transcription of every clip.

    `audio[i]` is clip i as a 16 kHz float32 waveform. Decoding the WAVs is
    done once by the caller, so file reading is not part of anyone's time.
    """
    from notetaker.config import get_settings
    from notetaker.transcribe import resolve_compute_type, resolve_device, transcribe

    settings = get_settings()
    device = resolve_device(device or settings.whisper_device)
    stored = stored_quantization(spec)
    compute_type = resolve_compute_type(
        compute_type or settings.whisper_compute_type, device, stored=stored
    )
    loader = loader or _default_loader

    t0 = clock()
    model = loader(spec, device, compute_type)
    load_seconds = clock() - t0

    def decode(model: Any, waveform: Any) -> str:
        return transcribe(
            waveform,
            model=model,
            language=language,
            beam_size=beam_size,
            word_timestamps=False,
            model_name=spec,
            max_chunk_s=0,  # clips are under 30 s: one window, no chunking
        ).text

    if warmup and len(audio):
        decode(model, audio[0])  # untimed, see the module docstring
    texts: list[str] = []
    elapsed = 0.0
    for waveform in audio:
        t0 = clock()
        texts.append(decode(model, waveform))
        elapsed += clock() - t0
    del model
    gc.collect()  # free the model before the next one loads
    return ModelRun(
        label=display_name(spec),
        spec=spec,
        quantization=stored,
        device=device,
        compute_type=compute_type,
        size_mb=model_size_mb(spec),
        load_seconds=round(load_seconds, 4),
        transcribe_seconds=round(elapsed, 4),
        audio_seconds=round(sum(c.duration for c in clips), 3),
        beam_size=beam_size,
        texts=texts,
    )


def load_eval_clips(
    data_dir: str | Path, split: str = "test", limit: int | None = None
) -> tuple[list[ClipInfo], list[Any]]:
    """Clips of a Step 6.1 dataset split, with their decoded audio."""
    raw = load_clips(data_dir, split)
    if limit is not None:
        raw = raw[:limit]
    if not raw:
        raise QuantizeEvalError(
            f"The {split} split has no clips. Try --split val if your dataset "
            "has no test lectures."
        )
    split_dir = Path(data_dir) / split
    clips = [
        ClipInfo(
            clip=c.path.relative_to(split_dir).as_posix(),
            lecture_id=c.lecture_id,
            reference=c.text,
            duration=c.duration,
        )
        for c in raw
    ]
    return clips, [read_wav(c.path) for c in raw]


def run_quantize_eval(
    models: Sequence[str],
    data_dir: str | Path,
    terms: list[Term],
    *,
    split: str = "test",
    runs_path: str | Path | None = None,
    limit: int | None = None,
    beam_size: int = 5,
    language: str | None = "en",
    device: str | None = None,
    compute_type: str | None = None,
    resamples: int = 2000,
    seed: int = 0,
    loader: Loader | None = None,
    log: Callable[[str], None] = print,
) -> QuantizeEvalReport:
    """Run every model on the split, save the runs, and score them."""
    if not models:
        raise QuantizeEvalError("Give at least one model")
    labels = [display_name(m) for m in models]
    if len(set(labels)) != len(labels):
        raise QuantizeEvalError(
            "Two models would get the same label in the report; give the "
            "export folders different names"
        )
    clips, audio = load_eval_clips(data_dir, split, limit)
    minutes = sum(c.duration for c in clips) / 60
    runs = []
    for number, spec in enumerate(models, start=1):
        log(
            f"[{number}/{len(models)}] {display_name(spec)}: {len(clips)} clips, "
            f"{minutes:.1f} min of audio"
        )
        run = run_model(
            spec,
            clips,
            audio,
            beam_size=beam_size,
            language=language,
            device=device,
            compute_type=compute_type,
            loader=loader,
        )
        log(f"    {run.compute_type} on {run.device}: RTF {run.rtf:.3f}")
        runs.append(run)
    path = save_runs(
        clips, runs, runs_path or Path("reports") / RUNS_FILENAME, split=split
    )
    log(f"Saved transcripts and timings to {path}")
    return evaluate_runs(
        clips, runs, terms, split=split, resamples=resamples, seed=seed
    )
