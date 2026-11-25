"""Export the fine-tuned Whisper to CTranslate2, optionally as int8 (Step 6.4a).

Why this exists:
    Step 6.2 trains a LoRA adapter on top of Hugging Face Whisper, which runs
    on PyTorch. The rest of the project (`notetaker transcribe`, the agent,
    the benchmark) runs Whisper through faster-whisper, which uses the
    CTranslate2 engine instead: about 4x faster and able to use 8-bit
    integer weights on a plain CPU. This module turns "base Whisper + LoRA
    adapter" into a folder faster-whisper can load directly.

The three steps:
    1. Merge. LoRA computes W x + (alpha / r) B A x. Because both terms are
       linear in x, we can fold the correction into the weight once:
           W_merged = W + (alpha / r) B A
       After that the model is an ordinary Whisper with the same shapes as
       the original, so any tool that understands Whisper can load it, and
       inference pays no extra cost for the adapter. peft's
       `merge_and_unload()` does exactly this for every adapted layer.
    2. Save as a normal Hugging Face model folder (config, weights,
       tokenizer, feature-extractor settings).
    3. Convert with CTranslate2's converter, choosing how weights are stored.

What int8 quantization is (the math, briefly):
    float32 stores each weight in 4 bytes. int8 stores it in 1 byte: for each
    row w of a weight matrix, CTranslate2 picks a scale
        s = max(|w_i|) / 127
    and stores q_i = round(w_i / s), an integer in [-127, 127]. At run time
    w_i is approximated by s * q_i. The rounding error per weight is at most
    s / 2, which is tiny compared to the weights themselves, so accuracy
    usually drops very little while the file shrinks about 4x and CPUs can
    use their fast integer instructions. Step 6.4b measures the actual
    speed / accuracy tradeoff on the lecture test clips.

Storage type vs compute type:
    `quantization` here decides how weights are *stored on disk*. When
    faster-whisper loads the folder, its `compute_type` decides how they are
    *used*, and CTranslate2 converts on load if needed. So one int8 export
    can run as int8 on a CPU and as int8_float16 on a GPU. Exporting int8 is
    the most flexible default: smallest file, runs everywhere.

Design notes:
    - torch, transformers and peft are only needed here (the [finetune]
      extra) and are imported lazily, so the rest of the package does not
      need them. ctranslate2 is always installed, as a dependency of
      faster-whisper, but is also imported lazily because it loads native
      libraries.
    - `export_ct2` takes the same `load_model` hook as `run_finetune`
      (dependency injection), so tests use a tiny random Whisper with a tiny
      real tokenizer and never download anything.
    - After converting, `verify_ct2` loads the result with faster-whisper and
      transcribes one second of silence. That catches a missing tokenizer or
      config file at export time instead of on the next real run.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from notetaker.finetune import (
    SUMMARY_FILENAME,
    WHISPER_SAMPLE_RATE,
    FinetuneConfig,
    FinetuneError,
    _import,
    load_finetuned,
)

# dir_size_mb moved to model_source.py in Step 6.4b (benchmark needs it
# without the fine-tuning code); imported here so existing callers still work.
from notetaker.model_source import dir_size_mb

# Weight storage types we offer. CTranslate2 knows a few more (int16,
# bfloat16, ...), but these four cover the useful cases:
#   int8          smallest (about 1/4 of float32), fast on CPU, fine on GPU
#   int8_float16  int8 weights, float16 for everything else; GPU only
#   float16       half size, the usual GPU choice, slow on most CPUs
#   float32       full precision, the reference for accuracy comparisons
QUANTIZATIONS = ("int8", "int8_float16", "float16", "float32")

# Files faster-whisper reads next to the converted weights. The converter
# copies them over from the merged Hugging Face folder.
#   tokenizer.json            turns token ids back into text
#   preprocessor_config.json  how audio becomes a log-mel spectrogram
#                             (80 mel bins for Whisper small, 128 for large-v3)
COPY_FILES = ("tokenizer.json", "preprocessor_config.json")

# What the export writes next to the CTranslate2 files, so a folder records
# where it came from (which adapter, which base model, which precision).
EXPORT_SUMMARY = "export.json"

# Files CTranslate2's converter writes. Used to check an output folder looks
# like a complete export and to refuse to delete folders that are not ours.
CT2_FILES = ("model.bin", "config.json", "vocabulary.json")


# ---------------------------------------------------------------------------
# Step 1 + 2: merge the adapter into the base weights and save
# ---------------------------------------------------------------------------


def save_merged(
    model_dir: str | Path,
    out_dir: str | Path,
    *,
    load_model: Callable[[FinetuneConfig], tuple[Any, Any, Any]] | None = None,
) -> FinetuneConfig:
    """Fold the LoRA adapter into Whisper and save a plain Hugging Face folder.

    Loads `<model_dir>/finetune.json` + `adapter/` (via `load_finetuned`),
    merges, and writes config, weights, tokenizer and feature-extractor
    settings to `out_dir`. Runs on the CPU: merging is a handful of matrix
    products, and keeping it off the GPU avoids memory trouble on Colab.
    Returns the training config (it names the base model).
    """
    model, feature_extractor, tokenizer, config = load_finetuned(
        model_dir, device="cpu", load_model=load_model
    )
    # merge_and_unload: W <- W + (alpha / r) B A in every adapted layer, then
    # drop the peft wrapper so what remains is a normal Whisper model.
    merged = model.merge_and_unload()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    feature_extractor.save_pretrained(str(out))
    _ensure_fast_tokenizer_file(out)
    return config


def _ensure_fast_tokenizer_file(folder: Path) -> None:
    """Make sure `tokenizer.json` exists; faster-whisper needs it.

    A "fast" tokenizer (Rust `tokenizers` library) saves everything in one
    tokenizer.json. An older "slow" Python tokenizer saves vocab.json +
    merges.txt instead. In that case we load the folder as a fast tokenizer,
    which converts the files, and save again.
    """
    if (folder / "tokenizer.json").is_file():
        return
    transformers = _import("transformers")
    fast = transformers.WhisperTokenizerFast.from_pretrained(str(folder))
    fast.save_pretrained(str(folder))
    if not (folder / "tokenizer.json").is_file():  # pragma: no cover - defensive
        raise FinetuneError(f"Could not write tokenizer.json into {folder}")


# ---------------------------------------------------------------------------
# Step 3: convert to CTranslate2
# ---------------------------------------------------------------------------


def convert_to_ct2(
    hf_dir: str | Path, out_dir: str | Path, quantization: str = "int8"
) -> Path:
    """Convert a Hugging Face Whisper folder to CTranslate2 format.

    Writes model.bin (the weights in the chosen storage type), config.json,
    vocabulary.json, plus copies of the files in COPY_FILES. `out_dir` must
    not exist yet or must be empty (`export_ct2` handles replacing).
    """
    if quantization not in QUANTIZATIONS:
        raise FinetuneError(
            f"Unknown quantization {quantization!r}; choose from {QUANTIZATIONS}"
        )
    hf_dir = Path(hf_dir)
    missing = [name for name in COPY_FILES if not (hf_dir / name).is_file()]
    if missing:
        raise FinetuneError(f"{hf_dir} is missing {', '.join(missing)}")
    from ctranslate2.converters import TransformersConverter

    converter = TransformersConverter(str(hf_dir), copy_files=list(COPY_FILES))
    # force=True lets the converter write into an existing empty folder
    # (tempfile.mkdtemp and pytest's tmp_path create folders up front).
    converter.convert(str(out_dir), quantization=quantization, force=True)
    return Path(out_dir)


def verify_ct2(ct2_dir: str | Path, compute_type: str = "int8") -> int:
    """Load an export with faster-whisper on the CPU and decode 1 s of silence.

    This is a smoke test, not an accuracy check: it proves every file the
    runtime needs is present and the weights load. Returns the number of
    segments decoded (usually 0 for silence; the number itself is
    unimportant). `compute_type` must be CPU-friendly (int8 or float32).
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(str(ct2_dir), device="cpu", compute_type=compute_type)
    silence = np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32)
    # transcribe() returns a lazy generator; list() makes it actually run.
    segments, _info = model.transcribe(silence, language="en", beam_size=1)
    return len(list(segments))


# ---------------------------------------------------------------------------
# The whole export
# ---------------------------------------------------------------------------


@dataclass
class ExportResult:
    """What `export_ct2` produced, also saved as `<out_dir>/export.json`."""

    out_dir: str
    source_model_dir: str
    base_model: str
    quantization: str
    merged_mb: float  # float32 Hugging Face weights after merging
    ct2_mb: float  # the CTranslate2 folder
    seconds: float
    verified: bool

    @property
    def compression(self) -> float:
        """How many times smaller the export is than the merged float32 model."""
        return self.merged_mb / self.ct2_mb if self.ct2_mb > 0 else float("nan")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["compression"] = round(self.compression, 2)
        return data


def _prepare_out_dir(out_dir: Path, force: bool) -> None:
    """Refuse to clobber an existing folder unless asked; never delete a folder
    that does not look like one of our exports (a typo in --out should not be
    able to wipe, say, the adapter folder)."""
    if not out_dir.exists():
        return
    if out_dir.is_dir() and not any(out_dir.iterdir()):
        return  # empty folder: fine to use
    if not force:
        raise FinetuneError(f"{out_dir} already exists; pass force=True to replace it")
    if not (out_dir / EXPORT_SUMMARY).is_file():
        raise FinetuneError(
            f"{out_dir} is not a previous export (no {EXPORT_SUMMARY}); "
            "refusing to delete it. Choose another --out folder."
        )
    shutil.rmtree(out_dir)


def export_ct2(
    model_dir: str | Path,
    out_dir: str | Path,
    *,
    quantization: str = "int8",
    force: bool = False,
    verify: bool = True,
    keep_merged: str | Path | None = None,
    load_model: Callable[[FinetuneConfig], tuple[Any, Any, Any]] | None = None,
    log: Callable[[str], None] = print,
) -> ExportResult:
    """Merge the adapter in `model_dir`, convert to CTranslate2, and check it.

    Output folder (`out_dir`), loadable with
    `faster_whisper.WhisperModel(out_dir)`:
        model.bin, config.json, vocabulary.json   CTranslate2 model
        tokenizer.json, preprocessor_config.json  copied for faster-whisper
        export.json                               where it came from + sizes

    The merged Hugging Face model is an intermediate step written to a
    temporary folder and deleted afterwards, unless `keep_merged` names a
    folder to keep it in (useful for Step 6.4b, which compares precisions,
    so the merge does not have to be repeated).
    """
    if quantization not in QUANTIZATIONS:
        raise FinetuneError(
            f"Unknown quantization {quantization!r}; choose from {QUANTIZATIONS}"
        )
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    if not (model_dir / SUMMARY_FILENAME).is_file():
        raise FinetuneError(
            f"{model_dir} has no {SUMMARY_FILENAME}; run `notetaker finetune` first"
        )
    if out_dir.resolve() == model_dir.resolve():
        raise FinetuneError("--out must be a different folder from --model-dir")
    _prepare_out_dir(out_dir, force)
    start = time.perf_counter()

    # `with tempfile.TemporaryDirectory()` creates a folder and deletes it,
    # with everything inside, when the block ends, even if an error occurs.
    with tempfile.TemporaryDirectory(prefix="notetaker-merged-") as tmp:
        merged_dir = Path(keep_merged) if keep_merged else Path(tmp)
        log(f"Merging the LoRA adapter from {model_dir} ...")
        config = save_merged(model_dir, merged_dir, load_model=load_model)
        merged_mb = dir_size_mb(merged_dir)
        log(f"Converting to CTranslate2 ({quantization}) ...")
        convert_to_ct2(merged_dir, out_dir, quantization)

    verified = False
    if verify:
        # float16-family exports are meant for GPUs; on a CPU we still load
        # them, but as int8 (CTranslate2 converts on load), to check files.
        log("Checking the export loads in faster-whisper ...")
        verify_ct2(out_dir, compute_type="int8")
        verified = True

    result = ExportResult(
        out_dir=str(out_dir),
        source_model_dir=str(model_dir),
        base_model=config.base_model,
        quantization=quantization,
        merged_mb=round(merged_mb, 2),
        ct2_mb=round(dir_size_mb(out_dir), 2),
        seconds=round(time.perf_counter() - start, 2),
        verified=verified,
    )
    (out_dir / EXPORT_SUMMARY).write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    log(
        f"Exported {result.ct2_mb:.1f} MB ({quantization}) to {out_dir}; "
        f"merged float32 model was {result.merged_mb:.1f} MB "
        f"({result.compression:.1f}x smaller)"
    )
    return result
