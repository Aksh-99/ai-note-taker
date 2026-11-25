"""Where a Whisper model comes from: a size name or a local CTranslate2 folder.

Why this exists (Step 6.4b):
    Until now every command took a Whisper *size* ("tiny", "small", ...) and
    faster-whisper downloaded the matching weights from the Hugging Face Hub.
    Step 6.4a produces something new: a folder on disk holding our fine-tuned
    Whisper in CTranslate2 format, stored as int8, float16 or float32. To
    compare those precisions (and to actually use the fine-tuned model) the
    `transcribe` and `benchmark` commands must accept such a folder wherever
    they accepted a size. This module holds the small amount of logic that
    decides which kind of model a string names, checks a folder is usable,
    and reads what the export recorded about itself.

Conveniently, faster-whisper's `WhisperModel(...)` already accepts either a
size name or a folder path as its first argument. So loading needs no change;
what needs care is everything around loading:
    - validation: a typo in a folder path should give a clear error before
      any audio is decoded, and pointing at the LoRA *adapter* folder (an
      easy mistake) should say "run export-ct2 first";
    - labels: reports and transcripts should say "whisper-small-lora-ct2-int8",
      not "/home/me/projects/ai-note-taker/models/whisper-small-lora-ct2-int8";
    - precision: an int8 export should run as int8 and a float32 export as
      float32 when the compute type is left on "auto" (see
      `auto_compute_type`), otherwise a precision comparison would quietly
      compare int8 with int8;
    - size on disk, one of the columns of the speed / accuracy tradeoff.

Name vs folder, when both could apply:
    A size name always wins. If you happen to have a folder called `small`
    in the current directory, `--model small` still means the Hub model;
    write `--model ./small` to use the folder. Anything that is not a known
    size is treated as a folder path.

This module imports nothing heavy (no faster-whisper, no torch) at the top,
so argument parsing stays instant.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from notetaker.config import WhisperModel

# The sizes the project officially supports (same list as the settings).
MODEL_SIZES: tuple[str, ...] = get_args(WhisperModel)

# Files a CTranslate2 Whisper folder must contain for faster-whisper to load
# it. (The converter also writes vocabulary.json, and our exports add
# tokenizer.json, but faster-whisper can work around those two being absent;
# it cannot work without these.)
REQUIRED_FILES = ("model.bin", "config.json")

# Written by `notetaker export-ct2` (Step 6.4a). Duplicated here as a plain
# string, rather than imported from quantize.py, so this module does not pull
# in the fine-tuning code.
EXPORT_SUMMARY = "export.json"

# Files that mark a folder as a LoRA adapter from `notetaker finetune` (PEFT
# writes adapter_config.json). Used only to give a more helpful error.
ADAPTER_MARKERS = ("adapter_config.json", "adapter_model.safetensors")


class ModelSourceError(ValueError):
    """The model string is neither a known size nor a usable local folder."""


@dataclass(frozen=True)
class ModelSource:
    """Everything we know about a model before loading it.

    Attributes:
        spec: exactly what to pass to faster-whisper's `WhisperModel(...)`:
            a size name, or the folder path as a string.
        label: short human-readable name for reports and transcripts.
        local: True for a folder on disk, False for a Hub size name.
        quantization: how the weights are stored ("int8", "float16", ...),
            read from the folder's export.json; None if unknown (always None
            for Hub models, which faster-whisper stores as float16).
        size_mb: total size of the folder in megabytes; None for Hub models
            (see `model_size_mb` for the cached-download case).
    """

    spec: str
    label: str
    local: bool
    quantization: str | None = None
    size_mb: float | None = None


def dir_size_mb(path: str | Path) -> float:
    """Total size of every file under `path`, in megabytes (1 MB = 2**20).

    `rglob("*")` walks the folder recursively. `stat()` follows symbolic
    links, which matters for the Hugging Face cache: it stores each file once
    as a "blob" and links to it, and we want the size of the real file.
    """
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total / 2**20


def read_export_summary(folder: str | Path) -> dict | None:
    """The folder's export.json as a dict, or None if missing or unreadable."""
    path = Path(folder) / EXPORT_SUMMARY
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # OSError: no such file. ValueError: not valid JSON. Either way the
        # folder still loads fine, we just know less about it.
        return None
    return data if isinstance(data, dict) else None


def is_size_name(spec: str) -> bool:
    return spec in MODEL_SIZES


def stored_quantization(spec: str) -> str | None:
    """Storage type recorded by export-ct2, or None. Never raises.

    This is the lenient lookup used at load time: a size name, a folder made
    by some other tool, or a damaged export.json all simply give None.
    """
    if is_size_name(spec) or not Path(spec).is_dir():
        return None
    summary = read_export_summary(spec)
    if summary is None:
        return None
    value = summary.get("quantization")
    return value if isinstance(value, str) else None


def display_name(spec: str) -> str:
    """Short label: the folder's own name for a local folder, else `spec`.

    `Path.resolve()` turns "." or "../models/x/" into an absolute path first,
    so the label is the real folder name rather than "." or "".
    """
    if is_size_name(spec) or not Path(spec).is_dir():
        return spec
    return Path(spec).resolve().name


def resolve_model_source(spec: str) -> ModelSource:
    """Decide what `spec` names and check it is usable. Strict: raises.

    Raises:
        ModelSourceError with a message that says how to fix the problem.
    """
    spec = str(spec).strip()
    if not spec:
        raise ModelSourceError("model must not be empty")
    if is_size_name(spec):
        return ModelSource(spec=spec, label=spec, local=False)

    folder = Path(spec).expanduser()
    if not folder.exists():
        raise ModelSourceError(
            f"'{spec}' is not a Whisper size ({', '.join(MODEL_SIZES)}) "
            "and no folder with that path exists"
        )
    if not folder.is_dir():
        raise ModelSourceError(
            f"'{spec}' is a file; expected a Whisper size or a CTranslate2 "
            "model folder (the output of `notetaker export-ct2`)"
        )
    missing = [name for name in REQUIRED_FILES if not (folder / name).is_file()]
    if missing:
        if any((folder / marker).is_file() for marker in ADAPTER_MARKERS):
            # The most likely mistake: passing the training output.
            raise ModelSourceError(
                f"'{spec}' is a LoRA adapter folder, which faster-whisper cannot "
                f"load directly. Convert it first:  notetaker export-ct2 "
                f"--model-dir {spec}"
            )
        raise ModelSourceError(
            f"'{spec}' is not a CTranslate2 model folder (missing "
            f"{', '.join(missing)}). Create one with `notetaker export-ct2`."
        )
    return ModelSource(
        spec=str(folder),
        label=display_name(str(folder)),
        local=True,
        quantization=stored_quantization(str(folder)),
        size_mb=round(dir_size_mb(folder), 2),
    )


def model_size_mb(spec: str) -> float | None:
    """Size on disk of a model, if it can be found without downloading.

    For a folder this is just its size. For a size name we ask faster-whisper
    where the Hub download lives with `local_files_only=True`, which only
    looks in the local cache and raises instead of downloading. So this is
    safe to call anywhere (including CI): worst case it returns None.
    """
    if not is_size_name(spec):
        return round(dir_size_mb(spec), 2) if Path(spec).is_dir() else None
    try:
        from faster_whisper.utils import download_model

        return round(dir_size_mb(download_model(spec, local_files_only=True)), 2)
    except Exception:  # noqa: BLE001
        # Not cached, faster-whisper not importable, no network needed but
        # some Hub-library error anyway: the size is simply unknown.
        return None


# How each storage type should be *computed* when the user leaves the compute
# type on "auto". Keyed by (storage type, device).
#   - On a CPU there is no fast float16 math, so float16 weights run as
#     float32, and int8 weights run as int8 (fast integer instructions).
#   - On a GPU, int8 weights run as int8_float16: int8 matrix multiplies with
#     float16 for everything else, CTranslate2's usual GPU int8 mode.
# The point: "auto" runs a model at the precision it was exported in, as far
# as the device allows, so comparing exports really compares precisions.
_AUTO_COMPUTE = {
    ("int8", "cpu"): "int8",
    ("int8_float16", "cpu"): "int8",
    ("float16", "cpu"): "float32",
    ("float32", "cpu"): "float32",
    ("int8", "cuda"): "int8_float16",
    ("int8_float16", "cuda"): "int8_float16",
    ("float16", "cuda"): "float16",
    ("float32", "cuda"): "float32",
}


def auto_compute_type(stored: str | None, device: str) -> str:
    """Compute type for "auto", given how weights are stored (if known).

    Without a known storage type we fall back to the project's original rule:
    float16 on a GPU, int8 on a CPU.
    """
    if stored is not None and (stored, device) in _AUTO_COMPUTE:
        return _AUTO_COMPUTE[(stored, device)]
    return "float16" if device == "cuda" else "int8"


def parse_model_arg(value: str) -> str:
    """argparse `type=` function for --model: a size name or a model folder.

    argparse calls this on the raw string. Raising ArgumentTypeError makes
    argparse print a clean "argument -m/--model: ..." usage error (exit code
    2), exactly like an invalid `choices` value used to.
    """
    try:
        return resolve_model_source(value).spec
    except ModelSourceError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
