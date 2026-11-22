"""Where am I running, and what fits on this GPU? (Step 6.2b)

Small helpers for the free-GPU fine-tuning notebook
(`notebooks/finetune_whisper_small.ipynb`). They live in the package rather
than in notebook cells so they can be unit tested in CI: notebook code is
never run by the test suite, package code is.

Nothing here imports torch. The notebook asks torch for the GPU name and
memory and passes the plain numbers in, which keeps these functions pure
(same input, same output) and trivial to test.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from notetaker.finetune import FinetuneConfig

Platform = Literal["colab", "kaggle", "local"]

# Effective batch (batch_size * grad_accum_steps) of the FinetuneConfig
# defaults. Changing the per-step batch to fit memory should not change the
# optimisation itself, so we keep this product fixed.
DEFAULT_EFFECTIVE_BATCH = 16


def detect_platform(environ: Mapping[str, str] | None = None) -> Platform:
    """Guess the notebook host from environment variables it always sets.

    Colab sets COLAB_RELEASE_TAG (and older images COLAB_GPU); Kaggle sets
    KAGGLE_KERNEL_RUN_TYPE. Passing `environ` explicitly is for tests; by
    default the real process environment is read.
    """
    env = os.environ if environ is None else environ
    if "COLAB_RELEASE_TAG" in env or "COLAB_GPU" in env:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in env:
        return "kaggle"
    return "local"


@dataclass(frozen=True)
class RunPaths:
    """Folders the notebook uses. `persist` is what survives the session."""

    work: Path  # repo checkout and downloaded audio (lost when the VM resets)
    data: Path  # NOTETAKER_DATA_DIR: raw lectures + fine-tuning clips
    output: Path  # adapter + finetune.json (small, worth keeping)
    persist: Path  # where the zipped result is copied at the end


def default_paths(platform: Platform, *, drive_mounted: bool = False) -> RunPaths:
    """Sensible folders for each host.

    - Colab: /content is the scratch disk. If Google Drive is mounted the
      result is also copied to MyDrive, which survives a disconnect.
    - Kaggle: only /kaggle/working is saved as the notebook's "Output"
      (20 GB limit), so the small results go there and the big audio goes
      to /kaggle/temp, which is not saved.
    - Local: everything under the current folder (the repo's gitignored
      data/ and models/ folders).
    """
    if platform == "colab":
        work = Path("/content")
        persist = (
            Path("/content/drive/MyDrive/notetaker")
            if drive_mounted
            else Path("/content/results")
        )
        return RunPaths(
            work=work,
            data=work / "data",
            output=work / "models" / "whisper-small-lora",
            persist=persist,
        )
    if platform == "kaggle":
        keep = Path("/kaggle/working")
        return RunPaths(
            work=Path("/kaggle/temp"),
            data=Path("/kaggle/temp/data"),
            output=keep / "whisper-small-lora",
            persist=keep,
        )
    if platform == "local":
        here = Path(".")
        return RunPaths(
            work=here,
            data=here / "data",
            output=here / "models" / "whisper-small-lora",
            persist=here / "models",
        )
    raise ValueError(f"Unknown platform {platform!r}")


def suggest_batch(
    gpu_memory_gb: float, effective_batch: int = DEFAULT_EFFECTIVE_BATCH
) -> tuple[int, int]:
    """(batch_size, grad_accum_steps) for Whisper small + LoRA in fp16.

    Rough memory rule from the PEFT Whisper example: batch 8 fits a 16 GB T4
    (free Colab and Kaggle report about 15 GB usable). Smaller cards halve
    the batch and double the accumulation, so batch_size * grad_accum_steps
    stays `effective_batch` and the gradient of each optimizer step is the
    same average over the same number of clips.
    """
    if gpu_memory_gb <= 0:
        raise ValueError("gpu_memory_gb must be positive")
    if effective_batch < 1:
        raise ValueError("effective_batch must be at least 1")
    if gpu_memory_gb >= 14:
        batch = 8
    elif gpu_memory_gb >= 10:
        batch = 4
    elif gpu_memory_gb >= 6:
        batch = 2
    else:
        batch = 1
    batch = min(batch, effective_batch)
    # Ceiling division, so the effective batch is never smaller than asked.
    accum = -(-effective_batch // batch)
    return batch, accum


def notebook_config(
    gpu_memory_gb: float | None,
    *,
    max_steps: int = 1000,
    smoke_test: bool = False,
    **overrides: Any,
) -> FinetuneConfig:
    """The FinetuneConfig the notebook trains with.

    - `gpu_memory_gb=None` means no GPU: fp16 off, batch 1 (only sensible
      for a smoke test).
    - `smoke_test=True` is a 20-step run with an eval every 10 steps, to
      check the whole pipeline works (about a minute on a T4) before
      committing to the full run.
    The result is validated, so a bad override fails here, not an hour in.
    """
    if gpu_memory_gb is None:
        batch, accum = 1, DEFAULT_EFFECTIVE_BATCH
        fp16 = False
    else:
        batch, accum = suggest_batch(gpu_memory_gb)
        fp16 = True
    steps = 20 if smoke_test else max_steps
    settings: dict[str, Any] = {
        "batch_size": batch,
        "grad_accum_steps": accum,
        "fp16": fp16,
        "max_steps": steps,
        "warmup_steps": min(50, steps // 10),
        "eval_every": 10 if smoke_test else min(100, steps),
    }
    settings.update(overrides)
    config = FinetuneConfig.from_json(settings)  # validates
    return config


def write_config(config: FinetuneConfig, path: str | Path) -> Path:
    """Save a config as JSON for `notetaker finetune --config <path>`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_json(), indent=2) + "\n", encoding="utf-8")
    return path
