"""Tests for the notebook helpers in notetaker.runtime (Step 6.2b)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from notetaker.finetune import FinetuneConfig, FinetuneError
from notetaker.runtime import (
    DEFAULT_EFFECTIVE_BATCH,
    default_paths,
    detect_platform,
    notebook_config,
    suggest_batch,
    write_config,
)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"COLAB_RELEASE_TAG": "release-colab_2026"}, "colab"),
        ({"COLAB_GPU": "1"}, "colab"),
        ({"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, "kaggle"),
        ({"HOME": "/home/me"}, "local"),
        ({}, "local"),
    ],
)
def test_detect_platform(environ: dict[str, str], expected: str) -> None:
    assert detect_platform(environ) == expected


def test_detect_platform_reads_the_real_environment(monkeypatch) -> None:
    for name in ("COLAB_RELEASE_TAG", "COLAB_GPU", "KAGGLE_KERNEL_RUN_TYPE"):
        monkeypatch.delenv(name, raising=False)
    assert detect_platform() == "local"
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    assert detect_platform() == "kaggle"


def test_default_paths_colab_with_and_without_drive() -> None:
    plain = default_paths("colab")
    assert plain.data == Path("/content/data")
    assert plain.output == Path("/content/models/whisper-small-lora")
    assert plain.persist == Path("/content/results")
    drive = default_paths("colab", drive_mounted=True)
    assert drive.persist == Path("/content/drive/MyDrive/notetaker")


def test_default_paths_kaggle_keeps_only_small_results() -> None:
    paths = default_paths("kaggle")
    # Big audio outside the saved folder; adapter and zip inside it.
    assert not paths.data.is_relative_to("/kaggle/working")
    assert paths.output.is_relative_to("/kaggle/working")
    assert paths.persist == Path("/kaggle/working")


def test_default_paths_local_and_unknown() -> None:
    assert default_paths("local").data == Path("data")
    with pytest.raises(ValueError, match="Unknown platform"):
        default_paths("laptop")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("memory", "expected"),
    [(15.8, (8, 2)), (40.0, (8, 2)), (12.0, (4, 4)), (8.0, (2, 8)), (4.0, (1, 16))],
)
def test_suggest_batch_keeps_the_effective_batch(
    memory: float, expected: tuple[int, int]
) -> None:
    batch, accum = suggest_batch(memory)
    assert (batch, accum) == expected
    assert batch * accum == DEFAULT_EFFECTIVE_BATCH


def test_suggest_batch_edge_cases() -> None:
    # Never a smaller effective batch than asked (ceiling division).
    assert suggest_batch(8.0, effective_batch=5) == (2, 3)
    # The batch never exceeds the effective batch.
    assert suggest_batch(16.0, effective_batch=4) == (4, 1)
    with pytest.raises(ValueError):
        suggest_batch(0)
    with pytest.raises(ValueError):
        suggest_batch(16.0, effective_batch=0)


def test_notebook_config_full_run_on_a_t4() -> None:
    config = notebook_config(15.8, max_steps=1000)
    assert (config.batch_size, config.grad_accum_steps) == (8, 2)
    assert config.fp16
    assert (config.max_steps, config.warmup_steps, config.eval_every) == (1000, 50, 100)
    # Everything else keeps the FinetuneConfig defaults.
    assert config.lora_r == FinetuneConfig().lora_r
    assert config.base_model == "openai/whisper-small"


def test_notebook_config_smoke_test_and_no_gpu() -> None:
    smoke = notebook_config(None, smoke_test=True)
    assert (smoke.max_steps, smoke.warmup_steps, smoke.eval_every) == (20, 2, 10)
    assert not smoke.fp16
    assert (smoke.batch_size, smoke.grad_accum_steps) == (1, 16)


def test_notebook_config_short_run_and_overrides() -> None:
    config = notebook_config(15.8, max_steps=30, lora_r=8)
    assert (config.warmup_steps, config.eval_every, config.lora_r) == (3, 30, 8)
    with pytest.raises(FinetuneError, match="Unknown config keys"):
        notebook_config(15.8, lora_rank=8)
    with pytest.raises(FinetuneError, match="learning_rate"):
        notebook_config(15.8, learning_rate=-1.0)


def test_write_config_round_trips_through_the_cli_loader(tmp_path: Path) -> None:
    config = notebook_config(12.0, max_steps=200)
    path = write_config(config, tmp_path / "nested" / "cfg.json")
    assert json.loads(path.read_text())["batch_size"] == 4
    assert FinetuneConfig.from_file(path) == config
