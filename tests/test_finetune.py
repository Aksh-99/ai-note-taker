"""Tests for the LoRA fine-tuning module that do NOT need PyTorch (Step 6.2a).

Settings, the learning-rate schedule, reading the Step 6.1 dataset, the
training plan, the dataset class (with fake feature extractor and tokenizer)
and `notetaker finetune --dry-run`. The training itself is tested in
test_finetune_training.py, which is skipped when torch is not installed.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from notetaker.audio import save_wav
from notetaker.cli import main
from notetaker.finetune import (
    WHISPER_MAX_LABEL_TOKENS,
    ClipDataset,
    FinetuneConfig,
    FinetuneError,
    load_clips,
    lr_multiplier,
    plan_markdown,
    plan_training,
    read_wav,
)

SR = 16_000


def make_split(
    root: Path,
    split: str,
    clips: list[tuple[str, float, str]],
) -> Path:
    """Write `<root>/<split>/metadata.jsonl` + WAVs like finetune-data does.

    `clips` holds (lecture id, seconds, text) per clip.
    """
    split_dir = root / split
    with_counts: dict[str, int] = {}
    rows = []
    for lecture, seconds, text in clips:
        number = with_counts.get(lecture, 0)
        with_counts[lecture] = number + 1
        name = f"{lecture}/{lecture}_{number:05d}.wav"
        t = np.arange(int(seconds * SR)) / SR
        save_wav(0.3 * np.sin(2 * np.pi * 220 * t), split_dir / name, SR)
        rows.append(
            {
                "file_name": name,
                "text": text,
                "lecture_id": lecture,
                "start": 0.0,
                "end": seconds,
                "duration": seconds,
            }
        )
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return split_dir


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    root = tmp_path / "finetune"
    make_split(
        root,
        "train",
        [
            ("lec-a", 1.0, "the eigenvalue"),
            ("lec-a", 2.0, "of a matrix"),
            ("lec-b", 1.5, "gradient descent"),
            ("lec-b", 0.5, "step size"),
        ],
    )
    make_split(root, "val", [("lec-c", 1.0, "Bellman equation")])
    return root


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_default_config_is_valid_and_effective_batch() -> None:
    config = FinetuneConfig()
    config.validate()
    assert config.effective_batch == 16
    assert config.max_label_tokens == WHISPER_MAX_LABEL_TOKENS


def test_config_round_trips_through_json(tmp_path: Path) -> None:
    config = FinetuneConfig(lora_r=8, target_modules=("q_proj", "k_proj"))
    path = tmp_path / "c.json"
    path.write_text(json.dumps(config.to_json()), encoding="utf-8")
    assert FinetuneConfig.from_file(path) == config


@pytest.mark.parametrize(
    "changes",
    [
        {"lora_r": 0},
        {"batch_size": 0},
        {"lora_dropout": 1.0},
        {"learning_rate": 0.0},
        {"warmup_steps": 1000},  # must be below max_steps
        {"max_label_tokens": 449},
        {"max_grad_norm": 0.0},
        {"target_modules": ()},
    ],
)
def test_bad_config_values_are_rejected(changes: dict) -> None:
    with pytest.raises(FinetuneError):
        FinetuneConfig(**changes).validate()


def test_unknown_config_key_is_an_error() -> None:
    with pytest.raises(FinetuneError, match="lora_rank"):
        FinetuneConfig.from_json({"lora_rank": 8})


def test_unreadable_config_file(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(FinetuneError, match="JSON object"):
        FinetuneConfig.from_file(path)
    with pytest.raises(FinetuneError, match="Cannot read"):
        FinetuneConfig.from_file(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------


def test_lr_schedule_warms_up_then_decays_linearly() -> None:
    values = [lr_multiplier(s, warmup_steps=4, total_steps=12) for s in range(13)]
    assert values[:5] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert values[8] == pytest.approx(0.5)
    assert values[12] == 0.0
    # Never negative, even past the end.
    assert lr_multiplier(20, 4, 12) == 0.0


def test_lr_schedule_without_warmup_starts_at_full_rate() -> None:
    assert lr_multiplier(0, 0, 10) == 1.0
    assert lr_multiplier(5, 0, 10) == 0.5


# ---------------------------------------------------------------------------
# Reading the dataset
# ---------------------------------------------------------------------------


def test_load_clips_reads_metadata(dataset_dir: Path) -> None:
    clips = load_clips(dataset_dir, "train")
    assert [c.text for c in clips] == [
        "the eigenvalue",
        "of a matrix",
        "gradient descent",
        "step size",
    ]
    assert clips[0].path.is_file()
    assert {c.lecture_id for c in clips} == {"lec-a", "lec-b"}


def test_load_clips_errors(dataset_dir: Path) -> None:
    with pytest.raises(FinetuneError, match="finetune-data"):
        load_clips(dataset_dir, "test")  # never built
    with pytest.raises(FinetuneError, match="Unknown split"):
        load_clips(dataset_dir, "dev")
    (dataset_dir / "val" / "lec-c" / "lec-c_00000.wav").unlink()
    with pytest.raises(FinetuneError, match="Missing audio"):
        load_clips(dataset_dir, "val")


def test_load_clips_rejects_bad_rows_and_long_clips(tmp_path: Path) -> None:
    split = make_split(tmp_path, "train", [("lec", 1.0, "hi")])
    meta = split / "metadata.jsonl"
    row = json.loads(meta.read_text())
    meta.write_text(json.dumps({**row, "duration": 31.0}) + "\n")
    with pytest.raises(FinetuneError, match="longer than"):
        load_clips(tmp_path, "train")
    del row["text"]
    meta.write_text(json.dumps(row) + "\n")
    with pytest.raises(FinetuneError, match="line 1"):
        load_clips(tmp_path, "train")


def test_read_wav_round_trips_save_wav(tmp_path: Path) -> None:
    audio = np.linspace(-0.5, 0.5, SR, dtype=np.float32)
    save_wav(audio, tmp_path / "a.wav", SR)
    back = read_wav(tmp_path / "a.wav")
    assert back.dtype == np.float32
    np.testing.assert_allclose(back, audio, atol=1e-4)


def test_read_wav_rejects_other_formats(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SR)
        wav_file.writeframes(b"\x00" * 400)
    with pytest.raises(FinetuneError, match="mono"):
        read_wav(path)
    save_wav(np.zeros(100), tmp_path / "8k.wav", 8000)
    with pytest.raises(FinetuneError, match="8000"):
        read_wav(tmp_path / "8k.wav")


# ---------------------------------------------------------------------------
# Training plan
# ---------------------------------------------------------------------------


def test_plan_counts_clips_hours_and_epochs(dataset_dir: Path) -> None:
    config = FinetuneConfig(
        batch_size=2, grad_accum_steps=1, max_steps=6, warmup_steps=0
    )
    plan = plan_training(config, dataset_dir)
    assert plan.splits["train"].clips == 4
    assert plan.splits["train"].lectures == 2
    assert plan.splits["train"].hours == pytest.approx(5.0 / 3600, abs=1e-3)
    assert plan.steps_per_epoch == 2
    assert plan.epochs == 3.0
    text = plan_markdown(plan, config)
    assert "| train | 4 |" in text and "3.0 epochs" in text
    assert "overfit" not in text


def test_plan_partial_batch_counts_as_a_step_and_warns_on_many_epochs(
    dataset_dir: Path,
) -> None:
    config = FinetuneConfig(batch_size=3, grad_accum_steps=1)  # 4 clips / 3
    plan = plan_training(config, dataset_dir)
    assert plan.steps_per_epoch == 2
    assert "overfit" in plan_markdown(plan, config)  # 1000 steps = 500 epochs


def test_plan_needs_a_val_split(tmp_path: Path) -> None:
    make_split(tmp_path, "train", [("lec", 1.0, "hi")])
    with pytest.raises(FinetuneError, match="val"):
        plan_training(FinetuneConfig(), tmp_path)


# ---------------------------------------------------------------------------
# Dataset class with fakes
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """One id per character, wrapped in start (1) and end (2) tokens."""

    def __call__(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[1] + [3 + ord(c) % 90 for c in text] + [2])


class FakeFeatures:
    """Records what it was given and returns a fixed-shape 'spectrogram'."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, audio, sampling_rate, return_tensors):
        self.calls.append((len(audio), sampling_rate))
        return SimpleNamespace(input_features=np.ones((1, 4, 6), dtype=np.float64))


def test_clip_dataset_tokenizes_and_loads_lazily(dataset_dir: Path) -> None:
    features = FakeFeatures()
    data = ClipDataset(load_clips(dataset_dir, "train"), features, FakeTokenizer())
    assert len(data) == 4
    assert features.calls == []  # no audio read until an item is requested
    item = data[1]
    assert features.calls == [(2 * SR, SR)]
    assert item["input_features"].shape == (4, 6)
    assert item["input_features"].dtype == np.float32
    assert item["labels"][0] == 1 and item["labels"][-1] == 2
    assert len(item["labels"]) == len("of a matrix") + 2


def test_clip_dataset_drops_labels_over_the_limit(dataset_dir: Path) -> None:
    # "step size" is 9 characters + 2 special tokens = 11 tokens.
    data = ClipDataset(
        load_clips(dataset_dir, "train"),
        FakeFeatures(),
        FakeTokenizer(),
        max_label_tokens=11,
    )
    assert data.dropped_too_long == 3
    assert [clip.text for clip, _ in data.items] == ["step size"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_dry_run_prints_plan(dataset_dir: Path, capsys, tmp_path: Path) -> None:
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({"batch_size": 4, "grad_accum_steps": 1}))
    code = main(
        [
            "finetune",
            "--data-dir",
            str(dataset_dir),
            "--config",
            str(config),
            "--max-steps",
            "3",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "| val | 1 |" in out
    assert "3 steps = 3.0 epochs" in out
    assert "Dry run" in out


def test_cli_reports_errors(tmp_path: Path, capsys) -> None:
    assert main(["finetune", "--data-dir", str(tmp_path), "--dry-run"]) == 1
    assert "finetune-data" in capsys.readouterr().out
    assert main(["finetune", "--data-dir", str(tmp_path), "--lora-r", "0"]) == 1
    assert "lora_r" in capsys.readouterr().out
