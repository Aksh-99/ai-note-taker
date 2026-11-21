"""Training tests for LoRA fine-tuning (Step 6.2a), on a tiny random Whisper.

Skipped unless the optional [finetune] extra (torch, transformers, peft) is
installed. Nothing is downloaded: the model is built from a small
WhisperConfig with random weights (about 37k parameters), the feature
extractor is constructed locally, and the tokenizer is a character-level
fake. That is enough to check the mechanics that matter: label padding,
which weights train, that the loss goes down, and what gets saved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from notetaker import finetune  # noqa: E402  (imports must follow the skips)
from notetaker.cli import main  # noqa: E402
from notetaker.finetune import (  # noqa: E402
    IGNORE_INDEX,
    ClipDataset,
    FinetuneConfig,
    WhisperCollator,
    apply_lora,
    count_parameters,
    evaluate_loss,
    load_clips,
    run_finetune,
)

# pytest puts tests/ on sys.path (its default "prepend" import mode), so the
# helpers in test_finetune.py can be imported by module name.
from test_finetune import FakeTokenizer, make_split  # noqa: E402

START, END = 1, 2  # token ids used by FakeTokenizer


def tiny_whisper() -> object:
    config = transformers.WhisperConfig(
        vocab_size=100,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        num_mel_bins=80,
        max_source_positions=1500,  # 3000 mel frames / 2 after the conv stride
        max_target_positions=64,
        pad_token_id=0,
        bos_token_id=START,
        eos_token_id=END,
        decoder_start_token_id=START,
        # Larger random weights than the default 0.02. The frozen output
        # layer (the token embeddings) then gives logits big enough that
        # adapters on q/v can move the predictions; with tiny random
        # embeddings every logit is near 0 and the loss cannot fall far below
        # ln(vocab size). Real Whisper has trained embeddings, so no issue.
        init_std=0.5,
    )
    torch.manual_seed(0)
    return transformers.WhisperForConditionalGeneration(config)


def tiny_loader(config: FinetuneConfig):
    return tiny_whisper(), transformers.WhisperFeatureExtractor(), FakeTokenizer()


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    root = tmp_path / "finetune"
    make_split(root, "train", [("lec-a", 0.5, "abc"), ("lec-a", 0.6, "abd")])
    make_split(root, "val", [("lec-b", 0.5, "abc")])
    return root


def small_config(**changes) -> FinetuneConfig:
    base = dict(
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        learning_rate=3e-2,
        batch_size=2,
        grad_accum_steps=1,
        max_steps=25,
        warmup_steps=0,
        eval_every=5,
        fp16=False,
    )
    return FinetuneConfig(**{**base, **changes})


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


def test_collator_strips_start_token_and_pads_with_ignore_index() -> None:
    examples = [
        {
            "input_features": np.zeros((80, 3000), np.float32),
            "labels": [START, 5, 6, END],
        },
        {"input_features": np.ones((80, 3000), np.float32), "labels": [START, 7, END]},
    ]
    batch = WhisperCollator(decoder_start_token_id=START)(examples)
    assert batch["input_features"].shape == (2, 80, 3000)
    assert batch["labels"].tolist() == [[5, 6, END], [7, END, IGNORE_INDEX]]
    assert batch["labels"].dtype == torch.long


def test_collator_keeps_labels_without_start_token() -> None:
    examples = [{"input_features": np.zeros((2, 2), np.float32), "labels": [5, END]}]
    batch = WhisperCollator(decoder_start_token_id=START)(examples)
    assert batch["labels"].tolist() == [[5, END]]


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


def test_lora_trains_only_adapter_weights() -> None:
    model = apply_lora(tiny_whisper(), small_config())
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)
    # Adapters on q_proj and v_proj only, in encoder and decoder.
    assert any("q_proj.lora_A" in n for n in trainable_names)
    assert not any("k_proj" in n or "out_proj" in n for n in trainable_names)
    trainable, total = count_parameters(model)
    assert 0 < trainable < total


def test_lora_starts_as_the_unchanged_model() -> None:
    """B is initialised to zero, so the wrapped model's output is identical."""
    features = torch.randn(1, 80, 3000)
    labels = torch.tensor([[5, 6, END]])
    plain = tiny_whisper().eval()
    wrapped = apply_lora(tiny_whisper(), small_config()).eval()
    with torch.no_grad():
        a = plain(input_features=features, labels=labels).loss
        b = wrapped(input_features=features, labels=labels).loss
    assert float(a) == pytest.approx(float(b), abs=1e-6)


def test_evaluate_loss_is_token_weighted(dataset_dir: Path) -> None:
    model = apply_lora(tiny_whisper(), small_config())
    data = ClipDataset(
        load_clips(dataset_dir, "train"),
        transformers.WhisperFeatureExtractor(),
        FakeTokenizer(),
    )
    collate = WhisperCollator(START)
    one_batch = torch.utils.data.DataLoader(data, batch_size=2, collate_fn=collate)
    per_item = torch.utils.data.DataLoader(data, batch_size=1, collate_fn=collate)
    # Same labels lengths here, so batch-of-2 and one-at-a-time must agree.
    assert evaluate_loss(model, one_batch, "cpu") == pytest.approx(
        evaluate_loss(model, per_item, "cpu"), rel=1e-4
    )
    assert model.training  # evaluate_loss switches back to training mode


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------


def test_run_finetune_lowers_loss_and_saves_adapter(
    dataset_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    logs: list[str] = []
    summary = run_finetune(
        small_config(),
        dataset_dir,
        out,
        device="cpu",
        load_model=tiny_loader,
        log=logs.append,
    )
    history = summary["history"]
    assert len(history) == 25
    # Overfitting two short clips: the loss must fall a lot.
    assert history[-1]["train_loss"] < 0.5 * history[0]["train_loss"]
    val_rows = [r for r in history if r["val_loss"] is not None]
    assert [r["step"] for r in val_rows] == [5, 10, 15, 20, 25]
    assert summary["best_val_loss"] == pytest.approx(
        min(r["val_loss"] for r in val_rows), abs=1e-4
    )
    assert summary["parameters"]["trainable"] < summary["parameters"]["total"]
    assert summary["dropped_too_long"] == {"train": 0, "val": 0}
    assert (out / "adapter" / "adapter_config.json").is_file()
    saved = json.loads((out / "finetune.json").read_text())
    assert saved["best_step"] == summary["best_step"]
    assert saved["config"]["lora_r"] == 4
    assert any("Trainable parameters" in line for line in logs)
    assert any("CPU" in line for line in logs)


def test_run_finetune_is_reproducible(dataset_dir: Path, tmp_path: Path) -> None:
    config = small_config(max_steps=4, eval_every=2)
    a = run_finetune(
        config,
        dataset_dir,
        tmp_path / "a",
        device="cpu",
        load_model=tiny_loader,
        log=lambda _: None,
    )
    b = run_finetune(
        config,
        dataset_dir,
        tmp_path / "b",
        device="cpu",
        load_model=tiny_loader,
        log=lambda _: None,
    )
    losses = [[r["train_loss"] for r in s["history"]] for s in (a, b)]
    assert losses[0] == pytest.approx(losses[1], rel=1e-5)


def test_saved_adapter_reloads_onto_the_base_model(
    dataset_dir: Path, tmp_path: Path
) -> None:
    peft = pytest.importorskip("peft")
    run_finetune(
        small_config(max_steps=10),
        dataset_dir,
        tmp_path / "out",
        device="cpu",
        load_model=tiny_loader,
        log=lambda _: None,
    )
    reloaded = peft.PeftModel.from_pretrained(
        tiny_whisper(), str(tmp_path / "out" / "adapter")
    )
    lora_b = [p for n, p in reloaded.named_parameters() if "lora_B" in n]
    # Training moved B away from its all-zero start, and that was saved.
    assert lora_b and any(float(p.abs().sum()) > 0 for p in lora_b)


def test_run_finetune_refuses_to_overwrite(dataset_dir: Path, tmp_path: Path) -> None:
    config = small_config(max_steps=1, eval_every=1)
    out = tmp_path / "out"
    run_finetune(
        config,
        dataset_dir,
        out,
        device="cpu",
        load_model=tiny_loader,
        log=lambda _: None,
    )
    with pytest.raises(finetune.FinetuneError, match="exists"):
        run_finetune(config, dataset_dir, out, device="cpu", load_model=tiny_loader)
    run_finetune(
        config,
        dataset_dir,
        out,
        device="cpu",
        load_model=tiny_loader,
        force=True,
        log=lambda _: None,
    )


def test_run_finetune_when_every_label_is_too_long(
    dataset_dir: Path, tmp_path: Path
) -> None:
    with pytest.raises(finetune.FinetuneError, match="token limit"):
        run_finetune(
            small_config(max_label_tokens=3),
            dataset_dir,
            tmp_path / "o",
            device="cpu",
            load_model=tiny_loader,
            log=lambda _: None,
        )


def test_cli_runs_training(
    dataset_dir: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(finetune, "_load_pretrained", tiny_loader)
    out = tmp_path / "out"
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps(small_config().to_json()))
    code = main(
        [
            "finetune",
            "--data-dir",
            str(dataset_dir),
            "-o",
            str(out),
            "--config",
            str(config),
            "--max-steps",
            "2",
            "--eval-every",
            "1",
            "--device",
            "cpu",
        ]
    )
    text = capsys.readouterr().out
    assert code == 0, text
    assert "step 2/2" in text
    assert (out / "adapter").is_dir()
    assert json.loads((out / "finetune.json").read_text())["config"]["max_steps"] == 2
