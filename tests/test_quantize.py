"""Tests for the CTranslate2 export (Step 6.4a) that need no PyTorch.

These cover argument checks, the folder-safety rules and the CLI wiring, so
they run in the main CI job. The real merge + convert + load tests, which
need the [finetune] extra, are in test_quantize_export.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from notetaker import quantize
from notetaker.cli import main
from notetaker.finetune import FinetuneConfig, FinetuneError
from notetaker.quantize import (
    EXPORT_SUMMARY,
    convert_to_ct2,
    dir_size_mb,
    export_ct2,
)


def fake_model_dir(root: Path) -> Path:
    """A folder that passes export_ct2's first check (has finetune.json)."""
    root.mkdir(parents=True)
    summary = {"config": FinetuneConfig().to_json()}
    (root / "finetune.json").write_text(json.dumps(summary))
    (root / "adapter").mkdir()
    return root


def test_dir_size_mb_counts_nested_files(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x" * 2**20)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"x" * 2**19)
    assert dir_size_mb(tmp_path) == pytest.approx(1.5)
    assert dir_size_mb(tmp_path / "sub") == pytest.approx(0.5)


def test_convert_rejects_unknown_quantization(tmp_path: Path) -> None:
    with pytest.raises(FinetuneError, match="Unknown quantization"):
        convert_to_ct2(tmp_path, tmp_path / "out", quantization="int4")


def test_convert_needs_tokenizer_and_preprocessor_files(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text("{}")
    with pytest.raises(FinetuneError, match="preprocessor_config.json"):
        convert_to_ct2(tmp_path, tmp_path / "out")


def test_export_needs_a_finished_finetune_run(tmp_path: Path) -> None:
    with pytest.raises(FinetuneError, match="run `notetaker finetune`"):
        export_ct2(tmp_path, tmp_path / "out", log=lambda _: None)


def test_export_rejects_unknown_quantization(tmp_path: Path) -> None:
    model_dir = fake_model_dir(tmp_path / "m")
    with pytest.raises(FinetuneError, match="Unknown quantization"):
        export_ct2(model_dir, tmp_path / "out", quantization="int3")


def test_export_refuses_to_write_into_the_model_dir(tmp_path: Path) -> None:
    model_dir = fake_model_dir(tmp_path / "m")
    with pytest.raises(FinetuneError, match="different folder"):
        export_ct2(model_dir, model_dir, force=True)


def test_export_will_not_overwrite_without_force(tmp_path: Path) -> None:
    model_dir = fake_model_dir(tmp_path / "m")
    out = tmp_path / "out"
    out.mkdir()
    (out / EXPORT_SUMMARY).write_text("{}")
    with pytest.raises(FinetuneError, match="force=True"):
        export_ct2(model_dir, out)


def test_force_never_deletes_a_folder_that_is_not_an_export(tmp_path: Path) -> None:
    model_dir = fake_model_dir(tmp_path / "m")
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "notes.txt").write_text("keep me")
    with pytest.raises(FinetuneError, match="refusing to delete"):
        export_ct2(model_dir, precious, force=True)
    assert (precious / "notes.txt").read_text() == "keep me"


def test_cli_default_output_folder_and_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    calls = {}

    def fake_export(model_dir, out, **kwargs):
        calls.update(model_dir=model_dir, out=out, **kwargs)
        return quantize.ExportResult(
            out_dir=str(out),
            source_model_dir=str(model_dir),
            base_model="openai/whisper-small",
            quantization=kwargs["quantization"],
            merged_mb=4.0,
            ct2_mb=1.0,
            seconds=0.1,
            verified=kwargs["verify"],
        )

    monkeypatch.setattr(quantize, "export_ct2", fake_export)
    model_dir = tmp_path / "models" / "whisper-small-lora"
    code = main(["export-ct2", "--model-dir", str(model_dir), "-q", "float16"])
    assert code == 0
    # Default --out sits next to the model folder and names the precision.
    assert calls["out"] == tmp_path / "models" / "whisper-small-lora-ct2-float16"
    assert calls["quantization"] == "float16"
    assert calls["verify"] is True and calls["force"] is False
    assert "export.json" in capsys.readouterr().out

    main(["export-ct2", "--model-dir", str(model_dir), "--out", "x", "--no-verify"])
    assert calls["out"] == Path("x") and calls["verify"] is False
    assert calls["quantization"] == "int8"  # the default


def test_cli_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = main(["export-ct2", "--model-dir", str(tmp_path / "nothing")])
    assert code == 1
    assert "run `notetaker finetune`" in capsys.readouterr().out


def test_cli_rejects_unknown_quantization() -> None:
    with pytest.raises(SystemExit):
        main(["export-ct2", "-q", "int4"])
