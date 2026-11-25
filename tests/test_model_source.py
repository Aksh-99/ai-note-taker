"""Tests for local CTranslate2 model folders as a model source (Step 6.4b).

No real model is loaded. A "model folder" here is a temp folder with empty
placeholder files named like a CTranslate2 export (model.bin, config.json)
plus, where relevant, an export.json like the one `notetaker export-ct2`
writes. That is enough to test everything except faster-whisper itself;
`tests/test_quantize_export.py` runs the real thing on a tiny Whisper.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import notetaker.transcribe as tr
from notetaker import benchmark as bm
from notetaker.benchmark import benchmark_model, to_markdown
from notetaker.cli import describe_model, main
from notetaker.config import get_settings
from notetaker.model_source import (
    ModelSourceError,
    auto_compute_type,
    dir_size_mb,
    display_name,
    model_size_mb,
    parse_model_arg,
    resolve_model_source,
    stored_quantization,
)
from notetaker.transcribe import Segment, Transcript, resolve_compute_type

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Default settings, and run inside tmp_path so outputs land there."""
    monkeypatch.chdir(tmp_path)
    for var in (
        "NOTETAKER_OUTPUT_DIR",
        "NOTETAKER_WHISPER_MODEL",
        "NOTETAKER_WHISPER_DEVICE",
        "NOTETAKER_WHISPER_COMPUTE_TYPE",
        "NOTETAKER_LANGUAGE",
    ):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    tr.load_model.cache_clear()
    yield
    get_settings.cache_clear()
    tr.load_model.cache_clear()


def make_export(
    folder: Path, quantization: str | None = "int8", weights_bytes: int = 2**20
) -> Path:
    """Fake export-ct2 output: placeholder files with a known total size."""
    folder.mkdir(parents=True)
    (folder / "model.bin").write_bytes(b"\0" * weights_bytes)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    if quantization is not None:
        summary = {"quantization": quantization, "base_model": "openai/whisper-small"}
        (folder / "export.json").write_text(json.dumps(summary), encoding="utf-8")
    return folder


@pytest.fixture
def export_dir(tmp_path) -> Path:
    return make_export(tmp_path / "models" / "whisper-small-lora-ct2-int8")


# ---------------------------------------------------------------------------
# resolve_model_source: what a string names
# ---------------------------------------------------------------------------


def test_size_names_are_hub_models():
    source = resolve_model_source("small")
    assert (source.spec, source.label, source.local) == ("small", "small", False)
    assert source.quantization is None and source.size_mb is None


def test_export_folder_is_read(export_dir):
    source = resolve_model_source(str(export_dir))
    assert source.local
    assert source.spec == str(export_dir)
    assert source.label == "whisper-small-lora-ct2-int8"
    assert source.quantization == "int8"
    # 1 MiB of weights plus two tiny JSON files.
    assert source.size_mb == pytest.approx(1.0, abs=0.01)


def test_relative_path_and_dot_get_real_folder_name(export_dir, monkeypatch):
    monkeypatch.chdir(export_dir)
    assert resolve_model_source(".").label == "whisper-small-lora-ct2-int8"
    monkeypatch.chdir(export_dir.parent)
    assert resolve_model_source("whisper-small-lora-ct2-int8/").local


def test_size_name_wins_over_a_folder_with_the_same_name(tmp_path):
    make_export(tmp_path / "small")  # tmp_path is the working directory
    assert resolve_model_source("small").local is False
    assert resolve_model_source("./small").local is True


def test_folder_without_export_json_still_works(tmp_path):
    folder = make_export(tmp_path / "other-tool", quantization=None)
    source = resolve_model_source(str(folder))
    assert source.local and source.quantization is None


def test_broken_export_json_is_ignored(export_dir):
    (export_dir / "export.json").write_text("{not json", encoding="utf-8")
    assert resolve_model_source(str(export_dir)).quantization is None


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda p: None, "no folder with that path exists"),
        (lambda p: p.write_text("x"), "is a file"),
        (lambda p: p.mkdir(), "missing model.bin, config.json"),
    ],
    ids=["missing", "file", "empty-folder"],
)
def test_bad_paths_explain_themselves(tmp_path, make, message):
    path = tmp_path / "thing"
    make(path)
    with pytest.raises(ModelSourceError, match=message):
        resolve_model_source(str(path))


def test_unknown_name_lists_the_sizes():
    with pytest.raises(ModelSourceError, match="tiny, base, small"):
        resolve_model_source("huge")


def test_empty_model_is_rejected():
    with pytest.raises(ModelSourceError, match="empty"):
        resolve_model_source("  ")


def test_adapter_folder_points_to_export_ct2(tmp_path):
    # The output of `notetaker finetune` is an adapter, not a CT2 model.
    adapter = tmp_path / "whisper-small-lora" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    with pytest.raises(ModelSourceError, match="notetaker export-ct2"):
        resolve_model_source(str(adapter))


def test_lenient_helpers_never_raise(tmp_path, export_dir):
    assert stored_quantization("small") is None
    assert stored_quantization(str(tmp_path / "nope")) is None
    assert stored_quantization(str(export_dir)) == "int8"
    assert display_name("tiny") == "tiny"
    assert display_name("not/a/folder") == "not/a/folder"
    assert display_name(str(export_dir)) == "whisper-small-lora-ct2-int8"


def test_dir_size_counts_nested_files(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "x").write_bytes(b"\0" * 2**19)
    (tmp_path / "a" / "b" / "y").write_bytes(b"\0" * 2**19)
    assert dir_size_mb(tmp_path / "a") == pytest.approx(1.0)


def test_model_size_never_downloads(export_dir, monkeypatch):
    calls = []

    def fake_download(name, local_files_only=False):
        calls.append(local_files_only)
        raise FileNotFoundError("not in the cache")

    import faster_whisper.utils

    monkeypatch.setattr(faster_whisper.utils, "download_model", fake_download)
    assert model_size_mb("tiny") is None
    assert calls == [True]  # only ever asked the local cache
    assert model_size_mb(str(export_dir)) == pytest.approx(1.0, abs=0.01)


def test_model_size_of_a_cached_hub_model(tmp_path, monkeypatch):
    cached = make_export(tmp_path / "hf-cache" / "tiny", weights_bytes=3 * 2**20)
    import faster_whisper.utils

    monkeypatch.setattr(
        faster_whisper.utils, "download_model", lambda name, **kw: str(cached)
    )
    assert model_size_mb("tiny") == pytest.approx(3.0, abs=0.01)


# ---------------------------------------------------------------------------
# Compute type for "auto"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stored", "device", "expected"),
    [
        (None, "cpu", "int8"),  # the original rule is unchanged
        (None, "cuda", "float16"),
        ("int8", "cpu", "int8"),
        ("float32", "cpu", "float32"),  # not silently squeezed to int8
        ("float16", "cpu", "float32"),  # CPUs have no fast float16 math
        ("int8_float16", "cpu", "int8"),
        ("int8", "cuda", "int8_float16"),
        ("float16", "cuda", "float16"),
        ("float32", "cuda", "float32"),
        ("bfloat16", "cpu", "int8"),  # unknown storage: fall back
    ],
)
def test_auto_compute_type(stored, device, expected):
    assert auto_compute_type(stored, device) == expected
    assert resolve_compute_type("auto", device, stored=stored) == expected


def test_explicit_compute_type_always_wins():
    assert resolve_compute_type("float32", "cpu", stored="int8") == "float32"


def test_load_model_passes_the_folder_and_its_precision(tmp_path, monkeypatch):
    """load_model must hand faster-whisper the path and a matching precision."""
    import faster_whisper

    seen = {}

    class RecordingWhisperModel:
        def __init__(self, name, device, compute_type):
            seen.update(name=name, device=device, compute_type=compute_type)

    monkeypatch.setattr(faster_whisper, "WhisperModel", RecordingWhisperModel)
    folder = make_export(tmp_path / "ct2-float32", quantization="float32")
    tr.load_model(str(folder), device="cpu")
    assert seen == {"name": str(folder), "device": "cpu", "compute_type": "float32"}


# ---------------------------------------------------------------------------
# transcribe / benchmark with a local folder (fake model, no weights loaded)
# ---------------------------------------------------------------------------


class FakeModel:
    def transcribe(self, audio, **kwargs):
        segment = SimpleNamespace(
            id=1, start=0.0, end=1.0, text=" Eigenvalues", avg_logprob=-0.1,
            no_speech_prob=0.0, words=None,
        )  # fmt: skip
        info = SimpleNamespace(language="en", language_probability=1.0, duration=1.0)
        return iter([segment]), info


def test_transcript_records_the_folder_name_not_the_path(export_dir):
    audio = np.zeros(16_000, dtype=np.float32)
    result = tr.transcribe(audio, model=FakeModel(), model_name=str(export_dir))
    assert result.model == "whisper-small-lora-ct2-int8"


def test_benchmark_result_for_a_local_export(export_dir):
    seen = {}

    def loader(name, device, compute_type):
        seen.update(name=name, compute_type=compute_type)
        return FakeModel()

    result = benchmark_model(
        np.zeros(16_000, dtype=np.float32), str(export_dir), device="cpu",
        loader=loader,
    )  # fmt: skip
    assert seen == {"name": str(export_dir), "compute_type": "int8"}
    assert result.model == "whisper-small-lora-ct2-int8"
    assert result.model_path == str(export_dir.resolve())
    assert result.quantization == "int8"
    assert result.model_size_mb == pytest.approx(1.0, abs=0.01)
    # Old benchmark.json files (no new keys) still load.
    old = {k: v for k, v in result.to_dict().items() if k != "model_path"}
    assert bm.BenchmarkResult.from_dict(old).model_path is None


def test_markdown_has_stored_and_size_columns(export_dir):
    local = bm.BenchmarkResult(
        model="whisper-small-lora-ct2-int8", device="cpu", compute_type="int8",
        audio_seconds=60.0, load_seconds=1.0, transcribe_seconds=[6.0],
        quantization="int8", model_size_mb=233.4,
    )  # fmt: skip
    hub = bm.BenchmarkResult(
        model="small", device="cpu", compute_type="int8", audio_seconds=60.0,
        load_seconds=1.0, transcribe_seconds=[9.0],
    )  # fmt: skip
    md = to_markdown([hub, local], "lec.mp3")
    assert "| Model | Stored | Compute | Size (MB) |" in md
    assert "| whisper-small-lora-ct2-int8 | int8 | int8 | 233 |" in md
    assert "| small | - | int8 | n/a |" in md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_model_arg_turns_errors_into_usage_errors(export_dir):
    assert parse_model_arg("base") == "base"
    assert parse_model_arg(str(export_dir)) == str(export_dir)
    with pytest.raises(argparse.ArgumentTypeError, match="not a Whisper size"):
        parse_model_arg("huge")


def test_describe_model(export_dir, tmp_path):
    assert describe_model("small") == "whisper-small"
    assert describe_model(str(export_dir)) == (
        "local model whisper-small-lora-ct2-int8 (int8)"
    )
    plain = make_export(tmp_path / "plain", quantization=None)
    assert describe_model(str(plain)) == "local model plain"


CANNED = Transcript(
    segments=[
        Segment(id=1, start=0.0, end=1.0, text="Hi.", avg_logprob=-0.1,
                no_speech_prob=0.0)
    ],
    language="en", language_probability=1.0, duration=1.0, model="x",
)  # fmt: skip


def test_cli_transcribe_accepts_an_export_folder(
    export_dir, tmp_path, monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(
        tr, "transcribe", lambda audio, **kw: calls.append(kw) or CANNED
    )
    audio = tmp_path / "lec.mp3"
    audio.write_bytes(b"fake")  # never decoded: transcribe is faked
    assert main(["transcribe", str(audio), "-m", str(export_dir)]) == 0
    assert calls[0]["model_name"] == str(export_dir)
    assert "local model whisper-small-lora-ct2-int8 (int8)" in capsys.readouterr().out


def test_cli_transcribe_rejects_an_adapter_folder(tmp_path, capsys):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    audio = tmp_path / "lec.mp3"
    audio.write_bytes(b"fake")
    with pytest.raises(SystemExit) as exit_info:
        main(["transcribe", str(audio), "-m", str(adapter)])
    assert exit_info.value.code == 2  # argparse usage error, before any work
    assert "export-ct2" in capsys.readouterr().err


@needs_ffmpeg
def test_cli_benchmark_compares_a_size_and_two_exports(tmp_path, monkeypatch):
    tone = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=2", str(tone)],
        check=True,
    )  # fmt: skip
    int8 = make_export(tmp_path / "m-int8", "int8")
    fp32 = make_export(tmp_path / "m-float32", "float32", weights_bytes=4 * 2**20)
    used = {}

    def loader(name, device, compute_type):
        used[name] = compute_type
        return FakeModel()

    monkeypatch.setattr(bm, "_default_loader", loader)
    out = tmp_path / "report"
    code = main(
        ["benchmark", str(tone), "-m", "tiny", str(int8), str(fp32),
         "--no-isolate", "-o", str(out)]
    )  # fmt: skip
    assert code == 0
    # On this CPU-only test, "auto" runs each export at its own precision.
    assert used == {"tiny": "int8", str(int8): "int8", str(fp32): "float32"}
    rows = json.loads((out / "benchmark.json").read_text())["results"]
    assert [r["model"] for r in rows] == ["tiny", "m-int8", "m-float32"]
    assert [r["quantization"] for r in rows] == [None, "int8", "float32"]
    assert rows[2]["model_size_mb"] == pytest.approx(4.0, abs=0.01)
    assert "| m-float32 | float32 | float32 | 4 |" in (out / "benchmark.md").read_text()
