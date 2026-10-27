"""Tests for notetaker.benchmark and `notetaker benchmark`.

No real Whisper model is loaded. `benchmark_model` accepts a `loader` and a
`clock`, so the tests hand it a fake model and a fake clock that advances by
known amounts. That makes every timing in the result exactly predictable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from notetaker import benchmark as bm
from notetaker.benchmark import BenchmarkResult, benchmark_model, to_markdown
from notetaker.cli import main
from notetaker.config import get_settings

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)
SR = 16_000


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch):
    for var in (
        "NOTETAKER_LANGUAGE",
        "NOTETAKER_WHISPER_DEVICE",
        "NOTETAKER_WHISPER_COMPUTE_TYPE",
    ):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeModel:
    """Same `.transcribe()` shape as faster-whisper, fixed two-word output."""

    def transcribe(self, audio, **kwargs):
        segment = SimpleNamespace(
            id=0,
            start=0.0,
            end=1.0,
            text=" Hello world",
            avg_logprob=-0.1,
            no_speech_prob=0.0,
            words=None,
        )
        info = SimpleNamespace(language="en", language_probability=1.0, duration=1.0)
        return iter([segment]), info


def fake_loader(model_name, device, compute_type):
    """Module-level (not nested) so it can be pickled into a child process."""
    return FakeModel()


class FakeClock:
    """Each call returns the next value from a list, like a stopwatch."""

    def __init__(self, ticks):
        self.ticks = list(ticks)

    def __call__(self):
        return self.ticks.pop(0)


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# BenchmarkResult arithmetic
# ---------------------------------------------------------------------------


def make_result(**overrides):
    values = dict(
        model="tiny",
        device="cpu",
        compute_type="int8",
        audio_seconds=100.0,
        load_seconds=1.5,
        transcribe_seconds=[30.0, 20.0, 25.0],
        peak_rss_mb=900.0,
        baseline_rss_mb=300.0,
        n_words=42,
    )
    values.update(overrides)
    return BenchmarkResult(**values)


def test_rtf_uses_median_of_repeats():
    r = make_result()
    assert r.median_transcribe_seconds == 25.0
    assert r.rtf == pytest.approx(0.25)
    assert r.speedup == pytest.approx(4.0)


def test_model_rss_is_peak_minus_baseline():
    assert make_result().model_rss_mb == pytest.approx(600.0)
    assert make_result(peak_rss_mb=None).model_rss_mb is None


def test_zero_length_audio_gives_nan_rtf():
    assert np.isnan(make_result(audio_seconds=0.0).rtf)


def test_dict_round_trip_ignores_derived_keys():
    r = make_result()
    data = r.to_dict()
    assert data["rtf"] == pytest.approx(0.25)
    assert BenchmarkResult.from_dict(data) == r


# ---------------------------------------------------------------------------
# benchmark_model with a fake model and clock
# ---------------------------------------------------------------------------


def test_benchmark_model_timings_are_exact():
    # Clock reads: load start/end, then start/end for each of 3 repeats.
    clock = FakeClock([0.0, 2.0, 10.0, 13.0, 20.0, 21.0, 30.0, 32.0])
    result = benchmark_model(
        tone(4.0),
        "base",
        repeats=3,
        device="cpu",
        compute_type="int8",
        loader=fake_loader,
        clock=clock,
    )
    assert result.model == "base"
    assert result.device == "cpu" and result.compute_type == "int8"
    assert result.audio_seconds == pytest.approx(4.0)
    assert result.load_seconds == pytest.approx(2.0)
    assert result.transcribe_seconds == pytest.approx([3.0, 1.0, 2.0])
    assert result.rtf == pytest.approx(0.5)  # median 2 s for 4 s of audio
    assert result.n_segments == 1 and result.n_words == 2


def test_benchmark_model_resolves_auto_device(monkeypatch):
    monkeypatch.setattr(bm, "resolve_device", lambda d: "cpu")
    seen = {}

    def loader(name, device, compute_type):
        seen.update(device=device, compute_type=compute_type)
        return FakeModel()

    benchmark_model(tone(1.0), "tiny", loader=loader)
    assert seen == {"device": "cpu", "compute_type": "int8"}


def test_benchmark_model_rejects_zero_repeats():
    with pytest.raises(ValueError):
        benchmark_model(tone(1.0), "tiny", repeats=0, loader=fake_loader)


def test_peak_rss_is_positive_on_unix():
    value = bm.peak_rss_mb()
    if value is not None:  # None on Windows
        assert value > 10


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_markdown_has_one_row_per_model():
    md = to_markdown([make_result(), make_result(model="small")], "lec.mp3")
    assert "# Whisper benchmark: lec.mp3" in md
    rows = [line for line in md.splitlines() if line.startswith("| tiny")]
    assert len(rows) == 1
    assert "0.250" in rows[0] and "4.0" in rows[0] and "600" in rows[0]
    assert "| small |" in md


def test_markdown_shows_na_without_memory_numbers():
    md = to_markdown([make_result(peak_rss_mb=None)], "x")
    assert "n/a" in md


def test_write_report(tmp_path):
    json_path, md_path = bm.write_report([make_result()], tmp_path / "r", "lec.mp3")
    data = json.loads(json_path.read_text())
    assert data["audio"] == "lec.mp3"
    assert data["results"][0]["model"] == "tiny"
    assert md_path.read_text().startswith("# Whisper benchmark")


# ---------------------------------------------------------------------------
# File-based paths (real ffmpeg, fake model)
# ---------------------------------------------------------------------------


@pytest.fixture
def tone_file(tmp_path):
    path = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=3", str(path)],
        check=True,
    )  # fmt: skip
    return path


@needs_ffmpeg
def test_max_seconds_trims_audio(tone_file):
    audio = bm.load_benchmark_audio(tone_file, max_seconds=1.0)
    assert len(audio) == SR


@needs_ffmpeg
def test_run_isolated_in_child_process(tone_file):
    result = bm.run_isolated(
        tone_file, "tiny", device="cpu", compute_type="int8", loader=fake_loader
    )
    assert result.model == "tiny"
    assert result.audio_seconds == pytest.approx(3.0, abs=0.05)
    assert result.n_words == 2


@needs_ffmpeg
def test_cli_benchmark_end_to_end(tone_file, tmp_path, monkeypatch, capsys):
    # Swap the real loader for the fake one; --no-isolate keeps everything in
    # this process so the monkeypatch applies.
    monkeypatch.setattr(bm, "_default_loader", fake_loader)
    out_dir = tmp_path / "report"
    code = main(
        ["benchmark", str(tone_file), "-m", "tiny", "base", "--no-isolate",
         "--repeats", "2", "-o", str(out_dir)]
    )  # fmt: skip
    assert code == 0
    data = json.loads((out_dir / "benchmark.json").read_text())
    assert [r["model"] for r in data["results"]] == ["tiny", "base"]
    assert all(len(r["transcribe_seconds"]) == 2 for r in data["results"])
    assert "RTF" in capsys.readouterr().out


def test_cli_benchmark_missing_file(capsys):
    assert main(["benchmark", "nope.mp3"]) == 1
    assert "file not found" in capsys.readouterr().out


def test_cli_benchmark_bad_repeats(tmp_path, capsys):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    assert main(["benchmark", str(f), "--repeats", "0"]) == 1


def test_cli_benchmark_rejects_unknown_model():
    with pytest.raises(SystemExit):
        main(["benchmark", "a.wav", "-m", "huge"])


@needs_ffmpeg
def test_cli_benchmark_keeps_going_when_one_model_fails(
    tone_file, tmp_path, monkeypatch, capsys
):
    def flaky_loader(name, device, compute_type):
        if name == "base":
            raise OSError("download blocked")
        return FakeModel()

    monkeypatch.setattr(bm, "_default_loader", flaky_loader)
    out_dir = tmp_path / "report"
    code = main(
        ["benchmark", str(tone_file), "-m", "tiny", "base", "--no-isolate",
         "-o", str(out_dir)]
    )  # fmt: skip
    assert code == 0
    assert "failed: OSError: download blocked" in capsys.readouterr().out
    data = json.loads((out_dir / "benchmark.json").read_text())
    assert [r["model"] for r in data["results"]] == ["tiny"]


@needs_ffmpeg
def test_cli_benchmark_fails_when_every_model_fails(tone_file, monkeypatch, capsys):
    def broken_loader(name, device, compute_type):
        raise OSError("download blocked")

    monkeypatch.setattr(bm, "_default_loader", broken_loader)
    assert main(["benchmark", str(tone_file), "-m", "tiny", "--no-isolate"]) == 1
    assert "no model could be benchmarked" in capsys.readouterr().out


@pytest.mark.slow
@needs_ffmpeg
def test_real_tiny_model_benchmark(tone_file):
    """Downloads the real tiny model (~75 MB). Run locally: pytest -m slow."""
    result = bm.run_isolated(tone_file, "tiny", repeats=1)
    assert result.load_seconds > 0
    assert result.rtf > 0
