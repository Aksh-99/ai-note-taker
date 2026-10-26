"""Tests for `notetaker transcribe`.

No real Whisper model is ever loaded here. Most tests replace the
`transcribe` function with a fake that returns a fixed Transcript; the
end-to-end test runs the real audio pipeline (ffmpeg) but swaps the model for
a fake one, so it checks everything except the neural network itself.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

import notetaker.transcribe as tr
from notetaker.audio import AudioLoadError
from notetaker.cli import main
from notetaker.config import get_settings
from notetaker.transcribe import Segment, Transcript

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)

CANNED = Transcript(
    segments=[
        Segment(
            id=0,
            start=0.0,
            end=1.5,
            text="Welcome to linear algebra.",
            avg_logprob=-0.2,
            no_speech_prob=0.01,
        )
    ],
    language="en",
    language_probability=0.99,
    duration=1.5,
    model="tiny",
)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Run each test inside a temp folder with default settings.

    `monkeypatch.chdir` makes the default `outputs/` folder land in tmp_path,
    so tests never write into the real repo.
    """
    monkeypatch.chdir(tmp_path)
    for var in (
        "NOTETAKER_OUTPUT_DIR",
        "NOTETAKER_WHISPER_MODEL",
        "NOTETAKER_LANGUAGE",
    ):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_transcribe(monkeypatch):
    """Replace notetaker.transcribe.transcribe and record how it was called."""
    calls = []

    def fake(audio, **kwargs):
        calls.append((audio, kwargs))
        return CANNED

    monkeypatch.setattr(tr, "transcribe", fake)
    return calls


@pytest.fixture
def audio_file(tmp_path):
    # The fake never reads the file, it only has to exist.
    path = tmp_path / "lecture01.mp3"
    path.write_bytes(b"not really audio")
    return path


def test_writes_json_and_markdown_to_default_folder(
    audio_file, fake_transcribe, capsys
):
    assert main(["transcribe", str(audio_file)]) == 0
    out = audio_file.parent / "outputs" / "lecture01"
    data = json.loads((out / "transcript.json").read_text(encoding="utf-8"))
    assert data["segments"][0]["text"] == "Welcome to linear algebra."
    md = (out / "transcript.md").read_text(encoding="utf-8")
    assert md.startswith("# lecture01\n")  # title defaults to the file name
    assert "**[00:00:00]** Welcome to linear algebra." in md
    assert "Done: 1 segments" in capsys.readouterr().out


def test_options_are_passed_through(audio_file, fake_transcribe, tmp_path):
    out = tmp_path / "custom"
    code = main(
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(out),
            "--model",
            "tiny",
            "--language",
            "en",
            "--beam-size",
            "1",
            "--title",
            "Lecture 1: Vectors",
        ]
    )
    assert code == 0
    _, kwargs = fake_transcribe[0]
    # on_chunk is a function (the progress printer), so check it separately.
    assert callable(kwargs.pop("on_chunk"))
    assert kwargs == {
        "language": "en",
        "beam_size": 1,
        "model_name": "tiny",
        "max_chunk_s": None,  # None = use NOTETAKER_CHUNK_MAX_SECONDS
    }
    assert (out / "transcript.md").read_text().startswith("# Lecture 1: Vectors")


def test_model_defaults_to_settings(audio_file, fake_transcribe, monkeypatch):
    monkeypatch.setenv("NOTETAKER_WHISPER_MODEL", "base")
    get_settings.cache_clear()
    assert main(["transcribe", str(audio_file)]) == 0
    assert fake_transcribe[0][1]["model_name"] == "base"


def test_missing_file_fails_cleanly(fake_transcribe, capsys):
    assert main(["transcribe", "nope.mp3"]) == 1
    assert "file not found" in capsys.readouterr().out
    assert fake_transcribe == []  # never tried to transcribe


def test_bad_beam_size_fails_cleanly(audio_file, fake_transcribe):
    assert main(["transcribe", str(audio_file), "--beam-size", "0"]) == 1


def test_max_chunk_seconds_is_passed_through(audio_file, fake_transcribe):
    assert main(["transcribe", str(audio_file), "--max-chunk-seconds", "120"]) == 0
    assert fake_transcribe[0][1]["max_chunk_s"] == 120.0


def test_negative_max_chunk_seconds_fails_cleanly(audio_file, fake_transcribe):
    assert main(["transcribe", str(audio_file), "--max-chunk-seconds", "-5"]) == 1
    assert fake_transcribe == []


def test_progress_printer_only_speaks_for_multi_chunk_runs(capsys):
    from notetaker.chunking import Chunk
    from notetaker.cli import print_chunk_progress

    chunk = Chunk(start_sample=9_600_000, end_sample=14_400_000, sample_rate=16000,
                  speech_ratio=0.8)  # fmt: skip
    print_chunk_progress(1, 1, chunk)
    assert capsys.readouterr().out == ""
    print_chunk_progress(3, 18, chunk)
    assert capsys.readouterr().out == "  chunk 3/18 [00:10:00 - 00:15:00]\n"


def test_unknown_model_is_rejected_by_argparse(audio_file):
    with pytest.raises(SystemExit) as exit_info:
        main(["transcribe", str(audio_file), "--model", "huge"])
    assert exit_info.value.code == 2


def test_audio_errors_become_exit_code_1(audio_file, monkeypatch, capsys):
    def broken(audio, **kwargs):
        raise AudioLoadError("ffmpeg could not read lecture01.mp3")

    monkeypatch.setattr(tr, "transcribe", broken)
    assert main(["transcribe", str(audio_file)]) == 1
    assert "could not read" in capsys.readouterr().out


class FakeModel:
    """Same shape as faster-whisper's WhisperModel.transcribe()."""

    def transcribe(self, audio, **kwargs):
        self.samples = len(audio)
        segment = SimpleNamespace(
            id=0,
            start=0.0,
            end=0.5,
            text=" A tone.",
            avg_logprob=-0.3,
            no_speech_prob=0.2,
            words=[],
        )
        info = SimpleNamespace(
            language="en", language_probability=0.9, duration=len(audio) / 16000
        )
        return iter([segment]), info


@needs_ffmpeg
def test_end_to_end_with_real_audio_and_fake_model(tmp_path, monkeypatch):
    # Generate a half-second 440 Hz tone. Media files are never committed.
    wav = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=0.5", str(wav)],
        check=True,
    )  # fmt: skip
    model = FakeModel()
    # load_model is what transcribe() calls when no model is passed in.
    monkeypatch.setattr(tr, "load_model", lambda name=None: model)

    assert main(["transcribe", str(wav), "-o", str(tmp_path / "out")]) == 0
    assert model.samples == 8000  # 0.5 s at 16 kHz reached the model
    loaded = Transcript.load_json(tmp_path / "out" / "transcript.json")
    assert loaded.segments[0].text == "A tone."
    assert loaded.duration == 0.5
