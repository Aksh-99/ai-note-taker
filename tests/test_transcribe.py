"""Tests for notetaker.transcribe.

Almost every test here uses `FakeModel`, a stand-in with the same
`.transcribe()` method as faster-whisper's WhisperModel. It returns segments we
wrote by hand, so we can check our conversion logic exactly without
downloading a model. CI runs only these fast tests.

The one test marked `slow` loads the real `tiny` Whisper model (~75 MB
download on first run). Run it locally with `pytest -m slow`.
"""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from notetaker import transcribe as tr
from notetaker.config import get_settings
from notetaker.transcribe import Transcript, Word, transcribe

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch):
    """Make every test start from default settings.

    `autouse=True` applies this fixture to every test in the file. We clear
    the settings cache before and after, so an env var set in one test cannot
    leak into the next.
    """
    monkeypatch.delenv("NOTETAKER_LANGUAGE", raising=False)
    monkeypatch.delenv("NOTETAKER_WHISPER_MODEL", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def raw_word(start, end, word, probability=0.9):
    # SimpleNamespace builds a throwaway object with these attributes, which
    # is all our converter reads from faster-whisper's Word objects.
    return SimpleNamespace(start=start, end=end, word=word, probability=probability)


def raw_segment(id, start, end, text, words=None):
    return SimpleNamespace(
        id=id,
        start=start,
        end=end,
        text=text,
        avg_logprob=-0.21,
        no_speech_prob=0.01,
        words=words,
    )


class FakeModel:
    """Mimics faster-whisper: returns (lazy generator of segments, info).

    It also records the arguments it received, so tests can check what we
    passed to the model.
    """

    def __init__(self, segments, language="en", language_probability=1.0):
        self.segments = segments
        self.info = SimpleNamespace(
            language=language,
            language_probability=language_probability,
            duration=2.5,
        )
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        # A generator, like the real thing: nothing runs until iterated.
        return (s for s in self.segments), self.info


SEGMENTS = [
    raw_segment(
        1,
        0.0,
        1.2,
        " Today we cover eigenvalues.",
        words=[
            raw_word(0.0, 0.31, " Today"),
            raw_word(0.31, 0.5, " we"),
            raw_word(0.5, 0.8, " cover"),
            raw_word(0.8, 1.2, " eigenvalues.", probability=0.71234567),
        ],
    ),
    raw_segment(2, 1.2, 2.5, " Let A be a matrix.", words=None),
]


def silence(seconds=1.0):
    return np.zeros(int(16_000 * seconds), dtype=np.float32)


def test_converts_segments_and_words():
    result = transcribe(silence(), model=FakeModel(SEGMENTS))
    assert isinstance(result, Transcript)
    assert [s.id for s in result.segments] == [1, 2]
    first = result.segments[0]
    # Leading spaces that Whisper adds are stripped.
    assert first.text == "Today we cover eigenvalues."
    assert first.start == 0.0 and first.end == 1.2
    assert first.duration == pytest.approx(1.2)
    assert first.words[3] == Word(0.8, 1.2, "eigenvalues.", 0.7123)
    # Segments without word timestamps get an empty list, not None.
    assert result.segments[1].words == []


def test_transcript_metadata_and_full_text():
    result = transcribe(silence(), model=FakeModel(SEGMENTS, language_probability=0.98))
    assert result.language == "en"
    assert result.language_probability == 0.98
    assert result.duration == 2.5
    assert result.model == "small"  # the configured default
    assert result.text == "Today we cover eigenvalues. Let A be a matrix."


def test_passes_decoding_options_to_model():
    model = FakeModel(SEGMENTS)
    transcribe(silence(), model=model, beam_size=1, word_timestamps=False)
    audio, kwargs = model.calls[0]
    assert audio.dtype == np.float32
    assert kwargs == {"language": "en", "beam_size": 1, "word_timestamps": False}


def test_language_falls_back_to_settings(monkeypatch):
    monkeypatch.setenv("NOTETAKER_LANGUAGE", "fr")
    model = FakeModel(SEGMENTS)
    transcribe(silence(), model=model)
    assert model.calls[0][1]["language"] == "fr"
    # An explicit argument beats the setting.
    transcribe(silence(), model=model, language="de")
    assert model.calls[1][1]["language"] == "de"


def test_blank_language_means_autodetect(monkeypatch):
    monkeypatch.setenv("NOTETAKER_LANGUAGE", "")
    model = FakeModel(SEGMENTS)
    transcribe(silence(), model=model)
    assert model.calls[0][1]["language"] is None


def test_empty_audio_gives_empty_transcript():
    result = transcribe(silence(0.5), model=FakeModel([]))
    assert result.segments == []
    assert result.text == ""


def test_rejects_bad_inputs():
    with pytest.raises(ValueError, match="beam_size"):
        transcribe(silence(), model=FakeModel([]), beam_size=0)
    stereo = np.zeros((16_000, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="1-D"):
        transcribe(stereo, model=FakeModel([]))


def test_json_round_trip(tmp_path):
    original = transcribe(silence(), model=FakeModel(SEGMENTS))
    path = original.save_json(tmp_path / "out" / "transcript.json")
    assert path.is_file()
    assert Transcript.load_json(path) == original


def test_resolve_device_and_precision(monkeypatch):
    assert tr.resolve_device("cpu") == "cpu"
    monkeypatch.setattr(tr, "cuda_available", lambda: False)
    assert tr.resolve_device("auto") == "cpu"
    monkeypatch.setattr(tr, "cuda_available", lambda: True)
    assert tr.resolve_device("auto") == "cuda"
    assert tr.resolve_compute_type("auto", "cpu") == "int8"
    assert tr.resolve_compute_type("auto", "cuda") == "float16"
    assert tr.resolve_compute_type("float32", "cpu") == "float32"


def test_load_model_uses_settings_without_downloading(monkeypatch):
    """Swap the real WhisperModel class for a recorder, so we can check the
    arguments load_model() would use without downloading anything."""
    import faster_whisper

    created = []

    class RecordingWhisperModel:
        def __init__(self, name, device, compute_type):
            created.append((name, device, compute_type))

    monkeypatch.setattr(faster_whisper, "WhisperModel", RecordingWhisperModel)
    monkeypatch.setattr(tr, "cuda_available", lambda: False)
    monkeypatch.setenv("NOTETAKER_WHISPER_MODEL", "base")
    tr.load_model.cache_clear()
    try:
        tr.load_model()
        tr.load_model()  # cached: must not construct a second model
        tr.load_model("tiny", compute_type="float32")
    finally:
        tr.load_model.cache_clear()
    assert created == [("base", "cpu", "int8"), ("tiny", "cpu", "float32")]


@needs_ffmpeg
def test_file_input_is_decoded_at_16khz(tmp_path, monkeypatch):
    """A file path goes through load_audio at 16 kHz, even if the configured
    sample rate is different, because Whisper only accepts 16 kHz."""
    monkeypatch.setenv("NOTETAKER_SAMPLE_RATE", "8000")
    clip = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "aevalsrc=0.5*sin(2*PI*440*t):s=44100:d=0.5", str(clip)],
        check=True,
    )  # fmt: skip
    model = FakeModel([])
    transcribe(clip, model=model)
    audio, _ = model.calls[0]
    assert audio.shape == (8000,)  # 0.5 s x 16000 Hz


@pytest.mark.slow
def test_real_tiny_model_on_silence():
    """End-to-end with the real `tiny` model. Silence should produce little or
    no text, and must not crash. Skipped in CI (downloads ~75 MB)."""
    model = tr.load_model("tiny", device="cpu", compute_type="int8")
    result = transcribe(silence(2.0), model=model, language="en", model_name="tiny")
    assert result.duration == pytest.approx(2.0, abs=0.05)
    assert result.model == "tiny"
    assert all(s.start <= s.end for s in result.segments)
