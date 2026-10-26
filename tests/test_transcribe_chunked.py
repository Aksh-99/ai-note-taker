"""Tests for chunked transcription of long recordings (Step 1.4).

`ChunkModel` is a fake Whisper: for every piece of audio it receives, it
reports one segment from 0.5 s to 1.0 s (relative to the piece, exactly like
the real model) with one word. That lets us check the stitching math: each
segment must come back shifted by its chunk's start time.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from notetaker.config import get_settings
from notetaker.transcribe import Segment, Word, shift_segment, transcribe

SR = 16_000


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch):
    for var in ("NOTETAKER_LANGUAGE", "NOTETAKER_CHUNK_MAX_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def pause(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


class ChunkModel:
    def __init__(self, detected_language="en"):
        self.detected_language = detected_language
        self.calls = []  # (number of samples, kwargs) per call

    def transcribe(self, audio, **kwargs):
        self.calls.append((len(audio), kwargs))
        n = len(self.calls)
        word = SimpleNamespace(start=0.5, end=1.0, word=" hello", probability=0.9)
        segment = SimpleNamespace(
            id=1,  # every chunk restarts at id 1, like the real model
            start=0.5,
            end=1.0,
            text=f" piece {n}",
            avg_logprob=-0.2,
            no_speech_prob=0.01,
            words=[word],
        )
        info = SimpleNamespace(
            language=kwargs["language"] or self.detected_language,
            language_probability=0.87,
            duration=len(audio) / SR,
        )
        return iter([segment]), info


# Pauses at 3.0-4.0 s and 6.5-7.0 s; with max 5 s the cuts are at 3.5 and
# 6.75 s (see test_chunking.py), giving three chunks.
LECTURE = np.concatenate([tone(3.0), pause(1.0), tone(2.5), pause(0.5), tone(3.0)])


def test_long_audio_is_split_and_timestamps_are_shifted():
    model = ChunkModel()
    result = transcribe(LECTURE, model=model, max_chunk_s=5)

    assert len(model.calls) == 3
    # The pieces handed to the model add up to the whole recording.
    assert sum(n for n, _ in model.calls) == len(LECTURE)
    assert [s.text for s in result.segments] == ["piece 1", "piece 2", "piece 3"]
    assert [s.id for s in result.segments] == [1, 2, 3]  # unique, not 1, 1, 1
    # Segment at 0.5 s inside chunks starting at 0, 3.5 and 6.75 s.
    starts = [s.start for s in result.segments]
    assert starts == pytest.approx([0.5, 4.0, 7.25], abs=0.03)
    # Words move with their segment.
    assert result.segments[2].words[0].start == pytest.approx(7.25, abs=0.03)
    assert result.duration == pytest.approx(10.0)


def test_max_chunk_zero_sends_everything_at_once():
    model = ChunkModel()
    result = transcribe(LECTURE, model=model, max_chunk_s=0)
    assert len(model.calls) == 1
    assert model.calls[0][0] == len(LECTURE)
    assert result.segments[0].start == 0.5


def test_default_chunk_length_comes_from_settings(monkeypatch):
    monkeypatch.setenv("NOTETAKER_CHUNK_MAX_SECONDS", "5")
    get_settings.cache_clear()
    model = ChunkModel()
    transcribe(LECTURE, model=model)
    assert len(model.calls) == 3


def test_silent_chunks_are_skipped_but_time_still_counts():
    # 4 s speech, 8 s of silence, 4 s speech: the middle chunk is pure silence.
    audio = np.concatenate([tone(4.0), pause(8.0), tone(4.0)])
    model = ChunkModel()
    result = transcribe(audio, model=model, max_chunk_s=5)
    # The planner makes 4 chunks here; the 2 middle ones are pure silence and
    # are never sent, so the model runs only twice.
    assert len(model.calls) == 2
    assert result.duration == pytest.approx(16.0)
    # The last segment still lands in the final stretch of speech.
    assert result.segments[-1].start > 11.0


def test_detected_language_is_locked_for_later_chunks(monkeypatch):
    monkeypatch.setenv("NOTETAKER_LANGUAGE", "")  # blank = auto-detect
    get_settings.cache_clear()
    model = ChunkModel(detected_language="fr")
    result = transcribe(LECTURE, model=model, max_chunk_s=5)
    languages = [kwargs["language"] for _, kwargs in model.calls]
    assert languages == [None, "fr", "fr"]
    assert result.language == "fr"
    assert result.language_probability == 0.87


def test_progress_callback_sees_every_sent_chunk():
    seen = []

    def record(index, total, chunk):
        seen.append((index, total, chunk.start))

    transcribe(LECTURE, model=ChunkModel(), max_chunk_s=5, on_chunk=record)
    assert [(i, total) for i, total, _ in seen] == [(1, 3), (2, 3), (3, 3)]
    assert [start for _, _, start in seen] == pytest.approx([0, 3.5, 6.75], abs=0.03)


def test_negative_max_chunk_is_rejected():
    with pytest.raises(ValueError, match="max_chunk_s"):
        transcribe(LECTURE, model=ChunkModel(), max_chunk_s=-1)


def test_shift_segment_moves_segment_and_words():
    seg = Segment(
        id=1, start=1.0, end=2.0, text="x", avg_logprob=-0.1, no_speech_prob=0.0,
        words=[Word(start=1.2, end=1.5, text="x", probability=0.9)],
    )  # fmt: skip
    moved = shift_segment(seg, 600.0, new_id=42)
    assert (moved.id, moved.start, moved.end) == (42, 601.0, 602.0)
    assert (moved.words[0].start, moved.words[0].end) == (601.2, 601.5)
    assert seg.start == 1.0  # the original is untouched (frozen dataclass)
