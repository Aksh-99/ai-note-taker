"""Tests for notetaker.chunking (energy VAD and chunk planning).

All audio is synthesized with NumPy: a sine tone stands in for speech and
zeros (or very faint noise) stand in for a pause. That way every expected cut
point is known exactly in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

from notetaker.chunking import (
    Chunk,
    frame_energy_db,
    plan_chunks,
    silent_runs,
    speech_mask,
)

SR = 16_000


def tone(seconds: float, amplitude: float = 0.3, freq: float = 220) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def pause(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def assert_tiles(chunks: list[Chunk], n_samples: int) -> None:
    """Chunks must cover [0, n_samples) exactly once, in order."""
    assert chunks[0].start_sample == 0
    assert chunks[-1].end_sample == n_samples
    for left, right in zip(chunks[:-1], chunks[1:], strict=True):
        assert left.end_sample == right.start_sample


# ---------------------------------------------------------------------------
# Frame energy and the speech mask
# ---------------------------------------------------------------------------


def test_energy_of_a_sine_matches_the_formula():
    # A sine of amplitude A has RMS A / sqrt(2), so dB = 20 log10(A / sqrt 2).
    # 500 Hz fits exactly 15 cycles in a 30 ms frame and 0.96 s is exactly 32
    # frames, so every frame sees whole cycles and no zero padding: the
    # formula then holds to rounding error.
    energy = frame_energy_db(tone(0.96, amplitude=0.5, freq=500), SR)
    expected = 20 * np.log10(0.5 / np.sqrt(2))  # about -9.03 dB
    assert energy.shape == (32,)
    assert np.allclose(energy, expected, atol=1e-3)


def test_halving_amplitude_costs_about_6_db():
    loud = frame_energy_db(tone(0.3, amplitude=0.4), SR).mean()
    quiet = frame_energy_db(tone(0.3, amplitude=0.2), SR).mean()
    assert loud - quiet == pytest.approx(20 * np.log10(2), abs=0.01)


def test_frame_count_rounds_up_and_silence_is_very_negative():
    # 1000 samples / 480 per 30 ms frame = 2.08, so 3 frames.
    energy = frame_energy_db(np.zeros(1000), SR)
    assert energy.shape == (3,)
    assert np.all(energy < -150)
    assert frame_energy_db(np.zeros(0), SR).shape == (0,)


def test_speech_mask_separates_tone_from_pause():
    signal = np.concatenate([tone(1.0), pause(1.0)])
    mask = speech_mask(frame_energy_db(signal, SR))
    half = len(mask) // 2
    assert mask[: half - 1].all()  # first second: speech
    assert not mask[half + 1 :].any()  # second second: silence


def test_faint_hiss_alone_never_counts_as_speech():
    # Noise with std 1e-3 is about -60 dB, below the -50 dB floor. Without the
    # floor, the relative threshold would call the louder half of it speech.
    rng = np.random.default_rng(0)
    hiss = rng.normal(0, 1e-3, size=SR * 2)
    assert not speech_mask(frame_energy_db(hiss, SR)).any()


def test_silent_runs_finds_every_run():
    mask = np.array([True, False, False, True, False])
    assert silent_runs(mask) == [(1, 3), (4, 5)]
    assert silent_runs(np.array([True, True])) == []
    assert silent_runs(np.array([False, False])) == [(0, 2)]


# ---------------------------------------------------------------------------
# Chunk planning
# ---------------------------------------------------------------------------


def test_short_audio_is_a_single_chunk():
    signal = tone(3.0)
    chunks = plan_chunks(signal, SR, max_chunk_s=10)
    assert len(chunks) == 1
    assert (chunks[0].start_sample, chunks[0].end_sample) == (0, len(signal))
    assert chunks[0].speech_ratio == 1.0


def test_cuts_land_in_the_middle_of_pauses():
    # Speech with pauses at 3.0-4.0 s and 6.5-7.0 s, 10 s in total.
    signal = np.concatenate([tone(3.0), pause(1.0), tone(2.5), pause(0.5), tone(3.0)])
    chunks = plan_chunks(signal, SR, max_chunk_s=5)
    assert_tiles(chunks, len(signal))
    cuts = [c.start for c in chunks[1:]]
    # Expected midpoints 3.5 s and 6.75 s, to within one 30 ms frame.
    assert cuts == pytest.approx([3.5, 6.75], abs=0.03)


def test_longest_pause_in_the_window_wins():
    # Window for the first cut is [5, 10] s. It holds a 0.4 s pause at 5.5 s
    # and a 1.0 s pause at 7.5 s; the longer one should be chosen.
    signal = np.concatenate([tone(5.5), pause(0.4), tone(1.6), pause(1.0), tone(6.0)])
    chunks = plan_chunks(signal, SR, max_chunk_s=10)
    assert chunks[1].start == pytest.approx(8.0, abs=0.03)


def test_pauses_shorter_than_min_silence_are_ignored():
    # Only a 0.1 s gap: too short to count, so the cut falls back to the
    # quietest frame, which is inside that gap anyway (it is pure zeros).
    signal = np.concatenate([tone(4.0), pause(0.1), tone(4.0)])
    chunks = plan_chunks(signal, SR, max_chunk_s=6, min_silence_s=0.3)
    assert 4.0 <= chunks[1].start <= 4.1


def test_no_pause_falls_back_to_the_quietest_frame():
    # A tone whose loudness swells and fades with a 4 s period: no silence at
    # all, but it is quietest at t = 2, 6, 10 ... seconds.
    t = np.arange(20 * SR) / SR
    envelope = 0.2 + 0.15 * np.cos(2 * np.pi * t / 4)
    signal = (envelope * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    chunks = plan_chunks(signal, SR, max_chunk_s=5)
    assert_tiles(chunks, len(signal))
    # First window is [2.5, 5] s; its quietest point is at the edge, 2.5 s is
    # closer to the trough at 2 s than 5 s is to the trough at 6 s.
    assert chunks[1].start == pytest.approx(2.5, abs=0.03)


@pytest.mark.parametrize("seed", range(5))
def test_chunks_always_tile_and_respect_the_limits(seed):
    rng = np.random.default_rng(seed)
    # Random "lecture": alternating tones and pauses of random lengths.
    parts = []
    for _ in range(30):
        parts.append(tone(rng.uniform(0.5, 6.0), amplitude=rng.uniform(0.05, 0.5)))
        parts.append(pause(rng.uniform(0.0, 1.5)))
    signal = np.concatenate(parts)
    max_s = 8.0
    chunks = plan_chunks(signal, SR, max_chunk_s=max_s)
    assert_tiles(chunks, len(signal))
    assert all(c.duration <= max_s + 1e-9 for c in chunks)
    # Every chunk but the last is at least half the maximum.
    assert all(c.duration >= max_s / 2 - 0.03 for c in chunks[:-1])


def test_silent_chunk_has_zero_speech_ratio():
    signal = np.concatenate([tone(4.0), pause(8.0), tone(4.0)])
    chunks = plan_chunks(signal, SR, max_chunk_s=5)
    assert any(c.speech_ratio == 0.0 for c in chunks)
    assert chunks[0].speech_ratio > 0.5


def test_chunk_times_in_seconds():
    chunk = Chunk(start_sample=16_000, end_sample=48_000, sample_rate=SR,
                  speech_ratio=1.0)  # fmt: skip
    assert (chunk.start, chunk.end, chunk.duration) == (1.0, 3.0, 2.0)


def test_bad_arguments_raise():
    with pytest.raises(ValueError, match="positive"):
        plan_chunks(tone(1.0), SR, max_chunk_s=0)
    with pytest.raises(ValueError, match="1-D"):
        plan_chunks(np.zeros((2, 100)), SR, max_chunk_s=5)
    with pytest.raises(ValueError, match="too short"):
        plan_chunks(tone(1.0), SR, max_chunk_s=0.03)
