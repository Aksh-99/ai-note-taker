"""Tests for notetaker.audio.

No audio files are committed to the repo. Instead each test asks ffmpeg to
synthesize a tiny clip (half a second of a pure tone) into a temporary folder,
which pytest deletes afterwards. A pure tone is ideal for testing because we
know exactly what the decoded signal should look like mathematically.
"""

from __future__ import annotations

import shutil
import subprocess
import wave

import numpy as np
import pytest

from notetaker import audio
from notetaker.audio import (
    AudioLoadError,
    FFmpegNotFoundError,
    duration_seconds,
    load_audio,
    peak_normalize,
    save_wav,
)

# Tests that need the real ffmpeg program are skipped (with a reason shown in
# the summary) on machines without it. CI installs ffmpeg, so they always run
# there.
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)


def make_clip(path, expression, sample_rate=44_100, seconds=0.5):
    """Synthesize a clip with ffmpeg's `aevalsrc` filter.

    `expression` is a formula in t (time in seconds), e.g. "sin(2*PI*440*t)".
    Two formulas separated by "|" give a stereo clip (left|right). We use
    44.1 kHz stereo on purpose: it is NOT Whisper's format, so the loader has
    to resample and downmix it.
    """
    source = f"aevalsrc={expression}:s={sample_rate}:d={seconds}"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", source, str(path)],
        check=True,
    )  # fmt: skip
    return path


def dominant_frequency(signal, sample_rate):
    """Frequency (Hz) with the largest magnitude in the signal's spectrum.

    `rfft` is the FFT for real-valued signals; `rfftfreq` gives the frequency
    in Hz of each FFT bin. The argmax bin is the strongest pure tone.
    """
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1 / sample_rate)
    return freqs[np.argmax(spectrum)]


@needs_ffmpeg
def test_load_resamples_to_16k_mono_float(tmp_path):
    clip = make_clip(tmp_path / "tone.wav", "0.5*sin(2*PI*440*t)|0.5*sin(2*PI*440*t)")
    waveform = load_audio(clip, sample_rate=16_000)

    assert waveform.dtype == np.float32
    assert waveform.ndim == 1  # mono: a single row of samples
    # 0.5 s at 16 kHz is 8000 samples. Resampling can add or drop a sample or
    # two at the edges, so allow a tiny tolerance.
    assert abs(len(waveform) - 8000) <= 2
    assert np.max(np.abs(waveform)) <= 1.0


@needs_ffmpeg
def test_tone_frequency_and_amplitude_survive_resampling(tmp_path):
    clip = make_clip(tmp_path / "tone.wav", "0.5*sin(2*PI*440*t)")
    waveform = load_audio(clip, sample_rate=16_000)

    # FFT resolution is sample_rate / N = 16000 / 8000 = 2 Hz per bin.
    assert dominant_frequency(waveform, 16_000) == pytest.approx(440, abs=2)
    # The peak of A*sin(...) is A = 0.5 (up to small 16-bit rounding error).
    assert np.max(np.abs(waveform)) == pytest.approx(0.5, abs=0.01)


@needs_ffmpeg
def test_stereo_is_averaged_to_mono(tmp_path):
    # Left channel is sin, right is -sin. Mono = (L + R) / 2 = 0 everywhere,
    # which proves the channels are averaged rather than one being dropped.
    clip = make_clip(
        tmp_path / "cancel.wav", "0.5*sin(2*PI*440*t)|-0.5*sin(2*PI*440*t)"
    )
    waveform = load_audio(clip, sample_rate=16_000)
    assert np.max(np.abs(waveform)) < 0.01


@needs_ffmpeg
def test_default_rate_comes_from_settings(tmp_path, monkeypatch):
    # monkeypatch sets an environment variable for this test only.
    monkeypatch.setenv("NOTETAKER_SAMPLE_RATE", "8000")
    audio.get_settings.cache_clear()  # force config to re-read the env
    try:
        clip = make_clip(tmp_path / "tone.wav", "0.5*sin(2*PI*440*t)")
        assert abs(len(load_audio(clip)) - 4000) <= 2  # 0.5 s at 8 kHz
    finally:
        audio.get_settings.cache_clear()  # don't leak 8 kHz into other tests


@needs_ffmpeg
def test_compressed_format_mp3(tmp_path):
    # mp3 is lossy, so only check the tone frequency, not exact sample values.
    clip = make_clip(tmp_path / "tone.mp3", "0.5*sin(2*PI*440*t)")
    waveform = load_audio(clip, sample_rate=16_000)
    assert dominant_frequency(waveform, 16_000) == pytest.approx(440, abs=2)


def test_missing_file_raises(tmp_path):
    with pytest.raises(AudioLoadError, match="not found"):
        load_audio(tmp_path / "nope.mp3")


@needs_ffmpeg
def test_non_audio_file_raises(tmp_path):
    bogus = tmp_path / "notes.mp3"
    bogus.write_text("this is text, not audio")
    with pytest.raises(AudioLoadError, match="could not decode"):
        load_audio(bogus)


def test_missing_ffmpeg_gives_helpful_error(tmp_path, monkeypatch):
    # Pretend ffmpeg is not installed by making `which` find nothing.
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    clip = tmp_path / "x.wav"
    clip.write_bytes(b"")
    with pytest.raises(FFmpegNotFoundError, match="install"):
        load_audio(clip)


def test_invalid_sample_rate_rejected(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        load_audio(tmp_path / "x.wav", sample_rate=0)


def test_peak_normalize_scales_loudest_sample():
    quiet = np.array([0.0, 0.1, -0.2, 0.05], dtype=np.float32)
    loud = peak_normalize(quiet, target_peak=0.9)
    assert np.max(np.abs(loud)) == pytest.approx(0.9)
    # Pure rescaling: every sample is multiplied by the same constant (4.5).
    np.testing.assert_allclose(loud, quiet * 4.5, rtol=1e-6)


def test_peak_normalize_leaves_silence_alone():
    silence = np.zeros(10, dtype=np.float32)
    np.testing.assert_array_equal(peak_normalize(silence), silence)


def test_peak_normalize_rejects_bad_target():
    with pytest.raises(ValueError):
        peak_normalize(np.ones(3), target_peak=1.5)


def test_duration_seconds():
    assert duration_seconds(np.zeros(32_000), sample_rate=16_000) == 2.0


def test_save_wav_writes_16k_mono_16bit(tmp_path):
    t = np.arange(16_000) / 16_000
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    out = save_wav(tone, tmp_path / "sub" / "out.wav", sample_rate=16_000)

    with wave.open(str(out), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnframes() == 16_000


@needs_ffmpeg
def test_save_then_load_round_trip(tmp_path):
    # Values beyond [-1, 1] must be clipped, not wrapped around.
    original = np.array([0.0, 0.25, -0.5, 2.0, -2.0] * 200, dtype=np.float32)
    path = save_wav(original, tmp_path / "rt.wav", sample_rate=16_000)
    restored = load_audio(path, sample_rate=16_000)
    np.testing.assert_allclose(restored, np.clip(original, -1, 1), atol=1e-3)
