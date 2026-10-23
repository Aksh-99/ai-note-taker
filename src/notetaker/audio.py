"""Load any audio or video file as a clean 16 kHz mono waveform for Whisper.

The problem:
    Lecture recordings arrive as mp3, m4a, wav, even mp4 video, at sample rates
    like 44.1 kHz or 48 kHz, often in stereo. Whisper was trained on exactly one
    format: 16 kHz, mono, floating-point samples between -1 and 1. Feeding it
    anything else gives worse transcripts or errors.

The approach:
    We let ffmpeg (a free, battle-tested command-line tool) do all decoding,
    mixing to mono, and resampling, then read its output straight into a NumPy
    array. Concretely, ffmpeg writes raw 16-bit integers to its standard output
    (a "pipe"), we read those bytes in Python and rescale them to floats. No
    temporary files are written, so this is fast and leaves no clutter.

    The resulting array is just a 1-D signal x[n], sampled every 1/16000 s.
    `len(audio) / sample_rate` is the duration in seconds.

ffmpeg must be installed separately (it is not a Python package):
    Ubuntu/Debian: sudo apt install ffmpeg      macOS: brew install ffmpeg
    Windows: winget install ffmpeg
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from notetaker.config import get_settings

# 16-bit audio stores each sample as an integer in [-32768, 32767]. Dividing by
# 32768 maps that range onto [-1, 1), the float convention Whisper expects.
INT16_SCALE = 32768.0


class AudioError(RuntimeError):
    """Base class for audio problems, so callers can catch them all at once."""


class FFmpegNotFoundError(AudioError):
    """ffmpeg is not installed or not on the PATH."""


class AudioLoadError(AudioError):
    """ffmpeg could not read the file (missing, corrupt, or not audio)."""


def find_ffmpeg() -> str:
    """Return the path to the ffmpeg program, or raise a helpful error.

    `shutil.which` searches the folders on your PATH, the same way the shell
    finds a command when you type its name.
    """
    path = shutil.which("ffmpeg")
    if path is None:
        raise FFmpegNotFoundError(
            "ffmpeg was not found. Install it (e.g. `sudo apt install ffmpeg` "
            "or `brew install ffmpeg`) and make sure it is on your PATH."
        )
    return path


def _resolve_sample_rate(sample_rate: int | None) -> int:
    """Use the explicit rate if given, otherwise the configured default."""
    if sample_rate is None:
        sample_rate = get_settings().sample_rate
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    return sample_rate


def load_audio(path: str | Path, sample_rate: int | None = None) -> np.ndarray:
    """Decode a media file into a mono float32 waveform at `sample_rate` Hz.

    Args:
        path: any file ffmpeg understands (mp3, wav, m4a, flac, mp4, ...).
        sample_rate: target rate in Hz. Defaults to the configured value
            (16000 unless overridden via NOTETAKER_SAMPLE_RATE).

    Returns:
        1-D NumPy array of float32 samples in [-1, 1).

    Raises:
        AudioLoadError: the file does not exist or could not be decoded.
        FFmpegNotFoundError: ffmpeg is not installed.
    """
    path = Path(path)
    sample_rate = _resolve_sample_rate(sample_rate)
    # Check existence ourselves: ffmpeg's own error message for this is noisy.
    if not path.is_file():
        raise AudioLoadError(f"Audio file not found: {path}")

    # The ffmpeg command, one option per line so it is easy to read:
    command = [
        find_ffmpeg(),
        "-nostdin",  # never wait for keyboard input (it would hang forever)
        "-hide_banner",  # skip the version banner in error output
        "-loglevel", "error",  # only print real errors
        "-i", str(path),  # input file
        "-vn",  # ignore any video stream (lecture recordings are often mp4)
        "-ac", "1",  # 1 audio channel: stereo is averaged down to mono
        "-ar", str(sample_rate),  # resample to the target rate
        "-f", "s16le",  # output format: raw signed 16-bit little-endian ints
        "-",  # "-" means write to standard output instead of a file
    ]  # fmt: skip
    # `capture_output=True` collects stdout (the audio bytes) and stderr (error
    # text) in memory. `check=False` because we inspect the exit code ourselves
    # to raise a friendlier error.
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        raise AudioLoadError(f"ffmpeg could not decode {path}: {message}")
    if not result.stdout:
        raise AudioLoadError(f"No audio stream found in {path}")

    # Reinterpret the raw bytes as int16 numbers (2 bytes each), then rescale.
    # `np.frombuffer` makes no copy; `astype` creates the float array we return.
    samples = np.frombuffer(result.stdout, dtype=np.int16)
    return samples.astype(np.float32) / INT16_SCALE


def peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Scale a waveform so its loudest sample has magnitude `target_peak`.

    Quiet classroom recordings can sit near zero amplitude. Multiplying by the
    constant c = target_peak / max|x[n]| makes the volume consistent without
    changing the shape of the signal (it is a pure rescaling, so no distortion).
    Silent input (all zeros) is returned unchanged to avoid dividing by zero.
    """
    if not 0.0 < target_peak <= 1.0:
        raise ValueError(f"target_peak must be in (0, 1], got {target_peak}")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak == 0.0:
        return audio.astype(np.float32, copy=True)
    return (audio * (target_peak / peak)).astype(np.float32)


def duration_seconds(audio: np.ndarray, sample_rate: int | None = None) -> float:
    """Length of a waveform in seconds: number of samples / samples per second."""
    return len(audio) / _resolve_sample_rate(sample_rate)


def save_wav(
    audio: np.ndarray, path: str | Path, sample_rate: int | None = None
) -> Path:
    """Write a mono float waveform to a 16-bit WAV file.

    Useful for caching a normalized copy of a long lecture, or for saving
    chunks later (Step 1.4). Uses Python's built-in `wave` module, no ffmpeg.
    Values outside [-1, 1] are clipped so they cannot wrap around when
    converted to integers (which would sound like loud clicks).
    """
    path = Path(path)
    sample_rate = _resolve_sample_rate(sample_rate)
    clipped = np.clip(audio, -1.0, 1.0)
    # Map [-1, 1] back to int16. 32767 (not 32768) so that +1.0 still fits.
    as_int16 = np.round(clipped * (INT16_SCALE - 1)).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    # `with` closes the file even if an error happens mid-write.
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 2 bytes = 16 bits per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(as_int16.tobytes())
    return path
