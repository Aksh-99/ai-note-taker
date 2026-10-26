"""Split long recordings into chunks at natural pauses (voice activity detection).

The problem:
    A 90-minute lecture is 86.4 million samples at 16 kHz. Handing the whole
    thing to Whisper in one call works in principle, but memory and run time
    for one call grow with the length (the decoder keeps every segment, word
    timestamp, and its decoding context for the whole file), a crash at minute
    80 loses everything, and there is no way to show progress. So we cut the
    audio into pieces of at most a few minutes and transcribe them one by one.

Why not cut every N seconds?
    A fixed cut lands in the middle of a word about as often as not, and
    Whisper then mishears the half-words on both sides of the cut. Instead we
    look for a pause (a stretch of silence) near the point where we want to
    cut, and cut in the middle of it. Finding the pauses is called voice
    activity detection, or VAD.

The VAD used here (energy-based, pure NumPy, no model download):
    1. Slice the waveform x[n] into short frames (30 ms = 480 samples).
    2. For each frame compute its loudness in decibels relative to full scale:
           rms = sqrt(mean(x[n]^2))          (root mean square amplitude)
           dB  = 20 * log10(rms)             (0 dB = loudest possible signal)
       Halving the amplitude subtracts about 6 dB. Typical speech sits around
       -30 to -15 dB, a quiet room around -60 dB or lower.
    3. Call a frame "speech" if it is louder than a threshold. The threshold
       adapts to the recording: it is 35 dB below the loud end of this file
       (the 95th percentile of frame loudness), but never below an absolute
       floor of -50 dB. A quiet recording therefore gets a lower threshold
       than a loud one, while pure hiss never counts as speech.

    This is cruder than a neural VAD such as Silero, but lecture audio is one
    person talking with clear pauses, so energy works well, and it keeps the
    test suite and CI free of model downloads.

How a cut point is chosen (greedy, left to right):
    From the current chunk start, look at the window between max_chunk/2 and
    max_chunk seconds ahead. Among the silent runs inside that window that
    last at least `min_silence_s`, pick the LONGEST one (a long pause is the
    likeliest place for a new sentence or topic) and cut at its midpoint. If
    the speaker never pauses for that long, fall back to the quietest single
    frame in the window. Then repeat from the cut. Every chunk is therefore
    between max_chunk/2 and max_chunk long, except possibly the last one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Defaults, also referenced by tests and the CLI.
DEFAULT_FRAME_MS = 30.0
DEFAULT_MIN_SILENCE_S = 0.3
# How far below the loud end of the recording a frame must be to count as
# silence, and the absolute loudness a frame must exceed to ever count as
# speech. Both in dB relative to full scale (dBFS).
DEFAULT_RELATIVE_DB = 35.0
DEFAULT_FLOOR_DB = -50.0

# Added inside the logarithm so a frame of exact zeros gives a very negative
# number (-200 dB) instead of log10(0) = -infinity.
_EPS = 1e-10


@dataclass(frozen=True)
class Chunk:
    """One piece of the recording, as a half-open sample range [start, end).

    Storing sample indices (integers) rather than seconds (floats) means the
    chunks tile the waveform exactly: chunk k ends at the very sample where
    chunk k+1 starts, with no rounding gaps or overlaps.
    """

    start_sample: int
    end_sample: int
    sample_rate: int
    # Fraction of this chunk's frames that the VAD called speech, in [0, 1].
    # 0.0 means the chunk is silence only and can be skipped.
    speech_ratio: float

    @property
    def start(self) -> float:
        """Start time in seconds from the beginning of the recording."""
        return self.start_sample / self.sample_rate

    @property
    def end(self) -> float:
        return self.end_sample / self.sample_rate

    @property
    def duration(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate


def _frame_length(sample_rate: int, frame_ms: float) -> int:
    frame_len = int(round(sample_rate * frame_ms / 1000))
    if frame_len < 1:
        raise ValueError(f"frame_ms={frame_ms} is shorter than one sample")
    return frame_len


def frame_energy_db(
    waveform: np.ndarray, sample_rate: int, frame_ms: float = DEFAULT_FRAME_MS
) -> np.ndarray:
    """Loudness of each consecutive frame, in dB relative to full scale.

    Returns an array with one value per frame. The last frame is padded with
    zeros if the waveform length is not a multiple of the frame length.
    """
    frame_len = _frame_length(sample_rate, frame_ms)
    n_frames = -(-len(waveform) // frame_len)  # ceiling division
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    padded = np.zeros(n_frames * frame_len, dtype=np.float64)
    padded[: len(waveform)] = waveform
    # Reshape the 1-D signal into a (frames x samples-per-frame) matrix, so
    # one vectorised mean over axis 1 gives every frame's power at once.
    frames = padded.reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return 20.0 * np.log10(rms + _EPS)


def speech_mask(
    energy_db: np.ndarray,
    relative_db: float = DEFAULT_RELATIVE_DB,
    floor_db: float = DEFAULT_FLOOR_DB,
) -> np.ndarray:
    """Boolean array: True where a frame is loud enough to be speech."""
    if energy_db.size == 0:
        return np.zeros(0, dtype=bool)
    # The 95th percentile (not the max) as "loud end", so one cough or a
    # dropped microphone does not drag the threshold up for the whole file.
    reference = float(np.percentile(energy_db, 95))
    threshold = max(floor_db, reference - relative_db)
    return energy_db > threshold


def silent_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find maximal runs of False (silence) in a speech mask.

    Returns half-open frame ranges [a, b). Example:
        mask = [T, F, F, T, F]  ->  [(1, 3), (4, 5)]

    Trick: pad the silence indicator with 0 on both sides and take the
    difference between neighbours. A run of 1s then shows up as a +1 where it
    starts and a -1 just after it ends, so np.flatnonzero finds all run edges
    in one vectorised pass instead of a Python loop over every frame.
    """
    silent = np.concatenate(([0], (~mask).astype(np.int8), [0]))
    edges = np.diff(silent)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _choose_cut(
    lo: int,
    hi: int,
    runs: list[tuple[int, int]],
    energy_db: np.ndarray,
    min_silence_frames: int,
) -> int:
    """Pick the frame index to cut at, somewhere in [lo, hi)."""
    best_cut, best_len = None, 0
    for a, b in runs:
        # Only the part of the silent run that lies inside the window counts.
        a, b = max(a, lo), min(b, hi)
        length = b - a
        # `>=` so that among equally long pauses the later one wins, which
        # keeps chunks closer to the maximum length (fewer chunks overall).
        if length >= min_silence_frames and length >= best_len:
            best_cut, best_len = (a + b) // 2, length
    if best_cut is not None:
        return best_cut
    # No real pause: cut at the quietest frame of the window instead.
    return lo + int(np.argmin(energy_db[lo:hi]))


def plan_chunks(
    waveform: np.ndarray,
    sample_rate: int,
    max_chunk_s: float,
    *,
    min_silence_s: float = DEFAULT_MIN_SILENCE_S,
    frame_ms: float = DEFAULT_FRAME_MS,
    relative_db: float = DEFAULT_RELATIVE_DB,
    floor_db: float = DEFAULT_FLOOR_DB,
) -> list[Chunk]:
    """Split a waveform into consecutive chunks of at most `max_chunk_s`.

    The chunks cover every sample exactly once, in order. A recording that is
    already short enough comes back as a single chunk.

    Args:
        waveform: 1-D float signal.
        sample_rate: samples per second of `waveform`.
        max_chunk_s: longest allowed chunk, in seconds.
        min_silence_s: shortest pause that counts as a good place to cut.
        frame_ms: VAD frame length in milliseconds.
        relative_db, floor_db: VAD threshold settings (see `speech_mask`).
    """
    if waveform.ndim != 1:
        raise ValueError(f"Expected a 1-D waveform, got shape {waveform.shape}")
    if max_chunk_s <= 0:
        raise ValueError(f"max_chunk_s must be positive, got {max_chunk_s}")

    frame_len = _frame_length(sample_rate, frame_ms)
    energy = frame_energy_db(waveform, sample_rate, frame_ms)
    mask = speech_mask(energy, relative_db, floor_db)
    n_frames = len(energy)

    max_frames = int(max_chunk_s * sample_rate // frame_len)
    if max_frames < 2:
        raise ValueError(
            f"max_chunk_s={max_chunk_s} is too short for {frame_ms} ms VAD frames"
        )
    min_silence_frames = max(1, int(round(min_silence_s * 1000 / frame_ms)))
    runs = silent_runs(mask)

    # Cut points in frame units. Frame boundaries convert to samples by
    # multiplying by frame_len, so the chunks still tile the signal exactly.
    boundaries = [0]
    while n_frames - boundaries[-1] > max_frames:
        start = boundaries[-1]
        lo, hi = start + max_frames // 2, start + max_frames
        boundaries.append(_choose_cut(lo, hi, runs, energy, min_silence_frames))
    boundaries.append(n_frames)

    chunks = []
    for a, b in zip(boundaries[:-1], boundaries[1:], strict=True):
        chunks.append(
            Chunk(
                start_sample=a * frame_len,
                # The last frame may be zero padding past the real end.
                end_sample=min(b * frame_len, len(waveform)),
                sample_rate=sample_rate,
                speech_ratio=round(float(mask[a:b].mean()), 4) if b > a else 0.0,
            )
        )
    return chunks
