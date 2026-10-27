"""Speed and memory benchmark for Whisper model sizes (tiny / base / small ...).

Why this exists:
    Bigger Whisper models are more accurate but slower and hungrier. Before
    choosing a default (and before the accuracy numbers of Phase 2), we need
    hard numbers for the other side of the tradeoff on the machine at hand.

What it measures, per model:
    load_seconds        time to load the weights (downloads are excluded only
                        if the model is already cached, so run once to warm up)
    transcribe_seconds  wall-clock time of each repeated transcription
    RTF                 real-time factor = transcribe time / audio duration.
                        RTF 0.25 means one hour of lecture takes 15 minutes.
                        Lower is better; below 1 is faster than real time.
                        "x real time" in the report is simply 1 / RTF.
    peak RAM            the most memory the process ever held (peak resident
                        set size, "RSS": the RAM pages actually in use).
    model RAM           peak RAM minus the baseline measured just before the
                        model was loaded, i.e. roughly what the model and its
                        decoding cost on top of Python + the audio itself.

Why each model runs in its own process (isolation):
    The operating system only reports the *peak* memory of a process, and a
    peak never goes down. If tiny, base and small ran one after another in one
    process, small's number would also include whatever tiny left behind, and
    tiny's number could never be measured after small. So by default each
    model is benchmarked in a fresh child process (Python's `multiprocessing`
    with the "spawn" start method, which starts a clean interpreter instead of
    copying the parent's memory). Each child's peak is then only its own.

Why the median of repeats:
    The first run often pays one-off costs (CPU caches, lazy initialisation),
    and background programs add noise. The median of several runs ignores a
    single slow outlier, which the mean would not.

Testing:
    `benchmark_model` takes the model loader and the clock as arguments
    (dependency injection, same pattern as `transcribe`), so tests use a fake
    model and a fake clock with exactly known timings and never download.
"""

from __future__ import annotations

import json
import multiprocessing
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from notetaker.config import get_settings
from notetaker.transcribe import (
    WHISPER_SAMPLE_RATE,
    SpeechModel,
    load_model,
    resolve_compute_type,
    resolve_device,
    transcribe,
)

# A loader takes (model_name, device, compute_type) and returns a model.
# `load_model` from transcribe.py has exactly this shape.
Loader = Callable[[str, str, str], SpeechModel]


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------


def peak_rss_mb() -> float | None:
    """Peak resident memory of this process so far, in megabytes.

    Uses the standard-library `resource` module, which exists on Linux and
    macOS but not on Windows; there we return None and the report shows "n/a"
    instead of crashing. Note the unit quirk: Linux reports kilobytes, macOS
    reports bytes.
    """
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Timings and memory for one model on one recording."""

    model: str
    device: str
    compute_type: str
    audio_seconds: float
    load_seconds: float
    # One entry per repeat, in seconds.
    transcribe_seconds: list[float] = field(default_factory=list)
    beam_size: int = 5
    peak_rss_mb: float | None = None
    baseline_rss_mb: float | None = None
    n_segments: int = 0
    n_words: int = 0

    @property
    def median_transcribe_seconds(self) -> float:
        return statistics.median(self.transcribe_seconds)

    @property
    def rtf(self) -> float:
        """Real-time factor: seconds of compute per second of audio."""
        if self.audio_seconds <= 0:
            return float("nan")
        return self.median_transcribe_seconds / self.audio_seconds

    @property
    def speedup(self) -> float:
        """How many times faster than real time (1 / RTF)."""
        rtf = self.rtf
        return 1.0 / rtf if rtf > 0 else float("inf")

    @property
    def model_rss_mb(self) -> float | None:
        """Memory added by loading and running the model (peak - baseline)."""
        if self.peak_rss_mb is None or self.baseline_rss_mb is None:
            return None
        return self.peak_rss_mb - self.baseline_rss_mb

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON, including the derived numbers for convenience."""
        data = asdict(self)
        data.update(
            median_transcribe_seconds=self.median_transcribe_seconds,
            rtf=self.rtf,
            speedup=self.speedup,
            model_rss_mb=self.model_rss_mb,
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Inverse of to_dict; ignores the derived keys."""
        names = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in names})


# ---------------------------------------------------------------------------
# Running a benchmark
# ---------------------------------------------------------------------------


def _default_loader(model_name: str, device: str, compute_type: str) -> SpeechModel:
    return load_model(model_name, device=device, compute_type=compute_type)


def benchmark_model(
    audio: np.ndarray,
    model_name: str,
    *,
    repeats: int = 1,
    beam_size: int = 5,
    language: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    loader: Loader | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    """Load one model and time `repeats` transcriptions of a 16 kHz waveform.

    `clock` defaults to `time.perf_counter`, Python's highest-resolution timer
    for measuring durations (unlike `time.time`, it never jumps if the system
    clock is adjusted).
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    # Looked up at call time (not as a default argument value), so tests can
    # swap `_default_loader` out with monkeypatch and the CLI picks it up.
    loader = loader or _default_loader
    settings = get_settings()
    device = resolve_device(device or settings.whisper_device)
    compute_type = resolve_compute_type(
        compute_type or settings.whisper_compute_type, device
    )
    audio_seconds = len(audio) / WHISPER_SAMPLE_RATE

    # Baseline taken *before* the model exists: Python, numpy and the audio.
    baseline = peak_rss_mb()

    t0 = clock()
    model = loader(model_name, device, compute_type)
    load_seconds = clock() - t0

    times: list[float] = []
    transcript = None
    for _ in range(repeats):
        t0 = clock()
        transcript = transcribe(
            audio,
            model=model,
            language=language,
            beam_size=beam_size,
            model_name=model_name,
        )
        times.append(clock() - t0)

    assert transcript is not None  # repeats >= 1 guarantees one run
    return BenchmarkResult(
        model=model_name,
        device=device,
        compute_type=compute_type,
        audio_seconds=audio_seconds,
        load_seconds=load_seconds,
        transcribe_seconds=times,
        beam_size=beam_size,
        peak_rss_mb=peak_rss_mb(),
        baseline_rss_mb=baseline,
        n_segments=len(transcript.segments),
        n_words=len(transcript.text.split()),
    )


def load_benchmark_audio(path: str | Path, max_seconds: float | None = None):
    """Decode a file to 16 kHz mono, optionally keeping only the first part.

    `max_seconds` makes quick runs possible on a long lecture: benchmarking
    the first 10 minutes gives a good RTF estimate in a fraction of the time.
    """
    from notetaker.audio import load_audio

    audio = load_audio(path, sample_rate=WHISPER_SAMPLE_RATE)
    if max_seconds is not None and max_seconds > 0:
        audio = audio[: int(max_seconds * WHISPER_SAMPLE_RATE)]
    return audio


def _isolated_worker(
    path: str, max_seconds: float | None, model_name: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Runs inside the child process. Returns a plain dict, because results
    cross the process boundary by pickling and plain types are safest."""
    audio = load_benchmark_audio(path, max_seconds)
    return benchmark_model(audio, model_name, **kwargs).to_dict()


def run_isolated(
    path: str | Path,
    model_name: str,
    *,
    max_seconds: float | None = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """Benchmark one model in a brand-new process (see module docstring).

    `ProcessPoolExecutor(max_workers=1)` is just a convenient way to run one
    function in a child process and get its return value (or its exception)
    back. Extra keyword arguments go to `benchmark_model`; a custom `loader`
    must be a module-level function so it can be pickled to the child.
    """
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        future = pool.submit(
            _isolated_worker, str(path), max_seconds, model_name, kwargs
        )
        return BenchmarkResult.from_dict(future.result())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_mb(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def to_markdown(results: Sequence[BenchmarkResult], audio_name: str) -> str:
    """Render results as a Markdown report with one table row per model."""
    if not results:
        return f"# Whisper benchmark: {audio_name}\n\nNo results.\n"
    first = results[0]
    lines = [
        f"# Whisper benchmark: {audio_name}",
        "",
        f"- Audio length: {first.audio_seconds:.1f} s",
        f"- Device: {first.device}, beam size {first.beam_size}, "
        f"{len(first.transcribe_seconds)} timed run(s) per model (median shown)",
        f"- Python {sys.version.split()[0]} on {sys.platform}",
        "",
        "RTF = transcription time / audio length (lower is better). "
        "Model RAM = peak RAM minus the baseline before loading the model.",
        "",
        "| Model | Compute | Load (s) | Transcribe (s) | RTF | x real time "
        "| Peak RAM (MB) | Model RAM (MB) | Words |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.model} | {r.compute_type} | {r.load_seconds:.2f} "
            f"| {r.median_transcribe_seconds:.2f} | {r.rtf:.3f} "
            f"| {r.speedup:.1f} | {_fmt_mb(r.peak_rss_mb)} "
            f"| {_fmt_mb(r.model_rss_mb)} | {r.n_words} |"
        )
    return "\n".join(lines) + "\n"


def write_report(
    results: Sequence[BenchmarkResult], out_dir: str | Path, audio_name: str
) -> tuple[Path, Path]:
    """Write `benchmark.json` (all numbers) and `benchmark.md` (the table)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "benchmark.json"
    md_path = out / "benchmark.md"
    payload = {"audio": audio_name, "results": [r.to_dict() for r in results]}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(results, audio_name), encoding="utf-8")
    return json_path, md_path
