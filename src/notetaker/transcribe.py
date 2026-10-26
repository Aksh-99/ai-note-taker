"""Speech to text with faster-whisper, returning timestamped segments.

What faster-whisper is:
    OpenAI's Whisper is a free, open speech-recognition model. faster-whisper
    runs the same weights through CTranslate2, an inference engine written in
    C++ that is about 4x faster than the original PyTorch code and can use
    8-bit integers (int8) on a plain CPU. The weights download once from the
    Hugging Face Hub (free, no key) and are cached on disk afterwards.

What this module returns:
    A `Transcript`: a list of `Segment`s, each one a sentence-sized piece of
    speech with a start time, end time, and text. Each segment also carries
    word-level timestamps, so later steps can cite "lecture 3 at 12:41" or
    jump a video player to the exact word.

    Whisper also reports two confidence numbers per segment that are useful
    for evaluation later:
      avg_logprob     mean log-probability of the predicted tokens. Closer to
                      0 means more confident (exp(-0.2) ~ 0.82 per token).
      no_speech_prob  probability the segment is silence or noise, not speech.

Design note (dependency injection):
    `transcribe()` accepts an optional `model` argument. In normal use we leave
    it empty and a real faster-whisper model is loaded. Tests instead pass a
    tiny fake object with the same `.transcribe()` method, so the test suite
    (and CI) never downloads a model. This "pass the heavy dependency in"
    pattern is called dependency injection, and it is the standard way to make
    ML code testable.

Long recordings (Step 1.4):
    Audio longer than `NOTETAKER_CHUNK_MAX_SECONDS` (5 minutes by default) is
    split at pauses by `notetaker.chunking`, each chunk is transcribed on its
    own, and the results are stitched back together with every timestamp
    shifted by the chunk's start time. Callers see one ordinary Transcript.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from notetaker.audio import load_audio
from notetaker.chunking import Chunk, plan_chunks
from notetaker.config import get_settings

# Whisper only understands 16 kHz audio. faster-whisper assumes any NumPy array
# it is given is already at this rate, so we always decode at exactly 16000 Hz
# here, whatever NOTETAKER_SAMPLE_RATE says.
WHISPER_SAMPLE_RATE = 16_000

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
# `@dataclass` auto-generates __init__, __repr__ and __eq__ from the field
# list, so these classes are just typed records. `frozen=True` makes them
# read-only after creation, which prevents accidental edits to a transcript.


@dataclass(frozen=True)
class Word:
    """One word with its time span (seconds from the start of the audio)."""

    start: float
    end: float
    text: str
    # Whisper's probability for this word, in [0, 1].
    probability: float


@dataclass(frozen=True)
class Segment:
    """A sentence-sized chunk of speech."""

    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    # `field(default_factory=list)` gives every Segment its own new list.
    # A plain `= []` default would be shared by all instances (a classic bug).
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Transcript:
    """Everything one transcription run produced, plus how it was made."""

    segments: list[Segment]
    language: str
    # How sure Whisper was about the language, in [0, 1]. Equals 1.0 when the
    # language was given explicitly instead of detected.
    language_probability: float
    duration: float  # length of the audio in seconds
    model: str  # which Whisper size produced this, for reproducibility

    @property
    def text(self) -> str:
        """The full transcript as one string."""
        return " ".join(s.text for s in self.segments if s.text)

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dicts and lists, ready for `json.dump`.

        `asdict` recurses into nested dataclasses (segments, then words).
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        """Rebuild a Transcript from `to_dict()` output (e.g. a loaded JSON)."""
        segments = [
            Segment(
                **{k: v for k, v in seg.items() if k != "words"},
                words=[Word(**w) for w in seg.get("words", [])],
            )
            for seg in data["segments"]
        ]
        return cls(
            segments=segments,
            language=data["language"],
            language_probability=data["language_probability"],
            duration=data["duration"],
            model=data["model"],
        )

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # ensure_ascii=False keeps accented names readable instead of é.
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> Transcript:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


class SpeechModel(Protocol):
    """Anything with faster-whisper's `transcribe` method.

    A `Protocol` describes a shape ("has a method called transcribe") rather
    than a parent class. The real WhisperModel and the fake model in the tests
    both fit it without inheriting from anything. This is Python's version of
    "duck typing", made visible to type checkers.
    """

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Iterable[Any], Any]: ...


def cuda_available() -> bool:
    """True if CTranslate2 can see at least one NVIDIA GPU."""
    try:
        import ctranslate2
    except ImportError:
        return False
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except RuntimeError:
        # Raised on some machines where CUDA libraries are half-installed.
        return False


def resolve_device(device: str) -> str:
    """Turn the 'auto' setting into a concrete 'cuda' or 'cpu'."""
    if device == "auto":
        return "cuda" if cuda_available() else "cpu"
    return device


def resolve_compute_type(compute_type: str, device: str) -> str:
    """Pick numeric precision when the setting is 'auto'.

    GPUs are fastest with float16 (16-bit floats). CPUs have no fast float16
    math, but they do have fast 8-bit integer math, so int8 is the best CPU
    choice: weights are stored as int8 plus a scale factor per row, which cuts
    memory about 4x versus float32 with a very small accuracy cost.
    """
    if compute_type == "auto":
        return "float16" if device == "cuda" else "int8"
    return compute_type


# lru_cache means each (model, device, precision) combination is loaded only
# once per process. Loading `small` takes a few seconds and ~500 MB of RAM, so
# transcribing ten files in a row should not reload it ten times.
@lru_cache(maxsize=2)
def load_model(
    model_name: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> SpeechModel:
    """Load (downloading on first use) a faster-whisper model.

    Every argument defaults to the configured setting (NOTETAKER_WHISPER_*).
    """
    settings = get_settings()
    model_name = model_name or settings.whisper_model
    device = resolve_device(device or settings.whisper_device)
    compute_type = resolve_compute_type(
        compute_type or settings.whisper_compute_type, device
    )
    # Imported here rather than at the top of the file: faster-whisper pulls in
    # heavy native libraries, and this way `import notetaker.transcribe` stays
    # instant for code (and tests) that never loads a real model.
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def _convert_segment(raw: Any) -> Segment:
    """Copy one faster-whisper segment into our own Segment type.

    We convert instead of passing faster-whisper's objects around so the rest
    of the project depends only on our types. If we later swap the speech
    engine (e.g. a fine-tuned model in Phase 6), nothing downstream changes.
    """
    words = [
        Word(
            start=round(float(w.start), 3),
            end=round(float(w.end), 3),
            text=w.word.strip(),
            probability=round(float(w.probability), 4),
        )
        for w in (raw.words or [])
    ]
    return Segment(
        id=int(raw.id),
        start=round(float(raw.start), 3),
        end=round(float(raw.end), 3),
        # Whisper puts a leading space before each segment's text; drop it.
        text=raw.text.strip(),
        avg_logprob=round(float(raw.avg_logprob), 4),
        no_speech_prob=round(float(raw.no_speech_prob), 4),
        words=words,
    )


def shift_segment(segment: Segment, offset: float, new_id: int) -> Segment:
    """Move a segment (and its words) later in time by `offset` seconds.

    Whisper timestamps each chunk from 0, so a segment at 12.0 s inside a chunk
    that starts at 600 s really happened at 612 s of the lecture. Segments are
    frozen dataclasses, so instead of editing one in place we build a copy
    with `dataclasses.replace`, which copies every field except those named.
    """
    words = [
        replace(w, start=round(w.start + offset, 3), end=round(w.end + offset, 3))
        for w in segment.words
    ]
    return replace(
        segment,
        id=new_id,
        start=round(segment.start + offset, 3),
        end=round(segment.end + offset, 3),
        words=words,
    )


# Type of the optional progress callback: called as on_chunk(index, total,
# chunk) just before each chunk is transcribed (index counts from 1).
ChunkCallback = Callable[[int, int, Chunk], None]


def transcribe(
    audio: str | Path | np.ndarray,
    model: SpeechModel | None = None,
    *,
    language: str | None = None,
    beam_size: int = 5,
    word_timestamps: bool = True,
    model_name: str | None = None,
    max_chunk_s: float | None = None,
    on_chunk: ChunkCallback | None = None,
) -> Transcript:
    """Transcribe a file or waveform into a timestamped `Transcript`.

    Args:
        audio: a path to any media file (decoded with our ffmpeg loader into
            16 kHz mono), or a 1-D float waveform already sampled at 16 kHz.
        model: a loaded model. Leave as None to load the configured one; tests
            pass a fake here.
        language: spoken language code like "en". None uses the configured
            value; if that is also empty, Whisper detects the language (on the
            first chunk, and that answer is reused for the rest).
        beam_size: how many candidate transcriptions the decoder keeps at each
            step (beam search). 1 is greedy and fastest; 5 is Whisper's
            standard setting and is usually a little more accurate.
        word_timestamps: also return per-word start and end times.
        model_name: label stored in the result. Defaults to the configured
            model size.
        max_chunk_s: longest piece sent to the model at once, in seconds.
            None uses NOTETAKER_CHUNK_MAX_SECONDS; 0 disables chunking.
        on_chunk: optional progress callback, see `ChunkCallback`.
    """
    if beam_size < 1:
        raise ValueError(f"beam_size must be at least 1, got {beam_size}")
    settings = get_settings()
    language = language or settings.language
    model_name = model_name or settings.whisper_model
    if max_chunk_s is None:
        max_chunk_s = settings.chunk_max_seconds
    if max_chunk_s < 0:
        raise ValueError(f"max_chunk_s must be >= 0, got {max_chunk_s}")

    # Decode with our own loader so every input path goes through the same
    # tested ffmpeg pipeline (mono, 16 kHz, float32 in [-1, 1)).
    if isinstance(audio, np.ndarray):
        waveform = np.asarray(audio, dtype=np.float32)
    else:
        waveform = load_audio(audio, sample_rate=WHISPER_SAMPLE_RATE)
    if waveform.ndim != 1:
        raise ValueError(
            f"Expected a 1-D mono waveform, got an array of shape {waveform.shape}"
        )

    if max_chunk_s > 0:
        chunks = plan_chunks(waveform, WHISPER_SAMPLE_RATE, max_chunk_s)
    else:
        chunks = [Chunk(0, len(waveform), WHISPER_SAMPLE_RATE, speech_ratio=1.0)]

    if model is None:
        model = load_model(model_name)

    segments: list[Segment] = []
    detected_language: str | None = None
    language_probability: float | None = None
    duration = 0.0
    for index, chunk in enumerate(chunks, start=1):
        # Skip chunks that are pure silence: there is nothing to transcribe,
        # and Whisper is known to "hallucinate" text (e.g. "Thank you.") on
        # silent input. A recording that fits in one chunk is always sent,
        # so Whisper itself decides what a short clip contains.
        if len(chunks) > 1 and chunk.speech_ratio == 0.0:
            duration += chunk.duration
            continue
        if on_chunk is not None:
            on_chunk(index, len(chunks), chunk)
        # Slicing a NumPy array gives a "view" (no copy), so this costs no
        # extra memory however long the lecture is.
        piece = waveform[chunk.start_sample : chunk.end_sample]
        raw_segments, info = model.transcribe(
            piece,
            language=language,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
        )
        if detected_language is None:
            detected_language = info.language
            language_probability = float(info.language_probability)
            # Lock the language for the remaining chunks, so an auto-detected
            # English lecture is not suddenly decoded as Welsh in chunk 7.
            language = language or info.language
        # Important: faster-whisper returns a *generator*. No audio is actually
        # decoded until we loop over it, so this loop is where the real work
        # (and nearly all the run time) happens.
        for raw in raw_segments:
            segment = _convert_segment(raw)
            # Renumber 1, 2, 3, ... (faster-whisper's own convention) so ids
            # stay unique across chunks instead of restarting in each one.
            segments.append(shift_segment(segment, chunk.start, len(segments) + 1))
        duration += float(info.duration)

    return Transcript(
        segments=segments,
        language=detected_language or language or "",
        language_probability=round(
            language_probability if language_probability is not None else 0.0, 4
        ),
        duration=round(duration, 3),
        model=model_name,
    )
