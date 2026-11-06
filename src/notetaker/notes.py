"""Turn a transcript into structured lecture notes with an LLM.

This is the first "generation" step of Phase 3. The pipeline is:

    transcript.json
        -> paragraphs        (export.group_paragraphs: pause-based blocks)
        -> chunks            (paragraphs packed up to a word budget)
        -> one LLM call per chunk, asking for JSON notes
        -> validated sections (heading, summary, key points, start time)
        -> notes.md + notes.json

Why chunks?
    A 75-minute lecture is about 10,000 spoken words, roughly 13,000 tokens.
    Small free models (Llama 3.1 8B on Ollama, Groq's free tier) either have
    a limited context window or get noticeably worse at following
    instructions when the input is that long. So we cut the transcript into
    pieces of about 1,200 words (a few minutes of speech) and summarize each
    one on its own (the "map" step). The "reduce" step in mapreduce.py then
    merges the per-chunk sections into the lecture's main topics and writes
    an overview; pass `reduce=True` to `generate_notes` to run it.

Why JSON instead of asking for Markdown directly?
    Free-form Markdown is hard to check. With JSON we can validate the shape
    (every section has a heading and at least one key point), reject junk,
    and attach data the model should not invent, such as the timestamp of
    each section. Validation uses pydantic models: you declare the expected
    fields and types once, and pydantic raises a clear error listing every
    field that is missing or wrong.

The "repair" loop
    Small models sometimes return broken JSON or the wrong shape. Instead of
    failing, we send the error message back to the model ("your reply was
    invalid because ...; return only the corrected JSON") and try again, up
    to `max_repairs` times. This is cheap and fixes most failures, because the
    model sees exactly what was wrong.

Timestamps
    Every paragraph in the prompt is prefixed with its start time, e.g.
    `[00:12:41] So the determinant ...`. The model is asked to copy the time
    of the paragraph where each section begins. We then parse that string,
    and if it is missing, malformed, or outside the chunk's time range, we
    fall back to the chunk's start. So a section's time is always a real
    moment in the lecture, never a hallucinated one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from notetaker.export import format_timestamp, group_paragraphs
from notetaker.llm import LLMClient, LLMResponse, Message
from notetaker.transcribe import Segment, Transcript

# About 1,200 words is 7-9 minutes of lecture speech and ~1,600 tokens: small
# enough for any free model's context, large enough to hold a full idea.
DEFAULT_CHUNK_WORDS = 1200

# Cap the reply length so a rambling model cannot run up time on a free tier.
DEFAULT_MAX_TOKENS = 1500

SYSTEM_PROMPT = """\
You are a careful teaching assistant who writes study notes from lecture \
transcripts. Use only what the lecturer actually says; never add facts, \
formulas or examples that are not in the transcript. Keep technical terms \
exactly as spoken. Answer with JSON only, no prose before or after it."""

USER_PROMPT_TEMPLATE = """\
Below is part {part} of {total} of a lecture transcript. Each paragraph starts \
with its timestamp [HH:MM:SS].

Write structured study notes for THIS PART ONLY as a JSON object of this shape:

{{
  "sections": [
    {{
      "heading": "short topic title (3 to 8 words)",
      "start": "HH:MM:SS timestamp of the paragraph where this topic begins",
      "summary": "one or two sentences on what this topic is about",
      "key_points": ["a specific fact, definition or step", "..."]
    }}
  ]
}}

Rules:
- 1 to 4 sections, in the order the topics are taught.
- 2 to 6 key points per section, each a complete, self-contained sentence.
- Copy "start" exactly from a timestamp that appears below.
- Skip small talk, logistics and repeated filler.

Transcript part {part} of {total}:
{transcript}"""

REPAIR_PROMPT = """\
Your previous reply could not be used: {error}
Return only the corrected JSON object with a "sections" list as described."""


# ---------------------------------------------------------------------------
# Chunking the transcript
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptChunk:
    """A run of consecutive paragraphs that goes to the LLM in one call."""

    index: int  # 0-based position in the lecture
    start: float  # seconds
    end: float  # seconds
    # Each paragraph as (start time in seconds, text).
    paragraphs: list[tuple[float, str]] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(text.split()) for _, text in self.paragraphs)

    def to_prompt_text(self) -> str:
        """Render as timestamped lines: `[00:12:41] paragraph text`."""
        return "\n".join(
            f"[{format_timestamp(start)}] {text}" for start, text in self.paragraphs
        )


def chunk_transcript(
    transcript: Transcript, max_words: int = DEFAULT_CHUNK_WORDS
) -> list[TranscriptChunk]:
    """Pack pause-based paragraphs into chunks of at most `max_words` words.

    This is greedy packing: keep adding the next paragraph while the chunk
    stays under budget, otherwise start a new chunk. We never split inside a
    paragraph (it would cut a sentence in half), so a single paragraph longer
    than the budget becomes a chunk of its own. Paragraphs are at most ~60 s,
    so in practice that only happens with a tiny `max_words`.
    """
    if max_words <= 0:
        raise ValueError(f"max_words must be positive, got {max_words}")

    paragraphs: list[list[Segment]] = group_paragraphs(transcript.segments)
    chunks: list[TranscriptChunk] = []
    current: list[list[Segment]] = []
    current_words = 0

    def flush() -> None:
        # A nested function ("closure") can read `current` and `chunks` from
        # the enclosing function, so the flushing logic is written only once.
        if not current:
            return
        chunks.append(
            TranscriptChunk(
                index=len(chunks),
                start=current[0][0].start,
                end=current[-1][-1].end,
                paragraphs=[
                    (para[0].start, " ".join(seg.text for seg in para))
                    for para in current
                ],
            )
        )

    for para in paragraphs:
        words = sum(len(seg.text.split()) for seg in para)
        if current and current_words + words > max_words:
            flush()
            current, current_words = [], 0
        current.append(para)
        current_words += words
    flush()
    return chunks


# ---------------------------------------------------------------------------
# Validating what the model returns
# ---------------------------------------------------------------------------

_TIMESTAMP = re.compile(r"^\s*(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\s*$")


def parse_timestamp(text: str) -> float | None:
    """Parse 'HH:MM:SS' or 'MM:SS' into seconds; None if it is not one."""
    match = _TIMESTAMP.match(text.strip("[] ")) if text else None
    if match is None:
        return None
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    if minutes >= 60 or seconds >= 60:
        return None
    return float(hours * 3600 + minutes * 60 + seconds)


class SectionSchema(BaseModel):
    """What one section must look like in the model's JSON reply.

    A pydantic `BaseModel` checks types on creation. `Field(min_length=1)`
    means "at least one character / item". Validators below clean the text
    (strip whitespace, drop empty bullet points) before the checks run.
    """

    heading: str = Field(min_length=1)
    start: str = ""
    summary: str = ""
    key_points: list[str] = Field(min_length=1)

    @field_validator("heading", "summary", "start", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("key_points", mode="before")
    @classmethod
    def _clean_points(cls, value: Any) -> Any:
        if isinstance(value, str):  # a model sometimes sends one string
            value = [value]
        if isinstance(value, list):
            return [p.strip() for p in value if isinstance(p, str) and p.strip()]
        return value


class ChunkNotesSchema(BaseModel):
    """The whole reply for one chunk: a non-empty list of sections."""

    sections: list[SectionSchema] = Field(min_length=1)


class NotesError(RuntimeError):
    """The model never produced usable notes for a chunk."""


def validate_reply(text: str) -> ChunkNotesSchema:
    """Parse and validate one reply. Raises ValueError with a readable reason.

    Accepts a bare list of sections too, since small models often drop the
    outer object. Both json.JSONDecodeError and pydantic's ValidationError are
    subclasses of ValueError, so callers only need to catch one type.
    """
    data = LLMResponse(text=text, model="", provider="").json()
    if isinstance(data, list):
        data = {"sections": data}
    try:
        return ChunkNotesSchema.model_validate(data)
    except ValidationError as error:
        # Shorten pydantic's multi-line message into one line for the repair
        # prompt, e.g. "sections.0.key_points: List should have at least 1 item".
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in error.errors()[:5]
        )
        raise ValueError(f"wrong JSON shape ({problems})") from None


# ---------------------------------------------------------------------------
# The notes themselves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoteSection:
    heading: str
    start: float  # seconds, always inside the chunk it came from
    summary: str
    key_points: list[str]
    chunk: int  # which chunk produced it (first chunk, for a merged topic)
    # After the reduce step: indices into `Notes.detail_sections` of the
    # per-chunk sections this topic was merged from. Empty for map output.
    sources: list[int] = field(default_factory=list)


@dataclass
class Notes:
    """Structured notes for one lecture plus how they were made."""

    title: str
    sections: list[NoteSection]
    model: str
    provider: str
    chunks: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    latency_s: float = 0.0
    # Filled by the reduce step (mapreduce.py); defaults mean "map only", so
    # notes.json files written before Step 3.3 still load.
    overview: str = ""
    takeaways: list[str] = field(default_factory=list)
    # The per-chunk sections before merging, kept for debugging and for the
    # faithfulness eval in Step 3.7. Empty when no reduce step ran.
    detail_sections: list[NoteSection] = field(default_factory=list)
    reduce_calls: int = 0
    reduce_levels: int = 0
    # Set when the reduce step failed and we fell back to the map notes.
    reduce_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Notes:
        sections = [NoteSection(**s) for s in data["sections"]]
        details = [NoteSection(**s) for s in data.get("detail_sections", [])]
        return cls(**{**data, "sections": sections, "detail_sections": details})

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        lines.append(
            f"_Generated by {self.provider} · {self.model} from {self.chunks} "
            f"transcript chunk(s). Timestamps link back to the recording._"
        )
        lines.append("")
        if self.overview:
            lines += [f"**Overview.** {self.overview}", ""]
        if not self.sections:
            lines += ["_No notes: the transcript was empty._", ""]
        for section in self.sections:
            lines.append(f"## {section.heading} `[{format_timestamp(section.start)}]`")
            lines.append("")
            if section.summary:
                lines += [section.summary, ""]
            lines += [f"- {point}" for point in section.key_points]
            lines.append("")
        if self.takeaways:
            lines += ["## Key takeaways", ""]
            lines += [f"- {point}" for point in self.takeaways]
            lines.append("")
        return "\n".join(lines)


def _resolve_start(raw: str, chunk: TranscriptChunk) -> float:
    """Model's timestamp if it lies inside the chunk, else the chunk start."""
    seconds = parse_timestamp(raw)
    # The chunk start is floored to whole seconds when shown in the prompt,
    # so allow 1 s of slack below it.
    if seconds is None or not (chunk.start - 1 <= seconds <= chunk.end):
        return chunk.start
    return max(seconds, chunk.start)


def summarize_chunk(
    client: LLMClient,
    chunk: TranscriptChunk,
    total: int,
    *,
    max_repairs: int = 1,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
) -> tuple[list[NoteSection], list[LLMResponse]]:
    """Ask the LLM for notes on one chunk, repairing bad JSON if needed.

    Returns the sections and every raw response (for token bookkeeping).
    """
    if max_repairs < 0:
        raise ValueError(f"max_repairs must be >= 0, got {max_repairs}")
    last_error: ValueError | None = None
    prompt = USER_PROMPT_TEMPLATE.format(
        part=chunk.index + 1, total=total, transcript=chunk.to_prompt_text()
    )
    messages = [Message("system", SYSTEM_PROMPT), Message("user", prompt)]
    responses: list[LLMResponse] = []
    for _ in range(max_repairs + 1):
        reply = client.chat(messages, json_mode=True, max_tokens=max_tokens)
        responses.append(reply)
        try:
            parsed = validate_reply(reply.text)
        except ValueError as error:
            # Keep the bad reply in the conversation so the model sees what
            # it wrote, then explain what was wrong with it.
            messages = [
                *messages,
                Message("assistant", reply.text),
                Message("user", REPAIR_PROMPT.format(error=error)),
            ]
            last_error = error
            continue
        sections = [
            NoteSection(
                heading=s.heading,
                start=_resolve_start(s.start, chunk),
                summary=s.summary,
                key_points=s.key_points,
                chunk=chunk.index,
            )
            for s in parsed.sections
        ]
        return sections, responses
    raise NotesError(
        f"chunk {chunk.index + 1}/{total}: no valid notes after "
        f"{max_repairs + 1} attempt(s): {last_error}"
    )


def generate_notes(
    transcript: Transcript,
    client: LLMClient,
    *,
    title: str = "Lecture notes",
    max_words: int = DEFAULT_CHUNK_WORDS,
    max_repairs: int = 1,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    on_chunk: Callable[[int, int, TranscriptChunk], None] | None = None,
    reduce: bool = False,
    reduce_words: int | None = None,
) -> Notes:
    """Full pipeline: chunk the transcript and summarize every chunk.

    `on_chunk(i, total, chunk)` is an optional progress callback; the CLI
    uses it to print a line before each (slow) LLM call.

    With `reduce=True` the per-chunk sections are then merged into topics
    with an overview (see mapreduce.py). `reduce_words` is the word budget
    per reduce call (None: the module default). If the reduce step fails,
    the map notes are returned unchanged with `reduce_error` set, because
    good per-chunk notes are still worth keeping.
    """
    chunks = chunk_transcript(transcript, max_words=max_words)
    sections: list[NoteSection] = []
    responses: list[LLMResponse] = []
    for chunk in chunks:
        if on_chunk is not None:
            on_chunk(chunk.index, len(chunks), chunk)
        chunk_sections, chunk_responses = summarize_chunk(
            client, chunk, len(chunks), max_repairs=max_repairs, max_tokens=max_tokens
        )
        sections.extend(chunk_sections)
        responses.extend(chunk_responses)

    # Sort by time: within a chunk the model may list topics out of order.
    # `sorted` is stable, so sections with equal times keep their order.
    sections = sorted(sections, key=lambda s: (s.chunk, s.start))
    notes = Notes(
        title=title,
        sections=sections,
        model=getattr(client, "model", "unknown"),
        provider=getattr(client, "provider", "unknown"),
        chunks=len(chunks),
        # `or 0` treats a missing count (None) as zero when adding up.
        prompt_tokens=sum(r.prompt_tokens or 0 for r in responses),
        completion_tokens=sum(r.completion_tokens or 0 for r in responses),
        llm_calls=len(responses),
        latency_s=sum(r.latency_s for r in responses),
    )
    if reduce:
        # Imported here, not at the top: mapreduce.py imports this module,
        # and a top-level import in both directions would be circular.
        from notetaker.mapreduce import DEFAULT_REDUCE_WORDS, reduce_notes

        notes = reduce_notes(
            notes,
            client,
            max_words=reduce_words or DEFAULT_REDUCE_WORDS,
            max_repairs=max_repairs,
            max_tokens=max_tokens,
        )
    return notes


def write_notes(notes: Notes, out_dir: str | Path) -> tuple[Path, Path]:
    """Write notes.json and notes.md into `out_dir`; return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "notes.json"
    md_path = out_dir / "notes.md"
    json_path.write_text(
        json.dumps(notes.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(notes.to_markdown(), encoding="utf-8")
    return json_path, md_path
