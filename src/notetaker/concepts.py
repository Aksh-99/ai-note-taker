"""Extract a glossary of key concepts and their definitions from a lecture.

Pipeline:

    transcript.json
        -> chunks                   (same chunker as notes.py)
        -> one LLM call per chunk   ("list the technical terms defined here")
        -> validated replies        (pydantic schema + repair loop)
        -> grounding check          (the term must really occur in that chunk)
        -> merge across chunks      (same term from two chunks -> one entry)
        -> mention index            (every paragraph that uses the term)
        -> concepts.json + concepts.md

Schema validation happens twice, in both directions:

1. On the way IN, every model reply is checked against `ChunkConceptsSchema`.
   A bad reply goes back to the model with the exact error (the same repair
   loop as notes.py).
2. On the way OUT, the final file is itself a pydantic model (`ConceptList`).
   `write_concepts` serializes it and `load_concepts` re-validates it, so a
   hand-edited or corrupted concepts.json fails loudly instead of breaking a
   later step (flashcards in 3.5 and quizzes in 3.6 will read this file).
   `concepts_json_schema()` exports the same rules as a standard JSON Schema
   document (schemas/concepts.schema.json), which other tools and languages
   can use without Python.

What the model does vs. what code does
    The model only does the language part: spotting which phrases are
    technical terms and paraphrasing the lecturer's definition. Everything
    checkable is done in code. A term the model returns must appear in the
    chunk it read (otherwise it is a hallucination or a term the lecturer
    never said, and it is dropped). Timestamps are not asked from the model
    at all: we search the transcript for the term and record every paragraph
    that mentions it. So "first mentioned at 00:12:41" is a fact about the
    transcript, not a guess. Whether the definition itself is faithful to
    the lecture is harder to check and is the job of Step 3.7.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from notetaker.export import format_timestamp
from notetaker.llm import LLMClient, LLMResponse, Message
from notetaker.mapreduce import dedupe_points
from notetaker.notes import DEFAULT_CHUNK_WORDS, TranscriptChunk, chunk_transcript
from notetaker.transcribe import Transcript

# A glossary reply is shorter than a notes reply.
DEFAULT_MAX_TOKENS = 1000

# A "term" longer than this is a sentence, not a concept name.
MAX_TERM_WORDS = 8

SYSTEM_PROMPT = """\
You are a careful teaching assistant who builds glossaries from lecture \
transcripts. Only list terms the lecturer actually says, and define them the \
way the lecturer does; never add outside knowledge. Answer with JSON only."""

USER_PROMPT_TEMPLATE = """\
Below is part {part} of {total} of a lecture transcript.

List the key technical concepts that are introduced, defined or explained in \
THIS PART, as a JSON object of this shape:

{{
  "concepts": [
    {{
      "term": "the concept name exactly as the lecturer says it",
      "definition": "one or two sentences: what it is, in the lecturer's terms",
      "aliases": ["other names or abbreviations used for it, if any"]
    }}
  ]
}}

Rules:
- 0 to 8 concepts; an empty list is fine if nothing technical is explained.
- "term" is a short noun phrase (at most {max_words} words) that appears in \
the text below.
- Skip people's names, course logistics and everyday words.

Transcript part {part} of {total}:
{transcript}"""

REPAIR_PROMPT = """\
Your previous reply could not be used: {error}
Return only the corrected JSON object with a "concepts" list as described."""


# ---------------------------------------------------------------------------
# Schemas: what a model reply must look like
# ---------------------------------------------------------------------------


class ConceptSchema(BaseModel):
    """One concept as the model must return it."""

    term: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)

    # mode="before" runs the function on the raw value, before pydantic's
    # own type check, so we can clean up harmless sloppiness first.
    @field_validator("term", "definition", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip().strip("\"'*`") if isinstance(value, str) else value

    @field_validator("term")
    @classmethod
    def _short_term(cls, value: str) -> str:
        # A plain validator (mode="after") runs once the value is a valid
        # str; raising ValueError here becomes a normal validation error.
        if len(value.split()) > MAX_TERM_WORDS:
            raise ValueError(f"term must be at most {MAX_TERM_WORDS} words")
        return value

    @field_validator("aliases", mode="before")
    @classmethod
    def _clean_aliases(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            return [a.strip() for a in value if isinstance(a, str) and a.strip()]
        return value


class ChunkConceptsSchema(BaseModel):
    """The whole reply for one chunk. An empty list is allowed."""

    concepts: list[ConceptSchema]


class ConceptsError(RuntimeError):
    """The model never produced a usable concept list for a chunk."""


def _short_errors(error: ValidationError) -> str:
    """Pydantic's multi-line error as one line for the repair prompt."""
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()[:5]
    )


def validate_concepts_reply(text: str) -> ChunkConceptsSchema:
    """Parse and validate one reply; raise ValueError with a readable reason.

    Besides the requested shape, two common small-model variants are accepted:
    a bare list of concepts, and a flat {"term": "definition", ...} object.
    """
    data = LLMResponse(text=text, model="", provider="").json()
    if isinstance(data, list):
        data = {"concepts": data}
    elif (
        isinstance(data, dict)
        and "concepts" not in data
        and data
        and all(isinstance(v, str) for v in data.values())
    ):
        data = {"concepts": [{"term": k, "definition": v} for k, v in data.items()]}
    try:
        return ChunkConceptsSchema.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"wrong JSON shape ({_short_errors(error)})") from None


# ---------------------------------------------------------------------------
# Matching terms against the transcript
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Very light plural folding so 'eigenvalues' matches 'eigenvalue'.

    Real stemmers (Porter, Snowball) are more thorough but also merge words
    we want apart (e.g. 'general' and 'generate'). For matching a glossary
    term against a transcript, dropping a plural ending is enough.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def term_words(text: str) -> tuple[str, ...]:
    """Lowercase, split on anything that is not a letter or digit, stem.

    'Back-propagation' -> ('back', 'propagation'); 'ReLUs' -> ('relu',).
    A tuple (not a list) so it can be a dictionary key when merging.
    """
    return tuple(_stem(w) for w in _WORD.findall(text.lower()))


def contains_phrase(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    """True if `needle` occurs as consecutive words inside `haystack`.

    A sliding window: compare the needle with every slice of the same length.
    O(len(haystack) * len(needle)), which is tiny for one paragraph.
    """
    n = len(needle)
    if n == 0:
        return False
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


# ---------------------------------------------------------------------------
# Output schema: the concepts.json file
# ---------------------------------------------------------------------------


class Concept(BaseModel):
    """One glossary entry in concepts.json."""

    term: str = Field(min_length=1, description="Concept name as spoken")
    definition: str = Field(min_length=1, description="Definition from the lecture")
    aliases: list[str] = Field(default_factory=list, description="Other names used")
    first_mention: float = Field(ge=0, description="Seconds into the lecture")
    mentions: list[float] = Field(
        default_factory=list,
        description="Start time (s) of every paragraph that mentions the term",
    )
    chunks: list[int] = Field(
        default_factory=list, description="Chunks the concept was extracted from"
    )
    other_definitions: list[str] = Field(
        default_factory=list,
        description="Differently worded definitions from later chunks",
    )


class ConceptList(BaseModel):
    """The whole concepts.json file: entries plus how they were made."""

    title: str
    concepts: list[Concept]
    model: str
    provider: str
    chunks: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_s: float = Field(default=0.0, ge=0)
    # Terms the model returned that never occur in the chunk it read.
    dropped_ungrounded: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        lines.append(
            f"_{len(self.concepts)} concept(s) extracted by {self.provider} · "
            f"{self.model}. Times show where each term is first mentioned._"
        )
        lines.append("")
        if not self.concepts:
            lines += ["_No concepts found._", ""]
        for c in self.concepts:
            alias = f" (also: {', '.join(c.aliases)})" if c.aliases else ""
            lines.append(
                f"- **{c.term}**{alias} `[{format_timestamp(c.first_mention)}]`: "
                f"{c.definition}"
            )
        lines.append("")
        return "\n".join(lines)


def concepts_json_schema() -> dict[str, Any]:
    """The concepts.json format as a standard JSON Schema document."""
    return ConceptList.model_json_schema()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclass
class _Draft:
    """A concept while we are still merging it (mutable, unlike the output)."""

    term: str
    definition: str
    keys: set[tuple[str, ...]]  # term + alias word tuples, for matching
    aliases: list[str] = field(default_factory=list)
    chunks: list[int] = field(default_factory=list)
    other_definitions: list[str] = field(default_factory=list)


def extract_chunk_concepts(
    client: LLMClient,
    chunk: TranscriptChunk,
    total: int,
    *,
    max_repairs: int = 1,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
) -> tuple[list[ConceptSchema], list[str], list[LLMResponse]]:
    """Ask for one chunk's concepts. Returns (grounded, dropped terms, replies)."""
    if max_repairs < 0:
        raise ValueError(f"max_repairs must be >= 0, got {max_repairs}")
    prompt = USER_PROMPT_TEMPLATE.format(
        part=chunk.index + 1,
        total=total,
        max_words=MAX_TERM_WORDS,
        transcript=chunk.to_prompt_text(),
    )
    messages = [Message("system", SYSTEM_PROMPT), Message("user", prompt)]
    responses: list[LLMResponse] = []
    last_error: ValueError | None = None
    for _ in range(max_repairs + 1):
        reply = client.chat(messages, json_mode=True, max_tokens=max_tokens)
        responses.append(reply)
        try:
            parsed = validate_concepts_reply(reply.text)
        except ValueError as error:
            messages = [
                *messages,
                Message("assistant", reply.text),
                Message("user", REPAIR_PROMPT.format(error=error)),
            ]
            last_error = error
            continue
        chunk_words = term_words(" ".join(text for _, text in chunk.paragraphs))
        grounded, dropped = [], []
        for concept in parsed.concepts:
            # Accept the concept if the term or any alias is really in the text.
            names = [concept.term, *concept.aliases]
            if any(contains_phrase(chunk_words, term_words(n)) for n in names):
                grounded.append(concept)
            else:
                dropped.append(concept.term)
        return grounded, dropped, responses
    raise ConceptsError(
        f"chunk {chunk.index + 1}/{total}: no valid concepts after "
        f"{max_repairs + 1} attempt(s): {last_error}"
    )


def merge_concepts(
    per_chunk: list[tuple[int, list[ConceptSchema]]],
) -> list[_Draft]:
    """Combine the same concept found in several chunks into one entry.

    Two concepts are "the same" if any of their names (term or alias) match
    after `term_words` normalization, so "SVD" in one chunk merges with
    "singular value decomposition (aliases: SVD)" in another. The first
    chunk's wording wins (that is usually where the term is introduced);
    later definitions are kept in `other_definitions` unless they are near
    copies. Order of the result = order of first appearance.
    """
    drafts: list[_Draft] = []
    for chunk_index, concepts in per_chunk:
        for c in concepts:
            keys = {k for k in (term_words(n) for n in [c.term, *c.aliases]) if k}
            match = next((d for d in drafts if d.keys & keys), None)
            if match is None:
                drafts.append(
                    _Draft(
                        term=c.term,
                        definition=c.definition,
                        keys=keys,
                        aliases=[],
                        chunks=[chunk_index],
                    )
                )
                match = drafts[-1]
            else:
                match.keys |= keys
                if chunk_index not in match.chunks:
                    match.chunks.append(chunk_index)
                kept = dedupe_points(
                    [match.definition, *match.other_definitions, c.definition]
                )
                match.other_definitions = kept[1:]
            # Aliases: every name except the main term, without duplicates.
            main = term_words(match.term)
            for name in [c.term, *c.aliases]:
                key = term_words(name)
                if key != main and key not in {term_words(a) for a in match.aliases}:
                    match.aliases.append(name)
    return drafts


def find_mentions(
    keys: set[tuple[str, ...]], chunks: list[TranscriptChunk]
) -> list[float]:
    """Start time of every paragraph in the lecture that uses any of `keys`."""
    times = []
    for chunk in chunks:
        for start, text in chunk.paragraphs:
            words = term_words(text)
            if any(contains_phrase(words, k) for k in keys):
                times.append(start)
    return times


def extract_concepts(
    transcript: Transcript,
    client: LLMClient,
    *,
    title: str = "Key concepts",
    max_words: int = DEFAULT_CHUNK_WORDS,
    max_repairs: int = 1,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    on_chunk: Callable[[int, int, TranscriptChunk], None] | None = None,
) -> ConceptList:
    """Full pipeline: chunk, extract per chunk, ground, merge, index mentions."""
    chunks = chunk_transcript(transcript, max_words=max_words)
    per_chunk: list[tuple[int, list[ConceptSchema]]] = []
    dropped: list[str] = []
    responses: list[LLMResponse] = []
    for chunk in chunks:
        if on_chunk is not None:
            on_chunk(chunk.index, len(chunks), chunk)
        grounded, bad, replies = extract_chunk_concepts(
            client, chunk, len(chunks), max_repairs=max_repairs, max_tokens=max_tokens
        )
        per_chunk.append((chunk.index, grounded))
        dropped.extend(bad)
        responses.extend(replies)

    concepts = []
    for draft in merge_concepts(per_chunk):
        mentions = find_mentions(draft.keys, chunks)
        # Grounding guarantees at least one mention; the fallback only
        # protects against an edge case in normalization.
        first = mentions[0] if mentions else chunks[draft.chunks[0]].start
        concepts.append(
            Concept(
                term=draft.term,
                definition=draft.definition,
                aliases=draft.aliases,
                first_mention=first,
                mentions=mentions,
                chunks=draft.chunks,
                other_definitions=draft.other_definitions,
            )
        )
    concepts.sort(key=lambda c: c.first_mention)  # stable: ties keep order
    return ConceptList(
        title=title,
        concepts=concepts,
        model=getattr(client, "model", "unknown"),
        provider=getattr(client, "provider", "unknown"),
        chunks=len(chunks),
        llm_calls=len(responses),
        prompt_tokens=sum(r.prompt_tokens or 0 for r in responses),
        completion_tokens=sum(r.completion_tokens or 0 for r in responses),
        latency_s=sum(r.latency_s for r in responses),
        dropped_ungrounded=dropped,
    )


def write_concepts(concepts: ConceptList, out_dir: str | Path) -> tuple[Path, Path]:
    """Write concepts.json and concepts.md into `out_dir`; return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "concepts.json"
    md_path = out_dir / "concepts.md"
    json_path.write_text(concepts.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(concepts.to_markdown(), encoding="utf-8")
    return json_path, md_path


def load_concepts(path: str | Path) -> ConceptList:
    """Read concepts.json and validate it; raises ValueError if it is invalid."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        return ConceptList.model_validate_json(text)
    except ValidationError as error:
        reason = _short_errors(error)
        raise ValueError(f"{path} is not a valid concepts file: {reason}") from None
