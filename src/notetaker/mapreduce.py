"""The "reduce" half of map-reduce summarization for long lectures.

Step 3.2 (notes.py) is the "map" step: every ~8-minute transcript chunk is
summarized on its own into 1-4 sections. For a 90-minute lecture that gives
30-40 sections, and neighbouring chunks often repeat a topic ("Eigenvalues",
then "Eigenvalues continued"). This module merges them into the lecture's
main topics and adds a short overview and key takeaways.

How one reduce call works
    The model sees the sections as a numbered list:

        [3] (00:16:02) Computing eigenvalues: summary... ; point; point

    and answers with JSON that GROUPS section numbers into topics:

        {"topics": [{"heading": "...", "summary": "...", "sections": [3, 4]}]}

    The model only writes the new heading and summary. Everything that must
    be factual is copied from the map sections in code: a topic's timestamp
    is the earliest start of its sections, and its key points are the key
    points of its sections with near-duplicates removed. So the reduce step
    cannot invent a timestamp or a key point, and no detail is lost.

Validation and repair
    The ids are checked in code: an id that does not exist, or one used in
    two topics, is an error, and the model is asked to fix its reply (the
    same repair loop as notes.py). An id the model forgot is not an error:
    that section is attached to the topic of the section just before it in
    time (or just after, if it is the first). This guarantees every map
    section ends up in exactly one topic.

Hierarchical reduce (the "recursive" part)
    If the numbered list is longer than the word budget (`max_words`), it
    would not fit well into a small model's prompt. Then we cut the list into
    consecutive batches that fit, reduce each batch into topics, and repeat
    on the (shorter) list of topics. Each level shrinks the list, so after
    one or two levels it fits and the final call also writes the overview.
    This is the same idea as a merge tree: n items, reduced in groups, gives
    about log(n) levels.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, replace
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from notetaker.export import format_timestamp
from notetaker.llm import LLMClient, LLMError, LLMResponse, Message
from notetaker.notes import Notes, NoteSection, NotesError

# Word budget for the numbered section list in one reduce prompt. 2,500
# words is ~3,300 tokens, comfortable for an 8B model's context.
DEFAULT_REDUCE_WORDS = 2500

# Only the first few key points of each section go into the prompt: they are
# enough to tell what a section is about, and they keep the prompt short.
# (All key points are still kept in the output, merged in code.)
PROMPT_POINTS_PER_SECTION = 3

# Stop after this many intermediate levels, whatever happens.
MAX_LEVELS = 3

# Two key points whose word-level similarity is at or above this count as
# the same point said twice (see `dedupe_points`). Deliberately strict:
# wrongly dropping a point loses information, keeping a near-copy is only
# a little untidy.
DUPLICATE_RATIO = 0.9

REDUCE_SYSTEM_PROMPT = """\
You are a careful teaching assistant who organizes study notes. Use only \
the information in the notes you are given; never add facts. Answer with \
JSON only, no prose before or after it."""

REDUCE_PROMPT_TEMPLATE = """\
Below are notes on consecutive parts of {scope}, as numbered sections in \
time order. Each line is: [id] (start time) heading: summary; key points.

Group the sections into the main topics of the lecture. Return a JSON \
object of this shape:

{{
{overview_fields}  "topics": [
    {{
      "heading": "short topic title (3 to 8 words)",
      "summary": "one to three sentences on what this topic covers",
      "sections": [1, 2]
    }}
  ]
}}

Rules:
- Put every section id in exactly one topic.
- A topic joins sections about the same idea; these are usually neighbours.
- Between {low} and {high} topics, listed in the order they are taught.
{overview_rules}
Sections:
{sections}"""

OVERVIEW_FIELDS = """\
  "overview": "two to four sentences on what the whole lecture covers",
  "takeaways": ["the most important idea, as one sentence", "..."],
"""

OVERVIEW_RULES = "- 3 to 6 takeaways, each a complete sentence.\n"

REDUCE_REPAIR_PROMPT = """\
Your previous reply could not be used: {error}
Return only the corrected JSON object with a "topics" list as described."""


# ---------------------------------------------------------------------------
# Schema for the model's reply
# ---------------------------------------------------------------------------

_ID = re.compile(r"^\s*\[?\s*[Ss]?\s*(\d+)\s*\]?\s*$")


class TopicSchema(BaseModel):
    """One topic in the reply: a new heading/summary plus section ids."""

    heading: str = Field(min_length=1)
    summary: str = ""
    sections: list[int] = Field(min_length=1)

    @field_validator("heading", "summary", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("sections", mode="before")
    @classmethod
    def _ids(cls, value: Any) -> Any:
        # Models write ids as 3, "3", "S3" or "[3]"; accept all of them.
        # Anything else is left as is, so pydantic reports it as an error.
        if isinstance(value, (int, str)):
            value = [value]
        if not isinstance(value, list):
            return value
        out: list[Any] = []
        for item in value:
            match = _ID.match(item) if isinstance(item, str) else None
            out.append(int(match.group(1)) if match else item)
        return out


class ReduceSchema(BaseModel):
    """The whole reduce reply. Overview fields are optional (final level)."""

    overview: str = ""
    takeaways: list[str] = Field(default_factory=list)
    topics: list[TopicSchema] = Field(min_length=1)

    @field_validator("overview", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("takeaways", mode="before")
    @classmethod
    def _clean(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            return [p.strip() for p in value if isinstance(p, str) and p.strip()]
        return value


def validate_reduce_reply(text: str, n_items: int) -> ReduceSchema:
    """Parse one reply and check its ids. Raises ValueError with a reason.

    Ids in the prompt are 1-based (people and models count from 1); the
    caller converts them to 0-based list positions.
    """
    data = LLMResponse(text=text, model="", provider="").json()
    if isinstance(data, list):  # a bare list of topics
        data = {"topics": data}
    try:
        parsed = ReduceSchema.model_validate(data)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in error.errors()[:5]
        )
        raise ValueError(f"wrong JSON shape ({problems})") from None

    seen: set[int] = set()
    for topic in parsed.topics:
        for item_id in topic.sections:
            if not 1 <= item_id <= n_items:
                raise ValueError(
                    f"section id {item_id} does not exist (ids are 1 to {n_items})"
                )
            if item_id in seen:
                raise ValueError(f"section id {item_id} is in more than one topic")
            seen.add(item_id)
    return parsed


# ---------------------------------------------------------------------------
# Pure helpers (no LLM): rendering, packing, merging
# ---------------------------------------------------------------------------


def render_item(item_id: int, section: NoteSection) -> str:
    """One prompt line: `[3] (00:16:02) Heading: summary; point; point`."""
    parts = [section.summary] if section.summary else []
    parts += section.key_points[:PROMPT_POINTS_PER_SECTION]
    body = "; ".join(parts)
    return f"[{item_id}] ({format_timestamp(section.start)}) {section.heading}: {body}"


def render_words(sections: list[NoteSection]) -> int:
    return sum(len(render_item(i + 1, s).split()) for i, s in enumerate(sections))


def pack_batches(
    sections: list[NoteSection], max_words: int
) -> list[list[NoteSection]]:
    """Cut the list into consecutive batches of at most `max_words` words.

    Greedy packing, as in `notes.chunk_transcript`. Every batch gets at least
    two sections (a batch of one has nothing to merge), so a budget that is
    too small still makes progress.
    """
    batches: list[list[NoteSection]] = []
    current: list[NoteSection] = []
    words = 0
    for section in sections:
        size = len(render_item(1, section).split())
        if len(current) >= 2 and words + size > max_words:
            batches.append(current)
            current, words = [], 0
        current.append(section)
        words += size
    if current:
        # Fold a lonely last section into the previous batch.
        if len(current) == 1 and batches:
            batches[-1].extend(current)
        else:
            batches.append(current)
    return batches


def _words(text: str) -> list[str]:
    """Lowercase words without punctuation: 'Det(A) = 0!' -> [det, a, 0]."""
    return re.findall(r"[a-z0-9]+", text.lower())


def dedupe_points(points: list[str], ratio: float = DUPLICATE_RATIO) -> list[str]:
    """Drop key points that repeat an earlier one almost word for word.

    Points are compared as lists of lowercase words, using difflib's
    similarity ratio: 2 * (matching words) / (total words in both), so 1.0
    means identical and 0.0 means nothing in common. Comparing words, not
    characters, matters: "the determinant is zero" and "the determinant is
    nonzero" share most characters but say opposite things. With 5 of 6
    words matching they score 0.83, below the 0.9 threshold, so both stay.
    This is O(n^2) comparisons, fine for the ~10-30 points of one topic.
    """
    kept: list[str] = []
    kept_words: list[list[str]] = []
    for point in points:
        words = _words(point)
        if not words:
            continue
        if any(
            difflib.SequenceMatcher(None, words, other).ratio() >= ratio
            for other in kept_words
        ):
            continue
        kept.append(point)
        kept_words.append(words)
    return kept


def assign_ids(groups: list[list[int]], n_items: int) -> list[list[int]]:
    """Give every 0-based id 0..n_items-1 exactly one group.

    `groups` come from a validated reply, so they have no duplicates or
    out-of-range ids, but some ids may be missing. A missing id joins the
    group of the nearest id before it (in time order), or after it if there
    is none before. Groups are returned sorted, each with sorted ids.
    """
    owner: dict[int, int] = {}
    for g, ids in enumerate(groups):
        for i in ids:
            owner[i] = g
    if not owner:
        raise ValueError("no section ids assigned")
    for i in range(n_items):
        if i in owner:
            continue
        before = [j for j in owner if j < i]
        after = [j for j in owner if j > i]
        neighbour = max(before) if before else min(after)
        owner[i] = owner[neighbour]
    result: list[list[int]] = [[] for _ in groups]
    for i in range(n_items):
        result[owner[i]].append(i)
    return [ids for ids in result if ids]


def merge_group(heading: str, summary: str, members: list[NoteSection]) -> NoteSection:
    """Build one topic from its member sections (facts copied, not generated)."""
    members = sorted(members, key=lambda s: s.start)
    points = dedupe_points([p for s in members for p in s.key_points])
    sources = sorted({i for s in members for i in s.sources})
    return NoteSection(
        heading=heading,
        start=members[0].start,
        summary=summary,
        key_points=points,
        chunk=min(s.chunk for s in members),
        sources=sources,
    )


def topic_range(n_items: int, final: bool) -> tuple[int, int]:
    """How many topics to ask for: a hint, not checked in code.

    Final level: roughly one topic per 3 sections, between 2 and 8.
    Intermediate level: roughly one per 2 sections, so each level halves.
    """
    if final:
        high = max(1, min(8, math.ceil(n_items / 2)))
        low = max(1, min(high, n_items // 4, 3))
    else:
        high = max(1, math.ceil(n_items / 2))
        low = max(1, n_items // 4)
    return low, high


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


@dataclass
class ReduceResult:
    topics: list[NoteSection]
    overview: str
    takeaways: list[str]
    responses: list[LLMResponse]


def reduce_once(
    client: LLMClient,
    sections: list[NoteSection],
    *,
    final: bool,
    max_repairs: int = 1,
    max_tokens: int | None = None,
) -> ReduceResult:
    """One reduce call: group `sections` into topics (repairing bad JSON)."""
    if max_repairs < 0:
        raise ValueError(f"max_repairs must be >= 0, got {max_repairs}")
    low, high = topic_range(len(sections), final)
    prompt = REDUCE_PROMPT_TEMPLATE.format(
        scope="one lecture" if final else "a section of one lecture",
        overview_fields=OVERVIEW_FIELDS if final else "",
        overview_rules=OVERVIEW_RULES if final else "",
        low=low,
        high=high,
        sections="\n".join(render_item(i + 1, s) for i, s in enumerate(sections)),
    )
    messages = [Message("system", REDUCE_SYSTEM_PROMPT), Message("user", prompt)]
    responses: list[LLMResponse] = []
    last_error: ValueError | None = None
    for _ in range(max_repairs + 1):
        reply = client.chat(messages, json_mode=True, max_tokens=max_tokens)
        responses.append(reply)
        try:
            parsed = validate_reduce_reply(reply.text, len(sections))
        except ValueError as error:
            messages = [
                *messages,
                Message("assistant", reply.text),
                Message("user", REDUCE_REPAIR_PROMPT.format(error=error)),
            ]
            last_error = error
            continue
        # 1-based ids from the model -> 0-based list positions.
        raw_groups = [[i - 1 for i in t.sections] for t in parsed.topics]
        # Keep each group's heading/summary attached while ids are completed:
        # remember which topic each first id belonged to.
        labels = {min(g): t for g, t in zip(raw_groups, parsed.topics, strict=True)}
        groups = assign_ids(raw_groups, len(sections))
        topics = []
        for ids in groups:
            # A completed group still contains its original ids, so exactly
            # one of the labels' keys is in it: the smallest original id.
            label = labels[min(i for i in ids if i in labels)]
            topics.append(
                merge_group(label.heading, label.summary, [sections[i] for i in ids])
            )
        topics.sort(key=lambda t: t.start)
        return ReduceResult(
            topics=topics,
            overview=parsed.overview if final else "",
            takeaways=parsed.takeaways if final else [],
            responses=responses,
        )
    raise NotesError(
        f"reduce: no valid topics after {max_repairs + 1} attempt(s): {last_error}"
    )


def reduce_sections(
    client: LLMClient,
    sections: list[NoteSection],
    *,
    max_words: int = DEFAULT_REDUCE_WORDS,
    max_repairs: int = 1,
    max_tokens: int | None = None,
) -> tuple[ReduceResult, int]:
    """Hierarchical reduce. Returns the final result and the number of
    intermediate levels that were needed (0 if everything fit at once).

    `sections` should carry `sources` (their own index for map output);
    merged topics carry the union of their members' sources.
    """
    if max_words <= 0:
        raise ValueError(f"max_words must be positive, got {max_words}")
    items = list(sections)
    responses: list[LLMResponse] = []
    levels = 0
    while render_words(items) > max_words and len(items) > 2 and levels < MAX_LEVELS:
        merged: list[NoteSection] = []
        for batch in pack_batches(items, max_words):
            result = reduce_once(
                client,
                batch,
                final=False,
                max_repairs=max_repairs,
                max_tokens=max_tokens,
            )
            merged.extend(result.topics)
            responses.extend(result.responses)
        levels += 1
        if len(merged) >= len(items):
            # The model merged nothing; another level would not help.
            items = merged
            break
        items = merged
    final = reduce_once(
        client, items, final=True, max_repairs=max_repairs, max_tokens=max_tokens
    )
    final.responses = responses + final.responses
    return final, levels


def reduce_notes(
    notes: Notes,
    client: LLMClient,
    *,
    max_words: int = DEFAULT_REDUCE_WORDS,
    max_repairs: int = 1,
    max_tokens: int | None = None,
) -> Notes:
    """Return a copy of map `notes` with sections merged into topics.

    Fewer than two sections: nothing to merge, the notes come back as they
    are. If the LLM fails (bad JSON after every repair, or an API error),
    the map notes are returned with `reduce_error` explaining why.
    """
    if len(notes.sections) < 2:
        return notes
    details = [replace(s, sources=[i]) for i, s in enumerate(notes.sections)]
    try:
        result, levels = reduce_sections(
            client,
            details,
            max_words=max_words,
            max_repairs=max_repairs,
            max_tokens=max_tokens,
        )
    except (NotesError, LLMError) as error:
        return replace(notes, reduce_error=str(error))
    calls = result.responses
    return replace(
        notes,
        sections=result.topics,
        overview=result.overview,
        takeaways=result.takeaways,
        detail_sections=details,
        reduce_calls=len(calls),
        reduce_levels=levels,
        llm_calls=notes.llm_calls + len(calls),
        prompt_tokens=notes.prompt_tokens + sum(r.prompt_tokens or 0 for r in calls),
        completion_tokens=notes.completion_tokens
        + sum(r.completion_tokens or 0 for r in calls),
        latency_s=notes.latency_s + sum(r.latency_s for r in calls),
    )
