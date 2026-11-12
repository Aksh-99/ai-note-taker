"""Find the moments in a lecture where the lecturer assigns work.

Two halves, built in Steps 4.1 and 4.2:

1. A keyword scan (Step 4.1) finds *candidate* sentences, free and offline,
   with timestamps.
2. An LLM reads only those candidates (Step 4.2) and turns the real ones into
   structured tasks: a short title, a kind (homework, exam, ...) and the
   deadline phrase copied word for word. Code in `notetaker.dates` then turns
   that phrase into a calendar date relative to the lecture date. Without an
   LLM (`use_llm=False`) a simple rule-based fallback does the same job less
   cleverly.

Why split it like this? It is the same "retrieve, then read" idea as RAG and
the faithfulness eval. A 90-minute lecture has maybe three sentences about
homework. Sending all 12,000 words to an LLM just to find them is slow and
burns free-tier quota, while a regex finds them in milliseconds. The LLM in
Step 4.2 then only reads the few candidates (with a little context around
each), which is cheaper and also more accurate, because the model is not
distracted by an hour of unrelated maths.

The scan is tuned for recall over precision: a false alarm ("this proof is
due to Gauss") costs one short LLM call later, while a missed deadline is the
one error a student really cares about.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from notetaker.dates import resolve_due
from notetaker.export import format_timestamp
from notetaker.llm import LLMClient, LLMError, LLMResponse, Message
from notetaker.transcribe import Transcript

# ---------------------------------------------------------------------------
# Cue words
# ---------------------------------------------------------------------------

# Each cue is a compiled regular expression. `\b` is a "word boundary", so
# r"\bdue\b" matches "due" but not "residue" or "subdue". `s?` makes a final
# "s" optional (assignment / assignments). re.IGNORECASE ignores capitals.
TASK_CUES: dict[str, re.Pattern[str]] = {
    "due": re.compile(r"\bdue\b", re.IGNORECASE),
    "deadline": re.compile(r"\bdeadlines?\b", re.IGNORECASE),
    "homework": re.compile(r"\bhomeworks?\b", re.IGNORECASE),
    "assignment": re.compile(r"\bassignments?\b", re.IGNORECASE),
    "problem set": re.compile(r"\b(?:problem sets?|p-?sets?)\b", re.IGNORECASE),
    "exam": re.compile(r"\b(?:exams?|midterms?|quiz(?:zes)?)\b", re.IGNORECASE),
    "submit": re.compile(
        r"\b(?:submit|hand (?:it )?in|turn (?:it )?in)\b", re.IGNORECASE
    ),
    "reading": re.compile(
        r"\b(?:read|reading)\b.{0,30}\b(?:chapters?|sections?|pages?|notes)\b",
        re.IGNORECASE,
    ),
    "project": re.compile(
        r"\b(?:project proposal|final project|lab report)\b", re.IGNORECASE
    ),
}

# Time expressions. On their own they are not a task ("last Friday we saw
# eigenvectors"), but next to a task cue they make a real deadline likely,
# so they raise the score and Step 4.2 will resolve them to calendar dates.
TIME_CUE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"tomorrow|tonight|today|next (?:week|class|lecture|time)|"
    r"end of (?:the )?(?:week|month|term|semester)|"
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|midnight|noon|"
    r"\d{1,2}(?::\d\d)?\s?(?:am|pm)|\d{1,2}/\d{1,2})\b",
    re.IGNORECASE,
)

# A sentence ends at ".", "!" or "?" followed by whitespace. The `(?<=...)`
# is a "lookbehind": it requires the punctuation before the split point but
# keeps it attached to the sentence instead of throwing it away.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

# The kinds of task a student can get. `Literal[...]` restricts a field to
# exactly these strings, and pydantic rejects anything else.
TaskKind = Literal["homework", "exam", "reading", "project", "other"]
KINDS: tuple[str, ...] = ("homework", "exam", "reading", "project", "other")

# Words small models use instead of our kind names, mapped to ours.
KIND_SYNONYMS: dict[str, str] = {
    "assignment": "homework",
    "problem set": "homework",
    "pset": "homework",
    "exercise": "homework",
    "lab": "project",
    "report": "project",
    "midterm": "exam",
    "final": "exam",
    "quiz": "exam",
    "test": "exam",
    "read": "reading",
}


class TaskCandidate(BaseModel):
    """One sentence that probably assigns work, with where it was said."""

    text: str = Field(min_length=1)
    # The sentence plus its neighbours, so the LLM can read things like
    # "That one." -> "Problem set 3 is due Friday. That one."
    context: str
    start: float = Field(ge=0)  # seconds into the lecture
    cues: list[str] = Field(min_length=1)  # which TASK_CUES matched
    time_phrases: list[str] = Field(default_factory=list)  # e.g. ["Friday"]
    score: float = Field(ge=0)


class Task(BaseModel):
    """One real piece of work for the student, with a resolved deadline."""

    title: str = Field(min_length=1)  # "Problem set 3"
    kind: TaskKind = "other"
    details: str = ""  # "questions 1 to 5", "chapters 4 and 5"
    due_phrase: str | None = None  # the words the lecturer used: "Friday at 5pm"
    due_date: date | None = None  # resolved by notetaker.dates, None if unknown
    due_time: str | None = None  # "HH:MM", 24 h
    date_rule: str = ""  # how due_date was found, for a human to check
    date_ambiguous: bool = False  # True: double-check this one
    start: float = Field(ge=0)  # where it was said, seconds
    mentions: list[float] = Field(default_factory=list)  # every time it came up
    evidence: str  # the candidate sentence it came from
    source: Literal["llm", "rules"] = "llm"


class TaskList(BaseModel):
    """Everything the tasks stage found in one lecture."""

    title: str
    # "keyword-candidates" (scan only), "llm" or "rules" (Step 4.2).
    method: str = "keyword-candidates"
    reference_date: date | None = None  # the lecture date deadlines are relative to
    candidates: list[TaskCandidate] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    rejected: list[int] = Field(
        default_factory=list
    )  # candidate numbers judged "not a task"
    warnings: list[str] = Field(default_factory=list)
    model: str | None = None
    provider: str | None = None
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.reference_date is not None:
            lines += [f"Lecture date: {self.reference_date.isoformat()}", ""]
        if self.tasks:
            lines.append("## To do")
            lines.append("")
            for t in self.tasks:
                lines.append("- [ ] " + _task_line(t))
            lines.append("")
            if any(t.date_ambiguous for t in self.tasks):
                lines.append(
                    "(?) = the date phrase was ambiguous, please double-check."
                )
                lines.append("")
        elif self.method in ("llm", "rules"):
            lines += ["No homework, deadlines or exams were found.", ""]
        if not self.candidates:
            if self.method == "keyword-candidates":
                lines.append("No homework, deadlines or exams were mentioned.")
            return "\n".join(lines).rstrip() + "\n"
        lines.append("## Candidate sentences (keyword scan)")
        lines.append("")
        for i, c in enumerate(self.candidates, start=1):
            when = f" (said: {', '.join(c.time_phrases)})" if c.time_phrases else ""
            mark = " (not a task)" if i in self.rejected else ""
            lines.append(f"- **[{format_timestamp(c.start)}]** {c.text}{when}{mark}")
        for w in self.warnings:
            lines += ["", f"Warning: {w}"]
        return "\n".join(lines) + "\n"


def _task_line(t: Task) -> str:
    """One readable line: title, kind, due date and where it was said."""
    if t.due_date is not None:
        when = t.due_date.strftime("%a %Y-%m-%d")
        if t.due_time:
            when += f" {t.due_time}"
        due = f"due **{when}**" + (" (?)" if t.date_ambiguous else "")
    elif t.due_phrase:
        due = f'due "{t.due_phrase}" (date not resolved)'
    else:
        due = "no deadline given"
    details = f": {t.details}" if t.details else ""
    return f"**{t.title}** ({t.kind}){details}, {due} [{format_timestamp(t.start)}]"


# ---------------------------------------------------------------------------
# Sentence splitting with timestamps
# ---------------------------------------------------------------------------


@dataclass
class _Sentence:
    text: str
    start: float


def split_sentences(transcript: Transcript) -> list[_Sentence]:
    """Split the transcript into sentences, each with its start time.

    Whisper segments do not line up with sentences: one sentence can span
    two segments and one segment can hold two sentences. So we glue all
    segment texts into one string, remember the character offset where each
    segment begins, split that string into sentences, and give every
    sentence the start time of the segment holding its first character.
    """
    pieces: list[str] = []
    offsets: list[tuple[int, float]] = []  # (char offset, segment start time)
    position = 0
    for segment in transcript.segments:
        text = segment.text.strip()
        if not text:
            continue
        offsets.append((position, segment.start))
        pieces.append(text)
        position += len(text) + 1  # +1 for the joining space
    full = " ".join(pieces)

    sentences: list[_Sentence] = []
    cursor = 0  # where we are in `full`
    seg_index = 0
    for raw in SENTENCE_END.split(full):
        begin = full.find(raw, cursor)
        cursor = begin + len(raw)
        # Advance to the last segment that starts at or before `begin`.
        # Both lists are sorted, so one forward pass is enough (no search).
        while seg_index + 1 < len(offsets) and offsets[seg_index + 1][0] <= begin:
            seg_index += 1
        text = raw.strip()
        if text:
            sentences.append(_Sentence(text, offsets[seg_index][1]))
    return sentences


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


def find_task_candidates(
    transcript: Transcript, *, context_sentences: int = 1
) -> list[TaskCandidate]:
    """Return every sentence with at least one task cue, in lecture order.

    score = number of distinct task cues + 1 if a time phrase is present.
    "Problem set 3 is due Friday" scores 3 (problem set, due, time), while a
    passing "on the exam" scores 1. Step 4.2 can read the highest first.
    """
    sentences = split_sentences(transcript)
    found: list[TaskCandidate] = []
    for i, sentence in enumerate(sentences):
        cues = [name for name, rx in TASK_CUES.items() if rx.search(sentence.text)]
        if not cues:
            continue
        times = [m.group(0) for m in TIME_CUE.finditer(sentence.text)]
        lo = max(0, i - context_sentences)
        hi = min(len(sentences), i + context_sentences + 1)
        found.append(
            TaskCandidate(
                text=sentence.text,
                context=" ".join(s.text for s in sentences[lo:hi]),
                start=sentence.start,
                cues=cues,
                # dict.fromkeys removes repeats but keeps first-seen order.
                time_phrases=list(dict.fromkeys(times)),
                score=len(cues) + (1.0 if times else 0.0),
            )
        )
    return found


# ---------------------------------------------------------------------------
# Step 4.2: the LLM reads the candidates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a careful teaching assistant who extracts student to-dos \
(homework, exams, readings, projects and their deadlines) from short \
lecture excerpts. Answer with JSON only."""

USER_PROMPT_TEMPLATE = """\
Below are numbered excerpts from a lecture. A keyword scan flagged each one \
because it might assign work to the students, but many are false alarms.

For every excerpt that really gives students something to do or prepare \
for (homework, problem set, exam, quiz, reading, project, lab report), \
return one task. Skip false alarms such as "this result is due to Gauss", \
past events ("last week's exam"), or machine-learning "test sets".

Return a JSON object: {{"tasks": [...]}}. Each task has:
- "excerpt": the excerpt number it comes from
- "title": a short name, at most 8 words (e.g. "Problem set 3")
- "kind": one of homework, exam, reading, project, other
- "details": extra specifics in at most 15 words, or ""
- "due_phrase": the deadline words copied exactly from the excerpt \
(e.g. "Friday at 5pm", "next Tuesday"), or null if no deadline is said

Do NOT convert the deadline to a date; copy the words. If no excerpt is a \
real task, return {{"tasks": []}}.

Excerpts:
{excerpts}"""

REPAIR_PROMPT = """\
Your previous reply could not be used: {error}
Return only the corrected JSON object with a "tasks" list as described."""

DEFAULT_BATCH_SIZE = 8  # candidates per LLM call
DEFAULT_MAX_CANDIDATES = 40  # more than this in one lecture: keep the best
DEFAULT_MAX_TOKENS = 1024


class TaskSchema(BaseModel):
    """What one task in a model reply must look like."""

    excerpt: int
    title: str = Field(min_length=1, max_length=120)
    kind: TaskKind = "other"
    details: str = ""
    due_phrase: str | None = None

    # A "before" validator runs on the raw value, before type checking, so
    # it can clean up what small models send ("Midterm" -> "exam", "" ->
    # None) instead of rejecting the whole reply.
    @field_validator("kind", mode="before")
    @classmethod
    def _normalise_kind(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        key = value.strip().lower()
        if key in KINDS:
            return key
        return KIND_SYNONYMS.get(key, "other")

    @field_validator("title", "details", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return (
            value.strip()
            if isinstance(value, str)
            else ("" if value is None else value)
        )

    @field_validator("due_phrase", mode="before")
    @classmethod
    def _empty_is_none(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().strip('"').strip()
            return value or None
        return value


class TasksReplySchema(BaseModel):
    tasks: list[TaskSchema] = Field(default_factory=list)


def validate_tasks_reply(text: str, n_excerpts: int) -> TasksReplySchema:
    """Parse one reply; raise ValueError with a reason the model can act on.

    Accepts a bare list as well as {"tasks": [...]}, and "candidate" or "id"
    instead of "excerpt", common small-model variants.
    """
    data = LLMResponse(text=text, model="", provider="").json()
    if isinstance(data, list):
        data = {"tasks": data}
    if isinstance(data, dict):
        for item in data.get("tasks") or []:
            if isinstance(item, dict) and "excerpt" not in item:
                for alias in ("candidate", "id", "number", "index"):
                    if alias in item:
                        item["excerpt"] = item.pop(alias)
                        break
    try:
        reply = TasksReplySchema.model_validate(data)
    except ValidationError as error:
        reasons = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in error.errors()[:5]
        )
        raise ValueError(f"wrong JSON shape ({reasons})") from None
    bad = [t.excerpt for t in reply.tasks if not 1 <= t.excerpt <= n_excerpts]
    if bad:
        raise ValueError(f"excerpt numbers {bad} do not exist (use 1 to {n_excerpts})")
    return reply


_WORDS = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORDS.findall(text.lower())


def phrase_in_text(phrase: str, text: str) -> bool:
    """True if `phrase` occurs in `text`, ignoring case and punctuation.

    Used to check the model copied the deadline instead of inventing one.
    Comparing word lists (not raw strings) means "Friday, 5 p.m." matches
    "friday 5 p m" and extra spaces do not matter.
    """
    needle, hay = _words(phrase), _words(text)
    if not needle:
        return False
    n = len(needle)
    return any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


def _format_excerpts(batch: list[TaskCandidate]) -> str:
    return "\n".join(
        f"{i}. [{format_timestamp(c.start)}] {c.context}"
        for i, c in enumerate(batch, start=1)
    )


def _ask_batch(
    client: LLMClient,
    batch: list[TaskCandidate],
    *,
    max_repairs: int,
    max_tokens: int | None,
) -> tuple[TasksReplySchema, list[LLMResponse]]:
    """One LLM call (plus repairs) for one batch of candidates."""
    prompt = USER_PROMPT_TEMPLATE.format(excerpts=_format_excerpts(batch))
    messages = [Message("system", SYSTEM_PROMPT), Message("user", prompt)]
    responses: list[LLMResponse] = []
    last_error: ValueError | None = None
    for _ in range(max_repairs + 1):
        reply = client.chat(messages, json_mode=True, max_tokens=max_tokens)
        responses.append(reply)
        try:
            return validate_tasks_reply(reply.text, len(batch)), responses
        except ValueError as error:
            # Show the model its own reply and the problem, then ask again.
            messages = [
                *messages,
                Message("assistant", reply.text),
                Message("user", REPAIR_PROMPT.format(error=error)),
            ]
            last_error = error
    raise _BatchFailed(str(last_error), responses)


class _BatchFailed(Exception):
    """Internal: a batch never produced valid JSON. Carries its replies."""

    def __init__(self, reason: str, responses: list[LLMResponse]):
        super().__init__(reason)
        self.responses = responses


# ---------------------------------------------------------------------------
# Rule-based fallback (no LLM)
# ---------------------------------------------------------------------------

# Cue name -> task kind, for the fallback.
CUE_KINDS: dict[str, str] = {
    "homework": "homework",
    "assignment": "homework",
    "problem set": "homework",
    "exam": "exam",
    "reading": "reading",
    "project": "project",
}

# Grab a name like "problem set 3", "homework two", "the midterm".
TITLE_PATTERN = re.compile(
    r"\b(?:problem sets?|p-?sets?|homeworks?|assignments?|labs?|"
    r"midterms?|exams?|quiz(?:zes)?|final project|project proposal|lab report)"
    r"(?:\s+(?:number\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten))?\b",
    re.IGNORECASE,
)


# Words that change a time phrase's meaning and sit just before it
# ("next Friday", "this Monday") or a day number just after a month
# ("December 3rd"). TIME_CUE alone finds only "Friday" / "December".
_BEFORE_TIME = re.compile(r"(?:\b(?:next|this|coming)\s+)$", re.IGNORECASE)
_AFTER_TIME = re.compile(r"^(?:\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)?\b)", re.IGNORECASE)


def keyword_due_phrase(candidate: TaskCandidate) -> str | None:
    """The deadline words in a candidate sentence, found without an LLM.

    Takes the stretch of the sentence from the first time phrase to the last
    (so "Friday at 5pm" stays together), widened to include a "next" before
    it and a day number after it.
    """
    text = candidate.text
    matches = list(TIME_CUE.finditer(text))
    if not matches:
        return None
    begin, end = matches[0].start(), matches[-1].end()
    if m := _BEFORE_TIME.search(text[:begin]):
        # `:=` (the "walrus" operator) assigns and tests in one step.
        begin = m.start()
    if m := _AFTER_TIME.match(text[end:]):
        end += m.end()
    return text[begin:end].strip(" ,.;")


def rule_based_task(candidate: TaskCandidate) -> TaskSchema | None:
    """Guess a task from a candidate alone. None if it does not look like one.

    Only candidates that mention a time ("Friday", "5pm") count, because a
    cue word without any deadline ("on the exam") is usually not a to-do.
    """
    if not candidate.time_phrases:
        return None
    kind = next((CUE_KINDS[c] for c in candidate.cues if c in CUE_KINDS), "other")
    m = TITLE_PATTERN.search(candidate.text)
    title = m.group(0) if m else kind.capitalize()
    title = title[0].upper() + title[1:]
    return TaskSchema(
        excerpt=1,
        title=title,
        kind=kind,
        due_phrase=keyword_due_phrase(candidate),
    )


# ---------------------------------------------------------------------------
# Building the final list
# ---------------------------------------------------------------------------


def _make_task(
    parsed: TaskSchema,
    candidate: TaskCandidate,
    *,
    source: Literal["llm", "rules"],
    reference: date | None,
    class_days: tuple[int, ...],
) -> Task:
    phrase = parsed.due_phrase
    rule_note = ""
    if phrase and not phrase_in_text(phrase, candidate.context):
        # The model "quoted" words that are not there. Fall back to the time
        # words the keyword scan saw, which are real by construction.
        phrase = keyword_due_phrase(candidate)
        rule_note = " (model's phrase not in transcript; used keyword times)"
    resolved = resolve_due(phrase, reference, class_days=class_days)
    return Task(
        title=parsed.title,
        kind=parsed.kind,
        details=parsed.details,
        due_phrase=phrase,
        due_date=resolved.date,
        due_time=resolved.time,
        date_rule=resolved.rule + rule_note,
        date_ambiguous=resolved.ambiguous,
        start=candidate.start,
        mentions=[candidate.start],
        evidence=candidate.text,
        source=source,
    )


def merge_tasks(tasks: list[Task]) -> list[Task]:
    """Merge repeats ("problem set 3 is due Friday" said twice) into one task.

    Two tasks are the same if their titles have the same words, ignoring
    case, punctuation and filler like "the". The first mention is kept and
    a later mention may fill in a missing deadline.
    """
    filler = {"the", "a", "an", "your", "our", "this", "next"}
    merged: dict[tuple[str, ...], Task] = {}
    for task in tasks:
        key = tuple(w for w in _words(task.title) if w not in filler)
        if key not in merged:
            merged[key] = task
            continue
        kept = merged[key]
        kept.mentions.append(task.start)
        if kept.due_date is None and task.due_date is not None:
            # model_copy(update=...) makes a new model with some fields
            # replaced; pydantic models are not edited field by field here so
            # the date, time and rule always stay consistent with each other.
            merged[key] = kept.model_copy(
                update={
                    k: getattr(task, k)
                    for k in (
                        "due_phrase",
                        "due_date",
                        "due_time",
                        "date_rule",
                        "date_ambiguous",
                    )
                }
            )
    return list(merged.values())


def _sort_key(task: Task) -> tuple[int, date, str, float]:
    # Dated tasks first (soonest first), undated ones after, by lecture time.
    return (
        task.due_date is None,
        task.due_date or date.max,
        task.due_time or "99:99",
        task.start,
    )


def extract_tasks(
    transcript: Transcript,
    client: LLMClient | None = None,
    *,
    title: str = "Tasks",
    lecture_date: date | None = None,
    class_days: tuple[int, ...] = (),
    use_llm: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_repairs: int = 1,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    on_batch: Callable[[int, int], None] | None = None,
) -> TaskList:
    """The tasks stage: keyword scan, then LLM (or rules), then date resolution.

    With `use_llm=False` or `client=None` no LLM is called and the rule-based
    fallback builds tasks from candidates that mention a time.

    If one LLM batch fails (bad JSON after repairs, or an API error), its
    candidates go through the rule-based fallback instead and a warning is
    recorded. A missed deadline is the one mistake that matters most here,
    so a partial failure should degrade quality, not lose tasks.
    """
    if batch_size <= 0 or max_candidates <= 0 or max_repairs < 0:
        raise ValueError("batch_size and max_candidates must be > 0, max_repairs >= 0")
    candidates = find_task_candidates(transcript)
    llm = use_llm and client is not None
    result = TaskList(
        title=title,
        method="llm" if llm else "rules",
        reference_date=lecture_date,
        candidates=candidates,
    )
    if llm:
        result.model = getattr(client, "model", "unknown")
        result.provider = getattr(client, "provider", "unknown")

    # Which candidates to read. Too many means the cue words fire on
    # something the lecture is *about* (an ML lecture on test sets), so keep
    # the highest scores, then restore lecture order. sorted() is stable:
    # equal scores keep their original order.
    chosen = list(range(len(candidates)))
    if len(chosen) > max_candidates:
        chosen = sorted(chosen, key=lambda i: -candidates[i].score)[:max_candidates]
        chosen.sort()
        result.warnings.append(
            f"{len(candidates)} candidates found; only the {max_candidates} "
            "highest-scoring were read"
        )

    tasks: list[Task] = []
    responses: list[LLMResponse] = []

    def add_rules(indices: list[int]) -> None:
        for i in indices:
            guess = rule_based_task(candidates[i])
            if guess is None:
                result.rejected.append(i + 1)
                continue
            tasks.append(
                _make_task(
                    guess,
                    candidates[i],
                    source="rules",
                    reference=lecture_date,
                    class_days=class_days,
                )
            )

    if not llm:
        add_rules(chosen)
    else:
        batches = [
            chosen[k : k + batch_size] for k in range(0, len(chosen), batch_size)
        ]
        for number, indices in enumerate(batches):
            if on_batch is not None:
                on_batch(number, len(batches))
            batch = [candidates[i] for i in indices]
            try:
                reply, replies = _ask_batch(
                    client, batch, max_repairs=max_repairs, max_tokens=max_tokens
                )
            except (_BatchFailed, LLMError) as error:
                responses.extend(getattr(error, "responses", []))
                result.warnings.append(
                    f"batch {number + 1}/{len(batches)} fell back to rules: {error}"
                )
                add_rules(indices)
                continue
            responses.extend(replies)
            used = set()
            for parsed in reply.tasks:
                index = indices[parsed.excerpt - 1]  # excerpt numbers start at 1
                used.add(index)
                tasks.append(
                    _make_task(
                        parsed,
                        candidates[index],
                        source="llm",
                        reference=lecture_date,
                        class_days=class_days,
                    )
                )
            result.rejected.extend(i + 1 for i in indices if i not in used)

    result.tasks = sorted(merge_tasks(tasks), key=_sort_key)
    result.rejected.sort()
    result.llm_calls = len(responses)
    result.prompt_tokens = sum(r.prompt_tokens or 0 for r in responses)
    result.completion_tokens = sum(r.completion_tokens or 0 for r in responses)
    result.latency_s = sum(r.latency_s for r in responses)
    if lecture_date is None and any(t.due_phrase for t in result.tasks):
        result.warnings.append(
            "no lecture date given, so deadlines were not turned into dates"
        )
    return result


def write_tasks(tasks: TaskList, out_dir: str | Path) -> tuple[Path, Path]:
    """Write tasks.json and tasks.md into `out_dir`; return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tasks.json"
    md_path = out_dir / "tasks.md"
    json_path.write_text(tasks.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(tasks.to_markdown(), encoding="utf-8")
    return json_path, md_path


def load_tasks(path: str | Path) -> TaskList:
    """Read tasks.json back, re-validating it."""
    return TaskList.model_validate_json(Path(path).read_text(encoding="utf-8"))
