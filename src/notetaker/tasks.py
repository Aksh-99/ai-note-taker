"""Find the moments in a lecture where the lecturer assigns work.

Step 4.1 needs a "tasks" stage at the end of the agent pipeline. Full deadline
extraction (an LLM turning "the problem set is due Friday" into a structured
task with a real date) arrives in Step 4.2. This module is the cheap first
half of that design: a keyword scan that finds *candidate* sentences, free
and offline, with timestamps.

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
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from notetaker.export import format_timestamp
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


class TaskCandidate(BaseModel):
    """One sentence that probably assigns work, with where it was said."""

    text: str = Field(min_length=1)
    # The sentence plus its neighbours, so Step 4.2's LLM can read things
    # like "That one." -> "Problem set 3 is due Friday. That one."
    context: str
    start: float = Field(ge=0)  # seconds into the lecture
    cues: list[str] = Field(min_length=1)  # which TASK_CUES matched
    time_phrases: list[str] = Field(default_factory=list)  # e.g. ["Friday"]
    score: float = Field(ge=0)


class TaskList(BaseModel):
    """Everything the tasks stage found in one lecture."""

    title: str
    # How the candidates were found. Step 4.2 adds LLM-extracted tasks with
    # resolved dates next to the raw candidates.
    method: str = "keyword-candidates"
    candidates: list[TaskCandidate] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        if not self.candidates:
            lines.append("No homework, deadlines or exams were mentioned.")
            return "\n".join(lines) + "\n"
        lines.append(
            "Sentences where the lecturer seems to assign work "
            "(keyword scan, not yet checked by an LLM):"
        )
        lines.append("")
        for c in self.candidates:
            when = f" (said: {', '.join(c.time_phrases)})" if c.time_phrases else ""
            lines.append(f"- **[{format_timestamp(c.start)}]** {c.text}{when}")
        return "\n".join(lines) + "\n"


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


def extract_tasks(transcript: Transcript, *, title: str = "Tasks") -> TaskList:
    """The tasks stage of the pipeline (keyword version, Step 4.1)."""
    return TaskList(title=title, candidates=find_task_candidates(transcript))


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
