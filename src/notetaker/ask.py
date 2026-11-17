"""Answer questions about your lectures, with citations to lecture + timestamp.

This is the "generation" half of RAG (retrieval-augmented generation). Step
5.1 built the search index and Step 5.2 filled it; this module uses it:

    question
        -> retrieve        top-k passages from the FAISS index (VectorStore)
        -> number them     [1] Lecture 5 [00:12:41] ... [2] Lecture 7 ...
        -> one LLM call    "answer using ONLY these sources, cite like [2]"
        -> check citations every [n] must point at a real retrieved passage
        -> Answer          text + the cited passages with lecture and time

Why retrieve first instead of just asking the model?
    A free 8B model has never heard your professor's lectures. Even a large
    model will happily make up a plausible answer. Retrieval hands the model
    the exact transcript passages that matter, so the answer is grounded in
    what was actually said, and every claim can be traced to a moment in a
    recording you can jump to and re-watch.

Why numbered sources instead of asking the model for timestamps?
    Models are bad at copying long ids and times exactly, and a hallucinated
    timestamp looks just as convincing as a real one. A small integer is easy
    to copy, and we can check it: `[3]` is only accepted if passage 3 was
    really retrieved. The lecture title and timestamp are then filled in by
    our code from the passage's metadata, never by the model. So a citation
    may be the wrong passage (the model's judgement), but it is never a
    moment that does not exist.

Renumbering
    The model might cite [4] and [1] and never [2]. The final answer renumbers
    citations in order of first appearance ([4] becomes [1], [1] becomes [2])
    so the reader sees a tidy 1, 2, 3 that matches the source list below it,
    the same convention as a research paper.

Saying "I don't know"
    Two safety valves stop the model from answering questions the lectures
    never covered. Before any LLM call, `min_score` can reject a question
    whose best passage is not similar enough (no tokens spent). After the
    call, the model itself can return `"answerable": false` when the sources
    do not contain the answer.

Offline mode
    `use_llm=False` skips the model and returns the best passage verbatim as
    the answer. That is a useful baseline, and it is how the command works on
    a machine without Ollama or a free API key.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from notetaker.export import format_timestamp
from notetaker.llm import LLMClient, LLMResponse, Message
from notetaker.vectorstore import SearchHit, VectorStore

# Five passages of up to 120 words is about 600 words (800 tokens) of
# context: small enough for any free model, enough to cover most questions.
DEFAULT_K = 5

# Answers should be a short paragraph; this caps a rambling model.
DEFAULT_MAX_TOKENS = 600

# Answer modes, stored in Answer.mode so callers (and the retrieval eval in
# Step 5.4) can tell how an answer was produced.
MODE_LLM = "llm"  # the model wrote the answer from the sources
MODE_EXTRACTIVE = "extractive"  # no model: the best passage, quoted
MODE_NO_MATCH = "no-match"  # nothing relevant enough was retrieved

NO_MATCH_TEXT = (
    "I could not find this in the indexed lectures: no passage matched the "
    "question closely enough."
)

SYSTEM_PROMPT = """\
You are a teaching assistant answering a student's question about their \
lectures. Use only the numbered transcript sources you are given; never add \
facts, formulas or examples that are not in them. Cite sources by number. \
Answer with JSON only, no prose before or after it."""

USER_PROMPT_TEMPLATE = """\
Question: {question}

Sources (numbered excerpts from lecture transcripts):

{sources}

Answer the question in a short paragraph using ONLY the sources above. After \
each sentence, cite the source(s) it comes from by number in square brackets, \
for example [2] or [1][3]. Only cite numbers from the list above. If the \
sources do not contain the answer, set "answerable" to false and say briefly \
that the lectures do not cover it.

Return a JSON object of this shape:
{{"answer": "your answer with citations like [1]", "answerable": true}}"""

REPAIR_PROMPT = """\
Your previous reply could not be used: {error}
Return only the corrected JSON object, with no text before or after it."""

NO_CITATION_ERROR = (
    "the answer cites no sources. Add the number of the source each sentence "
    "comes from in square brackets, e.g. [1]"
)

# A citation marker: one or more integers in square brackets, separated by
# commas, e.g. "[2]" or "[1, 3]". The (?:...) group is "non-capturing": it
# groups the ", 3" repeats without creating an extra match group.
_MARKER = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


class AskError(RuntimeError):
    """The model never produced a usable answer."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    """A retrieved passage as shown to the reader, with its display number."""

    number: int  # [number] in the answer text
    passage_id: str
    lecture_id: str
    title: str
    start: float  # seconds into that lecture's recording
    end: float
    score: float  # cosine similarity to the question
    text: str

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start)

    def label(self) -> str:
        """Human-readable reference, e.g. 'Lecture 5: Eigenvalues [00:12:41]'."""
        return f"{self.title or self.lecture_id} [{self.timestamp}]"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp
        return data

    @classmethod
    def from_hit(cls, hit: SearchHit, number: int) -> Citation:
        p = hit.passage
        return cls(
            number=number,
            passage_id=p.passage_id,
            lecture_id=p.lecture_id,
            title=p.title,
            start=p.start,
            end=p.end,
            score=hit.score,
            text=p.text,
        )


@dataclass
class Answer:
    """The answer to one question, plus everything needed to check it."""

    question: str
    text: str  # answer with citation markers renumbered 1, 2, 3, ...
    answerable: bool
    mode: str  # MODE_LLM, MODE_EXTRACTIVE or MODE_NO_MATCH
    citations: list[Citation] = field(default_factory=list)  # cited, in order
    retrieved: list[Citation] = field(default_factory=list)  # all, by rank
    warnings: list[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    llm_calls: int = 0
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "answerable": self.answerable,
            "mode": self.mode,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved": [c.to_dict() for c in self.retrieved],
            "warnings": self.warnings,
            "provider": self.provider,
            "model": self.model,
            "llm_calls": self.llm_calls,
            "latency_s": round(self.latency_s, 3),
        }

    def to_markdown(self, excerpt_words: int = 40) -> str:
        """Answer, then a numbered source list with short excerpts."""
        lines = [f"## {self.question}", "", self.text, ""]
        if self.citations:
            lines += ["**Sources**", ""]
            for c in self.citations:
                lines.append(
                    f"{c.number}. {c.label()} (similarity {c.score:.2f}): "
                    f'"{_excerpt(c.text, excerpt_words)}"'
                )
            lines.append("")
        for warning in self.warnings:
            lines.append(f"> Note: {warning}")
        if self.warnings:
            lines.append("")
        return "\n".join(lines)


def _excerpt(text: str, words: int) -> str:
    parts = text.split()
    if len(parts) <= words:
        return text.strip()
    return " ".join(parts[:words]) + " ..."


# ---------------------------------------------------------------------------
# Prompt building and reply checking
# ---------------------------------------------------------------------------


def format_sources(hits: list[SearchHit]) -> str:
    """Number the retrieved passages for the prompt, 1-based, best first."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        p = hit.passage
        span = f"{format_timestamp(p.start)}-{format_timestamp(p.end)}"
        blocks.append(f"[{i}] {p.title or p.lecture_id}, {span}\n{p.text.strip()}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[SearchHit]) -> list[Message]:
    prompt = USER_PROMPT_TEMPLATE.format(
        question=question.strip(), sources=format_sources(hits)
    )
    return [Message("system", SYSTEM_PROMPT), Message("user", prompt)]


class AnswerSchema(BaseModel):
    """The JSON reply we expect from the model."""

    answer: str = Field(min_length=1)
    answerable: bool = True

    @field_validator("answer", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


def validate_reply(text: str) -> AnswerSchema:
    """Parse one reply. Raises ValueError with a reason the model can read.

    json.JSONDecodeError and pydantic's ValidationError both subclass
    ValueError, so callers catch a single type.
    """
    data = LLMResponse(text=text, model="", provider="").json()
    if isinstance(data, str):  # a bare JSON string: treat it as the answer
        data = {"answer": data}
    try:
        return AnswerSchema.model_validate(data)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'reply'}: {e['msg']}"
            for e in error.errors()[:5]
        )
        raise ValueError(f"wrong JSON shape ({problems})") from None


@dataclass(frozen=True)
class ResolvedCitations:
    """Result of checking and renumbering the [n] markers in an answer."""

    text: str  # answer with valid markers renumbered
    order: list[int]  # original source numbers, in order of first citation
    unknown: list[str]  # markers that point at no retrieved source, e.g. "[7]"


def resolve_citations(text: str, n_sources: int) -> ResolvedCitations:
    """Check every [n] marker against the n_sources retrieved passages.

    A bracket group counts as a citation only if every number in it is a real
    source (1..n_sources). Anything else is left in the text untouched,
    because brackets also appear in ordinary maths, like the interval [0, 1].
    Single numbers out of range (such as "[7]" with five sources) are almost
    certainly invented citations, so they are reported in `unknown`.

    Valid markers are renumbered in order of first appearance and a group
    like "[3, 1]" is written as "[1][2]", one number per bracket.
    """
    new_number: dict[int, int] = {}  # original source number -> display number
    unknown: list[str] = []

    # re.sub can take a function instead of a replacement string: it is
    # called once per match and returns the text to put in its place. That
    # lets us renumber as we scan left to right.
    def replace(match: re.Match[str]) -> str:
        numbers = [int(n) for n in re.split(r"\s*,\s*", match.group(1))]
        if not all(1 <= n <= n_sources for n in numbers):
            if len(numbers) == 1:
                unknown.append(match.group(0))
            return match.group(0)
        out = []
        for n in numbers:
            if n not in new_number:
                new_number[n] = len(new_number) + 1
            display = f"[{new_number[n]}]"
            if display not in out:  # "[2, 2]" becomes a single "[1]"
                out.append(display)
        return "".join(out)

    renumbered = _MARKER.sub(replace, text)
    # dicts remember insertion order, so the keys are in first-cited order.
    return ResolvedCitations(renumbered, list(new_number), unknown)


# ---------------------------------------------------------------------------
# The main entry point
# ---------------------------------------------------------------------------


def ask(
    store: VectorStore,
    question: str,
    client: LLMClient | None = None,
    *,
    k: int = DEFAULT_K,
    lecture_ids: list[str] | None = None,
    min_score: float | None = None,
    use_llm: bool = True,
    max_repairs: int = 1,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
) -> Answer:
    """Answer `question` from the lectures in `store`, citing passages.

    `client` is any object with a `chat()` method (a real LLMClient, a
    FakeLLMClient in tests, or a cassette ReplayClient); it is not needed
    when `use_llm=False`. `max_repairs` is how many times the model is asked
    to fix an unusable reply (broken JSON, or an answer with no citations).
    """
    question = question.strip()
    if not question:
        raise ValueError("the question is empty")
    if max_repairs < 0:
        raise ValueError(f"max_repairs must be >= 0, got {max_repairs}")
    if use_llm and client is None:
        raise ValueError("use_llm=True needs an LLM client")

    hits = store.search(question, k=k, lecture_ids=lecture_ids)
    retrieved = [Citation.from_hit(h, h.rank) for h in hits]
    answer = Answer(question=question, text="", answerable=False, mode=MODE_NO_MATCH)
    answer.retrieved = retrieved

    # Safety valve 1: nothing relevant was found, so do not ask the model.
    if not hits or (min_score is not None and hits[0].score < min_score):
        answer.text = NO_MATCH_TEXT
        if hits:
            answer.warnings.append(
                f"best similarity {hits[0].score:.2f} is below --min-score "
                f"{min_score:.2f}"
            )
        return answer

    if not use_llm:
        # Extractive baseline: the single best passage, verbatim.
        best = retrieved[0]
        answer.mode = MODE_EXTRACTIVE
        answer.answerable = True
        answer.text = f'"{best.text.strip()}" [1]'
        answer.citations = [best]
        return answer

    answer.mode = MODE_LLM
    answer.provider = getattr(client, "provider", "")
    answer.model = getattr(client, "model", "")
    messages = build_messages(question, hits)
    start = time.perf_counter()
    last_error: ValueError | None = None
    parsed: AnswerSchema | None = None
    resolved: ResolvedCitations | None = None
    # The last valid-but-uncited reply, kept in case a repair attempt then
    # comes back as broken JSON: an uncited answer beats no answer.
    uncited: tuple[AnswerSchema, ResolvedCitations] | None = None

    for attempt in range(max_repairs + 1):
        reply = client.chat(messages, json_mode=True, max_tokens=max_tokens)
        answer.llm_calls += 1
        error: ValueError | None = None
        try:
            parsed = validate_reply(reply.text)
            resolved = resolve_citations(parsed.answer, len(hits))
            if parsed.answerable and not resolved.order:
                error = ValueError(NO_CITATION_ERROR)
                uncited = (parsed, resolved)
        except ValueError as exc:
            parsed, resolved, error = None, None, exc
            if uncited is not None:
                parsed, resolved = uncited
        if error is None:
            break
        last_error = error
        # An uncited answer is still an answer: keep it if we are out of
        # repairs. Broken JSON is not, so that case falls through to AskError.
        if attempt < max_repairs:
            # Show the model its own reply and what was wrong with it.
            messages = [
                *messages,
                Message("assistant", reply.text),
                Message("user", REPAIR_PROMPT.format(error=error)),
            ]
    answer.latency_s = time.perf_counter() - start

    if parsed is None or resolved is None:
        raise AskError(
            f"no usable answer after {answer.llm_calls} attempt(s): {last_error}"
        )

    answer.text = resolved.text
    answer.answerable = parsed.answerable
    by_number = {c.number: c for c in retrieved}
    answer.citations = [
        Citation(**{**asdict(by_number[old]), "number": new})
        for new, old in enumerate(resolved.order, start=1)
    ]
    if resolved.unknown:
        answer.warnings.append(
            "ignored citation(s) to sources that were not retrieved: "
            + ", ".join(resolved.unknown)
        )
    if parsed.answerable and not answer.citations:
        answer.warnings.append(
            "the model did not cite any source; check the answer against the "
            "retrieved passages"
        )
    return answer
