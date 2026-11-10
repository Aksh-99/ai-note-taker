"""Faithfulness eval: does the generated material only say what the lecture says?

Steps 3.2 to 3.6 produce notes, a glossary, flashcards and a quiz. Each one
already has cheap guards (schema validation, "the answer's words must occur in
the chunk"), but word overlap cannot tell "the gradient points uphill" from
"the gradient points downhill". This module measures faithfulness properly
with an LLM-as-judge, using a free model.

Pipeline:

    notes.json / concepts.json / flashcards.json / quiz.json
        -> claims                (one checkable statement each)
    transcript.json
        -> passages              (pause-based paragraphs with timestamps)
    each claim
        -> evidence retrieval    (top-k passages by IDF-weighted word overlap)
    claims in small batches
        -> judge LLM call        (verdict + exact supporting quote + reason)
        -> validation + repair   (same loop as the generators)
        -> quote verification    (is the quote really in the transcript?)
    -> metrics + report          (faithfulness_<kind>.json + .md)

Key ideas, in the order an interviewer might ask about them:

1. Claims, not documents. Asking "are these notes faithful? 1 to 10" gives a
   vague, unrepeatable number. Asking "is THIS sentence supported by THESE
   passages?" is a narrow yes/no question that even small models answer
   well, and the result points at exactly which bullet is wrong.

2. Retrieval before judging. A 75-minute transcript does not fit a free
   model's context, and long inputs make judges sloppy. So each claim gets
   the few passages most likely to contain its evidence. The score is IDF
   weighted overlap: a shared rare word ("eigenvector") counts much more than
   a shared common one ("matrix" said 200 times). This is the core idea of
   BM25, the classic search ranking function, in its simplest form.

3. Four verdicts. supported / partial / unsupported / contradicted. "partial"
   catches the most common note-taking error: an over-generalized or
   over-specific version of what was said. "unsupported" is a hallucination
   (possibly true, but not from this lecture); "contradicted" is worse, the
   lecture says something different.

4. Trust, but verify the judge. The judge must copy an exact quote from the
   passages for every non-"unsupported" verdict. Code then checks that the
   quote really occurs in the transcript. A "supported" verdict whose quote
   cannot be found is counted as only partial in the strict score. Judges
   (like students) sometimes claim support they cannot show.

5. A no-LLM baseline. Every claim also gets a lexical support score (share of
   its content words found in the evidence). `--lexical-only` turns that into
   verdicts without any LLM call, which is free, instant and runs in CI. When
   the judge runs, we report how often it agrees with the baseline and the
   chance-corrected agreement (Cohen's kappa). If kappa is near 1, the judge
   adds little over word overlap; if it is low, the disagreements are exactly
   the cases worth reading (paraphrases the baseline missed, or negations it
   could not see).

Known biases of LLM judges (and what we do about them):
    * Self-preference: a model rates its own writing higher. The CLI warns
      when the judge is the same model that generated the material; use a
      different (ideally larger) free model as judge, e.g. Groq's free
      llama-3.3-70b-versatile.
    * Randomness: the judge runs at temperature 0.
    * Leniency without evidence: the quote check above.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from notetaker.concepts import ConceptList, term_words
from notetaker.export import format_timestamp, group_paragraphs
from notetaker.flashcards import FlashcardDeck, content_words
from notetaker.llm import LLMClient, LLMResponse, Message
from notetaker.notes import Notes
from notetaker.quiz import Quiz
from notetaker.transcribe import Transcript

# Claims judged per LLM call. Batching cuts the number of calls (free tiers
# limit requests per minute) while keeping each prompt short and focused.
DEFAULT_BATCH_SIZE = 6

# Passages retrieved per claim.
DEFAULT_TOP_K = 3

# Word budget for all evidence in one judge prompt (about 2,000 tokens).
DEFAULT_EVIDENCE_WORDS = 1500

DEFAULT_MAX_TOKENS = 1500

# Lexical-only verdicts: share of a claim's content words found in its
# evidence. Chosen to match the flashcard grounding threshold (0.5).
LEXICAL_SUPPORTED = 0.8
LEXICAL_PARTIAL = 0.5

# A quote counts as found if this share of its words appear as one
# contiguous run in the transcript (allows for small copying slips).
QUOTE_MATCH_RATIO = 0.8

ClaimKind = Literal[
    "overview", "summary", "key_point", "takeaway", "definition", "flashcard", "quiz"
]
TargetKind = Literal["notes", "concepts", "flashcards", "quiz"]
Verdict = Literal["supported", "partial", "unsupported", "contradicted", "unjudged"]
JUDGED_VERDICTS = ("supported", "partial", "unsupported", "contradicted")

SYSTEM_PROMPT = """\
You are a strict fact-checker. You decide whether statements from study \
materials are backed by excerpts of a lecture transcript. Use ONLY the \
excerpts: outside knowledge does not count, even if the statement is true. \
Answer with JSON only."""

USER_PROMPT_TEMPLATE = """\
Lecture transcript excerpts (each starts with an id and its timestamp):

{evidence}

Statements to check:

{claims}

For EVERY statement give one verdict:
- "supported": the excerpts say this (a faithful paraphrase is fine).
- "partial": part of it is in the excerpts, but part is missing, more \
specific, or more general than what was said.
- "unsupported": the excerpts do not say this.
- "contradicted": the excerpts say something different or the opposite.

Answer as a JSON object of this shape:

{{
  "verdicts": [
    {{
      "claim": 1,
      "verdict": "supported",
      "quote": "the exact words from one excerpt that back your verdict",
      "reason": "one short sentence"
    }}
  ]
}}

Rules:
- One entry per statement, using the statement numbers above.
- "quote" is copied word for word from an excerpt (no ids or timestamps). \
Leave it empty only for "unsupported"."""

REPAIR_PROMPT = """\
Your previous reply could not be used: {error}
Return only the corrected JSON object with a "verdicts" list, one entry for \
each of the {count} statements."""


# ---------------------------------------------------------------------------
# Claims: turning each kind of output into checkable statements
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """One statement from the generated material."""

    id: int = Field(ge=1)
    text: str = Field(min_length=1)
    kind: ClaimKind
    source: str = Field(description="Where it came from, e.g. 'section 2, point 3'")
    time: float | None = Field(
        default=None, description="Lecture time (s) the generator linked it to"
    )


def _number(drafts: list[tuple[str, ClaimKind, str, float | None]]) -> list[Claim]:
    """Give claims ids 1, 2, 3, ... and drop empty texts."""
    claims = []
    for text, kind, source, time in drafts:
        if text.strip():
            claims.append(
                Claim(
                    id=len(claims) + 1,
                    text=text.strip(),
                    kind=kind,
                    source=source,
                    time=time,
                )
            )
    return claims


def claims_from_notes(notes: Notes) -> list[Claim]:
    """Overview, every section summary and key point, and the takeaways.

    We check the FINAL sections (what a student reads). After a reduce step
    those are the merged topics; the per-chunk drafts are not re-checked.
    """
    drafts: list[tuple[str, ClaimKind, str, float | None]] = []
    if notes.overview:
        drafts.append((notes.overview, "overview", "overview", None))
    for s_i, section in enumerate(notes.sections, start=1):
        where = f"section {s_i} ({section.heading})"
        if section.summary:
            drafts.append(
                (section.summary, "summary", f"{where}, summary", section.start)
            )
        for p_i, point in enumerate(section.key_points, start=1):
            drafts.append((point, "key_point", f"{where}, point {p_i}", section.start))
    for t_i, point in enumerate(notes.takeaways, start=1):
        drafts.append((point, "takeaway", f"takeaway {t_i}", None))
    return _number(drafts)


def claims_from_concepts(concepts: ConceptList) -> list[Claim]:
    """One claim per glossary entry: 'term: definition'."""
    return _number(
        [
            (
                f"{c.term}: {c.definition}",
                "definition",
                f"concept '{c.term}'",
                c.first_mention,
            )
            for c in concepts.concepts
        ]
    )


def claims_from_flashcards(deck: FlashcardDeck) -> list[Claim]:
    """A card is a claim that its answer is the right answer to its question."""
    return _number(
        [
            (
                f"{c.question} Answer: {c.answer}",
                "flashcard",
                f"card {i}",
                c.source_time,
            )
            for i, c in enumerate(deck.cards, start=1)
        ]
    )


def claims_from_quiz(quiz: Quiz) -> list[Claim]:
    """A question plus its correct option. Distractors are meant to be false."""
    return _number(
        [
            (
                f"{q.question} Answer: {q.options[q.answer]}",
                "quiz",
                f"question {i}",
                q.source_time,
            )
            for i, q in enumerate(quiz.questions, start=1)
        ]
    )


@dataclass(frozen=True)
class Target:
    """A loaded file to evaluate: its kind, claims and who generated it."""

    kind: TargetKind
    title: str
    claims: list[Claim]
    model: str
    provider: str


def load_target(path: str | Path) -> Target:
    """Read any Phase 3 output file and extract its claims.

    The kind is detected from the top-level keys, so the user just points at
    a file. Each file is re-validated by its own model on the way in, so a
    corrupted file fails here with a clear message.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a notes/concepts/flashcards/quiz file")
    try:
        if "sections" in data:
            notes = Notes.from_dict(data)
            return Target(
                "notes",
                notes.title,
                claims_from_notes(notes),
                notes.model,
                notes.provider,
            )
        if "concepts" in data:
            cl = ConceptList.model_validate(data)
            return Target(
                "concepts", cl.title, claims_from_concepts(cl), cl.model, cl.provider
            )
        if "cards" in data:
            deck = FlashcardDeck.model_validate(data)
            return Target(
                "flashcards",
                deck.title,
                claims_from_flashcards(deck),
                deck.model,
                deck.provider,
            )
        if "questions" in data:
            quiz = Quiz.model_validate(data)
            return Target(
                "quiz", quiz.title, claims_from_quiz(quiz), quiz.model, quiz.provider
            )
    except (ValidationError, TypeError, KeyError) as error:
        # TypeError/KeyError come from Notes.from_dict (a plain dataclass).
        raise ValueError(f"{path} could not be read: {error}") from None
    raise ValueError(
        f"{path} is not a notes, concepts, flashcards or quiz file "
        "(expected a 'sections', 'concepts', 'cards' or 'questions' key)"
    )


def sample_claims(claims: list[Claim], limit: int | None) -> list[Claim]:
    """Keep at most `limit` claims, spread evenly over the lecture.

    Taking the first N would only check the first few minutes. Instead we
    take every (n / limit)-th claim, a deterministic "systematic sample", so
    reruns check the same claims and the whole lecture is covered.
    """
    if limit is None or limit >= len(claims):
        return claims
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    step = len(claims) / limit
    return [claims[int(i * step)] for i in range(limit)]


# ---------------------------------------------------------------------------
# Evidence retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Passage:
    """One transcript paragraph, the unit of evidence."""

    index: int
    start: float
    text: str
    words: frozenset[str]

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def transcript_passages(transcript: Transcript) -> list[Passage]:
    """Pause-based paragraphs (the same ones shown in transcript.md)."""
    passages = []
    for para in group_paragraphs(transcript.segments):
        text = " ".join(seg.text.strip() for seg in para)
        passages.append(
            Passage(len(passages), para[0].start, text, frozenset(content_words(text)))
        )
    return passages


class Retriever:
    """Ranks passages for a claim by IDF-weighted word overlap.

    IDF ("inverse document frequency") of a word w is

        idf(w) = log((N + 1) / (df(w) + 1)) + 1

    where N is the number of passages and df(w) how many contain w. A word in
    every passage gets the minimum weight 1; a word in a single passage gets
    about log(N/2) + 1. The +1 terms ("smoothing") avoid dividing by zero and
    keep every weight positive. A passage's score for a claim is the sum of
    idf over the content words they share.
    """

    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        n = len(passages)
        df = Counter(w for p in passages for w in p.words)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}

    def score(self, claim_words: set[str], passage: Passage) -> float:
        return sum(self.idf[w] for w in claim_words & passage.words)

    def top_k(self, claim: Claim, k: int = DEFAULT_TOP_K) -> list[int]:
        """Indices of the best `k` passages (best first).

        Ties go to the passage closest to the time the generator linked the
        claim to. If the claim has a time, the passage playing at that moment
        is always added as an extra candidate: summaries often paraphrase so
        freely that they share few words with their own source.
        """
        if not self.passages:
            return []
        words = set(content_words(claim.text))
        time = claim.time

        def key(p: Passage) -> tuple[float, float]:
            distance = abs(p.start - time) if time is not None else p.index
            # Sorting ascending, so negate the score to put the best first.
            return (-self.score(words, p), distance)

        ranked = [p for p in sorted(self.passages, key=key) if self.score(words, p) > 0]
        chosen = [p.index for p in ranked[:k]]
        if time is not None:
            playing = self.passage_at(time)
            if playing not in chosen:
                chosen.append(playing)
        return chosen

    def passage_at(self, time: float) -> int:
        """Index of the last passage starting at or before `time`."""
        index = 0
        for p in self.passages:
            if p.start <= time + 1:  # +1 s: timestamps in outputs are floored
                index = p.index
        return index


def lexical_support(claim: Claim, evidence: list[Passage]) -> float:
    """Share of the claim's distinct content words found in its evidence."""
    words = set(content_words(claim.text))
    if not words:
        return 0.0
    found = set().union(*(p.words for p in evidence)) if evidence else set()
    return len(words & found) / len(words)


def lexical_verdict(score: float) -> Verdict:
    if score >= LEXICAL_SUPPORTED:
        return "supported"
    if score >= LEXICAL_PARTIAL:
        return "partial"
    return "unsupported"


def select_evidence(
    ranked: list[list[int]], passages: list[Passage], budget: int
) -> list[int]:
    """Merge the ranked lists of a batch into one evidence set under a budget.

    Round robin by rank: first every claim's best passage, then every claim's
    second best, and so on, stopping additions once the word budget is used.
    Best passages are always kept, so no claim is left without evidence.
    Returned in lecture order, which reads more naturally for the judge.
    """
    chosen: list[int] = []
    words = 0
    depth = max((len(r) for r in ranked), default=0)
    for rank in range(depth):
        for indices in ranked:
            if rank >= len(indices) or indices[rank] in chosen:
                continue
            size = passages[indices[rank]].word_count
            if rank == 0 or words + size <= budget:
                chosen.append(indices[rank])
                words += size
    return sorted(chosen)


# ---------------------------------------------------------------------------
# Validating the judge's reply
# ---------------------------------------------------------------------------

# Words small models use for our four verdicts.
_VERDICT_ALIASES = {
    "supported": "supported",
    "support": "supported",
    "yes": "supported",
    "true": "supported",
    "faithful": "supported",
    "entailed": "supported",
    "entailment": "supported",
    "partial": "partial",
    "partially supported": "partial",
    "partially": "partial",
    "partly supported": "partial",
    "unsupported": "unsupported",
    "not supported": "unsupported",
    "no": "unsupported",
    "not mentioned": "unsupported",
    "neutral": "unsupported",
    "hallucinated": "unsupported",
    "contradicted": "contradicted",
    "contradiction": "contradicted",
    "contradicts": "contradicted",
    "false": "contradicted",
    "refuted": "contradicted",
}


class VerdictSchema(BaseModel):
    """One verdict as the judge must return it."""

    claim: int
    verdict: Literal["supported", "partial", "unsupported", "contradicted"]
    quote: str = ""
    reason: str = ""

    @field_validator("claim", mode="before")
    @classmethod
    def _claim_number(cls, value: Any) -> Any:
        # Accept "3", "#3" or "claim 3" as well as 3.
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            return int(digits) if digits else value
        return value

    @field_validator("verdict", mode="before")
    @classmethod
    def _verdict_word(cls, value: Any) -> Any:
        if isinstance(value, str):
            word = value.strip().strip(".\"'").lower().replace("_", " ")
            return _VERDICT_ALIASES.get(word, word)
        return value

    @field_validator("quote", "reason", mode="before")
    @classmethod
    def _text(cls, value: Any) -> Any:
        if value is None:
            return ""
        return value.strip().strip("\"'") if isinstance(value, str) else value


class JudgeReplySchema(BaseModel):
    verdicts: list[VerdictSchema]


_FIELD_ALIASES = {
    "id": "claim",
    "statement": "claim",
    "claim_id": "claim",
    "label": "verdict",
    "evidence": "quote",
    "explanation": "reason",
    "rationale": "reason",
}


def validate_judge_reply(text: str, count: int) -> dict[int, VerdictSchema]:
    """Parse one reply into {claim number: verdict}; raise ValueError if bad.

    Every number 1..count must appear. Numbers outside that range are an
    error (the model is confused about which statement is which); a repeated
    number keeps its first verdict.
    """
    data = LLMResponse(text=text, model="", provider="").json()
    if isinstance(data, list):
        data = {"verdicts": data}
    elif isinstance(data, dict) and "verdicts" not in data:
        for other in ("results", "claims", "judgments", "statements"):
            if other in data:
                data = {"verdicts": data[other]}
                break
    if isinstance(data, dict) and isinstance(data.get("verdicts"), list):
        data = {
            "verdicts": [
                {
                    _FIELD_ALIASES.get(str(k).lower(), str(k).lower()): v
                    for k, v in item.items()
                }
                if isinstance(item, dict)
                else item
                for item in data["verdicts"]
            ]
        }
    try:
        parsed = JudgeReplySchema.model_validate(data)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in error.errors()[:5]
        )
        raise ValueError(f"wrong JSON shape ({problems})") from None
    verdicts: dict[int, VerdictSchema] = {}
    for v in parsed.verdicts:
        if not 1 <= v.claim <= count:
            raise ValueError(
                f"statement number {v.claim} does not exist (1 to {count})"
            )
        verdicts.setdefault(v.claim, v)
    missing = [i for i in range(1, count + 1) if i not in verdicts]
    if missing:
        raise ValueError(f"no verdict for statement(s) {missing}")
    return verdicts


def quote_found(quote: str, passages: list[Passage]) -> bool:
    """True if the quote (nearly) occurs word for word in one passage.

    Both sides are normalized to lowercase word lists, so punctuation and
    case do not matter. `SequenceMatcher.get_matching_blocks` lines the two
    word lists up and returns the runs of words they share (like `diff`
    does for lines). We accept the quote if

    * the shared runs cover at least QUOTE_MATCH_RATIO of the quote's words,
      which forgives a changed "a"/"the" or a fixed typo, and
    * they sit inside a stretch of the passage at most 1.5x the quote's
      length, so a "quote" stitched together from far-apart words is
      rejected.

    An invented quote fails the first test.
    """
    q = list(term_words(quote))
    if not q:
        return False
    for p in passages:
        words = list(term_words(p.text))
        matcher = SequenceMatcher(None, q, words, autojunk=False)
        # The last block is always a size-0 sentinel; drop empty blocks.
        blocks = [b for b in matcher.get_matching_blocks() if b.size]
        if not blocks:
            continue
        matched = sum(b.size for b in blocks)
        span = blocks[-1].b + blocks[-1].size - blocks[0].b
        if matched / len(q) >= QUOTE_MATCH_RATIO and span <= 1.5 * len(q) + 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class JudgedClaim(BaseModel):
    """A claim with its evidence and verdict."""

    id: int
    text: str
    kind: ClaimKind
    source: str
    time: float | None = None
    verdict: Verdict
    quote: str = ""
    quote_verified: bool | None = Field(
        default=None,
        description="Quote found in the transcript (None: none was needed)",
    )
    reason: str = ""
    evidence_times: list[float] = Field(
        default_factory=list, description="Start (s) of the passages the judge saw"
    )
    lexical_support: float = Field(ge=0, le=1)


class Summary(BaseModel):
    """Aggregate metrics over all claims."""

    claims: int = Field(ge=0)
    judged: int = Field(ge=0)
    counts: dict[str, int]
    faithfulness: float | None = Field(
        description="(supported + 0.5 * partial) / judged"
    )
    strict_faithfulness: float | None = Field(
        description="As above, but 'supported' without a verified quote counts 0.5"
    )
    hallucination_rate: float | None = Field(
        description="(unsupported + contradicted) / judged"
    )
    quote_verified_rate: float | None = Field(
        description="Share of verdicts needing a quote whose quote is in the transcript"
    )
    lexical_agreement: float | None = Field(
        default=None, description="Judge vs lexical baseline, both made yes/no"
    )
    lexical_kappa: float | None = Field(
        default=None, description="Cohen's kappa of the same comparison"
    )
    by_kind: dict[str, float | None] = Field(
        default_factory=dict, description="Faithfulness per claim kind"
    )


class FaithfulnessReport(BaseModel):
    """The whole faithfulness_<kind>.json file."""

    title: str
    target: str = Field(description="File that was evaluated")
    target_kind: TargetKind
    generator_model: str
    generator_provider: str
    judge_model: str
    judge_provider: str
    summary: Summary
    claims: list[JudgedClaim]
    total_claims: int = Field(ge=0, description="Claims in the file before sampling")
    llm_calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_s: float = Field(default=0.0, ge=0)
    batch_errors: list[str] = Field(default_factory=list)

    @property
    def judge_is_generator(self) -> bool:
        return (self.judge_provider, self.judge_model) == (
            self.generator_provider,
            self.generator_model,
        )

    def to_markdown(self) -> str:
        s = self.summary

        def pct(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.1%}"

        lines = [f"# Faithfulness: {self.title}", ""]
        lines.append(
            f"_{self.target_kind} from `{self.target}` "
            f"(generated by {self.generator_provider} · {self.generator_model}), "
            f"judged by {self.judge_provider} · {self.judge_model}. "
            f"{s.claims} of {self.total_claims} claim(s) checked._"
        )
        lines.append("")
        if self.judge_is_generator:
            lines += [
                "> Warning: the judge is the model that wrote this material. "
                "LLM judges tend to favour their own writing, so treat the "
                "score as optimistic.",
                "",
            ]
        lines += [
            "| Metric | Value |",
            "|---|---|",
            f"| Faithfulness | {pct(s.faithfulness)} |",
            f"| Strict faithfulness (quote verified) | {pct(s.strict_faithfulness)} |",
            f"| Hallucination rate | {pct(s.hallucination_rate)} |",
            f"| Judge quotes found in transcript | {pct(s.quote_verified_rate)} |",
        ]
        if s.lexical_kappa is not None or s.lexical_agreement is not None:
            kappa = "n/a" if s.lexical_kappa is None else f"{s.lexical_kappa:.2f}"
            lines.append(
                f"| Agreement with lexical baseline | {pct(s.lexical_agreement)} "
                f"(kappa {kappa}) |"
            )
        lines += ["", "| Verdict | Claims |", "|---|---|"]
        lines += [
            f"| {v} | {s.counts.get(v, 0)} |" for v in (*JUDGED_VERDICTS, "unjudged")
        ]
        if len(s.by_kind) > 1:
            lines += ["", "| Claim kind | Faithfulness |", "|---|---|"]
            lines += [f"| {k} | {pct(v)} |" for k, v in s.by_kind.items()]
        lines.append("")

        flagged = [
            c
            for c in self.claims
            if c.verdict != "supported" or c.quote_verified is False
        ]
        lines += ["## Claims to review", ""]
        if not flagged:
            lines += ["_None: every claim is supported with a verified quote._", ""]
        for c in flagged:
            when = "" if c.time is None else f" `[{format_timestamp(c.time)}]`"
            label = c.verdict.upper()
            if c.verdict == "supported":
                label += " (quote not found)"
            lines.append(f"- **{label}** {c.source}{when}: {c.text}")
            if c.reason:
                lines.append(f"  - Judge: {c.reason}")
            if c.quote:
                mark = "" if c.quote_verified else " (not found in transcript)"
                lines.append(f'  - Quote: "{c.quote}"{mark}')
            if c.evidence_times:
                times = ", ".join(format_timestamp(t) for t in c.evidence_times)
                lines.append(f"  - Evidence checked: {times}")
        if self.batch_errors:
            lines += ["", "## Judge errors", ""]
            lines += [f"- {e}" for e in self.batch_errors]
        lines.append("")
        return "\n".join(lines)


def faithfulness_json_schema() -> dict[str, Any]:
    """The faithfulness report format as a standard JSON Schema document."""
    return FaithfulnessReport.model_json_schema()


def _score(claims: list[JudgedClaim], strict: bool = False) -> float | None:
    judged = [c for c in claims if c.verdict in JUDGED_VERDICTS]
    if not judged:
        return None
    points = 0.0
    for c in judged:
        if c.verdict == "supported":
            points += 0.5 if strict and c.quote_verified is False else 1.0
        elif c.verdict == "partial":
            points += 0.5
    return points / len(judged)


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Agreement between two yes/no raters, corrected for chance.

        kappa = (p_o - p_e) / (1 - p_e)

    p_o is the observed share of items where they agree. p_e is the agreement
    expected if both answered at random with their own yes-rates:
    p_e = P(a yes) P(b yes) + P(a no) P(b no). kappa = 1 is perfect
    agreement, 0 is no better than chance. Undefined (None) when p_e = 1,
    i.e. both raters always give the same single answer.
    """
    n = len(a)
    if n == 0 or n != len(b):
        return None
    p_o = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    p_e = pa * pb + (1 - pa) * (1 - pb)
    if math.isclose(p_e, 1.0):
        return None
    return (p_o - p_e) / (1 - p_e)


def summarize(claims: list[JudgedClaim], compare_lexical: bool) -> Summary:
    counts = Counter(c.verdict for c in claims)
    judged = [c for c in claims if c.verdict in JUDGED_VERDICTS]
    quotes = [c.quote_verified for c in judged if c.quote_verified is not None]
    agreement = kappa = None
    if compare_lexical and judged:
        # Collapse to yes/no: supported or partial counts as "backed".
        judge = [c.verdict in ("supported", "partial") for c in judged]
        lexical = [lexical_verdict(c.lexical_support) != "unsupported" for c in judged]
        agreement = sum(x == y for x, y in zip(judge, lexical, strict=True)) / len(
            judged
        )
        kappa = cohen_kappa(judge, lexical)
    kinds = list(dict.fromkeys(c.kind for c in claims))  # unique, in order
    return Summary(
        claims=len(claims),
        judged=len(judged),
        counts={v: counts.get(v, 0) for v in (*JUDGED_VERDICTS, "unjudged")},
        faithfulness=_score(claims),
        strict_faithfulness=_score(claims, strict=True),
        hallucination_rate=(
            (counts["unsupported"] + counts["contradicted"]) / len(judged)
            if judged
            else None
        ),
        quote_verified_rate=sum(quotes) / len(quotes) if quotes else None,
        lexical_agreement=agreement,
        lexical_kappa=kappa,
        by_kind={k: _score([c for c in claims if c.kind == k]) for k in kinds},
    )


# ---------------------------------------------------------------------------
# Running the judge
# ---------------------------------------------------------------------------


def build_prompt(claims: list[Claim], evidence: list[Passage]) -> str:
    """The judge prompt for one batch. Claims are renumbered 1..n."""
    evidence_text = "\n".join(
        f"[P{p.index + 1}] [{format_timestamp(p.start)}] {p.text}" for p in evidence
    )
    claims_text = "\n".join(f"{i}. {c.text}" for i, c in enumerate(claims, start=1))
    return USER_PROMPT_TEMPLATE.format(evidence=evidence_text, claims=claims_text)


def judge_batch(
    client: LLMClient,
    claims: list[Claim],
    evidence: list[Passage],
    *,
    max_repairs: int = 1,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
) -> tuple[dict[int, VerdictSchema] | None, str | None, list[LLMResponse]]:
    """Ask the judge about one batch, repairing invalid replies.

    Returns (verdicts keyed 1..n, None, responses) on success, or
    (None, error message, responses) if no valid reply came back. A failed
    batch does not stop the eval: its claims are marked "unjudged", because
    one bad reply should not throw away an hour of free-tier calls.
    """
    if max_repairs < 0:
        raise ValueError(f"max_repairs must be >= 0, got {max_repairs}")
    messages = [
        Message("system", SYSTEM_PROMPT),
        Message("user", build_prompt(claims, evidence)),
    ]
    responses: list[LLMResponse] = []
    error: ValueError | None = None
    for _ in range(max_repairs + 1):
        # temperature=0: the judge should give the same verdict every run.
        reply = client.chat(
            messages, json_mode=True, max_tokens=max_tokens, temperature=0.0
        )
        responses.append(reply)
        try:
            return validate_judge_reply(reply.text, len(claims)), None, responses
        except ValueError as exc:
            error = exc
            messages = [
                *messages,
                Message("assistant", reply.text),
                Message("user", REPAIR_PROMPT.format(error=exc, count=len(claims))),
            ]
    return (
        None,
        f"no valid verdicts after {max_repairs + 1} attempt(s): {error}",
        responses,
    )


def evaluate_faithfulness(
    target: Target,
    transcript: Transcript,
    client: LLMClient | None,
    *,
    target_path: str = "",
    max_claims: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    top_k: int = DEFAULT_TOP_K,
    evidence_words: int = DEFAULT_EVIDENCE_WORDS,
    max_repairs: int = 1,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    on_batch: Callable[[int, int], None] | None = None,
) -> FaithfulnessReport:
    """Full eval. `client=None` gives the free lexical-only baseline."""
    if batch_size <= 0 or top_k <= 0:
        raise ValueError("batch_size and top_k must be positive")
    claims = sample_claims(target.claims, max_claims)
    passages = transcript_passages(transcript)
    retriever = Retriever(passages)
    ranked = {c.id: retriever.top_k(c, top_k) for c in claims}

    judged: list[JudgedClaim] = []
    responses: list[LLMResponse] = []
    errors: list[str] = []
    batches = [claims[i : i + batch_size] for i in range(0, len(claims), batch_size)]
    for b_i, batch in enumerate(batches):
        if on_batch is not None and client is not None:
            on_batch(b_i, len(batches))
        evidence_ids = select_evidence(
            [ranked[c.id] for c in batch], passages, evidence_words
        )
        evidence = [passages[i] for i in evidence_ids]
        verdicts: dict[int, VerdictSchema] | None = None
        if client is not None and evidence:
            verdicts, error, replies = judge_batch(
                client, batch, evidence, max_repairs=max_repairs, max_tokens=max_tokens
            )
            responses.extend(replies)
            if error is not None:
                errors.append(f"batch {b_i + 1}/{len(batches)}: {error}")
        elif client is not None:
            errors.append(f"batch {b_i + 1}/{len(batches)}: transcript is empty")

        for n, claim in enumerate(batch, start=1):
            own = [passages[i] for i in ranked[claim.id]]
            lex = lexical_support(claim, own)
            record: dict[str, Any] = {
                **claim.model_dump(),
                "lexical_support": lex,
                "evidence_times": [p.start for p in evidence],
            }
            if client is None:
                record["verdict"] = lexical_verdict(lex)
                record["evidence_times"] = [p.start for p in own]
            elif verdicts is None:
                record["verdict"] = "unjudged"
            else:
                v = verdicts[n]
                record.update(verdict=v.verdict, quote=v.quote, reason=v.reason)
                # Every verdict except "unsupported" must show a quote; a
                # missing one counts as unverified, not as "no opinion".
                if v.quote:
                    record["quote_verified"] = quote_found(v.quote, evidence)
                elif v.verdict != "unsupported":
                    record["quote_verified"] = False
            judged.append(JudgedClaim(**record))

    return FaithfulnessReport(
        title=target.title,
        target=target_path,
        target_kind=target.kind,
        generator_model=target.model,
        generator_provider=target.provider,
        judge_model=getattr(client, "model", "unknown")
        if client
        else "lexical-overlap",
        judge_provider=getattr(client, "provider", "unknown") if client else "none",
        summary=summarize(judged, compare_lexical=client is not None),
        claims=judged,
        total_claims=len(target.claims),
        llm_calls=len(responses),
        prompt_tokens=sum(r.prompt_tokens or 0 for r in responses),
        completion_tokens=sum(r.completion_tokens or 0 for r in responses),
        latency_s=sum(r.latency_s for r in responses),
        batch_errors=errors,
    )


def write_report(report: FaithfulnessReport, out_dir: str | Path) -> tuple[Path, Path]:
    """Write faithfulness_<kind>.json and .md; the kind keeps reports apart."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"faithfulness_{report.target_kind}"
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return json_path, md_path


def load_report(path: str | Path) -> FaithfulnessReport:
    """Read a report and validate it; raises ValueError if it is invalid."""
    try:
        return FaithfulnessReport.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except ValidationError as error:
        raise ValueError(
            f"{path} is not a valid faithfulness report: {error}"
        ) from None
