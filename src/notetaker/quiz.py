"""Generate multiple-choice quizzes (one correct answer plus distractors).

Pipeline:

    transcript.json
        -> chunks                    (same chunker as notes.py)
        -> one LLM call per chunk    ("write quiz questions about this part")
        -> validated replies         (pydantic schema + repair loop)
        -> grounding check           (the correct answer's words must be in
                                      the chunk, same test as flashcards.py)
        -> shuffled options          (deterministic, so the answer is not
                                      always option A)
        -> source timestamp          (the paragraph the question is about)
    concepts.json (optional, Step 3.4)
        -> "Which term matches this definition?" questions whose wrong
           options are other glossary terms; no LLM call needed
    both
        -> de-duplication            (same question twice -> one question)
        -> quiz.json + quiz.md       (questions first, answer key at the end)

Vocabulary: in a multiple-choice question the "stem" is the question, the
"key" is the correct option and the "distractors" are the wrong options. Good
distractors are plausible to someone who did not follow the lecture but
clearly wrong to someone who did.

Why the model returns the key and the distractors separately, instead of a
list of options plus the index of the right one:

* Index mistakes. Small models often get the index wrong (off by one, or
  pointing at a distractor). With separate fields there is no index to get
  wrong; we build the option list ourselves.
* Position bias. Models put the correct answer first far more often than
  1 in 4. If we kept their order, a student could score well by always
  answering A. We shuffle in code instead.

Two cheap quality signals are computed without a second model:

* Grounding (from flashcards.py): at least half of the key's content words
  must occur in the chunk, otherwise the question is dropped.
* Length cue: test-wise students know that the longest option is often the
  right one, because writers add qualifiers to make the key fully correct.
  We record the share of questions where the key is strictly the longest
  option. With 4 options a fair quiz should be near 25%; a much higher value
  means the distractors are too short and the prompt needs work.
"""

from __future__ import annotations

import difflib
import random
import re
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from notetaker.concepts import ConceptList, term_words
from notetaker.export import format_timestamp
from notetaker.flashcards import (
    DUPLICATE_RATIO,
    MIN_SUPPORT,
    best_paragraph_time,
    content_words,
    support,
)
from notetaker.llm import LLMClient, LLMResponse, Message
from notetaker.notes import DEFAULT_CHUNK_WORDS, TranscriptChunk, chunk_transcript
from notetaker.transcribe import Transcript

DEFAULT_MAX_TOKENS = 1500

# A standard multiple-choice item: 1 key + 3 distractors = 4 options (A-D).
NUM_DISTRACTORS = 3
# We accept 2 distractors (3 options) from a model that runs out of ideas,
# because a 3-option question is still a fair test; fewer is a true/false.
MIN_DISTRACTORS = 2

MAX_QUESTION_WORDS = 50
MAX_OPTION_WORDS = 30

# Options like these break when the options are shuffled ("all of the above"
# must be last to make sense), so they are rejected and the model is asked
# again.
_BANNED_OPTIONS = re.compile(r"\b(all|none|both|neither) of (the )?(above|these)\b")

LETTERS = "ABCDEFGH"

SYSTEM_PROMPT = """\
You are a careful teaching assistant who writes multiple-choice quiz \
questions from lecture transcripts. The correct answer must come from what \
the lecturer says; never add outside knowledge. Answer with JSON only."""

USER_PROMPT_TEMPLATE = """\
Below is part {part} of {total} of a lecture transcript.

Write multiple-choice questions that test understanding of THIS PART, as a \
JSON object of this shape:

{{
  "questions": [
    {{
      "question": "a specific question",
      "correct": "the correct answer",
      "distractors": ["wrong answer 1", "wrong answer 2", "wrong answer 3"],
      "explanation": "one sentence on why the correct answer is right"
    }}
  ]
}}

Rules:
- 0 to {max_questions} questions; an empty list is fine if the part has no \
content worth testing (for example course logistics).
- Exactly {n_distractors} distractors per question. Distractors must be \
plausible (common misconceptions, related but different concepts) but \
clearly wrong given the lecture, and about as long as the correct answer.
- No "all of the above", "none of the above" or "both" options, and no two \
options that mean the same thing.
- The question must make sense on its own, without the transcript (name the \
concept, do not say "the lecturer" or "this part").
- Questions at most {max_q} words, each option at most {max_o} words. Use \
the lecturer's own wording for the correct answer where possible.

Transcript part {part} of {total}:
{transcript}"""

REPAIR_PROMPT = """\
Your previous reply could not be used: {error}
Return only the corrected JSON object with a "questions" list as described."""


# ---------------------------------------------------------------------------
# Schemas: what a model reply must look like
# ---------------------------------------------------------------------------


def _strip_text(value: Any) -> Any:
    return value.strip().strip("\"'*`") if isinstance(value, str) else value


def option_key(text: str) -> tuple[str, ...]:
    """Normalized form of an option, used to spot two options that are equal.

    "The Gradient." and "the gradient" are the same option to a student, so
    we compare lowercased, punctuation-free, plural-folded words.
    """
    return tuple(term_words(text))


class QuestionSchema(BaseModel):
    """One question as the model must return it."""

    question: str = Field(min_length=1)
    correct: str = Field(min_length=1)
    distractors: list[str] = Field(min_length=MIN_DISTRACTORS)
    explanation: str = ""

    # mode="before": clean raw values before pydantic checks their types.
    @field_validator("question", "correct", "explanation", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return _strip_text(value)

    @field_validator("distractors", mode="before")
    @classmethod
    def _strip_list(cls, value: Any) -> Any:
        if isinstance(value, list):
            # Drop empty strings: some models pad the list with "".
            return [v for v in (_strip_text(x) for x in value) if v != ""]
        return value

    @field_validator("question")
    @classmethod
    def _short_question(cls, value: str) -> str:
        if len(value.split()) > MAX_QUESTION_WORDS:
            raise ValueError(f"question must be at most {MAX_QUESTION_WORDS} words")
        return value

    # mode="after" runs once all fields are validated, so it can compare them.
    @model_validator(mode="after")
    def _check_options(self) -> QuestionSchema:
        options = [self.correct, *self.distractors]
        for option in options:
            if len(option.split()) > MAX_OPTION_WORDS:
                raise ValueError(
                    f"each option must be at most {MAX_OPTION_WORDS} words"
                )
            if _BANNED_OPTIONS.search(option.lower()):
                raise ValueError(f"option {option!r} does not survive shuffling")
        keys = [option_key(o) for o in options]
        if len(set(keys)) != len(keys):
            raise ValueError("options must all be different")
        return self


class ChunkQuizSchema(BaseModel):
    """The whole reply for one chunk. An empty list is allowed."""

    questions: list[QuestionSchema]


class QuizError(RuntimeError):
    """The model never produced a usable question list for a chunk."""


def _short_errors(error: ValidationError) -> str:
    """Pydantic's multi-line error as one line for the repair prompt."""
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'question'}: {e['msg']}"
        for e in error.errors()[:5]
    )


# Small models use many spellings for the fields; map them to ours.
_KEY_ALIASES = {
    "q": "question",
    "stem": "question",
    "answer": "correct",
    "correct_answer": "correct",
    "key": "correct",
    "wrong": "distractors",
    "wrong_answers": "distractors",
    "incorrect": "distractors",
    "incorrect_answers": "distractors",
    "rationale": "explanation",
}


def validate_quiz_reply(text: str) -> ChunkQuizSchema:
    """Parse and validate one reply; raise ValueError with a readable reason.

    Accepted besides the requested shape: a bare list of questions, the list
    under "quiz" or "items", and common key spellings (answer, wrong_answers).
    """
    data = LLMResponse(text=text, model="", provider="").json()
    if isinstance(data, list):
        data = {"questions": data}
    elif isinstance(data, dict) and "questions" not in data:
        for other in ("quiz", "items", "mcqs"):
            if other in data:
                data = {"questions": data[other]}
                break
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        data = {
            "questions": [
                {
                    _KEY_ALIASES.get(str(k).lower(), str(k).lower()): v
                    for k, v in q.items()
                }
                if isinstance(q, dict)
                else q
                for q in data["questions"]
            ]
        }
    try:
        return ChunkQuizSchema.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"wrong JSON shape ({_short_errors(error)})") from None


# ---------------------------------------------------------------------------
# Output schema: the quiz.json file
# ---------------------------------------------------------------------------


class QuizQuestion(BaseModel):
    """One question in quiz.json, options already shuffled."""

    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=MIN_DISTRACTORS + 1)
    answer: int = Field(ge=0, description="Index of the correct option (0 = A)")
    explanation: str = ""
    source_time: float = Field(ge=0, description="Seconds into the lecture")
    origin: Literal["llm", "concept"] = Field(
        description="'llm' = generated from a chunk, 'concept' = from concepts.json"
    )
    chunk: int | None = Field(
        default=None, description="Chunk the question was generated from"
    )
    support: float = Field(
        ge=0, le=1, description="Share of the key's content words found in the source"
    )

    @model_validator(mode="after")
    def _answer_in_range(self) -> QuizQuestion:
        if self.answer >= len(self.options):
            raise ValueError(
                f"answer index {self.answer} but only {len(self.options)} options"
            )
        return self

    @property
    def answer_letter(self) -> str:
        return LETTERS[self.answer]


class Quiz(BaseModel):
    """The whole quiz.json file: questions plus how they were made."""

    title: str
    questions: list[QuizQuestion]
    model: str
    provider: str
    chunks: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_s: float = Field(default=0.0, ge=0)
    seed: int = Field(default=0, description="Seed used to shuffle the options")
    # Questions whose correct answer was not supported by the transcript.
    dropped_ungrounded: list[str] = Field(default_factory=list)
    dropped_duplicates: int = Field(default=0, ge=0)

    def answer_positions(self) -> dict[str, int]:
        """How often the key sits at A, B, C, ... (should be roughly even)."""
        width = max((len(q.options) for q in self.questions), default=0)
        counts = {LETTERS[i]: 0 for i in range(width)}
        for q in self.questions:
            counts[q.answer_letter] += 1
        return counts

    def longest_option_rate(self) -> float | None:
        """Share of questions whose key is strictly the longest option.

        Measured in words. None for an empty quiz. See the module docstring:
        near 1/len(options) is fair, much higher leaks the answer.
        """
        if not self.questions:
            return None
        hits = 0
        for q in self.questions:
            lengths = [len(o.split()) for o in q.options]
            key_len = lengths[q.answer]
            if all(key_len > n for i, n in enumerate(lengths) if i != q.answer):
                hits += 1
        return hits / len(self.questions)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        lines.append(
            f"_{len(self.questions)} question(s). The answer key is at the end._"
        )
        lines.append("")
        if not self.questions:
            lines += ["_No questions._", ""]
        for i, q in enumerate(self.questions, start=1):
            lines.append(f"**{i}.** {q.question}")
            lines.append("")
            lines += [f"- {LETTERS[j]}. {opt}" for j, opt in enumerate(q.options)]
            lines.append("")
        if self.questions:
            lines += ["## Answer key", ""]
            for i, q in enumerate(self.questions, start=1):
                why = f" {q.explanation}" if q.explanation else ""
                lines.append(
                    f"{i}. **{q.answer_letter}**.{why} "
                    f"`[{format_timestamp(q.source_time)}]`"
                )
            lines.append("")
        return "\n".join(lines)


def quiz_json_schema() -> dict[str, Any]:
    """The quiz.json format as a standard JSON Schema document."""
    return Quiz.model_json_schema()


# ---------------------------------------------------------------------------
# Shuffling
# ---------------------------------------------------------------------------


def shuffle_options(
    correct: str, distractors: list[str], *, seed: int, salt: str
) -> tuple[list[str], int]:
    """Put the key among the distractors at a random position.

    Returns (options, index of the key). The randomness is seeded so the
    same input always gives the same quiz: rerunning the command does not
    reshuffle the answers, and tests can check exact outputs.

    The per-question seed mixes the global `seed` with a checksum of `salt`
    (the question text). We use zlib.crc32 rather than Python's built-in
    hash(), because hash() of a string is randomized every time Python
    starts (a security feature), which would make the order change between
    runs.
    """
    rng = random.Random(seed * 1_000_003 + zlib.crc32(salt.encode("utf-8")))
    options = [correct, *distractors]
    order = list(range(len(options)))
    rng.shuffle(order)
    return [options[i] for i in order], order.index(0)


# ---------------------------------------------------------------------------
# Generation from the transcript
# ---------------------------------------------------------------------------


def generate_chunk_questions(
    client: LLMClient,
    chunk: TranscriptChunk,
    total: int,
    *,
    max_questions: int = 4,
    max_repairs: int = 1,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    min_support: float = MIN_SUPPORT,
    seed: int = 0,
) -> tuple[list[QuizQuestion], list[str], list[LLMResponse]]:
    """Ask for one chunk's questions. Returns (grounded questions, dropped
    question texts, replies)."""
    if max_repairs < 0:
        raise ValueError(f"max_repairs must be >= 0, got {max_repairs}")
    prompt = USER_PROMPT_TEMPLATE.format(
        part=chunk.index + 1,
        total=total,
        max_questions=max_questions,
        n_distractors=NUM_DISTRACTORS,
        max_q=MAX_QUESTION_WORDS,
        max_o=MAX_OPTION_WORDS,
        transcript=chunk.to_prompt_text(),
    )
    messages = [Message("system", SYSTEM_PROMPT), Message("user", prompt)]
    responses: list[LLMResponse] = []
    last_error: ValueError | None = None
    for _ in range(max_repairs + 1):
        reply = client.chat(messages, json_mode=True, max_tokens=max_tokens)
        responses.append(reply)
        try:
            parsed = validate_quiz_reply(reply.text)
        except ValueError as error:
            # Repair loop (same as notes.py): show the model its reply and
            # the exact problem, and ask again.
            messages = [
                *messages,
                Message("assistant", reply.text),
                Message("user", REPAIR_PROMPT.format(error=error)),
            ]
            last_error = error
            continue
        chunk_text = " ".join(text for _, text in chunk.paragraphs)
        questions, dropped = [], []
        # The model may ignore the limits; we enforce them here.
        for item in parsed.questions[:max_questions]:
            score = support(item.correct, chunk_text)
            if score < min_support:
                dropped.append(item.question)
                continue
            options, answer = shuffle_options(
                item.correct,
                item.distractors[:NUM_DISTRACTORS],
                seed=seed,
                salt=item.question,
            )
            questions.append(
                QuizQuestion(
                    question=item.question,
                    options=options,
                    answer=answer,
                    explanation=item.explanation,
                    source_time=best_paragraph_time(
                        f"{item.question} {item.correct}", chunk
                    ),
                    origin="llm",
                    chunk=chunk.index,
                    support=round(score, 3),
                )
            )
        return questions, dropped, responses
    raise QuizError(
        f"chunk {chunk.index + 1}/{total}: no valid quiz questions after "
        f"{max_repairs + 1} attempt(s): {last_error}"
    )


# ---------------------------------------------------------------------------
# Generation from the glossary (no LLM)
# ---------------------------------------------------------------------------


def mask_term(definition: str, names: list[str]) -> str:
    """Replace the term (and its aliases) inside its own definition by "____".

    "SVD is short for singular value decomposition, ..." would give the
    answer away in a "which term is this?" question. Matching is
    case-insensitive and on whole words; longest names first so "singular
    value decomposition" is masked before "singular" could be.
    """
    for name in sorted(names, key=len, reverse=True):
        if not name.strip():
            continue
        pattern = r"\b" + re.escape(name.strip()) + r"s?\b"
        definition = re.sub(pattern, "____", definition, flags=re.IGNORECASE)
    return definition


def concept_questions(
    concepts: ConceptList, *, seed: int = 0, n_distractors: int = NUM_DISTRACTORS
) -> list[QuizQuestion]:
    """'Which term matches this definition?' questions from the glossary.

    The distractors are other terms from the same lecture, which makes them
    plausible for free: they are real concepts the student has heard. We
    pick the terms first mentioned closest in time to the key, because
    concepts introduced together are usually related (eigenvalue vs
    eigenvector) and so are harder to tell apart than random ones.
    Needs at least n_distractors + 1 concepts; otherwise returns [].
    """
    items = concepts.concepts
    if len(items) < n_distractors + 1:
        return []
    questions = []
    for i, c in enumerate(items):
        others = [o for j, o in enumerate(items) if j != i]
        # Sort by time distance; ties broken by list order (sort is stable).
        others.sort(key=lambda o: abs(o.first_mention - c.first_mention))
        distractors = [o.term for o in others[:n_distractors]]
        stem = mask_term(c.definition, [c.term, *c.aliases])
        question = f"Which term matches this definition: {stem}"
        options, answer = shuffle_options(c.term, distractors, seed=seed, salt=question)
        questions.append(
            QuizQuestion(
                question=question,
                options=options,
                answer=answer,
                explanation=f"{c.term}: {c.definition}",
                source_time=c.first_mention,
                origin="concept",
                support=1.0,
            )
        )
    return questions


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def dedupe_questions(
    questions: list[QuizQuestion], ratio: float = DUPLICATE_RATIO
) -> tuple[list[QuizQuestion], int]:
    """Drop questions whose stem repeats an earlier one; return (kept, dropped).

    Same method as `dedupe_cards` in flashcards.py: compare content-word
    lists with difflib's ratio (2 * matching / total). Earlier ones win.
    """
    kept: list[QuizQuestion] = []
    kept_words: list[list[str]] = []
    for q in questions:
        words = content_words(q.question)
        if any(
            difflib.SequenceMatcher(None, words, other).ratio() >= ratio
            for other in kept_words
        ):
            continue
        kept.append(q)
        kept_words.append(words)
    return kept, len(questions) - len(kept)


def generate_quiz(
    transcript: Transcript | None,
    client: LLMClient | None,
    *,
    concepts: ConceptList | None = None,
    title: str = "Quiz",
    max_words: int = DEFAULT_CHUNK_WORDS,
    max_questions: int = 4,
    max_repairs: int = 1,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    min_support: float = MIN_SUPPORT,
    seed: int = 0,
    on_chunk: Callable[[int, int, TranscriptChunk], None] | None = None,
) -> Quiz:
    """Full pipeline. Give a transcript + client for LLM questions, `concepts`
    for glossary questions, or both. With client=None only concept questions
    are made."""
    if client is None and concepts is None:
        raise ValueError("need an LLM client, a concept list, or both")
    llm_questions: list[QuizQuestion] = []
    dropped: list[str] = []
    responses: list[LLMResponse] = []
    chunks: list[TranscriptChunk] = []
    if client is not None:
        if transcript is None:
            raise ValueError("an LLM client needs a transcript to read")
        chunks = chunk_transcript(transcript, max_words=max_words)
        for chunk in chunks:
            if on_chunk is not None:
                on_chunk(chunk.index, len(chunks), chunk)
            qs, bad, replies = generate_chunk_questions(
                client,
                chunk,
                len(chunks),
                max_questions=max_questions,
                max_repairs=max_repairs,
                max_tokens=max_tokens,
                min_support=min_support,
                seed=seed,
            )
            llm_questions.extend(qs)
            dropped.extend(bad)
            responses.extend(replies)

    # LLM questions first here: they test understanding, while glossary
    # questions only test vocabulary, so the LLM version wins a tie. Then
    # sort by time (stable sort) so the quiz follows the lecture.
    glossary = concept_questions(concepts, seed=seed) if concepts else []
    kept, n_duplicates = dedupe_questions(llm_questions + glossary)
    kept.sort(key=lambda q: q.source_time)
    return Quiz(
        title=title,
        questions=kept,
        model=getattr(client, "model", "none") if client else "none",
        provider=getattr(client, "provider", "none") if client else "none",
        chunks=len(chunks),
        llm_calls=len(responses),
        prompt_tokens=sum(r.prompt_tokens or 0 for r in responses),
        completion_tokens=sum(r.completion_tokens or 0 for r in responses),
        latency_s=sum(r.latency_s for r in responses),
        seed=seed,
        dropped_ungrounded=dropped,
        dropped_duplicates=n_duplicates,
    )


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def write_quiz(quiz: Quiz, out_dir: str | Path) -> tuple[Path, Path]:
    """Write quiz.json and quiz.md; return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "quiz.json"
    md_path = out_dir / "quiz.md"
    json_path.write_text(quiz.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(quiz.to_markdown(), encoding="utf-8")
    return json_path, md_path


def load_quiz(path: str | Path) -> Quiz:
    """Read quiz.json and validate it; raises ValueError if invalid."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        return Quiz.model_validate_json(text)
    except ValidationError as error:
        reason = _short_errors(error)
        raise ValueError(f"{path} is not a valid quiz file: {reason}") from None
