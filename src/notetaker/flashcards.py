"""Generate study flashcards (question / answer pairs) and export them to Anki.

Pipeline:

    transcript.json
        -> chunks                    (same chunker as notes.py)
        -> one LLM call per chunk    ("write flashcards about this part")
        -> validated replies         (pydantic schema + repair loop)
        -> grounding check           (the answer's words must be in the chunk)
        -> source timestamp          (the paragraph the card is about)
    concepts.json (optional, Step 3.4)
        -> one "What is X?" card per concept, no LLM call needed
    both
        -> de-duplication            (same question asked twice -> one card)
        -> flashcards.json + flashcards.md + flashcards.csv (Anki import)

Same division of labour as concepts.py: the model only writes the question
and the answer (the language part). Everything checkable is done in code:

* Grounding. A card is only useful if its answer is what the lecturer said.
  We cannot check meaning cheaply, but we can check vocabulary: at least half
  of the answer's content words (not "the", "of", "is"...) must occur in the
  chunk the model read. An answer built from words the lecturer never used
  is most likely outside knowledge or a hallucination, so it is dropped.
  The deeper faithfulness check is Step 3.7 (LLM-as-judge).
* Timestamps. The model is not asked for one. We pick the paragraph of the
  chunk that shares the most content words with the card, so every card can
  link back to a real moment in the recording.
* Duplicates. Neighbouring chunks often produce the same question. Questions
  are compared on their content words (so "What is an eigenvalue?" and
  "What is the eigenvalue?" match) and only the first copy is kept.

Anki import format
    Anki (the free spaced-repetition app) imports plain CSV. Since Anki 2.1.55
    a file can start with "#key:value" header lines that tell the importer the
    separator, the deck, the note type and which column holds tags, so the
    import works with no manual setup. Older Anki versions do not understand
    these lines; `write_anki_csv(..., header=False)` (CLI: --plain-csv) writes
    the bare three-column file instead.
"""

from __future__ import annotations

import csv
import difflib
import io
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from notetaker.concepts import ConceptList, term_words
from notetaker.export import format_timestamp
from notetaker.llm import LLMClient, LLMResponse, Message
from notetaker.notes import DEFAULT_CHUNK_WORDS, TranscriptChunk, chunk_transcript
from notetaker.transcribe import Transcript

DEFAULT_MAX_TOKENS = 1200

# Length limits keep cards atomic: one fact per card is the main rule of
# good spaced-repetition cards, and long answers are hard to grade yourself.
MAX_QUESTION_WORDS = 40
MAX_ANSWER_WORDS = 60

# Fraction of the answer's content words that must occur in the chunk.
MIN_SUPPORT = 0.5

# Two questions whose content words match at least this much are duplicates.
DUPLICATE_RATIO = 0.9

SYSTEM_PROMPT = """\
You are a careful teaching assistant who writes study flashcards from lecture \
transcripts. Every answer must come from what the lecturer says; never add \
outside knowledge. Answer with JSON only."""

USER_PROMPT_TEMPLATE = """\
Below is part {part} of {total} of a lecture transcript.

Write flashcards that test the important ideas of THIS PART, as a JSON object \
of this shape:

{{
  "cards": [
    {{"question": "a specific question", "answer": "a short answer"}}
  ]
}}

Rules:
- 0 to {max_cards} cards; an empty list is fine if the part has no content \
worth testing (for example course logistics).
- One fact per card. The question must make sense on its own, without the \
transcript (name the concept, do not say "the lecturer" or "this part").
- Questions at most {max_q} words, answers at most {max_a} words, using the \
lecturer's own wording where possible.
- Mix "what is", "why" and "how" questions; avoid yes/no questions.

Transcript part {part} of {total}:
{transcript}"""

REPAIR_PROMPT = """\
Your previous reply could not be used: {error}
Return only the corrected JSON object with a "cards" list as described."""


# ---------------------------------------------------------------------------
# Content words: the vocabulary used for grounding, timestamps and dedupe
# ---------------------------------------------------------------------------

# Function words that carry no subject matter. "not" and "no" are left OUT of
# this list on purpose: "Why is the matrix not invertible?" and "Why is the
# matrix invertible?" are different questions and must not be merged.
STOPWORDS = frozenset(
    """
    a an the and or but if then so of to in on at by for with from into onto
    as is are was were be been being am do does did done has have had having
    it its this that these those there here what which who whom whose when
    where why how can could will would should shall may might must we you he
    she they i me us our your their his her them my one also just very
    about over under than more most such some any each other only same lecture
    lecturer called call means mean
    """.split()  # noqa: SIM905 (a word block is easier to read than 100 strings)
)


def content_words(text: str) -> list[str]:
    """Lowercased, plural-folded words of `text` minus stopwords.

    Reuses `term_words` from concepts.py so a card and the glossary normalize
    text the same way. Single letters are dropped unless they are digits
    ("a" is noise, but "2" in "rank 2" is content).
    """
    return [
        w
        for w in term_words(text)
        if w not in STOPWORDS and (len(w) > 1 or w.isdigit())
    ]


def support(answer: str, source_text: str) -> float:
    """Fraction of the answer's distinct content words that occur in the source.

    A set-overlap score in [0, 1]. Distinct words, so repeating a word that is
    in the text cannot inflate the score. An answer with no content words at
    all ("Yes.") scores 0: there is nothing to check, and it is a bad card.
    """
    answer_words = set(content_words(answer))
    if not answer_words:
        return 0.0
    return len(answer_words & set(content_words(source_text))) / len(answer_words)


# ---------------------------------------------------------------------------
# Schemas: what a model reply must look like
# ---------------------------------------------------------------------------


def _strip_text(value: Any) -> Any:
    return value.strip().strip("\"'*`") if isinstance(value, str) else value


class CardSchema(BaseModel):
    """One card as the model must return it."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)

    # mode="before": clean the raw value before pydantic checks it is a str.
    @field_validator("question", "answer", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return _strip_text(value)

    @field_validator("question")
    @classmethod
    def _short_question(cls, value: str) -> str:
        if len(value.split()) > MAX_QUESTION_WORDS:
            raise ValueError(f"question must be at most {MAX_QUESTION_WORDS} words")
        return value

    @field_validator("answer")
    @classmethod
    def _short_answer(cls, value: str) -> str:
        if len(value.split()) > MAX_ANSWER_WORDS:
            raise ValueError(f"answer must be at most {MAX_ANSWER_WORDS} words")
        return value


class ChunkCardsSchema(BaseModel):
    """The whole reply for one chunk. An empty list is allowed."""

    cards: list[CardSchema]


class FlashcardsError(RuntimeError):
    """The model never produced a usable card list for a chunk."""


def _short_errors(error: ValidationError) -> str:
    """Pydantic's multi-line error as one line for the repair prompt."""
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()[:5]
    )


# Small models use many spellings for the two fields; map them to ours.
_KEY_ALIASES = {
    "q": "question",
    "front": "question",
    "prompt": "question",
    "a": "answer",
    "back": "answer",
    "response": "answer",
}


def validate_cards_reply(text: str) -> ChunkCardsSchema:
    """Parse and validate one reply; raise ValueError with a readable reason.

    Accepted besides the requested shape: a bare list of cards, the list under
    "flashcards" instead of "cards", and Q/A, front/back key names.
    """
    data = LLMResponse(text=text, model="", provider="").json()
    if isinstance(data, list):
        data = {"cards": data}
    elif isinstance(data, dict) and "cards" not in data and "flashcards" in data:
        data = {"cards": data["flashcards"]}
    if isinstance(data, dict) and isinstance(data.get("cards"), list):
        data = {
            "cards": [
                {
                    _KEY_ALIASES.get(str(k).lower(), str(k).lower()): v
                    for k, v in c.items()
                }
                if isinstance(c, dict)
                else c
                for c in data["cards"]
            ]
        }
    try:
        return ChunkCardsSchema.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"wrong JSON shape ({_short_errors(error)})") from None


# ---------------------------------------------------------------------------
# Output schema: the flashcards.json file
# ---------------------------------------------------------------------------


class Flashcard(BaseModel):
    """One card in flashcards.json."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    source_time: float = Field(ge=0, description="Seconds into the lecture")
    origin: Literal["llm", "concept"] = Field(
        description="'llm' = generated from a chunk, 'concept' = from concepts.json"
    )
    chunk: int | None = Field(
        default=None, description="Chunk the card was generated from (llm cards)"
    )
    support: float = Field(
        ge=0, le=1, description="Share of answer content words found in the source"
    )


class FlashcardDeck(BaseModel):
    """The whole flashcards.json file: cards plus how they were made."""

    title: str
    cards: list[Flashcard]
    model: str
    provider: str
    chunks: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_s: float = Field(default=0.0, ge=0)
    # Questions whose answers were not supported by the transcript.
    dropped_ungrounded: list[str] = Field(default_factory=list)
    # How many cards were removed as repeats of an earlier card.
    dropped_duplicates: int = Field(default=0, ge=0)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        lines.append(
            f"_{len(self.cards)} flashcard(s). Times link each card to the "
            "moment in the lecture it comes from._"
        )
        lines.append("")
        if not self.cards:
            lines += ["_No flashcards._", ""]
        for i, card in enumerate(self.cards, start=1):
            lines.append(
                f"**Q{i}.** {card.question} `[{format_timestamp(card.source_time)}]`"
            )
            lines.append("")
            lines.append(f"> {card.answer}")
            lines.append("")
        return "\n".join(lines)


def flashcards_json_schema() -> dict[str, Any]:
    """The flashcards.json format as a standard JSON Schema document."""
    return FlashcardDeck.model_json_schema()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def best_paragraph_time(text: str, chunk: TranscriptChunk) -> float:
    """Start time of the chunk paragraph sharing the most content words.

    This is a tiny retrieval problem: score each paragraph by word overlap
    with the card and take the best one (ties go to the earliest paragraph,
    which is usually where an idea is introduced).
    """
    card_words = set(content_words(text))
    best_time, best_score = chunk.start, -1
    for start, paragraph in chunk.paragraphs:
        score = len(card_words & set(content_words(paragraph)))
        if score > best_score:
            best_time, best_score = start, score
    return best_time


def generate_chunk_cards(
    client: LLMClient,
    chunk: TranscriptChunk,
    total: int,
    *,
    max_cards: int = 6,
    max_repairs: int = 1,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    min_support: float = MIN_SUPPORT,
) -> tuple[list[Flashcard], list[str], list[LLMResponse]]:
    """Ask for one chunk's cards. Returns (grounded cards, dropped questions,
    replies)."""
    if max_repairs < 0:
        raise ValueError(f"max_repairs must be >= 0, got {max_repairs}")
    prompt = USER_PROMPT_TEMPLATE.format(
        part=chunk.index + 1,
        total=total,
        max_cards=max_cards,
        max_q=MAX_QUESTION_WORDS,
        max_a=MAX_ANSWER_WORDS,
        transcript=chunk.to_prompt_text(),
    )
    messages = [Message("system", SYSTEM_PROMPT), Message("user", prompt)]
    responses: list[LLMResponse] = []
    last_error: ValueError | None = None
    for _ in range(max_repairs + 1):
        reply = client.chat(messages, json_mode=True, max_tokens=max_tokens)
        responses.append(reply)
        try:
            parsed = validate_cards_reply(reply.text)
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
        cards, dropped = [], []
        # The model may ignore the card limit; we enforce it here.
        for card in parsed.cards[:max_cards]:
            score = support(card.answer, chunk_text)
            if score < min_support:
                dropped.append(card.question)
                continue
            cards.append(
                Flashcard(
                    question=card.question,
                    answer=card.answer,
                    source_time=best_paragraph_time(
                        f"{card.question} {card.answer}", chunk
                    ),
                    origin="llm",
                    chunk=chunk.index,
                    support=round(score, 3),
                )
            )
        return cards, dropped, responses
    raise FlashcardsError(
        f"chunk {chunk.index + 1}/{total}: no valid flashcards after "
        f"{max_repairs + 1} attempt(s): {last_error}"
    )


def concept_cards(concepts: ConceptList) -> list[Flashcard]:
    """One 'What is X?' card per glossary entry. No LLM call needed.

    The definitions were already grounded and timestamped in Step 3.4, so
    these cards are free and reliable; support is 1.0 by construction.
    """
    cards = []
    for c in concepts.concepts:
        alias = f" ({', '.join(c.aliases)})" if c.aliases else ""
        cards.append(
            Flashcard(
                question=f"What is {c.term}{alias}?",
                answer=c.definition,
                source_time=c.first_mention,
                origin="concept",
                support=1.0,
            )
        )
    return cards


def dedupe_cards(
    cards: list[Flashcard], ratio: float = DUPLICATE_RATIO
) -> tuple[list[Flashcard], int]:
    """Drop cards whose question repeats an earlier one; return (kept, dropped).

    Questions are compared as lists of content words with difflib's ratio,
    2 * matching / total, like `dedupe_points` in mapreduce.py. Removing
    stopwords first is what makes "What is an eigenvalue?" and "What is the
    eigenvalue?" identical (both become ["eigenvalue"]). Earlier cards win,
    so pass concept cards first if they should be preferred.
    """
    kept: list[Flashcard] = []
    kept_words: list[list[str]] = []
    for card in cards:
        words = content_words(card.question)
        if any(
            difflib.SequenceMatcher(None, words, other).ratio() >= ratio
            for other in kept_words
        ):
            continue
        kept.append(card)
        kept_words.append(words)
    return kept, len(cards) - len(kept)


def generate_flashcards(
    transcript: Transcript | None,
    client: LLMClient | None,
    *,
    concepts: ConceptList | None = None,
    title: str = "Flashcards",
    max_words: int = DEFAULT_CHUNK_WORDS,
    max_cards: int = 6,
    max_repairs: int = 1,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    min_support: float = MIN_SUPPORT,
    on_chunk: Callable[[int, int, TranscriptChunk], None] | None = None,
) -> FlashcardDeck:
    """Full pipeline. Give a transcript + client for LLM cards, `concepts` for
    glossary cards, or both. With client=None only concept cards are made."""
    if client is None and concepts is None:
        raise ValueError("need an LLM client, a concept list, or both")
    llm_cards: list[Flashcard] = []
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
            cards, bad, replies = generate_chunk_cards(
                client,
                chunk,
                len(chunks),
                max_cards=max_cards,
                max_repairs=max_repairs,
                max_tokens=max_tokens,
                min_support=min_support,
            )
            llm_cards.extend(cards)
            dropped.extend(bad)
            responses.extend(replies)

    # Concept cards first so they win ties in dedupe, then sort by time so the
    # deck follows the lecture. sort() is stable: equal times keep this order.
    all_cards = (concept_cards(concepts) if concepts else []) + llm_cards
    kept, n_duplicates = dedupe_cards(all_cards)
    kept.sort(key=lambda c: c.source_time)
    return FlashcardDeck(
        title=title,
        cards=kept,
        model=getattr(client, "model", "none") if client else "none",
        provider=getattr(client, "provider", "none") if client else "none",
        chunks=len(chunks),
        llm_calls=len(responses),
        prompt_tokens=sum(r.prompt_tokens or 0 for r in responses),
        completion_tokens=sum(r.completion_tokens or 0 for r in responses),
        latency_s=sum(r.latency_s for r in responses),
        dropped_ungrounded=dropped,
        dropped_duplicates=n_duplicates,
    )


# ---------------------------------------------------------------------------
# Files: JSON, Markdown and the Anki CSV
# ---------------------------------------------------------------------------


def anki_tag(text: str) -> str:
    """Make a string safe as an Anki tag.

    Anki separates tags with spaces, so a tag cannot contain one. We
    lowercase, turn runs of anything that is not a letter, digit, '-' or ':'
    into '_', and trim underscores. ':' is kept because Anki uses '::' for
    tag hierarchies (e.g. 'linear_algebra::lecture01').
    """
    return re.sub(r"[^a-z0-9:\-]+", "_", text.lower()).strip("_") or "lecture"


def anki_csv_text(
    deck: FlashcardDeck,
    *,
    deck_name: str | None = None,
    header: bool = True,
) -> str:
    """The deck as Anki-importable CSV text (columns: Front, Back, Tags).

    The back of each card ends with the lecture timestamp so you can jump to
    the recording when a card is unclear. Fields go through the csv module,
    which quotes commas, quotes and newlines correctly; never build CSV rows
    with string joins, a comma inside an answer would shift every column.
    """
    name = (deck_name or deck.title).replace("\n", " ").strip() or "Flashcards"
    base_tag = anki_tag(name)
    buffer = io.StringIO()
    if header:
        # Anki file headers (2.1.55+). They are not CSV rows, so they are
        # written directly, before the csv writer starts.
        buffer.write("#separator:Comma\n")
        buffer.write("#html:false\n")
        buffer.write("#notetype:Basic\n")
        buffer.write(f"#deck:{name}\n")
        buffer.write("#columns:Front,Back,Tags\n")
        buffer.write("#tags column:3\n")
    # lineterminator="\n" keeps line endings the same on every OS.
    writer = csv.writer(buffer, lineterminator="\n")
    for card in deck.cards:
        back = f"{card.answer} [{format_timestamp(card.source_time)}]"
        writer.writerow([card.question, back, f"{base_tag} {card.origin}"])
    return buffer.getvalue()


def write_flashcards(
    deck: FlashcardDeck,
    out_dir: str | Path,
    *,
    deck_name: str | None = None,
    anki_header: bool = True,
) -> tuple[Path, Path, Path]:
    """Write flashcards.json, flashcards.md and flashcards.csv; return paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "flashcards.json"
    md_path = out_dir / "flashcards.md"
    csv_path = out_dir / "flashcards.csv"
    json_path.write_text(deck.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(deck.to_markdown(), encoding="utf-8")
    # newline="" stops Python from translating "\n" again on Windows.
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(anki_csv_text(deck, deck_name=deck_name, header=anki_header))
    return json_path, md_path, csv_path


def load_flashcards(path: str | Path) -> FlashcardDeck:
    """Read flashcards.json and validate it; raises ValueError if invalid."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        return FlashcardDeck.model_validate_json(text)
    except ValidationError as error:
        reason = _short_errors(error)
        raise ValueError(f"{path} is not a valid flashcards file: {reason}") from None
