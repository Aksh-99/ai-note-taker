"""Tests for flashcard generation and Anki export (src/notetaker/flashcards.py).

No live LLM calls: `FakeLLMClient` returns scripted JSON replies.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from notetaker import cli
from notetaker.concepts import Concept, ConceptList, write_concepts
from notetaker.flashcards import (
    Flashcard,
    FlashcardDeck,
    FlashcardsError,
    anki_csv_text,
    anki_tag,
    best_paragraph_time,
    concept_cards,
    content_words,
    dedupe_cards,
    flashcards_json_schema,
    generate_chunk_cards,
    generate_flashcards,
    load_flashcards,
    support,
    validate_cards_reply,
    write_flashcards,
)
from notetaker.llm import FakeLLMClient
from notetaker.notes import chunk_transcript
from notetaker.transcribe import Segment, Transcript

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "flashcards.schema.json"


def make_transcript(spans: list[tuple[float, float, str]]) -> Transcript:
    segments = [
        Segment(id=i, start=s, end=e, text=t, avg_logprob=-0.2, no_speech_prob=0.01)
        for i, (s, e, t) in enumerate(spans)
    ]
    return Transcript(
        segments=segments,
        language="en",
        language_probability=1.0,
        duration=spans[-1][1] if spans else 0.0,
        model="small",
    )


# 5 s pauses split these into four paragraphs.
LECTURE = make_transcript(
    [
        (
            0.0,
            10.0,
            "An eigenvalue is the factor by which a matrix stretches a vector.",
        ),
        (15.0, 25.0, "The determinant is zero exactly when the matrix is singular."),
        (30.0, 40.0, "Gradient descent moves the weights against the gradient."),
        (45.0, 55.0, "The learning rate controls the step size of each update."),
    ]
)


def card(q: str, a: str) -> dict:
    return {"question": q, "answer": a}


def reply(*cards: dict) -> str:
    return json.dumps({"cards": list(cards)})


GOOD = card(
    "What does an eigenvalue measure?",
    "The factor by which a matrix stretches a vector.",
)
HALLUCINATED = card(
    "Who invented eigenvalues?", "Augustin Cauchy in eighteen twenty nine."
)


def one_chunk():
    return chunk_transcript(LECTURE)[0]


# --- content words and support ----------------------------------------------


def test_content_words_drop_stopwords_but_keep_negation_and_digits():
    assert content_words("What is the rank of a matrix?") == ["rank", "matrix"]
    assert "not" in content_words("Why is it not invertible?")
    assert content_words("rank 2") == ["rank", "2"]


def test_support_is_share_of_answer_words_in_source():
    source = "the determinant is zero when the matrix is singular"
    assert support("The matrix is singular", source) == 1.0
    assert support("singular quaternion", source) == 0.5
    assert support("Yes.", source) == 0.0  # nothing checkable


# --- validation -------------------------------------------------------------


def test_valid_reply_parses_and_strips():
    parsed = validate_cards_reply(reply(card(' "What is X?" ', "**an answer**")))
    assert parsed.cards[0].question == "What is X?"
    assert parsed.cards[0].answer == "an answer"


@pytest.mark.parametrize(
    "text",
    [
        json.dumps([card("q?", "a")]),  # bare list
        json.dumps({"flashcards": [card("q?", "a")]}),  # other list key
        json.dumps({"cards": [{"Q": "q?", "A": "a"}]}),  # short keys
        json.dumps({"cards": [{"front": "q?", "back": "a"}]}),  # Anki words
        "```json\n" + reply(card("q?", "a")) + "\n```",  # code fence
    ],
)
def test_common_variants_are_accepted(text):
    parsed = validate_cards_reply(text)
    assert (parsed.cards[0].question, parsed.cards[0].answer) == ("q?", "a")


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        json.dumps({"cards": [{"question": "q?"}]}),  # no answer
        json.dumps({"cards": [card("", "a")]}),  # empty question
        json.dumps({"cards": [card("q?", "word " * 61)]}),  # answer too long
        json.dumps({"something": "else"}),
    ],
)
def test_invalid_replies_raise_value_error(text):
    with pytest.raises(ValueError):
        validate_cards_reply(text)


def test_empty_list_is_valid():
    assert validate_cards_reply(reply()).cards == []


# --- one chunk ----------------------------------------------------------------


def test_ungrounded_answers_are_dropped():
    client = FakeLLMClient(replies=[reply(GOOD, HALLUCINATED)])
    cards, dropped, replies = generate_chunk_cards(client, one_chunk(), 1)
    assert [c.question for c in cards] == [GOOD["question"]]
    assert dropped == [HALLUCINATED["question"]]
    assert cards[0].support == 1.0 and cards[0].origin == "llm"
    assert len(replies) == 1


def test_timestamp_points_at_best_matching_paragraph():
    q = card("What does the learning rate control?", "The step size of each update.")
    cards, _, _ = generate_chunk_cards(
        FakeLLMClient(replies=[reply(q)]), one_chunk(), 1
    )
    assert cards[0].source_time == 45.0
    assert best_paragraph_time("determinant singular", one_chunk()) == 15.0


def test_max_cards_is_enforced_in_code():
    many = [card(f"What is the matrix fact {i}?", "a matrix vector") for i in range(9)]
    cards, _, _ = generate_chunk_cards(
        FakeLLMClient(replies=[reply(*many)]), one_chunk(), 1, max_cards=3
    )
    assert len(cards) == 3


def test_repair_loop_recovers_and_sends_error_back():
    client = FakeLLMClient(replies=["oops", reply(GOOD)])
    cards, _, replies = generate_chunk_cards(client, one_chunk(), 1)
    assert len(cards) == 1 and len(replies) == 2
    assert "could not be used" in client.calls[1][-1].content


def test_gives_up_after_max_repairs():
    client = FakeLLMClient(replies=["oops"])
    with pytest.raises(FlashcardsError, match="no valid flashcards after 2"):
        generate_chunk_cards(client, one_chunk(), 1, max_repairs=1)
    with pytest.raises(ValueError):
        generate_chunk_cards(client, one_chunk(), 1, max_repairs=-1)


def test_prompt_contains_timestamped_transcript_and_limits():
    client = FakeLLMClient(replies=[reply()])
    generate_chunk_cards(client, one_chunk(), 1, max_cards=4)
    prompt = client.calls[0][-1].content
    assert "[00:00:15] The determinant is zero" in prompt
    assert "0 to 4 cards" in prompt


# --- concepts, dedupe, full pipeline -----------------------------------------


GLOSSARY = ConceptList(
    title="Key concepts",
    concepts=[
        Concept(
            term="eigenvalue",
            definition="The factor by which a matrix stretches a vector.",
            first_mention=0.0,
            mentions=[0.0],
        ),
        Concept(
            term="singular value decomposition",
            definition="A factorization of any matrix.",
            aliases=["SVD"],
            first_mention=20.0,
        ),
    ],
    model="m",
    provider="p",
    chunks=1,
    llm_calls=1,
)


def test_concept_cards_need_no_llm():
    cards = concept_cards(GLOSSARY)
    assert cards[0].question == "What is eigenvalue?"
    assert cards[1].question == "What is singular value decomposition (SVD)?"
    assert all(c.origin == "concept" and c.support == 1.0 for c in cards)


def make_card(q: str, t: float = 0.0) -> Flashcard:
    return Flashcard(question=q, answer="a", source_time=t, origin="llm", support=1)


def test_dedupe_ignores_stopwords_but_keeps_negation():
    cards = [
        make_card("What is an eigenvalue?"),
        make_card("What is the eigenvalue?"),
        make_card("Why is the matrix invertible?"),
        make_card("Why is the matrix not invertible?"),
    ]
    kept, dropped = dedupe_cards(cards)
    assert dropped == 1
    assert [c.question for c in kept] == [
        "What is an eigenvalue?",
        "Why is the matrix invertible?",
        "Why is the matrix not invertible?",
    ]


def test_generate_flashcards_end_to_end_with_concepts():
    llm_reply = reply(
        card("What is eigenvalue?", "The factor a matrix stretches a vector by."),
        card(
            "What does gradient descent do?", "Moves the weights against the gradient."
        ),
        HALLUCINATED,
    )
    client = FakeLLMClient(replies=[llm_reply])
    deck = generate_flashcards(LECTURE, client, concepts=GLOSSARY, title="L1")
    questions = [c.question for c in deck.cards]
    # The concept card wins the duplicate "What is eigenvalue?" question.
    assert questions.count("What is eigenvalue?") == 1
    assert deck.cards[0].origin == "concept"
    assert deck.dropped_duplicates == 1
    assert deck.dropped_ungrounded == [HALLUCINATED["question"]]
    # Sorted by time: eigenvalue (0) < SVD (20) < gradient descent (30).
    assert [c.source_time for c in deck.cards] == [0.0, 20.0, 30.0]
    assert deck.llm_calls == 1 and deck.provider == "fake"


def test_concepts_only_makes_no_calls_and_needs_input():
    deck = generate_flashcards(None, None, concepts=GLOSSARY)
    assert len(deck.cards) == 2 and deck.llm_calls == 0 and deck.model == "none"
    with pytest.raises(ValueError):
        generate_flashcards(LECTURE, None)
    with pytest.raises(ValueError):
        generate_flashcards(None, FakeLLMClient())


# --- files, Anki CSV and schema ---------------------------------------------


def sample_deck() -> FlashcardDeck:
    return FlashcardDeck(
        title="Linear Algebra, Lecture 1",
        cards=[
            Flashcard(
                question='What is "rank", formally?',
                answer="The number of independent columns,\nnot rows.",
                source_time=75.0,
                origin="llm",
                chunk=0,
                support=0.8,
            )
        ],
        model="m",
        provider="p",
        chunks=1,
        llm_calls=1,
    )


def test_anki_tag_has_no_spaces():
    assert anki_tag("Linear Algebra, Lecture 1") == "linear_algebra_lecture_1"
    assert anki_tag("math::la") == "math::la"
    assert anki_tag("!!!") == "lecture"


def test_anki_csv_header_and_quoting():
    text = anki_csv_text(sample_deck(), deck_name="Math::LA")
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert "#separator:Comma" in header and "#deck:Math::LA" in header
    assert "#tags column:3" in header
    body = "".join(line + "\n" for line in text.split("\n") if not line.startswith("#"))
    rows = list(csv.reader(io.StringIO(body)))
    rows = [r for r in rows if r]
    # Commas, quotes and the newline survive the round trip in one row.
    assert rows == [
        [
            'What is "rank", formally?',
            "The number of independent columns,\nnot rows. [00:01:15]",
            "math::la llm",
        ]
    ]


def test_plain_csv_has_no_header():
    text = anki_csv_text(sample_deck(), header=False)
    assert not text.startswith("#")
    assert next(csv.reader(io.StringIO(text)))[2] == "linear_algebra_lecture_1 llm"


def test_write_load_round_trip_and_markdown(tmp_path):
    json_path, md_path, csv_path = write_flashcards(sample_deck(), tmp_path / "out")
    assert load_flashcards(json_path) == sample_deck()
    md = md_path.read_text()
    assert '**Q1.** What is "rank", formally? `[00:01:15]`' in md
    assert csv_path.read_text().startswith("#separator:Comma")
    empty = sample_deck().model_copy(update={"cards": []})
    assert "_No flashcards._" in empty.to_markdown()


def test_load_rejects_invalid_file(tmp_path):
    bad = tmp_path / "flashcards.json"
    bad.write_text(json.dumps({"title": "x", "cards": [{"question": ""}]}))
    with pytest.raises(ValueError, match="not a valid flashcards file"):
        load_flashcards(bad)


def test_committed_schema_matches_code():
    # If this fails, regenerate the file with:
    #   notetaker flashcards --schema > schemas/flashcards.schema.json
    assert json.loads(SCHEMA_PATH.read_text()) == flashcards_json_schema()


# --- CLI ----------------------------------------------------------------------


def run_cli(argv: list[str], client=None) -> int:
    args = cli.build_parser().parse_args(["flashcards", *argv])
    return cli.run_flashcards(args, client=client)


def test_cli_writes_three_files_next_to_transcript(tmp_path, capsys):
    folder = tmp_path / "lecture01"
    folder.mkdir()
    LECTURE.save_json(folder / "transcript.json")
    client = FakeLLMClient(replies=[reply(GOOD, HALLUCINATED)])
    assert run_cli([str(folder / "transcript.json")], client=client) == 0
    out = capsys.readouterr().out
    assert "Dropped 1 card(s)" in out and "1 card(s)" in out
    deck = load_flashcards(folder / "flashcards.json")
    assert deck.title == "Flashcards: lecture01"
    assert (folder / "flashcards.md").exists()
    assert "#deck:Flashcards: lecture01" in (folder / "flashcards.csv").read_text()


def test_cli_concepts_only_needs_no_llm(tmp_path):
    write_concepts(GLOSSARY, tmp_path)
    argv = ["--concepts-only", "--concepts", str(tmp_path / "concepts.json")]
    assert run_cli([*argv, "--plain-csv", "--deck", "LA"]) == 0
    text = (tmp_path / "flashcards.csv").read_text()
    assert not text.startswith("#") and "What is eigenvalue?" in text


def test_cli_schema_flag_prints_schema(capsys):
    assert run_cli(["--schema"]) == 0
    assert json.loads(capsys.readouterr().out) == flashcards_json_schema()


def test_cli_errors(tmp_path, capsys):
    assert run_cli([]) == 1
    assert run_cli(["--concepts-only"]) == 1
    assert run_cli([str(tmp_path / "missing.json")]) == 1
    junk = tmp_path / "junk.json"
    junk.write_text("{}")
    assert run_cli([str(junk), "--max-cards", "0"]) == 1
    assert run_cli(["--concepts-only", "--concepts", str(junk)]) == 1
    out = capsys.readouterr().out
    assert "give a transcript.json" in out and "needs --concepts" in out
    assert "no such file" in out and "not a valid concepts file" in out


def test_cli_reports_unusable_model_output(tmp_path, capsys):
    path = tmp_path / "transcript.json"
    LECTURE.save_json(path)
    assert run_cli([str(path)], client=FakeLLMClient(replies=["nope"])) == 1
    assert "no valid flashcards" in capsys.readouterr().out
