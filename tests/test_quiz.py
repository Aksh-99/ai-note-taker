"""Tests for multiple-choice quiz generation (src/notetaker/quiz.py).

No live LLM calls: `FakeLLMClient` returns scripted JSON replies.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from notetaker import cli
from notetaker.concepts import Concept, ConceptList, write_concepts
from notetaker.llm import FakeLLMClient
from notetaker.notes import chunk_transcript
from notetaker.quiz import (
    Quiz,
    QuizError,
    QuizQuestion,
    concept_questions,
    dedupe_questions,
    generate_chunk_questions,
    generate_quiz,
    load_quiz,
    mask_term,
    option_key,
    quiz_json_schema,
    shuffle_options,
    validate_quiz_reply,
    write_quiz,
)
from notetaker.transcribe import Segment, Transcript

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "quiz.schema.json"


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


def item(q: str, correct: str, distractors: list[str], why: str = "") -> dict:
    return {
        "question": q,
        "correct": correct,
        "distractors": distractors,
        "explanation": why,
    }


def reply(*items: dict) -> str:
    return json.dumps({"questions": list(items)})


GOOD = item(
    "What does the learning rate control in gradient descent?",
    "The step size of each update",
    ["The number of layers", "The batch size", "The loss function"],
    "The lecturer says it controls the step size.",
)
HALLUCINATED = item(
    "Who introduced gradient descent?",
    "Augustin Cauchy in 1847",
    ["Isaac Newton", "Carl Gauss", "Alan Turing"],
)


def one_chunk():
    return chunk_transcript(LECTURE)[0]


# --- validation -------------------------------------------------------------


def test_valid_reply_parses_and_strips():
    parsed = validate_quiz_reply(
        reply(item(' "Q?" ', "**yes it is**", [" no way ", "", "maybe so"]))
    )
    q = parsed.questions[0]
    assert (q.question, q.correct) == ("Q?", "yes it is")
    assert q.distractors == ["no way", "maybe so"]  # empty padding removed


@pytest.mark.parametrize(
    "text",
    [
        json.dumps([item("q?", "right", ["w1", "w2", "w3"])]),  # bare list
        json.dumps({"quiz": [item("q?", "right", ["w1", "w2", "w3"])]}),
        json.dumps(
            {
                "questions": [
                    {"stem": "q?", "answer": "right", "wrong_answers": ["w1", "w2"]}
                ]
            }
        ),
        "```json\n" + reply(item("q?", "right", ["w1", "w2", "w3"])) + "\n```",
    ],
)
def test_common_variants_are_accepted(text):
    q = validate_quiz_reply(text).questions[0]
    assert (q.question, q.correct) == ("q?", "right")
    assert q.distractors[:2] == ["w1", "w2"]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("not json", "Expecting value"),
        (reply(item("q?", "right", ["only one"])), "distractors"),
        (reply(item("q?", "Right.", ["right", "w2", "w3"])), "different"),
        (reply(item("q?", "right", ["w1", "None of the above", "w3"])), "shuffling"),
        (reply(item("q?", "word " * 31, ["w1", "w2", "w3"])), "30 words"),
        (reply(item("word " * 51, "right", ["w1", "w2", "w3"])), "50 words"),
        (json.dumps({"something": "else"}), "questions"),
    ],
)
def test_invalid_replies_raise(text, reason):
    with pytest.raises(ValueError, match=reason):
        validate_quiz_reply(text)


def test_option_key_ignores_case_punctuation_and_plurals():
    assert option_key("The Gradients.") == option_key("the gradient")


# --- shuffling ----------------------------------------------------------------


def test_shuffle_is_deterministic_and_keeps_the_key():
    a = shuffle_options("key", ["x", "y", "z"], seed=0, salt="q1")
    b = shuffle_options("key", ["x", "y", "z"], seed=0, salt="q1")
    assert a == b
    options, answer = a
    assert options[answer] == "key" and sorted(options) == ["key", "x", "y", "z"]


def test_shuffle_spreads_the_key_over_all_positions():
    # Over many questions the key should land on every letter, roughly evenly
    # (this is the position-bias fix: the model always puts it first).
    positions = Counter(
        shuffle_options("key", ["x", "y", "z"], seed=0, salt=f"q{i}")[1]
        for i in range(400)
    )
    assert set(positions) == {0, 1, 2, 3}
    assert all(60 <= n <= 140 for n in positions.values())


def test_seed_changes_the_order():
    orders = {
        tuple(shuffle_options("k", ["a", "b", "c"], seed=s, salt="q")[0])
        for s in range(10)
    }
    assert len(orders) > 1


# --- one chunk ----------------------------------------------------------------


def test_chunk_questions_are_grounded_shuffled_and_timestamped():
    client = FakeLLMClient(replies=[reply(GOOD, HALLUCINATED)])
    questions, dropped, replies = generate_chunk_questions(client, one_chunk(), 1)
    assert dropped == ["Who introduced gradient descent?"]
    assert len(replies) == 1
    (q,) = questions
    assert q.options[q.answer] == "The step size of each update"
    assert len(q.options) == 4
    assert q.source_time == 45.0  # the learning-rate paragraph
    assert q.origin == "llm" and q.chunk == 0 and q.support == 1.0
    # The prompt asks for exactly 3 distractors and contains the transcript.
    prompt = client.calls[0][1].content
    assert "Exactly 3 distractors" in prompt and "[00:00:45]" in prompt


def test_extra_distractors_and_questions_are_cut():
    many = item(GOOD["question"], GOOD["correct"], ["a1", "b2", "c3", "d4", "e5"])
    other = item("What is singular?", "determinant is zero", ["x1", "y2", "z3"])
    client = FakeLLMClient(replies=[reply(many, other)])
    questions, _, _ = generate_chunk_questions(client, one_chunk(), 1, max_questions=1)
    assert len(questions) == 1 and len(questions[0].options) == 4


def test_repair_loop_then_success():
    client = FakeLLMClient(replies=["oops", reply(GOOD)])
    questions, _, replies = generate_chunk_questions(client, one_chunk(), 1)
    assert len(questions) == 1 and len(replies) == 2
    assert "could not be used" in client.calls[1][-1].content


def test_gives_up_after_repairs():
    client = FakeLLMClient(replies=["oops"])
    with pytest.raises(QuizError, match="no valid quiz questions after 2"):
        generate_chunk_questions(client, one_chunk(), 1, max_repairs=1)
    with pytest.raises(ValueError):
        generate_chunk_questions(client, one_chunk(), 1, max_repairs=-1)


# --- glossary questions -------------------------------------------------------


def concept(term: str, definition: str, t: float, aliases=()) -> Concept:
    return Concept(
        term=term,
        definition=definition,
        aliases=list(aliases),
        first_mention=t,
        mentions=[t],
    )


GLOSSARY = ConceptList(
    title="Glossary",
    concepts=[
        concept("eigenvalue", "The factor by which a matrix stretches a vector.", 0),
        concept("determinant", "A number that is zero for singular matrices.", 15),
        concept(
            "singular value decomposition",
            "SVD factors any matrix into rotations and a scaling.",
            20,
            aliases=["SVD"],
        ),
        concept("gradient descent", "Moving weights against the gradient.", 30),
        concept("learning rate", "The step size of each update.", 45),
    ],
    model="m",
    provider="p",
    chunks=1,
    llm_calls=1,
)


def test_mask_term_hides_the_answer():
    text = "SVD factors any matrix; singular value decompositions are useful."
    masked = mask_term(text, ["singular value decomposition", "SVD"])
    assert masked == "____ factors any matrix; ____ are useful."


def test_concept_questions_use_nearby_terms_as_distractors():
    questions = concept_questions(GLOSSARY)
    assert len(questions) == 5
    svd = next(q for q in questions if "rotations" in q.question)
    assert "SVD" not in svd.question and "____" in svd.question
    assert svd.options[svd.answer] == "singular value decomposition"
    # The three concepts first mentioned closest to t=20 (15, 30, 0).
    assert set(svd.options) - {"singular value decomposition"} == {
        "determinant",
        "gradient descent",
        "eigenvalue",
    }
    assert svd.origin == "concept" and svd.source_time == 20


def test_concept_questions_need_enough_terms():
    small = GLOSSARY.model_copy(update={"concepts": GLOSSARY.concepts[:3]})
    assert concept_questions(small) == []


# --- pipeline -----------------------------------------------------------------


def make_question(q: str, t: float, origin="llm") -> QuizQuestion:
    return QuizQuestion(
        question=q,
        options=["a", "b", "c", "d"],
        answer=0,
        source_time=t,
        origin=origin,
        support=1.0,
    )


def test_dedupe_questions_keeps_first():
    kept, n = dedupe_questions(
        [
            make_question("What is an eigenvalue?", 0),
            make_question("What is the eigenvalue?", 5),
            make_question("What is a determinant?", 9),
        ]
    )
    assert [q.source_time for q in kept] == [0, 9] and n == 1


def test_generate_quiz_combines_sources_sorted_by_time():
    client = FakeLLMClient(replies=[reply(GOOD, HALLUCINATED)])
    quiz = generate_quiz(LECTURE, client, concepts=GLOSSARY, title="T", seed=3)
    assert [q.source_time for q in quiz.questions] == sorted(
        q.source_time for q in quiz.questions
    )
    assert Counter(q.origin for q in quiz.questions) == {"llm": 1, "concept": 5}
    assert quiz.llm_calls == 1 and quiz.chunks == 1 and quiz.seed == 3
    assert quiz.dropped_ungrounded == ["Who introduced gradient descent?"]


def test_generate_quiz_without_llm_and_input_checks():
    quiz = generate_quiz(None, None, concepts=GLOSSARY)
    assert quiz.llm_calls == 0 and quiz.model == "none" and len(quiz.questions) == 5
    with pytest.raises(ValueError):
        generate_quiz(None, None)
    with pytest.raises(ValueError):
        generate_quiz(None, FakeLLMClient())


def test_quality_stats():
    quiz = Quiz(
        title="t",
        questions=[
            QuizQuestion(
                question="q1",
                options=["the long correct answer", "no", "nope", "never"],
                answer=0,
                source_time=0,
                origin="llm",
                support=1.0,
            ),
            QuizQuestion(
                question="q2",
                options=["short", "a b", "c d", "e f"],
                answer=2,
                source_time=0,
                origin="llm",
                support=1.0,
            ),
        ],
        model="m",
        provider="p",
        chunks=1,
        llm_calls=1,
    )
    assert quiz.longest_option_rate() == 0.5  # q2's key ties, so it does not count
    assert quiz.answer_positions() == {"A": 1, "B": 0, "C": 1, "D": 0}
    empty = quiz.model_copy(update={"questions": []})
    assert empty.longest_option_rate() is None and empty.answer_positions() == {}


def test_answer_index_must_be_in_range():
    with pytest.raises(ValueError, match="only 3 options"):
        QuizQuestion(
            question="q",
            options=["a", "b", "c"],
            answer=3,
            source_time=0,
            origin="llm",
            support=1.0,
        )


# --- files and schema ---------------------------------------------------------


def test_write_load_round_trip_and_markdown(tmp_path):
    quiz = generate_quiz(None, None, concepts=GLOSSARY, title="LA quiz")
    json_path, md_path = write_quiz(quiz, tmp_path / "out")
    assert load_quiz(json_path) == quiz
    md = md_path.read_text()
    assert md.startswith("# LA quiz") and "## Answer key" in md
    assert "- A. " in md and "- D. " in md
    first = quiz.questions[0]
    assert f"1. **{first.answer_letter}**." in md
    empty = quiz.model_copy(update={"questions": []})
    assert "_No questions._" in empty.to_markdown()


def test_load_rejects_invalid_file(tmp_path):
    bad = tmp_path / "quiz.json"
    bad.write_text(json.dumps({"title": "x", "questions": [{"question": ""}]}))
    with pytest.raises(ValueError, match="not a valid quiz file"):
        load_quiz(bad)


def test_committed_schema_matches_code():
    # If this fails, regenerate the file with:
    #   notetaker quiz --schema > schemas/quiz.schema.json
    assert json.loads(SCHEMA_PATH.read_text()) == quiz_json_schema()


# --- CLI ----------------------------------------------------------------------


def run_cli(argv: list[str], client=None) -> int:
    args = cli.build_parser().parse_args(["quiz", *argv])
    return cli.run_quiz(args, client=client)


def test_cli_writes_files_next_to_transcript(tmp_path, capsys):
    folder = tmp_path / "lecture01"
    folder.mkdir()
    LECTURE.save_json(folder / "transcript.json")
    client = FakeLLMClient(replies=[reply(GOOD, HALLUCINATED)])
    assert run_cli([str(folder / "transcript.json")], client=client) == 0
    out = capsys.readouterr().out
    assert "Dropped 1 question(s)" in out and "1 question(s)" in out
    assert "longest option" in out
    quiz = load_quiz(folder / "quiz.json")
    assert quiz.title == "Quiz: lecture01"
    assert (folder / "quiz.md").exists()


def test_cli_concepts_only_needs_no_llm(tmp_path, capsys):
    write_concepts(GLOSSARY, tmp_path)
    argv = ["--concepts-only", "--concepts", str(tmp_path / "concepts.json")]
    assert run_cli([*argv, "--seed", "7", "-o", str(tmp_path / "q")]) == 0
    quiz = load_quiz(tmp_path / "q" / "quiz.json")
    assert len(quiz.questions) == 5 and quiz.seed == 7


def test_cli_warns_when_glossary_too_small(tmp_path, capsys):
    small = GLOSSARY.model_copy(update={"concepts": GLOSSARY.concepts[:2]})
    write_concepts(small, tmp_path)
    argv = ["--concepts-only", "--concepts", str(tmp_path / "concepts.json")]
    assert run_cli(argv) == 0
    assert "at least 4 concepts" in capsys.readouterr().out


def test_cli_schema_flag_prints_schema(capsys):
    assert run_cli(["--schema"]) == 0
    assert json.loads(capsys.readouterr().out) == quiz_json_schema()


def test_cli_errors(tmp_path, capsys):
    assert run_cli([]) == 1
    assert run_cli(["--concepts-only"]) == 1
    assert run_cli([str(tmp_path / "missing.json")]) == 1
    junk = tmp_path / "junk.json"
    junk.write_text("{}")
    assert run_cli([str(junk), "--max-questions", "0"]) == 1
    assert run_cli(["--concepts-only", "--concepts", str(junk)]) == 1
    out = capsys.readouterr().out
    assert "give a transcript.json" in out and "needs --concepts" in out
    assert "no such file" in out and "not a valid concepts file" in out


def test_cli_reports_unusable_model_output(tmp_path, capsys):
    path = tmp_path / "transcript.json"
    LECTURE.save_json(path)
    assert run_cli([str(path)], client=FakeLLMClient(replies=["nope"])) == 1
    assert "no valid quiz questions" in capsys.readouterr().out


def test_main_dispatches_quiz(tmp_path, capsys):
    write_concepts(GLOSSARY, tmp_path)
    argv = ["quiz", "--concepts-only", "--concepts", str(tmp_path / "concepts.json")]
    assert cli.main(argv) == 0
    assert (tmp_path / "quiz.md").exists()
