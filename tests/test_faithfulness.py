"""Tests for the faithfulness eval (src/notetaker/faithfulness.py).

No live LLM calls: `FakeLLMClient` plays the judge with scripted JSON replies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from notetaker import cli
from notetaker.concepts import Concept, ConceptList, write_concepts
from notetaker.faithfulness import (
    Claim,
    JudgedClaim,
    Retriever,
    claims_from_notes,
    cohen_kappa,
    evaluate_faithfulness,
    faithfulness_json_schema,
    lexical_support,
    lexical_verdict,
    load_report,
    load_target,
    quote_found,
    sample_claims,
    select_evidence,
    summarize,
    transcript_passages,
    validate_judge_reply,
    write_report,
)
from notetaker.flashcards import Flashcard, FlashcardDeck, write_flashcards
from notetaker.llm import FakeLLMClient
from notetaker.notes import Notes, NoteSection, write_notes
from notetaker.quiz import Quiz, QuizQuestion, write_quiz
from notetaker.transcribe import Segment, Transcript

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "faithfulness.schema.json"
)


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


# 5 s pauses split these into four paragraphs (passages).
LECTURE = make_transcript(
    [
        (
            0.0,
            10.0,
            "An eigenvalue is the factor by which a matrix stretches a vector.",
        ),
        (15.0, 25.0, "The determinant of a matrix is the product of its eigenvalues."),
        (30.0, 40.0, "Gradient descent moves the weights against the gradient."),
        (45.0, 55.0, "The learning rate controls the step size of gradient descent."),
    ]
)


def make_notes(model: str = "gen-model", provider: str = "ollama") -> Notes:
    return Notes(
        title="Linear algebra",
        sections=[
            NoteSection(
                heading="Eigenvalues",
                start=0.0,
                summary="Eigenvalues describe how a matrix stretches vectors.",
                key_points=[
                    "The determinant equals the product of the eigenvalues.",
                    "Eigenvalues are always positive.",
                ],
                chunk=0,
            ),
            NoteSection(
                heading="Optimization",
                start=30.0,
                summary="",
                key_points=["The learning rate sets the step size."],
                chunk=0,
            ),
        ],
        model=model,
        provider=provider,
        chunks=1,
        overview="A lecture on eigenvalues and gradient descent.",
        takeaways=["Gradient descent moves against the gradient."],
    )


def verdict_reply(*items: tuple[int, str, str]) -> str:
    return json.dumps(
        {
            "verdicts": [
                {"claim": n, "verdict": v, "quote": q, "reason": "because"}
                for n, v, q in items
            ]
        }
    )


# --- claims ------------------------------------------------------------------


def test_claims_from_notes_covers_every_part_in_order():
    claims = claims_from_notes(make_notes())
    kinds = [c.kind for c in claims]
    assert kinds == [
        "overview",
        "summary",
        "key_point",
        "key_point",
        "key_point",
        "takeaway",
    ]
    assert [c.id for c in claims] == list(range(1, 7))
    assert claims[2].source == "section 1 (Eigenvalues), point 1"
    assert claims[4].time == 30.0
    assert claims[0].time is None


def test_load_target_detects_each_kind(tmp_path):
    write_notes(make_notes(), tmp_path / "n")
    assert load_target(tmp_path / "n" / "notes.json").kind == "notes"

    concepts = ConceptList(
        title="g",
        concepts=[
            Concept(term="eigenvalue", definition="a stretch factor", first_mention=0.0)
        ],
        model="m",
        provider="p",
        chunks=1,
        llm_calls=1,
    )
    write_concepts(concepts, tmp_path / "c")
    target = load_target(tmp_path / "c" / "concepts.json")
    assert target.kind == "concepts"
    assert target.claims[0].text == "eigenvalue: a stretch factor"

    deck = FlashcardDeck(
        title="d",
        cards=[
            Flashcard(
                question="What is the determinant?",
                answer="The product of the eigenvalues.",
                source_time=15.0,
                origin="llm",
                support=1.0,
            )
        ],
        model="m",
        provider="p",
        chunks=1,
        llm_calls=1,
    )
    write_flashcards(deck, tmp_path / "f")
    target = load_target(tmp_path / "f" / "flashcards.json")
    assert target.kind == "flashcards"
    assert target.claims[0].text.endswith("Answer: The product of the eigenvalues.")

    quiz = Quiz(
        title="q",
        questions=[
            QuizQuestion(
                question="What controls the step size?",
                options=["momentum", "the learning rate", "the loss"],
                answer=1,
                source_time=45.0,
                origin="llm",
                support=1.0,
            )
        ],
        model="m",
        provider="p",
        chunks=1,
        llm_calls=1,
    )
    write_quiz(quiz, tmp_path / "q")
    target = load_target(tmp_path / "q" / "quiz.json")
    assert target.kind == "quiz"
    assert target.claims[0].text.endswith("Answer: the learning rate")


def test_load_target_rejects_unknown_and_invalid_files(tmp_path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"hello": 1}))
    with pytest.raises(ValueError, match="expected"):
        load_target(other)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"cards": [{"question": ""}]}))
    with pytest.raises(ValueError, match="could not be read"):
        load_target(bad)
    listing = tmp_path / "list.json"
    listing.write_text("[]")
    with pytest.raises(ValueError):
        load_target(listing)


def test_sample_claims_spreads_over_the_lecture():
    claims = [
        Claim(id=i, text=f"claim {i}", kind="key_point", source="s")
        for i in range(1, 11)
    ]
    assert [c.id for c in sample_claims(claims, 5)] == [1, 3, 5, 7, 9]
    assert sample_claims(claims, None) == claims
    assert sample_claims(claims, 50) == claims
    with pytest.raises(ValueError):
        sample_claims(claims, 0)


# --- retrieval ---------------------------------------------------------------


def test_passages_follow_paragraphs():
    passages = transcript_passages(LECTURE)
    assert [p.start for p in passages] == [0.0, 15.0, 30.0, 45.0]
    assert "determinant" in passages[1].words


def test_retriever_prefers_rare_shared_words():
    passages = transcript_passages(LECTURE)
    retriever = Retriever(passages)
    # "matrix" occurs in two passages, "determinant" in one: idf is higher.
    assert retriever.idf["determinant"] > retriever.idf["matrix"]
    claim = Claim(
        id=1, text="The determinant of a matrix", kind="key_point", source="s"
    )
    assert retriever.top_k(claim, 1) == [1]


def test_retriever_adds_the_passage_playing_at_the_claim_time():
    retriever = Retriever(transcript_passages(LECTURE))
    claim = Claim(
        id=1, text="Determinant facts.", kind="summary", source="s", time=46.0
    )
    assert retriever.top_k(claim, 1) == [1, 3]
    # A claim with no shared words still gets the passage at its time.
    vague = Claim(
        id=2, text="Some vague remark.", kind="summary", source="s", time=31.0
    )
    assert retriever.top_k(vague, 3) == [2]


def test_retriever_ties_go_to_the_closest_passage():
    lecture = make_transcript(
        [
            (0.0, 5.0, "Vectors here."),
            (10.0, 15.0, "Vectors again."),
            (20.0, 25.0, "Other."),
        ]
    )
    retriever = Retriever(transcript_passages(lecture))
    claim = Claim(id=1, text="vectors", kind="key_point", source="s", time=12.0)
    assert retriever.top_k(claim, 1)[0] == 1


def test_retriever_on_empty_transcript():
    retriever = Retriever([])
    assert retriever.top_k(Claim(id=1, text="x", kind="quiz", source="s"), 3) == []


def test_select_evidence_round_robin_under_budget():
    passages = transcript_passages(LECTURE)  # each passage is 9-11 words
    # Best passages are always kept; second-rank ones only if they fit.
    assert select_evidence([[3, 0], [1, 2]], passages, budget=25) == [1, 3]
    assert select_evidence([[3, 0], [1, 2]], passages, budget=100) == [0, 1, 2, 3]
    assert select_evidence([[1], [1]], passages, budget=5) == [1]


def test_lexical_support_and_verdict():
    passages = transcript_passages(LECTURE)
    claim = Claim(
        id=1, text="The determinant is a product.", kind="key_point", source="s"
    )
    assert lexical_support(claim, [passages[1]]) == 1.0
    assert lexical_support(claim, [passages[0]]) == 0.0
    assert lexical_support(claim, []) == 0.0
    assert lexical_verdict(0.9) == "supported"
    assert lexical_verdict(0.6) == "partial"
    assert lexical_verdict(0.2) == "unsupported"


# --- judge reply validation ----------------------------------------------------


def test_validate_judge_reply_accepts_the_requested_shape():
    out = validate_judge_reply(
        verdict_reply((1, "supported", "q"), (2, "contradicted", "r")), 2
    )
    assert out[1].verdict == "supported"
    assert out[2].verdict == "contradicted"


def test_validate_judge_reply_accepts_common_variants():
    reply = json.dumps(
        {
            "results": [
                {"id": "1", "label": "Partially Supported", "evidence": "x"},
                {"claim_id": "#2", "verdict": "not_supported", "quote": None},
                {"statement": "claim 3", "verdict": "Yes."},
            ]
        }
    )
    out = validate_judge_reply(reply, 3)
    assert [out[i].verdict for i in (1, 2, 3)] == [
        "partial",
        "unsupported",
        "supported",
    ]
    assert out[1].quote == "x"
    assert out[2].quote == ""


def test_validate_judge_reply_keeps_first_of_repeated_numbers():
    reply = verdict_reply((1, "supported", ""), (1, "unsupported", ""))
    assert validate_judge_reply(reply, 1)[1].verdict == "supported"


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (verdict_reply((1, "supported", "")), r"no verdict for statement\(s\) \[2\]"),
        (verdict_reply((1, "yes", ""), (3, "yes", "")), "does not exist"),
        (verdict_reply((1, "maybe", ""), (2, "yes", "")), "wrong JSON shape"),
        ("not json", "Expecting value"),
    ],
)
def test_validate_judge_reply_rejects_bad_replies(reply, message):
    with pytest.raises(ValueError, match=message):
        validate_judge_reply(reply, 2)


def test_quote_found_tolerates_small_slips_but_not_inventions():
    passages = transcript_passages(LECTURE)
    assert quote_found("the PRODUCT of its eigenvalues!", passages)
    # One word changed in a 10-word quote: still 90% contiguous.
    assert quote_found(
        "The determinant of the matrix is the product of its eigenvalues", passages
    )
    assert not quote_found("eigenvalues are always positive numbers", passages)
    assert not quote_found("", passages)


# --- metrics -------------------------------------------------------------------


def test_cohen_kappa_known_values():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    # Agree on 2 of 4 with 50% yes-rates each: exactly chance level.
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohen_kappa([True, True], [True, True]) is None
    assert cohen_kappa([], []) is None


def judged(
    verdict: str, *, verified: bool | None = None, lex: float = 1.0, kind="key_point"
):
    return JudgedClaim(
        id=1,
        text="t",
        kind=kind,
        source="s",
        verdict=verdict,
        quote_verified=verified,
        lexical_support=lex,
    )


def test_summarize_scores():
    claims = [
        judged("supported", verified=True),
        judged("supported", verified=False),
        judged("partial", verified=True, lex=0.0),
        judged("contradicted", verified=True, kind="takeaway"),
        judged("unjudged"),
    ]
    s = summarize(claims, compare_lexical=True)
    assert s.judged == 4
    assert s.counts == {
        "supported": 2,
        "partial": 1,
        "unsupported": 0,
        "contradicted": 1,
        "unjudged": 1,
    }
    assert s.faithfulness == pytest.approx(2.5 / 4)
    assert s.strict_faithfulness == pytest.approx(2.0 / 4)
    assert s.hallucination_rate == pytest.approx(1 / 4)
    assert s.quote_verified_rate == pytest.approx(3 / 4)
    # Judge yes/no: Y Y Y N; lexical: Y Y N Y -> agree on 2 of 4.
    assert s.lexical_agreement == pytest.approx(0.5)
    assert s.by_kind == {"key_point": pytest.approx(2.5 / 3), "takeaway": 0.0}


def test_summarize_with_nothing_judged():
    s = summarize([judged("unjudged")], compare_lexical=True)
    assert s.faithfulness is None
    assert s.hallucination_rate is None
    assert s.lexical_agreement is None


# --- end to end ----------------------------------------------------------------


def notes_target(tmp_path, **kwargs):
    write_notes(make_notes(**kwargs), tmp_path)
    return load_target(tmp_path / "notes.json")


def test_evaluate_with_fake_judge(tmp_path):
    target = notes_target(tmp_path)
    judge = FakeLLMClient(
        replies=[
            verdict_reply(
                (1, "supported", "eigenvalue is the factor"),
                (2, "supported", "a matrix stretches a vector"),
                (3, "supported", "product of its eigenvalues"),
                (4, "unsupported", ""),
                (5, "supported", "controls the step size"),
                (6, "supported", "invented quote that is nowhere"),
            )
        ],
        model="judge-model",
        provider="groq",
    )
    report = evaluate_faithfulness(target, LECTURE, judge, target_path="notes.json")
    assert report.llm_calls == 1
    by_id = {c.id: c for c in report.claims}
    assert by_id[4].verdict == "unsupported"
    assert by_id[3].quote_verified is True
    assert by_id[6].quote_verified is False
    assert by_id[4].quote_verified is None
    assert report.summary.faithfulness == pytest.approx(5 / 6)
    assert report.summary.strict_faithfulness == pytest.approx(4.5 / 6)
    assert not report.judge_is_generator
    # The prompt holds both the statements and the timestamped evidence.
    prompt = judge.calls[0][1].content
    assert "4. Eigenvalues are always positive." in prompt
    assert "[00:00:15] The determinant" in prompt
    assert judge.calls[0][0].role == "system"


def test_evaluate_batches_and_repairs(tmp_path):
    target = notes_target(tmp_path)
    judge = FakeLLMClient(
        replies=[
            verdict_reply((1, "supported", "")),  # batch 1: missing 2 and 3
            verdict_reply(*[(n, "supported", "") for n in (1, 2, 3)]),  # repaired
            verdict_reply(*[(n, "partial", "") for n in (1, 2, 3)]),  # batch 2
        ]
    )
    report = evaluate_faithfulness(target, LECTURE, judge, batch_size=3)
    assert report.llm_calls == 3
    assert [c.verdict for c in report.claims] == ["supported"] * 3 + ["partial"] * 3
    # The repair prompt says what was missing.
    assert "statement(s) [2, 3]" in judge.calls[1][-1].content
    assert report.batch_errors == []


def test_failed_batch_is_unjudged_not_fatal(tmp_path):
    target = notes_target(tmp_path)
    judge = FakeLLMClient(replies=["garbage", "still garbage"])
    report = evaluate_faithfulness(target, LECTURE, judge, max_repairs=1)
    assert {c.verdict for c in report.claims} == {"unjudged"}
    assert len(report.batch_errors) == 1
    assert report.summary.faithfulness is None
    assert "Judge errors" in report.to_markdown()


def test_lexical_only_needs_no_client(tmp_path):
    target = notes_target(tmp_path)
    report = evaluate_faithfulness(target, LECTURE, None)
    assert report.llm_calls == 0
    assert report.judge_model == "lexical-overlap"
    assert report.summary.lexical_kappa is None
    by_id = {c.id: c for c in report.claims}
    # "equals" is not in the lecture: 3 of 4 content words -> partial.
    assert by_id[3].verdict == "partial"
    assert by_id[3].lexical_support == 0.75
    assert by_id[4].verdict == "unsupported"  # "always positive" is not said


def test_max_claims_and_bad_arguments(tmp_path):
    target = notes_target(tmp_path)
    report = evaluate_faithfulness(target, LECTURE, None, max_claims=2)
    assert report.summary.claims == 2
    assert report.total_claims == 6
    with pytest.raises(ValueError):
        evaluate_faithfulness(target, LECTURE, None, batch_size=0)


def test_empty_transcript_marks_claims_unjudged(tmp_path):
    target = notes_target(tmp_path)
    empty = Transcript(
        segments=[],
        language="en",
        language_probability=1.0,
        duration=0.0,
        model="small",
    )
    report = evaluate_faithfulness(target, empty, FakeLLMClient())
    assert {c.verdict for c in report.claims} == {"unjudged"}
    assert "transcript is empty" in report.batch_errors[0]


# --- files and schema ------------------------------------------------------------


def test_write_and_load_round_trip(tmp_path):
    target = notes_target(tmp_path)
    report = evaluate_faithfulness(target, LECTURE, None)
    json_path, md_path = write_report(report, tmp_path / "out")
    assert json_path.name == "faithfulness_notes.json"
    assert load_report(json_path) == report
    md = md_path.read_text()
    assert md.startswith("# Faithfulness: Linear algebra")
    assert "Claims to review" in md


def test_load_report_rejects_invalid(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"title": "x"}))
    with pytest.raises(ValueError, match="not a valid faithfulness report"):
        load_report(bad)


def test_markdown_warns_when_judge_is_generator(tmp_path):
    target = notes_target(tmp_path, model="fake-model", provider="fake")
    judge = FakeLLMClient(
        replies=[verdict_reply(*[(n, "supported", "") for n in range(1, 7)])]
    )
    report = evaluate_faithfulness(target, LECTURE, judge)
    assert report.judge_is_generator
    md = report.to_markdown()
    assert "Warning" in md
    assert "_None: every claim is supported" not in md  # no quotes were given
    assert "Claims to review" in md


def test_committed_schema_matches_code():
    # If this fails, regenerate the file:
    #   notetaker faithfulness --schema > schemas/faithfulness.schema.json
    assert json.loads(SCHEMA_PATH.read_text()) == faithfulness_json_schema()


# --- CLI ---------------------------------------------------------------------------


def write_lecture(tmp_path: Path) -> Path:
    LECTURE.save_json(tmp_path / "transcript.json")
    write_notes(make_notes(), tmp_path)
    return tmp_path / "notes.json"


def run_cli(argv: list[str], client=None) -> int:
    args = cli.build_parser().parse_args(["faithfulness", *argv])
    return cli.run_faithfulness(args, client=client)


def test_cli_lexical_only_writes_report(tmp_path, capsys):
    notes_path = write_lecture(tmp_path)
    assert run_cli([str(notes_path), "--lexical-only"]) == 0
    out = capsys.readouterr().out
    assert "no LLM calls" in out
    assert (tmp_path / "faithfulness_notes.md").exists()
    assert load_report(tmp_path / "faithfulness_notes.json").summary.claims == 6


def test_cli_with_fake_judge_and_self_judge_warning(tmp_path, capsys):
    LECTURE.save_json(tmp_path / "transcript.json")
    write_notes(make_notes(model="fake-model", provider="fake"), tmp_path)
    judge = FakeLLMClient(
        replies=[verdict_reply(*[(n, "supported", "") for n in range(1, 7)])]
    )
    out_dir = tmp_path / "reports"
    code = run_cli([str(tmp_path / "notes.json"), "-o", str(out_dir)], client=judge)
    assert code == 0
    out = capsys.readouterr().out
    assert "Warning: the judge is the model" in out
    assert "Faithfulness 100.0%" in out
    assert (out_dir / "faithfulness_notes.json").exists()


def test_cli_explicit_transcript_and_errors(tmp_path, capsys):
    notes_path = write_lecture(tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    LECTURE.save_json(other / "t.json")
    assert (
        run_cli([str(notes_path), "-t", str(other / "t.json"), "--lexical-only"]) == 0
    )

    assert run_cli([]) == 1
    assert run_cli([str(tmp_path / "missing.json")]) == 1
    assert run_cli([str(notes_path), "-t", str(tmp_path / "nope.json")]) == 1
    assert run_cli([str(notes_path), "--batch-size", "0"]) == 1
    assert run_cli([str(notes_path), "--max-claims", "0"]) == 1
    (tmp_path / "junk.json").write_text("{not json")
    assert run_cli([str(tmp_path / "junk.json")]) == 1
    out = capsys.readouterr().out
    assert "no such file" in out


def test_cli_schema_flag_prints_schema(capsys):
    assert run_cli(["--schema"]) == 0
    assert json.loads(capsys.readouterr().out) == faithfulness_json_schema()


def test_main_dispatches_faithfulness(tmp_path):
    notes_path = write_lecture(tmp_path)
    assert cli.main(["faithfulness", str(notes_path), "--lexical-only"]) == 0
