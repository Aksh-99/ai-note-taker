"""Tests for src/notetaker/ask.py and the `notetaker ask` command.

All offline: a HashingEmbedder index stands in for the neural model and a
FakeLLMClient returns scripted replies, so we can test exactly how the code
treats good answers, invented citations, broken JSON and refusals.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from notetaker.ask import (
    MODE_EXTRACTIVE,
    MODE_LLM,
    MODE_NO_MATCH,
    NO_CITATION_ERROR,
    NO_MATCH_TEXT,
    AskError,
    ask,
    build_messages,
    format_sources,
    resolve_citations,
    validate_reply,
)
from notetaker.cli import build_parser, main, run_ask
from notetaker.embeddings import HashingEmbedder
from notetaker.llm import FakeLLMClient
from notetaker.transcribe import Segment, Transcript
from notetaker.vectorstore import VectorStore


def make_transcript(texts: list[str], seconds_each: float = 30.0) -> Transcript:
    segments = [
        Segment(
            id=i,
            start=i * seconds_each,
            end=(i + 1) * seconds_each,
            text=t,
            avg_logprob=-0.2,
            no_speech_prob=0.01,
        )
        for i, t in enumerate(texts)
    ]
    return Transcript(
        segments=segments,
        language="en",
        language_probability=1.0,
        duration=len(texts) * seconds_each,
        model="test",
    )


LA = [
    "Today we study eigenvalues and eigenvectors of a square matrix.",
    "An eigenvector is a vector that the matrix only stretches, never rotates.",
    "To find eigenvalues we solve the characteristic polynomial det(A - lambda I).",
]
ML = [
    "Gradient descent updates the weights against the gradient of the loss.",
    "The learning rate controls the step size of each gradient descent update.",
]


@pytest.fixture
def store() -> VectorStore:
    """Two small lectures; max_words=12 gives one passage per sentence-ish."""
    s = VectorStore(HashingEmbedder(dim=256))
    s.add_transcript(
        make_transcript(LA), "la-05", title="Linear Algebra 5", max_words=12
    )
    s.add_transcript(make_transcript(ML), "ml-02", title="ML 2", max_words=12)
    return s


def reply(answer: str, answerable: bool = True) -> str:
    return json.dumps({"answer": answer, "answerable": answerable})


# ---------------------------------------------------------------------------
# Citation checking and renumbering
# ---------------------------------------------------------------------------


def test_citations_renumbered_in_order_of_first_use() -> None:
    r = resolve_citations("A is true [3]. B too [1]. Again A [3].", n_sources=5)
    assert r.text == "A is true [1]. B too [2]. Again A [1]."
    assert r.order == [3, 1]
    assert r.unknown == []


def test_citation_groups_are_split_and_deduplicated() -> None:
    r = resolve_citations("Both [2, 4]. Same [4,4].", n_sources=4)
    assert r.text == "Both [1][2]. Same [2]."
    assert r.order == [2, 4]


def test_invented_citation_is_reported_and_left_alone() -> None:
    r = resolve_citations("Real [1]. Invented [7].", n_sources=3)
    assert r.text == "Real [1]. Invented [7]."
    assert r.order == [1]
    assert r.unknown == ["[7]"]


def test_math_brackets_are_not_treated_as_citations() -> None:
    # [0, 1] is an interval and [2, 9] has a number out of range: both untouched.
    r = resolve_citations("x lies in [0, 1] and [2, 9] [2].", n_sources=3)
    assert r.text == "x lies in [0, 1] and [2, 9] [1]."
    assert r.order == [2]
    assert r.unknown == []  # multi-number groups are not flagged


def test_no_markers() -> None:
    r = resolve_citations("No citations here.", n_sources=3)
    assert r.order == [] and r.text == "No citations here."


# ---------------------------------------------------------------------------
# Prompt and reply parsing
# ---------------------------------------------------------------------------


def test_format_sources_numbers_passages_with_title_and_span(store) -> None:
    hits = store.search("eigenvector stretches", k=2)
    text = format_sources(hits)
    assert text.startswith("[1] ")
    assert "[2] " in text
    assert hits[0].passage.text.strip() in text
    assert "Linear Algebra 5, 00:00:" in text


def test_build_messages_contains_question_and_sources(store) -> None:
    hits = store.search("eigenvector", k=2)
    system, user = build_messages("  What is an eigenvector? ", hits)
    assert system.role == "system" and "only" in system.content
    assert "Question: What is an eigenvector?" in user.content
    assert format_sources(hits) in user.content


def test_validate_reply_accepts_fenced_json_and_bare_string() -> None:
    fenced = '```json\n{"answer": " Yes [1]. ", "answerable": true}\n```'
    assert validate_reply(fenced).answer == "Yes [1]."
    assert validate_reply('"Just text [1]"').answer == "Just text [1]"


@pytest.mark.parametrize("bad", ["not json", '{"answerable": true}', '{"answer": ""}'])
def test_validate_reply_rejects_bad_replies(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_reply(bad)


# ---------------------------------------------------------------------------
# ask(): the main flow
# ---------------------------------------------------------------------------


def test_answer_with_renumbered_citations_from_index_metadata(store) -> None:
    fake = FakeLLMClient(replies=[reply("It is only stretched [2]. See also [1].")])
    answer = ask(store, "What is an eigenvector?", fake, k=3)
    assert answer.mode == MODE_LLM
    assert answer.answerable is True
    assert answer.text == "It is only stretched [1]. See also [2]."
    assert answer.llm_calls == 1 and len(fake.calls) == 1
    assert [c.number for c in answer.citations] == [1, 2]
    # Citation 1 is the passage the model called [2]: retrieved rank 2.
    assert answer.citations[0].passage_id == answer.retrieved[1].passage_id
    assert answer.citations[1].passage_id == answer.retrieved[0].passage_id
    # Titles and times come from the index, never from the model.
    for c in answer.citations:
        assert c.title in {"Linear Algebra 5", "ML 2"}
        assert c.label().endswith(f"[{c.timestamp}]")
    assert answer.warnings == []
    assert answer.provider == "fake" and answer.model == "fake-model"


def test_retrieved_keeps_every_passage_by_rank(store) -> None:
    fake = FakeLLMClient(replies=[reply("Yes [1].")])
    answer = ask(store, "eigenvalues", fake, k=4)
    assert [c.number for c in answer.retrieved] == [1, 2, 3, 4]
    scores = [c.score for c in answer.retrieved]
    assert scores == sorted(scores, reverse=True)


def test_invented_citation_gives_a_warning(store) -> None:
    fake = FakeLLMClient(replies=[reply("Stretched [1], and more [9].")])
    answer = ask(store, "eigenvector", fake, k=3)
    assert len(answer.citations) == 1
    assert any("[9]" in w for w in answer.warnings)


def test_broken_json_is_repaired(store) -> None:
    fake = FakeLLMClient(replies=["Sure! Here it is", reply("Stretched [1].")])
    answer = ask(store, "eigenvector", fake, k=3, max_repairs=1)
    assert answer.llm_calls == 2
    assert answer.text == "Stretched [1]."
    # The repair turn shows the model its bad reply and the reason.
    second = fake.calls[1]
    assert second[-2].role == "assistant" and second[-2].content == "Sure! Here it is"
    assert "could not be used" in second[-1].content


def test_uncited_answer_is_sent_back_for_citations(store) -> None:
    fake = FakeLLMClient(replies=[reply("It is stretched."), reply("Stretched [1].")])
    answer = ask(store, "eigenvector", fake, k=3)
    assert answer.llm_calls == 2
    assert NO_CITATION_ERROR in fake.calls[1][-1].content
    assert len(answer.citations) == 1 and answer.warnings == []


def test_uncited_answer_kept_with_warning_when_out_of_repairs(store) -> None:
    fake = FakeLLMClient(replies=[reply("It is stretched.")])
    answer = ask(store, "eigenvector", fake, k=3, max_repairs=0)
    assert answer.text == "It is stretched."
    assert answer.citations == []
    assert any("did not cite" in w for w in answer.warnings)


def test_uncited_answer_survives_a_broken_repair(store) -> None:
    fake = FakeLLMClient(replies=[reply("It is stretched."), "{broken"])
    answer = ask(store, "eigenvector", fake, k=3, max_repairs=1)
    assert answer.llm_calls == 2
    assert answer.text == "It is stretched."
    assert any("did not cite" in w for w in answer.warnings)


def test_always_broken_json_raises(store) -> None:
    fake = FakeLLMClient(replies=["nope"])
    with pytest.raises(AskError, match="2 attempt"):
        ask(store, "eigenvector", fake, k=3, max_repairs=1)


def test_model_can_say_not_answerable(store) -> None:
    fake = FakeLLMClient(replies=[reply("The lectures do not cover this.", False)])
    answer = ask(store, "eigenvector history", fake, k=3)
    assert answer.answerable is False
    assert answer.citations == []
    assert answer.llm_calls == 1  # no repair asked for: no citation is fine here
    assert answer.warnings == []


def test_min_score_skips_the_llm(store) -> None:
    fake = FakeLLMClient(replies=[reply("should not be used [1]")])
    answer = ask(store, "eigenvector", fake, min_score=1.01)
    assert answer.mode == MODE_NO_MATCH
    assert answer.text == NO_MATCH_TEXT and answer.answerable is False
    assert fake.calls == []
    assert answer.retrieved  # still shown, for transparency
    assert "below --min-score" in answer.warnings[0]


def test_no_overlap_with_any_lecture_is_no_match(store) -> None:
    # With no shared words the hashing embedder's similarity is near 0 (not
    # exactly 0: two different words can land on the same hashed coordinate).
    fake = FakeLLMClient()
    answer = ask(store, "zebra xylophone", fake, min_score=0.3)
    assert answer.retrieved[0].score < 0.3
    assert answer.mode == MODE_NO_MATCH and fake.calls == []


def test_empty_index_is_no_match() -> None:
    empty = VectorStore(HashingEmbedder(dim=64))
    answer = ask(empty, "anything", FakeLLMClient())
    assert answer.mode == MODE_NO_MATCH and answer.retrieved == []
    assert answer.warnings == []


def test_extractive_mode_needs_no_client(store) -> None:
    answer = ask(store, "learning rate step size", use_llm=False)
    assert answer.mode == MODE_EXTRACTIVE
    assert answer.text.endswith("[1]")
    assert answer.citations[0].lecture_id == "ml-02"
    assert answer.citations[0].text.strip() in answer.text


def test_lecture_filter(store) -> None:
    fake = FakeLLMClient(replies=[reply("Yes [1].")])
    answer = ask(store, "gradient eigenvalues", fake, lecture_ids=["la-05"])
    assert {c.lecture_id for c in answer.retrieved} == {"la-05"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"question": "   "}, "empty"),
        ({"max_repairs": -1}, "max_repairs"),
        ({"client": None}, "needs an LLM client"),
    ],
)
def test_bad_arguments(store, kwargs, message) -> None:
    params = {"question": "eigenvector", "client": FakeLLMClient(), **kwargs}
    with pytest.raises(ValueError, match=message):
        ask(store, params.pop("question"), params.pop("client"), **params)


def test_markdown_and_dict(store) -> None:
    fake = FakeLLMClient(replies=[reply("Stretched [1], bad [8].")])
    answer = ask(store, "eigenvector", fake, k=2)
    md = answer.to_markdown()
    assert md.startswith("## eigenvector\n\nStretched [1], bad [8].")
    assert "**Sources**" in md and "1. Linear Algebra 5 [00:" in md
    assert "> Note: ignored citation" in md
    data = answer.to_dict()
    assert data["mode"] == "llm" and data["answerable"] is True
    assert data["citations"][0]["timestamp"] == answer.citations[0].timestamp
    assert len(data["retrieved"]) == 2
    json.dumps(data)  # everything is JSON-serialisable


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def index_dir(tmp_path: Path, store: VectorStore) -> Path:
    return store.save(tmp_path / "index")


def test_cli_no_llm(index_dir: Path, capsys) -> None:
    code = main(["ask", "learning rate", "-i", str(index_dir), "--no-llm"])
    out = capsys.readouterr().out
    assert code == 0
    assert "**Sources**" in out and "ML 2 [00:" in out
    assert "extractive, no LLM call" in out


def test_cli_missing_index(tmp_path: Path, capsys) -> None:
    code = main(["ask", "q", "-i", str(tmp_path / "none"), "--no-llm"])
    out = capsys.readouterr().out
    assert code == 1 and "notetaker ingest" in out


def test_cli_unknown_lecture(index_dir: Path, capsys) -> None:
    args = ["ask", "q", "-i", str(index_dir), "--no-llm", "-l", "bio-01"]
    assert main(args) == 1
    out = capsys.readouterr().out
    assert "not in the index: bio-01" in out and "la-05" in out


def test_cli_bad_top_k(index_dir: Path, capsys) -> None:
    assert main(["ask", "q", "-i", str(index_dir), "--no-llm", "-k", "0"]) == 1


def test_cli_with_client_writes_outputs(index_dir: Path, tmp_path: Path, capsys):
    md_path = tmp_path / "out" / "answer.md"
    args = build_parser().parse_args(
        ["ask", "What is an eigenvector?", "-i", str(index_dir), "-o", str(md_path)]
    )
    fake = FakeLLMClient(replies=[reply("Only stretched [1].")])
    assert run_ask(args, client=fake) == 0
    out = capsys.readouterr().out
    assert "Only stretched [1]." in out and "1 LLM call(s)" in out
    assert md_path.read_text(encoding="utf-8").startswith("## What is an eigen")


def test_cli_json_output_is_pure_json(index_dir: Path, tmp_path: Path, capsys):
    json_path = tmp_path / "answer.json"
    args = build_parser().parse_args(
        ["ask", "eigenvector", "-i", str(index_dir), "--json", "-o", str(json_path)]
    )
    assert run_ask(args, client=FakeLLMClient(replies=[reply("Yes [1].")])) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)  # stdout parses as JSON: no status lines
    assert data["answer"] == "Yes [1]."
    assert f"Wrote {json_path}" in captured.err
    assert json.loads(json_path.read_text(encoding="utf-8"))["mode"] == "llm"


def test_cli_ask_error_exit_code(index_dir: Path, capsys) -> None:
    args = build_parser().parse_args(["ask", "eigenvector", "-i", str(index_dir)])
    assert run_ask(args, client=FakeLLMClient(replies=["broken"])) == 1
    assert "no usable answer" in capsys.readouterr().out


def test_cli_record_then_replay(index_dir: Path, tmp_path: Path, capsys) -> None:
    cassette = tmp_path / "ask.json"
    question = "What is an eigenvector?"
    args = build_parser().parse_args(
        ["ask", question, "-i", str(index_dir), "--record-llm", str(cassette)]
    )
    fake = FakeLLMClient(replies=[reply("Only stretched [1].")])
    assert run_ask(args, client=fake) == 0
    recorded = capsys.readouterr().out
    assert cassette.exists()

    # Replay goes through main(): no client, no settings, no network.
    code = main(["ask", question, "-i", str(index_dir), "--replay-llm", str(cassette)])
    replayed = capsys.readouterr().out
    assert code == 0
    answer_part = recorded.split("(")[0]
    assert answer_part in replayed

    # A different question is a different prompt: the cassette has no reply.
    code = main(
        ["ask", "learning rate", "-i", str(index_dir), "--replay-llm", str(cassette)]
    )
    assert code == 1


def test_cli_replay_missing_cassette(index_dir: Path, tmp_path: Path, capsys):
    missing = tmp_path / "nope.json"
    assert main(["ask", "q", "-i", str(index_dir), "--replay-llm", str(missing)]) == 1
    assert "no such cassette" in capsys.readouterr().out
