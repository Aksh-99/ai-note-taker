"""Tests for the retrieval eval (Step 5.4): labels, metrics, reports and CLI.

Everything runs offline with the model-free HashingEmbedder, so no model is
downloaded. The metric functions are checked against hand-computed values
and, for the chance baseline, against brute force over every permutation.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import pytest

from notetaker.cli import main
from notetaker.embeddings import HashingEmbedder
from notetaker.retrieval_eval import (
    LabeledQuestion,
    QuestionSet,
    RelevantSpan,
    RetrievalEvalError,
    build_store,
    chance_mrr,
    chance_recall_at,
    check_labels,
    evaluate,
    judge,
    load_corpus_titles,
    load_questions,
    to_json,
    to_markdown,
)
from notetaker.transcribe import Segment, Transcript
from notetaker.vectorstore import Passage, SearchHit, VectorStore

ROOT = Path(__file__).resolve().parent.parent
DEMO_QUESTIONS = ROOT / "eval" / "retrieval" / "questions.json"


def passage(lecture: str, start: float, end: float, n: int = 0) -> Passage:
    return Passage(f"{lecture}#{n:04d}", lecture, start, end, f"text {n}")


def hits_for(passages: list[Passage]) -> list[SearchHit]:
    return [SearchHit(p, 1.0 - 0.1 * i, i + 1) for i, p in enumerate(passages)]


def question(*spans: tuple[str, float, float], qid: str = "q1") -> LabeledQuestion:
    return LabeledQuestion(qid, "?", tuple(RelevantSpan(*s) for s in spans))


def write_questions(path: Path, questions: list[dict], **extra) -> Path:
    path.write_text(json.dumps({"questions": questions, **extra}), encoding="utf-8")
    return path


def make_transcript(sentences: list[str], seconds_each: float = 10.0) -> Transcript:
    segments = [
        Segment(i, i * seconds_each, (i + 1) * seconds_each, text, -0.2, 0.01)
        for i, text in enumerate(sentences)
    ]
    return Transcript(segments, "en", 1.0, len(sentences) * seconds_each, "test")


# ---------------------------------------------------------------------------
# Relevance by time overlap
# ---------------------------------------------------------------------------


class TestOverlap:
    def test_overlapping_passage_in_same_lecture_is_relevant(self):
        span = RelevantSpan("a", 10.0, 20.0)
        assert span.overlaps(passage("a", 15.0, 30.0))
        assert span.overlaps(passage("a", 0.0, 12.0))
        assert span.overlaps(passage("a", 12.0, 18.0))  # inside the span
        assert span.overlaps(passage("a", 0.0, 100.0))  # contains the span

    def test_touching_at_one_point_does_not_count(self):
        span = RelevantSpan("a", 10.0, 20.0)
        assert not span.overlaps(passage("a", 20.0, 30.0))
        assert not span.overlaps(passage("a", 0.0, 10.0))

    def test_other_lecture_never_counts(self):
        assert not RelevantSpan("a", 10.0, 20.0).overlaps(passage("b", 10.0, 20.0))

    def test_question_relevant_if_any_span_matches(self):
        q = question(("a", 0, 5), ("b", 50, 60))
        assert q.is_relevant(passage("b", 55, 70))
        assert not q.is_relevant(passage("b", 0, 40))


# ---------------------------------------------------------------------------
# Per-question metrics
# ---------------------------------------------------------------------------


class TestJudge:
    def test_first_rank_and_reciprocal_rank(self):
        ps = [passage("x", 0, 10, 0), passage("a", 100, 110, 1), passage("a", 5, 15, 2)]
        result = judge(question(("a", 0, 20)), hits_for(ps), ps)
        assert result.relevant_flags == [False, False, True]
        assert result.first_rank == 3
        assert result.reciprocal_rank == pytest.approx(1 / 3)
        assert result.recall_at(1) == 0.0
        assert result.recall_at(2) == 0.0
        assert result.recall_at(3) == 1.0

    def test_not_found_scores_zero(self):
        ps = [passage("x", 0, 10)]
        result = judge(question(("a", 0, 20)), hits_for(ps), ps + [passage("a", 0, 5)])
        assert result.first_rank is None
        assert result.reciprocal_rank == 0.0
        assert result.recall_at(10) == 0.0
        assert result.relevant_in_index == 1

    def test_multi_span_recall_is_fraction_of_spans_found(self):
        # Four spans; the top 3 reach two of them, the 5th hit reaches a third.
        q = question(("a", 0, 10), ("b", 0, 10), ("c", 0, 10), ("d", 0, 10))
        ps = [
            passage("b", 0, 5, 0),
            passage("x", 0, 5, 1),
            passage("a", 0, 5, 2),
            passage("b", 5, 9, 3),  # second hit on span b does not add recall
            passage("c", 0, 5, 4),
        ]
        result = judge(q, hits_for(ps), ps)
        assert result.span_ranks == [3, 1, 5, None]
        assert result.recall_at(1) == 0.25
        assert result.recall_at(3) == 0.5
        assert result.recall_at(5) == 0.75
        assert result.first_rank == 1


# ---------------------------------------------------------------------------
# Chance baseline
# ---------------------------------------------------------------------------


def brute_force_chance(n: int, r: int, k: int, depth: int) -> tuple[float, float]:
    """Average recall@k and MRR@depth over every ordering of n passages.

    Passages 0..r-1 are the relevant ones. Tiny n only (n! orderings).
    """
    recall_total = mrr_total = 0.0
    orders = list(itertools.permutations(range(n)))
    for order in orders:
        ranks = [i + 1 for i, p in enumerate(order) if p < r]
        recall_total += 1.0 if ranks and ranks[0] <= k else 0.0
        mrr_total += 1.0 / ranks[0] if ranks and ranks[0] <= depth else 0.0
    return recall_total / len(orders), mrr_total / len(orders)


class TestChance:
    @pytest.mark.parametrize(
        ("n", "r", "k", "depth"),
        [(5, 1, 1, 5), (5, 2, 2, 3), (6, 3, 3, 6), (6, 1, 4, 2), (4, 4, 1, 4)],
    )
    def test_matches_brute_force(self, n, r, k, depth):
        recall, mrr = brute_force_chance(n, r, k, depth)
        assert chance_recall_at(n, r, k) == pytest.approx(recall)
        assert chance_mrr(n, r, depth) == pytest.approx(mrr)

    def test_simple_closed_forms(self):
        # One relevant passage among n: recall@k = k / n, MRR = H_n / n.
        assert chance_recall_at(20, 1, 5) == pytest.approx(5 / 20)
        harmonic = sum(1 / i for i in range(1, 21))
        assert chance_mrr(20, 1, 20) == pytest.approx(harmonic / 20)

    def test_edge_cases(self):
        assert chance_recall_at(10, 0, 5) == 0.0
        assert chance_mrr(10, 0, 5) == 0.0
        assert chance_recall_at(3, 1, 10) == 1.0  # k larger than the index
        assert chance_recall_at(10, 8, 3) == 1.0  # cannot miss with 3 picks


# ---------------------------------------------------------------------------
# Loading question files
# ---------------------------------------------------------------------------


GOOD = {
    "id": "q1",
    "question": "What is X?",
    "tags": ["lexical"],
    "relevant": [{"lecture_id": "a", "start": 1, "end": 2}],
}


class TestLoadQuestions:
    def test_loads_and_resolves_paths_relative_to_file(self, tmp_path):
        path = write_questions(
            tmp_path / "q.json", [GOOD], corpus="lectures", titles="t.json"
        )
        qs = load_questions(path)
        assert qs.questions[0] == LabeledQuestion(
            "q1", "What is X?", (RelevantSpan("a", 1.0, 2.0),), ("lexical",)
        )
        assert qs.corpus == tmp_path / "lectures"
        assert qs.titles == tmp_path / "t.json"
        assert qs.lecture_ids() == {"a"}
        assert qs.tags() == ["lexical"]

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"id": ""}, "missing a string 'id'"),
            ({"question": " "}, "missing the 'question'"),
            ({"relevant": []}, "non-empty list"),
            ({"relevant": [{"lecture_id": "a", "start": 1}]}, "needs lecture_id"),
            ({"relevant": [{"lecture_id": "a", "start": 5, "end": 2}]}, "start < end"),
            ({"relevant": [{"lecture_id": "a", "start": -1, "end": 2}]}, "start < end"),
            ({"tags": "lexical"}, "list of strings"),
        ],
    )
    def test_rejects_bad_questions(self, tmp_path, change, message):
        path = write_questions(tmp_path / "q.json", [{**GOOD, **change}])
        with pytest.raises(RetrievalEvalError, match=message):
            load_questions(path)

    def test_rejects_duplicate_ids_and_empty_files(self, tmp_path):
        with pytest.raises(RetrievalEvalError, match="duplicate"):
            load_questions(write_questions(tmp_path / "d.json", [GOOD, GOOD]))
        with pytest.raises(RetrievalEvalError, match="no questions"):
            load_questions(write_questions(tmp_path / "e.json", []))

    def test_missing_and_invalid_files(self, tmp_path):
        with pytest.raises(RetrievalEvalError, match="not found"):
            load_questions(tmp_path / "nope.json")
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(RetrievalEvalError, match="not valid JSON"):
            load_questions(bad)
        bad.write_text("[]", encoding="utf-8")
        with pytest.raises(RetrievalEvalError, match="'questions' list"):
            load_questions(bad)

    def test_titles_fallbacks(self, tmp_path):
        qs = QuestionSet([], titles=tmp_path / "missing.json")
        assert load_corpus_titles(qs, None) == {}  # optional file may be absent
        given = tmp_path / "t.json"
        given.write_text(json.dumps({"a": "Lecture A"}), encoding="utf-8")
        assert load_corpus_titles(qs, given) == {"a": "Lecture A"}


# ---------------------------------------------------------------------------
# Evaluating an index
# ---------------------------------------------------------------------------


@pytest.fixture
def small_store() -> VectorStore:
    store = VectorStore(HashingEmbedder())
    store.add_transcript(
        make_transcript(
            [
                "eigenvalues solve the determinant equation",
                "the trace equals the sum of eigenvalues",
                "symmetric matrices have real eigenvalues",
            ]
        ),
        lecture_id="la",
        max_words=6,
    )
    store.add_transcript(
        make_transcript(
            [
                "gradient descent takes small steps downhill",
                "the learning rate controls the step size",
                "overfitting means memorizing the training noise",
            ]
        ),
        lecture_id="ml",
        max_words=7,
    )
    return store


def small_questions() -> QuestionSet:
    return QuestionSet(
        [
            LabeledQuestion(
                "q1", "what does the learning rate control",
                (RelevantSpan("ml", 10, 20),),
                ("lexical",),
            ),
            LabeledQuestion(
                "q2", "trace and sum of eigenvalues", (RelevantSpan("la", 10, 20),),
                ("lexical",),
            ),
            LabeledQuestion(
                "q3", "a question with no shared words at all",
                (RelevantSpan("la", 20, 30), RelevantSpan("ml", 20, 30)),
                ("multi",),
            ),
        ],
        path=Path("small.json"),
    )  # fmt: skip


class TestEvaluate:
    def test_scores(self, small_store):
        report = evaluate(small_store, small_questions(), ks=[3, 1])
        assert report.ks == (1, 3)  # sorted, deduplicated
        assert report.n_passages == 6
        assert report.n_lectures == 2
        by_id = {r.question.id: r for r in report.results}
        assert by_id["q1"].first_rank == 1
        assert by_id["q2"].first_rank == 1
        assert report.recall(1) >= 2 / 3
        assert report.recall(3) >= report.recall(1)  # recall never falls with k
        assert report.recall(1, tag="lexical") == 1.0
        assert report.mrr(tag="lexical") == 1.0
        assert report.count("multi") == 1
        assert 0 < report.chance_recall[1] < report.chance_recall[3] < 1
        assert report.mean_latency_ms() >= 0
        assert report.label == "hashing-512"

    def test_rejects_bad_k(self, small_store):
        with pytest.raises(ValueError, match="positive"):
            evaluate(small_store, small_questions(), ks=[0, 5])
        with pytest.raises(ValueError, match="positive"):
            evaluate(small_store, small_questions(), ks=[])

    def test_unknown_lecture_is_an_error_that_lists_indexed_ones(self, small_store):
        qs = QuestionSet([question(("typo", 0, 10))])
        with pytest.raises(RetrievalEvalError, match=r"\['typo'\].*\['la', 'ml'\]"):
            check_labels(small_store, qs)

    def test_span_beyond_the_transcript_is_an_error(self, small_store):
        qs = QuestionSet([question(("la", 500, 510), qid="late")])
        with pytest.raises(RetrievalEvalError, match="late"):
            evaluate(small_store, qs)

    def test_label_includes_passage_size(self, small_store):
        report = evaluate(
            small_store,
            small_questions(),
            settings={"max_words": 60, "max_seconds": 90},
        )
        assert report.label == "hashing-512 @ 60w"
        assert report.to_dict()["settings"] == {"max_words": 60, "max_seconds": 90}

    def test_json_and_markdown(self, small_store):
        qs = small_questions()
        a = evaluate(small_store, qs, settings={"max_words": 6, "max_seconds": 90.0})
        b = evaluate(small_store, qs, ks=[1, 3, 5, 10])
        b.ks = a.ks  # same columns for the shared table
        md = to_markdown([a, b], qs)
        assert "| `hashing-512` | 6 (6 words / 90 s) |" in md
        assert "| chance | 6 |" in md
        assert md.count("| chance |") == 1  # same passage count, one row
        assert "lexical (n=2)" in md and "multi (n=1)" in md
        assert "| q3 | a question with no shared words at all | multi |" in md
        assert "spans)" in md  # multi-span questions show spans found

        data = json.loads(to_json([a], qs))
        run = data["runs"][0]
        assert data["questions"] == 3
        assert set(run["recall"]) == {"1", "3", "5", "10"}
        assert run["per_question"][0]["top"][0]["relevant"] is True
        assert run["per_question"][0]["first_rank"] == 1

    def test_markdown_needs_a_report(self):
        with pytest.raises(ValueError):
            to_markdown([], small_questions())


class TestBuildStore:
    def test_ids_titles_and_passage_size_follow_ingest(self, tmp_path):
        for name, sentences in {
            "la-05": ["one two three"] * 6,
            "ml-02": ["x y"] * 4,
        }.items():
            make_transcript(sentences).save_json(tmp_path / name / "transcript.json")
        store = build_store(
            tmp_path, embedder=HashingEmbedder(), titles={"la-05": "Eigen"}, max_words=6
        )
        assert store.lectures() == ["la-05", "ml-02"]
        assert store.passages[0].title == "Eigen"
        assert store.passages[-1].title == ""
        bigger = build_store(tmp_path, embedder_name="hashing-512", max_words=100)
        assert len(bigger) < len(store)

    def test_empty_folder(self, tmp_path):
        with pytest.raises(RetrievalEvalError, match="no files"):
            build_store(tmp_path, embedder=HashingEmbedder())


# ---------------------------------------------------------------------------
# The committed demo set
# ---------------------------------------------------------------------------


def load_demo_builder():
    """Import scripts/build_retrieval_demo.py (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "build_retrieval_demo", ROOT / "scripts" / "build_retrieval_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDemoSet:
    def test_committed_files_match_the_generator(self):
        # If this fails, run: python scripts/build_retrieval_demo.py
        for path, content in load_demo_builder().build().items():
            assert path.read_text(encoding="utf-8") == content, path

    def test_demo_eval_runs_and_beats_chance(self):
        qs = load_questions(DEMO_QUESTIONS)
        assert len(qs.questions) == 26
        store = build_store(
            qs.corpus, embedder_name="hashing-512", titles=load_corpus_titles(qs, None)
        )
        report = evaluate(store, qs)
        assert report.n_lectures == 4
        assert store.passages[0].title  # titles were picked up
        recalls = [report.recall(k) for k in report.ks]
        assert recalls == sorted(recalls)
        for k in report.ks:
            assert report.recall(k) > report.chance_recall[k]
        assert report.mrr() > 2 * report.chance_mrr
        # Word matching does well when the question shares words with the
        # answer and worse on paraphrases: the gap neural embeddings target.
        assert report.recall(5, "lexical") > report.recall(5, "paraphrase")
        assert report.recall(5, "lexical") >= 0.9


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_demo_run_with_report(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(ROOT)  # default question path is relative to the repo
        out = tmp_path / "reports" / "retrieval.md"
        code = main(
            [
                "eval-retrieval",
                "--embedder", "hashing-512",
                "--max-words", "120", "60",
                "-k", "1", "5",
                "--show-misses",
                "-o", str(out),
            ]
        )  # fmt: skip
        printed = capsys.readouterr().out
        assert code == 0
        assert "26 labeled questions" in printed
        assert "hashing-512 @ 120w" in printed and "hashing-512 @ 60w" in printed
        assert printed.count("chance (") == 2  # two passage counts
        assert "Not fully found in the top 5" in printed
        assert out.exists() and out.with_suffix(".json").exists()
        data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        assert len(data["runs"]) == 2
        assert data["runs"][0]["ks"] == [1, 5]

    def test_existing_index(self, tmp_path, capsys, small_store):
        small_store.save(tmp_path / "index")
        qpath = tmp_path / "q.json"
        qpath.write_text(
            json.dumps(
                {
                    "questions": [
                        {
                            "id": "q1",
                            "question": "learning rate step size",
                            "relevant": [{"lecture_id": "ml", "start": 10, "end": 20}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        code = main(
            ["eval-retrieval", "-q", str(qpath), "-i", str(tmp_path / "index"),
             "--embedder", "ignored"]
        )  # fmt: skip
        printed = capsys.readouterr().out
        assert code == 0
        assert "--embedder is ignored" in printed
        assert "R@1=1.000" in printed and "MRR=1.000" in printed

    def test_errors_exit_1(self, tmp_path, capsys):
        assert main(["eval-retrieval", "-q", str(tmp_path / "none.json")]) == 1
        assert "not found" in capsys.readouterr().out
        qpath = write_questions(tmp_path / "q.json", [GOOD])  # no corpus key
        assert main(["eval-retrieval", "-q", str(qpath)]) == 1
        assert "--index or --transcripts" in capsys.readouterr().out
        assert main(["eval-retrieval", "-q", str(qpath), "-i", str(tmp_path)]) == 1
        assert "no index" in capsys.readouterr().out

    def test_index_and_transcripts_are_exclusive(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["eval-retrieval", "-i", "a", "--transcripts", "b"])
