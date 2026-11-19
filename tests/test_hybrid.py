"""Tests for hybrid search (Step 5.5): fusion maths, the retriever, the eval CLI.

Offline only: vector search uses the model-free HashingEmbedder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from notetaker import cli
from notetaker.bm25 import BM25Index
from notetaker.embeddings import HashingEmbedder
from notetaker.hybrid import (
    RETRIEVERS,
    HybridRetriever,
    Retriever,
    linear_fusion,
    make_retriever,
    min_max,
    reciprocal_rank_fusion,
)
from notetaker.retrieval_eval import (
    build_store,
    evaluate,
    load_corpus_titles,
    load_questions,
)
from notetaker.vectorstore import Passage, SearchHit, VectorStore

ROOT = Path(__file__).resolve().parent.parent
DEMO_QUESTIONS = ROOT / "eval" / "retrieval" / "questions.json"


def p(name: str) -> Passage:
    return Passage(name, "lec", 0.0, 1.0, name)


def ranking(*names: str, scores: list[float] | None = None) -> list[SearchHit]:
    scores = scores or [1.0 - 0.1 * i for i in range(len(names))]
    return [
        SearchHit(p(n), s, i + 1)
        for i, (n, s) in enumerate(zip(names, scores, strict=True))
    ]


def ids(hits: list[SearchHit]) -> list[str]:
    return [h.passage.passage_id for h in hits]


class TestRRF:
    def test_scores_match_formula(self):
        fused = reciprocal_rank_fusion(
            [ranking("a", "b"), ranking("b", "c")], k=3, c=60
        )
        score = {h.passage.passage_id: h.score for h in fused}
        assert score["b"] == pytest.approx(1 / 62 + 1 / 61)  # rank 2 and rank 1
        assert score["a"] == pytest.approx(1 / 61)
        assert score["c"] == pytest.approx(1 / 62)
        # b is in both lists, so it wins.
        assert ids(fused) == ["b", "a", "c"]
        assert [h.rank for h in fused] == [1, 2, 3]

    def test_agreement_beats_one_strong_vote(self):
        # "x" is first in one list only; "y" is third in both.
        fused = reciprocal_rank_fusion(
            [ranking("x", "q", "y"), ranking("r", "s", "y")], k=1
        )
        assert ids(fused) == ["y"]

    def test_ties_broken_by_best_rank_then_id(self):
        # a and c both get exactly 1/61; tie broken by passage id.
        fused = reciprocal_rank_fusion([ranking("c"), ranking("a")], k=2)
        assert ids(fused) == ["a", "c"]

    def test_weights(self):
        fused = reciprocal_rank_fusion(
            [ranking("a"), ranking("b")], k=2, weights=[1.0, 3.0]
        )
        assert ids(fused) == ["b", "a"]

    def test_k_truncates_and_empty_lists(self):
        assert len(reciprocal_rank_fusion([ranking("a", "b", "c")], k=2)) == 2
        assert reciprocal_rank_fusion([[], []], k=5) == []

    def test_bad_arguments(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([ranking("a")], k=1, c=-1)
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([ranking("a")], k=1, weights=[1.0, 2.0])


class TestLinear:
    def test_min_max(self):
        assert min_max([2.0, 4.0, 3.0]) == [0.0, 1.0, 0.5]
        assert min_max([5.0, 5.0]) == [1.0, 1.0]
        assert min_max([]) == []

    def test_alpha_extremes_reproduce_each_list(self):
        vec = ranking("a", "b", "c", scores=[0.9, 0.5, 0.1])
        kw = ranking("c", "b", "a", scores=[12.0, 6.0, 1.0])
        assert ids(linear_fusion(vec, kw, k=3, alpha=1.0)) == ["a", "b", "c"]
        assert ids(linear_fusion(vec, kw, k=3, alpha=0.0)) == ["c", "b", "a"]

    def test_scores_match_formula(self):
        vec = ranking("a", "b", scores=[0.8, 0.4])  # normalised: a=1, b=0
        kw = ranking("b", "c", scores=[10.0, 2.0])  # normalised: b=1, c=0
        fused = linear_fusion(vec, kw, k=3, alpha=0.25)
        score = {h.passage.passage_id: h.score for h in fused}
        assert score["a"] == pytest.approx(0.25)
        assert score["b"] == pytest.approx(0.75)
        assert score["c"] == pytest.approx(0.0)

    def test_bad_alpha(self):
        with pytest.raises(ValueError):
            linear_fusion([], [], k=1, alpha=1.5)


@pytest.fixture
def demo():
    qs = load_questions(DEMO_QUESTIONS)
    store = build_store(
        qs.corpus, embedder_name="hashing-512", titles=load_corpus_titles(qs, None)
    )
    return qs, store


class TestHybridRetriever:
    def test_is_a_retriever_with_a_descriptive_name(self, demo):
        _, store = demo
        hybrid = HybridRetriever(store)
        assert isinstance(hybrid, Retriever)
        assert isinstance(store, Retriever)
        assert isinstance(BM25Index(store.passages), Retriever)
        assert hybrid.name == "hybrid-rrf(hashing-512+bm25)"
        assert HybridRetriever(store, "linear").name.startswith("hybrid-linear(")
        assert hybrid.passages is store.passages
        assert hybrid.lectures() == store.lectures()

    def test_search_returns_fused_ranking(self, demo):
        _, store = demo
        hybrid = HybridRetriever(store, pool=50)
        query = "what is the gradient descent learning rate"
        hits = hybrid.search(query, k=5)
        assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        # Same result as fusing the two lists by hand.
        manual = reciprocal_rank_fusion(
            [store.search(query, k=50), BM25Index(store.passages).search(query, k=50)],
            k=5,
        )
        assert ids(hits) == ids(manual)

    def test_lecture_filter_applies_to_both_sides(self, demo):
        _, store = demo
        lecture = store.lectures()[0]
        hits = HybridRetriever(store).search("matrix", k=10, lecture_ids=[lecture])
        assert hits and all(h.passage.lecture_id == lecture for h in hits)

    def test_empty_query_and_store(self, demo):
        _, store = demo
        assert HybridRetriever(store).search("  ") == []
        assert HybridRetriever(VectorStore(HashingEmbedder())).search("x") == []

    def test_bad_arguments(self, demo):
        _, store = demo
        with pytest.raises(ValueError):
            HybridRetriever(store, "max")
        with pytest.raises(ValueError):
            HybridRetriever(store, pool=0)
        with pytest.raises(ValueError):
            HybridRetriever(store, alpha=2)
        with pytest.raises(ValueError):
            HybridRetriever(store, bm25=BM25Index(store.passages[:2]))
        with pytest.raises(ValueError):
            HybridRetriever(store).search("x", k=0)

    def test_make_retriever(self, demo):
        _, store = demo
        assert make_retriever(store, "vector") is store
        assert isinstance(make_retriever(store, "bm25", b=0.5), BM25Index)
        assert make_retriever(store, "hybrid").fusion == "rrf"
        linear = make_retriever(store, "hybrid-linear", alpha=0.3)
        assert linear.fusion == "linear" and linear.alpha == 0.3
        with pytest.raises(ValueError, match="unknown retriever"):
            make_retriever(store, "magic")

    def test_cli_choices_match(self):
        # cli.py keeps its own copy so the parser does not import FAISS.
        assert cli.RETRIEVERS == RETRIEVERS


class TestEvalComparison:
    def test_every_retriever_beats_chance_on_demo(self, demo):
        qs, store = demo
        for kind in RETRIEVERS:
            report = evaluate(make_retriever(store, kind), qs)
            assert report.recall(5) > report.chance_recall[5], kind
            assert report.mrr() > report.chance_mrr, kind

    def test_bm25_beats_hashing_on_demo(self, demo):
        # The committed result in reports/hybrid.md: real IDF weighting and
        # length normalisation beat a plain hashed bag of words.
        qs, store = demo
        vector = evaluate(store, qs)
        bm25 = evaluate(make_retriever(store, "bm25"), qs)
        assert bm25.recall(5) > vector.recall(5)
        assert bm25.mrr() > vector.mrr()

    def test_cli_compares_retrievers(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(ROOT)
        out = tmp_path / "hybrid.md"
        code = cli.main(
            [
                "eval-retrieval",
                "--embedder", "hashing-512",
                "--max-words", "120", "60",
                "--retriever", "vector", "bm25", "hybrid", "hybrid-linear",
                "--alpha", "0.3",
                "-o", str(out),
            ]
        )  # fmt: skip
        printed = capsys.readouterr().out
        assert code == 0
        assert "bm25 @ 120w" in printed and "bm25 @ 60w" in printed
        assert "hybrid-rrf(hashing-512+bm25) @ 120w" in printed
        data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        # 4 retrievers x 2 passage sizes.
        assert len(data["runs"]) == 8
        assert {r["settings"]["retriever"] for r in data["runs"]} == set(RETRIEVERS)
        assert "| Retriever |" in out.read_text(encoding="utf-8")

    def test_cli_bm25_runs_once_per_passage_size(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(ROOT)
        out = tmp_path / "r.md"
        code = cli.main(
            [
                "eval-retrieval",
                "--embedder", "hashing-512",
                "--embedder", "hashing-256",
                "--retriever", "bm25", "hybrid",
                "-o", str(out),
            ]
        )  # fmt: skip
        assert code == 0
        data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        names = [r["embedder"] for r in data["runs"]]
        assert names.count("bm25") == 1
        assert len(names) == 3  # bm25 + one hybrid per embedder

    def test_cli_with_existing_index(self, tmp_path, capsys, monkeypatch, demo):
        _, store = demo
        store.save(tmp_path / "index")
        monkeypatch.chdir(ROOT)
        code = cli.main(
            [
                "eval-retrieval",
                "--index",
                str(tmp_path / "index"),
                "--retriever",
                "hybrid",
            ]
        )
        assert code == 0
        assert "hybrid-rrf(hashing-512+bm25)" in capsys.readouterr().out
