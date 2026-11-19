"""Tests for BM25 keyword search (Step 5.5).

Scores are checked against the formula computed by hand on a tiny corpus,
plus the properties the formula is designed to have: rare words weigh more,
repeated words saturate, long passages are penalised (unless b = 0).
"""

from __future__ import annotations

import math

import pytest

from notetaker.bm25 import BM25Index
from notetaker.vectorstore import Passage


def passages(*texts: str, lecture: str = "lec") -> list[Passage]:
    return [
        Passage(f"{lecture}#{i:04d}", lecture, 10.0 * i, 10.0 * (i + 1), t)
        for i, t in enumerate(texts)
    ]


CORPUS = passages(
    "eigenvalues of a matrix",  # tokens: eigenvalues, matrix
    "matrix multiplication",  # matrix, multiplication
    "gradient descent step size",  # gradient, descent, step, size
)


class TestFormula:
    def test_idf_matches_hand_computation(self):
        bm25 = BM25Index(CORPUS)
        # "matrix" is in 2 of 3 passages: ln(1 + (3 - 2 + 0.5) / (2 + 0.5)).
        assert bm25.idf["matrix"] == pytest.approx(math.log(1 + 1.5 / 2.5))
        # "gradient" is in 1 of 3: ln(1 + 2.5 / 1.5), a larger weight.
        assert bm25.idf["gradient"] == pytest.approx(math.log(1 + 2.5 / 1.5))
        assert bm25.idf["gradient"] > bm25.idf["matrix"] > 0

    def test_score_matches_hand_computation(self):
        k1, b = 1.5, 0.75
        bm25 = BM25Index(CORPUS, k1=k1, b=b)
        avgdl = (2 + 2 + 4) / 3
        assert bm25.avgdl == pytest.approx(avgdl)
        # Passage 0 for query "eigenvalues": tf = 1, |d| = 2.
        idf = math.log(1 + 2.5 / 1.5)
        norm = 1 - b + b * 2 / avgdl
        expected = idf * 1 * (k1 + 1) / (1 + k1 * norm)
        scores = bm25.scores("eigenvalues")
        assert scores[0] == pytest.approx(expected)
        assert scores[1] == 0 and scores[2] == 0

    def test_scores_add_over_distinct_query_terms(self):
        bm25 = BM25Index(CORPUS)
        both = bm25.scores("matrix eigenvalues")
        assert both == pytest.approx(bm25.scores("matrix") + bm25.scores("eigenvalues"))

    def test_repeated_query_words_count_once(self):
        bm25 = BM25Index(CORPUS)
        assert bm25.scores("matrix matrix") == pytest.approx(bm25.scores("matrix"))

    def test_stopwords_and_unknown_words_score_zero(self):
        bm25 = BM25Index(CORPUS)
        assert not bm25.scores("the of and").any()
        assert not bm25.scores("zebra").any()


class TestProperties:
    def test_term_frequency_saturates(self):
        bm25 = BM25Index(passages("cat", "cat cat", "cat cat cat cat", "dog"), b=0)
        s = bm25.scores("cat")
        # More occurrences score higher, but each extra one adds less.
        assert s[0] < s[1] < s[2]
        assert (s[1] - s[0]) > (s[2] - s[1]) / 2
        # The tf part is bounded by k1 + 1, so the score by idf * (k1 + 1).
        assert s[2] < bm25.idf["cat"] * (bm25.k1 + 1)

    def test_long_passages_are_penalised(self):
        texts = ("cat dog", "cat dog fish bird horse cow sheep goat")
        normal = BM25Index(passages(*texts)).scores("cat")
        assert normal[0] > normal[1]
        # With b = 0 there is no length normalisation: equal tf, equal score.
        flat = BM25Index(passages(*texts), b=0).scores("cat")
        assert flat[0] == pytest.approx(flat[1])

    def test_rare_word_beats_common_word(self):
        bm25 = BM25Index(
            passages("matrix rank", "matrix trace", "matrix bellman", "matrix norm")
        )
        hits = bm25.search("matrix bellman", k=4)
        assert hits[0].passage.text == "matrix bellman"


class TestSearch:
    def test_ranked_hits_with_ranks_and_scores(self):
        hits = BM25Index(CORPUS).search("matrix eigenvalues", k=5)
        assert [h.passage.passage_id for h in hits] == ["lec#0000", "lec#0001"]
        assert [h.rank for h in hits] == [1, 2]
        assert hits[0].score > hits[1].score > 0

    def test_zero_score_passages_are_left_out(self):
        # Only two passages mention "matrix"; the third is not padded in.
        assert len(BM25Index(CORPUS).search("matrix", k=3)) == 2

    def test_k_limits_results(self):
        assert len(BM25Index(CORPUS).search("matrix", k=1)) == 1

    def test_ties_keep_passage_order(self):
        bm25 = BM25Index(passages("cat", "cat", "cat"))
        hits = bm25.search("cat", k=3)
        assert [h.passage.passage_id for h in hits] == [
            "lec#0000",
            "lec#0001",
            "lec#0002",
        ]

    def test_lecture_filter(self):
        corpus = passages("matrix a", lecture="la") + passages("matrix b", lecture="ml")
        bm25 = BM25Index(corpus)
        assert bm25.lectures() == ["la", "ml"]
        hits = bm25.search("matrix", k=5, lecture_ids=["ml"])
        assert [h.passage.lecture_id for h in hits] == ["ml"]

    def test_empty_query_and_empty_index(self):
        assert BM25Index(CORPUS).search("   ") == []
        empty = BM25Index([])
        assert len(empty) == 0 and empty.search("matrix") == []

    def test_all_empty_passages_do_not_divide_by_zero(self):
        bm25 = BM25Index(passages("", "the of"))
        assert bm25.search("anything") == []

    @pytest.mark.parametrize("kwargs", [{"k1": -1.0}, {"b": 1.5}, {"b": -0.1}], ids=str)
    def test_bad_parameters(self, kwargs):
        with pytest.raises(ValueError):
            BM25Index(CORPUS, **kwargs)

    def test_bad_k(self):
        with pytest.raises(ValueError):
            BM25Index(CORPUS).search("matrix", k=0)

    def test_vocabulary(self):
        assert BM25Index(CORPUS).vocabulary_size == 7
