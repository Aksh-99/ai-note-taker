"""BM25: the classic keyword-search ranking formula, for hybrid search (Step 5.5).

Why add keyword search when we already have embeddings?
    Embeddings are good at meaning ("how do I get the eigenvalues?" finds
    "solve det(A - lambda I) = 0") but weak at exact rare words: a course
    code, a person's name, "Kolmogorov", "Bellman". A rare word that appears
    in the question and in one passage is very strong evidence, and a
    keyword ranker sees it directly. Combining the two ("hybrid search", see
    hybrid.py) usually beats either one alone. BM25 is the standard keyword
    ranker: it is what Elasticsearch and Lucene use by default.

The formula (Robertson and Zaragoza, "The Probabilistic Relevance Framework:
BM25 and Beyond", 2009). For a query q and a passage d:

    score(q, d) = sum over distinct terms t in q of
                  idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * norm(d))

    norm(d) = 1 - b + b * |d| / avgdl

where
    tf(t, d)  how many times t occurs in d
    |d|       number of tokens in d; avgdl is the mean over all passages
    idf(t)    ln(1 + (N - n_t + 0.5) / (n_t + 0.5)), with N passages of
              which n_t contain t. Rare terms get a large weight, terms in
              nearly every passage a weight close to 0. (The "1 +" inside
              the log is Lucene's variant: it keeps idf positive even for a
              term in more than half the passages.)

Reading the tf part as a function of x = tf: x (k1 + 1) / (x + k1 * norm)
    is 0 at x = 0, increasing, and tends to k1 + 1 as x grows. So the first
    occurrence of a word counts most and repeats count less and less
    ("saturation"); saying "matrix" 20 times does not make a passage 20 times
    more relevant. k1 (default 1.5) sets how fast it saturates.
    norm(d) > 1 for passages longer than average, which lowers their tf part:
    a long passage contains more words just by being long, so a match in it
    is weaker evidence. b in [0, 1] (default 0.75) sets how strongly length
    is penalised; b = 0 turns length normalisation off.

Implementation:
    An "inverted index": for every term, the list of passages that contain it
    and how often (its "postings"). A query then only touches the passages
    that share at least one word with it, instead of every passage. Scores
    are accumulated with numpy, one term at a time.

Tokenisation reuses `embeddings.tokenize` (lowercase words, stopwords
removed), so BM25 and the hashing embedder agree on what a "word" is.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from notetaker.embeddings import tokenize
from notetaker.vectorstore import Passage, SearchHit

# Standard defaults from the BM25 literature (also Lucene's defaults are
# k1 = 1.2, b = 0.75; 1.5 is common in research code and works well on short
# passages). Exposed as arguments so the retrieval eval can tune them.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


@dataclass(frozen=True)
class _Postings:
    """Where one term occurs: passage row numbers and counts, as parallel arrays."""

    rows: np.ndarray  # int64, row i of the index = passages[i]
    counts: np.ndarray  # float64, tf of the term in each of those passages


class BM25Index:
    """Keyword search over passages with the BM25 formula.

    Usage:
        bm25 = BM25Index(store.passages)
        hits = bm25.search("what is the bellman equation", k=5)

    It has the same `search(query, k, lecture_ids)` method and returns the
    same `SearchHit` objects as `VectorStore`, so code that searches does not
    care which one it was given (this shared shape is often called an
    "interface"). Unlike a vector index it needs no model and nothing saved
    to disk: building it from a few thousand passages takes milliseconds, so
    it is simply rebuilt from the passages whenever it is needed.
    """

    name = "bm25"

    def __init__(
        self, passages: Sequence[Passage], k1: float = DEFAULT_K1, b: float = DEFAULT_B
    ) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be between 0 and 1, got {b}")
        self.passages = list(passages)
        self.k1 = k1
        self.b = b

        # Count words once per passage. Counter maps word -> occurrences.
        counted = [Counter(tokenize(p.text)) for p in self.passages]
        self.doc_lengths = np.array([sum(c.values()) for c in counted], dtype=float)
        n = len(self.passages)
        # Guard against dividing by zero when every passage is empty.
        self.avgdl = (
            float(self.doc_lengths.mean()) if n and self.doc_lengths.any() else 1.0
        )

        # Build the inverted index: term -> (rows, counts). We first collect
        # plain Python lists, then freeze them into numpy arrays.
        rows: dict[str, list[int]] = {}
        tfs: dict[str, list[int]] = {}
        for row, counts in enumerate(counted):
            for term, tf in counts.items():
                rows.setdefault(term, []).append(row)
                tfs.setdefault(term, []).append(tf)
        self._postings = {
            term: _Postings(
                np.array(rows[term], dtype=np.int64), np.array(tfs[term], dtype=float)
            )
            for term in rows
        }
        # idf depends only on how many passages contain the term (n_t).
        self.idf = {term: self._idf(n, len(rows[term])) for term in rows}
        # norm(d) from the module docstring, precomputed for every passage.
        self._norm = 1.0 - b + b * self.doc_lengths / self.avgdl

    @staticmethod
    def _idf(n_passages: int, n_containing: int) -> float:
        """ln(1 + (N - n_t + 0.5) / (n_t + 0.5)); always > 0."""
        return math.log(1.0 + (n_passages - n_containing + 0.5) / (n_containing + 0.5))

    def __len__(self) -> int:
        return len(self.passages)

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def lectures(self) -> list[str]:
        """Lecture ids in passage order (same as VectorStore.lectures)."""
        return list(dict.fromkeys(p.lecture_id for p in self.passages))

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of the query against every passage, as an array of length N.

        Repeated query words are counted once ("eigen eigen" = "eigen"); a
        question rarely repeats a word on purpose, and this keeps scores
        comparable between short and long questions. Words not in any passage
        contribute nothing.
        """
        total = np.zeros(len(self.passages), dtype=float)
        for term in dict.fromkeys(tokenize(query)):  # distinct, in order
            post = self._postings.get(term)
            if post is None:
                continue
            tf = post.counts
            tf_part = tf * (self.k1 + 1.0) / (tf + self.k1 * self._norm[post.rows])
            # np.add.at adds into `total` at the given rows. (Plain
            # `total[rows] += x` is also safe here because each passage
            # appears at most once per term, but add.at states the intent.)
            np.add.at(total, post.rows, self.idf[term] * tf_part)
        return total

    def search(
        self, query: str, k: int = 5, lecture_ids: list[str] | None = None
    ) -> list[SearchHit]:
        """The k highest-scoring passages, best first.

        Passages with score 0 share no word with the query, so BM25 has no
        opinion about them. They are left out rather than returned in an
        arbitrary order, which means fewer than k hits can come back. Ties
        are broken by passage order so results are reproducible.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not self.passages or not query.strip():
            return []
        scores = self.scores(query)
        if lecture_ids is not None:
            wanted = set(lecture_ids)
            mask = np.array([p.lecture_id in wanted for p in self.passages])
            scores = np.where(mask, scores, 0.0)
        # argsort sorts ascending, so sort the negated scores. kind="stable"
        # keeps equal scores in passage order (the tie-break above).
        order = np.argsort(-scores, kind="stable")
        hits: list[SearchHit] = []
        for row in order[:k]:
            if scores[row] <= 0:
                break  # everything after this is 0 too
            hits.append(
                SearchHit(
                    passage=self.passages[row],
                    score=float(scores[row]),
                    rank=len(hits) + 1,
                )
            )
        return hits
