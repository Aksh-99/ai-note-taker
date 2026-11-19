"""Hybrid search: combine keyword (BM25) and vector rankings (Step 5.5).

The two retrievers fail in different ways:
    vector search (embeddings) understands paraphrases but can miss an exact
        rare word, e.g. a name or a symbol the model has never seen;
    BM25 (bm25.py) nails exact words but knows nothing about meaning, so
        "how do I get the eigenvalues" will not match "det(A - lambda I) = 0".
Fusing their rankings lets each cover the other's blind spot. This module
implements the two standard ways of fusing.

1. Reciprocal Rank Fusion (RRF), the default
   (Cormack, Clarke and Buettcher, SIGIR 2009). Each retriever votes with
   the *rank* it gave a passage, never its raw score:

        rrf(d) = sum over retrievers i of  w_i / (c + rank_i(d))

   A passage missing from a retriever's list gets nothing from it. c = 60 is
   the constant from the paper; it flattens the curve so that rank 1 vs
   rank 2 matters, but not enormously (1/61 vs 1/62). Why ranks? BM25 scores
   are unbounded (0 to 20+) while cosine similarities live in [-1, 1], so
   adding raw scores would let BM25 dominate. Ranks are on the same scale
   for every retriever, so RRF needs no tuning and no score calibration.
   That robustness is why it is the usual first choice.

2. Linear (convex) combination of normalised scores:

        lin(d) = alpha * s_vec(d) + (1 - alpha) * s_bm25(d)

   where each retriever's scores are min-max scaled to [0, 1] within its own
   candidate list, (s - min) / (max - min), and a missing passage counts 0.
   alpha in [0, 1] is a knob: 1 = pure vectors, 0 = pure BM25. This can beat
   RRF when tuned, but the best alpha depends on the data, and min-max
   scaling is sensitive to outliers. Included so the eval can compare.

Each retriever only contributes its top `pool` candidates (default 50).
Fusing complete rankings would be exact but wasteful at scale; passages
outside both top-50 lists are very unlikely to end up in the fused top 10.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from notetaker.bm25 import DEFAULT_B, DEFAULT_K1, BM25Index
from notetaker.vectorstore import Passage, SearchHit, VectorStore

DEFAULT_RRF_C = 60  # the constant from the RRF paper
DEFAULT_ALPHA = 0.5
DEFAULT_POOL = 50

# Names accepted by `make_retriever` and by `--retriever` on the command line.
RETRIEVERS = ("vector", "bm25", "hybrid", "hybrid-linear")


@runtime_checkable
class Retriever(Protocol):
    """Anything that can rank passages for a query.

    VectorStore, BM25Index and HybridRetriever all have this shape, so the
    retrieval eval and the ask command can take any of them. (A Protocol
    describes required attributes and methods instead of a parent class;
    see embeddings.Embedder for the same pattern.)
    """

    name: str
    passages: list[Passage]

    def lectures(self) -> list[str]: ...

    def search(
        self, query: str, k: int = 5, lecture_ids: list[str] | None = None
    ) -> list[SearchHit]: ...


# ---------------------------------------------------------------------------
# Fusion functions: pure maths on ranked lists, easy to test by hand
# ---------------------------------------------------------------------------


def _passage_lookup(rankings: Sequence[Sequence[SearchHit]]) -> dict[str, Passage]:
    """passage_id -> Passage for every passage in any ranking."""
    return {h.passage.passage_id: h.passage for hits in rankings for h in hits}


def _best_rank(pid: str, rankings: Sequence[Sequence[SearchHit]]) -> int:
    """The best (smallest) rank any retriever gave this passage, for tie-breaks."""
    ranks = [h.rank for hits in rankings for h in hits if h.passage.passage_id == pid]
    return min(ranks)


def _to_hits(
    fused: dict[str, float], rankings: Sequence[Sequence[SearchHit]], k: int
) -> list[SearchHit]:
    """Sort fused scores into SearchHits, best first, with fresh ranks 1..k.

    Ties (common with RRF, e.g. rank 1 in one list vs rank 1 in the other)
    are broken by the best individual rank, then by passage id, so the order
    is fully deterministic.
    """
    lookup = _passage_lookup(rankings)
    order = sorted(fused, key=lambda pid: (-fused[pid], _best_rank(pid, rankings), pid))
    return [
        SearchHit(passage=lookup[pid], score=fused[pid], rank=i + 1)
        for i, pid in enumerate(order[:k])
    ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]],
    k: int,
    c: float = DEFAULT_RRF_C,
    weights: Sequence[float] | None = None,
) -> list[SearchHit]:
    """Fuse several rankings with RRF; returns the top k. Score = RRF value."""
    if c < 0:
        raise ValueError(f"c must be >= 0, got {c}")
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("need one weight per ranking")
    fused: dict[str, float] = {}
    for hits, w in zip(rankings, weights, strict=True):
        for h in hits:
            pid = h.passage.passage_id
            fused[pid] = fused.get(pid, 0.0) + w / (c + h.rank)
    return _to_hits(fused, rankings, k)


def min_max(scores: Sequence[float]) -> list[float]:
    """Scale scores to [0, 1]: (s - min) / (max - min).

    If every score is equal there is no spread to scale, so all become 1.0
    (they are all "the best this retriever found").
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def linear_fusion(
    vector_hits: Sequence[SearchHit],
    keyword_hits: Sequence[SearchHit],
    k: int,
    alpha: float = DEFAULT_ALPHA,
) -> list[SearchHit]:
    """alpha * normalised vector score + (1 - alpha) * normalised BM25 score."""
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    fused: dict[str, float] = {}
    for hits, w in ((vector_hits, alpha), (keyword_hits, 1.0 - alpha)):
        for h, s in zip(hits, min_max([h.score for h in hits]), strict=True):
            pid = h.passage.passage_id
            fused[pid] = fused.get(pid, 0.0) + w * s
    return _to_hits(fused, [vector_hits, keyword_hits], k)


# ---------------------------------------------------------------------------
# The hybrid retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Vector search and BM25 over the same passages, fused into one ranking.

    Usage:
        store = VectorStore.load("index/")
        hybrid = HybridRetriever(store)            # RRF
        hits = hybrid.search("bellman equation", k=5)

    The BM25 index is built from `store.passages` on construction, so an
    existing vector index needs nothing extra on disk.
    """

    def __init__(
        self,
        store: VectorStore,
        fusion: str = "rrf",
        *,
        bm25: BM25Index | None = None,
        rrf_c: float = DEFAULT_RRF_C,
        alpha: float = DEFAULT_ALPHA,
        pool: int = DEFAULT_POOL,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        if fusion not in ("rrf", "linear"):
            raise ValueError(f"fusion must be 'rrf' or 'linear', got {fusion!r}")
        if pool <= 0:
            raise ValueError(f"pool must be positive, got {pool}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        self.store = store
        self.bm25 = bm25 if bm25 is not None else BM25Index(store.passages, k1=k1, b=b)
        if len(self.bm25) != len(store):
            raise ValueError(
                "the BM25 index and the vector store hold different passages"
            )
        self.fusion = fusion
        self.rrf_c = rrf_c
        self.alpha = alpha
        self.pool = pool

    @property
    def name(self) -> str:
        """E.g. "hybrid-rrf(hashing-512+bm25)", used as the row label in reports."""
        return f"hybrid-{self.fusion}({self.store.embedder.name}+bm25)"

    @property
    def passages(self) -> list[Passage]:
        return self.store.passages

    def lectures(self) -> list[str]:
        return self.store.lectures()

    def search(
        self, query: str, k: int = 5, lecture_ids: list[str] | None = None
    ) -> list[SearchHit]:
        """Top k passages by fused score, best first."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if len(self.store) == 0 or not query.strip():
            return []
        pool = max(k, self.pool)
        vector_hits = self.store.search(query, k=pool, lecture_ids=lecture_ids)
        keyword_hits = self.bm25.search(query, k=pool, lecture_ids=lecture_ids)
        if self.fusion == "rrf":
            return reciprocal_rank_fusion([vector_hits, keyword_hits], k, c=self.rrf_c)
        return linear_fusion(vector_hits, keyword_hits, k, alpha=self.alpha)


def make_retriever(store: VectorStore, kind: str = "vector", **options) -> Retriever:
    """Build a retriever over `store` by name: one of RETRIEVERS.

    "vector" returns the store itself (it already is a retriever).
    Extra keyword options go to HybridRetriever (rrf_c, alpha, pool, k1, b).
    """
    if kind == "vector":
        return store
    if kind == "bm25":
        bm25_opts = {key: options[key] for key in ("k1", "b") if key in options}
        return BM25Index(store.passages, **bm25_opts)
    if kind == "hybrid":
        return HybridRetriever(store, "rrf", **options)
    if kind == "hybrid-linear":
        return HybridRetriever(store, "linear", **options)
    raise ValueError(f"unknown retriever {kind!r}; choose from {', '.join(RETRIEVERS)}")
