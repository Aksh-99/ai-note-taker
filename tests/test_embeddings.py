"""Tests for src/notetaker/embeddings.py.

Everything here is offline: the HashingEmbedder needs no model, and the
SentenceTransformerEmbedder is tested with a small fake model object. One
test marked `slow` loads the real MiniLM model and is skipped in CI.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from notetaker.embeddings import (
    DEFAULT_MODEL,
    Embedder,
    EmbeddingError,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    l2_normalize,
    make_embedder,
    tokenize,
)


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Eigenvalues of A, um, don't change!") == [
        "eigenvalues",
        "don't",
        "change",
    ]


def test_l2_normalize_unit_rows_and_zero_rows():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], [0.6, 0.8], rtol=1e-6)
    np.testing.assert_array_equal(out[1], [0.0, 0.0])  # no NaN


class TestHashingEmbedder:
    def test_satisfies_protocol_and_shapes(self):
        emb = HashingEmbedder(dim=64)
        assert isinstance(emb, Embedder)
        vecs = emb.embed(["matrix multiplication", "gradient descent", ""])
        assert vecs.shape == (3, 64)
        assert vecs.dtype == np.float32
        assert emb.embed_query("matrix").shape == (64,)
        assert emb.embed([]).shape == (0, 64)

    def test_vectors_are_unit_length_except_empty(self):
        vecs = HashingEmbedder().embed(["eigenvalues and eigenvectors", "the a of"])
        assert np.linalg.norm(vecs[0]) == pytest.approx(1.0, abs=1e-5)
        # Only stopwords -> no features -> zero vector, not NaN.
        assert np.linalg.norm(vecs[1]) == 0.0

    def test_shared_words_score_higher_than_unrelated(self):
        emb = HashingEmbedder()
        q = emb.embed_query("how do we compute eigenvalues of a matrix")
        related, unrelated = emb.embed(
            [
                "to compute the eigenvalues of the matrix we solve det(A - lambda I)",
                "the french revolution began in 1789 with the estates general",
            ]
        )
        assert float(q @ related) > float(q @ unrelated) + 0.2

    def test_deterministic_across_processes(self):
        # Python's hash() changes per process; ours must not, or a saved index
        # would stop matching new queries.
        code = (
            "from notetaker.embeddings import HashingEmbedder;"
            "print(HashingEmbedder(dim=32).embed_query('singular value').tolist())"
        )
        runs = [
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True
            ).stdout
            for _ in range(2)
        ]
        assert runs[0] == runs[1]
        local = HashingEmbedder(dim=32).embed_query("singular value").tolist()
        assert runs[0].strip() == str(local)

    def test_bigrams_change_the_vector(self):
        a = HashingEmbedder(dim=256).embed_query("linear algebra")
        b = HashingEmbedder(dim=256, bigrams=False).embed_query("linear algebra")
        assert not np.allclose(a, b)

    def test_names(self):
        assert HashingEmbedder(dim=128).name == "hashing-128"
        assert HashingEmbedder(dim=128, bigrams=False).name == "hashing-128-unigram"

    def test_rejects_bad_dim(self):
        with pytest.raises(ValueError):
            HashingEmbedder(dim=0)


class FakeSTModel:
    """Stands in for sentence_transformers.SentenceTransformer.

    It maps each text to a fixed, un-normalised vector based on its length so
    we can check that our wrapper normalises and applies prefixes.
    """

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[list[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.array(
            [[len(t), 1.0] + [0.0] * (self.dim - 2) for t in texts], dtype=np.float64
        )


class TestSentenceTransformerEmbedder:
    def test_wraps_model_and_normalises(self):
        fake = FakeSTModel()
        emb = SentenceTransformerEmbedder("fake/model", model=fake)
        assert isinstance(emb, Embedder)
        assert emb.dim == 4
        vecs = emb.embed(["abc", "a"])
        assert vecs.shape == (2, 4) and vecs.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-6)

    def test_prefixes(self):
        fake = FakeSTModel()
        emb = SentenceTransformerEmbedder(
            "e5", model=fake, query_prefix="query: ", passage_prefix="passage: "
        )
        emb.embed(["x"])
        emb.embed_query("y")
        assert fake.calls == [["passage: x"], ["query: y"]]

    def test_newer_dimension_method_name(self):
        class NewerModel(FakeSTModel):
            def get_embedding_dimension(self) -> int:
                return 6

        assert SentenceTransformerEmbedder("m", model=NewerModel(dim=6)).dim == 6

    def test_bad_model_output_raises(self):
        class Broken(FakeSTModel):
            def encode(self, texts, **kwargs):
                return np.zeros((1, self.dim))

        emb = SentenceTransformerEmbedder("m", model=Broken())
        with pytest.raises(EmbeddingError):
            emb.embed(["a", "b"])

    def test_model_not_loaded_until_used(self, monkeypatch):
        import notetaker.embeddings as mod

        def boom():
            raise AssertionError("should not import sentence-transformers yet")

        monkeypatch.setattr(mod, "_import_sentence_transformers", boom)
        SentenceTransformerEmbedder("anything")  # creating it is free

    def test_rejects_bad_batch_size(self):
        with pytest.raises(ValueError):
            SentenceTransformerEmbedder(batch_size=0)


def test_make_embedder_by_name():
    h = make_embedder("hashing-64")
    assert isinstance(h, HashingEmbedder) and h.dim == 64 and h.bigrams
    assert make_embedder("hashing-64-unigram").bigrams is False
    st = make_embedder("BAAI/bge-small-en-v1.5")
    assert isinstance(st, SentenceTransformerEmbedder)
    assert st.name == "BAAI/bge-small-en-v1.5"


def test_make_embedder_defaults_to_settings(monkeypatch):
    from notetaker.config import get_settings

    monkeypatch.setenv("NOTETAKER_EMBEDDING_MODEL", "hashing-16")
    get_settings.cache_clear()
    try:
        assert make_embedder().name == "hashing-16"
    finally:
        get_settings.cache_clear()


@pytest.mark.slow
def test_real_minilm_ranks_paraphrase_first():
    """Loads the real model (downloads ~90 MB once). Skipped in CI."""
    pytest.importorskip("sentence_transformers")
    emb = SentenceTransformerEmbedder(DEFAULT_MODEL)
    assert emb.dim == 384
    q = emb.embed_query("How do I find the eigenvalues of a matrix?")
    docs = emb.embed(
        [
            "We solve the characteristic equation det(A minus lambda I) equals zero.",
            "The French Revolution began in 1789.",
        ]
    )
    assert float(q @ docs[0]) > float(q @ docs[1])
