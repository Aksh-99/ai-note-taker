"""Tests for src/notetaker/vectorstore.py (passages + FAISS index).

All offline: the HashingEmbedder stands in for the neural model, so no
weights are downloaded. FAISS itself is real.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from notetaker.embeddings import HashingEmbedder
from notetaker.transcribe import Segment, Transcript
from notetaker.vectorstore import (
    MANIFEST_FILE,
    Passage,
    VectorStore,
    VectorStoreError,
    lecture_id_from_path,
    split_passages,
)


def make_transcript(texts: list[str], seconds_each: float = 10.0) -> Transcript:
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


LINEAR_ALGEBRA = make_transcript(
    [
        "Today we study eigenvalues and eigenvectors of a square matrix.",
        "To find eigenvalues we solve the characteristic polynomial det(A - lambda I).",
        "Each eigenvalue has an eigenvector that the matrix only stretches.",
        "Next time we cover diagonalization and powers of a matrix.",
    ]
)
MACHINE_LEARNING = make_transcript(
    [
        "Gradient descent updates the weights against the gradient of the loss.",
        "The learning rate controls the step size of each gradient update.",
        "Backpropagation computes gradients layer by layer with the chain rule.",
    ]
)


def build_store() -> VectorStore:
    store = VectorStore(HashingEmbedder())
    store.add_transcript(LINEAR_ALGEBRA, "la-05", title="Linear Algebra 5")
    store.add_transcript(MACHINE_LEARNING, "ml-03", title="Machine Learning 3")
    return store


# ---- passages ---------------------------------------------------------------


class TestSplitPassages:
    def test_word_limit_and_overlap(self):
        t = make_transcript(["one two three four"] * 5)  # 4 words per segment
        ps = split_passages(t, "lec", max_words=8, overlap_segments=1)
        # Windows of 2 segments, each starting 1 segment after the last began.
        assert [p.segment_ids for p in ps] == [[0, 1], [1, 2], [2, 3], [3, 4]]
        assert ps[0].start == 0.0 and ps[0].end == 20.0
        assert ps[1].passage_id == "lec#0001"

    def test_no_overlap(self):
        t = make_transcript(["one two three four"] * 5)
        ps = split_passages(t, "lec", max_words=8, overlap_segments=0)
        assert [p.segment_ids for p in ps] == [[0, 1], [2, 3], [4]]

    def test_time_limit(self):
        t = make_transcript(["word"] * 6, seconds_each=30.0)
        ps = split_passages(
            t, "lec", max_words=1000, max_seconds=60, overlap_segments=0
        )
        assert [p.segment_ids for p in ps] == [[0, 1], [2, 3], [4, 5]]

    def test_oversized_segment_kept_and_loop_ends(self):
        t = make_transcript(["a b c d e f g h i j", "k"])
        ps = split_passages(t, "lec", max_words=3, overlap_segments=5)
        assert [p.segment_ids for p in ps] == [[0], [1]]

    def test_skips_empty_segments_and_keeps_metadata(self):
        t = make_transcript(["  hello there  ", "", "general kenobi"])
        (p,) = split_passages(t, "lec", title="Star Wars", source="x/transcript.json")
        assert p.text == "hello there general kenobi"
        assert p.segment_ids == [0, 2]
        assert p.title == "Star Wars" and p.source == "x/transcript.json"

    def test_empty_transcript(self):
        assert split_passages(make_transcript([]), "lec") == []

    @pytest.mark.parametrize(
        "kwargs", [{"max_words": 0}, {"max_seconds": 0}, {"overlap_segments": -1}]
    )
    def test_rejects_bad_options(self, kwargs):
        with pytest.raises(ValueError):
            split_passages(LINEAR_ALGEBRA, "lec", **kwargs)


def test_passage_citation_and_roundtrip():
    p = Passage("la#0001", "la", 761.4, 790.0, "text", title="Lecture 5")
    assert p.timestamp == "00:12:41"
    assert p.citation() == "Lecture 5 [00:12:41]"
    assert Passage("x#0", "x", 0, 1, "t").citation() == "x [00:00:00]"
    assert Passage.from_dict(p.to_dict()) == p


@pytest.mark.parametrize(
    "path, expected",
    [
        ("outputs/la-05/transcript.json", "la-05"),
        ("transcripts/ml_lecture3.json", "ml_lecture3"),
    ],
)
def test_lecture_id_from_path(path, expected):
    assert lecture_id_from_path(path) == expected


# ---- the store ----------------------------------------------------------------


class TestVectorStore:
    def test_search_finds_right_lecture_and_timestamp(self):
        store = build_store()
        hits = store.search(
            "how to find eigenvalues with the characteristic polynomial"
        )
        assert hits[0].passage.lecture_id == "la-05"
        assert hits[0].rank == 1
        assert hits[0].passage.start <= 10.0 < hits[0].passage.end + 1e-9
        top_ml = store.search("backpropagation chain rule gradients", k=1)[0]
        assert top_ml.passage.lecture_id == "ml-03"

    def test_scores_sorted_and_match_numpy(self):
        store = build_store()
        query = "gradient of the loss"
        hits = store.search(query, k=len(store))
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        # FAISS must agree with a brute-force dot product.
        q = store.embedder.embed_query(query)
        brute = store.vectors() @ q
        best = int(np.argmax(brute))
        assert hits[0].passage == store.passages[best]
        assert hits[0].score == pytest.approx(float(brute[best]), abs=1e-5)

    def test_k_larger_than_index_and_edge_cases(self):
        store = build_store()
        assert len(store.search("matrix", k=1000)) == len(store)
        assert store.search("   ") == []
        assert VectorStore(HashingEmbedder()).search("matrix") == []
        with pytest.raises(ValueError):
            store.search("matrix", k=0)

    def test_lecture_filter(self):
        store = build_store()
        hits = store.search("eigenvalues matrix", k=3, lecture_ids=["ml-03"])
        assert hits and all(h.passage.lecture_id == "ml-03" for h in hits)
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))

    def test_reindex_replaces_instead_of_duplicating(self):
        store = build_store()
        before = len(store)
        store.add_transcript(LINEAR_ALGEBRA, "la-05", title="Linear Algebra 5")
        assert len(store) == before
        assert store.lectures() == ["ml-03", "la-05"]
        assert store.vectors().shape == (before, store.dim)
        with pytest.raises(VectorStoreError):
            store.add_transcript(LINEAR_ALGEBRA, "la-05", replace=False)

    def test_remove_lecture_keeps_rows_aligned(self):
        store = build_store()
        removed = store.remove_lecture("la-05")
        assert removed > 0 and store.lectures() == ["ml-03"]
        assert store.remove_lecture("nope") == 0
        # After removal, row i must still be passage i's vector.
        expected = store.embedder.embed([p.text for p in store.passages])
        np.testing.assert_allclose(store.vectors(), expected, atol=1e-6)
        store.remove_lecture("ml-03")
        assert len(store) == 0 and store.search("gradient") == []

    def test_duplicate_passage_ids_rejected(self):
        store = VectorStore(HashingEmbedder())
        p = Passage("x#0000", "x", 0, 1, "matrix")
        store.add_passages([p])
        with pytest.raises(VectorStoreError):
            store.add_passages([p])
        assert store.add_passages([]) == 0

    def test_bad_embedder_shape_rejected(self):
        class Wrong(HashingEmbedder):
            def embed(self, texts):
                return np.zeros((len(texts), self.dim + 1), dtype=np.float32)

        with pytest.raises(VectorStoreError):
            VectorStore(Wrong(dim=8)).add_passages([Passage("x#0", "x", 0, 1, "t")])


class TestSaveLoad:
    def test_roundtrip_same_results(self, tmp_path):
        store = build_store()
        store.save(tmp_path / "index")
        manifest = json.loads((tmp_path / "index" / MANIFEST_FILE).read_text())
        assert manifest["embedder"] == "hashing-512"
        assert manifest["count"] == len(store)
        assert manifest["lectures"] == ["la-05", "ml-03"]

        # No embedder given: rebuilt from the name in the manifest.
        loaded = VectorStore.load(tmp_path / "index")
        assert loaded.passages == store.passages
        q = "learning rate step size"
        assert [h.passage for h in loaded.search(q)] == [
            h.passage for h in store.search(q)
        ]

    def test_wrong_embedder_rejected(self, tmp_path):
        build_store().save(tmp_path)
        with pytest.raises(VectorStoreError, match="built with"):
            VectorStore.load(tmp_path, embedder=HashingEmbedder(dim=64))

    def test_missing_index(self, tmp_path):
        with pytest.raises(VectorStoreError, match="no index"):
            VectorStore.load(tmp_path)

    def test_inconsistent_files_rejected(self, tmp_path):
        build_store().save(tmp_path)
        lines = (tmp_path / "passages.jsonl").read_text().splitlines()
        (tmp_path / "passages.jsonl").write_text("\n".join(lines[:-1]) + "\n")
        with pytest.raises(VectorStoreError, match="inconsistent"):
            VectorStore.load(tmp_path)

    def test_old_format_rejected(self, tmp_path):
        build_store().save(tmp_path)
        manifest = json.loads((tmp_path / MANIFEST_FILE).read_text())
        manifest["format_version"] = 0
        (tmp_path / MANIFEST_FILE).write_text(json.dumps(manifest))
        with pytest.raises(VectorStoreError, match="format"):
            VectorStore.load(tmp_path)

    def test_empty_store_roundtrip(self, tmp_path):
        VectorStore(HashingEmbedder(dim=16)).save(tmp_path)
        loaded = VectorStore.load(tmp_path)
        assert len(loaded) == 0 and loaded.dim == 16
