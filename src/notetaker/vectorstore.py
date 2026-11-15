"""A searchable index of lecture passages, with lecture + timestamp metadata.

This is the storage half of search across lectures (Phase 5). It answers:
"which moments in which lectures talk about X?"

Pipeline:
    transcript.json --split_passages--> passages (about 30-60 s of speech each,
                                        with start/end times)
                    --embedder--------> one unit vector per passage
                    --FAISS-----------> an index that returns the k vectors
                                        closest to a query vector

Why passages instead of whole lectures or single segments?
    A whole lecture is one vector for an hour of talk, so it cannot point to
    *where* something was said and its meaning is a blur of every topic. A
    single Whisper segment (one sentence) is too short to carry context:
    "so that's the answer" means nothing alone. A window of a few sentences,
    capped by word count and duration, is the usual middle ground. Windows
    overlap by one segment so an idea split across a boundary is still whole
    in at least one passage.

Why FAISS?
    FAISS (Facebook AI Similarity Search, MIT licence) is the standard free
    library for nearest-neighbour search over vectors. We use its simplest
    index, `IndexFlatIP`: it stores every vector and, for a query q, computes
    the inner product q . v with all of them and returns the k largest. That
    is exact (no approximation) and, because our vectors have length 1, the
    inner product equals cosine similarity. A semester of lectures is a few
    thousand passages, which exact search handles in well under a
    millisecond; approximate indexes (IVF, HNSW) only pay off at millions.

On disk an index is a folder with three files:
    index.faiss      the vectors, in FAISS's binary format
    passages.jsonl   one JSON object per line: text + lecture + timestamps,
                     in the same order as the vectors (row i <-> line i)
    manifest.json    which embedding model made the vectors, their dimension
                     and a count, checked on load so a mismatch fails loudly
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from notetaker.embeddings import Embedder, make_embedder
from notetaker.export import format_timestamp
from notetaker.transcribe import Transcript

# Bump this if the on-disk layout changes, so old indexes are rejected with a
# clear message instead of being misread.
FORMAT_VERSION = 1

INDEX_FILE = "index.faiss"
PASSAGES_FILE = "passages.jsonl"
MANIFEST_FILE = "manifest.json"

# Passage size defaults. 120 words is roughly 45 s of lecture speech
# (lecturers speak about 150 words per minute) and comfortably inside the
# 256-token input limit of all-MiniLM-L6-v2; longer text would be cut off.
DEFAULT_MAX_WORDS = 120
DEFAULT_MAX_SECONDS = 90.0
DEFAULT_OVERLAP_SEGMENTS = 1


class VectorStoreError(RuntimeError):
    """The index is missing, corrupt, or was built with a different model."""


@dataclass(frozen=True)
class Passage:
    """A short stretch of one lecture: the unit that search returns."""

    passage_id: str  # unique, e.g. "linear-algebra-l05#0007"
    lecture_id: str  # which lecture it came from
    start: float  # seconds from the start of that lecture's audio
    end: float
    text: str
    title: str = ""  # human-readable lecture name, for citations
    source: str = ""  # path of the transcript it was built from
    segment_ids: list[int] = field(default_factory=list)

    @property
    def timestamp(self) -> str:
        """Start time as HH:MM:SS, the form used in citations."""
        return format_timestamp(self.start)

    def citation(self) -> str:
        """Short reference such as "Lecture 5: Eigenvalues [00:12:41]"."""
        return f"{self.title or self.lecture_id} [{self.timestamp}]"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Passage:
        return cls(**data)


@dataclass(frozen=True)
class SearchHit:
    """One search result: a passage, its similarity to the query and its rank."""

    passage: Passage
    score: float  # cosine similarity in [-1, 1]; higher is more similar
    rank: int  # 1 = best match


def lecture_id_from_path(path: str | Path) -> str:
    """Pick a lecture id for a transcript file.

    The CLI writes `<output_dir>/<lecture>/transcript.json`, so for a file
    literally named transcript.json the folder name is the useful id;
    otherwise the file name without its extension is used.
    """
    path = Path(path)
    if path.stem == "transcript" and path.parent.name:
        return path.parent.name
    return path.stem


def split_passages(
    transcript: Transcript,
    lecture_id: str,
    title: str = "",
    source: str = "",
    max_words: int = DEFAULT_MAX_WORDS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    overlap_segments: int = DEFAULT_OVERLAP_SEGMENTS,
) -> list[Passage]:
    """Group a transcript's segments into overlapping, time-stamped passages.

    Greedy, left to right: start a window at segment i and keep adding the
    next segment while the window stays within `max_words` words and
    `max_seconds` seconds. A single segment that is longer than the limits on
    its own still becomes a passage (it is never dropped). The next window
    starts `overlap_segments` segments before the end of this one, but always
    at least one segment further on, so the loop is guaranteed to finish.
    """
    if max_words <= 0 or max_seconds <= 0:
        raise ValueError("max_words and max_seconds must be positive")
    if overlap_segments < 0:
        raise ValueError("overlap_segments must be >= 0")

    segments = [s for s in transcript.segments if s.text.strip()]
    passages: list[Passage] = []
    i = 0
    while i < len(segments):
        j = i
        words = 0
        while j < len(segments):
            seg_words = len(segments[j].text.split())
            span = segments[j].end - segments[i].start
            # Always take the first segment; after that, respect both limits.
            if j > i and (words + seg_words > max_words or span > max_seconds):
                break
            words += seg_words
            j += 1
        window = segments[i:j]
        passages.append(
            Passage(
                passage_id=f"{lecture_id}#{len(passages):04d}",
                lecture_id=lecture_id,
                start=window[0].start,
                end=window[-1].end,
                text=" ".join(s.text.strip() for s in window),
                title=title,
                source=source,
                segment_ids=[s.id for s in window],
            )
        )
        if j >= len(segments):
            break
        i = max(j - overlap_segments, i + 1)
    return passages


def _as_faiss_matrix(vectors: np.ndarray) -> np.ndarray:
    """FAISS's C++ code needs a contiguous float32 2-D array; make sure of it."""
    return np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))


class VectorStore:
    """Passages plus their vectors in a FAISS inner-product index.

    Usage:
        store = VectorStore(embedder)
        store.add_transcript(transcript, lecture_id="la-05", title="Eigenvalues")
        hits = store.search("how do we find eigenvalues?", k=3)
        store.save("index/")
        store = VectorStore.load("index/")  # rebuilds the same embedder
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.passages: list[Passage] = []
        self._index = faiss.IndexFlatIP(embedder.dim)

    def __len__(self) -> int:
        return len(self.passages)

    @property
    def dim(self) -> int:
        return self._index.d

    def lectures(self) -> list[str]:
        """Lecture ids in the index, in the order they were added."""
        # dict.fromkeys keeps first-seen order and drops repeats.
        return list(dict.fromkeys(p.lecture_id for p in self.passages))

    # ---- adding and removing ---------------------------------------------

    def add_passages(self, passages: list[Passage]) -> int:
        """Embed passages and append them to the index. Returns how many."""
        if not passages:
            return 0
        existing = {p.passage_id for p in self.passages}
        clashes = [p.passage_id for p in passages if p.passage_id in existing]
        if clashes:
            raise VectorStoreError(f"passage ids already in the index: {clashes[:3]}")
        vectors = self.embedder.embed([p.text for p in passages])
        if vectors.shape != (len(passages), self.dim):
            raise VectorStoreError(
                f"embedder returned shape {vectors.shape}, expected "
                f"({len(passages)}, {self.dim})"
            )
        self._index.add(_as_faiss_matrix(vectors))
        self.passages.extend(passages)
        return len(passages)

    def add_transcript(
        self,
        transcript: Transcript,
        lecture_id: str,
        title: str = "",
        source: str = "",
        replace: bool = True,
        **split_options: Any,
    ) -> int:
        """Split a transcript into passages and index them. Returns how many.

        With `replace=True` (the default) any passages already indexed for
        this lecture are removed first, so re-indexing an updated transcript
        does not leave stale duplicates behind.
        """
        if lecture_id in self.lectures():
            if not replace:
                raise VectorStoreError(f"lecture {lecture_id!r} is already indexed")
            self.remove_lecture(lecture_id)
        passages = split_passages(
            transcript, lecture_id, title=title, source=source, **split_options
        )
        return self.add_passages(passages)

    def vectors(self) -> np.ndarray:
        """All stored vectors as an (n, dim) array, row i for passage i."""
        if len(self) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        # A flat index keeps the raw vectors, so they can be read back exactly.
        return self._index.reconstruct_n(0, self._index.ntotal)

    def remove_lecture(self, lecture_id: str) -> int:
        """Drop every passage of one lecture. Returns how many were removed.

        A flat index has no cheap "delete row" that keeps rows aligned with
        our passage list, so we keep the surviving vectors and build a fresh
        index from them. That costs one copy of the vectors, which is nothing
        at lecture-collection scale, and no re-embedding is needed.
        """
        keep = np.array([p.lecture_id != lecture_id for p in self.passages], dtype=bool)
        removed = int((~keep).sum())
        if removed == 0:
            return 0
        kept_vectors = self.vectors()[keep]
        self.passages = [p for p, k in zip(self.passages, keep, strict=True) if k]
        self._index = faiss.IndexFlatIP(self.dim)
        if len(kept_vectors):
            self._index.add(_as_faiss_matrix(kept_vectors))
        return removed

    # ---- searching ---------------------------------------------------------

    def search(
        self, query: str, k: int = 5, lecture_ids: list[str] | None = None
    ) -> list[SearchHit]:
        """Return the k passages most similar to `query`, best first.

        `lecture_ids` restricts the search to some lectures. FAISS's flat
        index has no metadata, so we ask it for every passage ranked and keep
        the first k that pass the filter. With exact search that is still
        correct, and at our scale still fast.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if len(self) == 0 or not query.strip():
            return []
        q = self.embedder.embed_query(query).reshape(1, -1)
        fetch = len(self) if lecture_ids is not None else min(k, len(self))
        # FAISS searches a batch of queries at once and returns two arrays of
        # shape (n_queries, fetch): similarity scores and row numbers. We sent
        # one query, so we read row 0. Missing results are marked with -1.
        scores, rows = self._index.search(_as_faiss_matrix(q), fetch)
        wanted = set(lecture_ids) if lecture_ids is not None else None
        hits: list[SearchHit] = []
        for score, row in zip(scores[0], rows[0], strict=True):
            if row < 0:
                continue
            passage = self.passages[row]
            if wanted is not None and passage.lecture_id not in wanted:
                continue
            hits.append(
                SearchHit(passage=passage, score=float(score), rank=len(hits) + 1)
            )
            if len(hits) == k:
                break
        return hits

    # ---- saving and loading ------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "embedder": self.embedder.name,
            "dim": self.dim,
            "metric": "cosine (inner product of unit vectors)",
            "count": len(self),
            "lectures": self.lectures(),
        }

    def save(self, directory: str | Path) -> Path:
        """Write the index folder (created if needed). Returns its path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / INDEX_FILE))
        with (directory / PASSAGES_FILE).open("w", encoding="utf-8") as f:
            for passage in self.passages:
                f.write(json.dumps(passage.to_dict(), ensure_ascii=False) + "\n")
        (directory / MANIFEST_FILE).write_text(
            json.dumps(self.manifest(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(
        cls, directory: str | Path, embedder: Embedder | None = None
    ) -> VectorStore:
        """Read an index folder written by `save`.

        Without an `embedder`, the one named in the manifest is rebuilt, which
        guarantees queries are embedded by the same model as the passages.
        If an embedder is passed, its name and dimension must match the
        manifest: comparing vectors from two different models is meaningless
        (like measuring one distance in metres and another in miles), so a
        mismatch is an error rather than a silent bad search.
        """
        directory = Path(directory)
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.exists():
            raise VectorStoreError(f"no index found at {directory} (missing manifest)")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise VectorStoreError(
                f"index format {manifest.get('format_version')} is not supported "
                f"(expected {FORMAT_VERSION}); rebuild the index"
            )
        if embedder is None:
            embedder = make_embedder(manifest["embedder"])
        elif embedder.name != manifest["embedder"]:
            raise VectorStoreError(
                f"index was built with {manifest['embedder']!r} but the embedder "
                f"given is {embedder.name!r}; rebuild the index or use that model"
            )

        index = faiss.read_index(str(directory / INDEX_FILE))
        lines = (directory / PASSAGES_FILE).read_text(encoding="utf-8").splitlines()
        passages = [Passage.from_dict(json.loads(line)) for line in lines if line]
        if index.d != manifest["dim"] or embedder.dim != index.d:
            raise VectorStoreError(
                f"dimension mismatch: index {index.d}, manifest {manifest['dim']}, "
                f"embedder {embedder.dim}"
            )
        if not (index.ntotal == len(passages) == manifest["count"]):
            raise VectorStoreError(
                f"index is inconsistent: {index.ntotal} vectors, {len(passages)} "
                f"passages, manifest says {manifest['count']}"
            )

        store = cls(embedder)
        store._index = index
        store.passages = passages
        return store
