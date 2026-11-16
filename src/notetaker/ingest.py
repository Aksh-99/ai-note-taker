"""Index every transcript in a folder into one searchable vector store.

This is Step 5.2: the `notetaker ingest` command. Step 5.1 built the store
(`vectorstore.VectorStore`); this module decides *which* transcripts go into
it and keeps the index in sync with the folder as lectures are added,
re-transcribed or deleted.

    outputs/                          index/
      la-05/transcript.json   --->      index.faiss     (vectors)
      la-06/transcript.json             passages.jsonl  (text + timestamps)
      ml-02/transcript.json             manifest.json   (embedding model)
                                        ingest.json     (what was indexed, from
                                                         which file, file hash)

Incremental by design
    Embedding is the slow part (a neural model runs over every passage), so
    re-running `ingest` after adding one new lecture should not re-embed the
    other forty. For each transcript we store a SHA-256 hash of the file's
    bytes in ingest.json. A hash is a fingerprint: identical bytes always give
    the same 64-character hex string, and any change, even one character,
    gives a completely different one. So "has this file changed since we
    indexed it?" becomes "is the hash different?", with no need to keep a
    copy of the old file or trust file modification times (which change on
    `git checkout` or copying even when the content did not).

    A lecture is re-indexed when its file hash changes, its title changes, or
    the passage-splitting settings change (different settings give different
    passages, so every lecture must be redone). Everything else is skipped.

Lecture ids
    Each lecture needs a stable id; it is what citations and filters use. The
    rule from Step 5.1 is reused: `outputs/la-05/transcript.json` -> "la-05".
    If two files would get the same id (say `fall/la-05/transcript.json` and
    `spring/la-05/transcript.json`), both fall back to their path relative to
    the folder with slashes turned into dashes ("fall-la-05", "spring-la-05"),
    so nothing silently overwrites anything.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from notetaker.embeddings import Embedder, make_embedder
from notetaker.transcribe import Transcript
from notetaker.vectorstore import (
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_WORDS,
    DEFAULT_OVERLAP_SEGMENTS,
    MANIFEST_FILE,
    VectorStore,
    lecture_id_from_path,
    split_passages,
)

STATE_FILE = "ingest.json"
STATE_FORMAT_VERSION = 1
DEFAULT_PATTERN = "transcript.json"
DEFAULT_INDEX_DIR = Path("index")

# The possible outcomes for one lecture in a run, as shown by the CLI.
ADDED, UPDATED, UNCHANGED, REMOVED, MISSING, FAILED = (
    "added",
    "updated",
    "unchanged",
    "removed",
    "missing",
    "failed",
)


class IngestError(RuntimeError):
    """Ingest cannot run at all (no transcripts, wrong embedder, bad titles)."""


@dataclass(frozen=True)
class SplitOptions:
    """How transcripts are cut into passages (see vectorstore.split_passages)."""

    max_words: int = DEFAULT_MAX_WORDS
    max_seconds: float = DEFAULT_MAX_SECONDS
    overlap_segments: int = DEFAULT_OVERLAP_SEGMENTS

    def __post_init__(self) -> None:
        # __post_init__ runs right after the dataclass's generated __init__,
        # so bad settings fail here, before any file is touched.
        if self.max_words <= 0 or self.max_seconds <= 0:
            raise ValueError("max_words and max_seconds must be positive")
        if self.overlap_segments < 0:
            raise ValueError("overlap_segments must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LectureResult:
    """What happened to one lecture during an ingest run."""

    lecture_id: str
    status: str  # one of ADDED, UPDATED, UNCHANGED, REMOVED, MISSING, FAILED
    source: str = ""
    passages: int = 0
    reason: str = ""  # why it was re-indexed, or the error if it failed


@dataclass
class IngestReport:
    """Summary of a run: one LectureResult per lecture plus index totals."""

    index_dir: Path
    embedder: str
    results: list[LectureResult] = field(default_factory=list)
    total_passages: int = 0
    total_lectures: int = 0
    dry_run: bool = False
    saved: bool = False

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)

    @property
    def changed(self) -> bool:
        """Did this run change what is in the index?"""
        return any(r.status in (ADDED, UPDATED, REMOVED) for r in self.results)

    @property
    def ok(self) -> bool:
        return self.count(FAILED) == 0

    def summary(self) -> str:
        counts = ", ".join(
            f"{self.count(s)} {s}"
            for s in (ADDED, UPDATED, UNCHANGED, REMOVED, MISSING, FAILED)
            if self.count(s) or s in (ADDED, UPDATED, UNCHANGED)
        )
        verb = "would have" if self.dry_run else "has"
        return (
            f"{counts}. Index {verb} {self.total_passages} passage(s) "
            f"from {self.total_lectures} lecture(s)"
        )


# ---------------------------------------------------------------------------
# Finding files and naming lectures
# ---------------------------------------------------------------------------


def find_transcripts(folder: str | Path, pattern: str = DEFAULT_PATTERN) -> list[Path]:
    """All files under `folder` (any depth) whose name matches `pattern`.

    `pattern` is a glob such as "transcript.json" or "*.json". Results are
    sorted so runs are reproducible regardless of the filesystem's order.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise IngestError(f"not a folder: {folder}")
    # rglob = "recursive glob": searches every subfolder, like `find -name`.
    return sorted(p for p in folder.rglob(pattern) if p.is_file())


def _slug_from_relative(path: Path, root: Path) -> str:
    """`fall/la-05/transcript.json` -> "fall-la-05" (used when ids collide)."""
    rel = path.relative_to(root)
    parts = list(rel.parent.parts)
    if rel.stem != "transcript" or not parts:
        parts.append(rel.stem)
    return "-".join(parts)


def assign_lecture_ids(paths: list[Path], root: str | Path) -> dict[Path, str]:
    """Give every transcript a unique lecture id (see the module docstring)."""
    root = Path(root)
    short = {p: lecture_id_from_path(p) for p in paths}
    counts = Counter(short.values())
    ids = {
        p: (_slug_from_relative(p, root) if counts[sid] > 1 else sid)
        for p, sid in short.items()
    }
    # The fallback is unique because two different files cannot have the same
    # relative path, but a fallback could still equal some *other* short id
    # (a folder literally named "fall-la-05"). Fail loudly in that odd case.
    clashes = [i for i, n in Counter(ids.values()).items() if n > 1]
    if clashes:
        raise IngestError(f"cannot give unique lecture ids to: {clashes}")
    return ids


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes, read in 1 MB blocks so memory stays small."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        # iter(callable, sentinel) keeps calling f.read until it returns b"".
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_titles(path: str | Path | None) -> dict[str, str]:
    """Read an optional JSON file mapping lecture id -> human-readable title.

    Example: {"la-05": "18.06 Lecture 21: Eigenvalues"}. Titles appear in
    citations; a lecture without one is cited by its id.
    """
    if path is None:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise IngestError(f"titles file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise IngestError(f"titles file is not valid JSON: {path} ({error})") from error
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise IngestError('titles file must be a JSON object like {"la-05": "Title"}')
    return data


# ---------------------------------------------------------------------------
# The state file (ingest.json)
# ---------------------------------------------------------------------------


def load_state(index_dir: str | Path) -> dict[str, Any]:
    """Read ingest.json, or an empty state if it is missing or unreadable.

    Losing the state is never fatal: without it every lecture just looks
    "changed" and is re-embedded once, which is slow but correct.
    """
    path = Path(index_dir) / STATE_FILE
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"lectures": {}}
    if state.get("format_version") != STATE_FORMAT_VERSION:
        return {"lectures": {}}
    state.setdefault("lectures", {})
    return state


def save_state(
    index_dir: str | Path,
    embedder_name: str,
    options: SplitOptions,
    lectures: dict[str, dict[str, Any]],
) -> Path:
    """Write ingest.json. No timestamps, so an unchanged index gives the same file."""
    path = Path(index_dir) / STATE_FILE
    state = {
        "format_version": STATE_FORMAT_VERSION,
        "embedder": embedder_name,
        "options": options.to_dict(),
        # Sorted by id so the file diffs cleanly in git if someone commits it.
        "lectures": dict(sorted(lectures.items())),
    }
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return path


def index_exists(index_dir: str | Path) -> bool:
    return (Path(index_dir) / MANIFEST_FILE).exists()


# ---------------------------------------------------------------------------
# The ingest run
# ---------------------------------------------------------------------------


def _open_store(
    index_dir: Path,
    embedder: Embedder | None,
    embedder_name: str | None,
    rebuild: bool,
) -> VectorStore:
    """Load the existing index, or start an empty one.

    The model that built an existing index wins: its vectors are only
    comparable with vectors from the same model. Asking for a different one
    is an error that points at --rebuild, rather than a silent mix of two
    incompatible vector spaces.
    """
    if index_exists(index_dir) and not rebuild:
        manifest = json.loads((index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        wanted = embedder.name if embedder is not None else embedder_name
        if wanted is not None and wanted != manifest.get("embedder"):
            raise IngestError(
                f"{index_dir} was built with {manifest.get('embedder')!r}, not "
                f"{wanted!r}; use --rebuild to re-embed everything with the new model"
            )
        return VectorStore.load(index_dir, embedder=embedder)
    if embedder is None:
        # None means "the model from settings" (NOTETAKER_EMBEDDING_MODEL).
        embedder = make_embedder(embedder_name)
    return VectorStore(embedder)


def ingest_folder(
    folder: str | Path,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    embedder: Embedder | None = None,
    embedder_name: str | None = None,
    pattern: str = DEFAULT_PATTERN,
    titles: dict[str, str] | None = None,
    options: SplitOptions | None = None,
    rebuild: bool = False,
    prune: bool = False,
    dry_run: bool = False,
    on_result: Callable[[LectureResult], None] | None = None,
) -> IngestReport:
    """Bring the index at `index_dir` in line with the transcripts in `folder`.

    - new transcripts are added, changed ones re-indexed, unchanged skipped
    - lectures in the index whose transcript is gone are reported as
      "missing" and kept, unless `prune=True` removes them
    - `rebuild=True` ignores any existing index and starts from scratch
    - `dry_run=True` reports what would happen without embedding or saving
    - a transcript that cannot be read is reported as "failed"; the others
      are still indexed (one bad file should not block a whole semester)

    `embedder` (an object) is mainly for tests; `embedder_name` (e.g.
    "hashing-512") is what the CLI passes. `on_result` is called as each
    lecture finishes, so the CLI can print progress.
    """
    folder = Path(folder)
    index_dir = Path(index_dir)
    titles = titles or {}
    options = options or SplitOptions()

    paths = find_transcripts(folder, pattern)
    if not paths and not prune:
        raise IngestError(f"no files matching {pattern!r} under {folder}")
    ids = assign_lecture_ids(paths, folder)

    store = _open_store(index_dir, embedder, embedder_name, rebuild)
    state = {"lectures": {}} if rebuild else load_state(index_dir)
    previous: dict[str, dict[str, Any]] = state["lectures"]
    options_changed = bool(previous) and state.get("options") != options.to_dict()

    report = IngestReport(
        index_dir=index_dir, embedder=store.embedder.name, dry_run=dry_run
    )
    indexed = set(store.lectures())
    records: dict[str, dict[str, Any]] = {}

    def record(result: LectureResult) -> None:
        report.results.append(result)
        if on_result is not None:
            on_result(result)

    for path in paths:
        lecture_id = ids[path]
        title = titles.get(lecture_id, "")
        try:
            sha = file_sha256(path)
        except OSError as error:
            record(LectureResult(lecture_id, FAILED, str(path), reason=str(error)))
            continue
        old = previous.get(lecture_id)

        # Work out whether (and why) this lecture needs embedding.
        if lecture_id not in indexed:
            status, reason = ADDED, "new"
        elif old is None:
            status, reason = UPDATED, "not in ingest.json"
        elif options_changed:
            status, reason = UPDATED, "passage settings changed"
        elif old.get("sha256") != sha:
            status, reason = UPDATED, "transcript changed"
        elif old.get("title", "") != title:
            status, reason = UPDATED, "title changed"
        else:
            records[lecture_id] = old
            record(LectureResult(lecture_id, UNCHANGED, str(path), old["passages"]))
            continue

        try:
            transcript = Transcript.load_json(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            # json.JSONDecodeError is a ValueError; a JSON file that is not a
            # transcript fails with KeyError/TypeError in Transcript.from_dict.
            why = f"{type(error).__name__}: {error}"
            record(LectureResult(lecture_id, FAILED, str(path), reason=why))
            # Keep the old version searchable rather than dropping the lecture.
            if old is not None and lecture_id in indexed:
                records[lecture_id] = old
            continue

        if dry_run:
            # Count passages without embedding: splitting is cheap, embedding
            # is the expensive step we are skipping.
            n = len(split_passages(transcript, lecture_id, **options.to_dict()))
        else:
            n = store.add_transcript(
                transcript,
                lecture_id=lecture_id,
                title=title,
                source=str(path),
                replace=True,
                **options.to_dict(),
            )
        records[lecture_id] = {
            "source": str(path),
            "sha256": sha,
            "title": title,
            "passages": n,
        }
        record(LectureResult(lecture_id, status, str(path), n, reason))

    # Lectures in the index that no file in the folder produced this run.
    seen = set(ids.values())
    for lecture_id in [i for i in store.lectures() if i not in seen]:
        old = previous.get(lecture_id, {})
        n = sum(1 for p in store.passages if p.lecture_id == lecture_id)
        if prune:
            if not dry_run:
                store.remove_lecture(lecture_id)
            record(LectureResult(lecture_id, REMOVED, old.get("source", ""), n))
        else:
            records[lecture_id] = old or {"source": "", "sha256": "", "passages": n}
            reason = "no transcript in this folder; kept (use --prune to remove)"
            record(LectureResult(lecture_id, MISSING, old.get("source", ""), n, reason))

    if dry_run:
        report.total_passages = sum(r["passages"] for r in records.values())
        report.total_lectures = len(records)
        return report

    report.total_passages = len(store)
    report.total_lectures = len(store.lectures())
    # Save when something changed, or when there is no index on disk yet (so
    # a first run always leaves an index behind), or when ingest.json is
    # stale (e.g. it was deleted). An all-unchanged run writes nothing.
    stale_state = records != previous or options_changed
    if report.changed or not index_exists(index_dir) or rebuild or stale_state:
        store.save(index_dir)
        save_state(index_dir, store.embedder.name, options, records)
        report.saved = True
    return report
