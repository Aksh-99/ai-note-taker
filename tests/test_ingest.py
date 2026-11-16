"""Tests for src/notetaker/ingest.py and the `notetaker ingest` command.

Offline: the HashingEmbedder stands in for the neural model. To check that
unchanged lectures are really skipped (not just reported as skipped), a
counting wrapper records how many texts were sent to the embedder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from notetaker.cli import main
from notetaker.embeddings import HashingEmbedder
from notetaker.ingest import (
    ADDED,
    FAILED,
    MISSING,
    REMOVED,
    STATE_FILE,
    UNCHANGED,
    UPDATED,
    IngestError,
    SplitOptions,
    assign_lecture_ids,
    file_sha256,
    find_transcripts,
    ingest_folder,
    load_state,
    load_titles,
)
from notetaker.transcribe import Segment, Transcript
from notetaker.vectorstore import VectorStore


class CountingEmbedder(HashingEmbedder):
    """A HashingEmbedder that counts how many passages it has embedded."""

    def __init__(self) -> None:
        super().__init__(dim=256)
        self.embedded = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embedded += len(texts)
        return super().embed(texts)


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


LA = [
    "Today we study eigenvalues and eigenvectors of a square matrix.",
    "To find eigenvalues we solve the characteristic polynomial.",
    "Each eigenvalue has an eigenvector that the matrix only stretches.",
]
ML = [
    "Gradient descent updates the weights against the gradient of the loss.",
    "The learning rate controls the step size of each update.",
]
BIO = ["Mitochondria produce ATP through cellular respiration."]


def write_lecture(root: Path, name: str, texts: list[str]) -> Path:
    path = root / name / "transcript.json"
    make_transcript(texts).save_json(path)
    return path


@pytest.fixture
def lectures(tmp_path: Path) -> Path:
    root = tmp_path / "outputs"
    write_lecture(root, "la-05", LA)
    write_lecture(root, "ml-02", ML)
    # Files that are not transcript.json are ignored by default.
    (root / "la-05" / "notes.json").write_text("{}", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_find_transcripts_recursive_and_sorted(lectures: Path) -> None:
    write_lecture(lectures, "deep/nested/bio-01", BIO)
    found = find_transcripts(lectures)
    assert [p.parent.name for p in found] == ["bio-01", "la-05", "ml-02"]
    assert all(p.name == "transcript.json" for p in found)


def test_find_transcripts_custom_pattern(lectures: Path) -> None:
    names = {p.name for p in find_transcripts(lectures, "*.json")}
    assert names == {"transcript.json", "notes.json"}


def test_find_transcripts_rejects_non_folder(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="not a folder"):
        find_transcripts(tmp_path / "nope")


def test_lecture_ids_use_folder_name(lectures: Path) -> None:
    ids = assign_lecture_ids(find_transcripts(lectures), lectures)
    assert sorted(ids.values()) == ["la-05", "ml-02"]


def test_lecture_ids_disambiguate_collisions(tmp_path: Path) -> None:
    a = write_lecture(tmp_path, "fall/la-05", LA)
    b = write_lecture(tmp_path, "spring/la-05", LA)
    c = write_lecture(tmp_path, "ml-02", ML)
    ids = assign_lecture_ids([a, b, c], tmp_path)
    assert ids == {a: "fall-la-05", b: "spring-la-05", c: "ml-02"}


def test_lecture_ids_for_plain_json_files(tmp_path: Path) -> None:
    a = tmp_path / "x" / "lec1.json"
    b = tmp_path / "y" / "lec1.json"
    ids = assign_lecture_ids([a, b], tmp_path)
    assert ids == {a: "x-lec1", b: "y-lec1"}


def test_file_sha256_detects_any_change(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_text("eigenvalue", encoding="utf-8")
    first = file_sha256(path)
    assert first == file_sha256(path)  # deterministic
    assert len(first) == 64
    path.write_text("eigenvalues", encoding="utf-8")
    assert file_sha256(path) != first


@pytest.mark.parametrize(
    "kwargs", [{"max_words": 0}, {"max_seconds": -1}, {"overlap_segments": -1}]
)
def test_split_options_validated(kwargs) -> None:
    with pytest.raises(ValueError):
        SplitOptions(**kwargs)


def test_load_titles(tmp_path: Path) -> None:
    assert load_titles(None) == {}
    path = tmp_path / "titles.json"
    path.write_text(json.dumps({"la-05": "Eigenvalues"}), encoding="utf-8")
    assert load_titles(path) == {"la-05": "Eigenvalues"}


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"la-05": 3}'])
def test_load_titles_rejects_bad_files(tmp_path: Path, content: str) -> None:
    path = tmp_path / "titles.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(IngestError):
        load_titles(path)


def test_load_titles_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="not found"):
        load_titles(tmp_path / "missing.json")


def test_load_state_tolerates_missing_or_corrupt(tmp_path: Path) -> None:
    assert load_state(tmp_path) == {"lectures": {}}
    (tmp_path / STATE_FILE).write_text("{broken", encoding="utf-8")
    assert load_state(tmp_path) == {"lectures": {}}
    (tmp_path / STATE_FILE).write_text('{"format_version": 99}', encoding="utf-8")
    assert load_state(tmp_path) == {"lectures": {}}


# ---------------------------------------------------------------------------
# ingest_folder
# ---------------------------------------------------------------------------


def statuses(report) -> dict[str, str]:
    return {r.lecture_id: r.status for r in report.results}


def test_first_run_indexes_everything(lectures: Path, tmp_path: Path) -> None:
    index = tmp_path / "index"
    report = ingest_folder(lectures, index, embedder=HashingEmbedder(dim=256))
    assert statuses(report) == {"la-05": ADDED, "ml-02": ADDED}
    assert report.saved and report.ok
    store = VectorStore.load(index)
    assert set(store.lectures()) == {"la-05", "ml-02"}
    assert report.total_passages == len(store)
    hit = store.search("how do we find eigenvalues", k=1)[0]
    assert hit.passage.lecture_id == "la-05"
    assert hit.passage.source.endswith("la-05/transcript.json")
    state = load_state(index)
    assert state["embedder"] == "hashing-256"
    assert state["lectures"]["la-05"]["sha256"] == file_sha256(
        lectures / "la-05" / "transcript.json"
    )


def test_second_run_skips_unchanged(lectures: Path, tmp_path: Path) -> None:
    index = tmp_path / "index"
    ingest_folder(lectures, index, embedder=HashingEmbedder(dim=256))
    before = (index / "index.faiss").read_bytes()
    counter = CountingEmbedder()
    report = ingest_folder(lectures, index, embedder=counter)
    assert statuses(report) == {"la-05": UNCHANGED, "ml-02": UNCHANGED}
    assert not report.saved  # nothing changed, nothing written
    assert (index / "index.faiss").read_bytes() == before
    assert counter.embedded == 0


def test_only_new_and_changed_lectures_are_embedded(
    lectures: Path, tmp_path: Path
) -> None:
    index = tmp_path / "index"
    ingest_folder(lectures, index, embedder=HashingEmbedder(dim=256))
    write_lecture(lectures, "bio-01", BIO)
    write_lecture(lectures, "ml-02", [*ML, "Momentum smooths noisy gradient steps."])

    # Load with a counting embedder of the same name to see what gets embedded.
    counter = CountingEmbedder()
    report = ingest_folder(lectures, index, embedder=counter)
    assert statuses(report) == {"bio-01": ADDED, "la-05": UNCHANGED, "ml-02": UPDATED}
    updated = next(r for r in report.results if r.lecture_id == "ml-02")
    assert updated.reason == "transcript changed"
    # Only bio-01 (1 passage) and the new ml-02 passages were embedded.
    expected = sum(r.passages for r in report.results if r.status != UNCHANGED)
    assert counter.embedded == expected

    store = VectorStore.load(index)
    # Re-indexing replaced ml-02 instead of duplicating it.
    ids = [p.passage_id for p in store.passages]
    assert len(ids) == len(set(ids))
    hit = store.search("momentum gradient", k=1)[0]
    assert hit.passage.lecture_id == "ml-02"


def test_title_change_reindexes_and_appears_in_citations(
    lectures: Path, tmp_path: Path
) -> None:
    index = tmp_path / "index"
    ingest_folder(lectures, index, embedder=HashingEmbedder(dim=256))
    report = ingest_folder(
        lectures, index, titles={"la-05": "18.06 Lecture 21: Eigenvalues"}
    )
    assert statuses(report)["la-05"] == UPDATED
    store = VectorStore.load(index)
    hit = store.search("eigenvalues", k=1, lecture_ids=["la-05"])[0]
    assert hit.passage.citation().startswith("18.06 Lecture 21: Eigenvalues [")


def test_changed_split_options_reindex_everything(
    lectures: Path, tmp_path: Path
) -> None:
    index = tmp_path / "index"
    ingest_folder(lectures, index, embedder=HashingEmbedder(dim=256))
    report = ingest_folder(lectures, index, options=SplitOptions(max_words=10))
    assert set(statuses(report).values()) == {UPDATED}
    assert all(r.reason == "passage settings changed" for r in report.results)
    assert load_state(index)["options"]["max_words"] == 10


def test_deleted_transcript_kept_unless_pruned(lectures: Path, tmp_path: Path) -> None:
    index = tmp_path / "index"
    ingest_folder(lectures, index, embedder=HashingEmbedder(dim=256))
    (lectures / "ml-02" / "transcript.json").unlink()

    report = ingest_folder(lectures, index)
    assert statuses(report) == {"la-05": UNCHANGED, "ml-02": MISSING}
    assert "ml-02" in VectorStore.load(index).lectures()

    report = ingest_folder(lectures, index, prune=True)
    assert statuses(report) == {"la-05": UNCHANGED, "ml-02": REMOVED}
    assert VectorStore.load(index).lectures() == ["la-05"]
    assert set(load_state(index)["lectures"]) == {"la-05"}


def test_bad_transcript_fails_alone(lectures: Path, tmp_path: Path) -> None:
    (lectures / "broken").mkdir()
    (lectures / "broken" / "transcript.json").write_text("{not json", "utf-8")
    (lectures / "wrong").mkdir()
    (lectures / "wrong" / "transcript.json").write_text('{"a": 1}', "utf-8")
    index = tmp_path / "index"
    report = ingest_folder(lectures, index, embedder=HashingEmbedder(dim=256))
    got = statuses(report)
    assert got["broken"] == FAILED and got["wrong"] == FAILED
    assert got["la-05"] == ADDED and got["ml-02"] == ADDED
    assert not report.ok
    assert report.saved  # the good lectures were still indexed
    assert set(VectorStore.load(index).lectures()) == {"la-05", "ml-02"}


def test_corrupted_update_keeps_old_version(lectures: Path, tmp_path: Path) -> None:
    index = tmp_path / "index"
    ingest_folder(lectures, index, embedder=HashingEmbedder(dim=256))
    (lectures / "la-05" / "transcript.json").write_text("{oops", "utf-8")
    report = ingest_folder(lectures, index)
    assert statuses(report)["la-05"] == FAILED
    store = VectorStore.load(index)
    assert "la-05" in store.lectures()
    assert store.search("eigenvalues", k=1)[0].passage.lecture_id == "la-05"


def test_dry_run_embeds_and_writes_nothing(lectures: Path, tmp_path: Path) -> None:
    index = tmp_path / "index"
    counter = CountingEmbedder()
    report = ingest_folder(lectures, index, embedder=counter, dry_run=True)
    assert statuses(report) == {"la-05": ADDED, "ml-02": ADDED}
    assert report.total_passages > 0 and report.total_lectures == 2
    assert counter.embedded == 0
    assert not index.exists()


def test_existing_index_keeps_its_model(lectures: Path, tmp_path: Path) -> None:
    index = tmp_path / "index"
    ingest_folder(lectures, index, embedder=HashingEmbedder(dim=256))
    with pytest.raises(IngestError, match="--rebuild"):
        ingest_folder(lectures, index, embedder_name="hashing-512")
    report = ingest_folder(lectures, index, embedder_name="hashing-512", rebuild=True)
    assert set(statuses(report).values()) == {ADDED}
    assert VectorStore.load(index).embedder.name == "hashing-512"


def test_lost_state_file_reindexes_once(lectures: Path, tmp_path: Path) -> None:
    index = tmp_path / "index"
    ingest_folder(lectures, index, embedder=HashingEmbedder(dim=256))
    (index / STATE_FILE).unlink()
    report = ingest_folder(lectures, index)
    assert set(statuses(report).values()) == {UPDATED}
    assert (index / STATE_FILE).exists()
    again = ingest_folder(lectures, index)
    assert set(statuses(again).values()) == {UNCHANGED}


def test_empty_folder_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(IngestError, match="no files matching"):
        ingest_folder(tmp_path / "empty", tmp_path / "index")


def test_progress_callback_sees_every_result(lectures: Path, tmp_path: Path) -> None:
    seen = []
    ingest_folder(
        lectures,
        tmp_path / "index",
        embedder=HashingEmbedder(dim=256),
        on_result=seen.append,
    )
    assert [r.lecture_id for r in seen] == ["la-05", "ml-02"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_ingest_round_trip(lectures: Path, tmp_path: Path, capsys) -> None:
    index = tmp_path / "index"
    args = ["ingest", str(lectures), "--index", str(index), "--embedder", "hashing-256"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "+ la-05: added" in out
    assert "2 added" in out and "Wrote" in out

    assert main(["ingest", str(lectures), "-i", str(index)]) == 0
    out = capsys.readouterr().out
    assert "= la-05: unchanged" in out
    assert "Wrote" not in out


def test_cli_ingest_titles_and_dry_run(lectures: Path, tmp_path: Path, capsys) -> None:
    titles = tmp_path / "titles.json"
    titles.write_text(json.dumps({"la-05": "Eigenvalues"}), encoding="utf-8")
    index = tmp_path / "index"
    code = main(
        [
            "ingest",
            str(lectures),
            "-i",
            str(index),
            "--embedder",
            "hashing-256",
            "--titles",
            str(titles),
            "--dry-run",
        ]
    )
    assert code == 0
    assert "Dry run, nothing saved" in capsys.readouterr().out
    assert not index.exists()


def test_cli_ingest_errors(lectures: Path, tmp_path: Path, capsys) -> None:
    assert main(["ingest", str(tmp_path / "missing")]) == 1
    assert "Error: not a folder" in capsys.readouterr().out

    index = tmp_path / "index"
    main(["ingest", str(lectures), "-i", str(index), "--embedder", "hashing-256"])
    capsys.readouterr()
    code = main(["ingest", str(lectures), "-i", str(index), "--embedder", "x-model"])
    assert code == 1
    assert "--rebuild" in capsys.readouterr().out

    code = main(["ingest", str(lectures), "-i", str(index), "--max-words", "0"])
    assert code == 1
    assert "Error:" in capsys.readouterr().out


def test_cli_ingest_failed_file_gives_exit_code_1(
    lectures: Path, tmp_path: Path, capsys
) -> None:
    (lectures / "bad").mkdir()
    (lectures / "bad" / "transcript.json").write_text("{", encoding="utf-8")
    index = tmp_path / "index"
    code = main(
        ["ingest", str(lectures), "-i", str(index), "--embedder", "hashing-256"]
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "! bad: failed" in out
    assert "1 failed" in out
