"""Tests for the on-disk job store (notetaker.jobs). No web server, no model."""

from __future__ import annotations

import io
import json
import threading

import pytest

from notetaker.jobs import (
    JobNotFoundError,
    JobStore,
    UploadTooLargeError,
    run_job,
    transcribe_runner,
)
from notetaker.transcribe import Segment, Transcript


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs")


def make_job(store, name="lecture01.mp3", data=b"fake audio", **kwargs):
    return store.create(name, io.BytesIO(data), **kwargs)


def test_create_saves_upload_and_record(store):
    job = make_job(store, title="Linear Algebra 1", language="en")
    assert job.status == "queued"
    assert job.title == "Linear Algebra 1"
    assert job.language == "en"
    folder = store.job_dir(job.id)
    assert (folder / "input.mp3").read_bytes() == b"fake audio"
    saved = json.loads((folder / "job.json").read_text())
    assert saved["id"] == job.id
    assert saved["filename"] == "lecture01.mp3"


def test_title_defaults_to_file_stem_and_blank_language_is_none(store):
    job = make_job(store, "Lecture 3.M4A", language="")
    assert job.title == "Lecture 3"
    assert job.language is None
    # Extension is lower-cased for the stored copy.
    assert store.input_path(job).name == "input.m4a"


def test_path_parts_in_filename_are_dropped(store):
    job = make_job(store, "../../evil.mp3")
    assert job.filename == "evil.mp3"
    assert store.input_path(job).parent == store.job_dir(job.id)


def test_copy_handles_more_than_one_chunk(store):
    data = bytes(range(256)) * 9000  # about 2.2 MiB, so 3 read chunks
    job = make_job(store, data=data)
    assert store.input_path(job).read_bytes() == data


def test_too_large_upload_is_rejected_and_cleaned_up(store):
    with pytest.raises(UploadTooLargeError, match="1 MB"):
        make_job(store, data=b"x" * (2 * 2**20), max_bytes=2**20)
    assert list(store.root.iterdir()) == []


def test_upload_exactly_at_limit_is_accepted(store):
    job = make_job(store, data=b"x" * 100, max_bytes=100)
    assert store.get(job.id).status == "queued"


def test_empty_upload_is_rejected_and_cleaned_up(store):
    with pytest.raises(ValueError, match="empty"):
        make_job(store, data=b"")
    assert list(store.root.iterdir()) == []


def test_get_unknown_and_malformed_ids(store):
    with pytest.raises(JobNotFoundError):
        store.get("0" * 32)
    for bad in ["../etc", "ABC", "", "0" * 31, "g" * 32]:
        with pytest.raises(JobNotFoundError):
            store.get(bad)


def test_list_is_newest_first(store, monkeypatch):
    times = iter(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"])
    monkeypatch.setattr("notetaker.jobs._now", lambda: next(times))
    first = make_job(store, "a.mp3")
    second = make_job(store, "b.mp3")
    assert [j.id for j in store.list()] == [second.id, first.id]


def test_list_ignores_stray_folders(store):
    make_job(store)
    (store.root / "not-a-job").mkdir()
    (store.root / ("a" * 32)).mkdir()  # job-shaped name but no job.json
    assert len(store.list()) == 1


def test_update_changes_fields_and_validates(store):
    job = make_job(store)
    updated = store.update(job.id, status="running")
    assert updated.status == "running"
    assert store.get(job.id).status == "running"
    with pytest.raises(ValueError):
        store.update(job.id, status="dne")
    assert store.get(job.id).status == "running"  # bad update not saved
    assert not list(store.job_dir(job.id).glob("*.tmp"))


def test_concurrent_updates_do_not_corrupt(store):
    job = make_job(store)

    def worker(i):
        for _ in range(20):
            store.update(job.id, duration=float(i))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.get(job.id).duration in {0.0, 1.0, 2.0, 3.0}


def test_output_path_only_allows_listed_outputs(store):
    job = make_job(store)
    store.update(job.id, outputs=["transcript.md"])
    assert store.output_path(job.id, "transcript.md").name == "transcript.md"
    for name in ["job.json", "input.mp3", "../job.json"]:
        with pytest.raises(FileNotFoundError):
            store.output_path(job.id, name)


def test_run_job_success(store):
    job = make_job(store)
    seen = {}

    def runner(j, audio_path, out_dir):
        seen["status"] = store.get(j.id).status
        seen["audio"] = audio_path.read_bytes()
        (out_dir / "result.txt").write_text("hi")
        return {"outputs": ["result.txt"], "duration": 12.5}

    done = run_job(store, job.id, runner)
    assert seen == {"status": "running", "audio": b"fake audio"}
    assert done.status == "done"
    assert done.outputs == ["result.txt"]
    assert done.duration == 12.5
    assert done.error is None


def test_run_job_failure_is_recorded(store):
    job = make_job(store)

    def boom(*_):
        raise RuntimeError("ffmpeg could not read this file")

    failed = run_job(store, job.id, boom)
    assert failed.status == "failed"
    assert failed.error == "ffmpeg could not read this file"


def test_run_job_failure_without_message_uses_type_name(store):
    job = make_job(store)

    def boom(*_):
        raise KeyError

    assert run_job(store, job.id, boom).error == "KeyError"


def test_transcribe_runner_writes_transcript_files(store, monkeypatch):
    job = make_job(store, title="Week 1", language="en")
    calls = {}

    def fake_transcribe(audio, model=None, language=None, model_name=None, **kw):
        calls.update(audio=audio, language=language, model_name=model_name)
        return Transcript(
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=2.0,
                    text="Hello class.",
                    avg_logprob=-0.1,
                    no_speech_prob=0.0,
                )
            ],
            language="en",
            language_probability=1.0,
            duration=2.0,
            model="tiny",
        )

    monkeypatch.setattr("notetaker.transcribe.transcribe", fake_transcribe)
    done = run_job(store, job.id, transcribe_runner(model_name="tiny"))
    assert done.status == "done", done.error
    assert done.outputs == ["transcript.json", "transcript.md"]
    assert done.duration == 2.0
    assert calls["language"] == "en"
    assert calls["model_name"] == "tiny"
    assert calls["audio"] == store.input_path(job)
    md = store.output_path(job.id, "transcript.md").read_text()
    assert "Week 1" in md and "Hello class." in md
