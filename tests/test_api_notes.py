"""Tests for the notes stage of API jobs (Step 7.1b).

Two kinds of fake keep these fast and offline:

* a fake transcription runner that writes a transcript.json (no Whisper);
* fake LLM clients: a `FakeLLMClient` for simple cases, and the recorded
  cassette from Step 4.4 (`ReplayClient`) for an end-to-end check that an
  API job writes exactly the same files as `notetaker agent` does.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from notetaker.agent import STEP_ORDER, PipelineOptions  # noqa: E402
from notetaker.api import create_app, parse_job_options, parse_steps  # noqa: E402
from notetaker.cassette import ReplayClient  # noqa: E402
from notetaker.config import get_settings  # noqa: E402
from notetaker.jobs import JobStore, notes_runner, run_job  # noqa: E402
from notetaker.llm import FakeLLMClient, LLMConfigError  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "agent"
FIXTURE_OPTIONS = json.loads((FIXTURES / "options.json").read_text(encoding="utf-8"))


def fixture_transcriber(job, audio_path, out_dir):
    """Pretend Whisper heard the fixture lecture: copy its transcript."""
    shutil.copy(FIXTURES / "transcript.json", out_dir / "transcript.json")
    return {"outputs": ["transcript.json"], "duration": 210.0}


def no_client():
    raise AssertionError("an LLM client was built for a job that needs none")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in ("NOTETAKER_OUTPUT_DIR", "NOTETAKER_API_MAX_UPLOAD_MB"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs")


def upload(client, name="lecture21.mp3", data=b"fake audio", **form):
    return client.post("/jobs", files={"file": (name, data)}, data=form)


# ---- parsing the form fields ------------------------------------------------


def test_parse_steps():
    assert parse_steps(None) == []
    assert parse_steps("  ") == []
    assert parse_steps("all") == list(STEP_ORDER)
    assert parse_steps("ALL") == list(STEP_ORDER)
    # Any order and duplicates in, pipeline order out.
    assert parse_steps("quiz, notes,quiz") == ["notes", "quiz"]
    with pytest.raises(ValueError, match="summary"):
        parse_steps("notes,summary")
    with pytest.raises(ValueError, match="unknown"):
        parse_steps("all,notes")  # "all" only works on its own


def test_parse_job_options():
    options = parse_job_options("tasks", "2026-11-13", "mon, wed,fri")
    assert options["steps"] == ["tasks"]
    assert options["lecture_date"].isoformat() == "2026-11-13"
    assert options["class_days"] == (0, 2, 4)
    empty = parse_job_options("", "", "")
    assert empty == {"steps": [], "lecture_date": None, "class_days": ()}
    with pytest.raises(ValueError, match="lecture_date"):
        parse_job_options("", "13/11/2026", "")
    with pytest.raises(ValueError, match="weekday"):
        parse_job_options("", "", "mon,funday")


# ---- the runner -------------------------------------------------------------


def test_job_without_steps_is_transcript_only(store):
    runner = notes_runner(fixture_transcriber, client_factory=no_client)
    with open(FIXTURES / "transcript.json", "rb") as fake_audio:
        job = store.create("a.mp3", fake_audio)
    done = run_job(store, job.id, runner)
    assert done.status == "done"
    assert done.outputs == ["transcript.json"]
    assert done.step_errors == []


def test_offline_tasks_need_no_llm(store):
    """task_llm=False: the tasks stage uses keyword rules, no client at all."""
    runner = notes_runner(
        fixture_transcriber,
        client_factory=no_client,
        options=PipelineOptions(task_llm=False),
    )
    with open(FIXTURES / "transcript.json", "rb") as fake_audio:
        job = store.create("a.mp3", fake_audio, steps=["tasks"])
    done = run_job(store, job.id, runner)
    assert done.status == "done", done.error
    assert done.step_errors == []
    for name in ("tasks.json", "todo.json", "tasks.ics", "pipeline.json"):
        assert name in done.outputs
        assert (store.job_dir(job.id) / name).is_file()
    assert "notes.json" not in done.outputs


def test_missing_llm_key_keeps_the_transcript(store):
    def unconfigured():
        raise LLMConfigError("llm_provider=groq needs a free key")

    runner = notes_runner(fixture_transcriber, client_factory=unconfigured)
    with open(FIXTURES / "transcript.json", "rb") as fake_audio:
        job = store.create("a.mp3", fake_audio, steps=["notes"])
    done = run_job(store, job.id, runner)
    assert done.status == "done"
    assert done.outputs == ["transcript.json"]
    assert done.step_errors[0]["step"] == "llm"
    assert "free key" in done.step_errors[0]["error"]


def test_failing_stage_is_recorded_and_others_kept(store):
    """An LLM that only ever says garbage: every LLM stage fails softly."""
    client = FakeLLMClient(replies=["this is not JSON"])
    runner = notes_runner(fixture_transcriber, client_factory=lambda: client)
    with open(FIXTURES / "transcript.json", "rb") as fake_audio:
        job = store.create("a.mp3", fake_audio, steps=["notes", "quiz"])
    done = run_job(store, job.id, runner)
    assert done.status == "done"
    assert [e["step"] for e in done.step_errors] == ["notes", "quiz"]
    # The transcript and the run summary are still there.
    assert done.outputs == ["transcript.json", "pipeline.json"]


# ---- through the HTTP API ---------------------------------------------------


def replay_app(store):
    """The app with the fixture transcript and the recorded LLM replies."""
    options = PipelineOptions(
        chunk_words=FIXTURE_OPTIONS["chunk_words"],
        max_repairs=FIXTURE_OPTIONS["max_repairs"],
        reduce=FIXTURE_OPTIONS["reduce"],
        seed=FIXTURE_OPTIONS["seed"],
    )
    runner = notes_runner(
        fixture_transcriber,
        client_factory=lambda: ReplayClient(FIXTURES / "cassette.json"),
        options=options,
    )
    return TestClient(create_app(store, runner))


def test_api_job_with_all_steps_matches_the_agent_golden_files(store):
    client = replay_app(store)
    reply = upload(
        client,
        title=FIXTURE_OPTIONS["title"],
        steps="all",
        lecture_date=FIXTURE_OPTIONS["lecture_date"],
        class_days="mon,wed,fri",
    )
    assert reply.status_code == 202
    created = reply.json()
    assert created["steps"] == list(STEP_ORDER)
    assert created["lecture_date"] == "2026-11-13"
    assert created["class_days"] == [0, 2, 4]

    job = client.get(f"/jobs/{created['id']}").json()
    assert job["status"] == "done", job["error"]
    assert job["step_errors"] == []
    expected = sorted(p.name for p in (FIXTURES / "expected").iterdir())
    assert sorted(job["outputs"]) == sorted(["transcript.json", *expected])

    # Same prompts, same recorded replies -> same files as `notetaker agent`.
    for name in ("notes.json", "flashcards.csv", "quiz.json", "todo.json"):
        got = client.get(f"/jobs/{created['id']}/files/{name}")
        assert got.status_code == 200, name
        want = (FIXTURES / "expected" / name).read_text(encoding="utf-8")
        assert got.text == want, f"{name} differs from the golden file"

    notes = client.get(f"/jobs/{created['id']}/notes")
    assert notes.status_code == 200
    assert notes.json() == json.loads(
        (FIXTURES / "expected" / "notes.json").read_text(encoding="utf-8")
    )

    # Downloads get content types that apps understand.
    csv = client.get(f"/jobs/{created['id']}/files/flashcards.csv")
    assert csv.headers["content-type"].startswith("text/csv")
    ics = client.get(f"/jobs/{created['id']}/files/tasks.ics")
    assert ics.headers["content-type"].startswith("text/calendar")
    assert "BEGIN:VCALENDAR" in ics.text


@pytest.mark.parametrize(
    ("form", "message"),
    [
        ({"steps": "notes,summary"}, "unknown step"),
        ({"steps": "all", "lecture_date": "next friday"}, "lecture_date"),
        ({"steps": "tasks", "class_days": "mon,someday"}, "weekday"),
    ],
)
def test_bad_notes_fields_are_400_and_save_nothing(store, form, message):
    client = TestClient(create_app(store, fixture_transcriber))
    reply = upload(client, **form)
    assert reply.status_code == 400
    assert message in reply.json()["detail"]
    assert list(store.root.iterdir()) == []


def test_notes_endpoint_errors(store):
    def unconfigured():
        raise LLMConfigError("no key")

    client = TestClient(
        create_app(store, notes_runner(fixture_transcriber, unconfigured))
    )
    # Transcript-only job: notes were never asked for.
    plain = upload(client).json()["id"]
    reply = client.get(f"/jobs/{plain}/notes")
    assert reply.status_code == 404
    assert "steps=all" in reply.json()["detail"]

    # Notes asked for, but the LLM could not be reached: 404 with the reason.
    failed = upload(client, steps="notes").json()["id"]
    reply = client.get(f"/jobs/{failed}/notes")
    assert reply.status_code == 404
    assert "no key" in reply.json()["detail"]

    # Unknown job.
    assert client.get(f"/jobs/{'0' * 32}/notes").status_code == 404


def test_notes_before_done_is_409(store):
    client = TestClient(create_app(store, fixture_transcriber))
    job = upload(client, steps="notes").json()["id"]
    store.update(job, status="running")
    reply = client.get(f"/jobs/{job}/notes")
    assert reply.status_code == 409
    assert "running" in reply.json()["detail"]
