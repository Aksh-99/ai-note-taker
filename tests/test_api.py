"""Tests for the FastAPI app (notetaker.api) and `notetaker serve`.

FastAPI's TestClient calls the app in-process, with no network and no
server. Background tasks run right after each response, before the client
call returns, so a job is already finished when the next request is made.
No real Whisper model is loaded: tests pass fake runners or a fake model.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

# Skip this whole file if the API extra is not installed (e.g. the
# fine-tuning CI job, which installs only .[dev,finetune]).
pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from notetaker.api import create_app  # noqa: E402
from notetaker.cli import main  # noqa: E402
from notetaker.config import get_settings  # noqa: E402
from notetaker.jobs import JobStore, transcribe_runner  # noqa: E402
from notetaker.transcribe import Transcript  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)

TRANSCRIPT = {
    "segments": [
        {
            "id": 0,
            "start": 0.0,
            "end": 1.0,
            "text": "Eigenvalues today.",
            "avg_logprob": -0.1,
            "no_speech_prob": 0.0,
            "words": [],
        }
    ],
    "language": "en",
    "language_probability": 1.0,
    "duration": 1.0,
    "model": "fake",
}


def fake_runner(job, audio_path, out_dir):
    """Writes canned results, like transcribe_runner but with no model."""
    Transcript.from_dict(TRANSCRIPT).save_json(out_dir / "transcript.json")
    (out_dir / "transcript.md").write_text(f"# {job.title}\n\nEigenvalues today.\n")
    return {"outputs": ["transcript.json", "transcript.md"], "duration": 1.0}


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


def client_for(store, runner=fake_runner, **kwargs):
    return TestClient(create_app(store, runner, **kwargs))


def upload(client, name="lecture01.mp3", data=b"fake audio", **form):
    return client.post("/jobs", files={"file": (name, data)}, data=form)


def test_health():
    reply = client_for(None).get("/health")
    assert reply.status_code == 200
    assert reply.json()["status"] == "ok"


def test_default_store_lives_under_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTETAKER_OUTPUT_DIR", str(tmp_path / "out"))
    get_settings.cache_clear()
    client = TestClient(create_app(runner=fake_runner))
    job = upload(client).json()
    assert (tmp_path / "out" / "jobs" / job["id"] / "job.json").is_file()


def test_upload_runs_job_and_results_are_fetchable(store):
    client = client_for(store)
    reply = upload(client, title="Week 1", language="en")
    assert reply.status_code == 202
    job = reply.json()
    assert job["status"] == "queued"  # the reply is sent before the work runs
    assert job["title"] == "Week 1"
    assert reply.headers["location"] == f"/jobs/{job['id']}"

    status = client.get(f"/jobs/{job['id']}").json()
    assert status["status"] == "done"
    assert status["outputs"] == ["transcript.json", "transcript.md"]
    assert status["duration"] == 1.0

    transcript = client.get(f"/jobs/{job['id']}/transcript")
    assert transcript.status_code == 200
    assert transcript.json()["segments"][0]["text"] == "Eigenvalues today."

    md = client.get(f"/jobs/{job['id']}/files/transcript.md")
    assert md.status_code == 200
    assert md.headers["content-type"].startswith("text/markdown")
    assert "# Week 1" in md.text


def test_list_jobs(store):
    client = client_for(store)
    first = upload(client, "a.mp3").json()["id"]
    second = upload(client, "b.wav").json()["id"]
    ids = [j["id"] for j in client.get("/jobs").json()["jobs"]]
    assert set(ids) == {first, second}


def test_blank_form_fields_fall_back(store):
    job = upload(client_for(store), "Lecture 2.mp3", title="  ", language="").json()
    assert job["title"] == "Lecture 2"
    assert job["language"] is None


@pytest.mark.parametrize("name", ["notes.pdf", "noextension"])
def test_unsupported_file_type(store, name):
    reply = upload(client_for(store), name)
    assert reply.status_code == 415
    assert "unsupported file type" in reply.json()["detail"]
    assert list(store.root.iterdir()) == []


def test_upload_too_large(store):
    reply = upload(client_for(store, max_upload_mb=1), data=b"x" * (2**20 + 1))
    assert reply.status_code == 413
    assert list(store.root.iterdir()) == []


def test_upload_limit_comes_from_settings(store, monkeypatch):
    monkeypatch.setenv("NOTETAKER_API_MAX_UPLOAD_MB", "0.001")  # about 1 KB
    get_settings.cache_clear()
    assert upload(client_for(store), data=b"x" * 2000).status_code == 413
    assert upload(client_for(store), data=b"x" * 500).status_code == 202


def test_empty_upload(store):
    reply = upload(client_for(store), data=b"")
    assert reply.status_code == 400
    assert "empty" in reply.json()["detail"]


def test_missing_file_field(store):
    assert client_for(store).post("/jobs", data={"title": "x"}).status_code == 422


@pytest.mark.parametrize("job_id", ["0" * 32, "not-a-job"])
def test_unknown_job_is_404(store, job_id):
    client = client_for(store)
    for path in ["", "/transcript", "/files/transcript.md"]:
        assert client.get(f"/jobs/{job_id}{path}").status_code == 404


def test_failed_job_reports_error(store):
    def broken(*_):
        raise RuntimeError("ffmpeg could not read lecture01.mp3")

    client = client_for(store, broken)
    job_id = upload(client).json()["id"]
    status = client.get(f"/jobs/{job_id}").json()
    assert status["status"] == "failed"
    assert "could not read" in status["error"]
    reply = client.get(f"/jobs/{job_id}/transcript")
    assert reply.status_code == 409
    assert "could not read" in reply.json()["detail"]


def test_transcript_before_done_is_409(store):
    # A runner that never got to run: create the job directly in the store.
    import io

    job = store.create("a.mp3", io.BytesIO(b"x"))
    reply = client_for(store).get(f"/jobs/{job.id}/transcript")
    assert reply.status_code == 409
    assert "queued" in reply.json()["detail"]


def test_files_endpoint_refuses_unlisted_files(store):
    client = client_for(store)
    job_id = upload(client).json()["id"]
    for name in ["job.json", "input.mp3", "..%2Fjob.json"]:
        assert client.get(f"/jobs/{job_id}/files/{name}").status_code == 404


class FakeModel:
    """Same shape as faster-whisper's WhisperModel.transcribe()."""

    def transcribe(self, audio, **kwargs):
        self.samples = len(audio)
        segment = SimpleNamespace(
            id=0,
            start=0.0,
            end=0.5,
            text=" A tone.",
            avg_logprob=-0.3,
            no_speech_prob=0.2,
            words=[],
        )
        info = SimpleNamespace(
            language="en", language_probability=1.0, duration=len(audio) / 16000
        )
        return iter([segment]), info


@needs_ffmpeg
def test_end_to_end_real_audio_fake_model(store, tmp_path):
    """Upload a real (generated) wav; ffmpeg decodes it, a fake model 'hears' it."""
    wav = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=0.5", str(wav)],
        check=True,
    )  # fmt: skip
    model = FakeModel()
    client = client_for(store, transcribe_runner(model=model))
    job_id = upload(client, "tone.wav", wav.read_bytes(), title="Tone").json()["id"]
    status = client.get(f"/jobs/{job_id}").json()
    assert status["status"] == "done", status["error"]
    assert model.samples == 8000  # 0.5 s at 16 kHz reached the model
    assert status["duration"] == 0.5
    body = client.get(f"/jobs/{job_id}/transcript").json()
    assert body["segments"][0]["text"] == "A tone."
    assert "A tone." in client.get(f"/jobs/{job_id}/files/transcript.md").text


# ---- notetaker serve -------------------------------------------------------


def test_serve_starts_uvicorn(monkeypatch, capsys):
    calls = {}
    fake_uvicorn = SimpleNamespace(run=lambda app, host, port: calls.update(
        app=app, host=host, port=port))  # fmt: skip
    # Put a fake uvicorn in sys.modules so `import uvicorn` inside run_serve
    # picks it up and no real server starts.
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    assert main(["serve", "--port", "8123", "-m", "tiny"]) == 0
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 8123
    assert "/health" in {route.path for route in calls["app"].routes}
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8123" in out
    assert "tiny" in out


def test_serve_rejects_bad_port(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=None))
    assert main(["serve", "--port", "0"]) == 1
    assert "--port" in capsys.readouterr().out


def test_serve_without_api_extra(monkeypatch, capsys):
    # A None entry in sys.modules makes `import uvicorn` raise ImportError.
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert main(["serve"]) == 1
    assert "pip install" in capsys.readouterr().out


def test_serve_rejects_bad_model(capsys):
    with pytest.raises(SystemExit):
        main(["serve", "-m", "not-a-model-or-folder"])
