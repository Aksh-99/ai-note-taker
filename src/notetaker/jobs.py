"""A tiny on-disk job store for the web API (Step 7.1a).

The problem it solves:
    Transcribing a lecture takes minutes, far longer than a web request should
    stay open. So the API does not answer "here is your transcript"; it answers
    "I have started job 3f2a..., ask me later". Something has to remember each
    job between requests: what was uploaded, whether it is still running, and
    where its results are. That something is this module.

Layout on disk (one folder per job, under NOTETAKER_OUTPUT_DIR/jobs/):

    outputs/jobs/3f2a9c.../
        job.json          the Job record below (status, timestamps, outputs)
        input.mp3         the uploaded file, saved under a fixed name
        transcript.json   results, written by the runner when the job is done
        transcript.md

Why files and not a database?
    One folder per job is easy to inspect by hand, survives a server restart,
    and needs no extra service. A real deployment with many users would swap
    this for Redis or Postgres; because the API only talks to `JobStore`'s
    methods, that swap would not touch the API code. (Hiding storage behind a
    small class like this is often called the "repository pattern".)

No FastAPI imports here, so the store is testable with plain pytest.
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Literal

from pydantic import BaseModel, Field

# The life of a job: queued -> running -> done (or failed).
JobStatus = Literal["queued", "running", "done", "failed"]

# Job ids are uuid4().hex: 32 lowercase hex characters, e.g. "3f2a9c...".
# Checking every id against this pattern before touching the disk blocks
# "path traversal" attacks, where a malicious id like "../../etc" would make
# us read or write outside the jobs folder.
_JOB_ID = re.compile(r"^[0-9a-f]{32}$")

# Copying an upload in pieces of 1 MiB keeps memory flat however big it is.
_COPY_CHUNK_BYTES = 1024 * 1024


class JobNotFoundError(KeyError):
    """No job with this id exists (or the id is malformed)."""


class UploadTooLargeError(ValueError):
    """The uploaded file is bigger than the configured limit."""


def _now() -> str:
    """Current time as an ISO 8601 string in UTC, e.g. 2026-11-26T22:01:05Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Job(BaseModel):
    """Everything the API knows about one uploaded recording.

    A pydantic model rather than a dataclass, because pydantic gives us JSON
    reading and writing with validation for free, and FastAPI can use the
    same class to describe its responses in the auto-generated docs.
    """

    id: str
    status: JobStatus = "queued"
    filename: str = Field(description="The name of the file as uploaded.")
    title: str
    language: str | None = None
    created_at: str
    updated_at: str
    # Seconds of audio, filled in when transcription finishes.
    duration: float | None = None
    # Human-readable reason, only set when status == "failed".
    error: str | None = None
    # Result file names inside the job folder, e.g. ["transcript.json", ...].
    outputs: list[str] = Field(default_factory=list)


# A runner does the real work for one job. It receives the job record, the
# path of the uploaded file and the folder to write results into, and returns
# a dict of fields to record on the job (at least "outputs"). The API is given
# a runner instead of calling Whisper directly, so tests can pass a fake one
# (dependency injection again, as in transcribe.py).
JobRunner = Callable[[Job, Path, Path], dict]


class JobStore:
    """Create, read and update jobs stored as folders under `root`."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # The web server handles requests on several threads at once, and the
        # background worker updates jobs from another thread. A lock makes
        # each read-modify-write of a job.json happen as one uninterruptible
        # unit, so two threads can never interleave and lose an update.
        self._lock = threading.Lock()

    # ---- paths ----------------------------------------------------------

    def job_dir(self, job_id: str) -> Path:
        """The folder of one job. Raises JobNotFoundError for a bad id."""
        if not _JOB_ID.match(job_id):
            raise JobNotFoundError(job_id)
        return self.root / job_id

    def input_path(self, job: Job) -> Path:
        """Where the uploaded file is stored: input<original extension>."""
        return self.job_dir(job.id) / f"input{Path(job.filename).suffix.lower()}"

    def output_path(self, job_id: str, name: str) -> Path:
        """Path of a result file, only if the job actually listed it.

        Checking against `outputs` (an allow-list) means a request can never
        fetch job.json, the raw upload, or anything outside the folder.
        """
        job = self.get(job_id)
        if name not in job.outputs:
            raise FileNotFoundError(name)
        return self.job_dir(job_id) / name

    # ---- create / read --------------------------------------------------

    def create(
        self,
        filename: str,
        source: BinaryIO,
        *,
        title: str | None = None,
        language: str | None = None,
        max_bytes: int | None = None,
    ) -> Job:
        """Save an upload and register a new queued job.

        `source` is any readable binary stream (an open file, or the upload
        stream FastAPI gives us). It is copied in 1 MiB pieces, and copying
        stops as soon as `max_bytes` is exceeded, so a huge upload never
        fills memory or the disk. On any error the half-made folder is
        removed again.
        """
        # Only keep the last part of the name: a client could send
        # "../../evil.mp3", and we never want that path on our disk.
        filename = Path(filename).name or "upload"
        now = _now()
        job = Job(
            id=uuid.uuid4().hex,
            filename=filename,
            title=title or Path(filename).stem,
            language=language or None,
            created_at=now,
            updated_at=now,
        )
        folder = self.job_dir(job.id)
        folder.mkdir(parents=True)
        try:
            written = 0
            with open(self.input_path(job), "wb") as out:
                while chunk := source.read(_COPY_CHUNK_BYTES):
                    written += len(chunk)
                    if max_bytes is not None and written > max_bytes:
                        raise UploadTooLargeError(
                            f"file is larger than the {max_bytes // 2**20} MB limit"
                        )
                    out.write(chunk)
            if written == 0:
                raise ValueError("the uploaded file is empty")
            self._write(job)
        except BaseException:
            # BaseException (not just Exception) so even Ctrl+C cleans up.
            _remove_tree(folder)
            raise
        return job

    def get(self, job_id: str) -> Job:
        path = self.job_dir(job_id) / "job.json"
        if not path.is_file():
            raise JobNotFoundError(job_id)
        return Job.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[Job]:
        """All jobs, newest first."""
        jobs = []
        for folder in self.root.iterdir():
            if _JOB_ID.match(folder.name) and (folder / "job.json").is_file():
                jobs.append(self.get(folder.name))
        # ISO timestamps sort correctly as plain strings; the id breaks ties.
        return sorted(jobs, key=lambda j: (j.created_at, j.id), reverse=True)

    # ---- update ---------------------------------------------------------

    def update(self, job_id: str, **changes: object) -> Job:
        """Change some fields of a job and stamp `updated_at`."""
        with self._lock:
            job = self.get(job_id)
            # model_validate re-checks the types, so a typo such as
            # status="dne" is rejected instead of being saved.
            job = Job.model_validate(
                {**job.model_dump(), **changes, "updated_at": _now()}
            )
            self._write(job)
        return job

    def _write(self, job: Job) -> None:
        """Save job.json atomically.

        Writing straight into job.json would let a reader on another thread
        see a half-written file. Instead we write a temporary file and then
        `os.replace` it over the old one; the operating system guarantees a
        rename is all-or-nothing, so readers see either the old or the new
        version, never a mix.
        """
        path = self.job_dir(job.id) / "job.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)


def run_job(store: JobStore, job_id: str, runner: JobRunner) -> Job:
    """Run one job from start to finish, recording progress in the store.

    queued -> running, then the runner does the work; on success the job
    becomes "done" with whatever the runner reported, otherwise "failed"
    with the error message. Catching every Exception here is deliberate:
    this runs in the background after the HTTP response was sent, so an
    uncaught error would vanish into the server log and leave the job stuck
    on "running" forever. Recording it lets the user see what went wrong.
    """
    job = store.update(job_id, status="running", error=None)
    try:
        result = runner(job, store.input_path(job), store.job_dir(job_id))
    except Exception as error:  # noqa: BLE001 (see docstring)
        message = str(error) or type(error).__name__
        return store.update(job_id, status="failed", error=message)
    return store.update(job_id, status="done", **result)


def transcribe_runner(
    model_name: str | None = None, model: object | None = None
) -> JobRunner:
    """Build the real runner: Whisper transcription plus transcript files.

    This is a "factory": a function that returns a configured function. The
    returned `run` remembers `model_name` and `model` (a closure), so the API
    can hold one runner and call it for every job. Loading the model is
    cached inside `transcribe.load_model`, so only the first job pays for it.
    """

    def run(job: Job, audio_path: Path, out_dir: Path) -> dict:
        # Imported inside so creating the API does not import faster-whisper.
        from notetaker.export import write_outputs
        from notetaker.transcribe import transcribe

        transcript = transcribe(
            audio_path, model=model, language=job.language, model_name=model_name
        )
        json_path, md_path = write_outputs(transcript, out_dir, title=job.title)
        return {
            "outputs": [json_path.name, md_path.name],
            "duration": round(transcript.duration, 3),
        }

    return run


def _remove_tree(folder: Path) -> None:
    """Delete a job folder (flat: it never has subfolders)."""
    if not folder.exists():
        return
    for item in folder.iterdir():
        item.unlink()
    folder.rmdir()
