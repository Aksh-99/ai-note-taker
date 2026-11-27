"""HTTP API for the note taker, built with FastAPI (Step 7.1a).

Endpoints (open http://localhost:8000/docs for interactive docs, which
FastAPI generates automatically from the code below):

    GET  /health                       is the server up?
    POST /jobs                         upload a recording, start a job (202)
    GET  /jobs                         list jobs, newest first
    GET  /jobs/{id}                    one job's status
    GET  /jobs/{id}/transcript         the transcript as JSON, once done
    GET  /jobs/{id}/files/{name}       download a result file (e.g. transcript.md)

The "submit, then poll" pattern:
    Transcription takes minutes, so POST /jobs saves the file, answers at once
    with status 202 Accepted and the new job's id, and does the work after the
    response has been sent. The client then asks GET /jobs/{id} every few
    seconds until the status is "done" (or "failed"). This keeps every
    request short, which matters because browsers, proxies and free hosting
    platforms cut off requests that stay open too long.

    For now the work runs through FastAPI's `BackgroundTasks` (a function
    queued to run in the same server process right after the response).
    Step 7.2 replaces this with a proper job queue so jobs survive restarts
    and do not compete with requests.

Run it with `notetaker serve` (needs `pip install -e ".[api]"`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from notetaker import __version__
from notetaker.config import get_settings
from notetaker.jobs import (
    Job,
    JobNotFoundError,
    JobRunner,
    JobStore,
    UploadTooLargeError,
    run_job,
    transcribe_runner,
)

# File types we accept. ffmpeg can decode far more, but an allow-list gives a
# clear error for obvious mistakes (uploading a PDF) before any work starts.
AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".mkv"}
)

# Content types for result downloads, so browsers show them sensibly.
_MEDIA_TYPES = {".json": "application/json", ".md": "text/markdown; charset=utf-8"}


def create_app(
    store: JobStore | None = None,
    runner: JobRunner | None = None,
    *,
    max_upload_mb: float | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    An "app factory" instead of a module-level `app = FastAPI()`: each call
    makes a fresh app wired to the given store and runner. Tests pass a store
    in a temp folder and a fake runner, so no model is loaded and nothing is
    written to the real outputs folder. Leaving the arguments empty gives the
    real setup (jobs under NOTETAKER_OUTPUT_DIR/jobs, Whisper as the runner).
    """
    settings = get_settings()
    store = store or JobStore(settings.output_dir / "jobs")
    runner = runner or transcribe_runner()
    limit_mb = settings.api_max_upload_mb if max_upload_mb is None else max_upload_mb
    max_bytes = int(limit_mb * 1024 * 1024)

    app = FastAPI(
        title="notetaker",
        version=__version__,
        description="AI lecture note taker built on free, open models.",
    )

    def get_job_or_404(job_id: str) -> Job:
        try:
            return store.get(job_id)
        except JobNotFoundError:
            # HTTPException turns into a JSON error reply: {"detail": "..."}.
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"no job with id {job_id!r}"
            ) from None

    # Each decorated function below handles one URL. FastAPI reads the type
    # hints to parse and validate the request: `job_id: str` comes from the
    # path, `File(...)` from a multipart upload, `Form(...)` from form fields.
    # `response_model` documents (and filters) what the reply looks like.

    @app.get("/health")
    def health() -> dict[str, str]:
        """Cheap liveness check for monitoring and hosting platforms."""
        return {"status": "ok", "version": __version__}

    @app.post("/jobs", status_code=status.HTTP_202_ACCEPTED, response_model=Job)
    def create_job(
        background: BackgroundTasks,
        response: Response,
        file: Annotated[UploadFile, File(description="Lecture audio or video")],
        title: Annotated[str | None, Form()] = None,
        language: Annotated[
            str | None, Form(description="e.g. 'en'; empty = use the setting")
        ] = None,
    ) -> Job:
        """Upload a recording and start transcribing it in the background."""
        filename = file.filename or ""
        suffix = Path(filename).suffix.lower()
        if suffix not in AUDIO_EXTENSIONS:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"unsupported file type {suffix or '(none)'!r}; "
                f"use one of {', '.join(sorted(AUDIO_EXTENSIONS))}",
            )
        try:
            # `file.file` is the raw upload stream; the store copies it to
            # disk piece by piece, stopping early if it is over the limit.
            job = store.create(
                filename,
                file.file,
                title=(title or "").strip() or None,
                language=(language or "").strip() or None,
                max_bytes=max_bytes,
            )
        except UploadTooLargeError as error:
            # 413 "Content Too Large". Written as a number because Starlette
            # renamed its constant for it between versions.
            raise HTTPException(413, str(error)) from None
        except ValueError as error:  # e.g. an empty file
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from None

        # Queue the work to run after this response is sent.
        background.add_task(run_job, store, job.id, runner)
        # The Location header is the HTTP convention for "your new thing
        # lives at this URL"; clients can poll it.
        response.headers["Location"] = f"/jobs/{job.id}"
        return job

    @app.get("/jobs")
    def list_jobs() -> dict[str, list[Job]]:
        return {"jobs": store.list()}

    @app.get("/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str) -> Job:
        return get_job_or_404(job_id)

    @app.get("/jobs/{job_id}/transcript")
    def get_transcript(job_id: str) -> Any:
        """The transcript JSON (segments with timestamps), once the job is done."""
        job = get_job_or_404(job_id)
        if job.status != "done":
            # 409 Conflict: the request is valid but the job is not in a
            # state that allows it yet. The client should keep polling.
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"job is {job.status}; the transcript is ready when it is done"
                + (f" (error: {job.error})" if job.error else ""),
            )
        path = store.job_dir(job_id) / "transcript.json"
        return FileResponse(path, media_type="application/json")

    @app.get("/jobs/{job_id}/files/{name}")
    def get_file(job_id: str, name: str) -> FileResponse:
        get_job_or_404(job_id)
        try:
            path = store.output_path(job_id, name)
        except FileNotFoundError:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"job has no output named {name!r}"
            ) from None
        return FileResponse(
            path,
            media_type=_MEDIA_TYPES.get(path.suffix, "application/octet-stream"),
            filename=name,
        )

    return app
