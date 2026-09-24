"""FastAPI application: HTTP routes, auth/rate-limit dependency, and wiring.

Run locally with:  uvicorn app.main:app --reload

The API process does only fast work per request (validate, store the upload,
insert a row, enqueue an id) and answers ``202 Accepted``; transcription runs
on ``Worker`` tasks started in the lifespan below. In production the workers
would be separate processes/containers consuming a durable queue.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    Security,
    UploadFile,
    status,
)
from fastapi.security import APIKeyHeader
from starlette.responses import JSONResponse

from app.audio import AudioProcessingError, probe_duration_seconds
from app.config import Settings, get_settings
from app.queue_backend import InMemoryQueue
from app.rate_limit import RateLimiter
from app.schemas import ErrorResponse, JobCreatedResponse
from app.storage_backend import LocalDiskStorage, ObjectTooLargeError
from app.store import JobStatus, JobStore
from app.transcription_engine import TranscriptionEngine, get_engine
from app.worker import TranscriptionPipeline, Worker, write_dead_letter

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"
UPLOAD_PATH = f"{API_PREFIX}/transcriptions"
# Multipart framing (boundaries, part headers) on top of the file itself.
MULTIPART_OVERHEAD_BYTES = 64 * 1024
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 10.0


@dataclass
class Services:
    """Everything the routes need, built once in the lifespan."""

    settings: Settings
    store: JobStore
    storage: LocalDiskStorage
    queue: InMemoryQueue
    limiter: RateLimiter
    workers: list[Worker] = field(default_factory=list)
    worker_tasks: list[asyncio.Task[None]] = field(default_factory=list)


def get_services(request: Request) -> Services:
    return request.app.state.services


# --- auth + rate limiting ----------------------------------------------------

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,  # we raise our own 401 with a consistent error body
    description="API key issued to the calling service.",
)


def caller_id_for(api_key: str) -> str:
    """Stable, non-secret identifier for an API key.

    Jobs are owned by this id, so the database never stores raw keys: a
    leaked database dump doesn't leak working credentials.
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def require_api_key(
    response: Response,
    api_key: str | None = Security(api_key_header),
    services: Services = Depends(get_services),
) -> str:
    """Authenticate the caller, then apply their rate limit. Returns caller id.

    Auth and rate limiting live in one dependency attached to the router, so
    any route added to it gets both; neither can be forgotten per-route.
    """
    if not api_key or not any(
        # Constant-time comparison: response timing doesn't reveal how much
        # of a guessed key was correct.
        hmac.compare_digest(api_key.encode("utf-8"), valid.encode("utf-8"))
        for valid in services.settings.api_keys
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    caller_id = caller_id_for(api_key)
    decision = services.limiter.check(caller_id)
    rate_headers = {
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
    }
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry in {decision.retry_after_seconds}s.",
            headers={**rate_headers, "Retry-After": str(decision.retry_after_seconds)},
        )
    response.headers.update(rate_headers)
    return caller_id


router = APIRouter(prefix=API_PREFIX, dependencies=[Depends(require_api_key)])

ERROR_RESPONSES = {
    code: {"model": ErrorResponse}
    for code in (400, 401, 404, 413, 422, 429)
}


# --- routes ------------------------------------------------------------------


@router.post(
    "/transcriptions",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobCreatedResponse,
    responses={k: v for k, v in ERROR_RESPONSES.items() if k != 404},
    summary="Upload audio for transcription",
)
async def create_transcription(
    file: UploadFile = File(..., description="Audio file: wav, mp3, m4a, flac, ogg or mp4."),
    caller_id: str = Depends(require_api_key),
    services: Services = Depends(get_services),
) -> JobCreatedResponse:
    """Store the upload, queue a job, and return immediately with ``202``.

    Poll ``status_url`` for progress and the transcript.
    """
    settings = services.settings
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if not filename or extension not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type {extension or '(none)'!r}. Allowed: {allowed}.",
        )

    job_id = uuid.uuid4().hex
    # The storage key comes from the job id, never from the client's filename.
    key = f"{job_id}{extension}"
    try:
        await asyncio.to_thread(
            services.storage.save, key, file.file, settings.max_upload_bytes
        )
    except ObjectTooLargeError:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_bytes}-byte upload limit.",
        )
    finally:
        await file.close()

    # Cheap early validation: a header read, not a decode. Corrupt or
    # mislabeled files are rejected now with a clear error instead of being
    # queued and failing minutes later.
    try:
        with services.storage.as_local_file(key) as path:
            duration = await asyncio.to_thread(probe_duration_seconds, path)
    except AudioProcessingError as exc:
        services.storage.delete(key)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            # Report the caller's filename, not our internal storage key.
            detail=exc.message.replace(key, filename),
        )

    job = services.store.create_job(
        user_id=caller_id,
        original_filename=filename,
        file_path=key,
        duration_seconds=round(duration, 2),
        job_id=job_id,
    )
    await services.queue.enqueue(job.id)
    logger.info(
        "job queued",
        extra={"job_id": job.id, "caller_id": caller_id, "duration_seconds": job.duration_seconds},
    )
    return JobCreatedResponse(
        job_id=job.id,
        status=job.status,
        status_url=f"{API_PREFIX}/transcriptions/{job.id}",
    )


# --- app assembly ------------------------------------------------------------


async def recover_unfinished_jobs(services: Services) -> int:
    """Re-queue work that was in flight when the process last stopped.

    The in-memory queue is lost on restart but the database isn't, so it's
    the source of truth. A job left in ``processing`` was interrupted
    mid-attempt: that counts as a failed attempt, so a file that crashes the
    process every time still ends up dead-lettered instead of looping.
    """
    settings = services.settings
    requeued = 0
    for job in services.store.list_unfinished_jobs():
        if job.status == JobStatus.PROCESSING:
            if job.retry_count >= settings.max_retries:
                failed = services.store.mark_failed(
                    job.id, "interrupted", "Processing was interrupted and retries are exhausted."
                )
                write_dead_letter(
                    settings.dead_letter_dir,
                    failed,
                    attempts=job.retry_count + 1,
                    reason="retries_exhausted",
                    source="startup-recovery",
                )
                continue
            services.store.mark_retrying(
                job.id, "interrupted", "Processing was interrupted by a service restart."
            )
        await services.queue.enqueue(job.id)
        requeued += 1
    return requeued


def create_app(
    settings: Settings | None = None,
    engine: TranscriptionEngine | None = None,
) -> FastAPI:
    """Application factory. Tests pass their own settings/engine."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or get_settings()
        cfg.ensure_dirs()
        services = Services(
            settings=cfg,
            store=JobStore(
                cfg.db_path,
                transcript_dir=cfg.transcript_dir,
                inline_max_chars=cfg.inline_transcript_max_chars,
            ),
            storage=LocalDiskStorage(cfg.audio_dir),
            queue=InMemoryQueue(),
            limiter=RateLimiter(cfg.rate_limit_requests, cfg.rate_limit_window_seconds),
        )
        pipeline = TranscriptionPipeline.from_settings(engine or get_engine(cfg), cfg)
        for i in range(cfg.worker_count):
            worker = Worker(
                services.store,
                services.queue,
                services.storage,
                pipeline,
                max_retries=cfg.max_retries,
                retry_backoff_base_seconds=cfg.retry_backoff_base_seconds,
                dead_letter_dir=cfg.dead_letter_dir,
                name=f"worker-{i}",
            )
            services.workers.append(worker)
            services.worker_tasks.append(asyncio.create_task(worker.run_forever()))
        app.state.services = services

        recovered = await recover_unfinished_jobs(services)
        logger.info(
            "service started",
            extra={
                "engine": cfg.transcription_engine,
                "workers": cfg.worker_count,
                "recovered_jobs": recovered,
            },
        )
        try:
            yield
        finally:
            for worker in services.workers:
                worker.stop()
            done, pending = await asyncio.wait(
                services.worker_tasks, timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS
            )
            for task in pending:  # a job still running past the grace period
                task.cancel()
            await services.queue.close()
            services.store.close()
            logger.info("service stopped")

    app = FastAPI(
        title="Audio Transcription Service",
        version="1.0.0",
        description=(
            "Asynchronous speech-to-text: upload audio, receive a job id, poll "
            "for the transcript with per-segment timestamps."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def reject_oversized_uploads(request: Request, call_next):
        """Reject an oversized upload from its Content-Length, before the body
        is read. FastAPI parses the whole multipart body before the route
        runs, so the size cap inside the route alone would still accept (and
        spool to disk) a huge body first. Uploads without Content-Length
        (chunked encoding) are still capped while being saved."""
        if request.method == "POST" and request.url.path == UPLOAD_PATH:
            declared = request.headers.get("content-length")
            limit = request.app.state.services.settings.max_upload_bytes
            if declared and declared.isdigit() and int(declared) > limit + MULTIPART_OVERHEAD_BYTES:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "error_code": "file_too_large",
                        "detail": f"File exceeds the {limit}-byte upload limit.",
                    },
                )
        return await call_next(request)

    app.include_router(router)
    return app


app = create_app()
