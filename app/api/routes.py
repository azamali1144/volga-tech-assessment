from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, Request, Response, UploadFile, status

from app.api.dependencies import Services, get_services, require_api_key
from app.api.errors import ApiError
from app.api.schemas import (
    ErrorResponse,
    HealthChecks,
    HealthResponse,
    JobCreatedResponse,
    JobListResponse,
    JobResponse,
    JobSummary,
    TranscriptOut,
)
from app.infrastructure.database import MAX_LIST_LIMIT, JobStatus
from app.infrastructure.storage import ObjectTooLargeError
from app.services.audio import AudioProcessingError, probe_duration_seconds

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"
UPLOAD_PATH = f"{API_PREFIX}/transcriptions"


router = APIRouter(prefix=API_PREFIX, dependencies=[Depends(require_api_key)])

ERROR_RESPONSES = {
    code: {"model": ErrorResponse}
    for code in (400, 401, 404, 413, 422, 429, 500)
}


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
    settings = services.settings
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if not filename or extension not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions))
        raise ApiError(
            status.HTTP_400_BAD_REQUEST,
            "unsupported_format",
            f"Unsupported file type {extension or '(none)'!r}. Allowed: {allowed}.",
        )

    job_id = uuid.uuid4().hex
    key = f"{job_id}{extension}"
    try:
        await asyncio.to_thread(
            services.storage.save, key, file.file, settings.max_upload_bytes
        )
    except ObjectTooLargeError:
        raise ApiError(
            status.HTTP_413_CONTENT_TOO_LARGE,
            "file_too_large",
            f"File exceeds the {settings.max_upload_bytes}-byte upload limit.",
        )
    finally:
        await file.close()

    try:
        with services.storage.as_local_file(key) as path:
            duration = await asyncio.to_thread(probe_duration_seconds, path)
    except AudioProcessingError as exc:
        services.storage.delete(key)
        raise AudioProcessingError(exc.error_code, exc.message.replace(key, filename)) from exc

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


@router.get(
    "/transcriptions/{job_id}",
    response_model=JobResponse,
    responses={k: v for k, v in ERROR_RESPONSES.items() if k in (401, 404, 429)},
    summary="Get a job's status and, once completed, its transcript",
)
async def get_transcription(
    job_id: str,
    caller_id: str = Depends(require_api_key),
    services: Services = Depends(get_services),
) -> JobResponse:
    job = services.store.get_job(job_id)
    if job is None or job.user_id != caller_id:
        raise ApiError(status.HTTP_404_NOT_FOUND, "job_not_found", "Job not found.")

    transcript = None
    if job.status == JobStatus.COMPLETED:
        content = await asyncio.to_thread(services.store.get_transcript, job.id)
        if content is not None:
            transcript = TranscriptOut(version=job.trans_version, **content)
    return JobResponse.from_job(job, transcript=transcript)


@router.get(
    "/transcriptions",
    response_model=JobListResponse,
    responses={k: v for k, v in ERROR_RESPONSES.items() if k in (401, 422, 429)},
    summary="List your jobs, newest first",
)
async def list_transcriptions(
    limit: int = Query(20, ge=1, le=MAX_LIST_LIMIT),
    offset: int = Query(0, ge=0),
    status_filter: JobStatus | None = Query(None, alias="status"),
    caller_id: str = Depends(require_api_key),
    services: Services = Depends(get_services),
) -> JobListResponse:
    store = services.store
    jobs = store.list_jobs(caller_id, limit=limit, offset=offset, status=status_filter)
    has_more = bool(
        store.list_jobs(caller_id, limit=1, offset=offset + limit, status=status_filter)
    )
    return JobListResponse(
        items=[JobSummary.from_job(job) for job in jobs],
        limit=limit,
        offset=offset,
        next_offset=offset + limit if has_more else None,
    )


def health(request: Request, response: Response) -> HealthResponse:
    services: Services = request.app.state.services
    workers_running = sum(1 for task in services.worker_tasks if not task.done())
    checks = HealthChecks(database=services.store.ping(), workers=workers_running > 0)
    healthy = checks.database and checks.workers
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if healthy else "unhealthy",
        checks=checks,
        queue_depth=services.queue.size(),
        delayed_retries=services.queue.delayed_count,
        workers_running=workers_running,
    )
