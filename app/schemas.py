"""Pydantic request/response models for the HTTP API.

These are the public contract: FastAPI validates against them and generates
the OpenAPI docs (``/docs``) from them. They're deliberately separate from the
internal ``Job`` / ``TranscriptionResult`` types, so internals can change
without breaking callers, and fields that must not leak (storage keys, user
ids, raw internal error text) never reach a response.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.store import Job, JobStatus

# Internal failures are recorded in full on the job row and in logs; callers
# get a generic message instead of stack-trace-flavoured exception text.
INTERNAL_ERROR_MESSAGE = "Transcription failed due to an internal error."


class SegmentOut(BaseModel):
    id: int = Field(description="Position of the segment in the transcript, from 0.")
    start: float = Field(description="Start time in seconds from the beginning of the audio.")
    end: float = Field(description="End time in seconds from the beginning of the audio.")
    text: str


class TranscriptOut(BaseModel):
    version: int = Field(description="Transcript version; increments if the job is reprocessed.")
    text: str = Field(description="Full transcript text.")
    language: str | None = Field(default=None, description="Detected language code, e.g. 'en'.")
    duration: float | None = Field(default=None, description="Audio duration in seconds.")
    segments: list[SegmentOut]


class JobError(BaseModel):
    code: str = Field(description="Stable, machine-readable error code, e.g. 'invalid_audio'.")
    message: str


class JobSummary(BaseModel):
    """A job without its transcript: used in listings and as a base."""

    model_config = ConfigDict(use_enum_values=True)

    job_id: str
    status: JobStatus
    original_filename: str
    duration_seconds: float | None = None
    language: str | None = None
    retry_count: int = Field(description="Retries used so far (attempts - 1).")
    error: JobError | None = Field(
        default=None,
        description="Why the last attempt failed. Present while retrying and once failed.",
    )
    created_at: datetime
    updated_at: datetime
    failed_at: datetime | None = None

    @classmethod
    def from_job(cls, job: Job, **extra: Any) -> "JobSummary":
        error = None
        if job.error_code:
            message = (
                INTERNAL_ERROR_MESSAGE
                if job.error_code == "internal_error"
                else job.error_message or ""
            )
            error = JobError(code=job.error_code, message=message)
        return cls(
            job_id=job.id,
            status=job.status,
            original_filename=job.original_filename,
            duration_seconds=job.duration_seconds,
            language=job.language,
            retry_count=job.retry_count,
            error=error,
            created_at=job.created_at,
            updated_at=job.updated_at,
            failed_at=job.failed_at,
            **extra,
        )


class JobResponse(JobSummary):
    """Full job status, including the transcript once ``status == completed``."""

    transcript: TranscriptOut | None = Field(
        default=None, description="Present only when the job has completed."
    )

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "job_id": "3f1c2b9e8d7a4c6b9e0f1a2b3c4d5e6f",
                "status": "completed",
                "original_filename": "sales-call.mp3",
                "duration_seconds": 5.2,
                "language": "en",
                "retry_count": 0,
                "error": None,
                "created_at": "2026-09-24T10:00:00.000+00:00",
                "updated_at": "2026-09-24T10:00:04.512+00:00",
                "failed_at": None,
                "transcript": {
                    "version": 1,
                    "text": "Hello, thanks for calling. How can I help?",
                    "language": "en",
                    "duration": 5.2,
                    "segments": [
                        {"id": 0, "start": 0.0, "end": 2.1, "text": "Hello, thanks for calling."},
                        {"id": 1, "start": 2.4, "end": 5.2, "text": "How can I help?"},
                    ],
                },
            }
        },
    )


class JobCreatedResponse(BaseModel):
    """Returned with ``202 Accepted`` as soon as the upload is stored and queued."""

    model_config = ConfigDict(use_enum_values=True)

    job_id: str
    status: JobStatus
    status_url: str = Field(description="Poll this URL for progress and the result.")


class JobListResponse(BaseModel):
    items: list[JobSummary]
    limit: int
    offset: int
    next_offset: int | None = Field(
        default=None, description="Pass as `offset` to fetch the next page; null on the last page."
    )


class HealthChecks(BaseModel):
    database: bool
    workers: bool


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when every check passes, otherwise 'unhealthy'.")
    checks: HealthChecks
    queue_depth: int = Field(description="Jobs waiting to be picked up by a worker.")
    delayed_retries: int = Field(description="Failed attempts waiting out their backoff.")
    workers_running: int


class ErrorResponse(BaseModel):
    """Body of every non-2xx response."""

    error_code: str = Field(description="Stable, machine-readable error code.")
    detail: str = Field(description="Human-readable explanation.")
