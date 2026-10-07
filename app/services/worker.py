from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from app.infrastructure.database import InvalidTransitionError, Job, JobNotFoundError, JobStore
from app.infrastructure.queue import JobQueue
from app.infrastructure.storage import ObjectStorage
from app.services.pipeline import TranscriptionPipeline

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        store: JobStore,
        queue: JobQueue,
        storage: ObjectStorage,
        pipeline: TranscriptionPipeline,
        *,
        max_retries: int = 3,
        retry_backoff_base_seconds: float = 2.0,
        dead_letter_dir: str | Path = "storage/dead_letter",
        poll_timeout_seconds: float = 1.0,
        name: str = "worker",
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.store = store
        self.queue = queue
        self.storage = storage
        self.pipeline = pipeline
        self.max_retries = max_retries
        self.retry_backoff_base_seconds = retry_backoff_base_seconds
        self.dead_letter_dir = Path(dead_letter_dir)
        self.poll_timeout_seconds = poll_timeout_seconds
        self.name = name
        self._stopping = asyncio.Event()

    def backoff_seconds(self, retry_number: int) -> float:
        return self.retry_backoff_base_seconds * 2 ** (retry_number - 1)

    def stop(self) -> None:
        self._stopping.set()

    async def run_forever(self) -> None:
        logger.info("worker started", extra={"worker": self.name})
        while not self._stopping.is_set():
            job_id = await self._next_job()
            if job_id is None:
                continue
            try:
                await self._process_once(job_id)
            except Exception:
                logger.exception(
                    "unexpected error processing job",
                    extra={"worker": self.name, "job_id": job_id},
                )
        logger.info("worker stopped", extra={"worker": self.name})

    async def _next_job(self) -> str | None:
        dequeue = asyncio.ensure_future(self.queue.dequeue(timeout=self.poll_timeout_seconds))
        stopping = asyncio.ensure_future(self._stopping.wait())
        try:
            await asyncio.wait({dequeue, stopping}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stopping.cancel()
        if dequeue.done():
            return dequeue.result()
        dequeue.cancel()
        try:
            await dequeue
        except asyncio.CancelledError:
            pass
        return None

    async def _process_once(self, job_id: str) -> None:
        try:
            job = self.store.mark_processing(job_id)
        except (JobNotFoundError, InvalidTransitionError) as exc:
            logger.warning(
                "skipping job that can't be claimed",
                extra={"worker": self.name, "job_id": job_id, "reason": str(exc)},
            )
            return

        logger.info(
            "job processing",
            extra={"worker": self.name, "job_id": job_id, "attempt": job.retry_count + 1},
        )
        try:
            with self.storage.as_local_file(job.file_path) as audio_path:
                result = await self.pipeline.run(audio_path)
            version = self.store.save_transcript(job_id, result.to_dict())
            self.store.mark_completed(
                job_id, language=result.language, duration_seconds=result.duration
            )
        except Exception as exc:
            await self._handle_failure(job, exc)
            return

        logger.info(
            "job completed",
            extra={
                "worker": self.name,
                "job_id": job_id,
                "transcript_version": version,
                "duration_seconds": result.duration,
                "segments": len(result.segments),
            },
        )

    async def _handle_failure(self, job: Job, exc: Exception) -> None:
        code, message = describe_error(exc)
        attempt = job.retry_count + 1
        retryable = is_retryable(code)

        if retryable and job.retry_count < self.max_retries:
            retrying = self.store.mark_retrying(job.id, code, message)
            delay = self.backoff_seconds(retrying.retry_count)
            await self.queue.enqueue_after(job.id, delay)
            logger.warning(
                "job attempt failed; retry scheduled",
                extra={
                    "worker": self.name,
                    "job_id": job.id,
                    "attempt": attempt,
                    "error_code": code,
                    "retry_in_seconds": delay,
                },
            )
            return

        reason = "retries_exhausted" if retryable else "non_retryable"
        failed = self.store.mark_failed(job.id, code, message)
        self._write_dead_letter(failed, attempts=attempt, reason=reason)
        logger.error(
            "job failed permanently",
            extra={
                "worker": self.name,
                "job_id": job.id,
                "attempts": attempt,
                "error_code": code,
                "reason": reason,
            },
        )

    def _write_dead_letter(self, job: Job, *, attempts: int, reason: str) -> None:
        write_dead_letter(
            self.dead_letter_dir, job, attempts=attempts, reason=reason, source=self.name
        )


def write_dead_letter(
    dead_letter_dir: str | Path, job: Job, *, attempts: int, reason: str, source: str
) -> None:
    record = {
        "job_id": job.id,
        "user_id": job.user_id,
        "original_filename": job.original_filename,
        "file_path": job.file_path,
        "error_code": job.error_code,
        "error_message": job.error_message,
        "attempts": attempts,
        "reason": reason,
        "failed_at": job.failed_at,
        "source": source,
    }
    try:
        directory = Path(dead_letter_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{job.id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        logger.exception("could not write dead-letter record", extra={"job_id": job.id})


NON_RETRYABLE_ERROR_CODES = frozenset(
    {
        "invalid_audio",
        "normalization_failed",
        "file_not_found",
        "audio_not_found",
        "file_too_large",
        "invalid_storage_key",
        "ffmpeg_not_found",
        "engine_unavailable",
    }
)


def is_retryable(error_code: str) -> bool:
    return error_code not in NON_RETRYABLE_ERROR_CODES


def describe_error(exc: BaseException) -> tuple[str, str]:
    code = getattr(exc, "error_code", None)
    if isinstance(code, str) and code:
        return code, str(exc)
    return "internal_error", f"{type(exc).__name__}: {exc}"[:1000]
