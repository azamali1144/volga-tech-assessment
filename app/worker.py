from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import tempfile
from pathlib import Path

from app.audio import normalize_to_wav, split_into_chunks, wav_duration_seconds
from app.config import Settings
from app.queue_backend import JobQueue
from app.storage_backend import ObjectStorage
from app.store import InvalidTransitionError, Job, JobNotFoundError, JobStore
from app.transcription_engine import (
    TIMESTAMP_PRECISION,
    TranscriptionEngine,
    TranscriptionResult,
    merge_chunk_results,
)

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    def __init__(
        self,
        engine: TranscriptionEngine,
        *,
        chunk_threshold_seconds: float = 300.0,
        chunk_length_seconds: float = 240.0,
        chunk_overlap_seconds: float = 5.0,
        max_concurrent_chunks: int = 2,
        sample_rate: int = 16_000,
        work_dir: str | Path | None = None,
    ) -> None:
        if max_concurrent_chunks < 1:
            raise ValueError("max_concurrent_chunks must be >= 1")
        self.engine = engine
        self.chunk_threshold_seconds = chunk_threshold_seconds
        self.chunk_length_seconds = chunk_length_seconds
        self.chunk_overlap_seconds = chunk_overlap_seconds
        self.max_concurrent_chunks = max_concurrent_chunks
        self.sample_rate = sample_rate
        self.work_dir = Path(work_dir) if work_dir is not None else None

    @classmethod
    def from_settings(
        cls, engine: TranscriptionEngine, settings: Settings
    ) -> "TranscriptionPipeline":
        return cls(
            engine,
            chunk_threshold_seconds=settings.chunk_threshold_seconds,
            chunk_length_seconds=settings.chunk_length_seconds,
            chunk_overlap_seconds=settings.chunk_overlap_seconds,
            max_concurrent_chunks=settings.max_concurrent_chunk_transcriptions,
            sample_rate=settings.sample_rate,
            work_dir=settings.storage_dir / "tmp",
        )

    async def run(self, audio_path: str | Path) -> TranscriptionResult:
        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="transcribe-", dir=self.work_dir, ignore_cleanup_errors=True
        ) as tmp:
            tmp_dir = Path(tmp)
            wav = await asyncio.to_thread(
                normalize_to_wav, audio_path, tmp_dir / "normalized.wav", self.sample_rate
            )
            duration = wav_duration_seconds(wav)

            if duration <= self.chunk_threshold_seconds:
                result = await asyncio.to_thread(self.engine.transcribe, wav)
            else:
                result = await self._transcribe_chunked(wav, tmp_dir / "chunks")

        return dataclasses.replace(result, duration=round(duration, TIMESTAMP_PRECISION))

    async def _transcribe_chunked(self, wav: Path, chunk_dir: Path) -> TranscriptionResult:
        chunks = await asyncio.to_thread(
            split_into_chunks,
            wav,
            chunk_dir,
            self.chunk_length_seconds,
            self.chunk_overlap_seconds,
        )
        semaphore = asyncio.Semaphore(self.max_concurrent_chunks)
        failed = asyncio.Event()

        async def transcribe_chunk(chunk_path: Path) -> TranscriptionResult | None:
            async with semaphore:
                if failed.is_set():
                    return None
                try:
                    return await asyncio.to_thread(self.engine.transcribe, chunk_path)
                except BaseException:
                    failed.set()
                    raise

        outcomes = await asyncio.gather(
            *(transcribe_chunk(c.path) for c in chunks), return_exceptions=True
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        return merge_chunk_results(chunks, outcomes)


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
