"""Background processing: the transcription pipeline and the worker loop.

``TranscriptionPipeline`` is the pure core: audio file in, transcript out. It
knows nothing about queues, databases or HTTP, so it can be tested (and
reused, e.g. from a CLI or a batch job) on its own.
"""

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
    """normalize -> (single pass | chunk -> transcribe chunks -> merge).

    All blocking work (ffmpeg, WAV slicing, the engine) runs in worker
    threads via ``asyncio.to_thread``, so the event loop — which also serves
    the API in this single-process demo — stays responsive.
    """

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
        """Transcribe ``audio_path`` (any ffmpeg-readable format).

        Intermediate files (normalized WAV, chunks) live in a private temp
        directory that is removed when this returns — on success or failure.
        """
        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)
        # ignore_cleanup_errors: on Windows a file still held open elsewhere
        # can't be deleted; failing the job over temp cleanup would be wrong.
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

        # The WAV header is the authoritative duration; engines may omit it.
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
                # Once any chunk has failed the whole job will be retried, so
                # don't start more (possibly minutes-long) chunk transcriptions.
                if failed.is_set():
                    return None
                try:
                    return await asyncio.to_thread(self.engine.transcribe, chunk_path)
                except BaseException:
                    failed.set()
                    raise

        # return_exceptions=True: wait for every in-flight chunk to finish
        # before returning. A thread started by to_thread can't be cancelled,
        # so returning early would delete the temp dir under a running engine.
        outcomes = await asyncio.gather(
            *(transcribe_chunk(c.path) for c in chunks), return_exceptions=True
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        return merge_chunk_results(chunks, outcomes)


class Worker:
    """Consumer side of the queue: claim a job, run the pipeline, persist.

    Several ``Worker`` instances can share one queue and store (several
    asyncio tasks here, or separate processes with a durable queue): the
    store's compare-and-set on ``queued -> processing`` means a job id
    delivered twice is only ever processed once.
    """

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
        """Delay before retry ``retry_number`` (1-based): base * 2^(n-1)."""
        return self.retry_backoff_base_seconds * 2 ** (retry_number - 1)

    def stop(self) -> None:
        """Ask ``run_forever`` to exit after the current job (graceful)."""
        self._stopping.set()

    async def run_forever(self) -> None:
        logger.info("worker started", extra={"worker": self.name})
        while not self._stopping.is_set():
            job_id = await self._next_job()
            if job_id is None:
                continue  # idle or stopping; loop round to re-check
            try:
                await self._process_once(job_id)
            except Exception:
                # _process_once handles job failures itself; anything reaching
                # here is a bug or an infrastructure fault (e.g. the database
                # is down). Log it and keep the worker alive for other jobs.
                logger.exception(
                    "unexpected error processing job",
                    extra={"worker": self.name, "job_id": job_id},
                )
        logger.info("worker stopped", extra={"worker": self.name})

    async def _next_job(self) -> str | None:
        """Wait for a job id, but return ``None`` as soon as ``stop()`` is called.

        Without this, an idle worker would only notice shutdown when its
        dequeue timeout expired, delaying every graceful stop by up to
        ``poll_timeout_seconds``. Cancelling a pending ``dequeue`` loses
        nothing: an item is only removed from the queue once the get
        completes.
        """
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
            # Duplicate delivery, or a message for a job that already finished
            # or no longer exists: nothing to do.
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
        """Retry transient failures with exponential backoff; dead-letter the rest.

        ``job`` is the row as claimed for this attempt, so ``job.retry_count``
        is the number of retries already used. A job gets at most
        ``1 + max_retries`` attempts in total.
        """
        code, message = describe_error(exc)
        attempt = job.retry_count + 1
        retryable = is_retryable(code)

        if retryable and job.retry_count < self.max_retries:
            retrying = self.store.mark_retrying(job.id, code, message)
            delay = self.backoff_seconds(retrying.retry_count)
            # Delayed re-enqueue: the worker moves straight on to other jobs
            # instead of sleeping through the backoff.
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
    """Record a permanently failed job for manual review.

    The database row (status=failed) is the source of truth; this file is the
    review queue — in production, a real dead-letter queue (SQS DLQ). A
    failure to write it is logged but never raised: the job is already
    correctly marked failed.
    """
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


# Failures that another attempt can't fix: the input itself is bad or gone,
# or the deployment can't transcribe at all. Everything else (engine crashes,
# timeouts, I/O hiccups, unexpected exceptions) is treated as transient.
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
    """Map an exception to a stable ``(error_code, message)`` for the job row.

    Domain errors (``AudioProcessingError``) carry their own code; anything
    else is recorded as ``internal_error`` with its type, so raw exception
    text from third-party libraries never becomes the whole public message.
    """
    code = getattr(exc, "error_code", None)
    if isinstance(code, str) and code:
        return code, str(exc)
    return "internal_error", f"{type(exc).__name__}: {exc}"[:1000]
