"""Background processing: the transcription pipeline and the worker loop.

``TranscriptionPipeline`` is the pure core: audio file in, transcript out. It
knows nothing about queues, databases or HTTP, so it can be tested (and
reused, e.g. from a CLI or a batch job) on its own.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
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
        poll_timeout_seconds: float = 1.0,
        name: str = "worker",
    ) -> None:
        self.store = store
        self.queue = queue
        self.storage = storage
        self.pipeline = pipeline
        self.poll_timeout_seconds = poll_timeout_seconds
        self.name = name
        self._stopping = asyncio.Event()

    def stop(self) -> None:
        """Ask ``run_forever`` to exit after the current job (graceful)."""
        self._stopping.set()

    async def run_forever(self) -> None:
        logger.info("worker started", extra={"worker": self.name})
        while not self._stopping.is_set():
            job_id = await self.queue.dequeue(timeout=self.poll_timeout_seconds)
            if job_id is None:
                continue  # idle; loop round to re-check for shutdown
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
        code, message = describe_error(exc)
        self.store.mark_failed(job.id, code, message)
        logger.error(
            "job failed",
            extra={"worker": self.name, "job_id": job.id, "error_code": code},
        )


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
