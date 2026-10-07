from __future__ import annotations

import asyncio
import dataclasses
import tempfile
from pathlib import Path

from app.core.config import Settings
from app.services.audio import normalize_to_wav, split_into_chunks, wav_duration_seconds
from app.services.transcription.engines import TranscriptionEngine
from app.services.transcription.merge import merge_chunk_results
from app.services.transcription.models import TIMESTAMP_PRECISION, TranscriptionResult


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
