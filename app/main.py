from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status

from app.api.dependencies import Services
from app.api.errors import error_response, install_error_handlers
from app.api.routes import UPLOAD_PATH, health, router
from app.api.schemas import HealthResponse
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.infrastructure.database import JobStatus, JobStore
from app.infrastructure.queue import InMemoryQueue
from app.infrastructure.rate_limit import RateLimiter
from app.infrastructure.storage import LocalDiskStorage
from app.services.pipeline import TranscriptionPipeline
from app.services.transcription.engines import TranscriptionEngine, get_engine
from app.services.worker import Worker, write_dead_letter

logger = logging.getLogger(__name__)

MULTIPART_OVERHEAD_BYTES = 64 * 1024
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 10.0


async def recover_unfinished_jobs(services: Services) -> int:
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
    cfg = settings or get_settings()
    configure_logging(cfg.log_level, cfg.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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
            for task in pending:
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
        if request.method == "POST" and request.url.path == UPLOAD_PATH:
            declared = request.headers.get("content-length")
            limit = request.app.state.services.settings.max_upload_bytes
            if declared and declared.isdigit() and int(declared) > limit + MULTIPART_OVERHEAD_BYTES:
                return error_response(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    "file_too_large",
                    f"File exceeds the {limit}-byte upload limit.",
                )
        return await call_next(request)

    install_error_handlers(app)
    app.include_router(router)
    app.add_api_route(
        "/healthz",
        health,
        methods=["GET"],
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse, "description": "A check failed."}},
        summary="Health check (no auth)",
        tags=["ops"],
    )
    return app


app = create_app()
