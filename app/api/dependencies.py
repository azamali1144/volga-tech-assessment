from __future__ import annotations

import asyncio
import hashlib
import hmac
from dataclasses import dataclass, field

from fastapi import Depends, Request, Response, Security, status
from fastapi.security import APIKeyHeader

from app.api.errors import ApiError
from app.core.config import Settings
from app.infrastructure.database import JobStore
from app.infrastructure.queue import InMemoryQueue
from app.infrastructure.rate_limit import RateLimiter
from app.infrastructure.storage import LocalDiskStorage
from app.services.worker import Worker


@dataclass
class Services:
    settings: Settings
    store: JobStore
    storage: LocalDiskStorage
    queue: InMemoryQueue
    limiter: RateLimiter
    workers: list[Worker] = field(default_factory=list)
    worker_tasks: list[asyncio.Task[None]] = field(default_factory=list)


def get_services(request: Request) -> Services:
    return request.app.state.services


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API key issued to the calling service.",
)


def caller_id_for(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def require_api_key(
    response: Response,
    api_key: str | None = Security(api_key_header),
    services: Services = Depends(get_services),
) -> str:
    if not api_key or not any(
        hmac.compare_digest(api_key.encode("utf-8"), valid.encode("utf-8"))
        for valid in services.settings.api_keys
    ):
        raise ApiError(
            status.HTTP_401_UNAUTHORIZED,
            "unauthorized",
            "Missing or invalid API key.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    caller_id = caller_id_for(api_key)
    decision = services.limiter.check(caller_id)
    rate_headers = {
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
    }
    if not decision.allowed:
        raise ApiError(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate_limited",
            f"Rate limit exceeded. Retry in {decision.retry_after_seconds}s.",
            headers={**rate_headers, "Retry-After": str(decision.retry_after_seconds)},
        )
    response.headers.update(rate_headers)
    return caller_id
