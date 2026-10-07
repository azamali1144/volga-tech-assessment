from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int


class RateLimiter:
    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = clock()

    def check(self, key: str) -> RateLimitDecision:
        now = self._clock()
        cutoff = now - self.window_seconds
        with self._lock:
            self._maybe_sweep(now, cutoff)
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) < self.max_requests:
                hits.append(now)
                return RateLimitDecision(
                    allowed=True,
                    limit=self.max_requests,
                    remaining=self.max_requests - len(hits),
                    retry_after_seconds=0,
                )

            wait = hits[0] + self.window_seconds - now
            return RateLimitDecision(
                allowed=False,
                limit=self.max_requests,
                remaining=0,
                retry_after_seconds=max(1, math.ceil(wait)),
            )

    def _maybe_sweep(self, now: float, cutoff: float) -> None:
        if now - self._last_sweep < self.window_seconds:
            return
        self._last_sweep = now
        stale = [key for key, hits in self._hits.items() if not hits or hits[-1] <= cutoff]
        for key in stale:
            del self._hits[key]

    @property
    def tracked_keys(self) -> int:
        with self._lock:
            return len(self._hits)
