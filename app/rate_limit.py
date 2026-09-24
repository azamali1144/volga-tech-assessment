"""Per-API-key sliding-window rate limiting.

Each key keeps the timestamps of its recent *allowed* requests. A request is
allowed if fewer than ``max_requests`` of them fall within the last
``window_seconds``. Unlike a fixed window (reset every minute on the minute),
a sliding window can't be gamed by bursting at a window boundary to get 2x
the limit.

This is in-process state, so it limits per API instance: behind a load
balancer with N instances a key could get up to N x the limit. Production
would keep the same algorithm in Redis (a sorted set per key, trimmed with
ZREMRANGEBYSCORE) so all instances share one count.
"""

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
    retry_after_seconds: int  # 0 when allowed; whole seconds for Retry-After


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
        self._clock = clock  # injectable, so tests don't have to sleep
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = clock()

    def check(self, key: str) -> RateLimitDecision:
        """Record a request for ``key`` if allowed, and report the decision.

        Rejected requests are not recorded, so a client hammering the API
        while limited doesn't push its own recovery further away.
        """
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

            # Full: the oldest hit in the window is the next to expire.
            wait = hits[0] + self.window_seconds - now
            return RateLimitDecision(
                allowed=False,
                limit=self.max_requests,
                remaining=0,
                retry_after_seconds=max(1, math.ceil(wait)),
            )

    def _maybe_sweep(self, now: float, cutoff: float) -> None:
        """Drop keys with no hits inside the window, at most once per window.

        Keeps memory bounded by the number of *recently active* keys rather
        than every key ever seen. Amortized: one pass per window.
        """
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
