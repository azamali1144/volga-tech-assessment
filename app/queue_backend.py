"""Job queue abstraction (Adapter pattern).

The API enqueues job ids; workers dequeue them. Only ids travel through the
queue: the job's state lives in the database, so a message is tiny and a
redelivered or duplicate message is harmless (the store's status
compare-and-set rejects a second claim).

``InMemoryQueue`` wraps ``asyncio.Queue`` for this single-process demo. A
production ``SQSQueue``/``RabbitMQQueue`` would implement the same methods;
delayed delivery maps to SQS ``DelaySeconds`` or a RabbitMQ delayed exchange.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable


class QueueClosedError(RuntimeError):
    pass


@runtime_checkable
class JobQueue(Protocol):
    async def enqueue(self, job_id: str) -> None:
        """Make ``job_id`` available to workers now."""
        ...

    async def enqueue_after(self, job_id: str, delay_seconds: float) -> None:
        """Make ``job_id`` available after ``delay_seconds``, without blocking
        the caller (used for retry backoff)."""
        ...

    async def dequeue(self, timeout: float | None = None) -> str | None:
        """Wait for the next job id; ``None`` if ``timeout`` elapses first.

        The timeout lets a worker loop wake up periodically (e.g. to notice
        shutdown) instead of blocking forever.
        """
        ...

    def size(self) -> int:
        """Messages ready now (excludes delayed ones). For metrics/health."""
        ...


class InMemoryQueue:
    """In-process queue: fast and zero-setup, but not durable.

    Anything still queued (or waiting out a retry delay) is lost if the
    process stops — the main reason production uses SQS/RabbitMQ instead.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        # Strong references to pending delayed-delivery tasks: asyncio only
        # keeps weak references, so an unreferenced task can be GC'd mid-sleep.
        self._delayed: set[asyncio.Task[None]] = set()
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise QueueClosedError("queue is closed")

    async def enqueue(self, job_id: str) -> None:
        self._check_open()
        await self._queue.put(job_id)

    async def enqueue_after(self, job_id: str, delay_seconds: float) -> None:
        self._check_open()
        if delay_seconds <= 0:
            await self.enqueue(job_id)
            return

        async def deliver_later() -> None:
            await asyncio.sleep(delay_seconds)
            if not self._closed:
                await self._queue.put(job_id)

        task = asyncio.create_task(deliver_later(), name=f"delayed-enqueue:{job_id}")
        self._delayed.add(task)
        task.add_done_callback(self._delayed.discard)

    async def dequeue(self, timeout: float | None = None) -> str | None:
        self._check_open()
        if timeout is None:
            return await self._queue.get()
        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    def size(self) -> int:
        return self._queue.qsize()

    @property
    def delayed_count(self) -> int:
        return len(self._delayed)

    async def close(self) -> None:
        """Stop accepting work and cancel pending delayed deliveries."""
        self._closed = True
        for task in list(self._delayed):
            task.cancel()
        if self._delayed:
            await asyncio.gather(*self._delayed, return_exceptions=True)
