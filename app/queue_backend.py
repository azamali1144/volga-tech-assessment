from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable


class QueueClosedError(RuntimeError):
    pass


@runtime_checkable
class JobQueue(Protocol):
    async def enqueue(self, job_id: str) -> None:
        ...

    async def enqueue_after(self, job_id: str, delay_seconds: float) -> None:
        ...

    async def dequeue(self, timeout: float | None = None) -> str | None:
        ...

    def size(self) -> int:
        ...


class InMemoryQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
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
        self._closed = True
        for task in list(self._delayed):
            task.cancel()
        if self._delayed:
            await asyncio.gather(*self._delayed, return_exceptions=True)
