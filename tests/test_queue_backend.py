import asyncio
import time
import unittest

from app.queue_backend import InMemoryQueue, JobQueue, QueueClosedError


class InMemoryQueueTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.queue = InMemoryQueue()

    async def asyncTearDown(self):
        await self.queue.close()

    async def test_satisfies_job_queue_protocol(self):
        self.assertIsInstance(self.queue, JobQueue)

    async def test_fifo_order(self):
        for job_id in ("a", "b", "c"):
            await self.queue.enqueue(job_id)
        self.assertEqual(self.queue.size(), 3)
        self.assertEqual([await self.queue.dequeue() for _ in range(3)], ["a", "b", "c"])
        self.assertEqual(self.queue.size(), 0)

    async def test_dequeue_timeout_returns_none(self):
        started = time.perf_counter()
        self.assertIsNone(await self.queue.dequeue(timeout=0.05))
        self.assertGreaterEqual(time.perf_counter() - started, 0.04)

    async def test_blocked_consumer_wakes_on_enqueue(self):
        consumer = asyncio.create_task(self.queue.dequeue())
        await asyncio.sleep(0.01)
        self.assertFalse(consumer.done())
        await self.queue.enqueue("job-1")
        self.assertEqual(await asyncio.wait_for(consumer, 1), "job-1")

    async def test_enqueue_after_delays_delivery_without_blocking_caller(self):
        started = time.perf_counter()
        await self.queue.enqueue_after("later", 0.1)
        self.assertLess(time.perf_counter() - started, 0.05)  # caller not blocked
        self.assertEqual((self.queue.size(), self.queue.delayed_count), (0, 1))

        await self.queue.enqueue("now")
        self.assertEqual(await self.queue.dequeue(timeout=1), "now")  # jumps ahead
        self.assertEqual(await self.queue.dequeue(timeout=1), "later")
        self.assertGreaterEqual(time.perf_counter() - started, 0.09)
        await asyncio.sleep(0.01)  # done-callbacks run on the next loop step
        self.assertEqual(self.queue.delayed_count, 0)  # finished tasks released

    async def test_zero_delay_enqueues_immediately(self):
        await self.queue.enqueue_after("x", 0)
        self.assertEqual(self.queue.size(), 1)

    async def test_close_cancels_pending_delayed_deliveries(self):
        await self.queue.enqueue_after("never", 0.05)
        await self.queue.close()
        self.assertEqual(self.queue.delayed_count, 0)
        await asyncio.sleep(0.1)
        self.assertEqual(self.queue.size(), 0)
        with self.assertRaises(QueueClosedError):
            await self.queue.enqueue("x")
        with self.assertRaises(QueueClosedError):
            await self.queue.dequeue(timeout=0.01)


if __name__ == "__main__":
    unittest.main()
