import asyncio
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from app.audio import AudioProcessingError
from app.queue_backend import InMemoryQueue
from app.storage_backend import LocalDiskStorage, ObjectNotFoundError
from app.store import JobStatus, JobStore
from app.transcription_engine import EngineUnavailableError, MockEngine
from app.worker import (
    TranscriptionPipeline,
    Worker,
    describe_error,
    is_retryable,
)
from tests.conftest_helpers import FFMPEG_AVAILABLE, SKIP_REASON_NO_FFMPEG, make_tone_wav

SETTLE_TIMEOUT = 10.0


class RecordingEngine(MockEngine):
    """MockEngine that records calls and concurrency, and can be told to fail
    its first ``fail_times`` calls (or the calls numbered in ``fail_on``)."""

    def __init__(self, fail_times=0, exc=None, delay=0.0, fail_on=()):
        super().__init__(segment_seconds=2.0)
        self.fail_times = fail_times
        self.fail_on = set(fail_on)
        self.exc = exc or RuntimeError("transient engine failure")
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.peak_active = 0
        self._lock = threading.Lock()

    def transcribe(self, audio_path):
        with self._lock:
            self.calls += 1
            call = self.calls
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if call <= self.fail_times or call in self.fail_on:
                raise self.exc
            return super().transcribe(audio_path)
        finally:
            with self._lock:
                self.active -= 1


class _TempDirCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def tone(self, name="in.wav", seconds=3.0):
        return make_tone_wav(self.tmp / "src" / name, seconds=seconds)

    def pipeline(self, engine, **overrides):
        options = dict(
            chunk_threshold_seconds=30,
            chunk_length_seconds=10,
            chunk_overlap_seconds=2,
            max_concurrent_chunks=2,
            work_dir=self.tmp / "work",
        )
        options.update(overrides)
        return TranscriptionPipeline(engine, **options)

    def assert_work_dir_empty(self):
        work = self.tmp / "work"
        leftovers = list(work.iterdir()) if work.exists() else []
        self.assertEqual(leftovers, [], "pipeline left temp files behind")


@unittest.skipUnless(FFMPEG_AVAILABLE, SKIP_REASON_NO_FFMPEG)
class TranscriptionPipelineTest(_TempDirCase):
    async def test_short_audio_is_transcribed_in_a_single_pass(self):
        engine = RecordingEngine()
        result = await self.pipeline(engine).run(self.tone(seconds=8))
        self.assertEqual(engine.calls, 1)
        self.assertEqual(result.duration, 8.0)
        self.assertEqual(result.segments[-1].end, 8.0)
        self.assert_work_dir_empty()

    async def test_long_audio_is_chunked_and_merged_like_a_single_pass(self):
        engine = RecordingEngine()
        # 10s chunks with 2s overlap advance by 8s: chunk offsets land on the
        # mock's 2s segment grid, so the merge must reproduce a single pass.
        result = await self.pipeline(engine).run(self.tone(seconds=65))
        self.assertEqual(engine.calls, 8)  # windows start at 0, 8, ..., 56

        single = await self.pipeline(RecordingEngine(), chunk_threshold_seconds=1000).run(
            self.tone("again.wav", seconds=65)
        )
        self.assertEqual(
            [(s.start, s.end) for s in result.segments],
            [(s.start, s.end) for s in single.segments],
        )
        self.assertEqual(result.duration, 65.0)
        self.assert_work_dir_empty()

    async def test_chunk_concurrency_is_bounded(self):
        engine = RecordingEngine(delay=0.05)
        await self.pipeline(engine, max_concurrent_chunks=3).run(self.tone(seconds=65))
        self.assertEqual(engine.calls, 8)
        self.assertEqual(engine.peak_active, 3)

    async def test_failing_chunk_stops_new_chunks_and_cleans_up(self):
        engine = RecordingEngine(fail_on={2}, exc=RuntimeError("chunk 2 failed"))
        with self.assertRaisesRegex(RuntimeError, "chunk 2 failed"):
            await self.pipeline(engine, max_concurrent_chunks=1).run(self.tone(seconds=65))
        self.assertEqual(engine.calls, 2)  # chunks 3-8 never started
        self.assert_work_dir_empty()

    async def test_invalid_input_raises_audio_error_and_cleans_up(self):
        bogus = self.tmp / "bogus.mp3"
        bogus.write_bytes(b"not audio")
        with self.assertRaises(AudioProcessingError) as ctx:
            await self.pipeline(RecordingEngine()).run(bogus)
        self.assertEqual(ctx.exception.error_code, "normalization_failed")
        self.assert_work_dir_empty()

    async def test_event_loop_stays_responsive_during_transcription(self):
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.005)

        task = asyncio.create_task(ticker())
        try:
            await self.pipeline(RecordingEngine(delay=0.05)).run(self.tone(seconds=65))
        finally:
            task.cancel()
        self.assertGreater(ticks, 10)


@unittest.skipUnless(FFMPEG_AVAILABLE, SKIP_REASON_NO_FFMPEG)
class WorkerTest(_TempDirCase):
    async def asyncSetUp(self):
        self.store = JobStore(self.tmp / "jobs.db", transcript_dir=self.tmp / "transcripts")
        self.storage = LocalDiskStorage(self.tmp / "objects")
        self.queue = InMemoryQueue()
        self.dead_letter_dir = self.tmp / "dead_letter"
        self.workers = []
        self.tasks = []

    async def asyncTearDown(self):
        await self.stop_workers()
        await self.queue.close()
        self.store.close()

    def start_worker(self, engine, count=1, **options):
        settings = dict(
            max_retries=3,
            retry_backoff_base_seconds=0.01,
            dead_letter_dir=self.dead_letter_dir,
            poll_timeout_seconds=0.02,
        )
        settings.update(options)
        pipeline = self.pipeline(engine)
        for i in range(count):
            worker = Worker(self.store, self.queue, self.storage, pipeline, name=f"w{i}", **settings)
            self.workers.append(worker)
            self.tasks.append(asyncio.create_task(worker.run_forever()))
        return self.workers[-1]

    async def stop_workers(self):
        for worker in self.workers:
            worker.stop()
        if self.tasks:
            await asyncio.wait_for(asyncio.gather(*self.tasks), 5)
        self.workers, self.tasks = [], []

    def submit(self, seconds=3.0, data=None, name="call.wav"):
        job = self.store.create_job("alice", name, f"audio/{name}")
        payload = data if data is not None else self.tone(name, seconds).read_bytes()
        self.storage.save(job.file_path, io.BytesIO(payload))
        return job

    async def wait_until_settled(self, *job_ids):
        deadline = time.monotonic() + SETTLE_TIMEOUT
        while time.monotonic() < deadline:
            statuses = {self.store.get_job(j).status for j in job_ids}
            if statuses <= {JobStatus.COMPLETED, JobStatus.FAILED}:
                return
            await asyncio.sleep(0.01)
        self.fail(f"jobs did not settle: {[self.store.get_job(j).status for j in job_ids]}")

    def dead_letters(self):
        if not self.dead_letter_dir.exists():
            return []
        return [json.loads(p.read_text()) for p in sorted(self.dead_letter_dir.glob("*.json"))]

    async def test_successful_job_is_completed_with_transcript(self):
        engine = RecordingEngine()
        self.start_worker(engine)
        job = self.submit(seconds=5)
        await self.queue.enqueue(job.id)
        await self.wait_until_settled(job.id)

        done = self.store.get_job(job.id)
        self.assertEqual(done.status, JobStatus.COMPLETED)
        self.assertEqual((done.retry_count, done.trans_version), (0, 1))
        self.assertEqual((done.language, done.duration_seconds), ("en", 5.0))
        transcript = self.store.get_transcript(job.id)
        self.assertEqual(transcript["duration"], 5.0)
        self.assertEqual(len(transcript["segments"]), 3)
        self.assertEqual(self.dead_letters(), [])
        self.assert_work_dir_empty()

    async def test_transient_failures_then_success(self):
        engine = RecordingEngine(fail_times=2)
        self.start_worker(engine)
        job = self.submit()
        await self.queue.enqueue(job.id)
        await self.wait_until_settled(job.id)

        done = self.store.get_job(job.id)
        self.assertEqual(done.status, JobStatus.COMPLETED)
        self.assertEqual(done.retry_count, 2)
        self.assertEqual(engine.calls, 3)
        self.assertIsNone(done.error_code)  # stale error from attempt 2 cleared
        self.assertIsNotNone(self.store.get_transcript(job.id))
        self.assertEqual(self.dead_letters(), [])

    async def test_permanent_failure_goes_to_dead_letter(self):
        engine = RecordingEngine(fail_times=10**6)  # never succeeds
        self.start_worker(engine, max_retries=3)
        job = self.submit()
        await self.queue.enqueue(job.id)
        await self.wait_until_settled(job.id)

        failed = self.store.get_job(job.id)
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertEqual(failed.retry_count, 3)
        self.assertEqual(engine.calls, 4)  # 1 attempt + 3 retries
        self.assertEqual(failed.error_code, "internal_error")
        self.assertIn("transient engine failure", failed.error_message)
        self.assertIsNotNone(failed.failed_at)
        self.assertIsNone(self.store.get_transcript(job.id))

        (record,) = self.dead_letters()  # exactly one
        self.assertEqual(record["job_id"], job.id)
        self.assertEqual(record["attempts"], 4)
        self.assertEqual(record["reason"], "retries_exhausted")
        self.assertEqual(record["error_code"], "internal_error")
        self.assertEqual(record["failed_at"], failed.failed_at)

    async def test_non_retryable_failure_is_dead_lettered_without_retrying(self):
        engine = RecordingEngine()
        self.start_worker(engine)
        job = self.submit(data=b"this is not audio", name="corrupt.mp3")
        await self.queue.enqueue(job.id)
        await self.wait_until_settled(job.id)

        failed = self.store.get_job(job.id)
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertEqual(failed.error_code, "normalization_failed")
        self.assertEqual(failed.retry_count, 0)
        self.assertEqual(engine.calls, 0)
        (record,) = self.dead_letters()
        self.assertEqual((record["attempts"], record["reason"]), (1, "non_retryable"))

    async def test_engine_unavailable_is_not_retried(self):
        engine = RecordingEngine(fail_times=10**6, exc=EngineUnavailableError("no whisper"))
        self.start_worker(engine)
        job = self.submit()
        await self.queue.enqueue(job.id)
        await self.wait_until_settled(job.id)
        self.assertEqual(engine.calls, 1)
        self.assertEqual(self.store.get_job(job.id).error_code, "engine_unavailable")

    async def test_missing_audio_object_fails_permanently(self):
        self.start_worker(RecordingEngine())
        job = self.store.create_job("alice", "gone.wav", "audio/gone.wav")  # never uploaded
        await self.queue.enqueue(job.id)
        await self.wait_until_settled(job.id)
        failed = self.store.get_job(job.id)
        self.assertEqual((failed.error_code, failed.retry_count), ("audio_not_found", 0))

    async def test_retries_use_exponential_backoff_via_delayed_enqueue(self):
        delays = []
        original = self.queue.enqueue_after

        async def spy(job_id, delay_seconds):
            delays.append(delay_seconds)
            await original(job_id, delay_seconds)

        self.queue.enqueue_after = spy
        self.start_worker(RecordingEngine(fail_times=10**6), max_retries=3, retry_backoff_base_seconds=0.01)
        job = self.submit()
        await self.queue.enqueue(job.id)
        await self.wait_until_settled(job.id)
        self.assertEqual(delays, [0.01, 0.02, 0.04])

    async def test_long_audio_goes_through_chunked_path_end_to_end(self):
        engine = RecordingEngine()
        self.start_worker(engine)
        job = self.submit(seconds=65)
        await self.queue.enqueue(job.id)
        await self.wait_until_settled(job.id)

        self.assertEqual(self.store.get_job(job.id).status, JobStatus.COMPLETED)
        self.assertEqual(engine.calls, 8)
        segments = self.store.get_transcript(job.id)["segments"]
        self.assertEqual((segments[0]["start"], segments[-1]["end"]), (0.0, 65.0))
        for a, b in zip(segments, segments[1:]):
            self.assertGreaterEqual(b["start"], a["end"])

    async def test_duplicate_delivery_is_processed_once(self):
        engine = RecordingEngine(delay=0.05)
        self.start_worker(engine, count=3)
        job = self.submit()
        for _ in range(3):
            await self.queue.enqueue(job.id)
        await self.wait_until_settled(job.id)
        await asyncio.sleep(0.1)  # let the duplicate messages be consumed
        self.assertEqual(engine.calls, 1)
        self.assertEqual(self.store.get_job(job.id).trans_version, 1)

    async def test_unknown_job_id_is_skipped_and_worker_keeps_going(self):
        self.start_worker(RecordingEngine())
        await self.queue.enqueue("no-such-job")
        job = self.submit()
        await self.queue.enqueue(job.id)
        await self.wait_until_settled(job.id)
        self.assertEqual(self.store.get_job(job.id).status, JobStatus.COMPLETED)

    async def test_worker_survives_unexpected_infrastructure_errors(self):
        self.start_worker(RecordingEngine())
        original = self.store.mark_processing
        broken_once = []

        def flaky_mark_processing(job_id):
            if not broken_once:
                broken_once.append(job_id)
                raise OSError("database is locked")
            return original(job_id)

        self.store.mark_processing = flaky_mark_processing
        first, second = self.submit(name="a.wav"), self.submit(name="b.wav")
        with self.assertLogs("app.worker", level="ERROR") as logs:
            await self.queue.enqueue(first.id)
            await self.queue.enqueue(second.id)
            await self.wait_until_settled(second.id)
        self.assertIn("unexpected error processing job", logs.output[0])
        self.assertFalse(self.tasks[0].done())  # worker still running
        self.assertEqual(self.store.get_job(second.id).status, JobStatus.COMPLETED)
        self.assertEqual(self.store.get_job(first.id).status, JobStatus.QUEUED)

    async def test_stop_exits_promptly_when_idle(self):
        worker = self.start_worker(RecordingEngine())
        await asyncio.sleep(0.05)
        started = time.monotonic()
        worker.stop()
        await asyncio.wait_for(self.tasks[0], 1)
        self.assertLess(time.monotonic() - started, 0.5)

    async def test_multiple_workers_drain_many_jobs(self):
        engine = RecordingEngine(delay=0.02)
        self.start_worker(engine, count=3)
        jobs = [self.submit(name=f"j{i}.wav") for i in range(9)]
        for job in jobs:
            await self.queue.enqueue(job.id)
        await self.wait_until_settled(*(j.id for j in jobs))
        self.assertEqual(
            {self.store.get_job(j.id).status for j in jobs}, {JobStatus.COMPLETED}
        )
        self.assertEqual(engine.calls, 9)


class ErrorClassificationTest(unittest.TestCase):
    def test_domain_errors_keep_their_code(self):
        self.assertEqual(
            describe_error(AudioProcessingError("invalid_audio", "bad file")),
            ("invalid_audio", "bad file"),
        )
        self.assertEqual(describe_error(ObjectNotFoundError("gone"))[0], "audio_not_found")
        self.assertEqual(describe_error(EngineUnavailableError("x"))[0], "engine_unavailable")

    def test_unknown_errors_become_internal_error_with_type(self):
        code, message = describe_error(ZeroDivisionError("division by zero"))
        self.assertEqual(code, "internal_error")
        self.assertEqual(message, "ZeroDivisionError: division by zero")
        self.assertLessEqual(len(describe_error(ValueError("x" * 5000))[1]), 1000)

    def test_retryability(self):
        for code in ("invalid_audio", "normalization_failed", "audio_not_found", "engine_unavailable"):
            with self.subTest(code=code):
                self.assertFalse(is_retryable(code))
        for code in ("internal_error", "ffmpeg_timeout", "storage_error"):
            with self.subTest(code=code):
                self.assertTrue(is_retryable(code))

    def test_backoff_is_exponential(self):
        worker = Worker(None, None, None, None, retry_backoff_base_seconds=2.0)
        self.assertEqual([worker.backoff_seconds(n) for n in (1, 2, 3, 4)], [2, 4, 8, 16])

    def test_negative_max_retries_rejected(self):
        with self.assertRaises(ValueError):
            Worker(None, None, None, None, max_retries=-1)


if __name__ == "__main__":
    unittest.main()
