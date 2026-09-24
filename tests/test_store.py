import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from app.store import (
    InvalidTransitionError,
    JobNotFoundError,
    JobStatus,
    JobStore,
    TranscriptStorageError,
)


class _StoreTestCase(unittest.TestCase):
    inline_max_chars = 200

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.store = self.open_store()

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def open_store(self):
        return JobStore(
            self.tmp / "jobs.db",
            transcript_dir=self.tmp / "transcripts",
            inline_max_chars=self.inline_max_chars,
        )

    def new_job(self, user_id="alice", name="a.mp3"):
        return self.store.create_job(user_id, name, f"audio/{name}")


class CreateAndGetJobTest(_StoreTestCase):
    def test_new_job_is_queued_with_clean_counters(self):
        job = self.store.create_job("alice", "call.mp3", "audio/x.mp3", duration_seconds=12.5)
        self.assertEqual(job.status, JobStatus.QUEUED)
        self.assertEqual((job.retry_count, job.trans_version), (0, 0))
        self.assertIsNone(job.error_code)
        self.assertEqual(job.duration_seconds, 12.5)
        self.assertEqual(job.created_at, job.updated_at)
        self.assertEqual(self.store.get_job(job.id), job)

    def test_caller_supplied_id_is_used_and_must_be_unique(self):
        job = self.store.create_job("alice", "a.mp3", "k", job_id="fixed-id")
        self.assertEqual(job.id, "fixed-id")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.create_job("alice", "b.mp3", "k", job_id="fixed-id")

    def test_unknown_job_returns_none(self):
        self.assertIsNone(self.store.get_job("does-not-exist"))

    def test_data_survives_reopening_the_database(self):
        job = self.new_job()
        self.store.close()
        self.store = self.open_store()
        self.assertEqual(self.store.get_job(job.id), job)

    def test_concurrent_inserts_from_many_threads(self):
        def insert_many():
            for _ in range(25):
                self.new_job(user_id="bulk")

        threads = [threading.Thread(target=insert_many) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(self.store.list_jobs("bulk", limit=100)), 100)
        self.assertEqual(
            len(self.store.list_jobs("bulk", limit=100, offset=100)), 100
        )


class StatusLifecycleTest(_StoreTestCase):
    def test_retry_then_success_path(self):
        job = self.new_job()
        self.store.mark_processing(job.id)
        self.store.mark_retrying(job.id, "transient", "first failure")
        self.store.mark_processing(job.id)
        retried = self.store.mark_retrying(job.id, "transient", "second failure")
        self.assertEqual((retried.status, retried.retry_count), (JobStatus.RETRYING, 2))
        self.assertEqual(retried.error_message, "second failure")

        self.store.mark_processing(job.id)
        done = self.store.mark_completed(job.id, language="en", duration_seconds=42.0)
        self.assertEqual(done.status, JobStatus.COMPLETED)
        self.assertEqual(done.retry_count, 2)             # history kept
        self.assertIsNone(done.error_code)                 # stale error cleared
        self.assertIsNone(done.error_message)
        self.assertIsNone(done.failed_at)
        self.assertEqual((done.language, done.duration_seconds), ("en", 42.0))

    def test_completed_keeps_existing_values_when_none_given(self):
        job = self.store.create_job("alice", "a.mp3", "k", duration_seconds=9.0)
        self.store.mark_processing(job.id)
        done = self.store.mark_completed(job.id)
        self.assertEqual(done.duration_seconds, 9.0)

    def test_failure_records_error_and_timestamp(self):
        job = self.new_job()
        self.store.mark_processing(job.id)
        failed = self.store.mark_failed(job.id, "invalid_audio", "bad file")
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertEqual((failed.error_code, failed.error_message), ("invalid_audio", "bad file"))
        self.assertIsNotNone(failed.failed_at)

    def test_updated_at_advances_on_transition(self):
        job = self.new_job()
        processing = self.store.mark_processing(job.id)
        self.assertGreaterEqual(processing.updated_at, job.updated_at)
        self.assertEqual(processing.created_at, job.created_at)

    def test_terminal_states_reject_every_transition(self):
        done = self.new_job()
        self.store.mark_processing(done.id)
        self.store.mark_completed(done.id)
        failed = self.new_job()
        self.store.mark_failed(failed.id, "x", "y")

        for job_id in (done.id, failed.id):
            for transition in (
                lambda: self.store.mark_processing(job_id),
                lambda: self.store.mark_retrying(job_id, "x", "y"),
                lambda: self.store.mark_completed(job_id),
                lambda: self.store.mark_failed(job_id, "x", "y"),
            ):
                with self.subTest(job_id=job_id), self.assertRaises(InvalidTransitionError):
                    transition()

    def test_illegal_shortcuts_rejected(self):
        job = self.new_job()
        with self.assertRaises(InvalidTransitionError) as ctx:
            self.store.mark_completed(job.id)  # queued -> completed
        self.assertEqual(ctx.exception.current, JobStatus.QUEUED)
        with self.assertRaises(InvalidTransitionError):
            self.store.mark_retrying(job.id, "x", "y")  # queued -> retrying
        self.assertEqual(self.store.get_job(job.id).status, JobStatus.QUEUED)

    def test_rejected_transition_changes_nothing(self):
        job = self.new_job()
        self.store.mark_processing(job.id)
        before = self.store.mark_completed(job.id)
        with self.assertRaises(InvalidTransitionError):
            self.store.mark_retrying(job.id, "x", "y")
        self.assertEqual(self.store.get_job(job.id), before)

    def test_unknown_job_raises_not_found(self):
        with self.assertRaises(JobNotFoundError):
            self.store.mark_processing("ghost")

    def test_only_one_of_many_racing_workers_claims_a_job(self):
        job = self.new_job()
        wins = []

        def claim():
            try:
                self.store.mark_processing(job.id)
                wins.append(1)
            except InvalidTransitionError:
                pass

        threads = [threading.Thread(target=claim) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(wins), 1)


class TranscriptPersistenceTest(_StoreTestCase):
    small = {"text": "hi", "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hi"}]}
    large = {"text": "word " * 100, "segments": []}

    def transcript_rows(self, job_id):
        return self.store._conn.execute(
            "SELECT version, content, content_path, char_count FROM transcripts "
            "WHERE job_id = ? ORDER BY version",
            (job_id,),
        ).fetchall()

    def test_no_transcript_yet_returns_none(self):
        self.assertIsNone(self.store.get_transcript(self.new_job().id))

    def test_small_transcript_stored_inline(self):
        job = self.new_job()
        self.assertEqual(self.store.save_transcript(job.id, self.small), 1)
        (row,) = self.transcript_rows(job.id)
        self.assertIsNotNone(row["content"])
        self.assertIsNone(row["content_path"])
        self.assertLessEqual(row["char_count"], self.inline_max_chars)
        self.assertEqual(self.store.get_transcript(job.id), self.small)
        self.assertFalse((self.tmp / "transcripts").exists())

    def test_large_transcript_stored_as_file(self):
        job = self.new_job()
        self.store.save_transcript(job.id, self.large)
        (row,) = self.transcript_rows(job.id)
        self.assertIsNone(row["content"])
        self.assertGreater(row["char_count"], self.inline_max_chars)
        path = Path(row["content_path"])
        self.assertEqual(path, self.tmp / "transcripts" / f"{job.id}.v1.json")
        self.assertTrue(path.is_file())
        self.assertEqual(self.store.get_transcript(job.id), self.large)

    def test_non_ascii_text_round_trips(self):
        job = self.new_job()
        content = {"text": "Grüße, 你好, مرحبا", "segments": []}
        self.store.save_transcript(job.id, content)
        self.assertEqual(self.store.get_transcript(job.id), content)

    def test_versions_increment_and_old_versions_are_kept(self):
        job = self.new_job()
        self.assertEqual(self.store.save_transcript(job.id, self.small), 1)
        self.assertEqual(self.store.save_transcript(job.id, self.large), 2)
        self.assertEqual(self.store.get_job(job.id).trans_version, 2)
        self.assertEqual(self.store.get_transcript(job.id), self.large)      # latest
        self.assertEqual(self.store.get_transcript(job.id, version=1), self.small)
        self.assertIsNone(self.store.get_transcript(job.id, version=3))

    def test_saving_for_unknown_job_raises_not_found(self):
        with self.assertRaises(JobNotFoundError):
            self.store.save_transcript("ghost", self.small)

    def test_missing_transcript_file_raises_storage_error(self):
        job = self.new_job()
        self.store.save_transcript(job.id, self.large)
        Path(self.transcript_rows(job.id)[0]["content_path"]).unlink()
        with self.assertRaises(TranscriptStorageError):
            self.store.get_transcript(job.id)

    def test_failed_save_leaves_no_orphan_file_and_version_unchanged(self):
        job = self.new_job()
        self.store.save_transcript(job.id, self.small)
        # Occupy version 2 behind the store's back so the next insert clashes.
        with self.store._conn:
            self.store._conn.execute(
                "INSERT INTO transcripts (job_id, version, content, char_count, created_at) "
                "VALUES (?, 2, '{}', 2, 'now')",
                (job.id,),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save_transcript(job.id, self.large)
        self.assertFalse((self.tmp / "transcripts" / f"{job.id}.v2.json").exists())
        self.assertEqual(self.store.get_job(job.id).trans_version, 1)

    def test_schema_rejects_inconsistent_rows(self):
        job = self.new_job()
        with self.assertRaises(sqlite3.IntegrityError):  # both inline and file
            with self.store._conn:
                self.store._conn.execute(
                    "INSERT INTO transcripts (job_id, version, content, content_path, "
                    "char_count, created_at) VALUES (?, 1, '{}', 'x.json', 2, 'now')",
                    (job.id,),
                )
        with self.assertRaises(sqlite3.IntegrityError):  # no such job
            with self.store._conn:
                self.store._conn.execute(
                    "INSERT INTO transcripts (job_id, version, content, char_count, "
                    "created_at) VALUES ('ghost', 1, '{}', 2, 'now')"
                )


class ListJobsTest(_StoreTestCase):
    def test_newest_first_and_scoped_to_user(self):
        alice = [self.new_job("alice", f"a{i}.mp3") for i in range(5)]
        self.new_job("bob", "b.mp3")
        listed = self.store.list_jobs("alice")
        self.assertEqual([j.id for j in listed], [j.id for j in reversed(alice)])
        self.assertEqual([j.original_filename for j in self.store.list_jobs("bob")], ["b.mp3"])
        self.assertEqual(self.store.list_jobs("nobody"), [])

    def test_pagination(self):
        jobs = [self.new_job("alice", f"a{i}.mp3") for i in range(5)]
        newest_first = [j.id for j in reversed(jobs)]
        page1 = [j.id for j in self.store.list_jobs("alice", limit=2, offset=0)]
        page2 = [j.id for j in self.store.list_jobs("alice", limit=2, offset=2)]
        page3 = [j.id for j in self.store.list_jobs("alice", limit=2, offset=4)]
        self.assertEqual(page1 + page2 + page3, newest_first)

    def test_status_filter(self):
        jobs = [self.new_job("alice", f"a{i}.mp3") for i in range(3)]
        self.store.mark_processing(jobs[1].id)
        listed = self.store.list_jobs("alice", status=JobStatus.PROCESSING)
        self.assertEqual([j.id for j in listed], [jobs[1].id])

    def test_limit_is_clamped(self):
        for i in range(3):
            self.new_job("alice", f"a{i}.mp3")
        self.assertEqual(len(self.store.list_jobs("alice", limit=0)), 1)
        self.assertEqual(len(self.store.list_jobs("alice", limit=10**6)), 3)
        self.assertEqual(len(self.store.list_jobs("alice", offset=-5)), 3)


if __name__ == "__main__":
    unittest.main()
