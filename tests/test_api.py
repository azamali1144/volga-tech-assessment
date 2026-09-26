"""HTTP-layer tests: the real app (routes, auth, rate limiting, error
handlers, lifespan, background worker) driven through FastAPI's TestClient,
with the mock engine. Requires requirements-dev.txt; skipped otherwise."""

import tempfile
import time
import unittest
from pathlib import Path

from app.config import Settings
from app.store import JobStatus
from app.transcription_engine import MockEngine
from tests.conftest_helpers import (
    FFMPEG_AVAILABLE,
    SKIP_REASON_NO_FFMPEG,
    make_tone_file,
    make_tone_wav,
)

try:
    from fastapi.testclient import TestClient

    from app.main import caller_id_for, create_app

    CLIENT_AVAILABLE = True
except (ImportError, RuntimeError):
    # Starlette's TestClient raises RuntimeError (not ImportError) when no
    # HTTP client library is installed, i.e. requirements-dev.txt is missing.
    CLIENT_AVAILABLE = False

KEY = "test-key"
OTHER_KEY = "other-key"
AUTH = {"X-API-Key": KEY}
OTHER_AUTH = {"X-API-Key": OTHER_KEY}
UPLOAD_URL = "/api/v1/transcriptions"
POLL_TIMEOUT_SECONDS = 10.0


class FailingEngine(MockEngine):
    def transcribe(self, audio_path):
        raise RuntimeError("model exploded at /srv/secret/path.py:42")


@unittest.skipUnless(CLIENT_AVAILABLE, "pip install -r requirements-dev.txt")
class _ApiTestCase(unittest.TestCase):
    settings_overrides: dict = {}
    engine_factory = MockEngine
    raise_server_exceptions = True

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        options = dict(
            storage_dir=self.tmp / "storage",
            api_keys=frozenset({KEY, OTHER_KEY}),
            chunk_threshold_seconds=30,
            chunk_length_seconds=10,
            chunk_overlap_seconds=2,
            retry_backoff_base_seconds=0.01,
            log_level="CRITICAL",  # keep test output quiet
        )
        options.update(self.settings_overrides)
        self.settings = Settings(**options)
        app = create_app(settings=self.settings, engine=self.engine_factory())
        self.client = TestClient(app, raise_server_exceptions=self.raise_server_exceptions)
        self.client.__enter__()  # runs the lifespan: store, queue, workers

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self._tmp.cleanup()

    @property
    def services(self):
        return self.client.app.state.services

    def upload(self, path, headers=AUTH, filename=None):
        with open(path, "rb") as f:
            return self.client.post(
                UPLOAD_URL, headers=headers, files={"file": (filename or Path(path).name, f)}
            )

    def wait_for_terminal(self, status_url, headers=AUTH):
        deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            body = self.client.get(status_url, headers=headers).json()
            if body["status"] in ("completed", "failed"):
                return body
            time.sleep(0.02)
        self.fail(f"job at {status_url} did not finish")

    def assert_error(self, response, status_code, error_code):
        self.assertEqual(response.status_code, status_code, response.text)
        body = response.json()
        self.assertEqual(set(body), {"error_code", "detail"})
        self.assertEqual(body["error_code"], error_code)
        return body


@unittest.skipUnless(FFMPEG_AVAILABLE, SKIP_REASON_NO_FFMPEG)
class UploadAndPollTest(_ApiTestCase):
    def test_upload_poll_completed_round_trip(self):
        audio = make_tone_file(self.tmp / "call.mp3", seconds=5)
        response = self.upload(audio)

        self.assertEqual(response.status_code, 202, response.text)
        created = response.json()
        self.assertEqual(created["status"], "queued")
        self.assertEqual(created["status_url"], f"{UPLOAD_URL}/{created['job_id']}")
        self.assertEqual(response.headers["X-RateLimit-Limit"], str(self.settings.rate_limit_requests))
        self.assertIn("X-RateLimit-Remaining", response.headers)

        job = self.wait_for_terminal(created["status_url"])
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["original_filename"], "call.mp3")
        self.assertAlmostEqual(job["duration_seconds"], 5.0, delta=0.1)
        self.assertIsNone(job["error"])
        transcript = job["transcript"]
        self.assertEqual((transcript["version"], transcript["language"]), (1, "en"))
        segments = transcript["segments"]
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertAlmostEqual(segments[-1]["end"], 5.0, delta=0.1)
        self.assertEqual(set(segments[0]), {"id", "start", "end", "text"})

    def test_long_audio_is_chunked_and_merged(self):
        audio = make_tone_file(self.tmp / "podcast.flac", seconds=45)
        job = self.wait_for_terminal(self.upload(audio).json()["status_url"])
        segments = job["transcript"]["segments"]
        self.assertEqual((segments[0]["start"], segments[-1]["end"]), (0.0, 45.0))
        for a, b in zip(segments, segments[1:]):
            self.assertGreaterEqual(b["start"], a["end"])

    def test_every_supported_format_is_accepted(self):
        for ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg"):
            with self.subTest(ext=ext):
                audio = make_tone_file(self.tmp / f"tone{ext}", seconds=2)
                self.assertEqual(self.upload(audio).status_code, 202)
        video = make_tone_file(self.tmp / "clip.mp4", seconds=2, with_video=True)
        self.assertEqual(self.upload(video).status_code, 202)

    def test_extension_check_is_case_insensitive(self):
        audio = make_tone_wav(self.tmp / "loud.wav", seconds=1)
        self.assertEqual(self.upload(audio, filename="LOUD.WAV").status_code, 202)

    def test_raw_api_key_is_never_stored(self):
        created = self.upload(make_tone_wav(self.tmp / "a.wav", seconds=1)).json()
        job = self.services.store.get_job(created["job_id"])
        self.assertEqual(job.user_id, caller_id_for(KEY))
        self.assertNotIn(KEY, job.user_id)

    def test_client_filename_never_becomes_storage_key(self):
        audio = make_tone_wav(self.tmp / "a.wav", seconds=1)
        created = self.upload(audio, filename="../../evil.wav").json()
        job = self.services.store.get_job(created["job_id"])
        self.assertEqual(job.original_filename, "evil.wav")
        self.assertEqual(job.file_path, f"{created['job_id']}.wav")

    def test_corrupt_audio_rejected_at_upload(self):
        bogus = self.tmp / "voicemail.mp3"
        bogus.write_bytes(b"definitely not audio")
        body = self.assert_error(self.upload(bogus), 422, "invalid_audio")
        self.assertIn("voicemail.mp3", body["detail"])        # caller's name...
        self.assertNotIn(str(self.tmp), body["detail"])       # ...not server paths
        self.assertEqual(self.services.store.list_jobs(caller_id_for(KEY)), [])
        self.assertEqual(list(self.settings.audio_dir.iterdir()), [])  # nothing kept

    def test_other_callers_job_is_indistinguishable_from_missing(self):
        created = self.upload(make_tone_wav(self.tmp / "a.wav", seconds=1)).json()
        mine = self.client.get(created["status_url"], headers=OTHER_AUTH)
        missing = self.client.get(f"{UPLOAD_URL}/does-not-exist", headers=OTHER_AUTH)
        self.assert_error(mine, 404, "job_not_found")
        self.assertEqual(mine.json(), missing.json())


@unittest.skipUnless(FFMPEG_AVAILABLE, SKIP_REASON_NO_FFMPEG)
class ListingTest(_ApiTestCase):
    def test_listing_is_newest_first_paginated_and_scoped(self):
        wav = make_tone_wav(self.tmp / "a.wav", seconds=1)
        ids = [self.upload(wav, filename=f"f{i}.wav").json()["job_id"] for i in range(5)]
        self.upload(wav, headers=OTHER_AUTH)

        page1 = self.client.get(UPLOAD_URL, params={"limit": 2}, headers=AUTH).json()
        page2 = self.client.get(UPLOAD_URL, params={"limit": 2, "offset": 2}, headers=AUTH).json()
        page3 = self.client.get(UPLOAD_URL, params={"limit": 2, "offset": 4}, headers=AUTH).json()
        listed = [j["job_id"] for page in (page1, page2, page3) for j in page["items"]]
        self.assertEqual(listed, ids[::-1])
        self.assertEqual((page1["next_offset"], page2["next_offset"], page3["next_offset"]), (2, 4, None))
        self.assertNotIn("transcript", page1["items"][0])  # summaries only

        other = self.client.get(UPLOAD_URL, headers=OTHER_AUTH).json()
        self.assertEqual(len(other["items"]), 1)

    def test_status_filter(self):
        created = self.upload(make_tone_wav(self.tmp / "a.wav", seconds=1)).json()
        self.wait_for_terminal(created["status_url"])
        completed = self.client.get(UPLOAD_URL, params={"status": "completed"}, headers=AUTH).json()
        queued = self.client.get(UPLOAD_URL, params={"status": "queued"}, headers=AUTH).json()
        self.assertEqual([j["job_id"] for j in completed["items"]], [created["job_id"]])
        self.assertEqual(queued["items"], [])


@unittest.skipUnless(FFMPEG_AVAILABLE, SKIP_REASON_NO_FFMPEG)
class FailedJobTest(_ApiTestCase):
    engine_factory = FailingEngine
    settings_overrides = {"max_retries": 1}

    def test_failed_job_reports_code_but_not_internal_details(self):
        created = self.upload(make_tone_wav(self.tmp / "a.wav", seconds=1)).json()
        job = self.wait_for_terminal(created["status_url"])
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["retry_count"], 1)
        self.assertIsNotNone(job["failed_at"])
        self.assertIsNone(job["transcript"])
        self.assertEqual(job["error"]["code"], "internal_error")
        self.assertNotIn("/srv/secret", job["error"]["message"])  # generic message only
        dead_letters = list(self.settings.dead_letter_dir.glob("*.json"))
        self.assertEqual(len(dead_letters), 1)


class RequestValidationTest(_ApiTestCase):
    """Paths that are rejected before any audio processing (no ffmpeg needed)."""

    settings_overrides = {"max_upload_bytes": 10_000}

    def test_missing_or_wrong_api_key(self):
        for headers in ({}, {"X-API-Key": "wrong"}):
            with self.subTest(headers=headers):
                response = self.client.get(UPLOAD_URL, headers=headers)
                self.assert_error(response, 401, "unauthorized")
                self.assertEqual(response.headers["WWW-Authenticate"], "APIKey")

    def test_unsupported_extension(self):
        notes = self.tmp / "notes.txt"
        notes.write_text("hello")
        body = self.assert_error(self.upload(notes), 400, "unsupported_format")
        self.assertIn(".mp3", body["detail"])

    def test_oversized_upload_rejected_from_content_length(self):
        big = self.tmp / "big.wav"
        big.write_bytes(b"\0" * (10_000 + 100_000))  # beyond the multipart slack
        self.assert_error(self.upload(big), 413, "file_too_large")
        self.assertEqual(list(self.settings.audio_dir.iterdir()), [])

    def test_oversized_upload_within_slack_rejected_while_saving(self):
        # Passes the Content-Length pre-check (limit + 64 KiB framing slack)
        # but is caught by the streaming size cap while being stored.
        big = self.tmp / "big.wav"
        big.write_bytes(b"\0" * 30_000)
        self.assert_error(self.upload(big), 413, "file_too_large")
        self.assertEqual(list(self.settings.audio_dir.iterdir()), [])

    def test_missing_file_field(self):
        body = self.assert_error(
            self.client.post(UPLOAD_URL, headers=AUTH), 422, "validation_error"
        )
        self.assertIn("file", body["detail"])

    def test_invalid_query_parameters(self):
        for params in ({"limit": 0}, {"limit": 101}, {"offset": -1}, {"status": "bogus"}):
            with self.subTest(params=params):
                self.assert_error(
                    self.client.get(UPLOAD_URL, params=params, headers=AUTH), 422, "validation_error"
                )

    def test_unknown_job(self):
        self.assert_error(self.client.get(f"{UPLOAD_URL}/nope", headers=AUTH), 404, "job_not_found")

    def test_framework_errors_use_the_same_shape(self):
        self.assert_error(self.client.get("/api/v1/nope", headers=AUTH), 404, "not_found")
        self.assert_error(self.client.delete(UPLOAD_URL, headers=AUTH), 405, "method_not_allowed")

    def test_healthz_needs_no_auth(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["checks"], {"database": True, "workers": True})
        self.assertEqual(body["workers_running"], 1)

    def test_openapi_docs_are_served(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)
        paths = self.client.get("/openapi.json").json()["paths"]
        self.assertIn(UPLOAD_URL, paths)
        self.assertIn(f"{UPLOAD_URL}/{{job_id}}", paths)


class RateLimitTest(_ApiTestCase):
    settings_overrides = {"rate_limit_requests": 3}

    def test_limit_per_key_with_retry_after(self):
        remaining = [
            self.client.get(UPLOAD_URL, headers=AUTH).headers["X-RateLimit-Remaining"]
            for _ in range(3)
        ]
        self.assertEqual(remaining, ["2", "1", "0"])
        limited = self.client.get(UPLOAD_URL, headers=AUTH)
        self.assert_error(limited, 429, "rate_limited")
        self.assertGreaterEqual(int(limited.headers["Retry-After"]), 1)
        # Another key has its own budget.
        self.assertEqual(self.client.get(UPLOAD_URL, headers=OTHER_AUTH).status_code, 200)

    def test_unauthenticated_requests_do_not_consume_a_keys_budget(self):
        for _ in range(5):
            self.client.get(UPLOAD_URL, headers={"X-API-Key": "wrong"})
        self.assertEqual(self.client.get(UPLOAD_URL, headers=AUTH).status_code, 200)


class UnexpectedErrorTest(_ApiTestCase):
    raise_server_exceptions = False

    def test_unhandled_exception_becomes_generic_500(self):
        def broken(job_id):
            raise RuntimeError("secret internal detail")

        self.services.store.get_job = broken
        body = self.assert_error(
            self.client.get(f"{UPLOAD_URL}/anything", headers=AUTH), 500, "internal_error"
        )
        self.assertNotIn("secret", body["detail"])


class StartupRecoveryTest(unittest.TestCase):
    """Jobs left unfinished by a previous process are picked up on startup."""

    @unittest.skipUnless(CLIENT_AVAILABLE and FFMPEG_AVAILABLE, "needs httpx2 and ffmpeg")
    def test_interrupted_and_queued_jobs_are_finished_after_restart(self):
        from app.store import JobStore
        from app.storage_backend import LocalDiskStorage

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            settings = Settings(
                storage_dir=tmp / "storage", api_keys=frozenset({KEY}), log_level="CRITICAL"
            )
            settings.ensure_dirs()
            # Simulate a previous process that died: one job mid-processing,
            # one still queued, both with audio already stored.
            store = JobStore(settings.db_path, transcript_dir=settings.transcript_dir)
            storage = LocalDiskStorage(settings.audio_dir)
            owner = caller_id_for(KEY)
            job_ids = []
            for name in ("interrupted.wav", "queued.wav"):
                job = store.create_job(owner, name, f"x-{name}")
                with open(make_tone_wav(tmp / name, seconds=1), "rb") as f:
                    storage.save(job.file_path, f)
                job_ids.append(job.id)
            store.mark_processing(job_ids[0])
            store.close()

            with TestClient(create_app(settings=settings, engine=MockEngine())) as client:
                deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
                while time.monotonic() < deadline:
                    jobs = [client.get(f"{UPLOAD_URL}/{j}", headers=AUTH).json() for j in job_ids]
                    if all(j["status"] == "completed" for j in jobs):
                        break
                    time.sleep(0.02)
            self.assertEqual([j["status"] for j in jobs], ["completed", "completed"])
            self.assertEqual(jobs[0]["retry_count"], 1)  # the interruption counted as an attempt
            self.assertEqual(jobs[1]["retry_count"], 0)


if __name__ == "__main__":
    unittest.main()
