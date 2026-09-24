"""Job and transcript persistence (Repository pattern).

``JobStore`` is the only code that talks to the database. Nothing else in the
app imports ``sqlite3``, so moving to PostgreSQL means changing this file:
the schema below uses only portable types (TEXT ids and ISO-8601 UTC
timestamps map to UUID/TIMESTAMPTZ; INTEGER/REAL map directly).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"


# Lifecycle:  queued -> processing -> completed
#                          |   ^
#                          v   |
#                        retrying            (any non-terminal) -> failed
#
# Each transition lists the states it may start from. Enforcing this in the
# UPDATE's WHERE clause makes every transition an atomic compare-and-set: if
# two workers ever raced on one job, only one transition would win.
ALLOWED_FROM: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.PROCESSING: frozenset({JobStatus.QUEUED, JobStatus.RETRYING}),
    JobStatus.RETRYING: frozenset({JobStatus.PROCESSING}),
    JobStatus.COMPLETED: frozenset({JobStatus.PROCESSING}),
    JobStatus.FAILED: frozenset(
        {JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.RETRYING}
    ),
}
TERMINAL_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED})
MAX_LIST_LIMIT = 100


class JobNotFoundError(LookupError):
    pass


class TranscriptStorageError(RuntimeError):
    """A transcript row points at a file that can't be read."""


class InvalidTransitionError(RuntimeError):
    def __init__(self, job_id: str, current: JobStatus, target: JobStatus) -> None:
        super().__init__(
            f"Job {job_id}: cannot move from {current.value!r} to {target.value!r}"
        )
        self.job_id = job_id
        self.current = current
        self.target = target


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    file_path         TEXT NOT NULL,          -- object-storage key of the upload
    duration_seconds  REAL,
    status            TEXT NOT NULL,
    retry_count       INTEGER NOT NULL DEFAULT 0,
    error_code        TEXT,
    error_message     TEXT,
    failed_at         TEXT,
    language          TEXT,
    trans_version     INTEGER NOT NULL DEFAULT 0,  -- latest transcript version
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON jobs (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);

CREATE TABLE IF NOT EXISTS transcripts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    version      INTEGER NOT NULL,
    content      TEXT,             -- JSON, when stored inline
    content_path TEXT,             -- file path, when too large to inline
    char_count   INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    UNIQUE (job_id, version),
    CHECK ((content IS NULL) != (content_path IS NULL))
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class Job:
    id: str
    user_id: str
    original_filename: str
    file_path: str
    duration_seconds: float | None
    status: JobStatus
    retry_count: int
    error_code: str | None
    error_message: str | None
    failed_at: str | None
    language: str | None
    trans_version: int
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        data: dict[str, Any] = dict(row)
        data["status"] = JobStatus(data["status"])
        return cls(**data)


class JobStore:
    """Thread-safe SQLite-backed store for jobs and transcripts.

    One connection is shared by the API handlers and the worker, so it's
    opened with ``check_same_thread=False`` and every access goes through a
    lock. Queries are tiny, so the lock is never held for long; a Postgres
    version would use a connection pool instead.

    Transcripts whose JSON is at most ``inline_max_chars`` are stored in the
    ``transcripts`` row; larger ones are written to ``transcript_dir`` and only
    the path is stored, keeping the table small and fast for the common case.
    """

    def __init__(
        self,
        db_path: str | Path,
        transcript_dir: str | Path = "storage/transcripts",
        inline_max_chars: int = 20_000,
    ) -> None:
        self.db_path = str(db_path)
        self.transcript_dir = Path(transcript_dir)
        self.inline_max_chars = inline_max_chars
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if self.db_path != ":memory:":
                # WAL lets readers proceed while a write is in progress.
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "JobStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- jobs -------------------------------------------------------------

    def create_job(
        self,
        user_id: str,
        original_filename: str,
        file_path: str,
        duration_seconds: float | None = None,
        job_id: str | None = None,
    ) -> Job:
        """Insert a new job in ``queued`` state.

        ``job_id`` can be supplied so the caller can derive the storage key
        from it before the row exists; otherwise a UUID4 is generated.
        """
        job_id = job_id or uuid.uuid4().hex
        now = utc_now()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO jobs (id, user_id, original_filename, file_path,
                                  duration_seconds, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    user_id,
                    original_filename,
                    file_path,
                    duration_seconds,
                    JobStatus.QUEUED.value,
                    now,
                    now,
                ),
            )
        job = self.get_job(job_id)
        assert job is not None
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return Job.from_row(row) if row else None

    def list_jobs(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        status: JobStatus | None = None,
    ) -> list[Job]:
        """A caller's jobs, newest first. Always scoped to one ``user_id`` so
        one API key can never see another's jobs. Served by the
        ``(user_id, created_at DESC)`` index."""
        limit = max(1, min(limit, MAX_LIST_LIMIT))
        offset = max(0, offset)
        sql = "SELECT * FROM jobs WHERE user_id = ?"
        params: list[Any] = [user_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(JobStatus(status).value)
        # rowid (insertion order) breaks ties between jobs created in the same
        # millisecond, so "newest first" and pagination stay exact. On
        # Postgres this would be a BIGSERIAL column.
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [Job.from_row(row) for row in rows]

    # --- status transitions -----------------------------------------------

    def _transition(
        self, job_id: str, target: JobStatus, set_sql: str = "", params: tuple = ()
    ) -> Job:
        allowed = sorted(status.value for status in ALLOWED_FROM[target])
        placeholders = ", ".join("?" for _ in allowed)
        extra = f", {set_sql}" if set_sql else ""
        with self._lock, self._conn:
            cursor = self._conn.execute(
                f"""
                UPDATE jobs SET status = ?, updated_at = ?{extra}
                WHERE id = ? AND status IN ({placeholders})
                """,
                (target.value, utc_now(), *params, job_id, *allowed),
            )
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobNotFoundError(job_id)
        job = Job.from_row(row)
        if cursor.rowcount == 0:
            raise InvalidTransitionError(job_id, job.status, target)
        return job

    def mark_processing(self, job_id: str) -> Job:
        """A worker has picked the job up (first attempt or a retry)."""
        return self._transition(job_id, JobStatus.PROCESSING)

    def mark_retrying(self, job_id: str, error_code: str, error_message: str) -> Job:
        """An attempt failed transiently; record why and count the retry."""
        return self._transition(
            job_id,
            JobStatus.RETRYING,
            "retry_count = retry_count + 1, error_code = ?, error_message = ?",
            (error_code, error_message),
        )

    def mark_completed(
        self,
        job_id: str,
        language: str | None = None,
        duration_seconds: float | None = None,
    ) -> Job:
        """Transcription succeeded.

        Error fields from earlier failed attempts are cleared so a completed
        job doesn't present a stale error to callers; ``retry_count`` is kept
        as the record of how many attempts it took.
        """
        return self._transition(
            job_id,
            JobStatus.COMPLETED,
            "language = COALESCE(?, language), "
            "duration_seconds = COALESCE(?, duration_seconds), "
            "error_code = NULL, error_message = NULL",
            (language, duration_seconds),
        )

    def mark_failed(self, job_id: str, error_code: str, error_message: str) -> Job:
        """Permanent failure: retries exhausted or the error isn't retryable."""
        return self._transition(
            job_id,
            JobStatus.FAILED,
            "error_code = ?, error_message = ?, failed_at = ?",
            (error_code, error_message, utc_now()),
        )

    # --- transcripts ------------------------------------------------------

    def save_transcript(self, job_id: str, content: dict[str, Any]) -> int:
        """Store a new transcript version for ``job_id``; return its version.

        Versions start at 1 and increment on every save (e.g. reprocessing),
        and ``jobs.trans_version`` always points at the latest. Old versions
        are kept for auditability. Content is any JSON-serializable dict, so
        the store stays independent of the engine's result types.
        """
        payload = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        char_count = len(payload)
        inline = char_count <= self.inline_max_chars

        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT trans_version FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(job_id)
            version = row["trans_version"] + 1

            content_path = None
            if not inline:
                content_path = self._write_transcript_file(job_id, version, payload)
            try:
                self._conn.execute(
                    """
                    INSERT INTO transcripts (job_id, version, content, content_path,
                                             char_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        version,
                        payload if inline else None,
                        str(content_path) if content_path else None,
                        char_count,
                        utc_now(),
                    ),
                )
                self._conn.execute(
                    "UPDATE jobs SET trans_version = ?, updated_at = ? WHERE id = ?",
                    (version, utc_now(), job_id),
                )
            except BaseException:
                # The row never committed, so don't leave an orphaned file.
                if content_path is not None:
                    content_path.unlink(missing_ok=True)
                raise
        return version

    def _write_transcript_file(self, job_id: str, version: int, payload: str) -> Path:
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        path = self.transcript_dir / f"{job_id}.v{version}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)  # atomic: readers never see a half-written file
        return path

    def get_transcript(
        self, job_id: str, version: int | None = None
    ) -> dict[str, Any] | None:
        """Return a transcript (latest version by default), or None if absent."""
        with self._lock:
            if version is None:
                row = self._conn.execute(
                    """
                    SELECT t.content, t.content_path FROM transcripts t
                    JOIN jobs j ON j.id = t.job_id AND j.trans_version = t.version
                    WHERE t.job_id = ?
                    """,
                    (job_id,),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT content, content_path FROM transcripts "
                    "WHERE job_id = ? AND version = ?",
                    (job_id, version),
                ).fetchone()
        if row is None:
            return None
        if row["content"] is not None:
            return json.loads(row["content"])
        try:
            return json.loads(Path(row["content_path"]).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise TranscriptStorageError(
                f"Transcript file for job {job_id} is missing or unreadable"
            ) from exc
