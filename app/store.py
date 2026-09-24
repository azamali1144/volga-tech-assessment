"""Job and transcript persistence (Repository pattern).

``JobStore`` is the only code that talks to the database. Nothing else in the
app imports ``sqlite3``, so moving to PostgreSQL means changing this file:
the schema below uses only portable types (TEXT ids and ISO-8601 UTC
timestamps map to UUID/TIMESTAMPTZ; INTEGER/REAL map directly).
"""

from __future__ import annotations

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
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
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
