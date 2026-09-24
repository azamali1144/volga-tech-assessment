"""Structured logging.

Every log line is one JSON object, so fields passed via ``extra=`` (job_id,
error_code, attempt, ...) become queryable fields in a log aggregator
(CloudWatch, Datadog, Loki, ELK) instead of substrings to grep for:

    logger.info("job completed", extra={"job_id": job.id, "segments": 12})
    -> {"ts": "...", "level": "INFO", "logger": "app.worker",
        "message": "job completed", "job_id": "...", "segments": 12}

``LOG_FORMAT=text`` switches to a human-readable line for local development.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# Attributes every LogRecord has; anything else on a record came from
# ``extra=`` and is emitted as its own field.
_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None))
) | {"message", "asctime", "color_message", "taskName"}

# Loggers that install their own handlers; routed through ours instead so the
# whole process emits one consistent format.
_THIRD_PARTY_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(record).items()
        if key not in _STANDARD_ATTRS and not key.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Structured fields never overwrite the core ones.
        for key, value in _extra_fields(record).items():
            entry.setdefault(key, value)
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            entry["stack_info"] = self.formatStack(record.stack_info)
        # default=str: an unexpected extra (a Path, a datetime) must never
        # make logging itself raise.
        return json.dumps(entry, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """``2026-09-25 10:00:00,123 INFO app.worker: job completed job_id=... segments=12``"""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = _extra_fields(record)
        if extras:
            line += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        return line


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install a single stdout handler on the root logger. Idempotent.

    Logs go to stdout (not files) per twelve-factor practice: the container
    runtime or process manager collects and ships them.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())

    for name in _THIRD_PARTY_LOGGERS:
        third_party = logging.getLogger(name)
        third_party.handlers.clear()
        third_party.propagate = True
