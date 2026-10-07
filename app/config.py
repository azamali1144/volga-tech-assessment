from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

VALID_ENGINES = frozenset({"mock", "whisper"})
VALID_LOG_FORMATS = frozenset({"json", "text"})
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_csv(name: str, default: str) -> frozenset[str]:
    raw = os.getenv(name, default)
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    api_keys: frozenset[str] = frozenset({"dev-local-key"})
    rate_limit_requests: int = 60
    rate_limit_window_seconds: float = 60.0

    max_upload_bytes: int = 200 * 1024 * 1024
    allowed_extensions: frozenset[str] = frozenset(
        {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4"}
    )

    sample_rate: int = 16_000
    chunk_threshold_seconds: float = 300.0
    chunk_length_seconds: float = 240.0
    chunk_overlap_seconds: float = 5.0
    max_concurrent_chunk_transcriptions: int = 2

    transcription_engine: str = "mock"
    whisper_model: str = "base"

    max_retries: int = 3
    retry_backoff_base_seconds: float = 2.0
    worker_count: int = 1

    log_level: str = "INFO"
    log_format: str = "json"

    storage_dir: Path = Path("storage")
    inline_transcript_max_chars: int = 20_000
    audio_dir: Path | None = None
    transcript_dir: Path | None = None
    dead_letter_dir: Path | None = None
    db_path: Path | None = None

    def __post_init__(self) -> None:
        derived = {
            "audio_dir": self.storage_dir / "audio",
            "transcript_dir": self.storage_dir / "transcripts",
            "dead_letter_dir": self.storage_dir / "dead_letter",
            "db_path": self.storage_dir / "jobs.db",
        }
        for attr, default in derived.items():
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, default)
        self._validate()

    def _validate(self) -> None:
        if self.transcription_engine not in VALID_ENGINES:
            raise ValueError(
                f"TRANSCRIPTION_ENGINE must be one of {sorted(VALID_ENGINES)}, "
                f"got {self.transcription_engine!r}"
            )
        if self.log_format not in VALID_LOG_FORMATS:
            raise ValueError(f"LOG_FORMAT must be one of {sorted(VALID_LOG_FORMATS)}")
        if self.log_level not in VALID_LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(VALID_LOG_LEVELS)}")
        if not self.api_keys:
            raise ValueError("API_KEYS must contain at least one key")
        if self.chunk_length_seconds <= 0:
            raise ValueError("CHUNK_LENGTH_SECONDS must be positive")
        if not 0 <= self.chunk_overlap_seconds < self.chunk_length_seconds:
            raise ValueError(
                "CHUNK_OVERLAP_SECONDS must be >= 0 and smaller than CHUNK_LENGTH_SECONDS"
            )
        for name in (
            "max_upload_bytes",
            "max_concurrent_chunk_transcriptions",
            "rate_limit_requests",
            "worker_count",
            "inline_transcript_max_chars",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name.upper()} must be positive")
        if self.max_retries < 0:
            raise ValueError("MAX_RETRIES must be >= 0")

    def ensure_dirs(self) -> None:
        for path in (
            self.storage_dir,
            self.audio_dir,
            self.transcript_dir,
            self.dead_letter_dir,
            self.db_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        storage_dir = Path(_env_str("STORAGE_DIR", str(d.storage_dir)))

        def _env_path(name: str) -> Path | None:
            raw = os.getenv(name)
            return Path(raw) if raw else None

        return cls(
            api_keys=_env_csv("API_KEYS", ",".join(sorted(d.api_keys))),
            rate_limit_requests=_env_int("RATE_LIMIT_REQUESTS", d.rate_limit_requests),
            rate_limit_window_seconds=_env_float(
                "RATE_LIMIT_WINDOW_SECONDS", d.rate_limit_window_seconds
            ),
            max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", d.max_upload_bytes),
            allowed_extensions=frozenset(
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in _env_csv(
                    "ALLOWED_EXTENSIONS", ",".join(sorted(d.allowed_extensions))
                )
            ),
            sample_rate=_env_int("SAMPLE_RATE", d.sample_rate),
            chunk_threshold_seconds=_env_float(
                "CHUNK_THRESHOLD_SECONDS", d.chunk_threshold_seconds
            ),
            chunk_length_seconds=_env_float("CHUNK_LENGTH_SECONDS", d.chunk_length_seconds),
            chunk_overlap_seconds=_env_float(
                "CHUNK_OVERLAP_SECONDS", d.chunk_overlap_seconds
            ),
            max_concurrent_chunk_transcriptions=_env_int(
                "MAX_CONCURRENT_CHUNK_TRANSCRIPTIONS", d.max_concurrent_chunk_transcriptions
            ),
            transcription_engine=_env_str(
                "TRANSCRIPTION_ENGINE", d.transcription_engine
            ).lower(),
            whisper_model=_env_str("WHISPER_MODEL", d.whisper_model),
            max_retries=_env_int("MAX_RETRIES", d.max_retries),
            retry_backoff_base_seconds=_env_float(
                "RETRY_BACKOFF_BASE_SECONDS", d.retry_backoff_base_seconds
            ),
            worker_count=_env_int("WORKER_COUNT", d.worker_count),
            log_level=_env_str("LOG_LEVEL", d.log_level).upper(),
            log_format=_env_str("LOG_FORMAT", d.log_format).lower(),
            storage_dir=storage_dir,
            inline_transcript_max_chars=_env_int(
                "INLINE_TRANSCRIPT_MAX_CHARS", d.inline_transcript_max_chars
            ),
            audio_dir=_env_path("AUDIO_DIR"),
            transcript_dir=_env_path("TRANSCRIPT_DIR"),
            dead_letter_dir=_env_path("DEAD_LETTER_DIR"),
            db_path=_env_path("DB_PATH"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
