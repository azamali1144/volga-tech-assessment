"""Transcription engines behind one interface (Strategy pattern).

The rest of the app depends only on ``TranscriptionEngine`` and the
``Segment``/``TranscriptionResult`` shapes below; which concrete engine runs
is a configuration choice, not something callers know about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

TIMESTAMP_PRECISION = 2  # seconds, rounded to 10ms


@dataclass(frozen=True)
class Segment:
    """One timestamped span of speech. Times are seconds from the start of
    whatever audio was transcribed (a chunk, until merged; then the file)."""

    id: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "start": self.start, "end": self.end, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        return cls(
            id=int(data["id"]),
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data["text"]),
        )


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    duration: float | None = None  # seconds of audio covered

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionResult":
        return cls(
            text=str(data["text"]),
            segments=[Segment.from_dict(s) for s in data.get("segments", [])],
            language=data.get("language"),
            duration=data.get("duration"),
        )


@runtime_checkable
class TranscriptionEngine(Protocol):
    """Anything that turns an audio file into a ``TranscriptionResult``.

    ``transcribe`` is deliberately synchronous: real engines (Whisper) are
    CPU/GPU-bound blocking calls, so the async worker runs them via
    ``asyncio.to_thread`` rather than every engine reimplementing that.
    Input is always a normalized 16kHz mono WAV (see ``app.audio``).
    """

    name: str

    def transcribe(self, audio_path: Path) -> TranscriptionResult: ...
