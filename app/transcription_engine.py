"""Transcription engines behind one interface (Strategy pattern).

The rest of the app depends only on ``TranscriptionEngine`` and the
``Segment``/``TranscriptionResult`` shapes below; which concrete engine runs
is a configuration choice, not something callers know about.
"""

from __future__ import annotations

import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.audio import AudioProcessingError
from app.config import Settings

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


class EngineUnavailableError(RuntimeError):
    """The configured engine can't run in this environment (e.g. missing deps).

    This is a deployment problem, not a property of the audio, so retrying the
    job won't help; the worker treats it as a permanent failure.
    """


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, EOFError, OSError):
        return None


class WhisperEngine:
    """OpenAI Whisper, run locally.

    The ``whisper`` package (and PyTorch) is imported and the model loaded on
    first use, not at construction or import time. That keeps app startup and
    the test suite fast, and means a deployment using the mock engine never
    needs the multi-GB dependency installed at all.
    """

    name = "whisper"

    def __init__(self, model_name: str = "base", language: str | None = None) -> None:
        self.model_name = model_name
        self.language = language
        self._model: Any = None
        self._fp16 = False
        self._load_lock = threading.Lock()
        # Whisper's transcribe() temporarily installs forward hooks on the
        # shared model for its KV cache, so two concurrent calls on one model
        # would corrupt each other's decoding. Serialize per engine instance;
        # scale Whisper throughput with more worker processes, not threads.
        self._transcribe_lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is None:
            with self._load_lock:
                if self._model is None:  # double-checked: load exactly once
                    try:
                        import torch
                        import whisper
                    except ImportError as exc:
                        raise EngineUnavailableError(
                            "TRANSCRIPTION_ENGINE=whisper requires the openai-whisper "
                            "package: pip install -r requirements-whisper.txt"
                        ) from exc
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._fp16 = device == "cuda"  # fp16 is unsupported on CPU
                    self._model = whisper.load_model(self.model_name, device=device)
        return self._model

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        model = self._get_model()
        with self._transcribe_lock:
            raw = model.transcribe(
                str(audio_path), language=self.language, fp16=self._fp16
            )
        return self._to_result(raw, _wav_duration(Path(audio_path)))

    @staticmethod
    def _to_result(raw: dict[str, Any], duration: float | None) -> TranscriptionResult:
        """Map Whisper's output dict onto the shared result shape."""
        segments = []
        for seg in raw.get("segments", []):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            segments.append(
                Segment(
                    id=len(segments),
                    start=round(float(seg["start"]), TIMESTAMP_PRECISION),
                    end=round(float(seg["end"]), TIMESTAMP_PRECISION),
                    text=text,
                )
            )
        return TranscriptionResult(
            text=" ".join(s.text for s in segments),
            segments=segments,
            language=raw.get("language"),
            duration=round(duration, TIMESTAMP_PRECISION) if duration is not None else None,
        )


class MockEngine:
    """Deterministic stand-in for a real engine: no model, no network.

    Emits one segment every ``segment_seconds`` across the audio's actual
    duration, with text naming the segment's time span. The same file always
    yields the same result, and because the output follows the real audio
    length, chunking and merging are exercised exactly as they would be with
    Whisper. Used by the test suite and for offline development.
    """

    name = "mock"

    def __init__(self, segment_seconds: float = 2.0, delay_seconds: float = 0.0) -> None:
        if segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        self.segment_seconds = segment_seconds
        # Optional artificial latency, to make queueing/concurrency observable
        # when poking at the running service by hand.
        self.delay_seconds = delay_seconds

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        duration = _wav_duration(Path(audio_path))
        if duration is None:
            raise AudioProcessingError(
                "invalid_audio", f"Not a readable WAV file: {Path(audio_path).name}"
            )
        if self.delay_seconds:
            time.sleep(self.delay_seconds)

        segments = []
        start = 0.0
        while start < duration:
            end = min(start + self.segment_seconds, duration)
            segments.append(
                Segment(
                    id=len(segments),
                    start=round(start, TIMESTAMP_PRECISION),
                    end=round(end, TIMESTAMP_PRECISION),
                    text=f"mock speech {start:.2f}-{end:.2f}",
                )
            )
            start = len(segments) * self.segment_seconds  # no float drift
        return TranscriptionResult(
            text=" ".join(s.text for s in segments),
            segments=segments,
            language="en",
            duration=round(duration, TIMESTAMP_PRECISION),
        )


def get_engine(settings: Settings) -> TranscriptionEngine:
    """Build the engine selected by ``TRANSCRIPTION_ENGINE``.

    The only place that knows the concrete engine classes; swapping engines is
    a config change, never a code change for callers.
    """
    if settings.transcription_engine == "whisper":
        return WhisperEngine(model_name=settings.whisper_model)
    if settings.transcription_engine == "mock":
        return MockEngine()
    raise ValueError(f"Unknown transcription engine: {settings.transcription_engine!r}")
