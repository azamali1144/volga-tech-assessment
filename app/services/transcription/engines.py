from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.core.config import Settings
from app.services.audio import wav_duration_seconds
from app.services.transcription.models import (
    TIMESTAMP_PRECISION,
    Segment,
    TranscriptionResult,
    Word,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class TranscriptionEngine(Protocol):
    name: str

    def transcribe(self, audio_path: Path) -> TranscriptionResult: ...


class EngineUnavailableError(RuntimeError):
    error_code = "engine_unavailable"


class WhisperEngine:
    name = "whisper"

    def __init__(self, model_name: str = "base", language: str | None = None) -> None:
        self.model_name = model_name
        self.language = language
        self._model: Any = None
        self._fp16 = False
        self._load_lock = threading.Lock()
        self._transcribe_lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        import torch
                        import whisper
                    except ImportError as exc:
                        raise EngineUnavailableError(
                            "TRANSCRIPTION_ENGINE=whisper requires the openai-whisper "
                            "package: pip install -r requirements-whisper.txt"
                        ) from exc
                    except OSError as exc:
                        logger.error("could not load the whisper engine", exc_info=True)
                        raise EngineUnavailableError(
                            "The Whisper engine could not be loaded on this server "
                            "(PyTorch failed to load). See the server logs."
                        ) from exc
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._fp16 = device == "cuda"
                    self._model = whisper.load_model(self.model_name, device=device)
        return self._model

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        model = self._get_model()
        with self._transcribe_lock:
            raw = model.transcribe(
                str(audio_path),
                language=self.language,
                fp16=self._fp16,
                word_timestamps=True,
            )
        return self._to_result(raw, wav_duration_seconds(audio_path))

    @staticmethod
    def _to_result(raw: dict[str, Any], duration: float | None) -> TranscriptionResult:
        segments = []
        for seg in raw.get("segments", []):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            words = tuple(
                Word(
                    start=round(float(w["start"]), TIMESTAMP_PRECISION),
                    end=round(float(w["end"]), TIMESTAMP_PRECISION),
                    text=str(w["word"]),
                )
                for w in seg.get("words") or ()
                if str(w.get("word", "")).strip()
            )
            segments.append(
                Segment(
                    id=len(segments),
                    start=round(float(seg["start"]), TIMESTAMP_PRECISION),
                    end=round(float(seg["end"]), TIMESTAMP_PRECISION),
                    text=text,
                    words=words,
                )
            )
        return TranscriptionResult(
            text=" ".join(s.text for s in segments),
            segments=segments,
            language=raw.get("language"),
            duration=round(duration, TIMESTAMP_PRECISION) if duration is not None else None,
        )


class MockEngine:
    name = "mock"

    def __init__(
        self,
        segment_seconds: float = 2.0,
        delay_seconds: float = 0.0,
        word_timestamps: bool = True,
    ) -> None:
        if segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        self.segment_seconds = segment_seconds
        self.word_timestamps = word_timestamps
        self.delay_seconds = delay_seconds

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        duration = wav_duration_seconds(audio_path)
        if self.delay_seconds:
            time.sleep(self.delay_seconds)

        segments = []
        start = 0.0
        while start < duration:
            end = min(start + self.segment_seconds, duration)
            text = f"mock speech {start:.2f}-{end:.2f}"
            segments.append(
                Segment(
                    id=len(segments),
                    start=round(start, TIMESTAMP_PRECISION),
                    end=round(end, TIMESTAMP_PRECISION),
                    text=text,
                    words=self._spread_words(text, start, end) if self.word_timestamps else (),
                )
            )
            start = len(segments) * self.segment_seconds
        return TranscriptionResult(
            text=" ".join(s.text for s in segments),
            segments=segments,
            language="en",
            duration=round(duration, TIMESTAMP_PRECISION),
        )


    @staticmethod
    def _spread_words(text: str, start: float, end: float) -> tuple[Word, ...]:
        tokens = text.split()
        step = (end - start) / len(tokens)
        return tuple(
            Word(
                start=round(start + i * step, TIMESTAMP_PRECISION),
                end=round(end if i == len(tokens) - 1 else start + (i + 1) * step, TIMESTAMP_PRECISION),
                text=f" {token}",
            )
            for i, token in enumerate(tokens)
        )


def get_engine(settings: Settings) -> TranscriptionEngine:
    if settings.transcription_engine == "whisper":
        return WhisperEngine(model_name=settings.whisper_model)
    if settings.transcription_engine == "mock":
        return MockEngine()
    raise ValueError(f"Unknown transcription engine: {settings.transcription_engine!r}")
