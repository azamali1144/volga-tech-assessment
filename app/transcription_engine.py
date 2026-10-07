from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from app.audio import AudioChunk, wav_duration_seconds
from app.config import Settings

logger = logging.getLogger(__name__)

TIMESTAMP_PRECISION = 2


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Word":
        return cls(start=float(data["start"]), end=float(data["end"]), text=str(data["text"]))


@dataclass(frozen=True)
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }
        if self.words:
            data["words"] = [w.to_dict() for w in self.words]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        return cls(
            id=int(data["id"]),
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data["text"]),
            words=tuple(Word.from_dict(w) for w in data.get("words", ())),
        )


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    duration: float | None = None

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


def merge_chunk_results(
    chunks: Sequence[AudioChunk], results: Sequence[TranscriptionResult]
) -> TranscriptionResult:
    if len(chunks) != len(results):
        raise ValueError("chunks and results must be the same length")
    if not chunks:
        return TranscriptionResult(text="", segments=[], language=None, duration=0.0)

    pairs = sorted(zip(chunks, results), key=lambda pair: pair[0].start)
    ordered_chunks = [c for c, _ in pairs]

    cuts = [
        (nxt.start + prev.end) / 2
        for prev, nxt in zip(ordered_chunks, ordered_chunks[1:])
    ]
    upper_bounds = [*cuts, float("inf")]

    all_segments = [seg for _, result in pairs for seg in result.segments]
    if all_segments and all(seg.words for seg in all_segments):
        segments = _merge_by_words(pairs, upper_bounds)
    else:
        segments = _merge_by_segments(pairs, upper_bounds)

    languages = [r.language for _, r in pairs if r.language]
    return TranscriptionResult(
        text=" ".join(s.text for s in segments),
        segments=segments,
        language=max(languages, key=languages.count) if languages else None,
        duration=round(ordered_chunks[-1].end, TIMESTAMP_PRECISION),
    )


def join_words(words: Sequence[Word]) -> str:
    return "".join(w.text for w in words).strip()


def _normalize_word(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _merge_by_words(
    pairs: list[tuple[AudioChunk, TranscriptionResult]], upper_bounds: list[float]
) -> list[Segment]:
    segments: list[Segment] = []
    kept_until = float("-inf")
    last_word: Word | None = None
    tail_cut = False

    for (chunk, result), upper in zip(pairs, upper_bounds):
        first_in_chunk = True
        for seg in sorted(result.segments, key=lambda s: (s.start, s.end)):
            kept: list[Word] = []
            head_cut = False
            seg_tail_cut = False
            for word in sorted(seg.words, key=lambda w: (w.start, w.end)):
                start, end = word.start + chunk.start, word.end + chunk.start
                center = (start + end) / 2
                if center >= upper:
                    seg_tail_cut = True
                    continue
                if center < kept_until:
                    head_cut = head_cut or not kept
                    continue
                if (
                    first_in_chunk
                    and last_word is not None
                    and start < last_word.end
                    and _normalize_word(word.text) == _normalize_word(last_word.text)
                ):
                    first_in_chunk = False
                    head_cut = head_cut or not kept
                    continue
                first_in_chunk = False
                start = max(start, kept_until)
                if start >= end:
                    continue
                last_word = Word(
                    start=round(start, TIMESTAMP_PRECISION),
                    end=round(end, TIMESTAMP_PRECISION),
                    text=word.text,
                )
                kept.append(last_word)
                kept_until = end
            if not kept:
                continue

            if head_cut and tail_cut and segments:
                prev = segments[-1]
                words = prev.words + tuple(kept)
                segments[-1] = Segment(
                    id=prev.id,
                    start=prev.start,
                    end=kept[-1].end,
                    text=join_words(words),
                    words=words,
                )
            elif not head_cut and not seg_tail_cut:
                seg_start = round(seg.start + chunk.start, TIMESTAMP_PRECISION)
                if segments:
                    seg_start = max(seg_start, segments[-1].end)
                segments.append(
                    Segment(
                        id=len(segments),
                        start=min(seg_start, kept[0].start),
                        end=max(round(seg.end + chunk.start, TIMESTAMP_PRECISION), kept[-1].end),
                        text=seg.text,
                        words=tuple(kept),
                    )
                )
            else:
                segments.append(
                    Segment(
                        id=len(segments),
                        start=kept[0].start,
                        end=kept[-1].end,
                        text=join_words(kept),
                        words=tuple(kept),
                    )
                )
            tail_cut = seg_tail_cut
    return segments


def _merge_by_segments(
    pairs: list[tuple[AudioChunk, TranscriptionResult]], upper_bounds: list[float]
) -> list[Segment]:
    segments: list[Segment] = []
    kept_until = float("-inf")
    for (chunk, result), upper in zip(pairs, upper_bounds):
        for seg in sorted(result.segments, key=lambda s: (s.start, s.end)):
            start, end = seg.start + chunk.start, seg.end + chunk.start
            center = (start + end) / 2
            if not kept_until <= center < upper:
                continue
            start = max(start, kept_until)
            if start >= end:
                continue
            kept_until = end
            segments.append(
                Segment(
                    id=len(segments),
                    start=round(start, TIMESTAMP_PRECISION),
                    end=round(end, TIMESTAMP_PRECISION),
                    text=seg.text,
                )
            )
    return segments
