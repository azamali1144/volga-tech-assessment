"""Transcription engines behind one interface (Strategy pattern).

The rest of the app depends only on ``TranscriptionEngine`` and the
``Segment``/``TranscriptionResult`` shapes below; which concrete engine runs
is a configuration choice, not something callers know about.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from app.audio import AudioChunk, wav_duration_seconds
from app.config import Settings

TIMESTAMP_PRECISION = 2  # seconds, rounded to 10ms


@dataclass(frozen=True)
class Word:
    """One word with its own timing, on the same timeline as its segment.

    Following Whisper's convention, ``text`` includes any leading space
    (" quarter", but "%" or "," attached to the previous word), so joining
    words with ``join_words`` reproduces the original spacing exactly.
    """

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
    """One timestamped span of speech. Times are seconds from the start of
    whatever audio was transcribed (a chunk, until merged; then the file).

    ``words`` is optional: engines that report word timings fill it, and the
    chunk merge then works word by word instead of segment by segment.
    """

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

    error_code = "engine_unavailable"


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
                str(audio_path),
                language=self.language,
                fp16=self._fp16,
                # Per-word timings let the chunk merge cut between words
                # rather than between whole (multi-second) segments.
                word_timestamps=True,
            )
        return self._to_result(raw, wav_duration_seconds(audio_path))

    @staticmethod
    def _to_result(raw: dict[str, Any], duration: float | None) -> TranscriptionResult:
        """Map Whisper's output dict onto the shared result shape."""
        segments = []
        for seg in raw.get("segments", []):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            words = tuple(
                Word(
                    start=round(float(w["start"]), TIMESTAMP_PRECISION),
                    end=round(float(w["end"]), TIMESTAMP_PRECISION),
                    text=str(w["word"]),  # keeps Whisper's leading space
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
    """Deterministic stand-in for a real engine: no model, no network.

    Emits one segment every ``segment_seconds`` across the audio's actual
    duration, with text naming the segment's time span. The same file always
    yields the same result, and because the output follows the real audio
    length, chunking and merging are exercised exactly as they would be with
    Whisper. Like Whisper, it reports per-word timings (words evenly spaced
    across each segment) unless ``word_timestamps=False``, which exercises the
    segment-level merge fallback. Used by the test suite and offline dev.
    """

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
        # Optional artificial latency, to make queueing/concurrency observable
        # when poking at the running service by hand.
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
            start = len(segments) * self.segment_seconds  # no float drift
        return TranscriptionResult(
            text=" ".join(s.text for s in segments),
            segments=segments,
            language="en",
            duration=round(duration, TIMESTAMP_PRECISION),
        )


    @staticmethod
    def _spread_words(text: str, start: float, end: float) -> tuple[Word, ...]:
        """Split ``text`` into words evenly spaced over ``[start, end]``."""
        tokens = text.split()
        step = (end - start) / len(tokens)
        return tuple(
            Word(
                start=round(start + i * step, TIMESTAMP_PRECISION),
                end=round(end if i == len(tokens) - 1 else start + (i + 1) * step, TIMESTAMP_PRECISION),
                text=f" {token}",  # Whisper-style leading space
            )
            for i, token in enumerate(tokens)
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


# ---------------------------------------------------------------------------
# Chunk merging
# ---------------------------------------------------------------------------


def merge_chunk_results(
    chunks: Sequence[AudioChunk], results: Sequence[TranscriptionResult]
) -> TranscriptionResult:
    """Stitch per-chunk transcripts into one transcript on the file's timeline.

    Chunks overlap (see ``app.audio.plan_chunks``), so speech in an overlap
    region is transcribed twice. The rules:

    1. **Shift.** Times are chunk-local; add the chunk's ``start`` to put them
       on the global timeline.
    2. **Cut each overlap at its midpoint.** Each overlap ``[next.start,
       prev.end]`` gets one cut at its midpoint. Audio right at a chunk's edge
       is where a model hears the least context and most often clips or
       mishears a word; cutting at the midpoint means everything kept is at
       least ``overlap / 2`` seconds from the edge of the chunk it came from.
    3. **Keep by center.** Walking chunks in order, an item is kept if its
       center is before this chunk's cut (upper bound) and at or after the end
       of what has been kept so far (lower bound). An item centered *exactly*
       on the cut belongs to the later chunk (half-open interval): kept once,
       never twice, never dropped. Using "where the transcript so far ends" as
       the lower bound, rather than the fixed cut, means the later chunk picks
       up exactly where the earlier one stopped, so no stretch falls between.
    4. **Clamp** so timestamps never go backwards or overlap.

    **Granularity.** When every segment carries word timings (Whisper with
    ``word_timestamps=True``), the rules apply to individual *words*. A word
    lasts well under a second and the cut sits ``overlap / 2`` from either
    chunk edge, so any word near the cut was heard in full by both chunks:
    a word clipped at a chunk edge is replaced by the neighbour's complete
    one, and nothing is lost or repeated at the seam. Segments are then
    rebuilt from the words each chunk kept, preserving the chunks' segment
    boundaries; a sentence split by the cut (its start kept by one chunk, its
    end by the next) is joined back into one segment. As a guard against
    word-timing jitter between chunks, a chunk's first kept word is skipped
    if it is the previous word again (same text, overlapping in time).

    Without word timings, the same rules apply to whole segments. That is
    exact when both chunks segment the seam identically, but because a
    segment can't be split, a seam error of up to half a segment (a few
    repeated or dropped words) is possible when they don't.

    The same idea (keep the middle of each chunk's window, discard the edges)
    is what Hugging Face's ASR pipeline does with ``stride_length_s``.
    """
    if len(chunks) != len(results):
        raise ValueError("chunks and results must be the same length")
    if not chunks:
        return TranscriptionResult(text="", segments=[], language=None, duration=0.0)

    pairs = sorted(zip(chunks, results), key=lambda pair: pair[0].start)
    ordered_chunks = [c for c, _ in pairs]

    # Cut after chunk i sits at the midpoint of its overlap with chunk i + 1.
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
        # Most common detected language; ties go to the earliest chunk.
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
    # Whether the last emitted segment lost trailing words to the cut, i.e. a
    # sentence the next chunk will finish.
    tail_cut = False

    for (chunk, result), upper in zip(pairs, upper_bounds):
        first_in_chunk = True
        for seg in sorted(result.segments, key=lambda s: (s.start, s.end)):
            kept: list[Word] = []
            head_cut = False  # words dropped before the first kept one
            seg_tail_cut = False  # words dropped after the last kept one
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
                    # The previous chunk's last word again, with its timing
                    # shifted just past the cut: the same word, not a repeat.
                    # (A genuinely repeated word starts after the first ends.)
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
                # The previous chunk kept the start of this sentence and this
                # chunk its end: join them back into one segment.
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
                # Fully kept: use the engine's own text and bounds, clamped
                # after the previous segment.
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
    kept_until = float("-inf")  # end of the transcript assembled so far
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
