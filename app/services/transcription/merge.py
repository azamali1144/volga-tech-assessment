from __future__ import annotations

from typing import Sequence

from app.services.audio import AudioChunk
from app.services.transcription.models import (
    TIMESTAMP_PRECISION,
    Segment,
    TranscriptionResult,
    Word,
)


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
