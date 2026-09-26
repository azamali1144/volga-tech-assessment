import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from app.audio import AudioChunk, AudioProcessingError, split_into_chunks
from app.config import Settings
from app.transcription_engine import (
    EngineUnavailableError,
    MockEngine,
    Segment,
    TranscriptionEngine,
    TranscriptionResult,
    WhisperEngine,
    Word,
    get_engine,
    merge_chunk_results,
)
from tests.conftest_helpers import make_tone_wav


class _TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


def _chunk(index, start, end):
    return AudioChunk(index=index, path=Path(f"chunk{index}.wav"), start=start, end=end)


def _result(*segments, language="en"):
    segs = [Segment(i, s, e, t) for i, (s, e, t) in enumerate(segments)]
    return TranscriptionResult(text=" ".join(s.text for s in segs), segments=segs, language=language)


def _spans(result):
    return [(s.start, s.end, s.text) for s in result.segments]


class DataShapeTest(unittest.TestCase):
    def test_result_round_trips_through_dict(self):
        r = _result((0.0, 1.25, "hi"), (1.5, 2.0, "there"))
        self.assertEqual(TranscriptionResult.from_dict(r.to_dict()), r)

    def test_words_round_trip_and_are_omitted_when_absent(self):
        seg = Segment(0, 0.0, 1.0, "hi there", words=(Word(0.0, 0.4, "hi"), Word(0.5, 1.0, "there")))
        self.assertEqual(Segment.from_dict(seg.to_dict()), seg)
        self.assertNotIn("words", Segment(0, 0.0, 1.0, "hi").to_dict())


class MockEngineTest(_TempDirTestCase):
    def test_is_deterministic(self):
        wav = make_tone_wav(self.tmp / "a.wav", seconds=7.3)
        engine = MockEngine()
        self.assertEqual(engine.transcribe(wav), engine.transcribe(wav))

    def test_segments_tile_the_full_duration(self):
        wav = make_tone_wav(self.tmp / "a.wav", seconds=7.3)
        result = MockEngine(segment_seconds=2.0).transcribe(wav)
        self.assertEqual(
            [(s.start, s.end) for s in result.segments],
            [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 7.3)],
        )
        self.assertEqual(result.duration, 7.3)
        self.assertEqual([s.id for s in result.segments], [0, 1, 2, 3])

    def test_satisfies_engine_protocol(self):
        self.assertIsInstance(MockEngine(), TranscriptionEngine)

    def test_words_are_spread_evenly_across_each_segment(self):
        wav = make_tone_wav(self.tmp / "a.wav", seconds=3.0)
        first = MockEngine(segment_seconds=3.0).transcribe(wav).segments[0]
        self.assertEqual([w.text for w in first.words], [" mock", " speech", " 0.00-3.00"])
        self.assertEqual([(w.start, w.end) for w in first.words], [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])

    def test_word_timestamps_can_be_disabled(self):
        wav = make_tone_wav(self.tmp / "a.wav", seconds=3.0)
        result = MockEngine(word_timestamps=False).transcribe(wav)
        self.assertTrue(all(s.words == () for s in result.segments))

    def test_unreadable_input_raises_audio_error(self):
        bad = self.tmp / "bad.wav"
        bad.write_bytes(b"not a wav")
        with self.assertRaises(AudioProcessingError) as ctx:
            MockEngine().transcribe(bad)
        self.assertEqual(ctx.exception.error_code, "invalid_audio")


class GetEngineTest(unittest.TestCase):
    def test_selects_engine_from_settings(self):
        self.assertIsInstance(get_engine(Settings(transcription_engine="mock")), MockEngine)
        engine = get_engine(Settings(transcription_engine="whisper", whisper_model="tiny"))
        self.assertIsInstance(engine, WhisperEngine)
        self.assertEqual(engine.model_name, "tiny")


class _FakeWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, language=None, fp16=None, word_timestamps=False):
        self.calls.append(
            {"path": path, "language": language, "fp16": fp16, "word_timestamps": word_timestamps}
        )
        return {
            "language": "en",
            "segments": [
                {
                    "id": 0, "start": 0.0, "end": 1.23456, "text": " Hello world.",
                    "words": [
                        {"word": " Hello", "start": 0.0, "end": 0.504, "probability": 0.9},
                        {"word": " world.", "start": 0.6, "end": 1.23456, "probability": 0.9},
                    ],
                },
                {"id": 1, "start": 1.5, "end": 1.9, "text": "   ", "words": []},
                {
                    "id": 2, "start": 2.004, "end": 3.999, "text": " How are you?",
                    "words": [
                        {"word": " How", "start": 2.004, "end": 2.4, "probability": 0.9},
                        {"word": " ", "start": 2.4, "end": 2.41, "probability": 0.1},
                        {"word": " are", "start": 2.41, "end": 2.9, "probability": 0.9},
                        {"word": " you?", "start": 2.9, "end": 3.999, "probability": 0.9},
                    ],
                },
            ],
        }


class WhisperEngineTest(_TempDirTestCase):
    """Runs against fake ``whisper``/``torch`` modules: no model download."""

    def setUp(self):
        super().setUp()
        self.model = _FakeWhisperModel()
        self.loads = []

        def load_model(name, device):
            self.loads.append((name, device))
            return self.model

        fake_whisper = types.SimpleNamespace(load_model=load_model)
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False)
        )
        patcher = mock.patch.dict(sys.modules, {"whisper": fake_whisper, "torch": fake_torch})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.wav = make_tone_wav(self.tmp / "a.wav", seconds=4.2)

    def test_maps_whisper_output_to_shared_shape(self):
        result = WhisperEngine("tiny").transcribe(self.wav)
        self.assertEqual(
            _spans(result),
            [(0.0, 1.23, "Hello world."), (2.0, 4.0, "How are you?")],
        )
        self.assertEqual([s.id for s in result.segments], [0, 1])  # blank dropped, renumbered
        self.assertEqual(result.text, "Hello world. How are you?")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.duration, 4.2)

    def test_model_loads_lazily_and_exactly_once(self):
        engine = WhisperEngine("tiny")
        self.assertEqual(self.loads, [])
        threads = [threading.Thread(target=engine.transcribe, args=(self.wav,)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(self.loads, [("tiny", "cpu")])
        self.assertEqual(len(self.model.calls), 8)

    def test_fp16_disabled_on_cpu_and_language_passed_through(self):
        WhisperEngine("tiny", language="fr").transcribe(self.wav)
        self.assertEqual(self.model.calls[0]["fp16"], False)
        self.assertEqual(self.model.calls[0]["language"], "fr")

    def test_requests_and_maps_word_timestamps(self):
        result = WhisperEngine("tiny").transcribe(self.wav)
        self.assertTrue(self.model.calls[0]["word_timestamps"])
        first, second = result.segments
        self.assertEqual(
            [(w.start, w.end, w.text) for w in first.words],
            [(0.0, 0.5, " Hello"), (0.6, 1.23, " world.")],
        )
        # Blank words are dropped; the rest keep their timings and spacing.
        self.assertEqual([w.text for w in second.words], [" How", " are", " you?"])


class WhisperEngineMissingDependencyTest(unittest.TestCase):
    def test_missing_package_raises_engine_unavailable(self):
        # A None entry in sys.modules makes the import raise ImportError.
        with mock.patch.dict(sys.modules, {"whisper": None}):
            with self.assertRaises(EngineUnavailableError) as ctx:
                WhisperEngine().transcribe(Path("unused.wav"))
        self.assertIn("requirements-whisper.txt", str(ctx.exception))


class MergeSeamCasesTest(unittest.TestCase):
    """Hand-built seam scenarios. Chunk 0 = [0, 10], chunk 1 = [8, 18]:
    the overlap is [8, 10] and the cut sits at its midpoint, 9.0."""

    chunks = [_chunk(0, 0, 10), _chunk(1, 8, 18)]

    def merge(self, first, second):
        return merge_chunk_results(self.chunks, [first, second])

    def test_timestamps_are_shifted_by_chunk_offset(self):
        merged = self.merge(_result((1, 3, "a")), _result((5, 7, "b")))
        self.assertEqual(_spans(merged), [(1, 3, "a"), (13, 15, "b")])

    def test_word_centered_exactly_on_cut_is_kept_once_by_later_chunk(self):
        merged = self.merge(
            _result((8.5, 9.5, "boundary-first")),
            _result((0.5, 1.5, "boundary-second")),  # 8.5-9.5 globally
        )
        self.assertEqual(_spans(merged), [(8.5, 9.5, "boundary-second")])

    def test_phrase_straddling_cut_kept_once_by_chunk_holding_most_of_it(self):
        merged = self.merge(
            _result((7.0, 9.6, "hello there")),
            _result((0.2, 1.6, "there")),  # 8.2-9.6: tail of the same phrase
        )
        self.assertEqual(_spans(merged), [(7.0, 9.6, "hello there")])

    def test_word_clipped_at_chunk_edge_replaced_by_complete_version(self):
        merged = self.merge(
            _result((9.3, 10.0, "extraord")),       # cut off by chunk 0's edge
            _result((1.3, 2.4, "extraordinary")),   # 9.3-10.4, heard in full
        )
        self.assertEqual(_spans(merged), [(9.3, 10.4, "extraordinary")])

    def test_differently_segmented_seam_leaves_no_hole(self):
        merged = self.merge(
            _result((6, 8, "a"), (8, 10, "b")),
            _result((0, 1.2, "b-part"), (1.2, 3, "b-rest")),
        )
        spans = [(s, e) for s, e, _ in _spans(merged)]
        self.assertEqual(spans, [(6, 8), (8, 9.2), (9.2, 11)])

    def test_overlapping_kept_segment_is_clamped_not_backwards(self):
        merged = self.merge(
            _result((7.0, 8.9, "a")),
            _result((0.6, 2.0, "b")),  # 8.6-10.0, center 9.3 >= cut
        )
        self.assertEqual(_spans(merged), [(7.0, 8.9, "a"), (8.9, 10.0, "b")])

    def test_input_order_does_not_matter(self):
        first, second = _result((1, 3, "a")), _result((5, 7, "b"))
        forward = merge_chunk_results(self.chunks, [first, second])
        reverse = merge_chunk_results(self.chunks[::-1], [second, first])
        self.assertEqual(forward, reverse)

    def test_language_is_majority_vote(self):
        chunks = [_chunk(0, 0, 10), _chunk(1, 8, 18), _chunk(2, 16, 20)]
        results = [_result(language="en"), _result(language="fr"), _result(language="fr")]
        self.assertEqual(merge_chunk_results(chunks, results).language, "fr")

    def test_empty_and_mismatched_inputs(self):
        empty = merge_chunk_results([], [])
        self.assertEqual((empty.text, empty.segments, empty.duration), ("", [], 0.0))
        with self.assertRaises(ValueError):
            merge_chunk_results(self.chunks, [_result()])


def _wseg(seg_id, *words):
    """Segment from (start, end, text) words, Whisper-style leading spaces."""
    ws = tuple(Word(start, end, text) for start, end, text in words)
    return Segment(seg_id, ws[0].start, ws[-1].end, "".join(w.text for w in ws).strip(), words=ws)


def _wresult(*segments):
    return TranscriptionResult(text=" ".join(s.text for s in segments), segments=list(segments), language="en")


def _texts(result):
    return [s.text for s in result.segments]


def _all_words(result):
    return [w.text.strip() for s in result.segments for w in s.words]


class WordLevelMergeTest(unittest.TestCase):
    """Chunk 0 = [0, 20], chunk 1 = [15, 35]: overlap [15, 20], cut at 17.5.
    Chunk 1's word times below are chunk-local (global = local + 15)."""

    chunks = [_chunk(0, 0, 20), _chunk(1, 15, 35)]

    def merge(self, first, second):
        return merge_chunk_results(self.chunks, [first, second])

    def test_word_clipped_at_chunk_edge_is_replaced_and_nothing_is_lost(self):
        # Real case: chunk 0 heard "quarter" cut off at its edge as "court.",
        # and a segment-level merge also lost "driven mostly by".
        first = _wresult(_wseg(
            0,
            (16.0, 16.4, " Revenue"), (16.5, 16.9, " grew"), (17.0, 17.3, " by"),
            (17.8, 18.4, " the"), (18.5, 19.2, " previous"), (19.3, 20.0, " court."),
        ))
        second = _wresult(
            _wseg(0, (1.0, 1.4, " Revenue"), (1.5, 1.9, " grew"), (2.0, 2.3, " by"),
                  (2.8, 3.4, " the"), (3.5, 4.2, " previous"), (4.3, 5.0, " quarter,")),
            _wseg(1, (5.1, 5.6, " driven"), (5.7, 6.1, " mostly"), (6.2, 6.4, " by")),
        )
        merged = self.merge(first, second)
        self.assertEqual(
            _all_words(merged),
            ["Revenue", "grew", "by", "the", "previous", "quarter,", "driven", "mostly", "by"],
        )
        self.assertNotIn("court", " ".join(_texts(merged)))

    def test_nothing_is_repeated_when_chunks_segment_differently(self):
        # Real case: the segment merge produced "take-home / assessment. /
        # home assessment."
        first = _wresult(_wseg(
            0, (16.2, 16.8, " a"), (16.9, 17.3, " short"), (17.4, 18.2, " take-home"),
            (18.3, 19.4, " assessment."),
        ))
        second = _wresult(
            _wseg(0, (1.2, 1.8, " a"), (1.9, 2.3, " short"), (2.4, 2.9, " take-")),
            _wseg(1, (3.0, 3.2, " home"), (3.3, 4.4, " assessment."), (5.0, 5.6, " Finally,")),
        )
        merged = self.merge(first, second)
        self.assertEqual(_all_words(merged), ["a", "short", "take-", "home", "assessment.", "Finally,"])

    def test_word_centered_exactly_on_cut_is_kept_once_by_later_chunk(self):
        first = _wresult(_wseg(0, (16.0, 16.5, " before"), (17.0, 18.0, " boundary")))
        second = _wresult(_wseg(0, (1.0, 1.5, " before"), (2.0, 3.0, " boundary"), (3.5, 4.0, " after")))
        merged = self.merge(first, second)
        self.assertEqual(_all_words(merged), ["before", "boundary", "after"])
        boundary = merged.segments[-1].words[-2]
        self.assertEqual((boundary.start, boundary.end), (17.0, 18.0))

    def test_sentence_split_by_the_cut_is_joined_into_one_segment(self):
        first = _wresult(_wseg(0, (16.0, 16.6, " Revenue"), (16.7, 17.2, " grew"), (17.6, 18.0, " by"), (18.1, 18.9, " twelve")))
        second = _wresult(_wseg(0, (1.0, 1.6, " Revenue"), (1.7, 2.2, " grew"), (2.6, 3.0, " by"), (3.1, 3.9, " twelve"), (4.0, 4.9, " percent.")))
        merged = self.merge(first, second)
        self.assertEqual(_texts(merged), ["Revenue grew by twelve percent."])
        self.assertEqual((merged.segments[0].start, merged.segments[0].end), (16.0, 19.9))

    def test_punctuation_spacing_survives_rebuilding(self):
        first = _wresult(_wseg(0, (16.0, 16.5, " grew"), (16.6, 17.0, " by"), (17.6, 18.0, " 12"), (18.0, 18.2, "%")))
        second = _wresult(_wseg(0, (1.0, 1.5, " grew"), (1.6, 2.0, " by"), (2.6, 3.0, " 12"), (3.0, 3.2, "%"), (3.3, 3.6, " growth.")))
        self.assertEqual(_texts(self.merge(first, second)), ["grew by 12% growth."])

    def test_seam_word_with_shifted_timing_is_not_duplicated(self):
        # Chunk 0 keeps "assessment." (17.0-17.9, center 17.45 < cut 17.5).
        # Chunk 1 times the same word 0.5s later (17.5-18.4): its center 17.95
        # is past everything kept so far, so the center rule alone would keep
        # it twice. Same text + overlapping in time = the same word.
        first = _wresult(_wseg(0, (16.4, 16.9, " the"), (17.0, 17.9, " assessment.")))
        second = _wresult(_wseg(0, (1.4, 1.9, " the"), (2.5, 3.4, " assessment."), (3.6, 4.2, " Finally,")))
        self.assertEqual(_all_words(self.merge(first, second)), ["the", "assessment.", "Finally,"])

    def test_genuinely_repeated_word_at_seam_is_kept(self):
        first = _wresult(_wseg(0, (16.4, 17.0, " very")))
        second = _wresult(_wseg(0, (1.4, 2.0, " very"), (2.1, 2.7, " very"), (2.8, 3.4, " good.")))
        self.assertEqual(_all_words(self.merge(first, second)), ["very", "very", "good."])

    def test_fully_kept_segments_keep_engine_text_and_bounds(self):
        first = _wresult(Segment(0, 1.0, 3.2, "Hello there.", words=(Word(1.1, 1.6, " Hello"), Word(1.7, 3.0, " there."))))
        second = _wresult(Segment(0, 5.0, 7.0, "Later text.", words=(Word(5.0, 5.8, " Later"), Word(5.9, 7.0, " text."))))
        merged = self.merge(first, second)
        self.assertEqual(_spans(merged), [(1.0, 3.2, "Hello there."), (20.0, 22.0, "Later text.")])

    def test_falls_back_to_segment_merge_when_any_segment_lacks_words(self):
        first = _wresult(_wseg(0, (1.0, 2.0, " hi")))
        second = _result((5.0, 7.0, "no word timings"))
        merged = self.merge(first, second)
        self.assertEqual(_spans(merged), [(1.0, 2.0, "hi"), (20.0, 22.0, "no word timings")])
        self.assertEqual(merged.segments[1].words, ())


class MergeWithRealChunksTest(_TempDirTestCase):
    """Split real WAVs, transcribe each chunk with MockEngine, merge, and
    compare against transcribing the whole file in one pass."""

    engine = MockEngine(segment_seconds=2.0)

    def chunk_and_merge(self, seconds, length, overlap, engine=None):
        engine = engine or self.engine
        wav = make_tone_wav(self.tmp / f"{seconds}-{length}-{overlap}.wav", seconds=seconds)
        chunks = split_into_chunks(wav, self.tmp / f"c-{seconds}-{length}-{overlap}", length, overlap)
        merged = merge_chunk_results(chunks, [engine.transcribe(c.path) for c in chunks])
        return wav, chunks, merged

    def assert_well_formed_timeline(self, merged, duration):
        segs = merged.segments
        self.assertEqual(segs[0].start, 0.0)
        self.assertAlmostEqual(segs[-1].end, duration, places=2)
        self.assertEqual([s.id for s in segs], list(range(len(segs))))
        for a, b in zip(segs, segs[1:]):
            self.assertGreaterEqual(b.start, a.end, "timeline went backwards / overlapped")
        for s in segs:
            self.assertLess(s.start, s.end)

    def test_merge_chunk_results_dedupes_overlap_and_spans_full_duration(self):
        """When chunk boundaries line up with the segmentation, chunked +
        merged must equal a single pass: no gaps, no duplicates, full span."""
        for seconds, length, overlap in ((25, 10, 2), (123.45, 30, 6), (600, 240, 4)):
            with self.subTest(seconds=seconds, length=length, overlap=overlap):
                wav, chunks, merged = self.chunk_and_merge(seconds, length, overlap)
                self.assertGreater(len(chunks), 1)
                single = self.engine.transcribe(wav)
                self.assertEqual(
                    [(s.start, s.end) for s in merged.segments],
                    [(s.start, s.end) for s in single.segments],
                )
                self.assertEqual(merged.duration, single.duration)
                self.assert_well_formed_timeline(merged, seconds)

    MISALIGNED = ((660, 240, 5), (61, 20, 3), (100, 7, 2.5))

    def test_misaligned_seams_lose_nothing_with_word_timings(self):
        """Chunk offsets off the mock's 2s grid make the chunks segment each
        seam differently. Merging by words, the timeline is still covered
        with no gap longer than half a word (the mock's words are ~0.67s)."""
        half_word = self.engine.segment_seconds / 3 / 2
        for seconds, length, overlap in self.MISALIGNED:
            with self.subTest(seconds=seconds, length=length, overlap=overlap):
                _, _, merged = self.chunk_and_merge(seconds, length, overlap)
                self.assert_well_formed_timeline(merged, seconds)
                words = [w for seg in merged.segments for w in seg.words]
                for a, b in zip(words, words[1:]):
                    self.assertGreaterEqual(b.start, a.end)
                    self.assertLessEqual(b.start - a.end, half_word + 0.01, "gap too large")

    def test_segment_fallback_error_is_bounded_by_half_a_segment(self):
        """Without word timings the merge works on whole segments; the
        documented worst case at a misaligned seam is half a segment."""
        engine = MockEngine(segment_seconds=2.0, word_timestamps=False)
        half_segment = engine.segment_seconds / 2
        for seconds, length, overlap in self.MISALIGNED:
            with self.subTest(seconds=seconds, length=length, overlap=overlap):
                _, _, merged = self.chunk_and_merge(seconds, length, overlap, engine)
                self.assert_well_formed_timeline(merged, seconds)
                segs = merged.segments
                for a, b in zip(segs, segs[1:]):
                    self.assertLess(b.start - a.end, half_segment, "gap too large")


if __name__ == "__main__":
    unittest.main()
