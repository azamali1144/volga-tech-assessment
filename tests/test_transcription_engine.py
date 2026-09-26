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
        self.assertEqual([w.text for w in first.words], ["mock", "speech", "0.00-3.00"])
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
            [(0.0, 0.5, "Hello"), (0.6, 1.23, "world.")],
        )
        # Blank words are dropped; the rest keep their own timings.
        self.assertEqual([w.text for w in second.words], ["How", "are", "you?"])


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


class MergeWithRealChunksTest(_TempDirTestCase):
    """Split real WAVs, transcribe each chunk with MockEngine, merge, and
    compare against transcribing the whole file in one pass."""

    engine = MockEngine(segment_seconds=2.0)

    def chunk_and_merge(self, seconds, length, overlap):
        wav = make_tone_wav(self.tmp / f"{seconds}-{length}-{overlap}.wav", seconds=seconds)
        chunks = split_into_chunks(wav, self.tmp / f"c-{seconds}-{length}-{overlap}", length, overlap)
        merged = merge_chunk_results(chunks, [self.engine.transcribe(c.path) for c in chunks])
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

    def test_misaligned_segmentation_error_is_bounded_by_half_a_segment(self):
        """Chunk offsets that don't line up with the mock's 2s grid make the
        two chunks segment each seam differently: the documented worst case."""
        half_segment = self.engine.segment_seconds / 2
        for seconds, length, overlap in ((660, 240, 5), (61, 20, 3), (100, 7, 2.5)):
            with self.subTest(seconds=seconds, length=length, overlap=overlap):
                _, _, merged = self.chunk_and_merge(seconds, length, overlap)
                self.assert_well_formed_timeline(merged, seconds)
                segs = merged.segments
                for a, b in zip(segs, segs[1:]):
                    self.assertLess(b.start - a.end, half_segment, "gap too large")


if __name__ == "__main__":
    unittest.main()
