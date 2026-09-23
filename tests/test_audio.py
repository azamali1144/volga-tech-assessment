import tempfile
import unittest
from pathlib import Path

from app.audio import (
    AudioProcessingError,
    normalize_to_wav,
    plan_chunks,
    probe_duration_seconds,
    split_into_chunks,
)
from tests.conftest_helpers import (
    FFMPEG_AVAILABLE,
    SKIP_REASON_NO_FFMPEG,
    make_tone_file,
    make_tone_wav,
    read_frames,
    wav_info,
)


class _TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class PlanChunksTest(unittest.TestCase):
    """Pure overlap/offset math; no audio files involved."""

    def test_windows_overlap_and_cover_full_duration(self):
        self.assertEqual(plan_chunks(25, 10, 2), [(0, 10), (8, 18), (16, 25)])

    def test_default_settings_on_eleven_minute_file(self):
        self.assertEqual(
            plan_chunks(660, 240, 5), [(0, 240), (235, 475), (470, 660)]
        )

    def test_exact_fit_does_not_emit_empty_trailing_chunk(self):
        self.assertEqual(plan_chunks(20, 10, 0), [(0, 10), (10, 20)])

    def test_audio_shorter_than_one_chunk_gives_single_window(self):
        self.assertEqual(plan_chunks(3, 10, 2), [(0, 3)])

    def test_invariants_hold_across_many_shapes(self):
        for duration in (1, 9.99, 10, 10.01, 59.5, 600, 7200.3):
            for length, overlap in ((10, 0), (10, 2), (240, 5), (30, 29)):
                with self.subTest(duration=duration, length=length, overlap=overlap):
                    windows = plan_chunks(duration, length, overlap)
                    self.assertEqual(windows[0][0], 0)
                    self.assertEqual(windows[-1][1], duration)
                    for (s1, e1), (s2, e2) in zip(windows, windows[1:]):
                        self.assertAlmostEqual(e1 - s2, overlap)  # exact overlap
                        self.assertLess(s1, s2)                    # always advances
                        self.assertLess(e1, duration)              # only last one reaches the end
                    for s, e in windows:
                        self.assertLessEqual(e - s, length + 1e-9)
                        self.assertGreater(e - s, 0)

    def test_invalid_arguments_rejected(self):
        for args in ((0, 10, 2), (10, 0, 0), (10, 10, 10), (10, 10, 12), (10, 10, -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                plan_chunks(*args)


class SplitIntoChunksTest(_TempDirTestCase):
    def test_chunk_files_match_reported_offsets(self):
        wav = make_tone_wav(self.tmp / "in.wav", seconds=25)
        chunks = split_into_chunks(wav, self.tmp / "chunks", chunk_length=10, overlap=2)

        self.assertEqual([c.index for c in chunks], [0, 1, 2])
        self.assertEqual([(c.start, c.end) for c in chunks], [(0, 10), (8, 18), (16, 25)])
        for c in chunks:
            channels, rate, bits, duration = wav_info(c.path)
            self.assertEqual((channels, rate, bits), (1, 16_000, 16))
            self.assertAlmostEqual(duration, c.duration, places=6)

    def test_chunks_reconstruct_original_audio_exactly(self):
        """Dropping each chunk's overlap and concatenating gives back the input,
        sample for sample: nothing lost or duplicated at the seams."""
        wav = make_tone_wav(self.tmp / "in.wav", seconds=23.37)
        chunks = split_into_chunks(wav, self.tmp / "chunks", chunk_length=7, overlap=1.5)

        rebuilt = bytearray()
        bytes_per_second = 16_000 * 2
        for prev, chunk in zip([None] + chunks, chunks):
            data = read_frames(chunk.path)
            if prev is not None:
                data = data[round((prev.end - chunk.start) * bytes_per_second):]
            rebuilt += data
        self.assertEqual(bytes(rebuilt), read_frames(wav))

    def test_short_audio_yields_single_chunk(self):
        wav = make_tone_wav(self.tmp / "in.wav", seconds=3)
        chunks = split_into_chunks(wav, self.tmp / "chunks", chunk_length=10, overlap=2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0].start, chunks[0].end), (0, 3))

    def test_non_wav_input_rejected(self):
        bogus = self.tmp / "in.wav"
        bogus.write_bytes(b"definitely not a wav file")
        with self.assertRaises(AudioProcessingError) as ctx:
            split_into_chunks(bogus, self.tmp / "chunks", 10, 2)
        self.assertEqual(ctx.exception.error_code, "invalid_audio")

    def test_missing_file_rejected(self):
        with self.assertRaises(AudioProcessingError) as ctx:
            split_into_chunks(self.tmp / "nope.wav", self.tmp / "chunks", 10, 2)
        self.assertEqual(ctx.exception.error_code, "file_not_found")


@unittest.skipUnless(FFMPEG_AVAILABLE, SKIP_REASON_NO_FFMPEG)
class ProbeDurationTest(_TempDirTestCase):
    def test_probes_common_formats(self):
        for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            with self.subTest(ext=ext):
                path = make_tone_file(self.tmp / f"tone{ext}", seconds=4)
                self.assertAlmostEqual(probe_duration_seconds(path), 4.0, delta=0.1)

    def test_corrupt_file_rejected_without_leaking_server_path(self):
        bogus = self.tmp / "bogus.mp3"
        bogus.write_bytes(b"not audio at all")
        with self.assertRaises(AudioProcessingError) as ctx:
            probe_duration_seconds(bogus)
        self.assertEqual(ctx.exception.error_code, "invalid_audio")
        self.assertNotIn(str(self.tmp), ctx.exception.message)

    def test_missing_file_rejected(self):
        with self.assertRaises(AudioProcessingError) as ctx:
            probe_duration_seconds(self.tmp / "missing.mp3")
        self.assertEqual(ctx.exception.error_code, "file_not_found")


@unittest.skipUnless(FFMPEG_AVAILABLE, SKIP_REASON_NO_FFMPEG)
class NormalizeToWavTest(_TempDirTestCase):
    def test_output_is_16k_mono_16bit_with_same_duration(self):
        for ext in (".mp3", ".flac", ".ogg", ".m4a", ".wav"):
            with self.subTest(ext=ext):
                src = make_tone_file(self.tmp / f"src{ext}", seconds=3, sample_rate=44_100, channels=2)
                out = normalize_to_wav(src, self.tmp / "out" / f"{ext[1:]}.wav")
                channels, rate, bits, duration = wav_info(out)
                self.assertEqual((channels, rate, bits), (1, 16_000, 16))
                self.assertAlmostEqual(duration, 3.0, delta=0.1)

    def test_video_container_keeps_only_audio(self):
        src = make_tone_file(self.tmp / "clip.mp4", seconds=2, with_video=True)
        out = normalize_to_wav(src, self.tmp / "clip.wav")
        channels, rate, _, duration = wav_info(out)
        self.assertEqual((channels, rate), (1, 16_000))
        self.assertAlmostEqual(duration, 2.0, delta=0.1)

    def test_custom_sample_rate(self):
        src = make_tone_file(self.tmp / "src.mp3", seconds=1)
        out = normalize_to_wav(src, self.tmp / "out.wav", sample_rate=8_000)
        self.assertEqual(wav_info(out)[1], 8_000)

    def test_corrupt_input_fails_and_leaves_no_partial_output(self):
        bogus = self.tmp / "bogus.mp3"
        bogus.write_bytes(b"not audio at all")
        out = self.tmp / "bogus.wav"
        with self.assertRaises(AudioProcessingError) as ctx:
            normalize_to_wav(bogus, out)
        self.assertEqual(ctx.exception.error_code, "normalization_failed")
        self.assertFalse(out.exists())

    def test_normalized_output_feeds_chunker(self):
        """The two halves of the module compose: normalize, then split."""
        src = make_tone_file(self.tmp / "long.mp3", seconds=12)
        wav = normalize_to_wav(src, self.tmp / "long.wav")
        chunks = split_into_chunks(wav, self.tmp / "chunks", chunk_length=5, overlap=1)
        self.assertEqual(len(chunks), 3)
        self.assertAlmostEqual(chunks[-1].end, 12.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
