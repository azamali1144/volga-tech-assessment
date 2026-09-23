import os
import unittest
from pathlib import Path
from unittest import mock

from app.config import Settings


class SettingsDefaultsTest(unittest.TestCase):
    def test_defaults_are_valid_and_paths_derive_from_storage_dir(self):
        s = Settings(storage_dir=Path("somewhere"))
        self.assertEqual(s.transcription_engine, "mock")
        self.assertIn("dev-local-key", s.api_keys)
        self.assertEqual(s.audio_dir, Path("somewhere") / "audio")
        self.assertEqual(s.transcript_dir, Path("somewhere") / "transcripts")
        self.assertEqual(s.dead_letter_dir, Path("somewhere") / "dead_letter")
        self.assertEqual(s.db_path, Path("somewhere") / "jobs.db")

    def test_explicit_sub_path_is_not_overridden(self):
        s = Settings(storage_dir=Path("a"), db_path=Path("b/custom.db"))
        self.assertEqual(s.db_path, Path("b/custom.db"))


class SettingsValidationTest(unittest.TestCase):
    def test_overlap_must_be_smaller_than_chunk_length(self):
        with self.assertRaises(ValueError):
            Settings(chunk_length_seconds=10, chunk_overlap_seconds=10)

    def test_unknown_engine_rejected(self):
        with self.assertRaises(ValueError):
            Settings(transcription_engine="nope")

    def test_empty_api_keys_rejected(self):
        with self.assertRaises(ValueError):
            Settings(api_keys=frozenset())


class SettingsFromEnvTest(unittest.TestCase):
    def test_reads_and_parses_env_vars(self):
        env = {
            "API_KEYS": "k1, k2 ,",
            "MAX_RETRIES": "5",
            "CHUNK_OVERLAP_SECONDS": "2.5",
            "TRANSCRIPTION_ENGINE": "WHISPER",
            "ALLOWED_EXTENSIONS": "wav,.MP3",
            "STORAGE_DIR": "data",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            s = Settings.from_env()
        self.assertEqual(s.api_keys, frozenset({"k1", "k2"}))
        self.assertEqual(s.max_retries, 5)
        self.assertEqual(s.chunk_overlap_seconds, 2.5)
        self.assertEqual(s.transcription_engine, "whisper")
        self.assertEqual(s.allowed_extensions, frozenset({".wav", ".mp3"}))
        self.assertEqual(s.db_path, Path("data") / "jobs.db")

    def test_bad_integer_gives_clear_error(self):
        with mock.patch.dict(os.environ, {"MAX_RETRIES": "three"}, clear=True):
            with self.assertRaisesRegex(ValueError, "MAX_RETRIES"):
                Settings.from_env()

    def test_empty_env_uses_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Settings.from_env(), Settings())


if __name__ == "__main__":
    unittest.main()
