import io
import os
import tempfile
import unittest
from pathlib import Path

from app.storage_backend import (
    InvalidKeyError,
    LocalDiskStorage,
    ObjectNotFoundError,
    ObjectStorage,
    ObjectTooLargeError,
)


class LocalDiskStorageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "objects"
        self.storage = LocalDiskStorage(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_satisfies_object_storage_protocol(self):
        self.assertIsInstance(self.storage, ObjectStorage)

    def test_save_open_round_trip(self):
        data = os.urandom(3 * 1024 * 1024 + 7)  # spans several copy buffers
        written = self.storage.save("audio/job1.mp3", io.BytesIO(data))
        self.assertEqual(written, len(data))
        self.assertTrue(self.storage.exists("audio/job1.mp3"))
        with self.storage.open("audio/job1.mp3") as f:
            self.assertEqual(f.read(), data)

    def test_save_overwrites_existing_object(self):
        self.storage.save("k.bin", io.BytesIO(b"old"))
        self.storage.save("k.bin", io.BytesIO(b"new"))
        with self.storage.open("k.bin") as f:
            self.assertEqual(f.read(), b"new")

    def test_as_local_file_yields_real_path_inside_root(self):
        self.storage.save("audio/a.wav", io.BytesIO(b"abc"))
        with self.storage.as_local_file("audio/a.wav") as path:
            self.assertTrue(path.is_file())
            self.assertTrue(path.is_relative_to(self.root.resolve()))
            self.assertEqual(path.read_bytes(), b"abc")

    def test_size_cap_rejects_and_leaves_nothing_behind(self):
        with self.assertRaises(ObjectTooLargeError) as ctx:
            self.storage.save("audio/big.mp3", io.BytesIO(b"x" * 2001), max_bytes=2000)
        self.assertEqual(ctx.exception.max_bytes, 2000)
        self.assertFalse(self.storage.exists("audio/big.mp3"))
        self.assertEqual(os.listdir(self.root / "audio"), [])  # no .part files

    def test_size_cap_allows_exact_limit(self):
        self.assertEqual(
            self.storage.save("a.bin", io.BytesIO(b"x" * 2000), max_bytes=2000), 2000
        )

    def test_oversized_upload_does_not_clobber_existing_object(self):
        self.storage.save("a.bin", io.BytesIO(b"original"))
        with self.assertRaises(ObjectTooLargeError):
            self.storage.save("a.bin", io.BytesIO(b"x" * 100), max_bytes=10)
        with self.storage.open("a.bin") as f:
            self.assertEqual(f.read(), b"original")

    def test_path_traversal_and_malformed_keys_rejected(self):
        bad_keys = [
            "../evil", "audio/../../evil", "audio/..", "/etc/passwd",
            "C:/Windows/x", "a\\b", "", ".hidden", "audio//x", "audio/x/",
            "a b", "audio/%2e%2e/x", None,
        ]
        for key in bad_keys:
            with self.subTest(key=key):
                with self.assertRaises(InvalidKeyError):
                    self.storage.save(key, io.BytesIO(b"x"))
                with self.assertRaises(InvalidKeyError):
                    self.storage.exists(key)
        # Nothing was written anywhere outside the storage root.
        outside = [p for p in Path(self._tmp.name).rglob("*") if not p.is_relative_to(self.root)]
        self.assertEqual(outside, [])

    def test_missing_object(self):
        with self.assertRaises(ObjectNotFoundError):
            self.storage.open("audio/nope.mp3")
        with self.assertRaises(ObjectNotFoundError):
            with self.storage.as_local_file("audio/nope.mp3"):
                pass
        self.assertFalse(self.storage.exists("audio/nope.mp3"))

    def test_delete_is_idempotent(self):
        self.storage.save("a.bin", io.BytesIO(b"x"))
        self.storage.delete("a.bin")
        self.storage.delete("a.bin")
        self.assertFalse(self.storage.exists("a.bin"))

    def test_presigned_upload_is_explicitly_unsupported_locally(self):
        with self.assertRaises(NotImplementedError):
            self.storage.presigned_upload_url("audio/a.mp3")
        with self.assertRaises(InvalidKeyError):
            self.storage.presigned_upload_url("../evil")


if __name__ == "__main__":
    unittest.main()
