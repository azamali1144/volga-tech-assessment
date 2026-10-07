from __future__ import annotations

import os
import re
import uuid
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, Protocol, runtime_checkable

COPY_BUFFER_BYTES = 1024 * 1024
_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$")


class StorageError(Exception):
    error_code = "storage_error"


class InvalidKeyError(StorageError, ValueError):
    error_code = "invalid_storage_key"


class ObjectNotFoundError(StorageError, FileNotFoundError):
    error_code = "audio_not_found"


class ObjectTooLargeError(StorageError):
    error_code = "file_too_large"

    def __init__(self, max_bytes: int) -> None:
        super().__init__(f"Object exceeds the {max_bytes}-byte limit")
        self.max_bytes = max_bytes


@runtime_checkable
class ObjectStorage(Protocol):
    def save(self, key: str, source: BinaryIO, max_bytes: int | None = None) -> int:
        ...

    def open(self, key: str) -> BinaryIO:
        ...

    def as_local_file(self, key: str) -> AbstractContextManager[Path]:
        ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None:
        ...

    def presigned_upload_url(self, key: str, expires_seconds: int = 900) -> str:
        ...


class LocalDiskStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        if not isinstance(key, str) or not _KEY_PATTERN.fullmatch(key) or ".." in key.split("/"):
            raise InvalidKeyError(f"Invalid storage key: {key!r}")
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise InvalidKeyError(f"Storage key escapes the storage root: {key!r}")
        return path

    def save(self, key: str, source: BinaryIO, max_bytes: int | None = None) -> int:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
        written = 0
        try:
            with open(tmp, "wb") as out:
                while chunk := source.read(COPY_BUFFER_BYTES):
                    written += len(chunk)
                    if max_bytes is not None and written > max_bytes:
                        raise ObjectTooLargeError(max_bytes)
                    out.write(chunk)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return written

    def open(self, key: str) -> BinaryIO:
        path = self._path_for(key)
        try:
            return open(path, "rb")
        except FileNotFoundError as exc:
            raise ObjectNotFoundError(f"No such object: {key}") from exc

    @contextmanager
    def as_local_file(self, key: str) -> Iterator[Path]:
        path = self._path_for(key)
        if not path.is_file():
            raise ObjectNotFoundError(f"No such object: {key}")
        yield path

    def exists(self, key: str) -> bool:
        return self._path_for(key).is_file()

    def delete(self, key: str) -> None:
        self._path_for(key).unlink(missing_ok=True)

    def presigned_upload_url(self, key: str, expires_seconds: int = 900) -> str:
        self._path_for(key)
        raise NotImplementedError(
            "Local disk storage can't issue presigned URLs; uploads go through "
            "the API. An S3-backed ObjectStorage would return a presigned PUT URL."
        )
