"""Audio processing: duration probing and format normalization via ffmpeg.

ffmpeg/ffprobe are invoked as subprocesses (argument lists, never a shell
string), so user-supplied filenames can't inject shell commands. Any failure
is raised as ``AudioProcessingError`` with a stable ``error_code`` that the
API and worker can surface without leaking raw ffmpeg output to callers.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

DEFAULT_SAMPLE_RATE = 16_000
# Generous upper bounds so a wedged ffmpeg process can't hang a worker forever.
PROBE_TIMEOUT_SECONDS = 30
NORMALIZE_TIMEOUT_SECONDS = 30 * 60


class AudioProcessingError(Exception):
    """Raised when an input file can't be probed, decoded, or converted."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise AudioProcessingError(
            "ffmpeg_not_found",
            f"'{name}' was not found on PATH; install ffmpeg to process audio",
        )
    return path


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError(
            "ffmpeg_timeout", f"{Path(cmd[0]).name} timed out after {timeout}s"
        ) from exc


def _last_stderr_line(stderr: str, *paths: Path) -> str:
    """Last meaningful ffmpeg error line, with server-side paths redacted."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    line = lines[-1] if lines else "unknown error"
    for p in paths:
        line = line.replace(str(p), p.name)
    return line


def probe_duration_seconds(path: str | Path) -> float:
    """Return the duration of an audio/video file in seconds, via ffprobe."""
    path = Path(path)
    if not path.is_file():
        raise AudioProcessingError("file_not_found", f"No such file: {path.name}")

    ffprobe = _require_binary("ffprobe")
    result = _run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ],
        timeout=PROBE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise AudioProcessingError(
            "invalid_audio",
            f"Could not read audio file: {_last_stderr_line(result.stderr, path)}",
        )

    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (ValueError, KeyError, TypeError) as exc:
        raise AudioProcessingError(
            "invalid_audio", "Could not determine audio duration"
        ) from exc

    if duration <= 0:
        raise AudioProcessingError("invalid_audio", "Audio file has no duration")
    return duration


def normalize_to_wav(
    src: str | Path,
    dst: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    """Convert any ffmpeg-readable input to mono 16-bit PCM WAV at ``sample_rate``.

    This is the single format boundary for the pipeline: everything after this
    step (chunking, transcription) only ever sees one well-known format.
    Video inputs (e.g. .mp4) are handled by dropping the video stream.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise AudioProcessingError("file_not_found", f"No such file: {src.name}")

    ffmpeg = _require_binary("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(src),
            "-vn",                    # drop any video stream
            "-ac", "1",               # mono
            "-ar", str(sample_rate),  # resample
            "-c:a", "pcm_s16le",      # 16-bit PCM
            str(dst),
        ],
        timeout=NORMALIZE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        dst.unlink(missing_ok=True)  # don't leave a half-written file behind
        raise AudioProcessingError(
            "normalization_failed",
            f"Could not convert audio: {_last_stderr_line(result.stderr, src, dst)}",
        )
    return dst
