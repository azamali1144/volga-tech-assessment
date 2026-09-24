"""Audio processing: duration probing, format normalization, and chunking.

ffmpeg/ffprobe are invoked as subprocesses (argument lists, never a shell
string), so user-supplied filenames can't inject shell commands. Any failure
is raised as ``AudioProcessingError`` with a stable ``error_code`` that the
API and worker can surface without leaking raw ffmpeg output to callers.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
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



def wav_duration_seconds(path: str | Path) -> float:
    """Duration of a PCM WAV from its header: exact and instant, no subprocess.

    For files already normalized by ``normalize_to_wav``; anything else should
    go through ``probe_duration_seconds``.
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except FileNotFoundError as exc:
        raise AudioProcessingError("file_not_found", f"No such file: {path.name}") from exc
    except (wave.Error, EOFError, OSError, ZeroDivisionError) as exc:
        raise AudioProcessingError(
            "invalid_audio", f"Not a readable WAV file: {path.name}"
        ) from exc

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioChunk:
    """One slice of a longer recording.

    ``start``/``end`` are positions in the *original* file, in seconds. The
    merge step uses ``start`` to shift chunk-local timestamps back onto the
    global timeline.
    """

    index: int
    path: Path
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_chunks(
    duration: float, chunk_length: float, overlap: float
) -> list[tuple[float, float]]:
    """Compute ``(start, end)`` windows covering ``[0, duration]``.

    Consecutive windows overlap by ``overlap`` seconds so a word spoken right
    at a boundary is fully contained in at least one chunk. Windows advance by
    ``chunk_length - overlap``; the last window is clamped to ``duration``.
    Pure arithmetic, no I/O — this is what the overlap/offset tests target.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if not 0 <= overlap < chunk_length:
        raise ValueError("overlap must be >= 0 and smaller than chunk_length")

    step = chunk_length - overlap
    windows: list[tuple[float, float]] = []
    i = 0
    while True:
        # Multiply rather than accumulate, so float error doesn't drift
        # across hundreds of chunks in a multi-hour file.
        start = i * step
        end = min(start + chunk_length, duration)
        windows.append((start, end))
        if end >= duration:
            return windows
        i += 1


def split_into_chunks(
    wav_path: str | Path,
    out_dir: str | Path,
    chunk_length: float,
    overlap: float,
) -> list[AudioChunk]:
    """Split a normalized WAV into overlapping chunk files.

    Expects the output of ``normalize_to_wav`` (PCM WAV). Slicing is done on
    raw frames with the stdlib ``wave`` module instead of spawning ffmpeg per
    chunk: it's sample-exact (chunk offsets are exactly what the merge step
    assumes), needs no re-encoding, and costs one pass over the file.
    """
    wav_path, out_dir = Path(wav_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        src = wave.open(str(wav_path), "rb")
    except FileNotFoundError as exc:
        raise AudioProcessingError(
            "file_not_found", f"No such file: {wav_path.name}"
        ) from exc
    except (wave.Error, EOFError) as exc:
        raise AudioProcessingError(
            "invalid_audio", f"Not a PCM WAV file: {wav_path.name}"
        ) from exc

    with src:
        params = src.getparams()
        rate = params.framerate
        total_frames = params.nframes
        if total_frames == 0:
            raise AudioProcessingError("invalid_audio", "Audio file has no duration")

        chunks: list[AudioChunk] = []
        windows = plan_chunks(total_frames / rate, chunk_length, overlap)
        for index, (start_s, end_s) in enumerate(windows):
            # Snap to whole frames; report the snapped times so offsets used by
            # the merge step match the audio actually in each chunk file.
            start_f = round(start_s * rate)
            end_f = min(round(end_s * rate), total_frames)
            src.setpos(start_f)
            frames = src.readframes(end_f - start_f)

            chunk_path = out_dir / f"{wav_path.stem}.chunk{index:04d}.wav"
            with wave.open(str(chunk_path), "wb") as dst:
                dst.setparams(params)
                dst.writeframes(frames)

            chunks.append(
                AudioChunk(
                    index=index,
                    path=chunk_path,
                    start=start_f / rate,
                    end=end_f / rate,
                )
            )
    return chunks
