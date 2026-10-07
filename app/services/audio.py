from __future__ import annotations

import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SAMPLE_RATE = 16_000
PROBE_TIMEOUT_SECONDS = 30
NORMALIZE_TIMEOUT_SECONDS = 30 * 60


class AudioProcessingError(Exception):
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
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    line = lines[-1] if lines else "unknown error"
    for p in paths:
        line = line.replace(str(p), p.name)
    return line


def probe_duration_seconds(path: str | Path) -> float:
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
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(dst),
        ],
        timeout=NORMALIZE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        dst.unlink(missing_ok=True)
        raise AudioProcessingError(
            "normalization_failed",
            f"Could not convert audio: {_last_stderr_line(result.stderr, src, dst)}",
        )
    return dst


def wav_duration_seconds(path: str | Path) -> float:
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


@dataclass(frozen=True)
class AudioChunk:
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
