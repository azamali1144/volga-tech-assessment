"""Synthetic audio generators for tests.

No fixture files are checked in: every test builds the exact audio it needs,
so durations, sample rates and channel counts are known precisely.
"""

from __future__ import annotations

import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
SKIP_REASON_NO_FFMPEG = "ffmpeg/ffprobe not on PATH"


def make_tone_wav(
    path: str | Path,
    seconds: float,
    sample_rate: int = 16_000,
    channels: int = 1,
    frequency: float = 440.0,
) -> Path:
    """Write a 16-bit PCM sine-wave WAV using only the stdlib (no ffmpeg)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = round(seconds * sample_rate)
    amplitude = 0.3 * 32767
    frames = bytearray()
    for i in range(n_frames):
        sample = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
        frames += struct.pack("<h", sample) * channels
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames))
    return path


def make_tone_file(
    path: str | Path,
    seconds: float,
    sample_rate: int = 44_100,
    channels: int = 2,
    frequency: float = 440.0,
    with_video: bool = False,
) -> Path:
    """Write a sine tone in whatever format ``path``'s extension implies, via ffmpeg.

    Defaults to 44.1kHz stereo on purpose, so normalization has real work to do.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    if with_video:
        cmd += ["-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=64x48:rate=5"]
    cmd += [
        "-f", "lavfi",
        "-i", f"sine=frequency={frequency}:duration={seconds}:sample_rate={sample_rate}",
        "-ac", str(channels),
    ]
    if with_video:
        cmd += ["-shortest"]
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def wav_info(path: str | Path) -> tuple[int, int, int, float]:
    """Return ``(channels, sample_rate, sample_width_bits, duration_seconds)``."""
    with wave.open(str(path), "rb") as w:
        return (
            w.getnchannels(),
            w.getframerate(),
            w.getsampwidth() * 8,
            w.getnframes() / w.getframerate(),
        )


def read_frames(path: str | Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        return w.readframes(w.getnframes())
