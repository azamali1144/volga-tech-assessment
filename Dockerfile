# Audio transcription service.
#
#   docker build -t volga-transcription .                                # mock engine, small image
#   docker build -t volga-transcription --build-arg INSTALL_WHISPER=true .  # + Whisper (CPU)
#   docker run -p 8000:8000 -v transcription-data:/data volga-transcription

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg does all format decoding/normalization (see app/audio.py).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before code, so editing app/ doesn't invalidate this layer.
# With INSTALL_WHISPER=true, PyTorch comes from the CPU-only wheel index: the
# default wheel bundles CUDA and is several GB larger.
ARG INSTALL_WHISPER=false
COPY requirements.txt requirements-whisper.txt ./
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_WHISPER" = "true" ]; then \
         pip install torch --index-url https://download.pytorch.org/whl/cpu \
         && pip install -r requirements-whisper.txt; \
       fi

COPY app ./app

# Run unprivileged; /data holds everything stateful (audio, transcripts,
# SQLite db, dead-letter records) so it can be a mounted volume.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown appuser:appuser /data
USER appuser

ENV STORAGE_DIR=/data
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
