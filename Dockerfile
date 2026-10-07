FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG INSTALL_WHISPER=false
COPY requirements.txt requirements-whisper.txt ./
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_WHISPER" = "true" ]; then \
         pip install torch --index-url https://download.pytorch.org/whl/cpu \
         && pip install -r requirements-whisper.txt; \
       fi

COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data /home/appuser/.cache \
    && chown appuser:appuser /data /home/appuser/.cache
USER appuser

ENV STORAGE_DIR=/data
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
