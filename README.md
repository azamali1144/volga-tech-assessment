# volga-tech-assessment: Audio Transcription Pipeline

A transcription service meant to be called by other systems. A caller uploads an audio file and gets a job id back immediately. A background worker transcribes the file with [Whisper](https://github.com/openai/whisper), and the caller polls for the text with per-segment timestamps.

The brief asked for engineering decisions rather than a trained model, so most of this README explains *why* the service is built the way it is. Every claim below points at the code that implements it and, where possible, the test that proves it.

## Contents

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [API](#api)
- [Design decisions](#design-decisions)
- [What's real vs. mocked](#whats-real-vs-mocked)
- [Scaling to production](#scaling-to-production)
- [Testing](#testing)
- [Configuration](#configuration)
- [Known limitations](#known-limitations)

## Quick start

**Prerequisites:** Python 3.13 (the version it's tested on) and [ffmpeg](https://ffmpeg.org/) on your `PATH`. On Windows, `winget install Gyan.FFmpeg`; on macOS, `brew install ffmpeg`; on Debian/Ubuntu, `apt install ffmpeg`.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**macOS / Linux:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

This starts the service with the **mock** engine, which returns deterministic fake transcripts and needs no model download. For real transcription:

```bash
pip install -r requirements-whisper.txt     # adds openai-whisper + PyTorch (~2 GB)
TRANSCRIPTION_ENGINE=whisper uvicorn app.main:app              # macOS / Linux
$env:TRANSCRIPTION_ENGINE='whisper'; uvicorn app.main:app      # PowerShell
```

To configure the service, copy `.env.example` to `.env`, edit it, and run `uvicorn app.main:app --env-file .env`.

Interactive API docs are at <http://localhost:8000/docs>. A full round trip (the default API key locally is `dev-local-key`):

```bash
curl -s -X POST http://localhost:8000/api/v1/transcriptions \
  -H "X-API-Key: dev-local-key" -F "file=@call.mp3"
# {"job_id":"3f1c…","status":"queued","status_url":"/api/v1/transcriptions/3f1c…"}

curl -s http://localhost:8000/api/v1/transcriptions/3f1c… -H "X-API-Key: dev-local-key"
# {"status":"completed", …, "transcript":{"text":"…","segments":[{"start":0.0,"end":2.1,"text":"…"}, …]}}
```

**Docker:**

```bash
docker build -t volga-transcription .                                   # mock engine, small image
docker build -t volga-transcription --build-arg INSTALL_WHISPER=true .  # with Whisper (CPU)
docker run -p 8000:8000 -v transcription-data:/data volga-transcription
```

**Tests** use the mock engine and synthetic audio, so they need no model and no fixture files:

```bash
pip install -r requirements-dev.txt     # adds the HTTP client used by the API tests
python -m unittest discover -s tests -p "test_*.py" -v
```

## Architecture

```
  client ──POST /api/v1/transcriptions──▶ ┌──────────────────────────────────────┐
                                          │ FastAPI app  (app/main.py)           │
                                          │  1. auth + rate limit (one dependency)│
                                          │  2. check extension                  │
                                          │  3. stream upload to storage, cap size│──▶ ObjectStorage
                                          │  4. ffprobe header → 422 if corrupt   │    (LocalDiskStorage;
                                          │  5. insert job row (queued)          │     S3 in prod)
                                          │  6. enqueue job id                   │
  client ◀──202 {job_id, status_url}───── │  7. return 202                       │──▶ JobStore
                                          └──────────────────┬───────────────────┘    (SQLite;
                                                             │ job id                 Postgres in prod)
                                                             ▼
                                          ┌──────────────────────────────────────┐
                                          │ JobQueue (InMemoryQueue; SQS in prod)│
                                          └──────────────────┬───────────────────┘
                                                             │ dequeue
                                                             ▼
                                          ┌──────────────────────────────────────┐
                                          │ Worker  (app/worker.py)              │
                                          │  claim job: queued → processing      │
                                          │  TranscriptionPipeline.run():        │
                                          │    normalize to 16 kHz mono WAV      │
                                          │    ≤ 5 min → transcribe once         │
                                          │    > 5 min → split into overlapping  │
                                          │              chunks → transcribe     │
                                          │              (bounded concurrency)   │
                                          │              → merge_chunk_results   │
                                          │  save transcript (versioned)         │
                                          │  mark completed                      │
                                          │  on failure: retry with backoff, or  │
                                          │  mark failed + dead-letter record    │
                                          └──────────────────────────────────────┘

  client ──GET /api/v1/transcriptions/{id}──▶ status, plus transcript once completed
```

Each box on the right is a small interface with one demo implementation: `ObjectStorage`, `JobQueue`, `JobStore` and `TranscriptionEngine`. The production version of each can be swapped in without changing the code that calls it (see [What's real vs. mocked](#whats-real-vs-mocked)).

**Code layout**

| Module | Responsibility | Pattern |
|---|---|---|
| `app/main.py` | Routes, auth/rate-limit dependency, error handlers, startup/shutdown | Dependency injection |
| `app/worker.py` | `TranscriptionPipeline` (pure: audio in, transcript out), plus `Worker` (queue loop, retries, dead-letter) | Producer/consumer |
| `app/audio.py` | ffmpeg normalization, duration probing, chunk planning and splitting | |
| `app/transcription_engine.py` | `WhisperEngine`, `MockEngine`, `merge_chunk_results` | Strategy |
| `app/store.py` | SQLite job and transcript persistence, status state machine | Repository |
| `app/storage_backend.py` | Object storage interface and local-disk implementation | Adapter |
| `app/queue_backend.py` | Job queue interface and in-memory implementation | Adapter |
| `app/rate_limit.py` | Per-key sliding-window rate limiter | |
| `app/schemas.py` | Pydantic request/response models, which also generate `/docs` | |
| `app/config.py` | Every setting, read from environment variables and validated at startup | |
| `app/logging_config.py` | JSON structured logging | |

## API

Routes under `/api/v1` require an `X-API-Key` header.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/transcriptions` | Upload audio (multipart field `file`). Returns `202` with `{job_id, status, status_url}`. |
| `GET` | `/api/v1/transcriptions/{job_id}` | Status and, once `completed`, the transcript: `text`, `language`, `duration`, `version`, `segments[{id, start, end, text}]`. |
| `GET` | `/api/v1/transcriptions?limit=&offset=&status=` | The caller's jobs, newest first, paginated (`next_offset`). Summaries only. |
| `GET` | `/healthz` | Unauthenticated health probe. `200` or `503`, plus queue depth. |

A job moves through `queued → processing → completed`. A failed attempt goes `processing → retrying → processing`, and a job that fails for good ends in `failed`. While a job is retrying, and after it has failed, its `error: {code, message}` explains why.

**Errors.** Every non-2xx response has the same body, `{"error_code": "...", "detail": "..."}`:

| Status | `error_code` | When |
|---|---|---|
| 400 | `unsupported_format` | Extension isn't one of `.wav .mp3 .m4a .flac .ogg .mp4` |
| 401 | `unauthorized` | Missing or unknown API key |
| 404 | `job_not_found` | Unknown job id, or a job owned by another caller (the two are indistinguishable on purpose) |
| 413 | `file_too_large` | Upload exceeds `MAX_UPLOAD_BYTES` |
| 422 | `invalid_audio` | The file isn't decodable audio (checked at upload) |
| 422 | `validation_error` | Bad query or form input |
| 429 | `rate_limited` | Over the per-key limit; the response includes `Retry-After` |
| 500 | `internal_error` | Unexpected failure; full details are logged, and nothing internal is returned |

Successful authenticated responses carry `X-RateLimit-Limit` and `X-RateLimit-Remaining` headers.

## Design decisions

### Accepting and validating audio

`POST /api/v1/transcriptions` takes `multipart/form-data`. Validation happens in layers, cheapest first, so bad input is rejected as early and as cheaply as possible:

1. **Extension allow-list.** This is only a first filter, because an extension can lie.
2. **Size, checked twice:**
   - **Before the body is read.** FastAPI reads the whole multipart upload before a route runs, so a size check inside the route alone would still accept a huge body first. A middleware therefore rejects oversized uploads from their `Content-Length` header up front.
   - **While saving.** The same limit is enforced while the file is streamed to storage in 1 MiB pieces. This catches uploads sent without `Content-Length` (chunked encoding), and a rejected upload leaves no partial file behind.
3. **Decodability.** `ffprobe` reads the file's header (it doesn't decode the audio), so a corrupt or mislabeled file gets an immediate `422 invalid_audio`. Without this check the file would be queued and fail minutes later.

The storage key is derived from the server-generated job id, never from the client's filename, and keys are validated against path traversal (`app/storage_backend.py`, `tests/test_storage_backend.py`).

### Transcribing with timestamps

`WhisperEngine` (`app/transcription_engine.py`) wraps `whisper.transcribe(..., word_timestamps=True)`. It converts Whisper's output into `Segment(id, start, end, text, words)`, with times in seconds rounded to 10 ms and empty segments dropped. Each segment keeps its per-word timings, which the chunk merge relies on. The model loads lazily, on first use rather than at import, so the API starts instantly and a mock-engine deployment never installs PyTorch.

This has been run for real, not only against a fake: Whisper `base` on CPU, with speech generated by Windows text-to-speech. A 73.7 s meeting took about 7 s in one pass, and every word came out right, including "12%", "back-end" and "take-home".

Whisper's `transcribe()` isn't safe to run twice at once on the same model: it temporarily attaches hooks to the model for decoding, and concurrent calls would interfere. Calls to one engine instance are therefore serialized with a lock. Whisper throughput scales with more worker processes, not more threads.

### Handling different audio formats

Every file is normalized once to **16 kHz mono 16-bit PCM WAV** with ffmpeg (`normalize_to_wav`) before anything else touches it. Everything downstream (chunking, the engine) deals with exactly one format. ffmpeg already handles the full codec matrix, including the audio track of `.mp4` files, with no extra Python dependencies. It is always invoked with an argument list, never a shell string, and with a timeout.

### Long audio: chunking and merging

Files longer than `CHUNK_THRESHOLD_SECONDS` (default 5 min) are split into windows of `CHUNK_LENGTH_SECONDS` (240 s) that overlap by `CHUNK_OVERLAP_SECONDS` (5 s).

- **Cutting.** The chunk plan (`plan_chunks`) is pure arithmetic. The actual cutting uses the stdlib `wave` module on the normalized WAV, so it is sample-exact: each chunk's reported offset is exactly the audio in its file. A test rejoins the chunks without their overlaps and gets the original file back byte for byte.
- **Transcribing.** Chunks are transcribed in parallel, with `asyncio.Semaphore(MAX_CONCURRENT_CHUNK_TRANSCRIPTIONS)` limiting how many run at once. If one chunk fails, no new chunks are started.

**The merge (`merge_chunk_results`).** Speech in an overlap is transcribed twice, once by each chunk, so the merge has to keep exactly one copy of every **word**:

1. **Shift** every word by its chunk's start time, putting it on the file's timeline.
2. **Cut each overlap at its midpoint**, rather than where the earlier chunk ends. Audio right at a chunk's edge has the least context, so that is where words get clipped or misheard. With the cut at the midpoint, everything kept is at least `overlap/2` seconds (2.5 s by default) from the edge of the chunk it came from.
3. **Keep each word by its centre point.** A word is kept only if:
   - its centre is before this chunk's cut, and
   - its centre is at or after the end of the transcript assembled so far.
4. **Clamp** so timestamps never overlap or go backwards.
5. **Rebuild segments** from the kept words, preserving each chunk's segment boundaries. A sentence split by the cut (its start kept by one chunk, its end by the next) is joined back into one segment.

*What if a word falls exactly on the boundary?*
- **A word centred exactly on the cut** belongs to the later chunk, because each chunk keeps only words centred *before* its cut. It is kept once, never twice and never dropped.
- **A word clipped at a chunk's edge** is replaced by the neighbouring chunk's complete version. A word lasts well under a second and the cut is 2.5 s from either edge, so any word near the cut was heard in full by both chunks.
- **The same word timed slightly differently by each chunk** (word-timing jitter) is kept once: a chunk's first word is dropped if it has the same text as the previous kept word *and* overlaps it in time. A genuinely repeated word ("very, very") starts after the first ends, so it is kept.

Each of these cases has its own test in `tests/test_transcription_engine.py`, built from real Whisper output.

**Why word level, and the evidence.** The first version merged whole segments. It passed every mock-engine test, then broke on real speech: Whisper's segments are 4–7 s long, about as long as the overlap. With 20 s chunks on the 73.7 s meeting recording, the segment merge:
- turned "previous **quarter, driven mostly by**" into "previous **court.**", because the word "quarter" was cut off at a chunk edge and "driven mostly by" was dropped
- repeated "home assessment" at another seam

The word-level merge reproduces **all 181 words of the single-pass transcript**, with 20 s chunks and with 30 s chunks. The only differences are Whisper choosing to write "second"/"third" as "2"/"3" when it has less context, which is the model, not the merge.

On the mock engine, when chunk offsets line up with its segments, the merged output is **identical** to a single pass. `test_merge_chunk_results_dedupes_overlap_and_spans_full_duration` checks this on 25 s, 123 s and 600 s files. When they don't line up, no gap exceeds half a word.

Engines that don't report word timings fall back to the same rules applied to whole segments (see [Known limitations](#known-limitations)).

### Concurrent uploads

The upload endpoint does only fast work: validate, stream to storage, insert one row, enqueue one id. It returns `202` in milliseconds. Transcription happens on workers pulling from a queue, fully decoupled from the request. N simultaneous uploads become N quick requests plus N queued jobs, which workers drain at whatever rate they can sustain. The alternative, N connections each held open for the length of a transcription, doesn't scale.

- **The API process holds no state that ties a job to it.** Jobs live in the database and audio in object storage, so any instance can serve any request.
- **Blocking work never runs on the event loop.** ffmpeg, WAV splitting and the engine all run through `asyncio.to_thread`, so the API stays responsive while transcription runs (a test checks this).
- **Several workers can share one queue safely.** Claiming a job is an atomic update of the form `UPDATE … WHERE status IN (…)`, so a job id delivered twice is processed once. A test sends the same id three times to three workers and checks that the engine runs once.

### Storing audio and transcripts

- **Audio** lives in object storage, keyed by job id (`LocalDiskStorage` here, S3 in production).
- **Job metadata** is one row per job in `jobs`: `id, user_id, original_filename, file_path, duration_seconds, status, retry_count, error_code, error_message, failed_at, language, trans_version, created_at, updated_at`. The schema uses only portable types, so it maps directly onto PostgreSQL.
- **Transcripts** live in a separate, **versioned** `transcripts` table; `jobs.trans_version` points at the latest version, and reprocessing adds a new one. Transcripts up to `INLINE_TRANSCRIPT_MAX_CHARS` (20k characters of JSON) are stored inline in the row. Larger ones, such as a two-hour meeting, go to a file, and only the path is stored. The table stays small and fast for the common case.
- **Integrity** is enforced by the database itself:
  - a transcript must belong to an existing job (foreign key)
  - a row stores its transcript inline or as a file path, never both and never neither (CHECK constraint)
  - a version number can't repeat within a job (UNIQUE constraint)
- **Consistency.** Files are written to a temporary name and then renamed into place, so a reader never sees half a file. If a transcript's database write fails, its file is deleted.
- **No raw API keys in the database.** Jobs are owned by a SHA-256-derived caller id, so a leaked database doesn't leak working credentials.
- **Encryption at rest** isn't implemented, since there's no KMS locally. The storage and store interfaces are where it would go: S3 server-side encryption for audio, and database or column encryption for transcript text.

### Retrying and recovering failed transcriptions

Failures are classified before anything is retried (`Worker._handle_failure`):

| Kind | Examples | Handling |
|---|---|---|
| **Permanent** | `invalid_audio`, `normalization_failed`, `audio_not_found`, `engine_unavailable`, `ffmpeg_not_found` | Failed immediately. Retrying can't fix bad input or a broken deployment |
| **Transient** | Engine crashes, timeouts, unexpected exceptions | Marked `retrying` and re-enqueued after `RETRY_BACKOFF_BASE_SECONDS × 2^(n−1)` (2 s, 4 s, 8 s, …) |

- **Backoff doesn't block the worker.** The retry is a delayed re-enqueue, so the worker carries straight on with other jobs.
- **`MAX_RETRIES` counts retries, not attempts.** A job gets at most `1 + MAX_RETRIES` attempts (4 by default).
- **When a job fails for good:** its status becomes `failed` with `failed_at` set, and a dead-letter record is written to `DEAD_LETTER_DIR/<job_id>.json` for manual review. The record holds the job, the error code and message, the number of attempts, the reason (`retries_exhausted` or `non_retryable`) and the time.

**Startup recovery.** The in-memory queue doesn't survive a restart, but the database does. At startup the service re-queues every job left unfinished. A job found in `processing` was interrupted mid-attempt, and that counts as a failed attempt. A file that crashes the service every time therefore still ends up dead-lettered instead of looping forever.

Proven by `tests/test_worker.py`:
- `test_transient_failures_then_success`: two failures, then `completed` with `retry_count == 2`.
- `test_permanent_failure_goes_to_dead_letter`: always failing, so `failed` after 4 attempts, with exactly one dead-letter record.
- Other tests cover non-retryable errors, the backoff delays, duplicate delivery, and the worker surviving an infrastructure error.

### Exposing it as an API

- **Auth and rate limiting live in one dependency attached to the router**, so a new route can't accidentally skip either.
  - **Keys** are compared in constant time.
  - **The rate limiter** is a per-key **sliding window**. Unlike a fixed window, it can't be gamed by bursting at a window boundary to get twice the limit, and rejected requests don't extend the caller's block.
- **Pydantic models** for every request and response generate the OpenAPI docs at `/docs`. The response models are separate from the internal types, so storage keys, owner ids and raw internal error text never reach a response. An `internal_error` is shown to callers as a generic message.
- **Versioned** under `/api/v1`.
- **Observability:**
  - **Logs** are JSON, one object per line (`LOG_FORMAT=text` for local reading). `job_id`, `attempt`, `error_code` and `worker` are queryable fields.
  - **`/healthz`** returns `503` if the database is unreachable or no worker is running, and reports queue depth for autoscaling.

## What's real vs. mocked

The brief allows mocked infrastructure. The point is that each seam is clean, so here is exactly what's simplified and what swapping in the real thing involves:

| Concern | This repo | Production | Swap |
|---|---|---|---|
| Object storage | `LocalDiskStorage` | S3 / GCS / Azure Blob | New class with the same six `ObjectStorage` methods (`app/storage_backend.py`) |
| Job queue | `InMemoryQueue` (`asyncio.Queue`) | SQS / RabbitMQ | New class with `enqueue` / `enqueue_after` / `dequeue` / `size`; delays map to SQS `DelaySeconds` |
| Database | SQLite (WAL mode) | PostgreSQL | Only `app/store.py` touches the database; the schema is portable |
| Rate limiter | In-process sliding window | Same algorithm in Redis | A sorted set per key, shared across instances |
| Transcription | `mock` for tests/dev, `whisper` for real | Whisper (larger model, GPU), or a hosted ASR API | `TRANSCRIPTION_ENGINE` env var; no code change |
| Workers | Asyncio tasks inside the API process | Separate worker containers | `Worker` has no dependency on the API; it only needs a store, queue and storage |

## Scaling to production

In priority order:

1. **Durable queue and Postgres.** These are the two places the demo trades durability for zero setup. Startup recovery narrows the gap: queued work survives a restart because the database is the source of truth. A real queue is still needed for multiple processes.
2. **Workers as separate, autoscaled deployments**, scaled on queue depth (already exposed by `/healthz`). Whisper throughput scales with processes and GPUs, not threads.
3. **Redis-backed rate limiting**, so the limit is shared across API instances.
4. **Direct-to-S3 uploads via presigned URLs** (`ObjectStorage.presigned_upload_url` is the placeholder), so large files never pass through the API.
5. **Real identity** (OAuth2/JWT with tenant ids) instead of static API keys.
6. **Encryption at rest** for audio and transcripts.
7. **Per-chunk resumable retries** for long files. Today a retry re-transcribes the whole file.

## Testing

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -p "test_*.py" -v    # 175 tests, ~15 s
```

The tests generate their own audio (sine tones, via the stdlib or ffmpeg) and use `MockEngine`, so they need no model download and no fixture files.

| File | Tests | Covers |
|---|---:|---|
| `test_audio.py` | 19 | Duration probing and normalization across formats; chunk maths invariants; sample-exact reconstruction from chunks |
| `test_transcription_engine.py` | 35 | Mock determinism; Whisper output and word-timing mapping, lazy single model load and missing-dependency error (with a fake `whisper` module); word-level merge seam cases taken from real Whisper output; segment-level fallback; single-pass equivalence |
| `test_store.py` | 27 | Status state machine, including illegal transitions and racing claims; inline vs file transcripts; versioning; integrity constraints; per-user listing and pagination |
| `test_api.py` | 25 | The real app through FastAPI's `TestClient`: upload → poll → completed round trip, chunked long audio, every supported format, listing and pagination, per-caller isolation, every error code in the table above, rate limiting, generic 500s, `/healthz`, and restart recovery |
| `test_worker.py` | 26 | Pipeline (single-pass and chunked paths, concurrency limit, temp-file cleanup); worker retry, backoff, dead-letter, duplicate delivery, infrastructure errors, prompt shutdown |
| `test_storage_backend.py` | 11 | Round trips, size cap with no partial files, path traversal |
| `test_rate_limit.py` | 10 | Sliding window, boundary burst, `Retry-After`, memory sweep, thread safety |
| `test_queue_backend.py` | 7 | FIFO order, timeouts, delayed delivery, shutdown |
| `test_config.py` / `test_logging_config.py` | 8 / 8 | Env parsing and validation; JSON log output |

Tests that need ffmpeg or the dev HTTP client are skipped, not failed, when those aren't installed.

## Configuration

Every setting is an environment variable with a safe local default. [`.env.example`](.env.example) lists all 19, with explanations. The main ones:

| Variable | Default | Meaning |
|---|---|---|
| `API_KEYS` | `dev-local-key` | Comma-separated accepted keys |
| `TRANSCRIPTION_ENGINE` | `mock` | `mock` or `whisper` |
| `WHISPER_MODEL` | `base` | `tiny` … `large` |
| `MAX_UPLOAD_BYTES` | 200 MB | Upload size cap |
| `CHUNK_THRESHOLD_SECONDS` / `CHUNK_LENGTH_SECONDS` / `CHUNK_OVERLAP_SECONDS` | 300 / 240 / 5 | Long-audio chunking |
| `MAX_RETRIES` / `RETRY_BACKOFF_BASE_SECONDS` | 3 / 2 | Retry policy |
| `WORKER_COUNT` | 1 | Worker tasks in the process |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` | 60 / 60 | Per-key limit |
| `STORAGE_DIR` | `storage` | Root for audio, transcripts, database and dead letters |
| `LOG_FORMAT` / `LOG_LEVEL` | `json` / `INFO` | Logging |

Invalid values stop the service at startup with a message naming the setting, rather than surfacing halfway through a job.

## Known limitations

- **The in-memory queue isn't durable.** Startup recovery re-queues unfinished jobs from the database, but a real queue is needed before running more than one process.
- **The rate limiter is per process.** With N instances, a key can get up to N× its limit.
- **The merge is only word-exact when the engine reports word timings.** Whisper does. An engine that only gives segment timings falls back to segment-level merging. When two chunks divide a seam into differently bounded segments, that fallback's error at the seam is up to half a segment: a few words may repeat or be dropped.
- **Small chunks lose model context.** The merge keeps every word, but Whisper sometimes formats a phrase differently when it hears only 20–30 s ("Third, hiring" became "3. Hiring"). The 4-minute default chunks make this rare.
- **Retries redo the whole file.** For long audio, retrying only the failed chunks would save work.
- **Auth is a static API key.** That's fine for service-to-service use in a demo, but not a multi-tenant identity system.
