# Audio Transcription Service

This is my submission for the Volga Partners. It's a small transcription service that other systems call over HTTP: you upload an audio file, get a job id back straight away, and poll until the transcript (with per-segment timestamps) is ready. Transcription is done with OpenAI's [Whisper](https://github.com/openai/whisper).

The brief was more interested in engineering decisions than in the model itself, so most of this README is about why things are built the way they are, and what I'd change before running it for real.

## The problem and the framing

- **The problem:** teams have a lot of recorded speech (sales calls, meetings, podcasts) that can't be searched until it's transcribed.
- **What this is:** a **transcription microservice that other systems call**, not an app for people. That one decision drives most of the others: the API is job-based (submit, then poll) because the callers are systems, not a person watching a spinner.
- **The four use cases, and the design choice each one justifies:**
  - an overnight call-centre batch -> the async queue
  - a meetings product that needs results quickly -> a fast path for short files
  - 1–3 hour podcasts -> chunking long files
  - a compliance audit trail -> retries, dead-letter records and versioned transcripts


## Running it

Application requires Python 3.13 and ffmpeg on your PATH (`winget install Gyan.FFmpeg` on Windows, `brew install ffmpeg` on macOS, `apt install ffmpeg` on Debian/Ubuntu).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
uvicorn app.main:app --reload
```

That starts it with a mock engine, which returns fake but deterministic transcripts. It's handy for trying the API without downloading a model. For real transcription, install Whisper as well (it pulls in PyTorch, roughly 2 GB) and switch the engine:

```bash
pip install -r requirements-whisper.txt
TRANSCRIPTION_ENGINE=whisper uvicorn app.main:app           # macOS / Linux
$env:TRANSCRIPTION_ENGINE="whisper"; uvicorn app.main:app   # PowerShell
```

The Swagger UI is at http://localhost:8000/docs. Click "Authorize" and enter `dev-local-key` (the default local key), or use curl:

```bash
curl -X POST http://localhost:8000/api/v1/transcriptions \
  -H "X-API-Key: dev-local-key" -F "file=@call.mp3"
# {"job_id":"3f1c...","status":"queued","status_url":"/api/v1/transcriptions/3f1c..."}

curl http://localhost:8000/api/v1/transcriptions/3f1c... -H "X-API-Key: dev-local-key"
```

All settings are environment variables. `.env.example` lists them with their defaults; copy it to `.env` and start with `uvicorn app.main:app --env-file .env`.

### Docker

```bash
# mock engine (~860 MB image)
docker build -t volga-transcription .
docker run -p 8000:8000 -v transcription-data:/data volga-transcription

# with Whisper, CPU build of PyTorch (~3.6 GB image, first build takes a while)
docker build -t volga-transcription:whisper --build-arg INSTALL_WHISPER=true .
docker run -p 8000:8000 -e TRANSCRIPTION_ENGINE=whisper \
  -v transcription-data:/data -v whisper-models:/home/appuser/.cache \
  volga-transcription:whisper
```

Everything stateful (audio, transcripts, the SQLite file, dead-letter records) lives under `/data`, so jobs survive the container being recreated. The second volume caches the Whisper model. Without it, each new container downloads the model again before its first job (about 74 s for the first job, versus 9 s with the cache in my testing). The container runs as a non-root user and has a health check that calls `/healthz`.

### Tests

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -p "test_*.py"
```

There are 175 tests and they take about 15 seconds. They generate their own audio (sine tones) and use the mock engine, so there's no model download and no fixture files in the repo. Tests that need ffmpeg or the dev HTTP client are skipped if those aren't installed.

## API

Everything under `/api/v1` needs an `X-API-Key` header.

A job goes `queued → processing → completed`. If an attempt fails it goes to `retrying` and back to `processing`, and if it runs out of retries it ends up `failed`. The `error` field on the job says what went wrong.

## Design decisions

### Validating uploads

I do the cheap checks first, so a bad upload gets turned away as early as possible.

First I look at the file extension. It's a quick check, but I don't fully trust it, because anyone can rename a file.

Then I check the size, and I actually do this twice. FastAPI reads the whole upload before my code even runs. So if I only checked inside the route, someone could still send a 5 GB file and the server would take all of it first. To stop that, a small middleware looks at the `Content-Length` header and says no before the body is read. Some clients don't send that header, though, so I check the size again while saving the file.

Last, I run `ffprobe` on the file. It only reads the start of the file, so it's fast. If the file isn't real audio, the caller gets a 422 right away. Without this, a broken file would sit in the queue and only fail a few minutes later.

One more thing: I never use the filename the client sends when saving the file. The saved name comes from the job id. I also check every storage path, so nobody can trick the server into writing outside the storage folder.


### Transcription

The real transcription is done by `WhisperEngine`. I ask Whisper for timings on every single word (`word_timestamps=True`), not just on each sentence. I need those word timings later, when I glue long files back together.

The model only loads when the first job comes in, not when the server starts. So the API starts right away. And if you only use the mock engine, you never have to install PyTorch at all.

Something I learned along the way: Whisper can't safely do two transcriptions at the same time with the same model. While it's working, it changes some things inside the model, so two jobs at once would mess each other up. So I put a lock around it, and each engine handles one job at a time. If you want more speed, the answer is to run more worker processes, not more threads.

To test it for real, I made some speech with the Windows text-to-speech voice and ran it through the `base` model on a normal CPU. A 74-second meeting took about 7 seconds, and every word came out right, even tricky ones like "12%", "back-end" and "take-home".

### Long audio and the chunk merge

Files longer than 5 minutes are cut into 4-minute chunks, and each chunk overlaps the next by 5 seconds so no word gets cut in half. The chunks are transcribed side by side.

Anything said in an overlap gets transcribed twice, so when I put the chunks back together I merge word by word. I put one cut point in the middle of each overlap, and keep each word from the chunk whose side of the cut it falls on. I cut in the middle because the edge of a chunk is where Whisper most often chops or mishears a word. A word sitting exactly on the cut goes to the later chunk, so it's kept exactly once.

My first version merged whole sentences instead of words. It passed my mock tests, but on real Whisper output it turned "the previous quarter, driven mostly by" into "the previous court." and repeated "home assessment" somewhere else. Whisper's sentences are about as long as the overlap, so they can't be split cleanly. Switching to single words fixed it: the chunked transcript now has exactly the same 181 words as transcribing the whole file in one go.

### Handling many uploads at once

An upload only does the quick stuff: check the file, save it, add a row, and put the job in the queue. It answers in milliseconds. The slow part, the transcription, happens later in a background worker. So if lots of files arrive at once, the queue just gets longer. Nobody is left waiting on an open connection.

The slow work runs in separate threads, so the server keeps answering requests while a file is being transcribed. And if the same job ever gets picked up twice, only one worker actually runs it.

### Storage

There are two tables. `jobs` keeps track of each upload: its status, retries, errors and so on. `transcripts` holds the results, and it keeps old versions, so re-running a job adds a new version instead of overwriting the old one.

Short transcripts go straight into the database. Long ones (over 20,000 characters) are saved as a file, and the database just keeps the path. That keeps the table small and quick.

I never save the real API keys. Each job is linked to a hash of the key instead, so a leaked database doesn't give anyone a working key.

### Retries and failures

Some errors aren't worth retrying. If the audio is broken or the file is missing, trying again won't help, so the job fails right away. Other errors, like the engine crashing, get retried after 2, then 4, then 8 seconds. The worker doesn't sit and wait during that time. It moves on to other jobs.

A job gets up to 4 tries (1 plus 3 retries). If it still fails, it's marked `failed` and a small JSON file is saved in a dead-letter folder so someone can check what went wrong.

The queue lives in memory, so a restart would normally lose it. To fix that, when the service starts it looks in the database and puts any unfinished jobs back in the queue.

## Known limitations

- The queue lives in memory. Unfinished jobs come back after a restart, but for more than one server you'd want a real queue.
- The rate limit is counted per server, so with several servers a key could go over its limit.
- The word-by-word merge needs word timings. Whisper gives them, but an engine that doesn't may repeat or drop a few words where chunks join.
- Very short chunks give Whisper less to go on, so it sometimes writes things a bit differently ("Third" became "3."). The 4-minute default mostly avoids this.
- If one chunk fails, the retry redoes the whole file, not just that chunk.
- Login is a simple API key. Fine for a demo, not for real users.
