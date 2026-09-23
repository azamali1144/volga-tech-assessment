# volga-tech-assessment: Audio Transcription Pipeline

A transcription service that other systems call over an API. A caller uploads an audio file and gets back a job id straight away. A background worker then transcribes the file with Whisper, and the caller polls for the text with per-segment timestamps.

> **Status:** work in progress. This README grows as each part of the service is built.

## Planned scope

- Upload endpoint with extension checks and a size-capped streaming save
- Conversion of any supported format to 16kHz mono WAV with ffmpeg
- Splitting of long recordings into overlapping chunks, merged back into one transcript
- Whisper transcription engine, plus a deterministic mock engine for tests
- Job tracking in SQLite (status, retries, errors), with transcripts versioned
- Background worker with retry, exponential backoff and dead-letter handling
- API key auth, per-key rate limiting and structured JSON logging
- Docker image with ffmpeg included

## Project layout

```
app/       application code
tests/     unit tests
storage/   runtime data (gitignored)
```
