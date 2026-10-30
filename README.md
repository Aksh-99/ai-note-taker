# ai-note-taker

[![CI](https://github.com/Aksh-99/ai-note-taker/actions/workflows/ci.yml/badge.svg)](https://github.com/Aksh-99/ai-note-taker/actions/workflows/ci.yml)

An AI note taker that listens to lecture recordings and turns them into study material. Built end to end on free, open models.

**What it does**
- Transcribes lectures with Whisper (fine-tuned for technical vocabulary)
- Writes structured notes, key concepts, flashcards, and quizzes
- Pulls out deadlines and assignments into a to-do list and calendar file
- Lets you search across every lecture you've recorded, with answers cited to the exact timestamp

**What makes it an ML engineering project, not just an API wrapper**
- A real evaluation set with word error rate and technical-term recall
- LoRA fine-tuning of Whisper with before/after results
- Quantization with speed vs accuracy tradeoffs
- Retrieval evals (recall@k, MRR) for the search feature
- CI, Docker, and a deployed demo

## Stack
Python · faster-whisper · Hugging Face Transformers + PEFT · Ollama / Groq free tier · LangGraph · sentence-transformers · FAISS · FastAPI · Gradio · GitHub Actions · Hugging Face Spaces

## Requirements
Python 3.10+ and [ffmpeg](https://ffmpeg.org/) for decoding audio (`sudo apt install ffmpeg`, `brew install ffmpeg`, or `winget install ffmpeg`). Then install the package with `make install`.

Whisper model weights are not installed with the package. faster-whisper downloads them for free from the Hugging Face Hub the first time a model is used (about 75 MB for `tiny`, 480 MB for `small`) and caches them for later runs.

## Configuration
All settings come from environment variables (prefixed `NOTETAKER_`) or a local `.env` file. Start from the template, then check what the app picked up:

```bash
cp .env.example .env
notetaker config
```

The defaults are free and local (faster-whisper `small`, Ollama), so no API key is needed. `.env` is gitignored.

## Usage
Transcribe a lecture recording (any format ffmpeg reads, including video):

```bash
notetaker transcribe lecture01.mp3                 # writes outputs/lecture01/
notetaker transcribe lecture01.mp4 -m base -l en   # smaller model, skip language detection
```

Each run writes two files: `transcript.json` (segments, word timestamps, and confidence scores, used by every later step) and `transcript.md` (readable paragraphs, each labelled with its start time like `[00:12:41]`). Run `notetaker transcribe -h` for all options.

Long recordings are split automatically at pauses into pieces of at most 5 minutes, transcribed one at a time, and stitched back together with correct timestamps, so a 90-minute lecture never goes to Whisper in one call. Change the limit with `--max-chunk-seconds` (or `NOTETAKER_CHUNK_MAX_SECONDS`); `0` turns splitting off.

Compare the speed and memory of Whisper sizes on your own machine:

```bash
notetaker benchmark lecture01.mp3 -m tiny base small --max-seconds 600 --repeats 3
```

This writes `reports/benchmark/benchmark.md` (a table of load time, transcription time, real-time factor, and peak RAM per model) plus `benchmark.json`. Each model runs in its own fresh process so its memory number is not inflated by the models measured before it. Run it once beforehand so model downloads are not counted in the load times.

Download the evaluation set (MIT OpenCourseWare lectures plus their official human-made captions, used as reference transcripts for measuring accuracy):

```bash
notetaker evalset --list                      # show the lectures in eval/ocw_lectures.json
notetaker evalset                             # audio + transcripts into data/raw/ocw/<id>/
notetaker evalset --only mit-18.06-lec01      # just one lecture
notetaker evalset --no-audio                  # transcripts only (a few KB each)
```

Each lecture folder gets `audio.flac` (16 kHz mono, ready for Whisper), `reference.json` (timed caption cues), `reference.txt`, and `meta.json` (source URLs, license, date). Only the audio track is saved; the video is streamed through ffmpeg and never stored. Re-running skips what is already there (`--force` re-downloads). OCW material is CC BY-NC-SA licensed and large, so `data/raw/` is gitignored and each person downloads their own copy. If OCW changes a page layout, add `audio_url` / `transcript_url` to that lecture in the manifest.

Before transcripts are scored, both the model output and the reference go through the same text normalizer (lowercase, no punctuation, caption tags and speaker labels removed, numbers spelled out, filler words dropped), so only real recognition mistakes count. To see exactly what the scorer will compare:

```bash
notetaker normalize data/raw/ocw/mit-18.06-lec01/reference.json
notetaker normalize out/transcript.json --keep-digits --keep-fillers
```

Score a model on the evaluation set with word error rate (WER) and character error rate (CER). Transcribe each lecture into `outputs/eval/<model>/<id>/`, then point `notetaker wer` at that folder:

```bash
for d in data/raw/ocw/*/; do
  id=$(basename "$d")
  notetaker transcribe "$d/audio.flac" -m small -l en -o "outputs/eval/whisper-small/$id"
done
notetaker wer --hyps outputs/eval/whisper-small            # all lectures, report in reports/wer/
notetaker wer --hyps outputs/eval/whisper-small --max-seconds 600   # first 10 minutes only
notetaker wer reference.txt my_transcript.txt              # quick score of one pair
```

This writes `reports/wer/<system>.md` (per-lecture WER, substitutions / deletions / insertions, CER, and the most common word confusions), `<system>.json`, and `reports/wer/summary.md`, a table with one row per model that is rebuilt every time a new model is scored. The headline number is pooled WER: total errors divided by total reference words across all lectures.

## Status
Early development. See [ROADMAP.md](ROADMAP.md) for the plan and [DEVLOG.md](DEVLOG.md) for progress.
