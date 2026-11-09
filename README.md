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

WER weighs every word equally, but for lecture notes the technical vocabulary matters most. Technical-term recall counts how many of the curated terms in `eval/technical_terms.json` (about 60 terms such as "eigenvalue", "dynamic programming", "sample space", "backpropagation", each with the spellings that count as the same term) that the lecturer said also appear in the transcript:

```bash
notetaker terms --hyps outputs/eval/whisper-small          # all lectures, report in reports/terms/
notetaker terms reference.txt my_transcript.txt            # quick check of one pair
```

This writes `reports/terms/<system>.md` (recall and precision per lecture, recall by subject, and the most-missed terms), `<system>.json`, and `reports/terms/summary.md`. Counts are clipped, so a term earns at most as many hits as it was spoken; precision shows whether the model writes jargon that was never said. Edit the term list freely to match your own courses.

Once at least one model has a WER report, combine everything into the baseline report, the reference point for later fine-tuning and quantization:

```bash
notetaker baseline                    # reads reports/wer, reports/terms, reports/benchmark
notetaker baseline --rtf-budget 1.0   # allow slower models in the recommendation
```

This writes `reports/baseline.md` and `reports/baseline.json`: one row per model size with WER, CER, term recall, real-time factor and RAM, the Pareto frontier (models no other model beats on both accuracy and speed), the most accurate model within the speed budget, and what each step up in size buys in WER versus extra compute. Report names are matched across folders, so `whisper-small` in the WER report and `small` in the benchmark land in the same row. See `reports/baseline.md` for the full command loop.

### LLM setup (for notes, flashcards and quizzes)

Notes generation talks to an LLM through one client (`src/notetaker/llm.py`) that supports three free options. Pick one with `NOTETAKER_LLM_PROVIDER` in `.env`:

| Provider | Cost | Setup | Default model |
|---|---|---|---|
| `ollama` (default) | free, runs locally | install [Ollama](https://ollama.com), then `ollama pull llama3.1:8b` | `llama3.1:8b` |
| `groq` | free tier, rate-limited | free key from console.groq.com, set `GROQ_API_KEY` | `llama-3.1-8b-instant` |
| `huggingface` | free monthly credits | free token from huggingface.co/settings/tokens, set `HF_TOKEN` | `meta-llama/Llama-3.1-8B-Instruct` |

Check that it works:

```bash
notetaker llm "Explain eigenvalues in one sentence"
notetaker llm "List three linear algebra terms" --json --provider groq
```

Rate limits (HTTP 429) and server errors are retried with exponential backoff (1s, 2s, 4s, ...), honoring the server's `Retry-After`. Tests never call a real LLM: they use a fake transport or `FakeLLMClient`.

Turn a transcript into structured study notes:

```bash
notetaker notes outputs/lecture01/transcript.json              # notes.md + notes.json next to it
notetaker notes outputs/lecture01/transcript.json -p groq --chunk-words 2000
```

The transcript is split into chunks of about 1,200 words (a few minutes of speech, cut only at pauses). Each chunk goes to the LLM, which returns JSON sections (heading, one-line summary, key points, start time). Replies are validated, and broken JSON is sent back to the model once with the error so it can fix it. Every section heading carries a timestamp like `[00:12:41]`; timestamps the model invents outside its chunk are replaced by the chunk's start, so they always point at a real moment in the recording.

Then a "reduce" step merges the per-chunk sections into the lecture's main topics, adds a short overview and key takeaways (map-reduce summarization, `src/notetaker/mapreduce.py`). The model only groups numbered sections and writes the new headings and summaries; each topic's timestamp and key points are copied from its sections in code, with near-duplicate points removed, so nothing is invented and nothing is lost. If the section list is too long for one prompt (`--reduce-words`, default 2,500), it is merged in batches and then merged again. `--no-reduce` keeps the plain per-chunk notes, and if merging fails the per-chunk notes are still written.

Extract a glossary of key concepts and their definitions:

```bash
notetaker concepts outputs/lecture01/transcript.json           # concepts.md + concepts.json next to it
notetaker concepts --schema                                     # print the JSON Schema of concepts.json
```

Each chunk goes to the LLM, which lists the technical terms explained in it with a definition and any aliases (e.g. "SVD"). Replies are validated with pydantic (broken JSON gets one repair attempt). A term that does not actually occur in the chunk is dropped and reported, the same concept found in several chunks is merged into one entry (matched on its name or aliases, ignoring case, punctuation and plurals), and timestamps are found by searching the transcript, not asked from the model: each entry lists the time of every paragraph that mentions it. The output file has a published schema in [`schemas/concepts.schema.json`](schemas/concepts.schema.json), and `load_concepts` re-validates it on read, so later steps (flashcards, quizzes) can trust its shape.

Turn a lecture into flashcards you can study in Anki:

```bash
notetaker flashcards outputs/lecture01/transcript.json                     # flashcards.md + .json + .csv
notetaker flashcards outputs/lecture01/transcript.json \
    --concepts outputs/lecture01/concepts.json --deck "Linear Algebra::L01" # add glossary cards
notetaker flashcards --concepts-only --concepts outputs/lecture01/concepts.json  # no LLM calls at all
```

The LLM writes up to 6 question/answer cards per chunk. A card is kept only if at least half of its answer's content words (ignoring words like "the" or "is") occur in the transcript chunk, so answers built from outside knowledge are dropped. Each card gets the timestamp of the paragraph it overlaps most, repeated questions are removed, and `--concepts` adds a "What is X?" card for every glossary entry. `flashcards.csv` imports directly with Anki's File > Import: header lines set the deck, note type and tags (use `--plain-csv` for Anki older than 2.1.55). The JSON format is published in [`schemas/flashcards.schema.json`](schemas/flashcards.schema.json).

Make a multiple-choice practice quiz:

```bash
notetaker quiz outputs/lecture01/transcript.json                          # quiz.md + quiz.json
notetaker quiz outputs/lecture01/transcript.json --concepts outputs/lecture01/concepts.json
notetaker quiz --concepts-only --concepts outputs/lecture01/concepts.json  # no LLM calls at all
```

For each chunk the LLM writes up to 4 questions, each with the correct answer and 3 distractors as separate fields. The code builds the options and shuffles them with a fixed seed (`--seed`), because models tend to put the right answer first. A question is kept only if its correct answer is grounded in the transcript (same content-word test as the flashcards), options that repeat each other or say "all/none of the above" trigger a repair request, and each question links to the paragraph it is about. `--concepts` adds "which term matches this definition?" questions whose wrong options are the glossary terms introduced closest in time, with the term masked out of its own definition. `quiz.md` lists the questions with an answer key at the end, and the CLI reports how often the correct answer is the longest option (a classic giveaway; about 25% is fair with 4 options). The JSON format is published in [`schemas/quiz.schema.json`](schemas/quiz.schema.json).

## Status
Early development. See [ROADMAP.md](ROADMAP.md) for the plan and [DEVLOG.md](DEVLOG.md) for progress.
