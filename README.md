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

## Configuration
All settings come from environment variables (prefixed `NOTETAKER_`) or a local `.env` file. Start from the template, then check what the app picked up:

```bash
cp .env.example .env
notetaker config
```

The defaults are free and local (faster-whisper `small`, Ollama), so no API key is needed. `.env` is gitignored.

## Status
Early development. See [ROADMAP.md](ROADMAP.md) for the plan and [DEVLOG.md](DEVLOG.md) for progress.
