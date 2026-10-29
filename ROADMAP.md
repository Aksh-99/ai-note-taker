# Roadmap

Each nightly run picks the **first unchecked step**, builds it, ticks it off, and logs it in `DEVLOG.md`.
Steps are sized to roughly 1-5% of the project. Edit freely: reorder, add, or delete steps to steer the work.

Rule for every step: free/open models only, no paid API keys. Code must run and tests must pass before pushing.

## Phase 0: Foundations
- [x] 0.1 Repo scaffold: README, ROADMAP, DEVLOG
- [x] 0.2 Python project setup: `pyproject.toml`, `src/notetaker/` package, `.gitignore`, `requirements.txt`
- [x] 0.3 Tooling: ruff + pytest config, one smoke test, `Makefile` with `make test` / `make lint`
- [x] 0.4 GitHub Actions CI: lint + tests on every push
- [x] 0.5 Config module (pydantic-settings) and `.env.example` documenting every setting

## Phase 1: Transcription
- [x] 1.1 Audio loading + normalization (ffmpeg, 16kHz mono) with tests on a tiny generated clip
- [x] 1.2 Transcribe with faster-whisper (small model), return segments with timestamps
- [x] 1.3 CLI: `notetaker transcribe lecture.mp3` writes `transcript.json` + `transcript.md`
- [x] 1.4 Chunk long audio (VAD-based) so 90-minute lectures don't blow up memory
- [x] 1.5 Benchmark script: real-time factor and memory for tiny/base/small models

## Phase 2: Evaluation set (the ML-engineer part)
- [x] 2.1 Script to download a few MIT OpenCourseWare lectures + their official transcripts
- [x] 2.2 Text normalization for fair comparison (case, punctuation, numbers)
- [ ] 2.3 WER / CER evaluation with jiwer, results table saved to `reports/`
- [ ] 2.4 Technical-term recall metric (e.g. "eigenvalue", "backprop") from a curated term list
- [ ] 2.5 Baseline report: model size vs WER vs speed, written up in `reports/baseline.md`

## Phase 3: Notes generation
- [ ] 3.1 LLM client abstraction supporting Ollama (local) and Groq / HF free tier (hosted)
- [ ] 3.2 Prompted summarizer: transcript chunks to structured notes (headings, key points)
- [ ] 3.3 Map-reduce summarization for long lectures
- [ ] 3.4 Key concepts + definitions extraction as JSON with schema validation
- [ ] 3.5 Flashcard generation (Q/A pairs), exportable to Anki CSV
- [ ] 3.6 Quiz generation (multiple choice with distractors)
- [ ] 3.7 Notes quality eval: faithfulness check against transcript (LLM-as-judge with a free model)

## Phase 4: Agent layer
- [ ] 4.1 LangGraph agent skeleton: transcript -> notes -> tasks pipeline as a graph
- [ ] 4.2 Deadline/assignment extraction ("homework due Friday") with date resolution
- [ ] 4.3 To-do export (JSON + .ics calendar file)
- [ ] 4.4 Agent tests with recorded fixtures (no live LLM calls in CI)

## Phase 5: Search across lectures (RAG)
- [ ] 5.1 Embeddings with sentence-transformers, store in FAISS/Chroma with lecture + timestamp metadata
- [ ] 5.2 Ingest command: index all transcripts in a folder
- [ ] 5.3 Ask command: answer questions with citations to lecture + timestamp
- [ ] 5.4 Retrieval eval: small labeled question set, recall@k and MRR
- [ ] 5.5 Hybrid search (BM25 + vectors) and compare against 5.4 baseline

## Phase 6: Model improvement
- [ ] 6.1 Build fine-tuning dataset from eval-set audio (train/val split, no leakage)
- [ ] 6.2 LoRA fine-tune Whisper small on technical lecture audio (Colab/Kaggle free GPU notebook)
- [ ] 6.3 Evaluate fine-tuned vs base: WER + technical-term recall, write up in `reports/finetune.md`
- [ ] 6.4 int8 quantization (CTranslate2) of the best model, speed/accuracy tradeoff table

## Phase 7: API + UI
- [ ] 7.1 FastAPI backend: upload audio, job status, fetch results
- [ ] 7.2 Background job queue for transcription
- [ ] 7.3 Web UI (Gradio or Streamlit): upload, view notes, flashcards, search
- [ ] 7.4 Dockerfile + docker-compose for local run

## Phase 8: Ship it
- [ ] 8.1 Deploy to Hugging Face Spaces (free tier)
- [ ] 8.2 Monitoring: request logs, latency, error rate
- [ ] 8.3 README polish: architecture diagram, demo GIF, results tables, how to run
- [ ] 8.4 Write-up: `docs/case-study.md` covering problem, approach, metrics, lessons
