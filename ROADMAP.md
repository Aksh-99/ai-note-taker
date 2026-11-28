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
- [x] 2.3 WER / CER evaluation with jiwer, results table saved to `reports/`
- [x] 2.4 Technical-term recall metric (e.g. "eigenvalue", "backprop") from a curated term list
- [x] 2.5 Baseline report: model size vs WER vs speed, written up in `reports/baseline.md`

## Phase 3: Notes generation
- [x] 3.1 LLM client abstraction supporting Ollama (local) and Groq / HF free tier (hosted)
- [x] 3.2 Prompted summarizer: transcript chunks to structured notes (headings, key points)
- [x] 3.3 Map-reduce summarization for long lectures
- [x] 3.4 Key concepts + definitions extraction as JSON with schema validation
- [x] 3.5 Flashcard generation (Q/A pairs), exportable to Anki CSV
- [x] 3.6 Quiz generation (multiple choice with distractors)
- [x] 3.7 Notes quality eval: faithfulness check against transcript (LLM-as-judge with a free model)

## Phase 4: Agent layer
- [x] 4.1 LangGraph agent skeleton: transcript -> notes -> tasks pipeline as a graph
- [x] 4.2 Deadline/assignment extraction ("homework due Friday") with date resolution
- [x] 4.3 To-do export (JSON + .ics calendar file)
- [x] 4.4 Agent tests with recorded fixtures (no live LLM calls in CI)

## Phase 5: Search across lectures (RAG)
- [x] 5.1 Embeddings with sentence-transformers, store in FAISS/Chroma with lecture + timestamp metadata
- [x] 5.2 Ingest command: index all transcripts in a folder
- [x] 5.3 Ask command: answer questions with citations to lecture + timestamp
- [x] 5.4 Retrieval eval: small labeled question set, recall@k and MRR
- [x] 5.5 Hybrid search (BM25 + vectors) and compare against 5.4 baseline

## Phase 6: Model improvement
- [x] 6.1 Build fine-tuning dataset from eval-set audio (train/val split, no leakage)
- [x] 6.2a LoRA training code: data loading + collator, LoRA setup, training loop, `notetaker finetune` CLI (tested on a tiny random Whisper)
- [x] 6.2b Colab/Kaggle free-GPU notebook that runs `notetaker finetune` on Whisper small and saves the adapter
- [x] 6.3a Evaluation code: `notetaker finetune-eval` (adapter off vs on on the test clips, pooled WER/CER + term recall, paired bootstrap interval, `reports/finetune.md`), notebook section, tests on a tiny Whisper
- [x] 6.3b (moved to 8.2b: the real GPU run needs a human-started Colab/Kaggle session)
- [x] 6.4a `notetaker export-ct2`: merge the LoRA adapter into Whisper, convert to CTranslate2 (int8 / float16 / float32 storage), check it loads in faster-whisper; notebook section; tests on a tiny Whisper
- [x] 6.4b Local CT2 folders as models: `transcribe` / `benchmark` take an export-ct2 folder wherever they take a size (validation, labels, "auto" compute type follows the stored precision, size on disk in the benchmark table)
- [x] 6.4c Speed/accuracy tradeoff table: time + score each precision (WER, term recall, RTF, size) on the test clips, `reports/quantization.md`: `notetaker quantize-eval` + notebook cell (real numbers come with 8.2b)

## Phase 7: API + UI
- [x] 7.1a FastAPI backend for transcription: on-disk job store, `POST /jobs` upload (type + size checks), job status, transcript + file downloads, `notetaker serve`; tests with fake runners
- [x] 7.1b Notes for API jobs: optional notes stage after transcription (agent pipeline: notes, flashcards, quiz, tasks), results listed in the job and downloadable; tests with a fake LLM
- [ ] 7.2 Background job queue for transcription
- [ ] 7.3 Web UI (Gradio or Streamlit): upload, view notes, flashcards, search
- [ ] 7.4 Dockerfile + docker-compose for local run

## Phase 8: Ship it
- [ ] 8.1 Deploy to Hugging Face Spaces (free tier)
- [ ] 8.2 Monitoring: request logs, latency, error rate
- [ ] 8.2b Real results (needs Akshara to start a free Colab/Kaggle GPU session): run `notebooks/finetune_whisper_small.ipynb` with `SMOKE_TEST = False`, commit `reports/finetune.md` + `reports/finetune.json` + `reports/quantization.md` and put the headline numbers in the README (6.3b and 6.4 real runs)
- [ ] 8.3 README polish: architecture diagram, demo GIF, results tables, how to run
- [ ] 8.4 Write-up: `docs/case-study.md` covering problem, approach, metrics, lessons
