# Dev Log

One entry per work session, newest first. Each entry covers what was built, how it works, and what's next.

---

## 2026-10-27 · Step 2.1: Download MIT OCW lectures and official transcripts

**What I did**
- Added `eval/ocw_lectures.json`, a committed manifest of four first lectures from MIT OpenCourseWare: 18.06 Linear Algebra, 6.006 Algorithms, 6.034 Artificial Intelligence and 6.041 Probability. Everyone evaluates on the same list
- Added `src/notetaker/evalset.py`: loads and validates the manifest, finds the video (.mp4) and caption (.srt/.vtt) links on each lecture page, parses SRT and WebVTT captions into timed cues, and extracts 16 kHz mono FLAC audio with ffmpeg straight from the video URL
- New CLI command `notetaker evalset` with `--list`, `--only ID ...`, `--no-audio`, `--force` and `-o`. Output goes to `data/raw/ocw/<id>/` (gitignored: the files are large, and the CC BY-NC-SA license attaches attribution and non-commercial conditions to any copy, so each person downloads their own): `audio.flac`, `reference.json`, `reference.txt`, `meta.json`
- A manifest entry can give `audio_url` / `transcript_url` directly, which skips the page scan if OCW ever changes its HTML
- Added 31 tests in `tests/test_evalset.py` with a fake "internet" (a dict from URL to text), plus a real ffmpeg extraction from a generated stereo 44.1 kHz tone that checks the result is 16 kHz mono. `make check` passes (148 tests, lint clean)
- Honest caveat: the build sandbox cannot reach ocw.mit.edu, so the four page URLs and the page layout have not been checked against the live site yet. Run `notetaker evalset --no-audio` locally first; any lecture that fails prints which manifest field to fill in

**How it works**
OCW publishes professionally written captions for its lecture videos, and those are the "ground truth" for measuring word error rate in Step 2.3. The downloader reads each lecture page with Python's built-in HTML parser and collects every link, then picks the archive.org .mp4 and the .srt caption file (falling back to any .mp4 and to .vtt). Caption files are blocks separated by blank lines, each with a timing line `00:01:02,500 --> 00:01:05,000` followed by the text; one regular expression handles both SRT (comma before the milliseconds) and WebVTT (dot, optional hours), and formatting tags like `<i>` are stripped. For audio, ffmpeg is given the video URL directly and told to drop the video stream, mix to mono and resample to 16 kHz, so a few hundred MB of video turns into roughly 50 MB of lossless FLAC without the video ever touching the disk. Every file is first written as `name.part` and then renamed: a rename is atomic, so an interrupted run never leaves a truncated file under the real name, which is what makes "skip files that already exist" safe on re-runs. The network and ffmpeg functions are passed into `download_lecture` as arguments (dependency injection), so tests swap in fakes and run offline.

**Next up**
Step 2.2: Text normalization for fair comparison (case, punctuation, numbers).

---

## 2026-10-26 · Step 1.5: Speed and memory benchmark for Whisper sizes

**What I did**
- Added `src/notetaker/benchmark.py`: `benchmark_model()` loads one Whisper size, times the load and N repeated transcriptions, and records peak memory; `BenchmarkResult` derives the median time, real-time factor (RTF), "x real time", and model RAM; `write_report()` saves `benchmark.json` and a Markdown table `benchmark.md`
- New CLI command: `notetaker benchmark lecture.mp3 -m tiny base small --repeats 3 --max-seconds 600`, writing to `reports/benchmark/` by default
- Each model runs in a fresh child process by default (`--no-isolate` turns this off), so memory numbers do not leak from one model into the next
- If one model fails (download blocked, out of memory), the error is printed and the remaining models still run; the command only fails if none succeed
- `--max-seconds` benchmarks just the start of a long lecture for a quick estimate
- Added 20 tests in `tests/test_benchmark.py` using a fake model and a fake clock, so every timing is exactly known, plus a real child-process run with the fake loader. `make check` passes (117 tests, lint clean). The `slow` test with the real `tiny` model could not run tonight (Hugging Face downloads are blocked in the build sandbox), so run `notetaker benchmark` on a real lecture locally to get the first numbers

**How it works**
The headline metric is the real-time factor: compute time divided by audio length, so RTF 0.25 means an hour of lecture takes 15 minutes, and anything under 1 keeps up with live speech. Each model is timed over several runs and the median is kept, because the first run often pays one-off warm-up costs and the median, unlike the mean, ignores one slow outlier. Timing uses `time.perf_counter`, a high-resolution clock that never jumps when the system time changes. Memory is the process's peak resident set size (the RAM pages it actually used), read from the operating system via Python's `resource` module. A peak can only go up, so measuring tiny, base and small in one process would make every later number include the earlier models; instead each model gets a brand-new Python process ("spawn" start method), and its peak is reported along with the baseline measured just before the model was loaded, so the difference is roughly what the model itself costs. The loader and the clock are passed in as arguments (dependency injection), which is what lets the tests replace them with fakes and check the arithmetic exactly.

**Next up**
Step 2.1: Script to download a few MIT OpenCourseWare lectures + their official transcripts.

---

## 2026-10-25 · Step 1.4: VAD-based chunking for long lectures

**What I did**
- Added `src/notetaker/chunking.py`: an energy-based voice activity detector (pure NumPy, no model download) and `plan_chunks()`, which splits a waveform into consecutive pieces of at most `max_chunk_s` seconds, cutting in the middle of pauses
- `transcribe()` now splits anything longer than `NOTETAKER_CHUNK_MAX_SECONDS` (new setting, default 300 s; 0 turns it off), transcribes the chunks one at a time, and stitches them into one `Transcript`: timestamps of segments and words are shifted by each chunk's start, segment ids are renumbered 1, 2, 3, ... across chunks
- Chunks that are pure silence are skipped (Whisper tends to invent text like "Thank you." on silence), and a language auto-detected on the first chunk is locked in for the rest
- CLI: new `--max-chunk-seconds` option and a progress line per chunk (`chunk 3/18 [00:10:00 - 00:15:00]`)
- Documented the setting in `.env.example` and the README
- Added 30 tests (`tests/test_chunking.py`, `tests/test_transcribe_chunked.py`, plus CLI tests) on synthetic tones and pauses where every cut point is known in advance, including a randomized check that chunks always tile the audio exactly. `make check` passes (98 tests, lint clean). Planning the cuts for 90 minutes of audio takes under a second

**How it works**
The detector slices the audio into 30 ms frames and computes each frame's loudness in decibels, 20 log10(RMS). A frame counts as speech if it is louder than a threshold that adapts to the recording: 35 dB below its 95th-percentile loudness, but never below -50 dB, so quiet recordings still work and faint hiss never counts as speech. Runs of non-speech frames are found in one vectorized pass (pad the silence indicator with zeros, take `np.diff`, and the +1 and -1 entries mark where runs start and end). Cutting is greedy: from the current start, look in the window between half the maximum and the maximum length, pick the longest pause of at least 0.3 s there and cut at its midpoint; if the lecturer never pauses, cut at the quietest frame instead. So every chunk is between 2.5 and 5 minutes except the last, and no cut lands mid-word. Chunks are stored as integer sample ranges, so they tile the signal exactly with no rounding gaps. Each chunk is a NumPy view (a slice, not a copy), and Whisper only ever holds one chunk's decoding state at a time, which bounds its memory regardless of lecture length. One honest limit: the decoded waveform itself is still loaded whole (about 345 MB of float32 for 90 minutes); streaming the decode from ffmpeg would remove that if it ever matters.

**Next up**
Step 1.5: Benchmark script: real-time factor and memory for tiny/base/small models.

---

## 2026-10-24 · Step 1.3: `notetaker transcribe` CLI

**What I did**
- Added a `notetaker transcribe <file>` command that transcribes one recording and writes `transcript.json` and `transcript.md` into `outputs/<file name>/` (or any folder given with `-o`)
- Options: `--model` (choices come straight from the config's allowed Whisper sizes), `--language`, `--beam-size`, and `--title` for the Markdown heading
- Added `src/notetaker/export.py` with `format_timestamp` (761.4 s becomes `00:12:41`), `group_paragraphs`, `to_markdown`, and `write_outputs`
- Clear errors with exit code 1 for a missing file, a bad beam size, a missing ffmpeg, or an unreadable file, instead of a Python traceback
- Added a Usage section to the README and gitignored `outputs/` so generated transcripts are never committed
- Added 21 tests (`tests/test_export.py`, `tests/test_cli_transcribe.py`), including an end-to-end run on a generated tone through the real ffmpeg pipeline with a fake model. `make check` passes (68 tests, lint clean) in a fresh virtualenv

**How it works**
The command is a thin layer: argparse reads the arguments, `transcribe()` from Step 1.2 does the speech recognition, and the export module turns the result into files. Whisper returns sentence-sized segments, so printing one per line would give a 90-minute lecture well over a thousand choppy lines. Instead, consecutive segments are joined into paragraphs, and a new paragraph starts when the gap between one segment's end and the next one's start is at least 2 seconds (a natural pause), or when the paragraph already spans 60 seconds (so a lecturer who never pauses still gets readable blocks). Each paragraph is labelled with its start time, floored to the second, so jumping to that time in a video player lands just before the speech. The JSON file keeps everything (word timestamps, confidence scores) so later steps never need to rerun Whisper. In the tests, pytest's `monkeypatch` temporarily swaps the real `transcribe` function (or just the model loader) for a fake, which lets the whole command run in milliseconds with no model download.

**Next up**
Step 1.4: Chunk long audio (VAD-based) so 90-minute lectures don't blow up memory.

---

## 2026-10-23 · Step 1.2: Transcription with faster-whisper

**What I did**
- Added `src/notetaker/transcribe.py` with `transcribe(audio)`, which takes a file path or a 16 kHz waveform and returns a `Transcript`: a list of `Segment`s (start, end, text, confidence scores), each holding `Word`s with their own timestamps
- Added `load_model()`, which loads the configured Whisper size once and reuses it, and picks the hardware automatically: GPU with float16 if one is present, otherwise CPU with int8
- `Transcript` saves to and loads from JSON (`save_json` / `load_json`), so a 90-minute lecture only has to be transcribed once
- File inputs are always decoded at 16 kHz, even if `NOTETAKER_SAMPLE_RATE` is changed, because Whisper only understands 16 kHz
- Added `faster-whisper` as a dependency and a README note that model weights download on first use and are never committed
- Added 12 tests in `tests/test_transcribe.py` using a fake model, so CI never downloads weights. `make check` passes (47 tests, lint clean). The one `slow` test that runs the real `tiny` model could not run in tonight's build sandbox (Hugging Face downloads are blocked there), so run `pytest -m slow` once locally to confirm the end-to-end path

**How it works**
faster-whisper runs OpenAI's open Whisper weights on CTranslate2, a C++ inference engine that is several times faster than the original PyTorch code. Whisper is an encoder-decoder transformer: the encoder turns 30-second windows of the audio's log-mel spectrogram into feature vectors, and the decoder writes text token by token, also predicting timestamp tokens that mark where each segment starts and ends. Beam search (beam size 5) keeps the 5 most probable partial transcriptions at each step instead of greedily taking the single best token, which usually fixes small mistakes. Word timestamps come from cross-attention: for each output word, the model's attention over the audio frames shows which frames it was "listening to", and aligning those gives each word a start and end time. On CPU the weights are quantized to int8 (an 8-bit integer plus a scale per row), cutting memory about 4x versus float32 for a tiny accuracy cost. For testing I used dependency injection: `transcribe()` accepts any object with a `.transcribe()` method, so the tests hand it a fake model with known outputs and check the conversion exactly, with no download.

**Next up**
Step 1.3: CLI `notetaker transcribe lecture.mp3` that writes `transcript.json` and `transcript.md`.

---

## 2026-10-22 · Step 1.1: Audio loading and normalization

**What I did**
- Added `src/notetaker/audio.py` with `load_audio(path)`, which turns any file ffmpeg can read (mp3, wav, m4a, mp4 video, ...) into a 16 kHz mono float32 NumPy array in [-1, 1), the exact format Whisper expects
- Added helpers: `peak_normalize` (consistent volume for quiet recordings), `duration_seconds`, and `save_wav` (writes 16-bit mono WAV with Python's built-in `wave` module, for caching and for chunking in Step 1.4)
- Added clear error types: `FFmpegNotFoundError` with install instructions, and `AudioLoadError` for missing, corrupt, or non-audio files
- The default sample rate comes from the config module (`NOTETAKER_SAMPLE_RATE`, 16000)
- Added `numpy` as a dependency, an "Install ffmpeg" step in CI, and a Requirements section in the README
- Added 15 tests in `tests/test_audio.py`; no audio is committed, each test synthesizes a half-second tone with ffmpeg in a temp folder. `make check` passes (36 tests, lint clean) in a fresh virtualenv

**How it works**
ffmpeg does the hard part: decoding the file, averaging stereo channels down to mono, resampling to 16 kHz, and writing the result as raw 16-bit integers to its standard output. Python reads those bytes through a pipe (no temp files), reinterprets them as int16 with `np.frombuffer`, and divides by 32768 to get floats in [-1, 1). The tests check the math directly: a 440 Hz sine recorded at 44.1 kHz stereo must come back with 8000 samples (0.5 s x 16000), its FFT peak must still be at 440 Hz (bin width 16000 / 8000 = 2 Hz), and its amplitude must still be 0.5. A clip with sin on the left and -sin on the right must decode to near silence, which proves the channels are averaged rather than one being dropped. Peak normalization multiplies the whole signal by one constant, target / max|x[n]|, so it changes loudness without distorting the waveform.

**Next up**
Step 1.2: Transcribe with faster-whisper (small model), return segments with timestamps.

---

## 2026-10-21 · Step 0.5: Config module and `.env.example`

**What I did**
- Added `src/notetaker/config.py`: one typed `Settings` class (built on pydantic-settings) holding every knob the roadmap needs, grouped by phase: paths, Whisper model/device/precision/language/sample rate, LLM provider/model/temperature, Ollama URL, free-tier API keys, and the embedding model
- Added `.env.example`, a commented template documenting every setting with its default and allowed values
- Added a `notetaker config` command that prints the active settings with API keys masked, and prints a clear error (exit code 1) for invalid values
- Added `pydantic-settings` as the first runtime dependency and a Configuration section to the README
- Added 16 tests in `tests/test_config.py` covering defaults, env vars, `.env`, priority order, invalid values, blank values, secret masking, and a guard test that fails if a setting is ever missing from `.env.example`; `make check` passes (21 tests, lint clean) in a fresh virtualenv

**How it works**
Instead of hard-coding values like "use the small Whisper model" across many files, every module asks `get_settings()` for them. pydantic-settings looks for each value in three places, in priority order: a real environment variable such as `NOTETAKER_WHISPER_MODEL=base`, then a `.env` file, then the default in the code. It also checks each value against its type, so a typo like `smal` or a temperature of 5 fails immediately at startup with a readable message instead of breaking halfway through a long job. Choices like the LLM provider are restricted to a fixed list (`ollama`, `groq`, `huggingface`), which also enforces the free-models-only rule in code. API keys are stored as `SecretStr`, which prints as asterisks, so they cannot leak into logs; the real `.env` is gitignored and only the blank template is committed. `get_settings()` is cached, so the environment is read once and the whole program shares one settings object.

**Next up**
Step 1.1: Audio loading and normalization (ffmpeg, 16 kHz mono) with tests on a tiny generated clip.

---

## 2026-10-20 · Step 0.4: GitHub Actions CI

**What I did**
- Added `.github/workflows/ci.yml`, which runs `make lint` and `make test` on every push to `main`, every pull request, and on demand from the Actions tab
- Tests run on both Python 3.10 (the oldest version the package supports) and 3.12, using a build matrix
- Turned on pip caching, auto-cancel of outdated runs, a 15-minute timeout, and read-only permissions for the workflow
- Added a CI status badge to the top of `README.md`
- Checked locally: the workflow file passes `actionlint`, and `make check` (lint plus all 5 tests) is clean

**How it works**
GitHub Actions is a free service that starts a fresh Linux machine whenever an event like a push happens, then runs the steps listed in the workflow file: download the code, install Python, install the package with `make install`, then run `make lint` and `make test`. If any step fails, the commit gets a red X and the badge turns red, so a broken change is caught right away instead of weeks later. Using the same `make` commands locally and in CI means a passing `make check` on my laptop predicts a passing CI run. The matrix runs the identical job once per Python version, which catches code that accidentally depends on a newer language feature. CI deliberately skips tests marked `slow` and never calls a live LLM or downloads model weights, so it stays fast, free, and does not need any API keys.

**Next up**
Step 0.5: Config module (pydantic-settings) and `.env.example` documenting every setting.

---

## 2026-10-19 · Step 0.3: Tooling (ruff, pytest, Makefile)

**What I did**
- Added `[tool.pytest.ini_options]` and `[tool.ruff]` settings to `pyproject.toml`, so both tools are configured in the same file as the package
- Turned on a wider set of ruff rule families: style (E, W), real bugs (F), import order (I), bug patterns (B), modern syntax (UP), and simplifications (SIM)
- Registered a `slow` pytest marker for future tests that load real models or audio, and made unknown markers an error
- Added a `Makefile` with `make install`, `make lint`, `make format`, `make test`, and `make check` (lint plus tests, the same thing CI will run)
- Added `tests/test_cli.py` with smoke tests for `--version`, a bad flag, and the program name; all 5 tests pass and lint is clean

**How it works**
A linter reads code without running it and flags likely bugs and style problems; ruff does this plus formatting, so one tool replaces three older ones. pytest finds every `test_*.py` file under `tests/` and runs every `test_*` function in it; a test passes if its `assert` statements hold. The `slow` marker lets me tag expensive tests (like running Whisper on real audio) so the default `make test` and CI skip them and stay fast and free. The Makefile is just a list of named shortcuts, so `make check` always runs exactly the same commands locally and in CI, which avoids "works on my machine" surprises.

**Next up**
Step 0.4: GitHub Actions CI (lint + tests on every push).

---

## 2026-10-18 · Step 0.2: Python project setup

**What I did**
- Added `pyproject.toml` declaring the `notetaker` package, Python 3.10+, and a `dev` extra with pytest and ruff
- Created the `src/notetaker/` package with a version string and a minimal `notetaker` CLI (`notetaker --version`)
- Added `requirements.txt` (installs the package in editable mode with dev tools) and a `.gitignore` that keeps out caches, `.env` secrets, audio, and model weights
- Added two smoke tests in `tests/test_package.py`; both pass, and ruff lint and format checks are clean

**How it works**
`pyproject.toml` is the one config file modern Python tools read. It says what the package is called, what it depends on, and that installing it should create a `notetaker` command that runs `notetaker.cli:main`. The code uses the "src layout", meaning it lives in `src/notetaker` instead of the repo root. That forces tests to import the installed package rather than whatever files happen to sit next to them, so packaging mistakes show up early. Installing with `pip install -e ".[dev]"` (editable mode) links the install to the source folder, so edits take effect immediately. The version is read from installed package metadata, so it is defined once in `pyproject.toml`. The `.gitignore` blocks audio, model files, and `.env` so large files and secrets never land in git.

**Next up**
Step 0.3: Tooling (ruff + pytest config in `pyproject.toml`, a `Makefile` with `make test` / `make lint`).

---

## 2026-10-17 · Step 0.1: Repo scaffold

**What I did**
- Wrote `README.md` describing the project, its goals, and the planned stack
- Wrote `ROADMAP.md`: the full project split into ~45 small steps across 9 phases
- Started this dev log

**Why**
The project gets built in small nightly increments, so the plan has to live in the repo itself. Every session reads the roadmap, picks the next step, and logs it here.

**Next up**
Step 0.2: Python project setup (package layout, dependencies, `.gitignore`).
