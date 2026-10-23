# Dev Log

One entry per work session, newest first. Each entry covers what was built, how it works, and what's next.

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
