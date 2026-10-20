# Dev Log

One entry per work session, newest first. Each entry covers what was built, how it works, and what's next.

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
