# A Makefile gives short, memorable commands for common tasks.
# Usage: `make install`, `make lint`, `make test`, `make check`.
# Each "target" below is a name, a colon, then indented shell commands.
# (Make requires a real TAB character before each command line.)

# .PHONY tells make these targets are commands, not files to build.
.PHONY: install lint format test check record-fixtures

# Install the package in editable mode plus dev tools (pytest, ruff).
install:
	pip install -e ".[dev]"

# Lint (find problems) and verify formatting without changing files.
lint:
	ruff check .
	ruff format --check .

# Auto-fix what can be fixed: sort imports, reformat code.
format:
	ruff check --fix .
	ruff format .

# Run the test suite, skipping tests marked slow (real models or audio).
test:
	pytest -m "not slow"

# Everything CI runs, in one command. Run this before every push.
check: lint test

# Re-seed the agent test cassette and refresh the golden output files
# (Step 4.4). Uses the offline stand-in replies; to capture a real free
# model instead run: python scripts/record_agent_fixture.py --provider ollama
# Review `git diff tests/fixtures/agent/` before committing.
record-fixtures:
	python scripts/record_agent_fixture.py --scripted
	UPDATE_GOLDEN=1 pytest tests/test_agent_fixtures.py -q
