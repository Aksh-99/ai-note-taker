"""Static checks on the fine-tuning notebook (Step 6.2b).

CI cannot run the notebook (it needs a GPU and downloads Whisper and the
lectures), so these tests catch the mistakes that would otherwise only show
up on Colab: broken JSON, Python syntax errors, `notetaker` commands with
flags that do not exist, committed outputs, and anything that looks like a
secret.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
from pathlib import Path

import pytest

from notetaker.cli import build_parser

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks" / ("finetune_whisper_small.ipynb")
)


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells(notebook: dict) -> list[str]:
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def all_text(notebook: dict) -> str:
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_notebook_is_valid_nbformat_4(notebook: dict) -> None:
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))  # nbformat 4.5 requires unique ids
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert isinstance(cell["source"], list)


def test_notebook_is_committed_without_outputs(notebook: dict) -> None:
    # Outputs can leak paths, tokens or huge logs into git; keep it clean.
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_python_in_code_cells_parses(notebook: dict) -> None:
    for index, source in enumerate(code_cells(notebook)):
        # Lines starting with `!` are shell commands (IPython syntax), not
        # Python; blank them out but keep the line numbers.
        python = "\n".join(
            "" if line.lstrip().startswith(("!", "%")) else line
            for line in source.splitlines()
        )
        try:
            ast.parse(python)
        except SyntaxError as error:  # pragma: no cover - message on failure
            pytest.fail(f"code cell {index}: {error}")


def notetaker_commands(notebook: dict) -> list[str]:
    commands = []
    for source in code_cells(notebook):
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("!notetaker "):
                commands.append(stripped.removeprefix("!notetaker "))
    return commands


def test_every_notetaker_command_parses(notebook: dict) -> None:
    commands = notetaker_commands(notebook)
    assert len(commands) >= 5  # evalset, finetune-data x2, finetune x2
    parser = build_parser()
    for command in commands:
        # IPython replaces {NAME} with the Python value before running the
        # line. Substitute a dummy path; {ONLY} becomes a real --only list.
        filled = command.replace("{ONLY}", "--only mit-18.06-lec01")
        filled = re.sub(r"\{[^}]+\}", "/tmp/x", filled)
        try:
            parser.parse_args(shlex.split(filled))
        except SystemExit:  # argparse exits on unknown flags
            pytest.fail(f"notetaker does not accept: {command}")


def test_notebook_covers_the_whole_pipeline(notebook: dict) -> None:
    commands = " ".join(notetaker_commands(notebook))
    for step in (
        "evalset",
        "finetune-data",
        "finetune ",
        "finetune-eval",
        "export-ct2",
    ):
        assert step in commands
    text = all_text(notebook)
    for name in ("load_finetuned", "transcribe_clips", "disable_adapter"):
        assert name in text
    assert "make_archive" in text  # the adapter is saved at the end


def test_notebook_has_no_secrets_or_paid_apis(notebook: dict) -> None:
    text = all_text(notebook)
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)  # Hugging Face token
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", text)  # OpenAI-style key
    assert not re.search(r"gsk_[A-Za-z0-9]{20,}", text)  # Groq key
    assert "openai" not in text.lower().replace("openai/whisper", "")


def test_prose_has_no_em_dashes(notebook: dict) -> None:
    em_dash = chr(0x2014)  # written as a code point so this file has none
    assert em_dash not in all_text(notebook)
