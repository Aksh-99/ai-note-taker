"""Re-record the agent test cassette (tests/fixtures/agent/cassette.json).

Run this when a prompt changes (the replay tests then fail with a
`CassetteMiss`) or to swap the replies for a different free model:

    # with a local Ollama (free, no key)
    python scripts/record_agent_fixture.py --provider ollama --model llama3.1:8b

    # or a hosted free tier (needs GROQ_API_KEY or HF_TOKEN in .env)
    python scripts/record_agent_fixture.py --provider groq

    # offline: replay the hand-written stand-in replies instead of a model
    python scripts/record_agent_fixture.py --scripted

Then refresh the expected outputs and review the diff before committing:

    UPDATE_GOLDEN=1 pytest tests/test_agent_fixtures.py
    git diff tests/fixtures/agent/

`make record-fixtures` runs both steps with the --scripted replies.

This script is never run in CI: recording is the one moment we *want* live
LLM calls. Everything after that replays from the file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from notetaker.agent import PipelineOptions, run_pipeline
from notetaker.cassette import RecordingClient
from notetaker.config import get_settings
from notetaker.llm import LLMResponse, as_messages, make_client
from notetaker.transcribe import Transcript

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "agent"

# Words that appear only in one stage's system prompt, used by the scripted
# client to tell which stage is asking. Checked in tests/test_agent_fixtures.py.
STAGE_MARKERS: dict[str, str] = {
    "student to-dos": "tasks",
    "organizes study notes": "reduce",
    "builds glossaries": "concepts",
    "study flashcards": "flashcards",
    "multiple-choice": "quiz",
    "writes study notes": "notes",
}


def stage_of(system_prompt: str) -> str:
    """Name of the stage whose system prompt this is ("other" if unknown)."""
    return next(
        (s for key, s in STAGE_MARKERS.items() if key in system_prompt), "other"
    )


@dataclass
class ScriptedClient:
    """Stand-in model: hands out pre-written replies, one queue per stage.

    Used to seed the cassette when no real model is reachable. A real
    recording (Ollama, Groq, HF) is better, because it captures how an
    actual open model formats its answers.
    """

    replies: dict[str, list[str]]
    model: str = "scripted-stand-in"
    provider: str = "scripted"
    used: dict[str, int] = field(default_factory=dict)

    def chat(self, messages, **kwargs: Any) -> LLMResponse:
        messages = list(messages)
        stage = stage_of(messages[0].content if messages else "")
        queue = self.replies.get(stage)
        if not queue:
            raise KeyError(f"no scripted replies for stage {stage!r}")
        index = self.used.get(stage, 0)
        self.used[stage] = index + 1
        text = queue[min(index, len(queue) - 1)]  # repeat the last when used up
        return LLMResponse(text=text, model=self.model, provider=self.provider)

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any):
        return self.chat(as_messages(prompt, system), **kwargs)


def load_options(fixture_dir: Path = FIXTURE_DIR) -> tuple[str, PipelineOptions]:
    """Read options.json: the exact settings the fixture run uses."""
    raw = json.loads((fixture_dir / "options.json").read_text(encoding="utf-8"))
    options = PipelineOptions(
        chunk_words=raw["chunk_words"],
        max_repairs=raw["max_repairs"],
        reduce=raw["reduce"],
        seed=raw["seed"],
        lecture_date=date.fromisoformat(raw["lecture_date"]),
        class_days=tuple(raw["class_days"]),
    )
    return raw["title"], options


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--provider", choices=["ollama", "groq", "huggingface"])
    source.add_argument(
        "--scripted",
        action="store_true",
        help="use the stand-in replies in scripted_replies.json (offline)",
    )
    parser.add_argument("--model", default=None, help="model name for --provider")
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args(argv)

    title, options = load_options(args.fixture_dir)
    transcript = Transcript.load_json(args.fixture_dir / "transcript.json")
    if args.scripted:
        raw = json.loads((args.fixture_dir / "scripted_replies.json").read_text())
        replies = {k: v for k, v in raw.items() if not k.startswith("_")}
        inner: Any = ScriptedClient(replies)
        note = (
            "Seeded from scripted_replies.json (hand-written stand-in replies), "
            "not a live model. Re-record with --provider to capture real output."
        )
    else:
        inner = make_client(get_settings(), provider=args.provider, model=args.model)
        note = f"Recorded live from {inner.provider} · {inner.model}."

    recorder = RecordingClient(inner, args.fixture_dir / "cassette.json", note=note)
    print(f"Recording {transcript.duration:.0f}s lecture with {inner.provider}")
    result = run_pipeline(transcript, recorder, options, title=title)
    path = recorder.save()
    print(f"Wrote {path} ({len(recorder.cassette.interactions)} interactions)")
    for error in result.errors:
        print(f"Warning: stage {error['step']} failed: {error['error']}")
    print("Next: UPDATE_GOLDEN=1 pytest tests/test_agent_fixtures.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
