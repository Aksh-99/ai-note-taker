"""Tests for the LangGraph pipeline (src/notetaker/agent.py).

No live LLM calls. `RoutingFakeClient` answers each request according to
which stage sent it (recognised from the system prompt), so the tests do not
depend on the exact number or order of calls inside each stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest

from notetaker import agent, cli
from notetaker.agent import (
    STEP_ORDER,
    PipelineOptions,
    build_pipeline,
    pipeline_mermaid,
    run_pipeline,
    write_pipeline,
)
from notetaker.llm import LLMResponse, as_messages
from notetaker.transcribe import Segment, Transcript


def make_transcript(spans: list[tuple[float, float, str]]) -> Transcript:
    segments = [
        Segment(id=i, start=s, end=e, text=t, avg_logprob=-0.2, no_speech_prob=0.01)
        for i, (s, e, t) in enumerate(spans)
    ]
    return Transcript(
        segments=segments,
        language="en",
        language_probability=1.0,
        duration=spans[-1][1] if spans else 0.0,
        model="small",
    )


LECTURE = make_transcript(
    [
        (0.0, 10.0, "An eigenvalue is how much a matrix stretches a vector."),
        (15.0, 25.0, "The determinant is zero exactly when the matrix is singular."),
        (30.0, 40.0, "Problem set two is due Friday."),
    ]
)

NOTES_REPLY = json.dumps(
    {
        "sections": [
            {
                "heading": "Eigenvalues",
                "start": "00:00:00",
                "summary": "What an eigenvalue is.",
                "key_points": ["A matrix stretches a vector by its eigenvalue."],
            }
        ]
    }
)
CONCEPTS_REPLY = json.dumps(
    {
        "concepts": [
            {
                "term": "eigenvalue",
                "definition": "the factor by which a matrix stretches a vector",
            }
        ]
    }
)
CARDS_REPLY = json.dumps(
    {
        "cards": [
            {
                "question": "When is the determinant zero?",
                "answer": "Exactly when the matrix is singular.",
            }
        ]
    }
)
QUIZ_REPLY = json.dumps(
    {
        "questions": [
            {
                "question": "When is the determinant zero?",
                "correct": "When the matrix is singular",
                "distractors": ["Always", "When the matrix is square", "Never"],
                "explanation": "Said at 00:00:15.",
            }
        ]
    }
)

TASKS_REPLY = json.dumps(
    {
        "tasks": [
            {
                "excerpt": 1,
                "title": "Problem set two",
                "kind": "homework",
                "details": "",
                "due_phrase": "Friday",
            }
        ]
    }
)

# Words that appear only in each stage's system prompt.
ROUTES = {
    "student to-dos": "tasks",
    "builds glossaries": "concepts",
    "study flashcards": "flashcards",
    "multiple-choice": "quiz",
    "writes study notes": "notes",
}


@dataclass
class RoutingFakeClient:
    """Fake LLM that picks its reply by the stage that is asking."""

    replies: dict[str, str] = field(
        default_factory=lambda: {
            "notes": NOTES_REPLY,
            "concepts": CONCEPTS_REPLY,
            "flashcards": CARDS_REPLY,
            "quiz": QUIZ_REPLY,
            "tasks": TASKS_REPLY,
        }
    )
    model: str = "fake-model"
    provider: str = "fake"
    calls: list[str] = field(default_factory=list)  # stage name per call

    def chat(self, messages, **kwargs: Any) -> LLMResponse:
        messages = list(messages)
        system = messages[0].content if messages else ""
        stage = next((s for key, s in ROUTES.items() if key in system), "other")
        self.calls.append(stage)
        text = self.replies.get(stage, "{}")
        return LLMResponse(text=text, model=self.model, provider=self.provider)

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any):
        return self.chat(as_messages(prompt, system), **kwargs)


NO_REDUCE = PipelineOptions(reduce=False)


def test_cli_step_names_match_the_pipeline():
    # cli.py keeps its own copy so the parser need not import langgraph.
    assert cli.STEP_NAMES == STEP_ORDER


def test_options_validate_and_keep_pipeline_order():
    options = PipelineOptions(steps=("tasks", "notes"))
    assert options.steps == ("notes", "tasks")
    assert options.needs_llm
    # The tasks stage uses the LLM unless told to use offline rules.
    assert PipelineOptions(steps=("tasks",)).needs_llm
    assert not PipelineOptions(steps=("tasks",), task_llm=False).needs_llm
    with pytest.raises(ValueError):
        PipelineOptions(steps=("summarize",))
    with pytest.raises(ValueError):
        PipelineOptions(steps=())
    with pytest.raises(ValueError):
        PipelineOptions(chunk_words=0)


def test_llm_steps_need_a_client():
    with pytest.raises(ValueError):
        build_pipeline(None, PipelineOptions(steps=("notes",)))
    with pytest.raises(ValueError, match="need an LLM client"):
        build_pipeline(None, PipelineOptions(steps=("tasks",)))
    # The offline (rule-based) tasks step builds fine without one.
    build_pipeline(None, PipelineOptions(steps=("tasks",), task_llm=False))


def test_full_pipeline_runs_every_stage_in_order():
    client = RoutingFakeClient()
    started: list[str] = []
    result = run_pipeline(
        LECTURE, client, NO_REDUCE, title="L1", on_step=started.append
    )
    assert result.ok, result.errors
    assert started == list(STEP_ORDER)
    assert [r["step"] for r in result.log] == ["check_transcript", *STEP_ORDER]
    # The stage order shows up in the calls the LLM received.
    assert client.calls == ["notes", "concepts", "flashcards", "quiz", "tasks"]
    assert result.notes.sections[0].heading == "Eigenvalues"
    assert result.concepts.concepts[0].term == "eigenvalue"
    assert result.tasks.candidates[0].time_phrases == ["Friday"]
    assert result.notes.title == "Notes: L1"
    assert result.tasks.tasks[0].title == "Problem set two"
    assert result.tasks.method == "llm"
    assert result.llm_calls == 5
    assert result.provider == "fake"


def test_later_stages_reuse_the_glossary_from_state():
    result = run_pipeline(LECTURE, RoutingFakeClient(), NO_REDUCE)
    # One LLM card plus one "What is eigenvalue?" card from the concepts stage.
    origins = sorted(c.origin for c in result.flashcards.cards)
    assert origins == ["concept", "llm"]
    assert len(result.quiz.questions) >= 1


def test_failed_stage_is_recorded_and_the_rest_still_run():
    client = RoutingFakeClient()
    client.replies["concepts"] = "not json at all"
    result = run_pipeline(LECTURE, client, NO_REDUCE)
    assert not result.ok
    assert [e["step"] for e in result.errors] == ["concepts"]
    assert result.concepts is None
    statuses = {r["step"]: r["status"] for r in result.log}
    assert statuses["concepts"] == "failed"
    assert statuses["flashcards"] == statuses["quiz"] == statuses["tasks"] == "ok"
    # Without a glossary the deck has only LLM cards.
    assert {c.origin for c in result.flashcards.cards} == {"llm"}


def test_empty_transcript_stops_at_the_conditional_edge():
    client = RoutingFakeClient()
    result = run_pipeline(make_transcript([]), client, NO_REDUCE)
    assert client.calls == []
    empty = {"step": "check_transcript", "error": "transcript is empty"}
    assert result.errors == [empty]
    assert result.notes is None and result.tasks is None


def test_subset_of_steps_only_builds_those_nodes():
    client = RoutingFakeClient()
    options = PipelineOptions(steps=("concepts", "quiz"))
    result = run_pipeline(LECTURE, client, options)
    assert result.ok
    assert set(client.calls) == {"concepts", "quiz"}
    assert result.notes is None and result.flashcards is None
    nodes = set(build_pipeline(client, options).get_graph().nodes)
    assert {"concepts", "quiz", "check_transcript"} <= nodes
    assert "notes" not in nodes


def test_mermaid_shows_the_graph():
    text = pipeline_mermaid()
    assert "notes --> concepts" in text
    assert "quiz --> tasks" in text
    assert "check_transcript -." in text  # dotted = conditional edge


def test_write_pipeline_outputs(tmp_path):
    result = run_pipeline(LECTURE, RoutingFakeClient(), NO_REDUCE, title="L1")
    paths = write_pipeline(result, tmp_path)
    names = {p.name for p in paths}
    assert {
        "notes.json",
        "concepts.json",
        "flashcards.csv",
        "quiz.json",
        "tasks.json",
        "todo.json",
        "tasks.ics",
        "pipeline.json",
    } <= names
    summary = json.loads((tmp_path / "pipeline.json").read_text())
    assert summary["steps"] == list(STEP_ORDER)
    assert summary["llm_calls"] == 5
    assert summary["errors"] == []


def _write_transcript(tmp_path, transcript=LECTURE):
    folder = tmp_path / "lecture01"
    return transcript.save_json(folder / "transcript.json")


def test_cli_agent_end_to_end(tmp_path, capsys):
    path = _write_transcript(tmp_path)
    args = cli.build_parser().parse_args(
        ["agent", str(path), "--no-reduce", "--lecture-date", "2026-11-11"]
    )
    code = cli.run_agent(args, client=RoutingFakeClient())
    out = capsys.readouterr().out
    assert code == 0, out
    assert "notes -> concepts -> flashcards -> quiz -> tasks" in out
    assert "Found 1 task(s) in 1 candidate" in out
    # 2026-11-11 is a Wednesday, so "Friday" is the 13th.
    assert "Problem set two (homework): 2026-11-13" in out
    assert (path.parent / "tasks.md").exists()
    summary = json.loads((path.parent / "pipeline.json").read_text())
    assert summary["title"] == "lecture01"


def test_cli_agent_offline_tasks_only(tmp_path, capsys):
    path = _write_transcript(tmp_path)
    # No client and no LLM settings needed for the offline step.
    args = ["agent", str(path), "--steps", "tasks", "--no-task-llm"]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "no LLM needed" in out
    assert "Problem set two (homework)" in out


def test_tasks_stage_gets_lecture_date_and_class_days():
    options = PipelineOptions(
        steps=("tasks",), lecture_date=date(2026, 11, 11), class_days=(0, 2)
    )
    result = run_pipeline(LECTURE, RoutingFakeClient(), options)
    task = result.tasks.tasks[0]
    assert task.due_date == date(2026, 11, 13)
    assert result.tasks.reference_date == date(2026, 11, 11)


def test_cli_agent_rejects_bad_lecture_date(tmp_path, capsys):
    path = _write_transcript(tmp_path)
    assert cli.main(["agent", str(path), "--lecture-date", "Friday"]) == 1
    assert "YYYY-MM-DD" in capsys.readouterr().out


def test_cli_agent_failure_exit_code(tmp_path, capsys):
    path = _write_transcript(tmp_path)
    client = RoutingFakeClient()
    client.replies["quiz"] = "garbage"
    args = cli.build_parser().parse_args(["agent", str(path), "--no-reduce"])
    assert cli.run_agent(args, client=client) == 1
    out = capsys.readouterr().out
    assert "Error in quiz" in out
    # Stages that worked were still written.
    assert (path.parent / "notes.json").exists()


def test_cli_agent_graph_and_errors(tmp_path, capsys):
    assert cli.main(["agent", "--graph"]) == 0
    assert "graph TD" in capsys.readouterr().out
    assert cli.main(["agent"]) == 1
    assert cli.main(["agent", str(tmp_path / "missing.json")]) == 1
    assert "no such file" in capsys.readouterr().out


def test_module_does_not_import_langgraph_at_top_level():
    # langgraph is imported inside build_pipeline, so the agent module (and
    # the CLI) load quickly. The graph classes are not module attributes.
    assert not hasattr(agent, "StateGraph")
