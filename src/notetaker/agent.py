"""The whole lecture pipeline as a LangGraph graph (Step 4.1).

Until now every stage was its own CLI command: `notetaker notes`, then
`notetaker concepts`, then `notetaker flashcards --concepts concepts.json`,
and so on, with the user passing files between them by hand. This module
wires the stages into one pipeline:

    check_transcript -> notes -> concepts -> flashcards -> quiz -> tasks

What LangGraph adds
-------------------
LangGraph (free, open source, runs locally) describes a program as a graph:

* **State**: one dictionary that flows through the graph. Here it holds the
  transcript and every artifact produced so far (`PipelineState` below).
* **Nodes**: plain Python functions `state -> partial update`. A node never
  edits the state in place; it returns only the keys it changed, and
  LangGraph merges them in. That makes each stage easy to test on its own.
* **Edges**: which node runs next. A *conditional edge* calls a small
  "router" function to choose the next node at run time; we use one to stop
  early when the transcript is empty.

For a straight line of five steps a for-loop would do. The graph pays off in
the next steps: Step 4.2 adds a deadline-extraction node, Phase 5 adds
retrieval, and later we can add loops ("faithfulness too low -> regenerate")
or run independent branches in parallel by adding edges, without rewriting
the control flow. `notetaker agent --graph` also prints the graph as a
Mermaid diagram, which renders on GitHub, so the README shows the real
pipeline instead of a hand-drawn picture that can drift out of date.

Design choices
--------------
* **Sequential, not parallel.** flashcards and quiz do not depend on each
  other and could run side by side, but free tiers have tight rate limits
  (Groq allows about 30 requests a minute), and a local Ollama model serves
  one request at a time anyway. A straight line is also deterministic, which
  keeps the tests simple.
* **Fail soft.** If one stage fails (an LLM returns garbage three times, a
  free-tier quota runs out), the error is recorded in the state and the
  remaining stages still run. Flashcards without a glossary are still
  useful; losing all outputs because the quiz failed is not.
* **Dependencies by closure.** Nodes receive only the state, but they also
  need the LLM client and the options. `build_pipeline` defines the node
  functions *inside* itself, so they can read `client` and `options` from
  the enclosing function (a "closure"). This is a simple form of dependency
  injection: tests pass a `FakeLLMClient` and nothing touches the network.
"""

from __future__ import annotations

import json
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from notetaker.concepts import ConceptList, ConceptsError, extract_concepts
from notetaker.concepts import write_concepts as _write_concepts
from notetaker.flashcards import (
    FlashcardDeck,
    FlashcardsError,
    generate_flashcards,
    write_flashcards,
)
from notetaker.llm import LLMError
from notetaker.notes import Notes, NotesError, generate_notes, write_notes
from notetaker.quiz import Quiz, QuizError, generate_quiz, write_quiz
from notetaker.tasks import TaskList, extract_tasks, write_tasks
from notetaker.transcribe import Transcript

# The stages in the order they run. A tuple (not a list) because it is a
# constant that nothing should modify.
STEP_ORDER: tuple[str, ...] = ("notes", "concepts", "flashcards", "quiz", "tasks")

# Stages that call the LLM. "tasks" is an offline keyword scan in Step 4.1
# (the LLM half of deadline extraction is Step 4.2).
LLM_STEPS = frozenset({"notes", "concepts", "flashcards", "quiz"})

# Errors a stage may raise that mean "this stage failed, keep going". Any
# other exception is a real bug and should crash loudly, not be hidden.
STAGE_ERRORS: tuple[type[Exception], ...] = (
    LLMError,
    NotesError,
    ConceptsError,
    FlashcardsError,
    QuizError,
    ValueError,
)


# ---------------------------------------------------------------------------
# State and options
# ---------------------------------------------------------------------------


class PipelineState(TypedDict, total=False):
    """The dictionary that flows through the graph.

    A `TypedDict` is a normal dict with declared keys and types, so editors
    and type checkers know what is inside. `total=False` means a key may be
    missing (e.g. `quiz` before the quiz node has run).

    `Annotated[list, operator.add]` attaches a *reducer*: when a node returns
    `{"log": [record]}`, LangGraph appends it to the existing list (old + new)
    instead of replacing the list. Keys without a reducer are overwritten.
    """

    transcript: Transcript
    title: str
    notes: Notes | None
    concepts: ConceptList | None
    flashcards: FlashcardDeck | None
    quiz: Quiz | None
    tasks: TaskList | None
    log: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[dict[str, str]], operator.add]


@dataclass
class PipelineOptions:
    """Knobs for one pipeline run. Defaults match the single-stage commands."""

    steps: tuple[str, ...] = STEP_ORDER
    chunk_words: int = 1200
    max_repairs: int = 1
    reduce: bool = True  # merge chunk notes into topics (Step 3.3)
    max_cards: int = 6
    max_questions: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        # __post_init__ runs right after the dataclass's generated __init__,
        # a good place to validate. Keep the canonical order whatever order
        # the user typed, and reject unknown names early.
        unknown = [s for s in self.steps if s not in STEP_ORDER]
        if unknown:
            raise ValueError(
                f"unknown step(s) {unknown}; choose from {', '.join(STEP_ORDER)}"
            )
        if not self.steps:
            raise ValueError("choose at least one step")
        self.steps = tuple(s for s in STEP_ORDER if s in self.steps)
        if self.chunk_words <= 0 or self.max_repairs < 0:
            raise ValueError("chunk_words must be > 0 and max_repairs >= 0")

    @property
    def needs_llm(self) -> bool:
        return any(s in LLM_STEPS for s in self.steps)


@dataclass
class PipelineResult:
    """The final state, unpacked into named fields for convenience."""

    title: str
    steps: tuple[str, ...]
    notes: Notes | None = None
    concepts: ConceptList | None = None
    flashcards: FlashcardDeck | None = None
    quiz: Quiz | None = None
    tasks: TaskList | None = None
    log: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    model: str | None = None
    provider: str | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def llm_calls(self) -> int:
        return sum(r.get("llm_calls", 0) for r in self.log)

    def summary(self) -> dict[str, Any]:
        """A small JSON-ready record of the run (written as pipeline.json)."""
        return {
            "title": self.title,
            "steps": list(self.steps),
            "provider": self.provider,
            "model": self.model,
            "llm_calls": self.llm_calls,
            "seconds": round(sum(r["seconds"] for r in self.log), 3),
            "log": self.log,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Building the graph
# ---------------------------------------------------------------------------


def _calls(artifact: Any) -> int:
    """LLM calls an artifact reports (Notes, ConceptList, ... all have one)."""
    return int(getattr(artifact, "llm_calls", 0) or 0)


def build_pipeline(
    client: Any,
    options: PipelineOptions | None = None,
    *,
    on_step: Callable[[str], None] | None = None,
):
    """Build and compile the graph. Returns a LangGraph `CompiledStateGraph`.

    `client` is any object with the LLMClient `chat()` method (a real client
    or `FakeLLMClient`); it may be None when only offline steps are chosen.
    `on_step(name)` is called as each stage starts, for CLI progress lines.
    """
    # Imported here so `import notetaker` stays fast and the rest of the
    # package works even if langgraph is not installed.
    from langgraph.graph import END, START, StateGraph

    options = options or PipelineOptions()
    if options.needs_llm and client is None:
        raise ValueError(f"steps {list(options.steps)} need an LLM client")

    # --- the stage functions ------------------------------------------------
    # Each takes the state and returns the new artifact. They read `client`
    # and `options` from build_pipeline's scope (the closure described at the
    # top of the file).

    def run_notes(state: PipelineState) -> Notes:
        return generate_notes(
            state["transcript"],
            client,
            title=f"Notes: {state['title']}",
            max_words=options.chunk_words,
            max_repairs=options.max_repairs,
            reduce=options.reduce,
        )

    def run_concepts(state: PipelineState) -> ConceptList:
        return extract_concepts(
            state["transcript"],
            client,
            title=f"Key concepts: {state['title']}",
            max_words=options.chunk_words,
            max_repairs=options.max_repairs,
        )

    def run_flashcards(state: PipelineState) -> FlashcardDeck:
        # .get(): the concepts stage may be switched off or may have failed.
        # Then the deck is built from the transcript alone.
        return generate_flashcards(
            state["transcript"],
            client,
            concepts=state.get("concepts"),
            title=f"Flashcards: {state['title']}",
            max_words=options.chunk_words,
            max_cards=options.max_cards,
            max_repairs=options.max_repairs,
        )

    def run_quiz(state: PipelineState) -> Quiz:
        return generate_quiz(
            state["transcript"],
            client,
            concepts=state.get("concepts"),
            title=f"Quiz: {state['title']}",
            max_words=options.chunk_words,
            max_questions=options.max_questions,
            max_repairs=options.max_repairs,
            seed=options.seed,
        )

    def run_tasks(state: PipelineState) -> TaskList:
        return extract_tasks(state["transcript"], title=f"Tasks: {state['title']}")

    stages: dict[str, Callable[[PipelineState], Any]] = {
        "notes": run_notes,
        "concepts": run_concepts,
        "flashcards": run_flashcards,
        "quiz": run_quiz,
        "tasks": run_tasks,
    }

    def as_node(name: str, stage: Callable[[PipelineState], Any]):
        """Wrap a stage into a LangGraph node: time it, log it, catch errors.

        This is the "decorator pattern" done by hand: a function that takes a
        function and returns a new function with extra behaviour around it.
        Writing the timing and error handling once here keeps every stage
        function above a plain one-liner.
        """

        def node(state: PipelineState) -> dict[str, Any]:
            if on_step is not None:
                on_step(name)
            began = time.perf_counter()
            try:
                artifact = stage(state)
            except STAGE_ERRORS as error:
                seconds = time.perf_counter() - began
                return {
                    name: None,
                    "log": [{"step": name, "status": "failed", "seconds": seconds}],
                    "errors": [
                        {"step": name, "error": f"{type(error).__name__}: {error}"}
                    ],
                }
            seconds = time.perf_counter() - began
            record = {
                "step": name,
                "status": "ok",
                "seconds": round(seconds, 3),
                "llm_calls": _calls(artifact),
            }
            return {name: artifact, "log": [record]}

        return node

    def check_transcript(state: PipelineState) -> dict[str, Any]:
        """First node: refuse an empty transcript before spending LLM calls."""
        words = len(state["transcript"].text.split())
        if words == 0:
            return {
                "log": [{"step": "check_transcript", "status": "empty", "seconds": 0}],
                "errors": [
                    {"step": "check_transcript", "error": "transcript is empty"}
                ],
            }
        record = {"step": "check_transcript", "status": "ok", "seconds": 0}
        return {"log": [record]}

    def route_after_check(state: PipelineState) -> str:
        """Router for the conditional edge: continue, or stop at END."""
        return "stop" if state.get("errors") else "continue"

    # --- wiring -------------------------------------------------------------
    graph = StateGraph(PipelineState)
    graph.add_node("check_transcript", check_transcript)
    for name in options.steps:
        graph.add_node(name, as_node(name, stages[name]))

    graph.add_edge(START, "check_transcript")
    # The mapping turns the router's return value into the next node's name.
    graph.add_conditional_edges(
        "check_transcript",
        route_after_check,
        {"continue": options.steps[0], "stop": END},
    )
    # Chain the chosen steps: zip pairs each step with the one after it.
    # strict=False because the second list is one shorter on purpose.
    for current, following in zip(options.steps, options.steps[1:], strict=False):
        graph.add_edge(current, following)
    graph.add_edge(options.steps[-1], END)
    return graph.compile()


def pipeline_mermaid(options: PipelineOptions | None = None) -> str:
    """The graph as Mermaid text, without needing an LLM client."""
    options = options or PipelineOptions()
    # A dummy client is fine: building the graph never calls it.
    graph = build_pipeline(object(), options)
    return graph.get_graph().draw_mermaid()


# ---------------------------------------------------------------------------
# Running and saving
# ---------------------------------------------------------------------------


def run_pipeline(
    transcript: Transcript,
    client: Any,
    options: PipelineOptions | None = None,
    *,
    title: str = "lecture",
    on_step: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Run the whole graph on one transcript and return the final state."""
    options = options or PipelineOptions()
    graph = build_pipeline(client, options, on_step=on_step)
    initial: PipelineState = {
        "transcript": transcript,
        "title": title,
        "log": [],
        "errors": [],
    }
    final = graph.invoke(initial)
    return PipelineResult(
        title=title,
        steps=options.steps,
        notes=final.get("notes"),
        concepts=final.get("concepts"),
        flashcards=final.get("flashcards"),
        quiz=final.get("quiz"),
        tasks=final.get("tasks"),
        log=final.get("log", []),
        errors=final.get("errors", []),
        model=getattr(client, "model", None) if options.needs_llm else None,
        provider=getattr(client, "provider", None) if options.needs_llm else None,
    )


def write_pipeline(
    result: PipelineResult,
    out_dir: str | Path,
    *,
    deck_name: str | None = None,
) -> list[Path]:
    """Write every artifact the run produced, plus pipeline.json.

    Uses the same writers as the single-stage commands, so `notetaker agent`
    and `notetaker notes` give byte-identical notes.json files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if result.notes is not None:
        paths.extend(write_notes(result.notes, out_dir))
    if result.concepts is not None:
        paths.extend(_write_concepts(result.concepts, out_dir))
    if result.flashcards is not None:
        paths.extend(write_flashcards(result.flashcards, out_dir, deck_name=deck_name))
    if result.quiz is not None:
        paths.extend(write_quiz(result.quiz, out_dir))
    if result.tasks is not None:
        paths.extend(write_tasks(result.tasks, out_dir))
    summary_path = out_dir / "pipeline.json"
    summary_path.write_text(
        json.dumps(result.summary(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths.append(summary_path)
    return paths
