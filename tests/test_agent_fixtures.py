"""End-to-end agent tests that replay recorded LLM replies (Step 4.4).

Everything lives in tests/fixtures/agent/ (see the README there):

    transcript.json   a 3.5 minute lecture on eigenvalues, with homework,
                      a reading, a midterm and one false alarm ("due to Cauchy")
    options.json      the pipeline settings for the fixture run
    cassette.json     every LLM request the run makes, with the reply it got
    expected/         the files `notetaker agent` should write ("golden files")

The main test replays the cassette through the real pipeline and compares
every output file with expected/ byte for byte. That catches any change in
behaviour, from a parsing tweak to a reworded Markdown heading. If the
change is intended, refresh the golden files and review the diff:

    UPDATE_GOLDEN=1 pytest tests/test_agent_fixtures.py
    git diff tests/fixtures/agent/expected/

No network, no API key, no model: the conftest guard would fail the test if
anything tried to reach a live LLM.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

import notetaker.notes
from notetaker import cli
from notetaker.agent import STEP_ORDER, run_pipeline, write_pipeline
from notetaker.cassette import Cassette, CassetteMiss, ReplayClient
from notetaker.llm import FakeLLMClient
from notetaker.transcribe import Transcript

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "agent"
EXPECTED = FIXTURES / "expected"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"


def _load_recorder_script():
    """Import scripts/record_agent_fixture.py as a module.

    scripts/ is not a package, so a normal `import` cannot find it.
    importlib loads a module straight from a file path instead.
    """
    path = ROOT / "scripts" / "record_agent_fixture.py"
    spec = importlib.util.spec_from_file_location("record_agent_fixture", path)
    module = importlib.util.module_from_spec(spec)
    # Register it before running it: @dataclass looks the module up in
    # sys.modules while the file executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


recorder_script = _load_recorder_script()
TITLE, OPTIONS = recorder_script.load_options(FIXTURES)


def stable_summary(text: str) -> str:
    """pipeline.json minus wall-clock timings, which change every run."""
    summary = json.loads(text)
    summary.pop("seconds", None)
    for record in summary["log"]:
        record.pop("seconds", None)
    return json.dumps(summary, indent=2, ensure_ascii=False) + "\n"


def replay_run(cassette_path: Path = FIXTURES / "cassette.json"):
    client = ReplayClient(cassette_path)
    transcript = Transcript.load_json(FIXTURES / "transcript.json")
    result = run_pipeline(transcript, client, OPTIONS, title=TITLE)
    return result, client


# iCalendar requires a DTSTAMP (the moment the file was written) on every
# event, so tasks.ics differs on every run. Swap it for a fixed placeholder.
DTSTAMP = re.compile(r"^DTSTAMP:\d{8}T\d{6}Z\r$", re.MULTILINE)


def read_outputs(folder: Path, *, provider: str | None = None) -> dict[str, str]:
    """Every output file as text, with run-to-run noise removed.

    `provider`, if given, is replaced by "replay:<provider>" so a live run
    and its replay can be compared (files mention which client wrote them).
    """
    files = {}
    for path in sorted(folder.iterdir()):
        # newline="" keeps \r\n line endings (tasks.ics needs them) intact.
        text = path.read_text(encoding="utf-8", newline="")
        if path.name == "pipeline.json":
            text = stable_summary(text)
        text = DTSTAMP.sub("DTSTAMP:20000101T000000Z\r", text)
        if provider is not None:
            text = text.replace(f'"{provider}"', f'"replay:{provider}"')
            text = text.replace(f"by {provider} ", f"by replay:{provider} ")
        files[path.name] = text
    return files


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    """Run the replayed pipeline once and share it across the tests below."""
    result, client = replay_run()
    out = tmp_path_factory.mktemp("agent_out")
    write_pipeline(result, out)
    return result, client, out


# ---- the golden-file test -------------------------------------------------------


def test_outputs_match_golden_files(replayed):
    result, _, out = replayed
    actual = read_outputs(out)
    if UPDATE:
        EXPECTED.mkdir(exist_ok=True)
        for old in EXPECTED.iterdir():
            old.unlink()
        for name, text in actual.items():
            (EXPECTED / name).write_text(text, encoding="utf-8", newline="")
        pytest.skip("golden files rewritten; review `git diff` and commit")
    expected = read_outputs(EXPECTED)
    assert sorted(actual) == sorted(expected), "different set of output files"
    for name in expected:
        # One assert per file, so a failure names the file that changed.
        assert actual[name] == expected[name], (
            f"{name} changed. If intended: UPDATE_GOLDEN=1 pytest {__file__}"
        )


# ---- what the recorded run should show -----------------------------------------


def test_replay_uses_every_recorded_call_exactly(replayed):
    result, client, _ = replayed
    assert result.ok, result.errors
    assert client.unused() == [], "cassette has calls the pipeline no longer makes"
    assert len(client.calls) == len(client.cassette.interactions) == result.llm_calls
    assert [r["step"] for r in result.log] == ["check_transcript", *STEP_ORDER]


def test_notes_went_through_repair_and_map_reduce(replayed):
    result, _, _ = replayed
    notes = result.notes
    # Chunk 1's first reply had prose before the JSON, so it was repaired:
    # 2 map calls + 1 repair + 1 reduce call.
    assert notes.llm_calls == 4
    assert [s.heading for s in notes.sections] == [
        "Eigenvalues: definition and how to find them",
        "Worked example and sanity checks",
    ]
    assert notes.overview.startswith("The lecture introduces eigenvalues")


def test_tasks_resolve_dates_and_skip_the_false_alarm(replayed):
    result, _, _ = replayed
    tasks = {t.title: t for t in result.tasks.tasks}
    assert sorted(tasks) == ["Midterm", "Problem set 4", "Read sections 6.1 and 6.2"]
    # Lecture on Friday 2026-11-13: Monday is the 16th, "next Wednesday" the
    # 18th (flagged ambiguous), and the midterm keeps its explicit date.
    assert str(tasks["Read sections 6.1 and 6.2"].due_date) == "2026-11-16"
    pset = tasks["Problem set 4"]
    assert str(pset.due_date) == "2026-11-18" and pset.date_ambiguous
    assert str(tasks["Midterm"].due_date) == "2026-11-24"
    # The duplicate "Problem set 4" from the overlapping excerpt was merged.
    assert len(result.tasks.tasks) == 3


def test_glossary_from_state_reaches_flashcards(replayed):
    result, _, _ = replayed
    origins = {card.origin for card in result.flashcards.cards}
    assert origins == {"llm", "concept"}


def test_calendar_file_is_valid_icalendar(replayed):
    icalendar = pytest.importorskip("icalendar")
    _, _, out = replayed
    calendar = icalendar.Calendar.from_ical((out / "tasks.ics").read_bytes())
    summaries = sorted(str(e["SUMMARY"]) for e in calendar.walk("VEVENT"))
    assert summaries == [
        "Due: Midterm",
        "Due: Problem set 4 (?)",
        "Due: Read sections 6.1 and 6.2",
    ]


# ---- stale cassettes fail loudly ------------------------------------------------


def test_changed_prompt_raises_cassette_miss(monkeypatch):
    # Simulate someone editing the notes prompt without re-recording.
    reworded = notetaker.notes.SYSTEM_PROMPT.replace("careful", "meticulous")
    monkeypatch.setattr(notetaker.notes, "SYSTEM_PROMPT", reworded)
    with pytest.raises(CassetteMiss, match="re-record"):
        replay_run()


def test_cassette_records_where_it_came_from():
    cassette = Cassette.load(FIXTURES / "cassette.json")
    assert cassette.note, "cassette should say how it was recorded"
    assert cassette.provider in {"scripted", "ollama", "groq", "huggingface"}


def test_stage_markers_match_every_system_prompt():
    # The scripted client routes by these phrases; each stage's real system
    # prompt must contain exactly one of them.
    import notetaker.concepts
    import notetaker.flashcards
    import notetaker.mapreduce
    import notetaker.quiz
    import notetaker.tasks

    prompts = {
        "notes": notetaker.notes.SYSTEM_PROMPT,
        "reduce": notetaker.mapreduce.REDUCE_SYSTEM_PROMPT,
        "concepts": notetaker.concepts.SYSTEM_PROMPT,
        "flashcards": notetaker.flashcards.SYSTEM_PROMPT,
        "quiz": notetaker.quiz.SYSTEM_PROMPT,
        "tasks": notetaker.tasks.SYSTEM_PROMPT,
    }
    for stage, prompt in prompts.items():
        assert recorder_script.stage_of(prompt) == stage


def test_scripted_recording_reproduces_the_committed_cassette(tmp_path):
    committed = Cassette.load(FIXTURES / "cassette.json")
    if committed.provider != "scripted":
        pytest.skip("cassette was recorded from a live model")
    for name in ("transcript.json", "options.json", "scripted_replies.json"):
        (tmp_path / name).write_bytes((FIXTURES / name).read_bytes())
    assert recorder_script.main(["--scripted", "--fixture-dir", str(tmp_path)]) == 0
    fresh = (tmp_path / "cassette.json").read_text(encoding="utf-8")
    assert fresh == (FIXTURES / "cassette.json").read_text(encoding="utf-8")


# ---- the CLI flags ----------------------------------------------------------------


def _cli_args(out: Path, *extra: str) -> list[str]:
    return [
        "agent",
        str(FIXTURES / "transcript.json"),
        "-o",
        str(out),
        "--title",
        TITLE,
        "--chunk-words",
        str(OPTIONS.chunk_words),
        "--lecture-date",
        str(OPTIONS.lecture_date),
        "--class-days",
        "mon",
        "wed",
        "fri",
        *extra,
    ]


def test_cli_replay_reproduces_the_golden_files(tmp_path, capsys):
    out = tmp_path / "out"
    code = cli.main(_cli_args(out, "--replay-llm", str(FIXTURES / "cassette.json")))
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert "replay:" in printed and "never replayed" not in printed
    assert read_outputs(out) == read_outputs(EXPECTED)


def test_cli_record_then_replay_round_trip(tmp_path, capsys):
    cassette = tmp_path / "rec.json"
    replies = json.loads((FIXTURES / "scripted_replies.json").read_text())
    replies.pop("_about")
    client = recorder_script.ScriptedClient(replies)
    args = cli.build_parser().parse_args(
        _cli_args(tmp_path / "a", "--record-llm", str(cassette))
    )
    assert cli.run_agent(args, client=client) == 0
    assert "Recorded 11 LLM call(s)" in capsys.readouterr().out
    # Replaying our own recording gives the same files as the live run.
    assert cli.main(_cli_args(tmp_path / "b", "--replay-llm", str(cassette))) == 0
    live = read_outputs(tmp_path / "a", provider="scripted")
    assert live == read_outputs(tmp_path / "b")


def test_cli_replay_with_stale_cassette_exits_1(tmp_path, capsys):
    # A cassette holding one unrelated call: the very first request misses.
    recorder = recorder_script.RecordingClient(FakeLLMClient(replies=["{}"]))
    recorder.complete("unrelated")
    stale = recorder.save(tmp_path / "stale.json")
    assert cli.main(_cli_args(tmp_path / "o", "--replay-llm", str(stale))) == 1
    assert "no recorded reply" in capsys.readouterr().out


def test_cli_replay_missing_or_broken_cassette(tmp_path, capsys):
    assert cli.main(_cli_args(tmp_path, "--replay-llm", str(tmp_path / "nope"))) == 1
    assert "no such cassette" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert cli.main(_cli_args(tmp_path, "--replay-llm", str(bad))) == 1
    assert "not a version 1 cassette" in capsys.readouterr().out


def test_cli_record_and_replay_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["agent", "t.json", "--record-llm", "a", "--replay-llm", "b"]
        )
