"""Tests for task extraction (src/notetaker/tasks.py): keyword scan, LLM, rules."""

from __future__ import annotations

import json
from datetime import date

import pytest

from notetaker import cli
from notetaker.llm import FakeLLMClient, LLMError
from notetaker.tasks import (
    Task,
    TaskList,
    extract_tasks,
    find_task_candidates,
    keyword_due_phrase,
    load_tasks,
    merge_tasks,
    phrase_in_text,
    rule_based_task,
    split_sentences,
    validate_tasks_reply,
    write_tasks,
)
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
        (0.0, 10.0, "Today we study eigenvalues. They describe stretching."),
        # One sentence split across two Whisper segments.
        (12.0, 20.0, "Remember that problem set three is"),
        (20.0, 25.0, "due Friday at 5pm. Questions go on Piazza."),
        (30.0, 40.0, "The residue theorem is not on the midterm."),
        (45.0, 55.0, "We split the data into a training set and a test set."),
    ]
)


def test_sentences_get_the_start_of_their_first_segment():
    sentences = split_sentences(LECTURE)
    texts = [s.text for s in sentences]
    assert texts[2] == "Remember that problem set three is due Friday at 5pm."
    assert sentences[2].start == 12.0
    # "Questions go on Piazza." begins inside the segment that starts at 20 s.
    assert sentences[3].start == 20.0
    assert sentences[0].start == sentences[1].start == 0.0


def test_scan_finds_deadline_with_cues_time_and_context():
    found = find_task_candidates(LECTURE)
    first = found[0]
    assert first.start == 12.0
    assert set(first.cues) == {"due", "problem set"}
    assert first.time_phrases == ["Friday", "5pm"]
    assert first.score == 3.0
    # Context holds the neighbouring sentences.
    assert "They describe stretching." in first.context
    assert "Questions go on Piazza." in first.context


def test_word_boundaries_avoid_false_matches():
    found = find_task_candidates(LECTURE)
    texts = [c.text for c in found]
    # "residue" contains "due" but is not a match; "midterm" is.
    midterm = [c for c in found if "midterm" in c.text][0]
    assert midterm.cues == ["exam"]
    # "test set" in an ML lecture is not an exam.
    assert not any("training set" in t for t in texts)
    assert len(found) == 2


def test_no_candidates_and_empty_transcript():
    calm = make_transcript([(0.0, 5.0, "Eigenvalues are lovely.")])
    assert find_task_candidates(calm) == []
    assert find_task_candidates(make_transcript([])) == []
    md = extract_tasks(calm, title="T").to_markdown()
    assert "No homework" in md


def test_write_and_load_roundtrip(tmp_path):
    tasks = extract_tasks(LECTURE, title="Tasks: L1")
    json_path, md_path = write_tasks(tasks, tmp_path)
    loaded = load_tasks(json_path)
    assert isinstance(loaded, TaskList)
    assert loaded == tasks
    md = md_path.read_text()
    assert "[00:00:12]" in md and "Friday" in md


# ---------------------------------------------------------------------------
# Step 4.2: LLM extraction, rule fallback, date resolution
# ---------------------------------------------------------------------------

WED = date(2026, 11, 11)

# Candidates in LECTURE, in order:
#   1. "Remember that problem set three is due Friday at 5pm."   (12 s)
#   2. "The residue theorem is not on the midterm."              (30 s)
#   3. "...a training set and a test set" is NOT a candidate (no cue).
PSET_REPLY = json.dumps(
    {
        "tasks": [
            {
                "excerpt": 1,
                "title": "Problem set 3",
                "kind": "Problem Set",
                "details": "",
                "due_phrase": "Friday at 5pm",
            }
        ]
    }
)


def test_llm_tasks_get_resolved_dates_and_rejections():
    client = FakeLLMClient(replies=[PSET_REPLY])
    tasks = extract_tasks(LECTURE, client, lecture_date=WED)
    assert tasks.method == "llm"
    assert len(client.calls) == 1
    (task,) = tasks.tasks
    assert task.title == "Problem set 3"
    assert task.kind == "homework"  # "Problem Set" mapped by the validator
    assert task.due_date == date(2026, 11, 13)
    assert task.due_time == "17:00"
    assert task.start == 12.0 and task.source == "llm"
    assert tasks.rejected == [2]  # the midterm sentence is not a task
    assert tasks.llm_calls == 1 and tasks.model == "fake-model"
    # The prompt holds numbered excerpts with timestamps and context.
    prompt = client.calls[0][1].content
    assert "1. [00:00:12]" in prompt and "2. [00:00:30]" in prompt


def test_invented_due_phrase_is_replaced_by_keyword_times():
    reply = json.loads(PSET_REPLY)
    reply["tasks"][0]["due_phrase"] = "next Tuesday"  # not in the transcript
    tasks = extract_tasks(LECTURE, FakeLLMClient([json.dumps(reply)]), lecture_date=WED)
    task = tasks.tasks[0]
    assert task.due_phrase == "Friday at 5pm"
    assert task.due_date == date(2026, 11, 13)
    assert "not in transcript" in task.date_rule


def test_repair_loop_then_success():
    client = FakeLLMClient(replies=["not json", PSET_REPLY])
    tasks = extract_tasks(LECTURE, client, lecture_date=WED)
    assert len(client.calls) == 2
    assert "could not be used" in client.calls[1][-1].content
    assert tasks.tasks[0].title == "Problem set 3"
    assert not tasks.warnings


def test_failed_batch_falls_back_to_rules():
    client = FakeLLMClient(replies=["still not json"])
    tasks = extract_tasks(LECTURE, client, lecture_date=WED, max_repairs=1)
    assert len(client.calls) == 2
    assert any("fell back to rules" in w for w in tasks.warnings)
    (task,) = tasks.tasks  # the deadline is not lost
    assert task.source == "rules"
    assert task.title == "Problem set three"
    assert task.due_date == date(2026, 11, 13)
    assert tasks.rejected == [2]  # no time phrase, so no rule-based task


def test_api_error_also_falls_back():
    class Broken:
        model, provider = "x", "y"

        def chat(self, *args, **kwargs):
            raise LLMError("rate limited")

    tasks = extract_tasks(LECTURE, Broken(), lecture_date=WED)
    assert tasks.tasks[0].source == "rules"
    assert "rate limited" in tasks.warnings[0]


def test_rules_only_mode_makes_no_calls():
    client = FakeLLMClient()
    tasks = extract_tasks(LECTURE, client, lecture_date=WED, use_llm=False)
    assert client.calls == [] and tasks.method == "rules"
    assert tasks.tasks[0].due_date == date(2026, 11, 13)


def test_no_lecture_date_keeps_the_phrase_and_warns():
    tasks = extract_tasks(LECTURE, FakeLLMClient([PSET_REPLY]))
    task = tasks.tasks[0]
    assert task.due_date is None and task.due_phrase == "Friday at 5pm"
    assert any("no lecture date" in w for w in tasks.warnings)
    assert 'due "Friday at 5pm" (date not resolved)' in tasks.to_markdown()


def test_batches_and_candidate_cap():
    spans = [
        (i * 10.0, i * 10.0 + 5, f"Homework {i} is due Friday.") for i in range(1, 6)
    ]
    lecture = make_transcript(spans)
    client = FakeLLMClient(replies=['{"tasks": []}'])
    tasks = extract_tasks(lecture, client, batch_size=2)
    assert len(client.calls) == 3  # 5 candidates in batches of 2
    assert tasks.rejected == [1, 2, 3, 4, 5]
    capped = extract_tasks(lecture, FakeLLMClient(['{"tasks": []}']), max_candidates=3)
    assert "only the 3 highest-scoring" in capped.warnings[0]
    with pytest.raises(ValueError):
        extract_tasks(lecture, client, batch_size=0)


def test_repeated_mentions_merge_and_sort_by_due_date():
    lecture = make_transcript(
        [
            (0.0, 5.0, "Read chapter four by next Tuesday."),
            (10.0, 15.0, "The midterm is on November 30."),
            (20.0, 25.0, "Again, read chapter four by next Tuesday."),
        ]
    )
    reply = json.dumps(
        [
            {
                "excerpt": 1,
                "title": "Read chapter 4",
                "kind": "read",
                "due_phrase": "next Tuesday",
            },
            {
                "id": 2,
                "title": "Midterm",
                "kind": "midterm",
                "due_phrase": "November 30",
            },
            {
                "excerpt": 3,
                "title": "read Chapter 4",
                "kind": "reading",
                "due_phrase": "next Tuesday",
            },
        ]
    )
    tasks = extract_tasks(lecture, FakeLLMClient([reply]), lecture_date=WED)
    assert [t.title for t in tasks.tasks] == ["Read chapter 4", "Midterm"]
    reading, midterm = tasks.tasks
    assert reading.kind == "reading" and reading.mentions == [0.0, 20.0]
    assert reading.due_date == date(2026, 11, 17) and reading.date_ambiguous
    assert midterm.kind == "exam" and midterm.due_date == date(2026, 11, 30)


def test_merge_fills_a_missing_date_from_a_later_mention():
    first = Task(title="The final project", evidence="a", start=1.0)
    later = Task(
        title="final project",
        evidence="b",
        start=9.0,
        due_phrase="December 5",
        due_date=date(2026, 12, 5),
        date_rule="month and day",
    )
    (merged,) = merge_tasks([first, later])
    assert merged.due_date == date(2026, 12, 5) and merged.start == 1.0
    assert merged.mentions == [9.0]


def test_validate_reply_errors():
    with pytest.raises(ValueError, match="do not exist"):
        validate_tasks_reply('{"tasks": [{"excerpt": 3, "title": "x"}]}', 2)
    with pytest.raises(ValueError, match="wrong JSON shape"):
        validate_tasks_reply('{"tasks": [{"excerpt": 1}]}', 2)
    reply = validate_tasks_reply(
        '{"tasks": [{"candidate": 1, "title": " PS1 ", "kind": "weird", '
        '"due_phrase": ""}]}',
        1,
    )
    task = reply.tasks[0]
    assert (task.excerpt, task.title, task.kind, task.due_phrase) == (
        1,
        "PS1",
        "other",
        None,
    )
    assert validate_tasks_reply('{"tasks": []}', 3).tasks == []


def test_phrase_in_text_ignores_case_and_punctuation():
    assert phrase_in_text("Friday, 5 p.m.", "due friday 5 p.m. sharp")
    assert not phrase_in_text("Friday at noon", "due Friday at 5pm")
    assert not phrase_in_text("...", "anything")


def test_rule_based_task_needs_a_time():
    found = find_task_candidates(LECTURE)
    assert rule_based_task(found[1]) is None  # "not on the midterm"
    guess = rule_based_task(found[0])
    assert (guess.title, guess.kind, guess.due_phrase) == (
        "Problem set three",
        "homework",
        "Friday at 5pm",
    )


def test_markdown_and_round_trip(tmp_path):
    tasks = extract_tasks(LECTURE, FakeLLMClient([PSET_REPLY]), lecture_date=WED)
    md = tasks.to_markdown()
    assert "Lecture date: 2026-11-11" in md
    assert "- [ ] **Problem set 3** (homework), due **Fri 2026-11-13 17:00**" in md
    assert "(not a task)" in md
    json_path, _ = write_tasks(tasks, tmp_path)
    assert load_tasks(json_path) == tasks


def _save(tmp_path):
    return LECTURE.save_json(tmp_path / "lec1" / "transcript.json")


def test_cli_tasks_with_fake_llm(tmp_path, capsys):
    path = _save(tmp_path)
    args = cli.build_parser().parse_args(
        ["tasks", str(path), "--lecture-date", "2026-11-11"]
    )
    assert cli.run_tasks(args, client=FakeLLMClient([PSET_REPLY])) == 0
    out = capsys.readouterr().out
    assert "Problem set 3 (homework): 2026-11-13 17:00" in out
    saved = load_tasks(path.parent / "tasks.json")
    assert saved.title == "Tasks: lec1"
    # Step 4.3: the to-do export is written alongside.
    ics = (path.parent / "tasks.ics").read_text(encoding="utf-8")
    assert "DTSTART:20261113T170000" in ics
    assert (path.parent / "todo.json").exists()


def test_cli_tasks_offline_and_errors(tmp_path, capsys):
    path = _save(tmp_path)
    assert cli.main(["tasks", str(path), "--no-llm", "--class-days", "mon"]) == 0
    out = capsys.readouterr().out
    assert "offline rules" in out and '"Friday at 5pm"' in out
    assert cli.main(["tasks", str(path), "--no-llm", "--lecture-date", "x"]) == 1
    assert cli.main(["tasks", str(path), "--no-llm", "--class-days", "xyz"]) == 1
    assert cli.main(["tasks", str(tmp_path / "nope.json"), "--no-llm"]) == 1
    assert cli.main(["tasks", str(path), "--batch-size", "0"]) == 1


def test_keyword_due_phrase_keeps_next_and_day_numbers():
    lecture = make_transcript(
        [
            (0.0, 5.0, "Problem set four is due next Friday at 11:59 pm."),
            (6.0, 9.0, "The midterm is on December 3rd, in this room."),
            (10.0, 12.0, "Homework is posted."),
        ]
    )
    found = find_task_candidates(lecture)
    phrases = [keyword_due_phrase(c) for c in found]
    assert phrases == ["next Friday at 11:59 pm", "December 3rd", None]
