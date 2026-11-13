"""Tests for the to-do export (src/notetaker/todo.py): todo.json and tasks.ics.

The .ics output is checked two ways: by looking at the exact text we write,
and by parsing it with the independent `icalendar` package (a dev-only
dependency), which is the closest offline stand-in for "a real calendar app
can import this file".
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import icalendar
import pytest

from notetaker import cli
from notetaker.tasks import Task, TaskList, write_tasks
from notetaker.todo import (
    MAX_LINE_BYTES,
    IcsOptions,
    calendar_count,
    escape_text,
    fold_line,
    task_uid,
    to_ics,
    to_todo,
    write_todo,
)

NOW = datetime(2026, 11, 12, 9, 30, tzinfo=timezone.utc)


def make_task(**overrides) -> Task:
    fields = dict(
        title="Problem set 3",
        kind="homework",
        details="questions 1 to 5",
        due_phrase="Friday at 5pm",
        due_date=date(2026, 11, 13),
        due_time="17:00",
        date_rule="weekday: next Friday after the lecture",
        start=12.0,
        mentions=[12.0, 300.0],
        evidence="Problem set three is due Friday at 5pm.",
        source="llm",
    )
    fields.update(overrides)
    return Task(**fields)


TASKS = TaskList(
    title="Tasks: 18.06 lecture 4",
    method="llm",
    reference_date=date(2026, 11, 11),
    tasks=[
        make_task(),
        make_task(
            title="Midterm",
            kind="exam",
            details="",
            due_phrase="December 3rd",
            due_date=date(2026, 12, 3),
            due_time=None,
            start=30.0,
            mentions=[30.0],
            evidence="The midterm is on December 3rd, in this room.",
        ),
        make_task(
            title="Read chapter 4",
            kind="reading",
            details="",
            due_phrase=None,
            due_date=None,
            due_time=None,
            date_rule="",
            start=45.0,
            mentions=[45.0],
            evidence="Please read chapter 4, sections 1; 2 and 3.",
            source="rules",
        ),
    ],
)


def parse(text: str) -> icalendar.Calendar:
    return icalendar.Calendar.from_ical(text)


# ---------------------------------------------------------------------------
# todo.json
# ---------------------------------------------------------------------------


def test_todo_items_have_one_due_field_and_lecture_time():
    todo = to_todo(TASKS)
    assert todo.title == "Tasks: 18.06 lecture 4"
    assert todo.lecture_date == date(2026, 11, 11)
    pset, midterm, reading = todo.items
    assert (pset.due, pset.all_day) == ("2026-11-13T17:00", False)
    assert (midterm.due, midterm.all_day) == ("2026-12-03", True)
    assert (reading.due, reading.all_day) == (None, False)
    assert pset.said_at == "00:00:12" and pset.mentions == [12.0, 300.0]
    assert not any(item.done for item in todo.items)
    assert todo.undated == 1


def test_uid_is_stable_ignores_date_and_spacing_but_not_title():
    a = task_uid(make_task(), "Course")
    assert a.endswith("@notetaker")
    # A corrected date keeps the same UID, so a calendar moves the event.
    assert task_uid(make_task(due_date=date(2026, 11, 20)), "Course") == a
    assert task_uid(make_task(title="  problem   SET 3 "), "course") == a
    assert task_uid(make_task(title="Problem set 4"), "Course") != a
    assert task_uid(make_task(), "Other course") != a


def test_write_todo_is_deterministic(tmp_path):
    json_path, ics_path = write_todo(TASKS, tmp_path, now=NOW)
    first = (json_path.read_bytes(), ics_path.read_bytes())
    write_todo(TASKS, tmp_path, now=NOW)
    assert (json_path.read_bytes(), ics_path.read_bytes()) == first
    data = json.loads(first[0])
    assert data["format_version"] == 1
    assert data["items"][0]["id"] == task_uid(TASKS.tasks[0], TASKS.title)
    # Real CRLFs on disk, never doubled into "\r\r\n".
    assert b"\r\n" in first[1] and b"\r\r\n" not in first[1]


# ---------------------------------------------------------------------------
# iCalendar text helpers
# ---------------------------------------------------------------------------


def test_escape_text_handles_special_characters():
    assert escape_text("a,b;c\\d") == "a\\,b\\;c\\\\d"
    assert escape_text("line1\r\nline2\rline3\n") == "line1\\nline2\\nline3\\n"


def test_fold_line_respects_byte_limit_and_utf8():
    line = "DESCRIPTION:" + "é" * 100  # each é is 2 bytes
    folded = fold_line(line)
    physical = folded.split("\r\n")
    assert len(physical) > 1
    assert all(len(p.encode("utf-8")) <= MAX_LINE_BYTES for p in physical)
    assert all(p.startswith(" ") for p in physical[1:])
    # Unfolding (remove CRLF + one space) gives the original back, so no
    # character was cut in half.
    assert folded.replace("\r\n ", "") == line
    assert fold_line("SHORT:1") == "SHORT:1"


# ---------------------------------------------------------------------------
# tasks.ics
# ---------------------------------------------------------------------------


def test_ics_events_parse_with_icalendar():
    text = to_ics(TASKS, now=NOW)
    assert text.startswith("BEGIN:VCALENDAR\r\nVERSION:2.0\r\n")
    assert text.endswith("END:VCALENDAR\r\n")
    cal = parse(text)
    assert str(cal["X-WR-CALNAME"]) == "Tasks: 18.06 lecture 4"
    events = list(cal.walk("VEVENT"))
    assert len(events) == 2  # the undated reading is not on the calendar

    pset, midterm = events
    assert str(pset["SUMMARY"]) == "Due: Problem set 3"
    start = pset.decoded("DTSTART")
    assert start == datetime(2026, 11, 13, 17, 0) and start.tzinfo is None
    assert pset.decoded("DTEND") - start == timedelta(minutes=30)
    assert pset.decoded("DTSTAMP") == NOW
    assert str(pset["UID"]) == task_uid(TASKS.tasks[0], TASKS.title)
    description = str(pset["DESCRIPTION"])
    assert 'Lecturer said: "Friday at 5pm"' in description
    assert "Said at 00:00:12" in description and "\n" in description

    # All-day event: plain dates, end date exclusive.
    assert midterm.decoded("DTSTART") == date(2026, 12, 3)
    assert midterm.decoded("DTEND") == date(2026, 12, 4)

    alarms = list(pset.walk("VALARM"))
    assert len(alarms) == 1
    assert alarms[0].decoded("TRIGGER") == timedelta(days=-1)


def test_ics_escaped_text_round_trips():
    tricky = make_task(
        title="Lab 2, part A; with \\ backslash",
        evidence="Lab two, part A; due Friday.",
    )
    cal = parse(to_ics(TaskList(title="T", tasks=[tricky]), now=NOW))
    (event,) = cal.walk("VEVENT")
    assert str(event["SUMMARY"]) == "Due: Lab 2, part A; with \\ backslash"


def test_ics_vtodo_option_and_no_alarm():
    options = IcsOptions(component="todo", reminder_minutes=None)
    text = to_ics(TASKS, options, now=NOW)
    assert "VEVENT" not in text and "VALARM" not in text
    todos = list(parse(text).walk("VTODO"))
    assert [str(t["SUMMARY"]) for t in todos] == ["Problem set 3", "Midterm"]
    assert todos[0].decoded("DUE") == datetime(2026, 11, 13, 17, 0)
    assert todos[1].decoded("DUE") == date(2026, 12, 3)
    assert str(todos[0]["STATUS"]) == "NEEDS-ACTION"


def test_ambiguous_dates_are_marked_and_custom_reminder():
    tasks = TaskList(title="T", tasks=[make_task(date_ambiguous=True)])
    options = IcsOptions(event_minutes=90, reminder_minutes=150)
    (event,) = parse(to_ics(tasks, options, now=NOW)).walk("VEVENT")
    assert str(event["SUMMARY"]).endswith("(?)")
    assert "ambiguous" in str(event["DESCRIPTION"])
    assert event.decoded("DTEND") - event.decoded("DTSTART") == timedelta(minutes=90)
    (alarm,) = event.walk("VALARM")
    assert alarm.decoded("TRIGGER") == -timedelta(hours=2, minutes=30)


def test_empty_task_list_is_still_a_valid_calendar():
    cal = parse(to_ics(TaskList(title="Nothing"), now=NOW))
    assert list(cal.walk("VEVENT")) == []
    assert calendar_count([]) == 0


@pytest.mark.parametrize(
    "kwargs",
    [{"component": "meeting"}, {"event_minutes": 0}, {"reminder_minutes": -5}],
)
def test_bad_ics_options(kwargs):
    with pytest.raises(ValueError):
        IcsOptions(**kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_todo_reexports_tasks_json(tmp_path, capsys):
    json_path, _ = write_tasks(TASKS, tmp_path / "lec")
    out = tmp_path / "out"
    code = cli.main(
        ["todo", str(json_path), "-o", str(out), "--vtodo", "--no-reminder"]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "3 task(s): 2 on the calendar, 1 without a date" in printed
    text = (out / "tasks.ics").read_text(encoding="utf-8")
    assert "BEGIN:VTODO" in text and "VALARM" not in text
    assert len(json.loads((out / "todo.json").read_text())["items"]) == 3


def test_cli_todo_errors(tmp_path, capsys):
    json_path, _ = write_tasks(TASKS, tmp_path)
    assert cli.main(["todo", str(tmp_path / "missing.json")]) == 1
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "tasks"}')
    assert cli.main(["todo", str(bad)]) == 1
    assert cli.main(["todo", str(json_path), "--event-minutes", "0"]) == 1
    assert "Error" in capsys.readouterr().out
