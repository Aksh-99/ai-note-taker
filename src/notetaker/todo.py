"""Export the tasks stage as a to-do list (todo.json) and a calendar (tasks.ics).

Step 4.2 produced `tasks.json`: a detailed record of *how* each task was
found (candidate sentences, LLM token counts, rejected excerpts...). That is
great for debugging but awkward for anything else. This module turns it into
two files other programs can use directly:

    todo.json   A flat, stable list of to-do items, one per task, with a
                single `due` field ("2026-11-13T17:00", "2026-11-13" or
                null). Easy to load into a script, a to-do app or a web UI
                (step 7). It contains no timestamps of when it was made, so
                running the export twice gives byte-identical files, which
                keeps git diffs clean.

    tasks.ics   An iCalendar file (RFC 5545), the plain-text format every
                calendar app imports: Google Calendar, Apple Calendar,
                Outlook, Thunderbird. Double-click it and the deadlines
                appear in your calendar with a reminder the day before.

Why write the .ics by hand instead of using a library?
    The subset we need is small (a calendar, some events, an alarm), and
    writing it ourselves keeps the install small and shows the format is not
    magic. The tests parse our output with the independent `icalendar`
    package (a dev-only dependency) to prove real software can read it.

The few iCalendar rules that matter here:
    * Lines end in CRLF ("\\r\\n"), not just "\\n".
    * A line may be at most 75 *bytes* long. Longer lines are "folded": cut,
      and continued on the next line which starts with a single space.
      Cutting must not split a multi-byte UTF-8 character (e.g. an accented
      letter is 2 bytes), see `fold_line`.
    * In text values, backslash, comma, semicolon and newline are special
      and must be escaped as \\\\ \\, \\; and \\n, see `escape_text`.
    * Every component needs a UID, a globally unique id. If you import the
      same file twice, the calendar uses the UID to *update* the existing
      event instead of adding a duplicate. We derive it from a hash of the
      course title, task title and kind, so re-running the pipeline after
      fixing a date moves the event instead of duplicating it.

Dates and times:
    A task with a time ("Friday at 5pm") becomes a timed event that starts
    at the deadline. A task with only a date becomes an all-day event (in
    iCalendar the end date of an all-day event is *exclusive*, so a one-day
    event on the 13th ends on the 14th). Times are written as "floating"
    local time (no time zone), meaning "5pm wherever you are", which is how
    a lecturer means it. Tasks with no resolved date cannot go on a
    calendar; they stay in todo.json and are counted as skipped.

VEVENT or VTODO?
    iCalendar has a to-do component (VTODO) with a DUE date, which Apple
    Reminders and Thunderbird understand, but Google Calendar ignores it.
    So the default is events (VEVENT), which work everywhere, and
    `component="todo"` switches to VTODO.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from notetaker.export import format_timestamp
from notetaker.tasks import Task, TaskKind, TaskList

TODO_FORMAT_VERSION = 1  # bump if the todo.json layout ever changes
PRODID = "-//ai-note-taker//notetaker//EN"  # identifies the app that made the file
MAX_LINE_BYTES = 75  # RFC 5545 section 3.1
DEFAULT_EVENT_MINUTES = 30  # length of a timed deadline event
DEFAULT_REMINDER_MINUTES = 24 * 60  # alarm one day before

Component = Literal["event", "todo"]


# ---------------------------------------------------------------------------
# todo.json
# ---------------------------------------------------------------------------


class TodoItem(BaseModel):
    """One to-do item: everything an app needs, nothing about how it was found."""

    id: str  # same value as the calendar UID, so the two files can be matched
    title: str
    kind: TaskKind
    details: str = ""
    due: str | None = None  # "YYYY-MM-DDTHH:MM", "YYYY-MM-DD" or None
    all_day: bool = False  # True when only a date (no time) is known
    due_phrase: str | None = None  # the lecturer's own words, e.g. "Friday at 5pm"
    ambiguous: bool = False  # the date phrase could mean two things
    done: bool = False  # always False on export; a to-do app flips it
    said_at: str  # "HH:MM:SS" into the lecture, to jump to in the recording
    said_at_seconds: float
    mentions: list[float] = Field(default_factory=list)
    evidence: str  # the sentence the task came from
    source: Literal["llm", "rules"]


class TodoExport(BaseModel):
    """The whole todo.json file."""

    format_version: int = TODO_FORMAT_VERSION
    title: str
    lecture_date: date | None = None
    items: list[TodoItem] = Field(default_factory=list)
    undated: int = 0  # how many items have no due date (not on the calendar)


def task_uid(task: Task, calendar_title: str) -> str:
    """A stable unique id for a task, used as the iCalendar UID.

    A hash is a fixed-length fingerprint of some text: the same input always
    gives the same output, and different inputs almost surely differ. We hash
    the course title + task title + kind, lowercased and with spaces
    collapsed, so cosmetic changes do not create a "new" task. The due date
    is deliberately left out: if a date gets corrected, the UID stays the
    same and the calendar moves the event instead of duplicating it.
    """
    key = "|".join(
        " ".join(part.lower().split())
        for part in (calendar_title, task.title, task.kind)
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{digest}@notetaker"


def _due_string(task: Task) -> str | None:
    if task.due_date is None:
        return None
    if task.due_time:
        return f"{task.due_date.isoformat()}T{task.due_time}"
    return task.due_date.isoformat()


def to_todo(tasks: TaskList) -> TodoExport:
    """Flatten a TaskList into the simple todo.json structure."""
    items = [
        TodoItem(
            id=task_uid(t, tasks.title),
            title=t.title,
            kind=t.kind,
            details=t.details,
            due=_due_string(t),
            all_day=t.due_date is not None and not t.due_time,
            due_phrase=t.due_phrase,
            ambiguous=t.date_ambiguous,
            said_at=format_timestamp(t.start),
            said_at_seconds=t.start,
            mentions=t.mentions,
            evidence=t.evidence,
            source=t.source,
        )
        for t in tasks.tasks
    ]
    return TodoExport(
        title=tasks.title,
        lecture_date=tasks.reference_date,
        items=items,
        undated=sum(item.due is None for item in items),
    )


# ---------------------------------------------------------------------------
# iCalendar text helpers
# ---------------------------------------------------------------------------


def escape_text(value: str) -> str:
    r"""Escape a TEXT value (RFC 5545 section 3.3.11).

    Order matters: backslashes first, otherwise the backslashes we add for
    commas would themselves get doubled. Windows "\r\n" and lone "\r" are
    normalised to "\n" first, then every newline becomes the two characters
    backslash + n.
    """
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return (
        value.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
    )


def fold_line(line: str, limit: int = MAX_LINE_BYTES) -> str:
    """Fold one content line so no physical line exceeds `limit` bytes.

    The limit is in UTF-8 bytes, not characters: "é" is one character but two
    bytes. We add characters one at a time and start a new physical line just
    before the byte count would pass the limit, so a character is never cut
    in half. Continuation lines begin with one space, which a reader strips,
    so they have one byte less room.
    """
    pieces: list[str] = []
    current = ""
    used = 0  # bytes on the current physical line
    room = limit
    for char in line:
        size = len(char.encode("utf-8"))
        if used + size > room:
            pieces.append(current)
            current, used = "", 0
            room = limit - 1  # the leading space takes one byte
        current += char
        used += size
    pieces.append(current)
    return "\r\n ".join(pieces)


def _format_date(day: date) -> str:
    return day.strftime("%Y%m%d")


def _format_local(moment: datetime) -> str:
    # Floating local time: no trailing "Z" and no TZID.
    return moment.strftime("%Y%m%dT%H%M%S")


def _format_utc(moment: datetime) -> str:
    # DTSTAMP must be UTC, marked with a trailing "Z" ("Zulu" time).
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _duration(minutes: int) -> str:
    """Minutes as an iCalendar duration: 90 -> 'PT1H30M', 1440 -> 'P1D'."""
    days, rest = divmod(minutes, 24 * 60)
    hours, mins = divmod(rest, 60)
    text = "P" + (f"{days}D" if days else "")
    if hours or mins:
        text += "T" + (f"{hours}H" if hours else "") + (f"{mins}M" if mins else "")
    return text if text != "P" else "PT0M"


def _due_datetime(task: Task) -> datetime | None:
    """The deadline as a naive (floating) datetime, or None if date-only."""
    if task.due_date is None or not task.due_time:
        return None
    hour, minute = (int(x) for x in task.due_time.split(":"))
    return datetime.combine(task.due_date, time(hour, minute))


def _description(task: Task) -> str:
    """The multi-line notes shown when you open the event."""
    lines = [f"{task.kind.capitalize()} from lecture."]
    if task.details:
        lines.append(f"Details: {task.details}")
    if task.due_phrase:
        lines.append(f'Lecturer said: "{task.due_phrase}"')
    if task.date_ambiguous:
        lines.append("The date phrase was ambiguous, please double-check.")
    lines.append(f"Said at {format_timestamp(task.start)}: {task.evidence}")
    if task.date_rule:
        lines.append(f"Date rule: {task.date_rule}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# tasks.ics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IcsOptions:
    """Choices for the calendar export.

    component         "event" (VEVENT, works in every calendar app) or "todo"
                      (VTODO, for task apps such as Apple Reminders).
    event_minutes     length of a timed event that starts at the deadline.
    reminder_minutes  alarm this many minutes before; None for no alarm.
    """

    component: Component = "event"
    event_minutes: int = DEFAULT_EVENT_MINUTES
    reminder_minutes: int | None = DEFAULT_REMINDER_MINUTES

    def __post_init__(self) -> None:
        # __post_init__ runs right after the dataclass's generated __init__,
        # so bad values are rejected the moment the object is created.
        if self.component not in ("event", "todo"):
            raise ValueError("component must be 'event' or 'todo'")
        if self.event_minutes <= 0:
            raise ValueError("event_minutes must be > 0")
        if self.reminder_minutes is not None and self.reminder_minutes < 0:
            raise ValueError("reminder_minutes must be >= 0 (or None)")


def _component_lines(
    task: Task, calendar_title: str, options: IcsOptions, stamp: str
) -> list[str]:
    """The unfolded content lines for one task, or [] if it has no date."""
    if task.due_date is None:
        return []
    name = "VEVENT" if options.component == "event" else "VTODO"
    summary = ("Due: " if options.component == "event" else "") + task.title
    if task.date_ambiguous:
        summary += " (?)"
    lines = [
        f"BEGIN:{name}",
        f"UID:{task_uid(task, calendar_title)}",
        f"DTSTAMP:{stamp}",
        f"SUMMARY:{escape_text(summary)}",
        f"DESCRIPTION:{escape_text(_description(task))}",
        f"CATEGORIES:{escape_text(task.kind)}",
    ]
    moment = _due_datetime(task)
    if options.component == "event":
        if moment is not None:
            end = moment + timedelta(minutes=options.event_minutes)
            lines += [f"DTSTART:{_format_local(moment)}", f"DTEND:{_format_local(end)}"]
        else:
            # All-day: VALUE=DATE marks a date with no time; the end is
            # exclusive, so a one-day event ends on the following day.
            nxt = task.due_date + timedelta(days=1)
            lines += [
                f"DTSTART;VALUE=DATE:{_format_date(task.due_date)}",
                f"DTEND;VALUE=DATE:{_format_date(nxt)}",
            ]
        lines.append("TRANSP:TRANSPARENT")  # a deadline does not make you "busy"
    else:
        # Parameters such as VALUE=DATE go between the name and the colon.
        if moment is not None:
            lines.append(f"DUE:{_format_local(moment)}")
        else:
            lines.append(f"DUE;VALUE=DATE:{_format_date(task.due_date)}")
        lines.append("STATUS:NEEDS-ACTION")
    if options.reminder_minutes is not None:
        # A negative TRIGGER duration means "before the start (or due) time".
        lines += [
            "BEGIN:VALARM",
            "ACTION:DISPLAY",
            f"DESCRIPTION:{escape_text(summary)}",
            f"TRIGGER:-{_duration(options.reminder_minutes)}",
            "END:VALARM",
        ]
    lines.append(f"END:{name}")
    return lines


def to_ics(
    tasks: TaskList,
    options: IcsOptions | None = None,
    *,
    now: datetime | None = None,
) -> str:
    """Render the dated tasks as an iCalendar document (CRLF line endings).

    `now` fills DTSTAMP (when the file was made, required by the standard).
    Tests pass a fixed value so the output is reproducible.
    """
    options = options or IcsOptions()
    stamp = _format_utc(now or datetime.now(timezone.utc))
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:{PRODID}",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        # Non-standard but widely supported: the name a calendar app shows
        # when you subscribe to or import the file.
        f"X-WR-CALNAME:{escape_text(tasks.title)}",
    ]
    for task in tasks.tasks:
        lines += _component_lines(task, tasks.title, options, stamp)
    lines.append("END:VCALENDAR")
    return "".join(fold_line(line) + "\r\n" for line in lines)


def calendar_count(tasks: Iterable[Task]) -> int:
    """How many tasks can go on a calendar (those with a resolved date)."""
    return sum(t.due_date is not None for t in tasks)


def write_todo(
    tasks: TaskList,
    out_dir: str | Path,
    options: IcsOptions | None = None,
    *,
    now: datetime | None = None,
) -> tuple[Path, Path]:
    """Write todo.json and tasks.ics into `out_dir`; return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "todo.json"
    ics_path = out_dir / "tasks.ics"
    todo = to_todo(tasks)
    json_path.write_text(
        json.dumps(todo.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # newline="" stops Python from translating our "\r\n" into "\r\r\n" on
    # Windows; the CRLFs are already in the text.
    with ics_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(to_ics(tasks, options, now=now))
    return json_path, ics_path
