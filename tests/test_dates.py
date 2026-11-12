"""Tests for deadline phrase -> calendar date resolution (notetaker.dates)."""

from __future__ import annotations

from datetime import date

import pytest

from notetaker.dates import parse_time, parse_weekdays, resolve_due

WED = date(2026, 11, 11)  # a Wednesday
FRI = date(2026, 11, 13)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("Friday", date(2026, 11, 13)),
        ("by Monday", date(2026, 11, 16)),
        # A bare weekday equal to today means next week...
        ("Wednesday", date(2026, 11, 18)),
        # ...but "this Wednesday" said on a Wednesday means today.
        ("this Wednesday", WED),
        ("tomorrow", date(2026, 11, 12)),
        ("tonight", WED),
        ("in 3 days", date(2026, 11, 14)),
        ("in two weeks", date(2026, 11, 25)),
        ("in a week", date(2026, 11, 18)),
        ("end of the week", FRI),
        ("end of month", date(2026, 11, 30)),
        ("November 20th", date(2026, 11, 20)),
        ("Nov. 20", date(2026, 11, 20)),
        ("the twentieth of November", date(2026, 11, 20)),
        ("20 November", date(2026, 11, 20)),
        ("December the first", date(2026, 12, 1)),
        ("2026-12-01", date(2026, 12, 1)),
        ("the 15th", date(2026, 11, 15)),
        ("the 3rd", date(2026, 12, 3)),  # the 3rd of November has passed
    ],
)
def test_phrases_resolve_relative_to_the_lecture(phrase, expected):
    assert resolve_due(phrase, WED).date == expected


def test_month_day_in_the_past_rolls_into_next_year():
    resolved = resolve_due("January 5", date(2026, 12, 10))
    assert resolved.date == date(2027, 1, 5)
    assert "next year" in resolved.rule


def test_next_weekday_is_the_following_week_and_flagged():
    resolved = resolve_due("next Friday", WED)
    assert resolved.date == date(2026, 11, 20)
    assert resolved.ambiguous
    # Said on a Monday, "next Friday" still skips the current week.
    assert resolve_due("next Friday", date(2026, 11, 9)).date == date(2026, 11, 20)


def test_end_of_week_on_friday_is_today():
    assert resolve_due("end of week", FRI).date == FRI


def test_numeric_dates_are_us_month_day_and_maybe_ambiguous():
    resolved = resolve_due("12/1", WED)
    assert resolved.date == date(2026, 12, 1)
    assert resolved.ambiguous  # could be 12 January elsewhere
    assert not resolve_due("12/25", WED).ambiguous
    assert resolve_due("13/1", WED).date is None  # month 13 does not exist


def test_next_class_uses_class_days():
    no_schedule = resolve_due("next class", WED)
    assert no_schedule.date is None and "class-days" in no_schedule.rule
    # Course meets Monday and Wednesday: next class after a Wednesday is Monday.
    assert resolve_due("next class", WED, class_days=(0, 2)).date == date(2026, 11, 16)
    # Meets Wednesday and Friday: Friday.
    assert resolve_due("next lecture", WED, class_days=(2, 4)).date == FRI


def test_next_week_without_a_day():
    resolved = resolve_due("next week", WED)
    assert resolved.date == date(2026, 11, 18) and resolved.ambiguous


def test_time_of_day_is_parsed_with_the_date():
    resolved = resolve_due("Friday at 5pm", WED)
    assert (resolved.date, resolved.time) == (FRI, "17:00")
    assert resolve_due("tomorrow by midnight", WED).time == "23:59"


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("5pm", "17:00"),
        ("5 p.m.", "17:00"),
        ("11:59 pm", "23:59"),
        ("12 am", "00:00"),
        ("12pm", "12:00"),
        ("9:30am", "09:30"),
        ("noon", "12:00"),
        ("midnight", "23:59"),
        ("17:00", "17:00"),
        ("Friday", None),
        ("13pm", None),
    ],
)
def test_parse_time(phrase, expected):
    assert parse_time(phrase) == expected


def test_unresolvable_and_missing_inputs():
    assert resolve_due("end of semester", WED).date is None
    assert resolve_due("soon", WED).rule == "no date found in phrase"
    assert resolve_due(None, WED).rule == "no deadline phrase"
    assert resolve_due("  ", WED).date is None
    # No lecture date: no date, but the time is still read.
    no_ref = resolve_due("Friday at 5pm", None)
    assert no_ref.date is None and no_ref.time == "17:00"
    assert resolve_due("February 30", WED).date is None


def test_parse_weekdays():
    assert parse_weekdays(["Mon", "wed", "Friday"]) == (0, 2, 4)
    assert parse_weekdays(["thurs", "weds", "tue"]) == (1, 2, 3)
    with pytest.raises(ValueError):
        parse_weekdays(["funday"])
    with pytest.raises(ValueError):
        parse_weekdays(["t"])  # too short: tuesday or thursday?
