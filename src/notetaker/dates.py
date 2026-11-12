"""Turn spoken deadline phrases ("Friday at 5pm") into calendar dates.

Why is this plain code and not part of the LLM prompt? Language models are
good at reading ("which sentence is a homework deadline?") but unreliable at
calendar arithmetic ("what date is the Friday after 2026-11-11?"): they guess
weekdays, forget month lengths and leap years, and give different answers on
different runs. So the work is split:

    LLM:  reads the sentence, copies the deadline phrase verbatim
    code: resolves the phrase against the lecture date, deterministically

Every resolution records the rule that fired and whether the phrase was
ambiguous ("next Friday" means different things to different people), so a
student can see *why* a date was chosen and double-check the risky ones.

Conventions (documented because they are choices, not facts):
- A bare weekday means its next occurrence strictly after the lecture day:
  "due Friday" said on a Wednesday is two days later, said on a Friday it is
  a week later (nobody announces homework due the same afternoon).
- "next <weekday>" means that weekday in the following calendar week
  (weeks start on Monday), flagged ambiguous.
- "midnight" means 23:59 of the named day, the usual meaning of "due Friday
  at midnight". "noon" is 12:00.
- Numeric dates are US style, month/day ("3/14" is March 14).
- A month and day with no year means the next such date on or after the
  lecture, so a December lecture saying "due January 5" lands in next year.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Python's date.weekday() numbers days Monday=0 ... Sunday=6.
WEEKDAYS: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
# Short forms, for the --class-days option ("mon wed").
WEEKDAY_ABBREVIATIONS: dict[str, int] = {name[:3]: i for name, i in WEEKDAYS.items()}

MONTHS: dict[str, int] = {
    name.lower(): i for i, name in enumerate(calendar.month_name) if name
}
# "Jan", "Feb", ... and "Sept", which people really write.
MONTHS.update({name[:3]: i for name, i in list(MONTHS.items())})
MONTHS["sept"] = 9

# Ordinal words, because Whisper sometimes writes "the fifteenth".
_ORDINAL_WORDS = [
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
]
ORDINALS: dict[str, int] = {word: i + 1 for i, word in enumerate(_ORDINAL_WORDS)}
for _i, _word in enumerate(_ORDINAL_WORDS[:9]):
    # twenty-first ... twenty-ninth, thirtieth, thirty-first.
    ORDINALS[f"twenty-{_word}"] = ORDINALS[f"twenty {_word}"] = 21 + _i
ORDINALS["thirtieth"] = 30
ORDINALS["thirty-first"] = ORDINALS["thirty first"] = 31

NUMBER_WORDS: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
# Building regexes from the dictionaries keeps them in sync: adding a month
# name above automatically makes it matchable. Longer names go first in an
# alternation so "september" wins over "sep".


def _alternation(words) -> str:
    return "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))


_MONTH = _alternation(MONTHS)
_WEEKDAY = _alternation(WEEKDAYS)
_ORDINAL = _alternation(ORDINALS)
# A day number: "15", "15th", "1st", or an ordinal word.
_DAY = rf"(?:(?P<dnum>\d{{1,2}})(?:st|nd|rd|th)?|(?P<dword>{_ORDINAL}))"
_DAY2 = _DAY.replace("dnum", "dnum2").replace("dword", "dword2")

ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
# "November 20", "Nov. 20th", "November the twentieth"
MONTH_DAY = re.compile(rf"\b(?P<month>{_MONTH})\.?\s+(?:the\s+)?{_DAY}\b", re.I)
# "20 November", "the 20th of November"
DAY_MONTH = re.compile(
    rf"\b(?:the\s+)?{_DAY2}\s+(?:of\s+)?(?P<month2>{_MONTH})\b", re.I
)
# "11/20" (US month/day); the lookarounds avoid matching inside "1/2/3/4".
NUMERIC_DATE = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])")
# "the 15th" with no month
BARE_DAY = re.compile(rf"\bthe\s+{_DAY}\b", re.I)
RELATIVE_DAYS = re.compile(
    rf"\bin\s+(?P<n>\d+|{_alternation(NUMBER_WORDS)})\s+(?P<unit>days?|weeks?)\b",
    re.I,
)
WEEKDAY = re.compile(rf"\b(?P<mod>next|this|coming)?\s*(?P<day>{_WEEKDAY})\b", re.I)

# Times: "5pm", "5 p.m.", "11:59 pm", "17:00", "noon", "midnight".
TIME_12H = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s?m\b\.?", re.I)
TIME_24H = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
NOON = re.compile(r"\bnoon\b", re.I)
MIDNIGHT = re.compile(r"\bmidnight\b", re.I)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedDate:
    """The outcome of resolving one phrase.

    `frozen=True` makes instances immutable (like a tuple with names), which
    is right for a value that is computed once and only read afterwards.
    """

    date: date | None  # None: no date could be worked out
    time: str | None  # "HH:MM" (24 h) or None when no time was said
    rule: str  # short description of how the date was found, for humans
    ambiguous: bool = False  # True when a reasonable person could disagree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_weekdays(names: list[str] | tuple[str, ...]) -> tuple[int, ...]:
    """'mon', 'Wednesday', ... -> sorted weekday numbers (Monday=0)."""
    days = set()
    for name in names:
        key = name.strip().lower().rstrip(".,")
        # Any prefix of at least 3 letters counts: "wed", "thur", "friday".
        matches = [
            i for full, i in WEEKDAYS.items() if len(key) >= 3 and full.startswith(key)
        ]
        if key in {"weds", "thurs"}:
            matches = [WEEKDAY_ABBREVIATIONS[key[:3]]]
        if len(matches) != 1:
            raise ValueError(f"not a weekday: {name!r}")
        days.add(matches[0])
    return tuple(sorted(days))


def _days_until(reference: date, weekday: int, *, allow_today: bool) -> int:
    """Days from `reference` to the next `weekday` (0..6, or 1..7).

    The modulo trick: (target - today) % 7 is the forward distance on a
    7-day clock face, e.g. Wednesday(2) -> Friday(4) is 2, Friday -> Monday
    is (0 - 4) % 7 = 3.
    """
    delta = (weekday - reference.weekday()) % 7
    if delta == 0 and not allow_today:
        delta = 7
    return delta


def _day_value(match: re.Match[str], num: str, word: str) -> int | None:
    if match.group(num):
        return int(match.group(num))
    if match.group(word):
        return ORDINALS[match.group(word).lower()]
    return None


def _month_day(reference: date, month: int, day: int) -> tuple[date | None, bool]:
    """Next valid (month, day) on or after `reference`; (None, False) if invalid.

    Returns (date, rolled_over) where rolled_over means "pushed to next year".
    """
    for year in (reference.year, reference.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:  # e.g. February 30
            return None, False
        if candidate >= reference:
            return candidate, year != reference.year
    return None, False  # unreachable: next year is always >= reference


def parse_time(phrase: str) -> str | None:
    """The first clock time in `phrase` as "HH:MM", or None."""
    m = TIME_12H.search(phrase)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        if not 1 <= hour <= 12 or minute > 59:
            return None
        # 12 am is 00:xx and 12 pm is 12:xx; otherwise pm adds 12 hours.
        hour = hour % 12 + (12 if m.group(3).lower() == "p" else 0)
        return f"{hour:02d}:{minute:02d}"
    if MIDNIGHT.search(phrase):
        return "23:59"
    if NOON.search(phrase):
        return "12:00"
    m = TIME_24H.search(phrase)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return None


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------


def resolve_due(
    phrase: str | None,
    reference: date | None,
    *,
    class_days: tuple[int, ...] = (),
) -> ResolvedDate:
    """Resolve a deadline phrase said in a lecture held on `reference`.

    `class_days` (Monday=0) lets "next class" / "next lecture" resolve to the
    next day the course meets. The rules are tried from most to least
    specific; the first that matches decides the date. The time of day is
    parsed separately, so "Friday at 5pm" gets both.
    """
    if not phrase or not phrase.strip():
        return ResolvedDate(None, None, "no deadline phrase")
    text = phrase.strip()
    time_of_day = parse_time(text)
    if reference is None:
        return ResolvedDate(None, time_of_day, "no lecture date given")
    lower = text.lower()

    def done(when: date | None, rule: str, ambiguous: bool = False) -> ResolvedDate:
        return ResolvedDate(when, time_of_day, rule, ambiguous)

    # 1. Fully explicit dates.
    m = ISO_DATE.search(text)
    if m:
        try:
            return done(date(*map(int, m.groups())), "explicit date")
        except ValueError:
            pass
    for pattern, num, word, month_key in (
        (MONTH_DAY, "dnum", "dword", "month"),
        (DAY_MONTH, "dnum2", "dword2", "month2"),
    ):
        m = pattern.search(text)
        if m:
            month = MONTHS[m.group(month_key).lower()]
            day = _day_value(m, num, word)
            when, rolled = _month_day(reference, month, day or 0)
            if when is not None:
                rule = "month and day" + (" (next year)" if rolled else "")
                return done(when, rule)
    m = NUMERIC_DATE.search(text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            when, rolled = _month_day(reference, month, day)
            if when is not None:
                # 3/4 could be March 4 (US) or 3 April (most other places).
                return done(when, "numeric month/day (US)", ambiguous=day <= 12)

    # 2. Words relative to the lecture day.
    if re.search(r"\b(?:today|tonight|end of (?:the )?day)\b", lower):
        return done(reference, "same day")
    if re.search(r"\btomorrow\b", lower):
        return done(reference + timedelta(days=1), "next day")
    m = RELATIVE_DAYS.search(text)
    if m:
        raw = m.group("n").lower()
        n = int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
        days = n * (7 if m.group("unit").lower().startswith("week") else 1)
        return done(reference + timedelta(days=days), f"in {days} day(s)")
    if re.search(r"\bend of (?:the )?week\b", lower):
        friday = reference + timedelta(days=_days_until(reference, 4, allow_today=True))
        return done(friday, "end of week (Friday)")
    if re.search(r"\bend of (?:the )?month\b", lower):
        last = calendar.monthrange(reference.year, reference.month)[1]
        return done(reference.replace(day=last), "end of month")

    # 3. Weekdays. Checked after "end of week" so "by end of week, Friday"
    #    still works, and before "next week" so "next Friday" is not read
    #    as just "next week".
    m = WEEKDAY.search(text)
    if m:
        weekday = WEEKDAYS[m.group("day").lower()]
        modifier = (m.group("mod") or "").lower()
        if modifier == "next":
            # Monday of next week, then step to the named day.
            next_monday = reference + timedelta(days=7 - reference.weekday())
            return done(
                next_monday + timedelta(days=weekday),
                "next <weekday> = that day in the following week",
                ambiguous=True,
            )
        # "this Wednesday" said on a Wednesday means today; a bare
        # "Wednesday" said on a Wednesday means next week (see conventions).
        delta = _days_until(reference, weekday, allow_today=modifier == "this")
        return done(reference + timedelta(days=delta), "next occurrence of weekday")

    if re.search(r"\bnext (?:class|lecture|time|session)\b", lower):
        if not class_days:
            return done(None, "next class (needs --class-days)", ambiguous=True)
        delta = min(_days_until(reference, d, allow_today=False) for d in class_days)
        return done(reference + timedelta(days=delta), "next class day")
    if re.search(r"\bnext week\b", lower):
        return done(
            reference + timedelta(days=7), "next week (same weekday)", ambiguous=True
        )

    # 4. "the 15th" with no month: the next 15th on or after the lecture.
    m = BARE_DAY.search(text)
    if m:
        day = _day_value(m, "dnum", "dword") or 0
        for months_ahead in (0, 1, 2):
            year = reference.year + (reference.month - 1 + months_ahead) // 12
            month = (reference.month - 1 + months_ahead) % 12 + 1
            try:
                candidate = date(year, month, day)
            except ValueError:
                continue
            if candidate >= reference:
                return done(candidate, "day of month", ambiguous=True)

    if re.search(r"\bend of (?:the )?(?:term|semester|quarter)\b", lower):
        return done(None, "end of term (date not known)", ambiguous=True)
    return done(None, "no date found in phrase")
