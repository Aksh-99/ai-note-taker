"""Tests for notetaker.export (timestamps, paragraphs, Markdown, files)."""

from __future__ import annotations

import pytest

from notetaker.export import (
    format_timestamp,
    group_paragraphs,
    to_markdown,
    write_outputs,
)
from notetaker.transcribe import Segment, Transcript


def seg(id, start, end, text):
    return Segment(
        id=id, start=start, end=end, text=text, avg_logprob=-0.2, no_speech_prob=0.01
    )


def make_transcript(segments, duration=120.0):
    return Transcript(
        segments=segments,
        language="en",
        language_probability=1.0,
        duration=duration,
        model="small",
    )


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00"),
        (59.9, "00:00:59"),  # floors, never rounds up
        (761.4, "00:12:41"),
        (3600, "01:00:00"),
        (5400.5, "01:30:00"),  # a 90-minute lecture
    ],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_format_timestamp_rejects_negative():
    with pytest.raises(ValueError):
        format_timestamp(-1)


def test_pause_starts_new_paragraph():
    segments = [
        seg(0, 0.0, 2.0, "Hello."),
        seg(1, 2.3, 4.0, "Today: eigenvalues."),  # 0.3 s gap, same paragraph
        seg(2, 7.0, 9.0, "First, a definition."),  # 3 s gap, new paragraph
    ]
    paras = group_paragraphs(segments, pause_seconds=2.0)
    assert [[s.id for s in p] for p in paras] == [[0, 1], [2]]


def test_long_monologue_is_split_by_max_length():
    # 10 back-to-back 10-second segments, no pauses at all.
    segments = [seg(i, 10.0 * i, 10.0 * (i + 1), f"s{i}") for i in range(10)]
    paras = group_paragraphs(segments, max_paragraph_seconds=30.0)
    # Every paragraph spans at most 30 s, and nothing is lost or reordered.
    assert all(p[-1].end - p[0].start <= 30.0 for p in paras)
    assert [s.id for p in paras for s in p] == list(range(10))
    assert len(paras) == 4  # 3 + 3 + 3 + 1 segments


def test_empty_segments_are_dropped():
    paras = group_paragraphs([seg(0, 0, 1, ""), seg(1, 1, 2, "Hi.")])
    assert [[s.id for s in p] for p in paras] == [[1]]


def test_group_paragraphs_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        group_paragraphs([], pause_seconds=0)


def test_markdown_has_header_and_timestamped_paragraphs():
    t = make_transcript(
        [
            seg(0, 0.0, 2.0, "Hello."),
            seg(1, 2.3, 4.0, "Today: eigenvalues."),
            seg(2, 65.0, 68.0, "Next topic."),
        ],
        duration=761.4,
    )
    md = to_markdown(t, title="Lecture 1")
    assert md.startswith("# Lecture 1\n")
    assert "- **Duration:** 00:12:41" in md
    assert "- **Model:** whisper-small" in md
    assert "**[00:00:00]** Hello. Today: eigenvalues." in md
    assert "**[00:01:05]** Next topic." in md


def test_markdown_for_silence():
    md = to_markdown(make_transcript([]))
    assert "_No speech detected._" in md


def test_write_outputs_creates_both_files(tmp_path):
    t = make_transcript([seg(0, 0.0, 1.0, "Hi.")])
    json_path, md_path = write_outputs(t, tmp_path / "nested" / "lec", title="Lec")
    assert json_path.name == "transcript.json"
    assert md_path.name == "transcript.md"
    # The JSON must load back into an identical Transcript.
    assert Transcript.load_json(json_path) == t
    assert md_path.read_text(encoding="utf-8").startswith("# Lec")
