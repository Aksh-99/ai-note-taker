"""Tests for the keyword task-candidate scan (src/notetaker/tasks.py)."""

from __future__ import annotations

from notetaker.tasks import (
    TaskList,
    extract_tasks,
    find_task_candidates,
    load_tasks,
    split_sentences,
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
