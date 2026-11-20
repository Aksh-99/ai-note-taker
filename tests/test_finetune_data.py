"""Tests for the fine-tuning dataset builder (Step 6.1).

Everything runs offline: lectures are tiny folders with a hand-written
reference.json, and audio is either a fake loader returning a synthetic
waveform or (for the CLI test) a short WAV decoded by the real ffmpeg.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from notetaker.audio import save_wav
from notetaker.cli import main
from notetaker.finetune_data import (
    Cue,
    FinetuneDataError,
    Segment,
    assign_splits,
    build_dataset,
    check_leakage,
    clean_caption_text,
    find_lectures,
    load_cues,
    load_metadata,
    merge_cues,
    summary_markdown,
)

SR = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cue(start: float, end: float, text: str = "words here") -> Cue:
    return Cue(start, end, text)


def make_lecture(
    root: Path,
    lecture_id: str,
    cues: list[tuple[float, float, str]],
    *,
    course: str | None = None,
    audio_seconds: float | None = None,
) -> Path:
    """Write a lecture folder like `notetaker evalset` does."""
    folder = root / lecture_id
    folder.mkdir(parents=True)
    (folder / "reference.json").write_text(
        json.dumps([{"start": s, "end": e, "text": t} for s, e, t in cues]),
        encoding="utf-8",
    )
    if course is not None:
        (folder / "meta.json").write_text(json.dumps({"course": course}))
    if audio_seconds is not None:
        # A ramp makes every sample position identifiable, so tests can check
        # that each clip was cut from exactly the right place.
        n = round(audio_seconds * SR)
        audio = (np.arange(n, dtype=np.float32) % 1000) / 2000
        save_wav(audio, folder / "audio.flac", SR)  # ffmpeg sniffs the format
    return folder


def fake_load(path: Path, sample_rate: int) -> np.ndarray:
    """Read the WAV written by make_lecture without ffmpeg."""
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768


def four_lectures(root: Path, *, with_audio: bool = True) -> None:
    """Four lectures, each 60 s of audio with captions every 5 s."""
    for i, lecture_id in enumerate(["lec-a", "lec-b", "lec-c", "lec-d"]):
        cues = [
            (t, t + 4.5, f"Lecture {lecture_id} sentence number {t:g} about eigen")
            for t in range(0, 55, 5)
        ]
        make_lecture(
            root,
            lecture_id,
            cues,
            course=f"course-{i}",
            audio_seconds=60.0 if with_audio else None,
        )


# ---------------------------------------------------------------------------
# Cleaning and reading cues
# ---------------------------------------------------------------------------


def test_clean_caption_text_removes_markup_keeps_case_and_punctuation():
    assert clean_caption_text(">> PROFESSOR: So, [LAUGHTER] it's Ax = b.") == (
        "So, it's Ax = b."
    )
    assert clean_caption_text("[MUSIC]") == ""


def test_load_cues_sorts_and_validates(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            [{"start": 5, "end": 6, "text": "b"}, {"start": 1, "end": 2, "text": "a"}]
        )
    )
    assert [c.text for c in load_cues(path)] == ["a", "b"]
    path.write_text("[{}]")
    with pytest.raises(FinetuneDataError, match="Could not read"):
        load_cues(path)


# ---------------------------------------------------------------------------
# merge_cues
# ---------------------------------------------------------------------------


def test_merge_joins_neighbours_up_to_max_seconds():
    cues = [cue(t, t + 4.8, f"c{t}") for t in range(0, 60, 5)]
    segments, stats = merge_cues(cues, "x", max_seconds=30, pad=0)
    # 0..29.8 holds six cues (last ends at 29.8), the next six form a second.
    assert [s.text.split()[0] for s in segments] == ["c0", "c30"]
    assert segments[0].end == pytest.approx(29.8)
    assert all(s.duration <= 30 for s in segments)
    assert stats.cues == 12
    assert [s.index for s in segments] == [0, 1]


def test_merge_cuts_at_long_silence():
    cues = [cue(0, 4), cue(4.5, 8), cue(12, 15)]  # 4 s gap before the third
    segments, _ = merge_cues(cues, "x", max_gap=1.5, pad=0)
    assert [(s.start, s.end) for s in segments] == [(0, 8), (12, 15)]


def test_merge_treats_overlapping_cues_as_no_gap():
    segments, _ = merge_cues([cue(0, 4), cue(3.5, 7)], "x", pad=0)
    assert len(segments) == 1 and segments[0].end == 7


def test_merge_drops_empty_long_and_short():
    cues = [
        cue(0, 3, "[APPLAUSE]"),  # empty after cleaning
        cue(10, 45, "a monologue"),  # one cue longer than 30 s
        cue(50, 50.4, "hi"),  # too short on its own
        cue(60, 64, "kept"),
    ]
    segments, stats = merge_cues(cues, "x", min_seconds=1.0, pad=0)
    assert [s.text for s in segments] == ["kept"]
    assert (stats.empty_after_cleaning, stats.too_long, stats.too_short) == (1, 1, 1)


def test_padding_only_uses_half_the_silence():
    # Gap of 0.2 s between groups would need max_gap < 0.2 to split them.
    cues = [cue(1.0, 4.0, "a"), cue(4.2, 7.0, "b")]
    segments, _ = merge_cues(cues, "x", max_gap=0.1, pad=0.5)
    first, second = segments
    assert first.start == pytest.approx(0.5)  # full pad into leading silence
    assert first.end == pytest.approx(4.1)  # only half of the 0.2 s gap
    assert second.start == pytest.approx(4.1)
    assert first.end <= second.start  # padding never makes clips overlap


def test_padding_clamped_to_zero_audio_end_and_window():
    segments, _ = merge_cues([cue(0.1, 5)], "x", pad=0.5, audio_seconds=5.2)
    assert segments[0].start == 0.0 and segments[0].end == pytest.approx(5.2)
    # A group of exactly 30 s cannot be padded past the window.
    segments, _ = merge_cues([cue(10, 25), cue(25, 40)], "x", pad=0.5)
    assert segments[0].duration == pytest.approx(30.0)


def test_merge_drops_segments_past_audio_end():
    segments, stats = merge_cues(
        [cue(0, 4), cue(10, 14)], "x", audio_seconds=12, max_gap=1
    )
    assert len(segments) == 1 and stats.past_audio_end == 1


@pytest.mark.parametrize(
    "options",
    [
        {"max_seconds": 31},
        {"max_seconds": 0},
        {"min_seconds": 40},
        {"max_gap": -1},
        {"pad": -0.1},
    ],
)
def test_merge_rejects_bad_options(options):
    with pytest.raises(ValueError):
        merge_cues([cue(0, 1)], "x", **options)


# ---------------------------------------------------------------------------
# assign_splits
# ---------------------------------------------------------------------------


def test_assign_splits_is_deterministic_and_complete():
    groups = {f"l{i}": f"l{i}" for i in range(8)}
    first = assign_splits(groups, seed=3)
    assert first == assign_splits(groups, seed=3)
    counts = {s: list(first.values()).count(s) for s in ("train", "val", "test")}
    assert counts == {"train": 4, "val": 2, "test": 2}
    assert assign_splits(groups, seed=4) != first  # seed changes the split


def test_adding_a_lecture_does_not_reshuffle_old_ones_relative_order():
    groups = {f"l{i}": f"l{i}" for i in range(4)}
    before = assign_splits(groups, val_fraction=0, test_fraction=0.25)
    after = assign_splits({**groups, "l9": "l9"}, val_fraction=0, test_fraction=0.2)
    # Same test count (1); the new lecture either takes the slot or not, but
    # no other old lecture moves into test.
    moved = {k for k in groups if before[k] != after[k]}
    assert moved <= {k for k, v in before.items() if v == "test"}


def test_explicit_splits_and_unknown_ids():
    groups = {"a": "a", "b": "b", "c": "c"}
    split = assign_splits(groups, test=["b"], val=["c"])
    assert split == {"a": "train", "b": "test", "c": "val"}
    with pytest.raises(FinetuneDataError, match="Unknown"):
        assign_splits(groups, test=["zzz"])


def test_groups_stay_together():
    groups = {"a1": "A", "a2": "A", "b1": "B", "c1": "C"}
    split = assign_splits(groups, test=["a1"], val_fraction=0)
    assert split["a1"] == split["a2"] == "test"
    with pytest.raises(FinetuneDataError, match="cannot be split"):
        assign_splits(groups, test=["a1"], val=["a2"])


def test_needs_a_training_lecture():
    with pytest.raises(FinetuneDataError, match="No lectures left"):
        assign_splits({"a": "a", "b": "b"})
    with pytest.raises(FinetuneDataError, match="fraction"):
        assign_splits({"a": "a"}, val_fraction=1.0)
    with pytest.raises(FinetuneDataError, match="No lectures"):
        assign_splits({})


# ---------------------------------------------------------------------------
# check_leakage
# ---------------------------------------------------------------------------


def seg(lecture_id: str, text: str = "short", index: int = 0) -> Segment:
    return Segment(lecture_id, index, 0.0, 1.0, text)


def test_check_leakage_clean_split_passes():
    report = check_leakage(
        {"train": [seg("a")], "val": [seg("b")], "test": [seg("c")]},
        {"a": "a", "b": "b", "c": "c"},
    )
    assert report.ok and not report.warnings


def test_check_leakage_catches_lecture_and_group_overlap():
    report = check_leakage(
        {"train": [seg("a"), seg("b")], "test": [seg("a", index=1), seg("c")]},
        {"a": "A", "b": "B", "c": "B"},
    )
    assert not report.ok
    assert any("lecture a" in e for e in report.errors)
    assert any("group 'B'" in e for e in report.errors)


def test_check_leakage_warns_on_repeated_long_sentences_only():
    long_text = "Today we will prove that every symmetric matrix has real eigenvalues."
    report = check_leakage(
        {
            "train": [seg("a", long_text), seg("a", "Any questions?", 1)],
            "test": [seg("b", long_text.upper()), seg("b", "any questions", 1)],
        },
        {"a": "a", "b": "b"},
    )
    assert report.ok  # duplicates are warnings, not errors
    assert len(report.warnings) == 1 and "symmetric" in report.warnings[0]


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------


def test_build_writes_audiofolder_layout(tmp_path):
    four_lectures(tmp_path / "ocw")
    out = tmp_path / "ft"
    summary = build_dataset(tmp_path / "ocw", out, load=fake_load)

    lectures = [lec for s in summary.splits.values() for lec in s.lectures]
    assert sorted(lectures) == ["lec-a", "lec-b", "lec-c", "lec-d"]
    assert {len(summary.splits[s].lectures) for s in ("val", "test")} == {1}

    for split in ("train", "val", "test"):
        rows = load_metadata(out / split)
        assert len(rows) == summary.splits[split].clips > 0
        # Every clip comes from a lecture assigned to this split.
        assert {r["lecture_id"] for r in rows} == set(summary.splits[split].lectures)
        for row in rows:
            path = out / split / row["file_name"]
            with wave.open(str(path), "rb") as handle:
                seconds = handle.getnframes() / handle.getframerate()
            assert seconds == pytest.approx(row["duration"], abs=1e-3)
            assert seconds <= 30.0
            assert row["text"].startswith("Lecture ")

    saved = json.loads((out / "dataset.json").read_text())
    assert saved["splits"]["train"]["clips"] == summary.splits["train"].clips
    assert saved["leakage"]["errors"] == []


def test_clips_are_cut_from_the_right_samples(tmp_path):
    make_lecture(tmp_path / "ocw", "only", [(2.0, 5.0, "hello")], audio_seconds=8)
    make_lecture(tmp_path / "ocw", "other", [(1.0, 3.0, "bye")], audio_seconds=4)
    out = tmp_path / "ft"
    build_dataset(
        tmp_path / "ocw", out, test=["other"], val_fraction=0, pad=0, load=fake_load
    )
    row = load_metadata(out / "train")[0]
    clip = fake_load(out / "train" / row["file_name"], SR)
    full = fake_load(tmp_path / "ocw" / "only" / "audio.flac", SR)
    np.testing.assert_allclose(clip, full[2 * SR : 5 * SR], atol=1e-4)


def test_dry_run_reads_only_captions(tmp_path):
    four_lectures(tmp_path / "ocw", with_audio=False)

    def no_audio(path, sr):
        raise AssertionError("dry run must not load audio")

    summary = build_dataset(
        tmp_path / "ocw", tmp_path / "ft", dry_run=True, load=no_audio
    )
    assert summary.dry_run and sum(s.clips for s in summary.splits.values()) > 0
    assert not (tmp_path / "ft").exists()
    assert "| train |" in summary_markdown(summary)


def test_group_by_course_keeps_courses_together(tmp_path):
    root = tmp_path / "ocw"
    for lecture_id, course in [("a1", "A"), ("a2", "A"), ("b1", "B"), ("c1", "C")]:
        make_lecture(root, lecture_id, [(0, 3, "x y z")], course=course)
    summary = build_dataset(
        root, tmp_path / "ft", group_by="course", test=["a2"], val_fraction=0,
        dry_run=True,
    )  # fmt: skip
    assert summary.splits["test"].lectures == ["a1", "a2"]


def test_lectures_without_audio_are_skipped(tmp_path):
    four_lectures(tmp_path / "ocw")
    (tmp_path / "ocw" / "lec-d" / "audio.flac").unlink()
    summary = build_dataset(tmp_path / "ocw", tmp_path / "ft", load=fake_load)
    assert summary.skipped_lectures == ["lec-d"]
    assert "Skipped" in summary_markdown(summary)


def test_refuses_to_overwrite_without_force(tmp_path):
    four_lectures(tmp_path / "ocw")
    out = tmp_path / "ft"
    (out / "notes").mkdir(parents=True)  # something of the user's
    build_dataset(tmp_path / "ocw", out, load=fake_load)
    with pytest.raises(FinetuneDataError, match="--force"):
        build_dataset(tmp_path / "ocw", out, load=fake_load)
    first = len(load_metadata(out / "train"))
    build_dataset(tmp_path / "ocw", out, load=fake_load, force=True)
    # Rebuilt, not appended to, and the user's folder is untouched.
    assert len(load_metadata(out / "train")) == first
    assert (out / "notes").is_dir()


def test_missing_or_empty_eval_dir(tmp_path):
    with pytest.raises(FinetuneDataError, match="notetaker evalset"):
        find_lectures(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FinetuneDataError, match="No lecture folders"):
        find_lectures(tmp_path / "empty")
    make_lecture(tmp_path / "noaudio", "x", [(0, 2, "a")])
    with pytest.raises(FinetuneDataError, match="audio.flac"):
        build_dataset(tmp_path / "noaudio", tmp_path / "ft")


# ---------------------------------------------------------------------------
# CLI (real ffmpeg decoding of the tiny WAV files)
# ---------------------------------------------------------------------------


def test_cli_dry_run_and_build(tmp_path, capsys):
    four_lectures(tmp_path / "ocw")
    out = tmp_path / "ft"
    base = ["finetune-data", "--eval-dir", str(tmp_path / "ocw"), "-o", str(out)]

    assert main([*base, "--dry-run", "--test", "lec-b"]) == 0
    printed = capsys.readouterr().out
    assert "| test | lec-b |" in printed and "nothing was written" in printed

    assert main([*base, "--test", "lec-b", "--val", "lec-c"]) == 0
    assert "Wrote" in capsys.readouterr().out
    assert {r["lecture_id"] for r in load_metadata(out / "test")} == {"lec-b"}

    assert main(base) == 1  # exists already, no --force
    assert "--force" in capsys.readouterr().out
    assert main([*base, "--test", "nope"]) == 1
