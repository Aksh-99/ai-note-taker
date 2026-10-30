"""Tests for WER / CER evaluation (src/notetaker/wer.py and `notetaker wer`).

Expected numbers are worked out by hand in the comments, so each test also
documents how the metric is computed.
"""

import json

import pytest

from notetaker.cli import main
from notetaker.wer import (
    ErrorCounts,
    PairScore,
    SystemReport,
    find_pairs,
    load_timed_text,
    safe_filename,
    score_files,
    score_texts,
    summary_markdown,
    system_markdown,
    write_reports,
)

# --- ErrorCounts --------------------------------------------------------------


def test_error_counts_rate_and_length():
    c = ErrorCounts(hits=7, substitutions=1, deletions=2, insertions=1)
    assert c.reference_length == 10  # H + S + D
    assert c.errors == 4  # S + D + I
    assert c.rate == pytest.approx(0.4)


def test_error_counts_empty_rate_is_zero():
    assert ErrorCounts().rate == 0.0


def test_error_counts_add_pools_counts():
    a = ErrorCounts(hits=9, substitutions=1)  # 10% WER on 10 words
    b = ErrorCounts(hits=50, deletions=40, insertions=10)  # 50% WER on 90 words
    total = a + b
    assert total == ErrorCounts(hits=59, substitutions=1, deletions=40, insertions=10)
    # Pooled: 51 errors / 100 words = 51%, NOT the mean of 10% and ~55.6%.
    assert total.rate == pytest.approx(0.51)


# --- score_texts --------------------------------------------------------------


def test_identical_texts_score_zero():
    s = score_texts("the eigenvalues of a matrix", "the eigenvalues of a matrix")
    assert s.wer == 0.0
    assert s.cer == 0.0
    assert s.words.hits == 5


def test_one_substitution_one_deletion_one_insertion():
    # ref: one two three four five six seven eight        (8 words)
    # hyp: one two tree  four      six seven eight nine
    # alignment: three->tree (S), "five" missing (D), "nine" extra (I)
    s = score_texts(
        "one two three four five six seven eight",
        "one two tree four six seven eight nine",
    )
    assert s.words == ErrorCounts(hits=6, substitutions=1, deletions=1, insertions=1)
    assert s.wer == pytest.approx(3 / 8)
    assert s.top_substitutions == [("three", "tree", 1)]


def test_wer_can_exceed_one():
    # 1 reference word, model says 3 words: 1 S + 2 I = 3 errors / 1 word.
    s = score_texts("hello", "yellow fellow mellow")
    assert s.wer == pytest.approx(3.0)


def test_empty_hypothesis_is_all_deletions():
    s = score_texts("a b c", "")
    assert s.words == ErrorCounts(deletions=3)
    assert s.wer == 1.0


def test_empty_reference_raises():
    with pytest.raises(ValueError, match="empty"):
        score_texts("[MUSIC]", "hello", name="lec")


def test_normalization_removes_style_differences():
    ref = "PROFESSOR: So, there are 3 eigenvalues. [LAUGHTER]"
    hyp = "so there are three eigenvalues"
    assert score_texts(ref, hyp).wer == 0.0
    # Without normalization the same pair looks terrible.
    raw = score_texts(ref, hyp, normalize=False)
    assert raw.wer > 0.5


def test_cer_is_gentler_than_wer_on_split_words():
    # "eigenvalues" -> "eigen values": 1 S + 1 I at word level (2 / 2 = 100%),
    # but only 1 inserted space out of 13 characters at character level.
    s = score_texts("the eigenvalues", "the eigen values")
    assert s.wer == pytest.approx(1.0)
    assert s.chars.errors == 1
    assert s.cer == pytest.approx(1 / len("the eigenvalues"))


def test_top_substitutions_sorted_by_count_and_limited():
    ref = "a b a b a c"
    hyp = "x y x y x z"
    s = score_texts(ref, hyp, top_k=2)
    assert s.top_substitutions == [("a", "x", 3), ("b", "y", 2)]


def test_pair_score_round_trips_through_dict():
    s = score_texts("one two three", "one too three", name="lec")
    s.max_seconds = 60.0
    again = PairScore.from_dict(json.loads(json.dumps(s.to_dict())))
    assert again.words == s.words
    assert again.chars == s.chars
    assert again.top_substitutions == s.top_substitutions
    assert again.max_seconds == 60.0
    assert again.name == "lec"


# --- loading files --------------------------------------------------------------


def _write_reference(path, cues):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([{"start": a, "end": b, "text": t} for a, b, t in cues]),
        encoding="utf-8",
    )


def _write_transcript(path, segments):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "language": "en",
                "duration": 100.0,
                "segments": [{"start": a, "end": b, "text": t} for a, b, t in segments],
            }
        ),
        encoding="utf-8",
    )


def test_load_timed_text_cuts_by_start_time(tmp_path):
    ref = tmp_path / "reference.json"
    _write_reference(ref, [(0, 5, "one"), (5, 12, "two"), (12, 20, "three")])
    assert load_timed_text(ref).split() == ["one", "two", "three"]
    # "two" starts at 5 < 10 so it is kept whole; "three" starts at 12.
    assert load_timed_text(ref, max_seconds=10).split() == ["one", "two"]


def test_load_timed_text_reads_transcript_json_and_plain_text(tmp_path):
    hyp = tmp_path / "transcript.json"
    _write_transcript(hyp, [(0, 3, "hello"), (3, 6, "world")])
    assert load_timed_text(hyp).split() == ["hello", "world"]
    txt = tmp_path / "ref.txt"
    txt.write_text("plain words", encoding="utf-8")
    assert load_timed_text(txt) == "plain words"
    with pytest.raises(ValueError, match="timestamps"):
        load_timed_text(txt, max_seconds=5)


def test_score_files_uses_cut_on_both_sides(tmp_path):
    ref = tmp_path / "lec" / "reference.json"
    hyp = tmp_path / "lec" / "transcript.json"
    _write_reference(ref, [(0, 5, "the matrix"), (60, 65, "is singular")])
    _write_transcript(hyp, [(0, 5, "the matrix"), (60, 65, "is circular")])
    assert score_files(ref, hyp).wer == pytest.approx(1 / 4)
    cut = score_files(ref, hyp, max_seconds=30)
    assert cut.wer == 0.0
    assert cut.max_seconds == 30
    assert cut.name == "lec"  # defaults to the transcript's folder name


def test_find_pairs_matches_by_id_and_reports_missing(tmp_path):
    refs, hyps = tmp_path / "refs", tmp_path / "hyps"
    for lecture_id in ["b-lec", "a-lec", "c-lec"]:
        _write_reference(refs / lecture_id / "reference.json", [(0, 1, "x")])
    for lecture_id in ["a-lec", "b-lec"]:
        _write_transcript(hyps / lecture_id / "transcript.json", [(0, 1, "x")])
    pairs, missing = find_pairs(refs, hyps)
    assert [p[0] for p in pairs] == ["a-lec", "b-lec"]  # sorted
    assert missing == ["c-lec"]


# --- reports --------------------------------------------------------------------


def _report(system, pairs):
    return SystemReport(
        system=system,
        scores=[score_texts(r, h, name=n) for n, r, h in pairs],
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_system_report_pools_and_round_trips():
    report = _report(
        "whisper-small",
        [("short", "a b", "a x"), ("long", "a b c d e f g h", "a b c d e f g h")],
    )
    # 1 error over 2 + 8 = 10 words, not mean(50%, 0%) = 25%.
    assert report.wer == pytest.approx(0.1)
    again = SystemReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert again.wer == pytest.approx(report.wer)
    assert again.system == "whisper-small"


def test_markdown_tables():
    report = _report("whisper-tiny", [("lec1", "the cat sat", "the cat sit")])
    md = system_markdown(report)
    assert "| lec1 | 3 | 33.33% | 1 | 0 | 0 |" in md
    assert "| sat | sit | 1 |" in md
    other = _report("whisper-small", [("lec2", "hello world", "hello world")])
    summary = summary_markdown([report, other])
    lines = summary.splitlines()
    assert "| System | Pooled WER | Pooled CER | lec1 | lec2 |" in lines
    # Best system first, and "-" where a system skipped a lecture.
    rows = [line for line in lines if line.startswith("| whisper")]
    assert rows[0].startswith("| whisper-small | 0.00%")
    assert rows[1].endswith("| 33.33% | - |")


def test_safe_filename():
    assert safe_filename("whisper small/int8") == "whisper-small-int8"
    assert safe_filename("..") == "system"


def test_write_reports_rebuilds_summary_from_all_systems(tmp_path):
    write_reports(_report("tiny", [("lec", "a b c d", "a b x d")]), tmp_path)
    paths = write_reports(_report("small", [("lec", "a b c d", "a b c d")]), tmp_path)
    assert [p.name for p in paths] == ["small.json", "small.md", "summary.md"]
    (tmp_path / "unrelated.json").write_text("[1, 2]")  # must be ignored
    write_reports(_report("base", [("lec", "a b c d", "a b c")]), tmp_path)
    summary = (tmp_path / "summary.md").read_text()
    assert "| tiny |" in summary and "| small |" in summary and "| base |" in summary


# --- CLI ------------------------------------------------------------------------


def test_cli_pair_mode(tmp_path, capsys):
    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    ref.write_text("The matrix is singular.", encoding="utf-8")
    hyp.write_text("the matrix is circular", encoding="utf-8")
    assert main(["wer", str(ref), str(hyp)]) == 0
    out = capsys.readouterr().out
    assert "WER 25.00%" in out
    assert "S=1 D=0 I=0 of 4 words" in out


def test_cli_folder_mode_writes_reports(tmp_path, capsys):
    refs, hyps, out_dir = tmp_path / "refs", tmp_path / "small", tmp_path / "rep"
    _write_reference(refs / "lec1" / "reference.json", [(0, 2, "a b c d")])
    _write_reference(refs / "lec2" / "reference.json", [(0, 2, "e f")])
    _write_transcript(hyps / "lec1" / "transcript.json", [(0, 2, "a b x d")])
    code = main(["wer", "--refs", str(refs), "--hyps", str(hyps), "-o", str(out_dir)])
    assert code == 0
    out = capsys.readouterr().out
    assert "lec2: no" in out  # missing transcript is reported, not fatal
    assert "Pooled over 1 lecture(s): WER 25.00%" in out
    data = json.loads((out_dir / "small.json").read_text())
    assert data["system"] == "small"  # default name = --hyps folder name
    assert data["pooled"]["words"]["substitutions"] == 1
    assert (out_dir / "summary.md").is_file()


@pytest.mark.parametrize(
    "argv",
    [
        ["wer"],  # nothing to score
        ["wer", "only-one-file.txt"],
        ["wer", "a.txt", "b.txt", "--hyps", "x"],  # both modes at once
        ["wer", "a.txt", "b.txt", "--max-seconds", "0"],
    ],
)
def test_cli_rejects_bad_arguments(argv, capsys):
    assert main(argv) == 1
    assert "Error" in capsys.readouterr().out


def test_cli_folder_mode_with_nothing_to_score(tmp_path, capsys):
    (tmp_path / "refs").mkdir()
    code = main(
        ["wer", "--refs", str(tmp_path / "refs"), "--hyps", str(tmp_path / "h")]
    )
    assert code == 1
    assert "no lecture" in capsys.readouterr().out
