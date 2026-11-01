"""Tests for the baseline report (src/notetaker/baseline.py, `notetaker baseline`).

All inputs are small hand-made reports, so expected numbers can be checked by
hand in the comments. No models, no audio.
"""

import json
import math

import pytest

from notetaker.baseline import (
    BaselineRow,
    build_rows,
    comparability_notes,
    load_benchmark,
    load_wer_reports,
    model_key,
    params_millions,
    pareto_front,
    recommend,
    step_ups,
    to_markdown,
    write_baseline,
)
from notetaker.benchmark import BenchmarkResult, write_report
from notetaker.cli import main
from notetaker.term_recall import TermCounts, TermReport, TermScore
from notetaker.term_recall import write_reports as write_term_reports
from notetaker.wer import ErrorCounts, PairScore, SystemReport
from notetaker.wer import write_reports as write_wer_reports

# --- helpers ------------------------------------------------------------------


def wer_report(system, errors_by_lecture, n=100, max_seconds=None):
    """A SystemReport where lecture i has `errors` substitutions out of n words."""
    scores = [
        PairScore(
            name=name,
            words=ErrorCounts(hits=n - e, substitutions=e),
            chars=ErrorCounts(hits=5 * n - e, substitutions=e),
            max_seconds=max_seconds,
        )
        for name, e in errors_by_lecture.items()
    ]
    return SystemReport(system=system, scores=scores)


def term_report(system, said, written):
    counts = TermCounts(
        reference={"eigenvalue": said}, hypothesis={"eigenvalue": written}
    )
    return TermReport(system=system, scores=[TermScore("lec1", counts)])


def bench(model, transcribe_s, audio_s=100.0, device="cpu", compute="int8"):
    return BenchmarkResult(
        model=model,
        device=device,
        compute_type=compute,
        audio_seconds=audio_s,
        load_seconds=1.0,
        transcribe_seconds=[transcribe_s],
        peak_rss_mb=500.0,
        baseline_rss_mb=200.0,
    )


def row(model, wer, rtf, params=None):
    return BaselineRow(model=model, wer=wer, rtf=rtf, params_m=params)


# --- model_key / params_millions ----------------------------------------------


@pytest.mark.parametrize(
    "label, key",
    [
        ("small", "small"),
        ("whisper-small", "small"),
        ("Whisper Small", "small"),
        ("faster-whisper_small", "small"),
        ("whisper-small-int8", "small-int8"),
        ("whisper-small-raw", "small-raw"),
        ("whisper/large-v3", "large-v3"),
    ],
)
def test_model_key(label, key):
    assert model_key(label) == key


@pytest.mark.parametrize(
    "key, params",
    [
        ("tiny", 39),
        ("small", 244),
        ("small.en", 244),
        ("small-int8", 244),
        ("large-v3-int8", 1550),
        ("large-v3-turbo", 809),
        ("distil-large-v3", 756),
        ("smallish", None),  # must not match "small" without a separator
        ("my-model", None),
    ],
)
def test_params_millions(key, params):
    assert params_millions(key) == params


# --- build_rows ---------------------------------------------------------------


def test_build_rows_joins_three_reports_by_model():
    rows, warnings = build_rows(
        [
            wer_report("whisper-small", {"lec1": 10}),
            wer_report("whisper-tiny", {"lec1": 30}),
        ],
        [term_report("whisper-small", said=10, written=8)],
        [bench("small", 40.0), bench("tiny", 10.0)],
    )
    assert warnings == []
    # Sorted by size: tiny (39M) before small (244M).
    assert [r.model for r in rows] == ["tiny", "small"]
    tiny, small = rows
    assert small.wer == pytest.approx(0.10)
    assert small.cer == pytest.approx(10 / 500)
    assert small.term_recall == pytest.approx(0.8)
    assert small.term_precision == pytest.approx(1.0)
    assert small.rtf == pytest.approx(0.4)  # 40 s for 100 s of audio
    assert small.speedup == pytest.approx(2.5)
    assert small.model_rss_mb == pytest.approx(300.0)
    assert tiny.term_recall is None  # no term report for tiny
    assert small.reference_words == 100


def test_build_rows_keeps_models_missing_from_some_reports():
    rows, _ = build_rows([wer_report("whisper-base", {"a": 5})], [], [bench("tiny", 5)])
    by_model = {r.model: r for r in rows}
    assert by_model["base"].rtf is None
    assert by_model["tiny"].wer is None


def test_build_rows_warns_on_duplicate_keys():
    rows, warnings = build_rows(
        [wer_report("small", {"a": 1}), wer_report("whisper-small", {"a": 9})],
        bench_results=[bench("small", 1), bench("small", 2)],
    )
    assert len(rows) == 1
    assert rows[0].wer == pytest.approx(0.01)  # first one kept
    assert rows[0].rtf == pytest.approx(0.01)
    assert len(warnings) == 2


def test_build_rows_turns_nan_into_missing():
    rows, _ = build_rows([], [], [bench("tiny", 1.0, audio_s=0.0)])  # RTF is NaN
    assert rows[0].rtf is None
    assert rows[0].speedup is None


def test_unknown_sizes_sort_last():
    rows, _ = build_rows([wer_report("custom", {"a": 1}), wer_report("base", {"a": 1})])
    assert [r.model for r in rows] == ["base", "custom"]


# --- pareto_front / recommend / step_ups -------------------------------------


def test_pareto_front_drops_dominated_models():
    rows = [
        row("tiny", 0.30, 0.05),
        row("base", 0.20, 0.10),
        row("slowbad", 0.25, 0.50),  # base is better AND faster: dominated
        row("small", 0.12, 0.40),
        row("nospeed", 0.01, None),  # incomplete rows are never on the front
    ]
    assert pareto_front(rows) == {"tiny", "base", "small"}


def test_pareto_ties_are_not_dominated():
    # Identical points do not dominate each other (no strict improvement).
    assert pareto_front([row("a", 0.1, 0.1), row("b", 0.1, 0.1)]) == {"a", "b"}


def test_recommend_picks_most_accurate_within_budget():
    rows = [row("tiny", 0.30, 0.05), row("base", 0.20, 0.10), row("small", 0.12, 0.6)]
    best, reason = recommend(rows, rtf_budget=0.5)
    assert best.model == "base"
    assert "90-minute lecture takes about 9 minutes" in reason  # 90 * 0.10
    best, _ = recommend(rows, rtf_budget=1.0)
    assert best.model == "small"


def test_recommend_breaks_wer_ties_by_speed():
    best, _ = recommend([row("a", 0.1, 0.4), row("b", 0.1, 0.2)])
    assert best.model == "b"


def test_recommend_when_nothing_fits_or_nothing_complete():
    best, reason = recommend([row("small", 0.1, 2.0)], rtf_budget=0.5)
    assert best is None and "fastest is small" in reason
    best, reason = recommend([row("small", 0.1, None)])
    assert best is None and "No model" in reason


def test_step_ups_numbers():
    rows = [
        row("tiny", 0.30, 0.05, 39),
        row("base", 0.20, 0.10, 74),
        row("small", 0.15, None, 244),
    ]
    s1, s2 = step_ups(rows)
    # tiny -> base: 10 points, 10/30 = 33.3% relative, 0.10/0.05 = 2x compute.
    assert (s1.smaller, s1.larger) == ("tiny", "base")
    assert s1.wer_drop_points == pytest.approx(10.0)
    assert s1.wer_drop_relative == pytest.approx(1 / 3)
    assert s1.compute_ratio == pytest.approx(2.0)
    # base -> small: 5 points, 25% relative, unknown compute.
    assert s2.wer_drop_relative == pytest.approx(0.25)
    assert s2.compute_ratio is None


# --- comparability -------------------------------------------------------------


def test_comparability_flags_different_lecture_sets_and_cuts():
    rows, _ = build_rows(
        [
            wer_report("tiny", {"a": 1, "b": 1}),
            wer_report("base", {"a": 1}, max_seconds=600.0),
        ]
    )
    notes = " ".join(comparability_notes(rows))
    assert "different sets of lectures" in notes
    assert "different lengths of audio" in notes
    assert "No benchmark result for: tiny, base" in notes


def test_comparability_flags_mixed_devices():
    rows, _ = build_rows(
        [], [], [bench("tiny", 1), bench("base", 1, device="cuda", compute="float16")]
    )
    notes = " ".join(comparability_notes(rows))
    assert "different devices" in notes
    assert "No WER report for: tiny, base" in notes


def test_comparable_runs_have_no_notes():
    rows, _ = build_rows(
        [wer_report("tiny", {"a": 3}), wer_report("base", {"a": 2})],
        [],
        [bench("tiny", 5), bench("base", 10)],
    )
    assert comparability_notes(rows) == []


# --- markdown and files ----------------------------------------------------------


def _full_rows():
    return build_rows(
        [
            wer_report("whisper-tiny", {"lec1": 30, "lec2": 20}),
            wer_report("whisper-base", {"lec1": 20, "lec2": 10}),
            wer_report("whisper-small", {"lec1": 10, "lec2": 6}),
        ],
        [term_report("whisper-small", 10, 9)],
        [bench("tiny", 5), bench("base", 10), bench("small", 40)],
    )[0]


def test_markdown_contains_table_recommendation_and_steps():
    md = to_markdown(_full_rows(), rtf_budget=0.5, bench_audio="lec1/audio.flac")
    assert "| small | 244 | 8.00% |" in md  # (10 + 6) / 200
    assert "Budget: RTF <= 0.5. small:" in md  # RTF 0.4 fits
    assert "It transcribes 90.0% of spoken technical terms." in md
    assert "| tiny -> base |" in md
    assert "## WER per lecture" in md
    assert "lec1/audio.flac" in md
    assert "## Caveats" not in md  # everything was measured the same way
    assert chr(0x2014) not in md  # no em dashes in generated prose


def test_markdown_handles_missing_data():
    rows, _ = build_rows([wer_report("custom", {"a": 1})])
    md = to_markdown(rows)
    assert "| custom | n/a | 1.00% |" in md
    assert "No model has both" in md
    assert "## Caveats" in md


def test_write_baseline_writes_md_and_json(tmp_path):
    md_path, json_path = write_baseline(_full_rows(), tmp_path / "out" / "baseline.md")
    assert md_path.read_text().startswith("# Baseline")
    data = json.loads(json_path.read_text())
    assert data["recommended"] == "small"
    assert data["pareto_front"] == ["base", "small", "tiny"]
    assert [r["model"] for r in data["rows"]] == ["tiny", "base", "small"]
    assert not any(
        isinstance(v, float) and math.isnan(v) for v in data["rows"][0].values()
    )


def test_loaders_skip_bad_files_and_missing_paths(tmp_path):
    write_wer_reports(wer_report("whisper-tiny", {"a": 1}), tmp_path / "wer")
    (tmp_path / "wer" / "junk.json").write_text("{not json")
    reports, warnings = load_wer_reports(tmp_path / "wer")
    assert [r.system for r in reports] == ["whisper-tiny"]
    assert len(warnings) == 1 and "junk.json" in warnings[0]
    assert load_wer_reports(tmp_path / "nope") == ([], [])
    assert load_benchmark(tmp_path / "nope.json") == ([], None)


# --- CLI ------------------------------------------------------------------------


def test_cli_baseline_end_to_end(tmp_path, capsys):
    """Write real report files with the real writers, then run the command."""
    write_wer_reports(wer_report("whisper-tiny", {"lec1": 30}), tmp_path / "wer")
    write_wer_reports(wer_report("whisper-small", {"lec1": 10}), tmp_path / "wer")
    write_term_reports(term_report("whisper-small", 10, 7), tmp_path / "terms")
    write_report(
        [bench("tiny", 5), bench("small", 40)], tmp_path / "bench", "lec1.flac"
    )
    out = tmp_path / "baseline.md"

    code = main(
        [
            "baseline",
            "--wer", str(tmp_path / "wer"),
            "--terms", str(tmp_path / "terms"),
            "--benchmark", str(tmp_path / "bench" / "benchmark.json"),
            "-o", str(out),
        ]
    )  # fmt: skip
    assert code == 0
    printed = capsys.readouterr().out
    assert "2 model(s). Recommendation: small" in printed
    md = out.read_text()
    assert "| small | 244 | 10.00% |" in md
    assert "70.0%" in md  # term recall 7 / 10
    assert (tmp_path / "baseline.json").exists()


def test_cli_baseline_without_wer_reports_fails(tmp_path, capsys):
    assert main(["baseline", "--wer", str(tmp_path / "none")]) == 1
    assert "no WER reports" in capsys.readouterr().out


def test_cli_baseline_rejects_bad_budget(tmp_path, capsys):
    assert main(["baseline", "--rtf-budget", "0"]) == 1
    assert "--rtf-budget" in capsys.readouterr().out


def test_cli_baseline_works_with_wer_only(tmp_path, capsys):
    write_wer_reports(wer_report("whisper-base", {"lec1": 12}), tmp_path / "wer")
    code = main(
        [
            "baseline",
            "--wer", str(tmp_path / "wer"),
            "--terms", str(tmp_path / "t"),
            "--benchmark", str(tmp_path / "b.json"),
            "-o", str(tmp_path / "r" / "baseline.md"),
        ]
    )  # fmt: skip
    assert code == 0
    printed = capsys.readouterr().out
    assert "speed columns will be n/a" in printed
    assert "| base | 74 | 12.00% |" in (tmp_path / "r" / "baseline.md").read_text()
