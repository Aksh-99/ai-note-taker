"""Tests for technical-term recall (src/notetaker/term_recall.py, `notetaker terms`).

Expected numbers are worked out by hand in the comments.
"""

import json
from pathlib import Path

import pytest

from notetaker.cli import main
from notetaker.term_recall import (
    DEFAULT_TERMS_PATH,
    TermCounts,
    TermReport,
    TermScore,
    count_term,
    count_terms,
    load_terms,
    make_term,
    score_term_files,
    score_term_texts,
    summary_markdown,
    system_markdown,
    write_reports,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

EIGEN = make_term(
    "eigenvalue",
    ["eigenvalue", "eigenvalues", "eigen value", "eigen values"],
    domain="linear-algebra",
)
ELIM = make_term(
    "elimination", ["elimination", "Gaussian elimination"], domain="linear-algebra"
)
BACKPROP = make_term(
    "backpropagation",
    ["backpropagation", "back-propagation", "backprop"],
    domain="machine-learning",
)
TERMS = [EIGEN, ELIM, BACKPROP]

# --- make_term / load_terms -----------------------------------------------------


def test_make_term_normalizes_and_sorts_longest_first():
    # "back-propagation" normalizes to two words, so it comes first.
    assert BACKPROP.variants[0] == ("back", "propagation")
    assert ("backprop",) in BACKPROP.variants


def test_make_term_defaults_to_name():
    assert make_term("Binary Search").variants == (("binary", "search"),)


def test_make_term_rejects_empty_spellings():
    with pytest.raises(ValueError, match="empty"):
        make_term("x", ["!!!", "  "])


def test_make_term_deduplicates_spellings():
    # Both spellings normalize to "eigenvalue".
    assert make_term("e", ["Eigenvalue", "eigenvalue."]).variants == (("eigenvalue",),)


def test_load_terms_reads_list(tmp_path):
    path = tmp_path / "terms.json"
    path.write_text(json.dumps({"terms": [{"term": "matrix", "domain": "la"}]}))
    (term,) = load_terms(path)
    assert term.name == "matrix" and term.domain == "la"
    assert term.variants == (("matrix",),)


def test_load_terms_rejects_duplicates(tmp_path):
    path = tmp_path / "terms.json"
    path.write_text(json.dumps([{"term": "a"}, {"term": "a"}]))
    with pytest.raises(ValueError, match="duplicate"):
        load_terms(path)


@pytest.mark.parametrize("payload", [{"terms": []}, {"terms": [{"x": 1}]}, {}])
def test_load_terms_rejects_bad_shapes(tmp_path, payload):
    path = tmp_path / "terms.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_terms(path)


def test_committed_term_list_is_valid():
    # Guards the curated list itself: it must load, and be a useful size.
    terms = load_terms(REPO_ROOT / DEFAULT_TERMS_PATH)
    assert len(terms) >= 40
    assert all(t.domain for t in terms)


# --- counting -------------------------------------------------------------------


def test_count_term_counts_all_variants():
    words = ["the", "eigenvalue", "and", "two", "eigen", "values", "and", "eigenvalues"]
    assert count_term(words, EIGEN) == 3


def test_count_term_does_not_overlap():
    # "gaussian elimination" is one match; the inner "elimination" is not
    # counted a second time.
    words = ["gaussian", "elimination", "then", "elimination", "again"]
    assert count_term(words, ELIM) == 2


def test_count_term_needs_whole_words():
    # "eigen" alone is not a variant, so nothing matches.
    assert count_term(["eigen", "stuff"], EIGEN) == 0


def test_count_term_variant_at_end_of_text():
    # A two-word variant whose first word is the last word must not crash.
    assert count_term(["so", "eigen"], EIGEN) == 0


def test_count_terms_normalizes_text():
    counts = count_terms("Back-propagation! And BACKPROP.", TERMS)
    assert counts == {"eigenvalue": 0, "elimination": 0, "backpropagation": 2}


def test_count_terms_without_normalization_is_literal():
    counts = count_terms("Eigenvalue eigenvalue", [EIGEN], normalize=False)
    assert counts == {"eigenvalue": 1}


# --- TermCounts -----------------------------------------------------------------


def test_term_counts_clipped_recall_and_precision():
    # eigenvalue: said 4, written 2 -> 2 hits
    # elimination: said 1, written 3 -> 1 hit (clipped, extra 2 hurt precision)
    # backprop: said 0, written 1 -> 0 hits
    c = TermCounts(
        reference={"eigenvalue": 4, "elimination": 1, "backpropagation": 0},
        hypothesis={"eigenvalue": 2, "elimination": 3, "backpropagation": 1},
    )
    assert c.total_hits == 3
    assert c.recall == pytest.approx(3 / 5)
    assert c.precision == pytest.approx(3 / 6)
    assert c.misses() == [("eigenvalue", 4, 2)]


def test_term_counts_empty_is_zero():
    c = TermCounts({}, {})
    assert c.recall == 0.0 and c.precision == 0.0


def test_term_counts_add_pools():
    a = TermCounts({"x": 2}, {"x": 1})
    b = TermCounts({"x": 1, "y": 3}, {"y": 3})
    pooled = a + b
    assert pooled.reference == {"x": 3, "y": 3}
    assert pooled.hypothesis == {"x": 1, "y": 3}
    # hits: x min(3,1)=1, y 3 -> 4 of 6
    assert pooled.recall == pytest.approx(4 / 6)


def test_misses_sorted_by_loss_then_name():
    c = TermCounts({"b": 3, "a": 3, "c": 5}, {"c": 1})
    assert [t for t, _, _ in c.misses()] == ["c", "a", "b"]


# --- scoring texts and files ------------------------------------------------------


def test_score_term_texts_end_to_end():
    ref = "Gaussian elimination finds the eigenvalues. Then backprop."
    hyp = "Gaussian elimination finds the I can values. Then back prop."
    score = score_term_texts(ref, hyp, TERMS, name="lec")
    # said: elimination 1, eigenvalue 1, backprop 1. Written: elimination 1.
    assert score.recall == pytest.approx(1 / 3)
    assert score.precision == pytest.approx(1.0)
    assert {t for t, _, _ in score.counts.misses()} == {"eigenvalue", "backpropagation"}


def test_score_term_files_respects_max_seconds(tmp_path):
    ref = tmp_path / "ref" / "reference.json"
    hyp = tmp_path / "hyp" / "transcript.json"
    ref.parent.mkdir()
    hyp.parent.mkdir()
    cues = [
        {"start": 0.0, "end": 5.0, "text": "eigenvalue"},
        {"start": 700.0, "end": 705.0, "text": "eigenvalue"},
    ]
    ref.write_text(json.dumps(cues))
    hyp.write_text(json.dumps({"segments": cues[:1]}))
    full = score_term_files(ref, hyp, TERMS)
    cut = score_term_files(ref, hyp, TERMS, max_seconds=600)
    assert full.recall == pytest.approx(0.5)
    assert cut.recall == pytest.approx(1.0)
    assert cut.max_seconds == 600
    assert full.name == "hyp"  # defaults to the transcript's folder name


def test_term_score_round_trips_through_dict():
    score = TermScore("lec", TermCounts({"x": 2, "y": 0}, {"x": 1}), max_seconds=60)
    data = score.to_dict()
    assert data["reference_counts"] == {"x": 2}  # zeros dropped
    back = TermScore.from_dict(data)
    assert back.recall == pytest.approx(0.5)
    assert back.max_seconds == 60


# --- reports ----------------------------------------------------------------------


def _report(system="sys", hyp_eigen=1):
    scores = [
        TermScore("lec-a", TermCounts({"eigenvalue": 2}, {"eigenvalue": hyp_eigen})),
        TermScore("lec-b", TermCounts({"backpropagation": 2}, {"backpropagation": 2})),
    ]
    domains = {"eigenvalue": "linear-algebra", "backpropagation": "machine-learning"}
    return TermReport(system=system, scores=scores, domains=domains)


def test_report_pools_and_breaks_down_by_domain():
    report = _report()
    # hits: eigen 1 of 2, backprop 2 of 2 -> 3 of 4
    assert report.recall == pytest.approx(0.75)
    assert report.recall_by_domain() == {
        "linear-algebra": (1, 2),
        "machine-learning": (2, 2),
    }


def test_system_markdown_contains_tables():
    md = system_markdown(_report())
    assert "| **pooled** | 4 | 3 | **75.0%** |" in md
    assert "## Recall by domain" in md
    assert "| eigenvalue | 2 | 1 | 1 |" in md


def test_summary_sorts_best_recall_first():
    md = summary_markdown([_report("worse", hyp_eigen=0), _report("better", 2)])
    assert md.index("| better |") < md.index("| worse |")


def test_write_reports_round_trip_and_summary(tmp_path):
    write_reports(_report("model-a"), tmp_path)
    paths = write_reports(_report("model b", hyp_eigen=2), tmp_path)
    assert [p.name for p in paths] == ["model-b.json", "model-b.md", "summary.md"]
    summary = (tmp_path / "summary.md").read_text()
    assert "model-a" in summary and "model b" in summary
    back = TermReport.from_dict(json.loads((tmp_path / "model-b.json").read_text()))
    assert back.recall == pytest.approx(1.0)


# --- CLI --------------------------------------------------------------------------


def _write_terms(path):
    path.write_text(
        json.dumps(
            {
                "terms": [
                    {"term": "eigenvalue", "variants": ["eigenvalue", "eigen value"]}
                ]
            }
        )
    )
    return path


def test_cli_pair_mode(tmp_path, capsys):
    terms = _write_terms(tmp_path / "t.json")
    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    ref.write_text("Eigenvalue and eigenvalue.")
    hyp.write_text("eigen value and I can value")
    assert main(["terms", str(ref), str(hyp), "--terms", str(terms)]) == 0
    out = capsys.readouterr().out
    assert "term recall 50.0% (1 of 2 said)" in out
    assert "eigenvalue (1/2)" in out


def test_cli_folder_mode_writes_report(tmp_path, capsys):
    terms = _write_terms(tmp_path / "t.json")
    refs, hyps, out_dir = tmp_path / "refs", tmp_path / "hyps" / "tiny", tmp_path / "o"
    for lecture in ("lec1", "lec2"):
        (refs / lecture).mkdir(parents=True)
        (refs / lecture / "reference.json").write_text(
            json.dumps([{"start": 0, "end": 1, "text": "eigenvalue"}])
        )
    (hyps / "lec1").mkdir(parents=True)
    (hyps / "lec1" / "transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "eigen value"}]})
    )
    args = ["terms", "--terms", str(terms), "--refs", str(refs), "--hyps", str(hyps)]
    assert main([*args, "-o", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert "lec2: no" in out  # missing transcript reported, not fatal
    assert "term recall 100.0%" in out
    assert (out_dir / "tiny.json").is_file()
    assert (out_dir / "summary.md").is_file()


def test_cli_bad_term_file(tmp_path, capsys):
    assert main(["terms", "a", "b", "--terms", str(tmp_path / "nope.json")]) == 1
    assert "could not load term list" in capsys.readouterr().out


def test_cli_needs_files_or_hyps(capsys):
    terms = str(REPO_ROOT / DEFAULT_TERMS_PATH)
    assert main(["terms", "--terms", terms]) == 1
    assert main(["terms", "only-one", "--terms", terms]) == 1
    assert main(["terms", "a", "b", "--max-seconds", "0", "--terms", terms]) == 1
