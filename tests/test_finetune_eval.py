"""Tests for comparing fine-tuned vs base Whisper (Step 6.3).

These need no torch: they score hand-written transcripts, which is exactly
what `notetaker finetune-eval --from-transcripts` does. The end-to-end run
with a real (tiny, random) model lives in test_finetune_training.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from notetaker.cli import main
from notetaker.finetune import FinetuneConfig, FinetuneError
from notetaker.finetune_eval import (
    BASE,
    TUNED,
    ClipTranscripts,
    evaluate_transcripts,
    load_transcripts,
    paired_bootstrap,
    save_transcripts,
    score_clips,
    term_changes,
    to_markdown,
    training_summary,
    verdict,
    write_report,
)
from notetaker.term_recall import make_term

TERMS = [
    make_term("eigenvalue", ["eigenvalue", "eigenvalues"], domain="la"),
    make_term("matrix", ["matrix", "matrices"], domain="la"),
]


def row(clip: str, lecture: str, ref: str, base: str, tuned: str) -> ClipTranscripts:
    return ClipTranscripts(clip, lecture, ref, base, tuned)


# Two lectures. The fine-tuned model fixes "eigenvalue" in both clips of
# lecture A and makes one new mistake in lecture B.
ROWS = [
    row(
        "a/a_0.wav",
        "lec-a",
        "the eigenvalue of this matrix is two",
        "the eigen valley of this matrix is two",
        "the eigenvalue of this matrix is two",
    ),
    row(
        "a/a_1.wav",
        "lec-a",
        "every eigenvalue is real",
        "every i can value is real",
        "every eigenvalue is real",
    ),
    row(
        "b/b_0.wav",
        "lec-b",
        "now multiply the matrices",
        "now multiply the matrices",
        "now multiply the mattresses",
    ),
]


# ---------------------------------------------------------------------------
# Saving and loading transcripts
# ---------------------------------------------------------------------------


def test_transcripts_round_trip(tmp_path: Path) -> None:
    path = save_transcripts(ROWS, tmp_path / "sub" / "t.jsonl")
    assert load_transcripts(path) == ROWS
    # One JSON object per line.
    assert len(path.read_text().splitlines()) == len(ROWS)


def test_load_transcripts_errors(tmp_path: Path) -> None:
    with pytest.raises(FinetuneError, match="No transcripts file"):
        load_transcripts(tmp_path / "missing.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(FinetuneError, match="no transcripts"):
        load_transcripts(empty)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"clip": "x"}\n')
    with pytest.raises(FinetuneError, match="line 1"):
        load_transcripts(bad)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_clips_counts_errors_per_system() -> None:
    scores, skipped = score_clips(ROWS, TERMS)
    assert skipped == []
    first = scores[0]
    assert first.reference_words == 7
    # "eigenvalue" -> "eigen valley": one substitution plus one insertion.
    assert first.errors(BASE) == 2
    assert first.errors(TUNED) == 0
    assert first.terms[BASE].hits("eigenvalue") == 0
    assert first.terms[TUNED].hits("eigenvalue") == 1
    assert scores[2].errors(BASE) == 0 and scores[2].errors(TUNED) == 1


def test_clips_with_empty_reference_are_skipped() -> None:
    rows = [*ROWS, row("c/c_0.wav", "lec-c", "[MUSIC]", "la la", "")]
    scores, skipped = score_clips(rows, TERMS)
    assert skipped == ["c/c_0.wav"]
    assert len(scores) == 3


def test_term_changes_sorted_by_size_of_change() -> None:
    scores, _ = score_clips(ROWS, TERMS)
    changes = term_changes(scores)
    assert [(c.term, c.said, c.base_hits, c.tuned_hits) for c in changes] == [
        ("eigenvalue", 2, 0, 2),
        ("matrix", 2, 2, 1),
    ]
    assert changes[0].change == 2 and changes[1].change == -1


def test_evaluate_pools_errors_not_rates() -> None:
    report = evaluate_transcripts(ROWS, TERMS, resamples=200)
    base, tuned = report.overall[BASE], report.overall[TUNED]
    total_words = 7 + 4 + 4
    # Base errors: 2 (eigen valley) + 3 ("i can value": 1 sub + 2 ins) + 0.
    assert base.words.reference_length == total_words
    assert base.wer == pytest.approx(5 / total_words)
    assert tuned.wer == pytest.approx(1 / total_words)
    assert base.term_recall == pytest.approx(2 / 4)
    assert tuned.term_recall == pytest.approx(3 / 4)
    assert report.clips == 3
    assert report.clips_per_lecture == {"lec-a": 2, "lec-b": 1}
    assert report.lectures["lec-b"][TUNED].wer == pytest.approx(1 / 4)
    assert report.bootstrap.delta == pytest.approx(-4 / total_words)


def test_evaluate_needs_something_to_score() -> None:
    with pytest.raises(FinetuneError, match="nothing to score"):
        evaluate_transcripts([row("x", "l", "", "a", "b")], TERMS)


def test_examples_show_most_improved_and_worsened() -> None:
    report = evaluate_transcripts(ROWS, TERMS, resamples=10, examples=1)
    kinds = [(e["kind"], e["clip"]) for e in report.examples]
    # a_1 improved most (3 -> 0 errors); b_0 is the only clip that got worse.
    assert kinds == [("improved", "a/a_1.wav"), ("worsened", "b/b_0.wav")]


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_identical_systems_give_a_zero_interval() -> None:
    boot = paired_bootstrap([1, 2, 3], [1, 2, 3], [10, 10, 10], resamples=100)
    assert boot.delta == boot.low == boot.high == 0.0
    assert boot.share_improved == 0.0


def test_bootstrap_consistent_improvement_is_below_zero() -> None:
    rng = np.random.default_rng(1)
    words = rng.integers(20, 60, size=200)
    base = rng.binomial(words, 0.2)
    tuned = np.maximum(base - rng.integers(1, 4, size=200), 0)
    boot = paired_bootstrap(base, tuned, words, resamples=500, seed=3)
    assert boot.low <= boot.delta <= boot.high < 0
    assert boot.share_improved == 1.0
    assert "below zero" in verdict(boot)


def test_bootstrap_noise_only_includes_zero() -> None:
    rng = np.random.default_rng(2)
    words = np.full(300, 40)
    base = rng.binomial(words, 0.2)
    tuned = rng.binomial(words, 0.2)  # same error rate, independent noise
    boot = paired_bootstrap(base, tuned, words, resamples=1000)
    assert boot.low < 0 < boot.high
    assert "cannot tell" in verdict(boot)


def test_bootstrap_is_reproducible_and_validates_input() -> None:
    args = ([3, 1, 4, 1], [2, 1, 3, 2], [10, 12, 9, 11])
    assert paired_bootstrap(*args, seed=7) == paired_bootstrap(*args, seed=7)
    with pytest.raises(ValueError, match="same"):
        paired_bootstrap([1], [1, 2], [3, 4])
    with pytest.raises(ValueError, match="same"):
        paired_bootstrap([], [], [])
    with pytest.raises(ValueError, match="resamples"):
        paired_bootstrap([1], [1], [3], resamples=0)


def test_verdict_for_a_worse_model() -> None:
    boot = paired_bootstrap([0] * 50, [2] * 50, [10] * 50, resamples=50)
    assert "raised WER" in verdict(boot)


# ---------------------------------------------------------------------------
# Training summary and report files
# ---------------------------------------------------------------------------


def write_run(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": FinetuneConfig(max_steps=300).to_json(),
        "plan": {"train": {"hours": 2.5}},
        "parameters": {"trainable": 1_769_472, "total": 243_504_384},
        "best_step": 200,
        "best_val_loss": 0.41,
    }
    (model_dir / "finetune.json").write_text(json.dumps(summary))


def test_training_summary_reads_finetune_json(tmp_path: Path) -> None:
    write_run(tmp_path)
    info = training_summary(tmp_path)
    assert info["base_model"] == "openai/whisper-small"
    assert info["max_steps"] == 300
    assert info["best_step"] == 200
    assert info["trainable_parameters"] == 1_769_472
    assert info["train_hours"] == 2.5
    assert training_summary(tmp_path / "nowhere") == {}


def test_markdown_and_json_report(tmp_path: Path) -> None:
    write_run(tmp_path / "model")
    report = evaluate_transcripts(
        ROWS, TERMS, resamples=100, training=training_summary(tmp_path / "model")
    )
    md = to_markdown(report)
    assert md.startswith("# Fine-tuned vs base Whisper")
    assert "| base | 33.3% |" in md
    assert "| fine-tuned | 6.7% |" in md
    assert "WER change: **-26.7 pp**" in md
    assert "| eigenvalue | 2 | 0 | 2 | +2 |" in md
    assert "| lec-a | 2 |" in md
    assert "Trainable parameters: 1,769,472" in md
    assert "\u2014" not in md  # no em dashes in prose
    paths = write_report(report, tmp_path / "reports")
    assert [p.name for p in paths] == ["finetune.json", "finetune.md"]
    data = json.loads(paths[0].read_text())
    assert data["clips"] == 3
    assert data["overall"]["fine-tuned"]["wer"] == pytest.approx(1 / 15, abs=1e-6)
    assert data["terms"][0] == {
        "term": "eigenvalue",
        "said": 2,
        "base_hits": 0,
        "tuned_hits": 2,
        "change": 2,
    }
    assert data["training"]["best_step"] == 200


def test_markdown_when_nothing_changed() -> None:
    same = [row("a", "l", "the matrix", "the matrix", "the matrix")]
    md = to_markdown(evaluate_transcripts(same, TERMS, resamples=10))
    assert "No term was recalled a different number of times." in md
    assert "## Examples" not in md
    assert "## Training run" not in md


# ---------------------------------------------------------------------------
# CLI (re-score mode: no model needed)
# ---------------------------------------------------------------------------


def test_cli_rescores_saved_transcripts(tmp_path: Path, capsys) -> None:
    transcripts = save_transcripts(ROWS, tmp_path / "t.jsonl")
    terms = tmp_path / "terms.json"
    terms.write_text(json.dumps({"terms": [{"term": "eigenvalue"}]}))
    out = tmp_path / "reports"
    code = main(
        [
            "finetune-eval",
            "--from-transcripts",
            str(transcripts),
            "--terms",
            str(terms),
            "--bootstrap",
            "50",
            "-o",
            str(out),
        ]
    )
    text = capsys.readouterr().out
    assert code == 0, text
    assert "WER 33.3% -> 6.7%" in text
    assert "term recall 0.0% -> 100.0%" in text
    assert (out / "finetune.md").is_file() and (out / "finetune.json").is_file()


def test_cli_limit_and_errors(tmp_path: Path, capsys) -> None:
    transcripts = save_transcripts(ROWS, tmp_path / "t.jsonl")
    out = tmp_path / "reports"
    base_args = ["finetune-eval", "--from-transcripts", str(transcripts), "-o"]
    assert main([*base_args, str(out), "--limit", "1", "--bootstrap", "10"]) == 0
    assert json.loads((out / "finetune.json").read_text())["clips"] == 1

    assert main([*base_args, str(out), "--limit", "0"]) == 1
    assert "--limit" in capsys.readouterr().out
    assert main([*base_args, str(out), "--terms", str(tmp_path / "no.json")]) == 1
    assert "term list" in capsys.readouterr().out
    missing = ["finetune-eval", "--from-transcripts", str(tmp_path / "no.jsonl")]
    assert main(missing) == 1
    assert "No transcripts file" in capsys.readouterr().out
    assert main([*base_args, str(out), "--bootstrap", "0"]) == 1
    assert "resamples" in capsys.readouterr().out
