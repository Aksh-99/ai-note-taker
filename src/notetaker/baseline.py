"""Baseline report: Whisper model size vs accuracy vs speed, in one table.

Phase 2 produced three separate measurements, each in its own folder:

    reports/benchmark/benchmark.json   speed + memory per model   (Step 1.5)
    reports/wer/<system>.json          WER / CER per model        (Step 2.3)
    reports/terms/<system>.json        technical-term recall      (Step 2.4)

This module joins them into `reports/baseline.md`, the "before" picture that
every later improvement (fine-tuning in Phase 6, quantization in 6.4) is
compared against. It answers the practical question: which model size should
be the default on this machine?

The join problem:
    The three reports name models differently. The benchmark says "small",
    while WER / term reports are named after the transcript folder, e.g.
    "whisper-small". `model_key()` maps every label to one canonical key
    ("small") so rows can be matched. A model missing from one report still
    gets a row; the missing cells show "n/a".

Two ideas from the report worth knowing for interviews:

* Pareto frontier. A model is "dominated" if some other model is at least as
  accurate AND at least as fast, and strictly better in one of the two. The
  models that nobody dominates form the Pareto frontier: the only sensible
  choices, because for any other model there is a free upgrade. Which
  frontier model to pick depends on how you weigh speed against accuracy.

* Speed budget. To turn the frontier into one recommendation we fix a budget
  on the real-time factor (RTF, compute seconds per audio second) and take
  the most accurate model within it. The default budget RTF <= 0.5 means a
  90-minute lecture is transcribed in at most 45 minutes, so notes are ready
  well before the next class.

Everything here is pure Python over JSON files: no models, no downloads, so
it runs in CI and the tests use small hand-made reports.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from notetaker.benchmark import BenchmarkResult
from notetaker.term_recall import TermReport
from notetaker.wer import SystemReport

# Default real-time-factor budget for the recommendation (see module docstring).
DEFAULT_RTF_BUDGET = 0.5

# Parameter counts in millions, from the Whisper paper (Radford et al. 2022,
# Table 1) and the model cards for the later turbo / distil releases. The
# English-only ".en" variants have the same size as their multilingual twins.
WHISPER_PARAMS_M: dict[str, int] = {
    "tiny": 39,
    "base": 74,
    "small": 244,
    "medium": 769,
    "large": 1550,
    "large-v1": 1550,
    "large-v2": 1550,
    "large-v3": 1550,
    "large-v3-turbo": 809,
    "turbo": 809,
    "distil-small": 166,
    "distil-medium": 394,
    "distil-large-v2": 756,
    "distil-large-v3": 756,
}


# ---------------------------------------------------------------------------
# Matching model names across reports
# ---------------------------------------------------------------------------


def model_key(label: str) -> str:
    """Canonical model name used to join the three reports.

    "whisper-small", "Whisper Small", "faster-whisper_small" and "small" all
    become "small". Anything after the size is kept ("small-int8",
    "small-raw"), because those really are different systems.
    """
    key = label.strip().lower()
    # Spaces, underscores and slashes all count as separators.
    key = re.sub(r"[\s_/]+", "-", key)
    for prefix in ("faster-whisper-", "openai-whisper-", "whisper-"):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    return key


def params_millions(key: str) -> int | None:
    """Parameter count for a canonical key, or None if we do not know it.

    Tries an exact match first, then the longest known name that the key
    starts with, so "small.en" and "small-int8" both find "small" and
    "large-v3-int8" finds "large-v3" rather than plain "large". Checking the
    longest names first is what makes that work.
    """
    if key in WHISPER_PARAMS_M:
        return WHISPER_PARAMS_M[key]
    for name in sorted(WHISPER_PARAMS_M, key=len, reverse=True):
        if key.startswith((name + "-", name + ".")):
            return WHISPER_PARAMS_M[name]
    return None


# ---------------------------------------------------------------------------
# One row of the baseline table
# ---------------------------------------------------------------------------


@dataclass
class BaselineRow:
    """Everything we know about one model. Any field can be missing (None)."""

    model: str
    params_m: int | None = None
    # From the WER report.
    wer_system: str | None = None
    wer: float | None = None
    cer: float | None = None
    reference_words: int | None = None
    lectures: list[str] = field(default_factory=list)
    lecture_wer: dict[str, float] = field(default_factory=dict)
    # Seconds scored per lecture (None = whole lecture), for comparability checks.
    max_seconds: list[float | None] = field(default_factory=list)
    # From the term report.
    term_recall: float | None = None
    term_precision: float | None = None
    # From the benchmark.
    rtf: float | None = None
    load_seconds: float | None = None
    model_rss_mb: float | None = None
    peak_rss_mb: float | None = None
    device: str | None = None
    compute_type: str | None = None

    @property
    def speedup(self) -> float | None:
        """Times faster than real time (1 / RTF)."""
        if self.rtf is None or self.rtf <= 0:
            return None
        return 1.0 / self.rtf

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["speedup"] = self.speedup
        return data


def _finite(x: Any) -> float | None:
    """Return x as a float, or None for missing / NaN / infinite values.

    JSON from older runs can contain NaN (Python's json writes it), and a NaN
    would silently poison every comparison (NaN < 1 and NaN > 1 are both
    False), so we turn it into "missing" at the door.
    """
    if x is None:
        return None
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def build_rows(
    wer_reports: Iterable[SystemReport],
    term_reports: Iterable[TermReport] = (),
    bench_results: Iterable[BenchmarkResult] = (),
) -> tuple[list[BaselineRow], list[str]]:
    """Join the three kinds of report into one row per model.

    Returns (rows, warnings). Warnings are human-readable notes, e.g. when two
    reports map to the same model key and one of them had to be ignored.
    Rows are sorted by model size (unknown sizes last), then by name.
    """
    rows: dict[str, BaselineRow] = {}
    warnings: list[str] = []

    def row_for(label: str) -> BaselineRow:
        # dict.setdefault would build a BaselineRow even when one exists;
        # this only creates it the first time a key is seen.
        key = model_key(label)
        if key not in rows:
            rows[key] = BaselineRow(model=key, params_m=params_millions(key))
        return rows[key]

    for report in wer_reports:
        row = row_for(report.system)
        if row.wer_system is not None:
            warnings.append(
                f"WER reports '{row.wer_system}' and '{report.system}' both map to "
                f"model '{row.model}'; kept '{row.wer_system}'."
            )
            continue
        row.wer_system = report.system
        row.wer = report.wer
        row.cer = report.cer
        row.reference_words = report.words.reference_length
        row.lectures = sorted(s.name for s in report.scores)
        row.lecture_wer = {s.name: s.wer for s in report.scores}
        row.max_seconds = sorted(
            {s.max_seconds for s in report.scores},
            key=lambda v: math.inf if v is None else v,
        )

    seen_terms: dict[str, str] = {}
    for report in term_reports:
        row = row_for(report.system)
        if row.model in seen_terms:
            warnings.append(
                f"Term reports '{seen_terms[row.model]}' and '{report.system}' both "
                f"map to model '{row.model}'; kept '{seen_terms[row.model]}'."
            )
            continue
        seen_terms[row.model] = report.system
        row.term_recall = report.recall
        row.term_precision = report.precision

    seen_bench: set[str] = set()
    for result in bench_results:
        row = row_for(result.model)
        if row.model in seen_bench:
            warnings.append(
                f"Benchmark has more than one result for model '{row.model}'; "
                "kept the first."
            )
            continue
        seen_bench.add(row.model)
        row.rtf = _finite(result.rtf)
        row.load_seconds = _finite(result.load_seconds)
        row.model_rss_mb = _finite(result.model_rss_mb)
        row.peak_rss_mb = _finite(result.peak_rss_mb)
        row.device = result.device
        row.compute_type = result.compute_type

    ordered = sorted(
        rows.values(),
        key=lambda r: (r.params_m is None, r.params_m or 0, r.model),
    )
    return ordered, warnings


# ---------------------------------------------------------------------------
# Analysis: Pareto frontier, recommendation, step-ups, comparability
# ---------------------------------------------------------------------------


def _complete(rows: Iterable[BaselineRow]) -> list[BaselineRow]:
    """Rows that have both an accuracy (WER) and a speed (RTF) number."""
    return [r for r in rows if r.wer is not None and r.rtf is not None]


def pareto_front(rows: Sequence[BaselineRow]) -> set[str]:
    """Model keys on the WER-vs-RTF Pareto frontier (lower is better for both).

    Row r is dominated if another row o has o.wer <= r.wer and o.rtf <= r.rtf
    with at least one strict inequality. With a handful of models the simple
    O(n^2) double loop is the clearest way to write it.
    """
    candidates = _complete(rows)
    front: set[str] = set()
    for r in candidates:
        dominated = any(
            o.wer <= r.wer and o.rtf <= r.rtf and (o.wer < r.wer or o.rtf < r.rtf)
            for o in candidates
            if o is not r
        )
        if not dominated:
            front.add(r.model)
    return front


def recommend(
    rows: Sequence[BaselineRow], rtf_budget: float = DEFAULT_RTF_BUDGET
) -> tuple[BaselineRow | None, str]:
    """Most accurate model whose RTF fits the budget, plus a one-line reason.

    Ties on WER go to the faster model. If nothing fits, returns None and
    names the fastest model so the reader knows how far off the budget is.
    """
    candidates = _complete(rows)
    if not candidates:
        return None, "No model has both a WER and a benchmark result yet."
    within = [r for r in candidates if r.rtf <= rtf_budget]
    if not within:
        fastest = min(candidates, key=lambda r: r.rtf)
        return None, (
            f"No model fits the budget RTF <= {rtf_budget:g}; the fastest is "
            f"{fastest.model} at RTF {fastest.rtf:.3f}. Try a GPU, int8, or a "
            "larger budget."
        )
    best = min(within, key=lambda r: (r.wer, r.rtf))
    minutes = 90 * best.rtf
    return best, (
        f"{best.model}: lowest WER ({_pct(best.wer)}) among models with "
        f"RTF <= {rtf_budget:g}; a 90-minute lecture takes about "
        f"{minutes:.0f} minutes on this machine."
    )


@dataclass(frozen=True)
class StepUp:
    """What moving from one model size to the next bigger one buys and costs."""

    smaller: str
    larger: str
    wer_drop_points: float  # absolute WER drop in percentage points
    wer_drop_relative: float  # drop as a fraction of the smaller model's WER
    compute_ratio: float | None  # larger RTF / smaller RTF (None if unknown)


def step_ups(rows: Sequence[BaselineRow]) -> list[StepUp]:
    """Compare each model with the next bigger one that also has a WER.

    Rows are already sorted by size, so consecutive pairs are size steps.
    The relative drop is the number papers quote ("small cuts WER by 30%
    relative to base"); the compute ratio is its price.
    """
    with_wer = [r for r in rows if r.wer is not None and r.params_m is not None]
    out: list[StepUp] = []
    for a, b in zip(with_wer, with_wer[1:], strict=False):
        drop = a.wer - b.wer
        relative = drop / a.wer if a.wer > 0 else 0.0
        ratio = b.rtf / a.rtf if a.rtf and b.rtf is not None else None
        out.append(StepUp(a.model, b.model, 100 * drop, relative, ratio))
    return out


def comparability_notes(rows: Sequence[BaselineRow]) -> list[str]:
    """Warn when the numbers in the table were not measured the same way.

    A WER on the first 10 minutes is not comparable to a WER on whole
    lectures, and neither is a WER over a different set of lectures. Speed
    measured on different devices is not comparable either.
    """
    notes: list[str] = []
    scored = [r for r in rows if r.wer is not None]
    lecture_sets = {tuple(r.lectures) for r in scored}
    if len(lecture_sets) > 1:
        notes.append(
            "Models were scored on different sets of lectures, so their WERs "
            "are not directly comparable. Transcribe the same lectures with "
            "every model."
        )
    cuts = {tuple(r.max_seconds) for r in scored}
    if len(cuts) > 1:
        notes.append(
            "Models were scored on different lengths of audio (--max-seconds "
            "differs); re-score them with the same setting."
        )
    devices = {(r.device, r.compute_type) for r in rows if r.rtf is not None}
    if len(devices) > 1:
        notes.append(
            "Speed was measured with different devices or compute types: "
            + ", ".join(f"{d}/{c}" for d, c in sorted(devices, key=str))
            + "."
        )
    missing_bench = [r.model for r in scored if r.rtf is None]
    if missing_bench:
        notes.append(
            "No benchmark result for: " + ", ".join(missing_bench) + ". Run "
            "`notetaker benchmark` with these models to fill in speed."
        )
    missing_wer = [r.model for r in rows if r.wer is None]
    if missing_wer:
        notes.append(
            "No WER report for: " + ", ".join(missing_wer) + ". Transcribe the "
            "eval set with these models and run `notetaker wer`."
        )
    return notes


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _pct(rate: float | None, digits: int = 1) -> str:
    return "n/a" if rate is None else f"{100 * rate:.{digits}f}%"


def _num(value: float | None, fmt: str) -> str:
    return "n/a" if value is None else format(value, fmt)


def _cut_label(cuts: list[float | None]) -> str:
    if not cuts:
        return "n/a"
    return ", ".join("full" if c is None else f"first {c:g} s" for c in cuts)


def to_markdown(
    rows: Sequence[BaselineRow],
    *,
    rtf_budget: float = DEFAULT_RTF_BUDGET,
    bench_audio: str | None = None,
    warnings: Sequence[str] = (),
    created_at: str | None = None,
) -> str:
    """Render the full baseline report."""
    created_at = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    front = pareto_front(rows)
    best, reason = recommend(rows, rtf_budget)
    scored = [r for r in rows if r.wer is not None]
    lectures = sorted({x for r in scored for x in r.lectures})

    lines = [
        "# Baseline: Whisper model size vs accuracy vs speed",
        "",
        f"Generated by `notetaker baseline` at {created_at}. "
        "This is the reference point for fine-tuning (Phase 6) and quantization.",
        "",
        "## Setup",
        "",
        f"- Evaluation set: {len(lectures)} MIT OpenCourseWare lecture(s): "
        + (", ".join(lectures) if lectures else "none scored yet"),
        f"- Audio scored per lecture: {_cut_label(scored[0].max_seconds)}"
        if scored
        else "- Audio scored per lecture: n/a",
        f"- Speed measured on: {bench_audio or 'n/a'}",
        "- WER / CER are pooled over lectures (total errors / total reference "
        "words); both texts are normalized first (Step 2.2)",
        "- Term recall = technical-term occurrences transcribed / spoken (Step 2.4)",
        "- RTF = transcription time / audio length; lower is faster",
        "",
        "## Results",
        "",
        "| Model | Params (M) | WER | CER | Term recall | Term precision "
        "| RTF | x real time | Model RAM (MB) | Pareto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        star = "yes" if r.model in front else ""
        lines.append(
            f"| {r.model} | {_num(r.params_m, ',')} | {_pct(r.wer, 2)} "
            f"| {_pct(r.cer, 2)} | {_pct(r.term_recall)} | {_pct(r.term_precision)} "
            f"| {_num(r.rtf, '.3f')} | {_num(r.speedup, '.1f')} "
            f"| {_num(r.model_rss_mb, ',.0f')} | {star} |"
        )

    lines += [
        "",
        "Pareto = no other model is both at least as accurate and at least as "
        "fast. Only these models are sensible choices; which one depends on the "
        "speed you can afford.",
        "",
        "## Recommendation",
        "",
        f"Budget: RTF <= {rtf_budget:g}. " + reason,
    ]
    if best is not None and best.term_recall is not None:
        lines.append(
            f"It transcribes {_pct(best.term_recall)} of spoken technical terms."
        )

    steps = step_ups(rows)
    if steps:
        lines += [
            "",
            "## What each size step buys",
            "",
            "| Step | WER drop (points) | WER drop (relative) | Compute cost |",
            "|---|---:|---:|---:|",
        ]
        for s in steps:
            cost = "n/a" if s.compute_ratio is None else f"{s.compute_ratio:.1f}x"
            lines.append(
                f"| {s.smaller} -> {s.larger} | {s.wer_drop_points:+.2f} "
                f"| {100 * s.wer_drop_relative:+.1f}% | {cost} |"
            )
        lines += [
            "",
            "Diminishing returns show up as a shrinking WER drop while the "
            "compute cost keeps growing.",
        ]

    if len(lectures) > 1:
        lines += [
            "",
            "## WER per lecture",
            "",
            "| Model | " + " | ".join(lectures) + " |",
            "|---|" + "---:|" * len(lectures),
        ]
        for r in scored:
            cells = [_pct(r.lecture_wer.get(x), 2) for x in lectures]
            lines.append(f"| {r.model} | " + " | ".join(cells) + " |")

    notes = list(warnings) + comparability_notes(rows)
    if notes:
        lines += ["", "## Caveats", ""] + [f"- {n}" for n in notes]

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "notetaker evalset",
        "notetaker benchmark data/raw/ocw/<id>/audio.flac -m tiny base small "
        "--max-seconds 600",
        "# for each model: transcribe every lecture into outputs/eval/whisper-<m>/,",
        "# then: notetaker wer --hyps ...; notetaker terms --hyps ...",
        "notetaker baseline",
        "```",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Loading report files and writing the result
# ---------------------------------------------------------------------------


def _load_json_reports(folder: Path, cls: Any, kind: str):
    """Load every `*.json` in a folder with `cls.from_dict`, skipping bad files.

    Returns (reports, warnings). A broken or foreign JSON file should not
    stop the whole report, so it becomes a warning instead of a crash.
    """
    reports, warnings = [], []
    if not folder.is_dir():
        return reports, warnings
    for path in sorted(folder.glob("*.json")):
        try:
            reports.append(cls.from_dict(json.loads(path.read_text("utf-8"))))
        except (OSError, ValueError, KeyError, TypeError) as error:
            warnings.append(f"Skipped {kind} file {path.name}: {error}")
    return reports, warnings


def load_wer_reports(folder: str | Path) -> tuple[list[SystemReport], list[str]]:
    return _load_json_reports(Path(folder), SystemReport, "WER")


def load_term_reports(folder: str | Path) -> tuple[list[TermReport], list[str]]:
    return _load_json_reports(Path(folder), TermReport, "term")


def load_benchmark(path: str | Path) -> tuple[list[BenchmarkResult], str | None]:
    """Read `benchmark.json` from `notetaker benchmark`. Missing file -> ([], None)."""
    path = Path(path)
    if not path.is_file():
        return [], None
    data = json.loads(path.read_text("utf-8"))
    results = [BenchmarkResult.from_dict(d) for d in data.get("results", [])]
    return results, data.get("audio")


def write_baseline(
    rows: Sequence[BaselineRow],
    out_path: str | Path,
    *,
    rtf_budget: float = DEFAULT_RTF_BUDGET,
    bench_audio: str | None = None,
    warnings: Sequence[str] = (),
) -> tuple[Path, Path]:
    """Write `baseline.md` and a `baseline.json` next to it with every number."""
    md_path = Path(out_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = md_path.with_suffix(".json")
    best, _ = recommend(rows, rtf_budget)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rtf_budget": rtf_budget,
        "benchmark_audio": bench_audio,
        "recommended": best.model if best else None,
        "pareto_front": sorted(pareto_front(rows)),
        "rows": [r.to_dict() for r in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        to_markdown(
            rows,
            rtf_budget=rtf_budget,
            bench_audio=bench_audio,
            warnings=warnings,
            created_at=payload["created_at"],
        ),
        encoding="utf-8",
    )
    return md_path, json_path
