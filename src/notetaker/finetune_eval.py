"""Evaluate a LoRA fine-tuned Whisper against its base model (Step 6.3).

The question this answers:
    Did fine-tuning on lecture audio (Steps 6.1 and 6.2) actually help, and
    is the improvement bigger than what random luck in the choice of test
    clips could explain?

How the comparison is made fair:
    - Same clips: every held-out clip is transcribed twice by ONE model
      object, once with the LoRA adapter switched off (exactly the pretrained
      Whisper, see `finetune.load_finetuned`) and once with it on. Same
      audio, same greedy decoding, same normalization, same term list.
    - Same metrics as the Phase 2 baseline: pooled WER / CER (wer.py) and
      technical-term recall / precision (term_recall.py).
    - Held-out lectures: the default split is "test", lectures the adapter
      never saw in training and that were not used to pick the best step
      (that was the "val" split, so val numbers are slightly optimistic).

The statistics (paired bootstrap):
    The test set is a sample of clips, so the WER difference we measure
    (fine-tuned minus base) is itself a random quantity. To get a 95%
    confidence interval we resample: draw n clips WITH replacement from the
    n test clips, recompute both pooled WERs on that resample, take the
    difference, repeat B times (default 2000), and read off the 2.5th and
    97.5th percentiles. "Paired" means each resample keeps a clip's base and
    fine-tuned errors together, which cancels out clip difficulty (a hard,
    mumbled clip is hard for both models). If the whole interval is below
    zero, the improvement is unlikely to be luck.

    Caveat, stated in the report: clips from the same lecture are correlated
    (same speaker, same topic), so treating clips as independent makes the
    interval a little too narrow. With only a handful of test lectures a
    lecture-level bootstrap would be too coarse to be useful, so we resample
    clips and say so.

Two ways to run:
    1. Full run (needs the [finetune] extra and, in practice, a GPU):
       `run_finetune_eval` loads base + adapter, transcribes, saves the
       transcripts to JSONL, scores them and writes the report.
    2. Re-score only: `evaluate_transcripts` on a saved JSONL. No torch, no
       model, runs in seconds. This is what CI tests, and it lets you change
       the report or the term list without re-running the GPU part.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from notetaker.finetune import (
    SUMMARY_FILENAME,
    FinetuneConfig,
    FinetuneError,
    load_clips,
    load_finetuned,
    transcribe_clips,
)
from notetaker.normalize import normalize_text
from notetaker.term_recall import Term, TermCounts, score_term_texts
from notetaker.wer import ErrorCounts, score_texts

# The two systems being compared, in report order.
BASE, TUNED = "base", "fine-tuned"
SYSTEMS = (BASE, TUNED)

# Default file names. The transcripts can be re-scored later without a GPU.
TRANSCRIPTS_FILENAME = "eval_transcripts.jsonl"
REPORT_JSON = "finetune.json"
REPORT_MD = "finetune.md"


# ---------------------------------------------------------------------------
# Saved transcripts: one JSON object per clip
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipTranscripts:
    """One clip's reference text and the two models' transcripts."""

    clip: str  # file name relative to the split folder, e.g. "lec/lec_00003.wav"
    lecture_id: str
    reference: str
    base: str
    tuned: str

    def text(self, system: str) -> str:
        return self.base if system == BASE else self.tuned


def save_transcripts(rows: Sequence[ClipTranscripts], path: str | Path) -> Path:
    """Write rows as JSONL (one JSON object per line, easy to append or grep)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def load_transcripts(path: str | Path) -> list[ClipTranscripts]:
    """Read a JSONL written by `save_transcripts`, with clear errors."""
    path = Path(path)
    if not path.is_file():
        raise FinetuneError(f"No transcripts file {path}")
    rows: list[ClipTranscripts] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(ClipTranscripts(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as error:
            raise FinetuneError(f"{path} line {number}: bad row ({error})") from error
    if not rows:
        raise FinetuneError(f"{path} has no transcripts")
    return rows


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class ClipScore:
    """Word errors and term counts for one clip, for both systems."""

    clip: str
    lecture_id: str
    reference_words: int
    words: dict[str, ErrorCounts]  # system -> word error counts
    chars: dict[str, ErrorCounts]
    terms: dict[str, TermCounts]

    def errors(self, system: str) -> int:
        return self.words[system].errors


@dataclass
class Bootstrap:
    """Paired bootstrap result for "fine-tuned WER minus base WER"."""

    delta: float  # measured difference on the real test set (negative = better)
    low: float  # 2.5th percentile of the resampled differences
    high: float  # 97.5th percentile
    share_improved: float  # fraction of resamples where fine-tuned was better
    resamples: int
    seed: int


def paired_bootstrap(
    base_errors: Sequence[int],
    tuned_errors: Sequence[int],
    reference_words: Sequence[int],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> Bootstrap:
    """95% interval for the pooled WER difference, resampling clips.

    Pooled WER on a resample is sum(errors) / sum(reference words) over the
    drawn clips, so we never average per-clip WERs (that would let a 3-word
    clip count as much as a 70-word one).

    Vectorised with numpy: `idx` is a (resamples x n) matrix of random clip
    indices, so `errors[idx]` picks the errors of every drawn clip for every
    resample at once and `.sum(axis=1)` totals each resample. That replaces
    a slow Python double loop with one array operation.
    """
    base = np.asarray(base_errors, dtype=float)
    tuned = np.asarray(tuned_errors, dtype=float)
    ref = np.asarray(reference_words, dtype=float)
    if not (len(base) == len(tuned) == len(ref)) or len(ref) == 0:
        raise ValueError("need the same, non-zero number of clips for each input")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    delta = (tuned.sum() - base.sum()) / ref.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ref), size=(resamples, len(ref)))
    words = ref[idx].sum(axis=1)
    # A resample could in theory draw only clips with zero reference words;
    # score_texts rejects empty references, so every clip has >= 1 word.
    deltas = (tuned[idx].sum(axis=1) - base[idx].sum(axis=1)) / words
    low, high = np.percentile(deltas, [2.5, 97.5])
    return Bootstrap(
        delta=float(delta),
        low=float(low),
        high=float(high),
        share_improved=float((deltas < 0).mean()),
        resamples=resamples,
        seed=seed,
    )


def score_clips(
    rows: Sequence[ClipTranscripts], terms: list[Term]
) -> tuple[list[ClipScore], list[str]]:
    """Score every clip for both systems.

    Returns (scores, skipped). A clip whose reference is empty after
    normalization (e.g. a caption that was only "[MUSIC]") cannot have a WER,
    so it is skipped and listed rather than crashing the whole evaluation.
    """
    scores: list[ClipScore] = []
    skipped: list[str] = []
    for row in rows:
        if not normalize_text(row.reference).split():
            skipped.append(row.clip)
            continue
        words: dict[str, ErrorCounts] = {}
        chars: dict[str, ErrorCounts] = {}
        term_counts: dict[str, TermCounts] = {}
        for system in SYSTEMS:
            pair = score_texts(row.reference, row.text(system), name=row.clip)
            words[system] = pair.words
            chars[system] = pair.chars
            term_counts[system] = score_term_texts(
                row.reference, row.text(system), terms, name=row.clip
            ).counts
        scores.append(
            ClipScore(
                clip=row.clip,
                lecture_id=row.lecture_id,
                reference_words=words[BASE].reference_length,
                words=words,
                chars=chars,
                terms=term_counts,
            )
        )
    return scores, skipped


def _pool_words(scores: Sequence[ClipScore], system: str) -> ErrorCounts:
    return sum((s.words[system] for s in scores), ErrorCounts())


def _pool_chars(scores: Sequence[ClipScore], system: str) -> ErrorCounts:
    return sum((s.chars[system] for s in scores), ErrorCounts())


def _pool_terms(scores: Sequence[ClipScore], system: str) -> TermCounts:
    return sum((s.terms[system] for s in scores), TermCounts({}, {}))


@dataclass
class SystemSummary:
    """Pooled metrics for one system over a set of clips."""

    wer: float
    cer: float
    term_recall: float
    term_precision: float
    words: ErrorCounts
    term_hits: int
    term_reference: int

    @classmethod
    def of(cls, scores: Sequence[ClipScore], system: str) -> SystemSummary:
        words = _pool_words(scores, system)
        terms = _pool_terms(scores, system)
        return cls(
            wer=words.rate,
            cer=_pool_chars(scores, system).rate,
            term_recall=terms.recall,
            term_precision=terms.precision,
            words=words,
            term_hits=terms.total_hits,
            term_reference=terms.total_reference,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "wer": round(self.wer, 6),
            "cer": round(self.cer, 6),
            "term_recall": round(self.term_recall, 6),
            "term_precision": round(self.term_precision, 6),
            "words": asdict(self.words),
            "term_hits": self.term_hits,
            "term_reference": self.term_reference,
        }


@dataclass(frozen=True)
class TermChange:
    """How often one term was transcribed correctly, before vs after."""

    term: str
    said: int
    base_hits: int
    tuned_hits: int

    @property
    def change(self) -> int:
        return self.tuned_hits - self.base_hits


def term_changes(scores: Sequence[ClipScore]) -> list[TermChange]:
    """Per-term hits for both systems, biggest changes first.

    Hits are counted per clip and then summed (min(said, written) inside each
    clip), so a term written in the wrong clip does not count as recalled.
    """
    said: dict[str, int] = {}
    hits = {system: {} for system in SYSTEMS}
    for score in scores:
        for system in SYSTEMS:
            counts = score.terms[system]
            for name, n in counts.reference.items():
                if system == BASE:
                    said[name] = said.get(name, 0) + n
                hits[system][name] = hits[system].get(name, 0) + counts.hits(name)
    changes = [
        TermChange(name, n, hits[BASE].get(name, 0), hits[TUNED].get(name, 0))
        for name, n in said.items()
        if n
    ]
    return sorted(changes, key=lambda c: (-abs(c.change), -c.said, c.term))


@dataclass
class FinetuneEvalReport:
    """Everything the Step 6.3 report shows, plus enough to rebuild it."""

    split: str
    overall: dict[str, SystemSummary]
    lectures: dict[str, dict[str, SystemSummary]]
    clips_per_lecture: dict[str, int]
    bootstrap: Bootstrap
    terms: list[TermChange]
    examples: list[dict[str, Any]]  # most improved / most worsened clips
    skipped: list[str]
    training: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def clips(self) -> int:
        return sum(self.clips_per_lecture.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "created_at": self.created_at,
            "clips": self.clips,
            "skipped": self.skipped,
            "overall": {s: v.to_dict() for s, v in self.overall.items()},
            "wer_delta": asdict(self.bootstrap),
            "lectures": {
                lec: {
                    "clips": self.clips_per_lecture[lec],
                    **{s: v.to_dict() for s, v in by_system.items()},
                }
                for lec, by_system in self.lectures.items()
            },
            "terms": [asdict(t) | {"change": t.change} for t in self.terms],
            "examples": self.examples,
            "training": self.training,
        }


def _examples(
    rows: Sequence[ClipTranscripts], scores: Sequence[ClipScore], k: int
) -> list[dict[str, Any]]:
    """The k clips that improved most and the k that got worse most.

    Numbers say how much; examples say what kind of mistakes were fixed (or
    introduced), which is what you want to show in a write-up.
    """
    by_clip = {row.clip: row for row in rows}
    ranked = sorted(scores, key=lambda s: (s.errors(TUNED) - s.errors(BASE), s.clip))
    improved = [s for s in ranked if s.errors(TUNED) < s.errors(BASE)][:k]
    worsened = [s for s in reversed(ranked) if s.errors(TUNED) > s.errors(BASE)][:k]
    out = []
    for kind, group in (("improved", improved), ("worsened", worsened)):
        for score in group:
            row = by_clip[score.clip]
            out.append(
                {
                    "kind": kind,
                    "clip": score.clip,
                    "reference": row.reference,
                    "base": row.base,
                    "tuned": row.tuned,
                    "base_errors": score.errors(BASE),
                    "tuned_errors": score.errors(TUNED),
                    "reference_words": score.reference_words,
                }
            )
    return out


def training_summary(model_dir: str | Path) -> dict[str, Any]:
    """The few training facts worth quoting next to the results.

    Reads `<model_dir>/finetune.json` from `run_finetune`. Missing or broken
    files give an empty dict: the evaluation itself does not depend on it.
    """
    path = Path(model_dir) / SUMMARY_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        config = FinetuneConfig.from_json(data["config"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, FinetuneError):
        return {}
    params = data.get("parameters", {})
    plan = data.get("plan", {})
    return {
        "base_model": config.base_model,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "max_steps": config.max_steps,
        "effective_batch": config.effective_batch,
        "best_step": data.get("best_step"),
        "best_val_loss": data.get("best_val_loss"),
        "trainable_parameters": params.get("trainable"),
        "total_parameters": params.get("total"),
        "train_hours": plan.get("train", {}).get("hours"),
    }


def evaluate_transcripts(
    rows: Sequence[ClipTranscripts],
    terms: list[Term],
    *,
    split: str = "test",
    resamples: int = 2000,
    seed: int = 0,
    examples: int = 3,
    training: dict[str, Any] | None = None,
) -> FinetuneEvalReport:
    """Score saved transcripts and build the report (no model, no torch)."""
    scores, skipped = score_clips(rows, terms)
    if not scores:
        raise FinetuneError("No clip has a non-empty reference; nothing to score")
    by_lecture: dict[str, list[ClipScore]] = {}
    for score in scores:
        by_lecture.setdefault(score.lecture_id, []).append(score)
    lectures = {
        lec: {system: SystemSummary.of(group, system) for system in SYSTEMS}
        for lec, group in sorted(by_lecture.items())
    }
    return FinetuneEvalReport(
        split=split,
        overall={system: SystemSummary.of(scores, system) for system in SYSTEMS},
        lectures=lectures,
        clips_per_lecture={lec: len(g) for lec, g in sorted(by_lecture.items())},
        bootstrap=paired_bootstrap(
            [s.errors(BASE) for s in scores],
            [s.errors(TUNED) for s in scores],
            [s.reference_words for s in scores],
            resamples=resamples,
            seed=seed,
        ),
        terms=term_changes(scores),
        examples=_examples(rows, scores, examples),
        skipped=skipped,
        training=training or {},
    )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _pct(rate: float) -> str:
    return f"{100 * rate:.1f}%"


def _pp(delta: float) -> str:
    """A difference of two rates in percentage points, with its sign."""
    return f"{100 * delta:+.1f} pp"


def verdict(boot: Bootstrap) -> str:
    """One plain sentence on whether the WER change is likely real."""
    if boot.high < 0:
        return "Fine-tuning lowered WER, and the whole 95% interval is below zero."
    if boot.low > 0:
        return "Fine-tuning raised WER, and the whole 95% interval is above zero."
    return (
        "The 95% interval includes zero, so this test set cannot tell the two "
        "models apart on WER."
    )


def to_markdown(report: FinetuneEvalReport, top_terms: int = 10) -> str:
    """Render the report as GitHub-flavoured markdown."""
    base, tuned = report.overall[BASE], report.overall[TUNED]
    boot = report.bootstrap
    lines = [
        "# Fine-tuned vs base Whisper",
        "",
        f"Split: `{report.split}` ({report.clips} clips from "
        f"{len(report.lectures)} lectures, {base.words.reference_length:,} "
        f"reference words). Generated {report.created_at}.",
        "",
        "Both systems are the same model object with the LoRA adapter off "
        "(base) and on (fine-tuned), greedy decoding, same normalization and "
        "term list as the Phase 2 baseline.",
        "",
        "## Headline",
        "",
        "| System | WER | CER | Term recall | Term precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in report.overall.items():
        lines.append(
            f"| {name} | {_pct(summary.wer)} | {_pct(summary.cer)} | "
            f"{_pct(summary.term_recall)} | {_pct(summary.term_precision)} |"
        )
    lines += [
        "",
        f"WER change: **{_pp(boot.delta)}** (95% paired bootstrap interval "
        f"{_pp(boot.low)} to {_pp(boot.high)}; fine-tuned better in "
        f"{_pct(boot.share_improved)} of {boot.resamples:,} resamples). "
        f"{verdict(boot)}",
        "",
        f"Term recall change: {_pp(tuned.term_recall - base.term_recall)} "
        f"({base.term_hits} to {tuned.term_hits} of {base.term_reference} "
        "spoken term occurrences).",
        "",
        "## Per lecture",
        "",
        "| Lecture | Clips | Base WER | Tuned WER | Change | Base terms | "
        "Tuned terms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lec, by_system in report.lectures.items():
        b, t = by_system[BASE], by_system[TUNED]
        lines.append(
            f"| {lec} | {report.clips_per_lecture[lec]} | {_pct(b.wer)} | "
            f"{_pct(t.wer)} | {_pp(t.wer - b.wer)} | {_pct(b.term_recall)} | "
            f"{_pct(t.term_recall)} |"
        )
    changed = [c for c in report.terms if c.change][:top_terms]
    lines += ["", "## Terms that changed most", ""]
    if changed:
        lines += [
            "| Term | Said | Base hits | Tuned hits | Change |",
            "|---|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {c.term} | {c.said} | {c.base_hits} | {c.tuned_hits} | {c.change:+d} |"
            for c in changed
        ]
    else:
        lines.append("No term was recalled a different number of times.")
    if report.examples:
        lines += ["", "## Examples", ""]
        for ex in report.examples:
            lines += [
                f"**{ex['kind'].capitalize()}** `{ex['clip']}` "
                f"({ex['base_errors']} to {ex['tuned_errors']} errors in "
                f"{ex['reference_words']} words)",
                "",
                f"- Reference: {ex['reference']}",
                f"- Base: {ex['base']}",
                f"- Fine-tuned: {ex['tuned']}",
                "",
            ]
        lines.pop()  # drop the trailing blank line of the last example
    if report.training:
        tr = report.training
        lines += ["", "## Training run", ""]
        labels = [
            ("base_model", "Base model"),
            ("lora_r", "LoRA rank r"),
            ("lora_alpha", "LoRA alpha"),
            ("max_steps", "Max steps"),
            ("effective_batch", "Effective batch"),
            ("best_step", "Best step (lowest val loss)"),
            ("best_val_loss", "Best val loss"),
            ("trainable_parameters", "Trainable parameters"),
            ("train_hours", "Training audio (hours)"),
        ]
        for key, label in labels:
            value = tr.get(key)
            if value is not None:
                text = f"{value:,}" if isinstance(value, int) else str(value)
                lines.append(f"- {label}: {text}")
    lines += [
        "",
        "## How to read this",
        "",
        "- WER and CER are pooled: total errors over total reference words, "
        "not an average of per-clip rates.",
        "- The interval resamples clips with replacement. Clips from the same "
        "lecture are correlated, so the true interval is somewhat wider.",
        "- Decoding here is Hugging Face `generate` on 30 s clips, not "
        "faster-whisper on whole lectures, so absolute numbers are not "
        "directly comparable with `reports/baseline.md`; the base vs "
        "fine-tuned difference is the fair comparison.",
    ]
    if report.skipped:
        lines.append(
            f"- {len(report.skipped)} clip(s) skipped because their reference "
            "was empty after normalization."
        )
    return "\n".join(lines) + "\n"


def write_report(report: FinetuneEvalReport, out_dir: str | Path) -> list[Path]:
    """Write `finetune.json` and `finetune.md` into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / REPORT_JSON
    md_path = out_dir / REPORT_MD
    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", "utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return [json_path, md_path]


# ---------------------------------------------------------------------------
# Full run: load the model, transcribe both ways, save, score
# ---------------------------------------------------------------------------


def transcribe_both(
    model: Any,
    feature_extractor: Any,
    tokenizer: Any,
    clips: Sequence[Any],
    split_dir: str | Path,
    *,
    batch_size: int = 8,
    max_new_tokens: int = 225,
    log: Callable[[str], None] = print,
) -> list[ClipTranscripts]:
    """Transcribe every clip with the adapter on, then off.

    `model.disable_adapter()` is a context manager: inside the `with` block
    peft skips the LoRA term (alpha / r) B A x, so the same weights act as the
    original pretrained Whisper; on leaving the block the adapter is back.
    """
    log(f"Transcribing {len(clips)} clips with the adapter on...")
    tuned = transcribe_clips(
        model,
        feature_extractor,
        tokenizer,
        clips,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    log("...and with the adapter off (base model)")
    with model.disable_adapter():
        base = transcribe_clips(
            model,
            feature_extractor,
            tokenizer,
            clips,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
    split_dir = Path(split_dir)
    return [
        ClipTranscripts(
            clip=clip.path.relative_to(split_dir).as_posix(),
            lecture_id=clip.lecture_id,
            reference=clip.text,
            base=b,
            tuned=t,
        )
        for clip, b, t in zip(clips, base, tuned, strict=True)
    ]


def run_finetune_eval(
    model_dir: str | Path,
    data_dir: str | Path,
    terms: list[Term],
    *,
    split: str = "test",
    transcripts_path: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 225,
    device: str | None = None,
    resamples: int = 2000,
    seed: int = 0,
    load_model: Callable[[FinetuneConfig], tuple[Any, Any, Any]] | None = None,
    log: Callable[[str], None] = print,
) -> FinetuneEvalReport:
    """Load base + adapter, transcribe `split`, save transcripts, score them.

    `limit` keeps only the first N clips (a quick smoke test). `load_model`
    is the same test hook as in `run_finetune`.
    """
    clips = load_clips(data_dir, split)
    if limit is not None:
        clips = clips[:limit]
    if not clips:
        raise FinetuneError(
            f"The {split} split has no clips. Try --split val if your dataset "
            "has no test lectures."
        )
    model, feature_extractor, tokenizer, _ = load_finetuned(
        model_dir, device=device, load_model=load_model
    )
    rows = transcribe_both(
        model,
        feature_extractor,
        tokenizer,
        clips,
        Path(data_dir) / split,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        log=log,
    )
    path = save_transcripts(
        rows, transcripts_path or Path(model_dir) / TRANSCRIPTS_FILENAME
    )
    log(f"Saved transcripts to {path}")
    return evaluate_transcripts(
        rows,
        terms,
        split=split,
        resamples=resamples,
        seed=seed,
        training=training_summary(model_dir),
    )
