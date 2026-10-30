"""Word error rate (WER) and character error rate (CER) evaluation.

This is how we measure transcription quality against the official MIT
OpenCourseWare captions downloaded by `notetaker evalset` (Step 2.1).

The pipeline for one lecture:

    reference.json  ──load──> text ──normalize──> words ─┐
                                                         ├─ align (jiwer) ─> counts
    transcript.json ──load──> text ──normalize──> words ─┘

* Both sides go through the SAME `normalize_text()` (Step 2.2), so style
  differences ("3" vs "three", "Okay," vs "ok") are not counted as errors.
* jiwer aligns the two word sequences with the classic edit-distance
  (Levenshtein) dynamic program and counts:
      H = hits (correct words)      S = substitutions (wrong word)
      D = deletions (word missed)   I = insertions (extra word)
* WER = (S + D + I) / N, where N = H + S + D is the number of reference words.
  It can exceed 100% when the model inserts many extra words.
* CER is the same formula on characters instead of words. It is kinder to
  near-misses: "eigenvalues" vs "eigen values" is 2 word errors but only
  1 character error (one inserted space).

Corpus-level numbers are computed by POOLING the counts over all lectures
(sum the errors, divide by the total reference words), not by averaging the
per-lecture WERs. Pooling weights every word equally, so a 5-minute lecture
cannot swing the result as much as a 90-minute one. This is the convention
used by the Whisper paper and most ASR benchmarks.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jiwer

from notetaker.normalize import normalize_text

# ---------------------------------------------------------------------------
# Error counts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorCounts:
    """Alignment counts for one comparison (words or characters).

    `frozen=True` makes instances read-only, so a result cannot be changed by
    accident after it is computed. Adding two ErrorCounts (`a + b`) sums them,
    which is exactly what corpus-level pooling needs.
    """

    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def reference_length(self) -> int:
        """N: how many units the reference has (every one is a hit, S or D)."""
        return self.hits + self.substitutions + self.deletions

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """(S + D + I) / N. Defined as 0 when there is nothing to compare."""
        n = self.reference_length
        return self.errors / n if n else 0.0

    def __add__(self, other: ErrorCounts) -> ErrorCounts:
        # Defining __add__ lets us write `a + b` and also use sum(..., start).
        return ErrorCounts(
            hits=self.hits + other.hits,
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
        )


@dataclass
class PairScore:
    """WER and CER for one (reference, hypothesis) pair, plus common mistakes."""

    name: str
    words: ErrorCounts
    chars: ErrorCounts
    # Most frequent (reference word, hypothesis word, count) substitutions.
    top_substitutions: list[tuple[str, str, int]] = field(default_factory=list)
    # Seconds of audio the comparison covered (None = whole file).
    max_seconds: float | None = None

    @property
    def wer(self) -> float:
        return self.words.rate

    @property
    def cer(self) -> float:
        return self.chars.rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wer": round(self.wer, 6),
            "cer": round(self.cer, 6),
            "words": asdict(self.words),
            "chars": asdict(self.chars),
            "top_substitutions": [list(t) for t in self.top_substitutions],
            "max_seconds": self.max_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PairScore:
        return cls(
            name=data["name"],
            words=ErrorCounts(**data["words"]),
            chars=ErrorCounts(**data["chars"]),
            top_substitutions=[tuple(t) for t in data.get("top_substitutions", [])],
            max_seconds=data.get("max_seconds"),
        )


def _counts(output: Any) -> ErrorCounts:
    """Pull the four counts out of a jiwer WordOutput / CharacterOutput."""
    return ErrorCounts(
        hits=output.hits,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
    )


def substitution_pairs(
    reference: list[str], hypothesis: list[str], alignment: list[Any]
) -> Counter[tuple[str, str]]:
    """Count which reference words were replaced by which hypothesis words.

    jiwer describes an alignment as "chunks": runs of equal / substitute /
    insert / delete with start and end indices into each word list. In a
    substitute chunk the two ranges have the same length, so zip() pairs
    them one to one.
    """
    pairs: Counter[tuple[str, str]] = Counter()
    for chunk in alignment:
        if chunk.type != "substitute":
            continue
        ref_words = reference[chunk.ref_start_idx : chunk.ref_end_idx]
        hyp_words = hypothesis[chunk.hyp_start_idx : chunk.hyp_end_idx]
        pairs.update(zip(ref_words, hyp_words, strict=True))
    return pairs


def score_texts(
    reference: str,
    hypothesis: str,
    *,
    name: str = "",
    normalize: bool = True,
    top_k: int = 10,
) -> PairScore:
    """Compute WER and CER of `hypothesis` against `reference`.

    Args:
        reference: the ground-truth text (e.g. official captions).
        hypothesis: the model's transcript.
        name: label for this pair (usually the lecture id).
        normalize: run both through `normalize_text` first (recommended;
            False is only useful to see how much normalization matters).
        top_k: how many of the most common substitutions to keep.

    Raises:
        ValueError: if the reference is empty after normalization, since WER
            divides by the number of reference words.
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    # split() then join() collapses runs of whitespace, so the character
    # comparison is not thrown off by double spaces or newlines.
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        raise ValueError(f"{name or 'reference'}: reference text is empty")

    word_out = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    char_out = jiwer.process_characters(" ".join(ref_words), " ".join(hyp_words))
    # process_words works on a batch, so alignments is a list with one entry
    # per sentence; we passed a single string, hence [0].
    subs = substitution_pairs(ref_words, hyp_words, word_out.alignments[0])
    return PairScore(
        name=name,
        words=_counts(word_out),
        chars=_counts(char_out),
        top_substitutions=[(r, h, n) for (r, h), n in subs.most_common(top_k)],
    )


# ---------------------------------------------------------------------------
# Loading transcripts, optionally only the first N seconds
# ---------------------------------------------------------------------------


def load_timed_text(path: str | Path, max_seconds: float | None = None) -> str:
    """Read the spoken text of a transcript, optionally cut at `max_seconds`.

    Understands the files this project writes:
    * `transcript.json` from `notetaker transcribe` ({"segments": [...]})
    * `reference.json` from `notetaker evalset` ([{"start", "end", "text"}])
    * anything else is read as plain text (no cutting possible).

    Cutting lets you score just the first 10 minutes of a lecture, which is
    handy on a CPU laptop. A segment is kept if it STARTS before the cut, so
    a sentence spanning the boundary is kept whole on both sides. Captions
    and Whisper segments do not end at the same instants, so expect a few
    extra errors right at the boundary; they are tiny next to 10 minutes of
    speech.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".json":
        if max_seconds is not None:
            raise ValueError(
                f"{path}: --max-seconds needs timestamps, so use a .json file"
            )
        return raw
    data = json.loads(raw)
    items = data.get("segments", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a list of segments or cues")
    texts = []
    for item in items:
        if max_seconds is not None and float(item.get("start", 0.0)) >= max_seconds:
            continue
        texts.append(str(item.get("text", "")))
    return "\n".join(texts)


def score_files(
    reference_path: str | Path,
    hypothesis_path: str | Path,
    *,
    name: str | None = None,
    max_seconds: float | None = None,
    normalize: bool = True,
) -> PairScore:
    """Load two transcript files and score them with `score_texts`."""
    reference = load_timed_text(reference_path, max_seconds)
    hypothesis = load_timed_text(hypothesis_path, max_seconds)
    score = score_texts(
        reference,
        hypothesis,
        name=name or Path(hypothesis_path).parent.name,
        normalize=normalize,
    )
    score.max_seconds = max_seconds
    return score


def find_pairs(
    refs_dir: str | Path, hyps_dir: str | Path
) -> tuple[list[tuple[str, Path, Path]], list[str]]:
    """Match each lecture's reference to the model's transcript by lecture id.

    Expected layout (what `evalset` and `transcribe -o` produce):
        refs_dir/<id>/reference.json
        hyps_dir/<id>/transcript.json

    Returns (pairs, missing): pairs as (id, reference, hypothesis), sorted by
    id so reports are stable, and the ids that have a reference but no
    transcript yet (reported, not fatal).
    """
    pairs: list[tuple[str, Path, Path]] = []
    missing: list[str] = []
    for ref in sorted(Path(refs_dir).glob("*/reference.json")):
        lecture_id = ref.parent.name
        hyp = Path(hyps_dir) / lecture_id / "transcript.json"
        if hyp.is_file():
            pairs.append((lecture_id, ref, hyp))
        else:
            missing.append(lecture_id)
    return pairs, missing


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class SystemReport:
    """All per-lecture scores for one system (e.g. "whisper-small")."""

    system: str
    scores: list[PairScore]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def words(self) -> ErrorCounts:
        """Pooled word counts over every lecture."""
        return sum((s.words for s in self.scores), ErrorCounts())

    @property
    def chars(self) -> ErrorCounts:
        return sum((s.chars for s in self.scores), ErrorCounts())

    @property
    def wer(self) -> float:
        return self.words.rate

    @property
    def cer(self) -> float:
        return self.chars.rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "created_at": self.created_at,
            "pooled": {
                "wer": round(self.wer, 6),
                "cer": round(self.cer, 6),
                "words": asdict(self.words),
                "chars": asdict(self.chars),
            },
            "lectures": [s.to_dict() for s in self.scores],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SystemReport:
        return cls(
            system=data["system"],
            scores=[PairScore.from_dict(d) for d in data["lectures"]],
            created_at=data.get("created_at", ""),
        )


def _pct(rate: float) -> str:
    return f"{100 * rate:.2f}%"


def system_markdown(report: SystemReport) -> str:
    """Per-lecture table for one system, with its most common confusions."""
    lines = [
        f"# WER / CER: {report.system}",
        "",
        f"- Scored at {report.created_at} (Python {sys.version.split()[0]})",
        "- Both texts normalized with `notetaker.normalize.normalize_text`",
        "- Pooled = total errors / total reference words over all lectures",
        "",
        "| Lecture | Ref words | WER | Sub | Del | Ins | CER | Cut at (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in report.scores:
        w = s.words
        cut = "full" if s.max_seconds is None else f"{s.max_seconds:g}"
        lines.append(
            f"| {s.name} | {w.reference_length:,} | {_pct(s.wer)} "
            f"| {w.substitutions:,} | {w.deletions:,} | {w.insertions:,} "
            f"| {_pct(s.cer)} | {cut} |"
        )
    w = report.words
    lines.append(
        f"| **pooled** | {w.reference_length:,} | **{_pct(report.wer)}** "
        f"| {w.substitutions:,} | {w.deletions:,} | {w.insertions:,} "
        f"| **{_pct(report.cer)}** | |"
    )
    for s in report.scores:
        if not s.top_substitutions:
            continue
        lines += ["", f"## Most common substitutions: {s.name}", ""]
        lines += ["| Reference | Model said | Count |", "|---|---|---:|"]
        lines += [f"| {r} | {h} | {n} |" for r, h, n in s.top_substitutions]
    return "\n".join(lines) + "\n"


def summary_markdown(reports: list[SystemReport]) -> str:
    """One row per system, one WER column per lecture: the comparison table."""
    lectures = sorted({s.name for r in reports for s in r.scores})
    header = "| System | Pooled WER | Pooled CER | " + " | ".join(lectures) + " |"
    rule = "|---|---:|---:|" + "---:|" * len(lectures)
    lines = [
        "# WER summary",
        "",
        "Lower is better. Lecture columns show WER. "
        "Regenerated by `notetaker wer` from every `*.json` in this folder.",
        "",
        header,
        rule,
    ]
    for report in sorted(reports, key=lambda r: r.wer):
        by_name = {s.name: s for s in report.scores}
        cells = [_pct(by_name[x].wer) if x in by_name else "-" for x in lectures]
        lines.append(
            f"| {report.system} | {_pct(report.wer)} | {_pct(report.cer)} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines) + "\n"


def safe_filename(system: str) -> str:
    """Turn a system label like "whisper small/int8" into "whisper-small-int8"."""
    cleaned = "".join(c if c.isalnum() or c in "-_." else "-" for c in system)
    return cleaned.strip("-.") or "system"


def write_reports(report: SystemReport, out_dir: str | Path) -> list[Path]:
    """Save `<system>.json` + `<system>.md`, then rebuild `summary.md`.

    The summary is rebuilt from every JSON in the folder, so scoring a new
    model adds a row next to the earlier ones without re-running them.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(report.system)
    json_path = out / f"{stem}.json"
    md_path = out / f"{stem}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", "utf-8")
    md_path.write_text(system_markdown(report), encoding="utf-8")

    all_reports = []
    for path in sorted(out.glob("*.json")):
        try:
            all_reports.append(SystemReport.from_dict(json.loads(path.read_text())))
        except (ValueError, KeyError, TypeError):
            # Some other JSON file lives here; it is not ours, so skip it.
            continue
    summary_path = out / "summary.md"
    summary_path.write_text(summary_markdown(all_reports), encoding="utf-8")
    return [json_path, md_path, summary_path]
