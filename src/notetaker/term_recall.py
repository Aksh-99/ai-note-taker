"""Technical-term recall: did the model get the words that matter?

WER (Step 2.3) treats every word the same, so mishearing "the" costs exactly
as much as mishearing "eigenvalue". For lecture notes that is the wrong
priority: a transcript that drops "the" is still useful, one that writes
"I can value" instead of "eigenvalue" is not. This module measures how well
a model transcribes a curated list of technical terms
(`eval/technical_terms.json`).

The idea, for one lecture:

    reference ──normalize──> words ──count each term──> R_t  (times said)
    transcript ─normalize──> words ──count each term──> H_t  (times written)

    hits_t = min(R_t, H_t)      "clipped" count, as in BLEU
    term recall    = sum_t hits_t / sum_t R_t
    term precision = sum_t hits_t / sum_t H_t

* Recall asks: of all the technical terms the lecturer said, what fraction
  appear in the transcript? This is the headline number.
* Precision asks the reverse: of the technical terms the transcript
  contains, what fraction were really said? Low precision means the model
  is "hallucinating" jargon.
* Clipping with min() stops a model from gaming the metric: writing
  "eigenvalue" 100 times when the lecturer said it 5 times earns 5 hits,
  not 100.

Why counts instead of a word-by-word alignment? Counting does not care
WHERE a term appears, so a term that the model wrote in a slightly
different spot (common at chunk boundaries, Step 1.4) still counts. The
cost is that a miss in one place can be cancelled by a hallucination of the
same term elsewhere; with real lectures that is rare, and the precision
number makes it visible if it happens.

Matching happens on normalized text (Step 2.2), and each term lists the
spellings that count as the same term ("eigenvalue", "eigen values", ...),
so a model is not punished for a harmless split or plural.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from notetaker.normalize import normalize_text
from notetaker.wer import load_timed_text, safe_filename

# Default location of the curated list, relative to the repo root.
DEFAULT_TERMS_PATH = Path("eval/technical_terms.json")

# ---------------------------------------------------------------------------
# The term list
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Term:
    """One technical term and the word sequences that count as saying it.

    `variants` holds each spelling AFTER normalization, as a tuple of words,
    e.g. ("eigen", "values"). Tuples (not lists) are used because they are
    immutable, which `frozen=True` dataclasses need to stay hashable.
    """

    name: str
    domain: str
    variants: tuple[tuple[str, ...], ...]


def make_term(name: str, variants: list[str] | None = None, domain: str = "") -> Term:
    """Build a Term, normalizing every spelling the same way transcripts are.

    Variants are sorted longest first so that, when two spellings start at
    the same word ("gaussian elimination" and "gaussian"), the longer one
    wins and the shorter one is not double counted inside it.
    """
    spellings = variants or [name]
    normalized: set[tuple[str, ...]] = set()
    for spelling in spellings:
        words = tuple(normalize_text(spelling).split())
        if words:
            normalized.add(words)
    if not normalized:
        raise ValueError(f"term {name!r}: every spelling is empty after normalizing")
    ordered = tuple(sorted(normalized, key=lambda v: (-len(v), v)))
    return Term(name=name, domain=domain, variants=ordered)


def load_terms(path: str | Path = DEFAULT_TERMS_PATH) -> list[Term]:
    """Read the curated term list from JSON.

    Expected shape: {"terms": [{"term": ..., "domain": ..., "variants": [...]}]}
    (a bare list of those objects also works). Term names must be unique,
    since reports are keyed by name.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("terms") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: expected a non-empty list under 'terms'")
    terms: list[Term] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not item.get("term"):
            raise ValueError(f"{path}: every entry needs a 'term' field: {item!r}")
        name = str(item["term"])
        if name in seen:
            raise ValueError(f"{path}: duplicate term {name!r}")
        seen.add(name)
        terms.append(
            make_term(name, item.get("variants"), domain=str(item.get("domain", "")))
        )
    return terms


# ---------------------------------------------------------------------------
# Counting terms in a word list
# ---------------------------------------------------------------------------


def _index_words(words: list[str]) -> dict[str, list[int]]:
    """Map each word to the positions where it occurs (an "inverted index").

    Building this once per text means each term only checks the positions
    where its first word appears, instead of scanning the whole lecture.
    """
    index: dict[str, list[int]] = defaultdict(list)
    for position, word in enumerate(words):
        index[word].append(position)
    return index


def count_term(
    words: list[str], term: Term, index: dict[str, list[int]] | None = None
) -> int:
    """How many times `term` (any variant) occurs in `words`, without overlaps.

    Scans candidate start positions left to right. At each one the longest
    matching variant is taken and the scan jumps past it, so "eigen values
    eigen values" counts 2, never 3 or 4.
    """
    if index is None:
        index = _index_words(words)
    starts = sorted({p for v in term.variants for p in index.get(v[0], [])})
    count = 0
    next_free = 0  # first position not already used by an earlier match
    for start in starts:
        if start < next_free:
            continue
        for variant in term.variants:  # longest first
            if tuple(words[start : start + len(variant)]) == variant:
                count += 1
                next_free = start + len(variant)
                break
    return count


def count_terms(
    text: str, terms: list[Term], *, normalize: bool = True
) -> dict[str, int]:
    """Count every term in `text`. Returns {term name: count} for all terms."""
    if normalize:
        text = normalize_text(text)
    words = text.split()
    index = _index_words(words)
    return {term.name: count_term(words, term, index) for term in terms}


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


@dataclass
class TermCounts:
    """Reference and hypothesis counts per term, for one lecture or pooled.

    Keeping the raw counts (not just a percentage) lets the summary pool
    lectures correctly and lets later steps (2.5, 6.3) compare models term
    by term.
    """

    reference: dict[str, int]
    hypothesis: dict[str, int]

    def hits(self, name: str) -> int:
        return min(self.reference.get(name, 0), self.hypothesis.get(name, 0))

    @property
    def total_reference(self) -> int:
        return sum(self.reference.values())

    @property
    def total_hypothesis(self) -> int:
        return sum(self.hypothesis.values())

    @property
    def total_hits(self) -> int:
        names = set(self.reference) | set(self.hypothesis)
        return sum(self.hits(n) for n in names)

    @property
    def recall(self) -> float:
        """Share of spoken term occurrences that made it into the transcript."""
        n = self.total_reference
        return self.total_hits / n if n else 0.0

    @property
    def precision(self) -> float:
        """Share of transcribed term occurrences that were really spoken."""
        n = self.total_hypothesis
        return self.total_hits / n if n else 0.0

    def misses(self) -> list[tuple[str, int, int]]:
        """Terms said more often than transcribed, as (name, said, written).

        Sorted by how many occurrences were lost, then by name, so the worst
        offenders come first and the order is stable between runs.
        """
        out = [
            (name, said, self.hypothesis.get(name, 0))
            for name, said in self.reference.items()
            if said > self.hypothesis.get(name, 0)
        ]
        return sorted(out, key=lambda t: (-(t[1] - t[2]), t[0]))

    def __add__(self, other: TermCounts) -> TermCounts:
        # Element-wise sum of the two dictionaries, used for pooling.
        ref = dict(self.reference)
        hyp = dict(self.hypothesis)
        for name, n in other.reference.items():
            ref[name] = ref.get(name, 0) + n
        for name, n in other.hypothesis.items():
            hyp[name] = hyp.get(name, 0) + n
        return TermCounts(reference=ref, hypothesis=hyp)


def _drop_zeros(counts: dict[str, int]) -> dict[str, int]:
    return {name: n for name, n in sorted(counts.items()) if n}


@dataclass
class TermScore:
    """Term recall for one (reference, transcript) pair."""

    name: str
    counts: TermCounts
    max_seconds: float | None = None

    @property
    def recall(self) -> float:
        return self.counts.recall

    @property
    def precision(self) -> float:
        return self.counts.precision

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "recall": round(self.recall, 6),
            "precision": round(self.precision, 6),
            "max_seconds": self.max_seconds,
            # Zero counts are left out to keep the JSON readable.
            "reference_counts": _drop_zeros(self.counts.reference),
            "hypothesis_counts": _drop_zeros(self.counts.hypothesis),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TermScore:
        return cls(
            name=data["name"],
            counts=TermCounts(
                reference=dict(data["reference_counts"]),
                hypothesis=dict(data["hypothesis_counts"]),
            ),
            max_seconds=data.get("max_seconds"),
        )


def score_term_texts(
    reference: str,
    hypothesis: str,
    terms: list[Term],
    *,
    name: str = "",
    normalize: bool = True,
) -> TermScore:
    """Count the terms on both sides and wrap them in a TermScore."""
    return TermScore(
        name=name,
        counts=TermCounts(
            reference=count_terms(reference, terms, normalize=normalize),
            hypothesis=count_terms(hypothesis, terms, normalize=normalize),
        ),
    )


def score_term_files(
    reference_path: str | Path,
    hypothesis_path: str | Path,
    terms: list[Term],
    *,
    name: str | None = None,
    max_seconds: float | None = None,
    normalize: bool = True,
) -> TermScore:
    """Load two transcripts (optionally the first N seconds) and score them."""
    score = score_term_texts(
        load_timed_text(reference_path, max_seconds),
        load_timed_text(hypothesis_path, max_seconds),
        terms,
        name=name or Path(hypothesis_path).parent.name,
        normalize=normalize,
    )
    score.max_seconds = max_seconds
    return score


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class TermReport:
    """All per-lecture term scores for one system (e.g. "whisper-small")."""

    system: str
    scores: list[TermScore]
    # Term name -> domain, so the report can break recall down by subject.
    domains: dict[str, str] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def counts(self) -> TermCounts:
        """Counts pooled over every lecture (sum, then divide)."""
        return sum((s.counts for s in self.scores), TermCounts({}, {}))

    @property
    def recall(self) -> float:
        return self.counts.recall

    @property
    def precision(self) -> float:
        return self.counts.precision

    def recall_by_domain(self) -> dict[str, tuple[int, int]]:
        """{domain: (hits, said)} over the pooled counts, for domains that occur."""
        pooled = self.counts
        out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for name, said in pooled.reference.items():
            if not said:
                continue
            domain = self.domains.get(name) or "other"
            out[domain][0] += pooled.hits(name)
            out[domain][1] += said
        return {d: (h, n) for d, (h, n) in sorted(out.items())}

    def to_dict(self) -> dict[str, Any]:
        pooled = self.counts
        return {
            "system": self.system,
            "created_at": self.created_at,
            "pooled": {
                "recall": round(self.recall, 6),
                "precision": round(self.precision, 6),
                "said": pooled.total_reference,
                "written": pooled.total_hypothesis,
                "hits": pooled.total_hits,
            },
            "domains": self.domains,
            "lectures": [s.to_dict() for s in self.scores],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TermReport:
        return cls(
            system=data["system"],
            scores=[TermScore.from_dict(d) for d in data["lectures"]],
            domains=dict(data.get("domains", {})),
            created_at=data.get("created_at", ""),
        )


def _pct(rate: float) -> str:
    return f"{100 * rate:.1f}%"


def system_markdown(report: TermReport, top_misses: int = 15) -> str:
    """Per-lecture table, recall by domain, and the most-missed terms."""
    lines = [
        f"# Technical-term recall: {report.system}",
        "",
        f"- Scored at {report.created_at}",
        "- Recall = term occurrences transcribed / term occurrences spoken",
        "- Precision = transcribed occurrences that were really spoken",
        "- Counts are clipped: a term earns at most as many hits as it was said",
        "",
        "| Lecture | Terms said | Hits | Recall | Precision | Cut at (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in report.scores:
        c = s.counts
        cut = "full" if s.max_seconds is None else f"{s.max_seconds:g}"
        lines.append(
            f"| {s.name} | {c.total_reference:,} | {c.total_hits:,} "
            f"| {_pct(s.recall)} | {_pct(s.precision)} | {cut} |"
        )
    pooled = report.counts
    lines.append(
        f"| **pooled** | {pooled.total_reference:,} | {pooled.total_hits:,} "
        f"| **{_pct(report.recall)}** | **{_pct(report.precision)}** | |"
    )

    by_domain = report.recall_by_domain()
    if by_domain:
        lines += ["", "## Recall by domain", ""]
        lines += ["| Domain | Terms said | Hits | Recall |", "|---|---:|---:|---:|"]
        for domain, (hits, said) in by_domain.items():
            lines.append(f"| {domain} | {said:,} | {hits:,} | {_pct(hits / said)} |")

    misses = pooled.misses()[:top_misses]
    if misses:
        lines += ["", "## Most-missed terms (pooled)", ""]
        lines += ["| Term | Said | Transcribed | Lost |", "|---|---:|---:|---:|"]
        lines += [f"| {t} | {said} | {got} | {said - got} |" for t, said, got in misses]
    return "\n".join(lines) + "\n"


def summary_markdown(reports: list[TermReport]) -> str:
    """One row per system: pooled recall / precision plus recall per lecture."""
    lectures = sorted({s.name for r in reports for s in r.scores})
    lines = [
        "# Technical-term recall summary",
        "",
        "Higher is better. Lecture columns show recall. "
        "Regenerated by `notetaker terms` from every `*.json` in this folder.",
        "",
        "| System | Recall | Precision | " + " | ".join(lectures) + " |",
        "|---|---:|---:|" + "---:|" * len(lectures),
    ]
    for report in sorted(reports, key=lambda r: -r.recall):
        by_name = {s.name: s for s in report.scores}
        cells = [_pct(by_name[x].recall) if x in by_name else "-" for x in lectures]
        lines.append(
            f"| {report.system} | {_pct(report.recall)} | {_pct(report.precision)} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_reports(report: TermReport, out_dir: str | Path) -> list[Path]:
    """Save `<system>.json` + `<system>.md`, then rebuild `summary.md`.

    Same pattern as the WER reports: the summary is rebuilt from every JSON
    in the folder, so a newly scored model joins the earlier rows.
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
            all_reports.append(TermReport.from_dict(json.loads(path.read_text())))
        except (ValueError, KeyError, TypeError):
            continue  # not one of our report files
    summary_path = out / "summary.md"
    summary_path.write_text(summary_markdown(all_reports), encoding="utf-8")
    return [json_path, md_path, summary_path]
