"""Measure how well search finds the right passage: recall@k and MRR.

This is Step 5.4. Steps 5.1 to 5.3 built search and question answering, but
"it seems to work on the questions I tried" is not a number. A retrieval eval
is a fixed list of questions where a human has marked *where* the answer is.
We run every question through the index and check how high the right
passage comes up. The same set then scores any change: a different embedding
model, a different passage size, or hybrid search in Step 5.5.

Labels are time spans, not passage ids
    Each question lists one or more `relevant` spans: a lecture id plus a
    start and end time in seconds, e.g. "la-05 from 731.2 s to 774.9 s".
    A retrieved passage counts as relevant if it is from that lecture and its
    time range overlaps the span. Passage ids would be simpler but they break
    the moment passages are cut differently (`--max-words 60` renumbers every
    passage), and comparing passage sizes is exactly the kind of experiment
    this eval is for. A time span stays correct whatever the chunking.

Any retriever can be scored (Step 5.5)
    `evaluate` only needs `.search`, `.passages`, `.lectures()` and `.name`,
    so the same questions score vector search, BM25 and hybrid search
    (hybrid.py). `notetaker eval-retrieval --retriever vector bm25 hybrid`
    puts them side by side in one table.

    Two intervals [a, b] and [c, d] overlap when a < d and c < b. Touching at
    a single point (b == c) does not count, so a passage that ends exactly
    where the answer starts is not given credit.

The metrics (all between 0 and 1, higher is better)
    Search returns a ranked list. Let D be how deep we look (the largest k
    asked for, 10 by default).

    recall@k
        For one question: the fraction of its relevant spans that are covered
        by at least one of the top k passages. Most questions have a single
        span, so this is just 1 if the answer is in the top k and 0 if not
        (some papers call that "hit rate@k"). A question whose answer is
        spread over four lectures scores 0.5 if the top k reach two of them.
        The reported recall@k is the mean over questions.

        Why it matters here: `notetaker ask` shows the LLM the top 5
        passages. If the answer is not among them, no model can answer
        correctly, so recall@5 is a ceiling on answer quality.

    MRR (mean reciprocal rank)
        For one question: 1 / (rank of the first relevant passage), or 0 if
        none is in the top D. Rank 1 gives 1.0, rank 2 gives 0.5, rank 4
        gives 0.25. The mean over questions rewards putting the answer at the
        very top, which recall@5 cannot see (rank 1 and rank 5 both count as
        a hit). Strictly this is MRR@D, since anything below D scores 0.

    chance
        The report also shows what a random ordering of the passages would
        score on average. With N passages of which r are relevant, the chance
        that k random picks miss all of them is C(N-r, k) / C(N, k), so
        expected recall@k = 1 - C(N-r, k) / C(N, k) (C is "n choose k").
        This anchors the real numbers: recall@5 = 0.6 means little on its
        own, but a lot if chance is 0.15.

Small eval sets are noisy
    With 26 questions, one question flipping moves recall by about 0.04. Treat
    differences smaller than that as ties. The report prints the question
    count next to every score so this is visible.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from notetaker.embeddings import Embedder, make_embedder
from notetaker.hybrid import Retriever
from notetaker.ingest import assign_lecture_ids, find_transcripts, load_titles
from notetaker.transcribe import Transcript
from notetaker.vectorstore import (
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_WORDS,
    Passage,
    SearchHit,
    VectorStore,
)

DEFAULT_QUESTIONS = Path("eval/retrieval/questions.json")
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


class RetrievalEvalError(RuntimeError):
    """The question file is malformed or does not match the index."""


# ---------------------------------------------------------------------------
# The labeled question set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelevantSpan:
    """Where an answer is: a stretch of time in one lecture."""

    lecture_id: str
    start: float  # seconds
    end: float

    def overlaps(self, passage: Passage) -> bool:
        """True if the passage is from this lecture and shares some time with it."""
        return (
            passage.lecture_id == self.lecture_id
            and passage.start < self.end
            and self.start < passage.end
        )


@dataclass(frozen=True)
class LabeledQuestion:
    """One question plus where its answer is."""

    id: str
    question: str
    relevant: tuple[RelevantSpan, ...]
    tags: tuple[str, ...] = ()

    def is_relevant(self, passage: Passage) -> bool:
        return any(span.overlaps(passage) for span in self.relevant)


@dataclass
class QuestionSet:
    """A loaded question file.

    `corpus` and `titles` are optional paths stored in the file (relative to
    the file's own folder), so `notetaker eval-retrieval` can find the
    transcripts the labels were written against without extra flags.
    """

    questions: list[LabeledQuestion]
    path: Path | None = None
    description: str = ""
    corpus: Path | None = None
    titles: Path | None = None

    def lecture_ids(self) -> set[str]:
        return {s.lecture_id for q in self.questions for s in q.relevant}

    def tags(self) -> list[str]:
        """Every tag used, in first-seen order."""
        return list(dict.fromkeys(t for q in self.questions for t in q.tags))


def _parse_question(raw: Any, where: str) -> LabeledQuestion:
    """Validate one question object, with an error message that says where."""
    if not isinstance(raw, dict):
        raise RetrievalEvalError(f"{where}: each question must be a JSON object")
    qid = raw.get("id")
    text = raw.get("question")
    if not isinstance(qid, str) or not qid.strip():
        raise RetrievalEvalError(f"{where}: missing a string 'id'")
    if not isinstance(text, str) or not text.strip():
        raise RetrievalEvalError(f"question {qid}: missing the 'question' text")
    spans_raw = raw.get("relevant")
    if not isinstance(spans_raw, list) or not spans_raw:
        raise RetrievalEvalError(
            f"question {qid}: 'relevant' must be a non-empty list of spans"
        )
    spans = []
    for span in spans_raw:
        try:
            lecture_id = span["lecture_id"]
            start, end = float(span["start"]), float(span["end"])
        except (TypeError, KeyError, ValueError) as error:
            raise RetrievalEvalError(
                f"question {qid}: each span needs lecture_id, start and end "
                f"(got {span!r})"
            ) from error
        if not isinstance(lecture_id, str) or not lecture_id:
            raise RetrievalEvalError(f"question {qid}: lecture_id must be a string")
        if not 0 <= start < end:
            raise RetrievalEvalError(
                f"question {qid}: span must have 0 <= start < end, got "
                f"{start}..{end} in {lecture_id}"
            )
        spans.append(RelevantSpan(lecture_id, start, end))
    tags = raw.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise RetrievalEvalError(f"question {qid}: 'tags' must be a list of strings")
    return LabeledQuestion(qid, text.strip(), tuple(spans), tuple(tags))


def load_questions(path: str | Path) -> QuestionSet:
    """Read and validate a question file (format: eval/retrieval/questions.json)."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RetrievalEvalError(f"question file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise RetrievalEvalError(f"{path} is not valid JSON ({error})") from error
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise RetrievalEvalError(f"{path} must be an object with a 'questions' list")

    questions = [
        _parse_question(raw, f"{path} question #{i + 1}")
        for i, raw in enumerate(data["questions"])
    ]
    if not questions:
        raise RetrievalEvalError(f"{path} has no questions")
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            raise RetrievalEvalError(f"duplicate question id {q.id!r} in {path}")
        seen.add(q.id)

    def local(key: str) -> Path | None:
        # Paths inside the file are relative to the file, so the eval set can
        # be moved or copied as one folder.
        value = data.get(key)
        return (path.parent / value) if isinstance(value, str) and value else None

    return QuestionSet(
        questions=questions,
        path=path,
        description=str(data.get("description", "")),
        corpus=local("corpus"),
        titles=local("titles"),
    )


# ---------------------------------------------------------------------------
# Scoring one question
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    """What search returned for one question, and how it was judged."""

    question: LabeledQuestion
    hits: list[SearchHit]
    relevant_flags: list[bool]  # relevant_flags[i] is about hits[i]
    # span_ranks[j] = rank of the first hit covering relevant span j, or None.
    span_ranks: list[int | None]
    relevant_in_index: int  # passages in the whole index that are relevant
    seconds: float  # time for this one search

    @property
    def first_rank(self) -> int | None:
        """Rank (1-based) of the first relevant passage, or None if not found."""
        for hit, relevant in zip(self.hits, self.relevant_flags, strict=True):
            if relevant:
                return hit.rank
        return None

    def recall_at(self, k: int) -> float:
        """Fraction of this question's spans reached by the top k passages."""
        found = sum(1 for r in self.span_ranks if r is not None and r <= k)
        return found / len(self.span_ranks)

    @property
    def reciprocal_rank(self) -> float:
        rank = self.first_rank
        return 1.0 / rank if rank else 0.0


def judge(
    question: LabeledQuestion, hits: list[SearchHit], passages: Sequence[Passage]
) -> QuestionResult:
    """Mark each hit relevant or not and find where each span was first reached.

    `passages` is the whole index, used only to count how many passages are
    relevant at all (needed for the chance baseline and for spotting labels
    that point at nothing).
    """
    flags = [question.is_relevant(h.passage) for h in hits]
    span_ranks: list[int | None] = []
    for span in question.relevant:
        rank = next((h.rank for h in hits if span.overlaps(h.passage)), None)
        span_ranks.append(rank)
    relevant_in_index = sum(1 for p in passages if question.is_relevant(p))
    return QuestionResult(question, hits, flags, span_ranks, relevant_in_index, 0.0)


# ---------------------------------------------------------------------------
# The chance baseline (expected scores of a random ranking)
# ---------------------------------------------------------------------------


def chance_recall_at(n: int, r: int, k: int) -> float:
    """Expected recall@k if the n passages were shuffled and r are relevant.

    P(all k random picks miss) = C(n - r, k) / C(n, k): the number of ways to
    pick k from the irrelevant passages over the number of ways to pick k
    from all. math.comb returns 0 when k > n - r, which correctly gives a
    probability of 1 of hitting something.
    """
    if r <= 0 or n <= 0:
        return 0.0
    k = min(k, n)
    return 1.0 - math.comb(n - r, k) / math.comb(n, k)


def chance_mrr(n: int, r: int, depth: int) -> float:
    """Expected 1/rank of the first relevant passage in a random ranking.

    The first relevant passage is at rank i when the i - 1 passages before it
    are all irrelevant and passage i is relevant. Counting arrangements,
    P(first at i) = C(n - i, r - 1) / C(n, r) (choose where the other r - 1
    relevant passages go, all after position i). Ranks below `depth` score 0,
    matching how the real MRR is cut off.
    """
    if r <= 0 or n <= 0:
        return 0.0
    total = math.comb(n, r)
    return sum(
        math.comb(n - i, r - 1) / total / i for i in range(1, min(depth, n - r + 1) + 1)
    )


def _span_chance(span: RelevantSpan, passages: Sequence[Passage], k: int) -> float:
    r = sum(1 for p in passages if span.overlaps(p))
    return chance_recall_at(len(passages), r, k)


# ---------------------------------------------------------------------------
# Scoring a whole run
# ---------------------------------------------------------------------------


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


@dataclass
class RetrievalReport:
    """Scores for one index (one embedder + passage settings) on a question set."""

    embedder: str
    ks: tuple[int, ...]
    n_passages: int
    n_lectures: int
    results: list[QuestionResult]
    chance_recall: dict[int, float] = field(default_factory=dict)
    chance_mrr: float = 0.0
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return max(self.ks)

    @property
    def label(self) -> str:
        """Short name for tables: the embedder, plus passage size if known."""
        if "max_words" in self.settings:
            return f"{self.embedder} @ {self.settings['max_words']}w"
        return self.embedder

    def _subset(self, tag: str | None) -> list[QuestionResult]:
        if tag is None:
            return self.results
        return [r for r in self.results if tag in r.question.tags]

    def count(self, tag: str | None = None) -> int:
        return len(self._subset(tag))

    def recall(self, k: int, tag: str | None = None) -> float:
        return _mean(r.recall_at(k) for r in self._subset(tag))

    def mrr(self, tag: str | None = None) -> float:
        return _mean(r.reciprocal_rank for r in self._subset(tag))

    def mean_latency_ms(self) -> float:
        return 1000 * _mean(r.seconds for r in self.results)

    def misses(self, k: int) -> list[QuestionResult]:
        """Questions where some relevant span was not in the top k."""
        return [r for r in self.results if r.recall_at(k) < 1.0]

    def to_dict(self) -> dict[str, Any]:
        """Plain data for JSON output: aggregates plus every question's ranks."""
        return {
            "embedder": self.embedder,
            "settings": self.settings,
            "passages": self.n_passages,
            "lectures": self.n_lectures,
            "questions": len(self.results),
            "ks": list(self.ks),
            "recall": {str(k): round(self.recall(k), 4) for k in self.ks},
            "mrr": round(self.mrr(), 4),
            "chance": {
                "recall": {str(k): round(v, 4) for k, v in self.chance_recall.items()},
                "mrr": round(self.chance_mrr, 4),
            },
            "mean_latency_ms": round(self.mean_latency_ms(), 3),
            "per_question": [
                {
                    "id": r.question.id,
                    "question": r.question.question,
                    "tags": list(r.question.tags),
                    "first_rank": r.first_rank,
                    "span_ranks": r.span_ranks,
                    "top": [
                        {
                            "rank": h.rank,
                            "passage_id": h.passage.passage_id,
                            "citation": h.passage.citation(),
                            "score": round(h.score, 4),
                            "relevant": flag,
                        }
                        for h, flag in zip(r.hits, r.relevant_flags, strict=True)
                    ],
                }
                for r in self.results
            ],
        }


def check_labels(store: Retriever, questions: QuestionSet) -> None:
    """Fail loudly if labels point at lectures or times the index does not have.

    Without this, a typo in a lecture id (or evaluating against the wrong
    index) silently scores 0 for that question and drags the average down,
    which looks like a bad model rather than a bad label.
    """
    indexed = set(store.lectures())
    unknown = sorted(questions.lecture_ids() - indexed)
    if unknown:
        raise RetrievalEvalError(
            f"questions refer to lectures that are not in the index: {unknown}. "
            f"Indexed lectures: {sorted(indexed)}"
        )
    empty = [
        q.id
        for q in questions.questions
        for span in q.relevant
        if not any(span.overlaps(p) for p in store.passages)
    ]
    if empty:
        raise RetrievalEvalError(
            f"these questions have a span that no passage overlaps (check the "
            f"times against the transcript): {sorted(set(empty))}"
        )


def evaluate(
    store: Retriever,
    questions: QuestionSet,
    ks: Sequence[int] = DEFAULT_KS,
    settings: dict[str, Any] | None = None,
) -> RetrievalReport:
    """Run every question through `store.search` and score the rankings.

    `store` can be any retriever (hybrid.Retriever): a VectorStore, a
    BM25Index or a HybridRetriever. The report is labeled with its `name`.
    """
    ks = tuple(sorted(set(ks)))
    if not ks or ks[0] <= 0:
        raise ValueError(f"every k must be a positive integer, got {list(ks)}")
    check_labels(store, questions)
    depth = max(ks)
    passages = store.passages

    results = []
    for q in questions.questions:
        # perf_counter is a high-resolution clock meant for timing code.
        t0 = time.perf_counter()
        hits = store.search(q.question, k=depth)
        elapsed = time.perf_counter() - t0
        result = judge(q, hits, passages)
        result.seconds = elapsed
        results.append(result)

    # Chance baseline: per span for recall (to match how recall is computed),
    # per question for MRR (first relevant passage of any span).
    n = len(passages)
    chance_recall = {
        k: _mean(
            _mean(_span_chance(s, passages, k) for s in q.relevant)
            for q in questions.questions
        )
        for k in ks
    }
    chance_mrr_value = _mean(chance_mrr(n, r.relevant_in_index, depth) for r in results)

    return RetrievalReport(
        embedder=store.name,
        ks=ks,
        n_passages=n,
        n_lectures=len(store.lectures()),
        results=results,
        chance_recall=chance_recall,
        chance_mrr=chance_mrr_value,
        settings=dict(settings or {}),
    )


# ---------------------------------------------------------------------------
# Building an index in memory from a folder of transcripts
# ---------------------------------------------------------------------------


def build_store(
    folder: str | Path,
    *,
    embedder: Embedder | None = None,
    embedder_name: str | None = None,
    titles: dict[str, str] | None = None,
    pattern: str = "transcript.json",
    max_words: int = DEFAULT_MAX_WORDS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> VectorStore:
    """Index every transcript under `folder` in memory (nothing saved).

    Lecture ids follow the same rule as `notetaker ingest`, so labels written
    for an ingested index also work here and vice versa.
    """
    paths = find_transcripts(folder, pattern)
    if not paths:
        raise RetrievalEvalError(f"no files matching {pattern!r} under {folder}")
    ids = assign_lecture_ids(paths, folder)
    store = VectorStore(
        embedder if embedder is not None else make_embedder(embedder_name)
    )
    titles = titles or {}
    for path in paths:
        lecture_id = ids[path]
        store.add_transcript(
            Transcript.load_json(path),
            lecture_id=lecture_id,
            title=titles.get(lecture_id, ""),
            source=str(path),
            max_words=max_words,
            max_seconds=max_seconds,
        )
    return store


def load_corpus_titles(questions: QuestionSet, override: Path | None) -> dict[str, str]:
    """Titles from --titles if given, else from the question file, else none."""
    path = override if override is not None else questions.titles
    if path is not None and not path.exists() and override is None:
        return {}
    return load_titles(path)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _settings_text(report: RetrievalReport) -> str:
    s = report.settings
    if "max_words" in s:
        return f"{s['max_words']} words / {s['max_seconds']:g} s"
    return s.get("index", "")


def to_markdown(reports: list[RetrievalReport], questions: QuestionSet) -> str:
    """A report comparing one or more runs on the same question set."""
    if not reports:
        raise ValueError("no reports to write")
    ks = reports[0].ks
    first = reports[0]
    lines = [
        "# Retrieval eval: recall@k and MRR",
        "",
        f"Question set: `{questions.path}` ({len(questions.questions)} questions, "
        f"{len(questions.lecture_ids())} lectures). Index: {first.n_passages} "
        f"passages. Generated by `notetaker eval-retrieval`.",
        "",
        "recall@k: share of answer spans found in the top k passages. "
        "MRR: mean of 1 / rank of the first relevant passage "
        f"(0 if not in the top {first.depth}). "
        "Chance: expected score of a random ordering. "
        f"With {len(questions.questions)} questions, one question is worth about "
        f"{1 / len(questions.questions):.2f}, so smaller gaps are ties.",
        "",
        "## Overall",
        "",
        "| Retriever | Passages | "
        + " | ".join(f"recall@{k}" for k in ks)
        + " | MRR | ms / query |",
        "|---|---|" + "---|" * len(ks) + "---|---|",
    ]
    for rep in reports:
        lines.append(
            f"| `{rep.embedder}` | {rep.n_passages} ({_settings_text(rep)}) | "
            + " | ".join(_fmt(rep.recall(k)) for k in ks)
            + f" | {_fmt(rep.mrr())} | {rep.mean_latency_ms():.2f} |"
        )
    # Chance depends only on how many passages there are and which are
    # relevant, so runs with the same passage count share one chance row.
    chance_runs = {rep.n_passages: rep for rep in reports}
    for n, rep in chance_runs.items():
        lines.append(
            f"| chance | {n} | "
            + " | ".join(_fmt(rep.chance_recall[k]) for k in ks)
            + f" | {_fmt(rep.chance_mrr)} | |"
        )

    tags = questions.tags()
    if tags:
        k_mid = 5 if 5 in ks else ks[len(ks) // 2]
        lines += [
            "",
            f"## By question type (recall@{k_mid} / MRR)",
            "",
            "| Run | " + " | ".join(f"{t} (n={first.count(t)})" for t in tags) + " |",
            "|---|" + "---|" * len(tags),
        ]
        for rep in reports:
            cells = [f"{_fmt(rep.recall(k_mid, t))} / {_fmt(rep.mrr(t))}" for t in tags]
            lines.append(f"| `{rep.label}` | " + " | ".join(cells) + " |")

    lines += ["", "## Per question (rank of first relevant passage, - = not found)", ""]
    lines.append(
        "| Id | Question | Tags | " + " | ".join(r.label for r in reports) + " |"
    )
    lines.append("|---|---|---|" + "---|" * len(reports))
    for i, q in enumerate(questions.questions):
        ranks = []
        for rep in reports:
            res = rep.results[i]
            rank = str(res.first_rank) if res.first_rank else "-"
            if len(res.span_ranks) > 1:
                found = sum(1 for r in res.span_ranks if r is not None)
                rank += f" ({found}/{len(res.span_ranks)} spans)"
            ranks.append(rank)
        lines.append(
            f"| {q.id} | {q.question} | {', '.join(q.tags)} | "
            + " | ".join(ranks)
            + " |"
        )
    return "\n".join(lines) + "\n"


def to_json(reports: list[RetrievalReport], questions: QuestionSet) -> str:
    data = {
        "questions_file": str(questions.path) if questions.path else None,
        "questions": len(questions.questions),
        "runs": [rep.to_dict() for rep in reports],
    }
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
