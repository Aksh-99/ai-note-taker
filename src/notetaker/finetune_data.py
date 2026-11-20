"""Build a Whisper fine-tuning dataset from the evaluation lectures.

Why this exists:
    Phase 6 fine-tunes Whisper on technical lecture audio (Step 6.2) and then
    checks whether it got better (Step 6.3). Both need data in the shape
    Whisper trains on: short audio clips (at most 30 seconds, the length of
    Whisper's input window) each paired with the exact words spoken in it.
    The MIT OCW lectures from Step 2.1 already have that information: the
    official captions are timed cues, so we cut the long lecture audio at cue
    boundaries and keep each clip's caption text as its label.

The most important rule: no leakage.
    "Leakage" means information from the test data sneaking into training.
    The result looks great on paper and means nothing. For speech there are
    three ways it happens, and this module guards against each:

    1. The same lecture on both sides. Neighbouring clips share the topic,
       the vocabulary and even half-finished sentences, so a random split of
       clips would let the model memorise the test lecture. We therefore split
       whole LECTURES, never individual clips.
    2. The same speaker on both sides. A model can learn one professor's
       voice and accent. `group_by="course"` keeps every lecture of a course
       (usually one lecturer) on the same side. With one lecture per course,
       as in the current manifest, both options give the same split.
    3. The same sentence on both sides. Lecturers repeat themselves, and two
       lectures of a course can share a recap. `check_leakage` reports clips
       whose normalized text appears in more than one split.

    Rule 1 and 2 violations are errors (the build stops). Rule 3 is a warning,
    because short stock phrases ("any questions?") legitimately recur; only
    clips with at least `min_duplicate_words` words are reported.

    It also matters for Step 6.3: the base-vs-fine-tuned comparison must use
    ONLY the test lectures. The baseline in reports/baseline.md used all
    lectures, so Step 6.3 re-scores the base model on the test split too.

Output layout (Hugging Face "audiofolder" format):
    <out_dir>/
        dataset.json              settings, which lecture went where, counts
        train/metadata.jsonl      one JSON line per clip
        train/<lecture id>/<lecture id>_00000.wav ...
        val/...
        test/...

    Each metadata line looks like
        {"file_name": "mit-18.06-lec01/mit-18.06-lec01_00012.wav",
         "text": "So the row picture is ...", "lecture_id": "mit-18.06-lec01",
         "start": 301.2, "end": 327.9, "duration": 26.7}
    which is exactly what `datasets.load_dataset("audiofolder", data_dir=...)`
    expects, so the Colab notebook in Step 6.2 can load it in one line.

Design note (dependency injection, as in evalset.py):
    `build_dataset` receives the "load audio" and "write a clip" functions as
    arguments. Real runs use ffmpeg and the WAV writer from audio.py; tests
    pass a fake loader that returns a synthetic waveform, so the whole build
    is tested offline in milliseconds without downloading any lecture.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from notetaker.audio import load_audio, save_wav
from notetaker.normalize import normalize_text, strip_caption_markup

# Whisper always looks at 30 seconds of audio at a time. Longer clips would
# be cut off during training, so no clip may be longer than this.
WHISPER_WINDOW_S = 30.0

# The three splits, in the order they are reported.
SPLITS = ("train", "val", "test")
Split = Literal["train", "val", "test"]
GroupBy = Literal["lecture", "course"]

# Type aliases for the injected functions.
LoadAudio = Callable[[Path, int], np.ndarray]
WriteClip = Callable[[np.ndarray, Path, int], Path]


class FinetuneDataError(RuntimeError):
    """The dataset could not be built (message says why)."""


# ---------------------------------------------------------------------------
# Turning caption cues into training segments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cue:
    """One caption cue: text shown on screen from `start` to `end` seconds."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    """One training example: an audio time range and the words spoken in it."""

    lecture_id: str
    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SegmentStats:
    """Why cues did not become training data, for the summary report."""

    cues: int = 0
    empty_after_cleaning: int = 0
    too_long: int = 0
    too_short: int = 0
    past_audio_end: int = 0

    def add(self, other: SegmentStats) -> None:
        for name in asdict(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))


def clean_caption_text(text: str) -> str:
    """Remove caption-only clutter but keep case and punctuation.

    Sound tags like [LAUGHTER] and speaker labels like "PROFESSOR:" are not
    spoken, so they must not be in the label (the model would learn to invent
    them). Case and punctuation ARE kept: Whisper is trained to write cased,
    punctuated text, and fine-tuning on lowercased text would teach it to
    stop doing that. WER is computed after normalization anyway (Step 2.2).
    """
    return " ".join(strip_caption_markup(text).split())


def load_cues(path: str | Path) -> list[Cue]:
    """Read a reference.json written by `notetaker evalset`."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cues = [Cue(float(c["start"]), float(c["end"]), str(c["text"])) for c in raw]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise FinetuneDataError(
            f"Could not read caption cues from {path}: {error}"
        ) from error
    # Caption files are almost always in order, but sorting costs nothing and
    # the merging below relies on it.
    return sorted(cues, key=lambda c: (c.start, c.end))


def merge_cues(
    cues: list[Cue],
    lecture_id: str,
    *,
    max_seconds: float = WHISPER_WINDOW_S,
    min_seconds: float = 1.0,
    max_gap: float = 1.5,
    pad: float = 0.2,
    audio_seconds: float | None = None,
) -> tuple[list[Segment], SegmentStats]:
    """Join consecutive caption cues into segments of at most `max_seconds`.

    Caption cues are short (2 to 6 seconds), shorter than ideal for training,
    so we glue neighbours together greedily: keep adding the next cue while
    the segment stays within `max_seconds` and the silence before the cue is
    at most `max_gap`. A long gap usually means something unlabeled happened
    (writing on the board, a student question without captions), so it is a
    natural place to cut.

    Cue timings are a little imprecise (caption writers line them up by eye),
    and cutting exactly at the first word's start can clip its first sound.
    So each segment is widened by up to `pad` seconds on each side, but never
    past the halfway point of the silence to its neighbour. That way the
    padding can only add silence, never a word that belongs to the next clip.

    Dropped, and counted in the returned stats:
      * cues with no spoken text after cleaning (e.g. just "[MUSIC]"),
      * a single cue longer than `max_seconds` (we cannot cut it without
        knowing where inside it each word falls),
      * segments shorter than `min_seconds` (too little audio to learn from),
      * segments that end after the audio does (captions longer than the
        recording, which happens when a video was trimmed).
    """
    if max_seconds <= 0 or max_seconds > WHISPER_WINDOW_S:
        raise ValueError(f"max_seconds must be in (0, {WHISPER_WINDOW_S}]")
    if min_seconds < 0 or min_seconds >= max_seconds:
        raise ValueError("min_seconds must be >= 0 and below max_seconds")
    if max_gap < 0 or pad < 0:
        raise ValueError("max_gap and pad must be >= 0")

    stats = SegmentStats(cues=len(cues))

    # Pass 1: clean the text and drop cues that cannot be used on their own.
    usable: list[Cue] = []
    for cue in cues:
        text = clean_caption_text(cue.text)
        if not text:
            stats.empty_after_cleaning += 1
        elif cue.end - cue.start > max_seconds:
            stats.too_long += 1
        elif cue.end > cue.start:
            usable.append(Cue(cue.start, cue.end, text))

    # Pass 2: greedy merge. `groups` is a list of runs of consecutive cues.
    groups: list[list[Cue]] = []
    for cue in usable:
        if groups:
            current = groups[-1]
            gap = cue.start - current[-1].end
            # Overlapping cues (gap < 0) happen in captions; treat as no gap.
            fits = cue.end - current[0].start <= max_seconds
            if fits and gap <= max_gap:
                current.append(cue)
                continue
        groups.append([cue])

    # Pass 3: pad into the surrounding silence, then filter.
    segments: list[Segment] = []
    for i, group in enumerate(groups):
        start, end = group[0].start, group[-1].end
        prev_end = groups[i - 1][-1].end if i > 0 else 0.0
        next_start = groups[i + 1][0].start if i + 1 < len(groups) else math.inf
        # Half the silence on each side (0 if the neighbours overlap).
        room_before = max(0.0, (start - prev_end) / 2) if i > 0 else start
        room_after = max(0.0, (next_start - end) / 2)
        new_start = max(0.0, start - min(pad, room_before))
        new_end = end + min(pad, room_after)
        if audio_seconds is not None:
            if end > audio_seconds:
                stats.past_audio_end += 1
                continue
            new_end = min(new_end, audio_seconds)
        # Padding must not push a segment past the 30 s window either.
        excess = (new_end - new_start) - max_seconds
        if excess > 0:
            # Take the excess back from the padding, end side first.
            trim_end = min(excess, new_end - end)
            new_end -= trim_end
            new_start += excess - trim_end
        if end - start < min_seconds:
            stats.too_short += 1
            continue
        segments.append(
            Segment(
                lecture_id=lecture_id,
                index=len(segments),
                start=round(new_start, 3),
                end=round(new_end, 3),
                text=" ".join(c.text for c in group),
            )
        )
    return segments, stats


# ---------------------------------------------------------------------------
# Splitting lectures into train / val / test without leakage
# ---------------------------------------------------------------------------


def _rank_key(seed: int, group: str) -> str:
    """A pseudo-random but reproducible sort key for a group.

    Hashing "seed:group" gives every group a fixed random-looking position.
    Unlike shuffling a list, the order of two groups never depends on which
    OTHER groups exist, so adding a new lecture to the manifest does not
    reshuffle where the old ones went (only the counts per split can shift).
    """
    return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()


def _count(fraction: float, total: int) -> int:
    """How many groups a fraction means: round half up, at least 1 if > 0.

    (Python's round() uses "banker's rounding", round(2.5) == 2, which is
    surprising here, so we round half up with floor(x + 0.5) instead.)
    """
    if fraction <= 0:
        return 0
    return max(1, math.floor(fraction * total + 0.5))


def assign_splits(
    groups: dict[str, str],
    *,
    val: list[str] | None = None,
    test: list[str] | None = None,
    val_fraction: float = 0.25,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, Split]:
    """Decide which split every lecture goes to.

    Args:
        groups: lecture id -> group key. Lectures with the same key always go
            to the same split (key = lecture id for group_by="lecture", the
            course name for group_by="course").
        val, test: lecture ids to force into that split (their whole group
            comes along). When either is given, the fractions are ignored for
            that split and the rest goes to train.
        val_fraction, test_fraction: share of GROUPS (not clips) for each
            split when it is not given explicitly.
        seed: changes the random-looking assignment.

    Returns:
        lecture id -> "train" | "val" | "test".
    """
    if not groups:
        raise FinetuneDataError("No lectures to split")
    for name, fraction in (("val", val_fraction), ("test", test_fraction)):
        if not 0 <= fraction < 1:
            raise FinetuneDataError(f"{name} fraction must be in [0, 1)")

    named = {"val": list(val or []), "test": list(test or [])}
    unknown = [i for ids in named.values() for i in ids if i not in groups]
    if unknown:
        raise FinetuneDataError(
            f"Unknown lecture id(s) {unknown}; known: {sorted(groups)}"
        )

    group_split: dict[str, Split] = {}
    for split in ("test", "val"):
        for lecture_id in named[split]:
            key = groups[lecture_id]
            if group_split.get(key, split) != split:
                raise FinetuneDataError(
                    f"{lecture_id} is in group {key!r}, which is already in "
                    f"{group_split[key]}; a group cannot be split"
                )
            group_split[key] = split

    # Fill the splits that were not named from the remaining groups, in hash
    # order: test first, then val, everything else is train.
    remaining = sorted(
        {g for g in groups.values() if g not in group_split},
        key=lambda g: _rank_key(seed, g),
    )
    total = len(set(groups.values()))
    for split, fraction in (("test", test_fraction), ("val", val_fraction)):
        if named[split]:
            continue
        for key in remaining[: _count(fraction, total)]:
            group_split[key] = split
        remaining = [g for g in remaining if g not in group_split]
    for key in remaining:
        group_split[key] = "train"

    if "train" not in group_split.values():
        raise FinetuneDataError(
            f"No lectures left for training ({total} group(s) in total). Add "
            "lectures to the manifest or lower --val-fraction / --test-fraction."
        )
    return {lecture_id: group_split[key] for lecture_id, key in groups.items()}


@dataclass
class LeakageReport:
    """Result of `check_leakage`: errors stop the build, warnings do not."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_leakage(
    segments_by_split: dict[str, list[Segment]],
    groups: dict[str, str],
    *,
    min_duplicate_words: int = 8,
) -> LeakageReport:
    """Verify that nothing is shared between splits.

    This re-checks what `assign_splits` promised, on the final data. That is
    deliberate: if a later change to the code ever broke the split logic,
    this independent check would still catch it before any training run.
    """
    report = LeakageReport()

    def splits_of(key_of: Callable[[Segment], str]) -> dict[str, set[str]]:
        seen: dict[str, set[str]] = {}
        for split, segments in segments_by_split.items():
            for segment in segments:
                seen.setdefault(key_of(segment), set()).add(split)
        return seen

    for lecture_id, where in sorted(splits_of(lambda s: s.lecture_id).items()):
        if len(where) > 1:
            report.errors.append(f"lecture {lecture_id} is in {sorted(where)}")
    for key, where in sorted(splits_of(lambda s: groups[s.lecture_id]).items()):
        if len(where) > 1:
            report.errors.append(f"group {key!r} is in {sorted(where)}")

    # Same sentence in two splits. Compare normalized text so that "Okay, so"
    # and "okay so" count as the same.
    text_splits: dict[str, set[str]] = {}
    for split, segments in segments_by_split.items():
        for segment in segments:
            words = normalize_text(segment.text)
            if len(words.split()) >= min_duplicate_words:
                text_splits.setdefault(words, set()).add(split)
    for words, where in sorted(text_splits.items()):
        if len(where) > 1:
            preview = words if len(words) <= 60 else words[:57] + "..."
            report.warnings.append(f"same text in {sorted(where)}: {preview!r}")
    return report


# ---------------------------------------------------------------------------
# Building the dataset on disk
# ---------------------------------------------------------------------------


@dataclass
class LectureSource:
    """One downloaded lecture folder from `notetaker evalset`."""

    lecture_id: str
    course: str
    folder: Path

    @property
    def audio(self) -> Path:
        return self.folder / "audio.flac"

    @property
    def reference(self) -> Path:
        return self.folder / "reference.json"


def find_lectures(eval_dir: str | Path) -> list[LectureSource]:
    """Every lecture folder under `eval_dir` that has a reference.json.

    The course name comes from meta.json (written by the downloader) and is
    only needed for group_by="course"; without it, the lecture id is used.
    """
    eval_dir = Path(eval_dir)
    if not eval_dir.is_dir():
        raise FinetuneDataError(
            f"No eval folder at {eval_dir}. Download the lectures first with "
            "`notetaker evalset`."
        )
    lectures = []
    for folder in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        if not (folder / "reference.json").is_file():
            continue
        course = folder.name
        meta = folder / "meta.json"
        if meta.is_file():
            # contextlib.suppress(ValueError) is a tidy "try: ... except
            # ValueError: pass". A broken meta.json only loses the course
            # grouping, so it is not worth stopping the build for.
            with contextlib.suppress(ValueError):
                course = json.loads(meta.read_text(encoding="utf-8")).get(
                    "course", course
                )
        lectures.append(LectureSource(folder.name, course, folder))
    if not lectures:
        raise FinetuneDataError(f"No lecture folders with reference.json in {eval_dir}")
    return lectures


@dataclass
class SplitSummary:
    """Counts for one split, as stored in dataset.json."""

    lectures: list[str] = field(default_factory=list)
    clips: int = 0
    seconds: float = 0.0
    words: int = 0

    @property
    def hours(self) -> float:
        return self.seconds / 3600


@dataclass
class DatasetSummary:
    """Everything `build_dataset` did, written to dataset.json."""

    out_dir: str
    settings: dict
    splits: dict[str, SplitSummary]
    dropped: SegmentStats
    leakage: LeakageReport
    skipped_lectures: list[str] = field(default_factory=list)
    dry_run: bool = False

    def to_json(self) -> dict:
        return {
            "out_dir": self.out_dir,
            "dry_run": self.dry_run,
            "settings": self.settings,
            "splits": {
                name: {**asdict(s), "hours": round(s.hours, 4)}
                for name, s in self.splits.items()
            },
            "dropped": asdict(self.dropped),
            "leakage": asdict(self.leakage),
            "skipped_lectures": self.skipped_lectures,
        }


def _clear_output(out_dir: Path, force: bool) -> None:
    """Refuse to mix a new build with old clips, unless `force` is given.

    With force, only the folders and file this module creates are removed,
    never anything else a user may have put in `out_dir`.
    """
    ours = [out_dir / s for s in SPLITS] + [out_dir / "dataset.json"]
    existing = [p for p in ours if p.exists()]
    if existing and not force:
        raise FinetuneDataError(
            f"{out_dir} already has a dataset; use --force to rebuild it"
        )
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def build_dataset(
    eval_dir: str | Path,
    out_dir: str | Path,
    *,
    group_by: GroupBy = "lecture",
    val: list[str] | None = None,
    test: list[str] | None = None,
    val_fraction: float = 0.25,
    test_fraction: float = 0.25,
    seed: int = 0,
    max_seconds: float = WHISPER_WINDOW_S,
    min_seconds: float = 1.0,
    max_gap: float = 1.5,
    pad: float = 0.2,
    sample_rate: int = 16_000,
    dry_run: bool = False,
    force: bool = False,
    load: LoadAudio = load_audio,
    write: WriteClip = save_wav,
) -> DatasetSummary:
    """Cut the eval lectures into labeled clips and split them by lecture.

    `dry_run=True` only reads the captions: it reports how many clips and
    hours each split would get, and runs the leakage check, without loading
    any audio or writing any file. Handy to try split settings in a second.
    Lectures without audio.flac are skipped in a real run (and listed).
    """
    eval_dir, out_dir = Path(eval_dir), Path(out_dir)
    lectures = find_lectures(eval_dir)
    skipped = (
        []
        if dry_run
        else [lec.lecture_id for lec in lectures if not lec.audio.is_file()]
    )
    lectures = [lec for lec in lectures if lec.lecture_id not in skipped]
    if not lectures:
        raise FinetuneDataError(
            f"None of the lectures in {eval_dir} has audio.flac. Run "
            "`notetaker evalset` without --no-audio."
        )

    groups = {
        lec.lecture_id: (lec.course if group_by == "course" else lec.lecture_id)
        for lec in lectures
    }
    split_of = assign_splits(
        groups,
        val=val,
        test=test,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    segment_options = {
        "max_seconds": max_seconds,
        "min_seconds": min_seconds,
        "max_gap": max_gap,
        "pad": pad,
    }

    def segment_all(lengths: dict[str, float] | None):
        """Segment every lecture; `lengths` = audio seconds per lecture."""
        stats, by_split = SegmentStats(), {s: [] for s in SPLITS}
        for lec in lectures:
            segments, lecture_stats = merge_cues(
                load_cues(lec.reference),
                lec.lecture_id,
                audio_seconds=None if lengths is None else lengths[lec.lecture_id],
                **segment_options,
            )
            stats.add(lecture_stats)
            by_split[split_of[lec.lecture_id]].extend(segments)
        return by_split, stats

    # Phase 1 (cheap, captions only): segment and check leakage BEFORE the
    # slow audio work, so a problem is reported in a second, not after
    # minutes of decoding that would be thrown away.
    by_split, dropped = segment_all(None)
    leakage = check_leakage(by_split, groups)
    if not leakage.ok:
        raise FinetuneDataError("Leakage between splits: " + "; ".join(leakage.errors))

    if not dry_run:
        _clear_output(out_dir, force)
        lengths: dict[str, float] = {}
        for lec in lectures:
            # One lecture's audio at 16 kHz float32 is about 230 MB per
            # hour, so we hold only one lecture in memory at a time.
            audio = load(lec.audio, sample_rate)
            lengths[lec.lecture_id] = len(audio) / sample_rate
            segments, _ = merge_cues(
                load_cues(lec.reference),
                lec.lecture_id,
                audio_seconds=lengths[lec.lecture_id],
                **segment_options,
            )
            _write_clips(
                segments, audio, out_dir, split_of[lec.lecture_id], sample_rate, write
            )
            del audio  # free the memory before loading the next lecture
        # Phase 2 dropped clips that run past the end of the audio. Redo the
        # bookkeeping with the real lengths (merge_cues is deterministic, so
        # this matches exactly what was written) and re-check: removing
        # clips can only remove leakage, but the duplicate-text warnings
        # should describe the final data.
        by_split, dropped = segment_all(lengths)
        leakage = check_leakage(by_split, groups)

    splits = {}
    for split in SPLITS:
        segments = by_split[split]
        splits[split] = SplitSummary(
            lectures=sorted(i for i, s in split_of.items() if s == split),
            clips=len(segments),
            seconds=round(sum(s.duration for s in segments), 3),
            words=sum(len(s.text.split()) for s in segments),
        )
    summary = DatasetSummary(
        out_dir=str(out_dir),
        settings={
            "eval_dir": str(eval_dir),
            "group_by": group_by,
            "val": list(val or []),
            "test": list(test or []),
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "seed": seed,
            "max_seconds": max_seconds,
            "min_seconds": min_seconds,
            "max_gap": max_gap,
            "pad": pad,
            "sample_rate": sample_rate,
        },
        splits=splits,
        dropped=dropped,
        leakage=leakage,
        skipped_lectures=skipped,
        dry_run=dry_run,
    )
    if not dry_run:
        (out_dir / "dataset.json").write_text(
            json.dumps(summary.to_json(), indent=2) + "\n", encoding="utf-8"
        )
    return summary


def clip_name(segment: Segment) -> str:
    """Relative path of a clip inside its split folder."""
    return f"{segment.lecture_id}/{segment.lecture_id}_{segment.index:05d}.wav"


def _write_clips(
    segments: list[Segment],
    audio: np.ndarray,
    out_dir: Path,
    split: str,
    sample_rate: int,
    write: WriteClip,
) -> None:
    """Cut each segment out of the lecture audio and append its metadata line.

    Sample index = time in seconds x samples per second, so the clip for
    [start, end) is audio[round(start * sr) : round(end * sr)].
    """
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    # "a" = append: several lectures share one metadata.jsonl per split.
    with (split_dir / "metadata.jsonl").open("a", encoding="utf-8") as meta:
        for segment in segments:
            first = round(segment.start * sample_rate)
            last = round(segment.end * sample_rate)
            write(audio[first:last], split_dir / clip_name(segment), sample_rate)
            line = {
                "file_name": clip_name(segment),
                "text": segment.text,
                "lecture_id": segment.lecture_id,
                "start": segment.start,
                "end": segment.end,
                "duration": round(segment.duration, 3),
            }
            meta.write(json.dumps(line, ensure_ascii=False) + "\n")


def load_metadata(split_dir: str | Path) -> list[dict]:
    """Read a split's metadata.jsonl back (used by tests and Step 6.3)."""
    path = Path(split_dir) / "metadata.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summary_markdown(summary: DatasetSummary) -> str:
    """A short human-readable table of the build, printed by the CLI."""
    lines = [
        "| Split | Lectures | Clips | Hours | Words |",
        "|---|---|---:|---:|---:|",
    ]
    for name in SPLITS:
        s = summary.splits[name]
        lectures = ", ".join(s.lectures) or "-"
        lines.append(f"| {name} | {lectures} | {s.clips} | {s.hours:.2f} | {s.words} |")
    d = summary.dropped
    lines.append("")
    lines.append(
        f"Cues: {d.cues}. Dropped: {d.empty_after_cleaning} empty after "
        f"cleaning, {d.too_long} single cues over {summary.settings['max_seconds']:g}"
        f" s, {d.too_short} segments under {summary.settings['min_seconds']:g} s, "
        f"{d.past_audio_end} past the end of the audio."
    )
    if summary.skipped_lectures:
        lines.append(f"Skipped (no audio.flac): {', '.join(summary.skipped_lectures)}")
    for warning in summary.leakage.warnings:
        lines.append(f"Warning: {warning}")
    return "\n".join(lines)
