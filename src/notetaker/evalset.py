"""Download MIT OpenCourseWare lectures and their official transcripts.

Why this exists:
    To measure how good our transcripts are (word error rate, Step 2.3) we need
    lectures where a human already wrote down exactly what was said. MIT
    OpenCourseWare (OCW) publishes lecture videos together with professionally
    made caption files, under a free CC BY-NC-SA license. Those captions are our
    "ground truth" (the reference transcript).

What gets saved, per lecture, in `<data_dir>/raw/ocw/<lecture id>/`:
    audio.flac       16 kHz mono audio, exactly what Whisper wants. FLAC is
                     lossless but about half the size of WAV.
    reference.json   the caption cues: [{"start", "end", "text"}, ...]
    reference.txt    the same cues as plain text, one cue per line
    meta.json        where everything came from, the license, and the date

    `data/raw/` is gitignored: the files are large, and the CC BY-NC-SA license
    attaches attribution and non-commercial conditions to every copy, so each
    person downloads their own instead of us re-hosting them in git.

How the download works:
    The list of lectures lives in a small JSON "manifest" (eval/ocw_lectures.json)
    that IS committed, so everyone evaluates on the same lectures. For each
    entry we
      1. fetch the lecture's OCW web page and scan its HTML for links to the
         video (.mp4) and the caption file (.srt or .vtt),
      2. download and parse the captions into timed cues,
      3. let ffmpeg read the video straight from its URL and keep only the
         audio track, resampled to 16 kHz mono. The video itself is never
         written to disk, which saves a few hundred MB per lecture.
    A manifest entry may also give `audio_url` / `transcript_url` directly,
    which skips the page scan for that file (useful if OCW changes its layout).

Design note (dependency injection):
    `download_lecture` receives the "fetch a URL" and "extract audio" functions
    as arguments instead of calling the network directly. Real runs use the
    defaults below; tests pass fakes that return canned HTML and captions, so
    the whole pipeline is tested offline, in milliseconds.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

from notetaker.audio import find_ffmpeg

# The manifest committed with the repo. Paths are relative to the repo root,
# which is where the CLI is normally run from.
DEFAULT_MANIFEST = Path("eval/ocw_lectures.json")

# Whisper's expected format (see audio.py). Stored once here so the saved
# FLAC files are ready to transcribe with no further conversion.
EVAL_SAMPLE_RATE = 16_000

# Lecture ids become folder names, so we only allow characters that are safe
# in a path on every operating system (no slashes, no spaces, no "..").
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Some web servers refuse requests that do not look like they come from a
# browser or a named tool, so we identify ourselves politely.
USER_AGENT = "notetaker-evalset/0.1 (+https://github.com/Aksh-99/ai-note-taker)"

# Type aliases for the injected functions, to keep signatures readable.
FetchText = Callable[[str], str]
ExtractAudio = Callable[[str, Path], None]


class EvalSetError(RuntimeError):
    """A lecture could not be downloaded or parsed (message says why)."""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LectureSpec:
    """One lecture from the manifest.

    `frozen=True` makes instances read-only, so a spec cannot be changed by
    accident halfway through a download.
    """

    id: str
    course: str
    title: str
    page_url: str
    audio_url: str | None = None
    transcript_url: str | None = None


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> list[LectureSpec]:
    """Read and validate the lecture manifest.

    Raises EvalSetError with a clear message for a missing file, a missing
    field, an unsafe id, or two lectures sharing an id.
    """
    path = Path(path)
    if not path.is_file():
        raise EvalSetError(f"Manifest not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvalSetError(f"Manifest {path} is not valid JSON: {error}") from error

    entries = data.get("lectures") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise EvalSetError(f"Manifest {path} needs a non-empty 'lectures' list")

    specs: list[LectureSpec] = []
    seen: set[str] = set()
    for number, entry in enumerate(entries, start=1):
        missing = [
            key
            for key in ("id", "course", "title", "page_url")
            if not isinstance(entry.get(key), str) or not entry[key].strip()
        ]
        if missing:
            raise EvalSetError(f"Lecture #{number} in {path} is missing {missing}")
        lecture_id = entry["id"]
        if not _ID_PATTERN.match(lecture_id) or ".." in lecture_id:
            raise EvalSetError(
                f"Lecture id {lecture_id!r} may only use letters, digits, '.', "
                "'_' and '-' (it becomes a folder name)"
            )
        if lecture_id in seen:
            raise EvalSetError(f"Lecture id {lecture_id!r} appears twice in {path}")
        seen.add(lecture_id)
        specs.append(
            LectureSpec(
                id=lecture_id,
                course=entry["course"],
                title=entry["title"],
                page_url=entry["page_url"],
                audio_url=entry.get("audio_url") or None,
                transcript_url=entry.get("transcript_url") or None,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Finding the video and caption links on an OCW page
# ---------------------------------------------------------------------------


class _LinkCollector(HTMLParser):
    """Collect every URL in an HTML page's href= and src= attributes.

    Python's built-in HTMLParser reads the page tag by tag and calls
    `handle_starttag` for each opening tag, e.g. <a href="...">. We subclass it
    and override that one method ("override" = replace the parent's version
    with our own). Using a real parser instead of a regex copes with quoting
    styles, attribute order, and line breaks inside tags.
    """

    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in ("href", "src") and value:
                self.urls.append(value.strip())


@dataclass(frozen=True)
class MediaLinks:
    """The links we care about on a lecture page (None = not found)."""

    video_url: str | None
    caption_url: str | None


def _extension(url: str) -> str:
    """File extension of a URL's path, lower case, ignoring ?query strings."""
    return Path(urlparse(url).path).suffix.lower()


def find_media_links(html: str, base_url: str) -> MediaLinks:
    """Scan a lecture page for its video file and its caption file.

    Relative links like "/courses/.../captions.srt" are turned into full URLs
    with `urljoin(base_url, link)`. Among several candidates we prefer:
      * video: an archive.org .mp4 (OCW hosts its downloads there), else any .mp4
      * captions: .srt, else .vtt (both are timed text; SRT is the OCW default)
    """
    collector = _LinkCollector()
    collector.feed(html)
    # dict.fromkeys keeps the first occurrence of each URL, in page order
    # (a common idiom for "remove duplicates but keep the order").
    urls = list(dict.fromkeys(urljoin(base_url, u) for u in collector.urls))

    videos = [u for u in urls if _extension(u) == ".mp4"]
    videos.sort(key=lambda u: "archive.org" not in urlparse(u).netloc)
    captions = [u for u in urls if _extension(u) in (".srt", ".vtt")]
    captions.sort(key=lambda u: _extension(u) != ".srt")
    # The sorts above are "stable": items that compare equal keep page order,
    # and False sorts before True, so preferred links move to the front.
    return MediaLinks(
        video_url=videos[0] if videos else None,
        caption_url=captions[0] if captions else None,
    )


# ---------------------------------------------------------------------------
# Parsing caption files (SRT and WebVTT)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptionCue:
    """One caption: the text shown on screen between `start` and `end` seconds."""

    start: float
    end: float
    text: str


# Matches a timing line in either format:
#   SRT:    00:01:02,500 --> 00:01:05,000
#   WebVTT: 00:01:02.500 --> 00:01:05.000   (hours are optional in VTT: 01:02.500)
_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})"
_TIMING_LINE = re.compile(rf"^\s*{_TIME}\s*-->\s*{_TIME}")
# Formatting tags inside captions, e.g. <i>, </b>, <c.yellow>, <00:00:01.000>.
_TAG = re.compile(r"<[^>]*>")


def _seconds(hours: str | None, minutes: str, secs: str, millis: str) -> float:
    """Convert timestamp pieces into seconds (milliseconds padded: "5" -> 500)."""
    return (
        int(hours or 0) * 3600
        + int(minutes) * 60
        + int(secs)
        + int(millis.ljust(3, "0")) / 1000
    )


def parse_captions(text: str) -> list[CaptionCue]:
    """Parse an SRT or WebVTT caption file into a list of cues.

    Both formats are blocks separated by blank lines. Each block has a timing
    line ("start --> end") followed by one or more lines of text; SRT also puts
    a running number above the timing line, and VTT starts with a "WEBVTT"
    header. So the rule is simple: find the timing line in each block, and
    everything after it is the caption text. Blocks without a timing line
    (numbers, the header, NOTE comments) are skipped.
    """
    # Normalize Windows (\r\n) and old Mac (\r) line endings, and drop a
    # byte-order mark some editors put at the very start of the file.
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[CaptionCue] = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            match = _TIMING_LINE.match(line)
            if match is None:
                continue
            groups = match.groups()  # 8 pieces: 4 for start, 4 for end
            start = _seconds(*groups[:4])
            end = _seconds(*groups[4:])
            words = " ".join(_TAG.sub("", t).strip() for t in lines[i + 1 :])
            words = " ".join(words.split())  # collapse repeated spaces
            if words:
                cues.append(CaptionCue(start=start, end=end, text=words))
            break
    if not cues:
        raise EvalSetError("No caption cues found (is this an SRT or VTT file?)")
    return cues


def cues_to_text(cues: list[CaptionCue]) -> str:
    """The plain reference transcript: one cue per line."""
    return "\n".join(cue.text for cue in cues) + "\n"


# ---------------------------------------------------------------------------
# Real network and ffmpeg implementations (replaced by fakes in tests)
# ---------------------------------------------------------------------------


def fetch_text(url: str, timeout: float = 60.0) -> str:
    """Download a web page or text file and return it as a string."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # Servers say which text encoding they used in the Content-Type
            # header; fall back to UTF-8 when they do not.
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except OSError as error:  # URLError and timeouts are both OSErrors
        raise EvalSetError(f"Could not download {url}: {error}") from error


def extract_audio(source: str, dest: Path) -> None:
    """Save only the audio of `source` (a URL or a local file) as 16 kHz FLAC.

    ffmpeg can read directly from an http(s) URL, so the video is streamed
    through and only the small audio file is written.
    """
    command = [
        find_ffmpeg(),
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",  # overwrite dest if it exists (we always write a fresh temp file)
        "-i", source,
        "-vn",  # drop the video stream
        "-ac", "1",  # mono
        "-ar", str(EVAL_SAMPLE_RATE),  # 16 kHz
        "-c:a", "flac",  # lossless compression
        "-f", "flac",  # name the format explicitly: dest ends in .part
        str(dest),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        raise EvalSetError(f"ffmpeg could not extract audio from {source}: {message}")


# ---------------------------------------------------------------------------
# Downloading one lecture
# ---------------------------------------------------------------------------


@dataclass
class LectureFiles:
    """Where one lecture's files ended up, and whether we downloaded anything."""

    lecture_id: str
    folder: Path
    audio: Path
    reference_json: Path
    reference_txt: Path
    meta: Path
    skipped: bool = False
    num_cues: int = 0
    notes: list[str] = field(default_factory=list)


def _write_atomically(path: Path, text: str) -> None:
    """Write a text file so it is either complete or absent, never half-written.

    We write to "<name>.part" first and then rename. A rename within one folder
    is atomic (it happens all at once), so if the program is interrupted, the
    real file name never points at a truncated file. That matters because a
    re-run skips lectures whose files already exist.
    """
    temp = path.with_name(path.name + ".part")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def lecture_paths(spec: LectureSpec, out_dir: Path) -> LectureFiles:
    """The file layout for one lecture (nothing is created yet)."""
    folder = Path(out_dir) / spec.id
    return LectureFiles(
        lecture_id=spec.id,
        folder=folder,
        audio=folder / "audio.flac",
        reference_json=folder / "reference.json",
        reference_txt=folder / "reference.txt",
        meta=folder / "meta.json",
    )


def download_lecture(
    spec: LectureSpec,
    out_dir: str | Path,
    *,
    fetch: FetchText = fetch_text,
    extract: ExtractAudio = extract_audio,
    force: bool = False,
    with_audio: bool = True,
) -> LectureFiles:
    """Download one lecture's reference transcript and (optionally) its audio.

    Args:
        spec: the lecture, from the manifest.
        out_dir: parent folder; files go into `out_dir/<spec.id>/`.
        fetch: function returning the text at a URL (injected for tests).
        extract: function saving a URL's audio to a path (injected for tests).
        force: re-download even if the files already exist.
        with_audio: False downloads only the transcript (fast, a few KB).

    Returns:
        LectureFiles with the paths and `skipped=True` if nothing was needed.
    """
    files = lecture_paths(spec, Path(out_dir))
    have_reference = files.reference_json.is_file() and files.reference_txt.is_file()
    have_audio = files.audio.is_file() or not with_audio
    if have_reference and have_audio and not force:
        files.skipped = True
        return files

    # Only fetch the HTML page if the manifest did not give both links.
    caption_url = spec.transcript_url
    video_url = spec.audio_url
    if caption_url is None or (with_audio and video_url is None):
        links = find_media_links(fetch(spec.page_url), spec.page_url)
        caption_url = caption_url or links.caption_url
        video_url = video_url or links.video_url
    if caption_url is None:
        raise EvalSetError(
            f"{spec.id}: no .srt/.vtt caption link on {spec.page_url}. Add a "
            "'transcript_url' for this lecture to the manifest."
        )
    if with_audio and video_url is None:
        raise EvalSetError(
            f"{spec.id}: no .mp4 video link on {spec.page_url}. Add an "
            "'audio_url' for this lecture to the manifest."
        )

    files.folder.mkdir(parents=True, exist_ok=True)

    # 1. Reference transcript (small, so always refreshed when we get here).
    cues = parse_captions(fetch(caption_url))
    files.num_cues = len(cues)
    _write_atomically(
        files.reference_json,
        json.dumps([asdict(c) for c in cues], indent=2, ensure_ascii=False) + "\n",
    )
    _write_atomically(files.reference_txt, cues_to_text(cues))

    # 2. Audio (large): only when missing or forced.
    if with_audio and (force or not files.audio.is_file()):
        temp = files.audio.with_name(files.audio.name + ".part")
        try:
            extract(video_url, temp)
            temp.replace(files.audio)
        finally:
            # On failure, do not leave a half-written .part file behind.
            temp.unlink(missing_ok=True)

    # 3. Provenance: where it came from and under which license.
    meta = {
        **asdict(spec),
        "resolved_audio_url": video_url if with_audio else None,
        "resolved_transcript_url": caption_url,
        "num_cues": len(cues),
        "reference_duration_s": round(cues[-1].end, 3),
        "sample_rate": EVAL_SAMPLE_RATE,
        "license": "CC BY-NC-SA 4.0, MIT OpenCourseWare",
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_atomically(files.meta, json.dumps(meta, indent=2) + "\n")
    return files


def select_lectures(
    specs: list[LectureSpec], only: list[str] | None
) -> list[LectureSpec]:
    """Keep only the lectures named in `only` (all of them if `only` is empty)."""
    if not only:
        return specs
    known = {spec.id for spec in specs}
    unknown = [lecture_id for lecture_id in only if lecture_id not in known]
    if unknown:
        raise EvalSetError(f"Unknown lecture id(s) {unknown}; known: {sorted(known)}")
    wanted = set(only)
    return [spec for spec in specs if spec.id in wanted]
