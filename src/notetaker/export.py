"""Turn a `Transcript` into files people (and later steps) can read.

Two output formats:
    transcript.json  the full machine-readable result (segments, words,
                     confidence scores). Every later step (notes, search,
                     evaluation) reads this file, so a lecture is transcribed
                     only once.
    transcript.md    a human-readable version: a short header with metadata,
                     then the speech grouped into timestamped paragraphs.

How paragraphs are formed:
    Whisper's segments are sentence-sized, so printing one per line gives
    hundreds of choppy lines for a long lecture. Instead we glue consecutive
    segments together and start a new paragraph when either
      * the speaker pauses for at least `pause_seconds` (a natural break), or
      * the current paragraph already spans `max_paragraph_seconds`
        (so a lecturer who never pauses still gets readable blocks).
    Each paragraph is labelled with the time it starts, like `[00:12:41]`,
    which matches what you would type into a video player to jump there.
"""

from __future__ import annotations

from pathlib import Path

from notetaker.transcribe import Segment, Transcript

# Defaults chosen by reading a few lectures: a 2 s silence is usually a change
# of thought, and 60 s keeps paragraphs to roughly 120-180 spoken words.
DEFAULT_PAUSE_SECONDS = 2.0
DEFAULT_MAX_PARAGRAPH_SECONDS = 60.0


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS, e.g. 761.4 -> '00:12:41'.

    We truncate (floor) rather than round, so a paragraph starting at 59.9 s is
    labelled 00:00:59. Jumping there in a player lands just before the speech
    instead of just after its first word.
    """
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")
    total = int(seconds)  # int() floors positive numbers
    # divmod(a, b) returns (a // b, a % b) in one call.
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def group_paragraphs(
    segments: list[Segment],
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    max_paragraph_seconds: float = DEFAULT_MAX_PARAGRAPH_SECONDS,
) -> list[list[Segment]]:
    """Split segments into paragraphs using pauses and a maximum length.

    Returns a list of paragraphs, each a non-empty list of segments in their
    original order. Segments with empty text (Whisper sometimes emits these on
    silence) are dropped.
    """
    if pause_seconds <= 0 or max_paragraph_seconds <= 0:
        raise ValueError("pause_seconds and max_paragraph_seconds must be positive")

    paragraphs: list[list[Segment]] = []
    current: list[Segment] = []
    for seg in segments:
        if not seg.text:
            continue
        if current:
            gap = seg.start - current[-1].end
            span = seg.end - current[0].start
            if gap >= pause_seconds or span > max_paragraph_seconds:
                paragraphs.append(current)
                current = []
        current.append(seg)
    if current:  # the last paragraph never hits a break, so flush it here
        paragraphs.append(current)
    return paragraphs


def to_markdown(
    transcript: Transcript,
    title: str = "Transcript",
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    max_paragraph_seconds: float = DEFAULT_MAX_PARAGRAPH_SECONDS,
) -> str:
    """Render a transcript as Markdown with timestamped paragraphs."""
    lines = [
        f"# {title}",
        "",
        f"- **Duration:** {format_timestamp(transcript.duration)}",
        f"- **Language:** {transcript.language}",
        f"- **Model:** whisper-{transcript.model}",
        f"- **Segments:** {len(transcript.segments)}",
        "",
    ]
    paragraphs = group_paragraphs(
        transcript.segments, pause_seconds, max_paragraph_seconds
    )
    if not paragraphs:
        lines.append("_No speech detected._")
        lines.append("")
    for para in paragraphs:
        text = " ".join(seg.text for seg in para)
        lines.append(f"**[{format_timestamp(para[0].start)}]** {text}")
        lines.append("")  # a blank line separates Markdown paragraphs
    return "\n".join(lines)


def write_outputs(
    transcript: Transcript, out_dir: str | Path, title: str = "Transcript"
) -> tuple[Path, Path]:
    """Write transcript.json and transcript.md into `out_dir`.

    Returns the two paths (json, markdown) so the caller can report them.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = transcript.save_json(out_dir / "transcript.json")
    md_path = out_dir / "transcript.md"
    md_path.write_text(to_markdown(transcript, title=title), encoding="utf-8")
    return json_path, md_path
