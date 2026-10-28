"""Tests for the evaluation-set downloader (notetaker.evalset + CLI).

Nothing here touches the internet. A fake `fetch` returns canned HTML and
caption text from a dict of URL -> text, and a fake `extract` writes a few
bytes instead of running ffmpeg. One test runs the real ffmpeg extraction on
a tiny generated WAV file, which ffmpeg treats exactly like a URL.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

import notetaker.evalset as ev
from notetaker.cli import main
from notetaker.config import get_settings
from notetaker.evalset import (
    CaptionCue,
    EvalSetError,
    LectureSpec,
    cues_to_text,
    download_lecture,
    find_media_links,
    load_manifest,
    parse_captions,
    select_lectures,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = "https://ocw.mit.edu/courses/18-06/resources/lecture-1/"

SRT = """1
00:00:00,500 --> 00:00:03,000
The following content is provided
under a <i>Creative Commons</i> license.

2
00:00:03,000 --> 00:00:06,250
OK, this is linear algebra,
lecture one.
"""

VTT = """WEBVTT
Kind: captions

NOTE this comment block is ignored

intro
00:01.000 --> 00:02.500 align:start
Hello <c.yellow>world</c>.

01:00:00.000 --> 01:00:01.5
Last line.
"""

PAGE_HTML = f"""
<html><body>
  <a href="/courses/18-06/lec1.pdf">Transcript (PDF)</a>
  <a href="https://example.com/mirror/lec1.mp4">Mirror</a>
  <a class="download" href='https://archive.org/download/MIT18.06/lec1.mp4'>
    Download video</a>
  <track src="/courses/18-06/lec1.vtt" kind="captions">
  <a href="/courses/18-06/lec1.srt?raw=1">Captions</a>
  <a href="{PAGE}">this page</a>
</body></html>
"""

SPEC = LectureSpec(id="lec1", course="18.06", title="Lecture 1", page_url=PAGE)


class FakeWeb:
    """Stands in for the internet: a dict from URL to text, plus a call log."""

    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.fetched: list[str] = []
        self.extracted: list[str] = []

    def fetch(self, url: str) -> str:
        self.fetched.append(url)
        if url not in self.pages:
            raise EvalSetError(f"404 {url}")
        return self.pages[url]

    def extract(self, url: str, dest: Path) -> None:
        self.extracted.append(url)
        dest.write_bytes(b"fLaC fake audio")


def standard_web() -> FakeWeb:
    return FakeWeb(
        {
            PAGE: PAGE_HTML,
            "https://ocw.mit.edu/courses/18-06/lec1.srt?raw=1": SRT,
        }
    )


# ---- manifest ---------------------------------------------------------------


def write_manifest(tmp_path: Path, lectures: object) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"lectures": lectures}))
    return path


def test_committed_manifest_is_valid() -> None:
    specs = load_manifest(REPO_ROOT / "eval" / "ocw_lectures.json")
    assert len(specs) >= 3
    assert all(spec.page_url.startswith("https://ocw.mit.edu/") for spec in specs)


def test_manifest_reads_optional_overrides(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path,
        [
            {
                "id": "a",
                "course": "c",
                "title": "t",
                "page_url": "https://x/",
                "transcript_url": "https://x/a.srt",
            }
        ],
    )
    (spec,) = load_manifest(path)
    assert spec.transcript_url == "https://x/a.srt"
    assert spec.audio_url is None


@pytest.mark.parametrize(
    ("lectures", "message"),
    [
        ([], "non-empty"),
        ([{"id": "a", "course": "c", "title": "t"}], "missing ['page_url']"),
        ([{"id": "../x", "course": "c", "title": "t", "page_url": "u"}], "folder"),
        ([{"id": "a b", "course": "c", "title": "t", "page_url": "u"}], "folder"),
        (
            [{"id": "a", "course": "c", "title": "t", "page_url": "u"}] * 2,
            "twice",
        ),
    ],
)
def test_manifest_rejects_bad_entries(
    tmp_path: Path, lectures: object, message: str
) -> None:
    with pytest.raises(EvalSetError, match=message.replace("[", r"\[")):
        load_manifest(write_manifest(tmp_path, lectures))


def test_manifest_missing_or_invalid_json(tmp_path: Path) -> None:
    with pytest.raises(EvalSetError, match="not found"):
        load_manifest(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(EvalSetError, match="not valid JSON"):
        load_manifest(bad)


def test_select_lectures() -> None:
    other = LectureSpec(id="lec2", course="c", title="t", page_url="u")
    assert select_lectures([SPEC, other], None) == [SPEC, other]
    assert select_lectures([SPEC, other], ["lec2"]) == [other]
    with pytest.raises(EvalSetError, match="Unknown"):
        select_lectures([SPEC], ["nope"])


# ---- link finding -------------------------------------------------------------


def test_find_media_links_prefers_archive_mp4_and_srt() -> None:
    links = find_media_links(PAGE_HTML, PAGE)
    assert links.video_url == "https://archive.org/download/MIT18.06/lec1.mp4"
    # Relative link resolved against the page; query string kept.
    assert links.caption_url == "https://ocw.mit.edu/courses/18-06/lec1.srt?raw=1"


def test_find_media_links_falls_back_to_vtt_and_reports_missing() -> None:
    html = '<track src="captions/lec.vtt">'
    links = find_media_links(html, PAGE)
    assert links.caption_url == PAGE + "captions/lec.vtt"
    assert links.video_url is None
    assert find_media_links("<p>nothing</p>", PAGE).caption_url is None


# ---- caption parsing --------------------------------------------------------


def test_parse_srt() -> None:
    cues = parse_captions(SRT)
    assert cues == [
        CaptionCue(
            0.5,
            3.0,
            "The following content is provided under a Creative Commons license.",
        ),
        CaptionCue(3.0, 6.25, "OK, this is linear algebra, lecture one."),
    ]


def test_parse_vtt_with_header_notes_settings_and_tags() -> None:
    cues = parse_captions(VTT)
    assert cues == [
        CaptionCue(1.0, 2.5, "Hello world."),
        # "1.5" seconds: a short millisecond field is padded, so .5 = 500 ms.
        CaptionCue(3600.0, 3601.5, "Last line."),
    ]


def test_parse_handles_windows_line_endings_and_bom() -> None:
    cues = parse_captions("﻿" + SRT.replace("\n", "\r\n"))
    assert [c.start for c in cues] == [0.5, 3.0]


def test_parse_rejects_non_caption_text() -> None:
    with pytest.raises(EvalSetError, match="No caption cues"):
        parse_captions("<html>not captions</html>")


def test_cues_to_text() -> None:
    text = cues_to_text(parse_captions(SRT))
    assert text.splitlines()[1] == "OK, this is linear algebra, lecture one."
    assert text.endswith("\n")


# ---- downloading -------------------------------------------------------------


def test_download_lecture_writes_all_files(tmp_path: Path) -> None:
    web = standard_web()
    files = download_lecture(SPEC, tmp_path, fetch=web.fetch, extract=web.extract)

    assert not files.skipped
    assert files.folder == tmp_path / "lec1"
    assert files.audio.read_bytes() == b"fLaC fake audio"
    assert json.loads(files.reference_json.read_text())[1]["end"] == 6.25
    assert files.reference_txt.read_text().startswith("The following content")
    meta = json.loads(files.meta.read_text())
    assert meta["resolved_audio_url"] == web.extracted[0]
    assert meta["num_cues"] == 2
    assert meta["reference_duration_s"] == 6.25
    assert "CC BY-NC-SA" in meta["license"]
    # No temporary .part files are left behind.
    assert not list(files.folder.glob("*.part"))


def test_second_run_skips_and_force_redownloads(tmp_path: Path) -> None:
    web = standard_web()
    download_lecture(SPEC, tmp_path, fetch=web.fetch, extract=web.extract)
    again = download_lecture(SPEC, tmp_path, fetch=web.fetch, extract=web.extract)
    assert again.skipped
    assert len(web.extracted) == 1  # nothing fetched the second time
    download_lecture(SPEC, tmp_path, fetch=web.fetch, extract=web.extract, force=True)
    assert len(web.extracted) == 2


def test_missing_audio_is_downloaded_on_rerun(tmp_path: Path) -> None:
    web = standard_web()
    files = download_lecture(SPEC, tmp_path, fetch=web.fetch, extract=web.extract)
    files.audio.unlink()
    again = download_lecture(SPEC, tmp_path, fetch=web.fetch, extract=web.extract)
    assert not again.skipped
    assert again.audio.is_file()


def test_no_audio_mode_skips_video(tmp_path: Path) -> None:
    web = standard_web()
    files = download_lecture(
        SPEC, tmp_path, fetch=web.fetch, extract=web.extract, with_audio=False
    )
    assert web.extracted == []
    assert not files.audio.exists()
    assert files.reference_txt.is_file()
    assert json.loads(files.meta.read_text())["resolved_audio_url"] is None


def test_manifest_urls_skip_the_page(tmp_path: Path) -> None:
    spec = LectureSpec(
        id="x",
        course="c",
        title="t",
        page_url="https://unreachable/",
        audio_url="https://cdn/x.mp4",
        transcript_url="https://cdn/x.srt",
    )
    web = FakeWeb({"https://cdn/x.srt": SRT})
    download_lecture(spec, tmp_path, fetch=web.fetch, extract=web.extract)
    assert web.fetched == ["https://cdn/x.srt"]
    assert web.extracted == ["https://cdn/x.mp4"]


def test_page_without_captions_gives_helpful_error(tmp_path: Path) -> None:
    web = FakeWeb({PAGE: '<a href="v.mp4">video</a>'})
    with pytest.raises(EvalSetError, match="transcript_url"):
        download_lecture(SPEC, tmp_path, fetch=web.fetch, extract=web.extract)


def test_page_without_video_gives_helpful_error(tmp_path: Path) -> None:
    web = FakeWeb({PAGE: '<a href="c.srt">captions</a>', PAGE + "c.srt": SRT})
    with pytest.raises(EvalSetError, match="audio_url"):
        download_lecture(SPEC, tmp_path, fetch=web.fetch, extract=web.extract)


def test_failed_extraction_leaves_no_partial_audio(tmp_path: Path) -> None:
    web = standard_web()

    def broken_extract(url: str, dest: Path) -> None:
        dest.write_bytes(b"half")
        raise EvalSetError("connection reset")

    with pytest.raises(EvalSetError, match="connection reset"):
        download_lecture(SPEC, tmp_path, fetch=web.fetch, extract=broken_extract)
    folder = tmp_path / "lec1"
    assert not (folder / "audio.flac").exists()
    assert not list(folder.glob("*.part"))


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)


@needs_ffmpeg
def test_real_ffmpeg_extracts_16k_mono_flac(tmp_path: Path) -> None:
    # A 1-second stereo 44.1 kHz tone, standing in for a lecture video.
    source = tmp_path / "video.wav"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=1:sample_rate=44100",
            "-ac", "2", str(source),
        ],
        check=True,
    )  # fmt: skip
    dest = tmp_path / "audio.flac.part"
    ev.extract_audio(str(source), dest)
    # Decode back to WAV with ffmpeg to read the format with the stdlib.
    check = tmp_path / "check.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(dest), str(check)],
        check=True,
    )
    with wave.open(str(check)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16_000
        assert abs(wav.getnframes() - 16_000) < 200


@needs_ffmpeg
def test_real_ffmpeg_error_is_wrapped(tmp_path: Path) -> None:
    with pytest.raises(EvalSetError, match="could not extract"):
        ev.extract_audio(str(tmp_path / "missing.mp4"), tmp_path / "out.flac")


# ---- CLI ----------------------------------------------------------------------


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    return write_manifest(
        tmp_path,
        [
            {"id": "lec1", "course": "18.06", "title": "Lecture 1", "page_url": PAGE},
            {"id": "lec2", "course": "6.006", "title": "Lecture 2", "page_url": "x"},
        ],
    )


def test_cli_list(manifest: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["evalset", "--manifest", str(manifest), "--list"]) == 0
    out = capsys.readouterr().out
    assert "lec1" in out and "6.006: Lecture 2" in out


def test_cli_downloads_selected_lecture(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    web = standard_web()
    # download_lecture's defaults were bound when the module loaded, so we
    # wrap it to pass the fakes in explicitly.
    real = ev.download_lecture
    monkeypatch.setattr(
        ev,
        "download_lecture",
        lambda spec, out, **kw: real(
            spec, out, fetch=web.fetch, extract=web.extract, **kw
        ),
    )
    out_dir = tmp_path / "out"
    code = main(
        ["evalset", "--manifest", str(manifest), "--only", "lec1", "-o", str(out_dir)]
    )
    assert code == 0
    assert (out_dir / "lec1" / "audio.flac").is_file()
    assert not (out_dir / "lec2").exists()
    assert "1/1 lectures ready" in capsys.readouterr().out


def test_cli_reports_failures_and_continues(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    web = standard_web()  # knows lec1's page, not lec2's
    real = ev.download_lecture
    monkeypatch.setattr(
        ev,
        "download_lecture",
        lambda spec, out, **kw: real(
            spec, out, fetch=web.fetch, extract=web.extract, **kw
        ),
    )
    code = main(["evalset", "--manifest", str(manifest), "-o", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert code == 1
    assert "failed: 404" in out
    assert "1/2 lectures ready" in out


def test_cli_default_output_dir_uses_data_dir(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOTETAKER_DATA_DIR", str(tmp_path / "d"))
    get_settings.cache_clear()
    seen: list[Path] = []
    monkeypatch.setattr(
        ev,
        "download_lecture",
        lambda spec, out, **kw: (
            seen.append(Path(out)) or ev.lecture_paths(spec, Path(out))
        ),
    )
    try:
        assert main(["evalset", "--manifest", str(manifest), "--no-audio"]) == 0
    finally:
        get_settings.cache_clear()
    assert seen[0] == tmp_path / "d" / "raw" / "ocw"


def test_cli_bad_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["evalset", "--manifest", str(tmp_path / "none.json")]) == 1
    assert "Manifest not found" in capsys.readouterr().out
