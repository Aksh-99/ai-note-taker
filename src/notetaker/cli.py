"""Command-line entry point for `notetaker`.

Commands so far:
    notetaker --version   print the version
    notetaker config      print the active settings (API keys masked)
    notetaker transcribe lecture.mp3
                          transcribe a recording into transcript.json and
                          transcript.md (in outputs/<file name>/ by default)
"""

import argparse
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from notetaker import __version__
from notetaker.audio import AudioError
from notetaker.chunking import Chunk
from notetaker.config import WhisperModel, get_settings


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Kept separate from main() so tests can inspect the parser without
    actually running the program.
    """
    parser = argparse.ArgumentParser(
        prog="notetaker",
        description="AI lecture note taker built on free, open models.",
    )
    parser.add_argument(
        "--version", action="version", version=f"notetaker {__version__}"
    )
    # Subcommands (like `git commit` vs `git push`). `dest="command"` stores
    # which one was chosen, so main() can dispatch on it.
    subcommands = parser.add_subparsers(dest="command", metavar="<command>")
    subcommands.add_parser(
        "config", help="show active settings (from env vars, .env, defaults)"
    )

    tr = subcommands.add_parser(
        "transcribe",
        help="transcribe an audio/video file to transcript.json + transcript.md",
    )
    tr.add_argument("audio", type=Path, help="any file ffmpeg can read (mp3, mp4...)")
    tr.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="folder for the outputs (default: <NOTETAKER_OUTPUT_DIR>/<file name>)",
    )
    tr.add_argument(
        "-m",
        "--model",
        # get_args() pulls the allowed strings out of the Literal type in
        # config.py, so the CLI and the settings can never disagree.
        choices=get_args(WhisperModel),
        default=None,
        help="Whisper size (default: NOTETAKER_WHISPER_MODEL)",
    )
    tr.add_argument(
        "-l", "--language", default=None, help="language code like 'en' (default: auto)"
    )
    tr.add_argument(
        "--beam-size", type=int, default=5, help="beam search width (1 = greedy)"
    )
    tr.add_argument(
        "--title", default=None, help="heading for transcript.md (default: file name)"
    )
    tr.add_argument(
        "--max-chunk-seconds",
        type=float,
        default=None,
        help=(
            "split long recordings at pauses into pieces of at most this many "
            "seconds (default: NOTETAKER_CHUNK_MAX_SECONDS; 0 = no splitting)"
        ),
    )
    return parser


def show_config() -> int:
    """Print every setting as `name = value`, with secrets masked."""
    try:
        settings = get_settings()
    except ValidationError as error:
        # A bad value in the environment or .env: explain it instead of
        # dumping a long traceback. Exit code 1 signals failure to the shell.
        print(f"Invalid configuration:\n{error}")
        return 1
    for name, value in settings.redacted().items():
        print(f"{name} = {value}")
    return 0


def print_chunk_progress(index: int, total: int, chunk: Chunk) -> None:
    """Show which piece of a long recording is being transcribed.

    Passed to transcribe() as a "callback": a function handed to another
    function, which calls it back at the right moments (here, once per chunk).
    Silent for short files that fit in a single chunk.
    """
    from notetaker.export import format_timestamp

    if total > 1:
        print(
            f"  chunk {index}/{total} "
            f"[{format_timestamp(chunk.start)} - {format_timestamp(chunk.end)}]"
        )


def run_transcribe(args: argparse.Namespace) -> int:
    """Handle `notetaker transcribe`: transcribe one file and write outputs."""
    # Imported here so `notetaker --version` and `notetaker config` do not pay
    # the import cost of the transcription stack.
    from notetaker.export import write_outputs
    from notetaker.transcribe import transcribe

    audio_path: Path = args.audio
    if not audio_path.is_file():
        print(f"Error: file not found: {audio_path}")
        return 1
    if args.beam_size < 1:
        print("Error: --beam-size must be at least 1")
        return 1
    if args.max_chunk_seconds is not None and args.max_chunk_seconds < 0:
        print("Error: --max-chunk-seconds must be 0 or more")
        return 1
    try:
        settings = get_settings()
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1

    # `.stem` is the file name without its extension: lecture01.mp3 -> lecture01.
    # One folder per recording keeps notes, flashcards, etc. from later steps
    # next to the transcript they came from.
    out_dir = args.output_dir or settings.output_dir / audio_path.stem
    model_name = args.model or settings.whisper_model
    print(f"Transcribing {audio_path} with whisper-{model_name} ...")
    try:
        transcript = transcribe(
            audio_path,
            language=args.language,
            beam_size=args.beam_size,
            model_name=model_name,
            max_chunk_s=args.max_chunk_seconds,
            on_chunk=print_chunk_progress,
        )
    except AudioError as error:
        # Covers both "ffmpeg is not installed" and "ffmpeg could not read
        # this file". The message already says what to do.
        print(f"Error: {error}")
        return 1

    json_path, md_path = write_outputs(
        transcript, out_dir, title=args.title or audio_path.stem
    )
    print(
        f"Done: {len(transcript.segments)} segments, "
        f"{transcript.duration:.1f} s of audio, language '{transcript.language}'."
    )
    print(f"  {json_path}")
    print(f"  {md_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    `argv=None` means "read the real command-line arguments". Tests pass an
    explicit list instead, which lets them call main() like a normal function.
    The return value is the process exit code (0 means success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "config":
        return show_config()
    if args.command == "transcribe":
        return run_transcribe(args)
    # No subcommand given: show usage so the user sees what is available.
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
