"""Command-line entry point for `notetaker`.

Commands so far:
    notetaker --version   print the version
    notetaker config      print the active settings (API keys masked)
    notetaker transcribe lecture.mp3
                          transcribe a recording into transcript.json and
                          transcript.md (in outputs/<file name>/ by default)
    notetaker benchmark lecture.mp3 -m tiny base small
                          measure speed (real-time factor) and memory of
                          several Whisper sizes, report in reports/benchmark/
    notetaker evalset     download the MIT OCW evaluation lectures (audio +
                          official transcripts) into data/raw/ocw/
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

    bm = subcommands.add_parser(
        "benchmark",
        help="compare speed and memory of Whisper sizes on one recording",
    )
    bm.add_argument("audio", type=Path, help="a lecture recording ffmpeg can read")
    bm.add_argument(
        "-m",
        "--models",
        nargs="+",  # one or more values: -m tiny base small
        choices=get_args(WhisperModel),
        default=["tiny", "base", "small"],
        help="Whisper sizes to compare (default: tiny base small)",
    )
    bm.add_argument(
        "--repeats", type=int, default=1, help="timed runs per model (median kept)"
    )
    bm.add_argument(
        "--beam-size", type=int, default=5, help="beam search width (1 = greedy)"
    )
    bm.add_argument(
        "-l", "--language", default=None, help="language code like 'en' (default: auto)"
    )
    bm.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="only use the first N seconds of the recording, for a quicker run",
    )
    bm.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("reports/benchmark"),
        help="folder for benchmark.json + benchmark.md (default: reports/benchmark)",
    )
    bm.add_argument(
        "--no-isolate",
        action="store_true",
        help=(
            "run all models in this process (faster start-up, but the memory "
            "numbers of later models include earlier ones)"
        ),
    )
    ev = subcommands.add_parser(
        "evalset",
        help="download the MIT OpenCourseWare evaluation lectures + transcripts",
    )
    ev.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval/ocw_lectures.json"),
        help="JSON list of lectures (default: eval/ocw_lectures.json)",
    )
    ev.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="where to save them (default: <NOTETAKER_DATA_DIR>/raw/ocw)",
    )
    ev.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="ID",
        help="download just these lecture ids (default: all)",
    )
    ev.add_argument("--list", action="store_true", help="list the lectures and exit")
    ev.add_argument(
        "--no-audio",
        action="store_true",
        help="only download the reference transcripts (a few KB each)",
    )
    ev.add_argument(
        "--force", action="store_true", help="re-download files that already exist"
    )
    nm = subcommands.add_parser(
        "normalize",
        help="print a transcript in the normalized form used for WER scoring",
    )
    nm.add_argument(
        "file",
        type=Path,
        help="transcript.json, reference.json, or any plain-text file",
    )
    nm.add_argument(
        "--keep-digits",
        action="store_true",
        help='leave numbers as digits instead of spelling them ("3" not "three")',
    )
    nm.add_argument(
        "--keep-fillers",
        action="store_true",
        help="keep filler words such as um, uh, hmm",
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


def run_benchmark(args: argparse.Namespace) -> int:
    """Handle `notetaker benchmark`: time each model and write a report."""
    from notetaker.benchmark import (
        WHISPER_SAMPLE_RATE,
        benchmark_model,
        load_benchmark_audio,
        run_isolated,
        write_report,
    )

    audio_path: Path = args.audio
    if not audio_path.is_file():
        print(f"Error: file not found: {audio_path}")
        return 1
    if args.repeats < 1 or args.beam_size < 1:
        print("Error: --repeats and --beam-size must be at least 1")
        return 1
    try:
        get_settings()
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1

    options = {
        "repeats": args.repeats,
        "beam_size": args.beam_size,
        "language": args.language,
    }
    results = []
    try:
        # Decode once up front: this validates the file before any model loads,
        # and the in-process mode reuses the waveform for every model.
        audio = load_benchmark_audio(audio_path, args.max_seconds)
        seconds = len(audio) / WHISPER_SAMPLE_RATE
        print(f"Benchmarking {audio_path} ({seconds:.1f} s of audio)")
        for name in args.models:
            print(f"  whisper-{name} ...", flush=True)
            try:
                if args.no_isolate:
                    result = benchmark_model(audio, name, **options)
                else:
                    result = run_isolated(
                        audio_path, name, max_seconds=args.max_seconds, **options
                    )
            except Exception as error:  # noqa: BLE001
                # One model failing (most often: its weights could not be
                # downloaded, or it ran out of memory) should not throw away
                # the numbers already measured for the others.
                print(f"    failed: {type(error).__name__}: {error}")
                continue
            results.append(result)
            print(
                f"    load {result.load_seconds:.1f} s, "
                f"transcribe {result.median_transcribe_seconds:.1f} s, "
                f"RTF {result.rtf:.3f} ({result.speedup:.1f}x real time)"
            )
    except AudioError as error:
        print(f"Error: {error}")
        return 1

    if not results:
        print("Error: no model could be benchmarked (see messages above)")
        return 1
    json_path, md_path = write_report(results, args.output_dir, audio_path.name)
    print(f"Report written:\n  {json_path}\n  {md_path}")
    return 0


def run_evalset(args: argparse.Namespace) -> int:
    """Handle `notetaker evalset`: download the evaluation lectures."""
    from notetaker.evalset import (
        EvalSetError,
        download_lecture,
        load_manifest,
        select_lectures,
    )

    try:
        specs = select_lectures(load_manifest(args.manifest), args.only)
    except EvalSetError as error:
        print(f"Error: {error}")
        return 1
    if args.list:
        for spec in specs:
            print(f"{spec.id:<20} {spec.course}: {spec.title}")
        return 0
    try:
        out_dir = args.output_dir or get_settings().data_dir / "raw" / "ocw"
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1

    failures = 0
    for spec in specs:
        print(f"{spec.id}: {spec.title}", flush=True)
        try:
            files = download_lecture(
                spec, out_dir, force=args.force, with_audio=not args.no_audio
            )
        except Exception as error:  # noqa: BLE001
            # Network and ffmpeg errors come in many types. One broken lecture
            # should not stop the others, so report it and carry on.
            print(f"  failed: {error}")
            failures += 1
            continue
        if files.skipped:
            print(f"  already downloaded in {files.folder} (use --force to redo)")
        else:
            print(f"  saved {files.num_cues} caption cues to {files.folder}")
    done = len(specs) - failures
    print(f"{done}/{len(specs)} lectures ready in {out_dir}")
    return 0 if failures == 0 else 1


def run_normalize(args: argparse.Namespace) -> int:
    """Handle `notetaker normalize`: show exactly what the WER scorer will see."""
    from notetaker.normalize import load_text, normalize_text

    try:
        text = load_text(args.file)
    except (OSError, ValueError) as error:
        # OSError: missing/unreadable file. ValueError: bad JSON (the
        # json module's JSONDecodeError is a subclass of ValueError).
        print(f"Error: {error}")
        return 1
    print(
        normalize_text(
            text,
            numbers_to_words=not args.keep_digits,
            remove_fillers=not args.keep_fillers,
        )
    )
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
    if args.command == "benchmark":
        return run_benchmark(args)
    if args.command == "evalset":
        return run_evalset(args)
    if args.command == "normalize":
        return run_normalize(args)
    # No subcommand given: show usage so the user sees what is available.
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
