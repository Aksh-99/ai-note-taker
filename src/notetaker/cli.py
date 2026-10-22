"""Command-line entry point for `notetaker`.

Commands so far:
    notetaker --version   print the version
    notetaker config      print the active settings (API keys masked)

More subcommands, such as `notetaker transcribe lecture.mp3`, arrive in Phase 1.
"""

import argparse

from pydantic import ValidationError

from notetaker import __version__
from notetaker.config import get_settings


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
    # No subcommand given: show usage so the user sees what is available.
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
