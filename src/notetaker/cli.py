"""Command-line entry point for `notetaker`.

Right now it only reports the version. Subcommands such as
`notetaker transcribe lecture.mp3` arrive in Phase 1.
"""

import argparse

from notetaker import __version__


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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    `argv=None` means "read the real command-line arguments". Tests pass an
    explicit list instead, which lets them call main() like a normal function.
    The return value is the process exit code (0 means success).
    """
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
