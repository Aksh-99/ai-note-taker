"""Smoke tests for the command-line interface.

A "smoke test" is the cheapest possible check that something basically
works: if it fails, something is badly broken ("where there's smoke...").
"""

import pytest

from notetaker import __version__
from notetaker.cli import build_parser, main


def test_version_flag_prints_version(capsys):
    # argparse's --version action prints and then raises SystemExit(0).
    # pytest.raises catches that so the test can keep checking things.
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0

    # capsys is a built-in pytest "fixture" that captures what the code
    # printed. pytest passes it in automatically because of the argument name.
    printed = capsys.readouterr().out
    assert printed.strip() == f"notetaker {__version__}"


def test_unknown_flag_is_rejected():
    # Bad arguments should exit with code 2 (argparse's usage-error code),
    # not crash with a Python traceback.
    with pytest.raises(SystemExit) as exit_info:
        main(["--not-a-real-flag"])
    assert exit_info.value.code == 2


def test_parser_program_name():
    assert build_parser().prog == "notetaker"
