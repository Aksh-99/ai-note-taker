"""Basic checks that the package installs and imports correctly."""

import notetaker
from notetaker.cli import main


def test_version_is_set():
    # If packaging is broken, the version falls back to "0.0.0".
    assert notetaker.__version__ != "0.0.0"


def test_cli_runs():
    # main() with no arguments prints help and exits cleanly.
    assert main([]) == 0
