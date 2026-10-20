"""notetaker: turn lecture recordings into notes, flashcards, and searchable text.

This file marks the folder as a Python package. Anything defined here is
available as `notetaker.<name>` after `import notetaker`.
"""

# importlib.metadata reads the version from the installed package metadata
# (which comes from pyproject.toml), so the version number lives in one place.
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("notetaker")
except PackageNotFoundError:
    # Happens if someone imports the code without installing it first.
    __version__ = "0.0.0"

__all__ = ["__version__"]
