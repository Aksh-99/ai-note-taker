"""Shared pytest setup for every test file.

pytest loads `conftest.py` automatically before running tests in this
folder, so anything defined here applies everywhere without imports.
"""

from __future__ import annotations

import pytest

import notetaker.llm


class LiveLLMCallError(AssertionError):
    """A test tried to reach a real LLM API."""


@pytest.fixture(autouse=True)
def _no_live_llm_calls(request, monkeypatch):
    """Guard rail: fail any (non-slow) test that would make a live LLM call.

    `autouse=True` means every test gets this fixture without asking for it.
    It swaps the module-level `urllib_transport` (the one function that
    really sends HTTP requests to Ollama, Groq or Hugging Face) for one that
    raises. Real clients pick up `urllib_transport` when they are created,
    so any client built inside a test is covered. `monkeypatch` undoes the
    swap after each test.

    This turns the project rule "CI never calls a live LLM" from a promise
    into something the test suite checks. Tests marked `@pytest.mark.slow`
    are allowed through, since those are opt-in and skipped in CI.
    """
    if request.node.get_closest_marker("slow"):
        return

    def refuse(url, headers, payload, timeout):
        raise LiveLLMCallError(
            f"test tried to call a live LLM at {url}; use FakeLLMClient, a "
            "fake transport or a recorded cassette (notetaker.cassette)"
        )

    monkeypatch.setattr(notetaker.llm, "urllib_transport", refuse)
