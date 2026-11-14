"""Record real LLM calls once, replay them forever (Step 4.4).

The problem
-----------
The agent pipeline (Step 4.1) makes a dozen LLM calls per lecture. Testing it
against a live model is slow, costs free-tier quota, needs a key or a running
Ollama, and gives a slightly different answer every run. Hand-written fake
replies (like `RoutingFakeClient` in tests/test_agent.py) fix that, but they
only ever contain JSON *we* imagined, which is usually cleaner than what a
small open model really sends back.

The pattern: a "cassette"
-------------------------
This is the record/replay idea popularised by Ruby's VCR library and
Python's vcrpy, applied at the LLM-client level instead of the HTTP level:

* **Record** once: wrap a real client in `RecordingClient`. It passes every
  request through unchanged and writes each (request, reply) pair to a JSON
  file, the "cassette".
* **Replay** in tests and CI: `ReplayClient` reads the cassette and answers
  each request with the reply that was recorded for it. No network, same
  output every time, and the replies are what a real model actually said.

How a request finds its reply
-----------------------------
Each request is turned into a *key*: a SHA-256 hash of its messages plus the
settings that change the answer (JSON mode, max tokens). Hashing gives a
short fixed-length fingerprint, so equal requests get equal keys and any
change, even one character in a prompt, gives a different key.

That second property is deliberate. If someone edits a prompt, the old
recording no longer describes what the code sends, so replay stops with a
`CassetteMiss` error that shows which recorded request was closest and where
the text first differs. The fix is to re-record (`make record-fixtures`).
This turns "the prompt changed" into a visible, reviewable event instead of
a test that silently keeps passing on stale data.

The model name and temperature are stored but are *not* part of the key, so
a cassette recorded with Groq replays fine on a machine configured for
Ollama.

Repeated requests
-----------------
The same request can legitimately be sent twice in one run (two identical
chunks, or a stage that retries). So each key maps to a *queue* of replies,
handed out in recorded order. If a queue runs out, its last reply is reused,
which keeps replay deterministic.

`CassetteMiss` is a plain `RuntimeError`, not an `LLMError`, on purpose: the
pipeline treats `LLMError` as "this stage failed, carry on", and a stale
cassette is a test-setup problem that should stop the run loudly.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from notetaker.llm import LLMResponse, Message, as_messages

# Bump when the file layout changes, so old cassettes fail with a clear error.
CASSETTE_VERSION = 1


class CassetteMiss(RuntimeError):
    """Replay was asked for a request that is not on the cassette."""


class CassetteFormatError(ValueError):
    """The cassette file is not one this code understands."""


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def request_key(
    messages: Sequence[Message],
    *,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> str:
    """Fingerprint of one chat request (first 16 hex chars of SHA-256).

    `json.dumps(..., sort_keys=True)` gives a canonical string: the same
    request always serialises to the same bytes, so it always hashes to the
    same key. 16 hex characters = 64 bits, plenty to avoid accidental
    collisions among a few hundred recorded requests.
    """
    canonical = json.dumps(
        {
            "messages": [m.to_dict() for m in messages],
            "json_mode": bool(json_mode),
            "max_tokens": max_tokens,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# The cassette file
# ---------------------------------------------------------------------------


@dataclass
class Interaction:
    """One recorded request and the reply it got."""

    key: str
    messages: list[Message]
    json_mode: bool
    max_tokens: int | None
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "request": {
                "messages": [m.to_dict() for m in self.messages],
                "json_mode": self.json_mode,
                "max_tokens": self.max_tokens,
            },
            "response": {
                "text": self.text,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Interaction:
        request, response = data["request"], data["response"]
        messages = [Message(m["role"], m["content"]) for m in request["messages"]]
        interaction = cls(
            key=data["key"],
            messages=messages,
            json_mode=bool(request.get("json_mode", False)),
            max_tokens=request.get("max_tokens"),
            text=response["text"],
            prompt_tokens=response.get("prompt_tokens"),
            completion_tokens=response.get("completion_tokens"),
        )
        # Guard against hand edits: if someone changed a prompt in the file
        # but not its key, replay would match the wrong request.
        expected = request_key(
            messages, json_mode=interaction.json_mode, max_tokens=interaction.max_tokens
        )
        if expected != interaction.key:
            raise CassetteFormatError(
                f"interaction {interaction.key} was edited: its request now "
                f"hashes to {expected}. Re-record instead of editing by hand."
            )
        return interaction


@dataclass
class Cassette:
    """All interactions of one recording session, plus where they came from."""

    provider: str = "unknown"
    model: str = "unknown"
    note: str = ""
    interactions: list[Interaction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CASSETTE_VERSION,
            "provider": self.provider,
            "model": self.model,
            "note": self.note,
            "interactions": [i.to_dict() for i in self.interactions],
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # indent=2 + a trailing newline keeps git diffs of re-recordings
        # readable: a changed reply shows up as a few changed lines.
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        path.write_text(text + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Cassette:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise CassetteFormatError(f"{path} is not valid JSON: {error}") from error
        if not isinstance(data, dict) or data.get("version") != CASSETTE_VERSION:
            raise CassetteFormatError(
                f"{path} is not a version {CASSETTE_VERSION} cassette"
            )
        return cls(
            provider=data.get("provider", "unknown"),
            model=data.get("model", "unknown"),
            note=data.get("note", ""),
            interactions=[Interaction.from_dict(i) for i in data["interactions"]],
        )


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


class RecordingClient:
    """Wrap a real client and write down every request and reply.

    This is the "decorator" (or "wrapper") pattern: it has the same `chat()`
    and `complete()` methods as the client it wraps, so the pipeline cannot
    tell the difference, but it does a little extra work on the side.
    Call `save()` (or use it in a `with` block) to write the cassette.
    """

    def __init__(self, inner: Any, path: Path | None = None, *, note: str = ""):
        self.inner = inner
        self.path = Path(path) if path is not None else None
        self.model = getattr(inner, "model", "unknown")
        self.provider = getattr(inner, "provider", "unknown")
        self.cassette = Cassette(provider=self.provider, model=self.model, note=note)

    def chat(
        self,
        messages: Iterable[Message],
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = list(messages)
        reply = self.inner.chat(
            messages, json_mode=json_mode, max_tokens=max_tokens, **kwargs
        )
        key = request_key(messages, json_mode=json_mode, max_tokens=max_tokens)
        self.cassette.interactions.append(
            Interaction(
                key=key,
                messages=messages,
                json_mode=json_mode,
                max_tokens=max_tokens,
                text=reply.text,
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
            )
        )
        return reply

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any):
        return self.chat(as_messages(prompt, system), **kwargs)

    def save(self, path: Path | None = None) -> Path:
        target = path or self.path
        if target is None:
            raise ValueError("no path given for the cassette")
        return self.cassette.save(target)

    # `with RecordingClient(real, "c.json") as client:` saves on exit, even
    # if the pipeline raised halfway, so a partial recording is not lost.
    def __enter__(self) -> RecordingClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.path is not None:
            self.save()


class ReplayClient:
    """Answer requests from a cassette. Never touches the network."""

    def __init__(self, cassette: Cassette | Path | str):
        if not isinstance(cassette, Cassette):
            cassette = Cassette.load(Path(cassette))
        self.cassette = cassette
        # Report the recorded model, so pipeline.json says which real model
        # produced these replies.
        self.model = cassette.model
        self.provider = f"replay:{cassette.provider}"
        # key -> queue of interactions, in recorded order. `defaultdict(deque)`
        # creates an empty queue the first time a key is seen.
        self._queues: dict[str, deque[Interaction]] = defaultdict(deque)
        for interaction in cassette.interactions:
            self._queues[interaction.key].append(interaction)
        self._last: dict[str, Interaction] = {}
        self.calls: list[str] = []  # keys in the order they were asked for

    def chat(
        self,
        messages: Iterable[Message],
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = list(messages)
        key = request_key(messages, json_mode=json_mode, max_tokens=max_tokens)
        self.calls.append(key)
        queue = self._queues.get(key)
        if queue:
            interaction = queue.popleft()
            self._last[key] = interaction
        elif key in self._last:
            interaction = self._last[key]  # queue used up: repeat the last one
        else:
            raise CassetteMiss(self._explain_miss(messages, key))
        return LLMResponse(
            text=interaction.text,
            model=self.model,
            provider=self.provider,
            prompt_tokens=interaction.prompt_tokens,
            completion_tokens=interaction.completion_tokens,
            latency_s=0.0,
        )

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any):
        return self.chat(as_messages(prompt, system), **kwargs)

    def unused(self) -> list[Interaction]:
        """Recorded interactions that replay never asked for.

        After a full run this should be empty. Leftovers mean the code now
        makes fewer calls than when the cassette was recorded.
        """
        return [i for queue in self._queues.values() for i in queue]

    def _explain_miss(self, messages: list[Message], key: str) -> str:
        """Point at the closest recorded request and where it differs."""
        lines = [
            f"no recorded reply for request {key} "
            f"({len(self.cassette.interactions)} interactions on the cassette).",
            "A prompt or setting changed since recording: re-record with "
            "`make record-fixtures`.",
        ]
        wanted = _flatten(messages)
        best, best_ratio = None, -1.0
        for interaction in self.cassette.interactions:
            # SequenceMatcher.ratio() is 1.0 for identical text, 0.0 for
            # nothing in common. quick_ratio() is a cheap upper bound, used
            # to skip candidates that cannot beat the best so far.
            matcher = difflib.SequenceMatcher(
                None, wanted, _flatten(interaction.messages)
            )
            if matcher.quick_ratio() <= best_ratio:
                continue
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best, best_ratio = interaction, ratio
        if best is not None:
            recorded = _flatten(best.messages)
            # Index of the first differing character. strict=False: the two
            # texts usually differ in length, and zip stops at the shorter.
            pairs = zip(wanted, recorded, strict=False)
            at = next(
                (i for i, (a, b) in enumerate(pairs) if a != b),
                min(len(wanted), len(recorded)),
            )
            lines.append(
                f"Closest recording: {best.key} ({best_ratio:.0%} similar), "
                f"first difference at character {at}:"
            )
            lines.append(f"  sent:     {wanted[max(0, at - 40) : at + 60]!r}")
            lines.append(f"  recorded: {recorded[max(0, at - 40) : at + 60]!r}")
        return "\n".join(lines)


def _flatten(messages: Sequence[Message]) -> str:
    return "\n".join(f"[{m.role}] {m.content}" for m in messages)
