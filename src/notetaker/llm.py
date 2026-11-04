"""One small interface for every LLM this project is allowed to use.

Phase 3 turns transcripts into notes, flashcards and quizzes. All of that is
"send some messages to a chat model, get text back", but the free options
speak slightly different dialects:

    * Ollama (local, free, no key):   POST {base}/api/chat
    * Groq (hosted, free tier):       POST https://api.groq.com/openai/v1/chat/completions
    * Hugging Face Inference Providers (hosted, free monthly credits):
                                      POST https://router.huggingface.co/v1/chat/completions

Groq and Hugging Face both copy the OpenAI "chat completions" format, so they
share one class here. Ollama has its own format and gets its own class.

The pattern: an abstract base class
    `LLMClient` defines the public method everybody calls, `chat()`. It also
    owns the shared logic (retries, timing, error messages). Each subclass only
    fills in two small "hooks": how to build the HTTP request and how to read
    the reply. The summarizer in Step 3.2 will only ever see `LLMClient`, so
    switching from a local model to a hosted one is a config change, not a
    code change. This is the same idea as a function that takes any norm
    ||.|| as an argument instead of hard-coding the Euclidean one.

Testing without the network
    The HTTP call itself is a separate function (a "transport") passed into
    the client. Real use gets `urllib_transport`, which really posts JSON.
    Tests pass a fake transport that returns a canned dict, so CI never makes
    a live LLM call. `FakeLLMClient` goes one step further for later steps:
    it returns scripted replies and records every prompt it was sent.

No new dependencies: the HTTP code uses only Python's standard library.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from notetaker.config import LLMProvider, Settings, get_settings

Role = Literal["system", "user", "assistant"]

# The free model each provider uses when NOTETAKER_LLM_MODEL is empty. All
# three are the same open model family (Llama 3.1 8B Instruct), which keeps
# results roughly comparable when you switch provider.
DEFAULT_MODELS: dict[str, str] = {
    "ollama": "llama3.1:8b",
    "groq": "llama-3.1-8b-instant",
    "huggingface": "meta-llama/Llama-3.1-8B-Instruct",
}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
HF_BASE_URL = "https://router.huggingface.co/v1"

# HTTP status codes worth retrying: 408 request timeout, 429 "too many
# requests" (free tiers rate-limit a lot), and 5xx server-side hiccups.
# A 400 or 401 means our request or key is wrong, so retrying cannot help.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


# `frozen=True` makes instances immutable (like a tuple with named fields),
# so a message list can be reused across calls without surprises.
@dataclass(frozen=True)
class Message:
    """One turn of a chat: who said it (`role`) and what they said."""

    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LLMResponse:
    """What came back, plus bookkeeping that later evals and cost logs use."""

    text: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    attempts: int = 1

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens

    def json(self) -> Any:
        """Parse `text` as JSON (for requests made with `json_mode=True`).

        Models sometimes wrap JSON in a ```json ... ``` markdown fence even
        when asked not to, so a surrounding fence is removed first.
        """
        return json.loads(strip_code_fence(self.text))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Base class for every LLM failure, so callers can catch just one type."""


class LLMConfigError(LLMError):
    """Bad setup we can detect before sending anything (e.g. missing key)."""


class LLMHTTPError(LLMError):
    """The server answered with an error status, or could not be reached.

    `status` is None for network failures (connection refused, timeout).
    `retry_after` is the server's own "wait this many seconds" hint, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        # No status = network problem (server busy, timeout), worth a retry.
        return self.status is None or self.status in RETRYABLE_STATUS


class LLMResponseError(LLMError):
    """The server answered 200 OK but the body was not what we expected."""


# ---------------------------------------------------------------------------
# Transport: the one place that touches the network
# ---------------------------------------------------------------------------

# Type of a transport: (url, headers, json payload, timeout) -> parsed JSON.
Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


def urllib_transport(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """POST `payload` as JSON to `url` and return the decoded JSON reply.

    Every failure becomes an `LLMHTTPError` so the retry loop has a single
    error type to look at.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            raw = reply.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        # HTTPError must be caught before URLError because it is a subclass.
        detail = error.read().decode("utf-8", errors="replace")
        raise LLMHTTPError(
            f"HTTP {error.code} from {url}: {_short(detail)}",
            status=error.code,
            retry_after=_parse_retry_after(error.headers.get("Retry-After")),
        ) from error
    except urllib.error.URLError as error:
        raise LLMHTTPError(f"could not reach {url}: {error.reason}") from error
    except TimeoutError as error:
        message = f"timed out after {timeout:.0f}s waiting for {url}"
        raise LLMHTTPError(message) from error
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        message = f"reply from {url} was not JSON: {_short(raw)}"
        raise LLMResponseError(message) from error


def _parse_retry_after(value: str | None) -> float | None:
    """Read a Retry-After header given in seconds (the form APIs use)."""
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # The rarer HTTP-date form: ignore it and use our backoff.


def _short(text: str, limit: int = 300) -> str:
    """Trim long error bodies so a message stays one readable line."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def strip_code_fence(text: str) -> str:
    """Remove a ```lang ... ``` fence around the whole text, if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    lines = lines[1:]  # Drop the opening ``` line (it may carry a language tag).
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def as_messages(
    prompt: str | Sequence[Message], system: str | None = None
) -> list[Message]:
    """Accept either a plain prompt string or a ready list of messages."""
    # list(prompt) copies, so inserting the system message below never edits
    # the caller's own list.
    messages = [Message("user", prompt)] if isinstance(prompt, str) else list(prompt)
    if system is not None:
        messages.insert(0, Message("system", system))
    if not messages:
        raise ValueError("need at least one message")
    return messages


# ---------------------------------------------------------------------------
# The shared client
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Base class: retries, timing and errors live here, once.

    `ABC` ("abstract base class") means this class cannot be used directly;
    it only exists to be subclassed. Methods marked `@abstractmethod` are the
    holes each subclass must fill. Python refuses to create a subclass object
    that forgot one, so a half-written provider fails immediately.
    """

    provider: str = "abstract"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.2,
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        # "Dependency injection": the network call and the sleep function are
        # arguments instead of being hard-wired. Tests swap in fakes so they
        # run instantly and offline.
        self._transport = transport or urllib_transport
        self._sleep = sleep

    # ---- hooks each provider fills in ----------------------------------
    @abstractmethod
    def _request(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Return (url, headers, payload) for one chat request."""

    @abstractmethod
    def _parse(self, data: dict[str, Any]) -> tuple[str, int | None, int | None]:
        """Return (text, prompt_tokens, completion_tokens) from a reply."""

    def _explain(self, error: LLMHTTPError) -> str:
        """Provider-specific hint appended to a final error. Optional hook."""
        return ""

    # ---- public API -----------------------------------------------------
    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send a conversation and return the model's reply.

        `json_mode=True` asks the provider to return valid JSON only (both
        Ollama and the OpenAI-style APIs support this). Your prompt should
        still say "answer in JSON" and describe the shape you want.
        """
        messages = list(messages)
        if not messages:
            raise ValueError("need at least one message")
        temp = self.temperature if temperature is None else temperature
        url, headers, payload = self._request(messages, temp, max_tokens, json_mode)

        start = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                data = self._transport(url, headers, payload, self.timeout)
                break
            except LLMHTTPError as error:
                if not error.retryable or attempt > self.max_retries:
                    hint = self._explain(error)
                    message = f"{self.provider} request failed: {error}"
                    raise LLMHTTPError(
                        message + (f"\nHint: {hint}" if hint else ""),
                        status=error.status,
                        retry_after=error.retry_after,
                    ) from error
                self._sleep(self.backoff_delay(attempt, error.retry_after))

        try:
            text, prompt_tokens, completion_tokens = self._parse(data)
        except (KeyError, IndexError, TypeError) as error:
            raise LLMResponseError(
                f"unexpected {self.provider} reply shape: {_short(json.dumps(data))}"
            ) from error
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=time.perf_counter() - start,
            attempts=attempt,
        )

    def complete(
        self, prompt: str, *, system: str | None = None, **kwargs: Any
    ) -> LLMResponse:
        """Shortcut for the common case: one prompt, optional system message."""
        return self.chat(as_messages(prompt, system), **kwargs)

    def backoff_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait before retry number `attempt` (1, 2, 3, ...).

        Exponential backoff: base * 2^(attempt-1), i.e. 1s, 2s, 4s, 8s, ...,
        capped at `backoff_max`. Doubling means a short blip costs little,
        while a real outage is not hammered with requests. If the server sent
        Retry-After we trust it (still capped), since it knows its own limit.
        """
        if retry_after is not None:
            return min(retry_after, self.backoff_max)
        return min(self.backoff_base * 2 ** (attempt - 1), self.backoff_max)

    def __repr__(self) -> str:
        # Deliberately leaves out any API key.
        return f"{type(self).__name__}(model={self.model!r})"


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class OllamaClient(LLMClient):
    """Local models through Ollama (https://ollama.com). Free, offline, no key.

    Setup: install Ollama, then `ollama pull llama3.1:8b`. The server listens
    on http://localhost:11434 by default.
    """

    provider = "ollama"

    def __init__(
        self, model: str, *, base_url: str = "http://localhost:11434", **kwargs: Any
    ) -> None:
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")

    def _request(self, messages, temperature, max_tokens, json_mode):
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens  # Ollama's name for max_tokens.
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            # Without this Ollama streams one JSON object per token.
            "stream": False,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"
        return f"{self.base_url}/api/chat", {}, payload

    def _parse(self, data):
        return (
            data["message"]["content"],
            data.get("prompt_eval_count"),
            data.get("eval_count"),
        )

    def _explain(self, error):
        if error.status is None:
            return (
                f"is Ollama running at {self.base_url}? Start it with "
                "`ollama serve` (or open the Ollama app)."
            )
        if error.status == 404:
            return f"model not downloaded yet; run `ollama pull {self.model}`."
        return ""


class OpenAICompatibleClient(LLMClient):
    """Any server that speaks the OpenAI chat-completions format.

    Groq and Hugging Face both do, so they are thin subclasses below that
    only set the address and the name of the key.
    """

    provider = "openai-compatible"
    key_name = "API key"

    def __init__(self, model: str, *, base_url: str, api_key: str, **kwargs: Any):
        super().__init__(model, **kwargs)
        if not api_key:
            raise LLMConfigError(f"{self.provider} needs {self.key_name}; none was set")
        self.base_url = base_url.rstrip("/")
        # Leading underscore = "private, do not print". It is never included
        # in repr() or error messages.
        self._api_key = api_key

    def _request(self, messages, temperature, max_tokens, json_mode):
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        return f"{self.base_url}/chat/completions", headers, payload

    def _parse(self, data):
        usage = data.get("usage") or {}
        return (
            data["choices"][0]["message"]["content"],
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )

    def _explain(self, error):
        if error.status == 401:
            return f"{self.key_name} was rejected; check it is copied correctly."
        if error.status == 429:
            return "free-tier rate limit hit; wait a minute or use Ollama locally."
        if error.status == 404:
            return f"model {self.model!r} not found on {self.provider}."
        return ""


class GroqClient(OpenAICompatibleClient):
    """Groq's free tier (https://console.groq.com). Very fast, rate-limited."""

    provider = "groq"
    key_name = "GROQ_API_KEY"

    def __init__(
        self, model: str, *, api_key: str, base_url: str = GROQ_BASE_URL, **kw: Any
    ) -> None:
        super().__init__(model, base_url=base_url, api_key=api_key, **kw)


class HuggingFaceClient(OpenAICompatibleClient):
    """Hugging Face Inference Providers (free monthly credits with an HF token)."""

    provider = "huggingface"
    key_name = "HF_TOKEN"

    def __init__(
        self, model: str, *, api_key: str, base_url: str = HF_BASE_URL, **kw: Any
    ) -> None:
        super().__init__(model, base_url=base_url, api_key=api_key, **kw)


# ---------------------------------------------------------------------------
# A fake for tests
# ---------------------------------------------------------------------------


@dataclass
class FakeLLMClient:
    """Stand-in client for tests: scripted replies, no network.

    It has the same `chat()` / `complete()` methods as a real client, so code
    written against `LLMClient` accepts it ("duck typing": if it has the
    methods we call, it works). Replies are handed out in order; when they
    run out, the last one repeats. Every call is stored in `calls` so a test
    can check exactly which prompt the summarizer built.
    """

    replies: list[str] = field(default_factory=lambda: ["fake reply"])
    model: str = "fake-model"
    provider: str = "fake"
    calls: list[list[Message]] = field(default_factory=list)

    def chat(self, messages: Iterable[Message], **kwargs: Any) -> LLMResponse:
        messages = list(messages)
        self.calls.append(messages)
        index = min(len(self.calls), len(self.replies)) - 1
        text = self.replies[index]
        return LLMResponse(text=text, model=self.model, provider=self.provider)

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any):
        return self.chat(as_messages(prompt, system), **kwargs)


# ---------------------------------------------------------------------------
# Factory: settings -> the right client
# ---------------------------------------------------------------------------


def make_client(
    settings: Settings | None = None,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
    transport: Transport | None = None,
) -> LLMClient:
    """Build the client described by the settings (or by the overrides).

    A "factory" function hides the `if provider == ...` choice in one place,
    so the rest of the code just asks for "the configured LLM".
    """
    settings = settings or get_settings()
    provider = provider or settings.llm_provider
    model = model or settings.llm_model or DEFAULT_MODELS[provider]
    common: dict[str, Any] = {
        "temperature": settings.llm_temperature,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
        "transport": transport,
    }
    if provider == "ollama":
        return OllamaClient(model, base_url=settings.ollama_base_url, **common)
    if provider == "groq":
        key = settings.groq_api_key
        if key is None:
            raise LLMConfigError(
                "llm_provider=groq needs a free key: create one at "
                "https://console.groq.com/keys and set GROQ_API_KEY in .env"
            )
        return GroqClient(model, api_key=key.get_secret_value(), **common)
    if provider == "huggingface":
        key = settings.hf_token
        if key is None:
            raise LLMConfigError(
                "llm_provider=huggingface needs a free token: create one at "
                "https://huggingface.co/settings/tokens and set HF_TOKEN in .env"
            )
        return HuggingFaceClient(model, api_key=key.get_secret_value(), **common)
    raise LLMConfigError(f"unknown LLM provider {provider!r}")
