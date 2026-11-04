"""Tests for the LLM client layer (src/notetaker/llm.py).

No test here touches the network. Each client gets a fake "transport" (the
function that would normally do the HTTP POST), which records what it was
asked to send and returns a canned reply in the provider's real format.
That checks both directions: we build the request each API expects, and we
read the reply shape each API returns.
"""

from __future__ import annotations

import json
import os

import pytest

from notetaker import cli
from notetaker.config import Settings, get_settings
from notetaker.llm import (
    DEFAULT_MODELS,
    FakeLLMClient,
    GroqClient,
    HuggingFaceClient,
    LLMConfigError,
    LLMHTTPError,
    LLMResponse,
    LLMResponseError,
    Message,
    OllamaClient,
    as_messages,
    make_client,
    strip_code_fence,
    urllib_transport,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Same isolation as test_config: no real env vars or .env leak in."""
    for name in list(os.environ):
        if name.startswith("NOTETAKER_") or name in {"GROQ_API_KEY", "HF_TOKEN"}:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class RecordingTransport:
    """Fake network: returns scripted replies or raises scripted errors."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, url, headers, payload, timeout):
        self.requests.append(
            {"url": url, "headers": headers, "payload": payload, "timeout": timeout}
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def ollama_reply(text="hi", prompt_tokens=7, completion_tokens=3):
    return {
        "model": "llama3.1:8b",
        "message": {"role": "assistant", "content": text},
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
    }


def openai_reply(text="hi", prompt_tokens=11, completion_tokens=4):
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def no_sleep(seconds):
    """Record nothing, wait for nothing: keeps retry tests instant."""


# ---- Ollama -------------------------------------------------------------


def test_ollama_builds_native_chat_request():
    transport = RecordingTransport(ollama_reply("A matrix scales a vector."))
    client = OllamaClient(
        "llama3.1:8b",
        base_url="http://localhost:11434/",
        temperature=0.3,
        timeout=9,
        transport=transport,
    )
    reply = client.complete("What is an eigenvector?", system="Be brief.")

    sent = transport.requests[0]
    assert sent["url"] == "http://localhost:11434/api/chat"  # trailing / removed
    assert sent["timeout"] == 9
    payload = sent["payload"]
    assert payload["model"] == "llama3.1:8b"
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0.3}
    assert "format" not in payload
    assert payload["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What is an eigenvector?"},
    ]
    assert reply.text == "A matrix scales a vector."
    assert (reply.prompt_tokens, reply.completion_tokens) == (7, 3)
    assert reply.total_tokens == 10
    assert reply.provider == "ollama"
    assert reply.attempts == 1


def test_ollama_json_mode_and_max_tokens():
    transport = RecordingTransport(ollama_reply('{"a": 1}'))
    client = OllamaClient("m", transport=transport)
    reply = client.complete("x", json_mode=True, max_tokens=50, temperature=0.0)
    payload = transport.requests[0]["payload"]
    assert payload["format"] == "json"
    assert payload["options"] == {"temperature": 0.0, "num_predict": 50}
    assert reply.json() == {"a": 1}


def test_ollama_connection_error_gets_a_hint():
    transport = RecordingTransport(LLMHTTPError("could not reach x: refused"))
    client = OllamaClient("m", max_retries=0, transport=transport)
    with pytest.raises(LLMHTTPError, match="ollama serve"):
        client.complete("x")


def test_ollama_missing_model_hint():
    transport = RecordingTransport(LLMHTTPError("HTTP 404", status=404))
    client = OllamaClient("qwen2.5:7b", transport=transport, sleep=no_sleep)
    with pytest.raises(LLMHTTPError, match="ollama pull qwen2.5:7b") as info:
        client.complete("x")
    assert info.value.status == 404
    assert len(transport.requests) == 1  # 404 is not retried


# ---- OpenAI-compatible (Groq, Hugging Face) -----------------------------


@pytest.mark.parametrize(
    ("cls", "url"),
    [
        (GroqClient, "https://api.groq.com/openai/v1/chat/completions"),
        (HuggingFaceClient, "https://router.huggingface.co/v1/chat/completions"),
    ],
)
def test_openai_style_request_and_reply(cls, url):
    transport = RecordingTransport(openai_reply("Gradient descent."))
    client = cls("some-model", api_key="secret-123", transport=transport)
    reply = client.chat(
        [Message("user", "Name an optimizer.")], max_tokens=20, json_mode=True
    )
    sent = transport.requests[0]
    assert sent["url"] == url
    assert sent["headers"] == {"Authorization": "Bearer secret-123"}
    assert sent["payload"] == {
        "model": "some-model",
        "messages": [{"role": "user", "content": "Name an optimizer."}],
        "temperature": 0.2,
        "max_tokens": 20,
        "response_format": {"type": "json_object"},
    }
    assert reply.text == "Gradient descent."
    assert (reply.prompt_tokens, reply.completion_tokens) == (11, 4)


def test_api_key_never_in_repr():
    client = GroqClient("m", api_key="super-secret")
    assert "super-secret" not in repr(client)


def test_openai_style_requires_key():
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        GroqClient("m", api_key="")


def test_missing_usage_gives_none_tokens():
    transport = RecordingTransport({"choices": [{"message": {"content": "ok"}}]})
    reply = GroqClient("m", api_key="k", transport=transport).complete("x")
    assert reply.text == "ok"
    assert reply.total_tokens is None


def test_unexpected_reply_shape_is_a_clear_error():
    transport = RecordingTransport({"error": "weird"})
    client = HuggingFaceClient("m", api_key="k", transport=transport)
    with pytest.raises(LLMResponseError, match="unexpected huggingface reply"):
        client.complete("x")


def test_bad_key_is_not_retried_and_explained():
    transport = RecordingTransport(LLMHTTPError("HTTP 401", status=401))
    client = GroqClient("m", api_key="k", transport=transport, sleep=no_sleep)
    with pytest.raises(LLMHTTPError, match="GROQ_API_KEY was rejected"):
        client.complete("x")
    assert len(transport.requests) == 1


# ---- Retries and backoff ------------------------------------------------


def test_retries_rate_limit_then_succeeds():
    waits = []
    transport = RecordingTransport(
        LLMHTTPError("HTTP 429", status=429),
        LLMHTTPError("HTTP 503", status=503),
        openai_reply("finally"),
    )
    client = GroqClient(
        "m", api_key="k", transport=transport, sleep=waits.append, max_retries=3
    )
    reply = client.complete("x")
    assert reply.text == "finally"
    assert reply.attempts == 3
    assert waits == [1.0, 2.0]  # exponential: 1s, then 2s


def test_retry_after_header_wins_but_is_capped():
    waits = []
    transport = RecordingTransport(
        LLMHTTPError("HTTP 429", status=429, retry_after=5.0),
        LLMHTTPError("HTTP 429", status=429, retry_after=999.0),
        openai_reply(),
    )
    client = GroqClient(
        "m", api_key="k", transport=transport, sleep=waits.append, backoff_max=30
    )
    client.complete("x")
    assert waits == [5.0, 30.0]


def test_gives_up_after_max_retries():
    waits = []
    errors = [LLMHTTPError("HTTP 500", status=500) for _ in range(3)]
    transport = RecordingTransport(*errors)
    client = OllamaClient("m", transport=transport, sleep=waits.append, max_retries=2)
    with pytest.raises(LLMHTTPError, match="ollama request failed") as info:
        client.complete("x")
    assert info.value.status == 500
    assert len(transport.requests) == 3  # 1 try + 2 retries
    assert len(waits) == 2


@pytest.mark.parametrize(
    ("attempt", "expected"), [(1, 1.0), (2, 2.0), (3, 4.0), (6, 30.0), (20, 30.0)]
)
def test_backoff_delay_doubles_and_caps(attempt, expected):
    client = OllamaClient("m", backoff_base=1.0, backoff_max=30.0)
    assert client.backoff_delay(attempt) == expected


def test_retryable_status_codes():
    assert LLMHTTPError("x").retryable  # network error
    assert LLMHTTPError("x", status=429).retryable
    assert LLMHTTPError("x", status=502).retryable
    assert not LLMHTTPError("x", status=400).retryable
    assert not LLMHTTPError("x", status=401).retryable


# ---- Helpers ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```\n', '{"a": 1}'),
        ('  ```json\n{"a": 1}', '{"a": 1}'),  # unclosed fence
    ],
)
def test_strip_code_fence(text, expected):
    assert strip_code_fence(text) == expected


def test_response_json_handles_fenced_reply():
    reply = LLMResponse(
        text='```json\n{"terms": ["svd"]}\n```', model="m", provider="p"
    )
    assert reply.json() == {"terms": ["svd"]}


def test_as_messages_does_not_mutate_input():
    original = [Message("user", "hi")]
    result = as_messages(original, system="sys")
    assert [m.role for m in result] == ["system", "user"]
    assert original == [Message("user", "hi")]


def test_empty_conversation_rejected():
    with pytest.raises(ValueError):
        OllamaClient("m").chat([])


def test_fake_client_scripts_replies_and_records_calls():
    fake = FakeLLMClient(replies=["one", "two"])
    assert fake.complete("a").text == "one"
    assert fake.complete("b", system="s").text == "two"
    assert fake.complete("c").text == "two"  # last reply repeats
    assert fake.calls[1] == [Message("system", "s"), Message("user", "b")]


def test_urllib_transport_unreachable_server_raises_llm_error():
    # Port 9 ("discard") on localhost is closed on normal machines and in CI,
    # so the connection is refused immediately. Still no external network.
    with pytest.raises(LLMHTTPError) as info:
        urllib_transport("http://127.0.0.1:9/api/chat", {}, {"x": 1}, timeout=2)
    assert info.value.status is None
    assert info.value.retryable


# ---- Factory ------------------------------------------------------------


def test_factory_defaults_to_local_ollama():
    client = make_client(Settings())
    assert isinstance(client, OllamaClient)
    assert client.model == DEFAULT_MODELS["ollama"]
    assert client.base_url == "http://localhost:11434"
    assert client.max_retries == 3


def test_factory_uses_provider_default_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    client = make_client(Settings(), provider="groq")
    assert isinstance(client, GroqClient)
    assert client.model == "llama-3.1-8b-instant"


def test_factory_respects_settings(monkeypatch):
    monkeypatch.setenv("NOTETAKER_LLM_PROVIDER", "huggingface")
    monkeypatch.setenv("NOTETAKER_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("NOTETAKER_LLM_TIMEOUT_SECONDS", "30")
    client = make_client(Settings())
    assert isinstance(client, HuggingFaceClient)
    assert client.model == "Qwen/Qwen2.5-7B-Instruct"
    assert client.timeout == 30.0


@pytest.mark.parametrize(
    ("provider", "needed"), [("groq", "GROQ_API_KEY"), ("huggingface", "HF_TOKEN")]
)
def test_factory_explains_missing_key(provider, needed):
    with pytest.raises(LLMConfigError, match=needed):
        make_client(Settings(), provider=provider)


def test_blank_model_setting_means_provider_default(monkeypatch):
    monkeypatch.setenv("NOTETAKER_LLM_MODEL", "")
    assert Settings().llm_model is None


# ---- CLI ----------------------------------------------------------------


def _cli_args(*argv):
    return cli.build_parser().parse_args(["llm", *argv])


def test_cli_llm_prints_reply_and_stats(capsys):
    transport = RecordingTransport(ollama_reply("Eigenvalues scale eigenvectors."))
    code = cli.run_llm(_cli_args("What is an eigenvalue?"), transport=transport)
    out = capsys.readouterr().out
    assert code == 0
    assert "Eigenvalues scale eigenvectors." in out
    assert "7 in / 3 out tokens" in out
    assert "llama3.1:8b" in out


def test_cli_llm_json_mode(capsys):
    transport = RecordingTransport(ollama_reply('```json\n{"term": "svd"}\n```'))
    code = cli.run_llm(_cli_args("x", "--json"), transport=transport)
    out = capsys.readouterr().out
    assert code == 0
    assert json.loads(out.split("\n[")[0]) == {"term": "svd"}


def test_cli_llm_bad_json_fails(capsys):
    transport = RecordingTransport(ollama_reply("not json at all"))
    code = cli.run_llm(_cli_args("x", "--json"), transport=transport)
    assert code == 1
    assert "not valid JSON" in capsys.readouterr().out


def test_cli_llm_missing_key_is_friendly(capsys):
    code = cli.main(["llm", "hi", "--provider", "groq"])
    assert code == 1
    assert "console.groq.com" in capsys.readouterr().out


# ---- Live check (skipped in CI) ----------------------------------------


@pytest.mark.slow
def test_live_ollama_round_trip():
    """Real call to a local Ollama. Run with: pytest -m slow -k live_ollama."""
    client = make_client(Settings(), provider="ollama")
    try:
        reply = client.complete("Reply with the single word: ready", max_tokens=5)
    except LLMHTTPError as error:
        pytest.skip(f"Ollama not available: {error}")
    assert reply.text.strip()
