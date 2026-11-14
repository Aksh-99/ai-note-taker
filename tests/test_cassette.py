"""Tests for record/replay of LLM calls (src/notetaker/cassette.py)."""

from __future__ import annotations

import json

import pytest

from notetaker.cassette import (
    Cassette,
    CassetteFormatError,
    CassetteMiss,
    RecordingClient,
    ReplayClient,
    request_key,
)
from notetaker.llm import FakeLLMClient, LLMError, Message

SYSTEM = Message("system", "You are terse.")


def ask(text: str) -> list[Message]:
    return [SYSTEM, Message("user", text)]


def record(replies: list[str], prompts: list[str], **kwargs) -> RecordingClient:
    recorder = RecordingClient(FakeLLMClient(replies=replies, model="m1"))
    for prompt in prompts:
        recorder.chat(ask(prompt), **kwargs)
    return recorder


# ---- keys -----------------------------------------------------------------


def test_key_is_stable_and_sensitive_to_every_detail():
    key = request_key(ask("hi"), json_mode=True, max_tokens=10)
    assert key == request_key(ask("hi"), json_mode=True, max_tokens=10)
    assert len(key) == 16
    # Any change in text, role order, JSON mode or token limit is a new key.
    assert key != request_key(ask("hi!"), json_mode=True, max_tokens=10)
    assert key != request_key(ask("hi"), json_mode=False, max_tokens=10)
    assert key != request_key(ask("hi"), json_mode=True, max_tokens=11)
    assert key != request_key(list(reversed(ask("hi"))), json_mode=True, max_tokens=10)


# ---- recording --------------------------------------------------------------


def test_recording_passes_replies_through_and_stores_them():
    recorder = record(["a", "b"], ["one", "two"], json_mode=True, max_tokens=50)
    assert recorder.model == "m1" and recorder.provider == "fake"
    first, second = recorder.cassette.interactions
    assert (first.text, second.text) == ("a", "b")
    assert first.messages == ask("one")
    assert first.json_mode is True and first.max_tokens == 50
    assert first.key == request_key(ask("one"), json_mode=True, max_tokens=50)
    # The wrapped client still saw the calls.
    assert len(recorder.inner.calls) == 2


def test_save_and_load_round_trip(tmp_path):
    recorder = record(["réponse: café"], ["question"])
    recorder.cassette.note = "unit test"
    path = recorder.save(tmp_path / "sub" / "c.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["version"] == 1 and raw["model"] == "m1"
    # ensure_ascii=False keeps accented text readable in the file.
    assert "café" in path.read_text(encoding="utf-8")
    loaded = Cassette.load(path)
    assert loaded.note == "unit test"
    assert loaded.interactions[0].text == "réponse: café"
    assert loaded.interactions[0].messages == ask("question")


def test_with_block_saves_even_when_the_run_crashes(tmp_path):
    path = tmp_path / "c.json"
    recorder = RecordingClient(FakeLLMClient(replies=["x"]), path)
    with pytest.raises(RuntimeError), recorder as client:
        client.complete("hello", system="sys")
        raise RuntimeError("pipeline blew up")
    assert len(Cassette.load(path).interactions) == 1


def test_save_needs_a_path():
    with pytest.raises(ValueError):
        record(["x"], ["y"]).save()


# ---- replay -----------------------------------------------------------------


def test_replay_answers_by_content_not_by_order():
    cassette = record(["A", "B"], ["first", "second"]).cassette
    replay = ReplayClient(cassette)
    # Asked in the opposite order, each prompt still gets its own reply.
    assert replay.chat(ask("second")).text == "B"
    assert replay.chat(ask("first")).text == "A"
    assert replay.unused() == []
    assert replay.provider == "replay:fake" and replay.model == "m1"


def test_repeated_requests_replay_in_recorded_order_then_repeat_last():
    # The same prompt sent twice got two different replies while recording.
    cassette = record(["v1", "v2"], ["same", "same"]).cassette
    replay = ReplayClient(cassette)
    assert [replay.chat(ask("same")).text for _ in range(3)] == ["v1", "v2", "v2"]


def test_replay_keeps_token_counts_and_supports_complete():
    recorder = RecordingClient(FakeLLMClient(replies=["ok"]))
    recorder.complete("p", system="You are terse.")
    recorder.cassette.interactions[0].prompt_tokens = 7
    reply = ReplayClient(recorder.cassette).complete("p", system="You are terse.")
    assert reply.text == "ok" and reply.prompt_tokens == 7


def test_settings_are_part_of_the_match():
    cassette = record(["x"], ["p"], json_mode=True).cassette
    with pytest.raises(CassetteMiss):
        ReplayClient(cassette).chat(ask("p"), json_mode=False)


def test_unused_reports_calls_that_never_happened():
    replay = ReplayClient(record(["a", "b"], ["one", "two"]).cassette)
    replay.chat(ask("one"))
    assert [i.text for i in replay.unused()] == ["b"]


def test_miss_explains_the_closest_recording():
    cassette = record(["x"], ["Summarize part 1 of the lecture."]).cassette
    with pytest.raises(CassetteMiss) as caught:
        ReplayClient(cassette).chat(ask("Summarise part 1 of the lecture."))
    message = str(caught.value)
    assert "make record-fixtures" in message
    assert cassette.interactions[0].key in message
    assert "Summarise" in message and "Summarize" in message


def test_miss_is_not_an_llm_error():
    # The pipeline swallows LLMError per stage; a stale cassette must not be
    # mistaken for a model failure.
    assert not issubclass(CassetteMiss, LLMError)


# ---- file validation ----------------------------------------------------------


def test_load_rejects_hand_edited_prompts(tmp_path):
    path = record(["x"], ["original"]).save(tmp_path / "c.json")
    raw = json.loads(path.read_text())
    raw["interactions"][0]["request"]["messages"][1]["content"] = "edited"
    path.write_text(json.dumps(raw))
    with pytest.raises(CassetteFormatError, match="edited"):
        Cassette.load(path)


def test_load_accepts_hand_edited_replies(tmp_path):
    # Only the request is hashed, so tweaking a reply to test an edge case
    # is allowed.
    path = record(["x"], ["q"]).save(tmp_path / "c.json")
    raw = json.loads(path.read_text())
    raw["interactions"][0]["response"]["text"] = "y"
    path.write_text(json.dumps(raw))
    assert ReplayClient(path).chat(ask("q")).text == "y"


@pytest.mark.parametrize("content", ["not json", "[]", '{"version": 99}'])
def test_load_rejects_unknown_files(tmp_path, content):
    path = tmp_path / "c.json"
    path.write_text(content)
    with pytest.raises(CassetteFormatError):
        Cassette.load(path)
