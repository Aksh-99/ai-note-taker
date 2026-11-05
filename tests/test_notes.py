"""Tests for the notes summarizer (src/notetaker/notes.py).

No live LLM calls: every test uses `FakeLLMClient`, which returns scripted
replies and records each prompt, so we can check both what we send and how
we handle good, bad and half-broken replies.
"""

from __future__ import annotations

import json

import pytest

from notetaker import cli
from notetaker.llm import FakeLLMClient, LLMResponse
from notetaker.notes import (
    Notes,
    NoteSection,
    NotesError,
    TranscriptChunk,
    chunk_transcript,
    generate_notes,
    parse_timestamp,
    summarize_chunk,
    validate_reply,
    write_notes,
)
from notetaker.transcribe import Segment, Transcript


def make_transcript(spans: list[tuple[float, float, str]]) -> Transcript:
    segments = [
        Segment(id=i, start=s, end=e, text=t, avg_logprob=-0.2, no_speech_prob=0.01)
        for i, (s, e, t) in enumerate(spans)
    ]
    end = spans[-1][1] if spans else 0.0
    return Transcript(
        segments=segments,
        language="en",
        language_probability=1.0,
        duration=end,
        model="small",
    )


def words(n: int, word: str = "eigen") -> str:
    return " ".join([word] * n)


# Three paragraphs: segments separated by 5 s pauses (> the 2 s break), so
# group_paragraphs makes one paragraph per segment. 10, 10 and 5 words.
THREE_PARAS = make_transcript(
    [(0.0, 10.0, words(10)), (15.0, 25.0, words(10)), (30.0, 40.0, words(5))]
)


def reply(*sections: dict) -> str:
    return json.dumps({"sections": list(sections)})


GOOD_SECTION = {
    "heading": "Eigenvalues",
    "start": "00:00:15",
    "summary": "What an eigenvalue is.",
    "key_points": ["Av = lambda v defines an eigenpair.", "  ", "Trace is a sum."],
}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunking_packs_paragraphs_under_budget():
    chunks = chunk_transcript(THREE_PARAS, max_words=20)
    # 10 + 10 fits in 20; adding 5 more would exceed it.
    assert [c.word_count for c in chunks] == [20, 5]
    assert [c.index for c in chunks] == [0, 1]
    assert (chunks[0].start, chunks[0].end) == (0.0, 25.0)
    assert (chunks[1].start, chunks[1].end) == (30.0, 40.0)


def test_chunking_large_budget_gives_one_chunk():
    chunks = chunk_transcript(THREE_PARAS, max_words=10_000)
    assert len(chunks) == 1
    assert chunks[0].word_count == 25


def test_oversized_paragraph_becomes_its_own_chunk():
    chunks = chunk_transcript(THREE_PARAS, max_words=3)
    assert [c.word_count for c in chunks] == [10, 10, 5]


def test_empty_transcript_gives_no_chunks():
    assert chunk_transcript(make_transcript([])) == []


def test_chunking_rejects_bad_budget():
    with pytest.raises(ValueError):
        chunk_transcript(THREE_PARAS, max_words=0)


def test_prompt_text_has_timestamps():
    chunk = chunk_transcript(THREE_PARAS, max_words=20)[0]
    lines = chunk.to_prompt_text().splitlines()
    assert lines[0].startswith("[00:00:00] eigen")
    assert lines[1].startswith("[00:00:15] eigen")


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("00:12:41", 761.0),
        ("12:41", 761.0),
        ("[01:00:00]", 3600.0),
        (" 0:05:09 ", 309.0),
        ("12:61", None),
        ("noon", None),
        ("", None),
    ],
)
def test_parse_timestamp(text, seconds):
    assert parse_timestamp(text) == seconds


def test_validate_reply_cleans_points():
    parsed = validate_reply(reply(GOOD_SECTION))
    assert parsed.sections[0].key_points == [
        "Av = lambda v defines an eigenpair.",
        "Trace is a sum.",
    ]


def test_validate_reply_accepts_bare_list_and_code_fence():
    text = "```json\n" + json.dumps([GOOD_SECTION]) + "\n```"
    assert validate_reply(text).sections[0].heading == "Eigenvalues"


def test_validate_reply_accepts_single_string_key_point():
    section = {**GOOD_SECTION, "key_points": "Only one point."}
    assert validate_reply(reply(section)).sections[0].key_points == ["Only one point."]


@pytest.mark.parametrize(
    "text",
    [
        "not json at all",
        json.dumps({"sections": []}),
        reply({"heading": "No points", "key_points": []}),
        reply({"heading": "", "key_points": ["x"]}),
        json.dumps({"notes": "wrong key"}),
    ],
)
def test_validate_reply_rejects_bad_shapes(text):
    with pytest.raises(ValueError):
        validate_reply(text)


# ---------------------------------------------------------------------------
# Summarizing one chunk
# ---------------------------------------------------------------------------


def test_summarize_chunk_builds_prompt_and_sections():
    client = FakeLLMClient(replies=[reply(GOOD_SECTION)])
    chunk = chunk_transcript(THREE_PARAS, max_words=20)[0]
    sections, responses = summarize_chunk(client, chunk, total=2)

    assert len(responses) == 1
    system, user = client.calls[0]
    assert system.role == "system" and "JSON only" in system.content
    assert "part 1 of 2" in user.content
    assert "[00:00:15]" in user.content

    (section,) = sections
    assert section.heading == "Eigenvalues"
    assert section.start == 15.0
    assert section.chunk == 0


@pytest.mark.parametrize("start", ["99:00:00", "garbage", "", "00:00:35"])
def test_bad_or_out_of_range_start_falls_back_to_chunk_start(start):
    # Chunk 0 covers 0-25 s, so 35 s is outside it and is not trusted.
    client = FakeLLMClient(replies=[reply({**GOOD_SECTION, "start": start})])
    chunk = chunk_transcript(THREE_PARAS, max_words=20)[0]
    sections, _ = summarize_chunk(client, chunk, total=1)
    assert sections[0].start == chunk.start


def test_repair_loop_recovers_from_bad_json():
    client = FakeLLMClient(replies=["oops, not json", reply(GOOD_SECTION)])
    chunk = chunk_transcript(THREE_PARAS, max_words=20)[0]
    sections, responses = summarize_chunk(client, chunk, total=1, max_repairs=1)

    assert len(responses) == 2 and len(sections) == 1
    # The second call carries the bad reply and an explanation of the error.
    second = client.calls[1]
    assert [m.role for m in second] == ["system", "user", "assistant", "user"]
    assert second[2].content == "oops, not json"
    assert "could not be used" in second[3].content


def test_gives_up_after_max_repairs():
    client = FakeLLMClient(replies=["bad"])
    chunk = chunk_transcript(THREE_PARAS, max_words=20)[0]
    with pytest.raises(NotesError, match="2 attempt"):
        summarize_chunk(client, chunk, total=1, max_repairs=1)
    assert len(client.calls) == 2


def test_negative_max_repairs_rejected():
    chunk = chunk_transcript(THREE_PARAS, max_words=20)[0]
    with pytest.raises(ValueError):
        summarize_chunk(FakeLLMClient(), chunk, total=1, max_repairs=-1)


# ---------------------------------------------------------------------------
# Whole pipeline and outputs
# ---------------------------------------------------------------------------


class CountingClient(FakeLLMClient):
    """Fake client that also reports token counts, to test bookkeeping."""

    def chat(self, messages, **kwargs):
        base = super().chat(messages, **kwargs)
        return LLMResponse(
            text=base.text,
            model=self.model,
            provider=self.provider,
            prompt_tokens=100,
            completion_tokens=20,
            latency_s=0.5,
        )


def test_generate_notes_end_to_end():
    later = {**GOOD_SECTION, "heading": "Determinants", "start": "00:00:30"}
    client = CountingClient(replies=[reply(GOOD_SECTION), reply(later)])
    seen = []
    notes = generate_notes(
        THREE_PARAS,
        client,
        title="Lecture 1",
        max_words=20,
        on_chunk=lambda i, total, chunk: seen.append((i, total)),
    )
    assert seen == [(0, 2), (1, 2)]
    assert [s.heading for s in notes.sections] == ["Eigenvalues", "Determinants"]
    assert notes.chunks == 2 and notes.llm_calls == 2
    assert (notes.prompt_tokens, notes.completion_tokens) == (200, 40)
    assert notes.latency_s == pytest.approx(1.0)
    assert notes.provider == "fake"


def test_sections_sorted_by_time_within_chunk():
    early = {**GOOD_SECTION, "heading": "Early", "start": "00:00:00"}
    client = FakeLLMClient(replies=[reply(GOOD_SECTION, early)])
    notes = generate_notes(THREE_PARAS, client, max_words=10_000)
    assert [s.heading for s in notes.sections] == ["Early", "Eigenvalues"]


def test_markdown_rendering():
    notes = Notes(
        title="Lecture 1",
        sections=[NoteSection("Eigenvalues", 761.0, "Summary.", ["Point."], 0)],
        model="m",
        provider="p",
        chunks=1,
    )
    md = notes.to_markdown()
    assert md.startswith("# Lecture 1")
    assert "## Eigenvalues `[00:12:41]`" in md
    assert "- Point." in md


def test_empty_notes_markdown():
    notes = Notes(title="T", sections=[], model="m", provider="p", chunks=0)
    assert "No notes" in notes.to_markdown()


def test_write_and_reload_round_trip(tmp_path):
    client = FakeLLMClient(replies=[reply(GOOD_SECTION)])
    notes = generate_notes(THREE_PARAS, client, max_words=10_000)
    json_path, md_path = write_notes(notes, tmp_path / "out")
    assert md_path.read_text(encoding="utf-8").startswith("# Lecture notes")
    reloaded = Notes.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
    assert reloaded == notes


def test_empty_transcript_makes_no_llm_calls():
    client = FakeLLMClient()
    notes = generate_notes(make_transcript([]), client)
    assert notes.sections == [] and client.calls == []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_cli(argv, client):
    args = cli.build_parser().parse_args(argv)
    return cli.run_notes(args, client=client)


def test_cli_writes_notes_next_to_transcript(tmp_path, capsys):
    lecture = tmp_path / "lecture01"
    THREE_PARAS.save_json(lecture / "transcript.json")
    client = FakeLLMClient(replies=[reply(GOOD_SECTION)])

    code = run_cli(["notes", str(lecture / "transcript.json")], client)

    assert code == 0
    md = (lecture / "notes.md").read_text(encoding="utf-8")
    assert md.startswith("# Notes: lecture01")
    assert (lecture / "notes.json").exists()
    assert "chunk 1/1" in capsys.readouterr().out


def test_cli_missing_file(tmp_path, capsys):
    code = run_cli(["notes", str(tmp_path / "nope.json")], FakeLLMClient())
    assert code == 1
    assert "no such file" in capsys.readouterr().out


def test_cli_not_a_transcript(tmp_path, capsys):
    path = tmp_path / "transcript.json"
    path.write_text('{"hello": 1}', encoding="utf-8")
    assert run_cli(["notes", str(path)], FakeLLMClient()) == 1
    assert "not a transcript" in capsys.readouterr().out


def test_cli_reports_unusable_model_output(tmp_path, capsys):
    THREE_PARAS.save_json(tmp_path / "transcript.json")
    client = FakeLLMClient(replies=["never json"])
    code = run_cli(
        ["notes", str(tmp_path / "transcript.json"), "--max-repairs", "0"], client
    )
    assert code == 1
    assert "no valid notes" in capsys.readouterr().out


def test_cli_rejects_bad_chunk_words(tmp_path):
    THREE_PARAS.save_json(tmp_path / "transcript.json")
    argv = ["notes", str(tmp_path / "transcript.json"), "--chunk-words", "0"]
    assert run_cli(argv, FakeLLMClient()) == 1


def test_chunk_dataclass_word_count():
    chunk = TranscriptChunk(0, 0.0, 1.0, [(0.0, "a b c"), (0.5, "d")])
    assert chunk.word_count == 4
