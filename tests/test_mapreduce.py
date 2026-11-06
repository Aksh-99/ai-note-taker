"""Tests for the reduce step (src/notetaker/mapreduce.py).

As everywhere in Phase 3, no live LLM calls: `FakeLLMClient` returns
scripted replies, and a small "smart" fake below answers reduce prompts by
reading the section ids out of the prompt, so the hierarchical path can be
tested on any number of sections.
"""

from __future__ import annotations

import json
import re

import pytest

from notetaker import cli
from notetaker.llm import FakeLLMClient, LLMError, LLMResponse
from notetaker.mapreduce import (
    assign_ids,
    dedupe_points,
    merge_group,
    pack_batches,
    reduce_notes,
    reduce_once,
    reduce_sections,
    render_item,
    render_words,
    topic_range,
    validate_reduce_reply,
)
from notetaker.notes import Notes, NoteSection, NotesError, generate_notes
from notetaker.transcribe import Segment, Transcript


def section(heading: str, start: float, points=None, chunk=0, sources=None):
    return NoteSection(
        heading=heading,
        start=start,
        summary=f"About {heading.lower()}.",
        key_points=points or [f"{heading} point one.", f"{heading} point two."],
        chunk=chunk,
        sources=sources if sources is not None else [],
    )


FOUR = [
    section("Eigenvalues", 0.0, chunk=0),
    section("Eigenvalues continued", 400.0, chunk=1),
    section("Determinants", 800.0, chunk=1),
    section("Determinant rules", 1200.0, chunk=2),
]


def topics_reply(*topics, overview="", takeaways=None) -> str:
    data = {
        "topics": [
            {"heading": h, "summary": f"{h}.", "sections": ids} for h, ids in topics
        ]
    }
    if overview:
        data["overview"] = overview
    if takeaways is not None:
        data["takeaways"] = takeaways
    return json.dumps(data)


def notes_from(sections) -> Notes:
    return Notes(
        title="L",
        sections=list(sections),
        model="fake-model",
        provider="fake",
        chunks=3,
        prompt_tokens=10,
        completion_tokens=5,
        llm_calls=3,
        latency_s=1.0,
    )


# ---------------------------------------------------------------------------
# Validating replies
# ---------------------------------------------------------------------------


def test_validate_accepts_good_reply_and_id_spellings():
    text = json.dumps(
        {
            "overview": "  A lecture.  ",
            "takeaways": ["One.", "  ", 3],
            "topics": [
                {"heading": "A", "sections": ["S1", "[2]"]},
                {"heading": "B", "sections": 3},
            ],
        }
    )
    parsed = validate_reduce_reply(text, 3)
    assert [t.sections for t in parsed.topics] == [[1, 2], [3]]
    assert parsed.overview == "A lecture."
    assert parsed.takeaways == ["One."]


def test_validate_accepts_bare_list_and_code_fence():
    text = '```json\n[{"heading": "A", "sections": [1, 2]}]\n```'
    assert validate_reduce_reply(text, 2).topics[0].sections == [1, 2]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", None),
        ('{"topics": []}', "wrong JSON shape"),
        ('{"topics": [{"heading": "", "sections": [1]}]}', "wrong JSON shape"),
        ('{"topics": [{"heading": "A", "sections": ["x"]}]}', "wrong JSON shape"),
        ('{"topics": [{"heading": "A", "sections": [5]}]}', "does not exist"),
        ('{"topics": [{"heading": "A", "sections": [0]}]}', "does not exist"),
        (
            '{"topics": [{"heading": "A", "sections": [1]},'
            ' {"heading": "B", "sections": [1, 2]}]}',
            "more than one topic",
        ),
    ],
)
def test_validate_rejects_bad_replies(text, message):
    with pytest.raises(ValueError, match=message):
        validate_reduce_reply(text, 2)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_render_item_shows_id_time_and_first_points():
    s = section("Eigenvalues", 761.0, points=["p1", "p2", "p3", "p4"])
    line = render_item(3, s)
    assert line.startswith("[3] (00:12:41) Eigenvalues: About eigenvalues.")
    assert "p3" in line and "p4" not in line  # only the first 3 points


def test_dedupe_points_drops_near_duplicates_only():
    points = [
        "The determinant is zero for singular matrices.",
        "the determinant is zero for singular matrices!",  # same point
        "The determinant is zero for all singular matrices.",  # 1 word added
        "Eigenvalues solve det(A - lambda I) = 0.",
        "   ",
    ]
    assert dedupe_points(points) == [points[0], points[3]]


def test_dedupe_keeps_points_that_differ_in_one_key_word():
    points = ["The determinant is zero.", "The determinant is nonzero."]
    assert dedupe_points(points) == points
    assert dedupe_points(["Step one.", "Step two."]) == ["Step one.", "Step two."]


def test_assign_ids_fills_gaps_from_previous_neighbour():
    # ids 1 and 4 missing: 1 joins 0's group, 4 joins 3's group.
    assert assign_ids([[0], [2, 3]], 5) == [[0, 1], [2, 3, 4]]
    # id 0 missing and nothing before it: joins the group after it.
    assert assign_ids([[1], [2]], 3) == [[0, 1], [2]]


def test_assign_ids_needs_at_least_one_id():
    with pytest.raises(ValueError):
        assign_ids([], 3)


def test_merge_group_copies_facts_from_members():
    a = section("A", 300.0, points=["Shared point.", "Only A."], chunk=2, sources=[1])
    b = section("B", 100.0, points=["shared point", "Only B."], chunk=1, sources=[0])
    topic = merge_group("New heading", "New summary.", [a, b])
    assert topic.start == 100.0  # earliest member, never model-written
    assert topic.chunk == 1
    assert topic.sources == [0, 1]
    # B comes first in time; its "shared point" wins, A's copy is dropped.
    assert topic.key_points == ["shared point", "Only B.", "Only A."]


def test_pack_batches_respects_budget_and_min_two():
    items = [section(f"T{i}", i * 60.0) for i in range(7)]
    per_item = len(render_item(1, items[0]).split())
    batches = pack_batches(items, per_item * 3)
    assert [len(b) for b in batches] == [3, 4]  # lonely last item folded in
    assert [s for b in batches for s in b] == items  # order kept, nothing lost
    # A tiny budget still puts two sections in each batch.
    assert all(len(b) >= 2 for b in pack_batches(items, 1))


@pytest.mark.parametrize(
    ("n", "final", "expected"),
    [(2, True, (1, 1)), (12, True, (3, 6)), (40, True, (3, 8)), (10, False, (2, 5))],
)
def test_topic_range(n, final, expected):
    assert topic_range(n, final) == expected


# ---------------------------------------------------------------------------
# One reduce call
# ---------------------------------------------------------------------------


def test_reduce_once_final_builds_topics_and_prompt():
    client = FakeLLMClient(
        replies=[
            topics_reply(
                ("Determinants", [3, 4]),
                ("Eigenvalues", [1, 2]),  # listed out of order on purpose
                overview="Linear algebra.",
                takeaways=["Eigenvalues matter."],
            )
        ]
    )
    items = [s for s in (replace_sources(s, [i]) for i, s in enumerate(FOUR))]
    result = reduce_once(client, items, final=True)
    assert [t.heading for t in result.topics] == ["Eigenvalues", "Determinants"]
    assert [t.start for t in result.topics] == [0.0, 800.0]
    assert [t.sources for t in result.topics] == [[0, 1], [2, 3]]
    assert result.overview == "Linear algebra."
    assert result.takeaways == ["Eigenvalues matter."]
    prompt = client.calls[0][1].content
    assert "[4] (00:20:00) Determinant rules" in prompt
    assert '"overview"' in prompt


def test_reduce_once_intermediate_ignores_overview():
    client = FakeLLMClient(replies=[topics_reply(("All", [1, 2]), overview="x")])
    result = reduce_once(client, FOUR[:2], final=False)
    assert result.overview == "" and result.takeaways == []
    assert '"overview"' not in client.calls[0][1].content


def test_reduce_once_attaches_forgotten_sections():
    # Section 2 is in no topic: it joins section 1's topic.
    client = FakeLLMClient(replies=[topics_reply(("A", [1]), ("B", [3, 4]))])
    result = reduce_once(client, FOUR, final=True)
    assert [len(t.key_points) for t in result.topics] == [4, 4]
    assert result.topics[0].heading == "A"


def test_reduce_once_repairs_bad_ids():
    client = FakeLLMClient(
        replies=[topics_reply(("A", [1, 9])), topics_reply(("A", [1, 2]))]
    )
    result = reduce_once(client, FOUR[:2], final=True)
    assert len(result.responses) == 2
    repair = client.calls[1][-1].content
    assert "section id 9 does not exist" in repair


def test_reduce_once_gives_up():
    client = FakeLLMClient(replies=["nope"])
    with pytest.raises(NotesError, match="2 attempt"):
        reduce_once(client, FOUR, final=True, max_repairs=1)
    with pytest.raises(ValueError):
        reduce_once(client, FOUR, final=True, max_repairs=-1)


# ---------------------------------------------------------------------------
# Hierarchical reduce
# ---------------------------------------------------------------------------


class PairingClient(FakeLLMClient):
    """Answers any reduce prompt by pairing up consecutive section ids.

    It reads the ids `[1] ... [n]` from the last user message, so it works
    for every batch and level without scripting each reply by hand.
    """

    def chat(self, messages, **kwargs):
        messages = list(messages)
        self.calls.append(messages)
        ids = [int(i) for i in re.findall(r"^\[(\d+)\]", messages[-1].content, re.M)]
        pairs = [ids[i : i + 2] for i in range(0, len(ids), 2)]
        data = {
            "overview": "Whole lecture.",
            "takeaways": ["Big idea."],
            "topics": [
                {"heading": f"Topic {p[0]}", "summary": "s", "sections": p}
                for p in pairs
            ],
        }
        return LLMResponse(
            text=json.dumps(data),
            model=self.model,
            provider=self.provider,
            prompt_tokens=7,
            completion_tokens=3,
            latency_s=0.25,
        )


def replace_sources(s: NoteSection, sources: list[int]) -> NoteSection:
    from dataclasses import replace

    return replace(s, sources=sources)


def many_sections(n: int) -> list[NoteSection]:
    return [
        replace_sources(section(f"T{i}", i * 60.0, chunk=i // 3), [i]) for i in range(n)
    ]


def test_reduce_sections_single_level_when_it_fits():
    client = PairingClient()
    result, levels = reduce_sections(client, many_sections(6), max_words=10_000)
    assert levels == 0 and len(client.calls) == 1
    assert len(result.topics) == 3


def test_reduce_sections_goes_hierarchical_and_keeps_coverage():
    items = many_sections(24)
    budget = render_words(items) // 4  # forces several batches
    client = PairingClient()
    result, levels = reduce_sections(client, items, max_words=budget)
    assert levels >= 1
    assert len(client.calls) > 1
    # Every map section is in exactly one final topic, in time order.
    all_sources = [i for t in result.topics for i in t.sources]
    assert sorted(all_sources) == list(range(24))
    assert len(all_sources) == len(set(all_sources))
    starts = [t.start for t in result.topics]
    assert starts == sorted(starts)
    # Only the final call asks for the overview.
    asked = ['"overview"' in c[-1].content for c in client.calls]
    assert asked == [False] * (len(asked) - 1) + [True]
    assert result.overview == "Whole lecture."


def test_reduce_sections_stops_when_model_merges_nothing():
    class Identity(FakeLLMClient):
        def chat(self, messages, **kwargs):
            messages = list(messages)
            self.calls.append(messages)
            n = len(re.findall(r"^\[\d+\]", messages[-1].content, re.M))
            topics = [{"heading": f"H{i}", "sections": [i]} for i in range(1, n + 1)]
            return LLMResponse(
                text=json.dumps({"topics": topics}), model="m", provider="p"
            )

    client = Identity()
    result, levels = reduce_sections(client, many_sections(8), max_words=1)
    assert levels == 1  # gave up after one level that did not shrink
    assert len(result.topics) == 8


def test_reduce_sections_rejects_bad_budget():
    with pytest.raises(ValueError):
        reduce_sections(PairingClient(), many_sections(3), max_words=0)


# ---------------------------------------------------------------------------
# reduce_notes and the full pipeline
# ---------------------------------------------------------------------------


def test_reduce_notes_merges_and_adds_bookkeeping():
    reduced = reduce_notes(notes_from(FOUR), PairingClient())
    assert [t.heading for t in reduced.sections] == ["Topic 1", "Topic 3"]
    assert reduced.overview == "Whole lecture."
    assert reduced.takeaways == ["Big idea."]
    assert [s.sources for s in reduced.detail_sections] == [[0], [1], [2], [3]]
    assert reduced.reduce_calls == 1 and reduced.reduce_error is None
    assert reduced.llm_calls == 4
    assert (reduced.prompt_tokens, reduced.completion_tokens) == (17, 8)
    assert reduced.latency_s == pytest.approx(1.25)


def test_reduce_notes_skips_single_section():
    notes = notes_from(FOUR[:1])
    client = FakeLLMClient()
    assert reduce_notes(notes, client) is notes
    assert client.calls == []


def test_reduce_notes_falls_back_on_failure():
    notes = notes_from(FOUR)
    reduced = reduce_notes(notes, FakeLLMClient(replies=["bad"]), max_repairs=0)
    assert reduced.sections == notes.sections
    assert "reduce" in reduced.reduce_error

    class Down(FakeLLMClient):
        def chat(self, messages, **kwargs):
            raise LLMError("rate limited")

    assert reduce_notes(notes, Down()).reduce_error == "rate limited"


def test_markdown_shows_overview_and_takeaways():
    reduced = reduce_notes(notes_from(FOUR), PairingClient())
    md = reduced.to_markdown()
    assert "**Overview.** Whole lecture." in md
    assert md.index("**Overview.**") < md.index("## Topic 1")
    assert md.rstrip().endswith("- Big idea.")


def test_round_trip_and_old_json_still_loads():
    reduced = reduce_notes(notes_from(FOUR), PairingClient())
    assert Notes.from_dict(json.loads(json.dumps(reduced.to_dict()))) == reduced
    # A notes.json from Step 3.2 has none of the new fields.
    old = notes_from(FOUR).to_dict()
    for key in (
        "overview",
        "takeaways",
        "detail_sections",
        "reduce_calls",
        "reduce_levels",
        "reduce_error",
    ):
        old.pop(key)
    for s in old["sections"]:
        s.pop("sources")
    assert Notes.from_dict(old).sections[0].sources == []


def make_transcript(n: int) -> Transcript:
    # Segments 5 s apart (> the 2 s paragraph break): one paragraph each.
    segments = [
        Segment(
            id=i,
            start=i * 15.0,
            end=i * 15.0 + 10,
            text="word " * 10,
            avg_logprob=-0.2,
            no_speech_prob=0.01,
        )
        for i in range(n)
    ]
    return Transcript(
        segments=segments,
        language="en",
        language_probability=1.0,
        duration=n * 15.0,
        model="small",
    )


class MapThenPair(PairingClient):
    """Map prompts get one section per chunk; reduce prompts get pairs."""

    def chat(self, messages, **kwargs):
        messages = list(messages)
        if "Group the sections" in messages[-1].content or (
            len(messages) > 1 and "Group the sections" in messages[1].content
        ):
            return super().chat(messages, **kwargs)
        self.calls.append(messages)
        ts = re.search(r"\[(\d\d:\d\d:\d\d)\]", messages[-1].content).group(1)
        text = json.dumps(
            {
                "sections": [
                    {"heading": f"At {ts}", "start": ts, "key_points": [f"Point {ts}."]}
                ]
            }
        )
        return LLMResponse(text=text, model=self.model, provider=self.provider)


def test_generate_notes_with_reduce():
    notes = generate_notes(make_transcript(4), MapThenPair(), max_words=10, reduce=True)
    assert len(notes.detail_sections) == 4 and len(notes.sections) == 2
    assert notes.llm_calls == 5  # 4 map + 1 reduce
    assert notes.sections[1].key_points == ["Point 00:00:30.", "Point 00:00:45."]


def test_generate_notes_reduce_off_by_default():
    notes = generate_notes(make_transcript(4), MapThenPair(), max_words=10)
    assert notes.detail_sections == [] and len(notes.sections) == 4


def run_cli(argv, client):
    return cli.run_notes(cli.build_parser().parse_args(argv), client=client)


def test_cli_reduces_by_default(tmp_path, capsys):
    path = tmp_path / "lec" / "transcript.json"
    make_transcript(4).save_json(path)
    assert run_cli(["notes", str(path), "--chunk-words", "10"], MapThenPair()) == 0
    out = capsys.readouterr().out
    assert "Merged 4 chunk section(s) into 2 topic(s)" in out
    data = json.loads((tmp_path / "lec" / "notes.json").read_text(encoding="utf-8"))
    assert data["overview"] == "Whole lecture."


def test_cli_no_reduce_and_warning(tmp_path, capsys):
    path = tmp_path / "transcript.json"
    make_transcript(4).save_json(path)
    argv = ["notes", str(path), "--chunk-words", "10", "--no-reduce"]
    assert run_cli(argv, MapThenPair()) == 0
    assert "Merged" not in capsys.readouterr().out

    class BadReduce(MapThenPair):
        def chat(self, messages, **kwargs):
            messages = list(messages)
            if "Group the sections" in messages[1].content:
                self.calls.append(messages)
                return LLMResponse(text="oops", model="m", provider="p")
            return super().chat(messages, **kwargs)

    assert run_cli(["notes", str(path), "--chunk-words", "10"], BadReduce()) == 0
    assert "Warning: could not merge" in capsys.readouterr().out


def test_cli_rejects_bad_reduce_words(tmp_path):
    path = tmp_path / "transcript.json"
    make_transcript(2).save_json(path)
    assert run_cli(["notes", str(path), "--reduce-words", "0"], MapThenPair()) == 1
