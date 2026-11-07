"""Tests for key-concept extraction (src/notetaker/concepts.py).

No live LLM calls: `FakeLLMClient` returns scripted JSON replies, so we can
test validation, the repair loop, grounding, merging and the CLI offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from notetaker import cli
from notetaker.concepts import (
    ConceptList,
    ConceptsError,
    concepts_json_schema,
    contains_phrase,
    extract_chunk_concepts,
    extract_concepts,
    load_concepts,
    merge_concepts,
    term_words,
    validate_concepts_reply,
    write_concepts,
)
from notetaker.llm import FakeLLMClient
from notetaker.notes import chunk_transcript
from notetaker.transcribe import Segment, Transcript

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "concepts.schema.json"


def make_transcript(spans: list[tuple[float, float, str]]) -> Transcript:
    segments = [
        Segment(id=i, start=s, end=e, text=t, avg_logprob=-0.2, no_speech_prob=0.01)
        for i, (s, e, t) in enumerate(spans)
    ]
    return Transcript(
        segments=segments,
        language="en",
        language_probability=1.0,
        duration=spans[-1][1] if spans else 0.0,
        model="small",
    )


# Pauses of 5 s split segments into separate paragraphs (break is 2 s).
LECTURE = make_transcript(
    [
        (0.0, 10.0, "Today we define an eigenvalue of a matrix A."),
        (15.0, 25.0, "The singular value decomposition, or SVD, factors any matrix."),
        (30.0, 40.0, "Eigenvalues tell us how a matrix stretches vectors."),
        (45.0, 55.0, "We will use the SVD again for low rank approximation."),
    ]
)


def concept(term: str, definition: str = "a thing the lecturer defines", **kw):
    return {"term": term, "definition": definition, **kw}


def reply(*concepts: dict) -> str:
    return json.dumps({"concepts": list(concepts)})


# --- validation -------------------------------------------------------------


def test_valid_reply_parses():
    parsed = validate_concepts_reply(
        reply(concept("eigenvalue", "a scalar", aliases="lambda"))
    )
    assert parsed.concepts[0].term == "eigenvalue"
    assert parsed.concepts[0].aliases == ["lambda"]  # single string -> list


def test_empty_list_is_valid():
    assert validate_concepts_reply(reply()).concepts == []


def test_bare_list_and_flat_dict_accepted():
    assert validate_concepts_reply(json.dumps([concept("SVD")])).concepts[0].term == (
        "SVD"
    )
    flat = validate_concepts_reply(json.dumps({"SVD": "a factorization"}))
    assert flat.concepts[0].definition == "a factorization"


def test_code_fence_and_stray_quotes_cleaned():
    text = "```json\n" + reply(concept("**SVD**", "  a factorization ")) + "\n```"
    parsed = validate_concepts_reply(text)
    assert parsed.concepts[0].term == "SVD"
    assert parsed.concepts[0].definition == "a factorization"


@pytest.mark.parametrize(
    "text",
    [
        "not json at all",
        json.dumps({"concepts": [{"term": "SVD"}]}),  # no definition
        json.dumps({"concepts": [{"term": "", "definition": "x"}]}),
        reply(concept("this term is far too long to be a real concept name")),
        json.dumps({"other": 3}),
    ],
)
def test_invalid_replies_raise_value_error(text):
    with pytest.raises(ValueError):
        validate_concepts_reply(text)


# --- matching helpers -------------------------------------------------------


def test_term_words_normalizes_case_punctuation_and_plurals():
    assert term_words("Back-Propagation") == ("back", "propagation")
    assert term_words("Eigenvalues") == term_words("eigenvalue")
    assert term_words("probabilities") == ("probability",)
    # Words ending in ss/us/is keep their s.
    assert term_words("loss basis bonus") == ("loss", "basis", "bonus")


def test_contains_phrase_needs_consecutive_words():
    text = term_words("the singular value decomposition of A")
    assert contains_phrase(text, term_words("singular value"))
    assert not contains_phrase(text, term_words("value singular"))
    assert not contains_phrase(text, ())


# --- one chunk --------------------------------------------------------------


def test_ungrounded_terms_are_dropped():
    chunk = chunk_transcript(LECTURE)[0]
    client = FakeLLMClient(
        replies=[reply(concept("eigenvalue"), concept("Fourier transform"))]
    )
    grounded, dropped, _ = extract_chunk_concepts(client, chunk, 1)
    assert [c.term for c in grounded] == ["eigenvalue"]
    assert dropped == ["Fourier transform"]


def test_alias_can_ground_a_term():
    chunk = chunk_transcript(LECTURE)[0]
    client = FakeLLMClient(replies=[reply(concept("singular values", aliases=["SVD"]))])
    grounded, dropped, _ = extract_chunk_concepts(client, chunk, 1)
    assert len(grounded) == 1 and dropped == []


def test_repair_loop_recovers_and_sends_error_back():
    chunk = chunk_transcript(LECTURE)[0]
    client = FakeLLMClient(replies=["oops", reply(concept("SVD"))])
    grounded, _, responses = extract_chunk_concepts(client, chunk, 1)
    assert len(responses) == 2 and grounded[0].term == "SVD"
    repair_prompt = client.calls[1][-1].content
    assert "could not be used" in repair_prompt


def test_gives_up_after_max_repairs():
    chunk = chunk_transcript(LECTURE)[0]
    client = FakeLLMClient(replies=["bad"])
    with pytest.raises(ConceptsError, match="2 attempt"):
        extract_chunk_concepts(client, chunk, 1, max_repairs=1)
    with pytest.raises(ValueError):
        extract_chunk_concepts(client, chunk, 1, max_repairs=-1)


def test_prompt_contains_timestamped_transcript():
    chunk = chunk_transcript(LECTURE)[0]
    client = FakeLLMClient(replies=[reply()])
    extract_chunk_concepts(client, chunk, 3)
    prompt = client.calls[0][-1].content
    assert "part 1 of 3" in prompt
    assert "[00:00:15] The singular value decomposition" in prompt


# --- merging ----------------------------------------------------------------


def test_merge_by_alias_keeps_first_wording():
    from notetaker.concepts import ConceptSchema

    first = ConceptSchema(
        term="singular value decomposition",
        definition="factors any matrix into rotations and a scaling",
        aliases=["SVD"],
    )
    later = ConceptSchema(term="SVD", definition="used for low rank approximation")
    repeat = ConceptSchema(
        term="SVDs", definition="factors any matrix into rotations and a scaling"
    )
    merged = merge_concepts([(0, [first]), (1, [later]), (2, [repeat])])
    assert len(merged) == 1
    m = merged[0]
    assert m.term == "singular value decomposition"
    assert m.aliases == ["SVD"]
    assert m.chunks == [0, 1, 2]
    # The repeat is a near copy of the main definition, so it is not kept.
    assert m.other_definitions == ["used for low rank approximation"]


def test_distinct_terms_stay_separate():
    from notetaker.concepts import ConceptSchema

    a = ConceptSchema(term="eigenvalue", definition="x")
    b = ConceptSchema(term="eigenvector", definition="y")
    assert len(merge_concepts([(0, [a, b])])) == 2


# --- full pipeline ----------------------------------------------------------


def test_extract_concepts_end_to_end():
    # max_words=20 puts two paragraphs in each chunk -> two LLM calls.
    client = FakeLLMClient(
        replies=[
            reply(
                concept(
                    "singular value decomposition", "factors a matrix", aliases=["SVD"]
                ),
                concept("eigenvalue", "how much a vector is stretched"),
            ),
            reply(concept("SVD", "used for low rank approximation"), concept("qubit")),
        ]
    )
    result = extract_concepts(LECTURE, client, title="L1", max_words=20)
    assert result.chunks == 2 and result.llm_calls == 2
    assert result.dropped_ungrounded == ["qubit"]
    terms = [c.term for c in result.concepts]
    assert terms == ["eigenvalue", "singular value decomposition"]  # by time
    eig, svd = result.concepts
    # Mentions come from the transcript, not the model: plural matches too.
    assert eig.mentions == [0.0, 30.0] and eig.first_mention == 0.0
    assert svd.mentions == [15.0, 45.0]
    assert svd.chunks == [0, 1]


def test_empty_transcript_makes_no_calls():
    client = FakeLLMClient()
    result = extract_concepts(make_transcript([]), client)
    assert result.concepts == [] and client.calls == []


# --- files and schema -------------------------------------------------------


def test_write_load_round_trip_and_markdown(tmp_path):
    client = FakeLLMClient(replies=[reply(concept("SVD", "a factorization"))])
    result = extract_concepts(LECTURE, client, title="Glossary")
    json_path, md_path = write_concepts(result, tmp_path)
    assert load_concepts(json_path) == result
    md = md_path.read_text()
    assert md.startswith("# Glossary")
    assert "- **SVD** `[00:00:15]`: a factorization" in md


def test_load_rejects_invalid_file(tmp_path):
    bad = tmp_path / "concepts.json"
    bad.write_text(json.dumps({"title": "x", "concepts": [{"term": ""}]}))
    with pytest.raises(ValueError, match="not a valid concepts file"):
        load_concepts(bad)


def test_committed_schema_matches_code():
    # If this fails, regenerate it with:
    #   notetaker concepts --schema > schemas/concepts.schema.json
    assert json.loads(SCHEMA_PATH.read_text()) == concepts_json_schema()


def test_schema_describes_required_fields():
    schema = concepts_json_schema()
    assert {"title", "concepts", "model"} <= set(schema["required"])
    assert schema["$defs"]["Concept"]["properties"]["term"]["minLength"] == 1


def test_markdown_with_no_concepts():
    empty = ConceptList(
        title="T", concepts=[], model="m", provider="p", chunks=0, llm_calls=0
    )
    assert "_No concepts found._" in empty.to_markdown()


# --- CLI --------------------------------------------------------------------


def run_cli(argv: list[str], client=None) -> int:
    args = cli.build_parser().parse_args(["concepts", *argv])
    return cli.run_concepts(args, client=client)


def test_cli_writes_files_next_to_transcript(tmp_path, capsys):
    folder = tmp_path / "lecture01"
    folder.mkdir()
    LECTURE.save_json(folder / "transcript.json")
    client = FakeLLMClient(replies=[reply(concept("eigenvalue"), concept("qubit"))])
    assert run_cli([str(folder / "transcript.json")], client=client) == 0
    out = capsys.readouterr().out
    assert "Dropped 1 term(s) not found in the transcript: qubit" in out
    data = load_concepts(folder / "concepts.json")
    assert data.title == "Key concepts: lecture01"
    assert (folder / "concepts.md").exists()


def test_cli_schema_flag_prints_schema(capsys):
    assert run_cli(["--schema"]) == 0
    assert json.loads(capsys.readouterr().out) == concepts_json_schema()


def test_cli_errors(tmp_path, capsys):
    assert run_cli([]) == 1
    assert run_cli([str(tmp_path / "missing.json")]) == 1
    junk = tmp_path / "junk.json"
    junk.write_text("{}")
    assert run_cli([str(junk)]) == 1
    assert run_cli([str(junk), "--chunk-words", "0"]) == 1
    out = capsys.readouterr().out
    assert "give a transcript.json" in out and "no such file" in out


def test_cli_reports_unusable_model_output(tmp_path, capsys):
    path = tmp_path / "transcript.json"
    LECTURE.save_json(path)
    assert run_cli([str(path)], client=FakeLLMClient(replies=["nope"])) == 1
    assert "no valid concepts" in capsys.readouterr().out
