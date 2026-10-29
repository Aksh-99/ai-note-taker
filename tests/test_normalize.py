"""Tests for text normalization (src/notetaker/normalize.py).

The key property: two texts that SAY the same thing must normalize to the
same string, so WER only counts real recognition mistakes.
"""

import json

import pytest

from notetaker.cli import main
from notetaker.normalize import (
    expand_numbers,
    load_text,
    normalize_text,
    normalize_words,
    number_to_words,
    ordinal_to_words,
    strip_caption_markup,
)

# --- numbers ----------------------------------------------------------------


# pytest.mark.parametrize runs the same test once per (input, expected) pair,
# and reports each pair separately if it fails.
@pytest.mark.parametrize(
    ("digits", "words"),
    [
        ("0", "zero"),
        ("7", "seven"),
        ("13", "thirteen"),
        ("20", "twenty"),
        ("42", "forty two"),
        ("100", "one hundred"),
        ("105", "one hundred five"),
        ("999", "nine hundred ninety nine"),
        ("1000", "one thousand"),
        ("1024", "one thousand twenty four"),
        ("2026", "two thousand twenty six"),
        ("1000000", "one million"),
        ("3000500", "three million five hundred"),
        ("1000000000000", "one trillion"),
        ("007", "seven"),
    ],
)
def test_number_to_words(digits, words):
    assert number_to_words(digits) == words


def test_very_long_numbers_are_read_digit_by_digit():
    assert number_to_words("1234567890123456") == (
        "one two three four five six seven eight nine zero one two three four five six"
    )


def test_number_to_words_rejects_non_digits():
    with pytest.raises(ValueError):
        number_to_words("12a")


@pytest.mark.parametrize(
    ("digits", "words"),
    [
        ("1", "first"),
        ("2", "second"),
        ("3", "third"),
        ("4", "fourth"),
        ("5", "fifth"),
        ("8", "eighth"),
        ("9", "ninth"),
        ("12", "twelfth"),
        ("20", "twentieth"),
        ("21", "twenty first"),
        ("100", "one hundredth"),
    ],
)
def test_ordinal_to_words(digits, words):
    assert ordinal_to_words(digits) == words


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,024 points", "one thousand twenty four points"),
        ("10,000,000", "ten million"),
        ("the 2nd row", "the second row"),
        ("50% done", "fifty percent done"),
        ("$5", "five dollars"),
        ("pi is 3.14", "pi is three point one four"),
        (".5", "point five"),
        ("mp3", "mp three"),
        ("3d plot", "three d plot"),
        # A list like "1,2,3" is not a thousands separator.
        ("1,2,3", "one , two , three"),
    ],
)
def test_expand_numbers(text, expected):
    # expand_numbers pads with spaces; compare after collapsing them.
    assert " ".join(expand_numbers(text).split()) == expected


# --- caption markup ---------------------------------------------------------


def test_sound_tags_are_removed():
    text = "so [LAUGHTER] that works (APPLAUSE) [MUSIC PLAYING]"
    assert strip_caption_markup(text).split() == ["so", "that", "works"]


def test_lowercase_parentheses_are_kept_because_they_are_math():
    assert normalize_text("take f(x) and g(y)") == "take f x and g y"


def test_single_capital_in_parentheses_is_kept():
    # "(A)" is usually a spoken answer option or matrix name.
    assert normalize_text("option (A)") == "option a"


def test_speaker_labels_are_removed():
    text = "PROFESSOR: Good morning.\nGILBERT STRANG: Let's start.\n>> AUDIENCE: Yes."
    assert normalize_text(text) == "good morning lets start yes"


def test_a_normal_sentence_with_a_colon_is_kept():
    assert normalize_text("Remember this: rank matters") == "remember this rank matters"


# --- the full normalizer ----------------------------------------------------


def test_example_from_module_docstring():
    reference = "So, there are 3 eigenvalues."
    whisper = "so there are three eigenvalues"
    assert normalize_text(reference) == normalize_text(whisper)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Don’t worry.", "don't worry"),
        ("eigen-value", "eigen value"),
        ("OK, so", "okay so"),
        ("x²", "x squared"),
        ("a + b = c", "a plus b equals c"),
        ("Um, the, uh, matrix", "the matrix"),
        ("３ apples", "three apples"),  # full-width digit
        ("PROFESSOR: The 1st step.", "the first step"),
        ("it's 100%", "its one hundred percent"),
    ],
)
def test_style_variants_normalize_to_the_same_text(a, b):
    assert normalize_text(a) == normalize_text(b)


def test_output_is_lowercase_words_and_single_spaces():
    out = normalize_text("  Hello,\tWORLD!!  How's   it\n going?  ")
    assert out == "hello world hows it going"


def test_empty_and_punctuation_only_text():
    assert normalize_text("") == ""
    assert normalize_text("... !!! [NOISE]") == ""


def test_options_turn_steps_off():
    assert normalize_text("Um, 3 cats", numbers_to_words=False) == "3 cats"
    assert normalize_text("Um, 3 cats", remove_fillers=False) == "um three cats"


def test_normalize_words_returns_a_list():
    assert normalize_words("The 2 rows.") == ["the", "two", "rows"]


@pytest.mark.parametrize(
    "text",
    [
        "PROFESSOR: So the 2nd eigenvalue is 3.5, OK? [LAUGHTER]",
        "It’s x² + 50% of $10 - um - right?",
        "Section 3.2.1, the 100th time.",
    ],
)
def test_normalizing_twice_changes_nothing(text):
    # "Idempotent": normalize(normalize(x)) == normalize(x). If this failed,
    # the scorer's result would depend on whether a text was cleaned already.
    once = normalize_text(text)
    assert normalize_text(once) == once


# --- loading files and the CLI ----------------------------------------------


def test_load_text_reads_all_three_formats(tmp_path):
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({"segments": [{"text": "a"}, {"text": "b"}]}))
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps([{"start": 0, "end": 1, "text": "c"}]))
    plain = tmp_path / "reference.txt"
    plain.write_text("d\ne\n")

    assert load_text(transcript) == "a\nb"
    assert load_text(reference) == "c"
    assert load_text(plain) == "d\ne\n"


def test_load_text_rejects_unexpected_json(tmp_path):
    path = tmp_path / "weird.json"
    path.write_text(json.dumps({"segments": "not a list"}))
    with pytest.raises(ValueError):
        load_text(path)


def test_cli_normalize_prints_normalized_text(tmp_path, capsys):
    path = tmp_path / "reference.txt"
    path.write_text("PROFESSOR: Um, 2 rows.")
    assert main(["normalize", str(path)]) == 0
    assert capsys.readouterr().out.strip() == "two rows"

    assert main(["normalize", str(path), "--keep-digits", "--keep-fillers"]) == 0
    assert capsys.readouterr().out.strip() == "um 2 rows"


def test_cli_normalize_missing_file(tmp_path, capsys):
    assert main(["normalize", str(tmp_path / "nope.txt")]) == 1
    assert "Error" in capsys.readouterr().out
