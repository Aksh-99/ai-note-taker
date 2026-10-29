"""Text normalization so transcripts can be compared fairly.

Why this exists:
    Word error rate (WER, Step 2.3) counts how many words differ between our
    transcript and the human reference. Taken literally it punishes harmless
    differences in *writing style*, not mistakes in *hearing*:

        reference:  "So, there are 3 eigenvalues."
        whisper:    "so there are three eigenvalues"

    Literally that is 2 errors out of 5 words (40% WER), yet the model heard
    everything correctly. So before scoring, both texts go through the SAME
    normalizer, which maps every style variant to one canonical form:

        "so there are three eigenvalues"   (both sides)

    This is standard practice: OpenAI's Whisper paper uses a similar
    normalizer, and published WER numbers are always "after normalization".

What the normalizer does, in order (the order matters, see comments below):
    1. Caption clutter is removed: [LAUGHTER], (APPLAUSE), ">>" and speaker
       labels like "PROFESSOR:" or "GILBERT STRANG:". Nobody says these aloud.
    2. Unicode is tidied: curly quotes become straight, "x²" becomes
       "x squared", fancy digits and ligatures become plain ones (NFKC).
    3. Everything is lowercased.
    4. Numbers become words: "3" -> "three", "1,024" -> "one thousand twenty
       four", "2nd" -> "second", "50%" -> "fifty percent", "3.14" -> "three
       point one four", "$5" -> "five dollars".
    5. A few symbols are spoken: "+" plus, "=" equals, "&" and, "×" times.
    6. Apostrophes are dropped ("don't" -> "dont") and all other punctuation
       becomes a space, so "eigen-value" and "eigen value" match.
    7. Filler words (um, uh, hmm, ...) are removed, because caption writers
       usually leave them out while some models keep them.
    8. A few spelling variants are unified ("ok" -> "okay").

Known limits (worth knowing, and worth saying in an interview):
    * Numbers only go digits -> words. If the reference spells "nineteen
      ninety nine" and the model writes "1999", we produce "one thousand nine
      hundred ninety nine" and they still differ. Course numbers like "6.006"
      (said "six double-oh-six") have the same issue. Both are rare in
      lecture speech, and they affect every model equally.
    * English only. Other languages would need their own number words.

The whole module is pure Python with no dependencies, so it is fast, easy
to test, and behaves identically on every machine.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Numbers to words
# ---------------------------------------------------------------------------

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]  # fmt: skip
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
    "ninety",
]  # fmt: skip
# Each scale word is 1000 times the previous one.
_SCALES = ["", "thousand", "million", "billion", "trillion"]

# Integers with more digits than this are read digit by digit instead
# (a phone number or a long ID is never said as "quadrillions").
_MAX_DIGITS = 15


def _under_thousand(n: int) -> list[str]:
    """Words for 0 < n < 1000, e.g. 342 -> ["three", "hundred", "forty", "two"]."""
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words += [_ONES[hundreds], "hundred"]
    if rest >= 20:
        tens, ones = divmod(rest, 10)
        words.append(_TENS[tens])
        if ones:
            words.append(_ONES[ones])
    elif rest:
        words.append(_ONES[rest])
    return words


def number_to_words(digits: str) -> str:
    """Spell a string of digits as English words.

    >>> number_to_words("2024")
    'two thousand twenty four'

    How it works: split the number into groups of three digits from the
    right (2,024 -> [2, 24]); each group is spelled on its own and followed
    by its scale word (thousand, million, ...). No "and" and no hyphens,
    because punctuation is stripped later anyway and "and" is often not said.
    """
    if not digits.isdigit():
        raise ValueError(f"expected only digits, got {digits!r}")
    if len(digits) > _MAX_DIGITS:
        return " ".join(_ONES[int(d)] for d in digits)
    n = int(digits)
    if n == 0:
        return "zero"
    words: list[str] = []
    scale = 0
    while n:
        n, group = divmod(n, 1000)
        if group:
            part = _under_thousand(group)
            if _SCALES[scale]:
                part.append(_SCALES[scale])
            # We build from the lowest group upward, so prepend each new part.
            words = part + words
        scale += 1
    return " ".join(words)


# Irregular ordinals; everything else just gets "th" (with y -> ieth).
_ORDINAL_IRREGULAR = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}


def ordinal_to_words(digits: str) -> str:
    """Spell an ordinal: "21" -> "twenty first", "30" -> "thirtieth".

    Only the LAST word of the cardinal changes, so "100" -> "one hundredth".
    """
    words = number_to_words(digits).split()
    last = words[-1]
    if last in _ORDINAL_IRREGULAR:
        words[-1] = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def _digits_one_by_one(digits: str) -> str:
    """Read digits one at a time: "14" -> "one four" (as in 3.14)."""
    return " ".join(_ONES[int(d)] for d in digits)


# Regular expressions for the number forms we expand. Each one is compiled
# once at import time, which is faster than recompiling on every call.
#
# (?<![\d.]) and (?![\d]) are "lookarounds": they check the neighbouring
# character without consuming it, so "1,000" matches as a whole number but
# the "000" inside it is never matched on its own.
_THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d{3}\b)")
_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_DOLLARS = re.compile(r"\$\s?(\d+(?:\.\d+)?)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_DECIMAL = re.compile(r"(?<![\d.])(\d*)\.(\d+)(?![\d])")
_LETTER_DIGIT = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")
_INTEGER = re.compile(r"\d+")


def _spell_decimal(match: re.Match[str]) -> str:
    whole, frac = match.group(1), match.group(2)
    head = number_to_words(whole) + " " if whole else ""
    return f" {head}point {_digits_one_by_one(frac)} "


def expand_numbers(text: str) -> str:
    """Replace every number in (lowercased) text with words.

    The order of the substitutions matters: "1,000th" must lose its comma
    before ordinals are found, and "$3.50" must be handled before the plain
    decimal rule would turn the "3.50" into words without the "dollars".
    """
    text = _THOUSANDS_SEP.sub("", text)
    text = _ORDINAL.sub(lambda m: f" {ordinal_to_words(m.group(1))} ", text)
    # Rewrite "$5" as "5 dollars" and "50%" as "50 percent"; the number
    # itself is spelled by the decimal / integer rules below.
    text = _DOLLARS.sub(lambda m: f" {m.group(1)} dollars ", text)
    text = _PERCENT.sub(lambda m: f" {m.group(1)} percent ", text)
    text = _DECIMAL.sub(_spell_decimal, text)
    # "mp3" -> "mp 3", "3d" -> "3 d", so the digits can be spelled.
    text = _LETTER_DIGIT.sub(" ", text)
    return _INTEGER.sub(lambda m: f" {number_to_words(m.group(0))} ", text)


# ---------------------------------------------------------------------------
# The full normalizer
# ---------------------------------------------------------------------------

# [LAUGHTER], [INAUDIBLE], [MUSIC PLAYING]: square brackets are never speech.
_SQUARE_BRACKETS = re.compile(r"\[[^\]]*\]")
# (APPLAUSE), (LAUGHS): parentheses only when the inside is ALL CAPS, because
# a lowercase "f(x)" in a math lecture is real speech we must keep.
_CAPS_PARENS = re.compile(r"\([A-Z][A-Z' -]*[A-Z]\)")
# Speaker labels at the start of a line or after a ">>" marker, e.g.
# "PROFESSOR:", "GILBERT STRANG:", ">> AUDIENCE:". At least two capitals so
# that a sentence like "I: the identity" is not mistaken for a label.
_SPEAKER_LABEL = re.compile(r"(?m)(?:^|(?<=>>))\s*[A-Z][A-Z.' -]*[A-Z]:(?=\s|$)")
_CHEVRONS = re.compile(r">>+")

# Characters NFKC would turn into plain digits ("²" -> "2"), losing the
# meaning, so we translate them first.
_POWERS = {"²": " squared ", "³": " cubed "}
_QUOTES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'", "`": "'"})
_SYMBOLS = {
    "+": " plus ",
    "=": " equals ",
    "&": " and ",
    "×": " times ",
    "±": " plus or minus ",
    "%": " percent ",
}

_APOSTROPHE = re.compile(r"'")
# \w is letters, digits and underscore in any script. Anything else, and the
# underscore, becomes a space.
_NOT_WORD = re.compile(r"[\W_]+")

FILLER_WORDS = frozenset(
    {"um", "umm", "uh", "uhh", "uhm", "erm", "er", "ah", "hmm", "hm", "mm", "mhm"}
)
SPELLING_VARIANTS = {
    "ok": "okay",
    "alright": "all right",
    "gonna": "going to",
    "wanna": "want to",
}


def strip_caption_markup(text: str) -> str:
    """Remove sound tags, speaker labels and ">>" markers (case-sensitive,
    so this must run BEFORE lowercasing)."""
    text = _SQUARE_BRACKETS.sub(" ", text)
    text = _CAPS_PARENS.sub(" ", text)
    text = _SPEAKER_LABEL.sub(" ", text)
    return _CHEVRONS.sub(" ", text)


def normalize_text(
    text: str,
    *,
    numbers_to_words: bool = True,
    remove_fillers: bool = True,
) -> str:
    """Return the canonical, comparison-ready form of `text`.

    The result is lowercase words separated by single spaces, with no
    punctuation. Applying it twice gives the same result as applying it once
    (it is "idempotent"), which the tests check.
    """
    text = strip_caption_markup(text)
    for char, words in _POWERS.items():
        text = text.replace(char, words)
    text = text.translate(_QUOTES)
    # NFKC folds look-alike characters into one standard form: full-width
    # "３" -> "3", the ligature "ﬁ" -> "fi", a non-breaking space -> a space.
    text = unicodedata.normalize("NFKC", text).lower()
    if numbers_to_words:
        text = expand_numbers(text)
    for symbol, words in _SYMBOLS.items():
        text = text.replace(symbol, words)
    text = _APOSTROPHE.sub("", text)
    text = _NOT_WORD.sub(" ", text)

    words: list[str] = []
    for word in text.split():
        if remove_fillers and word in FILLER_WORDS:
            continue
        words.extend(SPELLING_VARIANTS.get(word, word).split())
    return " ".join(words)


def normalize_words(text: str, **options: bool) -> list[str]:
    """Same as `normalize_text`, but returned as a list of words."""
    return normalize_text(text, **options).split()


def load_text(path: str | Path) -> str:
    """Read the spoken text out of any transcript file this project writes.

    * `transcript.json` from `notetaker transcribe` ({"segments": [...]})
    * `reference.json` from `notetaker evalset` ([{"text": ...}, ...])
    * anything else is read as plain text (e.g. `reference.txt`).
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".json":
        return raw
    data = json.loads(raw)
    items = data.get("segments", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a list of segments or cues")
    return "\n".join(str(item.get("text", "")) for item in items)
