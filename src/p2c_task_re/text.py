"""Text normalization and the quote locator (PLAN §3: V1 provenance check and the S1 fact sheet).

`locate_quote` finds a quote in a paper's text and returns character offsets into the *original*
text, so a fact's span can be stored and re-read verbatim. Matching ignores differences that do not
change what a sentence says: whitespace and line breaks, curly vs straight quotes, the various
dashes and the minus sign vs a hyphen, and Unicode compatibility forms (NFKC: µ → μ, ⁺ → +, ﬁ → fi).
It is otherwise exact: case, words and punctuation must match.
"""

import unicodedata
from collections.abc import Sequence

from p2c_task_re.data.jats import Section

_QUOTES_AND_DASHES = str.maketrans(
    {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u201a": "'",  # single low-9 quotation mark
        "\u201b": "'",  # single high-reversed-9 quotation mark
        "\u2032": "'",  # prime
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u201e": '"',  # double low-9 quotation mark
        "\u201f": '"',  # double high-reversed-9 quotation mark
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "\u2212": "-",  # minus sign
    }
)


def _pieces(s: str) -> list[tuple[str, int, int]]:
    """Split `s` into (normalized text, start, end) pieces, one per character cluster.

    A cluster is a character plus any combining marks after it, so NFKC can compose them. Whitespace
    becomes a single space; invisible format characters (soft hyphen, zero-width space) become "".
    """
    pieces: list[tuple[str, int, int]] = []
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and unicodedata.combining(s[j]):
            j += 1
        cluster = s[i:j]
        if cluster.isspace():
            piece = " "
        elif unicodedata.category(cluster[0]) == "Cf":
            piece = ""
        else:
            piece = unicodedata.normalize("NFKC", cluster).translate(_QUOTES_AND_DASHES)
        pieces.append((piece, i, j))
        i = j
    return pieces


def normalize(s: str) -> str:
    """NFKC, unify quotes/dashes/minus, collapse whitespace runs to one space, strip the ends."""
    return " ".join("".join(piece for piece, _, _ in _pieces(s)).split())


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Return (start, end) of `quote` in `text`, or None.

    Offsets index the original `text`, so `text[start:end]` is the passage as the paper writes it.
    The first verbatim occurrence wins; failing that, the first normalized one. Whitespace is ignored
    entirely when matching (a line break, a doubled space or a missing space all match), as are the
    other differences listed in the module docstring. Leading/trailing whitespace of `quote` is dropped.
    """
    quote = quote.strip()
    if quote and (at := text.find(quote)) >= 0:  # fast path: a verbatim copy
        return at, at + len(quote)
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for piece, start, end in _pieces(text):
        for ch in piece.replace(" ", ""):
            chars.append(ch)
            starts.append(start)
            ends.append(end)
    needle = "".join(normalize(quote).split())
    if not needle:
        return None
    at = "".join(chars).find(needle)
    if at < 0:
        return None
    return starts[at], ends[at + len(needle) - 1]


def section_at(sections: Sequence[Section], offset: int) -> Section | None:
    """The innermost section whose span contains `offset`, or None (e.g. a blank separator line)."""
    containing = [s for s in sections if s.start <= offset < s.end]
    return max(containing, key=lambda s: s.depth, default=None)
