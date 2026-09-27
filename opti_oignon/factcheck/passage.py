"""The passage check: fold v1, locating a quote in a chunk whose hash is recomputed, sentences, windows.

A passage counts only where the host finds it, character for character after a
fixed fold, inside a chunk whose SHA-256 the host recomputes. For one quote
against one chunk:

1. the chunk's hash is recomputed; a mismatch refuses the chunk ("chunk_changed")
   and nothing is located in it;
2. a chunk that is not NFC is refused ("chunk_not_nfc"), never repaired: the hash
   and the offsets are of the stored text;
3. the quote is NFC-normalised and folded; empty is "quote_empty", under the
   floor is "quote_too_short";
4. the chunk (its plain view, for markdown) is folded with an alignment map;
5. every occurrence is found, overlapping allowed, up to a cap (beyond it, the
   first ones and a flag); only aligned occurrences count, mapped back to
   code points of the chunk's own text; none is "quote_not_found".

A span never crosses a chunk: each chunk is searched on its own, after its own
hash check, never a join of chunks.

Fold v1 maps, one code point at a time, the classes of ``FOLD_TABLE``, and
nothing else: no case folding, no digit folding, never NFKC (compatibility
forms "may remove distinctions that are important to the semantics": NFKC turns
ten with a superscript nine into 109). An expansion (a ligature, the ellipsis)
is atomic: an occurrence must start on the first part of an expansion and end
on its last. A removed code point produces nothing, so a span never starts or
ends on one; a collapsed run of whitespace maps to its first code point.
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from . import markup

checkpoint_before_apply = True

FOLD_VERSION = 1

FOLD_TABLE = (
    ("whitespace", "str.isspace", " "),
    ("invisible", (0x00AD, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF), ""),
    ("apostrophe", (0x2018, 0x2019, 0x201A, 0x201B, 0x02BC, 0x2032, 0x00B4, 0x0060, 0xFF07), "'"),
    ("double_quote", (0x201C, 0x201D, 0x201E, 0x201F, 0x00AB, 0x00BB, 0x2033, 0xFF02), '"'),
    ("dash", (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212, 0xFE63, 0xFF0D), "-"),
    ("ellipsis", (0x2026,), "..."),
    ("ligature", (0xFB00,), "ff"),
    ("ligature", (0xFB01,), "fi"),
    ("ligature", (0xFB02,), "fl"),
    ("ligature", (0xFB03,), "ffi"),
    ("ligature", (0xFB04,), "ffl"),
    ("ligature", (0xFB05,), "st"),
    ("ligature", (0xFB06,), "st"),
)

_LOOKUP = {chr(point): becomes for _, points, becomes in FOLD_TABLE if not isinstance(points, str)
           for point in points}

SPLITTER_VERSION = 1
BOUNDARY = r"(?<=[.!?])\s+"
# A period after one of these never ends a sentence; nor does a period after a
# single letter (an initial: "George W. Bush", "U.S."). A cut that remains
# uncertain is read as a cut context, never as a whole sentence (see
# ``uncertain_end``).
ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "cf.", "vs.", "et al.", "Dr.", "Mr.", "Mrs.", "Prof.", "Fig.",
                 "p.", "pp.", "no.", "M.", "Mme.",
                 "No.", "Nos.", "Ms.", "Jr.", "Sr.", "St.", "Ste.", "Mt.", "Pr.", "Gen.", "Gov.", "Sen.",
                 "Rep.", "Pres.", "Col.", "Capt.", "Lt.", "Sgt.", "Hon.", "Rev.", "Inc.", "Ltd.", "Co.",
                 "Corp.", "Dept.", "Univ.", "Vol.", "vol.", "approx.", "ca.", "a.m.", "p.m.", "Jan.", "Feb.",
                 "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "Mlle.",
                 "Mgr.", "Me.", "env.", "av.", "apr.")
UNCERTAIN_END = ("a capitalised word of one to three letters before the period, a Roman numeral "
                 "(Henri IV.) excepted")
# Enough of the text before a period to read the word it ends.
_TAIL = 24

_BOUNDARY = re.compile(BOUNDARY)
_INITIAL = re.compile(r"(?<![^\W_])[^\W\d_]\.$")
_SHORT_CAPITAL = re.compile(r"(?<![^\W_])([^\W\d_]{1,3})\.$")
_ROMAN = re.compile(r"^[IVXLCDM]{2,}$")
_DOTTED = re.compile(r"(?:[^\W\d_]+\.)+$")
_SINGLE_WORD = frozenset(a for a in ABBREVIATIONS if " " not in a)
_SEVERAL_WORDS = tuple(a for a in ABBREVIATIONS if " " in a)


def fold_rules():
    """Fold v1, for the rules digest."""
    table = []
    for name, points, becomes in FOLD_TABLE:
        table.append([name, points if isinstance(points, str) else [f"U+{p:04X}" for p in points], becomes])
    return {"version": FOLD_VERSION, "table": table}


def splitter_rules():
    """The sentence splitter, for the rules digest."""
    return {"version": SPLITTER_VERSION, "boundary": BOUNDARY, "abbreviations": list(ABBREVIATIONS),
            "initials": "a period after a single letter never ends a sentence",
            "uncertain_end": UNCERTAIN_END,
            "blocks": "a blank line, a heading and a line opening a list item end a sentence"}


@dataclass(frozen=True)
class Folded:
    """Folded text; for each folded code point, the original index and its part (k, m)."""

    text: str
    origin: tuple
    part: tuple


def fold(text):
    """Fold v1 of ``text``, with its alignment map."""
    out, origin, part = [], [], []
    last_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if not last_space:
                out.append(" ")
                origin.append(index)
                part.append((1, 1))
                last_space = True
            continue
        becomes = _LOOKUP.get(char)
        if becomes is None:
            out.append(char)
            origin.append(index)
            part.append((1, 1))
            last_space = False
        elif becomes:
            size = len(becomes)
            for k, piece in enumerate(becomes, 1):
                out.append(piece)
                origin.append(index)
                part.append((k, size))
            last_space = False
    return Folded("".join(out), tuple(origin), tuple(part))


def fold_text(text):
    """Fold v1 of ``text``, the string alone."""
    return fold(text).text


def fold_quote(quote):
    """A quote as it is searched: NFC, folded, stripped at both ends."""
    return fold_text(unicodedata.normalize("NFC", quote)).strip(" ")


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Located:
    """What the passage check found: a refusal, or occurrences in code points."""

    refusal: object
    occurrences: tuple
    count: int
    truncated: bool


def locate(chunk_text, chunk_sha256, quote, *, min_quote_chars, max_occurrences, markdown=False):
    """Every aligned occurrence of ``quote`` in one chunk, at the chunk's own code points."""
    if sha256(chunk_text) != chunk_sha256:
        return Located("chunk_changed", (), 0, False)
    if not unicodedata.is_normalized("NFC", chunk_text):
        return Located("chunk_not_nfc", (), 0, False)
    folded_quote = fold_quote(quote)
    if not folded_quote:
        return Located("quote_empty", (), 0, False)
    if len(folded_quote) < min_quote_chars:
        return Located("quote_too_short", (), 0, False)
    view = markup.plain(chunk_text) if markdown else None
    folded = fold(view.text if view else chunk_text)
    found = []
    truncated = False
    last = len(folded_quote) - 1
    position = folded.text.find(folded_quote)
    while position != -1:
        first_part = folded.part[position]
        last_part = folded.part[position + last]
        if first_part[0] == 1 and last_part[0] == last_part[1]:
            if len(found) >= max_occurrences:
                truncated = True
                break
            start = folded.origin[position]
            end = folded.origin[position + last] + 1
            if view is not None:
                start, end = markup.to_original(view, start, end)
            found.append((start, end))
        position = folded.text.find(folded_quote, position + 1)
    if not found:
        return Located("quote_not_found", (), 0, False)
    return Located(None, tuple(found), len(found), truncated)


@dataclass(frozen=True)
class Sentence:
    """One sentence of a view, in plain coordinates."""

    start: int
    end: int
    kind: str
    block: int
    line: int
    list_id: int
    index: int


def _abbreviated(before):
    """Whether the period ending ``before`` belongs to an initial or an abbreviation of the list."""
    if _INITIAL.search(before):
        return True
    dotted = _DOTTED.search(before)
    if dotted and (dotted.start() == 0 or not before[dotted.start() - 1].isalnum()):
        word = dotted.group(0)
        if any(word[cut:] in _SINGLE_WORD for cut in [0] + [i + 1 for i, c in enumerate(word[:-1]) if c == "."]):
            return True
    for abbreviation in _SEVERAL_WORDS:
        if before.endswith(abbreviation):
            head = before[: len(before) - len(abbreviation)]
            if not head or not head[-1].isalnum():
                return True
    return False


def sentences(view):
    """Every sentence of the prose, list items and quotations of a view, in order."""
    out = []
    lines = view.lines
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.kind not in markup.SENTENCE_KINDS or line.block < 0:
            index += 1
            continue
        first = index
        while (index + 1 < len(lines) and lines[index + 1].block == line.block
               and lines[index + 1].kind in markup.SENTENCE_KINDS):
            index += 1
        block_start, block_end = lines[first].start, lines[index].end
        kind = markup.LIST_ITEM if line.kind in (markup.LIST_ITEM, markup.LIST_CONT) else line.kind
        text = view.text[block_start:block_end]
        cuts = [0]
        for match in _BOUNDARY.finditer(text):
            if not _abbreviated(text[max(0, match.start() - _TAIL): match.start()]):
                cuts.append(match.end())
        cuts.append(len(text))
        for left, right in zip(cuts, cuts[1:]):
            piece = text[left:right]
            stripped = piece.strip()
            if not stripped:
                continue
            start = block_start + left + (len(piece) - len(piece.lstrip()))
            end = start + len(stripped)
            out.append(Sentence(start, end, kind, line.block, view.line_at(start), line.list_id, len(out)))
        index += 1
    return out


def split_text(text, *, markdown=False):
    """The plain text of every sentence of ``text``."""
    view = markup.plain(text) if markdown else markup.verbatim(text)
    return [view.text[s.start:s.end] for s in sentences(view)]


def uncertain_end(text):
    """The word before a sentence's final period when that period may not end it, else None.

    A period after a capitalised word of one to three letters that is not in
    the list ("Pvt.", "Bob.") may be an abbreviation the list does not know: the
    sentence after it may be the tail of this one, and is never read as whole.
    """
    match = _SHORT_CAPITAL.search(text.rstrip()[-_TAIL:])
    if match and match.group(1)[0].isupper() and not _ROMAN.match(match.group(1)):
        return match.group(1) + "."
    return None


@dataclass(frozen=True)
class Window:
    """The context a sentence is read in.

    Ranges are in plain coordinates of the chunk's view, except a lead-in
    marked "original" (a line of markup the subset does not read, whose raw
    text is read). ``headings`` is the path of heading lines above the
    sentence, nearest first; ``lead_ins`` every line that introduces it,
    nearest first: for a list item, each parent item then the list's lead-in;
    for a paragraph, a one-line label set above it.
    """

    sentence: tuple
    before: object
    after: object
    heading: object
    heading_text: object
    heading_from: object
    headings: tuple
    locator_heading: object
    lead_ins: tuple
    quoted: bool
    list_item: bool
    first_in_chunk: bool
    last_in_chunk: bool
    cut_after: object

    @property
    def lead_in(self):
        """The nearest lead-in in plain coordinates, or None."""
        return next(((a, b) for where, a, b in self.lead_ins if where == "plain"), None)


def _one_line_paragraph(lines, index):
    line = lines[index]
    above = lines[index - 1] if index > 0 else None
    below = lines[index + 1] if index + 1 < len(lines) else None
    return ((above is None or above.block != line.block) and (below is None or below.block != line.block))


def window(view, sents, index, *, locator_heading=None):
    """The window of sentence ``index``: its neighbours, its heading path and what leads into it."""
    sentence = sents[index]
    lines = view.lines

    def neighbour(position):
        if not 0 <= position < len(sents):
            return None
        other = sents[position]
        if other.block == sentence.block:
            return other
        if (sentence.kind == markup.LIST_ITEM and other.kind == markup.LIST_ITEM
                and other.list_id == sentence.list_id):
            return other
        return None

    before = neighbour(index - 1)
    after = neighbour(index + 1)
    headings = []
    level = None
    for line in reversed(lines[: sentence.line]):
        if line.kind != markup.HEADING:
            continue
        if level is None or 0 < line.level < level:
            headings.append((line.start, line.end))
            level = line.level
            if level <= 1:
                break
    heading = headings[0] if headings else None
    if heading is not None:
        heading_text, heading_from = view.text[heading[0]:heading[1]], "line"
    elif locator_heading:
        heading_text, heading_from = str(locator_heading), "locator"
    else:
        heading_text = heading_from = None
    lead_ins = []
    first_line = next(i for i, line in enumerate(lines) if line.block == sentence.block)
    if sentence.kind == markup.LIST_ITEM:
        indent = lines[first_line].indent
        for line in reversed(lines[:first_line]):
            if line.list_id != sentence.list_id:
                continue
            if line.kind == markup.LIST_ITEM and 0 <= line.indent < indent:
                lead_ins.append(("plain", line.start, line.end))
                indent = line.indent
        first = min(i for i, line in enumerate(lines) if line.list_id == sentence.list_id)
        above = lines[:first]
    else:
        above = lines[:first_line]
    for position in range(len(above) - 1, -1, -1):
        line = above[position]
        if line.kind == markup.BLANK:
            continue
        if line.kind in (markup.PROSE, markup.QUOTED) and (
                sentence.kind == markup.LIST_ITEM or (line.labelish and _one_line_paragraph(lines, position))):
            lead_ins.append(("plain", line.start, line.end))
        elif line.kind == markup.UNPARSED:
            lead_ins.append(("original", line.ostart, line.oend))
        break
    cut_after = None
    if index > 0 and sents[index - 1].block == sentence.block:
        previous = sents[index - 1]
        cut_after = uncertain_end(view.text[previous.start:previous.end])
    return Window(
        sentence=(sentence.start, sentence.end),
        before=(before.start, before.end) if before else None,
        after=(after.start, after.end) if after else None,
        heading=heading, heading_text=heading_text, heading_from=heading_from,
        headings=tuple(headings), locator_heading=str(locator_heading) if locator_heading else None,
        lead_ins=tuple(lead_ins),
        quoted=sentence.kind == markup.QUOTED,
        list_item=sentence.kind == markup.LIST_ITEM,
        first_in_chunk=sentence.index == 0,
        last_in_chunk=sentence.index == len(sents) - 1,
        cut_after=cut_after,
    )
