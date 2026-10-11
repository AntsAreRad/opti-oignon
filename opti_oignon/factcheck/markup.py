"""A closed markdown subset: a plain view of a text, and where each of its characters came from.

Chat answers are markdown, and so are the owner's notes. ``plain(text)`` reads a
closed, versioned subset and returns the text a sentence is read in, with a map
from every plain character to its position in the original text, so that every
span and every claim offset points into what was written. It is not the
browser's renderer: what the subset does not name is marked "not read" and is
never read as prose, so a sentence inside raw HTML or a comment never becomes a
claim or a passage; nothing is guessed.

| Construct                                   | Plain view     | Line kind or mark                  |
|---------------------------------------------|----------------|------------------------------------|
| fenced code (backticks, tildes), indented   | removed        | ``code``                           |
| ATX and setext headings                     | heading text   | ``heading``                        |
| list markers ``-`` ``*`` ``+`` ``1.`` ``1)``| removed        | ``list_item``                      |
| blockquote ``>``                            | removed        | ``quoted``                         |
| emphasis ``*`` ``**`` ``_`` ``__``          | removed        | none                               |
| strikethrough ``~~``                        | kept, marked   | its characters marked struck       |
| inline code                                 | its text       | its characters marked code         |
| links, autolinks                            | the link text  | none                               |
| images                                      | removed        | the line marked ``image``          |
| table rows (pipes, with a delimiter row)    | the row        | ``table_row``                      |
| raw HTML, footnote and link definitions,    | removed        | ``markup_unparsed``                |
| thematic breaks, inline HTML                | inline: kept   | the line marked ``html``           |

Struck text is text its author took back: a sentence that holds any of it is
never a claim ("markup not read") and never a passage that supports one.

``verbatim(text)`` is the view of a source that is not markdown: the text
itself, unchanged, read for its lines and paragraphs, its ``#`` heading lines
and the lines that open a list item, and nothing else. A heading line is a
boundary and the heading of what follows it, never a sentence.
"""

import re
from dataclasses import dataclass

checkpoint_before_apply = True

MARKUP_VERSION = 1

PROSE = "prose"
BLANK = "blank"
HEADING = "heading"
LIST_ITEM = "list_item"
LIST_CONT = "list_cont"
QUOTED = "quoted"
CODE = "code"
FENCE = "fence"
TABLE_ROW = "table_row"
TABLE_DELIM = "table_delim"
SETEXT = "setext"
UNPARSED = "markup_unparsed"

# Line kinds whose text is read as prose.
SENTENCE_KINDS = (PROSE, LIST_ITEM, LIST_CONT, QUOTED)
# Line kinds whose content reaches the plain view.
CONTENT_KINDS = (PROSE, LIST_ITEM, LIST_CONT, QUOTED, HEADING, TABLE_ROW)

SUBSET = (
    ("fenced_code", "``` or ~~~, at most three spaces of indent", "removed", CODE),
    ("indented_code", "four spaces or a tab after a blank line", "removed", CODE),
    ("atx_heading", "one to six #", "heading text", HEADING),
    ("setext_heading", "a single prose line underlined with = or -", "heading text", HEADING),
    ("list_item", "- * + or up to nine digits and . or )", "marker removed", LIST_ITEM),
    ("blockquote", ">", "marker removed", QUOTED),
    ("emphasis", "** __ * _ in pairs, _ never inside a word", "delimiters removed", "none"),
    ("strikethrough", "~~ in pairs", "delimiters removed, the text kept and marked struck",
     UNPARSED),
    ("inline_code", "a backtick run and its match", "its text", CODE),
    ("link", "[text](url)", "the text", "none"),
    ("autolink", "<scheme://...> or <user@host>", "the address", "none"),
    ("image", "![alt](url)", "removed", "image"),
    ("table", "a pipe row followed by a pipe delimiter row", "the row", TABLE_ROW),
    ("html_block", "a line opening an HTML tag or comment, to the next blank line", "removed", UNPARSED),
    ("definition", "[label]: or [^label]:", "removed", UNPARSED),
    ("thematic_break", "three or more - * or _", "removed", UNPARSED),
    ("inline_html", "a tag inside a line", "kept", "html"),
    ("verbatim", "a source that is not markdown: # heading lines and list-item lines only",
     "the text unchanged", "heading, list_item"),
)

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+|$)")
_ATX_CLOSE = re.compile(r"(?<![ \t])[ \t]+#+[ \t]*$")
_SETEXT = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
_BREAK = re.compile(r"^ {0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$")
_LIST = re.compile(r"^([ \t]*)([-*+]|\d{1,9}[.)])(?:[ \t]+|$)")
_QUOTE = re.compile(r"^ {0,3}>[ ]?")
_HTML_BLOCK = re.compile(r"^ {0,3}<(?:/?[A-Za-z][A-Za-z0-9-]*(?:[\s/>]|$)|!--)")
_DEFINITION = re.compile(r"^ {0,3}\[\^?[^\]]+\]:")
# The blanks after the last cell and after a closing pipe are one run when no pipe stands between them.
_DELIM = re.compile(r"^ {0,3}\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*(?:\|[ \t]*)?$")
_INDENTED = re.compile(r"^(?: {4}|\t)")
_LEADING = re.compile(r"^[ \t]*")

_TICKS = re.compile(r"`+")
# The openings of an image and of a link, the end of a target, and what follows it; ``_targets`` reads them.
_IMAGE_HEAD = re.compile(r"!\[")
_LINK_HEAD = re.compile(r"(?<!!)\[")
_TARGET_STOP = re.compile(r"[)\s]")
# A target never ends in a blank, so the blanks before a title are read only where a run of them starts.
_TARGET_TAIL = re.compile(r"(?:(?<!\s)\s+\"[^\"]*\")?\)")
# The same addresses in linear time: the host up to its first dot after its first character, read one way.
_AUTOLINK = re.compile(r"<((?:https?|ftp)://[^\s<>]+|[^\s<>@]+@[^\s<>@][^\s<>@.]*\.[^\s<>]+)>")
_INLINE_HTML = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
_STRIKE = re.compile(r"(~~)(?=\S)(.+?)(?<=\S)~~")
# The blanks around a closing colon are one run when no colon stands between them.
_WHOLLY_EMPHASISED = re.compile(r"^(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1[ \t]*(?::[ \t]*)?$")
_EMPHASIS = (
    re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1"),
    re.compile(r"(?<![*\w])(\*)(?=[^\s*])(.+?)(?<=[^\s*])\*(?![*\w])"),
    re.compile(r"(?<![_\w])(_)(?=[^\s_])(.+?)(?<=[^\s_])_(?![_\w])"),
)


@dataclass(frozen=True)
class Line:
    """One line: its kind, its content in plain coordinates, the whole line in the original.

    ``level`` is a heading's level (0 for any other line); ``indent`` a list
    item's indent in columns (-1 for any other line); ``labelish`` says that a
    prose line is wholly emphasised or ends with a colon, the shape of a label
    set above what it introduces.
    """

    kind: str
    start: int
    end: int
    ostart: int
    oend: int
    block: int
    list_id: int
    image: bool
    html: bool
    level: int = 0
    indent: int = -1
    labelish: bool = False


@dataclass(frozen=True)
class View:
    """A plain view: its text, the original position of each character, its lines."""

    text: str
    origin: tuple
    lines: tuple
    inline_removed: frozenset
    code_chars: frozenset
    html_chars: frozenset
    images: tuple
    markdown: bool
    original: str = ""
    struck_chars: frozenset = frozenset()

    def line_at(self, position):
        """Index of the line holding plain ``position``."""
        low, high = 0, len(self.lines) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if self.lines[middle].start <= position:
                low = middle
            else:
                high = middle - 1
        return low


def rules():
    """The subset, for the rules digest."""
    return {"version": MARKUP_VERSION, "subset": [list(row) for row in SUBSET]}


def to_original(view, start, end):
    """Original (start, end) of the plain range [start, end); end is exclusive."""
    if end <= start:
        position = view.origin[start] if start < len(view.origin) else len(view.origin)
        return position, position
    return view.origin[start], view.origin[end - 1] + 1


def _raw_lines(text):
    lines = []
    start = 0
    while True:
        cut = text.find("\n", start)
        if cut == -1:
            lines.append((start, len(text), False))
            break
        lines.append((start, cut, True))
        start = cut + 1
    return lines


def verbatim(text):
    """The text as it is: lines, paragraphs, ``#`` heading lines and list-item lines, nothing else read."""
    lines = []
    block = -1
    list_id = -1
    previous = BLANK
    last_nonblank = None
    for ostart, oend, _ in _raw_lines(text):
        line = text[ostart:oend]
        start, end, level, indent = ostart, oend, 0, -1
        list_line = -1
        if not line.strip():
            kind, line_block = BLANK, -1
        elif (match := _ATX.match(line)):
            kind, level = HEADING, len(match.group(1))
            content = line[match.end():]
            closing = _ATX_CLOSE.search(content)
            if closing:
                content = content[:closing.start()]
            start = ostart + match.end()
            end = start + len(content.rstrip())
            block += 1
            line_block = block
        elif (match := _LIST.match(line)):
            kind, indent = LIST_ITEM, len(match.group(1).expandtabs(4))
            start = ostart + match.end()
            block += 1
            if last_nonblank not in (LIST_ITEM, LIST_CONT):
                list_id += 1
            line_block, list_line = block, list_id
        elif previous in (LIST_ITEM, LIST_CONT) and line[:1] in (" ", "\t"):
            kind, line_block, list_line = LIST_CONT, block, list_id
            start = ostart + _LEADING.match(line).end()
        else:
            kind = PROSE
            if previous != PROSE:
                block += 1
            line_block = block
        labelish = kind == PROSE and line.rstrip().endswith(":")
        lines.append(Line(kind, start, end, ostart, oend, line_block, list_line, False, False, level, indent,
                          labelish))
        previous = kind
        if kind != BLANK:
            last_nonblank = kind
    return View(text, tuple(range(len(text))), tuple(lines), frozenset(), frozenset(), frozenset(), (),
                False, text, frozenset())


def _classify(text):
    """Kind and content range, in original coordinates, of every line."""
    raw = _raw_lines(text)
    out = []
    fence = None
    in_html = False
    in_table = False
    delimiter_due = False
    for index, (ostart, oend, _) in enumerate(raw):
        line = text[ostart:oend]
        previous = out[-1]["kind"] if out else None
        last_nonblank = next((entry["kind"] for entry in reversed(out) if entry["kind"] != BLANK), None)
        entry = {"kind": PROSE, "cstart": ostart, "cend": oend, "ostart": ostart, "oend": oend,
                 "level": 0, "indent": -1, "emphasised": bool(_WHOLLY_EMPHASISED.match(line.strip()))}
        out.append(entry)
        if fence:
            stripped = line.strip()
            closing = bool(stripped) and set(stripped) == {fence[0]} and len(stripped) >= len(fence)
            entry["kind"] = FENCE if closing else CODE
            if closing:
                fence = None
            continue
        if in_html:
            in_html = bool(line.strip())
            entry["kind"] = UNPARSED if in_html else BLANK
            continue
        if in_table:
            if "|" in line and line.strip():
                entry["kind"] = TABLE_DELIM if delimiter_due else TABLE_ROW
                delimiter_due = False
                continue
            in_table = False
        following = text[raw[index + 1][0]:raw[index + 1][1]] if index + 1 < len(raw) else ""
        if not line.strip():
            entry["kind"] = BLANK
        elif (match := _FENCE.match(line)):
            entry["kind"] = FENCE
            fence = match.group(1)
        elif _HTML_BLOCK.match(line):
            entry["kind"] = UNPARSED
            in_html = True
        elif _DEFINITION.match(line):
            entry["kind"] = UNPARSED
        elif (_INDENTED.match(line) and previous in (None, BLANK, CODE)
              and last_nonblank not in (LIST_ITEM, LIST_CONT)):
            entry["kind"] = CODE
        elif (match := _ATX.match(line)):
            entry["kind"] = HEADING
            entry["level"] = len(match.group(1))
            content = line[match.end():]
            closing = _ATX_CLOSE.search(content)
            if closing:
                content = content[:closing.start()]
            entry["cstart"] = ostart + match.end()
            entry["cend"] = entry["cstart"] + len(content.rstrip())
        elif "|" in line and "|" in following and _DELIM.match(following):
            entry["kind"] = TABLE_ROW
            in_table = True
            delimiter_due = True
        elif (_SETEXT.match(line) and previous == PROSE
              and (len(out) < 3 or out[-3]["kind"] not in (PROSE, LIST_CONT))):
            out[-2]["kind"] = HEADING
            out[-2]["level"] = 1 if line.strip().startswith("=") else 2
            entry["kind"] = SETEXT
        elif _BREAK.match(line):
            entry["kind"] = UNPARSED
        elif (match := _QUOTE.match(line)):
            position = match.end()
            while (inner := _QUOTE.match(line[position:])):
                position += inner.end()
            entry["kind"] = QUOTED
            entry["cstart"] = ostart + position
        elif (match := _LIST.match(line)):
            entry["kind"] = LIST_ITEM
            entry["indent"] = len(match.group(1).expandtabs(4))
            entry["cstart"] = ostart + match.end()
        elif last_nonblank in (LIST_ITEM, LIST_CONT) and (previous != BLANK or line[:1] in (" ", "\t")):
            entry["kind"] = LIST_CONT
            entry["cstart"] = ostart + _LEADING.match(line).end()
        else:
            entry["cstart"] = ostart + _LEADING.match(line).end()
    return out


def _code_spans(segment):
    """(start, inner start, inner end, end) of each code span of a line's content, in order, exactly as
    ``(`+)(.+?)(?<!`)\\1(?!`)`` finds them, in time linear in the content.

    A run of backticks opens with its first k backticks, k being the longest length at most its own that a later
    whole run of the same line has, and closes at the first such run; a run that finds none opens nothing. Each
    run's closing is found from the right, trying lengths from its own down: no more tries than its backticks.
    """
    runs = [(match.start(), match.end() - match.start()) for match in _TICKS.finditer(segment)]
    closing, nearest = [None] * len(runs), {}
    for index in range(len(runs) - 1, -1, -1):
        start, size = runs[index]
        if index + 1 < len(runs) and segment.find("\n", start + size, runs[index + 1][0]) >= 0:
            nearest = {}
        closing[index] = next(((nearest[length], length) for length in range(size, 0, -1) if length in nearest),
                              None)
        nearest[size] = index
    spans, index = [], 0
    while index < len(runs):
        if closing[index] is None:
            index += 1
            continue
        (start, _size), (close, length) = runs[index], closing[index]
        spans.append((start, start + length, runs[close][0], runs[close][0] + length))
        index = close + 1
    return spans


def _targets(subject, head, least):
    """(start, text end, end) of each ``[text](target "title")`` form of a line's content, in order, exactly as
    ``finditer`` finds them with ``!\\[[^\\]]*\\]\\([^)\\s]*(?:\\s+\\"[^\\"]*\\")?\\)`` (``head`` ``_IMAGE_HEAD``,
    ``least`` 0) or ``(?<!!)\\[([^\\]]+)\\]\\([^)\\s]*(?:\\s+\\"[^\\"]*\\")?\\)`` (``_LINK_HEAD``, 1), in time linear in
    the content.

    Those patterns read the text and the target again from every opening of a long run. A text ends at the first
    ``]`` after its opening, a target at the first ``)`` or blank after its ``(``, and what may follow a target --
    a title, then ``)`` -- is read at that end: so the next ``]`` and the next end of a target are kept with the place
    they were searched from, and what follows each end is read once.
    """
    found, at, tails = [], 0, {}
    bracket = (0, -2)  # (searched from, the next "]" from there or -1); -2 before any search
    stop = (0, -1)  # (searched from, where a target from there ends); -1 before any search
    while True:
        opening = head.search(subject, at)
        if opening is None:
            return found
        text = opening.end()
        if not (bracket[0] <= text and (bracket[1] == -1 or text <= bracket[1])):
            bracket = (text, subject.find("]", text))
        close = bracket[1]
        if close == -1:
            return found  # no "]" after this opening, so none after any opening after it
        if close - text < least or subject[close + 1:close + 2] != "(":
            at = opening.start() + 1
            continue
        target = close + 2
        if not (stop[0] <= target <= stop[1]):
            hit = _TARGET_STOP.search(subject, target)
            stop = (target, len(subject) if hit is None else hit.start())
        if stop[1] not in tails:
            tail = _TARGET_TAIL.match(subject, stop[1])
            tails[stop[1]] = None if tail is None else tail.end()
        end = tails[stop[1]]
        if end is None:
            at = opening.start() + 1
            continue
        found.append((opening.start(), close, end))
        at = end


def _inline(text, cstart, cend):
    """Plain characters of one line's content, with what each came from."""
    segment = text[cstart:cend]
    size = len(segment)
    deleted = [False] * size
    protected = [False] * size
    code = [False] * size
    html = [False] * size
    images = []
    for start, inner_start, inner_end, end in _code_spans(segment):
        for position in range(start, inner_start):
            deleted[position] = True
        for position in range(inner_end, end):
            deleted[position] = True
        for position in range(inner_start, inner_end):
            code[position] = True
            protected[position] = True
    subject = "".join("\x00" if protected[i] or deleted[i] else segment[i] for i in range(size))
    for start, _close, end in _targets(subject, _IMAGE_HEAD, 0):
        for position in range(start, end):
            deleted[position] = True
        images.append((cstart + start, cstart + end))
    for start, close, end in _targets(subject, _LINK_HEAD, 1):
        if any(deleted[start:end]):
            continue
        deleted[start] = True
        for position in range(close, end):
            deleted[position] = True
    for match in _AUTOLINK.finditer(subject):
        if any(deleted[match.start():match.end()]):
            continue
        deleted[match.start()] = True
        deleted[match.end() - 1] = True
    for match in _INLINE_HTML.finditer(subject):
        if any(deleted[match.start():match.end()]) or any(protected[match.start():match.end()]):
            continue
        for position in range(match.start(), match.end()):
            html[position] = True
    struck = [False] * size
    live = [i for i in range(size) if not deleted[i]]
    view = "".join("\x00" if protected[i] or html[i] else segment[i] for i in live)
    for match in _STRIKE.finditer(view):
        head = live[match.start():match.start() + 2]
        tail = live[match.end() - 2:match.end()]
        for position in head + tail:
            deleted[position] = True
        for position in live[match.start() + 2:match.end() - 2]:
            struck[position] = True
    for _ in range(4):
        live = [i for i in range(size) if not deleted[i]]
        view = "".join("\x00" if protected[i] or html[i] else segment[i] for i in live)
        changed = False
        for pattern in _EMPHASIS:
            for match in pattern.finditer(view):
                width = len(match.group(1))
                head = live[match.start():match.start() + width]
                tail = live[match.end() - width:match.end()]
                if any(deleted[i] for i in head + tail):
                    continue
                for position in head + tail:
                    deleted[position] = True
                changed = True
            if changed:
                break
        if not changed:
            break
    removed = frozenset(cstart + i for i in range(size) if deleted[i])
    kept = [i for i in range(size) if not deleted[i]]
    return kept, code, html, images, removed, struck


def plain(text):
    """The plain view of ``text`` read through the markdown subset."""
    entries = _classify(text)
    chars = []
    origin = []
    lines = []
    inline_removed = set()
    code_chars = set()
    html_chars = set()
    struck_chars = set()
    images = []
    block = -1
    list_id = -1
    previous_kind = None
    last_nonblank = None
    for entry in entries:
        kind = entry["kind"]
        start = len(chars)
        image = False
        has_html = False
        if kind in CONTENT_KINDS:
            kept, code, html, found_images, removed, struck = _inline(text, entry["cstart"], entry["cend"])
            inline_removed |= removed
            images.extend(found_images)
            image = bool(found_images)
            for position in kept:
                if code[position]:
                    code_chars.add(len(chars))
                if html[position]:
                    html_chars.add(len(chars))
                    has_html = True
                if struck[position]:
                    struck_chars.add(len(chars))
                chars.append(text[entry["cstart"] + position])
                origin.append(entry["cstart"] + position)
        end = len(chars)
        if kind == PROSE:
            if previous_kind != PROSE:
                block += 1
            line_block, line_list = block, -1
        elif kind == QUOTED:
            if previous_kind != QUOTED:
                block += 1
            line_block, line_list = block, -1
        elif kind == LIST_ITEM:
            block += 1
            if last_nonblank not in (LIST_ITEM, LIST_CONT):
                list_id += 1
            line_block, line_list = block, list_id
        elif kind == LIST_CONT:
            line_block, line_list = block, list_id
        elif kind == HEADING:
            block += 1
            line_block, line_list = block, -1
        else:
            line_block, line_list = -1, -1
        content = "".join(chars[start:end]).rstrip()
        labelish = kind == PROSE and (entry["emphasised"] or content.endswith(":"))
        lines.append(Line(kind, start, end, entry["ostart"], entry["oend"], line_block, line_list, image,
                          has_html, entry["level"], entry["indent"], labelish))
        if entry["oend"] < len(text):
            chars.append("\n")
            origin.append(entry["oend"])
        previous_kind = kind
        if kind != BLANK:
            last_nonblank = kind
    return View("".join(chars), tuple(origin), tuple(lines), frozenset(inline_removed),
                frozenset(code_chars), frozenset(html_chars), tuple(images), True, text, frozenset(struck_chars))
