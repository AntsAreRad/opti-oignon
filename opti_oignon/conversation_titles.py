"""The automatic title of a conversation: a few of the words the user typed in its first turns, and nothing else.

Pure and stdlib-only: no I/O and no import of the package, so that it loads alone where it is tested. The store
hands it the stored user turns, decrypted -- a turn it could not decrypt as ``SEALED`` --, and the configuration read
from ``config/titles.yaml`` (``read_config``); it hands back the title, the digest of the words that made it and the
place of the turn they came from (``first_typed_source``), nothing, or that it cannot tell (``UNREADABLE``: a turn
before any that titles is still sealed), and the store keeps the title it has.

The words are a turn's typed text only: the whole content of a turn the user typed with no part from elsewhere, or
else its typed parts outside any run of parts that holds a document -- the rule ``memory/auto_capture.typed_turns``
holds, of which ``typed_text`` is a copy a contract keeps equal. A refined, legacy or document turn, a withdrawn turn
and a turn the grammar refuses give none. A turn that names a password -- anywhere in its content, typed words,
pasted parts and a withdrawn turn alike -- gives none, and neither does any later turn: the words around the name,
before it as after it, may be the secret. A password is named when one of titles.yaml's ``password_names`` matches
as its pattern says (``names_password``). Of the other turns' words a title keeps words of letters only: a word
holding a character of Unicode's number categories goes whole -- a number, a version, a code, and the common password
made of a word and digits --, and so do code, an address, a path, a key written with other characters and a string
switching case often. A number written in words, or in ideographs (which are letters, a numeric value or not),
stays. In a script
written without spaces a run of its letters is a word, ended by a digit, a mark of punctuation or another script.
What no rule here can tell is a secret made of letters alone, typed in a turn that names no password as titles.yaml
lists them -- a name that mostly means something else, like pass or pw, is not listed --, or alone in a turn before
the one that does. A title is bounded in code points, cut at a space, one
line, and says when it was cut.
"""

import bisect
import hashlib
import itertools
import re
import unicodedata
from dataclasses import dataclass

PLACEHOLDER = "New conversation"
# Where a title came from: the placeholder, rule (a), the model's refinement (block 180), the user (or an API caller
# that named one), or a time before this module; only the first three are ever written again automatically.
SOURCES = ("default", "auto", "refined", "manual", "legacy")
AUTOMATIC = ("default", "auto", "refined")
# What the store hands for a turn it could not decrypt; what first_typed_source answers when such a turn comes before
# any that titles: it cannot tell -- the turn may title, or name a password -- and the store keeps the title it has.
SEALED = ("sealed",)
UNREADABLE = ("unreadable",)

_RSQUO = chr(0x2019)
# The Tibetan syllable marks: they join a word's syllables, as a hyphen joins two letters.
_TSHEG = (chr(0x0F0B), chr(0x0F0C))
# What a word is stripped of at its ends: ASCII punctuation, the typographic quotes, guillemets and the ellipsis.
_PUNCT = ".,;:!?()[]{}<>\"'" + "".join(chr(code) for code in (0x2018, 0x2019, 0x201C, 0x201D, 0x00AB, 0x00BB, 0x2026))
_FENCE = ("```", "~~~")
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_MARKER = re.compile(r"^\s*(?:#{1,6}|>|[-*+]|\d{1,3}[.)])\s+")
_EMPHASIS = re.compile(r"(?:(?<=\s)|^)[*_]+|[*_]+(?=\s|$)")
_SPACE = re.compile(r"\s+")
_LETTERS = re.compile(r"[^\W\d_]+")
# Where the text is cut into segments when a password's name is looked for: any character that is neither a letter,
# a digit, an underscore, a hyphen (the ASCII one, U+2010, U+2011) nor an apostrophe -- a space, a path's slash, an
# address's dot, a sigil.
_BREAK = re.compile(r"[^\w\-'" + chr(0x2019) + chr(0x2010) + chr(0x2011) + r"]+")
# What is not a letter in a segment, the NUL that parts the segments aside (NFKD never makes one).
_NONLETTER = re.compile(r"[^\w\x00]|[\d_]")
# Letters that draw nothing: a title made of them, of combining marks alone or of blank symbols says nothing.
_BLANK = frozenset(chr(code) for code in (0x115F, 0x1160, 0x2800, 0x3164, 0xFFA0))
# Scripts written without spaces between words -- Thai and Lao, Tibetan, Myanmar and its extensions, Khmer, Tai Le,
# New Tai Lue, Buginese, Tai Tham, Balinese, Sundanese, Javanese, Cham, Tai Viet, the ideographic iteration marks,
# kana, the ideographs, halfwidth kana: a run of their letters is one word however long.
_UNSPACED = ((0x0E00, 0x0EFF), (0x0F00, 0x0FFF), (0x1000, 0x109F), (0x1780, 0x17FF), (0x1950, 0x19DF),
             (0x1A00, 0x1AAF), (0x1B00, 0x1BBF), (0x3005, 0x3006), (0x3040, 0x30FF), (0x3400, 0x4DBF),
             (0x4E00, 0x9FFF), (0xA980, 0xA9FF), (0xAA00, 0xAADF), (0xF900, 0xFAFF), (0xFF66, 0xFF9F),
             (0x20000, 0x2FA1F))
# The punctuation and symbols of the ideographic, vertical, compatibility and fullwidth blocks: they separate words.
_SEPARATING = ((0x3000, 0x303F), (0xFE10, 0xFE19), (0xFE30, 0xFE4F), (0xFF00, 0xFFEF))


class TitlesConfigError(ValueError):
    """``config/titles.yaml`` holds an unknown key, misses one or gives one the wrong type: no automatic title."""


@dataclass(frozen=True)
class DeriveConfig:
    max_chars: int
    min_letters: int
    max_user_turns: int
    max_token_chars: int
    max_case_switches: int
    max_read_chars: int
    ellipsis: str
    # Each password name as its words' patterns: a tuple of (mode, letters), mode one of _MODES.
    password_names: tuple
    # What each digit or symbol written for a letter is read as when a password's name is looked for (str.translate).
    lookalikes: dict


@dataclass(frozen=True)
class TitlesConfig:
    enabled: bool
    derive: DeriveConfig
    manual_max_chars: int
    search_scan_limit: int


# The keys of each section and their types: the loader reads these and only these.
_SECTIONS = {"enabled": bool, "derive": dict, "manual": dict, "search": dict}
_DERIVE = {
    "max_chars": int, "min_letters": int, "max_user_turns": int, "max_token_chars": int, "max_case_switches": int,
    "max_read_chars": int, "ellipsis": str, "password_names": list, "password_lookalikes": dict,
}
_MANUAL = {"max_chars": int}
_SEARCH = {"scan_limit": int}
# The integers that may be zero; every other one is at least 1.
_MAY_BE_ZERO = {"max_case_switches", "min_letters"}
# How a pattern's word matches the text: x a whole run (or x with a trailing s); x* a run's start; *x a run's end (or
# the end x and a trailing s); ~x~ anywhere in a run; *x* anywhere in a segment; +x anywhere in a segment longer than
# x (nor x and a trailing s). A run is a word's letters cut at camelCase, a segment an identifier's letters joined.
_MODES = ("whole", "start", "end", "within", "anywhere", "inside")
_ON_SEGMENTS = ("anywhere", "inside")
# A phrase's words joined, matched anywhere in one word of the text, from this many letters (wifipass, motdepasse).
_JOINED_LETTERS = 6


def _section(data, shape, where):
    if not isinstance(data, dict):
        raise TitlesConfigError(f"{where} is not a mapping")
    unknown = sorted(set(data) - set(shape), key=str)
    if unknown:
        raise TitlesConfigError(f"{where} holds an unknown key: {unknown[0]}")
    missing = sorted(set(shape) - set(data))
    if missing:
        raise TitlesConfigError(f"{where} misses the key {missing[0]}")
    for key, kind in shape.items():
        value = data[key]
        # A bool is an int to Python; here neither stands for the other.
        if type(value) is not kind:
            raise TitlesConfigError(f"{where}.{key} is not a {kind.__name__}")
        if kind is int and value < (0 if key in _MAY_BE_ZERO else 1):
            raise TitlesConfigError(f"{where}.{key} is out of range")
    return data


# Every combining mark and every format character of the planes in use (0 and 1), of the tags and of the variation
# selectors' supplement -- a soft hyphen, a zero-width space or joiner, a direction mark, a byte order mark --, built
# at first use and published whole: str.translate takes them off at the speed of C. They join no word apart, and a
# name holds none.
_MARK_TABLE = [None]


def _marks():
    table = _MARK_TABLE[0]
    if table is None:
        table = {code: None for code in (*range(0x20000), *range(0xE0000, 0xE0080), *range(0xE0100, 0xE01F0))
                 if unicodedata.category(chr(code)) in ("Mn", "Mc", "Me", "Cf")}
        _MARK_TABLE[0] = table
    return table


def _bare(word):
    """``word`` as a password's name is compared: case folded, its accents and other marks taken off -- a word typed
    with an accent matches the plain one the configuration lists."""
    return unicodedata.normalize("NFKD", word.casefold()).translate(_marks())


def _is_letter(char):
    return unicodedata.category(char)[0] in "LM"


def _pattern(entry):
    """A ``password_names`` entry as its words' (mode, letters), or None when it is not one."""
    if not isinstance(entry, str) or not entry.split():
        return None
    out = []
    for word in entry.split():
        if word.startswith("+"):
            mode, core = "inside", word[1:]
        elif len(word) > 2 and word.startswith("~") and word.endswith("~"):
            mode, core = "within", word[1:-1]
        elif len(word) > 2 and word.startswith("*") and word.endswith("*"):
            mode, core = "anywhere", word[1:-1]
        elif word.endswith("*"):
            mode, core = "start", word[:-1]
        elif word.startswith("*"):
            mode, core = "end", word[1:]
        else:
            mode, core = "whole", word
        letters = _bare(core)
        if not letters or not all(unicodedata.category(char)[0] == "L" for char in letters):
            return None
        out.append((mode, letters))
    return tuple(out)


def read_config(data):
    """The configuration from a parsed ``titles.yaml``: every key known and typed, or ``TitlesConfigError``."""
    top = _section(data, _SECTIONS, "titles.yaml")
    derive = _section(top["derive"], _DERIVE, "derive")
    names = []
    for entry in derive["password_names"]:
        pattern = _pattern(entry)
        if pattern is None:
            raise TitlesConfigError("derive.password_names holds an entry that is not words of letters and their marks")
        names.append(pattern)
    lookalikes = {}
    for key, value in derive["password_lookalikes"].items():
        if not (isinstance(key, str) and len(key) == 1 and not _is_letter(key) and not key.isspace()
                and isinstance(value, str) and len(value) == 1 and _is_letter(value)):
            raise TitlesConfigError("derive.password_lookalikes maps one character that is no letter to one letter")
        lookalikes[ord(key)] = value
    if len(derive["ellipsis"]) >= derive["max_chars"]:
        raise TitlesConfigError("derive.ellipsis is as long as a title may be")
    manual = _section(top["manual"], _MANUAL, "manual")
    search = _section(top["search"], _SEARCH, "search")
    plain = ("password_names", "password_lookalikes")
    return TitlesConfig(
        enabled=top["enabled"],
        derive=DeriveConfig(
            **{key: derive[key] for key in _DERIVE if key not in plain},
            password_names=tuple(names),
            lookalikes=lookalikes,
        ),
        manual_max_chars=manual["max_chars"],
        search_scan_limit=search["scan_limit"],
    )


def typed_text(content, origin, segments):
    """The words the user typed in one stored user turn, as ``memory/auto_capture.typed_turns`` reads them, or "".

    The turn's whole content when it is typed and carries no segment; else its typed segments, outside any run of
    adjacent segments that holds a document (words typed beside a pasted document take their sense from it). A
    segment that does not read as [start, end, base], sorted, disjoint and inside the content gives nothing.
    """
    if not isinstance(content, str) or not isinstance(origin, str) or not isinstance(segments, (list, tuple)):
        return ""
    if not segments:
        parts = [content] if origin == "typed" else []
    else:
        parts, end = [], 0
        for segment in segments:
            if not isinstance(segment, (list, tuple)) or len(segment) != 3:
                return ""
            start, stop, _label = segment
            if type(start) is not int or type(stop) is not int or not end <= start < stop <= len(content):
                return ""
            end = stop
        run_start = 0
        while run_start < len(segments):
            run_end = run_start + 1
            while run_end < len(segments) and segments[run_end][0] == segments[run_end - 1][1]:
                run_end += 1
            run = segments[run_start:run_end]
            if not any(label == "document" for _start, _stop, label in run):
                parts.extend(content[start:stop] for start, stop, label in run if label == "typed")
            run_start = run_end
    return "\n".join(part.strip() for part in parts if part.strip())


def basis_of(text):
    """The digest of the typed words a title was made from: SHA-256 of their NFC form."""
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()


def _unspaced(char):
    code = ord(char)
    return any(low <= code <= high for low, high in _UNSPACED)


def _runs(segment):
    """The runs of letters of one segment of the text as a password's name is compared, an identifier written in
    camelCase cut where a lowercase letter meets an uppercase one that begins a run of two letters or more (dbPassword,
    adminPw -- never PwC)."""
    out = []
    for match in _LETTERS.finditer(segment):
        run, start = match.group(), 0
        # Only a run with an uppercase letter past its first can be camelCase; its last letter never begins a word
        # of its own: PwC stays one.
        if run[1:] != run[1:].lower():
            for index in range(1, len(run) - 1):
                if unicodedata.category(run[index - 1]) == "Ll" and unicodedata.category(run[index]) in ("Lu", "Lt"):
                    out.append(_bare(run[start:index]))
                    start = index
        out.append(_bare(run[start:]))
    return out


def _fits(mode, core, unit, as_typed):
    if mode == "whole":
        return unit in (core, core + "s")
    if mode == "start":
        return unit.startswith(core)
    if mode == "end":
        return unit.endswith(core) or unit.endswith(core + "s")
    if mode in ("anywhere", "within"):
        return core in unit
    # A longer word, as typed: a symbol read as a letter would lengthen the word it begins ($PWD).
    return as_typed and core in unit and unit not in (core, core + "s")


def _hits(core, everything, starts):
    """The indices of the segments whose letters hold ``core``: found in the NUL-joined letters at C speed, each
    place mapped to its segment by the segments' offsets ``starts``."""
    found, at = [], everything.find(core)
    while at != -1:
        index = bisect.bisect_right(starts, at) - 1
        if not found or found[-1] != index:
            found.append(index)
        at = everything.find(core, at + 1)
    return found


def _named(name, parts, letters, everything, starts, as_typed):
    """Whether ``name`` matches the text cut into ``parts`` (its segments as typed) and ``letters`` (each one's
    letters, joined by NULs in ``everything`` from the offsets ``starts``): one word against the runs of the segments
    that hold its letters (whole, start, end, within) or against those segments (anywhere, inside); several words
    against as many consecutive segments, or consecutive runs, around the places of their longest word, or, joined,
    anywhere in one segment (a joined form of six letters or more). Only the segments that hold a name's letters are
    read: a long text costs what its names' places cost."""
    if len(name) == 1:
        mode, core = name[0]
        hits = _hits(core, everything, starts)
        if mode in _ON_SEGMENTS:
            return any(_fits(mode, core, letters[index], as_typed) for index in hits)
        return any(_fits(mode, core, run, as_typed) for index in hits for run in _runs(parts[index]))
    size = len(name)
    cores = [core for _mode, core in name]
    longest = max(range(size), key=lambda position: len(cores[position]))
    hits = _hits(cores[longest], everything, starts)
    for hit in hits:
        first = hit - longest
        if 0 <= first and first + size <= len(letters) and all(
                _fits(mode, core, letters[first + offset], as_typed) for offset, (mode, core) in enumerate(name)):
            return True
    # Consecutive runs: those of the segments near each place of the longest word that hold one of the name's words,
    # a segment's runs following its neighbour's only when the two segments are next to each other.
    near = sorted({index for hit in hits for index in range(max(0, hit - size + 1), min(len(letters), hit + size))
                   if any(core in letters[index] for core in cores)})
    sequence = [(index, run) for index in near for run in _runs(parts[index])]
    for begin in range(len(sequence) - size + 1):
        window = sequence[begin:begin + size]
        if all(later[0] - earlier[0] in (0, 1) for earlier, later in zip(window, window[1:])) \
                and all(_fits(mode, core, run, as_typed) for (mode, core), (_index, run) in zip(name, window)):
            return True
    joined = "".join(cores)
    return len(joined) >= _JOINED_LETTERS and bool(_hits(joined, everything, starts))


def names_password(text, cfg):
    """Whether ``text`` names a password: one of ``cfg.password_names`` matches.

    ``cfg`` is a ``DeriveConfig``. Combining marks and format characters are taken off first (a soft hyphen or a
    zero-width space inside a name hides nothing). The text is read twice -- as it is, and with each digit or symbol
    of ``cfg.lookalikes`` read as the letter it is written for -- and cut into segments wherever a character neither a
    letter, a digit, an underscore, a hyphen nor an apostrophe stands (a space, a path's slash, an address's dot, a
    sigil), each the letters it holds, case and marks aside: one pass of C for the whole text. A name none of whose
    segments holds all its letters is not looked for further; for the others, only the segments that hold its words
    are read again, cut into their runs, an identifier in camelCase cut into its words. A name of one word matches
    as its pattern says -- whole, start, end and within against a run, anywhere and inside against a segment, inside
    only as typed (a symbol read as a letter would lengthen the word it begins: $PWD) --; a name of several words
    matches as many consecutive segments or runs, and, its words joined, anywhere in one segment.
    """
    if not isinstance(text, str) or not text:
        return False
    plain = unicodedata.normalize("NFC", text).translate(_marks())
    for as_typed, reading in ((True, plain), (False, plain.translate(cfg.lookalikes))):
        parts = _BREAK.sub(" ", reading).split()
        letters = _bare(_NONLETTER.sub("", "\0".join(parts))).split("\0")
        everything = "\0".join(letters)
        starts = list(itertools.accumulate((len(unit) + 1 for unit in letters), initial=0))
        for name in cfg.password_names:
            if all(core in everything for _mode, core in name) \
                    and _named(name, parts, letters, everything, starts, as_typed):
                return True
    return False


def settings_digest(cfg):
    """The digest of every setting of ``cfg`` (a ``DeriveConfig``) a title depends on. The store marks each title it
    drew, and each placeholder it settled, with it and with the print of this module's code: either changed, and the
    title is read again."""
    names = sorted(repr(name) for name in cfg.password_names)
    lookalikes = sorted((key, value) for key, value in cfg.lookalikes.items())
    shape = (cfg.max_chars, cfg.min_letters, cfg.max_user_turns, cfg.max_token_chars, cfg.max_case_switches,
             cfg.max_read_chars, cfg.ellipsis, names, lookalikes)
    return hashlib.sha256(repr(shape).encode("utf-8")).hexdigest()


def _case_switches(word):
    """How many times a lowercase letter is followed by an uppercase one: a random-case string switches often."""
    return sum(1 for first, second in zip(word, word[1:])
               if unicodedata.category(first) == "Ll" and unicodedata.category(second) in ("Lu", "Lt"))


def _letters_only(word):
    for index, char in enumerate(word):
        if _is_letter(char):
            continue
        # A hyphen, an apostrophe or a Tibetan syllable mark between two letters belongs to the word.
        if char in ("-", "'", _RSQUO) + _TSHEG and 0 < index < len(word) - 1 \
                and _is_letter(word[index - 1]) and _is_letter(word[index + 1]):
            continue
        return False
    return True


def _kept(word, cfg):
    """Whether ``word``, its punctuation stripped, may stand in a title: letters only -- a hyphen, an apostrophe or a
    syllable mark between two of them aside --, a letter among them rather than marks alone, no longer than
    ``max_token_chars`` unless of a script written without spaces, and not switching case often. A digit or any
    other character Unicode counts as a number, anywhere in it, and the word goes."""
    if not word or not _letters_only(word) or not any(unicodedata.category(char)[0] == "L" for char in word):
        return False
    long_enough = len(word) <= cfg.max_token_chars or all(_unspaced(char) or not _is_letter(char) for char in word)
    return long_enough and _case_switches(word) <= cfg.max_case_switches


def _plain_lines(text):
    """The text's lines with fenced code, inline code, a leading Markdown marker and emphasis runs taken out."""
    lines, fenced = [], False
    for line in text.split("\n"):
        if line.strip().startswith(_FENCE):
            fenced = not fenced
            continue
        if fenced:
            continue
        line = _INLINE_CODE.sub(" ", line)
        line = _MARKER.sub("", line, count=1)
        lines.append(_EMPHASIS.sub(" ", line))
    return lines


def _spacing(char):
    """Whether ``char`` separates words: a control, format, private, unassigned or surrogate character, a line or
    paragraph separator, or a mark of punctuation or a symbol of the scripts written without spaces -- their own
    (the Tibetan syllable marks aside, which join), the ideographic, the vertical, the compatibility and the fullwidth
    ones."""
    category = unicodedata.category(char)
    if category[0] == "C" or category in ("Zl", "Zp"):
        return True
    if char in _TSHEG or category[0] not in "PSZ":
        return False
    code = ord(char)
    return _unspaced(char) or any(low <= code <= high for low, high in _SEPARATING)


def _folded(text):
    """Every character that separates words a space, and a space where a run of a script written without spaces
    meets anything else -- a digit, a mark of punctuation, another script --; white space collapsed, the ends
    stripped. A combining mark stays with the letter before it, a Tibetan syllable mark with its syllables."""
    out, inside = [], None
    for char in text:
        if _spacing(char):
            out.append(" ")
            inside = None
            continue
        if unicodedata.category(char)[0] == "M":
            out.append(char)
            continue
        unspaced = _unspaced(char) and (_is_letter(char) or char in _TSHEG)
        if inside is not None and unspaced != inside:
            out.append(" ")
        out.append(char)
        inside = unspaced
    return _SPACE.sub(" ", "".join(out)).strip()


def _head(text, limit):
    """The first ``limit`` characters of ``text``, cut back to a space and before an inline code span the cut would
    leave open: a word or a code span is never half read."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    space = max(head.rfind(" "), head.rfind("\n"))
    if space > 0:
        head = head[:space]
    if head.count("`") % 2:
        head = head[: head.rindex("`")]
    return head


def derive_title(text, cfg):
    """Rule (a): a title of the plain words of ``text``, or None when it names a password or its plain words hold
    too few letters.

    ``cfg`` is a ``DeriveConfig``. The title holds no character of the C categories, no line break and no character of
    Unicode's number categories (N); it is drawn from the first ``cfg.max_read_chars`` characters of ``text``, and is
    at most ``cfg.max_chars`` code points, cut at a space with ``cfg.ellipsis`` after it.
    """
    if not isinstance(text, str) or not text:
        return None
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    if names_password(text, cfg):
        return None
    # The title is drawn from the typed words' first characters; the password's name was looked for in all of them.
    # A slash command, a path, an address: each holds a character no plain word holds, and goes whole.
    tokens = _folded(" ".join(_plain_lines(_head(text, cfg.max_read_chars)))).split(" ")
    kept = [word for word in (token.strip(_PUNCT + "".join(_TSHEG)) for token in tokens) if _kept(word, cfg)]
    if sum(1 for word in kept for char in word if unicodedata.category(char)[0] == "L") < cfg.min_letters:
        return None
    title = " ".join(kept)
    if len(title) > cfg.max_chars:
        cut = title[: cfg.max_chars - len(cfg.ellipsis)]
        if " " in cut:
            cut = cut[: cut.rindex(" ")]
        else:
            while cut and unicodedata.category(title[len(cut)])[0] == "M":
                cut = cut[:-1]
        title = cut.rstrip() + cfg.ellipsis
    return title


def first_typed_source(turns, cfg, *, defect=None):
    """``(title, basis, index)`` from the first of ``turns`` whose typed words make a title; None when none does, or
    when a turn names a password before any titles; ``UNREADABLE`` when a turn before any that titles is ``SEALED``.

    ``turns`` are the conversation's stored user turns in order, each ``(content, origin, segments, context)`` with
    the content decrypted, or ``SEALED`` when the store could not decrypt it; at most ``cfg.derive.max_user_turns`` of
    them are read. Each turn's whole content -- its typed words and its pasted parts, a withdrawn turn's too -- is read
    for a password's name: a turn that names one ends the reading, and neither it nor a later turn titles. A sealed
    turn cannot be read for one: nothing after it may title. Then a turn whose context holds ``withdrawn``, or that
    ``defect(role, origin, segments, length)`` refuses, gives nothing. ``basis`` is the digest of the typed words the
    title was made from, ``index`` the place of their turn among ``turns``.
    """
    for index, (content, origin, segments, context) in enumerate(list(turns)[: cfg.derive.max_user_turns]):
        if content is SEALED:
            return UNREADABLE
        if not isinstance(content, str):
            continue
        if names_password(content, cfg.derive):
            return None
        if "withdrawn" in (context or ()):
            continue
        if defect is not None and (not isinstance(segments, (list, tuple))
                                   or defect("user", origin, segments, len(content)) is not None):
            continue
        text = typed_text(content, origin, segments)
        if not text:
            continue
        title = derive_title(text, cfg.derive)
        if title is not None:
            return title, basis_of(text), index
    return None


def fold_manual(title, max_chars, prefixes=(), *, cut=False):
    """A title the user typed as it is stored, or None when it is not one.

    NFC; every control, format or separator character a space; white space collapsed; the ends stripped. None when
    it begins as an encrypted value does (``prefixes``: it would be served as one, empty), when it is longer than
    ``max_chars`` code points (None: no bound) -- unless ``cut``, for a title another device kept: then it is cut
    there, never inside a combining sequence --, or when nothing is left that draws (blanks, combining marks alone).
    """
    if not isinstance(title, str):
        return None
    folded = _SPACE.sub(" ", "".join(" " if unicodedata.category(char)[0] == "C"
                                     or unicodedata.category(char) in ("Zl", "Zp") else char
                                     for char in unicodedata.normalize("NFC", title))).strip()
    if prefixes and folded.startswith(tuple(prefixes)):
        return None
    if max_chars is not None and len(folded) > max_chars:
        if not cut:
            return None
        end = max_chars
        while end and unicodedata.category(folded[end])[0] == "M":
            end -= 1
        folded = folded[:end].rstrip()
    if not any(unicodedata.category(char)[0] in "LNPS" and char not in _BLANK for char in folded):
        return None
    return folded


def fold_search(text):
    """``text`` as a search compares it: NFC, case folded, white space collapsed."""
    return _SPACE.sub(" ", unicodedata.normalize("NFC", str(text)).casefold()).strip()
