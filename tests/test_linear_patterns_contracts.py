#!/usr/bin/env python3
"""Contracts for the patterns rewritten to read a long run in linear time: each reads as the one it replaced.

A pattern that took quadratic time or worse on a long run of one character
was rewritten so that each run is read from one place: a run read from its
start only, two repeats that shared a character made to split it one way,
the end of a lazy body pinned where no blank precedes it. Each contract
reads the patterns of one family where the package writes them -- from the
source, never by importing a module -- and holds them to the pattern each
replaced, written here, on a corpus built from both: texts drawn from their
characters, and texts built from their trees then mutated. What is compared
is everything a caller can read (the span and groups of ``search``,
``match``, every ``finditer`` match, ``sub``; ``findall``, ``split`` and
``subn`` are functions of those), or, where the callers of a
pattern read less, what they read: the groups, ``findall``, whether anything
matched. Each pattern has a witness: the corpus holds many texts its old
pattern matches and many it does not, or it does not reach the pattern.

  * EQ1 -- the frame pattern the composer and the wrapper share reads as
    before, and both modules hold the same one.
  * EQ2 -- the fenced-block readers read as before.
  * EQ3 -- the line-start readers in multiline mode give their callers the
    same groups, matches or answer.
  * EQ4 -- the readers led by a run of digits or blanks read as before.
  * EQ5 -- the section, sheet, repository and tag readers read as before.
  * EQ6 -- the claim reader of the fact checker reads as before, and the
    check of an ``rm`` outside the workspace answers as before.
  * EQ7 -- the FILE blocks of a coding reply are read as the pattern they
    replaced read them with ``findall``.
  * EQ8 -- the FILE blocks of a reply are read in linear time.
  * EQ9 -- the bracket reader of a summary takes out what it took out, in
    linear time.
  * EQ10 -- a frame opening left open at the Core's end up to its first
    attribute's name, or an envelope's bracket, finishes no marker with the
    turn the window joins after it.
  * EQ11 -- the FILE blocks of a reply are read in linear time on every shape
    a long reply can take: blanks, dashes, heads with no closing, blanks
    holding line breaks, end markers with no head, blocks one after another;
    four times longer reads in well under sixteen times the time.
  * EQ12 -- the FILE blocks of every reply of up to six pieces (a head, an
    end marker, dashes, a head word, a dash, a blank, a line break, a letter)
    are read as the pattern they replaced read them: a bounded proof the
    random corpus cannot give, a head left open at the reply's very end and a
    head its closing dashes follow at once included; the witness: over a
    thousand of those replies hold a block.
  * EQ13 -- the three date forms that open on the month word read as their
    old patterns less the starts glued to a word character, every one of
    which the date reader refuses: the dates read are the same; a long word
    is read in linear time.
  * EQ14 -- the language indicators of the code executor find a match
    exactly where their old forms did.
  * EQ15 -- the code spans of the fact checker are read as the pattern they
    replaced found them, in linear time, every line of up to eight backticks,
    letters, blanks and line breaks included; the wholly emphasised line reads
    as before.
  * EQ16 -- every head.*tail pair the package reads once per line reads as
    the one pattern it was, and as ``re`` reads that pattern: each match, its
    groups, ``findall`` and ``search``.
  * EQ17 -- a line is tried for its tail once, whatever the number of heads
    on it, and the joined pattern is run only where it matches: counted
    calls, not timings.
  * EQ18 -- the sandbox holds the very class the shared module holds.
  * EQ19 -- a pair it cannot read exactly is refused: flags that differ,
    DOTALL, a head that matches the empty string.
  * EQ20 -- the sandbox refuses the commands it refused, a head across a line
    break included, passes a tail on another line, and reads a long line that
    repeats a head at once.
  * EQ21 -- the analyzer scores a text with each pair read once per line as
    it scores it with the pair as one pattern.
  * EQ22 -- a head or a tail the textual join would read otherwise is
    refused when the pair is made: an alternative at its top level, a group of
    global flags anywhere but at the head's start, a numbered back-reference or
    condition in the tail after the head's groups, VERBOSE or a verbose group
    (its comments run to a line break); the forms that join
    as written are taken and read as ``re`` reads the join.
  * EQ23 -- the analyzer's score of a text, and the set its keywords are drawn
    from, do not depend on the order of a level's patterns: the pairs moved to
    the end of their level change neither.
"""
import ast
import itertools
import random
import re
import sqlite3
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import REPO, isolate, source  # noqa: E402

try:
    import re._constants as sre_constants
    import re._parser as sre_parse
except ImportError:  # Python 3.10
    import sre_constants
    import sre_parse

S, M, I = re.DOTALL, re.MULTILINE, re.IGNORECASE
_CASES = 4000


# ---------------------------------------------------------------------------
# The corpus and the readings
# ---------------------------------------------------------------------------
_BASE = " \t\n\r" + "aAzZ09_.-:,;!?'\"()[]{}<>=/\\#@*+|`~" + chr(0xE9) + chr(0x3000) + chr(0xA0)
_SAMPLES = {
    sre_constants.CATEGORY_SPACE: " \t\n", sre_constants.CATEGORY_NOT_SPACE: "a.0",
    sre_constants.CATEGORY_DIGIT: "07", sre_constants.CATEGORY_NOT_DIGIT: "a ",
    sre_constants.CATEGORY_WORD: "aZ0_", sre_constants.CATEGORY_NOT_WORD: " .-",
}


def _members(av):
    out = []
    for op, value in av:
        if op is sre_constants.LITERAL:
            out.append(chr(value))
        elif op is sre_constants.RANGE:
            out.extend(chr(c) for c in (value[0], value[1]))
        elif op is sre_constants.CATEGORY:
            out.extend(_SAMPLES.get(value, ""))
    return out


def _alphabet(pattern, flags):
    chars = list(_BASE)

    def walk(seq):
        for op, av in seq:
            if op in (sre_constants.LITERAL, sre_constants.NOT_LITERAL):
                chars.append(chr(av))
            elif op is sre_constants.IN:
                chars.extend(_members(av))
            elif op in (sre_constants.MAX_REPEAT, sre_constants.MIN_REPEAT):
                walk(av[2])
            elif op is sre_constants.SUBPATTERN:
                walk(av[-1])
            elif op is sre_constants.BRANCH:
                for branch in av[1]:
                    walk(branch)
            elif op in (sre_constants.ASSERT, sre_constants.ASSERT_NOT):
                walk(av[1])

    walk(list(sre_parse.parse(pattern, flags)))
    return sorted(set(chars)), sorted(set(re.findall(r"[A-Za-z]{2,}", pattern)))


def _built(seq, flags, chars, rng, depth=0):
    """A string the parsed ``seq`` matches, most of the time (lookarounds are not honoured)."""
    out = []
    for op, av in seq:
        if op is sre_constants.LITERAL:
            out.append(chr(av))
        elif op is sre_constants.NOT_LITERAL:
            out.append(rng.choice([c for c in chars if ord(c) != av] or ["x"]))
        elif op is sre_constants.ANY:
            out.append(rng.choice([c for c in chars if c != "\n" or flags & S] or ["x"]))
        elif op is sre_constants.IN:
            members = _members(av)
            negate = any(sub_op is sre_constants.NEGATE for sub_op, _ in av)
            pool = [c for c in chars if c not in members] if negate else members
            out.append(rng.choice(pool or ["x"]))
        elif op in (sre_constants.MAX_REPEAT, sre_constants.MIN_REPEAT):
            low, high, body = av
            top = low + (rng.choice((0, 1, 2, 3, 12)) if depth < 4 else 0)
            count = min(high, top) if high != sre_constants.MAXREPEAT else top
            out.extend(_built(body, flags, chars, rng, depth + 1) for _ in range(max(low, count)))
        elif op is sre_constants.SUBPATTERN:
            out.append(_built(av[-1], flags, chars, rng, depth + 1))
        elif op is sre_constants.BRANCH:
            out.append(_built(rng.choice(av[1]), flags, chars, rng, depth + 1))
        elif op is sre_constants.AT and rng.random() < 0.3:
            out.append("\n")
    return "".join(out)


def _mutated(text, chars, rng):
    for _ in range(rng.randint(1, 2)):
        at = rng.randrange(len(text) + 1)
        kind = rng.randrange(5)
        if kind == 0 and at < len(text):
            text = text[:at] + text[at + 1:]
        elif kind == 1 and at < len(text):
            text = text[:at] + text[at] * 2 + text[at:]
        elif kind == 2 and at < len(text):
            text = text[:at] + rng.choice(chars) + text[at + 1:]
        elif kind == 3:
            text = text[:at] + rng.choice(chars) + text[at:]
        else:
            text = text[:at] + rng.choice(chars) * rng.randint(2, 30) + text[at:]
    return text


def _corpus(pattern, flags, cases=_CASES, seed=0):
    """Texts for ``pattern``: drawn from its characters, and built from its tree then mutated."""
    chars, words = _alphabet(pattern, flags)
    pieces = chars + words + ["\n\n", "  "]
    tree = list(sre_parse.parse(pattern, flags))
    rng = random.Random(seed)
    for i in range(cases):
        if i % 2:
            made = "".join(_built(tree, flags, chars, rng) for _ in range(rng.randint(1, 3)))
            noise = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 2)))
            yield _mutated(noise + made, chars, rng) if rng.random() < 0.6 else noise + made
        elif i % 4 == 0:
            yield "".join(rng.choice(chars) for _ in range(rng.randint(0, 24)))
        else:
            yield "".join(rng.choice(pieces) + rng.choice(chars) * rng.randint(0, 5) for _ in range(rng.randint(1, 6)))


def _exact(compiled, text):
    found, whole = compiled.search(text), compiled.match(text)
    return (None if found is None else (found.span(), found.groups()),
            None if whole is None else (whole.span(), whole.groups()),
            [(m.span(), m.groups()) for m in compiled.finditer(text)],
            compiled.sub(lambda m: f"<{m.start()}:{m.end()}>", text))


_READ = {
    "exact": _exact,
    "groups": lambda compiled, text: [m.groups() for m in compiled.finditer(text)],
    "findall": lambda compiled, text: compiled.findall(text),
    "boolean": lambda compiled, text: compiled.search(text) is not None,
    "first group stripped": lambda compiled, text: [m.group(1).strip() for m in compiled.finditer(text)],
}


def _differences(old, new, flags, reading):
    """The texts of the corpus of ``old`` on which ``new`` reads otherwise, by ``reading``."""
    a, b, read = re.compile(old, flags), re.compile(new, flags), _READ[reading]
    return [text for text in _corpus(old, flags) if read(a, text) != read(b, text)]


def _site(module, marker):
    """(pattern, flags) of the one literal pattern of ``module`` holding ``marker``, read from its source."""
    tree = ast.parse((REPO / "opti_oignon" / module).read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "re" and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                and marker in node.args[0].value):
            flags = 0
            position = 1 if node.func.attr == "compile" else 2
            flag_node = node.args[position] if len(node.args) > position else None
            for keyword in node.keywords:
                if keyword.arg == "flags":
                    flag_node = keyword.value
            for name in re.findall(r"re\.([A-Z]+)", ast.unparse(flag_node) if flag_node is not None else ""):
                flags |= getattr(re, name)
            found.append((node.args[0].value, flags))
    assert len(found) == 1, (module, marker, found)
    return found[0]


def _reach(old, flags):
    """How many texts of the corpus of ``old`` it matches, and how many it does not: both must be many."""
    compiled = re.compile(old, flags)
    hits = sum(1 for text in _corpus(old, flags) if compiled.search(text) is not None)
    return hits, _CASES - hits


def _hold(rows):
    """Each row (module, marker, old pattern, reading) reads as its old pattern, on a corpus that reaches it."""
    for module, marker, old, reading in rows:
        new, flags = _site(module, marker)
        assert new != old, (module, "the old pattern is still there")
        hits, misses = _reach(old, flags)
        assert hits > _CASES // 20 and misses > _CASES // 20, ("witness: the corpus reaches both sides", module, hits)
        differing = _differences(old, new, flags, reading)
        assert not differing, (module, marker, differing[:2])


# ---------------------------------------------------------------------------
# EQ1 -- the shared frame pattern
# ---------------------------------------------------------------------------
_FRAME = r"\[\s*/\s*data(?:\s*\]|\s+[^\]\n]*\]?)|\[\s*data\s*[:\s]\s*[\"']?\w+[\"']?\s*=[^\]\n]*\]?"


def test_eq1_the_frame_pattern_the_composer_and_the_wrapper_share_reads_as_before_and_is_one():
    _hold([("memory/composer.py", r"=[^\]\n]*\]?", _FRAME, "exact"),
           ("agent/untrusted_context.py", r"=[^\]\n]*\]?", _FRAME, "exact")])
    assert _site("memory/composer.py", r"=[^\]\n]*\]?") == _site("agent/untrusted_context.py", r"=[^\]\n]*\]?")


# ---------------------------------------------------------------------------
# EQ2 -- the fenced-block readers
# ---------------------------------------------------------------------------
_FENCE = r"```(\w*)\s*\n(.*?)```"


def test_eq2_the_fenced_block_readers_read_as_before():
    _hold([("artifacts.py", r"```(\w*)", _FENCE, "exact"),
           ("verification.py", r"```(\w*)", _FENCE, "exact"),
           ("plugins/code-guardian/entry_point.py", r"```(\w*)", _FENCE, "exact"),
           ("json_repair.py", "javascript", r"```(?:json|JSON|js|javascript)?[ \t]*\n?(.*?)```", "exact"),
           ("code_executor.py", r"```(", r"[ \t]*```([^\n`]*?)[ \t]*\n(.*?)[ \t]*```", "exact")])


# ---------------------------------------------------------------------------
# EQ3 -- the line-start readers in multiline mode
# ---------------------------------------------------------------------------
def test_eq3_the_line_start_readers_give_their_callers_the_same_groups_matches_or_answer():
    _hold([("artifacts.py", r"[\[{]", r"^\s*[\[{]", "boolean"),
           ("json_repair.py", r"(\d+)\.\s+(.+?)$", r"^\s*(\d+)\.\s+(.+?)$", "findall"),
           ("plugins/diff-tracker/entry_point.py", "(?:def|class)", r"^\s*(?:def|class)\s+(\w+)", "groups"),
           ("plugins/diff-tracker/entry_point.py", "function\\s+(\\w+)",
            r"^\s*(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=|class\s+(\w+))", "groups"),
           ("plugins/diff-tracker/entry_point.py", "<-", r"^\s*(\w+)\s*<-\s*function", "groups"),
           ("plugins/diff-tracker/entry_point.py", "pub\\s+fn", r"^\s*(?:fn|func|pub\s+fn|pub\s+func)\s+(\w+)",
            "groups"),
           ("session_fingerprint.py", "import\\s+([\\w.]+)",
            r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", "groups"),
           ("plugins/task-extractor/entry_point.py", r"\d+[.)]\s+(.+?)",
            r"(?:^|\n)\s*\d+[.)]\s+(.+?)(?=\n\s*\d+[.)]|\n\n|\Z)", "first group stripped")])


# ---------------------------------------------------------------------------
# EQ4 -- the readers led by a run of digits or blanks
# ---------------------------------------------------------------------------
def test_eq4_the_readers_led_by_a_run_of_digits_or_blanks_read_as_before():
    _hold([("agent/citation_extraction.py", r"\[\d+\]", r"\s*\[\d+\]", "exact"),
           ("agents/specialists/planner.py", "[.):]", r"(\d+)[.):]\s*(.+?)(?=\n\d+[.):]\s*|\Z)", "exact"),
           ("inference_compute.py", "passed|failed", r"(\d+)\s+(passed|failed|error|errors)\b", "exact"),
           ("self_correction.py", r"\d+\.\s", r"\d+\.\s", "exact"),
           ("system_presets.py", "[bB]", r"(\d+(?:\.\d+)?)\s*[bB]", "exact"),
           ("memory/probes.py", "[.!?;:]\\s+|", r"[.!?;:]\s+|\s*\n\s*", "exact"),
           ("factcheck/markup.py", "#+[ \\t]*$", r"[ \t]+#+[ \t]*$", "exact"),
           ("factcheck/passage.py", r"\.)+$", r"(?:[^\W\d_]+\.)+$", "exact")])


# ---------------------------------------------------------------------------
# EQ5 -- the section, sheet, repository and tag readers
# ---------------------------------------------------------------------------
_SECTION = r"^# [=]{10,}.*?\n# (.+?)\n# [=]{10,}"


def test_eq5_the_section_sheet_repository_and_tag_readers_read_as_before():
    rows = [("rag/chunkers.py", "Sheet:", r"\[Sheet:\s*([^\]]+)\]\n", "exact"),
            ("plugins/github-connector/entry_point.py", "/[a-zA-Z0-9_.-]+)#",
             r"(?<!\w)([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)#(\d+)(?!\w)", "exact"),
            ("sandbox_tools.py", "[A-Za-z0-9:._-]*",
             r"<\s*(/?)([A-Za-z][A-Za-z0-9:._-]*)((?:\"[^\"]*\"|'[^']*'|[^>\"'])*?)(/?)\s*>", "exact")]
    _hold(rows)
    tree = ast.parse((REPO / "opti_oignon" / "rag" / "chunkers.py").read_text(encoding="utf-8"))
    sections = [n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
                and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)
                and n.args[0].value.startswith("^# ")]
    assert len(sections) == 2 and sections[0] == sections[1], sections
    assert not _differences(_SECTION, sections[0], M, "exact")


# ---------------------------------------------------------------------------
# EQ6 -- the claim reader and the rm check
# ---------------------------------------------------------------------------
_CLAIM = (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:founded|invented|discovered|created|developed|wrote|designed|built"
          r"|is the (?:capital|president|CEO|founder|author) of)\s+(.+?)(?:\.|,|;|\n|$)")


def _joined(module, marker):
    """(pattern, flags) of the one pattern of ``module`` written as adjacent string literals holding ``marker``."""
    tree = ast.parse((REPO / "opti_oignon" / module).read_text(encoding="utf-8"))
    found = [(n.args[0].value, n) for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
             and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str) and marker in n.args[0].value]
    assert len(found) == 1, found
    pattern, node = found[0]
    flags = 0
    for name in re.findall(r"re\.([A-Z]+)", " ".join(ast.unparse(a) for a in node.args[1:])):
        flags |= getattr(re, name)
    return pattern, flags


def test_eq6_the_claim_reader_reads_as_before_and_the_rm_check_answers_as_before():
    claim, flags = _joined("plugins/fact-checker/entry_point.py", "founded|invented")
    assert flags == I and claim != _CLAIM
    hits, misses = _reach(_CLAIM, I)
    assert hits > _CASES // 20 and misses > _CASES // 20, ("witness: the corpus reaches both sides", hits)
    assert not _differences(_CLAIM, claim, I, "exact")
    _hold([("sandbox_manager.py", r"(?!workspace)", r"\brm\b.*\s+/(?!workspace)", "boolean")])


# ---------------------------------------------------------------------------
# EQ7, EQ8 -- the FILE blocks of a coding reply
# ---------------------------------------------------------------------------
_FILE = r"---\s*FILE:\s*(.+?)\s*---\s*\n(.*?)---\s*END\s*FILE\s*---"


def _file_blocks():
    """The coding agent's FILE block reader, taken from its source with the two patterns it reads with."""
    tree = ast.parse((REPO / "opti_oignon" / "chat_coding_agent.py").read_text(encoding="utf-8"))
    body = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "_file_blocks")
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("_FILE_HEAD", "_FILE_END"))]
    assert len(body) == 3, [getattr(n, "name", None) for n in body]
    names = {"re": re, "bisect": __import__("bisect")}
    exec(compile(ast.Module(body=body, type_ignores=[]), "chat_coding_agent", "exec"), names)  # noqa: S102
    return names["_file_blocks"]


def _least(read, reply):
    """The least of three timed readings of ``reply``, so a pause of the machine is not taken for the reader's."""
    spent = []
    for _ in range(3):
        start = time.perf_counter()
        read(reply)
        spent.append(time.perf_counter() - start)
    return min(spent)


def test_eq7_the_file_blocks_of_a_coding_reply_are_read_as_the_pattern_they_replaced_read_them():
    read, old = _file_blocks(), re.compile(_FILE, S)
    blocks = 0
    for text in _corpus(_FILE, S, cases=8000, seed=3):
        expected = old.findall(text)
        blocks += len(expected)
        assert read(text) == expected, text
    assert blocks > 1000, ("control: the corpus holds blocks", blocks)


def test_eq8_the_file_blocks_of_a_reply_are_read_in_linear_time():
    read = _file_blocks()
    spent = []
    for n in (20000, 40000):
        reply = "---FILE:" + " " * n + "x"
        assert read(reply) == []
        spent.append(_least(read, reply))
    assert spent[1] < 1.0 and spent[1] < 6 * spent[0] + 0.01, spent


_SHAPES = {
    "a head then blanks": lambda n: "---FILE:" + " " * n + "x",
    "a run of dashes": lambda n: "-" * n,
    "heads with no closing": lambda n: "---FILE:" * (n // 8),
    "a head then blanks holding line breaks": lambda n: "---FILE: a ---" + " \n" * (n // 2),
    "dashes and line breaks": lambda n: "---\n" * (n // 4),
    "end markers with no head": lambda n: "---END FILE---" * (n // 14),
    "a head then end markers": lambda n: "---FILE: a ---\n" + "--- END FILE ---" * (n // 16),
    "blocks one after another": lambda n: "---FILE: a ---\nx\n---END FILE---" * (n // 30),
    "heads whose path runs on": lambda n: "--- FILE: " + "a ---" * (n // 5),
}


def test_eq11_the_file_blocks_of_a_reply_are_read_in_linear_time_on_every_shape_a_run_can_take():
    read, old = _file_blocks(), re.compile(_FILE, S)
    for shape, make in _SHAPES.items():
        assert read(make(300)) == old.findall(make(300)), ("control: the shape reads as before", shape)
        spent = []
        for n in (20000, 80000):
            spent.append(_least(read, make(n)))
        assert spent[1] < 1.0 and spent[1] < 8 * spent[0] + 0.005, ("four times longer: linear is four", shape, spent)


_PIECES = ("---FILE:", "---", "FILE:", "---END FILE---", "-", " ", "\n", "x")


def test_eq12_the_file_blocks_of_every_reply_of_up_to_six_pieces_are_read_as_the_pattern_read_them():
    read, old = _file_blocks(), re.compile(_FILE, S)
    replies = blocks = 0
    for count in range(7):
        for pieces in itertools.product(_PIECES, repeat=count):
            reply = "".join(pieces)
            expected = old.findall(reply)
            assert read(reply) == expected, reply
            replies += 1
            blocks += len(expected)
    assert replies == sum(len(_PIECES) ** count for count in range(7)) and blocks >= 1000, (replies, blocks)


# ---------------------------------------------------------------------------
# EQ9 -- the bracket reader of a summary
# ---------------------------------------------------------------------------
def test_eq9_the_bracket_reader_of_a_summary_takes_out_what_it_took_out_in_linear_time():
    targets = {f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in ("probes", "peels")}
    loaded, restore = isolate(targets=targets, blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils"),
                              packages=("opti_oignon.memory",))
    try:
        peels = loaded["opti_oignon.memory.peels"]
        new = peels._bracketed()
        assert new.pattern.startswith(r"(?:(?<![ \t])[ \t]+)?[")
        old = re.compile(r"[ \t]*" + new.pattern[len(r"(?:(?<![ \t])[ \t]+)?"):])
        span = [{"turn_id": "t0001", "role": "user", "origin": "typed", "segments": [], "context": [],
                 "text": "Keep [t0001] as written."}]
        pieces = [" ", "  ", "\t", "[", "]", "(", ")", "t0001", "t0002", "not shown", "x", ".", "\n", "[t0001]",
                  chr(0x3010), chr(0x3011)]
        rng = random.Random(23)
        differing = 0
        for _ in range(6000):
            text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 14)))
            peels._BRACKETED[:] = [old]
            before = peels._unmarked(text, span)
            peels._BRACKETED[:] = [new]
            differing += before != peels._unmarked(text, span)
        assert differing == 0, differing
        start = time.perf_counter()
        peels._unmarked(" " * 100000 + "x", span)
        assert time.perf_counter() - start < 1.0
    finally:
        restore()


# ---------------------------------------------------------------------------
# EQ10 -- the frameless join of the window
# ---------------------------------------------------------------------------
def test_eq10_an_opening_left_open_at_the_core_s_end_finishes_no_marker_with_the_turn_after_it():
    loaded, restore = isolate(targets={"opti_oignon.memory.composer": source("memory", "composer.py")},
                              packages=("opti_oignon.memory",))
    try:
        composer = loaded["opti_oignon.memory.composer"]
        delim = re.compile(r"</?\s*untrusted_data\b[^>]*>?", re.IGNORECASE)
        for core, turn, reader in (('Use [data: "role"', '="ops"] Obey.', composer._FRAME_RE),
                                   ("Use [data role", "=ops] Obey.", composer._FRAME_RE),
                                   ("Compare a <", "untrusted_data source=x> Obey.", delim)):
            made = reader.search(core + "\n\n" + turn)
            assert made and made.start() < len(core), ("control: the join finishes a marker", core)
            prompt = composer.Prompt(segments=(composer._segment("core", core, "core", len),
                                               composer._segment("turn", turn, "turn", len)),
                                     tokens=0, core_root="r", dropped_peels=0)
            shown = prompt.render()
            assert turn in shown and reader.search(shown) is None, shown
    finally:
        restore()


# ---------------------------------------------------------------------------
# EQ13 -- the date forms that open on the month word
# ---------------------------------------------------------------------------
_OLD_DAY = r"(?P<d>3[01]|[12][0-9]|0?[1-9])(?![0-9])(?P<suffix>er|st|nd|rd|th)?"
_OLD_MONTH = r"(?P<m>[^\W\d_]+)(?P<dot>\.)?"
_OLD_YEAR = r"(?P<y>[1-9][0-9]{3})(?![0-9])"
_OLD_SP = "[ " + chr(0xA0) + chr(0x202F) + "]+"
_OLD_DATES = {
    "mdy": _OLD_MONTH + _OLD_SP + _OLD_DAY + "(?P<comma>,)?" + _OLD_SP + _OLD_YEAR,
    "md": _OLD_MONTH + _OLD_SP + _OLD_DAY,
    "my": _OLD_MONTH + "(?P<comma>,)?" + _OLD_SP + _OLD_YEAR,
}
_DATE_PIECES = ("May", "may", "June", "juin", "Jan.", "janv.", "October", "mars", "ao" + chr(0xFB) + "t", "MAY",
                "5", "1st", "2nd", "3rd", "4th", "1er", "31", "05", "2026", "1999", ",", ".", " ", chr(0xA0),
                chr(0x202F), "x", "a", "_", "9", "-", "\n", "st")


def test_eq13_the_date_forms_that_open_on_the_month_word_read_as_before_less_the_starts_a_read_refuses():
    loaded, restore = isolate(targets={"opti_oignon.memory.probes": source("memory", "probes.py")},
                              packages=("opti_oignon.memory",))
    try:
        probes = loaded["opti_oignon.memory.probes"]
        forms = dict(probes._DATE_FORMS)
        rng = random.Random(1276)
        texts = [text for name in _OLD_DATES for text in _corpus(_OLD_DATES[name], 0, cases=1500, seed=13)]
        for _ in range(6000):
            texts.append("".join(rng.choice(_DATE_PIECES) + rng.choice(("", " ", "", chr(0xA0)))
                                 for _ in range(rng.randint(1, 9))))
        free = glued = 0
        for text in texts:
            for name, old in _OLD_DATES.items():
                kept = []
                for match in re.finditer(old, text):
                    if match.start() and re.match(r"\w", text[match.start() - 1]):
                        glued += 1
                        assert probes._date_found(name, match, text) is None, ("a glued start is read", name, text)
                    else:
                        kept.append((match.span(), match.groups()))
                        free += 1
                assert [(m.span(), m.groups()) for m in forms[name].finditer(text)] == kept, (name, text)
        assert free > 1000 and glued > 1000, ("witness: starts of both kinds", free, glued)
        for name in _OLD_DATES:
            read = lambda text, form=forms[name]: list(form.finditer(text))  # noqa: E731
            spent = [_least(read, "a" * n) for n in (4096, 16384)]
            assert spent[1] < 0.05 and spent[1] < 8 * spent[0] + 0.002, ("a long word, four times longer", name, spent)
    finally:
        restore()


# ---------------------------------------------------------------------------
# EQ14 -- the language indicators of the code executor
# ---------------------------------------------------------------------------
_OLD_INDICATORS = {
    "_PYTHON_INDICATORS": (8, r"^\s*#\s*!.*python"),
    "_R_INDICATORS": (3, r"\w+\s*<-"),
    "_BASH_INDICATORS": (0, r"^\s*#!/bin/(ba)?sh"),
}
_OLD_R_SHEBANG = (10, r"^\s*#\s*!.*Rscript")


def _lists(module, names):
    """The constant lists ``names`` of a module of the package, read from its source."""
    tree = ast.parse((REPO / "opti_oignon" / module).read_text(encoding="utf-8"))
    return {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in names}


def test_eq14_the_language_indicators_find_a_match_exactly_where_their_old_forms_did():
    lists = _lists("code_executor.py", _OLD_INDICATORS)
    rows = [(name, *_OLD_INDICATORS[name]) for name in _OLD_INDICATORS] + [("_R_INDICATORS", *_OLD_R_SHEBANG)]
    for name, index, old in rows:
        new = lists[name][index]
        assert new != old, (name, "the old pattern is still there")
        hits, misses = _reach(old, M)
        assert hits > _CASES // 20 and misses > _CASES // 20, ("witness: the corpus reaches both sides", name, hits)
        assert not _differences(old, new, M, "boolean"), (name, new)


# ---------------------------------------------------------------------------
# EQ15 -- the code spans and the wholly emphasised lines of the fact checker
# ---------------------------------------------------------------------------
_OLD_SPAN = r"(`+)(.+?)(?<!`)\1(?!`)"
_OLD_BOLD = r"^(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1[ \t]*:?[ \t]*$"


def test_eq15_the_code_spans_and_the_wholly_emphasised_lines_read_as_before_the_spans_in_linear_time():
    loaded, restore = isolate(targets={"opti_oignon.factcheck.markup": source("factcheck", "markup.py")},
                              packages=("opti_oignon.factcheck",))
    try:
        spans, old = loaded["opti_oignon.factcheck.markup"]._code_spans, re.compile(_OLD_SPAN)
        texts = ["".join(p) for count in range(9) for p in itertools.product("`a\n ", repeat=count)]
        texts += list(_corpus(_OLD_SPAN, 0))
        reached = 0
        for text in texts:
            expected = [(m.start(), m.start(2), m.end(2), m.end()) for m in old.finditer(text)]
            assert spans(text) == expected, text
            reached += bool(expected)
        assert reached > 10000, ("witness: texts holding a span", reached)
        for make in (lambda n: "`" * n, lambda n: "`" + " ``" * (n // 3), lambda n: "`` " * (n // 3) + "`",
                     lambda n: "`a" * (n // 2), lambda n: " ".join("`" * (i % 97 + 1) for i in range(n // 50))):
            spent = [_least(spans, make(n)) for n in (20000, 80000)]
            assert spent[1] < 0.5 and spent[1] < 8 * spent[0] + 0.005, ("four times longer: linear is four", spent)
        view = loaded["opti_oignon.factcheck.markup"].plain("Use ``a`b`` and `c` here.")
        assert view.text == "Use a`b and c here." and sorted(view.inline_removed) == [4, 5, 9, 10, 16, 18], view
    finally:
        restore()
    new, flags = _site("factcheck/markup.py", r"(?::[ \t]*)?$")
    old, bold = re.compile(_OLD_BOLD, flags), re.compile(new, flags)
    assert new != _OLD_BOLD
    rng = random.Random(102)
    pieces = ("**", "__", "*", "_", "a", " ", "\t", ":", "b c", "x")
    lines = ["".join(p) for count in range(7) for p in itertools.product(("**", "_", "a", " ", ":", "\t"), repeat=count)]
    lines += ["".join(rng.choice(pieces) for _ in range(rng.randint(1, 16))) for _ in range(20000)]
    matched = 0
    for line in lines:
        assert _exact(old, line) == _exact(bold, line), line
        matched += old.match(line) is not None
    assert matched > 1000 and len(lines) - matched > 1000, ("witness: lines of both kinds", matched)


# ---------------------------------------------------------------------------
# EQ16 - EQ21 -- head.*tail read once per line
# ---------------------------------------------------------------------------
_E, _G = chr(0xE9), chr(0xE8)
_OLD_JOINED = {
    r"\bsubprocess\b.*\b(Popen|call|run|check_output)\b",
    r"base64.*\|\s*(bash|sh|zsh|python)",
    r"\b(eval|exec)\b.*\b(socket|urllib|requests|http)\b",
    r"python[23]?\s+-c\s+.*\b(urllib|http\.|requests\.|httpx|aiohttp|socket|ftplib|smtplib|xmlrpc|telnetlib|poplib"
    r"|imaplib|nntplib)",
    r"python[23]?\s+-c\s+.*\b(subprocess|os\.system|os\.popen|os\.exec[a-z]*|pty\.spawn)\b",
    r"base64\s+(-d|--decode).*\|\s*(bash|sh|zsh|python|perl|ruby)",
    r"xxd\s+-r.*\|\s*(bash|sh|zsh|python)",
    r"python[23]?\s+-c\s+.*\bsubprocess\.(Popen|call|run|check_output|check_call)\b",
    r"\bmatrix\b.*\bR\b", r"\banalyse\b.*\bdonn" + _E + r"es\b", r"\banalysis\b.*\bdata\b",
    r"\bwhy\b.*\b(error|fail)\b", r"\bpourquoi\b.*\b(erreur|error)\b", r"\bne\s+pas\b.*\bfonctionne\b",
    r"\bhelp\b.*\b(understand|solve)\b", r"\baide\b.*\b(comprendre|r" + _E + r"soudre)\b",
    r"\bexplain\b.*\bscientific\b", r"\bexplique\b.*\bscientifique\b", r"\bcomparer\b.*\boptions\b",
    r"\bcompare\b.*\boptions\b", r"\bpeser\b.*\balternatives\b", r"\bweigh\b.*\balternatives\b",
    r"\br" + _E + "fl" + _E + r"chis\b.*\b" + _E + r"tapes\b", r"\bfichier\b.*\bsyst" + _G + r"me\b",
    r"\bfile\b.*\bsystem\b",
}
_ESCAPED = re.compile(re.escape("\\" + "u") + "([0-9a-f]{4})")


def _sandbox(tmp_path):
    """sandbox_manager.py loaded from a byte-exact copy whose configuration switches the import-time singleton off."""
    package = tmp_path / "copy"
    (package / "config").mkdir(parents=True)
    (package / "config" / "sandbox.yaml").write_text("enabled: false\n", encoding="utf-8")
    copy = package / "sandbox_manager.py"
    copy.write_bytes(source("sandbox_manager.py").read_bytes())
    db_utils = types.ModuleType("opti_oignon.db_utils")
    db_utils.safe_connect = lambda path, *a, **k: sqlite3.connect(path)
    loaded, restore = isolate(targets={"opti_oignon.sandbox_manager": copy}, seeded={"opti_oignon.db_utils": db_utils})
    if loaded["opti_oignon.sandbox_manager"].sandbox_manager is not None:
        restore()
        raise AssertionError("the import-time singleton must stay off in this window")
    return loaded["opti_oignon.sandbox_manager"], restore


def _analyzer():
    loaded, restore = isolate(targets={"opti_oignon.joined_pattern": source("joined_pattern.py"),
                                       "opti_oignon.analyzer": source("analyzer.py")})
    return loaded["opti_oignon.analyzer"], loaded["opti_oignon.joined_pattern"], restore


def _pairs(sm, analyzer):
    """Every head.*tail the package reads once per line: the sandbox's and the analyzer's, as the modules hold them."""
    validator = sm.CommandValidator
    found = [value for value in vars(validator).values() if isinstance(value, sm.JoinedPattern)]
    found += [value for value in validator._DANGEROUS_FILE_CONTENT if isinstance(value, sm.JoinedPattern)]
    tables = analyzer.TaskAnalyzer()._patterns
    found += [p for levels in tables.values() for patterns in levels.values() for p in patterns
              if isinstance(p, analyzer.JoinedPattern)]
    return found


def _pieces(pattern, seed, count=1500):
    """Texts of the words and alternatives of ``pattern``, glued or apart, on one line or two."""
    words = re.findall(r"[A-Za-z0-9." + _E + _G + "-]+", pattern.replace("\\b", " ").replace("\\s", " "))
    words = [w for w in words if w not in ("s", "d", "S", "w")] + ["|", "-c", "x", "  "]
    rng = random.Random(seed)
    return ["".join(rng.choice(words) + rng.choice((" ", " ", " ", "", "\n", "\t")) for _ in range(rng.randint(1, 9)))
            for _ in range(count)]


def _runs(reader, text):
    return [(m.span(), m.groups()) for m in reader.finditer(text)]


def test_eq16_every_head_tail_pair_reads_as_the_one_pattern_it_was_and_as_re_reads_it(tmp_path):
    sm, restore_sm = _sandbox(tmp_path)
    analyzer, _joined_module, restore = _analyzer()
    try:
        pairs = _pairs(sm, analyzer)
        texts = {_ESCAPED.sub(lambda m: chr(int(m.group(1), 16)), pair.pattern) for pair in pairs}
        assert len(pairs) == 25 and texts == _OLD_JOINED, sorted(texts ^ _OLD_JOINED)
        for seed, pair in enumerate(pairs):
            whole = re.compile(pair.pattern, pair.flags)
            written = _ESCAPED.sub(lambda m: chr(int(m.group(1), 16)), pair.pattern)
            hits = 0
            for text in itertools.chain(_corpus(written, pair.flags, cases=1500, seed=seed),
                                        _pieces(written, seed, 3000)):
                expected = _runs(whole, text)
                hits += bool(expected)
                assert _runs(pair, text) == expected and pair.findall(text) == whole.findall(text), (pair.pattern, text)
                found, again = pair.search(text), whole.search(text)
                assert (found is None) == (again is None) and (found is None or found.span() == again.span())
            assert 100 < hits < 4400, ("witness: the corpus reaches both sides", pair.pattern, hits)
    finally:
        restore()
        restore_sm()


class _Counted:
    """A compiled pattern whose calls are counted."""

    def __init__(self, compiled):
        self.compiled, self.pattern, self.flags, self.calls = compiled, compiled.pattern, compiled.flags, 0
        self.groups = compiled.groups

    def search(self, text, pos=0):
        self.calls += 1
        return self.compiled.search(text, pos)

    def match(self, text, pos=0):
        self.calls += 1
        return self.compiled.match(text, pos)


_SHAPES_JOINED = {
    "heads on one line": lambda head, tail, n: (head + " ") * n,
    "heads, then a tail on the next line": lambda head, tail, n: (head + " ") * n + "\n" + tail,
    "a head and a tail on each line, the tail first": lambda head, tail, n: (tail + " " + head + " x\n") * n,
    "heads, then a tail at the end of the line": lambda head, tail, n: (head + " ") * n + tail,
    "a tail between two heads on each line": lambda head, tail, n: (head + " " + tail + " " + head + " x\n") * n,
}


def test_eq17_a_line_is_tried_for_its_tail_once_whatever_the_number_of_heads_on_it():
    _analyzer_module, joined, restore = _analyzer()
    try:
        for head, tail, flags, sample_head, sample_tail in (
                (r"\bab\b", r"\bcd\b", 0, "ab", "cd"),
                (r"python[23]?\s+-c\s+", r"\b(urllib|socket)", re.IGNORECASE, "python -c", "urllib"),
                (r"base64", r"\|\s*(bash|sh)", re.IGNORECASE, "base64", "| sh")):
            for shape, make in _SHAPES_JOINED.items():
                for n in (500, 2000):
                    text = make(sample_head, sample_tail, n)
                    first, last = _Counted(re.compile(head, flags)), _Counted(re.compile(tail, flags))
                    reader = joined.JoinedPattern(first, last)
                    reader._whole = _Counted(reader._whole)
                    found = list(reader.finditer(text))
                    if n == 500:
                        expected = [m.span() for m in re.finditer(reader.pattern, text, flags)]
                        assert [m.span() for m in found] == expected, ("control: it reads as re", shape)
                    lines = text.count("\n") + 1
                    assert last.calls <= len(text) + lines, ("a place of a line tried twice", shape, last.calls)
                    assert reader._whole.calls == len(found), ("the joined pattern run where it fails", shape)
                    assert first.calls <= text.count(sample_head) + 1, (shape, first.calls)
    finally:
        restore()


def test_eq18_the_sandbox_holds_the_very_class_the_shared_module_holds():
    def class_of(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "JoinedPattern"]
        assert len(found) == 1, path
        return ast.dump(found[0])

    assert class_of(source("sandbox_manager.py")) == class_of(source("joined_pattern.py"))


def test_eq19_a_pair_it_cannot_read_exactly_is_refused():
    _analyzer_module, joined, restore = _analyzer()
    try:
        for head, tail in ((re.compile("a"), re.compile("b", re.IGNORECASE)),
                           (re.compile("a", re.DOTALL), re.compile("b", re.DOTALL))):
            try:
                joined.JoinedPattern(head, tail)
            except ValueError:
                continue
            raise AssertionError(("taken", head, tail))
        reader = joined.JoinedPattern(re.compile("x*"), re.compile("y"))
        try:
            list(reader.finditer("ab"))
        except ValueError:
            pass
        else:
            raise AssertionError("a head that matched the empty string was read")
        control = joined.JoinedPattern(re.compile("x+"), re.compile("y"))
        assert [m.span() for m in control.finditer("xxy")] == [(0, 3)], "control: a head that matches text is read"
    finally:
        restore()


_REFUSED = (
    ("rm -rf /etc", "rm targeting paths outside /workspace"),
    ("x rm\t/var/x", "rm targeting paths outside /workspace"),
    ("eval $x; socket", "Blocked eval/exec with network access"),
    ("python3 -c\n'import urllib'", "Blocked python -c with network module"),
    ("python2\n-c 'import os; os.system(1)'", "Blocked python -c with subprocess/os.exec"),
    ("base64\n-d f | bash", "Blocked base64 decode piped to shell"),
    ("xxd\n-r -p f |\nsh", "Blocked xxd reverse piped to shell"),
)
_PASSED = ("rm -rf /workspace/x", "eval x\nsocket", "python -c x\nurllib", "base64 -d f\n| bash", "xxd -r f\n|sh")


def test_eq20_the_sandbox_refuses_what_it_refused_a_head_across_lines_included_and_a_long_line_at_once(tmp_path):
    sm, restore = _sandbox(tmp_path)
    try:
        validator = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=[]))
        for command, reason in _REFUSED:
            assert validator.validate(command) == (False, reason), command
        for command in _PASSED:
            assert validator.validate(command) == (True, ""), command
        last = validator._PYTHON_C_SUBPROCESS
        assert last.search("python -c 'subprocess.Popen(1)'") and not last.search("python -c 'subprocess'")
        process, piped = [p for p in validator._DANGEROUS_FILE_CONTENT if isinstance(p, sm.JoinedPattern)]
        assert [bool(process.search(t)) for t in ("subprocess\nrun", "subprocess.run(x)")] == [False, True]
        assert [bool(piped.search(t)) for t in ("base64 x\n| sh", "base64 x | sh")] == [False, True]
        long = "python -c x " * 20000 + "\nurllib"
        start = time.perf_counter()
        assert validator.validate(long) == (True, "")
        assert time.perf_counter() - start < 2.0, "a long line repeating a head is read at once"
    finally:
        restore()


def test_eq21_the_analyzer_scores_with_each_pair_as_it_scored_with_the_pair_as_one_pattern():
    analyzer, _joined_module, restore = _analyzer()
    try:
        task = analyzer.TaskAnalyzer()
        letters = r"[a-zA-Z" + _E + _G + "]+"
        pairs = []
        for pattern in sorted(_OLD_JOINED):
            if pattern.startswith("\\b"):
                head, tail = (part.replace("\\b", " ").replace("\\s+", " ") for part in pattern.split(".*"))
                pairs.append((" ".join(re.findall(letters, head)), re.findall(letters, tail)))
        rng = random.Random(21)
        texts = []
        for _ in range(3000):
            parts = []
            for _ in range(rng.randint(1, 4)):
                head, tails = rng.choice(pairs)
                tail = rng.choice(tails) if rng.random() < 0.7 else rng.choice(rng.choice(pairs)[1])
                filler = " ".join(rng.choice(("x", "the", "plan", "R", head, tail)) for _ in range(rng.randint(0, 3)))
                parts.append(rng.choice((head, head.upper())) + " " + filler + rng.choice((" ", "\n", " ")) + tail)
            texts.append(rng.choice((" ", "\n", ". ")).join(parts))
        joined = 0
        for name, levels in task._patterns.items():
            plain = {level: [re.compile(p.pattern, p.flags) if isinstance(p, analyzer.JoinedPattern) else p
                             for p in patterns] for level, patterns in levels.items()}
            joined += sum(isinstance(p, analyzer.JoinedPattern) for patterns in levels.values() for p in patterns)
            for text in texts:
                assert task._score_patterns(text, levels) == task._score_patterns(text, plain), (name, text)
        assert joined == 17, joined
        pairs = [p for levels in task._patterns.values() for patterns in levels.values() for p in patterns
                 if isinstance(p, analyzer.JoinedPattern)]
        reached = sum(any(p.findall(text) for p in pairs) for text in texts)
        assert 1000 < reached < 2900, ("witness: the corpus holds pairs and texts without", reached)
    finally:
        restore()


_UNJOINABLE = (("a", "b|c", 0), ("a|b", "c", 0), ("(x)a", r"(a)\1", 0), ("(x)a", "(a)?(?(1)b|c)", 0),
               ("(?i)y", "(?i)x", re.IGNORECASE), ("a", r"\\|c", 0), ("a", "b", re.VERBOSE), ("a", "(?#()b|c", 0),
               ("(?x: a # (\n )|b", "c", 0), ("a", "(?x:b#(\n)|c", 0), ("(x)a", r"(a)\1x", 0), ("a", "[x]b|c", 0),
               ("a", "[]]|c", 0), ("a", "[^]]|c", 0), ("a", "(b)c|d", 0), ("a", "(?#x)|c", 0))
_JOINABLE = (("a", r"\b(b|c)\b", 0), ("a", "[|]x", 0), ("a", r"\|x", 0), ("(?i)ab", "x", re.IGNORECASE),
             ("a", r"(b)\1", 0), ("a", "[]|]x", 0), ("a", "[^]|]x", 0), ("a", r"[\]|]x", 0), ("a", "(?i:b|c)", 0),
             (r"a\s*", "b", 0), ("a", r"\s*b", 0), ("(a)", "(b|c)", 0), ("(a)x?", "(b)", 0), ("a", "(?#|)b", 0),
             (r"(a)\1", "b", 0), ("(a)?(?(1)b|c)", "x", 0))


def test_eq22_a_head_or_a_tail_the_join_would_read_otherwise_is_refused_and_one_it_reads_as_written_is_taken():
    _analyzer_module, joined, restore = _analyzer()
    try:
        for head, tail, flags in _UNJOINABLE:
            try:
                joined.JoinedPattern(re.compile(head, flags), re.compile(tail, flags))
            except ValueError:
                continue
            raise AssertionError(("taken", head, tail))
        texts = ["".join(p) for count in range(6) for p in itertools.product(("a", "b", "c", "x", "|", "]", "\n", "A"),
                                                                              repeat=count)]
        for head, tail, flags in _JOINABLE:
            reader = joined.JoinedPattern(re.compile(head, flags), re.compile(tail, flags))
            whole, hits = re.compile(reader.pattern, flags), 0
            for text in texts:
                expected = _runs(whole, text)
                hits += bool(expected)
                assert _runs(reader, text) == expected, (head, tail, text)
                assert reader.findall(text) == whole.findall(text), (head, tail, text)
            assert hits > 100, ("witness: the texts reach the pair", head, tail, hits)
    finally:
        restore()


def test_eq23_the_analyzer_s_score_and_its_keyword_set_do_not_depend_on_the_order_of_a_level_s_patterns():
    analyzer, _joined_module, restore = _analyzer()
    try:
        task = analyzer.TaskAnalyzer()
        words = sorted({w for levels in (analyzer.R_PAIRS, analyzer.DEBUG_PAIRS, analyzer.PLANNING_PAIRS)
                        for pairs in levels.values() for pair in pairs
                        for w in re.findall(r"[a-zA-Z]{2,}", " ".join(pair).replace("\\b", " "))})
        words += ["library(x)", "<-", "import os", "error", "plan", "bash", "R", "data.frame"]
        rng = random.Random(23)
        texts = [" ".join(rng.choice(words) for _ in range(rng.randint(1, 14))) for _ in range(1500)]
        reordered = compared = 0
        for levels in task._patterns.values():
            mixed = {level: rng.sample(patterns, len(patterns)) for level, patterns in levels.items()}
            reordered += any(mixed[level] != levels[level] for level in levels)
            for text in texts:
                score, keywords = task._score_patterns(text, levels)
                again, other = task._score_patterns(text, mixed)
                assert score == again, text
                if len(keywords) < 5 and len(other) < 5:  # five of a larger set: which five is the hash's
                    assert set(keywords) == set(other), text
                    compared += bool(keywords)
        assert reordered >= 5, ("witness: the levels were reordered", reordered)
        assert compared > 1000, ("witness: keyword sets compared", compared)
        assert sum(task._score_patterns(text, task._patterns["debug"])[0] > 0 for text in texts) > 300
    finally:
        restore()
