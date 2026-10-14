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
  * EQ24 -- the section, sheet, repository and tag readers read as before
    (it gives way for EQ5): a sheet marker or a tag left open takes its run
    at once, the sheets are split at the real markers only, and the tags are
    read once per place by the Svelte checker's walker.
  * EQ25 -- the claim reader reads as before (it gives way for EQ6): a run of
    words with no verb in it opens no claim and is taken at once; the claims
    the fact checker extracts are the ones it extracted; the rm check answers
    as before.
  * EQ26 -- the XML tool calls of a reply are read as before; an opening left
    open, without its ">" or its closing tag, takes the rest at once.
  * EQ27 -- a plan is read from a planner's reply as the three patterns read
    it, by find and rfind.
  * EQ28 -- the think blocks of a reply are taken out as before; an opening
    left open is kept, as it was.
  * EQ29 -- the Svelte scan takes out comments and script and style bodies as
    before, names compared as a back-reference compares them (dotted and
    dotless i and long s read otherwise by a literal).
  * EQ30 -- the Svelte scan reads each place once: counted calls, not timings.
  * EQ31 -- the fenced blocks of skills, teacher drafts, beautified text and
    corrections are read as before.
  * EQ32 -- the page strippers take out what they took out; hidden CSS is
    matched where it was.
  * EQ33 -- the hidden-CSS reader reads each closing parenthesis once:
    counted calls, not timings.
  * EQ34 -- the images, links, addresses and table rows of a line read as
    before.
  * EQ35 -- an image or a link shares its closing bracket and its target's
    end with every opening before them: counted calls, not timings.
  * EQ36 -- the line and marker readers give their callers the same groups,
    spans or answer.
  * EQ37 -- the chunkers find the functions and classes they found, a class
    without bases inside the bases of one left open included.
  * EQ38 -- a suggested rule finds what it found; the system-tag rule also
    reads an opening left without its end.
  * EQ39 -- a statement of the drift ledger is read into the parts the
    statement pattern gave it, and where.
  * EQ40 -- a statement reads what follows each place once: counted calls,
    not timings.
  * EQ41 -- a python -c subprocess call is refused with the reason its own
    check gives, read before the broader one that answered first.
  * EQ42 -- every reader that takes a run at once reads each unit of its head
    and a character in linear time: four times as many units in well under
    sixteen times the time, the units the cost guard does not pump included.
  * EQ43 -- the section, sheet, repository and tag readers read as before
    (gives way for EQ24): the GitHub connector's repository references are
    read one run of name characters at a time, as its old pattern found them,
    in linear time.
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


# ---------------------------------------------------------------------------
# EQ24 to EQ38 -- the patterns the cost guard owed, read in linear time
# ---------------------------------------------------------------------------
# Each reader is held to the pattern it replaced, written here, as its caller reads it. Its code is run from the
# source by name, never by importing its module. A reader that takes at once a run where every start fails alike
# marks its real matches with an empty group ``real``; its caller keeps those. The caller's old reading is had by
# giving it the old pattern followed by an empty ``real`` group, which every match sets.
def _with_real(old, flags):
    """The old pattern, every match of which sets ``real``: a caller that keeps real matches reads it as before."""
    return re.compile("(?:" + old + ")(?P<real>)", flags)


def _defs(module, *names, **given):
    """A namespace holding the top-level definitions ``names`` of ``module``, run from its source."""
    import __future__

    import dataclasses
    tree = ast.parse((REPO / "opti_oignon" / module).read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        named = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            named = [node.name]
        elif isinstance(node, ast.Assign):
            named = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            named = [node.target.id]
        if set(named) & set(names):
            body.append(node)
    space = {"re": re, "json": __import__("json"), "dataclass": dataclasses.dataclass, "field": dataclasses.field,
             "__name__": __name__, **given}
    code = compile(ast.Module(body=body, type_ignores=[]), module, "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, space)  # noqa: S102
    missing = [name for name in names if name not in space]
    assert not missing, (module, missing)
    return space


def _method(module, name, owner=None, **given):
    """Method ``name`` of class ``owner`` of ``module`` (the one class that holds it when None), run from its source
    as a plain function."""
    import __future__
    tree = ast.parse((REPO / "opti_oignon" / module).read_text(encoding="utf-8"))
    found = [item for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and owner in (None, node.name)
             for item in node.body
             if isinstance(item, ast.FunctionDef) and item.name == name]
    assert len(found) == 1, (module, name, len(found))
    found[0].decorator_list = []
    space = {"re": re, "json": __import__("json"), **given}
    code = compile(ast.Module(body=found, type_ignores=[]), module, "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, space)  # noqa: S102
    return space[name]


def _statement(module, function, marker):
    """The one assignment of ``function`` in ``module`` that holds ``marker``, as a function of its ``text``."""
    tree = ast.parse((REPO / "opti_oignon" / module).read_text(encoding="utf-8"))
    found = [s for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == function
             for s in node.body if isinstance(s, ast.Assign) and marker in ast.unparse(s)]
    assert len(found) == 1, (module, function, marker, len(found))
    code = compile(ast.Module(body=found, type_ignores=[]), module, "exec")

    def run(text):
        space = {"re": re, "text": text}
        exec(code, space)  # noqa: S102
        return space["text"]
    return run


def _class_patterns(module, cls):
    """The compiled patterns that class ``cls`` of ``module`` holds as attributes, by name, run from its source."""
    tree = ast.parse((REPO / "opti_oignon" / module).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    space = {"re": re}
    for item in node.body:
        if isinstance(item, ast.Assign) and isinstance(item.value, ast.Call) and "re.compile" in ast.unparse(item.value):
            exec(compile(ast.Module(body=[item], type_ignores=[]), module, "exec"), space)  # noqa: S102
    return {name: value for name, value in space.items() if isinstance(value, re.Pattern)}


def _drawn(pieces, seed, cases=_CASES):
    """Texts of up to fourteen of ``pieces``, glued."""
    rng = random.Random(seed)
    return ["".join(rng.choice(pieces) for _ in range(rng.randint(0, 14))) for _ in range(cases)]


def _both(old, flags, pieces, seed=0):
    """The corpus of an old pattern (texts built from its tree, then mutated), and texts drawn from ``pieces``."""
    return list(_corpus(old, flags, seed=seed)) + _drawn(pieces, seed + 1)


def _agree(texts, old, new, least=_CASES // 40, hit=bool):
    """``new`` reads every text as ``old`` does; ``old`` finds something (``hit``) in many texts, and nothing in many."""
    found = 0
    for text in texts:
        expected = old(text)
        found += bool(hit(expected))
        assert new(text) == expected, (text, expected)
    assert least < found < len(texts) - least, ("witness: the corpus reaches both sides", found, len(texts))


def _real(compiled, text):
    """The real matches of a reader that takes a run at once, as (span, the groups before ``real``)."""
    before = compiled.groupindex["real"] - 1
    return [(m.span(), m.groups()[:before]) for m in compiled.finditer(text) if m.group("real") is not None]


def _all(compiled, text):
    return [(m.span(), m.groups()) for m in compiled.finditer(text)]


class _Counted:
    """A compiled pattern whose ``search`` and ``match`` calls are counted, with the characters each search passed
    before what it found (or to the end)."""

    def __init__(self, compiled):
        self.compiled, self.calls, self.passed = compiled, 0, 0

    def search(self, *args):
        self.calls += 1
        found = self.compiled.search(*args)
        self.passed += (len(args[0]) if found is None else found.start()) - (args[1] if len(args) > 1 else 0)
        return found

    def match(self, *args):
        self.calls += 1
        return self.compiled.match(*args)

    def fullmatch(self, *args):
        self.calls += 1
        return self.compiled.fullmatch(*args)

    def __getattr__(self, name):
        return getattr(self.compiled, name)


class _Text(str):
    """A text whose ``find`` and ``rfind`` calls are counted, with the characters they passed."""

    def find(self, *args):
        found = str.find(self, *args)
        self.calls = getattr(self, "calls", 0) + 1
        start = args[1] if len(args) > 1 else 0
        self.passed = getattr(self, "passed", 0) + ((len(self) if found == -1 else found) - start)
        return found


def _every(tokens, most):
    """Every text of at most ``most`` of ``tokens``: a bounded proof the random corpus cannot give."""
    for count in range(most + 1):
        for parts in itertools.product(tokens, repeat=count):
            yield "".join(parts)


def _plugin(name):
    """The entry point of plugin ``name``, loaded alone from its source."""
    dotted = "opti_oignon.plugins." + name.replace("-", "_")
    loaded, restore = isolate(targets={dotted + ".entry_point": source("plugins", name, "entry_point.py")},
                              packages=("opti_oignon.plugins", dotted))
    return loaded[dotted + ".entry_point"], restore


# ---------------------------------------------------------------------------
# EQ24 -- the section, sheet, repository and tag readers (gives way for EQ5)
# ---------------------------------------------------------------------------
_TAG = r"<\s*(/?)([A-Za-z][A-Za-z0-9:._-]*)((?:\"[^\"]*\"|'[^']*'|[^>\"'])*?)(/?)\s*>"
_SHEET = r"\[Sheet:\s*([^\]]+)\]\n"


def _sheet_split():
    """The statements of ``_chunk_by_sheets`` from its pattern to its list of sheets, as a function of the content."""
    tree = ast.parse((REPO / "opti_oignon" / "rag" / "chunkers.py").read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_chunk_by_sheets")
    first = next(i for i, s in enumerate(method.body) if "re.compile" in ast.unparse(s))
    last = next(i for i, s in enumerate(method.body) if "sheets.append" in ast.unparse(s))
    split = ast.parse("def split(content):\n    return sheets").body[0]
    split.body = method.body[first:last + 1] + split.body
    space = {"re": re}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[split], type_ignores=[])), "chunkers", "exec"), space)  # noqa: S102
    return space["split"]


def test_eq24_the_section_sheet_repository_and_tag_readers_read_as_before_with_runs_left_open_taken_at_once():
    _hold([("plugins/github-connector/entry_point.py", "/[a-zA-Z0-9_.-]+)#",
            r"(?<!\w)([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)#(\d+)(?!\w)", "exact")])
    tree = ast.parse((REPO / "opti_oignon" / "rag" / "chunkers.py").read_text(encoding="utf-8"))
    sections = [n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
                and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)
                and n.args[0].value.startswith("^# ")]
    assert len(sections) == 2 and sections[0] == sections[1], sections
    assert not _differences(_SECTION, sections[0], M, "exact")
    split, old = _sheet_split(), re.compile(_SHEET)
    _agree(_both(_SHEET, 0, ["[Sheet:", " ", "a", "]", "\n", "]\n", "x", "\t", "[Sheet: a]\n"]),
           old.split, split, hit=lambda sheets: len(sheets) > 1)
    assert split("[Sheet: a" * 300 + "]\n") == old.split("[Sheet: a" * 300 + "]\n")
    tags = _defs("sandbox_tools.py", "_TAG_HEAD_RE", "_TAG_END_RE", "_tags")["_tags"]
    _agree(_both(_TAG, 0, ["<", "</", "< ", "a", "b1", "/", ">", " ", "\n", '"', "'", "x=", "<a", "/>", "\t"]),
           re.compile(_TAG).findall, tags)
    _agree(list(_every(["<", "/", "a", " ", ">", '"', "'"], 6)), re.compile(_TAG).findall, tags)


# ---------------------------------------------------------------------------
# EQ25 -- the claim reader and the rm check (gives way for EQ6)
# ---------------------------------------------------------------------------
def test_eq25_the_claim_reader_reads_as_before_a_run_of_words_with_no_verb_taken_at_once():
    claim, flags = _joined("plugins/fact-checker/entry_point.py", "founded|invented")
    assert flags == I and claim != _CLAIM
    texts = _both(_CLAIM, I, ["Ada", " ", "Lovelace", "founded", "is the capital of", "x", ".", ",", "\n", "aa", "1"])
    _agree(texts, lambda t: _all(re.compile(_CLAIM, I), t), lambda t: _real(re.compile(claim, I), t))
    fact, restore = _plugin("fact-checker")
    try:
        def read(text):
            return [(c.text, c.start, c.end) for c in
                    fact.extract_claims(text, aggressiveness="moderate", max_claims=100_000)]
        new = fact._ENTITY_CLAIM_PATTERN
        fact._ENTITY_CLAIM_PATTERN = _with_real(_CLAIM, I)
        expected = [read(text) for text in texts[::5]]
        fact._ENTITY_CLAIM_PATTERN = new
        assert [read(text) for text in texts[::5]] == expected
        assert sum(map(bool, expected)) > 100, "witness: claims are read"
    finally:
        restore()
    _hold([("sandbox_manager.py", r"(?!workspace)", r"\brm\b.*\s+/(?!workspace)", "boolean")])


# ---------------------------------------------------------------------------
# EQ26 -- the tool calls of a reply
# ---------------------------------------------------------------------------
_INVOKE = r"<invoke\b([^>]*)>(.*?)</invoke>"
_PARAM = r"<param(?:eter)?\b([^>]*)>(.*?)</param(?:eter)?>"


def test_eq26_the_xml_tool_calls_of_a_reply_are_read_as_before_an_opening_left_open_taken_at_once():
    loaded, restore = isolate(targets={"opti_oignon.agent.tool_parsing": source("agent", "tool_parsing.py")},
                              packages=("opti_oignon.agent",))
    try:
        parsing = loaded["opti_oignon.agent.tool_parsing"]
        pieces = ["<invoke", "<INVOKE", "<invokex", " name='a'", ' name="b"', ">", "</invoke>", "<param",
                  "<parameter", " name='p'", "</param>", "</parameter>", "\n", " ", "x", "1", "<", "/",
                  "<invoke name='a'>", "<parameter name='p'>v</parameter>", "</invoke>"]
        texts = _both(_INVOKE, S | I, pieces) + _drawn(pieces, 5)
        _agree(texts, lambda t: _all(re.compile(_INVOKE, S | I), t), lambda t: _real(parsing._XML_INVOKE_RE, t))
        _agree(texts, lambda t: _all(re.compile(_PARAM, S | I), t), lambda t: _real(parsing._XML_PARAM_RE, t))

        def read(text):
            return [(call.name, call.arguments, call.raw) for call in parsing.parse_xml_blocks(text)]
        new = parsing._XML_INVOKE_RE, parsing._XML_PARAM_RE
        parsing._XML_INVOKE_RE, parsing._XML_PARAM_RE = _with_real(_INVOKE, S | I), _with_real(_PARAM, S | I)
        expected = [read(text) for text in texts]
        parsing._XML_INVOKE_RE, parsing._XML_PARAM_RE = new
        assert [read(text) for text in texts] == expected
        assert sum(map(bool, expected)) > 200, "witness: calls are read"
        assert sum(1 for calls in expected for _name, arguments, _raw in calls if arguments) > 50, (
            "witness: calls with arguments are read")
        assert read("<invoke name='a'><parameter name='p'>1</parameter></invoke>") == [
            ("a", {"p": 1}, "<invoke name='a'><parameter name='p'>1</parameter></invoke>")]
        assert read("<invoke name='a'><parameter name='p'>" * 3000) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# EQ27 -- the plan of a planner's reply
# ---------------------------------------------------------------------------
def test_eq27_a_plan_is_read_from_a_reply_as_the_three_patterns_read_it():
    import json

    class Quiet:
        @staticmethod
        def warning(*_args):
            return None

    parse = _method("agents/dynamic_pipeline.py", "_parse_plan_json", logger=Quiet)

    def old(raw):
        if not raw:
            return None
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass
        for pattern in (r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```", r"\{[\s\S]*\}"):
            match = re.search(pattern, raw)
            if match:
                try:
                    return json.loads((match.group(1) if "```" in pattern else match.group(0)).strip())
                except json.JSONDecodeError:
                    continue
        start, end = raw.find("{"), raw.rfind("}")
        try:
            if start != -1 and end > start:
                return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
        return None

    pieces = ["```json", "```", "{", "}", '{"a": 1}', "[1]", " ", "\n", "x", "1", '"', ":", ",", "\t"]
    _agree(_drawn(pieces, 0) + _drawn(pieces, 1), old, lambda raw: parse(None, raw))
    assert parse(None, "```json" * 2000) is None and parse(None, "{" * 4000 + "x") is None


# ---------------------------------------------------------------------------
# EQ28 -- the think blocks of a reply
# ---------------------------------------------------------------------------
def test_eq28_the_think_blocks_of_a_reply_are_taken_out_as_before_an_opening_left_open_kept():
    pieces = ["<think>", "</think>", "<THINK>", "</Think>", "<think", "think>", "\n", " ", "x", "<", ">"]
    texts = _both(r"<think>.*?</think>", S | I, pieces)
    blocks = re.compile(r"<think>.*?</think>", S)
    for module in ("context_summary.py", "memory/legacy.py"):
        clean = _method(module, "_clean_think_tags")

        def old(text):
            return blocks.sub("", text).replace("<think>", "").replace("</think>", "").strip()

        _agree(texts, lambda text: (old(text), blocks.search(text) is not None),
               lambda text, clean=clean: (clean(None, text), blocks.search(text) is not None), hit=lambda e: e[1])
        assert clean(None, "<think>" * 3 + "a</think>b") == "b" and clean(None, "x<think>" * 4) == "x" * 4
    folded = re.compile(r"<think>.*?</think>", S | I)
    for module in ("memory/curation.py", "memory/extraction.py"):
        space = _defs(module, "_THINK_RE", "_drop_closed")
        _agree(texts, lambda text: (folded.sub("", text), folded.search(text) is not None),
               lambda text, s=space: (s["_THINK_RE"].sub(s["_drop_closed"], text), folded.search(text) is not None),
               hit=lambda e: e[1])
        assert space["_THINK_RE"].sub(space["_drop_closed"], "<THINK>a" * 3 + "</think>b") == "b"


# ---------------------------------------------------------------------------
# EQ29, EQ30 -- the Svelte checker's scan
# ---------------------------------------------------------------------------
_SCRIPT_STYLE = r"(<(script|style)\b[^>]*>).*?(</\2\s*>)"
_DOTLESS, _DOTTED, _LONG_S = chr(0x131), chr(0x130), chr(0x17F)


def _scan():
    return _defs("sandbox_tools.py", "_TAG_HEAD_RE", "_TAG_END_RE", "_SCRIPT_STYLE_RE", "_SCRIPT_STYLE_ENDS",
                 "_SAME_NAME_RE", "_closing_tag", "_blank_script_style", "_tags")


def test_eq29_the_svelte_scan_takes_out_comments_and_script_bodies_as_before_names_compared_as_a_back_reference():
    blank = _scan()["_blank_script_style"]
    old = re.compile(_SCRIPT_STYLE, I | S)
    pieces = ["<script", "<style", "<SCRIPT", "<scr" + _DOTTED + "pt", "<" + _LONG_S + "cript", ">", "</script>",
              "</style >", "</SCRIPT>", "</" + _LONG_S + "cript>", "</scr" + _DOTLESS + "pt>", " a", "\n", "x", "<"]
    texts = _both(_SCRIPT_STYLE, I | S, pieces)
    _agree(texts, lambda t: (old.sub(lambda m: m.group(1) + m.group(3), t), old.search(t) is not None),
           lambda t: (blank(t), old.search(t) is not None), hit=lambda e: e[1])
    assert blank("<" + _LONG_S + "cript>a</script>") == "<" + _LONG_S + "cript>a</script>", "a literal would close"
    assert blank("<style>a<script>b</script>c</style>") == "<style></style>"
    _agree(list(_every(["<script", "<style", ">", "</script>", "</style>", "a", "<" + _LONG_S + "cript",
                        "</" + _LONG_S + "cript>"], 5)),
           lambda t: (old.sub(lambda m: m.group(1) + m.group(3), t), old.search(t) is not None),
           lambda t: (blank(t), old.search(t) is not None), hit=lambda e: e[1])
    comments = _statement("sandbox_tools.py", "_strip_for_tag_scan", "<!--")
    wanted = re.compile(r"<!--.*?-->", S)
    _agree(_both(r"<!--.*?-->", S, ["<!--", "-->", "<!-->", "--", ">", "x", "\n", "<", "!"]),
           lambda t: (wanted.sub("", t), wanted.search(t) is not None),
           lambda t: (comments(t), wanted.search(t) is not None), hit=lambda e: e[1])
    assert comments("<!--" * 5 + "a-->b<!--c") == "b<!--c"


def test_eq30_the_svelte_scan_reads_each_place_once_counted_calls_not_timings():
    for shape, last in (("<a ", ""), ("< /aa", ""), ("<a x='", ""), ('<a "<a ', ""), ("<b\n", ""), ("<script", ""),
                        ("<style x", ""), ("<script>", ""), ("</script>a", ""), ("<script ", ">"), ("<a ", ">")):
        for size in (400, 1600):
            space = _scan()
            end, head = _Counted(space["_TAG_END_RE"]), _Counted(space["_TAG_HEAD_RE"])
            ends = {name: _Counted(pattern) for name, pattern in space["_SCRIPT_STYLE_ENDS"].items()}
            space.update(_TAG_END_RE=end, _TAG_HEAD_RE=head, _SCRIPT_STYLE_ENDS=ends)
            text = _Text(shape * size + last)
            space["_tags"](text)
            assert 0 < end.calls <= len(text) + 1, ("a place walked twice, or none counted", shape, end.calls)
            assert head.calls <= text.count("<") + 1, (shape, head.calls)
            assert getattr(text, "passed", 0) <= len(text), ("a quoted run read twice", shape)
            assert head.passed <= len(text), ("a head searched twice", shape)
            text = _Text(shape * size + last)
            space["_blank_script_style"](text)
            assert sum(e.calls for e in ends.values()) <= 4, ("a closing tag searched again", shape)
            opened = "<script" in shape and ">" in shape + last
            assert not opened or sum(e.calls for e in ends.values()) > 0, ("no closing tag counted", shape)
            assert sum(e.passed for e in ends.values()) <= 2 * len(text), ("a closing tag searched again", shape)
            assert getattr(text, "passed", 0) <= 2 * len(text), ("a > searched again", shape)


# ---------------------------------------------------------------------------
# EQ31 -- fenced blocks
# ---------------------------------------------------------------------------
_FENCED = r"```[^\n]*\n(.*?)```"


def test_eq31_the_fenced_blocks_of_skills_drafts_beautified_text_and_corrections_read_as_before():
    pieces = ["```", "```skill", "```SKILL", "```skills", "```py", "\n", " ", "x", "`", "``", "{", "}", "json"]
    texts = _both(_FENCED, S, pieces)
    space = _defs("agent/skills.py", "SECTION_VERIFICATION", "_extract_section", "_FENCE_RE", "_verification_commands")
    old = re.compile(_FENCED, S)

    def commands_old(body):
        section = space["_extract_section"](body, space["SECTION_VERIFICATION"])
        return [m.group(1).strip() for m in old.finditer(section) if m.group(1).strip()] if section else []

    _agree(["## Verification\n" + t for t in texts], commands_old, space["_verification_commands"])
    teacher = _defs("agent/teacher.py", "_SKILL_FENCE_RE", "_FIELD_RE_TMPL", "_SLUG_RE", "_slug", "_field",
                    "TeacherSkillDraft", "SOURCE_TEACHER", "extract_skill_draft")
    skill = re.compile(r"```skill\b[^\n]*\n(.*?)```", S | I)

    def draft_old(text):
        found = skill.search(text) if text else None
        return found.group(1).strip() or None if found else None

    _agree(_both(skill.pattern, S | I, pieces), draft_old,
           lambda t: (lambda d: None if d is None else d.content)(teacher["extract_skill_draft"](t)))
    beautifier, restore = _plugin("markdown-beautifier")
    try:
        new = beautifier._FENCED_BLOCK_RE
        beautifier._FENCED_BLOCK_RE = _with_real(r"(```[^\n]*\n[\s\S]*?```)", S)
        expected = [beautifier._protect_fenced_blocks(text) for text in texts]
        beautifier._FENCED_BLOCK_RE = new
        assert [beautifier._protect_fenced_blocks(text) for text in texts] == expected
        assert sum(1 for _protected, kept in expected if kept) > 200, "witness: blocks are protected"
        assert beautifier._protect_fenced_blocks("```\n" + "x" * 50) == ("```\n" + "x" * 50, [])
    finally:
        restore()
    corrected = _method("self_correction.py", "_parse_json_response")
    fence = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', S)

    def json_old(text):
        import json
        if not text:
            return None
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        for match, group in ((fence.search(text), 1),
                             (re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL), 0)):
            if match:
                try:
                    return json.loads(match.group(group))
                except json.JSONDecodeError:
                    pass
        return None

    jpieces = ["```", "```json", "{", "}", '{"a": 1}', '"b"', ":", " ", "\n", "x", ","]
    _agree(_both(fence.pattern, S, jpieces), json_old, corrected)
    assert corrected("```json {" + "x" * 50) is None


# ---------------------------------------------------------------------------
# EQ32, EQ33 -- pages and documents
# ---------------------------------------------------------------------------
_HIDDEN = (r"(?i)(display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|opacity\s*:\s*0|"
           r"position\s*:\s*absolute\s*;\s*left\s*:\s*-\d+|color\s*:\s*(?:white|#fff(?:fff)?|rgba\([^)]*,\s*0\s*\)))")
_HIDDEN_PIECES = ["color", "COLOR", ":", " ", "rgba(", "RGBA(", ",", "0", "00", ")", "1", "white", "#fff", "#ffffff",
                  "display:none", "opacity:0", "position: absolute; left: -9", "font-size:0", "\n", "x", "(", "\t"]


def _hidden():
    return _defs("rag_sanitizer.py", "_HIDDEN_CSS_FIXED", "_HIDDEN_RGBA_HEAD", "_zero_alpha_comma", "_HiddenCss",
                 "_HIDDEN_CSS")


def test_eq32_the_page_strippers_take_out_what_they_took_out_hidden_css_matched_where_it_was():
    tags = _statement("rag_store.py", "_extract_html_text", "<[^>]+>")
    _agree(_both(r"<[^>]+>", 0, ["<", ">", "<>", "a", " ", "\n", "<a>", "<<", ">>"]),
           lambda t: (re.sub(r"<[^>]+>", " ", t), re.search(r"<[^>]+>", t) is not None),
           lambda t: (tags(t), re.search(r"<[^>]+>", t) is not None), hit=lambda e: e[1])
    css, old = _hidden()["_HIDDEN_CSS"], re.compile(_HIDDEN)
    texts = _both(_HIDDEN, 0, _HIDDEN_PIECES)
    _agree(texts, lambda t: [m.span() for m in old.finditer(t)], lambda t: list(css.spans(t)))
    _agree(texts, lambda t: (old.sub("[R]", t), old.search(t) is not None),
           lambda t: (css.sub("[R]", t), old.search(t) is not None), hit=lambda e: e[1])
    _agree(list(_every(["color:rgba(", "color:", ",", "0", " ", ")", "1", "display:none"], 5)),
           lambda t: [m.span() for m in old.finditer(t)], lambda t: list(css.spans(t)))


def test_eq33_the_hidden_css_reader_reads_each_closing_once_counted_calls_not_timings():
    for shape, last in (("color: rgba(", ")"), ("color:rgba(1,", ")"), ("color: rgba(x) ", ")"), ("display: ", ")"),
                        ("color:rgba(1, 0", ")"), ("color:rgba(0, 0) ", "display:none"),
                        ("display:none ", "color:rgba(1, 0)"), ("color:rgba(0, 0) ", ""), ("display:none ", "")):
        for size in (400, 1600):
            space = _hidden()
            head, fixed = _Counted(space["_HIDDEN_RGBA_HEAD"]), _Counted(space["_HIDDEN_CSS_FIXED"])
            commas, comma = [], space["_zero_alpha_comma"]
            space.update(_HIDDEN_RGBA_HEAD=head, _HIDDEN_CSS_FIXED=fixed,
                         _zero_alpha_comma=lambda text, close, comma=comma: commas.append(close) or comma(text, close))
            text = _Text(shape * size + last)
            list(space["_HiddenCss"]().spans(text))
            assert 0 < head.calls <= size + 2 and 0 < fixed.calls <= size + 2, (shape, head.calls, fixed.calls)
            assert head.passed <= 2 * len(text) and fixed.passed <= 2 * len(text), (shape, head.passed, fixed.passed)
            assert len(commas) == len(set(commas)), ("a closing read twice", shape)
            assert getattr(text, "passed", 0) <= 2 * len(text), ("a ) searched again", shape)


# ---------------------------------------------------------------------------
# EQ34, EQ35 -- the inline readers of the fact checker
# ---------------------------------------------------------------------------
_IMAGE = r"!\[[^\]]*\]\([^)\s]*(?:\s+\"[^\"]*\")?\)"
_LINK = r"(?<!!)\[([^\]]+)\]\([^)\s]*(?:\s+\"[^\"]*\")?\)"


def _markup():
    return _defs("factcheck/markup.py", "_IMAGE_HEAD", "_LINK_HEAD", "_TARGET_STOP", "_TARGET_TAIL", "_targets")


_OLD_TARGET_LOOPS = """
for match in re.finditer(_OLD_IMAGE, subject):
    for position in range(match.start(), match.end()):
        deleted[position] = True
    images.append((cstart + match.start(), cstart + match.end()))
for match in re.finditer(_OLD_LINK, subject):
    if any(deleted[match.start():match.end()]):
        continue
    deleted[match.start()] = True
    for position in range(match.end(1), match.end()):
        deleted[position] = True
"""


def _inlines():
    """The line reader of the fact checker, and the same with its image and link loops as they were written."""
    names = ("_TICKS", "_IMAGE_HEAD", "_LINK_HEAD", "_TARGET_STOP", "_TARGET_TAIL", "_AUTOLINK", "_INLINE_HTML",
             "_STRIKE", "_EMPHASIS", "_code_spans", "_targets", "_inline")
    new = _defs("factcheck/markup.py", *names)["_inline"]
    tree = ast.parse((REPO / "opti_oignon" / "factcheck" / "markup.py").read_text(encoding="utf-8"))
    inline = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_inline")
    loops = [i for i, s in enumerate(inline.body) if isinstance(s, ast.For) and "_targets" in ast.unparse(s.iter)]
    assert len(loops) == 2 and loops[1] == loops[0] + 1, loops
    inline.body[loops[0]:loops[1] + 1] = ast.parse(_OLD_TARGET_LOOPS).body
    space = _defs("factcheck/markup.py", *names[:-1], _OLD_IMAGE=_IMAGE, _OLD_LINK=_LINK)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[inline], type_ignores=[])), "markup", "exec"), space)  # noqa: S102
    return new, space["_inline"]


def test_eq34_the_images_links_addresses_and_table_rows_of_a_line_read_as_before():
    _hold([("factcheck/markup.py", r"[^\s<>@.]*",
            r"<((?:https?|ftp)://[^\s<>]+|[^\s<>@]+@[^\s<>@]+\.[^\s<>]+)>", "exact")])
    delim, _flags = _site("factcheck/markup.py", r"(?:\|[ \t]*)?$")
    old_delim = r"^ {0,3}\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$"
    _agree(_both(old_delim, 0, [" ", "|", "-", ":", "\t", "--", "x", "\n", "| --- |", "|-|"]),
           lambda t: (lambda m: m and m.span())(re.match(old_delim, t)),
           lambda t: (lambda m: m and m.span())(re.match(delim, t)))
    space = _markup()
    pieces = ["![", "[", "]", "(", ")", "](", " ", "\t", '"', "a", "!", "\n", "x y", ' "t"']
    _agree(_both(_IMAGE, 0, pieces), lambda t: [(m.start(), m.end()) for m in re.finditer(_IMAGE, t)],
           lambda t: [(start, end) for start, _close, end in space["_targets"](t, space["_IMAGE_HEAD"], 0)])
    _agree(_both(_LINK, 0, pieces, seed=1), lambda t: [(m.start(), m.end(1), m.end()) for m in re.finditer(_LINK, t)],
           lambda t: space["_targets"](t, space["_LINK_HEAD"], 1))
    short = list(_every(["![", "[", "]", "(", "](", ")", " ", "a", '"'], 5))
    _agree(short, lambda t: [(m.start(), m.end()) for m in re.finditer(_IMAGE, t)],
           lambda t: [(start, end) for start, _close, end in space["_targets"](t, space["_IMAGE_HEAD"], 0)])
    _agree(short, lambda t: [(m.start(), m.end(1), m.end()) for m in re.finditer(_LINK, t)],
           lambda t: space["_targets"](t, space["_LINK_HEAD"], 1))
    new, old = _inlines()
    _agree(_both(_IMAGE, 0, pieces + ["<a@b.c>", "~~x~~", "`c`", "**b**"], seed=2) + short[::3],
           lambda t: old(t, 0, len(t)), lambda t: new(t, 0, len(t)), hit=lambda e: e[3] or e[4])


def test_eq35_an_image_or_a_link_shares_its_closing_bracket_and_target_end_counted_calls_not_timings():
    for shape, last in (("![a](", ""), ("[a](", ""), ("![a](x ", ""), ("[a](x \"t", ""), ("![", ""), ("[a]", ""),
                        ("![a](x![a](", ""), ("[a](x)", ""), ("![", "]"), ("[a", "]"), ("![a](", ")"), ("[a](", ")")):
        for size in (400, 1600):
            space = _markup()
            stop, tail = _Counted(space["_TARGET_STOP"]), _Counted(space["_TARGET_TAIL"])
            space.update(_TARGET_STOP=stop, _TARGET_TAIL=tail)
            for head, least in ((space["_IMAGE_HEAD"], 0), (space["_LINK_HEAD"], 1)):
                text = _Text(shape * size + last)
                space["_targets"](text, head, least)
                assert getattr(text, "passed", 0) <= len(text), ("a ] searched again", shape)
            ends = shape.count(")") + sum(char.isspace() for char in shape)
            assert stop.calls <= 2 * size + 2 and stop.passed <= 2 * len(text), (shape, stop.calls, stop.passed)
            assert "(" not in shape or (stop.calls > 0 and tail.calls > 0), ("no target end counted", shape)
            assert tail.calls <= 2 * (ends * size + 1), ("what follows a target's end read again", shape, tail.calls)


# ---------------------------------------------------------------------------
# EQ36 -- line readers and marker readers
# ---------------------------------------------------------------------------
_ANSWER = (r"(?:^|\n)\s*(?:\*\*)?(?:final answer|answer|conclusion|result|therefore|thus|in summary|to summarize|"
           r"in conclusion|so,?)(?:\*\*)?[:\s]|(?:^|\n)\s*(?:\*\*)?(?:TL;?DR|TLDR)(?:\*\*)?[:\s]|(?:^|\n)\s*---+\s*\n")
_REASONING = (r"(?:^|\n)\s*(?:\*\*)?(?:reasoning|thought process|analysis|let me think|let's think|step \d|thinking)"
              r"(?:\*\*)?[:\s]")
_VERBS = ("install", "configure", "setup", "set up", "create", "add", "remove", "update", "check", "verify", "ensure",
          "run", "test", "deploy", "build", "fix", "implement", "write", "review", "backup", "migrate", "download",
          "upload")
_IMPERATIVE = r"(?:^|\n)\s*(?:" + "|".join(_VERBS) + r")\s+.+?(?:\.|$)"


def test_eq36_the_line_and_marker_readers_give_their_callers_the_same_groups_spans_or_answer():
    tree = ast.parse(source("artifacts.py").read_text(encoding="utf-8"))
    titles = [n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
              and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str) and "<title>" in n.args[0].value]
    assert len(titles) == 2 and titles[0] == titles[1], titles
    old_title = re.compile(r"<title>(.*?)</title>", I)
    _agree(_both(old_title.pattern, I, ["<title>", "</title>", "<TITLE>", "\n", " ", "x", "<", ">"]),
           lambda t: (lambda m: m and m.group(1))(old_title.search(t)),
           lambda t: (lambda m: m and m.group(1))(re.search(titles[0], t, I)))
    _hold([("artifacts.py", r"\w[^\n]*\n", r"^(import|from)\s+\w+.*\n.*(def|class)\s+\w+", "boolean")])
    speedup, flags = _site("speculative_decoding.py", "speedup")
    old_speedup = r"speculative.*speedup[:\s]+([\d.]+)\s*x"
    _agree(_both(old_speedup, I, ["speculative", "speedup", ":", " ", "2.5", "x", "\n", "y"]),
           lambda t: (lambda m: m and m.group(1))(re.search(old_speedup, t, I)),
           lambda t: (lambda m: m and m.group(1))(re.search(speedup, t, flags)))
    fingerprint = _defs("session_fingerprint.py", "_IMPORT_PATTERNS", "_BUG_PATTERNS")
    typed = next(p for p, category in fingerprint["_BUG_PATTERNS"] if category == "type")
    assert not _differences(r"type\s*error|expected.*got|incompatible type", typed, 0, "boolean")
    imports = next(p for p, lang in fingerprint["_IMPORT_PATTERNS"] if lang == "javascript" and "from" in p)
    old_import = r"^import\s+.*\s+from\s+['\"]"
    _agree([line.strip() for text in _both(old_import, 0, ["import", " ", "\t", "x", "from", "'", "\r"])
            for line in text.split("\n")],
           lambda line: bool(re.match(old_import, line)), lambda line: bool(re.match(imports, line)), least=50)
    triggers = _defs("conversation_compressor.py", "_RETRIEVAL_TRIGGER_PATTERNS")["_RETRIEVAL_TRIGGER_PATTERNS"]
    previous = next(p for p in triggers if r"previous\s+" in p)
    assert not _differences(r"\bin\s+(?:a|our|the|my|this)?\s*previous\s+(?:message|conversation|turn)\b", previous,
                            I, "boolean")
    _hold([("plugins/chain-of-thought-enforcer/entry_point.py", r"1\)|2\)",
            r"(?:first|second|third|1\)|2\)|3\)|a\)|b\)|c\)|\band\b.*\?)", "boolean")])
    cot, restore = _plugin("chain-of-thought-enforcer")
    try:
        pieces = ["\n", " ", "\t", "answer", "Answer:", "**", "final answer", "so,", "TL;DR", "---", ":", "x",
                  "reasoning:", "step 1", "Let me think "]
        texts = _both(_ANSWER, I | M, pieces) + _both(_REASONING, I | M, pieces, seed=2)
        new = cot._ANSWER_MARKER_RE, cot._REASONING_MARKER_RE
        cot._ANSWER_MARKER_RE, cot._REASONING_MARKER_RE = _with_real(_ANSWER, I | M), _with_real(_REASONING, I | M)
        expected = [cot.split_reasoning_and_answer(text) for text in texts]
        cot._ANSWER_MARKER_RE, cot._REASONING_MARKER_RE = new
        assert [cot.split_reasoning_and_answer(text) for text in texts] == expected
        assert sum(1 for reasoning, _answer in expected if reasoning) > 200, "witness: replies are split"
        replies = [text for text in texts if text.strip()]
        hooked = [cot.hook_post_inference(types.SimpleNamespace(data={"response": text}, config={})) for text in replies]
        cot._ANSWER_MARKER_RE, cot._REASONING_MARKER_RE = _with_real(_ANSWER, I | M), _with_real(_REASONING, I | M)
        assert [cot.hook_post_inference(types.SimpleNamespace(data={"response": text}, config={}))
                for text in replies] == hooked
        cot._ANSWER_MARKER_RE, cot._REASONING_MARKER_RE = new
        assert sum(map(bool, hooked)) > 100, "witness: replies are formatted"
        only_answer = types.SimpleNamespace(data={"response": "We weigh it.\nAnswer: 42"}, config={})
        assert cot.hook_post_inference(only_answer) is not None, "an answer marker alone is enough to format"
    finally:
        restore()
    tasks, restore = _plugin("task-extractor")
    try:
        texts = _both(_IMPERATIVE, I | M, ["\n", " ", "install", "run", "set up", "x", ".", "Fix ", "\t", "y z"])
        new = tasks._IMPERATIVE_RE
        tasks._IMPERATIVE_RE = _with_real(_IMPERATIVE, I | M)
        expected = [tasks.extract_tasks(text, max_tasks=10_000) for text in texts]
        tasks._IMPERATIVE_RE = new
        assert [tasks.extract_tasks(text, max_tasks=10_000) for text in texts] == expected
        assert sum(map(bool, expected)) > 200, "witness: tasks are read"
    finally:
        restore()


# ---------------------------------------------------------------------------
# EQ37 -- the chunkers' boundaries
# ---------------------------------------------------------------------------
_DEF = r"^(async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->.*?)?:"
_CLASS = r"^class\s+(\w+)\s*(?:\([^)]*\))?\s*:"
_R_FUNCTION = r"^(\w+)\s*<-\s*function\s*\([^)]*\)\s*\{"


def _r_boundaries():
    """The statements of the R chunker's chunk that gather its boundaries, as a function of (self, content)."""
    tree = ast.parse((REPO / "opti_oignon" / "rag" / "chunkers.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RChunker")
    chunk = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "chunk")
    first = next(i for i, s in enumerate(chunk.body) if ast.unparse(s) == "boundaries = []")
    last = next(i for i, s in enumerate(chunk.body) if "FUNCTION_PATTERN" in ast.unparse(s))
    gather = ast.parse("def gather(self, content):\n    return boundaries").body[0]
    gather.body = chunk.body[first:last + 1] + gather.body
    space = {"re": re}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[gather], type_ignores=[])), "chunkers", "exec"), space)  # noqa: S102
    return space["gather"]


def test_eq37_the_chunkers_find_the_functions_and_classes_they_found_a_class_without_bases_inside_one_left_open():
    patterns = _class_patterns("rag/chunkers.py", "CodeChunker")
    find = _method("rag/chunkers.py", "_find_boundaries", "CodeChunker")
    classes = _method("rag/chunkers.py", "_classes")
    holder = types.SimpleNamespace(**patterns)
    holder._classes = lambda content: classes(holder, content)

    def old(content):
        found = [(m.start(), m.group(1).strip(), "section") for m in patterns["SECTION_PATTERN"].finditer(content)]
        found += [(m.start(), m.group(1), "class") for m in re.finditer(_CLASS, content, M)]
        found += [(m.start(), m.group(2), "function") for m in re.finditer(_DEF, content, M)]
        return found

    pieces = ["def", "async", " ", "f", "(", ")", ":", "\n", "->", "x", "class", " A", "R", "# " + "=" * 10]
    _agree(_both(_DEF, M, pieces) + _both(_CLASS, M, pieces, seed=1), old, lambda content: find(holder, content))
    assert classes(holder, "class A(\nclass B:\n)") == [(9, "B")], "a class without bases inside one left open"
    assert classes(holder, "class A(\nclass B:\n):") == [(0, "A")], "a class without bases inside one with bases"
    assert classes(holder, "class A(\nclass B:\n):\nclass C:") == [(0, "A"), (21, "C")]
    _agree(list(_every(["class A", "(", ")", ":", "\n", " "], 6)),
           lambda t: [(m.start(), m.group(1)) for m in re.finditer(_CLASS, t, M)], lambda t: classes(holder, t))
    r_patterns = _class_patterns("rag/chunkers.py", "RChunker")
    r_bounds = _r_boundaries()

    def r_old(content):
        found = [(m.start(), m.group(1).strip(), "section") for m in r_patterns["SECTION_PATTERN"].finditer(content)]
        return found + [(m.start(), m.group(1), "function") for m in re.finditer(_R_FUNCTION, content, M)]

    _agree(_both(_R_FUNCTION, M, ["f", " ", "<-", "function", "(", ")", "{", "\n", "x", "# " + "=" * 10]),
           r_old, lambda content: r_bounds(types.SimpleNamespace(**r_patterns), content))


# ---------------------------------------------------------------------------
# EQ38 -- the rules a red-team finding suggests
# ---------------------------------------------------------------------------
_FORGET = r"forget\s+(?:everything|all|your)\s+(?:\w+\s+)*(instructions|rules|guidelines)"
_SYSTEM_TAG = r"</?(?:system|admin|root|override)[^>]*>"


def test_eq38_a_suggested_rule_finds_what_it_found_a_system_tag_rule_also_an_opening_left_without_its_end():
    markers = _defs("redteam/feedback.py", "_INJECTION_MARKERS")["_INJECTION_MARKERS"]
    forget = next(pattern for pattern, name in markers if name == "forget_rules")
    assert not _differences(_FORGET, forget, I, "boolean")
    _agree(_drawn(["forget ", "forget all ", "forget your ", " ", "everything ", "all ", "the ", "a", "rules", "x",
                   "\n", "aforget", ".", "instructions"], 3),
           lambda t: re.search(_FORGET, t, I) is not None, lambda t: re.search(forget, t, I) is not None)
    text = (REPO / "opti_oignon" / "redteam" / "feedback.py").read_text(encoding="utf-8")
    wider = re.findall(r'r"(</\?\(\?:system[^"]*)"', text)
    assert len(wider) == 2 and wider[0] == wider[1] and wider[0] != _SYSTEM_TAG, wider
    found = 0
    for case in _both(_SYSTEM_TAG, I, ["<", "</", "system", "ROOT", ">", " ", "x", "\n", "<system>"]):
        old, new = re.search(_SYSTEM_TAG, case, I), re.search(wider[0], case, I)
        if old is not None:
            found += 1
            assert new is not None and new.start() <= old.start(), case
            if case.count("<") == 1:
                assert new.span() == old.span(), case
    assert found > _CASES // 20, ("witness", found)
    assert re.search(wider[0], "<system role" + " x" * 50, I), "an opening left without its end is read"


# ---------------------------------------------------------------------------
# EQ39, EQ40 -- the statements of the drift ledger
# ---------------------------------------------------------------------------
_STATEMENT = (r"^\s*(?P<subject>.+?)\s+(?P<neg>does not |do not |did not |is not |are not |was not |never )?"
              r"(?P<verb>lives?|works?|is|are|was|were|prefers?|likes?)(?:\s+(?P<particle>in|at|on|called|born on))?"
              r"\s+(?P<object>.+?)\s*\.?\s*$")
_PARTS = ("subject", "neg", "verb", "particle", "object")


def _statements():
    return _defs("memory/drift.py", "_NEGATIONS", "_VERBS", "_PARTICLES", "_Statement", "_statement")


def test_eq39_a_statement_is_read_into_the_parts_the_statement_pattern_gave_it_and_where():
    read, old = _statements()["_statement"], re.compile(_STATEMENT, I)

    def spans(match):
        return None if match is None else tuple((match.start(n), match.end(n), match.group(n)) for n in _PARTS)

    pieces = [" ", "  ", "\n", "\t", "The user", "Ada", "lives", "live", "LIVES", "is", "is not ", "does not ", "never ",
              "in", "born on", "called", "on", "at", "Paris", ".", "x", "works", "prefers", "like", "isn", "inside",
              "live" + _LONG_S, " .", "\n."]
    _agree(_both(_STATEMENT, I, pieces) + _drawn(pieces, 7), lambda t: spans(old.match(t)), lambda t: spans(read(t)))
    assert spans(read("   is happy")) == spans(old.match("   is happy")), "a subject of one blank, as the pattern read"
    _agree(list(_every(["a", " ", "\n", "is", "is not ", "in", ".", "x", "lives", "\t"], 5)),
           lambda t: spans(old.match(t)), lambda t: spans(read(t)))
    for statement in ("Ada is .", "Ada is not is Paris", "a lives in in x", "Ada is not\n", " a is x. ", "a  is \n x"):
        assert spans(read(statement)) == spans(old.match(statement)), statement


def test_eq40_a_statement_reads_what_follows_each_place_once_counted_calls_not_timings():
    for shape in (("a", " ", "is", "\n"), ("a", " ", "lives in", "\n"), ("a b", " ", "x", " "), ("", " ", "is", "")):
        for size in (300, 1200):
            space = _statements()
            verbs = [_Counted(pattern) for pattern in space["_VERBS"]]
            space["_VERBS"] = tuple(verbs)
            subject, blanks, verb, after = shape
            space["_statement"](subject + blanks * size + verb + after * size)
            assert 0 < sum(v.calls for v in verbs) <= 200 + 2 * size, ("the words after a place read again", shape)


# ---------------------------------------------------------------------------
# EQ41 -- the sandbox's subprocess-call check
# ---------------------------------------------------------------------------
def test_eq41_a_python_c_subprocess_call_is_refused_with_the_reason_its_own_check_gives(tmp_path):
    sm, restore = _sandbox(tmp_path)
    try:
        validator = sm.CommandValidator(sm.SandboxConfig(blocked_commands=[], blocked_patterns=[]))
        for call in ("Popen", "call", "run", "check_output", "check_call"):
            command = f"python3 -c 'import subprocess; subprocess.{call}([1])'"
            assert validator.validate(command) == (False, "Blocked python -c with subprocess calls"), command
        assert validator.validate("python -c 'import os; os.system(1)'") == (
            False, "Blocked python -c with subprocess/os.exec"), "the broader check still refuses the rest"
    finally:
        restore()


# ---------------------------------------------------------------------------
# EQ42 -- the readers that take a run at once, on every unit of a head and a character
# ---------------------------------------------------------------------------
_UNIT_CHARACTERS = " \t\na0_.-:,[](){}<>=/\\'\"!?#@*+|;~" + chr(0xE9) + chr(0x3000) + chr(0x2028) + "bxZ9"


def _takers():
    """(name, compiled, head) of each reader that takes a run at once, as the package writes it."""
    found = []
    for module, names, head in (("agent/tool_parsing.py", ("_XML_INVOKE_RE",), "<invoke"),
                                ("agent/tool_parsing.py", ("_XML_PARAM_RE",), "<param"),
                                ("agent/skills.py", ("_FENCE_RE",), "```"),
                                ("agent/teacher.py", ("_SKILL_FENCE_RE",), "```skill"),
                                ("memory/curation.py", ("_THINK_RE",), "<think>"),
                                ("memory/extraction.py", ("_THINK_RE",), "<think>"),
                                ("plugins/markdown-beautifier/entry_point.py", ("_FENCED_BLOCK_RE",), "```"),
                                ("plugins/task-extractor/entry_point.py", ("_IMPERATIVE_VERBS", "_IMPERATIVE_RE"), "\n"),
                                ("plugins/fact-checker/entry_point.py", ("_ENTITY_CLAIM_PATTERN",), "Aa "),
                                ("plugins/chain-of-thought-enforcer/entry_point.py",
                                 ("_ANSWER_MARKERS", "_ANSWER_MARKER_RE", "_REASONING_MARKERS", "_REASONING_MARKER_RE"),
                                 "\n")):
        space = _defs(module, *names)
        found += [(f"{module}:{name}", space[name], head) for name in names if isinstance(space[name], re.Pattern)]
    for module, marker, head in (("context_summary.py", "<think>(?:", "<think>"), ("memory/legacy.py", "<think>(?:", "<think>"),
                                 ("sandbox_tools.py", "<!--(?:", "<!--"), ("self_correction.py", "```(?:json)?", "```json {"),
                                 ("rag_store.py", "(?P<real>)", "<")):
        pattern, flags = _site(module, marker)
        found.append((f"{module}:{marker}", re.compile(pattern, flags), head))
    chunkers = _class_patterns("rag/chunkers.py", "CodeChunker")
    found += [("chunkers:def", chunkers["FUNCTION_PATTERN"], "def f("),
              ("chunkers:class", chunkers["CLASS_BASES_PATTERN"], "class A("),
              ("chunkers:r", _class_patterns("rag/chunkers.py", "RChunker")["FUNCTION_PATTERN"], "f <- function("),
              ("chunkers:sheet", re.compile(*_site("rag/chunkers.py", "Sheet:")), "[Sheet:")]
    return found


def _scan_time(compiled, text):
    """The least of three processor times of a full scan of ``text``, every match read: the time the scan took on
    the processor, not the wall's, which another process sharing the core would lengthen."""
    spent = []
    for _ in range(3):
        start = time.process_time()
        for _match in compiled.finditer(text):
            pass
        spent.append(time.process_time() - start)
    return min(spent)


def _linear(compiled, unit, size=2000):
    """True when four times as many units take well under sixteen times the time (a quadratic reader takes it)."""
    short, long = _scan_time(compiled, unit * size), _scan_time(compiled, unit * size * 4)
    return long <= 8 * short + 0.002


def test_eq42_every_reader_that_takes_a_run_at_once_reads_each_head_and_character_in_linear_time():
    old = re.compile(r"<[^>]+>")
    assert not _linear(old, "< "), "witness: a quadratic reader is told from a linear one"
    takers = _takers()
    assert len(takers) == 20, [name for name, _compiled, _head in takers]
    slow = []
    for name, compiled, head in takers:
        assert "real" in compiled.groupindex, name
        for character in _UNIT_CHARACTERS:
            if not _linear(compiled, head + character) and not _linear(compiled, head + character):
                slow.append((name, head + character))
    assert not slow, slow


# ---------------------------------------------------------------------------
# EQ43 -- the section, sheet, repository and tag readers, the repository references read run by run (gives way
# for EQ24)
# ---------------------------------------------------------------------------
_REPO_REF = r"(?<!\w)([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)#(\d+)(?!\w)"
_REPO_REF_BEFORE = r"(?<!\w)(?:(?<![.-])|(?<=\w[.-]))([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)#(\d+)(?!\w)"


def test_eq43_the_section_sheet_repository_and_tag_readers_read_as_before_the_references_one_run_at_a_time():
    refs = _defs("plugins/github-connector/entry_point.py", "_REF_RUN", "_REF_DIGITS", "_word",
                 "_owner_repo_refs")["_owner_repo_refs"]
    old = re.compile(_REPO_REF)

    def before(text):
        return [(m.start(), m.group(1), m.group(2)) for m in old.finditer(text)]

    pieces = ["a", "B", "1", "_", ".", "-", "/", "#", "#1", "12", " ", "\n", "x/y#3", chr(0xE9), chr(0x663), "!"]
    _agree(_both(_REPO_REF, 0, pieces), before, refs)
    _agree(list(_every(["a", "1", ".", "/", "#", " ", chr(0xE9)], 6)), before, refs)
    for unit in ("aa.", "a.-", "1.", "a-b."):
        short, long = (_cpu_of(refs, unit * size) for size in (2000, 8000))
        assert long <= 8 * short + 0.002, ("the reader reads a run once", unit, short, long)
    assert not _linear(re.compile(_REPO_REF_BEFORE), "aa."), "witness: the pattern it replaced read a run per start"
    tree = ast.parse((REPO / "opti_oignon" / "rag" / "chunkers.py").read_text(encoding="utf-8"))
    sections = [n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
                and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)
                and n.args[0].value.startswith("^# ")]
    assert len(sections) == 2 and sections[0] == sections[1], sections
    assert not _differences(_SECTION, sections[0], M, "exact")
    split, old_split = _sheet_split(), re.compile(_SHEET)
    _agree(_both(_SHEET, 0, ["[Sheet:", " ", "a", "]", "\n", "]\n", "x", "\t", "[Sheet: a]\n"]),
           old_split.split, split, hit=lambda sheets: len(sheets) > 1)
    assert split("[Sheet: a" * 300 + "]\n") == old_split.split("[Sheet: a" * 300 + "]\n")
    tags = _defs("sandbox_tools.py", "_TAG_HEAD_RE", "_TAG_END_RE", "_tags")["_tags"]
    _agree(_both(_TAG, 0, ["<", "</", "< ", "a", "b1", "/", ">", " ", "\n", '"', "'", "x=", "<a", "/>", "\t"]),
           re.compile(_TAG).findall, tags)
    _agree(list(_every(["<", "/", "a", " ", ">", '"', "'"], 6)), re.compile(_TAG).findall, tags)


def _cpu_of(read, text):
    """The least of three processor times of ``read(text)``."""
    best = None
    for _ in range(3):
        start = time.process_time()
        read(text)
        spent = time.process_time() - start
        best = spent if best is None else min(best, spent)
    return best


# ---------------------------------------------------------------------------
# EQ44 -- the rm check (gives way for EQ25, whose claim reader is no longer delivered)
# ---------------------------------------------------------------------------
def test_eq44_the_rm_check_answers_as_before():
    _hold([("sandbox_manager.py", r"(?!workspace)", r"\brm\b.*\s+/(?!workspace)", "boolean")])


# ---------------------------------------------------------------------------
# EQ45 -- the readers that take a run at once, as delivered (gives way for EQ42)
# ---------------------------------------------------------------------------
def _takers_delivered():
    """(name, compiled, head) of each reader that takes a run at once, as the package delivers it: the readers of
    ``_takers`` but the fact checker's, no longer delivered."""
    found = []
    for module, names, head in (("agent/tool_parsing.py", ("_XML_INVOKE_RE",), "<invoke"),
                                ("agent/tool_parsing.py", ("_XML_PARAM_RE",), "<param"),
                                ("agent/skills.py", ("_FENCE_RE",), "```"),
                                ("agent/teacher.py", ("_SKILL_FENCE_RE",), "```skill"),
                                ("memory/curation.py", ("_THINK_RE",), "<think>"),
                                ("memory/extraction.py", ("_THINK_RE",), "<think>"),
                                ("plugins/markdown-beautifier/entry_point.py", ("_FENCED_BLOCK_RE",), "```"),
                                ("plugins/task-extractor/entry_point.py", ("_IMPERATIVE_VERBS", "_IMPERATIVE_RE"), "\n"),
                                ("plugins/chain-of-thought-enforcer/entry_point.py",
                                 ("_ANSWER_MARKERS", "_ANSWER_MARKER_RE", "_REASONING_MARKERS", "_REASONING_MARKER_RE"),
                                 "\n")):
        space = _defs(module, *names)
        found += [(f"{module}:{name}", space[name], head) for name in names if isinstance(space[name], re.Pattern)]
    for module, marker, head in (("context_summary.py", "<think>(?:", "<think>"), ("memory/legacy.py", "<think>(?:", "<think>"),
                                 ("sandbox_tools.py", "<!--(?:", "<!--"), ("self_correction.py", "```(?:json)?", "```json {"),
                                 ("rag_store.py", "(?P<real>)", "<")):
        pattern, flags = _site(module, marker)
        found.append((f"{module}:{marker}", re.compile(pattern, flags), head))
    chunkers = _class_patterns("rag/chunkers.py", "CodeChunker")
    found += [("chunkers:def", chunkers["FUNCTION_PATTERN"], "def f("),
              ("chunkers:class", chunkers["CLASS_BASES_PATTERN"], "class A("),
              ("chunkers:r", _class_patterns("rag/chunkers.py", "RChunker")["FUNCTION_PATTERN"], "f <- function("),
              ("chunkers:sheet", re.compile(*_site("rag/chunkers.py", "Sheet:")), "[Sheet:")]
    return found


def test_eq45_every_delivered_reader_that_takes_a_run_at_once_reads_each_head_and_character_in_linear_time():
    old = re.compile(r"<[^>]+>")
    assert not _linear(old, "< "), "witness: a quadratic reader is told from a linear one"
    takers = _takers_delivered()
    assert len(takers) == 19, [name for name, _compiled, _head in takers]
    slow = []
    for name, compiled, head in takers:
        assert "real" in compiled.groupindex, name
        for character in _UNIT_CHARACTERS:
            if not _linear(compiled, head + character) and not _linear(compiled, head + character):
                slow.append((name, head + character))
    assert not slow, slow
