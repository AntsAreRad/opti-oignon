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
"""
import ast
import itertools
import random
import re
import sys
import time
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
