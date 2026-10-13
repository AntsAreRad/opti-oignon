#!/usr/bin/env python3
"""Contracts for the engine without backtracking read by the native core: the programs ``linear_engine`` compiles are
read in Rust, by strangling, with the Python read as the reference.

The compilation stays in Python (the parse of ``re``, the sets of characters read from ``re``, the Pike programs,
the refusals by name); the native core receives the programs as data and reads them step for step as the reference
does. Its answers must be the reference's, and so ``re``'s, and its count of work must be the reference's to the
unit: the contracts that hold the reference to linear work then hold the native read by the same numbers.

  * LR1 -- every call answers as the reference and as ``re`` do, with the
    same count of work, on the seeded corpora of LE1 and LE29 read by the
    native core.
  * LR2 -- the native core reads a pattern when it is built and reads the
    programs' format, and says so; a pattern it does not read (groups in a
    positive lookaround, more than 64 sets of characters), a core of
    another format, or a read not asked for, is read by the reference, by
    name; a program out of shape is refused by the core, by name.
  * LR3 -- the native read's work doubles with the text where the
    reference's does: it is the reference's count, at three sizes, for a
    search, every match in turn and the lookarounds.
  * LR4 -- windows of two positions and a cache of three moves change no
    answer and no count of work of the native read.
  * LR5 -- a long native read releases the interpreter: another thread runs
    while one native step reads (a pass, or the matches of a search); threads
    that read one pattern at once get the answers each would get alone; and
    a finditer iterator shared by threads gives each match once.
  * LR6 -- a text is read as the code points Python holds, whatever its
    width: a lone surrogate (which no UTF-8 conversion takes) and a
    character beyond the basic plane are one position each, as for ``re``.
  * LR7 -- what a read keeps and bounds is taken when it begins: the classes
    kept are let go between reads, never during one, and the bounds of a
    text's read hold for every later pass of that text; so a long-lived
    pattern, a small cache of classes, and a bound changed between two
    reads of one text change no answer and no count of work.
  * LR8 -- a failure of the native core is never the caller's: patterns
    compiled at once by several threads are read; a program out of shape
    is refused by the core by name, never a crash; a core that fails in a
    read hands that read, and the pattern's next ones, to the reference.
  * LR9 -- the native core is loaded only by the interpreter ABI it was
    built for: the artefact carries that interpreter's SOABI, read from the
    file before anything is loaded; one built for another, or carrying no
    stamp, is not loaded (one built under 3.13 crashes 3.14 at its first
    call), and the reference reads.
  * LR10 -- a step of finditer reads up to the end of its own match and no
    further, natively as in the reference: the work after each step is the
    reference's, and a long match later in the text is not read before it
    is asked for.

LR1 to LR10 need the built artefact, as the native core's contracts do (``scripts/build_oo_core.sh`` run once).

Local-only (the public distribution ships no tests).
"""
import os
import random
import re
import shutil
import subprocess
import sys
import sysconfig
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import isolate, source  # noqa: E402
from test_linear_engine_contracts import _LE24_PATTERNS, _MEASURED, _answers, _corpus  # noqa: E402

pytestmark = pytest.mark.skipif(sys.version_info[:2] not in _MEASURED,
                                reason="the engine without backtracking is not measured on this version of Python")

BUDGET_S = {
    "test_lr1_every_call_answers_as_the_reference_and_re_with_the_same_work": 30.0,
    "test_lr2_the_native_core_reads_what_it_holds_and_the_reference_the_rest_by_name": 6.0,
    "test_lr3_the_native_work_doubles_with_the_text_as_the_reference_s": 20.0,
    "test_lr4_small_windows_and_a_small_cache_change_no_answer_and_no_work": 20.0,
    "test_lr5_a_long_native_read_lets_other_threads_run_and_threads_get_their_own_answers": 10.0,
    "test_lr6_a_text_is_read_as_the_code_points_python_holds": 6.0,
    "test_lr7_what_a_read_keeps_and_bounds_is_taken_when_it_begins": 10.0,
    "test_lr8_a_failure_of_the_native_core_is_never_the_caller_s": 10.0,
    "test_lr9_the_native_core_is_loaded_only_by_the_interpreter_it_was_built_for": 10.0,
    "test_lr10_a_step_of_finditer_reads_up_to_its_own_match_and_no_further": 10.0,
}


def _loaded():
    """(linear_engine, restore): the engine and its native core's loader, loaded from their files in one window. A core
    that does not load fails the contract with the window already restored, so no later contract inherits it."""
    loaded, restore = isolate(targets={"opti_oignon.linear_pattern": source("linear_pattern.py"),
                                       "opti_oignon.linear_engine": source("linear_engine.py"),
                                       "opti_oignon.native": source("native", "__init__.py")})
    engine = loaded["opti_oignon.linear_engine"]
    try:
        assert loaded["opti_oignon.native"].load() is not None, \
            "oo_core is not built: run scripts/build_oo_core.sh on this machine"
    except AssertionError:
        restore()
        raise
    return engine, restore


def _pair(engine, pattern, flags=0):
    """(native, reference): the pattern compiled twice, read by the native core and by the reference."""
    engine.USE_NATIVE = True
    native = engine.compile(pattern, flags)
    engine.USE_NATIVE = False
    try:
        reference = engine.compile(pattern, flags)
    finally:
        engine.USE_NATIVE = True
    return native, reference


def _still_native(native, what):
    """The native core read every call made on ``native``: none failed over to the reference, which would answer
    and count exactly as the reference does, and so hide the core's defect from a contract of equivalence."""
    assert native._native is not None and native.native_reason == "read by the native core", (what, native.native_reason)


# ---------------------------------------------------------------------------
# LR1 -- the corpora of LE1 and LE29
# ---------------------------------------------------------------------------
def test_lr1_every_call_answers_as_the_reference_and_re_with_the_same_work():
    engine, restore = _loaded()
    try:
        read_natively = checks = 0
        for seed, patterns, texts, longest, wide in ((1401, 900, 6, 7, False), (2901, 1500, 6, 8, True)):
            corpus, _refused = _corpus(engine, seed, patterns, texts, longest, wide)
            for number, (pattern, flags, compiled, _read, cases) in enumerate(corpus):
                native, reference = _pair(engine, pattern, flags)
                started = native._native is not None
                read_natively += started
                for index, (text, pos, endpos) in enumerate(cases):
                    whole = index % 3 == 0
                    want = _answers(compiled, text, pos, endpos, whole)
                    assert _answers(reference, text, pos, endpos, whole) == want, (pattern, text, pos, endpos)
                    assert _answers(native, text, pos, endpos, whole) == want, (pattern, flags, text, pos, endpos)
                    assert native.work == reference.work, (pattern, text, native.work, reference.work)
                    checks += 1
                if started:
                    _still_native(native, pattern)
        assert checks == (900 + 1500) * 6
        # Where re's search tries a start (its compiled set of first characters, under the pattern's own type flags):
        # the native search skips the starts re's search skips, where re's match takes them.
        chars = "".join(chr(point) for point in (0x85, 0xE9, 0x660, 0xA0, 0x2028, 0x3000, 0x1D7CE))
        skipped = 0
        for pattern, flags in _LE24_PATTERNS:
            native, reference = _pair(engine, pattern, flags)
            compiled = re.compile(pattern, flags)
            # short texts, and long ones whose skipped starts sit past the first words of the map of starts
            long_texts = [" " * gap + "\x85b" + "\u3000c" * 3 for gap in (31, 63, 64, 100, 200)]
            for text in [chars, "\x85b", "\xe9!", "\u0660x", "!\xe9", "a" + chars + " b", " \x85 ", "x\u3000y"] + long_texts:
                want = _answers(compiled, text)
                assert _answers(native, text) == want == _answers(reference, text), (ascii(pattern), ascii(text))
                assert native.work == reference.work, (pattern, ascii(text))
                found, first = compiled.search(text), compiled.match(text)
                skipped += first is not None and (found is None or found.span() != first.span())
            _still_native(native, pattern)
        assert skipped >= 3, skipped
        # Not blind: the native core reads nearly every pattern of the corpora, and the work it counts is not nil.
        assert read_natively > 0.9 * (900 + 1500), read_natively
    finally:
        restore()


# ---------------------------------------------------------------------------
# LR2 -- the strangler, by name
# ---------------------------------------------------------------------------
def test_lr2_the_native_core_reads_what_it_holds_and_the_reference_the_rest_by_name():
    engine, restore = _loaded()
    try:
        native, _reference = _pair(engine, r"(\w+)\s(?=x)")
        assert native._native is not None and native.native_reason == "read by the native core"
        many = "".join(chr(0x4E00 + k) for k in range(70))      # seventy literals: seventy sets, none of them scanned
        for pattern, why in ((r"a(?=(b))", "groups in a positive lookaround"), (many + "|x", "more than 64 sets")):
            read = engine.compile(pattern)
            assert read._native is None and why in read.native_reason, (pattern, read.native_reason)
            for text in ("ab", "x" + many + many[:3], ""):
                assert _answers(read, text) == _answers(re.compile(pattern), text), pattern
        # A count or a position out of a C ssize_t raises what re raises, and leaves the core in place.
        read = engine.compile("-")
        for call in (lambda r: r.subn("x", "a-b", 2 ** 64), lambda r: r.split("a-b", 2 ** 63),
                     lambda r: r.search("a-b", 2 ** 63), lambda r: r.match("a-b", 0, -2 ** 63 - 1)):
            for target in (read, re.compile("-")):
                with pytest.raises(OverflowError, match="too large to convert to C ssize_t"):
                    call(target)
        assert read._native is not None and read.subn("x", "a-b", 2 ** 63 - 1) == ("axb", 1)
        # re converts a count or a position before it reads the template or the string's type: so does the engine.
        for call in (lambda r: r.subn("\\9", "a-b", 2 ** 63), lambda r: r.subn(5, "a-b", 2 ** 63),
                     lambda r: r.search(5, 2 ** 63), lambda r: r.split(b"a-b", 2 ** 64)):
            for target in (read, re.compile("-")):
                with pytest.raises(OverflowError):
                    call(target)

        class Two:
            def __index__(self):
                return 2

        for target in (read, re.compile("-")):
            assert target.subn("x", "a-b-c-d", Two()) == ("axbxc-d", 2) and target.split("a-b-c", True) == ["a", "b-c"]
        # The bounds by their exact value: 64 sets of characters, 64 tests, read by the core; one more, by the reference.
        for count, why in ((64, None), (65, "more than 64 sets")):
            read = engine.compile("".join(chr(0x4E00 + k) for k in range(count)))
            assert (read._native is not None) == (why is None) and (why is None or why in read.native_reason), count
        for count, why in ((64, None), (65, "more than 64 sets of characters or tests")):
            read = engine.compile("(?<!a)" * count + "x")
            assert len(read._shared.tests) == count and len(read._shared.sets) == 2, count
            assert (read._native is not None) == (why is None) and (why is None or why in read.native_reason), count
            assert _answers(read, "xaxbx") == _answers(re.compile(read.pattern), "xaxbx")
        # The edges of a C ssize_t are taken, as re takes them.
        for target in (engine.compile("x"), re.compile("x")):
            assert target.search("ax", -2 ** 63).span() == target.search("ax", 0, 2 ** 63 - 1).span() == (1, 2)
        # A core that is not loaded, and a plan that cannot be made: declined by name, the reference reads.
        from opti_oignon import native as loader
        loaded = loader.load
        loader.load = lambda: None
        try:
            read = engine.compile("a")
            assert read._native is None and "not loaded here" in read.native_reason and read.search("ba").span() == (1, 2)
        finally:
            loader.load = loaded
        made = engine._native_plan

        def broken(*args):
            raise RuntimeError("a plan that cannot be made")

        engine._native_plan = broken
        try:
            read = engine.compile("a")
            assert read._native is None and "declined it (RuntimeError" in read.native_reason
        finally:
            engine._native_plan = made
        engine.USE_NATIVE = False
        try:
            assert engine.compile("a").native_reason == "not asked"
        finally:
            engine.USE_NATIVE = True
        engine.NATIVE_FORMAT = 2
        try:
            read = engine.compile("a")
            assert read._native is None and "format 1" in read.native_reason, read.native_reason
        finally:
            engine.NATIVE_FORMAT = 1
        # A program out of shape is refused by the core, by name, and never read.
        from opti_oignon import native as loader
        core = loader.load()
        bad_plan = (1, [[(97, 97)]], [], [], [], [], [([(2, 7, -1, -1), (3, -1, -1, -1)], [0, 0], [], False)], (0, 2, False),
                    None)
        with pytest.raises(ValueError, match="a jump out of the program"):
            core.LinearProgram(bad_plan)
        with pytest.raises(ValueError, match="format 9"):
            core.LinearProgram((9,) + bad_plan[1:])
    finally:
        restore()


# ---------------------------------------------------------------------------
# LR3 -- linear work, the reference's count
# ---------------------------------------------------------------------------
def test_lr3_the_native_work_doubles_with_the_text_as_the_reference_s():
    engine, restore = _loaded()
    try:
        for pattern, unit, call in (("(?:a|a)*b", "a", "search"), ("b.*c|b", "b", "finditer"),
                                    (r"(?=.*x)\w", "ab", "search"), (r"(\w+\s?)*$", "ab ", "search"),
                                    (r"(?<=a)b", "ab", "finditer")):
            works = []
            for size in (1000, 2000, 4000):
                native, reference = _pair(engine, pattern)
                text = unit * (size // len(unit))
                for read in (native, reference):
                    found = getattr(read, call)(text)
                    if call == "finditer":
                        list(found)
                assert native.work == reference.work, (pattern, size, native.work, reference.work)
                _still_native(native, pattern)
                works.append(native.work)
            assert works[1] <= 2.2 * works[0] and works[2] <= 2.2 * works[1], (pattern, works)
    finally:
        restore()


# ---------------------------------------------------------------------------
# LR4 -- small windows, a small cache
# ---------------------------------------------------------------------------
def test_lr4_small_windows_and_a_small_cache_change_no_answer_and_no_work():
    engine, restore = _loaded()
    try:
        # The corpus of LE1, read under windows of one to three positions and caches of one to three moves: the
        # restarts of the caches (the numbers of states let go) and the windows read again, in every pattern.
        corpus, _refused = _corpus(engine, 1401, 300, 4, 9)
        for number, (pattern, flags, compiled, _read, cases) in enumerate(corpus):
            engine.WINDOW, engine.MOST_MOVES = 1 + number % 3, 1 + number // 3 % 3
            native, reference = _pair(engine, pattern, flags)
            if native._native is None:
                continue
            for text, pos, endpos in cases:
                want = _answers(compiled, text, pos, endpos)
                assert _answers(native, text, pos, endpos) == want == _answers(reference, text, pos, endpos), pattern
                assert native.work == reference.work, (pattern, text, engine.WINDOW, engine.MOST_MOVES)
            _still_native(native, pattern)
        engine.WINDOW, engine.MOST_MOVES = 2, 3
        rng = random.Random(4)
        patterns = [r"(a|)*b", r"b.*c|b", r"(?<=a|b)\w+", r"(?=\w*x)\w+", r"\b\w+\b", r"(?:x*)*y", r"(\w+\s?)+$",
                    r"(?<!a)b(?!c)", r"^\s*(\w+)", r"(?m)^a|b$"]
        for pattern in patterns:
            native, reference = _pair(engine, pattern)
            assert native._native is not None, pattern
            for _ in range(40):
                text = "".join(rng.choice("abcx \n") for _ in range(rng.randint(0, 30)))
                want = _answers(re.compile(pattern), text)
                assert _answers(reference, text) == want and _answers(native, text) == want, (pattern, text)
                assert native.work == reference.work, (pattern, text)
            _still_native(native, pattern)
    finally:
        restore()


# ---------------------------------------------------------------------------
# LR5 -- one read at a time
# ---------------------------------------------------------------------------
def test_lr5_a_long_native_read_lets_other_threads_run_and_threads_get_their_own_answers():
    engine, restore = _loaded()
    try:
        # One long native step (the backward pass of a long text, one call into Rust): another thread counts while
        # it runs. The interpreter hands the GIL over only every half second here, so the other thread can count
        # during the step only if the step itself releases the GIL.
        long_read, _reference = _pair(engine, r"(?:a|b)*c|\w+x")
        text = "ab " * 600_000
        counted, go, done = [0], threading.Event(), threading.Event()

        def count():
            go.wait()
            while not done.is_set():
                counted[0] += 1

        # The read itself (the classes of the text, the lookarounds' tables): one native step, the GIL released.
        switch = sys.getswitchinterval()
        sys.setswitchinterval(0.5)
        counter = threading.Thread(target=count)
        counter.start()
        try:
            go.set()
            context = long_read._context(text, len(text))
            during = counted[0]
        finally:
            done.set()
            counter.join()
            sys.setswitchinterval(switch)
        assert during > 100, ("the other thread did not run while the text was classed", during)
        counted, go, done = [0], threading.Event(), threading.Event()

        def count():
            go.wait()
            while not done.is_set():
                counted[0] += 1

        switch = sys.getswitchinterval()
        sys.setswitchinterval(0.5)
        counter = threading.Thread(target=count)
        counter.start()
        try:
            go.set()
            context.read.pass_(False)
            during = counted[0]
        finally:
            done.set()
            counter.join()
            sys.setswitchinterval(switch)
        assert during > 100, ("the other thread did not run during the native step", during)
        # The matches of a search, found in one native step (findall's), let another thread run as well.
        words, _reference = _pair(engine, r"\w+")
        context = words._context(text, len(text))
        context.read.pass_(False)
        counted[0] = 0
        go.clear()
        done.clear()
        sys.setswitchinterval(0.5)
        counter = threading.Thread(target=count)
        counter.start()
        try:
            go.set()
            found, _at, _refuse, finished = context.read.steps(0, None, 0)
            during = counted[0]
        finally:
            done.set()
            counter.join()
            sys.setswitchinterval(switch)
        assert finished and len(found) == 600_000 and during > 100, (len(found), during)
        # One finditer iterator, shared by threads: every match comes out once, in order within each thread.
        shared_read, _reference = _pair(engine, r"(?:a|a)*b|c")
        steps, taken, errors = shared_read.finditer("c" * 20_000), [], []

        def take():
            try:
                taken.extend(found.start() for found in steps)
            except Exception as exc:  # noqa: BLE001 - the contract reports what escaped
                errors.append(repr(exc))

        switch = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            takers = [threading.Thread(target=take) for _ in range(4)]
            for taker in takers:
                taker.start()
            for taker in takers:
                taker.join()
        finally:
            sys.setswitchinterval(switch)
        assert not errors and sorted(taken) == list(range(20_000)), (errors, len(taken))
        native, _reference = _pair(engine, r"(\w+)\s+(\w+)")
        texts = ["ab cd ef " * (50 + k) for k in range(8)]
        alone = [[m.span(2) for m in re.finditer(r"(\w+)\s+(\w+)", text)] for text in texts]
        got = [None] * len(texts)

        def read(k):
            got[k] = [m.span(2) for m in native.finditer(texts[k])]

        threads = [threading.Thread(target=read, args=(k,)) for k in range(len(texts))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert got == alone
    finally:
        restore()



# ---------------------------------------------------------------------------
# LR6 -- code points, whatever their width
# ---------------------------------------------------------------------------
def test_lr6_a_text_is_read_as_the_code_points_python_holds():
    engine, restore = _loaded()
    try:
        texts = ["a\ud800b", "\udfff\ud800", "x\U0001f600y\U0001f600", "\xe9\ud800\U0001f600 ab", "\ud800" * 5, ""]
        for pattern in (r".", r"\w+", r"[^a]+", r"(?<=\ud800)b", r"\b\w", r"(.)(?=\U0001f600)", r"(?s).{2}$", r"\W+"):
            native, reference = _pair(engine, pattern)
            assert native._native is not None, pattern
            for text in texts:
                want = _answers(re.compile(pattern), text)
                assert _answers(native, text) == want == _answers(reference, text), (pattern, ascii(text))
                assert native.work == reference.work, (pattern, ascii(text))
            _still_native(native, pattern)
        # Not blind: a lone surrogate is a text no UTF-8 conversion takes.
        with pytest.raises(UnicodeEncodeError):
            "a\ud800b".encode("utf-8")
    finally:
        restore()


# ---------------------------------------------------------------------------
# LR7 -- classes and bounds, taken when a read begins
# ---------------------------------------------------------------------------
def test_lr7_what_a_read_keeps_and_bounds_is_taken_when_it_begins():
    engine, restore = _loaded()
    try:
        # A long-lived pattern and a small cache of classes: many texts of many distinct characters.
        engine.MOST_CLASSES = 3
        rng = random.Random(7)
        for pattern in (r"(?s)(.)+", r"\w+", r"(?<=\w)\W", r"\b\w+\b|(?=\d)"):
            native, reference = _pair(engine, pattern)
            for _ in range(12):
                text = "".join(chr(rng.choice((0x61, 0x4E00, 0x1F600, 0x30, 0x20)) + rng.randrange(40))
                               for _ in range(rng.randint(0, 60)))
                assert _answers(native, text) == _answers(reference, text) == _answers(re.compile(pattern), text)
                assert native.work == reference.work, (pattern, native.work, reference.work)
            _still_native(native, pattern)
        # Reads made between two steps of a search, which let the kept classes go: the search counts what it would
        # count alone.
        for pattern in (r"\w", r"(?<=\w)\W|\d"):
            native, reference = _pair(engine, pattern)
            for read in (native, reference):
                steps = read.finditer("abcdef12 gh" * 3)
                next(steps)
                read.search("xyz\u4e00\u4e01")
                read.findall("\U0001f600 qq")
                list(steps)
            assert native.work == reference.work, (pattern, native.work, reference.work)
        # The bound on kept classes changed between the reads of one pattern: each read takes the bound of its own time.
        native, reference = _pair(engine, r"\w+|(?=\d)")
        for read in (native, reference):
            for bound, text in ((3, "abc def"), (1 << 16, "ghi jkl \u4e00"), (0, "mno"), (2, "abc pqr")):
                engine.MOST_CLASSES = bound
                read.findall(text)
        assert native.work == reference.work, (native.work, reference.work)
        engine.MOST_CLASSES = 1 << 16
        # A bound changed between two reads of one text: the text's later passes keep the bounds it began with.
        for pattern, text, bound in ((r"(\w+\s?)*x", "ab cd " * 40, ("MOST_MOVES", 3)),
                                     (r"[ab ]*x", "ab " * 200, ("WINDOW", 2))):
            native, reference = _pair(engine, pattern)
            saved = getattr(engine, bound[0])
            try:
                for read in (native, reference):
                    setattr(engine, bound[0], saved)
                    read.search(text)
                    setattr(engine, bound[0], bound[1])
                    read.fullmatch(text, 301 if bound[0] == "WINDOW" else 0)
            finally:
                setattr(engine, bound[0], saved)
            assert native.work == reference.work, (pattern, bound, native.work, reference.work)
    finally:
        restore()


# ---------------------------------------------------------------------------
# LR8 -- the native core's failures are not the caller's
# ---------------------------------------------------------------------------
class PanicException(BaseException):
    """A stand-in for pyo3's panic, a BaseException the reference takes for a failure of the core."""


def test_lr8_a_failure_of_the_native_core_is_never_the_caller_s():
    engine, restore = _loaded()
    try:
        # Several threads compile their first patterns at once: each is read by the core.
        engine._WORD_RANGES = None
        reasons, errors = [], []

        def compile_one(k):
            try:
                reasons.append(engine.compile(rf"\w{{{k + 1}}}x").native_reason)
            except Exception as exc:  # noqa: BLE001 - the contract reports what escaped
                errors.append(repr(exc))

        threads = [threading.Thread(target=compile_one, args=(k,)) for k in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors and reasons == ["read by the native core"] * 6, (errors, reasons)
        # Programs out of shape: refused by the core, by name, and the interpreter stands.
        from opti_oignon import native as loader
        core = loader.load()

        def plan(code, depth, ncaps=2, looks=(), programs=()):
            return (1, [[(97, 97)]], [], [], [(1, 0, True, False)] if looks else [], list(looks),
                    [(code, depth, [], False)] + list(programs), (0, ncaps, False), None)

        good = [(6, 0, -1, -1), (0, 0, -1, -1), (6, 1, -1, -1), (3, -1, -1, -1)]
        core.LinearProgram(plan(good, [0, 0, 0, 0]))
        for bad, why in ((plan([(6, 99, -1, -1)] + good[1:], [0, 0, 0, 0]), "a mark out of the program"),
                         (plan([(7, -5, -1, -1)] + good[1:], [0, 0, 0, 0]), "a mark out of the program"),
                         (plan(good, [0, 1 << 40, 0, 0]), "a depth out of the program"),
                         (plan(good, [0, 0, 0, 0], ncaps=1), "a count of groups out of shape"),
                         (plan(good, [0, 0, 0, 0], looks=[(False, 0)]), "without its own program"),
                         (plan(good, [0, 0, 0, 0], looks=[(False, 1)],
                               programs=[([(5, 0, -1, 0), (3, -1, -1, -1)], [0, 0], [(0, 0, True)], False)]),
                          "a table not yet made"),
                         (plan([(6, 2, -1, -1)] + good[1:], [0, 0, 0, 0]), "a mark out of the program"),
                         (plan([(7, 5, -1, -1)] + good[1:], [0, 0, 0, 0]), "a mark out of the program"),
                         (plan([good[0], (8, 5, 2, 3)] + good[2:], [0, 0, 0, 0]), "a check out of the program"),
                         (plan([good[0], (8, 0, 4, 3)] + good[2:], [0, 0, 0, 0]), "a check out of the program"),
                         (plan(good, [0, 0, 5, 0]), "a depth out of the program"),
                         (plan([good[0], (0, 1, -1, -1)] + good[2:], [0, 0, 0, 0]), "a step on a set it does not hold"),
                         (plan([good[0], (4, 0, -1, -1)] + good[2:], [0, 0, 0, 0]), "a test it does not hold"),
                         (plan([good[0], (1, 2, 4, -1)] + good[2:], [0, 0, 0, 0]), "a split out of the program"),
                         (plan([good[0], (2, 4, -1, -1)] + good[2:], [0, 0, 0, 0]), "a jump out of the program"),
                         (plan(good, [0, 0, 0, 0], ncaps=1 << 17), "a count of groups out of shape"),
                         ((1, [[(98, 98), (97, 97)]], [], [], [], [], [(good, [0, 0, 0, 0], [], False)], (0, 2, False),
                           None), "out of order"),
                         ((1, [[(97, 97)]], [], [], [], [], [(good, [0, 0, 0, 0], [], False)], (0, 2, False),
                           [(5, 9), (7, 12)]), "out of order"),
                         ((1, [[(97, 97)]] * 65, [], [], [], [], [(good, [0, 0, 0, 0], [], False)], (0, 2, False),
                           None), "more than 64 sets"),
                         ((1, [[(97, 97)]], [], [], [(0, 9, False, False)], [], [(good, [0, 0, 0, 0], [], False)],
                           (0, 2, False), None), "an anchor it does not know"),
                         ((1, [[(97, 97)]], [], [], [], [], [(good, [0, 0, 0, 0], [(0, 0, True)], False)], (0, 2, False),
                           None), "a lookaround test it does not hold")):
            with pytest.raises(ValueError, match=why):
                core.LinearProgram(bad)
        # A core that fails in a read: the reference reads it, and the pattern's next reads, and says why.
        for failure in (ValueError("linear read: a read panicked"), PanicException("index out of bounds")):
            read = engine.compile(r"(\w+)\s(\w+)")
            assert read._native is not None

            class Failing:
                def read(self, *args, failure=failure):
                    raise failure

            read._native = Failing()
            for call in ("search", "match", "fullmatch", "findall"):
                assert _answers(read, "ab cd ef") == _answers(re.compile(r"(\w+)\s(\w+)"), "ab cd ef"), call
            assert read._native is None and "the native core failed" in read.native_reason, read.native_reason
        # Two live searches when the core fails: each goes on with the reference, from where it stood.
        read = engine.compile(r"(\w+)\s")
        first, second = read.finditer("ab cd ef gh "), read.finditer("ij kl mn ")
        assert next(first).group(1) == "ab" and next(second).group(1) == "ij"

        class FailingRead:
            work = 0

            def steps(self, *args):
                raise PanicException("a read panicked")

        for steps in (first, second):
            steps._state.context.read = FailingRead()
        assert [found.group(1) for found in first] == ["cd", "ef", "gh"]
        assert [found.group(1) for found in second] == ["kl", "mn"]
        assert read._native is None and "the native core failed" in read.native_reason
        # An error of the Python around the core (memory for the results) is the caller's, as re's would be: the
        # core stays, and the search has not moved on.
        read = engine.compile(r"(\w)(\w)")
        made = read._found

        def short_of_memory(*args):
            read._found = made
            raise MemoryError

        read._found = short_of_memory
        with pytest.raises(MemoryError):
            read.subn("<\\2>", "ab cd ef gh ij kl")
        assert read.subn("<\\2>", "ab cd ef gh ij kl") == re.subn(r"(\w)(\w)", "<\\2>", "ab cd ef gh ij kl")
        assert read.native_reason == "read by the native core"
        # match and fullmatch read through other calls into the core (the pass of the end only, a start, the machine):
        # a failure of any of them hands the read to the reference.
        class FailingProgram:
            def __init__(self, inner, failing):
                self.inner, self.failing = inner, failing

            def read(self, *args):
                return FailingCalls(self.inner.read(*args), self.failing)

        class FailingCalls:
            def __init__(self, inner, failing):
                self.inner, self.failing = inner, failing

            def __getattr__(self, name):
                if name == self.failing:
                    def fail(*args):
                        raise PanicException(f"{name} panicked")
                    return fail
                return getattr(self.inner, name)

        for failing in ("pass_", "starts", "pike"):
            for call in ("match", "fullmatch"):
                read = engine.compile(r"(\w+)\s?")
                read._native = FailingProgram(read._native, failing)
                got = getattr(read, call)("ab ")
                want = getattr(re.compile(r"(\w+)\s?"), call)("ab ")
                assert (got.span(), got.groups()) == (want.span(), want.groups()), (failing, call)
                assert read._native is None and "the native core failed" in read.native_reason, (failing, call)
        # One of two live searches fails: the other goes on with the reference from its next step.
        read = engine.compile(r"(\w+)\s")
        first, second = read.finditer("ab cd ef "), read.finditer("gh ij kl ")
        next(first), next(second)
        first._state.context.read = FailingRead()
        assert [found.group(1) for found in first] == ["cd", "ef"]
        assert type(second._state.context) is engine._NativeContext
        assert [found.group(1) for found in second] == ["ij", "kl"]
        assert type(second._state.context) is not engine._NativeContext, "the second search left the core"
        # pyo3 reports a lack of memory for the Python objects of the results as a panic, "Python API call failed": the
        # caller's MemoryError too, and the core stays.
        read = engine.compile(r"\w+")
        steps = read.finditer("ab cd ef")
        next(steps)

        class ConversionFailed:
            work = 0

            def steps(self, *args):
                raise PanicException("Python API call failed")

        live = steps._state.context.read
        steps._state.context.read = ConversionFailed()
        with pytest.raises(MemoryError):
            next(steps)
        assert read._native is not None and read.native_reason == "read by the native core"
        steps._state.context.read = live
        assert [found.group() for found in steps] == ["cd", "ef"]
        # A read whose count of work fails after a failure of the core: the last count stands, nothing escapes.
        read = engine.compile(r"\w+")
        steps = read.finditer("ab cd ef")
        next(steps)

        class NoCount:
            @property
            def work(self):
                raise PanicException("a read panicked")

            def steps(self, *args):
                raise PanicException("a read panicked")

        steps._state.context.read = NoCount()
        assert [found.group() for found in steps] == ["cd", "ef"] and read._native is None
        # A lack of memory in a call into the core is the caller's MemoryError, and the core stays.
        read = engine.compile(r"\w+")

        class ShortOfMemory:
            def read(self, *args):
                raise MemoryError("no memory for the classes")

        kept = read._native
        read._native = ShortOfMemory()
        with pytest.raises(MemoryError):
            read.findall("ab cd")
        read._native = kept
        assert read.findall("ab cd") == ["ab", "cd"] and read.native_reason == "read by the native core"
        # Not blind: an interrupt is not taken for a failure of the core.
        read = engine.compile("a")

        class Interrupted:
            def read(self, *args):
                raise KeyboardInterrupt

        read._native = Interrupted()
        with pytest.raises(KeyboardInterrupt):
            read.search("a")
        # A buffer the core cannot have under an address-space limit: MemoryError for the caller, never an abort.
        child = subprocess.run([sys.executable, "-c", _SHORT_OF_MEMORY, str(Path(__file__).resolve().parents[1])],
                               capture_output=True, text=True, timeout=120)
        assert child.returncode == 0 and child.stdout.strip() == "MemoryError", (child.returncode, child.stdout,
                                                                                 child.stderr[-300:])
    finally:
        restore()


_SHORT_OF_MEMORY = """
import resource, sys
sys.path.insert(0, sys.argv[1])
from opti_oignon import linear_engine
read = linear_engine.compile(r"\\w+\\d")
assert read._native is not None, read.native_reason
text = "ab " * 20_000_000
with open("/proc/self/status") as status:
    size = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmSize"))
resource.setrlimit(resource.RLIMIT_AS, (size + 20 * 2 ** 20, resource.RLIM_INFINITY))
try:
    read.search(text)
    print("read")
except MemoryError:
    print("MemoryError")
"""


# ---------------------------------------------------------------------------
# LR9 -- the interpreter an artefact was built for
# ---------------------------------------------------------------------------
def _loader_in(root, artefact_bytes):
    """The package's loader, copied into a tree of its own with an artefact of the given bytes, loaded afresh."""
    import importlib.util
    native = root / "opti_oignon" / "native"
    native.mkdir(parents=True)
    (native / "__init__.py").write_bytes(source("native", "__init__.py").read_bytes())
    (native / "oo_core.so").write_bytes(artefact_bytes)
    spec = importlib.util.spec_from_file_location(f"lr9_loader_{root.name}", str(native / "__init__.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lr9_the_native_core_is_loaded_only_by_the_interpreter_it_was_built_for(tmp_path, caplog):
    _engine, restore = _loaded()
    try:
        from opti_oignon import native as loader
        artefact = loader._artefact().read_bytes()
        here = sysconfig.get_config_var("SOABI")
        assert loader.stamp(loader._artefact()) == here and loader.load().PYTHON_TAG == here
        # The same bytes, the stamp naming another interpreter (a tag of the same length), or no stamp: not loaded.
        other = here[:-1] + ("0" if here[-1] != "0" else "1")
        mark = f"\noo_core built for python: {here}\n".encode()
        assert artefact.count(mark) == 1, "the artefact carries one stamp"
        cases = {"other": artefact.replace(mark, mark.replace(here.encode(), other.encode())),
                 "unstamped": artefact.replace(mark, b"\n" * len(mark)), "same": artefact}
        for name, data in cases.items():
            caplog.clear()
            module = _loader_in(tmp_path / name, data)
            found = module.load()
            if name == "same":
                # Not blind: the bytes that differ are the stamp's only, and the artefact as built is loaded.
                assert found is not None and found.PYTHON_TAG == here, name
                continue
            assert found is None, f"an artefact {name} is loaded"
            assert any("not for this interpreter" in record.getMessage() for record in caplog.records), name
        # The build script names the interpreter pyo3 builds against (what the stamp is read from), and checks the
        # artefact with that same interpreter: a stand-in cargo writes what it was given, a stand-in loader says
        # which interpreter loads it.
        root = tmp_path / "script"
        (root / "scripts").mkdir(parents=True)
        shutil.copy2(Path(__file__).resolve().parents[1] / "scripts" / "build_oo_core.sh", root / "scripts")
        (root / "rust" / "oo_core").mkdir(parents=True)
        (root / "rust" / "oo_core" / "Cargo.toml").write_text('[package]\nname = "oo_core"\n', encoding="utf-8")
        native = root / "opti_oignon" / "native"
        native.mkdir(parents=True)
        (root / "opti_oignon" / "__init__.py").write_text("", encoding="utf-8")
        (native / "__init__.py").write_text(
            "import sys\nfrom pathlib import Path\nfrom types import SimpleNamespace\n\n\ndef load():\n"
            "    built = (Path(__file__).resolve().parent / 'oo_core.so').read_text().strip()\n"
            "    return SimpleNamespace(VERSION=built + ' checked by ' + sys.executable)\n", encoding="utf-8")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "cargo").write_text(
            "#!/usr/bin/env bash\nwhile [ $# -gt 0 ]; do\n  if [ \"$1\" = \"--manifest-path\" ]; then manifest=\"$2\"; shift; fi\n"
            "  shift\ndone\nmkdir -p \"$(dirname \"$manifest\")/target/release\"\n"
            "echo \"built for $PYO3_PYTHON\" > \"$(dirname \"$manifest\")/target/release/liboo_core.so\"\n", encoding="utf-8")
        (bin_dir / "cargo").chmod(0o755)
        named = bin_dir / "python-for-the-core"
        named.symlink_to(sys.executable)
        env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", PYO3_PYTHON=str(named),
                   PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        run = subprocess.run(["bash", str(root / "scripts" / "build_oo_core.sh")], cwd=root, env=env,
                             capture_output=True, text=True, timeout=120)
        assert run.returncode == 0, run.stderr
        assert run.stdout.strip().splitlines()[-1] == f"oo_core built for {named} checked by {named}", run.stdout
    finally:
        restore()


# ---------------------------------------------------------------------------
# LR10 -- a step of finditer, and no further
# ---------------------------------------------------------------------------
def test_lr10_a_step_of_finditer_reads_up_to_its_own_match_and_no_further():
    engine, restore = _loaded()
    try:
        text = "a" * 10 + "b" + "x" * 50_000 + "y"
        native, reference = _pair(engine, r"a|bx*y")
        steps = {"native": native.finditer(text), "reference": reference.finditer(text)}
        for _ in range(8):
            assert next(steps["native"]).span() == next(steps["reference"]).span()
            assert native.work == reference.work, (native.work, reference.work)
        early = native.work
        assert [found.span() for found in steps["native"]] == [(8, 9), (9, 10), (10, 50_012)]
        # Not blind: the long match, read only when asked for, costs most of the work.
        assert early * 4 < native.work, (early, native.work)
        _still_native(native, r"a|bx*y")
    finally:
        restore()
