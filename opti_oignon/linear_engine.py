"""An engine without backtracking for a regular expression the linear check refuses: the semantics of Python's own
engine, groups included, in time linear in the text for one search and for every match in turn.

A pattern that comes from outside the code is judged by ``linear_pattern`` before it runs. One the check calls
linear runs under ``re``; one it refuses is read here, unless this engine refuses it too. What a read here costs
rests on no premise about how ``re`` walks a pattern.

THE PROGRAM. The parsed pattern is compiled into a Pike program: a step on one set of characters, ordered splits
(alternatives in their order, a greedy repeat before a lazy one), jumps, group marks, anchors, and lookarounds read
from oracle tables. Python's engine takes an optional round of a loop only if the round before it read something,
the first one always, and leaves the loop after an empty one. A thread carries one integer j: the loops of its
chain (the enclosing loops in an optional round whose body can read nothing) from index j inward have read nothing
since their round began. Rounds begin later the deeper they sit, so which of them have read is always an outer part
of the chain: one integer is enough, and a position holds at most m * (d + 1) threads (m instructions, d the depth
of such loops).

THE READ. A pass from the end of the text to its start finds, at every position, the instructions from which a match
can still be completed: a lazy automaton over sets of instructions (a map of bits, one per instruction), its moves
kept up to MOST_MOVES, then started over. The leftmost match starts at the first position where the program's first
instruction is one of them, among the starts re's search tries (re tries a start only in the set it compiles from
the pattern's first class, under the pattern's own type flags: where a group of other type flags holds that class,
re skips starts its match takes, and so does this search). From there the Pike machine follows the threads in order
of priority, groups included, and drops a thread that reads into a place from which no match can be completed. Once a
match is found, every thread that remains is of higher priority and will end in a match of its own, so the machine
stops exactly where the leftmost-first match ends: each match reads its own span only, and ``finditer`` (lazy, as
re's), ``findall``, ``sub`` and ``split`` read in time linear in the text, as one ``search`` does (an engine that
runs one search after another, as the regex crate documents of its own iterators, may take quadratic time there).
The passes of the last text read are kept for its next read, so a reader that takes one match after another from a
moving position reads the text once in all.

WHAT IT COSTS. Time: O(n * m) for a text of n characters and a program of m steps, the constant of Python; where
the states the text visits repeat (a list of words, a class run), the moves are kept and a character costs about
the same whatever m. Memory: the text's own size in bytes for the starts, as much again for each lookaround's table,
m / 8 bytes every WINDOW positions for the backward pass and one window, and at most MOST_MOVES sets of m / 8
bytes for each automaton's moves. The text read is an exact ``str``: a subclass's methods never run during a read.
What a read keeps and bounds is taken when it begins: the classes kept are let go between reads, never during one,
and a text's passes keep the bounds its read began with.

THE NATIVE READ. Where the native core (``oo_core``) is built and reads this format of program, the programs
compiled here are handed to it as data and read in Rust, step for step as here, with the same count of work, the
interpreter released meanwhile; it costs two bytes more per position (the class of each character). A pattern it
does not read (groups in a positive lookaround, more than 64 sets of characters or tests) is read here, by name,
and so is every read after a failure of the core.

LOOKAROUNDS. Each is an oracle table, one pass per lookaround, the innermost first: a lookahead by the backward
pass of its body, a lookbehind by a forward pass. The groups of a positive lookaround are read again where the
winning thread passed it; such a lookaround is passed at most once on a path (in no repeat that may take it twice),
and it is bounded in width (MOST_LOOK_WIDTH), so that this second read costs no more than that width.

THE CHARACTERS. Each set of characters is the one ``linear_pattern`` reads from ``re`` over every code point for the
item as written under its own flags. Characters with the same answers for every set and for the anchors fall into
one class, and the automata move on classes.

REFUSED, NEVER GUESSED (``Unreadable``, a ``re.error`` that names the reason): a back-reference; a conditional
(``re`` can answer from the marks of a path it tried and left); an atomic group or a possessive repeat of more than
one set of characters (of one set, read exactly as a run, then a negative lookahead); groups in a positive
lookaround that a repeat may pass twice, or that is wider than MOST_LOOK_WIDTH; a bytes pattern; a program of more
than MOST_INSTRUCTIONS steps (instructions and rounds of repeats) or MOST_SETS sets of characters; a Python version
this engine was not measured on against ``re``.
"""
import bisect
import operator
import re
import sys
import threading
import types

from opti_oignon import linear_pattern as _check

try:
    import re._compiler as _compiler
except ImportError:  # Python 3.10
    import sre_compile as _compiler

# Module conventions (project-wide).
checkpoint_before_apply = True

# The steps of all the programs of one pattern (instructions, and rounds of repeats, which a body that reads nothing
# makes without an instruction), and its sets of characters, at most: past either, the pattern is refused, so the
# time a compilation takes and the time per character a read may take are bounded too.
MOST_INSTRUCTIONS = 2000
MOST_SETS = 256
# The moves one lazy automaton keeps before it starts over, and the positions between two kept sets of the
# backward pass.
MOST_MOVES = 20_000
WINDOW = 512
# The widest lookaround whose groups are read again.
MOST_LOOK_WIDTH = 256
# The strings of the prefilter, and their length, at most.
MOST_STRINGS = 64
MOST_LENGTH = 32
# The characters whose class is kept between reads.
MOST_CLASSES = 1 << 16
# The versions of Python this engine was measured on against ``re``. Not 3.10: its ``re`` keeps the groups of a
# negative lookaround that failed, and can report a group that ends before it starts.
MEASURED = ((3, 11), (3, 12), (3, 13), (3, 14))
# The read side in the native core (``oo_core``), when it is built and reads this format of program: a pattern it
# does not read (groups in a positive lookaround, more than 64 sets of characters or tests) is read here.
NATIVE_FORMAT = 1
USE_NATIVE = True

_C = _check._constants
_P = _check._parser
_POSSESSIVE = getattr(_C, "POSSESSIVE_REPEAT", None)
_ATOMIC = getattr(_C, "ATOMIC_GROUP", None)
_SINGLE = _check.SINGLE

CHAR, SPLIT, JMP, MATCH, ASSERT, ORACLE, SAVE, MARK, CHECK, GUARD = range(10)
FAIL = -1
BIG = 1 << 30

_WORD = re.compile(r"\w")
_WORD_ASCII = re.compile(r"\w", re.ASCII)


class Unreadable(re.error):
    """A pattern this engine does not read; ``reason`` names what it holds."""

    def __init__(self, reason):
        super().__init__(f"pattern not read: {reason}")
        self.reason = reason


def _children(op, av):
    """The parts of an item that may hold a group the engine reads (a conditional, or an atomic group or a possessive
    repeat that holds one, is refused when it is built)."""
    if op is _C.SUBPATTERN:
        return [av[-1]]
    if op is _C.BRANCH:
        return list(av[1])
    if op in (_C.MAX_REPEAT, _C.MIN_REPEAT):
        return [av[2]]
    if op in (_C.ASSERT, _C.ASSERT_NOT):
        return [av[1]]
    return []


def _groups_in(seq, out):
    for op, av in seq:
        if op is _C.SUBPATTERN and av[0]:
            out.add(av[0])
        for sub in _children(op, av):
            _groups_in(sub, out)
    return out


def _simple(body):
    """The engine's test for a one-character repeat: one item that reads one character, maybe in a plain group."""
    items = list(body)
    if len(items) != 1:
        return False
    op, av = items[0]
    if op is _C.SUBPATTERN:
        return av[0] is None and _simple(av[-1])
    return op in _SINGLE


class _Look:
    def __init__(self, behind, positive, width, groups):
        self.behind, self.positive, self.width, self.groups = behind, positive, width, groups
        self.index = None
        self.test = None
        self.scan = None       # forward program without groups: the oracle table's
        self.exact = None      # forward program with groups, for a positive lookaround that holds some
        self.slot = None       # the caps slot of the position where the winning thread passed it


class _Shared:
    """What the programs of one pattern share: sets of characters, tests, lookarounds, classes of characters."""

    def __init__(self, flags):
        self.flags = flags
        self.sets, self.set_ids = [], {}
        self.tests, self.test_ids = [], {}
        self.asserts = []          # test ids of the anchors
        self.looks, self.by_body = [], {}
        self.size = 0
        self.ncaps = 0
        self.classes, self.signatures, self.infos = {}, {}, []
        self.contexts = {}
        # what reading the sets may cost, as for the check: past READS passes over every code point, refused
        self.machine = _check._Automaton()

    def set_id(self, op, av, flags):
        """The number of the set one item accepts: the code points ``linear_pattern`` reads from the engine for
        it (the sets the check of the same pattern just read, most often), as sorted disjoint ranges."""
        key = _check._key(op, av, flags)
        found = self.set_ids.get(key)
        if found is None:
            ranges = _check._members(op, av, flags, self.machine)
            found = self.set_ids[key] = self._add(ranges)
        return found

    def _add(self, ranges):
        if len(self.sets) >= MOST_SETS:
            raise Unreadable(f"more than {MOST_SETS} sets of characters")
        self.sets.append(([first for first, _ in ranges], ranges))
        return len(self.sets) - 1

    def never(self):
        """The set that holds no character: the step of a lookahead that never holds, ``(?!)``."""
        found = self.set_ids.get(None)
        if found is None:
            found = self.set_ids[None] = self._add(())
        return found

    def test_id(self, test):
        found = self.test_ids.get(test)
        if found is None:
            found = self.test_ids[test] = len(self.tests)
            self.tests.append(test)
            if test[0] == "at":
                self.asserts.append(found)
        return found

    def classify(self, ch, classes):
        """The class of a character, kept in ``classes``: its answer for every set, whether it ends a line, and
        whether it is a word character in Unicode and in ASCII."""
        members, point = 0, ord(ch)
        for set_id, (firsts, ranges) in enumerate(self.sets):
            at = bisect.bisect_right(firsts, point) - 1
            if at >= 0 and ranges[at][1] >= point:
                members |= 1 << set_id
        signature = (members, ch == "\n", _WORD.fullmatch(ch) is not None, _WORD_ASCII.fullmatch(ch) is not None)
        found = self.signatures.get(signature)
        if found is None:
            found = self.signatures[signature] = len(self.infos)
            self.infos.append(signature)
        classes[ch] = found
        return found

    def anchors(self, before, here, first, last, penultimate, most_moves):
        """The anchors that hold at a position, as bits over their test ids, from the classes on each side."""
        key = (before, here, first, last, penultimate)
        found = self.contexts.get(key)
        if found is None:
            found = 0
            prev = None if before is None else self.infos[before]
            cur = None if here is None else self.infos[here]
            for test_id in self.asserts:
                _, kind, flags = self.tests[test_id]
                if _anchor(kind, flags, prev, cur, first, last, penultimate):
                    found |= 1 << test_id
            if len(self.contexts) >= most_moves:
                self.contexts.clear()
            self.contexts[key] = found
        return found


def _anchor(kind, flags, prev, cur, first, last, penultimate):
    """Whether an anchor holds between a character of class info prev (None at the start) and one of cur (None at
    the end); first, last, penultimate: the position is 0, the end, one before the end."""
    if kind is _C.AT_BEGINNING:
        return first or (bool(flags & re.MULTILINE) and prev[1])
    if kind is _C.AT_BEGINNING_STRING:
        return first
    if kind is _C.AT_END:
        if flags & re.MULTILINE:
            return last or cur[1]
        return last or (penultimate and cur[1])
    if kind is _C.AT_END_STRING:
        return last
    if kind in (_C.AT_BOUNDARY, _C.AT_NON_BOUNDARY):
        if first and last and sys.version_info < (3, 14):
            return False                      # before 3.14 the engine reads neither in an empty text
        word = 3 if flags & re.ASCII else 2
        before = prev is not None and prev[word]
        after = cur is not None and cur[word]
        return (before != after) == (kind is _C.AT_BOUNDARY)
    raise Unreadable(f"an anchor this engine does not model ({kind})")


def _past_anchor(kind, flags, text, pos, end):
    """Whether an anchor holds at a position past the end (pos > end >= 0: match with pos past endpos), read from
    the text itself: neither end of the text is there, and no character after it."""
    if kind is _C.AT_BEGINNING:
        return bool(flags & re.MULTILINE) and text[pos - 1] == "\n"
    if kind is _C.AT_END:
        return bool(flags & re.MULTILINE) and pos < len(text) and text[pos] == "\n"
    if kind in (_C.AT_BOUNDARY, _C.AT_NON_BOUNDARY):
        if end == 0 and sys.version_info < (3, 14):
            return False
        word = _WORD_ASCII if flags & re.ASCII else _WORD
        return (word.fullmatch(text[pos - 1]) is not None) == (kind is _C.AT_BOUNDARY)
    return False


class _Program:
    """One Pike program. exact: groups, empty rounds and the one-character rule past the end are read; otherwise
    the program answers whether a match exists, which none of them changes."""

    def __init__(self, shared, exact):
        self.shared, self.exact = shared, exact
        self.code, self.depth = [], []
        self.looks_with_groups = []
        self.repeats = 0           # the repeats that may take a round twice, around the instruction being built

    def charge(self):
        """Count one step of the program: an instruction, or a round of a repeat (a body that reads nothing, an
        empty group, makes a round without an instruction)."""
        shared = self.shared
        if shared.size >= MOST_INSTRUCTIONS:
            raise Unreadable(f"a program of more than {MOST_INSTRUCTIONS} steps")
        shared.size += 1

    def emit(self, ins, depth):
        self.charge()
        self.code.append(ins)
        self.depth.append(depth)
        return len(self.code) - 1

    def seq(self, seq, flags, chain):
        for op, av in seq:
            self.one(op, av, flags, chain)

    def one(self, op, av, flags, chain):
        exact = self.exact
        if op in _SINGLE:
            self.emit((CHAR, self.shared.set_id(op, av, flags)), chain)
        elif op is _C.SUBPATTERN:
            group = av[0]
            if group and exact:
                self.emit((SAVE, 2 * group), chain)
            self.seq(av[-1], _check._inside(flags, av), chain)
            if group and exact:
                self.emit((SAVE, 2 * group + 1), chain)
        elif op is _C.BRANCH:
            branches, jumps = av[1], []
            for index, branch in enumerate(branches):
                if index < len(branches) - 1:
                    split = self.emit(None, chain)
                    self.seq(branch, flags, chain)
                    jumps.append(self.emit(None, chain))
                    self.code[split] = (SPLIT, split + 1, len(self.code))
                else:
                    self.seq(branch, flags, chain)
            for at in jumps:
                self.code[at] = (JMP, len(self.code))
        elif op in (_C.MAX_REPEAT, _C.MIN_REPEAT):
            self.repeat(op is _C.MAX_REPEAT, av, flags, chain)
        elif _POSSESSIVE is not None and op is _POSSESSIVE:
            self.possessive(av, flags, chain)
        elif _ATOMIC is not None and op is _ATOMIC:
            self.atomic(av, flags, chain)
        elif op is _C.AT:
            self.emit((ASSERT, self.shared.test_id(("at", av, flags))), chain)
        elif op in (_C.ASSERT, _C.ASSERT_NOT):
            look = self.look(op, av, flags)
            slot = look.slot if exact and look.exact is not None else None
            if slot is not None:
                if self.repeats:
                    raise Unreadable("groups in a lookaround that a repeat may pass twice")
                if look not in self.looks_with_groups:
                    self.looks_with_groups.append(look)
            self.emit((ORACLE, look.test, slot, look.index), chain)
        elif op is _C.GROUPREF_EXISTS:
            raise Unreadable("a conditional (re may answer it from a path it left)")
        elif op is _C.FAILURE:
            self.emit((CHAR, self.shared.never()), chain)
        elif op is _C.GROUPREF:
            raise Unreadable("a back-reference")
        else:
            raise Unreadable(f"an item this engine does not model ({op})")

    def repeat(self, greedy, av, flags, chain):
        least, most, body = av
        twice = most > 1
        self.repeats += twice
        try:
            self._repeat(greedy, least, most, body, flags, chain)
        finally:
            self.repeats -= twice

    def _repeat(self, greedy, least, most, body, flags, chain):
        if _simple(body):
            # the engine's one-character repeat fails outright past the end (match with pos past endpos), in a
            # lookaround's body too
            self.emit((GUARD,), chain)
        for _ in range(least):
            self.charge()
            self.seq(body, flags, chain)
        if most == least:
            return
        nullable = self.exact and body.getwidth()[0] == 0
        inner = chain + 1 if nullable else chain
        if most == _C.MAXREPEAT:
            top = self.emit(None, chain)
            begin = len(self.code)
            if nullable:
                self.emit((MARK, chain), inner)
            self.seq(body, flags, inner)
            if not nullable:
                self.emit((JMP, top), inner)
                out = len(self.code)
                self.code[top] = (SPLIT, begin, out) if greedy else (SPLIT, out, begin)
            elif greedy:
                check = self.emit(None, inner)
                out = len(self.code)
                self.code[top] = (SPLIT, begin, out)
                self.code[check] = (CHECK, chain, top, out)
            else:
                split = self.emit(None, inner)
                check = self.emit((CHECK, chain, begin, FAIL), inner)
                out = len(self.code)
                self.code[top] = (SPLIT, out, begin)
                self.code[split] = (SPLIT, out, check)
            return
        rounds = most - least
        if greedy:
            splits, checks = [], []
            for index in range(rounds):
                self.charge()
                splits.append(self.emit(None, chain))
                if nullable:
                    self.emit((MARK, chain), inner)
                self.seq(body, flags, inner)
                if nullable and index < rounds - 1:
                    checks.append(self.emit(None, inner))
            out = len(self.code)
            for at in splits:
                self.code[at] = (SPLIT, at + 1, out)
            for index, at in enumerate(checks):
                self.code[at] = (CHECK, chain, splits[index + 1], out)
            return
        first, begins, pending = self.emit(None, chain), [], []
        for index in range(rounds):
            self.charge()
            begins.append(len(self.code))
            if nullable:
                self.emit((MARK, chain), inner)
            self.seq(body, flags, inner)
            if index < rounds - 1:
                split = self.emit(None, inner)
                pending.append((split, self.emit(None, inner) if nullable else None))
        out = len(self.code)
        self.code[first] = (SPLIT, out, begins[0])
        for index, (split, check) in enumerate(pending):
            following = begins[index + 1]
            if check is None:
                self.code[split] = (SPLIT, out, following)
            else:
                self.code[split] = (SPLIT, out, check)
                self.code[check] = (CHECK, chain, following, FAIL)

    def possessive(self, av, flags, chain):
        """A possessive repeat of one set of characters, read exactly: X{m,}+ is X{m,}(?!X), and X{m,n}+ is X{n},
        or else X{m,n-1}(?!X) (a run shorter than n, taken whole)."""
        least, most, body = av
        if not _simple(body):
            raise Unreadable("a possessive repeat of more than one set of characters")
        if least == most:
            self.repeat(True, av, flags, chain)
            return
        if most == _C.MAXREPEAT:
            self.run_then_not(least, most, body, flags, chain)
            return
        split = self.emit(None, chain)
        self.repeat(True, (most, most, body), flags, chain)
        jump = self.emit(None, chain)
        other = len(self.code)
        self.run_then_not(least, most - 1, body, flags, chain)
        self.code[split] = (SPLIT, split + 1, other)
        self.code[jump] = (JMP, len(self.code))

    def run_then_not(self, least, most, body, flags, chain):
        self.repeat(True, (least, most, body), flags, chain)
        look = self.look(_C.ASSERT_NOT, (1, body), flags)
        self.emit((ORACLE, look.test, None, look.index), chain)

    def atomic(self, body, flags, chain):
        """An atomic group, read exactly where it changes nothing or is a possessive repeat in disguise."""
        items = list(body)
        if all(op in _SINGLE or op is _C.AT for op, _ in items):
            self.seq(body, flags, chain)
            return
        if len(items) == 1 and items[0][0] in (_C.MAX_REPEAT, _C.MIN_REPEAT) and _simple(items[0][1][2]):
            op, (least, most, inner) = items[0]
            if op is _C.MAX_REPEAT:
                self.possessive((least, most, inner), flags, chain)
            else:
                self.repeat(True, (least, least, inner), flags, chain)
            return
        raise Unreadable("an atomic group of more than one set of characters")

    def look(self, op, av, flags):
        direction, body = av
        shared = self.shared
        found = shared.by_body.get(id(body))
        if found is not None and found.positive == (op is _C.ASSERT):
            return found
        numbers = _groups_in(body, set())
        groups = bool(numbers)
        low, high = body.getwidth()
        width = low if direction < 0 else None      # re refuses a lookbehind of variable width
        look = _Look(direction < 0, op is _C.ASSERT, width, groups)
        look.numbers = numbers
        if groups and look.positive and high > MOST_LOOK_WIDTH:
            raise Unreadable(f"groups in a lookaround that may be wider than {MOST_LOOK_WIDTH}")
        scan = _Program(shared, False)
        scan.seq(body, flags, 0)
        scan.emit((MATCH,), 0)
        scan.finish()
        look.scan = scan
        if groups and look.positive:
            exact = _Program(shared, True)
            exact.seq(body, flags, 0)
            exact.emit((MATCH,), 0)
            exact.finish()
            look.exact = exact
            look.slot = shared.ncaps
            shared.ncaps += 1
        look.index = len(shared.looks)
        look.test = shared.test_id(("look", look.index, look.positive))
        shared.looks.append(look)
        shared.by_body[id(body)] = look
        return look

    def finish(self):
        """What the passes read of the program: its steps on characters, its ends, and each step that reads nothing
        taken backward, with the test it needs."""
        code = self.code
        self.chars = [(pc, ins[1]) for pc, ins in enumerate(code) if ins[0] == CHAR]
        self.ends = [pc for pc, ins in enumerate(code) if ins[0] == MATCH]
        back = [[] for _ in code]
        tests = set()
        for pc, ins in enumerate(code):
            kind = ins[0]
            if kind == JMP:
                back[ins[1]].append((pc, None))
            elif kind == SPLIT:
                back[ins[1]].append((pc, None))
                back[ins[2]].append((pc, None))
            elif kind in (SAVE, MARK, GUARD):
                back[pc + 1].append((pc, None))
            elif kind == CHECK:
                for target in ins[2:]:
                    if target != FAIL:
                        back[target].append((pc, None))
            elif kind in (ASSERT, ORACLE):
                back[pc + 1].append((pc, ins[1]))
                tests.add(ins[1])
        self.back = back
        looks = sorted(test_id for test_id in tests if self.shared.tests[test_id][0] == "look")
        self.oracle_tests = [(test_id, self.shared.tests[test_id][1], self.shared.tests[test_id][2])
                             for test_id in looks]
        self.anchored = any(self.shared.tests[test_id][0] == "at" for test_id in tests)
        # a set of instructions is a map of bits, one per instruction: m / 8 bytes, whatever how many it holds
        self.width = len(code) // 8 + 1
        self.empty = bytes(self.width)
        self.cache = _Cache()


class _Cache:
    """The moves of one lazy automaton over sets of instructions, numbered, kept up to MOST_MOVES."""

    def __init__(self):
        self.moves, self.states, self.numbers = {}, [], {}

    def number(self, state):
        found = self.numbers.get(state)
        if found is None:
            found = self.numbers[state] = len(self.states)
            self.states.append(state)
        return found

    def clear(self):
        self.moves.clear()
        self.states.clear()
        self.numbers.clear()


def _within(ranges, ch):
    """Whether a character is in a set of sorted disjoint ranges (firsts, ranges)."""
    firsts, spans = ranges
    point = ord(ch)
    at = bisect.bisect_right(firsts, point) - 1
    return at >= 0 and spans[at][1] >= point


def _holds(state, pc):
    """Whether a set of instructions (a map of bits) holds ``pc``."""
    return state[pc >> 3] >> (pc & 7) & 1


def _completes(program, following, members, bits, may_end):
    """The instructions at a position from which a match can be completed: those that reach, by steps that read
    nothing and whose tests hold here, an end (when a match may end here), or a step on this character (members:
    its answer for every set; None at the end) into one of ``following`` at the next position."""
    stack = []
    if members is not None:
        for pc, set_id in program.chars:
            after = pc + 1
            if members >> set_id & 1 and following[after >> 3] >> (after & 7) & 1:
                stack.append(pc)
    if may_end:
        stack.extend(program.ends)
    back, seen = program.back, bytearray(program.width)
    while stack:
        pc = stack.pop()
        if seen[pc >> 3] >> (pc & 7) & 1:
            continue
        seen[pc >> 3] |= 1 << (pc & 7)
        for before, test_id in back[pc]:
            if test_id is None or bits >> test_id & 1:
                stack.append(before)
    return bytes(seen)


class _Context:
    """What a read of one text needs at every position: the class of each character, the tests that hold."""

    def __init__(self, shared, text, end, oracles):
        self.shared, self.text, self.end, self.oracles = shared, text, end, oracles
        self.work = 0
        # the bounds of the read, as they stand when it begins: every pass of this text, now or at a later read, keeps
        # them (the native core's read takes them at the same moment)
        self.most_moves, self.window = MOST_MOVES, WINDOW
        # the classes of the text, all made when the read begins (as the native core makes them) and kept with it:
        # another read, made between two steps of this one, changes nothing in what this one counts; the classes the
        # pattern keeps for the next read are let go once they number MOST_CLASSES or more
        classes = shared.classes
        for ch in set(text[:end]):
            if ch not in classes:
                self.work += len(shared.sets) + 2
                shared.classify(ch, classes)
        self.classes = classes
        if len(classes) >= MOST_CLASSES:
            shared.classes = {}

    def cls(self, at):
        if at >= self.end or at < 0:
            return None
        return self.classes[self.text[at]]

    def members(self, cls):
        return None if cls is None else self.shared.infos[cls][0]

    def bits(self, program, at, here):
        """The tests of ``program`` that hold at ``at`` (here: the class of the character there), as bits."""
        bits = 0
        if program.anchored:
            end = self.end
            before = self.cls(at - 1) if at > 0 else None
            bits = self.shared.anchors(before, here, at == 0, at == end, at == end - 1, self.most_moves)
        for test_id, index, positive in program.oracle_tests:
            if (self.oracles[index][at] == 1) == positive:
                bits |= 1 << test_id
        return bits


class _Alive:
    """For every position of [start, end], the instructions from which a match can be completed, by one backward
    pass: whether the program's first instruction is one of them is kept for every position, the whole set every
    WINDOW positions; a window is read again, once, when the Pike machine reaches it."""

    def __init__(self, program, context, start, end, final_only):
        self.program, self.context, self.start, self.end, self.final_only = program, context, start, end, final_only
        self.marks, self.window, self.sets, self.base = {}, None, None, None
        self.size = size = context.window
        starts = bytearray(end - start + 1)
        state = program.empty
        context.work += end - start + 1
        for at in range(end, start - 1, -1):
            state = self.move(state, at)
            if state[0] & 1:
                starts[at - start] = 1
            if at == end or not (at - start) % size:
                self.marks[at] = state
        self.starts = starts

    def move(self, following, at):
        program, context = self.program, self.context
        cache = program.cache if not self.final_only else program.final_cache
        cls = context.cls(at)
        bits = context.bits(program, at, cls) if program.anchored or program.oracle_tests else 0
        may_end = not self.final_only or at == self.end
        number = cache.number(following)
        key = (number, cls, bits, may_end)
        found = cache.moves.get(key)
        if found is not None:
            return cache.states[found]
        state = _completes(program, following, context.members(cls), bits, may_end)
        context.work += len(program.code) + len(program.chars)
        if len(cache.moves) >= context.most_moves:
            cache.clear()                 # the move is not kept: its key names a state of the cache let go
            return state
        cache.moves[key] = cache.number(state)
        return state

    def next_start(self, at):
        """The first position from ``at`` on where a match starts, or None."""
        if at > self.end:
            return None
        found = self.starts.find(1, at - self.start)
        return None if found < 0 else self.start + found

    def at(self, at):
        window = (at - self.start) // self.size
        if window != self.window:
            base = self.start + window * self.size
            top = min(base + self.size, self.end)
            state = self.marks[top]
            sets = [None] * (top - base + 1)
            sets[-1] = state
            self.context.work += top - base
            for place in range(top - 1, base - 1, -1):
                state = self.move(state, place)
                sets[place - base] = state
            self.window, self.sets, self.base = window, sets, base
        return self.sets[at - self.base]


class _NativeContext:
    """A read made by the native core: its work, its end, and the read itself."""

    def __init__(self, read, end):
        self.read, self.end, self._work = read, end, 0

    @property
    def work(self):
        """The read's work, or the last count it gave when the core cannot give one now (it failed)."""
        try:
            self._work = self.read.work
        except BaseException as exc:  # noqa: BLE001 - a panic of the core is a BaseException
            if not _native_failure(exc):
                raise
        return self._work


class _NativeAlive:
    """The native core's backward pass of the main program, as the read path asks it: a start, the next start."""

    def __init__(self, read, final_only, end):
        self.read, self.final_only, self.end = read, final_only, end
        self.starts = self

    def __getitem__(self, at):
        return _core(self.read.starts, self.final_only, at)


_AT_CODES = {_C.AT_BEGINNING: 0, _C.AT_BEGINNING_STRING: 1, _C.AT_END: 2, _C.AT_END_STRING: 3, _C.AT_BOUNDARY: 4,
             _C.AT_NON_BOUNDARY: 5}
_WORD_RANGES = None


def _word_ranges():
    """The word characters as re reads them over every code point, in Unicode and in ASCII: read once, by
    ``linear_pattern`` as every set is, and kept (published by one assignment: a thread that compiles at the same
    time sees none or both)."""
    global _WORD_RANGES
    found = _WORD_RANGES
    if found is None:
        machine, word = _check._Automaton(), [(_C.CATEGORY, _C.CATEGORY_WORD)]
        found = ([tuple(span) for span in _check._members(_C.IN, word, re.UNICODE, machine)],
                 [tuple(span) for span in _check._members(_C.IN, word, re.ASCII, machine)])
        _WORD_RANGES = found
    return found


def _native_failure(exc):
    """Whether an exception is one the native core may raise: then the reference reads, and a stop asked for (an
    interrupt, an exit) is never taken for one."""
    return isinstance(exc, Exception) or type(exc).__name__ == "PanicException"


class _CoreFailed(Exception):
    """A call into the native core that failed during a read: the reference reads it instead. Only a call into the
    core raises it: an error of the Python around it (a template, memory for the results) is the caller's, as for
    re."""


def _core(call, *args):
    """A call into the native core: a failure of the core becomes ``_CoreFailed``; a lack of memory, for the core's
    buffers or for the Python objects of its results (which pyo3 reports as a panic, "Python API call failed"), is
    the caller's MemoryError, as it would be for the reference, and the core stays."""
    try:
        return call(*args)
    except MemoryError:
        raise
    except BaseException as exc:  # noqa: BLE001 - a panic of the core is a BaseException
        if not _native_failure(exc):
            raise
        if type(exc).__name__ == "PanicException" and "Python API call failed" in str(exc):
            raise MemoryError(str(exc)) from exc
        raise _CoreFailed(f"{type(exc).__name__}: {exc}") from exc


def _native(shared, main, tried):
    """(the native core's reading of the programs, why) or (None, why the reference reads them): any failure to
    hand the programs over is a decline, by name."""
    try:
        return _native_plan(shared, main, tried)
    except BaseException as exc:  # noqa: BLE001 - a panic of the core is a BaseException
        if not _native_failure(exc):
            raise
        return None, f"the native core declined it ({type(exc).__name__}: {exc})"


def _native_plan(shared, main, tried):
    try:
        from opti_oignon import native
    except ImportError:
        return None, "the native core cannot be loaded"
    core = native.load()
    if core is None or not hasattr(core, "LinearProgram"):
        return None, "the native core is not loaded here (not built, or built for another interpreter)"
    if getattr(core, "LINEAR_FORMAT", None) != NATIVE_FORMAT:
        return None, f"the native core reads programs of format {getattr(core, 'LINEAR_FORMAT', None)}"
    if main.looks_with_groups:
        return None, "groups in a positive lookaround"
    if len(shared.sets) > 64 or len(shared.tests) > 64:
        return None, "more than 64 sets of characters or tests"
    programs, numbers = [], {}

    def number(program):
        if id(program) not in numbers:
            numbers[id(program)] = len(programs)
            programs.append(program)
        return numbers[id(program)]

    number(main)
    looks = [(look.behind, number(look.scan)) for look in shared.looks]
    plans = []
    for program in programs:
        code = []
        for ins in program.code:
            args = [FAIL if arg is None else arg for arg in ins[1:4]]
            code.append((ins[0], *(args + [FAIL] * (3 - len(args)))))
        plans.append((code, list(program.depth), list(program.oracle_tests), bool(program.anchored)))
    tests = []
    for test in shared.tests:
        if test[0] == "at":
            tests.append((0, _AT_CODES[test[1]], bool(test[2] & re.MULTILINE), bool(test[2] & re.ASCII)))
        else:
            tests.append((1, test[1], bool(test[2]), False))
    word, word_ascii = _word_ranges()
    plan = (NATIVE_FORMAT, [list(ranges) for _firsts, ranges in shared.sets], word, word_ascii, tests, looks, plans,
            (0, shared.ncaps, sys.version_info >= (3, 14)), None if tried is None else list(tried[1]))
    try:
        return core.LinearProgram(plan), "read by the native core"
    except ValueError as exc:
        return None, str(exc)


def _oracle(look, context):
    """For every position 0..end, whether the lookaround's body holds there: from it ahead, or up to it behind."""
    program, end = look.scan, context.end
    if not look.behind:
        alive = _Alive(program, context, 0, end, False)
        return alive.starts + b"\x00"
    table = bytearray(end + 2)
    cache, state = program.cache, program.empty
    context.work += end + 1
    for at in range(0, end + 1):
        cls = context.cls(at)
        bits = context.bits(program, at, cls) if program.anchored or program.oracle_tests else 0
        number = cache.number(state)
        key = (number, cls, bits, "ahead")
        found = cache.moves.get(key)
        if found is None:
            state, matched = _forward_set(program, state, context.members(cls), bits)
            context.work += len(program.code) + len(program.chars)
            if len(cache.moves) >= context.most_moves:
                cache.clear()
            else:
                cache.moves[key] = cache.number((state, matched))
        else:
            state, matched = cache.states[found]
        table[at] = matched
    return table


def _forward_set(program, state, members, bits):
    """Whether a match ends at a position from ``state`` (the instructions there, a map of bits) or from a start
    there, and the instructions at the next position once its character (members; None at the end) is read."""
    code = program.code
    stack = [0] + [pc for pc in range(len(code)) if state[pc >> 3] >> (pc & 7) & 1]
    seen, live, matched = bytearray(program.width), bytearray(program.width), False
    while stack:
        pc = stack.pop()
        if seen[pc >> 3] >> (pc & 7) & 1:
            continue
        seen[pc >> 3] |= 1 << (pc & 7)
        ins = code[pc]
        kind = ins[0]
        if kind == JMP:
            stack.append(ins[1])
        elif kind == SPLIT:
            stack.append(ins[1])
            stack.append(ins[2])
        elif kind in (SAVE, MARK, GUARD):
            stack.append(pc + 1)
        elif kind == CHECK:
            stack.extend(target for target in ins[2:] if target != FAIL)
        elif kind in (ASSERT, ORACLE):
            if bits >> ins[1] & 1:
                stack.append(pc + 1)
        elif kind == MATCH:
            matched = True
        elif members is not None and members >> ins[1] & 1:
            after = pc + 1
            live[after >> 3] |= 1 << (after & 7)
    return bytes(live), matched


def _past(look, text, at, end, past):
    """Whether a lookaround holds at a position past the end (match only): every read fails there, so only a body
    that reads nothing can hold."""
    if look.width:
        return False
    code, seen, stack = look.scan.code, set(), [0]
    while stack:
        pc = stack.pop()
        if pc in seen:
            continue
        seen.add(pc)
        ins = code[pc]
        kind = ins[0]
        if kind == JMP:
            stack.append(ins[1])
        elif kind == SPLIT:
            stack.extend(ins[1:3])
        elif kind in (SAVE, MARK):
            stack.append(pc + 1)
        elif kind == CHECK:
            stack.extend(target for target in ins[2:] if target != FAIL)
        elif kind in (ASSERT, ORACLE):
            if _past_test(look.scan.shared, ins[1], text, at, end, past):
                stack.append(pc + 1)
        elif kind == MATCH:
            return True
    return False


def _past_test(shared, test_id, text, at, end, past):
    test = shared.tests[test_id]
    if test[0] == "at":
        return _past_anchor(test[1], test[2], text, at, end)
    return past[test[1]] == test[2]


class Match:
    """A match, as ``re.Match`` answers for it."""

    def __init__(self, pattern, string, text, pos, endpos, caps, lastindex):
        self.re, self.string, self.pos, self.endpos = pattern, string, pos, endpos
        self._text, self._caps, self.lastindex = text, caps, lastindex

    def _index(self, group):
        if isinstance(group, str):
            found = self.re.groupindex.get(group)
            if found is None:
                raise IndexError("no such group")
            return found
        try:
            index = operator.index(group)
        except TypeError:
            raise IndexError("no such group") from None
        if not 0 <= index <= self.re.groups:
            raise IndexError("no such group")
        return index

    def _one(self, index, default=None):
        start = self._caps[2 * index]
        return default if start is None else self._text[start:self._caps[2 * index + 1]]

    def group(self, *groups):
        if not groups:
            return self._one(0)
        if len(groups) == 1:
            return self._one(self._index(groups[0]))
        return tuple(self._one(self._index(group)) for group in groups)

    def __getitem__(self, group):
        return self._one(self._index(group))

    def groups(self, default=None):
        return tuple(self._one(index, default) for index in range(1, self.re.groups + 1))

    def groupdict(self, default=None):
        return {name: self._one(index, default) for name, index in self.re.groupindex.items()}

    def span(self, group=0):
        index = self._index(group)
        start = self._caps[2 * index]
        return (-1, -1) if start is None else (start, self._caps[2 * index + 1])

    def start(self, group=0):
        return self.span(group)[0]

    def end(self, group=0):
        return self.span(group)[1]

    @property
    def regs(self):
        return tuple(self.span(index) for index in range(self.re.groups + 1))

    @property
    def lastgroup(self):
        if self.lastindex is None:
            return None
        for name, index in self.re.groupindex.items():
            if index == self.lastindex:
                return name
        return None

    def expand(self, template):
        return _expand(self.re._template(template), self)

    def __bool__(self):
        return True

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        return f"<linear_engine.Match object; span={self.span()!r}, match={self.group()!r}>"


class _Search:
    """Where a search of one text stands: the next position, the empty match it must not repeat there, the text's
    context and backward pass once made."""

    def __init__(self, string, text, start, end, done):
        self.string, self.text, self.start, self.end, self.done = string, text, start, end, done
        self.at, self.refuse, self.context, self.alive = start, None, None, None


class _Matches:
    """finditer's iterator, as re's: one match at each step, the pattern's lock held while it is found (a step reads
    up to the end of its own match, and no further)."""

    def __init__(self, pattern, state):
        self._pattern, self._state = pattern, state

    def __iter__(self):
        return self

    def __next__(self):
        with self._pattern._lock:
            found = self._pattern._next(self._state)
        if found is None:
            raise StopIteration
        return found


def _search_set(parsed, flags, machine):
    """The set of characters re's search tries a start at, when it compiles one, as sorted disjoint ranges (read as
    every set is, by ``linear_pattern``, under the pattern's own type flags); None when it tries every start. re compiles its charset prefix under the pattern's own type flags, whatever the flags
    of the group the class sits in, so a class in a group of other type flags (``(?a:\\S)``) is tried at fewer places
    than it matches: re's search then skips a start its match takes, and so does this one. A literal prefix skips
    only to where a match may start, and changes no answer."""
    lo = parsed.getwidth()[0]
    if lo == 0 or (flags & re.IGNORECASE and flags & re.LOCALE):
        return None
    prefix, _skip, _whole = _compiler._get_literal_prefix(parsed, flags)
    if prefix:
        return None
    charset = _compiler._get_charset_prefix(parsed, flags)
    if not charset:
        return None
    ranges = _check._members(_C.IN, charset, re.ASCII if flags & re.ASCII else re.UNICODE, machine)
    return [first for first, _ in ranges], ranges


def _ssize(value):
    """An index or a count as re takes one, a C ssize_t: one out of that range raises what re raises."""
    value = operator.index(value)
    if not -sys.maxsize - 1 <= value <= sys.maxsize:
        raise OverflowError("Python int too large to convert to C ssize_t")
    return value


def _expand(parts, match):
    out = []
    for at, part in enumerate(parts):
        if at & 1:
            value = match.group(part)
            if value:
                out.append(value)
        elif part:
            out.append(part)
    return "".join(out)


class Pattern:
    """A compiled pattern read without backtracking; answers as ``re.Pattern`` does for the calls it has."""

    def __init__(self, pattern, flags=0):
        if sys.version_info[:2] not in MEASURED:
            raise Unreadable("this engine was not measured against re on Python "
                             f"{sys.version_info[0]}.{sys.version_info[1]}")
        if not isinstance(pattern, str):
            raise Unreadable("only a text pattern is read")
        re.compile(pattern, flags)
        try:
            parsed = _P.parse(pattern, flags)     # re refuses LOCALE in a text pattern
            self.pattern, self.flags = pattern, int(int(flags) | parsed.state.flags)
            self.groups = parsed.state.groups - 1
            self.groupindex = types.MappingProxyType(dict(parsed.state.groupdict))
            shared = _Shared(self.flags)
            shared.ncaps = 2 * (self.groups + 1)
            main = _Program(shared, True)
            main.emit((SAVE, 0), 0)
            main.seq(list(parsed), self.flags, 0)
            main.emit((SAVE, 1), 0)
            main.emit((MATCH,), 0)
            main.finish()
            main.final_cache = _Cache()
            self._tried = _search_set(parsed, self.flags, shared.machine)
        except _check._Unchecked as exc:
            raise Unreadable(f"it holds {exc}") from None
        except RecursionError:
            raise Unreadable("it is too deep") from None
        self._shared, self._main = shared, main
        self._needles = _required(parsed, self.flags)
        self._templates = {}
        # one read at a time: a read fills the caches of the pattern (moves, classes) that the next one reads
        self._lock = threading.Lock()
        # the backward pass and the lookarounds' tables of the last text read, for the next read of the same text
        # (a reader that takes one token after another from a moving position reads the text once in all)
        self._last = None
        self.work = 0
        self._native, self.native_reason = _native(shared, main, self._tried) if USE_NATIVE else (None, "not asked")

    def __repr__(self):
        return f"linear_engine.compile({self.pattern!r}, {self.flags})"

    # -- a compiled pattern is a value, as re's is: a copy is itself, and it pickles as the call that makes it ------

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return compile, (self.pattern, self.flags)

    def __eq__(self, other):
        return type(other) is type(self) and (other.pattern, other.flags) == (self.pattern, self.flags)

    def __hash__(self):
        return hash((self.pattern, self.flags))

    # -- the calls of re.Pattern ----------------------------------------------------------------------------------

    def search(self, string, pos=0, endpos=sys.maxsize):
        state = self._start(string, pos, endpos)
        with self._lock:
            return self._next(state)

    def match(self, string, pos=0, endpos=sys.maxsize):
        return self._anchored(string, pos, endpos, False)

    def fullmatch(self, string, pos=0, endpos=sys.maxsize):
        return self._anchored(string, pos, endpos, True)

    def finditer(self, string, pos=0, endpos=sys.maxsize):
        return _Matches(self, self._start(string, pos, endpos))

    def findall(self, string, pos=0, endpos=sys.maxsize):
        state, found_all = self._all(string, pos, endpos, raw=True)
        if found_all and type(found_all[0]) is list:
            # read by the native core: the groups' strings straight from their ends, no match objects made
            text, groups = state.text, self.groups
            if groups == 0:
                return [text[caps[0]:caps[1]] for caps in found_all]
            if groups == 1:
                return [text[caps[2]:caps[3]] if caps[2] is not None else "" for caps in found_all]
            return [tuple(text[caps[2 * k]:caps[2 * k + 1]] if caps[2 * k] is not None else ""
                          for k in range(1, groups + 1)) for caps in found_all]
        out = []
        for found in found_all:
            if self.groups == 0:
                out.append(found.group())
            elif self.groups == 1:
                out.append(found.group(1) or "")
            else:
                out.append(tuple(value or "" for value in found.groups()))
        return out

    def split(self, string, maxsplit=0):
        state, matches = self._all(string, 0, sys.maxsize, _ssize(maxsplit))
        text, out, last = state.text, [], 0
        for found in matches:
            out.append(text[last:found.start()])
            out.extend(found.groups())
            last = found.end()
        out.append(text[last:])
        return out

    def sub(self, repl, string, count=0):
        return self.subn(repl, string, count)[0]

    def subn(self, repl, string, count=0):
        count = _ssize(count)                 # re converts the count before it reads the template
        if callable(repl):
            make = repl
        else:
            parts = self._template(repl)
            if len(parts) == 1:
                literal = parts[0] or ""

                def make(found):
                    return literal
            else:

                def make(found):
                    return _expand(parts, found)
        state, matches = self._all(string, 0, sys.maxsize, count)
        text, out, last = state.text, [], 0
        for found in matches:
            out.append(text[last:found.start()])
            item = make(found)
            if item is not None:
                out.append(item)
            last = found.end()
        out.append(text[last:])
        return "".join(out), len(matches)

    def _template(self, repl):
        parts = self._templates.get(repl)
        if parts is None:
            if not isinstance(repl, str):
                raise TypeError(f"expected str instance, {type(repl).__name__} found")
            if sys.version_info >= (3, 12):
                parts = list(_P.parse_template(repl, self))
            else:
                groups, literals = _P.parse_template(repl, self)
                parts, at = [], 0
                for index, group in groups:
                    parts.append("".join(item for item in literals[at:index] if item))
                    parts.append(group)
                    at = index + 1
                parts.append("".join(item for item in literals[at:] if item))
            if len(self._templates) < 64:
                self._templates[repl] = parts
        return parts

    # -- the read -------------------------------------------------------------------------------------------------

    def _bounds(self, string, pos, endpos):
        """(text, start, end): the text to read, as an exact str (a subclass's methods are never called: no code of
        the caller runs while a read holds the lock), pos and endpos clamped as re clamps them."""
        pos, endpos = _ssize(pos), _ssize(endpos)
        if not isinstance(string, str):
            if isinstance(string, (bytes, bytearray, memoryview)):
                raise TypeError("cannot use a string pattern on a bytes-like object")
            raise TypeError(f"expected string or bytes-like object, got '{type(string).__name__}'")
        text = string if type(string) is str else str.__str__(string)
        length = len(text)
        start = min(max(pos, 0), length)
        end = min(max(endpos, 0), length)
        return text, start, end

    def _context(self, text, end):
        if self._native is not None:
            return _NativeContext(_core(self._native.read, text, end, MOST_MOVES, WINDOW, MOST_CLASSES), end)
        context = _Context(self._shared, text, end, [])
        for look in self._shared.looks:
            context.oracles.append(_oracle(look, context))
        return context

    def _absent(self, text, start, end):
        return self._needles is not None and not any(text.find(needle, start, end) >= 0 for needle in self._needles)

    def _pass(self, string, text, end, final_only):
        """The context (classes, lookaround tables) and the backward pass over [0, end] of a text, kept for the
        next read of the same text; and the work this call spent making them."""
        last = self._last
        spent = 0
        if last is None or last[0] is not string or last[1] != end:
            context = self._context(text, end)
            spent = context.work
            last = self._last = [string, end, context, None, None]
        context, index = last[2], 4 if final_only else 3
        if last[index] is None:
            before = context.work
            if type(context) is _NativeContext:
                _core(context.read.pass_, final_only)
                last[index] = _NativeAlive(context.read, final_only, end)
            else:
                last[index] = _Alive(self._main, context, 0, end, final_only)
            spent += context.work - before
        return context, last[index], spent

    def _start(self, string, pos, endpos):
        """The state of a search of ``string`` from pos to endpos, every match in turn (done at once when nothing
        can match there: past the end, or a string every match holds absent)."""
        text, start, end = self._bounds(string, pos, endpos)
        return _Search(string, text, start, end, start > end or self._absent(text, start, end))

    def _all(self, string, pos, endpos, most=0, raw=False):
        """(state, matches): every match in turn, at most ``most`` when it is not 0, none when it is negative (as re
        counts), found under one hold of the lock; raw: the native core's matches as their caps, no match objects."""
        state = self._start(string, pos, endpos)
        if most < 0:
            return state, []
        with self._lock:
            return state, self._nexts(state, most, raw)

    def _native_off(self, failed):
        """After a call into the native core failed during a read: the core is let go for this pattern (the
        reference reads from now on, every live search included, and says why)."""
        self._native, self.native_reason = None, f"the native core failed ({failed}); the reference reads"
        self._last = None

    def _next(self, state):
        """The next match of a search, or None (the lock held)."""
        found = self._nexts(state, 1)
        return found[0] if found else None

    def _nexts(self, state, most, raw=False):
        """The next matches of a search, at most ``most`` (all when 0), the lock held; a failure of the native core
        hands the read to the reference, from where it stands."""
        try:
            return self._nexts_read(state, most, raw)
        except _CoreFailed as failed:
            self._native_off(failed)
            state.context = state.alive = None
            return self._nexts_read(state, most, raw)

    def _nexts_read(self, state, most, raw):
        if state.done:
            return []
        if type(state.context) is _NativeContext and self._native is None:
            # the core was let go for this pattern by another search's failure: this one goes on with the reference
            state.context = state.alive = None
        if state.alive is None:
            state.context, state.alive, spent = self._pass(state.string, state.text, state.end, False)
            self.work += spent
        context, alive = state.context, state.alive
        before = context.work
        try:
            if type(context) is _NativeContext:
                found, at, refuse, done = _core(context.read.steps, state.at, state.refuse, most)
                if not raw:
                    found = [self._found(caps, state.string, state.text, state.start, state.end, context)
                             for caps in found]
                # the search moves on only once its matches are made: an error before leaves it where it stood
                state.at, state.refuse, state.done = at, refuse, done
                return found
            out = []
            while not (most and len(out) >= most):
                one = self._step(state, context, alive)
                if one is None:
                    state.done = True
                    break
                out.append(one)
            return out
        finally:
            self.work += context.work - before

    def _step(self, state, context, alive):
        end, refuse = state.end, state.refuse
        if state.at > end:
            return None
        begin = self._begin(alive, state.text, state.at)
        if begin is None:
            return None
        caps = None
        if begin == refuse:
            caps = self._pike(self._main, context, begin, False, refuse, alive)
            if caps is None:
                begin = self._begin(alive, state.text, begin + 1)
                if begin is None:
                    return None
        if caps is None:
            caps = self._pike(self._main, context, begin, False, None, alive)
            if caps is None:
                raise AssertionError("a start the backward pass found holds no match")
        found = self._found(caps, state.string, state.text, state.start, end, context)
        state.at = found.end()
        state.refuse = state.at if found.end() == found.start() else None
        return found

    def _begin(self, alive, text, at):
        """The first start from ``at`` on where a match starts and re's search tries one (re tries a start only in
        its search set, when it compiles one), or None."""
        begin = alive.next_start(at)
        tried = self._tried
        while tried is not None and begin is not None and begin < alive.end and not _within(tried, text[begin]):
            begin = alive.next_start(begin + 1)
        return begin

    def _anchored(self, string, pos, endpos, final_only):
        text, start, end = self._bounds(string, pos, endpos)
        if start > end:
            if final_only:
                return None
            with self._lock:
                return self._past(string, text, start, end)
        if self._absent(text, start, end):
            return None
        with self._lock:
            try:
                return self._anchored_read(string, text, start, end, final_only)
            except _CoreFailed as failed:
                self._native_off(failed)
                return self._anchored_read(string, text, start, end, final_only)

    def _anchored_read(self, string, text, start, end, final_only):
        context, alive, spent = self._pass(string, text, end, final_only)
        self.work += spent
        before = context.work
        try:
            if not alive.starts[start]:
                return None
            caps = self._pike(self._main, context, start, final_only, None, alive)
            if caps is None:
                raise AssertionError("a start the backward pass found holds no match")
            return self._found(caps, string, text, start, end, context)
        finally:
            self.work += context.work - before

    def _past(self, string, text, start, end):
        """match with pos past endpos: the engine still tries the pattern there, where every read fails."""
        past = []
        for look in self._shared.looks:
            past.append(_past(look, text, start, end, past))
        caps = self._past_first(self._main, text, start, end, past)
        return None if caps is None else self._found(caps, string, text, start, end, past)

    def _past_first(self, program, string, at, end, past):
        """The caps of the first thread that ends at a position past the end, where every read fails, or None;
        ``past``: whether each lookaround holds there."""
        shared = self._shared
        bits = 0
        for test_id in range(len(shared.tests)):
            if _past_test(shared, test_id, string, at, end, past):
                bits |= 1 << test_id
        threads = []
        self._close(program, 0, BIG, (None,) * shared.ncaps + (None,), at, end, bits, set(), threads)
        for pc, caps in threads:
            if program.code[pc][0] == MATCH:
                return caps
        return None

    def _close(self, program, pc, j, caps, at, end, bits, seen, out):
        """Append to ``out``, in order of priority, the threads (a step on a character, or an end) reached from
        (pc, j) at ``at`` by steps that read nothing; ``seen`` holds the keys already taken there."""
        code, depth = program.code, program.depth
        stack = [(pc, j, caps)]
        while stack:
            pc, j, caps = stack.pop()
            if pc == FAIL:
                continue
            ins = code[pc]
            kind = ins[0]
            key = (pc, 0 if kind == CHAR or kind == MATCH else min(j, depth[pc]))
            if key in seen:
                continue
            seen.add(key)
            if kind == JMP:
                stack.append((ins[1], j, caps))
            elif kind == SPLIT:
                stack.append((ins[2], j, caps))
                stack.append((ins[1], j, caps))
            elif kind == SAVE:
                changed = list(caps)
                changed[ins[1]] = at
                if ins[1] & 1 and ins[1] > 1:
                    changed[-1] = ins[1] // 2
                stack.append((pc + 1, j, tuple(changed)))
            elif kind == ASSERT:
                if bits >> ins[1] & 1:
                    stack.append((pc + 1, j, caps))
            elif kind == ORACLE:
                if bits >> ins[1] & 1:
                    if ins[2] is not None:
                        changed = list(caps)
                        changed[ins[2]] = at
                        changed[-1] = ("look", ins[3], caps[-1])
                        caps = tuple(changed)
                    stack.append((pc + 1, j, caps))
            elif kind == MARK:
                stack.append((pc + 1, min(j, ins[1]), caps))
            elif kind == GUARD:
                if at <= end:
                    stack.append((pc + 1, j, caps))
            elif kind == CHECK:
                stack.append((ins[2] if ins[1] < j else ins[3], j, caps))
            else:
                out.append((pc, caps))

    def _pike(self, program, context, start, final_only, refuse, alive, stop_at=None):
        """The caps of the leftmost-first match from ``start``, or None; a thread that reads into a place from
        which ``alive`` says no match can be completed is dropped (no pruning when alive is None)."""
        if type(context) is _NativeContext:
            # the native core reads the main program only: a pattern whose lookarounds hold groups is declined
            assert program is self._main and stop_at is None, "the native core reads the main program only"
            caps = _core(context.read.pike, start, final_only, refuse)
            return None if caps is None else tuple(caps)
        code, end = program.code, context.end
        finish = end if stop_at is None else stop_at
        caps0 = (None,) * self._shared.ncaps + (None,)
        at = start
        here = context.cls(at)
        bits = context.bits(program, at, here) if program.anchored or program.oracle_tests else 0
        threads = []
        self._close(program, 0, BIG, caps0, at, end, bits, set(), threads)
        matched, work = None, 0
        while threads:
            work += len(threads)
            following, seen = [], set()
            live = None
            if at < finish:
                members = context.members(here)
                nxt = at + 1
                there = context.cls(nxt)
                nbits = context.bits(program, nxt, there) if program.anchored or program.oracle_tests else 0
                live = alive.at(nxt) if alive is not None else None
            for pc, caps in threads:
                ins = code[pc]
                if ins[0] == MATCH:
                    if final_only and at != end:
                        continue
                    if stop_at is not None and at != stop_at:
                        continue
                    if refuse is not None and at == refuse:
                        continue
                    matched = caps
                    break
                if at < finish and members >> ins[1] & 1 and (live is None or _holds(live, pc + 1)):
                    self._close(program, pc + 1, BIG, caps, nxt, end, nbits, seen, following)
            work += len(seen)
            if at >= finish:
                break
            at, here, bits, threads = nxt, there, nbits, following
        context.work += work
        return matched

    def _found(self, caps, string, text, start, end, context):
        caps = list(caps)
        last = self._resolve(self._main, caps, context, text, end)
        return Match(self, string, text, start, end, tuple(caps[: 2 * (self.groups + 1)]), last)

    def _resolve(self, program, caps, context, string, end):
        """The groups of the positive lookarounds that hold some, read again where the winning thread passed each
        (once at most); and the last group closed, which may be one of theirs."""
        lasts = {}
        for look in program.looks_with_groups:
            at = caps[look.slot]
            if at is None:
                continue
            sub = list(self._look_caps(look, at, context, string, end))
            lasts[look.index] = self._resolve(look.exact, sub, context, string, end)
            for group in look.numbers:
                caps[2 * group], caps[2 * group + 1] = sub[2 * group], sub[2 * group + 1]
        marker = caps[-1]
        while isinstance(marker, tuple):
            _, index, before = marker
            if lasts.get(index) is not None:
                return lasts[index]
            marker = before
        return marker

    def _look_caps(self, look, at, context, string, end):
        """The caps of a positive lookaround's own read where it held; ``context`` is the list of what each
        lookaround answers past the end when the match was tried there."""
        if isinstance(context, list):
            caps = self._past_first(look.exact, string, at, end, context)
        elif look.behind:
            caps = self._pike(look.exact, context, at - look.width, False, None, None, stop_at=at)
        else:
            caps = self._pike(look.exact, context, at, False, None, None)
        if caps is None:
            raise AssertionError("a lookaround that held does not hold again")
        return caps


# -- the prefilter -------------------------------------------------------------------------------------------------


def _required(parsed, flags):
    """A set of non-empty strings, one of which every match holds, or None."""
    return _info(list(parsed), flags)[1]


def _best(sets):
    good = [found for found in sets if found and "" not in found]
    if not good:
        return None
    return max(good, key=lambda found: (min(map(len, found)), -len(found)))


def _join(left, right):
    """Every string of left followed by one of right, or None past the bounds (never a part of them)."""
    if right is None or len(left) * len(right) > MOST_STRINGS:
        return None
    joined = {a + b for a in left for b in right}
    return joined if all(len(one) <= MOST_LENGTH for one in joined) else None


def _info(seq, flags):
    """(exact, factors): every string the sequence can match (or None), and a set one of which each match holds."""
    run, exact, factors = {""}, {""}, []
    for op, av in seq:
        mine, factor = _node(op, av, flags)
        if factor:
            factors.append(factor)
        joined = _join(run, mine)
        if joined is not None:
            run = joined
            if exact is not None:
                exact = run
            continue
        factors.append(run)
        exact = None
        run = mine if mine is not None else {""}
    factors.append(run)
    return exact, _best(factors)


def _node(op, av, flags):
    if op is _C.LITERAL:
        ch = chr(av)
        if flags & re.IGNORECASE and not (ch.isascii() and not ch.isalpha()):
            return None, None
        return {ch}, {ch}
    if op is _C.SUBPATTERN:
        return _info(list(av[-1]), _check._inside(flags, av))
    if op is _C.BRANCH:
        parts = [_info(list(branch), flags) for branch in av[1]]
        exact = None
        if all(part[0] is not None for part in parts):
            exact = set().union(*(part[0] for part in parts))
            if len(exact) > MOST_STRINGS:
                exact = None
        factors = None
        if all(part[1] for part in parts):
            factors = set().union(*(part[1] for part in parts))
            if len(factors) > MOST_STRINGS:
                factors = None
        return exact, factors
    if op in (_C.MAX_REPEAT, _C.MIN_REPEAT) or (_POSSESSIVE is not None and op is _POSSESSIVE):
        least, most, body = av
        inner = _info(list(body), flags)
        exact = None
        if least == most and inner[0] is not None and least <= MOST_LENGTH:
            exact = {""}
            for _ in range(least):
                exact = _join(exact, inner[0])
                if exact is None:
                    break
        return exact, (inner[1] if least >= 1 else None)
    if _ATOMIC is not None and op is _ATOMIC:
        return _info(list(av), flags)
    if op in (_C.AT, _C.ASSERT, _C.ASSERT_NOT):
        return {""}, None
    return None, None


def compile(pattern, flags=0):  # noqa: A001 - the name re gives the call
    """A ``Pattern`` read without backtracking; ``Unreadable`` (a ``re.error``) for a pattern this engine does not
    read, ``re.error`` for an invalid one."""
    return Pattern(pattern, flags)
