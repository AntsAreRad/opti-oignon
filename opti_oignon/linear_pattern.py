"""Linear patterns: whether a regular expression that comes from outside the code reads any text in linear time,
decided from its structure before it runs.

Python's engine backtracks. A pattern whose repeats can split one run of
characters in more than one way, or whose leading repeat is tried again from
every position of a run, takes quadratic, cubic or exponential time on a long
run of such characters. A pattern written in the package is held to linear
time by a guard that measures it; a pattern given by a model, a configuration
file or a person is not in the package, and is judged here when it is
compiled.

THE CHECK. The pattern is parsed and built into a Thompson automaton, whose
paths are the paths the engine walks. A loop on every character in front of
it stands for the start positions a search tries: one start behind ``\\A``, or
behind ``^`` outside multiline mode; the start of each line behind ``^`` in
multiline mode; every position otherwise. A loop's way back is cut where its
body was entered by empty steps alone: the engine takes such an empty round
once, then loops no more and goes on past the loop, a way of its own; every
empty path that leads to one same character step counts as a step of its own. A state from which every way on ends the match by empty
steps, with no assertion on the way, is terminal: the engine ends there in a
match at its first dead end and never walks back, though at each position on
its way it first tries every empty path from where it stands, counted as from
any state. The automaton reads a text
in time linear in its length exactly when it is neither exponentially
ambiguous (two distinct cycles through one state on one word: some pair of
states reaches itself both ways) nor infinitely ambiguous (states p and q
with p to p, p to q and q to q on one word): the first is found among the
strongly connected components of the pairs of states, the second by a path
from (p, p, q) to (p, q, q) among the triples. The ways one start tries --
the paths that read one same word, counted along every word -- are at most
WAYS: a pattern linear in the text may still try two ways a round of a counted
repeat, a number no text needs to be long to reach. Every match in turn costs no
more: each start position is tried by one search only, so ``findall``,
``finditer``, ``sub`` and ``split`` read in linear time what ``search`` does.

The sets of characters each item accepts are read from the engine itself over
every code point, under the item's own flags: case folding and Unicode classes
are exact, and two sets share a character exactly when the partition of every
code point they induce has a class inside both.

A lookaround whose body has no unbounded repeat, no back-reference, no more
than LOOKAROUND_WAYS ways through it and none of more than STATES steps (a
character, an anchor, a round of a repeat, a nested lookaround's own) costs a
bounded time wherever it is tried, and only removes paths: it is read
as an assertion. A repeat counted past STATES is refused, as its automaton
would be.

SOUND, NOT COMPLETE. What the check does not model -- a lookaround beyond
that bound, a back-reference, a conditional, an automaton of more than STATES
states, an analysis of more than STEPS steps, sets of characters written in
more than SOURCES characters, a pattern too deep to parse --
is refused, never guessed. Atomic groups and possessive repeats are read as
plain ones, and the order in which the engine tries branches and repeats is
not read: a pattern linear only by that order (a lazy repeat, a branch that
takes a run at once after one that fails) is refused too.

JOINS. ``joins_exactly(head, tail)`` tells whether a reader that takes the
first way ``head`` matches from a place, then looks for ``tail`` from its end
to the end of its line, finds whatever ``head.*tail`` finds: when ``head``
ends one way only from any place (no word it matches is a prefix of another),
or ends with a greedy repeat of one class of characters after a part that
ends one way only, and ``tail`` cannot start with a character of that class.

The sets of characters are read in one pass over every code point for each
set that holds a class shorthand, a negation or any character, and in one pass
for all the others of one pattern and flags together (then each from the
characters that pass holds); the text of every code point is built at the
first read and kept, about 4.5 MB.
"""
import functools
import re
import sys

try:
    import re._constants as _constants
    import re._parser as _parser
except ImportError:  # Python 3.10
    import sre_constants as _constants
    import sre_parse as _parser

# Module conventions (project-wide).
checkpoint_before_apply = True

# The largest automaton the check builds, the most steps its analysis takes, and the most sets of characters it
# reads from the engine for one pattern (each a pass over every code point): past any, the pattern is refused, so a
# pattern built to be costly to check costs a bounded time.
STATES = 600
STEPS = 400_000
READS = 32
# The characters the class patterns one check reads may be written in: a pass over every code point costs about
# their number for each, so a pattern built of huge classes costs a bounded time too.
SOURCES = 1024
# The ways through a lookaround's body it may take and still be read as an assertion.
LOOKAROUND_WAYS = 256
# The ways one start may try on one word, and the steps one try of a lookaround may take, at most: past either, a
# pattern linear in the text is refused for what it costs at every start.
WAYS = 4096
# The steps of lookaround tries one start may take, each try counted on every way that may reach it.
TRIES = 16 * WAYS
# The sets of characters kept between checks, at most; the characters one shared pass may hold and still be split
# item by item.
KEPT = 1024
FEW = 4096

REPEATS = tuple(op for op in (_constants.MAX_REPEAT, _constants.MIN_REPEAT,
                              getattr(_constants, "POSSESSIVE_REPEAT", None)) if op is not None)
ATOMIC = getattr(_constants, "ATOMIC_GROUP", None)
SINGLE = (_constants.LITERAL, _constants.NOT_LITERAL, _constants.ANY, _constants.IN)
CATEGORIES = {
    _constants.CATEGORY_DIGIT: r"\d", _constants.CATEGORY_NOT_DIGIT: r"\D",
    _constants.CATEGORY_SPACE: r"\s", _constants.CATEGORY_NOT_SPACE: r"\S",
    _constants.CATEGORY_WORD: r"\w", _constants.CATEGORY_NOT_WORD: r"\W",
}
# The flags that change which characters one item accepts: multiline and verbose change none.
CHARACTER_FLAGS = re.IGNORECASE | re.ASCII | re.LOCALE | re.UNICODE | re.DOTALL

_RANGES = {}


class CostlyPattern(re.error):
    """A pattern refused because it may read a long text in more than linear time, or cannot be checked."""


class _Unchecked(Exception):
    """The pattern holds what the check does not model, or is too large to check."""


def _inside(flags, av):
    """The flags inside a group: the pattern's, with the group's own added and removed; a type flag the group sets
    (ASCII, LOCALE, UNICODE) takes the place of the pattern's, as the engine combines them."""
    if av[1] & _parser.TYPE_FLAGS:
        flags &= ~_parser.TYPE_FLAGS
    return (flags | av[1]) & ~av[2]


def _single(op, av):
    """A pattern for one single-character item of a parsed tree."""
    if op is _constants.LITERAL:
        return re.escape(chr(av))
    if op is _constants.NOT_LITERAL:
        return "[^" + re.escape(chr(av)) + "]"
    if op is _constants.ANY:
        return "."
    parts, negate = [], ""
    for sub_op, sub_av in av:
        if sub_op is _constants.NEGATE:
            negate = "^"
        elif sub_op is _constants.LITERAL:
            parts.append(re.escape(chr(sub_av)))
        elif sub_op is _constants.RANGE:
            parts.append(re.escape(chr(sub_av[0])) + "-" + re.escape(chr(sub_av[1])))
        elif sub_op is _constants.CATEGORY and sub_av in CATEGORIES:
            parts.append(CATEGORIES[sub_av])
        else:
            raise _Unchecked(f"a class member the check does not model ({sub_op})")
    if not parts:
        return r"[\s\S]" if negate else r"[^\s\S]"
    return "[" + negate + "".join(parts) + "]"


_EVERY = []


def _key(op, av, flags):
    """(single-character pattern, the flags that change what it accepts) for one item."""
    single = _single(op, av)
    return single, flags & (CHARACTER_FLAGS if single == "." else CHARACTER_FLAGS & ~re.DOTALL)


def _wide(op, av):
    """True for an item that may accept many characters: a class shorthand, a negation, any character."""
    if op is _constants.IN:
        return any(sub_op in (_constants.NEGATE, _constants.CATEGORY) for sub_op, _ in av)
    return op is not _constants.LITERAL


def _charge(machine, passes=1):
    """Charge the check for passes over every code point; past READS, the pattern is refused."""
    machine.reads -= passes
    if machine.reads < 0:
        raise _Unchecked(f"more than {READS} passes over every code point to read its sets of characters")


def _scan(single, flags, machine=None):
    """The code points ``single`` accepts under ``flags``, read from the engine in one pass over every code point;
    charged to ``machine``'s check when given (a prefetch charges what a check pays, read or kept)."""
    if machine is not None:
        _charge(machine)
    if not _EVERY:
        _EVERY.append("".join(map(chr, range(sys.maxunicode + 1))))
    compiled = re.compile("(?:" + single + ")+", flags)
    return tuple((found.start(), found.end() - 1) for found in compiled.finditer(_EVERY[0]))


def _keep(key, ranges):
    if len(_RANGES) >= KEPT:
        _RANGES.clear()
    _RANGES[key] = ranges


def _hold(machine, key, ranges):
    """Hold a set read for the check, and keep it between checks."""
    machine.ranges[key] = ranges
    _keep(key, ranges)


def _known(machine, key):
    """The set the check holds, or an earlier check kept, or None."""
    ranges = machine.ranges.get(key)
    return _RANGES.get(key) if ranges is None else ranges


def _size(sets):
    """The code points the union of ``sets`` (each sorted disjoint ranges) holds."""
    total, reach = 0, -1
    for first, last in sorted(span for ranges in sets for span in ranges):
        if last > reach:
            total += last - max(first, reach + 1) + 1
            reach = last
    return total


def _prefetch(seq, flags, machine):
    """Read the sets of the parsed ``seq``'s items: each wide one alone, the others of one flags in one pass
    together, then each from the characters that pass holds when they are FEW or less, alone otherwise. A set an
    earlier check kept is not read again, but costs the check what reading it would: no verdict hangs on what
    earlier checks kept."""
    wide, narrow = {}, {}

    def walk(seq, flags):
        for op, av in seq:
            if op in SINGLE:
                if op is _constants.LITERAL and not flags & re.IGNORECASE:
                    continue
                key = _key(op, av, flags)
                if _wide(op, av):
                    wide[key] = None
                else:
                    narrow.setdefault(key[1], {})[key[0]] = None
            elif op is _constants.SUBPATTERN:
                walk(av[-1], _inside(flags, av))
            elif op in REPEATS:
                walk(av[2], flags)
            elif op is ATOMIC:
                walk(av, flags)
            elif op is _constants.BRANCH:
                for branch in av[1]:
                    walk(branch, flags)

    walk(seq, flags)
    sources = sum(len(key[0]) for key in wide) + sum(len(single) for singles in narrow.values() for single in singles)
    if sources > SOURCES:
        raise _Unchecked(f"sets of characters written in more than {SOURCES} characters")
    for key in wide:
        _charge(machine)
        ranges = _known(machine, key)
        _hold(machine, key, _scan(key[0], key[1]) if ranges is None else ranges)
    for item_flags, singles in narrow.items():
        singles = list(singles)
        found = {single: _known(machine, (single, item_flags)) for single in singles}
        missing = [single for single in singles if found[single] is None]
        _charge(machine)
        shared = _scan("|".join(missing), item_flags) if missing else ()
        size = _size([shared] + [ranges for ranges in found.values() if ranges is not None])
        if len(missing) == 1:
            found[missing[0]] = shared
        if len(singles) > 1 and size > FEW:
            _charge(machine, len(singles))
            for single in missing if len(missing) > 1 else ():
                found[single] = _scan(single, item_flags)
        elif len(missing) > 1:
            machine.spend(len(singles) * size)
            points = [point for first, last in shared for point in range(first, last + 1)]
            for single in missing:
                item = re.compile(single, item_flags)
                found[single] = _ranges([point for point in points if item.fullmatch(chr(point))])
        elif len(singles) > 1:
            machine.spend(len(singles) * size)
        for single in singles:
            _hold(machine, (single, item_flags), found[single])


def _ranges(points):
    """Sorted code points as sorted disjoint (first, last) ranges."""
    out = []
    for point in points:
        if out and out[-1][1] == point - 1:
            out[-1] = (out[-1][0], point)
        else:
            out.append((point, point))
    return tuple(out)


def _members(op, av, flags, machine):
    """The code points one item accepts under ``flags``, as sorted disjoint (first, last) ranges: a literal read
    with case is itself; any other item is read from the engine over every code point, once for the check
    whatever another check empties, and kept between checks."""
    if op is _constants.LITERAL and not flags & re.IGNORECASE:
        return ((av, av),)
    key = _key(op, av, flags)
    ranges = machine.ranges.get(key)
    ranges = _RANGES.get(key) if ranges is None else ranges
    if ranges is None:
        ranges = _scan(key[0], key[1], machine)
        _keep(key, ranges)
    machine.ranges[key] = ranges
    return ranges


class _Automaton:
    """A Thompson automaton: empty steps, character steps by set, the body each loop enters, the sets of characters
    the check has read, and what it may still spend."""

    def __init__(self):
        self.empty, self.steps, self.sets, self.bodies, self.exits = [], [], [], {}, {}
        self._ids, self.ranges, self.bounded, self.looked = {}, {}, {}, 0
        self.budget, self.reads = STEPS, READS

    def state(self):
        if len(self.empty) >= STATES:
            raise _Unchecked(f"an automaton of more than {STATES} states")
        self.empty.append([])
        self.steps.append([])
        return len(self.empty) - 1

    def set_id(self, ranges):
        if ranges not in self._ids:
            self._ids[ranges] = len(self.sets)
            self.sets.append(ranges)
        return self._ids[ranges]

    def spend(self, count=1):
        self.budget -= count
        if self.budget < 0:
            raise _Unchecked(f"an analysis of more than {STEPS} steps")


def _build(seq, flags, machine, start):
    """The automaton of the parsed ``seq`` from state ``start``, under ``flags``; its end state."""
    current = start
    for op, av in seq:
        if op in SINGLE:
            after = machine.state()
            machine.steps[current].append((machine.set_id(_members(op, av, flags, machine)), after))
            current = after
        elif op is _constants.SUBPATTERN:
            current = _build(av[-1], _inside(flags, av), machine, current)
        elif op is ATOMIC:
            current = _build(av, flags, machine, current)
        elif op is _constants.BRANCH:
            end = machine.state()
            for branch in av[1]:
                entry = machine.state()
                machine.empty[current].append(entry)
                machine.empty[_build(branch, flags, machine, entry)].append(end)
            current = end
        elif op in REPEATS:
            least, most, body = av
            if least > STATES or most != _constants.MAXREPEAT and most - least > STATES:
                raise _Unchecked(f"a repeat counted past {STATES}")
            for _ in range(least):
                machine.spend()
                current = _build(body, flags, machine, current)
            if most == _constants.MAXREPEAT:
                loop, end, entry = machine.state(), machine.state(), machine.state()
                machine.empty[current].append(loop)
                machine.empty[loop].append(entry)
                machine.bodies[loop], machine.exits[loop] = entry, end
                machine.empty[_build(body, flags, machine, entry)].append(("back", loop))
                machine.empty[loop].append(end)
                current = end
            else:
                end = machine.state()
                for _ in range(most - least):
                    machine.spend()
                    machine.empty[current].append(end)
                    current = _build(body, flags, machine, current)
                machine.empty[current].append(end)
                current = end
        elif op is _constants.AT or (op in (_constants.ASSERT, _constants.ASSERT_NOT)
                                     and _bounded(av[1], machine)):
            if op is not _constants.AT:
                machine.looked += machine.bounded[id(av[1])][1]
                if machine.looked > WAYS:
                    raise _Unchecked(f"lookarounds that take more than {WAYS} steps in all from one start")
            after = machine.state()
            machine.empty[current].append(("assert", after))
            current = after
        elif op in (_constants.ASSERT, _constants.ASSERT_NOT):
            raise _Unchecked(f"a lookaround of more than {LOOKAROUND_WAYS} ways or {STATES} steps")
        else:
            raise _Unchecked(f"a construct the check does not model ({op})")
    return current


def _ways(seq):
    """The ways through the parsed ``seq``, a body ``_rounds`` has read, counted to LOOKAROUND_WAYS and one past, a
    lookaround in it tried on each of them (its ways multiply theirs)."""
    ways = 1
    for op, av in seq:
        if op in SINGLE or op is _constants.AT:
            continue
        if op in (_constants.ASSERT, _constants.ASSERT_NOT):
            inner = _ways(av[1])
        elif op is _constants.SUBPATTERN:
            inner = _ways(av[-1])
        elif op is ATOMIC:
            inner = _ways(av)
        elif op is _constants.BRANCH:
            inner = sum(_ways(branch) for branch in av[1])
        else:
            inner = _counted(_ways(av[2]), av[0], av[1])
        ways = min(ways * inner, LOOKAROUND_WAYS + 1)
    return ways


def _rounds(seq):
    """The steps one way through the parsed ``seq`` may take, counted to STATES and one past: a character, an
    anchor, a lookaround and the steps it takes, each round of a repeat (an empty round counted as one)."""
    cap, total = STATES + 1, 0
    for op, av in seq:
        if op in SINGLE or op is _constants.AT:
            inner = 1
        elif op in (_constants.ASSERT, _constants.ASSERT_NOT):
            inner = 1 + _rounds(av[1])
        elif op is _constants.SUBPATTERN:
            inner = _rounds(av[-1])
        elif op is ATOMIC:
            inner = _rounds(av)
        elif op is _constants.BRANCH:
            inner = max(_rounds(branch) for branch in av[1])
        elif op in REPEATS and av[1] != _constants.MAXREPEAT:
            inner = min(av[1], cap) * max(1, _rounds(av[2]))
        else:
            raise _Unchecked("a lookaround with an unbounded repeat, a back-reference or a conditional")
        total = min(total + inner, cap)
    return total


def _bounded(body, machine):
    """True when a lookaround's parsed body costs a bounded time wherever it is tried: no more than
    LOOKAROUND_WAYS ways through it, those of the lookarounds in it multiplied in, each of no more than STATES
    steps (a character, an anchor, a round of a repeat, a nested lookaround's own); judged once for the check,
    however often the body is built. ``_build`` adds the steps of one try (ways times steps) of every lookaround it
    builds: a start that walks them all, or one try of any, takes no more than WAYS steps."""
    if id(body) not in machine.bounded:
        machine.spend()
        rounds = _rounds(body)
        ways = _ways(body) if rounds <= STATES else LOOKAROUND_WAYS + 1
        machine.bounded[id(body)] = (ways <= LOOKAROUND_WAYS, rounds * ways)
    return machine.bounded[id(body)][0]


def _counted(ways, least, most):
    """The ways through a body of ``ways`` ways repeated ``least`` to ``most`` times, counted to LOOKAROUND_WAYS
    and one past, in a time bounded by that count, never by the repeat's."""
    cap = LOOKAROUND_WAYS + 1
    if ways == 1:
        return min(most - least + 1, cap)
    total, power = 0, 1
    for count in range(most + 1):
        if count >= least:
            total += power
        if total >= cap or power >= cap:
            return cap
        power *= ways
    return total


def _closure(machine, start, final, through_assertions=False):
    """The character steps between the states a character enters, each counted once per empty path that leads to it
    (to WAYS and one past), an empty round of a loop going on past it; and the states from which an empty path with
    no assertion (or any, ``through_assertions``) ends the match. The empty paths from each state to the end, through
    any assertion, are counted too (``machine.finals``): the engine tries them all where the end is refused."""
    kept = {start} | {target for row in machine.steps for _, target in row}
    memo = {}

    def reach(state, entered):
        key = (state, entered)
        if key in memo:
            return memo[key]
        machine.spend()
        found, ends, finals = {}, state == final, int(state == final)
        for set_id, target in machine.steps[state]:
            found[(set_id, target)] = min(WAYS + 1, found.get((set_id, target), 0) + 1)
        for target in machine.empty[state]:
            inner, sure = entered, True
            if isinstance(target, tuple) and target[0] == "assert":
                target, sure = target[1], False
            elif isinstance(target, tuple) and target[1] in entered:
                # An empty round: the engine loops no more and goes on past the loop, a way of its own.
                target, inner = machine.exits[target[1]], entered - {target[1]}
            elif isinstance(target, tuple):
                target = target[1]
            elif machine.bodies.get(state) == target:
                inner = entered | {state}
            below, below_ends, below_finals = reach(target, inner)
            ends = ends or ((sure or through_assertions) and below_ends)
            finals = min(WAYS + 1, finals + below_finals)
            for step, count in below.items():
                found[step] = min(WAYS + 1, found.get(step, 0) + count)
        memo[key] = (found, ends, finals)
        return memo[key]

    edges, ending = {}, set()
    machine.finals = {}
    for state in kept:
        found, ends, machine.finals[state] = reach(state, frozenset())
        if ends:
            ending.add(state)
        for (set_id, target), count in found.items():
            edges[(state, set_id, target)] = count
    return edges, ending


def _classes(sets, machine):
    """The distinct membership masks of the partition of every code point the sets induce: a sweep over the points
    where a set begins or ends."""
    toggles = {}
    for index, ranges in enumerate(sets):
        machine.spend(len(ranges))
        for first, last in ranges:
            toggles[first] = toggles.get(first, 0) ^ (1 << index)
            toggles[last + 1] = toggles.get(last + 1, 0) ^ (1 << index)
    masks, mask, previous = set(), 0, 0
    for point in sorted(toggles):
        if point > previous:
            masks.add(mask)
        mask ^= toggles[point]
        previous = point
    if previous <= sys.maxunicode:
        masks.add(mask)
    return masks


def _components(roots, successors, machine):
    """Tarjan's strongly connected components of what ``roots`` reach: {node: component}, [members...]."""
    index, low, on_stack, stack, component, groups = {}, {}, set(), [], {}, []
    counter = 0
    for root in roots:
        if root in index:
            continue
        work = [(root, iter(successors(root)))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, pending = work[-1]
            advanced = False
            for nxt in pending:
                machine.spend()
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(successors(nxt))))
                    advanced = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                members = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component[member] = len(groups)
                    members.append(member)
                    if member == node:
                        break
                groups.append(members)
    return component, groups


def _ambiguity(pattern, flags):
    """'linear', 'polynomial' or 'exponential' for every search of the pattern on any text."""
    parsed = _parser.parse(pattern, flags)
    flags = parsed.state.flags
    machine = _Automaton()
    anything = machine.set_id(((0, sys.maxunicode),))
    newline = machine.set_id(((10, 10),))
    other = machine.set_id(((0, 9), (11, sys.maxunicode)))
    items = list(parsed)
    lead = items[0] if items and items[0][0] is _constants.AT else (None, None)
    init, start = machine.state(), machine.state()
    looping = {init}
    machine.empty[init].append(start)
    if lead[1] is _constants.AT_BEGINNING_STRING or (lead[1] is _constants.AT_BEGINNING
                                                     and not flags & re.MULTILINE):
        pass                                                   # one start position
    elif lead[1] is _constants.AT_BEGINNING:
        inside = machine.state()                               # a line start (init), or inside a line
        looping.add(inside)
        machine.steps[init].append((newline, init))
        machine.steps[init].append((other, inside))
        machine.steps[inside].append((other, inside))
        machine.steps[inside].append((newline, init))
    else:
        machine.steps[init].append((anything, init))           # a search tries every start position
    _prefetch(items, flags, machine)
    final = _build(items, flags, machine, start)
    edges, ending = _closure(machine, init, final)
    masks = _classes(machine.sets, machine)
    # Terminal states: every way on ends the match by empty steps; the engine never walks back out of them.
    terminal = set(ending)
    while True:
        leaving = {source for (source, _, target) in edges if source in terminal and target not in terminal}
        if not leaving:
            break
        terminal -= leaving
    every, edges = edges, {key: count for key, count in edges.items() if key[0] not in terminal}

    met = {}

    def meet(*ids):
        need = 0
        for set_id in ids:
            need |= 1 << set_id
        if need not in met:
            machine.spend(len(masks))
            met[need] = any(mask & need == need for mask in masks)
        return met[need]

    out = {}
    for (source, set_id, target), count in edges.items():
        out.setdefault(source, []).append((set_id, target, count))
    pairs, apart = {}, set()

    def successors(pair):
        if pair not in pairs:
            left, right = pair
            found = []
            for left_set, left_target, left_count in out.get(left, ()):
                for right_set, right_target, right_count in out.get(right, ()):
                    machine.spend()
                    if meet(left_set, right_set):
                        nxt = (left_target, right_target)
                        found.append(nxt)
                        if not (left == right and left_set == right_set and left_target == right_target
                                and left_count == 1):
                            apart.add((pair, nxt))
            pairs[pair] = found
        return pairs[pair]

    component, groups = _components([(init, init)], successors, machine)
    for members in groups:
        diagonal = [pair for pair in members if pair[0] == pair[1]]
        if not diagonal:
            continue
        inside = set(members)
        if any(pair[0] != pair[1] for pair in members) or any(
                (pair, nxt) in apart for pair in members for nxt in pairs[pair] if nxt in inside):
            return "exponential"
    cyclic = {pair for members in groups if len(members) > 1 or members[0] in pairs[members[0]] for pair in members}
    for p, q in sorted(pair for pair in cyclic if pair[0] != pair[1]):
        if (p, p) not in cyclic or (q, q) not in cyclic:
            continue
        goal, seen, todo = (p, q, q), {(p, p, q)}, [(p, p, q)]
        while todo:
            a, b, c = todo.pop()
            for a_set, a_target, _ in out.get(a, ()):
                if component.get((a_target, a_target)) != component.get((p, p)):
                    continue
                for b_set, b_target, _ in out.get(b, ()):
                    for c_set, c_target, _ in out.get(c, ()):
                        machine.spend()
                        if component.get((c_target, c_target)) != component.get((q, q)) or not meet(
                                a_set, b_set, c_set):
                            continue
                        nxt = (a_target, b_target, c_target)
                        if nxt == goal:
                            return "polynomial"
                        if nxt not in seen:
                            seen.add(nxt)
                            todo.append(nxt)
    # The ways one start tries: the paths that read one same word, counted along every word a class of characters
    # at a time. Linear in the text, a pattern may still try more ways from every start than any text is long (two
    # ways a round, forty rounds).
    moves, fan = {}, dict(machine.finals)
    for (source, set_id, target), count in edges.items():
        if not (source == init and target in looping):
            fan[source] = min(WAYS + 1, fan.get(source, 0) + count)
            for mask in masks:
                machine.spend()
                if mask >> set_id & 1:
                    moves.setdefault((source, mask), []).append((target, count))
    ordered, widest = sorted(masks), 1
    # A terminal state is left by one way only, the first that reads (it ends in a match), but at each position on
    # that way every empty path from where it stands is tried first.
    after, weight, reached, todo = {}, dict(machine.finals), {init}, [init]
    for (source, _, target), count in every.items():
        after.setdefault(source, []).append(target)
        if source in terminal and not (source == init and target in looping):
            weight[source] = min(WAYS + 1, weight.get(source, 0) + count)
    while todo:
        for target in after.get(todo.pop(), ()):
            machine.spend()
            if target not in reached:
                reached.add(target)
                todo.append(target)
    for state in sorted(reached & terminal):
        if weight.get(state, 0) > WAYS:
            return "exponential"
        widest = max(widest, weight.get(state, 0))
    seen, todo = set(), [((init, 1),)]
    while todo:
        vector = todo.pop()
        machine.spend(len(ordered))
        # Every empty path from where the ways stand is tried, to a step or to the end: an end refused (an assertion
        # after it, an empty match where every match in turn wants one that reads) tries them all.
        tried = sum(count * fan.get(state, 0) for state, count in vector)
        if tried > WAYS:
            return "exponential"
        widest = max(widest, tried)
        for mask in ordered:
            following = {}
            for state, count in vector:
                for target, times in moves.get((state, mask), ()):
                    machine.spend()
                    following[target] = min(WAYS + 1, following.get(target, 0) + count * times)
            key = tuple(sorted(following.items()))
            if key and key not in seen:
                seen.add(key)
                todo.append(key)
    if machine.looked * widest > TRIES:
        raise _Unchecked(f"lookarounds tried on up to {widest} ways from one start, more than {TRIES} steps")
    return "linear"


def _ends_once(seq, flags, machine):
    """True when no text the parsed ``seq`` matches is a proper prefix of another it matches (assertions read as
    passed): from any place, it can end one way only. Two ways read one same text side by side, on a character
    both their sets hold at every step; it is not so when one may end where the other may still read on to an end
    (one way alone is the two ways on one path)."""
    start = machine.state()
    final = _build(seq, flags, machine, start)
    edges, ending = _closure(machine, start, final, through_assertions=True)
    masks = _classes(machine.sets, machine)
    out, into = {}, {}
    for source, set_id, target in edges:
        if machine.sets[set_id]:
            out.setdefault(source, set()).add((set_id, target))
            into.setdefault(target, set()).add(source)
    # The states from which one character or more reaches an end.
    onward, todo = set(), list(ending)
    while todo:
        for source in into.get(todo.pop(), ()):
            machine.spend()
            if source not in onward:
                onward.add(source)
                todo.append(source)
    met = {}
    seen, todo = {(start, start)}, [(start, start)]
    while todo:
        left, right = todo.pop()
        if left in ending and right in onward:
            return False
        for left_set, left_target in out.get(left, ()):
            for right_set, right_target in out.get(right, ()):
                machine.spend()
                need = 1 << left_set | 1 << right_set
                if need not in met:
                    machine.spend(len(masks))
                    met[need] = any(mask & need == need for mask in masks)
                if met[need] and (left_target, right_target) not in seen:
                    seen.add((left_target, right_target))
                    todo.append((left_target, right_target))
    return True


def _joins(head, tail, flags):
    joined = _parser.parse(head + ".*" + tail, flags)
    flags = joined.state.flags
    items = list(joined)
    count = len(list(_parser.parse(head, flags)))
    gap = items[count] if count < len(items) else (None, None)
    if flags & re.DOTALL or gap[0] is not _constants.MAX_REPEAT or gap[1][:2] != (0, _constants.MAXREPEAT) or list(
            gap[1][2]) != [(_constants.ANY, None)]:
        return False
    head_items, tail_items = items[:count], items[count + 1:]
    machine = _Automaton()
    _prefetch(head_items + tail_items, flags, machine)
    last = head_items[-1] if head_items else (None, None)
    greedy = (_constants.MAX_REPEAT, getattr(_constants, "POSSESSIVE_REPEAT", None))
    run = None
    if last[0] in greedy and last[1][1] == _constants.MAXREPEAT and len(last[1][2]) == 1 and list(
            last[1][2])[0][0] in SINGLE:
        op, av = list(last[1][2])[0]
        run = machine.set_id(_members(op, av, flags, machine))
        head_items = head_items[:-1]
    if not _ends_once(head_items, flags, machine):
        return False
    if run is None:
        return True
    start = machine.state()
    final = _build(tail_items, flags, machine, start)
    edges, ending = _closure(machine, start, final, through_assertions=True)
    if start in ending:
        return False
    firsts = {set_id for source, set_id, _target in edges if source == start}
    masks = _classes(machine.sets, machine)
    return not any(mask >> run & 1 and mask >> set_id & 1 for mask in masks for set_id in firsts)


def joins_exactly(head, tail, flags=0):
    """True when a reader that takes the first way ``head`` matches from a place, then looks for ``tail`` from where
    it ends to the end of its line, finds whatever a search of ``head.*tail`` under ``flags`` finds: ``head`` ends
    one way only from any place, or ends with a greedy repeat of one class of characters after a part that ends one
    way only, and ``tail`` cannot start with a character of that class. False when the check cannot tell."""
    if not isinstance(flags, int):
        return False
    try:
        return _joins(head, tail, int(flags))
    except (_Unchecked, RecursionError, OverflowError, ValueError, TypeError, re.error):
        return False


@functools.lru_cache(maxsize=256)
def _judged(pattern, flags):
    """(verdict, why): 'linear' and '', or a refusal and its reason."""
    if not isinstance(pattern, str):
        return "unchecked", "only a text pattern is checked"
    try:
        found = _ambiguity(pattern, flags)
    except _Unchecked as exc:
        return "unchecked", f"it holds {exc}"
    except RecursionError:
        return "unchecked", "it is too deep to check"
    if found == "linear":
        return "linear", ""
    return found, ("a text can make it take exponential time" if found == "exponential"
                   else "a long text can make it take quadratic time or worse")


def verdict(pattern, flags=0):
    """'linear' when every search of the pattern (``search``, ``match``, from any position), and every match in
    turn (``findall``, ``finditer``, ``sub``, ``split``), reads any text in linear time -- not ``fullmatch``, which
    walks back out of the states where a search ends;
    'polynomial' or 'exponential' when a text can make it slower; 'unchecked' when the check cannot tell. A
    pattern that does not compile raises ``re.error``, as does one the engine refuses with another error (a
    repeat count too large, flags that cannot stand together, a nesting too deep)."""
    try:
        re.compile(pattern, flags)
    except (RecursionError, OverflowError, ValueError, TypeError) as exc:
        raise re.error(f"{exc.__class__.__name__}: {exc}", pattern) from None
    return _judged(pattern, int(flags))[0]


def compile_linear(pattern, flags=0):
    """``re.compile(pattern, flags)`` for a pattern that reads any text in linear time; ``CostlyPattern`` (a
    ``re.error``) for any other, so a caller that refuses an invalid pattern refuses it alike: one the engine finds
    too large to read (a repeat count, a nesting) as a ``CostlyPattern``; flags it cannot hold together, or a
    pattern or flags of the wrong type, as an invalid one, a ``re.error``."""
    try:
        compiled = re.compile(pattern, flags)
        found, why = _judged(pattern, int(flags))
    except (RecursionError, OverflowError) as exc:
        raise CostlyPattern(f"pattern refused: it is too large to read ({exc.__class__.__name__})",
                            pattern) from None
    except (ValueError, TypeError) as exc:
        raise re.error(f"{exc.__class__.__name__}: {exc}", pattern) from None
    if found != "linear":
        raise CostlyPattern(f"pattern refused: {why}", pattern)
    return compiled
