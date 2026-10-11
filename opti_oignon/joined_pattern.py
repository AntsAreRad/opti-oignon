"""Two patterns joined by ``.*``, searched in time linear in the text.

``re.compile(head + ".*" + tail)`` starts over at every place ``head``
matches and runs ``.*`` to the end of the line and back each time: a line
that repeats ``head`` with no ``tail`` after it takes quadratic time.
``JoinedPattern`` reads each line once instead. For a place where ``head``
matches, it asks whether ``tail`` can start between the end of that head and
the end of its line; a line is tried from its end back, once, and what was
found is kept. The joined pattern is run only from a place it is known to
match from, so each match is the very one ``re`` returns.

Exact when the first way ``head`` matches from a place is the one the joined
pattern can succeed with, if it succeeds there at all (as when ``head``
matches one way only, or its other ways end inside a run of blanks that
``tail`` cannot start in), and when ``head`` never matches the empty string.
What can be checked is refused when the pair is made: flags that differ,
DOTALL or VERBOSE; a head or a tail the textual join would read otherwise
(an alternative at its top level, a group of global flags anywhere but at the
head's start, a verbose group, a numbered back-reference or condition in the
tail once the head's groups come before it).

``sandbox_manager`` holds a copy of the class (it is loaded alone in contract
windows); a contract holds the two equal.
"""

import re

# Module conventions (project-wide).
checkpoint_before_apply = True


class JoinedPattern:
    """``head``, then anything on the same line, then ``tail``: what a compiled ``head.*tail`` answers to
    ``search``, ``finditer`` and ``findall``, in time linear in the text."""

    def __init__(self, head, tail):
        if head.flags != tail.flags or head.flags & (re.DOTALL | re.VERBOSE):
            raise ValueError("head and tail take the same flags, neither DOTALL nor VERBOSE")
        if not (self._joinable(head.pattern, True, False) and self._joinable(tail.pattern, False, head.groups > 0)):
            raise ValueError("a head or a tail the join would read otherwise")
        self.flags = head.flags
        self.pattern = head.pattern + ".*" + tail.pattern
        self._head, self._tail = head, tail
        self._whole = re.compile(self.pattern, self.flags)

    def finditer(self, text):
        pos, reach = 0, {}
        while True:
            head = self._head.search(text, pos)
            if head is None:
                return
            start, end = head.span()
            if start == end:
                raise ValueError("head matched the empty string")
            stop = text.find("\n", end)
            match = None
            if self._reaches(text, end, len(text) if stop < 0 else stop, reach):
                match = self._whole.match(text, start)
            if match is None:
                pos = start + 1
                continue
            yield match
            pos = match.end()

    def _reaches(self, text, end, stop, reach):
        """True when ``tail`` matches from a place between ``end`` and ``stop``, the end of its line. A line is
        tried from its end back, once: ``reach[stop]`` is the lowest place tried and the highest place ``tail``
        matched from, or -1."""
        low, high = reach.get(stop, (stop + 1, -1))
        while high < 0 and low > end:
            low -= 1
            if self._tail.match(text, low):
                high = low
        reach[stop] = (low, high)
        return high >= end

    @staticmethod
    def _joinable(pattern, first, numbered):
        """True when ``pattern`` reads the same as a part of the join: no alternative at its top level, no group of
        global flags unless ``first`` and at its start, and, when ``numbered``, no numbered back-reference or
        condition (the head's groups would come before the ones it names)."""
        depth, at, inside = 0, 0, False
        while at < len(pattern):
            char = pattern[at]
            if char == "\\":
                if numbered and not inside and pattern[at + 1:at + 2] in tuple("123456789"):
                    return False
                at += 2
                continue
            if inside:
                inside = char != "]"
            elif char == "[":
                inside = True
                at += 2 if pattern[at + 1:at + 2] == "]" else 3 if pattern[at + 1:at + 3] == "^]" else 1
                continue
            elif char == "(":
                if pattern.startswith("(?#", at):
                    close = pattern.find(")", at)
                    at = len(pattern) if close < 0 else close + 1
                    continue
                if re.match(r"\(\?[aiLmsu]*x[aiLmsux]*(?:-[imsx]*)?:", pattern[at:]):
                    return False  # a verbose group: its comments run to a line break the scan does not read
                group = re.match(r"\(\?(?:([aiLmsux]+)\)|\((\d))", pattern[at:])
                if group and group.group(1) and not (first and at == 0):
                    return False
                if group and group.group(2) and numbered:
                    return False
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "|" and depth == 0:
                return False
            at += 1
        return True

    def search(self, text):
        return next(self.finditer(text), None)

    def findall(self, text):
        groups = self._whole.groups
        if groups == 0:
            return [match.group() for match in self.finditer(text)]
        if groups == 1:
            return [match.groups("")[0] for match in self.finditer(text)]
        return [match.groups("") for match in self.finditer(text)]
