#!/usr/bin/env python3
"""Contracts for the chat loader's words for the engines that build their own context.

With the onion on, the reasoning engine, the consensus engine and
self-correct's correction pass say in a status line, once they have returned,
that the onion's window is not theirs. The loader shows a server status only
when its closed table maps it (``lib/chat/loaderWords.ts``).

  * LD23 -- each of those lines, as the agentic executor writes it, opens
    with the caveat's prefix and is said in the table's words -- the engine
    named, building its own context -- which pass both of the garden's nets
    and are printable ASCII; a line of another prefix, or naming another
    engine, is not said.

Local-only (the public distribution ships no tests). The table runs under Node
through the loader contracts' own driver (tests/test_chat_loader_contracts.py),
and the nets through ``allium/ethics.py`` loaded by the shared isolation window.
"""

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_chat_loader_contracts as _ld  # noqa: E402
from _isolation import REPO  # noqa: E402

BUDGET_S = {
    "test_ld23_an_engine_that_builds_its_own_context_is_said_in_the_tables_words": 6.0,
}


def _outside_lines():
    """The status lines the agentic executor says when an engine builds its own context, read off its source."""
    tree = ast.parse((REPO / "opti_oignon" / "agentic_executor.py").read_text(encoding="utf-8"))
    (found,) = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(getattr(target, "id", "") == "_OUTSIDE_WINDOW" for target in node.targets)]
    return ast.literal_eval(found)


def test_ld23_an_engine_that_builds_its_own_context_is_said_in_the_tables_words():
    lines = _outside_lines()
    assert list(lines) == ["reasoning", "consensus", "self_correct"], lines
    assert all(line.startswith("[!] ") for line in lines.values()), ("the caveat's own prefix", lines)
    near = [line.replace("[!] ", "[i] ", 1) for line in lines.values()] + ["[!] The kitchen builds its own context"]
    out = _ld._node("words", {"sample": _ld._SAMPLE, "ranAs": [], "statuses": [*lines.values(), *near],
                              "replays": {}})
    said = out["statuses"]
    assert said[:3] == ["The reasoning engine builds its own context", "The consensus engine builds its own context",
                        "The self-correction pass builds its own context"], said
    assert said[3:] == [None] * len(near), ("a line of another prefix or another engine is not said", said)
    ethics, restore = _ld._nets()
    try:
        assert ethics.class_hits("Thinking"), "control: the class net refuses a governed word"
        refused = {w: (ethics.class_hits(w), ethics.banned_hits(w)) for w in said[:3]
                   if ethics.class_hits(w) or ethics.banned_hits(w)}
    finally:
        restore()
    assert not refused, f"the words pass both nets: {refused}"
    assert not [w for w in said[:3] if re.search(r"[^ -~]", w)], "printable ASCII"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
