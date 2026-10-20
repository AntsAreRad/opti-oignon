#!/usr/bin/env python3
"""Helpers for the journal's contracts: the shared window, a turn's counter, and conversations made by hand.

The onion's modules and the untrusted-data wrapper load from source through
the shared isolation window (``_isolation.isolate``); the native core is
loaded only where a contract asks for it, the reference assembly otherwise.
A conversation's evictions are made with the state's own writers --
``Flesh.evict_span`` and ``PeelTree.add`` -- as the queue makes them, so a
contract on the journal reads a region, receipts and anchors it chose,
never what a gate decided.

Local-only (the public distribution ships no tests).
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian")
LIBRARIAN = "opti_oignon.memory.librarian"
COMPOSER = "opti_oignon.memory.composer"
PEELS = "opti_oignon.memory.peels"
WRAPPER = "opti_oignon.agent.untrusted_context"
NATIVE = "opti_oignon.native"
ONION_YAML = REPO / "opti_oignon" / "config" / "onion.yaml"
SYSTEM = ("You are a careful assistant. Answer from what you are shown.",)
MANIFEST = "Tools: none."


def open_window(*, native=False, store=False):
    """``(loaded, restore)``: the onion's modules and the wrapper, from source; the native core and the onion store
    only if asked."""
    targets = {f"opti_oignon.memory.{name}": source("memory", f"{name}.py")
               for name in MODULES + (("onion_store",) if store else ())}
    targets[WRAPPER] = source("agent", "untrusted_context.py")
    blocked = ("opti_oignon.inference_backend", "opti_oignon.db_utils")
    if native:
        targets[NATIVE] = source("native", "__init__.py")
    else:
        blocked += (NATIVE,)
    loaded, restore = isolate(targets=targets, blocked=blocked, packages=("opti_oignon.memory", "opti_oignon.agent"))
    loaded[LIBRARIAN].reset_librarian()
    return loaded, restore


def count(text):
    """A turn's counter that is not the default: two tokens a word, and one more for every forty characters."""
    return 2 * len(text.split()) + len(text) // 40


def other_count(text):
    """Another model's counter: three tokens a word."""
    return 3 * len(text.split())


def config(lib, **over):
    fields = dict(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=4, temperature=0.1, num_predict=128)
    fields.update(over)
    return lib.LibrarianConfig(**fields)


def typed(text, **extra):
    """A user message typed whole."""
    return dict({"role": "user", "content": text, "origin": "typed", "segments": [[0, len(text), "typed"]]}, **extra)


def answer(text, **extra):
    return dict({"role": "assistant", "content": text}, **extra)


def with_document(words, document, **extra):
    """A user message: words typed, then a document attached after a line the executor writes."""
    head = f"{words}\n\nDocument provided:\n"
    text = head + document
    return dict({"role": "user", "content": text, "origin": "typed",
                 "segments": [[0, len(words), "typed"], [len(head), len(text), "document"]]}, **extra)


def pasted(words, paste, **extra):
    """A user message: words typed, then text pasted in their own run, with no character between them."""
    text = words + paste
    return dict({"role": "user", "content": text, "origin": "typed",
                 "segments": [[0, len(words), "typed"], [len(words), len(text), "document"]]}, **extra)


def rag(*texts, label=None):
    """The retrieval a turn brings, as a caller hands it over: its kind and its items, each a text and its label."""
    return {"kind": "rag", "items": [(text, label) for text in texts]}


def web(*texts, suffix="", label=None):
    """The web results a turn brings, and the sentence that cites them (``suffix``) when given."""
    part = {"kind": "web", "items": [(text, label) for text in texts]}
    if suffix:
        part["suffix"] = suffix
    return part


def conversation(n, start=1):
    """``n`` messages, a typed question then an answer, each naming its own service and day."""
    out = []
    for i in range(start, start + n):
        if i % 2:
            out.append(typed(f"Turn {i}: did Alice review service {i} on 2026-03-{i % 28 + 1:02d}?"))
        else:
            out.append(answer(f"Turn {i}: yes, Alice reviewed service {i - 1} and it stays on the new cluster."))
    return out


def accept(loaded, state, n, summary):
    """The oldest ``n`` Flesh turns leave under an accepted receipt, ``summary`` the peel that stands for them."""
    peels = loaded[PEELS]
    with state.lock:
        receipt = state.flesh.evict_span(n, state.cellar, state.ledger, kind="accepted")
        state.tree.add(peels.Peel(id=peels.peel_id(summary, [receipt.key]), text=summary, level=0,
                                  sources=(receipt.key,), children=(),
                                  source_digest=peels.source_digest(state.cellar, [receipt.key]),
                                  probes_passed=0, probes_total=0))
    return receipt


def hold(state, n, anchors=()):
    """The oldest ``n`` Flesh turns leave under a held receipt, with the places of its anchors."""
    with state.lock:
        return state.flesh.evict_span(n, state.cellar, state.ledger, kind="held", anchors=anchors)


def bare(state, n):
    """The oldest ``n`` Flesh turns leave under a bare receipt."""
    with state.lock:
        return state.flesh.evict_span(n, state.cellar, state.ledger, kind="bare")


def plan(lib, cid, question, *, wrap, model="model-a", counter=count, system=SYSTEM, messages=None, **kw):
    """One turn's plan: ``question`` a message, or the words of a typed one."""
    turn = question if isinstance(question, dict) else typed(question)
    return lib.plan_window(cid, model=model, count=counter, system=system, turn=turn, wrap=wrap, messages=messages,
                           **kw)


def window(loaded, cid, question, *, num_ctx=None, **kw):
    """Plan then compose one turn's window, sent at the plan's rung unless ``num_ctx`` says otherwise, with the real
    wrapper unless ``wrap`` is given."""
    lib = loaded[LIBRARIAN]
    kw.setdefault("wrap", loaded[WRAPPER].wrap)
    return lib.compose_window(plan(lib, cid, question, **kw), num_ctx=num_ctx)


def canonical(win):
    """A window's messages as the bytes a journal serializes them to."""
    return b"".join(role.encode("utf-8") + b"\x00" + content.encode("utf-8", "surrogatepass") + b"\x1e"
                    for role, content, _label in win.messages)


def roles(win):
    return [role for role, _content, _label in win.messages]


def budgets_at_home(loaded, home):
    """The shipped budgets, read again with ``home`` as the rung a caller that names none composes under."""
    return replace(loaded[COMPOSER].load_budgets(), home=home)


def snug(loaded, question, *, budgets=None, system=SYSTEM, counter=count):
    """``budgets`` -- the shipped ones when None -- with the caps of the parts a window lays whole at what a turn
    asking ``question`` under ``system`` lays of them, at every rung: the system prompt and the turn's words as
    counted, the Core, empty, at one token, its cap never none. They leave the memory one token to borrow
    (``idle_room``), so a contract written on a cap that binds holds under the caps it was written for."""
    budgets = budgets or loaded[COMPOSER].load_budgets()
    used, words = sum(counter(text) for text in system), counter(question)
    return replace(budgets, tables=tuple((rung, replace(table, system=used, core=1, turn=words))
                                         for rung, table in budgets.tables))
