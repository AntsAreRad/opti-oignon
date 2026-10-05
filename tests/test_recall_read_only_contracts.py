#!/usr/bin/env python3
"""What the onion promises about reading a receipt and closing one.

A receipt stands for a span of turns the window let go, and it shows in the
digest until it is closed. Reading the span used to close it: one verb did
both. A recall a model could ever trigger -- and the design kept a tool for
it in view -- would then let an injected instruction read the receipts away
and erase the trace of what the window dropped.

Reading and closing are two verbs now. A recall hands back the verbatim span
and changes no receipt. Closing is the user's verb, through the route that
passes the user as actor; any other actor is refused by name and nothing
changes. No module a model can reach imports either verb: only the user's
two surfaces, the HTTP route and the terminal session, and the librarian
that defines them.

Loaded through the shared isolation window with the real schemas, the
dependency layer stood in for, and the librarian loaded from source over a
blocked registry; the handlers are called as functions.
"""

import ast
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_ONION = ("probes", "core_store", "receipts", "composer", "peels", "librarian")


def _deps():
    module = types.ModuleType("opti_oignon.api.deps")
    module.MEMORY_AVAILABLE = False
    module.memory_manager = None
    return module


def _open():
    targets = {f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _ONION}
    targets["opti_oignon.api.schemas"] = source("api", "schemas.py")
    targets["opti_oignon.api.routes_memory"] = source("api", "routes_memory.py")
    loaded, restore = isolate(
        targets=targets,
        blocked=("opti_oignon.inference_backend",),
        seeded={"opti_oignon.api.deps": _deps()},
        packages=("opti_oignon.memory", "opti_oignon.api"),
    )
    lib = loaded["opti_oignon.memory.librarian"]
    lib.reset_librarian()
    lib.onion_enabled = lambda path=None: True
    return loaded["opti_oignon.api.routes_memory"], lib, loaded, restore


def _messages(n):
    out = []
    for i in range(1, n + 1):
        role = "user" if i % 2 else "assistant"
        out.append({"role": role, "content": f"Turn {i}: Alice reviewed service {i} on 2026-03-{i:02d} and we agreed that service {i} stays on the new cluster."})
    return out


def _faithful(turns):
    return " ".join(str(t.get("text", "")) for t in turns)


def _with_receipts(routes, lib, loaded):
    """A conversation whose window let go of at least two spans."""
    routes.onion_pin("c1", routes.OnionPinRequest(text="Alice owns the services."))
    peels = loaded["opti_oignon.memory.peels"]
    composer = loaded["opti_oignon.memory.composer"]
    gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
    budget = composer.Budget(window=2000, reserve=200, core=300, receipts=300, peels=800, flesh=200, turn=200)
    state = lib.peek_state("c1")
    state.mirror(_messages(12))
    while lib.curate(state, _faithful, gate=gate, budget=budget).evicted:
        pass
    receipts = routes.onion_state("c1")["receipts"]
    assert len(receipts) >= 2, "control: open receipts to read"
    return state, receipts


def _open_keys(routes):
    return [r["key"] for r in routes.onion_state("c1")["receipts"]]


# ---------------------------------------------------------------------------
# ro1 -- the route's recall leaves the receipt open
# ---------------------------------------------------------------------------

def test_ro1_a_recall_through_the_route_hands_back_the_span_and_leaves_the_receipt_open():
    routes, lib, loaded, restore = _open()
    try:
        state, receipts = _with_receipts(routes, lib, loaded)
        key = receipts[0]["key"]
        recalled = routes.onion_recall("c1", key)
        assert recalled["span"][0]["text"].startswith("Turn 1:"), "the verbatim span"
        assert key in _open_keys(routes), "reading a span does not close its receipt"
    finally:
        restore()


# ---------------------------------------------------------------------------
# ro2 -- the librarian's recall changes no receipt
# ---------------------------------------------------------------------------

def test_ro2_the_librarian_s_recall_changes_no_receipt():
    routes, lib, loaded, restore = _open()
    try:
        state, receipts = _with_receipts(routes, lib, loaded)
        before = [(r.key, r.resolved) for r in state.ledger.all()]
        span = lib.recall("c1", receipts[0]["key"])
        assert span, "the span comes back"
        assert [(r.key, r.resolved) for r in lib.peek_state("c1").ledger.all()] == before
    finally:
        restore()


# ---------------------------------------------------------------------------
# ro3 -- closing a receipt is the user's verb, through its route
# ---------------------------------------------------------------------------

def test_ro3_the_user_closes_a_receipt_through_its_route():
    routes, lib, loaded, restore = _open()
    try:
        state, receipts = _with_receipts(routes, lib, loaded)
        key = receipts[0]["key"]
        answered = routes.onion_resolve("c1", key)
        assert answered["key"] == key and answered["resolved"] is True
        assert key not in _open_keys(routes)
        assert receipts[1]["key"] in _open_keys(routes), "only the named receipt closes"
    finally:
        restore()


# ---------------------------------------------------------------------------
# ro4 -- any other actor is refused by name, and nothing changes
# ---------------------------------------------------------------------------

def test_ro4_a_close_by_any_other_actor_is_refused_by_name_and_changes_nothing():
    routes, lib, loaded, restore = _open()
    try:
        state, receipts = _with_receipts(routes, lib, loaded)
        key = receipts[0]["key"]
        for actor in ("assistant", "model", "tool", "", None):
            with pytest.raises(PermissionError, match=repr(actor)):
                lib.resolve_receipt("c1", key, actor=actor)
        assert key in _open_keys(routes)
    finally:
        restore()


# ---------------------------------------------------------------------------
# ro5 -- only the user's surfaces import the two verbs
# ---------------------------------------------------------------------------

_VERBS = {"recall", "resolve_receipt"}
_ALLOWED = {
    "opti_oignon/api/routes_memory.py",
    "opti_oignon/cli/session.py",
    "opti_oignon/memory/librarian.py",
}


def _users_of_the_verbs():
    found = set()
    root = REPO / "opti_oignon"
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("data", "__pycache__")]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = Path(dirpath) / name
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                hit = (
                    isinstance(node, ast.Attribute) and node.attr in _VERBS
                ) or (
                    isinstance(node, ast.ImportFrom)
                    and (node.module or "").endswith("librarian")
                    and any(a.name in _VERBS for a in node.names)
                ) or (
                    isinstance(node, ast.FunctionDef) and node.name in _VERBS
                    and name == "librarian.py"
                )
                if hit:
                    found.add(path.relative_to(REPO).as_posix())
    return found


def test_ro5_only_the_user_s_surfaces_and_the_librarian_name_the_two_verbs():
    found = _users_of_the_verbs()
    assert {"opti_oignon/api/routes_memory.py", "opti_oignon/cli/session.py"} <= found, (
        "the scan must see the user's own surfaces, or it proves nothing"
    )
    assert sorted(found - _ALLOWED) == []
