#!/usr/bin/env python3
"""Contracts for the user's surface on the onion Core: the memory routes.

The Core changes only on an explicit user action; until this block nothing
let a user act. The route is that action: scoped to a conversation, it
pins, supersedes and recalls through the librarian, passes the user as
actor, and returns every refusal by name. It imports the librarian inside
its handlers, so the route module stays importable where the onion is not
and the onion stays out of the module scope of everything but the
executor's guarded import.

  * OR1 -- the four handlers reach the librarian with the conversation id
    and the user as actor, and hand back what it answers: the pinned id,
    the successor id, the Core entries and open receipts, the verbatim
    span.
  * OR2 -- every refusal travels by name with its own status: the onion
    switched off is 503, an actor refusal 403, an unknown entry or key 404,
    an already superseded entry or an over-cap pin 409, and none of them
    leaves a partial state.
  * OR3 -- the route module imports the onion only inside its handlers,
    the handlers never name an actor other than the user, and the route
    module itself is importable with the onion unreachable.
  * OR6 -- the proposals route lists the decisions the user typed that the
    queue offers the Core, word for word, with their id, turn, origin and
    day, and nothing else.
  * OR7 -- one proposal accepted pins its exact words to the Core as the
    user and answers the entry; one declined leaves the Core as it was.
  * OR8 -- an unknown or decided proposal is refused by name, the onion
    switched off is 503 on each proposals route, and nothing changes.
  * OR9 -- the status route is declared before the conversation's, so that
    "status" is never read as a conversation id.

Local-only (the public distribution ships no tests). The route module is
loaded through the shared isolation window with the real schemas, the
dependency layer stood in for, and the librarian loaded from source over a
blocked registry; the handlers are called as functions, the exception
FastAPI would turn into a response is read directly.
"""

import ast
import hashlib
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_ONION = ("probes", "core_store", "receipts", "composer", "peels", "librarian")
_ROUTES = source("api", "routes_memory.py")


def _deps():
    module = types.ModuleType("opti_oignon.api.deps")
    module.MEMORY_AVAILABLE = False
    module.memory_manager = None
    return module


def _open(*, with_onion=True, enabled=True):
    targets = {}
    if with_onion:
        targets.update({f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _ONION})
    targets["opti_oignon.api.schemas"] = source("api", "schemas.py")
    targets["opti_oignon.api.routes_memory"] = _ROUTES
    loaded, restore = isolate(
        targets=targets,
        blocked=("opti_oignon.inference_backend",),
        seeded={"opti_oignon.api.deps": _deps()},
        packages=("opti_oignon.memory", "opti_oignon.api"),
    )
    routes = loaded["opti_oignon.api.routes_memory"]
    lib = loaded.get("opti_oignon.memory.librarian")
    if lib is not None:
        lib.reset_librarian()
        lib.onion_enabled = lambda path=None: enabled
    return routes, lib, loaded, restore


def _messages(n):
    out = []
    for i in range(1, n + 1):
        role = "user" if i % 2 else "assistant"
        out.append({"role": role, "content": f"Turn {i}: Alice reviewed service {i} on 2026-03-{i:02d} and we agreed that service {i} stays on the new cluster."})
    return out


def _faithful(turns):
    return " ".join(str(t.get("text", "")) for t in turns)


def _status(routes, call):
    """The status FastAPI would answer, read from the exception the handler raises."""
    with pytest.raises(routes.HTTPException) as raised:
        call()
    return raised.value.status_code, str(raised.value.detail)


# ---------------------------------------------------------------------------
# OR1 -- the handlers reach the librarian as the user
# ---------------------------------------------------------------------------
def test_or1_the_handlers_reach_the_librarian_as_the_user_and_hand_back_its_answers():
    routes, lib, loaded, restore = _open()
    try:
        seen = []
        real_pin = lib.pin

        def recording_pin(conversation_id, text, *, actor, **kwargs):
            seen.append((conversation_id, text, actor))
            return real_pin(conversation_id, text, actor=actor, **kwargs)

        lib.pin = recording_pin
        pinned = routes.onion_pin("c1", routes.OnionPinRequest(text="Answers cite their source."))
        assert seen == [("c1", "Answers cite their source.", "user")], "the conversation id and the user as actor"
        assert pinned["conversation_id"] == "c1" and len(pinned["id"]) == 64
        succeeded = routes.onion_supersede("c1", routes.OnionSupersedeRequest(old_id=pinned["id"], text="Answers cite their source, always."))
        assert succeeded["id"] != pinned["id"]

        peels = loaded["opti_oignon.memory.peels"]
        composer = loaded["opti_oignon.memory.composer"]
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        budget = composer.Budget(window=2000, reserve=200, core=300, receipts=300, peels=800, flesh=200, turn=200)
        state = lib.peek_state("c1")
        state.mirror(_messages(12))
        while lib.curate(state, _faithful, gate=gate, budget=budget).evicted:
            pass
        listed = routes.onion_state("c1")
        assert [(e["status"], e["superseded_by"] is not None) for e in listed["core"]] == [("superseded", True), ("active", False)]
        assert len(listed["receipts"]) >= 2 and all(len(r["key"]) == 64 and r["stub"] for r in listed["receipts"])
        key = listed["receipts"][0]["key"]
        recalled = routes.onion_recall("c1", key)
        assert recalled["key"] == key and recalled["span"][0]["text"].startswith("Turn 1:"), "the verbatim span"
        assert key not in [r["key"] for r in routes.onion_state("c1")["receipts"]], "resolved, so no longer open"
        assert routes.onion_state("nobody") == {"conversation_id": "nobody", "core": [], "receipts": []}
        assert lib.peek_state("nobody") is None, "listing an unknown conversation creates nothing"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OR2 -- refusals by name, with their status
# ---------------------------------------------------------------------------
def test_or2_every_refusal_travels_by_name_with_its_status_and_leaves_no_partial_state():
    routes, lib, loaded, restore = _open(enabled=False)
    try:
        status, detail = _status(routes, lambda: routes.onion_pin("c1", routes.OnionPinRequest(text="x")))
        assert status == 503 and "switched off" in detail
        assert lib.peek_state("c1") is None
    finally:
        restore()

    routes, lib, loaded, restore = _open()
    try:
        composer = loaded["opti_oignon.memory.composer"]
        small = composer.Budget(window=2000, reserve=200, core=40, receipts=300, peels=800, flesh=200, turn=200)
        real_load = composer.load_budget
        lib.load_budget = lambda path=None: small
        composer.load_budget = lambda path=None: small
        pinned = routes.onion_pin("c1", routes.OnionPinRequest(text="Answers cite their source."))
        status, detail = _status(routes, lambda: routes.onion_pin("c1", routes.OnionPinRequest(text=" ".join(["word"] * 200))))
        assert status == 409 and "cap of 40" in detail, "an over-cap pin is refused before it lands, with the cap named"
        assert [e["text"] for e in routes.onion_state("c1")["core"]] == ["Answers cite their source."]
        status, detail = _status(routes, lambda: routes.onion_supersede("c1", routes.OnionSupersedeRequest(old_id="0" * 64, text="y")))
        assert status == 404
        routes.onion_supersede("c1", routes.OnionSupersedeRequest(old_id=pinned["id"], text="Answers cite their source, always."))
        status, detail = _status(routes, lambda: routes.onion_supersede("c1", routes.OnionSupersedeRequest(old_id=pinned["id"], text="z")))
        assert status == 409 and "already superseded" in detail
        status, detail = _status(routes, lambda: routes.onion_recall("c1", "0" * 64))
        assert status == 404 and "not in the ledger" in detail
        status, detail = _status(routes, lambda: routes.onion_recall("nobody", "0" * 64))
        assert status == 404 and "no onion state" in detail
        status, detail = _status(routes, lambda: routes.onion_pin("c1", routes.OnionPinRequest(text="   ")))
        assert status == 409 and "empty" in detail

        real_pin = lib.pin
        lib.pin = lambda *a, **k: real_pin(*a, **{**k, "actor": "model"})
        status, detail = _status(routes, lambda: routes.onion_pin("c1", routes.OnionPinRequest(text="The model pins.")))
        assert status == 403 and "explicit user action" in detail, "the store's actor refusal travels as 403"
        composer.load_budget = real_load
    finally:
        restore()


# ---------------------------------------------------------------------------
# OR3 -- the route's own shape
# ---------------------------------------------------------------------------
def test_or3_the_route_imports_the_onion_in_its_handlers_only_and_names_no_other_actor():
    text = _ROUTES.read_text(encoding="utf-8")
    tree = ast.parse(text)
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("librarian" in (getattr(n, "module", "") or "") or any(a.name == "librarian" for a in n.names) for n in top), (
        "the librarian is never imported at module scope"
    )
    inner = [
        n for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
        for n in ast.walk(f) if isinstance(n, ast.ImportFrom) and any(a.name == "librarian" for a in n.names)
    ]
    assert len(inner) >= 1, "and it is imported inside a handler helper"
    actors = {
        kw.value.value for n in ast.walk(tree) if isinstance(n, ast.Call)
        for kw in n.keywords if kw.arg == "actor" and isinstance(kw.value, ast.Constant)
    }
    assert actors == {"user"}, f"the route passes the user as actor and nothing else: {actors}"

    routes, lib, loaded, restore = _open(with_onion=False)
    try:
        assert lib is None and routes.router is not None, "the route module is importable with the onion unreachable"
        status, detail = _status(routes, lambda: routes.onion_state("c1"))
        assert status == 503 and "not available" in detail
    finally:
        restore()


# ---------------------------------------------------------------------------
# OR4 -- what OR1 pinned, now that reading a span leaves its receipt open
# ---------------------------------------------------------------------------
def test_or4_the_handlers_reach_the_librarian_as_the_user_and_a_recall_leaves_the_receipt_open():
    routes, lib, loaded, restore = _open()
    try:
        seen = []
        real_pin = lib.pin

        def recording_pin(conversation_id, text, *, actor, **kwargs):
            seen.append((conversation_id, text, actor))
            return real_pin(conversation_id, text, actor=actor, **kwargs)

        lib.pin = recording_pin
        pinned = routes.onion_pin("c1", routes.OnionPinRequest(text="Answers cite their source."))
        assert seen == [("c1", "Answers cite their source.", "user")], "the conversation id and the user as actor"
        assert pinned["conversation_id"] == "c1" and len(pinned["id"]) == 64
        succeeded = routes.onion_supersede("c1", routes.OnionSupersedeRequest(old_id=pinned["id"], text="Answers cite their source, always."))
        assert succeeded["id"] != pinned["id"]

        peels = loaded["opti_oignon.memory.peels"]
        composer = loaded["opti_oignon.memory.composer"]
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        budget = composer.Budget(window=2000, reserve=200, core=300, receipts=300, peels=800, flesh=200, turn=200)
        state = lib.peek_state("c1")
        state.mirror(_messages(12))
        while lib.curate(state, _faithful, gate=gate, budget=budget).evicted:
            pass
        listed = routes.onion_state("c1")
        assert [(e["status"], e["superseded_by"] is not None) for e in listed["core"]] == [("superseded", True), ("active", False)]
        assert len(listed["receipts"]) >= 2 and all(len(r["key"]) == 64 and r["stub"] for r in listed["receipts"])
        key = listed["receipts"][0]["key"]
        recalled = routes.onion_recall("c1", key)
        assert recalled["key"] == key and recalled["span"][0]["text"].startswith("Turn 1:"), "the verbatim span"
        assert key in [r["key"] for r in routes.onion_state("c1")["receipts"]], "read, and still open"
        closed = routes.onion_resolve("c1", key)
        assert closed == {"conversation_id": "c1", "key": key, "resolved": True}
        assert key not in [r["key"] for r in routes.onion_state("c1")["receipts"]], "closed by the user"
        assert routes.onion_state("nobody") == {"conversation_id": "nobody", "core": [], "receipts": []}
        assert lib.peek_state("nobody") is None, "listing an unknown conversation creates nothing"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OR5 -- the code route reads a block by its key, as the user
# ---------------------------------------------------------------------------
def test_or5_the_code_route_reads_a_block_by_its_key_and_refuses_an_unknown_one_by_name():
    routes, lib, loaded, restore = _open()
    try:
        peels = loaded["opti_oignon.memory.peels"]
        body = "SELECT name FROM nodes WHERE alive = 1"
        state = lib.state_for("c1")
        state.mirror([
            {"role": "user", "origin": "typed", "content": f"Here is the query.\n```sql\n{body}\n```\nCarol runs it every morning."},
            {"role": "assistant", "origin": "assistant", "content": "Noted: Carol runs the query every morning."},
        ])
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        step = peels.evict_gated(flesh=state.flesh, cellar=state.cellar, ledger=state.ledger,
                                 tree=state.tree, gate=gate, summarize=_faithful)
        assert step.evicted, step.reason
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
        got = routes.onion_code("c1", digest)
        assert got == {"conversation_id": "c1", "key": f"code:{digest}", "language": "sql", "code": body}
        assert _status(routes, lambda: routes.onion_code("c1", "f" * 12)) == (404, "no code block behind code:" + "f" * 12)
        assert _status(routes, lambda: routes.onion_code("nobody", digest))[0] == 404
        assert _status(routes, lambda: routes.onion_code("c1", "XYZ"))[0] == 404
    finally:
        restore()


# ---------------------------------------------------------------------------
# OR6-OR9 -- the proposals, decided by the user; the status route's place
# ---------------------------------------------------------------------------
_DECIDED = "We keep Docker on the build server."
_LOSSY = "Alice moved the build to Berlin on 2026-03-04. The Berlin build runs 12 jobs a day. Bob checks the logs every morning."


def _held_decision(lib, loaded, cid):
    """A typed decision the queue holds in the Cellar and offers the Core, in conversation ``cid``."""
    from dataclasses import replace

    peels, composer = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.composer"]
    state = lib.state_for(cid)
    state.mirror([
        {"role": "user", "origin": "typed", "segments": [],
         "content": "Alice moved the build to Berlin on 2026-03-04. " + _DECIDED},
        {"role": "assistant", "origin": "assistant", "segments": [],
         "content": "Noted: the Berlin build runs 12 jobs a day. Bob checks the logs every morning."},
    ])
    budget = composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=1, turn=60)
    outcome = lib.curate(state, lambda turns: _LOSSY, gate=replace(peels.load_gate(), span_turns=2), budget=budget,
                         ladder=replace(peels.load_ladder(), rho=0.1))
    assert outcome.rung == "held", "control: held, and offered"
    return state


def test_or6_the_proposals_route_lists_the_decisions_word_for_word():
    routes, lib, loaded, restore = _open()
    try:
        _held_decision(lib, loaded, "c1")
        listed = routes.onion_proposals("c1")
        assert set(listed) == {"conversation_id", "proposals"} and listed["conversation_id"] == "c1"
        (offered,) = listed["proposals"]
        assert set(offered) == {"id", "text", "turn_id", "origin", "made_on"}
        assert (offered["text"], offered["turn_id"], offered["origin"]) == (_DECIDED, "t0001", "typed")
        assert len(offered["id"]) == 64 and offered["made_on"] == lib._today()
        assert routes.onion_proposals("nobody") == {"conversation_id": "nobody", "proposals": []}
    finally:
        restore()


def test_or7_one_proposal_accepted_pins_its_words_and_one_declined_leaves_the_core():
    routes, lib, loaded, restore = _open()
    try:
        _held_decision(lib, loaded, "c1")
        _held_decision(lib, loaded, "c2")
        first = routes.onion_proposals("c1")["proposals"][0]["id"]
        accepted = routes.onion_accept_proposal("c1", first)
        assert accepted["decision"] == "accepted" and accepted["id"] == first
        assert [(e["id"], e["text"]) for e in routes.onion_state("c1")["core"]] == [(accepted["entry_id"], _DECIDED)]
        second = routes.onion_proposals("c2")["proposals"][0]["id"]
        declined = routes.onion_decline_proposal("c2", second)
        assert declined == {"conversation_id": "c2", "id": second, "decision": "declined", "entry_id": None}
        assert routes.onion_state("c2")["core"] == [] and routes.onion_proposals("c2")["proposals"] == []
    finally:
        restore()


def test_or8_an_unknown_or_decided_proposal_is_refused_by_name_and_the_onion_off_is_503():
    routes, lib, loaded, restore = _open()
    try:
        _held_decision(lib, loaded, "c1")
        pid = routes.onion_proposals("c1")["proposals"][0]["id"]
        code, detail = _status(routes, lambda: routes.onion_accept_proposal("c1", "f" * 64))
        assert code == 404 and "no proposal" in detail
        routes.onion_decline_proposal("c1", pid)
        code, detail = _status(routes, lambda: routes.onion_accept_proposal("c1", pid))
        assert code == 404 and "not open" in detail
        assert routes.onion_state("c1")["core"] == [], "nothing pinned"
        lib.onion_enabled = lambda path=None: False
        for call in (lambda: routes.onion_proposals("c1"), lambda: routes.onion_accept_proposal("c1", pid),
                     lambda: routes.onion_decline_proposal("c1", pid)):
            assert _status(routes, call)[0] == 503
    finally:
        restore()


def test_or9_the_status_route_is_declared_before_the_conversation_s():
    routes, lib, loaded, restore = _open()
    try:
        paths = [getattr(route, "path", "") for route in routes.router.routes]
        assert "/api/memory/onion/status" in paths and "/api/memory/onion/{conv_id}" in paths, "control"
        assert paths.index("/api/memory/onion/status") < paths.index("/api/memory/onion/{conv_id}")
    finally:
        restore()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
