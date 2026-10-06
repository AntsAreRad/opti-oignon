#!/usr/bin/env python3
"""Contracts for the origin of a conversation turn.

A turn is saved with the origin of its words: typed by the user, the user's
question as the model rewrote it, a document the user attached, the model's
answer, or ``legacy`` for a turn written before origins were kept or by a
path that cannot vouch for one. Only typed text may later make a decision
probe, so the origin is a trust label. It has to be true where a turn is
written, survive wherever the turn travels, and never reach the model.
Segments are bounds on a turn's content, never text: the parts that came
from elsewhere, each with its own base. A character no segment covers
belongs to no one and is never probed.

  * OT1 -- the grammar admits exactly eight origins, and each role its own.
  * OT2 -- an origin outside the grammar is refused by name, rule by rule.
  * OT3 -- a segment list outside the grammar is refused by name, rule by
    rule.
  * OT4 -- the grammar stands in three modules, each loaded alone where it
    is tested, and the three are one text.
  * OT5 -- a fresh store carries the two columns; an older store gains them
    at its first connection and every row it held reads as legacy.
  * OT6 -- a saved turn reads back with its origin and segments through the
    mirror's read, and the context read still hands over role and content
    only.
  * OT7 -- a turn saved without an origin is legacy.
  * OT8 -- an origin outside the grammar is refused before anything is
    written.
  * OT9 -- a synced conversation lands as legacy whatever its payload
    claims.
  * OT10 -- a migrated history entry lands as legacy.
  * OT11 -- the executor saves a typed question as typed, with no segment,
    and the answer as the assistant's.
  * OT12 -- a question the model rewrote is saved as refined; a rewrite
    that changed nothing leaves it typed.
  * OT13 -- an attached document is a segment of its own, and the separator
    the executor writes between the two belongs to no one.
  * OT14 -- an answer is flagged web exactly when web results reached its
    prompt.
  * OT15 -- an answer served from the cache is saved with the same origins.
  * OT16 -- after the turn, the librarian is offered the conversation
    through the origin read. Supersedes XW4, whose fake store has no such
    read.
  * OT17 to OT20 -- each agentic pipeline that saves its own turn saves the
    question as typed and the answer flagged tool. They supersede the four
    contracts of the pipeline-persistence suite, whose fake store predates
    the origin, and keep everything those four held.
  * OT21 -- the coding agent saves the question as typed and the answer
    flagged tool.
  * OT22 -- a branch store gains the two columns, and its older rows read as
    legacy.
  * OT23 -- a message the user posts to a branch is typed and any other
    role is legacy; the branch store refuses an origin outside the grammar.
  * OT24 -- a merge copies each message with its origin and segments.
  * OT25 -- every message write site in the package is accounted for: the
    sites that name an origin, the benches and evaluations that keep the
    legacy default, and every insert statement names the origin.
  * OT26 -- the mirror carries origin and segments into the Flesh.
  * OT27 -- a declaration outside the grammar is mirrored as legacy with no
    segment, and the log names the turn, never its text.
  * OT28 -- the summariser is handed turn, role and text, never an origin.
  * OT29 -- an evicted span keeps its labels in the Cellar, and its receipt
    and its peel answer for their union.
  * OT30 -- a probe carries the base of the piece it was drawn from and the
    role of its turn; the separator is never drawn from.
  * OT31 -- the native core draws the same probes, piece by piece, and is
    the one that drew them.

Local-only (the public distribution ships no tests).
"""

import ast
import dataclasses
import json
import logging
import os
import sqlite3
import sys
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402
from _registry_bridge import seed_registry  # noqa: E402

_CANONICAL = {
    "typed", "refined", "document", "legacy",
    "assistant", "assistant+tool", "assistant+web", "assistant+tool+web",
}
_SEPARATOR_WORDS = "Document provided"
_QUESTION = "Where does Alice meet Bob on 2024-03-15?"
_DOCUMENT = "Contoso opens the venue to 40 guests."


def _raw(path):
    return sqlite3.connect(str(path))


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def _plain_connect(path, **kw):
    return sqlite3.connect(path, check_same_thread=kw.get("check_same_thread", False))


def _store(tmp_path):
    """conversation.py alone, over plain sqlite in ``tmp_path``, the field key a reversible marker."""
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = _plain_connect
    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = Path(tmp_path)
    loaded, restore = isolate(
        targets={"opti_oignon.conversation": source("conversation.py")},
        seeded={"opti_oignon.db_utils": db, "opti_oignon.config": cfg},
    )
    mod = loaded["opti_oignon.conversation"]
    mod._encrypt = lambda v: "E:" + v
    mod._decrypt = lambda v: v[2:] if isinstance(v, str) and v.startswith("E:") else v
    return mod, restore


def _branches(tmp_path):
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = _plain_connect
    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = Path(tmp_path)
    loaded, restore = isolate(
        targets={"opti_oignon.conversation_branches": source("conversation_branches.py")},
        blocked=("opti_oignon.context_manager",),
        seeded={"opti_oignon.db_utils": db, "opti_oignon.config": cfg},
    )
    return loaded["opti_oignon.conversation_branches"], restore


def _branch_manager(cb, path):
    return cb.ConversationBranchManager(db_path=path, config=json.loads(json.dumps(cb._DEFAULT_CONFIG)))


_ONION = ("probes", "core_store", "receipts", "composer", "peels", "librarian")


def _librarian():
    loaded, restore = isolate(
        targets={f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _ONION},
        blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils"),
        packages=("opti_oignon.memory",),
    )
    lib = loaded["opti_oignon.memory.librarian"]
    lib.reset_librarian()
    return lib, loaded, restore


def _memory(*names, native=False):
    targets = {f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in names}
    blocked = ()
    if native:
        targets["opti_oignon.native"] = source("native", "__init__.py")
    else:
        blocked = ("opti_oignon.native",)
    loaded, restore = isolate(targets=targets, blocked=blocked, packages=("opti_oignon.memory",))
    return loaded, restore


def _with_document(question=_QUESTION, document=_DOCUMENT, separator="\n\n---\nDocument provided:\n"):
    content = question + separator + document
    segments = [[0, len(question), "typed"], [len(content) - len(document), len(content), "document"]]
    return content, segments


# ---------------------------------------------------------------------------
# OT1 -- the closure of the grammar
# ---------------------------------------------------------------------------
def test_ot1_the_grammar_admits_exactly_eight_origins_and_each_role_its_own(tmp_path):
    mod, restore = _store(tmp_path)
    try:
        bases = ("typed", "refined", "document", "assistant", "legacy", "", "Typed", "user", "system", "tool", "web")
        flag_lists = ([], ["tool"], ["web"], ["tool", "web"], ["web", "tool"], ["tool", "tool"], ["web", "web"],
                      [""], ["remote"], ["Web"])
        accepted = {}
        for role in ("user", "assistant", "system", "tool", None):
            for base in bases:
                for flags in flag_lists:
                    origin = "+".join([base] + flags)
                    if mod._origin_defect(role, origin, [], 0) is None:
                        accepted.setdefault(role, set()).add(origin)
        everything = set().union(*accepted.values()) if accepted else set()
        assert len(everything) >= 8, f"the grammar admits something: {sorted(everything)}"
        assert everything == _CANONICAL
        assert accepted["user"] == {"typed", "refined", "document", "legacy"}
        assert accepted["assistant"] == {"assistant", "assistant+tool", "assistant+web", "assistant+tool+web", "legacy"}
        for role in ("system", "tool", None):
            assert accepted[role] == {"legacy"}, f"a {role} turn can only be legacy"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT2 -- an origin outside the grammar, refused by name
# ---------------------------------------------------------------------------
def test_ot2_an_origin_outside_the_grammar_is_refused_by_name(tmp_path):
    mod, restore = _store(tmp_path)
    try:
        cases = [
            ("user", 7, "non-empty string"),
            ("user", "", "non-empty string"),
            ("user", None, "non-empty string"),
            ("user", "typd", "origin base 'typd'"),
            ("assistant", "assistant+cache", "origin flag 'cache'"),
            ("assistant", "assistant+", "origin flag ''"),
            ("assistant", "assistant+web+tool", "once each, in order"),
            ("assistant", "assistant+web+web", "once each, in order"),
            ("user", "typed+web", "a typed origin carries no flag"),
            ("user", "document+tool", "a document origin carries no flag"),
            ("assistant", "typed", "role 'assistant' cannot carry typed"),
            ("system", "typed", "role 'system' cannot carry typed"),
            ("user", "assistant", "role 'user' cannot carry assistant"),
            (["user"], "typed", "cannot carry typed"),
        ]
        for role, origin, reason in cases:
            defect = mod._origin_defect(role, origin, [], 0)
            assert defect is not None, f"{origin!r} as {role!r} lies outside the grammar"
            assert reason in defect, f"{origin!r} as {role!r}: the refusal names its rule, got {defect!r}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT3 -- a segment list outside the grammar, refused by name
# ---------------------------------------------------------------------------
def test_ot3_a_segment_list_outside_the_grammar_is_refused_by_name(tmp_path):
    mod, restore = _store(tmp_path)
    try:
        good = [[0, 5, "typed"], [9, 20, "document"]]
        assert mod._origin_defect("user", "typed", good, 20) is None, "control: two disjoint parts in bounds"
        assert mod._origin_defect("assistant", "assistant+web", [[0, 20, "assistant"]], 20) is None
        cases = [
            ("user", "typed", "0-5", "segments are a list"),
            ("user", "typed", None, "segments are a list"),
            ("user", "typed", [[0, 5]], "[start, end, base]"),
            ("user", "typed", [[0, 5, "typed", 1]], "[start, end, base]"),
            ("user", "typed", ["typed"], "[start, end, base]"),
            ("user", "typed", [[0.0, 5, "typed"]], "integers"),
            ("user", "typed", [[True, 5, "typed"]], "integers"),
            ("user", "typed", [[5, 5, "typed"]], "overlaps, is empty or leaves the content"),
            ("user", "typed", [[0, 21, "typed"]], "overlaps, is empty or leaves the content"),
            ("user", "typed", [[-1, 5, "typed"]], "overlaps, is empty or leaves the content"),
            ("user", "typed", [[0, 10, "typed"], [5, 15, "document"]], "overlaps, is empty or leaves the content"),
            ("user", "typed", [[10, 15, "typed"], [0, 5, "document"]], "overlaps, is empty or leaves the content"),
            ("user", "typed", [[0, 5, "assistant"]], "role 'user' carries no assistant segment"),
            ("user", "typed", [[0, 5, "legacy"]], "role 'user' carries no legacy segment"),
            ("user", "typed", [[0, 5, "typed+web"]], "carries no typed+web segment"),
            ("assistant", "assistant", [[0, 5, "typed"]], "role 'assistant' carries no typed segment"),
            ("user", "legacy", [[0, 5, "typed"]], "a legacy turn has no segments"),
        ]
        for role, origin, segments, reason in cases:
            defect = mod._origin_defect(role, origin, segments, 20)
            assert defect is not None, f"{segments!r} lies outside the grammar"
            assert reason in defect, f"{segments!r}: the refusal names its rule, got {defect!r}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT4 -- three homes, one text
# ---------------------------------------------------------------------------
_GRAMMAR_HOMES = (
    ("conversation.py",),
    ("conversation_branches.py",),
    ("memory", "probes.py"),
)
_GRAMMAR_NAMES = ("_ORIGIN_BASES", "_ORIGIN_FLAGS", "_ORIGIN_ROLES", "_origin_defect")


def _grammar_nodes(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in _GRAMMAR_NAMES):
            found.setdefault(node.targets[0].id, []).append(ast.dump(node.value))
        elif isinstance(node, ast.FunctionDef) and node.name in _GRAMMAR_NAMES:
            found.setdefault(node.name, []).append(ast.dump(node))
    return found


def test_ot4_the_three_copies_of_the_grammar_are_one_text():
    homes = [REPO / "opti_oignon" / Path(*parts) for parts in _GRAMMAR_HOMES]
    readings = {}
    for path in homes:
        found = _grammar_nodes(path)
        for name in _GRAMMAR_NAMES:
            assert len(found.get(name, [])) == 1, f"{path.name} holds {name} once at module level"
        readings[path] = {name: found[name][0] for name in _GRAMMAR_NAMES}
    assert len(readings) == 3
    first, *others = homes
    for other in others:
        for name in _GRAMMAR_NAMES:
            assert readings[other][name] == readings[first][name], (
                f"{name} in {other.relative_to(REPO)} is not the text of {first.relative_to(REPO)}"
            )


# ---------------------------------------------------------------------------
# OT5 -- the columns, fresh and migrated
# ---------------------------------------------------------------------------
_OLD_SCHEMA = """
CREATE TABLE conversations (
    id TEXT PRIMARY KEY, title TEXT DEFAULT 'New conversation', created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, model TEXT, task_type TEXT, preset TEXT, metadata TEXT DEFAULT '{}'
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL, timestamp TEXT NOT NULL, token_estimate INTEGER DEFAULT 0, model TEXT,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
"""


def _columns(path, table):
    conn = _raw(path)
    try:
        return {row[1]: row for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def test_ot5_a_fresh_store_carries_the_origin_columns_and_an_older_one_gains_them_as_legacy(tmp_path):
    mod, restore = _store(tmp_path)
    try:
        fresh = tmp_path / "fresh.db"
        mod.ConversationManager(db_path=fresh).create_conversation(title="t")
        cols = _columns(fresh, "messages")
        assert cols["origin"][3] == 1 and cols["origin"][4] == "'legacy'"
        assert cols["segments"][3] == 1 and cols["segments"][4] == "'[]'"

        old = tmp_path / "old.db"
        conn = _raw(old)
        conn.executescript(_OLD_SCHEMA)
        conn.execute("INSERT INTO conversations (id, created_at, updated_at) VALUES ('c-old', 't0', 't0')")
        conn.execute("INSERT INTO messages (conversation_id, role, content, timestamp) VALUES ('c-old', 'user', 'E:hello', 't1')")
        conn.execute("INSERT INTO messages (conversation_id, role, content, timestamp) VALUES ('c-old', 'assistant', 'E:hi', 't2')")
        conn.commit()
        conn.close()
        assert "origin" not in _columns(old, "messages"), "control: the older file has no origin"

        rows = mod.ConversationManager(db_path=old).get_mirror_messages("c-old")
        assert len(rows) >= 2, "the rows the older file held are read"
        assert [(r["role"], r["content"]) for r in rows] == [("user", "hello"), ("assistant", "hi")]
        assert all(r["origin"] == "legacy" and r["segments"] == [] for r in rows)
        assert {"origin", "segments"} <= set(_columns(old, "messages"))
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT6 -- read back through the mirror's read; the context read unchanged
# ---------------------------------------------------------------------------
def test_ot6_a_saved_turn_reads_back_with_its_origin_and_the_context_read_does_not_carry_it(tmp_path):
    mod, restore = _store(tmp_path)
    try:
        mgr = mod.ConversationManager(db_path=tmp_path / "c.db")
        conv = mgr.create_conversation(title="t")
        content, segments = _with_document()
        mgr.add_message(conv.id, "system", "Be brief.")
        mgr.add_message(conv.id, "user", content, origin="typed", segments=segments)
        mgr.add_message(conv.id, "assistant", "At the venue.", model="m", origin="assistant+web")
        rows = mgr.get_mirror_messages(conv.id)
        assert [(r["role"], r["content"], r["origin"], r["segments"]) for r in rows] == [
            ("user", content, "typed", segments),
            ("assistant", "At the venue.", "assistant+web", []),
        ]
        context = mgr.get_context_messages(conv.id)
        assert context == [{"role": "user", "content": content}, {"role": "assistant", "content": "At the venue."}]
        assert all(set(m) == {"role", "content"} for m in context), "the model is never handed an origin"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT7 -- no origin given: legacy
# ---------------------------------------------------------------------------
def test_ot7_a_turn_saved_without_an_origin_is_legacy(tmp_path):
    mod, restore = _store(tmp_path)
    try:
        path = tmp_path / "c.db"
        mgr = mod.ConversationManager(db_path=path)
        conv = mgr.create_conversation(title="t")
        assert mgr.add_message(conv.id, "user", "Plain words.") is not None
        rows = mgr.get_mirror_messages(conv.id)
        assert [(r["origin"], r["segments"]) for r in rows] == [("legacy", [])]
        conn = _raw(path)
        try:
            stored = conn.execute("SELECT origin, segments FROM messages").fetchall()
        finally:
            conn.close()
        assert stored == [("legacy", "[]")]
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT8 -- refused before anything is written
# ---------------------------------------------------------------------------
def test_ot8_an_origin_outside_the_grammar_is_refused_before_anything_is_written(tmp_path):
    mod, restore = _store(tmp_path)
    try:
        path = tmp_path / "c.db"
        mgr = mod.ConversationManager(db_path=path)
        conv = mgr.create_conversation(title="t")
        assert mgr.add_message(conv.id, "user", "Kept.", origin="typed") is not None, "control: a good turn lands"
        refusals = [
            dict(role="user", content="Lost.", origin="typd"),
            dict(role="assistant", content="Lost.", origin="typed"),
            dict(role="user", content="Lost.", origin="typed", segments=[[0, 99, "typed"]]),
            dict(role="user", content="Lost.", origin="legacy", segments=[[0, 2, "typed"]]),
        ]
        for kwargs in refusals:
            role, content = kwargs.pop("role"), kwargs.pop("content")
            with pytest.raises(mod.OriginError) as caught:
                mgr.add_message(conv.id, role, content, **kwargs)
            assert "origin" in str(caught.value) or "segment" in str(caught.value)
        conn = _raw(path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        finally:
            conn.close()
        assert count == 1, "a refused turn writes nothing"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT9 -- a synced conversation lands as legacy
# ---------------------------------------------------------------------------
def test_ot9_a_synced_conversation_lands_as_legacy_whatever_its_payload_claims(tmp_path):
    mod, restore = _store(tmp_path)
    try:
        path = tmp_path / "c.db"
        mgr = mod.ConversationManager(db_path=path)
        claimed = [
            {"role": "user", "content": "We pick Monday.", "timestamp": "t1", "token_estimate": 1, "model": None,
             "metadata": {"origin": "typed"}, "origin": "typed", "segments": [[0, 5, "typed"]]},
            {"role": "assistant", "content": "Monday it is.", "timestamp": "t2", "token_estimate": 1, "model": None,
             "metadata": {}, "origin": "typed"},
        ]
        payload = {"user_id": "u", "conversation": {
            "id": "c-sync", "title": "s", "created_at": "t0", "updated_at": "t2", "model": None,
            "task_type": None, "preset": None, "metadata": {}, "messages": claimed}}
        assert mgr.apply_synced_conversation(payload) is True
        rows = mgr.get_mirror_messages("c-sync")
        assert len(rows) >= 2, "control: the synced turns landed"
        assert all(r["origin"] == "legacy" and r["segments"] == [] for r in rows)
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT10 -- a migrated history entry lands as legacy
# ---------------------------------------------------------------------------
def test_ot10_a_migrated_history_entry_lands_as_legacy(tmp_path):
    mod, restore = _store(tmp_path)
    try:
        path = tmp_path / "c.db"
        mgr = mod.ConversationManager(db_path=path)
        mgr._migrate_single_entry({"question": "Shall we ship Friday?", "response": "Yes.", "timestamp": "t0"})
        conn = _raw(path)
        try:
            stored = conn.execute("SELECT role, origin, segments FROM messages ORDER BY id").fetchall()
        finally:
            conn.close()
        assert len(stored) >= 2, "control: the entry became two turns"
        assert stored == [("user", "legacy", "[]"), ("assistant", "legacy", "[]")]
    finally:
        restore()


# ---------------------------------------------------------------------------
# The executor's window: the real executor over a scripted registry, a store
# that keeps what it is handed, and a librarian that records what it is
# offered.
# ---------------------------------------------------------------------------
_EXECUTOR = "opti_oignon.executor"
_WRAPPER = "opti_oignon.agent.untrusted_context"
_LIBRARIAN = "opti_oignon.memory.librarian"


class _Scripted:
    def __init__(self, reply=("At the ", "venue.")):
        self.calls, self.reply = [], reply

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        head, tail = self.reply
        return iter([{"message": {"content": head}}, {"message": {"content": tail}, "done": True}])


class _Store:
    """A conversation store that keeps what it is handed, origins included."""

    def __init__(self):
        self.saved = []

    def add_message(self, conversation_id, role, content, model=None, metadata=None, *, origin="legacy", segments=()):
        self.saved.append({"conversation_id": conversation_id, "role": role, "content": content,
                           "origin": origin, "segments": [list(s) for s in segments]})

    def _of(self, conversation_id):
        return [m for m in self.saved if m["conversation_id"] == conversation_id]

    def get_context_messages(self, conversation_id, **kwargs):
        return [{"role": m["role"], "content": m["content"]} for m in self._of(conversation_id)]

    def get_mirror_messages(self, conversation_id):
        return [{"role": m["role"], "content": m["content"], "origin": m["origin"], "segments": m["segments"]}
                for m in self._of(conversation_id)]

    def get_conversation(self, conversation_id):
        return SimpleNamespace(id=conversation_id, messages=self.get_context_messages(conversation_id), metadata={})

    def update_conversation_metadata(self, conversation_id, *args, **kwargs):
        return None


class _Librarian:
    def __init__(self):
        self.curate_calls = []

    def onion_enabled(self, path=None):
        return True

    def memory_block(self, conversation_id, question=None, **kwargs):
        return ""

    def maybe_curate(self, conversation_id, messages, **kwargs):
        self.curate_calls.append((conversation_id, [dict(m) for m in (messages or [])]))
        return True


class _Cache:
    enabled = True

    def make_cache_key(self, *args, **kwargs):
        return "c" * 16

    def make_conversation_cache_key(self, *args, **kwargs):
        return "c" * 16

    def get(self, key):
        return SimpleNamespace(response="Cached: at the venue.")


_WEB_RESULT = {"title": "Venue", "snippet": "The hall opens on Monday.", "url": "https://example.org/venue"}


def _executor(*, web_results=None, cache=None):
    scripted = _Scripted()
    ollama_stub = types.ModuleType("ollama")
    ollama_stub.chat = scripted.chat
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(get_model=lambda *a, **k: "test-model:1b", get_temperature=lambda *a, **k: 0.2)
    router = types.ModuleType("opti_oignon.router")
    router.RoutingResult = type("RoutingResult", (), {})
    retrieval = types.ModuleType("opti_oignon.memory.retrieval")
    retrieval.build_memory_block = lambda *a, **k: ""
    retrieval.working_memory_block = lambda *a, **k: ""
    store = _Store()
    conversation = types.ModuleType("opti_oignon.conversation")
    conversation.conversation_manager = store
    librarian = _Librarian()
    lib_module = types.ModuleType(_LIBRARIAN)
    lib_module.onion_enabled = librarian.onion_enabled
    lib_module.memory_block = librarian.memory_block
    lib_module.maybe_curate = librarian.maybe_curate
    seeded = {
        "opti_oignon.config": cfg,
        "opti_oignon.router": router,
        "opti_oignon.memory.retrieval": retrieval,
        "opti_oignon.conversation": conversation,
        _LIBRARIAN: lib_module,
    }
    if web_results is not None:
        switch = types.ModuleType("opti_oignon.search_killswitch")
        switch.search_killswitch = SimpleNamespace(is_killed=lambda: False)
        engine = types.ModuleType("opti_oignon.web_search")
        engine.web_search_engine = SimpleNamespace(search=lambda query, max_results=5: list(web_results))
        seeded["opti_oignon.search_killswitch"] = switch
        seeded["opti_oignon.web_search"] = engine
    if cache is not None:
        cache_module = types.ModuleType("opti_oignon.response_cache")
        cache_module.response_cache = cache
        seeded["opti_oignon.response_cache"] = cache_module
    targets = {
        _WRAPPER: source("agent", "untrusted_context.py"),
        "opti_oignon.context_dedup": source("context_dedup.py"),
        _EXECUTOR: source("executor.py"),
    }
    seed_registry(seeded, scripted)
    had, prev = "ollama" in sys.modules, sys.modules.get("ollama")
    sys.modules["ollama"] = ollama_stub
    loaded, win_restore = isolate(targets=targets, seeded=seeded, packages=("opti_oignon.agent", "opti_oignon.memory"))

    def restore():
        win_restore()
        if had:
            sys.modules["ollama"] = prev
        else:
            sys.modules.pop("ollama", None)

    return loaded[_EXECUTOR], scripted, store, librarian, restore


def _routing():
    return SimpleNamespace(model="test-model:1b", task_type="general", temperature=0.2,
                           prompt_variant="standard", timeout=30)


def _drive(gen):
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def _saved(store, conversation_id="conv-1"):
    return [(m["role"], m["origin"], m["segments"]) for m in store._of(conversation_id)]


# ---------------------------------------------------------------------------
# OT11 -- typed and answered
# ---------------------------------------------------------------------------
def test_ot11_the_executor_saves_a_typed_question_as_typed_and_its_answer_as_the_assistant():
    mod, scripted, store, librarian, restore = _executor()
    try:
        _drive(mod.Executor().execute(_QUESTION, _routing(), refine=False, conversation_id="conv-1"))
        assert len(store.saved) >= 2, "control: the turn was saved"
        assert _saved(store) == [("user", "typed", []), ("assistant", "assistant", [])]
        assert store.saved[0]["content"] == _QUESTION
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT12 -- rewritten by the model: refined; rewritten to the same words: typed
# ---------------------------------------------------------------------------
def test_ot12_a_question_the_model_rewrote_is_saved_refined_and_an_unchanged_one_typed():
    rewritten = "On 2024-03-15, where do Alice and Bob meet?"
    for refine_to, expected in ((rewritten, "refined"), (_QUESTION, "typed")):
        mod, scripted, store, librarian, restore = _executor()
        try:
            ex = mod.Executor()
            ex.refine_question = lambda question, document=None, model=None, temperature=0.3, _to=refine_to: (_to, None)
            _drive(ex.execute(_QUESTION, _routing(), refine=True, conversation_id="conv-1"))
            assert len(store.saved) >= 2, "control: the turn was saved"
            assert store.saved[0]["content"] == refine_to
            assert _saved(store)[0] == ("user", expected, []), f"a rewrite to {refine_to!r} is {expected}"
        finally:
            restore()


# ---------------------------------------------------------------------------
# OT13 -- the document is a segment of its own
# ---------------------------------------------------------------------------
def test_ot13_an_attached_document_is_its_own_segment_and_the_separator_belongs_to_no_one():
    mod, scripted, store, librarian, restore = _executor()
    try:
        _drive(mod.Executor().execute(_QUESTION, _routing(), document=_DOCUMENT, refine=False, conversation_id="conv-1"))
        assert len(store.saved) >= 2, "control: the turn was saved"
        user = store.saved[0]
        content = user["content"]
        assert content.startswith(_QUESTION) and content.endswith(_DOCUMENT), "control: the turn joins the two"
        doc_start = len(content) - len(_DOCUMENT)
        assert user["origin"] == "typed"
        assert user["segments"] == [[0, len(_QUESTION), "typed"], [doc_start, len(content), "document"]]
        unclaimed = content[len(_QUESTION):doc_start]
        assert _SEPARATOR_WORDS in unclaimed, "the words the executor writes between the two are covered by no segment"
        assert store.saved[1]["origin"] == "assistant"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT14 -- web results in the prompt: the answer is flagged web
# ---------------------------------------------------------------------------
def test_ot14_an_answer_is_flagged_web_exactly_when_web_results_reached_its_prompt():
    mod, scripted, store, librarian, restore = _executor(web_results=[_WEB_RESULT])
    try:
        _drive(mod.Executor().execute(_QUESTION, _routing(), refine=False, web_search=True, conversation_id="conv-1"))
        sent = json.dumps(scripted.calls[0]["messages"]) if scripted.calls else ""
        assert "The hall opens on Monday." in sent, "control: the web results reached the prompt"
        assert _saved(store) == [("user", "typed", []), ("assistant", "assistant+web", [])]
    finally:
        restore()
    mod, scripted, store, librarian, restore = _executor(web_results=[])
    try:
        _drive(mod.Executor().execute(_QUESTION, _routing(), refine=False, web_search=True, conversation_id="conv-1"))
        assert len(store.saved) >= 2, "control: the turn was saved"
        assert _saved(store)[1] == ("assistant", "assistant", []), "a search that brought nothing flags nothing"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT15 -- served from the cache, saved with the same origins
# ---------------------------------------------------------------------------
def test_ot15_an_answer_served_from_the_cache_is_saved_with_the_same_origins():
    mod, scripted, store, librarian, restore = _executor(cache=_Cache())
    try:
        ex = mod.Executor()
        ex._cache_enabled = True
        _drive(ex.execute(_QUESTION, _routing(), document=_DOCUMENT, refine=False, conversation_id="conv-1"))
        assert ex._last_cache_hit is True, "control: the cache answered"
        assert len(store.saved) >= 2, "control: the turn was saved on the hit"
        user, answer = store.saved[0], store.saved[1]
        content = user["content"]
        assert answer["content"] == "Cached: at the venue."
        assert (user["origin"], answer["origin"]) == ("typed", "assistant")
        assert user["segments"] == [[0, len(_QUESTION), "typed"],
                                    [len(content) - len(_DOCUMENT), len(content), "document"]]
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT16 -- the mirror is fed through the origin read (supersedes XW4)
# ---------------------------------------------------------------------------
def test_ot16_after_the_turn_the_librarian_is_offered_the_conversation_through_the_origin_read():
    mod, scripted, store, librarian, restore = _executor()
    try:
        _drive(mod.Executor().execute(_QUESTION, _routing(), refine=False, conversation_id="conv-1"))
        assert store.saved, "control: the turn was saved"
        assert len(librarian.curate_calls) == 1
        cid, messages = librarian.curate_calls[0]
        assert cid == "conv-1"
        assert messages == store.get_mirror_messages("conv-1")
        assert [m["origin"] for m in messages] == ["typed", "assistant"]
        assert messages[-1]["content"] == "At the venue."
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT17 to OT20 -- the agentic pipelines (supersede the pipeline-persistence
# suite's four contracts)
# ---------------------------------------------------------------------------
_AGENT_CONV = "conv-agentic"


class _AgentStore:
    """Mirrors the store's keyword surface, origin included."""

    def __init__(self):
        self.saved = []

    def add_message(self, conv_id=None, role=None, content=None, model=None, metadata=None, *, origin="legacy",
                    segments=()):
        self.saved.append({"conv_id": conv_id, "role": role, "content": content, "model": model,
                           "origin": origin, "segments": list(segments)})
        return {"ok": True}


class _PipelineResult:
    def __init__(self, text):
        self.final_response = text
        self.model = "fake-model"
        self.tier_index = 0
        self.tier_name = "fast"
        self.score = 0.9
        self.total_latency_ms = 1.0
        self.draft_accepted = True
        self.iterations = 1
        self.convergence_score = 1.0


class _Cascade:
    enabled = True

    def cascade(self, query, task_type=None):
        return _PipelineResult("cascade answer")


class _Speculative:
    enabled = True

    def generate(self, query, task_type=None):
        return _PipelineResult("speculative answer")


class _Reasoner:
    """Streams like Executor.execute and, like a persist=False call, saves nothing."""

    def __init__(self):
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        yield "reasoning part. "
        yield "more reasoning."


class _ToolResult:
    def __init__(self):
        self.tool_calls = ["write_file"]
        self.response = "TOOL_OUTPUT_BLOCK"


class _Tools:
    def should_use_tools(self, message, model):
        return True

    def execute_with_tools(self, **kwargs):
        return _ToolResult()


def _agentic(store):
    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = store
    loaded, restore = isolate(
        targets={"opti_oignon.agentic_executor": source("agentic_executor.py")},
        seeded={"opti_oignon.conversation": conv},
    )
    return loaded["opti_oignon.agentic_executor"], restore


def _agent_routing():
    return SimpleNamespace(model="m", task_type=None)


def _quiet(agent):
    agent._get_conversation_context = lambda cid: []
    agent.get_tool_history = lambda cid: []
    agent._record_tool_calls = lambda cid, tc: None
    agent._emit_tool_call = lambda tc: None
    return agent


def _agent_origins(store):
    return [(m["role"], m["origin"], m["segments"]) for m in store.saved]


def test_ot17_the_cascading_pipeline_saves_a_typed_question_and_a_tool_flagged_answer():
    store = _AgentStore()
    mod, restore = _agentic(store)
    try:
        mod.CASCADING_INFERENCE_AVAILABLE = True
        agent = mod.AgenticExecutor(cascading_inference=_Cascade())
        list(agent._execute_cascading_pipeline("q", _agent_routing(), _AGENT_CONV, None))
    finally:
        restore()
    assert len(store.saved) == 2, "the cascading pipeline persists user and assistant"
    assert store.saved[1]["content"] == "cascade answer" and store.saved[1]["model"] == "fake-model"
    assert _agent_origins(store) == [("user", "typed", []), ("assistant", "assistant+tool", [])]


def test_ot18_the_speculative_pipeline_saves_a_typed_question_and_a_tool_flagged_answer():
    store = _AgentStore()
    mod, restore = _agentic(store)
    try:
        mod.SPECULATIVE_GENERATION_AVAILABLE = True
        agent = mod.AgenticExecutor(speculative_generator=_Speculative())
        list(agent._execute_speculative_pipeline("q", _agent_routing(), _AGENT_CONV, None))
    finally:
        restore()
    assert len(store.saved) == 2, "the speculative pipeline persists user and assistant"
    assert store.saved[1]["content"] == "speculative answer" and store.saved[1]["model"] == "fake-model"
    assert _agent_origins(store) == [("user", "typed", []), ("assistant", "assistant+tool", [])]


def test_ot19_the_think_and_tools_pipeline_saves_a_typed_question_and_a_tool_flagged_answer():
    store = _AgentStore()
    mod, restore = _agentic(store)
    try:
        mod.TOOL_EXECUTOR_AVAILABLE = True
        reasoner = _Reasoner()
        agent = _quiet(mod.AgenticExecutor(executor=reasoner, tool_executor=_Tools()))
        list(agent._execute_think_tools_pipeline("do a thing", _agent_routing(), _AGENT_CONV, None))
    finally:
        restore()
    assert reasoner.calls and reasoner.calls[0].get("persist") is False, "the executor's own save is suppressed"
    assert len(store.saved) == 2, "user and assistant persisted exactly once"
    answer = store.saved[1]["content"]
    assert "reasoning" in answer and "TOOL_OUTPUT_BLOCK" in answer, "reasoning and tool output both persist"
    assert _agent_origins(store) == [("user", "typed", []), ("assistant", "assistant+tool", [])]


def test_ot20_the_tools_pipeline_saves_a_typed_question_and_a_tool_flagged_answer():
    store = _AgentStore()
    mod, restore = _agentic(store)
    try:
        mod.TOOL_EXECUTOR_AVAILABLE = True
        agent = _quiet(mod.AgenticExecutor(tool_executor=_Tools()))
        list(agent._execute_tools_pipeline("q", _agent_routing(), _AGENT_CONV, None))
    finally:
        restore()
    assert len(store.saved) == 2, "the tools pipeline persists user and assistant"
    assert store.saved[1]["content"] == "TOOL_OUTPUT_BLOCK"
    assert _agent_origins(store) == [("user", "typed", []), ("assistant", "assistant+tool", [])]


# ---------------------------------------------------------------------------
# OT21 -- the coding agent
# ---------------------------------------------------------------------------
class _CodingStore:
    def __init__(self):
        self.saved = []

    def add_message(self, conv_id, role, content, model=None, metadata=None, *, origin="legacy", segments=()):
        self.saved.append({"conv_id": conv_id, "role": role, "content": content, "origin": origin,
                           "segments": list(segments)})


def test_ot21_the_coding_agent_saves_a_typed_question_and_a_tool_flagged_answer():
    store = _CodingStore()
    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = store
    loaded, restore = isolate(
        targets={"opti_oignon.chat_coding_agent": source("chat_coding_agent.py")},
        seeded={"opti_oignon.conversation": conv},
    )
    try:
        mod = loaded["opti_oignon.chat_coding_agent"]
        assert mod.CONVERSATION_AVAILABLE is True, "control: the agent reaches the store"
        session = object.__new__(mod.ChatCodingSession)
        session._conversation_id = "conv-code"
        session._save_turn_to_conversation("Fix the failing test.", "Fixed: the import was wrong.", "coder:7b")
    finally:
        restore()
    assert len(store.saved) == 2, "control: the coding turn was saved"
    assert [(m["role"], m["origin"], m["segments"]) for m in store.saved] == [
        ("user", "typed", []), ("assistant", "assistant+tool", [])]


# ---------------------------------------------------------------------------
# OT22 -- the branch store gains the columns
# ---------------------------------------------------------------------------
_OLD_BRANCH_SCHEMA = """
CREATE TABLE branches (
    branch_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, parent_branch_id TEXT,
    fork_message_id INTEGER NOT NULL, name TEXT NOT NULL, color TEXT NOT NULL DEFAULT '#B59E7D',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, metadata TEXT DEFAULT '{}'
);
CREATE TABLE branch_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, branch_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
    role TEXT NOT NULL, content TEXT NOT NULL, timestamp TEXT NOT NULL, token_estimate INTEGER DEFAULT 0,
    model TEXT, metadata TEXT DEFAULT '{}',
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id) ON DELETE CASCADE
);
"""


def test_ot22_a_branch_store_gains_the_origin_columns_and_its_older_rows_read_as_legacy(tmp_path):
    cb, restore = _branches(tmp_path)
    try:
        old = tmp_path / "branches-old.db"
        conn = _raw(old)
        conn.executescript(_OLD_BRANCH_SCHEMA)
        conn.execute("INSERT INTO branches (branch_id, conversation_id, fork_message_id, name, created_at, updated_at)"
                     " VALUES ('b1', 'c1', 1, 'b', 't0', 't0')")
        conn.execute("INSERT INTO branch_messages (branch_id, conversation_id, role, content, timestamp)"
                     " VALUES ('b1', 'c1', 'user', 'older words', 't1')")
        conn.commit()
        conn.close()
        assert "origin" not in _columns(old, "branch_messages"), "control: the older file has no origin"
        mgr = _branch_manager(cb, old)
        assert [m.content for m in mgr.get_branch_only_messages("b1")] == ["older words"]
        cols = _columns(old, "branch_messages")
        assert cols["origin"][4] == "'legacy'" and cols["segments"][4] == "'[]'"
        conn = _raw(old)
        try:
            assert conn.execute("SELECT origin, segments FROM branch_messages").fetchall() == [("legacy", "[]")]
        finally:
            conn.close()
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT23 -- posted by the user: typed; anything else: legacy
# ---------------------------------------------------------------------------
def test_ot23_a_message_the_user_posts_to_a_branch_is_typed_and_any_other_role_legacy(tmp_path):
    cb, restore = _branches(tmp_path)
    try:
        assert cb.posted_origin("user") == "typed"
        for role in ("assistant", "system", "tool", "", None):
            assert cb.posted_origin(role) == "legacy", f"a {role!r} message posted by a client is legacy"
        path = tmp_path / "b.db"
        mgr = _branch_manager(cb, path)
        branch = mgr.fork("c1", fork_message_id=1, name="b")
        assert mgr.add_branch_message(branch.branch_id, "c1", "user", "Mine.", origin=cb.posted_origin("user"))
        with pytest.raises(cb.OriginError):
            mgr.add_branch_message(branch.branch_id, "c1", "assistant", "Not mine.", origin="typed")
        conn = _raw(path)
        try:
            rows = conn.execute("SELECT role, origin FROM branch_messages ORDER BY id").fetchall()
        finally:
            conn.close()
        assert rows == [("user", "typed")], "the refused message wrote nothing"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT24 -- a merge copies origin and segments
# ---------------------------------------------------------------------------
def test_ot24_a_merge_copies_each_message_with_its_origin_and_segments(tmp_path):
    cb, restore = _branches(tmp_path)
    try:
        path = tmp_path / "b.db"
        mgr = _branch_manager(cb, path)
        source_branch = mgr.fork("c1", fork_message_id=1, name="from")
        target_branch = mgr.fork("c1", fork_message_id=1, name="to")
        content, segments = _with_document()
        mgr.add_branch_message(source_branch.branch_id, "c1", "user", content, origin="typed", segments=segments)
        mgr.add_branch_message(source_branch.branch_id, "c1", "assistant", "Noted.", origin="assistant+web")
        mgr.add_branch_message(source_branch.branch_id, "c1", "system", "Be brief.")
        merged = mgr.merge_messages(source_branch.branch_id, target_branch.branch_id)
        assert len(merged) == 3, "control: the three messages were copied"
        conn = _raw(path)
        try:
            rows = conn.execute(
                "SELECT role, origin, segments FROM branch_messages WHERE branch_id = ? ORDER BY id",
                (target_branch.branch_id,),
            ).fetchall()
        finally:
            conn.close()
        assert [(r[0], r[1], json.loads(r[2])) for r in rows] == [
            ("user", "typed", segments), ("assistant", "assistant+web", []), ("system", "legacy", [])]
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT25 -- the census of message write sites
# ---------------------------------------------------------------------------
_NAMED = {
    "opti_oignon/executor.py": 4,
    "opti_oignon/agentic_executor.py": 2,
    "opti_oignon/chat_coding_agent.py": 2,
    "opti_oignon/conversation_branches.py": 1,
    "opti_oignon/api/routes_branches.py": 1,
}
_DEFAULTED = {"opti_oignon/performance_benchmark.py", "opti_oignon/agent_eval/fidelity.py"}
_WRITERS = ("add_message", "add_branch_message")


def _is_main_guard(test):
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__")


def _census():
    named, defaulted, forwarded, selftest, inserts = Counter(), Counter(), Counter(), Counter(), []
    for root, dirs, files in os.walk(REPO / "opti_oignon"):
        dirs[:] = sorted(d for d in dirs if d not in ("data", "__pycache__"))
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = Path(root) / name
            rel = path.relative_to(REPO).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            guarded = set()
            for node in tree.body:
                if isinstance(node, ast.If) and _is_main_guard(node.test):
                    guarded.update(id(sub) for sub in ast.walk(node))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                    if called not in _WRITERS:
                        continue
                    if id(node) in guarded:
                        selftest[rel] += 1
                    elif any(k.arg is None for k in node.keywords):
                        forwarded[rel] += 1
                    elif any(k.arg == "origin" for k in node.keywords):
                        named[rel] += 1
                    else:
                        defaulted[rel] += 1
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    text = " ".join(node.value.split())
                    if "INSERT INTO messages" in text or "INSERT INTO branch_messages" in text:
                        inserts.append((rel, node.lineno, "origin" in text))
    return named, defaulted, forwarded, selftest, inserts


def test_ot25_every_message_write_site_in_the_package_is_accounted_for():
    named, defaulted, forwarded, selftest, inserts = _census()
    assert sum(named.values()) >= 10, f"the census finds the sites that name an origin: {dict(named)}"
    assert dict(named) == _NAMED, f"the sites that name an origin: {dict(named)}"
    assert set(defaulted) == _DEFAULTED, f"only benches and evaluations keep the legacy default: {dict(defaulted)}"
    assert dict(forwarded) == {"opti_oignon/conversation.py": 1}, "one envelope forwards its keywords"
    assert len(inserts) >= 5, f"the census finds the insert statements: {inserts}"
    assert all(has for _rel, _line, has in inserts), (
        f"every insert into a message table names the origin: {[i for i in inserts if not i[2]]}"
    )


# ---------------------------------------------------------------------------
# OT26 -- the mirror carries the labels
# ---------------------------------------------------------------------------
def test_ot26_the_mirror_carries_origin_and_segments_into_the_flesh():
    lib, loaded, restore = _librarian()
    try:
        content, segments = _with_document()
        state = lib.OnionState()
        added = state.mirror([
            {"role": "user", "content": content, "origin": "typed", "segments": segments},
            {"role": "assistant", "content": "At the venue.", "origin": "assistant+web", "segments": []},
            {"role": "user", "content": "Older words."},
        ])
        assert added == 3, "control: three turns mirrored"
        turns = state.flesh.turns()
        assert [(t["turn_id"], t["role"], t["origin"], t["segments"]) for t in turns] == [
            ("t0001", "user", "typed", segments),
            ("t0002", "assistant", "assistant+web", []),
            ("t0003", "user", "legacy", []),
        ]
        assert turns[0]["text"] == content
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT27 -- a declaration outside the grammar: legacy, said without the text
# ---------------------------------------------------------------------------
def test_ot27_a_declaration_outside_the_grammar_is_mirrored_legacy_and_said_without_the_text(caplog):
    lib, loaded, restore = _librarian()
    try:
        state = lib.OnionState()
        with caplog.at_level(logging.WARNING):
            added = state.mirror([
                {"role": "assistant", "content": "SECRET-ONE is mine.", "origin": "typed"},
                {"role": "user", "content": "SECRET-TWO here.", "origin": "typed", "segments": [[0, 99, "typed"]]},
                {"role": "user", "content": "SECRET-THREE.", "origin": "typed+web"},
                {"role": "user", "content": "SECRET-FOUR.", "origin": "typed", "segments": None},
                {"role": "user", "content": "Fine words.", "origin": "typed"},
            ])
        assert added == 5
        turns = state.flesh.turns()
        assert [(t["origin"], t["segments"]) for t in turns[:4]] == [("legacy", [])] * 4
        assert (turns[4]["origin"], turns[4]["segments"]) == ("typed", []), "control: a good declaration stands"
        said = "\n".join(r.getMessage() for r in caplog.records)
        for turn_id in ("t0001", "t0002", "t0003", "t0004"):
            assert turn_id in said, f"the refusal for {turn_id} is said"
        assert "t0005" not in said
        assert "SECRET" not in said, "the log names the turn, never its text"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT28 -- the summariser never sees an origin
# ---------------------------------------------------------------------------
class _Backend:
    def __init__(self):
        self.calls = []

    def generate(self, model, messages, options=None, keep_alive="30m", think=False, images=None):
        self.calls.append(messages)
        return SimpleNamespace(content="summary")


def test_ot28_the_summariser_is_handed_turn_role_and_text_never_an_origin():
    lib, loaded, restore = _librarian()
    try:
        content, segments = _with_document()
        backend = _Backend()
        cfg = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=4,
                                  temperature=0.1, num_predict=128)
        summarize = lib.registry_summarizer(cfg, resolve=lambda model: backend)
        summarize([
            {"turn_id": "t0001", "role": "user", "text": content, "origin": "typed", "segments": segments},
            {"turn_id": "t0002", "role": "assistant", "text": "Noted.", "origin": "assistant+web", "segments": []},
        ])
        assert backend.calls, "control: the summariser was asked"
        quoted = [m for m in backend.calls[0] if m["role"] == "user"][0]["content"]
        lines = [json.loads(line) for line in quoted.splitlines()]
        assert len(lines) == 2
        assert all(set(line) == {"turn", "role", "text"} for line in lines), "turn, role and text, nothing else"
        assert "assistant+web" not in quoted and "segments" not in quoted
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT29 -- labels survive eviction: Cellar, receipt, peel
# ---------------------------------------------------------------------------
def test_ot29_an_evicted_span_keeps_its_labels_and_its_receipt_and_peel_answer_for_their_union():
    loaded, restore = _memory("probes", "receipts", "peels")
    try:
        receipts = loaded["opti_oignon.memory.receipts"]
        peels = loaded["opti_oignon.memory.peels"]
        content, segments = _with_document()
        flesh = receipts.Flesh([
            {"turn_id": "t0001", "role": "user", "text": content, "origin": "typed", "segments": segments},
            {"turn_id": "t0002", "role": "assistant", "text": "Alice meets Bob at Contoso.", "origin": "assistant+web",
             "segments": []},
            {"turn_id": "t0003", "role": "user", "text": "Later words.", "origin": "typed", "segments": []},
        ])
        cellar, ledger, tree = receipts.Cellar(), receipts.ReceiptLedger(), peels.PeelTree()
        gate = peels.Gate(decision_threshold=0.0, episodic_threshold=0.0, span_turns=2)
        outcome = peels.evict_gated(flesh=flesh, cellar=cellar, ledger=ledger, tree=tree, gate=gate,
                                    summarize=lambda turns: " ".join(t["text"] for t in turns))
        assert outcome.evicted, f"control: the span left the Flesh ({outcome.reason})"
        span = cellar.get(outcome.receipt.key)
        assert [(t["origin"], t["segments"]) for t in span] == [("typed", segments), ("assistant+web", [])]
        union = ("assistant+web", "document", "typed")
        assert len(ledger.origins(outcome.receipt.key, cellar)) >= 3
        assert ledger.origins(outcome.receipt.key, cellar) == union
        assert peels.peel_origins(outcome.peel, cellar) == union
        assert receipts.span_origins([{"turn_id": "t9", "text": "x"}]) == ("legacy",), "a span that declares nothing"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT30 -- each probe knows its piece
# ---------------------------------------------------------------------------
def test_ot30_a_probe_carries_the_base_of_its_piece_and_the_role_of_its_turn():
    loaded, restore = _memory("probes")
    try:
        probes = loaded["opti_oignon.memory.probes"]
        content, segments = _with_document()
        drawn = probes.generate_probes([
            {"turn_id": "t1", "role": "user", "text": content, "origin": "typed", "segments": segments},
            {"turn_id": "t2", "role": "assistant", "text": "Bob confirmed Contoso.", "origin": "assistant+web"},
        ])
        found = {(p.turn_id, p.answer, p.origin, p.role) for p in drawn}
        assert len(found) >= 6, f"the span is rich: {sorted(found)}"
        for expected in (("t1", "Alice", "typed", "user"), ("t1", "Bob", "typed", "user"),
                         ("t1", "2024-03-15", "typed", "user"), ("t1", "Contoso", "document", "user"),
                         ("t1", "40", "document", "user"), ("t2", "Contoso", "assistant+web", "assistant")):
            assert expected in found, f"{expected} is drawn with its piece's base and its turn's role"
        answers = {p.answer for p in drawn}
        assert "Document" not in answers and "provided" not in answers, "the separator is never drawn from"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OT31 -- the native core draws the same, piece by piece
# ---------------------------------------------------------------------------
class _Counting:
    def __init__(self, core):
        self.core, self.draws = core, 0

    def __getattr__(self, name):
        return getattr(self.core, name)

    def probe_generate(self, *args, **kwargs):
        self.draws += 1
        return self.core.probe_generate(*args, **kwargs)


def test_ot31_the_native_core_draws_the_same_probes_piece_by_piece():
    content, segments = _with_document()
    span = [
        {"turn_id": "t1", "role": "user", "text": content, "origin": "typed", "segments": segments},
        {"turn_id": "t2", "role": "assistant", "text": "Bob confirmed Contoso.", "origin": "assistant+web"},
    ]
    loaded, restore = _memory("probes", native=False)
    try:
        reference = loaded["opti_oignon.memory.probes"].generate_probes(span)
    finally:
        restore()
    loaded, restore = _memory("probes", native=True)
    try:
        probes = loaded["opti_oignon.memory.probes"]
        core = loaded["opti_oignon.native"].load()
        assert core is not None, "the native core is built on this machine"
        counting = _Counting(core)
        probes._native = lambda: counting
        native = probes.generate_probes(span)
    finally:
        restore()
    assert counting.draws == 1, "the native core drew the span"
    assert len(reference) >= 6
    assert {p.origin for p in native} == {"typed", "document", "assistant+web"}, "drawn piece by piece"
    assert "Document" not in {p.answer for p in native}, "the separator is never drawn from"
    # Two windows load two modules, so two Probe classes: compare the fields.
    assert [dataclasses.astuple(p) for p in native] == [dataclasses.astuple(p) for p in reference], (
        "probe for probe, origin and role included"
    )
