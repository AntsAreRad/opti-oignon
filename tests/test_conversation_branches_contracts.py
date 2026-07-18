#!/usr/bin/env python3
"""What branching promises about the forks it keeps and the history it rebuilds.

Branching runs on its own SQLite file, and the whole store answers to two
demands at once. Every write is DATA: a branch name, a colour, a conversation
id, a merged-message tag -- a value carrying SQL metacharacters has to
round-trip as a string, never execute, so every statement is parameterised.
And the store cascades on its own key: deleting a branch takes its messages
with it, and re-parents the branch's children onto the deleted branch's own
parent so the tree never dangles.

The manager talks to disk through one seam. Every connection is opened by the
module's ``safe_connect`` and every connection turns foreign keys on, so the
cascade the schema declares is the cascade that actually runs. When the seam is
gone the manager cannot even initialise, which is the honest failure: the
availability flag flips and the module-level singleton becomes None rather than
handing back a half-built store.

Reads degrade rather than raise. A listing, a lookup, a stats roll-up on a
branch that has no rows comes back empty or zeroed, never as an exception,
because a read that throws takes a UI panel down with it. Rebuilding a branch's
full history is a pure splice once the shared prefix is in hand: the caller may
hand in the main-conversation messages, and the shared part is exactly those up
to and including the fork point -- a message without an id is dropped, and the
branch's own messages are appended after. Reading that shared prefix from the
separate conversation store is the same read fenced by an id bound, and a
metadata blob that will not parse decodes to an empty mapping rather than
sinking the row.

One promise spent a stretch of its life BROKEN and these contracts said so out
loud instead of freezing the defect: the dynamic-update column allowlist
rejected the always-present timestamp column, so a rename, a recolour or a
metadata merge was swallowed whole. The allowlist now admits the timestamp
column, the two update contracts assert persistence as plain assertions, and
the expected-failure markers that stood witness are gone -- promoted, exactly
as their reason strings demanded.

The merge tag keeps provenance through ``original_message_id`` -- note that the
timestamps of merged messages are NOT preserved, whatever the config key of
that name suggests; that key is inert and its behaviour is pinned here as the
inertia it is. Config loading fills the default schema, applies an override
file when present, drops a foreign TOP-LEVEL section, and -- pinned exactly --
does NOT drop a foreign sub-key inside a known section.

Loaded through the shared isolation window. The connection factory, the data
directory, and the token estimator are the only project seams the store
reaches; each is seeded or blocked, and the token path's absence is proven, so
no real backend is ever touched by these contracts.
"""

import json
import sqlite3
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.conversation_branches"

# A token value the fallback (length // 4) can never produce for our inputs, so
# a contract that sees it knows the estimator delegated to the seeded seam.
_CM_TOKENS = 777


def _default_connect(path, **kwargs):
    return sqlite3.connect(str(path), **kwargs)


def _load(data_dir, *, connect=_default_connect, seed_cm=False):
    """Load the branching module in isolation.

    data_dir  -- seeded as ``config.DATA_DIR``. Contains the module-level
                 singleton's own database and is where the separate
                 conversation store is looked up.
    connect   -- the stand-in ``safe_connect``. The default is a plain sqlite
                 connection; a caller can pass a counting or a raising factory
                 to pin the seam itself.
    seed_cm   -- when true, a context-manager stand-in whose ``estimate_tokens``
                 returns ``_CM_TOKENS`` is seeded so the delegated token path
                 runs; when false the name is blocked and the length fallback
                 runs instead.
    """
    seeded = {}
    blocked = []

    du = types.ModuleType("opti_oignon.db_utils")
    du.safe_connect = connect
    seeded["opti_oignon.db_utils"] = du

    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = Path(data_dir)
    seeded["opti_oignon.config"] = cfg

    if seed_cm:
        cm = types.ModuleType("opti_oignon.context_manager")
        cm.estimate_tokens = lambda text, model=None: _CM_TOKENS
        seeded["opti_oignon.context_manager"] = cm
    else:
        blocked.append("opti_oignon.context_manager")

    loaded, restore = isolate(
        targets={_TARGET: source("conversation_branches.py")},
        blocked=blocked,
        seeded=seeded,
    )
    return loaded[_TARGET], restore


# --- fixtures -------------------------------------------------------------
# A manager is always built with an explicit database path (distinct from the
# singleton's) and an explicit config derived from the module defaults, so a
# contract never depends on whatever the on-disk config file happens to hold.

def _cfg(cb, **overrides):
    base = json.loads(json.dumps(cb._DEFAULT_CONFIG))
    for section, values in overrides.items():
        base.setdefault(section, {})
        base[section].update(values)
    return base


def _mgr(cb, tmp_path, name="t.db", **overrides):
    return cb.ConversationBranchManager(
        db_path=Path(tmp_path) / name,
        config=_cfg(cb, **overrides) if overrides else None,
    )


def _seed_conversation_db(tmp_path, rows):
    """Write a minimal conversation store the shared-prefix reads consult."""
    conv = Path(tmp_path) / "conversations.db"
    c = sqlite3.connect(str(conv))
    c.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id TEXT, "
        "role TEXT, content TEXT, timestamp TEXT, token_estimate INTEGER, "
        "model TEXT, metadata TEXT)"
    )
    c.executemany(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    c.commit()
    c.close()
    return conv


# =========================================================================
# Helpers and data classes
# =========================================================================

def test_b1_dataclasses_roundtrip_to_dict(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        br = cb.Branch("bid", "cid", None, 3, "n", "#111111", "t0", "t1",
                       {"k": [1, 2]})
        assert br.to_dict() == {
            "branch_id": "bid", "conversation_id": "cid",
            "parent_branch_id": None, "fork_message_id": 3, "name": "n",
            "color": "#111111", "created_at": "t0", "updated_at": "t1",
            "metadata": {"k": [1, 2]},
        }
        msg = cb.BranchMessage(5, "bid", "cid", "user", "hi", "ts",
                               token_estimate=9, model="m", metadata={"a": 1})
        d = msg.to_dict()
        assert d["id"] == 5 and d["metadata"] == {"a": 1} and d["model"] == "m"

        cmp = cb.BranchComparison("a", "b", "A", "B", [{"id": 1}], [], [], 2)
        assert cmp.to_dict()["shared_messages"] == [{"id": 1}]
        assert cmp.to_dict()["fork_message_id"] == 2
    finally:
        restore()


def test_b2_tree_node_to_dict_is_recursive(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        child = cb.BranchTreeNode("c", "child", "#222222", 4, 2, "m", "t")
        root = cb.BranchTreeNode(None, "Main", "#B59E7D", None, 0, None, None,
                                 children=[child])
        d = root.to_dict()
        assert d["branch_id"] is None and d["name"] == "Main"
        assert len(d["children"]) == 1
        assert d["children"][0]["branch_id"] == "c", (
            "to_dict must recurse into children, not leave node objects"
        )
    finally:
        restore()


def test_b3_message_to_ollama_format_is_role_and_content_only(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        msg = cb.BranchMessage(1, "b", "c", "assistant", "text", "ts",
                               token_estimate=9, model="llama", metadata={"x": 1})
        assert msg.to_ollama_format() == {"role": "assistant", "content": "text"}, (
            "the ollama view must carry only role and content -- no id, model, "
            "timestamp or metadata leaks into what is sent to the backend"
        )
    finally:
        restore()


def test_b4_estimate_tokens_empty_fallback_and_delegated(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        assert cb._estimate_tokens("") == 0
        assert cb._HAS_CONTEXT_MANAGER is False
        assert cb._estimate_tokens("abcd") == 1, "no estimator: length over four"
    finally:
        restore()

    cb2, restore2 = _load(tmp_path, seed_cm=True)
    try:
        assert cb2._HAS_CONTEXT_MANAGER is True
        assert cb2._estimate_tokens("abcd") == _CM_TOKENS, (
            "with an estimator present the module must delegate, not fall back"
        )
    finally:
        restore2()


def test_b5_load_config_returns_default_schema_without_override(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        empty = Path(tmp_path) / "no_config_here"
        empty.mkdir()
        saved = cb.__file__
        cb.__file__ = str(empty / "conversation_branches.py")
        try:
            loaded = cb._load_config()
        finally:
            cb.__file__ = saved
        assert loaded == cb._DEFAULT_CONFIG, (
            "with no override file present, loading yields the default schema "
            "verbatim -- the fallback path, not a partial merge"
        )
    finally:
        restore()


def test_b6_load_config_overrides_and_filters_top_level_only(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        cdir = Path(tmp_path) / "cfg"
        (cdir / "config").mkdir(parents=True)
        (cdir / "config" / "branches.yaml").write_text(
            "branches:\n  max_per_conversation: 7\n  unknown_sub: 42\n"
            "totally_foreign:\n  a: 1\n",
            encoding="utf-8",
        )
        saved = cb.__file__
        cb.__file__ = str(cdir / "conversation_branches.py")
        try:
            merged = cb._load_config()
        finally:
            cb.__file__ = saved
        assert merged["branches"]["max_per_conversation"] == 7, "override applies"
        assert "totally_foreign" not in merged, (
            "a foreign top-level section is dropped: only default sections survive"
        )
        assert merged["branches"]["unknown_sub"] == 42, (
            "a foreign sub-key inside a known section is NOT filtered -- pinned "
            "exactly, because the merge shallow-updates the section dict"
        )
        assert merged["merge"]["max_messages_per_merge"] == 200, (
            "an untouched default section is preserved intact"
        )
    finally:
        restore()


# =========================================================================
# Connection seam, metadata decoding, injection safety
# =========================================================================

def test_b7_all_db_access_flows_through_the_connection_seam(tmp_path):
    calls = {"n": 0}

    def counting(path, **kwargs):
        calls["n"] += 1
        return sqlite3.connect(str(path), **kwargs)

    cb, restore = _load(tmp_path, connect=counting)
    try:
        m = _mgr(cb, tmp_path)
        calls["n"] = 0
        b = m.fork("c", 1)
        m.list_branches("c")
        m.get_branch(b.branch_id)
        m.add_branch_message(b.branch_id, "c", "user", "x")
        m.delete_branch(b.branch_id)
        assert calls["n"] > 0, (
            "every database touch must be opened through safe_connect; a path "
            "that reaches sqlite directly would bypass the encrypted-connection "
            "seam and never increment this counter"
        )
    finally:
        restore()


def test_b8_row_converters_tolerate_unparseable_metadata(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        b = m.fork("c", 1)
        # Corrupt the stored metadata directly, then read it back.
        raw = sqlite3.connect(str(Path(tmp_path) / "t.db"))
        raw.execute("UPDATE branches SET metadata = ? WHERE branch_id = ?",
                    ("this is not json", b.branch_id))
        raw.commit()
        raw.close()
        got = m.get_branch(b.branch_id)
        assert got is not None and got.metadata == {}, (
            "a metadata column that will not parse must decode to an empty "
            "mapping, never raise out of the row converter"
        )

        m.add_branch_message(b.branch_id, "c", "user", "hi")
        raw = sqlite3.connect(str(Path(tmp_path) / "t.db"))
        raw.execute("UPDATE branch_messages SET metadata = ? WHERE branch_id = ?",
                    ("{bad", b.branch_id))
        raw.commit()
        raw.close()
        msgs = m.get_branch_only_messages(b.branch_id)
        assert msgs and msgs[0].metadata == {}
    finally:
        restore()


def test_b9_names_and_ids_with_sql_metacharacters_roundtrip(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        evil_conv = "c'); DROP TABLE branches;--"
        evil_name = 'nasty"; DELETE FROM branches;--'
        b = m.fork(evil_conv, 1, name=evil_name)
        assert b is not None and b.name == evil_name
        listed = m.list_branches(evil_conv)
        assert [x.name for x in listed] == [evil_name], (
            "a conversation id and a branch name carrying SQL metacharacters "
            "must round-trip as data; that only holds under parameterisation"
        )
    finally:
        restore()


# =========================================================================
# Fork / list / get
# =========================================================================

def test_b10_fork_persists_fields_and_returns_branch(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        b = m.fork("conv", 5, name="explore", color="#ABCabc",
                   metadata={"why": "test"})
        assert b is not None
        assert (b.conversation_id, b.fork_message_id, b.name, b.color) == (
            "conv", 5, "explore", "#ABCabc")
        assert b.metadata == {"why": "test"}
        again = m.get_branch(b.branch_id)
        assert again is not None and again.name == "explore"
        assert again.metadata == {"why": "test"}
    finally:
        restore()


def test_b11_fork_autofills_name_and_cycles_colour(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path, branches={
            "default_name_template": "Branch {n}",
            "color_palette": ["#AAA111", "#BBB222"],
        })
        first = m.fork("c", 1)
        second = m.fork("c", 2)
        assert first.name == "Branch 1" and second.name == "Branch 2", (
            "auto names count up: {n} is the existing count plus one"
        )
        assert first.color == "#AAA111" and second.color == "#BBB222", (
            "auto colours walk the palette by branch count modulo its length"
        )
    finally:
        restore()


def test_b12_fork_limit_blocks_and_zero_disables(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        capped = _mgr(cb, tmp_path, name="cap.db",
                      branches={"max_per_conversation": 2})
        made = [capped.fork("c", 1) for _ in range(3)]
        assert made[0] is not None and made[1] is not None
        assert made[2] is None, "the third fork past a limit of two is refused"

        unlimited = _mgr(cb, tmp_path, name="free.db",
                         branches={"max_per_conversation": 0})
        assert all(unlimited.fork("c", 1) is not None for _ in range(5)), (
            "a limit of zero disables the guard entirely"
        )
    finally:
        restore()


def test_b13_fork_rejects_nonexistent_parent(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        orphan = m.fork("c", 1, parent_branch_id="no-such-branch")
        assert orphan is None, (
            "foreign keys are on, so a fork naming a parent that does not exist "
            "violates the constraint and returns None rather than a dangling row"
        )
    finally:
        restore()


def test_b14_list_branches_orders_by_creation_and_empty_is_empty(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        assert m.list_branches("absent") == []
        a = m.fork("c", 1, name="a")
        b = m.fork("c", 2, name="b")
        c = m.fork("c", 3, name="c")
        names = [x.name for x in m.list_branches("c")]
        assert names == ["a", "b", "c"], "listing is ordered by creation time"
        assert {a.branch_id, b.branch_id, c.branch_id} == {
            x.branch_id for x in m.list_branches("c")}
    finally:
        restore()


def test_b15_get_branch_returns_object_or_none(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        assert m.get_branch("missing") is None
        b = m.fork("c", 1, name="here")
        got = m.get_branch(b.branch_id)
        assert got is not None and got.name == "here"
    finally:
        restore()


# =========================================================================
# Update -- INTENDED behaviour, currently broken (expected failure, strict)
# =========================================================================

def test_b16_update_branch_persists_rename_and_recolour(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        b = m.fork("c", 1, name="old", color="#000000")
        updated = m.update_branch(b.branch_id, name="new", color="#FFFFFF")
        assert updated is not None
        assert updated.name == "new" and updated.color == "#FFFFFF"
        assert m.get_branch(b.branch_id).name == "new"
    finally:
        restore()


def test_b17_update_branch_merges_metadata(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        b = m.fork("c", 1, metadata={"keep": 1})
        updated = m.update_branch(b.branch_id, metadata={"add": 2})
        assert updated is not None
        assert updated.metadata == {"keep": 1, "add": 2}
    finally:
        restore()


# =========================================================================
# Delete
# =========================================================================

def test_b18_delete_branch_missing_returns_false(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        assert m.delete_branch("nope") is False
    finally:
        restore()


def test_b19_delete_branch_reparents_children_to_grandparent(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        gp = m.fork("c", 1, name="gp")
        p = m.fork("c", 2, name="p", parent_branch_id=gp.branch_id)
        ch = m.fork("c", 3, name="ch", parent_branch_id=p.branch_id)
        m.add_branch_message(p.branch_id, "c", "user", "doomed")

        assert m.delete_branch(p.branch_id) is True
        assert m.get_branch(p.branch_id) is None
        child = m.get_branch(ch.branch_id)
        assert child is not None and child.parent_branch_id == gp.branch_id, (
            "a deleted branch's children re-parent onto its own parent, not to "
            "null -- the tree closes over the hole"
        )
        assert m.get_branch_only_messages(p.branch_id) == [], (
            "the deleted branch's messages are gone with it"
        )
    finally:
        restore()


def test_b20_delete_all_branches_counts_scopes_and_is_injection_safe(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        assert m.delete_all_branches("empty") == 0
        evil = "c'); DROP TABLE branches;--"
        m.fork(evil, 1)
        m.fork(evil, 2)
        m.fork("other", 1, name="survivor")
        n = m.delete_all_branches(evil)
        assert n == 2, "the return value is the number of branches removed"
        assert [x.name for x in m.list_branches("other")] == ["survivor"], (
            "a metacharacter-laden conversation id scopes the delete as data; "
            "sibling conversations are untouched"
        )
    finally:
        restore()


# =========================================================================
# Branch messages and shared-prefix reads
# =========================================================================

def test_b21_add_message_to_missing_branch_returns_none(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        assert m.add_branch_message("ghost", "c", "user", "hi") is None
    finally:
        restore()


def test_b22_add_message_inserts_bumps_branch_and_estimates(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        b = m.fork("c", 1)
        before = m.get_branch(b.branch_id).updated_at
        msg = m.add_branch_message(b.branch_id, "c", "assistant",
                                   "abcdefgh", model="llama")
        assert msg is not None and msg.id is not None
        assert msg.token_estimate == 2, "eight characters over four is two tokens"
        assert msg.model == "llama" and msg.role == "assistant"
        stored = m.get_branch_only_messages(b.branch_id)
        assert [x.id for x in stored] == [msg.id]
        assert m.get_branch(b.branch_id).updated_at >= before, (
            "adding a message stamps the branch's updated_at"
        )
    finally:
        restore()


def test_b23_branch_only_messages_order_and_empty(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        b = m.fork("c", 1)
        assert m.get_branch_only_messages(b.branch_id) == []
        m.add_branch_message(b.branch_id, "c", "user", "one")
        m.add_branch_message(b.branch_id, "c", "assistant", "two")
        m.add_branch_message(b.branch_id, "c", "user", "three")
        contents = [x.content for x in m.get_branch_only_messages(b.branch_id)]
        assert contents == ["one", "two", "three"], (
            "branch messages come back in chronological, then insertion, order"
        )
    finally:
        restore()


def test_b24_full_history_splices_shared_prefix_inclusively(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        assert m.get_branch_messages_full("c", "missing") == []
        b = m.fork("c", fork_message_id=2)
        m.add_branch_message(b.branch_id, "c", "user", "branch-first")
        main = [
            {"id": 1, "role": "user", "content": "m1"},
            {"id": 2, "role": "assistant", "content": "fork-point"},
            {"id": 3, "role": "user", "content": "after-fork"},
            {"role": "user", "content": "no-id"},
        ]
        full = m.get_branch_messages_full("c", b.branch_id, main_messages=main)
        assert [x.get("content") for x in full] == [
            "m1", "fork-point", "branch-first"], (
            "the shared prefix is main messages with id up to and INCLUDING the "
            "fork point; an id-less message is dropped; the branch's own "
            "messages follow"
        )
    finally:
        restore()


def test_b25_shared_prefix_reads_empty_without_conversation_store(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        # No conversations.db was seeded under DATA_DIR.
        assert m._get_main_messages_up_to("c", 10) == []
        assert m._get_main_messages_after("c", 0) == []
    finally:
        restore()


def test_b26_shared_prefix_reads_filter_and_decode(tmp_path):
    _seed_conversation_db(tmp_path, [
        (1, "c", "user", "b1", "t1", 1, None, '{"k": 1}'),
        (2, "c", "assistant", "b2", "t2", 2, "llama", None),
        (3, "c", "user", "b3", "t3", 3, None, "not-json"),
        (9, "other", "user", "x", "t9", 1, None, None),
    ])
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        up = m._get_main_messages_up_to("c", 2)
        assert [r["id"] for r in up] == [1, 2], "id up to the bound, this convo"
        assert up[0]["metadata"] == {"k": 1}, "well-formed metadata decodes"

        after = m._get_main_messages_after("c", 2)
        assert [r["id"] for r in after] == [3], "id strictly beyond the bound"
        assert after[0]["metadata"] == {}, (
            "a metadata blob that will not parse decodes to an empty mapping"
        )
    finally:
        restore()


# =========================================================================
# Stats, tree, comparison, merge, config, availability
# =========================================================================

def test_b27_branch_stats_empty_and_populated(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        b = m.fork("c", 1)
        assert m.get_branch_stats(b.branch_id) == {
            "message_count": 0, "last_activity": None,
            "total_tokens": 0, "last_model": None,
        }
        m.add_branch_message(b.branch_id, "c", "user", "aaaa", model=None)
        m.add_branch_message(b.branch_id, "c", "assistant", "bbbbbbbb",
                             model="llama3")
        stats = m.get_branch_stats(b.branch_id)
        assert stats["message_count"] == 2
        assert stats["total_tokens"] == 3, "one plus two tokens by length"
        assert stats["last_model"] == "llama3", (
            "last model is the most recent message that named one"
        )
    finally:
        restore()


def test_b28_branch_tree_roots_at_main_and_nests(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        top = m.fork("c", 1, name="top")
        child = m.fork("c", 2, name="child", parent_branch_id=top.branch_id)
        m.add_branch_message(child.branch_id, "c", "user", "hi")

        tree = m.get_branch_tree("c")
        assert tree.branch_id is None and tree.name == "Main"
        assert [n.name for n in tree.children] == ["top"], (
            "a branch forked from main hangs off the root"
        )
        top_node = tree.children[0]
        assert [n.name for n in top_node.children] == ["child"], (
            "a branch forked from another branch nests under it"
        )
        assert top_node.children[0].message_count == 1, (
            "node message counts are sourced from each branch's stats"
        )
    finally:
        restore()


def test_b29_compare_branches_fork_points_and_shared_slice(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        a = m.fork("c", 5, name="A")
        b = m.fork("c", 8, name="B")
        assert m.compare_branches("c", None, None) is None, (
            "two mains have nothing to compare"
        )
        assert m.compare_branches("c", "ghost", b.branch_id) is None, (
            "a missing branch aborts the comparison"
        )
        main = [{"id": i, "role": "user", "content": f"m{i}"} for i in range(1, 10)]
        cmp = m.compare_branches("c", a.branch_id, b.branch_id, main_messages=main)
        assert cmp is not None and cmp.fork_message_id == 5, (
            "the common fork is the earlier of the two fork points"
        )
        assert [x["id"] for x in cmp.shared_messages] == [1, 2, 3, 4, 5], (
            "shared messages are the main prefix up to the common fork"
        )
    finally:
        restore()


def test_b30_merge_filters_caps_tags_and_needs_target(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path, merge={"max_messages_per_merge": 2})
        src = m.fork("c", 1, name="src")
        picked = [m.add_branch_message(src.branch_id, "c", "user", f"s{i}")
                  for i in range(4)]

        dst = m.fork("c", 1, name="dst")
        merged = m.merge_messages(src.branch_id, dst.branch_id)
        assert len(merged) == 2, "the merge is capped at the configured maximum"
        assert all("merged_from" in x.metadata for x in merged)
        assert all(x.metadata.get("original_message_id") is not None
                   for x in merged), "provenance is tagged onto every copy"

        one = m.fork("c", 1, name="one")
        chosen = m.merge_messages(src.branch_id, one.branch_id,
                                  message_ids=[picked[0].id])
        assert [x.content for x in chosen] == ["s0"], (
            "an explicit id list selects exactly those source messages"
        )
        assert m.merge_messages(src.branch_id, "no-target") == [], (
            "a merge into a branch that does not exist yields nothing"
        )
    finally:
        restore()


def test_b31_get_config_returns_a_copy(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        m = _mgr(cb, tmp_path)
        cfg = m.get_config()
        cfg["branches"] = "mutated"
        assert m.get_config()["branches"] != "mutated", (
            "get_config hands back a copy; mutating it must not reach the manager"
        )
    finally:
        restore()


def test_b32_availability_reflects_manager_initialisation(tmp_path):
    cb, restore = _load(tmp_path)
    try:
        assert cb.BRANCHES_AVAILABLE is True
        assert cb.branch_manager is not None, (
            "a clean load builds the module singleton and flags availability"
        )
    finally:
        restore()

    def _refuse(path, **kwargs):
        raise RuntimeError("connection factory unavailable")

    cb2, restore2 = _load(tmp_path, connect=_refuse)
    try:
        assert cb2.BRANCHES_AVAILABLE is False, (
            "if the manager cannot initialise, the flag flips rather than "
            "handing back a half-built store"
        )
        assert cb2.branch_manager is None
    finally:
        restore2()
