#!/usr/bin/env python3
"""Contracts for the componion's journal: one chain per being, refused by name where it breaks.

A being's life is a chain of facts. Each fact has an eid that depends only on
its envelope, and each has a local link ``sha256({eid, prev, seq})``; a keyed
anchor in the store covers the head, the write generation and the local
tables, and the signed audit log carries anchors across days and after every
destruction. The membrane is the single writer into a life.

  * AJ1 -- two writers interleaved on one file keep one chain: the second
    does not read the head while the first holds it, both land with
    consecutive seqs and distinct oseqs, and the chain verifies from genesis.
  * AJ2 -- a byte flipped in a body, in a link, or in the file itself is
    refused as ``body`` or ``link`` at the event it lives in; an empty chain
    is refused ``genesis`` and a missing anchor ``anchor``, never passed.
  * AJ3 -- two links swapped are refused where the order breaks, although
    the head and the anchor still check.
  * AJ4 -- a truncated tail is refused ``truncated`` by the keyed anchor,
    naming both ends; an anchor rewritten under a wrong key, or unkeyed, is
    refused ``anchor``, and under the right key it opens.
  * AJ5 -- an older copy of the store is refused ``older`` against the
    cross-anchors in the real signed audit log, including a same-day
    rollback over a forget; a consistent rollback of the store and the audit
    together is not detected, and that scope is pinned.
  * AJ6 -- after a forget the chain still verifies, the redacted eid being
    recomputed from its envelope, and the payload's key, its ciphertext and
    its reference are gone from the file set -- even a copy a foreign
    connection left behind with ``secure_delete`` off; the checkpoints from
    the forgotten minute on are dropped, the earlier ones kept.
  * AJ7 -- the surface and the actor come from the transport alone: a
    claimed surface is never read, a token counts only while it is valid,
    the attended test binds only consent verbs, another account is refused
    ``owner``, and a tainted event changes no table and no byte.
  * AJ8 -- a flood saturates the daily budget merged across origins
    exactly, the excess is counted, a new day opens a new budget; the kinds
    table has one source, and the law's pin is total.
  * AJ9 -- the eid does not depend on the device: two stores sown from the
    same draws give the same eids for the same facts at different seqs.
  * AJ10 -- resuming is explicit and recorded: a refused store is refused
    the same way however often it is opened, and not one byte changes; a
    resume confirmed with the wrong numbers is refused; the right numbers
    set the broken tail aside behind a ``resumed`` fact that jumps past
    every anchored event, re-issue the forget whose forgetter was set aside,
    and the anchors inside the gap are satisfied by it.
  * AJ11 -- both engines answer the fact identity with the same bytes: two
    thousand envelopes, uppercase hex included, reach every refusal of
    ``fact_envelope`` and its accepted answer; five hundred facts through
    ``fact_id``; the golden redacted envelope has the golden eid in both.
  * AJ12 -- a long-lived process refuses the rollback a fresh process
    refuses: what it remembers of the audit never outranks the audit, whether
    it closed the being or holds it open.
  * AJ13 -- every recorded destruction is still in effect when everything
    else verifies: a destroyed key back, a forgotten body back, the rhythm
    floor lowered or an ended season reopened is refused ``destroyed``.
  * AJ14 -- a resume after a rollback counts every event the audit anchored
    past the copy, and its generation jumps past the audit's, so the being
    reopens alive even when the resume's own anchor is refused, owing it
    until the next write settles it.
  * AJ15 -- a resume acts only on the state it verified: another resume
    landing in between refuses it ``confirm``; and it enforces a destruction
    record still pending, so a word put back behind a cut forget is gone.
  * AJ16 -- a resume never records a key it keeps as destroyed: a stray row
    holding the secret's bytes is dropped without naming the secret, and a
    record naming the secret refuses the resume by name, writing nothing.

Local-only. The platform loads through the shared isolation window with the
platform's configuration, keys, mode, audit log and user modules proven
unreachable; every seam is injected (``tests/_allium_store_support.py``).
"""

import copy
import hashlib
import importlib.util
import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_store_support as support  # noqa: E402
from _allium_window import native_module, open_allium  # noqa: E402
from _isolation import REPO  # noqa: E402

BUDGET_S = {
    "test_aj1_two_interleaved_writers_keep_one_chain_and_lose_no_fact": 2.0,
    "test_aj2_a_flipped_byte_is_refused_at_its_event_and_an_empty_chain_never_passes": 2.0,
    "test_aj3_a_reordered_link_is_refused_where_the_order_breaks": 2.0,
    "test_aj4_a_truncation_is_refused_by_the_keyed_anchor": 2.0,
    "test_aj5_an_older_copy_is_refused_against_the_cross_anchors": 2.0,
    "test_aj6_after_a_forget_the_chain_verifies_and_the_key_ciphertext_and_ref_are_gone": 2.0,
    "test_aj7_the_surface_and_actor_come_from_the_transport_and_a_tainted_event_never_lands": 2.0,
    "test_aj8_a_flood_saturates_the_merged_daily_budget_exactly_and_the_pin_is_total": 2.0,
    "test_aj9_the_eid_does_not_depend_on_the_device": 2.0,
    "test_aj10_resuming_is_explicit_recorded_and_keeps_forgotten_things_forgotten": 2.0,
    "test_aj11_both_engines_answer_the_fact_identity_with_the_same_bytes": 2.0,
    "test_aj12_a_long_lived_process_refuses_the_rollback_a_fresh_process_refuses": 2.0,
    "test_aj13_every_recorded_destruction_is_still_in_effect_when_the_rest_verifies": 2.0,
    "test_aj14_a_resume_after_a_rollback_jumps_past_every_anchor_even_when_its_own_is_refused": 2.0,
    "test_aj15_a_resume_acts_only_on_the_state_it_verified_and_enforces_pending_records": 2.0,
    "test_aj16_a_resume_never_records_a_key_it_keeps_as_destroyed": 2.0,
}


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


@pytest.fixture
def p():
    platform, restore = support.open_platform()
    try:
        yield platform
    finally:
        restore()


def _refusal(p, given, user="local"):
    """What a fresh process gets when it opens ``user``'s being: the refusal, or ``None``."""
    target = support.store(p, given)
    try:
        target.open(user)
    except p.store.StoreRefused as refusal:
        return refusal
    finally:
        target.close()
    return None


def _grown(p, tmp_path, suite, n):
    """A sown being with ``n`` acts after its genesis, closed; ``(seams, path)``."""
    given = support.seams(p, tmp_path, suite=suite)
    target = support.store(p, given)
    being = support.sow(p, target)
    support.build(being, n, support.cli(p))
    target.close()
    return given, support.store_path(p, given)


def _links(path):
    return support.read(path, "SELECT seq, prev, eid, link FROM links ORDER BY seq")


# ---------------------------------------------------------------------------
# AJ1 -- two writers, one chain
# ---------------------------------------------------------------------------
def test_aj1_two_interleaved_writers_keep_one_chain_and_lose_no_fact(p, tmp_path):
    given, path = _grown(p, tmp_path, "aj1", 0)
    a_at_head = threading.Event()
    release = threading.Event()
    b_at_head = threading.Event()
    b_started = threading.Event()
    returned = {}

    def stage_a(name):
        if name == "head":
            a_at_head.set()
            release.wait(3.0)

    def stage_b(name):
        if name == "head":
            b_at_head.set()

    store_a = support.store(p, dict(given, connect=support.sqlite_seam(), stage=stage_a))
    store_b = support.store(p, dict(given, connect=support.sqlite_seam(), stage=stage_b, persistence=dict(
        given["persistence"], busy_timeout_ms=2000)))
    results = {}

    def run(name, being, act):
        if name == "b":
            b_started.set()
        try:
            results[name] = being.append("act", {"act": act}, transport=support.cli(p))
        except BaseException as exc:  # noqa: BLE001 - the contract reads what each writer got
            results[name] = exc
        returned[name] = time.monotonic()

    threads = []
    try:
        being_a = store_a.open("local")
        being_b = store_b.open("local")
        assert being_a is not None and being_b is not None
        before = len(_links(path))
        thread_a = threading.Thread(target=run, args=("a", being_a, "greet"))
        thread_b = threading.Thread(target=run, args=("b", being_b, "play"))
        threads = [thread_a, thread_b]
        thread_a.start()
        assert a_at_head.wait(3.0), "the first writer reached the head"
        thread_b.start()
        assert b_started.wait(3.0), "the second writer started its append before the window opens"
        assert not b_at_head.wait(0.1), "the second writer read the head while the first held it"
        released = time.monotonic()
        release.set()
        for thread in threads:
            thread.join(5.0)
        assert not any(thread.is_alive() for thread in threads)
    finally:
        release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(5.0)
        store_a.close()
        store_b.close()
    a, b = results["a"], results["b"]
    assert isinstance(a, p.membrane.Appended), a
    assert isinstance(b, p.membrane.Appended), b
    assert b_at_head.is_set()
    assert returned["b"] >= released, "the second writer returned only after the first released the head"
    assert sorted([a.seq, b.seq]) == [before, before + 1], (a, b, before)
    assert sorted([a.oseq, b.oseq]) == [before, before + 1], "each fact took its own oseq"
    links = _links(path)
    assert len(links) == before + 2, "no fact was lost"
    assert [row[0] for row in links] == list(range(before + 2))
    assert _refusal(p, given) is None, "a fresh process verifies the chain from genesis"
    fresh = support.store(p, given)
    try:
        assert fresh.status("local").status == "alive"
    finally:
        fresh.close()


# ---------------------------------------------------------------------------
# AJ2 -- a flipped byte, an empty chain, a missing anchor
# ---------------------------------------------------------------------------
def test_aj2_a_flipped_byte_is_refused_at_its_event_and_an_empty_chain_never_passes(p, tmp_path):
    given = support.seams(p, tmp_path, suite="aj2")
    target = support.store(p, given)
    being = support.sow(p, target)
    word = support.canary(p, "aj2")
    for index in range(1, 20):
        if index == 7:
            written = being.append("name", {"name": word}, transport=support.cli(p))
        else:
            written = being.append("act", {"act": support.ACTS[index % 5]}, transport=support.cli(p))
        assert written.seq == index
    target.close()
    path = support.store_path(p, given)
    clean = path.read_bytes()
    assert _refusal(p, given) is None, "the untouched store opens"

    def at_seven(column, table, flip):
        def change(conn):
            eid = conn.execute("SELECT eid FROM links WHERE seq = 7").fetchone()[0]
            value = conn.execute(f"SELECT {column} FROM {table} WHERE eid = ?", (eid,)).fetchone()[0]
            conn.execute(f"UPDATE {table} SET {column} = ? WHERE eid = ?", (flip(value), eid))
        support.edit(path, change)

    def swap_letter(value):
        text = bytes(value).decode("ascii")
        at = text.index(word)
        changed = "b" if text[at] != "b" else "c"
        return (text[:at] + changed + text[at + 1:]).encode("ascii")

    def swap_hex(value):
        return ("0" if value[10] != "0" else "1").join((value[:10], value[11:]))

    # (a) one character of the body at seq 7
    at_seven("body", "bodies", swap_letter)
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused), refusal
    assert (refusal.seq, refusal.reason) == (7, "body")
    assert "event #7" in str(refusal)
    path.write_bytes(clean)

    # (b) one character of the link at seq 7
    at_seven("link", "links", swap_hex)
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused), refusal
    assert (refusal.seq, refusal.reason) == (7, "link")
    path.write_bytes(clean)

    # (c) one byte of the file itself
    needle = word.encode("ascii")
    assert support.count_bytes(path, needle) == 1, "the canary lives in the file exactly once"
    data = bytearray(path.read_bytes())
    at = data.index(needle)
    data[at] = 0x62 if data[at] != 0x62 else 0x63
    path.write_bytes(bytes(data))
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused), refusal
    assert (refusal.seq, refusal.reason) == (7, "body")
    path.write_bytes(clean)

    # (d) the silent zeros: no rows at all, no anchor at all
    def empty(conn):
        for table in ("links", "bodies", "facts"):
            conn.execute(f"DELETE FROM {table}")
    support.edit(path, empty)
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused), refusal
    assert (refusal.seq, refusal.reason) == (0, "genesis")
    path.write_bytes(clean)
    support.edit(path, lambda conn: conn.execute("DELETE FROM meta WHERE key = 'anchor'"))
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused), refusal
    assert (refusal.seq, refusal.reason) == (0, "anchor")
    path.write_bytes(clean)
    assert _refusal(p, given) is None, "the restored store opens again"

    # (e) the envelope's own columns at seq 7, the body, its digest and the link untouched: only the eid sees them
    for column, value in (("t", 1), ("kind", "act"), ("oseq", 1000)):
        support.edit(path, lambda conn, column=column, value=value: conn.execute(
            f"UPDATE facts SET {column} = ? WHERE eid = (SELECT eid FROM links WHERE seq = 7)", (value,)))
        refusal = _refusal(p, given)
        assert isinstance(refusal, p.store.ChainRefused), (column, refusal)
        assert (refusal.seq, refusal.reason) == (7, "eid"), (column, refusal)
        path.write_bytes(clean)


# ---------------------------------------------------------------------------
# AJ3 -- a reordered link
# ---------------------------------------------------------------------------
def test_aj3_a_reordered_link_is_refused_where_the_order_breaks(p, tmp_path):
    given, path = _grown(p, tmp_path, "aj3", 19)
    links = _links(path)
    clean = path.read_bytes()
    e7, e8 = links[7][2], links[8][2]

    def swap(conn):
        conn.execute("UPDATE links SET eid = 'placeholder' WHERE seq = 7")
        conn.execute("UPDATE links SET eid = ? WHERE seq = 8", (e7,))
        conn.execute("UPDATE links SET eid = ? WHERE seq = 7", (e8,))
    support.edit(path, swap)
    swapped = _links(path)
    assert (swapped[7][2], swapped[8][2]) == (e8, e7)
    assert [row[3] for row in swapped] == [row[3] for row in links], "the link column is untouched"
    # Witness: the head still recomputes, and the anchor over it still verifies.
    seq, prev, eid, link = swapped[-1]
    assert p.chain.link(eid, prev, seq) == link
    meta = {key: p.wire.parse(bytes(value)) for key, value in support.read(path, "SELECT key, value FROM meta")}
    conn = sqlite3.connect(str(path))
    try:
        local = p.chain.local_digest(p.chain.read_local(conn), meta)
    finally:
        conn.close()
    expected = p.chain.anchor_value(being=meta["being"], gen=meta["gen"], head_seq=seq, head_link=link, local=local,
                                    owner=meta["owner"], soil=meta["soil"], key_id=support.KEY_ID,
                                    anchor_key=support.KEY)
    assert meta["anchor"] == expected, "only a check of every row can see the swap"
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused), refusal
    assert (refusal.seq, refusal.reason) == (7, "link")

    # Another sound fact in place of seq 7, with its own link over seq 6: seq 8 no longer follows it.
    path.write_bytes(clean)
    [(being, origin, laws)] = support.read(path, "SELECT being, origin, laws FROM facts WHERE eid = ?", (e7,))
    fact = {"being": being, "body": {"act": "water"}, "kind": "act", "laws": laws, "origin": origin, "oseq": 1000,
            "t": 0}
    made = p.journal.fact_id(fact)
    own_link = p.chain.link(made["eid"], links[6][3], 7)

    def substitute(conn):
        conn.execute("INSERT INTO facts (eid, being, t, kind, origin, oseq, laws, body_sha256) "
                     "VALUES (?, ?, 0, 'act', ?, 1000, ?, ?)", (made["eid"], being, origin, laws, made["body"]))
        conn.execute("INSERT INTO bodies (eid, body, redacted_by) VALUES (?, ?, NULL)",
                     (made["eid"], p.wire.emit(fact["body"])))
        conn.execute("UPDATE links SET eid = ?, link = ? WHERE seq = 7", (made["eid"], own_link))
        conn.execute("DELETE FROM bodies WHERE eid = ?", (e7,))
        conn.execute("DELETE FROM facts WHERE eid = ?", (e7,))
    support.edit(path, substitute)
    substituted = _links(path)
    seq, prev, eid, link = substituted[7]
    assert (prev, eid) == (links[6][3], made["eid"]) and p.chain.link(eid, prev, seq) == link, \
        "witness: seq 7's own link recomputes"
    assert substituted[8] == links[8], "seq 8 is untouched"
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused), refusal
    assert (refusal.seq, refusal.reason) == (8, "link")


# ---------------------------------------------------------------------------
# AJ4 -- a truncation
# ---------------------------------------------------------------------------
def _ocj(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _prefix_verifies(path, last):
    """A verifier written apart from the platform: links, body digests and eids of rows 0..=last."""
    import hashlib

    rows = support.read(path, "SELECT l.seq, l.prev, l.eid, l.link, f.being, f.t, f.kind, f.origin, f.oseq, "
                              "f.laws, f.body_sha256, b.body FROM links l JOIN facts f ON f.eid = l.eid "
                              "JOIN bodies b ON b.eid = l.eid ORDER BY l.seq")
    prev = "0" * 64
    for expected, row in enumerate(rows):
        seq, stored_prev, eid, link, being, t, kind, origin, oseq, laws, digest, body = row
        envelope = {"being": being, "body": digest, "kind": kind, "laws": laws, "origin": origin, "oseq": oseq,
                    "t": t}
        if (seq != expected or stored_prev != prev or hashlib.sha256(bytes(body)).hexdigest() != digest
                or hashlib.sha256(_ocj(envelope)).hexdigest() != eid
                or hashlib.sha256(_ocj({"eid": eid, "prev": prev, "seq": seq})).hexdigest() != link):
            return False
        prev = link
    return len(rows) == last + 1


def test_aj4_a_truncation_is_refused_by_the_keyed_anchor(p, tmp_path):
    given, path = _grown(p, tmp_path, "aj4", 19)
    assert len(_links(path)) == 20

    def cut(conn):
        eids = [row[0] for row in conn.execute("SELECT eid FROM links WHERE seq >= 17")]
        conn.execute("DELETE FROM links WHERE seq >= 17")
        for eid in eids:
            conn.execute("DELETE FROM bodies WHERE eid = ?", (eid,))
            conn.execute("DELETE FROM facts WHERE eid = ?", (eid,))
    support.edit(path, cut)
    assert _prefix_verifies(path, 16), "witness: the rows left are a sound prefix 0..=16"
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused), refusal
    assert refusal.reason == "truncated"
    assert "#19" in str(refusal) and "#16" in str(refusal), str(refusal)
    assert refusal.kept_seq == 16
    cut_bytes = path.read_bytes()

    support.rewrite_anchor(p, path, seq=16, key_id=support.KEY_ID, anchor_key=support.WRONG_KEY)
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused) and refusal.reason == "anchor", refusal
    path.write_bytes(cut_bytes)
    support.rewrite_anchor(p, path, seq=16, key_id="nokey", anchor_key=None)
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused) and refusal.reason == "anchor", refusal
    path.write_bytes(cut_bytes)
    support.rewrite_anchor(p, path, seq=16, key_id=support.KEY_ID, anchor_key=support.KEY)
    assert _refusal(p, given) is None, "witness: under the right key the shorter chain is whole"


# ---------------------------------------------------------------------------
# AJ5 -- an older copy, against the cross-anchors
# ---------------------------------------------------------------------------
def _audit_files(audit):
    folder = Path(audit._db_path).parent
    return {child.name: child.read_bytes() for child in sorted(folder.iterdir()) if child.is_file()}


def _restore_audit(audit, files):
    folder = Path(audit._db_path).parent
    for child in sorted(folder.iterdir()):
        if child.is_file() and child.name not in files:
            child.unlink()
    for name, data in files.items():
        folder.joinpath(name).write_bytes(data)


def test_aj5_an_older_copy_is_refused_against_the_cross_anchors(p, tmp_path):
    audit = support.load_audit(tmp_path)
    given = support.seams(p, tmp_path, suite="aj5", audit=audit)
    target = support.store(p, given)
    being = support.sow(p, target)
    support.build(being, 20, support.cli(p))
    target.close()
    path = support.store_path(p, given)
    copy_a = path.read_bytes()
    audit_a = _audit_files(audit)

    given["clock"].advance_days(1)
    target = support.store(p, given)
    being = target.open("local")
    support.build(being, 20, support.cli(p))
    being_tag = being.being_tag
    target.close()
    latest = path.read_bytes()
    seal = p.anchors.seal_key(support.KEY)
    found = p.anchors.anchors(audit, being_tag, support.KEY_ID, seal, given["cipher"][1])
    assert found["latest"][1] == 21, "the daily anchor of day 1 landed at seq 21"
    keys = [event["details"]["key"] for event in audit.get_events(limit=50, event_type="allium_anchor")]
    assert keys == [support.KEY_ID, support.KEY_ID], "two daily anchors, sealed under the key"

    path.write_bytes(copy_a)
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused) and refusal.reason == "older", refusal

    path.write_bytes(latest)
    target = support.store(p, given)
    being = target.open("local")
    taught = being.append("lang_teach", {}, transport=support.cli(p), payload="shallot")
    target.close()
    copy_b = path.read_bytes()
    target = support.store(p, given)
    being = target.open("local")
    forgot = being.append("lang_forget", {"target": taught.eid}, transport=support.cli(p))
    assert isinstance(forgot, p.membrane.Appended)
    target.close()
    assert _refusal(p, given) is None, "after the forget the store opens"
    after_forget = path.read_bytes()
    path.write_bytes(copy_b)
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused) and refusal.reason == "older", refusal

    # Copy A with its generation raised past every anchor: its store anchor names another generation.
    latest_gen = p.anchors.anchors(audit, being_tag, support.KEY_ID, seal, given["cipher"][1])["latest"][0]
    path.write_bytes(copy_a)
    support.edit(path, lambda conn: conn.execute("UPDATE meta SET value = ? WHERE key = 'gen'",
                                                 (p.wire.emit(latest_gen + 1),)))
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused) and refusal.reason == "anchor", refusal

    # A rollback over a write that moves the generation and not the seq: still older.
    path.write_bytes(after_forget)
    target = support.store(p, given)
    target.open("local").heard_note(5, "leek", {"turn": 1})
    target.close()
    copy_c = path.read_bytes()
    target = support.store(p, given)
    target.open("local").heard_end_season(5)
    target.close()
    assert _refusal(p, given) is None, "after the season ended the store opens"
    path.write_bytes(copy_c)
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused) and refusal.reason == "older", refusal

    # The honest scope: the store and the audit rolled back together are not detected.
    path.write_bytes(copy_a)
    _restore_audit(audit, audit_a)
    given["audit"] = type(audit)(db_path=audit._db_path)
    target = support.store(p, given)
    try:
        assert target.status("local").status == "alive"
    finally:
        target.close()


# ---------------------------------------------------------------------------
# AJ6 -- a forget leaves a chain that verifies and no bytes of what it forgot
# ---------------------------------------------------------------------------
def test_aj6_after_a_forget_the_chain_verifies_and_the_key_ciphertext_and_ref_are_gone(p, tmp_path):
    given = support.seams(p, tmp_path, suite="aj6")
    target = support.store(p, given)
    being = support.sow(p, target)
    # The store's own connection: the connect seam turned secure_delete off, the store turns it on.
    store_conn = given["connect"].connections[-1]
    path = support.store_path(p, given)
    genesis_eid = _links(path)[0][2]
    early = being.checkpoint_put(0, being.laws, genesis_eid, {"events": [genesis_eid]})
    given["clock"].wall += 5 * 60
    word = support.canary(p, "aj6")
    taught = being.append("lang_teach", {}, transport=support.cli(p), payload=word)
    assert isinstance(taught, p.membrane.Appended) and taught.t == 5, taught
    late = being.checkpoint_put(taught.t, being.laws, taught.eid, {"events": [genesis_eid, taught.eid]})
    assert (early.t, late.t) == (0, 5)
    [(ref, ct)] = support.read(path, "SELECT ref, ct FROM payloads")
    [(key,)] = support.read(path, "SELECT key FROM keys WHERE name = ?", ("payload:" + ref,))
    key, ct = bytes(key), bytes(ct)
    assert key in given["entropy"].chunks, "the payload key is a drawn key"

    # A foreign connection with secure_delete off writes the same key bytes and deletes them.
    def foreign(sql, params):
        def change(conn):
            conn.execute("PRAGMA secure_delete = OFF")
            conn.execute(sql, params)
        support.edit(path, change)
    foreign("INSERT INTO keys (name, key) VALUES ('scratch', ?)", (key,))
    foreign("DELETE FROM keys WHERE name = 'scratch'", ())
    assert support.count_bytes(path, key) == 2, "witness: the live key and the copy the foreign history left"
    assert support.count_bytes(path, ct) >= 1
    assert support.count_bytes(path, ref.encode("ascii")) >= 1
    assert support.count_bytes(path, word.encode("ascii")) == 0, "the word is sealed"
    assert being.unseal(ref) == word

    forgot = being.append("lang_forget", {"target": taught.eid}, transport=support.cli(p))
    assert isinstance(forgot, p.membrane.Appended), forgot
    assert support.count_bytes(path, key) == 0, "the key is gone, the foreign copy with it"
    assert support.count_bytes(path, ct) == 0
    assert support.count_bytes(path, ref.encode("ascii")) == 0
    assert support.read(path, "SELECT body, redacted_by FROM bodies WHERE eid = ?", (taught.eid,)) == [
        (None, forgot.eid)]
    assert store_conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    assert support.read(path, "SELECT t FROM checkpoints ORDER BY t") == [(0,)], "from the forgotten minute on"
    record = _anchor_records(p, given, being.being_tag)[-1]
    assert record["why"] == "destroy", record
    assert record["destroyed"] == [p.anchors.fingerprint(key)] and record["forgot"] == [taught.eid], record
    with pytest.raises(p.membrane.MembraneRefused) as info:
        being.unseal(ref)
    assert info.value.code == "payload"
    being.verify()
    target.close()
    assert _refusal(p, given) is None, "the redacted fact's eid is recomputed from its envelope"


# ---------------------------------------------------------------------------
# AJ7 -- surfaces, actors, taint
# ---------------------------------------------------------------------------
_REFUSED_VERBS = ("grant", "revoke", "compost", "export", "import", "sow", "resume", "finish_sowing")
_REFUSED_KINDS = {
    "lang_forget": {"target": "0" * 64}, "forget_rhythm": {}, "laws_pin": {}, "laws_unpin": {}, "evolve": {},
    "name": {"name": "sprout"}, "dream_depth": {"depth": "deep"}, "rest_begin": {},
}


def _code(p, call):
    try:
        call()
    except p.membrane.MembraneRefused as refusal:
        return refusal.code
    except p.store.StoreRefused as refusal:
        return "store:" + refusal.code
    return None


def test_aj7_the_surface_and_actor_come_from_the_transport_and_a_tainted_event_never_lands(p, tmp_path):
    M = p.membrane
    given = support.seams(p, tmp_path, suite="aj7")
    target = support.store(p, given)
    being = support.sow(p, target)
    now = given["clock"].wall
    synthetic = {"role": "admin", "sub": "local", "type": "access"}
    local = M.Transport("web", principal=synthetic, claimed="web_session")
    assert M.surface_of(local, now) == "web_local", "the claimed surface is never read"
    for verb in _REFUSED_VERBS:
        assert _code(p, lambda verb=verb: M.permit(verb, local, now)) == "surface", verb
    assert _code(p, lambda: target.sow(transport=local, law="fixture", tz_minutes=0,
                                       rhythm_consent=False)) == "surface"
    assert _code(p, lambda: target.finish_sowing(transport=local, confirm="00000000")) == "surface"
    assert _code(p, lambda: target.resume(transport=local, confirm=(0, 0))) == "surface"
    assert _code(p, lambda: target.unclaimed(transport=local)) == "owner", "the synthetic principal is no administrator"
    for kind, body in _REFUSED_KINDS.items():
        assert _code(p, lambda kind=kind, body=body: being.append(kind, body, transport=local)) == "surface", kind
    assert isinstance(being.append("act", {"act": "touch"}, transport=local), M.Appended)
    assert isinstance(being.append("lang_teach", {}, transport=local, payload="leek"), M.Appended)

    token = support.principal("local", now)
    session = M.Transport("web", principal=token)
    assert M.surface_of(session, now) == "web_session"
    assert M.permit("grant", session, now) == "web_session"
    assert M.surface_of(M.Transport("web", principal=dict(token, exp=now)), now) == "web_local"
    assert M.surface_of(M.Transport("web", principal=dict(token, iat=True)), now) == "web_local"
    plain = M.Transport("cli", attended=False)
    assert isinstance(being.append("act", {"act": "warm"}, transport=plain), M.Appended)
    assert _code(p, lambda: M.permit("grant", plain, now)) == "attended"
    assert M.permit("grant", M.Transport("cli", attended=True), now) == "cli_tty"
    assert _code(p, lambda: M.permit("grant", M.Transport("light_hook"), now)) == "surface"
    assert _code(p, lambda: being.append("presence_hour", {"active": True, "observed_hour": 3},
                                         transport=local)) == "surface"
    assert _code(p, lambda: being.rhythm_put(local, 3, True)) == "surface"
    assert _code(p, lambda: M.surface_of(M.Transport("tool"), now)) == "surface"
    assert _code(p, lambda: being.append("act", {"act": "play"}, transport=M.Transport("tool"))) == "surface"
    assert _code(p, lambda: being.append("act", {"act": "play"}, transport=M.Transport("sync"))) == "surface"

    # Another account on someone's store.
    shared = support.seams(p, tmp_path.joinpath("multi"), suite="aj7", index=1, single_user=lambda: False)
    multi = support.store(p, shared)
    alice = multi.sow(transport=support.web(p, "alice", now), law="fixture", tz_minutes=0, rhythm_consent=False)
    assert isinstance(alice.append("act", {"act": "water"}, transport=support.web(p, "alice", now)), M.Appended)
    assert _code(p, lambda: alice.append("act", {"act": "water"},
                                         transport=support.web(p, "bob", now))) == "owner"
    assert _code(p, lambda: M.actor_of(M.Transport("cli"), False, now)) == "owner"
    multi.close()

    # A tainted event changes no table and no byte.
    path = support.store_path(p, given)
    before = (support.counts(path), support.sha256_file(path))
    assert _code(p, lambda: being.append("act", {"act": "greet"}, transport=plain, grant_ref="g1")) == "taint"
    assert _code(p, lambda: being.append("act", {"act": "greet"}, transport=plain,
                                         grant_ref=("g1", "g2"))) == "taint"
    assert (support.counts(path), support.sha256_file(path)) == before
    target.close()


# ---------------------------------------------------------------------------
# AJ8 -- budgets, merged across origins; the table and the pin
# ---------------------------------------------------------------------------
def _load_script(name, module_name):
    path = REPO.joinpath("scripts", name)
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    return module


def test_aj8_a_flood_saturates_the_merged_daily_budget_exactly_and_the_pin_is_total(p, tmp_path):
    M = p.membrane
    given = support.seams(p, tmp_path, suite="aj8")
    target = support.store(p, given)
    being = support.sow(p, target)
    other = "b2" * 8
    support.land(being, other, 0, 0, "move_pot", {})
    support.land(being, other, 1, 0, "move_pot", {})
    outcomes = [being.append("move_pot", {}, transport=support.cli(p)) for _ in range(5)]
    accepted = [outcome for outcome in outcomes if isinstance(outcome, M.Appended)]
    dropped = [outcome for outcome in outcomes if isinstance(outcome, M.Dropped)]
    assert (len(accepted), len(dropped)) == (1, 4), outcomes
    assert all(outcome.reason == "budget" for outcome in dropped)
    path = support.store_path(p, given)
    assert support.read(path, "SELECT day, kind, dropped FROM overflow") == [(0, "move_pot", 4)]
    given["clock"].advance_days(1)
    fresh = being.append("move_pot", {}, transport=support.cli(p))
    assert isinstance(fresh, M.Appended) and fresh.t == 1440, fresh
    target.close()
    assert _refusal(p, given) is None, "the landed facts, the drops and the new day verify"

    author = _load_script("allium_author_journal.py", "_aj8_author")
    files = author.author()
    assert sorted(files) == ["journal_v1.json"]
    on_disk = REPO.joinpath("opti_oignon", "allium", "tables", "journal_v1.json").read_text(encoding="ascii")
    assert files["journal_v1.json"] == on_disk, "the kinds table has one source"

    law = p.lawfiles.law("fixture")
    table_bytes = p.lawfiles.table_bytes("journal_v1")
    pin = M.validate_pin(law, table_bytes)
    assert pin["budgets"]["move_pot"] == 3 and pin["budgets"]["act"] == 64
    wrong_digest = copy.deepcopy(law)
    wrong_digest["journal"]["table"]["sha256"] = "0" * 64
    missing_reserved = copy.deepcopy(law)
    del missing_reserved["journal"]["budgets"]["braid"]
    exempt_budgeted = copy.deepcopy(law)
    exempt_budgeted["journal"]["budgets"]["resumed"] = 1
    for bad in (wrong_digest, missing_reserved, exempt_budgeted):
        with pytest.raises(p.store.StoreRefused) as info:
            M.validate_pin(bad, table_bytes)
        assert info.value.code == "law", info.value
    for budget, accepted in ((0, False), (1, True), (4096, True), (4097, False)):
        ranged = copy.deepcopy(law)
        ranged["journal"]["budgets"]["move_pot"] = budget
        if accepted:
            assert M.validate_pin(ranged, table_bytes)["budgets"]["move_pot"] == budget, "witness: in range"
        else:
            with pytest.raises(p.store.StoreRefused) as info:
                M.validate_pin(ranged, table_bytes)
            assert info.value.code == "law", (budget, info.value)


# ---------------------------------------------------------------------------
# AJ9 -- the eid is the fact's, not the device's
# ---------------------------------------------------------------------------
def test_aj9_the_eid_does_not_depend_on_the_device(p, tmp_path):
    M = p.membrane
    given_a = support.seams(p, tmp_path.joinpath("a"), suite="aj9")
    given_b = support.seams(p, tmp_path.joinpath("b"), suite="aj9")
    store_a = support.store(p, given_a)
    store_b = support.store(p, given_b)
    being_a = support.sow(p, store_a)
    being_b = support.sow(p, store_b)
    assert (being_a.being, being_a.origin) == (being_b.being, being_b.origin), "the same draws"
    other = "c3" * 8
    for oseq in range(3):
        support.land(being_b, other, oseq, 0, "act", {"act": "warm"})
    facts = [("act", {"act": act}) for act in support.ACTS]
    written_a = [being_a.append(kind, body, transport=support.cli(p)) for kind, body in facts]
    written_b = [being_b.append(kind, body, transport=support.cli(p)) for kind, body in facts]
    assert all(isinstance(item, M.Appended) for item in written_a + written_b)
    assert [item.eid for item in written_a] == [item.eid for item in written_b], "the same facts, the same eids"
    assert [item.oseq for item in written_a] == [item.oseq for item in written_b] == [1, 2, 3, 4, 5]
    assert [item.seq for item in written_a] == [1, 2, 3, 4, 5]
    assert [item.seq for item in written_b] == [4, 5, 6, 7, 8]
    links_a = {row[2]: row[3] for row in _links(support.store_path(p, given_a))}
    links_b = {row[2]: row[3] for row in _links(support.store_path(p, given_b))}
    for item in written_a:
        assert links_a[item.eid] != links_b[item.eid], "the link is local, the eid is not"
    store_a.close()
    store_b.close()


# ---------------------------------------------------------------------------
# AJ10 -- restore, never repair
# ---------------------------------------------------------------------------
def _anchor_records(p, given, being_tag):
    """Every cross-anchor record of one being in the audit stand-in, oldest first, opened under the seal key."""
    seal = p.anchors.seal_key(support.KEY)
    records = []
    for event in reversed(given["audit"].get_events(limit=256, event_type="allium_anchor")):
        details = event["details"]
        if details["tag"] == being_tag:
            records.append(p.wire.parse(given["cipher"][1](seal, bytes.fromhex(details["sealed"]))))
    return records


def test_aj10_resuming_is_explicit_recorded_and_keeps_forgotten_things_forgotten(p, tmp_path):
    given = support.seams(p, tmp_path, suite="aj10")
    target = support.store(p, given)
    being = support.sow(p, target)
    being_tag = being.being_tag
    path = support.store_path(p, given)
    taught = ref = None
    for seq in range(1, 20):
        if seq == 5:
            taught = written = being.append("lang_teach", {}, transport=support.cli(p), payload="scallion")
            [(ref,)] = support.read(path, "SELECT ref FROM payloads")
        elif seq == 9:
            written = being.append("lang_forget", {"target": taught.eid}, transport=support.cli(p))
        else:
            if seq == 15:
                given["clock"].advance_days(1)
            written = being.append("act", {"act": support.ACTS[seq % 5]}, transport=support.cli(p))
        assert isinstance(written, p.membrane.Appended) and written.seq == seq, written
    target.close()
    anchored = {record["seq"]: record for record in _anchor_records(p, given, being_tag)}
    assert sorted(anchored) == [1, 9, 15], "a daily anchor at 1 and 15, a destruction anchor at 9"
    links = _links(path)
    support.edit(path, lambda conn: conn.execute("UPDATE bodies SET body = ? WHERE eid = ?",
                                                 (b'{"act":"none"}', links[7][2])))
    broken = support.sha256_file(path)

    # Refused the same way, however often; nothing is repaired behind the person's back.
    for _ in range(3):
        refusal = _refusal(p, given)
        assert isinstance(refusal, p.store.ChainRefused), refusal
        assert (refusal.seq, refusal.reason, refusal.kept_seq, refusal.discarded) == (7, "body", 6, 13), refusal
        assert support.sha256_file(path) == broken
    looked = support.store(p, given)
    status = looked.status("local")
    looked.close()
    assert status.status == "unreadable" and status.offer is not None, status
    assert (status.offer.kept_seq, status.offer.discarded) == (6, 13), status.offer

    resumer = support.store(p, given)
    with pytest.raises(p.store.ResumeRefused) as info:
        resumer.resume(transport=support.cli(p), confirm=(6, 12))
    assert info.value.code == "confirm"
    assert support.sha256_file(path) == broken, "a refused resume writes nothing"
    resumer.resume(transport=support.cli(p), confirm=(6, 13))
    resumer.close()

    rows = support.read(path, "SELECT l.seq, l.prev, l.eid, f.kind, b.body FROM links l JOIN facts f "
                              "ON f.eid = l.eid JOIN bodies b ON b.eid = l.eid ORDER BY l.seq")
    assert [row[0] for row in rows] == [0, 1, 2, 3, 4, 5, 6, 20, 21], "the tail is set aside, the seq jumps"
    removed = [row[2] for row in links[7:]]
    resumed = rows[7]
    assert resumed[3] == "resumed" and resumed[1] == links[6][3], resumed
    digest = hashlib.sha256(_ocj(removed)).hexdigest()
    assert p.wire.parse(bytes(resumed[4])) == {"digest": digest, "removed": 13}
    reissued = rows[8]
    assert reissued[3] == "lang_forget" and p.wire.parse(bytes(reissued[4])) == {"target": taught.eid}
    assert support.read(path, "SELECT body, redacted_by FROM bodies WHERE eid = ?", (taught.eid,)) == [
        (None, reissued[2])], "the forget whose forgetter was set aside is issued again"
    for eid in removed:
        assert support.read(path, "SELECT COUNT(*) FROM facts WHERE eid = ?", (eid,)) == [(0,)]
    latest = _anchor_records(p, given, being_tag)[-1]
    assert (latest["why"], latest["seq"]) == ("resume", 21), "the resume is recorded in the audit log"

    fresh = support.store(p, given)
    reopened = fresh.open("local")
    assert fresh.status("local").status == "alive"
    assert reopened.append("act", {"act": "warm"}, transport=support.cli(p)).seq == 22
    with pytest.raises(p.membrane.MembraneRefused) as info:
        reopened.unseal(ref)
    assert info.value.code == "payload", "the forgotten word stays forgotten"
    fresh.close()

    # The anchor at 15 names an event the resume set aside: the gap satisfies it, and nothing else would.
    conn = sqlite3.connect(str(path))
    conn.isolation_level = None
    try:
        verified = p.chain.verify(conn, name_tag=p.anchors.owner_tag("local"), name_soil="encrypted",
                                  key_id=support.KEY_ID, anchor_key=support.KEY)
        assert verified.gaps == [[6, 20]], verified.gaps
        at_15 = (anchored[15]["gen"], 15, anchored[15]["head"])
        p.chain.check_cross(verified, at_15, lambda seq: p.chain.link_at(conn, seq))
        without_gap = p.chain.Verified(**dict(verified.__dict__, gaps=[]))
        with pytest.raises(p.store.ChainRefused) as info:
            p.chain.check_cross(without_gap, at_15, lambda seq: p.chain.link_at(conn, seq))
        assert info.value.reason == "fork"
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# AJ11 -- both engines, one fact identity
# ---------------------------------------------------------------------------
_GOLDEN_BODY = "2e23a73a0d01ba27f7d814553b7572b21b115624c5e682488d1b70f080e0192f"
_GOLDEN_EID = "b1e8fbdce5df4fbc2bce7261fb945d2a33ae0bd5d5c539fe1cff7edcf5585d3a"
_GOLDEN_FACT = {"being": "0f" * 16, "body": {"note": "a first word", "turn": 3}, "kind": "sow", "laws": 0,
                "origin": "a1" * 8, "oseq": 0, "t": 0}
_DETAILS = ("being", "body", "fact", "fields", "kind", "laws", "origin", "oseq", "t")
_KIND_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789_"
_MAX_INT = (1 << 53) - 1


def _hex(stream, n):
    raw = b""
    while len(raw) < n:
        raw += stream.next_u64().to_bytes(8, "big")
    return raw[:n].hex()


def _natural(stream):
    pick = stream.below(3)
    return 0 if pick == 0 else _MAX_INT if pick == 1 else stream.below(_MAX_INT + 1)


def _kind(stream, length):
    return "".join(_KIND_ALPHABET[stream.below(len(_KIND_ALPHABET))] for _ in range(length))


def _upper_once(stream, value):
    """The value with one hex letter made upper case (or one digit replaced by ``A``)."""
    letters = [at for at, char in enumerate(value) if "a" <= char <= "f"]
    if letters:
        at = letters[stream.below(len(letters))]
        return value[:at] + value[at].upper() + value[at + 1:]
    at = stream.below(len(value))
    return value[:at] + "A" + value[at + 1:]


def _resized(stream, value):
    return value[1:] if stream.below(2) == 0 else value + "0"


def _not_text(stream):
    return (1, None, True, [])[stream.below(4)]


def _not_int(stream):
    return ("0", None, False, {})[stream.below(4)]


def _drawn_fact(stream):
    signed = _natural(stream) * (1 if stream.below(2) == 0 else -1)
    return {"being": _hex(stream, 16), "body": {"n": signed, "w": _hex(stream, 4)},
            "kind": _kind(stream, stream.below(32) + 1), "laws": _natural(stream), "origin": _hex(stream, 8),
            "oseq": _natural(stream), "t": _natural(stream) * (1 if stream.below(2) == 0 else -1)}


def _drawn_envelope(stream, fact, digest):
    """An envelope of ``fact``, sound or broken on purpose, and the refusal detail it must draw."""
    envelope = dict(fact, body=digest)
    pick = stream.below(16)
    if pick == 0:
        del envelope[sorted(envelope)[stream.below(7)]]
        return envelope, "fields"
    if pick == 1:
        return dict(envelope, seq=1), "fields"
    if pick == 2:
        return _not_text(stream), "fact"
    if pick in (3, 6):
        name = "being" if pick == 3 else "origin"
        how = stream.below(3)
        broken = (_upper_once(stream, envelope[name]) if how == 0 else _resized(stream, envelope[name]) if how == 1
                  else _not_text(stream))
        return dict(envelope, **{name: broken}), name
    if pick == 4:
        how = stream.below(5)
        broken = ("", _kind(stream, 33), _kind(stream, 3) + "A", _kind(stream, 3) + "." + _kind(stream, 2),
                  None)[how]
        return dict(envelope, kind=broken if how < 4 else _not_text(stream)), "kind"
    if pick in (5, 7):
        name = "laws" if pick == 5 else "oseq"
        broken = -(min(_natural(stream) + 1, _MAX_INT)) if stream.below(2) == 0 else _not_int(stream)
        return dict(envelope, **{name: broken}), name
    if pick == 8:
        return dict(envelope, t=_not_int(stream)), "t"
    if pick == 9:
        how = stream.below(4)
        broken = (_upper_once(stream, digest), _resized(stream, digest), fact["body"], None)[how]
        return dict(envelope, body=broken if how < 3 else _not_text(stream)), "body"
    return envelope, None


def test_aj11_both_engines_answer_the_fact_identity_with_the_same_bytes():
    loaded, restore = open_allium(native=True)
    try:
        native = native_module(loaded)
        ref = loaded["opti_oignon.allium.ref.protocol"]
        journal = loaded["opti_oignon.allium.ref.journal"]
        wire = loaded["opti_oignon.allium.wire"]
        rng = loaded["opti_oignon.allium.rng"]

        def both(request):
            data = wire.emit(request) if isinstance(request, dict) else request
            mine = ref.call(data)
            assert bytes(native.allium_call(data)) == mine, data[:200]
            return wire.parse(mine)

        seen = {}
        built = 0
        case = 0
        upper = 0
        while built < 2000:
            stream = rng.Stream(bytes(32), "test.aj9", case)
            case += 1
            envelopes = []
            first = None
            for _ in range(stream.below(4) + 1):
                fact = _drawn_fact(stream)
                envelope, defect = _drawn_envelope(stream, fact, journal.body_digest(fact["body"]))
                if isinstance(envelope, dict) and any(isinstance(envelope.get(name), str) and
                                                      envelope[name] != envelope[name].lower()
                                                      for name in ("being", "body", "origin")):
                    upper += 1
                first = defect if first is None else first
                envelopes.append(envelope)
                built += 1
            answer = both({"envelopes": envelopes, "op": "fact_envelope", "v": 1})
            if first is None:
                assert sorted(answer) == ["eids"] and len(answer["eids"]) == len(envelopes), answer
                seen["accepted"] = seen.get("accepted", 0) + 1
            else:
                assert answer == {"detail": first, "refused": "bad_fact"}, (first, answer)
                seen[first] = seen.get(first, 0) + 1
        assert upper >= 100, "witness: uppercase hex in beings, bodies and origins is sent"
        assert sorted(seen) == sorted(_DETAILS + ("accepted",)), seen
        assert min(seen.values()) >= 1, seen

        # The list itself, before any envelope is read.
        op = {"op": "fact_envelope", "v": 1}
        assert both(dict(op, envelopes={})) == {"detail": "envelopes", "refused": "bad_request"}
        assert both(dict(op, envelopes=[], more=1)) == {"detail": "fields", "refused": "bad_request"}
        assert both(op) == {"detail": "fields", "refused": "bad_request"}
        assert both(dict(op, envelopes=[])) == {"eids": []}
        over = b'{"envelopes":[' + b",".join([b"0"] * 100001) + b'],"op":"fact_envelope","v":1}'
        assert both(over) == {"detail": "items", "refused": "limit"}

        # Five hundred facts through fact_id, every one accepted, with the same bytes.
        for index in range(500):
            answer = both({"fact": _drawn_fact(rng.Stream(bytes(32), "test.aj9.fact_id", index)), "op": "fact_id",
                           "v": 1})
            assert sorted(answer) == ["body", "eid"], answer

        # The golden fact and its redacted envelope, in both engines.
        assert both({"fact": _GOLDEN_FACT, "op": "fact_id", "v": 1}) == {"body": _GOLDEN_BODY, "eid": _GOLDEN_EID}
        redacted = dict(_GOLDEN_FACT, body=_GOLDEN_BODY)
        assert both(dict(op, envelopes=[redacted, redacted])) == {"eids": [_GOLDEN_EID, _GOLDEN_EID]}
    finally:
        restore()


# ---------------------------------------------------------------------------
# AJ12 -- a long-lived process refuses the rollback a fresh process refuses
# ---------------------------------------------------------------------------
def test_aj12_a_long_lived_process_refuses_the_rollback_a_fresh_process_refuses(p, tmp_path):
    given = support.seams(p, tmp_path, suite="aj12")
    cli = support.cli(p)
    first = support.store(p, dict(given))
    being = support.sow(p, first)
    taught = being.append("lang_teach", {}, transport=cli, payload=support.canary(p, "aj12"))
    first.close()
    path = support.store_path(p, given)
    before_forget = path.read_bytes()

    # (1) The server opened the being once and closed it; another process forgets; the file is rolled back.
    server = support.store(p, dict(given))
    assert server.open("local") is not None
    server.close()
    other = support.store(p, dict(given))
    assert isinstance(other.open("local").append("lang_forget", {"target": taught.eid}, transport=cli),
                      p.membrane.Appended)
    other.close()
    after_forget = path.read_bytes()
    path.write_bytes(before_forget)
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused) and refusal.reason == "older", "witness: a fresh process"
    with pytest.raises(p.store.ChainRefused) as info:
        server.open("local")
    assert info.value.reason == "older", info.value
    server.close()

    # (2) The server holds the being open while another process teaches, then forgets, and the file is rolled back.
    path.write_bytes(after_forget)
    held = support.store(p, dict(given))
    assert held.open("local") is not None
    other = support.store(p, dict(given))
    again = other.open("local").append("lang_teach", {}, transport=cli, payload=support.canary(p, "aj12b"))
    other.close()
    before_second = path.read_bytes()
    other = support.store(p, dict(given))
    assert isinstance(other.open("local").append("lang_forget", {"target": again.eid}, transport=cli),
                      p.membrane.Appended)
    other.close()
    path.write_bytes(before_second)
    refusal = _refusal(p, given)
    assert isinstance(refusal, p.store.ChainRefused) and refusal.reason == "older", "witness: a fresh process"
    status = held.status("local")
    assert (status.status, status.reason) == ("unreadable", "older"), status
    held.close()


# ---------------------------------------------------------------------------
# AJ16 -- a resume never records a key it keeps as destroyed
# ---------------------------------------------------------------------------
def _look(p, given, user="local"):
    target = support.store(p, given)
    try:
        return target.status(user)
    finally:
        target.close()


def test_aj16_a_resume_never_records_a_key_it_keeps_as_destroyed(p, tmp_path):
    given = support.seams(p, tmp_path, suite="aj16")
    target = support.store(p, given)
    being = support.sow(p, target)
    support.build(being, 3, support.cli(p))
    being_tag = being.being_tag
    target.close()
    path = support.store_path(p, given)
    clean = path.read_bytes()
    [(secret,)] = support.read(path, "SELECT key FROM keys WHERE name = 'being_secret'")
    secret_fp = p.anchors.fingerprint(bytes(secret))

    # (1) A whole pending record naming the secret's fingerprint: the resume is refused by name and writes nothing.
    record = {"destroyed": [secret_fp], "ended": [], "forgot": [], "rhythm_floor": 0}
    support.edit(path, lambda conn: conn.execute("UPDATE meta SET value = ? WHERE key = 'cross_pending'",
                                                 (p.wire.emit([record]),)))
    status = _look(p, given)
    assert (status.status, status.reason) == ("unreadable", "anchor"), status
    edited = support.sha256_file(path)
    entries = len(given["audit"].entries)
    resumer = support.store(p, given)
    with pytest.raises(p.store.StoreRefused) as info:
        resumer.resume(transport=support.cli(p), confirm=(status.offer.kept_seq, status.offer.discarded))
    resumer.close()
    assert not isinstance(info.value, p.store.ChainRefused) and info.value.code == "local", info.value
    assert "secret" in str(info.value), str(info.value)
    assert support.sha256_file(path) == edited, "a refused resume writes nothing"
    assert len(given["audit"].entries) == entries, "and anchors nothing"

    # (2) A stray key row holding the secret's bytes: the resume drops the row and records nothing of the secret.
    path.write_bytes(clean)
    stray = "payload:" + "ab" * 16
    support.edit(path, lambda conn: conn.execute("INSERT INTO keys (name, key) VALUES (?, ?)", (stray, bytes(secret))))
    status = _look(p, given)
    assert (status.status, status.reason) == ("unreadable", "anchor"), status
    resumer = support.store(p, given)
    resumer.resume(transport=support.cli(p), confirm=(status.offer.kept_seq, status.offer.discarded))
    resumer.close()
    assert support.read(path, "SELECT name FROM keys ORDER BY name") == [("being_secret",)]
    latest = _anchor_records(p, given, being_tag)[-1]
    assert latest["why"] == "resume" and secret_fp not in latest["destroyed"], latest
    assert _refusal(p, given) is None, "a fresh process opens the resumed being"

# ---------------------------------------------------------------------------
# AJ13 -- every recorded destruction is still in effect
# ---------------------------------------------------------------------------
def test_aj13_every_recorded_destruction_is_still_in_effect_when_the_rest_verifies(p, tmp_path):
    cli = support.cli(p)
    hook = p.membrane.Transport("light_hook")
    given = support.seams(p, tmp_path, suite="aj13")
    target = support.store(p, given)
    being = support.sow(p, target, rhythm_consent=True)
    path = support.store_path(p, given)
    being.rhythm_put(hook, 7, True)
    being.rhythm_put(hook, 8, True)
    [(rhythm_key,)] = support.read(path, "SELECT key FROM keys WHERE name = 'rhythm'")
    assert isinstance(being.append("forget_rhythm", {}, transport=cli), p.membrane.Appended)
    being.heard_note(0, "hello", {"turn": 1})
    being.heard_end_season(0)
    taught = being.append("lang_teach", {}, transport=cli, payload="leek")
    [(taught_body,)] = support.read(path, "SELECT body FROM bodies WHERE eid = ?", (taught.eid,))
    forgot = being.append("lang_forget", {"target": taught.eid}, transport=cli)
    target.close()
    head = forgot.seq
    assert support.read(path, "SELECT MAX(seq) FROM links") == [(head,)]
    clean = path.read_bytes()

    def rewritten(change, seq=head):
        """The clean store changed, and its anchor rewritten under the right key at the same generation."""
        path.write_bytes(clean)
        support.edit(path, change)
        support.rewrite_anchor(p, path, seq=seq, key_id=support.KEY_ID, anchor_key=support.KEY)
        return _refusal(p, given)

    assert rewritten(lambda conn: None) is None, "witness: a rewritten anchor over the untouched store opens"

    def key_back(conn):
        conn.execute("INSERT INTO keys (name, key) VALUES ('rhythm', ?)", (bytes(rhythm_key),))

    def floor_lowered(conn):
        conn.execute("UPDATE meta SET value = ? WHERE key = 'rhythm_floor'", (p.wire.emit(0),))

    def season_reopened(conn):
        conn.execute("UPDATE meta SET value = ? WHERE key = 'heard_ended'", (p.wire.emit([]),))

    def body_back(conn):
        conn.execute("DELETE FROM links WHERE seq = ?", (head,))
        conn.execute("DELETE FROM bodies WHERE eid = ?", (forgot.eid,))
        conn.execute("DELETE FROM facts WHERE eid = ?", (forgot.eid,))
        conn.execute("UPDATE bodies SET body = ?, redacted_by = NULL WHERE eid = ?", (bytes(taught_body), taught.eid))

    for name, change, seq in (("a destroyed key back", key_back, head), ("the rhythm floor lowered", floor_lowered, head),
                              ("an ended season reopened", season_reopened, head),
                              ("a forgotten body back", body_back, head - 1)):
        refusal = rewritten(change, seq)
        assert isinstance(refusal, p.store.ChainRefused) and refusal.reason == "destroyed", (name, refusal)


# ---------------------------------------------------------------------------
# AJ14 -- a resume after a rollback jumps past the audit
# ---------------------------------------------------------------------------
class _Refusing(support.MemoryAudit):
    """The audit stand-in, refusing every write while ``refusing`` is set, as a full disk would."""

    def __init__(self):
        super().__init__()
        self.refusing = False

    def append_event(self, event_type, **kwargs):
        if self.refusing:
            raise OSError("the audit log refuses the write")
        return super().append_event(event_type, **kwargs)


def _meta(p, path, key):
    return p.wire.parse(bytes(support.read(path, "SELECT value FROM meta WHERE key = ?", (key,))[0][0]))


def test_aj14_a_resume_after_a_rollback_jumps_past_every_anchor_even_when_its_own_is_refused(p, tmp_path):
    cli = support.cli(p)
    audit = _Refusing()
    given = support.seams(p, tmp_path, suite="aj14", audit=audit)
    target = support.store(p, given)
    being = support.sow(p, target)
    support.build(being, 3, cli)
    being_tag = being.being_tag
    target.close()
    path = support.store_path(p, given)
    copy_a = path.read_bytes()
    given["clock"].advance_days(1)
    target = support.store(p, given)
    being = target.open("local")
    support.build(being, 2, cli)
    taught = being.append("lang_teach", {}, transport=cli, payload="leek")
    forgot = being.append("lang_forget", {"target": taught.eid}, transport=cli)
    target.close()
    anchored = {record["seq"]: record for record in _anchor_records(p, given, being_tag)}
    assert sorted(anchored) == [1, 4, forgot.seq], sorted(anchored)

    path.write_bytes(copy_a)
    gen_a = _meta(p, path, "gen")
    assert anchored[forgot.seq]["gen"] > gen_a + 1, "witness: the audit's generation is past the copy's next one"
    status = _look(p, given)
    assert (status.status, status.reason) == ("unreadable", "older"), status
    assert (status.offer.kept_seq, status.offer.discarded) == (3, forgot.seq - 3), status.offer

    # The resume's own anchor is refused: the being reopens alive all the same, owing it.
    audit.refusing = True
    entries = len(audit.entries)
    resumer = support.store(p, given)
    assert resumer.resume(transport=cli, confirm=(status.offer.kept_seq, status.offer.discarded)) is not None
    resumer.close()
    assert len(audit.entries) == entries, "witness: nothing reached the audit"
    assert _meta(p, path, "gen") > anchored[forgot.seq]["gen"]
    status = _look(p, given)
    assert status.status == "alive" and "anchor_owed" in status.hints, status

    # The next write settles what is owed.
    audit.refusing = False
    target = support.store(p, given)
    written = target.open("local").append("act", {"act": "warm"}, transport=cli)
    target.close()
    status = _look(p, given)
    assert status.status == "alive" and "anchor_owed" not in status.hints, status
    assert _meta(p, path, "cross_pending") == []
    assert _anchor_records(p, given, being_tag)[-1]["seq"] == written.seq


# ---------------------------------------------------------------------------
# AJ15 -- a resume acts only on what it verified, and enforces pending records
# ---------------------------------------------------------------------------
def test_aj15_a_resume_acts_only_on_the_state_it_verified_and_enforces_pending_records(p, tmp_path):
    cli = support.cli(p)

    # (1) Another resume lands between this one's verification and its transaction.
    given = support.seams(p, tmp_path.joinpath("race"), suite="aj15")
    target = support.store(p, given)
    support.build(support.sow(p, target), 9, cli)
    target.close()
    path = support.store_path(p, given)
    links = _links(path)
    support.edit(path, lambda conn: conn.execute("UPDATE bodies SET body = ? WHERE eid = ?",
                                                 (b'{"act":"none"}', links[5][2])))
    status = _look(p, given)
    assert (status.status, status.reason) == ("unreadable", "body"), status
    confirm = (status.offer.kept_seq, status.offer.discarded)
    landed = []

    def another_resume(name):
        if name == "resume" and not landed:
            other = support.store(p, given)
            landed.append(other.resume(transport=cli, confirm=confirm) is not None)
            other.close()

    racer = support.store(p, dict(given, stage=another_resume))
    with pytest.raises(p.store.ResumeRefused) as info:
        racer.resume(transport=cli, confirm=confirm)
    racer.close()
    assert landed == [True], "witness: the other resume landed in between"
    assert info.value.code == "confirm", info.value
    kinds = [row[0] for row in support.read(path, "SELECT f.kind FROM links l JOIN facts f ON f.eid = l.eid "
                                                  "ORDER BY l.seq")]
    assert kinds.count("resumed") == 1, kinds
    assert _refusal(p, given) is None

    # (2) A forget whose record is still pending, cut from the tail, and its word put back by a tool.
    audit = _Refusing()
    given = support.seams(p, tmp_path.joinpath("pending"), suite="aj15", index=1, audit=audit)
    target = support.store(p, given)
    being = support.sow(p, target)
    word = support.canary(p, "aj15")
    taught = being.append("lang_teach", {}, transport=cli, payload=word)
    path = support.store_path(p, given)
    [(body,)] = support.read(path, "SELECT body FROM bodies WHERE eid = ?", (taught.eid,))
    [(ref, ct)] = support.read(path, "SELECT ref, ct FROM payloads")
    [(key,)] = support.read(path, "SELECT key FROM keys WHERE name = ?", ("payload:" + ref,))
    audit.refusing = True
    forgot = being.append("lang_forget", {"target": taught.eid}, transport=cli)
    target.close()
    audit.refusing = False
    assert [item["forgot"] for item in _meta(p, path, "cross_pending")] == [[taught.eid]], "witness: only pending"

    def put_back(conn):
        conn.execute("DELETE FROM links WHERE seq = ?", (forgot.seq,))
        conn.execute("DELETE FROM bodies WHERE eid = ?", (forgot.eid,))
        conn.execute("DELETE FROM facts WHERE eid = ?", (forgot.eid,))
        conn.execute("UPDATE bodies SET body = ?, redacted_by = NULL WHERE eid = ?", (bytes(body), taught.eid))
        conn.execute("INSERT INTO payloads (ref, eid, ct) VALUES (?, ?, ?)", (ref, taught.eid, bytes(ct)))
        conn.execute("INSERT INTO keys (name, key) VALUES (?, ?)", ("payload:" + ref, bytes(key)))
    support.edit(path, put_back)
    status = _look(p, given)
    assert (status.status, status.reason) == ("unreadable", "truncated"), status
    resumer = support.store(p, given)
    reopened = resumer.resume(transport=cli, confirm=(status.offer.kept_seq, status.offer.discarded))
    with pytest.raises(p.membrane.MembraneRefused) as info:
        reopened.unseal(ref)
    assert info.value.code == "payload", "the word the pending record forgot stays forgotten"
    resumer.close()
    assert support.read(path, "SELECT COUNT(*) FROM keys WHERE name = ?", ("payload:" + ref,)) == [(0,)]
    assert support.read(path, "SELECT body FROM bodies WHERE eid = ?", (taught.eid,)) == [(None,)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
