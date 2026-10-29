"""Contracts for a deleted skill that keeps no words, and whose deletion reaches the paired devices.

Deleting a skill archived its final version into its history first -- one more full copy of its text --, unlinked its
file and two of its sidecars without overwriting them, and left its history, its sync mark and its folder, named by
the skill. Its tombstone named nothing, so a paired device refused it at the landing and kept the skill, serving it on;
a deletion whose tombstone was not journalled left the skill's versions in the sync journal. The review queue kept
the decided proposals that held its text, and the deferred ledger a paired device's offer of it. The wipe never
removed a skill, and the export carried none. Now a deletion forgets: the journal's versions first, the folder
overwritten with its history, the decided proposals, the offers its tombstone outclocks; the tombstone names the
skill, and a paired device lands it -- one from a build before this by its key -- and forgets the skill the same way.

  * SK1 -- A DELETED SKILL KEEPS NO WORDS: every file under the data folder read back, the journal, the review queue
    and the ledger included; a pending proposal and a newer offer, listed for a decision, are kept, and an offer at
    the tombstone's clock is judged as the merge would; a draft's deletion journals nothing.
  * SK2 -- ITS TOMBSTONE NAMES IT, AND A PAIRED DEVICE FORGETS IT: through the real engine's round; a tombstone from a
    build before this lands by its key; one whose key names no skill lands nowhere; one that finds an offer waiting
    takes its place, the offer's words gone, and lands when approved.
  * SK3 -- WHAT DELETIONS BEFORE THIS LEFT IS SWEPT by the next one -- a dead write's leftover, a proposal decided
    after its skill's deletion --; a write in progress is left alone (it holds the registry's lock while it makes its
    folder), and what is no deletion's residue too: a linked category, a folder of the user's own; a tombstone the
    journal did not take is said, owed, and published by the next deletion, the offers its floor outclocks forgotten.
  * SK4 -- A STEP THAT FAILS IS NAMED: a history that cannot be removed, or a journal that is there and cannot be
    written or loaded, refuses, the skill listed with its text, and deleting it again finishes it; a queue or a
    ledger that cannot be reached is named, and the deletion goes on.
  * SK5 -- A STALE VERSION AFTER A DELETION LOSES: the tombstone, or the floor a deletion left when its tombstone was
    not taken; a newer one lands.
  * SK6 -- THE WIPE WITH THE SKILLS NAMED: every skill and draft through its deletion, the offers cleared, a skill that
    refuses named.
  * SK7 -- EVERY REFUSAL BEFORE ANYTHING IS REMOVED.
  * SK8 -- THE EXPORT CARRIES EVERY SKILL AND ITS HISTORY.
  * SK9 -- THE REQUEST'S WORDS ARE FOUR, and failures carry the inventory's labels (kb11 of
    tests/test_knowledge_forgetting_contracts.py, which held them three, re-asserted unchanged with the fourth).
  * SK10 -- THE PANELS AND THE ROUTE: the "Your data" panel offers the skills apart; the route and the skills panel
    say what a deletion could not finish, and a refusal is a 503.
  * SK11 -- THE OVERWRITE IS ONE: its five copies are the same text (pf13 of tests/test_project_forgetting_contracts.py,
    which counted four, re-asserted unchanged).

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import ast
import inspect
import json
import sqlite3
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_sk1_a_deleted_skill_keeps_no_words": 10.0,
    "test_sk2_its_tombstone_names_it_and_a_paired_device_forgets_it": 15.0,
    "test_sk3_what_deletions_before_this_left_is_swept_and_an_owed_tombstone_published": 12.0,
    "test_sk4_a_step_that_fails_is_named_and_the_skill_kept_while_its_text_stays": 10.0,
    "test_sk5_a_stale_version_after_a_deletion_loses": 15.0,
    "test_sk6_the_wipe_with_the_skills_named": 15.0,
    "test_sk7_every_refusal_before_anything_is_removed": 10.0,
    "test_sk8_the_export_carries_every_skill_and_its_history": 10.0,
    "test_sk9_the_requests_words_are_four_and_failures_carry_the_inventorys_labels": 15.0,
    "test_sk10_the_panels_and_the_route_say_what_a_deletion_left": 15.0,
    "test_sk11_the_overwrite_is_one_text_in_its_five_copies": 15.0,
}

BODY, OLD, DECLINED, SETTLED = "ZANZIBAR-STEP-9", "ZANZIBAR-OLD-9", "ZANZIBAR-DECLINED-9", "ZANZIBAR-SETTLED-9"
PENDING, CLAIMED, OFFER, TEMP = "ZANZIBAR-PENDING-9", "ZANZIBAR-CLAIMED-9", "ZANZIBAR-OFFER-9", "ZANZIBAR-TEMP-9"
KEPT, DRAFT = "KEPT-SKILL-9", "DRAFT-SKILL-9"
_GONE = tuple(word.encode() for word in (BODY, OLD, DECLINED, SETTLED, OFFER, TEMP))
_SKILLS_LABEL = "skills (written by the user or accepted from the agent)"
_OFFERS_LABEL = "skills a paired device offered (waiting for this device's approval)"
_VEILID = ("records", "reconcile", "change_feed", "signing", "deferred_ledger", "peers", "producers", "protocol",
           "review", "sync_engine")
_UPDATED = "2026-10-29T12:00:00+00:00"


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _plain(path, *, check_same_thread=False, **_options):
    return sqlite3.connect(str(path), check_same_thread=check_same_thread, timeout=0.3)


def _held(data, *words):
    """Which of ``words`` any file under ``data`` still holds."""
    found = set()
    for path in data.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            found.update(w for w in words if w in raw)
    return found


def _body(word):
    return f"## When to Use\nShipping {word}.\n\n## Procedure\n{word} first, then announce.\n"


class _Signer:
    """A device with no signing backend: records are journalled unsigned and accepted unverified."""

    def verify_available(self):
        return False

    def public_key(self):
        return b""

    def sign(self, _data):
        raise RuntimeError("no signing backend")


class _Peer:
    """The paired device: its journal as a list of versions, served from the watermark asked, under its epoch."""

    def __init__(self, protocol, records, versions=(), epoch="epoch-1"):
        self.protocol, self.records = protocol, records
        self.versions, self.epoch = list(versions), epoch

    def fetch(self, request):
        cursor = int(request["watermark"])
        chunk = self.versions[cursor:]
        return {"v": self.protocol.PROTOCOL_VERSION, "type": self.protocol.MSG_RECORD_BATCH, "device": "dev-b",
                "high_water": len(self.versions) if chunk else cursor,
                "records": self.records.encode_records(chunk), "epoch": self.epoch}


def _world(tmp_path, *, wipe=False, single=True):
    """This device over ``tmp_path/data``: the skills' registry, the review queue, the sync engine with its journal, its
    paired set and its deferred ledger, each from its source; with ``wipe``, the account wipe and the inventory, one
    account ``u-1``, single-user or not."""
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    audit = []

    def effective_user_id(user_id, single_user_mode=True):
        return "local" if (single_user_mode or user_id is None) else user_id

    seeded = {
        "opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain),
        "opti_oignon.db_encryption": _module("opti_oignon.db_encryption", get_encrypted_connection=_plain,
                                             SQLCIPHER_AVAILABLE=False),
        "opti_oignon.config": _module("opti_oignon.config", DATA_DIR=data),
        "opti_oignon.user_isolation": _module("opti_oignon.user_isolation", DEFAULT_LOCAL_USER="local",
                                              effective_user_id=effective_user_id),
        "opti_oignon.veilid.guard": _module("opti_oignon.veilid.guard", veilid_available=lambda: True,
                                            assert_sync_allowed=lambda: None),
        "opti_oignon.signed_audit_log": _module("opti_oignon.signed_audit_log",
                                                chain_log=lambda **details: audit.append(details)),
    }
    targets = {f"opti_oignon.veilid.{name}": source("veilid", f"{name}.py") for name in _VEILID}
    targets.update({
        "opti_oignon.agent.skills": source("agent", "skills.py"),
        "opti_oignon.pending_writes": source("pending_writes.py"),
    })
    if wipe:
        seeded["opti_oignon.run_journal"] = _module("opti_oignon.run_journal", wipe=lambda: {"wiped": True})
        seeded["opti_oignon.auth"] = _module("opti_oignon.auth", auth_manager=SimpleNamespace(
            single_user_mode=single, count_users=lambda: 1,
            get_user=lambda uid: SimpleNamespace(id=uid) if uid == "u-1" else None))
        targets["opti_oignon.user_data_manager"] = source("user_data_manager.py")
        targets["opti_oignon.store_inventory"] = source("store_inventory.py")
    loaded, restore = isolate(targets=targets, seeded=seeded, packages=("opti_oignon.veilid", "opti_oignon.agent"))
    try:
        mod = {name.rsplit(".", 1)[-1]: module for name, module in loaded.items()}
        feed_module, engine_module, ledger_module = mod["change_feed"], mod["sync_engine"], mod["deferred_ledger"]
        feed = feed_module.ChangeFeed(root=data)
        feed_module.set_change_feed(feed)
        ledger = ledger_module.DeferredLedger(root=data)
        ledger_module.set_deferred_ledger(ledger)
        peers = mod["peers"].PeerStore(root=data)
        engine = engine_module.SyncEngine(device="dev-a", feed=feed, store=peers, signer=_Signer(), ledger=ledger)
        engine_module.set_sync_engine(engine)
        engine.register_peer("dev-b", "route-b")
        sk = mod["skills"]
        registry = sk.SkillRegistry(data / "skills")
        sk._REGISTRY = registry
        pw = mod["pending_writes"]
        queue = pw.get_pending_store()
    except BaseException:
        restore()
        raise

    def close():
        try:
            engine_module.set_sync_engine(None)
            feed_module.reset_change_feed()
            ledger_module.set_deferred_ledger(None)
            ledger.close()
            sk._REGISTRY = None
            pw._QUEUE = None
        finally:
            restore()

    return SimpleNamespace(data=data, mod=mod, sk=sk, skills=registry, pw=pw, queue=queue, feed=feed, ledger=ledger,
                           dl=ledger_module, engine=engine, audit=audit, udm=mod.get("user_data_manager"),
                           inventory=mod.get("store_inventory"), restore=close)


def _plant(world, name="zanzibar", category="ops", *, proposals=True):
    """A published skill holding words everywhere the registry keeps them -- its file, the version its history
    archived, its counter, its sync mark and approval, a write's leftover --, its versions in the journal, and, with
    ``proposals``, its decided proposals in the review queue (declined; accepted and written), one pending and one
    accepted whose write is under way; its key."""
    sk, reg = world.sk, world.skills
    reg.add(name, category, _body(OLD), source=sk.SOURCE_MANUAL, status=sk.STATUS_PUBLISHED)
    reg.update(name, category, body=_body(BODY))
    folder = world.data / "skills" / category / name
    reg.increment_usage(name, category)
    reg._write_origin(folder / sk.ORIGIN_FILENAME, ["0" * 64])
    (folder / sk.APPROVED_FILENAME).write_text(json.dumps({"approved": ["0" * 64]}), encoding="utf-8")
    (folder / f".{sk.SKILL_FILENAME}.1.1.tmp").write_bytes(f"{TEMP} a write that died".encode())
    if proposals:
        def propose(word):
            text = _body(word)
            arguments = {"name": name, "category": category, "text": text, "sha256": sk.text_digest(text),
                         "base_sha256": "0" * 64, "source": "agent"}
            return world.queue.propose("skills", "edit", arguments, {"via": "the agent"})[0]

        world.queue.decline(propose(DECLINED).id)
        settled = propose(SETTLED)
        world.queue.claim(settled.id)
        world.queue.settle(settled.id, "published")
        propose(PENDING)
        world.queue.claim(propose(CLAIMED).id)
    return sk._skill_sync_key(category, name)


def _offer(world, key, clock, word):
    """A paired device's version of the skill, deferred for this device's approval."""
    records = world.mod["records"]
    category, name = key.split("/")
    record = records.new_record(records.RecordKind.SKILL, key,
                                {"user_id": "local", "skill": {"category": category, "name": name,
                                                               "markdown": _body(word)}},
                                device="dev-b", clock=clock, updated_at=_UPDATED)
    world.ledger.offer(record, peer_id="dev-b")


def _journalled(world, key):
    """The versions the journal holds of a skill."""
    return [r for r in world.feed.current_records() if r.kind.value == "skill" and r.record_id == key]


def _delete(world, name="zanzibar", category="ops", *, draft=False):
    """The user deletes the skill by the digest of its text shown; what stays, named (a build before this one names
    nothing: it is asked without)."""
    left = []
    current = world.skills.get(name, category, draft=draft)
    named = world.skills.delete_named
    said = {"left": left} if "left" in inspect.signature(named).parameters else {}
    assert named(name, category, draft=draft, sha256=current.digest(), **said), "control: deleted"
    return left


def _round(world, versions, epoch="epoch-1"):
    protocol, records = world.mod["protocol"], world.mod["records"]
    return world.engine.run_round("dev-b", _Peer(protocol, records, versions, epoch=epoch),
                                  approval_fn=lambda _conv, _label, _args: True)


# ---------------------------------------------------------------------------
# SK1 -- a deleted skill keeps no words
# ---------------------------------------------------------------------------
def test_sk1_a_deleted_skill_keeps_no_words(tmp_path, monkeypatch):
    world = _world(tmp_path)
    try:
        key = _plant(world)
        world.skills.add("kept", "ops", _body(KEPT), source=world.sk.SOURCE_MANUAL,
                         status=world.sk.STATUS_PUBLISHED)
        _offer(world, key, 1, OFFER)
        assert _held(world.data, *_GONE) == set(_GONE), "control: the words are everywhere the skill keeps them"
        left = _delete(world)
        assert left == [], f"a deletion that finished names something: {left}"
        assert not (world.data / "skills" / "ops" / "zanzibar").exists(), "the skill's folder stayed"
        assert _held(world.data, *_GONE) == set(), f"words of the deleted skill stayed: {_held(world.data, *_GONE)}"
        assert world.skills.get("kept", "ops") is not None and _held(world.data, KEPT.encode()), "another skill went"
        # Listed for a decision: the proposal pending, and the acceptance whose write is under way.
        assert _held(world.data, PENDING.encode(), CLAIMED.encode()) == {PENDING.encode(), CLAIMED.encode()}
        assert world.ledger.get("skill", key) is None, "the offer the tombstone outclocks stayed"
        (tomb,) = _journalled(world, key)
        assert tomb.deleted and tomb.payload["skill"] == {"category": "ops", "name": "zanzibar"}, tomb.payload
        assert [a["action"] for a in world.audit if a.get("name") == "zanzibar"][-1] == "delete"
        # A paired device's version newer than the deletion is listed for approval: kept.
        _plant(world, proposals=False)
        _offer(world, key, 99, "NEWER-OFFER-9")
        assert _delete(world) == []
        assert world.ledger.get("skill", key) is not None and _held(world.data, b"NEWER-OFFER-9")
        # Its own deletion forgets its decided proposals, with no sweep after it.
        _plant(world)
        assert world.skills.forget_skill("zanzibar", "ops", sweep=False) and not _held(
            world.data, DECLINED.encode(), SETTLED.encode()), "a deletion left its decided proposals to a sweep"
        # A draft's deletion is device-local: its folder goes, nothing is journalled; with no sweep after it, its folder
        # and its emptied category go all the same.
        world.skills.add("drafty", "misc", _body(DRAFT))
        before = len(world.feed.current_records())
        assert _delete(world, "drafty", "misc", draft=True) == []
        assert not (world.data / "skills" / ".drafts" / "misc").exists() and not _held(world.data, DRAFT.encode())
        assert len(world.feed.current_records()) == before, "a draft's deletion journalled something"
        world.skills.add("drafty", "misc", _body(DRAFT))
        left = []
        assert world.skills.forget_skill("drafty", "misc", draft=True, sweep=False, left=left) and left == []
        assert not (world.data / "skills" / ".drafts" / "misc").exists(), "a deletion left its folder to a sweep"
        assert world.skills.delete("zanzibar", "ops") is False, "a skill deleted twice"
        # A registry rooted in another folder asks none of the stores beside this one.
        _plant(world)
        other = world.sk.SkillRegistry(tmp_path / "elsewhere" / "skills")
        other.add("zanzibar", "ops", _body("ELSEWHERE-9"), source=world.sk.SOURCE_MANUAL,
                  status=world.sk.STATUS_PUBLISHED)
        assert other.forget_skill("zanzibar", "ops") and _held(world.data, DECLINED.encode(), SETTLED.encode()) == {
            DECLINED.encode(), SETTLED.encode()}, "a registry of another folder forgot this one's proposals"
    finally:
        world.restore()
    # An offer at the tombstone's own clock: gone when the tombstone beats it, kept when it beats the tombstone.
    for beaten in (True, False):
        world = _world(tmp_path / f"tie-{beaten}")
        try:
            key = _plant(world, proposals=False)
            sk, records, reconcile = world.sk, world.mod["records"], world.mod["reconcile"]
            monkeypatch.setattr(sk, "_now", lambda: _UPDATED)
            tomb = records.new_record(records.RecordKind.SKILL, key, sk._tombstone_payload("ops", "zanzibar"),
                                      device="dev-a", clock=3, deleted=True, updated_at=_UPDATED)
            for salt in range(200):
                offer = records.new_record(records.RecordKind.SKILL, key,
                                           {"user_id": "local", "skill": {"category": "ops", "name": "zanzibar",
                                                                          "markdown": _body(f"TIE-{salt}")}},
                                           device="dev-b", clock=3, updated_at=_UPDATED)
                if (reconcile.choose_winner([offer, tomb]) is tomb) == beaten:
                    break
            else:
                raise AssertionError("control: no offer found on that side of the tie")
            world.ledger.offer(offer, peer_id="dev-b")
            _delete(world)
            (held,) = _journalled(world, key)
            assert held.content_hash == tomb.content_hash, "control: the tombstone judged is the one journalled"
            assert (world.ledger.get("skill", key) is None) is beaten, (
                "an offer the tombstone beats stayed" if beaten else "an offer that beats the tombstone went")
            assert bool(_held(world.data, f"TIE-{salt}".encode())) is not beaten, "the files say otherwise"
        finally:
            monkeypatch.undo()
            world.restore()
    # A device that never synced: its deletion makes no journal, nor a ledger, to be asked.
    world = _world(tmp_path / "unsynced")
    try:
        monkeypatch.setattr(sys.modules["opti_oignon.veilid.guard"], "veilid_available", lambda: False)
        _plant(world, proposals=False)
        assert _delete(world) == [] and world.skills.get("zanzibar", "ops") is None
        assert not (world.data / "veilid_change_feed.db").exists(), "a deletion made the sync journal"
        assert not (world.data / "veilid_deferred.db").exists(), "a deletion made the deferred ledger"
    finally:
        monkeypatch.undo()
        world.restore()


# ---------------------------------------------------------------------------
# SK2 -- its tombstone names it, and a paired device forgets it
# ---------------------------------------------------------------------------
def test_sk2_its_tombstone_names_it_and_a_paired_device_forgets_it(tmp_path):
    there = _world(tmp_path / "there")
    try:
        key = _plant(there, proposals=False)
        _delete(there)
        (sent,) = _journalled(there, key)
        payload, clock = dict(sent.payload), sent.clock
    finally:
        there.restore()
    assert payload == {"user_id": "local", "skill": {"category": "ops", "name": "zanzibar"}}, payload
    for case, carried in (("named", payload), ("from a build before this", {})):
        world = _world(tmp_path / case.replace(" ", "-"))
        try:
            assert _plant(world) == key
            assert world.skills.get("zanzibar", "ops") is not None, "control: the skill is here"
            records = world.mod["records"]
            tomb = records.new_record(records.RecordKind.SKILL, key, carried, device="dev-b", clock=clock,
                                      deleted=True, updated_at=_UPDATED)
            _round(world, [tomb])
            assert world.skills.get("zanzibar", "ops") is None, f"{case}: the paired device kept the skill"
            assert not (world.data / "skills" / "ops" / "zanzibar").exists(), f"{case}: the folder stayed"
            assert _held(world.data, *_GONE) == set(), f"{case}: words stayed: {_held(world.data, *_GONE)}"
            assert _held(world.data, PENDING.encode()), f"{case}: a pending proposal went"
            (journalled,) = _journalled(world, key)
            assert journalled.deleted, f"{case}: the tombstone was not journalled"
            # A tombstone whose key names no skill lands nowhere, and the round goes on.
            stray = records.new_record(records.RecordKind.SKILL, "Ops/../x", {}, device="dev-b", clock=1,
                                       deleted=True, updated_at=_UPDATED)
            _round(world, [tomb, stray])
            assert _journalled(world, "Ops/../x") == [], f"{case}: a key that names no skill was journalled"
        finally:
            world.restore()
    # An offer waiting for approval holds the tombstone with it: the tombstone takes its place, the offer's words
    # gone from the ledger's files, and the approval lands it as a round does.
    world = _world(tmp_path / "waiting")
    try:
        _plant(world, proposals=False)
        _offer(world, key, 1, OFFER)
        records = world.mod["records"]
        tomb = records.new_record(records.RecordKind.SKILL, key, payload, device="dev-b", clock=clock, deleted=True,
                                  updated_at=_UPDATED)
        result = _round(world, [tomb])
        assert result.applied == 0 and world.skills.get("zanzibar", "ops") is not None, "control: held for approval"
        assert world.ledger.get("skill", key).clock == clock and not _held(world.data, OFFER.encode()), (
            "the offer the tombstone replaced kept its words")
        world.engine.approve_deferred("skill", key)
        assert world.skills.get("zanzibar", "ops") is None and _held(world.data, *_GONE) == set(), (
            "an approved tombstone left the skill")
        (journalled,) = _journalled(world, key)
        assert journalled.deleted and world.ledger.get("skill", key) is None
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SK3 -- what deletions before this left is swept; an owed tombstone published
# ---------------------------------------------------------------------------
def test_sk3_what_deletions_before_this_left_is_swept_and_an_owed_tombstone_published(tmp_path, monkeypatch):
    world = _world(tmp_path)
    try:
        skills = world.data / "skills"
        gone = skills / "ops" / "old-gone"
        (gone / ".versions").mkdir(parents=True)
        (gone / ".versions" / "v1.md").write_text(_body(OLD), encoding="utf-8")
        (gone / "_origin.json").write_text('{"origin": "sync", "adopted": []}', encoding="utf-8")
        draft = skills / ".drafts" / "ops" / "old-draft"
        draft.mkdir(parents=True)
        (draft / "_usage.json").write_text('{"uses": 2}', encoding="utf-8")
        dead = skills / "misc" / "died-writing"
        dead.mkdir(parents=True)
        (dead / ".SKILL.md.7.7.tmp").write_bytes(b"DIED-WRITING-9")
        world.skills.add("other", "misc", _body(KEPT), source=world.sk.SOURCE_MANUAL,
                         status=world.sk.STATUS_PUBLISHED)
        assert _delete(world, "other", "misc") == []
        assert not gone.exists() and not draft.exists() and not _held(world.data, OLD.encode()), (
            "what a deletion before this left was not swept")
        assert not dead.exists() and not _held(world.data, b"DIED-WRITING-9"), "a dead write's leftover stayed"
        # A write holds the registry's lock while it makes its folder: a sweep waits for it, the folder kept.
        sk, made, go = world.sk, threading.Event(), threading.Event()
        real_write = sk._write_atomic

        def slow(path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            made.set()
            go.wait(5)
            real_write(path, data)

        monkeypatch.setattr(sk, "_write_atomic", slow)
        writer = threading.Thread(target=lambda: world.skills.add("racing", "misc", _body(KEPT),
                                                                  source=sk.SOURCE_MANUAL,
                                                                  status=sk.STATUS_PUBLISHED), daemon=True)
        writer.start()
        assert made.wait(5), "control: the write made its folder"
        sweeper = threading.Thread(target=world.skills.sweep_orphans, daemon=True)
        sweeper.start()
        sweeper.join(0.3)
        waited = sweeper.is_alive()
        go.set()
        writer.join(5)
        sweeper.join(5)
        monkeypatch.setattr(sk, "_write_atomic", real_write)
        assert waited and world.skills.get("racing", "misc") is not None, "a sweep took the folder a write was making"
        # The journal does not take the tombstone: said, owed, the words gone all the same.
        key = _plant(world, proposals=False)
        real = world.engine.publish_skill

        def refused(*args, **kwargs):
            raise RuntimeError("the journal refused")

        monkeypatch.setattr(world.engine, "publish_skill", refused)
        _offer(world, key, 2, OFFER)
        left = _delete(world)
        assert left == ["paired devices (not told of the skill 'ops/zanzibar''s deletion: the sync journal did not "
                        "take its tombstone; the next deletion or wipe tells them)"], left
        assert world.feed.owed("skill") == [key] and _journalled(world, key) == []
        assert world.ledger.get("skill", key) is None, "an offer at the floor the deletion left stayed"
        assert _held(world.data, *_GONE) == set(), "a deletion its peers were not told of kept words"
        monkeypatch.setattr(world.engine, "publish_skill", real)
        world.skills.add("third", "misc", _body(KEPT), source=world.sk.SOURCE_MANUAL,
                         status=world.sk.STATUS_PUBLISHED)
        assert _delete(world, "third", "misc") == []
        (tomb,) = _journalled(world, key)
        assert tomb.deleted and tomb.payload["skill"] == {"category": "ops", "name": "zanzibar"}, tomb.payload
        assert world.feed.owed("skill") == []
        # What is no deletion's residue stays: a category linked from elsewhere, a folder of the user's own.
        outside = tmp_path / "outside" / "old" / ".versions"
        outside.mkdir(parents=True)
        (outside / "v1.md").write_bytes(b"OUTSIDE-WORDS-9")
        (skills / "linked").symlink_to(tmp_path / "outside")
        mine = skills / "ops" / "mine"
        (mine / "references").mkdir(parents=True)
        (mine / "Skill.md").write_bytes(b"MINE-WORDS-9")
        (mine / "references" / "guide.md").write_bytes(b"MINE-GUIDE-9")
        # A proposal kept for a decision at the deletion, declined after it; a declined proposal of a new skill.
        _plant(world, proposals=False)
        text = _body(PENDING)
        arguments = {"name": "zanzibar", "category": "ops", "text": text, "sha256": world.sk.text_digest(text),
                     "base_sha256": "0" * 64, "source": "agent"}
        later = world.queue.propose("skills", "edit", arguments, {"via": "the agent"})[0]
        fresh = _body("NEW-SKILL-9")
        new = world.queue.propose("skills", "add", {"name": "fresh", "category": "ops", "text": fresh,
                                                    "sha256": world.sk.text_digest(fresh), "source": "agent"},
                                  {"via": "the agent"})[0]
        world.queue.decline(new.id)
        failed_text = _body("NEW-ACCEPTED-9")
        accepted = world.queue.propose("skills", "add", {"name": "never", "category": "ops", "text": failed_text,
                                                         "sha256": world.sk.text_digest(failed_text),
                                                         "source": "agent"}, {"via": "the agent"})[0]
        world.queue.claim(accepted.id)
        world.queue.settle(accepted.id, "failed: target changed")
        linked = skills / "ops" / "linked-residue"
        (linked / ".versions").mkdir(parents=True)
        (tmp_path / "usage-elsewhere.json").write_bytes(b"ELSEWHERE-USAGE-9")
        (linked / "_usage.json").symlink_to(tmp_path / "usage-elsewhere.json")
        assert _delete(world) == [] and _held(world.data, PENDING.encode()), "control: kept for a decision"
        world.queue.decline(later.id)
        world.skills.add("fourth", "misc", _body(KEPT), source=world.sk.SOURCE_MANUAL,
                         status=world.sk.STATUS_PUBLISHED)
        assert _delete(world, "fourth", "misc") == []
        assert not _held(world.data, PENDING.encode()), "a proposal decided after its skill's deletion kept its text"
        assert _held(world.data, b"NEW-SKILL-9", b"NEW-ACCEPTED-9") == {b"NEW-SKILL-9", b"NEW-ACCEPTED-9"}, (
            "a decided proposal of a skill that never was went")
        assert (linked / "_usage.json").is_symlink() and (tmp_path / "usage-elsewhere.json").exists(), (
            "a folder holding a link was swept as residue")
        # A live skill of a category the listing hides keeps its decided proposals through another deletion's sweep.
        world.skills.add("notes", "_private", _body("PRIVATE-WORDS-9"), source=world.sk.SOURCE_MANUAL,
                         status=world.sk.STATUS_PUBLISHED)
        hidden = _body("PRIVATE-EDIT-9")
        edit = world.queue.propose("skills", "edit", {"name": "notes", "category": "_private", "text": hidden,
                                                      "sha256": world.sk.text_digest(hidden), "base_sha256": "0" * 64,
                                                      "source": "agent"}, {"via": "the agent"})[0]
        world.queue.decline(edit.id)
        world.skills.add("fifth", "misc", _body(KEPT), source=world.sk.SOURCE_MANUAL,
                         status=world.sk.STATUS_PUBLISHED)
        assert _delete(world, "fifth", "misc") == [] and _held(world.data, b"PRIVATE-EDIT-9"), (
            "a live skill's decided proposal was swept")
        assert (outside / "v1.md").read_bytes() == b"OUTSIDE-WORDS-9" and (skills / "linked").is_symlink()
        assert _held(mine, b"MINE-WORDS-9", b"MINE-GUIDE-9") == {b"MINE-WORDS-9", b"MINE-GUIDE-9"}, (
            "a folder of the user's own was swept")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SK4 -- a step that fails is named, the skill kept while its text stays
# ---------------------------------------------------------------------------
def test_sk4_a_step_that_fails_is_named_and_the_skill_kept_while_its_text_stays(tmp_path, monkeypatch):
    world = _world(tmp_path)
    try:
        sk = world.sk
        key = _plant(world)
        real = sk._forget_entry

        def stuck(path, said):
            if path.name == ".versions":
                raise PermissionError("held")
            return real(path, said)

        monkeypatch.setattr(sk, "_forget_entry", stuck)
        current = world.skills.get("zanzibar", "ops")
        with pytest.raises(sk.SkillNotForgotten, match="it stays listed, delete it again"):
            world.skills.delete_named("zanzibar", "ops", draft=False, sha256=current.digest())
        assert world.skills.get("zanzibar", "ops") is not None and _held(world.data, BODY.encode()), (
            "a deletion that refused took the skill's text")
        monkeypatch.setattr(sk, "_forget_entry", real)
        assert _delete(world) == [] and _held(world.data, *_GONE) == set(), "deleting it again did not finish it"
        # A journal that cannot be written refuses before anything is removed; deleting again, once it can, finishes.
        _plant(world, proposals=False)

        def unwritable(*args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(world.feed, "forget_record", unwritable)
        current = world.skills.get("zanzibar", "ops")
        with pytest.raises(sk.SkillNotForgotten, match="the sync journal could not erase its versions"):
            world.skills.delete_named("zanzibar", "ops", draft=False, sha256=current.digest())
        monkeypatch.undo()
        assert world.skills.get("zanzibar", "ops") is not None and _held(world.data, BODY.encode()), (
            "a deletion its journal refused took the skill's text")
        assert [r.deleted for r in _journalled(world, key)] == [False], "a refused deletion journalled a tombstone"
        # A journal that is there and that this process cannot load refuses the same way.
        monkeypatch.delattr(sys.modules["opti_oignon.veilid.change_feed"], "forget_record")
        with pytest.raises(sk.SkillNotForgotten, match="the sync journal could not erase its versions"):
            world.skills.delete_named("zanzibar", "ops", draft=False, sha256=current.digest())
        monkeypatch.undo()
        assert world.skills.get("zanzibar", "ops") is not None and _held(world.data, BODY.encode())
        assert _delete(world) == [] and _held(world.data, *_GONE) == set()
        # A queue or a ledger that cannot be reached: named, and the deletion goes on.
        cases = (
            (world.pw, "forget_decided", "pending writes (the skill's decided proposals not removed: RuntimeError; "
                                         "the wipe removes them)"),
            (world.dl, "forget_offers", f"{_OFFERS_LABEL} (not removed: RuntimeError; the wipe removes them)"),
        )
        for owner, name, said in cases:
            _plant(world, proposals=owner is world.pw)
            if owner is world.dl:
                _offer(world, key, 1, OFFER)

            def broken(*args, **kwargs):
                raise RuntimeError("not reachable")

            monkeypatch.setattr(owner, name, broken)
            left = _delete(world)
            monkeypatch.undo()
            assert left == [said], (name, left)
            assert world.skills.get("zanzibar", "ops") is None, name
        (tomb,) = _journalled(world, key)
        assert tomb.deleted and not _held(world.data, BODY.encode(), OLD.encode())
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SK5 -- a stale version after a deletion loses
# ---------------------------------------------------------------------------
def test_sk5_a_stale_version_after_a_deletion_loses(tmp_path, monkeypatch):
    for case in ("tombstone", "floor"):
        world = _world(tmp_path / case)
        try:
            key = _plant(world, proposals=False)
            (live,) = [r for r in _journalled(world, key) if not r.deleted]
            if case == "floor":
                monkeypatch.setattr(world.engine, "publish_skill",
                                    lambda *a, **k: (_ for _ in ()).throw(RuntimeError("refused")))
            _delete(world)
            monkeypatch.undo()
            _round(world, [live])
            assert world.skills.get("zanzibar", "ops") is None, f"{case}: a stale version brought the skill back"
            assert not _held(world.data, BODY.encode(), OLD.encode()), f"{case}: a stale version wrote its words"
            held = _journalled(world, key)
            assert all(r.deleted for r in held) and (held if case == "tombstone" else not held), (case, held)
            records = world.mod["records"]
            newer = records.new_record(records.RecordKind.SKILL, key, live.payload, device="dev-b",
                                       clock=live.clock + 5, updated_at=_UPDATED)
            _round(world, [live, newer], epoch="epoch-2")
            assert world.skills.get("zanzibar", "ops") is not None, f"{case}: a newer version did not land"
        finally:
            world.restore()


# ---------------------------------------------------------------------------
# SK6 -- the wipe with the skills named
# ---------------------------------------------------------------------------
def _wipe_plant(world):
    key = _plant(world)
    sk = world.sk
    world.skills.add("kept", "ops", _body(KEPT), source=sk.SOURCE_MANUAL, status=sk.STATUS_PUBLISHED)
    world.skills.add("drafty", "misc", _body(DRAFT))
    gone = world.data / "skills" / "ops" / "old-gone" / ".versions"
    gone.mkdir(parents=True)
    (gone / "v1.md").write_text(_body("RESIDUE-WORDS-9"), encoding="utf-8")
    _offer(world, key, 99, OFFER)
    handmade = world.data / "skills" / "Ops" / "handmade"
    handmade.mkdir(parents=True)
    (handmade / "SKILL.md").write_text(_body("HANDMADE-WORDS-9"), encoding="utf-8")
    world.skills.add("notes", "_private", _body("PRIVATE-WORDS-9"), source=sk.SOURCE_MANUAL,
                     status=sk.STATUS_PUBLISHED)
    return key


def test_sk6_the_wipe_with_the_skills_named(tmp_path, monkeypatch):
    world = _world(tmp_path / "w", wipe=True)
    try:
        udm = world.udm
        key = _wipe_plant(world)
        preview = udm.wipe_preview("u-1", remove=["skills"])
        counts = preview["counts"]["skills"]
        assert {k: counts[k] for k in ("skills", "published", "drafts", "versions")} == {
            "skills": 5, "published": 4, "drafts": 1, "versions": 1}, counts
        assert preview["phrase"] == f"remove 5 skills {counts['code']}" and len(counts["code"]) == 4, preview
        assert "paired devices" in preview["notice"] and preview["on_request"]["skills"] == [_SKILLS_LABEL,
                                                                                            _OFFERS_LABEL]
        answer = udm.UserDataDeleter().delete_all("u-1", remove=["skills"], confirm=preview["phrase"])
        assert answer["content_removed"]["skills"] == 5, answer["content_removed"]
        assert not [line for line in answer["not_covered"] if line.startswith(("skills", "paired devices"))], (
            answer["not_covered"])
        words = (*_GONE, KEPT.encode(), DRAFT.encode(), b"RESIDUE-WORDS-9", PENDING.encode(), CLAIMED.encode(),
                 b"HANDMADE-WORDS-9", b"PRIVATE-WORDS-9")
        assert _held(world.data, *words) == set(), f"the wipe left words: {_held(world.data, *words)}"
        assert world.skills.skill_keys() == [] and world.ledger.count() == 0
        tombs = {r.record_id: r for r in world.feed.current_records() if r.kind.value == "skill"}
        assert sorted(tombs) == ["_private/notes", "ops/kept", key] and all(t.deleted for t in tombs.values()), tombs
    finally:
        world.restore()
    # A skill whose deletion refuses at every pass is named; the others go.
    world = _world(tmp_path / "stuck", wipe=True)
    try:
        udm = world.udm
        _wipe_plant(world)
        forget = world.skills.forget_skill

        def refusing(name, category, **kwargs):
            if name == "kept":
                raise world.sk.SkillNotForgotten("held")
            return forget(name, category, **kwargs)

        monkeypatch.setattr(world.skills, "forget_skill", refusing)
        phrase = udm.wipe_preview("u-1", remove=["skills"])["phrase"]
        answer = udm.UserDataDeleter().delete_all("u-1", remove=["skills"], confirm=phrase)
        assert answer["content_removed"]["skills"] == 4, answer["content_removed"]
        assert f"{_SKILLS_LABEL} (the skill 'ops/kept' not deleted: SkillNotForgotten)" in answer["not_covered"], (
            answer["not_covered"])
        assert world.skills.get("kept", "ops") is not None
        # A folder linked under a name the registry does not keep is never followed to the skill it leads to.
        real = world.data / "skills" / "ops" / "kept"
        (world.data / "skills" / "Ops").mkdir(parents=True, exist_ok=True)
        (world.data / "skills" / "Ops" / "alias").symlink_to(real)
        assert world.skills.forget_skill("alias", "Ops") is False and (real / "SKILL.md").is_file()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SK7 -- every refusal before anything is removed
# ---------------------------------------------------------------------------
def test_sk7_every_refusal_before_anything_is_removed(tmp_path, monkeypatch):
    world = _world(tmp_path / "one", wipe=True)
    try:
        udm = world.udm
        key = _plant(world)
        before = len(_journalled(world, key))
        deleter = udm.UserDataDeleter()
        statuses = {}
        for case, call in (
            ("no phrase", lambda: deleter.delete_all("u-1", remove=["skills"])),
            ("another phrase", lambda: deleter.delete_all("u-1", remove=["skills"], confirm="remove 9 skills 0000")),
            ("a word it does not know", lambda: deleter.delete_all("u-1", remove=["skill"], confirm="x")),
        ):
            with pytest.raises(udm.WipeRefused) as refused:
                call()
            statuses[case] = refused.value.status
        monkeypatch.setattr(udm, "_get_skill_registry", lambda: (_ for _ in ()).throw(OSError("unreadable")))
        for case, call in (("unreadable preview", lambda: udm.wipe_preview("u-1", remove=["skills"])),
                           ("unreadable wipe", lambda: deleter.delete_all("u-1", remove=["skills"], confirm="x"))):
            with pytest.raises(udm.WipeRefused) as refused:
                call()
            statuses[case] = refused.value.status
        monkeypatch.undo()
        assert statuses == {"no phrase": 422, "another phrase": 409, "a word it does not know": 422,
                            "unreadable preview": 503, "unreadable wipe": 503}, statuses
        assert world.skills.get("zanzibar", "ops") is not None and _held(world.data, *_GONE) == set(_GONE) - {
            OFFER.encode()}, "a refused wipe removed something"
        assert len(_journalled(world, key)) == before
    finally:
        world.restore()
    world = _world(tmp_path / "many", wipe=True, single=False)
    try:
        _plant(world, proposals=False)
        with pytest.raises(world.udm.WipeRefused) as refused:
            world.udm.wipe_preview("u-1", remove=["skills"])
        assert refused.value.status == 409 and world.skills.get("zanzibar", "ops") is not None
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SK8 -- the export carries every skill and its history
# ---------------------------------------------------------------------------
def test_sk8_the_export_carries_every_skill_and_its_history(tmp_path):
    world = _world(tmp_path / "one", wipe=True)
    try:
        _plant(world, proposals=False)
        world.skills.add("drafty", "misc", _body(DRAFT))
        world.skills.add("notes", "_private", _body("PRIVATE-WORDS-9"), source=world.sk.SOURCE_MANUAL,
                         status=world.sk.STATUS_PUBLISHED)
        exported = world.udm.UserDataExporter().export("u-1")
        skills = {(s["category"], s["name"], s["status"]): s for s in exported["skills"]}
        assert sorted(skills) == [("_private", "notes", "published"), ("misc", "drafty", "draft"),
                                  ("ops", "zanzibar", "published")], sorted(skills)
        zanzibar = skills[("ops", "zanzibar", "published")]
        assert BODY in zanzibar["markdown"] and [v["file"] for v in zanzibar["versions"]] == ["v1.md"]
        assert OLD in zanzibar["versions"][0]["markdown"] and DRAFT in skills[("misc", "drafty", "draft")]["markdown"]
        meta = exported["export_metadata"]
        assert meta["format_version"] == "1.4" and meta["failed"] == [], meta
        assert _SKILLS_LABEL not in meta["not_covered"] and _OFFERS_LABEL in meta["not_covered"], meta["not_covered"]
    finally:
        world.restore()
    world = _world(tmp_path / "many", wipe=True, single=False)
    try:
        _plant(world, proposals=False)
        assert world.udm.UserDataExporter().export("u-1")["skills"] == [], "carried for one name among others"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SK9 -- the request's words are four
# ---------------------------------------------------------------------------
def test_sk9_the_requests_words_are_four_and_failures_carry_the_inventorys_labels(tmp_path):
    # kb11 of tests/test_knowledge_forgetting_contracts.py, which held the words three, re-asserted unchanged for
    # the projects and the knowledge base, with the fourth.
    from test_knowledge_forgetting_contracts import _LABELS, _PROJECT_LABELS, _constant
    from test_knowledge_forgetting_contracts import _world as _knowledge_world

    loaded, restore = isolate(targets={"opti_oignon.store_inventory": source("store_inventory.py")})
    try:
        inventory = loaded["opti_oignon.store_inventory"]
        assert inventory.REQUESTS == ("notes", "projects", "knowledge", "skills")
        named = [e.label for e in inventory.STORES if e.request == inventory.PROJECTS]
        assert tuple(named) == _PROJECT_LABELS, named
        named = [e.label for e in inventory.STORES if e.request == inventory.KNOWLEDGE]
        assert tuple(named) == _LABELS, named
        assert all(e.holds in (inventory.WORDS, inventory.DRAWN) for e in inventory.STORES
                   if e.request == inventory.KNOWLEDGE)
        named = [e.label for e in inventory.STORES if e.request == inventory.SKILLS]
        assert tuple(named) == (_SKILLS_LABEL, _OFFERS_LABEL), named
        assert all(e.holds == inventory.WORDS for e in inventory.STORES if e.request == inventory.SKILLS)
    finally:
        restore()
    assert _constant(source("project_context.py"), "_INDEX_LABEL") == _PROJECT_LABELS[2]
    assert _constant(source("projects.py"), "_FILES_LABEL") == _PROJECT_LABELS[1]
    assert (_constant(source("agent", "skills.py"), "_SKILLS_LABEL"),
            _constant(source("agent", "skills.py"), "_OFFERS_LABEL")) == (_SKILLS_LABEL, _OFFERS_LABEL)
    # The stores a deletion asks beside the registry, by the names their own modules spell.
    for name, module, constant in (("_JOURNAL_FILE", ("veilid", "change_feed.py"), "DB_FILENAME"),
                                   ("_LEDGER_FILE", ("veilid", "deferred_ledger.py"), "DB_FILENAME"),
                                   ("_QUEUE_FILE", ("pending_writes.py",), "_FILE_NAME")):
        assert _constant(source("agent", "skills.py"), name) == _constant(source(*module), constant), name
    world = _knowledge_world(tmp_path / "kb", wipe=True)
    try:
        rs = world.rs
        assert (rs._DOCUMENTS_LABEL, rs._AUDIT_LABEL, rs._JOBS_LABEL) == _LABELS
        assert tuple(world.udm.WIPE_WHEN_NAMED["knowledge"]) == _LABELS
        assert tuple(world.udm.WIPE_WHEN_NAMED["skills"]) == (_SKILLS_LABEL, _OFFERS_LABEL)
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SK10 -- the panels and the route say what a deletion left
# ---------------------------------------------------------------------------
_RULES = "frontend/src/lib/settings/yourData.ts"
_DRIVER = r"""
const m = await import(process.env.OO_YOUR_DATA);
const ok = (cond, what) => { if (!cond) { console.log('FAIL ' + what); process.exit(1); } };
ok(m.wipeQuery(['skills', 'notes'], 'remove 1 note and 3 skills 8f1e') ===
   '?remove=notes&remove=skills&confirm=remove%201%20note%20and%203%20skills%208f1e', 'both words, sorted');
const gate = { exportOffered: true, typedWipe: 'delete my data', named: ['skills'],
               phrase: 'remove 3 skills 8f1e', typedPhrase: '', busy: false };
ok(!m.canWipe(gate), 'the phrase asked for the skills');
ok(m.canWipe({ ...gate, typedPhrase: 'remove 3 skills 8f1e' }), 'the phrase typed');
ok(m.counted(1, 'skill') === '1 skill' && m.counted(2, 'draft') === '2 drafts', 'counted');
console.log('PASS rules');
"""


def test_sk10_the_panels_and_the_route_say_what_a_deletion_left(tmp_path, monkeypatch):
    from _frontend import read, run_ts
    from test_approval_shows_contracts import _routes_world

    run_ts({"OO_YOUR_DATA": _RULES}, _DRIVER, "rules")
    panel = read("frontend/src/lib/components/settings/YourDataPanel.svelte")
    assert 'label="Also delete my skills"' in panel and "on:change={toggleSkills}" in panel
    assert ".concat(skills ? ['skills'] : [])" in panel, "the word named"
    assert "{#if preview?.on_request?.skills}" in panel and "counted(skillsCounted.skills, 'skill')" in panel
    skills_panel = read("frontend/src/lib/components/panels/SkillsPanel.svelte")
    assert "const answer = await deleteSkill(skill.category, skill.name, skill.status, shown.sha256);" in skills_panel
    assert "if (!answer.deleted) throw new Error(" in skills_panel, "a deletion that did not happen said done"
    assert "answer.not_forgotten ?? []" in skills_panel and "not all of it went:" in skills_panel
    assert "if (typeof said === 'string') toastWarning(said);" in skills_panel
    assert "not_forgotten?: string[];" in read("frontend/src/lib/api/skills.ts")
    world = _routes_world(tmp_path / "routes")
    try:
        routes, registry, skills = world.routes, world.registry, world.skills
        body = _body(BODY)

        def published():
            registry.add("greet", "general", body, status=skills.STATUS_PUBLISHED)
            return registry.get("greet", "general").digest()

        answers = [routes.skill_delete_payload(registry, "general", "greet", status="published", sha256=published())]
        monkeypatch.setattr(skills, "_forget_proposals",
                            lambda category, name, data, present=None: (
                                [] if present is not None else ["pending writes (held)"]))
        answers.append(routes.skill_delete_payload(registry, "general", "greet", status="published",
                                                   sha256=published()))
        monkeypatch.undo()
        real = skills._forget_entry
        monkeypatch.setattr(skills, "_forget_entry", lambda path, said: (
            (_ for _ in ()).throw(PermissionError("held")) if path.name == "_usage.json" else real(path, said)))
        digest = published()
        registry.increment_usage("greet", "general")
        routes._resolve_skill_registry = lambda: registry
        with pytest.raises(routes.HTTPException) as refused:
            routes.delete_skill("general", "greet", "published", digest)
        kept = registry.get("greet", "general")
    finally:
        world.restore()
    assert answers == [{"deleted": True}, {"deleted": True, "not_forgotten": ["pending writes (held)"]}], answers
    assert refused.value.status_code == 503 and "it stays listed, delete it again" in refused.value.detail
    assert kept is not None, "a refused deletion took the skill"


# ---------------------------------------------------------------------------
# SK11 -- the overwrite is one text in its five copies
# ---------------------------------------------------------------------------
def test_sk11_the_overwrite_is_one_text_in_its_five_copies(tmp_path, monkeypatch):
    # pf13 of tests/test_project_forgetting_contracts.py, which counted four copies, re-asserted unchanged on them;
    # the fifth, the skills', is the same text.
    import test_project_forgetting_contracts as pf

    copies = pf._copies()
    assert sorted(copies) == ["agent/skills.py", "history.py", "learned_router.py", "notes/blob_store.py",
                              "projects.py"], sorted(copies)
    assert len(set(copies.values())) == 1, "the copies of the overwrite differ"
    four = {path: text for path, text in copies.items() if path != "agent/skills.py"}
    monkeypatch.setattr(pf, "_copies", lambda: dict(four))
    pf.test_pf13_no_deletion_overwrites_a_file_another_link_names(tmp_path)
    tree = ast.parse(source("agent", "skills.py").read_text(encoding="utf-8"))
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"_forget_entry", "_forget_folder", "_overwrite_and_remove", "_made_writable"} <= defined
    assert REPO.is_dir()
