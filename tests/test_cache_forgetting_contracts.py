"""Contracts for a deletion that leaves no answer drawn from what it deleted in the caches.

The response cache and the semantic cache keep answers by the request and the prompt they were given for -- a prompt
built from the knowledge base's documents, a project's files, the notes, the skills --, and no entry says which of
those it held. A conversation's deletion forgot its cached answers; the deletion of a knowledge-base document or
collection, a project, a note or a skill forgot none: an answer drawn from the deleted words stayed on disk until its
time ran out, and an answer a request already under way put after the deletion was drawn from them too. A note's
decided proposals stayed in the review queue, the one whose acceptance made the note with its title and body. Now each
of those deletions empties both caches whole, overwritten (``cached_answers``), an answer asked before the emptying is
refused by the cache, and a note's deletion forgets its decided proposals.

  * CF1 -- A REMOVED DOCUMENT OR COLLECTION LEAVES NO CACHED ANSWER: through the real knowledge base, both caches'
    files read back; the answer of a request under way is refused.
  * CF2 -- A DELETED PROJECT, OR A FILE REMOVED FROM ONE, LEAVES NONE: through the real projects store.
  * CF3 -- A DELETED NOTE LEAVES NONE, a paired device's deletion of it neither: through the real notes store.
  * CF4 -- A DELETED SKILL LEAVES NONE, a paired device's deletion of it neither: through the real registry and round.
  * CF5 -- AN ANSWER ASKED BEFORE AN EMPTYING IS NEVER PUT: in both caches, the mark taken when the request began
    (the executor's first act) and handed back right before each put, two requests on one key each judged by its
    own; with no mark handed back, a key looked up again keeping the oldest, many lookups between; one asked after it
    is put, and one put with no lookup before it too.
  * CF6 -- A DELETED NOTE'S DECIDED PROPOSALS GO: an update or a deletion aimed at it, the acceptance that made it; one
    pending, and a declined one that would have made a note, stay; one decided after the deletion goes with the next.
  * CF7 -- A CACHE THAT CANNOT BE EMPTIED IS NAMED, and the deletion goes on; another folder's cache is left.
  * CF8 -- AN ERASED FACT LEAVES NONE: a hard deletion, an erasure from the trash, a paired device's erasure, the
    user's clear, through the real facts store, the caches emptied once the row is gone, outside the store's lock.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import ast
import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import source  # noqa: E402

BUDGET_S = {
    "test_cf1_a_removed_document_or_collection_leaves_no_cached_answer": 10.0,
    "test_cf2_a_deleted_project_leaves_no_cached_answer": 10.0,
    "test_cf3_a_deleted_note_leaves_no_cached_answer_nor_a_paired_devices_deletion_of_it": 10.0,
    "test_cf4_a_deleted_skill_leaves_no_cached_answer_nor_a_paired_devices_deletion_of_it": 10.0,
    "test_cf5_an_answer_asked_before_an_emptying_is_never_put": 10.0,
    "test_cf6_a_deleted_notes_decided_proposals_go": 10.0,
    "test_cf7_a_cache_that_cannot_be_emptied_is_named_and_another_folders_is_left": 10.0,
    "test_cf8_an_erased_fact_leaves_no_cached_answer": 10.0,
}

_CACHES = {
    "opti_oignon.response_cache": source("response_cache.py"),
    "opti_oignon.semantic_cache": source("semantic_cache.py"),
    "opti_oignon.cached_answers": source("cached_answers.py"),
}


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _plain(path, *, check_same_thread=False, **_options):
    return sqlite3.connect(str(path), check_same_thread=check_same_thread, timeout=0.3)


def _held(data, *words):
    """Which of ``words`` any file under ``data`` -- or the file ``data`` -- still holds."""
    found = set()
    for path in ([data] if data.is_file() else data.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            found.update(w for w in words if w in raw)
    return found


def _with_caches(suite, monkeypatch, *, queue=False):
    """``suite``'s own world, its isolation window opened with the two caches and their forgetting loaded from their
    sources too (and, with ``queue``, the review queue): the stores' deletions reach them as the package wires them."""
    real = suite.isolate

    def isolate(*, targets, seeded=None, packages=(), **rest):
        extra = dict(_CACHES)
        stand_ins = {}
        if queue:
            extra["opti_oignon.pending_writes"] = source("pending_writes.py")
            stand_ins = {
                "opti_oignon.db_encryption": _module("opti_oignon.db_encryption", get_encrypted_connection=_plain,
                                                     SQLCIPHER_AVAILABLE=False),
                "opti_oignon.user_isolation": _module(
                    "opti_oignon.user_isolation", DEFAULT_LOCAL_USER="local",
                    effective_user_id=lambda user_id, single_user_mode=True: "local"),
            }
        return real(targets={**dict(targets), **extra}, seeded={**stand_ins, **(seeded or {})}, packages=packages,
                    **rest)

    monkeypatch.setattr(suite, "isolate", isolate)


def _loaded(world):
    """The modules a world loaded, by their dotted names."""
    held = getattr(world, "loaded", None)
    return held if held is not None else {f"opti_oignon.{name}": module for name, module in world.mod.items()}


def _cache(world, word):
    """Both caches over the world's data folder, each holding an answer drawn from ``word``, and a request under way:
    its lookups missed, its answer not yet put."""
    loaded = _loaded(world)
    rc = loaded["opti_oignon.response_cache"].response_cache
    sc = loaded["opti_oignon.semantic_cache"].semantic_cache
    sc._config.update(enabled=True, semantic_match_enabled=False, exact_match_enabled=True)
    sc.embeddings_available = False
    key = rc.make_cache_key("m", f"a system prompt with {word}", "the question")
    assert rc.get(key) is None
    assert rc.put("m", f"a system prompt with {word}", "the question", f"an answer drawn from {word}")
    assert sc.get("the question", context_fingerprint=f"fp-{word}") is None
    assert sc.put("the question", f"an answer drawn from {word}", model="m", context_fingerprint=f"fp-{word}")
    later = rc.make_cache_key("m", f"a system prompt with {word}", "a later question")
    assert rc.get(later) is None and sc.get("a later question", context_fingerprint=f"fp-{word}") is None
    return SimpleNamespace(rc=rc, sc=sc, later=later, word=word.encode())


def _answered(data, cache, what):
    """No file under ``data`` holds an answer drawn from the word; the answer under way is refused."""
    assert not _held(data, cache.word), f"{what}: a cached answer drawn from the deleted words stayed"
    assert cache.rc.put("m", f"a system prompt with {cache.word.decode()}", "a later question",
                        f"a late answer drawn from {cache.word.decode()}") == "", f"{what}: a late answer was cached"
    assert cache.sc.put("a later question", f"a late answer drawn from {cache.word.decode()}", model="m",
                        context_fingerprint=f"fp-{cache.word.decode()}") == "", f"{what}: a late answer was cached"
    assert not _held(data, cache.word), f"{what}: a late answer reached a file"


# ---------------------------------------------------------------------------
# CF1 -- a removed document or collection leaves no cached answer
# ---------------------------------------------------------------------------
def test_cf1_a_removed_document_or_collection_leaves_no_cached_answer(tmp_path, monkeypatch):
    import test_knowledge_forgetting_contracts as kb

    _with_caches(kb, monkeypatch)
    for case in ("document", "collection"):
        world = kb._world(tmp_path / case)
        try:
            kb._plant(world, "doc-gone", kb.GONE, kb.GONE_FILE, kb.GONE_QUERY)
            cache = _cache(world, f"CACHED-KB-{case.upper()}-9")
            assert _held(world.data, cache.word), "control: the caches hold the answer"
            if case == "document":
                removed, left = world.store.forget_document("doc-gone")
            else:
                removed, left = world.store.forget_collection("papers")
            assert removed and left == [], (case, left)
            _answered(world.data, cache, f"a removed {case}")
        finally:
            world.restore()


# ---------------------------------------------------------------------------
# CF2 -- a deleted project leaves no cached answer
# ---------------------------------------------------------------------------
def test_cf2_a_deleted_project_leaves_no_cached_answer(tmp_path, monkeypatch):
    import test_project_forgetting_contracts as pf

    _with_caches(pf, monkeypatch)
    world = pf._world(tmp_path)
    try:
        planted = pf._plant(world)
        cache = _cache(world, "CACHED-PROJECT-FILE-9")
        removed, left = world.store.forget_file(planted.file.id)
        assert removed and left == [], left
        _answered(world.data, cache, "a project file removed")
        cache = _cache(world, "CACHED-PROJECT-9")
        deleted, left = world.store.forget_project(planted.project.id)
        assert deleted and left == [], left
        _answered(world.data, cache, "a deleted project")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CF3 -- a deleted note leaves no cached answer, a paired device's deletion of it neither
# ---------------------------------------------------------------------------
def test_cf3_a_deleted_note_leaves_no_cached_answer_nor_a_paired_devices_deletion_of_it(tmp_path, monkeypatch):
    import test_note_forgetting_contracts as nf

    _with_caches(nf, monkeypatch, queue=True)
    world = nf._world(tmp_path)
    try:
        note = world.notes.add_note(nf.TITLE, body_crdt=nf.BODY, tags=f'["{nf.TAG}"]')
        cache = _cache(world, "CACHED-NOTE-9")
        died, left = world.notes.forget_note(note.id)
        assert died and left == [], left
        _answered(world.data, cache, "a deleted note")
        # A paired device's deletion: the note dead here, its tombstone journalled, then forgotten.
        other = world.notes.add_note("PEER-TITLE-9", body_crdt=b"PEER-BODY-9", tags="[]")
        cache = _cache(world, "CACHED-PEER-NOTE-9")
        with world.notes._lock, world.notes._conn() as conn:
            conn.execute("UPDATE note SET deleted = 1 WHERE id = ?", (other.id,))
        assert world.notes.forget_tombstoned(other.id) == []
        _answered(world.data, cache, "a paired device's deletion of a note")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CF4 -- a deleted skill leaves no cached answer, a paired device's deletion of it neither
# ---------------------------------------------------------------------------
def test_cf4_a_deleted_skill_leaves_no_cached_answer_nor_a_paired_devices_deletion_of_it(tmp_path, monkeypatch):
    import test_skill_forgetting_contracts as sk

    _with_caches(sk, monkeypatch)
    world = sk._world(tmp_path / "here")
    try:
        key = sk._plant(world, proposals=False)
        cache = _cache(world, "CACHED-SKILL-9")
        assert sk._delete(world) == []
        _answered(world.data, cache, "a deleted skill")
        (sent,) = sk._journalled(world, key)
        payload, clock = dict(sent.payload), sent.clock
    finally:
        world.restore()
    world = sk._world(tmp_path / "peer")
    try:
        sk._plant(world, proposals=False)
        cache = _cache(world, "CACHED-PEER-SKILL-9")
        records = world.mod["records"]
        tomb = records.new_record(records.RecordKind.SKILL, key, payload, device="dev-b", clock=clock, deleted=True,
                                  updated_at=sk._UPDATED)
        sk._round(world, [tomb])
        assert world.skills.get("zanzibar", "ops") is None, "control: the paired device's deletion landed"
        _answered(world.data, cache, "a paired device's deletion of a skill")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CF8 -- an erased fact leaves no cached answer
# ---------------------------------------------------------------------------
def test_cf8_an_erased_fact_leaves_no_cached_answer(tmp_path, monkeypatch):
    import test_landing_verdict_contracts as lw

    _with_caches(lw, monkeypatch)
    world = lw._world(tmp_path)
    try:
        world.facts.add("The user's sister is called FACT-GONE-9.", "fact", fact_id="fact-gone")
        cache = _cache(world, "CACHED-FACT-9")
        assert world.facts.hard_delete("fact-gone")
        _answered(world.data, cache, "a fact deleted")
        world.facts.add("The user's brother is called FACT-TRASHED-9.", "fact", fact_id="fact-trashed")
        assert world.facts.soft_delete("fact-trashed")
        cache = _cache(world, "CACHED-TRASHED-FACT-9")
        assert world.facts.erase("fact-trashed", trashed_only=True)
        _answered(world.data, cache, "a fact erased from the trash")
        # Emptied once the row is gone, outside the store's lock.
        world.facts.add("The user's cousin is called FACT-ORDER-9.", "fact", fact_id="fact-order")
        cache = _cache(world, "CACHED-ORDER-FACT-9")
        seen = []
        empty = cache.sc.empty

        def watched():
            seen.append((world.facts.get("fact-order"), not world.facts._lock._is_owned()))
            return empty()

        monkeypatch.setattr(cache.sc, "empty", watched)
        assert world.facts.hard_delete("fact-order")
        monkeypatch.setattr(cache.sc, "empty", empty)
        assert seen == [(None, True)], seen
        # A paired device's erasure; the user's clear; a folder holding the semantic cache's file only.
        world.facts.add("The user's aunt is called FACT-PEER-9.", "fact", fact_id="fact-peer")
        cache = _cache(world, "CACHED-PEER-FACT-9")
        assert world.facts.apply_synced_memory_canonical("fact-peer", {}, deleted=True)
        _answered(world.data, cache, "a paired device's erasure of a fact")
        world.facts.add("The user's uncle is called FACT-CLEAR-9.", "fact", fact_id="fact-clear")
        cache = _cache(world, "CACHED-CLEAR-FACT-9")
        (world.data / "response_cache.db").unlink()
        assert world.facts.clear() == 1
        assert not _held(world.data / "semantic_cache.db", cache.word), "the facts cleared: an answer stayed"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CF5 -- an answer asked before an emptying is never put
# ---------------------------------------------------------------------------
def test_cf5_an_answer_asked_before_an_emptying_is_never_put(tmp_path, monkeypatch):
    import test_knowledge_forgetting_contracts as kb

    _with_caches(kb, monkeypatch)
    world = kb._world(tmp_path)
    try:
        cache = _cache(world, "CACHED-EMPTIED-9")
        rc, sc = cache.rc, cache.sc
        before = rc.make_cache_key("m", "a prompt", "asked before")
        assert rc.get(before) is None and sc.get("asked before", context_fingerprint="fp") is None
        rc.empty()
        sc.empty()
        assert rc.put("m", "a prompt", "asked before", "BEFORE-ANSWER-9") == ""
        assert sc.put("asked before", "BEFORE-ANSWER-9", model="m", context_fingerprint="fp") == ""
        assert rc.get(before) is None and sc.get("asked before", context_fingerprint="fp") is None
        # Asked after the emptying: put; put with no lookup before it (warming): put too.
        assert rc.put("m", "a prompt", "asked before", "AFTER-ANSWER-9") != ""
        assert sc.put("asked before", "AFTER-ANSWER-9", model="m", context_fingerprint="fp") != ""
        assert rc.put("m", "a prompt", "warmed", "WARMED-ANSWER-9") != ""
        assert _held(world.data, b"BEFORE-ANSWER-9") == set()
        assert _held(world.data, b"AFTER-ANSWER-9", b"WARMED-ANSWER-9") == {b"AFTER-ANSWER-9", b"WARMED-ANSWER-9"}
        # A key looked up again after the emptying keeps the oldest mark: the first request's answer is refused, in
        # the semantic cache under another model too.
        again = rc.make_cache_key("m", "a prompt", "asked twice")
        assert rc.get(again) is None and sc.get("asked twice", model="m", context_fingerprint="fp") is None
        rc.empty()
        sc.empty()
        assert rc.get(again) is None and sc.get("asked twice", model="other", context_fingerprint="fp") is None
        assert rc.put("m", "a prompt", "asked twice", "TWICE-ANSWER-9") == ""
        assert sc.put("asked twice", "TWICE-ANSWER-9", model="m", context_fingerprint="fp") == ""
        # A request marked when it began, before its prompt drew on any store, then emptied before its lookup.
        rc_module, sc_module = _loaded(world)["opti_oignon.response_cache"], _loaded(world)["opti_oignon.semantic_cache"]
        rc_module.begin_request()
        sc_module.begin_request()
        rc.empty()
        sc.empty()
        drawn = rc.make_cache_key("m", "a prompt drawn before", "asked after")
        assert rc.get(drawn) is None and sc.get("asked after", context_fingerprint="fp-drawn") is None
        assert rc.put("m", "a prompt drawn before", "asked after", "DRAWN-ANSWER-9") == ""
        assert sc.put("asked after", "DRAWN-ANSWER-9", model="m", context_fingerprint="fp-drawn") == ""
        rc_module.begin_request()
        sc_module.begin_request()
        # A request's mark outlives the lookups of many others, well within the bound.
        first = rc.make_cache_key("m", "a prompt", "the first of many")
        assert rc.get(first) is None
        for n in range(200):
            assert rc.get(rc.make_cache_key("m", "a prompt", f"another {n}")) is None
        rc.empty()
        assert rc.put("m", "a prompt", "the first of many", "MANY-ANSWER-9") == ""
        assert _held(world.data, b"TWICE-ANSWER-9", b"DRAWN-ANSWER-9", b"MANY-ANSWER-9") == set()
        # Two requests on one key: each is judged by its own beginning, handed back at its put, never by the other's.
        ca = _loaded(world)["opti_oignon.cached_answers"]
        marks_a = ca.begin_cached_request()
        assert sorted(marks_a) == ["opti_oignon.response_cache", "opti_oignon.semantic_cache"], marks_a
        shared = rc.make_cache_key("m", "a prompt", "asked by two")
        assert rc.get(shared) is None and sc.get("asked by two", context_fingerprint="fp-two") is None
        marks_b = ca.begin_cached_request()
        assert rc.get(shared) is None and sc.get("asked by two", context_fingerprint="fp-two") is None
        rc.empty()
        sc.empty()
        for marks in (marks_a, marks_b):
            ca.declare_cached_request(marks, "opti_oignon.response_cache")
            assert rc.put("m", "a prompt", "asked by two", "TWO-BEFORE-9") == ""
            ca.declare_cached_request(marks, "opti_oignon.semantic_cache")
            assert sc.put("asked by two", "TWO-BEFORE-9", model="m", context_fingerprint="fp-two") == ""
        late = ca.begin_cached_request()
        assert rc.get(shared) is None and sc.get("asked by two", context_fingerprint="fp-two") is None
        early = marks_a
        ca.declare_cached_request(late, "opti_oignon.response_cache")
        assert rc.put("m", "a prompt", "asked by two", "TWO-AFTER-9") != ""
        ca.declare_cached_request(late, "opti_oignon.semantic_cache")
        assert sc.put("asked by two", "TWO-AFTER-9", model="m", context_fingerprint="fp-two") != ""
        ca.declare_cached_request(early, "opti_oignon.response_cache")
        assert rc.put("m", "a prompt", "asked by two", "TWO-EARLY-9") == ""
        ca.declare_cached_request(early, "opti_oignon.semantic_cache")
        assert sc.put("asked by two", "TWO-EARLY-9", model="m", context_fingerprint="fp-two") == ""
        assert _held(world.data, b"TWO-BEFORE-9", b"TWO-EARLY-9") == set() and _held(world.data, b"TWO-AFTER-9")
        # An emptying of a cache whose file is not there still refuses what was asked before it.
        rc_module.begin_request()
        fresh = rc_module.ResponseCache(db_path=tmp_path / "elsewhere" / "response_cache.db")
        asked = fresh.make_cache_key("m", "p", "q")
        assert fresh.get(asked) is None
        (tmp_path / "elsewhere" / "response_cache.db").unlink()
        fresh.empty()
        assert fresh.put("m", "p", "q", "FRESH-LATE-9") == ""
    finally:
        world.restore()
    # The executor marks its request first, before its prompt draws on any store, and hands its marks back right before
    # each of its puts.
    tree = ast.parse(source("executor.py").read_text(encoding="utf-8"))
    (execute,) = [node for cls in tree.body if isinstance(cls, ast.ClassDef) for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "execute"]
    assert ast.unparse(execute.body[1]) == "_cached_marks = _begin_cached_request()", ast.unparse(execute.body[1])
    puts = 0
    for block in ast.walk(execute):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(block, field, None)
            if not isinstance(statements, list):
                continue
            for index, statement in enumerate(statements):
                calls = [node for node in ast.walk(statement) if isinstance(node, ast.Call)
                         and ast.unparse(node.func) in ("_response_cache.put", "_semantic_cache.put")]
                if calls and not isinstance(statement, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                    puts += 1
                    cache = "opti_oignon.semantic_cache" if "_semantic_cache.put" in ast.unparse(statement) else (
                        "opti_oignon.response_cache")
                    assert index > 0 and ast.unparse(statements[index - 1]) == (
                        f"_declare_cached_request(_cached_marks, '{cache}')"), ast.unparse(statement)[:80]
    assert puts == 2, puts
    for helper, delegate in (("_begin_cached_request", "begin_cached_request"),
                             ("_declare_cached_request", "declare_cached_request")):
        (function,) = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == helper]
        assert delegate in ast.unparse(function), helper


# ---------------------------------------------------------------------------
# CF6 -- a deleted note's decided proposals go
# ---------------------------------------------------------------------------
def test_cf6_a_deleted_notes_decided_proposals_go(tmp_path, monkeypatch):
    import test_note_forgetting_contracts as nf

    _with_caches(nf, monkeypatch, queue=True)
    world = nf._world(tmp_path)
    try:
        pw = _loaded(world)["opti_oignon.pending_writes"]
        queue = pw.get_pending_store()
        made = queue.propose("notes", "make", {"title": "MADE-TITLE-9", "body": "MADE-BODY-9", "tags": "[]"},
                             {"via": "the agent"})[0]
        queue.claim(made.id)
        note = world.notes.add_note("MADE-TITLE-9", body_crdt=b"MADE-BODY-9", tags="[]",
                                    note_id=pw.accepted_note_id(made.id))
        queue.settle(made.id, f"Note created [{note.id}] MADE-TITLE-9.")
        update = queue.propose("notes", "update", {"note_id": note.id, "title": "UPDATE-TITLE-9", "tags": "[]"},
                               {"via": "the agent"})[0]
        queue.decline(update.id)
        waiting = queue.propose("notes", "update", {"note_id": note.id, "title": "WAITING-TITLE-9", "tags": "[]"},
                                {"via": "the agent"})[0]
        never = queue.propose("notes", "make", {"title": "NEVER-TITLE-9", "body": "NEVER-BODY-9", "tags": "[]"},
                              {"via": "the agent"})[0]
        queue.decline(never.id)
        died, left = world.notes.forget_note(note.id)
        assert died and left == [], left
        assert _held(world.data, b"MADE-BODY-9", b"UPDATE-TITLE-9") == set(), (
            "a deleted note's decided proposals kept its words")
        assert _held(world.data, b"WAITING-TITLE-9", b"NEVER-BODY-9") == {b"WAITING-TITLE-9", b"NEVER-BODY-9"}, (
            "a proposal listed for a decision, or one that made no note, went")
        # Decided after the deletion: forgotten by the next deletion's sweep.
        queue.decline(waiting.id)
        other = world.notes.add_note("OTHER-9", body_crdt=b"OTHER-9", tags="[]")
        assert world.notes.forget_note(other.id) == (True, [])
        assert not _held(world.data, b"WAITING-TITLE-9"), "a proposal decided after its note's deletion stayed"
        assert _held(world.data, b"NEVER-BODY-9")
        # The note's own deletion forgets them, with no sweep after it.
        third = world.notes.add_note("THIRD-9", body_crdt=b"THIRD-9", tags="[]")
        aimed = queue.propose("notes", "update", {"note_id": third.id, "title": "AIMED-TITLE-9", "tags": "[]"},
                              {"via": "the agent"})[0]
        queue.decline(aimed.id)
        assert world.notes.forget_note(third.id, sweep=False) == (True, [])
        assert not _held(world.data, b"AIMED-TITLE-9"), "a note's deletion left its decided proposals to a sweep"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CF7 -- a cache that cannot be emptied is named; another folder's cache is left
# ---------------------------------------------------------------------------
def test_cf7_a_cache_that_cannot_be_emptied_is_named_and_another_folders_is_left(tmp_path, monkeypatch):
    import test_knowledge_forgetting_contracts as kb

    _with_caches(kb, monkeypatch)
    world = kb._world(tmp_path / "one")
    try:
        kb._plant(world, "doc-gone", kb.GONE, kb.GONE_FILE, kb.GONE_QUERY)
        cache = _cache(world, "CACHED-STUCK-9")

        def stuck():
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(cache.sc, "empty", stuck)
        removed, left = world.store.forget_document("doc-gone")
        assert removed, "the deletion stopped at a cache"
        assert left == ["semantic cache (the answers it keeps not emptied: OperationalError; the next deletion or the "
                        "wipe empties it)"], left
        monkeypatch.undo()
        assert not _held(world.data / "response_cache.db", cache.word), "the cache that could be emptied was not"
    finally:
        world.restore()
    # The process's caches kept in another folder are another store's: a deletion here leaves them.
    _with_caches(kb, monkeypatch)
    world = kb._world(tmp_path / "two")
    try:
        cache = _cache(world, "CACHED-ELSEWHERE-9")
        forget = _loaded(world)["opti_oignon.cached_answers"].forget_cached_answers
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "response_cache.db").write_bytes(b"")
        (elsewhere / "semantic_cache.db").write_bytes(b"")
        assert forget(elsewhere) == []
        for name in ("response_cache.db", "semantic_cache.db"):
            assert _held(world.data / name, cache.word), f"another folder's deletion emptied this {name}"
    finally:
        world.restore()
