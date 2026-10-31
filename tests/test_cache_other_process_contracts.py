"""Contracts for a deletion another process makes on the same data folder, and a request under way here.

Each answer a cache takes is judged by the beginning of its request or its turn: a cache emptied since refuses it. The
count of emptyings, the marks and the forgotten conversations live in a process, and the server and ``oo chat`` run
the executor in two processes on one data folder. Now a deletion is told to every process through the cache's file:
each emptying, and each forgetting of a conversation, draws the file's generation (SQLite's user_version) anew in its
own write, before its entries go; each put reads the generation in the write it is about to make, holding the file's
write lock, and one it did not draw or read before is an emptying it counts. Two windows on one folder stand for two
processes here: each loads both caches from their source, with its own module state; the files are shared.

  * XP1 -- ANOTHER PROCESS'S EMPTYING REFUSES A REQUEST UNDER WAY HERE: in both caches, the question's embedding too,
    a declared put and one judged by its lookup; a request begun after it is put; and the other way round.
  * XP2 -- THE GENERATION IS READ IN THE PUT'S OWN WRITE: an emptying that comes while a put writes waits for it and
    deletes its entry, in each of the three writes; the response cache draws its generation before its entries go,
    so a put between the two is refused; each write judges its mark before it opens the file, opens its write
    before it reads the generation, and judges again after.
  * XP3 -- THE GENERATION: drawn anew, never 0 nor the one it replaces; a refused put writes nothing and leaves the
    file free; a process's first read counts one emptying only of a file ever emptied; a file made again counts; a
    process's own emptying, or forgetting, is not counted twice, however they interleave (round 2).
  * XP4 -- ANOTHER PROCESS'S FORGETTING OF A CONVERSATION REFUSES A PUT UNDER WAY HERE: the conversation's answer is
    kept by neither cache; and a forgetting here never swallows a deletion there it had not read: its own draw
    counts what it replaced, at a process's first touch of a file too (round 1); a forgetting here between a put's
    judgment and its write refuses the put (round 2).
  * XP5 -- THROUGH THE REAL EXECUTOR: the other process's deletion during a call refuses its answer and its question;
    a turn begun before it has its later call refused; a call begun after is put.
  * XP6 -- NOTHING HERE WAITS ON THE FILE IN VAIN (rounds 1 and 2): a put this process's own count already refuses,
    or of a conversation it forgot, does not open the file; an emptying, a forgetting or a put here that waits for
    another process's write holds no lock the requests here take.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import ast
import hashlib
import sqlite3
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_turn_cache_mark_contracts as tj  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_xp1_another_processs_emptying_refuses_a_request_under_way_here": 5.0,
    "test_xp2_the_generation_is_read_in_the_puts_own_write": 8.0,
    "test_xp3_the_generation_is_drawn_anew_and_read_once_per_change": 5.0,
    "test_xp4_another_processs_forgetting_of_a_conversation_refuses_a_put_under_way_here": 5.0,
    "test_xp5_through_the_real_executor_another_processs_deletion_refuses_the_call_under_way": 10.0,
    "test_xp6_nothing_here_waits_on_the_file_in_vain": 8.0,
}

_RC, _SC = tj._RC, tj._SC
_WAIT = 3.0
# How long a racing emptying is given to finish while a put holds its write: it must still be waiting after it.
_RACE = 0.3


def _window(data, *, factory=sqlite3.Connection):
    """One process's view of the folder ``data``: both caches and their forgetting loaded from their source, with
    their own counts, marks and generations; the files are the folder's. Connections wait for a writer, as the real
    layer's do."""
    def connect(path, *, check_same_thread=False, **_options):
        return sqlite3.connect(str(path), check_same_thread=check_same_thread, timeout=_WAIT, factory=factory)

    seeded = tj._base_seeds(data)
    seeded["opti_oignon.db_utils"] = tj._module("opti_oignon.db_utils", safe_connect=connect)
    loaded, close = isolate(targets=dict(tj._CACHES), seeded=seeded, packages=("opti_oignon",))
    rc, sc = tj._ready(loaded)
    return SimpleNamespace(loaded=loaded, rc=rc, sc=sc, RC=loaded[_RC], SC=loaded[_SC], close=close)


def _hooked(hooks):
    """A connection that runs ``hooks["read"]`` once as it reads the file's generation, ``hooks["begin"]`` once right
    before it opens a write, and ``hooks["commit"]`` once right after a commit."""
    class Hooked(sqlite3.Connection):
        def execute(self, sql, *args):
            said = " ".join(sql.split()).upper()
            if said == "PRAGMA USER_VERSION" and "read" in hooks:
                hooks.pop("read")()
            if said == "BEGIN IMMEDIATE" and "begin" in hooks:
                hooks.pop("begin")()
            return super().execute(sql, *args)

        def commit(self):
            super().commit()
            if "commit" in hooks:
                hooks.pop("commit")()

    return Hooked


def _generation(path):
    conn = sqlite3.connect(str(path), timeout=_WAIT)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _free(path):
    """Whether another writer takes the file's write lock at once."""
    conn = sqlite3.connect(str(path), timeout=0.05)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.rollback()
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def _key(win, content):
    return win.rc.make_cache_key("m", "a prompt", content)


def _counts(win):
    return win.RC._EMPTIED[0], win.SC._EMPTIED[0]


def _warm(win, tag):
    """A put in both caches, as any earlier request makes one: the files are there, and this process read them."""
    assert win.rc.put("m", "a prompt", f"warm {tag}", f"WARM-{tag}") != ""
    assert win.sc.put(f"warm {tag}", f"WARM-{tag}", model="m", context_fingerprint="fp") != ""


# ---------------------------------------------------------------------------
# XP1 -- another process's emptying refuses a request under way here
# ---------------------------------------------------------------------------

def test_xp1_another_processs_emptying_refuses_a_request_under_way_here(tmp_path):
    data = tmp_path / "data"
    here = there = None
    try:
        here = _window(data)
        there = _window(data)
        _warm(here, "XP1")
        # A request here begins, and misses.
        marks = here.RC.begin_request(), here.SC.begin_request()
        assert here.rc.get(_key(here, "asked late")) is None
        assert here.rc.get(_key(here, "asked by its lookup")) is None
        assert here.sc.get("asked late", context_fingerprint="fp") is None
        # The other process deletes: both caches emptied there.
        there.rc.empty()
        there.sc.empty()
        assert _counts(here) == (0, 0), "the witness: this process's count moved without reading the file"
        assert tj._held(data, b"WARM-XP1") == set()
        # The answer drawn before it, put after it: refused in both caches, its question's embedding too, and the
        # put judged by its lookup's mark.
        here.RC.declare_request(marks[0])
        assert here.rc.put("m", "a prompt", "asked late", "LATE-ANSWER-XP1") == ""
        assert here.rc.put("m", "a prompt", "asked by its lookup", "LOOKUP-ANSWER-XP1") == ""
        here.SC.declare_request(marks[1])
        assert here.sc.put("asked late", "LATE-ANSWER-XP1", model="m", context_fingerprint="fp") == ""
        here.SC.declare_request(marks[1])
        assert here.sc.store_embedding("late-key", "m", "LATE-QUESTION-XP1", [1.0, 0.0]) is False
        assert tj._held(data, b"LATE-ANSWER-XP1", b"LOOKUP-ANSWER-XP1", b"LATE-QUESTION-XP1") == set()
        # A request begun after the emptying: put.
        after = here.RC.begin_request(), here.SC.begin_request()
        here.RC.declare_request(after[0])
        assert here.rc.put("m", "a prompt", "asked after", "AFTER-ANSWER-XP1") != ""
        here.SC.declare_request(after[1])
        assert here.sc.put("asked after", "AFTER-ANSWER-XP1", model="m", context_fingerprint="fp") != ""
        here.SC.declare_request(after[1])
        assert here.sc.store_embedding("after-key", "m", "AFTER-QUESTION-XP1", [1.0, 0.0]) is True
        assert tj._held(data, b"AFTER-ANSWER-XP1", b"AFTER-QUESTION-XP1") == {
            b"AFTER-ANSWER-XP1", b"AFTER-QUESTION-XP1"}
        # The other way round: emptied here, refused there.
        theirs = there.RC.begin_request(), there.SC.begin_request()
        here.rc.empty()
        here.sc.empty()
        there.RC.declare_request(theirs[0])
        assert there.rc.put("m", "a prompt", "theirs", "THEIR-ANSWER-XP1") == ""
        there.SC.declare_request(theirs[1])
        assert there.sc.put("theirs", "THEIR-ANSWER-XP1", model="m", context_fingerprint="fp") == ""
        assert tj._held(data, b"THEIR-ANSWER-XP1") == set()
    finally:
        for win in (there, here):
            if win is not None:
                win.close()


# ---------------------------------------------------------------------------
# XP2 -- the generation is read in the put's own write
# ---------------------------------------------------------------------------

def _raced(hooks, empty):
    """Arm ``hooks`` so that, as the put reads the generation, the other process's ``empty`` starts and is given
    ``_RACE`` to finish; what was seen."""
    seen = {}

    def during():
        worker = threading.Thread(target=lambda: seen.setdefault("gone", empty()), name="xp2-other-process")
        worker.start()
        worker.join(_RACE)
        seen["waited"], seen["worker"] = worker.is_alive(), worker

    hooks["read"] = during
    return seen


def test_xp2_the_generation_is_read_in_the_puts_own_write(tmp_path):
    data = tmp_path / "data"
    hooks = {}
    here = there = None
    try:
        here = _window(data, factory=_hooked(hooks))
        there = _window(data)
        _warm(here, "XP2")
        # An emptying that comes while a put writes waits for the write, then deletes its entry: in each write.
        writes = (
            (lambda mark: (here.RC.declare_request(mark[0]),
                           here.rc.put("m", "a prompt", "raced", "RACED-RC-XP2"))[1] != "",
             there.rc.empty, b"RACED-RC-XP2"),
            (lambda mark: (here.SC.declare_request(mark[1]),
                           here.sc.put("raced", "RACED-SC-XP2", model="m", context_fingerprint="fp"))[1] != "",
             there.sc.empty, b"RACED-SC-XP2"),
            (lambda mark: (here.SC.declare_request(mark[1]),
                           here.sc.store_embedding("raced-key", "m", "RACED-EMB-XP2", [1.0, 0.0]))[1] is True,
             there.sc.empty, b"RACED-EMB-XP2"),
        )
        for index, (write, empty, word) in enumerate(writes):
            # A put first: this process reads what the last race left, so the request below begins after it.
            _warm(here, f"XP2-{index}")
            mark = here.RC.begin_request(), here.SC.begin_request()
            seen = _raced(hooks, empty)
            took = write(mark)
            hooks.pop("read", None)
            assert "worker" in seen, f"the put read no generation of the file: {word}"
            seen["worker"].join(_WAIT)
            assert not seen["worker"].is_alive() and "gone" in seen, f"the other process's emptying never ended: {word}"
            assert seen["waited"], f"the other process emptied the file while a put held its write: {word}"
            assert took, f"the put that took the write first was refused: {word}"
            assert tj._held(data, word) == set(), f"the emptying that waited left the put's entry: {word}"
        # The response cache draws its generation before its entries go: a put that comes between the two is
        # refused, and nothing of it stays.
        there_hooks = {}
        there.close()
        there = _window(data, factory=_hooked(there_hooks))
        assert there.rc.get(_key(there, "a lookup there")) is None
        mark = here.RC.begin_request()
        late = {}

        def between():
            here.RC.declare_request(mark)
            late["key"] = here.rc.put("m", "a prompt", "between", "BETWEEN-ANSWER-XP2")

        there_hooks["commit"] = between
        there.rc.empty()
        assert "key" in late, "the witness: the emptying never committed"
        assert late["key"] == "", "a put between the generation and the entries' going was not refused"
        assert tj._held(data, b"BETWEEN-ANSWER-XP2") == set()
    finally:
        for win in (there, here):
            if win is not None:
                win.close()
    # Read in the syntax tree: each write judges its mark before it opens the file, opens its write before it reads
    # the generation, and judges the mark again after.
    for name, cls, method in (("response_cache.py", "ResponseCache", "put"), ("semantic_cache.py", "SemanticCache",
                              "put"), ("semantic_cache.py", "SemanticCache", "store_embedding")):
        tree = ast.parse(source(name).read_text(encoding="utf-8"))
        (node,) = [n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls for n in c.body
                   if isinstance(n, ast.FunctionDef) and n.name == method]
        text = ast.unparse(node)
        assert text.count("conn.execute('BEGIN IMMEDIATE')") == 1 and text.count("_outdated(mark)") == 2, (name, method)
        assert text.index("_outdated(mark)") < text.index("self._get_connection()") < (
            text.index("conn.execute('BEGIN IMMEDIATE')")) < text.index("_absorb(conn, self._db_path)") < (
            text.rindex("_outdated(mark)")), (name, method)


# ---------------------------------------------------------------------------
# XP3 -- the generation is drawn anew and read once per change
# ---------------------------------------------------------------------------

def test_xp3_the_generation_is_drawn_anew_and_read_once_per_change(tmp_path):
    data = tmp_path / "data"
    rc_file, sc_file = data / "response_cache.db", data / "semantic_cache.db"
    wins = []
    try:
        first = _window(data)
        wins.append(first)
        _warm(first, "XP3")
        assert (_generation(rc_file), _generation(sc_file)) == (0, 0), "a file never emptied has the generation 0"
        # Its own emptying, counted once: a request begun after it is put; one begun before it is not.
        before = first.RC.begin_request(), first.SC.begin_request()
        first.rc.empty()
        first.sc.empty()
        after = first.RC.begin_request(), first.SC.begin_request()
        first.RC.declare_request(before[0])
        assert first.rc.put("m", "a prompt", "before own", "BEFORE-OWN-XP3") == ""
        first.SC.declare_request(before[1])
        assert first.sc.put("before own", "BEFORE-OWN-XP3", model="m", context_fingerprint="fp") == ""
        first.RC.declare_request(after[0])
        assert first.rc.put("m", "a prompt", "after own", "AFTER-OWN-XP3") != ""
        first.SC.declare_request(after[1])
        assert first.sc.put("after own", "AFTER-OWN-XP3", model="m", context_fingerprint="fp") != ""
        # Its own forgetting of a conversation, counted once too.
        first.rc.forget_conversation("conv-forgotten-here")
        first.sc.forget_conversation("conv-forgotten-here")
        after = first.RC.begin_request(), first.SC.begin_request()
        first.RC.declare_request(after[0])
        assert first.rc.put("m", "a prompt", "after forgetting", "AFTER-FORGET-XP3") != ""
        first.SC.declare_request(after[1])
        assert first.sc.put("after forgetting", "AFTER-FORGET-XP3", model="m", context_fingerprint="fp") != ""
        # A process that never read a file ever emptied: a request begun before its first write is refused, once;
        # one begun after it is put. A refused put writes nothing, and leaves the file free.
        fresh = _window(data)
        wins.append(fresh)
        before = fresh.RC.begin_request(), fresh.SC.begin_request()
        held = _generation(rc_file), _generation(sc_file)
        fresh.RC.declare_request(before[0])
        assert fresh.rc.put("m", "a prompt", "first read", "FIRST-READ-XP3") == ""
        fresh.SC.declare_request(before[1])
        assert fresh.sc.put("first read", "FIRST-READ-XP3", model="m", context_fingerprint="fp") == ""
        assert tj._held(data, b"FIRST-READ-XP3") == set()
        assert (_generation(rc_file), _generation(sc_file)) == held and _free(rc_file) and _free(sc_file)
        again = fresh.RC.begin_request(), fresh.SC.begin_request()
        fresh.RC.declare_request(again[0])
        assert fresh.rc.put("m", "a prompt", "second read", "SECOND-READ-XP3") != ""
        fresh.SC.declare_request(again[1])
        assert fresh.sc.put("second read", "SECOND-READ-XP3", model="m", context_fingerprint="fp") != ""
        # A process that never read a file never emptied: nothing counted.
        other = tmp_path / "other"
        untouched = _window(other)
        wins.append(untouched)
        _warm(untouched, "XP3-OTHER")
        newcomer = _window(other)
        wins.append(newcomer)
        mark = newcomer.RC.begin_request(), newcomer.SC.begin_request()
        newcomer.RC.declare_request(mark[0])
        assert newcomer.rc.put("m", "a prompt", "never emptied", "NEVER-EMPTIED-XP3") != ""
        newcomer.SC.declare_request(mark[1])
        assert newcomer.sc.put("never emptied", "NEVER-EMPTIED-XP3", model="m", context_fingerprint="fp") != ""
        # A file removed and made again, its generation back to 0: a request begun before is refused.
        mark = fresh.RC.begin_request(), fresh.SC.begin_request()
        for path in data.iterdir():
            if path.is_file():
                path.unlink()
        remade = _window(data)
        wins.append(remade)
        _warm(remade, "XP3-REMADE")
        assert (_generation(rc_file), _generation(sc_file)) == (0, 0)
        fresh.RC.declare_request(mark[0])
        assert fresh.rc.put("m", "a prompt", "remade", "REMADE-ANSWER-XP3") == ""
        fresh.SC.declare_request(mark[1])
        assert fresh.sc.put("remade", "REMADE-ANSWER-XP3", model="m", context_fingerprint="fp") == ""
        assert tj._held(data, b"REMADE-ANSWER-XP3") == set()
        # Drawn anew: never 0, never the one it replaces, in each cache.
        for module, cache, path in ((remade.RC, remade.rc, rc_file), (remade.SC, remade.sc, sc_file)):
            conn = sqlite3.connect(str(path))
            conn.execute("PRAGMA user_version = 5")
            conn.commit()
            conn.close()
            draws, bounds, drawn = [4, 0x7FFFFFFE], [], module.secrets
            module.secrets = SimpleNamespace(randbelow=lambda bound: bounds.append(bound) or draws.pop(0))
            try:
                cache.empty()
            finally:
                module.secrets = drawn
            assert bounds == [0x7FFFFFFF, 0x7FFFFFFF], f"{path.name} drew from {bounds}"
            assert _generation(path) == 0x7FFFFFFF, f"{path.name} kept {_generation(path)}"
        # Its own deletions counted once each, however they interleave: a forgetting here that comes the moment an
        # emptying's draw is committed takes neither for another process's, and a request begun after both is put
        # (round 2).
        hooks = {}
        mine = _window(tmp_path / "own", factory=_hooked(hooks))
        wins.append(mine)
        _warm(mine, "XP3-OWN")
        for cache in (mine.rc, mine.sc):
            began = _counts(mine)
            hooks["commit"] = lambda: (mine.rc.forget_conversation("conv-own"), mine.sc.forget_conversation("conv-own"))
            cache.empty()
            assert "commit" not in hooks, "the witness: the emptying committed nothing"
            raised = tuple(after - before for after, before in zip(_counts(mine), began))
            assert raised == ((1, 0) if cache is mine.rc else (0, 1)), f"its own deletions counted {raised}"
            after = _begin(mine)
            mine.RC.declare_request(after[0])
            assert mine.rc.put("m", "a prompt", "own after", "OWN-AFTER-XP3") != ""
            mine.SC.declare_request(after[1])
            assert mine.sc.put("own after", "OWN-AFTER-XP3", model="m", context_fingerprint="fp") != ""
    finally:
        for win in reversed(wins):
            win.close()


# ---------------------------------------------------------------------------
# XP4 -- another process's forgetting of a conversation refuses a put under way here
# ---------------------------------------------------------------------------

def _refused_both(win, marks, content, word, conversation_id=None):
    """Whether the request whose ``marks`` these are has its answer refused by both caches of ``win``."""
    win.RC.declare_request(marks[0])
    rc_key = win.rc.put("m", "a prompt", content, word, conversation_id=conversation_id)
    win.SC.declare_request(marks[1])
    sc_key = win.sc.put(content, word, model="m", conversation_id=conversation_id, context_fingerprint="fp")
    return rc_key == "" and sc_key == ""


def _begin(win):
    return win.RC.begin_request(), win.SC.begin_request()


def test_xp4_another_processs_forgetting_of_a_conversation_refuses_a_put_under_way_here(tmp_path):
    data = tmp_path / "data"
    wins = []
    try:
        here, there = _window(data), _window(data)
        wins += [here, there]
        _warm(here, "XP4")
        assert there.rc.get(_key(there, "a lookup there")) is None
        # A turn of the conversation here begins; the other process deletes the conversation.
        marks = _begin(here)
        there.rc.forget_conversation("conv-gone")
        there.sc.forget_conversation("conv-gone")
        assert "conv-gone" not in here.RC._FORGOTTEN, "the witness: the forgetting reached this process's memory"
        assert _refused_both(here, marks, "in the gone one", "GONE-ANSWER-XP4", "conv-gone")
        assert tj._held(data, b"GONE-ANSWER-XP4") == set()
        # A turn begun after it, in another conversation: put.
        after = _begin(here)
        here.RC.declare_request(after[0])
        assert here.rc.put("m", "a prompt", "in another", "OTHER-ANSWER-XP4", conversation_id="conv-kept") != ""
        here.SC.declare_request(after[1])
        assert here.sc.put("in another", "OTHER-ANSWER-XP4", model="m", conversation_id="conv-kept",
                           context_fingerprint="fp") != ""
        # A forgetting here reads what the other process drew before it: an emptying there, not read here yet, is
        # counted, never swallowed by this process's own draw.
        marks = _begin(here)
        there.rc.empty()
        there.sc.empty()
        here.rc.forget_conversation("conv-unrelated")
        here.sc.forget_conversation("conv-unrelated")
        assert _refused_both(here, marks, "after a forgetting here", "SWALLOWED-ANSWER-XP4")
        # Nor a forgetting there by one here.
        marks = _begin(here)
        there.rc.forget_conversation("conv-gone-too")
        there.sc.forget_conversation("conv-gone-too")
        here.rc.forget_conversation("conv-other")
        here.sc.forget_conversation("conv-other")
        assert _refused_both(here, marks, "in the gone one too", "GONE-TOO-ANSWER-XP4", "conv-gone-too")
        # A process whose first touch of a file ever emptied is a forgetting: a request it began before is refused.
        fresh = _window(data)
        wins.append(fresh)
        marks = _begin(fresh)
        fresh.rc.forget_conversation("conv-first-touch")
        fresh.sc.forget_conversation("conv-first-touch")
        assert _refused_both(fresh, marks, "first touch", "FIRST-TOUCH-ANSWER-XP4")
        assert tj._held(data, b"SWALLOWED-ANSWER-XP4", b"GONE-TOO-ANSWER-XP4", b"FIRST-TOUCH-ANSWER-XP4") == set()
        # A forgetting here that comes between a put's judgment in memory and its write: the put, judged again in its
        # write, is refused (round 2).
        hooks = {}
        mine = _window(tmp_path / "between", factory=_hooked(hooks))
        wins.append(mine)
        _warm(mine, "XP4-BETWEEN")
        for name, cache in (("response", mine.rc), ("semantic", mine.sc)):
            conversation = f"conv-between-{name}"
            hooks["begin"] = lambda c=cache, conv=conversation: c.forget_conversation(conv)
            marks = _begin(mine)
            if cache is mine.rc:
                mine.RC.declare_request(marks[0])
                key = cache.put("m", "a prompt", "between", "BETWEEN-ANSWER-XP4", conversation_id=conversation)
            else:
                mine.SC.declare_request(marks[1])
                key = cache.put("between", "BETWEEN-ANSWER-XP4", model="m", conversation_id=conversation,
                                context_fingerprint="fp")
            assert "begin" not in hooks, f"the witness: the {name} put opened no write"
            assert key == "", f"the {name} put of a conversation forgotten before its write was kept"
        assert tj._held(tmp_path / "between", b"BETWEEN-ANSWER-XP4") == set()
    finally:
        for win in reversed(wins):
            win.close()


# ---------------------------------------------------------------------------
# XP5 -- through the real executor, another process's deletion refuses the call under way
# ---------------------------------------------------------------------------

def test_xp5_through_the_real_executor_another_processs_deletion_refuses_the_call_under_way(tmp_path):
    # The other process: both caches of the folder, loaded first; its window closed, its modules kept.
    there = _window(tmp_path)
    there.close()
    world = tj._exec_world(tmp_path)
    try:
        rc_file, sc_file = tmp_path / "response_cache.db", tmp_path / "semantic_cache.db"
        world.loaded[_SC]._get_embedding = lambda text, model=None: [
            (b - 128) / 128 for b in hashlib.sha256(text.encode("utf-8")).digest()]
        assert "FIRST-ANSWER-XP5" in tj._ask(world, "a first FIRST-QUESTION-XP5", "FIRST-ANSWER-XP5")
        assert tj._in_file(rc_file, b"FIRST-ANSWER-XP5") and tj._in_file(sc_file, b"FIRST-QUESTION-XP5"), (
            "the witness: the first answer, or its question's embedding, was not put")
        # The other process deletes while this call is answered: its answer, and its question, kept nowhere.
        chat = world.client.chat

        def deleted_meanwhile(**kwargs):
            there.rc.empty()
            there.sc.empty()
            return chat(**kwargs)

        world.client.chat = deleted_meanwhile
        assert "DURING-ANSWER-XP5" in tj._ask(world, "asked DURING-QUESTION-XP5", "DURING-ANSWER-XP5")
        world.client.chat = chat
        assert tj._held(tmp_path, b"FIRST-ANSWER-XP5") == set(), "the witness: the other process emptied nothing"
        assert tj._held(tmp_path, b"DURING-ANSWER-XP5", b"DURING-QUESTION-XP5") == set(), (
            "a call under way when another process deleted put its answer or its question")
        # A turn begun before the other process's deletion: its later call refused. A call begun after: put.
        turn = tj._turn()
        assert "TURN-ANSWER-XP5" in tj._ask(world, "the turn's first", "TURN-ANSWER-XP5", run=turn)
        assert tj._in_file(rc_file, b"TURN-ANSWER-XP5")
        there.rc.empty()
        there.sc.empty()
        assert "LATER-ANSWER-XP5" in tj._ask(world, "the turn's later", "LATER-ANSWER-XP5", run=turn)
        assert tj._held(tmp_path, b"LATER-ANSWER-XP5") == set(), "a later call of a turn begun before put"
        assert "AFTER-ANSWER-XP5" in tj._ask(world, "asked after", "AFTER-ANSWER-XP5")
        assert tj._in_file(rc_file, b"AFTER-ANSWER-XP5") and tj._in_file(sc_file, b"AFTER-ANSWER-XP5")
    finally:
        world.close()


# ---------------------------------------------------------------------------
# XP6 -- nothing here waits on the file in vain
# ---------------------------------------------------------------------------

def _hold(path):
    """Another process's write in progress on the file at ``path``: its write lock, held until released."""
    conn = sqlite3.connect(str(path), timeout=0.1)
    conn.execute("BEGIN IMMEDIATE")
    return conn


def _counted(opened):
    """A connection that says it was opened."""
    class Counted(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            opened.append(1)
            super().__init__(*args, **kwargs)

    return Counted


def test_xp6_nothing_here_waits_on_the_file_in_vain(tmp_path):
    data = tmp_path / "data"
    rc_file, sc_file = data / "response_cache.db", data / "semantic_cache.db"
    opened = []
    here = _window(data, factory=_counted(opened))
    holders = []
    try:
        _warm(here, "XP6")
        # A put of a conversation this process forgot never opens the file (round 2).
        here.rc.forget_conversation("conv-forgotten-xp6")
        here.sc.forget_conversation("conv-forgotten-xp6")
        opened.clear()
        assert _refused_both(here, _begin(here), "forgotten", "FORGOTTEN-XP6", "conv-forgotten-xp6")
        assert opened == [], f"a put this process refuses opened the file {len(opened)} time(s)"
        marks = _begin(here)
        here.rc.empty()
        here.sc.empty()
        # Another process writes both files. A put this process's own count already refuses returns at once: the
        # file is not opened, its write lock not waited for.
        holders += [_hold(rc_file), _hold(sc_file)]
        opened.clear()
        began = time.monotonic()
        assert _refused_both(here, marks, "known stale", "KNOWN-STALE-XP6")
        here.SC.declare_request(marks[1])
        assert here.sc.store_embedding("stale-key", "m", "KNOWN-STALE-QUESTION-XP6", [1.0, 0.0]) is False
        waited = time.monotonic() - began
        assert waited < _RACE, f"a put this process already refuses waited {waited:.2f} s on the file"
        assert opened == [], f"a put this process refuses opened the file {len(opened)} time(s)"
        assert not getattr(here.sc, "_layer_failure_logged", False), "a refused put was taken for a storage failure"
        for holder in holders:
            holder.rollback()
        # An emptying, a forgetting or a put here that waits for the other process's write holds no lock the
        # requests here take (the forgetting and the put: round 2).
        def put_rc():
            here.RC.declare_request(here.RC.begin_request())
            return here.rc.put("m", "a prompt", "a waiting put", "WAITING-PUT-XP6") != ""

        def put_sc():
            here.SC.declare_request(here.SC.begin_request())
            return here.sc.put("a waiting put", "WAITING-PUT-XP6", model="m", context_fingerprint="fp") != ""

        def ask_rc():
            return here.rc.get(_key(here, "while it waits"))

        def ask_sc():
            return here.sc.get("while it waits", context_fingerprint="fp")

        cases = (("the emptying", rc_file, here.rc.empty, ask_rc), ("the emptying", sc_file, here.sc.empty, ask_sc),
                 ("the forgetting", rc_file, lambda: here.rc.forget_conversation("conv-waiting"), ask_rc),
                 ("the forgetting", sc_file, lambda: here.sc.forget_conversation("conv-waiting"), ask_sc),
                 ("the put", rc_file, put_rc, ask_rc), ("the put", sc_file, put_sc, ask_sc))
        for what, path, act, ask in cases:
            holder = _hold(path)
            holders.append(holder)
            ended = {}
            worker = threading.Thread(target=lambda a=act: ended.setdefault("done", a()), name="xp6-waiting")
            worker.start()
            worker.join(_RACE)
            assert worker.is_alive(), f"the witness: {what} did not wait for the other process's write ({path.name})"
            began = time.monotonic()
            _begin(here)
            assert ask() is None
            waited = time.monotonic() - began
            holder.rollback()
            worker.join(_WAIT)
            assert not worker.is_alive() and "done" in ended, f"{what} never ended once the write was done"
            assert waited < _RACE, f"a request here waited {waited:.2f} s behind {what} waiting on {path.name}"
            assert what != "the put" or ended["done"] is True, f"{what} was refused once the write was done"
    finally:
        for holder in holders:
            holder.close()
        here.close()
