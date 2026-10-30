"""Contracts for a turn whose later calls are judged by the turn's beginning in the caches.

Each answer a cache takes is judged by the beginning of the request that made it: a cache emptied since -- a deletion
of what the answer may have been drawn from -- refuses it. A turn makes many requests: a pipeline's steps, the agentic
executor's passes, the self-correction's pass on its draft; each marked its own beginning, after the calls before it
drew on the stores, though its prompt repeats their words. Now the first call of a turn marks it, before anything of
the turn draws on a store, and every later call of the turn takes those marks; a turn is known by the stop all its
calls share. The executor's cascade and speculative paths mark their own request.

  * TJ1 -- A LATER CALL OF A TURN IS JUDGED BY THE TURN'S BEGINNING: through the real executor and both real caches,
    a later call of a turn, on the turn's thread or another, has its answer refused after an emptying the turn
    outlived, and its question, which repeats the turn's words, is kept nowhere; a call of another turn, or of none,
    begun after it is put; a turn marked before its first call, then emptied, has that call's answer refused.
  * TJ2 -- THE ROUTE, THE PIPELINE AND THE AGENTIC EXECUTOR MARK THE TURN FIRST AND HAND IT ON: the real pipeline
    runner marks a turn before its first step, the real agentic executor before its first pass, each later step or
    pass begins from those marks; every pass hands the executor its turn; the chat route marks each turn it opens.
  * TJ3 -- A TURN'S MARKS GO WITH IT: weakly held; a stop that is not an Event, or a call with no turn, is judged by
    its own beginning; a cache begins a later call from its turn's count only when that count is older, on any
    thread, and from its first count when the turn began before it could mark; a turn is marked under one lock from
    counts read without the caches' locks, so marking never waits on a put.
  * TJ4 -- SUPERSEDES CF5 of ``tests/test_cache_forgetting_contracts.py``: its body whole, the executor's first act
    taking the mark with its run.
  * TJ5 -- THE CASCADE'S AND THE SPECULATIVE PATH'S ANSWERS ARE JUDGED BY THEIR OWN BEGINNING: one emptied during
    the call is refused, one begun after an emptying is put whatever an earlier request on its thread saw.
  * TJ6 -- A DELETION IN A FOLDER WHOSE CACHES HAVE NO FILE YET REFUSES A REQUEST UNDER WAY: through the real knowledge
    base, its caches loaded, their files not made; each store asks when a cache is loaded; another folder's
    deletion leaves them.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import ast
import gc
import hashlib
import sqlite3
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import seed_registry  # noqa: E402

BUDGET_S = {
    "test_tj1_a_later_call_of_a_turn_is_judged_by_the_turns_beginning": 10.0,
    "test_tj2_the_pipeline_and_the_agentic_executor_mark_the_turn_first_and_hand_it_on": 10.0,
    "test_tj3_a_turns_marks_go_with_it_and_a_turn_that_cannot_be_held_is_judged_call_by_call": 5.0,
    "test_tj4_an_answer_asked_before_an_emptying_is_never_put_and_the_executor_marks_with_its_run": 10.0,
    "test_tj5_the_cascades_and_the_speculative_paths_answers_are_judged_by_their_own_beginning": 10.0,
    "test_tj6_a_deletion_in_a_folder_whose_caches_have_no_file_yet_refuses_a_request_under_way": 10.0,
}

_WAIT = 3.0
_RC, _SC = "opti_oignon.response_cache", "opti_oignon.semantic_cache"
_CACHES = {
    _RC: source("response_cache.py"),
    _SC: source("semantic_cache.py"),
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
    """Which of ``words`` any file under ``data`` still holds."""
    found = set()
    for path in data.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            found.update(w for w in words if w in raw)
    return found


def _in_file(path, word):
    return path.is_file() and word in path.read_bytes()


def _routing(model="m"):
    return SimpleNamespace(model=model, task_type="general", temperature=0.2, prompt_variant="standard", timeout=30,
                           routing_reason="contract", images=None)


def _turn(results=None):
    """A turn as the chat route hands it on: its stop, its results."""
    return SimpleNamespace(stop=threading.Event(), results={} if results is None else results)


def _drive(gen):
    chunks = []
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as stop:
        return chunks, stop.value


def _base_seeds(data):
    return {
        "opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain),
        "opti_oignon.config": _module("opti_oignon.config", DATA_DIR=data, config=SimpleNamespace(
            get_model=lambda *a, **k: "test-model:1b", get_temperature=lambda *a, **k: 0.2)),
    }


def _ready(loaded):
    """Both caches of a window, the semantic one matching exactly (no embeddings)."""
    sc = loaded[_SC].semantic_cache
    sc._config.update(enabled=True, semantic_match_enabled=False, exact_match_enabled=True)
    sc.embeddings_available = False
    return loaded[_RC].response_cache, sc


def _empty(loaded):
    """A deletion, as the caches see one: both emptied."""
    rc, sc = loaded[_RC].response_cache, loaded[_SC].semantic_cache
    rc.empty()
    sc.empty()


def _count(loaded):
    return {name: loaded[name]._EMPTIED[0] for name in (_RC, _SC)}


class _Client:
    """The model behind the registry: answers each call with ``answer``."""

    def __init__(self):
        self.answer = "an answer"
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        return iter([{"message": {"content": self.answer}}])


class _Recorder:
    def record(self, **fields):
        return True


def _exec_world(data):
    """The real executor over both real caches and their forgetting, the caches' files in ``data``."""
    client = _Client()
    seeded = _base_seeds(data)
    seeded.update({
        "opti_oignon.router": _module("opti_oignon.router", RoutingResult=type("RoutingResult", (), {})),
        "opti_oignon.context_ledger": _module("opti_oignon.context_ledger", get_context_ledger=_Recorder),
        "opti_oignon.conversation": _module("opti_oignon.conversation", conversation_manager=SimpleNamespace(
            get_context_messages=lambda conv_id: [], add_message=lambda *a, **k: None,
            update_conversation_metadata=lambda *a, **k: None, get_conversation=lambda conv_id: None)),
        "opti_oignon.memory.auto_capture": _module("opti_oignon.memory.auto_capture",
                                                   maybe_capture=lambda conv_id, messages: None),
        "opti_oignon.memory.librarian": _module("opti_oignon.memory.librarian",
                                                maybe_curate=lambda conv_id, messages: None,
                                                memory_block=lambda conv_id, question: "",
                                                onion_enabled=lambda: False),
    })
    seed_registry(seeded, client)
    loaded, close = isolate(
        targets={**_CACHES, "opti_oignon.context_dedup": source("context_dedup.py"),
                 "opti_oignon.executor": source("executor.py")},
        seeded=seeded, packages=("opti_oignon.memory",),
    )
    _ready(loaded)
    return SimpleNamespace(loaded=loaded, client=client, executor=loaded["opti_oignon.executor"],
                           ca=loaded["opti_oignon.cached_answers"], close=close)


def _ask(world, question, answer, **kwargs):
    """One executor call, answered with ``answer``; the reply."""
    world.client.answer = answer
    _chunks, (_refined, response) = _drive(world.executor.Executor().execute(question, _routing(), refine=False,
                                                                             **kwargs))
    return response


# ---------------------------------------------------------------------------
# TJ1 -- a later call of a turn is judged by the turn's beginning
# ---------------------------------------------------------------------------

def test_tj1_a_later_call_of_a_turn_is_judged_by_the_turns_beginning(tmp_path):
    world = _exec_world(tmp_path)
    try:
        rc_file, sc_file = tmp_path / "response_cache.db", tmp_path / "semantic_cache.db"
        # The embedding model answers, so a question's embedding is stored beside an answer the cache takes.
        world.loaded[_SC]._get_embedding = lambda text, model=None: [
            (b - 128) / 128 for b in hashlib.sha256(text.encode("utf-8")).digest()]
        turn = _turn()
        # The turn's first call: answered and put, in both caches, its question's embedding beside it.
        assert "FIRST-ANSWER-T1" in _ask(world, "the first step FIRST-QUESTION-T1", "FIRST-ANSWER-T1", run=turn)
        assert _in_file(rc_file, b"FIRST-ANSWER-T1") and _in_file(sc_file, b"FIRST-ANSWER-T1"), (
            "the witness: the turn's first answer reached neither cache")
        assert _in_file(sc_file, b"FIRST-QUESTION-T1"), "the witness: no question's embedding was stored"
        # A deletion: both caches emptied, overwritten.
        assert world.ca.forget_cached_answers(tmp_path) == []
        assert _held(tmp_path, b"FIRST-ANSWER-T1") == set()
        # The turn's later call -- a step whose prompt repeats the words the first drew --: answered, never put, and
        # its question, which repeats them, kept nowhere either.
        assert "LATER-ANSWER-T1" in _ask(world, "the second step LATER-QUESTION-T1", "LATER-ANSWER-T1", run=turn)
        assert _held(tmp_path, b"LATER-ANSWER-T1") == set(), "a later call of the turn put its answer"
        assert _held(tmp_path, b"LATER-QUESTION-T1") == set(), "a refused answer's question was stored"
        # And on another thread: the turn's marks are the turn's, not a thread's.
        replies = []
        worker = threading.Thread(target=lambda: replies.append(
            _ask(world, "the third step", "THREAD-ANSWER-T1", run=turn)), name="tj1-later-step")
        worker.start()
        worker.join(_WAIT)
        assert not worker.is_alive() and replies and "THREAD-ANSWER-T1" in replies[0]
        assert _held(tmp_path, b"THREAD-ANSWER-T1") == set(), "a later call of the turn on another thread put"
        # Another turn begun after the deletion, the same question: put. A call with no turn: put.
        assert "OTHER-ANSWER-T1" in _ask(world, "the second step LATER-QUESTION-T1", "OTHER-ANSWER-T1", run=_turn())
        assert _in_file(rc_file, b"OTHER-ANSWER-T1") and _in_file(sc_file, b"OTHER-ANSWER-T1")
        assert "ALONE-ANSWER-T1" in _ask(world, "a question alone", "ALONE-ANSWER-T1")
        assert _in_file(rc_file, b"ALONE-ANSWER-T1") and _in_file(sc_file, b"ALONE-ANSWER-T1")
        # A turn marked before its first call -- the pipeline's, the agentic executor's beginning --, then a
        # deletion: its first executor call is refused too.
        early = _turn()
        world.ca.begin_cached_request(early)
        assert world.ca.forget_cached_answers(tmp_path) == []
        assert "EARLY-ANSWER-T1" in _ask(world, "asked early", "EARLY-ANSWER-T1", run=early)
        assert _held(tmp_path, b"EARLY-ANSWER-T1") == set(), "a call of a turn marked before the deletion put"
    finally:
        world.close()
    # Read in the syntax tree: the question's embedding is judged like a put, the request's mark handed to the
    # semantic cache right before it.
    tree = ast.parse(source("executor.py").read_text(encoding="utf-8"))
    (execute,) = [node for cls in tree.body if isinstance(cls, ast.ClassDef) for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "execute"]
    stores = 0
    for block in ast.walk(execute):
        statements = getattr(block, "body", None)
        if not isinstance(statements, list):
            continue
        for index, statement in enumerate(statements):
            if isinstance(statement, ast.Expr) and "_semantic_cache.store_embedding(" in ast.unparse(statement):
                stores += 1
                assert index > 0 and ast.unparse(statements[index - 1]) == (
                    f"_declare_cached_request(_cached_marks, '{_SC}')"), "the embedding stored without the mark"
    assert stores == 1, stores


# ---------------------------------------------------------------------------
# TJ2 -- the pipeline and the agentic executor mark the turn first and hand it on
# ---------------------------------------------------------------------------

class _EmptiedOnce(dict):
    """A turn's results, which empty both caches the first time ``hook`` is touched: the moment after the runner
    took its turn in hand and before any of its calls began."""

    def __init__(self, empty, hook):
        super().__init__()
        self._empty, self._hook, self.emptied = empty, hook, 0

    def _once(self):
        if not self.emptied:
            self.emptied = 1
            self._empty()

    def pop(self, *args):
        if self._hook == "pop":
            self._once()
        return super().pop(*args)

    def update(self, *args, **kwargs):
        if self._hook == "update":
            self._once()
        return super().update(*args, **kwargs)


def _leaf_world(tmp_path, name, filename):
    loaded, close = isolate(targets={**_CACHES, f"opti_oignon.{name}": source(filename)},
                            seeded=_base_seeds(tmp_path), packages=("opti_oignon",))
    _ready(loaded)
    return loaded, close


def test_tj2_the_pipeline_and_the_agentic_executor_mark_the_turn_first_and_hand_it_on(tmp_path):
    # The pipeline runner: its turn marked before the first step; each step begins from those marks.
    loaded, close = _leaf_world(tmp_path / "pipeline", "pipelines", "pipelines.py")
    try:
        pl, ca = loaded["opti_oignon.pipelines"], loaded["opti_oignon.cached_answers"]
        pl._resolve_emergency_stop = lambda: None
        pl._resolve_resource_governor = lambda: None

        class Steps:
            def __init__(self):
                self.marks, self.runs = [], []

            def execute(self, **kw):
                self.runs.append(kw.get("run"))
                self.marks.append(ca.begin_cached_request(kw.get("run")))
                if len(self.marks) == 1:
                    _empty(loaded)
                yield f"the output of step {len(self.marks)}"

        router = SimpleNamespace(enabled=False, override_routing=lambda routing, step_type: routing)
        pipe = pl.ExecutionPipeline(id="p", name="P", steps=[
            pl.ExecutionStep("direct", label="one"), pl.ExecutionStep("direct", label="two"),
            pl.ExecutionStep("direct", label="three"),
        ])
        began = _count(loaded)
        turn = _turn(_EmptiedOnce(lambda: _empty(loaded), "pop"))
        steps = Steps()
        list(pl.PipelineRunner(agentic_executor=steps, smart_router=router).execute(
            pipe, "hello", SimpleNamespace(model="m"), run=turn))
        assert turn.results.emptied == 1, "the witness: nothing emptied the caches before the first step"
        assert len(steps.marks) == 3 and all(run is turn for run in steps.runs), steps.runs
        assert steps.marks == [began] * 3, f"the steps of a turn began at {steps.marks}, the turn at {began}"
        assert _count(loaded) != began
        ca.declare_cached_request(steps.marks[-1], _RC)
        assert loaded[_RC].response_cache.put("m", "a prompt", "the last step", "STEP-ANSWER-T2") == ""
        assert _held(tmp_path / "pipeline", b"STEP-ANSWER-T2") == set()
        # The witness: a run with no turn, each step judged by its own beginning.
        alone = Steps()
        list(pl.PipelineRunner(agentic_executor=alone, smart_router=router).execute(
            pipe, "hello", SimpleNamespace(model="m")))
        assert alone.marks[1] == _count(loaded) and alone.marks[0] != alone.marks[1], alone.marks
    finally:
        close()
    # The agentic executor: its turn marked before its first pass; a later pass of the turn begins from those marks.
    loaded, close = _leaf_world(tmp_path / "agentic", "agentic_executor", "agentic_executor.py")
    try:
        ae, ca = loaded["opti_oignon.agentic_executor"], loaded["opti_oignon.cached_answers"]

        class Base:
            def __init__(self):
                self.marks, self.runs = [], []

            def execute(self, **kw):
                self.runs.append(kw.get("run"))
                self.marks.append(ca.begin_cached_request(kw.get("run")))
                yield "an answer of the pass"

        began = _count(loaded)
        turn = _turn(_EmptiedOnce(lambda: _empty(loaded), "update"))
        base = Base()
        agent = ae.AgenticExecutor(executor=base, default_model="m")
        assert [c for c in agent.execute("q", _routing(), run=turn) if isinstance(c, str)]
        assert turn.results.emptied == 1, "the witness: nothing emptied the caches before the first pass"
        assert base.marks == [began], f"the turn's first pass began at {base.marks}, the turn at {began}"
        assert base.runs[0].stop is turn.stop, "the pass was not handed the turn"
        _empty(loaded)
        list(agent.execute("q again", _routing(), run=turn))
        assert base.marks[1] == began, f"the turn's later pass began at {base.marks[1]}, the turn at {began}"
        fresh = _turn()
        list(agent.execute("q", _routing(), run=fresh))
        assert base.marks[2] == _count(loaded), "a new turn did not begin at its own beginning"
    finally:
        close()
    # Read in the syntax tree: the runner and the agentic executor mark the turn before anything of it, every pass
    # hands the executor the turn, and both marks reach the caches' forgetting.
    pipelines = ast.parse(source("pipelines.py").read_text(encoding="utf-8"))
    (runner,) = [n for c in pipelines.body if isinstance(c, ast.ClassDef) and c.name == "PipelineRunner"
                 for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "execute"]
    assert ast.unparse(runner.body[1]) == "_begin_cached_turn(run)", ast.unparse(runner.body[1])
    agentic = ast.parse(source("agentic_executor.py").read_text(encoding="utf-8"))
    (execute,) = [n for c in agentic.body if isinstance(c, ast.ClassDef) and c.name == "AgenticExecutor"
                  for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "execute"]
    statements = [ast.unparse(s) for s in execute.body]
    built = next(i for i, s in enumerate(statements) if s.startswith("turn = _Turn("))
    assert statements[built + 1] == "_begin_cached_turn(turn)", statements[built + 1]
    passes = [n for n in ast.walk(agentic) if isinstance(n, ast.Call)
              and ast.unparse(n.func) == "self._executor.execute"]
    assert len(passes) == 4, len(passes)
    for call in passes:
        handed = {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg}
        assert handed.get("run") == "turn", f"a pass hands the executor no turn: {ast.unparse(call)[:80]}"
    for tree in (pipelines, agentic):
        (helper,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_begin_cached_turn"]
        assert "mark_cached_turn(run)" in ast.unparse(helper)
    # The chat route marks each turn as it opens it, before anything of the turn -- the coding agent's phases too --
    # draws on a store.
    route = ast.parse(source("api/routes_chat.py").read_text(encoding="utf-8"))
    (opened,) = [n for n in route.body if isinstance(n, ast.FunctionDef) and n.name == "_open_turn"]
    statements = [ast.unparse(s) for s in opened.body]
    assert statements[1:3] == ["turn = ChatTurn(conversation_id)", "_begin_cached_turn(turn)"], statements[:3]
    (helper,) = [n for n in route.body if isinstance(n, ast.FunctionDef) and n.name == "_begin_cached_turn"]
    assert "mark_cached_turn(turn)" in ast.unparse(helper)


# ---------------------------------------------------------------------------
# TJ3 -- a turn's marks go with it; a turn that cannot be held is judged call by call
# ---------------------------------------------------------------------------

class _Unheld:
    """A stop that cannot be referred to weakly."""

    __slots__ = ()


class _OtherStop:
    """A stop of another kind than the Event every layer of a turn shares."""


def test_tj3_a_turns_marks_go_with_it_and_a_turn_that_cannot_be_held_is_judged_call_by_call(tmp_path, monkeypatch):
    loaded, close = isolate(targets=dict(_CACHES), seeded=_base_seeds(tmp_path), packages=("opti_oignon",))
    try:
        ca, rcm = loaded["opti_oignon.cached_answers"], loaded[_RC]
        rc, _sc = _ready(loaded)
        # Weakly held: the turn's marks go when the turn does.
        before = len(ca._TURNS)
        turn = _turn()
        first = ca.begin_cached_request(turn)
        assert len(ca._TURNS) == before + 1 and ca._TURNS[turn.stop] == first
        del turn
        gc.collect()
        assert len(ca._TURNS) == before, "a turn's marks outlived it"
        # A stop that cannot be held, a stop of another kind, and a call with no turn: each call judged by its own
        # beginning.
        for run in (SimpleNamespace(stop=_Unheld()), SimpleNamespace(stop=_OtherStop()), SimpleNamespace(), None):
            began = ca.begin_cached_request(run)
            _empty(loaded)
            assert ca.begin_cached_request(run) == _count(loaded) != began, run
        assert len(ca._TURNS) == before
        # A cache begins a later call from its turn's count only when that count is older.
        now = rcm._EMPTIED[0]
        assert rcm.begin_request(now - 1) == now - 1 and rcm._REQUEST.count == now - 1
        assert rcm.begin_request(now + 5) == now and rcm.begin_request() == now
        # On any thread: a later call of the turn begun on another thread notes its lookups with the turn's count,
        # so an answer put there with no mark handed back is refused after an emptying the turn outlived.
        turn = _turn()
        ca.begin_cached_request(turn)
        _empty(loaded)
        outcome = []

        def later_call():
            ca.begin_cached_request(turn)
            key = rc.make_cache_key("m", "a prompt", "asked on another thread")
            outcome.append(rc.get(key))
            outcome.append(rc.put("m", "a prompt", "asked on another thread", "THREAD-ANSWER-T3"))

        worker = threading.Thread(target=later_call, name="tj3-later-call")
        worker.start()
        worker.join(_WAIT)
        assert not worker.is_alive() and outcome == [None, ""], outcome
        assert _held(tmp_path, b"THREAD-ANSWER-T3") == set()
        # A cache the turn's first call could not mark -- not loaded yet -- begins a later call from its first count:
        # every emptying of it came after the turn began. A turn whose first call marked nothing is held all the same.
        late = _turn()
        monkeypatch.setattr(ca, "_CACHE_MODULES", (_RC,))
        first = ca.begin_cached_request(late)
        assert sorted(first) == [_RC]
        bare = _turn()
        monkeypatch.setattr(ca, "_CACHE_MODULES", ())
        assert ca.begin_cached_request(bare) == {} and ca._TURNS[bare.stop] == {}
        monkeypatch.setattr(ca, "_CACHE_MODULES", (_RC, _SC))
        _empty(loaded)
        assert ca.begin_cached_request(late) == {_RC: first[_RC], _SC: 0}
        assert ca.begin_cached_request(bare) == {_RC: 0, _SC: 0}
        # A turn is marked from the caches' counts read under the turn's lock, so two calls of one turn marking at
        # once take the same marks; a request then begins outside it.
        seen = []
        real_emptied, real_begin = rcm.emptied, rcm.begin_request
        monkeypatch.setattr(rcm, "emptied", lambda: seen.append(("count", ca._TURNS_LOCK.locked())) or real_emptied())
        monkeypatch.setattr(rcm, "begin_request",
                            lambda since=None: seen.append(("begin", ca._TURNS_LOCK.locked())) or real_begin(since))
        ca.begin_cached_request(_turn())
        assert seen == [("count", True), ("begin", False)], seen
        monkeypatch.setattr(rcm, "emptied", real_emptied)
        # Marking a turn never waits on a cache's own lock, which a put holds over its disk work: the chat route marks
        # each turn on its event loop.
        marked = []
        with rcm._MARKS_LOCK, loaded[_SC]._MARKS_LOCK:
            worker = threading.Thread(target=lambda: marked.append(ca.mark_cached_turn(_turn())), name="tj3-mark")
            worker.start()
            worker.join(_WAIT)
            assert not worker.is_alive(), "marking a turn waited on a cache's lock"
        assert marked and sorted(marked[0]) == [_RC, _SC], marked
    finally:
        close()


# ---------------------------------------------------------------------------
# TJ4 -- supersedes CF5: an answer asked before an emptying is never put; the executor marks with its run
# ---------------------------------------------------------------------------

def test_tj4_an_answer_asked_before_an_emptying_is_never_put_and_the_executor_marks_with_its_run(
        tmp_path, monkeypatch):
    import test_cache_forgetting_contracts as cf
    import test_knowledge_forgetting_contracts as kb

    _cache, _held_cf, _loaded = cf._cache, cf._held, cf._loaded
    cf._with_caches(kb, monkeypatch)
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
        assert _held_cf(world.data, b"BEFORE-ANSWER-9") == set()
        assert _held_cf(world.data, b"AFTER-ANSWER-9", b"WARMED-ANSWER-9") == {b"AFTER-ANSWER-9",
                                                                              b"WARMED-ANSWER-9"}
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
        rc_module, sc_module = _loaded(world)[_RC], _loaded(world)[_SC]
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
        assert _held_cf(world.data, b"TWICE-ANSWER-9", b"DRAWN-ANSWER-9", b"MANY-ANSWER-9") == set()
        # Two requests on one key: each is judged by its own beginning, handed back at its put, never by the other's.
        ca = _loaded(world)["opti_oignon.cached_answers"]
        marks_a = ca.begin_cached_request()
        assert sorted(marks_a) == [_RC, _SC], marks_a
        shared = rc.make_cache_key("m", "a prompt", "asked by two")
        assert rc.get(shared) is None and sc.get("asked by two", context_fingerprint="fp-two") is None
        marks_b = ca.begin_cached_request()
        assert rc.get(shared) is None and sc.get("asked by two", context_fingerprint="fp-two") is None
        rc.empty()
        sc.empty()
        for marks in (marks_a, marks_b):
            ca.declare_cached_request(marks, _RC)
            assert rc.put("m", "a prompt", "asked by two", "TWO-BEFORE-9") == ""
            ca.declare_cached_request(marks, _SC)
            assert sc.put("asked by two", "TWO-BEFORE-9", model="m", context_fingerprint="fp-two") == ""
        late = ca.begin_cached_request()
        assert rc.get(shared) is None and sc.get("asked by two", context_fingerprint="fp-two") is None
        early = marks_a
        ca.declare_cached_request(late, _RC)
        assert rc.put("m", "a prompt", "asked by two", "TWO-AFTER-9") != ""
        ca.declare_cached_request(late, _SC)
        assert sc.put("asked by two", "TWO-AFTER-9", model="m", context_fingerprint="fp-two") != ""
        ca.declare_cached_request(early, _RC)
        assert rc.put("m", "a prompt", "asked by two", "TWO-EARLY-9") == ""
        ca.declare_cached_request(early, _SC)
        assert sc.put("asked by two", "TWO-EARLY-9", model="m", context_fingerprint="fp-two") == ""
        assert _held_cf(world.data, b"TWO-BEFORE-9", b"TWO-EARLY-9") == set() and _held_cf(world.data,
                                                                                          b"TWO-AFTER-9")
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
    # The executor marks its request first -- with its run, so a later call of a turn takes the turn's marks --,
    # before its prompt draws on any store, and hands its marks back right before each of its puts.
    tree = ast.parse(source("executor.py").read_text(encoding="utf-8"))
    (execute,) = [node for cls in tree.body if isinstance(cls, ast.ClassDef) for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "execute"]
    assert ast.unparse(execute.body[1]) == "_cached_marks = _begin_cached_request(run)", ast.unparse(execute.body[1])
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
                    cache = _SC if "_semantic_cache.put" in ast.unparse(statement) else _RC
                    assert index > 0 and ast.unparse(statements[index - 1]) == (
                        f"_declare_cached_request(_cached_marks, '{cache}')"), ast.unparse(statement)[:80]
    assert puts == 2, puts
    for helper, delegate in (("_begin_cached_request", "begin_cached_request(run)"),
                             ("_declare_cached_request", "declare_cached_request")):
        (function,) = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == helper]
        assert delegate in ast.unparse(function), helper


# ---------------------------------------------------------------------------
# TJ5 -- the cascade's and the speculative path's answers are judged by their own beginning
# ---------------------------------------------------------------------------

def test_tj5_the_cascades_and_the_speculative_paths_answers_are_judged_by_their_own_beginning(tmp_path):
    world = _exec_world(tmp_path)
    try:
        ex = world.executor
        sc_file = tmp_path / "semantic_cache.db"
        during = {"empty": False}

        def answered(word):
            if during["empty"]:
                _empty(world.loaded)
            return word

        ex.CASCADING_AVAILABLE = True
        ex._cascading_inference = SimpleNamespace(enabled=True, tiers=[], cascade=lambda query, task_type: (
            SimpleNamespace(final_response=answered(f"CASCADE-{query}"), model_used="m", tier_name="t")))
        ex.SPECULATIVE_AVAILABLE = True
        ex._speculative_generator = SimpleNamespace(
            enabled=True, verify_model="", draft_model="",
            generate=lambda *a, **k: SimpleNamespace(
                final_response=answered(f"SPECULATIVE-{k.get('query', a[0] if a else '')}"),
                draft_accepted=True, draft_model="m", verify_model="m"))
        paths = (("cascade", "execute_cascade"), ("speculative", "execute_speculative"))
        for word, method in paths:
            # An earlier request on this thread began before an emptying: a call begun after it is put all the same.
            world.ca.begin_cached_request()
            _empty(world.loaded)
            during["empty"] = False
            result = getattr(ex.Executor(), method)(f"{word}-after")
            assert result is not None and result.final_response.endswith(f"{word}-after")
            assert _in_file(sc_file, result.final_response.encode()), f"the {word} answer begun after an emptying"
            # Emptied during the call: refused.
            during["empty"] = True
            result = getattr(ex.Executor(), method)(f"{word}-during")
            assert result is not None
            assert _held(tmp_path, result.final_response.encode()) == set(), f"the {word} answer emptied during it"
        # Read in the syntax tree: each path marks its request first and hands the mark back right before its put.
        tree = ast.parse(source("executor.py").read_text(encoding="utf-8"))
        for _word, method in paths:
            (function,) = [n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body
                           if isinstance(n, ast.FunctionDef) and n.name == method]
            assert ast.unparse(function.body[1]) == "_cached_marks = _begin_cached_request()", method
            puts = 0
            for block in ast.walk(function):
                statements = getattr(block, "body", None)
                if not isinstance(statements, list):
                    continue
                for index, statement in enumerate(statements):
                    if "_semantic_cache.put(" in ast.unparse(statement) and not isinstance(
                            statement, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                        puts += 1
                        assert index > 0 and ast.unparse(statements[index - 1]) == (
                            f"_declare_cached_request(_cached_marks, '{_SC}')"), method
            assert puts == 1, (method, puts)
    finally:
        world.close()


# ---------------------------------------------------------------------------
# TJ6 -- a deletion in a folder whose caches have no file yet refuses a request under way
# ---------------------------------------------------------------------------

def test_tj6_a_deletion_in_a_folder_whose_caches_have_no_file_yet_refuses_a_request_under_way(tmp_path, monkeypatch):
    import test_cache_forgetting_contracts as cf
    import test_knowledge_forgetting_contracts as kb

    cf._with_caches(kb, monkeypatch)
    world = kb._world(tmp_path)
    try:
        kb._plant(world, "doc-gone", kb.GONE, kb.GONE_FILE, kb.GONE_QUERY)
        loaded = cf._loaded(world)
        rc, sc = _ready(loaded)
        ca = loaded["opti_oignon.cached_answers"]
        files = (world.data / "response_cache.db", world.data / "semantic_cache.db")
        assert not any(f.exists() for f in files), "the witness: a cache's file was already made"
        # A request under way, marked before its lookup; the document it drew on removed.
        marks = ca.begin_cached_request()
        removed, left = world.store.forget_document("doc-gone")
        assert removed and left == [], left
        assert _count(loaded) != marks, "the removal counted no emptying"
        # Its lookup makes the files; its answer is refused in both.
        assert rc.get(rc.make_cache_key("m", "a prompt drawn before", "asked late")) is None
        assert sc.get("asked late", context_fingerprint="fp-late") is None
        ca.declare_cached_request(marks, _RC)
        assert rc.put("m", "a prompt drawn before", "asked late", "NOFILE-ANSWER-T6") == ""
        ca.declare_cached_request(marks, _SC)
        assert sc.put("asked late", "NOFILE-ANSWER-T6", model="m", context_fingerprint="fp-late") == ""
        assert _held(world.data, b"NOFILE-ANSWER-T6") == set()
        # A request begun after the removal: put.
        after = ca.begin_cached_request()
        ca.declare_cached_request(after, _RC)
        assert rc.put("m", "a prompt drawn after", "asked after", "AFTER-ANSWER-T6") != ""
        assert _held(world.data, b"AFTER-ANSWER-T6")
        # A deletion in another folder, which holds no cache file, leaves these caches and their count alone.
        before = _count(loaded)
        assert ca.forget_cached_answers(tmp_path / "elsewhere") == []
        assert _count(loaded) == before and _held(world.data, b"AFTER-ANSWER-T6"), "another folder's caches emptied"
        # A cache module whose instance names no file -- a stand-in -- is no folder's: a deletion reports nothing of it.
        monkeypatch.setattr(loaded[_RC], "response_cache", SimpleNamespace())
        assert ca.forget_cached_answers(tmp_path / "elsewhere") == []
    finally:
        world.restore()
    # Read in the syntax tree: each store that empties the caches asks when a cache is loaded, not only when a file is
    # there; the facts store the same way.
    files = "(data / 'response_cache.db').exists() or (data / 'semantic_cache.db').exists()"
    loaded_any = ("any((sys.modules.get(f'opti_oignon.{name}') is not None for name in ('response_cache', "
                  "'semantic_cache')))")
    for name in ("rag_store.py", "projects.py", "notes/notes_store.py", "agent/skills.py"):
        tree = ast.parse(source(name).read_text(encoding="utf-8"))
        (helper,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_forget_cached_answers"]
        assert ast.unparse(helper.body[1].test) == f"not (_caches_loaded() or {files})", name
        (loaded_fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_caches_loaded"]
        assert ast.unparse(loaded_fn.body[1]) == f"return {loaded_any}", name
    facts = ast.parse(source("memory/canonical_store.py").read_text(encoding="utf-8"))
    (method,) = [n for c in facts.body if isinstance(c, ast.ClassDef) for n in c.body
                 if isinstance(n, ast.FunctionDef) and n.name == "_forget_cached_answers"]
    checks = [ast.unparse(s) for s in method.body[2:4]]
    assert checks == [f"loaded = {loaded_any}", f"if not (loaded or {files}):\n    return"], checks
