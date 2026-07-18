#!/usr/bin/env python3
"""What the benchmark API routes promise over admission, history and configuration.

The module under contract is a FastAPI router: nineteen HTTP endpoints and one
WebSocket wired around three module-level singletons -- a run-state object that
serialises benchmark execution, a history store imported behind an availability
flag, and an emergency-stop admission guard imported the same way. The suite
drives the real router through a real test client, so the status codes, the
request validation and the response shapes pinned here are the ones the wire
actually carries, not a re-implementation of them.

Three shapes of the module's behaviour are worded by the code more narrowly than
a first reading suggests, and this suite pins the code:

  * The admission guard is consulted only when its module imported. When that
    import fails, run admission is OPEN -- the handler skips the check and
    launches the worker. Elsewhere in the project the same guard surface is
    held fail-closed (an indeterminable stop state refuses a mutating trigger);
    here the code chooses the opposite, and the suite pins the open door as the
    behaviour it is. Closing it would be a behaviour change, tracked outside
    this suite.

  * Config validation only inspects dict-form role assignments. A role mapped
    to a bare string resolves in the roles listing, but the validator walks
    past it without checking the named model against the installed set, so a
    string-form assignment can never raise a warning. The suite pins both
    halves: the listing that understands the string form and the validator
    that is blind to it.

  * The benchmark config loader answers a MISSING file with a four-key default
    and an EMPTY file with a bare empty mapping. Handlers `.get` their way
    around either, but the two degraded shapes are not the same shape, and the
    suite pins each one where it arises.

Every dependency the router reaches for is manufactured, not hoped for. The
inference client is declared unreachable and proven so before the module loads,
in this environment and in any other, so the installed-model list is empty by
construction. The history store is seeded as a deterministic recorder; one
window removes it entirely to prove the import fallback flips the availability
flag and degrades every history endpoint to the same refusal. The worker thread
is replaced with a recorder at the launch seam, so an admitted run is pinned by
the payload handed to the thread rather than by racing a daemon. The scoring
helpers are pure and are pinned directly, boundary by boundary. Loaded through
the shared isolation window; no real backend is ever touched.
"""

import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.api.routes_benchmark"
_PREFIX = "/api/benchmark"


# ---------------------------------------------------------------------------
# Seeds: a recorder store, a controllable admission guard, a thread recorder.


class _Store:
    """Deterministic stand-in for the benchmark history store.

    Every method records its call and answers from a knob the contract set
    beforehand, so a passthrough is pinned by comparing what went in with what
    came out on both faces of the route.
    """

    def __init__(self):
        self.calls = []
        self.runs_result = []
        self.count_result = 0
        self.detail = {}
        self.deleted = set()
        self.compare_result = SimpleNamespace(runs=[], matrix={}, deltas={}, regressions=[])
        self.trends = SimpleNamespace(
            model="", run_ids=[], run_dates=[], avg_scores=[], avg_times=[]
        )
        self.conn = None

    def get_runs(self, run_type="llm", limit=20, offset=0):
        self.calls.append(("get_runs", run_type, limit, offset))
        return self.runs_result

    def get_run_count(self, run_type="llm"):
        self.calls.append(("get_run_count", run_type))
        return self.count_result

    def get_run_detail(self, run_id):
        self.calls.append(("get_run_detail", run_id))
        return self.detail.get(run_id)

    def delete_run(self, run_id):
        self.calls.append(("delete_run", run_id))
        return run_id in self.deleted

    def compare_runs(self, run_ids):
        self.calls.append(("compare_runs", tuple(run_ids)))
        return self.compare_result

    def get_model_trends(self, model, last_n_runs=10):
        self.calls.append(("get_model_trends", model, last_n_runs))
        return self.trends

    def save_run(self, record):
        self.calls.append(("save_run", record))

    def save_result(self, record):
        self.calls.append(("save_result", record))

    def _get_conn(self):
        return self.conn


class _Conn:
    """Wrapper handing the route a connection whose close() is observable.

    The route closes its connection in a finally block; the wrapper records
    the close instead of severing the underlying handle, so the contract can
    still read the table afterwards to pin what was persisted. The underlying
    handle is opened cross-thread-safe because the test client serves the
    handler from a worker thread.
    """

    def __init__(self, real):
        self._real = real
        self.closed = False

    def execute(self, *args):
        return self._real.execute(*args)

    def commit(self):
        return self._real.commit()

    def close(self):
        self.closed = True


class _AttrBag:
    """Record stand-in: keyword fields become attributes, nothing more."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _history_module(store):
    mod = types.ModuleType("opti_oignon.benchmark_history")
    mod.BenchmarkRunRecord = _AttrBag
    mod.BenchmarkResultRecord = _AttrBag
    mod.benchmark_history = store
    return mod


def _stop_module(exc=None):
    """Admission-guard stand-in: records each consultation, raises on demand."""
    mod = types.ModuleType("opti_oignon.emergency_stop")
    calls = []

    def guard_http():
        calls.append("admission")
        if exc is not None:
            raise exc

    mod.guard_http = guard_http
    mod.calls = calls
    return mod


def _thread_stub(record):
    """Stand-in for the threading module at the launch seam.

    The handler under contract only reaches for ``threading.Thread``; the
    stand-in records the constructed thread and makes ``start`` a no-op, so an
    admitted run is pinned by its launch payload instead of by racing a real
    daemon against the assertions.
    """

    class _Thread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=""):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            self.daemon = daemon
            self.name = name
            self.started = False
            record.append(self)

        def start(self):
            self.started = True

    return SimpleNamespace(Thread=_Thread)


def _load(*, store=None, stop=None, block_history=False, block_stop=False):
    """Load the benchmark routes in isolation and mount them on a test app.

    store         -- the history stand-in behind the availability flag; a
                     fresh recorder unless the contract brings its own.
    stop          -- the admission-guard stand-in; a silent recorder unless
                     the contract needs one that raises.
    block_history -- declare the history module UNREACHABLE and prove it, so
                     the module's own import fallback is what runs.
    block_stop    -- declare the admission-guard module UNREACHABLE and prove
                     it, selecting the branch where the guard is None.

    The inference client is ALWAYS declared unreachable and proven so before
    the module executes: the installed-model list must be empty by
    construction, in this environment and in any other, or every contract that
    reasons about it proves nothing.

    Returns ``(module, client, store, stop, restore)``.
    """
    seeded = {}
    blocked = ["ollama"]

    if block_history:
        blocked.append("opti_oignon.benchmark_history")
        store = None
    else:
        if store is None:
            store = _Store()
        seeded["opti_oignon.benchmark_history"] = _history_module(store)

    if block_stop:
        blocked.append("opti_oignon.emergency_stop")
        stop = None
    else:
        if stop is None:
            stop = _stop_module()
        seeded["opti_oignon.emergency_stop"] = stop

    loaded, restore = isolate(
        targets={_TARGET: source("api", "routes_benchmark.py")},
        blocked=tuple(blocked),
        seeded=seeded,
        packages=("opti_oignon.api",),
    )
    module = loaded[_TARGET]
    app = FastAPI()
    app.include_router(module.router)
    client = TestClient(app)
    return module, client, store, stop, restore


def _pin_config(module, *, benchmark=None, models=None):
    """Replace the YAML loaders on the loaded module with fixed mappings."""
    if benchmark is not None:
        module._load_benchmark_config = lambda: benchmark
    if models is not None:
        module._load_models_config = lambda: models


# ---------------------------------------------------------------------------
# Admission and run-state


def test_r1_stop_guard_refuses_before_any_other_answer():
    stop = _stop_module(exc=HTTPException(status_code=503, detail={"stopped": True}))
    rb, client, _, _, restore = _load(stop=stop)
    try:
        _pin_config(rb, benchmark={"tasks": {"t": {}}})
        resp = client.post(f"{_PREFIX}/llm/run", json={"models": ["m"]})
        assert resp.status_code == 503 and resp.json()["detail"] == {"stopped": True}, (
            "an engaged stop refuses the run admission with its own payload"
        )
        rb._state.start("busy")
        resp = client.post(f"{_PREFIX}/llm/run", json={"models": ["m"]})
        assert resp.status_code == 503, (
            "the stop guard is consulted BEFORE the single-run lock: an engaged stop "
            "answers 503 even when the lock would have answered 409"
        )
        assert stop.calls == ["admission", "admission"]
    finally:
        restore()


def test_r2_absent_stop_module_leaves_admission_open():
    threads = []
    rb, client, _, _, restore = _load(block_stop=True)
    try:
        assert rb._emergency_stop is None, (
            "a failed guard import leaves the module-level handle None"
        )
        rb.threading = _thread_stub(threads)
        _pin_config(rb, benchmark={"tasks": {"t1": {"name": "a"}}})
        resp = client.post(f"{_PREFIX}/llm/run", json={"models": ["m1"]})
        assert resp.status_code == 200 and threads and threads[0].started, (
            "with the guard module unreachable the run is ADMITTED: the handler skips "
            "the check it cannot import. Pinned as the behaviour the code chooses; the "
            "fail-closed posture used elsewhere would be a behaviour change"
        )
    finally:
        restore()


def test_r3_single_run_lock_answers_409_while_running():
    rb, client, _, stop, restore = _load()
    try:
        rb._state.start("busy")
        resp = client.post(f"{_PREFIX}/llm/run", json={"models": ["m"]})
        assert resp.status_code == 409 and "already running" in resp.json()["detail"]
        assert stop.calls == ["admission"], (
            "the silent guard was consulted before the lock refused"
        )
    finally:
        restore()


def test_r4_no_models_and_none_installed_is_a_400():
    rb, client, _, _, restore = _load()
    try:
        _pin_config(rb, benchmark={"tasks": {"t": {}}})
        resp = client.post(f"{_PREFIX}/llm/run", json={"models": []})
        assert resp.status_code == 400 and "No models" in resp.json()["detail"], (
            "an empty request over an empty installed set is refused before any task "
            "resolution, even though the pinned config has tasks to offer"
        )
    finally:
        restore()


def test_r5_unknown_suite_is_a_404():
    rb, client, _, _, restore = _load()
    try:
        _pin_config(rb, benchmark={"suites": {}, "tasks": {"t": {}}})
        resp = client.post(
            f"{_PREFIX}/llm/run", json={"models": ["m"], "suite_id": "missing"}
        )
        assert resp.status_code == 404 and "missing" in resp.json()["detail"]
    finally:
        restore()


def test_r6_no_valid_tasks_is_a_400_on_both_branches():
    rb, client, _, _, restore = _load()
    try:
        _pin_config(rb, benchmark={"tasks": {}})
        resp = client.post(f"{_PREFIX}/llm/run", json={"models": ["m"], "tasks": ["ghost"]})
        assert resp.status_code == 400 and "No valid tasks" in resp.json()["detail"], (
            "a requested task list that filters to nothing is refused"
        )
        resp = client.post(f"{_PREFIX}/llm/run", json={"models": ["m"]})
        assert resp.status_code == 400 and "No valid tasks" in resp.json()["detail"], (
            "an empty catalogue leaves the default all-tasks selection empty too"
        )
    finally:
        restore()


def test_r7_admitted_run_launches_the_worker_with_the_resolved_payload():
    threads = []
    cfg = {
        "suites": {"sw": {"tasks": ["t1"]}},
        "tasks": {"t1": {"name": "a"}, "t2": {}, "t3": {}},
        "scoring": {"knob": 1},
    }
    rb, client, _, stop, restore = _load()
    try:
        rb.threading = _thread_stub(threads)
        _pin_config(rb, benchmark=cfg)

        body = {"models": ["m1", "m2"], "temperature": 0.5, "timeout": 60, "max_tokens": 200}
        resp = client.post(f"{_PREFIX}/llm/run", json=body)
        assert resp.status_code == 200
        payload = resp.json()
        run_id = payload["run_id"]
        assert payload["status"] == "running"
        assert payload["models"] == ["m1", "m2"]
        assert payload["tasks"] == ["t1", "t2", "t3"], "no filter selects the whole catalogue"
        assert payload["total_tests"] == 6, (
            "the announced total is models TIMES tasks -- two by three is six, a "
            "product no same-operand sum could imitate"
        )

        assert len(threads) == 1 and threads[0].started
        worker = threads[0]
        assert worker.target is rb._run_benchmark_thread
        assert worker.daemon is True
        assert worker.name == f"benchmark-{run_id[:8]}"
        assert worker.args == (
            run_id, ["m1", "m2"], cfg["tasks"], ["t1", "t2", "t3"], 0.5, 60, 200,
            {"knob": 1},
        ), "the worker receives exactly the resolved run, not a re-derived one"
        assert stop.calls == ["admission"]
        assert rb._state.is_running() and rb._state.current_run_id == run_id

        rb._state.finish("completed")
        resp = client.post(f"{_PREFIX}/llm/run", json={"models": ["m"], "tasks": ["t2", "zz"]})
        assert resp.status_code == 200 and resp.json()["tasks"] == ["t2"], (
            "a finished run releases the lock, and a requested list is filtered "
            "against the catalogue"
        )

        rb._state.finish("completed")
        resp = client.post(
            f"{_PREFIX}/llm/run", json={"models": ["m"], "suite_id": "sw", "tasks": ["t2"]}
        )
        assert resp.status_code == 200 and resp.json()["tasks"] == ["t1"], (
            "a suite id overrides an explicit task list"
        )
    finally:
        restore()


def test_r8_cancel_requires_a_running_run_then_flags_it():
    rb, client, _, _, restore = _load()
    try:
        resp = client.post(f"{_PREFIX}/llm/cancel")
        assert resp.status_code == 409, "cancelling an idle state is refused"
        rb._state.start("rid")
        resp = client.post(f"{_PREFIX}/llm/cancel")
        assert resp.status_code == 200
        assert resp.json() == {"status": "cancel_requested", "run_id": "rid"}
        assert rb._state.is_cancelled()
        resp = client.post(f"{_PREFIX}/llm/cancel")
        assert resp.status_code == 200, (
            "a second cancel while the worker is still winding down is not an error"
        )
    finally:
        restore()


def test_r9_status_tracks_the_state_and_keeps_the_last_run_id():
    rb, client, _, _, restore = _load()
    try:
        assert client.get(f"{_PREFIX}/llm/status").json() == {
            "running": False, "run_id": None, "status": "idle",
        }
        rb._state.start("rid1")
        assert client.get(f"{_PREFIX}/llm/status").json() == {
            "running": True, "run_id": "rid1", "status": "running",
        }
        rb._state.finish("completed")
        assert client.get(f"{_PREFIX}/llm/status").json() == {
            "running": False, "run_id": "rid1", "status": "completed",
        }, "finishing keeps the run id in view; only the next start replaces it"
    finally:
        restore()


def test_r10_run_request_bounds_are_validated_on_the_wire():
    rb, client, _, _, restore = _load()
    try:
        for bad in (
            {"temperature": -0.1},
            {"temperature": 2.1},
            {"timeout": 29},
            {"timeout": 1801},
            {"max_tokens": 99},
            {"max_tokens": 4097},
        ):
            resp = client.post(f"{_PREFIX}/llm/run", json=bad)
            assert resp.status_code == 422, f"{bad} must be refused by validation"

        _pin_config(rb, benchmark={"tasks": {"t": {}}})
        rb.threading = _thread_stub([])
        boundary = {"models": ["m"], "temperature": 0.0, "timeout": 30, "max_tokens": 100}
        assert client.post(f"{_PREFIX}/llm/run", json=boundary).status_code == 200, (
            "the declared bounds are inclusive"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# History endpoints


def test_r11_every_history_endpoint_degrades_to_the_same_503():
    rb, client, _, _, restore = _load(block_history=True)
    try:
        probes = (
            ("GET", f"{_PREFIX}/runs", None),
            ("GET", f"{_PREFIX}/runs/x", None),
            ("DELETE", f"{_PREFIX}/runs/x", None),
            ("GET", f"{_PREFIX}/compare?runs=a,b", None),
            ("GET", f"{_PREFIX}/trends/m", None),
            ("POST", f"{_PREFIX}/llm/user-score",
             {"run_id": "r", "model": "m", "task": "t", "score": 5.0}),
        )
        for method, url, body in probes:
            resp = client.request(method, url, json=body)
            assert resp.status_code == 503, f"{method} {url} must refuse without a store"
            assert "not available" in resp.json()["detail"]
    finally:
        restore()


def test_r12_run_listing_bounds_its_pagination():
    rb, client, _, _, restore = _load()
    try:
        assert client.get(f"{_PREFIX}/runs", params={"limit": 0}).status_code == 422
        assert client.get(f"{_PREFIX}/runs", params={"limit": 101}).status_code == 422
        assert client.get(f"{_PREFIX}/runs", params={"offset": -1}).status_code == 422
        ok = client.get(f"{_PREFIX}/runs", params={"limit": 1, "offset": 0})
        assert ok.status_code == 200
        assert client.get(f"{_PREFIX}/runs", params={"limit": 100}).status_code == 200, (
            "the declared bounds are inclusive"
        )
    finally:
        restore()


def test_r13_run_listing_forwards_the_query_and_wraps_the_answer():
    store = _Store()
    store.runs_result = [{"id": "r1"}]
    store.count_result = 7
    rb, client, _, _, restore = _load(store=store)
    try:
        resp = client.get(
            f"{_PREFIX}/runs", params={"run_type": "perf", "limit": 5, "offset": 3}
        )
        assert resp.status_code == 200
        assert resp.json() == {"runs": [{"id": "r1"}], "total": 7, "limit": 5, "offset": 3}
        assert ("get_runs", "perf", 5, 3) in store.calls, (
            "the query parameters reach the store unchanged"
        )
        assert ("get_run_count", "perf") in store.calls
    finally:
        restore()


def test_r14_run_detail_is_a_passthrough_or_a_404():
    store = _Store()
    store.detail = {"ok": {"id": "ok", "results": [{"score": 1}]}}
    rb, client, _, _, restore = _load(store=store)
    try:
        resp = client.get(f"{_PREFIX}/runs/ok")
        assert resp.status_code == 200 and resp.json() == store.detail["ok"], (
            "what the store hands back is what the wire carries"
        )
        assert client.get(f"{_PREFIX}/runs/nope").status_code == 404
    finally:
        restore()


def test_r15_delete_answers_deleted_or_404():
    store = _Store()
    store.deleted = {"gone"}
    rb, client, _, _, restore = _load(store=store)
    try:
        resp = client.delete(f"{_PREFIX}/runs/gone")
        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted", "run_id": "gone"}
        assert client.delete(f"{_PREFIX}/runs/keep").status_code == 404
        assert ("delete_run", "keep") in store.calls
    finally:
        restore()


def test_r16_compare_needs_two_ids_and_strips_what_it_splits():
    store = _Store()
    store.compare_result = SimpleNamespace(
        runs=[{"id": "a"}], matrix={"m": 1}, deltas={"d": 2}, regressions=["r"]
    )
    rb, client, _, _, restore = _load(store=store)
    try:
        assert client.get(f"{_PREFIX}/compare").status_code == 422, (
            "the id list is a required parameter"
        )
        assert client.get(f"{_PREFIX}/compare", params={"runs": "a"}).status_code == 400
        assert client.get(f"{_PREFIX}/compare", params={"runs": "a, ,,"}).status_code == 400, (
            "blank fragments do not count towards the two-id floor"
        )
        resp = client.get(f"{_PREFIX}/compare", params={"runs": " a , b "})
        assert resp.status_code == 200
        assert resp.json() == {
            "runs": [{"id": "a"}], "matrix": {"m": 1}, "deltas": {"d": 2}, "regressions": ["r"],
        }
        assert ("compare_runs", ("a", "b")) in store.calls, (
            "the ids reach the store stripped of their whitespace"
        )
    finally:
        restore()


def test_r17_trends_bounds_last_n_and_maps_the_five_fields():
    store = _Store()
    store.trends = SimpleNamespace(
        model="mx", run_ids=["a"], run_dates=["d"], avg_scores=[1.0], avg_times=[2.0]
    )
    rb, client, _, _, restore = _load(store=store)
    try:
        assert client.get(f"{_PREFIX}/trends/mx", params={"last_n": 0}).status_code == 422
        assert client.get(f"{_PREFIX}/trends/mx", params={"last_n": 51}).status_code == 422
        resp = client.get(f"{_PREFIX}/trends/mx", params={"last_n": 7})
        assert resp.status_code == 200
        assert resp.json() == {
            "model": "mx", "run_ids": ["a"], "run_dates": ["d"],
            "avg_scores": [1.0], "avg_times": [2.0],
        }
        assert ("get_model_trends", "mx", 7) in store.calls
        client.get(f"{_PREFIX}/trends/mx")
        assert ("get_model_trends", "mx", 10) in store.calls, "the default window is ten runs"
    finally:
        restore()


# ---------------------------------------------------------------------------
# User scoring


def test_r18_user_score_refuses_in_order_store_run_result():
    rb, client, _, _, restore = _load(block_history=True)
    try:
        body = {"run_id": "r", "model": "m", "task": "t", "score": 5.0}
        assert client.post(f"{_PREFIX}/llm/user-score", json=body).status_code == 503
    finally:
        restore()

    rb, client, store, _, restore = _load()
    try:
        body = {"run_id": "r", "model": "m", "task": "t", "score": 5.0}
        resp = client.post(f"{_PREFIX}/llm/user-score", json=body)
        assert resp.status_code == 404 and "Run" in resp.json()["detail"], (
            "an unknown run is refused before any connection is opened"
        )
    finally:
        restore()

    store = _Store()
    store.detail = {"r": {"id": "r"}}
    real = sqlite3.connect(":memory:", check_same_thread=False)
    real.row_factory = sqlite3.Row
    real.execute(
        "CREATE TABLE benchmark_results ("
        "id TEXT PRIMARY KEY, run_id TEXT, model TEXT, task TEXT, "
        "auto_score REAL, user_score REAL, score REAL)"
    )
    real.commit()
    store.conn = _Conn(real)
    rb, client, _, _, restore = _load(store=store)
    try:
        _pin_config(rb, benchmark={})
        body = {"run_id": "r", "model": "m", "task": "t", "score": 5.0}
        resp = client.post(f"{_PREFIX}/llm/user-score", json=body)
        assert resp.status_code == 404 and "Result" in resp.json()["detail"], (
            "a known run without the addressed result is refused after the lookup"
        )
        assert store.conn.closed, "the connection is closed even on the refusal path"
    finally:
        restore()


def test_r19_user_score_persists_the_weighted_final_score():
    def _rig():
        store = _Store()
        store.detail = {"run1": {"id": "run1"}}
        real = sqlite3.connect(":memory:", check_same_thread=False)
        real.row_factory = sqlite3.Row
        real.execute(
            "CREATE TABLE benchmark_results ("
            "id TEXT PRIMARY KEY, run_id TEXT, model TEXT, task TEXT, "
            "auto_score REAL, user_score REAL, score REAL)"
        )
        real.execute(
            "INSERT INTO benchmark_results VALUES ('res1', 'run1', 'm1', 't1', 6.0, NULL, 6.0)"
        )
        real.commit()
        store.conn = _Conn(real)
        return store, real

    store, real = _rig()
    rb, client, _, _, restore = _load(store=store)
    try:
        _pin_config(rb, benchmark={"scoring": {"user_weight": 0.25, "auto_weight": 0.75}})
        body = {"run_id": "run1", "model": "m1", "task": "t1", "score": 8.0}
        resp = client.post(f"{_PREFIX}/llm/user-score", json=body)
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "final_score": 6.5}, (
            "the final score is the CONFIGURED blend of the user and auto scores, "
            "rounded to two places -- not the built-in weights"
        )
        row = real.execute(
            "SELECT user_score, score FROM benchmark_results WHERE id = 'res1'"
        ).fetchone()
        assert (row["user_score"], row["score"]) == (8.0, 6.5), (
            "both the raw user score and the blended final score are persisted"
        )
        assert store.conn.closed, "the connection is closed after the write"
    finally:
        restore()

    store, real = _rig()
    rb, client, _, _, restore = _load(store=store)
    try:
        _pin_config(rb, benchmark={})
        body = {"run_id": "run1", "model": "m1", "task": "t1", "score": 8.0}
        resp = client.post(f"{_PREFIX}/llm/user-score", json=body)
        assert resp.json()["final_score"] == 7.2, (
            "without a scoring config the blend falls back to six-tenths user, "
            "four-tenths auto"
        )
    finally:
        restore()


def test_r20_user_score_bounds_are_validated_and_inclusive():
    rb, client, _, _, restore = _load()
    try:
        base = {"run_id": "r", "model": "m", "task": "t"}
        assert client.post(
            f"{_PREFIX}/llm/user-score", json={**base, "score": -0.1}
        ).status_code == 422
        assert client.post(
            f"{_PREFIX}/llm/user-score", json={**base, "score": 10.1}
        ).status_code == 422
        assert client.post(
            f"{_PREFIX}/llm/user-score", json={"run_id": "r", "model": "m", "score": 5.0}
        ).status_code == 422, "every addressing field is required"
        for boundary in (0.0, 10.0):
            resp = client.post(f"{_PREFIX}/llm/user-score", json={**base, "score": boundary})
            assert resp.status_code == 404, (
                "the boundary scores clear validation and fail later, on the lookup"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Model configuration endpoints


def test_r21_config_update_refuses_empty_then_maps_the_save_verdict():
    rb, client, _, _, restore = _load()
    try:
        resp = client.put(f"{_PREFIX}/models/config", json={"config": {}})
        assert resp.status_code == 400 and "Empty config" in resp.json()["detail"]

        rb._save_models_config = lambda cfg: False
        resp = client.put(f"{_PREFIX}/models/config", json={"config": {"k": 1}})
        assert resp.status_code == 500, "a failed save is the caller's problem to see"

        saved = []
        rb._save_models_config = lambda cfg: saved.append(cfg) or True
        resp = client.put(f"{_PREFIX}/models/config", json={"config": {"k": 1}})
        assert resp.status_code == 200 and resp.json() == {"status": "saved"}
        assert saved == [{"k": 1}], "the body's config reaches the writer unchanged"
    finally:
        restore()


def test_r22_role_update_builds_the_assignment_and_preserves_the_rest():
    rb, client, _, _, restore = _load()
    try:
        _pin_config(rb, models={"routing": {"other": {"primary": "keep"}}})
        saved = []
        rb._save_models_config = lambda cfg: saved.append(cfg) or True

        resp = client.put(f"{_PREFIX}/models/config/roles/chat", json={})
        assert resp.status_code == 400 and not saved, (
            "an all-empty assignment is refused before anything is written"
        )

        resp = client.put(
            f"{_PREFIX}/models/config/roles/chat", json={"primary": "p1", "quality": "q1"}
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "updated", "role": "chat",
            "assignment": {"primary": "p1", "quality": "q1"},
        }, "only the fields the caller set make it into the assignment"
        assert saved[0]["routing"]["chat"] == {"primary": "p1", "quality": "q1"}
        assert saved[0]["routing"]["other"] == {"primary": "keep"}, (
            "updating one role leaves the other roles as they were"
        )

        rb._save_models_config = lambda cfg: False
        resp = client.put(f"{_PREFIX}/models/config/roles/chat", json={"fast": "f1"})
        assert resp.status_code == 500
    finally:
        restore()


def test_r23_validate_flags_unknown_models_but_is_blind_to_string_roles():
    rb, client, _, _, restore = _load()
    try:
        body = {"config": {
            "routing": {"chat": {"primary": "ghost", "fast": ""}, "solo": "phantom-model"},
            "fallback_order": ["phantom"],
        }}
        resp = client.post(f"{_PREFIX}/models/config/validate", json=body)
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["warnings"] == [
            {"role": "chat", "priority": "primary", "model": "ghost", "issue": "not_installed"},
            {"role": "fallback_order", "priority": "", "model": "phantom",
             "issue": "not_installed"},
        ], (
            "a dict-form assignment and the fallback list are checked; an empty slot is "
            "skipped; a STRING-form assignment is walked past unchecked, so a role mapped "
            "to a bare model name can never raise a warning here even though the roles "
            "listing understands that form"
        )
        assert payload["valid"] is False and payload["installed_count"] == 0

        rb._get_installed_models = lambda: ["good"]
        body = {"config": {"routing": {"r": {"primary": "good", "fast": "bad"}}}}
        payload = client.post(f"{_PREFIX}/models/config/validate", json=body).json()
        assert [w["model"] for w in payload["warnings"]] == ["bad"]
        assert payload["valid"] is False and payload["installed_count"] == 1

        body = {"config": {"routing": {"r": {"primary": "good"}}, "fallback_order": ["good"]}}
        payload = client.post(f"{_PREFIX}/models/config/validate", json=body).json()
        assert payload == {"valid": True, "warnings": [], "installed_count": 1}

        _pin_config(rb, models={"routing": {"z": {"primary": "nope"}}})
        rb._get_installed_models = lambda: []
        payload = client.post(f"{_PREFIX}/models/config/validate", json={"config": {}}).json()
        assert [w["model"] for w in payload["warnings"]] == ["nope"], (
            "an empty body falls back to the config on disk"
        )
    finally:
        restore()


def test_r24_roles_listing_reads_dict_and_string_forms_and_drops_the_rest():
    rb, client, _, _, restore = _load()
    try:
        _pin_config(rb, models={"routing": {
            "chat": {"primary": "a", "fast": "b", "quality": "c"},
            "code": "solo",
            "weird": 42,
        }})
        resp = client.get(f"{_PREFIX}/models/config/roles")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["roles"] == [
            {"role": "chat", "primary": "a", "fast": "b", "quality": "c"},
            {"role": "code", "primary": "solo", "fast": "", "quality": ""},
        ], (
            "a dict assignment maps its three slots, a string assignment becomes the "
            "primary, and any other shape is dropped from the listing"
        )
        assert payload["installed_models"] == []
    finally:
        restore()


def test_r25_installed_models_reports_the_list_and_its_count():
    rb, client, _, _, restore = _load()
    try:
        assert client.get(f"{_PREFIX}/models/installed").json() == {"models": [], "count": 0}
        rb._get_installed_models = lambda: ["a", "b"]
        assert client.get(f"{_PREFIX}/models/installed").json() == {
            "models": ["a", "b"], "count": 2,
        }
    finally:
        restore()


# ---------------------------------------------------------------------------
# Suite and task catalogue


def test_r26_suite_listing_counts_dangling_ids_but_categorises_known_ones():
    rb, client, _, _, restore = _load()
    try:
        _pin_config(rb, benchmark={
            "suites": {
                "s1": {"name": "N", "description": "D", "tasks": ["t1", "t2", "zz"]},
                "s2": {"tasks": []},
            },
            "tasks": {"t1": {"category": "code"}, "t2": {}},
        })
        resp = client.get(f"{_PREFIX}/suites")
        assert resp.status_code == 200
        suites = {s["id"]: s for s in resp.json()["suites"]}
        one = suites["s1"]
        assert (one["name"], one["description"]) == ("N", "D")
        assert one["task_count"] == 3 and one["tasks"] == ["t1", "t2", "zz"], (
            "the count includes a dangling id the catalogue does not know"
        )
        assert sorted(one["categories"]) == ["code", "general"], (
            "categories come only from ids the catalogue KNOWS, with the default filled "
            "in -- the dangling id contributes nothing"
        )
        two = suites["s2"]
        assert (two["name"], two["task_count"], two["categories"]) == ("s2", 0, []), (
            "a nameless suite answers to its id"
        )
    finally:
        restore()


def test_r27_suite_detail_is_404_or_expands_every_listed_id():
    rb, client, _, _, restore = _load()
    try:
        _pin_config(rb, benchmark={
            "suites": {"s1": {"name": "N", "tasks": ["t1", "zz"]}},
            "tasks": {"t1": {
                "name": "T1", "description": "dd", "category": "code", "prompt": "pp",
                "expected_keywords": ["k"], "max_expected_time": 9, "scoring_method": "llm",
            }},
        })
        resp = client.get(f"{_PREFIX}/suites/nope")
        assert resp.status_code == 404 and "nope" in resp.json()["detail"]

        resp = client.get(f"{_PREFIX}/suites/s1")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["id"] == "s1" and payload["name"] == "N"
        known, dangling = payload["tasks"]
        assert known == {
            "id": "t1", "name": "T1", "description": "dd", "category": "code",
            "prompt": "pp", "expected_keywords": ["k"], "max_expected_time": 9,
            "scoring_method": "llm",
        }
        assert dangling == {
            "id": "zz", "name": "zz", "description": "", "category": "general",
            "prompt": "", "expected_keywords": [], "max_expected_time": 300,
            "scoring_method": "keywords",
        }, "a dangling id still yields a row, filled entirely with the defaults"
    finally:
        restore()


def test_r28_task_listing_maps_the_catalogue_without_the_prompts():
    rb, client, _, _, restore = _load()
    try:
        _pin_config(rb, benchmark={"tasks": {
            "t1": {"name": "T1", "description": "dd", "category": "code",
                   "prompt": "secret", "max_expected_time": 9, "scoring_method": "llm"},
            "t2": {},
        }})
        resp = client.get(f"{_PREFIX}/tasks")
        assert resp.status_code == 200
        tasks = {t["id"]: t for t in resp.json()["tasks"]}
        assert tasks["t1"] == {
            "id": "t1", "name": "T1", "description": "dd", "category": "code",
            "max_expected_time": 9, "scoring_method": "llm",
        }, "the listing carries the metadata and leaves the prompt out"
        assert tasks["t2"] == {
            "id": "t2", "name": "t2", "description": "", "category": "general",
            "max_expected_time": 300, "scoring_method": "keywords",
        }
    finally:
        restore()


# ---------------------------------------------------------------------------
# Pure scoring helpers


def test_r29_calculate_score_boundary_by_boundary():
    rb, _, _, _, restore = _load()
    try:
        score = rb._calculate_score
        assert score("x" * 201, []) == (7, [], [])
        assert score("x" * 200, []) == (5, [], []), "the length tiers are strict"
        assert score("x" * 51, []) == (5, [], [])
        assert score("x" * 50, []) == (3, [], [])
        assert score("", []) == (3, [], [])

        got = score("alpha beta " + "x" * 140, ["alpha", "beta", "gamma"])
        assert got == (6, ["alpha", "beta"], ["gamma"]), (
            "two of three keywords truncate to six, and both lists name their members"
        )

        four_of_five = "a1 b2 c3 d4 " + "x" * 300
        assert score(four_of_five, ["a1", "b2", "c3", "d4", "e5"])[0] == 9, (
            "a four-fifths hit over a long answer earns the completeness bonus"
        )
        assert score("a1 " + "x" * 300, ["a1"])[0] == 10, (
            "the bonus never pushes past the ceiling"
        )
        assert score("a1", ["a1"])[0] == 8, "a very short answer pays the two-point toll"
        assert score("zz", ["a1"])[0] == 0, "the toll never pushes below the floor"

        got = score("some foo here", ["Foo"])
        assert got == (8, ["Foo"], []), (
            "matching is case-insensitive and the found list keeps the keyword's casing"
        )
    finally:
        restore()


def test_r30_refusal_detection_is_pattern_based_and_case_insensitive():
    rb, _, _, _, restore = _load()
    try:
        for text in (
            "I'm sorry, that is out of scope",
            "well i CANNOT help with that",
            "As an AI model, hard pass",
            "I am not able to comply",
            "my expertise is elsewhere",
            "I don't have that information",
        ):
            assert rb._is_refusal(text), text
        assert not rb._is_refusal("Paris is the capital of France.")
        assert not rb._is_refusal("")
    finally:
        restore()


# ---------------------------------------------------------------------------
# Degradation without the optional imports


def test_r31_missing_history_module_flips_the_flag_and_nulls_the_handle():
    rb, _, _, _, restore = _load(block_history=True)
    try:
        assert rb.HISTORY_AVAILABLE is False
        assert rb.benchmark_history is None
    finally:
        restore()


def test_r32_missing_inference_client_yields_an_empty_installed_list():
    rb, _, _, _, restore = _load()
    try:
        assert rb.OLLAMA_AVAILABLE is False, (
            "the client's absence is manufactured by the window, not assumed"
        )
        assert rb._get_installed_models() == []
    finally:
        restore()


def test_r33_single_test_without_a_client_degrades_to_an_error_result():
    rb, _, _, _, restore = _load()
    try:
        task = {"prompt": "p", "expected_keywords": ["k1"], "max_expected_time": 5,
                "category": "code", "name": "T"}
        got = rb._execute_single_test(
            model="m", task_id="t", task=task,
            temperature=0.1, timeout=30, max_tokens=100, scoring_config={},
        )
        assert got["status"] == "error", (
            "a missing client is an error, not a timeout: the message names the client "
            "and the elapsed time is nowhere near the budget"
        )
        assert "Ollama not available" in got["error_message"]
        assert (got["score"], got["auto_score"]) == (0.0, 0.0)
        assert got["response_preview"] == ""
        assert got["keywords_found"] == [] and got["keywords_missing"] == ["k1"]
        assert (got["model"], got["task"], got["task_name"], got["category"]) == (
            "m", "t", "T", "code",
        )
        assert got["time_seconds"] >= 0
    finally:
        restore()


def test_r34_config_loaders_degrade_without_raising_and_round_trip():
    rb, _, _, _, restore = _load()
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            assert rb.YAML_AVAILABLE is True

            rb.BENCHMARK_CONFIG_PATH = tmp / "absent.yaml"
            assert rb._load_benchmark_config() == {
                "suites": {}, "tasks": {}, "runner": {}, "scoring": {},
            }, "a missing file answers with the four-key default"

            empty = tmp / "empty.yaml"
            empty.write_text("", encoding="utf-8")
            rb.BENCHMARK_CONFIG_PATH = empty
            assert rb._load_benchmark_config() == {}, (
                "an EMPTY file answers with a bare empty mapping -- a different shape "
                "from the missing-file default, pinned as the two degradations they are"
            )

            bench = tmp / "bench.yaml"
            bench.write_text(
                yaml.safe_dump({"suites": {"a": {"tasks": []}}, "tasks": {}}),
                encoding="utf-8",
            )
            rb.BENCHMARK_CONFIG_PATH = bench
            assert rb._load_benchmark_config() == {"suites": {"a": {"tasks": []}}, "tasks": {}}

            rb.MODELS_CONFIG_PATH = tmp / "absent_models.yaml"
            assert rb._load_models_config() == {}

            out = tmp / "models.yaml"
            rb.MODELS_CONFIG_PATH = out
            assert rb._save_models_config({"k": [1, 2]}) is True
            assert yaml.safe_load(out.read_text(encoding="utf-8")) == {"k": [1, 2]}
            assert rb._load_models_config() == {"k": [1, 2]}, "the writer round-trips"

            rb.MODELS_CONFIG_PATH = tmp / "no_such_dir" / "m.yaml"
            assert rb._save_models_config({"k": 1}) is False, (
                "an unwritable destination is a False, not a raise"
            )
    finally:
        restore()
