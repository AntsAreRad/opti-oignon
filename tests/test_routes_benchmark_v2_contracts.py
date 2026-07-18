#!/usr/bin/env python3
"""What the second benchmark API surface promises over admission, aggregation and management.

The module under contract is a FastAPI router: twenty-six endpoints wired around six
singletons that a dependency module binds behind availability flags -- an evaluator, a
run engine, a judge score store, a recommender, an auto-trigger manager and a custom
profile store. The router holds no state of its own; every answer is a guarded
delegation plus a mapping through response models. The suite drives the real router
through a real test client, and the schema module is loaded REAL from its own source
file, so the response validation exercised on the wire is the production one -- a
permissive stand-in would wave through shapes the true classes refuse, and one
contract proves the refusal.

Several shapes of the module's behaviour are worded by the code more narrowly than a
first reading suggests, and this suite pins the code:

  * Degradation splits into two families. Five read endpoints answer 200 with an
    all-default body when their singleton is down; every acting endpoint refuses with
    503. The sibling benchmark surface degrades uniformly; this one chooses the split,
    and the suite pins both halves.

  * Run admission consults no stop guard and no operating mode: its whole order is
    availability, three request checks, a busy gate, then profile existence -- and the
    existence check stands only while the evaluator does. With the evaluator down, an
    unknown profile is admitted straight to the engine. Category checks on custom
    profiles lean on the same crutch.

  * The busy gate is an attribute read with an open default: an engine that lacks the
    attribute is treated as idle. The default only swallows AttributeError, though.
    The project binds these singletons through deferred import proxies whose flag
    reports that the module FILE exists, so a module that exists but cannot import
    reaches the wire as a 500 from the first attribute touch, not as the 503 of a
    lowered flag. One contract manufactures exactly that proxy.

  * Failure envelopes are asymmetric: an engine that raises at launch is wrapped as a
    500 with the reason, while a recommender that raises on apply is a 200 that says
    applied false. Disabling the auto trigger hardcodes running false in the answer
    while enabling re-reads the live attribute. Each is pinned where it stands.

Every dependency the router reaches for is manufactured, not hoped for: the dependency
module is seeded with recorders whose knobs each contract sets, and the inference
client is declared unreachable and proven so before anything executes. Loaded through
the shared isolation window; no real backend is ever touched.
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.api.routes_benchmark_v2"
_SCHEMAS = "opti_oignon.api.schemas"
_PREFIX = "/api/benchmark/v2"


# ---------------------------------------------------------------------------
# Seeds: a dependency module whose flags follow its singletons, plus recorders.


_FLAG_FOR = {
    "benchmark_evaluator": "BENCHMARK_V2_AVAILABLE",
    "benchmark_runner": "BENCHMARK_RUNNER_AVAILABLE",
    "judge_store": "BENCHMARK_JUDGE_AVAILABLE",
    "benchmark_recommender": "BENCHMARK_RECOMMENDATIONS_AVAILABLE",
    "auto_trigger": "AUTO_TRIGGER_AVAILABLE",
    "custom_profile_store": "CUSTOM_PROFILES_AVAILABLE",
}


def _deps_module(**singletons):
    """Stand-in for the dependency module: six flags, six singleton handles.

    Every name the router imports is present. A singleton passed here raises its
    own availability flag; everything else stays down with a None handle, which
    is the dependency module's own degraded shape.
    """
    module = ModuleType("opti_oignon.api.deps")
    for name, flag in _FLAG_FOR.items():
        setattr(module, flag, False)
        setattr(module, name, None)
    for name, value in singletons.items():
        setattr(module, name, value)
        setattr(module, _FLAG_FOR[name], True)
    return module


class _Recorder:
    """Deterministic stand-in whose attributes and method answers are both knobs.

    Names given in ``attrs`` become plain instance attributes and are read as
    such -- an attribute read is not a call, and the busy gate in production
    reads an attribute. Every other public name resolves to a recording method
    that answers from ``ret``; a knob holding an exception instance raises it
    instead. A passthrough is pinned by comparing the call log with the wire.
    """

    def __init__(self, attrs=None, ret=None):
        self.calls = []
        self._ret = dict(ret or {})
        for name, value in (attrs or {}).items():
            setattr(self, name, value)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def _method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            value = self._ret.get(name)
            if isinstance(value, Exception):
                raise value
            return value

        return _method


class _BareRunner:
    """Engine stand-in that genuinely LACKS the busy attribute.

    The recorder above answers every name, so it cannot express absence; the
    open default of the busy gate needs an object on which the attribute read
    truly fails.
    """

    def __init__(self):
        self.calls = []

    def start_run(self, **kwargs):
        self.calls.append(("start_run", kwargs))
        return "run-bare"


class _RaisingProxy:
    """Stand-in for a deferred import proxy whose resolution failed.

    The dependency module raises its flags on module-file EXISTENCE and binds
    the singleton through a proxy that re-raises the import failure on first
    attribute access. ``getattr`` with a default swallows only AttributeError,
    so this error reaches the wire.
    """

    def __getattr__(self, name):
        raise ImportError(f"deferred import failed resolving {name!r}")


class _ToDict:
    """Record stand-in exposing exactly one behaviour: ``to_dict``."""

    def __init__(self, **fields):
        self._fields = dict(fields)

    def to_dict(self):
        return dict(self._fields)


def _progress(**over):
    """A live progress record shaped the way the engine hands one over."""
    base = {
        "run_id": "r1",
        "status": SimpleNamespace(value="running"),
        "total_questions": 0,
        "completed_questions": 0,
        "current_model": "",
        "current_question": "",
        "elapsed_ms": 0.0,
        "error": "",
    }
    base.update(over)
    return SimpleNamespace(**base)


def _load(**singletons):
    """Load the router in isolation and mount it on a test app.

    The schema module is loaded REAL from its source file, ahead of the router,
    so the response models the router names are the true Pydantic classes and
    the validation on the wire is the production one. The dependency module is
    seeded; each contract raises exactly the flags it needs. The inference
    client is ALWAYS declared unreachable and proven so before the module
    executes.

    Returns ``(module, client, restore)``. The client keeps server errors as
    500 answers instead of re-raising them, because two contracts pin a 500.
    """
    loaded, restore = isolate(
        targets={
            _SCHEMAS: source("api", "schemas.py"),
            _TARGET: source("api", "routes_benchmark_v2.py"),
        },
        blocked=("ollama",),
        seeded={"opti_oignon.api.deps": _deps_module(**singletons)},
        packages=("opti_oignon.api",),
    )
    module = loaded[_TARGET]
    app = FastAPI()
    app.include_router(module.router)
    client = TestClient(app, raise_server_exceptions=False)
    return module, client, restore


# ---------------------------------------------------------------------------
# Profiles and the two degraded families


def test_v1_profiles_map_the_evaluator_through_the_true_schema():
    evaluator = _Recorder(
        attrs={
            "available_profiles": [{
                "id": "p1", "name": "Coding", "description": "d",
                "categories": ["code"], "weight_preset": "speed",
                "custom": True, "surplus": "dropped",
            }],
            "available_categories": ["code", "chat"],
        },
        ret={"question_count": 7},
    )
    _, client, restore = _load(benchmark_evaluator=evaluator)
    try:
        resp = client.get(f"{_PREFIX}/profiles")
        body = resp.json()
        assert resp.status_code == 200 and body["total_questions"] == 7
        assert body["available_categories"] == ["code", "chat"]
        assert body["profiles"] == [{
            "id": "p1", "name": "Coding", "description": "d",
            "categories": ["code"], "weight_preset": "speed", "custom": True,
        }], "the true schema keeps its own fields and drops the surplus key"
        assert evaluator.calls == [("question_count", (), {})]
    finally:
        restore()

    evaluator = _Recorder(
        attrs={"available_profiles": [{"id": 123}], "available_categories": []},
        ret={"question_count": 0},
    )
    _, client, restore = _load(benchmark_evaluator=evaluator)
    try:
        resp = client.get(f"{_PREFIX}/profiles")
        assert resp.status_code == 500, (
            "the schema classes are the real ones: an integer offered to a string "
            "field is refused on the wire, where a permissive stand-in would have "
            "waved it through"
        )
    finally:
        restore()


def test_v2_missing_singletons_leave_the_reading_family_empty_not_refusing():
    _, client, restore = _load()
    try:
        resp = client.get(f"{_PREFIX}/profiles")
        assert resp.status_code == 200 and resp.json() == {
            "profiles": [], "available_categories": [], "total_questions": 0,
        }, "the profile listing degrades to an all-default body, not to a refusal"
        resp = client.get(f"{_PREFIX}/profiles/custom")
        assert resp.status_code == 200 and resp.json() == {"profiles": [], "count": 0}
        resp = client.get(f"{_PREFIX}/auto-trigger/status")
        body = resp.json()
        assert resp.status_code == 200 and body["enabled"] is False
        assert body["poll_interval_seconds"] == 120.0
        resp = client.get(f"{_PREFIX}/auto-trigger/config")
        assert resp.status_code == 200 and resp.json()["trigger_models"] == "all_new"
        resp = client.get(f"{_PREFIX}/auto-trigger/events")
        assert resp.status_code == 200 and resp.json() == {"events": [], "count": 0}
    finally:
        restore()


def test_v3_missing_singletons_refuse_the_acting_family_with_503():
    engine = "Benchmark runner not available"
    reco = "Benchmark recommendations not available"
    custom = "Custom profiles not available"
    trigger = "Auto-trigger not available"
    rows = [
        ("post", "/run", {"json": {"models": ["m"], "profile": "p"}}, engine),
        ("get", "/status/r1", {}, engine),
        ("post", "/cancel/r1", {}, engine),
        ("get", "/results/r1", {}, engine),
        ("get", "/compare", {}, engine),
        ("get", "/history", {}, engine),
        ("get", "/leaderboard", {}, engine),
        ("get", "/head-to-head", {"params": {"model_a": "a", "model_b": "b"}}, engine),
        ("get", "/trends", {"params": {"model": "m"}}, engine),
        ("get", "/export/r1", {}, engine),
        ("get", "/recommendations", {}, reco),
        ("post", "/recommendations/apply", {}, reco),
        ("post", "/profiles/custom", {"json": {"name": "n"}}, custom),
        ("put", "/profiles/custom/p1", {"json": {"name": "n"}}, custom),
        ("delete", "/profiles/custom/p1", {}, custom),
        ("post", "/profiles/preview", {"json": ["a"]}, custom),
        ("put", "/auto-trigger/config", {"json": {"enabled": True}}, trigger),
        ("post", "/auto-trigger/enable", {}, trigger),
        ("post", "/auto-trigger/disable", {}, trigger),
        ("post", "/auto-trigger/test-poll", {}, trigger),
        ("post", "/auto-trigger/reset", {}, trigger),
    ]
    _, client, restore = _load()
    try:
        for method, path, kwargs, detail in rows:
            resp = getattr(client, method)(f"{_PREFIX}{path}", **kwargs)
            assert resp.status_code == 503 and resp.json()["detail"] == detail, (
                f"{method.upper()} {path} belongs to the refusing family and names "
                f"its own missing singleton"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Run admission


def test_v4_run_admission_refuses_in_a_fixed_order():
    _, client, restore = _load()
    try:
        resp = client.post(f"{_PREFIX}/run", json={"models": []})
        assert resp.status_code == 503, (
            "availability is answered before any request validation"
        )
    finally:
        restore()

    runner = _Recorder(attrs={"is_busy": True}, ret={"start_run": "never"})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.post(f"{_PREFIX}/run", json={"models": []})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "At least one model is required", (
            "an empty model list is refused before the busy gate is consulted"
        )
        resp = client.post(f"{_PREFIX}/run", json={"models": ["a"], "profile": ""})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Profile name is required"
        resp = client.post(f"{_PREFIX}/run", json={"models": ["a"], "use_judge": True})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "judge_model is required when use_judge is true"
        resp = client.post(
            f"{_PREFIX}/run",
            json={"models": ["a"], "custom_weights": {"accuracy": "heavy"}},
        )
        assert resp.status_code == 422, (
            "the request model is the real one: a non-numeric weight is refused at "
            "the edge"
        )
        assert runner.calls == [], "no refused admission reaches the engine"
    finally:
        restore()

    runner = _Recorder(attrs={"is_busy": False}, ret={"start_run": "run-1"})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.post(f"{_PREFIX}/run", json={"models": ["a"]})
        assert resp.status_code == 200 and resp.json()["profile"] == "all_round", (
            "an OMITTED profile falls back to the request model's default and passes "
            "the emptiness check; only an explicit empty string is refused"
        )
        assert runner.calls[0][2]["profile"] == "all_round"
    finally:
        restore()


def test_v5_the_busy_gate_is_an_attribute_read_with_an_open_default():
    runner = _Recorder(attrs={"is_busy": True}, ret={"start_run": "never"})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.post(f"{_PREFIX}/run", json={"models": ["a"], "profile": "p"})
        assert resp.status_code == 409
        assert resp.json()["detail"] == (
            "A benchmark run is already in progress. Cancel it or wait for completion."
        )
        assert runner.calls == [], "a busy engine is never asked to start"
    finally:
        restore()

    bare = _BareRunner()
    _, client, restore = _load(benchmark_runner=bare)
    try:
        resp = client.post(f"{_PREFIX}/run", json={"models": ["a"], "profile": "p"})
        assert resp.status_code == 200 and resp.json()["run_id"] == "run-bare", (
            "an engine that LACKS the busy attribute is treated as idle: the gate "
            "reads the attribute with an open default instead of requiring it"
        )
        assert bare.calls and bare.calls[0][0] == "start_run"
    finally:
        restore()


def test_v6_profile_existence_holds_only_while_the_evaluator_stands():
    evaluator = _Recorder(ret={"get_profile_config": None})
    runner = _Recorder(attrs={"is_busy": False}, ret={"start_run": "run-ok"})
    _, client, restore = _load(benchmark_runner=runner, benchmark_evaluator=evaluator)
    try:
        resp = client.post(f"{_PREFIX}/run", json={"models": ["a"], "profile": "ghost"})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Profile 'ghost' not found"
        assert evaluator.calls == [("get_profile_config", ("ghost",), {})]
        assert runner.calls == []
    finally:
        restore()

    evaluator = _Recorder(ret={"get_profile_config": {"categories": ["c"]}})
    runner = _Recorder(attrs={"is_busy": False}, ret={"start_run": "run-ok"})
    _, client, restore = _load(benchmark_runner=runner, benchmark_evaluator=evaluator)
    try:
        resp = client.post(f"{_PREFIX}/run", json={"models": ["a"], "profile": "known"})
        assert resp.status_code == 200
        assert evaluator.calls == [("get_profile_config", ("known",), {})]
    finally:
        restore()

    runner = _Recorder(attrs={"is_busy": False}, ret={"start_run": "run-ok"})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.post(f"{_PREFIX}/run", json={"models": ["a"], "profile": "ghost"})
        assert resp.status_code == 200 and resp.json()["run_id"] == "run-ok", (
            "with the evaluator down the existence check is SKIPPED: an unknown "
            "profile is admitted straight to the engine"
        )
    finally:
        restore()


def test_v7_an_admitted_run_forwards_the_request_and_hardcodes_running():
    runner = _Recorder(attrs={"is_busy": False}, ret={"start_run": "run-xyz"})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.post(f"{_PREFIX}/run", json={
            "models": ["a", "b"], "profile": "quick",
            "use_judge": True, "judge_model": "j",
            "custom_weights": {"accuracy": 1.0},
        })
        assert resp.status_code == 200
        assert resp.json() == {
            "run_id": "run-xyz", "profile": "quick",
            "models": ["a", "b"], "status": "running",
        }, "the answer says running whatever state the engine will report first"
        assert runner.calls == [("start_run", (), {
            "profile": "quick", "models": ["a", "b"],
            "use_judge": True, "judge_model": "j",
            "custom_weights": {"accuracy": 1.0},
        })], "the engine receives exactly the five request fields, as keywords"
    finally:
        restore()


def test_v8_an_engine_that_raises_at_launch_is_wrapped_as_a_500():
    runner = _Recorder(
        attrs={"is_busy": False},
        ret={"start_run": RuntimeError("gpu melted")},
    )
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.post(f"{_PREFIX}/run", json={"models": ["a"], "profile": "p"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Failed to start benchmark run: gpu melted"
    finally:
        restore()


def test_v9_a_raised_flag_with_a_dead_proxy_surfaces_as_a_500():
    _, client, restore = _load(benchmark_runner=_RaisingProxy())
    try:
        resp = client.post(f"{_PREFIX}/run", json={"models": ["a"], "profile": "p"})
        assert resp.status_code == 500, (
            "the availability flag reports that the module FILE exists; when the "
            "deferred proxy re-raises its import failure on the first attribute "
            "touch, the open default of the busy gate cannot swallow an ImportError "
            "and the wire carries a 500 -- not the 503 of a lowered flag"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Status and cancellation


def test_v10_status_maps_a_live_progress_field_by_field():
    prog = _progress(
        run_id="r1", status=SimpleNamespace(value="running"),
        total_questions=10, completed_questions=3,
        current_model="m", current_question="q7", elapsed_ms=12.5, error="",
    )
    runner = _Recorder(ret={"get_progress": prog})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/status/r1")
        assert resp.status_code == 200
        assert resp.json() == {
            "run_id": "r1", "status": "running",
            "total_questions": 10, "completed_questions": 3,
            "current_model": "m", "current_question": "q7",
            "elapsed_ms": 12.5, "error": "",
        }
        assert runner.calls == [("get_progress", ("r1",), {})], (
            "a live progress answers by itself; the results store is not consulted"
        )
    finally:
        restore()


def test_v11_status_of_a_finished_run_is_synthesised_from_the_store():
    runner = _Recorder(ret={
        "get_progress": None,
        "get_results": {"run_id": "other", "status": "failed", "error": "boom"},
    })
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/status/r9")
        assert resp.status_code == 200
        assert resp.json() == {
            "run_id": "r9", "status": "failed",
            "total_questions": 0, "completed_questions": 0,
            "current_model": "", "current_question": "",
            "elapsed_ms": 0.0, "error": "boom",
        }, (
            "a finished run is synthesised: identity from the PATH, status and error "
            "from the store, and the question counters flattened to zero"
        )
    finally:
        restore()

    runner = _Recorder(ret={"get_progress": None, "get_results": {"run_id": "r9"}})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/status/r9")
        body = resp.json()
        assert resp.status_code == 200
        assert body["status"] == "completed" and body["error"] == "", (
            "a stored record without a status is presumed completed"
        )
    finally:
        restore()

    runner = _Recorder(ret={"get_progress": None, "get_results": None})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/status/rx")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Run 'rx' not found"
    finally:
        restore()

    runner = _Recorder(ret={"get_progress": None, "get_results": {}})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/status/rx")
        assert resp.status_code == 404, (
            "an EMPTY stored record is indistinguishable from an absent one: the "
            "fallback tests truthiness, not presence"
        )
    finally:
        restore()


def test_v12_cancel_answers_cancelling_or_a_conflated_404():
    runner = _Recorder(ret={"cancel_run": True})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.post(f"{_PREFIX}/cancel/r1")
        assert resp.status_code == 200
        assert resp.json() == {"run_id": "r1", "status": "cancelling"}
        assert runner.calls == [("cancel_run", ("r1",), {})]
    finally:
        restore()

    runner = _Recorder(ret={"cancel_run": False})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.post(f"{_PREFIX}/cancel/r1")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Run 'r1' not found or already completed", (
            "one 404 covers both an unknown run and a finished one; the wording "
            "carries the conflation"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Results and judge ride-along


def test_v13_results_fill_the_gaps_and_prefer_the_stored_identity():
    data = {
        "run_id": "inner", "profile": "p",
        "model_scores": {"a": {"accuracy_avg": 1.5}},
        "question_results": {"a": [{"question_id": "q1"}]},
    }
    runner = _Recorder(ret={"get_results": data})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/results/outer")
        body = resp.json()
        assert resp.status_code == 200
        assert body["run_id"] == "inner", "the stored identity wins over the path"
        assert body["model_scores"]["a"] == {
            "model": "a", "accuracy_avg": 1.5, "code_avg": 0.0,
            "structure_avg": 0.0, "speed_avg": 0.0, "composite": 0.0,
            "questions_evaluated": 0,
        }, "a partial score record is completed with zeros and named after its key"
        assert body["question_results"]["a"][0]["question_id"] == "q1"
        assert body["question_results"]["a"][0]["composite_score"] == 0.0
        assert body["question_results"]["a"][0]["details"] == {}
        assert body["status"] == "" and body["weight_preset"] == "balanced"
        assert body["custom_weights"] is None and body["error"] == ""
        assert body["judge_scores"] == [] and body["judge_summary"] == {}, (
            "with the judge family down the judge fields stay at their defaults and "
            "no judge store is ever consulted"
        )
    finally:
        restore()

    runner = _Recorder(ret={"get_results": None})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/results/rx")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Run 'rx' not found"
    finally:
        restore()


def test_v14_judge_scores_ride_along_best_effort():
    judge = _Recorder(ret={
        "get_scores_for_run": [{"q": 1}],
        "get_summary_for_run": {"avg": 2},
    })
    runner = _Recorder(ret={"get_results": {"run_id": "r1"}})
    _, client, restore = _load(benchmark_runner=runner, judge_store=judge)
    try:
        resp = client.get(f"{_PREFIX}/results/r1")
        body = resp.json()
        assert resp.status_code == 200
        assert body["judge_scores"] == [{"q": 1}]
        assert body["judge_summary"] == {"avg": 2}
        assert judge.calls == [
            ("get_scores_for_run", ("r1",), {}),
            ("get_summary_for_run", ("r1",), {}),
        ]
    finally:
        restore()

    judge = _Recorder(ret={"get_scores_for_run": RuntimeError("judge db gone")})
    runner = _Recorder(ret={"get_results": {"run_id": "r1"}})
    _, client, restore = _load(benchmark_runner=runner, judge_store=judge)
    try:
        resp = client.get(f"{_PREFIX}/results/r1")
        body = resp.json()
        assert resp.status_code == 200
        assert body["judge_scores"] == [] and body["judge_summary"] == {}, (
            "a judge store that raises is swallowed: the results still answer, with "
            "the judge fields at their defaults"
        )
        assert judge.calls == [("get_scores_for_run", ("r1",), {})]
    finally:
        restore()


# ---------------------------------------------------------------------------
# Comparison, history, leaderboard, head-to-head, trends


def test_v15_compare_normalises_the_filter_and_echoes_three_fields():
    runner = _Recorder(ret={"compare": {
        "models": [{"model": "a"}], "profile_filter": "pf", "model_filter": ["a"],
    }})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/compare", params={
            "models": "a, ,b,,c ", "profile": "pf", "limit": 5,
        })
        assert resp.status_code == 200
        assert runner.calls == [("compare", (), {
            "models": ["a", "b", "c"], "profile": "pf", "limit": 5,
        })], "the comma filter is split, stripped and cleared of blanks"
        assert resp.json() == {
            "models": [{"model": "a"}], "profile_filter": "pf", "model_filter": ["a"],
        }
        runner.calls.clear()
        resp = client.get(f"{_PREFIX}/compare")
        assert resp.status_code == 200
        assert runner.calls == [("compare", (), {
            "models": None, "profile": None, "limit": 10,
        })], "no filter forwards None, not an empty list"
        assert client.get(f"{_PREFIX}/compare", params={"limit": 0}).status_code == 422
        assert client.get(f"{_PREFIX}/compare", params={"limit": 101}).status_code == 422
        assert client.get(f"{_PREFIX}/compare", params={"limit": 1}).status_code == 200
        assert client.get(f"{_PREFIX}/compare", params={"limit": 100}).status_code == 200
    finally:
        restore()


def test_v16_history_forwards_the_filters_and_counts_the_page():
    runs = [
        {
            "run_id": "r2", "profile": "p", "models": ["a"], "status": "completed",
            "started_at": 2.0, "duration_ms": 5.0,
            "model_scores": {"a": {"composite": 3.0}},
        },
        {"run_id": "r1"},
    ]
    runner = _Recorder(ret={"history": runs})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/history", params={
            "limit": 7, "profile": "p", "model": "a",
        })
        body = resp.json()
        assert resp.status_code == 200
        assert runner.calls == [("history", (), {
            "limit": 7, "profile": "p", "model": "a",
        })]
        first = body["runs"][0]
        assert first["run_id"] == "r2" and first["weight_preset"] == "balanced"
        assert first["model_scores"]["a"]["composite"] == 3.0
        assert first["model_scores"]["a"]["model"] == "a"
        assert body["runs"][1]["run_id"] == "r1"
        assert body["total"] == 2, (
            "the count is the PAGE length, not whatever total the store may hold"
        )
        assert client.get(f"{_PREFIX}/history", params={"limit": 0}).status_code == 422
        assert client.get(f"{_PREFIX}/history", params={"limit": 201}).status_code == 422
        assert client.get(f"{_PREFIX}/history", params={"limit": 200}).status_code == 200
    finally:
        restore()


def test_v17_leaderboard_ranks_the_compare_answer_and_never_fills_judge_avg():
    runner = _Recorder(ret={"compare": {"models": [
        {
            "model": "x", "avg_composite": 9.0, "avg_accuracy": 1.0,
            "avg_code": 2.0, "avg_structure": 3.0, "avg_speed": 4.0,
            "run_count": 5, "last_run": 6.0,
        },
        {"model": "y"},
    ]}})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/leaderboard", params={"profile": "p"})
        body = resp.json()
        assert resp.status_code == 200
        assert runner.calls == [("compare", (), {"profile": "p", "limit": 20})]
        assert body["entries"][0] == {
            "rank": 1, "model": "x", "composite": 9.0, "accuracy_avg": 1.0,
            "code_avg": 2.0, "structure_avg": 3.0, "speed_avg": 4.0,
            "judge_avg": 0.0, "run_count": 5, "last_run": 6.0,
        }, "aggregate keys are renamed on the way out and ranking starts at one"
        assert body["entries"][1]["rank"] == 2
        assert body["entries"][1]["judge_avg"] == 0.0, (
            "the judge average exists in the schema but the route never fills it"
        )
        assert body["profile"] == "p" and body["total"] == 2
        runner.calls.clear()
        resp = client.get(f"{_PREFIX}/leaderboard")
        assert resp.json()["profile"] == "" and runner.calls[0][2]["profile"] is None
    finally:
        restore()


def test_v18_head_to_head_declares_winners_per_metric_strictly():
    runner = _Recorder(ret={"compare": {"models": [
        {
            "model": "x", "avg_accuracy": 2.0, "avg_code": 1.0,
            "avg_structure": 3.0, "avg_speed": 1.0, "avg_composite": 2.0,
        },
        {
            "model": "y", "avg_accuracy": 1.0, "avg_code": 1.0,
            "avg_structure": 4.0, "avg_speed": 1.0, "avg_composite": 3.0,
        },
    ]}})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/head-to-head", params={
            "model_a": "x", "model_b": "y",
        })
        body = resp.json()
        assert resp.status_code == 200
        assert runner.calls == [("compare", (), {"models": ["x", "y"], "profile": None})]
        assert [m["metric"] for m in body["metrics"]] == [
            "accuracy", "code", "structure", "speed", "composite",
        ]
        assert [m["winner"] for m in body["metrics"]] == ["x", "tie", "y", "tie", "y"], (
            "a metric is won strictly: equal values are a tie, not a win"
        )
        assert body["model_a_wins"] == 1 and body["model_b_wins"] == 2
        assert body["ties"] == 2 and body["overall_winner"] == "y"
    finally:
        restore()


def test_v19_head_to_head_on_unknown_models_ties_everything():
    runner = _Recorder(ret={"compare": {"models": []}})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/head-to-head", params={
            "model_a": "x", "model_b": "y",
        })
        body = resp.json()
        assert resp.status_code == 200
        assert body["ties"] == 5 and body["overall_winner"] == "tie", (
            "models the store never saw are compared on default zeros: five ties and "
            "no 404 -- absence is indistinguishable from an all-zero record"
        )
        assert client.get(
            f"{_PREFIX}/head-to-head", params={"model_a": "x"},
        ).status_code == 422, "both model names are required at the edge"
    finally:
        restore()


def test_v20_trends_walk_history_oldest_first_and_skip_foreign_runs():
    runs = [
        {
            "run_id": "new", "started_at": 3.0, "profile": "p",
            "model_scores": {"mx": {
                "composite": 3.0, "accuracy_avg": 1.0, "code_avg": 2.0,
                "structure_avg": 3.0, "speed_avg": 4.0,
            }},
        },
        {"run_id": "mid", "started_at": 2.0, "model_scores": {"other": {}}},
        {
            "run_id": "old", "started_at": 1.0, "profile": "p",
            "model_scores": {"mx": {"composite": 1.0}},
        },
    ]
    runner = _Recorder(ret={"history": runs})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/trends", params={"model": "mx"})
        body = resp.json()
        assert resp.status_code == 200
        assert runner.calls == [("history", (), {
            "limit": 50, "profile": None, "model": "mx",
        })]
        assert [p["run_id"] for p in body["points"]] == ["old", "new"], (
            "the newest-first history is walked in reverse into a chronological "
            "series, and a run without this model contributes no point"
        )
        assert body["points"][1] == {
            "run_id": "new", "timestamp": 3.0, "composite": 3.0, "accuracy": 1.0,
            "code": 2.0, "structure": 3.0, "speed": 4.0, "profile": "p",
        }
    finally:
        restore()


def _trend_runs(values):
    """Newest-first history whose chronological composites are ``values``."""
    runs = []
    for stamp, composite in enumerate(values, 1):
        runs.append({
            "run_id": f"r{stamp}", "started_at": float(stamp),
            "model_scores": {"mx": {"composite": composite}},
        })
    runs.reverse()
    return runs


def test_v21_trend_direction_needs_three_points_and_five_percent():
    runner = _Recorder(ret={"history": _trend_runs([1.0, 2.0])})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        body = client.get(f"{_PREFIX}/trends", params={"model": "mx"}).json()
        assert body["trend_direction"] == "stable"
        assert body["regression_detected"] is False, (
            "fewer than three points never move the needle"
        )
        runner._ret["history"] = _trend_runs([1.0, 1.0, 1.0, 1.2, 1.2, 1.2])
        body = client.get(f"{_PREFIX}/trends", params={"model": "mx"}).json()
        assert body["trend_direction"] == "improving"
        assert body["regression_detected"] is False
        runner._ret["history"] = _trend_runs([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        body = client.get(f"{_PREFIX}/trends", params={"model": "mx"}).json()
        assert body["trend_direction"] == "declining"
        assert body["regression_detected"] is True, (
            "only a decline counts as a regression"
        )
        runner._ret["history"] = _trend_runs([1.0, 1.0, 1.0, 1.02, 1.02, 1.02])
        body = client.get(f"{_PREFIX}/trends", params={"model": "mx"}).json()
        assert body["trend_direction"] == "stable", (
            "a move inside the five percent band is noise"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Recommendations


def test_v22_recommendations_fall_back_to_generation_then_to_empty():
    snap = SimpleNamespace(
        snapshot_id="s1", created_at=1.0, profile="p",
        recommendations=[_ToDict(
            role="quality", model="m1", composite_score=9.5, reason="top",
        )],
        applied=True, applied_at=2.0,
    )
    reco = _Recorder(ret={"get_latest": snap})
    _, client, restore = _load(benchmark_recommender=reco)
    try:
        body = client.get(f"{_PREFIX}/recommendations").json()
        assert body["snapshot_id"] == "s1" and body["applied"] is True
        entry = body["recommendations"][0]
        assert entry["role"] == "quality" and entry["model"] == "m1"
        assert entry["composite_score"] == 9.5 and entry["reason"] == "top"
        assert entry["speed_score"] == 0.0, "missing entry fields settle on defaults"
        assert reco.calls == [("get_latest", (), {})], (
            "a stored snapshot answers by itself; generation is not attempted"
        )
    finally:
        restore()

    reco = _Recorder(ret={"get_latest": None, "generate_from_history": snap})
    _, client, restore = _load(benchmark_recommender=reco)
    try:
        body = client.get(f"{_PREFIX}/recommendations").json()
        assert body["snapshot_id"] == "s1"
        assert reco.calls == [
            ("get_latest", (), {}),
            ("generate_from_history", (), {}),
        ], "with no stored snapshot the route tries to generate one from history"
    finally:
        restore()

    reco = _Recorder(ret={
        "get_latest": None,
        "generate_from_history": RuntimeError("no history"),
    })
    _, client, restore = _load(benchmark_recommender=reco)
    try:
        resp = client.get(f"{_PREFIX}/recommendations")
        assert resp.status_code == 200
        assert resp.json() == {
            "snapshot_id": "", "created_at": 0.0, "profile": "",
            "recommendations": [], "applied": False, "applied_at": 0.0,
        }, "a generator that raises is swallowed into the all-default body"
        assert reco.calls[-1][0] == "generate_from_history"
    finally:
        restore()


def test_v23_apply_forwards_the_router_answer_with_defaults():
    reco = _Recorder(ret={"apply_to_smart_router": {
        "applied": True, "snapshot_id": "s1", "changes": {"k": 1}, "error": "",
    }})
    _, client, restore = _load(benchmark_recommender=reco)
    try:
        resp = client.post(f"{_PREFIX}/recommendations/apply")
        assert resp.status_code == 200
        assert resp.json() == {
            "applied": True, "snapshot_id": "s1", "changes": {"k": 1}, "error": "",
        }
        assert reco.calls == [("apply_to_smart_router", (), {})]
    finally:
        restore()

    reco = _Recorder(ret={"apply_to_smart_router": {}})
    _, client, restore = _load(benchmark_recommender=reco)
    try:
        assert client.post(f"{_PREFIX}/recommendations/apply").json() == {
            "applied": False, "snapshot_id": "", "changes": {}, "error": "",
        }, "a bare answer from the recommender settles on defaults"
    finally:
        restore()


def test_v24_apply_failure_is_a_200_that_says_so():
    reco = _Recorder(ret={"apply_to_smart_router": RuntimeError("router down")})
    _, client, restore = _load(benchmark_recommender=reco)
    try:
        resp = client.post(f"{_PREFIX}/recommendations/apply")
        assert resp.status_code == 200
        assert resp.json() == {
            "applied": False, "snapshot_id": "", "changes": {}, "error": "router down",
        }, (
            "a recommender that raises is answered as a SUCCESS status carrying "
            "applied false and the reason -- the launch path wraps the same failure "
            "as a 500; the two envelopes are deliberately unalike"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Export


def test_v25_export_polices_the_format_at_the_edge():
    runner = _Recorder(ret={"get_results": {"run_id": "r1"}})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/export/r1", params={"format": "xml"})
        assert resp.status_code == 422, "only json and csv pass the format pattern"
        resp = client.get(f"{_PREFIX}/export/r1")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json"), (
            "the default format is json"
        )
        resp = client.get(f"{_PREFIX}/export/r1", params={"format": "csv"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
    finally:
        restore()


def test_v26_json_export_carries_eleven_keys_and_names_the_path():
    data = {
        "run_id": "inner", "profile": "p", "models": ["a"], "status": "completed",
        "started_at": 1.0, "finished_at": 2.0, "duration_ms": 3.0,
        "model_scores": {"a": {"x": 1}}, "question_results": {"a": []},
        "custom_weights": {"accuracy": 1.0}, "error": "e",
    }
    runner = _Recorder(ret={"get_results": data})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/export/outer")
        body = resp.json()
        assert resp.status_code == 200
        assert set(body) == {
            "run_id", "profile", "models", "status", "started_at", "finished_at",
            "duration_ms", "weight_preset", "model_scores", "question_results",
            "judge_scores",
        }, (
            "the export is its own eleven-key shape: custom weights, error and the "
            "judge summary of the results endpoint are not part of it"
        )
        assert body["run_id"] == "inner"
        assert resp.headers["content-disposition"] == (
            'attachment; filename="benchmark_outer.json"'
        ), "the body names the store while the filename names the path"
        assert body["judge_scores"] == []
        assert body["model_scores"] == {"a": {"x": 1}}, (
            "export forwards the stored scores untouched, without the response "
            "model mapping"
        )
    finally:
        restore()


def test_v27_csv_export_orders_nine_columns_and_walks_the_questions():
    data = {
        "question_results": {
            "a": [{
                "question_id": "q1", "category": "c1", "accuracy_score": 1.5,
                "code_score": 2.0, "structure_score": 3.0, "speed_score": 4.0,
                "composite_score": 2.5,
            }],
            "b": [{"question_id": "q2", "category": "c2"}],
        },
    }
    runner = _Recorder(ret={"get_results": data})
    _, client, restore = _load(benchmark_runner=runner)
    try:
        resp = client.get(f"{_PREFIX}/export/r1", params={"format": "csv"})
        assert resp.status_code == 200
        assert resp.headers["content-disposition"] == (
            'attachment; filename="benchmark_r1.csv"'
        )
        lines = resp.text.splitlines()
        assert lines[0] == (
            "run_id,model,question_id,category,accuracy_score,code_score,"
            "structure_score,speed_score,composite_score"
        )
        assert lines[1] == "r1,a,q1,c1,1.5,2.0,3.0,4.0,2.5"
        assert lines[2] == "r1,b,q2,c2,0.0,0.0,0.0,0.0,0.0", (
            "one row per question, model by model, gaps filled with zeros"
        )
        assert len(lines) == 3
    finally:
        restore()


# ---------------------------------------------------------------------------
# Custom profile store


def test_v28_create_rejects_at_two_layers_and_polices_the_weights():
    store = _Recorder(ret={"create": _ToDict(profile_id="cp1", name="n")})
    _, client, restore = _load(custom_profile_store=store)
    try:
        assert client.post(f"{_PREFIX}/profiles/custom", json={}).status_code == 422, (
            "the name is the one field the request model itself requires"
        )
        resp = client.post(f"{_PREFIX}/profiles/custom", json={"name": "   "})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Profile name is required", (
            "a blank name passes the schema and is refused by the handler: two "
            "layers, two answers"
        )
        resp = client.post(f"{_PREFIX}/profiles/custom", json={
            "name": "n", "custom_weights": {"accuracy": 1.0},
        })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail.startswith("Custom weights missing keys:")
        for key in ("code", "structure", "speed"):
            assert key in detail
        assert store.calls == []
        resp = client.post(f"{_PREFIX}/profiles/custom", json={
            "name": "  n  ",
            "custom_weights": {
                "accuracy": 1.0, "code": 1.0, "structure": 1.0,
                "speed": 1.0, "bonus": 9.0,
            },
        })
        assert resp.status_code == 201
        assert store.calls == [("create", (), {
            "name": "n", "description": "", "categories": [],
            "weight_preset": "balanced",
            "custom_weights": {
                "accuracy": 1.0, "code": 1.0, "structure": 1.0,
                "speed": 1.0, "bonus": 9.0,
            },
            "timeout": 45, "max_response_tokens": 800,
            "expected_length_range": [10, 600],
        })], (
            "the stored name is stripped, the schema defaults travel to the store, "
            "and a surplus weight key is tolerated -- only MISSING keys are policed"
        )
        body = resp.json()
        assert body["profile_id"] == "cp1" and body["name"] == "n"
    finally:
        restore()


def test_v29_category_checks_stand_only_with_the_evaluator():
    evaluator = _Recorder(attrs={"available_categories": ["good"]})
    store = _Recorder(ret={
        "create": _ToDict(profile_id="cp1"),
        "update": _ToDict(profile_id="p1"),
    })
    _, client, restore = _load(custom_profile_store=store, benchmark_evaluator=evaluator)
    try:
        resp = client.post(f"{_PREFIX}/profiles/custom", json={
            "name": "n", "categories": ["good", "bad"],
        })
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Unknown categories: bad"
        resp = client.put(f"{_PREFIX}/profiles/custom/p1", json={
            "categories": ["bad"],
        })
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Unknown categories: bad"
        assert store.calls == [], "an unknown category never reaches the store"
    finally:
        restore()

    store = _Recorder(ret={
        "create": _ToDict(profile_id="cp1"),
        "update": _ToDict(profile_id="p1"),
    })
    _, client, restore = _load(custom_profile_store=store)
    try:
        resp = client.post(f"{_PREFIX}/profiles/custom", json={
            "name": "n", "categories": ["bad"],
        })
        assert resp.status_code == 201
        resp = client.put(f"{_PREFIX}/profiles/custom/p1", json={
            "categories": ["bad"],
        })
        assert resp.status_code == 200
        assert [c[0] for c in store.calls] == ["create", "update"], (
            "with the evaluator down the category check is SKIPPED on both writes: "
            "the same names are forwarded unchallenged"
        )
    finally:
        restore()


def test_v30_create_maps_store_errors_by_their_wording():
    store = _Recorder(ret={"create": ValueError("profile already exists")})
    _, client, restore = _load(custom_profile_store=store)
    try:
        resp = client.post(f"{_PREFIX}/profiles/custom", json={"name": "n"})
        assert resp.status_code == 409
        assert resp.json()["detail"] == "profile already exists"
        store._ret["create"] = ValueError("weights out of range")
        resp = client.post(f"{_PREFIX}/profiles/custom", json={"name": "n"})
        assert resp.status_code == 400, (
            "a ValueError is a conflict only when it SAYS so; any other wording is "
            "a bad request"
        )
        store._ret["create"] = RuntimeError("disk full")
        resp = client.post(f"{_PREFIX}/profiles/custom", json={"name": "n"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Failed to create profile: disk full"
    finally:
        restore()


def test_v31_update_filters_nulls_refuses_empty_and_404s_after_the_store():
    store = _Recorder(ret={"update": None})
    _, client, restore = _load(custom_profile_store=store)
    try:
        resp = client.put(f"{_PREFIX}/profiles/custom/p1", json={})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "No fields to update"
        resp = client.put(f"{_PREFIX}/profiles/custom/p1", json={
            "name": None, "timeout": None,
        })
        assert resp.status_code == 400, (
            "a field sent as null is indistinguishable from an omitted one: the "
            "None filter drops it, so this route cannot CLEAR a field"
        )
        assert store.calls == []
        resp = client.put(f"{_PREFIX}/profiles/custom/p1", json={"description": "d"})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Custom profile 'p1' not found"
        assert store.calls == [("update", ("p1", {"description": "d"}), {})]
        store._ret["update"] = _ToDict(profile_id="p1", name="n")
        resp = client.put(f"{_PREFIX}/profiles/custom/p1", json={"description": "d"})
        assert resp.status_code == 200 and resp.json()["profile_id"] == "p1"
        store._ret["update"] = ValueError("name already exists")
        resp = client.put(f"{_PREFIX}/profiles/custom/p1", json={"name": "x"})
        assert resp.status_code == 409
        store._ret["update"] = ValueError("nope")
        resp = client.put(f"{_PREFIX}/profiles/custom/p1", json={"name": "x"})
        assert resp.status_code == 400
    finally:
        restore()


def test_v32_delete_answers_flatly_or_404s():
    store = _Recorder(ret={"delete": True})
    _, client, restore = _load(custom_profile_store=store)
    try:
        resp = client.delete(f"{_PREFIX}/profiles/custom/p1")
        assert resp.status_code == 200
        assert resp.json() == {"profile_id": "p1", "deleted": True}
        assert store.calls == [("delete", ("p1",), {})]
        store._ret["delete"] = False
        resp = client.delete(f"{_PREFIX}/profiles/custom/p2")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Custom profile 'p2' not found"
    finally:
        restore()


def test_v33_listing_custom_profiles_counts_what_it_maps():
    store = _Recorder(ret={"list_profiles": [
        _ToDict(profile_id="a", name="A"),
        _ToDict(profile_id="b"),
    ]})
    _, client, restore = _load(custom_profile_store=store)
    try:
        resp = client.get(f"{_PREFIX}/profiles/custom")
        body = resp.json()
        assert resp.status_code == 200
        assert body["count"] == 2
        assert body["profiles"][0]["profile_id"] == "a"
        assert body["profiles"][0]["name"] == "A"
        assert body["profiles"][1]["name"] == "", (
            "each record travels through its own to_dict and settles on defaults"
        )
    finally:
        restore()


def test_v34_preview_takes_a_bare_array_and_passes_questions_conditionally():
    store = _Recorder(ret={"get_question_preview": {
        "category_counts": {"a": 2}, "total": 2,
    }})
    _, client, restore = _load(custom_profile_store=store)
    try:
        resp = client.post(f"{_PREFIX}/profiles/preview", json=["a", "b"])
        assert resp.status_code == 200
        assert resp.json() == {"category_counts": {"a": 2}, "total": 2}
        assert store.calls == [("get_question_preview", (["a", "b"], None), {})], (
            "with the evaluator down the question bank travels as None"
        )
        assert client.post(
            f"{_PREFIX}/profiles/preview", json={"x": 1},
        ).status_code == 422, "the body is a bare array of names, nothing else"
    finally:
        restore()

    store = _Recorder(ret={"get_question_preview": {"category_counts": {}, "total": 0}})
    evaluator = _Recorder(attrs={"questions": ["Q1"]})
    _, client, restore = _load(custom_profile_store=store, benchmark_evaluator=evaluator)
    try:
        client.post(f"{_PREFIX}/profiles/preview", json=["a"])
        assert store.calls == [("get_question_preview", (["a"], ["Q1"]), {})], (
            "with the evaluator up its question bank rides along"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Auto trigger


def test_v35_auto_trigger_readings_expand_and_filter_the_surplus():
    trigger = _Recorder(attrs={
        "status": {"enabled": True, "known_models": 3, "surplus": 1},
        "config": {"trigger_profile": "quick", "surplus": 2},
    })
    _, client, restore = _load(auto_trigger=trigger)
    try:
        resp = client.get(f"{_PREFIX}/auto-trigger/status")
        body = resp.json()
        assert resp.status_code == 200
        assert body["enabled"] is True and body["known_models"] == 3
        assert "surplus" not in body, (
            "the reading is expanded into the real model, which silently drops the "
            "keys it does not know"
        )
        assert body["running"] is False, "unnamed fields settle on defaults"
        resp = client.get(f"{_PREFIX}/auto-trigger/config")
        body = resp.json()
        assert body["trigger_profile"] == "quick" and "surplus" not in body
        assert body["trigger_models"] == "all_new"
    finally:
        restore()


def test_v36_config_update_filters_nulls_and_forwards_the_rest():
    trigger = _Recorder(ret={"update_config": {
        "enabled": False, "poll_interval_seconds": 30.0,
    }})
    _, client, restore = _load(auto_trigger=trigger)
    try:
        resp = client.put(f"{_PREFIX}/auto-trigger/config", json={})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "No fields to update"
        resp = client.put(f"{_PREFIX}/auto-trigger/config", json={
            "enabled": False, "judge_model": None,
        })
        assert resp.status_code == 200
        assert trigger.calls == [("update_config", ({"enabled": False},), {})], (
            "false SURVIVES the None filter while a null field is dropped: disabling "
            "by configuration is expressible, clearing a field is not"
        )
        body = resp.json()
        assert body["enabled"] is False and body["poll_interval_seconds"] == 30.0
        assert body["trigger_profile"] == "all_round"
    finally:
        restore()


def test_v37_enable_reports_the_thread_while_disable_asserts_it_down():
    trigger = _Recorder(attrs={"running": True})
    _, client, restore = _load(auto_trigger=trigger)
    try:
        resp = client.post(f"{_PREFIX}/auto-trigger/enable")
        assert resp.status_code == 200
        assert resp.json() == {"enabled": True, "running": True}
        assert trigger.calls == [("enable", (), {})], (
            "enabling re-reads the live attribute after the call"
        )
        resp = client.post(f"{_PREFIX}/auto-trigger/disable")
        assert resp.status_code == 200
        assert resp.json() == {"enabled": False, "running": False}, (
            "disabling HARDCODES running false in the answer -- the attribute still "
            "says true and is never consulted"
        )
        assert trigger.calls[-1] == ("disable", (), {})
    finally:
        restore()


def test_v38_test_poll_and_reset_pass_straight_through():
    trigger = _Recorder(ret={"test_poll": {
        "ok": True, "snapshot_models": 2, "model_names": ["a"],
        "diff": {"added": ["a"]},
    }})
    _, client, restore = _load(auto_trigger=trigger)
    try:
        resp = client.post(f"{_PREFIX}/auto-trigger/test-poll")
        body = resp.json()
        assert resp.status_code == 200
        assert body["ok"] is True and body["snapshot_models"] == 2
        assert body["model_names"] == ["a"] and body["diff"] == {"added": ["a"]}
        assert body["error"] == ""
        assert trigger.calls == [("test_poll", (), {})]
        resp = client.post(f"{_PREFIX}/auto-trigger/reset")
        assert resp.status_code == 200
        assert resp.json() == {"reset": True}
        assert trigger.calls[-1] == ("reset_snapshot", (), {})
    finally:
        restore()
