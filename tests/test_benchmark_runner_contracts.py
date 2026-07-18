#!/usr/bin/env python3
"""What the benchmark engine promises to the admission gate that fronts it.

The router in front of this engine refuses new work with a conflict answer
whenever the engine reports itself busy, and that report is the first promise
pinned here: busyness is a locked property over the live progress table, true
from the moment a run is registered -- before its worker thread has even
started -- and false again the moment the run reaches a terminal status. The
engine itself never serialises: a second run started while one is in flight is
accepted and executed concurrently, so the single-run discipline lives
entirely in the gate that consults the property, not in the engine behind it.
Finished runs stay in the progress table, same live object, so progress
remains consultable after the fact.

The run body is wrapped in a fail-safe outer guard. A crash anywhere inside
marks the run failed with the real error, persists a minimal failed record,
clears any pending cancel flag and fires the final callback -- so a crashed
run can never leave the engine permanently busy. The early refusals walk a
different road: a missing evaluator, an unknown profile or an empty question
list mark the progress with the true reason but persist NOTHING, and the
synchronous entry point then reports a failure whose message says only that
no stored result was found; the real reason survives in the progress record
alone. The synchronous return never carries model scores even on success --
scores live in the store and are reached through the results accessors.

Cancellation is cooperative and coarse: the flag is polled between questions
and between models, the partial run is persisted as cancelled with whatever
was measured, and the flag is dropped when the run terminalises. Cancelling
a run that already finished is still accepted, and that flag then lingers,
inert, because nothing will ever consume it.

Per-model resource admission follows the benchmark discipline: admit or
refuse, never downsize. A refused model is recorded as not admitted with the
governor's reason and skipped without ever touching the query seam, while the
run continues with the remaining models. The bounded queue entry point is
preferred over the plain one when the governor offers it, the caller is
tagged, and an admission that expects a load invalidates the governor
snapshot with the granted context. The gate fails open -- absent, disabled or
broken governors admit everything. Eviction between models goes through the
backend registry once per admitted model, sums whatever each backend reports,
absorbs individual failures, and obeys both its switch and the skip path.

The judge hand-off is conjunctive -- the caller must ask, name a judge model,
an instance must be bound and the judge module's own flag must be up -- and a
cancelled run never reaches it. The judge receives every question-response
pair with its model tag plus the very query seam the run used, and a judge
crash is absorbed without failing the run. Retention cleanup runs only after
completed runs, with the horizon read from the evaluator's profile data.

The transport helper has exactly one all-zero shape, returned when the client
library is absent; a client that breaks mid-stream keeps the elapsed time.
The streaming arm measures first-token latency at the first content-bearing
chunk, folds both chunk shapes, prefers the exact token count reported at the
end of the stream over the chunk approximation, and caches one client per
timeout. An empty generation zeroes every axis without consulting any
evaluator, and a code question without a sandbox is zeroed the same silent
way. Composites are renormalised over the axes actually evaluated.

The store attaches filtered scores without narrowing the run selection when
asked for one model's history, aggregates completed runs only, removes the
child rows of expired runs and reports how many runs fell -- and re-saving a
run replaces the run row while quietly duplicating its child rows.

Everything the module reaches for is seeded or declared unreachable and
proven so. The connection seam is seeded with a redirect so the module-level
singleton, which opens the default results database at load time, can never
touch the shipped file.
"""

import contextlib
import sqlite3
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.benchmark_runner"
_TARGET_SOURCE = source("benchmark_runner.py")
_SHIPPED_DEFAULT_DB = str(source("data", "benchmark_results.db"))

_DB_UTILS = "opti_oignon.db_utils"
_EVALUATOR = "opti_oignon.benchmark_evaluator"
_JUDGE = "opti_oignon.benchmark_judge"
_GOVERNOR = "opti_oignon.resource_governor"
_BACKENDS = "opti_oignon.inference_backend"
_SANDBOX = "opti_oignon.sandbox_manager"

_DEADLINE = 5.0


# ---------------------------------------------------------------------------
# Rigging: the connection redirect, the evaluator and judge stand-ins, the
# window loader, and the call-time seams the module resolves on every use.


def _redirecting_db_utils(tmp):
    """Connection seam that maps the shipped default database into the test
    directory, so the module-level singleton opens a scratch file instead of
    the repository's own results database. Every other path passes through.
    """
    module = ModuleType(_DB_UTILS)

    def safe_connect(path, **kwargs):
        text = str(path)
        if text == _SHIPPED_DEFAULT_DB:
            text = str(Path(tmp) / "module-default.db")
        return sqlite3.connect(text, **kwargs)

    module.safe_connect = safe_connect
    return module


def _evaluator_module():
    """Stand-in for the evaluator module: every name the engine imports, with
    result classes shaped exactly as the engine touches them and evaluation
    functions that count their consultations and answer fixed scores.
    """
    module = ModuleType(_EVALUATOR)
    module.BENCHMARK_EVALUATOR_AVAILABLE = True
    probe = {
        "accuracy": 0,
        "code": 0,
        "structure": 0,
        "performance": 0,
        "composite": [],
        "custom": 0,
        "preset": 0,
    }

    class AccuracyResult:
        def __init__(self, question_id="", score=0.0, method="", details="",
                     matched_answer=""):
            self.question_id = question_id
            self.score = score
            self.method = method
            self.details = details
            self.matched_answer = matched_answer

    class CodeResult:
        def __init__(self, question_id="", score=0.0, details="", compiles=False,
                     runs=False, output_matches=False, tests_pass=False):
            self.question_id = question_id
            self.score = score
            self.details = details
            self.compiles = compiles
            self.runs = runs
            self.output_matches = output_matches
            self.tests_pass = tests_pass

    class StructuralResult:
        def __init__(self, repetition_score=0.0, lexical_diversity=0.0,
                     length_appropriateness=0.0, format_compliance=0.0,
                     composite=0.0):
            self.repetition_score = repetition_score
            self.lexical_diversity = lexical_diversity
            self.length_appropriateness = length_appropriateness
            self.format_compliance = format_compliance
            self.composite = composite

    class PerformanceResult:
        def __init__(self, ttft_ms=0.0, tokens_per_second=0.0,
                     total_time_ms=0.0, score=0.0):
            self.ttft_ms = ttft_ms
            self.tokens_per_second = tokens_per_second
            self.total_time_ms = total_time_ms
            self.score = score

    class QuestionResult:
        def __init__(self, question_id="", category="", prompt="", response=""):
            self.question_id = question_id
            self.category = category
            self.prompt = prompt
            self.response = response
            self.accuracy = None
            self.code = None
            self.structure = None
            self.performance = None
            self.composite_score = 0.0

    class WeightPreset:
        def __init__(self, accuracy=0.35, code=0.25, structure=0.25, speed=0.15):
            self.accuracy = accuracy
            self.code = code
            self.structure = structure
            self.speed = speed

    class Question:
        def __init__(self, question_id, category, prompt):
            self.id = question_id
            self.category = category
            self.prompt = prompt

    def evaluate_accuracy(question, response):
        probe["accuracy"] += 1
        return AccuracyResult(question_id=question.id, score=0.8, method="fixed")

    def evaluate_code(question, response):
        probe["code"] += 1
        return CodeResult(question_id=question.id, score=0.6, compiles=True)

    def evaluate_structure(response, expected_range, format_check):
        probe["structure"] += 1
        return StructuralResult(composite=0.7)

    def evaluate_performance(ttft_ms=0.0, tokens_per_second=0.0, total_time_ms=0.0):
        probe["performance"] += 1
        return PerformanceResult(ttft_ms=ttft_ms, tokens_per_second=tokens_per_second,
                                 total_time_ms=total_time_ms, score=0.9)

    def compute_composite_score(accuracy, code, structure, speed, weights,
                                evaluated=None):
        probe["composite"].append(
            (accuracy, code, structure, speed, tuple(sorted(evaluated or ()))),
        )
        return 0.5

    class BenchmarkEvaluator:
        def __init__(self, profiles=None, questions=None, custom=None,
                     presets=None, profiles_data=None):
            self._profiles = profiles or {}
            self._questions = questions or {}
            self._custom = custom or {}
            self._presets = presets or {}
            self.profiles_data = profiles_data or {}

        def get_profile_config(self, profile):
            return self._profiles.get(profile)

        def get_questions_for_profile(self, profile):
            return self._questions.get(profile, [])

        def get_custom_weights(self, profile):
            probe["custom"] += 1
            return self._custom.get(profile)

        def get_weights(self, name):
            probe["preset"] += 1
            return self._presets.get(name, WeightPreset())

    module.AccuracyResult = AccuracyResult
    module.BenchmarkEvaluator = BenchmarkEvaluator
    module.CodeResult = CodeResult
    module.PerformanceResult = PerformanceResult
    module.Question = Question
    module.QuestionResult = QuestionResult
    module.StructuralResult = StructuralResult
    module.WeightPreset = WeightPreset
    module.benchmark_evaluator = BenchmarkEvaluator()
    module.compute_composite_score = compute_composite_score
    module.evaluate_accuracy = evaluate_accuracy
    module.evaluate_code = evaluate_code
    module.evaluate_structure = evaluate_structure
    module.evaluate_performance = evaluate_performance
    module.probe = probe
    return module


def _judge_module(available=True):
    module = ModuleType(_JUDGE)
    module.BENCHMARK_JUDGE_AVAILABLE = available

    class BenchmarkJudge:
        pass

    module.BenchmarkJudge = BenchmarkJudge
    module.benchmark_judge = None
    return module


def _load(tmp, *, ollama=None, judge_available=True):
    """Load the engine from source inside the shared window.

    The direct inference client is unreachable by default; a transport test
    hands in its own stand-in instead. The governor, the backend registry and
    the sandbox are declared unreachable and proven so; the first two are
    resolved by the engine at call time, so a test seeds them straight into
    the module cache for the duration of one call path.
    """
    evaluator = _evaluator_module()
    seeds = {
        _DB_UTILS: _redirecting_db_utils(tmp),
        _EVALUATOR: evaluator,
        _JUDGE: _judge_module(available=judge_available),
    }
    blocks = [_GOVERNOR, _BACKENDS, _SANDBOX]
    if ollama is None:
        blocks.append("ollama")
    else:
        seeds["ollama"] = ollama
    loaded, restore = isolate(
        targets={_TARGET: _TARGET_SOURCE},
        blocked=tuple(blocks),
        seeded=seeds,
    )
    return loaded[_TARGET], evaluator, restore


def _profiled(evaluator, *, profile="p1", retention=1000, custom=None,
              questions=None):
    """A configured evaluator instance: one profile, one plain question and
    one code-generation question unless the test brings its own roster.
    """
    make = evaluator.Question
    if questions is None:
        questions = [make("q1", "general", "P1"), make("q2", "code_generation", "P2")]
    return evaluator.BenchmarkEvaluator(
        profiles={profile: {
            "weight_preset": "balanced",
            "timeout": 7,
            "max_response_tokens": 55,
            "expected_length_range": [5, 100],
            "format_check": "prose",
        }},
        questions={profile: questions},
        custom={profile: custom} if custom is not None else {},
        presets={"balanced": evaluator.WeightPreset()},
        profiles_data={"runner": {"results_retention_days": retention}},
    )


def _plain_answer(model, prompt, timeout, max_tokens):
    return ("a steady answer", 12.0, 100.0, 42)


def _poll(condition, deadline=_DEADLINE):
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


def _settle(runner, module, *run_ids):
    """Every spawned run must reach a terminal status before the window
    closes; a worker thread must never outlive its window.
    """
    live = (module.RunStatus.PENDING, module.RunStatus.RUNNING)

    def done(run_id):
        progress = runner.get_progress(run_id)
        return progress is None or progress.status not in live

    for run_id in run_ids:
        assert _poll(lambda: done(run_id)), f"run did not settle: {run_id}"


@contextlib.contextmanager
def _seeded_at_call_time(name, module):
    """Place a stand-in where the engine's call-time resolution looks first,
    and put the neutralised entry back when the path has been walked.
    """
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        sys.modules[name] = previous


class _Decision:
    def __init__(self, admitted, reason="", load_expected=False, num_ctx=0):
        self.admitted = admitted
        self.reason = reason
        self.load_expected = load_expected
        self.num_ctx = num_ctx


class _Governor:
    def __init__(self, log, *, enabled=True, refused=(), load_expected=True):
        self.config = SimpleNamespace(enabled=enabled)
        self._log = log
        self._refused = set(refused)
        self._load_expected = load_expected

    def admit_or_wait(self, model, ctx, caller=""):
        self._log.append(("admit_or_wait", model, ctx, caller))
        if model in self._refused:
            return _Decision(False, reason="no room")
        return _Decision(True, load_expected=self._load_expected, num_ctx=4096)

    def admit(self, model, ctx, caller=""):
        self._log.append(("admit", model))
        return _Decision(True)

    def invalidate_on_load(self, model, num_ctx):
        self._log.append(("invalidate_on_load", model, num_ctx))

    def invalidate_on_evict(self, argument):
        self._log.append(("invalidate_on_evict", argument))


def _governor_seed(governor):
    module = ModuleType(_GOVERNOR)
    module.FEATURE_AVAILABLE = True

    def get_resource_governor():
        return governor

    module.get_resource_governor = get_resource_governor
    return module


def _broken_governor_seed():
    module = ModuleType(_GOVERNOR)
    module.FEATURE_AVAILABLE = True

    def get_resource_governor():
        raise RuntimeError("governor down")

    module.get_resource_governor = get_resource_governor
    return module


class _Backend:
    def __init__(self, log, tag, *, unloaded=0, failing=False):
        self._log = log
        self._tag = tag
        self._unloaded = unloaded
        self._failing = failing

    def unload_all(self):
        self._log.append(self._tag)
        if self._failing:
            raise RuntimeError("unload down")
        return self._unloaded


def _backend_seed(*backends):
    module = ModuleType(_BACKENDS)
    registry = SimpleNamespace(backends=lambda: list(backends))

    def get_backend_registry():
        return registry

    module.get_backend_registry = get_backend_registry
    return module


class _ObjectChunk:
    """Typed-object chunk shape; a role-only preamble carries no content."""

    def __init__(self, content=None, eval_count=None, role_only=False):
        if role_only:
            self.message = SimpleNamespace(content=None)
        elif content is not None:
            self.message = SimpleNamespace(content=content)
        else:
            self.message = None
        if eval_count is not None:
            self.eval_count = eval_count


class _StreamClient:
    instances = []

    def __init__(self, timeout=None):
        type(self).instances.append(timeout)
        self.timeout = timeout
        self.asked = []

    def chat(self, model, messages, stream, options):
        self.asked.append((model, messages, options))
        return iter([
            _ObjectChunk(role_only=True),
            {"message": {"content": "He"}},
            _ObjectChunk(content="llo"),
            {"eval_count": 7},
        ])


class _BreakingClient:
    def __init__(self, timeout=None):
        self.timeout = timeout

    def chat(self, model, messages, stream, options):
        raise RuntimeError("transport down")


def _ollama_seed(client_class):
    module = ModuleType("ollama")
    module.Client = client_class
    return module


# ---------------------------------------------------------------------------
# Load: the module-level singleton and the wiring the window seeded.


def test_r1_the_module_singleton_rises_through_the_connection_seam(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        assert module.BENCHMARK_RUNNER_AVAILABLE is True
        assert isinstance(module.benchmark_runner, module.BenchmarkRunner)
        assert module.benchmark_runner.store is module.results_store
        # The singleton opened the DEFAULT database at load time -- through
        # the seeded redirect, never through the shipped file.
        assert (tmp_path / "module-default.db").exists()
    finally:
        restore()


def test_r2_optional_dependency_wiring_mirrors_what_the_window_seeded(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        assert module.BENCHMARK_EVALUATOR_AVAILABLE is True
        assert module.benchmark_evaluator is evaluator.benchmark_evaluator
        assert module.BENCHMARK_JUDGE_AVAILABLE is True
        assert module._default_judge is None
        assert module.SANDBOX_AVAILABLE is False
        assert module.sandbox_manager is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# Busyness and the run lifecycle: the property the admission gate consults.


def test_r3_a_fresh_engine_is_idle_and_knows_no_run(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        assert runner.is_busy is False
        assert runner.get_progress("run-unknown") is None
    finally:
        restore()


def test_r4_a_run_counts_as_busy_from_registration_through_running(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    gate = threading.Event()
    entered = threading.Event()
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")

        def held_answer(model, prompt, timeout, max_tokens):
            entered.set()
            gate.wait(_DEADLINE)
            return ("held", 1.0, 10.0, 3)

        run_id = runner.start_run("p1", ["m1"], query_fn=held_answer)
        # Registered before the worker thread even starts: already busy.
        assert runner.is_busy is True
        assert entered.wait(_DEADLINE)
        assert runner.get_progress(run_id).status is module.RunStatus.RUNNING
        assert runner.is_busy is True
        gate.set()
        _settle(runner, module, run_id)
        assert runner.is_busy is False
    finally:
        gate.set()
        restore()


def test_r5_a_finished_run_leaves_the_engine_idle_and_its_progress_retained(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        result = runner.run_sync("p1", ["m1"], query_fn=_plain_answer)
        assert result.status is module.RunStatus.COMPLETED
        assert runner.is_busy is False
        first = runner.get_progress(result.run_id)
        assert first is not None
        assert first.status is module.RunStatus.COMPLETED
        # The very same live progress object stays consultable afterwards.
        assert runner.get_progress(result.run_id) is first
        assert first.completed_questions == 2
    finally:
        restore()


def test_r6_the_engine_itself_never_serialises_concurrent_runs(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    gate = threading.Event()
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        started = []

        def held_answer(model, prompt, timeout, max_tokens):
            started.append(model)
            gate.wait(_DEADLINE)
            return ("held", 1.0, 10.0, 3)

        first = runner.start_run("p1", ["m1"], query_fn=held_answer)
        second = runner.start_run("p1", ["m2"], query_fn=held_answer)
        assert second != first
        # The second run is accepted while the first is in flight; the
        # one-run-at-a-time discipline is the gate's, not the engine's.
        assert runner.get_progress(second) is not None
        assert runner.is_busy is True
        assert _poll(lambda: set(started) == {"m1", "m2"})
        gate.set()
        _settle(runner, module, first, second)
        assert runner.get_progress(first).status is module.RunStatus.COMPLETED
        assert runner.get_progress(second).status is module.RunStatus.COMPLETED
    finally:
        gate.set()
        restore()


def test_r7_progress_updates_for_an_unknown_run_vanish_silently(tmp_path):
    # The worker thread reports through this path; an unregistered run must
    # vanish silently rather than raise mid-thread, and a partial update
    # must keep every field it did not carry.
    module, evaluator, restore = _load(tmp_path)
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        assert runner._update_progress("run-unknown", module.RunStatus.FAILED,
                                       error="lost") is None
        assert runner.get_progress("run-unknown") is None
        result = runner.run_sync("p1", ["m1"], query_fn=_plain_answer)
        progress = runner.get_progress(result.run_id)
        assert progress.total_questions == 2
        runner._update_progress(result.run_id, module.RunStatus.FAILED)
        assert progress.status is module.RunStatus.FAILED
        assert progress.total_questions == 2
        assert progress.error == ""
    finally:
        restore()


# ---------------------------------------------------------------------------
# The synchronous path: persistence, plumbing, and what the return carries.


def test_r8_a_completed_run_persists_run_scores_and_question_rows(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        result = runner.run_sync("p1", ["m1"], query_fn=_plain_answer)
        row = runner.store.get_run(result.run_id)
        assert row["status"] == "completed"
        assert row["models"] == ["m1"]
        assert row["custom_weights"] is None
        scores = row["model_scores"]["m1"]
        assert scores["questions_evaluated"] == 2
        assert scores["not_admitted"] == 0
        assert scores["accuracy_avg"] == 0.8
        assert scores["code_avg"] == 0.0
        assert scores["structure_avg"] == 0.7
        assert scores["speed_avg"] == 0.9
        assert scores["composite"] == 0.5
        details = runner.store.get_run_details(result.run_id)
        questions = details["question_results"]["m1"]
        assert [entry["question_id"] for entry in questions] == ["q1", "q2"]
        assert all(entry["composite_score"] == 0.5 for entry in questions)
        assert "code" in questions[1]["details"]
    finally:
        restore()


def test_r9_the_synchronous_return_carries_status_and_timing_but_no_scores(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        result = runner.run_sync("p1", ["m1"], query_fn=_plain_answer)
        assert result.status is module.RunStatus.COMPLETED
        assert result.model_scores == {}
        assert result.started_at > 0.0
        assert result.finished_at >= result.started_at
        assert result.duration_ms >= 0.0
        assert result.weight_preset == "balanced"
        assert result.error == ""
    finally:
        restore()


def test_r10_profile_timeout_and_token_budget_reach_the_query_seam(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        seen = []

        def recording_answer(model, prompt, timeout, max_tokens):
            seen.append((model, prompt, timeout, max_tokens))
            return ("a steady answer", 12.0, 100.0, 42)

        runner.run_sync("p1", ["m1"], query_fn=recording_answer)
        assert seen == [("m1", "P1", 7, 55), ("m1", "P2", 7, 55)]
    finally:
        restore()


def test_r11_weight_resolution_prefers_the_call_then_the_profile_then_the_preset(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        probe = evaluator.probe

        # The call's own weights win; neither evaluator road is consulted.
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "a.db")
        result = runner.run_sync("p1", ["m1"], query_fn=_plain_answer,
                                 custom_weights={"accuracy": 0.5})
        assert (probe["custom"], probe["preset"]) == (0, 0)
        assert runner.store.get_run(result.run_id)["custom_weights"] == {
            "accuracy": 0.5,
        }

        # The profile's own custom weights come next and are persisted whole.
        custom = evaluator.WeightPreset(accuracy=0.4, code=0.3, structure=0.2,
                                        speed=0.1)
        runner = module.BenchmarkRunner(
            evaluator=_profiled(evaluator, custom=custom),
            db_path=tmp_path / "b.db",
        )
        result = runner.run_sync("p1", ["m1"], query_fn=_plain_answer)
        assert (probe["custom"], probe["preset"]) == (1, 0)
        assert runner.store.get_run(result.run_id)["custom_weights"] == {
            "accuracy": 0.4, "code": 0.3, "structure": 0.2, "speed": 0.1,
        }

        # With neither, the named preset is resolved and nothing is persisted.
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "c.db")
        result = runner.run_sync("p1", ["m1"], query_fn=_plain_answer)
        assert (probe["custom"], probe["preset"]) == (2, 1)
        assert runner.store.get_run(result.run_id)["custom_weights"] is None
    finally:
        restore()


def test_r12_composites_renormalise_over_the_axes_actually_evaluated(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        runner.run_sync("p1", ["m1"], query_fn=_plain_answer)
        axes = [entry[4] for entry in evaluator.probe["composite"]]
        assert axes == [
            ("accuracy", "speed", "structure"),
            ("code", "speed", "structure"),
            ("accuracy", "code", "speed", "structure"),
        ]
    finally:
        restore()


# ---------------------------------------------------------------------------
# Early refusals and the fail-safe outer guard.


def test_r13_early_failures_mark_progress_but_persist_nothing(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "a.db")
        result = runner.run_sync("absent", ["m1"], query_fn=_plain_answer)
        assert result.status is module.RunStatus.FAILED
        # The synchronous return masks the reason; the progress keeps it.
        assert result.error == "Run result not found in store"
        progress = runner.get_progress(result.run_id)
        assert progress.status is module.RunStatus.FAILED
        assert progress.error == "Profile 'absent' not found"
        assert runner.store.get_run(result.run_id) is None
        assert runner.is_busy is False

        empty = _profiled(evaluator)
        empty._questions["p1"] = []
        runner = module.BenchmarkRunner(evaluator=empty, db_path=tmp_path / "b.db")
        result = runner.run_sync("p1", ["m1"], query_fn=_plain_answer)
        assert result.error == "Run result not found in store"
        progress = runner.get_progress(result.run_id)
        assert progress.error == "No questions found for profile 'p1'"
        assert runner.store.get_run(result.run_id) is None
    finally:
        restore()


def test_r14_a_missing_evaluator_fails_the_run_before_anything_else(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        # The constructor falls back to the module-level evaluator on any
        # falsy argument, so the bare-engine state is set directly.
        runner._evaluator = None
        called = []
        result = runner.run_sync("p1", ["m1"],
                                 query_fn=lambda *a: called.append(a))
        assert result.status is module.RunStatus.FAILED
        progress = runner.get_progress(result.run_id)
        assert progress.error == "Evaluator not available"
        assert runner.store.get_run(result.run_id) is None
        assert called == []
    finally:
        restore()


def test_r15_a_crash_in_the_run_body_leaves_no_zombie(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        reported = []

        def breaking_answer(model, prompt, timeout, max_tokens):
            raise RuntimeError("kaput")

        result = runner.run_sync("p1", ["m1"], query_fn=breaking_answer,
                                 progress_callback=reported.append)
        progress = runner.get_progress(result.run_id)
        assert progress.status is module.RunStatus.FAILED
        assert progress.error == "kaput"
        assert runner.is_busy is False
        row = runner.store.get_run(result.run_id)
        assert row["status"] == "failed"
        assert row["error"] == "kaput"
        assert row["model_scores"] == {}
        assert reported and reported[-1].status is module.RunStatus.FAILED
        # A minimal record exists, so the synchronous return reads it whole.
        assert result.status is module.RunStatus.FAILED
        assert result.error == "kaput"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Cancellation: cooperative, coarse, persisted partial.


def test_r16_cancellation_persists_the_partial_run_and_clears_the_flag(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    gate = threading.Event()
    entered = threading.Event()
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        asked = []

        def held_answer(model, prompt, timeout, max_tokens):
            asked.append((model, prompt))
            entered.set()
            gate.wait(_DEADLINE)
            return ("held", 1.0, 10.0, 3)

        run_id = runner.start_run("p1", ["m1", "m2"], query_fn=held_answer)
        assert entered.wait(_DEADLINE)
        assert runner.cancel_run(run_id) is True
        gate.set()
        _settle(runner, module, run_id)
        assert runner.get_progress(run_id).status is module.RunStatus.CANCELLED
        row = runner.store.get_run(run_id)
        assert row["status"] == "cancelled"
        # The flag was seen after the first question: the first model keeps
        # its one measurement, the second model was never queried.
        assert asked == [("m1", "P1")]
        assert set(row["model_scores"]) == {"m1"}
        assert row["model_scores"]["m1"]["questions_evaluated"] == 1
        # Terminalisation dropped the pending flag; nothing public reflects
        # the set, so it is observed directly.
        assert run_id not in runner._cancelled
        assert runner.is_busy is False
    finally:
        gate.set()
        restore()


def test_r17_cancelling_an_unknown_run_is_refused(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        assert runner.cancel_run("run-unknown") is False
    finally:
        restore()


def test_r18_cancelling_a_finished_run_is_accepted_and_the_flag_lingers(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        first = runner.run_sync("p1", ["m1"], query_fn=_plain_answer)
        assert runner.get_progress(first.run_id).status is module.RunStatus.COMPLETED
        # Finished runs stay registered, so the cancel is still accepted --
        # and the flag then lingers with nothing left to consume it.
        assert runner.cancel_run(first.run_id) is True
        assert first.run_id in runner._cancelled
        # The lingering flag is keyed to the old run and stays inert.
        second = runner.run_sync("p1", ["m1"], query_fn=_plain_answer)
        assert second.status is module.RunStatus.COMPLETED
        assert first.run_id in runner._cancelled
    finally:
        restore()


# ---------------------------------------------------------------------------
# Per-model admission: admit or refuse, never downsize; the gate fails open.


def test_r19_a_refused_model_is_recorded_and_skipped_while_the_run_continues(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        log = []
        seed = _governor_seed(_Governor(log, refused={"m2"}))
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        asked = []

        def recording_answer(model, prompt, timeout, max_tokens):
            asked.append(model)
            return ("a steady answer", 12.0, 100.0, 42)

        with _seeded_at_call_time(_GOVERNOR, seed):
            result = runner.run_sync("p1", ["m1", "m2"],
                                     query_fn=recording_answer,
                                     evict_between=False)
        assert result.status is module.RunStatus.COMPLETED
        assert "m2" not in asked
        assert "m1" in asked
        scores = runner.store.get_run(result.run_id)["model_scores"]
        assert scores["m2"]["not_admitted"] == 1
        assert scores["m2"]["admission_reason"] == "no room"
        assert scores["m2"]["questions_evaluated"] == 0
        assert scores["m1"]["not_admitted"] == 0
    finally:
        restore()


def test_r20_an_admission_expecting_a_load_invalidates_and_the_queue_entry_wins(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        log = []
        seed = _governor_seed(_Governor(log))
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "a.db")
        with _seeded_at_call_time(_GOVERNOR, seed):
            runner.run_sync("p1", ["m1"], query_fn=_plain_answer,
                            evict_between=False)
        assert ("invalidate_on_load", "m1", 4096) in log
        # The bounded queue entry is preferred over the plain one, and the
        # caller is tagged on every admission.
        assert not any(entry[0] == "admit" for entry in log)
        admissions = [entry for entry in log if entry[0] == "admit_or_wait"]
        assert admissions and all(entry[3] == "benchmark" for entry in admissions)

        log.clear()
        quiet = _governor_seed(_Governor(log, load_expected=False))
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "b.db")
        with _seeded_at_call_time(_GOVERNOR, quiet):
            runner.run_sync("p1", ["m1"], query_fn=_plain_answer,
                            evict_between=False)
        assert not any(entry[0] == "invalidate_on_load" for entry in log)
    finally:
        restore()


def test_r21_the_admission_gate_fails_open_when_the_governor_is_gone(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        asked = []

        def recording_answer(model, prompt, timeout, max_tokens):
            asked.append(model)
            return ("a steady answer", 12.0, 100.0, 42)

        # Absent: the window blocks the governor module outright.
        runner.run_sync("p1", ["m1"], query_fn=recording_answer,
                        evict_between=False)
        assert asked.count("m1") == 2

        # Disabled: even a refusing governor admits everything.
        log = []
        disabled = _governor_seed(_Governor(log, enabled=False, refused={"m1"}))
        with _seeded_at_call_time(_GOVERNOR, disabled):
            runner.run_sync("p1", ["m1"], query_fn=recording_answer,
                            evict_between=False)
        assert asked.count("m1") == 4

        # Broken: a governor accessor that raises degrades to no gate.
        with _seeded_at_call_time(_GOVERNOR, _broken_governor_seed()):
            runner.run_sync("p1", ["m1"], query_fn=recording_answer,
                            evict_between=False)
        assert asked.count("m1") == 6
    finally:
        restore()


# ---------------------------------------------------------------------------
# Eviction between models: registry route, best-effort sum, switch and skip.


def test_r22_eviction_runs_once_per_admitted_model_and_sums_best_effort(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        unloads = []
        backends = _backend_seed(
            _Backend(unloads, "first", unloaded=2),
            _Backend(unloads, "second", failing=True),
        )
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "runs.db")
        with _seeded_at_call_time(_BACKENDS, backends):
            runner.run_sync("p1", ["m1", "m2"], query_fn=_plain_answer)
        # One clean-slate pass after each admitted model, every backend asked.
        assert unloads == ["first", "second", "first", "second"]

        # The helper's own promise: failures are absorbed into the sum, and
        # a reachable governor sees its snapshot invalidated afterwards.
        unloads.clear()
        log = []
        governor = _governor_seed(_Governor(log))
        with _seeded_at_call_time(_BACKENDS, backends), \
                _seeded_at_call_time(_GOVERNOR, governor):
            assert module._evict_loaded_models() == 2
        assert unloads == ["first", "second"]
        assert ("invalidate_on_evict", None) in log
    finally:
        restore()


def test_r23_eviction_obeys_its_switch_and_skipped_models_trigger_none(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        unloads = []
        backends = _backend_seed(_Backend(unloads, "first", unloaded=1))
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "a.db")
        with _seeded_at_call_time(_BACKENDS, backends):
            runner.run_sync("p1", ["m1"], query_fn=_plain_answer,
                            evict_between=False)
        assert unloads == []

        # A refused model never reaches the clean-slate pass either.
        log = []
        refusing = _governor_seed(_Governor(log, refused={"m1"}))
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "b.db")
        with _seeded_at_call_time(_BACKENDS, backends), \
                _seeded_at_call_time(_GOVERNOR, refusing):
            result = runner.run_sync("p1", ["m1"], query_fn=_plain_answer)
        assert result.status is module.RunStatus.COMPLETED
        assert unloads == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# The judge hand-off: conjunctive gate, full payload, crash absorbed.


def test_r24_the_judge_gate_is_conjunctive_and_hands_over_the_full_payload(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        received = {}

        class Judge:
            def evaluate_run(self, run_id, judge_model, question_responses,
                             query_fn):
                received.update(run_id=run_id, judge_model=judge_model,
                                pairs=question_responses, query_fn=query_fn)
                return SimpleNamespace(total_evaluations=1, total_tokens=2)

        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "a.db",
                                        judge=Judge())
        result = runner.run_sync("p1", ["m1"], query_fn=_plain_answer,
                                 use_judge=True, judge_model="jm")
        assert received["run_id"] == result.run_id
        assert received["judge_model"] == "jm"
        assert received["query_fn"] is _plain_answer
        assert {pair["model"] for pair in received["pairs"]} == {"m1"}
        assert {pair["question_id"] for pair in received["pairs"]} == {"q1", "q2"}
        assert all(pair["response"] == "a steady answer"
                   for pair in received["pairs"])

        received.clear()
        runner.run_sync("p1", ["m1"], query_fn=_plain_answer,
                        use_judge=False, judge_model="jm")
        assert received == {}
        runner.run_sync("p1", ["m1"], query_fn=_plain_answer,
                        use_judge=True, judge_model="")
        assert received == {}
    finally:
        restore()

    # Fourth operand: the judge module's own flag, read where it was bound.
    module, evaluator, restore = _load(tmp_path, judge_available=False)
    try:
        received = {}

        class Judge:
            def evaluate_run(self, **kwargs):
                received.update(kwargs)

        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "b.db",
                                        judge=Judge())
        result = runner.run_sync("p1", ["m1"], query_fn=_plain_answer,
                                 use_judge=True, judge_model="jm")
        assert result.status is module.RunStatus.COMPLETED
        assert received == {}
    finally:
        restore()


def test_r25_a_judge_crash_is_absorbed_and_a_cancelled_run_never_judges(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    gate = threading.Event()
    entered = threading.Event()
    try:
        class BreakingJudge:
            def evaluate_run(self, **kwargs):
                raise RuntimeError("judge down")

        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "a.db",
                                        judge=BreakingJudge())
        result = runner.run_sync("p1", ["m1"], query_fn=_plain_answer,
                                 use_judge=True, judge_model="jm")
        assert result.status is module.RunStatus.COMPLETED

        reached = []

        class Judge:
            def evaluate_run(self, **kwargs):
                reached.append(kwargs)

        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "b.db",
                                        judge=Judge())

        def held_answer(model, prompt, timeout, max_tokens):
            entered.set()
            gate.wait(_DEADLINE)
            return ("held", 1.0, 10.0, 3)

        run_id = runner.start_run("p1", ["m1"], query_fn=held_answer,
                                  use_judge=True, judge_model="jm")
        assert entered.wait(_DEADLINE)
        assert runner.cancel_run(run_id) is True
        gate.set()
        _settle(runner, module, run_id)
        assert runner.get_progress(run_id).status is module.RunStatus.CANCELLED
        assert reached == []
    finally:
        gate.set()
        restore()


# ---------------------------------------------------------------------------
# Retention: completed runs only, horizon read from the evaluator's data.


def test_r26_retention_cleanup_follows_completed_runs_only(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    gate = threading.Event()
    entered = threading.Event()
    try:
        horizons = []

        class SpyStore(module.ResultsStore):
            def cleanup(self, retention_days=90):
                horizons.append(retention_days)
                return super().cleanup(retention_days)

        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        store=SpyStore(tmp_path / "runs.db"))
        runner.run_sync("p1", ["m1"], query_fn=_plain_answer)
        assert horizons == [1000]

        horizons.clear()
        runner.run_sync("absent", ["m1"], query_fn=_plain_answer)
        assert horizons == []

        def held_answer(model, prompt, timeout, max_tokens):
            entered.set()
            gate.wait(_DEADLINE)
            return ("held", 1.0, 10.0, 3)

        run_id = runner.start_run("p1", ["m1"], query_fn=held_answer)
        assert entered.wait(_DEADLINE)
        assert runner.cancel_run(run_id) is True
        gate.set()
        _settle(runner, module, run_id)
        assert runner.get_progress(run_id).status is module.RunStatus.CANCELLED
        assert horizons == []
    finally:
        gate.set()
        restore()


# ---------------------------------------------------------------------------
# The transport helper: one all-zero shape, first-content latency, exact
# counts, one cached client per timeout.


def test_r27_the_transport_returns_empty_shapes_when_the_client_is_gone(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        # Absent library: the only all-zero answer the helper ever gives.
        assert module._query_ollama("m", "p") == ("", 0.0, 0.0, 0)

        # A client that breaks mid-call keeps the elapsed time.
        with _seeded_at_call_time("ollama", _ollama_seed(_BreakingClient)):
            response, first_ms, total_ms, tokens = module._query_ollama(
                "m", "p", timeout=2,
            )
        module._OLLAMA_CLIENTS.clear()
        assert (response, first_ms, tokens) == ("", 0.0, 0)
        assert total_ms > 0.0

        # The chunk folders normalise both shapes and every missing form.
        assert module._chunk_message_text({"message": None}) == ""
        assert module._chunk_message_text({"message": {"content": None}}) == ""
        assert module._chunk_message_text(_ObjectChunk(content="x")) == "x"
        assert module._chunk_eval_count({"eval_count": "5"}) == 5
        assert module._chunk_eval_count({"eval_count": "x"}) == 0
        assert module._chunk_eval_count(SimpleNamespace()) == 0
    finally:
        restore()


def test_r28_the_stream_arm_measures_first_content_and_caches_per_timeout(tmp_path):
    _StreamClient.instances = []
    module, evaluator, restore = _load(tmp_path,
                                       ollama=_ollama_seed(_StreamClient))
    try:
        response, first_ms, total_ms, tokens = module._query_ollama(
            "m", "the prompt", timeout=9, max_tokens=33,
        )
        # The role-only preamble did not start the clock; the joined content
        # and the exact reported count did come through.
        assert response == "Hello"
        assert first_ms > 0.0
        assert total_ms >= first_ms
        assert tokens == 7
        assert _StreamClient.instances == [9]
        module._query_ollama("m", "the prompt", timeout=9)
        assert _StreamClient.instances == [9]
        module._query_ollama("m", "the prompt", timeout=4)
        assert _StreamClient.instances == [9, 4]
        # The prompt and the token budget reached the client untouched.
        client = module._OLLAMA_CLIENTS[9]
        model, messages, options = client.asked[0]
        assert model == "m"
        assert messages == [{"role": "user", "content": "the prompt"}]
        assert options == {"num_predict": 33}
    finally:
        restore()


# ---------------------------------------------------------------------------
# Evaluation edges: empty generations and code without a sandbox are zeroed
# without consulting any evaluator.


def test_r29_empty_responses_and_missing_sandbox_zero_without_evaluators(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        probe = evaluator.probe
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "a.db")
        result = runner.run_sync("p1", ["m1"],
                                 query_fn=lambda *a: ("", 0.0, 50.0, 0))
        assert (probe["accuracy"], probe["code"]) == (0, 0)
        assert (probe["structure"], probe["performance"]) == (0, 0)
        details = runner.store.get_run_details(result.run_id)
        for entry in details["question_results"]["m1"]:
            assert entry["composite_score"] == 0.0
            assert entry["accuracy_score"] == 0.0

        # A non-empty code answer without a sandbox is zeroed the same
        # silent way; only the plain question consults its evaluator.
        runner = module.BenchmarkRunner(evaluator=_profiled(evaluator),
                                        db_path=tmp_path / "b.db")
        result = runner.run_sync("p1", ["m1"], query_fn=_plain_answer)
        assert probe["code"] == 0
        assert probe["accuracy"] == 1
        scores = runner.store.get_run(result.run_id)["model_scores"]["m1"]
        assert scores["code_avg"] == 0.0
    finally:
        restore()


# ---------------------------------------------------------------------------
# The store on its own: attachment filtering, completed-only aggregation,
# child cleanup, and the duplication a re-save quietly performs.


def test_r30_the_store_filters_attachments_and_duplicates_children_on_resave(tmp_path):
    module, evaluator, restore = _load(tmp_path)
    try:
        store = module.ResultsStore(tmp_path / "history.db")
        now = time.time()
        first = module.RunResult(
            run_id="run-first", profile="pa", models=["m1"],
            status=module.RunStatus.COMPLETED,
            started_at=now - 300, finished_at=now - 200, duration_ms=5.0,
            model_scores={"m1": module.ModelScore(model="m1", composite=0.9)},
        )
        second = module.RunResult(
            run_id="run-second", profile="pb", models=["m2"],
            status=module.RunStatus.FAILED,
            started_at=now - 100, finished_at=now - 90, duration_ms=5.0,
            model_scores={"m2": module.ModelScore(model="m2", composite=0.1)},
        )
        third = module.RunResult(
            run_id="run-third", profile="pa", models=["m3"],
            status=module.RunStatus.COMPLETED,
            started_at=now - 50, finished_at=now - 40, duration_ms=5.0,
            model_scores={"m3": module.ModelScore(model="m3", composite=0.4)},
        )
        for result in (first, second, third):
            store.save_run(result)

        # One model's history: every run still answers, newest first, and
        # only the score attachment is narrowed to the asked-for model.
        history = store.get_history(model="m1")
        assert [entry["run_id"] for entry in history] == [
            "run-third", "run-second", "run-first",
        ]
        assert set(history[2]["model_scores"]) == {"m1"}
        assert history[0]["model_scores"] == {}
        assert history[1]["model_scores"] == {}

        # Aggregation sees completed runs only, best composite first.
        compared = store.compare_models()
        assert [entry["model"] for entry in compared["models"]] == ["m1", "m3"]

        # Expired runs fall with their children, and the count is reported.
        expired = module.RunResult(
            run_id="run-old", profile="pa", models=["m1"],
            status=module.RunStatus.COMPLETED,
            started_at=now - 200 * 86400, finished_at=now - 200 * 86400,
            duration_ms=5.0,
            model_scores={"m1": module.ModelScore(
                model="m1",
                question_results=[{
                    "question_id": "qa", "category": "general",
                    "prompt": "p", "response": "r", "details": {"kept": 1},
                }],
            )},
        )
        store.save_run(expired)
        connection = sqlite3.connect(str(tmp_path / "history.db"))
        try:
            count_children = (
                "SELECT COUNT(*) FROM benchmark_question_results"
                " WHERE run_id = 'run-old'"
            )
            assert connection.execute(count_children).fetchone()[0] == 1
            assert store.cleanup(90) == 1
            assert store.get_run("run-old") is None
            assert connection.execute(count_children).fetchone()[0] == 0

            # A re-save replaces the run row but plainly re-inserts the
            # children: the score rows double.
            store.save_run(first)
            count_scores = (
                "SELECT COUNT(*) FROM benchmark_model_scores"
                " WHERE run_id = 'run-first'"
            )
            assert connection.execute(count_scores).fetchone()[0] == 2
        finally:
            connection.close()
        assert store.get_run("run-first")["status"] == "completed"
    finally:
        restore()
