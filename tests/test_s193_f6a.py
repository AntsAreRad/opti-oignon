#!/usr/bin/env python3
"""
S193 F6a — benchmark core fixes (runner, evaluator, judge, routes).

Covers:
  - BJD-01: score_exact no longer matches empty / one-character answers
  - BJD-02: failed/empty generations score zero (performance + runner path)
  - BJD-03: composite renormalizes over the axes actually evaluated
  - BJD-04: an unparseable judge response is flagged as an error
  - BMK-01: both-form (dict/object) ollama chunk parsing in both query fns
  - BMK-02: per-call timeout enforced via a timeout-bound ollama.Client
  - BMK-03: exact eval_count from the final chunk preferred over chunk count
  - BMK-04: v2 run endpoint refuses concurrent runs (source assertion)
  - BMK-05: configured retention cleanup wired at run completion
  - BMK-06: v1 WS broadcast schedules puts on the client's event loop
"""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

_PROJECT = Path(__file__).resolve().parent.parent


def _load_module(name: str, rel_path: str):
    """Load a module directly from file path.

    Registers the module in sys.modules before exec_module (S192 loader
    idiom) so dataclass processing and intra-module lookups stay safe.
    """
    full = _PROJECT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(full))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_eval_mod = _load_module("s193_benchmark_evaluator", "opti_oignon/benchmark_evaluator.py")
_runner_mod = _load_module("s193_benchmark_runner", "opti_oignon/benchmark_runner.py")
_judge_mod = _load_module("s193_benchmark_judge", "opti_oignon/benchmark_judge.py")

score_exact = _eval_mod.score_exact
evaluate_performance = _eval_mod.evaluate_performance
compute_composite_score = _eval_mod.compute_composite_score
WeightPreset = _eval_mod.WeightPreset
BenchmarkEvaluator = _eval_mod.BenchmarkEvaluator


# ---------------------------------------------------------------------------
# Fake ollama module (dict-form and object-form chunks)
# ---------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChunk:
    """Object-form chunk: attribute access only, no .get()."""

    def __init__(self, content, eval_count=None):
        self.message = _FakeMessage(content)
        if eval_count is not None:
            self.eval_count = eval_count


def _make_fake_ollama(chunks, record):
    """Build a fake ollama module whose Client records its timeout."""
    fake = types.ModuleType("ollama")

    class Client:
        def __init__(self, **kwargs):
            record.append(kwargs)

        def chat(self, **kwargs):
            return iter(chunks)

        def generate(self, **kwargs):
            return {"response": "ok"}

    fake.Client = Client
    return fake


@pytest.fixture
def fake_ollama_env():
    """Inject a fake ollama and clear per-timeout client caches."""
    saved = sys.modules.get("ollama")

    def install(chunks):
        record = []
        sys.modules["ollama"] = _make_fake_ollama(chunks, record)
        _runner_mod._OLLAMA_CLIENTS.clear()
        _judge_mod._JUDGE_CLIENTS.clear()
        return record

    yield install
    if saved is not None:
        sys.modules["ollama"] = saved
    else:
        sys.modules.pop("ollama", None)
    _runner_mod._OLLAMA_CLIENTS.clear()
    _judge_mod._JUDGE_CLIENTS.clear()


# ---------------------------------------------------------------------------
# BJD-01 — exact scoring guards
# ---------------------------------------------------------------------------

class TestBJD01ExactGuards:
    def test_empty_response_scores_zero(self):
        score, matched = score_exact("", ["Paris"])
        assert score == 0.0
        assert matched == ""

    def test_one_char_answer_scores_zero(self):
        score, _ = score_exact("a", ["Paris"])
        assert score == 0.0

    def test_empty_expected_entry_never_matches(self):
        score, _ = score_exact("anything at all", [""])
        assert score == 0.0

    def test_legit_reverse_containment_still_matches(self):
        score, matched = score_exact("Paris", ["Paris, France"])
        assert score == 1.0
        assert matched == "Paris, France"

    def test_forward_containment_still_matches(self):
        score, _ = score_exact("The capital of France is Paris.", ["Paris"])
        assert score == 1.0

    def test_two_char_numeric_answer_still_matches(self):
        score, _ = score_exact("42", ["42 degrees"])
        assert score == 1.0


# ---------------------------------------------------------------------------
# BJD-02 — failed generation scores zero
# ---------------------------------------------------------------------------

class TestBJD02FailedGeneration:
    def test_tokenless_generation_scores_zero(self):
        result = evaluate_performance(
            ttft_ms=0.0, tokens_per_second=0.0, total_time_ms=120.0,
        )
        assert result.score == 0.0

    def test_real_generation_unaffected(self):
        result = evaluate_performance(
            ttft_ms=100, tokens_per_second=50, total_time_ms=2000,
        )
        assert result.score > 0.8


# ---------------------------------------------------------------------------
# BJD-03 — composite renormalization
# ---------------------------------------------------------------------------

class TestBJD03Renormalization:
    def test_no_code_axis_reaches_one(self):
        score = compute_composite_score(
            1.0, 0.0, 1.0, 1.0, WeightPreset(),
            evaluated={"accuracy", "structure", "speed"},
        )
        assert score == pytest.approx(1.0)

    def test_code_only_question_excludes_accuracy(self):
        score = compute_composite_score(
            0.0, 1.0, 1.0, 1.0, WeightPreset(),
            evaluated={"code", "structure", "speed"},
        )
        assert score == pytest.approx(1.0)

    def test_default_behaviour_unchanged_without_evaluated(self):
        score = compute_composite_score(1.0, 0.0, 1.0, 1.0, WeightPreset())
        assert score == pytest.approx(0.75)

    def test_all_axes_evaluated_matches_default(self):
        all_axes = {"accuracy", "code", "structure", "speed"}
        a = compute_composite_score(0.8, 0.6, 0.7, 0.9, WeightPreset())
        b = compute_composite_score(
            0.8, 0.6, 0.7, 0.9, WeightPreset(), evaluated=all_axes,
        )
        assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# Runner end-to-end: empty responses zeroed, non-code profile renormalized
# ---------------------------------------------------------------------------

_QUESTIONS_YAML = """
general_knowledge:
  - id: q1
    prompt: "Capital of France?"
    expected: ["Paris"]
    scoring: exact
"""

_PROFILES_YAML = """
profiles:
  tiny:
    name: Tiny
    categories: [general_knowledge]
    weight_preset: balanced
    timeout: 7
    expected_length_range: [1, 600]
weight_presets:
  balanced:
    accuracy: 0.35
    code: 0.25
    structure: 0.25
    speed: 0.15
runner:
  results_retention_days: 30
"""


def _make_runner(tmpdir):
    qpath = Path(tmpdir) / "questions.yaml"
    ppath = Path(tmpdir) / "profiles.yaml"
    qpath.write_text(_QUESTIONS_YAML, encoding="utf-8")
    ppath.write_text(_PROFILES_YAML, encoding="utf-8")
    evaluator = BenchmarkEvaluator(questions_path=qpath, profiles_path=ppath)
    store = _runner_mod.ResultsStore(os.path.join(tmpdir, "bench.db"))
    return _runner_mod.BenchmarkRunner(evaluator=evaluator, store=store)


class TestRunnerIntegration:
    def test_empty_response_run_scores_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            result = runner.run_sync(
                "tiny", ["m1"],
                query_fn=lambda m, p, t, mt: ("", 0.0, 100.0, 0),
            )
            assert result.status == _runner_mod.RunStatus.COMPLETED
            stored = runner.get_results(result.run_id)
            ms = stored["model_scores"]["m1"]
            assert ms["composite"] == pytest.approx(0.0)
            assert ms["accuracy_avg"] == pytest.approx(0.0)
            assert ms["speed_avg"] == pytest.approx(0.0)

    def test_perfect_answer_non_code_profile_reaches_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            result = runner.run_sync(
                "tiny", ["m1"],
                # 50 tokens over 1000 ms -> 50 tok/s -> speed axis 1.0
                query_fn=lambda m, p, t, mt: ("Paris", 100.0, 1000.0, 50),
            )
            stored = runner.get_results(result.run_id)
            ms = stored["model_scores"]["m1"]
            # accuracy 1.0, structure 1.0, speed 1.0; code axis excluded
            # by renormalization (was capped at 0.75 before S193).
            assert ms["composite"] == pytest.approx(1.0)

    def test_retention_cleanup_called_with_configured_days(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _make_runner(tmpdir)
            calls = []
            original = runner._store.cleanup
            runner._store.cleanup = lambda days: calls.append(days) or original(days)
            runner.run_sync(
                "tiny", ["m1"],
                query_fn=lambda m, p, t, mt: ("Paris", 100.0, 2000.0, 50),
            )
            assert calls == [30]


# ---------------------------------------------------------------------------
# BMK-01 / BMK-02 / BMK-03 — query functions
# ---------------------------------------------------------------------------

class TestQueryOllama:
    def test_object_form_chunks_parsed(self, fake_ollama_env):
        fake_ollama_env([
            _FakeChunk("Hel"),
            _FakeChunk("lo"),
            _FakeChunk("", eval_count=17),
        ])
        text, ttft, total, tokens = _runner_mod._query_ollama("m", "p", 7, 100)
        assert text == "Hello"
        assert tokens == 17  # exact eval_count preferred (BMK-03)
        assert ttft > 0.0

    def test_dict_form_chunks_still_parsed(self, fake_ollama_env):
        fake_ollama_env([
            {"message": {"content": "Hi"}},
            {"message": {"content": ""}, "eval_count": 5},
        ])
        text, _, _, tokens = _runner_mod._query_ollama("m", "p", 7, 100)
        assert text == "Hi"
        assert tokens == 5

    def test_timeout_bound_client_used(self, fake_ollama_env):
        record = fake_ollama_env([_FakeChunk("x")])
        _runner_mod._query_ollama("m", "p", 13, 100)
        assert record and record[0].get("timeout") == 13

    def test_judge_object_form_and_timeout(self, fake_ollama_env):
        record = fake_ollama_env([_FakeChunk('{"accuracy": 8}')])
        text, _, _, _ = _judge_mod._query_judge("j", "p", 21, 100)
        assert text == '{"accuracy": 8}'
        assert record and record[0].get("timeout") == 21


# ---------------------------------------------------------------------------
# BJD-04 — unparseable judge output is an error, not a zero score
# ---------------------------------------------------------------------------

class TestBJD04JudgeParseFailure:
    def _judge(self, tmpdir):
        store = _judge_mod.JudgeStore(os.path.join(tmpdir, "judge.db"))
        return _judge_mod.BenchmarkJudge(store=store)

    def test_unparseable_response_sets_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            judge = self._judge(tmpdir)
            js = judge.evaluate(
                "q1", "question", "response", "m1", "j1",
                query_fn=lambda m, p, t, mt: (
                    "no rubric in this text at all", 1.0, 10.0, 3,
                ),
            )
            assert js.error != ""
            assert js.weighted_score == 0.0

    def test_valid_json_still_clean(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            judge = self._judge(tmpdir)
            payload = (
                '{"accuracy": 8, "relevance": 7, "completeness": 6, '
                '"conciseness": 9, "reasoning": 8, "justification": "ok"}'
            )
            js = judge.evaluate(
                "q1", "question", "response", "m1", "j1",
                query_fn=lambda m, p, t, mt: (payload, 1.0, 10.0, 3),
            )
            assert js.error == ""
            assert js.weighted_score > 0.0

    def test_unparseable_excluded_from_run_averages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            judge = self._judge(tmpdir)
            summary = judge.evaluate_run(
                "run-x", "j1",
                [
                    {"question_id": "q1", "question_text": "q",
                     "response": "r", "model": "m1"},
                ],
                query_fn=lambda m, p, t, mt: ("garbage text", 1.0, 10.0, 3),
            )
            assert summary.errors == 1
            assert summary.scores_by_model == {}


# ---------------------------------------------------------------------------
# BMK-04 / BMK-06 — route-level fixes (source + AST assertions; the route
# import chain is heavy, per the S192 convention)
# ---------------------------------------------------------------------------

class TestRouteSourceAssertions:
    def test_v2_run_has_concurrency_guard(self):
        src = (_PROJECT / "opti_oignon/api/routes_benchmark_v2.py").read_text()
        assert "S193 BMK-04" in src
        assert 'getattr(benchmark_runner, "is_busy", False)' in src
        assert "status_code=409" in src

    def test_v1_broadcast_is_thread_safe(self):
        src = (_PROJECT / "opti_oignon/api/routes_benchmark.py").read_text()
        assert "S193 BMK-06" in src
        assert "call_soon_threadsafe" in src
        assert "add_ws_client(queue, asyncio.get_running_loop())" in src

    def test_v1_generate_uses_timeout_client(self):
        src = (_PROJECT / "opti_oignon/api/routes_benchmark.py").read_text()
        assert "_get_v1_client(timeout).generate(" in src
        assert "_ollama.Client(timeout=timeout)" in src

    def test_route_files_parse(self):
        import ast
        for rel in (
            "opti_oignon/api/routes_benchmark.py",
            "opti_oignon/api/routes_benchmark_v2.py",
        ):
            ast.parse((_PROJECT / rel).read_text())


class TestRunStateBehaviour:
    """Behavioural check of the patched _RunState via direct load."""

    def _load_routes(self):
        return _load_module(
            "s193_routes_benchmark", "opti_oignon/api/routes_benchmark.py",
        )

    def test_broadcast_uses_client_loop(self):
        routes = self._load_routes()
        state = routes._RunState()

        class FakeQueue:
            def __init__(self):
                self.items = []

            def put_nowait(self, item):
                self.items.append(item)

        class FakeLoop:
            def __init__(self):
                self.scheduled = 0

            def call_soon_threadsafe(self, fn, *args):
                self.scheduled += 1
                fn(*args)

        q, loop = FakeQueue(), FakeLoop()
        state.add_ws_client(q, loop)
        state.broadcast({"type": "t"})
        assert loop.scheduled == 1
        assert q.items == [{"type": "t"}]
        state.remove_ws_client(q)
        state.broadcast({"type": "u"})
        assert q.items == [{"type": "t"}]

    def test_broadcast_without_loop_falls_back(self):
        routes = self._load_routes()
        state = routes._RunState()

        class FakeQueue:
            def __init__(self):
                self.items = []

            def put_nowait(self, item):
                self.items.append(item)

        q = FakeQueue()
        state.add_ws_client(q)
        state.broadcast({"type": "direct"})
        assert q.items == [{"type": "direct"}]
