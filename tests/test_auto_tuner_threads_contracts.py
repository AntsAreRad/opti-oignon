#!/usr/bin/env python3
"""The tuner measures the CPU thread count against the plan, and the plan reads what held.

The auto-tuner swept a fixed list of thread counts (2, 4, 6, 8) whatever the
machine, from a baseline in the middle of that list. Every trial reached the
engine without a ticket of its own, so the engine's backstop admitted it as a
user call -- one that may evict the user's models -- and the tuner never knew
which engine served a trial nor where the model computed. Its best count went
to a results file nothing planned from.

  The sweep:
    * AT4 -- "auto", the shipped default, sweeps the counts the governor's
      plan gives the machine: the plan, each fraction of it rounded up, and
      the fastest class's core count, deduplicated and never past the plan;
      a list in the file is swept as written, and a setting that cannot be
      read falls back to "auto". Superseded by AT12: the server's own quota
      no longer bounds the plan of an engine in a process of its own.
    * AT5 -- the baseline runs at the plan's count, so the gain is told
      against what the plan would do; without a plan, "auto" sends no count
      and a list keeps its middle.

  The trials:
    * AT6 -- each trial asks its own "tuner" ticket naming its engine, holds
      it around the engine call and sends the context it was admitted at; a
      held trial runs nothing and fails by the governor's reason; a refusal
      no wait can lift, or a baseline that could not be measured, ends the
      run.
    * AT7 -- each result says the engine that served it, where the model
      computed, and whether the engine applied the count asked.

  What a sweep leaves behind:
    * AT8 -- the best count is kept only if it holds: its confirmation ran,
      every thread trial was measured, by an engine that applies a count per
      call, on one placement known exactly, at no more than the plan, and no
      slower than the baseline.
    * AT9 -- when no count is kept, one last trial runs at the plan's count,
      so the model stays at the count its next decisions carry; when one is
      kept, the last trial ran at it.
    * AT10 -- the manager hands a kept count to the governor, which writes it
      with the machine's fingerprint and plans from it; a run that kept none
      hands nothing.
    * AT11 -- the status route serves the "auto" space, and the profile the
      API returns carries the kept count, or why none was kept.
    * AT12 -- AT4 again: the sweep follows the machine's plan whatever the
      server's quota, and never passes the plan.
    * AT13 -- a run that ends early after a thread trial (cancelled, or
      refused) leaves the resident pinned at the plan's count, not the
      trial's.
    * AT14 -- the pin follows the count the resident holds, not the last
      trial tried: trials the governor holds never reach the engine, and a
      kept count wins over the plan.
    * AT15 -- a run that changed no pin changes none, and the warm-up names
      the engine as a trial does.
    * AT16 -- a run whose trials reached no engine re-pins nothing, whatever
      another load did to the pin meanwhile.
    * AT17 -- the pins a run reads and restores live in a running governor:
      with none running, the run starts none.

Everything here is proven in the container: the tuner and the governor in the
shared isolation window, a scripted CPU topology, hand-set snapshots, and
engines that record what they are sent. Which count makes an engine fastest
on a given machine is owed to the machine.
"""

import shutil
import sqlite3
import sys
import tempfile
import threading
import types
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_RG = "opti_oignon.resource_governor"
_TUNER = "opti_oignon.auto_tuner"
# The two seams the governor resolves at the call: unreachable when the window
# opens, so neither an emergency stop nor a model window enters a decision.
_SEAMS = ("opti_oignon.context_manager", "opti_oignon.emergency_stop")
_CLOSERS = []
# Twelve cores, an eighth reserved rounded up (two): a plan of ten. Half of it
# is five, three quarters 7.5, rounded up to eight, and four cores are fast.
_PLAN = (10, [4, 5, 8, 10], "plan")
# The thread count is the only knob that moves the speed here: eight is the
# fastest, the plan's ten the baseline.
_SPEEDS = {4: 9.0, 5: 11.0, 8: 12.0, 10: 10.0}


def _db_utils():
    """A db_utils stand-in whose safe_connect is plain sqlite."""
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda p, **kw: sqlite3.connect(
        str(p), check_same_thread=kw.get("check_same_thread", False)
    )
    return db


def _open():
    """The governor and the tuner in one window: (governor module, tuner module)."""
    loaded, restore = isolate(
        targets={_RG: source("resource_governor.py"), _TUNER: source("auto_tuner.py")},
        blocked=_SEAMS,
        seeded={"opti_oignon.db_utils": _db_utils()},
    )
    _CLOSERS.append(restore)
    return loaded[_RG], loaded[_TUNER]


def _project_modules():
    """Every project entry of the module cache, by identity."""
    return {k: v for k, v in sys.modules.items() if k == "opti_oignon" or k.startswith("opti_oignon.")}


@pytest.fixture(autouse=True)
def _left_as_found():
    """No contract may leave a project module changed."""
    before = _project_modules()
    yield
    while _CLOSERS:
        _CLOSERS.pop()()
    assert _project_modules() == before, "every project module is left as the contract found it"


class _Clock:
    """A fixed monotonic stand-in: a hand-set snapshot never goes stale."""

    def __call__(self) -> float:
        return 1000.0


def _topology(physical=12, *, fast=4, quota=None):
    """A CPU topology as the governor reads it: two SMT siblings per core, the
    first ``fast`` cores in class 0, and the cgroup quota."""
    cores = tuple(
        types.SimpleNamespace(cpus=(n, n + physical), rank=None, perf_class=0 if n < fast else 1)
        for n in range(physical)
    )
    return types.SimpleNamespace(
        usable=tuple(sorted(cpu for core in cores for cpu in core.cpus)),
        cores=cores,
        physical=physical,
        smt=True,
        l3=(),
        numa=(),
        quota_cpus=quota,
        class_source="acpi_cppc",
    )


class _Hardware:
    """A hardware profile that answers the CPUs; its cards read nothing."""

    def __init__(self, topology=None, *, cards_absent=False):
        self.topology = topology
        self._absent = cards_absent

    def cpu_topology(self):
        return self.topology

    def cards_absent(self):
        return self._absent

    def placement(self):
        return None

    def pressure(self):
        return None

    def others_cpu_pressure(self):
        return None


def _governor(rg, *, hardware, weights=None, capacity=10.0, db=None):
    tmp = Path(tempfile.mkdtemp(prefix="at-"))
    _CLOSERS.append(lambda: shutil.rmtree(tmp, ignore_errors=True))
    meminfo = tmp / "meminfo"
    meminfo.write_text("MemTotal:       65536000 kB\nMemAvailable:   65536000 kB\n", encoding="utf-8")
    gov = rg.ResourceGovernor(
        config_path=str(tmp / "missing.yaml"),
        db_path=str(db or tmp / "governor.db"),
        warmup=None,
        registry=None,
        clock=_Clock(),
        meminfo_path=str(meminfo),
        vram_probe=None,
        hardware=hardware,
    )
    cfg = rg.GovernorConfig(
        total_vram_gb=capacity,
        safety_margin_gb=1.5,
        kv_coefficient=0.5,
        ctx_ladder=[8192, 4096, 2048],
        ctx_floor={"chat": 2048},
    )
    cfg.weights_override_models = dict(weights or {})
    gov._config = cfg
    return gov


def _no_card_snapshot(rg, gov):
    """A machine with no card: the CPU computes every model, from 62.5 GiB free."""
    gov._snapshot = rg.ResourceSnapshot(
        taken_at=1000.0,
        ttl_s=9999.0,
        loaded=[],
        capacity_gb=None,
        vram_in_use_gb=0.0,
        ram_available_mb=64000.0,
        cards_absent=True,
    )


class _Engine:
    """An engine as the tuner's trials ask it: every call recorded with the
    ticket the calling thread held, and counters that always report."""

    def __init__(self, rg, name="ollama", threads_per_call=True):
        self._rg = rg
        self.name = name
        self.threads_per_call = threads_per_call
        self.calls = []

    def generate(self, model=None, messages=None, options=None, **kwargs):
        self.calls.append({"model": model, "options": dict(options or {}), "ticket": self._rg.get_active_ticket()})
        return types.SimpleNamespace(
            content="a reply long enough to count",
            extra={
                "eval_count": 64,
                "eval_duration": 2_000_000_000,
                "prompt_eval_count": 16,
                "prompt_eval_duration": 250_000_000,
                "timings": {"predicted_per_second": 32.0, "prompt_per_second": 64.0},
            },
        )


class _Scripted:
    """A governor that answers each admission from a script and records who asked."""

    def __init__(self, *decisions):
        self.asked = []
        self._script = list(decisions)

    def admit_or_wait(
        self, model, requested_ctx=None, caller="benchmark", extra_models=None, digest=None, wait_s=None, engine=None
    ):
        self.asked.append((model, caller, engine))
        return self._script.pop(0) if len(self._script) > 1 else self._script[0]

    def note_held(self, decision):
        pass


class _Bench:
    """A benchmark whose speed is its thread count's (``speeds``), labelled
    as the trial says: ``engine``, ``placement``, ``source`` and ``applied``
    are values or functions of the count. ``later`` replaces a count's speed
    on every run after its first (its confirmation); ``fail`` makes those
    later runs errors. Every call is recorded."""

    def __init__(self, tuner, speeds, *, engine="ollama", placement="split:20", source="measured", applied=True,
                 later=None, fail=()):
        self._tuner = tuner
        self._speeds = dict(speeds)
        self._labels = {"engine": engine, "placement": placement, "source": source, "applied": applied}
        self._later = dict(later or {})
        self._fail = set(fail)
        self._runs = {}
        self.calls = []

    def _label(self, name, threads):
        value = self._labels[name]
        return value(threads) if callable(value) else value

    def __call__(self, params):
        tuner = self._tuner
        key = tuner._param_key(params)
        seen = self._runs.get(key, 0)
        self._runs[key] = seen + 1
        self.calls.append(dict(params))
        threads = params.get("threads")
        if seen and threads in self._fail:
            return tuner.BenchmarkResult(params=params, error="the engine stopped")
        speed = self._later[threads] if seen and threads in self._later else self._speeds[threads]
        return tuner.BenchmarkResult(
            params=params,
            tokens_per_second_tg=speed,
            tokens_per_second_pp=speed * 2,
            total_time_ms=1.0,
            source=self._label("source", threads),
            engine=self._label("engine", threads),
            placement=self._label("placement", threads),
            threads_applied=self._label("applied", threads),
        )


def _tuner(tuner, bench, *, threads="auto", plan=_PLAN):
    space = tuner.ParameterSpace.from_dict({"threads": threads})
    return tuner.AutoTuner(
        config=tuner.TunerConfig(warmup_runs=0, trials_per_param=1),
        param_space=space,
        benchmark_fn=bench,
        thread_plan=(lambda: plan) if plan is not None else None,
    )


def _thread_axis(sweep, defaults):
    """The thread counts of the points that leave every other knob at its default."""
    others = {k: v for k, v in defaults.items() if k != "threads"}
    return {p.get("threads") for p in sweep if {k: v for k, v in p.items() if k != "threads"} == others}


# ---------------------------------------------------------------------------
# AT4-AT5 -- the sweep
# ---------------------------------------------------------------------------


def test_at4_auto_sweeps_the_plan_its_fractions_and_the_fast_class_never_past_the_plan(caplog):
    """AT4 -- twelve cores of which four are fast, two reserved: the plan is
    ten, and "auto" sweeps 4, 5, 8 and 10. Under a quota of six CPUs the plan
    is six: 3, 4, 5, 6. On one class of cores the fast count is every core,
    held to the plan: 5, 8, 10. The shipped file says "auto" with the
    fractions one half and three quarters, and a missing or unreadable
    setting reads as "auto"; a list is swept as written. Sweeping a fixed
    list, or a count past the plan (the reserve, the quota) -> RED."""
    rg, tuner = _open()
    gov = _governor(rg, hardware=_Hardware(_topology(12, fast=4)))
    assert gov.thread_candidates("m", [0.5, 0.75]) == _PLAN
    quota = _governor(rg, hardware=_Hardware(_topology(12, fast=4, quota=6.0)))
    assert quota.thread_candidates("m", [0.5, 0.75]) == (6, [3, 4, 5, 6], "plan")
    uniform = _governor(rg, hardware=_Hardware(_topology(12, fast=12)))
    assert uniform.thread_candidates("m", [0.5, 0.75]) == (10, [5, 8, 10], "plan")

    auto = _tuner(tuner, _Bench(tuner, _SPEEDS))
    assert _thread_axis(auto._build_smart_sweep(), auto._default_params()) == {4, 5, 8, 10}
    listed = _tuner(tuner, _Bench(tuner, _SPEEDS), threads=[3, 7, 16])
    assert _thread_axis(listed._build_smart_sweep(), listed._default_params()) == {3, 7, 16, 10}

    shipped = yaml.safe_load(Path(source("config", "auto_tuner.yaml")).read_text(encoding="utf-8"))
    block = shipped["parameter_space"]
    assert (block["threads"], block["threads_fractions"]) == ("auto", [0.5, 0.75])
    space = tuner.ParameterSpace.from_dict(block)
    assert (space.threads_auto, space.threads_fractions) == (True, [0.5, 0.75])
    assert space.to_dict()["threads"] == "auto"
    assert tuner.ParameterSpace.from_dict({}).threads_auto is True
    unreadable = tuner.ParameterSpace.from_dict({"threads": "many", "threads_fractions": [1.5, "x", True]})
    assert (unreadable.threads_auto, unreadable.threads_fractions) == (True, [0.5, 0.75])
    assert "threads" in caplog.text
    cleaned = tuner.ParameterSpace.from_dict({"threads": [0, "x", 3, 3, True, 6]})
    assert (cleaned.threads_auto, cleaned.threads) == (False, [3, 6])


def test_at5_the_baseline_runs_at_the_plans_count():
    """AT5 -- the warm-up and the baseline run at the plan's count (ten),
    for "auto" and for a list alike, not at the list's middle. Without a
    plan (the CPUs unreadable), "auto" sends no count at all and the engine
    picks its own, while a list keeps its middle as before. A baseline at
    the middle of a list -> RED."""
    _, tuner = _open()
    for threads in ("auto", [2, 4, 6, 8]):
        bench = _Bench(tuner, {**_SPEEDS, 2: 8.0, 6: 9.5})
        run = tuner.AutoTuner(
            config=tuner.TunerConfig(warmup_runs=1, trials_per_param=1),
            param_space=tuner.ParameterSpace.from_dict({"threads": threads}),
            benchmark_fn=bench,
            thread_plan=lambda: _PLAN,
        )
        profile = run.run("m", tuner.TunerJob())
        assert [bench.calls[0]["threads"], bench.calls[1]["threads"]] == [10, 10], threads
        assert profile.all_results[0]["params"]["threads"] == 10, threads

    unplanned = _tuner(tuner, _Bench(tuner, _SPEEDS), plan=(None, [], "unknown"))
    assert "threads" not in unplanned._default_params()
    assert _thread_axis(unplanned._build_smart_sweep(), unplanned._default_params()) == {None}
    middle = _tuner(tuner, _Bench(tuner, _SPEEDS), threads=[2, 4, 6, 8], plan=None)
    assert middle._default_params()["threads"] == 6


# ---------------------------------------------------------------------------
# AT6-AT7 -- the trials
# ---------------------------------------------------------------------------


def test_at6_each_trial_asks_its_own_tuner_ticket_holds_it_and_abides_by_it():
    """AT6 -- an Ollama trial asks the governor as "tuner" naming "ollama",
    a llama.cpp trial naming "llama_cpp"; the engine is called with that
    ticket held and the context it was admitted at; no ticket is left behind.
    A held trial calls no engine and fails by the governor's reason. A
    refusal no wait can lift ends the run, failed, by that reason, and so
    does a baseline the governor held. A trial reaching the engine through
    its user backstop, or a sweep run on regardless -> RED."""
    rg, tuner = _open()
    admitted = rg.AdmissionDecision(admitted=True, model="m", num_ctx=4096, caller="tuner", placement="cpu")
    asked = _Scripted(admitted)
    rg._governor = asked
    ollama, llama = _Engine(rg), _Engine(rg, name="llama_cpp", threads_per_call=False)
    tuner.create_ollama_benchmark_fn("m", backend=ollama)({"threads": 6})
    tuner.create_llamacpp_benchmark_fn("m", backend=llama)({"threads": 6})
    assert asked.asked == [("m", "tuner", "ollama"), ("m", "tuner", "llama_cpp")]
    assert [call["ticket"] for call in ollama.calls + llama.calls] == [admitted, admitted]
    assert [call["options"]["num_ctx"] for call in ollama.calls + llama.calls] == [4096, 4096]
    assert rg.get_active_ticket() is None

    held = rg.AdmissionDecision(admitted=False, model="m", caller="tuner", reason="background_gate_interactive")
    rg._governor = _Scripted(held)
    quiet = _Engine(rg)
    result = tuner.create_ollama_benchmark_fn("m", backend=quiet)({"threads": 6})
    assert quiet.calls == [] and "background_gate_interactive" in result.error

    for decision in (
        rg.AdmissionDecision(admitted=False, model="m", caller="tuner", reason="background_cost_unknown"),
        held,
    ):
        rg._governor = _Scripted(decision)
        run = tuner.AutoTuner(
            config=tuner.TunerConfig(warmup_runs=0, trials_per_param=1),
            param_space=tuner.ParameterSpace(),
            benchmark_fn=tuner.create_ollama_benchmark_fn("m", backend=_Engine(rg)),
        )
        job = tuner.TunerJob()
        with pytest.raises(RuntimeError):
            run.run("m", job)
        assert job.status == "failed" and decision.reason in job.error


def test_at7_each_result_says_its_engine_where_the_model_computed_and_whether_the_count_applied():
    """AT7 -- through the real governor on a machine with no card, an Ollama
    trial says "ollama", "cpu", and that the count applied; the next trial,
    joining the load the first admitted, says "cpu" too; a llama.cpp trial
    says "llama_cpp", "cpu", and that its count did not apply (llama.cpp
    fixes it at the load); the serialised result carries all three. A result
    that cannot say who served it or where -> RED."""
    rg, tuner = _open()
    gov = _governor(rg, hardware=_Hardware(_topology(12), cards_absent=True), weights={"m": 4.0}, capacity=None)
    _no_card_snapshot(rg, gov)
    rg._governor = gov
    ollama = tuner.create_ollama_benchmark_fn("m", backend=_Engine(rg))
    first, second = ollama({"threads": 5}), ollama({"threads": 8})
    llama = tuner.create_llamacpp_benchmark_fn("m", backend=_Engine(rg, name="llama_cpp", threads_per_call=False))
    third = llama({"threads": 5})
    seen = [(r.error, r.engine, r.placement, r.threads_applied) for r in (first, second, third)]
    assert seen == [("", "ollama", "cpu", True), ("", "ollama", "cpu", True), ("", "llama_cpp", "cpu", False)]
    carried = first.to_dict()
    assert (carried["engine"], carried["placement"], carried["threads_applied"]) == ("ollama", "cpu", True)


# ---------------------------------------------------------------------------
# AT8-AT10 -- what a sweep leaves behind
# ---------------------------------------------------------------------------


def test_at8_the_best_count_is_kept_only_if_it_holds():
    """AT8 -- eight threads beat the plan's ten (12 tokens per second against
    10) and the kept count says so, with the engine, the placement and both
    rates. One condition broken at a time keeps nothing and says why: the
    confirmation fails; a thread trial is only estimated; the engine does
    not apply a count per call; the placement is unknown; the placement
    changed during the sweep; the best is past the plan; the confirmation
    is slower than the baseline. Keeping a count that did not hold -> RED."""
    _, tuner = _open()
    kept = _tuner(tuner, _Bench(tuner, _SPEEDS)).run("m", tuner.TunerJob())
    assert kept.threads_optimum == {
        "engine": "ollama",
        "placement": "split:20",
        "threads": 8,
        "threads_batch": 8,
        "tg": 12.0,
        "base_tg": 10.0,
    }
    assert kept.to_dict()["threads_optimum"] == kept.threads_optimum
    broken = {
        "confirmation": _Bench(tuner, _SPEEDS, fail={8}),
        "not measured": _Bench(tuner, _SPEEDS, source=lambda t: "estimated" if t == 5 else "measured"),
        "per call": _Bench(tuner, _SPEEDS, applied=False),
        "placement unknown": _Bench(tuner, _SPEEDS, placement=None),
        "placement changed during the sweep": _Bench(
            tuner, _SPEEDS, placement=lambda t: "split:24" if t == 5 else "split:20"
        ),
        "slower than the baseline": _Bench(tuner, _SPEEDS, later={8: 9.0}),
    }
    for why, bench in broken.items():
        profile = _tuner(tuner, bench).run("m", tuner.TunerJob())
        assert (profile.threads_optimum, why in profile.threads_note) == (None, True), why
    past = _tuner(tuner, _Bench(tuner, {**_SPEEDS, 12: 15.0}), threads=[4, 12]).run("m", tuner.TunerJob())
    assert (past.threads_optimum, "past the plan" in past.threads_note) == (None, True)


def test_at9_a_sweep_leaves_the_model_at_the_count_its_next_decisions_carry():
    """AT9 -- when the count is kept, the last trial ran at it (eight); when
    the confirmation does not hold, one more trial runs at the plan's count
    (ten), with every other knob at its default, so the model is not left
    at a count nobody plans. Ending the sweep on the rejected count -> RED."""
    _, tuner = _open()
    kept = _Bench(tuner, _SPEEDS)
    _tuner(tuner, kept).run("m", tuner.TunerJob())
    assert kept.calls[-1]["threads"] == 8
    for bench in (_Bench(tuner, _SPEEDS, later={8: 9.0}), _Bench(tuner, _SPEEDS, fail={8})):
        run = _tuner(tuner, bench)
        run.run("m", tuner.TunerJob())
        assert bench.calls[-1] == run._default_params()
        assert bench.calls[-1]["threads"] == 10


def _finish(model):
    """Wait for the manager's thread for ``model`` to end."""
    for thread in threading.enumerate():
        if thread.name == f"tuner-{model}":
            thread.join(timeout=30)
            assert not thread.is_alive(), "the tuning thread ends"


def test_at10_the_manager_hands_a_kept_count_to_the_governor_which_plans_from_it(tmp_path):
    """AT10 -- a run through the manager, with "auto" in its file, sweeps
    the governor's candidates and hands the kept count to it: the store
    holds eight with both rates and this machine's fingerprint, and the plan
    for that model, engine and placement is eight, said "measured". A run
    whose confirmation does not hold hands nothing, and that model keeps
    the plan. A kept count left in the results file alone -> RED."""
    rg, tuner = _open()
    gov = _governor(rg, hardware=_Hardware(_topology(12)))
    rg._governor = gov
    tuner._RESULTS_PATH = tmp_path / "tuner_results.json"
    config = tmp_path / "auto_tuner.yaml"
    config.write_text(
        "auto_tuner:\n  warmup_runs: 0\n  trials_per_param: 1\nparameter_space:\n  threads: auto\n", encoding="utf-8"
    )
    manager = tuner.AutoTunerManager(config_path=str(config))
    manager.start_tuning("m", _Bench(tuner, _SPEEDS))
    _finish("m")
    assert manager.get_job("m").status == "completed"
    row = gov.store.get_thread_optimum("m", "ollama", "split:20")
    assert (row["threads"], row["threads_batch"], row["tg"], row["base_tg"]) == (8, 8, 12.0, 10.0)
    assert row["fingerprint"] == gov.cpu_fingerprint() and row["fingerprint"]
    assert gov.plan_threads("m", "ollama", "split:20") == (8, 8, "measured")

    manager.start_tuning("n", _Bench(tuner, _SPEEDS, later={8: 9.0}))
    _finish("n")
    assert manager.get_job("n").status == "completed"
    assert gov.store.get_thread_optimum("n", "ollama", "split:20") is None
    assert gov.plan_threads("n", "ollama", "split:20") == (10, 10, "plan")


def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_at11_the_status_route_serves_auto_and_a_profile_carries_the_kept_count(tmp_path):
    """AT11 -- with the shipped file, the tuner's status route answers
    "auto" and its fractions, where the schema's list of numbers refused the
    word and the route failed; the profile schema carries the count a run
    kept, and, when none was kept, why. A status that cannot say "auto", or
    a kept count the API drops -> RED."""
    holder = {}
    deps = _stub(
        "opti_oignon.api.deps",
        AUTO_TUNER_AVAILABLE=True,
        INFERENCE_BACKEND_AVAILABLE=False,
        SPECULATIVE_DECODING_AVAILABLE=False,
        get_auto_tuner_manager=lambda: holder["manager"],
        get_speculative_decoding_manager=lambda: None,
        get_backend_registry=lambda: None,
    )
    loaded, restore = isolate(
        targets={
            _TUNER: source("auto_tuner.py"),
            "opti_oignon.api.schemas": source("api", "schemas.py"),
            "opti_oignon.api.routes_tuner": source("api", "routes_tuner.py"),
        },
        blocked=("requests",),
        seeded={
            "opti_oignon.inference_backend": _stub("opti_oignon.inference_backend", get_backend_registry=lambda: None),
            "opti_oignon.api.deps": deps,
        },
        packages=("opti_oignon.api",),
    )
    _CLOSERS.append(restore)
    tuner, schemas = loaded[_TUNER], loaded["opti_oignon.api.schemas"]
    tuner._RESULTS_PATH = tmp_path / "tuner_results.json"
    holder["manager"] = tuner.AutoTunerManager(config_path=source("config", "auto_tuner.yaml"))
    status = loaded["opti_oignon.api.routes_tuner"].get_tuner_status()
    assert (status.param_space.threads, status.param_space.threads_fractions) == ("auto", [0.5, 0.75])
    kept = {"engine": "ollama", "placement": "cpu", "threads": 8, "threads_batch": 8, "tg": 12.0, "base_tg": 10.0}
    served = schemas.TunerProfileSchema(**tuner.TunerProfile(model_name="m", threads_optimum=kept).to_dict())
    assert (served.threads_optimum, served.threads_note) == (kept, "")
    why = "the best count is past the plan"
    noted = schemas.TunerProfileSchema(**tuner.TunerProfile(model_name="n", threads_note=why).to_dict())
    assert (noted.threads_optimum, noted.threads_note) == (None, why)


# ---------------------------------------------------------------------------
# AT12-AT13 -- the machine's plan, and a run that ends early
# ---------------------------------------------------------------------------


def test_at12_auto_sweeps_the_machines_plan_whatever_the_servers_quota_and_never_past_it(caplog):
    """AT12 -- supersedes AT4, whose plan was held to the server's quota: the
    sweep measures an engine in a process of its own, planned on the
    machine's cores. Twelve cores of which four are fast, two reserved: the
    plan is ten, and "auto" sweeps 4, 5, 8 and 10, under a server quota of
    six CPUs as well; with half the cores reserved the plan is six: 3, 4, 5,
    6. On one class of cores the fast count is every core, held to the plan:
    5, 8, 10. The shipped file says "auto" with the fractions one half and
    three quarters, and a missing or unreadable setting reads as "auto"; a
    list is swept as written. Sweeping a fixed list, or a count past the
    plan (the reserve) -> RED."""
    rg, tuner = _open()
    gov = _governor(rg, hardware=_Hardware(_topology(12, fast=4)))
    assert gov.thread_candidates("m", [0.5, 0.75]) == _PLAN
    quota = _governor(rg, hardware=_Hardware(_topology(12, fast=4, quota=6.0)))
    assert quota.thread_candidates("m", [0.5, 0.75]) == _PLAN
    halved = _governor(rg, hardware=_Hardware(_topology(12, fast=4)))
    halved._config.threads_reserve_fraction, halved._config.threads_reserve_ceiling = 0.5, 8
    assert halved.thread_candidates("m", [0.5, 0.75]) == (6, [3, 4, 5, 6], "plan")
    uniform = _governor(rg, hardware=_Hardware(_topology(12, fast=12)))
    assert uniform.thread_candidates("m", [0.5, 0.75]) == (10, [5, 8, 10], "plan")

    auto = _tuner(tuner, _Bench(tuner, _SPEEDS))
    assert _thread_axis(auto._build_smart_sweep(), auto._default_params()) == {4, 5, 8, 10}
    listed = _tuner(tuner, _Bench(tuner, _SPEEDS), threads=[3, 7, 16])
    assert _thread_axis(listed._build_smart_sweep(), listed._default_params()) == {3, 7, 16, 10}

    shipped = yaml.safe_load(Path(source("config", "auto_tuner.yaml")).read_text(encoding="utf-8"))
    block = shipped["parameter_space"]
    assert (block["threads"], block["threads_fractions"]) == ("auto", [0.5, 0.75])
    space = tuner.ParameterSpace.from_dict(block)
    assert (space.threads_auto, space.threads_fractions) == (True, [0.5, 0.75])
    assert space.to_dict()["threads"] == "auto"
    assert tuner.ParameterSpace.from_dict({}).threads_auto is True
    unreadable = tuner.ParameterSpace.from_dict({"threads": "many", "threads_fractions": [1.5, "x", True]})
    assert (unreadable.threads_auto, unreadable.threads_fractions) == (True, [0.5, 0.75])
    assert "threads" in caplog.text
    cleaned = tuner.ParameterSpace.from_dict({"threads": [0, "x", 3, 3, True, 6]})
    assert (cleaned.threads_auto, cleaned.threads) == (False, [3, 6])


class _Pinning(_Bench):
    """A benchmark that pins each trial's count in the governor, as an Ollama
    head does (note_own_threads), and ends the run at its trial of ``stop``
    threads: a cancel, or a refusal no wait can lift."""

    def __init__(self, tuner, gov, *, stop, how, pin=True):
        super().__init__(tuner, _SPEEDS)
        self._gov, self._stop, self._how, self._pin = gov, stop, how, pin
        self.run = None

    def __call__(self, params):
        result = super().__call__(params)
        threads = params.get("threads")
        if self._pin:
            self._gov.pin_threads("m", threads, threads)
        if threads == self._stop:
            if self._how == "cancel":
                self.run.cancel()
            else:
                raise self._tuner.TunerRefused("refused by the resource governor: background_capacity_unknown")
        return result


def test_at13_a_run_that_ends_early_after_a_thread_trial_leaves_the_resident_pinned_at_the_plan():
    """AT13 -- a run cancelled during its thread axis, or ended by a refusal
    after a trial at another count, never runs its settling trial: the
    governor is told to pin the plan's count (ten) in place of the trial's,
    so the next call reloads the model once at the plan instead of every
    call keeping a count nobody plans; where no pin was made, none is.
    Leaving the resident pinned at the trial's count -> RED."""
    rg, tuner = _open()
    for how, raised in (("cancel", ValueError), ("refuse", tuner.TunerRefused)):
        gov = _governor(rg, hardware=_Hardware(_topology(12)))
        rg._governor = gov
        bench = _Pinning(tuner, gov, stop=5, how=how)
        bench.run = _tuner(tuner, bench)
        with pytest.raises(raised):
            bench.run.run("m", tuner.TunerJob())
        assert gov.pinned_threads("m") == (10, 10), how
    bare = _governor(rg, hardware=_Hardware(_topology(12)))
    rg._governor = bare
    quiet = _Pinning(tuner, bare, stop=5, how="cancel", pin=False)
    quiet.run = _tuner(tuner, quiet)
    with pytest.raises(ValueError):
        quiet.run.run("m", tuner.TunerJob())
    assert bare.pinned_threads("m") is None


class _Holding(_Bench):
    """A benchmark whose trials reach the engine and pin their count and
    placement, as an Ollama head does, except the trials at the plan's ten
    threads once a trial at eight has run: the governor holds those, they
    never reach the engine and pin nothing. ``stop`` ends the run at the
    first held trial: "cancel", "refuse", or None (the run goes on)."""

    def __init__(self, tuner, gov, *, stop=None, later=None):
        super().__init__(tuner, _SPEEDS, later=later)
        self._gov, self._stop = gov, stop
        self.run = None
        self.eight = False
        self.held = 0

    def __call__(self, params):
        threads = params.get("threads")
        if threads == 10 and self.eight:
            self.held += 1
            if self._stop == "cancel":
                self.run.cancel()
            elif self._stop == "refuse":
                raise self._tuner.TunerRefused("refused by the resource governor: background_capacity_unknown")
            return self._tuner.BenchmarkResult(
                params=params, error="held by the resource governor: background_gate_interactive"
            )
        result = super().__call__(params)
        self.eight = self.eight or threads == 8
        self._gov.pin_threads("m", threads, threads)
        self._gov.pin_placement("m", "split:20")
        return result


def test_at14_the_pin_follows_the_count_the_resident_holds_not_the_last_trial_tried():
    """AT14 -- a trial the governor holds never reaches the engine, so the
    resident keeps the count of the last trial that did: after the trial at
    eight, the trials at the plan's ten are held, and a run then cancelled,
    ended by a refusal, or completed with a confirmation that does not hold
    and its settling trial held as well, leaves the resident pinned at what a
    fresh load would get -- the plan's ten, or, once a count is kept for that
    model, engine and placement, that count (seven). Deciding by the last
    trial tried -> RED."""
    rg, tuner = _open()
    for stop, raised in (("cancel", ValueError), ("refuse", tuner.TunerRefused), (None, None)):
        gov = _governor(rg, hardware=_Hardware(_topology(12)))
        rg._governor = gov
        bench = _Holding(tuner, gov, stop=stop, later={8: 9.0})
        bench.run = _tuner(tuner, bench)
        if raised is None:
            bench.run.run("m", tuner.TunerJob())
        else:
            with pytest.raises(raised):
                bench.run.run("m", tuner.TunerJob())
        assert bench.held >= 1, stop
        assert gov.pinned_threads("m") == (10, 10), stop
    kept = _governor(rg, hardware=_Hardware(_topology(12)))
    assert kept.record_thread_optimum("m", "ollama", "split:20", threads=7, threads_batch=7, tg=12.0, base_tg=10.0)
    rg._governor = kept
    bench = _Holding(tuner, kept, stop="cancel")
    bench.run = _tuner(tuner, bench)
    with pytest.raises(ValueError):
        bench.run.run("m", tuner.TunerJob())
    assert kept.pinned_threads("m") == (7, 7)


class _Quiet(_Bench):
    """A benchmark whose every trial the governor holds: it reaches no
    engine and pins nothing, and the run is cancelled at the first."""

    def __init__(self, tuner):
        super().__init__(tuner, _SPEEDS)
        self.run = None
        self.held = 0

    def __call__(self, params):
        self.held += 1
        self.run.cancel()
        return self._tuner.BenchmarkResult(params=params, error="held by the resource governor: cpu_pressure")


class _WarmThenStop(_Bench):
    """A benchmark whose warm-up reaches the engine and pins its count and
    placement, as an Ollama head does, and that cancels the run there."""

    def __init__(self, tuner, gov):
        super().__init__(tuner, _SPEEDS)
        self._gov = gov
        self.run = None

    def __call__(self, params):
        result = super().__call__(params)
        self._gov.pin_threads("m", params.get("threads"), params.get("threads"))
        self._gov.pin_placement("m", "split:20")
        self.run.cancel()
        return result


def test_at15_a_run_that_changed_no_pin_changes_none_and_the_warm_up_counts_as_a_trial():
    """AT15 -- the re-pin is weighed against the pin the run found: a run
    whose every trial the governor held reached no engine, and leaves a
    resident pinned at its kept count (seven) as it was, rather than
    re-pinning it to the bare plan; a run cancelled after its warm-up, which
    reloaded the model at the plan's ten, has the governor pin the kept count
    back, the warm-up naming the engine as a trial does; and a run that
    reached the engine at the count it found pinned leaves that pin. Re-pinning
    a pin the run never changed, or forgetting the warm-up -> RED."""
    rg, tuner = _open()
    kept = _governor(rg, hardware=_Hardware(_topology(12)))
    assert kept.record_thread_optimum("m", "ollama", "split:20", threads=7, threads_batch=7, tg=12.0, base_tg=10.0)
    kept.pin_threads("m", 7, 7)
    kept.pin_placement("m", "split:20")
    rg._governor = kept
    quiet = _Quiet(tuner)
    quiet.run = _tuner(tuner, quiet)
    with pytest.raises((ValueError, RuntimeError)):
        quiet.run.run("m", tuner.TunerJob())
    assert quiet.held >= 1
    assert kept.pinned_threads("m") == (7, 7)
    warm = _WarmThenStop(tuner, kept)
    warm.run = tuner.AutoTuner(
        config=tuner.TunerConfig(warmup_runs=1, trials_per_param=1),
        param_space=tuner.ParameterSpace.from_dict({"threads": "auto"}),
        benchmark_fn=warm,
        thread_plan=lambda: _PLAN,
    )
    with pytest.raises(ValueError):
        warm.run.run("m", tuner.TunerJob())
    assert [call["threads"] for call in warm.calls] == [10]
    assert kept.pinned_threads("m") == (7, 7)
    # A run that reached the engine at the count it found pinned changed no
    # pin: the resident stays at ten, though a kept count would plan seven.
    kept.pin_threads("m", 10, 10)
    same = _WarmThenStop(tuner, kept)
    same.run = _tuner(tuner, same)
    with pytest.raises(ValueError):
        same.run.run("m", tuner.TunerJob())
    assert [call["threads"] for call in same.calls] == [10]
    assert kept.pinned_threads("m") == (10, 10)


class _Elsewhere(_Quiet):
    """A benchmark whose every trial the governor holds while another load,
    a chat say, brings the model in at the count kept for it (seven) and
    pins that; the run is cancelled at the first trial."""

    def __init__(self, tuner, gov):
        super().__init__(tuner)
        self._gov = gov

    def __call__(self, params):
        self._gov.pin_threads("m", 7, 7)
        self._gov.pin_placement("m", "split:20")
        return super().__call__(params)


def test_at16_a_run_whose_trials_reached_no_engine_re_pins_nothing():
    """AT16 -- a run none of whose trials reached an engine moved no pin
    itself: when another load pins the model meanwhile (a chat bringing it
    in at its kept count, seven), the run leaves that pin as it is rather
    than plan it again with no engine named, which would pin the bare plan
    and cost the chat a reload. Re-pinning after a run that reached no
    engine -> RED."""
    rg, tuner = _open()
    gov = _governor(rg, hardware=_Hardware(_topology(12)))
    assert gov.record_thread_optimum("m", "ollama", "split:20", threads=7, threads_batch=7, tg=12.0, base_tg=10.0)
    rg._governor = gov
    bench = _Elsewhere(tuner, gov)
    bench.run = _tuner(tuner, bench)
    assert gov.pinned_threads("m") is None
    with pytest.raises((ValueError, RuntimeError)):
        bench.run.run("m", tuner.TunerJob())
    assert bench.held >= 1
    assert gov.pinned_threads("m") == (7, 7)


def test_at17_a_run_reads_and_restores_pins_only_in_a_running_governor_and_starts_none():
    """AT17 -- a pin lives in a running governor, so a run reads and
    restores pins only there: with no governor running, a run, completed or
    cancelled, starts none (starting one would open its store and its
    refresh for nothing, a store in the package's data directory by
    default), and its trials still run as the benchmark gives them.
    Starting a governor to read a pin -> RED."""
    rg, tuner = _open()
    assert rg._governor is None
    done = _Bench(tuner, _SPEEDS)
    assert _tuner(tuner, done).run("m", tuner.TunerJob()).threads_optimum is not None
    stopped = _Pinning(tuner, None, stop=5, how="cancel", pin=False)
    stopped.run = _tuner(tuner, stopped)
    with pytest.raises(ValueError):
        stopped.run.run("m", tuner.TunerJob())
    assert (len(done.calls) > 0, rg._governor) == (True, None)
