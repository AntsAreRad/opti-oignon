#!/usr/bin/env python3
"""Resource Governor: admission knows who asks.

Every caller belongs to one of three closed classes, named in
resource_governor.yaml: ``interactive`` (a person is waiting on the answer:
chat, pipeline), ``user`` (asked by a person, not watched: benchmark,
agent_eval, the direct backstop, and any caller the file does not name) and
``background`` (asked by nobody: warm-ups, indexing, tuning, consolidation).
These contracts pin what the class changes:

  * The background never evicts: no eviction credit, no grant conditional on
    an eviction, no reload of a resident for more context, no split between
    VRAM and RAM unless the file allows it, and no load whose fit cannot be
    known (capacity or cost unknown), since the engine's own LRU would evict
    for it. A call on a resident at the context it holds loads nothing and
    stays admitted.
  * The background is evicted first: a resident only the background loaded,
    with no call in flight on it, is an eviction candidate before any other,
    whatever its idle time.
  * The in-flight registry: holding and releasing a ticket (the thread-local
    pass-through) counts the calls in flight per class; an entry older than
    its bound expires.
  * The background gate: held while an interactive call is in flight or
    waiting, or while the CPU pressure other programs suffer is above its
    mark (left below a lower one). Its reason is recorded, never counted as a
    resource refusal, and shown in the status.
  * The priority queue: ordered by class, then arrival; no caller passes a
    waiter of a higher class; within a class a waiter is passed at most
    ``max_bypass`` times; each class has its own depth and wait; the
    background waits by default; an emergency stop releases every waiter at
    its next wake.
  * The status names the calls in flight and waiting by class and the
    gate's state; the routes write the new scalar keys in range only and
    keep the gate's CPU marks in order, the class tables staying read-only.
  * The warm-up is the first background caller: it asks as the background
    and sends nothing while held; its keepalive ping never waits.
  * A background caller that kept a model resident for a run lets it go at
    the run's end: the governor unloads a guest only the background loaded,
    with no call in flight on it, and keeps one a higher class has used.

Isolation follows the governor suites: each contract opens the shared window
of ``tests/_isolation.py`` on the on-disk source, with a seeded ``db_utils``
and every other project module unreachable unless a contract seeds it, and
closes it after. Snapshots are injected fresh, so no reading of the machine
enters a decision; the clock is a settable stand-in; a held ticket is always
one a real admission handed out. Contracts that wait run the waiter on a
thread and release it by moving the clock and waking the queue, so none of
them can hang the suite.

Local-only. Runs under pytest or the __main__ runner.
"""

import sqlite3
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_RG = "opti_oignon.resource_governor"
_WARMUP = "opti_oignon.model_warmup"
_BACKEND = "opti_oignon.inference_backend"
_ROUTES = "opti_oignon.api.routes_governor"
_ESTOP = "opti_oignon.emergency_stop"
_CM = "opti_oignon.context_manager"
_GIB = 1024 ** 3
_ZERO = {"interactive": 0, "user": 0, "background": 0}
_CLOSERS = []


def _db_utils():
    """A db_utils stand-in whose safe_connect is plain sqlite."""
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda p, **kw: sqlite3.connect(
        str(p), check_same_thread=kw.get("check_same_thread", False)
    )
    return db


def _open(*extra, seeded=None, blocked=(_CM, _ESTOP), packages=()):
    """The governor and ``extra`` (name, path) modules after it, in one window."""
    targets = {_RG: source("resource_governor.py")}
    for name, path in extra:
        targets[name] = path
    seeds = {"opti_oignon.db_utils": _db_utils()}
    seeds.update(seeded or {})
    loaded, restore = isolate(targets=targets, blocked=blocked, seeded=seeds, packages=packages)
    _CLOSERS.append(restore)
    return loaded


def _rg():
    return _open()[_RG]


def _project_modules():
    """Every project entry of the module cache, by identity."""
    return {k: v for k, v in sys.modules.items() if k == "opti_oignon" or k.startswith("opti_oignon.")}


@pytest.fixture(autouse=True)
def _left_as_found():
    """No contract may leave a project module changed in the cache."""
    before = _project_modules()
    yield
    while _CLOSERS:
        _CLOSERS.pop()()
    assert _project_modules() == before, "every project module is left as the contract found it"


class _Clock:
    """A settable monotonic stand-in."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class _Hardware:
    """A hardware profile that answers only the CPU pressure others suffer."""

    def __init__(self, reading=None):
        self.reading = reading
        self.reads = 0

    def others_cpu_pressure(self):
        self.reads += 1
        return self.reading

    def pressure(self):
        return None

    def placement(self):
        return None


def _pressure(value, source="cgroups"):
    return {"source": source, "some_avg10": value, "cgroup": "app.slice/editor.scope", "count": 4}


def _warmup(keep_alive="10m"):
    """The keep_alive the evictable derivation reads, and nothing else."""
    return types.SimpleNamespace(keep_alive=keep_alive)


def _config(rg, *, capacity=10.0, weights=None, **fields):
    cfg = rg.GovernorConfig(total_vram_gb=capacity, safety_margin_gb=1.5, kv_coefficient=0.5)
    cfg.weights_override_models = dict(weights or {})
    for key, value in fields.items():
        setattr(cfg, key, value)
    return cfg


def _governor(rg, cfg, *, hardware=None, warmup=None, clock=None):
    """A real governor with injected seams and a hand-set config."""
    clk = clock or _Clock()
    tmp = tempfile.mkdtemp()
    gov = rg.ResourceGovernor(
        config_path="/nonexistent-resource-governor-config",
        db_path=str(Path(tmp) / "governor.db"),
        warmup=warmup,
        registry=None,
        clock=clk,
        meminfo_path="/nonexistent-meminfo",
        hardware=hardware,
    )
    gov._config = cfg
    return gov, clk


def _snapshot(rg, clk, *, capacity=10.0, in_use=0.0, loaded=(), ram_mb=64000.0):
    """A snapshot stamped now, so get_snapshot_fast returns it as-is."""
    return rg.ResourceSnapshot(
        taken_at=clk(),
        ttl_s=9999.0,
        loaded=list(loaded),
        capacity_gb=capacity,
        vram_in_use_gb=in_use,
        ram_available_mb=ram_mb,
    )


def _view(rg, name, gb, *, idle_s, keep_alive_s=600.0):
    """A resident whose idle time, derived against the wall clock, is ``idle_s``."""
    return rg.LoadedModelView(
        name=name, size_vram_bytes=int(gb * _GIB), expires_at=time.time() + keep_alive_s - idle_s
    )


def _idle_view(rg, name, gb):
    """A resident whose epoch-0 expiry makes it idle past any threshold."""
    return rg.LoadedModelView(name=name, size_vram_bytes=int(gb * _GIB), expires_at=0.0)


def _queued(gov):
    """The enqueue entries of the decisions ring, oldest first."""
    return [d for d in reversed(gov.store.recent_decisions(50)) if d["decision"] == "queue"]


def _record_attempts(gov):
    """Record the caller of every admission attempt the governor makes."""
    attempts = []
    real = gov.admit

    def admit(model, requested_ctx=None, caller="chat", **kw):
        attempts.append(caller)
        return real(model, requested_ctx, caller=caller, **kw)

    gov.admit = admit
    return attempts


def _until(predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _waiting(gov, model, caller, **kw):
    """admit_or_wait on its own thread; its decision lands in the dict."""
    out = {}

    def run():
        out["decision"] = gov.admit_or_wait(model, caller=caller, **kw)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, out


def _release(gov, clk, *threads):
    """Move the clock past every wait and wake the queue, then join."""
    clk.t += 1.0e6
    gov._notify_queue()
    for thread in threads:
        thread.join(5.0)


# ---------------------------------------------------------------------------
# The classes
# ---------------------------------------------------------------------------


def test_gc1_the_shipped_file_names_who_asks_and_the_background_defaults():
    """GC1 -- the shipped file maps each caller to its class and carries the
    decided defaults: the background waits in the queue (eight deep, two
    minutes), never splits, and is held by a gate whose in-flight entries
    expire after fifteen minutes and whose CPU marks are 10 and 5. Dropping a
    caller from the shipped table leaves it a user -> RED."""
    rg = _rg()
    cfg = rg.load_config(Path(source("config", "resource_governor.yaml")))
    callers = ("chat", "pipeline", "benchmark", "agent_eval", "direct", "warmup", "index", "tuner", "reverie")
    assert {c: cfg.class_of(c) for c in callers} == {
        "chat": "interactive",
        "pipeline": "interactive",
        "benchmark": "user",
        "agent_eval": "user",
        "direct": "user",
        "warmup": "background",
        "index": "background",
        "tuner": "background",
        "reverie": "background",
    }
    assert cfg.class_queued == {"interactive": False, "user": False, "background": True}
    assert (cfg.class_depth.get("background"), cfg.class_wait_s.get("background")) == (8, 120.0)
    assert cfg.background_allow_split is False
    assert cfg.queue_max_bypass == 2
    assert (cfg.background_gate_enabled, cfg.background_gate_in_flight_max_s) == (True, 900.0)
    assert (cfg.background_gate_cpu_enter, cfg.background_gate_cpu_exit) == (10.0, 5.0)


def test_gc2_a_decision_carries_the_class_of_its_caller_and_a_stranger_is_a_user():
    """GC2 -- every decision names its class, in the ticket and in its dict,
    and a caller the table does not name is a user, never the background.
    Defaulting a stranger to the background -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, weights={"m": 1.0}))
    gov._snapshot = _snapshot(rg, clk)
    seen = {}
    for caller in ("chat", "benchmark", "warmup", "never-named"):
        decision = gov.admit("m", 2048, caller=caller)
        assert decision.admitted is True
        assert decision.to_dict()["admission_class"] == decision.admission_class
        seen[caller] = decision.admission_class
    assert seen == {"chat": "interactive", "benchmark": "user", "warmup": "background", "never-named": "user"}


def test_gc3_the_classes_are_closed_and_a_file_may_add_or_move_a_caller():
    """GC3 -- the file's caller table is merged over the defaults: it may add
    a caller or move one to another class, but a class outside the three is
    refused and the caller keeps its class. Accepting any class name -> RED."""
    rg = _rg()
    path = Path(tempfile.mkdtemp()) / "resource_governor.yaml"
    path.write_text(
        "classes:\n  callers:\n    chat: urgent\n    nightly: background\n    benchmark: interactive\n",
        encoding="utf-8",
    )
    cfg = rg.load_config(path)
    assert cfg.class_of("chat") == "interactive"
    assert cfg.class_of("nightly") == "background"
    assert cfg.class_of("benchmark") == "interactive"
    assert cfg.class_of("never-named") == "user"


# ---------------------------------------------------------------------------
# The background never evicts
# ---------------------------------------------------------------------------


def test_gc4_the_background_never_gets_a_grant_conditional_on_an_eviction():
    """GC4 -- the frame where chat is granted CONDITIONAL on evicting an idle
    resident (budget 0.5 now, 4.5 with the eviction, cost 4.0; no split) is
    refused to the background, with no conditional flag: it counts no
    eviction credit. Giving the background the eviction credit -> RED."""
    rg = _rg()
    cfg = _config(rg, weights={"m": 3.0}, idle_evict_threshold_s=600.0, offload_enabled=False)
    gov, clk = _governor(rg, cfg, warmup=_warmup())
    gov._snapshot = _snapshot(rg, clk, in_use=8.0, loaded=[_idle_view(rg, "resident", 4.0)])
    chat = gov.admit("m", 2048, caller="chat")
    assert chat.admitted is True and chat.conditional_on_eviction is True
    background = gov.admit("m", 2048, caller="warmup")
    assert background.admitted is False
    assert background.conditional_on_eviction is False


def test_gc5_the_background_never_reloads_a_resident_for_more_context():
    """GC5 -- a resident holding 2048 tokens asked 8192 is a reload for a
    user (its memory freed first, then the whole cost) and a refusal for the
    background, which loads nothing. Letting the background reload -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=24.0))
    held = rg.LoadedModelView(name="m", size_vram_bytes=4 * _GIB, context_length=2048)
    gov._snapshot = _snapshot(rg, clk, capacity=24.0, in_use=4.0, loaded=[held])
    user = gov.admit("m", 8192, caller="benchmark")
    assert user.admitted is True and user.load_expected is True
    background = gov.admit("m", 8192, caller="warmup")
    assert background.admitted is False
    assert background.reason == "background_never_reloads"
    assert background.load_expected is False


def test_gc6_the_background_splits_between_vram_and_ram_only_when_allowed():
    """GC6 -- a 10 GiB model on an 8 GiB card is admitted split for a user;
    the background is refused the same split until the file allows it.
    Ignoring allow_split (either way) -> RED."""
    rg = _rg()
    cfg = _config(rg, capacity=8.0, weights={"m": 10.0})
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=8.0)
    user = gov.admit("m", None, caller="benchmark")
    assert user.admitted is True and user.partial_offload is True
    assert gov.admit("m", None, caller="warmup").admitted is False
    cfg.background_allow_split = True
    allowed = gov.admit("m", None, caller="warmup")
    assert allowed.admitted is True and allowed.partial_offload is True


def test_gc7_the_background_is_refused_a_load_when_the_free_memory_cannot_be_known():
    """GC7 -- with the card's capacity unknown a user is admitted fail-open
    (3.1); the background, which must not let the engine's own LRU evict for
    it, is refused by name. Failing the background open too -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=None, weights={"m": 2.0}))
    gov._snapshot = _snapshot(rg, clk, capacity=None)
    user = gov.admit("m", None, caller="benchmark")
    assert user.admitted is True and user.reason.startswith("capacity_unknown_fail_open")
    background = gov.admit("m", None, caller="warmup")
    assert background.admitted is False
    assert background.reason == "background_capacity_unknown"


def test_gc8_a_background_call_on_a_resident_at_its_context_is_admitted_and_loads_nothing():
    """GC8 -- a keepalive-like call on a resident at the context it holds
    loads nothing, so it stays admitted even with the capacity unknown.
    Refusing the background before the resident check -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=None))
    held = rg.LoadedModelView(name="m", size_vram_bytes=4 * _GIB, context_length=4096)
    gov._snapshot = _snapshot(rg, clk, capacity=None, loaded=[held])
    ping = gov.admit("m", None, caller="warmup")
    assert ping.admitted is True
    assert ping.load_expected is False
    assert ping.num_ctx == 4096


def test_gc9_the_background_is_refused_a_load_whose_cost_is_unknown():
    """GC9 -- a model nothing can price is never too large for a user (3.1);
    the background is refused it by name, since it cannot show the load fits
    the free memory. Treating the unknown cost as zero for the background ->
    RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=24.0))
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    assert gov.admit("never-measured", None, caller="benchmark").admitted is True
    background = gov.admit("never-measured", None, caller="warmup")
    assert background.admitted is False
    assert background.reason == "background_cost_unknown"


# ---------------------------------------------------------------------------
# The background is evicted first
# ---------------------------------------------------------------------------


def _two_residents(rg, clk):
    """``warmed`` idle five seconds, ``old`` idle well past the threshold."""
    return _snapshot(
        rg,
        clk,
        capacity=24.0,
        in_use=5.0,
        loaded=[_view(rg, "warmed", 3.0, idle_s=5.0), _view(rg, "old", 2.0, idle_s=5000.0)],
    )


def test_gc10_a_resident_only_the_background_loaded_is_evicted_first_whatever_its_idle_time():
    """GC10 -- an idle-threshold candidate list holds only ``old``; once
    ``warmed`` is known to be loaded by the background it comes first, idle
    five seconds or not. Dropping the background-first branch -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=24.0, idle_evict_threshold_s=600.0), warmup=_warmup())
    snapshot = _two_residents(rg, clk)
    assert [c[0] for c in gov._evictable_candidates(snapshot)] == ["old"]
    gov.note_loaded_by("warmed", "background")
    assert [c[0] for c in gov._evictable_candidates(snapshot)] == ["warmed", "old"]


def test_gc11_a_background_resident_with_a_call_in_flight_on_it_is_not_evicted_first():
    """GC11 -- while a background ticket on ``warmed`` is held, it is not a
    candidate; released, it is first again. Ignoring the calls in flight ->
    RED."""
    rg = _rg()
    cfg = _config(rg, capacity=24.0, weights={"warmed": 3.0}, idle_evict_threshold_s=600.0)
    gov, clk = _governor(rg, cfg, warmup=_warmup())
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    ticket = gov.admit("warmed", None, caller="warmup")
    gov.note_loaded_by("warmed", "background")
    snapshot = _two_residents(rg, clk)
    rg.set_active_ticket(ticket)
    try:
        in_flight = [c[0] for c in gov._evictable_candidates(snapshot)]
    finally:
        rg.clear_active_ticket()
    assert in_flight == ["old"]
    assert [c[0] for c in gov._evictable_candidates(snapshot)] == ["warmed", "old"]


def test_gc12_a_background_resident_a_higher_class_has_held_is_no_longer_evicted_first():
    """GC12 -- once a chat ticket on ``warmed`` has been held, the model is
    the chat's too: released, it is not evicted first. Keeping the
    background mark after a higher class used the model -> RED."""
    rg = _rg()
    cfg = _config(rg, capacity=24.0, weights={"warmed": 3.0}, idle_evict_threshold_s=600.0)
    gov, clk = _governor(rg, cfg, warmup=_warmup())
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    chat = gov.admit("warmed", None, caller="chat")
    gov.note_loaded_by("warmed", "background")
    rg.set_active_ticket(chat)
    rg.clear_active_ticket()
    assert [c[0] for c in gov._evictable_candidates(_two_residents(rg, clk))] == ["old"]


def test_gc13_the_engine_gate_marks_a_load_with_the_class_of_the_ticket_that_loads_it():
    """GC13 -- a load accounted through the engine gate under a background
    ticket is the background's; one under a user ticket is not. Not marking
    the load at the gate -> RED."""
    rg = _rg()
    cfg = _config(rg, capacity=24.0, weights={"warmed": 3.0, "asked": 3.0}, idle_evict_threshold_s=600.0)
    gov, clk = _governor(rg, cfg, warmup=_warmup())
    rg._governor = gov
    for model, caller in (("warmed", "warmup"), ("asked", "benchmark")):
        gov._snapshot = _snapshot(rg, clk, capacity=24.0)
        ticket = gov.admit(model, None, caller=caller)
        assert ticket.load_expected is True
        with rg.ticket_scope(ticket):
            rg.backend_admission_gate(model, {})
    snapshot = _snapshot(
        rg,
        clk,
        capacity=24.0,
        in_use=6.0,
        loaded=[_view(rg, "warmed", 3.0, idle_s=5.0), _view(rg, "asked", 3.0, idle_s=5.0)],
    )
    assert [c[0] for c in gov._evictable_candidates(snapshot)] == ["warmed"]


# ---------------------------------------------------------------------------
# The in-flight registry
# ---------------------------------------------------------------------------


def test_gc14_holding_and_releasing_a_ticket_count_the_calls_in_flight_per_thread_and_class():
    """GC14 -- set_active_ticket, ticket_scope and clear_active_ticket feed
    the governor's count of calls in flight: one ticket per thread, counted
    by class, back to zero once released. Not feeding the registry -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=24.0, weights={"m": 1.0}))
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    chat = gov.admit("m", None, caller="chat")
    background = gov.admit("m", None, caller="warmup")
    user = gov.admit("m", None, caller="benchmark")
    assert gov.in_flight_by_class() == _ZERO
    rg.set_active_ticket(chat)
    assert gov.in_flight_by_class() == {**_ZERO, "interactive": 1}
    with rg.ticket_scope(background):
        assert gov.in_flight_by_class() == {**_ZERO, "background": 1}
    assert gov.in_flight_by_class() == {**_ZERO, "interactive": 1}
    held, done = threading.Event(), threading.Event()

    def other():
        rg.set_active_ticket(user)
        held.set()
        done.wait(5.0)
        rg.clear_active_ticket()

    thread = threading.Thread(target=other, daemon=True)
    thread.start()
    assert held.wait(5.0)
    assert gov.in_flight_by_class() == {**_ZERO, "interactive": 1, "user": 1}
    done.set()
    thread.join(5.0)
    assert gov.in_flight_by_class() == {**_ZERO, "interactive": 1}
    rg.clear_active_ticket()
    assert gov.in_flight_by_class() == _ZERO


def test_gc15_an_in_flight_entry_older_than_its_bound_expires_and_holds_nothing():
    """GC15 -- a chat ticket never released (a leak) holds the background
    only until it is older than in_flight_max_s. Never expiring -> RED."""
    rg = _rg()
    cfg = _config(rg, capacity=24.0, weights={"m": 1.0}, background_gate_in_flight_max_s=60.0)
    gov, clk = _governor(rg, cfg)
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    rg.set_active_ticket(gov.admit("m", None, caller="chat"))
    try:
        assert gov.admit("m", None, caller="warmup").held_by == "interactive_in_flight"
        clk.t += 61.0
        assert gov.admit("m", None, caller="warmup").admitted is True
        assert gov.in_flight_by_class()["interactive"] == 0
    finally:
        rg.clear_active_ticket()


# ---------------------------------------------------------------------------
# The background gate
# ---------------------------------------------------------------------------


def test_gc16_an_interactive_call_in_flight_holds_the_background_and_its_release_opens_the_gate():
    """GC16 -- while a chat ticket is held the background is refused, held
    by name; released, the same call is admitted. Not consulting the calls in
    flight -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=24.0, weights={"m": 1.0}))
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    rg.set_active_ticket(gov.admit("m", None, caller="chat"))
    try:
        held = gov.admit("m", None, caller="warmup")
    finally:
        rg.clear_active_ticket()
    assert held.admitted is False
    assert held.held_by == "interactive_in_flight"
    assert held.reason == "background_held:interactive_in_flight"
    assert gov.admit("m", None, caller="warmup").admitted is True


def test_gc17_a_user_call_in_flight_does_not_hold_the_background():
    """GC17 -- only the interactive class holds the gate: a benchmark ticket
    in flight leaves the background admitted. Holding on any class -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=24.0, weights={"m": 1.0}))
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    rg.set_active_ticket(gov.admit("m", None, caller="benchmark"))
    try:
        background = gov.admit("m", None, caller="warmup")
    finally:
        rg.clear_active_ticket()
    assert background.admitted is True
    assert background.held_by is None


def test_gc18_the_cpu_pressure_others_suffer_holds_the_background_between_its_two_marks():
    """GC18 -- pressure readings 4, 12, 7, 4.9, 7 against marks 10 (enter)
    and 5 (exit): open, held, still held, open, still open. Dropping the
    exit mark (holding only at or above 10) -> RED."""
    rg = _rg()
    hardware = _Hardware()
    cfg = _config(
        rg,
        capacity=24.0,
        weights={"m": 1.0},
        background_gate_cpu_enter=10.0,
        background_gate_cpu_exit=5.0,
        snapshot_ttl_s=2.0,
    )
    gov, clk = _governor(rg, cfg, hardware=hardware)
    verdicts = []
    for value in (4.0, 12.0, 7.0, 4.9, 7.0):
        hardware.reading = _pressure(value)
        clk.t += 3.0  # past the freshness of the last reading
        gov._snapshot = _snapshot(rg, clk, capacity=24.0)
        verdicts.append(gov.admit("m", None, caller="warmup").held_by)
    assert verdicts == [None, "cpu_pressure", "cpu_pressure", None, None]


def test_gc19_a_pressure_nobody_can_read_leaves_the_gate_open():
    """GC19 -- the gate asks for the reading; none comes back, and the
    background is admitted with the pressure shown unknown. Holding on an
    unknown pressure -> RED."""
    rg = _rg()
    hardware = _Hardware(reading=None)
    gov, clk = _governor(rg, _config(rg, capacity=24.0, weights={"m": 1.0}), hardware=hardware)
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    assert gov.admit("m", None, caller="warmup").admitted is True
    assert hardware.reads >= 1
    assert gov.background_gate_state()["pressure"] is None


def test_gc20_a_disabled_gate_never_holds_the_background():
    """GC20 -- with the gate off, neither a chat in flight nor a pressure of
    50 holds the background. Ignoring the switch -> RED."""
    rg = _rg()
    hardware = _Hardware(reading=_pressure(50.0))
    cfg = _config(rg, capacity=24.0, weights={"m": 1.0}, background_gate_enabled=False)
    gov, clk = _governor(rg, cfg, hardware=hardware)
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    rg.set_active_ticket(gov.admit("m", None, caller="chat"))
    try:
        background = gov.admit("m", None, caller="warmup")
    finally:
        rg.clear_active_ticket()
    assert background.admitted is True


def test_gc21_a_held_background_is_recorded_with_its_reason_and_is_never_a_resource_refusal():
    """GC21 -- the hold is written to the decisions ring with its reason, but
    the refusal-rate window that drives backpressure does not count it.
    Counting a hold as a refusal -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=24.0, weights={"m": 1.0}))
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    rg.set_active_ticket(gov.admit("m", None, caller="chat"))
    try:
        before = gov._refusal_window_stats()
        gov.admit("m", None, caller="warmup")
        after = gov._refusal_window_stats()
    finally:
        rg.clear_active_ticket()
    reasons = [d["reason"] for d in gov.store.recent_decisions(10)]
    assert "background_held:interactive_in_flight" in reasons
    assert after == before


def test_gc22_an_interactive_caller_waiting_in_the_queue_holds_the_background():
    """GC22 -- a chat enrolled in the queue and waiting holds the background
    by name; once it leaves, the background is admitted. Ignoring the
    interactive waiters -> RED."""
    rg = _rg()
    cfg = _config(
        rg, capacity=8.0, weights={"big": 50.0, "m": 1.0}, offload_enabled=False,
        queue_enabled_per_caller={"chat": True},
    )
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=8.0)
    thread, _out = _waiting(gov, "big", "chat")
    try:
        assert _until(lambda: gov.queue_by_class()["interactive"] == 1)
        assert gov.admit("m", None, caller="warmup").held_by == "interactive_queued"
    finally:
        _release(gov, clk, thread)
    assert not thread.is_alive()
    assert gov.admit("m", None, caller="warmup").admitted is True


# ---------------------------------------------------------------------------
# The priority queue
# ---------------------------------------------------------------------------


def test_gc23_the_background_waits_in_the_queue_by_default_and_a_caller_may_opt_out():
    """GC23 -- a refused background call enqueues with no enrolment named;
    a caller the file opts out gets the plain refusal. Leaving the
    background out of the queue -> RED."""
    rg = _rg()
    cfg = _config(rg, capacity=8.0, weights={"big": 50.0}, offload_enabled=False, class_wait_s={"background": 0.0})
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=8.0)
    assert gov.admit_or_wait("big", caller="warmup").admitted is False
    assert [d["caller"] for d in _queued(gov)] == ["warmup"]
    cfg.queue_enabled_per_caller = {"warmup": False}
    gov.admit_or_wait("big", caller="warmup")
    assert [d["caller"] for d in _queued(gov)] == ["warmup"]


def test_gc24_each_class_has_its_own_depth_bound():
    """GC24 -- the user class at its bound of 2 refuses a third user waiter,
    and the background still enqueues under its own bound of 8. One bound
    for all classes -> RED."""
    rg = _rg()
    cfg = _config(
        rg, capacity=8.0, weights={"big": 50.0}, offload_enabled=False,
        queue_enabled_per_caller={"benchmark": True}, queue_depth=2, queue_wait_s=0.0,
        class_depth={"background": 8}, class_wait_s={"background": 0.0},
    )
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=8.0)
    for _ in range(2):
        gov._enqueue("user")
    gov.admit_or_wait("big", caller="benchmark")
    gov.admit_or_wait("big", caller="warmup")
    assert [d["caller"] for d in _queued(gov)] == ["warmup"]


def test_gc25_no_caller_passes_a_waiter_of_a_higher_class():
    """GC25 -- with a user waiting, a background call that fits does not
    even try: it waits out its bound behind the user. A background waiter
    blocks no user. Letting a lower class try past a higher one -> RED."""
    rg = _rg()
    cfg = _config(
        rg, capacity=24.0, weights={"m": 1.0}, queue_enabled_per_caller={"benchmark": True},
        queue_wait_s=0.0, class_wait_s={"background": 0.0},
    )
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    attempts = _record_attempts(gov)
    waiting_user = gov._enqueue("user")
    background = gov.admit_or_wait("m", caller="warmup")
    assert background.admitted is False
    assert background.reason == "queue_wait_expired"
    assert attempts == []
    gov._dequeue(waiting_user)
    gov._enqueue("background")
    assert gov.admit_or_wait("m", caller="benchmark").admitted is True
    assert attempts == ["benchmark"]


def test_gc26_a_waiter_is_passed_at_most_max_bypass_times_within_its_class():
    """GC26 -- with an allowance of one, a later user call that fits passes
    the waiting head once; the next one does not try and is refused. Not
    charging the pass to the head -> RED."""
    rg = _rg()
    cfg = _config(
        rg, capacity=24.0, weights={"m": 1.0}, queue_enabled_per_caller={"benchmark": True},
        queue_wait_s=0.0, queue_max_bypass=1,
    )
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    head = gov._enqueue("user")
    attempts = _record_attempts(gov)
    assert gov.admit_or_wait("m", caller="benchmark").admitted is True
    assert head.bypassed == 1
    second = gov.admit_or_wait("m", caller="benchmark")
    assert second.admitted is False
    assert attempts == ["benchmark"]


def test_gc27_an_allowance_of_zero_keeps_arrival_order_within_a_class():
    """GC27 -- with max_bypass 0 a later user call never passes the head,
    even when it fits and the head does not. Reading 0 as no bound -> RED."""
    rg = _rg()
    cfg = _config(
        rg, capacity=24.0, weights={"m": 1.0}, queue_enabled_per_caller={"benchmark": True},
        queue_wait_s=0.0, queue_max_bypass=0,
    )
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    gov._enqueue("user")
    attempts = _record_attempts(gov)
    later = gov.admit_or_wait("m", caller="benchmark")
    assert later.admitted is False
    assert attempts == []


def test_gc28_an_emergency_stop_releases_every_waiter_at_its_next_wake_eligible_or_not():
    """GC28 -- a background waiter queued behind a user (so never trying) is
    released at its next wake once the stop engages, with the stop's own
    refusal, and its slot is given back. Checking the stop only before an
    attempt -> RED (the waiter stays)."""
    estop = types.ModuleType(_ESTOP)
    estop.stopped = False
    estop.is_stopped = lambda: estop.stopped
    estop.refusal_payload = lambda: {"error": "emergency_stopped"}
    rg = _open(seeded={_ESTOP: estop}, blocked=(_CM,))[_RG]
    gov, clk = _governor(rg, _config(rg, capacity=8.0, weights={"big": 50.0}, offload_enabled=False))
    gov._snapshot = _snapshot(rg, clk, capacity=8.0)
    gov._enqueue("user")
    thread, out = _waiting(gov, "big", "warmup")
    try:
        assert _until(lambda: gov.queue_by_class()["background"] == 1)
        estop.stopped = True
        gov._notify_queue()
        thread.join(5.0)
        stayed = thread.is_alive()
    finally:
        _release(gov, clk, thread)
    assert stayed is False
    assert out["decision"].is_estop is True
    assert gov.queue_by_class()["background"] == 0


def test_gc29_a_waiter_is_admitted_when_memory_frees_and_its_slot_is_given_back():
    """GC29 -- a background call refused for memory waits; when the card
    frees and the queue is woken it is admitted, and the queue is empty
    after. Not re-running admission on a wake -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=8.0, weights={"m": 6.0}, offload_enabled=False))
    gov._snapshot = _snapshot(rg, clk, capacity=8.0, in_use=4.0)
    thread, out = _waiting(gov, "m", "warmup")
    try:
        assert _until(lambda: gov.queue_by_class()["background"] == 1)
        gov._snapshot = _snapshot(rg, clk, capacity=8.0, in_use=0.0)
        gov._notify_queue()
        thread.join(5.0)
        stayed = thread.is_alive()
    finally:
        _release(gov, clk, thread)
    assert stayed is False
    assert out["decision"].admitted is True
    assert gov.queue_by_class() == _ZERO


def test_gc30_a_caller_may_shorten_its_wait_but_never_lengthen_it_past_its_class():
    """GC30 -- with the background bound at 120 s, a caller asking 5 s
    leaves after 5, and one asking 500 s leaves after 120. Taking the
    caller's figure above the class bound -> RED."""
    rg = _rg()
    cfg = _config(rg, capacity=8.0, weights={"big": 50.0}, offload_enabled=False, class_wait_s={"background": 120.0})
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=8.0)
    short, _a = _waiting(gov, "big", "warmup", wait_s=5.0)
    long, _b = _waiting(gov, "big", "warmup", wait_s=500.0)
    try:
        assert _until(lambda: gov.queue_by_class()["background"] == 2)
        clk.t += 6.0
        gov._notify_queue()
        short.join(5.0)
        short_left, long_stayed = not short.is_alive(), long.is_alive()
        clk.t += 115.0  # 1121: past the class's own bound, short of 500
        gov._notify_queue()
        long.join(5.0)
        long_left = not long.is_alive()
    finally:
        _release(gov, clk, short, long)
    assert (short_left, long_stayed, long_left) == (True, True, True)


# ---------------------------------------------------------------------------
# The status
# ---------------------------------------------------------------------------


def test_gc31_the_status_names_the_calls_in_flight_and_queued_by_class_and_why_the_gate_holds():
    """GC31 -- the /status body carries a scheduling section: calls in flight
    and waiting, by class, and the gate's state, reason and the pressure
    reading it last saw. Leaving the section out -> RED."""
    loaded = _open((_ROUTES, source("api", "routes_governor.py")), packages=("opti_oignon.api",))
    rg, routes = loaded[_RG], loaded[_ROUTES]
    hardware = _Hardware(reading=_pressure(3.0))
    gov, clk = _governor(rg, _config(rg, capacity=24.0, weights={"m": 1.0}), hardware=hardware)
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    gov._enqueue("background")
    rg.set_active_ticket(gov.admit("m", None, caller="chat"))
    try:
        scheduling = routes.status_payload(gov)["scheduling"]
    finally:
        rg.clear_active_ticket()
    assert scheduling["in_flight"] == {**_ZERO, "interactive": 1}
    assert scheduling["queued"] == {**_ZERO, "background": 1}
    gate = scheduling["background_gate"]
    assert (gate["enabled"], gate["state"], gate["reason"]) == (True, "held", "interactive_in_flight")
    assert (gate["pressure"]["source"], gate["pressure"]["some_avg10"]) == ("cgroups", 3.0)


def test_gc34_the_routes_write_the_new_scalar_keys_in_range_only_and_keep_the_cpu_marks_in_order():
    """GC34 -- the allowance and the gate's scalars are written to the file
    and read back; a negative allowance, a mark above 100, an exit mark
    above the enter mark and the class tables are refused, the file left as
    it was. Dropping the ordered pair of the CPU marks -> RED."""
    loaded = _open((_ROUTES, source("api", "routes_governor.py")), packages=("opti_oignon.api",))
    rg, routes = loaded[_RG], loaded[_ROUTES]
    path = Path(tempfile.mkdtemp()) / "resource_governor.yaml"
    path.write_text(Path(source("config", "resource_governor.yaml")).read_text(encoding="utf-8"), encoding="utf-8")
    current = rg.load_config(path)

    def write(changes):
        return routes.config_write_payload(current, changes, path, lambda: None, lambda applied: None)

    write({
        "queue.max_bypass": 0,
        "background_gate.enabled": False,
        "background_gate.in_flight_max_s": 300.0,
        "background_gate.cpu_enter_some_avg10": 20.0,
    })
    written = rg.load_config(path)
    assert written.queue_max_bypass == 0
    assert written.background_gate_enabled is False
    assert (written.background_gate_in_flight_max_s, written.background_gate_cpu_enter) == (300.0, 20.0)
    before = path.read_bytes()
    for refused in (
        {"queue.max_bypass": -1},
        {"background_gate.cpu_enter_some_avg10": 101.0},
        {"background_gate.cpu_exit_some_avg10": 30.0},
        {"classes": {}},
    ):
        with pytest.raises(routes.ConfigWriteError):
            write(refused)
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# The warm-up, first background caller
# ---------------------------------------------------------------------------


class _Backend:
    """An engine head that records each call and the ticket it saw."""

    name = "ollama"

    def __init__(self, rg):
        self._rg = rg
        self.calls = []

    def generate(self, model, messages, options=None, keep_alive=None):
        ticket = self._rg.get_active_ticket()
        self.calls.append((model, getattr(ticket, "caller", None), getattr(ticket, "admission_class", None)))
        return {"message": {"content": ""}}

    def loaded_models(self):
        return []


def _warm_window():
    holder = {}
    seeded = types.ModuleType(_BACKEND)
    seeded.get_backend_registry = lambda: types.SimpleNamespace(
        resolve_backend=lambda model: holder["backend"], active=holder["backend"]
    )
    loaded = _open((_WARMUP, source("model_warmup.py")), seeded={_BACKEND: seeded})
    rg, warm = loaded[_RG], loaded[_WARMUP]
    holder["backend"] = _Backend(rg)
    return rg, warm, holder["backend"]


def _bounded(call, timeout=3.0):
    """Run ``call`` on its own thread: (finished, result)."""
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("result", call()), daemon=True)
    thread.start()
    thread.join(timeout)
    return not thread.is_alive(), out.get("result")


def test_gc32_the_warmup_asks_as_the_background_and_sends_nothing_while_held():
    """GC32 -- with a chat in flight the warm-up is refused by the gate and
    the engine is never asked; once released, the engine is asked once,
    under the warm-up's own background ticket. Warming up without asking the
    governor -> RED."""
    rg, warm, backend = _warm_window()
    gov, clk = _governor(rg, _config(rg, capacity=24.0, weights={"m": 1.0}))
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    rg.set_active_ticket(gov.admit("m", None, caller="chat"))
    try:
        finished, held = _bounded(lambda: warm.ModelWarmup().warmup("m", timeout=0.0))
    finally:
        rg.clear_active_ticket()
        _release(gov, clk)
    assert finished is True
    assert held.success is False
    assert "interactive_in_flight" in (held.error or "")
    assert backend.calls == []
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    finished, sent = _bounded(lambda: warm.ModelWarmup().warmup("m", timeout=0.0))
    assert finished is True and sent.success is True
    assert backend.calls == [("m", "warmup", "background")]


def test_gc33_the_keepalive_ping_asks_as_the_background_and_never_waits():
    """GC33 -- held by a chat in flight, the ping returns False at once,
    though the background may wait two minutes in the queue, and sends
    nothing; released, it is sent under a background ticket. Pinging
    through the waiting entry -> RED (the ping is still waiting)."""
    rg, warm, backend = _warm_window()
    cfg = _config(rg, capacity=24.0, weights={"m": 1.0}, class_wait_s={"background": 120.0})
    gov, clk = _governor(rg, cfg)
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    rg.set_active_ticket(gov.admit("m", None, caller="chat"))
    try:
        finished, pinged = _bounded(lambda: warm.ModelWarmup().send_keepalive("m"))
    finally:
        rg.clear_active_ticket()
        _release(gov, clk)
    assert finished is True
    assert pinged is False
    assert backend.calls == []
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    finished, pinged = _bounded(lambda: warm.ModelWarmup().send_keepalive("m"))
    assert finished is True and pinged is True
    assert backend.calls == [("m", "warmup", "background")]


# ---------------------------------------------------------------------------
# The loads admitted and not yet seen
# ---------------------------------------------------------------------------


class _Loaded:
    """The warm-up as the snapshot reads it: a keep_alive and the ps view."""

    def __init__(self, *views):
        self.keep_alive = "10m"
        self.views = list(views)

    def get_loaded_models(self):
        return list(self.views)


def _ps(name, gb, ctx):
    """One ps entry, never idle, the refresh turns into a resident view."""
    return types.SimpleNamespace(
        name=name, size_vram=int(gb * _GIB), size=int(gb * _GIB), expires_at=None, context_length=ctx, digest=None
    )


def _pending_models(gov):
    return [p["model"] for p in gov.pending_loads()]


def test_gc35_the_background_counts_every_load_admitted_and_not_yet_seen_and_the_foreground_does_not():
    """GC35 -- on a 16 GiB card (14.5 free after the margin), a chat load of
    5 GiB, its call over, and a background load of 5 GiB are admitted and
    not yet in the loaded view: a third background load of 5 GiB no longer
    fits what is free, while a user is still granted it. Pricing the
    background against the loaded view alone -> RED (both background loads
    take the same memory)."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=16.0, weights={"a": 4.0, "b": 4.0, "c": 4.0}))
    gov._snapshot = _snapshot(rg, clk, capacity=16.0)
    chat = gov.admit("a", 2048, caller="chat")
    gov.note_held(chat)
    gov.note_held(None)
    assert chat.load_expected is True
    assert gov.admit("b", 2048, caller="warmup").admitted is True
    third = gov.admit("c", 2048, caller="warmup")
    assert third.admitted is False
    assert third.reason == "vram_insufficient"
    assert gov.admit("c", 2048, caller="benchmark").admitted is True


def test_gc36_a_load_ends_once_a_refreshed_view_shows_the_model_at_its_context():
    """GC36 -- a background load of 5 GiB at 2048 tokens stays pending while
    the refreshed view shows the model at another context, and ends once a
    refresh shows it at its own; later, the model gone, its memory is free
    again for the background. Never ending a load the view shows -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=10.0, weights={"a": 4.0, "b": 4.0}))
    warm = _Loaded()
    gov._warmup = warm
    gov._snapshot = _snapshot(rg, clk, capacity=10.0)
    assert gov.admit("a", 2048, caller="warmup").load_expected is True
    warm.views = [_ps("a", 5.0, 1024)]
    gov.refresh(force=True)
    assert _pending_models(gov) == ["a"]
    warm.views = [_ps("a", 5.0, 2048)]
    gov.refresh(force=True)
    assert _pending_models(gov) == []
    warm.views = []
    gov.refresh(force=True)
    assert gov.admit("b", 2048, caller="warmup").admitted is True


def test_gc37_a_load_whose_call_has_ended_ends_when_a_later_view_does_not_show_it():
    """GC37 -- a chat load the engine never completed: while its call holds
    the ticket, a view without the model proves nothing; once the call has
    released it, the next view without the model ends the load and frees
    its memory for the background. Waiting for the model to appear -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=10.0, weights={"a": 4.0, "b": 4.0}))
    gov._warmup = _Loaded()
    gov._snapshot = _snapshot(rg, clk, capacity=10.0)
    chat = gov.admit("a", 2048, caller="chat")
    gov.note_held(chat)
    gov.refresh(force=True)
    assert _pending_models(gov) == ["a"]
    gov.note_held(None)
    clk.t += 1.0
    gov.refresh(force=True)
    assert _pending_models(gov) == []
    assert gov.admit("b", 2048, caller="warmup").admitted is True


def test_gc38_a_load_the_engine_gate_refuses_ends_at_once():
    """GC38 -- a user's split admission that the calling engine cannot place
    is refused at the engine gate: its load never happens and no longer
    holds memory. Leaving the refused load pending -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=8.0, weights={"m": 10.0}))
    gov._snapshot = _snapshot(rg, clk, capacity=8.0)
    split = gov.admit("m", None, caller="benchmark")
    assert split.partial_offload is True and split.num_gpu is None
    assert _pending_models(gov) == ["m"]
    with pytest.raises(rg.GovernorRefusal):
        rg._refuse_split(gov, split, "llama.cpp")
    assert _pending_models(gov) == []


def test_gc39_a_load_older_than_its_bound_ends_and_holds_nothing():
    """GC39 -- a user load that never shows in the loaded view holds the
    background's memory until background_gate.pending_load_max_s, then
    ends: the background is admitted. A load that never ends -> RED."""
    rg = _rg()
    cfg = _config(rg, capacity=10.0, weights={"a": 4.0, "b": 4.0}, background_gate_pending_load_max_s=60.0)
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=10.0)
    gov.admit("a", 2048, caller="benchmark")
    assert gov.admit("b", 2048, caller="warmup").admitted is False
    clk.t += 61.0
    gov._snapshot = _snapshot(rg, clk, capacity=10.0)
    assert gov.admit("b", 2048, caller="warmup").admitted is True
    assert _pending_models(gov) == ["b"]


def test_gc40_a_load_admitted_while_a_background_decision_runs_makes_it_try_again():
    """GC40 -- a user load admitted while a background admission is being
    decided, after it read the loads not yet seen: the background refuses
    itself by a reason waiting lifts, and claims nothing. Claiming the load
    without checking the loads again -> RED (both take the same memory)."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=10.0, weights={"a": 4.0, "b": 4.0}))
    gov._snapshot = _snapshot(rg, clk, capacity=10.0)
    real = gov.get_snapshot_fast
    raced = []

    def get_snapshot_fast():
        if not raced:
            raced.append(None)
            raced[0] = gov.admit("a", 2048, caller="benchmark")
        return real()

    gov.get_snapshot_fast = get_snapshot_fast
    background = gov.admit("b", 2048, caller="warmup")
    assert raced[0].admitted is True
    assert background.admitted is False
    assert background.reason == "background_pending_changed"
    assert _pending_models(gov) == ["a"]


def test_gc41_a_background_call_on_a_model_still_loading_joins_its_load_and_loads_nothing():
    """GC41 -- a user load of m at 8192 tokens not yet seen: a background
    call on m naming no context, or no more than 8192, is admitted at 8192
    and loads nothing; one asking more waits for the load, by a reason
    waiting lifts. Loading m a second time for the background -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=24.0, weights={"m": 2.0}))
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    assert gov.admit("m", 8192, caller="benchmark").load_expected is True
    for asked in (None, 4096, 8192):
        joined = gov.admit("m", asked, caller="warmup")
        assert (joined.admitted, joined.load_expected, joined.num_ctx) == (True, False, 8192), asked
    more = gov.admit("m", 16384, caller="warmup")
    assert more.admitted is False
    assert more.reason == "background_load_pending"


def test_gc42_an_interactive_admission_holds_the_background_before_its_ticket_is_held():
    """GC42 -- a chat admitted whose ticket no thread holds yet (the stream
    takes it on its own thread) holds the background by name. Counting only
    the tickets held -> RED (the background slips in)."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=24.0, weights={"m": 1.0, "n": 1.0}))
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    assert gov.admit("m", 2048, caller="chat").admitted is True
    background = gov.admit("n", 2048, caller="warmup")
    assert background.admitted is False
    assert background.held_by == "interactive_admitted"


def test_gc43_once_its_ticket_is_held_and_released_an_interactive_admission_holds_nothing():
    """GC43 -- the chat's ticket held moves the admission to the calls in
    flight, which hold the background; released, the gate opens at once,
    well within the grace an unheld admission is given. Keeping the
    admission once its ticket is held -> RED (the gate stays held)."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=24.0, weights={"m": 1.0, "n": 1.0}))
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    chat = gov.admit("m", 2048, caller="chat")
    gov.note_held(chat)
    assert gov.admit("n", 2048, caller="warmup").held_by == "interactive_in_flight"
    gov.note_held(None)
    assert gov.admit("n", 2048, caller="warmup").admitted is True


def test_gc44_an_interactive_admission_never_held_stops_holding_after_its_grace():
    """GC44 -- a chat admitted and never held (a call that admits without
    taking its ticket) holds the background for
    background_gate.admitted_grace_s, then no longer. An unheld admission
    holding without end -> RED."""
    rg = _rg()
    cfg = _config(rg, capacity=24.0, weights={"m": 1.0, "n": 1.0}, background_gate_admitted_grace_s=5.0)
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    gov.admit("m", 2048, caller="chat")
    clk.t += 4.0
    assert gov.admit("n", 2048, caller="warmup").held_by == "interactive_admitted"
    clk.t += 2.0
    assert gov.admit("n", 2048, caller="warmup").admitted is True


# ---------------------------------------------------------------------------
# What waiting cannot lift, and the machine without a card
# ---------------------------------------------------------------------------


def test_gc45_a_refusal_no_wait_can_lift_is_returned_at_once_and_never_queued():
    """GC45 -- with the card unreadable, or the model's cost unknown, a
    background caller that may wait two minutes is refused at once, by
    name, and never enters the queue. Queueing them -> RED (the caller
    waits out its bound)."""
    rg = _rg()
    cfg = _config(rg, capacity=None, weights={"m": 2.0}, class_wait_s={"background": 120.0})
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=None)
    try:
        blind_done, blind = _bounded(lambda: gov.admit_or_wait("m", 2048, caller="warmup"))
        gov._snapshot = _snapshot(rg, clk, capacity=24.0)
        unpriced_done, unpriced = _bounded(lambda: gov.admit_or_wait("never-measured", 2048, caller="warmup"))
    finally:
        _release(gov, clk)
    assert (blind_done, unpriced_done) == (True, True)
    assert blind.reason == "background_capacity_unknown"
    assert unpriced.reason == "background_cost_unknown"
    assert _queued(gov) == []


def test_gc46_on_a_machine_known_to_have_no_card_the_background_loads_into_free_ram_only():
    """GC46 -- the profile knows the machine has no card: a background load,
    weights and context both, is admitted when the RAM free above the
    reserve, less the loads not yet seen, holds it, and refused by a reason
    waiting lifts when it does not. Refusing every load there -> RED, and
    so is pricing the RAM without the loads not yet seen."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=None, weights={"a": 4.0, "b": 4.0}))
    gov._snapshot = rg.ResourceSnapshot(
        taken_at=clk(), ttl_s=9999.0, capacity_gb=None, ram_available_mb=13.0 * 1024, cards_absent=True
    )
    first = gov.admit("a", 2048, caller="warmup")
    assert (first.admitted, first.load_expected) == (True, True)
    second = gov.admit("b", 2048, caller="warmup")
    assert second.admitted is False
    assert second.reason == "ram_insufficient"


# ---------------------------------------------------------------------------
# The queue's turns, deadline and records
# ---------------------------------------------------------------------------


def test_gc47_two_callers_trying_at_once_never_pass_a_waiter_past_its_allowance():
    """GC47 -- with an allowance of one, a user call that fits passes the
    waiting head; another arriving while the first is still being decided
    may not try, and is refused. Checking the allowance and charging the
    pass in two steps -> RED (the head is passed twice)."""
    rg = _rg()
    cfg = _config(
        rg, capacity=24.0, weights={"m": 1.0}, queue_enabled_per_caller={"benchmark": True},
        queue_wait_s=0.0, queue_max_bypass=1,
    )
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    head = gov._enqueue("user")
    real = gov.admit
    inner = []

    def admit(model, requested_ctx=None, caller="chat", **kw):
        if not inner:
            inner.append(None)
            inner[0] = gov.admit_or_wait("m", caller="benchmark")
        return real(model, requested_ctx, caller=caller, **kw)

    gov.admit = admit
    assert gov.admit_or_wait("m", caller="benchmark").admitted is True
    assert inner[0].admitted is False
    assert head.bypassed == 1


def test_gc48_a_caller_refused_after_trying_gives_back_the_pass_it_took():
    """GC48 -- with an allowance of one, a user call too large to fit tries
    past the waiting head and is refused: the head has not been passed, so
    the next call that fits may still pass it. Keeping the pass of a
    refused try -> RED."""
    rg = _rg()
    cfg = _config(
        rg, capacity=24.0, weights={"m": 1.0, "big": 50.0}, offload_enabled=False,
        queue_enabled_per_caller={"benchmark": True}, queue_wait_s=0.0, queue_max_bypass=1,
    )
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    head = gov._enqueue("user")
    assert gov.admit_or_wait("big", caller="benchmark").admitted is False
    assert head.bypassed == 0
    assert gov.admit_or_wait("m", caller="benchmark").admitted is True
    assert head.bypassed == 1


def test_gc49_a_waiters_deadline_is_set_when_it_enters_the_queue_not_after_its_record():
    """GC49 -- a ring write that takes ten seconds of the clock does not
    lengthen a five-second wait: the deadline is set as the waiter enters
    the queue, so it leaves at its first wake. Setting it after the write
    -> RED (the waiter stays)."""
    rg = _rg()
    cfg = _config(rg, capacity=8.0, weights={"big": 50.0}, offload_enabled=False, class_wait_s={"background": 120.0})
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=8.0)
    real = gov.record_decision

    def record_decision(caller, model, requested_ctx, num_ctx, action, reason, *rest, **kw):
        if action == "queue":
            clk.t += 10.0
        return real(caller, model, requested_ctx, num_ctx, action, reason, *rest, **kw)

    gov.record_decision = record_decision
    thread, out = _waiting(gov, "big", "warmup", requested_ctx=2048, wait_s=5.0)
    try:
        thread.join(3.0)
        stayed = thread.is_alive()
    finally:
        _release(gov, clk, thread)
    assert stayed is False
    assert out["decision"].admitted is False


def test_gc50_a_waiters_retries_are_silent_and_only_its_entry_and_outcome_are_recorded():
    """GC50 -- a user caller enrolled in the queue retries at every wake;
    the ring and the refusal window keep its first refusal, its entry in
    the queue and its outcome, however many retries it made. Recording
    each retry -> RED (one waiter fills both)."""
    rg = _rg()
    # The refusal window outlasts the release's jump of the clock, which
    # would otherwise age the first refusal out of it.
    cfg = _config(
        rg, capacity=8.0, weights={"big": 50.0}, offload_enabled=False,
        queue_enabled_per_caller={"benchmark": True}, queue_wait_s=60.0, pressure_refusal_window_s=1.0e7,
    )
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=8.0)
    attempts = _record_attempts(gov)
    thread, _out = _waiting(gov, "big", "benchmark")
    try:
        for n in (2, 3, 4):
            assert _until(lambda: len(attempts) >= n or gov._notify_queue())
    finally:
        _release(gov, clk, thread)
    rows = [d for d in reversed(gov.store.recent_decisions(50)) if d["caller"] == "benchmark"]
    assert [d["decision"] for d in rows] == ["refuse", "queue", "refuse"]
    assert len(gov._refusal_events) == 2


def test_gc51_a_user_waiting_in_the_queue_holds_a_plain_background_admission():
    """GC51 -- with the user class enrolled in the queue and a user caller
    waiting, a background call through plain admit is held by name instead
    of taking memory the user waits for. Holding only for the interactive
    class -> RED."""
    rg = _rg()
    cfg = _config(
        rg, capacity=24.0, weights={"m": 1.0}, class_queued={"interactive": False, "user": True, "background": True}
    )
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    gov._enqueue("user")
    held = gov.admit("m", 2048, caller="warmup")
    assert held.admitted is False
    assert held.held_by == "user_queued"


def test_gc52_no_caller_passes_a_waiter_of_a_higher_class_and_the_gate_says_why():
    """GC52 -- with a user waiting, a background call that fits does not even
    try: it waits out its bound behind the user and leaves by the gate's
    reason, the user waiting. A background waiter blocks no user. Letting a
    lower class try past a higher one -> RED. Supersedes GC25, whose expiry
    reason predates the gate holding for every higher class."""
    rg = _rg()
    cfg = _config(
        rg, capacity=24.0, weights={"m": 1.0}, queue_enabled_per_caller={"benchmark": True},
        queue_wait_s=0.0, class_wait_s={"background": 0.0},
    )
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    attempts = _record_attempts(gov)
    waiting_user = gov._enqueue("user")
    background = gov.admit_or_wait("m", 2048, caller="warmup")
    assert background.admitted is False
    assert background.reason == "background_held:user_queued"
    assert attempts == []
    gov._dequeue(waiting_user)
    gov._enqueue("background")
    assert gov.admit_or_wait("m", caller="benchmark").admitted is True
    assert attempts == ["benchmark"]


def test_gc53_a_background_call_asking_more_than_a_resident_holds_is_served_by_it_when_it_may():
    """GC53 -- a resident holding 4096 tokens on a 6 GiB card: a background
    call asking 16384, which the dynamic stage brings down to 4096 for want
    of free memory, is served by the resident at its context, as any class
    is; so is a background caller whose floor the resident's context meets.
    Refusing the background before the dynamic stage -> RED."""
    rg = _rg()
    cfg = _config(rg, capacity=6.0, dynamic_ctx_enabled=True, ctx_ladder=[16384, 8192, 4096])
    gov, clk = _governor(rg, cfg)
    held = rg.LoadedModelView(name="m", size_vram_bytes=4 * _GIB, context_length=4096)
    gov._snapshot = _snapshot(rg, clk, capacity=6.0, in_use=4.0, loaded=[held])
    served = gov.admit("m", 16384, caller="warmup")
    assert (served.admitted, served.load_expected, served.num_ctx) == (True, False, 4096)
    cfg.dynamic_ctx_enabled = False
    cfg.ctx_floor = {"warmup": 4096}
    floored = gov.admit("m", 16384, caller="warmup")
    assert (floored.admitted, floored.load_expected, floored.num_ctx) == (True, False, 4096)


# ---------------------------------------------------------------------------
# Contracts superseded by the loads not yet seen and the gate's new holds
# ---------------------------------------------------------------------------


def test_gc54_the_background_never_gets_a_grant_conditional_on_an_eviction():
    """GC54 -- the frame where chat is granted CONDITIONAL on evicting an
    idle resident (budget 0.5 now, 4.5 with the eviction): a background load
    of another model of the same cost is refused, with no conditional flag:
    it counts no eviction credit. Giving the background the eviction credit
    -> RED. Supersedes GC4, whose background asked the chat's own model and
    now joins the chat's load instead of placing its own."""
    rg = _rg()
    cfg = _config(rg, weights={"m": 3.0, "n": 3.0}, idle_evict_threshold_s=600.0, offload_enabled=False)
    gov, clk = _governor(rg, cfg, warmup=_warmup())
    gov._snapshot = _snapshot(rg, clk, in_use=8.0, loaded=[_idle_view(rg, "resident", 4.0)])
    background = gov.admit("n", 2048, caller="warmup")
    assert background.admitted is False
    assert background.conditional_on_eviction is False
    chat = gov.admit("m", 2048, caller="chat")
    assert chat.admitted is True and chat.conditional_on_eviction is True


def test_gc55_the_background_splits_between_vram_and_ram_only_when_allowed():
    """GC55 -- a 10 GiB model on an 8 GiB card is admitted split for a user;
    on a governor of its own the background is refused the same split until
    the file allows it. Ignoring allow_split (either way) -> RED. Supersedes
    GC6, whose background followed the user's admission of the same model
    and now joins that load instead of placing its own."""
    rg = _rg()
    users, clk = _governor(rg, _config(rg, capacity=8.0, weights={"m": 10.0}))
    users._snapshot = _snapshot(rg, clk, capacity=8.0)
    user = users.admit("m", 1024, caller="benchmark")
    assert user.admitted is True and user.partial_offload is True
    cfg = _config(rg, capacity=8.0, weights={"m": 10.0})
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=8.0)
    assert gov.admit("m", 1024, caller="warmup").admitted is False
    cfg.background_allow_split = True
    allowed = gov.admit("m", 1024, caller="warmup")
    assert allowed.admitted is True and allowed.partial_offload is True


def test_gc56_holding_and_releasing_a_ticket_count_the_calls_in_flight_per_thread_and_class():
    """GC56 -- set_active_ticket, ticket_scope and clear_active_ticket feed
    the governor's count of calls in flight: one ticket per thread, counted
    by class, back to zero once released. Not feeding the registry -> RED.
    Supersedes GC14, whose background was admitted after a chat admission
    no thread held yet, which now holds the gate."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=24.0, weights={"m": 1.0}))
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    background = gov.admit("m", 2048, caller="warmup")
    chat = gov.admit("m", None, caller="chat")
    user = gov.admit("m", None, caller="benchmark")
    assert gov.in_flight_by_class() == _ZERO
    rg.set_active_ticket(chat)
    assert gov.in_flight_by_class() == {**_ZERO, "interactive": 1}
    with rg.ticket_scope(background):
        assert gov.in_flight_by_class() == {**_ZERO, "background": 1}
    assert gov.in_flight_by_class() == {**_ZERO, "interactive": 1}
    held, done = threading.Event(), threading.Event()

    def other():
        rg.set_active_ticket(user)
        held.set()
        done.wait(5.0)
        rg.clear_active_ticket()

    thread = threading.Thread(target=other, daemon=True)
    thread.start()
    assert held.wait(5.0)
    assert gov.in_flight_by_class() == {**_ZERO, "interactive": 1, "user": 1}
    done.set()
    thread.join(5.0)
    assert gov.in_flight_by_class() == {**_ZERO, "interactive": 1}
    rg.clear_active_ticket()
    assert gov.in_flight_by_class() == _ZERO


def test_gc57_a_waiter_is_admitted_when_memory_frees_and_its_slot_is_given_back():
    """GC57 -- a background call refused for memory waits; when the card
    frees and the queue is woken it is admitted, and the queue is empty
    after. Not re-running admission on a wake -> RED. Supersedes GC29, whose
    call named no context: the background is now charged the context it
    loads at, which that frame's freed card no longer holds."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=8.0, weights={"m": 5.0}, offload_enabled=False))
    gov._snapshot = _snapshot(rg, clk, capacity=8.0, in_use=4.0)
    thread, out = _waiting(gov, "m", "warmup", requested_ctx=2048)
    try:
        assert _until(lambda: gov.queue_by_class()["background"] == 1)
        gov._snapshot = _snapshot(rg, clk, capacity=8.0, in_use=0.0)
        gov._notify_queue()
        thread.join(5.0)
        stayed = thread.is_alive()
    finally:
        _release(gov, clk, thread)
    assert stayed is False
    assert out["decision"].admitted is True
    assert gov.queue_by_class() == _ZERO


def test_gc58_a_decision_carries_the_class_of_its_caller_and_a_stranger_is_a_user():
    """GC58 -- every decision names its class, in the ticket and in its dict,
    and a caller the table does not name is a user, never the background.
    Defaulting a stranger to the background -> RED. Supersedes GC2, whose
    background asked after a chat admission no thread held yet, which now
    holds the gate."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, weights={"m": 1.0}))
    gov._snapshot = _snapshot(rg, clk)
    seen = {}
    for caller in ("warmup", "chat", "benchmark", "never-named"):
        decision = gov.admit("m", 2048, caller=caller)
        assert decision.admitted is True
        assert decision.to_dict()["admission_class"] == decision.admission_class
        seen[caller] = decision.admission_class
    assert seen == {"chat": "interactive", "benchmark": "user", "warmup": "background", "never-named": "user"}


def test_gc59_the_background_is_refused_a_load_when_the_free_memory_cannot_be_known():
    """GC59 -- with the card's capacity unknown a user is admitted fail-open
    (3.1); the background, loading another model, which must not let the
    engine's own LRU evict for it, is refused by name. Failing the background
    open too -> RED. Supersedes GC7, whose background asked the user's own
    model and now joins the user's load instead of placing its own."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=None, weights={"m": 2.0, "n": 2.0}))
    gov._snapshot = _snapshot(rg, clk, capacity=None)
    user = gov.admit("m", None, caller="benchmark")
    assert user.admitted is True and user.reason.startswith("capacity_unknown_fail_open")
    background = gov.admit("n", None, caller="warmup")
    assert background.admitted is False
    assert background.reason == "background_capacity_unknown"


def test_gc60_the_background_is_refused_a_load_whose_cost_is_unknown():
    """GC60 -- a model nothing can price is never too large for a user (3.1);
    the background, loading another such model, is refused it by name, since
    it cannot show the load fits the free memory. Treating the unknown cost
    as zero for the background -> RED. Supersedes GC9, whose background
    asked the user's own model and now joins the user's load."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=24.0))
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    assert gov.admit("never-measured", None, caller="benchmark").admitted is True
    background = gov.admit("also-never-measured", None, caller="warmup")
    assert background.admitted is False
    assert background.reason == "background_cost_unknown"


# ---------------------------------------------------------------------------
# The context a background load is charged and sent
# ---------------------------------------------------------------------------


def _sent_options(backend):
    """The options of every call the engine head is asked, in order."""
    sent = []
    real = backend.generate

    def generate(model, messages, options=None, keep_alive=None):
        sent.append(dict(options or {}))
        return real(model, messages, options=options, keep_alive=keep_alive)

    backend.generate = generate
    return sent


def test_gc61_the_warmup_sends_the_context_it_was_admitted_at():
    """GC61 -- the warm-up, told no context, is admitted at the one the
    governor names for the model and sends it, so the engine loads what the
    admission priced. Sending no context (the engine loads at its own
    default, with a KV nobody charged) -> RED."""
    rg, warm, backend = _warm_window()
    gov, clk = _governor(rg, _config(rg, capacity=24.0, weights={"m": 1.0}, ctx_ladder=[8192, 3072]))
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    sent = _sent_options(backend)
    assert warm.ModelWarmup().warmup("m", timeout=0.0).success is True
    assert sent == [{"num_predict": 1, "num_ctx": 3072}]


def test_gc62_the_keepalive_ping_renews_a_resident_at_the_context_it_holds():
    """GC62 -- a resident holding 8192 tokens: the ping is admitted at that
    context and sends it, so the engine renews the model as it is instead of
    loading it again at its own default. Pinging with no context -> RED."""
    rg, warm, backend = _warm_window()
    gov, clk = _governor(rg, _config(rg, capacity=24.0))
    rg._governor = gov
    held = rg.LoadedModelView(name="m", size_vram_bytes=4 * _GIB, context_length=8192)
    gov._snapshot = _snapshot(rg, clk, capacity=24.0, in_use=4.0, loaded=[held])
    sent = _sent_options(backend)
    assert warm.ModelWarmup().send_keepalive("m") is True
    assert sent == [{"num_predict": 0, "num_ctx": 8192}]


def test_gc63_a_background_load_told_no_context_is_charged_the_one_it_will_load_at():
    """GC63 -- told no context, a background load is priced and admitted at
    the context the interactive class was last admitted at for the model,
    else at the model's output reserve, else at the ladder's smallest step.
    Forgetting the interactive class's own context -> RED (the warm model
    is loaded again at the first chat)."""
    cm = types.ModuleType(_CM)
    cm.get_model_limits = lambda model: types.SimpleNamespace(
        context_window=32768, max_output=2048 if model == "r" else 0
    )
    rg = _open(seeded={_CM: cm}, blocked=(_ESTOP,))[_RG]
    cfg = _config(rg, capacity=24.0, weights={"c": 1.0, "r": 1.0, "l": 1.0}, ctx_ladder=[8192, 3072])
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    gov.record_decision("chat", "c", 6000, 6144, "admit", "fits")
    decisions = [gov.admit(model, None, caller="warmup") for model in ("c", "r", "l")]
    assert [d.num_ctx for d in decisions] == [6144, 2048, 3072]
    assert [d.cost_gb for d in decisions] == [4.0, 2.0, 2.5]


def test_gc64_a_warmup_whose_admission_fails_sends_nothing():
    """GC64 -- a governor whose admission raises: the warm-up and the ping
    send nothing and say why, instead of reaching the engine without a
    ticket, where the backstop would admit them as a user that may evict.
    Loading without a ticket -> RED."""
    rg, warm, backend = _warm_window()

    class _Broken:
        def admit_or_wait(self, *args, **kwargs):
            raise RuntimeError("store unreadable")

        admit = admit_or_wait

    rg._governor = _Broken()
    result = warm.ModelWarmup().warmup("m", timeout=0.0)
    assert result.success is False
    assert "store unreadable" in (result.error or "")
    assert warm.ModelWarmup().send_keepalive("m") is False
    assert backend.calls == []


def test_gc65_the_resume_answers_at_once_and_its_warmup_waits_on_its_own_thread():
    """GC65 -- resuming after an emergency stop asks the warm-up of the model
    it names on a thread of its own: the resume answers at once while the
    warm-up waits its turn in the queue, and the warm-up still runs.
    Warming up on the resume's own thread -> RED (the resume waits with
    it)."""
    estop = _open((_ESTOP, source("emergency_stop.py")), blocked=(_CM,))[_ESTOP]
    started, release = threading.Event(), threading.Event()

    class _Warmer:
        def warmup(self, model):
            started.set()
            release.wait(5.0)
            return types.SimpleNamespace(success=True)

    backend = types.SimpleNamespace(name="ollama", health_check=lambda: True)
    estop._resolve_backend_registry = lambda: types.SimpleNamespace(active=backend)
    estop._resolve_warmup = lambda: _Warmer()
    try:
        finished, out = _bounded(lambda: estop._step_reconnect_ollama("m"), timeout=2.0)
        ran = started.wait(5.0)
    finally:
        release.set()
    assert (finished, ran) == (True, True)
    assert out["warmup"]["model"] == "m"


# ---------------------------------------------------------------------------
# A configuration write keeps the live state
# ---------------------------------------------------------------------------


def test_gc66_a_configuration_write_rebuilds_the_governor_with_its_live_state():
    """GC66 -- the config route rebuilds the governor from the file at once
    and hands the new one the live state: the chat in flight still holds the
    background, the background's guest is still marked, the waiter is still
    queued, the loads not yet seen still count, and the chat's release,
    through the new governor, ends its call. Dropping the governor for a
    fresh one -> RED (the chat in flight is forgotten mid-call)."""
    loaded = _open((_ROUTES, source("api", "routes_governor.py")), packages=("opti_oignon.api",))
    rg, routes = loaded[_RG], loaded[_ROUTES]
    tmp = Path(tempfile.mkdtemp())
    path = tmp / "resource_governor.yaml"
    path.write_text(Path(source("config", "resource_governor.yaml")).read_text(encoding="utf-8"), encoding="utf-8")
    rg._DEFAULT_CONFIG_PATH = path
    rg._DEFAULT_DB_PATH = tmp / "fresh.db"
    # Built on the file the route writes, as the governor of the process is.
    clk = _Clock()
    gov = rg.ResourceGovernor(
        config_path=str(path), db_path=str(tmp / "governor.db"), warmup=None, registry=None,
        clock=clk, meminfo_path="/nonexistent-meminfo", hardware=None,
    )
    gov._config = _config(rg, capacity=10.0, weights={"a": 4.0, "b": 4.0})
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=10.0)
    assert gov.admit("b", 2048, caller="warmup").load_expected is True
    gov.note_loaded_by("b", "background")
    rg.set_active_ticket(gov.admit("a", 2048, caller="chat"))
    gov._enqueue("user")
    write = next(r.endpoint for r in routes.router.routes if r.path == "/api/governor/config" and "POST" in r.methods)
    try:
        write({"queue.max_bypass": 1})
        new = rg.get_resource_governor()
        kept = (
            new is not gov,
            new.config.queue_max_bypass,
            new.in_flight_by_class()["interactive"],
            new.background_gate_state()["reason"],
            new.queue_by_class()["user"],
            _pending_models(new),
            new._owners["b"][0],
        )
    finally:
        rg.clear_active_ticket()
    assert kept == (True, 1, 1, "interactive_in_flight", 1, ["b", "a"], "background")
    assert new.in_flight_by_class()["interactive"] == 0


# ---------------------------------------------------------------------------
# The file's classes and bounds, and what no contract held yet
# ---------------------------------------------------------------------------


def test_gc67_a_class_the_file_names_as_no_string_is_refused_and_the_file_still_loads():
    """GC67 -- a caller the file moves to a list or a mapping instead of a
    class name is refused with a warning and keeps its class, the rest of
    the table applies, and the file loads. Testing the name against the
    classes before its type -> RED (the load raises, or the table is lost)."""
    rg = _rg()
    path = Path(tempfile.mkdtemp()) / "resource_governor.yaml"
    path.write_text(
        "classes:\n  callers:\n    warmup: [user]\n    stranger: {class: interactive}\n    nightly: background\n",
        encoding="utf-8",
    )
    cfg = rg.load_config(path)
    assert [cfg.class_of(c) for c in ("warmup", "stranger", "nightly")] == ["background", "user", "background"]


def test_gc68_the_background_is_refused_a_load_whose_extra_model_nothing_can_price():
    """GC68 -- a background load of a priced model with an extra model (a
    draft) nothing can price is refused by name, as an unpriced model is,
    and never waits. Pricing the unknown extra as nothing -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=24.0, weights={"m": 1.0}))
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    refused = gov.admit("m", 2048, caller="warmup", extra_models=["never-measured-draft"])
    assert refused.admitted is False
    assert refused.reason == "background_cost_unknown"


def test_gc69_a_waiter_admitted_on_a_retry_passes_the_waiters_of_its_class_ahead_of_it():
    """GC69 -- a user waiter behind a head of its class is admitted once
    memory frees: the head is charged the pass, as for a newcomer. Charging
    only the newcomers' passes -> RED."""
    rg = _rg()
    cfg = _config(
        rg, capacity=8.0, weights={"m": 5.0}, offload_enabled=False,
        queue_enabled_per_caller={"benchmark": True}, queue_wait_s=60.0, queue_max_bypass=2,
    )
    gov, clk = _governor(rg, cfg)
    gov._snapshot = _snapshot(rg, clk, capacity=8.0, in_use=4.0)
    head = gov._enqueue("user")
    thread, out = _waiting(gov, "m", "benchmark", requested_ctx=2048)
    try:
        assert _until(lambda: gov.queue_by_class()["user"] == 2)
        gov._snapshot = _snapshot(rg, clk, capacity=8.0, in_use=0.0)
        gov._notify_queue()
        thread.join(5.0)
        stayed = thread.is_alive()
    finally:
        _release(gov, clk, thread)
    assert stayed is False
    assert out["decision"].admitted is True
    assert head.bypassed == 1


def test_gc70_the_gates_bounds_are_written_in_range_only_and_a_zero_bound_is_refused():
    """GC70 -- the grace an unheld admission holds the gate and the bound of
    a load not yet seen are written to the file and read back; a negative
    grace, or a load or in-flight bound of zero, is refused at the route,
    the file left as it was, and a zero bound in the file keeps its default.
    Accepting an in-flight bound of zero -> RED (it ends the hold of every
    call in flight at once)."""
    loaded = _open((_ROUTES, source("api", "routes_governor.py")), packages=("opti_oignon.api",))
    rg, routes = loaded[_RG], loaded[_ROUTES]
    tmp = Path(tempfile.mkdtemp())
    path = tmp / "resource_governor.yaml"
    path.write_text(Path(source("config", "resource_governor.yaml")).read_text(encoding="utf-8"), encoding="utf-8")
    current = rg.load_config(path)

    def write(changes):
        return routes.config_write_payload(current, changes, path, lambda: None, lambda applied: None)

    write({"background_gate.admitted_grace_s": 3.0, "background_gate.pending_load_max_s": 120.0})
    written = rg.load_config(path)
    assert (written.background_gate_admitted_grace_s, written.background_gate_pending_load_max_s) == (3.0, 120.0)
    before = path.read_bytes()
    for refused in (
        {"background_gate.admitted_grace_s": -1.0},
        {"background_gate.pending_load_max_s": 0.0},
        {"background_gate.in_flight_max_s": 0.0},
    ):
        with pytest.raises(routes.ConfigWriteError):
            write(refused)
    assert path.read_bytes() == before
    zero = tmp / "zero.yaml"
    zero.write_text("background_gate:\n  in_flight_max_s: 0\n  pending_load_max_s: 0\n", encoding="utf-8")
    kept = rg.load_config(zero)
    assert (kept.background_gate_in_flight_max_s, kept.background_gate_pending_load_max_s) == (900.0, 600.0)


# ---------------------------------------------------------------------------
# Across a reload; a model loads once; the background's guests go first
# ---------------------------------------------------------------------------


def _reloaded(rg, gov, cfg, snapshot=None):
    """The governor a configuration write builds from ``gov``, the process's
    governor, with ``cfg`` as the file it read. Built under the queue's lock,
    so no waiter the reload wakes sees the new governor before its file."""
    rg._governor = gov
    with gov._queue_cond:
        new = rg.reload_resource_governor()
        new._config = cfg
        if snapshot is not None:
            new._snapshot = snapshot
    return new


def _memory_reading(value):
    return {"memory": {"some": {"avg10": value}}}


def test_gc71_a_waiter_that_never_tried_is_refused_by_the_governor_a_reload_built():
    """GC71 -- a background waiter, held by a user waiting ahead of it, never
    gets a turn; meanwhile a reload reads a file that switches the gate off.
    At the end of its wait it is refused as queue_wait_expired, not held,
    and the refusal is recorded by the governor the process now asks.
    Reading the gate through the replaced governor -> RED (held:
    user_queued); recording through it -> RED (the current governor never
    records it)."""
    rg = _rg()
    cfg = dict(capacity=24.0, weights={"m": 1.0})
    gov, clk = _governor(rg, _config(rg, **cfg))
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    gov._enqueue("user")
    thread, out = _waiting(gov, "m", "warmup")
    recorded = []
    try:
        assert _until(lambda: gov.queue_by_class()["background"] == 1)
        new = _reloaded(rg, gov, _config(rg, background_gate_enabled=False, **cfg), _snapshot(rg, clk, capacity=24.0))
        real = new.record_decision

        def record(caller, model, requested_ctx, admitted_ctx, decision, reason=""):
            recorded.append((caller, reason))
            return real(caller, model, requested_ctx, admitted_ctx, decision, reason)

        new.record_decision = record
    finally:
        _release(gov, clk, thread)
    decision = out["decision"]
    assert (decision.admitted, decision.reason, decision.held_by) == (False, "queue_wait_expired", None)
    assert recorded == [("warmup", "queue_wait_expired")]


def test_gc72_a_caller_holding_the_governor_a_reload_replaced_waits_as_the_current_file_says():
    """GC72 -- the file a reload read makes the benchmark wait, the replaced
    one did not: a caller that still asks the governor the reload replaced,
    behind a user head it may not pass, enters the queue and is refused at
    the end of its wait without trying. Reading the class's queueing from
    the replaced governor's file -> RED (admitted at once, past the head)."""
    rg = _rg()
    cfg = dict(capacity=24.0, weights={"m": 1.0}, queue_max_bypass=0)
    gov, clk = _governor(rg, _config(rg, **cfg))
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    gov._enqueue("user")
    new = _reloaded(
        rg, gov, _config(rg, queue_enabled_per_caller={"benchmark": True}, **cfg), _snapshot(rg, clk, capacity=24.0)
    )
    attempts = _record_attempts(new)
    decision = gov.admit_or_wait("m", caller="benchmark", wait_s=0.0)
    assert (decision.admitted, decision.reason, attempts) == (False, "queue_wait_expired", [])
    assert len(_queued(new)) == 1


def test_gc73_a_waiter_tries_by_the_allowance_of_the_file_a_reload_read():
    """GC73 -- a benchmark waits behind a user head it may not pass (an
    allowance of zero); a reload reads an allowance of one: woken, the
    waiter passes the head once and is admitted. Judging its turn by the
    replaced governor's file -> RED (it waits out its wait)."""
    rg = _rg()
    cfg = dict(capacity=24.0, weights={"m": 1.0}, queue_enabled_per_caller={"benchmark": True}, queue_wait_s=60.0)
    gov, clk = _governor(rg, _config(rg, queue_max_bypass=0, **cfg))
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    head = gov._enqueue("user")
    thread, out = _waiting(gov, "m", "benchmark")
    try:
        assert _until(lambda: gov.queue_by_class()["user"] == 2)
        _reloaded(rg, gov, _config(rg, queue_max_bypass=1, **cfg), _snapshot(rg, clk, capacity=24.0))
        thread.join(5.0)
        stayed = thread.is_alive()
    finally:
        _release(gov, clk, thread)
    assert stayed is False
    assert out["decision"].admitted is True
    assert head.bypassed == 1


def test_gc74_a_refusal_before_a_reload_counts_in_the_window_of_the_governor_it_builds():
    """GC74 -- a user refused for memory before a configuration write, and
    one after it, both count in the refusal window of the governor the
    write built: the pressure a run of refusals signals survives the write.
    A window of its own for each governor -> RED (the first refusal is
    forgotten)."""
    rg = _rg()
    cfg = dict(capacity=8.0, weights={"m": 20.0}, offload_enabled=False)
    gov, clk = _governor(rg, _config(rg, **cfg))
    gov._snapshot = _snapshot(rg, clk, capacity=8.0)
    assert gov.admit("m", 2048, caller="benchmark").admitted is False
    new = _reloaded(rg, gov, _config(rg, **cfg), _snapshot(rg, clk, capacity=8.0))
    assert new.admit("m", 2048, caller="benchmark").admitted is False
    assert new._refusal_window_stats()[1:] == (2, 2)


def test_gc75_a_reload_under_sustained_pressure_keeps_its_timer_and_the_keep_alive_to_restore():
    """GC75 -- soft pressure begins, a reload comes before it has lasted
    pressure_sustain_s, and the governor the reload built shortens the
    warm-up's keep_alive once it has; a second reload, then the pressure
    clears, and the keep_alive the pressure replaced is restored. Starting
    the timer again at a reload -> RED (nothing shortened); forgetting the
    keep_alive to restore -> RED (left short until a restart)."""
    rg = _rg()
    warmup = _warmup("10m")
    cfg = dict(capacity=10.0, pressure_sustain_s=10.0, pressure_keep_alive="1m")
    gov, clk = _governor(rg, _config(rg, **cfg), warmup=warmup)
    gov._snapshot = _snapshot(rg, clk, capacity=10.0, in_use=9.0)
    assert gov.pressure_state()["level"] == "soft"
    clk.t += 6.0
    first = _reloaded(rg, gov, _config(rg, **cfg), _snapshot(rg, clk, capacity=10.0, in_use=9.0))
    clk.t += 6.0
    first.pressure_state()
    shortened = warmup.keep_alive
    second = _reloaded(rg, first, _config(rg, **cfg), _snapshot(rg, clk, capacity=10.0, in_use=0.0))
    assert second.pressure_state()["level"] == "none"
    assert (shortened, warmup.keep_alive) == ("1m", "10m")


def test_gc76_a_reload_keeps_the_side_of_each_hysteresis_and_its_marks_judge_the_next_reading():
    """GC76 -- memory pressure entered (12 over its enter mark of 10) and
    the background gate held by the CPU pressure others suffer (12 over
    10); after a reload a reading between the exit and the enter marks
    keeps both (7: still under pressure, still held). A reload whose file
    moves the CPU exit mark above that reading releases the gate at once:
    the reading is taken again and the new marks judge it. Starting the
    memory or the CPU hysteresis again at a reload -> RED; keeping the
    replaced governor's reading -> RED (held until it ages out)."""
    rg = _rg()
    hardware = _Hardware(_pressure(12.0))
    gov, clk = _governor(rg, _config(rg), hardware=hardware)
    assert gov._update_memory_pressure(_memory_reading(12.0)) is True
    assert gov.background_gate_state()["reason"] == "cpu_pressure"
    hardware.reading = _pressure(7.0)
    new = _reloaded(rg, gov, _config(rg))
    kept = (new._update_memory_pressure(_memory_reading(7.0)), new.background_gate_state()["reason"])
    moved = _reloaded(rg, new, _config(rg, background_gate_cpu_exit=8.0))
    assert kept == (True, "cpu_pressure")
    assert moved.background_gate_state()["reason"] is None


def test_gc77_two_loads_of_one_model_not_yet_seen_count_once_for_the_background_at_the_largest():
    """GC77 -- on a 16 GiB card (14.5 free after the margin), two user loads
    of ``a`` not yet seen, at 5 and then 6 GiB, are one load to come: the
    background is refused 9 GiB (8.5 free once ``a`` counts at its
    largest) and granted 5. Counting each load -> RED (5 refused: 3.5
    free); counting the first -> RED (9 granted: 9.5 free)."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=16.0, weights={"a": 4.0, "b": 4.0, "c": 8.0}))
    gov._snapshot = _snapshot(rg, clk, capacity=16.0)
    assert gov.admit("a", 2048, caller="benchmark").load_expected is True
    assert gov.admit("a", 4096, caller="benchmark").load_expected is True
    assert [p["vram_gb"] for p in gov.pending_loads()] == [5.0, 6.0]
    big = gov.admit("c", 2048, caller="warmup")
    assert (big.admitted, big.reason) == (False, "vram_insufficient")
    assert gov.admit("b", 2048, caller="warmup").admitted is True


def test_gc78_with_idle_eviction_off_the_backgrounds_guest_is_still_evicted_first():
    """GC78 -- with the idle threshold switched off (none, or negative) no
    resident is idle enough to evict, yet a resident only the background
    loaded is still a candidate, alone. Gating the background's guests on
    the idle threshold -> RED."""
    rg = _rg()
    found = []
    for threshold in (None, -1.0):
        gov, clk = _governor(rg, _config(rg, capacity=24.0, idle_evict_threshold_s=threshold), warmup=_warmup())
        snapshot = _two_residents(rg, clk)
        before = [c[0] for c in gov._evictable_candidates(snapshot)]
        gov.note_loaded_by("warmed", "background")
        found.append((before, [c[0] for c in gov._evictable_candidates(snapshot)]))
    assert found == [([], ["warmed"]), ([], ["warmed"])]


def test_gc79_what_the_pressure_readings_leave_is_read_and_written_under_the_shared_lock():
    """GC79 -- a governor and the one a reload builds share what the
    pressure readings leave, so both may touch it at once (each builds its
    snapshot under its own lock): every read and write of it, through the
    memory and the CPU hysteresis, the sustained pressure's timer and the
    keep_alive it replaced, holds the cache lock both share. The memory
    hysteresis judged without it -> RED; the status reading the replaced
    keep_alive without it -> RED."""
    rg = _rg()
    warmup = _warmup("10m")
    hardware = _Hardware(_pressure(12.0))
    gov, clk = _governor(
        rg, _config(rg, pressure_sustain_s=10.0, pressure_keep_alive="1m"), hardware=hardware, warmup=warmup
    )
    seen = []
    lock = gov._cache_lock
    fields = {"memory_pressure_active", "cpu_held", "pressure_soft_since", "keep_alive_original"}

    class Watched(rg._SharedPressure):
        def __getattribute__(self, name):
            if name in fields:
                seen.append((name, lock.locked()))
            return object.__getattribute__(self, name)

        def __setattr__(self, name, value):
            seen.append((name, lock.locked()))
            object.__setattr__(self, name, value)

    watched = Watched()
    seen.clear()
    gov._shared = watched
    gov._update_memory_pressure(_memory_reading(12.0))
    gov._update_memory_pressure(_memory_reading(3.0))
    gov.background_gate_state()
    for in_use, step in ((9.0, 0.0), (9.0, 11.0), (0.0, 1.0)):
        clk.t += step
        gov._snapshot = _snapshot(rg, clk, capacity=10.0, in_use=in_use)
        gov.pressure_state()
    assert warmup.keep_alive == "10m"
    assert {name for name, _ in seen} == fields
    assert [entry for entry in seen if not entry[1]] == []


# ---------------------------------------------------------------------------
# GC80-GC81 -- a background guest released at the end of its run
# ---------------------------------------------------------------------------


def test_gc80_a_guest_only_the_background_loaded_is_released_at_the_end_of_its_run():
    """GC80 -- a model the background alone loaded, with no call in flight
    on it, goes through the governor's eviction when the run that kept it
    resident lets it go; a model the governor never saw loaded is not
    touched. Releasing without asking who loaded it -> RED."""
    rg = _rg()
    gov, clk = _governor(rg, _config(rg, capacity=24.0, weights={"lib": 3.0}))
    evicted = []
    gov.evict_model = lambda name, **kw: evicted.append(name) or True
    assert gov.release_guest("never-loaded") is False and evicted == []
    gov.note_loaded_by("lib", "background")
    assert gov.release_guest("lib") is True
    assert evicted == ["lib"]


def test_gc81_a_model_a_higher_class_used_or_one_with_a_call_in_flight_is_not_released():
    """GC81 -- once a chat ticket on the model has been held, it is the
    chat's own and stays; a guest with the librarian's call in flight on it
    stays while the call runs, and goes once it returned. Ignoring the
    owner's class or the calls in flight -> RED."""
    rg = _rg()
    cfg = _config(rg, capacity=24.0, weights={"lib": 3.0, "shared": 3.0})
    gov, clk = _governor(rg, cfg, warmup=_warmup())
    rg._governor = gov
    gov._snapshot = _snapshot(rg, clk, capacity=24.0)
    evicted = []
    gov.evict_model = lambda name, **kw: evicted.append(name) or True
    gov.note_loaded_by("shared", "background")
    chat = gov.admit("shared", None, caller="chat")
    rg.set_active_ticket(chat)
    rg.clear_active_ticket()
    assert gov.release_guest("shared") is False, "the chat's own now"
    ticket = gov.admit("lib", None, caller="librarian")
    assert ticket.admitted and ticket.admission_class == "background", "control: the librarian asks as the background"
    gov.note_loaded_by("lib", "background")
    rg.set_active_ticket(ticket)
    try:
        assert gov.release_guest("lib") is False, "a call in flight on it"
    finally:
        rg.clear_active_ticket()
    assert gov.release_guest("lib") is True
    assert evicted == ["lib"]


# ---------------------------------------------------------------------------
# __main__ runner (parity with the sibling suites)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
        finally:
            while _CLOSERS:
                _CLOSERS.pop()()
    print(f"\n{len(tests) - failures} passed, {failures} failed")
    sys.exit(1 if failures else 0)
