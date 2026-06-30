#!/usr/bin/env python3
"""Resource Governor backfill, suite: the enforcement seam, the ctx ladder,
and the learned-ceiling store.

Second companion (additive; the S280 companion and every original test file are
left byte-untouched). Where ``test_resource_governor_contracts.py`` pinned the
admission math, the backpressure signal, and the runtime-limits applier, this
file pins the three highest-value surfaces named at the S280 close:

  The enforcement seam (``backend_admission_gate``) -- the point the four
  generate/stream heads call, where a refusal actually RAISES and blocks:
    * G1 fail-secure: a ticketless call whose admit refuses RAISES the typed
      GovernorRefusal (the core of enforcement; without it a refused decision
      is silently ignored and the load proceeds).
    * G2 a disabled governor stands down entirely -- it returns without ever
      consulting admit (a pure resource no-op; estop is enforced elsewhere).
    * G3 a matching admitted ticket is honoured without re-deciding and the
      load is accounted EXACTLY once, even across repeated gate calls (the
      load_expected latch).

  Admission ctx shaping (``ResourceGovernor.admit``) -- the per-caller floor
  and the downsize ladder:
    * D1 a caller WITH a floor that cannot fit at the requested ctx is stepped
      down the ladder to a ctx that fits (action "downsize").
    * D2 a caller WITHOUT a floor (benchmark/AGT/direct) is REFUSED instead of
      being silently downsized -- the admission guarantee those callers rely on.
    * D3 the requested ctx is clamped to the model's context window before the
      fit check (ModelLimits stays the authority).

  Learned ceiling (``AdaptStore`` fast-down / slow-up, DI-8):
    * E1 a load failure lowers the working ceiling to (observed in-use minus
      the safety margin) IMMEDIATELY (fast-down, below the observed point).
    * E2 the ceiling relaxes upward only after _CEILING_RELAX_AFTER_SUCCESSES
      above-ceiling successes (slow-up; one fluke success does not race it back).
    * E3 the ceiling never drops below the configured floor.

Isolation follows the S280 idiom and is re-declared here so this file is fully
self-contained (no cross-test import; each mutation node runs in a fresh
interpreter that re-reads the on-disk source). The module's top-level imports
are stdlib + ``yaml`` only, so it loads through ``spec_from_file_location`` with
the stubbed ``db_utils``; every external seam is resolved through ``sys.modules``
first with a fail-open fallback, so a hand-built snapshot, an injected clock, a
seeded ModelLimits stub, and a fake governor drive every path deterministically
-- no warmup, registry, Ollama, eviction, or audit-chain read happens.

The gate tests mutate the module singleton and the thread-local ticket; each
restores both in a finally so neither this suite nor a sibling companion sees
leaked state.

Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"
_RG_SRC = _OO / "resource_governor.py"


def _install_base_stubs():
    """Seed the minimal ``opti_oignon`` package + db_utils (notes idiom)."""
    if not isinstance(sys.modules.get("opti_oignon"), types.ModuleType) or (
        getattr(sys.modules.get("opti_oignon"), "__file__", "x") is not None
    ):
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = []
        sys.modules["opti_oignon"] = pkg
    if "opti_oignon.db_utils" not in sys.modules:
        db = types.ModuleType("opti_oignon.db_utils")
        db.safe_connect = lambda p, **kw: sqlite3.connect(
            str(p), check_same_thread=kw.get("check_same_thread", False)
        )
        sys.modules["opti_oignon.db_utils"] = db


def _load_rg():
    """Load resource_governor.py in isolation (idempotent; shares the cached
    module with the sibling companion when both run in one process)."""
    _install_base_stubs()
    cached = sys.modules.get("opti_oignon.resource_governor")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.resource_governor", _RG_SRC
    )
    rg = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.resource_governor"] = rg
    spec.loader.exec_module(rg)
    return rg


class _FakeClock:
    """A settable monotonic stand-in so windows/freshness are deterministic."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _governor(rg, *, clock=None, config=None):
    """A real governor with injected deps and a hand-set config (no real config
    file, no warmup, no registry, no /proc read)."""
    clk = clock or _FakeClock()
    tmp = tempfile.mkdtemp()
    gov = rg.ResourceGovernor(
        config_path="/nonexistent-resource-governor-config",
        db_path=str(Path(tmp) / "governor.db"),
        warmup=None,
        registry=None,
        clock=clk,
        meminfo_path="/nonexistent-meminfo",
    )
    gov._config = config if config is not None else rg.GovernorConfig()
    return gov, clk


def _fresh_snapshot(rg, clk, *, capacity, in_use, loaded=None, ram_mb=64000.0):
    """A snapshot stamped now() so get_snapshot_fast returns it as-is."""
    return rg.ResourceSnapshot(
        taken_at=clk(),
        ttl_s=9999.0,
        loaded=loaded or [],
        capacity_gb=capacity,
        vram_in_use_gb=in_use,
        ram_available_mb=ram_mb,
    )


def _store(rg):
    """An AdaptStore on a throwaway sqlite file (open-use-close per op)."""
    tmp = tempfile.mkdtemp()
    return rg.AdaptStore(db_path=str(Path(tmp) / "adapt.db"))


def _seed_ctx_window(window: int):
    """Install a context_manager stub whose ModelLimits has the given window
    (consumed by the module's sys.modules-first clamp resolver)."""
    mod = types.ModuleType("opti_oignon.context_manager")

    class _Limits:
        context_window = window

    mod.get_model_limits = lambda model: _Limits()
    sys.modules["opti_oignon.context_manager"] = mod
    return mod


def _clear_ctx_window():
    """Remove the context_manager stub so the clamp fails open (no clamp)."""
    sys.modules.pop("opti_oignon.context_manager", None)


class _FakeGovernor:
    """A minimal governor stand-in for the enforcement-seam tests.

    Exposes only what ``backend_admission_gate`` touches: a config with an
    ``enabled`` flag, an ``admit`` that returns a pre-set decision (and counts
    calls), an ``invalidate_on_load`` recorder, and a conditional-eviction hook
    that fails loudly -- the gate contracts never set a conditional grant, so
    reaching the host-side eviction would be a test bug, not silent.
    """

    def __init__(self, *, enabled=True, decision=None):
        self.config = types.SimpleNamespace(enabled=enabled)
        self._decision = decision
        self.admit_calls = 0
        self.invalidate_calls: list[tuple] = []

    def admit(self, model, requested, caller="direct"):
        self.admit_calls += 1
        if self._decision is None:
            raise AssertionError(
                "admit must not be consulted on this path"
            )
        return self._decision

    def invalidate_on_load(self, model, num_ctx):
        self.invalidate_calls.append((model, num_ctx))

    def _honour_conditional_eviction(self, decision):  # pragma: no cover
        raise AssertionError(
            "conditional eviction must not fire in the gate contracts"
        )


def _seat_governor(rg, fake):
    """Seat a fake as the module singleton get_resource_governor() returns."""
    rg._governor = fake


# ---------------------------------------------------------------------------
# Enforcement-seam contracts (backend_admission_gate)
# ---------------------------------------------------------------------------


def test_g1_ticketless_refusal_raises():
    """G1 -- a ticketless call whose admit refuses must RAISE GovernorRefusal
    carrying that decision. Neutralising the refusal guard lets a refused
    decision fall through and the gate returns None (the load would proceed
    unblocked) -> RED."""
    rg = _load_rg()
    refused = rg.AdmissionDecision(
        admitted=False, model="m", action="refuse", reason="vram_insufficient"
    )
    fake = _FakeGovernor(enabled=True, decision=refused)
    _seat_governor(rg, fake)
    rg.clear_active_ticket()
    try:
        raised = False
        captured = None
        try:
            rg.backend_admission_gate("m", None)
        except rg.GovernorRefusal as exc:
            raised = True
            captured = exc.decision
        assert raised, "a ticketless refusal must raise GovernorRefusal"
        assert captured is refused
        assert fake.admit_calls == 1  # admit was consulted exactly once
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_g2_disabled_governor_stands_down_without_admit():
    """G2 -- a disabled governor returns from the gate WITHOUT consulting admit
    (a pure resource no-op). Removing the early disabled return makes the gate
    proceed and call admit, which on the fake asserts loudly -> RED."""
    rg = _load_rg()
    fake = _FakeGovernor(enabled=False, decision=None)  # admit() would assert
    _seat_governor(rg, fake)
    rg.clear_active_ticket()
    try:
        out = rg.backend_admission_gate("m", {"num_ctx": 4096})
        assert out is None  # stands down quietly
        assert fake.admit_calls == 0  # admit never consulted when disabled
        assert fake.invalidate_calls == []
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_g3_matching_ticket_accounts_load_once():
    """G3 -- a matching admitted ticket is honoured without re-deciding, and the
    load is accounted EXACTLY once across repeated gate calls. The second call
    sees load_expected already latched off and does nothing further; defeating
    the latch re-accounts the load on every call -> RED."""
    rg = _load_rg()
    fake = _FakeGovernor(enabled=True, decision=None)  # admit() would assert
    _seat_governor(rg, fake)
    ticket = rg.AdmissionDecision(
        admitted=True,
        model="m",
        num_ctx=4096,
        action="admit",
        load_expected=True,
        conditional_on_eviction=False,
    )
    rg.set_active_ticket(ticket)
    try:
        rg.backend_admission_gate("m", None)  # call 1: accounts the load
        rg.backend_admission_gate("m", None)  # call 2: ticket consumed, no-op
        assert fake.admit_calls == 0  # a matching ticket never re-decides
        assert fake.invalidate_calls == [("m", 4096)]  # accounted once only
        assert ticket.load_expected is False  # latched off after the first
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


# ---------------------------------------------------------------------------
# Admission ctx-shaping contracts (admit: per-caller floor + downsize ladder)
# ---------------------------------------------------------------------------

# Shared cost frame for D1/D2: weights 5.0 GiB, kv 0.5 GiB/1024 tok.
#   budget_unconditional = capacity - in_use - margin = 10.0 - 0.0 - 1.5 = 8.5
#   cost(8192) = 5.0 + 8*0.5 = 9.0  -> does NOT fit
#   cost(4096) = 5.0 + 4*0.5 = 7.0  -> fits
def _ladder_config(rg):
    cfg = rg.GovernorConfig(
        total_vram_gb=10.0,
        safety_margin_gb=1.5,
        kv_coefficient=0.5,
        ctx_ladder=[8192, 4096, 2048],
        ctx_floor={"chat": 2048},
    )
    cfg.weights_override_models = {"m": 5.0}
    return cfg


def test_d1_floored_caller_laddered_to_downsize():
    """D1 -- a floored caller (chat, floor 2048) that cannot fit at 8192 is
    stepped down the ladder to 4096, which fits: action "downsize". Dropping
    the ladder extension leaves [8192] only, which does not fit, and the caller
    is refused instead -> RED."""
    rg = _load_rg()
    gov, clk = _governor(rg, config=_ladder_config(rg))
    gov._snapshot = _fresh_snapshot(rg, clk, capacity=10.0, in_use=0.0)
    decision = gov.admit("m", requested_ctx=8192, caller="chat")
    assert decision.admitted is True
    assert decision.action == "downsize"
    assert decision.num_ctx == 4096  # first ladder step that fits
    assert "ctx_laddered_to_fit" in decision.reason


def test_d2_floorless_caller_refused_not_downsized():
    """D2 -- a floorless caller (direct) under the SAME cost is REFUSED, never
    silently downsized; the admission guarantee for benchmark/AGT/direct.
    Giving floorless callers a default floor would ladder them down to a fit and
    admit -> RED."""
    rg = _load_rg()
    gov, clk = _governor(rg, config=_ladder_config(rg))
    gov._snapshot = _fresh_snapshot(rg, clk, capacity=10.0, in_use=0.0)
    decision = gov.admit("m", requested_ctx=8192, caller="direct")
    assert decision.admitted is False
    assert decision.action == "refuse"
    assert decision.reason == "vram_insufficient"
    # shortfall = cost(8192) - budget_with_eviction = 9.0 - 8.5 = 0.5
    assert decision.shortfall_gb == 0.5


def test_d3_requested_ctx_clamped_to_model_window():
    """D3 -- the requested ctx is clamped to the model's context window (2048)
    before the fit check; capacity is huge so the fit always holds and only the
    clamp is under test. Dropping the clamp admits at the requested 8192 -> RED."""
    rg = _load_rg()
    _seed_ctx_window(2048)
    try:
        cfg = rg.GovernorConfig(
            total_vram_gb=1000.0, safety_margin_gb=1.5, kv_coefficient=0.5
        )
        cfg.weights_override_models = {"m": 1.0}
        gov, clk = _governor(rg, config=cfg)
        gov._snapshot = _fresh_snapshot(rg, clk, capacity=1000.0, in_use=0.0)
        decision = gov.admit("m", requested_ctx=8192, caller="chat")
        assert decision.admitted is True
        assert decision.num_ctx == 2048  # clamped to the window, not 8192
        assert "clamped_to_model_limit" in decision.reason
    finally:
        _clear_ctx_window()


# ---------------------------------------------------------------------------
# Learned-ceiling contracts (AdaptStore fast-down / slow-up, DI-8)
# ---------------------------------------------------------------------------


def test_e1_failure_fast_down_below_observed():
    """E1 -- a load failure lowers the ceiling to (observed in-use minus the
    safety margin) immediately: observed 10.0, margin 1.5 -> 8.5, in one call.
    Adding the margin instead of subtracting it would set the ceiling ABOVE the
    point that just failed (a less-conservative regression) -> RED."""
    rg = _load_rg()
    store = _store(rg)
    new_ceiling = store.record_load_failure(
        observed_in_use_gb=10.0, safety_margin_gb=1.5, floor_gb=4.0, now=1000.0
    )
    assert new_ceiling == 8.5
    assert store.get_learned_ceiling() == 8.5  # persisted


def test_e2_slow_up_only_after_threshold_successes():
    """E2 -- after a failure pins the ceiling at 8.5, above-ceiling successes
    relax it only every _CEILING_RELAX_AFTER_SUCCESSES (5). The first four leave
    it unchanged; the fifth raises it by one step (1.0) to 9.5. Relaxing on the
    first success would race the ceiling back up after a single fluke -> RED."""
    rg = _load_rg()
    n = rg._CEILING_RELAX_AFTER_SUCCESSES
    step = rg._CEILING_RELAX_STEP_GB
    store = _store(rg)
    store.record_load_failure(
        observed_in_use_gb=10.0, safety_margin_gb=1.5, floor_gb=4.0, now=1000.0
    )  # ceiling 8.5, successes reset to 0
    for i in range(n - 1):
        store.record_load_success(
            total_in_use_gb=9.0, configured_capacity_gb=None, now=1000.0 + i
        )
    assert store.get_learned_ceiling() == 8.5  # still pinned after n-1
    final = store.record_load_success(
        total_in_use_gb=9.0, configured_capacity_gb=None, now=2000.0
    )
    assert final == 8.5 + step  # the nth success raises it one step
    assert store.get_learned_ceiling() == 8.5 + step


def test_e3_floor_never_crossed_downward():
    """E3 -- the ceiling never drops below the configured floor: an observed
    in-use of 2.0 with margin 1.5 would compute 0.5, but the floor (4.0) wins.
    Taking the min against the floor instead of the max would let the ceiling
    sink below the floor -> RED."""
    rg = _load_rg()
    store = _store(rg)
    new_ceiling = store.record_load_failure(
        observed_in_use_gb=2.0, safety_margin_gb=1.5, floor_gb=4.0, now=1000.0
    )
    assert new_ceiling == 4.0  # floor, not 0.5
    assert store.get_learned_ceiling() == 4.0


# ---------------------------------------------------------------------------
# __main__ runner (parity with the sibling companions)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [
        v
        for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures} passed, {failures} failed")
    sys.exit(1 if failures else 0)
