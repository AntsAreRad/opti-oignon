#!/usr/bin/env python3
"""Resource Governor backfill: the DoS-bound security contracts.

Companion suite (additive; no original test file is edited). ``resource_governor.py``
(2915 lines) is the system's DoS bound -- admission gate, runtime backpressure,
and the optional runtime limits applier -- and ships with no test today. This
companion pins the highest-value security contracts across the three families,
each mutation-proven red-before-green.

The module's top-level imports are stdlib + ``yaml`` only, so it loads in
isolation through ``spec_from_file_location`` with the stubbed ``db_utils``
idiom the sibling suites use. Every external dependency the admission path
reaches (the Ollama warmup, the backend registry, the emergency-stop flag, the
ModelLimits clamp) is resolved through ``sys.modules`` first with a fail-open
fallback, so a hand-built snapshot plus an injected clock drive ``admit`` fully
deterministically -- no warmup or registry read happens.

Contracts pinned (spec RESOURCE_GOVERNOR_SPEC.md Sections 4-6):

  Admission (``admit``):
    * C1 fail-secure: an engaged emergency stop refuses BEFORE the
      governor-disabled passthrough and BEFORE the fit math (is_estop).
    * C2 refuse-beyond-capacity: a cost that exceeds the budget even with
      eviction is refused (vram_insufficient), never admitted.
    * C3 bounded fail-open: with capacity unknown the VRAM half fails open,
      but the RAM half STILL guards (a known weight cost above MemAvailable
      refuses, ram_insufficient).

  Backpressure (``_pressure_from_snapshot`` / ``_record_admission``):
    * C5 escalation = the worse of the two signals: a high refusal rate alone
      (capacity otherwise fine) still escalates the level to soft.
    * C7 an estop refusal NEVER enters the refusal-rate window (otherwise the
      estop would self-amplify the DoS pressure signal).

  Runtime limits (``apply_llamacpp_rlimits``, child-process idiom):
    * C8 off-by-default: rlimits_enabled=False applies nothing ("disabled").
    * C9 the soft limit is clamped to the existing hard ceiling (the applier
      lowers SOFT only and never asks for soft above hard); the clamp is what
      lets an over-ceiling operator request apply at all instead of being
      rejected by setrlimit.

The limits applier latches its outcome once per process and ``setrlimit`` is
process-wide, so C8/C9 run in a fresh child interpreter (the idiom the module
docstring names) and re-import the on-disk source -- which keeps them honest
under the mutation harness too.

Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
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
    """Load resource_governor.py in isolation (idempotent)."""
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


def _seed_estop(stopped: bool):
    """Install a fresh emergency_stop stub with the given flag (consumed by
    the module's sys.modules-first resolver)."""
    mod = types.ModuleType("opti_oignon.emergency_stop")
    mod.is_stopped = lambda: stopped
    mod.refusal_payload = lambda: {
        "error": "emergency_stopped",
        "message": "stop engaged",
    }
    sys.modules["opti_oignon.emergency_stop"] = mod
    return mod


def _clear_estop():
    """Remove the estop stub so the resolver fails open to None (no stop)."""
    sys.modules.pop("opti_oignon.emergency_stop", None)


class _FakeClock:
    """A settable monotonic stand-in so windows/freshness are deterministic."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _governor(rg, *, clock=None, config=None):
    """A governor with injected deps and a hand-set config (no real config
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


# ---------------------------------------------------------------------------
# Admission contracts
# ---------------------------------------------------------------------------


def test_c1_estop_refuses_before_passthrough_and_fit():
    """C1 -- fail-secure: an engaged estop refuses even when the governor is
    DISABLED and the model would otherwise fit comfortably. The estop branch
    is checked first; neutralising it lets the request fall through to the
    disabled passthrough (or the fit) and be admitted -> RED."""
    rg = _load_rg()
    _seed_estop(True)
    try:
        # Disabled governor + huge capacity: every other path would admit.
        cfg = rg.GovernorConfig(enabled=False, total_vram_gb=1000.0)
        gov, clk = _governor(rg, config=cfg)
        gov._snapshot = _fresh_snapshot(rg, clk, capacity=1000.0, in_use=0.0)
        decision = gov.admit("anymodel", requested_ctx=4096, caller="chat")
        assert decision.admitted is False
        assert decision.is_estop is True
        assert decision.reason == "emergency_stopped"
    finally:
        _clear_estop()


def test_c2_refuse_beyond_capacity():
    """C2 -- a weight cost far above the budget (even counting eviction) is
    refused with vram_insufficient; flipping the no-fit verdict to a fit would
    admit an over-capacity load -> RED."""
    rg = _load_rg()
    _clear_estop()
    cfg = rg.GovernorConfig(total_vram_gb=8.0, safety_margin_gb=1.5)
    cfg.weights_override_models = {"bigmodel": 50.0}  # 50 GiB on an 8 GiB box
    gov, clk = _governor(rg, config=cfg)
    gov._snapshot = _fresh_snapshot(rg, clk, capacity=8.0, in_use=0.0)
    decision = gov.admit("bigmodel", requested_ctx=None, caller="chat")
    assert decision.admitted is False
    assert decision.reason == "vram_insufficient"
    # budget_unconditional = 8.0 - 0.0 - 1.5 = 6.5; shortfall = 50.0 - 6.5.
    assert decision.shortfall_gb == 43.5


def test_c3_capacity_unknown_fails_open_but_ram_still_guards():
    """C3 -- bounded fail-open: with capacity unknown the VRAM half admits,
    but a known weight cost above MemAvailable still refuses (ram_insufficient).
    Inverting the RAM comparison would admit an over-RAM load -> RED."""
    rg = _load_rg()
    _clear_estop()
    cfg = rg.GovernorConfig(total_vram_gb=None, safety_margin_gb=1.5)
    cfg.weights_override_models = {"bigmodel": 50.0}
    gov, clk = _governor(rg, config=cfg)
    # capacity None -> VRAM half disabled; ram tiny (~1 GiB) << 50 GiB weights.
    gov._snapshot = _fresh_snapshot(
        rg, clk, capacity=None, in_use=0.0, ram_mb=1000.0
    )
    decision = gov.admit("bigmodel", requested_ctx=None, caller="chat")
    assert decision.admitted is False
    assert decision.reason == "ram_insufficient"


def test_c3b_capacity_unknown_admits_when_ram_suffices():
    """C3 control -- the SAME capacity-unknown path admits when the weight cost
    fits RAM (proves the refusal above is the RAM guard, not a blanket refuse)."""
    rg = _load_rg()
    _clear_estop()
    cfg = rg.GovernorConfig(total_vram_gb=None)
    gov, clk = _governor(rg, config=cfg)
    gov._snapshot = _fresh_snapshot(
        rg, clk, capacity=None, in_use=0.0, ram_mb=64000.0
    )
    decision = gov.admit("smallmodel", requested_ctx=4096, caller="chat")
    assert decision.admitted is True
    assert decision.reason == "capacity_unknown_fail_open"


# ---------------------------------------------------------------------------
# Backpressure contracts
# ---------------------------------------------------------------------------


def test_c5_pressure_escalates_on_refusal_rate_alone():
    """C5 -- the pressure level is the WORSE of the ratio signal and the
    refusal-rate signal. With capacity otherwise fine (ratio 'none') but a
    refusal-heavy window (>= 3 decisions, >= 50% refused), the level must be
    'soft'. Forcing the level to the ratio signal alone keeps it 'none' -> RED."""
    rg = _load_rg()
    cfg = rg.GovernorConfig(
        total_vram_gb=10.0,
        pressure_soft_threshold=0.85,
        pressure_hard_threshold=0.95,
        pressure_refusal_window_s=60.0,
    )
    gov, clk = _governor(rg, config=cfg)
    # 2 of 3 refused = 0.67 >= 0.5 over 3 >= 3 decisions, all inside the window.
    for refused in (True, True, False):
        gov._refusal_events.append((clk(), refused))
    snap = _fresh_snapshot(rg, clk, capacity=10.0, in_use=1.0)  # ratio 0.10
    state = gov._pressure_from_snapshot(snap)
    assert state["level"] == "soft"
    assert state["ratio"] == 0.1  # the ratio half alone would have said 'none'


def test_c5b_no_pressure_when_ratio_low_and_window_clean():
    """C5 control -- a low ratio AND a clean window yields 'none' (proves the
    escalation above comes from the refusal rate, not an always-soft floor)."""
    rg = _load_rg()
    cfg = rg.GovernorConfig(total_vram_gb=10.0, pressure_refusal_window_s=60.0)
    gov, clk = _governor(rg, config=cfg)
    snap = _fresh_snapshot(rg, clk, capacity=10.0, in_use=1.0)
    assert gov._pressure_from_snapshot(snap)["level"] == "none"


def test_c7_estop_refusal_never_enters_refusal_window():
    """C7 -- an estop refusal must NOT feed the refusal-rate window, or the
    emergency stop would amplify its own backpressure signal. After an estop
    refusal the window stays empty; counting it would make it length 1 -> RED.
    A resource refusal on the same governor DOES enter, proving the seam is
    live and the estop exclusion is specific."""
    rg = _load_rg()
    _seed_estop(True)
    try:
        cfg = rg.GovernorConfig(total_vram_gb=8.0)
        gov, clk = _governor(rg, config=cfg)
        gov._snapshot = _fresh_snapshot(rg, clk, capacity=8.0, in_use=0.0)
        gov.admit("m", requested_ctx=4096, caller="chat")
        assert len(gov._refusal_events) == 0  # estop refusal excluded
    finally:
        _clear_estop()

    # Same kind of refusal, but a RESOURCE one, must be counted.
    cfg2 = rg.GovernorConfig(total_vram_gb=8.0, safety_margin_gb=1.5)
    cfg2.weights_override_models = {"bigmodel": 50.0}
    gov2, clk2 = _governor(rg, config=cfg2)
    gov2._snapshot = _fresh_snapshot(rg, clk2, capacity=8.0, in_use=0.0)
    d = gov2.admit("bigmodel", requested_ctx=None, caller="chat")
    assert d.admitted is False
    assert len(gov2._refusal_events) == 1  # resource refusal counted


# ---------------------------------------------------------------------------
# Runtime limits contracts (child-process idiom: fresh latch, isolated rlimit)
# ---------------------------------------------------------------------------

_CHILD_PREAMBLE = textwrap.dedent(
    """
    import importlib.util, json, sqlite3, sys, types, resource
    _OO = {oo!r}
    pkg = types.ModuleType("opti_oignon"); pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda p, **k: sqlite3.connect(str(p))
    sys.modules["opti_oignon.db_utils"] = db
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.resource_governor", _OO + "/resource_governor.py")
    rg = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.resource_governor"] = rg
    spec.loader.exec_module(rg)
    """
)


def _run_child(body: str) -> dict:
    """Run a child interpreter that imports the on-disk source and prints a
    single JSON result line; return the parsed dict."""
    script = _CHILD_PREAMBLE.format(oo=str(_OO)) + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, f"child failed: {proc.stderr}\n{proc.stdout}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, f"child produced no output: {proc.stderr}"
    return json.loads(lines[-1])


def test_c8_rlimits_off_by_default():
    """C8 -- with rlimits_enabled=False the applier touches nothing and reports
    'disabled'; skipping that early return would apply a limit -> RED. The live
    RLIMIT_DATA soft is asserted unchanged."""
    result = _run_child(
        """
        soft0, hard0 = resource.getrlimit(resource.RLIMIT_DATA)
        cfg = rg.GovernorConfig()
        cfg.rlimits_enabled = False
        cfg.rlimits_data_gb = 64.0   # would apply if the disable gate were gone
        out = rg.apply_llamacpp_rlimits(cfg)
        soft1, _ = resource.getrlimit(resource.RLIMIT_DATA)
        print(json.dumps({
            "applied": out["applied"],
            "reason": out["reason"],
            "data_bytes": out["data_bytes"],
            "soft_unchanged": soft1 == soft0,
        }))
        """
    )
    assert result["applied"] is False
    assert result["reason"] == "disabled"
    assert result["data_bytes"] is None
    assert result["soft_unchanged"] is True


def test_c9_soft_clamped_to_existing_hard():
    """C9 -- the applier lowers SOFT only and never above the existing hard
    ceiling. The child first lowers its own RLIMIT_DATA hard to a finite value,
    then asks for a soft far above it; the clamp (min(target, hard)) makes the
    limit apply at hard. Dropping the clamp asks setrlimit for soft>hard, which
    it rejects, so the limit fails to apply -> RED."""
    result = _run_child(
        """
        finite_hard = 8 * (1024 ** 3)               # 8 GiB, well above usage
        resource.setrlimit(resource.RLIMIT_DATA, (finite_hard, finite_hard))
        cfg = rg.GovernorConfig()
        cfg.rlimits_enabled = True
        cfg.rlimits_data_gb = 100.0                 # 100 GiB target > 8 GiB hard
        out = rg.apply_llamacpp_rlimits(cfg)
        soft1, hard1 = resource.getrlimit(resource.RLIMIT_DATA)
        print(json.dumps({
            "applied": out["applied"],
            "data_bytes": out["data_bytes"],
            "finite_hard": finite_hard,
            "soft_eq_hard": soft1 == finite_hard,
            "hard_preserved": hard1 == finite_hard,
        }))
        """
    )
    assert result["applied"] is True
    assert result["data_bytes"] == result["finite_hard"]   # clamped to hard
    assert result["soft_eq_hard"] is True
    assert result["hard_preserved"] is True


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
