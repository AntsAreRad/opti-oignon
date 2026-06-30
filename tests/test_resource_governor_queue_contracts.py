#!/usr/bin/env python3
"""Resource Governor backfill, suite: the bounded queue, the conditional-grant
decision, and the learned-cost store.

Third companion (additive; the S280 and S281 companions and every original test
file are left byte-untouched). It pins the governor surfaces that remained
container-isolable after S281, each mutation-proven red-before-green:

  The bounded admission queue (``admit_or_wait``, Section 5) -- only the
  NON-BLOCKING exits (queue_wait_s=0 throughout, so no wait slice is ever
  entered); the durable observable is the decisions ring (an enqueue writes a
  "queue" decision):
    * W1 an admitted decision stands down immediately and never enqueues.
    * W2 a non-enrolled refused caller returns the refusal without enqueuing.
    * W3 an enrolled refused caller already AT the depth bound stands (the bound
      is inclusive: at exactly the bound, no new enqueue).

  The conditional-grant decision (``admit``, Section 4.2) -- the DECISION only;
  the eviction act (``_honour_conditional_eviction``) is host-side and untouched:
    * C4a a fit reachable ONLY through evictable_now is granted CONDITIONAL on
      eviction.
    * C4b a fit that holds unconditionally is NOT conditional (the flag tracks
      the fit verdict, not an always-conditional path).

  The learned per-model cost store (``AdaptStore.record_model_cost`` /
  ``get_model_cost``):
    * P1 a recorded (name, digest) cost round-trips exactly.
    * P2 the exact-digest lookup is preferred over a newer digest-less row.
    * P3 (control) an absent model returns None -- never "too large" (3.1).

Isolation follows the S280/S281 idiom, re-declared here so this file is fully
self-contained. The conditional-grant path reads ``self._warmup.keep_alive`` and
a real ``time.time()`` for idle derivation; rather than patch the clock, an
injected fake warmup (keep_alive "10m") plus a loaded view whose ``expires_at``
is epoch 0 makes the derived idle time dwarf any sub-second drift in ``now``, so
evictability is deterministic. The queue tests force admit/refuse through a
pre-injected snapshot and read the enqueue back from the ring.

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
_GIB = 1024 ** 3


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
    module with the sibling companions when several run in one process)."""
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


def _evictable_view(rg, *, name="resident", gb=4.0):
    """A loaded view whose epoch-0 expiry makes its derived idle time dwarf any
    sub-second drift in the real now() -> deterministically evictable."""
    return rg.LoadedModelView(
        name=name, size_vram_bytes=int(gb * _GIB), expires_at=0.0
    )


def _warmup(keep_alive="10m"):
    """A minimal warmup stand-in carrying only the keep_alive the evictable
    derivation reads (parsed to seconds by the module)."""
    return types.SimpleNamespace(keep_alive=keep_alive)


def _queue_decisions(gov):
    """The 'queue' (enqueue) entries currently in the decisions ring."""
    return [d for d in gov._store.recent_decisions(50) if d["decision"] == "queue"]


# ---------------------------------------------------------------------------
# Bounded-queue contracts (admit_or_wait; non-blocking exits, ring observable)
# ---------------------------------------------------------------------------


def _queue_config(rg, *, capacity, enrolled, depth=2):
    cfg = rg.GovernorConfig(
        total_vram_gb=capacity,
        safety_margin_gb=1.5,
        kv_coefficient=0.5,
        queue_enabled_per_caller=({"benchmark": True} if enrolled else {}),
        queue_depth=depth,
        queue_wait_s=0.0,  # no blocking wait is ever entered
    )
    return cfg


def test_w1_admitted_returns_without_enqueue():
    """W1 -- an admitted decision returns immediately and never enqueues, even
    for an enrolled caller. Dropping the admitted short-circuit makes the admit
    fall into the queue block and record a "queue" decision -> RED."""
    rg = _load_rg()
    cfg = _queue_config(rg, capacity=10.0, enrolled=True)
    cfg.weights_override_models = {"m": 1.0}  # cost fits comfortably
    gov, clk = _governor(rg, config=cfg)
    gov._snapshot = _fresh_snapshot(rg, clk, capacity=10.0, in_use=0.0)
    decision = gov.admit_or_wait("m", requested_ctx=2048, caller="benchmark")
    assert decision.admitted is True
    assert _queue_decisions(gov) == []  # admitted never touches the queue


def test_w2_unenrolled_refusal_returns_without_enqueue():
    """W2 -- a non-enrolled caller whose admission refuses gets plain admit
    semantics (the refusal is returned, nothing enqueued). Removing the
    enrolment guard makes the non-enrolled caller enqueue -> RED."""
    rg = _load_rg()
    cfg = _queue_config(rg, capacity=8.0, enrolled=False)
    cfg.weights_override_models = {"m": 50.0}  # 50 GiB on an 8 GiB box -> refuse
    gov, clk = _governor(rg, config=cfg)
    gov._snapshot = _fresh_snapshot(rg, clk, capacity=8.0, in_use=0.0)
    decision = gov.admit_or_wait("m", requested_ctx=None, caller="benchmark")
    assert decision.admitted is False
    assert _queue_decisions(gov) == []  # not enrolled -> no enqueue


def test_w3_enrolled_refusal_at_depth_bound_does_not_enqueue():
    """W3 -- an enrolled refused caller already AT the depth bound stands on its
    refusal without a new enqueue (the bound is inclusive). Loosening the
    comparison from >= to > admits one more past the bound -> RED."""
    rg = _load_rg()
    cfg = _queue_config(rg, capacity=8.0, enrolled=True, depth=2)
    cfg.weights_override_models = {"m": 50.0}  # refuse
    gov, clk = _governor(rg, config=cfg)
    gov._snapshot = _fresh_snapshot(rg, clk, capacity=8.0, in_use=0.0)
    gov._queue_depth = gov._config.queue_depth  # already at the bound
    decision = gov.admit_or_wait("m", requested_ctx=None, caller="benchmark")
    assert decision.admitted is False
    assert _queue_decisions(gov) == []  # at the bound -> no new enqueue


# ---------------------------------------------------------------------------
# Conditional-grant contracts (admit; the decision only, no eviction act)
# ---------------------------------------------------------------------------

# Shared frame: weights 3.0 GiB, kv 0.5 GiB/1024 tok, one idle-evictable
# 4.0 GiB resident. cost(m, 2048) = 3.0 + 1.0 = 4.0.
def _evict_config(rg):
    cfg = rg.GovernorConfig(
        total_vram_gb=10.0,
        safety_margin_gb=1.5,
        kv_coefficient=0.5,
        idle_evict_threshold_s=600.0,
    )
    cfg.weights_override_models = {"m": 3.0}
    return cfg


def test_c4a_eviction_only_fit_is_conditional_grant():
    """C4a -- a fit reachable only by counting the idle-evictable resident is
    granted CONDITIONAL on eviction. budget_unconditional = 10-8-1.5 = 0.5 <
    4.0; budget_with_eviction = 0.5 + 4.0 = 4.5 >= 4.0. Forcing the conditional
    flag off marks an eviction-only fit as unconditional -> RED."""
    rg = _load_rg()
    gov, clk = _governor(rg, config=_evict_config(rg))
    gov._warmup = _warmup("10m")  # keep_alive_s = 600
    gov._snapshot = _fresh_snapshot(
        rg, clk, capacity=10.0, in_use=8.0, loaded=[_evictable_view(rg)]
    )
    decision = gov.admit("m", requested_ctx=2048, caller="chat")
    assert decision.admitted is True
    assert decision.conditional_on_eviction is True
    assert "conditional_on_eviction" in decision.reason


def test_c4b_unconditional_fit_is_not_conditional():
    """C4b -- with the same idle-evictable resident but budget that suffices
    unconditionally (in_use 2.0 -> budget 6.5 >= 4.0), the grant is NOT
    conditional: the flag tracks the fit verdict. Forcing the flag on marks an
    unconditional fit as conditional -> RED."""
    rg = _load_rg()
    gov, clk = _governor(rg, config=_evict_config(rg))
    gov._warmup = _warmup("10m")
    gov._snapshot = _fresh_snapshot(
        rg, clk, capacity=10.0, in_use=2.0, loaded=[_evictable_view(rg)]
    )
    decision = gov.admit("m", requested_ctx=2048, caller="chat")
    assert decision.admitted is True
    assert decision.conditional_on_eviction is False


# ---------------------------------------------------------------------------
# Learned-cost store contracts (record_model_cost / get_model_cost)
# ---------------------------------------------------------------------------


def test_p1_model_cost_roundtrips_by_name_and_digest():
    """P1 -- a recorded (name, digest) cost is read back exactly. Dropping the
    digest persistence stores an empty digest, so the exact lookup misses and
    falls back to a row whose digest no longer matches -> RED."""
    rg = _load_rg()
    store = _store(rg)
    store.record_model_cost(
        "m", "sha256:abc", 4 * _GIB, num_ctx=4096, observed_at=1000.0
    )
    got = store.get_model_cost("m", "sha256:abc")
    assert got is not None
    assert got["size_vram_bytes"] == 4 * _GIB
    assert got["digest"] == "sha256:abc"
    assert got["num_ctx"] == 4096


def test_p2_exact_digest_preferred_over_recency():
    """P2 -- the exact-digest row is preferred over a NEWER digest-less row for
    the same name. Skipping the exact lookup falls back to latest-by-time, which
    is the digest-less 2 GiB row, not the exact 4 GiB one -> RED."""
    rg = _load_rg()
    store = _store(rg)
    store.record_model_cost("m", "sha256:abc", 4 * _GIB, observed_at=1000.0)
    store.record_model_cost("m", "", 2 * _GIB, observed_at=2000.0)  # newer, no digest
    got = store.get_model_cost("m", "sha256:abc")
    assert got is not None
    assert got["size_vram_bytes"] == 4 * _GIB  # exact digest beats recency


def test_p3_absent_model_returns_none():
    """P3 (control) -- an absent model returns None (never "too large", 3.1);
    proves P1/P2 are not an artefact of always returning a row."""
    rg = _load_rg()
    store = _store(rg)
    assert store.get_model_cost("absent-model") is None


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
