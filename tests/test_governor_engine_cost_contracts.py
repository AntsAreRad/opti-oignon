#!/usr/bin/env python3
"""What the governor charges a load, and which engine it asks.

The governor asked every engine, in the order they were registered, for a
model's geometry and size, whichever engine was about to serve it; it priced
every model as a token generator, weights plus a KV cache that grows with the
context, whatever the engine served; it charged a model already loaded its KV
again, on top of the memory the model already held, so a full card turned a
resident model into a split with nothing on the GPU; and the eviction it
planned for a conditional grant priced the load its own way, without the
operator's overrides or the model's geometry, so it could evict nothing for a
load the admission had found too large.

  The engine that will serve:
    * HC1 -- the engine the registry resolves for the model is asked first,
      and the decision names it.
    * HC2 -- an engine named by the caller wins over the resolution.
    * HC3 -- a registry that cannot resolve keeps the registration order, and
      the decision names no engine.

  What an engine declares (a model that is not a token generator):
    * HC4 -- every engine today declares nothing, and prices as before.
    * HC5 -- a model with no KV cache is charged none at any context, and its
      context is never stepped down for memory.
    * HC6 -- a declared per-request state is charged once.
    * HC7 -- declared weights replace the estimate; an operator override still
      wins over them.
    * HC8 -- a declaration that does not hold together is ignored.

  A model already loaded:
    * HC9 -- asked at or below its loaded context, it costs nothing and is
      never split, even on a full card.
    * HC10 -- asked above its loaded context, it is a reload: its own memory
      is credited and the whole cost charged.
    * HC11 -- with its loaded context unknown, it is priced as before.

  The eviction a conditional grant plans:
    * HC12 -- it prices the load exactly as the admission that granted it.
    * HC13 -- a grant the admission did not price is priced the admission's
      way: the operator's overrides and the model's own KV coefficient.

  What a first version missed (an independent review):
    * HC14 -- admission reads the registry's cached resolution alone and
      calls no engine for it.
    * HC15 -- an engine name no backend carries serves nothing.
    * HC16 -- a resident that holds its context still pays for the draft
      that loads beside it.
    * HC17 -- a reload is priced from what the resident holds, and frees its
      RAM as well as its VRAM.
    * HC18 -- with the capacity unknown, a reload needs only the RAM it does
      not free.
    * HC19 -- a reload never steps below the context the model holds.
    * HC20 -- the dynamic context sizes a reload with the memory it frees.
    * HC21 -- an eviction counts no credit for a model gone since its grant.
    * HC22 -- a model is never its own eviction candidate.

Everything here is proven in the container, on scripted snapshots and fakes
loaded through the shared isolation window: no card, no socket, no model.
Whether an engine reloads a model for a different context is owed to the
machine.
"""

import sqlite3
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_RG = "opti_oignon.resource_governor"
_BACKEND = "opti_oignon.inference_backend"
_SEAMS = ("opti_oignon.context_manager", "opti_oignon.emergency_stop")
_GIB = 1024 ** 3
_CLOSERS = []


def _db_utils():
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda p, **kw: sqlite3.connect(
        str(p), check_same_thread=kw.get("check_same_thread", False)
    )
    return db


def _open(*extra):
    targets = {_RG: source("resource_governor.py")}
    for name, path in extra:
        targets[name] = path
    loaded, restore = isolate(
        targets=targets, blocked=_SEAMS, seeded={"opti_oignon.db_utils": _db_utils()}
    )
    _CLOSERS.append(restore)
    return loaded


def _project_modules():
    return {k: v for k, v in sys.modules.items() if k == "opti_oignon" or k.startswith("opti_oignon.")}


@pytest.fixture(autouse=True)
def _left_as_found():
    before = _project_modules()
    urlopen = urllib.request.urlopen
    yield
    urllib.request.urlopen = urlopen
    while _CLOSERS:
        _CLOSERS.pop()()
    assert _project_modules() == before, "every project module is left as the contract found it"


class _Clock:
    def __call__(self) -> float:
        return 1000.0


class _Warmup:
    def __init__(self, keep_alive="10m"):
        self.keep_alive = keep_alive

    def get_loaded_models(self):
        return []


def _config(rg, *, weights=None, kv=None, **fields):
    """10 GiB card, 1.5 GiB margin: 8.5 GiB free on an empty card."""
    cfg = rg.GovernorConfig(
        total_vram_gb=10.0,
        safety_margin_gb=1.5,
        kv_coefficient=0.5,
        ctx_ladder=[8192, 4096, 2048],
        ctx_floor={"chat": 2048},
    )
    cfg.weights_override_models = dict(weights or {})
    cfg.kv_override_models = dict(kv or {})
    for name, value in fields.items():
        setattr(cfg, name, value)
    return cfg


def _governor(rg, config, *, registry=None):
    tmp = Path(tempfile.mkdtemp(prefix="hc-"))
    meminfo = tmp / "meminfo"
    meminfo.write_text("MemAvailable:   65536000 kB\n", encoding="utf-8")
    gov = rg.ResourceGovernor(
        config_path=str(tmp / "missing.yaml"),
        db_path=str(tmp / "governor.db"),
        warmup=_Warmup(),
        registry=registry,
        clock=_Clock(),
        meminfo_path=str(meminfo),
        vram_probe=None,
    )
    gov._config = config
    return gov


def _snapshot(rg, gov, *, in_use=0.0, loaded=None, capacity=10.0, ram_mb=64000.0):
    gov._snapshot = rg.ResourceSnapshot(
        taken_at=1000.0,
        ttl_s=9999.0,
        loaded=list(loaded or []),
        capacity_gb=capacity,
        vram_in_use_gb=in_use,
        ram_available_mb=ram_mb,
    )
    return gov._snapshot


def _view(rg, name, gb, *, context=None, idle=False, total=None):
    """A loaded model; an epoch-0 expiry makes it idle past any threshold.

    ``total`` is the size Ollama reports beside size_vram (a split model).
    """
    return rg.LoadedModelView(
        name=name,
        size_vram_bytes=int(gb * _GIB),
        expires_at=0.0 if idle else None,
        context_length=context,
        size_bytes=int(total * _GIB) if total is not None else 0,
    )


class _Info:
    def __init__(self, mapping=None):
        self.extra = {"model_info": dict(mapping)} if mapping is not None else {}
        self.path = None
        self.parameter_size = None
        self.quantization_level = None
        self.size = None


class _Backend:
    def __init__(self, name, infos=None, costs=None):
        self.name = name
        self._infos = dict(infos or {})
        if costs is not None:
            self.cost_model = lambda model: costs.get(model)

    def model_info(self, model):
        return self._infos.get(model)


class _Registry:
    """``resolve`` is what the funnels resolved last (the registry's cache)."""

    def __init__(self, *backends, resolve=None):
        self._backends = list(backends)
        self._resolve = dict(resolve or {})

    def backends(self):
        return list(self._backends)

    def get(self, name):
        return next((b for b in self._backends if b.name == name), None)

    def cached_backend(self, model):
        return self.get(self._resolve[model]) if model in self._resolve else None

    def resolve_backend(self, model):
        raise AssertionError("admission must not resolve a backend by itself")


def _llama(layers):
    """32 heads over a 4096 embedding: 128 per head, 8 KV heads, f16."""
    return {
        "general.architecture": "llama",
        "llama.block_count": layers,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8,
        "llama.embedding_length": 4096,
    }


# ---------------------------------------------------------------------------
# HC1-HC3 -- the engine that will serve
# ---------------------------------------------------------------------------


def _two_engines(resolve=None):
    # 32 layers: 0.5 GiB of KV at 4096 tokens; 64 layers: 1.0 GiB.
    first = _Backend("ollama", {"m": _Info(_llama(32))})
    second = _Backend("llama_cpp", {"m": _Info(_llama(64))})
    return _Registry(first, second, resolve=resolve)


def test_hc1_the_engine_the_registry_resolves_is_asked_first_and_named():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"m": 1.0}), registry=_two_engines({"m": "llama_cpp"}))
    _snapshot(rg, gov)
    decision = gov.admit("m", requested_ctx=4096, caller="benchmark")
    assert decision.admitted is True
    assert decision.engine == "llama_cpp"
    assert decision.cost_gb == 2.0


def test_hc2_an_engine_the_caller_names_wins_over_the_resolution():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"m": 1.0}), registry=_two_engines({"m": "llama_cpp"}))
    _snapshot(rg, gov)
    decision = gov.admit("m", requested_ctx=4096, caller="benchmark", engine="ollama")
    assert decision.engine == "ollama"
    assert decision.cost_gb == 1.5


def test_hc3_a_registry_that_cannot_resolve_keeps_the_registration_order():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"m": 1.0}), registry=_two_engines())
    _snapshot(rg, gov)
    decision = gov.admit("m", requested_ctx=4096, caller="benchmark")
    assert decision.engine is None
    assert decision.cost_gb == 1.5


# ---------------------------------------------------------------------------
# HC4-HC8 -- what an engine declares
# ---------------------------------------------------------------------------


def test_hc4_every_engine_today_declares_nothing_and_prices_as_before():
    loaded = _open((_BACKEND, source("inference_backend.py")))
    rg, mod = loaded[_RG], loaded[_BACKEND]
    assert mod.InferenceBackend.cost_model(None, "m") is None
    for engine in (mod.OllamaBackend, mod.LlamaCppBackend, mod.LlamaServerBackend):
        assert engine.cost_model is mod.InferenceBackend.cost_model, engine.__name__
    silent = _Registry(_Backend("ollama", {"m": _Info(_llama(32))}, costs={}), resolve={"m": "ollama"})
    bare = _Registry(_Backend("ollama", {"m": _Info(_llama(32))}), resolve={"m": "ollama"})
    costs = []
    for registry in (silent, bare):
        gov = _governor(rg, _config(rg, weights={"m": 1.0}), registry=registry)
        _snapshot(rg, gov)
        costs.append(gov.admit("m", requested_ctx=4096, caller="benchmark").cost_gb)
    assert costs == [1.5, 1.5]


def test_hc5_a_model_with_no_kv_cache_is_charged_none_and_never_stepped_down():
    rg = _open()[_RG]
    encoder = _Backend("npu", costs={"e": {"kind": "encoder", "weights_gb": 9.0, "kv": False}})
    gov = _governor(rg, _config(rg, offload_prefer="speed"), registry=_Registry(encoder, resolve={"e": "npu"}))
    _snapshot(rg, gov)
    # 9.0 does not fit the 8.5 free. A generator would step down the ladder
    # first; without a KV cache a shorter context saves nothing, so the
    # split comes at the context asked.
    decision = gov.admit("e", requested_ctx=8192, caller="chat")
    assert decision.admitted is True
    assert (decision.action, decision.num_ctx) == ("admit", 8192)
    assert decision.partial_offload is True
    assert decision.cost_gb == 9.0


def test_hc6_a_declared_per_request_state_is_charged_once():
    rg = _open()[_RG]
    predictor = _Backend("npu", costs={"p": {"kind": "predictor", "weights_gb": 6.0, "state_gb": 3.0, "kv": False}})
    gov = _governor(rg, _config(rg), registry=_Registry(predictor, resolve={"p": "npu"}))
    _snapshot(rg, gov)
    decision = gov.admit("p", requested_ctx=4096, caller="benchmark")
    assert decision.cost_gb == 9.0
    assert decision.partial_offload is True
    assert (decision.vram_cost_gb, decision.ram_cost_gb) == (8.5, 0.5)


def test_hc7_declared_weights_replace_the_estimate_and_an_override_still_wins():
    rg = _open()[_RG]
    declared = _Backend("ollama", {"m": _Info(_llama(32))}, costs={"m": {"kind": "generator", "weights_gb": 3.0}})
    registry = _Registry(declared, resolve={"m": "ollama"})
    gov = _governor(rg, _config(rg), registry=registry)
    _snapshot(rg, gov)
    assert gov.admit("m", requested_ctx=4096, caller="benchmark").cost_gb == 3.5
    gov = _governor(rg, _config(rg, weights={"m": 5.0}), registry=registry)
    _snapshot(rg, gov)
    assert gov.admit("m", requested_ctx=4096, caller="benchmark").cost_gb == 5.5


def test_hc8_a_declaration_that_does_not_hold_together_is_ignored():
    rg = _open()[_RG]
    for bad in (
        {"kind": "oracle"},
        {"kind": "encoder", "weights_gb": -1.0},
        {"kind": "encoder", "state_gb": "lots"},
        {"kind": "encoder", "kv": "no"},
        {"kind": "encoder", "weights_gb": float("nan")},
        {"kind": "encoder", "weights_gb": 10 ** 400},
        {"kind": "generator", "kv": False},
        "encoder",
        7,
    ):
        engine = _Backend("ollama", {"m": _Info(_llama(32))}, costs={"m": bad})
        gov = _governor(rg, _config(rg, weights={"m": 1.0}), registry=_Registry(engine, resolve={"m": "ollama"}))
        _snapshot(rg, gov)
        # The generator's price: 1.0 of weights and 0.5 of KV at 4096 tokens.
        assert gov.admit("m", requested_ctx=4096, caller="benchmark").cost_gb == 1.5, bad


# ---------------------------------------------------------------------------
# HC9-HC11 -- a model already loaded
# ---------------------------------------------------------------------------


def test_hc9_a_resident_asked_at_or_below_its_loaded_context_costs_nothing_and_is_never_split():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg))
    # The card is full with the model itself: 9.5 of 10 GiB.
    _snapshot(rg, gov, in_use=9.5, loaded=[_view(rg, "m", 9.5, context=8192)])
    for ctx in (8192, 4096):
        decision = gov.admit("m", requested_ctx=ctx, caller="chat")
        assert decision.admitted is True, ctx
        assert decision.reason == "fits_resident", ctx
        assert decision.gpu_share is None, ctx
        assert decision.load_expected is False, ctx
        assert decision.cost_gb == 0.0, ctx


def test_hc10_a_resident_asked_above_its_loaded_context_is_a_reload_with_its_own_memory_credited():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"m": 4.0}))
    _snapshot(rg, gov, in_use=6.0, loaded=[_view(rg, "m", 6.0, context=2048)])
    decision = gov.admit("m", requested_ctx=8192, caller="chat")
    # 4.0 of weights and 4.0 of KV at 8192 tokens, against 10 - 6 - 1.5 + the
    # 6 the reload frees.
    assert decision.admitted is True
    assert decision.reason == "fits"
    assert decision.load_expected is True
    assert (decision.cost_gb, decision.credit_gb) == (8.0, 6.0)


def test_hc11_a_resident_with_its_loaded_context_unknown_is_priced_as_before():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"m": 4.0}))
    _snapshot(rg, gov, in_use=6.0, loaded=[_view(rg, "m", 6.0)])
    decision = gov.admit("m", requested_ctx=2048, caller="chat")
    assert decision.admitted is True
    assert decision.reason == "fits"
    assert decision.load_expected is False
    assert (decision.cost_gb, decision.credit_gb) == (1.0, 0.0)


# ---------------------------------------------------------------------------
# HC12-HC13 -- the eviction a conditional grant plans
# ---------------------------------------------------------------------------


def _with_idle_residents(rg, gov):
    """Two idle models of 3 GiB each: 2.5 free now, 8.5 after evicting both."""
    gov._warmup = _Warmup(keep_alive="10m")
    _snapshot(rg, gov, in_use=6.0, loaded=[_view(rg, "a", 3.0, idle=True), _view(rg, "b", 3.0, idle=True)])
    evicted = []
    gov.evict_model = lambda name, **kw: evicted.append(name) or True
    return evicted


def test_hc12_the_eviction_prices_the_load_exactly_as_the_admission_that_granted_it():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg))
    gov._warmup = _Warmup(keep_alive="10m")
    # m holds 5 GiB at 1024 (0.5 of KV): reloaded at 4096 it costs 4.5 + 2.0,
    # its 5 credited; three idle models of 1 GiB; 0.5 free.
    idle = [_view(rg, name, 1.0, idle=True) for name in ("a", "b", "c")]
    _snapshot(rg, gov, in_use=8.0, loaded=idle + [_view(rg, "m", 5.0, context=1024)])
    evicted = []
    gov.evict_model = lambda name, **kw: evicted.append(name) or True
    decision = gov.admit("m", requested_ctx=4096, caller="chat")
    assert decision.conditional_on_eviction is True
    assert (decision.cost_gb, decision.credit_gb) == (6.5, 5.0)
    gov._honour_conditional_eviction(decision)
    # 6.5 - 5.0 - 0.5: one gigabyte is needed, one eviction. Priced its own
    # way, as a resident's KV alone, the load would need 1.5 and evict two.
    assert len(evicted) == 1


def test_hc13_a_grant_the_admission_did_not_price_is_priced_the_admissions_way():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"m": 4.0}, kv={"m": 0.25}))
    evicted = _with_idle_residents(rg, gov)
    grant = rg.AdmissionDecision(
        admitted=True, model="m", num_ctx=4096, conditional_on_eviction=True, load_expected=True
    )
    gov._honour_conditional_eviction(grant)
    assert len(evicted) == 1


# ---------------------------------------------------------------------------
# HC14-HC22 -- what the first version missed
# ---------------------------------------------------------------------------


def test_hc14_admission_reads_the_registrys_cache_alone_and_calls_no_engine():
    loaded = _open((_BACKEND, source("inference_backend.py")))
    rg, mod = loaded[_RG], loaded[_BACKEND]

    class _Probe:
        display_name = "probe"

        def __init__(self, name):
            self.name = name
            self.checks = 0

        def health_check(self):
            self.checks += 1
            return True

        def model_info(self, model):
            return types.SimpleNamespace(name=model) if model == "m" else None

    registry = mod.BackendRegistry()
    probes = [_Probe("ollama"), _Probe("llama_cpp")]
    for probe in probes:
        registry.register(probe)
    assert registry.cached_backend("m") is None
    resolved = registry.resolve_backend("m")
    checks = sum(p.checks for p in probes)
    assert registry.cached_backend("m") is resolved
    assert sum(p.checks for p in probes) == checks
    # The governor reads that cache; resolving by itself would raise here.
    gov = _governor(rg, _config(rg, weights={"m": 1.0}), registry=_two_engines({"m": "llama_cpp"}))
    _snapshot(rg, gov)
    assert gov.admit("m", requested_ctx=4096, caller="benchmark").engine == "llama_cpp"


def test_hc15_an_engine_name_no_backend_carries_serves_nothing():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"m": 1.0}), registry=_two_engines({"m": "llama_cpp"}))
    _snapshot(rg, gov)
    decision = gov.admit("m", requested_ctx=4096, caller="benchmark", engine="tpu")
    assert decision.engine is None
    assert decision.cost_gb == 1.5


def test_hc16_a_resident_that_holds_its_context_still_pays_for_the_draft_that_loads():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"d": 4.0}, offload_enabled=False))
    _snapshot(rg, gov, in_use=9.5, loaded=[_view(rg, "m", 9.5, context=8192)])
    decision = gov.admit("m", requested_ctx=None, caller="direct", extra_models=["d"])
    # The 4 GiB draft against 10 - 9.5 - 1.5: no room, offload off.
    assert decision.admitted is False
    assert decision.reason == "vram_insufficient"
    assert decision.shortfall_gb == 5.0


def test_hc17_a_reload_is_priced_from_what_the_resident_holds_and_frees_its_ram_too():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg))
    # 6 GiB on the card, 2 more in RAM, loaded at 2048 (1.0 of KV): 7.0 of
    # weights; asked 4096, 7.0 + 2.0 = 9.0 against 10 - 6 - 1.5 + 6 = 8.5.
    # 4.25 GiB of RAM free: 0.25 over the reserve, 2.25 with what it frees.
    _snapshot(rg, gov, in_use=6.0, loaded=[_view(rg, "m", 6.0, context=2048, total=8.0)], ram_mb=4352.0)
    decision = gov.admit("m", requested_ctx=4096, caller="benchmark")
    assert decision.admitted is True
    assert decision.partial_offload is True
    assert (decision.cost_gb, decision.credit_gb) == (9.0, 6.0)
    assert (decision.vram_cost_gb, decision.ram_cost_gb) == (8.5, 0.5)


def test_hc18_with_the_capacity_unknown_a_reload_needs_only_the_ram_it_does_not_free():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg))
    # Held in RAM alone, 10 GiB at 4096 (2.0 of KV): 8.0 of weights, asked
    # 8192 with 5.86 GiB free; the reload frees the 10 it holds first.
    _snapshot(rg, gov, capacity=None, loaded=[_view(rg, "m", 0.0, context=4096, total=10.0)], ram_mb=6000.0)
    decision = gov.admit("m", requested_ctx=8192, caller="chat")
    assert decision.admitted is True
    assert decision.reason == "capacity_unknown_fail_open"
    assert decision.load_expected is True


def test_hc19_a_reload_never_steps_below_the_context_the_model_holds():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, offload_enabled=False))
    # 7 GiB loaded at 4096 (2.0 of KV): 5.0 of weights. At 8192, 9.0 against
    # 8.5; a reload at 4096 would only rebuild what it holds.
    _snapshot(rg, gov, in_use=7.0, loaded=[_view(rg, "m", 7.0, context=4096)])
    decision = gov.admit("m", requested_ctx=8192, caller="chat")
    assert decision.admitted is True
    assert (decision.action, decision.reason, decision.num_ctx) == ("downsize", "ctx_laddered_to_fit+fits_resident", 4096)
    assert decision.load_expected is False
    # A caller without a floor is never downsized: no room, no resident step.
    refused = gov.admit("m", requested_ctx=8192, caller="benchmark")
    assert refused.admitted is False


def test_hc20_the_dynamic_context_sizes_a_reload_with_the_memory_it_frees():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, dynamic_ctx_enabled=True))
    # 6 GiB loaded at 2048 (1.0 of KV): 5.0 of weights. With the 6 it frees,
    # 3.5 GiB are left for KV: 7168 tokens, the 4096 step.
    _snapshot(rg, gov, in_use=6.0, loaded=[_view(rg, "m", 6.0, context=2048)])
    decision = gov.admit("m", requested_ctx=8192, caller="chat")
    assert decision.admitted is True
    assert decision.num_ctx == 4096
    assert decision.load_expected is True


def test_hc21_an_eviction_counts_no_credit_for_a_model_gone_since_its_grant():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg))
    evicted = _with_idle_residents(rg, gov)
    grant = rg.AdmissionDecision(
        admitted=True, model="m", num_ctx=4096, conditional_on_eviction=True, load_expected=True,
        cost_gb=5.0, credit_gb=3.0,
    )
    # m is no longer loaded: nothing of it is freed, 5.0 against 2.5 free.
    gov._honour_conditional_eviction(grant)
    assert len(evicted) == 1


def test_hc22_a_model_is_never_its_own_eviction_candidate():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg))
    gov._warmup = _Warmup(keep_alive="10m")
    # m is idle and resident, its loaded context unknown: 4.0 of KV at 8192
    # against 1.5 free. Evicting itself would free nothing for itself.
    _snapshot(rg, gov, in_use=7.0, loaded=[_view(rg, "m", 7.0, idle=True)])
    decision = gov.admit("m", requested_ctx=8192, caller="benchmark")
    assert decision.conditional_on_eviction is False
    assert decision.partial_offload is True
