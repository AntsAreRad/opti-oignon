#!/usr/bin/env python3
"""What the governor promises when a model does not fit the GPU alone.

A model whose cost exceeded the free VRAM used to be refused, even when the
GPU and the system RAM together held it with room to spare; since the
capacity probe, the chat refused every model larger than the card. Ollama
places a model's layers between the two by itself. What the governor lacked
was the arithmetic, and a KV cost that knew the model it priced.

  The KV cache priced from the model's own geometry:
    * GO1 -- bytes per token = layers x KV heads x (key + value length) x 2
      (f16), from the model information Ollama reports.
    * GO2 -- an unreported head size is the embedding width over the heads,
      a model without a KV head count keeps one per head, and a per-layer
      KV head list is summed.
    * GO3 -- the same geometry is read from a GGUF file's header.
    * GO4 -- an operator override wins over the geometry, which wins over
      the flat coefficient.
    * GO5 -- the geometry is read once per model, and again after a load.

  Admission that knows the offload:
    * GO6 -- the GPU alone comes first: now, then after evicting idle models.
    * GO7 -- a model the GPU cannot hold even after eviction is admitted split
      when VRAM and RAM together hold it, at the requested context; the
      decision carries the GPU share, both costs, the layers on the GPU when
      their count is known, and no num_gpu.
    * GO8 -- the split counts only the VRAM free now; it never waits on an
      eviction.
    * GO9 -- when neither holds, the refusal names both shortfalls.
    * GO10 -- unreadable RAM means no split: the refusal is today's, word for
      word.
    * GO11 -- prefer: context keeps the requested context and splits before
      stepping down the ladder.
    * GO12 -- prefer: speed steps down the ladder on the GPU alone first and
      splits only at the last step, where the GPU holds the most.
    * GO13 -- callers without a floor are split at their own context, and the
      decision is recorded in the ring, whose columns do not change.
    * GO14 -- a split whose GPU share is under the configured minimum is
      refused, by name.
    * GO15 -- the split never eats the RAM reserve.
    * GO16 -- offload disabled gives back the historical decisions.

  The policy in YAML:
    * GO17 -- the offload block parses; the shipped file carries the decided
      defaults.
    * GO18 -- a value out of its range at load keeps the default.
    * GO19 -- the config routes show the block and write each key, in range
      only.

  The engines:
    * GO20 -- llama.cpp in process refuses a split by name before any load,
      and serves a model it already holds.
    * GO21 -- Ollama and llama-server run an admitted split as they are.

  The loaded view:
    * GO22 -- the total size Ollama reports is kept beside size_vram, from the
      ps answer through the warmup into the governor's view.
    * GO23 -- the learned cost is the total.
    * GO24 -- a split resident is resident, and only its VRAM part counts as
      VRAM in use.

  The reserve sized from the machine:
    * GO25 -- what GO17 pins, now that the shipped file leaves ram_reserve_gb
      null and the reserve is sized from the machine; GO17 stays in the tree
      word for word and is deselected by name.

Everything here is proven in the container, on scripted snapshots and fakes
loaded through the shared isolation window: no card, no socket, no model.
What an engine does with a split on the machine is owed there.
"""

import json
import sqlite3
import struct
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_RG = "opti_oignon.resource_governor"
_MM = "opti_oignon.model_manager"
_BACKEND = "opti_oignon.inference_backend"
_WARMUP = "opti_oignon.model_warmup"
_ROUTES = "opti_oignon.api.routes_governor"
# The two seams the governor resolves at the call: unreachable when the window
# opens, so neither an emergency stop nor a model window enters a decision.
_SEAMS = ("opti_oignon.context_manager", "opti_oignon.emergency_stop")
_GIB = 1024 ** 3
_CLOSERS = []


def _db_utils():
    """A db_utils stand-in whose safe_connect is plain sqlite."""
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda p, **kw: sqlite3.connect(
        str(p), check_same_thread=kw.get("check_same_thread", False)
    )
    return db


def _open(*extra, packages=()):
    """The governor and ``extra`` (name, path) modules after it, in one window."""
    targets = {_RG: source("resource_governor.py")}
    for name, path in extra:
        targets[name] = path
    loaded, restore = isolate(
        targets=targets,
        blocked=_SEAMS,
        seeded={"opti_oignon.db_utils": _db_utils()},
        packages=packages,
    )
    _CLOSERS.append(restore)
    return loaded


def _project_modules():
    """Every project entry of the module cache, by identity."""
    return {k: v for k, v in sys.modules.items() if k == "opti_oignon" or k.startswith("opti_oignon.")}


@pytest.fixture(autouse=True)
def _left_as_found():
    """No contract may leave a project module or the HTTP transport changed."""
    before = _project_modules()
    urlopen = urllib.request.urlopen
    yield
    urllib.request.urlopen = urlopen
    while _CLOSERS:
        _CLOSERS.pop()()
    assert _project_modules() == before, "every project module is left as the contract found it"


class _Clock:
    """A fixed monotonic stand-in: a hand-set snapshot never goes stale."""

    def __call__(self) -> float:
        return 1000.0


def _meminfo(directory: Path, ram_mb: float) -> str:
    """A meminfo file of our own: an absent path falls back to the host's RAM."""
    path = directory / "meminfo"
    path.write_text(f"MemAvailable:   {int(ram_mb * 1024)} kB\n", encoding="utf-8")
    return str(path)


def _config(rg, *, weights=None, capacity=10.0, **fields):
    """10 GiB card, 1.5 GiB margin: 8.5 GiB free on an empty card."""
    cfg = rg.GovernorConfig(
        total_vram_gb=capacity,
        safety_margin_gb=1.5,
        kv_coefficient=0.5,
        ctx_ladder=[8192, 4096, 2048],
        ctx_floor={"chat": 2048},
    )
    cfg.weights_override_models = dict(weights or {})
    for name, value in fields.items():
        setattr(cfg, name, value)
    return cfg


def _governor(rg, config=None, *, registry=None, warmup=None, ram_mb=64000.0):
    tmp = Path(tempfile.mkdtemp(prefix="go-"))
    gov = rg.ResourceGovernor(
        config_path=str(tmp / "missing.yaml"),
        db_path=str(tmp / "governor.db"),
        warmup=warmup,
        registry=registry,
        clock=_Clock(),
        meminfo_path=_meminfo(tmp, ram_mb),
        vram_probe=None,
    )
    if config is not None:
        gov._config = config
    return gov


def _snapshot(rg, gov, *, capacity=10.0, in_use=0.0, loaded=None, ram_mb=64000.0):
    """A hand-set snapshot the admission reads as it is."""
    gov._snapshot = rg.ResourceSnapshot(
        taken_at=1000.0,
        ttl_s=9999.0,
        loaded=list(loaded or []),
        capacity_gb=capacity,
        vram_in_use_gb=in_use,
        ram_available_mb=ram_mb,
    )
    return gov._snapshot


def _idle_view(rg, gb):
    """A resident whose epoch-0 expiry makes it idle past any threshold."""
    return rg.LoadedModelView(name="resident", size_vram_bytes=int(gb * _GIB), expires_at=0.0)


class _Warmup:
    """The warmup as the governor reads it: a keep_alive and a loaded set."""

    def __init__(self, loaded=None, keep_alive="10m"):
        self.keep_alive = keep_alive
        self._loaded = list(loaded or [])

    def get_loaded_models(self):
        return list(self._loaded)


def _resident(name, size_vram, size=None, digest=None):
    return types.SimpleNamespace(
        name=name,
        size_vram=size_vram,
        size=size,
        expires_at=None,
        context_length=None,
        digest=digest,
    )


class _Info:
    """A model_info answer: the raw mapping under ``extra``, and a file path."""

    def __init__(self, mapping=None, path=None):
        self._extra = {"model_info": dict(mapping)} if mapping is not None else {}
        self.path = path
        self.parameter_size = None
        self.quantization_level = None
        self.size = None
        self.extra_reads = 0

    @property
    def extra(self):
        self.extra_reads += 1
        return self._extra


class _Backend:
    def __init__(self, infos, name="ollama"):
        self.name = name
        self._infos = infos

    def model_info(self, model):
        return self._infos.get(model)


class _Registry:
    def __init__(self, *backends):
        self._backends = list(backends)

    def backends(self):
        return list(self._backends)


def _llama(*, layers, heads, kv_heads, embd, key=None, value=None):
    """The geometry keys Ollama's model_info and a GGUF header both carry."""
    mapping = {
        "general.architecture": "llama",
        "llama.block_count": layers,
        "llama.attention.head_count": heads,
        "llama.embedding_length": embd,
    }
    if kv_heads is not None:
        mapping["llama.attention.head_count_kv"] = kv_heads
    if key is not None:
        mapping["llama.attention.key_length"] = key
    if value is not None:
        mapping["llama.attention.value_length"] = value
    return mapping


def _gguf(path, fields):
    """A GGUF v3 header holding ``fields`` (strings and uint32) and no tensor."""

    def string(text):
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(fields))
    for key, value in fields.items():
        out += string(key)
        if isinstance(value, str):
            out += struct.pack("<I", 8) + string(value)
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", value)
    Path(path).write_bytes(out)


def _offload(cfg):
    return (
        cfg.offload_enabled,
        cfg.offload_prefer,
        cfg.offload_min_gpu_share,
        cfg.offload_ram_reserve_gb,
    )


# ---------------------------------------------------------------------------
# GO1-GO5 -- the KV cache priced from the model's own geometry
# ---------------------------------------------------------------------------


def test_go1_the_kv_cache_is_priced_from_the_layers_heads_and_lengths_ollama_reports():
    rg = _open()[_RG]
    info = _Info(_llama(layers=32, heads=32, kv_heads=8, embd=4096, key=64, value=64))
    gov = _governor(rg, _config(rg, weights={"m": 6.0}), registry=_Registry(_Backend({"m": info})))
    # 32 layers x 8 KV heads x (64 + 64) x 2 bytes = 65536 bytes per token:
    # 0.0625 GiB per 1024 tokens, where the flat coefficient says 0.5.
    assert gov.resolve_kv_coefficient("m") == 0.0625
    assert gov.estimate_kv_cache_gb(16384, model="m") == 1.0
    _snapshot(rg, gov)
    decision = gov.admit("m", requested_ctx=16384, caller="chat")
    # 6.0 + 1.0 = 7.0 fits the 8.5 GiB free; priced flat (6.0 + 8.0) it would not.
    assert decision.admitted is True
    assert decision.num_ctx == 16384
    assert decision.reason == "fits"


def test_go2_the_head_size_is_derived_when_unreported_and_per_layer_heads_are_summed():
    rg = _open()[_RG]
    infos = {
        # No key or value length: a head is 4096 / 32 = 128 wide.
        "derived": _Info(_llama(layers=32, heads=32, kv_heads=8, embd=4096)),
        # No KV head count: every head keeps its own keys and values.
        "mha": _Info(_llama(layers=16, heads=32, kv_heads=None, embd=4096)),
        # One KV head count per layer: the cache holds their sum.
        "per_layer": _Info(_llama(layers=4, heads=8, kv_heads=[8, 0, 8, 0], embd=1024, key=128, value=128)),
    }
    gov = _governor(rg, _config(rg), registry=_Registry(_Backend(infos)))
    # 32 x 8 x (128 + 128) x 2 = 131072 bytes per token.
    assert gov.resolve_kv_coefficient("derived") == 0.125
    # 16 x 32 x (128 + 128) x 2 = 262144 bytes per token.
    assert gov.resolve_kv_coefficient("mha") == 0.25
    # (8 + 0 + 8 + 0) x (128 + 128) x 2 = 8192 bytes per token.
    assert gov.resolve_kv_coefficient("per_layer") == 0.0078125


def test_go3_a_gguf_file_header_gives_the_same_geometry():
    loaded = _open((_MM, source("model_manager.py")))
    rg = loaded[_RG]
    path = Path(tempfile.mkdtemp(prefix="go-gguf-")) / "local.gguf"
    _gguf(path, _llama(layers=32, heads=32, kv_heads=8, embd=4096))
    # The in-process engine answers a file and no metadata mapping.
    backend = _Backend({"local.gguf": _Info(path=str(path))}, name="llama_cpp")
    gov = _governor(rg, _config(rg), registry=_Registry(backend))
    assert gov.resolve_kv_coefficient("local.gguf") == 0.125


def test_go4_an_operator_override_wins_over_the_geometry_which_wins_over_the_flat_rate():
    rg = _open()[_RG]
    geometry = _llama(layers=32, heads=32, kv_heads=8, embd=4096, key=64, value=64)
    infos = {name: _Info(geometry) for name in ("exact-model", "family-x:7b", "plain")}
    cfg = _config(rg)
    cfg.kv_override_models = {"exact-model": 0.3}
    cfg.kv_override_families = {"family-x": 0.2}
    gov = _governor(rg, cfg, registry=_Registry(_Backend(infos)))
    assert gov.resolve_kv_coefficient("exact-model") == 0.3
    assert gov.resolve_kv_coefficient("family-x:7b") == 0.2
    assert gov.resolve_kv_coefficient("plain") == 0.0625
    # Neither an override nor a geometry: the flat coefficient, as before.
    assert gov.resolve_kv_coefficient("unknown") == 0.5


def test_go5_the_geometry_is_read_once_per_model_and_again_after_a_load():
    rg = _open()[_RG]
    info = _Info(_llama(layers=32, heads=32, kv_heads=8, embd=4096, key=64, value=64))
    gov = _governor(rg, _config(rg, weights={"m": 1.0}), registry=_Registry(_Backend({"m": info})))
    _snapshot(rg, gov)
    gov.admit("m", requested_ctx=4096, caller="chat")
    gov.admit("m", requested_ctx=8192, caller="chat")
    assert info.extra_reads == 1
    gov.invalidate_on_load("m", 4096)
    _snapshot(rg, gov)
    gov.admit("m", requested_ctx=4096, caller="chat")
    assert info.extra_reads == 2


# ---------------------------------------------------------------------------
# GO6-GO16 -- admission that knows the offload
# ---------------------------------------------------------------------------


def test_go6_the_gpu_alone_comes_first_now_then_after_evicting_idle_models():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"m": 5.0}))
    _snapshot(rg, gov)
    whole = gov.admit("m", requested_ctx=4096, caller="chat")
    assert whole.admitted is True
    assert whole.reason == "fits"
    assert whole.gpu_share is None
    assert whole.partial_offload is False
    # An idle 4 GiB resident: 4.5 GiB free now, 8.5 after evicting it.
    gov = _governor(rg, _config(rg, weights={"m": 5.0}, idle_evict_threshold_s=600.0), warmup=_Warmup())
    _snapshot(rg, gov, in_use=4.0, loaded=[_idle_view(rg, 4.0)])
    evicting = gov.admit("m", requested_ctx=4096, caller="chat")
    # 7.0 does not fit 4.5 now and fits 8.5 after the eviction: the GPU alone,
    # granted on the eviction, before any split.
    assert evicting.admitted is True
    assert evicting.conditional_on_eviction is True
    assert evicting.gpu_share is None
    assert "partial_offload" not in evicting.reason


def test_go7_a_model_the_gpu_cannot_hold_is_admitted_split_at_the_requested_context():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"m": 9.0}))
    _snapshot(rg, gov)
    decision = gov.admit("m", requested_ctx=4096, caller="chat")
    # 9.0 + 2.0 = 11.0: 8.5 on the GPU, 2.5 in RAM (58.5 GiB usable).
    assert decision.admitted is True
    assert decision.action == "admit"
    assert decision.num_ctx == 4096
    assert decision.reason == "partial_offload"
    assert decision.partial_offload is True
    assert decision.vram_cost_gb == 8.5
    assert decision.ram_cost_gb == 2.5
    assert decision.gpu_share == pytest.approx(8.5 / 11.0)
    assert decision.gpu_layers is None  # no layer count is known
    assert decision.num_gpu is None  # Ollama places the layers itself
    assert decision.conditional_on_eviction is False
    # A known geometry names the layers on the GPU.
    info = _Info(_llama(layers=40, heads=32, kv_heads=8, embd=4096, key=128, value=128))
    gov = _governor(rg, _config(rg, weights={"m": 9.0}), registry=_Registry(_Backend({"m": info})))
    _snapshot(rg, gov)
    decision = gov.admit("m", requested_ctx=4096, caller="chat")
    # 40 x 8 x 256 x 2 bytes per token: 0.625 GiB at 4096 tokens, cost 9.625,
    # and 8.5 / 9.625 of 40 layers is 35.3: 35 layers on the GPU.
    assert decision.ram_cost_gb == 1.125
    assert decision.gpu_layers == 35
    carried = decision.to_dict()
    assert (carried["vram_cost_gb"], carried["ram_cost_gb"], carried["gpu_layers"]) == (8.5, 1.125, 35)
    assert carried["gpu_share"] == decision.gpu_share
    assert carried["num_gpu"] is None


def test_go8_the_split_counts_only_the_vram_free_now_and_never_waits_on_an_eviction():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"m": 9.0}, idle_evict_threshold_s=600.0), warmup=_Warmup())
    _snapshot(rg, gov, in_use=4.0, loaded=[_idle_view(rg, 4.0)])
    decision = gov.admit("m", requested_ctx=4096, caller="chat")
    # 11.0 fits neither 4.5 now nor 8.5 after the eviction. The split takes
    # the 4.5 free now and puts the other 6.5 in RAM, evicting nobody.
    assert decision.admitted is True
    assert decision.partial_offload is True
    assert decision.vram_cost_gb == 4.5
    assert decision.ram_cost_gb == 6.5
    assert decision.conditional_on_eviction is False


def test_go9_a_model_neither_holds_is_refused_with_both_shortfalls_named():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"m": 50.0}))
    # 16000 MB available: 15.625 GiB, less the 4 GiB reserve.
    _snapshot(rg, gov, ram_mb=16000.0)
    decision = gov.admit("m", requested_ctx=None, caller="chat")
    assert decision.admitted is False
    assert decision.action == "refuse"
    assert decision.reason == "vram_insufficient+ram_insufficient"
    # On the GPU alone, 50.0 - 8.5; split, 41.5 in RAM where 11.625 is usable.
    assert decision.shortfall_gb == 41.5
    assert decision.ram_shortfall_gb == 29.875
    payload = decision.refusal_payload()
    assert payload["shortfall_gb"] == 41.5
    assert payload["ram_shortfall_gb"] == 29.875
    assert "41.5 GB of VRAM" in payload["message"]
    assert "29.9 GB of RAM" in payload["message"]


def test_go10_unreadable_ram_means_no_split_and_the_historical_refusal():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"m": 50.0}))
    _snapshot(rg, gov, ram_mb=0.0)
    decision = gov.admit("m", requested_ctx=None, caller="chat")
    assert decision.admitted is False
    assert decision.reason == "vram_insufficient"
    assert decision.shortfall_gb == 41.5
    assert decision.ram_shortfall_gb is None
    payload = decision.refusal_payload()
    assert payload["message"] == (
        "Not enough resources to load m (short by 41.5 GB): evict idle models,"
        " pick a smaller model, or lower the context."
    )
    assert "ram_shortfall_gb" not in payload
    # A split was never priced: the RAM part is never a guess.
    assert decision.ram_cost_gb is None


def test_go11_prefer_context_splits_at_the_requested_context_before_stepping_down():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"m": 5.0}))
    _snapshot(rg, gov)
    decision = gov.admit("m", requested_ctx=8192, caller="chat")
    # 5.0 + 4.0 = 9.0 does not fit 8.5; 4096 would. The context is kept and
    # 0.5 GiB goes to RAM.
    assert gov.config.offload_prefer == "context"
    assert decision.admitted is True
    assert decision.action == "admit"
    assert decision.num_ctx == 8192
    assert decision.reason == "partial_offload"
    assert decision.vram_cost_gb == 8.5
    assert decision.ram_cost_gb == 0.5


def test_go12_prefer_speed_steps_down_on_the_gpu_first_and_splits_only_at_the_last_step():
    rg = _open()[_RG]
    gov = _governor(rg, _config(rg, weights={"m": 5.0}, offload_prefer="speed"))
    _snapshot(rg, gov)
    stepped = gov.admit("m", requested_ctx=8192, caller="chat")
    assert stepped.admitted is True
    assert stepped.action == "downsize"
    assert stepped.num_ctx == 4096
    assert stepped.reason == "ctx_laddered_to_fit"
    assert stepped.gpu_share is None
    # 8.0 GiB of weights: 12.0, 10.0 and 9.0 at the three steps, none on the
    # GPU alone. The split is taken at the last step, the GPU's largest share.
    gov = _governor(rg, _config(rg, weights={"m": 8.0}, offload_prefer="speed"))
    _snapshot(rg, gov)
    split = gov.admit("m", requested_ctx=8192, caller="chat")
    assert split.admitted is True
    assert split.action == "downsize"
    assert split.num_ctx == 2048
    assert split.reason == "ctx_laddered_to_fit+partial_offload"
    assert split.vram_cost_gb == 8.5
    assert split.ram_cost_gb == 0.5


def test_go13_callers_without_a_floor_are_split_at_their_context_and_recorded():
    rg = _open()[_RG]
    for caller in ("benchmark", "agent_eval", "direct"):
        gov = _governor(rg, _config(rg, weights={"m": 5.0}))
        _snapshot(rg, gov)
        decision = gov.admit("m", requested_ctx=8192, caller=caller)
        assert decision.admitted is True, caller
        assert decision.action == "admit", caller
        assert decision.num_ctx == 8192, caller
        assert decision.reason == "partial_offload", caller
        row = gov.store.recent_decisions(1)[0]
        assert {k: row[k] for k in ("caller", "model", "requested_ctx", "admitted_ctx", "decision", "reason")} == {
            "caller": caller,
            "model": "m",
            "requested_ctx": 8192,
            "admitted_ctx": 8192,
            "decision": "admit",
            "reason": "partial_offload",
        }
    conn = sqlite3.connect(gov.store._db_path)
    try:
        columns = [r[1] for r in conn.execute("PRAGMA table_info(decisions)")]
    finally:
        conn.close()
    assert columns == ["id", "ts", "caller", "model", "requested_ctx", "admitted_ctx", "decision", "reason"]


def test_go14_a_split_under_the_minimum_gpu_share_is_refused_by_name():
    rg = _open()[_RG]
    # 11.0 GiB at 4096 tokens: the GPU would hold 8.5 / 11.0, 77 per cent.
    # A caller without a floor has no lower step to fall back on.
    gov = _governor(rg, _config(rg, weights={"m": 9.0}, offload_min_gpu_share=0.78))
    _snapshot(rg, gov)
    refused = gov.admit("m", requested_ctx=4096, caller="benchmark")
    assert refused.admitted is False
    assert refused.reason == "vram_insufficient+gpu_share_below_minimum"
    assert refused.ram_shortfall_gb == 0.0
    message = refused.refusal_payload()["message"]
    assert "77%" in message and "78%" in message
    gov = _governor(rg, _config(rg, weights={"m": 9.0}, offload_min_gpu_share=0.77))
    _snapshot(rg, gov)
    admitted = gov.admit("m", requested_ctx=4096, caller="benchmark")
    assert admitted.admitted is True
    assert admitted.partial_offload is True


def test_go15_the_split_never_eats_the_ram_reserve():
    rg = _open()[_RG]
    # 2.5 GiB must go to RAM. 6656 MB is 6.5 GiB: 2.5 usable over a 4.0
    # reserve. A caller without a floor has no lower step to fall back on.
    gov = _governor(rg, _config(rg, weights={"m": 9.0}))
    _snapshot(rg, gov, ram_mb=6656.0)
    assert gov.admit("m", requested_ctx=4096, caller="benchmark").admitted is True
    gov = _governor(rg, _config(rg, weights={"m": 9.0}, offload_ram_reserve_gb=4.5))
    _snapshot(rg, gov, ram_mb=6656.0)
    refused = gov.admit("m", requested_ctx=4096, caller="benchmark")
    assert refused.admitted is False
    assert refused.reason == "vram_insufficient+ram_insufficient"
    assert refused.ram_shortfall_gb == 0.5


def test_go16_offload_disabled_gives_back_the_historical_decisions():
    rg = _open()[_RG]
    # The frame of C2: 50 GiB on an 8 GiB card.
    gov = _governor(rg, _config(rg, weights={"bigmodel": 50.0}, capacity=8.0, offload_enabled=False))
    _snapshot(rg, gov, capacity=8.0)
    refused = gov.admit("bigmodel", requested_ctx=None, caller="chat")
    assert refused.admitted is False
    assert refused.reason == "vram_insufficient"
    assert refused.shortfall_gb == 43.5
    assert refused.ram_shortfall_gb is None
    # The frame of D1 and D2: 9.0 GiB at 8192 tokens against 8.5 free.
    gov = _governor(rg, _config(rg, weights={"m": 5.0}, offload_enabled=False))
    _snapshot(rg, gov)
    stepped = gov.admit("m", requested_ctx=8192, caller="chat")
    assert stepped.action == "downsize"
    assert stepped.num_ctx == 4096
    assert stepped.reason == "ctx_laddered_to_fit"
    direct = gov.admit("m", requested_ctx=8192, caller="direct")
    assert direct.admitted is False
    assert direct.reason == "vram_insufficient"
    assert direct.shortfall_gb == 0.5


# ---------------------------------------------------------------------------
# GO17-GO19 -- the policy in YAML
# ---------------------------------------------------------------------------


def test_go17_the_offload_block_parses_and_the_shipped_file_carries_the_decided_defaults():
    rg = _open()[_RG]
    tmp = Path(tempfile.mkdtemp(prefix="go-cfg-"))
    path = tmp / "resource_governor.yaml"
    path.write_text(
        "offload:\n  enabled: false\n  prefer: speed\n  min_gpu_share: 0.25\n  ram_reserve_gb: 8.0\n",
        encoding="utf-8",
    )
    assert _offload(rg.load_config(path)) == (False, "speed", 0.25, 8.0)
    assert _offload(rg.load_config(tmp / "missing.yaml")) == (True, "context", 0.0, 4.0)
    shipped = Path(source("config", "resource_governor.yaml"))
    assert _offload(rg.load_config(shipped)) == (True, "context", 0.0, 4.0)
    raw = yaml.safe_load(shipped.read_text(encoding="utf-8"))
    assert raw["offload"] == {"enabled": True, "prefer": "context", "min_gpu_share": 0.0, "ram_reserve_gb": 4.0}


def test_go18_a_value_out_of_its_range_at_load_keeps_the_default():
    rg = _open()[_RG]
    tmp = Path(tempfile.mkdtemp(prefix="go-range-"))
    path = tmp / "resource_governor.yaml"
    cases = [
        ("offload:\n  prefer: fast\n  min_gpu_share: 1.5\n  ram_reserve_gb: -1\n", (True, "context", 0.0, 4.0)),
        ("offload:\n  min_gpu_share: -0.1\n  ram_reserve_gb: .nan\n", (True, "context", 0.0, 4.0)),
        ("offload:\n  prefer: 3\n  min_gpu_share: half\n  ram_reserve_gb: .inf\n", (True, "context", 0.0, 4.0)),
        ("offload: yes\n", (True, "context", 0.0, 4.0)),
        # The edges are in range.
        ("offload:\n  min_gpu_share: 1.0\n  ram_reserve_gb: 0.0\n", (True, "context", 1.0, 0.0)),
    ]
    for text, expected in cases:
        path.write_text(text, encoding="utf-8")
        assert _offload(rg.load_config(path)) == expected, text


def test_go19_the_config_routes_show_the_block_and_write_each_key_in_range_only():
    loaded = _open((_ROUTES, source("api", "routes_governor.py")), packages=("opti_oignon.api",))
    rg, routes = loaded[_RG], loaded[_ROUTES]
    gov = _governor(rg, rg.GovernorConfig())
    view = routes.config_read_payload(gov)
    assert view["config"]["offload"] == {
        "enabled": True,
        "prefer": "context",
        "min_gpu_share": 0.0,
        "ram_reserve_gb": 4.0,
    }
    keys = {"offload.enabled", "offload.prefer", "offload.min_gpu_share", "offload.ram_reserve_gb"}
    assert keys <= set(view["writable_keys"])
    path = Path(tempfile.mkdtemp(prefix="go-api-")) / "resource_governor.yaml"
    path.write_text(Path(source("config", "resource_governor.yaml")).read_text(encoding="utf-8"), encoding="utf-8")
    resets, audits = [], []
    out = routes.config_write_payload(
        gov.config,
        {"offload.enabled": False, "offload.prefer": "speed", "offload.min_gpu_share": 0.25, "offload.ram_reserve_gb": 6.0},
        path,
        lambda: resets.append(True),
        audits.append,
    )
    assert out["applied"]["offload.prefer"] == {"old": "context", "new": "speed"}
    assert _offload(rg.load_config(path)) == (False, "speed", 0.25, 6.0)
    assert resets == [True] and len(audits) == 1
    before = path.read_bytes()
    for bad in (
        {"offload.min_gpu_share": 1.5},
        {"offload.min_gpu_share": -0.5},
        {"offload.ram_reserve_gb": -1.0},
        {"offload.ram_reserve_gb": float("inf")},
        {"offload.prefer": "fast"},
    ):
        with pytest.raises(routes.ConfigWriteError) as refused:
            routes.config_write_payload(gov.config, bad, path, lambda: None, lambda changes: None)
        assert refused.value.status_code == 400, bad
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# GO20-GO21 -- the engines
# ---------------------------------------------------------------------------


class _GateGovernor:
    """What backend_admission_gate touches, with its admission scripted."""

    def __init__(self, decision):
        self.config = types.SimpleNamespace(enabled=True)
        self._decision = decision
        self.recorded = []
        self.loads = []

    def admit(self, model, requested, caller="direct"):
        return self._decision

    def _record_admission(self, decision):
        self.recorded.append(decision)

    def invalidate_on_load(self, model, num_ctx):
        self.loads.append((model, num_ctx))

    def _honour_conditional_eviction(self, decision):  # pragma: no cover
        raise AssertionError("a split is never conditional on an eviction")


def _split(rg, model):
    return rg.AdmissionDecision(
        admitted=True,
        model=model,
        num_ctx=4096,
        action="admit",
        reason="partial_offload",
        caller="chat",
        requested_ctx=4096,
        load_expected=True,
        gpu_share=8.5 / 11.0,
        vram_cost_gb=8.5,
        ram_cost_gb=2.5,
    )


class _Llama:
    """The in-process engine's model class; constructing it is a load."""

    built = []

    def __init__(self, **kwargs):
        _Llama.built.append(kwargs)

    def create_chat_completion(self, **kwargs):
        if kwargs.get("stream"):
            return iter([{"choices": [{"delta": {"content": "served"}, "finish_reason": "stop"}]}])
        return {"choices": [{"message": {"content": "served"}}]}


def test_go20_llama_cpp_in_process_refuses_a_split_by_name_before_any_load_and_serves_a_resident():
    loaded = _open((_BACKEND, source("inference_backend.py")))
    rg, mod = loaded[_RG], loaded[_BACKEND]
    mod._live_mode = lambda: "daily"
    mod.LLAMA_CPP_AVAILABLE = True
    _Llama.built = []
    mod._LlamaCpp = _Llama
    models = Path(tempfile.mkdtemp(prefix="go-models-"))
    (models / "big.gguf").write_bytes(b"GGUF")
    backend = mod.LlamaCppBackend(model_dirs=[str(models)])  # every layer on the GPU
    fake = _GateGovernor(_split(rg, "big.gguf"))
    rg._governor = fake
    messages = [{"role": "user", "content": "hi"}]
    try:
        with pytest.raises(rg.GovernorRefusal) as refused:
            backend.generate("big.gguf", messages)
        text = str(refused.value)
        assert "llama.cpp" in text and "split" in text and "Ollama" in text
        assert refused.value.decision.admitted is False
        assert refused.value.decision.reason == "partial_offload_unsupported"
        with pytest.raises(rg.GovernorRefusal):
            list(backend.stream("big.gguf", messages))
        assert _Llama.built == []  # no load was attempted
        assert fake.loads == []  # nothing was accounted
        assert [d.reason for d in fake.recorded] == ["partial_offload_unsupported"] * 2
        # A held ticket is refused the same way, before its load is accounted.
        ticket = _split(rg, "big.gguf")
        rg.set_active_ticket(ticket)
        try:
            with pytest.raises(rg.GovernorRefusal):
                backend.generate("big.gguf", messages)
        finally:
            rg.clear_active_ticket()
        assert ticket.load_expected is True
        assert fake.loads == []
        # A model the engine already holds loads nothing: it is served.
        backend._loaded_models["big.gguf"] = _Llama.__new__(_Llama)
        assert backend.generate("big.gguf", messages).content == "served"
        assert _Llama.built == []
        # An explicit layer count is the operator's own placement: it loads.
        # (The file's integrity gate is not what this contract is about.)
        mod._provenance_guard = lambda path: None
        placed = mod.LlamaCppBackend(model_dirs=[str(models)], n_gpu_layers=20)
        assert placed.generate("big.gguf", messages).content == "served"
        assert [built["n_gpu_layers"] for built in _Llama.built] == [20]
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


class _Ollama:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": "local"}}


class _Server:
    def __init__(self):
        self.urls = []

    def urlopen(self, req, timeout=None):
        self.urls.append(req.full_url)
        body = json.dumps({"choices": [{"message": {"content": "remote"}}]}).encode("utf-8")
        return _Response(body)


class _Response:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_go21_ollama_and_llama_server_run_an_admitted_split_as_they_are():
    loaded = _open((_BACKEND, source("inference_backend.py")))
    rg, mod = loaded[_RG], loaded[_BACKEND]
    mod._live_mode = lambda: "daily"
    fake = _GateGovernor(_split(rg, "big"))
    rg._governor = fake
    messages = [{"role": "user", "content": "hi"}]
    try:
        client = _Ollama()
        mod.OLLAMA_AVAILABLE = True
        mod._ollama_module = client
        reply = mod.OllamaBackend().generate("big", messages, options={"num_ctx": 4096})
        assert reply.content == "local"
        # Ollama places the layers itself: nothing beyond the context is sent.
        assert client.calls[0]["options"] == {"num_ctx": 4096}
        assert fake.loads == [("big", 4096)]
        server = _Server()
        mod.urllib.request.urlopen = server.urlopen
        reply = mod.LlamaServerBackend(host="http://127.0.0.1:8080").generate("big", messages)
        assert reply.content == "remote"
        assert server.urls == ["http://127.0.0.1:8080/v1/chat/completions"]
        assert fake.recorded == []  # nobody refused
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


# ---------------------------------------------------------------------------
# GO22-GO24 -- the loaded view
# ---------------------------------------------------------------------------


class _Bytes:
    """The client's ByteSize: an int through __int__."""

    def __init__(self, value):
        self._value = value

    def __int__(self):
        return self._value


def test_go22_the_total_size_is_kept_beside_size_vram_from_ps_to_the_governor():
    loaded = _open((_BACKEND, source("inference_backend.py")))
    rg, mod = loaded[_RG], loaded[_BACKEND]
    as_dict = mod._loaded_model_from_ps({"model": "a", "size_vram": 6 * _GIB, "size": 10 * _GIB}, "ollama")
    assert as_dict.size == 10 * _GIB
    assert as_dict.to_dict()["size"] == 10 * _GIB
    as_object = mod._loaded_model_from_ps(
        types.SimpleNamespace(name="b", model="b", size_vram=_Bytes(1), size=_Bytes(7 * _GIB)), "ollama"
    )
    assert as_object.size == 7 * _GIB and isinstance(as_object.size, int)
    assert mod._loaded_model_from_ps({"model": "c", "size_vram": 1}, "ollama").size is None
    # The warmup carries it.
    records = [_resident("a", 6 * _GIB, 10 * _GIB)]
    seeded = types.ModuleType(_BACKEND)
    seeded.get_backend_registry = lambda: types.SimpleNamespace(
        active=types.SimpleNamespace(name="ollama", loaded_models=lambda: list(records))
    )
    warm, restore = isolate(targets={_WARMUP: source("model_warmup.py")}, seeded={_BACKEND: seeded})
    _CLOSERS.append(restore)
    models = warm[_WARMUP].ModelWarmup().get_loaded_models()
    assert [(m.name, m.size_vram, m.size) for m in models] == [("a", 6 * _GIB, 10 * _GIB)]
    # The governor's view carries it.
    gov = _governor(rg, _config(rg, capacity=24.0), warmup=_Warmup(records))
    snap = gov.refresh(force=True)
    assert snap.loaded[0].size_bytes == 10 * _GIB
    assert snap.to_dict()["loaded"][0]["size_bytes"] == 10 * _GIB


def test_go23_the_learned_cost_is_the_total_and_size_vram_only_when_no_total_is_said():
    rg = _open()[_RG]
    warmup = _Warmup([_resident("m", 6 * _GIB, 10 * _GIB, "d1"), _resident("n", 3 * _GIB, None, "d2")])
    gov = _governor(rg, _config(rg, capacity=24.0), warmup=warmup)
    gov.invalidate_on_load("m", 4096)
    gov.invalidate_on_load("n", 2048)
    gov.refresh(force=True)
    assert gov.store.get_model_cost("m", "d1")["size_vram_bytes"] == 10 * _GIB
    assert gov.store.get_model_cost("n", "d2")["size_vram_bytes"] == 3 * _GIB


def test_go24_a_split_resident_is_resident_and_only_its_vram_part_is_vram_in_use():
    rg = _open()[_RG]
    # m: 6 GiB on the GPU, 4 in RAM. c: entirely in RAM.
    warmup = _Warmup([_resident("m", 6 * _GIB, 10 * _GIB), _resident("c", 0, 4 * _GIB)])
    gov = _governor(rg, _config(rg, weights={"m": 9.0}), warmup=warmup)
    snap = gov.refresh(force=True)
    assert snap.vram_in_use_gb == 6.0
    views = {v["name"]: v for v in snap.to_dict()["loaded"]}
    assert views["m"]["ram_bytes"] == 4 * _GIB
    assert views["c"]["ram_bytes"] == 4 * _GIB
    again = gov.admit("m", requested_ctx=2048, caller="chat")
    # Resident: no weight cost, 1.0 GiB of KV against the 2.5 left.
    assert again.admitted is True
    assert again.load_expected is False
    assert again.gpu_share is None
    in_ram = gov.admit("c", requested_ctx=2048, caller="chat")
    assert in_ram.admitted is True
    assert in_ram.load_expected is False


def test_go25_the_offload_block_parses_and_the_shipped_file_sizes_the_reserve_from_the_machine():
    rg = _open()[_RG]
    tmp = Path(tempfile.mkdtemp(prefix="go-cfg-"))
    path = tmp / "resource_governor.yaml"
    path.write_text(
        "offload:\n  enabled: false\n  prefer: speed\n  min_gpu_share: 0.25\n  ram_reserve_gb: 8.0\n",
        encoding="utf-8",
    )
    assert _offload(rg.load_config(path)) == (False, "speed", 0.25, 8.0)
    assert _offload(rg.load_config(tmp / "missing.yaml")) == (True, "context", 0.0, 4.0)
    shipped = Path(source("config", "resource_governor.yaml"))
    # null: the reserve is sized from the machine (the ram_reserve block).
    assert _offload(rg.load_config(shipped)) == (True, "context", 0.0, None)
    raw = yaml.safe_load(shipped.read_text(encoding="utf-8"))
    assert raw["offload"] == {"enabled": True, "prefer": "context", "min_gpu_share": 0.0, "ram_reserve_gb": None}
