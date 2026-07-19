#!/usr/bin/env python3
"""What the governor promises about a dynamic, ladder-quantized context.

A context size that drifts with every turn poisons the model's prompt
cache: the engine re-plans its KV allocation on every change of quantum.
The dynamic stage pinned here removes the drift without touching a single
decision the governor makes today when the stage is off.

Off means byte-identical. With the stage disabled -- the shipped default
-- a fitting request is admitted at its requested value verbatim, and an
unknown capacity still admits the request untouched. Nothing about the
historical decisions moves.

On means quantized and capped. The admitted context lands on the config
ladder: a request between steps rounds UP to the next step, so a growing
conversation keeps one stable quantum for many turns. A live ceiling is
derived from the VRAM left after weights through the model's KV
coefficient, itself landed on the ladder; a request above the ladder's
top answers the top. The stage never refuses: a budget below the lowest
step still answers the lowest step and leaves refusal to the fit math,
which keeps its own contract.

Unknown capacity is capped, not refused. When the machine cannot be
measured, the admission itself stays deliberately fail-open -- but the
context it grants is the configured conservative ceiling, never the raw
request. The fail-open half decides WHETHER; the fail-secure half decides
HOW MUCH.

The model window still rules first, the per-caller floors and the
downsize ladder keep working underneath, and the config block parses with
honest defaults. Loaded through the shared isolation window with a
scripted snapshot; no card, no socket, no database outside a temp path is
ever touched.
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_GOVERNOR = "opti_oignon.resource_governor"


def _load_governor(
    *,
    dynamic=False,
    unknown_ceiling=8192,
    capacity=24.0,
    in_use=2.0,
    weights=4.0,
    ladder=(32768, 16384, 8192, 4096),
    window=131072,
    kv=0.5,
    margin=1.5,
):
    loaded, restore = isolate(
        targets={_GOVERNOR: source("resource_governor.py")},
        seeded={},
        packages=("opti_oignon",),
    )
    rg = loaded[_GOVERNOR]
    tmp = tempfile.mkdtemp(prefix="dyn-ctx-")
    gov = rg.ResourceGovernor(
        config_path=str(Path(tmp) / "missing.yaml"),
        db_path=str(Path(tmp) / "adapt.db"),
    )
    cfg = gov._config
    cfg.enabled = True
    cfg.safety_margin_gb = margin
    cfg.kv_coefficient = kv
    cfg.ctx_ladder = list(ladder)
    cfg.ctx_floor = {"chat": min(ladder), "pipeline": min(ladder)}
    cfg.dynamic_ctx_enabled = bool(dynamic)
    cfg.dynamic_ctx_unknown_ceiling = int(unknown_ceiling)

    snap = SimpleNamespace(
        capacity_gb=capacity,
        vram_in_use_gb=in_use,
        ram_available_mb=32000.0,
        loaded=[],
        sources=["scripted"],
        expirations={},
    )
    gov.get_snapshot_fast = lambda: snap
    gov._pressure_from_snapshot = lambda s: {"level": "none"}
    gov._evictable_now_gb = lambda s: 0.0
    gov.estimate_model_vram_gb = lambda *a, **k: (weights, "scripted")
    gov.resolve_weights_override = lambda m: None
    rg._resolve_emergency_stop = lambda: None
    rg._resolve_context_manager = lambda: SimpleNamespace(
        get_model_limits=lambda m: SimpleNamespace(context_window=window)
    )
    return rg, gov, restore


# ---------------------------------------------------------------------------
# dc1 -- control: stage off, a fitting request passes verbatim
# ---------------------------------------------------------------------------

def test_dc1_stage_off_admits_the_requested_value_verbatim():
    rg, gov, restore = _load_governor(dynamic=False)
    try:
        decision = gov.admit("m", requested_ctx=5200, caller="chat")
        assert decision.admitted is True
        assert decision.num_ctx == 5200
        assert decision.reason == "fits"
    finally:
        restore()


# ---------------------------------------------------------------------------
# dc2 -- stage on: a request between steps rounds UP to the next step
# ---------------------------------------------------------------------------

def test_dc2_stage_on_rounds_the_request_up_to_the_next_ladder_step():
    rg, gov, restore = _load_governor(dynamic=True)
    try:
        decision = gov.admit("m", requested_ctx=5200, caller="chat")
        assert decision.admitted is True
        assert decision.num_ctx == 8192
        assert "ctx_quantized" in decision.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# dc3 -- stage on: two growing turns land on the same quantum
# ---------------------------------------------------------------------------

def test_dc3_growing_turns_share_one_stable_quantum():
    rg, gov, restore = _load_governor(dynamic=True)
    try:
        d1 = gov.admit("m", requested_ctx=5200, caller="chat")
        d2 = gov.admit("m", requested_ctx=7400, caller="chat")
        assert d1.num_ctx == d2.num_ctx == 8192
    finally:
        restore()


# ---------------------------------------------------------------------------
# dc4 -- stage on: the live ceiling caps a huge request to the fitting step
# ---------------------------------------------------------------------------

def test_dc4_live_ceiling_caps_the_request_to_the_highest_fitting_step():
    # kv budget = 24 - 2 - 1.5 - 4 = 16.5 GiB -> 33792 tokens -> step 32768
    rg, gov, restore = _load_governor(dynamic=True)
    try:
        decision = gov.admit("m", requested_ctx=100000, caller="chat")
        assert decision.admitted is True
        assert decision.num_ctx == 32768
        assert "ctx_quantized" in decision.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# dc5 -- stage on, capacity unknown: capped to the configured ceiling
# ---------------------------------------------------------------------------

def test_dc5_unknown_capacity_admits_but_caps_to_the_configured_ceiling():
    rg, gov, restore = _load_governor(
        dynamic=True, capacity=None, unknown_ceiling=8192
    )
    try:
        decision = gov.admit("m", requested_ctx=262144, caller="chat")
        assert decision.admitted is True, "the admission half stays fail-open"
        assert decision.num_ctx == 8192, "the context half is fail-secure"
        assert decision.reason == "capacity_unknown_fail_open+ctx_capped"
    finally:
        restore()


# ---------------------------------------------------------------------------
# dc6 -- stage on: a request already on the ladder is not inflated
# ---------------------------------------------------------------------------

def test_dc6_a_request_already_on_a_step_passes_unchanged():
    rg, gov, restore = _load_governor(dynamic=True)
    try:
        decision = gov.admit("m", requested_ctx=4096, caller="chat")
        assert decision.num_ctx == 4096
        assert "ctx_quantized" not in decision.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# dc7 -- stage on: above the ladder's top, the top answers
# ---------------------------------------------------------------------------

def test_dc7_a_request_above_the_ladder_top_lands_on_the_top():
    rg, gov, restore = _load_governor(dynamic=True, capacity=200.0)
    try:
        decision = gov.admit("m", requested_ctx=90000, caller="chat")
        assert decision.num_ctx == 32768
        assert "ctx_quantized" in decision.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# dc8 -- stage on: a starved budget answers the lowest step, never refuses
# ---------------------------------------------------------------------------

def test_dc8_a_starved_budget_answers_the_lowest_step_and_lets_fit_refuse():
    # kv budget = 8 - 2 - 1.5 - 4 = 0.5 GiB -> 1024 tokens < lowest step.
    # The stage still answers 4096; cost(4096) = 4 + 2 = 6 > 4.5 budget,
    # so the FIT math refuses -- the stage itself never does.
    rg, gov, restore = _load_governor(dynamic=True, capacity=8.0)
    try:
        decision = gov.admit("m", requested_ctx=5200, caller="chat")
        assert decision.admitted is False
        assert decision.reason == "vram_insufficient"
    finally:
        restore()


# ---------------------------------------------------------------------------
# dc9 -- control: the model window still clamps first, both modes agree
# ---------------------------------------------------------------------------

def test_dc9_model_window_clamp_still_rules_first():
    rg, gov, restore = _load_governor(dynamic=True, window=32768, capacity=200.0)
    try:
        decision = gov.admit("m", requested_ctx=999999, caller="chat")
        assert decision.num_ctx == 32768
        assert "clamped_to_model_limit" in decision.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# dc10 -- stage on: the downsize ladder keeps working underneath
# ---------------------------------------------------------------------------

def test_dc10_downsize_ladder_still_works_under_the_stage():
    # kv budget = 10 - 2 - 1.5 - 4 = 2.5 GiB -> 5120 tokens -> ceiling 4096.
    # Request 5200 quantizes toward 8192 but the ceiling lands it on 4096;
    # cost(4096) = 4 + 2 = 6 <= 6.5 budget -> admitted at the floor step.
    rg, gov, restore = _load_governor(dynamic=True, capacity=10.0)
    try:
        decision = gov.admit("m", requested_ctx=5200, caller="chat")
        assert decision.admitted is True
        assert decision.num_ctx == 4096
        assert "ctx_quantized" in decision.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# dc11 -- control: stage off, unknown capacity still passes verbatim
# ---------------------------------------------------------------------------

def test_dc11_stage_off_unknown_capacity_keeps_the_historical_passthrough():
    # The model-window clamp runs before the unknown-capacity branch, so
    # the historical passthrough hands over the POST-CLAMP value: with a
    # 131072 window, a 262144 request is admitted at 131072 untouched.
    rg, gov, restore = _load_governor(dynamic=False, capacity=None)
    try:
        decision = gov.admit("m", requested_ctx=262144, caller="chat")
        assert decision.admitted is True
        assert decision.num_ctx == 131072
        assert decision.reason == "capacity_unknown_fail_open"
    finally:
        restore()


# ---------------------------------------------------------------------------
# dc12 -- the config block parses, with honest defaults when absent
# ---------------------------------------------------------------------------

def test_dc12_config_block_parses_and_defaults_stay_conservative():
    rg, gov, restore = _load_governor()
    try:
        tmp = Path(tempfile.mkdtemp(prefix="dyn-cfg-"))
        cfg_path = tmp / "resource_governor.yaml"
        cfg_path.write_text(
            "enabled: true\n"
            "dynamic_ctx:\n"
            "  enabled: true\n"
            "  unknown_ctx_ceiling: 6144\n",
            encoding="utf-8",
        )
        cfg = rg.load_config(cfg_path)
        assert cfg.dynamic_ctx_enabled is True
        assert cfg.dynamic_ctx_unknown_ceiling == 6144

        bare = rg.load_config(tmp / "missing.yaml")
        assert bare.dynamic_ctx_enabled is False
        assert bare.dynamic_ctx_unknown_ceiling == 8192
    finally:
        restore()
