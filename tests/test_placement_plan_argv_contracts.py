#!/usr/bin/env python3
"""Contracts for a measured placement expressed as a llama-server command.

The repository never launches llama-server -- the process is started host-side
and consumed through LlamaServerBackend. A placement plan is therefore a
RECIPE: a description that resolves to an argv, reproducibly, so that the same
plan can be run again and compared. Until now the argv builder could express
none of the placement this block exists for. Of the six placements named in the
work order, three had no representation anywhere in the tree, and the other
three existed only as function parameters with no configuration path.

These contracts pin the shape of a plan and the argv it resolves to. They pin
nothing about which placement is fast: that is machine work, it is owed, and no
contract here claims otherwise.

  * PL1 -- experts are routed off the card only when the plan says so, and the
    routing expression reaches the command verbatim.
  * PL2 -- the target's own layer count reaches the card. Before this, the
    builder emitted a layer count for the DRAFT model only.
  * PL3 -- the KV cache is kept off the card only when the plan says so.
  * PL4 -- the context extension terms travel together, and only when asked.
  * PL5 -- flash attention and the two KV cache types come from the plan, so a
    placement is expressible without editing a call site.
  * PL6 -- an unrecognised KV cache type is refused loudly. The in-process
    backend deliberately fails open here, because a perf knob must not block a
    load; a recipe has no library to ask and no load to protect, so the same
    input is a refusal rather than a silent downgrade to the default.
  * PL7 -- the recipe is reproducible: the same plan and inputs answer the
    identical argv, and the builder stays pure.
  * PL8 -- a plan is loaded from YAML, and a per-model entry overrides the
    defaults rather than replacing or ignoring them.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window. The builder is pure by contract, so nothing here spawns a
process, reads a model, or touches a device.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _open():
    """Open the shared window on the placement module and the argv builder."""
    loaded, restore = isolate(
        targets={
            "opti_oignon.placement": source("placement.py"),
            "opti_oignon.speculative_decoding": source(
                "speculative_decoding.py",
            ),
        },
    )
    return (
        loaded["opti_oignon.placement"],
        loaded["opti_oignon.speculative_decoding"],
        restore,
    )


def _bare_config(sd):
    """A speculative config that contributes no draft flags of its own."""
    return sd.SpeculativeConfig(enabled=False)


def _build(sd, placement=None, **kw):
    return sd.build_llama_server_command(
        "/models/target.gguf", _bare_config(sd), placement=placement, **kw,
    )


def _pairs(cmd):
    """The argv as a set of tokens, for presence questions."""
    return list(cmd)


# ---------------------------------------------------------------------------
# PL1 -- experts leave the card only on request
# ---------------------------------------------------------------------------
def test_pl1_experts_are_routed_off_the_card_only_when_asked():
    pm, sd, restore = _open()
    try:
        without = _build(sd, pm.PlacementPlan())
        assert "-ot" not in without, (
            "a plan that says nothing about experts emits no routing flag"
        )
        assert "--n-cpu-moe" not in without, (
            "nor the count form of the same intent"
        )

        plan = pm.PlacementPlan(override_tensor=r"\.ffn_.*_exps\.=CPU")
        with_experts = _build(sd, plan)
        assert "-ot" in with_experts, "the routing flag is emitted on request"
        idx = with_experts.index("-ot")
        assert with_experts[idx + 1] == r"\.ffn_.*_exps\.=CPU", (
            "the routing expression reaches the command verbatim, since it is "
            "the operator's own and cannot be second-guessed here"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# PL2 -- the target's layers reach the card
# ---------------------------------------------------------------------------
def test_pl2_the_target_layer_count_reaches_the_card():
    pm, sd, restore = _open()
    try:
        without = _build(sd, pm.PlacementPlan())
        assert "-ngl" not in without, (
            "a plan that says nothing about layers emits no layer flag"
        )

        cmd = _build(sd, pm.PlacementPlan(n_gpu_layers=99))
        assert "-ngl" in cmd, (
            "the target's layer count is expressible; before this the builder "
            "emitted a layer count for the draft model only"
        )
        assert cmd[cmd.index("-ngl") + 1] == "99", "the count travels with it"
        assert isinstance(cmd[cmd.index("-ngl") + 1], str), (
            "every argv term is a string, as the rest of the command is"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# PL3 -- the KV cache stays off the card only on request
# ---------------------------------------------------------------------------
def test_pl3_kv_offload_is_refused_only_when_asked():
    pm, sd, restore = _open()
    try:
        without = _build(sd, pm.PlacementPlan())
        assert "--no-kv-offload" not in without, (
            "the default keeps the existing behaviour, whatever it is"
        )
        cmd = _build(sd, pm.PlacementPlan(no_kv_offload=True))
        assert "--no-kv-offload" in cmd, (
            "a plan that keeps the KV cache off the card says so in the command"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# PL4 -- the context extension terms travel together
# ---------------------------------------------------------------------------
def test_pl4_the_context_extension_terms_travel_together():
    pm, sd, restore = _open()
    try:
        without = _build(sd, pm.PlacementPlan())
        assert "--rope-scaling" not in without, (
            "a plan that does not extend context emits no extension terms"
        )

        plan = pm.PlacementPlan(
            rope_scaling="yarn", rope_scale=4.0, yarn_orig_ctx=32768,
        )
        cmd = _build(sd, plan)
        assert "--rope-scaling" in cmd, "the extension method is named"
        assert cmd[cmd.index("--rope-scaling") + 1] == "yarn"
        assert "--rope-scale" in cmd, "its factor travels with it"
        assert cmd[cmd.index("--rope-scale") + 1] == "4.0"
        assert "--yarn-orig-ctx" in cmd, (
            "the original training context travels with it too: the factor is "
            "meaningless without the length it multiplies"
        )
        assert cmd[cmd.index("--yarn-orig-ctx") + 1] == "32768"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PL5 -- the three existing knobs gain a plan, not only a call site
# ---------------------------------------------------------------------------
def test_pl5_flash_attention_and_kv_types_come_from_the_plan():
    pm, sd, restore = _open()
    try:
        plan = pm.PlacementPlan(
            flash_attn=True, type_k="q8_0", type_v="q8_0",
        )
        cmd = _build(sd, plan)
        assert "--flash-attn" in cmd, "flash attention is expressible by plan"
        assert cmd[cmd.index("--cache-type-k") + 1] == "q8_0", (
            "the K cache type comes from the plan"
        )
        assert cmd[cmd.index("--cache-type-v") + 1] == "q8_0", (
            "the V cache type comes from the plan"
        )

        bare = _build(sd, pm.PlacementPlan())
        assert "--flash-attn" not in bare, "and is absent when unasked"
        assert "--cache-type-k" not in bare, "as are the cache types"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PL6 -- an unrecognised KV cache type is refused, not dropped
# ---------------------------------------------------------------------------
def test_pl6_an_unknown_kv_cache_type_is_refused():
    pm, sd, restore = _open()
    try:
        good = pm.PlacementPlan(type_k="q8_0")
        assert good.validate() == [], (
            "a recognised type validates cleanly, so the refusal below is "
            "about the type and not about validation being always-on"
        )

        bad = pm.PlacementPlan(type_k="q8_bogus")
        errors = bad.validate()
        assert errors, "an unrecognised cache type is an error"
        assert any("q8_bogus" in e for e in errors), (
            "the refusal names the value it refused"
        )

        raised = ""
        try:
            _build(sd, bad)
        except ValueError as exc:
            raised = str(exc)
        assert "q8_bogus" in raised, (
            "building a command from an invalid plan raises rather than "
            "quietly emitting the default cache type"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# PL7 -- the recipe is reproducible
# ---------------------------------------------------------------------------
def test_pl7_the_same_plan_answers_the_same_command():
    pm, sd, restore = _open()
    try:
        plan = pm.PlacementPlan(
            n_gpu_layers=99,
            override_tensor=r"\.ffn_.*_exps\.=CPU",
            flash_attn=True,
            type_k="q8_0",
            type_v="q8_0",
        )
        first = _build(sd, plan)
        second = _build(sd, plan)
        assert first == second, (
            "the builder is pure: the same plan answers the identical argv, "
            "which is what makes a recipe comparable across runs"
        )
        assert len(first) > 5, (
            "the command carries real content, so equality is not between two "
            "empty lists"
        )

        other = _build(sd, pm.PlacementPlan(n_gpu_layers=40))
        assert other != first, "a different plan answers a different command"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PL8 -- a per-model entry overrides the defaults
# ---------------------------------------------------------------------------
def test_pl8_a_named_model_overrides_the_defaults(tmp_path):
    pm, _sd, restore = _open()
    try:
        config = tmp_path / "placement.yaml"
        config.write_text(
            "defaults:\n"
            "  n_gpu_layers: 99\n"
            "  flash_attn: true\n"
            "  type_k: q8_0\n"
            "models:\n"
            "  big-mixture:\n"
            "    n_gpu_layers: 12\n"
            "    override_tensor: \"\\\\.ffn_.*_exps\\\\.=CPU\"\n",
            encoding="utf-8",
        )

        base = pm.load_placement_plan("unlisted-model", config_path=config)
        assert base.n_gpu_layers == 99, "an unlisted model takes the defaults"
        assert base.flash_attn is True, "including the flags"
        assert base.override_tensor is None, "and nothing it never declared"

        named = pm.load_placement_plan("big-mixture", config_path=config)
        assert named.n_gpu_layers == 12, "a named model overrides a default"
        assert named.override_tensor == r"\.ffn_.*_exps\.=CPU", (
            "and adds what only it declares"
        )
        assert named.flash_attn is True, (
            "a per-model entry refines the defaults rather than replacing "
            "them: a key it does not mention keeps the default value"
        )
        assert named.type_k == "q8_0", "likewise for a default it is silent on"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    import tempfile

    simple = [
        ("PL1 experts routed only when asked", test_pl1_experts_are_routed_off_the_card_only_when_asked),
        ("PL2 target layer count reaches the card", test_pl2_the_target_layer_count_reaches_the_card),
        ("PL3 kv offload refused only when asked", test_pl3_kv_offload_is_refused_only_when_asked),
        ("PL4 context extension terms travel together", test_pl4_the_context_extension_terms_travel_together),
        ("PL5 flash and kv types come from the plan", test_pl5_flash_attention_and_kv_types_come_from_the_plan),
        ("PL6 unknown kv cache type is refused", test_pl6_an_unknown_kv_cache_type_is_refused),
        ("PL7 same plan answers same command", test_pl7_the_same_plan_answers_the_same_command),
    ]
    passed = 0
    total = len(simple) + 1
    for label, fn in simple:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            test_pl8_a_named_model_overrides_the_defaults(Path(tmp))
            print("PASS  PL8 named model overrides the defaults")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print("FAIL  PL8 named model overrides the defaults")
            traceback.print_exc()
    print(f"\n{passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
