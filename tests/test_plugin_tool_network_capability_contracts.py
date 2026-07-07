#!/usr/bin/env python3
"""Contracts for the plugin-provided tool network capability.

A tool a plugin contributes to the registry must carry a network flag that
reflects what the plugin actually declared in its validated manifest, not a
value the plugin picked for its own tool. The registry exposes a single
trusted entry point for this, and the mode-aware capability manifest -- the
one source of truth for what a model may call this turn -- must then mask or
expose the tool accordingly while the isolated mode is active.

These contracts pin the properties that make that trustworthy:

  * R1 -- a plugin WITHOUT the outbound-network permission that ships a tool
    claiming to be network-bound is registered as NON-network: the plugin's
    own flag is discarded (it is untrusted). The isolated mode does not mask
    it (nothing to mask -- its network imports are blocked anyway).
  * R2 -- a plugin WITH the outbound-network permission that ships a tool
    claiming NOT to be network-bound is registered as network: the isolated
    mode masks it. Without this the tool would leak in the isolated mode.
  * R3 -- fail-secure: when the permission set cannot be established (None),
    the tool is treated as network-bound and masked in the isolated mode; a
    known EMPTY permission set is a definite "no network" and stays exposed.
  * R4 -- the derivation touches only the network flag: name, description,
    parameters, handler and enabled state are preserved, and the registry's
    own per-call availability view honours the derived flag.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling harness
idiom: canonical dotted names so package and relative imports resolve
against the loaded copies, with a controllable stand-in for the mode (and,
for the end-to-end checks, the killswitch, profile store and estimator the
manifest reads).
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

MODEL = "qwen3:8b"


# ---------------------------------------------------------------------------
# Isolated loading (sibling-harness idiom)
# ---------------------------------------------------------------------------
def _load(*, bulbe=False):
    """Load the real registry and manifest under controllable stand-ins.

    Returns (tool_registry_module, capability_manifest_module, mode_stub,
    restore). The mode stand-in drives the isolated-mode invariant; the
    killswitch, profile store and estimator stand-ins keep the manifest's
    other probes deterministic and out of the way.
    """
    keys = (
        "opti_oignon", "opti_oignon.tool_registry",
        "opti_oignon.security_mode", "opti_oignon.search_killswitch",
        "opti_oignon.model_profiles", "opti_oignon.context_manager",
        "opti_oignon.capability_manifest",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    sm = types.ModuleType("opti_oignon.security_mode")
    sm._bulbe = bulbe
    sm.is_bulbe = lambda: sm._bulbe
    sys.modules["opti_oignon.security_mode"] = sm
    pkg.security_mode = sm

    ks = types.ModuleType("opti_oignon.search_killswitch")

    class _Killswitch:
        def is_killed(self):
            return False

    ks.search_killswitch = _Killswitch()
    sys.modules["opti_oignon.search_killswitch"] = ks
    pkg.search_killswitch = ks

    mp = types.ModuleType("opti_oignon.model_profiles")
    mp.get_profile = lambda name: None
    sys.modules["opti_oignon.model_profiles"] = mp
    pkg.model_profiles = mp

    cm = types.ModuleType("opti_oignon.context_manager")
    cm.estimate_tokens_calibrated = lambda text, model: len(text.split())
    sys.modules["opti_oignon.context_manager"] = cm
    pkg.context_manager = cm

    def _real(dotted, path):
        spec = importlib.util.spec_from_file_location(dotted, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    tr = _real("opti_oignon.tool_registry", _OO / "tool_registry.py")
    pkg.tool_registry = tr
    cmf = _real(
        "opti_oignon.capability_manifest", _OO / "capability_manifest.py",
    )
    pkg.capability_manifest = cmf

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return tr, cmf, sm, restore


# ---------------------------------------------------------------------------
# Local material
# ---------------------------------------------------------------------------
def _mk_plugin_tool(tr, name, *, claims_network):
    """A tool as a plugin might ship it, carrying its own network claim."""
    return tr.ToolDefinition(
        name=name,
        description=f"Plugin tool {name} that does one specific thing.",
        parameters={
            "arg": tr.ToolParam(
                name="arg", type="string", description="An argument.",
                required=True,
            ),
        },
        handler=lambda **kw: "",
        requires=[],
        enabled=True,
        network=claims_network,
    )


def _excluded_for_network(cmf, manifest, name):
    """True when the manifest excluded ``name`` for the network invariant."""
    return manifest.excluded.get(name) == cmf.REASON_NETWORK_MODE


# ---------------------------------------------------------------------------
# R1 -- no permission + a network claim -> registered non-network, not masked
# ---------------------------------------------------------------------------
def test_r1_no_permission_discards_plugin_network_claim():
    tr, cmf, _sm, restore = _load(bulbe=True)
    try:
        reg = tr.ToolRegistry()
        registered = reg.register_plugin_tool(
            _mk_plugin_tool(tr, "widget", claims_network=True),
            plugin_permissions=["conversation_read"],
        )
        assert registered.network is False, (
            "a plugin without the outbound-network permission must not "
            "produce a network tool, even when its own spec claims network"
        )
        # Isolated mode is active: a non-network tool is NOT masked for the
        # network reason.
        manifest = cmf.build_manifest(model=MODEL, registry=reg)
        assert not _excluded_for_network(cmf, manifest, "widget"), (
            "a correctly non-network plugin tool must not be masked by the "
            "isolated-mode network invariant"
        )
        assert any(t.name == "widget" for t in manifest.tools), (
            "the non-network plugin tool should be callable in the isolated "
            "mode"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# R2 -- permission present, no claim -> registered network, masked in isolated
# ---------------------------------------------------------------------------
def test_r2_permission_forces_network_and_isolated_masks_it():
    tr, cmf, sm, restore = _load(bulbe=True)
    try:
        reg = tr.ToolRegistry()
        registered = reg.register_plugin_tool(
            _mk_plugin_tool(tr, "fetcher", claims_network=False),
            plugin_permissions=["network_outbound"],
        )
        assert registered.network is True, (
            "a plugin holding the outbound-network permission must produce a "
            "network tool, even when its own spec claims no network"
        )
        # Isolated mode active -> masked for the network reason.
        masked = cmf.build_manifest(model=MODEL, registry=reg)
        assert _excluded_for_network(cmf, masked, "fetcher"), (
            "a network plugin tool must be masked while the isolated mode is "
            "active, or it leaks"
        )
        assert not any(t.name == "fetcher" for t in masked.tools), (
            "a masked tool must be absent from the callable set"
        )
        # Same registry, mode off -> announced again.
        sm._bulbe = False
        shown = cmf.build_manifest(model=MODEL, registry=reg)
        assert any(t.name == "fetcher" for t in shown.tools), (
            "outside the isolated mode the same network tool is announced"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# R3 -- fail-secure on indeterminate permissions; empty set stays exposed
# ---------------------------------------------------------------------------
def test_r3_indeterminate_fails_secure_empty_stays_exposed():
    tr, cmf, _sm, restore = _load(bulbe=True)
    try:
        reg = tr.ToolRegistry()
        # Indeterminate: permission set could not be established.
        indeterminate = reg.register_plugin_tool(
            _mk_plugin_tool(tr, "unknown", claims_network=False),
            plugin_permissions=None,
        )
        assert indeterminate.network is True, (
            "an indeterminable permission set must fail secure to a network "
            "tool so the isolated mode masks it"
        )
        manifest = cmf.build_manifest(model=MODEL, registry=reg)
        assert _excluded_for_network(cmf, manifest, "unknown"), (
            "the fail-secure tool must be masked while the isolated mode is "
            "active"
        )
        # A known EMPTY permission set is a definite absence of network.
        empty = reg.register_plugin_tool(
            _mk_plugin_tool(tr, "plain", claims_network=True),
            plugin_permissions=[],
        )
        assert empty.network is False, (
            "a known empty permission set means no network permission; the "
            "tool stays non-network and exposed"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# R4 -- only the network flag is derived; other fields and the gate survive
# ---------------------------------------------------------------------------
def test_r4_preserves_other_fields_and_registry_gate():
    tr, cmf, sm, restore = _load(bulbe=True)
    try:
        reg = tr.ToolRegistry()
        original = _mk_plugin_tool(tr, "gadget", claims_network=False)
        registered = reg.register_plugin_tool(
            original, plugin_permissions=["network_outbound"],
        )
        assert registered.name == "gadget"
        assert registered.description == original.description
        assert set(registered.parameters) == {"arg"}
        assert registered.handler is original.handler
        assert registered.enabled is True, (
            "the derivation must not disable the tool"
        )
        assert reg.get("gadget") is registered, (
            "the tool must actually be in the registry after registration"
        )
        # The registry's own availability view honours the derived flag:
        # a network tool is unavailable while the isolated mode is active.
        assert reg.is_available("gadget") is False, (
            "a derived-network tool must be unavailable in the isolated mode"
        )
        assert "gadget" not in {t.name for t in reg.list_available()}, (
            "a derived-network tool must be absent from list_available in "
            "the isolated mode"
        )
        sm._bulbe = False
        assert reg.is_available("gadget") is True, (
            "outside the isolated mode the tool is available again"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("R1 no perm discards plugin claim", test_r1_no_permission_discards_plugin_network_claim),
        ("R2 perm forces network, isolated masks", test_r2_permission_forces_network_and_isolated_masks_it),
        ("R3 indeterminate fail-secure, empty exposed", test_r3_indeterminate_fails_secure_empty_stays_exposed),
        ("R4 preserves fields and gate", test_r4_preserves_other_fields_and_registry_gate),
    ]
    passed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
