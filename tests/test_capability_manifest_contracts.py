#!/usr/bin/env python3
"""Contracts for the capability manifest.

The manifest is the per-request introspection of what the model can
actually call: it reads the tool registry (the single source of truth),
overlays the security mode, the web-search killswitch and the explicit
override, checks the target model's tool-calling aptitude, and produces
two coherent artifacts -- the tool set handed to the executor and a
concise, token-capped prompt block. These contracts pin the properties
that make it trustworthy:

  * Contract 1 -- in Daily mode the manifest announces web search in BOTH
    artifacts without looking at the message at all: the tool set contains
    web_search and the prompt block names it. Awareness is unconditional;
    the model decides.
  * Contract 2 -- the isolation invariant, both outcomes: with the
    isolated mode active, no network-flagged tool appears in the tool set
    NOR in the prompt block (local tools still do); with the mode off, the
    same registry announces them. Derived from the mode plus the flags on
    the registered definitions, never from a hand-maintained list.
  * Contract 3 -- fidelity: a disabled tool and a killswitched web search
    are absent from both artifacts, each with its recorded reason. No
    ghost tools.
  * Contract 4 -- a model with an explicit negative tool-calling verdict
    gets an empty tool set and an omitted prompt block; a profile listing
    the tool_use capability, or no profile at all, stays capable (the
    historical fallback protocol is preserved).
  * Contract 5 -- the prompt block cap is measured with the house
    estimator, not assumed: the recorded token count equals an independent
    re-measure and never exceeds the budget, while the TOOL SET stays
    complete (the cap degrades prose, never access).
  * Contract 6 -- overrides: False forces absence from both artifacts even
    in Daily; True keeps presence in Daily; the isolation invariant
    outranks True.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling harness
contracts: canonical dotted names so package and relative imports resolve
against the loaded copies, with controllable stand-ins for the mode, the
killswitch, the profile store and the token estimator.
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


# ---------------------------------------------------------------------------
# Isolated loading (sibling-harness idiom)
# ---------------------------------------------------------------------------
def _load(*, bulbe=False, killed=False, profile=None, estimator=None):
    """Load the real registry and manifest under controllable stand-ins.

    Returns (manifest_module, registry_module, mode_stub, killswitch_stub,
    restore). The estimator stand-in defaults to a whitespace word count so
    the measured-cap contract has a deterministic, independent yardstick.
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
        def __init__(self, engaged):
            self._engaged = engaged

        def is_killed(self):
            return self._engaged

    ks.search_killswitch = _Killswitch(killed)
    sys.modules["opti_oignon.search_killswitch"] = ks
    pkg.search_killswitch = ks

    mp = types.ModuleType("opti_oignon.model_profiles")
    mp._profile = profile
    mp.get_profile = lambda name: mp._profile
    sys.modules["opti_oignon.model_profiles"] = mp
    pkg.model_profiles = mp

    cm = types.ModuleType("opti_oignon.context_manager")
    cm.estimate_tokens_calibrated = estimator or (
        lambda text, model: len(text.split())
    )
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

    return cmf, tr, sm, ks, restore


# ---------------------------------------------------------------------------
# Local material
# ---------------------------------------------------------------------------
def _mk_tool(tr, name, *, network=False, enabled=True, description=None):
    return tr.ToolDefinition(
        name=name,
        description=description or f"Does the {name} operation on request.",
        parameters={},
        handler=lambda **kw: "",
        requires=[],
        enabled=enabled,
        network=network,
    )


def _mk_registry(tr, tools):
    reg = tr.ToolRegistry()
    for tool in tools:
        reg.register(tool)
    return reg


MODEL = "qwen3:8b"


class _Profile:
    """Minimal profile stand-in mirroring the fields the manifest reads."""

    def __init__(self, tool_calling=None, capabilities=None):
        self.tool_calling = tool_calling
        self.capabilities = list(capabilities or [])

    def has_capability(self, capability):
        return capability in self.capabilities


# ---------------------------------------------------------------------------
# Contract 1 -- Daily announces web search in both artifacts, unconditionally
# ---------------------------------------------------------------------------
def test_c1_daily_manifest_announces_web_search_in_both_artifacts():
    cmf, tr, _sm, _ks, restore = _load(bulbe=False)
    try:
        reg = _mk_registry(tr, [
            _mk_tool(tr, "web_search", network=True),
            _mk_tool(tr, "execute_code"),
        ])
        manifest = cmf.build_manifest(model=MODEL, registry=reg)
        names = [t.name for t in manifest.tools]
        assert "web_search" in names, f"web_search missing from tool set: {names}"
        assert "execute_code" in names, f"local tool missing: {names}"
        assert manifest.has_tools, "has_tools should be True"
        assert manifest.prompt_block, "prompt block should not be empty"
        assert "web_search" in manifest.prompt_block, (
            "prompt block does not name web_search"
        )
        # The manifest is message-independent by construction: build_manifest
        # accepts no message/keyword input at all.
        import inspect
        params = inspect.signature(cmf.build_manifest).parameters
        assert "message" not in params, (
            "build_manifest must not condition on the message"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- isolation invariant, both outcomes, flag-derived
# ---------------------------------------------------------------------------
def test_c2_isolated_mode_never_announces_network_split():
    cmf, tr, sm, _ks, restore = _load(bulbe=True)
    try:
        reg = _mk_registry(tr, [
            _mk_tool(tr, "web_search", network=True),
            _mk_tool(tr, "fetch_remote", network=True),
            _mk_tool(tr, "execute_code"),
        ])
        manifest = cmf.build_manifest(model=MODEL, registry=reg)
        names = [t.name for t in manifest.tools]
        assert "web_search" not in names, f"network tool leaked into set: {names}"
        assert "fetch_remote" not in names, f"network tool leaked into set: {names}"
        assert "execute_code" in names, f"local tool wrongly dropped: {names}"
        assert "web_search" not in manifest.prompt_block, (
            "network tool leaked into the prompt block"
        )
        assert "fetch_remote" not in manifest.prompt_block, (
            "network tool leaked into the prompt block"
        )
        assert "execute_code" in manifest.prompt_block, (
            "local tool missing from the prompt block"
        )
        assert manifest.excluded.get("web_search") == cmf.REASON_NETWORK_MODE
        assert manifest.excluded.get("fetch_remote") == cmf.REASON_NETWORK_MODE

        # Other outcome: same registry, mode off -- both network tools appear.
        sm._bulbe = False
        manifest2 = cmf.build_manifest(model=MODEL, registry=reg)
        names2 = [t.name for t in manifest2.tools]
        assert "web_search" in names2 and "fetch_remote" in names2, (
            f"network tools should be announced outside the isolated mode: {names2}"
        )
        assert "web_search" in manifest2.prompt_block
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- fidelity: disabled or killswitched -> absent, with reasons
# ---------------------------------------------------------------------------
def test_c3_unreachable_tools_are_absent_with_reasons():
    cmf, tr, _sm, ks, restore = _load(bulbe=False, killed=True)
    try:
        reg = _mk_registry(tr, [
            _mk_tool(tr, "web_search", network=True),
            _mk_tool(tr, "broken_tool", enabled=False),
            _mk_tool(tr, "execute_code"),
        ])
        manifest = cmf.build_manifest(model=MODEL, registry=reg)
        names = [t.name for t in manifest.tools]
        assert "web_search" not in names, (
            "killswitched web search must not appear in the tool set"
        )
        assert "web_search" not in manifest.prompt_block, (
            "killswitched web search must not appear in the prompt block"
        )
        assert manifest.excluded.get("web_search") == cmf.REASON_KILLSWITCH
        assert "broken_tool" not in names, "disabled tool leaked into the set"
        assert "broken_tool" not in manifest.prompt_block
        assert manifest.excluded.get("broken_tool") == cmf.REASON_DISABLED
        assert "execute_code" in names

        # Disengage the killswitch: web search comes back (per-request truth).
        ks.search_killswitch._engaged = False
        manifest2 = cmf.build_manifest(model=MODEL, registry=reg)
        assert "web_search" in [t.name for t in manifest2.tools]
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4 -- explicit negative verdict degrades; capable paths preserved
# ---------------------------------------------------------------------------
def test_c4_model_without_tool_calling_degrades_cleanly():
    cmf, tr, _sm, _ks, mp_restore = _load(
        bulbe=False, profile=_Profile(tool_calling=False),
    )
    try:
        reg = _mk_registry(tr, [
            _mk_tool(tr, "web_search", network=True),
            _mk_tool(tr, "execute_code"),
        ])
        manifest = cmf.build_manifest(model=MODEL, registry=reg)
        assert manifest.tools == (), (
            f"incapable model must get no tool set: {manifest.tools}"
        )
        assert not manifest.has_tools
        assert manifest.prompt_block == "", (
            "incapable model must get an omitted prompt block, no ghost tools"
        )
        assert manifest.model_tool_capable is False
        assert manifest.excluded.get("web_search") == cmf.REASON_MODEL
        assert manifest.excluded.get("execute_code") == cmf.REASON_MODEL
    finally:
        mp_restore()

    # Positive halves: an explicit True verdict, a tool_use capability, and
    # the no-profile default all stay capable.
    for profile in (
        _Profile(tool_calling=True),
        _Profile(capabilities=["tool_use"]),
        None,
    ):
        cmf, tr, _sm, _ks, restore = _load(bulbe=False, profile=profile)
        try:
            reg = _mk_registry(tr, [_mk_tool(tr, "execute_code")])
            manifest = cmf.build_manifest(model=MODEL, registry=reg)
            assert manifest.model_tool_capable is True, f"profile={profile}"
            assert manifest.has_tools, f"profile={profile}"
        finally:
            restore()


# ---------------------------------------------------------------------------
# Contract 5 -- the cap is measured with the house estimator and enforced
# ---------------------------------------------------------------------------
def test_c5_prompt_block_cap_is_measured_and_enforced():
    long_desc = (
        "Performs a thorough multi step operation over the working data "
        "and reports every intermediate detail it produced along the way "
        "so the caller can audit the whole run afterwards without guessing."
    )
    cmf, tr, _sm, _ks, restore = _load(bulbe=False)
    try:
        tools = [
            _mk_tool(tr, f"tool_number_{i:02d}", description=long_desc)
            for i in range(12)
        ]
        reg = _mk_registry(tr, tools)

        # Shrink path: the block degrades its prose under the budget while
        # the tool set stays complete.
        manifest = cmf.build_manifest(model=MODEL, registry=reg, token_budget=60)
        measured = len(manifest.prompt_block.split())
        assert manifest.prompt_tokens == measured, (
            f"recorded {manifest.prompt_tokens} != independent re-measure "
            f"{measured}: the count must come from the estimator"
        )
        assert manifest.prompt_tokens <= 60, (
            f"budget exceeded: {manifest.prompt_tokens} > 60"
        )
        assert len(manifest.tools) == 12, (
            "the cap must degrade prose, never the tool set"
        )

        # Floor path: a brutal budget still holds the invariant and the block
        # says how many tools exist instead of listing them.
        manifest2 = cmf.build_manifest(model=MODEL, registry=reg, token_budget=12)
        assert manifest2.prompt_tokens <= 12
        assert manifest2.prompt_tokens == len(manifest2.prompt_block.split())
        assert "12" in manifest2.prompt_block, (
            "the floor block should state the tool count"
        )
        assert len(manifest2.tools) == 12

        # Default budget: recorded and respected.
        manifest3 = cmf.build_manifest(model=MODEL, registry=reg)
        assert manifest3.token_budget == cmf.MANIFEST_TOKEN_BUDGET
        assert manifest3.prompt_tokens <= manifest3.token_budget
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 6 -- overrides force both directions; the invariant outranks True
# ---------------------------------------------------------------------------
def test_c6_override_semantics_and_precedence():
    cmf, tr, sm, _ks, restore = _load(bulbe=False)
    try:
        reg = _mk_registry(tr, [
            _mk_tool(tr, "web_search", network=True),
            _mk_tool(tr, "execute_code"),
        ])

        # False forces absence from BOTH artifacts, even in Daily.
        m_off = cmf.build_manifest(
            model=MODEL, registry=reg, web_search_override=False,
        )
        assert "web_search" not in [t.name for t in m_off.tools]
        assert "web_search" not in m_off.prompt_block
        assert m_off.excluded.get("web_search") == cmf.REASON_OVERRIDE
        assert "execute_code" in [t.name for t in m_off.tools]

        # True keeps presence in Daily.
        m_on = cmf.build_manifest(
            model=MODEL, registry=reg, web_search_override=True,
        )
        assert "web_search" in [t.name for t in m_on.tools]
        assert "web_search" in m_on.prompt_block

        # The isolation invariant outranks a True override.
        sm._bulbe = True
        m_forced = cmf.build_manifest(
            model=MODEL, registry=reg, web_search_override=True,
        )
        assert "web_search" not in [t.name for t in m_forced.tools], (
            "the mode invariant must outrank the override"
        )
        assert "web_search" not in m_forced.prompt_block
        assert m_forced.excluded.get("web_search") == cmf.REASON_NETWORK_MODE
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner (pytest picks up the test_ functions; direct execution works too)
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    names = sorted(n for n in globals() if n.startswith("test_"))
    selected = [
        n for n in names if not argv or any(fragment in n for fragment in argv)
    ]
    failures = 0
    for name in selected:
        try:
            globals()[name]()
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"{len(selected) - failures}/{len(selected)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
