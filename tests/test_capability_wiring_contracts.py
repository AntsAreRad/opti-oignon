#!/usr/bin/env python3
"""Contracts for the capability-manifest wiring in the executors.

The manifest replaces the keyword gate on tool ACCESS: in auto mode the
model always receives the full reachable tool set and decides by itself,
while the remaining heuristics keep shaping the pipeline (think versus
direct, the dedicated web fast path) and the explicit overrides keep
their forcing semantics. These contracts pin the wiring:

  * Contract W1 -- a message with no tool keyword at all still arms the
    tools pipeline when capabilities are present; without capabilities the
    legacy selection is byte-for-byte preserved (direct).
  * Contract W2 -- preserved semantics: the web keyword fast path still
    selects the dedicated pipeline in auto; a True override still forces
    it unconditionally; the forced-off branch keeps tools armed.
  * Contract W3 -- the registry itself is mode-aware per call, both
    outcomes: a network-flagged definition disappears from
    list_available() and is_available() while the isolated mode is
    active, and reappears when it is not. Flag-derived, no hand list.
  * Contract W4 -- the native decision consumes the manifest: the
    function-call schemas sent to the runtime are built from the manifest
    tool set (not the raw registry) and the decision messages carry the
    manifest prompt block.
  * Contract W5 -- the format= fallback carries the same truth: the
    decision prompt contains the manifest block and only the manifest
    tools; the registry prompt renderer accepts an explicit tool list.
  * Contract W6 -- the think+tools second phase is armed by the manifest,
    not by keywords: with a populated manifest the tools phase runs even
    when the keyword heuristic says no; without a manifest the legacy
    keyword verdict is preserved.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling harness
contracts: canonical dotted names so package and relative imports resolve
against the loaded copies, with a minimal pydantic stand-in only when the
real package is absent.
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
# Minimal pydantic stand-in (only when the real package is absent)
# ---------------------------------------------------------------------------
def _pydantic_shim() -> types.ModuleType:
    mod = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for name in getattr(self.__class__, "__annotations__", {}):
                default = getattr(self.__class__, name, None)
                if isinstance(default, (list, dict)):
                    default = type(default)(default)
                setattr(self, name, default)
            for key, value in kwargs.items():
                setattr(self, key, value)

    mod.BaseModel = BaseModel
    return mod


# ---------------------------------------------------------------------------
# Isolated loading (sibling-harness idiom)
# ---------------------------------------------------------------------------
_AGENTIC_KEYS = (
    "opti_oignon", "opti_oignon.agentic_executor",
)


def _load_agentic():
    """Load the agentic executor alone; every sibling import fails soft."""
    saved = {k: sys.modules.get(k) for k in _AGENTIC_KEYS}
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.agentic_executor", _OO / "agentic_executor.py",
    )
    ae = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.agentic_executor"] = ae
    spec.loader.exec_module(ae)
    pkg.agentic_executor = ae

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return ae, restore


_LOOP_KEYS = (
    "pydantic", "ollama", "opti_oignon", "opti_oignon.tool_calling",
    "opti_oignon.tool_registry", "opti_oignon.structured_output",
    "opti_oignon.response_hygiene", "opti_oignon.tool_executor",
    "opti_oignon.security_mode", "opti_oignon.config",
)


def _load_tool_loop(*, bulbe=False):
    """Load the real tool loop chain with a controllable mode stand-in."""
    saved = {k: sys.modules.get(k) for k in _LOOP_KEYS}

    try:
        import pydantic  # noqa: F401
    except ImportError:
        sys.modules["pydantic"] = _pydantic_shim()

    ollama_stub = types.ModuleType("ollama")
    ollama_stub.chat = lambda **kw: None
    sys.modules["ollama"] = ollama_stub

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    sm = types.ModuleType("opti_oignon.security_mode")
    sm._bulbe = bulbe
    sm.is_bulbe = lambda: sm._bulbe
    sys.modules["opti_oignon.security_mode"] = sm
    pkg.security_mode = sm

    def _real(dotted, path):
        spec = importlib.util.spec_from_file_location(dotted, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    pkg.tool_calling = _real(
        "opti_oignon.tool_calling", _OO / "tool_calling.py",
    )
    tr = _real("opti_oignon.tool_registry", _OO / "tool_registry.py")
    pkg.tool_registry = tr

    so = types.ModuleType("opti_oignon.structured_output")
    so.StructuredOutputEngine = object
    so.ToolCallRequest = object
    so.structured_output_engine = None
    so.STRUCTURED_OUTPUT_AVAILABLE = False
    sys.modules["opti_oignon.structured_output"] = so
    pkg.structured_output = so

    pkg.response_hygiene = _real(
        "opti_oignon.response_hygiene", _OO / "response_hygiene.py",
    )
    te = _real("opti_oignon.tool_executor", _OO / "tool_executor.py")
    pkg.tool_executor = te

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return te, tr, sm, ollama_stub, restore


# ---------------------------------------------------------------------------
# Local material
# ---------------------------------------------------------------------------
KEYWORDLESS = "Tell me about the aqueducts of ancient Lyon."


def _mk_tool(tr, name, *, network=False):
    return tr.ToolDefinition(
        name=name,
        description=f"Does the {name} operation on request.",
        parameters={},
        handler=lambda **kw: "",
        requires=[],
        enabled=True,
        network=network,
    )


def _mk_manifest(tools, block):
    return types.SimpleNamespace(
        tools=tuple(tools), prompt_block=block, has_tools=bool(tools),
    )


# ---------------------------------------------------------------------------
# Contract W1 -- keywordless auto arms tools iff capabilities are present
# ---------------------------------------------------------------------------
def test_w1_auto_keywordless_arms_tools_only_with_capabilities():
    ae, restore = _load_agentic()
    try:
        classification = ae._quick_classify(KEYWORDLESS)
        assert not any([
            classification["needs_tools"], classification["needs_web"],
            classification["is_code"], classification["is_complex"],
            classification["needs_reasoning"],
        ]), f"fixture message unexpectedly matched a heuristic: {classification}"

        armed = ae._select_pipeline(
            classification=classification,
            think_override=None,
            web_search_override=None,
            tool_executor_available=True,
            verification_available=False,
            capabilities_armed=True,
        )
        assert armed == ae.PIPELINE_TOOLS, (
            f"capabilities present must arm the tools pipeline, got {armed}"
        )

        legacy = ae._select_pipeline(
            classification=classification,
            think_override=None,
            web_search_override=None,
            tool_executor_available=True,
            verification_available=False,
            capabilities_armed=False,
        )
        assert legacy == ae.PIPELINE_DIRECT, (
            f"without capabilities the legacy selection must hold, got {legacy}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract W2 -- preserved heuristics and override pipeline semantics
# ---------------------------------------------------------------------------
def test_w2_preserved_heuristics_and_override_semantics():
    ae, restore = _load_agentic()
    try:
        web_classification = ae._quick_classify("what happened today in Lyon")
        assert web_classification["needs_web"], "fixture must hit the web keywords"
        fast_path = ae._select_pipeline(
            classification=web_classification,
            think_override=None,
            web_search_override=None,
            tool_executor_available=True,
            verification_available=False,
            capabilities_armed=True,
        )
        assert fast_path == ae.PIPELINE_WEB_SEARCH, (
            f"the web keyword fast path must be preserved, got {fast_path}"
        )

        forced = ae._select_pipeline(
            classification=ae._quick_classify(KEYWORDLESS),
            think_override=None,
            web_search_override=True,
            tool_executor_available=True,
            verification_available=False,
            capabilities_armed=False,
        )
        assert forced == ae.PIPELINE_WEB_SEARCH, (
            f"a True override must force the web pipeline, got {forced}"
        )

        forced_off = ae._select_pipeline(
            classification=ae._quick_classify(KEYWORDLESS),
            think_override=False,
            web_search_override=False,
            tool_executor_available=True,
            verification_available=False,
            capabilities_armed=True,
        )
        assert forced_off == ae.PIPELINE_TOOLS, (
            f"forced-off must keep armed tools available, got {forced_off}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract W3 -- the registry is mode-aware per call, both outcomes
# ---------------------------------------------------------------------------
def test_w3_registry_dynamic_network_gate_split():
    te, tr, sm, _ollama, restore = _load_tool_loop(bulbe=True)
    try:
        reg = tr.ToolRegistry()
        reg.register(_mk_tool(tr, "net_tool", network=True))
        reg.register(_mk_tool(tr, "local_tool"))

        names = [t.name for t in reg.list_available()]
        assert "net_tool" not in names, (
            f"network tool listed while the isolated mode is active: {names}"
        )
        assert "local_tool" in names
        assert reg.is_available("net_tool") is False
        assert reg.is_available("local_tool") is True

        sm._bulbe = False
        names2 = [t.name for t in reg.list_available()]
        assert "net_tool" in names2, (
            f"network tool must reappear outside the isolated mode: {names2}"
        )
        assert reg.is_available("net_tool") is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract W4 -- native decision consumes the manifest set and block
# ---------------------------------------------------------------------------
def test_w4_native_decision_uses_manifest_schemas_and_block():
    te, tr, _sm, ollama_stub, restore = _load_tool_loop(bulbe=False)
    try:
        reg = tr.ToolRegistry()
        web = _mk_tool(tr, "web_search", network=True)
        code = _mk_tool(tr, "execute_code")
        helper = _mk_tool(tr, "helper_local")
        for tool in (web, code, helper):
            reg.register(tool)

        manifest = _mk_manifest(
            [web, code], "CAPABILITY BLOCK MARKER: web_search, execute_code",
        )

        captured = {}

        def _chat(**kwargs):
            captured.update(kwargs)
            return {"message": {"content": "", "tool_calls": []}}

        ollama_stub.chat = _chat
        te.model_supports_native_tools = lambda model, **kw: True
        te.parse_native_tool_calls = lambda resp: []

        executor = te.ToolExecutor(
            registry=reg, structured_engine=None,
            default_model=MODEL, tool_transcript="flat",
        )
        decisions = executor._decide_tools(
            KEYWORDLESS, MODEL, [], [],
            native_transcript=[{"role": "user", "content": KEYWORDLESS}],
            manifest=manifest,
        )
        assert decisions == []
        assert "tools" in captured, "the native call never went out"
        schema_names = sorted(
            s["function"]["name"] for s in captured["tools"]
        )
        assert schema_names == ["execute_code", "web_search"], (
            f"schemas must come from the manifest set, got {schema_names}"
        )
        joined = "\n".join(
            str(m.get("content", "")) for m in captured["messages"]
        )
        assert "CAPABILITY BLOCK MARKER" in joined, (
            "the manifest prompt block is missing from the decision messages"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract W5 -- the format= fallback carries the block and filtered tools
# ---------------------------------------------------------------------------
def test_w5_format_fallback_prompt_carries_block_and_filtered_tools():
    te, tr, _sm, _ollama, restore = _load_tool_loop(bulbe=False)
    try:
        reg = tr.ToolRegistry()
        web = _mk_tool(tr, "web_search", network=True)
        code = _mk_tool(tr, "execute_code")
        helper = _mk_tool(tr, "helper_local")
        for tool in (web, code, helper):
            reg.register(tool)

        # The registry prompt renderer accepts an explicit tool list.
        subset_prompt = reg.get_tools_prompt(tools=[code])
        assert "execute_code" in subset_prompt
        assert "web_search" not in subset_prompt
        assert "helper_local" not in subset_prompt

        manifest = _mk_manifest(
            [web, code], "CAPABILITY BLOCK MARKER: web_search, execute_code",
        )
        te.model_supports_native_tools = lambda model, **kw: False

        executor = te.ToolExecutor(
            registry=reg, structured_engine=None,
            default_model=MODEL, tool_transcript="flat",
        )
        captured = {}

        def _capture_ask(message, model, tools_prompt, previous_results,
                         context_messages):
            captured["tools_prompt"] = tools_prompt
            return None

        executor._ask_llm_for_tool = _capture_ask
        decisions = executor._decide_tools(
            KEYWORDLESS, MODEL, [], [], manifest=manifest,
        )
        assert decisions == []
        prompt = captured.get("tools_prompt", "")
        assert "CAPABILITY BLOCK MARKER" in prompt, (
            "the manifest block is missing from the format= decision prompt"
        )
        assert "web_search" in prompt and "execute_code" in prompt
        assert "helper_local" not in prompt, (
            "the format= prompt must be filtered to the manifest set"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract W6 -- think+tools phase two is armed by the manifest
# ---------------------------------------------------------------------------
def test_w6_think_tools_phase_two_armed_by_manifest():
    ae, restore = _load_agentic()
    try:
        populated = _mk_manifest([object()], "block")
        empty = _mk_manifest([], "")

        assert ae._phase2_tools_armed(True, False, populated) is True, (
            "a populated manifest must arm phase two despite the keywords"
        )
        assert ae._phase2_tools_armed(True, True, None) is True, (
            "the legacy keyword verdict must be preserved without a manifest"
        )
        assert ae._phase2_tools_armed(True, False, None) is False, (
            "no manifest and no keywords must stay unarmed (legacy)"
        )
        assert ae._phase2_tools_armed(True, False, empty) is False, (
            "an empty manifest must not arm phase two"
        )
        assert ae._phase2_tools_armed(False, True, populated) is False, (
            "an unavailable tool executor can never be armed"
        )
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
