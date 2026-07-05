#!/usr/bin/env python3
"""Capability manifest -- per-request introspection of callable capabilities.

The manifest answers one question for every request: what can the target
model ACTUALLY call right now? It reads the tool registry (the single
source of truth for callable tools, builtins and sandbox swaps and any
plugin-registered definitions alike), overlays the security mode, the
web-search killswitch and the explicit web-search override, checks the
target model's tool-calling aptitude through the profile store, and
produces two coherent artifacts:

  * the TOOL SET handed to the executor -- always the full reachable set,
    so the model decides by itself instead of a keyword gate deciding for
    it; and
  * a concise PROMPT BLOCK telling the model what exists and when it is
    relevant -- one line per tool, capped by a measured token budget so
    small-context local models never pay an unbounded prompt tax.

Invariants:

  * While the isolated security mode is active, no network-flagged tool
    appears in either artifact. The verdict is derived per request from
    the mode probe plus the ``network`` flag carried by each registered
    definition -- never from a hand-maintained list -- and it outranks
    every override. An indeterminable mode fails closed.
  * Fidelity: a tool that is disabled, killswitched or otherwise
    unreachable is absent from both artifacts, with its reason recorded.
    No ghost tools.
  * A model with an explicit negative tool-calling verdict receives an
    empty tool set and an omitted prompt block; models without a verdict
    keep the historical fallback protocol, so nothing regresses.
  * The prompt block never exceeds its budget, measured with the same
    calibrated estimator the context budgets use; the cap degrades the
    prose, never the tool set.

The fine prompt wording lives in module constants so it can be tuned on
the host without touching any logic.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# New-module safety rule: any change this module drives through the
# system must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

# Default token budget for the prompt block, sized for small-context
# local models. Callers may pass a different budget per request.
MANIFEST_TOKEN_BUDGET = 350

# Exclusion reasons recorded in ``CapabilityManifest.excluded``.
REASON_NETWORK_MODE = "network_blocked_by_mode"
REASON_KILLSWITCH = "web_search_killswitch"
REASON_DISABLED = "disabled"
REASON_OVERRIDE = "override_off"
REASON_MODEL = "model_no_tool_calling"

# Prompt wording (host-tunable constants; logic never depends on the text).
MANIFEST_HEADER = (
    "Capabilities available this turn (call a tool only when it truly "
    "helps; otherwise answer directly):"
)
MANIFEST_SANDBOX_NOTE = (
    "File and code tools run inside a disposable sandbox with no host "
    "filesystem or network access; their outputs leave the sandbox only "
    "after explicit human approval."
)
MANIFEST_FLOOR_TEMPLATE = "{count} tools are callable this turn."

# Deterministic degradation ladder for the per-tool summary length
# (characters); the final rungs are the name-only tier and the count
# floor, applied in _render_block below.
_SUMMARY_TIERS = (110, 70, 40)

# The name of the web-search tool in the registry; used only to attach
# the killswitch and override verdicts to the right definition (the
# network invariant itself is flag-derived, not name-derived).
_WEB_SEARCH_TOOL = "web_search"


@dataclass(frozen=True)
class CapabilityManifest:
    """The two per-request artifacts plus their derivation record."""

    tools: tuple = ()
    prompt_block: str = ""
    prompt_tokens: int = 0
    token_budget: int = MANIFEST_TOKEN_BUDGET
    mode: str = "unknown"
    model: str = ""
    model_tool_capable: bool = True
    excluded: dict = field(default_factory=dict)

    @property
    def has_tools(self) -> bool:
        return bool(self.tools)


# ---------------------------------------------------------------------------
# Runtime probes (each defensive; failure directions documented)
# ---------------------------------------------------------------------------
def _resolve_mode() -> tuple[str, bool]:
    """Return ``(mode_label, network_allowed)``.

    Fail-closed: when the mode cannot be determined, network capability
    stays closed -- the isolation invariant is never opened by a probe
    failure.
    """
    try:
        from .security_mode import is_bulbe
        if is_bulbe():
            return "bulbe", False
        return "daily", True
    except Exception:
        return "unknown", False


def _web_search_killed() -> bool:
    """Whether the web-search killswitch is engaged.

    Fail-open on probe failure, mirroring the enforcement middleware: an
    absent killswitch module means the feature is not killed.
    """
    try:
        from .search_killswitch import search_killswitch
        return bool(search_killswitch.is_killed())
    except Exception:
        return False


def _model_tool_capable(model: str) -> bool:
    """Profile-backed tool-calling verdict for ``model``.

    An explicit ``tool_calling`` verdict on the profile wins in both
    directions. A profile listing the ``tool_use`` capability is capable.
    Otherwise -- including when no profile exists or the profile system is
    unavailable -- the model stays capable: the historical fallback
    protocol drives tools for models without native function calling, so
    the default preserves existing behavior. Only an explicit negative
    verdict degrades the manifest.
    """
    try:
        from .model_profiles import get_profile
        profile = get_profile(model)
        if profile is not None:
            verdict = getattr(profile, "tool_calling", None)
            if verdict is not None:
                return bool(verdict)
            if profile.has_capability("tool_use"):
                return True
    except Exception:
        pass
    return True


def model_tool_capable(model: str) -> bool:
    """Public tool-calling predicate for ``model``.

    The manifest's own verdict, exposed so callers other than the manifest
    builder (e.g. the router) can ask "can this model call tools?" without
    reaching into a private helper. The rule is unchanged: an explicit
    profile verdict wins in both directions, and a model with no profile --
    or no profile system -- stays capable, because the historical fallback
    protocol drives tools for models without native function calling. Only
    an explicit negative verdict returns False.
    """
    return _model_tool_capable(model)


def _measure_tokens(text: str, model: str) -> int:
    """Measure ``text`` with the house calibrated estimator.

    The same yardstick the context budgets use, so the manifest budget
    and the context budget speak the same unit. Conservative character
    fallback only when the estimator is unavailable.
    """
    if not text:
        return 0
    try:
        from .context_manager import estimate_tokens_calibrated
        return int(estimate_tokens_calibrated(text, model))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Prompt block rendering
# ---------------------------------------------------------------------------
def _tool_summary(tool, max_chars: int) -> str:
    """One-line summary: the first sentence of the description, capped."""
    description = " ".join(str(getattr(tool, "description", "") or "").split())
    first = description.split(". ", 1)[0].rstrip(".")
    if max_chars and len(first) > max_chars:
        cut = first[:max_chars].rsplit(" ", 1)[0]
        first = (cut or first[:max_chars]).rstrip(",;: ")
    return first


def _render_tier(tools, max_chars: int, sandbox_note: str) -> str:
    lines = [MANIFEST_HEADER]
    for tool in tools:
        if max_chars > 0:
            summary = _tool_summary(tool, max_chars)
            lines.append(f"- {tool.name}: {summary}" if summary
                         else f"- {tool.name}")
        else:
            lines.append(f"- {tool.name}")
    if sandbox_note and max_chars > 0:
        lines.append(sandbox_note)
    return "\n".join(lines)


def _render_block(tools, model: str, budget: int,
                  sandbox_note: str) -> tuple[str, int]:
    """Render the block under ``budget``, degrading deterministically.

    Ladder: full one-line summaries at decreasing lengths, then names
    only, then a count floor. The tool set is never affected -- only the
    prose degrades. Returns ``(block, measured_tokens)``.
    """
    if not tools:
        return "", 0
    for max_chars in _SUMMARY_TIERS:
        block = _render_tier(tools, max_chars, sandbox_note)
        tokens = _measure_tokens(block, model)
        if tokens <= budget:
            return block, tokens
    block = _render_tier(tools, 0, "")
    tokens = _measure_tokens(block, model)
    if tokens <= budget:
        return block, tokens
    block = MANIFEST_FLOOR_TEMPLATE.format(count=len(tools))
    tokens = _measure_tokens(block, model)
    if tokens <= budget:
        return block, tokens
    return "", 0


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------
def build_manifest(
    *,
    model: str,
    registry=None,
    web_search_override: bool | None = None,
    token_budget: int | None = None,
) -> CapabilityManifest:
    """Build the per-request capability manifest for ``model``.

    Message-independent by construction: awareness is unconditional and
    the model decides. Precedence of the exclusion verdicts, per tool:
    disabled, then the mode invariant (network flag), then the web-search
    killswitch, then a False override. A True override cannot resurrect
    an unreachable tool and never outranks the mode invariant.
    """
    if registry is None:
        try:
            from .tool_registry import tool_registry as registry
        except Exception:
            registry = None

    budget = MANIFEST_TOKEN_BUDGET if token_budget is None else int(token_budget)
    mode, network_allowed = _resolve_mode()
    capable = _model_tool_capable(model)

    try:
        all_tools = list(registry.list_all()) if registry is not None else []
    except Exception:
        all_tools = []

    if not capable:
        return CapabilityManifest(
            tools=(),
            prompt_block="",
            prompt_tokens=0,
            token_budget=budget,
            mode=mode,
            model=model,
            model_tool_capable=False,
            excluded={t.name: REASON_MODEL for t in all_tools},
        )

    killed = _web_search_killed()
    included = []
    excluded: dict = {}
    for tool in all_tools:
        if not getattr(tool, "enabled", True):
            excluded[tool.name] = REASON_DISABLED
            continue
        if getattr(tool, "network", False) and not network_allowed:
            excluded[tool.name] = REASON_NETWORK_MODE
            continue
        if tool.name == _WEB_SEARCH_TOOL and killed:
            excluded[tool.name] = REASON_KILLSWITCH
            continue
        if tool.name == _WEB_SEARCH_TOOL and web_search_override is False:
            excluded[tool.name] = REASON_OVERRIDE
            continue
        included.append(tool)

    sandbox_note = ""
    if getattr(registry, "sandbox_mode", False) or getattr(
        registry, "quick_sandbox_mode", False,
    ):
        sandbox_note = MANIFEST_SANDBOX_NOTE

    block, tokens = _render_block(included, model, budget, sandbox_note)
    manifest = CapabilityManifest(
        tools=tuple(included),
        prompt_block=block,
        prompt_tokens=tokens,
        token_budget=budget,
        mode=mode,
        model=model,
        model_tool_capable=True,
        excluded=excluded,
    )
    if excluded:
        logger.debug(
            "Capability manifest: %d tools announced, %d excluded (%s)",
            len(included), len(excluded),
            ", ".join(f"{k}={v}" for k, v in excluded.items()),
        )
    return manifest
