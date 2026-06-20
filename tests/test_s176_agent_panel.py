#!/usr/bin/env python3
"""Tests for S176 -- the agent panel (frontend, Theme 3 / Odysseus Core).

node_modules is absent in the sandbox, so the panel is checked by file content
and a structural tag-balance pass (Playwright is not feasible here). The panel
must build on the ds primitives, consume the loop's AgentEvent stream over the
agent API client, surface the round / step display and a cancel control, wire
the Bulbe approval prompts to the existing tool-call approval API, use aria-live
regions, use only --oo-* tokens (no raw hex), and be registered in
FRONTEND_REDESIGN_SPEC.md so the Theme 1 cartography invariant stays green.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "AgentPanel.svelte"
CLIENT = ROOT / "frontend" / "src" / "lib" / "api" / "agent.ts"
SPEC = ROOT / "FRONTEND_REDESIGN_SPEC.md"


def _panel() -> str:
    return PANEL.read_text(encoding="utf-8")


def _client() -> str:
    return CLIENT.read_text(encoding="utf-8")


# Existence and primitives


class TestExistence:
    def test_panel_exists(self):
        assert PANEL.exists()

    def test_client_exists(self):
        assert CLIENT.exists()

    def test_imports_ds_primitives(self):
        text = _panel()
        assert "from '$lib/ds'" in text
        for prim in ("Card", "Button", "Icon", "EmptyState", "InlineError"):
            assert prim in text

    def test_imports_agent_client(self):
        text = _panel()
        assert "from '$lib/api/agent'" in text


# Event-stream consumption


class TestEventStreamConsumption:
    def test_consumes_agent_event_type(self):
        text = _panel()
        assert "AgentEvent" in text

    def test_subscribes_to_stream(self):
        text = _panel()
        assert "connectAgentStream(" in text

    def test_handles_event_kinds(self):
        text = _panel()
        # The panel reacts to the loop's event kinds.
        for kind in ("round_start", "model_output", "tool_result", "done", "error", "verifier_output"):
            assert kind in text

    def test_renders_tool_results(self):
        text = _panel()
        assert "tool_result" in text
        assert "AgentToolResult" in text or "toolResult(" in text

    def test_accumulates_events(self):
        text = _panel()
        assert "pushEvent" in text


# Approval prompts wired to the existing API


class TestApprovalPrompts:
    def test_uses_pending_approvals(self):
        text = _panel()
        assert "getPendingApprovals" in text

    def test_approve_and_deny_calls(self):
        text = _panel()
        assert "approveToolCall(" in text
        assert "denyToolCall(" in text

    def test_approve_deny_handlers(self):
        text = _panel()
        assert "handleApprove" in text
        assert "handleDeny" in text

    def test_approve_deny_buttons(self):
        text = _panel()
        assert "Approve" in text
        assert "Deny" in text


# Round / step display and cancel control


class TestRoundAndCancel:
    def test_round_display(self):
        text = _panel()
        assert "currentRound" in text
        assert "Round" in text

    def test_cancel_control(self):
        text = _panel()
        assert "handleCancel" in text
        assert "cancelAgentRun(" in text
        assert "Cancel" in text

    def test_status_role(self):
        text = _panel()
        assert 'role="status"' in text


# Live regions


class TestLiveRegions:
    def test_aria_live_present(self):
        text = _panel()
        assert 'aria-live="polite"' in text

    def test_stream_is_live_region(self):
        text = _panel()
        # The streaming container announces updates.
        assert re.search(r'class="agent-stream"[^>]*aria-live', text) is not None


# Tokens and icons


class TestTokensAndIcons:
    def test_uses_oo_tokens(self):
        assert "var(--oo-" in _panel()

    def test_no_raw_hex_colors(self):
        text = _panel()
        hexes = re.findall(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b", text)
        assert hexes == [], f"raw hex colors present: {hexes}"

    def test_lucide_icons_via_icon(self):
        text = _panel()
        assert re.search(r"<Icon\s+name=", text) is not None

    def test_no_emojis(self):
        text = _panel()
        for ch in text:
            assert ord(ch) < 0x1F000, f"emoji-range char present: {ch!r}"


# Tag balance


class TestTagBalance:
    def _count(self, src, tag):
        opens = len(re.findall(r"<%s(\s[^>]*?)?>" % tag, src))
        selfc = len(re.findall(r"<%s(\s[^>]*?)?/>" % tag, src))
        closes = len(re.findall(r"</%s>" % tag, src))
        return opens - selfc, closes

    def test_block_balance(self):
        text = _panel()
        assert len(re.findall(r"\{#if\b", text)) == len(re.findall(r"\{/if\}", text))
        assert len(re.findall(r"\{#each\b", text)) == len(re.findall(r"\{/each\}", text))

    def test_paired_tags_balanced(self):
        text = _panel()
        for tag in ("script", "style", "section", "header", "Card"):
            o, c = self._count(text, tag)
            assert o == c, f"<{tag}> unbalanced: open(non-self)={o} close={c}"


# Cartography registration


class TestCartography:
    def test_registered_in_frontend_spec(self):
        text = SPEC.read_text(encoding="utf-8")
        assert "AgentPanel" in text

    def test_marked_new_for_s176(self):
        text = SPEC.read_text(encoding="utf-8")
        assert re.search(r"AgentPanel\.svelte`?\s*\|\s*NEW\s*\|\s*S176", text) is not None


# The agent API client (file content)


class TestClient:
    def test_agent_event_interface(self):
        text = _client()
        assert "export interface AgentEvent" in text
        assert "AgentEventKind" in text

    def test_event_kinds_constant(self):
        text = _client()
        assert "AGENT_EVENT_KINDS" in text
        for kind in ("round_start", "model_output", "tool_result", "done", "error", "verifier_output"):
            assert f"'{kind}'" in text

    def test_tool_result_interface(self):
        text = _client()
        assert "export interface AgentToolResult" in text
        for field in ("tool_name", "executed", "observation"):
            assert field in text

    def test_stream_subscription(self):
        text = _client()
        assert "export function connectAgentStream" in text
        assert "ReconnectingWebSocket" in text
        assert "wsUrl(" in text

    def test_parse_event_helper(self):
        text = _client()
        assert "parseAgentEvent" in text

    def test_run_controls(self):
        text = _client()
        assert "cancelAgentRun" in text
        assert "getAgentStatus" in text

    def test_reuses_existing_approval_api(self):
        text = _client()
        assert "from './toolCallApproval'" in text
        for fn in ("getPendingApprovals", "approveToolCall", "denyToolCall"):
            assert fn in text
