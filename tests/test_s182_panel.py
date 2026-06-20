#!/usr/bin/env python3
"""Tests for S182 Goal 2 -- the sharing-control panel (frontend, Theme 4 / Veilid Sync).

node_modules is absent in the sandbox, so the panel is checked by file content and
a structural tag-balance pass (Playwright is not feasible here). The panel must
build on the ds primitives, consume the sync API client (status and peers, pairing
generate / accept, relabel / unpair / run), surface the Bulbe refusal honestly
(disabled run and generate, pairing management still available), offer the pairing
ceremony (show this device's code, scan or paste a peer's), use an aria-live region,
use only --oo-* tokens (no raw hex), use lucide icons through Icon, and be
registered in FRONTEND_REDESIGN_SPEC.md so the Theme 1 cartography invariant stays
green.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "SyncPanel.svelte"
CLIENT = ROOT / "frontend" / "src" / "lib" / "api" / "sync.ts"
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

    def test_imports_sync_client(self):
        assert "from '$lib/api/sync'" in _panel()


# Status surface consumption


class TestStatusSurface:
    def test_consumes_status(self):
        text = _panel()
        assert "getSyncStatus(" in text

    def test_consumes_peers(self):
        text = _panel()
        assert "listSyncPeers(" in text

    def test_shows_running_state(self):
        text = _panel()
        assert "running" in text

    def test_shows_per_peer_last_sync(self):
        text = _panel()
        assert "last_sync" in text or "Last sync" in text

    def test_shows_watermark(self):
        text = _panel()
        assert "watermark" in text


# The Bulbe refusal is surfaced honestly


class TestBulbeHonesty:
    def test_reads_bulbe_flag(self):
        text = _panel()
        assert "bulbe_disabled" in text or "bulbeDisabled" in text

    def test_mentions_bulbe_disabled(self):
        text = _panel()
        assert "Bulbe" in text

    def test_run_disabled_under_bulbe(self):
        text = _panel()
        # The sync-now action's disabled state references the Bulbe flag.
        assert re.search(r"disabled=\{[^}]*bulbeDisabled", text) is not None

    def test_veilid_availability_surfaced(self):
        text = _panel()
        assert "veilid_available" in text or "veilidAvailable" in text


# The pairing ceremony


class TestPairing:
    def test_generate_self_payload(self):
        text = _panel()
        assert "getPairingSelf(" in text

    def test_accept_peer_payload(self):
        text = _panel()
        assert "acceptPairing(" in text

    def test_pairing_code_affordance(self):
        text = _panel()
        # The pairing code (what a QR encodes) is shown and a paste affordance exists.
        assert "qr-code" in text
        assert "textarea" in text

    def test_pairing_local_in_any_mode(self):
        text = _panel()
        # The accept path is not gated on the Bulbe flag (pairing is local).
        assert "Pair device" in text


# Peer management actions


class TestPeerActions:
    def test_run_action(self):
        text = _panel()
        assert "runSync(" in text and "syncNow(" in text

    def test_relabel_action(self):
        text = _panel()
        assert "relabelPeer(" in text

    def test_unpair_action(self):
        text = _panel()
        assert "unpairPeer(" in text

    def test_actions_disabled_while_busy(self):
        text = _panel()
        assert "busyPeer" in text


# Live regions


class TestLiveRegions:
    def test_aria_live_present(self):
        assert 'aria-live="polite"' in _panel()

    def test_peers_is_status_region(self):
        text = _panel()
        assert re.search(r'class="sync-peers"[^>]*role="status"', text) is not None


# Tokens and icons


class TestTokensAndIcons:
    def test_uses_oo_tokens(self):
        assert "var(--oo-" in _panel()

    def test_no_raw_hex_colors(self):
        text = _panel()
        hexes = re.findall(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b", text)
        assert hexes == [], f"raw hex colors present: {hexes}"

    def test_lucide_icons_via_icon(self):
        assert re.search(r"<Icon\s+name=", _panel()) is not None

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
        for tag in ("script", "style", "section", "header", "Card", "label", "textarea"):
            o, c = self._count(text, tag)
            assert o == c, f"<{tag}> unbalanced: open(non-self)={o} close={c}"


# Cartography registration


class TestCartography:
    def test_registered_in_frontend_spec(self):
        assert "SyncPanel" in SPEC.read_text(encoding="utf-8")

    def test_marked_new_for_s182(self):
        text = SPEC.read_text(encoding="utf-8")
        assert re.search(r"SyncPanel\.svelte`?\s*\|\s*NEW\s*\|\s*S182", text) is not None


# The sync API client (file content)


class TestClient:
    def test_uses_api_client(self):
        text = _client()
        assert "from './client'" in text
        for fn in ("apiGet", "apiPost", "apiDelete"):
            assert fn in text

    def test_status_interface(self):
        text = _client()
        assert "export interface SyncStatus" in text
        for field in ("running", "bulbe_disabled", "veilid_available", "peers"):
            assert field in text

    def test_peer_interface(self):
        text = _client()
        assert "export interface SyncPeer" in text
        for field in ("peer_id", "routing_key", "label", "watermark"):
            assert field in text

    def test_pairing_self_interface(self):
        text = _client()
        assert "export interface PairingSelf" in text
        assert "text" in text

    def test_status_function(self):
        assert "export async function getSyncStatus" in _client()

    def test_list_peers_function(self):
        assert "export async function listSyncPeers" in _client()

    def test_pairing_functions(self):
        text = _client()
        assert "export async function getPairingSelf" in text
        assert "export async function acceptPairing" in text

    def test_management_functions(self):
        text = _client()
        assert "export async function relabelPeer" in text
        assert "export async function unpairPeer" in text
        assert "export async function runSync" in text

    def test_documents_bulbe_and_kerckhoffs(self):
        text = _client().lower()
        assert "bulbe" in text
        assert "kerckhoffs" in text or "no secret" in text
