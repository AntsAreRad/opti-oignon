#!/usr/bin/env python3
"""Tests for S174 -- the memories panel (frontend).

node_modules is absent in the sandbox, so the panel is checked by file content
and a structural tag-balance pass (Playwright is not feasible here). The panel
must build on the ds primitives, wire the memories API client, group by category
with soft-delete / restore / edit, use only --oo-* tokens (no raw hex), and be
registered in FRONTEND_REDESIGN_SPEC.md so the Theme 1 cartography invariant
stays green.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "MemoriesPanel.svelte"
LEGACY = ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "MemoryPanel.svelte"
CLIENT = ROOT / "frontend" / "src" / "lib" / "api" / "memories.ts"
SPEC = ROOT / "FRONTEND_REDESIGN_SPEC.md"


def _panel() -> str:
    return PANEL.read_text(encoding="utf-8")


# Existence and primitives


class TestExistence:
    def test_panel_exists(self):
        assert PANEL.exists()

    def test_legacy_panel_intact(self):
        # The new panel is distinct; the legacy panel is not removed.
        assert LEGACY.exists()

    def test_imports_ds_primitives(self):
        text = _panel()
        assert "from '$lib/ds'" in text
        for prim in ("Card", "Tabs", "Modal", "Select", "Icon"):
            assert prim in text

    def test_wires_memories_client(self):
        text = _panel()
        assert "from '$lib/api/memories'" in text
        for fn in ("listMemories", "editMemory", "softDeleteMemory", "restoreMemory"):
            assert fn in text


# Behaviour surface


class TestBehaviour:
    def test_grouped_by_category(self):
        text = _panel()
        assert "CATEGORY_ORDER" in text
        assert "grouped" in text

    def test_active_archived_tabs(self):
        text = _panel()
        assert "id: 'active'" in text
        assert "id: 'archived'" in text

    def test_soft_delete_action(self):
        text = _panel()
        assert "doSoftDelete" in text
        assert "softDeleteMemory(" in text

    def test_restore_action(self):
        text = _panel()
        assert "doRestore" in text
        assert "restoreMemory(" in text

    def test_edit_action(self):
        text = _panel()
        assert "openEdit" in text
        assert "saveEdit" in text
        assert "editMemory(" in text

    def test_modal_uses_onclose_callback(self):
        text = _panel()
        # Modal takes an onClose callback prop (not an event).
        assert "onClose={closeEdit}" in text

    def test_select_change_has_typeof_guard(self):
        text = _panel()
        assert "onCategoryChange" in text
        assert "typeof value === 'string'" in text


# Tokens and icons


class TestTokensAndIcons:
    def test_uses_oo_tokens(self):
        text = _panel()
        assert "var(--oo-" in text

    def test_no_raw_hex_colors(self):
        text = _panel()
        hexes = re.findall(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b", text)
        assert hexes == [], f"raw hex colors present: {hexes}"

    def test_lucide_icons_via_icon(self):
        text = _panel()
        assert re.search(r'<Icon\s+name="', text) is not None

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
        for tag in ("script", "style", "section", "Card", "Modal", "textarea", "svelte:fragment"):
            o, c = self._count(text, tag.replace(":", r"\:") if ":" in tag else tag)
            assert o == c, f"<{tag}> unbalanced: open(non-self)={o} close={c}"


# Cartography registration


class TestCartography:
    def test_registered_in_frontend_spec(self):
        text = SPEC.read_text(encoding="utf-8")
        assert "MemoriesPanel" in text

    def test_marked_new_for_s174(self):
        text = SPEC.read_text(encoding="utf-8")
        # The disposition table marks it NEW in S174.
        assert re.search(r"MemoriesPanel\.svelte`?\s*\|\s*NEW\s*\|\s*S174", text) is not None


# Client typing (file content)


class TestClient:
    def test_record_type(self):
        text = CLIENT.read_text(encoding="utf-8")
        assert "export interface MemoryRecord" in text

    def test_six_categories(self):
        text = CLIENT.read_text(encoding="utf-8")
        for cat in ("identity", "preference", "fact", "contact", "project", "goal"):
            assert cat in text
