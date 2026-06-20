#!/usr/bin/env python3
"""Tests for S177 -- the skills-manager panel (frontend, Theme 3 / Odysseus Core).

node_modules is absent in the sandbox, so the panel is checked by file content
and a structural tag-balance pass (Playwright is not feasible here). The panel
must build on the ds primitives, browse the SKILL.md registry over the skills
API client (published skills and the agent-proposed drafts), expand a skill to
read its procedure, surface the approval-gated write actions (approve-and-publish
a draft, delete) with drafts clearly marked as awaiting approval, use an
aria-live region, use only --oo-* tokens (no raw hex), use lucide icons through
Icon, and be registered in FRONTEND_REDESIGN_SPEC.md so the Theme 1 cartography
invariant stays green.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "SkillsPanel.svelte"
CLIENT = ROOT / "frontend" / "src" / "lib" / "api" / "skills.ts"
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

    def test_imports_skills_client(self):
        text = _panel()
        assert "from '$lib/api/skills'" in text


# Registry browsing


class TestRegistryBrowsing:
    def test_lists_skills(self):
        text = _panel()
        assert "listSkills(" in text

    def test_distinguishes_drafts_and_published(self):
        text = _panel()
        # Both states are represented in the UI.
        assert "draft" in text and "published" in text
        assert "isDraft(" in text

    def test_filter_states(self):
        text = _panel()
        for f in ("all", "published", "drafts"):
            assert f"'{f}'" in text

    def test_status_icons(self):
        text = _panel()
        assert "STATUS_ICON" in text


# Reading a skill body


class TestSkillView:
    def test_fetches_full_body(self):
        text = _panel()
        assert "getSkill(" in text

    def test_expand_handler(self):
        text = _panel()
        assert "toggleBody(" in text

    def test_renders_body(self):
        text = _panel()
        assert "skill-body" in text


# Approval-gated write actions surfaced


class TestApprovalGatedActions:
    def test_publish_action(self):
        text = _panel()
        assert "publishSkill(" in text
        assert "handlePublish" in text

    def test_publish_labelled_as_approval(self):
        text = _panel()
        # Publishing a draft is surfaced as the human approval step.
        assert "Approve" in text and "publish" in text

    def test_delete_action(self):
        text = _panel()
        assert "deleteSkill(" in text
        assert "handleDelete" in text
        assert "Delete" in text

    def test_drafts_marked_awaiting_approval(self):
        text = _panel()
        assert "awaiting approval" in text

    def test_actions_disabled_while_busy(self):
        text = _panel()
        assert "busyKey" in text


# Live regions


class TestLiveRegions:
    def test_aria_live_present(self):
        text = _panel()
        assert 'aria-live="polite"' in text

    def test_list_is_status_region(self):
        text = _panel()
        assert re.search(r'class="skills-list"[^>]*role="status"', text) is not None


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
        assert "SkillsPanel" in text

    def test_marked_new_for_s177(self):
        text = SPEC.read_text(encoding="utf-8")
        assert re.search(r"SkillsPanel\.svelte`?\s*\|\s*NEW\s*\|\s*S177", text) is not None


# The skills API client (file content)


class TestClient:
    def test_skill_interface(self):
        text = _client()
        assert "export interface Skill" in text
        for field in ("name", "category", "status", "version", "source"):
            assert field in text

    def test_status_type(self):
        text = _client()
        assert "export type SkillStatus" in text
        assert "'draft'" in text and "'published'" in text

    def test_list_function(self):
        text = _client()
        assert "export async function listSkills" in text

    def test_get_function(self):
        text = _client()
        assert "export async function getSkill" in text

    def test_publish_function(self):
        text = _client()
        assert "export async function publishSkill" in text

    def test_delete_function(self):
        text = _client()
        assert "export async function deleteSkill" in text

    def test_is_draft_helper(self):
        text = _client()
        assert "export function isDraft" in text

    def test_uses_api_client(self):
        text = _client()
        assert "from './client'" in text
        for fn in ("apiGet", "apiPost", "apiDelete"):
            assert fn in text
