#!/usr/bin/env python3
"""Tests for S177 -- the on-disk SKILL.md registry (Theme 3 / Odysseus Core).

Covers ODYSSEUS_SPEC.md Section 6.1 / Section 6.3 (the registry half):

- Path safety: category and name are sanitised to a strict slug and a traversal
  payload can never resolve outside the registry root.
- Frontmatter: round-trips through the YAML-style block; tolerant of missing /
  malformed frontmatter.
- CRUD: add (draft / published), get, edit, patch (unique match), delete.
- Drafts versus published: kept apart on disk so a draft never shadows a live
  skill; the index lists both, distinctly.
- Version retention: each edit / patch / re-publish archives the prior version
  under .versions for audit.
- The usage sidecar: increment bumps _usage.json only and never rewrites the
  SKILL.md; a no-op when the skill is absent.
- Search relevance: name / category weigh more than the body; stable ordering.
- Cartography: skills.py is registered in ODYSSEUS_SPEC.md Section 10.

Loaded in isolation via spec_from_file_location with opti_oignon stubbed, and
rooted at a temporary directory, so the runtime collects without the backend
and leaves nothing on disk.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
AGENT = OO / "agent"
SPEC = ROOT / "ODYSSEUS_SPEC.md"


def _ensure_pkg():
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(OO)]
        sys.modules["opti_oignon"] = pkg
    if "opti_oignon.agent" not in sys.modules:
        apkg = types.ModuleType("opti_oignon.agent")
        apkg.__path__ = [str(AGENT)]
        sys.modules["opti_oignon.agent"] = apkg


def _ensure_agent(name: str):
    full = f"opti_oignon.agent.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(AGENT / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_pkg()
sk = _ensure_agent("skills")


# A standard structured body for tests.
def _body(when="When deploying a service to the cluster.", proc="run the deploy script"):
    return (
        f"## When to Use\n{when}\n\n"
        f"## Procedure\n{proc}\n\n"
        "## Pitfalls\nDo not skip the health check.\n\n"
        "## Verification\nConfirm the service responds on its port.\n"
    )


@pytest.fixture
def reg(tmp_path):
    return sk.SkillRegistry(root=tmp_path)


@pytest.fixture(autouse=True)
def _reset():
    sk.reset_skill_registry()
    yield
    sk.reset_skill_registry()


# Module conventions


class TestConventions:
    def test_checkpoint_flag(self):
        assert sk.checkpoint_before_apply is True

    def test_feature_flag(self):
        assert sk.FEATURE_AVAILABLE is True

    def test_singleton_reset(self):
        a = sk.get_skill_registry()
        sk.reset_skill_registry()
        b = sk.get_skill_registry()
        assert a is not b

    def test_set_registry(self, tmp_path):
        custom = sk.SkillRegistry(root=tmp_path)
        sk.set_skill_registry(custom)
        assert sk.get_skill_registry() is custom

    def test_status_and_source_constants(self):
        assert sk.STATUS_DRAFT == "draft"
        assert sk.STATUS_PUBLISHED == "published"
        assert sk.SOURCE_TEACHER == "teacher-escalation"

    def test_body_sections_order(self):
        assert sk.BODY_SECTIONS == (
            "When to Use",
            "Procedure",
            "Pitfalls",
            "Verification",
        )


# Path safety


class TestPathSafety:
    def test_segment_sanitises_disallowed(self):
        assert sk._safe_segment("My Skill!", "x") == "my-skill"
        assert sk._safe_segment("../etc", "x") == "etc"
        assert sk._safe_segment("a/b/c", "x") == "a-b-c"

    def test_segment_fallback_on_empty(self):
        assert sk._safe_segment("", "fallback") == "fallback"
        assert sk._safe_segment("...", "fallback") == "fallback"

    def test_traversal_payload_stays_under_root(self, reg):
        path = reg._skill_path("../../etc", "passwd", draft=False)
        assert path is not None
        assert reg._within_root(path.parent)

    def test_absolute_payload_stays_under_root(self, reg):
        path = reg._skill_path("/abs/evil", "/x", draft=False)
        assert path is not None
        assert reg._within_root(path.parent)

    def test_within_root_rejects_outside(self, reg, tmp_path):
        outside = tmp_path.parent / "outside-of-root"
        assert reg._within_root(outside) is False

    def test_add_with_traversal_name_lands_in_registry(self, reg):
        skill = reg.add("../../../escape", "../cat", _body(), status="published")
        got = reg.get(skill.name, skill.category)
        assert got is not None
        assert reg._within_root(reg._skill_dir(skill.category, skill.name, draft=False))


# Frontmatter


class TestFrontmatter:
    def test_round_trip(self):
        meta = {
            "name": "deploy-service",
            "category": "coding",
            "status": "published",
            "version": 2,
            "source": "agent",
            "created_at": "2026-06-02T00:00:00+00:00",
            "updated_at": "2026-06-02T01:00:00+00:00",
        }
        text = sk._serialise_frontmatter(meta) + "\n\n## When to Use\nbody\n"
        parsed, body = sk._parse_frontmatter(text)
        assert parsed["name"] == "deploy-service"
        assert parsed["version"] == "2"
        assert "## When to Use" in body

    def test_no_frontmatter_is_all_body(self):
        parsed, body = sk._parse_frontmatter("## When to Use\nno frontmatter here")
        assert parsed == {}
        assert body.startswith("## When to Use")

    def test_malformed_frontmatter_tolerated(self):
        # An opening fence with no closing fence is treated as body, not a crash.
        parsed, body = sk._parse_frontmatter("---\nname: x\nno closing fence")
        assert parsed == {}
        assert "name: x" in body

    def test_extract_section(self):
        body = _body(when="Trigger phrase here.")
        assert "Trigger phrase here." in sk._extract_section(body, "When to Use")
        assert "deploy script" in sk._extract_section(body, "Procedure")
        assert sk._extract_section(body, "Nonexistent") == ""


# CRUD


class TestCrud:
    def test_add_draft_then_get(self, reg):
        skill = reg.add("Deploy", "coding", _body(), source="agent", status="draft")
        assert skill.status == "draft"
        assert reg.get("deploy", "coding", draft=True) is not None
        assert reg.get("deploy", "coding", draft=False) is None

    def test_add_published_version_one(self, reg):
        skill = reg.add("Deploy", "coding", _body(), status="published")
        assert skill.status == "published"
        assert skill.version == 1
        assert reg.get("deploy", "coding") is not None

    def test_readd_published_bumps_version(self, reg):
        reg.add("Deploy", "coding", _body(), status="published")
        again = reg.add("Deploy", "coding", _body(proc="updated"), status="published")
        assert again.version == 2

    def test_edit_archives_and_bumps(self, reg):
        reg.add("Deploy", "coding", _body(), status="published")
        edited = reg.update("deploy", "coding", body=_body(proc="v2 steps"))
        assert edited is not None
        assert edited.version == 2
        assert "v2 steps" in edited.body

    def test_edit_missing_returns_none(self, reg):
        assert reg.update("missing", "coding", body="x") is None

    def test_patch_unique_match(self, reg):
        reg.add("Deploy", "coding", _body(proc="run the deploy script"), status="published")
        patched = reg.patch("deploy", "coding", "run the deploy script", "run deploy.sh")
        assert patched is not None
        assert "run deploy.sh" in patched.body
        assert patched.version == 2

    def test_patch_rejects_non_unique(self, reg):
        reg.add("Deploy", "coding", "## Procedure\nx x x", status="published")
        assert reg.patch("deploy", "coding", "x", "y") is None

    def test_patch_rejects_absent(self, reg):
        assert reg.patch("deploy", "coding", "anything", "y") is None

    def test_delete_published(self, reg):
        reg.add("Deploy", "coding", _body(), status="published")
        assert reg.delete("deploy", "coding") is True
        assert reg.get("deploy", "coding") is None

    def test_delete_draft(self, reg):
        reg.add("Deploy", "coding", _body(), status="draft")
        assert reg.delete("deploy", "coding", draft=True) is True
        assert reg.get("deploy", "coding", draft=True) is None

    def test_delete_missing_returns_false(self, reg):
        assert reg.delete("deploy", "coding") is False

    def test_view_and_view_ref(self, reg):
        reg.add("Deploy", "coding", _body(when="Use when deploying."), status="published")
        full = reg.view("deploy", "coding")
        assert "## When to Use" in full and "Procedure" in full
        ref = reg.view_ref("deploy", "coding")
        assert "deploy" in ref and "Use when deploying." in ref
        assert "Procedure" not in ref  # reference is the trigger only

    def test_view_absent_is_empty(self, reg):
        assert reg.view("nope", "coding") == ""
        assert reg.view_ref("nope", "coding") == ""


# Drafts versus published


class TestDraftsVsPublished:
    def test_draft_does_not_shadow_published(self, reg):
        reg.add("Deploy", "coding", _body(proc="published proc"), status="published")
        reg.add("Deploy", "coding", _body(proc="draft proc"), status="draft")
        assert "published proc" in reg.get("deploy", "coding").body
        assert "draft proc" in reg.get("deploy", "coding", draft=True).body

    def test_draft_version_anticipates_next(self, reg):
        reg.add("Deploy", "coding", _body(), status="published")  # v1
        draft = reg.add("Deploy", "coding", _body(proc="proposed"), status="draft")
        assert draft.version == 2  # the version it would become

    def test_index_separates(self, reg):
        reg.add("Alpha", "coding", _body(), status="published")
        reg.add("Beta", "writing", _body(), status="draft")
        idx = reg.index()
        pub_names = {s["name"] for s in idx["published"]}
        draft_names = {s["name"] for s in idx["drafts"]}
        assert pub_names == {"alpha"}
        assert draft_names == {"beta"}

    def test_list_include_drafts(self, reg):
        reg.add("Alpha", "coding", _body(), status="published")
        reg.add("Beta", "coding", _body(), status="draft")
        assert len(reg.list(include_drafts=False)) == 1
        assert len(reg.list(include_drafts=True)) == 2

    def test_list_filter_by_category(self, reg):
        reg.add("Alpha", "coding", _body(), status="published")
        reg.add("Gamma", "writing", _body(), status="published")
        assert {s.name for s in reg.list(category="coding")} == {"alpha"}


# Version retention


class TestVersionRetention:
    def _versions(self, reg, tmp_path, category, name):
        d = tmp_path / category / name / sk.VERSIONS_DIR
        return sorted(p.name for p in d.glob("*.md")) if d.is_dir() else []

    def test_edit_archives_prior(self, reg, tmp_path):
        reg.add("Deploy", "coding", _body(), status="published")  # v1
        reg.update("deploy", "coding", body=_body(proc="v2"))  # archives v1
        assert "v1.md" in self._versions(reg, tmp_path, "coding", "deploy")

    def test_multiple_edits_retain_all(self, reg, tmp_path):
        reg.add("Deploy", "coding", _body(), status="published")
        reg.update("deploy", "coding", body=_body(proc="v2"))
        reg.update("deploy", "coding", body=_body(proc="v3"))
        versions = self._versions(reg, tmp_path, "coding", "deploy")
        assert "v1.md" in versions and "v2.md" in versions

    def test_publish_over_existing_archives(self, reg, tmp_path):
        reg.add("Deploy", "coding", _body(), status="published")  # v1
        reg.add("Deploy", "coding", _body(proc="draft"), status="draft")
        reg.publish("deploy", "coding")  # archives the v1 published
        assert "v1.md" in self._versions(reg, tmp_path, "coding", "deploy")

    def test_delete_archives_final_version(self, reg, tmp_path):
        reg.add("Deploy", "coding", _body(), status="published")
        reg.delete("deploy", "coding")
        assert "v1.md" in self._versions(reg, tmp_path, "coding", "deploy")


# Publish path (registry-level; the gated tool path is tested separately)


class TestPublish:
    def test_publish_promotes_draft(self, reg):
        reg.add("Deploy", "coding", _body(), status="draft")
        published = reg.publish("deploy", "coding")
        assert published is not None
        assert published.status == "published"
        assert reg.get("deploy", "coding") is not None

    def test_publish_consumes_draft(self, reg):
        reg.add("Deploy", "coding", _body(), status="draft")
        reg.publish("deploy", "coding")
        assert reg.get("deploy", "coding", draft=True) is None

    def test_publish_missing_draft_returns_none(self, reg):
        assert reg.publish("deploy", "coding") is None

    def test_publish_preserves_source(self, reg):
        reg.add("Deploy", "coding", _body(), source="teacher-escalation", status="draft")
        published = reg.publish("deploy", "coding")
        assert published.source == "teacher-escalation"


# Usage sidecar


class TestUsageSidecar:
    def test_increment_bumps_only_sidecar(self, reg, tmp_path):
        reg.add("Deploy", "coding", _body(), status="published")
        skill_file = tmp_path / "coding" / "deploy" / sk.SKILL_FILENAME
        before = skill_file.read_text(encoding="utf-8")
        usage = reg.increment_usage("deploy", "coding")
        after = skill_file.read_text(encoding="utf-8")
        assert usage.uses == 1
        assert before == after  # SKILL.md never rewritten for a count bump
        assert (tmp_path / "coding" / "deploy" / sk.USAGE_FILENAME).is_file()

    def test_increment_accumulates(self, reg):
        reg.add("Deploy", "coding", _body(), status="published")
        reg.increment_usage("deploy", "coding")
        usage = reg.increment_usage("deploy", "coding")
        assert usage.uses == 2
        assert usage.last_used

    def test_increment_absent_is_noop(self, reg):
        usage = reg.increment_usage("nope", "coding")
        assert usage.uses == 0

    def test_get_usage_default(self, reg):
        reg.add("Deploy", "coding", _body(), status="published")
        assert reg.get_usage("deploy", "coding").uses == 0


# Search relevance


class TestSearch:
    def test_name_match_outranks_body(self, reg):
        reg.add("Deploy Service", "coding", _body(proc="kubernetes rollout"), status="published")
        reg.add("Write Tests", "coding", "## Procedure\ndeploy mentioned in body only", status="published")
        results = reg.search("deploy")
        assert results
        assert results[0].skill.name == "deploy-service"

    def test_category_contributes(self, reg):
        reg.add("Alpha", "kubernetes", _body(), status="published")
        results = reg.search("kubernetes")
        assert results and results[0].skill.name == "alpha"

    def test_no_overlap_dropped(self, reg):
        reg.add("Deploy", "coding", _body(), status="published")
        assert reg.search("astrophysics") == []

    def test_empty_query_returns_empty(self, reg):
        reg.add("Deploy", "coding", _body(), status="published")
        assert reg.search("") == []

    def test_limit_respected(self, reg):
        for i in range(5):
            reg.add(f"Deploy {i}", "coding", _body(), status="published")
        assert len(reg.search("deploy", limit=2)) == 2

    def test_relevant_returns_skills(self, reg):
        reg.add("Deploy", "coding", _body(), status="published")
        rel = reg.relevant("deploy")
        assert rel and isinstance(rel[0], sk.Skill)

    def test_search_can_include_drafts(self, reg):
        reg.add("Deploy", "coding", _body(), status="draft")
        assert reg.search("deploy", include_drafts=False) == []
        assert reg.search("deploy", include_drafts=True)


# Skill record


class TestSkillRecord:
    def test_to_markdown_has_frontmatter(self):
        s = sk.Skill(name="deploy", category="coding", body=_body())
        md = s.to_markdown()
        assert md.startswith("---")
        assert "name: deploy" in md
        assert "## When to Use" in md

    def test_reference_is_compact(self):
        s = sk.Skill(name="deploy", category="coding", version=3, body=_body(when="Trigger."))
        ref = s.reference()
        assert "deploy" in ref and "v3" in ref and "Trigger." in ref

    def test_to_dict_summary(self):
        s = sk.Skill(name="deploy", category="coding", body=_body(when="Use when X."))
        d = s.to_dict()
        assert d["name"] == "deploy"
        assert "Use when X." in d["summary"]


# Cartography


class TestCartography:
    def test_skills_registered_in_spec(self):
        text = SPEC.read_text(encoding="utf-8")
        assert "opti_oignon/agent/skills.py" in text

    def test_skill_md_path_in_spec(self):
        text = SPEC.read_text(encoding="utf-8")
        assert "data/skills/" in text
        assert "_usage.json" in text
