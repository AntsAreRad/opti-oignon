#!/usr/bin/env python3
"""S214 -- Sandbox Workspace cycle RELEASE: 3.8.0, CHANGELOG, docs, roll-up.

File-content assertions for the release session's whole surface: the version
sites exact at 3.8.0; the CHANGELOG top entry telling the cycle story bloc by
bloc with the security postures stated; the docs pages existing and carrying
the posture strings (default-off, Daily-only, approval-gated, fail-secure);
the spec section 13/14 close-out by source; the shakedown roll-up
(SANDBOX_CYCLE_LIVE_WALK.md, SW0 argv capture FIRST) with the handoff pointer
and routing; the DOC-01 absorption (the eight orphan docs pages joined to the
mkdocs nav, nothing deleted); and the registrations.

Supersessions carried by this session (deselect-plus-reassert; the originals
in tests/test_s208_sync_bloc4.py are never edited):
- test_s208_sync_bloc4.py::TestVersionRelease::test_version_file_is_370
- test_s208_sync_bloc4.py::TestVersionRelease::test_version_bare_no_rc
- test_s208_sync_bloc4.py::TestVersionRelease::test_pyproject_version_is_370_and_hardcoded
- test_s208_sync_bloc4.py::TestChangelogRelease::test_top_entry_is_370
- test_s208_sync_bloc4.py::TestDocsRelease::test_readme_refreshed_to_370_with_sync_section
  (the five 3.7.0 pins; each re-asserted at 3.8.0 here, in the s208-on-s182
  lineage. The rest of the s208 release section stays green: the 3.7.0
  CHANGELOG entry, the sync docs page, VEILID_SPEC, and the closed sync
  roadmap are untouched by this release.)
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FINAL_VERSION = "3.8.0"
PREVIOUS_VERSION = "3.7.0"


def _read(*parts) -> str:
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def _norm(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# A. Version sites (the five superseded s208 pins, re-asserted at 3.8.0)
# ---------------------------------------------------------------------------


class TestVersionRelease:
    """Reasserts the superseded s208 pins at 3.8.0."""

    def test_version_file_is_380(self):
        src = _read("opti_oignon", "__version__.py")
        assert '"3.8.0"' in src
        assert PREVIOUS_VERSION not in src

    def test_version_bare_no_rc(self):
        m = re.search(r'__version__\s*=\s*"([^"]+)"', _read("opti_oignon", "__version__.py"))
        assert m and re.match(r"^\d+\.\d+\.\d+$", m.group(1))
        assert m.group(1) == FINAL_VERSION

    def test_pyproject_version_is_380_and_hardcoded(self):
        src = _read("pyproject.toml")
        assert f'version = "{FINAL_VERSION}"' in src
        assert f'version = "{PREVIOUS_VERSION}"' not in src
        import tomllib

        data = tomllib.loads(src)
        assert "dynamic" not in data["project"]
        assert data["project"]["version"] == FINAL_VERSION

    def test_pyproject_consistent_with_version_file(self):
        m = re.search(r'__version__\s*=\s*"([^"]+)"', _read("opti_oignon", "__version__.py"))
        import tomllib

        data = tomllib.loads(_read("pyproject.toml"))
        assert data["project"]["version"] == m.group(1)

    def test_addopts_carries_the_new_version_supersessions(self):
        src = _read("pyproject.toml")
        for node in (
            "test_s208_sync_bloc4.py::TestVersionRelease::test_version_file_is_370",
            "test_s208_sync_bloc4.py::TestVersionRelease::test_version_bare_no_rc",
            "test_s208_sync_bloc4.py::TestVersionRelease::test_pyproject_version_is_370_and_hardcoded",
            "test_s208_sync_bloc4.py::TestChangelogRelease::test_top_entry_is_370",
            "test_s208_sync_bloc4.py::TestDocsRelease::test_readme_refreshed_to_370_with_sync_section",
        ):
            assert f"--deselect=tests/{node}" in src, node

    def test_addopts_still_carries_the_s182_lineage(self):
        src = _read("pyproject.toml")
        for node in (
            "test_s182_release.py::TestVersionBump::test_version_file_is_final",
            "test_s197_f10a.py::test_ds03_version_file_is_360",
        ):
            assert f"--deselect=tests/{node}" in src, node

    def test_historical_veilid_mentions_untouched(self):
        # The 3.7.0 strings in the veilid modules are HISTORY (the S208 grace
        # flip and the fleet upgrade order), not live version sites.
        assert "3.7.0" in _read("opti_oignon", "veilid", "signing.py")
        assert "3.7.0" in _read("opti_oignon", "veilid", "sync_engine.py")


# ---------------------------------------------------------------------------
# B. CHANGELOG: the cycle story bloc by bloc, the postures stated
# ---------------------------------------------------------------------------


class TestChangelogRelease:
    def setup_method(self):
        self.c = _read("CHANGELOG.md")

    def test_top_entry_is_380(self):
        entries = re.findall(r"## v(\d+\.\d+\.\d+)", self.c)
        assert entries and entries[0] == FINAL_VERSION

    def test_previous_entries_retained(self):
        assert "## v3.7.0 -- 2026-06-05 (S208)" in self.c
        assert "## v3.6.0 -- 2026-06-02 (S182)" in self.c

    def _entry(self) -> str:
        return self.c.split("## v3.8.0")[1].split("## v3.7.0")[0]

    def test_entry_tells_the_cycle_bloc_by_bloc(self):
        entry = _norm(self._entry())
        for term in (
            "S209", "seccomp", "rlimit", "cloistering", "tmpfs",
            "S210", "conversation binding", "TTL",
            "S211", "drag-and-drop", "baseline manifest",
            "S212", "diff", "deletions confirmed separately", "symlink-safe",
            "S213", "provision", "--require-hashes", "--unshare-net",
            "[SECURITY]",
        ):
            assert term in entry, term

    def test_entry_states_the_security_postures(self):
        entry = _norm(self._entry())
        for posture in ("default OFF", "Daily-only", "fail-secure", "approval gate"):
            assert posture in entry, posture
        assert "never model-triggerable" in entry

    def test_entry_states_kerckhoffs_and_s73(self):
        entry = _norm(self._entry())
        assert "Kerckhoffs" in entry
        assert "S73/S74" in entry

    def test_entry_no_marketing_prose(self):
        entry = self._entry().lower()
        for fluff in ("exciting", "amazing", "revolutionary", "game-chang", "seamless"):
            assert fluff not in entry, fluff

    def test_entry_records_the_walk_and_capture_first(self):
        entry = _norm(self._entry())
        assert "SANDBOX_CYCLE_LIVE_WALK.md" in entry
        assert "capture-before-deploy" in entry


# ---------------------------------------------------------------------------
# C. Docs: README, SECURITY.md, the new page, the nav, DOC-01 absorbed
# ---------------------------------------------------------------------------


class TestReadmeRelease:
    def setup_method(self):
        self.r = _read("README.md")

    def test_sits_between_refreshed_to_380(self):
        assert "Opti-Oignon v3.8.0 sits between" in self.r
        assert "Opti-Oignon v3.7.0 sits between" not in self.r

    def test_intro_carries_the_cycle_sentence(self):
        norm = _norm(self.r)
        assert "v3.8.0 (the sandbox workspace cycle)" in norm
        assert "task code never sees the network" in norm

    def test_features_section_added_and_previous_kept(self):
        assert "## Features Added in v3.8.0 (Sandbox Workspace Cycle)" in self.r
        assert "## Features Added in v3.7.0 (Sync Cycle)" in self.r
        assert "## Features Added in v3.5.0" in self.r

    def test_layer3_block_refreshed(self):
        norm = _norm(self.r)
        assert "seccomp denylist" in norm
        assert "diff-gated write-back" in norm
        assert "default-off, user-only, Daily-only, provision-phase egress" in norm
        assert "Sandbox copy-out restricted to data/" not in self.r


class TestSecurityMdRelease:
    def setup_method(self):
        self.s = _read("SECURITY.md")

    def test_layer3_states_the_containment(self):
        norm = _norm(self.s)
        for term in (
            "seccomp-BPF syscall denylist",
            "unshare-ipc/uts/cgroup",
            "tmpfs size cap",
            "strict mode refuses execution",
        ):
            assert term in norm, term

    def test_layer3_maps_the_four_s73_clauses(self):
        norm = _norm(self.s)
        assert "clause by clause" in norm
        assert "fully isolated, disposable environment" in norm
        assert "explicit copy-in" in norm
        assert "deletions are confirmed separately" in norm
        assert "hash-bound to the reviewed diff" in norm
        assert "Auto-apply does not exist" in norm

    def test_layer3_states_the_network_posture(self):
        norm = _norm(self.s)
        assert "default-off" in norm
        assert "Daily-only at a fail-secure binding-layer gate" in norm
        assert "never model-triggerable" in norm
        assert "task code never sees the network" in norm


class TestSandboxWorkspacesDocsPage:
    def setup_method(self):
        self.d = _read("docs", "agent", "sandbox-workspaces.md")

    def test_page_exists_with_the_sections(self):
        for header in (
            "# Sandbox Workspaces",
            "## Containment",
            "## The manager",
            "## Copy-in",
            "## Diff review and apply",
            "## The settings strip",
            "## The optional network",
            "## Honest limitations",
        ):
            assert header in self.d, header

    def test_page_states_the_postures(self):
        norm = _norm(self.d)
        assert "default-off" in norm
        assert "Daily-only" in norm
        assert "approval-gated" in norm
        assert "fail-secure" in norm
        assert "never model-triggerable" in norm

    def test_exfiltration_rationale_stated_plainly(self):
        norm = _norm(self.d)
        assert "a second exit that the approval gate does not cover" in norm
        assert "sharpened when host files have been cloned in" in norm

    def test_honest_limitations_register(self):
        norm = _norm(self.d)
        assert "bwrap is required" in norm
        assert "assures only on a real host" in norm
        assert "SANDBOX_CYCLE_LIVE_WALK.md" in norm

    def test_cross_reference_from_sandboxed_agent_page(self):
        src = _norm(_read("docs", "agent", "sandboxed-agent.md"))
        assert "sandbox-workspaces.md" in src


class TestMkdocsNavRelease:
    def setup_method(self):
        self.n = _read("mkdocs.yml")

    def test_new_page_in_nav(self):
        assert "Sandbox Workspaces: agent/sandbox-workspaces.md" in self.n

    def test_doc01_orphans_joined_to_nav(self):
        for page in (
            "SECURITY_AUDIT_S155.md",
            "security/DESIGN_NOTE_EX01_FAIL_SECURE_TOOL_GATE.md",
            "API_REFERENCE.md",
            "PLUGIN_DEVELOPMENT_GUIDE.md",
            "demo_scenarios.md",
            "BRANCH_PROTECTION.md",
            "ROADMAP_SIDE_QUESTS.md",
            "COLOR_INSPIRATION_S93.md",
        ):
            assert page in self.n, page

    def test_no_docs_page_left_orphan(self):
        pages = sorted(
            str(p.relative_to(ROOT / "docs"))
            for p in (ROOT / "docs").rglob("*.md")
        )
        orphans = [p for p in pages if p not in self.n]
        assert orphans == [], orphans

    def test_doc01_absorbed_without_deletion(self):
        # Absorption means joining the nav, never deleting or renaming.
        for page in (
            "docs/API_REFERENCE.md",
            "docs/api-reference.md",
            "docs/PLUGIN_DEVELOPMENT_GUIDE.md",
            "docs/plugin-development.md",
        ):
            assert (ROOT / page).is_file(), page


# ---------------------------------------------------------------------------
# D. Spec close-out: sections 13/14 by source, the status note, the 15 row
# ---------------------------------------------------------------------------


class TestSpecCloseOut:
    def setup_method(self):
        self.s = _read("SANDBOX_WORKSPACE_SPEC.md")
        self.norm = _norm(self.s)

    def test_header_status_note(self):
        assert "Status: RELEASED at S214 as v3.8.0" in self.norm
        assert "SANDBOX_CYCLE_LIVE_WALK.md" in self.norm

    def test_section13_delivery_record(self):
        assert "Delivery record: RELEASED" in self.norm
        assert "closed the cycle as v3.8.0" in self.norm
        assert "Bloc 0 S209, Bloc 1 S210, Bloc 2 S211, Bloc 3 S212, Bloc 4 S213" in self.norm

    def test_section14_answers_present(self):
        assert "Answered at the release (S214), with what shipped:" in self.norm
        assert "Workspace creation is EXPLICIT" in self.norm
        assert "`$HOME` plus the configured list, never `/`" in self.norm
        assert "The egress default is the PROVISION PHASE" in self.norm
        assert "Persistent workspaces stayed OPTIONAL" in self.norm

    def test_section14_questions_retained(self):
        # The questions stay on the record above their answers.
        assert "Open questions for the maintainer:" in self.norm
        assert "Spec assumes explicit" in self.norm

    def test_out_of_scope_unchanged(self):
        assert "Raw `--share-net` for the sandbox. Never" in self.norm
        assert "Webhooks. Remain permanently cancelled" in self.norm

    def test_section15_release_row(self):
        assert "Release (S214, `tests/test_s214_release.py`)" in self.norm
        assert "default-off, Daily-only, fail-secure, approval-gated" in self.norm


# ---------------------------------------------------------------------------
# E. The shakedown roll-up: the walk document and the handoff pointers
# ---------------------------------------------------------------------------


class TestLiveWalkPrepared:
    def setup_method(self):
        self.w = _read("SANDBOX_CYCLE_LIVE_WALK.md")

    def test_walk_has_the_seven_items(self):
        for n in range(0, 7):
            assert f"## SW{n}." in self.w, f"SW{n}"

    def test_argv_capture_is_the_first_item_and_first(self):
        norm = _norm(self.w)
        assert "ARGV CAPTURE ORDERING -- run this FIRST" in norm
        assert self.w.index("## SW0.") < self.w.index("## SW1.")
        assert "capture-before-deploy" in norm

    def test_walk_covers_the_mandated_surface(self):
        norm = _norm(self.w)
        for term in (
            "BLOC0_HOST_ASSURANCE",
            "Rendered manager walk",
            "Rendered copy-in walk",
            "REAL apply onto a host directory",
            "Live provision run",
            "real DNS",
            "rc 0",
            "Rendered settings-strip walk",
            "DISABLED with the refusal stated",
            "exfiltration warning",
        ):
            assert term in norm, term

    def test_walk_routes_per_the_register_discipline(self):
        norm = _norm(self.w)
        assert "shakedown findings register" in norm
        assert "never patched blind from the host" in norm

    def test_proxy_mode_stays_out_of_the_walk(self):
        norm = _norm(self.w)
        assert "prepared-and-labelled (not wired) and is NOT a walk item" in norm


class TestHandoffRollUp:
    def setup_method(self):
        self.h = _read("SHAKEDOWN_S198_HANDOFF.md")
        self.norm = _norm(self.h)

    def test_section_e_carries_the_rollup_pointer(self):
        assert "### Sandbox Workspace cycle roll-up (S214)" in self.h
        assert "SANDBOX_CYCLE_LIVE_WALK.md" in self.norm
        assert "The cycle LANDED (S209-S213, 3.8.0)" in self.norm

    def test_rollup_orders_the_capture_first(self):
        assert "SW0, capture-before-deploy" in self.norm

    def test_routing_updated(self):
        assert "consolidated in SANDBOX_CYCLE_LIVE_WALK.md (run it with this list; SW0 first)" in self.norm

    def test_bloc0_subsection_retained(self):
        assert "### BLOC0_HOST_ASSURANCE (S209)" in self.h


# ---------------------------------------------------------------------------
# F. Registrations and cross-spec invariants
# ---------------------------------------------------------------------------


class TestRegistrations:
    def test_frontend_redesign_untouched_rows_present(self):
        src = _read("FRONTEND_REDESIGN_SPEC.md")
        # No new component this session; the cycle's rows stay registered.
        assert "SandboxSettingsStrip.svelte" in src
        assert "S214" not in src

    def test_odysseus_spec_carries_no_s214_marker(self):
        assert "S214" not in _read("ODYSSEUS_SPEC.md")

    def test_auth_core_not_referenced_by_this_session(self):
        # The release touches no source module beyond the version literal.
        assert "3.8.0" not in _read("opti_oignon", "auth.py")
        assert "3.8.0" not in _read("opti_oignon", "auth_2fa.py")

    def test_frontend_package_json_versioning_independent(self):
        import json

        data = json.loads(_read("frontend", "package.json"))
        # The frontend keeps its own version line; the product bump never
        # touches it (dependency pins like svelte-check ^3.8.0 are unrelated).
        assert data["version"] != FINAL_VERSION
