"""S236 release suite: the cas 7 (remote inference) cycle close as v3.11.0.

Families:
A. Version sites at 3.11.0: the version file (exact and bare, no rc),
   pyproject hardcoded (no dynamic), pyproject/version-file consistency.
B. CHANGELOG: top entry v3.11.0 dated S236, the cycle told lot by lot
   (S234 the served handler single-reply, S235 streaming Option A pull
   plus the per-device grant, the RAG read-only sub-grant, revocation,
   the rate gate, and the control surface), the inherited tier 1
   posture stated, the honest host-assured limitations naming
   HOST_SHAKEDOWN_S236.md and the per-chunk latency, no marketing
   prose, prior entries retained intact.
C. README: sits-between refreshed to 3.11.0, the intro cycle sentence,
   the v3.11.0 features section ABOVE the retained v3.10.0 one, the
   remote-chat API row, and the re-assertions of the three truths the
   superseded s232 TestReadmeRelease nodes carried.
D. REMOTE_INFERENCE_SPEC close-out: the header Status note, the
   delivery record (lots in order, why 3.11.0), the host-walk named.
E. ROADMAP_POST_AUDIT: cas 7 rolled to LANDED and RELEASED at S236
   (v3.11.0) while the spec-contract pin and the historical AGT/governor
   release pins stay untouched.
F. Supersessions and the addopts lineage, id by id: the eight new S236
   deselects present; the six s232, the five s214 and the fourteen s228
   deselects untouched; the total deselect count grew by exactly eight
   (198 -> 206, nothing removed).
G. Docs: the new remote-inference page covers the channel surface, the
   mkdocs nav entry exists, veilid-sync.md carries the cross-reference.
H. Host runbook: HOST_SHAKEDOWN_S236.md exists and is a HOST_SHAKEDOWN-
   class document -- host-assured, never simulated, with the numbers to
   record and a findings register.

Red-before discipline: on the pristine S235 tree the S236 mechanics fail
(version still 3.10.0, no v3.11.0 changelog entry, no README section, no
Status note, no roll, no new deselects, no docs page, no runbook) while
the retained-truth guards (prior changelog entries, the AGT/governor
roadmap pins, the s232/s214/s228 deselect lineage) hold green by
construction. After the release edits the whole suite is green.

Supersessions this suite re-asserts (deselect-plus-reassert; originals
never edited):
- tests/test_s232_release.py::TestVersionRelease::test_version_file_is_3100,
  ::test_version_bare_no_rc,
  ::test_pyproject_version_is_3100_and_hardcoded;
  ::TestChangelogRelease::test_top_entry_is_3100;
  ::TestReadmeRelease::test_sits_between_refreshed_to_3100,
  ::test_superseded_readme_truths_reasserted;
  ::TestAddoptsLineage::test_count_grew_by_exactly_six (the count node,
  re-asserted at the new total 206).
- tests/test_s233_cas7_spec.py::TestSpecExists::test_spec_status_decided (the
  spec status moves DECIDED -> RELEASED at this release; re-asserted by
  TestSpecCloseout::test_header_status_note).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FINAL = "3.11.0"
PREVIOUS = "3.10.0"


def _read(*parts: str) -> str:
    return ROOT.joinpath(*parts).read_text(encoding="utf-8")


def _norm(text: str) -> str:
    """Whitespace-flattening helper (the S221 lesson) for document pins."""
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# A. Version sites at 3.11.0
# ---------------------------------------------------------------------------


class TestVersionRelease:
    def test_version_file_is_3110(self):
        src = _read("opti_oignon", "__version__.py")
        assert f'"{FINAL}"' in src
        assert f'"{PREVIOUS}"' not in src

    def test_version_bare_no_rc(self):
        m = re.search(
            r'__version__\s*=\s*"([^"]+)"', _read("opti_oignon", "__version__.py")
        )
        assert m is not None
        assert m.group(1) == FINAL

    def test_pyproject_version_is_3110_and_hardcoded(self):
        src = _read("pyproject.toml")
        assert f'version = "{FINAL}"' in src
        data = tomllib.loads(src)
        assert "dynamic" not in data["project"]
        assert data["project"]["version"] == FINAL

    def test_pyproject_matches_version_file(self):
        m = re.search(
            r'__version__\s*=\s*"([^"]+)"', _read("opti_oignon", "__version__.py")
        )
        data = tomllib.loads(_read("pyproject.toml"))
        assert m is not None
        assert data["project"]["version"] == m.group(1)


# ---------------------------------------------------------------------------
# B. CHANGELOG
# ---------------------------------------------------------------------------


class TestChangelogRelease:
    def setup_method(self):
        self.c = _read("CHANGELOG.md")

    def _entry(self) -> str:
        return self.c.split("## v3.11.0")[1].split("## v3.10.0")[0]

    def test_top_entry_is_3110(self):
        entries = re.findall(r"## v(\d+\.\d+\.\d+)", self.c)
        assert entries and entries[0] == FINAL

    def test_entry_header_dated_s236(self):
        assert "## v3.11.0 -- 2026-06-08 (S236)" in self.c

    def test_entry_tells_the_cycle_lot_by_lot(self):
        entry = _norm(self._entry())
        for term in (
            "cas 7",
            "S234",
            "single",
            "request id",
            "VL-01",
            "bounded surface",
            "executor.execute",
            "Bulbe",
            "S235",
            "Option A",
            "cursor",
            "done marker",
            "sub-grant",
            "RAG",
            "revocation",
            "unpair",
            "rate",
            "telemetry",
            "RemoteChannelPanel",
            "SyncPanel",
            "SYN-06",
            "audit-chain",
        ):
            assert term in entry, term

    def test_entry_states_the_inherited_posture(self):
        entry = _norm(self._entry())
        assert "tier 1" in entry
        assert "Daily-only" in entry
        assert "instantly revocable" in entry
        assert "no admission bypass" in entry

    def test_entry_honest_limitations(self):
        entry = _norm(self._entry())
        assert "HOST_SHAKEDOWN_S236.md" in entry
        assert "host-assured" in entry
        assert "release before the walk" in entry
        assert "per-chunk" in entry

    def test_entry_no_marketing_prose(self):
        entry = self._entry().lower()
        for fluff in (
            "exciting",
            "amazing",
            "revolutionary",
            "game-chang",
            "seamless",
        ):
            assert fluff not in entry, fluff

    def test_previous_entries_retained(self):
        assert "## v3.10.0 -- 2026-06-08 (S232)" in self.c
        assert "## v3.9.0 -- 2026-06-07 (S227)" in self.c
        assert "## v3.8.0 -- 2026-06-06 (S214)" in self.c
        assert "## v3.7.0 -- 2026-06-05 (S208)" in self.c


# ---------------------------------------------------------------------------
# C. README
# ---------------------------------------------------------------------------


class TestReadmeRelease:
    def setup_method(self):
        self.src = _read("README.md")

    def test_sits_between_refreshed_to_3110(self):
        assert f"Opti-Oignon v{FINAL} sits between" in self.src
        assert f"Opti-Oignon v{PREVIOUS} sits between" not in self.src

    def test_intro_carries_the_cycle_sentence(self):
        assert "the remote inference cycle" in self.src

    def test_features_section_above_retained_previous(self):
        new = "## Features Added in v3.11.0 (Remote Inference Cycle)"
        old = "## Features Added in v3.10.0 (Agent Performance Cycle)"
        assert new in self.src
        assert old in self.src
        assert self.src.index(new) < self.src.index(old)

    def test_superseded_readme_truths_reasserted(self):
        # The three truths of the deselected s232 TestReadmeRelease nodes:
        # the retained v3.10.0 features section, the sits-between line
        # (now at 3.11.0), the retained agent-performance cycle phrase.
        assert "## Features Added in v3.10.0 (Agent Performance Cycle)" in self.src
        assert f"Opti-Oignon v{FINAL} sits between" in self.src
        assert "the agent performance cycle" in self.src

    def test_remote_chat_api_row(self):
        assert "| `/api/sync/peers/{peer_id}/remote-chat` |" in self.src
        assert "channel telemetry" in self.src


# ---------------------------------------------------------------------------
# D. REMOTE_INFERENCE_SPEC close-out
# ---------------------------------------------------------------------------


class TestSpecCloseout:
    def setup_method(self):
        self.src = _norm(_read("REMOTE_INFERENCE_SPEC.md"))

    def test_header_status_note(self):
        assert "Status: RELEASED at S236 as v3.11.0" in self.src
        assert "HOST_SHAKEDOWN_S236.md" in self.src

    def test_delivery_record(self):
        assert "Delivery record: RELEASED" in self.src
        assert "Lot 1 S234" in self.src
        assert "Lot 2 S235" in self.src
        assert "Lot 3 S236" in self.src
        assert "feature-cycle-minor precedent" in self.src
        assert "release before the walk" in self.src

    def test_host_walk_named(self):
        assert "host-assured" in self.src
        assert "per-chunk" in self.src
        assert "two real devices" in self.src


# ---------------------------------------------------------------------------
# E. ROADMAP_POST_AUDIT roll
# ---------------------------------------------------------------------------


class TestRoadmapRoll:
    def setup_method(self):
        self.src = _norm(_read("ROADMAP_POST_AUDIT.md"))

    def test_cas7_rolled(self):
        assert "LANDED and RELEASED at S236 (v3.11.0)" in self.src

    def test_cas7_spec_contract_pin_retained(self):
        assert "REMOTE_INFERENCE_SPEC.md is the design contract" in self.src

    def test_historical_release_pins_untouched(self):
        assert "LANDED and RELEASED at S232 (v3.10.0)" in self.src
        assert "LANDED and RELEASED at S227 (v3.9.0)" in self.src


# ---------------------------------------------------------------------------
# F. Supersessions and the addopts lineage
# ---------------------------------------------------------------------------

S236_DESELECTS = (
    "test_s232_release.py::TestVersionRelease::test_version_file_is_3100",
    "test_s232_release.py::TestVersionRelease::test_version_bare_no_rc",
    "test_s232_release.py::TestVersionRelease::"
    "test_pyproject_version_is_3100_and_hardcoded",
    "test_s232_release.py::TestChangelogRelease::test_top_entry_is_3100",
    "test_s232_release.py::TestReadmeRelease::test_sits_between_refreshed_to_3100",
    "test_s232_release.py::TestReadmeRelease::test_superseded_readme_truths_reasserted",
    "test_s232_release.py::TestAddoptsLineage::test_count_grew_by_exactly_six",
    "test_s233_cas7_spec.py::TestSpecExists::test_spec_status_decided",
)

S232_DESELECTS = (
    "test_s227_governor_bloc4.py::TestVersionRelease::test_version_file_is_390",
    "test_s227_governor_bloc4.py::TestVersionRelease::test_version_bare_no_rc",
    "test_s227_governor_bloc4.py::TestVersionRelease::"
    "test_pyproject_version_is_390_and_hardcoded",
    "test_s227_governor_bloc4.py::TestChangelogRelease::test_top_entry_is_390",
    "test_s227_governor_bloc4.py::TestReadme::test_feature_section_and_intro",
    "test_s230_agt_lot3.py::TestDeliverablePins::test_version_holds",
)

S214_DESELECTS = (
    "test_s214_release.py::TestVersionRelease::test_version_file_is_380",
    "test_s214_release.py::TestVersionRelease::test_version_bare_no_rc",
    "test_s214_release.py::TestVersionRelease::"
    "test_pyproject_version_is_380_and_hardcoded",
    "test_s214_release.py::TestChangelogRelease::test_top_entry_is_380",
    "test_s214_release.py::TestReadmeRelease::test_sits_between_refreshed_to_380",
)

S228_DESELECTS = (
    "test_s176_tools.py::TestSchemas::test_six_schemas",
    "test_s176_tools.py::TestSchemas::test_handler_two_not_sandboxed",
    "test_s176_tools.py::TestRegistryPerMode::test_daily_handlers_are_non_sandbox_two",
    "test_s176_tools.py::TestRegistryPerMode::test_bulbe_has_no_handlers",
    "test_s176_tools.py::TestRegistryPerMode::test_bulbe_exposes_sandbox_only",
    "test_s176_tools.py::TestRegistryPerMode::test_unknown_mode_is_fail_secure_bulbe",
    "test_s176_tools.py::TestSchemas::"
    "test_sandbox_argument_names_cover_dispatch_lambdas",
    "test_s177_manage_skills.py::TestSchemaSupersede::test_seven_schemas",
    "test_s177_manage_skills.py::TestSchemaSupersede::"
    "test_manage_skills_is_third_non_sandbox",
    "test_s177_manage_skills.py::TestSchemaSupersede::test_daily_includes_manage_skills",
    "test_s177_manage_skills.py::TestSchemaSupersede::test_bulbe_excludes_manage_skills",
    "test_s222_agt_spec.py::TestSeamToolsSchemas::test_all_schemas_is_seven_today",
    "test_s222_agt_spec.py::TestSeamToolsSchemas::test_handler_names_are_the_three",
    "test_s222_agt_spec.py::TestSeamAllowlists::test_frozensets_exact",
)


class TestAddoptsLineage:
    def setup_method(self):
        self.src = _read("pyproject.toml")

    def test_carries_the_eight_s236_supersessions(self):
        for node in S236_DESELECTS:
            assert f"--deselect=tests/{node}" in self.src, node

    def test_s232_six_untouched(self):
        for node in S232_DESELECTS:
            assert f"--deselect=tests/{node}" in self.src, node

    def test_s214_five_untouched(self):
        for node in S214_DESELECTS:
            assert f"--deselect=tests/{node}" in self.src, node

    def test_s228_fourteen_untouched(self):
        for node in S228_DESELECTS:
            assert f"--deselect=tests/{node}" in self.src, node

    def test_count_grew_by_exactly_eight(self):
        # 198 deselects at the S235 close; the eight S236 supersessions
        # (the seven s232 release pins plus the s233 spec-status pin,
        # superseded as the spec status moves DECIDED -> RELEASED) join;
        # nothing is ever removed from the lineage.
        assert self.src.count("--deselect=") == 206


# ---------------------------------------------------------------------------
# G. Docs: the page, the nav, the cross-reference
# ---------------------------------------------------------------------------


class TestDocsRelease:
    def test_page_exists_and_covers_the_channel(self):
        src = _norm(_read("docs", "sync", "remote-inference.md"))
        for term in (
            "tier 1",
            "RAG",
            "sub-grant",
            "revocation",
            "rate",
            "telemetry",
            "Option A",
            "cursor",
            "Bulbe",
            "Daily",
            "audit",
            "HOST_SHAKEDOWN_S236.md",
            "SYN-06",
            "RemoteChannelPanel",
        ):
            assert term in src, term

    def test_nav_entry(self):
        nav = _read("mkdocs.yml")
        assert "Remote Inference: sync/remote-inference.md" in nav

    def test_cross_reference_in_veilid_sync(self):
        src = _read("docs", "sync", "veilid-sync.md")
        assert "remote-inference.md" in src


# ---------------------------------------------------------------------------
# H. Host runbook: the HOST_SHAKEDOWN-class document
# ---------------------------------------------------------------------------


class TestHostRunbook:
    def setup_method(self):
        self.src = _read("HOST_SHAKEDOWN_S236.md")

    def test_exists_and_is_findings_not_fixes(self):
        norm = _norm(self.src)
        assert "host-assured" in norm
        assert "never simulated" in norm
        assert "findings" in norm.lower()

    def test_covers_the_numbers_to_record(self):
        norm = _norm(self.src)
        for term in (
            "per-chunk",
            "revocation",
            "rate",
            "Bulbe",
            "Daily",
        ):
            assert term in norm, term

    def test_two_device_exercise(self):
        norm = _norm(self.src)
        assert ("two devices" in norm) or ("desktop-to-desktop" in norm)
