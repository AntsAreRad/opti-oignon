"""Tests for S164 -- v3.3.0 Release.

Validates version bump, CHANGELOG entry, README updates, SECURITY.md
creation, MkDocs accuracy, and release signing scripts.
"""

import importlib.util
import os
import re
import sys

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(relpath: str) -> str:
    """Read a file relative to project root."""
    with open(os.path.join(ROOT, relpath), encoding="utf-8") as f:
        return f.read()


def _load_version() -> str:
    """Load __version__ via importlib isolation."""
    path = os.path.join(ROOT, "opti_oignon", "__version__.py")
    spec = importlib.util.spec_from_file_location("__version__", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.__version__


# ---------------------------------------------------------------------------
# Class 1: Version
# ---------------------------------------------------------------------------

class TestVersion:
    """Version bump to 3.3.0."""

    def test_version_string(self):
        assert _load_version() == "3.3.0"

    def test_version_file_contains_330(self):
        content = _read("opti_oignon/__version__.py")
        assert '"3.3.0"' in content

    def test_version_file_no_old_version(self):
        content = _read("opti_oignon/__version__.py")
        assert "3.2.11" not in content

    def test_version_file_ast_valid(self):
        import ast
        content = _read("opti_oignon/__version__.py")
        ast.parse(content)

    def test_version_pep440_compliant(self):
        v = _load_version()
        assert re.match(r"^\d+\.\d+\.\d+$", v), f"Not PEP 440: {v}"


# ---------------------------------------------------------------------------
# Class 2: CHANGELOG
# ---------------------------------------------------------------------------

class TestChangelog:
    """CHANGELOG.md v3.3.0 entry."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read("CHANGELOG.md")

    def test_v330_entry_exists(self):
        assert "## v3.3.0" in self.content

    def test_v330_date(self):
        assert "v3.3.0 -- 2026-04-16 (S164)" in self.content

    def test_v330_before_v3211(self):
        pos_330 = self.content.index("## v3.3.0")
        pos_3211 = self.content.index("## v3.2.11")
        assert pos_330 < pos_3211

    def test_v330_has_added_section(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "### Added" in entry

    def test_v330_has_security_section(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "### Security" in entry

    def test_v330_has_changed_section(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "### Changed" in entry

    def test_v330_has_metrics_section(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "### Cumulative Metrics" in entry

    def test_v330_references_bloc1_theme(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "Theme engine" in entry or "theme engine" in entry

    def test_v330_references_bloc2_audit(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "security audit" in entry.lower()

    def test_v330_references_bloc3_streaming(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "backpressure" in entry.lower() or "streaming" in entry.lower()

    def test_v330_references_bloc4_ci(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "GitHub Actions" in entry

    def test_v330_references_docker(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "Docker" in entry

    def test_v330_references_mkdocs(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "MkDocs" in entry

    def test_v330_references_keyboard_shortcuts(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "Keyboard shortcuts" in entry or "keyboard shortcuts" in entry

    def test_v330_references_accessibility(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "ccessibility" in entry  # Accessibility or accessibility

    def test_v330_references_conversation_branching(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "branching" in entry.lower()

    def test_v330_references_async_plugins(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "Async" in entry or "async" in entry

    def test_v330_references_parallel_rag(self):
        entry = self.content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        assert "Parallel RAG" in entry or "parallel RAG" in entry


# ---------------------------------------------------------------------------
# Class 3: SECURITY.md
# ---------------------------------------------------------------------------

class TestSecurityMd:
    """Root-level SECURITY.md."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read("SECURITY.md")

    def test_file_exists(self):
        assert os.path.isfile(os.path.join(ROOT, "SECURITY.md"))

    def test_title(self):
        assert self.content.startswith("# Security")

    def test_threat_model_section(self):
        assert "## Threat Model" in self.content

    def test_defense_layers_section(self):
        assert "## Defense Layers" in self.content

    def test_six_layers_listed(self):
        for i in range(1, 7):
            assert f"### Layer {i}" in self.content

    def test_security_modes_section(self):
        assert "## Security Modes" in self.content

    def test_bulbe_mode_documented(self):
        assert "Bulbe mode" in self.content

    def test_daily_mode_documented(self):
        assert "Daily mode" in self.content

    def test_deployment_recommendations(self):
        assert "## Deployment Recommendations" in self.content

    def test_docker_security_posture(self):
        assert "Docker" in self.content
        assert "127.0.0.1" in self.content

    def test_docker_optional_documented(self):
        assert "optional" in self.content.lower()

    def test_native_recommended(self):
        assert "native" in self.content.lower()

    def test_release_signing_section(self):
        assert "## Release Signing" in self.content

    def test_sign_release_script_referenced(self):
        assert "sign_release.sh" in self.content

    def test_verify_release_script_referenced(self):
        assert "verify_release.sh" in self.content

    def test_cicd_signing_documented(self):
        assert "release.yml" in self.content or "release workflow" in self.content.lower()

    def test_csp_documented(self):
        assert "Content Security Policy" in self.content or "CSP" in self.content

    def test_vulnerability_reporting_section(self):
        assert "## Vulnerability Reporting" in self.content

    def test_audit_history_section(self):
        assert "## Security Audit History" in self.content

    def test_s155_s156_audit_referenced(self):
        assert "S155" in self.content and "S156" in self.content

    def test_further_reading_section(self):
        assert "## Further Reading" in self.content

    def test_kerckhoffs_principle(self):
        assert "Kerckhoffs" in self.content

    def test_no_french_text(self):
        french_markers = ["securite", "chiffrement", "connexion", "utilisateur"]
        lower = self.content.lower()
        for marker in french_markers:
            assert marker not in lower, f"French word found: {marker}"


# ---------------------------------------------------------------------------
# Class 4: README.md
# ---------------------------------------------------------------------------

class TestReadme:
    """README.md updated for v3.3.0."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read("README.md")

    def test_version_reference_v330(self):
        assert "v3.3.0" in self.content

    def test_no_stale_v320_in_description(self):
        # The first paragraph should mention v3.3.0, not v3.2.0
        first_para = self.content[:500]
        assert "v3.2.0" not in first_para

    def test_features_added_v330_section(self):
        assert "## Features Added in v3.3.0" in self.content

    def test_theme_engine_feature(self):
        section = self.content.split("## Features Added in v3.3.0")[1]
        section = section.split("## Features Added in v3.2.0")[0]
        assert "Theme Engine" in section

    def test_keyboard_shortcuts_feature(self):
        section = self.content.split("## Features Added in v3.3.0")[1]
        section = section.split("## Features Added in v3.2.0")[0]
        assert "Keyboard Shortcuts" in section

    def test_accessibility_feature(self):
        section = self.content.split("## Features Added in v3.3.0")[1]
        section = section.split("## Features Added in v3.2.0")[0]
        assert "Accessibility" in section

    def test_conversation_branching_feature(self):
        section = self.content.split("## Features Added in v3.3.0")[1]
        section = section.split("## Features Added in v3.2.0")[0]
        assert "Branching" in section

    def test_security_audit_feature(self):
        section = self.content.split("## Features Added in v3.3.0")[1]
        section = section.split("## Features Added in v3.2.0")[0]
        assert "Security Audit" in section

    def test_streaming_feature(self):
        section = self.content.split("## Features Added in v3.3.0")[1]
        section = section.split("## Features Added in v3.2.0")[0]
        assert "Streaming" in section

    def test_async_plugins_feature(self):
        section = self.content.split("## Features Added in v3.3.0")[1]
        section = section.split("## Features Added in v3.2.0")[0]
        assert "Async Plugins" in section

    def test_mkdocs_feature(self):
        section = self.content.split("## Features Added in v3.3.0")[1]
        section = section.split("## Features Added in v3.2.0")[0]
        assert "MkDocs" in section

    def test_ci_feature(self):
        section = self.content.split("## Features Added in v3.3.0")[1]
        section = section.split("## Features Added in v3.2.0")[0]
        assert "GitHub Actions CI" in section

    def test_release_automation_feature(self):
        section = self.content.split("## Features Added in v3.3.0")[1]
        section = section.split("## Features Added in v3.2.0")[0]
        assert "Release Automation" in section

    def test_docker_feature(self):
        section = self.content.split("## Features Added in v3.3.0")[1]
        section = section.split("## Features Added in v3.2.0")[0]
        assert "Docker" in section

    def test_endpoint_count_updated(self):
        assert "519+" in self.content

    def test_no_stale_494_endpoints(self):
        assert "494" not in self.content

    def test_test_count_updated(self):
        assert "9300" in self.content or "9,300" in self.content

    def test_no_stale_8172_tests(self):
        assert "8172" not in self.content

    def test_test_files_updated(self):
        assert "151 test files" in self.content

    def test_no_stale_137_test_files(self):
        assert "137 test files" not in self.content

    def test_module_count_updated(self):
        assert "~255" in self.content

    def test_component_count_updated(self):
        assert "137 Svelte" in self.content

    def test_session_range_updated(self):
        assert "S42-S164" in self.content

    def test_roadmap_includes_s151_s164(self):
        assert "S151-S164" in self.content

    def test_security_md_link(self):
        assert "[SECURITY.md](SECURITY.md)" in self.content


# ---------------------------------------------------------------------------
# Class 5: Stale version references
# ---------------------------------------------------------------------------

class TestStaleVersions:
    """No stale 3.2.11 version assertions outside historical entries."""

    def test_version_py_is_330(self):
        content = _read("opti_oignon/__version__.py")
        match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
        assert match and match.group(1) == "3.3.0"

    def test_no_3211_in_version_file(self):
        content = _read("opti_oignon/__version__.py")
        assert "3.2.11" not in content

    def test_changelog_first_entry_is_v330(self):
        content = _read("CHANGELOG.md")
        entries = re.findall(r"## v(\d+\.\d+\.\d+)", content)
        assert entries[0] == "3.3.0"

    def test_readme_description_has_v330(self):
        content = _read("README.md")
        first_line_with_version = [
            l for l in content.split("\n")
            if "Opti-Oignon v" in l
        ][0]
        assert "v3.3.0" in first_line_with_version

    def test_docs_index_has_v330(self):
        content = _read("docs/index.md")
        assert "v3.3.0" in content


# ---------------------------------------------------------------------------
# Class 6: MkDocs consistency
# ---------------------------------------------------------------------------

class TestMkDocs:
    """MkDocs configuration and page accuracy."""

    def test_mkdocs_yml_exists(self):
        assert os.path.isfile(os.path.join(ROOT, "mkdocs.yml"))

    def test_nav_has_security_guide(self):
        content = _read("mkdocs.yml")
        assert "Security Guide" in content

    def test_nav_has_red_team_guide(self):
        content = _read("mkdocs.yml")
        assert "Red Team Guide" in content

    def test_nav_has_keyboard_shortcuts(self):
        content = _read("mkdocs.yml")
        assert "Keyboard Shortcuts" in content

    def test_nav_has_architecture(self):
        content = _read("mkdocs.yml")
        assert "Architecture" in content

    def test_all_nav_files_exist(self):
        content = _read("mkdocs.yml")
        md_files = re.findall(r":\s+(\S+\.md)", content)
        for md_file in md_files:
            full_path = os.path.join(ROOT, "docs", md_file)
            assert os.path.isfile(full_path), f"Missing nav file: {md_file}"

    def test_index_version_is_v330(self):
        content = _read("docs/index.md")
        assert "v3.3.0" in content

    def test_module_map_test_count(self):
        content = _read("docs/architecture/module-map.md")
        assert "9300" in content or "9,300" in content

    def test_module_map_test_files(self):
        content = _read("docs/architecture/module-map.md")
        assert "151 files" in content


# ---------------------------------------------------------------------------
# Class 7: Release signing scripts
# ---------------------------------------------------------------------------

class TestReleaseScripts:
    """Release signing and verification scripts."""

    def test_sign_release_exists(self):
        path = os.path.join(ROOT, "scripts", "sign_release.sh")
        assert os.path.isfile(path)

    def test_verify_release_exists(self):
        path = os.path.join(ROOT, "scripts", "verify_release.sh")
        assert os.path.isfile(path)

    def test_sign_release_executable(self):
        path = os.path.join(ROOT, "scripts", "sign_release.sh")
        assert os.access(path, os.X_OK)

    def test_verify_release_executable(self):
        path = os.path.join(ROOT, "scripts", "verify_release.sh")
        assert os.access(path, os.X_OK)

    def test_sign_release_uses_gpg(self):
        content = _read("scripts/sign_release.sh")
        assert "gpg" in content

    def test_sign_release_creates_sha256(self):
        content = _read("scripts/sign_release.sh")
        assert "sha256sum" in content

    def test_sign_release_creates_sig(self):
        content = _read("scripts/sign_release.sh")
        assert "detach-sign" in content

    def test_verify_release_checks_signature(self):
        content = _read("scripts/verify_release.sh")
        assert "gpg" in content and "verify" in content

    def test_verify_release_checks_checksum(self):
        content = _read("scripts/verify_release.sh")
        assert "sha256sum" in content

    def test_verify_release_has_strict_mode(self):
        content = _read("scripts/verify_release.sh")
        assert "--strict" in content

    def test_verify_release_exit_codes_documented(self):
        content = _read("scripts/verify_release.sh")
        assert "Exit codes" in content or "exit" in content.lower()

    def test_sign_references_security_md(self):
        content = _read("scripts/sign_release.sh")
        assert "SECURITY.md" in content

    def test_verify_references_security_md(self):
        content = _read("scripts/verify_release.sh")
        assert "SECURITY.md" in content


# ---------------------------------------------------------------------------
# Class 8: Cross-file consistency
# ---------------------------------------------------------------------------

class TestCrossFileConsistency:
    """Cross-references between files are consistent."""

    def test_readme_links_to_security_md(self):
        content = _read("README.md")
        assert "SECURITY.md" in content

    def test_security_md_links_to_docs(self):
        content = _read("SECURITY.md")
        assert "docs/security/overview.md" in content

    def test_changelog_entry_count_plausible(self):
        content = _read("CHANGELOG.md")
        v330_entry = content.split("## v3.3.0")[1].split("## v3.2.11")[0]
        # Should have substantial content (at least 50 lines)
        lines = [l for l in v330_entry.strip().split("\n") if l.strip()]
        assert len(lines) >= 50, f"Only {len(lines)} non-empty lines"

    def test_security_md_references_audit_report(self):
        content = _read("SECURITY.md")
        assert "SECURITY_AUDIT_S155" in content

    def test_readme_and_docs_endpoint_count_match(self):
        readme = _read("README.md")
        api_ref = _read("docs/api-reference.md")
        assert "519" in readme and "519" in api_ref


# ---------------------------------------------------------------------------
# Class 9: pyproject.toml
# ---------------------------------------------------------------------------

class TestPyprojectToml:
    """pyproject.toml packaging configuration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read("pyproject.toml")

    def test_file_exists(self):
        assert os.path.isfile(os.path.join(ROOT, "pyproject.toml"))

    def test_version_matches(self):
        assert 'version = "3.3.0"' in self.content

    def test_version_consistent_with_version_py(self):
        v = _load_version()
        assert f'version = "{v}"' in self.content

    def test_name(self):
        assert 'name = "opti-oignon"' in self.content

    def test_requires_python(self):
        assert 'requires-python = ">=3.10"' in self.content

    def test_build_backend(self):
        assert 'build-backend = "setuptools.build_meta"' in self.content

    def test_license(self):
        assert 'license = "MIT"' in self.content

    def test_core_dep_fastapi(self):
        assert '"fastapi>=' in self.content

    def test_core_dep_ollama(self):
        assert '"ollama>=' in self.content

    def test_core_dep_chromadb(self):
        assert '"chromadb>=' in self.content

    def test_core_dep_bcrypt(self):
        assert '"bcrypt>=' in self.content

    def test_core_dep_cryptography(self):
        assert '"cryptography>=' in self.content

    def test_optional_llama(self):
        assert "llama-cpp-python" in self.content

    def test_optional_auth(self):
        assert "fido2" in self.content
        assert "pyotp" in self.content
        assert "qrcode" in self.content

    def test_optional_sqlcipher(self):
        assert "pysqlcipher3" in self.content

    def test_optional_dev(self):
        assert "pytest" in self.content
        assert "ruff" in self.content
        assert "mypy" in self.content

    def test_optional_docs(self):
        assert "mkdocs" in self.content
        assert "mkdocs-material" in self.content

    def test_optional_all_group(self):
        assert '"opti-oignon[auth,sqlcipher,llama,dev,docs]"' in self.content

    def test_cli_entry_point(self):
        assert 'oo = "opti_oignon.cli.main:cli"' in self.content

    def test_package_find_includes_opti_oignon(self):
        assert '"opti_oignon*"' in self.content

    def test_package_find_excludes_tests(self):
        assert '"tests*"' in self.content

    def test_pytest_config(self):
        assert "[tool.pytest.ini_options]" in self.content

    def test_ruff_config(self):
        assert "[tool.ruff]" in self.content

    def test_mypy_config(self):
        assert "[tool.mypy]" in self.content

    def test_coverage_config(self):
        assert "[tool.coverage.run]" in self.content

    def test_no_dynamic_version(self):
        # Version is hardcoded to avoid import chain issues
        assert "dynamic" not in self.content.split("[project]")[1].split("[")[0]

