#!/usr/bin/env python3
"""
Tests for S161 -- MkDocs Documentation Site.

Validates mkdocs.yml structure, doc file existence, internal links,
build script, and documentation content quality.
"""

import os
import re
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MKDOCS_YML = PROJECT_ROOT / "mkdocs.yml"
DOCS_DIR = PROJECT_ROOT / "docs"
REQUIREMENTS_DOCS = PROJECT_ROOT / "requirements-docs.txt"
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build_docs.sh"
EXTRA_CSS = DOCS_DIR / "stylesheets" / "extra.css"
INDEX_MD = DOCS_DIR / "index.md"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mkdocs_loader():
    """Create a YAML loader that handles !!python/name tags safely.

    MkDocs uses !!python/name: for superfences custom_fences format
    callbacks. yaml.safe_load rejects these tags, so we add a
    constructor that returns the tag string as a placeholder.
    """
    loader = yaml.SafeLoader
    # Build a subclass to avoid mutating the global SafeLoader
    custom = type("MkDocsLoader", (loader,), {})
    custom.add_multi_constructor(
        "tag:yaml.org,2002:python/name:",
        lambda loader, suffix, node: f"python:{suffix}",
    )
    return custom


@pytest.fixture(scope="module")
def mkdocs_config():
    """Load and parse mkdocs.yml."""
    with open(MKDOCS_YML, encoding="utf-8") as f:
        return yaml.load(f, Loader=_make_mkdocs_loader())


@pytest.fixture(scope="module")
def nav_files(mkdocs_config):
    """Extract all .md file paths from the nav structure."""
    files = []

    def _walk_nav(items):
        for item in items:
            if isinstance(item, str):
                files.append(item)
            elif isinstance(item, dict):
                for value in item.values():
                    if isinstance(value, str):
                        files.append(value)
                    elif isinstance(value, list):
                        _walk_nav(value)

    _walk_nav(mkdocs_config.get("nav", []))
    return files


@pytest.fixture(scope="module")
def all_doc_files():
    """Collect all markdown files under docs/."""
    return list(DOCS_DIR.rglob("*.md"))


# ---------------------------------------------------------------------------
# 1. mkdocs.yml validity
# ---------------------------------------------------------------------------

class TestMkdocsYml:
    """Tests for mkdocs.yml configuration."""

    def test_mkdocs_yml_exists(self):
        assert MKDOCS_YML.is_file(), "mkdocs.yml must exist at project root"

    def test_mkdocs_yml_valid_yaml(self):
        with open(MKDOCS_YML, encoding="utf-8") as f:
            config = yaml.load(f, Loader=_make_mkdocs_loader())
        assert isinstance(config, dict), "mkdocs.yml must parse as a dict"

    def test_site_name_present(self, mkdocs_config):
        assert "site_name" in mkdocs_config
        assert "Opti-Oignon" in mkdocs_config["site_name"]

    def test_theme_is_material(self, mkdocs_config):
        theme = mkdocs_config.get("theme", {})
        assert theme.get("name") == "material"

    def test_theme_has_palette(self, mkdocs_config):
        theme = mkdocs_config.get("theme", {})
        palette = theme.get("palette")
        assert isinstance(palette, list), "palette must be a list (light/dark)"
        assert len(palette) >= 2, "need at least light and dark schemes"

    def test_dark_light_toggle(self, mkdocs_config):
        theme = mkdocs_config.get("theme", {})
        palette = theme.get("palette", [])
        has_toggle = all("toggle" in p for p in palette)
        assert has_toggle, "each palette entry must have a toggle"

    def test_search_plugin_enabled(self, mkdocs_config):
        plugins = mkdocs_config.get("plugins", [])
        plugin_names = []
        for p in plugins:
            if isinstance(p, str):
                plugin_names.append(p)
            elif isinstance(p, dict):
                plugin_names.extend(p.keys())
        assert "search" in plugin_names, "search plugin must be enabled"

    def test_nav_present(self, mkdocs_config):
        nav = mkdocs_config.get("nav")
        assert isinstance(nav, list), "nav must be a list"
        assert len(nav) >= 5, "nav must have at least 5 top-level entries"

    def test_extra_css_referenced(self, mkdocs_config):
        extra_css = mkdocs_config.get("extra_css", [])
        assert any("extra.css" in c for c in extra_css), (
            "extra.css must be referenced in extra_css"
        )

    def test_markdown_extensions_present(self, mkdocs_config):
        extensions = mkdocs_config.get("markdown_extensions", [])
        ext_names = []
        for e in extensions:
            if isinstance(e, str):
                ext_names.append(e)
            elif isinstance(e, dict):
                ext_names.extend(e.keys())
        assert "admonition" in ext_names
        assert "tables" in ext_names

    def test_mermaid_support(self, mkdocs_config):
        extensions = mkdocs_config.get("markdown_extensions", [])
        has_superfences = False
        for e in extensions:
            if isinstance(e, dict) and "pymdownx.superfences" in e:
                has_superfences = True
            elif e == "pymdownx.superfences":
                has_superfences = True
        assert has_superfences, "pymdownx.superfences required for mermaid"

    def test_code_copy_feature(self, mkdocs_config):
        theme = mkdocs_config.get("theme", {})
        features = theme.get("features", [])
        assert "content.code.copy" in features


# ---------------------------------------------------------------------------
# 2. Nav-referenced files exist
# ---------------------------------------------------------------------------

class TestNavFiles:
    """All files referenced in the nav must exist."""

    def test_all_nav_files_exist(self, nav_files):
        missing = [f for f in nav_files if not (DOCS_DIR / f).is_file()]
        assert not missing, f"Missing nav files: {missing}"

    def test_nav_has_index(self, nav_files):
        assert "index.md" in nav_files

    def test_nav_has_getting_started(self, nav_files):
        gs = [f for f in nav_files if f.startswith("getting-started/")]
        assert len(gs) >= 3

    def test_nav_has_security(self, nav_files):
        sec = [f for f in nav_files if f.startswith("security/")]
        assert len(sec) >= 4

    def test_nav_has_architecture(self, nav_files):
        arch = [f for f in nav_files if f.startswith("architecture/")]
        assert len(arch) >= 3

    def test_nav_has_redteam(self, nav_files):
        rt = [f for f in nav_files if f.startswith("redteam/")]
        assert len(rt) >= 2

    def test_nav_has_user_guide(self, nav_files):
        ug = [f for f in nav_files if f.startswith("user-guide/")]
        assert len(ug) >= 4

    def test_nav_file_count(self, nav_files):
        assert len(nav_files) >= 20, f"Expected 20+ nav pages, got {len(nav_files)}"


# ---------------------------------------------------------------------------
# 3. Build script
# ---------------------------------------------------------------------------

class TestBuildScript:
    """Tests for scripts/build_docs.sh."""

    def test_build_script_exists(self):
        assert BUILD_SCRIPT.is_file()

    def test_build_script_executable(self):
        mode = BUILD_SCRIPT.stat().st_mode
        assert mode & stat.S_IXUSR, "build_docs.sh must be executable"

    def test_build_script_has_shebang(self):
        with open(BUILD_SCRIPT, encoding="utf-8") as f:
            first_line = f.readline().strip()
        assert first_line.startswith("#!/"), "must have a shebang line"
        assert "bash" in first_line

    def test_build_script_uses_set_e(self):
        content = BUILD_SCRIPT.read_text(encoding="utf-8")
        assert "set -e" in content or "set -eu" in content or "set -euo" in content

    def test_build_script_valid_bash(self):
        result = subprocess.run(
            ["bash", "-n", str(BUILD_SCRIPT)],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_build_script_references_mkdocs_yml(self):
        content = BUILD_SCRIPT.read_text(encoding="utf-8")
        assert "mkdocs.yml" in content or "MKDOCS_YML" in content

    def test_build_script_has_check_mode(self):
        content = BUILD_SCRIPT.read_text(encoding="utf-8")
        assert "check" in content, "build script should support check mode"

    def test_build_script_has_serve_mode(self):
        content = BUILD_SCRIPT.read_text(encoding="utf-8")
        assert "serve" in content, "build script should support serve mode"


# ---------------------------------------------------------------------------
# 4. Internal links
# ---------------------------------------------------------------------------

class TestInternalLinks:
    """No broken internal links in markdown files."""

    LINK_PATTERN = re.compile(r'\[(?:[^\]]*)\]\(([^)#][^)]*\.md(?:#[^)]*)?)\)')

    def test_no_broken_internal_links(self, all_doc_files):
        broken = []
        for mdfile in all_doc_files:
            content = mdfile.read_text(encoding="utf-8")
            for match in self.LINK_PATTERN.finditer(content):
                target = match.group(1).split("#")[0]
                if target.startswith("http"):
                    continue
                resolved = (mdfile.parent / target).resolve()
                if not resolved.is_file():
                    broken.append(f"{mdfile.relative_to(PROJECT_ROOT)} -> {target}")
        assert not broken, "Broken links:\n" + "\n".join(broken)


# ---------------------------------------------------------------------------
# 5. Requirements file
# ---------------------------------------------------------------------------

class TestRequirementsDocs:
    """Tests for requirements-docs.txt."""

    def test_requirements_file_exists(self):
        assert REQUIREMENTS_DOCS.is_file()

    def test_requires_mkdocs(self):
        content = REQUIREMENTS_DOCS.read_text(encoding="utf-8")
        assert "mkdocs" in content.lower()

    def test_requires_material(self):
        content = REQUIREMENTS_DOCS.read_text(encoding="utf-8")
        assert "mkdocs-material" in content.lower()

    def test_no_empty_lines_only(self):
        lines = [
            l.strip() for l in REQUIREMENTS_DOCS.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        assert len(lines) >= 2, "need at least mkdocs and mkdocs-material"


# ---------------------------------------------------------------------------
# 6. Custom CSS
# ---------------------------------------------------------------------------

class TestCustomCSS:
    """Tests for docs/stylesheets/extra.css."""

    def test_extra_css_exists(self):
        assert EXTRA_CSS.is_file()

    def test_uses_oo_variables(self):
        content = EXTRA_CSS.read_text(encoding="utf-8")
        assert "--oo-" in content, "must use --oo-* CSS variables"

    def test_maps_to_material_variables(self):
        content = EXTRA_CSS.read_text(encoding="utf-8")
        assert "--md-primary-fg-color" in content or "--md-accent-fg-color" in content

    def test_has_dark_scheme(self):
        content = EXTRA_CSS.read_text(encoding="utf-8")
        assert "slate" in content, "must define dark (slate) scheme styles"

    def test_no_hardcoded_hex_in_selectors(self):
        """CSS variable definitions may have hex values, but selectors should use vars."""
        content = EXTRA_CSS.read_text(encoding="utf-8")
        # Find property declarations that use raw hex outside of variable definitions
        lines = content.splitlines()
        violations = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip variable definitions (--oo-* or --md-*) and comments
            if stripped.startswith("--") or stripped.startswith("/*") or stripped.startswith("*"):
                continue
            # Check for hex color values in property assignments
            if re.search(r':\s*#[0-9a-fA-F]{3,8}\s*;', stripped):
                violations.append(f"Line {i}: {stripped}")
        assert not violations, "Hardcoded hex in CSS properties:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# 7. Content quality
# ---------------------------------------------------------------------------

class TestContentQuality:
    """Basic quality checks on documentation content."""

    def test_index_has_mermaid(self):
        content = INDEX_MD.read_text(encoding="utf-8")
        assert "```mermaid" in content, "index.md should have architecture diagram"

    def test_index_has_quick_start(self):
        content = INDEX_MD.read_text(encoding="utf-8")
        assert "quick start" in content.lower() or "Quick start" in content

    def test_all_docs_have_h1(self, all_doc_files):
        missing_h1 = []
        for mdfile in all_doc_files:
            # Skip legacy files not in nav
            relpath = str(mdfile.relative_to(DOCS_DIR))
            if relpath.startswith(("COLOR_", "ROADMAP_", "SECURITY_AUDIT", "demo_")):
                continue
            if relpath in ("API_REFERENCE.md", "PLUGIN_DEVELOPMENT_GUIDE.md"):
                continue
            content = mdfile.read_text(encoding="utf-8")
            if not re.search(r'^# ', content, re.MULTILINE):
                missing_h1.append(relpath)
        assert not missing_h1, f"Missing H1 heading: {missing_h1}"

    def test_no_french_in_nav_docs(self, all_doc_files):
        """Nav docs should be in English."""
        french_markers = ["utilisateur", "securite", "connexion", "parametres"]
        violations = []
        for mdfile in all_doc_files:
            relpath = str(mdfile.relative_to(DOCS_DIR))
            # Skip legacy files
            if relpath.startswith(("COLOR_", "ROADMAP_", "SECURITY_AUDIT", "demo_")):
                continue
            if relpath in ("API_REFERENCE.md", "PLUGIN_DEVELOPMENT_GUIDE.md"):
                continue
            content = mdfile.read_text(encoding="utf-8").lower()
            for marker in french_markers:
                if marker in content:
                    violations.append(f"{relpath}: contains '{marker}'")
        assert not violations, "French text found:\n" + "\n".join(violations)

    def test_no_emojis_in_docs(self, all_doc_files):
        """No emojis in documentation markdown."""
        emoji_pattern = re.compile(
            "[\U0001F300-\U0001F9FF\U0001FA00-\U0001FA6F\U00002702-\U000027B0]"
        )
        violations = []
        for mdfile in all_doc_files:
            relpath = str(mdfile.relative_to(DOCS_DIR))
            if relpath.startswith(("COLOR_", "ROADMAP_", "SECURITY_AUDIT", "demo_")):
                continue
            content = mdfile.read_text(encoding="utf-8")
            if emoji_pattern.search(content):
                violations.append(relpath)
        assert not violations, f"Emojis found in: {violations}"

    def test_security_docs_mention_bulbe(self):
        overview = (DOCS_DIR / "security" / "overview.md").read_text(encoding="utf-8")
        assert "Bulbe" in overview

    def test_cli_docs_mention_all_commands(self):
        cli_ref = (DOCS_DIR / "cli-reference.md").read_text(encoding="utf-8")
        for cmd in ["ask", "models", "status", "backup", "rag", "redteam", "config"]:
            assert cmd in cli_ref, f"CLI reference missing command: {cmd}"

    def test_architecture_has_mermaid_diagrams(self):
        for name in ("data-flow.md", "security-layers.md"):
            content = (DOCS_DIR / "architecture" / name).read_text(encoding="utf-8")
            assert "```mermaid" in content, f"{name} should have mermaid diagram"

    def test_contributing_mentions_importlib_pattern(self):
        content = (DOCS_DIR / "contributing.md").read_text(encoding="utf-8")
        assert "importlib" in content

    def test_plugin_dev_mentions_subprocess_isolation(self):
        content = (DOCS_DIR / "plugin-development.md").read_text(encoding="utf-8")
        assert "subprocess" in content.lower() or "sandbox" in content.lower()


# ---------------------------------------------------------------------------
# 8. Directory structure
# ---------------------------------------------------------------------------

class TestDirectoryStructure:
    """Verify expected directory layout."""

    @pytest.mark.parametrize("subdir", [
        "getting-started",
        "user-guide",
        "security",
        "redteam",
        "architecture",
        "stylesheets",
        "assets",
        "overrides",
    ])
    def test_docs_subdirectory_exists(self, subdir):
        assert (DOCS_DIR / subdir).is_dir(), f"docs/{subdir}/ must exist"

    def test_logo_in_docs_assets(self):
        assert (DOCS_DIR / "assets" / "opti-oignon.svg").is_file()
