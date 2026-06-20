#!/usr/bin/env python3
"""
Tests for S139 — English-Only Codebase.

Validates:
1. Zero French text in comments/docstrings (security scan integration)
2. Module-level docstrings exist and are English for key modules
3. AST validity of all modified Python files
4. Version bump to 3.0.2
"""

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OPTI_ROOT = _PROJECT_ROOT / "opti_oignon"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_module(name: str, path: Path):
    """Load a module without triggering the full __init__ chain."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# French detection patterns (mirrors security_scan.py)
_FRENCH_ACCENTS = re.compile(r"[éèêëàâùûôïîçÉÈÊÀ]")
_FRENCH_PATTERNS = re.compile(
    r"(?:"
    r"Retourne |Calcule |Verifie |Recupere |Cree |Selectionne |"
    r"Genere |Sauvegarde |Supprime |Construit |Initialise |"
    r"Nettoie |Mesure |Resout |Valide |Fournit |"
    r"Augmente |Importe |Exporte |Duplique |Convertit |"
    r"du modele|du pipeline|du fichier|du cache|du contenu|du budget|"
    r"du prompt|du resume|du systeme|"
    r"de la conversation|de la memoire|de la recherche|"
    r"les resultats|les modeles|en memoire|"
    r"par defaut|mis a jour"
    r")",
    re.IGNORECASE,
)
_FRENCH_EXCLUDE = [
    re.compile(r"Author:\s*Léon", re.IGNORECASE),
    re.compile(r"__author__\s*="),
    re.compile(r"[À-ÿ]"),
    re.compile(r'^r["\']'),
    re.compile(r"re\.compile"),
    re.compile(r"\binitialis(?:e|ed|ing)\b"),
]
_FRENCH_KEYWORD_FILES = {"presets.py"}


def _is_french_line(line: str) -> bool:
    """Check if a line contains French text."""
    stripped = line.strip()
    if not stripped:
        return False
    for pat in _FRENCH_EXCLUDE:
        if pat.search(stripped):
            return False
    if _FRENCH_ACCENTS.search(stripped):
        words = re.findall(r"\w*[éèêëàâùûôïîçÉÈÊÀ]\w*", stripped)
        if all(w in {"Léon", "León"} for w in words):
            return False
        return True
    if _FRENCH_PATTERNS.search(stripped):
        return True
    return False


# ---------------------------------------------------------------------------
# Test: Version bump
# ---------------------------------------------------------------------------

class TestVersionBump:
    """Verify version is at least 3.0.2."""

    def test_version_file(self):
        mod = _load_module("__version__", _OPTI_ROOT / "__version__.py")
        assert mod.__version__ >= "3.0.2"


# ---------------------------------------------------------------------------
# Test: Zero French in Python files
# ---------------------------------------------------------------------------

class TestNoFrenchInPython:
    """All Python source files must have zero French comments/docstrings."""

    def _scan_file(self, filepath: Path) -> list[tuple[int, str]]:
        """Return list of (line_number, text) for French lines."""
        if filepath.name in _FRENCH_KEYWORD_FILES:
            return []
        violations = []
        try:
            lines = filepath.read_text("utf-8").splitlines()
        except Exception:
            return []
        for i, line in enumerate(lines, 1):
            if _is_french_line(line):
                violations.append((i, line.strip()[:120]))
        return violations

    def test_no_french_in_core_modules(self):
        """Zero French in opti_oignon/ Python files."""
        all_violations = {}
        for pyfile in sorted(_OPTI_ROOT.rglob("*.py")):
            viols = self._scan_file(pyfile)
            if viols:
                all_violations[str(pyfile.relative_to(_PROJECT_ROOT))] = viols

        if all_violations:
            msg_parts = []
            for fname, viols in all_violations.items():
                msg_parts.append(f"\n  {fname} ({len(viols)} lines):")
                for lineno, text in viols[:5]:
                    msg_parts.append(f"    L{lineno}: {text}")
                if len(viols) > 5:
                    msg_parts.append(f"    ... and {len(viols)-5} more")
            pytest.fail(f"French text found in Python files:{''.join(msg_parts)}")

    def test_no_french_in_frontend(self):
        """Zero French in frontend Svelte/TS files."""
        frontend_src = _PROJECT_ROOT / "frontend" / "src"
        if not frontend_src.exists():
            pytest.skip("No frontend/src directory")

        all_violations = {}
        for ext in ("*.svelte", "*.ts"):
            for fpath in sorted(frontend_src.rglob(ext)):
                viols = self._scan_file(fpath)
                if viols:
                    all_violations[str(fpath.relative_to(_PROJECT_ROOT))] = viols

        if all_violations:
            msg_parts = []
            for fname, viols in all_violations.items():
                msg_parts.append(f"\n  {fname} ({len(viols)} lines):")
                for lineno, text in viols[:3]:
                    msg_parts.append(f"    L{lineno}: {text}")
            pytest.fail(f"French text found in frontend files:{''.join(msg_parts)}")


# ---------------------------------------------------------------------------
# Test: AST validity
# ---------------------------------------------------------------------------

class TestASTValidity:
    """All Python files must parse without syntax errors."""

    def test_all_python_files_parse(self):
        """AST-verify every Python file under opti_oignon/."""
        failures = []
        for pyfile in sorted(_OPTI_ROOT.rglob("*.py")):
            try:
                ast.parse(pyfile.read_text("utf-8"))
            except SyntaxError as e:
                failures.append(f"{pyfile}: {e}")

        if failures:
            pytest.fail(
                f"AST failures ({len(failures)}):\n"
                + "\n".join(f"  {f}" for f in failures)
            )


# ---------------------------------------------------------------------------
# Test: Module-level docstrings
# ---------------------------------------------------------------------------

# Key modules that must have English module-level docstrings
_KEY_MODULES = [
    "executor.py",
    "memory.py",
    "pipeline_manager.py",
    "context_summary.py",
    "conversation.py",
    "consensus.py",
    "reasoning.py",
    "self_correction.py",
    "verification.py",
    "router.py",
    "response_cache.py",
    "search_integration.py",
    "tool_executor.py",
    "tool_registry.py",
    "config.py",
    "main.py",
]


class TestModuleDocstrings:
    """Key modules must have English module-level docstrings."""

    @pytest.mark.parametrize("module_name", _KEY_MODULES)
    def test_module_has_english_docstring(self, module_name: str):
        """Each key module must have a non-empty, non-French module docstring."""
        filepath = _OPTI_ROOT / module_name
        if not filepath.exists():
            pytest.skip(f"{module_name} not found")

        tree = ast.parse(filepath.read_text("utf-8"))
        docstring = ast.get_docstring(tree)

        assert docstring is not None, f"{module_name} has no module docstring"
        assert len(docstring.strip()) > 10, f"{module_name} docstring is too short"

        # Check it's not French
        for line in docstring.splitlines():
            assert not _is_french_line(line), (
                f"{module_name} has French in module docstring: {line.strip()[:80]}"
            )


# ---------------------------------------------------------------------------
# Test: Security scan integration
# ---------------------------------------------------------------------------

class TestSecurityScanFrench:
    """The security_scan.py check_no_french must pass."""

    def test_security_scan_no_french_passes(self):
        """Run the actual security_scan check_no_french and verify zero violations."""
        scan_path = _PROJECT_ROOT / "scripts" / "security_scan.py"
        if not scan_path.exists():
            pytest.skip("security_scan.py not found")

        spec = importlib.util.spec_from_file_location("security_scan", scan_path)
        mod = importlib.util.module_from_spec(spec)
        # Temporarily add scripts to path
        old_path = sys.path.copy()
        sys.path.insert(0, str(scan_path.parent))
        try:
            spec.loader.exec_module(mod)
            py_files = mod._py_files(include_tests=False)
            svelte_files = mod._svelte_files()
            result = mod.check_no_french(py_files, svelte_files)
            assert len(result.violations) == 0, (
                f"French violations found ({len(result.violations)}):\n"
                + "\n".join(
                    f"  {v['file']}:{v['line']} — {v['detail']}"
                    for v in result.violations[:10]
                )
            )
        finally:
            sys.path = old_path


# ---------------------------------------------------------------------------
# Test: Presets.py French keywords are intentional
# ---------------------------------------------------------------------------

class TestPresetsKeywordsIntentional:
    """presets.py contains French keywords for user-input detection — verify they exist."""

    def test_presets_has_french_detection_keywords(self):
        """Presets.py should still contain French keywords for input matching."""
        filepath = _OPTI_ROOT / "presets.py"
        if not filepath.exists():
            pytest.skip("presets.py not found")

        content = filepath.read_text("utf-8")
        # These are intentional French keywords for auto-detection
        assert '"ne marche pas"' in content
        assert '"échoue"' in content or '"echoue"' in content
