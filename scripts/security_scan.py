#!/usr/bin/env python3
"""
security_scan.py -- Static Security Analysis for Opti-Oignon (S130, extended S155).

Pure static analysis: scans source files for common security anti-patterns
WITHOUT importing any opti_oignon modules.

Checks:
  1. No raw sqlite3.connect() -- must use get_encrypted_connection()
  2. No eval()/exec() outside sandbox
  3. No hardcoded secrets (passwords, API keys, tokens)
  4. No pickle.loads() on untrusted data
  5. All SQL parameterized -- no f-string in .execute()
  5b. No f-string SQL literals (S138)
  6. No shell=True in subprocess calls
  7. CSRF protection present on state-changing routes
  8. No hardcoded hex colors in Svelte (extends audit_colors.py)
  9. checkpoint_before_apply = True always hardcoded
  10. No French in code comments or UI text
  11. No unsafe yaml.load() without SafeLoader (S155)
  12. Path traversal risk detection (S155)
  13. SSRF vector detection (S155)
  14. Rate limiting on sensitive endpoints (S155)
  15. Cookie security flags (httponly, secure, samesite) (S155)
  16. Insecure random in security contexts (S155)
  17. Frontend hardcoded secrets in JS/TS/Svelte (S155)

Usage:
    python scripts/security_scan.py [--json] [--verbose]

Exit code 0 if all pass, 1 if any fail.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_BACKEND_DIR = _PROJECT_ROOT / "opti_oignon"
_FRONTEND_DIR = _PROJECT_ROOT / "frontend" / "src"
_TEST_DIR = _PROJECT_ROOT / "tests"


def _py_files(include_tests: bool = False) -> list[Path]:
    """Collect all Python source files."""
    files = list(_BACKEND_DIR.rglob("*.py"))
    if include_tests:
        files.extend(_TEST_DIR.rglob("*.py"))
    return [f for f in files if "__pycache__" not in str(f)]


def _svelte_files() -> list[Path]:
    """Collect all Svelte component files."""
    if not _FRONTEND_DIR.exists():
        return []
    return [f for f in _FRONTEND_DIR.rglob("*.svelte") if "node_modules" not in str(f)]


def _ts_files() -> list[Path]:
    """Collect all TypeScript files."""
    if not _FRONTEND_DIR.exists():
        return []
    return [f for f in _FRONTEND_DIR.rglob("*.ts") if "node_modules" not in str(f)]


def _read_lines(path: Path) -> list[str]:
    """Read file lines, returning empty list on error."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Check result
# ---------------------------------------------------------------------------

class CheckResult:
    """Result of a single security check."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self.passed = True
        self.violations: list[dict[str, Any]] = []

    def add_violation(self, file: str, line: int, detail: str) -> None:
        self.passed = False
        self.violations.append({
            "file": file,
            "line": line,
            "detail": detail,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "passed": self.passed,
            "violation_count": len(self.violations),
            "violations": self.violations,
        }


# ---------------------------------------------------------------------------
# Check 1: No raw sqlite3.connect()
# ---------------------------------------------------------------------------

_SQLITE_CONNECT_RE = re.compile(r"\bsqlite3\.connect\s*\(")

# S138: Only db_utils.py (the safe_connect wrapper) and db_encryption.py
# (the encryption layer itself) may call sqlite3.connect directly.
# Plugin entry_points are excluded because they run in sandboxes.
_SQLITE_ALLOWED = {
    "db_utils.py",
    "db_encryption.py",
}


def check_no_raw_sqlite(files: list[Path]) -> CheckResult:
    """Ensure no direct sqlite3.connect() outside allowed modules.

    S138 hardening: after migrating all core modules to safe_connect(),
    only the connection infrastructure (db_utils, db_encryption) may
    call sqlite3.connect directly.  Plugin entry_points under plugins/
    are excluded (sandboxed, no host DB access).
    """
    result = CheckResult(
        "no_raw_sqlite",
        "No sqlite3.connect() outside db_utils/db_encryption (use safe_connect)",
    )
    for fpath in files:
        if fpath.name in _SQLITE_ALLOWED:
            continue
        # Exclude plugin entry_points (sandboxed)
        if "plugins" + os.sep in str(fpath) or "/plugins/" in str(fpath):
            continue
        lines = _read_lines(fpath)
        for i, line in enumerate(lines, 1):
            if _SQLITE_CONNECT_RE.search(line):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                result.add_violation(
                    str(fpath.relative_to(_PROJECT_ROOT)), i,
                    f"Direct sqlite3.connect() call: {stripped[:80]}",
                )
    return result


# ---------------------------------------------------------------------------
# Check 2: No eval()/exec() outside sandbox
# ---------------------------------------------------------------------------

_EVAL_EXEC_RE = re.compile(r"\b(eval|exec)\s*\(")

# Files where eval/exec is allowed (sandbox, test utilities)
_EVAL_ALLOWED_PATTERNS = {
    "sandbox",
    "plugin_sandbox",
    "test_",
}


def check_no_eval_exec(files: list[Path]) -> CheckResult:
    """No eval()/exec() outside plugin sandbox."""
    result = CheckResult(
        "no_eval_exec",
        "No eval()/exec() outside plugin sandbox",
    )
    for fpath in files:
        fname = fpath.name
        if any(pat in fname for pat in _EVAL_ALLOWED_PATTERNS):
            continue
        for i, line in enumerate(_read_lines(fpath), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith(("\"\"\"", "'''")):
                continue
            m = _EVAL_EXEC_RE.search(line)
            if m:
                # Ignore string content (rough heuristic: check if inside quotes)
                before = line[:m.start()]
                # Skip if it looks like a string/docstring context
                if before.count('"') % 2 == 1 or before.count("'") % 2 == 1:
                    continue
                result.add_violation(
                    str(fpath.relative_to(_PROJECT_ROOT)), i,
                    f"Dangerous {m.group(1)}() call: {stripped[:80]}",
                )
    return result


# ---------------------------------------------------------------------------
# Check 3: No hardcoded secrets
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"""(?:password|passwd|pwd)\s*=\s*["'][^"']{4,}["']""", re.I),
    re.compile(r"""(?:api_key|apikey|api_secret)\s*=\s*["'][^"']{8,}["']""", re.I),
    re.compile(r"""(?:secret_key|secret)\s*=\s*["'][^"']{8,}["']""", re.I),
    re.compile(r"""(?:token)\s*=\s*["'][A-Za-z0-9_\-]{20,}["']""", re.I),
    re.compile(r"""(?:private_key)\s*=\s*["'][^"']{8,}["']""", re.I),
]

# Common false-positive patterns to exclude
_SECRET_FALSE_POSITIVES = {
    "password_hash",
    "password_field",
    "password_input",
    "password_min",
    "current_password",
    "new_password",
    "confirm_password",
    "password_strength",
    "secret_key_file",
    "min_length",
    '""',
    "''",
    "test_",
    "placeholder",
    "example",
    "PLACEHOLDER",
    "changeme",
    "your_",
    "xxx",
}


def check_no_hardcoded_secrets(files: list[Path]) -> CheckResult:
    """No hardcoded passwords, API keys, or tokens."""
    result = CheckResult(
        "no_hardcoded_secrets",
        "No hardcoded secrets (passwords, API keys, tokens)",
    )
    for fpath in files:
        for i, line in enumerate(_read_lines(fpath), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pat in _SECRET_PATTERNS:
                m = pat.search(line)
                if m:
                    matched_text = m.group(0).lower()
                    # Check false positives
                    if any(fp in matched_text for fp in _SECRET_FALSE_POSITIVES):
                        continue
                    # Skip type annotations and parameter defaults
                    if "str =" in line and ('""' in line or "''" in line):
                        continue
                    # Skip Field() definitions with description
                    if "Field(" in line:
                        continue
                    result.add_violation(
                        str(fpath.relative_to(_PROJECT_ROOT)), i,
                        f"Possible hardcoded secret: {stripped[:80]}",
                    )
    return result


# ---------------------------------------------------------------------------
# Check 4: No pickle.loads() on untrusted data
# ---------------------------------------------------------------------------

_PICKLE_RE = re.compile(r"\bpickle\.(loads?|Unpickler)\s*\(")


def check_no_pickle(files: list[Path]) -> CheckResult:
    """No pickle.loads() on potentially untrusted data."""
    result = CheckResult(
        "no_pickle_loads",
        "No pickle.loads() on untrusted data",
    )
    for fpath in files:
        for i, line in enumerate(_read_lines(fpath), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if _PICKLE_RE.search(line):
                result.add_violation(
                    str(fpath.relative_to(_PROJECT_ROOT)), i,
                    f"Dangerous pickle usage: {stripped[:80]}",
                )
    return result


# ---------------------------------------------------------------------------
# Check 5: All SQL parameterized (no f-string in .execute())
# ---------------------------------------------------------------------------

_EXECUTE_FSTRING_RE = re.compile(r"""\.execute\s*\(\s*f["']""")
_EXECUTE_FORMAT_RE = re.compile(r"""\.execute\s*\([^)]*\.format\s*\(""")
_EXECUTE_PERCENT_RE = re.compile(r"""\.execute\s*\([^)]*%\s*[\(]""")


def check_sql_parameterized(files: list[Path]) -> CheckResult:
    """All SQL queries must use parameterized queries."""
    result = CheckResult(
        "sql_parameterized",
        "All SQL uses parameterized queries (no f-strings in .execute())",
    )
    for fpath in files:
        for i, line in enumerate(_read_lines(fpath), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for pat in (_EXECUTE_FSTRING_RE, _EXECUTE_FORMAT_RE, _EXECUTE_PERCENT_RE):
                if pat.search(line):
                    result.add_violation(
                        str(fpath.relative_to(_PROJECT_ROOT)), i,
                        f"Non-parameterized SQL: {stripped[:80]}",
                    )
                    break
    return result


# ---------------------------------------------------------------------------
# Check 5b: No f-string SQL literals (S138)
# ---------------------------------------------------------------------------

# Catches f"SELECT ...FROM" / f"INSERT INTO" / f"UPDATE ...SET" /
# f"DELETE FROM" / f"ALTER TABLE" / f"CREATE TABLE|INDEX" patterns.
# Requires SQL keyword + its typical SQL follow-up to avoid false positives
# on logger messages like f"Pipeline deleted: {id}".
_FSTRING_SQL_RE = re.compile(
    r"""\bf(["'])"""
    r"""(?=.*\b("""
    r"""SELECT\s+.+?\s+FROM"""
    r"""|INSERT\s+INTO"""
    r"""|UPDATE\s+\w+\s+SET"""
    r"""|DELETE\s+FROM"""
    r"""|ALTER\s+TABLE"""
    r"""|CREATE\s+(TABLE|INDEX)"""
    r""")\b)""",
    re.IGNORECASE,
)

# Files allowed to contain f-string SQL patterns (test files, security scan itself)
_FSTRING_SQL_ALLOWED_PREFIXES = ("test_", "security_scan")


def check_no_fstring_sql(files: list[Path]) -> CheckResult:
    """No f-string literals containing SQL keywords in production code.

    S138 hardening: SQL queries must use str.format() or concatenation
    with validated identifiers, never f-strings, to make injection
    patterns visually obvious during review.
    """
    result = CheckResult(
        "no_fstring_sql",
        "No f-string SQL literals in production code",
    )
    for fpath in files:
        if any(fpath.name.startswith(p) for p in _FSTRING_SQL_ALLOWED_PREFIXES):
            continue
        # Exclude plugins (sandboxed)
        if "plugins" + os.sep in str(fpath) or "/plugins/" in str(fpath):
            continue
        for i, line in enumerate(_read_lines(fpath), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if _FSTRING_SQL_RE.search(line):
                result.add_violation(
                    str(fpath.relative_to(_PROJECT_ROOT)), i,
                    f"f-string SQL literal: {stripped[:80]}",
                )
    return result


# ---------------------------------------------------------------------------
# Check 6: No shell=True in subprocess calls
# ---------------------------------------------------------------------------

_SHELL_TRUE_RE = re.compile(r"\bsubprocess\.\w+\s*\([^)]*shell\s*=\s*True")


def check_no_shell_true(files: list[Path]) -> CheckResult:
    """No shell=True in subprocess calls."""
    result = CheckResult(
        "no_shell_true",
        "No shell=True in subprocess.* calls",
    )
    for fpath in files:
        lines = _read_lines(fpath)
        full_text = "\n".join(lines)
        # Multi-line check via full text
        for m in _SHELL_TRUE_RE.finditer(full_text):
            # Find the line number
            line_num = full_text[:m.start()].count("\n") + 1
            context = lines[line_num - 1].strip() if line_num <= len(lines) else ""
            if context.startswith("#"):
                continue
            result.add_violation(
                str(fpath.relative_to(_PROJECT_ROOT)), line_num,
                f"subprocess with shell=True: {context[:80]}",
            )
    return result


# ---------------------------------------------------------------------------
# Check 7: CSRF on state-changing routes
# ---------------------------------------------------------------------------

_STATE_CHANGE_RE = re.compile(
    r"""@router\.(post|put|delete|patch)\s*\("""
)


def check_csrf_protection(files: list[Path]) -> CheckResult:
    """State-changing routes should have CSRF protection.

    S136 audit fix: checks for CSRFMiddleware (not SecurityModeMiddleware,
    which does mode enforcement but NOT CSRF validation).
    """
    result = CheckResult(
        "csrf_protection",
        "POST/PUT/DELETE routes have CSRF protection (CSRFMiddleware)",
    )

    # Check for global CSRF middleware in app.py
    app_py = _BACKEND_DIR / "api" / "app.py"
    if app_py.exists():
        app_text = app_py.read_text(encoding="utf-8", errors="replace")
        if "CSRFMiddleware" in app_text and "add_middleware" in app_text:
            # Global CSRF middleware covers all routes
            return result

    # No global CSRF middleware found — this is a violation
    result.add_violation(
        str(app_py.relative_to(_PROJECT_ROOT)) if app_py.exists() else "api/app.py",
        1,
        "CSRFMiddleware not registered globally in app.py. "
        "All POST/PUT/DELETE/PATCH endpoints lack CSRF protection.",
    )
    return result


# ---------------------------------------------------------------------------
# Check 8: No hardcoded hex colors in Svelte
# ---------------------------------------------------------------------------

_HEX_IN_STYLE_RE = re.compile(
    r"""#[0-9a-fA-F]{3,8}\b""",
)

# Allowed patterns: var(--oo-*, #fallback) and rgba(0,0,0,alpha)
_HEX_FALLBACK_RE = re.compile(
    r"""var\(--oo-[\w-]+,\s*#[0-9a-fA-F]{3,8}\)"""
)
# HTML entities like &#9733; &#123; &#10005;
_HTML_ENTITY_RE = re.compile(r"""&#[0-9a-fA-F]{1,8};""")
_RGBA_ZERO_RE = re.compile(r"""rgba?\(\s*0\s*,\s*0\s*,\s*0""")


def check_no_hardcoded_colors(svelte_files: list[Path]) -> CheckResult:
    """No hardcoded hex colors in Svelte -- must use --oo-* CSS variables."""
    result = CheckResult(
        "no_hardcoded_colors",
        "No hardcoded hex colors in Svelte (use --oo-* CSS variables)",
    )
    for fpath in svelte_files:
        for i, line in enumerate(_read_lines(fpath), 1):
            stripped = line.strip()
            # Skip HTML comments
            if stripped.startswith("<!--"):
                continue
            # Skip JS comments
            if stripped.startswith("//") or stripped.startswith("/*"):
                continue
            # Check for hex colors
            hex_matches = _HEX_IN_STYLE_RE.findall(line)
            if not hex_matches:
                continue
            # Filter out valid fallback patterns and HTML entities
            cleaned = _HEX_FALLBACK_RE.sub("", line)
            cleaned = _HTML_ENTITY_RE.sub("", cleaned)
            remaining_hex = _HEX_IN_STYLE_RE.findall(cleaned)
            if remaining_hex:
                for h in remaining_hex:
                    result.add_violation(
                        str(fpath.relative_to(_PROJECT_ROOT)), i,
                        f"Hardcoded color {h}: {stripped[:80]}",
                    )
    return result


# ---------------------------------------------------------------------------
# Check 9: checkpoint_before_apply = True hardcoded
# ---------------------------------------------------------------------------

_CHECKPOINT_RE = re.compile(r"checkpoint_before_apply\s*=")


def check_checkpoint_hardcoded(files: list[Path]) -> CheckResult:
    """checkpoint_before_apply must always be True and never overridable."""
    result = CheckResult(
        "checkpoint_before_apply",
        "checkpoint_before_apply = True always hardcoded, never overridable",
    )
    found_any = False
    for fpath in files:
        for i, line in enumerate(_read_lines(fpath), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _CHECKPOINT_RE.search(line):
                found_any = True
                # Must be = True (not False, not variable)
                if "= True" not in line and "=True" not in line:
                    result.add_violation(
                        str(fpath.relative_to(_PROJECT_ROOT)), i,
                        f"checkpoint_before_apply not hardcoded True: {stripped[:80]}",
                    )
    if not found_any:
        # Not necessarily a violation -- might be defined elsewhere
        pass
    return result


# ---------------------------------------------------------------------------
# Check 10: No French in code comments or UI text
# ---------------------------------------------------------------------------

# Accented characters that indicate French text (excludes proper nouns)
_FRENCH_ACCENTS = re.compile(r"[éèêëàâùûôïîçÉÈÊÀ]")

# Non-accented French verb/phrase patterns in comments and docstrings
_FRENCH_PATTERNS = re.compile(
    r"(?:"
    # French verb forms at start of docstring/comment
    r"Retourne |Calcule |Verifie |Recupere |Cree |Selectionne |"
    r"Genere |Sauvegarde |Supprime |Construit |Initialise |"
    r"Nettoie |Mesure |Resout |Valide |Fournit |"
    r"Augmente |Importe |Exporte |Duplique |Convertit |"
    # French article + noun patterns
    r"du modele|du pipeline|du fichier|du cache|du contenu|du budget|"
    r"du prompt|du resume|du systeme|"
    r"de la conversation|de la memoire|de la recherche|"
    r"les resultats|les modeles|en memoire|"
    r"par defaut|mis a jour"
    r")",
    re.IGNORECASE,
)

# Lines/patterns to exclude from French detection (legitimate uses)
_FRENCH_EXCLUDE_PATTERNS = [
    re.compile(r"Author:\s*Léon", re.IGNORECASE),
    re.compile(r"__author__\s*="),
    re.compile(r"[À-ÿ]"),       # Regex character class for accented chars
    re.compile(r'^r["\']'),      # Raw string regex patterns
    re.compile(r"re\.compile"),   # Regex definitions
    re.compile(r"\binitialis(?:e|ed|ing)\b"),  # British English spelling
]

# Files to skip for French check (docs, readme, prompts)
_FRENCH_SKIP = {".md", ".txt", ".rst"}

# Files with intentional French keywords (detection lists for user input)
_FRENCH_KEYWORD_FILES = {"presets.py"}


def _is_french_line(line: str) -> bool:
    """Check if a line contains French text (not false positives)."""
    stripped = line.strip()
    if not stripped:
        return False

    # Skip excluded patterns
    for pattern in _FRENCH_EXCLUDE_PATTERNS:
        if pattern.search(stripped):
            return False

    # Check for accented characters (strong signal)
    if _FRENCH_ACCENTS.search(stripped):
        # Extract accented words — skip if only in proper nouns
        words = re.findall(r"\w*[éèêëàâùûôïîçÉÈÊÀ]\w*", stripped)
        if all(w in {"Léon", "León"} for w in words):
            return False
        return True

    # Check non-accented French patterns
    if _FRENCH_PATTERNS.search(stripped):
        return True

    return False


def check_no_french(py_files: list[Path], svelte_files: list[Path]) -> CheckResult:
    """No French in code comments or UI text."""
    result = CheckResult(
        "no_french_in_code",
        "No French in code comments or UI text",
    )
    all_files = list(py_files) + list(svelte_files) + list(_ts_files())
    for fpath in all_files:
        if fpath.suffix in _FRENCH_SKIP:
            continue
        # Skip files with intentional French keyword lists
        if fpath.name in _FRENCH_KEYWORD_FILES:
            continue
        for i, line in enumerate(_read_lines(fpath), 1):
            if _is_french_line(line):
                stripped = line.strip()
                result.add_violation(
                    str(fpath.relative_to(_PROJECT_ROOT)), i,
                    f"French text: {stripped[:80]}",
                )
    return result


# ---------------------------------------------------------------------------
# Check 11: Unsafe YAML deserialization (S155)
# ---------------------------------------------------------------------------

_YAML_LOAD_RE = re.compile(r"\byaml\.load\s*\(")
_YAML_SAFE_RE = re.compile(r"Loader\s*=\s*(yaml\.)?(SafeLoader|CSafeLoader|FullLoader)")


def check_no_unsafe_yaml(files: list[Path]) -> CheckResult:
    """No yaml.load() without SafeLoader (use yaml.safe_load instead)."""
    result = CheckResult(
        "no_unsafe_yaml",
        "No yaml.load() without SafeLoader (use yaml.safe_load)",
    )
    for fpath in files:
        for i, line in enumerate(_read_lines(fpath), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if _YAML_LOAD_RE.search(line):
                # Check if SafeLoader / CSafeLoader is specified
                if _YAML_SAFE_RE.search(line):
                    continue
                # Check if next line has Loader= (multi-line call)
                lines = _read_lines(fpath)
                if i < len(lines) and _YAML_SAFE_RE.search(lines[i]):
                    continue
                result.add_violation(
                    str(fpath.relative_to(_PROJECT_ROOT)), i,
                    f"Unsafe yaml.load() without SafeLoader: {stripped[:80]}",
                )
    return result


# ---------------------------------------------------------------------------
# Check 12: Path traversal risks (S155)
# ---------------------------------------------------------------------------

# Patterns that suggest user input flowing into path operations
_PATH_TRAVERSAL_PATTERNS = [
    # os.path.join with request/query/body parameters
    re.compile(r"os\.path\.join\s*\([^)]*(?:request|user_input|filename|upload|param)"),
    # Path() with user input
    re.compile(r"Path\s*\([^)]*(?:request|user_input|filename|upload|param)"),
    # open() with f-string containing user-controlled variables
    re.compile(r"open\s*\(\s*f[\"'].*\{(?:filename|user_input|upload|file_path|name)\}"),
]

# Files where path operations with dynamic input are expected and validated
_PATH_TRAVERSAL_ALLOWED = {
    "sandbox",
    "file_utils",
    "safe_path",
    "test_",
}


def check_path_traversal(files: list[Path]) -> CheckResult:
    """Detect potential path traversal via user-controlled path input."""
    result = CheckResult(
        "path_traversal",
        "No unvalidated user input in path operations",
    )
    for fpath in files:
        fname = fpath.name
        if any(pat in fname for pat in _PATH_TRAVERSAL_ALLOWED):
            continue
        for i, line in enumerate(_read_lines(fpath), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for pat in _PATH_TRAVERSAL_PATTERNS:
                if pat.search(line):
                    result.add_violation(
                        str(fpath.relative_to(_PROJECT_ROOT)), i,
                        f"Potential path traversal: {stripped[:80]}",
                    )
                    break
    return result


# ---------------------------------------------------------------------------
# Check 13: SSRF vectors (S155)
# ---------------------------------------------------------------------------

# Patterns where user-controlled URLs are passed to HTTP libraries
_SSRF_PATTERNS = [
    re.compile(r"requests\.(get|post|put|delete|head|patch)\s*\(\s*(?:f[\"']|url|target|endpoint)"),
    re.compile(r"httpx\.(get|post|put|delete|head|patch|AsyncClient)\s*\(\s*(?:f[\"']|url|target)"),
    re.compile(r"urllib\.request\.urlopen\s*\("),
    re.compile(r"aiohttp\.ClientSession\s*\(\s*\)\s*\.\s*(get|post)\s*\(\s*(?:f[\"']|url)"),
]

# Modules where outbound HTTP is expected and validated
_SSRF_ALLOWED = {
    "kill_switch",
    "ollama",
    "web_search",
    "search_engine",
    "test_",
    "audit_deps",
    "backends",
}


def check_ssrf_vectors(files: list[Path]) -> CheckResult:
    """Detect potential SSRF via user-controlled URLs in HTTP calls."""
    result = CheckResult(
        "ssrf_vectors",
        "No user-controlled URLs in outbound HTTP requests",
    )
    for fpath in files:
        fname = fpath.stem
        if any(pat in fname for pat in _SSRF_ALLOWED):
            continue
        for i, line in enumerate(_read_lines(fpath), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for pat in _SSRF_PATTERNS:
                if pat.search(line):
                    result.add_violation(
                        str(fpath.relative_to(_PROJECT_ROOT)), i,
                        f"Potential SSRF vector: {stripped[:80]}",
                    )
                    break
    return result


# ---------------------------------------------------------------------------
# Check 14: Missing rate limiting on sensitive endpoints (S155)
# ---------------------------------------------------------------------------

# Route files that handle sensitive operations
_RATE_LIMIT_REQUIRED_ROUTES = {
    "routes_auth",
    "routes_users",
    "routes_files",
    "routes_security",
}


def check_rate_limiting(files: list[Path]) -> CheckResult:
    """Sensitive endpoints (auth, upload, admin) should have rate limiting."""
    result = CheckResult(
        "rate_limiting",
        "Rate limiting present on sensitive endpoints (auth, upload, admin)",
    )
    for fpath in files:
        fname = fpath.stem
        if fname not in _RATE_LIMIT_REQUIRED_ROUTES:
            continue
        text = fpath.read_text(encoding="utf-8", errors="replace")
        # Check for any rate limiting mechanism
        has_rate_limit = any(marker in text for marker in (
            "rate_limit",
            "RateLimiter",
            "RateLimit",
            "throttle",
            "Throttle",
            "slowapi",
            "limiter",
        ))
        if not has_rate_limit:
            result.add_violation(
                str(fpath.relative_to(_PROJECT_ROOT)), 1,
                f"No rate limiting found in sensitive route module: {fname}",
            )
    return result


# ---------------------------------------------------------------------------
# Check 15: Cookie security flags (S155)
# ---------------------------------------------------------------------------

_SET_COOKIE_RE = re.compile(r"set_cookie\s*\(|\.set_cookie\s*\(")


def check_cookie_security(files: list[Path]) -> CheckResult:
    """Cookies must have httponly, secure, and samesite flags."""
    result = CheckResult(
        "cookie_security",
        "Cookies set with httponly, secure, and samesite flags",
    )
    for fpath in files:
        lines = _read_lines(fpath)
        full_text = "\n".join(lines)
        # Find all set_cookie calls
        for m in _SET_COOKIE_RE.finditer(full_text):
            line_num = full_text[:m.start()].count("\n") + 1
            # Extract the full call (approximate: up to 500 chars or closing paren)
            call_region = full_text[m.start():m.start() + 500]
            paren_depth = 0
            call_end = len(call_region)
            for ci, ch in enumerate(call_region):
                if ch == "(":
                    paren_depth += 1
                elif ch == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        call_end = ci + 1
                        break
            call_text = call_region[:call_end]

            missing = []
            if "httponly" not in call_text.lower():
                missing.append("httponly")
            if "secure" not in call_text.lower():
                missing.append("secure")
            if "samesite" not in call_text.lower():
                missing.append("samesite")

            if missing:
                context = lines[line_num - 1].strip() if line_num <= len(lines) else ""
                result.add_violation(
                    str(fpath.relative_to(_PROJECT_ROOT)), line_num,
                    f"Cookie missing flags ({', '.join(missing)}): {context[:80]}",
                )
    return result


# ---------------------------------------------------------------------------
# Check 16: Insecure random usage (S155)
# ---------------------------------------------------------------------------

_INSECURE_RANDOM_RE = re.compile(
    r"\brandom\.(randint|random|choice|randrange|sample|uniform|shuffle)\s*\("
)

# Contexts where random module is acceptable (non-security)
_RANDOM_ALLOWED = {
    "test_",
    "demo",
    "benchmark",
    "fixture",
}


def check_insecure_random(files: list[Path]) -> CheckResult:
    """No random module for security-sensitive contexts (use secrets module)."""
    result = CheckResult(
        "insecure_random",
        "No insecure random in security-sensitive code (use secrets/os.urandom)",
    )
    # Only flag random usage in security-related modules
    security_modules = {
        "auth", "session", "token", "crypto", "security",
        "csrf", "nonce", "key", "secret", "password",
    }
    for fpath in files:
        fname = fpath.stem.lower()
        if any(pat in fname for pat in _RANDOM_ALLOWED):
            continue
        # Only flag in security-related files
        if not any(sec in fname for sec in security_modules):
            continue
        for i, line in enumerate(_read_lines(fpath), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if _INSECURE_RANDOM_RE.search(line):
                result.add_violation(
                    str(fpath.relative_to(_PROJECT_ROOT)), i,
                    f"Insecure random in security context: {stripped[:80]}",
                )
    return result


# ---------------------------------------------------------------------------
# Check 17: Frontend secrets in JS/TS/Svelte (S155)
# ---------------------------------------------------------------------------

_FRONTEND_SECRET_PATTERNS = [
    re.compile(r"""(?:api_key|apiKey|API_KEY)\s*[:=]\s*['"][^'"]{8,}['"]"""),
    re.compile(r"""(?:secret|SECRET)\s*[:=]\s*['"][^'"]{8,}['"]"""),
    re.compile(r"""(?:token|TOKEN)\s*[:=]\s*['"][A-Za-z0-9_\-]{20,}['"]"""),
    re.compile(r"""(?:password|PASSWORD)\s*[:=]\s*['"][^'"]{4,}['"]"""),
    re.compile(r"""Authorization['"]?\s*:\s*['"]Bearer\s+[A-Za-z0-9_\-\.]{20,}['"]"""),
]

_FRONTEND_SECRET_FALSE_POSITIVES = {
    "placeholder", "example", "your_", "xxx", "test",
    "changeme", "PLACEHOLDER", "TODO", "env.",
    "import.meta.env", "process.env",
}


def check_frontend_secrets(svelte_files: list[Path], ts_files: list[Path]) -> CheckResult:
    """No hardcoded secrets in frontend code."""
    result = CheckResult(
        "no_frontend_secrets",
        "No hardcoded secrets in Svelte/TypeScript files",
    )
    all_files = list(svelte_files) + list(ts_files)
    for fpath in all_files:
        for i, line in enumerate(_read_lines(fpath), 1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*"):
                continue
            for pat in _FRONTEND_SECRET_PATTERNS:
                m = pat.search(line)
                if m:
                    matched = m.group(0).lower()
                    if any(fp in matched for fp in _FRONTEND_SECRET_FALSE_POSITIVES):
                        continue
                    result.add_violation(
                        str(fpath.relative_to(_PROJECT_ROOT)), i,
                        f"Possible hardcoded secret: {stripped[:80]}",
                    )
                    break
    return result


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def run_all_checks(verbose: bool = False) -> dict[str, Any]:
    """Run all security checks and return a JSON-serializable report."""
    py_files = _py_files(include_tests=False)
    svelte_files = _svelte_files()
    ts_files = _ts_files()

    checks = [
        check_no_raw_sqlite(py_files),
        check_no_eval_exec(py_files),
        check_no_hardcoded_secrets(py_files),
        check_no_pickle(py_files),
        check_sql_parameterized(py_files),
        check_no_fstring_sql(py_files),
        check_no_shell_true(py_files),
        check_csrf_protection(py_files),
        check_no_hardcoded_colors(svelte_files),
        check_checkpoint_hardcoded(py_files),
        check_no_french(py_files, svelte_files),
        # S155 additions
        check_no_unsafe_yaml(py_files),
        check_path_traversal(py_files),
        check_ssrf_vectors(py_files),
        check_rate_limiting(py_files),
        check_cookie_security(py_files),
        check_insecure_random(py_files),
        check_frontend_secrets(svelte_files, ts_files),
    ]

    passed = sum(1 for c in checks if c.passed)
    failed = sum(1 for c in checks if not c.passed)

    report = {
        "checks": [c.to_dict() for c in checks],
        "passed": passed,
        "failed": failed,
        "total": len(checks),
        "all_passed": failed == 0,
    }

    if verbose:
        for c in checks:
            status = "PASS" if c.passed else "FAIL"
            print(f"  [{status}] {c.name}: {c.description}")
            if not c.passed:
                for v in c.violations[:5]:
                    print(f"         {v['file']}:{v['line']} -- {v['detail']}")
                if len(c.violations) > 5:
                    print(f"         ... and {len(c.violations) - 5} more")

    return report


def main() -> int:
    """Entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Opti-Oignon Security Scanner (S130/S155)")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not args.json:
        print("Opti-Oignon Security Scanner (S130/S155)")
        print("=" * 50)

    report = run_all_checks(verbose=not args.json)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print()
        print(f"Results: {report['passed']}/{report['total']} passed, "
              f"{report['failed']} failed")
        if report["all_passed"]:
            print("All checks PASSED.")
        else:
            print("Some checks FAILED -- review violations above.")

    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
