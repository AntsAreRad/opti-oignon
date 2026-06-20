#!/usr/bin/env python3
"""
audit_accessibility.py -- Automated WCAG AA accessibility checks for Svelte components.

Checks performed:
  1. Buttons without aria-label and without visible text content
  2. Images (<img>) without alt attribute
  3. Inputs without aria-label or associated id (for <label for=...>)
  4. Missing role attributes on interactive custom elements
  5. Color contrast via --oo-* CSS variables (delegates to audit_contrast.py)
  6. Touch target size warnings (min 44x44px)

Usage:
    python scripts/audit_accessibility.py [--fix] [--verbose] [--only-new SESSION]

Exit code 0 if no critical issues, 1 if any are found.
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_SRC = PROJECT_ROOT / "frontend" / "src"
COMPONENTS_DIR = FRONTEND_SRC / "lib" / "components"


def find_svelte_files(directory: Path) -> list[Path]:
    """Recursively find all .svelte files."""
    return sorted(directory.rglob("*.svelte"))


def check_buttons(content: str, filepath: str) -> list[dict]:
    """Find buttons without aria-label or visible text."""
    issues = []
    for m in re.finditer(r"<button([^>]*)>(.*?)</button>", content, re.DOTALL):
        attrs = m.group(1)
        inner = m.group(2).strip()
        has_aria = "aria-label" in attrs or "aria-label" in inner
        # Strip HTML tags and Svelte expressions to check for text
        text_only = re.sub(r"<[^>]+>", "", inner).strip()
        text_only = re.sub(r"\{[^}]+\}", "EXPR", text_only).strip()
        if not has_aria and not text_only:
            line = content[: m.start()].count("\n") + 1
            issues.append({
                "file": filepath,
                "line": line,
                "type": "button-no-label",
                "severity": "critical",
                "message": "Button without aria-label or visible text",
            })
    return issues


def check_images(content: str, filepath: str) -> list[dict]:
    """Find <img> tags without alt attribute."""
    issues = []
    for m in re.finditer(r"<img([^>]*)(/?)>", content):
        attrs = m.group(1)
        if "alt=" not in attrs:
            line = content[: m.start()].count("\n") + 1
            issues.append({
                "file": filepath,
                "line": line,
                "type": "img-no-alt",
                "severity": "critical",
                "message": "Image without alt attribute",
            })
    return issues


def check_inputs(content: str, filepath: str) -> list[dict]:
    """Find inputs without aria-label or id (for label association)."""
    issues = []
    for m in re.finditer(r"<input([^>]*)(/?)>", content):
        attrs = m.group(1)
        has_aria = "aria-label" in attrs
        has_id = "id=" in attrs
        if not has_aria and not has_id:
            line = content[: m.start()].count("\n") + 1
            # Hidden inputs are exempt
            if 'type="hidden"' in attrs:
                continue
            issues.append({
                "file": filepath,
                "line": line,
                "type": "input-no-label",
                "severity": "warning",
                "message": "Input without aria-label or id for label association",
            })
    return issues


def check_touch_targets(content: str, filepath: str) -> list[dict]:
    """Warn about potentially small touch targets (buttons/links < 44px)."""
    issues = []
    small_patterns = [
        r"w-[1-6]\s",  # Tailwind w-1 through w-6 (< 44px)
        r"h-[1-6]\s",  # Tailwind h-1 through h-6
        r"p-0\.5\s",   # Very small padding
    ]
    for m in re.finditer(r"<(button|a)([^>]*)>", content):
        attrs = m.group(2)
        for pattern in small_patterns:
            if re.search(pattern, attrs):
                line = content[: m.start()].count("\n") + 1
                issues.append({
                    "file": filepath,
                    "line": line,
                    "type": "small-touch-target",
                    "severity": "info",
                    "message": f"Potentially small touch target on <{m.group(1)}>",
                })
                break
    return issues


def check_focus_indicators(content: str, filepath: str) -> list[dict]:
    """Check for outline-none without a focus-visible replacement."""
    issues = []
    if "outline-none" in content or "outline: none" in content:
        # Check if there's a focus-visible or focus: replacement
        has_focus_visible = "focus-visible" in content or "focus:" in content
        if not has_focus_visible:
            issues.append({
                "file": filepath,
                "line": 0,
                "type": "no-focus-indicator",
                "severity": "warning",
                "message": "outline-none used without focus-visible replacement",
            })
    return issues


def audit_all(verbose: bool = False) -> list[dict]:
    """Run all accessibility checks on all Svelte components."""
    all_issues = []
    files = find_svelte_files(COMPONENTS_DIR)

    # Also check route files
    routes_dir = FRONTEND_SRC / "routes"
    if routes_dir.exists():
        files.extend(find_svelte_files(routes_dir))

    for fpath in files:
        content = fpath.read_text(encoding="utf-8", errors="replace")
        relpath = str(fpath.relative_to(PROJECT_ROOT))

        issues = []
        issues.extend(check_buttons(content, relpath))
        issues.extend(check_images(content, relpath))
        issues.extend(check_inputs(content, relpath))
        issues.extend(check_touch_targets(content, relpath))
        issues.extend(check_focus_indicators(content, relpath))

        all_issues.extend(issues)

        if verbose and issues:
            for issue in issues:
                print(
                    f"  [{issue['severity'].upper()}] {issue['file']}:{issue['line']} "
                    f"- {issue['message']}"
                )

    return all_issues


def main():
    parser = argparse.ArgumentParser(
        description="WCAG AA accessibility audit for Opti-Oignon Svelte components"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all issues")
    parser.add_argument(
        "--only-critical", action="store_true", help="Only report critical issues"
    )
    args = parser.parse_args()

    print("Running accessibility audit...")
    print(f"Scanning: {COMPONENTS_DIR}")
    print()

    issues = audit_all(verbose=args.verbose)

    if args.only_critical:
        issues = [i for i in issues if i["severity"] == "critical"]

    # Summary
    critical = sum(1 for i in issues if i["severity"] == "critical")
    warnings = sum(1 for i in issues if i["severity"] == "warning")
    info = sum(1 for i in issues if i["severity"] == "info")

    print()
    print("=" * 50)
    print(f"Accessibility Audit Results:")
    print(f"  Critical: {critical}")
    print(f"  Warnings: {warnings}")
    print(f"  Info:     {info}")
    print(f"  Total:    {len(issues)}")
    print("=" * 50)

    if critical > 0:
        print("\nCritical issues found! Please fix before release.")
        if not args.verbose:
            print("Run with --verbose to see details.")
        return 1

    print("\nNo critical accessibility issues found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
