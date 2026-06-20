#!/usr/bin/env python3
"""
audit_colors.py -- Automated color scanner for Opti-Oignon Svelte components.
S92: Detects hardcoded colors, inline styles bypassing CSS variables,
and Tailwind color utility classes that override theme.css tokens.

Output: JSON report with file, line number, violation type, suggested fix.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any


# Patterns for detecting color violations
HEX_COLOR_RE = re.compile(
    r"""(?<!var\(--)#[0-9a-fA-F]{3,8}\b""",
    re.IGNORECASE,
)

RGBA_INLINE_RE = re.compile(
    r"""rgba?\(\s*\d+""",
    re.IGNORECASE,
)

HSLA_INLINE_RE = re.compile(
    r"""hsla?\(\s*\d+""",
    re.IGNORECASE,
)

INLINE_COLOR_STYLE_RE = re.compile(
    r"""style\s*=\s*["'][^"']*(?:(?:^|;\s*)(?:color|background-color|background|border-color|border)\s*:\s*(?!var\(--oo-))""",
    re.IGNORECASE,
)

BG_WHITE_RE = re.compile(
    r"""background(?:-color)?\s*:\s*white\b""",
    re.IGNORECASE,
)

COLOR_WHITE_RE = re.compile(
    r"""(?:^|;\s*)color\s*:\s*white\b""",
    re.IGNORECASE,
)

# Tailwind color utility classes that bypass theme variables
# Matches: text-{color}-{shade}, bg-{color}-{shade}, border-{color}-{shade}
TAILWIND_COLOR_RE = re.compile(
    r"""\b(?:text|bg|border|ring|from|to|via|outline|shadow|divide|placeholder)-"""
    r"""(?:red|green|blue|amber|yellow|emerald|orange|purple|pink|cyan|teal|"""
    r"""indigo|gray|slate|zinc|stone|neutral|lime|rose|fuchsia|violet|sky|white|black)-"""
    r"""\d{2,3}(?:\/\d+)?""",
)

# Simpler pattern for standalone color keywords in Tailwind
TAILWIND_NAMED_RE = re.compile(
    r"""\b(?:text|bg|border)-(?:white|black)\b"""
)

# Intentional exceptions: overlay backdrops, var() bracket notation
EXCEPTION_RE = re.compile(
    r"""bg-black/\d+|"""
    r"""\[var\(--oo-[^\]]+\)\]"""
)

# Allowed exceptions (CSS variable references, comments, Svelte template syntax)
SVELTE_EACH_RE = re.compile(r"""\{#each\b""")
COMMENT_RE = re.compile(r"""<!--.*?-->""", re.DOTALL)
CSS_VAR_RE = re.compile(r"""var\(--oo-[^)]+\)""")

# Map Tailwind colors to suggested CSS variable replacements
TAILWIND_TO_VAR: dict[str, str] = {
    "red": "--oo-error",
    "green": "--oo-success",
    "emerald": "--oo-success",
    "amber": "--oo-warning",
    "yellow": "--oo-warning",
    "blue": "--oo-info",
    "purple": "--oo-pipe-tools",
    "orange": "--oo-pipe-code",
    "pink": "--oo-pipe-reason",
    "cyan": "--oo-info",
    "teal": "--oo-pipe-correct",
}


def suggest_fix(violation_type: str, match_text: str) -> str:
    """Generate a suggested fix for a violation."""
    if violation_type == "hardcoded_hex":
        return "Replace with appropriate var(--oo-*) CSS variable"
    if violation_type == "inline_rgba":
        return "Replace with var(--oo-{semantic}-bg) or var(--oo-{semantic}-bd)"
    if violation_type == "inline_white":
        return "Replace 'white' with var(--oo-fg-primary) or var(--oo-bg-elevated)"
    if violation_type == "tailwind_color":
        for tw_name, css_var in TAILWIND_TO_VAR.items():
            if tw_name in match_text:
                return f"Replace with {css_var} or style using var({css_var})"
        return "Replace with appropriate var(--oo-*) CSS variable"
    return "Review and replace with theme CSS variable"


def is_in_script_section(line: str, in_script: bool) -> bool:
    """Track whether we are inside a <script> block."""
    if "<script" in line:
        return True
    if "</script>" in line:
        return False
    return in_script


def scan_file(filepath: Path) -> list[dict[str, Any]]:
    """Scan a single Svelte file for color violations."""
    violations: list[dict[str, Any]] = []
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return violations

    lines = content.split("\n")

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Skip HTML comments
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue

        # Skip lines that are only Svelte each blocks (false positives from hex in {#each})
        if SVELTE_EACH_RE.search(stripped):
            continue

        # Check for hardcoded hex colors
        hex_matches = HEX_COLOR_RE.findall(line)
        for match in hex_matches:
            # Filter out Svelte template syntax like {#each} which contains hex-like patterns
            # Also filter HTML entities like &#9733;
            if match.startswith("#") and not re.match(r"^#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?(?:[0-9a-fA-F]{2})?$", match):
                continue
            # Skip HTML entity references
            idx = line.find(match)
            if idx > 0 and line[idx - 1] == "&":
                continue
            violations.append({
                "file": str(filepath),
                "line": line_num,
                "type": "hardcoded_hex",
                "match": match,
                "context": stripped[:120],
                "fix": suggest_fix("hardcoded_hex", match),
            })

        # Check for inline rgba/hsla (not inside var(--oo-*))
        for pattern, vtype in [
            (RGBA_INLINE_RE, "inline_rgba"),
            (HSLA_INLINE_RE, "inline_hsla"),
        ]:
            if pattern.search(line):
                # Only flag if the rgba is NOT a fallback inside var()
                # e.g. var(--oo-error-bg, rgba(...)) is acceptable
                clean_line = re.sub(r"var\(--oo-[^)]*\)", "", line)
                if pattern.search(clean_line):
                    violations.append({
                        "file": str(filepath),
                        "line": line_num,
                        "type": vtype,
                        "match": pattern.search(clean_line).group(0),  # type: ignore[union-attr]
                        "context": stripped[:120],
                        "fix": suggest_fix(vtype, ""),
                    })

        # Check for background-color: white / color: white
        if BG_WHITE_RE.search(line):
            violations.append({
                "file": str(filepath),
                "line": line_num,
                "type": "inline_white",
                "match": "background-color: white",
                "context": stripped[:120],
                "fix": suggest_fix("inline_white", ""),
            })

        # Check for Tailwind color utility classes
        tw_matches = TAILWIND_COLOR_RE.findall(line)
        for match in tw_matches:
            # Skip if the match is inside a CSS variable bracket notation
            match_idx = line.find(match)
            if match_idx > 0 and line[match_idx - 1:match_idx] == '[':
                continue
            violations.append({
                "file": str(filepath),
                "line": line_num,
                "type": "tailwind_color",
                "match": match,
                "context": stripped[:120],
                "fix": suggest_fix("tailwind_color", match),
            })

        # Check for text-white / bg-white / bg-black standalone
        tw_named = TAILWIND_NAMED_RE.findall(line)
        for match in tw_named:
            # Skip bg-black/NN overlay patterns (intentional, handled by app.css)
            if match == "bg-black" and re.search(r'bg-black/\d+', line):
                continue
            # Skip if inside CSS variable bracket notation
            match_idx = line.find(match)
            if match_idx > 0 and line[match_idx - 1:match_idx] == '[':
                continue
            violations.append({
                "file": str(filepath),
                "line": line_num,
                "type": "tailwind_named_color",
                "match": match,
                "context": stripped[:120],
                "fix": "Replace with var(--oo-fg-primary) or var(--oo-bg-base)",
            })

    return violations


def scan_directory(root: Path) -> list[dict[str, Any]]:
    """Scan all Svelte files in a directory tree."""
    all_violations: list[dict[str, Any]] = []
    svelte_files = sorted(root.rglob("*.svelte"))
    for filepath in svelte_files:
        file_violations = scan_file(filepath)
        all_violations.extend(file_violations)
    return all_violations


def make_summary(violations: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a summary of violations."""
    by_type: dict[str, int] = {}
    by_file: dict[str, int] = {}
    for v in violations:
        by_type[v["type"]] = by_type.get(v["type"], 0) + 1
        by_file[v["file"]] = by_file.get(v["file"], 0) + 1

    return {
        "total_violations": len(violations),
        "by_type": dict(sorted(by_type.items(), key=lambda x: -x[1])),
        "by_file": dict(sorted(by_file.items(), key=lambda x: -x[1])),
        "files_with_violations": len(by_file),
    }


def main() -> None:
    """Entry point."""
    root = Path(__file__).resolve().parent.parent / "frontend" / "src"
    if not root.exists():
        print(f"Error: frontend source not found at {root}", file=sys.stderr)
        sys.exit(1)

    violations = scan_directory(root)
    summary = make_summary(violations)

    report = {
        "summary": summary,
        "violations": violations,
    }

    output_path = Path(__file__).resolve().parent.parent / "docs" / "color_audit_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Color audit complete: {summary['total_violations']} violations in {summary['files_with_violations']} files")
    print(f"Report saved to: {output_path}")
    print("\nBy type:")
    for vtype, count in summary["by_type"].items():
        print(f"  {vtype}: {count}")
    print("\nTop files:")
    for filepath, count in list(summary["by_file"].items())[:15]:
        short = filepath.split("frontend/src/")[-1] if "frontend/src/" in filepath else filepath
        print(f"  {short}: {count}")


if __name__ == "__main__":
    main()
