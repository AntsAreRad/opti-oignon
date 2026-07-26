#!/usr/bin/env python3
"""
audit_contrast.py — WCAG AA contrast validator for Opti-Oignon theme.css
Parses CSS custom properties from theme.css, resolves var() references,
computes relative luminance and contrast ratios for all fg/bg pairs.

WCAG AA requirements:
  - Normal text (< 18pt / < 14pt bold): contrast >= 4.5:1
  - Large text (>= 18pt / >= 14pt bold): contrast >= 3.0:1

Usage:
    python scripts/audit_contrast.py [--fix] [--verbose]

Exit code 0 if all pairs pass, 1 if any pair fails.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Minimum contrast ratios per WCAG AA
RATIO_NORMAL = 4.5
RATIO_LARGE = 3.0

# Pairs to check: (fg_var, bg_var, text_type)
# text_type: "normal" requires 4.5:1, "large" requires 3.0:1
CONTRAST_PAIRS = [
    # -- Dark mode primary text on backgrounds --
    ("--oo-fg-primary", "--oo-bg-base", "normal"),
    ("--oo-fg-primary", "--oo-bg-surface", "normal"),
    ("--oo-fg-primary", "--oo-bg-elevated", "normal"),
    ("--oo-fg-primary", "--oo-bg-overlay", "normal"),
    ("--oo-fg-secondary", "--oo-bg-base", "normal"),
    ("--oo-fg-secondary", "--oo-bg-surface", "normal"),
    ("--oo-fg-tertiary", "--oo-bg-base", "normal"),
    ("--oo-fg-tertiary", "--oo-bg-surface", "normal"),
    ("--oo-fg-muted", "--oo-bg-base", "large"),
    ("--oo-fg-muted", "--oo-bg-surface", "large"),

    # -- Accent on backgrounds --
    ("--oo-tobacco", "--oo-bg-base", "normal"),
    ("--oo-tobacco", "--oo-bg-surface", "normal"),
    ("--oo-sage", "--oo-bg-base", "normal"),
    ("--oo-sage", "--oo-bg-surface", "normal"),

    # -- Buttons: fg on bg --
    ("--oo-btn-primary-fg", "--oo-btn-primary-bg", "large"),
    ("--oo-btn-secondary-fg", "--oo-btn-secondary-bg", "normal"),
    ("--oo-fg-on-accent", "--oo-acc-500", "large"),

    # -- Semantic on their backgrounds (badge/pill text) --
    ("--oo-success", "--oo-bg-base", "normal"),
    ("--oo-success", "--oo-bg-surface", "normal"),
    ("--oo-error", "--oo-bg-base", "normal"),
    ("--oo-error", "--oo-bg-surface", "normal"),
    ("--oo-warning", "--oo-bg-base", "normal"),
    ("--oo-warning", "--oo-bg-surface", "normal"),
    ("--oo-info", "--oo-bg-base", "normal"),
    ("--oo-info", "--oo-bg-surface", "normal"),

    # -- Message bubbles --
    ("--oo-msg-user-fg", "--oo-bg-base", "normal"),
    ("--oo-msg-bot-fg", "--oo-msg-bot-bg", "normal"),

    # -- Sidebar --
    ("--oo-fg-primary", "--oo-sidebar-bg", "normal"),
    ("--oo-fg-tertiary", "--oo-sidebar-bg", "normal"),

    # -- Input --
    ("--oo-fg-primary", "--oo-input-bg", "normal"),
    ("--oo-fg-secondary", "--oo-input-bg", "normal"),
]


def parse_hex(hex_str: str) -> Optional[tuple]:
    """Parse a hex color string to (r, g, b) tuple (0-255)."""
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c * 2 for c in hex_str)
    if len(hex_str) != 6:
        return None
    try:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return (r, g, b)
    except ValueError:
        return None


def parse_rgba(rgba_str: str) -> Optional[tuple]:
    """Parse rgba() to (r, g, b, a) tuple. Returns None on failure."""
    match = re.match(
        r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+))?\s*\)",
        rgba_str.strip(),
    )
    if not match:
        return None
    r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
    a = float(match.group(4)) if match.group(4) else 1.0
    return (r, g, b, a)


def blend_on_bg(fg_rgba: tuple, bg_rgb: tuple) -> tuple:
    """Alpha-composite fg_rgba onto bg_rgb, returns (r, g, b)."""
    r_fg, g_fg, b_fg, a = fg_rgba
    r_bg, g_bg, b_bg = bg_rgb
    r = int(r_fg * a + r_bg * (1 - a))
    g = int(g_fg * a + g_bg * (1 - a))
    b = int(b_fg * a + b_bg * (1 - a))
    return (r, g, b)


def relative_luminance(rgb: tuple) -> float:
    """Compute WCAG relative luminance from (r, g, b) 0-255."""
    vals = []
    for c in rgb:
        s = c / 255.0
        if s <= 0.04045:
            vals.append(s / 12.92)
        else:
            vals.append(((s + 0.055) / 1.055) ** 2.4)
    return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]


def contrast_ratio(rgb1: tuple, rgb2: tuple) -> float:
    """Compute WCAG contrast ratio between two RGB colors."""
    l1 = relative_luminance(rgb1)
    l2 = relative_luminance(rgb2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def parse_color_value(value: str) -> Optional[tuple]:
    """Parse a CSS color value (hex or rgba) to (r, g, b)."""
    value = value.strip()
    if value.startswith("#"):
        return parse_hex(value)
    if value.startswith("rgba") or value.startswith("rgb("):
        rgba = parse_rgba(value)
        if rgba is None:
            return None
        if len(rgba) == 4 and rgba[3] < 1.0:
            # For semi-transparent colors, blend on a neutral mid-grey
            # to get a representative solid color
            return rgba[:3]
        return rgba[:3]
    return None


def extract_css_vars(css_text: str) -> dict:
    """
    Extract all --oo-* custom property declarations from CSS text.
    Returns dict mapping variable name to raw value string.
    Parses both :root/html.dark and html:not(.dark) blocks.
    """
    variables = {}
    # Match property declarations like --oo-foo: value;
    pattern = re.compile(r"(--oo-[\w-]+)\s*:\s*([^;]+);")
    for match in pattern.finditer(css_text):
        name = match.group(1).strip()
        value = match.group(2).strip()
        variables[name] = value
    return variables


def extract_themed_vars(css_text: str) -> tuple:
    """
    Extract variables separately for dark and light themes.
    Returns (dark_vars, light_vars) dicts.
    """
    dark_vars = {}
    light_vars = {}

    # Split into blocks by looking for the selectors
    # Strategy: find the dark block (:root, html.dark) and light block (html:not(.dark))
    # and parse each separately

    # Find all blocks with their selectors
    # Simplified approach: parse line by line tracking which block we are in
    lines = css_text.split("\n")
    current_block = None
    brace_depth = 0
    block_content = []

    for line in lines:
        stripped = line.strip()

        if current_block is None:
            if ":root" in stripped or "html.dark" in stripped:
                current_block = "dark"
                brace_depth = 0
                block_content = []
            elif "html:not(.dark)" in stripped:
                current_block = "light"
                brace_depth = 0
                block_content = []

        if current_block is not None:
            brace_depth += stripped.count("{") - stripped.count("}")
            block_content.append(line)
            if brace_depth <= 0 and "{" in "".join(block_content):
                content = "\n".join(block_content)
                parsed = extract_css_vars(content)
                if current_block == "dark":
                    dark_vars.update(parsed)
                else:
                    light_vars.update(parsed)
                current_block = None
                block_content = []

    return dark_vars, light_vars


def resolve_var(variables: dict, value: str, depth: int = 0) -> str:
    """Resolve var() references in a value string."""
    if depth > 10:
        return value
    match = re.match(r"var\((--[\w-]+)\)", value.strip())
    if match:
        ref = match.group(1)
        if ref in variables:
            return resolve_var(variables, variables[ref], depth + 1)
    return value


def resolve_to_rgb(variables: dict, var_name: str) -> Optional[tuple]:
    """Resolve a CSS variable name to an (r, g, b) tuple."""
    if var_name not in variables:
        return None
    raw = resolve_var(variables, variables[var_name])
    return parse_color_value(raw)


def audit_theme(
    variables: dict, theme_name: str, verbose: bool = False
) -> list:
    """
    Audit all contrast pairs for a given theme.
    Returns list of failure dicts.
    """
    failures = []
    passes = 0

    for fg_var, bg_var, text_type in CONTRAST_PAIRS:
        fg_rgb = resolve_to_rgb(variables, fg_var)
        bg_rgb = resolve_to_rgb(variables, bg_var)

        if fg_rgb is None or bg_rgb is None:
            if verbose:
                missing = []
                if fg_rgb is None:
                    missing.append(f"fg={fg_var}")
                if bg_rgb is None:
                    missing.append(f"bg={bg_var}")
                print(
                    f"  SKIP  {theme_name}: {fg_var} on {bg_var} "
                    f"(unresolved: {', '.join(missing)})"
                )
            continue

        ratio = contrast_ratio(fg_rgb, bg_rgb)
        required = RATIO_NORMAL if text_type == "normal" else RATIO_LARGE
        passed = ratio >= required

        if passed:
            passes += 1
            if verbose:
                print(
                    f"  PASS  {theme_name}: {fg_var} on {bg_var} "
                    f"= {ratio:.2f}:1 (need {required}:1)"
                )
        else:
            failures.append(
                {
                    "theme": theme_name,
                    "fg": fg_var,
                    "bg": bg_var,
                    "fg_hex": "#{:02x}{:02x}{:02x}".format(*fg_rgb),
                    "bg_hex": "#{:02x}{:02x}{:02x}".format(*bg_rgb),
                    "ratio": round(ratio, 2),
                    "required": required,
                    "text_type": text_type,
                }
            )
            print(
                f"  FAIL  {theme_name}: {fg_var} on {bg_var} "
                f"= {ratio:.2f}:1 (need {required}:1)"
            )

    return failures


def main():
    parser = argparse.ArgumentParser(
        description="WCAG AA contrast audit for Opti-Oignon theme.css"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show passing pairs too"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )
    args = parser.parse_args()

    # Find theme.css
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    theme_path = project_root / "frontend" / "src" / "styles" / "theme.css"

    if not theme_path.exists():
        print(f"ERROR: theme.css not found at {theme_path}", file=sys.stderr)
        sys.exit(1)

    css_text = theme_path.read_text(encoding="utf-8")
    dark_vars, light_vars = extract_themed_vars(css_text)

    print(f"Parsed {len(dark_vars)} dark vars, {len(light_vars)} light vars")
    print(f"Checking {len(CONTRAST_PAIRS)} pairs per theme...\n")

    print("=== Dark Theme ===")
    dark_failures = audit_theme(dark_vars, "dark", verbose=args.verbose)

    print("\n=== Light Theme ===")
    light_failures = audit_theme(light_vars, "light", verbose=args.verbose)

    all_failures = dark_failures + light_failures
    total_checked = len(CONTRAST_PAIRS) * 2
    total_passed = total_checked - len(all_failures)

    print(f"\n{'=' * 50}")
    print(f"Results: {total_passed}/{total_checked} pairs pass WCAG AA")

    if all_failures:
        print(f"FAILURES: {len(all_failures)}")
        for f in all_failures:
            print(
                f"  {f['theme']}: {f['fg']} ({f['fg_hex']}) on "
                f"{f['bg']} ({f['bg_hex']}) = {f['ratio']}:1 "
                f"(need {f['required']}:1, {f['text_type']} text)"
            )
    else:
        print("All pairs pass WCAG AA compliance!")

    # Save report
    report_path = project_root / "docs" / "contrast_audit_report.json"
    report = {
        "total_pairs": total_checked,
        "passed": total_passed,
        "failed": len(all_failures),
        "failures": all_failures,
        "dark_vars_count": len(dark_vars),
        "light_vars_count": len(light_vars),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    if args.json:
        print(json.dumps(report, indent=2))

    sys.exit(1 if all_failures else 0)


if __name__ == "__main__":
    main()
