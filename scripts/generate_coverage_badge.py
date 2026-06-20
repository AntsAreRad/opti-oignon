#!/usr/bin/env python3
"""
Generate an SVG coverage badge from pytest-cov JSON output.

Usage:
    python3 scripts/generate_coverage_badge.py coverage.json
    python3 scripts/generate_coverage_badge.py coverage.json --output assets/coverage-badge.svg

Reads the total coverage percentage from a pytest-cov JSON report
and generates a shields.io-style SVG badge.
"""

import json
import os
import sys


# Badge color thresholds
THRESHOLDS = [
    (90, "#4c1"),      # bright green
    (75, "#97ca00"),   # green
    (60, "#a4a61d"),   # yellow-green
    (40, "#dfb317"),   # yellow
    (20, "#fe7d37"),   # orange
    (0, "#e05d44"),    # red
]

# SVG template (shields.io flat style)
SVG_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="a">
    <rect width="{width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#a)">
    <rect width="61" height="20" fill="#555"/>
    <rect x="61" width="{value_width}" height="20" fill="{color}"/>
    <rect width="{width}" height="20" fill="url(#b)"/>
  </g>
  <g fill="#fff" text-anchor="middle"
     font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="30.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
    <text x="30.5" y="14">coverage</text>
    <text x="{text_x}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{text_x}" y="14">{label}</text>
  </g>
</svg>"""


def get_color(percentage):
    """Return badge color based on coverage percentage."""
    for threshold, color in THRESHOLDS:
        if percentage >= threshold:
            return color
    return THRESHOLDS[-1][1]


def generate_badge(percentage, output_path):
    """Generate an SVG badge file."""
    label = f"{percentage:.0f}%"
    # Approximate text width: ~7px per character + padding
    value_width = max(len(label) * 7 + 10, 35)
    width = 61 + value_width
    text_x = 61 + value_width / 2

    svg = SVG_TEMPLATE.format(
        width=width,
        value_width=value_width,
        color=get_color(percentage),
        text_x=text_x,
        label=label,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg.strip() + "\n")

    print(f"Badge generated: {output_path} ({label})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/generate_coverage_badge.py <coverage.json> [--output PATH]")
        sys.exit(1)

    json_path = sys.argv[1]
    output_path = "assets/coverage-badge.svg"

    # Parse --output flag
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    if not os.path.isfile(json_path):
        print(f"Error: {json_path} not found")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # pytest-cov JSON format: {"totals": {"percent_covered": 42.5, ...}, ...}
    totals = data.get("totals", {})
    percentage = totals.get("percent_covered", 0.0)

    generate_badge(percentage, output_path)


if __name__ == "__main__":
    main()
