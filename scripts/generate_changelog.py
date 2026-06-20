#!/usr/bin/env python3
"""
Generate a release changelog from git commits since the last tag.

Parses conventional commit prefixes (feat:, fix:, security:, docs:,
test:, ci:, refactor:, perf:, chore:) and groups them into sections.
Falls back to a flat list if no conventional commits are found.

Usage:
    python3 scripts/generate_changelog.py [--tag TAG] [--format md|text]

Output goes to stdout. Redirect to a file if needed:
    python3 scripts/generate_changelog.py > RELEASE_CHANGELOG.md

Exit codes:
    0 -- success
    1 -- git not available or not a git repository
"""

import argparse
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone


# Conventional commit categories and their display names
CATEGORIES: OrderedDict[str, str] = OrderedDict([
    ("feat", "Features"),
    ("fix", "Bug Fixes"),
    ("security", "Security"),
    ("perf", "Performance"),
    ("refactor", "Refactoring"),
    ("docs", "Documentation"),
    ("test", "Tests"),
    ("ci", "CI/CD"),
    ("chore", "Maintenance"),
])

# Pattern to match conventional commit prefix
# Matches: "feat: message", "fix(scope): message", "security: message"
CONVENTIONAL_RE = re.compile(
    r"^(?P<type>" + "|".join(CATEGORIES.keys()) + r")"
    r"(?:\([^)]*\))?"  # optional scope in parentheses
    r":\s*(?P<message>.+)$",
    re.IGNORECASE,
)


def run_git(*args: str) -> str:
    """Run a git command and return stripped stdout."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except FileNotFoundError:
        print("Error: git is not installed or not in PATH", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"Error: git {' '.join(args)} failed: {exc.stderr}", file=sys.stderr)
        sys.exit(1)


def get_previous_tag() -> str | None:
    """Find the most recent tag before HEAD."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0", "HEAD^"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def get_current_tag() -> str | None:
    """Get the tag pointing at HEAD, if any."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def get_commits_since(ref: str | None) -> list[dict[str, str]]:
    """
    Get commits since a reference (tag or None for all commits).

    Returns a list of dicts with 'hash', 'subject', 'body' keys.
    """
    # Format: short hash, subject, body separated by field markers
    sep = "---FIELD---"
    log_format = f"%h{sep}%s{sep}%b{sep}---COMMIT---"

    if ref:
        range_spec = f"{ref}..HEAD"
    else:
        range_spec = "HEAD"

    raw = run_git("log", range_spec, f"--pretty=format:{log_format}", "--no-merges")

    if not raw:
        return []

    commits = []
    for block in raw.split("---COMMIT---"):
        block = block.strip()
        if not block:
            continue
        parts = block.split(sep)
        if len(parts) >= 2:
            commits.append({
                "hash": parts[0].strip(),
                "subject": parts[1].strip(),
                "body": parts[2].strip() if len(parts) > 2 else "",
            })

    return commits


def categorize_commits(
    commits: list[dict[str, str]],
) -> tuple[OrderedDict[str, list[dict[str, str]]], list[dict[str, str]]]:
    """
    Sort commits into conventional-commit categories.

    Returns (categorized, uncategorized) where categorized is an
    OrderedDict mapping category key to list of commits, and
    uncategorized is the leftover list.
    """
    categorized: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for key in CATEGORIES:
        categorized[key] = []

    uncategorized = []

    for commit in commits:
        match = CONVENTIONAL_RE.match(commit["subject"])
        if match:
            commit_type = match.group("type").lower()
            commit["clean_message"] = match.group("message")
            categorized[commit_type].append(commit)
        else:
            commit["clean_message"] = commit["subject"]
            uncategorized.append(commit)

    return categorized, uncategorized


def format_markdown(
    current_tag: str | None,
    previous_tag: str | None,
    categorized: OrderedDict[str, list[dict[str, str]]],
    uncategorized: list[dict[str, str]],
) -> str:
    """Format the changelog as markdown."""
    lines = []

    # Header
    version_label = current_tag or "Unreleased"
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines.append(f"## {version_label} ({date_str})")
    lines.append("")

    has_content = False

    # Categorized sections
    for key, commits in categorized.items():
        if not commits:
            continue
        has_content = True
        section_name = CATEGORIES[key]
        lines.append(f"### {section_name}")
        lines.append("")
        for commit in commits:
            msg = commit["clean_message"]
            short_hash = commit["hash"]
            lines.append(f"- {msg} ({short_hash})")
        lines.append("")

    # Uncategorized
    if uncategorized:
        has_content = True
        lines.append("### Other Changes")
        lines.append("")
        for commit in uncategorized:
            msg = commit["clean_message"]
            short_hash = commit["hash"]
            lines.append(f"- {msg} ({short_hash})")
        lines.append("")

    if not has_content:
        lines.append("No changes recorded.")
        lines.append("")

    # Comparison link hint
    if previous_tag and current_tag:
        lines.append(
            f"**Full diff:** `{previous_tag}...{current_tag}`"
        )
        lines.append("")

    return "\n".join(lines)


def format_text(
    current_tag: str | None,
    previous_tag: str | None,
    categorized: OrderedDict[str, list[dict[str, str]]],
    uncategorized: list[dict[str, str]],
) -> str:
    """Format the changelog as plain text."""
    lines = []

    version_label = current_tag or "Unreleased"
    lines.append(f"Release: {version_label}")
    lines.append("")

    for key, commits in categorized.items():
        if not commits:
            continue
        section_name = CATEGORIES[key]
        lines.append(f"  {section_name}:")
        for commit in commits:
            lines.append(f"    - {commit['clean_message']}")
        lines.append("")

    if uncategorized:
        lines.append("  Other:")
        for commit in uncategorized:
            lines.append(f"    - {commit['clean_message']}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Generate release changelog from git commits."
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Base tag to compare against (default: auto-detect previous tag)",
    )
    parser.add_argument(
        "--format",
        choices=["md", "text"],
        default="md",
        dest="output_format",
        help="Output format (default: md)",
    )
    args = parser.parse_args()

    # Determine range
    current_tag = get_current_tag()
    previous_tag = args.tag or get_previous_tag()

    # Get commits
    commits = get_commits_since(previous_tag)

    if not commits:
        # Fallback: if no commits found (e.g., shallow clone),
        # produce a minimal changelog
        print(f"## {current_tag or 'Unreleased'}")
        print("")
        print("Release built from tag. See CHANGELOG.md for details.")
        return

    # Categorize
    categorized, uncategorized = categorize_commits(commits)

    # Format output
    if args.output_format == "md":
        output = format_markdown(current_tag, previous_tag, categorized, uncategorized)
    else:
        output = format_text(current_tag, previous_tag, categorized, uncategorized)

    print(output)


if __name__ == "__main__":
    main()
