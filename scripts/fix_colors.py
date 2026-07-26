#!/usr/bin/env python3
"""
fix_colors.py -- Automated batch color fixer, companion to audit_colors.py.
Replaces hardcoded Tailwind color classes and inline hex/rgba
with CSS variable references in all Svelte components.
"""

import re
from pathlib import Path

FRONTEND_SRC = Path(__file__).resolve().parent.parent / "frontend" / "src"

# ---- Tailwind class replacements ----
# Maps Tailwind color classes to theme-variable-based equivalents.
# For classes used in dark mode that are remapped in app.css for light mode,
# we keep the Tailwind class as-is if app.css handles it.
# But for cleanliness, we replace with style= or semantic classes.

# Simple class-to-class replacements (where a direct mapping exists)
TW_CLASS_MAP: dict[str, str] = {
    # --- Status dot / bar colors: replace with inline style ---
    # These need special handling, see below
}

# ---- Inline style replacements ----
INLINE_REPLACEMENTS: list[tuple[str, str]] = [
    # FeedbackWidget: hardcoded red hex
    ("'#f87171'", "'var(--oo-error)'"),
    # CompressionSettings: hardcoded emerald hex
    ("'color: #10b981;'", "'color: var(--oo-success);'"),
    # CacheStatsPanel, HumanizerPanel, ModelProfilePanel: white toggle knob
    ("background-color: white;", "background-color: var(--oo-toggle-knob);"),
    # OnionLoader: hardcoded accent hex
    ("color: string = '#B07D56'", "color: string = 'var(--oo-acc-500)'"),
    # ProxySettingsPanel: Tailwind green rgba
    ("rgba(34, 197, 94, 0.12)", "var(--oo-success-bg)"),
    ("rgba(34, 197, 94, 0.3)", "var(--oo-success-bd)"),
    # ProxySettingsPanel: Tailwind red rgba
    ("rgba(239, 68, 68, 0.12)", "var(--oo-error-bg)"),
    ("rgba(239, 68, 68, 0.3)", "var(--oo-error-bd)"),
    ("rgba(239, 68, 68, 0.15)", "var(--oo-error-bg)"),
    # ProxySettingsPanel: Tailwind amber rgba
    ("rgba(245, 158, 11, 0.08)", "var(--oo-warning-bg)"),
    ("rgba(245, 158, 11, 0.25)", "var(--oo-warning-bd)"),
    ("rgba(245, 158, 11, 0.10)", "var(--oo-warning-bg)"),
    ("rgba(245, 158, 11, 0.1)", "var(--oo-warning-bg)"),
    ("rgba(245, 158, 11, 0.12)", "var(--oo-warning-bg)"),
    ("rgba(245, 158, 11, 0.15)", "var(--oo-warning-bg)"),
    ("rgba(245, 158, 11, 0.3)", "var(--oo-warning-bd)"),
    # ConversationList: hardcoded accent rgba
    ("rgba(176, 125, 86, 0.1)", "var(--oo-msg-user-bg)"),
    ("rgba(176, 125, 86, 0.2)", "var(--oo-msg-user-bd)"),
    ("rgba(176, 125, 86, 0.15)", "var(--oo-msg-user-bg)"),
    ("rgba(176, 125, 86, 0.25)", "var(--oo-msg-user-bd)"),
    ("rgba(176, 125, 86, 0.3)", "var(--oo-acc-400)"),
    # PerformanceDashboard: black rgba
    ("rgba(0,0,0,0.2)", "var(--oo-bg-overlay)"),
    # OnboardingOverlay: dark overlay
    ("rgba(20, 18, 16, 0.85)", "var(--oo-bg-base)"),
]

# ---- Tailwind semantic class swaps ----
# These are the Tailwind color classes we want to replace in the Svelte template.
# Format: (old_class, new_class_or_style)
TW_SEMANTIC_SWAPS: list[tuple[str, str]] = [
    # --- ContextPanel bar colors ---
    ("'bg-red-500'", "'bg-[var(--oo-error)]'"),
    ("'bg-amber-500'", "'bg-[var(--oo-warning)]'"),
    ("'bg-yellow-500'", "'bg-[var(--oo-warning)]'"),
    ("'bg-emerald-500'", "'bg-[var(--oo-success)]'"),
    ("'text-emerald-400'", "'text-[var(--oo-success)]'"),
    ("'text-amber-400'", "'text-[var(--oo-warning)]'"),
    ("'text-red-400'", "'text-[var(--oo-error)]'"),
]

# ---- Full Tailwind class-to-variable map for bulk replacement ----
# These replacements happen on the raw text of each file.
BULK_CLASS_SWAPS: list[tuple[str, str]] = [
    # Text colors
    ("text-red-400", "text-[var(--oo-error)]"),
    ("text-red-300", "text-[var(--oo-error)]"),
    ("text-red-100", "text-[var(--oo-error)]"),
    ("text-green-400", "text-[var(--oo-success)]"),
    ("text-green-300", "text-[var(--oo-success)]"),
    ("text-emerald-400", "text-[var(--oo-success)]"),
    ("text-emerald-300", "text-[var(--oo-success)]"),
    ("text-amber-400", "text-[var(--oo-warning)]"),
    ("text-amber-300", "text-[var(--oo-warning)]"),
    ("text-yellow-400", "text-[var(--oo-warning)]"),
    ("text-yellow-300", "text-[var(--oo-warning)]"),
    ("text-blue-400", "text-[var(--oo-info)]"),
    ("text-blue-300", "text-[var(--oo-info)]"),
    ("text-purple-400", "text-[var(--oo-cat-purple)]"),
    ("text-orange-400", "text-[var(--oo-cat-orange)]"),
    ("text-pink-400", "text-[var(--oo-cat-pink)]"),
    ("text-gray-400", "text-[var(--oo-fg-tertiary)]"),

    # Background colors
    ("bg-red-950", "bg-[var(--oo-error-bg)]"),
    ("bg-red-900/30", "bg-[var(--oo-error-bg)]"),
    ("bg-red-900", "bg-[var(--oo-error-bg)]"),
    ("bg-red-800", "bg-[var(--oo-error-bg)]"),
    ("bg-red-700", "bg-[var(--oo-error)]"),
    ("bg-red-600/20", "bg-[var(--oo-error-bg)]"),
    ("bg-red-600", "bg-[var(--oo-error)]"),
    ("bg-red-500", "bg-[var(--oo-error)]"),
    ("bg-red-400", "bg-[var(--oo-error)]"),
    ("bg-green-900/30", "bg-[var(--oo-success-bg)]"),
    ("bg-green-900", "bg-[var(--oo-success-bg)]"),
    ("bg-green-500", "bg-[var(--oo-success)]"),
    ("bg-emerald-950", "bg-[var(--oo-success-bg)]"),
    ("bg-emerald-500", "bg-[var(--oo-success)]"),
    ("bg-emerald-400", "bg-[var(--oo-success)]"),
    ("bg-amber-900/30", "bg-[var(--oo-warning-bg)]"),
    ("bg-amber-500/10", "bg-[var(--oo-warning-bg)]"),
    ("bg-amber-500", "bg-[var(--oo-warning)]"),
    ("bg-amber-400", "bg-[var(--oo-warning)]"),
    ("bg-yellow-900", "bg-[var(--oo-warning-bg)]"),
    ("bg-yellow-500", "bg-[var(--oo-warning)]"),
    ("bg-blue-500/70", "bg-[var(--oo-info)]"),
    ("bg-blue-500", "bg-[var(--oo-info)]"),
    ("bg-blue-400", "bg-[var(--oo-info)]"),
    ("bg-purple-500/70", "bg-[var(--oo-cat-purple)]"),
    ("bg-purple-500", "bg-[var(--oo-cat-purple)]"),
    ("bg-emerald-500/70", "bg-[var(--oo-success)]"),
    ("bg-orange-500", "bg-[var(--oo-cat-orange)]"),
    ("bg-pink-500", "bg-[var(--oo-cat-pink)]"),
    ("bg-gray-500", "bg-[var(--oo-fg-muted)]"),

    # Border colors
    ("border-red-800/40", "border-[var(--oo-error-bd)]"),
    ("border-red-800", "border-[var(--oo-error-bd)]"),
    ("border-red-700", "border-[var(--oo-error-bd)]"),
    ("border-green-800/40", "border-[var(--oo-success-bd)]"),
    ("border-green-800", "border-[var(--oo-success-bd)]"),
    ("border-green-700", "border-[var(--oo-success-bd)]"),
    ("border-green-500", "border-[var(--oo-success-bd)]"),
    ("border-emerald-800", "border-[var(--oo-success-bd)]"),
    ("border-amber-800/40", "border-[var(--oo-warning-bd)]"),
    ("border-amber-500/20", "border-[var(--oo-warning-bd)]"),
    ("border-amber-500", "border-[var(--oo-warning-bd)]"),
    ("border-yellow-700", "border-[var(--oo-warning-bd)]"),
]

# Special patterns that need regex replacement
REGEX_SWAPS: list[tuple[str, str]] = [
    # ConsensusPanel: tierColor returns Tailwind classes
    (r"return 'text-green-400'", "return 'text-[var(--oo-success)]'"),
    (r"return 'text-red-400'", "return 'text-[var(--oo-error)]'"),
    (r"return 'text-amber-400'", "return 'text-[var(--oo-warning)]'"),
    (r"return 'bg-green-500'", "return 'bg-[var(--oo-success)]'"),
    (r"return 'bg-amber-500'", "return 'bg-[var(--oo-warning)]'"),
    (r"return 'bg-red-500'", "return 'bg-[var(--oo-error)]'"),
]


def fix_file(filepath: Path) -> int:
    """Apply all color fixes to a single file. Returns count of changes."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0

    original = content
    changes = 0

    # Apply inline replacements
    for old, new in INLINE_REPLACEMENTS:
        if old in content:
            content = content.replace(old, new)

    # Apply bulk class swaps (order matters: longer patterns first)
    sorted_swaps = sorted(BULK_CLASS_SWAPS, key=lambda x: -len(x[0]))
    for old_cls, new_cls in sorted_swaps:
        if old_cls in content:
            content = content.replace(old_cls, new_cls)

    # Apply regex swaps
    for pattern, replacement in REGEX_SWAPS:
        content = re.sub(re.escape(pattern), replacement, content)

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        changes = 1

    return changes


def main() -> None:
    """Fix all Svelte files."""
    svelte_files = sorted(FRONTEND_SRC.rglob("*.svelte"))
    total_changed = 0

    for filepath in svelte_files:
        changed = fix_file(filepath)
        if changed:
            short = str(filepath).split("frontend/src/")[-1]
            print(f"  Fixed: {short}")
            total_changed += 1

    print(f"\nTotal files modified: {total_changed}")


if __name__ == "__main__":
    main()
