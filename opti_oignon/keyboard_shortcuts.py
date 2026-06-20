#!/usr/bin/env python3
"""
Keyboard shortcuts registry and management.

Provides a centralized registry of keyboard shortcuts with:
- Default shortcut definitions
- Conflict detection between shortcuts
- Custom binding validation and serialization
- Reset-to-defaults support
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)

# Valid modifier keys
VALID_MODIFIERS = frozenset({"ctrl", "shift", "alt", "meta"})

# Valid special key names (non-printable)
VALID_SPECIAL_KEYS = frozenset({
    "enter", "escape", "tab", "space", "backspace", "delete",
    "arrowup", "arrowdown", "arrowleft", "arrowright",
    "home", "end", "pageup", "pagedown",
    "f1", "f2", "f3", "f4", "f5", "f6",
    "f7", "f8", "f9", "f10", "f11", "f12",
})

# Keys that commonly conflict with browser defaults
BROWSER_CONFLICT_KEYS: dict[str, str] = {
    "ctrl+t": "browser new tab",
    "ctrl+w": "browser close tab",
    "ctrl+n": "browser new window",
    "ctrl+shift+n": "browser incognito window",
    "ctrl+tab": "browser switch tab",
    "ctrl+shift+tab": "browser switch tab reverse",
    "ctrl+l": "browser address bar",
    "ctrl+d": "browser bookmark",
    "ctrl+h": "browser history",
    "ctrl+j": "browser downloads",
    "ctrl+p": "browser print",
    "ctrl+s": "browser save page",
    "ctrl+f": "browser find",
    "ctrl+g": "browser find next",
    "ctrl+r": "browser reload",
    "ctrl+shift+i": "browser dev tools",
    "ctrl+shift+j": "browser console",
    "ctrl+u": "browser view source",
    "f5": "browser reload",
    "f11": "browser fullscreen",
    "f12": "browser dev tools",
}

# Maximum custom bindings allowed
MAX_CUSTOM_BINDINGS = 50


@dataclass
class ShortcutBinding:
    """A single keyboard shortcut binding."""

    action: str
    key: str
    ctrl: bool = False
    shift: bool = False
    alt: bool = False
    meta: bool = False
    description: str = ""
    category: str = "general"

    def combo_string(self) -> str:
        """Return normalized combo string like 'ctrl+shift+t'."""
        parts: list[str] = []
        if self.ctrl:
            parts.append("ctrl")
        if self.shift:
            parts.append("shift")
        if self.alt:
            parts.append("alt")
        if self.meta:
            parts.append("meta")
        parts.append(self.key.lower())
        return "+".join(parts)

    def display_string(self) -> str:
        """Return human-readable display string like 'Ctrl + Shift + T'."""
        parts: list[str] = []
        if self.ctrl:
            parts.append("Ctrl")
        if self.shift:
            parts.append("Shift")
        if self.alt:
            parts.append("Alt")
        if self.meta:
            parts.append("Meta")
        key_display = self.key
        if len(key_display) == 1:
            key_display = key_display.upper()
        elif key_display == "enter":
            key_display = "Enter"
        elif key_display == "escape":
            key_display = "Esc"
        elif key_display == ",":
            key_display = ","
        elif key_display == "?":
            key_display = "?"
        parts.append(key_display)
        return " + ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShortcutBinding:
        """Deserialize from dictionary."""
        return cls(
            action=str(data.get("action", "")),
            key=str(data.get("key", "")),
            ctrl=bool(data.get("ctrl", False)),
            shift=bool(data.get("shift", False)),
            alt=bool(data.get("alt", False)),
            meta=bool(data.get("meta", False)),
            description=str(data.get("description", "")),
            category=str(data.get("category", "general")),
        )


# Default shortcuts registry
DEFAULT_SHORTCUTS: list[ShortcutBinding] = [
    ShortcutBinding(
        action="new_chat",
        key="n",
        ctrl=True,
        description="New conversation",
        category="navigation",
    ),
    ShortcutBinding(
        action="send_message",
        key="enter",
        ctrl=True,
        description="Send message",
        category="chat",
    ),
    ShortcutBinding(
        action="toggle_sidebar",
        key="b",
        ctrl=True,
        description="Toggle sidebar",
        category="navigation",
    ),
    ShortcutBinding(
        action="search_conversations",
        key="k",
        ctrl=True,
        description="Search conversations",
        category="navigation",
    ),
    ShortcutBinding(
        action="open_settings",
        key=",",
        ctrl=True,
        description="Open settings",
        category="navigation",
    ),
    ShortcutBinding(
        action="toggle_theme",
        key="t",
        ctrl=True,
        shift=True,
        description="Toggle theme",
        category="ui",
    ),
    ShortcutBinding(
        action="export_conversation",
        key="e",
        ctrl=True,
        shift=True,
        description="Export conversation",
        category="chat",
    ),
    ShortcutBinding(
        action="show_shortcuts",
        key="?",
        description="Show keyboard shortcuts",
        category="help",
    ),
    ShortcutBinding(
        action="close_dialog",
        key="escape",
        description="Close dialog or panel",
        category="ui",
    ),
]

# Build default map keyed by action
DEFAULT_SHORTCUTS_MAP: dict[str, ShortcutBinding] = {
    s.action: s for s in DEFAULT_SHORTCUTS
}


def get_default_shortcuts() -> list[dict[str, Any]]:
    """Return default shortcuts as serializable list."""
    return [s.to_dict() for s in DEFAULT_SHORTCUTS]


def get_default_shortcuts_map() -> dict[str, dict[str, Any]]:
    """Return default shortcuts as action -> binding map."""
    return {action: s.to_dict() for action, s in DEFAULT_SHORTCUTS_MAP.items()}


class ShortcutRegistry:
    """
    Centralized keyboard shortcut registry.

    Manages active shortcuts, detects conflicts, and supports
    custom binding overrides with reset-to-defaults.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, ShortcutBinding] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Load all default shortcuts into the registry."""
        self._bindings.clear()
        for shortcut in DEFAULT_SHORTCUTS:
            self._bindings[shortcut.action] = ShortcutBinding(
                action=shortcut.action,
                key=shortcut.key,
                ctrl=shortcut.ctrl,
                shift=shortcut.shift,
                alt=shortcut.alt,
                meta=shortcut.meta,
                description=shortcut.description,
                category=shortcut.category,
            )

    def get(self, action: str) -> ShortcutBinding | None:
        """Get binding for a given action."""
        return self._bindings.get(action)

    def get_all(self) -> dict[str, ShortcutBinding]:
        """Return all current bindings."""
        return dict(self._bindings)

    def get_all_serialized(self) -> dict[str, dict[str, Any]]:
        """Return all bindings as serializable dicts."""
        return {action: b.to_dict() for action, b in self._bindings.items()}

    def lookup_by_combo(self, combo: str) -> ShortcutBinding | None:
        """Find binding by combo string (e.g. 'ctrl+n')."""
        normalized = combo.strip().lower()
        for binding in self._bindings.values():
            if binding.combo_string() == normalized:
                return binding
        return None

    def register(
        self,
        action: str,
        key: str,
        ctrl: bool = False,
        shift: bool = False,
        alt: bool = False,
        meta: bool = False,
        description: str = "",
        category: str = "general",
    ) -> ShortcutBinding:
        """Register or update a shortcut binding."""
        binding = ShortcutBinding(
            action=action,
            key=key.lower() if len(key) > 1 else key,
            ctrl=ctrl,
            shift=shift,
            alt=alt,
            meta=meta,
            description=description or self._bindings.get(action, ShortcutBinding(action="", key="")).description,
            category=category,
        )
        self._bindings[action] = binding
        return binding

    def unregister(self, action: str) -> bool:
        """Remove a shortcut binding. Returns True if removed."""
        if action in self._bindings:
            del self._bindings[action]
            return True
        return False

    def detect_conflicts(self, exclude_action: str = "") -> list[dict[str, Any]]:
        """
        Detect shortcut combo collisions within the registry.

        Returns list of conflict dicts with 'combo', 'actions' keys.
        Optionally exclude a specific action from conflict detection.
        """
        combo_map: dict[str, list[str]] = {}
        for action, binding in self._bindings.items():
            if action == exclude_action:
                continue
            combo = binding.combo_string()
            if combo not in combo_map:
                combo_map[combo] = []
            combo_map[combo].append(action)

        conflicts: list[dict[str, Any]] = []
        for combo, actions in combo_map.items():
            if len(actions) > 1:
                conflicts.append({"combo": combo, "actions": actions})
        return conflicts

    def check_browser_conflicts(self) -> list[dict[str, Any]]:
        """
        Check which current bindings conflict with browser defaults.

        Returns list of dicts with 'action', 'combo', 'browser_function' keys.
        """
        warnings: list[dict[str, Any]] = []
        for action, binding in self._bindings.items():
            combo = binding.combo_string()
            if combo in BROWSER_CONFLICT_KEYS:
                warnings.append({
                    "action": action,
                    "combo": combo,
                    "browser_function": BROWSER_CONFLICT_KEYS[combo],
                })
        return warnings

    def reset_action(self, action: str) -> bool:
        """Reset a single action to its default binding."""
        if action in DEFAULT_SHORTCUTS_MAP:
            default = DEFAULT_SHORTCUTS_MAP[action]
            self._bindings[action] = ShortcutBinding(
                action=default.action,
                key=default.key,
                ctrl=default.ctrl,
                shift=default.shift,
                alt=default.alt,
                meta=default.meta,
                description=default.description,
                category=default.category,
            )
            return True
        return False

    def reset_all(self) -> None:
        """Reset all bindings to defaults."""
        self._load_defaults()

    def apply_custom_bindings(self, custom: dict[str, dict[str, Any]]) -> list[str]:
        """
        Apply custom binding overrides on top of defaults.

        Returns list of warning messages (e.g. unknown actions).
        """
        warnings: list[str] = []
        for action, binding_data in custom.items():
            if action not in DEFAULT_SHORTCUTS_MAP:
                warnings.append(f"Unknown action: {action}")
                continue
            validated = validate_binding(binding_data)
            if validated is None:
                warnings.append(f"Invalid binding for action: {action}")
                continue
            # Preserve description and category from defaults
            default = DEFAULT_SHORTCUTS_MAP[action]
            self._bindings[action] = ShortcutBinding(
                action=action,
                key=validated["key"],
                ctrl=validated.get("ctrl", False),
                shift=validated.get("shift", False),
                alt=validated.get("alt", False),
                meta=validated.get("meta", False),
                description=default.description,
                category=default.category,
            )
        return warnings

    def export_custom_diff(self) -> dict[str, dict[str, Any]]:
        """
        Export only bindings that differ from defaults.

        Returns action -> binding dict for changed bindings only.
        """
        diff: dict[str, dict[str, Any]] = {}
        for action, binding in self._bindings.items():
            default = DEFAULT_SHORTCUTS_MAP.get(action)
            if default is None:
                diff[action] = binding.to_dict()
                continue
            if binding.combo_string() != default.combo_string():
                diff[action] = {
                    "key": binding.key,
                    "ctrl": binding.ctrl,
                    "shift": binding.shift,
                    "alt": binding.alt,
                    "meta": binding.meta,
                }
        return diff


def validate_key(key: str) -> bool:
    """Validate that a key string is acceptable."""
    if not key or not isinstance(key, str):
        return False
    k = key.lower().strip()
    # Single character keys
    if len(k) == 1:
        return True
    # Special keys
    if k in VALID_SPECIAL_KEYS:
        return True
    # Punctuation that comes as-is
    if k in {",", ".", "/", ";", "'", "[", "]", "\\", "-", "=", "`", "?"}:
        return True
    return False


def validate_binding(data: dict[str, Any]) -> dict[str, Any] | None:
    """
    Validate a binding dict. Returns normalized dict or None if invalid.

    Expected keys: key (required), ctrl, shift, alt, meta (optional bools).
    """
    if not isinstance(data, dict):
        return None
    key = data.get("key")
    if not key or not isinstance(key, str):
        return None
    key_normalized = key.lower().strip() if len(key) > 1 else key
    if not validate_key(key_normalized):
        return None
    return {
        "key": key_normalized,
        "ctrl": bool(data.get("ctrl", False)),
        "shift": bool(data.get("shift", False)),
        "alt": bool(data.get("alt", False)),
        "meta": bool(data.get("meta", False)),
    }


def validate_custom_bindings(custom: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate a full custom bindings payload.

    Returns (is_valid, list_of_errors).
    """
    errors: list[str] = []
    if not isinstance(custom, dict):
        return False, ["Custom bindings must be a dictionary"]

    if len(custom) > MAX_CUSTOM_BINDINGS:
        errors.append(
            f"Too many custom bindings ({len(custom)}), max {MAX_CUSTOM_BINDINGS}"
        )

    for action, binding_data in custom.items():
        if not isinstance(action, str) or not action.strip():
            errors.append(f"Invalid action key: {action!r}")
            continue
        if action not in DEFAULT_SHORTCUTS_MAP:
            errors.append(f"Unknown action: {action}")
            continue
        if validate_binding(binding_data) is None:
            errors.append(f"Invalid binding for action: {action}")

    return len(errors) == 0, errors


def parse_combo_string(combo: str) -> dict[str, Any] | None:
    """
    Parse a combo string like 'ctrl+shift+t' into a binding dict.

    Returns None if the combo is invalid.
    """
    if not combo or not isinstance(combo, str):
        return None
    parts = [p.strip().lower() for p in combo.split("+")]
    if not parts:
        return None

    key = parts[-1]
    modifiers = set(parts[:-1])

    # Validate modifiers
    if not modifiers.issubset(VALID_MODIFIERS):
        return None

    if not validate_key(key):
        return None

    return {
        "key": key,
        "ctrl": "ctrl" in modifiers,
        "shift": "shift" in modifiers,
        "alt": "alt" in modifiers,
        "meta": "meta" in modifiers,
    }


def check_combo_browser_conflict(combo: str) -> str | None:
    """
    Check if a combo string conflicts with browser defaults.

    Returns the browser function name if conflict, None otherwise.
    """
    normalized = combo.strip().lower()
    return BROWSER_CONFLICT_KEYS.get(normalized)
