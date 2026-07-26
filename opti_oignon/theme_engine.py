#!/usr/bin/env python3
"""
Theme engine for Opti-Oignon.

Generates CSS custom property sets from user-chosen accent colors (HSL).
Supports warmth offset (hue shift), lightness offset, and saturation control.
Validates WCAG AA contrast ratios. Provides built-in preset themes
and manages user-defined custom presets.

All generated variables follow the --oo-* convention and are a strict
superset of the accent variables defined in theme.css.
"""

import colorsys
import json
import re
from typing import Any

# -- HSL / Hex conversion utilities --

def hsl_to_hex(h: float, s: float, l: float) -> str:
    """Convert HSL (h: 0-360, s: 0-100, l: 0-100) to hex string."""
    h_norm = (h % 360) / 360.0
    s_norm = max(0.0, min(100.0, s)) / 100.0
    l_norm = max(0.0, min(100.0, l)) / 100.0
    r, g, b = colorsys.hls_to_rgb(h_norm, l_norm, s_norm)
    return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to (r, g, b) tuple (0-255)."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def hex_to_hsl(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color to HSL (h: 0-360, s: 0-100, l: 0-100)."""
    r, g, b = hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    return (h * 360.0, s * 100.0, l * 100.0)


def is_valid_hex(color: str) -> bool:
    """Check if a string is a valid hex color (#RGB or #RRGGBB)."""
    return bool(re.match(r"^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$", color))


# -- WCAG contrast ratio --

def _relative_luminance(r: int, g: int, b: int) -> float:
    """Compute relative luminance per WCAG 2.1 definition."""
    def linearize(c: int) -> float:
        srgb = c / 255.0
        if srgb <= 0.04045:
            return srgb / 12.92
        return ((srgb + 0.055) / 1.055) ** 2.4

    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def validate_contrast(fg_hex: str, bg_hex: str) -> float:
    """
    Compute WCAG contrast ratio between two hex colors.

    Returns a float >= 1.0. WCAG AA requires >= 4.5 for normal text,
    >= 3.0 for large text.
    """
    r1, g1, b1 = hex_to_rgb(fg_hex)
    r2, g2, b2 = hex_to_rgb(bg_hex)
    lum1 = _relative_luminance(r1, g1, b1)
    lum2 = _relative_luminance(r2, g2, b2)
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)


def passes_wcag_aa(fg_hex: str, bg_hex: str, large_text: bool = False) -> bool:
    """Check if a color pair meets WCAG AA requirements."""
    ratio = validate_contrast(fg_hex, bg_hex)
    threshold = 3.0 if large_text else 4.5
    return ratio >= threshold


# -- Modifier functions --

def apply_warmth_offset(hue: int, warmth: int) -> int:
    """
    Shift hue towards warm (positive) or cool (negative).

    Warmth range: -30 to +30. Positive pushes towards orange/red,
    negative pushes towards blue/cyan.

    Returns adjusted hue (0-359).
    """
    warmth = max(-30, min(30, warmth))
    return (hue + warmth) % 360


def apply_lightness_offset(base_lightness: float, offset: int) -> float:
    """
    Apply a lightness offset to a base lightness value.

    Offset range: -50 to +50.
    Result clamped to 5-95 to avoid pure black/white.
    """
    offset = max(-50, min(50, offset))
    return max(5.0, min(95.0, base_lightness + offset))


# -- Accent scale generation --

_DARK_LIGHTNESS_STEPS: dict[str, float] = {
    "50": 95.0, "100": 88.0, "200": 75.0, "300": 62.0, "400": 50.0,
    "500": 50.0, "600": 42.0, "700": 34.0, "800": 26.0, "900": 18.0,
}

_LIGHT_LIGHTNESS_STEPS: dict[str, float] = {
    "50": 95.0, "100": 85.0, "200": 70.0, "300": 55.0, "400": 35.0,
    "500": 35.0, "600": 28.0, "700": 22.0, "800": 16.0, "900": 10.0,
}

_SATURATION_FACTORS: dict[str, float] = {
    "50": 0.30, "100": 0.50, "200": 0.70, "300": 0.85, "400": 1.00,
    "500": 1.00, "600": 0.95, "700": 0.85, "800": 0.75, "900": 0.65,
}

SCALE_KEYS = ["50", "100", "200", "300", "400", "500", "600", "700", "800", "900"]


def generate_accent_scale(
    hue: int,
    saturation: int = 70,
    mode: str = "dark",
    lightness_offset: int = 0,
    warmth: int = 0,
) -> dict[str, str]:
    """
    Generate a full accent color scale (50 through 900) from a base HSL hue.

    Args:
        hue: Hue value (0-359).
        saturation: Base saturation (0-100).
        mode: "dark" or "light".
        lightness_offset: Shift all lightness steps (-50 to +50).
        warmth: Hue shift towards warm/cool (-30 to +30).

    Returns:
        Dict mapping scale keys ("50"-"900") to hex color strings.
    """
    effective_hue = apply_warmth_offset(hue % 360, warmth)
    saturation = max(0, min(100, saturation))
    steps = _DARK_LIGHTNESS_STEPS if mode == "dark" else _LIGHT_LIGHTNESS_STEPS

    scale: dict[str, str] = {}
    for key in SCALE_KEYS:
        lightness = apply_lightness_offset(steps[key], lightness_offset)
        sat = saturation * _SATURATION_FACTORS[key]
        scale[key] = hsl_to_hex(effective_hue, sat, lightness)
    return scale


def generate_theme_variables(
    accent_hue: int,
    secondary_hue: int = -1,
    mode: str = "dark",
    accent_saturation: int = 70,
    secondary_saturation: int = 30,
    accent_lightness_offset: int = 0,
    secondary_lightness_offset: int = 0,
    accent_warmth: int = 0,
    secondary_warmth: int = 0,
) -> dict[str, str]:
    """
    Generate a complete set of CSS custom properties for a theme configuration.

    Includes all --oo-acc-* variables (50-900), --oo-accent-primary,
    secondary accent tokens (sage/tobacco), buttons, input focus, bubbles.
    Semantic colors are NOT included (they remain fixed in theme.css).
    """
    if secondary_hue < 0:
        secondary_hue = (accent_hue + 90) % 360

    accent_scale = generate_accent_scale(
        accent_hue, accent_saturation, mode, accent_lightness_offset, accent_warmth
    )
    secondary_scale = generate_accent_scale(
        secondary_hue, secondary_saturation, mode,
        secondary_lightness_offset, secondary_warmth
    )

    variables: dict[str, str] = {}

    for key, value in accent_scale.items():
        variables[f"oo-acc-{key}"] = value
    variables["oo-accent-primary"] = accent_scale["500"]

    variables["oo-sage"] = secondary_scale["500"]
    variables["oo-sage-bg"] = _hex_to_rgba(secondary_scale["500"], 0.12)
    variables["oo-sage-bd"] = _hex_to_rgba(secondary_scale["500"], 0.18)
    variables["oo-pine"] = secondary_scale["600"]

    variables["oo-tobacco"] = accent_scale["500"]
    variables["oo-tobacco-bg"] = _hex_to_rgba(accent_scale["500"], 0.10)
    variables["oo-tobacco-bd"] = _hex_to_rgba(accent_scale["500"], 0.15)

    variables["oo-btn-primary-bg"] = accent_scale["500"]
    if mode == "dark":
        variables["oo-btn-primary-fg"] = "#222224"
    else:
        variables["oo-btn-primary-fg"] = "#F0EBE4"
    variables["oo-btn-primary-hover"] = accent_scale["600"]

    variables["oo-input-focus"] = _hex_to_rgba(accent_scale["500"], 0.35)

    if mode == "dark":
        variables["oo-msg-user-bg"] = _hex_to_rgba(accent_scale["500"], 0.06)
        variables["oo-msg-user-bd"] = _hex_to_rgba(accent_scale["500"], 0.06)
    else:
        variables["oo-msg-user-bg"] = _hex_to_rgba(secondary_scale["500"], 0.14)
        variables["oo-msg-user-bd"] = _hex_to_rgba(secondary_scale["500"], 0.18)

    return variables


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba() string."""
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


# -- Built-in preset themes --
#
# The five built-in presets are renamed to the design-system presets
# (anthracite, parchment, slate, linen, high-contrast). The previous ids
# (default, ocean, forest, rose, monochrome) remain reserved for backward
# compatibility and resolve through _LEGACY_PRESET_ALIASES, so older stored
# selections still map to a valid preset. The curated per-preset foundation
# palettes live in BUILTIN_PRESET_PALETTES (see end of module).

_NEW_PRESET_IDS = frozenset(
    {"anthracite", "parchment", "slate", "linen", "high-contrast"}
)
_LEGACY_PRESET_IDS = frozenset(
    {"default", "ocean", "forest", "rose", "monochrome"}
)
# Reserve both old and new ids against custom-preset collisions.
BUILTIN_PRESET_IDS = _NEW_PRESET_IDS | _LEGACY_PRESET_IDS

# Best-effort migration of a legacy preset id to its closest new preset.
_LEGACY_PRESET_ALIASES: dict[str, str] = {
    "default": "anthracite",
    "ocean": "slate",
    "monochrome": "high-contrast",
    "forest": "anthracite",
    "rose": "anthracite",
}

_PRESET_THEMES: list[dict[str, Any]] = [
    {
        "id": "anthracite",
        "name": "Anthracite",
        "description": "Warm greyer dark with a tobacco accent (default dark)",
        "mode": "dark",
        "accent_hue": 35, "accent_saturation": 70,
        "secondary_hue": 130, "secondary_saturation": 12,
        "accent_lightness_offset": 0, "secondary_lightness_offset": 0,
        "accent_warmth": 0, "secondary_warmth": 0,
        "builtin": True,
    },
    {
        "id": "parchment",
        "name": "Parchment",
        "description": "Deep warm taupe light with a deep-tobacco accent (default light)",
        "mode": "light",
        "accent_hue": 30, "accent_saturation": 60,
        "secondary_hue": 130, "secondary_saturation": 18,
        "accent_lightness_offset": 0, "secondary_lightness_offset": 0,
        "accent_warmth": 0, "secondary_warmth": 0,
        "builtin": True,
    },
    {
        "id": "slate",
        "name": "Slate",
        "description": "Cool dark with a blue accent",
        "mode": "dark",
        "accent_hue": 205, "accent_saturation": 45,
        "secondary_hue": 150, "secondary_saturation": 25,
        "accent_lightness_offset": 0, "secondary_lightness_offset": 0,
        "accent_warmth": 0, "secondary_warmth": 0,
        "builtin": True,
    },
    {
        "id": "linen",
        "name": "Linen",
        "description": "Cool light with a deep-teal accent",
        "mode": "light",
        "accent_hue": 195, "accent_saturation": 45,
        "secondary_hue": 160, "secondary_saturation": 25,
        "accent_lightness_offset": 0, "secondary_lightness_offset": 0,
        "accent_warmth": 0, "secondary_warmth": 0,
        "builtin": True,
    },
    {
        "id": "high-contrast",
        "name": "High Contrast",
        "description": "Accessibility-first dark with maximum contrast",
        "mode": "dark",
        "accent_hue": 35, "accent_saturation": 100,
        "secondary_hue": 130, "secondary_saturation": 60,
        "accent_lightness_offset": 0, "secondary_lightness_offset": 0,
        "accent_warmth": 0, "secondary_warmth": 0,
        "builtin": True,
    },
]


def get_preset_themes() -> list[dict[str, Any]]:
    """Return the list of built-in preset themes."""
    return [preset.copy() for preset in _PRESET_THEMES]


def get_preset_by_id(preset_id: str) -> dict[str, Any] | None:
    """Return a single built-in preset by its id, or None if not found.

    Legacy preset ids (default, ocean, forest, rose, monochrome) are
    resolved through _LEGACY_PRESET_ALIASES to their closest new preset so
    older stored selections keep working after the rename.
    """
    for preset in _PRESET_THEMES:
        if preset["id"] == preset_id:
            return preset.copy()
    aliased = _LEGACY_PRESET_ALIASES.get(preset_id)
    if aliased:
        for preset in _PRESET_THEMES:
            if preset["id"] == aliased:
                return preset.copy()
    return None


# -- Custom preset validation --

_CUSTOM_PRESET_FIELDS = {
    "id", "name", "description",
    "accent_hue", "accent_saturation",
    "secondary_hue", "secondary_saturation",
    "accent_lightness_offset", "secondary_lightness_offset",
    "accent_warmth", "secondary_warmth",
}

MAX_CUSTOM_PRESETS = 20
MAX_PRESET_NAME_LEN = 50
MAX_PRESET_DESC_LEN = 200


def validate_custom_preset(preset: dict[str, Any]) -> list[str]:
    """
    Validate a user-submitted custom preset.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []

    if "name" not in preset or not isinstance(preset.get("name"), str):
        errors.append("name is required and must be a string")
    elif len(preset["name"].strip()) == 0:
        errors.append("name cannot be empty")
    elif len(preset["name"]) > MAX_PRESET_NAME_LEN:
        errors.append(f"name must be at most {MAX_PRESET_NAME_LEN} characters")

    if "description" in preset and isinstance(preset["description"], str):
        if len(preset["description"]) > MAX_PRESET_DESC_LEN:
            errors.append(
                f"description must be at most {MAX_PRESET_DESC_LEN} characters"
            )

    if "id" in preset and isinstance(preset["id"], str):
        if preset["id"] in BUILTIN_PRESET_IDS:
            errors.append(
                f"Cannot use reserved built-in preset id: {preset['id']}"
            )
        if not re.match(r"^[a-zA-Z0-9_-]+$", preset["id"]):
            errors.append("id must contain only alphanumeric, hyphens, underscores")
        if len(preset["id"]) > 50:
            errors.append("id must be at most 50 characters")

    errors.extend(_validate_theme_numerics(preset))
    return errors


def validate_theme_config(config: dict[str, Any]) -> list[str]:
    """
    Validate a user-submitted theme configuration.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []

    if "accent_hue" not in config:
        errors.append("accent_hue is required")

    if "mode" in config:
        if config["mode"] not in ("dark", "light"):
            errors.append("mode must be 'dark' or 'light'")

    errors.extend(_validate_theme_numerics(config))
    return errors


def _validate_theme_numerics(config: dict[str, Any]) -> list[str]:
    """Validate numeric theme fields shared between config and presets."""
    errors: list[str] = []

    for key in ("accent_hue", "secondary_hue"):
        if key in config:
            val = config[key]
            low = -1 if key == "secondary_hue" else 0
            if not isinstance(val, (int, float)) or val < low or val > 359:
                errors.append(f"{key} must be between {low} and 359")

    for key in ("accent_saturation", "secondary_saturation"):
        if key in config:
            val = config[key]
            if not isinstance(val, (int, float)) or val < 0 or val > 100:
                errors.append(f"{key} must be between 0 and 100")

    for key in ("accent_lightness_offset", "secondary_lightness_offset"):
        if key in config:
            val = config[key]
            if not isinstance(val, (int, float)) or val < -50 or val > 50:
                errors.append(f"{key} must be between -50 and 50")

    for key in ("accent_warmth", "secondary_warmth"):
        if key in config:
            val = config[key]
            if not isinstance(val, (int, float)) or val < -30 or val > 30:
                errors.append(f"{key} must be between -30 and 30")

    return errors


# -- Import/export helpers --

def validate_preset_import(data: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Validate imported preset data (from JSON).

    Returns:
        Tuple of (valid_presets, errors).
    """
    errors: list[str] = []

    if not isinstance(data, list):
        return [], ["Import data must be a JSON array of presets"]

    if len(data) > MAX_CUSTOM_PRESETS:
        return [], [f"Cannot import more than {MAX_CUSTOM_PRESETS} presets at once"]

    valid: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Item {i}: must be an object")
            continue
        item_errors = validate_custom_preset(item)
        if item_errors:
            for err in item_errors:
                errors.append(f"Item {i} ({item.get('name', '?')}): {err}")
        else:
            valid.append(item)

    return valid, errors


def export_presets(presets: list[dict[str, Any]]) -> str:
    """Serialize custom presets to JSON for export."""
    exportable = []
    for p in presets:
        entry = {k: v for k, v in p.items() if k in _CUSTOM_PRESET_FIELDS}
        exportable.append(entry)
    return json.dumps(exportable, indent=2, ensure_ascii=False)


# -- Design-system foundation schema --
# Curated per-preset foundation palettes (38 tokens each), the
# single source of truth shared with the split theme-<id>.css files.
BUILTIN_PRESET_PALETTES: dict[str, dict[str, str]] = {
    'anthracite': {
        'bg-base': '#1F1F22',
        'bg-surface': '#27272A',
        'bg-elevated': '#303034',
        'bg-overlay': '#3A3A3E',
        'bg-subtle': '#1A1A1D',
        'bg-hover': '#34343A',
        'bg-active': '#3E3E45',
        'fg-primary': '#ECE9E3',
        'fg-secondary': '#B5B1A9',
        'fg-tertiary': '#928D85',
        'fg-muted': '#8C8881',
        'fg-decorative': '#4A4A4C',
        'fg-on-accent': '#1F1F22',
        'bd-subtle': '#2A2A2D',
        'bd-default': '#2E2E32',
        'bd-strong': '#48484C',
        'acc-50': '#FDF6EE',
        'acc-100': '#F8E8D4',
        'acc-200': '#E8C9A0',
        'acc-300': '#D4AC78',
        'acc-400': '#C48838',
        'acc-500': '#C48838',
        'acc-600': '#A87030',
        'acc-700': '#7D5636',
        'acc-800': '#614328',
        'acc-900': '#4A321E',
        'success': '#86A189',
        'success-bg': 'rgba(134, 161, 137, 0.16)',
        'success-bd': 'rgba(134, 161, 137, 0.24)',
        'warning': '#D2C0A0',
        'warning-bg': 'rgba(210, 192, 160, 0.16)',
        'warning-bd': 'rgba(210, 192, 160, 0.24)',
        'error': '#D88679',
        'error-bg': 'rgba(216, 134, 121, 0.16)',
        'error-bd': 'rgba(216, 134, 121, 0.24)',
        'info': '#8AABBF',
        'info-bg': 'rgba(138, 171, 191, 0.16)',
        'info-bd': 'rgba(138, 171, 191, 0.24)',
    },
    'parchment': {
        'bg-base': '#E5DECE',
        'bg-surface': '#DDD5C3',
        'bg-elevated': '#ECE5D6',
        'bg-overlay': '#D2CAB8',
        'bg-subtle': '#D8D0BE',
        'bg-hover': '#D6CEBC',
        'bg-active': '#CFC6B3',
        'fg-primary': '#2D2C2A',
        'fg-secondary': '#4D4842',
        'fg-tertiary': '#555049',
        'fg-muted': '#6E675E',
        'fg-decorative': '#B8B1A4',
        'fg-on-accent': '#F0EBE4',
        'bd-subtle': '#CFC8BC',
        'bd-default': '#BEB7AB',
        'bd-strong': '#A8A298',
        'acc-50': '#F5EDE4',
        'acc-100': '#EAD8C4',
        'acc-200': '#D4B48A',
        'acc-300': '#B8925E',
        'acc-400': '#7A4E1E',
        'acc-500': '#7A4E1E',
        'acc-600': '#5E3C15',
        'acc-700': '#4E3012',
        'acc-800': '#3D250E',
        'acc-900': '#2E1B0A',
        'success': '#3D5240',
        'success-bg': 'rgba(61, 82, 64, 0.12)',
        'success-bd': 'rgba(61, 82, 64, 0.20)',
        'warning': '#5E5030',
        'warning-bg': 'rgba(94, 80, 48, 0.12)',
        'warning-bd': 'rgba(94, 80, 48, 0.20)',
        'error': '#7A3E34',
        'error-bg': 'rgba(122, 62, 52, 0.12)',
        'error-bd': 'rgba(122, 62, 52, 0.20)',
        'info': '#2E5A6E',
        'info-bg': 'rgba(46, 90, 110, 0.12)',
        'info-bd': 'rgba(46, 90, 110, 0.20)',
    },
    'slate': {
        'bg-base': '#1A1E24',
        'bg-surface': '#22272F',
        'bg-elevated': '#2B313A',
        'bg-overlay': '#353C46',
        'bg-subtle': '#15181D',
        'bg-hover': '#2F3640',
        'bg-active': '#39414C',
        'fg-primary': '#E4E8EC',
        'fg-secondary': '#B0B8C1',
        'fg-tertiary': '#8A929C',
        'fg-muted': '#828B95',
        'fg-decorative': '#444A52',
        'fg-on-accent': '#10141A',
        'bd-subtle': '#242A32',
        'bd-default': '#2A313A',
        'bd-strong': '#444C56',
        'acc-50': '#ECF3F7',
        'acc-100': '#DAE6F0',
        'acc-200': '#C3D7E6',
        'acc-300': '#A9C6DB',
        'acc-400': '#8EB4D0',
        'acc-500': '#7AA7C8',
        'acc-600': '#668CA8',
        'acc-700': '#537288',
        'acc-800': '#3F5768',
        'acc-900': '#2E3F4C',
        'success': '#7AB69A',
        'success-bg': 'rgba(122, 182, 154, 0.16)',
        'success-bd': 'rgba(122, 182, 154, 0.24)',
        'warning': '#D4B98F',
        'warning-bg': 'rgba(212, 185, 143, 0.16)',
        'warning-bd': 'rgba(212, 185, 143, 0.24)',
        'error': '#D88679',
        'error-bg': 'rgba(216, 134, 121, 0.16)',
        'error-bd': 'rgba(216, 134, 121, 0.24)',
        'info': '#8AB6D0',
        'info-bg': 'rgba(138, 182, 208, 0.16)',
        'info-bd': 'rgba(138, 182, 208, 0.24)',
    },
    'linen': {
        'bg-base': '#EAEBE7',
        'bg-surface': '#E3E4E0',
        'bg-elevated': '#F0F1EE',
        'bg-overlay': '#D9DAD5',
        'bg-subtle': '#DEDFDA',
        'bg-hover': '#DCDDD8',
        'bg-active': '#D3D4CE',
        'fg-primary': '#22272D',
        'fg-secondary': '#454C54',
        'fg-tertiary': '#4C5157',
        'fg-muted': '#666D75',
        'fg-decorative': '#B0B4AD',
        'fg-on-accent': '#F0F1EE',
        'bd-subtle': '#D2D3CD',
        'bd-default': '#C2C3BD',
        'bd-strong': '#A8A9A2',
        'acc-50': '#E2E8EB',
        'acc-100': '#C4D1D6',
        'acc-200': '#A1B5BE',
        'acc-300': '#7794A1',
        'acc-400': '#4D7384',
        'acc-500': '#2E5A6E',
        'acc-600': '#274C5C',
        'acc-700': '#1F3D4B',
        'acc-800': '#182F39',
        'acc-900': '#11222A',
        'success': '#2E6858',
        'success-bg': 'rgba(46, 104, 88, 0.12)',
        'success-bd': 'rgba(46, 104, 88, 0.20)',
        'warning': '#5E5030',
        'warning-bg': 'rgba(94, 80, 48, 0.12)',
        'warning-bd': 'rgba(94, 80, 48, 0.20)',
        'error': '#7A3E34',
        'error-bg': 'rgba(122, 62, 52, 0.12)',
        'error-bd': 'rgba(122, 62, 52, 0.20)',
        'info': '#2E5A6E',
        'info-bg': 'rgba(46, 90, 110, 0.12)',
        'info-bd': 'rgba(46, 90, 110, 0.20)',
    },
    'high-contrast': {
        'bg-base': '#000000',
        'bg-surface': '#0A0A0A',
        'bg-elevated': '#141414',
        'bg-overlay': '#1E1E1E',
        'bg-subtle': '#000000',
        'bg-hover': '#1A1A1A',
        'bg-active': '#242424',
        'fg-primary': '#FFFFFF',
        'fg-secondary': '#D6D6D6',
        'fg-tertiary': '#B8B8B8',
        'fg-muted': '#A8A8A8',
        'fg-decorative': '#6A6A6A',
        'fg-on-accent': '#000000',
        'bd-subtle': '#2A2A2A',
        'bd-default': '#3A3A3A',
        'bd-strong': '#6A6A6A',
        'acc-50': '#FFF4E5',
        'acc-100': '#FFEACB',
        'acc-200': '#FFDDAC',
        'acc-300': '#FFCE87',
        'acc-400': '#FFBE63',
        'acc-500': '#FFB347',
        'acc-600': '#D6963C',
        'acc-700': '#AD7A30',
        'acc-800': '#855D25',
        'acc-900': '#61441B',
        'success': '#7CE07C',
        'success-bg': 'rgba(124, 224, 124, 0.16)',
        'success-bd': 'rgba(124, 224, 124, 0.24)',
        'warning': '#FFD27A',
        'warning-bg': 'rgba(255, 210, 122, 0.16)',
        'warning-bd': 'rgba(255, 210, 122, 0.24)',
        'error': '#FF8A7A',
        'error-bg': 'rgba(255, 138, 122, 0.16)',
        'error-bd': 'rgba(255, 138, 122, 0.24)',
        'info': '#8FD0FF',
        'info-bg': 'rgba(143, 208, 255, 0.16)',
        'info-bd': 'rgba(143, 208, 255, 0.24)',
    },
}

BUILTIN_PRESET_MODES: dict[str, str] = {'anthracite': 'dark', 'parchment': 'light', 'slate': 'dark', 'linen': 'light', 'high-contrast': 'dark'}

# Canonical token pairs checked for WCAG AA. (fg, bg, required_ratio).
# Body-text pairs require 4.5; accent/large-element pairs require 3.0.
_CANONICAL_PAIRS: list[tuple[str, str, float]] = [
    ("fg-primary", "bg-base", 4.5),
    ("fg-primary", "bg-surface", 4.5),
    ("fg-secondary", "bg-base", 4.5),
    ("fg-secondary", "bg-surface", 4.5),
    ("acc-500", "bg-base", 3.0),
    ("fg-on-accent", "acc-500", 3.0),
]


def get_preset_palette(preset_id: str) -> dict[str, str] | None:
    """Return the curated 38-token foundation palette for a built-in preset.

    Legacy ids resolve through the same alias map as get_preset_by_id.
    """
    if preset_id in BUILTIN_PRESET_PALETTES:
        return dict(BUILTIN_PRESET_PALETTES[preset_id])
    aliased = _LEGACY_PRESET_ALIASES.get(preset_id)
    if aliased and aliased in BUILTIN_PRESET_PALETTES:
        return dict(BUILTIN_PRESET_PALETTES[aliased])
    return None


def generate_foundation_schema(
    accent_hue: int,
    mode: str = "dark",
    accent_saturation: int = 70,
    accent_lightness_offset: int = 0,
    accent_warmth: int = 0,
    base_preset: str | None = None,
) -> dict[str, str]:
    """Emit the full foundation schema (--oo-* tokens) for a custom preset.

    Background, foreground, border and semantic tokens are inherited from a
    mode-appropriate base preset (anthracite for dark, parchment for light);
    the accent ramp (--oo-acc-50..900) is generated from the chosen hue.
    Returns variable names WITHOUT the leading '--' (matching the convention
    used by generate_theme_variables).
    """
    if mode not in ("dark", "light"):
        raise ValueError("mode must be 'dark' or 'light'")
    if base_preset is None:
        base_preset = "anthracite" if mode == "dark" else "parchment"
    base = BUILTIN_PRESET_PALETTES.get(base_preset, BUILTIN_PRESET_PALETTES["anthracite"])

    out: dict[str, str] = {}
    # Inherit non-accent foundation tokens from the base preset.
    for token, value in base.items():
        if token.startswith("acc-"):
            continue
        out[f"oo-{token}"] = value
    # Generate the accent ramp from the requested hue.
    scale = generate_accent_scale(
        accent_hue, accent_saturation, mode, accent_lightness_offset, accent_warmth
    )
    for key, value in scale.items():
        out[f"oo-acc-{key}"] = value
    out["oo-accent-primary"] = scale["500"]
    return out


def preset_contrast_report(preset_id: str) -> dict[str, dict[str, object]]:
    """Compute WCAG ratios for the canonical token pairs of a built-in preset.

    Returns {pair_label: {"ratio": float, "required": float, "passes": bool}}.
    """
    palette = get_preset_palette(preset_id)
    if palette is None:
        raise KeyError(f"Unknown preset: {preset_id}")
    report: dict[str, dict[str, object]] = {}
    for fg_tok, bg_tok, required in _CANONICAL_PAIRS:
        fg, bg = palette.get(fg_tok), palette.get(bg_tok)
        if not (fg and bg and is_valid_hex(fg) and is_valid_hex(bg)):
            continue
        ratio = validate_contrast(fg, bg)
        report[f"{fg_tok}|{bg_tok}"] = {
            "ratio": round(ratio, 2),
            "required": required,
            "passes": ratio >= required,
        }
    return report


def validate_builtin_presets() -> dict[str, bool]:
    """Return {preset_id: all_canonical_pairs_pass} for the 5 built-in presets."""
    result: dict[str, bool] = {}
    for pid in BUILTIN_PRESET_PALETTES:
        report = preset_contrast_report(pid)
        result[pid] = all(p["passes"] for p in report.values())
    return result
