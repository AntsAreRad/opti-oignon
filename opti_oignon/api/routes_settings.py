#!/usr/bin/env python3
"""
Routes API pour la gestion de la configuration.

Endpoints pour lire/ecrire les preferences utilisateur
et recharger la configuration depuis le disque.
"""

import logging

from fastapi import APIRouter, HTTPException

from .deps import CONFIG_AVAILABLE, config
from .schemas import (
    CustomPresetCreateRequest,
    CustomPresetImportRequest,
    CustomPresetsExportResponse,
    KeyboardShortcutsResponse,
    KeyboardShortcutsUpdateRequest,
    KeyboardShortcutsUpdateResponse,
    SettingSetRequest,
    SettingsResponse,
    SettingValue,
    ThemeConfigRequest,
    ThemeConfigResponse,
    ThemePresetResponse,
    ThemePresetsListResponse,
)

# Import theme engine via importlib to avoid triggering __init__.py chain
import importlib.util as _ilu
import os as _os
import sys as _sys

_te_path = _os.path.join(
    _os.path.dirname(_os.path.dirname(__file__)), "theme_engine.py"
)
_te_spec = _ilu.spec_from_file_location("theme_engine", _te_path)
_te_mod = _ilu.module_from_spec(_te_spec)
_sys.modules["theme_engine"] = _te_mod  # Python 3.13: register before exec_module for dataclass safety
_te_spec.loader.exec_module(_te_mod)

generate_theme_variables = _te_mod.generate_theme_variables
get_preset_themes = _te_mod.get_preset_themes
get_preset_by_id = _te_mod.get_preset_by_id
validate_theme_config = _te_mod.validate_theme_config
validate_custom_preset = _te_mod.validate_custom_preset
validate_preset_import = _te_mod.validate_preset_import
export_presets = _te_mod.export_presets
BUILTIN_PRESET_IDS = _te_mod.BUILTIN_PRESET_IDS
MAX_CUSTOM_PRESETS = _te_mod.MAX_CUSTOM_PRESETS

# Import keyboard_shortcuts via importlib to avoid triggering __init__.py chain
_ks_path = _os.path.join(
    _os.path.dirname(_os.path.dirname(__file__)), "keyboard_shortcuts.py"
)
_ks_spec = _ilu.spec_from_file_location("keyboard_shortcuts", _ks_path)
_ks_mod = _ilu.module_from_spec(_ks_spec)
_sys.modules["keyboard_shortcuts"] = _ks_mod  # Python 3.13: @dataclass needs sys.modules entry
_ks_spec.loader.exec_module(_ks_mod)

ShortcutRegistry = _ks_mod.ShortcutRegistry
validate_custom_bindings = _ks_mod.validate_custom_bindings
get_default_shortcuts = _ks_mod.get_default_shortcuts
get_default_shortcuts_map = _ks_mod.get_default_shortcuts_map

KEYBOARD_SHORTCUTS_KEY = "keyboard_shortcuts_custom"

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _check_available():
    """Check that the config module is available."""
    if not CONFIG_AVAILABLE or config is None:
        raise HTTPException(
            status_code=503,
            detail="Config module not available",
        )


@router.get("", response_model=SettingsResponse)
def get_settings() -> dict:
    """Retrieve the complete configuration."""
    _check_available()

    data = config.as_dict()
    return SettingsResponse(
        models=data.get("models", {}),
        presets=data.get("presets", {}),
        user=data.get("user", {}),
    )


# -- Theme endpoints (S152) --
# These MUST be defined before the /{key} catch-all route.

THEME_PREF_KEY = "theme_config"
CUSTOM_PRESETS_KEY = "custom_theme_presets"


def _build_theme_variables(cfg: dict) -> dict[str, str]:
    """Build CSS variables dict from a theme config dict."""
    return generate_theme_variables(
        cfg.get("accent_hue", 35),
        cfg.get("secondary_hue", -1),
        cfg.get("mode", "dark"),
        cfg.get("accent_saturation", 70),
        cfg.get("secondary_saturation", 30),
        cfg.get("accent_lightness_offset", 0),
        cfg.get("secondary_lightness_offset", 0),
        cfg.get("accent_warmth", 0),
        cfg.get("secondary_warmth", 0),
    )


def _theme_response(cfg: dict, variables: dict[str, str]) -> ThemeConfigResponse:
    """Build a ThemeConfigResponse from a config dict."""
    return ThemeConfigResponse(
        accent_hue=cfg.get("accent_hue", 35),
        accent_saturation=cfg.get("accent_saturation", 70),
        secondary_hue=cfg.get("secondary_hue", 130),
        secondary_saturation=cfg.get("secondary_saturation", 12),
        accent_lightness_offset=cfg.get("accent_lightness_offset", 0),
        secondary_lightness_offset=cfg.get("secondary_lightness_offset", 0),
        accent_warmth=cfg.get("accent_warmth", 0),
        secondary_warmth=cfg.get("secondary_warmth", 0),
        mode=cfg.get("mode", "dark"),
        preset_id=cfg.get("preset_id"),
        variables=variables,
    )


def _load_custom_presets() -> list[dict]:
    """Load custom presets from user preferences."""
    _check_available()
    saved = config.get_user_preference(CUSTOM_PRESETS_KEY)
    if saved and isinstance(saved, list):
        return saved
    return []


def _save_custom_presets(presets: list[dict]) -> bool:
    """Persist custom presets to user preferences."""
    return config.set_user_preference(CUSTOM_PRESETS_KEY, presets)


@router.get("/theme/presets", response_model=ThemePresetsListResponse)
def get_theme_presets() -> ThemePresetsListResponse:
    """List all theme presets (built-in + custom)."""
    builtin = get_preset_themes()
    custom = []
    try:
        custom = _load_custom_presets()
    except Exception:
        pass  # Config unavailable, return builtins only
    all_presets = builtin + custom
    presets = [ThemePresetResponse(**p) for p in all_presets]
    return ThemePresetsListResponse(presets=presets)


@router.get("/theme", response_model=ThemeConfigResponse)
def get_theme() -> ThemeConfigResponse:
    """Load the current user theme configuration."""
    _check_available()

    saved = config.get_user_preference(THEME_PREF_KEY)
    if saved and isinstance(saved, dict):
        cfg = saved
    else:
        cfg = {
            "accent_hue": 35, "accent_saturation": 70,
            "secondary_hue": 130, "secondary_saturation": 12,
            "accent_lightness_offset": 0, "secondary_lightness_offset": 0,
            "accent_warmth": 0, "secondary_warmth": 0,
            "mode": "dark", "preset_id": "default",
        }

    variables = _build_theme_variables(cfg)
    return _theme_response(cfg, variables)


@router.post("/theme", response_model=ThemeConfigResponse)
def save_theme(request: ThemeConfigRequest) -> ThemeConfigResponse:
    """Save user theme configuration and return generated CSS variables."""
    _check_available()

    config_dict = request.model_dump()
    errors = validate_theme_config(config_dict)
    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))

    # If a preset id is given, load that preset's values
    if request.preset_id:
        # Check built-in first, then custom
        preset = get_preset_by_id(request.preset_id)
        if preset is None:
            custom = _load_custom_presets()
            for cp in custom:
                if cp.get("id") == request.preset_id:
                    preset = cp
                    break
        if preset is None:
            raise HTTPException(
                status_code=404, detail=f"Preset not found: {request.preset_id}"
            )
        for field in (
            "accent_hue", "accent_saturation",
            "secondary_hue", "secondary_saturation",
            "accent_lightness_offset", "secondary_lightness_offset",
            "accent_warmth", "secondary_warmth",
        ):
            if field in preset:
                config_dict[field] = preset[field]

    success = config.set_user_preference(THEME_PREF_KEY, config_dict)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save theme preference")

    variables = _build_theme_variables(config_dict)
    return _theme_response(config_dict, variables)


# -- Custom preset management (S152) --

@router.post("/theme/presets/custom", response_model=ThemePresetResponse)
def create_custom_preset(request: CustomPresetCreateRequest) -> ThemePresetResponse:
    """Create a new custom theme preset from the current configuration."""
    _check_available()

    import uuid as _uuid
    preset_data = request.model_dump()
    preset_data["id"] = f"custom-{_uuid.uuid4().hex[:8]}"
    preset_data["builtin"] = False
    if not preset_data.get("description"):
        preset_data["description"] = ""

    errors = validate_custom_preset(preset_data)
    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))

    existing = _load_custom_presets()
    if len(existing) >= MAX_CUSTOM_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_CUSTOM_PRESETS} custom presets reached"
        )

    existing.append(preset_data)
    if not _save_custom_presets(existing):
        raise HTTPException(status_code=500, detail="Failed to save custom preset")

    return ThemePresetResponse(**preset_data)


@router.delete("/theme/presets/custom/{preset_id}")
def delete_custom_preset(preset_id: str) -> dict:
    """Delete a custom theme preset by id."""
    _check_available()

    if preset_id in BUILTIN_PRESET_IDS:
        raise HTTPException(status_code=400, detail="Cannot delete built-in presets")

    existing = _load_custom_presets()
    filtered = [p for p in existing if p.get("id") != preset_id]
    if len(filtered) == len(existing):
        raise HTTPException(status_code=404, detail=f"Custom preset not found: {preset_id}")

    if not _save_custom_presets(filtered):
        raise HTTPException(status_code=500, detail="Failed to delete custom preset")

    return {"deleted": preset_id}


@router.get("/theme/presets/export", response_model=CustomPresetsExportResponse)
def export_custom_presets() -> CustomPresetsExportResponse:
    """Export all custom presets as downloadable JSON."""
    _check_available()
    custom = _load_custom_presets()
    return CustomPresetsExportResponse(presets_json=export_presets(custom))


@router.post("/theme/presets/import", response_model=ThemePresetsListResponse)
def import_custom_presets(request: CustomPresetImportRequest) -> ThemePresetsListResponse:
    """Import custom presets from JSON, merging with existing."""
    _check_available()

    valid, errors = validate_preset_import(request.presets)
    if errors and not valid:
        raise HTTPException(status_code=422, detail="; ".join(errors))

    existing = _load_custom_presets()
    existing_ids = {p.get("id") for p in existing}

    import uuid as _uuid
    added = []
    for preset in valid:
        if "id" not in preset or preset["id"] in existing_ids:
            preset["id"] = f"custom-{_uuid.uuid4().hex[:8]}"
        preset["builtin"] = False
        added.append(preset)

    total = existing + added
    if len(total) > MAX_CUSTOM_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Import would exceed {MAX_CUSTOM_PRESETS} custom presets limit"
        )

    if not _save_custom_presets(total):
        raise HTTPException(status_code=500, detail="Failed to import presets")

    # Return all presets (built-in + custom)
    all_presets = get_preset_themes() + total
    presets = [ThemePresetResponse(**p) for p in all_presets]
    return ThemePresetsListResponse(presets=presets)


# -- S153: Keyboard shortcuts endpoints --
# These MUST be defined before the /{key} catch-all route.

@router.get("/keyboard_shortcuts", response_model=KeyboardShortcutsResponse)
def get_keyboard_shortcuts() -> KeyboardShortcutsResponse:
    """Get current keyboard shortcuts with any custom overrides applied."""
    _check_available()

    registry = ShortcutRegistry()
    saved = config.get_user_preference(KEYBOARD_SHORTCUTS_KEY)
    custom_overrides: dict = {}
    if isinstance(saved, dict) and saved:
        warnings = registry.apply_custom_bindings(saved)
        if warnings:
            logger.warning("Keyboard shortcuts load warnings: %s", warnings)
        custom_overrides = saved

    return KeyboardShortcutsResponse(
        shortcuts=registry.get_all_serialized(),
        custom_overrides=custom_overrides,
        browser_conflicts=registry.check_browser_conflicts(),
    )


@router.put("/keyboard_shortcuts", response_model=KeyboardShortcutsUpdateResponse)
def update_keyboard_shortcuts(
    request: KeyboardShortcutsUpdateRequest,
) -> KeyboardShortcutsUpdateResponse:
    """Save custom keyboard shortcut bindings."""
    _check_available()

    custom = request.custom_bindings

    # Empty dict means reset to defaults
    if not custom:
        success = config.set_user_preference(KEYBOARD_SHORTCUTS_KEY, {})
        registry = ShortcutRegistry()
        return KeyboardShortcutsUpdateResponse(
            success=success,
            shortcuts=registry.get_all_serialized(),
            custom_overrides={},
            browser_conflicts=registry.check_browser_conflicts(),
            warnings=[],
        )

    # Validate
    is_valid, errors = validate_custom_bindings(custom)
    if not is_valid:
        raise HTTPException(status_code=422, detail="; ".join(errors))

    # Apply and save
    registry = ShortcutRegistry()
    warnings = registry.apply_custom_bindings(custom)
    diff = registry.export_custom_diff()

    success = config.set_user_preference(KEYBOARD_SHORTCUTS_KEY, diff)
    if not success:
        raise HTTPException(
            status_code=500, detail="Failed to save keyboard shortcuts"
        )

    return KeyboardShortcutsUpdateResponse(
        success=True,
        shortcuts=registry.get_all_serialized(),
        custom_overrides=diff,
        browser_conflicts=registry.check_browser_conflicts(),
        warnings=warnings,
    )


@router.get("/{key}", response_model=SettingValue)
def get_setting(key: str) -> dict:
    """Retrieve a specific user preference."""
    _check_available()

    value = config.get_user_preference(key)
    return SettingValue(key=key, value=value)


@router.put("/{key}", response_model=SettingValue)
def set_setting(key: str, request: SettingSetRequest) -> dict:
    """Definit une preference utilisateur."""
    _check_available()

    if not key.strip():
        raise HTTPException(status_code=422, detail="Key cannot be empty")

    success = config.set_user_preference(key, request.value)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save preference")

    return SettingValue(key=key, value=request.value)


@router.post("/reload")
def reload_config() -> dict:
    """Reload configuration from disk."""
    _check_available()

    try:
        config.reload()
        return {"reloaded": True}
    except Exception as e:
        logger.error(f"Config reload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Reload failed: {str(e)}",
        )
