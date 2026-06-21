"""
tests/test_s152_theme_engine.py -- S152 theme engine tests.

Verifies:
- HSL/hex color conversions and validation
- WCAG contrast ratio computation
- Accent scale generation (10 stops, dark/light modes)
- Warmth and lightness offset modifiers
- Full theme variable generation with all parameters
- Built-in preset structure and content
- Custom preset validation (name, id, numeric bounds)
- Import/export validation
- API endpoint schemas existence
- Version bump check
"""

import importlib.util
import json
import os
import re
import sys
import types

# -- Isolation stubs (standard pattern) --
for mod_name in [
    "opti_oignon",
    "opti_oignon.db_utils",
    "opti_oignon.config",
    "opti_oignon.auth",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THEME_ENGINE_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "theme_engine.py")
SCHEMAS_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "api", "schemas.py")
ROUTES_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "api", "routes_settings.py")
VERSION_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "__version__.py")
THEME_CSS_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "styles", "theme.css")
CUSTOMIZER_PATH = os.path.join(
    PROJECT_ROOT, "frontend", "src", "lib", "components", "panels",
    "ThemeCustomizer.svelte",
)
THEME_API_PATH = os.path.join(
    PROJECT_ROOT, "frontend", "src", "lib", "api", "theme.ts",
)


# -- Load theme engine via importlib --

def _load_module():
    spec = importlib.util.spec_from_file_location("theme_engine", THEME_ENGINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


te = _load_module()


def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Class 1: HSL/Hex conversions
# ---------------------------------------------------------------------------

class TestColorConversions:
    """Test color conversion utilities."""

    def test_hsl_to_hex_red(self):
        result = te.hsl_to_hex(0, 100, 50)
        assert result == "#FF0000"

    def test_hsl_to_hex_green(self):
        result = te.hsl_to_hex(120, 100, 50)
        assert result == "#00FF00"

    def test_hsl_to_hex_blue(self):
        result = te.hsl_to_hex(240, 100, 50)
        assert result == "#0000FF"

    def test_hsl_to_hex_black(self):
        result = te.hsl_to_hex(0, 0, 0)
        assert result == "#000000"

    def test_hsl_to_hex_white(self):
        result = te.hsl_to_hex(0, 0, 100)
        assert result == "#FFFFFF"

    def test_hsl_to_hex_wraps_at_360(self):
        a = te.hsl_to_hex(0, 70, 50)
        b = te.hsl_to_hex(360, 70, 50)
        assert a == b

    def test_hex_to_rgb_valid(self):
        assert te.hex_to_rgb("#FF8800") == (255, 136, 0)

    def test_hex_to_rgb_short_form(self):
        assert te.hex_to_rgb("#F80") == (255, 136, 0)

    def test_hex_to_rgb_invalid_raises(self):
        with pytest.raises(ValueError):
            te.hex_to_rgb("nope")

    def test_hex_to_hsl_roundtrip(self):
        original_hex = "#C48838"
        h, s, l = te.hex_to_hsl(original_hex)
        assert 0 <= h <= 360
        assert 0 <= s <= 100
        assert 0 <= l <= 100
        # Roundtrip should be close
        back = te.hsl_to_hex(h, s, l)
        r1, g1, b1 = te.hex_to_rgb(original_hex)
        r2, g2, b2 = te.hex_to_rgb(back)
        assert abs(r1 - r2) <= 1
        assert abs(g1 - g2) <= 1
        assert abs(b1 - b2) <= 1

    def test_is_valid_hex_six_digits(self):
        assert te.is_valid_hex("#C48838") is True

    def test_is_valid_hex_three_digits(self):
        assert te.is_valid_hex("#abc") is True

    def test_is_valid_hex_rejects_invalid(self):
        assert te.is_valid_hex("C48838") is False
        assert te.is_valid_hex("#GGG") is False
        assert te.is_valid_hex("") is False
        assert te.is_valid_hex("#12345") is False


# ---------------------------------------------------------------------------
# Class 2: WCAG contrast
# ---------------------------------------------------------------------------

class TestWCAGContrast:
    """Test WCAG contrast ratio computation."""

    def test_black_white_max_contrast(self):
        ratio = te.validate_contrast("#FFFFFF", "#000000")
        assert ratio == pytest.approx(21.0, abs=0.1)

    def test_same_color_min_contrast(self):
        ratio = te.validate_contrast("#888888", "#888888")
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_contrast_is_symmetric(self):
        r1 = te.validate_contrast("#C48838", "#222224")
        r2 = te.validate_contrast("#222224", "#C48838")
        assert r1 == pytest.approx(r2)

    def test_passes_wcag_aa_normal_text(self):
        # White on black should always pass
        assert te.passes_wcag_aa("#FFFFFF", "#000000") is True

    def test_passes_wcag_aa_fails_low_contrast(self):
        # Very similar colors
        assert te.passes_wcag_aa("#777777", "#888888") is False

    def test_passes_wcag_aa_large_text_threshold(self):
        # Large text has lower threshold (3.0)
        ratio = te.validate_contrast("#C48838", "#222224")
        if ratio >= 3.0:
            assert te.passes_wcag_aa("#C48838", "#222224", large_text=True) is True


# ---------------------------------------------------------------------------
# Class 3: Modifier functions
# ---------------------------------------------------------------------------

class TestModifiers:
    """Test warmth and lightness offset functions."""

    def test_warmth_positive_shift(self):
        result = te.apply_warmth_offset(35, 20)
        assert result == 55

    def test_warmth_negative_shift(self):
        result = te.apply_warmth_offset(35, -20)
        assert result == 15

    def test_warmth_wraps_around_360(self):
        result = te.apply_warmth_offset(350, 30)
        assert result == 20

    def test_warmth_wraps_around_zero(self):
        result = te.apply_warmth_offset(10, -30)
        assert result == 340

    def test_warmth_clamped_high(self):
        result = te.apply_warmth_offset(100, 100)
        # Clamped to +30
        assert result == 130

    def test_warmth_clamped_low(self):
        result = te.apply_warmth_offset(100, -100)
        # Clamped to -30
        assert result == 70

    def test_lightness_positive_offset(self):
        result = te.apply_lightness_offset(50.0, 20)
        assert result == 70.0

    def test_lightness_negative_offset(self):
        result = te.apply_lightness_offset(50.0, -20)
        assert result == 30.0

    def test_lightness_clamped_high(self):
        result = te.apply_lightness_offset(90.0, 50)
        assert result == 95.0  # clamped

    def test_lightness_clamped_low(self):
        result = te.apply_lightness_offset(10.0, -50)
        assert result == 5.0  # clamped

    def test_lightness_offset_zero_is_identity(self):
        result = te.apply_lightness_offset(42.0, 0)
        assert result == 42.0


# ---------------------------------------------------------------------------
# Class 4: Accent scale generation
# ---------------------------------------------------------------------------

class TestAccentScale:
    """Test accent scale generation."""

    def test_scale_has_10_keys(self):
        scale = te.generate_accent_scale(35, 70, "dark")
        assert len(scale) == 10

    def test_scale_keys_match_expected(self):
        scale = te.generate_accent_scale(35, 70, "dark")
        expected = {"50", "100", "200", "300", "400", "500", "600", "700", "800", "900"}
        assert set(scale.keys()) == expected

    def test_scale_values_are_valid_hex(self):
        scale = te.generate_accent_scale(210, 65, "dark")
        for key, val in scale.items():
            assert te.is_valid_hex(val), f"Invalid hex at {key}: {val}"

    def test_dark_light_produce_different_scales(self):
        dark = te.generate_accent_scale(35, 70, "dark")
        light = te.generate_accent_scale(35, 70, "light")
        assert dark["500"] != light["500"]

    def test_hue_zero_works(self):
        scale = te.generate_accent_scale(0, 70, "dark")
        assert len(scale) == 10
        for val in scale.values():
            assert te.is_valid_hex(val)

    def test_hue_359_works(self):
        scale = te.generate_accent_scale(359, 70, "dark")
        assert len(scale) == 10

    def test_saturation_zero_produces_greys(self):
        scale = te.generate_accent_scale(35, 0, "dark")
        # With zero saturation, all values should be achromatic (R=G=B)
        for val in scale.values():
            r, g, b = te.hex_to_rgb(val)
            assert abs(r - g) <= 1 and abs(g - b) <= 1, f"Not grey: {val}"

    def test_saturation_100_works(self):
        scale = te.generate_accent_scale(35, 100, "dark")
        assert len(scale) == 10

    def test_warmth_modifies_output(self):
        base = te.generate_accent_scale(35, 70, "dark", 0, 0)
        warm = te.generate_accent_scale(35, 70, "dark", 0, 20)
        assert base["500"] != warm["500"]

    def test_lightness_offset_modifies_output(self):
        base = te.generate_accent_scale(35, 70, "dark", 0, 0)
        lighter = te.generate_accent_scale(35, 70, "dark", 15, 0)
        assert base["500"] != lighter["500"]


# ---------------------------------------------------------------------------
# Class 5: Theme variable generation
# ---------------------------------------------------------------------------

class TestThemeVariables:
    """Test full CSS variable generation."""

    def test_variables_include_all_acc_keys(self):
        v = te.generate_theme_variables(35)
        for key in te.SCALE_KEYS:
            assert f"oo-acc-{key}" in v

    def test_variables_include_accent_primary(self):
        v = te.generate_theme_variables(35)
        assert "oo-accent-primary" in v

    def test_variables_include_sage_tokens(self):
        v = te.generate_theme_variables(35, 130)
        assert "oo-sage" in v
        assert "oo-sage-bg" in v
        assert "oo-sage-bd" in v
        assert "oo-pine" in v

    def test_variables_include_tobacco_tokens(self):
        v = te.generate_theme_variables(35)
        assert "oo-tobacco" in v
        assert "oo-tobacco-bg" in v
        assert "oo-tobacco-bd" in v

    def test_variables_include_button_tokens(self):
        v = te.generate_theme_variables(35)
        assert "oo-btn-primary-bg" in v
        assert "oo-btn-primary-fg" in v
        assert "oo-btn-primary-hover" in v

    def test_variables_include_input_focus(self):
        v = te.generate_theme_variables(35)
        assert "oo-input-focus" in v

    def test_variables_include_message_bubbles(self):
        v = te.generate_theme_variables(35)
        assert "oo-msg-user-bg" in v
        assert "oo-msg-user-bd" in v

    def test_dark_mode_btn_fg_is_dark(self):
        v = te.generate_theme_variables(35, mode="dark")
        assert v["oo-btn-primary-fg"] == "#222224"

    def test_light_mode_btn_fg_is_light(self):
        v = te.generate_theme_variables(35, mode="light")
        assert v["oo-btn-primary-fg"] == "#F0EBE4"

    def test_secondary_hue_auto_derived(self):
        v = te.generate_theme_variables(35, secondary_hue=-1)
        assert "oo-sage" in v  # Should still have sage derived from hue+90

    def test_all_modifiers_applied(self):
        v_base = te.generate_theme_variables(35, 130, "dark", 70, 30, 0, 0, 0, 0)
        v_mod = te.generate_theme_variables(35, 130, "dark", 70, 30, 10, -5, 15, -10)
        assert v_base["oo-acc-500"] != v_mod["oo-acc-500"]

    def test_variable_count_minimum(self):
        v = te.generate_theme_variables(35)
        assert len(v) >= 24


# ---------------------------------------------------------------------------
# Class 6: Built-in presets
# ---------------------------------------------------------------------------

class TestBuiltinPresets:
    """Test built-in preset themes."""

    def test_five_builtin_presets(self):
        presets = te.get_preset_themes()
        assert len(presets) == 5

    def test_preset_ids_are_unique(self):
        presets = te.get_preset_themes()
        ids = [p["id"] for p in presets]
        assert len(ids) == len(set(ids))

    def test_default_preset_exists(self):
        p = te.get_preset_by_id("default")
        assert p is not None
        assert p["name"] == "Sage & Tobacco"

    def test_all_presets_have_required_fields(self):
        required = {
            "id", "name", "description",
            "accent_hue", "accent_saturation",
            "secondary_hue", "secondary_saturation",
            "accent_lightness_offset", "secondary_lightness_offset",
            "accent_warmth", "secondary_warmth",
            "builtin",
        }
        for preset in te.get_preset_themes():
            for field in required:
                assert field in preset, f"Missing {field} in preset {preset['id']}"

    def test_all_presets_are_builtin(self):
        for preset in te.get_preset_themes():
            assert preset["builtin"] is True

    def test_get_preset_by_id_nonexistent(self):
        assert te.get_preset_by_id("nonexistent") is None

    def test_presets_are_copies(self):
        p1 = te.get_preset_themes()
        p2 = te.get_preset_themes()
        p1[0]["name"] = "MUTATED"
        assert p2[0]["name"] != "MUTATED"

    def test_builtin_preset_ids_frozenset(self):
        assert "default" in te.BUILTIN_PRESET_IDS
        assert "ocean" in te.BUILTIN_PRESET_IDS
        assert "forest" in te.BUILTIN_PRESET_IDS
        assert "rose" in te.BUILTIN_PRESET_IDS
        assert "monochrome" in te.BUILTIN_PRESET_IDS


# ---------------------------------------------------------------------------
# Class 7: Custom preset validation
# ---------------------------------------------------------------------------

class TestCustomPresetValidation:
    """Test custom preset validation logic."""

    def test_valid_custom_preset(self):
        errors = te.validate_custom_preset({
            "name": "My Theme", "id": "my-theme", "accent_hue": 200,
        })
        assert errors == []

    def test_missing_name_rejected(self):
        errors = te.validate_custom_preset({"id": "x", "accent_hue": 0})
        assert any("name" in e for e in errors)

    def test_empty_name_rejected(self):
        errors = te.validate_custom_preset({"name": "   ", "accent_hue": 0})
        assert any("empty" in e for e in errors)

    def test_name_too_long_rejected(self):
        errors = te.validate_custom_preset({
            "name": "x" * 51, "id": "x", "accent_hue": 0,
        })
        assert any("50" in e for e in errors)

    def test_builtin_id_rejected(self):
        errors = te.validate_custom_preset({
            "name": "Test", "id": "default", "accent_hue": 0,
        })
        assert any("reserved" in e.lower() or "built-in" in e.lower() for e in errors)

    def test_invalid_id_chars_rejected(self):
        errors = te.validate_custom_preset({
            "name": "Test", "id": "my theme!!", "accent_hue": 0,
        })
        assert any("alphanumeric" in e for e in errors)

    def test_warmth_out_of_range_rejected(self):
        errors = te.validate_custom_preset({
            "name": "Test", "id": "x", "accent_hue": 0, "accent_warmth": 50,
        })
        assert any("warmth" in e for e in errors)

    def test_lightness_offset_out_of_range_rejected(self):
        errors = te.validate_custom_preset({
            "name": "Test", "id": "x", "accent_hue": 0,
            "accent_lightness_offset": -60,
        })
        assert any("lightness" in e for e in errors)

    def test_description_too_long_rejected(self):
        errors = te.validate_custom_preset({
            "name": "Test", "id": "x", "accent_hue": 0,
            "description": "d" * 201,
        })
        assert any("200" in e for e in errors)


# ---------------------------------------------------------------------------
# Class 8: Theme config validation
# ---------------------------------------------------------------------------

class TestThemeConfigValidation:
    """Test theme config validation."""

    def test_valid_config(self):
        errors = te.validate_theme_config({
            "accent_hue": 35, "mode": "dark",
            "accent_saturation": 70, "secondary_hue": 130,
        })
        assert errors == []

    def test_missing_accent_hue(self):
        errors = te.validate_theme_config({"mode": "dark"})
        assert any("accent_hue" in e for e in errors)

    def test_invalid_mode(self):
        errors = te.validate_theme_config({
            "accent_hue": 35, "mode": "neon",
        })
        assert any("mode" in e for e in errors)

    def test_accent_hue_out_of_range(self):
        errors = te.validate_theme_config({"accent_hue": 400})
        assert any("accent_hue" in e for e in errors)

    def test_saturation_out_of_range(self):
        errors = te.validate_theme_config({
            "accent_hue": 35, "accent_saturation": 150,
        })
        assert any("saturation" in e for e in errors)

    def test_warmth_out_of_range(self):
        errors = te.validate_theme_config({
            "accent_hue": 35, "accent_warmth": 50,
        })
        assert any("warmth" in e for e in errors)


# ---------------------------------------------------------------------------
# Class 9: Import/export
# ---------------------------------------------------------------------------

class TestImportExport:
    """Test preset import/export functionality."""

    def test_import_valid_data(self):
        data = [{"name": "T1", "id": "t1", "accent_hue": 100}]
        valid, errors = te.validate_preset_import(data)
        assert len(valid) == 1
        assert errors == []

    def test_import_rejects_non_list(self):
        valid, errors = te.validate_preset_import({"name": "x"})
        assert len(valid) == 0
        assert any("array" in e.lower() for e in errors)

    def test_import_rejects_non_dict_items(self):
        valid, errors = te.validate_preset_import(["not a dict"])
        assert len(valid) == 0
        assert len(errors) > 0

    def test_import_too_many_presets(self):
        data = [
            {"name": f"T{i}", "id": f"t{i}", "accent_hue": i}
            for i in range(25)
        ]
        valid, errors = te.validate_preset_import(data)
        assert len(valid) == 0
        assert any("20" in e for e in errors)

    def test_import_partial_valid(self):
        data = [
            {"name": "Good", "id": "good", "accent_hue": 50},
            {"id": "bad"},  # missing name
        ]
        valid, errors = te.validate_preset_import(data)
        assert len(valid) == 1
        assert len(errors) > 0

    def test_export_strips_internal_fields(self):
        presets = [{"id": "x", "name": "T", "accent_hue": 35, "internal_key": "skip"}]
        exported = te.export_presets(presets)
        parsed = json.loads(exported)
        assert "internal_key" not in parsed[0]

    def test_export_preserves_custom_fields(self):
        presets = [{
            "id": "x", "name": "T", "accent_hue": 35,
            "accent_saturation": 80, "accent_warmth": 10,
        }]
        exported = te.export_presets(presets)
        parsed = json.loads(exported)
        assert parsed[0]["accent_warmth"] == 10
        assert parsed[0]["accent_saturation"] == 80

    def test_export_is_valid_json(self):
        presets = [{"id": "x", "name": "T", "accent_hue": 35}]
        exported = te.export_presets(presets)
        parsed = json.loads(exported)
        assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# Class 10: File existence and structure
# ---------------------------------------------------------------------------

class TestFileStructure:
    """Test that all S152 files exist and have expected content."""

    def test_theme_engine_exists(self):
        assert os.path.isfile(THEME_ENGINE_PATH)

    def test_theme_customizer_exists(self):
        assert os.path.isfile(CUSTOMIZER_PATH)

    def test_theme_api_ts_exists(self):
        assert os.path.isfile(THEME_API_PATH)

    def test_customizer_no_hardcoded_hex_in_styles(self):
        content = read_file(CUSTOMIZER_PATH)
        # Extract style block
        style_match = re.search(r"<style>(.*?)</style>", content, re.DOTALL)
        if style_match:
            style_block = style_match.group(1)
            hex_matches = re.findall(r"#[0-9A-Fa-f]{3,6}", style_block)
            assert len(hex_matches) == 0, f"Hardcoded hex in styles: {hex_matches}"

    def test_customizer_uses_oo_variables(self):
        content = read_file(CUSTOMIZER_PATH)
        assert "--oo-bg-surface" in content
        assert "--oo-fg-primary" in content
        assert "--oo-accent-primary" in content

    def test_theme_api_has_all_functions(self):
        content = read_file(THEME_API_PATH)
        expected = [
            "getThemeConfig", "saveThemeConfig", "getThemePresets",
            "createCustomPreset", "deleteCustomPreset",
            "exportCustomPresets", "importCustomPresets",
        ]
        for fn in expected:
            assert fn in content, f"Missing function: {fn}"

    def test_schemas_has_theme_classes(self):
        content = read_file(SCHEMAS_PATH)
        expected = [
            "ThemeConfigRequest", "ThemeConfigResponse",
            "ThemePresetResponse", "ThemePresetsListResponse",
            "CustomPresetCreateRequest", "CustomPresetImportRequest",
            "CustomPresetsExportResponse",
        ]
        for cls in expected:
            assert cls in content, f"Missing schema: {cls}"

    def test_routes_has_theme_endpoints(self):
        content = read_file(ROUTES_PATH)
        expected_paths = [
            "/theme/presets",
            "/theme",
            "/theme/presets/custom",
            "/theme/presets/export",
            "/theme/presets/import",
        ]
        for path in expected_paths:
            assert path in content, f"Missing route: {path}"

    def test_version_is_3_2_1(self):
        content = read_file(VERSION_PATH)
        assert '"3.2.1"' in content or "'3.2.1'" in content
