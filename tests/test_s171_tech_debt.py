#!/usr/bin/env python3
"""
S171 -- Tech Debt: E2E findings integration + CI hardening.

Phase-gated tests for the single Tech Debt session of ROADMAP_S165_S182
(Theme 2). Covers, in order of the session goals:

  Goal 1 (Phase 1) -- the seven hardening-pass code fixes:
    - pyproject: pysqlcipher3 -> sqlcipher3-binary, [all] isolated
    - CSRF middleware skips in single-user (non-Bulbe) mode
    - _deep_merge defensive validation (raises on structural type conflict)
    - smart_router pre-flight RAM check before model selection
    - frontend WebSocket client reconnect with exponential backoff (1-30s)
    - server WebSocket graceful close with proper close code on errors
    - client.ts formalized fetchApi shim (was imported but never exported)

Tests are file-content assertions plus importlib-isolated runtime checks for
the pure-Python helpers, following the established spec_from_file_location +
sys.modules stub pattern (the backend is not importable in the sandbox because
'ollama' is absent). Later phases append their own classes to this file.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend", "src")
CONFIG_DIR = os.path.join(BACKEND_DIR, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(*parts: str) -> str:
    """Read a repository file as UTF-8 text."""
    with open(os.path.join(PROJECT_ROOT, *parts), "r", encoding="utf-8") as fh:
        return fh.read()


def _load_system_presets_module():
    """Load system_presets.py in isolation (stubbed opti_oignon.config)."""
    import yaml

    config_mod = types.ModuleType("opti_oignon.config")
    config_mod.CONFIG_DIR = Path(CONFIG_DIR)
    config_mod.DATA_DIR = Path(DATA_DIR)

    def load_yaml(filepath):
        if not filepath.exists():
            return {}
        with open(filepath) as f:
            return yaml.safe_load(f) or {}

    def save_yaml(filepath, data):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        return True

    config_mod.load_yaml = load_yaml
    config_mod.save_yaml = save_yaml

    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [BACKEND_DIR]
        sys.modules["opti_oignon"] = pkg
    sys.modules["opti_oignon.config"] = config_mod

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.system_presets",
        os.path.join(BACKEND_DIR, "system_presets.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_smart_router_module():
    """Load smart_router.py in isolation (guarded relative imports degrade)."""
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [BACKEND_DIR]
        sys.modules["opti_oignon"] = pkg
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.smart_router",
        os.path.join(BACKEND_DIR, "smart_router.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.smart_router"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_speculative_module():
    """Load speculative_decoding.py in isolation (no runtime deps needed)."""
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [BACKEND_DIR]
        sys.modules["opti_oignon"] = pkg
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.speculative_decoding",
        os.path.join(BACKEND_DIR, "speculative_decoding.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.speculative_decoding"] = mod
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# Goal 1 / Phase 1 -- code fixes
# ===========================================================================

class TestSqlcipherMigration:
    """Fix 1a: pyproject [sqlcipher] migrated to sqlcipher3-binary."""

    def setup_method(self):
        self.content = _read("pyproject.toml")

    def test_sqlcipher_binary_present(self):
        assert "sqlcipher3-binary" in self.content

    def test_pysqlcipher3_removed_from_dependency(self):
        # The legacy package must no longer appear as a dependency spec.
        assert 'pysqlcipher3>=' not in self.content
        assert '"pysqlcipher3' not in self.content

    def test_sqlcipher_group_still_defined(self):
        assert "sqlcipher = [" in self.content


class TestAllGroupIsolation:
    """Fix 1b: [all] meta-group isolates platform-specific extras."""

    def setup_method(self):
        self.content = _read("pyproject.toml")

    def test_all_excludes_sqlcipher_and_llama(self):
        assert '"opti-oignon[auth,dev,docs]"' in self.content

    def test_old_all_string_gone(self):
        assert '"opti-oignon[auth,sqlcipher,llama,dev,docs]"' not in self.content

    def test_isolation_rationale_documented(self):
        assert "EXCLUDED from [all]" in self.content


class TestCsrfSingleUserSkip:
    """Fix 2: CSRF middleware skips validation in single-user (non-Bulbe) mode."""

    def setup_method(self):
        self.content = _read("opti_oignon", "api", "csrf_middleware.py")

    def test_single_user_helper_defined(self):
        assert "def _is_single_user_unauthenticated()" in self.content

    def test_bulbe_helper_defined(self):
        assert "def _is_bulbe()" in self.content

    def test_dispatch_guard_present(self):
        assert "if _is_single_user_unauthenticated():" in self.content

    def test_guard_checks_single_user_and_not_bulbe(self):
        assert "single_user_mode" in self.content
        assert "not _is_bulbe()" in self.content


class TestDeepMergeValidation:
    """Fix 3: _deep_merge raises on structural type conflict (runtime, isolated)."""

    @classmethod
    def setup_class(cls):
        cls.mod = _load_system_presets_module()

    # Legacy contract preserved (mirrors test_system_presets_s84).
    def test_simple_merge_preserved(self):
        assert self.mod._deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_scalar_override_preserved(self):
        assert self.mod._deep_merge({"a": 1}, {"a": 99}) == {"a": 99}

    def test_nested_merge_preserved(self):
        base = {"a": {"b": 1, "c": 2}}
        over = {"a": {"c": 99, "d": 3}}
        assert self.mod._deep_merge(base, over) == {"a": {"b": 1, "c": 99, "d": 3}}

    def test_empty_override_preserved(self):
        assert self.mod._deep_merge({"a": 1}, {}) == {"a": 1}

    # New defensive behavior.
    def test_dict_over_scalar_raises(self):
        try:
            self.mod._deep_merge({"a": 5}, {"a": {"x": 1}})
            assert False, "expected TypeError"
        except TypeError as exc:
            assert "type conflict" in str(exc)

    def test_scalar_over_dict_raises(self):
        try:
            self.mod._deep_merge({"a": {"x": 1}}, {"a": 5})
            assert False, "expected TypeError"
        except TypeError as exc:
            assert "type conflict" in str(exc)

    def test_non_dict_argument_raises(self):
        try:
            self.mod._deep_merge([], {})
            assert False, "expected TypeError"
        except TypeError:
            pass


class TestSmartRouterRamPreflight:
    """Fix 4: smart_router pre-flight RAM check (file-content + runtime)."""

    @classmethod
    def setup_class(cls):
        cls.mod = _load_smart_router_module()
        cls.content = _read("opti_oignon", "smart_router.py")

    def test_helpers_defined(self):
        assert "def _get_available_ram_mb()" in self.content
        assert "def _estimate_model_ram_mb(" in self.content

    def test_instance_flag_present(self):
        assert "self._ram_preflight = True" in self.content

    def test_config_toggle_read(self):
        assert '"ram_preflight" in config' in self.content

    def test_select_model_applies_budget(self):
        assert "ram_budget_mb" in self.content
        assert "RAM pre-flight excluded" in self.content

    def test_estimate_parses_billions(self):
        assert self.mod._estimate_model_ram_mb("32B") == 32.0 * 750.0
        assert self.mod._estimate_model_ram_mb("7b") == 7.0 * 750.0
        assert self.mod._estimate_model_ram_mb("1.5B") == 1.5 * 750.0

    def test_estimate_failopen_on_unknown(self):
        assert self.mod._estimate_model_ram_mb(None) == 0.0
        assert self.mod._estimate_model_ram_mb("") == 0.0
        assert self.mod._estimate_model_ram_mb("not-a-size") == 0.0

    def test_available_ram_non_negative(self):
        assert self.mod._get_available_ram_mb() >= 0.0


class TestWsClientReconnect:
    """Fix 5: frontend WebSocket client reconnects with exponential backoff."""

    def setup_method(self):
        self.content = _read("frontend", "src", "lib", "api", "client.ts")

    def test_reconnecting_class_exported(self):
        assert "export class ReconnectingWebSocket" in self.content

    def test_backoff_bounds_defined(self):
        assert "export const WS_BACKOFF_MIN_MS = 1000" in self.content
        assert "export const WS_BACKOFF_MAX_MS = 30000" in self.content

    def test_exponential_backoff_logic(self):
        assert "WS_BACKOFF_MIN_MS * 2 **" in self.content
        assert "Math.min(WS_BACKOFF_MAX_MS" in self.content

    def test_client_close_stops_reconnect(self):
        assert "closedByClient" in self.content


class TestWsServerGracefulClose:
    """Fix 6: server WebSocket endpoints close with explicit codes on errors."""

    def test_chat_close_codes(self):
        c = _read("opti_oignon", "api", "routes_chat.py")
        assert "WS_CLOSE_INTERNAL_ERROR = 1011" in c
        assert "WS_CLOSE_INVALID_DATA = 1003" in c
        assert "await websocket.close(code=WS_CLOSE_INTERNAL_ERROR)" in c
        assert "await websocket.close(code=WS_CLOSE_INVALID_DATA)" in c

    def test_benchmark_close_code(self):
        c = _read("opti_oignon", "api", "routes_benchmark.py")
        assert "WS_CLOSE_INTERNAL_ERROR = 1011" in c
        assert "await websocket.close(code=WS_CLOSE_INTERNAL_ERROR)" in c

    def test_live_metrics_close_code(self):
        c = _read("opti_oignon", "api", "routes_live_metrics.py")
        assert "WS_CLOSE_INTERNAL_ERROR = 1011" in c
        assert "await websocket.close(code=WS_CLOSE_INTERNAL_ERROR)" in c


class TestFetchApiShim:
    """Fix 7: client.ts exports a formalized fetchApi shim used by 4 modules."""

    def setup_method(self):
        self.content = _read("frontend", "src", "lib", "api", "client.ts")

    def test_fetch_api_exported(self):
        assert "export async function fetchApi" in self.content

    def test_options_interface_exported(self):
        assert "export interface FetchApiOptions" in self.content

    def test_routes_through_typed_helpers(self):
        assert "return apiPost<T>(path, parsedBody);" in self.content
        assert "return apiGet<T>(path);" in self.content

    def test_callers_still_import_fetch_api(self):
        for mod in ("securityMode", "toolCallApproval", "pluginAllowlist", "searchKillSwitch"):
            c = _read("frontend", "src", "lib", "api", f"{mod}.ts")
            assert "import { fetchApi } from './client';" in c


# ===========================================================================
# Goal 2 / Phase 2 -- CI improvements
# ===========================================================================

class TestCiTypeScriptCheck:
    """Goal 2: `npm run check` (svelte-check) wired into CI."""

    def setup_method(self):
        self.ci = _read(".github", "workflows", "ci.yml")

    def test_check_script_defined(self):
        import json
        scripts = json.loads(_read("frontend", "package.json"))["scripts"]
        assert scripts.get("check") == "svelte-check --tsconfig ./tsconfig.json"

    def test_ci_runs_check(self):
        assert "npm run check" in self.ci

    def test_frontend_job_present(self):
        assert "frontend:" in self.ci
        assert "Frontend Checks" in self.ci


class TestCiEslint:
    """Goal 2: eslint wired into CI with a config and a lint script."""

    def setup_method(self):
        import json
        self.ci = _read(".github", "workflows", "ci.yml")
        self.pkg = json.loads(_read("frontend", "package.json"))

    def test_lint_script_defined(self):
        assert self.pkg["scripts"].get("lint") == "eslint ."

    def test_eslint_in_dev_dependencies(self):
        dev = self.pkg["devDependencies"]
        assert "eslint" in dev
        assert "eslint-plugin-svelte" in dev
        assert "@typescript-eslint/parser" in dev

    def test_eslint_config_exists(self):
        cfg = _read("frontend", ".eslintrc.cjs")
        assert "plugin:svelte/recommended" in cfg
        assert "@typescript-eslint/parser" in cfg

    def test_ci_runs_lint(self):
        assert "npm run lint" in self.ci

    def test_lockfile_includes_eslint(self):
        import json
        lock = json.loads(_read("frontend", "package-lock.json"))
        root_dev = lock["packages"][""]["devDependencies"]
        assert "eslint" in root_dev
        assert "node_modules/eslint" in lock["packages"]


class TestCiInstallSmoke:
    """Goal 2: build + install the distribution in a fresh env, CLI smoke."""

    def setup_method(self):
        self.ci = _read(".github", "workflows", "ci.yml")

    def test_install_job_present(self):
        assert "install:" in self.ci
        assert "Install Smoke Test" in self.ci

    def test_builds_distribution(self):
        assert "python -m build" in self.ci

    def test_installs_in_fresh_venv(self):
        assert "python -m venv" in self.ci
        assert "pip install dist/*.whl" in self.ci

    def test_cli_smoke_invocation(self):
        assert "oo --version" in self.ci


# ===========================================================================
# Goal 3 / Phase 3 -- test migration + isolation + deselect re-evaluation
# ===========================================================================

class TestSpeculativeStatsIsolation:
    """Goal 3: speculative_decoding manager stats path is injectable.

    Replaces the stateful test_s110 assertion that shared
    data/speculative_stats.json across pytest invocations (deselected).
    """

    @classmethod
    def setup_class(cls):
        cls.mod = _load_speculative_module()
        cls.content = _read("opti_oignon", "speculative_decoding.py")

    def test_stats_path_param_in_signature(self):
        assert "stats_path: Optional[str] = None" in self.content

    def test_stats_path_used_not_global(self):
        assert "self._stats_path = Path(stats_path) if stats_path else _RESULTS_PATH" in self.content
        assert "self._stats_path.is_file()" in self.content

    def test_record_then_reset_isolated(self, tmp_path):
        p = str(tmp_path / "stats.json")
        mgr = self.mod.SpeculativeDecodingManager(stats_path=p)
        mgr.record_acceptance(16, 12, 2.0)
        assert mgr.stats.total_runs == 1
        mgr.reset_stats()
        assert mgr.stats.total_runs == 0

    def test_no_shared_state_leak(self, tmp_path):
        p1 = str(tmp_path / "a.json")
        p2 = str(tmp_path / "b.json")
        mgr1 = self.mod.SpeculativeDecodingManager(stats_path=p1)
        mgr1.record_acceptance(10, 8, 1.5)
        # A manager on a different path must not see mgr1's run.
        mgr2 = self.mod.SpeculativeDecodingManager(stats_path=p2)
        assert mgr2.stats.total_runs == 0

    def test_original_stateful_test_deselected(self):
        addopts = _read("pyproject.toml")
        assert (
            "--deselect=tests/test_s110_inference_accelerator.py::"
            "TestSpeculativeDecodingManager::test_record_and_reset_stats" in addopts
        )


class TestActualPackagingLayout:
    """Goal 3: retire stale app.py / setup.py assertions; assert real layout.

    opti_oignon/app.py and a top-level setup.py do not exist; the FastAPI app
    lives in opti_oignon/api/app.py, the version is hardcoded in
    opti_oignon/__version__.py, and the CLI entry point is opti_oignon.cli.main:cli.
    The stale tests are deselected; this is the replacement assertion.
    """

    def test_no_package_root_app_py(self):
        assert not os.path.isfile(os.path.join(BACKEND_DIR, "app.py"))

    def test_no_top_level_setup_py(self):
        assert not os.path.isfile(os.path.join(PROJECT_ROOT, "setup.py"))

    def test_fastapi_app_lives_under_api(self):
        assert os.path.isfile(os.path.join(BACKEND_DIR, "api", "app.py"))

    def test_version_is_hardcoded(self):
        import re
        vcontent = _read("opti_oignon", "__version__.py")
        # A literal assignment, not a value read from disk at import time.
        assert re.search(r'^__version__\s*=\s*["\']', vcontent, re.M)
        assert "open(" not in vcontent

    def test_cli_entry_point_in_pyproject(self):
        content = _read("pyproject.toml")
        assert 'oo = "opti_oignon.cli.main:cli"' in content

    def test_stale_app_setup_tests_deselected(self):
        addopts = _read("pyproject.toml")
        for target in (
            "tests/test_design_system_s83.py::TestVersionBump::test_app_py_version",
            "tests/test_palette_v4e_s93.py::TestVersionBump::test_app_py_version",
            "tests/test_ux_cleanup_s94.py::TestVersionAndIntegrity::test_version_setup_py",
            "tests/test_s125_security_hardening_p2.py::TestVersionBump::test_setup_reads_from_version_file",
        ):
            assert f"--deselect={target}" in addopts


class TestDeselectReEvaluation:
    """Goal 3: re-evaluate redesign-cycle deselects; prune stale ones."""

    def setup_method(self):
        self.addopts = _read("pyproject.toml")

    def test_stale_hex_deselect_removed(self):
        # test_model_manager_no_hex now passes after the frontend tokenization;
        # its deselect was pruned in S171.
        assert (
            "tests/test_s113_telemetry_dashboard_profiler.py::"
            "TestFrontendHexCompliance::test_model_manager_no_hex" not in self.addopts
        )

    def test_theme_transition_supersede_retained(self):
        # Superseded by tokenization (re-asserted in test_s170); deselect kept.
        assert (
            "--deselect=tests/test_palette_v4e_s93.py::"
            "TestThemeTransition::test_transition_properties" in self.addopts
        )

    def test_version_supersedes_retained_pending_final_bump(self):
        # The 6 test_s164_release version tests stay deselected until the
        # 3.4.0 final bump reconciliation (Phase 5 / Goal 5).
        assert (
            "--deselect=tests/test_s164_release.py::TestVersion::test_version_string"
            in self.addopts
        )


# ===========================================================================
# Goal 4 / Phase 4 -- residual French sweep
# ===========================================================================

class TestFrenchSweep:
    """Goal 4: residual French removed from touched backend files.

    The enforced standard (scripts/security_scan.py check_no_french, also
    exercised by test_s139) must report zero violations. The files touched in
    S171 (routes_chat.py) and the conversation module are additionally swept of
    accent-free French in comments / docstrings / log messages.
    """

    def test_security_scan_reports_zero_french(self):
        scan = os.path.join(PROJECT_ROOT, "scripts", "security_scan.py")
        if not os.path.isfile(scan):
            import pytest
            pytest.skip("security_scan.py not found")
        spec = importlib.util.spec_from_file_location("security_scan_s171", scan)
        mod = importlib.util.module_from_spec(spec)
        old = sys.path.copy()
        sys.path.insert(0, os.path.dirname(scan))
        try:
            spec.loader.exec_module(mod)
            result = mod.check_no_french(mod._py_files(include_tests=False), mod._svelte_files())
            assert len(result.violations) == 0
        finally:
            sys.path = old

    def test_routes_chat_swept(self):
        c = _read("opti_oignon", "api", "routes_chat.py")
        for token in ("deconnecte", "Recuperer", "les reponses", "metadonnees", "Interroge"):
            assert token not in c

    def test_conversation_swept(self):
        c = _read("opti_oignon", "conversation.py")
        for token in ("Erreur ", "Raccourci pour", "dernier msg est", "repertoire d'historique"):
            assert token not in c


# ===========================================================================
# Goal 5 / Phase 5 -- documentation + final version bump (3.4.0)
# ===========================================================================

class TestFinalVersionBump:
    """Goal 5: 3.4.0-rc graduated to the final 3.4.0."""

    def test_version_py_is_final(self):
        import re
        c = _read("opti_oignon", "__version__.py")
        m = re.search(r'__version__\s*=\s*"([^"]+)"', c)
        assert m and m.group(1) == "3.4.0"

    def test_pyproject_is_final(self):
        assert 'version = "3.4.0"' in _read("pyproject.toml")
        assert 'version = "3.4.0-rc"' not in _read("pyproject.toml")

    def test_no_rc_suffix_in_version_file(self):
        assert "3.4.0-rc" not in _read("opti_oignon", "__version__.py")

    def test_pep440_deselect_removed(self):
        # The bare 3.4.0 satisfies the original ^X.Y.Z$ PEP 440 check, so its
        # deselect was dropped in S171 (re-evaluation).
        assert (
            "tests/test_s164_release.py::TestVersion::test_version_pep440_compliant"
            not in _read("pyproject.toml")
        )

    def test_s170_supersede_reconciled_to_final(self):
        c = _read("tests", "test_s170_polish_accessibility.py")
        assert 'self._version_from_file() == "3.4.0"' in c
        assert 'self._version_from_file() == "3.4.0-rc"' not in c


class TestChangelogFinalEntry:
    """Goal 5: CHANGELOG has the v3.4.0 (S171) entry; RC and v3.3.0 retained."""

    def setup_method(self):
        self.c = _read("CHANGELOG.md")

    def test_v340_final_entry_present(self):
        assert "## v3.4.0 -- 2026-06-01 (S171)" in self.c

    def test_top_version_entry_is_340(self):
        import re
        entries = re.findall(r"## v(\d+\.\d+\.\d+)", self.c)
        assert entries and entries[0] == "3.4.0"

    def test_rc_and_v330_entries_retained(self):
        assert "## v3.4.0-rc -- 2026-06-01 (S170)" in self.c
        assert "## v3.3.0" in self.c


class TestDocsAndContributing:
    """Goal 5: conda install path + post-install checklist docs."""

    def test_contributing_has_conda_path(self):
        c = _read("CONTRIBUTING.md")
        assert "conda create -n opti-oignon" in c
        assert "conda-forge" in c

    def test_post_install_checklist_exists(self):
        page = _read("docs", "getting-started", "post-install-checklist.md")
        assert "security.yaml" in page
        assert "security_mode: daily" in page
        assert "Daily" in page and "Bulbe" in page

    def test_checklist_in_mkdocs_nav(self):
        nav = _read("mkdocs.yml")
        assert "getting-started/post-install-checklist.md" in nav
