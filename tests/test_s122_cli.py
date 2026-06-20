#!/usr/bin/env python3
"""
Tests for S122 -- CLI Companion ``oo``.

Test groups:
1.  CLIConfig: defaults, ws_base, NO_COLOR, to_dict
2.  CLIConfig persistence: save / load round-trip
3.  CLIConfig edge cases: invalid format, missing file, bad YAML
4.  OOClient: URL building, ws URL building
5.  OOClient: _handle_response error paths
6.  OOClient: CLIClientError fields
7.  Output formatters: format_models_table
8.  Output formatters: format_status
9.  Output helpers: echo_error, echo_success
10. Spinner: context manager protocol
11. Click CLI: all commands registered
12. Click CLI: help text present
13. Click CLI: ask command prompt resolution
14. Click CLI: config set / reset
15. Click CLI: backup subcommands
16. Click CLI: rag subcommands
17. Entry point: pyproject.toml oo script
18. Entry point: setup.py delegates to pyproject
19. Optional deps: cli extra defined
20. Package structure: all cli/ files exist
21. No Ollama imports in cli/ package
22. All code comments in English
23. AST validation: all S122 Python files
24. Version bump: 2.4.0

Target: ~42 tests
"""

import ast
import importlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "opti_oignon"
CLI_DIR = BACKEND_DIR / "cli"
VERSION_FILE = BACKEND_DIR / "__version__.py"
PYPROJECT = ROOT / "pyproject.toml"
SETUP_PY = ROOT / "setup.py"

# S122 files under test
S122_PYTHON_FILES = [
    CLI_DIR / "__init__.py",
    CLI_DIR / "config.py",
    CLI_DIR / "client.py",
    CLI_DIR / "main.py",
    CLI_DIR / "output.py",
]


# ---------------------------------------------------------------------------
# Isolated module loader (bypass opti_oignon/__init__.py -> ollama)
# ---------------------------------------------------------------------------

def _load(name: str, filepath: Path):
    """Load a single module in isolation."""
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Pre-load CLI modules in isolation
_config_mod = _load("opti_oignon.cli.config", CLI_DIR / "config.py")
_client_mod = _load("opti_oignon.cli.client", CLI_DIR / "client.py")
_output_mod = _load("opti_oignon.cli.output", CLI_DIR / "output.py")

CLIConfig = _config_mod.CLIConfig
load_config = _config_mod.load_config
OOClient = _client_mod.OOClient
CLIClientError = _client_mod.CLIClientError
format_models_table = _output_mod.format_models_table
format_status = _output_mod.format_status
Spinner = _output_mod.Spinner


# =========================================================================
# 1. CLIConfig defaults
# =========================================================================

class TestCLIConfigDefaults(unittest.TestCase):
    """Verify CLIConfig default values."""

    def test_default_api_url(self):
        cfg = CLIConfig()
        self.assertEqual(cfg.api_url, "http://localhost:8001")

    def test_default_model_none(self):
        cfg = CLIConfig()
        self.assertIsNone(cfg.default_model)

    def test_default_output_format(self):
        cfg = CLIConfig()
        self.assertEqual(cfg.output_format, "text")

    def test_default_color_true(self):
        # Ensure NO_COLOR not set for this test
        env = os.environ.copy()
        env.pop("NO_COLOR", None)
        with patch.dict(os.environ, env, clear=True):
            cfg = CLIConfig()
            self.assertTrue(cfg.color)

    def test_default_timeout(self):
        cfg = CLIConfig()
        self.assertEqual(cfg.timeout, 120)

    def test_trailing_slash_stripped(self):
        cfg = CLIConfig(api_url="http://host:9000/")
        self.assertEqual(cfg.api_url, "http://host:9000")

    def test_invalid_output_format_falls_back(self):
        cfg = CLIConfig(output_format="xml")
        self.assertEqual(cfg.output_format, "text")


# =========================================================================
# 2. CLIConfig ws_base derivation
# =========================================================================

class TestCLIConfigWSBase(unittest.TestCase):
    """Test WebSocket URL derivation."""

    def test_http_to_ws(self):
        cfg = CLIConfig(api_url="http://localhost:8001")
        self.assertEqual(cfg.ws_base, "ws://localhost:8001")

    def test_https_to_wss(self):
        cfg = CLIConfig(api_url="https://secure.host:443")
        self.assertEqual(cfg.ws_base, "wss://secure.host:443")

    def test_bare_host_gets_ws(self):
        cfg = CLIConfig(api_url="myhost:3000")
        self.assertEqual(cfg.ws_base, "ws://myhost:3000")


# =========================================================================
# 3. CLIConfig NO_COLOR
# =========================================================================

class TestCLIConfigNoColor(unittest.TestCase):
    """Verify NO_COLOR environment variable support."""

    def test_no_color_env_disables(self):
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            cfg = CLIConfig(color=True)
            self.assertFalse(cfg.color)

    def test_no_color_empty_string_disables(self):
        with patch.dict(os.environ, {"NO_COLOR": ""}):
            cfg = CLIConfig(color=True)
            self.assertFalse(cfg.color)


# =========================================================================
# 4. CLIConfig persistence (save / load round-trip)
# =========================================================================

class TestCLIConfigPersistence(unittest.TestCase):
    """Test save/load round-trip."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cli.yaml"
            cfg = CLIConfig(
                api_url="http://remote:9999",
                default_model="mistral",
                output_format="json",
                timeout=60,
            )
            cfg.save(path)
            loaded = load_config(path)
            self.assertEqual(loaded.api_url, "http://remote:9999")
            self.assertEqual(loaded.default_model, "mistral")
            self.assertEqual(loaded.output_format, "json")
            self.assertEqual(loaded.timeout, 60)

    def test_load_missing_file_returns_defaults(self):
        cfg = load_config(Path("/nonexistent/path/cli.yaml"))
        self.assertEqual(cfg.api_url, "http://localhost:8001")

    def test_load_bad_yaml_returns_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(": : : bad yaml [[[")
            f.flush()
            cfg = load_config(Path(f.name))
            self.assertEqual(cfg.api_url, "http://localhost:8001")
            os.unlink(f.name)

    def test_to_dict_has_required_keys(self):
        cfg = CLIConfig()
        d = cfg.to_dict()
        self.assertIn("api_url", d)
        self.assertIn("output_format", d)
        self.assertIn("color", d)
        self.assertIn("timeout", d)

    def test_to_dict_omits_none_model(self):
        cfg = CLIConfig(default_model=None)
        d = cfg.to_dict()
        self.assertNotIn("default_model", d)

    def test_to_dict_includes_model_when_set(self):
        cfg = CLIConfig(default_model="llama3")
        d = cfg.to_dict()
        self.assertEqual(d["default_model"], "llama3")


# =========================================================================
# 5. OOClient URL building
# =========================================================================

class TestOOClientURLs(unittest.TestCase):
    """Verify URL construction."""

    def setUp(self):
        self.client = OOClient(config=CLIConfig(api_url="http://myhost:8080"))

    def test_http_url(self):
        self.assertEqual(self.client._url("/api/models"), "http://myhost:8080/api/models")

    def test_ws_url(self):
        self.assertEqual(
            self.client._ws_url("/api/chat/stream"),
            "ws://myhost:8080/api/chat/stream",
        )

    def test_url_no_double_slash(self):
        url = self.client._url("/api/health/dashboard")
        self.assertNotIn("//api", url)


# =========================================================================
# 6. CLIClientError
# =========================================================================

class TestCLIClientError(unittest.TestCase):
    """CLIClientError carries status_code."""

    def test_with_status_code(self):
        err = CLIClientError("Not found", status_code=404)
        self.assertEqual(err.status_code, 404)
        self.assertIn("Not found", str(err))

    def test_without_status_code(self):
        err = CLIClientError("Connection failed")
        self.assertIsNone(err.status_code)


# =========================================================================
# 7. format_models_table
# =========================================================================

class TestFormatModelsTable(unittest.TestCase):
    """Test model list formatting."""

    def test_empty_list(self):
        out = format_models_table([], color=False)
        self.assertIn("No models", out)

    def test_single_model(self):
        models = [{"name": "llama3:8b", "size": "4.7 GB", "family": "llama", "quantization": "Q4_0"}]
        out = format_models_table(models, color=False)
        self.assertIn("llama3:8b", out)
        self.assertIn("4.7 GB", out)
        self.assertIn("llama", out)
        self.assertIn("Q4_0", out)

    def test_multiple_models_count(self):
        models = [
            {"name": "m1", "size": "1B", "family": "f1"},
            {"name": "m2", "size": "2B", "family": "f2"},
            {"name": "m3", "size": "3B", "family": "f3"},
        ]
        out = format_models_table(models, color=False)
        self.assertIn("3", out)  # count
        self.assertIn("model(s)", out)

    def test_missing_fields_handled(self):
        models = [{"name": "minimal"}]
        out = format_models_table(models, color=False)
        self.assertIn("minimal", out)


# =========================================================================
# 8. format_status
# =========================================================================

class TestFormatStatus(unittest.TestCase):
    """Test status dashboard formatting."""

    def test_basic_fields(self):
        data = {"version": "2.4.0", "model_count": 12, "uptime_seconds": 3600}
        out = format_status(data, color=False)
        self.assertIn("2.4.0", out)
        self.assertIn("12", out)
        self.assertIn("60 min", out)

    def test_ollama_connected(self):
        data = {"version": "2.4.0", "ollama_status": {"connected": True}}
        out = format_status(data, color=False)
        self.assertIn("connected", out)

    def test_ollama_disconnected(self):
        data = {"version": "2.4.0", "ollama_status": {"connected": False}}
        out = format_status(data, color=False)
        self.assertIn("disconnected", out)

    def test_empty_data(self):
        out = format_status({}, color=False)
        self.assertIn("Status", out)


# =========================================================================
# 9. Spinner
# =========================================================================

class TestSpinner(unittest.TestCase):
    """Spinner context manager protocol."""

    def test_spinner_context_manager(self):
        s = Spinner("test", enabled=False)
        with s:
            pass  # Should not raise

    def test_spinner_disabled_no_thread(self):
        s = Spinner("test", enabled=False)
        with s:
            self.assertIsNone(s._thread)


# =========================================================================
# 10. Click CLI command registration
# =========================================================================

class TestCLICommandRegistration(unittest.TestCase):
    """Verify all commands are registered on the Click group."""

    def setUp(self):
        # Load main.py in isolation -- it imports from sibling modules
        # which are already in sys.modules from the isolated load above
        self.main_mod = _load("opti_oignon.cli.main", CLI_DIR / "main.py")

    def test_cli_group_exists(self):
        self.assertTrue(hasattr(self.main_mod, "cli"))

    def test_ask_command(self):
        cmds = self.main_mod.cli.commands
        self.assertIn("ask", cmds)

    def test_models_command(self):
        self.assertIn("models", self.main_mod.cli.commands)

    def test_status_command(self):
        self.assertIn("status", self.main_mod.cli.commands)

    def test_backup_group(self):
        self.assertIn("backup", self.main_mod.cli.commands)
        backup_grp = self.main_mod.cli.commands["backup"]
        self.assertIn("export", backup_grp.commands)
        self.assertIn("import", backup_grp.commands)

    def test_rag_group(self):
        self.assertIn("rag", self.main_mod.cli.commands)
        rag_grp = self.main_mod.cli.commands["rag"]
        self.assertIn("ingest", rag_grp.commands)
        self.assertIn("query", rag_grp.commands)

    def test_config_group(self):
        self.assertIn("config", self.main_mod.cli.commands)
        cfg_grp = self.main_mod.cli.commands["config"]
        self.assertIn("set", cfg_grp.commands)
        self.assertIn("reset", cfg_grp.commands)

    def test_main_entry_function(self):
        self.assertTrue(callable(self.main_mod.main))


# =========================================================================
# 11. Prompt resolution
# =========================================================================

class TestPromptResolution(unittest.TestCase):
    """Test _resolve_prompt helper."""

    def setUp(self):
        self.main_mod = _load("opti_oignon.cli.main", CLI_DIR / "main.py")

    def test_direct_string(self):
        result = self.main_mod._resolve_prompt("hello", None, False)
        self.assertEqual(result, "hello")

    def test_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("prompt from file")
            f.flush()
            result = self.main_mod._resolve_prompt(None, f.name, False)
            self.assertEqual(result, "prompt from file")
            os.unlink(f.name)

    def test_empty_returns_empty(self):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            result = self.main_mod._resolve_prompt(None, None, False)
            self.assertEqual(result, "")


# =========================================================================
# 12. Package structure
# =========================================================================

class TestPackageStructure(unittest.TestCase):
    """All S122 files exist."""

    def test_cli_package_exists(self):
        self.assertTrue(CLI_DIR.is_dir())

    def test_all_files_exist(self):
        for f in S122_PYTHON_FILES:
            with self.subTest(file=f.name):
                self.assertTrue(f.exists(), f"Missing: {f}")

    def test_init_has_docstring(self):
        content = _read(CLI_DIR / "__init__.py")
        self.assertIn("S122", content)


# =========================================================================
# 13. No Ollama imports
# =========================================================================

class TestNoOllamaImports(unittest.TestCase):
    """CLI must not import ollama."""

    def test_no_ollama_in_cli(self):
        for f in S122_PYTHON_FILES:
            content = _read(f)
            self.assertNotIn("import ollama", content,
                             f"{f.name} must not import ollama")
            self.assertNotIn("from ollama", content,
                             f"{f.name} must not import from ollama")


# =========================================================================
# 14. English-only code comments
# =========================================================================

class TestEnglishComments(unittest.TestCase):
    """All code comments and docstrings in English (no French)."""

    FRENCH_PATTERNS = re.compile(
        r"\b(requete|reponse|envoie|connexion|modele|genere|erreur|"
        r"fermee|supprime|telecharge|parametr|configur[ée]|r[ée]cup[ée]r)\b",
        re.IGNORECASE,
    )

    def test_no_french_in_cli_files(self):
        for f in S122_PYTHON_FILES:
            content = _read(f)
            # Check only comments and docstrings (approximate)
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    match = self.FRENCH_PATTERNS.search(stripped)
                    self.assertIsNone(
                        match,
                        f"Possible French in {f.name}:{i}: '{match.group() if match else ''}'",
                    )


# =========================================================================
# 15. AST validation
# =========================================================================

class TestASTValidation(unittest.TestCase):
    """All S122 Python files must parse without errors."""

    def test_ast_valid(self):
        for f in S122_PYTHON_FILES:
            with self.subTest(file=f.name):
                source = _read(f)
                try:
                    ast.parse(source, filename=str(f))
                except SyntaxError as exc:
                    self.fail(f"SyntaxError in {f.name}: {exc}")


# =========================================================================
# 16. Entry point in pyproject.toml
# =========================================================================

class TestEntryPoints(unittest.TestCase):
    """Verify oo entry point is declared."""

    def test_pyproject_oo_script(self):
        content = _read(PYPROJECT)
        self.assertIn('oo = "opti_oignon.cli.main:main"', content)

    def test_pyproject_cli_extra(self):
        content = _read(PYPROJECT)
        self.assertIn("[project.optional-dependencies]", content)
        self.assertIn("cli", content)
        self.assertIn("click", content)
        self.assertIn("httpx", content)
        self.assertIn("websockets", content)

    def test_setup_py_exists(self):
        self.assertTrue(SETUP_PY.exists())


# =========================================================================
# 17. Version bump
# =========================================================================

class TestVersionBump(unittest.TestCase):
    """Verify version is 2.4.0."""

    def test_version_file(self):
        content = _read(VERSION_FILE)
        self.assertIn('"2.4.0"', content)


# =========================================================================
# 18. OOClient _handle_response
# =========================================================================

class TestOOClientHandleResponse(unittest.TestCase):
    """Test HTTP response handling logic."""

    def setUp(self):
        self.client = OOClient(config=CLIConfig())

    def test_400_raises_error(self):
        resp = MagicMock()
        resp.status_code = 400
        resp.json.return_value = {"detail": "Bad request"}
        resp.text = "Bad request"
        with self.assertRaises(CLIClientError) as ctx:
            self.client._handle_response(resp)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_500_raises_error(self):
        resp = MagicMock()
        resp.status_code = 500
        resp.json.return_value = {"detail": "Server error"}
        resp.text = "Server error"
        with self.assertRaises(CLIClientError) as ctx:
            self.client._handle_response(resp)
        self.assertEqual(ctx.exception.status_code, 500)

    def test_200_returns_json(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"ok": True}
        result = self.client._handle_response(resp)
        self.assertEqual(result, {"ok": True})

    def test_200_non_json_returns_text(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        resp.text = "plain text"
        result = self.client._handle_response(resp)
        self.assertEqual(result, "plain text")


# =========================================================================

if __name__ == "__main__":
    unittest.main()
