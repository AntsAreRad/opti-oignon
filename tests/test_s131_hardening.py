"""
Tests for S131 -- Conversation RAM Wipe + Ollama Log Control + Swap/Network Hardening.

Validates:
- Part 1: conversation_wipe.py (register, wipe, wipe_all, auto-wipe, Bulbe per-turn)
- Part 2: ollama_log_proxy.py (config detection, env recommendations, sanitize)
- Part 3: secure_bytes.py swap protection (check function, startup, Bulbe enforcement)
- Part 4: network_hardening.py (DNS check, proxy check, port check)
- Part 5: API routes (hardening endpoints in routes_security.py)
- Part 6: Frontend files (hardening.ts, HardeningPanel.svelte, SecurityPanel, ChatControlBar)
- Part 7: Version bump 2.9.1 -> 2.9.3, no French, INSTALL.md

Target: ~42 tests
"""

import ast
import importlib.util
import os
import re
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
FRONTEND_SRC = os.path.join(PROJECT_ROOT, "frontend", "src")
COMPONENTS_DIR = os.path.join(FRONTEND_SRC, "lib", "components", "settings")
CHAT_DIR = os.path.join(FRONTEND_SRC, "lib", "components", "chat")
API_TS_DIR = os.path.join(FRONTEND_SRC, "lib", "api")
API_DIR = os.path.join(BACKEND_DIR, "api")


def _load_module(name, path):
    """Load a Python module from file path without triggering __init__ imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if "opti_oignon" not in sys.modules:
        parent = types.ModuleType("opti_oignon")
        sys.modules["opti_oignon"] = parent
    if "opti_oignon.config" not in sys.modules:
        cfg_mod = types.ModuleType("opti_oignon.config")
        cfg_mod.DATA_DIR = tempfile.mkdtemp()
        sys.modules["opti_oignon.config"] = cfg_mod
    if "opti_oignon.db_encryption" not in sys.modules:
        dbe = types.ModuleType("opti_oignon.db_encryption")
        def _fake_conn(db_path, **kw):
            return sqlite3.connect(str(db_path), check_same_thread=False)
        dbe.get_encrypted_connection = _fake_conn
        dbe.SQLCIPHER_AVAILABLE = False
        sys.modules["opti_oignon.db_encryption"] = dbe
    # Stub for signed_audit_log
    if "opti_oignon.signed_audit_log" not in sys.modules:
        sal = types.ModuleType("opti_oignon.signed_audit_log")
        sal.chain_log = MagicMock(return_value=1)
        sal.signed_audit_log = None
        sal.SIGNED_AUDIT_AVAILABLE = True
        sys.modules["opti_oignon.signed_audit_log"] = sal
    # Stub for security_mode
    if "opti_oignon.security_mode" not in sys.modules:
        sm = types.ModuleType("opti_oignon.security_mode")
        sm.is_bulbe = MagicMock(return_value=False)
        sm.is_daily = MagicMock(return_value=True)
        sm.get_current_mode = MagicMock(return_value="daily")
        sys.modules["opti_oignon.security_mode"] = sm
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# =========================================================================
# Part 1: conversation_wipe.py
# =========================================================================

class TestConversationWipe(unittest.TestCase):
    """Tests for ConversationWipeManager."""

    def setUp(self):
        self.mod = _load_module(
            "opti_oignon.conversation_wipe",
            os.path.join(BACKEND_DIR, "conversation_wipe.py"),
        )
        # Fresh manager for each test
        self.mgr = self.mod.ConversationWipeManager()

    def test_feature_flag(self):
        """CONVERSATION_WIPE_AVAILABLE is True."""
        self.assertTrue(self.mod.CONVERSATION_WIPE_AVAILABLE)

    def test_register_and_wipe_list(self):
        """Register a list buffer and wipe it."""
        data = ["secret message 1", "secret message 2"]
        self.mgr.register_buffer("conv-1", data, "messages")
        result = self.mgr.wipe("conv-1")
        self.assertEqual(result.conversation_id, "conv-1")
        self.assertEqual(result.buffers_wiped, 1)
        self.assertTrue(result.success)
        # List should be cleared
        self.assertEqual(len(data), 0)

    def test_register_and_wipe_dict(self):
        """Register a dict buffer and wipe it."""
        data = {"role": "user", "content": "top secret"}
        self.mgr.register_buffer("conv-2", data)
        result = self.mgr.wipe("conv-2")
        self.assertGreater(result.buffers_wiped, 0)
        self.assertEqual(len(data), 0)

    def test_register_string_buffer(self):
        """Register a string (immutable) and wipe."""
        s = "sensitive system prompt"
        self.mgr.register_buffer("conv-3", s, "system_prompt")
        result = self.mgr.wipe("conv-3")
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.buffers_wiped, 1)

    def test_wipe_nonexistent(self):
        """Wipe a non-existent conversation returns zero buffers."""
        result = self.mgr.wipe("does-not-exist")
        self.assertEqual(result.buffers_wiped, 0)
        self.assertTrue(result.success)

    def test_wipe_all(self):
        """wipe_all clears all registered conversations."""
        self.mgr.register_buffer("c1", ["msg1"])
        self.mgr.register_buffer("c2", ["msg2"])
        self.mgr.register_buffer("c3", ["msg3"])
        results = self.mgr.wipe_all()
        self.assertEqual(len(results), 3)
        self.assertEqual(len(self.mgr.get_registered_conversations()), 0)

    def test_on_conversation_close_enabled(self):
        """on_conversation_close wipes when auto_wipe_on_close is true."""
        self.mgr._config = {"auto_wipe_on_close": True}
        self.mgr.register_buffer("conv-close", ["data"])
        result = self.mgr.on_conversation_close("conv-close")
        self.assertIsNotNone(result)
        self.assertEqual(result.buffers_wiped, 1)

    def test_on_conversation_close_disabled(self):
        """on_conversation_close returns None when disabled."""
        self.mgr._config = {"auto_wipe_on_close": False}
        self.mgr.register_buffer("conv-skip", ["data"])
        result = self.mgr.on_conversation_close("conv-skip")
        self.assertIsNone(result)

    def test_get_status(self):
        """get_status returns expected structure."""
        self.mgr.register_buffer("conv-s", ["x"])
        status = self.mgr.get_status()
        self.assertTrue(status["available"])
        self.assertEqual(status["active_conversations"], 1)
        self.assertEqual(status["total_registered_buffers"], 1)
        self.assertIn("memset_available", status)

    def test_wipe_result_has_memset_field(self):
        """WipeResult includes memset_available."""
        result = self.mod.WipeResult(conversation_id="test")
        self.assertIsInstance(result.memset_available, bool)

    def test_module_level_aliases(self):
        """Module-level convenience functions exist."""
        self.assertTrue(callable(self.mod.register_buffer))
        self.assertTrue(callable(self.mod.wipe_conversation))
        self.assertTrue(callable(self.mod.wipe_all_conversations))
        self.assertTrue(callable(self.mod.on_conversation_close))
        self.assertTrue(callable(self.mod.on_bulbe_response))


# =========================================================================
# Part 2: ollama_log_proxy.py
# =========================================================================

class TestOllamaLogProxy(unittest.TestCase):
    """Tests for Ollama log sanitization and config detection."""

    def setUp(self):
        self.mod = _load_module(
            "opti_oignon.ollama_log_proxy",
            os.path.join(BACKEND_DIR, "ollama_log_proxy.py"),
        )

    def test_feature_flag(self):
        """OLLAMA_LOG_PROXY_AVAILABLE is True."""
        self.assertTrue(self.mod.OLLAMA_LOG_PROXY_AVAILABLE)

    def test_sanitize_prompt_content(self):
        """Prompt content JSON is redacted."""
        line = 'DEBUG: {"prompt": "What is my password?"}'
        sanitized = self.mod.sanitize_ollama_prompt_log(line)
        self.assertIn("[REDACTED]", sanitized)
        self.assertNotIn("password", sanitized)

    def test_sanitize_email(self):
        """Email addresses are redacted."""
        line = "User email: john.doe@example.com"
        sanitized = self.mod.sanitize_ollama_prompt_log(line)
        self.assertIn("[REDACTED]", sanitized)
        self.assertNotIn("john.doe", sanitized)

    def test_sanitize_api_key(self):
        """API keys are redacted."""
        line = "api_key=sk-abc123def456"
        sanitized = self.mod.sanitize_ollama_prompt_log(line)
        self.assertIn("[REDACTED]", sanitized)
        self.assertNotIn("sk-abc123", sanitized)

    def test_sanitize_bearer_token(self):
        """Bearer tokens are redacted."""
        line = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9"
        sanitized = self.mod.sanitize_ollama_prompt_log(line)
        self.assertIn("[REDACTED]", sanitized)

    def test_sanitize_empty_string(self):
        """Empty string returns empty."""
        self.assertEqual(self.mod.sanitize_ollama_prompt_log(""), "")

    def test_recommendations_daily(self):
        """Daily mode recommendations include warn level."""
        recs = self.mod.get_ollama_env_recommendations("daily")
        self.assertEqual(recs["OLLAMA_LOG_LEVEL"], "warn")
        self.assertEqual(recs["OLLAMA_DEBUG"], "0")

    def test_recommendations_bulbe(self):
        """Bulbe mode recommendations include error level and 0 keepalive."""
        recs = self.mod.get_ollama_env_recommendations("bulbe")
        self.assertEqual(recs["OLLAMA_LOG_LEVEL"], "error")
        self.assertEqual(recs["OLLAMA_KEEP_ALIVE"], "0")

    def test_check_config_returns_dataclass(self):
        """check_ollama_log_config returns OllamaLogConfig."""
        cfg = self.mod.check_ollama_log_config()
        self.assertIsInstance(cfg, self.mod.OllamaLogConfig)
        self.assertIsInstance(cfg.log_level, str)
        self.assertIsInstance(cfg.is_verbose, bool)


# =========================================================================
# Part 3: Swap protection (secure_bytes.py additions)
# =========================================================================

class TestSwapProtection(unittest.TestCase):
    """Tests for swap encryption check in secure_bytes.py."""

    def setUp(self):
        self.mod = _load_module(
            "opti_oignon.secure_bytes",
            os.path.join(BACKEND_DIR, "secure_bytes.py"),
        )

    def test_swap_check_result_fields(self):
        """SwapCheckResult has all required fields."""
        r = self.mod.SwapCheckResult()
        self.assertFalse(r.swap_enabled)
        self.assertFalse(r.encrypted)
        self.assertTrue(r.safe)
        self.assertIsInstance(r.devices, list)
        self.assertTrue(r.platform_supported)

    def test_check_swap_encrypted_returns_result(self):
        """check_swap_encrypted returns a SwapCheckResult."""
        result = self.mod.check_swap_encrypted()
        self.assertIsInstance(result, self.mod.SwapCheckResult)

    def test_check_swap_no_swap_is_safe(self):
        """If /proc/swaps has no entries, result is safe."""
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            mock_open.return_value.readlines = MagicMock(
                return_value=["Filename\tType\tSize\tUsed\tPriority\n"]
            )
            with patch("os.path.isfile", return_value=True):
                with patch.object(self.mod, "sys") as mock_sys:
                    mock_sys.platform = "linux"
                    result = self.mod.check_swap_encrypted()
        # In container, likely no swap
        self.assertTrue(result.safe)

    def test_check_swap_non_linux(self):
        """Non-Linux platform returns graceful no-op."""
        original = sys.platform
        with patch.object(self.mod.sys, "platform", "darwin"):
            result = self.mod.check_swap_encrypted()
        self.assertFalse(result.platform_supported)
        self.assertTrue(result.safe)

    def test_swap_startup_check_callable(self):
        """swap_startup_check is callable."""
        self.assertTrue(callable(self.mod.swap_startup_check))

    def test_swap_startup_no_crash(self):
        """swap_startup_check does not crash in test container."""
        # Should not raise (no swap in container)
        self.mod.swap_startup_check()


# =========================================================================
# Part 4: network_hardening.py
# =========================================================================

class TestNetworkHardening(unittest.TestCase):
    """Tests for network hardening checks."""

    def setUp(self):
        self.mod = _load_module(
            "opti_oignon.network_hardening",
            os.path.join(BACKEND_DIR, "network_hardening.py"),
        )

    def test_feature_flag(self):
        """NETWORK_HARDENING_AVAILABLE is True."""
        self.assertTrue(self.mod.NETWORK_HARDENING_AVAILABLE)

    def test_dns_check_returns_result(self):
        """check_dns_encryption returns DnsCheckResult."""
        result = self.mod.check_dns_encryption()
        self.assertIsInstance(result, self.mod.DnsCheckResult)
        self.assertIn(result.protocol, ("doh", "dot", "plain", "stub", "unknown"))

    def test_proxy_check_no_config(self):
        """Proxy check with no config returns not configured."""
        result = self.mod.check_proxy_config()
        self.assertIsInstance(result, self.mod.ProxyCheckResult)
        # In test env, no proxy configured
        self.assertFalse(result.reachable)

    def test_port_check_returns_list(self):
        """check_listening_ports returns list of PortInfo."""
        result = self.mod.check_listening_ports()
        self.assertIsInstance(result, list)

    def test_full_status_structure(self):
        """get_full_network_status returns complete structure."""
        status = self.mod.get_full_network_status()
        self.assertTrue(status["available"])
        self.assertIn("dns", status)
        self.assertIn("proxy", status)
        self.assertIn("ports", status)
        self.assertIn("warnings", status)
        self.assertIsInstance(status["warnings"], list)

    def test_full_status_dns_fields(self):
        """DNS substatus has expected fields."""
        status = self.mod.get_full_network_status()
        dns = status["dns"]
        self.assertIn("encrypted", dns)
        self.assertIn("protocol", dns)
        self.assertIn("resolver", dns)

    def test_port_info_expected_field(self):
        """PortInfo dataclass has expected field."""
        pi = self.mod.PortInfo(port=8000, expected=True)
        self.assertTrue(pi.expected)
        pi2 = self.mod.PortInfo(port=9999, expected=False)
        self.assertFalse(pi2.expected)


# =========================================================================
# Part 5: API routes (routes_security.py hardening endpoints)
# =========================================================================

class TestRoutesSecurityHardening(unittest.TestCase):
    """Test that hardening endpoints exist in routes_security.py."""

    def setUp(self):
        self.path = os.path.join(API_DIR, "routes_security.py")
        with open(self.path, "r", encoding="utf-8") as f:
            self.content = f.read()
        self.tree = ast.parse(self.content)

    def test_ast_valid(self):
        """routes_security.py is valid Python."""
        self.assertIsNotNone(self.tree)

    def test_conversation_wipe_single_endpoint(self):
        """POST conversation-wipe/{conversation_id} endpoint exists."""
        self.assertIn("conversation_wipe_single", self.content)
        self.assertIn("/conversation-wipe/{conversation_id}", self.content)

    def test_conversation_wipe_all_endpoint(self):
        """POST conversation-wipe/all endpoint exists."""
        self.assertIn("conversation_wipe_all", self.content)
        self.assertIn("/conversation-wipe/all", self.content)

    def test_hardening_status_endpoint(self):
        """GET hardening/status endpoint exists."""
        self.assertIn("hardening_status", self.content)
        self.assertIn("/hardening/status", self.content)

    def test_hardening_network_endpoint(self):
        """GET hardening/network endpoint exists."""
        self.assertIn("hardening_network", self.content)
        self.assertIn("/hardening/network", self.content)


# =========================================================================
# Part 6: Frontend files
# =========================================================================

class TestFrontendS131(unittest.TestCase):
    """Validate frontend files for S131."""

    def test_hardening_ts_exists(self):
        """hardening.ts API client exists."""
        path = os.path.join(API_TS_DIR, "hardening.ts")
        self.assertTrue(os.path.isfile(path))

    def test_hardening_ts_exports(self):
        """hardening.ts exports key functions."""
        path = os.path.join(API_TS_DIR, "hardening.ts")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("wipeConversation", content)
        self.assertIn("wipeAllConversations", content)
        self.assertIn("getHardeningStatus", content)
        self.assertIn("getNetworkStatus", content)

    def test_hardening_ts_types(self):
        """hardening.ts defines required types."""
        path = os.path.join(API_TS_DIR, "hardening.ts")
        with open(path, "r") as f:
            content = f.read()
        for t in ("WipeResult", "HardeningStatus", "SwapStatus",
                   "OllamaLogStatus", "NetworkStatus"):
            self.assertIn(t, content, f"Type {t} missing")

    def test_hardening_panel_exists(self):
        """HardeningPanel.svelte exists."""
        path = os.path.join(COMPONENTS_DIR, "HardeningPanel.svelte")
        self.assertTrue(os.path.isfile(path))

    def test_hardening_panel_sections(self):
        """HardeningPanel has all 4 sections."""
        path = os.path.join(COMPONENTS_DIR, "HardeningPanel.svelte")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("Conversation RAM Wipe", content)
        self.assertIn("Ollama Logging", content)
        self.assertIn("Swap Protection", content)
        self.assertIn("Network Hardening", content)

    def test_hardening_panel_html_balance(self):
        """HardeningPanel.svelte has balanced div tags."""
        path = os.path.join(COMPONENTS_DIR, "HardeningPanel.svelte")
        with open(path, "r") as f:
            content = f.read()
        # Strip Svelte blocks to avoid false matches in {#if ...} etc.
        stripped = re.sub(r"\{[#/:][^}]*\}", "", content)
        opens = len(re.findall(r"<div[\s>]", stripped))
        closes = len(re.findall(r"</div>", stripped))
        self.assertEqual(opens, closes, f"div imbalance: {opens} opens, {closes} closes")

    def test_hardening_panel_css_vars(self):
        """HardeningPanel uses only --oo-* CSS variables (hex only in fallback)."""
        path = os.path.join(COMPONENTS_DIR, "HardeningPanel.svelte")
        with open(path, "r") as f:
            content = f.read()
        # Find hex colors: must be 3 or 6 hex digits preceded by #
        # Exclude Svelte syntax like {#each, {#if, etc.
        for match in re.finditer(r"(?<!\{)#[0-9a-fA-F]{3,8}\b", content):
            hex_val = match.group()
            start = max(0, match.start() - 50)
            context = content[start:match.start()]
            self.assertIn("var(--oo-", context,
                          f"Hex {hex_val} not in var() fallback: ...{context}")

    def test_security_panel_hardening_tab(self):
        """SecurityPanel.svelte includes Hardening tab."""
        path = os.path.join(COMPONENTS_DIR, "SecurityPanel.svelte")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("HardeningPanel", content)
        self.assertIn("hardening", content)
        self.assertIn("Hardening", content)

    def test_chat_control_bar_wipe_button(self):
        """ChatControlBar.svelte has wipe conversation button."""
        path = os.path.join(CHAT_DIR, "ChatControlBar.svelte")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("handleWipeConversation", content)
        self.assertIn("wipeAvailable", content)
        self.assertIn("conversation-wipe", content)

    def test_no_french_in_frontend(self):
        """No French text in new frontend files."""
        french = re.compile(
            r"\b(mot de passe|connexion|parametre|securite|effacer|supprim|"
            r"disponible|indisponible|erreur|valider)\b", re.I
        )
        files = [
            os.path.join(API_TS_DIR, "hardening.ts"),
            os.path.join(COMPONENTS_DIR, "HardeningPanel.svelte"),
        ]
        for fpath in files:
            with open(fpath, "r") as f:
                content = f.read()
            matches = french.findall(content)
            self.assertEqual(len(matches), 0,
                             f"French in {os.path.basename(fpath)}: {matches}")


# =========================================================================
# Part 7: Version bump + INSTALL.md + no French in backend
# =========================================================================

class TestVersionAndDocs(unittest.TestCase):
    """Version bump and documentation checks."""

    def test_version_bump(self):
        """Version is 2.9.3."""
        version_path = os.path.join(BACKEND_DIR, "__version__.py")
        with open(version_path, "r") as f:
            content = f.read()
        self.assertIn('"3.0.0"', content)

    def test_install_md_high_security(self):
        """INSTALL.md has High-Security Deployment section."""
        path = os.path.join(PROJECT_ROOT, "INSTALL.md")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("High-Security Deployment", content)
        self.assertIn("Swap Configuration", content)
        self.assertIn("zram", content)
        self.assertIn("LUKS", content)

    def test_security_yaml_hardening(self):
        """security.yaml has hardening section."""
        path = os.path.join(BACKEND_DIR, "config", "security.yaml")
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        self.assertIn("hardening", data)
        self.assertTrue(data["hardening"]["auto_wipe_on_close"])
        self.assertTrue(data["hardening"]["bulbe_wipe_per_turn"])
        self.assertTrue(data["hardening"]["require_encrypted_swap"])

    def test_security_yaml_ollama(self):
        """security.yaml has ollama section."""
        path = os.path.join(BACKEND_DIR, "config", "security.yaml")
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        self.assertIn("ollama", data)
        self.assertTrue(data["ollama"]["log_sanitization"])

    def test_no_french_in_new_backend_modules(self):
        """No French in new S131 backend modules."""
        french = re.compile(
            r"\b(mot de passe|connexion|parametre|securite|effacer|supprim|"
            r"configurer|disponible)\b", re.I
        )
        files = [
            os.path.join(BACKEND_DIR, "conversation_wipe.py"),
            os.path.join(BACKEND_DIR, "ollama_log_proxy.py"),
            os.path.join(BACKEND_DIR, "network_hardening.py"),
        ]
        for fpath in files:
            with open(fpath, "r") as f:
                content = f.read()
            matches = french.findall(content)
            self.assertEqual(len(matches), 0,
                             f"French in {os.path.basename(fpath)}: {matches}")

    def test_all_new_backend_ast_valid(self):
        """All new S131 backend files are valid Python."""
        files = [
            os.path.join(BACKEND_DIR, "conversation_wipe.py"),
            os.path.join(BACKEND_DIR, "ollama_log_proxy.py"),
            os.path.join(BACKEND_DIR, "network_hardening.py"),
            os.path.join(BACKEND_DIR, "secure_bytes.py"),
            os.path.join(API_DIR, "routes_security.py"),
        ]
        for fpath in files:
            with open(fpath, "r") as f:
                tree = ast.parse(f.read())
            self.assertIsNotNone(tree, f"AST failed for {fpath}")


if __name__ == "__main__":
    unittest.main()
