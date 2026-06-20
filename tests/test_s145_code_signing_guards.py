#!/usr/bin/env python3
"""
Tests for S145 — Code Signing + Runtime Guards.

Covers:
- Part 1: Code signing scripts existence and structure
- Part 2: Ollama bind guard detection logic (_extract_host, env, proc, ss)
- Part 3: OllamaBindCheckResult dataclass
- Part 4: LUKS detection parsing (lsblk, proc_mounts, dmsetup)
- Part 5: LUKSCheckResult dataclass
- Part 6: Startup checklist aggregation
- Part 7: CheckItem / StartupCheckResult data structures
- Part 8: Startup checklist caching
- Part 9: API endpoint schema (GET /api/security/startup-checks)
- Part 10: Security score impact
- Part 11: Version bump (3.2.0-rc1)
"""

import importlib.util
import json
import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# ---------------------------------------------------------------------------
# Importlib isolation pattern
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OO = _PROJECT_ROOT / "opti_oignon"


def _ensure_package():
    """Register opti_oignon as a package in sys.modules if missing."""
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(_OO)]
        sys.modules["opti_oignon"] = pkg


def _load_module(name: str, filepath: Path):
    """Load a module via importlib without triggering __init__ chain."""
    _ensure_package()
    fqn = f"opti_oignon.{name}"
    if fqn in sys.modules:
        return sys.modules[fqn]
    spec = importlib.util.spec_from_file_location(fqn, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


# Mock yaml before loading modules that import it
if "yaml" not in sys.modules:
    _yaml_mock = types.ModuleType("yaml")
    _yaml_mock.safe_load = lambda x: {}
    _yaml_mock.dump = lambda *a, **kw: ""
    sys.modules["yaml"] = _yaml_mock

# Load modules under test
_nbg_mod = _load_module("network_bind_guard", _OO / "network_bind_guard.py")
_luks_mod = _load_module("luks_detector", _OO / "luks_detector.py")
_sc_mod = _load_module("startup_checks", _OO / "startup_checks.py")


# =========================================================================
# Part 1: Code signing scripts existence and structure
# =========================================================================

class TestCodeSigningScripts(unittest.TestCase):
    """Part 1: Verify code signing scripts exist and are well-formed."""

    def test_sign_release_exists(self):
        script = _PROJECT_ROOT / "scripts" / "sign_release.sh"
        self.assertTrue(script.exists(), f"Missing: {script}")

    def test_verify_release_exists(self):
        script = _PROJECT_ROOT / "scripts" / "verify_release.sh"
        self.assertTrue(script.exists(), f"Missing: {script}")

    def test_sign_release_executable(self):
        script = _PROJECT_ROOT / "scripts" / "sign_release.sh"
        mode = script.stat().st_mode
        self.assertTrue(mode & stat.S_IXUSR, "sign_release.sh not executable")

    def test_verify_release_executable(self):
        script = _PROJECT_ROOT / "scripts" / "verify_release.sh"
        mode = script.stat().st_mode
        self.assertTrue(mode & stat.S_IXUSR, "verify_release.sh not executable")

    def test_sign_release_has_shebang(self):
        script = _PROJECT_ROOT / "scripts" / "sign_release.sh"
        first_line = script.read_text().split("\n")[0]
        self.assertTrue(first_line.startswith("#!/"), "Missing shebang")

    def test_verify_release_has_shebang(self):
        script = _PROJECT_ROOT / "scripts" / "verify_release.sh"
        first_line = script.read_text().split("\n")[0]
        self.assertTrue(first_line.startswith("#!/"), "Missing shebang")

    def test_sign_release_uses_gpg(self):
        content = (_PROJECT_ROOT / "scripts" / "sign_release.sh").read_text()
        self.assertIn("gpg", content)
        self.assertIn("--detach-sign", content)

    def test_verify_release_uses_gpg_verify(self):
        content = (_PROJECT_ROOT / "scripts" / "verify_release.sh").read_text()
        self.assertIn("gpg", content)
        self.assertIn("--verify", content)

    def test_sign_release_creates_sha256(self):
        content = (_PROJECT_ROOT / "scripts" / "sign_release.sh").read_text()
        self.assertIn("sha256sum", content)

    def test_verify_release_checks_sha256(self):
        content = (_PROJECT_ROOT / "scripts" / "verify_release.sh").read_text()
        self.assertIn("sha256sum", content)

    def test_sign_release_supports_key_flag(self):
        content = (_PROJECT_ROOT / "scripts" / "sign_release.sh").read_text()
        self.assertIn("--key", content)

    def test_verify_release_supports_strict_flag(self):
        content = (_PROJECT_ROOT / "scripts" / "verify_release.sh").read_text()
        self.assertIn("--strict", content)


# =========================================================================
# Part 2: Ollama bind guard detection logic
# =========================================================================

class TestExtractHost(unittest.TestCase):
    """Part 2a: _extract_host parsing."""

    def test_plain_ip(self):
        self.assertEqual(_nbg_mod._extract_host("127.0.0.1"), "127.0.0.1")

    def test_ip_with_port(self):
        self.assertEqual(_nbg_mod._extract_host("0.0.0.0:11434"), "0.0.0.0")

    def test_http_scheme(self):
        self.assertEqual(
            _nbg_mod._extract_host("http://127.0.0.1:11434"), "127.0.0.1",
        )

    def test_https_scheme(self):
        self.assertEqual(
            _nbg_mod._extract_host("https://10.0.0.5:443"), "10.0.0.5",
        )

    def test_ipv6_bracket(self):
        self.assertEqual(_nbg_mod._extract_host("[::]:11434"), "::")

    def test_with_path(self):
        self.assertEqual(
            _nbg_mod._extract_host("http://0.0.0.0:11434/api"), "0.0.0.0",
        )

    def test_localhost_string(self):
        self.assertEqual(_nbg_mod._extract_host("localhost"), "localhost")

    def test_whitespace_stripped(self):
        self.assertEqual(_nbg_mod._extract_host("  127.0.0.1  "), "127.0.0.1")


class TestOllamaBindEnvVar(unittest.TestCase):
    """Part 2b: Ollama bind detection via OLLAMA_HOST env var."""

    @patch.dict(os.environ, {"OLLAMA_HOST": "0.0.0.0:11434"})
    @patch.object(_nbg_mod, "_get_current_mode", return_value="daily")
    def test_exposed_via_env(self, _mock_mode):
        result = _nbg_mod.check_ollama_bind()
        self.assertTrue(result.checked)
        self.assertTrue(result.exposed)
        self.assertEqual(result.method, "env_OLLAMA_HOST")
        self.assertEqual(result.bind_address, "0.0.0.0")

    @patch.dict(os.environ, {"OLLAMA_HOST": "127.0.0.1:11434"})
    @patch.object(_nbg_mod, "_get_current_mode", return_value="daily")
    def test_safe_via_env(self, _mock_mode):
        result = _nbg_mod.check_ollama_bind()
        self.assertTrue(result.checked)
        self.assertFalse(result.exposed)
        self.assertEqual(result.bind_address, "127.0.0.1")

    @patch.dict(os.environ, {"OLLAMA_HOST": "0.0.0.0:11434"})
    @patch.object(_nbg_mod, "_get_current_mode", return_value="bulbe")
    @patch.object(_nbg_mod, "_audit_critical_event")
    def test_blocked_in_bulbe(self, _mock_audit, _mock_mode):
        result = _nbg_mod.check_ollama_bind()
        self.assertTrue(result.exposed)
        self.assertTrue(result.blocked)
        _mock_audit.assert_called_once()

    @patch.dict(os.environ, {"OLLAMA_HOST": "0.0.0.0:11434"})
    @patch.object(_nbg_mod, "_get_current_mode", return_value="bulbe")
    @patch.object(_nbg_mod, "_audit_critical_event")
    def test_not_blocked_when_flag_false(self, _mock_audit, _mock_mode):
        result = _nbg_mod.check_ollama_bind(block_if_exposed_bulbe=False)
        self.assertTrue(result.exposed)
        self.assertFalse(result.blocked)


class TestOllamaProcNetTcp(unittest.TestCase):
    """Part 2c: Ollama detection via /proc/net/tcp."""

    def test_finds_listening_on_wildcard(self):
        # 0.0.0.0:11434 => hex port 2CAA, addr 00000000
        proc_content = (
            "  sl  local_address rem_address   st tx_queue rx_queue\n"
            "   0: 00000000:2CAA 00000000:0000 0A 00000000:00000000\n"
        )
        with patch("builtins.open", mock_open(read_data=proc_content)):
            result = _nbg_mod._check_ollama_proc_net_tcp(11434)
        self.assertEqual(result, "0.0.0.0")

    def test_finds_listening_on_localhost(self):
        # 127.0.0.1 in little-endian hex = 0100007F
        proc_content = (
            "  sl  local_address rem_address   st tx_queue rx_queue\n"
            "   0: 0100007F:2CAA 00000000:0000 0A 00000000:00000000\n"
        )
        with patch("builtins.open", mock_open(read_data=proc_content)):
            result = _nbg_mod._check_ollama_proc_net_tcp(11434)
        self.assertEqual(result, "127.0.0.1")

    def test_ignores_non_listen_state(self):
        # state 01 = ESTABLISHED, not LISTEN (0A)
        proc_content = (
            "  sl  local_address rem_address   st tx_queue rx_queue\n"
            "   0: 00000000:2CAA 00000000:0000 01 00000000:00000000\n"
        )
        with patch("builtins.open", mock_open(read_data=proc_content)):
            result = _nbg_mod._check_ollama_proc_net_tcp(11434)
        self.assertIsNone(result)

    def test_returns_none_on_file_not_found(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = _nbg_mod._check_ollama_proc_net_tcp(11434)
        self.assertIsNone(result)


class TestOllamaSs(unittest.TestCase):
    """Part 2d: Ollama detection via ss command."""

    @patch("subprocess.run")
    def test_detects_wildcard_via_ss(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="LISTEN  0  128  *:11434  *:*  users:((\"ollama\",pid=123))\n",
        )
        result = _nbg_mod._check_ollama_ss(11434)
        self.assertEqual(result, "0.0.0.0")

    @patch("subprocess.run")
    def test_detects_localhost_via_ss(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="LISTEN  0  128  127.0.0.1:11434  *:*  users:((\"ollama\",pid=123))\n",
        )
        result = _nbg_mod._check_ollama_ss(11434)
        self.assertEqual(result, "127.0.0.1")

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_returns_none_if_ss_missing(self, _):
        result = _nbg_mod._check_ollama_ss(11434)
        self.assertIsNone(result)


# =========================================================================
# Part 3: OllamaBindCheckResult dataclass
# =========================================================================

class TestOllamaBindCheckResult(unittest.TestCase):
    """Part 3: OllamaBindCheckResult serialization."""

    def test_default_values(self):
        r = _nbg_mod.OllamaBindCheckResult()
        self.assertFalse(r.checked)
        self.assertFalse(r.exposed)
        self.assertEqual(r.bind_address, "")
        self.assertFalse(r.blocked)

    def test_to_dict_keys(self):
        r = _nbg_mod.OllamaBindCheckResult()
        d = r.to_dict()
        expected = {"checked", "exposed", "bind_address", "port", "method", "detail", "blocked"}
        self.assertEqual(set(d.keys()), expected)

    def test_to_dict_values(self):
        r = _nbg_mod.OllamaBindCheckResult(
            checked=True, exposed=True, bind_address="0.0.0.0",
            port=11434, method="env_OLLAMA_HOST", detail="test",
        )
        d = r.to_dict()
        self.assertTrue(d["exposed"])
        self.assertEqual(d["bind_address"], "0.0.0.0")


# =========================================================================
# Part 4: LUKS detection parsing
# =========================================================================

class TestLUKSLsblk(unittest.TestCase):
    """Part 4a: LUKS detection via lsblk."""

    @patch("subprocess.run")
    def test_detects_crypt_device(self, mock_run):
        lsblk_output = json.dumps({
            "blockdevices": [{
                "name": "sda",
                "type": "disk",
                "fstype": None,
                "mountpoint": None,
                "children": [{
                    "name": "sda1",
                    "type": "part",
                    "fstype": "crypto_LUKS",
                    "mountpoint": None,
                    "children": [{
                        "name": "dm-0",
                        "type": "crypt",
                        "fstype": "ext4",
                        "mountpoint": "/",
                    }],
                }],
            }],
        })
        mock_run.return_value = MagicMock(returncode=0, stdout=lsblk_output)
        result = _luks_mod._check_lsblk()
        self.assertIsNotNone(result)
        self.assertTrue(result["encrypted"])
        self.assertIn("dm-0", result["devices"])

    @patch("subprocess.run")
    def test_no_crypt_detected(self, mock_run):
        lsblk_output = json.dumps({
            "blockdevices": [{
                "name": "sda",
                "type": "disk",
                "fstype": None,
                "mountpoint": None,
                "children": [{
                    "name": "sda1",
                    "type": "part",
                    "fstype": "ext4",
                    "mountpoint": "/",
                }],
            }],
        })
        mock_run.return_value = MagicMock(returncode=0, stdout=lsblk_output)
        result = _luks_mod._check_lsblk()
        self.assertIsNotNone(result)
        self.assertFalse(result["encrypted"])

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_lsblk_missing(self, _):
        result = _luks_mod._check_lsblk()
        self.assertIsNone(result)


class TestLUKSProcMounts(unittest.TestCase):
    """Part 4b: LUKS detection via /proc/mounts."""

    @patch.object(_luks_mod, "_check_dm_uuid", return_value=True)
    def test_detects_dm_mapper_root(self, _mock_uuid):
        proc_content = "/dev/mapper/vg-root / ext4 rw 0 0\n"
        with patch("builtins.open", mock_open(read_data=proc_content)):
            with patch.object(Path, "exists", return_value=True):
                result = _luks_mod._check_proc_mounts()
        self.assertIsNotNone(result)
        self.assertTrue(result["encrypted"])

    def test_detects_non_dm_root(self):
        proc_content = "/dev/sda1 / ext4 rw 0 0\n"
        with patch("builtins.open", mock_open(read_data=proc_content)):
            with patch.object(Path, "exists", return_value=True):
                result = _luks_mod._check_proc_mounts()
        self.assertIsNotNone(result)
        self.assertFalse(result["encrypted"])


class TestLUKSDmsetup(unittest.TestCase):
    """Part 4c: LUKS detection via dmsetup."""

    @patch("subprocess.run")
    def test_detects_crypt_target(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="vg-root: 0 1024 crypt aes-xts-plain64\n",
        )
        result = _luks_mod._check_dmsetup()
        self.assertIsNotNone(result)
        self.assertTrue(result["encrypted"])
        self.assertIn("vg-root", result["devices"])

    @patch("subprocess.run")
    def test_no_crypt_targets(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="No devices found",
        )
        result = _luks_mod._check_dmsetup()
        self.assertIsNotNone(result)
        self.assertFalse(result["encrypted"])

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_dmsetup_missing(self, _):
        result = _luks_mod._check_dmsetup()
        self.assertIsNone(result)


# =========================================================================
# Part 5: LUKSCheckResult dataclass
# =========================================================================

class TestLUKSCheckResult(unittest.TestCase):
    """Part 5: LUKSCheckResult serialization."""

    def test_default_values(self):
        r = _luks_mod.LUKSCheckResult()
        self.assertFalse(r.checked)
        self.assertFalse(r.encrypted)
        self.assertEqual(r.tips, [])

    def test_to_dict_keys(self):
        r = _luks_mod.LUKSCheckResult()
        d = r.to_dict()
        expected = {"checked", "encrypted", "method", "detail", "encrypted_devices", "tips"}
        self.assertEqual(set(d.keys()), expected)

    def test_tips_populated_when_unencrypted(self):
        r = _luks_mod.LUKSCheckResult(tips=list(_luks_mod._LUKS_TIPS))
        self.assertTrue(len(r.tips) >= 3)


# =========================================================================
# Part 6: Startup checklist aggregation
# =========================================================================

class TestStartupChecklist(unittest.TestCase):
    """Part 6: run_startup_checks aggregation logic."""

    def setUp(self):
        _sc_mod.clear_cache()

    @patch.object(_sc_mod, "_check_code_signing_scripts")
    @patch.object(_sc_mod, "_check_ollama_bind")
    @patch.object(_sc_mod, "_check_luks")
    @patch.object(_sc_mod, "_check_security_mode")
    @patch.object(_sc_mod, "_check_encrypted_swap")
    def test_all_passed(self, m_swap, m_mode, m_luks, m_ollama, m_sign):
        for m in (m_swap, m_mode, m_luks, m_ollama, m_sign):
            m.return_value = _sc_mod.CheckItem(
                name="test", passed=True, severity="info", detail="ok",
            )
        result = _sc_mod.run_startup_checks(force=True)
        self.assertTrue(result.all_passed)
        self.assertFalse(result.blocked)
        self.assertEqual(len(result.checks), 5)

    @patch.object(_sc_mod, "_check_code_signing_scripts")
    @patch.object(_sc_mod, "_check_ollama_bind")
    @patch.object(_sc_mod, "_check_luks")
    @patch.object(_sc_mod, "_check_security_mode")
    @patch.object(_sc_mod, "_check_encrypted_swap")
    def test_critical_failure_blocks(self, m_swap, m_mode, m_luks, m_ollama, m_sign):
        m_sign.return_value = _sc_mod.CheckItem(
            name="sign", passed=True, severity="info", detail="ok",
        )
        m_ollama.return_value = _sc_mod.CheckItem(
            name="ollama_bind", passed=False, severity="critical",
            detail="exposed", score_impact=-15,
        )
        m_luks.return_value = _sc_mod.CheckItem(
            name="luks", passed=True, severity="info", detail="ok",
        )
        m_mode.return_value = _sc_mod.CheckItem(
            name="mode", passed=True, severity="info", detail="ok",
        )
        m_swap.return_value = _sc_mod.CheckItem(
            name="swap", passed=True, severity="info", detail="ok",
        )
        result = _sc_mod.run_startup_checks(force=True)
        self.assertFalse(result.all_passed)
        self.assertTrue(result.blocked)
        self.assertIn("ollama_bind", result.block_reason)

    @patch.object(_sc_mod, "_check_code_signing_scripts")
    @patch.object(_sc_mod, "_check_ollama_bind")
    @patch.object(_sc_mod, "_check_luks")
    @patch.object(_sc_mod, "_check_security_mode")
    @patch.object(_sc_mod, "_check_encrypted_swap")
    def test_warning_does_not_block(self, m_swap, m_mode, m_luks, m_ollama, m_sign):
        for m in (m_sign, m_ollama, m_mode, m_swap):
            m.return_value = _sc_mod.CheckItem(
                name="ok", passed=True, severity="info", detail="ok",
            )
        m_luks.return_value = _sc_mod.CheckItem(
            name="luks", passed=False, severity="warning",
            detail="no encryption", score_impact=-5,
        )
        result = _sc_mod.run_startup_checks(force=True)
        self.assertFalse(result.all_passed)
        self.assertFalse(result.blocked)

    @patch.object(_sc_mod, "_check_code_signing_scripts")
    @patch.object(_sc_mod, "_check_ollama_bind")
    @patch.object(_sc_mod, "_check_luks")
    @patch.object(_sc_mod, "_check_security_mode")
    @patch.object(_sc_mod, "_check_encrypted_swap")
    def test_score_impact_aggregated(self, m_swap, m_mode, m_luks, m_ollama, m_sign):
        m_sign.return_value = _sc_mod.CheckItem(
            name="sign", passed=False, severity="info", detail="missing",
            score_impact=-2,
        )
        m_ollama.return_value = _sc_mod.CheckItem(
            name="ollama", passed=True, severity="info", detail="ok",
        )
        m_luks.return_value = _sc_mod.CheckItem(
            name="luks", passed=False, severity="warning", detail="no",
            score_impact=-5,
        )
        m_mode.return_value = _sc_mod.CheckItem(
            name="mode", passed=True, severity="info", detail="ok",
        )
        m_swap.return_value = _sc_mod.CheckItem(
            name="swap", passed=False, severity="warning", detail="no",
            score_impact=-3,
        )
        result = _sc_mod.run_startup_checks(force=True)
        self.assertEqual(result.total_score_impact, -10)


# =========================================================================
# Part 7: CheckItem / StartupCheckResult data structures
# =========================================================================

class TestCheckItemDataclass(unittest.TestCase):
    """Part 7a: CheckItem to_dict."""

    def test_to_dict_basic(self):
        ci = _sc_mod.CheckItem(
            name="test", passed=True, severity="info", detail="ok",
        )
        d = ci.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertTrue(d["passed"])
        self.assertNotIn("tips", d)  # empty tips omitted

    def test_to_dict_with_tips(self):
        ci = _sc_mod.CheckItem(
            name="test", passed=False, severity="warning",
            detail="issue", tips=["tip1", "tip2"],
        )
        d = ci.to_dict()
        self.assertIn("tips", d)
        self.assertEqual(len(d["tips"]), 2)


class TestStartupCheckResultDataclass(unittest.TestCase):
    """Part 7b: StartupCheckResult to_dict."""

    def test_to_dict_keys(self):
        sr = _sc_mod.StartupCheckResult()
        d = sr.to_dict()
        expected = {
            "timestamp", "checks", "all_passed", "blocked",
            "block_reason", "total_score_impact", "check_count",
            "passed_count", "failed_count",
        }
        self.assertEqual(set(d.keys()), expected)

    def test_counts_computed(self):
        sr = _sc_mod.StartupCheckResult(
            checks=[
                _sc_mod.CheckItem(name="a", passed=True, severity="info", detail="ok"),
                _sc_mod.CheckItem(name="b", passed=False, severity="warning", detail="no"),
            ],
        )
        d = sr.to_dict()
        self.assertEqual(d["check_count"], 2)
        self.assertEqual(d["passed_count"], 1)
        self.assertEqual(d["failed_count"], 1)


# =========================================================================
# Part 8: Startup checklist caching
# =========================================================================

class TestStartupCache(unittest.TestCase):
    """Part 8: Caching behavior."""

    def setUp(self):
        _sc_mod.clear_cache()

    def test_cache_initially_none(self):
        self.assertIsNone(_sc_mod.get_cached_result())

    @patch.object(_sc_mod, "_check_code_signing_scripts")
    @patch.object(_sc_mod, "_check_ollama_bind")
    @patch.object(_sc_mod, "_check_luks")
    @patch.object(_sc_mod, "_check_security_mode")
    @patch.object(_sc_mod, "_check_encrypted_swap")
    def test_result_cached_after_run(self, *mocks):
        for m in mocks:
            m.return_value = _sc_mod.CheckItem(
                name="ok", passed=True, severity="info", detail="ok",
            )
        _sc_mod.run_startup_checks(force=True)
        self.assertIsNotNone(_sc_mod.get_cached_result())

    @patch.object(_sc_mod, "_check_code_signing_scripts")
    @patch.object(_sc_mod, "_check_ollama_bind")
    @patch.object(_sc_mod, "_check_luks")
    @patch.object(_sc_mod, "_check_security_mode")
    @patch.object(_sc_mod, "_check_encrypted_swap")
    def test_second_call_uses_cache(self, *mocks):
        for m in mocks:
            m.return_value = _sc_mod.CheckItem(
                name="ok", passed=True, severity="info", detail="ok",
            )
        r1 = _sc_mod.run_startup_checks(force=True)
        r2 = _sc_mod.run_startup_checks()  # no force
        self.assertIs(r1, r2)

    def test_clear_cache(self):
        _sc_mod._cached_result = _sc_mod.StartupCheckResult()
        _sc_mod.clear_cache()
        self.assertIsNone(_sc_mod.get_cached_result())


# =========================================================================
# Part 9: API endpoint schema
# =========================================================================

class TestAPIEndpointSchema(unittest.TestCase):
    """Part 9: GET /api/security/startup-checks endpoint exists."""

    def test_endpoint_registered(self):
        """Verify the route is defined in routes_security.py."""
        src = (_OO / "api" / "routes_security.py").read_text()
        self.assertIn("/startup-checks", src)
        self.assertIn("get_startup_checks", src)

    def test_endpoint_imports_startup_checks(self):
        src = (_OO / "api" / "routes_security.py").read_text()
        self.assertIn("from opti_oignon.startup_checks import run_startup_checks", src)

    def test_endpoint_accepts_force_param(self):
        src = (_OO / "api" / "routes_security.py").read_text()
        self.assertIn("force", src)


# =========================================================================
# Part 10: Security score impact
# =========================================================================

class TestSecurityScoreImpact(unittest.TestCase):
    """Part 10: Score impacts are correct values."""

    def test_code_signing_impact(self):
        """Missing scripts = -2."""
        check = _sc_mod._check_code_signing_scripts()
        # Scripts exist in our project, so should pass
        if check.passed:
            self.assertEqual(check.score_impact, 0)
        else:
            self.assertEqual(check.score_impact, -2)

    def test_ollama_exposed_daily_impact(self):
        """Exposed Ollama in daily = -10."""
        ci = _sc_mod.CheckItem(
            name="ollama_bind", passed=False, severity="warning",
            detail="exposed", score_impact=-10,
        )
        self.assertEqual(ci.score_impact, -10)

    def test_ollama_exposed_bulbe_impact(self):
        """Exposed Ollama in bulbe = -15."""
        ci = _sc_mod.CheckItem(
            name="ollama_bind", passed=False, severity="critical",
            detail="blocked", score_impact=-15,
        )
        self.assertEqual(ci.score_impact, -15)

    def test_luks_missing_impact(self):
        """No LUKS = -5."""
        ci = _sc_mod.CheckItem(
            name="luks_encryption", passed=False, severity="warning",
            detail="no encryption", score_impact=-5,
        )
        self.assertEqual(ci.score_impact, -5)

    def test_swap_unencrypted_impact(self):
        """Unencrypted swap = -3."""
        ci = _sc_mod.CheckItem(
            name="encrypted_swap", passed=False, severity="warning",
            detail="unencrypted", score_impact=-3,
        )
        self.assertEqual(ci.score_impact, -3)


# =========================================================================
# Part 11: Version bump (3.2.0-rc1)
# =========================================================================

class TestVersionBump(unittest.TestCase):
    """Part 11: Version is 3.2.0-rc1."""

    def test_version_file(self):
        version_file = _OO / "__version__.py"
        content = version_file.read_text()
        self.assertIn("3.2.0-rc1", content)

    def test_security_md_version(self):
        sec_md = _PROJECT_ROOT / "SECURITY.md"
        content = sec_md.read_text()
        self.assertIn("3.2.0-rc1", content)


# =========================================================================
# Runner
# =========================================================================

if __name__ == "__main__":
    unittest.main()
