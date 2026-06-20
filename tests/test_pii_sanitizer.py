#!/usr/bin/env python3
"""
Tests for PII Sanitizer module (S82).

Covers:
- Email sanitization
- IPv4/IPv6 sanitization
- File path sanitization
- Hostname sanitization
- Custom patterns
- Config toggling
- Preview API
- Edge cases
"""

import sys
import types
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module loading (test isolation without ollama)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Provide opti_oignon.config stub
_config_mod = types.ModuleType("opti_oignon.config")
_config_mod.CONFIG_DIR = _PROJECT_ROOT / "opti_oignon" / "config"

def _load_yaml(p):
    import yaml
    with open(p) as f:
        return yaml.safe_load(f) or {}

_config_mod.load_yaml = _load_yaml
sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))
sys.modules["opti_oignon.config"] = _config_mod

# Load pii_sanitizer module
_pii_path = _PROJECT_ROOT / "opti_oignon" / "pii_sanitizer.py"
_pii_spec = importlib.util.spec_from_file_location(
    "opti_oignon.pii_sanitizer", _pii_path
)
_pii_mod = importlib.util.module_from_spec(_pii_spec)
sys.modules["opti_oignon.pii_sanitizer"] = _pii_mod
_pii_spec.loader.exec_module(_pii_mod)

PIISanitizer = _pii_mod.PIISanitizer
PIISanitizeConfig = _pii_mod.PIISanitizeConfig
PIISanitizeResult = _pii_mod.PIISanitizeResult
PIIReplacement = _pii_mod.PIIReplacement


# ===========================================================================
# Tests
# ===========================================================================


class TestPIISanitizerEmails:
    """Test email address sanitization."""

    def test_simple_email(self):
        s = PIISanitizer()
        result = s.sanitize("contact user@example.com for help")
        assert "[EMAIL]" in result
        assert "user@example.com" not in result

    def test_multiple_emails(self):
        s = PIISanitizer()
        text = "send to alice@test.org and bob@corp.io"
        result = s.sanitize(text)
        assert result.count("[EMAIL]") == 2
        assert "alice@test.org" not in result
        assert "bob@corp.io" not in result

    def test_email_with_plus(self):
        s = PIISanitizer()
        result = s.sanitize("user+tag@gmail.com")
        assert "[EMAIL]" in result

    def test_email_disabled(self):
        config = PIISanitizeConfig(strip_emails=False)
        s = PIISanitizer(config)
        result = s.sanitize("user@example.com")
        assert "user@example.com" in result
        assert "[EMAIL]" not in result


class TestPIISanitizerIPs:
    """Test IP address sanitization."""

    def test_ipv4(self):
        s = PIISanitizer()
        result = s.sanitize("server at 192.168.1.100")
        assert "[IP]" in result
        assert "192.168.1.100" not in result

    def test_ipv4_multiple(self):
        s = PIISanitizer()
        result = s.sanitize("from 10.0.0.1 to 172.16.0.255")
        assert result.count("[IP]") == 2

    def test_ipv4_boundary_values(self):
        s = PIISanitizer()
        assert "[IP]" in s.sanitize("addr 0.0.0.0")
        assert "[IP]" in s.sanitize("addr 255.255.255.255")

    def test_ipv6_full(self):
        s = PIISanitizer()
        result = s.sanitize("addr 2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert "[IP]" in result

    def test_ip_disabled(self):
        config = PIISanitizeConfig(strip_ip_addresses=False)
        s = PIISanitizer(config)
        result = s.sanitize("server 192.168.1.1")
        assert "192.168.1.1" in result


class TestPIISanitizerPaths:
    """Test file path sanitization."""

    def test_absolute_path(self):
        s = PIISanitizer()
        result = s.sanitize("file at /home/leon/project/main.py")
        assert "[PATH]" in result
        assert "/home/leon" not in result

    def test_home_tilde_path(self):
        s = PIISanitizer()
        result = s.sanitize("config in ~/Documents/settings.yaml")
        assert "[PATH]" in result

    def test_relative_path(self):
        s = PIISanitizer()
        result = s.sanitize("see ./src/main.py")
        assert "[PATH]" in result

    def test_path_disabled(self):
        config = PIISanitizeConfig(strip_file_paths=False)
        s = PIISanitizer(config)
        result = s.sanitize("file /home/user/test.py")
        assert "/home/user/test.py" in result


class TestPIISanitizerHostnames:
    """Test internal hostname sanitization."""

    def test_dot_local(self):
        s = PIISanitizer()
        result = s.sanitize("connect to mypc.local")
        assert "[HOST]" in result
        assert "mypc.local" not in result

    def test_dot_internal(self):
        s = PIISanitizer()
        result = s.sanitize("server.internal is down")
        assert "[HOST]" in result

    def test_dot_lan(self):
        s = PIISanitizer()
        result = s.sanitize("printer.lan not responding")
        assert "[HOST]" in result

    def test_hostname_disabled(self):
        config = PIISanitizeConfig(strip_hostnames=False)
        s = PIISanitizer(config)
        result = s.sanitize("mypc.local")
        assert "mypc.local" in result


class TestPIISanitizerCustomPatterns:
    """Test custom regex patterns."""

    def test_custom_pattern(self):
        config = PIISanitizeConfig(
            custom_patterns=[
                {"pattern": r"\bPROJECT-\d+\b", "replacement": "[TICKET]", "label": "ticket"}
            ]
        )
        s = PIISanitizer(config)
        result = s.sanitize("fix PROJECT-1234 urgently")
        assert "[TICKET]" in result
        assert "PROJECT-1234" not in result

    def test_invalid_custom_pattern_ignored(self):
        config = PIISanitizeConfig(
            custom_patterns=[
                {"pattern": "[invalid(", "replacement": "[X]"}
            ]
        )
        # Should not raise, just log warning
        s = PIISanitizer(config)
        result = s.sanitize("test text")
        assert result == "test text"


class TestPIISanitizerConfig:
    """Test configuration behavior."""

    def test_disabled_returns_original(self):
        config = PIISanitizeConfig(enabled=False)
        s = PIISanitizer(config)
        text = "user@test.com at 192.168.1.1"
        assert s.sanitize(text) == text

    def test_empty_string(self):
        s = PIISanitizer()
        assert s.sanitize("") == ""

    def test_no_pii_returns_original(self):
        s = PIISanitizer()
        text = "python dataclass tutorial best practices"
        assert s.sanitize(text) == text

    def test_from_dict_defaults(self):
        config = PIISanitizeConfig.from_dict({})
        assert config.enabled is True
        assert config.strip_emails is True

    def test_from_dict_override(self):
        config = PIISanitizeConfig.from_dict({
            "enabled": True,
            "strip_emails": False,
            "strip_ip_addresses": True,
        })
        assert config.strip_emails is False
        assert config.strip_ip_addresses is True

    def test_update_config(self):
        s = PIISanitizer()
        new_config = PIISanitizeConfig(enabled=False)
        s.update_config(new_config)
        assert s.sanitize("user@test.com") == "user@test.com"


class TestPIISanitizerReport:
    """Test sanitize_with_report and preview."""

    def test_report_structure(self):
        s = PIISanitizer()
        result = s.sanitize_with_report("email user@test.com on 10.0.0.1")
        assert isinstance(result, PIISanitizeResult)
        assert result.was_modified is True
        assert len(result.replacements) == 2
        categories = {r.category for r in result.replacements}
        assert "email" in categories
        assert "ip" in categories

    def test_report_no_pii(self):
        s = PIISanitizer()
        result = s.sanitize_with_report("clean query about python")
        assert result.was_modified is False
        assert len(result.replacements) == 0

    def test_preview_dict(self):
        s = PIISanitizer()
        preview = s.preview("user@test.com on mypc.local")
        assert "original" in preview
        assert "sanitized" in preview
        assert "items" in preview
        assert "was_modified" in preview
        assert preview["was_modified"] is True
        assert len(preview["items"]) >= 2


class TestPIISanitizerCombined:
    """Test multiple PII types in a single query."""

    def test_all_types_combined(self):
        s = PIISanitizer()
        text = "user@test.com at 192.168.1.1 in /home/leon/code on mypc.local"
        result = s.sanitize_with_report(text)
        assert "[EMAIL]" in result.sanitized
        assert "[IP]" in result.sanitized
        assert "[PATH]" in result.sanitized
        assert "[HOST]" in result.sanitized
        assert result.was_modified is True
        assert len(result.replacements) == 4

    def test_email_before_hostname_ordering(self):
        """Emails should be sanitized before hostnames to avoid partial matches."""
        s = PIISanitizer()
        text = "admin@server.local"
        result = s.sanitize(text)
        # The email should be caught as email, not split into hostname
        assert "[EMAIL]" in result
