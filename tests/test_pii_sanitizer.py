#!/usr/bin/env python3
"""Tests for the PII sanitizer (pii_sanitizer.PIISanitizer).

On a privacy-first local platform this strips technical PII (emails, IPv4/IPv6
addresses, local file paths, internal hostnames) and any user-defined patterns
out of text before it leaves the trust boundary. It is pure regex substitution,
so every branch is testable.

Application order matters and is part of the contract: emails run before
hostnames, so an address with a ``.local`` domain is redacted whole rather than
split. The replacement report deduplicates the matched strings but every
occurrence in the text is still substituted.
"""

from opti_oignon.pii_sanitizer import (
    PIISanitizeConfig,
    PIISanitizer,
)


def _san(**cfg):
    return PIISanitizer(PIISanitizeConfig(**cfg))


def _categories(text, **cfg):
    return {r.category for r in _san(**cfg).sanitize_with_report(text).replacements}


# ===========================================================================
# Per-category redaction
# ===========================================================================

def test_redacts_email():
    assert _san().sanitize("contact user@example.com please") == "contact [EMAIL] please"


def test_redacts_ipv4():
    assert _san().sanitize("host 192.168.1.1 up") == "host [IP] up"


def test_redacts_ipv6():
    out = _san().sanitize("addr 2001:0db8:85a3:0000:0000:8a2e:0370:7334 end")
    assert out == "addr [IP] end"


def test_redacts_file_path():
    assert _san().sanitize("open /home/user/secret.txt now") == "open [PATH] now"


def test_redacts_hostname():
    assert _san().sanitize("ping db.internal here") == "ping [HOST] here"


# ===========================================================================
# Application order: email before hostname
# ===========================================================================

def test_email_with_local_domain_redacted_whole():
    # The .local domain must not be split off and hostname-redacted; the whole
    # address is one [EMAIL].
    assert _san().sanitize("mail admin@server.local done") == "mail [EMAIL] done"


# ===========================================================================
# Clean / empty input
# ===========================================================================

def test_clean_text_unchanged():
    result = _san().sanitize_with_report("nothing to redact here")
    assert result.sanitized == "nothing to redact here"
    assert result.was_modified is False
    assert result.replacements == []


def test_empty_text_passthrough():
    assert _san().sanitize("") == ""


# ===========================================================================
# Multiple categories + report dedup
# ===========================================================================

def test_multiple_categories():
    out = _san().sanitize("email a@b.com ip 10.0.0.1")
    assert out == "email [EMAIL] ip [IP]"
    assert _categories("email a@b.com ip 10.0.0.1") == {"email", "ip"}


def test_report_deduplicates_but_substitutes_all():
    text = "x@y.com and x@y.com"
    result = _san().sanitize_with_report(text)
    assert result.sanitized == "[EMAIL] and [EMAIL]"     # both occurrences replaced
    emails = [r for r in result.replacements if r.category == "email"]
    assert len(emails) == 1                               # but reported once


# ===========================================================================
# Config gating
# ===========================================================================

def test_disabled_is_passthrough():
    assert _san(enabled=False).sanitize("user@example.com 10.0.0.1") == "user@example.com 10.0.0.1"


def test_category_can_be_disabled():
    out = _san(strip_emails=False).sanitize("user@example.com and 10.0.0.1")
    assert "user@example.com" in out      # email left intact
    assert "[IP]" in out                  # ip still redacted


# ===========================================================================
# Report / preview
# ===========================================================================

def test_report_categories_are_labelled():
    cats = _categories("user@a.com 10.0.0.1 /etc/passwd host.lan")
    assert cats == {"email", "ip", "path", "hostname"}


def test_was_modified_flag():
    assert _san().sanitize_with_report("user@a.com").was_modified is True
    assert _san().sanitize_with_report("plain").was_modified is False


def test_preview_structure():
    preview = _san().preview("reach me at user@a.com")
    assert set(preview) == {"original", "sanitized", "items", "was_modified"}
    assert preview["was_modified"] is True
    item = preview["items"][0]
    assert item["category"] == "email"
    assert item["replacement"] == "[EMAIL]"
    assert item["original"] == "user@a.com"


# ===========================================================================
# Custom patterns
# ===========================================================================

def test_custom_pattern_redaction():
    san = _san(custom_patterns=[
        {"pattern": r"SECRET-\d+", "replacement": "[REDACTED]", "label": "secret"},
    ])
    result = san.sanitize_with_report("token SECRET-42 end")
    assert result.sanitized == "token [REDACTED] end"
    assert any(r.category == "secret" for r in result.replacements)


def test_invalid_custom_pattern_is_skipped():
    # An un-compilable pattern is dropped (logged) and never crashes sanitize.
    san = _san(custom_patterns=[{"pattern": "([unclosed", "replacement": "X"}])
    assert san.sanitize("user@a.com") == "[EMAIL]"   # still works


# ===========================================================================
# Config.from_dict
# ===========================================================================

def test_config_from_dict_overrides_and_defaults():
    cfg = PIISanitizeConfig.from_dict({"strip_emails": False})
    assert cfg.strip_emails is False
    assert cfg.strip_ip_addresses is True            # untouched -> default
    assert PIISanitizeConfig.from_dict({}).enabled is True
