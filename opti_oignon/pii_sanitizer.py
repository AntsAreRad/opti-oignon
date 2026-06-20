#!/usr/bin/env python3
"""
PII SANITIZER - Opti-Oignon
==============================

Strips personally identifiable information from text before sending
to external services (web search, proxy, etc.).

Configurable via web_search.yaml pii_sanitization section.

Usage:
    from opti_oignon.pii_sanitizer import PIISanitizer, pii_sanitizer

    cleaned = pii_sanitizer.sanitize("Search for user@mail.com logs on 192.168.1.1")
    # -> "Search for [EMAIL] logs on [IP]"

    report = pii_sanitizer.sanitize_with_report("my query with user@mail.com")
    # -> PIISanitizeResult(sanitized="my query with [EMAIL]", replacements=[...])

Author: Leon
"""

__version__ = "1.8.4"
__author__ = "Leon"

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PIISanitizeConfig:
    """
    Configuration for PII sanitization.

    Attributes:
        enabled: Whether sanitization is active
        strip_emails: Replace email addresses with [EMAIL]
        strip_ip_addresses: Replace IPv4/IPv6 addresses with [IP]
        strip_file_paths: Replace local file paths with [PATH]
        strip_hostnames: Replace internal hostnames with [HOST]
        custom_patterns: List of {pattern, replacement} dicts for user-defined rules
    """
    enabled: bool = True
    strip_emails: bool = True
    strip_ip_addresses: bool = True
    strip_file_paths: bool = True
    strip_hostnames: bool = True
    custom_patterns: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "PIISanitizeConfig":
        """Create config from a dictionary (YAML section)."""
        if not data:
            return cls()
        return cls(
            enabled=data.get("enabled", True),
            strip_emails=data.get("strip_emails", True),
            strip_ip_addresses=data.get("strip_ip_addresses", True),
            strip_file_paths=data.get("strip_file_paths", True),
            strip_hostnames=data.get("strip_hostnames", True),
            custom_patterns=data.get("custom_patterns", []),
        )


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class PIIReplacement:
    """A single PII replacement made during sanitization."""
    original: str
    replacement: str
    category: str  # email, ip, path, hostname, custom


@dataclass
class PIISanitizeResult:
    """Result of sanitization with detailed replacement report."""
    original: str
    sanitized: str
    replacements: list[PIIReplacement] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        """Whether any PII was found and replaced."""
        return len(self.replacements) > 0


# =============================================================================
# COMPILED REGEX PATTERNS
# =============================================================================

# Email: standard pattern covering most real-world addresses
_RE_EMAIL = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
)

# IPv4: 4 octets separated by dots (0-255 range validated)
_RE_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# IPv6: simplified pattern covering common formats
# Matches full form and :: abbreviated form
_RE_IPV6 = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
    r"|\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b"
    r"|\b::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}\b"
    r"|\b(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}\b"
)

# File paths: Unix-style absolute paths and common relative paths
# Matches /home/user/file.py, ~/Documents/report.pdf, ./src/main.py
_RE_FILE_PATH = re.compile(
    r"(?:~|\.\.?)?/(?:[a-zA-Z0-9_.\-]+/)*[a-zA-Z0-9_.\-]+"
)

# Internal hostnames: machine.local, server.internal, hostname.lan, etc.
_RE_HOSTNAME = re.compile(
    r"\b[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?"
    r"\.(?:local|internal|lan|localdomain|home|corp|intranet)\b",
    re.IGNORECASE,
)


# =============================================================================
# MAIN CLASS
# =============================================================================

class PIISanitizer:
    """
    Strips PII from text based on configurable rules.

    Each rule category can be independently enabled/disabled.
    Custom regex patterns can be added via configuration.

    Usage:
        sanitizer = PIISanitizer()
        clean_text = sanitizer.sanitize("email user@test.com on 10.0.0.1")
        # -> "email [EMAIL] on [IP]"
    """

    def __init__(self, config: PIISanitizeConfig | None = None):
        """
        Initialize the PII sanitizer.

        Args:
            config: Optional configuration. Uses defaults if None.
        """
        self.config = config or PIISanitizeConfig()
        self._custom_compiled: list[tuple[re.Pattern, str, str]] = []
        self._compile_custom_patterns()

    def _compile_custom_patterns(self) -> None:
        """Pre-compile custom regex patterns from config."""
        self._custom_compiled.clear()
        for entry in self.config.custom_patterns:
            pattern_str = entry.get("pattern", "")
            replacement = entry.get("replacement", "[REDACTED]")
            if not pattern_str:
                continue
            try:
                compiled = re.compile(pattern_str)
                label = entry.get("label", "custom")
                self._custom_compiled.append((compiled, replacement, label))
            except re.error as e:
                logger.warning(f"Invalid custom PII pattern '{pattern_str}': {e}")

    def update_config(self, config: PIISanitizeConfig) -> None:
        """Update configuration and recompile custom patterns."""
        self.config = config
        self._compile_custom_patterns()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def sanitize(self, text: str) -> str:
        """
        Sanitize text by replacing PII with placeholders.

        Args:
            text: Input text to sanitize

        Returns:
            Sanitized text with PII replaced by category placeholders
        """
        if not self.config.enabled or not text:
            return text

        result = self.sanitize_with_report(text)
        return result.sanitized

    def sanitize_with_report(self, text: str) -> PIISanitizeResult:
        """
        Sanitize text and return a detailed report of all replacements.

        Args:
            text: Input text to sanitize

        Returns:
            PIISanitizeResult with sanitized text and replacement details
        """
        if not text:
            return PIISanitizeResult(original=text, sanitized=text)

        if not self.config.enabled:
            return PIISanitizeResult(original=text, sanitized=text)

        replacements: list[PIIReplacement] = []
        sanitized = text

        # Order matters: emails before hostnames (emails contain domain-like parts)
        if self.config.strip_emails:
            sanitized, new_replacements = self._apply_pattern(
                sanitized, _RE_EMAIL, "[EMAIL]", "email"
            )
            replacements.extend(new_replacements)

        if self.config.strip_ip_addresses:
            sanitized, new_replacements = self._apply_pattern(
                sanitized, _RE_IPV4, "[IP]", "ip"
            )
            replacements.extend(new_replacements)
            sanitized, new_replacements = self._apply_pattern(
                sanitized, _RE_IPV6, "[IP]", "ip"
            )
            replacements.extend(new_replacements)

        if self.config.strip_file_paths:
            sanitized, new_replacements = self._apply_pattern(
                sanitized, _RE_FILE_PATH, "[PATH]", "path"
            )
            replacements.extend(new_replacements)

        if self.config.strip_hostnames:
            sanitized, new_replacements = self._apply_pattern(
                sanitized, _RE_HOSTNAME, "[HOST]", "hostname"
            )
            replacements.extend(new_replacements)

        # Custom patterns
        for compiled, replacement, label in self._custom_compiled:
            sanitized, new_replacements = self._apply_pattern(
                sanitized, compiled, replacement, label
            )
            replacements.extend(new_replacements)

        return PIISanitizeResult(
            original=text,
            sanitized=sanitized,
            replacements=replacements,
        )

    def preview(self, text: str) -> dict:
        """
        Preview what would be sanitized without modifying.

        Returns a dict suitable for UI display with original text,
        sanitized text, and list of found PII items.

        Args:
            text: Text to preview sanitization for

        Returns:
            Dict with keys: original, sanitized, items, was_modified
        """
        result = self.sanitize_with_report(text)
        return {
            "original": result.original,
            "sanitized": result.sanitized,
            "items": [
                {
                    "original": r.original,
                    "replacement": r.replacement,
                    "category": r.category,
                }
                for r in result.replacements
            ],
            "was_modified": result.was_modified,
        }

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    @staticmethod
    def _apply_pattern(
        text: str,
        pattern: re.Pattern,
        replacement: str,
        category: str,
    ) -> tuple[str, list[PIIReplacement]]:
        """
        Apply a regex pattern and collect replacements.

        Returns:
            Tuple of (modified text, list of replacements made)
        """
        replacements = []
        matches = list(pattern.finditer(text))

        if not matches:
            return text, replacements

        # Deduplicate matched strings for the report
        seen = set()
        for match in matches:
            original = match.group()
            if original not in seen:
                seen.add(original)
                replacements.append(PIIReplacement(
                    original=original,
                    replacement=replacement,
                    category=category,
                ))

        sanitized = pattern.sub(replacement, text)
        return sanitized, replacements


# =============================================================================
# CONFIG LOADING FROM YAML
# =============================================================================

def load_pii_config_from_yaml() -> PIISanitizeConfig:
    """
    Load PII sanitization config from web_search.yaml.

    Returns:
        PIISanitizeConfig loaded from YAML, or defaults if unavailable
    """
    try:
        from .config import load_yaml, CONFIG_DIR
        data = load_yaml(CONFIG_DIR / "web_search.yaml")
        pii_section = data.get("pii_sanitization", {})
        return PIISanitizeConfig.from_dict(pii_section)
    except Exception as e:
        logger.debug(f"Could not load PII config from YAML: {e}")
        return PIISanitizeConfig()


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

pii_sanitizer = PIISanitizer(load_pii_config_from_yaml())
