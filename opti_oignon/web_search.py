#!/usr/bin/env python3
"""
WEB SEARCH MODULE - Opti-Oignon
================================

Web search through the ddgs package, with caching, rate limiting,
token-budgeted formatting, SOCKS5/Tor proxy support, and PII sanitization.
With its default backend, ddgs sends the query to Wikipedia and to one or
more engines it picks at random among DuckDuckGo, Bing, Google, Brave,
Mojeek, Yahoo, Yandex and Mullvad; every result is labelled "duckduckgo"
all the same.

Every request this module makes -- a search, the proxy health check and its
Tor exit lookup -- opens one gate first, ahead of the cache. The gate refuses
while the search kill switch is engaged or cannot be read, and in any
security mode but exactly "daily", a mode that cannot be read included. A
refusal raises ``WebSearchRefused``, which names it, and it is never retried.
The search class is bound privately and constructed in one place,
``WebSearcher._ddgs``; nothing else in the package names it. The kill
switch's domain allowlist and its injection circuit breaker act on every real
search, cached results included.

``fetch_page`` fetches one web page for the knowledge base behind the same
gate, asked again before each redirect and after every read of the body,
and refused by name as ``PageFetchRefused``. It reaches only a public
address: user information, a scheme other than http or https, a port other
than 80 and 443 unless the configuration names it, and a host that resolves
to a loopback, private, link-local, site-local, unspecified, multicast,
reserved or shared address are refused, an IPv4 address inside IPv6 read as
the address it carries; so are an address this machine holds and one on a
network its routing tables reach without a gateway. The host is resolved
once per request and the connection goes only to addresses that were
checked; at most three redirects are followed, each checked as the first;
the body and the time are capped.

This module provides the search layer used by the ReAct integration (Session 6)
to inject web search results into LLM conversations.

Quick usage:
    from opti_oignon.web_search import web_searcher

    results = web_searcher.search("python dataclass tutorial")
    formatted = web_searcher.search_and_format("latest pandas release", token_budget=1500)

CLI:
    python -m opti_oignon.web_search "python dataclass tutorial"
    python -m opti_oignon.web_search --formatted "latest pandas release"

Author: Leon
"""

__version__ = "1.8.4"
__author__ = "Leon"

import codecs
import errno
import hashlib
import http.client
import ipaddress
import logging
import socket
import ssl
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

# =============================================================================
# CONDITIONAL IMPORTS
# =============================================================================

try:
    # New package name (ddgs >= 7.0)
    from ddgs import DDGS as _DDGS
    DDGS_AVAILABLE = True
    DuckDuckGoSearchException = Exception
    RatelimitException = Exception
    TimeoutException = Exception
except ImportError:
    try:
        # Legacy package name (duckduckgo_search < 7.0)
        from duckduckgo_search import DDGS as _DDGS  # type: ignore[no-redef]
        from duckduckgo_search.exceptions import (
            DuckDuckGoSearchException,
            RatelimitException,
            TimeoutException,
        )
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False
        _DDGS = None
        DuckDuckGoSearchException = Exception
        RatelimitException = Exception
        TimeoutException = Exception

try:
    from .pii_sanitizer import PIISanitizeConfig, PIISanitizer
    from .pii_sanitizer import pii_sanitizer as _default_pii
    PII_AVAILABLE = True
except ImportError:
    PII_AVAILABLE = False
    PIISanitizer = None
    PIISanitizeConfig = None
    _default_pii = None

logger = logging.getLogger(__name__)


# =============================================================================
# THE REQUEST GATE
# =============================================================================

_REFUSALS = {
    "kill_switch": "Web search is refused: the search kill switch is engaged.",
    "mode": "Web search is refused outside Daily mode.",
    "unreadable": "Web search is refused: the kill switch cannot be read.",
}


class WebSearchRefused(RuntimeError):
    """A web request refused by the gate, carrying the refusal by name."""

    def __init__(self, refusal: str):
        super().__init__(_REFUSALS[refusal])
        self.refusal = refusal


def search_refusal() -> str | None:
    """Why no web request may leave the process now, or None.

    ``kill_switch`` while the switch is engaged; ``unreadable`` when it cannot
    be read, its module absent included; ``mode`` in any mode but exactly
    ``"daily"``, and when the mode cannot be read. Both are asked at every
    request: the switch reads its record again, and the mode answers from
    the process's cache, which a mode change made in this process
    refreshes.
    """
    try:
        from opti_oignon.search_killswitch import search_killswitch
        if search_killswitch.is_killed():
            return "kill_switch"
    except Exception:
        return "unreadable"
    try:
        from opti_oignon.security_mode import get_current_mode
        mode = get_current_mode()
    except Exception:
        return "mode"
    return None if isinstance(mode, str) and mode == "daily" else "mode"


# =============================================================================
# SEARCH RESULT DATACLASS
# =============================================================================

@dataclass
class SearchResult:
    """
    A single web search result.

    Attributes:
        title: Page title
        snippet: Text excerpt / description
        url: Full URL
        source: Search engine identifier (for future multi-engine support)
    """
    title: str
    snippet: str
    url: str
    source: str = "duckduckgo"


# =============================================================================
# Search Result Sanitizer (Prompt Injection Defense)
# =============================================================================

# The prompt-injection patterns and the invisible-char / HTML-tag /
# hidden-CSS / base64-instruction strippers are defined once in rag_sanitizer
# (the single source of truth) and imported here, so the RAG sanitizer and the
# search-result sanitizer cannot drift apart. The injection list is
# (name, pattern, weight); the weight is unused on the search side.
from opti_oignon.rag_sanitizer import (
    _BASE64_INSTRUCTION,
    _HIDDEN_CSS,
    _HTML_TAGS,
    _INJECTION_PATTERNS,
    _INVISIBLE_CHARS,
)


def _load_search_safety_config() -> dict:
    """Load search safety config from security.yaml."""
    import yaml
    cfg_path = Path(__file__).parent / "config" / "security.yaml"
    try:
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("search_safety", {})
    except Exception:
        pass
    return {}


class SearchResultSanitizer:
    """Sanitizes search results to defend against indirect prompt injection.

    Performs multiple layers of defense:
    1. Strip HTML tags from titles and snippets
    2. Remove zero-width / invisible Unicode characters
    3. Detect and flag common injection patterns
    4. Limit snippet length to prevent context flooding
    5. Log detected injection attempts for audit
    """

    def __init__(self, config: dict | None = None):
        # An empty configuration is a configuration: only None reads the file.
        self._config = config if config is not None else _load_search_safety_config()
        self._enabled = self._config.get("enabled", True)
        self._max_snippet_length = self._config.get("max_snippet_length", 500)
        self._max_title_length = self._config.get("max_title_length", 200)
        self._strip_html = self._config.get("strip_html", True)
        self._strip_invisible = self._config.get("strip_invisible_chars", True)
        self._detect_injections = self._config.get("detect_injections", True)
        self._audit_log: list[dict] = []

    def sanitize_result(self, result: SearchResult) -> SearchResult:
        """Sanitize a single search result.

        Returns a new SearchResult with cleaned title and snippet.
        Detected injection patterns are logged and stripped.
        """
        if not self._enabled:
            return result

        title = self._clean_text(result.title, self._max_title_length, "title")
        snippet = self._clean_text(result.snippet, self._max_snippet_length, "snippet")

        return SearchResult(
            title=title,
            snippet=snippet,
            url=result.url,
            source=result.source,
        )

    def sanitize_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Sanitize a list of search results."""
        return [self.sanitize_result(r) for r in results]

    def get_audit_log(self) -> list[dict]:
        """Get the audit log of detected injection attempts."""
        return list(self._audit_log)

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._audit_log.clear()

    def absorb_audit_log(self, entries: list[dict]) -> None:
        """Append another sanitizer's detections, as the security events route reads them."""
        self._audit_log.extend(dict(entry) for entry in entries)

    def _clean_text(self, text: str, max_length: int, field_name: str) -> str:
        """Apply all sanitization layers to a text field.

        Defense-in-depth approach:
        1. Unicode NFKC normalization (collapse homoglyphs, fullwidth chars)
        2. Strip HTML tags
        3. Remove invisible/bidi/zero-width Unicode
        4. Remove base64-encoded data URIs (encoding bypass vector)
        5. Remove hidden CSS content markers
        6. Detect and neutralize injection patterns
        7. Truncate to max length
        8. Normalize whitespace
        """
        if not text:
            return text

        import unicodedata

        original = text

        # 0. Unicode NFKC normalization: collapses fullwidth Latin chars,
        # compatibility decomposition + canonical composition.
        # E.g. U+FF29 (fullwidth 'I') -> 'I', ligatures decomposed.
        text = unicodedata.normalize("NFKC", text)

        # 1. Strip HTML tags
        if self._strip_html:
            text = _HTML_TAGS.sub("", text)

        # 2. Remove invisible/zero-width/bidi characters
        if self._strip_invisible:
            text = _INVISIBLE_CHARS.sub("", text)

        # 3. Remove base64-encoded data URIs (encoding bypass vector)
        text = _BASE64_INSTRUCTION.sub("[encoded-content-removed]", text)

        # 4. Remove hidden CSS content markers
        text = _HIDDEN_CSS.sub("[hidden-content-removed]", text)

        # 5. Detect injection patterns
        if self._detect_injections:
            for pattern_name, pattern, _weight in _INJECTION_PATTERNS:
                match = pattern.search(text)
                if match:
                    self._log_injection(pattern_name, match.group(), field_name, original[:200])
                    # Replace the matched injection with a neutralization marker
                    text = pattern.sub("[content-filtered]", text)

        # 6. Truncate to max length
        if len(text) > max_length:
            text = text[:max_length].rsplit(" ", 1)[0] + "..."

        # 7. Normalize whitespace (collapse multiple spaces, strip)
        text = " ".join(text.split())

        return text

    def _log_injection(self, pattern: str, matched: str, field: str, context: str) -> None:
        """Log a detected injection attempt."""
        entry = {
            "pattern": pattern,
            "matched": matched[:100],
            "field": field,
            "context": context[:200],
        }
        self._audit_log.append(entry)
        logger.warning(
            "PROMPT INJECTION DETECTED: pattern=%s matched=%r in %s",
            pattern, matched[:60], field,
        )


# Module-level singleton
_search_sanitizer: SearchResultSanitizer | None = None


def get_search_sanitizer() -> SearchResultSanitizer:
    """Get or create the singleton SearchResultSanitizer."""
    global _search_sanitizer
    if _search_sanitizer is None:
        _search_sanitizer = SearchResultSanitizer()
    return _search_sanitizer


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WebSearchConfig:
    """
    Configuration for the web searcher.

    Attributes:
        cache_ttl: Cache time-to-live in seconds
        rate_limit_interval: Minimum seconds between requests
        default_max_results: Default number of results per search
        default_token_budget: Default token budget for formatted output
        region: DuckDuckGo region code (None = auto)
        safesearch: DuckDuckGo safesearch level
        timeout: Request timeout in seconds (direct connection)
        proxy: SOCKS5 proxy URL (None = direct, e.g. "socks5h://localhost:9050")
        proxy_timeout: Timeout when using proxy (typically longer for Tor)
        max_retries: Max retry attempts on transient failures
        retry_backoff: List of backoff delays in seconds per retry attempt
        pii_sanitization_enabled: Whether to sanitize queries before sending
    """
    cache_ttl: int = 300
    rate_limit_interval: float = 1.0
    default_max_results: int = 5
    default_token_budget: int = 1500
    region: str | None = None
    safesearch: str = "moderate"
    timeout: int = 10
    proxy: str | None = None
    proxy_timeout: int = 15
    max_retries: int = 3
    retry_backoff: list[int] = field(default_factory=lambda: [2, 5, 10])
    pii_sanitization_enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "WebSearchConfig":
        """Create config from a dictionary (YAML data)."""
        if not data:
            return cls()
        pii_section = data.get("pii_sanitization", {})
        return cls(
            cache_ttl=data.get("cache_ttl", 300),
            rate_limit_interval=data.get("rate_limit_interval", 1.0),
            default_max_results=data.get("default_max_results", 5),
            default_token_budget=data.get("default_token_budget", 1500),
            region=data.get("region"),
            safesearch=data.get("safesearch", "moderate"),
            timeout=data.get("timeout", 10),
            proxy=data.get("proxy"),
            proxy_timeout=data.get("proxy_timeout", 15),
            max_retries=data.get("max_retries", 3),
            retry_backoff=data.get("retry_backoff", [2, 5, 10]),
            pii_sanitization_enabled=pii_section.get("enabled", True) if pii_section else True,
        )


def _load_config_from_yaml() -> WebSearchConfig:
    """Load web search config from web_search.yaml."""
    try:
        from .config import CONFIG_DIR, load_yaml
        data = load_yaml(CONFIG_DIR / "web_search.yaml")
        return WebSearchConfig.from_dict(data)
    except Exception as e:
        logger.debug(f"Could not load web_search.yaml: {e}, using defaults")
        return WebSearchConfig()


# =============================================================================
# PROXY STATUS
# =============================================================================

@dataclass
class ProxyStatus:
    """Result of a proxy health check."""
    configured: bool = False
    proxy_url: str | None = None
    reachable: bool = False
    latency_ms: float | None = None
    exit_ip: str | None = None
    error: str | None = None


# =============================================================================
# MAIN CLASS
# =============================================================================

class WebSearcher:
    """
    Web search wrapper with caching, rate limiting, token-budgeted formatting,
    SOCKS5/Tor proxy support, PII sanitization, and retry with backoff.

    Usage:
        searcher = WebSearcher()
        results = searcher.search("query")
        formatted = searcher.search_and_format("query", token_budget=1500)
    """

    def __init__(
        self,
        config: WebSearchConfig | None = None,
        pii_sanitizer: "PIISanitizer | None" = None,
    ):
        """
        Initialize the web searcher.

        Args:
            config: Optional configuration override
            pii_sanitizer: Optional PII sanitizer instance (uses module singleton if None)
        """
        self.config = config or WebSearchConfig()

        # PII sanitizer: use provided, or module singleton, or None
        if pii_sanitizer is not None:
            self._pii = pii_sanitizer
        elif PII_AVAILABLE and _default_pii is not None:
            self._pii = _default_pii
        else:
            self._pii = None

        # In-memory cache: {query_hash: (timestamp, List[SearchResult])}
        self._cache: dict[str, tuple[float, list[SearchResult]]] = {}

        # Timestamp of last request (rate limiting)
        self._last_request_time: float = 0.0

        # Statistics
        self._stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "retries": 0,
            "pii_sanitizations": 0,
            "proxy_searches": 0,
        }

        if not DDGS_AVAILABLE:
            logger.warning(
                "Search package not installed. "
                "Install with: pip install ddgs"
            )

    # -------------------------------------------------------------------------
    # The gate
    # -------------------------------------------------------------------------

    def _require_open(self) -> None:
        """Raise ``WebSearchRefused`` unless a web request may leave now."""
        refusal = search_refusal()
        if refusal is not None:
            raise WebSearchRefused(refusal)

    def _ddgs(self, **kwargs):
        """The one construction of the search class, behind the gate."""
        self._require_open()
        return _DDGS(**kwargs)

    def _allowed(self, results: list[SearchResult]) -> list[SearchResult]:
        """The results the kill switch's domain allowlist lets through.

        Applied on the way out, so an allowlist change reaches cached results
        too. When the allowlist cannot be read, nothing passes.
        """
        try:
            from opti_oignon.search_killswitch import search_killswitch
            kept = list(search_killswitch.filter_results(results))
        except Exception as exc:
            logger.warning(
                "Search results withheld: the domain allowlist cannot be read (%s).",
                exc.__class__.__name__,
            )
            return []
        withheld = len(results) - len(kept)
        if withheld:
            logger.info(
                "Domain allowlist withheld %d of %d search result(s).",
                withheld, len(results),
            )
        return kept

    @staticmethod
    def _record_injection(sanitizer: "SearchResultSanitizer") -> None:
        """Feed the circuit breaker once for a search whose results carried an injection.

        Pattern names only: never the query, a snippet or a URL.
        """
        log = sanitizer.get_audit_log()
        if not log:
            return
        patterns = sorted({str(entry.get("pattern", "")) for entry in log})
        try:
            from opti_oignon.search_killswitch import search_killswitch
            search_killswitch.record_injection(details="patterns: " + ", ".join(patterns))
        except Exception as exc:
            logger.warning(
                "Search injection could not be recorded (%s).", exc.__class__.__name__,
            )

    # -------------------------------------------------------------------------
    # Proxy configuration
    # -------------------------------------------------------------------------

    @property
    def proxy_configured(self) -> bool:
        """Whether a proxy is configured."""
        return self.config.proxy is not None and self.config.proxy != ""

    @property
    def effective_timeout(self) -> int:
        """Return proxy_timeout if proxy is configured, else timeout."""
        if self.proxy_configured:
            return self.config.proxy_timeout
        return self.config.timeout

    def set_proxy(self, proxy_url: str | None) -> None:
        """
        Update the proxy configuration at runtime.

        Args:
            proxy_url: SOCKS5 proxy URL or None to disable
        """
        self.config.proxy = proxy_url
        logger.info(f"Proxy updated: {proxy_url or 'disabled (direct)'}")

    def check_proxy_status(self) -> ProxyStatus:
        """
        Check if the configured proxy is reachable.

        Tests connectivity by making a lightweight request through the proxy.
        For Tor, attempts to retrieve the exit node IP.

        Returns:
            ProxyStatus with connectivity details
        """
        if not self.proxy_configured:
            return ProxyStatus(configured=False)

        status = ProxyStatus(
            configured=True,
            proxy_url=self.config.proxy,
        )

        try:
            self._require_open()
            start = time.monotonic()
            if not DDGS_AVAILABLE:
                status.error = "duckduckgo-search not installed"
                return status

            ddgs = self._ddgs(
                proxy=self.config.proxy,
                timeout=self.config.proxy_timeout,
            )
            # Lightweight search to verify connectivity
            results = ddgs.text("test", max_results=1)  # noqa: F841
            elapsed = (time.monotonic() - start) * 1000

            status.reachable = True
            status.latency_ms = round(elapsed, 1)

            # Try to get exit IP for Tor proxies
            if "9050" in (self.config.proxy or ""):
                status.exit_ip = self._get_tor_exit_ip()

        except Exception as e:
            status.reachable = False
            status.error = str(e)
            logger.warning(f"Proxy health check failed: {e}")

        return status

    def _get_tor_exit_ip(self) -> str | None:
        """Attempt to retrieve Tor exit node IP via a check service."""
        self._require_open()
        try:
            import json
            import urllib.request

            proxy_handler = urllib.request.ProxyHandler({
                "https": self.config.proxy,
                "http": self.config.proxy,
            })
            opener = urllib.request.build_opener(proxy_handler)
            response = opener.open("https://check.torproject.org/api/ip", timeout=10)
            data = json.loads(response.read().decode())
            return data.get("IP")
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # PII sanitization
    # -------------------------------------------------------------------------

    def sanitize_query(self, query: str) -> tuple[str, bool]:
        """
        Sanitize a search query by stripping PII if enabled.

        Args:
            query: Raw search query

        Returns:
            Tuple of (sanitized_query, was_modified)
        """
        if not self.config.pii_sanitization_enabled:
            return query, False

        if self._pii is None:
            return query, False

        result = self._pii.sanitize_with_report(query)
        if result.was_modified:
            self._stats["pii_sanitizations"] += 1
            logger.info(
                f"PII sanitized: {len(result.replacements)} item(s) removed from query"
            )

        return result.sanitized, result.was_modified

    def preview_sanitization(self, query: str) -> dict:
        """
        Preview PII sanitization for a query (for UI display).

        Args:
            query: Raw query to preview

        Returns:
            Dict with original, sanitized, items, was_modified
        """
        if self._pii is None:
            return {
                "original": query,
                "sanitized": query,
                "items": [],
                "was_modified": False,
            }
        return self._pii.preview(query)

    # -------------------------------------------------------------------------
    # Main search
    # -------------------------------------------------------------------------

    def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[SearchResult]:
        """
        Search the web through the ddgs package, which with its default backend
        sends the query to Wikipedia and to one or more engines it picks at
        random, with optional proxy and PII sanitization.

        Args:
            query: Search query string
            max_results: Maximum number of results (default from config)

        Returns:
            List of SearchResult objects the domain allowlist lets through
            (may be empty on error)

        Raises:
            WebSearchRefused: When the gate refuses, before the cache is read
            RuntimeError: If ddgs/duckduckgo-search is not installed
        """
        self._require_open()
        if not DDGS_AVAILABLE:
            raise RuntimeError(
                "Search package not installed. "
                "Install with: pip install ddgs"
            )

        # Validate query
        query = (query or "").strip()
        if not query:
            logger.warning("Empty search query, returning empty list")
            return []

        max_results = max_results or self.config.default_max_results
        self._stats["total_searches"] += 1

        # PII sanitization
        sanitized_query, _ = self.sanitize_query(query)

        # Check cache (using sanitized query for key)
        cache_key = self._make_cache_key(sanitized_query, max_results)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            self._stats["cache_hits"] += 1
            logger.debug(f"Cache hit for: {sanitized_query!r}")
            return self._allowed(cached)

        self._stats["cache_misses"] += 1

        # Rate limiting
        self._enforce_rate_limit()

        # Execute search with retry logic
        results = self._search_with_retry(sanitized_query, max_results)

        if results is not None:
            # The circuit breaker may just have tripped on these results.
            self._require_open()
            self._put_in_cache(cache_key, results)
            logger.info(
                f"Search complete: {sanitized_query!r} -> {len(results)} result(s)"
                f"{' (via proxy)' if self.proxy_configured else ''}"
            )
            return self._allowed(results)

        return []

    def _search_with_retry(
        self,
        query: str,
        max_results: int,
    ) -> list[SearchResult] | None:
        """
        Execute a DDG search with configurable retry and backoff.

        Returns:
            List of results on success, None on exhausted retries
        """
        max_attempts = 1 + self.config.max_retries
        backoff = self.config.retry_backoff

        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                ddgs_kwargs = {
                    "timeout": self.effective_timeout,
                }
                if self.proxy_configured:
                    ddgs_kwargs["proxy"] = self.config.proxy
                    self._stats["proxy_searches"] += 1

                ddgs = self._ddgs(**ddgs_kwargs)
                raw_results = ddgs.text(
                    query,
                    region=self.config.region,
                    safesearch=self.config.safesearch,
                    max_results=max_results,
                )

                # Convert to SearchResult
                results = []
                for item in (raw_results or []):
                    results.append(SearchResult(
                        title=item.get("title", "").strip(),
                        snippet=item.get("body", "").strip(),
                        url=item.get("href", "").strip(),
                        source="duckduckgo",
                    ))

                # Sanitize results against prompt injection, on an instance of
                # this search's own so the breaker counts this search exactly;
                # its detections then join the shared log the security events
                # route reads.
                shared = get_search_sanitizer()
                sanitizer = SearchResultSanitizer(config=shared._config)
                results = sanitizer.sanitize_results(results)
                shared.absorb_audit_log(sanitizer.get_audit_log())
                self._record_injection(sanitizer)

                return results

            except WebSearchRefused:
                raise
            except RatelimitException as e:
                last_error = e
                logger.warning(
                    f"Rate limit on attempt {attempt + 1}/{max_attempts}: {e}"
                )
            except TimeoutException as e:
                last_error = e
                logger.warning(
                    f"Timeout on attempt {attempt + 1}/{max_attempts}: {e}"
                )
            except DuckDuckGoSearchException as e:
                last_error = e
                logger.warning(
                    f"DDG error on attempt {attempt + 1}/{max_attempts}: {e}"
                )
            except Exception as e:
                last_error = e
                logger.error(
                    f"Unexpected error on attempt {attempt + 1}/{max_attempts}: {e}"
                )

            # Backoff before retry (if not last attempt)
            if attempt < max_attempts - 1:
                delay = backoff[min(attempt, len(backoff) - 1)]
                logger.info(f"Retrying in {delay}s...")
                self._stats["retries"] += 1
                time.sleep(delay)

        # All retries exhausted
        self._stats["errors"] += 1
        logger.error(
            f"Search failed after {max_attempts} attempts for {query!r}: {last_error}"
        )
        return None

    # -------------------------------------------------------------------------
    # Formatted search with token budget
    # -------------------------------------------------------------------------

    def search_and_format(
        self,
        query: str,
        max_results: int | None = None,
        token_budget: int | None = None,
    ) -> str:
        """
        Search and return results formatted as text within a token budget.

        The output is a numbered list suitable for injection into LLM context.
        Results are truncated if they exceed the token budget.

        Args:
            query: Search query string
            max_results: Maximum number of results (default 3 for formatted output)
            token_budget: Maximum approximate tokens in output (default from config)

        Returns:
            Formatted string with search results, or empty string on error/no results
        """
        max_results = max_results or 3
        token_budget = token_budget or self.config.default_token_budget

        results = self.search(query, max_results=max_results)

        if not results:
            return ""

        return self._format_results(results, token_budget)

    # -------------------------------------------------------------------------
    # Cache
    # -------------------------------------------------------------------------

    def clear_cache(self) -> int:
        """
        Clear the entire search cache.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared: {count} entry(ies) removed")
        return count

    def get_cache_stats(self) -> dict:
        """
        Get cache and search statistics for debugging.

        Returns:
            Dictionary with stats (total_searches, cache_hits, cache_misses,
            errors, retries, pii_sanitizations, proxy_searches,
            cache_size, cache_entries)
        """
        self._evict_expired()

        return {
            **self._stats,
            "cache_size": len(self._cache),
            "cache_entries": list(self._cache.keys()),
            "ddgs_available": DDGS_AVAILABLE,
            "proxy_configured": self.proxy_configured,
            "pii_available": PII_AVAILABLE,
        }

    # -------------------------------------------------------------------------
    # Internal - Cache
    # -------------------------------------------------------------------------

    def _make_cache_key(self, query: str, max_results: int) -> str:
        """
        Generate a cache key from query and parameters.

        Normalizes query (lowercase, strip) before hashing for consistency.
        """
        normalized = query.lower().strip()
        raw = f"{normalized}|{max_results}"
        return hashlib.md5(raw.encode("utf-8"), usedforsecurity=False).hexdigest()

    def _get_from_cache(self, key: str) -> list[SearchResult] | None:
        """Return cached results if present and not expired, else None."""
        if key not in self._cache:
            return None

        timestamp, results = self._cache[key]
        age = time.time() - timestamp

        if age > self.config.cache_ttl:
            del self._cache[key]
            logger.debug(f"Cache expired for key {key[:8]}... (age={age:.0f}s)")
            return None

        return results

    def _put_in_cache(self, key: str, results: list[SearchResult]) -> None:
        """Store results in cache with current timestamp."""
        self._cache[key] = (time.time(), results)

    def _evict_expired(self) -> int:
        """Remove all expired cache entries. Returns count of evicted entries."""
        now = time.time()
        expired_keys = [
            k for k, (ts, _) in self._cache.items()
            if (now - ts) > self.config.cache_ttl
        ]
        for k in expired_keys:
            del self._cache[k]
        return len(expired_keys)

    # -------------------------------------------------------------------------
    # Internal - Rate Limiting
    # -------------------------------------------------------------------------

    def _enforce_rate_limit(self) -> None:
        """Enforce minimum interval between requests. Blocks if needed."""
        now = time.time()
        elapsed = now - self._last_request_time
        wait_time = self.config.rate_limit_interval - elapsed

        if wait_time > 0:
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
            time.sleep(wait_time)

        self._last_request_time = time.time()

    # -------------------------------------------------------------------------
    # Internal - Formatting
    # -------------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Rough token estimation (4 chars per token heuristic).

        This is intentionally simple - matches the pattern used in
        conversation.py and executor.py for consistency.
        """
        return max(1, len(text) // 4)

    def _format_results(
        self,
        results: list[SearchResult],
        token_budget: int,
    ) -> str:
        """
        Format search results as numbered text within a token budget.

        Format:
            [1] Title of result
            Snippet text truncated to fit budget...
            URL: https://example.com

            [2] Title of second result
            ...

        Results are added one by one until the budget is exhausted.
        Individual snippets are truncated if a single result exceeds
        remaining budget.
        """
        formatted_parts = []
        tokens_used = 0

        for i, result in enumerate(results, 1):
            entry = self._format_single_result(i, result)
            entry_tokens = self._estimate_tokens(entry)

            if tokens_used + entry_tokens <= token_budget:
                formatted_parts.append(entry)
                tokens_used += entry_tokens
            else:
                remaining_budget = token_budget - tokens_used
                if remaining_budget < 30:
                    break

                truncated = self._format_single_result_truncated(
                    i, result, remaining_budget
                )
                if truncated:
                    formatted_parts.append(truncated)
                break

        return "\n".join(formatted_parts)

    @staticmethod
    def _format_single_result(index: int, result: SearchResult) -> str:
        """Format a single search result as text."""
        lines = [
            f"[{index}] {result.title}",
            result.snippet,
            f"URL: {result.url}",
            "",
        ]
        return "\n".join(lines)

    def _format_single_result_truncated(
        self,
        index: int,
        result: SearchResult,
        token_budget: int,
    ) -> str | None:
        """
        Format a single result with truncated snippet to fit budget.

        Returns None if even the minimal version doesn't fit.
        """
        header = f"[{index}] {result.title}"
        url_line = f"URL: {result.url}"
        overhead = self._estimate_tokens(header + "\n" + url_line + "\n\n")

        snippet_budget = token_budget - overhead
        if snippet_budget <= 0:
            return None

        snippet = result.snippet
        max_chars = snippet_budget * 4
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "..."

        lines = [header, snippet, url_line, ""]
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        stats = self.get_cache_stats()
        proxy_info = f", proxy={'ON' if self.proxy_configured else 'OFF'}"
        return (
            f"<WebSearcher: "
            f"{stats['total_searches']} searches, "
            f"{stats['cache_size']} cached, "
            f"ddgs={'OK' if DDGS_AVAILABLE else 'missing'}"
            f"{proxy_info}>"
        )


# =============================================================================
# ONE WEB PAGE, BEHIND THE SAME GATE
# =============================================================================
#
# The knowledge base ingests a web page by URL. The fetch opens the gate
# above before every request, a redirect included, and after every read of
# the body, and reaches only a public address: the host is resolved once per
# request, every answer must be public -- neither this machine's nor on one
# of its links -- and the connection goes only to answers that were checked,
# in the resolver's order, so a name whose answer changes in between cannot
# steer it inside. The request still names the URL's host, and TLS verifies
# that name. A redirect is followed by hand and checked exactly as the first
# request, at most MAX_REDIRECTS times. The body is capped as it is read, and
# one time budget covers the whole fetch: a timer cuts a socket that stalls
# past it. Name resolution itself is bounded by the system resolver's own
# timeouts, not by that budget.

_PAGE_REFUSALS = {
    "kill_switch": "Fetching a web page is refused: the search kill switch is engaged.",
    "mode": "Fetching a web page is refused outside Daily mode.",
    "unreadable": "Fetching a web page is refused: the search kill switch cannot be read.",
}
_WEB_PORTS = {"http": 80, "https": 443}
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
# A fixed policy bound, not a setting: each redirect is one more destination
# to check, and three cover the common http-to-https and canonical-host hops.
MAX_REDIRECTS = 3
_READ_CHUNK = 64 * 1024
_SHARED_ADDRESS_SPACE = ipaddress.ip_network("100.64.0.0/10")
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")

# This machine's own network tables, read at every request: its IPv6
# addresses with their prefixes, and the routes it reaches without a gateway.
# A platform without such a table has none to read; one that cannot be read
# refuses the fetch.
_LINK_TABLES = {
    "addresses6": "/proc/net/if_inet6",
    "routes6": "/proc/net/ipv6_route",
    "routes4": "/proc/net/route",
}
_RTF_UP, _RTF_GATEWAY, _RTF_REJECT = 0x0001, 0x0002, 0x0200
_VISIBLE_ASCII = "".join(chr(code) for code in range(0x21, 0x7F))
# The encodings a web page may name -- those of the WHATWG Encoding Standard
# that Python decodes -- by Python's own name for them.
_PAGE_ENCODINGS = frozenset(codecs.lookup(name).name for name in (
    "utf-8", "utf-16", "utf-16-le", "utf-16-be", "ascii", "latin-1",
    "iso8859-2", "iso8859-3", "iso8859-4", "iso8859-5", "iso8859-6", "iso8859-7", "iso8859-8",
    "iso8859-10", "iso8859-13", "iso8859-14", "iso8859-15", "iso8859-16",
    "cp866", "cp874", "cp1250", "cp1251", "cp1252", "cp1253", "cp1254", "cp1255", "cp1256",
    "cp1257", "cp1258", "koi8-r", "koi8-u", "mac-roman", "mac-cyrillic",
    "gbk", "gb18030", "big5", "euc-jp", "iso2022-jp", "shift-jis", "euc-kr",
))

# The fetch's clock, read at every step of its time budget.
_clock = time.monotonic


class PageFetchRefused(RuntimeError):
    """A page fetch refused by the web gate, carrying the refusal by name."""

    def __init__(self, refusal: str):
        super().__init__(_PAGE_REFUSALS[refusal])
        self.refusal = refusal


class DestinationRefused(ValueError):
    """A URL, or a redirect, whose destination a page fetch does not reach."""


class PageFetchFailed(ValueError):
    """A fetch that was allowed and brought no page back."""


@dataclass
class FetchedPage:
    """One page: where it came from after redirects, its type, its bytes and its text."""

    url: str
    status: int
    content_type: str
    body: bytes
    text: str


class _TimedOut(Exception):
    """The fetch's time budget ran out."""


class _TooLarge(Exception):
    """The body passed the size cap."""


class _Failure(Exception):
    """Why an allowed request brought no page back; the caller names the URL."""


def _carried(ip):
    """The IPv4 address an IPv6 address carries (mapped, or behind the NAT64 prefix), or None."""
    if ip.version != 6:
        return None
    inner = ip.ipv4_mapped
    if inner is None and ip in _NAT64_PREFIX:
        inner = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return inner


def _address_class(address: str) -> str | None:
    """The class of a non-public address, or None for a public one.

    An IPv4 address inside IPv6 (mapped, or behind the NAT64 prefix) is read
    as the IPv4 address it carries. Anything the standard library does not
    call global is refused, under the most precise name it has; so is the
    deprecated IPv6 site-local range, which the standard library calls global.
    """
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return "unreadable"
    inner = _carried(ip)
    if inner is not None:
        found = _address_class(str(inner))
        return None if found is None else f"{found}, an IPv4 address inside IPv6"
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.version == 6 and ip.is_site_local:
        return "site-local"
    if ip.is_multicast:
        return "multicast"
    if ip.version == 4 and ip in _SHARED_ADDRESS_SPACE:
        return "shared address space"
    if ip.is_reserved:
        return "reserved"
    if ip.is_private:
        return "private"
    if not ip.is_global:
        return "not global"
    return None


def _refused(hop: int) -> str:
    return "Refused" if hop == 0 else f"Refused at redirect {hop}"


def _held_here(ip) -> bool | None:
    """Whether this machine holds the address, or None when that cannot be told.

    A datagram socket binds to an address only when the machine holds it;
    the bind sends nothing.
    """
    family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
    try:
        probe = socket.socket(family, socket.SOCK_DGRAM)
    except OSError as exc:
        return False if exc.errno == errno.EAFNOSUPPORT else None
    try:
        probe.bind((str(ip), 0))
    except OSError as exc:
        return False if exc.errno == errno.EADDRNOTAVAIL else None
    finally:
        probe.close()
    return True


def _native4(value: int) -> str:
    """An IPv4 address the kernel's route table prints as a native 32-bit number."""
    return str(ipaddress.IPv4Address(struct.pack("=I", value)))


def _addresses6(lines: list[str]) -> list:
    """Each IPv6 address of this machine, as the network its prefix names."""
    found = []
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 6 or len(fields[0]) != 32:
            raise ValueError(line)
        address = ipaddress.IPv6Address(int(fields[0], 16))
        found.append(ipaddress.IPv6Network((address, int(fields[2], 16)), strict=False))
    return found


def _routes6(lines: list[str]) -> list:
    """The IPv6 networks this machine reaches without a gateway, every table's."""
    found = []
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 10 or len(fields[0]) != 32 or len(fields[4]) != 32:
            raise ValueError(line)
        prefix, flags = int(fields[1], 16), int(fields[8], 16)
        if prefix == 0 or int(fields[4], 16) != 0 or not flags & _RTF_UP or flags & (_RTF_GATEWAY | _RTF_REJECT):
            continue
        found.append(ipaddress.IPv6Network((ipaddress.IPv6Address(int(fields[0], 16)), prefix), strict=False))
    return found


def _routes4(lines: list[str]) -> list:
    """The IPv4 networks this machine reaches without a gateway."""
    found = []
    for line in lines:
        fields = line.split()
        if not fields or fields[0] == "Iface":
            continue
        if len(fields) < 8:
            raise ValueError(line)
        destination, gateway, flags, mask = (int(fields[i], 16) for i in (1, 2, 3, 7))
        if mask == 0 or gateway != 0 or not flags & _RTF_UP or flags & (_RTF_GATEWAY | _RTF_REJECT):
            continue
        found.append(ipaddress.IPv4Network((_native4(destination), _native4(mask)), strict=False))
    return found


def _link_networks(hop: int) -> list:
    """The networks on this machine's links, read from its tables now."""
    networks = []
    for table, parse in (("addresses6", _addresses6), ("routes6", _routes6), ("routes4", _routes4)):
        try:
            with open(_LINK_TABLES[table], encoding="ascii") as handle:
                lines = handle.read().splitlines()
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError) as exc:
            raise DestinationRefused(
                f"{_refused(hop)}: this machine's network table {table} cannot be read ({exc}), "
                "so no address can be checked against its links."
            ) from None
        try:
            networks += parse(lines)
        except (ValueError, struct.error):
            raise DestinationRefused(
                f"{_refused(hop)}: this machine's network table {table} cannot be read "
                "(a line does not parse), so no address can be checked against its links."
            ) from None
    return networks


def _local_class(address: str, links: list, hop: int) -> str | None:
    """Why a public address is still not reached -- this machine, or a link of it -- or None."""
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    ip = _carried(ip) or ip
    held = _held_here(ip)
    if held is None:
        raise DestinationRefused(f"{_refused(hop)}: whether {ip} is this machine's own address cannot be told.")
    if held:
        return "this machine"
    for network in links:
        if network.version == ip.version and ip in network:
            return f"local network, on the link {network}"
    return None


def _gate() -> None:
    """The web gate, for a page fetch: raise its refusal by name, or return."""
    refusal = search_refusal()
    if refusal is not None:
        raise PageFetchRefused(refusal)


def _destination(url: str, ports: frozenset, hop: int) -> tuple[str, str, int, str]:
    """(scheme, host, port, request target) of a URL a fetch may try.

    Refuses a backslash, whitespace or a control character anywhere in the
    URL (a browser reads such a URL otherwise), a scheme other than http or
    https, user information, and a port other than 80 and 443 unless the
    configuration names it.
    """
    where = _refused(hop)
    if any(ch == chr(0x5C) or ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url):
        raise DestinationRefused(f"{where}: the URL carries a backslash, a space or a control character.")
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in _WEB_PORTS:
        raise DestinationRefused(
            f"{where}: only http and https are fetched, not {scheme or 'a URL without a scheme'}."
        )
    if "@" in parts.netloc:
        raise DestinationRefused(f"{where}: the URL carries user information.")
    host = parts.hostname
    if not host:
        raise DestinationRefused(f"{where}: the URL names no host.")
    try:
        port = parts.port
    except ValueError:
        raise DestinationRefused(f"{where}: the URL's port cannot be read.") from None
    port = _WEB_PORTS[scheme] if port is None else port
    if port not in (80, 443) and port not in ports:
        raise DestinationRefused(
            f"{where}: port {port} is not fetched; only 80 and 443, and the ports the configuration allows."
        )
    target = quote(parts.path or "/", safe=_VISIBLE_ASCII)
    if parts.query:
        target += "?" + quote(parts.query, safe=_VISIBLE_ASCII)
    return scheme, host, port, target


def _checked_addresses(host: str, port: int, hop: int) -> list[str]:
    """The addresses a request may connect to: every answer checked, in the resolver's order."""
    try:
        answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        raise _Failure(f"the host name does not resolve ({exc})") from None
    addresses, links = [], None
    for _family, _kind, _proto, _name, sockaddr in answers:
        address = str(sockaddr[0])
        found = _address_class(address)
        if found is None:
            if links is None:
                links = _link_networks(hop)
            found = _local_class(address, links, hop)
        if found is not None:
            raise DestinationRefused(f"{_refused(hop)}: the host resolves to a non-public address ({found}).")
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise _Failure("the host name does not resolve")
    return addresses


def _connect_checked(addresses: list[str], port: int, deadline: float) -> socket.socket:
    """A socket to the first checked address that answers, in order, within the budget.

    Only the addresses this request checked are tried, as the resolver
    ordered them; the name is never resolved again.
    """
    failure = None
    for address in addresses:
        remaining = deadline - _clock()
        if remaining <= 0:
            raise _TimedOut()
        try:
            return socket.create_connection((address, port), remaining)
        except OSError as exc:
            failure = exc
    raise failure


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """A connection to an address the fetch checked; the request still names the URL's host."""

    def __init__(self, host: str, port: int, addresses: list[str], deadline: float):
        super().__init__(host, port, timeout=max(deadline - _clock(), 0.001))
        self._addresses = list(addresses)
        self._deadline = deadline
        self._live = None
        self._expired = False

    def connect(self):
        self.sock = self._live = _connect_checked(self._addresses, self.port, self._deadline)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """The same over TLS: the certificate is verified against the URL's host name."""

    def __init__(self, host: str, port: int, addresses: list[str], deadline: float, context: ssl.SSLContext):
        super().__init__(host, port, timeout=max(deadline - _clock(), 0.001), context=context)
        self._addresses = list(addresses)
        self._deadline = deadline
        self._live = None
        self._expired = False

    def connect(self):
        raw = _connect_checked(self._addresses, self.port, self._deadline)
        tls = self._context.wrap_socket(raw, server_hostname=self.host, do_handshake_on_connect=False)
        self._live = tls
        tls.do_handshake()
        self.sock = tls


def _tls_context() -> ssl.SSLContext:
    """Certificates verified, against the host name the URL names."""
    return ssl.create_default_context()


def _expire(connection) -> None:
    """The time budget ran out: mark the request and cut its socket, which wakes a stalled read."""
    connection._expired = True
    live = connection._live
    if live is not None:
        try:
            live.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _read_capped(connection, response, deadline: float, max_bytes: int) -> bytes:
    """The body, read at most one socket read at a time, within the cap and the budget.

    The gate is asked again after every read: a mode that leaves Daily, or a
    switch engaged, while the page is read stops the reading there. The
    reading ends when the response is complete: its socket may be closed by
    then, the server having closed the connection, so nothing is asked of it
    once the response says so. A body that ends before its declared length
    is a failure, not a page.
    """
    chunks, total = [], 0
    while not response.isclosed():
        remaining = deadline - _clock()
        if remaining <= 0 or connection._expired:
            raise _TimedOut()
        if connection._live is not None:
            connection._live.settimeout(remaining)
        chunk = response.read1(_READ_CHUNK)
        _gate()
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise _TooLarge()
        chunks.append(chunk)
    if connection._expired:
        raise _TimedOut()
    if response.length:
        raise _Failure("the connection closed before the declared length was read")
    return b"".join(chunks)


def _one_request(scheme, host, port, target, addresses, headers, deadline, max_bytes):
    """One GET to a checked address: (status, Location or None, (content type, body) or None)."""
    remaining = deadline - _clock()
    if remaining <= 0:
        raise _TimedOut()
    if scheme == "https":
        connection = _PinnedHTTPSConnection(host, port, addresses, deadline, _tls_context())
    else:
        connection = _PinnedHTTPConnection(host, port, addresses, deadline)
    watch = threading.Timer(remaining, _expire, args=(connection,))
    watch.daemon = True
    watch.start()
    try:
        try:
            connection.connect()
            if connection._expired:
                raise _TimedOut()
            connection.request("GET", target, headers=headers)
            response = connection.getresponse()
            status = response.status
            if status in _REDIRECT_STATUSES:
                return status, response.getheader("Location") or "", None
            if status >= 400:
                raise _Failure(f"the server answered {status} {response.reason}".rstrip())
            encoding = (response.getheader("Content-Encoding") or "identity").strip().lower()
            if encoding != "identity":
                raise _Failure(f"the server sent a compressed body ({encoding}) though none was accepted")
            # A length that is not ASCII digits is a length unknown, as the
            # response itself reads it.
            declared = (response.getheader("Content-Length") or "").strip()
            if declared.isascii() and declared.isdigit() and int(declared) > max_bytes:
                raise _TooLarge()
            body = _read_capped(connection, response, deadline, max_bytes)
            return status, None, (response.getheader("Content-Type") or "", body)
        except (OSError, http.client.HTTPException):
            if connection._expired or deadline <= _clock():
                raise _TimedOut() from None
            raise
    finally:
        watch.cancel()
        connection.close()


def _charset(content_type: str) -> str:
    """The charset a Content-Type names, when it is one of the web's encodings; UTF-8 otherwise.

    A codec that is no text encoding (``hex``, ``zlib``), or one the web
    does not use (``punycode``, whose decoding is quadratic, ``idna``,
    ``rot13``), names nothing a page is read with.
    """
    for parameter in content_type.split(";")[1:]:
        name, _sep, value = parameter.partition("=")
        if name.strip().lower() == "charset":
            candidate = value.strip().strip("\"'")
            try:
                found = codecs.lookup(candidate).name
            except (LookupError, ValueError):
                break
            return found if found in _PAGE_ENCODINGS else "utf-8"
    return "utf-8"


def fetch_page(
    url: str,
    *,
    timeout: float = 30,
    max_bytes: int = 5 * 1024 * 1024,
    user_agent: str = "Opti-Oignon RAG/1.0",
    allowed_ports=(),
) -> FetchedPage:
    """Fetch one web page behind the web gate, from a public address only.

    Every request, the first and each redirect's, opens the gate, is checked
    for its destination, resolves the host once and connects only to the
    addresses it checked, the next one tried when one does not answer; the
    gate is asked again after every read of the body. ``allowed_ports`` adds
    ports to 80 and 443; ``max_bytes`` caps the body as it is read;
    ``timeout`` is one budget for the whole fetch, in seconds. The text is
    decoded with the charset the page names when it is one of the web's
    encodings, UTF-8 otherwise.

    Raises:
        PageFetchRefused: The gate refused, before any request of this fetch
            or of one of its redirects, or while the body was read.
        DestinationRefused: The URL or a redirect names a destination a
            fetch does not reach, or more than MAX_REDIRECTS redirects, or
            this machine's network tables cannot be read to tell.
        PageFetchFailed: A request was allowed and brought no page back.
    """
    ports = frozenset(
        port for port in allowed_ports
        if isinstance(port, int) and not isinstance(port, bool) and 0 < port < 65536
    )
    budget = float(timeout)
    deadline = _clock() + budget
    headers = {"User-Agent": user_agent, "Accept": "*/*", "Connection": "close"}
    current = url
    for hop in range(MAX_REDIRECTS + 1):
        _gate()
        scheme, host, port, target = _destination(current, ports, hop)
        try:
            addresses = _checked_addresses(host, port, hop)
            status, location, page = _one_request(scheme, host, port, target, addresses, headers, deadline, max_bytes)
        except _TooLarge:
            raise PageFetchFailed(f"Page too large: more than {max_bytes} bytes.") from None
        except _TimedOut:
            raise PageFetchFailed(f"Failed to fetch URL {current}: no complete answer within {budget:g} s.") from None
        except _Failure as failure:
            raise PageFetchFailed(f"Failed to fetch URL {current}: {failure}.") from None
        except (OSError, http.client.HTTPException) as exc:
            raise PageFetchFailed(f"Failed to fetch URL {current}: {exc}") from None
        if page is None:
            if not location:
                raise DestinationRefused(f"Refused at redirect {hop + 1}: the answer names no Location.")
            current = urljoin(current, location.strip())
            continue
        content_type, body = page
        return FetchedPage(
            url=current, status=status, content_type=content_type, body=body,
            text=body.decode(_charset(content_type), errors="replace"),
        )
    raise DestinationRefused(f"Refused: more than {MAX_REDIRECTS} redirects.")


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_yaml_config = _load_config_from_yaml()
web_searcher = WebSearcher(config=_yaml_config)

# Backward-compatible alias used by tool_registry
web_search_engine = web_searcher


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Shortcut to web_searcher.search()."""
    return web_searcher.search(query, max_results=max_results)


def search_and_format(
    query: str,
    max_results: int = 3,
    token_budget: int = 1500,
) -> str:
    """Shortcut to web_searcher.search_and_format()."""
    return web_searcher.search_and_format(
        query, max_results=max_results, token_budget=token_budget
    )


def is_available() -> bool:
    """Check if web search is available (duckduckgo-search installed)."""
    return DDGS_AVAILABLE


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Opti-Oignon Web Search Module - CLI Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m opti_oignon.web_search "python dataclass tutorial"
  python -m opti_oignon.web_search --formatted "latest pandas release"
  python -m opti_oignon.web_search --budget 500 --formatted "R tidyverse"
  python -m opti_oignon.web_search --test
  python -m opti_oignon.web_search --proxy-check
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query",
    )
    parser.add_argument(
        "--formatted", "-f",
        action="store_true",
        help="Use search_and_format() with token budget",
    )
    parser.add_argument(
        "--budget", "-b",
        type=int,
        default=1500,
        help="Token budget for formatted output (default: 1500)",
    )
    parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=5,
        help="Max number of results (default: 5)",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="SOCKS5 proxy URL (e.g. socks5h://localhost:9050)",
    )
    parser.add_argument(
        "--proxy-check",
        action="store_true",
        help="Check proxy connectivity and exit",
    )
    parser.add_argument(
        "--sanitize-preview",
        action="store_true",
        help="Preview PII sanitization for the query",
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run full test suite",
    )

    args = parser.parse_args()

    print("=== Opti-Oignon Web Search Module ===\n")
    print(f"Version: {__version__}")
    print(f"duckduckgo-search: {'installed' if DDGS_AVAILABLE else 'NOT INSTALLED'}")
    print(f"PII sanitizer: {'available' if PII_AVAILABLE else 'not available'}")
    print(f"Proxy: {web_searcher.config.proxy or 'disabled (direct)'}")
    print()

    if args.proxy:
        web_searcher.set_proxy(args.proxy)

    if args.proxy_check:
        print("Checking proxy status...")
        status = web_searcher.check_proxy_status()
        print(f"  Configured: {status.configured}")
        print(f"  URL: {status.proxy_url}")
        print(f"  Reachable: {status.reachable}")
        if status.latency_ms is not None:
            print(f"  Latency: {status.latency_ms}ms")
        if status.exit_ip:
            print(f"  Exit IP: {status.exit_ip}")
        if status.error:
            print(f"  Error: {status.error}")
        sys.exit(0)

    if args.sanitize_preview and args.query:
        print(f"PII sanitization preview for: {args.query!r}")
        preview = web_searcher.preview_sanitization(args.query)
        print(f"  Sanitized: {preview['sanitized']!r}")
        print(f"  Modified: {preview['was_modified']}")
        for item in preview["items"]:
            print(f"  - [{item['category']}] {item['original']!r} -> {item['replacement']!r}")
        sys.exit(0)

    if not DDGS_AVAILABLE:
        print("ERROR: duckduckgo-search is not installed.")
        print("Install with: pip install ddgs")
        sys.exit(1)

    if args.test:
        print("=" * 60)
        print("TEST SUITE")
        print("=" * 60)

        print("\n--- Test 1: Basic search ---")
        results = web_searcher.search("python dataclass tutorial", max_results=3)
        print(f"Results: {len(results)}")
        for r in results:
            print(f"  - {r.title}")
            print(f"    {r.url}")
            print(f"    {r.snippet[:80]}...")
            print()

        print("\n--- Test 2: Formatted search (budget=800) ---")
        formatted = web_searcher.search_and_format(
            "python dataclass tutorial", max_results=3, token_budget=800
        )
        print(formatted)
        print(f"[Estimated tokens: ~{WebSearcher._estimate_tokens(formatted)}]")

        print("\n--- Test 3: Cache hit ---")
        stats_before = web_searcher.get_cache_stats()
        results2 = web_searcher.search("python dataclass tutorial", max_results=3)
        stats_after = web_searcher.get_cache_stats()
        print(f"Cache hits before: {stats_before['cache_hits']}")
        print(f"Cache hits after: {stats_after['cache_hits']}")
        print(f"Cache hit detected: {stats_after['cache_hits'] > stats_before['cache_hits']}")

        print("\n--- Test 4: Empty query ---")
        empty = web_searcher.search("")
        print(f"Results for empty query: {len(empty)} (expected: 0)")

        print("\n--- Test 5: PII sanitization ---")
        preview = web_searcher.preview_sanitization(
            "error on user@example.com at /home/user/project"
        )
        print(f"  Original: {preview['original']!r}")
        print(f"  Sanitized: {preview['sanitized']!r}")
        print(f"  Items found: {len(preview['items'])}")

        print("\n--- Test 6: Statistics ---")
        stats = web_searcher.get_cache_stats()
        for key, value in stats.items():
            if key != "cache_entries":
                print(f"  {key}: {value}")

        print("\n--- Test 7: Clear cache ---")
        cleared = web_searcher.clear_cache()
        print(f"Entries cleared: {cleared}")
        print(f"Cache size after clear: {web_searcher.get_cache_stats()['cache_size']}")

        print("\n" + "=" * 60)
        print("TESTS COMPLETE")
        print("=" * 60)

    elif args.query:
        if args.formatted:
            print(f"Formatted search: {args.query!r}")
            print(f"Budget: {args.budget} tokens, Max: {args.max_results} results\n")
            print("-" * 60)
            output = web_searcher.search_and_format(
                args.query,
                max_results=args.max_results,
                token_budget=args.budget,
            )
            if output:
                print(output)
                print("-" * 60)
                print(f"[~{WebSearcher._estimate_tokens(output)} tokens]")
            else:
                print("No results.")
        else:
            print(f"Search: {args.query!r}")
            print(f"Max: {args.max_results} results\n")
            results = web_searcher.search(args.query, max_results=args.max_results)
            if results:
                for i, r in enumerate(results, 1):
                    print(f"[{i}] {r.title}")
                    print(f"    {r.snippet}")
                    print(f"    URL: {r.url}")
                    print()
            else:
                print("No results.")

        print(f"\n{web_searcher}")

    else:
        parser.print_help()
