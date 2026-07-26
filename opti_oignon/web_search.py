#!/usr/bin/env python3
"""
WEB SEARCH MODULE - Opti-Oignon
================================

DuckDuckGo-based web search with caching, rate limiting, token-budgeted
formatting, SOCKS5/Tor proxy support, and PII sanitization.

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

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# CONDITIONAL IMPORTS
# =============================================================================

try:
    # New package name (ddgs >= 7.0)
    from ddgs import DDGS
    DDGS_AVAILABLE = True
    DuckDuckGoSearchException = Exception
    RatelimitException = Exception
    TimeoutException = Exception
except ImportError:
    try:
        # Legacy package name (duckduckgo_search < 7.0)
        from duckduckgo_search import DDGS  # type: ignore[no-redef]
        from duckduckgo_search.exceptions import (
            DuckDuckGoSearchException,
            RatelimitException,
            TimeoutException,
        )
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False
        DDGS = None
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
        self._config = config or _load_search_safety_config()
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
            "S125 PROMPT INJECTION DETECTED: pattern=%s matched=%r in %s",
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
            start = time.monotonic()
            if not DDGS_AVAILABLE:
                status.error = "duckduckgo-search not installed"
                return status

            ddgs = DDGS(
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
        Search the web via DuckDuckGo with optional proxy and PII sanitization.

        Args:
            query: Search query string
            max_results: Maximum number of results (default from config)

        Returns:
            List of SearchResult objects (may be empty on error)

        Raises:
            RuntimeError: If ddgs/duckduckgo-search is not installed
        """
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
            return cached

        self._stats["cache_misses"] += 1

        # Rate limiting
        self._enforce_rate_limit()

        # Execute search with retry logic
        results = self._search_with_retry(sanitized_query, max_results)

        if results is not None:
            self._put_in_cache(cache_key, results)
            logger.info(
                f"Search complete: {sanitized_query!r} -> {len(results)} result(s)"
                f"{' (via proxy)' if self.proxy_configured else ''}"
            )
            return results

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

                ddgs = DDGS(**ddgs_kwargs)
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

                # Sanitize results against prompt injection
                results = get_search_sanitizer().sanitize_results(results)

                return results

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
            "error on user@example.com at /home/leon/project"
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
