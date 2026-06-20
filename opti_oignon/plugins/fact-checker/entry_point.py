"""
Fact-checker plugin for Opti-Oignon.

Parses factual claims in LLM responses (dates, numbers, proper nouns,
named entities) and cross-references them via DuckDuckGo web search.
Annotates inline with [verified], [unverified], or [conflict: ...].

Gracefully degrades to [unverified] when web search is unavailable.
"""

import logging
import re
from typing import Any, Optional

__plugin_name__: str = "fact-checker"
__plugin_version__: str = "1.0.0"

logger = logging.getLogger(__name__)

# =========================================================================
# Web search integration (optional)
# =========================================================================

_WEB_SEARCH_AVAILABLE = False
_web_search_fn = None

try:
    from opti_oignon.web_search import web_searcher, DDGS_AVAILABLE
    if DDGS_AVAILABLE:
        _WEB_SEARCH_AVAILABLE = True
        _web_search_fn = web_searcher.search
except Exception:
    pass

# =========================================================================
# Claim extraction patterns
# =========================================================================

# Dates: "in 1969", "on March 15, 2023", "January 2020", "2024-01-15"
_DATE_PATTERN = re.compile(
    r"\b(?:in|on|since|from|during|around|by)\s+"
    r"(?:"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:,\s*\d{4})?"
    r"|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{4}"
    r"|"
    r"\d{4}(?:-\d{2}(?:-\d{2})?)?"
    r")",
    re.IGNORECASE,
)

# Numeric claims: "X is 384,400 km", "costs $29.99", "weighs 150 kg"
_NUMERIC_CLAIM_PATTERN = re.compile(
    r"(?:is|are|was|were|approximately|about|roughly|exactly|nearly|over|under)"
    r"\s+[\$]?\d[\d,]*\.?\d*\s*"
    r"(?:%|percent|km|mi|miles|kg|lb|pounds|meters|feet|dollars|euros|billion|million|thousand)?",
    re.IGNORECASE,
)

# Proper nouns: sequences of capitalized words (2+ words, not at sentence start)
_PROPER_NOUN_PATTERN = re.compile(
    r"(?:(?<=[.!?]\s)|(?<=,\s)|(?<=;\s)|(?<=\n))([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
)

# Named entity claims: "X founded Y", "X invented Y", "X is the capital of Y"
_ENTITY_CLAIM_PATTERN = re.compile(
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
    r"\s+(?:founded|invented|discovered|created|developed|wrote|designed|built|is the (?:capital|president|CEO|founder|author) of)"
    r"\s+(.+?)(?:\.|,|;|\n|$)",
    re.IGNORECASE,
)

# Code block boundaries (to skip)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.DOTALL)

# Inline code (to skip)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")


# =========================================================================
# Claim extraction
# =========================================================================

class Claim:
    """A factual claim extracted from text."""

    __slots__ = ("text", "category", "start", "end", "context")

    def __init__(
        self,
        text: str,
        category: str,
        start: int,
        end: int,
        context: str = "",
    ) -> None:
        self.text = text
        self.category = category
        self.start = start
        self.end = end
        self.context = context

    def __repr__(self) -> str:
        return f"Claim({self.category!r}, {self.text!r})"


def _get_sentence_context(text: str, start: int, end: int) -> str:
    """Extract the sentence containing a match for search context."""
    # Find sentence boundaries
    sent_start = text.rfind(".", 0, start)
    sent_start = 0 if sent_start == -1 else sent_start + 1
    sent_end = text.find(".", end)
    sent_end = len(text) if sent_end == -1 else sent_end + 1
    return text[sent_start:sent_end].strip()


def _get_code_block_ranges(text: str) -> list[tuple[int, int]]:
    """Return list of (start, end) ranges for code blocks and inline code."""
    ranges: list[tuple[int, int]] = []
    for m in _CODE_BLOCK_RE.finditer(text):
        ranges.append((m.start(), m.end()))
    for m in _INLINE_CODE_RE.finditer(text):
        ranges.append((m.start(), m.end()))
    return ranges


def _in_code_block(pos: int, ranges: list[tuple[int, int]]) -> bool:
    """Check if a position falls within any code block range."""
    for start, end in ranges:
        if start <= pos < end:
            return True
    return False


def extract_claims(
    text: str,
    *,
    aggressiveness: str = "moderate",
    skip_code_blocks: bool = True,
    max_claims: int = 10,
) -> list[Claim]:
    """Extract factual claims from text.

    Parameters
    ----------
    text : str
        The LLM response text to analyze.
    aggressiveness : str
        Level of claim extraction: low, moderate, or high.
    skip_code_blocks : bool
        Whether to skip claims inside code blocks.
    max_claims : int
        Maximum number of claims to extract.

    Returns
    -------
    list[Claim]
        Extracted claims sorted by position.
    """
    claims: list[Claim] = []
    code_ranges = _get_code_block_ranges(text) if skip_code_blocks else []

    def _add_claims(pattern: re.Pattern[str], category: str) -> None:
        for m in pattern.finditer(text):
            if len(claims) >= max_claims:
                return
            if skip_code_blocks and _in_code_block(m.start(), code_ranges):
                continue
            context = _get_sentence_context(text, m.start(), m.end())
            claims.append(Claim(
                text=m.group(0).strip(),
                category=category,
                start=m.start(),
                end=m.end(),
                context=context,
            ))

    # Always extract dates and numbers
    _add_claims(_DATE_PATTERN, "date")
    _add_claims(_NUMERIC_CLAIM_PATTERN, "numeric")

    # Moderate+: add proper nouns and entity claims
    if aggressiveness in ("moderate", "high"):
        _add_claims(_ENTITY_CLAIM_PATTERN, "entity")

    # High: add proper noun sequences
    if aggressiveness == "high":
        _add_claims(_PROPER_NOUN_PATTERN, "proper_noun")

    # Sort by position, deduplicate overlapping claims
    claims.sort(key=lambda c: c.start)
    deduped: list[Claim] = []
    last_end = -1
    for claim in claims:
        if claim.start >= last_end:
            deduped.append(claim)
            last_end = claim.end
    return deduped[:max_claims]


# =========================================================================
# Verification via web search
# =========================================================================

def verify_claim(
    claim: Claim,
    *,
    max_results: int = 3,
    trusted_domains: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Verify a single claim against web search results.

    Returns
    -------
    dict with keys:
        status: "verified" | "unverified" | "conflict"
        detail: Optional explanation for conflicts
        sources: List of source URLs consulted
    """
    if not _WEB_SEARCH_AVAILABLE or _web_search_fn is None:
        return {"status": "unverified", "detail": None, "sources": []}

    # Build search query from claim context
    query = claim.context if claim.context else claim.text
    # Trim to reasonable length for search
    if len(query) > 150:
        query = query[:150]

    try:
        results = _web_search_fn(query, max_results=max_results)
    except Exception as exc:
        logger.debug("Web search failed for claim %r: %s", claim.text, exc)
        return {"status": "unverified", "detail": None, "sources": []}

    if not results:
        return {"status": "unverified", "detail": None, "sources": []}

    sources = [r.url for r in results if hasattr(r, "url")]
    snippets = [r.snippet for r in results if hasattr(r, "snippet")]

    # Check trusted domains first
    if trusted_domains:
        for result in results:
            url = getattr(result, "url", "")
            for domain in trusted_domains:
                if domain in url:
                    return {
                        "status": "verified",
                        "detail": None,
                        "sources": sources,
                    }

    # Simple verification heuristic: check if key terms from
    # the claim appear in search snippets
    claim_lower = claim.text.lower()
    # Extract key terms (numbers, proper nouns)
    key_terms = re.findall(r"\b\d[\d,]*\.?\d*\b", claim_lower)
    key_terms += [
        w for w in claim_lower.split()
        if len(w) > 3 and w[0].isalpha()
    ]

    if not key_terms:
        return {"status": "unverified", "detail": None, "sources": sources}

    snippets_text = " ".join(snippets).lower()
    matched = sum(1 for term in key_terms if term in snippets_text)
    match_ratio = matched / len(key_terms) if key_terms else 0

    if match_ratio >= 0.5:
        return {"status": "verified", "detail": None, "sources": sources}

    # Check for contradicting information
    # Look for negation near claim terms
    contradiction_words = ["not", "incorrect", "false", "wrong", "actually", "however"]
    has_contradiction = False
    conflict_detail = None
    for snippet in snippets:
        snippet_lower = snippet.lower()
        for neg in contradiction_words:
            if neg in snippet_lower:
                for term in key_terms:
                    if term in snippet_lower:
                        has_contradiction = True
                        conflict_detail = snippet[:120]
                        break
            if has_contradiction:
                break
        if has_contradiction:
            break

    if has_contradiction and conflict_detail:
        return {
            "status": "conflict",
            "detail": conflict_detail,
            "sources": sources,
        }

    return {"status": "unverified", "detail": None, "sources": sources}


# =========================================================================
# Response annotation
# =========================================================================

def annotate_response(
    text: str,
    claims: list[Claim],
    results: list[dict[str, Any]],
) -> str:
    """Insert verification badges into the response text.

    Inserts badges after each verified claim, working from the end
    of the text backwards to preserve positions.

    Parameters
    ----------
    text : str
        Original response text.
    claims : list[Claim]
        Extracted claims (sorted by position).
    results : list[dict]
        Verification results parallel to claims.

    Returns
    -------
    str
        Annotated response text.
    """
    if not claims or not results:
        return text

    # Work backwards to preserve string positions
    annotated = text
    for claim, result in reversed(list(zip(claims, results))):
        status = result.get("status", "unverified")
        if status == "verified":
            badge = " [verified]"
        elif status == "conflict":
            detail = result.get("detail", "")
            if detail:
                # Truncate conflict detail
                detail = detail[:80].rstrip()
                badge = f" [conflict: {detail}]"
            else:
                badge = " [conflict]"
        else:
            badge = " [unverified]"

        # Insert badge at end of claim
        insert_pos = claim.end
        if insert_pos <= len(annotated):
            annotated = annotated[:insert_pos] + badge + annotated[insert_pos:]

    return annotated


# =========================================================================
# Hook implementation
# =========================================================================

def hook_post_inference(ctx: Any) -> Optional[dict[str, Any]]:
    """Post-inference hook: verify factual claims in LLM response.

    Reads ctx.data["response"], extracts claims, verifies via web
    search, and returns modified response with inline annotations.
    """
    response = ctx.data.get("response", "")
    if not response or len(response) < 50:
        return None

    # Read config
    config = ctx.config or {}
    aggressiveness = config.get("aggressiveness", "moderate")
    max_checks = config.get("max_checks", 5)
    skip_code = config.get("skip_code_blocks", True)
    trusted_str = config.get("trusted_domains", "")
    trusted_domains = (
        [d.strip() for d in trusted_str.split(",") if d.strip()]
        if trusted_str
        else None
    )

    # Extract claims
    claims = extract_claims(
        response,
        aggressiveness=aggressiveness,
        skip_code_blocks=skip_code,
        max_claims=max_checks,
    )

    if not claims:
        return None

    # Verify each claim
    results: list[dict[str, Any]] = []
    for claim in claims:
        result = verify_claim(
            claim,
            max_results=3,
            trusted_domains=trusted_domains,
        )
        results.append(result)

    # Annotate
    annotated = annotate_response(response, claims, results)

    # Build summary
    verified_count = sum(1 for r in results if r["status"] == "verified")
    unverified_count = sum(1 for r in results if r["status"] == "unverified")
    conflict_count = sum(1 for r in results if r["status"] == "conflict")

    return {
        "response": annotated,
        "fact_check_summary": {
            "total_claims": len(claims),
            "verified": verified_count,
            "unverified": unverified_count,
            "conflicts": conflict_count,
        },
    }


# =========================================================================
# Hook registry
# =========================================================================

HOOKS = {
    "post_inference": hook_post_inference,
}


def init() -> None:
    """Plugin initialization."""
    if not _WEB_SEARCH_AVAILABLE:
        logger.info(
            "fact-checker: web search unavailable, "
            "all claims will be marked [unverified]"
        )


def shutdown() -> None:
    """Plugin shutdown."""
    pass
