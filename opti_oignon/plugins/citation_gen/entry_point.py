"""
Citation generator plugin for Opti-Oignon.

Generates formatted citations from RAG retrieval results. Supports
APA, MLA, and Chicago citation styles.
"""

from typing import Any

__plugin_name__: str = "citation-gen"
__plugin_version__: str = "1.0.0"

# Configuration defaults
_DEFAULT_STYLE = "apa"
_AUTO_CITE = True
_MAX_CITATIONS = 10

SUPPORTED_STYLES = ("apa", "mla", "chicago")


def _format_apa(source: dict[str, Any]) -> str:
    """Format a source in APA style.

    Expected source keys: author, title, year, source_file, url, page
    """
    author = source.get("author", "Unknown")
    year = source.get("year", "n.d.")
    title = source.get("title", source.get("source_file", "Untitled"))
    url = source.get("url", "")

    citation = f"{author} ({year}). {title}."
    if url:
        citation += f" Retrieved from {url}"
    return citation


def _format_mla(source: dict[str, Any]) -> str:
    """Format a source in MLA style."""
    author = source.get("author", "Unknown")
    title = source.get("title", source.get("source_file", "Untitled"))
    year = source.get("year", "n.d.")
    url = source.get("url", "")

    citation = f'{author}. "{title}." {year}.'
    if url:
        citation += f" {url}"
    return citation


def _format_chicago(source: dict[str, Any]) -> str:
    """Format a source in Chicago style."""
    author = source.get("author", "Unknown")
    title = source.get("title", source.get("source_file", "Untitled"))
    year = source.get("year", "n.d.")
    url = source.get("url", "")

    citation = f"{author}. {title}. {year}."
    if url:
        citation += f" {url}."
    return citation


_FORMATTERS = {
    "apa": _format_apa,
    "mla": _format_mla,
    "chicago": _format_chicago,
}


def format_citation(
    source: dict[str, Any],
    style: str = _DEFAULT_STYLE,
) -> str:
    """Format a single citation from a source dict.

    Parameters
    ----------
    source : dict
        Source metadata with keys like author, title, year, url.
    style : str
        Citation style: 'apa', 'mla', or 'chicago'.

    Returns
    -------
    str
        Formatted citation string.
    """
    formatter = _FORMATTERS.get(style.lower(), _format_apa)
    return formatter(source)


def format_citations(
    sources: list[dict[str, Any]],
    style: str = _DEFAULT_STYLE,
    max_citations: int = _MAX_CITATIONS,
) -> list[str]:
    """Format multiple citations.

    Returns list of formatted citation strings.
    """
    results: list[str] = []
    seen_titles: set[str] = set()

    for source in sources[:max_citations]:
        # Deduplicate by title
        title = source.get("title", source.get("source_file", ""))
        if title in seen_titles:
            continue
        seen_titles.add(title)
        results.append(format_citation(source, style))

    return results


def build_references_section(
    citations: list[str],
    style: str = _DEFAULT_STYLE,
) -> str:
    """Build a formatted references section from citation strings."""
    if not citations:
        return ""

    header = {
        "apa": "References",
        "mla": "Works Cited",
        "chicago": "Bibliography",
    }.get(style.lower(), "References")

    lines = [f"\n\n---\n**{header}**\n"]
    for i, citation in enumerate(citations, 1):
        lines.append(f"[{i}] {citation}")

    return "\n".join(lines)


def extract_sources_from_rag(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract source metadata from RAG context data.

    Looks for common keys: rag_results, sources, chunks, citations.
    """
    sources: list[dict[str, Any]] = []

    # Try various keys where RAG results might be stored
    for key in ("rag_results", "sources", "chunks", "citations", "rag_chunks"):
        items = data.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    sources.append(item)

    return sources


# =========================================================================
# Hook implementations
# =========================================================================

def hook_post_inference(ctx: Any) -> dict[str, Any] | None:
    """Auto-append citations to RAG-augmented responses.

    Checks ctx.data for RAG sources and appends a references section
    to the response text if auto_cite is enabled.
    """
    if not _AUTO_CITE:
        return None

    sources = extract_sources_from_rag(ctx.data)
    if not sources:
        return None

    response = ctx.data.get("response", "")
    if not response:
        return None

    style = ctx.config.get("citation_style", _DEFAULT_STYLE)
    citations = format_citations(sources, style=style)
    if not citations:
        return None

    refs_section = build_references_section(citations, style=style)
    return {
        "response": response + refs_section,
        "citations_added": len(citations),
        "citation_style": style,
    }


def hook_tool_call(ctx: Any) -> dict[str, Any] | None:
    """Handle direct citation formatting requests.

    Expects ctx.data:
        tool_name: "cite" or "citation" or "citation_gen"
        sources: list[dict] — source metadata
        style: str (optional)
    """
    tool_name = ctx.data.get("tool_name", "")
    if tool_name not in ("cite", "citation", "citation_gen"):
        return None

    sources = ctx.data.get("sources", [])
    if not isinstance(sources, list) or not sources:
        return {"result": None, "error": "No sources provided"}

    style = ctx.data.get("style", _DEFAULT_STYLE)
    if style not in SUPPORTED_STYLES:
        return {"result": None, "error": f"Unsupported style: {style}"}

    citations = format_citations(sources, style=style)
    refs = build_references_section(citations, style=style)

    return {
        "result": refs,
        "citations": citations,
        "style": style,
        "count": len(citations),
        "error": None,
    }


HOOKS = {
    "post_inference": hook_post_inference,
    "tool_call": hook_tool_call,
}


def init() -> None:
    """Plugin initialization."""
    pass


def shutdown() -> None:
    """Plugin shutdown."""
    pass
