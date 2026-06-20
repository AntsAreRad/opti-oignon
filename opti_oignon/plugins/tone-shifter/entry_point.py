"""
Tone-shifter plugin for Opti-Oignon.

Transforms LLM response tone via rule-based text processing.
No re-inference needed -- each mode applies regex and string
replacement rules sequentially to shift the response tone.

Supported modes:
    academic  -- hedging language, formal citations style
    casual    -- contractions, conversational markers
    eli5      -- simplify jargon, short sentences, analogies markers
    formal    -- remove contractions, professional phrasing
    concise   -- strip filler, compress paragraphs
    verbose   -- expand abbreviations, add transitions

Pure text processing -- no external dependencies, no permissions needed.
"""

import logging
import re
from typing import Any, Optional

__plugin_name__: str = "tone-shifter"
__plugin_version__: str = "1.0.0"

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration defaults
# =========================================================================

_DEFAULT_MODE = "none"

# =========================================================================
# Transformation rule type
# =========================================================================

# Each rule is (pattern, replacement, flags)
# pattern: regex string
# replacement: replacement string (supports backrefs)
# flags: re flags (0 for none)

Rule = tuple[str, str, int]

# =========================================================================
# Code block protection
# =========================================================================

_CODE_BLOCK_RE = re.compile(r"(```[\s\S]*?```|`[^`\n]+`)", re.DOTALL)


def _protect_code_blocks(text: str) -> tuple[str, list[str]]:
    """Replace code blocks with placeholders to avoid transforming them.

    Returns the modified text and a list of extracted code blocks.
    """
    blocks: list[str] = []

    def _replace(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f"__CODE_BLOCK_{len(blocks) - 1}__"

    protected = _CODE_BLOCK_RE.sub(_replace, text)
    return protected, blocks


def _restore_code_blocks(text: str, blocks: list[str]) -> str:
    """Restore code blocks from placeholders."""
    for i, block in enumerate(blocks):
        text = text.replace(f"__CODE_BLOCK_{i}__", block)
    return text


# =========================================================================
# Academic mode rules
# =========================================================================

ACADEMIC_RULES: list[Rule] = [
    # Add hedging to absolute statements
    (r"\bIt is\b", "It appears to be", 0),
    (r"\bThis is\b", "This can be considered", 0),
    (r"\bThis shows\b", "This suggests", 0),
    (r"\bThis proves\b", "This provides evidence that", 0),
    (r"\bclearly\b", "arguably", re.IGNORECASE),
    (r"\bobviously\b", "evidently", re.IGNORECASE),
    (r"\bdefinitely\b", "likely", re.IGNORECASE),
    (r"\balways\b", "typically", re.IGNORECASE),
    (r"\bnever\b", "rarely", re.IGNORECASE),
    # Formalize casual phrasing
    (r"\ba lot of\b", "a significant number of", re.IGNORECASE),
    (r"\bbig\b", "substantial", re.IGNORECASE),
    (r"\bget\b", "obtain", re.IGNORECASE),
    (r"\bkind of\b", "somewhat", re.IGNORECASE),
    (r"\bsort of\b", "to some extent", re.IGNORECASE),
    (r"\bguess\b", "hypothesize", re.IGNORECASE),
    (r"\bstuff\b", "material", re.IGNORECASE),
    (r"\bthings\b", "elements", re.IGNORECASE),
]

# =========================================================================
# Casual mode rules
# =========================================================================

CASUAL_RULES: list[Rule] = [
    # Add contractions
    (r"\bdo not\b", "don't", re.IGNORECASE),
    (r"\bcannot\b", "can't", re.IGNORECASE),
    (r"\bwill not\b", "won't", re.IGNORECASE),
    (r"\bis not\b", "isn't", re.IGNORECASE),
    (r"\bare not\b", "aren't", re.IGNORECASE),
    (r"\bshould not\b", "shouldn't", re.IGNORECASE),
    (r"\bwould not\b", "wouldn't", re.IGNORECASE),
    (r"\bcould not\b", "couldn't", re.IGNORECASE),
    (r"\bI am\b", "I'm", 0),
    (r"\bI have\b", "I've", 0),
    (r"\bI will\b", "I'll", 0),
    (r"\bit is\b", "it's", 0),
    (r"\bthat is\b", "that's", 0),
    (r"\bthey are\b", "they're", 0),
    # Casual markers
    (r"\bFurthermore\b", "Also", 0),
    (r"\bMoreover\b", "Plus", 0),
    (r"\bHowever\b", "But", 0),
    (r"\bTherefore\b", "So", 0),
    (r"\bConsequently\b", "So", 0),
    (r"\bIn addition\b", "Also", 0),
    (r"\bNevertheless\b", "Still", 0),
    (r"\butilize\b", "use", re.IGNORECASE),
    (r"\bcommence\b", "start", re.IGNORECASE),
    (r"\bterminate\b", "end", re.IGNORECASE),
    (r"\bfacilitate\b", "help", re.IGNORECASE),
    (r"\bimplement\b", "set up", re.IGNORECASE),
]

# =========================================================================
# ELI5 mode rules
# =========================================================================

ELI5_RULES: list[Rule] = [
    # Simplify jargon
    (r"\balgorithm\b", "step-by-step recipe", re.IGNORECASE),
    (r"\bparameter\b", "setting", re.IGNORECASE),
    (r"\bconfiguration\b", "setup", re.IGNORECASE),
    (r"\boptimize\b", "make better", re.IGNORECASE),
    (r"\boptimization\b", "making things better", re.IGNORECASE),
    (r"\blatency\b", "waiting time", re.IGNORECASE),
    (r"\bbandwidth\b", "how much can flow through", re.IGNORECASE),
    (r"\bdatabase\b", "organized collection of info", re.IGNORECASE),
    (r"\bAPI\b", "way for programs to talk to each other", 0),
    (r"\bserver\b", "computer that serves info", re.IGNORECASE),
    (r"\bfunction\b", "mini program", re.IGNORECASE),
    (r"\bvariable\b", "container for a value", re.IGNORECASE),
    (r"\bframeworks?\b", "toolkit", re.IGNORECASE),
    (r"\brepository\b", "code storage", re.IGNORECASE),
    (r"\bauthentication\b", "proving who you are", re.IGNORECASE),
    (r"\bencryption\b", "secret code scrambling", re.IGNORECASE),
    # Simplify transitions
    (r"\bFurthermore\b", "Also", 0),
    (r"\bConsequently\b", "So", 0),
    (r"\bNevertheless\b", "But", 0),
    (r"\bTherefore\b", "So", 0),
]

# =========================================================================
# Formal mode rules
# =========================================================================

FORMAL_RULES: list[Rule] = [
    # Remove contractions
    (r"\bdon't\b", "do not", re.IGNORECASE),
    (r"\bcan't\b", "cannot", re.IGNORECASE),
    (r"\bwon't\b", "will not", re.IGNORECASE),
    (r"\bisn't\b", "is not", re.IGNORECASE),
    (r"\baren't\b", "are not", re.IGNORECASE),
    (r"\bshouldn't\b", "should not", re.IGNORECASE),
    (r"\bwouldn't\b", "would not", re.IGNORECASE),
    (r"\bcouldn't\b", "could not", re.IGNORECASE),
    (r"\bI'm\b", "I am", 0),
    (r"\bI've\b", "I have", 0),
    (r"\bI'll\b", "I will", 0),
    (r"\bit's\b", "it is", 0),
    (r"\bthat's\b", "that is", 0),
    (r"\bthey're\b", "they are", 0),
    (r"\bwe're\b", "we are", 0),
    (r"\byou're\b", "you are", 0),
    (r"\blet's\b", "let us", 0),
    (r"\bhere's\b", "here is", 0),
    (r"\bthere's\b", "there is", 0),
    # Formalize casual words
    (r"\bget\b", "obtain", re.IGNORECASE),
    (r"\bgot\b", "obtained", re.IGNORECASE),
    (r"\bfix\b", "rectify", re.IGNORECASE),
    (r"\bbig\b", "significant", re.IGNORECASE),
    (r"\bhelp\b", "assist", re.IGNORECASE),
    (r"\bbuy\b", "purchase", re.IGNORECASE),
    (r"\bstart\b", "commence", re.IGNORECASE),
    (r"\bend\b", "conclude", re.IGNORECASE),
]

# =========================================================================
# Concise mode rules
# =========================================================================

# Filler phrases to strip entirely
_FILLER_PHRASES = [
    r"\bin other words,?\s*",
    r"\bthat being said,?\s*",
    r"\bit is worth noting that\s*",
    r"\bas mentioned (?:above|earlier|previously),?\s*",
    r"\bit should be noted that\s*",
    r"\bfor what it is worth,?\s*",
    r"\bat the end of the day,?\s*",
    r"\bbased on my (?:understanding|knowledge),?\s*",
    r"\bas a matter of fact,?\s*",
    r"\bto be (?:honest|fair),?\s*",
    r"\bneedless to say,?\s*",
    r"\bwith that (?:being )?said,?\s*",
    r"\bin (?:this|that) regard,?\s*",
    r"\ball things considered,?\s*",
    r"\bin order to\b",
    r"\bdue to the fact that\b",
    r"\bfor the purpose of\b",
    r"\bin the event that\b",
    r"\bprior to\b",
    r"\bsubsequent to\b",
    r"\bin the process of\b",
]

CONCISE_RULES: list[Rule] = [
    # Strip filler phrases
    *[(phrase, "", re.IGNORECASE) for phrase in _FILLER_PHRASES],
    # Compress verbose constructions
    (r"\bin order to\b", "to", re.IGNORECASE),
    (r"\bdue to the fact that\b", "because", re.IGNORECASE),
    (r"\bfor the purpose of\b", "for", re.IGNORECASE),
    (r"\bin the event that\b", "if", re.IGNORECASE),
    (r"\bprior to\b", "before", re.IGNORECASE),
    (r"\bsubsequent to\b", "after", re.IGNORECASE),
    (r"\bhas the ability to\b", "can", re.IGNORECASE),
    (r"\bis able to\b", "can", re.IGNORECASE),
    (r"\bin a manner that\b", "so that", re.IGNORECASE),
    (r"\bin the process of\b", "while", re.IGNORECASE),
    (r"\ba (?:large |significant |considerable )?number of\b", "many", re.IGNORECASE),
    # Clean up double spaces from removals
    (r"  +", " ", 0),
    # Clean up trailing spaces before punctuation
    (r" ([.,;:!?])", r"\1", 0),
]

# =========================================================================
# Verbose mode rules
# =========================================================================

VERBOSE_RULES: list[Rule] = [
    # Expand common abbreviations
    (r"\be\.g\.(?:\s|$|,)", "for example ", 0),
    (r"\bi\.e\.(?:\s|$|,)", "that is to say ", 0),
    (r"\betc\.(?:\s|$)", "and so on ", 0),
    (r"\bvs\.(?:\s|$)", "versus ", 0),
    (r"\bw/o\b", "without", 0),
    (r"\bw/\b", "with", 0),
    (r"\binfo\b", "information", re.IGNORECASE),
    (r"\bdocs\b", "documentation", re.IGNORECASE),
    (r"\bdoc\b", "document", re.IGNORECASE),
    (r"\bconfig\b", "configuration", re.IGNORECASE),
    (r"\brepo\b", "repository", re.IGNORECASE),
    (r"\benv\b", "environment", re.IGNORECASE),
    # Add transitional phrases
    (r"\. But\b", ". However,", 0),
    (r"\. So\b", ". Therefore,", 0),
    (r"\. Also\b", ". Furthermore,", 0),
    (r"\. Plus\b", ". Additionally,", 0),
]

# =========================================================================
# Mode registry
# =========================================================================

MODE_RULES: dict[str, list[Rule]] = {
    "academic": ACADEMIC_RULES,
    "casual": CASUAL_RULES,
    "eli5": ELI5_RULES,
    "formal": FORMAL_RULES,
    "concise": CONCISE_RULES,
    "verbose": VERBOSE_RULES,
}

AVAILABLE_MODES: frozenset[str] = frozenset(MODE_RULES.keys()) | {"none"}

# =========================================================================
# Transformation engine
# =========================================================================


def apply_rules(text: str, rules: list[Rule]) -> str:
    """Apply a list of regex replacement rules to text sequentially.

    Parameters
    ----------
    text : str
        Input text (code blocks should already be protected).
    rules : list[Rule]
        List of (pattern, replacement, flags) tuples.

    Returns
    -------
    str
        Transformed text.
    """
    for pattern, replacement, flags in rules:
        try:
            text = re.sub(pattern, replacement, text, flags=flags)
        except re.error:
            logger.warning("Invalid regex pattern in tone rule: %s", pattern)
            continue
    return text


def apply_custom_rules(text: str, custom: dict[str, str]) -> str:
    """Apply user-defined custom replacement rules.

    Parameters
    ----------
    text : str
        Input text.
    custom : dict[str, str]
        Mapping of regex patterns to replacement strings.

    Returns
    -------
    str
        Transformed text.
    """
    for pattern, replacement in custom.items():
        try:
            text = re.sub(pattern, replacement, text)
        except re.error:
            logger.warning("Invalid custom regex: %s", pattern)
            continue
    return text


def transform_tone(
    text: str,
    mode: str,
    custom_rules: Optional[dict[str, str]] = None,
) -> str:
    """Transform text tone according to mode and optional custom rules.

    Code blocks are protected from transformation.

    Parameters
    ----------
    text : str
        Input response text.
    mode : str
        Tone mode name (academic, casual, eli5, formal, concise, verbose).
    custom_rules : dict[str, str], optional
        Additional custom regex replacement rules.

    Returns
    -------
    str
        Transformed text.
    """
    if mode == "none" and not custom_rules:
        return text

    # Protect code blocks
    protected, blocks = _protect_code_blocks(text)

    # Apply mode rules
    if mode in MODE_RULES:
        protected = apply_rules(protected, MODE_RULES[mode])

    # Apply custom rules
    if custom_rules:
        protected = apply_custom_rules(protected, custom_rules)

    # Restore code blocks
    return _restore_code_blocks(protected, blocks)


# =========================================================================
# Hook implementation
# =========================================================================


def hook_post_inference(ctx: Any) -> Optional[dict[str, Any]]:
    """Post-inference hook: transform response tone.

    Reads the active mode from config and applies the corresponding
    transformation rules to the LLM response. Code blocks are
    preserved untouched.
    """
    response = ctx.data.get("response", "")
    if not response:
        return None

    config = ctx.config or {}
    mode = config.get("active_mode", _DEFAULT_MODE)
    custom = config.get("custom_rules", {})

    if mode == "none" and not custom:
        return None

    if mode not in AVAILABLE_MODES:
        logger.warning("Unknown tone mode: %s (available: %s)", mode, AVAILABLE_MODES)
        return None

    transformed = transform_tone(response, mode, custom_rules=custom or None)

    if transformed == response:
        return None

    return {
        "response": transformed,
        "tone_mode": mode,
    }


# =========================================================================
# Hook registry
# =========================================================================

HOOKS = {
    "post_inference": hook_post_inference,
}


def init() -> None:
    """Plugin initialization."""
    pass


def shutdown() -> None:
    """Plugin shutdown."""
    pass
