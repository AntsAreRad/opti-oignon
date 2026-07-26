#!/usr/bin/env python3
"""
PROJECT TRIGGERS -- 3-Level Query Relevance Detection
============================================================

Determines whether a user query is relevant to the active project,
using a 3-level escalation strategy for speed vs accuracy:

  Level 1 (regex, <1ms): Explicit triggers like "@project", "project files",
      "look in the files", "dans le projet", etc.
  Level 2 (term matching, <10ms): Match query terms against key_terms
      extracted from indexed project files.
  Level 3 (LLM classification, <500ms): Ask the LLM if the query is
      project-relevant. Only used when L1 + L2 are inconclusive.

The escalation stops as soon as a definitive answer is found.

Author: Leon
"""

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# CONDITIONAL IMPORTS
# =============================================================================

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# ProjectStore for fetching key_terms
try:
    from opti_oignon.projects import project_store
    PROJECTS_AVAILABLE = True
except ImportError:
    PROJECTS_AVAILABLE = False
    project_store = None

# Executor for LLM classification (Level 3)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# =============================================================================
# CONSTANTS
# =============================================================================

_CONFIG_DIR = Path(__file__).parent / "config"

# Level 1: regex patterns for explicit project references
# These indicate the user is clearly talking about project files/context
_EXPLICIT_PATTERNS_EN = [
    r"@project\b",
    r"\bproject\s+files?\b",
    r"\blook\s+(?:in|at|through)\s+(?:the\s+)?files?\b",
    r"\bsearch\s+(?:in\s+)?(?:the\s+)?(?:project|files?)\b",
    r"\bfrom\s+(?:the\s+)?(?:project|files?)\b",
    r"\bin\s+(?:the\s+)?(?:project|uploaded)\s+files?\b",
    r"\bproject\s+context\b",
    r"\buse\s+(?:the\s+)?project\b",
    r"\bcheck\s+(?:the\s+)?project\b",
    r"\bmy\s+(?:project\s+)?files?\b",
    r"\buploaded\s+(?:files?|documents?)\b",
]

_EXPLICIT_PATTERNS_FR = [
    r"\bdans\s+le\s+projet\b",
    r"\bfichiers?\s+(?:du\s+)?projet\b",
    r"\bcherche[rz]?\s+dans\s+(?:les\s+)?fichiers?\b",
    r"\bregarde[rz]?\s+(?:dans\s+)?(?:les\s+)?fichiers?\b",
    r"\bcontexte\s+(?:du\s+)?projet\b",
    r"\butilise[rz]?\s+le\s+projet\b",
    r"\bmes\s+fichiers?\b",
    r"\bdocuments?\s+(?:du\s+)?projet\b",
]

# Compile all L1 patterns once
_L1_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in _EXPLICIT_PATTERNS_EN + _EXPLICIT_PATTERNS_FR
]

# Word-boundary verdict patterns for the L3 classification.
# The previous substring parse checked "YES" first, so any answer
# containing YES anywhere (e.g. "NO. YES would imply ...") returned True
# -- the AGL-01 verdict-substring class. A naive \bNO\b alone would miss
# "NOT RELEVANT", so NOT counts as NO. Exactly one side must match; both
# or neither is ambiguous and returns None, falling through to the
# conservative not-relevant default.
_L3_YES_RE = re.compile(r"\bYES\b")
_L3_NO_RE = re.compile(r"\bNOT?\b")

# Level 3: LLM classification prompt template
_LLM_CLASSIFICATION_PROMPT = """You are a classifier. Determine if the user query is related to project files or context.

Project name: {project_name}
Project description: {project_description}
Indexed files: {file_list}

User query: {query}

Answer with ONLY one word: YES or NO
- YES if the query likely refers to, needs, or would benefit from the project's files or context
- NO if the query is general knowledge, unrelated, or self-contained

Answer:"""

# Default trigger config
_DEFAULT_TRIGGER_CONFIG = {
    "level1_enabled": True,
    "level2_enabled": True,
    "level3_enabled": True,
    "level2_min_matches": 2,
    "level2_min_score": 0.15,
    "level3_model": "",
    "level3_timeout_ms": 500,
    "level3_ollama_url": "http://localhost:11434",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RelevanceResult:
    """Result of trigger detection for a query.

    Attributes:
        relevant: Whether the query is project-relevant.
        confidence: Confidence level (0.0 to 1.0).
        trigger_level: Which level determined the result (1, 2, 3, or 0 if none).
        matched_terms: Terms that matched in L2 (if applicable).
        matched_pattern: Regex pattern that matched in L1 (if applicable).
        duration_ms: Total detection time in milliseconds.
        details: Additional details for debugging.
    """

    relevant: bool = False
    confidence: float = 0.0
    trigger_level: int = 0
    matched_terms: list[str] = field(default_factory=list)
    matched_pattern: str = ""
    duration_ms: float = 0.0
    details: str = ""


# =============================================================================
# HELPERS
# =============================================================================

def _load_trigger_config() -> dict[str, Any]:
    """Load trigger detection configuration from projects.yaml."""
    config = dict(_DEFAULT_TRIGGER_CONFIG)
    config_path = _CONFIG_DIR / "projects.yaml"
    if not YAML_AVAILABLE or not config_path.exists():
        return config

    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        trigger_cfg = raw.get("projects", {}).get("triggers", {})
        config.update(trigger_cfg)
    except Exception as e:
        logger.warning("Failed to load trigger config: %s", e)

    return config


def _tokenize_query(query: str) -> set[str]:
    """Tokenize a query into lowercase word tokens (3+ chars).

    Args:
        query: The user query string.

    Returns:
        Set of lowercase word tokens.
    """
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", query)
    return {w.lower() for w in words}


# =============================================================================
# TRIGGER DETECTOR
# =============================================================================

class ProjectTriggerDetector:
    """3-level trigger detection for project query relevance.

    Escalation order:
      L1 (regex) -> L2 (term matching) -> L3 (LLM classification)

    Each level is independently toggleable via config. The detector
    stops as soon as a definitive result is found.
    """

    def __init__(self, store: Any | None = None):
        """Initialize the trigger detector.

        Args:
            store: ProjectStore instance (uses global singleton if None).
        """
        self._config = _load_trigger_config()
        self._store = store if store is not None else project_store

    def detect(
        self,
        query: str,
        project_id: str,
        skip_l3: bool = False,
    ) -> RelevanceResult:
        """Run 3-level trigger detection on a query.

        Args:
            query: The user query to analyze.
            project_id: The project ID to check against.
            skip_l3: If True, skip LLM classification (for speed).

        Returns:
            RelevanceResult with detection outcome.
        """
        start = time.monotonic()
        result = RelevanceResult()

        if not query.strip():
            result.duration_ms = (time.monotonic() - start) * 1000
            return result

        # -- Level 1: Regex matching --
        if self._config.get("level1_enabled", True):
            l1_result = self._check_level1(query)
            if l1_result is not None:
                result.relevant = True
                result.confidence = 0.95
                result.trigger_level = 1
                result.matched_pattern = l1_result
                result.details = f"L1 regex match: {l1_result}"
                result.duration_ms = (time.monotonic() - start) * 1000
                logger.debug("Trigger L1 hit: %s (%.1fms)", l1_result, result.duration_ms)
                return result

        # -- Level 2: Term matching --
        if self._config.get("level2_enabled", True):
            l2_result = self._check_level2(query, project_id)
            if l2_result is not None:
                matched_terms, score = l2_result
                min_score = self._config.get("level2_min_score", 0.15)
                if score >= min_score:
                    result.relevant = True
                    result.confidence = min(0.85, 0.5 + score)
                    result.trigger_level = 2
                    result.matched_terms = matched_terms
                    result.details = f"L2 term match: {matched_terms} (score={score:.2f})"
                    result.duration_ms = (time.monotonic() - start) * 1000
                    logger.debug("Trigger L2 hit: %s (%.1fms)", matched_terms, result.duration_ms)
                    return result

        # -- Level 3: LLM classification --
        if (
            self._config.get("level3_enabled", True)
            and not skip_l3
        ):
            l3_result = self._check_level3(query, project_id)
            if l3_result is not None:
                result.relevant = l3_result
                result.confidence = 0.70 if l3_result else 0.60
                result.trigger_level = 3
                result.details = f"L3 LLM classification: {'relevant' if l3_result else 'not relevant'}"
                result.duration_ms = (time.monotonic() - start) * 1000
                logger.debug("Trigger L3: %s (%.1fms)", l3_result, result.duration_ms)
                return result

        # No level triggered: not relevant
        result.relevant = False
        result.confidence = 0.5
        result.details = "No trigger level matched"
        result.duration_ms = (time.monotonic() - start) * 1000
        return result

    # =========================================================================
    # LEVEL 1: REGEX
    # =========================================================================

    def _check_level1(self, query: str) -> str | None:
        """Check for explicit project references via regex.

        Args:
            query: The user query.

        Returns:
            The matched pattern string, or None if no match.
        """
        for pattern in _L1_PATTERNS:
            if pattern.search(query):
                return pattern.pattern
        return None

    # =========================================================================
    # LEVEL 2: TERM MATCHING
    # =========================================================================

    def _check_level2(
        self,
        query: str,
        project_id: str,
    ) -> tuple | None:
        """Match query terms against project file key_terms.

        Computes a relevance score based on the proportion of query
        tokens that match indexed key_terms.

        Args:
            query: The user query.
            project_id: The project to check against.

        Returns:
            Tuple of (matched_terms, score) if matches found, else None.
        """
        if self._store is None:
            return None

        # Collect all key_terms from indexed project files
        try:
            files = self._store.list_files(project_id)
        except Exception:
            return None

        all_terms: set[str] = set()
        for pf in files:
            if pf.indexed and pf.key_terms:
                for term in pf.key_terms:
                    all_terms.add(term.lower())

        if not all_terms:
            return None

        # Tokenize the query
        query_tokens = _tokenize_query(query)
        if not query_tokens:
            return None

        # Find matches
        matched = query_tokens & all_terms
        min_matches = self._config.get("level2_min_matches", 2)

        if len(matched) < min_matches:
            return None

        # Score = proportion of query tokens that match
        score = len(matched) / len(query_tokens)
        return (sorted(matched), score)

    # =========================================================================
    # LEVEL 3: LLM CLASSIFICATION
    # =========================================================================

    def _check_level3(
        self,
        query: str,
        project_id: str,
    ) -> bool | None:
        """Ask the LLM whether the query is project-relevant.

        Uses a direct Ollama API call with a strict timeout to stay
        within the 500ms budget.

        Args:
            query: The user query.
            project_id: The project ID.

        Returns:
            True if relevant, False if not, None if classification failed.
        """
        if not REQUESTS_AVAILABLE:
            return None

        if self._store is None:
            return None

        # Get project details for the prompt
        project = self._store.get_project(project_id)
        if project is None:
            return None

        # Get file list for context
        try:
            files = self._store.list_files(project_id)
            file_list = ", ".join(f.filename for f in files[:20])
            if len(files) > 20:
                file_list += f" ... and {len(files) - 20} more"
        except Exception:
            file_list = "(no files)"

        # Build the classification prompt
        prompt = _LLM_CLASSIFICATION_PROMPT.format(
            project_name=project.name,
            project_description=project.description or "(no description)",
            file_list=file_list or "(no files)",
            query=query,
        )

        # Determine model
        model = self._config.get("level3_model", "")
        if not model:
            # Try to use a fast model from the project's settings
            model = project.settings.get("default_model", "")
        if not model:
            # Fallback: try to detect an available model
            model = "qwen3:32b"

        # Make the Ollama API call with strict timeout
        timeout_ms = self._config.get("level3_timeout_ms", 500)
        timeout_s = max(timeout_ms / 1000, 0.5)
        ollama_url = self._config.get("level3_ollama_url", "http://localhost:11434")

        try:
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 5,
                    },
                },
                timeout=timeout_s,
            )
            response.raise_for_status()
            answer = response.json().get("response", "").strip().upper()

            has_yes = bool(_L3_YES_RE.search(answer))
            has_no = bool(_L3_NO_RE.search(answer))
            if has_yes and not has_no:
                return True
            if has_no and not has_yes:
                return False
            logger.debug("L3 ambiguous answer: '%s'", answer)
            return None

        except requests.exceptions.Timeout:
            logger.debug("L3 LLM classification timed out (%dms)", timeout_ms)
            return None
        except Exception as e:
            logger.debug("L3 LLM classification failed: %s", e)
            return None


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

trigger_detector = ProjectTriggerDetector()
