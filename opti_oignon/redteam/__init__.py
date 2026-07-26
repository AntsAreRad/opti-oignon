#!/usr/bin/env python3
"""
RED TEAM AUDIT FRAMEWORK — Opti-Oignon
==================================================

LLM-powered dynamic security testing of defense layers.
Generates adversarial attacks via local Ollama models, applies obfuscation
strategies, and evaluates each defense module's resilience.

100% local — no cloud calls, all generation through Ollama.

Architecture:
- generator.py   — AttackGenerator: produces raw attack strings per category
- strategies.py  — AttackStrategy: obfuscation transforms (base64, rot13, …)
- targets.py     — TargetAdapter: wraps each defense module for uniform testing
- config.py      — loads & validates config/redteam.yaml

Author: Leon
"""

__all__ = [
    "REDTEAM_ENABLED",
    "AttackGenerator",
    "AttackCategory",
    "AttackStrategy",
    "apply_strategy",
    "chain_strategies",
    "STRATEGY_REGISTRY",
    "TargetAdapter",
    "RAGSanitizerTarget",
    "RAGAugmenterTarget",
    "SearchSanitizerTarget",
    "PIISanitizerTarget",
    "SandboxTarget",
    "ChatTarget",
    "load_redteam_config",
    "RedTeamConfig",
    "SchedulerConfig",
    "RedTeamRunner",
    "CampaignRun",
    "RunProgress",
    "AttackScore",
    "CampaignScore",
    "score_result",
    "aggregate_scores",
    "generate_json_report",
    "generate_text_report",
    "generate_markdown_report",
    "Suggestion",
    "SuggestionStore",
    "suggestion_store",
    "extract_suggestions",
    "apply_suggestion_to_config",
]

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Feature flag — can be toggled via config/redteam.yaml or env var
REDTEAM_ENABLED: bool = True

try:
    from .config import RedTeamConfig, SchedulerConfig, load_redteam_config
except ImportError:
    logger.debug("redteam.config not available")

try:
    from .generator import AttackCategory, AttackGenerator
except ImportError:
    logger.debug("redteam.generator not available")

try:
    from .strategies import (
        STRATEGY_REGISTRY,
        AttackStrategy,
        apply_strategy,
        chain_strategies,
    )
except ImportError:
    logger.debug("redteam.strategies not available")

try:
    from .targets import (
        ChatTarget,
        PIISanitizerTarget,
        RAGAugmenterTarget,
        RAGSanitizerTarget,
        SandboxTarget,
        SearchSanitizerTarget,
        TargetAdapter,
    )
except ImportError:
    logger.debug("redteam.targets not available")

try:
    from .runner import CampaignRun, RedTeamRunner, RunProgress
except ImportError:
    logger.debug("redteam.runner not available")

try:
    from .scoring import AttackScore, CampaignScore, aggregate_scores, score_result
except ImportError:
    logger.debug("redteam.scoring not available")

try:
    from .reports import (
        generate_json_report,
        generate_markdown_report,
        generate_text_report,
    )
except ImportError:
    logger.debug("redteam.reports not available")

try:
    from .feedback import (
        Suggestion,
        SuggestionStore,
        suggestion_store,
        extract_suggestions,
        apply_suggestion_to_config,
    )
except ImportError:
    logger.debug("redteam.feedback not available")
