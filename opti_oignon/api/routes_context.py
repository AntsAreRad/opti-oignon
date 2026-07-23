#!/usr/bin/env python3
"""
API routes for context pipeline verification.

Provides a health endpoint for context window state,
l'utilisation des tokens, les statistiques de trimming et l'allocation
of the budget per model.
"""

import logging
from typing import Any

from fastapi import APIRouter, Query

from .deps import (
    CONTEXT_WINDOW_AVAILABLE,
    CONVERSATION_AVAILABLE,
    EXECUTOR_AVAILABLE,
    conversation_manager,
    executor,
    token_budget_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/context", tags=["context"])


def _estimate_tokens(text: str) -> int:
    """Estimation rapide du nombre de tokens (approximation ~4 chars/token).

    Args:
        text: Texte a estimer

    Returns:
        Estimation du nombre de tokens
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _get_conversation_context(conversation_id: str | None = None) -> dict[str, Any]:
    """Retrieve the context information for a conversation.

    Args:
        conversation_id: Conversation ID (optionnel)

    Returns:
        Dict with active conversation data
    """
    result = {
        "conversation_id": None,
        "model": None,
        "model_context_window": 0,
        "messages_count": 0,
        "estimated_tokens": 0,
        "usage_percent": 0.0,
        "trimming_active": False,
        "last_window_stats": {},
    }

    # Pas de gestionnaire de conversation disponible
    if not CONVERSATION_AVAILABLE or conversation_manager is None:
        return result

    # Determine target conversation
    target_id = conversation_id
    if not target_id:
        # Try the most recent conversation
        try:
            convs = conversation_manager.list_conversations()
            if convs:
                target_id = convs[0].get("id") if isinstance(convs[0], dict) else getattr(convs[0], "id", None)
        except Exception as e:
            logger.debug(f"Impossible de lister les conversations: {e}")
            return result

    if not target_id:
        return result

    result["conversation_id"] = target_id

    # Retrieve conversation messages
    try:
        msgs = conversation_manager.get_messages(target_id)
        result["messages_count"] = len(msgs) if msgs else 0

        # Estimer les tokens totaux
        total_tokens = 0
        for msg in (msgs or []):
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            total_tokens += _estimate_tokens(content)
        result["estimated_tokens"] = total_tokens
    except Exception as e:
        logger.debug(f"Impossible de recuperer les messages: {e}")

    # Retrieve the conversation model
    try:
        conv_data = conversation_manager.get_conversation(target_id)
        if conv_data:
            model = conv_data.get("model") if isinstance(conv_data, dict) else getattr(conv_data, "model", None)
            result["model"] = model
    except Exception as e:
        logger.debug(f"Impossible de recuperer la conversation: {e}")

    # Budget et fenetre de contexte
    model = result.get("model")
    if model and CONTEXT_WINDOW_AVAILABLE and token_budget_manager is not None:
        try:
            budget = token_budget_manager.get_budget(model)
            result["model_context_window"] = budget.context_window

            # Pourcentage d'utilisation
            if budget.context_window > 0:
                result["usage_percent"] = round(
                    (result["estimated_tokens"] / budget.context_window) * 100, 2
                )

            # Detection du trimming
            available = budget.available_for_history(0)
            result["trimming_active"] = result["estimated_tokens"] > available
        except Exception as e:
            logger.debug(f"Impossible de calculer le budget: {e}")

    # Stats de la derniere fenetre glissante (depuis l'executor)
    if EXECUTOR_AVAILABLE and executor is not None:
        try:
            window_stats = executor.last_window_stats
            if window_stats:
                result["last_window_stats"] = window_stats
                # Mettre a jour le statut trimming si on a des stats recentes
                if window_stats.get("dropped", 0) > 0:
                    result["trimming_active"] = True
        except Exception as e:
            logger.debug(f"Impossible de recuperer les stats de fenetre: {e}")

    return result


def _get_budget_allocation(model: str | None = None) -> dict[str, Any]:
    """Compute the budget allocation for a model.

    Args:
        model: Model name (optionnel)

    Returns:
        Dict with budget allocation
    """
    result = {
        "system_prompt": 0,
        "history": 0,
        "reserved_for_response": 0,
        "total_allocated": 0,
        "context_window": 0,
        "system_ratio": 0.0,
        "history_ratio": 0.0,
        "generation_ratio": 0.0,
    }

    if not CONTEXT_WINDOW_AVAILABLE or token_budget_manager is None:
        return result

    target_model = model or "default"

    try:
        budget = token_budget_manager.get_budget(target_model)
        result["system_prompt"] = budget.system_budget
        result["history"] = budget.history_budget
        result["reserved_for_response"] = budget.generation_budget
        result["total_allocated"] = budget.total_allocated
        result["context_window"] = budget.context_window
        result["system_ratio"] = round(budget.system_ratio, 3)
        result["history_ratio"] = round(budget.history_ratio, 3)
        result["generation_ratio"] = round(budget.generation_ratio, 3)
    except Exception as e:
        logger.debug(f"Impossible de calculer l'allocation: {e}")

    return result


@router.get("/health")
def context_health(
    conversation_id: str | None = Query(None, description="ID de conversation specifique"),
    model: str | None = Query(None, description="Modele pour l'allocation de budget"),
) -> dict:
    """Context pipeline health endpoint.

    Return the full context window state:
    - Disponibilite du module
    - Active conversation data
    - Token budget allocation
    """
    # Data de conversation
    conv_context = _get_conversation_context(conversation_id)

    # Model for budget: parameter takes priority, otherwise conversation model
    budget_model = model or conv_context.get("model")
    budget_allocation = _get_budget_allocation(budget_model)

    # Determiner le statut global
    status = "healthy"
    if not CONTEXT_WINDOW_AVAILABLE:
        status = "degraded"
    elif conv_context.get("trimming_active"):
        status = "trimming"
    elif conv_context.get("usage_percent", 0) > 90:
        status = "warning"

    return {
        "status": status,
        "context_window_available": CONTEXT_WINDOW_AVAILABLE,
        "executor_available": EXECUTOR_AVAILABLE,
        "conversation_available": CONVERSATION_AVAILABLE,
        "current_conversation": conv_context,
        "budget_allocation": budget_allocation,
    }


@router.get("/budget/{model_name}")
def get_model_budget(model_name: str) -> dict:
    """Retrieve the budget allocation for a specific model.

    Args:
        model_name: Ollama model name
    """
    if not CONTEXT_WINDOW_AVAILABLE or token_budget_manager is None:
        return {
            "available": False,
            "model": model_name,
            "budget": None,
        }

    allocation = _get_budget_allocation(model_name)

    return {
        "available": True,
        "model": model_name,
        "budget": allocation,
    }


@router.get("/stats")
def context_stats(
    conversation_id: str | None = Query(None, description="ID de conversation"),
) -> dict:
    """Statistiques detaillees de la fenetre glissante.

    Return stats of the last window operation
    glissante, incluant les messages gardes/supprimes et
    les metriques de tokens. Includes S123 optimization report
    when the context optimizer is active.
    """
    stats = {
        "available": False,
        "window_stats": {},
        "trimming_history": [],
        "optimization_report": None,
    }

    if EXECUTOR_AVAILABLE and executor is not None:
        try:
            window_stats = executor.last_window_stats
            if window_stats:
                stats["available"] = True
                stats["window_stats"] = window_stats
        except Exception as e:
            logger.debug(f"Impossible de recuperer les stats: {e}")

        # Include optimization report when available
        try:
            opt_report = executor.last_optimization_report
            if opt_report is not None:
                stats["optimization_report"] = opt_report.as_dict()
        except Exception as e:
            logger.debug(f"Could not retrieve optimization report: {e}")

    return stats
