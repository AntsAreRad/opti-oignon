#!/usr/bin/env python3
"""
API routes for Learned Router — Opti-Oignon S67.

Endpoints for training status inspection, manual retraining,
config management, live query classification, and A/B metrics.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/routing/learned", tags=["learned-routing"])


# =============================================================================
# Pydantic schemas
# =============================================================================

class TrainingResultResponse(BaseModel):
    """Result of a training run."""
    accuracy: float
    n_samples: int
    n_classes: int
    trained_at: float
    model_type: str
    cv_folds: int
    success: bool
    error: str


class LearnedRouterStatusResponse(BaseModel):
    """Full status snapshot for the learned router."""
    available: bool
    trained: bool
    enabled: bool
    sklearn_available: bool
    sample_count: int
    samples_since_retrain: int
    min_training_samples: int
    class_distribution: dict
    last_training: dict | None
    model_type: str
    confidence_threshold: float
    auto_retrain_interval: int


class LearnedRouterConfigResponse(BaseModel):
    """Current learned router configuration."""
    enabled: bool
    model_type: str
    confidence_threshold: float
    min_training_samples: int
    auto_retrain_interval: int
    feature_max_features: int
    feature_ngram_range: list
    logistic_max_iter: int
    logistic_C: float
    random_forest_n_estimators: int
    random_forest_max_depth: int | None
    max_stored_samples: int
    cv_folds: int


class LearnedRouterConfigUpdateRequest(BaseModel):
    """Partial config update for the learned router.

    All fields optional; only provided fields are updated.
    """
    enabled: bool | None = None
    model_type: str | None = Field(
        None,
        description="Classifier type: 'logistic' or 'random_forest'",
    )
    confidence_threshold: float | None = Field(None, ge=0.0, le=1.0)
    min_training_samples: int | None = Field(None, ge=5, le=10000)
    auto_retrain_interval: int | None = Field(None, ge=10, le=10000)
    feature_max_features: int | None = Field(None, ge=100, le=50000)
    max_stored_samples: int | None = Field(None, ge=100, le=100000)
    cv_folds: int | None = Field(None, ge=2, le=10)


class ClassifyRequest(BaseModel):
    """Request body for live query classification."""
    query: str = Field(..., min_length=1, max_length=5000)
    yaml_task_type: str = Field(
        default="general",
        description="Task type from YAML heuristic router (for comparison)",
    )


class ClassifyResponse(BaseModel):
    """Classification result for a single query."""
    ml_prediction: dict          # RoutingPrediction.to_dict()
    yaml_task_type: str          # Original YAML prediction
    final_task_type: str         # Task type that would be used
    routing_source: str          # 'learned' or 'yaml'
    confidence: float


class ABMetricsResponse(BaseModel):
    """A/B metrics comparing learned vs YAML routing."""
    total_decisions: int
    learned_count: int
    yaml_count: int
    learned_ratio: float
    avg_ml_confidence: float
    avg_ml_confidence_learned: float
    avg_ml_confidence_yaml: float
    class_agreement_rate: float
    top_disagreements: list
    decisions_by_source: dict
    window_hours: float
    confidence_histogram: list


# =============================================================================
# Dependency helpers
# =============================================================================

def _get_learned_router():
    """Return the LearnedRouter singleton or raise 503."""
    try:
        from opti_oignon.learned_router import LEARNED_ROUTER_AVAILABLE, learned_router
        if not LEARNED_ROUTER_AVAILABLE or learned_router is None:
            raise HTTPException(
                status_code=503,
                detail="Learned router unavailable (sklearn not installed)",
            )
        return learned_router
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Learned router module not available",
        )


def _get_metrics(router):
    """Return a LearnedRouterMetrics instance for the given router."""
    from opti_oignon.learned_router import LearnedRouterMetrics
    return LearnedRouterMetrics(router)


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/status", response_model=LearnedRouterStatusResponse)
def get_status() -> dict:
    """Return training status, sample counts, and current config summary.

    Returns 503 if sklearn is not installed.
    """
    lr = _get_learned_router()
    try:
        status = lr.get_status()
        return LearnedRouterStatusResponse(**status)
    except Exception as exc:
        logger.error("GET /status failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/train", response_model=TrainingResultResponse)
def trigger_training() -> dict:
    """Trigger a manual retraining of the ML classifier.

    Requires at least min_training_samples in the store.
    Returns the TrainingResult with accuracy and metadata.
    """
    lr = _get_learned_router()
    try:
        result = lr.train()
        return TrainingResultResponse(**result.to_dict())
    except Exception as exc:
        logger.error("POST /train failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/config", response_model=LearnedRouterConfigResponse)
def get_config() -> dict:
    """Return the current learned router configuration."""
    lr = _get_learned_router()
    try:
        cfg = lr.get_config()
        ngram = cfg.get("feature_ngram_range", [1, 2])
        if not isinstance(ngram, list):
            ngram = list(ngram)
        max_depth = cfg.get("random_forest_max_depth")
        return LearnedRouterConfigResponse(
            enabled=cfg.get("enabled", False),
            model_type=cfg.get("model_type", "logistic"),
            confidence_threshold=cfg.get("confidence_threshold", 0.70),
            min_training_samples=cfg.get("min_training_samples", 50),
            auto_retrain_interval=cfg.get("auto_retrain_interval", 100),
            feature_max_features=cfg.get("feature_max_features", 5000),
            feature_ngram_range=ngram,
            logistic_max_iter=cfg.get("logistic_max_iter", 1000),
            logistic_C=cfg.get("logistic_C", 1.0),
            random_forest_n_estimators=cfg.get("random_forest_n_estimators", 100),
            random_forest_max_depth=max_depth,
            max_stored_samples=cfg.get("max_stored_samples", 10000),
            cv_folds=cfg.get("cv_folds", 5),
        )
    except Exception as exc:
        logger.error("GET /config failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.put("/config")
def update_config(request: LearnedRouterConfigUpdateRequest) -> dict:
    """Apply a partial config update to the learned router.

    Only provided fields are updated; others remain unchanged.
    Enabling the router requires it to be trained first.
    """
    lr = _get_learned_router()
    updates = {k: v for k, v in request.model_dump().items() if v is not None}

    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided for update")

    # Guard: cannot enable an untrained router
    if updates.get("enabled") is True and not lr.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Cannot enable learned router before training. "
                   "POST /api/routing/learned/train first.",
        )

    # Validate model_type
    if "model_type" in updates and updates["model_type"] not in ("logistic", "random_forest"):
        raise HTTPException(
            status_code=400,
            detail="model_type must be 'logistic' or 'random_forest'",
        )

    try:
        lr.update_config(updates)
        return {"success": True, "updated": list(updates.keys())}
    except Exception as exc:
        logger.error("PUT /config failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/classify", response_model=ClassifyResponse)
def classify_query(request: ClassifyRequest) -> dict:
    """Classify a query using both ML model and YAML heuristic.

    Returns the ML prediction, the YAML comparison, and which source
    would be used based on the current confidence threshold.
    Useful for testing/previewing routing decisions before enabling.
    """
    lr = _get_learned_router()
    try:
        ml_pred = lr.classify(request.query)
        fallback_result = lr.classify_with_fallback(request.query, request.yaml_task_type)

        routing_source = "yaml" if fallback_result.fallback_used else "learned"
        final_task = fallback_result.task_type

        return ClassifyResponse(
            ml_prediction=ml_pred.to_dict(),
            yaml_task_type=request.yaml_task_type,
            final_task_type=final_task,
            routing_source=routing_source,
            confidence=ml_pred.confidence,
        )
    except Exception as exc:
        logger.error("POST /classify failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/metrics", response_model=ABMetricsResponse)
def get_ab_metrics(window_hours: float = 24.0) -> dict:
    """Return A/B comparison metrics between learned and YAML routing.

    Args:
        window_hours: Time window for aggregation (default 24h).
    """
    if window_hours <= 0 or window_hours > 8760:
        raise HTTPException(
            status_code=400,
            detail="window_hours must be between 0 and 8760",
        )

    lr = _get_learned_router()
    try:
        metrics_obj = _get_metrics(lr)
        result = metrics_obj.compute(window_hours=window_hours)
        histogram = metrics_obj.get_confidence_histogram(bins=10, window_hours=window_hours)
        d = result.to_dict()
        d["confidence_histogram"] = histogram
        return ABMetricsResponse(**d)
    except Exception as exc:
        logger.error("GET /metrics failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
