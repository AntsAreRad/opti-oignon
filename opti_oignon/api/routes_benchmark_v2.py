#!/usr/bin/env python3
"""
API routes for Benchmark V2.

Provides endpoints for listing profiles, starting benchmark runs,
polling progress, retrieving results, comparing models, viewing
historical data, LLM-as-Judge evaluation, leaderboard, head-to-head
comparison, trend analysis, recommendations, export, custom profile
CRUD, question preview, auto-trigger management, and test poll.
"""

import csv
import io
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from .deps import (
    AUTO_TRIGGER_AVAILABLE,
    BENCHMARK_JUDGE_AVAILABLE,
    BENCHMARK_RECOMMENDATIONS_AVAILABLE,
    BENCHMARK_RUNNER_AVAILABLE,
    BENCHMARK_V2_AVAILABLE,
    CUSTOM_PROFILES_AVAILABLE,
    auto_trigger,
    benchmark_evaluator,
    benchmark_recommender,
    benchmark_runner,
    custom_profile_store,
    judge_store,
)
from .schemas import (
    BenchmarkV2ApplyResponse,
    BenchmarkV2AutoTriggerConfigResponse,
    BenchmarkV2AutoTriggerConfigUpdate,
    BenchmarkV2AutoTriggerEventResponse,
    BenchmarkV2AutoTriggerEventsResponse,
    BenchmarkV2AutoTriggerStatusResponse,
    BenchmarkV2AutoTriggerTestPollResponse,
    BenchmarkV2CompareResponse,
    BenchmarkV2CustomProfileCreate,
    BenchmarkV2CustomProfileResponse,
    BenchmarkV2CustomProfilesListResponse,
    BenchmarkV2CustomProfileUpdate,
    BenchmarkV2HeadToHeadMetric,
    BenchmarkV2HeadToHeadResponse,
    BenchmarkV2HistoryEntry,
    BenchmarkV2HistoryResponse,
    BenchmarkV2LeaderboardEntry,
    BenchmarkV2LeaderboardResponse,
    BenchmarkV2ModelScore,
    BenchmarkV2ProfileSchema,
    BenchmarkV2ProfilesResponse,
    BenchmarkV2ProgressResponse,
    BenchmarkV2QuestionPreviewResponse,
    BenchmarkV2QuestionResult,
    BenchmarkV2RecommendationEntry,
    BenchmarkV2RecommendationsResponse,
    BenchmarkV2ResultsResponse,
    BenchmarkV2RunRequest,
    BenchmarkV2RunStarted,
    BenchmarkV2TrendPoint,
    BenchmarkV2TrendResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/benchmark/v2", tags=["benchmark_v2"])


# ---- Profiles ----

@router.get("/profiles", response_model=BenchmarkV2ProfilesResponse)
def list_profiles() -> dict:
    """List available benchmark profiles with their categories and weights."""
    if not BENCHMARK_V2_AVAILABLE or benchmark_evaluator is None:
        return BenchmarkV2ProfilesResponse()

    profiles = [
        BenchmarkV2ProfileSchema(**p)
        for p in benchmark_evaluator.available_profiles
    ]

    return BenchmarkV2ProfilesResponse(
        profiles=profiles,
        available_categories=benchmark_evaluator.available_categories,
        total_questions=benchmark_evaluator.question_count(),
    )


# ---- Run management ----

@router.post("/run", response_model=BenchmarkV2RunStarted)
def start_run(request: BenchmarkV2RunRequest) -> dict:
    """Start an asynchronous benchmark run.

    Runs the specified profile against the given models. Progress
    can be polled via GET /status/{run_id}. Optionally enables
    LLM-as-Judge evaluation with use_judge and judge_model.
    """
    if not BENCHMARK_RUNNER_AVAILABLE or benchmark_runner is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark runner not available",
        )

    if not request.models:
        raise HTTPException(
            status_code=400,
            detail="At least one model is required",
        )

    if not request.profile:
        raise HTTPException(
            status_code=400,
            detail="Profile name is required",
        )

    if request.use_judge and not request.judge_model:
        raise HTTPException(
            status_code=400,
            detail="judge_model is required when use_judge is true",
        )

    # BMK-04: timing-based speed scores are only valid when a single
    # benchmark owns the Ollama backend; mirror the v1 single-run guard.
    if getattr(benchmark_runner, "is_busy", False):
        raise HTTPException(
            status_code=409,
            detail="A benchmark run is already in progress. Cancel it or wait for completion.",
        )

    # Validate profile exists
    if BENCHMARK_V2_AVAILABLE and benchmark_evaluator is not None:
        config = benchmark_evaluator.get_profile_config(request.profile)
        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{request.profile}' not found",
            )

    try:
        run_id = benchmark_runner.start_run(
            profile=request.profile,
            models=request.models,
            use_judge=request.use_judge,
            judge_model=request.judge_model,
            custom_weights=request.custom_weights,
        )
        return BenchmarkV2RunStarted(
            run_id=run_id,
            profile=request.profile,
            models=request.models,
            status="running",
        )
    except Exception as e:
        logger.error("Failed to start benchmark run: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start benchmark run: {e}",
        )


@router.get("/status/{run_id}", response_model=BenchmarkV2ProgressResponse)
def get_run_status(run_id: str) -> dict:
    """Poll the progress of a running benchmark."""
    if not BENCHMARK_RUNNER_AVAILABLE or benchmark_runner is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark runner not available",
        )

    progress = benchmark_runner.get_progress(run_id)
    if progress is None:
        # Check if it's a completed run in the store
        stored = benchmark_runner.get_results(run_id)
        if stored:
            return BenchmarkV2ProgressResponse(
                run_id=run_id,
                status=stored.get("status", "completed"),
                total_questions=0,
                completed_questions=0,
                error=stored.get("error", ""),
            )
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found",
        )

    return BenchmarkV2ProgressResponse(
        run_id=progress.run_id,
        status=progress.status.value,
        total_questions=progress.total_questions,
        completed_questions=progress.completed_questions,
        current_model=progress.current_model,
        current_question=progress.current_question,
        elapsed_ms=progress.elapsed_ms,
        error=progress.error,
    )


@router.post("/cancel/{run_id}")
def cancel_run(run_id: str) -> dict:
    """Request cancellation of a running benchmark."""
    if not BENCHMARK_RUNNER_AVAILABLE or benchmark_runner is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark runner not available",
        )

    cancelled = benchmark_runner.cancel_run(run_id)
    if not cancelled:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found or already completed",
        )
    return {"run_id": run_id, "status": "cancelling"}


# ---- Results ----

@router.get("/results/{run_id}", response_model=BenchmarkV2ResultsResponse)
def get_run_results(run_id: str) -> dict:
    """Get detailed results for a completed benchmark run.

    Includes per-model scores, per-question evaluation details,
    and judge scores when available.
    """
    if not BENCHMARK_RUNNER_AVAILABLE or benchmark_runner is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark runner not available",
        )

    data = benchmark_runner.get_results(run_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found",
        )

    # Convert model scores
    model_scores = {}
    for model, ms in data.get("model_scores", {}).items():
        model_scores[model] = BenchmarkV2ModelScore(
            model=ms.get("model", model),
            accuracy_avg=ms.get("accuracy_avg", 0.0),
            code_avg=ms.get("code_avg", 0.0),
            structure_avg=ms.get("structure_avg", 0.0),
            speed_avg=ms.get("speed_avg", 0.0),
            composite=ms.get("composite", 0.0),
            questions_evaluated=ms.get("questions_evaluated", 0),
        )

    # Convert question results
    question_results = {}
    for model, qrs in data.get("question_results", {}).items():
        question_results[model] = [
            BenchmarkV2QuestionResult(
                question_id=qr.get("question_id", ""),
                category=qr.get("category", ""),
                prompt=qr.get("prompt", ""),
                response=qr.get("response", ""),
                accuracy_score=qr.get("accuracy_score", 0.0),
                code_score=qr.get("code_score", 0.0),
                structure_score=qr.get("structure_score", 0.0),
                speed_score=qr.get("speed_score", 0.0),
                composite_score=qr.get("composite_score", 0.0),
                details=qr.get("details", {}),
            )
            for qr in qrs
        ]

    # Fetch judge scores if available
    j_scores: list[dict] = []
    j_summary: dict = {}
    if BENCHMARK_JUDGE_AVAILABLE and judge_store is not None:
        try:
            j_scores = judge_store.get_scores_for_run(run_id)
            j_summary = judge_store.get_summary_for_run(run_id)
        except Exception as e:
            logger.debug("Could not fetch judge scores for %s: %s", run_id, e)

    return BenchmarkV2ResultsResponse(
        run_id=data.get("run_id", run_id),
        profile=data.get("profile", ""),
        models=data.get("models", []),
        status=data.get("status", ""),
        started_at=data.get("started_at", 0.0),
        finished_at=data.get("finished_at", 0.0),
        duration_ms=data.get("duration_ms", 0.0),
        weight_preset=data.get("weight_preset", "balanced"),
        custom_weights=data.get("custom_weights"),
        model_scores=model_scores,
        question_results=question_results,
        judge_scores=j_scores,
        judge_summary=j_summary,
        error=data.get("error", ""),
    )


# ---- Comparison ----

@router.get("/compare", response_model=BenchmarkV2CompareResponse)
def compare_models(
    models: str | None = Query(
        default=None,
        description="Comma-separated list of model names to compare",
    ),
    profile: str | None = Query(
        default=None,
        description="Filter by profile name",
    ),
    limit: int = Query(default=10, ge=1, le=100),
) -> dict:
    """Compare model performance across historical benchmark runs.

    Aggregates scores from completed runs, optionally filtered by
    model names and profile.
    """
    if not BENCHMARK_RUNNER_AVAILABLE or benchmark_runner is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark runner not available",
        )

    model_list = None
    if models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]

    result = benchmark_runner.compare(
        models=model_list,
        profile=profile,
        limit=limit,
    )

    return BenchmarkV2CompareResponse(
        models=result.get("models", []),
        profile_filter=result.get("profile_filter"),
        model_filter=result.get("model_filter"),
    )


# ---- History ----

@router.get("/history", response_model=BenchmarkV2HistoryResponse)
def get_history(
    limit: int = Query(default=50, ge=1, le=200),
    profile: str | None = Query(default=None),
    model: str | None = Query(default=None),
) -> dict:
    """Get historical benchmark runs with summary scores.

    Returns a list of past runs ordered by most recent first.
    """
    if not BENCHMARK_RUNNER_AVAILABLE or benchmark_runner is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark runner not available",
        )

    runs = benchmark_runner.history(
        limit=limit,
        profile=profile,
        model=model,
    )

    entries = []
    for run in runs:
        ms_dict = {}
        for m_name, ms in run.get("model_scores", {}).items():
            ms_dict[m_name] = BenchmarkV2ModelScore(
                model=ms.get("model", m_name),
                accuracy_avg=ms.get("accuracy_avg", 0.0),
                code_avg=ms.get("code_avg", 0.0),
                structure_avg=ms.get("structure_avg", 0.0),
                speed_avg=ms.get("speed_avg", 0.0),
                composite=ms.get("composite", 0.0),
                questions_evaluated=ms.get("questions_evaluated", 0),
            )
        entries.append(BenchmarkV2HistoryEntry(
            run_id=run.get("run_id", ""),
            profile=run.get("profile", ""),
            models=run.get("models", []),
            status=run.get("status", ""),
            started_at=run.get("started_at", 0.0),
            duration_ms=run.get("duration_ms", 0.0),
            weight_preset=run.get("weight_preset", "balanced"),
            custom_weights=run.get("custom_weights"),
            model_scores=ms_dict,
        ))

    return BenchmarkV2HistoryResponse(
        runs=entries,
        total=len(entries),
    )


# ---- Leaderboard ----

@router.get("/leaderboard", response_model=BenchmarkV2LeaderboardResponse)
def get_leaderboard(
    profile: str | None = Query(default=None, description="Filter by profile"),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    """Get ranked model leaderboard based on composite scores.

    Aggregates model performance from completed runs and ranks them.
    """
    if not BENCHMARK_RUNNER_AVAILABLE or benchmark_runner is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark runner not available",
        )

    result = benchmark_runner.compare(profile=profile, limit=limit)
    raw_models = result.get("models", [])

    entries = []
    for rank, m in enumerate(raw_models, 1):
        entries.append(BenchmarkV2LeaderboardEntry(
            rank=rank,
            model=m.get("model", ""),
            composite=m.get("avg_composite", 0.0),
            accuracy_avg=m.get("avg_accuracy", 0.0),
            code_avg=m.get("avg_code", 0.0),
            structure_avg=m.get("avg_structure", 0.0),
            speed_avg=m.get("avg_speed", 0.0),
            run_count=m.get("run_count", 0),
            last_run=m.get("last_run", 0.0),
        ))

    return BenchmarkV2LeaderboardResponse(
        profile=profile or "",
        entries=entries,
        total=len(entries),
    )


# ---- Head-to-Head ----

@router.get("/head-to-head", response_model=BenchmarkV2HeadToHeadResponse)
def head_to_head(
    model_a: str = Query(..., description="First model name"),
    model_b: str = Query(..., description="Second model name"),
    profile: str | None = Query(default=None, description="Filter by profile"),
) -> dict:
    """Side-by-side comparison of two models across all metrics.

    Compares aggregated scores and declares a winner per metric.
    """
    if not BENCHMARK_RUNNER_AVAILABLE or benchmark_runner is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark runner not available",
        )

    result = benchmark_runner.compare(
        models=[model_a, model_b],
        profile=profile,
    )
    raw_models = result.get("models", [])

    # Build lookup
    scores_map: dict[str, dict] = {}
    for m in raw_models:
        scores_map[m.get("model", "")] = m

    a_data = scores_map.get(model_a, {})
    b_data = scores_map.get(model_b, {})

    metric_keys = [
        ("accuracy", "avg_accuracy"),
        ("code", "avg_code"),
        ("structure", "avg_structure"),
        ("speed", "avg_speed"),
        ("composite", "avg_composite"),
    ]

    metrics = []
    a_wins = 0
    b_wins = 0
    ties = 0

    for display_name, db_key in metric_keys:
        a_val = a_data.get(db_key, 0.0)
        b_val = b_data.get(db_key, 0.0)

        if a_val > b_val:
            winner = model_a
            a_wins += 1
        elif b_val > a_val:
            winner = model_b
            b_wins += 1
        else:
            winner = "tie"
            ties += 1

        metrics.append(BenchmarkV2HeadToHeadMetric(
            metric=display_name,
            model_a_value=a_val,
            model_b_value=b_val,
            winner=winner,
        ))

    overall = model_a if a_wins > b_wins else (model_b if b_wins > a_wins else "tie")

    return BenchmarkV2HeadToHeadResponse(
        model_a=model_a,
        model_b=model_b,
        metrics=metrics,
        overall_winner=overall,
        model_a_wins=a_wins,
        model_b_wins=b_wins,
        ties=ties,
    )


# ---- Trends ----

@router.get("/trends", response_model=BenchmarkV2TrendResponse)
def get_trends(
    model: str = Query(..., description="Model name to track"),
    limit: int = Query(default=50, ge=1, le=200),
    profile: str | None = Query(default=None, description="Filter by profile"),
) -> dict:
    """Get temporal performance data for a model.

    Returns score data points over time and detects regressions.
    """
    if not BENCHMARK_RUNNER_AVAILABLE or benchmark_runner is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark runner not available",
        )

    runs = benchmark_runner.history(limit=limit, profile=profile, model=model)

    points = []
    for run in reversed(runs):
        ms = run.get("model_scores", {}).get(model)
        if not ms:
            continue
        points.append(BenchmarkV2TrendPoint(
            run_id=run.get("run_id", ""),
            timestamp=run.get("started_at", 0.0),
            composite=ms.get("composite", 0.0),
            accuracy=ms.get("accuracy_avg", 0.0),
            code=ms.get("code_avg", 0.0),
            structure=ms.get("structure_avg", 0.0),
            speed=ms.get("speed_avg", 0.0),
            profile=run.get("profile", ""),
        ))

    # Detect trend direction and regression
    trend_direction = "stable"
    regression_detected = False
    if len(points) >= 3:
        recent = [p.composite for p in points[-3:]]
        older = [p.composite for p in points[:3]]
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        if avg_recent > avg_older * 1.05:
            trend_direction = "improving"
        elif avg_recent < avg_older * 0.95:
            trend_direction = "declining"
            regression_detected = True

    return BenchmarkV2TrendResponse(
        model=model,
        points=points,
        trend_direction=trend_direction,
        regression_detected=regression_detected,
    )


# ---- Recommendations ----

@router.get("/recommendations", response_model=BenchmarkV2RecommendationsResponse)
def get_recommendations() -> dict:
    """Get current best-model suggestions based on benchmark data."""
    if not BENCHMARK_RECOMMENDATIONS_AVAILABLE or benchmark_recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark recommendations not available",
        )

    snapshot = benchmark_recommender.get_latest()
    if snapshot is None:
        # Try to generate from history
        try:
            snapshot = benchmark_recommender.generate_from_history()
        except Exception as e:
            logger.error("Failed to generate recommendations: %s", e)

    if snapshot is None:
        return BenchmarkV2RecommendationsResponse()

    entries = [
        BenchmarkV2RecommendationEntry(**r.to_dict())
        for r in snapshot.recommendations
    ]

    return BenchmarkV2RecommendationsResponse(
        snapshot_id=snapshot.snapshot_id,
        created_at=snapshot.created_at,
        profile=snapshot.profile,
        recommendations=entries,
        applied=snapshot.applied,
        applied_at=snapshot.applied_at,
    )


@router.post("/recommendations/apply", response_model=BenchmarkV2ApplyResponse)
def apply_recommendations() -> dict:
    """Apply current recommendations to smart router configuration.

    Sets the quality model as default and adjusts routing.
    """
    if not BENCHMARK_RECOMMENDATIONS_AVAILABLE or benchmark_recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark recommendations not available",
        )

    try:
        result = benchmark_recommender.apply_to_smart_router()
    except Exception as e:
        logger.error("Failed to apply recommendations: %s", e)
        return BenchmarkV2ApplyResponse(
            applied=False,
            error=str(e),
        )

    return BenchmarkV2ApplyResponse(
        applied=result.get("applied", False),
        snapshot_id=result.get("snapshot_id", ""),
        changes=result.get("changes", {}),
        error=result.get("error", ""),
    )


# ---- Export ----

@router.get("/export/{run_id}")
def export_results(
    run_id: str,
    format: str = Query(default="json", pattern="^(json|csv)$"),
) -> dict:
    """Export benchmark results as JSON or CSV.

    Returns the full run data in the requested format as a downloadable file.
    """
    if not BENCHMARK_RUNNER_AVAILABLE or benchmark_runner is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark runner not available",
        )

    data = benchmark_runner.get_results(run_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found",
        )

    if format == "csv":
        return _export_csv(run_id, data)
    return _export_json(run_id, data)


def _export_json(run_id: str, data: dict) -> JSONResponse:
    """Build JSON export response."""
    # Include judge scores if available
    j_scores: list[dict] = []
    if BENCHMARK_JUDGE_AVAILABLE and judge_store is not None:
        try:
            j_scores = judge_store.get_scores_for_run(run_id)
        except Exception:
            pass

    export_data = {
        "run_id": data.get("run_id", run_id),
        "profile": data.get("profile", ""),
        "models": data.get("models", []),
        "status": data.get("status", ""),
        "started_at": data.get("started_at", 0.0),
        "finished_at": data.get("finished_at", 0.0),
        "duration_ms": data.get("duration_ms", 0.0),
        "weight_preset": data.get("weight_preset", "balanced"),
        "model_scores": data.get("model_scores", {}),
        "question_results": data.get("question_results", {}),
        "judge_scores": j_scores,
    }

    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f'attachment; filename="benchmark_{run_id}.json"',
        },
    )


def _export_csv(run_id: str, data: dict) -> StreamingResponse:
    """Build CSV export as streaming response."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "run_id", "model", "question_id", "category",
        "accuracy_score", "code_score", "structure_score",
        "speed_score", "composite_score",
    ])

    # Rows from question results
    for model, qrs in data.get("question_results", {}).items():
        for qr in qrs:
            writer.writerow([
                run_id,
                model,
                qr.get("question_id", ""),
                qr.get("category", ""),
                qr.get("accuracy_score", 0.0),
                qr.get("code_score", 0.0),
                qr.get("structure_score", 0.0),
                qr.get("speed_score", 0.0),
                qr.get("composite_score", 0.0),
            ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="benchmark_{run_id}.csv"',
        },
    )


# =========================================================================
# Custom Profile CRUD
# =========================================================================

@router.get(
    "/profiles/custom",
    response_model=BenchmarkV2CustomProfilesListResponse,
)
def list_custom_profiles() -> dict:
    """List all user-defined custom benchmark profiles."""
    if not CUSTOM_PROFILES_AVAILABLE or custom_profile_store is None:
        return BenchmarkV2CustomProfilesListResponse()

    profiles = custom_profile_store.list_profiles()
    items = [
        BenchmarkV2CustomProfileResponse(**p.to_dict())
        for p in profiles
    ]
    return BenchmarkV2CustomProfilesListResponse(
        profiles=items,
        count=len(items),
    )


@router.post(
    "/profiles/custom",
    response_model=BenchmarkV2CustomProfileResponse,
    status_code=201,
)
def create_custom_profile(request: BenchmarkV2CustomProfileCreate) -> dict:
    """Create a new custom benchmark profile."""
    if not CUSTOM_PROFILES_AVAILABLE or custom_profile_store is None:
        raise HTTPException(
            status_code=503,
            detail="Custom profiles not available",
        )

    if not request.name.strip():
        raise HTTPException(
            status_code=400,
            detail="Profile name is required",
        )

    # Validate categories against available ones
    if BENCHMARK_V2_AVAILABLE and benchmark_evaluator is not None:
        valid_cats = set(benchmark_evaluator.available_categories)
        invalid = [c for c in request.categories if c not in valid_cats]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown categories: {', '.join(invalid)}",
            )

    # Validate custom weights if provided
    if request.custom_weights:
        required_keys = {"accuracy", "code", "structure", "speed"}
        missing = required_keys - set(request.custom_weights.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Custom weights missing keys: {', '.join(missing)}",
            )

    try:
        profile = custom_profile_store.create(
            name=request.name.strip(),
            description=request.description,
            categories=request.categories,
            weight_preset=request.weight_preset,
            custom_weights=request.custom_weights,
            timeout=request.timeout,
            max_response_tokens=request.max_response_tokens,
            expected_length_range=request.expected_length_range,
        )
        return BenchmarkV2CustomProfileResponse(**profile.to_dict())
    except ValueError as e:
        status = 409 if "already exists" in str(e) else 400
        raise HTTPException(status_code=status, detail=str(e))
    except Exception as e:
        logger.error("Failed to create custom profile: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create profile: {e}",
        )


@router.put(
    "/profiles/custom/{profile_id}",
    response_model=BenchmarkV2CustomProfileResponse,
)
def update_custom_profile(
    profile_id: str,
    request: BenchmarkV2CustomProfileUpdate,
) -> dict:
    """Update an existing custom benchmark profile."""
    if not CUSTOM_PROFILES_AVAILABLE or custom_profile_store is None:
        raise HTTPException(
            status_code=503,
            detail="Custom profiles not available",
        )

    updates = {
        k: v for k, v in request.model_dump().items() if v is not None
    }
    if not updates:
        raise HTTPException(
            status_code=400,
            detail="No fields to update",
        )

    # Validate categories if being updated
    if "categories" in updates and BENCHMARK_V2_AVAILABLE and benchmark_evaluator is not None:
        valid_cats = set(benchmark_evaluator.available_categories)
        invalid = [c for c in updates["categories"] if c not in valid_cats]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown categories: {', '.join(invalid)}",
            )

    try:
        profile = custom_profile_store.update(profile_id, updates)
    except ValueError as e:
        status = 409 if "already exists" in str(e) else 400
        raise HTTPException(status_code=status, detail=str(e))

    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"Custom profile '{profile_id}' not found",
        )

    return BenchmarkV2CustomProfileResponse(**profile.to_dict())


@router.delete("/profiles/custom/{profile_id}")
def delete_custom_profile(profile_id: str) -> dict:
    """Delete a custom benchmark profile."""
    if not CUSTOM_PROFILES_AVAILABLE or custom_profile_store is None:
        raise HTTPException(
            status_code=503,
            detail="Custom profiles not available",
        )

    deleted = custom_profile_store.delete(profile_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Custom profile '{profile_id}' not found",
        )

    return {"profile_id": profile_id, "deleted": True}


@router.post(
    "/profiles/preview",
    response_model=BenchmarkV2QuestionPreviewResponse,
)
def preview_profile_questions(categories: list[str]) -> dict:
    """Preview which questions will be included for given categories.

    Accepts a JSON array of category names in the request body.
    Returns counts per category and total question count.
    """
    if not CUSTOM_PROFILES_AVAILABLE or custom_profile_store is None:
        raise HTTPException(
            status_code=503,
            detail="Custom profiles not available",
        )

    questions = None
    if BENCHMARK_V2_AVAILABLE and benchmark_evaluator is not None:
        questions = benchmark_evaluator.questions

    result = custom_profile_store.get_question_preview(
        categories, questions,
    )
    return BenchmarkV2QuestionPreviewResponse(**result)


# =========================================================================
# Auto-Trigger Management
# =========================================================================

@router.get(
    "/auto-trigger/status",
    response_model=BenchmarkV2AutoTriggerStatusResponse,
)
def get_auto_trigger_status() -> dict:
    """Get current status of the auto-trigger system."""
    if not AUTO_TRIGGER_AVAILABLE or auto_trigger is None:
        return BenchmarkV2AutoTriggerStatusResponse()

    return BenchmarkV2AutoTriggerStatusResponse(**auto_trigger.status)


@router.get(
    "/auto-trigger/config",
    response_model=BenchmarkV2AutoTriggerConfigResponse,
)
def get_auto_trigger_config() -> dict:
    """Get current auto-trigger configuration."""
    if not AUTO_TRIGGER_AVAILABLE or auto_trigger is None:
        return BenchmarkV2AutoTriggerConfigResponse()

    return BenchmarkV2AutoTriggerConfigResponse(**auto_trigger.config)


@router.put(
    "/auto-trigger/config",
    response_model=BenchmarkV2AutoTriggerConfigResponse,
)
def update_auto_trigger_config(request: BenchmarkV2AutoTriggerConfigUpdate) -> dict:
    """Update auto-trigger configuration.

    Only provided fields are updated. Set enabled=true to start
    the polling thread, enabled=false to stop it.
    """
    if not AUTO_TRIGGER_AVAILABLE or auto_trigger is None:
        raise HTTPException(
            status_code=503,
            detail="Auto-trigger not available",
        )

    updates = {
        k: v for k, v in request.model_dump().items() if v is not None
    }
    if not updates:
        raise HTTPException(
            status_code=400,
            detail="No fields to update",
        )

    result = auto_trigger.update_config(updates)
    return BenchmarkV2AutoTriggerConfigResponse(**result)


@router.post("/auto-trigger/enable")
def enable_auto_trigger() -> dict:
    """Enable auto-trigger and start the polling thread.

    WARNING: Benchmarks will run automatically when new models are
    detected. This uses significant GPU/RAM resources.
    """
    if not AUTO_TRIGGER_AVAILABLE or auto_trigger is None:
        raise HTTPException(
            status_code=503,
            detail="Auto-trigger not available",
        )

    auto_trigger.enable()
    return {"enabled": True, "running": auto_trigger.running}


@router.post("/auto-trigger/disable")
def disable_auto_trigger() -> dict:
    """Disable auto-trigger and stop the polling thread."""
    if not AUTO_TRIGGER_AVAILABLE or auto_trigger is None:
        raise HTTPException(
            status_code=503,
            detail="Auto-trigger not available",
        )

    auto_trigger.disable()
    return {"enabled": False, "running": False}


@router.get(
    "/auto-trigger/events",
    response_model=BenchmarkV2AutoTriggerEventsResponse,
)
def get_auto_trigger_events() -> dict:
    """Get recent auto-trigger events (triggers, skips, errors)."""
    if not AUTO_TRIGGER_AVAILABLE or auto_trigger is None:
        return BenchmarkV2AutoTriggerEventsResponse()

    events = auto_trigger.events
    items = [
        BenchmarkV2AutoTriggerEventResponse(**e)
        for e in events
    ]
    return BenchmarkV2AutoTriggerEventsResponse(
        events=items,
        count=len(items),
    )


@router.post(
    "/auto-trigger/test-poll",
    response_model=BenchmarkV2AutoTriggerTestPollResponse,
)
def test_poll_auto_trigger() -> dict:
    """Run a single poll without triggering any benchmark.

    Verifies Ollama connectivity and shows whether model changes
    would be detected. Does not modify the stored snapshot.
    """
    if not AUTO_TRIGGER_AVAILABLE or auto_trigger is None:
        raise HTTPException(
            status_code=503,
            detail="Auto-trigger not available",
        )

    result = auto_trigger.test_poll()
    return BenchmarkV2AutoTriggerTestPollResponse(**result)


@router.post("/auto-trigger/reset")
def reset_auto_trigger_snapshot() -> dict:
    """Reset the known model snapshot.

    Useful after manual model management — the next poll will
    re-baseline without triggering.
    """
    if not AUTO_TRIGGER_AVAILABLE or auto_trigger is None:
        raise HTTPException(
            status_code=503,
            detail="Auto-trigger not available",
        )

    auto_trigger.reset_snapshot()
    return {"reset": True}
