#!/usr/bin/env python3
"""
API routes for the health dashboard and benchmarks.

Provides an aggregated dashboard endpoint (modules, cache, conversations,
memory) and performance benchmark endpoints.
"""

import logging

from fastapi import APIRouter, HTTPException

from .deps import (
    ARTIFACT_AVAILABLE,
    BENCHMARK_AVAILABLE,
    CODE_EXECUTOR_AVAILABLE,
    CONTEXT_WINDOW_AVAILABLE,
    CONVERSATION_AVAILABLE,
    EXECUTOR_AVAILABLE,
    MEMORY_AVAILABLE,
    MODEL_WARMUP_AVAILABLE,
    PIPELINE_AVAILABLE,
    PRESET_AVAILABLE,
    RESPONSE_CACHE_AVAILABLE,
    SEMANTIC_CACHE_AVAILABLE,
    perf_benchmark_runner,
    conversation_manager,
    executor,
    memory_manager,
    model_warmup,
    response_cache,
)
from .schemas import (
    BenchmarkResultSchema,
    CacheStatsSchema,
    HealthDashboard,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("/dashboard", response_model=HealthDashboard)
def health_dashboard() -> dict:
    """Full health dashboard with aggregated data."""
    dashboard = HealthDashboard(
        modules={
            "conversation": CONVERSATION_AVAILABLE,
            "presets": PRESET_AVAILABLE,
            "memory": MEMORY_AVAILABLE,
            "artifacts": ARTIFACT_AVAILABLE,
            "code_executor": CODE_EXECUTOR_AVAILABLE,
            "response_cache": RESPONSE_CACHE_AVAILABLE,
            "semantic_cache": SEMANTIC_CACHE_AVAILABLE,
            "pipelines": PIPELINE_AVAILABLE,
            "benchmarks": BENCHMARK_AVAILABLE,
            "model_warmup": MODEL_WARMUP_AVAILABLE,
            "context_window": CONTEXT_WINDOW_AVAILABLE,
        },
    )

    # Conversation count
    if CONVERSATION_AVAILABLE and conversation_manager is not None:
        try:
            convs = conversation_manager.list_conversations()
            dashboard.conversation_count = len(convs)
        except Exception as e:
            logger.debug(f"Could not count conversations: {e}")

    # Memory fact count
    if MEMORY_AVAILABLE and memory_manager is not None:
        try:
            dashboard.memory_fact_count = memory_manager.count_facts()
        except Exception as e:
            logger.debug(f"Could not count memory facts: {e}")

    # Cache statistics
    if RESPONSE_CACHE_AVAILABLE and response_cache is not None:
        try:
            stats = response_cache.get_stats()
            dashboard.cache_stats = CacheStatsSchema(
                total_entries=stats.total_entries,
                total_hits=stats.total_hits,
                total_misses=stats.total_misses,
                hit_rate=stats.hit_rate,
                entries_by_model=stats.entries_by_model,
                oldest_entry=stats.oldest_entry,
                total_size_bytes=stats.total_size_bytes,
            )
        except Exception as e:
            logger.debug(f"Could not get cache stats: {e}")

    # Model warmup status
    if MODEL_WARMUP_AVAILABLE and model_warmup is not None:
        try:
            dashboard.warmup_status = {
                "warmed_models": list(model_warmup.warmed_models),
                "is_warming": model_warmup.is_warming,
            }
        except Exception as e:
            logger.debug(f"Could not get warmup status: {e}")

    # Context health (S47)
    context_health_info = {"available": CONTEXT_WINDOW_AVAILABLE}
    if EXECUTOR_AVAILABLE and executor is not None:
        try:
            window_stats = executor.last_window_stats
            if window_stats:
                context_health_info["last_window_stats"] = window_stats
                context_health_info["trimming_active"] = window_stats.get("dropped", 0) > 0
        except Exception as e:
            logger.debug(f"Could not get context health: {e}")
    dashboard.context_health = context_health_info

    return dashboard


@router.post("/benchmarks")
def run_all_benchmarks(iterations: int = 200) -> dict:
    """Execute tous les benchmarks disponibles."""
    if not BENCHMARK_AVAILABLE or perf_benchmark_runner is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark module not available",
        )

    try:
        suite = perf_benchmark_runner.run_all(iterations=iterations)
        results = {}
        for name, result in suite.results.items():
            results[name] = BenchmarkResultSchema(
                name=result.name,
                iterations=result.iterations,
                total_time_ms=result.total_time_ms,
                mean_ms=result.mean_ms,
                median_ms=result.median_ms,
                min_ms=result.min_ms,
                max_ms=result.max_ms,
                stddev_ms=result.stddev_ms,
                p95_ms=result.p95_ms,
                p99_ms=result.p99_ms,
                throughput_ops=result.throughput_ops,
                error=result.error,
            )
        return {
            "timestamp": suite.timestamp,
            "version": suite.version,
            "total_time_ms": suite.total_time_ms,
            "results": results,
        }
    except Exception as e:
        logger.error(f"Benchmark suite error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmarks/{name}", response_model=BenchmarkResultSchema)
def run_single_benchmark(name: str, iterations: int = 200) -> dict:
    """Execute un benchmark specifique par nom."""
    if not BENCHMARK_AVAILABLE or perf_benchmark_runner is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark module not available",
        )

    try:
        result = perf_benchmark_runner.run(name, iterations=iterations)
        if result.error and "Unknown benchmark" in result.error:
            raise HTTPException(status_code=404, detail=result.error)
        return BenchmarkResultSchema(
            name=result.name,
            iterations=result.iterations,
            total_time_ms=result.total_time_ms,
            mean_ms=result.mean_ms,
            median_ms=result.median_ms,
            min_ms=result.min_ms,
            max_ms=result.max_ms,
            stddev_ms=result.stddev_ms,
            p95_ms=result.p95_ms,
            p99_ms=result.p99_ms,
            throughput_ops=result.throughput_ops,
            error=result.error,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Benchmark {name} error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
