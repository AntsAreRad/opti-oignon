#!/usr/bin/env python3
"""
API routes for Benchmark Dashboard (S60).

Provides endpoints for:
- LLM benchmark execution with live WebSocket progress
- Benchmark history (list, detail, compare, delete)
- Model configuration read/write
- Performance benchmark passthrough

The LLM benchmark runs in a background thread. Only one benchmark
can run at a time. Progress is emitted via WebSocket.
"""

import asyncio
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

# S215: emergency-stop admission guard (a stopped system refuses honestly)
try:
    from opti_oignon import emergency_stop as _emergency_stop
except Exception:
    _emergency_stop = None

logger = logging.getLogger(__name__)

# Conditional imports
try:
    from opti_oignon.benchmark_history import (
        BenchmarkResultRecord,
        BenchmarkRunRecord,
        benchmark_history,
    )
    HISTORY_AVAILABLE = True
except ImportError:
    HISTORY_AVAILABLE = False
    benchmark_history = None

try:
    import yaml as _yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# S193 BMK-02: per-timeout cached Ollama clients so the request timeout is
# enforced at the transport level (it previously only classified errors
# after an unbounded blocking call).
_V1_CLIENTS: dict[int, Any] = {}
_V1_CLIENTS_LOCK = threading.Lock()


def _get_v1_client(timeout: int) -> Any:
    """Return a cached ollama.Client bound to the given timeout."""
    with _V1_CLIENTS_LOCK:
        client = _V1_CLIENTS.get(timeout)
        if client is None:
            client = _ollama.Client(timeout=timeout)
            _V1_CLIENTS[timeout] = client
        return client

# Config paths
CONFIG_DIR = Path(__file__).parent.parent / "config"
DATA_DIR = Path(__file__).parent.parent / "data"
BENCHMARK_CONFIG_PATH = CONFIG_DIR / "benchmark.yaml"
MODELS_CONFIG_PATH = CONFIG_DIR / "models.yaml"

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

# S171: RFC 6455 WebSocket close codes for graceful server-side shutdown.
WS_CLOSE_INTERNAL_ERROR = 1011


# =============================================================================
# BENCHMARK CONFIG LOADER
# =============================================================================

def _load_benchmark_config() -> dict[str, Any]:
    """Load benchmark configuration from YAML."""
    if not YAML_AVAILABLE:
        return {"suites": {}, "tasks": {}, "runner": {}, "scoring": {}}
    try:
        with open(BENCHMARK_CONFIG_PATH, encoding="utf-8") as f:
            return _yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to load benchmark config: %s", e)
        return {"suites": {}, "tasks": {}, "runner": {}, "scoring": {}}


def _load_models_config() -> dict[str, Any]:
    """Load models configuration from YAML."""
    if not YAML_AVAILABLE:
        return {}
    try:
        with open(MODELS_CONFIG_PATH, encoding="utf-8") as f:
            return _yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to load models config: %s", e)
        return {}


def _save_models_config(config: dict[str, Any]) -> bool:
    """Save models configuration to YAML."""
    if not YAML_AVAILABLE:
        return False
    try:
        with open(MODELS_CONFIG_PATH, "w", encoding="utf-8") as f:
            _yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        logger.error("Failed to save models config: %s", e)
        return False


def _get_installed_models() -> list[str]:
    """Get list of installed Ollama models."""
    if not OLLAMA_AVAILABLE:
        return []
    try:
        response = _ollama.list()
        if hasattr(response, "models"):
            return [m.model for m in (response.models or [])]
        if isinstance(response, dict):
            return [m.get("name", "") for m in response.get("models", [])]
        return []
    except Exception as e:
        logger.debug("Cannot list Ollama models: %s", e)
        return []


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class BenchmarkRunRequest(BaseModel):
    """Request body for starting a benchmark run."""
    models: list[str] = Field(default_factory=list, description="Models to test (empty = all installed)")
    tasks: list[str] = Field(default_factory=list, description="Tasks to run (empty = all)")
    suite_id: str = Field(default="", description="Suite ID to use (overrides tasks)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout: int = Field(default=300, ge=30, le=1800)
    max_tokens: int = Field(default=1000, ge=100, le=4096)


class BenchmarkRunSummary(BaseModel):
    """Summary of a benchmark run."""
    id: str
    run_type: str = "llm"
    started_at: str = ""
    completed_at: str = ""
    status: str = "running"
    models_tested: list[str] = Field(default_factory=list)
    tasks_tested: list[str] = Field(default_factory=list)
    total_tests: int = 0
    avg_score: float | None = None
    best_model: str | None = None
    duration_sec: float | None = None


class BenchmarkResultItem(BaseModel):
    """Individual test result."""
    model: str = ""
    task: str = ""
    task_name: str = ""
    category: str = ""
    score: float = 0.0
    auto_score: float = 0.0
    user_score: float | None = None
    time_seconds: float = 0.0
    status: str = "success"
    response_preview: str = ""
    keywords_found: list[str] = Field(default_factory=list)
    keywords_missing: list[str] = Field(default_factory=list)
    error_message: str | None = None


class BenchmarkRunDetail(BaseModel):
    """Full detail of a benchmark run."""
    id: str
    run_type: str = "llm"
    started_at: str = ""
    completed_at: str = ""
    status: str = ""
    models: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    total_tests: int = 0
    avg_score: float | None = None
    best_model: str | None = None
    duration_sec: float | None = None
    results: list[BenchmarkResultItem] = Field(default_factory=list)
    global_ranking: list[dict[str, Any]] = Field(default_factory=list)
    best_by_category: dict[str, str] = Field(default_factory=dict)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ModelConfigUpdate(BaseModel):
    """Request body for updating model config."""
    config: dict[str, Any] = Field(default_factory=dict)


class ModelRoleUpdate(BaseModel):
    """Request body for updating a single role assignment."""
    primary: str = ""
    fast: str = ""
    quality: str = ""


class UserScoreRequest(BaseModel):
    """Request to submit a user score for a result."""
    run_id: str
    model: str
    task: str
    score: float = Field(ge=0.0, le=10.0)


# =============================================================================
# RUN STATE MANAGEMENT
# =============================================================================

class _RunState:
    """Thread-safe state for the current benchmark run."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current_run_id: str | None = None
        self.status: str = "idle"
        self.cancel_requested: bool = False
        self.progress: dict[str, Any] = {}
        self.ws_clients: list[
            tuple[asyncio.Queue, asyncio.AbstractEventLoop | None]
        ] = []
        self.ws_lock = threading.Lock()

    def is_running(self) -> bool:
        with self.lock:
            return self.status == "running"

    def start(self, run_id: str) -> None:
        with self.lock:
            self.current_run_id = run_id
            self.status = "running"
            self.cancel_requested = False
            self.progress = {}

    def finish(self, status: str = "completed") -> None:
        with self.lock:
            self.status = status
            self.cancel_requested = False

    def request_cancel(self) -> None:
        with self.lock:
            self.cancel_requested = True

    def is_cancelled(self) -> bool:
        with self.lock:
            return self.cancel_requested

    def add_ws_client(
        self,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        # S193 BMK-06: keep the client's event loop alongside its queue so
        # the worker thread can schedule puts on the right loop.
        with self.ws_lock:
            self.ws_clients.append((queue, loop))

    def remove_ws_client(self, queue: asyncio.Queue) -> None:
        with self.ws_lock:
            self.ws_clients = [
                (q, l) for (q, l) in self.ws_clients if q is not queue
            ]

    def broadcast(self, event: dict[str, Any]) -> None:
        """Send event to all connected WebSocket clients.

        S193 BMK-06: broadcast is called from the benchmark worker thread;
        asyncio queues are not thread-safe, so hand each put over to the
        client's event loop instead of mutating the queue cross-thread
        (the previous direct put_nowait could race and silently drop
        events, including the terminal one).
        """
        with self.ws_lock:
            clients = list(self.ws_clients)
        for q, loop in clients:
            try:
                if loop is not None:
                    loop.call_soon_threadsafe(q.put_nowait, event)
                else:
                    q.put_nowait(event)
            except Exception:
                pass


_state = _RunState()


# =============================================================================
# BENCHMARK RUNNER (BACKGROUND THREAD)
# =============================================================================

def _run_benchmark_thread(
    run_id: str,
    models: list[str],
    tasks: dict[str, dict[str, Any]],
    task_ids: list[str],
    temperature: float,
    timeout: int,
    max_tokens: int,
    scoring_config: dict[str, Any],
):
    """Execute benchmark in a background thread.

    Iterates over models x tasks, calls Ollama, scores results,
    persists to history, and broadcasts progress via WebSocket queues.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    total_tests = len(models) * len(task_ids)
    completed = 0
    start_time = time.time()
    results: list[dict[str, Any]] = []

    # Create the run record
    run_record = BenchmarkRunRecord(
        id=run_id,
        run_type="llm",
        started_at=started_at,
        status="running",
        models=models,
        tasks=task_ids,
        total_tests=total_tests,
        config_snapshot={"temperature": temperature, "timeout": timeout},
    )

    if HISTORY_AVAILABLE and benchmark_history:
        benchmark_history.save_run(run_record)

    try:
        for model in models:
            if _state.is_cancelled():
                break

            for task_id in task_ids:
                if _state.is_cancelled():
                    break

                task = tasks.get(task_id)
                if not task:
                    completed += 1
                    continue

                # Broadcast progress
                elapsed = time.time() - start_time
                avg_per_test = elapsed / max(completed, 1)
                remaining = avg_per_test * (total_tests - completed)

                _state.broadcast({
                    "type": "progress",
                    "data": {
                        "total_tests": total_tests,
                        "completed_tests": completed,
                        "current_model": model,
                        "current_task": task_id,
                        "current_task_name": task.get("name", task_id),
                        "percent": round((completed / total_tests) * 100, 1),
                        "elapsed_sec": int(elapsed),
                        "estimated_remaining_sec": int(remaining),
                    },
                })

                # Execute the test
                result = _execute_single_test(
                    model=model,
                    task_id=task_id,
                    task=task,
                    temperature=temperature,
                    timeout=timeout,
                    max_tokens=max_tokens,
                    scoring_config=scoring_config,
                )

                # Persist result
                result_record = BenchmarkResultRecord(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    model=model,
                    task=task_id,
                    task_name=task.get("name", task_id),
                    category=task.get("category", "general"),
                    score=result["score"],
                    auto_score=result["auto_score"],
                    user_score=result.get("user_score"),
                    time_seconds=result["time_seconds"],
                    status=result["status"],
                    response_preview=result["response_preview"],
                    keywords_found=result.get("keywords_found", []),
                    keywords_missing=result.get("keywords_missing", []),
                    error_message=result.get("error_message"),
                )
                if HISTORY_AVAILABLE and benchmark_history:
                    benchmark_history.save_result(result_record)

                results.append(result)
                completed += 1

                # Broadcast result
                _state.broadcast({
                    "type": "result",
                    "data": result,
                })

        # Finalize
        end_time = time.time()
        duration = end_time - start_time
        completed_at = datetime.now(timezone.utc).isoformat()

        # Calculate summary
        success_scores = [
            r["score"] for r in results if r["status"] == "success"
        ]
        avg_score = (
            round(sum(success_scores) / len(success_scores), 2)
            if success_scores else None
        )

        # Find best model
        model_avgs: dict[str, list[float]] = {}
        for r in results:
            if r["status"] == "success":
                model_avgs.setdefault(r["model"], []).append(r["score"])
        best_model = None
        if model_avgs:
            best_model = max(
                model_avgs,
                key=lambda m: sum(model_avgs[m]) / len(model_avgs[m]),
            )

        final_status = "cancelled" if _state.is_cancelled() else "completed"

        # Update run record
        run_record.completed_at = completed_at
        run_record.status = final_status
        run_record.avg_score = avg_score
        run_record.best_model = best_model
        run_record.duration_sec = round(duration, 2)

        if HISTORY_AVAILABLE and benchmark_history:
            benchmark_history.save_run(run_record)

        # Broadcast completion
        _state.broadcast({
            "type": "completed" if final_status == "completed" else "cancelled",
            "data": {
                "run_id": run_id,
                "status": final_status,
                "avg_score": avg_score,
                "best_model": best_model,
                "duration_sec": round(duration, 2),
                "total_completed": completed,
            },
        })

    except Exception as e:
        logger.error("Benchmark run %s failed: %s", run_id, e, exc_info=True)
        run_record.status = "error"
        run_record.error = str(e)[:500]
        run_record.completed_at = datetime.now(timezone.utc).isoformat()
        run_record.duration_sec = round(time.time() - start_time, 2)

        if HISTORY_AVAILABLE and benchmark_history:
            benchmark_history.save_run(run_record)

        _state.broadcast({
            "type": "error",
            "data": {"message": str(e)[:500], "run_id": run_id},
        })

    finally:
        _state.finish(run_record.status)


def _execute_single_test(
    model: str,
    task_id: str,
    task: dict[str, Any],
    temperature: float,
    timeout: int,
    max_tokens: int,
    scoring_config: dict[str, Any],
) -> dict[str, Any]:
    """Execute a single benchmark test against a model.

    Returns a result dict with score, timing, status, etc.
    """
    prompt = task.get("prompt", "")
    expected_keywords = task.get("expected_keywords", [])
    max_expected_time = task.get("max_expected_time", timeout)
    category = task.get("category", "general")

    start = time.time()

    try:
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama not available")

        response = _get_v1_client(timeout).generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        elapsed = time.time() - start
        response_text = ""
        if isinstance(response, dict):
            response_text = response.get("response", "")
        elif hasattr(response, "response"):
            response_text = response.response or ""

        # Check for refusal
        if _is_refusal(response_text):
            return {
                "model": model,
                "task": task_id,
                "task_name": task.get("name", task_id),
                "category": category,
                "score": 2.0,
                "auto_score": 2.0,
                "time_seconds": round(elapsed, 2),
                "status": "refused",
                "response_preview": response_text[:500],
                "keywords_found": [],
                "keywords_missing": expected_keywords,
                "error_message": None,
            }

        # Score
        auto_score, found, missing = _calculate_score(
            response_text, expected_keywords
        )

        # Time penalty
        penalty_factor = scoring_config.get("time_penalty_factor", 2.0)
        penalty_points = scoring_config.get("time_penalty_points", 2)
        if elapsed > max_expected_time * penalty_factor:
            auto_score = max(0, auto_score - penalty_points)

        return {
            "model": model,
            "task": task_id,
            "task_name": task.get("name", task_id),
            "category": category,
            "score": float(auto_score),
            "auto_score": float(auto_score),
            "time_seconds": round(elapsed, 2),
            "status": "success",
            "response_preview": response_text[:500],
            "keywords_found": found,
            "keywords_missing": missing,
            "error_message": None,
        }

    except Exception as e:
        elapsed = time.time() - start
        error_msg = str(e)[:200]
        status = "timeout" if (
            "timeout" in error_msg.lower() or elapsed > timeout
        ) else "error"

        return {
            "model": model,
            "task": task_id,
            "task_name": task.get("name", task_id),
            "category": category,
            "score": 0.0,
            "auto_score": 0.0,
            "time_seconds": round(elapsed, 2),
            "status": status,
            "response_preview": "",
            "keywords_found": [],
            "keywords_missing": expected_keywords,
            "error_message": error_msg,
        }


def _is_refusal(response: str) -> bool:
    """Detect if model refused to answer."""
    patterns = [
        r"I'm sorry",
        r"I cannot",
        r"I am not able",
        r"as an AI",
        r"my expertise is",
        r"I don't have",
    ]
    return any(re.search(p, response, re.IGNORECASE) for p in patterns)


def _calculate_score(
    response: str,
    expected_keywords: list[str],
) -> tuple:
    """Calculate score based on keyword matching.

    Returns (score, keywords_found, keywords_missing).
    """
    response_lower = response.lower()
    found = [k for k in expected_keywords if k.lower() in response_lower]
    missing = [k for k in expected_keywords if k.lower() not in response_lower]

    if not expected_keywords:
        if len(response) > 200:
            return 7, [], []
        elif len(response) > 50:
            return 5, [], []
        return 3, [], []

    ratio = len(found) / len(expected_keywords)
    score = int(ratio * 10)

    # Bonus for complete response
    if ratio >= 0.8 and len(response) > 300:
        score = min(10, score + 1)

    # Penalty for very short response
    if len(response) < 100:
        score = max(0, score - 2)

    return score, found, missing


# =============================================================================
# LLM BENCHMARK ENDPOINTS
# =============================================================================

@router.get("/suites")
async def list_suites() -> dict:
    """List available benchmark suites."""
    config = _load_benchmark_config()
    suites = config.get("suites", {})
    tasks = config.get("tasks", {})

    result = []
    for suite_id, suite in suites.items():
        task_list = suite.get("tasks", [])
        result.append({
            "id": suite_id,
            "name": suite.get("name", suite_id),
            "description": suite.get("description", ""),
            "task_count": len(task_list),
            "tasks": task_list,
            "categories": list(set(
                tasks.get(t, {}).get("category", "general")
                for t in task_list if t in tasks
            )),
        })
    return {"suites": result}


@router.get("/suites/{suite_id}")
async def get_suite_detail(suite_id: str) -> dict:
    """Get detailed information about a benchmark suite."""
    config = _load_benchmark_config()
    suites = config.get("suites", {})
    tasks = config.get("tasks", {})

    if suite_id not in suites:
        raise HTTPException(status_code=404, detail=f"Suite '{suite_id}' not found")

    suite = suites[suite_id]
    task_details = []
    for tid in suite.get("tasks", []):
        t = tasks.get(tid, {})
        task_details.append({
            "id": tid,
            "name": t.get("name", tid),
            "description": t.get("description", ""),
            "category": t.get("category", "general"),
            "prompt": t.get("prompt", ""),
            "expected_keywords": t.get("expected_keywords", []),
            "max_expected_time": t.get("max_expected_time", 300),
            "scoring_method": t.get("scoring_method", "keywords"),
        })

    return {
        "id": suite_id,
        "name": suite.get("name", suite_id),
        "description": suite.get("description", ""),
        "tasks": task_details,
    }


@router.get("/tasks")
async def list_tasks() -> dict:
    """List all available benchmark tasks."""
    config = _load_benchmark_config()
    tasks = config.get("tasks", {})
    result = []
    for tid, t in tasks.items():
        result.append({
            "id": tid,
            "name": t.get("name", tid),
            "description": t.get("description", ""),
            "category": t.get("category", "general"),
            "max_expected_time": t.get("max_expected_time", 300),
            "scoring_method": t.get("scoring_method", "keywords"),
        })
    return {"tasks": result}


@router.post("/llm/run")
async def start_llm_benchmark(request: BenchmarkRunRequest) -> dict:
    """Start an LLM benchmark run in the background.

    Only one benchmark can run at a time. Returns the run ID
    for tracking via WebSocket or polling.
    """
    if _emergency_stop is not None:
        _emergency_stop.guard_http()  # S215: refused, not hung
    if _state.is_running():
        raise HTTPException(
            status_code=409,
            detail="A benchmark is already running. Cancel it first or wait.",
        )

    # Resolve models
    models = request.models
    if not models:
        models = _get_installed_models()
    if not models:
        raise HTTPException(
            status_code=400,
            detail="No models specified and no Ollama models available.",
        )

    # Resolve tasks
    config = _load_benchmark_config()
    all_tasks = config.get("tasks", {})
    scoring_config = config.get("scoring", {})

    task_ids: list[str] = []
    if request.suite_id:
        suites = config.get("suites", {})
        suite = suites.get(request.suite_id)
        if not suite:
            raise HTTPException(
                status_code=404,
                detail=f"Suite '{request.suite_id}' not found",
            )
        task_ids = suite.get("tasks", [])
    elif request.tasks:
        task_ids = [t for t in request.tasks if t in all_tasks]
    else:
        task_ids = list(all_tasks.keys())

    if not task_ids:
        raise HTTPException(
            status_code=400,
            detail="No valid tasks to run.",
        )

    # Create run
    run_id = str(uuid.uuid4())
    _state.start(run_id)

    # Launch background thread
    thread = threading.Thread(
        target=_run_benchmark_thread,
        args=(
            run_id, models, all_tasks, task_ids,
            request.temperature, request.timeout,
            request.max_tokens, scoring_config,
        ),
        daemon=True,
        name=f"benchmark-{run_id[:8]}",
    )
    thread.start()

    return {
        "run_id": run_id,
        "status": "running",
        "models": models,
        "tasks": task_ids,
        "total_tests": len(models) * len(task_ids),
    }


@router.get("/llm/status")
async def get_llm_status() -> dict:
    """Get current benchmark run status."""
    return {
        "running": _state.is_running(),
        "run_id": _state.current_run_id,
        "status": _state.status,
    }


@router.post("/llm/cancel")
async def cancel_llm_benchmark() -> dict:
    """Request cancellation of the current benchmark run."""
    if not _state.is_running():
        raise HTTPException(status_code=409, detail="No benchmark is running.")
    _state.request_cancel()
    return {"status": "cancel_requested", "run_id": _state.current_run_id}


@router.post("/llm/user-score")
async def submit_user_score(request: UserScoreRequest) -> dict:
    """Submit a user score for a benchmark result."""
    if not HISTORY_AVAILABLE or not benchmark_history:
        raise HTTPException(status_code=503, detail="Benchmark history not available")

    # Find and update the result
    detail = benchmark_history.get_run_detail(request.run_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Run not found")

    conn = benchmark_history._get_conn()
    try:
        # Load scoring config for weighted average
        config = _load_benchmark_config()
        scoring = config.get("scoring", {})
        user_w = scoring.get("user_weight", 0.6)
        auto_w = scoring.get("auto_weight", 0.4)

        cursor = conn.execute(
            """SELECT id, auto_score FROM benchmark_results
               WHERE run_id = ? AND model = ? AND task = ?""",
            (request.run_id, request.model, request.task),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Result not found")

        auto_score = row["auto_score"]
        final_score = round(user_w * request.score + auto_w * auto_score, 2)

        conn.execute(
            """UPDATE benchmark_results
               SET user_score = ?, score = ?
               WHERE id = ?""",
            (request.score, final_score, row["id"]),
        )
        conn.commit()
        return {"status": "ok", "final_score": final_score}
    finally:
        conn.close()


# =============================================================================
# WEBSOCKET PROGRESS
# =============================================================================

@router.websocket("/llm/progress")
async def benchmark_progress_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for live benchmark progress.

    S136 audit fix: authenticates before processing.
    """
    await websocket.accept()

    # S136 audit fix: authenticate WebSocket connection
    try:
        from .routes_auth import authenticate_websocket
        user = await authenticate_websocket(websocket)
        if user is None:
            await websocket.send_json({"type": "error", "data": {"detail": "Authentication required"}})
            await websocket.close(code=4001)
            return
    except Exception:
        await websocket.send_json({"type": "error", "data": {"detail": "Authentication failed"}})
        await websocket.close(code=4001)
        return

    queue: asyncio.Queue = asyncio.Queue()
    # S193 BMK-06: capture this connection's loop so the worker thread
    # can schedule queue puts thread-safely.
    _state.add_ws_client(queue, asyncio.get_running_loop())

    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "data": {
                "running": _state.is_running(),
                "run_id": _state.current_run_id,
            },
        })

        # Forward events from queue to WebSocket
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(event)
                # Stop after terminal events
                if event.get("type") in ("completed", "cancelled", "error"):
                    break
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WebSocket benchmark error: %s", e)
        try:
            await websocket.close(code=WS_CLOSE_INTERNAL_ERROR)
        except Exception:
            pass
    finally:
        _state.remove_ws_client(queue)
        try:
            await websocket.close()
        except Exception:
            pass


# =============================================================================
# HISTORY ENDPOINTS
# =============================================================================

@router.get("/runs")
async def list_runs(
    run_type: str = Query(default="llm", description="Run type: llm or perf"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """List past benchmark runs."""
    if not HISTORY_AVAILABLE or not benchmark_history:
        raise HTTPException(status_code=503, detail="Benchmark history not available")

    runs = benchmark_history.get_runs(run_type=run_type, limit=limit, offset=offset)
    total = benchmark_history.get_run_count(run_type=run_type)
    return {"runs": runs, "total": total, "limit": limit, "offset": offset}


@router.get("/runs/{run_id}")
async def get_run_detail(run_id: str) -> dict:
    """Get full detail for a benchmark run."""
    if not HISTORY_AVAILABLE or not benchmark_history:
        raise HTTPException(status_code=503, detail="Benchmark history not available")

    detail = benchmark_history.get_run_detail(run_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Run not found")
    return detail


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str) -> dict:
    """Delete a benchmark run and its results."""
    if not HISTORY_AVAILABLE or not benchmark_history:
        raise HTTPException(status_code=503, detail="Benchmark history not available")

    deleted = benchmark_history.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "deleted", "run_id": run_id}


@router.get("/compare")
async def compare_runs(
    runs: str = Query(description="Comma-separated run IDs"),
) -> dict:
    """Compare two or more benchmark runs side by side."""
    if not HISTORY_AVAILABLE or not benchmark_history:
        raise HTTPException(status_code=503, detail="Benchmark history not available")

    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    if len(run_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="Provide at least 2 run IDs separated by commas.",
        )

    comparison = benchmark_history.compare_runs(run_ids)
    return {
        "runs": comparison.runs,
        "matrix": comparison.matrix,
        "deltas": comparison.deltas,
        "regressions": comparison.regressions,
    }


@router.get("/trends/{model}")
async def get_model_trends(
    model: str,
    last_n: int = Query(default=10, ge=1, le=50),
) -> dict:
    """Get performance trend for a model over recent runs."""
    if not HISTORY_AVAILABLE or not benchmark_history:
        raise HTTPException(status_code=503, detail="Benchmark history not available")

    trend = benchmark_history.get_model_trends(model, last_n_runs=last_n)
    return {
        "model": trend.model,
        "run_ids": trend.run_ids,
        "run_dates": trend.run_dates,
        "avg_scores": trend.avg_scores,
        "avg_times": trend.avg_times,
    }


# =============================================================================
# MODEL CONFIGURATION ENDPOINTS
# =============================================================================

@router.get("/models/config")
async def get_models_config() -> dict:
    """Get the full model routing configuration."""
    config = _load_models_config()
    installed = _get_installed_models()
    return {
        "config": config,
        "installed_models": installed,
    }


@router.put("/models/config")
async def update_models_config(body: ModelConfigUpdate) -> dict:
    """Save updated model routing configuration."""
    if not body.config:
        raise HTTPException(status_code=400, detail="Empty config")

    success = _save_models_config(body.config)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to save models config",
        )

    # Reload config singleton if available
    try:
        from opti_oignon.config import config as app_config
        if hasattr(app_config, "reload"):
            app_config.reload()
    except Exception:
        pass

    return {"status": "saved"}


@router.get("/models/config/roles")
async def get_config_roles() -> dict:
    """Get current role-to-model assignments."""
    config = _load_models_config()
    routing = config.get("routing", {})
    installed = _get_installed_models()

    roles = []
    for role, assignments in routing.items():
        if isinstance(assignments, dict):
            roles.append({
                "role": role,
                "primary": assignments.get("primary", ""),
                "fast": assignments.get("fast", ""),
                "quality": assignments.get("quality", ""),
            })
        elif isinstance(assignments, str):
            roles.append({
                "role": role,
                "primary": assignments,
                "fast": "",
                "quality": "",
            })

    return {"roles": roles, "installed_models": installed}


@router.put("/models/config/roles/{role}")
async def update_role_assignment(role: str, body: ModelRoleUpdate) -> dict:
    """Update model assignment for a specific role."""
    config = _load_models_config()
    routing = config.get("routing", {})

    assignment = {}
    if body.primary:
        assignment["primary"] = body.primary
    if body.fast:
        assignment["fast"] = body.fast
    if body.quality:
        assignment["quality"] = body.quality

    if not assignment:
        raise HTTPException(status_code=400, detail="No model assignments provided")

    routing[role] = assignment
    config["routing"] = routing

    success = _save_models_config(config)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save config")

    return {"status": "updated", "role": role, "assignment": assignment}


@router.post("/models/config/validate")
async def validate_models_config(body: ModelConfigUpdate) -> dict:
    """Validate a model config against installed Ollama models."""
    installed = set(_get_installed_models())
    config = body.config or _load_models_config()
    routing = config.get("routing", {})

    warnings = []
    for role, assignments in routing.items():
        if isinstance(assignments, dict):
            for priority, model in assignments.items():
                if model and model not in installed:
                    warnings.append({
                        "role": role,
                        "priority": priority,
                        "model": model,
                        "issue": "not_installed",
                    })

    fallback = config.get("fallback_order", [])
    for model in fallback:
        if model not in installed:
            warnings.append({
                "role": "fallback_order",
                "priority": "",
                "model": model,
                "issue": "not_installed",
            })

    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "installed_count": len(installed),
    }


@router.get("/models/installed")
async def list_installed_models() -> dict:
    """List all installed Ollama models."""
    models = _get_installed_models()
    return {"models": models, "count": len(models)}
