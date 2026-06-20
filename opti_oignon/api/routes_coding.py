#!/usr/bin/env python3
"""
API routes for the Coding Agent -- S74/S77/S78/S79.

Provides endpoints for the multi-step autonomous coding loop:
start task, generate/approve plan, execute steps, review diffs, apply changes.

S77: Background execution via _RunState singleton (SQ-07).
POST /api/coding/execute-all returns immediately, runs in background thread.
POST /api/coding/stop signals graceful stop.

S78: Coding History Analytics (SQ-08).
GET /api/coding/history/analytics returns full analytics payload.

S79: Export (JSON/CSV) and batch delete operations.
GET /api/coding/history/export?format=json|csv
POST /api/coding/history/batch-delete

WebSocket /ws/coding/live streams live progress events.

SECURITY: The apply endpoint ALWAYS requires prior human approval.
"""

import asyncio
import csv
import io
import json
import logging
import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .deps import (
    CODING_AGENT_AVAILABLE,
    coding_agent_instance,
    SANDBOX_AVAILABLE,
    sandbox_manager,
    CODING_HISTORY_AVAILABLE,
    coding_history_store,
)
from .schemas import (
    CodingTaskRequest,
    CodingPlanResponse,
    CodingPlanStepResponse,
    CodingCheckpointRequest,
    CodingStepResponse,
    CodingTestResultResponse,
    CodingDiffEntry,
    CodingDiffResponse,
    CodingApplyRequest,
    CodingApplyResponse,
    CodingStatusResponse,
    CodingHistoryEntryResponse,
    CodingTaskSummaryResponse,
    CodingTaskDetailResponse,
    CodingHistoryListResponse,
    CodingHistoryStatsResponse,
    CodingResumeRequest,
    CodingAnalyticsResponse,
    CodingModelSuccessRate,
    CodingModelAvgSteps,
    CodingAvgStepsOverall,
    CodingFailureReason,
    CodingTimeTrend,
    CodingTestPassRate,
    CodingStepsDistribution,
    CodingBatchDeleteRequest,
    CodingBatchDeleteResponse,
)

# S215: emergency-stop admission guard (a stopped system refuses honestly)
try:
    from opti_oignon import emergency_stop as _emergency_stop
except Exception:  # pragma: no cover - constrained environments only
    _emergency_stop = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/coding", tags=["coding"])


# ---------------------------------------------------------------------------
# WebSocket live progress
# ---------------------------------------------------------------------------

class _ProgressBroadcaster:
    """Thread-safe broadcaster for coding agent progress events.

    Manages WebSocket connections and dispatches events from
    the coding agent's progress callbacks.
    """

    def __init__(self):
        self._connections: list[WebSocket] = []
        self._lock = threading.Lock()
        self._event_queue: list[dict] = []
        self._max_queue = 500

    def add_connection(self, ws: WebSocket) -> None:
        with self._lock:
            self._connections.append(ws)

    def remove_connection(self, ws: WebSocket) -> None:
        with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)

    @property
    def connection_count(self) -> int:
        with self._lock:
            return len(self._connections)

    def on_event(self, event: dict) -> None:
        """Callback for coding agent progress events.

        This is called from the agent's thread, so we queue events
        for async dispatch.
        """
        with self._lock:
            if len(self._event_queue) < self._max_queue:
                self._event_queue.append(event)

    def drain_events(self) -> list[dict]:
        """Drain queued events for async dispatch."""
        with self._lock:
            events = self._event_queue[:]
            self._event_queue.clear()
            return events

    async def broadcast(self, event: dict) -> None:
        """Send event to all connected WebSocket clients."""
        dead: list[WebSocket] = []
        with self._lock:
            conns = self._connections[:]

        for ws in conns:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)

        if dead:
            with self._lock:
                for ws in dead:
                    if ws in self._connections:
                        self._connections.remove(ws)


_broadcaster = _ProgressBroadcaster()


# ---------------------------------------------------------------------------
# Background execution state (SQ-07)
# ---------------------------------------------------------------------------

class _RunState:
    """Singleton tracking background execute-all thread state.

    Pattern from S60 BenchmarkRunner: a single background thread
    runs execute_all_steps while the REST endpoint returns immediately.
    """

    def __init__(self):
        self.is_running: bool = False
        self.should_stop: bool = False
        self.error: str = ""
        self.thread: threading.Thread | None = None
        self.executed_count: int = 0
        self.task_id: str = ""
        self._lock = threading.Lock()

    def start(self, agent, task_id: str) -> bool:
        """Launch background execution thread.

        Args:
            agent: CodingAgent instance.
            task_id: Current task ID for tracking.

        Returns:
            True if started, False if already running.
        """
        with self._lock:
            if self.is_running:
                return False
            self.is_running = True
            self.should_stop = False
            self.error = ""
            self.executed_count = 0
            self.task_id = task_id
            self.thread = threading.Thread(
                target=self._run,
                args=(agent,),
                daemon=True,
                name="coding-execute-all",
            )
            self.thread.start()
            return True

    def stop(self) -> bool:
        """Signal graceful stop to the background thread.

        Returns:
            True if a running thread was signalled, False otherwise.
        """
        with self._lock:
            if not self.is_running:
                return False
            self.should_stop = True
            return True

    def _check_stop(self) -> bool:
        """Callback passed to execute_all_steps."""
        return self.should_stop

    def _run(self, agent) -> None:
        """Background thread target: runs execute_all_steps."""
        try:
            steps = agent.execute_all_steps(should_stop=self._check_stop)
            with self._lock:
                self.executed_count = len(steps)
        except Exception as exc:
            logger.error("Background execute-all failed: %s", exc)
            with self._lock:
                self.error = str(exc)
        finally:
            with self._lock:
                self.is_running = False
                self.thread = None

    def get_state(self) -> dict[str, Any]:
        """Get current run state as a dict."""
        with self._lock:
            return {
                "is_running": self.is_running,
                "should_stop": self.should_stop,
                "error": self.error,
                "executed_count": self.executed_count,
                "task_id": self.task_id,
            }


_run_state = _RunState()


# ---------------------------------------------------------------------------
# Helper: get or create agent
# ---------------------------------------------------------------------------

def _get_agent():
    """Get the coding agent instance, raising 503 if unavailable."""
    if not CODING_AGENT_AVAILABLE or coding_agent_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Coding agent not available. Requires sandbox_tools and sandbox_manager.",
        )
    return coding_agent_instance


def _ensure_agent_with_callback():
    """Get agent and ensure progress callback is registered."""
    agent = _get_agent()
    # Register broadcaster callback if not already done
    if _broadcaster.on_event not in agent._progress_callbacks:
        agent.add_progress_callback(_broadcaster.on_event)
    return agent


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@router.post("/start", response_model=CodingStatusResponse)
def start_coding_task(request: CodingTaskRequest) -> dict:
    """Start a new coding task.

    Creates a sandbox session, optionally injects the project,
    and returns initial status.
    """
    if _emergency_stop is not None:
        _emergency_stop.guard_http()  # S215: refused, not hung
    agent = _ensure_agent_with_callback()

    try:
        task_id = agent.start_task(
            task=request.task,
            project_path=request.project_path,
            allow_degraded=request.allow_degraded,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return _build_status_response(agent)


@router.post("/plan", response_model=CodingPlanResponse)
def generate_or_approve_plan(request: CodingCheckpointRequest | None = None) -> dict:
    """Generate a plan or respond to a plan checkpoint.

    - Without request body: generates a new plan from the task.
    - With checkpoint decision: approve, modify, or abort the plan.
    """
    agent = _ensure_agent_with_callback()

    if request is None or request.decision == "":
        # Generate plan
        try:
            plan = agent.generate_plan()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

        return _build_plan_response(plan)

    # Handle checkpoint
    decision = request.decision.lower()

    if decision == "abort":
        agent.abort()
        raise HTTPException(status_code=200, detail="Task aborted")

    if decision == "modify" and request.modified_plan:
        # Rebuild plan from modified data
        from opti_oignon.coding_agent import (
            CodingPlan, PlanStep, PlanStepType,
        )
        raw = request.modified_plan
        steps = []
        for i, s in enumerate(raw.get("steps", []), start=1):
            try:
                stype = PlanStepType(s.get("step_type", "bash"))
            except ValueError:
                stype = PlanStepType.BASH
            steps.append(PlanStep(
                step_number=i,
                step_type=stype,
                description=s.get("description", ""),
                file_path=s.get("file_path", ""),
                command=s.get("command", ""),
                content=s.get("content", ""),
                old_str=s.get("old_str", ""),
                new_str=s.get("new_str", ""),
            ))
        modified = CodingPlan(
            task=raw.get("task", agent._task),
            steps=steps,
            summary=raw.get("summary", ""),
            estimated_files=raw.get("estimated_files", 0),
        )
        agent.set_plan(modified)
        return _build_plan_response(modified)

    # decision == "approve" — just return the current plan
    if agent.plan is None:
        raise HTTPException(status_code=404, detail="No plan to approve")

    return _build_plan_response(agent.plan)


@router.post("/step", response_model=CodingStepResponse)
def execute_next_step() -> dict:
    """Execute the next step in the plan.

    Returns the step result. Returns 404 when all steps are done.
    """
    if _emergency_stop is not None:
        _emergency_stop.guard_http()  # S215: refused, not hung
    agent = _ensure_agent_with_callback()

    if agent.plan is None:
        raise HTTPException(status_code=409, detail="No plan. Call /plan first.")

    try:
        step = agent.execute_next_step()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    if step is None:
        raise HTTPException(status_code=404, detail="No more steps to execute")

    return CodingStepResponse(
        step_number=step.step_number,
        step_type=step.step_type.value,
        description=step.description,
        completed=step.completed,
        result=step.result[:2000],
        error=step.error[:2000],
    )


@router.post("/execute-all")
def execute_all_steps_background() -> dict:
    """Execute all remaining plan steps in a background thread.

    Returns immediately with {"started": true}. Progress is streamed
    via WebSocket /ws/coding/live. Poll GET /status or GET /execute-all/status
    for completion state.

    Returns 409 if already running or no plan available.
    """
    if _emergency_stop is not None:
        _emergency_stop.guard_http()  # S215: refused, not hung
    agent = _ensure_agent_with_callback()

    if agent.plan is None:
        raise HTTPException(
            status_code=409, detail="No plan. Call /plan first."
        )

    started = _run_state.start(agent, agent.task_id)
    if not started:
        raise HTTPException(
            status_code=409,
            detail="Background execution already running.",
        )

    return {"started": True, "task_id": agent.task_id}


@router.get("/execute-all/status")
def execute_all_status() -> dict:
    """Get the state of the background execute-all thread."""
    return _run_state.get_state()


@router.post("/stop")
def stop_execution() -> dict:
    """Signal graceful stop to the background execute-all thread.

    The current step finishes, then execution halts. Returns 409
    if no background execution is running.
    """
    stopped = _run_state.stop()
    if not stopped:
        raise HTTPException(
            status_code=409,
            detail="No background execution running to stop.",
        )

    return {"stopping": True, "task_id": _run_state.task_id}


@router.get("/status", response_model=CodingStatusResponse)
def get_coding_status() -> dict:
    """Get current coding agent status."""
    agent = _get_agent()
    return _build_status_response(agent)


@router.get("/diff", response_model=CodingDiffResponse)
def get_diffs() -> dict:
    """Generate and return diffs of all changes.

    Compares original injected files with current sandbox state.
    """
    agent = _ensure_agent_with_callback()

    try:
        diffs = agent.generate_diffs()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return CodingDiffResponse(
        count=len(diffs),
        diffs=[
            CodingDiffEntry(
                path=d.path,
                is_new=d.is_new,
                is_deleted=d.is_deleted,
                diff="\n".join(d.diff_lines)[:10000],
            )
            for d in diffs
        ],
    )


@router.post("/approve", response_model=CodingApplyResponse)
def approve_and_apply(request: CodingApplyRequest | None = None) -> dict:
    """Approve and apply sandbox changes to the real filesystem.

    SECURITY: This is the ONLY exit from the sandbox. The frontend
    must present diffs and get explicit user confirmation before
    calling this endpoint.
    """
    agent = _ensure_agent_with_callback()

    target = request.target_path if request else None

    try:
        result = agent.apply_changes(target_path=target)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return CodingApplyResponse(
        applied=result.get("applied", 0),
        files=result.get("files", []),
        errors=result.get("errors", []),
    )


@router.post("/abort")
def abort_task() -> dict:
    """Abort the current coding task and destroy the sandbox."""
    agent = _ensure_agent_with_callback()

    success = agent.abort()
    return {
        "aborted": True,
        "cleanup_success": success,
        "task_id": agent.task_id,
    }


# ---------------------------------------------------------------------------
# WebSocket: live progress
# ---------------------------------------------------------------------------

@router.websocket("/ws/coding/live")
async def websocket_coding_live(ws: WebSocket) -> None:
    """Stream live coding agent progress events.

    S136 audit fix: authenticates before processing.
    """
    await ws.accept()

    # S136 audit fix: authenticate WebSocket connection
    try:
        from .routes_auth import authenticate_websocket
        user = await authenticate_websocket(ws)
        if user is None:
            await ws.send_json({"type": "error", "detail": "Authentication required"})
            await ws.close(code=4001)
            return
    except Exception:
        await ws.send_json({"type": "error", "detail": "Authentication failed"})
        await ws.close(code=4001)
        return

    _broadcaster.add_connection(ws)
    logger.info(
        "Coding WebSocket connected (total: %d)",
        _broadcaster.connection_count,
    )

    try:
        # Send initial status
        agent = _get_agent()
        await ws.send_json({
            "type": "connected",
            "phase": agent.phase.value,
            "task_id": agent.task_id,
        })

        # Poll for events and forward to client
        while True:
            # Drain queued events from the agent thread
            events = _broadcaster.drain_events()
            for event in events:
                await ws.send_json(event)

            # Check for incoming messages (keepalive / commands)
            try:
                data = await asyncio.wait_for(
                    ws.receive_text(), timeout=1.0
                )
                # Client can send ping
                if data == "ping":
                    await ws.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Send heartbeat
                await ws.send_json({
                    "type": "heartbeat",
                    "timestamp": time.time(),
                })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("Coding WebSocket error: %s", exc)
    finally:
        _broadcaster.remove_connection(ws)
        logger.info(
            "Coding WebSocket disconnected (remaining: %d)",
            _broadcaster.connection_count,
        )


# ---------------------------------------------------------------------------
# Task History endpoints (S76)
# ---------------------------------------------------------------------------


@router.get("/history", response_model=CodingHistoryListResponse)
def list_task_history(
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
) -> dict:
    """List persisted coding agent tasks with optional status filter."""
    if not CODING_HISTORY_AVAILABLE or coding_history_store is None:
        return CodingHistoryListResponse(tasks=[], total=0)
    tasks = coding_history_store.list_tasks(
        limit=limit, offset=offset, status=status
    )
    total = coding_history_store.count_tasks(status=status)
    return CodingHistoryListResponse(
        tasks=[CodingTaskSummaryResponse(**t.to_dict()) for t in tasks],
        total=total,
    )


@router.get("/history/stats", response_model=CodingHistoryStatsResponse)
def get_history_stats() -> dict:
    """Get aggregate statistics for coding history."""
    if not CODING_HISTORY_AVAILABLE or coding_history_store is None:
        return CodingHistoryStatsResponse()
    stats = coding_history_store.get_stats()
    return CodingHistoryStatsResponse(**stats)


@router.get("/history/analytics", response_model=CodingAnalyticsResponse)
def get_history_analytics() -> dict:
    """Get full coding history analytics (S78 SQ-08).

    Aggregates success rates, step counts, model comparison,
    failure reasons, time trends, and test pass rates.
    All computation is done via SQL in CodingHistoryStore.
    """
    if not CODING_HISTORY_AVAILABLE or coding_history_store is None:
        return CodingAnalyticsResponse()
    raw = coding_history_store.get_analytics()
    return CodingAnalyticsResponse(
        total_tasks=raw.get("total_tasks", 0),
        completed_tasks=raw.get("completed_tasks", 0),
        overall_success_rate=raw.get("overall_success_rate", 0.0),
        success_rate_by_model=[
            CodingModelSuccessRate(**entry)
            for entry in raw.get("success_rate_by_model", [])
        ],
        avg_steps_by_model=[
            CodingModelAvgSteps(**entry)
            for entry in raw.get("avg_steps_by_model", [])
        ],
        avg_steps_overall=CodingAvgStepsOverall(
            **raw.get("avg_steps_overall", {})
        ),
        failure_reasons=[
            CodingFailureReason(**entry)
            for entry in raw.get("failure_reasons", [])
        ],
        time_trends=[
            CodingTimeTrend(**entry)
            for entry in raw.get("time_trends", [])
        ],
        test_pass_rate_per_task=[
            CodingTestPassRate(**entry)
            for entry in raw.get("test_pass_rate_per_task", [])
        ],
        steps_distribution=[
            CodingStepsDistribution(**entry)
            for entry in raw.get("steps_distribution", [])
        ],
    )


# ---------------------------------------------------------------------------
# Export & Batch Delete (S79)
# ---------------------------------------------------------------------------


@router.get("/history/export")
def export_history(format: str = "json") -> StreamingResponse:
    """Export task history as a downloadable file.

    Query params:
        format: 'json' (default) or 'csv'.

    Returns a streaming response with Content-Disposition header
    for file download.
    """
    if not CODING_HISTORY_AVAILABLE or coding_history_store is None:
        raise HTTPException(
            status_code=503, detail="Coding history not available"
        )

    fmt = format.lower().strip()

    if fmt == "csv":
        rows = coding_history_store.export_tasks_csv_rows()
        if not rows:
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "task_id", "task_text", "model", "status",
                "step_count", "test_runs", "pass_rate",
                "created_at", "completed_at", "duration_seconds",
            ])
            output.seek(0)
        else:
            output = io.StringIO()
            fieldnames = [
                "task_id", "task_text", "model", "status",
                "step_count", "test_runs", "pass_rate",
                "created_at", "completed_at", "duration_seconds",
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
            output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    "attachment; filename=coding_history.csv"
            },
        )

    elif fmt == "json":
        data = coding_history_store.export_tasks_json()
        content = json.dumps(data, indent=2, default=str)
        return StreamingResponse(
            io.StringIO(content),
            media_type="application/json",
            headers={
                "Content-Disposition":
                    "attachment; filename=coding_history.json"
            },
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported export format: {fmt}. Use 'json' or 'csv'.",
        )


@router.post(
    "/history/batch-delete",
    response_model=CodingBatchDeleteResponse,
)
def batch_delete_tasks(request: CodingBatchDeleteRequest) -> dict:
    """Batch delete coding tasks.

    Accepts either:
    - task_ids: list of task IDs to delete
    - before_date: ISO date string (e.g. '2025-01-01'); deletes tasks
      created before that date.

    If both are provided, task_ids takes precedence.
    """
    if not CODING_HISTORY_AVAILABLE or coding_history_store is None:
        raise HTTPException(
            status_code=503, detail="Coding history not available"
        )

    if request.task_ids is not None:
        deleted = coding_history_store.batch_delete_by_ids(request.task_ids)
        return CodingBatchDeleteResponse(deleted=deleted)

    if request.before_date is not None:
        # Parse ISO date to timestamp
        from datetime import datetime, timezone
        try:
            dt = datetime.fromisoformat(request.before_date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = dt.timestamp()
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format: {request.before_date}. "
                       f"Use ISO format (e.g. '2025-01-01'). Error: {exc}",
            )
        deleted = coding_history_store.batch_delete_before_date(ts)
        return CodingBatchDeleteResponse(deleted=deleted)

    raise HTTPException(
        status_code=400,
        detail="Provide either 'task_ids' or 'before_date'.",
    )


@router.get("/history/{task_id}", response_model=CodingTaskDetailResponse)
def get_task_detail(task_id: str) -> dict:
    """Get full detail of a persisted coding task."""
    if not CODING_HISTORY_AVAILABLE or coding_history_store is None:
        raise HTTPException(
            status_code=503, detail="Coding history not available"
        )
    detail = coding_history_store.get_task_detail(task_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return CodingTaskDetailResponse(**detail.to_dict())


@router.post("/resume/{task_id}")
def resume_task(task_id: str, request: CodingResumeRequest | None = None) -> dict:
    """Resume a previously interrupted coding task from its last checkpoint.

    Loads the checkpoint state and returns it for the frontend to
    re-initialize the agent. The actual agent re-creation happens
    via the normal start_coding_task flow with restored state.
    """
    if _emergency_stop is not None:
        _emergency_stop.guard_http()  # S215: refused, not hung
    if not CODING_HISTORY_AVAILABLE or coding_history_store is None:
        raise HTTPException(
            status_code=503, detail="Coding history not available"
        )
    checkpoint = coding_history_store.get_last_checkpoint(task_id)
    if checkpoint is None:
        raise HTTPException(
            status_code=404, detail="No checkpoint found for task"
        )
    return {
        "task_id": checkpoint.task_id,
        "task_text": checkpoint.task_text,
        "project_path": checkpoint.project_path,
        "model": (
            request.model if request and request.model
            else checkpoint.model
        ),
        "plan_json": checkpoint.plan_json,
        "current_step": checkpoint.current_step,
        "phase": checkpoint.phase,
        "originals_hash": checkpoint.originals_hash,
    }


@router.delete("/history/{task_id}")
def delete_task_history(task_id: str) -> dict:
    """Delete a persisted coding task and all associated records."""
    if not CODING_HISTORY_AVAILABLE or coding_history_store is None:
        raise HTTPException(
            status_code=503, detail="Coding history not available"
        )
    deleted = coding_history_store.delete_task(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"deleted": True, "task_id": task_id}


@router.post("/history/prune")
def prune_history() -> dict:
    """Remove old tasks based on retention policy."""
    if not CODING_HISTORY_AVAILABLE or coding_history_store is None:
        raise HTTPException(
            status_code=503, detail="Coding history not available"
        )
    pruned = coding_history_store.prune()
    return {"pruned": pruned}


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def _build_status_response(agent) -> CodingStatusResponse:
    """Build a CodingStatusResponse from agent state."""
    status = agent.get_status()

    plan_data = status.get("plan")
    plan_resp = None
    if plan_data:
        plan_resp = CodingPlanResponse(
            task=plan_data.get("task", ""),
            summary=plan_data.get("summary", ""),
            estimated_files=plan_data.get("estimated_files", 0),
            total_steps=plan_data.get("total_steps", 0),
            completed_steps=plan_data.get("completed_steps", 0),
            steps=[
                CodingPlanStepResponse(**s)
                for s in plan_data.get("steps", [])
            ],
        )

    return CodingStatusResponse(
        task_id=status.get("task_id", ""),
        task=status.get("task", ""),
        phase=status.get("phase", "idle"),
        session_active=status.get("session_active", False),
        plan=plan_resp,
        current_step=status.get("current_step", 0),
        total_steps=status.get("total_steps", 0),
        iteration=status.get("iteration", 0),
        max_iterations=status.get("max_iterations", 10),
        fix_count=status.get("fix_count", 0),
        max_fix_retries=status.get("max_fix_retries", 3),
        test_results=[
            CodingTestResultResponse(**t)
            for t in status.get("test_results", [])
        ],
        diffs=[
            CodingDiffEntry(**d)
            for d in status.get("diffs", [])
        ],
        history_count=status.get("history_count", 0),
        history=[
            CodingHistoryEntryResponse(**h)
            for h in status.get("history", [])
        ],
    )


def _build_plan_response(plan) -> CodingPlanResponse:
    """Build a CodingPlanResponse from a CodingPlan."""
    plan_dict = plan.to_dict()
    return CodingPlanResponse(
        task=plan_dict.get("task", ""),
        summary=plan_dict.get("summary", ""),
        estimated_files=plan_dict.get("estimated_files", 0),
        total_steps=plan_dict.get("total_steps", 0),
        completed_steps=plan_dict.get("completed_steps", 0),
        steps=[
            CodingPlanStepResponse(**s)
            for s in plan_dict.get("steps", [])
        ],
    )
