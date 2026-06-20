#!/usr/bin/env python3
"""
Agent eval API routes -- S230 (AGT_SPEC Section 7.4).

The cycle's ONLY new router. Five endpoints under /api/agent-eval, the
benchmark idiom throughout: POST /run answers 409 when a run is already in
progress, GET /status reads the runner snapshot, GET /results/{run_id} and
GET /history read the store, POST /cancel requests cooperative
cancellation. Auth parity with the existing routers (the SYN-06 idiom): a
router-level dependency on the session user, the same one routes_sync /
routes_security / routes_governor carry; the global deny-by-default
AuthMiddleware already covers the prefix, so this is parity, not a gap
closure.

No frontend in this cycle (FRD-03 territory); the response shapes are the
read-phase decisions of the landing lot and are ready for it.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module conventions (project-wide).
checkpoint_before_apply = True

try:
    from opti_oignon.agent_eval import (
        EvalRunner,
        get_eval_runner,
        load_suite,
        reset_eval_runner,
    )
    from opti_oignon.agent_eval import runner as _runner_mod

    _EVAL_OK = True
except Exception:  # pragma: no cover - partial build degradation
    EvalRunner = None  # type: ignore[assignment]
    get_eval_runner = None  # type: ignore[assignment]
    load_suite = None  # type: ignore[assignment]
    reset_eval_runner = None  # type: ignore[assignment]
    _runner_mod = None  # type: ignore[assignment]
    _EVAL_OK = False

FEATURE_AVAILABLE = _EVAL_OK


# ---------------------------------------------------------------------------
# FastAPI surface (guarded; thin wrappers over the runner and the store)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, Depends, HTTPException, Query
    from pydantic import BaseModel, Field

    # Auth parity (the SYN-06 idiom): require authentication on every
    # endpoint, the same per-router dependency the recent routers carry.
    try:
        from .routes_auth import _get_current_user

        _auth_dep = [Depends(_get_current_user)]
    except ImportError:  # pragma: no cover - auth module layout change
        _auth_dep = []

    router = APIRouter(
        prefix="/api/agent-eval", tags=["agent-eval"], dependencies=_auth_dep
    )

    class EvalRunRequest(BaseModel):
        models: list[str] | str = Field(
            description="Model names: a list, or a comma-separated string"
        )
        suite: str = "micro"
        repeats: int = 1
        evict_between: bool = True

    def _runner() -> Any:
        if not _EVAL_OK or get_eval_runner is None:
            raise HTTPException(
                status_code=503, detail="Agent eval harness not available"
            )
        runner = get_eval_runner()
        if not getattr(_runner_mod, "FEATURE_AVAILABLE", False):
            raise HTTPException(
                status_code=503,
                detail="Agent surface not available; eval runner disabled",
            )
        return runner

    def _models_list(raw: list[str] | str) -> list[str]:
        if isinstance(raw, str):
            return [m.strip() for m in raw.split(",") if m.strip()]
        return [m.strip() for m in raw if isinstance(m, str) and m.strip()]

    @router.post("/run")
    def eval_run(request: EvalRunRequest) -> dict[str, Any]:
        """Start an eval run (409 when one is already in progress)."""
        runner = _runner()
        models = _models_list(request.models)
        if not models:
            raise HTTPException(
                status_code=422, detail="models must name at least one model"
            )
        if request.repeats < 1:
            raise HTTPException(status_code=422, detail="repeats must be >= 1")
        try:
            load_suite(request.suite)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Suite '{request.suite}' not found"
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if runner.is_busy:
            raise HTTPException(
                status_code=409,
                detail=(
                    "An eval run is already in progress. Cancel it or wait"
                    " for completion."
                ),
            )
        try:
            run_id = runner.start_run(
                models=models,
                suite=request.suite,
                repeats=request.repeats,
                evict_between=request.evict_between,
            )
        except RuntimeError as exc:
            # The runner-level busy guard raced the route-level check.
            raise HTTPException(status_code=409, detail=str(exc))
        return {"run_id": run_id, "started": True}

    @router.get("/status")
    def eval_status() -> dict[str, Any]:
        """The runner snapshot: busy flag and run progress."""
        return _runner().status()

    @router.get("/results/{run_id}")
    def eval_results(run_id: str) -> dict[str, Any]:
        """Full results for one run: run row, task rows, summary."""
        details = _runner().store.get_run_details(run_id)
        if details is None:
            raise HTTPException(
                status_code=404, detail=f"Run '{run_id}' not found"
            )
        return details

    @router.get("/history")
    def eval_history(
        limit: int = Query(50, ge=1, le=500),
        suite: str | None = Query(None),
    ) -> dict[str, Any]:
        """Recent runs, newest first."""
        return {"runs": _runner().store.get_history(limit, suite)}

    @router.post("/cancel")
    def eval_cancel() -> dict[str, bool]:
        """Request cooperative cancellation of the active run."""
        return {"cancelled": _runner().cancel()}

except Exception:  # pragma: no cover - FastAPI absent (isolated tests)
    router = None  # type: ignore[assignment]
