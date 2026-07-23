#!/usr/bin/env python3
"""
Routes API pour l'execution de code.

Endpoints pour executer du code, extraire les blocs de code
of a texte, et reinitialiser le repertoire de travail persistant.
"""

import logging

from fastapi import APIRouter, HTTPException

from .deps import CODE_EXECUTOR_AVAILABLE, code_executor
from .schemas import (
    CodeBlockInfo,
    CodeBlocksRequest,
    CodeBlocksResponse,
    CodeExecuteRequest,
    CodeExecuteResponse,
)

logger = logging.getLogger(__name__)

# Emergency-stop admission guard (a stopped system refuses honestly)
try:
    from opti_oignon import emergency_stop as _emergency_stop
except Exception:
    _emergency_stop = None

router = APIRouter(prefix="/api/code", tags=["code"])


def _check_available():
    """Check that the code_executor module is available."""
    if not CODE_EXECUTOR_AVAILABLE or code_executor is None:
        raise HTTPException(
            status_code=503,
            detail="Code executor module not available",
        )


@router.post("/execute", response_model=CodeExecuteResponse)
def execute_code(request: CodeExecuteRequest) -> dict:
    """Execute un bloc de code dans un sous-processus sandboxe."""
    if _emergency_stop is not None:
        _emergency_stop.guard_http()  # Refused, not hung
    _check_available()

    if not request.code.strip():
        raise HTTPException(status_code=422, detail="Code cannot be empty")

    try:
        result = code_executor.execute(
            code=request.code,
            language=request.language,
            timeout=request.timeout,
            conv_id=request.conv_id,
        )
        return CodeExecuteResponse(
            success=result.success,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.return_code,
            execution_time=result.execution_time,
            language=result.language,
            truncated=result.truncated,
            error_message=result.error_message,
            output_files=result.output_files,
        )
    except Exception as e:
        logger.error(f"Code execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/blocks", response_model=CodeBlocksResponse)
def extract_code_blocks(request: CodeBlocksRequest) -> dict:
    """Extract code blocks from text (LLM response). Returns {blocks: [...]}."""
    _check_available()

    blocks = code_executor.extract_code_blocks(request.text)
    return CodeBlocksResponse(
        blocks=[
            CodeBlockInfo(
                code=b.code,
                language=b.language,
                start_pos=b.start_pos,
                end_pos=b.end_pos,
            )
            for b in blocks
        ]
    )


@router.post("/reset-workdir")
def reset_workdir(conv_id: str = "") -> dict:
    """Reset the persistent working directory of a conversation."""
    _check_available()

    if not conv_id:
        raise HTTPException(
            status_code=422,
            detail="conv_id query parameter is required",
        )

    success = code_executor.reset_persistent_dir(conv_id)
    return {"reset": success, "conv_id": conv_id}
