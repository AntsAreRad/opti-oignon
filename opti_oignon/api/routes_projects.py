#!/usr/bin/env python3
"""
API routes for Projects.

Provides endpoints for project CRUD, file upload/management,
output management, conversation linking, and project context
(indexation, trigger detection, context preview).
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Audit fix: require authentication for all endpoints
try:
    from .routes_auth import _get_current_user
    _auth_dep = [Depends(_get_current_user)]
except ImportError:
    _auth_dep = []

# Conditional imports
try:
    from opti_oignon.projects import (
        Project,
        ProjectFile,
        ProjectOutput,
        project_store,
    )
    PROJECTS_AVAILABLE = True
except ImportError:
    PROJECTS_AVAILABLE = False
    project_store = None

# Project context (indexation + retrieval)
try:
    from opti_oignon.project_context import (
        ProjectContextBuilder,  # noqa: F401
        ProjectIndexer,  # noqa: F401
        project_context_builder,
        project_indexer,
    )
    PROJECT_CONTEXT_AVAILABLE = True
except ImportError:
    PROJECT_CONTEXT_AVAILABLE = False
    project_indexer = None
    project_context_builder = None

# Project trigger detection
try:
    from opti_oignon.project_triggers import (
        ProjectTriggerDetector,  # noqa: F401
        trigger_detector,
    )
    PROJECT_TRIGGERS_AVAILABLE = True
except ImportError:
    PROJECT_TRIGGERS_AVAILABLE = False
    trigger_detector = None

router = APIRouter(prefix="/api", tags=["projects"], dependencies=_auth_dep)


# =============================================================================
# SCHEMAS
# =============================================================================

class ProjectCreateRequest(BaseModel):
    """Request body for creating a project."""
    name: str = Field(..., min_length=1, max_length=200, description="Project name")
    description: str = Field(default="", max_length=2000)
    system_instructions: str = Field(default="", max_length=10000)
    settings: dict[str, Any] = Field(default_factory=dict)


class ProjectUpdateRequest(BaseModel):
    """Request body for updating a project."""
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    system_instructions: str | None = Field(default=None, max_length=10000)
    settings: dict[str, Any] | None = None


class ProjectResponse(BaseModel):
    """Single project in API responses."""
    id: str
    name: str
    description: str = ""
    system_instructions: str = ""
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


class ProjectDetailResponse(BaseModel):
    """Full project detail with files, outputs, conversations, and stats."""
    id: str
    name: str
    description: str = ""
    system_instructions: str = ""
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    files: list[dict[str, Any]] = Field(default_factory=list)
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    conversations: list[dict[str, str]] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class ProjectFileResponse(BaseModel):
    """Single project file in API responses."""
    id: str
    project_id: str
    filename: str
    file_path: str = ""
    file_type: str = ""
    file_size_bytes: int = 0
    indexed: bool = False
    chunk_count: int = 0
    summary: str = ""
    key_terms: list[str] = Field(default_factory=list)
    uploaded_at: str = ""
    updated_at: str = ""


class ProjectOutputResponse(BaseModel):
    """Single project output in API responses."""
    id: str
    project_id: str
    source_conversation_id: str = ""
    filename: str
    file_path: str = ""
    output_type: str = "code"
    description: str = ""
    created_at: str = ""


class OutputCreateRequest(BaseModel):
    """Request body for creating a project output (metadata only, content via file upload)."""
    filename: str = Field(..., min_length=1, max_length=500)
    output_type: str = Field(default="code", max_length=50)
    description: str = Field(default="", max_length=2000)
    source_conversation_id: str = Field(default="", max_length=100)


class ConversationLinkRequest(BaseModel):
    """Request body for linking a conversation to a project."""
    conversation_id: str = Field(default="", max_length=100)


# =============================================================================
# HELPERS
# =============================================================================

def _require_projects():
    """Check that the projects module is available."""
    if not PROJECTS_AVAILABLE or project_store is None:
        raise HTTPException(
            status_code=503,
            detail="Projects module is not available",
        )
    if not project_store.enabled:
        raise HTTPException(
            status_code=503,
            detail="Projects feature is disabled",
        )


def _project_to_response(p: "Project") -> dict[str, Any]:
    """Convert a Project dataclass to a response dict."""
    return p.to_dict()


def _file_to_response(pf: "ProjectFile") -> dict[str, Any]:
    """Convert a ProjectFile dataclass to a response dict."""
    return pf.to_dict()


def _output_to_response(po: "ProjectOutput") -> dict[str, Any]:
    """Convert a ProjectOutput dataclass to a response dict."""
    return po.to_dict()


# =============================================================================
# PROJECT CRUD
# =============================================================================

@router.post("/projects", response_model=ProjectResponse, status_code=201)
async def create_project(req: ProjectCreateRequest) -> dict:
    """Create a new project."""
    _require_projects()
    try:
        project = project_store.create_project(
            name=req.name,
            description=req.description,
            system_instructions=req.system_instructions,
            settings=req.settings if req.settings else None,
        )
        return _project_to_response(project)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/projects", response_model=list[ProjectResponse])
async def list_projects() -> list:
    """List all projects."""
    _require_projects()
    projects = project_store.list_projects()
    return [_project_to_response(p) for p in projects]


@router.get("/projects/{project_id}", response_model=ProjectDetailResponse)
async def get_project(project_id: str) -> dict:
    """Get full project detail with files, outputs, conversations, and stats."""
    _require_projects()
    project = project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    files = project_store.list_files(project_id)
    outputs = project_store.list_outputs(project_id)
    conversations = project_store.list_conversations(project_id)
    stats = project_store.get_project_stats(project_id)

    result = project.to_dict()
    result["files"] = [_file_to_response(f) for f in files]
    result["outputs"] = [_output_to_response(o) for o in outputs]
    result["conversations"] = conversations
    result["stats"] = stats

    return result


@router.put("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, req: ProjectUpdateRequest) -> dict:
    """Update an existing project."""
    _require_projects()
    try:
        project = project_store.update_project(
            project_id=project_id,
            name=req.name,
            description=req.description,
            system_instructions=req.system_instructions,
            settings=req.settings,
        )
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return _project_to_response(project)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str) -> dict:
    """Delete a project and all associated data (files, outputs, conversation links)."""
    _require_projects()
    deleted = project_store.delete_project(project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "ok", "deleted": project_id}


# =============================================================================
# FILE MANAGEMENT
# =============================================================================

@router.post(
    "/projects/{project_id}/files",
    response_model=ProjectFileResponse,
    status_code=201,
)
async def upload_project_file(project_id: str, file: UploadFile = File(...)) -> dict:
    """Upload a file to a project.

    Accepts multipart file upload, validates size and extension,
    stores the file on disk, and registers it in the database.
    """
    _require_projects()

    if file.filename is None:
        raise HTTPException(status_code=422, detail="Filename is required")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {e}")

    try:
        pf = project_store.add_file(
            project_id=project_id,
            filename=file.filename,
            content=content,
        )
        return _file_to_response(pf)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/projects/{project_id}/files", response_model=list[ProjectFileResponse])
async def list_project_files(project_id: str) -> list:
    """List all files for a project."""
    _require_projects()
    # Verify project exists
    project = project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    files = project_store.list_files(project_id)
    return [_file_to_response(f) for f in files]


@router.get("/projects/{project_id}/files/{file_id}", response_model=ProjectFileResponse)
async def get_project_file(project_id: str, file_id: str) -> dict:
    """Get a specific project file's metadata."""
    _require_projects()
    pf = project_store.get_file(file_id)
    if pf is None or pf.project_id != project_id:
        raise HTTPException(status_code=404, detail="File not found")
    return _file_to_response(pf)


@router.delete("/projects/{project_id}/files/{file_id}")
async def delete_project_file(project_id: str, file_id: str) -> dict:
    """Remove a file from a project (deletes from disk and database)."""
    _require_projects()
    # Verify file belongs to this project
    pf = project_store.get_file(file_id)
    if pf is None or pf.project_id != project_id:
        raise HTTPException(status_code=404, detail="File not found")
    removed = project_store.remove_file(file_id)
    if not removed:
        raise HTTPException(status_code=404, detail="File not found")
    return {"status": "ok", "deleted": file_id}


# =============================================================================
# OUTPUT MANAGEMENT
# =============================================================================

@router.post(
    "/projects/{project_id}/outputs",
    response_model=ProjectOutputResponse,
    status_code=201,
)
async def upload_project_output(
    project_id: str,
    file: UploadFile = File(...),
    output_type: str = Query(default="code", max_length=50),
    description: str = Query(default="", max_length=2000),
    source_conversation_id: str = Query(default="", max_length=100),
) -> dict:
    """Upload an output file to a project."""
    _require_projects()

    if file.filename is None:
        raise HTTPException(status_code=422, detail="Filename is required")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {e}")

    try:
        po = project_store.add_output(
            project_id=project_id,
            filename=file.filename,
            content=content,
            output_type=output_type,
            description=description,
            source_conversation_id=source_conversation_id,
        )
        return _output_to_response(po)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/projects/{project_id}/outputs", response_model=list[ProjectOutputResponse])
async def list_project_outputs(project_id: str) -> list:
    """List all outputs for a project."""
    _require_projects()
    project = project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    outputs = project_store.list_outputs(project_id)
    return [_output_to_response(o) for o in outputs]


@router.delete("/projects/{project_id}/outputs/{output_id}")
async def delete_project_output(project_id: str, output_id: str) -> dict:
    """Remove an output from a project."""
    _require_projects()
    po = project_store.get_output(output_id)
    if po is None or po.project_id != project_id:
        raise HTTPException(status_code=404, detail="Output not found")
    removed = project_store.remove_output(output_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Output not found")
    return {"status": "ok", "deleted": output_id}


# =============================================================================
# CONVERSATION LINKING
# =============================================================================

@router.post("/projects/{project_id}/conversations/{conversation_id}", status_code=201)
async def link_conversation(project_id: str, conversation_id: str) -> dict:
    """Link a conversation to a project."""
    _require_projects()
    linked = project_store.link_conversation(project_id, conversation_id)
    if not linked:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "ok", "project_id": project_id, "conversation_id": conversation_id}


@router.delete("/projects/{project_id}/conversations/{conversation_id}")
async def unlink_conversation(project_id: str, conversation_id: str) -> dict:
    """Unlink a conversation from a project."""
    _require_projects()
    unlinked = project_store.unlink_conversation(project_id, conversation_id)
    if not unlinked:
        raise HTTPException(status_code=404, detail="Conversation link not found")
    return {"status": "ok", "project_id": project_id, "conversation_id": conversation_id}


@router.get("/projects/{project_id}/conversations")
async def list_project_conversations(project_id: str) -> dict:
    """List all conversations linked to a project."""
    _require_projects()
    project = project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    conversations = project_store.list_conversations(project_id)
    return {"project_id": project_id, "conversations": conversations}


# =============================================================================
# INDEXATION + CONTEXT
# =============================================================================

@router.post("/projects/{project_id}/files/{file_id}/index")
async def index_project_file(project_id: str, file_id: str) -> dict:
    """Trigger indexation of a single project file into ChromaDB.

    Chunks the file, generates embeddings, stores in the per-project
    collection, and updates the file record with summary and key_terms.
    """
    _require_projects()

    if not PROJECT_CONTEXT_AVAILABLE or project_indexer is None:
        raise HTTPException(
            status_code=503,
            detail="Project context module is not available",
        )

    # Verify file belongs to this project
    pf = project_store.get_file(file_id)
    if pf is None or pf.project_id != project_id:
        raise HTTPException(status_code=404, detail="File not found")

    result = project_indexer.index_file(project_id, file_id)
    if not result.success:
        raise HTTPException(
            status_code=422,
            detail=f"Indexation failed: {result.error}",
        )

    return {
        "status": "ok",
        "file_id": file_id,
        "filename": result.filename,
        "chunk_count": result.chunk_count,
        "summary_length": len(result.summary),
        "key_terms_count": len(result.key_terms),
    }


@router.post("/projects/{project_id}/reindex")
async def reindex_project(project_id: str) -> dict:
    """Reindex all files in a project.

    Processes each file sequentially: chunks, embeds, stores in ChromaDB.
    Returns a summary of results for all files.
    """
    _require_projects()

    if not PROJECT_CONTEXT_AVAILABLE or project_indexer is None:
        raise HTTPException(
            status_code=503,
            detail="Project context module is not available",
        )

    project = project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    results = project_indexer.reindex_project(project_id)
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    return {
        "status": "ok",
        "project_id": project_id,
        "total_files": len(results),
        "indexed": len(succeeded),
        "failed": len(failed),
        "total_chunks": sum(r.chunk_count for r in succeeded),
        "errors": [
            {"file_id": r.file_id, "filename": r.filename, "error": r.error}
            for r in failed
        ],
    }


@router.get("/projects/{project_id}/context")
async def preview_project_context(
    project_id: str,
    query: str = Query(..., min_length=1, description="Query to retrieve context for"),
    budget_tokens: int | None = Query(default=None, ge=100, le=32000),
) -> dict:
    """Preview the context that would be injected for a given query.

    Runs trigger detection and RAG retrieval to show what the LLM
    would receive as additional context. Useful for debugging and
    tuning project settings.
    """
    _require_projects()

    project = project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Run trigger detection if available
    trigger_result = None
    if PROJECT_TRIGGERS_AVAILABLE and trigger_detector is not None:
        try:
            tr = trigger_detector.detect(query, project_id, skip_l3=True)
            trigger_result = {
                "relevant": tr.relevant,
                "confidence": round(tr.confidence, 3),
                "trigger_level": tr.trigger_level,
                "matched_terms": tr.matched_terms,
                "matched_pattern": tr.matched_pattern,
                "duration_ms": round(tr.duration_ms, 2),
                "details": tr.details,
            }
        except Exception as e:
            trigger_result = {"error": str(e)}

    # Build context if available
    context_result = None
    if PROJECT_CONTEXT_AVAILABLE and project_context_builder is not None:
        try:
            ctx = project_context_builder.build_context(
                project_id, query, budget_tokens=budget_tokens,
            )
            context_result = {
                "context_text": ctx.context_text,
                "system_instructions": ctx.system_instructions,
                "chunks_used": ctx.chunks_used,
                "total_tokens_estimate": ctx.total_tokens_estimate,
                "source_files": ctx.source_files,
            }
        except Exception as e:
            context_result = {"error": str(e)}

    return {
        "project_id": project_id,
        "project_name": project.name,
        "query": query,
        "trigger": trigger_result,
        "context": context_result,
    }


@router.get("/projects/{project_id}/files/{file_id}/summary")
async def get_file_summary(project_id: str, file_id: str) -> dict:
    """Get the summary and key terms for an indexed project file.

    Returns the extractive summary and frequency-based key terms
    generated during indexation. Returns empty values if the file
    has not been indexed yet.
    """
    _require_projects()

    pf = project_store.get_file(file_id)
    if pf is None or pf.project_id != project_id:
        raise HTTPException(status_code=404, detail="File not found")

    return {
        "file_id": file_id,
        "filename": pf.filename,
        "indexed": pf.indexed,
        "chunk_count": pf.chunk_count,
        "summary": pf.summary,
        "key_terms": pf.key_terms,
    }
