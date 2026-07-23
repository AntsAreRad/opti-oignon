#!/usr/bin/env python3
"""
Conversation branching API routes.

POST   /api/branches/fork                      -- Fork conversation at message
GET    /api/branches/{conversation_id}          -- List branches for conversation
GET    /api/branches/detail/{branch_id}         -- Get single branch details
GET    /api/branches/{branch_id}/messages       -- Get branch messages (shared + branch)
PUT    /api/branches/{branch_id}                -- Rename/recolor branch
DELETE /api/branches/{branch_id}                -- Delete branch
GET    /api/branches/{conversation_id}/tree     -- Get branch tree structure
POST   /api/branches/compare                    -- Compare two branches side-by-side
POST   /api/branches/merge                      -- Merge messages between branches
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/branches", tags=["branches"])


# =============================================================================
# SCHEMAS
# =============================================================================

class ForkRequest(BaseModel):
    """Request body for forking a conversation."""
    conversation_id: str = Field(description="Conversation UUID to fork")
    fork_message_id: int = Field(description="Message ID at the fork point (last shared)")
    name: str | None = Field(default=None, description="Branch name (auto-generated if omitted)")
    color: str | None = Field(default=None, description="Branch color hex (auto-assigned if omitted)")
    parent_branch_id: str | None = Field(default=None, description="Parent branch ID (None = fork from main)")


class BranchUpdateRequest(BaseModel):
    """Request body for updating a branch."""
    name: str | None = Field(default=None, description="New branch name")
    color: str | None = Field(default=None, description="New branch color hex")
    metadata: dict[str, Any] | None = Field(default=None, description="Metadata to merge")


class CompareRequest(BaseModel):
    """Request body for comparing two branches."""
    conversation_id: str = Field(description="Conversation UUID")
    branch_a_id: str | None = Field(default=None, description="First branch ID (None = main)")
    branch_b_id: str | None = Field(default=None, description="Second branch ID (None = main)")


class MergeRequest(BaseModel):
    """Request body for merging messages between branches."""
    source_branch_id: str = Field(description="Source branch UUID")
    target_branch_id: str = Field(description="Target branch UUID")
    message_ids: list[int] | None = Field(default=None, description="Specific message IDs (None = all)")


class AddBranchMessageRequest(BaseModel):
    """Request body for adding a message to a branch."""
    conversation_id: str = Field(description="Conversation UUID")
    role: str = Field(description="Message role (user, assistant, system)")
    content: str = Field(description="Message content")
    model: str | None = Field(default=None, description="Model used")
    metadata: dict[str, Any] | None = Field(default=None, description="Extra metadata")


class BranchResponse(BaseModel):
    """Response for a single branch."""
    branch_id: str
    conversation_id: str
    parent_branch_id: str | None
    fork_message_id: int
    name: str
    color: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = {}
    stats: dict[str, Any] | None = None


class BranchMessageResponse(BaseModel):
    """Response for a branch message."""
    id: int
    branch_id: str
    conversation_id: str
    role: str
    content: str
    timestamp: str
    token_estimate: int = 0
    model: str | None = None
    metadata: dict[str, Any] = {}


# =============================================================================
# HELPERS
# =============================================================================

def _check_available():
    """Verify branches module is available."""
    from .deps import BRANCHES_AVAILABLE, branch_manager
    if not BRANCHES_AVAILABLE or branch_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Branches module not available",
        )
    return branch_manager


# =============================================================================
# ENDPOINTS
# =============================================================================

# ---------------------------------------------------------------------------
# Specific literal routes FIRST (before catch-all {id} parameters)
# ---------------------------------------------------------------------------

@router.post("/fork", response_model=BranchResponse)
def fork_conversation(req: ForkRequest) -> dict:
    """Fork a conversation at a specific message."""
    mgr = _check_available()
    branch = mgr.fork(
        conversation_id=req.conversation_id,
        fork_message_id=req.fork_message_id,
        name=req.name,
        color=req.color,
        parent_branch_id=req.parent_branch_id,
    )
    if not branch:
        raise HTTPException(
            status_code=400,
            detail="Failed to create branch (limit reached or invalid parameters)",
        )
    stats = mgr.get_branch_stats(branch.branch_id)
    return BranchResponse(**branch.to_dict(), stats=stats)


@router.post("/compare")
def compare_branches(req: CompareRequest) -> dict:
    """Compare two branches side-by-side."""
    mgr = _check_available()
    if req.branch_a_id is None and req.branch_b_id is None:
        raise HTTPException(
            status_code=400,
            detail="At least one branch ID must be provided",
        )
    comparison = mgr.compare_branches(
        conversation_id=req.conversation_id,
        branch_a_id=req.branch_a_id,
        branch_b_id=req.branch_b_id,
    )
    if not comparison:
        raise HTTPException(status_code=404, detail="Branch not found or comparison failed")
    return comparison.to_dict()


@router.post("/merge")
def merge_branches(req: MergeRequest) -> dict:
    """Merge messages from one branch into another."""
    mgr = _check_available()
    merged = mgr.merge_messages(
        source_branch_id=req.source_branch_id,
        target_branch_id=req.target_branch_id,
        message_ids=req.message_ids,
    )
    return {
        "merged_count": len(merged),
        "source_branch_id": req.source_branch_id,
        "target_branch_id": req.target_branch_id,
        "messages": [m.to_dict() for m in merged],
    }


@router.get("/detail/{branch_id}", response_model=BranchResponse)
def get_branch_detail(branch_id: str) -> dict:
    """Get details for a single branch."""
    mgr = _check_available()
    branch = mgr.get_branch(branch_id)
    if not branch:
        raise HTTPException(status_code=404, detail="Branch not found")
    stats = mgr.get_branch_stats(branch_id)
    return BranchResponse(**branch.to_dict(), stats=stats)


# ---------------------------------------------------------------------------
# Routes with {id}/suffix (specific suffixes match before bare {id})
# ---------------------------------------------------------------------------

@router.get("/{branch_id}/messages")
def get_branch_messages(
    branch_id: str,
    full: bool = Query(default=True, description="Include shared history (True) or branch-only (False)"),
) -> dict:
    """Get messages for a branch.

    With full=True (default), returns shared history + branch-specific messages.
    With full=False, returns only branch-specific messages.
    """
    mgr = _check_available()
    branch = mgr.get_branch(branch_id)
    if not branch:
        raise HTTPException(status_code=404, detail="Branch not found")

    if full:
        messages = mgr.get_branch_messages_full(
            conversation_id=branch.conversation_id,
            branch_id=branch_id,
        )
    else:
        msgs = mgr.get_branch_only_messages(branch_id)
        messages = [m.to_dict() for m in msgs]

    return {"branch_id": branch_id, "messages": messages, "count": len(messages)}


@router.post("/{branch_id}/messages", response_model=BranchMessageResponse)
def add_message_to_branch(branch_id: str, req: AddBranchMessageRequest) -> dict:
    """Add a message to a branch."""
    mgr = _check_available()
    msg = mgr.add_branch_message(
        branch_id=branch_id,
        conversation_id=req.conversation_id,
        role=req.role,
        content=req.content,
        model=req.model,
        metadata=req.metadata,
    )
    if not msg:
        raise HTTPException(status_code=400, detail="Failed to add message (branch not found?)")
    return BranchMessageResponse(**msg.to_dict())


@router.get("/{conversation_id}/tree")
def get_branch_tree(conversation_id: str) -> dict:
    """Get the branch tree structure for a conversation."""
    mgr = _check_available()
    tree = mgr.get_branch_tree(conversation_id)
    return tree.to_dict()


# ---------------------------------------------------------------------------
# Bare {id} routes LAST (catch-all pattern)
# ---------------------------------------------------------------------------

@router.get("/{conversation_id}", response_model=list[BranchResponse])
def list_branches(conversation_id: str) -> dict:
    """List all branches for a conversation."""
    mgr = _check_available()
    branches = mgr.list_branches(conversation_id)
    result = []
    for b in branches:
        stats = mgr.get_branch_stats(b.branch_id)
        result.append(BranchResponse(**b.to_dict(), stats=stats))
    return result


@router.put("/{branch_id}", response_model=BranchResponse)
def update_branch(branch_id: str, req: BranchUpdateRequest) -> dict:
    """Rename, recolor, or update metadata for a branch."""
    mgr = _check_available()
    updated = mgr.update_branch(
        branch_id=branch_id,
        name=req.name,
        color=req.color,
        metadata=req.metadata,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Branch not found")
    stats = mgr.get_branch_stats(branch_id)
    return BranchResponse(**updated.to_dict(), stats=stats)


@router.delete("/{branch_id}")
def delete_branch(branch_id: str) -> dict:
    """Delete a branch and all its messages."""
    mgr = _check_available()
    ok = mgr.delete_branch(branch_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Branch not found")
    return {"deleted": True, "branch_id": branch_id}
