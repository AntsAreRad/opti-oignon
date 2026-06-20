#!/usr/bin/env python3
"""
Routes API pour la gestion des artifacts.

Endpoints pour lister, consulter, supprimer, telecharger
et exporter les artifacts detectes dans les reponses LLM.
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from .deps import ARTIFACT_AVAILABLE, artifact_manager
from .schemas import ArtifactContent, ArtifactExport, ArtifactInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["artifacts"])


def _check_available():
    """Check that the artifacts module is available."""
    if not ARTIFACT_AVAILABLE or artifact_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Artifact module not available",
        )


@router.get(
    "/conversations/{conv_id}/artifacts",
    response_model=list[ArtifactInfo],
)
def list_artifacts(conv_id: str) -> list:
    """Liste les artifacts of ae conversation."""
    _check_available()
    artifacts = artifact_manager.get_artifacts(conv_id)
    return [
        ArtifactInfo(
            id=a.id,
            artifact_type=a.artifact_type,
            title=a.title,
            language=a.language,
            created_at=a.created_at,
            conversation_id=a.conversation_id,
            display_mode=a.display_mode,
            line_count=a.line_count,
            version=a.version,
            parent_id=a.parent_id,
        )
        for a in artifacts
    ]


@router.get("/artifacts/{artifact_id}", response_model=ArtifactContent)
def get_artifact(artifact_id: str, conv_id: str = "") -> dict:
    """Retrieve an artifact by its ID.

    Le conv_id est necessaire car les artifacts sont indexes par conversation.
    Si non fourni, on cherche dans toutes les conversations en cache.
    """
    _check_available()

    # Search dans une conversation specifique
    if conv_id:
        artifact = artifact_manager.get_artifact_by_id(conv_id, artifact_id)
        if artifact:
            return _artifact_to_content(artifact)
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Search dans toutes les conversations en cache
    for cid in artifact_manager.get_conversation_ids():
        artifact = artifact_manager.get_artifact_by_id(cid, artifact_id)
        if artifact:
            return _artifact_to_content(artifact)

    raise HTTPException(status_code=404, detail="Artifact not found")


@router.get("/artifacts/{artifact_id}/content")
def get_artifact_content(artifact_id: str, conv_id: str = "") -> PlainTextResponse:
    """Retrieve the raw content of an artifact."""
    _check_available()

    artifact = _find_artifact(artifact_id, conv_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    return PlainTextResponse(
        content=artifact.content,
        media_type=artifact.file_extension.lstrip(".") == "html"
        and "text/html"
        or "text/plain",
    )


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str, conv_id: str = "") -> PlainTextResponse:
    """Download an artifact as a file."""
    _check_available()

    artifact = _find_artifact(artifact_id, conv_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    return PlainTextResponse(
        content=artifact.content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{artifact.filename}"',
        },
    )


@router.delete("/artifacts/{artifact_id}")
def delete_artifact(artifact_id: str, conv_id: str = "") -> dict:
    """Delete an artifact."""
    _check_available()

    if conv_id:
        deleted = artifact_manager.delete_artifact(conv_id, artifact_id)
        if deleted:
            return {"deleted": True, "id": artifact_id}
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Chercher dans toutes les conversations
    for cid in artifact_manager.get_conversation_ids():
        if artifact_manager.delete_artifact(cid, artifact_id):
            return {"deleted": True, "id": artifact_id}

    raise HTTPException(status_code=404, detail="Artifact not found")


@router.get(
    "/conversations/{conv_id}/artifacts/export",
    response_model=list[ArtifactExport],
)
def export_artifacts(conv_id: str) -> list:
    """Export all artifacts from a conversation."""
    _check_available()
    exports = artifact_manager.export_artifacts(conv_id)
    return [ArtifactExport(**e) for e in exports]


# -- Helpers --

def _find_artifact(artifact_id: str, conv_id: str = ""):
    """Search for an artifact in a conversation or in the global cache."""
    if conv_id:
        return artifact_manager.get_artifact_by_id(conv_id, artifact_id)
    for cid in artifact_manager.get_conversation_ids():
        a = artifact_manager.get_artifact_by_id(cid, artifact_id)
        if a:
            return a
    return None


def _artifact_to_content(artifact) -> ArtifactContent:
    """Convert an Artifact to ArtifactContent schema."""
    return ArtifactContent(
        id=artifact.id,
        artifact_type=artifact.artifact_type,
        title=artifact.title,
        content=artifact.content,
        language=artifact.language,
        created_at=artifact.created_at,
        display_mode=artifact.display_mode,
        line_count=artifact.line_count,
        version=artifact.version,
        parent_id=artifact.parent_id,
        filename=artifact.filename,
    )
