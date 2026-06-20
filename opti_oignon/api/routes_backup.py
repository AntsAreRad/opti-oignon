#!/usr/bin/env python3
"""
API routes for configuration backup and restore -- Opti-Oignon S121, S125.

Endpoints:
  GET  /api/backup/sections           -- List available sections with item counts
  GET  /api/backup/export             -- Export full or partial backup JSON
  POST /api/backup/import             -- Import backup with merge/replace strategy
  POST /api/backup/preview            -- Preview import diff without applying
  POST /api/backup/export/encrypted   -- Export encrypted backup (S125)
  POST /api/backup/import/encrypted   -- Import encrypted backup (S125)
"""

import base64
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .deps import BACKUP_AVAILABLE, backup_manager
from .schemas import (
    BackupImportRequest,
    BackupImportResponse,
    BackupPreviewRequest,
    BackupPreviewResponse,
    BackupSectionInfo,
    BackupSectionsResponse,
)

logger = logging.getLogger(__name__)

# S136 audit fix: require authentication for all endpoints
try:
    from .routes_auth import _get_current_user
    _auth_dep = [Depends(_get_current_user)]
except ImportError:
    _auth_dep = []

router = APIRouter(prefix="/api/backup", tags=["backup"], dependencies=_auth_dep)


def _check_available() -> None:
    """Verify the backup manager is available."""
    if not BACKUP_AVAILABLE or backup_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Backup manager not available",
        )


# -----------------------------------------------------------------
# GET /api/backup/sections
# -----------------------------------------------------------------

@router.get("/sections", response_model=BackupSectionsResponse)
def list_backup_sections() -> dict:
    """List available backup sections with item counts."""
    _check_available()
    try:
        sections = backup_manager.list_sections()
        return BackupSectionsResponse(
            sections=[BackupSectionInfo(**s) for s in sections],
        )
    except Exception as exc:
        logger.error("Failed to list backup sections: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------
# GET /api/backup/export
# -----------------------------------------------------------------

@router.get("/export")
def export_backup(
    sections: Optional[str] = Query(
        None,
        description="Comma-separated section names to export. Omit for all.",
    ),
) -> JSONResponse:
    """Generate and download a backup JSON.

    Query params:
        sections: Comma-separated list of sections (optional, default=all).

    Returns:
        JSON backup file with Content-Disposition header for download.
    """
    _check_available()

    try:
        if sections:
            section_list = [s.strip() for s in sections.split(",") if s.strip()]
            data = backup_manager.export_sections(section_list)
        else:
            data = backup_manager.export_all()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Backup export failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(exc)}")

    return JSONResponse(
        content=data,
        headers={
            "Content-Disposition": "attachment; filename=opti-oignon.oo-backup.json",
        },
    )


# -----------------------------------------------------------------
# POST /api/backup/preview
# -----------------------------------------------------------------

@router.post("/preview", response_model=BackupPreviewResponse)
def preview_import(request: BackupPreviewRequest) -> dict:
    """Preview what an import would change without applying.

    Body:
        backup: The backup JSON object.
        strategy: 'merge' or 'replace' (default: merge).

    Returns:
        Preview with diff items and summary of changes.
    """
    _check_available()

    if not request.backup:
        raise HTTPException(status_code=400, detail="Empty backup data")

    try:
        preview = backup_manager.preview_import(
            request.backup,
            strategy=request.strategy,
        )
        return BackupPreviewResponse(**preview.to_dict())
    except Exception as exc:
        logger.error("Backup preview failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(exc)}")


# -----------------------------------------------------------------
# POST /api/backup/import
# -----------------------------------------------------------------

@router.post("/import", response_model=BackupImportResponse)
def import_backup(request: BackupImportRequest) -> JSONResponse:
    """Upload and apply a backup file.

    Body:
        backup: The backup JSON object.
        strategy: 'merge' (keep existing, add missing) or 'replace' (overwrite).

    Returns:
        Import result with success/failure details.
    """
    _check_available()

    if not request.backup:
        raise HTTPException(status_code=400, detail="Empty backup data")

    if request.strategy not in ("merge", "replace"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy: '{request.strategy}'. Must be 'merge' or 'replace'.",
        )

    try:
        result = backup_manager.import_backup(
            request.backup,
            strategy=request.strategy,
            allow_unsigned=request.allow_unsigned,
        )
        status_code = 200 if result.success else 207
        return JSONResponse(
            content=result.to_dict(),
            status_code=status_code,
        )
    except Exception as exc:
        logger.error("Backup import failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Import failed: {str(exc)}")


# -----------------------------------------------------------------
# S125: Encrypted backup endpoints
# -----------------------------------------------------------------

class EncryptedExportRequest(BaseModel):
    """Request body for encrypted backup export."""
    password: str = Field(min_length=8, description="Password for encryption")
    sections: str | None = Field(
        default=None,
        description="Comma-separated section names to export (default: all)",
    )


class EncryptedImportRequest(BaseModel):
    """Request body for encrypted backup import."""
    password: str = Field(description="Password used to encrypt the backup")
    encrypted_data: str = Field(description="Base64-encoded encrypted backup data")
    strategy: str = Field(default="merge", description="Import strategy: merge or replace")
    # BK-03 (S194): explicit user override for the BK-01 signature policy.
    allow_unsigned: bool = Field(
        default=False,
        description="Explicitly allow restoring an unsigned/unverifiable backup",
    )


@router.post("/export/encrypted")
def export_encrypted_backup(req: EncryptedExportRequest) -> dict:
    """Export an encrypted backup (S125).

    Encrypts the backup JSON with a user-provided password using
    PBKDF2 key derivation and AES encryption. Returns base64-encoded
    encrypted data for download.
    """
    _check_available()

    try:
        from opti_oignon.backup_manager import encrypt_backup as _encrypt_backup
    except ImportError:
        raise HTTPException(status_code=503, detail="Encryption module not available")

    # Export backup data
    try:
        if req.sections:
            section_list = [s.strip() for s in req.sections.split(",") if s.strip()]
            data = backup_manager.export_sections(section_list)
        else:
            data = backup_manager.export_all()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Backup export failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(exc)}")

    # Encrypt
    try:
        encrypted_bytes = _encrypt_backup(data, req.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Backup encryption failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Encryption failed: {str(exc)}")

    # Return base64-encoded for JSON transport
    encoded = base64.b64encode(encrypted_bytes).decode("ascii")

    return {
        "encrypted": True,
        "data": encoded,
        "filename": "opti-oignon-backup.oo-backup.enc",
        "size_bytes": len(encrypted_bytes),
    }


@router.post("/import/encrypted")
def import_encrypted_backup(req: EncryptedImportRequest) -> JSONResponse:
    """Import an encrypted backup (S125).

    Decrypts the backup with the provided password, then applies
    it using the specified strategy (merge or replace).
    """
    _check_available()

    try:
        from opti_oignon.backup_manager import decrypt_backup as _decrypt_backup
    except ImportError:
        raise HTTPException(status_code=503, detail="Encryption module not available")

    if req.strategy not in ("merge", "replace"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy: '{req.strategy}'. Must be 'merge' or 'replace'.",
        )

    # Decode base64
    try:
        encrypted_bytes = base64.b64decode(req.encrypted_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encrypted data")

    # Decrypt
    try:
        backup_data = _decrypt_backup(encrypted_bytes, req.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Backup decryption failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Decryption failed: {str(exc)}")

    # Import
    try:
        result = backup_manager.import_backup(
            backup_data,
            strategy=req.strategy,
            allow_unsigned=req.allow_unsigned,
        )
        status_code = 200 if result.success else 207
        return JSONResponse(
            content={
                "encrypted": True,
                "decrypted": True,
                **result.to_dict(),
            },
            status_code=status_code,
        )
    except Exception as exc:
        logger.error("Encrypted backup import failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Import failed: {str(exc)}")
