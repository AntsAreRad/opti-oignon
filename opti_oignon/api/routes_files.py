#!/usr/bin/env python3
"""
File Upload API Routes.

Endpoints for uploading text files and images with extension and
size validation. Returns content for attachment to the next chat
message.
"""

import base64
import logging
import os

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from .schemas import FileUploadResponse, ImageUploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])


# ---------------------------------------------------------------------------
# Rate limiting (S156 -- SA-155-050)
# ---------------------------------------------------------------------------

def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, considering proxy headers."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _check_upload_rate(request: Request) -> None:
    """Rate limit check for file upload endpoints.

    Raises HTTPException 429 if rate limit exceeded.
    """
    try:
        from opti_oignon.rate_limiter import rate_limit_check
    except ImportError:
        return  # Graceful degradation if module not available

    client_ip = _get_client_ip(request)
    allowed, info = rate_limit_check("file_upload", key=client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=info["message"],
            headers={"Retry-After": str(int(info["retry_after"] + 1))},
        )

# Allowed extensions (same whitelist as chat_ui.py)
ALLOWED_EXTENSIONS = {
    ".r", ".R", ".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml",
    ".csv", ".tsv", ".xml", ".html", ".css", ".js", ".ts", ".jsx", ".tsx",
    ".c", ".cpp", ".h", ".java", ".go", ".rs", ".lua", ".rb", ".pl",
    ".toml", ".ini", ".cfg", ".conf", ".log", ".tex", ".bib", ".nf",
}

# S48: Allowed image extensions
ALLOWED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
}

# Correspondance extension -> MIME type pour les images
_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

# Maximum file size (500 KB)
MAX_FILE_SIZE = 500_000

# S48: Maximum image size (10 MB)
MAX_IMAGE_SIZE = 10_000_000


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(request: Request, file: UploadFile = File(...)) -> dict:
    """Upload a text file with validation.

    Accept a multipart file, check extension and size,
    and return the content for attachment to the next message.
    """
    _check_upload_rate(request)

    if file.filename is None:
        raise HTTPException(status_code=422, detail="Filename is required")

    # Validate extension
    _, ext = os.path.splitext(file.filename)
    if ext not in ALLOWED_EXTENSIONS and ext.upper() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file extension: {ext}. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Read content
    try:
        content_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Read error: {e}")

    # Validate file size
    size = len(content_bytes)
    if size > MAX_FILE_SIZE:
        size_kb = size / 1024
        raise HTTPException(
            status_code=422,
            detail=f"File too large: {file.filename} ({size_kb:.0f}KB > "
            f"{MAX_FILE_SIZE // 1024}KB limit)",
        )

    # Decode text content
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content = content_bytes.decode("latin-1")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=422,
                detail="File is not a valid text file (encoding error)",
            )

    return FileUploadResponse(
        filename=file.filename,
        size_bytes=size,
        content=content,
        extension=ext,
    )


# ---------------------------------------------------------------------------
# S48: Image Upload
# ---------------------------------------------------------------------------

@router.post("/upload/image", response_model=ImageUploadResponse)
async def upload_image(request: Request, file: UploadFile = File(...)) -> dict:
    """Upload an image and return base64 data for Ollama.

    Accept a multipart image file, check extension and size,
    and return base64 data without data:... prefix for direct
    usage in ollama.chat()'s ``images`` field.
    """
    _check_upload_rate(request)

    if file.filename is None:
        raise HTTPException(status_code=422, detail="Filename is required")

    # Validate extension
    _, ext = os.path.splitext(file.filename)
    ext_lower = ext.lower()
    if ext_lower not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported image extension: {ext}. "
            f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}",
        )

    # Read content
    try:
        content_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Read error: {e}")

    # Validate file size
    size = len(content_bytes)
    if size > MAX_IMAGE_SIZE:
        size_mb = size / (1024 * 1024)
        raise HTTPException(
            status_code=422,
            detail=f"Image too large: {file.filename} ({size_mb:.1f}MB > "
            f"{MAX_IMAGE_SIZE // (1024 * 1024)}MB limit)",
        )

    # Base64 encoding
    b64_data = base64.b64encode(content_bytes).decode("ascii")

    # MIME type
    mime_type = _IMAGE_MIME_TYPES.get(ext_lower, "image/png")

    return ImageUploadResponse(
        filename=file.filename,
        size_bytes=size,
        base64_data=b64_data,
        mime_type=mime_type,
    )
