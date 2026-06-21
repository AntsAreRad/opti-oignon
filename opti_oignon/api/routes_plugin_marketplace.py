#!/usr/bin/env python3
"""
Plugin marketplace API routes (S102).

GET    /api/plugins/marketplace           -- Browse available plugins from index
GET    /api/plugins/marketplace/search    -- Search by keyword/tag/author/hook
POST   /api/plugins/marketplace/install   -- Install from remote URL
GET    /api/plugins/{name}/reviews        -- Get reviews for a plugin
POST   /api/plugins/{name}/reviews        -- Add a review
POST   /api/plugins/marketplace/template  -- Generate a plugin scaffold
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# MKT-01: defense-in-depth parity with routes_plugins (S136). The global
# AuthMiddleware already enforces auth on these non-public paths; this
# per-router dependency mirrors routes_plugins so the two /api/plugins
# routers are consistent.
try:
    from .routes_auth import _get_current_user
    _auth_dep = [Depends(_get_current_user)]
except Exception:
    def _get_current_user() -> dict:  # type: ignore[misc]
        """Fallback identity when the auth router is unavailable.

        REV-2 (S219): mirrors the routes_users fallback so the review
        author binding below always has an identity to bind to.
        """
        return {
            "sub": "local",
            "username": "local",
            "role": "admin",
            "type": "access",
        }
    _auth_dep = []

router = APIRouter(
    prefix="/api/plugins",
    tags=["plugin-marketplace"],
    dependencies=_auth_dep,
)


# =========================================================================
# PYDANTIC SCHEMAS
# =========================================================================

class IndexEntryResponse(BaseModel):
    name: str
    version: str
    description: str
    author: str
    url: str = ""
    tags: list[str] = []
    hooks: list[str] = []
    permissions: list[str] = []
    min_opti_version: str = "1.0.0"
    stars: int = 0
    downloads: int = 0
    sha256: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    average_rating: float = 0.0
    review_count: int = 0


class MarketplaceListResponse(BaseModel):
    plugins: list[IndexEntryResponse]
    total: int


class RemoteInstallRequest(BaseModel):
    url: str = Field(..., description="URL to plugin archive or GitHub repo")
    expected_sha256: str = Field("", description="Expected SHA-256 hash (optional)")
    auto_enable: bool = Field(False, description="Auto-enable after install")


class RemoteInstallResponse(BaseModel):
    success: bool
    name: str = ""
    version: str = ""
    message: str = ""
    error: str | None = None


class ReviewResponse(BaseModel):
    id: int
    plugin_name: str
    rating: int
    title: str = ""
    text: str = ""
    author: str = "anonymous"
    created_at: float = 0.0
    # REV-2 (S219): authenticated owner identity; None on legacy rows.
    user_id: str | None = None


class ReviewListResponse(BaseModel):
    reviews: list[ReviewResponse]
    total: int
    average_rating: float = 0.0
    rating_distribution: dict[int, int] = {}


class AddReviewRequest(BaseModel):
    # REV-2 (S219): no client-supplied author field. The author is
    # derived server-side from the authenticated identity; an author
    # key sent by an older client is ignored by the model.
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    title: str = Field("", description="Review title")
    text: str = Field("", description="Review body text")


class AddReviewResponse(BaseModel):
    success: bool
    review: ReviewResponse | None = None
    message: str = ""
    error: str | None = None


class TemplateRequest(BaseModel):
    name: str = Field(..., description="Plugin name (lowercase, alphanumeric + hyphens)")
    author: str = Field("Your Name", description="Plugin author")
    description: str = Field("A custom Opti-Oignon plugin.", description="Short description")
    version: str = Field("1.0.0", description="Initial version")
    hooks: list[str] = Field(default_factory=lambda: ["post_inference"], description="Hook points")
    permissions: list[str] = Field(default_factory=list, description="Permissions to request")


class TemplateResponse(BaseModel):
    success: bool
    path: str = ""
    files: list[str] = []
    message: str = ""
    error: str | None = None


# =========================================================================
# HELPERS
# =========================================================================

def _get_index():
    """Get the plugin index singleton, raise 503 if unavailable."""
    try:
        from opti_oignon.api.deps import (
            PLUGIN_INDEX_AVAILABLE,
            plugin_index_instance,
        )
        if not PLUGIN_INDEX_AVAILABLE or plugin_index_instance is None:
            raise HTTPException(
                status_code=503,
                detail="Plugin index not available",
            )
        return plugin_index_instance
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Plugin marketplace not available",
        )


def _get_installer():
    """Get the remote installer singleton, raise 503 if unavailable."""
    try:
        from opti_oignon.api.deps import (
            PLUGIN_INSTALLER_AVAILABLE,
            remote_installer_instance,
        )
        if not PLUGIN_INSTALLER_AVAILABLE or remote_installer_instance is None:
            raise HTTPException(
                status_code=503,
                detail="Plugin installer not available",
            )
        return remote_installer_instance
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Plugin installer not available",
        )


def _get_review_store():
    """Get the review store singleton, raise 503 if unavailable."""
    try:
        from opti_oignon.api.deps import (
            PLUGIN_REVIEWS_AVAILABLE,
            plugin_review_store_instance,
        )
        if not PLUGIN_REVIEWS_AVAILABLE or plugin_review_store_instance is None:
            raise HTTPException(
                status_code=503,
                detail="Plugin review store not available",
            )
        return plugin_review_store_instance
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Plugin review system not available",
        )


def _get_template_generator():
    """Get the template generator singleton, raise 503 if unavailable."""
    try:
        from opti_oignon.api.deps import (
            PLUGIN_TEMPLATE_AVAILABLE,
            plugin_template_instance,
        )
        if not PLUGIN_TEMPLATE_AVAILABLE or plugin_template_instance is None:
            raise HTTPException(
                status_code=503,
                detail="Plugin template generator not available",
            )
        return plugin_template_instance
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Plugin template system not available",
        )


def _enrich_with_ratings(
    entries: list[Any],
    review_store: Any,
) -> list[IndexEntryResponse]:
    """Convert index entries to response models enriched with rating data."""
    results = []
    for entry in entries:
        d = entry.to_dict() if hasattr(entry, "to_dict") else dict(entry)
        avg_rating = 0.0
        rev_count = 0
        try:
            summary = review_store.get_rating_summary(d["name"])
            avg_rating = summary.average_rating
            rev_count = summary.review_count
        except Exception:
            pass
        results.append(IndexEntryResponse(
            **d,
            average_rating=round(avg_rating, 2),
            review_count=rev_count,
        ))
    return results


# =========================================================================
# ENDPOINTS
# =========================================================================

@router.get("/marketplace", response_model=MarketplaceListResponse)
def browse_marketplace(
    sort_by: str = Query("stars", description="Sort by: name, stars, downloads, updated_at"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    refresh: bool = Query(False, description="Force refresh from remote index"),
) -> dict:
    """Browse available plugins from the marketplace index."""
    index = _get_index()

    # Optionally refresh from remote
    if refresh or index.is_stale:
        try:
            index.refresh_from_remote(force=refresh)
        except Exception as exc:
            logger.warning("Index refresh failed: %s", exc)

    entries = index.list_all(sort_by=sort_by, limit=limit, offset=offset)

    # Enrich with ratings
    try:
        review_store = _get_review_store()
        plugins = _enrich_with_ratings(entries, review_store)
    except Exception:
        plugins = [
            IndexEntryResponse(**e.to_dict(), average_rating=0.0, review_count=0)
            for e in entries
        ]

    return MarketplaceListResponse(
        plugins=plugins,
        total=index.count,
    )


@router.get("/marketplace/search", response_model=MarketplaceListResponse)
def search_marketplace(
    keyword: str = Query("", description="Search in name and description"),
    tag: str = Query("", description="Filter by tag"),
    author: str = Query("", description="Filter by author"),
    hook: str = Query("", description="Filter by hook type"),
    sort_by: str = Query("stars", description="Sort by: name, stars, downloads, updated_at"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
) -> dict:
    """Search the marketplace index by keyword, tag, author, or hook."""
    index = _get_index()

    entries = index.search(
        keyword=keyword,
        tag=tag,
        author=author,
        hook=hook,
        sort_by=sort_by,
        limit=limit,
    )

    # Enrich with ratings
    try:
        review_store = _get_review_store()
        plugins = _enrich_with_ratings(entries, review_store)
    except Exception:
        plugins = [
            IndexEntryResponse(**e.to_dict(), average_rating=0.0, review_count=0)
            for e in entries
        ]

    return MarketplaceListResponse(
        plugins=plugins,
        total=len(plugins),
    )


@router.post("/marketplace/install", response_model=RemoteInstallResponse)
def install_from_url(req: RemoteInstallRequest) -> dict:
    """Install a plugin from a remote URL."""
    installer = _get_installer()

    result = installer.install_from_url(
        req.url,
        expected_sha256=req.expected_sha256,
        auto_enable=req.auto_enable,
    )

    return RemoteInstallResponse(
        success=result["success"],
        name=result.get("name", ""),
        version=result.get("version", ""),
        message=result.get("message", ""),
        error=result.get("error"),
    )


@router.get("/{name}/reviews", response_model=ReviewListResponse)
def get_plugin_reviews(
    name: str,
    sort_by: str = Query("created_at", description="Sort by: created_at, rating"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> dict:
    """Get reviews and rating summary for a plugin."""
    store = _get_review_store()

    reviews = store.get_reviews(
        name, sort_by=sort_by, limit=limit, offset=offset,
    )
    summary = store.get_rating_summary(name)

    return ReviewListResponse(
        reviews=[
            ReviewResponse(**r.to_dict())
            for r in reviews
        ],
        total=summary.review_count,
        average_rating=round(summary.average_rating, 2),
        rating_distribution=summary.rating_distribution,
    )


@router.post("/{name}/reviews", response_model=AddReviewResponse)
def add_plugin_review(
    name: str,
    req: AddReviewRequest,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Add a review for a plugin.

    REV-2 (S219): the review author and owner are bound server-side to
    the authenticated identity (the router already carries the auth
    dependency, MKT-01); clients cannot review under an arbitrary name.
    """
    store = _get_review_store()

    caller_id = str(current_user.get("sub") or "local")
    display_author = str(current_user.get("username") or caller_id)

    try:
        review = store.add_review(
            plugin_name=name,
            rating=req.rating,
            title=req.title,
            text=req.text,
            author=display_author,
            user_id=caller_id,
        )
        return AddReviewResponse(
            success=True,
            review=ReviewResponse(**review.to_dict()),
            message=f"Review added for '{name}'",
        )
    except Exception as exc:
        logger.warning("Failed to add review for '%s': %s", name, exc)
        return AddReviewResponse(
            success=False,
            error=str(exc),
            message=f"Failed to add review: {exc}",
        )


@router.post("/marketplace/template", response_model=TemplateResponse)
def generate_plugin_template(req: TemplateRequest) -> dict:
    """Generate a new plugin scaffold from a template."""
    generator = _get_template_generator()

    result = generator.generate(
        name=req.name,
        author=req.author,
        description=req.description,
        version=req.version,
        hooks=req.hooks,
        permissions=req.permissions,
    )

    return TemplateResponse(
        success=result["success"],
        path=result.get("path", ""),
        files=result.get("files", []),
        message=(
            f"Plugin scaffold generated at {result.get('path', '')}"
            if result["success"]
            else f"Generation failed: {result.get('error', 'unknown')}"
        ),
        error=result.get("error"),
    )
