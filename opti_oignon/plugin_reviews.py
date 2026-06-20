#!/usr/bin/env python3
"""
Plugin review store for Opti-Oignon (S102).

PluginReviewStore: SQLite-backed local ratings (1-5 stars) and text
reviews per plugin. Compute averages, sort by rating/popularity/recency.
"""

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
# S136 audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)



@dataclass
class PluginReview:
    """A single review for a plugin."""

    id: int
    plugin_name: str
    rating: int  # 1-5
    title: str
    text: str
    author: str
    created_at: float
    # REV-2 (S219): authenticated owner identity; None on legacy rows
    # written before the user_id column existed.
    user_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for API responses."""
        return {
            "id": self.id,
            "plugin_name": self.plugin_name,
            "rating": self.rating,
            "title": self.title,
            "text": self.text,
            "author": self.author,
            "created_at": self.created_at,
            "user_id": self.user_id,
        }


@dataclass
class PluginRatingSummary:
    """Aggregated rating info for a plugin."""

    plugin_name: str
    average_rating: float
    review_count: int
    rating_distribution: dict[int, int]  # {1: count, 2: count, ...}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for API responses."""
        return {
            "plugin_name": self.plugin_name,
            "average_rating": round(self.average_rating, 2),
            "review_count": self.review_count,
            "rating_distribution": self.rating_distribution,
        }


class PluginReviewError(Exception):
    """Raised on review validation errors."""


class PluginReviewStore:
    """SQLite-backed review and rating store for plugins.

    Parameters
    ----------
    db_path : Path or str
        Path to the SQLite database file.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # -----------------------------------------------------------------
    # SQLite setup
    # -----------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = _safe_connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plugin_reviews (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    plugin_name TEXT NOT NULL,
                    rating      INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                    title       TEXT NOT NULL DEFAULT '',
                    text        TEXT NOT NULL DEFAULT '',
                    author      TEXT NOT NULL DEFAULT 'anonymous',
                    created_at  REAL NOT NULL DEFAULT 0,
                    user_id     TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reviews_plugin
                ON plugin_reviews (plugin_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reviews_rating
                ON plugin_reviews (rating)
            """)
            # REV-2 (S219): authenticated owner identity. Guarded additive
            # migration, same idiom as veilid/peers.py: legacy rows read
            # NULL (unattributable) and are never matched by the per-user
            # cascade delete by construction.
            cols = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(plugin_reviews)"
                ).fetchall()
            }
            if "user_id" not in cols:
                conn.execute(
                    "ALTER TABLE plugin_reviews ADD COLUMN user_id TEXT"
                )
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reviews_user
                ON plugin_reviews (user_id)
            """)
            conn.commit()
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # Add / delete reviews
    # -----------------------------------------------------------------

    def add_review(
        self,
        plugin_name: str,
        rating: int,
        *,
        title: str = "",
        text: str = "",
        author: str = "anonymous",
        user_id: Optional[str] = None,
    ) -> PluginReview:
        """Add a review for a plugin.

        Parameters
        ----------
        plugin_name : str
            Name of the plugin being reviewed.
        rating : int
            Rating from 1 to 5.
        title : str
            Optional review title.
        text : str
            Optional review body text.
        author : str
            Reviewer display name. REV-2 (S219): the API route derives
            this from the authenticated identity; it is no longer
            client-supplied free text.
        user_id : str, optional
            Authenticated owner identity (REV-2, S219). Scopes the
            review for per-user export and cascade delete. None only
            for legacy or identity-less callers.

        Returns
        -------
        PluginReview
            The created review.

        Raises
        ------
        PluginReviewError
            On validation failure.
        """
        if not plugin_name or not plugin_name.strip():
            raise PluginReviewError("Plugin name is required")

        if not isinstance(rating, int) or rating < 1 or rating > 5:
            raise PluginReviewError(
                f"Rating must be an integer between 1 and 5, got {rating}"
            )

        now = time.time()
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                INSERT INTO plugin_reviews
                    (plugin_name, rating, title, text, author, created_at,
                     user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plugin_name.strip(),
                    rating,
                    title.strip(),
                    text.strip(),
                    author.strip() or "anonymous",
                    now,
                    user_id,
                ),
            )
            conn.commit()
            review_id = cursor.lastrowid
        finally:
            conn.close()

        review = PluginReview(
            id=review_id,
            plugin_name=plugin_name.strip(),
            rating=rating,
            title=title.strip(),
            text=text.strip(),
            author=author.strip() or "anonymous",
            created_at=now,
            user_id=user_id,
        )
        logger.info(
            "Added review for '%s': %d stars by %s",
            plugin_name, rating, review.author,
        )
        return review

    def delete_review(self, review_id: int) -> bool:
        """Delete a review by ID. Returns True if deleted."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM plugin_reviews WHERE id = ?", (review_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # Query reviews
    # -----------------------------------------------------------------

    def get_reviews(
        self,
        plugin_name: str,
        *,
        sort_by: str = "created_at",
        limit: int = 50,
        offset: int = 0,
    ) -> list[PluginReview]:
        """Get reviews for a specific plugin.

        Parameters
        ----------
        plugin_name : str
            Plugin to get reviews for.
        sort_by : str
            Sort column: created_at, rating.
        limit : int
            Max results.
        offset : int
            Pagination offset.
        """
        allowed_sorts = {"created_at", "rating"}
        col = sort_by if sort_by in allowed_sorts else "created_at"

        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM plugin_reviews WHERE plugin_name = ? "
                "ORDER BY {} DESC LIMIT ? OFFSET ?".format(col),
                (plugin_name, limit, offset),
            ).fetchall()
            return [self._row_to_review(r) for r in rows]
        finally:
            conn.close()

    def get_rating_summary(self, plugin_name: str) -> PluginRatingSummary:
        """Get aggregated rating summary for a plugin."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt, COALESCE(AVG(rating), 0) as avg_r "
                "FROM plugin_reviews WHERE plugin_name = ?",
                (plugin_name,),
            ).fetchone()

            count = row["cnt"] if row else 0
            avg = float(row["avg_r"]) if row else 0.0

            # Rating distribution
            dist_rows = conn.execute(
                "SELECT rating, COUNT(*) as cnt FROM plugin_reviews "
                "WHERE plugin_name = ? GROUP BY rating ORDER BY rating",
                (plugin_name,),
            ).fetchall()
            distribution = {i: 0 for i in range(1, 6)}
            for dr in dist_rows:
                distribution[dr["rating"]] = dr["cnt"]

            return PluginRatingSummary(
                plugin_name=plugin_name,
                average_rating=avg,
                review_count=count,
                rating_distribution=distribution,
            )
        finally:
            conn.close()

    def get_top_rated(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Get plugins sorted by average rating (descending).

        Returns list of dicts with plugin_name, average_rating, review_count.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT plugin_name, AVG(rating) as avg_r, COUNT(*) as cnt "
                "FROM plugin_reviews GROUP BY plugin_name "
                "ORDER BY avg_r DESC, cnt DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "plugin_name": r["plugin_name"],
                    "average_rating": round(float(r["avg_r"]), 2),
                    "review_count": r["cnt"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_most_reviewed(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Get plugins sorted by review count (descending).

        Returns list of dicts with plugin_name, average_rating, review_count.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT plugin_name, AVG(rating) as avg_r, COUNT(*) as cnt "
                "FROM plugin_reviews GROUP BY plugin_name "
                "ORDER BY cnt DESC, avg_r DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "plugin_name": r["plugin_name"],
                    "average_rating": round(float(r["avg_r"]), 2),
                    "review_count": r["cnt"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_recent_reviews(self, *, limit: int = 20) -> list[PluginReview]:
        """Get the most recent reviews across all plugins."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM plugin_reviews ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_review(r) for r in rows]
        finally:
            conn.close()

    def delete_reviews_for_plugin(self, plugin_name: str) -> int:
        """Delete all reviews for a plugin. Returns count deleted."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM plugin_reviews WHERE plugin_name = ?",
                (plugin_name,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_reviews_for_user(self, user_id: str) -> list[PluginReview]:
        """All reviews owned by a user (REV-2, S219).

        Feeds the per-user data export (UD-04 family). Legacy rows with
        a NULL user_id are unattributable and never returned.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM plugin_reviews WHERE user_id = ? "
                "ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
            return [self._row_to_review(r) for r in rows]
        finally:
            conn.close()

    def delete_reviews_for_user(self, user_id: str) -> int:
        """Delete all reviews owned by a user. Returns count deleted.

        REV-2 (S219): the per-user cascade-delete hook. Legacy rows with
        a NULL user_id never match the equality predicate, so they stay
        untouched by construction (documented in ATREST_INVENTORY.md).
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM plugin_reviews WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    @property
    def total_reviews(self) -> int:
        """Total number of reviews in the store."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM plugin_reviews"
            ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    @staticmethod
    def _row_to_review(row: sqlite3.Row) -> PluginReview:
        """Convert a database row to a PluginReview."""
        return PluginReview(
            id=row["id"],
            plugin_name=row["plugin_name"],
            rating=row["rating"],
            title=row["title"],
            text=row["text"],
            author=row["author"],
            created_at=row["created_at"],
            user_id=row["user_id"],
        )


# =========================================================================
# Module-level singleton
# =========================================================================

PLUGIN_REVIEWS_AVAILABLE = True

try:
    from opti_oignon.config import DATA_DIR as _DATA_DIR

    _reviews_db = Path(_DATA_DIR) / "plugin_reviews.db"
    plugin_review_store = PluginReviewStore(db_path=_reviews_db)
except Exception as _exc:
    logger.debug("PluginReviewStore singleton init deferred: %s", _exc)
    plugin_review_store = None  # type: ignore[assignment]
