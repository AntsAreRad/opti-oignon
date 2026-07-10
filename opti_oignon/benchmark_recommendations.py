#!/usr/bin/env python3
"""
Benchmark Recommendations

Generates automated model recommendations based on benchmark results.
Identifies best models for different use cases (fast, quality, code, value)
and provides one-click apply to push assignments into smart_router config.

Features:
  - Best fast model: highest speed with acceptable quality
  - Best quality model: highest composite score
  - Best code model: highest code evaluation score
  - Best value model: best quality-to-speed ratio
  - Snapshot storage as latest recommendations
  - One-click apply to smart_router configuration
"""

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
# Audit hardening: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)


# ---------------------------------------------------------------------------
# Data directory (same as benchmark_runner)
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_DB_PATH = _DATA_DIR / "benchmark_results.db"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelRecommendation:
    """A single model recommendation for a specific role."""
    role: str = ""
    model: str = ""
    composite_score: float = 0.0
    speed_score: float = 0.0
    accuracy_score: float = 0.0
    code_score: float = 0.0
    structure_score: float = 0.0
    tokens_per_second: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model": self.model,
            "composite_score": self.composite_score,
            "speed_score": self.speed_score,
            "accuracy_score": self.accuracy_score,
            "code_score": self.code_score,
            "structure_score": self.structure_score,
            "tokens_per_second": self.tokens_per_second,
            "reason": self.reason,
        }


@dataclass
class RecommendationSnapshot:
    """Complete set of recommendations from a benchmark analysis."""
    snapshot_id: str = ""
    created_at: float = 0.0
    source_run_ids: list[str] = field(default_factory=list)
    profile: str = ""
    recommendations: list[ModelRecommendation] = field(default_factory=list)
    applied: bool = False
    applied_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "source_run_ids": self.source_run_ids,
            "profile": self.profile,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "applied": self.applied,
            "applied_at": self.applied_at,
        }

    def get_recommendation(self, role: str) -> ModelRecommendation | None:
        """Find a recommendation by role."""
        for r in self.recommendations:
            if r.role == role:
                return r
        return None


# ---------------------------------------------------------------------------
# Recommendation roles
# ---------------------------------------------------------------------------

ROLE_FAST = "fast"
ROLE_QUALITY = "quality"
ROLE_CODE = "code"
ROLE_VALUE = "value"

ALL_ROLES = (ROLE_FAST, ROLE_QUALITY, ROLE_CODE, ROLE_VALUE)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class RecommendationStore:
    """SQLite storage for recommendation snapshots.

    Uses the same database as benchmark_runner to keep results co-located.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or _DEFAULT_DB_PATH)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create recommendations table if it does not exist."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS benchmark_recommendations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        snapshot_id TEXT UNIQUE NOT NULL,
                        created_at REAL NOT NULL,
                        source_run_ids TEXT NOT NULL DEFAULT '[]',
                        profile TEXT NOT NULL DEFAULT '',
                        recommendations TEXT NOT NULL DEFAULT '[]',
                        applied INTEGER DEFAULT 0,
                        applied_at REAL DEFAULT 0
                    );

                    CREATE INDEX IF NOT EXISTS idx_rec_snapshot
                        ON benchmark_recommendations(snapshot_id);
                    CREATE INDEX IF NOT EXISTS idx_rec_created
                        ON benchmark_recommendations(created_at);
                """)
                conn.commit()
            finally:
                conn.close()

    def save_snapshot(self, snapshot: RecommendationSnapshot) -> None:
        """Persist a recommendation snapshot."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO benchmark_recommendations
                       (snapshot_id, created_at, source_run_ids, profile,
                        recommendations, applied, applied_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        snapshot.snapshot_id,
                        snapshot.created_at,
                        json.dumps(snapshot.source_run_ids),
                        snapshot.profile,
                        json.dumps([r.to_dict() for r in snapshot.recommendations]),
                        1 if snapshot.applied else 0,
                        snapshot.applied_at,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_latest(self) -> RecommendationSnapshot | None:
        """Get the most recent recommendation snapshot."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    """SELECT * FROM benchmark_recommendations
                       ORDER BY created_at DESC LIMIT 1""",
                ).fetchone()
                if not row:
                    return None
                return self._row_to_snapshot(row)
            finally:
                conn.close()

    def get_by_id(self, snapshot_id: str) -> RecommendationSnapshot | None:
        """Get a specific recommendation snapshot."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    "SELECT * FROM benchmark_recommendations WHERE snapshot_id = ?",
                    (snapshot_id,),
                ).fetchone()
                if not row:
                    return None
                return self._row_to_snapshot(row)
            finally:
                conn.close()

    def mark_applied(self, snapshot_id: str) -> bool:
        """Mark a snapshot as applied to smart_router."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                cur = conn.execute(
                    """UPDATE benchmark_recommendations
                       SET applied = 1, applied_at = ?
                       WHERE snapshot_id = ?""",
                    (time.time(), snapshot_id),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def get_history(self, limit: int = 20) -> list[RecommendationSnapshot]:
        """Get historical recommendation snapshots."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """SELECT * FROM benchmark_recommendations
                       ORDER BY created_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
                return [self._row_to_snapshot(r) for r in rows]
            finally:
                conn.close()

    @staticmethod
    def _row_to_snapshot(row: Any) -> RecommendationSnapshot:
        """Convert a database row to a RecommendationSnapshot."""
        recs_raw = json.loads(row["recommendations"])
        recs = [
            ModelRecommendation(**r) for r in recs_raw
        ]
        return RecommendationSnapshot(
            snapshot_id=row["snapshot_id"],
            created_at=row["created_at"],
            source_run_ids=json.loads(row["source_run_ids"]),
            profile=row["profile"],
            recommendations=recs,
            applied=bool(row["applied"]),
            applied_at=row["applied_at"],
        )


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

class BenchmarkRecommender:
    """Generates model recommendations from benchmark data.

    Analyzes model scores across recent benchmark runs and identifies
    the best model for each role (fast, quality, code, value).
    """

    def __init__(
        self,
        store: RecommendationStore | None = None,
        db_path: str | Path | None = None,
    ):
        self._store = store or RecommendationStore(db_path)
        self._db_path = str(db_path or _DEFAULT_DB_PATH)

    @property
    def store(self) -> RecommendationStore:
        return self._store

    def generate_from_scores(
        self,
        model_scores: list[dict[str, Any]],
        source_run_ids: list[str] | None = None,
        profile: str = "",
    ) -> RecommendationSnapshot:
        """Generate recommendations from a list of model score dicts.

        Each dict should have keys: model, accuracy_avg (or avg_accuracy),
        code_avg (or avg_code), structure_avg (or avg_structure),
        speed_avg (or avg_speed), composite (or avg_composite).

        Args:
            model_scores: List of per-model score dictionaries.
            source_run_ids: Run IDs that contributed to these scores.
            profile: Profile name used for the benchmark.

        Returns:
            RecommendationSnapshot with best model per role.
        """
        import uuid

        if not model_scores:
            return RecommendationSnapshot(
                snapshot_id=f"rec-{uuid.uuid4().hex[:12]}",
                created_at=time.time(),
                profile=profile,
            )

        # Normalize key names (support both run-level and compare-level keys)
        normalized = []
        for ms in model_scores:
            normalized.append({
                "model": ms.get("model", ""),
                "accuracy": ms.get("accuracy_avg", ms.get("avg_accuracy", 0.0)),
                "code": ms.get("code_avg", ms.get("avg_code", 0.0)),
                "structure": ms.get("structure_avg", ms.get("avg_structure", 0.0)),
                "speed": ms.get("speed_avg", ms.get("avg_speed", 0.0)),
                "composite": ms.get("composite", ms.get("avg_composite", 0.0)),
            })

        recommendations = []

        # Best quality: highest composite
        best_quality = max(normalized, key=lambda x: x["composite"])
        recommendations.append(ModelRecommendation(
            role=ROLE_QUALITY,
            model=best_quality["model"],
            composite_score=best_quality["composite"],
            speed_score=best_quality["speed"],
            accuracy_score=best_quality["accuracy"],
            code_score=best_quality["code"],
            structure_score=best_quality["structure"],
            reason=f"Highest composite score ({best_quality['composite']:.3f})",
        ))

        # Best fast: highest speed with composite >= 50% of best quality
        quality_threshold = best_quality["composite"] * 0.5
        fast_candidates = [
            m for m in normalized if m["composite"] >= quality_threshold
        ]
        if not fast_candidates:
            fast_candidates = normalized
        best_fast = max(fast_candidates, key=lambda x: x["speed"])
        recommendations.append(ModelRecommendation(
            role=ROLE_FAST,
            model=best_fast["model"],
            composite_score=best_fast["composite"],
            speed_score=best_fast["speed"],
            accuracy_score=best_fast["accuracy"],
            code_score=best_fast["code"],
            structure_score=best_fast["structure"],
            reason=f"Fastest with acceptable quality (speed {best_fast['speed']:.3f})",
        ))

        # Best code: highest code score
        best_code = max(normalized, key=lambda x: x["code"])
        recommendations.append(ModelRecommendation(
            role=ROLE_CODE,
            model=best_code["model"],
            composite_score=best_code["composite"],
            speed_score=best_code["speed"],
            accuracy_score=best_code["accuracy"],
            code_score=best_code["code"],
            structure_score=best_code["structure"],
            reason=f"Highest code evaluation score ({best_code['code']:.3f})",
        ))

        # Best value: highest composite * speed product (quality-speed ratio)
        def value_score(m: dict) -> float:
            c = m["composite"]
            s = m["speed"]
            if s <= 0 and c <= 0:
                return 0.0
            return c * (1.0 + s)

        best_value = max(normalized, key=value_score)
        vs = value_score(best_value)
        recommendations.append(ModelRecommendation(
            role=ROLE_VALUE,
            model=best_value["model"],
            composite_score=best_value["composite"],
            speed_score=best_value["speed"],
            accuracy_score=best_value["accuracy"],
            code_score=best_value["code"],
            structure_score=best_value["structure"],
            reason=f"Best quality-speed ratio (value {vs:.3f})",
        ))

        snapshot = RecommendationSnapshot(
            snapshot_id=f"rec-{uuid.uuid4().hex[:12]}",
            created_at=time.time(),
            source_run_ids=source_run_ids or [],
            profile=profile,
            recommendations=recommendations,
        )

        # Persist
        self._store.save_snapshot(snapshot)

        return snapshot

    def generate_from_run(self, run_id: str) -> RecommendationSnapshot | None:
        """Generate recommendations from a single benchmark run.

        Reads model scores from the benchmark_model_scores table.

        Args:
            run_id: The benchmark run to analyze.

        Returns:
            RecommendationSnapshot or None if run not found.
        """
        conn = _safe_connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Get run info
            run_row = conn.execute(
                "SELECT profile FROM benchmark_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if not run_row:
                return None

            # Get model scores
            rows = conn.execute(
                "SELECT * FROM benchmark_model_scores WHERE run_id = ?",
                (run_id,),
            ).fetchall()
            if not rows:
                return None

            model_scores = [dict(r) for r in rows]
            return self.generate_from_scores(
                model_scores,
                source_run_ids=[run_id],
                profile=run_row["profile"],
            )
        finally:
            conn.close()

    def generate_from_history(
        self,
        profile: str | None = None,
        limit: int = 10,
    ) -> RecommendationSnapshot | None:
        """Generate recommendations from aggregated historical runs.

        Uses the compare_models pattern from the benchmark store.

        Args:
            profile: Optional profile filter.
            limit: Number of recent runs to consider.

        Returns:
            RecommendationSnapshot or None if no data.
        """
        conn = _safe_connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Aggregate only over the latest `limit` completed
            # runs. The previous query applied LIMIT to the GROUP BY (capping
            # the number of MODELS) while the aggregate silently spanned the
            # entire history, and source_run_ids then misreported the scope.
            # Resolve the recent-run set once and reuse it for both.
            run_query = """
                SELECT br.run_id
                FROM benchmark_runs br
                WHERE br.status = 'completed'
            """
            run_params: list[Any] = []
            if profile:
                run_query += " AND br.profile = ?"
                run_params.append(profile)
            run_query += " ORDER BY br.started_at DESC"
            if limit:
                run_query += " LIMIT ?"
                run_params.append(limit)

            run_rows = conn.execute(run_query, run_params).fetchall()
            source_ids = [r["run_id"] for r in run_rows]
            if not source_ids:
                return None

            placeholders = ",".join("?" * len(source_ids))
            agg_query = f"""
                SELECT ms.model,
                       AVG(ms.accuracy_avg) as avg_accuracy,
                       AVG(ms.code_avg) as avg_code,
                       AVG(ms.structure_avg) as avg_structure,
                       AVG(ms.speed_avg) as avg_speed,
                       AVG(ms.composite) as avg_composite,
                       COUNT(*) as run_count
                FROM benchmark_model_scores ms
                WHERE ms.run_id IN ({placeholders})
                GROUP BY ms.model ORDER BY avg_composite DESC
            """
            rows = conn.execute(agg_query, source_ids).fetchall()
            if not rows:
                return None

            model_scores = [dict(r) for r in rows]

            return self.generate_from_scores(
                model_scores,
                source_run_ids=source_ids,
                profile=profile or "",
            )
        finally:
            conn.close()

    def get_latest(self) -> RecommendationSnapshot | None:
        """Get the most recent recommendation snapshot."""
        return self._store.get_latest()

    def apply_to_smart_router(
        self,
        snapshot: RecommendationSnapshot | None = None,
    ) -> dict[str, Any]:
        """Apply recommendations to the smart_router configuration.

        Sets the quality model as default_model, adjusts speed_preference
        based on the recommendation roles.

        Args:
            snapshot: Snapshot to apply. Uses latest if None.

        Returns:
            Dictionary describing what was applied.
        """
        if snapshot is None:
            snapshot = self._store.get_latest()
        if snapshot is None:
            return {"applied": False, "error": "No recommendations available"}

        try:
            from opti_oignon.smart_router import smart_router
        except ImportError:
            return {"applied": False, "error": "Smart router not available"}

        if smart_router is None:
            return {"applied": False, "error": "Smart router singleton is None"}

        applied_changes: dict[str, Any] = {
            "applied": True,
            "snapshot_id": snapshot.snapshot_id,
            "changes": {},
        }

        # Apply quality model as default
        quality_rec = snapshot.get_recommendation(ROLE_QUALITY)
        if quality_rec:
            smart_router.configure(default_model=quality_rec.model)
            applied_changes["changes"]["default_model"] = quality_rec.model

        # Mark snapshot as applied
        self._store.mark_applied(snapshot.snapshot_id)
        snapshot.applied = True
        snapshot.applied_at = time.time()

        return applied_changes


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

try:
    recommendation_store = RecommendationStore()
    benchmark_recommender = BenchmarkRecommender(store=recommendation_store)
    BENCHMARK_RECOMMENDATIONS_AVAILABLE = True
except Exception as e:
    logger.warning("BenchmarkRecommender init failed: %s", e)
    recommendation_store = None  # type: ignore[assignment]
    benchmark_recommender = None  # type: ignore[assignment]
    BENCHMARK_RECOMMENDATIONS_AVAILABLE = False
