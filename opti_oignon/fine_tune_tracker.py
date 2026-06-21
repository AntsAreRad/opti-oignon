#!/usr/bin/env python3
"""
FINE-TUNE TRACKER -- Variant Management & A/B Comparison (S96)
================================================================

Tracks fine-tuned model variants with their base model mappings,
training metadata, and performance history. Supports A/B comparison
between base and fine-tuned models via side-by-side inference.

SQLite-backed storage for variant registry and comparison results.

Author: Leon
"""

import concurrent.futures
import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# FT-04 (S194): guard the yaml import so a missing PyYAML degrades the
# module instead of breaking its import (VL-02 sibling-consistency class).
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - PyYAML is a core dependency
    yaml = None  # type: ignore[assignment]
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)
# S136 audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)


# =============================================================================
# CONSTANTS
# =============================================================================

_DATA_DIR = Path(__file__).parent / "data"
_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_DB_PATH = _DATA_DIR / "fine_tune_variants.db"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "fine_tune.yaml"

VARIANT_STATUS_ACTIVE = "active"
VARIANT_STATUS_INACTIVE = "inactive"
VARIANT_STATUS_TRAINING = "training"
VALID_STATUSES = {VARIANT_STATUS_ACTIVE, VARIANT_STATUS_INACTIVE, VARIANT_STATUS_TRAINING}

COMPARISON_STATUS_PENDING = "pending"
COMPARISON_STATUS_RUNNING = "running"
COMPARISON_STATUS_COMPLETED = "completed"
COMPARISON_STATUS_FAILED = "failed"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FineTuneVariant:
    """A fine-tuned model variant linked to a base model."""

    variant_id: str = ""
    name: str = ""
    base_model: str = ""
    variant_model: str = ""
    status: str = VARIANT_STATUS_ACTIVE
    created_at: str = ""
    updated_at: str = ""
    description: str = ""

    # Training metadata
    dataset_size: int = 0
    epochs: int = 0
    learning_rate: float = 0.0
    loss: float = 0.0
    training_duration_seconds: float = 0.0

    # Extra metadata (JSON-serialized)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.variant_id:
            self.variant_id = str(uuid.uuid4())[:12]
        now = datetime.utcnow().isoformat() + "Z"
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FineTuneVariant":
        """Create from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class ComparisonPrompt:
    """A single prompt used in an A/B comparison."""

    prompt: str = ""
    base_response: str = ""
    variant_response: str = ""
    base_latency_ms: float = 0.0
    variant_latency_ms: float = 0.0
    winner: str = ""  # "base", "variant", or "tie"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonResult:
    """Result of an A/B comparison between base and fine-tuned models."""

    comparison_id: str = ""
    variant_id: str = ""
    base_model: str = ""
    variant_model: str = ""
    status: str = COMPARISON_STATUS_PENDING
    created_at: str = ""
    completed_at: str = ""
    prompts: list[ComparisonPrompt] = field(default_factory=list)
    base_avg_latency_ms: float = 0.0
    variant_avg_latency_ms: float = 0.0
    base_wins: int = 0
    variant_wins: int = 0
    ties: int = 0
    summary: str = ""

    def __post_init__(self):
        if not self.comparison_id:
            self.comparison_id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = asdict(self)
        d["prompts"] = [p.to_dict() if hasattr(p, "to_dict") else p for p in self.prompts]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComparisonResult":
        """Create from dictionary."""
        prompts_raw = data.pop("prompts", [])
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        result = cls(**filtered)
        result.prompts = [
            ComparisonPrompt(**p) if isinstance(p, dict) else p
            for p in prompts_raw
        ]
        return result


# =============================================================================
# FINE-TUNE TRACKER
# =============================================================================

class FineTuneTracker:
    """Manages fine-tuned model variants and A/B comparisons.

    SQLite-backed registry for base -> fine-tuned model mappings,
    training metadata, and comparison history.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        config_path: Path | None = None,
    ):
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._lock = threading.Lock()
        self._config: dict[str, Any] = {}

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_config()
        self._init_db()

    def _load_config(self) -> None:
        """Load tracking configuration from YAML."""
        try:
            if YAML_AVAILABLE and self._config_path.exists():
                with open(self._config_path, encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Failed to load fine-tune tracker config: %s", exc)
            self._config = {}

    @property
    def enabled(self) -> bool:
        """Whether variant tracking is enabled."""
        return self._config.get("tracking", {}).get("enabled", True)

    @property
    def default_comparison_prompts(self) -> int:
        """Default number of prompts for A/B comparison."""
        return self._config.get("tracking", {}).get("default_comparison_prompts", 5)

    @property
    def comparison_timeout(self) -> int:
        """Timeout in seconds for model responses during comparison."""
        return self._config.get("tracking", {}).get("comparison_timeout", 120)

    def _get_conn(self) -> sqlite3.Connection:
        """Create a new SQLite connection with row factory."""
        conn = _safe_connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS variants (
                        variant_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        base_model TEXT NOT NULL,
                        variant_model TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'active',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        description TEXT DEFAULT '',
                        dataset_size INTEGER DEFAULT 0,
                        epochs INTEGER DEFAULT 0,
                        learning_rate REAL DEFAULT 0.0,
                        loss REAL DEFAULT 0.0,
                        training_duration_seconds REAL DEFAULT 0.0,
                        metadata TEXT DEFAULT '{}'
                    );

                    CREATE INDEX IF NOT EXISTS idx_variants_base
                        ON variants(base_model);
                    CREATE INDEX IF NOT EXISTS idx_variants_status
                        ON variants(status);

                    CREATE TABLE IF NOT EXISTS comparisons (
                        comparison_id TEXT PRIMARY KEY,
                        variant_id TEXT NOT NULL,
                        base_model TEXT NOT NULL,
                        variant_model TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        created_at TEXT NOT NULL,
                        completed_at TEXT DEFAULT '',
                        prompts_json TEXT DEFAULT '[]',
                        base_avg_latency_ms REAL DEFAULT 0.0,
                        variant_avg_latency_ms REAL DEFAULT 0.0,
                        base_wins INTEGER DEFAULT 0,
                        variant_wins INTEGER DEFAULT 0,
                        ties INTEGER DEFAULT 0,
                        summary TEXT DEFAULT '',
                        FOREIGN KEY (variant_id) REFERENCES variants(variant_id)
                            ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_comparisons_variant
                        ON comparisons(variant_id);
                """)
                conn.commit()
            finally:
                conn.close()

    # =========================================================================
    # VARIANT CRUD
    # =========================================================================

    def register_variant(self, variant: FineTuneVariant) -> FineTuneVariant:
        """Register a new fine-tuned variant.

        Args:
            variant: FineTuneVariant to register.

        Returns:
            The registered variant with generated ID.

        Raises:
            ValueError: If required fields are missing or variant_model already exists.
        """
        if not variant.name:
            raise ValueError("Variant name is required")
        if not variant.base_model:
            raise ValueError("Base model is required")
        if not variant.variant_model:
            raise ValueError("Variant model name is required")

        with self._lock:
            conn = self._get_conn()
            try:
                # Check for duplicate variant_model
                existing = conn.execute(
                    "SELECT variant_id FROM variants WHERE variant_model = ?",
                    (variant.variant_model,),
                ).fetchone()
                if existing:
                    raise ValueError(
                        f"Variant model '{variant.variant_model}' is already registered "
                        f"(id: {existing['variant_id']})"
                    )

                metadata_json = json.dumps(variant.metadata, ensure_ascii=False)
                conn.execute(
                    """INSERT INTO variants (
                        variant_id, name, base_model, variant_model, status,
                        created_at, updated_at, description,
                        dataset_size, epochs, learning_rate, loss,
                        training_duration_seconds, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        variant.variant_id,
                        variant.name,
                        variant.base_model,
                        variant.variant_model,
                        variant.status,
                        variant.created_at,
                        variant.updated_at,
                        variant.description,
                        variant.dataset_size,
                        variant.epochs,
                        variant.learning_rate,
                        variant.loss,
                        variant.training_duration_seconds,
                        metadata_json,
                    ),
                )
                conn.commit()
                return variant
            finally:
                conn.close()

    def get_variant(self, variant_id: str) -> FineTuneVariant | None:
        """Get a variant by ID."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM variants WHERE variant_id = ?",
                    (variant_id,),
                ).fetchone()
                return self._row_to_variant(row) if row else None
            finally:
                conn.close()

    def list_variants(
        self,
        base_model: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[FineTuneVariant]:
        """List registered variants with optional filters.

        Args:
            base_model: Filter by base model name.
            status: Filter by status.
            limit: Maximum results to return.

        Returns:
            List of FineTuneVariant objects.
        """
        clauses = []
        params: list[Any] = []

        if base_model:
            clauses.append("base_model = ?")
            params.append(base_model)
        if status:
            clauses.append("status = ?")
            params.append(status)

        where = "WHERE {}".format(" AND ".join(clauses)) if clauses else ""
        params.append(limit)

        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    f"SELECT * FROM variants {where} ORDER BY created_at DESC LIMIT ?",
                    params,
                ).fetchall()
                return [self._row_to_variant(r) for r in rows]
            finally:
                conn.close()

    def update_variant(
        self,
        variant_id: str,
        updates: dict[str, Any],
    ) -> FineTuneVariant | None:
        """Update variant fields.

        Args:
            variant_id: ID of variant to update.
            updates: Dictionary of field -> value pairs to update.

        Returns:
            Updated variant, or None if not found.
        """
        allowed_fields = {
            "name", "status", "description", "dataset_size", "epochs",
            "learning_rate", "loss", "training_duration_seconds", "metadata",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed_fields}
        if not filtered:
            return self.get_variant(variant_id)

        # Handle metadata serialization
        if "metadata" in filtered:
            filtered["metadata"] = json.dumps(filtered["metadata"], ensure_ascii=False)

        filtered["updated_at"] = datetime.utcnow().isoformat() + "Z"

        set_clause = ", ".join(f"{k} = ?" for k in filtered)
        params = list(filtered.values()) + [variant_id]

        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    f"UPDATE variants SET {set_clause} WHERE variant_id = ?",
                    params,
                )
                conn.commit()
                if cursor.rowcount == 0:
                    return None
            finally:
                conn.close()

        return self.get_variant(variant_id)

    def unregister_variant(self, variant_id: str) -> bool:
        """Unregister (delete) a variant and its comparison history.

        Args:
            variant_id: ID of variant to remove.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    "DELETE FROM variants WHERE variant_id = ?",
                    (variant_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    # =========================================================================
    # A/B COMPARISON
    # =========================================================================

    def create_comparison(
        self,
        variant_id: str,
        prompts: list[str],
    ) -> ComparisonResult:
        """Create a new A/B comparison record.

        Args:
            variant_id: ID of the variant to compare against its base.
            prompts: List of prompt strings to test.

        Returns:
            ComparisonResult with pending status.

        Raises:
            ValueError: If variant not found or prompts empty.
        """
        variant = self.get_variant(variant_id)
        if variant is None:
            raise ValueError(f"Variant '{variant_id}' not found")
        if not prompts:
            raise ValueError("At least one prompt is required for comparison")

        comparison = ComparisonResult(
            variant_id=variant_id,
            base_model=variant.base_model,
            variant_model=variant.variant_model,
            status=COMPARISON_STATUS_PENDING,
            prompts=[ComparisonPrompt(prompt=p) for p in prompts],
        )

        prompts_json = json.dumps(
            [p.to_dict() for p in comparison.prompts],
            ensure_ascii=False,
        )

        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO comparisons (
                        comparison_id, variant_id, base_model, variant_model,
                        status, created_at, prompts_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        comparison.comparison_id,
                        comparison.variant_id,
                        comparison.base_model,
                        comparison.variant_model,
                        comparison.status,
                        comparison.created_at,
                        prompts_json,
                    ),
                )
                conn.commit()
                return comparison
            finally:
                conn.close()

    def run_comparison(
        self,
        comparison_id: str,
        inference_fn: Any = None,
    ) -> ComparisonResult:
        """Execute an A/B comparison by querying both models.

        Args:
            comparison_id: ID of the comparison to run.
            inference_fn: Callable(model, prompt) -> str for model inference.
                If None, comparison is marked as failed.

        Returns:
            Updated ComparisonResult with responses and stats.
        """
        comparison = self.get_comparison(comparison_id)
        if comparison is None:
            raise ValueError(f"Comparison '{comparison_id}' not found")

        self._update_comparison_status(comparison_id, COMPARISON_STATUS_RUNNING)

        if inference_fn is None:
            self._update_comparison_status(
                comparison_id, COMPARISON_STATUS_FAILED,
                summary="No inference function provided",
            )
            comparison.status = COMPARISON_STATUS_FAILED
            comparison.summary = "No inference function provided"
            return comparison

        updated_prompts = []
        base_latencies = []
        variant_latencies = []

        for prompt_entry in comparison.prompts:
            prompt_text = prompt_entry.prompt

            # Query base model
            base_resp, base_lat = self._timed_inference(
                inference_fn, comparison.base_model, prompt_text
            )
            prompt_entry.base_response = base_resp
            prompt_entry.base_latency_ms = base_lat
            base_latencies.append(base_lat)

            # Query variant model
            var_resp, var_lat = self._timed_inference(
                inference_fn, comparison.variant_model, prompt_text
            )
            prompt_entry.variant_response = var_resp
            prompt_entry.variant_latency_ms = var_lat
            variant_latencies.append(var_lat)

            # Simple heuristic: longer response wins (placeholder for LLM judge)
            if len(var_resp) > len(base_resp) * 1.1:
                prompt_entry.winner = "variant"
            elif len(base_resp) > len(var_resp) * 1.1:
                prompt_entry.winner = "base"
            else:
                prompt_entry.winner = "tie"

            updated_prompts.append(prompt_entry)

        # Compute stats
        comparison.prompts = updated_prompts
        comparison.base_avg_latency_ms = (
            sum(base_latencies) / len(base_latencies) if base_latencies else 0.0
        )
        comparison.variant_avg_latency_ms = (
            sum(variant_latencies) / len(variant_latencies) if variant_latencies else 0.0
        )
        comparison.base_wins = sum(1 for p in updated_prompts if p.winner == "base")
        comparison.variant_wins = sum(1 for p in updated_prompts if p.winner == "variant")
        comparison.ties = sum(1 for p in updated_prompts if p.winner == "tie")
        comparison.status = COMPARISON_STATUS_COMPLETED
        comparison.completed_at = datetime.utcnow().isoformat() + "Z"

        # Generate summary
        total = len(updated_prompts)
        comparison.summary = (
            f"Base wins: {comparison.base_wins}/{total}, "
            f"Variant wins: {comparison.variant_wins}/{total}, "
            f"Ties: {comparison.ties}/{total}. "
            f"Avg latency - Base: {comparison.base_avg_latency_ms:.0f}ms, "
            f"Variant: {comparison.variant_avg_latency_ms:.0f}ms."
        )

        # Persist results
        self._save_comparison(comparison)
        return comparison

    def get_comparison(self, comparison_id: str) -> ComparisonResult | None:
        """Get a comparison by ID."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM comparisons WHERE comparison_id = ?",
                    (comparison_id,),
                ).fetchone()
                return self._row_to_comparison(row) if row else None
            finally:
                conn.close()

    def list_comparisons(
        self,
        variant_id: str | None = None,
        limit: int = 20,
    ) -> list[ComparisonResult]:
        """List comparison results with optional variant filter."""
        params: list[Any] = []
        where = ""
        if variant_id:
            where = "WHERE variant_id = ?"
            params.append(variant_id)
        params.append(limit)

        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    f"SELECT * FROM comparisons {where} ORDER BY created_at DESC LIMIT ?",
                    params,
                ).fetchall()
                return [self._row_to_comparison(r) for r in rows]
            finally:
                conn.close()

    def get_variant_stats(self, variant_id: str) -> dict[str, Any]:
        """Get aggregated stats for a variant across all comparisons.

        Returns:
            Dictionary with win rates, average latencies, and comparison count.
        """
        comparisons = self.list_comparisons(variant_id=variant_id, limit=100)
        completed = [c for c in comparisons if c.status == COMPARISON_STATUS_COMPLETED]

        if not completed:
            return {
                "variant_id": variant_id,
                "comparison_count": 0,
                "total_prompts": 0,
                "base_win_rate": 0.0,
                "variant_win_rate": 0.0,
                "tie_rate": 0.0,
                "avg_base_latency_ms": 0.0,
                "avg_variant_latency_ms": 0.0,
            }

        total_base_wins = sum(c.base_wins for c in completed)
        total_variant_wins = sum(c.variant_wins for c in completed)
        total_ties = sum(c.ties for c in completed)
        total_prompts = total_base_wins + total_variant_wins + total_ties

        avg_base_lat = (
            sum(c.base_avg_latency_ms for c in completed) / len(completed)
        )
        avg_variant_lat = (
            sum(c.variant_avg_latency_ms for c in completed) / len(completed)
        )

        return {
            "variant_id": variant_id,
            "comparison_count": len(completed),
            "total_prompts": total_prompts,
            "base_win_rate": total_base_wins / total_prompts if total_prompts else 0.0,
            "variant_win_rate": total_variant_wins / total_prompts if total_prompts else 0.0,
            "tie_rate": total_ties / total_prompts if total_prompts else 0.0,
            "avg_base_latency_ms": round(avg_base_lat, 1),
            "avg_variant_latency_ms": round(avg_variant_lat, 1),
        }

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _timed_inference(
        self,
        inference_fn: Any,
        model: str,
        prompt: str,
    ) -> tuple[str, float]:
        """Run inference with timing, bounded by `comparison_timeout`.

        FT-03 (S194): the call runs in a worker thread and is abandoned
        past the configured timeout. The thread itself cannot be killed
        and may keep holding a backend slot until the model returns, but
        the comparison no longer blocks on it.

        Returns:
            Tuple of (response_text, latency_ms).
        """
        timeout = self.comparison_timeout
        start = time.perf_counter()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(inference_fn, model, prompt)
            response = future.result(timeout=timeout)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return str(response), elapsed_ms
        except concurrent.futures.TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "Inference timed out after %ss for %s", timeout, model
            )
            return f"[Error: timeout after {timeout}s]", elapsed_ms
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("Inference failed for %s: %s", model, exc)
            return f"[Error: {exc}]", elapsed_ms
        finally:
            executor.shutdown(wait=False)

    def _update_comparison_status(
        self,
        comparison_id: str,
        status: str,
        summary: str = "",
    ) -> None:
        """Update comparison status in database."""
        with self._lock:
            conn = self._get_conn()
            try:
                params: list[Any] = [status]
                set_parts = ["status = ?"]
                if summary:
                    set_parts.append("summary = ?")
                    params.append(summary)
                if status == COMPARISON_STATUS_COMPLETED:
                    set_parts.append("completed_at = ?")
                    params.append(datetime.utcnow().isoformat() + "Z")
                params.append(comparison_id)
                conn.execute(
                    "UPDATE comparisons SET {} WHERE comparison_id = ?".format(
                        ", ".join(set_parts)
                    ),
                    params,
                )
                conn.commit()
            finally:
                conn.close()

    def _save_comparison(self, comparison: ComparisonResult) -> None:
        """Persist full comparison result to database."""
        prompts_json = json.dumps(
            [p.to_dict() for p in comparison.prompts],
            ensure_ascii=False,
        )
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE comparisons SET
                        status = ?,
                        completed_at = ?,
                        prompts_json = ?,
                        base_avg_latency_ms = ?,
                        variant_avg_latency_ms = ?,
                        base_wins = ?,
                        variant_wins = ?,
                        ties = ?,
                        summary = ?
                    WHERE comparison_id = ?""",
                    (
                        comparison.status,
                        comparison.completed_at,
                        prompts_json,
                        comparison.base_avg_latency_ms,
                        comparison.variant_avg_latency_ms,
                        comparison.base_wins,
                        comparison.variant_wins,
                        comparison.ties,
                        comparison.summary,
                        comparison.comparison_id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def _row_to_variant(self, row: sqlite3.Row) -> FineTuneVariant:
        """Convert a database row to FineTuneVariant."""
        metadata = {}
        try:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            pass

        return FineTuneVariant(
            variant_id=row["variant_id"],
            name=row["name"],
            base_model=row["base_model"],
            variant_model=row["variant_model"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            description=row["description"] or "",
            dataset_size=row["dataset_size"] or 0,
            epochs=row["epochs"] or 0,
            learning_rate=row["learning_rate"] or 0.0,
            loss=row["loss"] or 0.0,
            training_duration_seconds=row["training_duration_seconds"] or 0.0,
            metadata=metadata,
        )

    def _row_to_comparison(self, row: sqlite3.Row) -> ComparisonResult:
        """Convert a database row to ComparisonResult."""
        prompts = []
        try:
            raw = json.loads(row["prompts_json"]) if row["prompts_json"] else []
            prompts = [ComparisonPrompt(**p) for p in raw]
        except (json.JSONDecodeError, TypeError):
            pass

        return ComparisonResult(
            comparison_id=row["comparison_id"],
            variant_id=row["variant_id"],
            base_model=row["base_model"],
            variant_model=row["variant_model"],
            status=row["status"],
            created_at=row["created_at"],
            completed_at=row["completed_at"] or "",
            prompts=prompts,
            base_avg_latency_ms=row["base_avg_latency_ms"] or 0.0,
            variant_avg_latency_ms=row["variant_avg_latency_ms"] or 0.0,
            base_wins=row["base_wins"] or 0,
            variant_wins=row["variant_wins"] or 0,
            ties=row["ties"] or 0,
            summary=row["summary"] or "",
        )


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

FINE_TUNE_TRACKER_AVAILABLE = True

try:
    fine_tune_tracker = FineTuneTracker()
except Exception as exc:
    logger.warning("Failed to initialize FineTuneTracker: %s", exc)
    fine_tune_tracker = None
    FINE_TUNE_TRACKER_AVAILABLE = False
