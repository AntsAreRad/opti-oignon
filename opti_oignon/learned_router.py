#!/usr/bin/env python3
"""
LEARNED ROUTER -- ML-based Query Classification (S67)
======================================================

Supplements the YAML heuristic router with a sklearn classifier trained
on actual query history. When the model predicts with confidence above
the configured threshold, its prediction overrides the YAML router.
Below threshold, the YAML router is used as fallback.

Training data is collected automatically from routing decisions and
stored in a local SQLite table. The fitted model is persisted to disk
with joblib so retraining is not required on every restart.

Architecture:
    - LearnedRouter: main class (singleton via module-level `learned_router`)
    - TrainingResult: dataclass for train() return value
    - RoutingPrediction: dataclass for classify() return value
    - LEARNED_ROUTER_AVAILABLE: flag (False when sklearn/joblib missing)

Author: Leon
"""

import hashlib
import hmac
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)
# S136 audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)


# =============================================================================
# CONDITIONAL IMPORTS
# =============================================================================

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    joblib = None
    LogisticRegression = None
    RandomForestClassifier = None
    TfidfVectorizer = None
    cross_val_score = None
    Pipeline = None

# Whether this module is fully operational
LEARNED_ROUTER_AVAILABLE = SKLEARN_AVAILABLE

# =============================================================================
# CONSTANTS
# =============================================================================

_DATA_DIR = Path(__file__).parent / "data"
_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "learned_routing.yaml"
_DEFAULT_DB_PATH = _DATA_DIR / "learned_router.db"
_DEFAULT_MODEL_PATH = _DATA_DIR / "learned_router.pkl"


# ---------------------------------------------------------------------------
# Persisted-model integrity (LR-01, S185)
#
# joblib.load is pickle deserialization: loading a tampered or swapped
# learned_router.pkl is arbitrary code execution at init. The file is plaintext
# at rest (not a SQLCipher store) and writable by anything with FS access
# (backup-restore, a future Veilid sync of the data dir, an accidental commit).
# Before loading we verify a keyed MAC over the file (HMAC off the master key,
# the same construction as the SQLCipher subkey) and refuse on mismatch,
# missing MAC, or no key. The load fails safe -- without a key the model is not
# loaded (the router falls back to its heuristic) and is never deserialized
# blind. Kerckhoffs-clean: only the master key is secret; the derivation is open.
# ---------------------------------------------------------------------------

_MODEL_MAC_INFO = b"oo-learned-router-mac-v1"


def _model_mac_path(model_path) -> Path:
    """Sidecar path holding the keyed MAC for a persisted model."""
    return Path(f"{model_path}.mac")


def _router_master_key() -> bytes | None:
    """Return the master encryption key bytes, or None if unavailable.

    Isolated so the no-key path is explicit and testable.
    """
    try:
        from opti_oignon.encryption import get_encryption_key

        master = get_encryption_key()
        if not master:
            return None
        return master.as_bytes() if hasattr(master, "as_bytes") else master
    except Exception:
        return None


def _derive_model_mac_subkey(master_key: bytes | None) -> bytes | None:
    """Derive the learned-router MAC subkey from the master key (HMAC-SHA256).

    Domain-separated from the SQLCipher subkey (distinct info string).
    """
    if not master_key:
        return None
    return hmac.new(master_key, _MODEL_MAC_INFO, hashlib.sha256).digest()


def _compute_model_mac(model_path, subkey: bytes) -> str:
    with open(model_path, "rb") as fh:
        data = fh.read()
    return hmac.new(subkey, data, hashlib.sha256).hexdigest()


def write_model_mac(model_path, mac_path, master_key: bytes | None) -> bool:
    """Write a keyed MAC sidecar for a persisted model.

    Returns False (and writes nothing) when no key is available.
    """
    subkey = _derive_model_mac_subkey(master_key)
    if subkey is None:
        return False
    Path(mac_path).write_text(_compute_model_mac(model_path, subkey), encoding="utf-8")
    return True


def verify_model_mac(model_path, mac_path, master_key: bytes | None) -> bool:
    """Verify a model's keyed MAC before any deserialization.

    Returns False (fail-safe) when no key is available, no MAC sidecar exists,
    the files cannot be read, or the MAC does not match -- so a tampered or
    unauthenticated pickle is never loaded.
    """
    subkey = _derive_model_mac_subkey(master_key)
    if subkey is None:
        return False
    mac_p = Path(mac_path)
    if not mac_p.exists():
        return False
    try:
        stored = mac_p.read_text(encoding="utf-8").strip()
        actual = _compute_model_mac(model_path, subkey)
    except OSError:
        return False
    return hmac.compare_digest(stored, actual)

# Valid task types the classifier can predict (matches analyzer.TaskType labels)
KNOWN_TASK_TYPES = [
    "code_python",
    "code_r",
    "debug",
    "refactor",
    "reasoning",
    "planning",
    "analysis",
    "scientific_writing",
    "general",
    "quick_answer",
    "tool_use",
    "mathematical",
    "creative",
    "data_analysis",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrainingResult:
    """Result returned by LearnedRouter.train()."""
    accuracy: float = 0.0           # Cross-validated accuracy (0-1)
    n_samples: int = 0              # Number of training samples used
    n_classes: int = 0              # Number of distinct task types in training data
    trained_at: float = 0.0         # Unix timestamp of training completion
    model_type: str = ""            # "logistic" or "random_forest"
    cv_folds: int = 5               # Cross-validation folds used
    success: bool = False           # Whether training succeeded
    error: str = ""                 # Error message if success=False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "accuracy": round(self.accuracy, 4),
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "trained_at": self.trained_at,
            "model_type": self.model_type,
            "cv_folds": self.cv_folds,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class RoutingPrediction:
    """Result returned by LearnedRouter.classify()."""
    task_type: str = ""             # Predicted task type
    confidence: float = 0.0        # Confidence score (0-1, max class probability)
    model_type: str = ""           # Classifier type used
    fallback_used: bool = False    # Whether YAML fallback was applied
    top_classes: list[dict] = field(default_factory=list)  # Top-3 predictions

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_type": self.task_type,
            "confidence": round(self.confidence, 4),
            "model_type": self.model_type,
            "fallback_used": self.fallback_used,
            "top_classes": self.top_classes,
        }


# =============================================================================
# LEARNED ROUTER
# =============================================================================

class LearnedRouter:
    """ML-based query classifier that supplements the YAML heuristic router.

    Maintains its own SQLite table of (query_text, task_type) training
    samples. When sklearn is available and the model has been trained,
    classify_with_fallback() uses ML prediction if confidence is above
    the configured threshold, otherwise delegates to the YAML router.

    Usage:
        router = LearnedRouter()
        router.log_sample("how do I fix this R error", "debug")
        result = router.train()
        prediction = router.classify("write a python function")
    """

    def __init__(
        self,
        config_path: Path | None = None,
        db_path: Path | None = None,
        model_path: Path | None = None,
    ):
        """Initialize the learned router.

        Args:
            config_path: Path to learned_routing.yaml (None = default).
            db_path: Path to SQLite training store (None = default).
            model_path: Path to joblib model file (None = default).
        """
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._config: dict[str, Any] = {}
        self._pipeline: Any = None          # Fitted sklearn Pipeline
        self._last_training: TrainingResult | None = None
        self._samples_since_retrain: int = 0

        self._load_config()
        self._init_db()

        if SKLEARN_AVAILABLE:
            self._try_load_model()

        logger.info(
            "LearnedRouter initialized (sklearn=%s, trained=%s)",
            SKLEARN_AVAILABLE,
            self.is_trained,
        )

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load YAML configuration with defaults."""
        defaults: dict[str, Any] = {
            "enabled": False,
            "model_type": "logistic",
            "confidence_threshold": 0.70,
            "min_training_samples": 50,
            "auto_retrain_interval": 100,
            "feature_max_features": 5000,
            "feature_ngram_range": [1, 2],
            "logistic_max_iter": 1000,
            "logistic_C": 1.0,
            "random_forest_n_estimators": 100,
            "random_forest_max_depth": None,
            "max_stored_samples": 10000,
            "cv_folds": 5,
        }
        try:
            if self._config_path.exists():
                with open(self._config_path, encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh) or {}
                defaults.update(loaded)
        except Exception as exc:
            logger.warning("Could not load learned_routing.yaml: %s", exc)
        self._config = defaults

    def get_config(self) -> dict[str, Any]:
        """Return current configuration dict."""
        return dict(self._config)

    def update_config(self, updates: dict[str, Any]) -> None:
        """Apply partial config updates and persist to YAML.

        Args:
            updates: Dict of keys to update.
        """
        self._config.update(updates)
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(self._config, fh, default_flow_style=False)
        except Exception as exc:
            logger.warning("Could not persist learned_routing.yaml: %s", exc)

    # -------------------------------------------------------------------------
    # Database -- training sample store
    # -------------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Open a WAL-mode SQLite connection."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = _safe_connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Create training_samples table if it does not exist."""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    source TEXT DEFAULT 'router'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ts_task
                ON training_samples(task_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ts_timestamp
                ON training_samples(timestamp)
            """)
            # Routing decisions log (for A/B metrics)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS routing_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    ml_task_type TEXT DEFAULT '',
                    ml_confidence REAL DEFAULT 0.0,
                    yaml_task_type TEXT DEFAULT '',
                    routing_source TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rd_timestamp
                ON routing_decisions(timestamp)
            """)
            conn.commit()
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Training data management
    # -------------------------------------------------------------------------

    def log_sample(
        self,
        query_text: str,
        task_type: str,
        source: str = "router",
    ) -> None:
        """Add a labeled training sample to the store.

        Called by the executor/smart_router after every routing decision
        so the classifier accumulates real usage data over time.

        Args:
            query_text: Raw query string.
            task_type: Ground-truth task type label.
            source: Origin of the label ('router', 'user_feedback', etc.)
        """
        if not query_text or not task_type:
            return
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO training_samples (query_text, task_type, timestamp, source) "
                "VALUES (?, ?, ?, ?)",
                (query_text.strip(), task_type, time.time(), source),
            )
            conn.commit()
            self._samples_since_retrain += 1
            self._maybe_prune(conn)
        except Exception as exc:
            logger.debug("log_sample failed: %s", exc)
        finally:
            conn.close()

    def _maybe_prune(self, conn: sqlite3.Connection) -> None:
        """Prune oldest samples if store exceeds max_stored_samples."""
        max_samples = self._config.get("max_stored_samples", 10000)
        try:
            row = conn.execute("SELECT COUNT(*) FROM training_samples").fetchone()
            count = row[0] if row else 0
            if count > max_samples:
                excess = count - max_samples
                conn.execute(
                    "DELETE FROM training_samples WHERE id IN "
                    "(SELECT id FROM training_samples ORDER BY timestamp ASC LIMIT ?)",
                    (excess,),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("_maybe_prune failed: %s", exc)

    def get_sample_count(self) -> int:
        """Return total number of stored training samples."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) FROM training_samples").fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
        finally:
            conn.close()

    def get_class_distribution(self) -> dict[str, int]:
        """Return count of samples per task_type."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT task_type, COUNT(*) as cnt FROM training_samples "
                "GROUP BY task_type ORDER BY cnt DESC"
            ).fetchall()
            return {row["task_type"]: row["cnt"] for row in rows}
        except Exception:
            return {}
        finally:
            conn.close()

    def _load_training_data(self) -> tuple[list[str], list[str]]:
        """Load all (query_text, task_type) pairs from the DB.

        Returns:
            Tuple of (texts, labels).
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT query_text, task_type FROM training_samples "
                "ORDER BY timestamp ASC"
            ).fetchall()
            texts = [row["query_text"] for row in rows]
            labels = [row["task_type"] for row in rows]
            return texts, labels
        except Exception:
            return [], []
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Model training
    # -------------------------------------------------------------------------

    def _build_pipeline(self) -> Any:
        """Construct a fresh sklearn Pipeline based on current config.

        Returns:
            Unfitted sklearn Pipeline.
        """
        ngram_range = tuple(self._config.get("feature_ngram_range", [1, 2]))
        vectorizer = TfidfVectorizer(
            max_features=self._config.get("feature_max_features", 5000),
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
        )
        model_type = self._config.get("model_type", "logistic")
        if model_type == "random_forest":
            max_depth = self._config.get("random_forest_max_depth", None)
            classifier = RandomForestClassifier(
                n_estimators=self._config.get("random_forest_n_estimators", 100),
                max_depth=max_depth if max_depth else None,
                random_state=42,
                n_jobs=-1,
            )
        else:
            classifier = LogisticRegression(
                max_iter=self._config.get("logistic_max_iter", 1000),
                C=self._config.get("logistic_C", 1.0),
                random_state=42,
            )
        return Pipeline([("tfidf", vectorizer), ("clf", classifier)])

    def train(self, min_samples: int | None = None) -> TrainingResult:
        """Train (or retrain) the classifier from stored samples.

        Args:
            min_samples: Override for minimum required samples. If None,
                         uses configured min_training_samples.

        Returns:
            TrainingResult with accuracy, sample count, and metadata.
        """
        if not SKLEARN_AVAILABLE:
            return TrainingResult(
                success=False,
                error="sklearn/joblib not available",
                trained_at=time.time(),
            )

        threshold = min_samples or self._config.get("min_training_samples", 50)
        texts, labels = self._load_training_data()

        if len(texts) < threshold:
            return TrainingResult(
                success=False,
                n_samples=len(texts),
                error=f"Insufficient samples: {len(texts)} < {threshold}",
                trained_at=time.time(),
            )

        n_classes = len(set(labels))
        cv_folds = min(self._config.get("cv_folds", 5), n_classes, len(texts))
        cv_folds = max(cv_folds, 2)

        try:
            pipe = self._build_pipeline()
            scores = cross_val_score(pipe, texts, labels, cv=cv_folds, scoring="accuracy")
            accuracy = float(scores.mean())

            # Fit on full dataset
            pipe.fit(texts, labels)
            self._pipeline = pipe
            self._samples_since_retrain = 0

            # Persist model
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, str(self._model_path))
            # LR-01 (S185): write a keyed MAC so the artifact is authenticated
            # on load. Without a master key the model is persisted but cannot be
            # reloaded (verify fails safe); warn so the operator knows.
            if not write_model_mac(
                self._model_path, _model_mac_path(self._model_path),
                _router_master_key(),
            ):
                logger.warning(
                    "LearnedRouter: no encryption key available; model persisted "
                    "without an integrity MAC and will not be reloaded until "
                    "retrained with a key available"
                )

            result = TrainingResult(
                accuracy=accuracy,
                n_samples=len(texts),
                n_classes=n_classes,
                trained_at=time.time(),
                model_type=self._config.get("model_type", "logistic"),
                cv_folds=cv_folds,
                success=True,
            )
            self._last_training = result
            logger.info(
                "LearnedRouter trained: accuracy=%.3f, samples=%d, classes=%d",
                accuracy, len(texts), n_classes,
            )
            return result

        except Exception as exc:
            logger.error("Training failed: %s", exc)
            return TrainingResult(
                success=False,
                n_samples=len(texts),
                error=str(exc),
                trained_at=time.time(),
            )

    def _try_load_model(self) -> bool:
        """Attempt to load a previously saved model from disk.

        Returns:
            True if a model was successfully loaded.
        """
        if not SKLEARN_AVAILABLE or not self._model_path.exists():
            return False
        # LR-01 (S185): joblib.load is pickle deserialization (arbitrary code
        # execution on a tampered/swapped file). Verify a keyed MAC BEFORE
        # loading and refuse on mismatch, missing MAC, or no key (fail-safe).
        if not verify_model_mac(
            self._model_path, _model_mac_path(self._model_path),
            _router_master_key(),
        ):
            logger.warning(
                "LearnedRouter: refusing to load persisted model %s -- integrity "
                "MAC missing, invalid, or no encryption key (fail-safe skip; "
                "retrain to re-authenticate)",
                self._model_path,
            )
            self._pipeline = None
            return False
        try:
            self._pipeline = joblib.load(str(self._model_path))
            logger.info("LearnedRouter: loaded persisted model from %s", self._model_path)
            return True
        except Exception as exc:
            logger.warning("Could not load persisted model: %s", exc)
            self._pipeline = None
            return False

    @property
    def is_trained(self) -> bool:
        """True if a fitted model is currently available."""
        return self._pipeline is not None

    @property
    def last_training_result(self) -> TrainingResult | None:
        """Most recent TrainingResult, or None if never trained."""
        return self._last_training

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------

    def classify(self, query: str) -> RoutingPrediction:
        """Predict the task type for a query using the ML model.

        Args:
            query: Raw query text.

        Returns:
            RoutingPrediction with task_type, confidence, and top classes.
            If the model is not trained or unavailable, returns a prediction
            with task_type='general' and confidence=0.0.
        """
        if not SKLEARN_AVAILABLE or self._pipeline is None:
            return RoutingPrediction(
                task_type="general",
                confidence=0.0,
                model_type="none",
                fallback_used=True,
            )

        try:
            proba = self._pipeline.predict_proba([query])[0]
            classes = self._pipeline.classes_
            top_idx = int(proba.argmax())
            confidence = float(proba[top_idx])
            task_type = str(classes[top_idx])

            # Build top-3 list
            sorted_pairs = sorted(
                zip(classes, proba), key=lambda p: p[1], reverse=True
            )[:3]
            top_classes = [
                {"task_type": str(c), "confidence": round(float(p), 4)}
                for c, p in sorted_pairs
            ]

            return RoutingPrediction(
                task_type=task_type,
                confidence=confidence,
                model_type=self._config.get("model_type", "logistic"),
                fallback_used=False,
                top_classes=top_classes,
            )
        except Exception as exc:
            logger.debug("classify() failed: %s", exc)
            return RoutingPrediction(
                task_type="general",
                confidence=0.0,
                model_type="error",
                fallback_used=True,
            )

    def classify_with_fallback(
        self,
        query: str,
        fallback_task_type: str,
    ) -> RoutingPrediction:
        """Classify a query, falling back when confidence is too low.

        If the ML model predicts with confidence >= confidence_threshold
        and the feature is enabled, the ML prediction is returned. Otherwise
        the fallback_task_type (from YAML router) is returned instead.

        Args:
            query: Raw query text.
            fallback_task_type: Task type determined by the YAML router.

        Returns:
            RoutingPrediction; fallback_used=True when YAML task type is used.
        """
        enabled = self._config.get("enabled", False)
        threshold = self._config.get("confidence_threshold", 0.70)

        if not enabled or not self.is_trained:
            return RoutingPrediction(
                task_type=fallback_task_type,
                confidence=0.0,
                model_type="yaml_fallback",
                fallback_used=True,
            )

        prediction = self.classify(query)

        if prediction.confidence >= threshold:
            return prediction

        # Below threshold -- use YAML fallback
        return RoutingPrediction(
            task_type=fallback_task_type,
            confidence=prediction.confidence,
            model_type=self._config.get("model_type", "logistic"),
            fallback_used=True,
            top_classes=prediction.top_classes,
        )

    # -------------------------------------------------------------------------
    # Auto-retraining
    # -------------------------------------------------------------------------

    def auto_retrain_if_needed(self) -> TrainingResult | None:
        """Trigger retraining if enough new samples have accumulated.

        Called opportunistically (e.g., from log_sample or a background
        task). Returns TrainingResult if retraining was triggered, else None.

        Returns:
            TrainingResult if retrained, None otherwise.
        """
        interval = self._config.get("auto_retrain_interval", 100)
        if self._samples_since_retrain >= interval:
            logger.info(
                "Auto-retrain triggered (%d new samples)", self._samples_since_retrain
            )
            return self.train()
        return None

    # -------------------------------------------------------------------------
    # Decision logging (for A/B metrics)
    # -------------------------------------------------------------------------

    def log_routing_decision(
        self,
        query_text: str,
        ml_task_type: str,
        ml_confidence: float,
        yaml_task_type: str,
        routing_source: str,
    ) -> None:
        """Persist a routing decision for A/B comparison metrics.

        Args:
            query_text: Raw query.
            ml_task_type: Task type predicted by ML model.
            ml_confidence: ML model confidence score.
            yaml_task_type: Task type from YAML heuristic router.
            routing_source: 'learned' or 'yaml'.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO routing_decisions "
                "(query_text, ml_task_type, ml_confidence, yaml_task_type, "
                "routing_source, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    query_text[:500],  # Truncate for storage
                    ml_task_type,
                    ml_confidence,
                    yaml_task_type,
                    routing_source,
                    time.time(),
                ),
            )
            conn.commit()
        except Exception as exc:
            logger.debug("log_routing_decision failed: %s", exc)
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Status reporting
    # -------------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a complete status snapshot for API/frontend consumption.

        Returns:
            Dict with training state, sample counts, and config summary.
        """
        sample_count = self.get_sample_count()
        class_dist = self.get_class_distribution()
        training = self._last_training.to_dict() if self._last_training else None

        return {
            "available": SKLEARN_AVAILABLE,
            "trained": self.is_trained,
            "enabled": self._config.get("enabled", False),
            "sklearn_available": SKLEARN_AVAILABLE,
            "sample_count": sample_count,
            "samples_since_retrain": self._samples_since_retrain,
            "min_training_samples": self._config.get("min_training_samples", 50),
            "class_distribution": class_dist,
            "last_training": training,
            "model_type": self._config.get("model_type", "logistic"),
            "confidence_threshold": self._config.get("confidence_threshold", 0.70),
            "auto_retrain_interval": self._config.get("auto_retrain_interval", 100),
        }


# =============================================================================
# A/B METRICS
# =============================================================================

@dataclass
class ABMetricsResult:
    """Aggregated A/B comparison between learned and YAML routing."""

    total_decisions: int = 0
    learned_count: int = 0          # Decisions where ML model was used
    yaml_count: int = 0             # Decisions where YAML fallback was used
    learned_ratio: float = 0.0      # learned_count / total_decisions
    avg_ml_confidence: float = 0.0  # Mean ML confidence across all decisions
    avg_ml_confidence_learned: float = 0.0  # Mean confidence when ML was used
    avg_ml_confidence_yaml: float = 0.0     # Mean confidence when YAML fell back
    class_agreement_rate: float = 0.0       # Rate where ML == YAML task type
    top_disagreements: list[dict] = field(default_factory=list)  # ML vs YAML divergences
    decisions_by_source: dict[str, int] = field(default_factory=dict)
    window_hours: float = 24.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_decisions": self.total_decisions,
            "learned_count": self.learned_count,
            "yaml_count": self.yaml_count,
            "learned_ratio": round(self.learned_ratio, 4),
            "avg_ml_confidence": round(self.avg_ml_confidence, 4),
            "avg_ml_confidence_learned": round(self.avg_ml_confidence_learned, 4),
            "avg_ml_confidence_yaml": round(self.avg_ml_confidence_yaml, 4),
            "class_agreement_rate": round(self.class_agreement_rate, 4),
            "top_disagreements": self.top_disagreements,
            "decisions_by_source": self.decisions_by_source,
            "window_hours": self.window_hours,
        }


class LearnedRouterMetrics:
    """Aggregates routing decision logs for A/B comparison reporting.

    Reads from the routing_decisions table populated by
    LearnedRouter.log_routing_decision() and computes summary statistics
    comparing ML-based vs YAML heuristic routing choices.

    Usage:
        metrics = LearnedRouterMetrics(learned_router_instance)
        result = metrics.compute(window_hours=24)
    """

    def __init__(self, router: "LearnedRouter"):
        """Initialize with a LearnedRouter instance.

        Args:
            router: LearnedRouter whose DB will be queried.
        """
        self._router = router

    def compute(self, window_hours: float = 24.0) -> ABMetricsResult:
        """Compute A/B metrics over a recent time window.

        Args:
            window_hours: How many hours of history to include.

        Returns:
            ABMetricsResult with aggregated statistics.
        """
        since = time.time() - window_hours * 3600
        conn = self._router._get_conn()
        try:
            rows = conn.execute(
                "SELECT ml_task_type, ml_confidence, yaml_task_type, routing_source "
                "FROM routing_decisions WHERE timestamp >= ? ORDER BY timestamp DESC",
                (since,),
            ).fetchall()
        except Exception as exc:
            logger.debug("LearnedRouterMetrics.compute() query failed: %s", exc)
            return ABMetricsResult(window_hours=window_hours)
        finally:
            conn.close()

        if not rows:
            return ABMetricsResult(window_hours=window_hours)

        total = len(rows)
        learned_rows = [r for r in rows if r["routing_source"] == "learned"]
        yaml_rows = [r for r in rows if r["routing_source"] == "yaml"]

        # Confidence averages
        all_conf = [r["ml_confidence"] for r in rows]
        avg_conf = sum(all_conf) / len(all_conf) if all_conf else 0.0

        learned_conf = [r["ml_confidence"] for r in learned_rows]
        avg_learned_conf = sum(learned_conf) / len(learned_conf) if learned_conf else 0.0

        yaml_conf = [r["ml_confidence"] for r in yaml_rows]
        avg_yaml_conf = sum(yaml_conf) / len(yaml_conf) if yaml_conf else 0.0

        # Class agreement: cases where ML prediction matched YAML prediction
        agreements = sum(
            1 for r in rows
            if r["ml_task_type"] and r["yaml_task_type"]
            and r["ml_task_type"] == r["yaml_task_type"]
        )
        comparable = sum(
            1 for r in rows if r["ml_task_type"] and r["yaml_task_type"]
        )
        agreement_rate = agreements / comparable if comparable > 0 else 0.0

        # Top disagreements: (ml_task, yaml_task) pairs ranked by frequency
        disagreement_counts: dict[tuple, int] = {}
        for r in rows:
            ml = r["ml_task_type"]
            yaml_ = r["yaml_task_type"]
            if ml and yaml_ and ml != yaml_:
                key = (ml, yaml_)
                disagreement_counts[key] = disagreement_counts.get(key, 0) + 1

        top_disagreements = [
            {"ml_task_type": ml, "yaml_task_type": yaml_, "count": cnt}
            for (ml, yaml_), cnt in sorted(
                disagreement_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]
        ]

        return ABMetricsResult(
            total_decisions=total,
            learned_count=len(learned_rows),
            yaml_count=len(yaml_rows),
            learned_ratio=len(learned_rows) / total if total > 0 else 0.0,
            avg_ml_confidence=avg_conf,
            avg_ml_confidence_learned=avg_learned_conf,
            avg_ml_confidence_yaml=avg_yaml_conf,
            class_agreement_rate=agreement_rate,
            top_disagreements=top_disagreements,
            decisions_by_source={"learned": len(learned_rows), "yaml": len(yaml_rows)},
            window_hours=window_hours,
        )

    def get_confidence_histogram(
        self,
        bins: int = 10,
        window_hours: float = 24.0,
    ) -> list[dict]:
        """Compute a histogram of ML confidence scores.

        Args:
            bins: Number of histogram buckets (evenly spaced 0-1).
            bins: Number of histogram buckets.
            window_hours: Time window for filtering.

        Returns:
            List of dicts with 'bucket_min', 'bucket_max', 'count'.
        """
        since = time.time() - window_hours * 3600
        conn = self._router._get_conn()
        try:
            rows = conn.execute(
                "SELECT ml_confidence FROM routing_decisions WHERE timestamp >= ?",
                (since,),
            ).fetchall()
        except Exception:
            return []
        finally:
            conn.close()

        if not rows:
            return []

        bucket_size = 1.0 / bins
        counts = [0] * bins
        for r in rows:
            conf = r["ml_confidence"]
            idx = min(int(conf / bucket_size), bins - 1)
            counts[idx] += 1

        return [
            {
                "bucket_min": round(i * bucket_size, 2),
                "bucket_max": round((i + 1) * bucket_size, 2),
                "count": counts[i],
            }
            for i in range(bins)
        ]


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

try:
    learned_router = LearnedRouter()
    logger.debug("LearnedRouter singleton created")
except Exception as _exc:
    logger.warning("LearnedRouter singleton failed to initialize: %s", _exc)
    learned_router = None  # type: ignore[assignment]
    LEARNED_ROUTER_AVAILABLE = False
