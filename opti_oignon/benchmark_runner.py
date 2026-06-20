#!/usr/bin/env python3
"""
Benchmark Runner — S88

Orchestrates benchmark runs across models and profiles. Runs questions
sequentially or in parallel, collects evaluation results, computes
composite scores, and persists everything to SQLite.

Features:
  - Run a profile against one or multiple models
  - Progress callbacks for frontend streaming
  - SQLite results storage with temporal tracking
  - Cross-model and cross-run comparison
  - Configurable retention
"""

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)
# S136 audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from opti_oignon.benchmark_evaluator import (
        BenchmarkEvaluator,
        Question,
        QuestionResult,
        AccuracyResult,
        CodeResult,
        StructuralResult,
        PerformanceResult,
        WeightPreset,
        benchmark_evaluator,
        evaluate_accuracy,
        evaluate_code,
        evaluate_structure,
        evaluate_performance,
        compute_composite_score,
        BENCHMARK_EVALUATOR_AVAILABLE,
    )
except ImportError:
    try:
        # Fallback: direct relative import (when loaded via importlib)
        import importlib.util as _ilu
        _eval_spec = _ilu.spec_from_file_location(
            "_benchmark_evaluator",
            str(Path(__file__).parent / "benchmark_evaluator.py"),
        )
        _eval_mod = _ilu.module_from_spec(_eval_spec)
        import sys as _sys1
        _sys1.modules["_benchmark_evaluator"] = _eval_mod  # Python 3.13: @dataclass needs sys.modules
        _eval_spec.loader.exec_module(_eval_mod)
        BenchmarkEvaluator = _eval_mod.BenchmarkEvaluator
        Question = _eval_mod.Question
        QuestionResult = _eval_mod.QuestionResult
        AccuracyResult = _eval_mod.AccuracyResult
        CodeResult = _eval_mod.CodeResult
        StructuralResult = _eval_mod.StructuralResult
        PerformanceResult = _eval_mod.PerformanceResult
        WeightPreset = _eval_mod.WeightPreset
        benchmark_evaluator = _eval_mod.benchmark_evaluator
        evaluate_accuracy = _eval_mod.evaluate_accuracy
        evaluate_code = _eval_mod.evaluate_code
        evaluate_structure = _eval_mod.evaluate_structure
        evaluate_performance = _eval_mod.evaluate_performance
        compute_composite_score = _eval_mod.compute_composite_score
        BENCHMARK_EVALUATOR_AVAILABLE = _eval_mod.BENCHMARK_EVALUATOR_AVAILABLE
    except Exception:
        BENCHMARK_EVALUATOR_AVAILABLE = False
        benchmark_evaluator = None

try:
    from opti_oignon.sandbox_manager import sandbox_manager
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    sandbox_manager = None

try:
    from opti_oignon.benchmark_judge import (
        BenchmarkJudge,
        benchmark_judge as _default_judge,
        BENCHMARK_JUDGE_AVAILABLE,
    )
except ImportError:
    try:
        import importlib.util as _ilu2
        _judge_spec = _ilu2.spec_from_file_location(
            "_benchmark_judge",
            str(Path(__file__).parent / "benchmark_judge.py"),
        )
        _judge_mod = _ilu2.module_from_spec(_judge_spec)
        import sys as _sys2
        _sys2.modules["_benchmark_judge"] = _judge_mod  # Python 3.13: @dataclass needs sys.modules
        _judge_spec.loader.exec_module(_judge_mod)
        BenchmarkJudge = _judge_mod.BenchmarkJudge
        _default_judge = _judge_mod.benchmark_judge
        BENCHMARK_JUDGE_AVAILABLE = _judge_mod.BENCHMARK_JUDGE_AVAILABLE
    except Exception:
        BENCHMARK_JUDGE_AVAILABLE = False
        _default_judge = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_DB_PATH = _DATA_DIR / "benchmark_results.db"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class RunStatus(str, Enum):
    """Status of a benchmark run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunProgress:
    """Progress state for a benchmark run."""
    run_id: str
    status: RunStatus = RunStatus.PENDING
    total_questions: int = 0
    completed_questions: int = 0
    current_model: str = ""
    current_question: str = ""
    started_at: float = 0.0
    elapsed_ms: float = 0.0
    error: str = ""


@dataclass
class ModelScore:
    """Aggregated scores for a model within a run."""
    model: str
    accuracy_avg: float = 0.0
    code_avg: float = 0.0
    structure_avg: float = 0.0
    speed_avg: float = 0.0
    composite: float = 0.0
    questions_evaluated: int = 0
    question_results: list = field(default_factory=list)
    # S224 (R-01 benchmark semantics, spec 4.3): a model the governor
    # refuses is SKIPPED and recorded as not-admitted with the reason --
    # never silently downsized.
    not_admitted: bool = False
    admission_reason: str = ""


@dataclass
class RunResult:
    """Complete result of a benchmark run."""
    run_id: str
    profile: str
    models: list[str] = field(default_factory=list)
    status: RunStatus = RunStatus.PENDING
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_ms: float = 0.0
    model_scores: dict[str, ModelScore] = field(default_factory=dict)
    weight_preset: str = "balanced"
    custom_weights: dict[str, float] | None = None
    error: str = ""


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

class ResultsStore:
    """SQLite-backed storage for benchmark results."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or _DEFAULT_DB_PATH)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS benchmark_runs (
                        run_id TEXT PRIMARY KEY,
                        profile TEXT NOT NULL,
                        models TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        started_at REAL NOT NULL,
                        finished_at REAL DEFAULT 0,
                        duration_ms REAL DEFAULT 0,
                        weight_preset TEXT DEFAULT 'balanced',
                        custom_weights TEXT DEFAULT '',
                        error TEXT DEFAULT ''
                    );

                    CREATE TABLE IF NOT EXISTS benchmark_model_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        model TEXT NOT NULL,
                        accuracy_avg REAL DEFAULT 0,
                        code_avg REAL DEFAULT 0,
                        structure_avg REAL DEFAULT 0,
                        speed_avg REAL DEFAULT 0,
                        composite REAL DEFAULT 0,
                        questions_evaluated INTEGER DEFAULT 0,
                        not_admitted INTEGER DEFAULT 0,
                        admission_reason TEXT DEFAULT '',
                        FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
                    );

                    CREATE TABLE IF NOT EXISTS benchmark_question_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        model TEXT NOT NULL,
                        question_id TEXT NOT NULL,
                        category TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        response TEXT DEFAULT '',
                        accuracy_score REAL DEFAULT 0,
                        code_score REAL DEFAULT 0,
                        structure_score REAL DEFAULT 0,
                        speed_score REAL DEFAULT 0,
                        composite_score REAL DEFAULT 0,
                        details TEXT DEFAULT '{}',
                        timestamp REAL NOT NULL,
                        FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_runs_profile
                        ON benchmark_runs(profile);
                    CREATE INDEX IF NOT EXISTS idx_runs_started
                        ON benchmark_runs(started_at);
                    CREATE INDEX IF NOT EXISTS idx_scores_run
                        ON benchmark_model_scores(run_id);
                    CREATE INDEX IF NOT EXISTS idx_qresults_run
                        ON benchmark_question_results(run_id, model);
                """)
                # S224: additive columns for the not-admitted recording
                # (R-01 benchmark semantics). Guarded ALTERs migrate
                # pre-S224 databases; fresh databases already carry the
                # columns through the CREATE above (duplicate-column
                # errors are the expected no-op). House precedent:
                # auth_2fa, plugin_reviews, semantic_cache.
                for _ddl in (
                    "ALTER TABLE benchmark_model_scores "
                    "ADD COLUMN not_admitted INTEGER DEFAULT 0",
                    "ALTER TABLE benchmark_model_scores "
                    "ADD COLUMN admission_reason TEXT DEFAULT ''",
                ):
                    try:
                        conn.execute(_ddl)
                    except sqlite3.OperationalError:
                        pass
                conn.commit()
            finally:
                conn.close()

    def save_run(self, result: RunResult) -> None:
        """Persist a complete run result."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO benchmark_runs
                       (run_id, profile, models, status, started_at,
                        finished_at, duration_ms, weight_preset,
                        custom_weights, error)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        result.run_id,
                        result.profile,
                        json.dumps(result.models),
                        result.status.value,
                        result.started_at,
                        result.finished_at,
                        result.duration_ms,
                        result.weight_preset,
                        json.dumps(result.custom_weights) if result.custom_weights else "",
                        result.error,
                    ),
                )

                # Save model scores
                for model, ms in result.model_scores.items():
                    conn.execute(
                        """INSERT INTO benchmark_model_scores
                           (run_id, model, accuracy_avg, code_avg,
                            structure_avg, speed_avg, composite,
                            questions_evaluated, not_admitted,
                            admission_reason)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            result.run_id,
                            model,
                            ms.accuracy_avg,
                            ms.code_avg,
                            ms.structure_avg,
                            ms.speed_avg,
                            ms.composite,
                            ms.questions_evaluated,
                            int(getattr(ms, "not_admitted", False)),
                            str(getattr(ms, "admission_reason", "")),
                        ),
                    )

                    # Save individual question results
                    for qr in ms.question_results:
                        details = {}
                        if isinstance(qr, dict):
                            details = qr.get("details", {})
                            q_id = qr.get("question_id", "")
                            q_cat = qr.get("category", "")
                            q_prompt = qr.get("prompt", "")
                            q_resp = qr.get("response", "")
                            q_acc = qr.get("accuracy_score", 0.0)
                            q_code = qr.get("code_score", 0.0)
                            q_struct = qr.get("structure_score", 0.0)
                            q_speed = qr.get("speed_score", 0.0)
                            q_comp = qr.get("composite_score", 0.0)
                        else:
                            q_id = qr.question_id
                            q_cat = qr.category
                            q_prompt = qr.prompt
                            q_resp = qr.response
                            q_acc = qr.accuracy.score if qr.accuracy else 0.0
                            q_code = qr.code.score if qr.code else 0.0
                            q_struct = qr.structure.composite if qr.structure else 0.0
                            q_speed = qr.performance.score if qr.performance else 0.0
                            q_comp = qr.composite_score
                            details = _question_result_details(qr)

                        conn.execute(
                            """INSERT INTO benchmark_question_results
                               (run_id, model, question_id, category,
                                prompt, response, accuracy_score, code_score,
                                structure_score, speed_score, composite_score,
                                details, timestamp)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                result.run_id,
                                model,
                                q_id,
                                q_cat,
                                q_prompt,
                                q_resp,
                                q_acc,
                                q_code,
                                q_struct,
                                q_speed,
                                q_comp,
                                json.dumps(details),
                                time.time(),
                            ),
                        )

                conn.commit()
            finally:
                conn.close()

    def get_run(self, run_id: str) -> dict | None:
        """Retrieve a single run with model scores."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    "SELECT * FROM benchmark_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if not row:
                    return None

                run = dict(row)
                run["models"] = json.loads(run["models"])
                cw_raw = run.get("custom_weights", "")
                if cw_raw:
                    try:
                        run["custom_weights"] = json.loads(cw_raw)
                    except (json.JSONDecodeError, TypeError):
                        run["custom_weights"] = None
                else:
                    run["custom_weights"] = None

                # Fetch model scores
                scores = conn.execute(
                    "SELECT * FROM benchmark_model_scores WHERE run_id = ?",
                    (run_id,),
                ).fetchall()
                run["model_scores"] = {
                    s["model"]: dict(s) for s in scores
                }

                return run
            finally:
                conn.close()

    def get_run_details(self, run_id: str) -> dict | None:
        """Retrieve a run with model scores and per-question results."""
        run = self.get_run(run_id)
        if not run:
            return None

        with self._lock:
            conn = _safe_connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                qresults = conn.execute(
                    """SELECT * FROM benchmark_question_results
                       WHERE run_id = ? ORDER BY model, question_id""",
                    (run_id,),
                ).fetchall()
                by_model: dict[str, list[dict]] = {}
                for qr in qresults:
                    d = dict(qr)
                    d["details"] = json.loads(d.get("details", "{}"))
                    model = d["model"]
                    by_model.setdefault(model, []).append(d)
                run["question_results"] = by_model
                return run
            finally:
                conn.close()

    def get_history(
        self,
        limit: int = 50,
        profile: str | None = None,
        model: str | None = None,
    ) -> list[dict]:
        """Retrieve historical runs, optionally filtered."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                query = "SELECT * FROM benchmark_runs WHERE 1=1"
                params: list = []
                if profile:
                    query += " AND profile = ?"
                    params.append(profile)
                query += " ORDER BY started_at DESC LIMIT ?"
                params.append(limit)

                rows = conn.execute(query, params).fetchall()
                runs = []
                for row in rows:
                    run = dict(row)
                    run["models"] = json.loads(run["models"])
                    cw_raw = run.get("custom_weights", "")
                    if cw_raw:
                        try:
                            run["custom_weights"] = json.loads(cw_raw)
                        except (json.JSONDecodeError, TypeError):
                            run["custom_weights"] = None
                    else:
                        run["custom_weights"] = None
                    # Attach model scores
                    if model:
                        scores = conn.execute(
                            """SELECT * FROM benchmark_model_scores
                               WHERE run_id = ? AND model = ?""",
                            (run["run_id"], model),
                        ).fetchall()
                    else:
                        scores = conn.execute(
                            """SELECT * FROM benchmark_model_scores
                               WHERE run_id = ?""",
                            (run["run_id"],),
                        ).fetchall()
                    run["model_scores"] = {
                        s["model"]: dict(s) for s in scores
                    }
                    runs.append(run)
                return runs
            finally:
                conn.close()

    def compare_models(
        self,
        model_names: list[str] | None = None,
        profile: str | None = None,
        limit: int = 10,
    ) -> dict:
        """Compare models across their latest benchmark runs.

        Returns aggregated per-model stats from recent runs.
        """
        with self._lock:
            conn = _safe_connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                query = """
                    SELECT ms.model,
                           AVG(ms.accuracy_avg) as avg_accuracy,
                           AVG(ms.code_avg) as avg_code,
                           AVG(ms.structure_avg) as avg_structure,
                           AVG(ms.speed_avg) as avg_speed,
                           AVG(ms.composite) as avg_composite,
                           COUNT(*) as run_count,
                           MAX(br.started_at) as last_run
                    FROM benchmark_model_scores ms
                    JOIN benchmark_runs br ON ms.run_id = br.run_id
                    WHERE br.status = 'completed'
                """
                params: list = []
                if model_names:
                    placeholders = ",".join("?" * len(model_names))
                    query += f" AND ms.model IN ({placeholders})"
                    params.extend(model_names)
                if profile:
                    query += " AND br.profile = ?"
                    params.append(profile)
                query += " GROUP BY ms.model ORDER BY avg_composite DESC"
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                rows = conn.execute(query, params).fetchall()
                return {
                    "models": [dict(r) for r in rows],
                    "profile_filter": profile,
                    "model_filter": model_names,
                }
            finally:
                conn.close()

    def cleanup(self, retention_days: int = 90) -> int:
        """Remove runs older than retention period."""
        cutoff = time.time() - (retention_days * 86400)
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                # Get old run IDs
                old_ids = conn.execute(
                    "SELECT run_id FROM benchmark_runs WHERE started_at < ?",
                    (cutoff,),
                ).fetchall()
                if not old_ids:
                    return 0
                ids = [r[0] for r in old_ids]
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    "DELETE FROM benchmark_question_results WHERE run_id IN ({})".format(placeholders),
                    ids,
                )
                conn.execute(
                    "DELETE FROM benchmark_model_scores WHERE run_id IN ({})".format(placeholders),
                    ids,
                )
                conn.execute(
                    "DELETE FROM benchmark_runs WHERE run_id IN ({})".format(placeholders),
                    ids,
                )
                conn.commit()
                return len(ids)
            finally:
                conn.close()


def _question_result_details(qr: Any) -> dict:
    """Extract serializable details from a QuestionResult."""
    details: dict = {}
    if hasattr(qr, "accuracy") and qr.accuracy:
        details["accuracy"] = {
            "score": qr.accuracy.score,
            "matched": qr.accuracy.matched_answer,
            "method": qr.accuracy.method,
        }
    if hasattr(qr, "code") and qr.code:
        details["code"] = {
            "score": qr.code.score,
            "compiles": qr.code.compiles,
            "runs": qr.code.runs,
            "output_matches": qr.code.output_matches,
            "tests_pass": qr.code.tests_pass,
        }
    if hasattr(qr, "structure") and qr.structure:
        details["structure"] = {
            "repetition": qr.structure.repetition_score,
            "diversity": qr.structure.lexical_diversity,
            "length": qr.structure.length_appropriateness,
            "format": qr.structure.format_compliance,
            "composite": qr.structure.composite,
        }
    if hasattr(qr, "performance") and qr.performance:
        details["performance"] = {
            "ttft_ms": qr.performance.ttft_ms,
            "tps": qr.performance.tokens_per_second,
            "total_ms": qr.performance.total_time_ms,
            "score": qr.performance.score,
        }
    return details


# ---------------------------------------------------------------------------
# LLM query helper
# ---------------------------------------------------------------------------

# S193 BMK-02: per-timeout cached Ollama clients (RSN-02 idiom) so the
# profile timeout is actually enforced at the transport level; it was
# previously accepted and silently ignored.
_OLLAMA_CLIENTS: dict[int, Any] = {}
_OLLAMA_CLIENTS_LOCK = threading.Lock()


def _get_ollama_client(timeout: int) -> Any:
    """Return a cached ollama.Client bound to the given timeout."""
    import ollama
    with _OLLAMA_CLIENTS_LOCK:
        client = _OLLAMA_CLIENTS.get(timeout)
        if client is None:
            client = ollama.Client(timeout=timeout)
            _OLLAMA_CLIENTS[timeout] = client
        return client


def _chunk_message_text(chunk: Any) -> str:
    """Extract message content from a stream chunk (dict or object form).

    S193 BMK-01: the ollama client returns dicts in older versions and
    typed objects in newer ones; handle both (MEM-06 idiom) instead of
    the dict-only access that silently emptied every response.
    """
    if isinstance(chunk, dict):
        msg = chunk.get("message") or {}
    else:
        msg = getattr(chunk, "message", None)
    if msg is None:
        return ""
    if isinstance(msg, dict):
        return msg.get("content") or ""
    return getattr(msg, "content", "") or ""


def _chunk_eval_count(chunk: Any) -> int:
    """Extract the exact eval_count from a final stream chunk, if present."""
    if isinstance(chunk, dict):
        val = chunk.get("eval_count")
    else:
        val = getattr(chunk, "eval_count", None)
    try:
        return int(val) if val else 0
    except (TypeError, ValueError):
        return 0


def _query_ollama(
    model: str,
    prompt: str,
    timeout: int = 45,
    max_tokens: int = 800,
) -> tuple[str, float, float, int]:
    """Send a prompt to Ollama and return response with timing.

    Returns:
        Tuple of (response_text, ttft_ms, total_time_ms, token_count).
    """
    try:
        import ollama  # noqa: F401
    except ImportError:
        return "", 0.0, 0.0, 0

    start = time.time()
    ttft = 0.0
    chunks: list[str] = []
    token_count = 0
    eval_count = 0

    try:
        client = _get_ollama_client(timeout)
        stream = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={
                "num_predict": max_tokens,
            },
        )
        for chunk in stream:
            content = _chunk_message_text(chunk)
            if content:
                # S193 BMK-03: TTFT measured at the first content-bearing
                # chunk, not at a role-only preamble chunk.
                if ttft == 0.0:
                    ttft = (time.time() - start) * 1000
                chunks.append(content)
                token_count += 1  # Approximate: 1 chunk ~ 1 token
            ec = _chunk_eval_count(chunk)
            if ec:
                eval_count = ec

    except Exception as e:
        logger.error("Ollama query failed for model %s: %s", model, e)
        return "", 0.0, (time.time() - start) * 1000, 0

    total_ms = (time.time() - start) * 1000
    response = "".join(chunks)
    # S193 BMK-03: prefer the exact token count reported by Ollama in the
    # final chunk over the chunk-count approximation.
    if eval_count:
        token_count = eval_count
    return response, ttft, total_ms, token_count


# ---------------------------------------------------------------------------
# Resource Governor wiring (S224, R-01 benchmark semantics)
# ---------------------------------------------------------------------------


def _resolve_resource_governor() -> Any:
    """S224: lazy governor resolver; None means unguarded (fail-open).

    sys.modules is consulted first so a test-seeded or standalone-loaded
    module is reused as-is, then the package import; any error degrades
    to None (the availability-control posture).
    """
    try:
        import sys as _sys

        mod = _sys.modules.get("opti_oignon.resource_governor")
        if mod is None:
            from opti_oignon import resource_governor as mod  # type: ignore
        if mod is None or not getattr(mod, "FEATURE_AVAILABLE", False):
            return None
        return mod
    except Exception:
        return None


def _admit_benchmark_model(model: str) -> Any:
    """S224: per-model admission with benchmark semantics (spec 4.3):
    admit or refuse, NEVER downsize (the governor enforces it through
    the absent ctx floor for this caller). None when the governor is
    absent or disabled. A positive admission expecting a load is
    accounted here: the benchmark transport is a direct ollama call out
    of the mechanical seam's reach, so the funnel is the closest seam.
    """
    governor_module = _resolve_resource_governor()
    if governor_module is None:
        return None
    try:
        governor = governor_module.get_resource_governor()
        if not governor.config.enabled:
            return None
        # S225 (Bloc 2): route through the bounded opt-in queue entry.
        # With the shipped default (nobody enrolled) this degrades to
        # plain admit() bit for bit; enrolling "benchmark" in
        # queue.enabled_per_caller makes a refused admission wait
        # bounded-then-retry instead of skipping. Defensive getattr so
        # an injected pre-S225 governor still fails open.
        admit_fn = getattr(governor, "admit_or_wait", governor.admit)
        decision = admit_fn(model, None, caller="benchmark")
        if decision.admitted and decision.load_expected:
            governor.invalidate_on_load(model, decision.num_ctx)
        return decision
    except Exception as e:
        logger.debug("Benchmark admission failed open: %s", e)
        return None


def _evict_loaded_models() -> int:
    """S224: best-effort eviction between benchmark models through the
    registry's existing unload idiom (keep_alive=0, the S215 primitive);
    the governor snapshot is then invalidated so the next admission sees
    the clean slate. Returns the number of models unloaded (0 on any
    failure); every path fails open.
    """
    count = 0
    try:
        import sys as _sys

        ib = _sys.modules.get("opti_oignon.inference_backend")
        if ib is None:
            from opti_oignon import inference_backend as ib  # type: ignore
        registry = ib.get_backend_registry()
        for backend in registry.backends():
            unload = getattr(backend, "unload_all", None)
            if callable(unload):
                try:
                    count += int(unload() or 0)
                except Exception as e:
                    logger.debug("Evict-between unload failed: %s", e)
    except Exception as e:
        logger.debug("Evict-between unavailable: %s", e)
    governor_module = _resolve_resource_governor()
    if governor_module is not None:
        try:
            governor_module.get_resource_governor().invalidate_on_evict(None)
        except Exception:
            pass
    return count


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[RunProgress], None]


class BenchmarkRunner:
    """Orchestrates benchmark runs across models and profiles.

    Executes questions, evaluates responses, computes composite scores,
    and persists results to SQLite.
    """

    def __init__(
        self,
        evaluator: Any = None,
        store: ResultsStore | None = None,
        db_path: str | Path | None = None,
        judge: Any = None,
    ):
        self._evaluator = evaluator or benchmark_evaluator
        self._store = store or ResultsStore(db_path)
        self._judge = judge or _default_judge
        self._active_runs: dict[str, RunProgress] = {}
        self._lock = threading.Lock()
        self._cancelled: set[str] = set()

    @property
    def store(self) -> ResultsStore:
        return self._store

    @property
    def is_busy(self) -> bool:
        """Return True if any benchmark run is currently in progress."""
        with self._lock:
            for prog in self._active_runs.values():
                if prog.status in (RunStatus.PENDING, RunStatus.RUNNING):
                    return True
        return False

    def start_run(
        self,
        profile: str,
        models: list[str],
        progress_callback: ProgressCallback | None = None,
        query_fn: Any = None,
        use_judge: bool = False,
        judge_model: str = "",
        custom_weights: dict[str, float] | None = None,
        evict_between: bool = True,
    ) -> str:
        """Start a benchmark run in a background thread.

        Args:
            profile: Profile name from benchmark_profiles.yaml.
            models: List of Ollama model names to benchmark.
            progress_callback: Optional callback for progress updates.
            query_fn: Optional override for _query_ollama (for testing).
            use_judge: Whether to run LLM-as-Judge evaluation after metrics.
            judge_model: Model name to use as judge (required if use_judge).
            custom_weights: Optional custom weight overrides (accuracy, code,
                structure, speed). When provided, these take priority over
                the profile's weight_preset.

        Returns:
            Run ID string.
        """
        run_id = f"run-{uuid.uuid4().hex[:12]}"

        progress = RunProgress(
            run_id=run_id,
            status=RunStatus.PENDING,
        )
        with self._lock:
            self._active_runs[run_id] = progress

        thread = threading.Thread(
            target=self._execute_run,
            args=(run_id, profile, models, progress_callback, query_fn,
                  use_judge, judge_model, custom_weights, evict_between),
            daemon=True,
        )
        thread.start()

        return run_id

    def run_sync(
        self,
        profile: str,
        models: list[str],
        progress_callback: ProgressCallback | None = None,
        query_fn: Any = None,
        use_judge: bool = False,
        judge_model: str = "",
        custom_weights: dict[str, float] | None = None,
        evict_between: bool = True,
    ) -> RunResult:
        """Run a benchmark synchronously (for testing).

        Same as start_run but blocks until completion.
        """
        run_id = f"run-{uuid.uuid4().hex[:12]}"

        progress = RunProgress(
            run_id=run_id,
            status=RunStatus.PENDING,
        )
        with self._lock:
            self._active_runs[run_id] = progress

        self._execute_run(run_id, profile, models, progress_callback, query_fn,
                          use_judge, judge_model, custom_weights,
                          evict_between)

        # Retrieve result from store
        stored = self._store.get_run(run_id)
        if stored:
            return RunResult(
                run_id=run_id,
                profile=profile,
                models=models,
                status=RunStatus(stored["status"]),
                started_at=stored["started_at"],
                finished_at=stored["finished_at"],
                duration_ms=stored["duration_ms"],
                weight_preset=stored.get("weight_preset", "balanced"),
                custom_weights=stored.get("custom_weights"),
                error=stored.get("error", ""),
            )

        return RunResult(
            run_id=run_id,
            profile=profile,
            models=models,
            status=RunStatus.FAILED,
            error="Run result not found in store",
        )

    def get_progress(self, run_id: str) -> RunProgress | None:
        """Get current progress of a running benchmark."""
        with self._lock:
            return self._active_runs.get(run_id)

    def cancel_run(self, run_id: str) -> bool:
        """Request cancellation of a running benchmark."""
        with self._lock:
            if run_id in self._active_runs:
                self._cancelled.add(run_id)
                return True
        return False

    def get_results(self, run_id: str) -> dict | None:
        """Get detailed results for a completed run."""
        return self._store.get_run_details(run_id)

    def compare(
        self,
        models: list[str] | None = None,
        profile: str | None = None,
        limit: int = 10,
    ) -> dict:
        """Compare model performance across runs."""
        return self._store.compare_models(models, profile, limit)

    def history(
        self,
        limit: int = 50,
        profile: str | None = None,
        model: str | None = None,
    ) -> list[dict]:
        """Get historical benchmark runs."""
        return self._store.get_history(limit, profile, model)

    def cleanup(self, retention_days: int = 90) -> int:
        """Clean up old benchmark results."""
        return self._store.cleanup(retention_days)

    # -- Internal execution ------------------------------------------------

    def _execute_run(
        self,
        run_id: str,
        profile: str,
        models: list[str],
        progress_callback: ProgressCallback | None,
        query_fn: Any,
        use_judge: bool = False,
        judge_model: str = "",
        custom_weights: dict[str, float] | None = None,
        evict_between: bool = True,
    ) -> None:
        """Execute a benchmark run with a fail-safe outer guard.

        S193 BMK-12: an unhandled exception in the run body (e.g. a malformed
        profile field) must not leave a zombie RUNNING progress and a
        permanently-busy runner (is_busy would then lock the v2 run endpoint
        at 409). Mark the run FAILED, persist a minimal failed record, clear
        any cancel flag, and fire the final callback.
        """
        try:
            self._execute_run_impl(
                run_id, profile, models, progress_callback, query_fn,
                use_judge, judge_model, custom_weights, evict_between,
            )
        except Exception as e:
            logger.error(
                "Benchmark run %s crashed: %s", run_id, e, exc_info=True,
            )
            self._update_progress(run_id, RunStatus.FAILED, error=str(e))
            try:
                self._store.save_run(RunResult(
                    run_id=run_id,
                    profile=profile,
                    models=models,
                    status=RunStatus.FAILED,
                    error=str(e),
                ))
            except Exception:
                pass
            with self._lock:
                self._cancelled.discard(run_id)
            prog = self._active_runs.get(run_id)
            if progress_callback and prog:
                progress_callback(prog)

    def _execute_run_impl(
        self,
        run_id: str,
        profile: str,
        models: list[str],
        progress_callback: ProgressCallback | None,
        query_fn: Any,
        use_judge: bool = False,
        judge_model: str = "",
        custom_weights: dict[str, float] | None = None,
        evict_between: bool = True,
    ) -> None:
        """Execute a benchmark run (runs in thread or synchronously)."""
        qfn = query_fn or _query_ollama
        started_at = time.time()

        # Load profile config and questions
        if self._evaluator is None:
            self._update_progress(
                run_id, RunStatus.FAILED,
                error="Evaluator not available",
            )
            if progress_callback:
                progress_callback(self._active_runs.get(run_id, RunProgress(run_id=run_id)))
            return

        profile_config = self._evaluator.get_profile_config(profile)
        if not profile_config:
            self._update_progress(
                run_id, RunStatus.FAILED,
                error=f"Profile '{profile}' not found",
            )
            if progress_callback:
                progress_callback(self._active_runs.get(run_id, RunProgress(run_id=run_id)))
            return

        questions = self._evaluator.get_questions_for_profile(profile)
        if not questions:
            self._update_progress(
                run_id, RunStatus.FAILED,
                error=f"No questions found for profile '{profile}'",
            )
            if progress_callback:
                progress_callback(self._active_runs.get(run_id, RunProgress(run_id=run_id)))
            return

        weight_preset_name = profile_config.get("weight_preset", "balanced")

        # Resolve effective weights: explicit custom_weights > profile custom_weights > preset
        effective_custom_weights: dict[str, float] | None = None
        if custom_weights:
            weights = WeightPreset(
                accuracy=custom_weights.get("accuracy", 0.35),
                code=custom_weights.get("code", 0.25),
                structure=custom_weights.get("structure", 0.25),
                speed=custom_weights.get("speed", 0.15),
            )
            effective_custom_weights = custom_weights
        else:
            profile_custom = self._evaluator.get_custom_weights(profile)
            if profile_custom is not None:
                weights = profile_custom
                effective_custom_weights = {
                    "accuracy": profile_custom.accuracy,
                    "code": profile_custom.code,
                    "structure": profile_custom.structure,
                    "speed": profile_custom.speed,
                }
            else:
                weights = self._evaluator.get_weights(weight_preset_name)
        timeout = profile_config.get("timeout", 45)
        max_tokens = profile_config.get("max_response_tokens", 800)
        expected_range = tuple(profile_config.get("expected_length_range", [10, 600]))
        format_check = profile_config.get("format_check", "")
        total_questions = len(questions) * len(models)

        # Update progress: running
        self._update_progress(
            run_id, RunStatus.RUNNING,
            total=total_questions,
            started_at=started_at,
        )
        if progress_callback:
            progress_callback(self._active_runs[run_id])

        # Collect results per model
        model_scores: dict[str, ModelScore] = {}

        for model in models:
            if run_id in self._cancelled:
                break

            # S224: per-model resource admission (R-01, benchmark
            # semantics 4.3: admit or refuse, NEVER downsize). A refused
            # model is SKIPPED and recorded as not-admitted in results;
            # the run continues with the remaining models.
            _admission = _admit_benchmark_model(model)
            if _admission is not None and not _admission.admitted:
                model_scores[model] = ModelScore(
                    model=model,
                    not_admitted=True,
                    admission_reason=str(_admission.reason),
                )
                logger.warning(
                    "Benchmark: model %s not admitted (%s); skipped",
                    model, _admission.reason,
                )
                continue

            ms = ModelScore(model=model)
            accuracy_scores = []
            code_scores = []
            structure_scores = []
            speed_scores = []

            for question in questions:
                if run_id in self._cancelled:
                    break

                # Update current question in progress
                self._update_progress(
                    run_id, RunStatus.RUNNING,
                    current_model=model,
                    current_question=question.id,
                )
                if progress_callback:
                    progress_callback(self._active_runs[run_id])

                # Query the model
                response, ttft_ms, total_ms, token_count = qfn(
                    model, question.prompt, timeout, max_tokens,
                )

                # Calculate tokens per second
                duration_s = total_ms / 1000.0 if total_ms > 0 else 1.0
                tps = token_count / duration_s if duration_s > 0 else 0.0

                # Build QuestionResult
                qr = QuestionResult(
                    question_id=question.id,
                    category=question.category,
                    prompt=question.prompt,
                    response=response,
                )

                # Evaluate accuracy (for non-code-generation questions)
                is_code_gen = question.category == "code_generation"
                # S193 BJD-03: track the axes actually evaluated so the
                # composite renormalizes over them instead of carrying a
                # dead axis weight.
                evaluated_axes = {"structure", "speed"}
                evaluated_axes.add("code" if is_code_gen else "accuracy")

                if not response.strip():
                    # S193 BJD-02: a failed/empty generation scores zero on
                    # every axis instead of being evaluated as text (an
                    # empty response previously scored ~0.57 composite).
                    if is_code_gen:
                        qr.code = CodeResult(
                            question_id=question.id,
                            details="Empty response, not evaluated",
                        )
                        code_scores.append(0.0)
                    else:
                        qr.accuracy = AccuracyResult(
                            question_id=question.id,
                            score=0.0,
                            method="empty",
                            details="Empty response, not evaluated",
                        )
                        accuracy_scores.append(0.0)
                    qr.structure = StructuralResult()
                    structure_scores.append(0.0)
                    qr.performance = PerformanceResult(
                        ttft_ms=ttft_ms,
                        tokens_per_second=tps,
                        total_time_ms=total_ms,
                        score=0.0,
                    )
                    speed_scores.append(0.0)
                    qr.composite_score = 0.0
                else:
                    if not is_code_gen:
                        qr.accuracy = evaluate_accuracy(question, response)
                        accuracy_scores.append(qr.accuracy.score)

                    # Evaluate code (for code generation questions)
                    if is_code_gen and SANDBOX_AVAILABLE:
                        qr.code = evaluate_code(question, response)
                        code_scores.append(qr.code.score)
                    elif is_code_gen:
                        # No sandbox: score based on syntax check only
                        qr.code = CodeResult(
                            question_id=question.id,
                            details="Sandbox unavailable, code not executed",
                        )
                        code_scores.append(0.0)

                    # Evaluate structure
                    qr.structure = evaluate_structure(
                        response, expected_range, format_check,
                    )
                    structure_scores.append(qr.structure.composite)

                    # Evaluate performance
                    qr.performance = evaluate_performance(
                        ttft_ms=ttft_ms,
                        tokens_per_second=tps,
                        total_time_ms=total_ms,
                    )
                    speed_scores.append(qr.performance.score)

                    # Composite score for this question
                    acc = qr.accuracy.score if qr.accuracy else 0.0
                    cod = qr.code.score if qr.code else 0.0
                    qr.composite_score = compute_composite_score(
                        acc, cod, qr.structure.composite, qr.performance.score,
                        weights,
                        evaluated=evaluated_axes,
                    )

                ms.question_results.append(qr)

                # Update completed count
                with self._lock:
                    prog = self._active_runs.get(run_id)
                    if prog:
                        prog.completed_questions += 1
                        prog.elapsed_ms = (time.time() - started_at) * 1000

                if progress_callback:
                    progress_callback(self._active_runs[run_id])

            # Aggregate model scores
            ms.accuracy_avg = (
                sum(accuracy_scores) / len(accuracy_scores)
                if accuracy_scores else 0.0
            )
            ms.code_avg = (
                sum(code_scores) / len(code_scores)
                if code_scores else 0.0
            )
            ms.structure_avg = (
                sum(structure_scores) / len(structure_scores)
                if structure_scores else 0.0
            )
            ms.speed_avg = (
                sum(speed_scores) / len(speed_scores)
                if speed_scores else 0.0
            )
            # S193 BJD-03: aggregate composite renormalized over the axes
            # this model was actually evaluated on.
            model_axes = {"structure", "speed"}
            if accuracy_scores:
                model_axes.add("accuracy")
            if code_scores:
                model_axes.add("code")
            ms.composite = compute_composite_score(
                ms.accuracy_avg, ms.code_avg,
                ms.structure_avg, ms.speed_avg,
                weights,
                evaluated=model_axes,
            )
            ms.questions_evaluated = len(ms.question_results)

            model_scores[model] = ms

            # S224: evict-between is the runner's default (spec 4.3): a
            # clean slate after each benchmarked model, through the
            # existing keep_alive=0 unload idiom; the governor snapshot
            # is invalidated alongside.
            if evict_between:
                _evict_loaded_models()

        # -- LLM-as-Judge evaluation (S89) --
        judge_summary = None
        if use_judge and judge_model and self._judge and BENCHMARK_JUDGE_AVAILABLE:
            if run_id not in self._cancelled:
                try:
                    # Collect all question-response pairs for judge evaluation
                    judge_inputs: list[dict[str, Any]] = []
                    for model_name, ms_data in model_scores.items():
                        for qr in ms_data.question_results:
                            if isinstance(qr, dict):
                                judge_inputs.append({
                                    "question_id": qr.get("question_id", ""),
                                    "question_text": qr.get("prompt", ""),
                                    "response": qr.get("response", ""),
                                    "model": model_name,
                                })
                            else:
                                judge_inputs.append({
                                    "question_id": qr.question_id,
                                    "question_text": qr.prompt,
                                    "response": qr.response,
                                    "model": model_name,
                                })

                    judge_summary = self._judge.evaluate_run(
                        run_id=run_id,
                        judge_model=judge_model,
                        question_responses=judge_inputs,
                        query_fn=qfn,
                    )
                    logger.info(
                        "Judge evaluation complete for run %s: %d evals, %d tokens",
                        run_id,
                        judge_summary.total_evaluations,
                        judge_summary.total_tokens,
                    )
                except Exception as e:
                    logger.error("Judge evaluation failed for run %s: %s", run_id, e)

        # Determine final status
        finished_at = time.time()
        if run_id in self._cancelled:
            status = RunStatus.CANCELLED
            with self._lock:
                self._cancelled.discard(run_id)
        else:
            status = RunStatus.COMPLETED

        # Build and persist result
        result = RunResult(
            run_id=run_id,
            profile=profile,
            models=models,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=(finished_at - started_at) * 1000,
            model_scores=model_scores,
            weight_preset=weight_preset_name,
            custom_weights=effective_custom_weights,
        )

        try:
            self._store.save_run(result)
        except Exception as e:
            logger.error("Failed to save run %s: %s", run_id, e)
            result.error = str(e)
            result.status = RunStatus.FAILED

        # S193 BMK-05: enforce the configured retention so the results DB
        # does not grow without bound (results_retention_days existed in
        # config but cleanup() had no caller).
        if result.status == RunStatus.COMPLETED:
            try:
                pdata = getattr(self._evaluator, "profiles_data", None) or {}
                retention = int(
                    pdata.get("runner", {}).get("results_retention_days", 90)
                )
                removed = self._store.cleanup(retention)
                if removed:
                    logger.info(
                        "Benchmark retention removed %d old runs", removed,
                    )
            except Exception as e:
                logger.debug("Benchmark retention cleanup skipped: %s", e)

        # Final progress update
        self._update_progress(
            run_id, result.status,
            error=result.error,
        )
        if progress_callback:
            progress_callback(self._active_runs[run_id])

    def _update_progress(
        self,
        run_id: str,
        status: RunStatus,
        total: int = 0,
        started_at: float = 0.0,
        current_model: str = "",
        current_question: str = "",
        error: str = "",
    ) -> None:
        """Update the progress tracking for a run."""
        with self._lock:
            prog = self._active_runs.get(run_id)
            if not prog:
                return
            prog.status = status
            if total:
                prog.total_questions = total
            if started_at:
                prog.started_at = started_at
            if current_model:
                prog.current_model = current_model
            if current_question:
                prog.current_question = current_question
            if error:
                prog.error = error
            if started_at or prog.started_at:
                prog.elapsed_ms = (time.time() - (started_at or prog.started_at)) * 1000


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

try:
    results_store = ResultsStore()
    benchmark_runner = BenchmarkRunner(store=results_store)
    BENCHMARK_RUNNER_AVAILABLE = True
except Exception as e:
    logger.warning("BenchmarkRunner init failed: %s", e)
    results_store = None  # type: ignore[assignment]
    benchmark_runner = None  # type: ignore[assignment]
    BENCHMARK_RUNNER_AVAILABLE = False
