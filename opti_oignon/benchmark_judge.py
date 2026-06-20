#!/usr/bin/env python3
"""
Benchmark Judge — S89

LLM-as-Judge evaluation layer for the autonomous benchmark engine.
Uses a selected model to evaluate responses on a structured rubric
(accuracy, relevance, completeness, conciseness, reasoning), producing
JSON scores with justifications. Integrates with json_repair for
robust parsing and stores results in SQLite.

Features:
  - Configurable rubric weights via benchmark_judge.yaml
  - Structured JSON output with json_repair fallback
  - Token cost tracking per evaluation
  - SQLite persistence for judge scores
  - Weighted blending of autonomous + judge scores
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
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from opti_oignon.json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    try:
        import importlib.util as _ilu
        import sys as _sys
        _jr_spec = _ilu.spec_from_file_location(
            "_json_repair",
            str(Path(__file__).parent / "json_repair.py"),
        )
        _jr_mod = _ilu.module_from_spec(_jr_spec)
        _sys.modules["_json_repair"] = _jr_mod  # Python 3.13: register before exec_module for dataclass safety
        _jr_spec.loader.exec_module(_jr_mod)
        repair_json = _jr_mod.repair_json
        JSON_REPAIR_AVAILABLE = True
    except Exception:
        JSON_REPAIR_AVAILABLE = False
        repair_json = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).parent / "config"
_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_DB_PATH = _DATA_DIR / "benchmark_results.db"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "benchmark_judge.yaml"

_DEFAULT_CONFIG: dict[str, Any] = {
    "autonomous_weight": 0.5,
    "judge_weight": 0.5,
    "max_judge_tokens": 1024,
    "judge_timeout": 60,
    "rubric": {
        "accuracy": {"description": "Factual correctness", "weight": 0.25},
        "relevance": {"description": "Answers the actual question", "weight": 0.25},
        "completeness": {"description": "Covers all aspects", "weight": 0.20},
        "conciseness": {"description": "No unnecessary verbosity", "weight": 0.15},
        "reasoning": {"description": "Logical chain of thought", "weight": 0.15},
    },
    "judge_system_prompt": (
        "You are an expert evaluator assessing the quality of an AI model's response.\n"
        "Evaluate the response against the original question using the rubric.\n"
        "Each dimension is scored from 1 (worst) to 10 (best).\n\n"
        "Rubric:\n"
        "- Accuracy (1-10): Factual correctness\n"
        "- Relevance (1-10): Answers the actual question\n"
        "- Completeness (1-10): Covers all aspects\n"
        "- Conciseness (1-10): No unnecessary verbosity\n"
        "- Reasoning (1-10): Logical chain of thought\n\n"
        "Respond with valid JSON only:\n"
        '{"accuracy": <int>, "relevance": <int>, "completeness": <int>, '
        '"conciseness": <int>, "reasoning": <int>, "justification": "<brief>"}'
    ),
}

RUBRIC_DIMENSIONS = ("accuracy", "relevance", "completeness", "conciseness", "reasoning")


def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load judge configuration from YAML, falling back to defaults."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if YAML_AVAILABLE and path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            merged = dict(_DEFAULT_CONFIG)
            merged.update(data)
            return merged
        except Exception as e:
            logger.warning("Failed to load judge config from %s: %s", path, e)
    return dict(_DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JudgeScore:
    """Evaluation scores from the LLM judge for a single question."""
    question_id: str = ""
    model: str = ""
    judge_model: str = ""
    accuracy: int = 0
    relevance: int = 0
    completeness: int = 0
    conciseness: int = 0
    reasoning: int = 0
    justification: str = ""
    weighted_score: float = 0.0
    tokens_used: int = 0
    eval_time_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "question_id": self.question_id,
            "model": self.model,
            "judge_model": self.judge_model,
            "accuracy": self.accuracy,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "conciseness": self.conciseness,
            "reasoning": self.reasoning,
            "justification": self.justification,
            "weighted_score": self.weighted_score,
            "tokens_used": self.tokens_used,
            "eval_time_ms": self.eval_time_ms,
            "error": self.error,
        }


@dataclass
class JudgeRunSummary:
    """Summary of judge evaluations for a benchmark run."""
    run_id: str = ""
    judge_model: str = ""
    total_evaluations: int = 0
    total_tokens: int = 0
    avg_score: float = 0.0
    scores_by_model: dict[str, float] = field(default_factory=dict)
    dimension_averages: dict[str, float] = field(default_factory=dict)
    errors: int = 0


# ---------------------------------------------------------------------------
# Judge score persistence
# ---------------------------------------------------------------------------

class JudgeStore:
    """SQLite storage for judge evaluation results.

    Uses the same database as benchmark_runner to keep results co-located.
    Creates a separate benchmark_judge_scores table.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or _DEFAULT_DB_PATH)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create judge scores table if it does not exist."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS benchmark_judge_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        question_id TEXT NOT NULL,
                        model TEXT NOT NULL,
                        judge_model TEXT NOT NULL,
                        accuracy INTEGER DEFAULT 0,
                        relevance INTEGER DEFAULT 0,
                        completeness INTEGER DEFAULT 0,
                        conciseness INTEGER DEFAULT 0,
                        reasoning INTEGER DEFAULT 0,
                        justification TEXT DEFAULT '',
                        weighted_score REAL DEFAULT 0.0,
                        tokens_used INTEGER DEFAULT 0,
                        eval_time_ms REAL DEFAULT 0.0,
                        error TEXT DEFAULT '',
                        timestamp REAL NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_judge_run
                        ON benchmark_judge_scores(run_id);
                    CREATE INDEX IF NOT EXISTS idx_judge_run_model
                        ON benchmark_judge_scores(run_id, model);
                """)
                conn.commit()
            finally:
                conn.close()

    def save_score(self, run_id: str, score: JudgeScore) -> None:
        """Persist a single judge score."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                conn.execute(
                    """INSERT INTO benchmark_judge_scores
                       (run_id, question_id, model, judge_model,
                        accuracy, relevance, completeness, conciseness,
                        reasoning, justification, weighted_score,
                        tokens_used, eval_time_ms, error, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        score.question_id,
                        score.model,
                        score.judge_model,
                        score.accuracy,
                        score.relevance,
                        score.completeness,
                        score.conciseness,
                        score.reasoning,
                        score.justification,
                        score.weighted_score,
                        score.tokens_used,
                        score.eval_time_ms,
                        score.error,
                        time.time(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def save_scores_batch(self, run_id: str, scores: list[JudgeScore]) -> None:
        """Persist multiple judge scores in a single transaction."""
        if not scores:
            return
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                for score in scores:
                    conn.execute(
                        """INSERT INTO benchmark_judge_scores
                           (run_id, question_id, model, judge_model,
                            accuracy, relevance, completeness, conciseness,
                            reasoning, justification, weighted_score,
                            tokens_used, eval_time_ms, error, timestamp)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            run_id,
                            score.question_id,
                            score.model,
                            score.judge_model,
                            score.accuracy,
                            score.relevance,
                            score.completeness,
                            score.conciseness,
                            score.reasoning,
                            score.justification,
                            score.weighted_score,
                            score.tokens_used,
                            score.eval_time_ms,
                            score.error,
                            time.time(),
                        ),
                    )
                conn.commit()
            finally:
                conn.close()

    def get_scores_for_run(self, run_id: str) -> list[dict]:
        """Retrieve all judge scores for a given run."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """SELECT * FROM benchmark_judge_scores
                       WHERE run_id = ? ORDER BY model, question_id""",
                    (run_id,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_scores_for_model(self, run_id: str, model: str) -> list[dict]:
        """Retrieve judge scores for a specific model in a run."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """SELECT * FROM benchmark_judge_scores
                       WHERE run_id = ? AND model = ?
                       ORDER BY question_id""",
                    (run_id, model),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_summary_for_run(self, run_id: str) -> dict:
        """Compute aggregate judge stats for a run."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """SELECT model, judge_model,
                              AVG(weighted_score) as avg_score,
                              AVG(accuracy) as avg_accuracy,
                              AVG(relevance) as avg_relevance,
                              AVG(completeness) as avg_completeness,
                              AVG(conciseness) as avg_conciseness,
                              AVG(reasoning) as avg_reasoning,
                              SUM(tokens_used) as total_tokens,
                              COUNT(*) as count,
                              SUM(CASE WHEN error != '' THEN 1 ELSE 0 END) as errors
                       FROM benchmark_judge_scores
                       WHERE run_id = ?
                       GROUP BY model""",
                    (run_id,),
                ).fetchall()
                if not rows:
                    return {}
                result = {
                    "run_id": run_id,
                    "judge_model": rows[0]["judge_model"] if rows else "",
                    "total_tokens": sum(r["total_tokens"] for r in rows),
                    "models": {},
                }
                for r in rows:
                    result["models"][r["model"]] = {
                        "avg_score": round(r["avg_score"], 3),
                        "avg_accuracy": round(r["avg_accuracy"], 2),
                        "avg_relevance": round(r["avg_relevance"], 2),
                        "avg_completeness": round(r["avg_completeness"], 2),
                        "avg_conciseness": round(r["avg_conciseness"], 2),
                        "avg_reasoning": round(r["avg_reasoning"], 2),
                        "evaluations": r["count"],
                        "errors": r["errors"],
                    }
                return result
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Judge evaluator
# ---------------------------------------------------------------------------

class BenchmarkJudge:
    """LLM-as-Judge evaluation engine.

    Takes a question and response, queries a judge model for structured
    rubric scores, parses the JSON output (with json_repair fallback),
    and computes a weighted score.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        store: JudgeStore | None = None,
        config_path: str | Path | None = None,
    ):
        self._config = config or _load_config(config_path)
        self._store = store or JudgeStore()
        self._rubric = self._config.get("rubric", _DEFAULT_CONFIG["rubric"])
        self._rubric_weights = {
            dim: self._rubric.get(dim, {}).get("weight", 0.2)
            for dim in RUBRIC_DIMENSIONS
        }
        self._autonomous_weight = self._config.get("autonomous_weight", 0.5)
        self._judge_weight = self._config.get("judge_weight", 0.5)
        self._max_tokens = self._config.get("max_judge_tokens", 1024)
        self._timeout = self._config.get("judge_timeout", 60)
        self._system_prompt = self._config.get(
            "judge_system_prompt",
            _DEFAULT_CONFIG["judge_system_prompt"],
        )

    @property
    def store(self) -> JudgeStore:
        return self._store

    @property
    def autonomous_weight(self) -> float:
        return self._autonomous_weight

    @property
    def judge_weight(self) -> float:
        return self._judge_weight

    @property
    def rubric_weights(self) -> dict[str, float]:
        return dict(self._rubric_weights)

    def build_eval_prompt(self, question: str, response: str) -> str:
        """Build the evaluation prompt sent to the judge model.

        Args:
            question: The original benchmark question.
            response: The model's response to evaluate.

        Returns:
            Formatted prompt string.
        """
        return (
            f"Question:\n{question}\n\n"
            f"Response to evaluate:\n{response}\n\n"
            "Evaluate this response according to the rubric. "
            "Respond with JSON only."
        )

    def parse_judge_response(self, raw: str) -> dict[str, Any]:
        """Parse the judge model's JSON response.

        Uses json_repair for robustness when raw JSON is malformed.

        Args:
            raw: Raw text from the judge model.

        Returns:
            Dictionary with rubric scores and justification.
        """
        # Try direct parse first
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return self._validate_scores(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: json_repair
        if JSON_REPAIR_AVAILABLE and repair_json is not None:
            try:
                data = repair_json(raw)
                if isinstance(data, dict):
                    return self._validate_scores(data)
            except Exception as e:
                logger.debug("json_repair failed for judge response: %s", e)

        # Last resort: extract numbers with simple heuristic
        return self._extract_scores_heuristic(raw)

    def _validate_scores(self, data: dict) -> dict[str, Any]:
        """Validate and clamp score values to 1-10 range."""
        result: dict[str, Any] = {}
        for dim in RUBRIC_DIMENSIONS:
            val = data.get(dim, 0)
            try:
                val = int(val)
            except (TypeError, ValueError):
                val = 0
            result[dim] = max(1, min(10, val)) if val > 0 else 0
        result["justification"] = str(data.get("justification", ""))
        return result

    def _extract_scores_heuristic(self, raw: str) -> dict[str, Any]:
        """Last-resort extraction of scores from unstructured text."""
        import re
        result: dict[str, Any] = {dim: 0 for dim in RUBRIC_DIMENSIONS}
        result["justification"] = ""
        for dim in RUBRIC_DIMENSIONS:
            pattern = rf'{dim}\s*[:=]\s*(\d+)'
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                val = int(match.group(1))
                result[dim] = max(1, min(10, val))
        return result

    def compute_weighted_score(self, scores: dict[str, Any]) -> float:
        """Compute weighted judge score from rubric dimensions.

        Args:
            scores: Dictionary with dimension names as keys and int scores.

        Returns:
            Weighted score normalized to 0.0-1.0 range.
        """
        total = 0.0
        weight_sum = 0.0
        for dim in RUBRIC_DIMENSIONS:
            val = scores.get(dim, 0)
            if isinstance(val, (int, float)) and val > 0:
                w = self._rubric_weights.get(dim, 0.2)
                total += (val / 10.0) * w
                weight_sum += w
        if weight_sum > 0:
            return round(total / weight_sum, 4)
        return 0.0

    def blend_scores(
        self,
        autonomous_score: float,
        judge_score: float,
    ) -> float:
        """Blend autonomous and judge scores using configured weights.

        Args:
            autonomous_score: Score from S88 autonomous metrics (0-1).
            judge_score: Score from LLM judge (0-1).

        Returns:
            Blended composite score (0-1).
        """
        return round(
            autonomous_score * self._autonomous_weight
            + judge_score * self._judge_weight,
            4,
        )

    def evaluate(
        self,
        question_id: str,
        question_text: str,
        response: str,
        model: str,
        judge_model: str,
        query_fn: Any = None,
    ) -> JudgeScore:
        """Evaluate a single response using the judge model.

        Args:
            question_id: Identifier of the benchmark question.
            question_text: The benchmark question text.
            response: The model's response to evaluate.
            model: The model that produced the response.
            judge_model: The model to use as judge.
            query_fn: Optional override for the LLM query function.
                       Signature: (model, prompt, timeout, max_tokens)
                       -> (text, ttft_ms, total_ms, token_count).

        Returns:
            JudgeScore with rubric scores and metadata.
        """
        score = JudgeScore(
            question_id=question_id,
            model=model,
            judge_model=judge_model,
        )

        prompt = self.build_eval_prompt(question_text, response)
        qfn = query_fn or _query_judge

        start = time.time()
        try:
            raw_text, _ttft, _total_ms, token_count = qfn(
                judge_model,
                prompt,
                self._timeout,
                self._max_tokens,
            )
            score.tokens_used = token_count
        except Exception as e:
            score.error = str(e)
            score.eval_time_ms = (time.time() - start) * 1000
            return score

        score.eval_time_ms = (time.time() - start) * 1000

        if not raw_text:
            score.error = "Empty response from judge model"
            return score

        # Parse scores
        parsed = self.parse_judge_response(raw_text)
        # S193 BJD-04: a judge output that yields no rubric dimension at
        # all is a parse failure, not a legitimate zero score; flag it so
        # evaluate_run excludes it from the model averages instead of
        # silently averaging a 0.0 in.
        if all(int(parsed.get(dim, 0) or 0) <= 0 for dim in RUBRIC_DIMENSIONS):
            score.error = "Unparseable judge response (no rubric scores found)"
            score.justification = str(parsed.get("justification", ""))
            return score
        score.accuracy = parsed.get("accuracy", 0)
        score.relevance = parsed.get("relevance", 0)
        score.completeness = parsed.get("completeness", 0)
        score.conciseness = parsed.get("conciseness", 0)
        score.reasoning = parsed.get("reasoning", 0)
        score.justification = parsed.get("justification", "")
        score.weighted_score = self.compute_weighted_score(parsed)

        return score

    def evaluate_run(
        self,
        run_id: str,
        judge_model: str,
        question_responses: list[dict[str, Any]],
        query_fn: Any = None,
    ) -> JudgeRunSummary:
        """Evaluate all responses in a benchmark run.

        Args:
            run_id: The benchmark run identifier.
            judge_model: Model to use as judge.
            question_responses: List of dicts with keys:
                question_id, question_text, response, model.
            query_fn: Optional override for LLM query function.

        Returns:
            JudgeRunSummary with aggregate stats.
        """
        summary = JudgeRunSummary(
            run_id=run_id,
            judge_model=judge_model,
        )

        scores: list[JudgeScore] = []
        model_scores: dict[str, list[float]] = {}
        dimension_totals: dict[str, list[float]] = {
            dim: [] for dim in RUBRIC_DIMENSIONS
        }

        for qr in question_responses:
            js = self.evaluate(
                question_id=qr.get("question_id", ""),
                question_text=qr.get("question_text", qr.get("prompt", "")),
                response=qr.get("response", ""),
                model=qr.get("model", ""),
                judge_model=judge_model,
                query_fn=query_fn,
            )
            scores.append(js)

            if js.error:
                summary.errors += 1
            else:
                model_name = js.model
                model_scores.setdefault(model_name, []).append(js.weighted_score)
                for dim in RUBRIC_DIMENSIONS:
                    val = getattr(js, dim, 0)
                    if val > 0:
                        dimension_totals[dim].append(float(val))

        # Persist all scores
        self._store.save_scores_batch(run_id, scores)

        # Aggregate
        summary.total_evaluations = len(scores)
        summary.total_tokens = sum(s.tokens_used for s in scores)

        all_scores = []
        for m, vals in model_scores.items():
            avg = sum(vals) / len(vals) if vals else 0.0
            summary.scores_by_model[m] = round(avg, 4)
            all_scores.extend(vals)

        summary.avg_score = (
            round(sum(all_scores) / len(all_scores), 4)
            if all_scores else 0.0
        )

        for dim, vals in dimension_totals.items():
            summary.dimension_averages[dim] = (
                round(sum(vals) / len(vals), 2)
                if vals else 0.0
            )

        return summary


# ---------------------------------------------------------------------------
# Default query function (mirrors _query_ollama from benchmark_runner)
# ---------------------------------------------------------------------------

# S193 BMK-02: per-timeout cached Ollama clients (RSN-02 idiom) so
# judge_timeout is actually enforced at the transport level.
_JUDGE_CLIENTS: dict[int, Any] = {}
_JUDGE_CLIENTS_LOCK = threading.Lock()


def _get_judge_client(timeout: int) -> Any:
    """Return a cached ollama.Client bound to the given timeout."""
    import ollama
    with _JUDGE_CLIENTS_LOCK:
        client = _JUDGE_CLIENTS.get(timeout)
        if client is None:
            client = ollama.Client(timeout=timeout)
            _JUDGE_CLIENTS[timeout] = client
        return client


def _judge_chunk_text(chunk: Any) -> str:
    """Extract message content from a stream chunk (dict or object form).

    S193 BMK-01: handle both client forms (MEM-06 idiom) instead of the
    dict-only access that silently emptied every judge response.
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


def _query_judge(
    model: str,
    prompt: str,
    timeout: int = 60,
    max_tokens: int = 1024,
) -> tuple[str, float, float, int]:
    """Send a prompt to Ollama for judge evaluation.

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

    try:
        client = _get_judge_client(timeout)
        stream = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={
                "num_predict": max_tokens,
            },
        )
        for chunk in stream:
            content = _judge_chunk_text(chunk)
            if content:
                if ttft == 0.0:
                    ttft = (time.time() - start) * 1000
                chunks.append(content)
                token_count += 1
    except Exception as e:
        logger.error("Judge query failed for model %s: %s", model, e)
        return "", 0.0, (time.time() - start) * 1000, 0

    total_ms = (time.time() - start) * 1000
    response = "".join(chunks)
    return response, ttft, total_ms, token_count


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

try:
    _judge_config = _load_config()
    judge_store = JudgeStore()
    benchmark_judge = BenchmarkJudge(config=_judge_config, store=judge_store)
    BENCHMARK_JUDGE_AVAILABLE = True
except Exception as e:
    logger.warning("BenchmarkJudge init failed: %s", e)
    judge_store = None  # type: ignore[assignment]
    benchmark_judge = None  # type: ignore[assignment]
    BENCHMARK_JUDGE_AVAILABLE = False
