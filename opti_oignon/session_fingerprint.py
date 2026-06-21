#!/usr/bin/env python3
"""
SESSION FINGERPRINT - OPTI-OIGNON v1.7.7 (S75)
================================================

Lightweight session fingerprinting for coding agent context injection.
Replaces compressed history with a compact ~150-200 token fingerprint
that captures the essential state of an ongoing coding session.

10 Dimensions:
  D1  task_type       — classify from task text
  D2  stack           — detect from file extensions + imports
  D3  hot_files       — touch counter per file, top 5
  D4  recent_bugs     — regex classify test failures
  D5  test_health     — rolling pass rate, failure types
  D6  momentum        — steps completed/remaining, velocity
  D7  domain_terms    — TF-IDF on names from hot files (batch)
  D8  dep_clusters    — import graph communities (batch)
  D9  user_preferences — learned from checkpoint decisions (persistent SQLite)
  D10 context_anchors — explicit invariants from fix loops

No LLM calls — purely heuristic/statistical.
O(1) incremental updates for on_step/on_test/on_checkpoint.

Author: Leon
"""

import collections
import hashlib
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)
# S136 audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)


try:
    import yaml as _yaml
except ImportError:
    _yaml = None

try:
    import json as _json
except ImportError:
    _json = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "config", "fingerprint.yaml"
)

_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "dimension_weights": {
        "task_type": 1.0,
        "stack": 1.0,
        "hot_files": 1.0,
        "recent_bugs": 1.0,
        "test_health": 1.0,
        "momentum": 1.0,
        "domain_terms": 0.8,
        "dep_clusters": 0.6,
        "user_preferences": 0.9,
        "context_anchors": 1.0,
    },
    "tfidf_refresh_interval": 5,
    "max_anchors": 5,
    "max_hot_files": 5,
    "max_domain_terms": 10,
    "max_dep_clusters": 5,
    "max_bug_history": 20,
    "sqlite_path": "fingerprint.db",
    "serialization_format": "yaml",
}


@dataclass
class FingerprintConfig:
    """Configuration for session fingerprinting."""

    enabled: bool = True
    dimension_weights: dict[str, float] = field(default_factory=dict)
    tfidf_refresh_interval: int = 5
    max_anchors: int = 5
    max_hot_files: int = 5
    max_domain_terms: int = 10
    max_dep_clusters: int = 5
    max_bug_history: int = 20
    sqlite_path: str = "fingerprint.db"
    serialization_format: str = "yaml"


def _load_config() -> FingerprintConfig:
    """Load fingerprint configuration from YAML with safe defaults."""
    raw = dict(_DEFAULT_CONFIG)
    try:
        if _yaml is not None and os.path.isfile(_CONFIG_PATH):
            with open(_CONFIG_PATH, encoding="utf-8") as fh:
                loaded = _yaml.safe_load(fh) or {}
            raw.update(loaded)
    except Exception as exc:
        logger.warning("Failed to load fingerprint config: %s", exc)

    return FingerprintConfig(
        enabled=raw.get("enabled", True),
        dimension_weights=raw.get("dimension_weights", {}),
        tfidf_refresh_interval=raw.get("tfidf_refresh_interval", 5),
        max_anchors=raw.get("max_anchors", 5),
        max_hot_files=raw.get("max_hot_files", 5),
        max_domain_terms=raw.get("max_domain_terms", 10),
        max_dep_clusters=raw.get("max_dep_clusters", 5),
        max_bug_history=raw.get("max_bug_history", 20),
        sqlite_path=raw.get("sqlite_path", "fingerprint.db"),
        serialization_format=raw.get("serialization_format", "yaml"),
    )


# ---------------------------------------------------------------------------
# D1: Task Type Classification
# ---------------------------------------------------------------------------

class TaskType:
    """Task type constants."""
    CREATE = "create"
    REFACTOR = "refactor"
    BUG_FIX = "bug_fix"
    TEST = "test"
    DOCS = "docs"
    FEATURE = "feature"
    UNKNOWN = "unknown"


_TASK_PATTERNS: list[tuple[str, str]] = [
    (r"\b(fix|bug|error|crash|broken|issue|patch|hotfix)\b", TaskType.BUG_FIX),
    (r"\b(refactor|clean|reorganize|restructure|simplify|extract)\b", TaskType.REFACTOR),
    (r"\b(test|spec|coverage|assert|unittest|pytest)\b", TaskType.TEST),
    (r"\b(doc|readme|comment|docstring|changelog|guide)\b", TaskType.DOCS),
    (r"\b(create|new|add|implement|build|scaffold|init)\b", TaskType.CREATE),
    (r"\b(feature|enhance|improve|upgrade|extend|support)\b", TaskType.FEATURE),
]

# Complexity heuristics based on word count and keyword density
_COMPLEXITY_THRESHOLDS = {
    "simple": 20,
    "moderate": 60,
    "complex": 150,
}


def classify_task(text: str) -> dict[str, Any]:
    """Classify task type and complexity from task description.

    Returns:
        Dict with 'type', 'complexity', and 'confidence' keys.
    """
    if not text:
        return {"type": TaskType.UNKNOWN, "complexity": "simple", "confidence": 0.0}

    lower = text.lower()
    scores: dict[str, int] = collections.defaultdict(int)

    for pattern, task_type in _TASK_PATTERNS:
        matches = re.findall(pattern, lower)
        scores[task_type] += len(matches)

    if not any(scores.values()):
        task_type = TaskType.UNKNOWN
        confidence = 0.0
    else:
        task_type = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[task_type] / total if total > 0 else 0.0

    # Complexity from word count
    word_count = len(text.split())
    if word_count < _COMPLEXITY_THRESHOLDS["simple"]:
        complexity = "simple"
    elif word_count < _COMPLEXITY_THRESHOLDS["moderate"]:
        complexity = "moderate"
    elif word_count < _COMPLEXITY_THRESHOLDS["complex"]:
        complexity = "complex"
    else:
        complexity = "very_complex"

    return {
        "type": task_type,
        "complexity": complexity,
        "confidence": round(confidence, 3),
    }


# ---------------------------------------------------------------------------
# D2: Stack Detection
# ---------------------------------------------------------------------------

_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "react",
    ".tsx": "react-ts",
    ".svelte": "svelte",
    ".vue": "vue",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c-header",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".r": "r",
    ".R": "r",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".css": "css",
    ".scss": "scss",
    ".html": "html",
    ".md": "markdown",
    ".dockerfile": "docker",
}

_IMPORT_PATTERNS: list[tuple[str, str]] = [
    (r"^import\s+(\w+)", "python"),
    (r"^from\s+(\w+)\s+import", "python"),
    (r"^const\s+\w+\s*=\s*require\(", "javascript"),
    (r"^import\s+.*\s+from\s+['\"]", "javascript"),
    (r"^use\s+\w+::", "rust"),
    (r"^package\s+\w+", "go"),
    (r"^import\s+\w+\.\w+", "java"),
    (r"^require\s+['\"]", "ruby"),
    (r"^library\(", "r"),
    (r"^using\s+\w+", "csharp"),
]


def detect_stack(file_paths: list[str], file_contents: dict[str, str] | None = None) -> dict[str, Any]:
    """Detect technology stack from file extensions and import patterns.

    Args:
        file_paths: List of file paths touched in the session.
        file_contents: Optional dict mapping paths to file contents for import analysis.

    Returns:
        Dict with 'languages' (sorted by frequency), 'primary', 'frameworks'.
    """
    lang_counts: dict[str, int] = collections.defaultdict(int)

    # Extension-based detection
    for path in file_paths:
        _, ext = os.path.splitext(path.lower())
        if ext in _EXTENSION_MAP:
            lang_counts[_EXTENSION_MAP[ext]] += 1

    # Import-based detection from file contents
    frameworks: set[str] = set()
    if file_contents:
        for _path, content in file_contents.items():
            for line in content.split("\n")[:50]:
                line = line.strip()
                for pattern, lang in _IMPORT_PATTERNS:
                    if re.match(pattern, line):
                        lang_counts[lang] += 1
                        break
                # Framework detection
                if "fastapi" in line.lower():
                    frameworks.add("fastapi")
                elif "flask" in line.lower():
                    frameworks.add("flask")
                elif "django" in line.lower():
                    frameworks.add("django")
                elif "svelte" in line.lower():
                    frameworks.add("svelte")
                elif "react" in line.lower():
                    frameworks.add("react")
                elif "pytest" in line.lower():
                    frameworks.add("pytest")

    sorted_langs = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_langs[0][0] if sorted_langs else "unknown"

    return {
        "languages": {lang: count for lang, count in sorted_langs},
        "primary": primary,
        "frameworks": sorted(frameworks),
    }


# ---------------------------------------------------------------------------
# D3: Hot Files
# ---------------------------------------------------------------------------

@dataclass
class HotFilesTracker:
    """Track file touch frequency for the session."""

    _touches: dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))
    _sizes: dict[str, int] = field(default_factory=dict)

    def touch(self, path: str, size: int = 0) -> None:
        """Record a file touch."""
        self._touches[path] += 1
        if size > 0:
            self._sizes[path] = size

    def top(self, n: int = 5) -> list[dict[str, Any]]:
        """Return the top N most-touched files."""
        sorted_files = sorted(self._touches.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "path": path,
                "touches": count,
                "size": self._sizes.get(path, 0),
            }
            for path, count in sorted_files[:n]
        ]

    @property
    def file_count(self) -> int:
        """Total number of unique files touched."""
        return len(self._touches)

    @property
    def avg_file_size(self) -> int:
        """Average file size of files with known sizes."""
        sizes = [s for s in self._sizes.values() if s > 0]
        return int(sum(sizes) / len(sizes)) if sizes else 0

    def serialize(self, max_files: int = 5) -> dict[str, Any]:
        """Serialize hot files state."""
        return {
            "top": self.top(max_files),
            "file_count": self.file_count,
            "avg_file_size": self.avg_file_size,
        }


# ---------------------------------------------------------------------------
# D4: Recent Bugs
# ---------------------------------------------------------------------------

_BUG_PATTERNS: list[tuple[str, str]] = [
    (r"assert(ion)?error", "assertion"),
    (r"import\s*error|modulenotfounderror|no module named", "import"),
    (r"type\s*error|expected.*got|incompatible type", "type"),
    (r"syntax\s*error|unexpected token|invalid syntax", "syntax"),
    (r"runtime\s*error|recursion|stack overflow|segfault", "runtime"),
    (r"key\s*error|index\s*error|out of range", "index"),
    (r"attribute\s*error|has no attribute", "attribute"),
    (r"value\s*error|invalid (literal|value)", "value"),
    (r"file\s*not\s*found|no such file|permission denied", "io"),
    (r"timeout|timed? out|deadline exceeded", "timeout"),
]


def classify_bug(error_text: str) -> str:
    """Classify a test failure/error into a bug category.

    Returns:
        Bug category string (assertion, import, type, syntax, runtime, etc.)
    """
    if not error_text:
        return "unknown"

    lower = error_text.lower()
    for pattern, category in _BUG_PATTERNS:
        if re.search(pattern, lower):
            return category

    return "unknown"


@dataclass
class BugTracker:
    """Track recent bug classifications."""

    _bugs: list[dict[str, Any]] = field(default_factory=list)
    _max_history: int = 20

    def record(self, error_text: str, step: int = 0) -> str:
        """Record a bug from error text. Returns the classified category."""
        category = classify_bug(error_text)
        entry = {
            "category": category,
            "step": step,
            "timestamp": time.time(),
            "snippet": error_text[:200],
        }
        self._bugs.append(entry)
        if len(self._bugs) > self._max_history:
            self._bugs = self._bugs[-self._max_history:]
        return category

    @property
    def category_counts(self) -> dict[str, int]:
        """Count of bugs by category."""
        counts: dict[str, int] = collections.defaultdict(int)
        for bug in self._bugs:
            counts[bug["category"]] += 1
        return dict(counts)

    @property
    def recent(self) -> list[dict[str, Any]]:
        """Last 5 bugs."""
        return self._bugs[-5:]

    def serialize(self) -> dict[str, Any]:
        """Serialize bug tracker state."""
        return {
            "total": len(self._bugs),
            "categories": self.category_counts,
            "recent": [
                {"category": b["category"], "step": b["step"]}
                for b in self.recent
            ],
        }


# ---------------------------------------------------------------------------
# D5: Test Health
# ---------------------------------------------------------------------------

@dataclass
class SuiteHealthTracker:
    """Track rolling test pass rate and failure types."""

    _results: list[bool] = field(default_factory=list)
    _failure_types: list[str] = field(default_factory=list)
    _window_size: int = 20

    def record(self, passed: bool, failure_type: str = "") -> None:
        """Record a test run result."""
        self._results.append(passed)
        if not passed and failure_type:
            self._failure_types.append(failure_type)
        if len(self._results) > self._window_size:
            self._results = self._results[-self._window_size:]
        if len(self._failure_types) > self._window_size:
            self._failure_types = self._failure_types[-self._window_size:]

    @property
    def pass_rate(self) -> float:
        """Rolling pass rate over the window."""
        if not self._results:
            return 1.0
        return sum(1 for r in self._results if r) / len(self._results)

    @property
    def common_failure_types(self) -> list[tuple[str, int]]:
        """Most common failure types, sorted by frequency."""
        counts: dict[str, int] = collections.defaultdict(int)
        for ft in self._failure_types:
            counts[ft] += 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)

    @property
    def last_result(self) -> bool | None:
        """Last test result."""
        return self._results[-1] if self._results else None

    @property
    def total_runs(self) -> int:
        """Total test runs recorded."""
        return len(self._results)

    def serialize(self) -> dict[str, Any]:
        """Serialize test health state."""
        return {
            "pass_rate": round(self.pass_rate, 3),
            "total_runs": self.total_runs,
            "last_passed": self.last_result,
            "common_failures": dict(self.common_failure_types[:3]),
        }


# ---------------------------------------------------------------------------
# D6: Session Momentum
# ---------------------------------------------------------------------------

@dataclass
class MomentumTracker:
    """Track session progress and velocity."""

    steps_completed: int = 0
    steps_remaining: int = 0
    stuck_count: int = 0
    _step_timestamps: list[float] = field(default_factory=list)

    def complete_step(self) -> None:
        """Record a step completion."""
        self.steps_completed += 1
        if self.steps_remaining > 0:
            self.steps_remaining -= 1
        self._step_timestamps.append(time.time())

    def set_total_steps(self, total: int) -> None:
        """Set total steps from plan."""
        self.steps_remaining = max(0, total - self.steps_completed)

    def record_stuck(self) -> None:
        """Record a stuck event (fix loop, repeated failure)."""
        self.stuck_count += 1

    @property
    def velocity(self) -> float:
        """Steps per minute over the last 5 steps."""
        if len(self._step_timestamps) < 2:
            return 0.0
        recent = self._step_timestamps[-5:]
        elapsed = recent[-1] - recent[0]
        if elapsed <= 0:
            return 0.0
        return (len(recent) - 1) / (elapsed / 60.0)

    @property
    def progress_ratio(self) -> float:
        """Fraction of total steps completed."""
        total = self.steps_completed + self.steps_remaining
        if total <= 0:
            return 0.0
        return self.steps_completed / total

    def serialize(self) -> dict[str, Any]:
        """Serialize momentum state."""
        return {
            "completed": self.steps_completed,
            "remaining": self.steps_remaining,
            "stuck_count": self.stuck_count,
            "velocity": round(self.velocity, 2),
            "progress": round(self.progress_ratio, 3),
        }


# ---------------------------------------------------------------------------
# D7: Domain Terms (TF-IDF)
# ---------------------------------------------------------------------------

_NAME_PATTERN = re.compile(
    r"\b(?:def|class|function|const|let|var|type)\s+([a-zA-Z_]\w{2,})\b"
)
_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_STOP_WORDS = frozenset({
    "self", "none", "true", "false", "return", "import", "from",
    "class", "def", "function", "const", "let", "var", "type",
    "init", "main", "test", "str", "int", "float", "bool", "list",
    "dict", "set", "tuple", "any", "optional", "union",
})


def extract_terms(code: str) -> list[str]:
    """Extract meaningful identifier terms from code.

    Splits camelCase and snake_case, filters stop words.
    """
    identifiers = _NAME_PATTERN.findall(code)
    terms: list[str] = []

    for ident in identifiers:
        # Split snake_case
        parts = ident.split("_")
        for part in parts:
            # Split camelCase
            sub_parts = _CAMEL_SPLIT.split(part)
            for sp in sub_parts:
                sp_lower = sp.lower()
                if len(sp_lower) >= 3 and sp_lower not in _STOP_WORDS:
                    terms.append(sp_lower)

    return terms


def compute_tfidf(
    doc_terms: list[list[str]], max_terms: int = 10
) -> list[tuple[str, float]]:
    """Compute TF-IDF scores across documents.

    Args:
        doc_terms: List of term lists (one per document/file).
        max_terms: Maximum terms to return.

    Returns:
        Sorted list of (term, tfidf_score) tuples.
    """
    if not doc_terms:
        return []

    num_docs = len(doc_terms)

    # Document frequency
    df: dict[str, int] = collections.defaultdict(int)
    for terms in doc_terms:
        seen = set(terms)
        for t in seen:
            df[t] += 1

    # TF-IDF per term across all docs
    tfidf_scores: dict[str, float] = collections.defaultdict(float)
    for terms in doc_terms:
        if not terms:
            continue
        tf_counts = collections.Counter(terms)
        doc_len = len(terms)
        for term, count in tf_counts.items():
            tf = count / doc_len
            idf = math.log(1 + num_docs / (1 + df.get(term, 0)))
            tfidf_scores[term] += tf * idf

    sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_terms[:max_terms]


@dataclass
class DomainTermsTracker:
    """Track domain-specific terms via TF-IDF."""

    _file_terms: dict[str, list[str]] = field(default_factory=dict)
    _cached_tfidf: list[tuple[str, float]] = field(default_factory=list)
    _steps_since_refresh: int = 0
    _refresh_interval: int = 5
    _max_terms: int = 10

    def update_file(self, path: str, content: str) -> None:
        """Update terms for a file."""
        self._file_terms[path] = extract_terms(content)
        self._steps_since_refresh += 1

    def should_refresh(self) -> bool:
        """Check if TF-IDF should be recomputed."""
        return self._steps_since_refresh >= self._refresh_interval

    def refresh(self) -> list[tuple[str, float]]:
        """Recompute TF-IDF scores."""
        doc_terms = list(self._file_terms.values())
        self._cached_tfidf = compute_tfidf(doc_terms, self._max_terms)
        self._steps_since_refresh = 0
        return self._cached_tfidf

    @property
    def terms(self) -> list[tuple[str, float]]:
        """Current TF-IDF terms (may be stale until refresh)."""
        return list(self._cached_tfidf)

    def serialize(self) -> dict[str, Any]:
        """Serialize domain terms."""
        return {
            "terms": {t: round(s, 3) for t, s in self._cached_tfidf[:self._max_terms]},
            "file_count": len(self._file_terms),
        }


# ---------------------------------------------------------------------------
# D8: Dependency Clusters
# ---------------------------------------------------------------------------

_PYTHON_IMPORT = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE
)


def build_import_graph(file_contents: dict[str, str]) -> dict[str, set[str]]:
    """Build a module-level import graph from file contents.

    Args:
        file_contents: Dict mapping file paths to their contents.

    Returns:
        Adjacency dict: module -> set of imported modules.
    """
    graph: dict[str, set[str]] = collections.defaultdict(set)

    for path, content in file_contents.items():
        module = os.path.splitext(os.path.basename(path))[0]
        for match in _PYTHON_IMPORT.finditer(content):
            imported = match.group(1) or match.group(2)
            if imported:
                # Take root module
                root = imported.split(".")[0]
                if root != module:
                    graph[module].add(root)

    return dict(graph)


def find_clusters(graph: dict[str, set[str]], max_clusters: int = 5) -> list[list[str]]:
    """Find connected component clusters in the import graph.

    Simple BFS-based connected components.

    Args:
        graph: Adjacency dict from build_import_graph.
        max_clusters: Maximum clusters to return.

    Returns:
        List of clusters (each a sorted list of module names), largest first.
    """
    # Build undirected adjacency
    undirected: dict[str, set[str]] = collections.defaultdict(set)
    all_nodes: set[str] = set()
    for module, deps in graph.items():
        all_nodes.add(module)
        for dep in deps:
            all_nodes.add(dep)
            undirected[module].add(dep)
            undirected[dep].add(module)

    visited: set[str] = set()
    clusters: list[list[str]] = []

    for node in all_nodes:
        if node in visited:
            continue
        # BFS
        component: list[str] = []
        queue = [node]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in undirected.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        clusters.append(sorted(component))

    # Sort by size descending
    clusters.sort(key=len, reverse=True)
    return clusters[:max_clusters]


@dataclass
class DepClustersTracker:
    """Track dependency clusters from import graph."""

    _clusters: list[list[str]] = field(default_factory=list)
    _graph: dict[str, set[str]] = field(default_factory=dict)
    _computed: bool = False

    def compute(self, file_contents: dict[str, str], max_clusters: int = 5) -> None:
        """Compute clusters from file contents."""
        self._graph = build_import_graph(file_contents)
        self._clusters = find_clusters(self._graph, max_clusters)
        self._computed = True

    @property
    def clusters(self) -> list[list[str]]:
        """Current clusters."""
        return list(self._clusters)

    def serialize(self) -> dict[str, Any]:
        """Serialize dependency clusters."""
        return {
            "clusters": self._clusters,
            "module_count": len(set(
                m for c in self._clusters for m in c
            )),
        }


# ---------------------------------------------------------------------------
# D9: User Preferences (persistent SQLite)
# ---------------------------------------------------------------------------

class UserPreferencesStore:
    """Persistent store for user preference signals from checkpoints.

    Stores cumulative approve/modify/abort ratios and patterns.
    """

    def __init__(self, db_path: str = "fingerprint.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        try:
            with self._lock:
                conn = _safe_connect(self._db_path)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS preferences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        decision TEXT NOT NULL,
                        phase TEXT NOT NULL DEFAULT '',
                        context TEXT NOT NULL DEFAULT '',
                        timestamp REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS preference_summary (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
                conn.close()
        except Exception as exc:
            logger.warning("Failed to init fingerprint DB: %s", exc)

    def record(self, decision: str, phase: str = "", context: str = "") -> None:
        """Record a checkpoint decision.

        Args:
            decision: One of 'approve', 'modify', 'abort'.
            phase: The phase when the decision was made.
            context: Additional context string.
        """
        try:
            with self._lock:
                conn = _safe_connect(self._db_path)
                conn.execute(
                    "INSERT INTO preferences (decision, phase, context, timestamp) "
                    "VALUES (?, ?, ?, ?)",
                    (decision, phase, context, time.time()),
                )
                conn.commit()
                conn.close()
        except Exception as exc:
            logger.warning("Failed to record preference: %s", exc)

    def get_ratios(self) -> dict[str, float]:
        """Get cumulative decision ratios.

        Returns:
            Dict with 'approve', 'modify', 'abort' ratios (0.0-1.0).
        """
        try:
            with self._lock:
                conn = _safe_connect(self._db_path)
                cursor = conn.execute(
                    "SELECT decision, COUNT(*) FROM preferences GROUP BY decision"
                )
                counts = dict(cursor.fetchall())
                conn.close()

            total = sum(counts.values())
            if total == 0:
                return {"approve": 0.0, "modify": 0.0, "abort": 0.0}

            return {
                "approve": round(counts.get("approve", 0) / total, 3),
                "modify": round(counts.get("modify", 0) / total, 3),
                "abort": round(counts.get("abort", 0) / total, 3),
            }
        except Exception as exc:
            logger.warning("Failed to get preference ratios: %s", exc)
            return {"approve": 0.0, "modify": 0.0, "abort": 0.0}

    def get_phase_preferences(self) -> dict[str, dict[str, int]]:
        """Get decision counts per phase.

        Returns:
            Dict mapping phase -> {decision -> count}.
        """
        try:
            with self._lock:
                conn = _safe_connect(self._db_path)
                cursor = conn.execute(
                    "SELECT phase, decision, COUNT(*) FROM preferences "
                    "WHERE phase != '' GROUP BY phase, decision"
                )
                rows = cursor.fetchall()
                conn.close()

            result: dict[str, dict[str, int]] = collections.defaultdict(
                lambda: collections.defaultdict(int)
            )
            for phase, decision, count in rows:
                result[phase][decision] = count
            return {k: dict(v) for k, v in result.items()}
        except Exception as exc:
            logger.warning("Failed to get phase preferences: %s", exc)
            return {}

    @property
    def total_decisions(self) -> int:
        """Total number of recorded decisions."""
        try:
            with self._lock:
                conn = _safe_connect(self._db_path)
                cursor = conn.execute("SELECT COUNT(*) FROM preferences")
                count = cursor.fetchone()[0]
                conn.close()
                return count
        except Exception:
            return 0

    def serialize(self) -> dict[str, Any]:
        """Serialize user preferences."""
        return {
            "ratios": self.get_ratios(),
            "total_decisions": self.total_decisions,
            "phase_prefs": self.get_phase_preferences(),
        }

    def close(self) -> None:
        """No-op for API consistency (connections are per-call)."""
        pass


# ---------------------------------------------------------------------------
# D10: Context Anchors
# ---------------------------------------------------------------------------

@dataclass
class ContextAnchorsTracker:
    """Track explicit invariants discovered during fix loops.

    Max 5 entries. Each anchor is a short string describing a constraint
    that must be preserved.
    """

    _anchors: list[dict[str, Any]] = field(default_factory=list)
    _max_anchors: int = 5

    def add(self, text: str, source: str = "fix_loop") -> None:
        """Add a context anchor.

        Args:
            text: The invariant description (max 200 chars).
            source: Where this anchor was discovered.
        """
        if not text:
            return

        text = text[:200]

        # Deduplicate by content hash
        text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8]
        for anchor in self._anchors:
            if anchor.get("hash") == text_hash:
                return

        self._anchors.append({
            "text": text,
            "source": source,
            "hash": text_hash,
            "timestamp": time.time(),
        })

        # Evict oldest if over limit
        if len(self._anchors) > self._max_anchors:
            self._anchors = self._anchors[-self._max_anchors:]

    @property
    def anchors(self) -> list[str]:
        """Current anchor texts."""
        return [a["text"] for a in self._anchors]

    def serialize(self) -> dict[str, Any]:
        """Serialize context anchors."""
        return {
            "anchors": self.anchors,
            "count": len(self._anchors),
        }


# ---------------------------------------------------------------------------
# FingerprintManager
# ---------------------------------------------------------------------------

class FingerprintManager:
    """Lightweight session fingerprint manager.

    Maintains 10 dimensions incrementally updated via on_step/on_test/on_checkpoint.
    Serializes to a compact blob (~150-200 tokens) for context injection.

    Dimensions:
        D1  task_type       — from classify_task()
        D2  stack           — from detect_stack()
        D3  hot_files       — HotFilesTracker
        D4  recent_bugs     — BugTracker
        D5  test_health     — SuiteHealthTracker
        D6  momentum        — MomentumTracker
        D7  domain_terms    — DomainTermsTracker (batch, every N steps)
        D8  dep_clusters    — DepClustersTracker (batch, once at planning)
        D9  user_preferences — UserPreferencesStore (persistent SQLite)
        D10 context_anchors — ContextAnchorsTracker
    """

    def __init__(
        self,
        config: FingerprintConfig | None = None,
        preferences_store: UserPreferencesStore | None = None,
    ):
        """Initialize the fingerprint manager.

        Args:
            config: Configuration. Loaded from YAML if None.
            preferences_store: Persistent store for D9.
                Created with config sqlite_path if None.
        """
        self._config = config or _load_config()

        # D1: Task type (set once at task start)
        self._task_info: dict[str, Any] = {
            "type": TaskType.UNKNOWN,
            "complexity": "simple",
            "confidence": 0.0,
        }

        # D2: Stack (updated incrementally)
        self._file_paths: list[str] = []
        self._file_contents: dict[str, str] = {}
        self._stack: dict[str, Any] = {
            "languages": {},
            "primary": "unknown",
            "frameworks": [],
        }

        # D3: Hot files
        self._hot_files = HotFilesTracker()

        # D4: Recent bugs
        self._bugs = BugTracker()
        self._bugs._max_history = self._config.max_bug_history

        # D5: Test health
        self._test_health = SuiteHealthTracker()

        # D6: Momentum
        self._momentum = MomentumTracker()

        # D7: Domain terms
        self._domain_terms = DomainTermsTracker()
        self._domain_terms._refresh_interval = self._config.tfidf_refresh_interval
        self._domain_terms._max_terms = self._config.max_domain_terms

        # D8: Dependency clusters
        self._dep_clusters = DepClustersTracker()

        # D9: User preferences (persistent)
        if preferences_store is not None:
            self._preferences = preferences_store
        else:
            self._preferences = UserPreferencesStore(
                db_path=self._config.sqlite_path
            )

        # D10: Context anchors
        self._anchors = ContextAnchorsTracker()
        self._anchors._max_anchors = self._config.max_anchors

        # Internal state
        self._step_count = 0
        self._lock = threading.Lock()

    # -----------------------------------------------------------------
    # Task initialization
    # -----------------------------------------------------------------

    def set_task(self, task_text: str, total_steps: int = 0) -> None:
        """Initialize fingerprint for a new task.

        Args:
            task_text: The task description text.
            total_steps: Total planned steps (from CodingPlan).
        """
        self._task_info = classify_task(task_text)
        self._momentum.set_total_steps(total_steps)

    # -----------------------------------------------------------------
    # Incremental update hooks (O(1) per call)
    # -----------------------------------------------------------------

    def on_step(self, step: dict[str, Any]) -> None:
        """Update fingerprint after a coding step.

        Expected step keys:
            file_path: str — file affected
            step_type: str — 'create', 'edit', 'bash', 'test'
            content: str — file content (optional, for D7)
            size: int — file size in bytes (optional)
            completed: bool — whether step succeeded

        Updates: D3 (hot_files), D6 (momentum), D7 (domain_terms, batched).
        """
        with self._lock:
            self._step_count += 1
            file_path = step.get("file_path", "")
            content = step.get("content", "")
            size = step.get("size", len(content.encode()) if content else 0)
            completed = step.get("completed", True)

            # D3: Hot files
            if file_path:
                self._hot_files.touch(file_path, size)
                self._file_paths.append(file_path)

            # D6: Momentum
            if completed:
                self._momentum.complete_step()

            # D2: Stack update (lightweight, just extension counting)
            if file_path:
                self._stack = detect_stack(self._file_paths)

            # D7: Domain terms (batch, store content for later)
            if file_path and content:
                self._file_contents[file_path] = content
                self._domain_terms.update_file(file_path, content)
                if self._domain_terms.should_refresh():
                    self._domain_terms.refresh()

    def on_test(self, result: dict[str, Any]) -> None:
        """Update fingerprint after a test run.

        Expected result keys:
            passed: bool — whether tests passed
            output: str — test output text
            error: str — error text (if failed)

        Updates: D4 (recent_bugs), D5 (test_health), D6 (momentum on failure).
        """
        with self._lock:
            passed = result.get("passed", True)
            output = result.get("output", "")
            error = result.get("error", "")

            # D5: Test health
            failure_type = ""
            if not passed:
                error_text = error or output
                # D4: Recent bugs
                failure_type = self._bugs.record(
                    error_text, step=self._step_count
                )
                # D6: Momentum stuck
                self._momentum.record_stuck()

            self._test_health.record(passed, failure_type)

    def on_checkpoint(self, decision: dict[str, Any]) -> None:
        """Update fingerprint after a human checkpoint.

        Expected decision keys:
            action: str — 'approve', 'modify', 'abort'
            phase: str — current phase name
            context: str — optional context
            anchor: str — optional context anchor to add (D10)

        Updates: D9 (user_preferences), D10 (context_anchors).
        """
        with self._lock:
            action = decision.get("action", "")
            phase = decision.get("phase", "")
            context = decision.get("context", "")
            anchor = decision.get("anchor", "")

            # D9: User preferences (persistent)
            if action in ("approve", "modify", "abort"):
                self._preferences.record(action, phase, context)

            # D10: Context anchors
            if anchor:
                self._anchors.add(anchor, source=f"checkpoint:{phase}")

    def add_anchor(self, text: str, source: str = "manual") -> None:
        """Manually add a context anchor.

        Args:
            text: The invariant description.
            source: Where this anchor was discovered.
        """
        with self._lock:
            self._anchors.add(text, source)

    # -----------------------------------------------------------------
    # Batch operations (called infrequently)
    # -----------------------------------------------------------------

    def compute_dep_clusters(self, file_contents: dict[str, str] | None = None) -> None:
        """Compute dependency clusters from file contents.

        Called once at planning phase, or manually.
        """
        contents = file_contents or self._file_contents
        if contents:
            self._dep_clusters.compute(
                contents, self._config.max_dep_clusters
            )

    def refresh_domain_terms(self) -> None:
        """Force a TF-IDF refresh."""
        self._domain_terms.refresh()

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        """Serialize the fingerprint to a compact dict.

        Target: ~150-200 tokens when converted to YAML/JSON.
        """
        weights = self._config.dimension_weights

        fp: dict[str, Any] = {}

        # Only include dimensions with non-zero weight
        if weights.get("task_type", 1.0) > 0:
            fp["task"] = {
                "type": self._task_info["type"],
                "complexity": self._task_info["complexity"],
            }

        if weights.get("stack", 1.0) > 0:
            fp["stack"] = {
                "primary": self._stack["primary"],
                "langs": list(self._stack.get("languages", {}).keys())[:4],
            }
            if self._stack.get("frameworks"):
                fp["stack"]["fw"] = self._stack["frameworks"][:3]

        if weights.get("hot_files", 1.0) > 0:
            hot = self._hot_files.serialize(self._config.max_hot_files)
            if hot["file_count"] > 0:
                fp["files"] = {
                    "top": [
                        f["path"].rsplit("/", 1)[-1]
                        for f in hot["top"][:3]
                    ],
                    "n": hot["file_count"],
                }

        if weights.get("recent_bugs", 1.0) > 0:
            bugs = self._bugs.serialize()
            if bugs["total"] > 0:
                fp["bugs"] = bugs["categories"]

        if weights.get("test_health", 1.0) > 0:
            health = self._test_health.serialize()
            if health["total_runs"] > 0:
                fp["tests"] = {
                    "rate": health["pass_rate"],
                    "runs": health["total_runs"],
                    "last": health["last_passed"],
                }

        if weights.get("momentum", 1.0) > 0:
            mom = self._momentum.serialize()
            if mom["completed"] > 0 or mom["remaining"] > 0:
                fp["momentum"] = {
                    "done": mom["completed"],
                    "left": mom["remaining"],
                    "vel": mom["velocity"],
                }
                if mom["stuck_count"] > 0:
                    fp["momentum"]["stuck"] = mom["stuck_count"]

        if weights.get("domain_terms", 0.8) > 0:
            terms = self._domain_terms.serialize()
            if terms["terms"]:
                fp["terms"] = list(terms["terms"].keys())[:6]

        if weights.get("dep_clusters", 0.6) > 0:
            clusters = self._dep_clusters.serialize()
            if clusters["clusters"]:
                # Only include the largest cluster names
                fp["clusters"] = [
                    c[:3] for c in clusters["clusters"][:2]
                ]

        if weights.get("user_preferences", 0.9) > 0:
            prefs = self._preferences.serialize()
            if prefs["total_decisions"] > 0:
                fp["prefs"] = prefs["ratios"]

        if weights.get("context_anchors", 1.0) > 0:
            anchors = self._anchors.serialize()
            if anchors["count"] > 0:
                fp["anchors"] = anchors["anchors"]

        return fp

    def serialize_compact(self) -> str:
        """Serialize to a compact string for context injection.

        Returns YAML or JSON depending on config.
        """
        data = self.serialize()
        fmt = self._config.serialization_format

        if fmt == "yaml" and _yaml is not None:
            return _yaml.dump(data, default_flow_style=True, width=200).strip()
        else:
            return _json.dumps(data, separators=(",", ":"))

    # -----------------------------------------------------------------
    # Full dimension state (for API / debugging)
    # -----------------------------------------------------------------

    def get_full_state(self) -> dict[str, Any]:
        """Get the full state of all dimensions (not token-optimized)."""
        return {
            "d1_task_type": self._task_info,
            "d2_stack": self._stack,
            "d3_hot_files": self._hot_files.serialize(self._config.max_hot_files),
            "d4_recent_bugs": self._bugs.serialize(),
            "d5_test_health": self._test_health.serialize(),
            "d6_momentum": self._momentum.serialize(),
            "d7_domain_terms": self._domain_terms.serialize(),
            "d8_dep_clusters": self._dep_clusters.serialize(),
            "d9_user_preferences": self._preferences.serialize(),
            "d10_context_anchors": self._anchors.serialize(),
            "step_count": self._step_count,
            "config": {
                "enabled": self._config.enabled,
                "tfidf_refresh_interval": self._config.tfidf_refresh_interval,
                "max_anchors": self._config.max_anchors,
                "serialization_format": self._config.serialization_format,
            },
        }

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def config(self) -> FingerprintConfig:
        """Current configuration."""
        return self._config

    @property
    def step_count(self) -> int:
        """Number of steps processed."""
        return self._step_count

    @property
    def task_type(self) -> str:
        """Current task type classification."""
        return self._task_info.get("type", TaskType.UNKNOWN)

    @property
    def preferences_store(self) -> UserPreferencesStore:
        """Access to the persistent preferences store."""
        return self._preferences


# ---------------------------------------------------------------------------
# Module-level singleton + availability flag
# ---------------------------------------------------------------------------

fingerprint_config: FingerprintConfig = _load_config()

FINGERPRINT_AVAILABLE = True

try:
    fingerprint_manager = FingerprintManager(config=fingerprint_config)
except Exception as _exc:
    logger.warning("FingerprintManager init failed: %s", _exc)
    fingerprint_manager = None
    FINGERPRINT_AVAILABLE = False
