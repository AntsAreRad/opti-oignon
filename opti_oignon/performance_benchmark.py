#!/usr/bin/env python3
"""
PERFORMANCE BENCHMARK — Module de benchmarks de performance
============================================================

Measures the latency and throughput of Opti-Oignon subsystems:
- Cache exact (SHA-256 lookup)
- Cache semantique (cosine similarity search)
- Detection d'artefacts
- Fenetre de contexte (trimming)
- Operations conversation (SQLite)
- Memoire cross-conversation
- Token budget calculation
- Warmup status queries

Usage:
    from opti_oignon.performance_benchmark import benchmark_runner, run_all

    # Executer tous les benchmarks
    results = benchmark_runner.run_all()

    # Benchmark specifique
    result = benchmark_runner.run("response_cache")

    # Exporter en JSON
    benchmark_runner.export_json("benchmarks.json")

    # Rapport texte
    print(benchmark_runner.get_report())

Session 25 — H2
"""

import json
import logging
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.

    Attributes:
        name: Nom du benchmark
        iterations: Nombre d'iterations executees
        total_time_ms: Temps total en millisecondes
        mean_ms: Latence moyenne en ms
        median_ms: Latence mediane en ms
        min_ms: Latence minimale en ms
        max_ms: Latence maximale en ms
        stddev_ms: Ecart-type en ms
        p95_ms: 95eme percentile en ms
        p99_ms: 99eme percentile en ms
        throughput_ops: Operations par seconde
        metadata: Data supplementaires
        error: Message d'erreur si echec
    """
    name: str
    iterations: int = 0
    total_time_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    stddev_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    throughput_ops: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results.

    Attributes:
        timestamp: Horodatage ISO du run
        version: Version d'Opti-Oignon
        results: Resultats par benchmark
        total_time_ms: Temps total de la suite
    """
    timestamp: str = ""
    version: str = ""
    results: dict[str, BenchmarkResult] = field(default_factory=dict)
    total_time_ms: float = 0.0


# =============================================================================
# UTILITAIRES
# =============================================================================

def _measure(func: Callable, iterations: int = 100, warmup: int = 5) -> BenchmarkResult:
    """Execute a function N times and compute statistics.

    Args:
        func: Fonction a benchmarker (sans arguments)
        iterations: Nombre d'iterations
        warmup: Nombre d'iterations de chauffe (non comptees)

    Returns:
        BenchmarkResult avec statistiques
    """
    # Phase de chauffe
    for _ in range(warmup):
        try:
            func()
        except Exception:
            pass

    # Phase de mesure
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000  # en ms
        timings.append(elapsed)

    if not timings:
        return BenchmarkResult(name="", error="No timings recorded")

    total = sum(timings)
    sorted_t = sorted(timings)
    n = len(sorted_t)

    # Percentiles
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)

    return BenchmarkResult(
        name="",
        iterations=n,
        total_time_ms=total,
        mean_ms=statistics.mean(timings),
        median_ms=statistics.median(timings),
        min_ms=sorted_t[0],
        max_ms=sorted_t[-1],
        stddev_ms=statistics.stdev(timings) if n > 1 else 0.0,
        p95_ms=sorted_t[p95_idx],
        p99_ms=sorted_t[p99_idx],
        throughput_ops=(n / (total / 1000)) if total > 0 else 0.0,
    )


def _percentile(sorted_values: list, pct: float) -> float:
    """Compute a given percentile.

    Args:
        sorted_values: Liste triee de valeurs
        pct: Percentile (0.0-1.0)

    Returns:
        Valeur au percentile demande
    """
    if not sorted_values:
        return 0.0
    idx = min(int(len(sorted_values) * pct), len(sorted_values) - 1)
    return sorted_values[idx]


# =============================================================================
# BENCHMARKS INDIVIDUELS
# =============================================================================

def bench_response_cache(iterations: int = 200) -> BenchmarkResult:
    """Benchmark: Response cache (exact SHA-256 lookup).

    Measure the latency of put/get/miss on the response cache.
    """
    try:
        from .response_cache import ResponseCache
    except ImportError:
        return BenchmarkResult(name="response_cache", error="Module not available")

    tmpdir = tempfile.mkdtemp(prefix="bench_cache_")
    db_path = Path(tmpdir) / "bench_cache.db"

    try:
        cache = ResponseCache(db_path=db_path)

        # Pre-remplir avec des data
        for i in range(50):
            cache.put(
                model=f"model_{i % 3}",
                system_prompt="You are a helpful assistant.",
                user_content=f"Question {i}?",
                response=f"Response content for benchmark item {i}" * 10,
            )

        # Benchmark: cache hits (lecture de cles existantes)
        # Generate the key as the cache would
        hit_key = cache.make_cache_key("model_0", "You are a helpful assistant.", "Question 0?")
        hit_result = _measure(
            lambda: cache.get(hit_key),
            iterations=iterations,
        )
        hit_result.name = "response_cache_hit"
        hit_result.metadata["type"] = "exact_hit"

        # Benchmark: cache misses
        miss_counter = [0]
        def _miss_op():
            miss_counter[0] += 1
            cache.get(f"nonexistent_key_{miss_counter[0]}")

        miss_result = _measure(_miss_op, iterations=iterations)
        miss_result.name = "response_cache_miss"
        miss_result.metadata["type"] = "exact_miss"

        # Benchmark: cache put
        put_counter = [0]
        def _put_op():
            put_counter[0] += 1
            cache.put(
                model="bench_model",
                system_prompt="Benchmark system prompt.",
                user_content=f"Put question {put_counter[0]}?",
                response=f"New response {put_counter[0]}",
            )

        put_result = _measure(_put_op, iterations=iterations)
        put_result.name = "response_cache_put"
        put_result.metadata["type"] = "put"

        # Combine results (return hit as primary)
        hit_result.metadata["miss_mean_ms"] = miss_result.mean_ms
        hit_result.metadata["put_mean_ms"] = put_result.mean_ms
        hit_result.name = "response_cache"
        return hit_result

    finally:
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def bench_semantic_cache(iterations: int = 100) -> BenchmarkResult:
    """Benchmark: Semantic cache (cosine similarity search).

    Measure cosine similarity search latency.
    Utilise des embeddings simules (pas d'appel Ollama).
    """
    try:
        from .semantic_cache import SemanticCache, cosine_similarity
    except ImportError:
        return BenchmarkResult(name="semantic_cache", error="Module not available")

    import random

    tmpdir = tempfile.mkdtemp(prefix="bench_sem_")
    db_path = Path(tmpdir) / "bench_sem.db"

    try:
        cache = SemanticCache(db_path=db_path)

        # Generer des embeddings simules (dimension 384, comme mxbai-embed-large)
        dim = 384
        random.seed(42)

        entries = []
        for i in range(50):
            emb = [random.gauss(0, 1) for _ in range(dim)]
            # Normaliser
            norm = sum(x * x for x in emb) ** 0.5
            emb = [x / norm for x in emb]
            entries.append((f"bench_query_{i}", emb, f"bench_response_{i}", f"model_{i % 3}"))

        # Stocker les embeddings
        for query, emb, response, model in entries:
            try:
                cache.store_embedding(
                    cache_key=f"key_{query}",
                    model=model,
                    query_text=query,
                    embedding=emb,
                )
            except Exception:
                pass

        # Benchmark: cosine similarity calcul (brut)
        emb_a = entries[0][1]
        emb_b = entries[25][1]
        cosine_result = _measure(
            lambda: cosine_similarity(emb_a, emb_b),
            iterations=iterations,
        )
        cosine_result.name = "cosine_similarity"
        cosine_result.metadata["dimension"] = dim

        # Benchmark: find_similar_by_embedding (search dans le cache)
        def _search():
            cache.find_similar_by_embedding(
                query_embedding=emb_a,
                model="model_0",
                threshold=0.85,
            )

        search_result = _measure(_search, iterations=iterations)
        search_result.name = "semantic_cache"
        search_result.metadata["cosine_mean_ms"] = cosine_result.mean_ms
        search_result.metadata["entries_count"] = len(entries)
        search_result.metadata["dimension"] = dim
        return search_result

    finally:
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def bench_artifact_detection(iterations: int = 200) -> BenchmarkResult:
    """Benchmark: Artifact detection speed.

    Measure artifact detection latency in an LLM response.
    """
    try:
        from .artifacts import ArtifactDetector
    except ImportError:
        return BenchmarkResult(name="artifact_detection", error="Module not available")

    detector = ArtifactDetector()

    # Reponse LLM typique avec du code
    sample_response = '''Here's a Python script to analyze biodiversity data:

```python
import pandas as pd
import numpy as np
from scipy import stats

def calculate_shannon_index(counts):
    """Calculate Shannon diversity index."""
    total = sum(counts)
    proportions = [c / total for c in counts if c > 0]
    return -sum(p * np.log(p) for p in proportions)

# Load species data
df = pd.read_csv("species_counts.csv")
diversity = calculate_shannon_index(df["count"].values)
print(f"Shannon Index: {diversity:.4f}")
```

And here's an R version:

```r
library(vegan)

species_data <- read.csv("species_counts.csv")
shannon <- diversity(species_data$count, index = "shannon")
cat(sprintf("Shannon Index: %.4f\\n", shannon))
```

The Shannon index measures species diversity in a community.
'''

    result = _measure(
        lambda: detector.detect(sample_response),
        iterations=iterations,
    )
    result.name = "artifact_detection"
    result.metadata["response_length"] = len(sample_response)

    # Verifier la detection
    artifacts = detector.detect(sample_response)
    result.metadata["artifacts_found"] = len(artifacts)
    return result


def bench_context_window(iterations: int = 200) -> BenchmarkResult:
    """Benchmark: Context window trimming speed.

    Measure history trimming latency with importance scoring.
    """
    try:
        from .context_window import SlidingWindowManager, TokenBudgetManager
    except ImportError:
        return BenchmarkResult(name="context_window", error="Module not available")

    swm = SlidingWindowManager()
    tbm = TokenBudgetManager()

    # Historique de conversation long (40 messages)
    history = []
    for i in range(40):
        history.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message {i}: " + "Some content about bioinformatics analysis. " * 20,
        })

    # Budget pour qwen3:32b
    budget = tbm.get_budget("qwen3:32b")

    result = _measure(
        lambda: swm.prepare_messages(list(history), "qwen3:32b"),
        iterations=iterations,
    )
    result.name = "context_window"
    result.metadata["history_length"] = len(history)
    result.metadata["budget_tokens"] = budget.context_window

    # Mesurer also le calcul de budget
    budget_result = _measure(
        lambda: tbm.get_budget("qwen3:32b"),
        iterations=iterations,
    )
    result.metadata["budget_calc_mean_ms"] = budget_result.mean_ms
    return result


def bench_conversation_db(iterations: int = 200) -> BenchmarkResult:
    """Benchmark: Conversation database operations.

    Measure CRUD operation latency on the SQLite conversations database.
    """
    try:
        from .conversation import ConversationManager
    except ImportError:
        return BenchmarkResult(name="conversation_db", error="Module not available")

    # Utiliser un repertoire temporaire
    tmpdir = tempfile.mkdtemp(prefix="bench_conv_")
    db_path = Path(tmpdir) / "bench_conv.db"

    try:
        cm = ConversationManager(db_path=db_path)

        # Creer des conversations de test
        conv_ids = []
        for i in range(20):
            conv = cm.create_conversation(title=f"Bench conversation {i}")
            conv_ids.append(conv.id)
            # Ajouter des messages
            for j in range(5):
                cm.add_message(conv.id, "user", f"User message {j} in conv {i}")
                cm.add_message(conv.id, "assistant", f"Assistant reply {j} in conv {i}")

        # Benchmark: list conversations
        list_result = _measure(
            lambda: cm.list_conversations(),
            iterations=iterations,
        )

        # Benchmark: get messages
        target_id = conv_ids[0]
        msg_result = _measure(
            lambda: cm.get_messages(target_id),
            iterations=iterations,
        )

        # Benchmark: search
        search_result = _measure(
            lambda: cm.search_conversations("User message"),
            iterations=iterations,
        )

        # Combine results
        list_result.name = "conversation_db"
        list_result.metadata["list_mean_ms"] = list_result.mean_ms
        list_result.metadata["get_messages_mean_ms"] = msg_result.mean_ms
        list_result.metadata["search_mean_ms"] = search_result.mean_ms
        list_result.metadata["conv_count"] = len(conv_ids)
        list_result.metadata["messages_per_conv"] = 10
        return list_result

    finally:
        # Nettoyage
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def bench_memory(iterations: int = 200) -> BenchmarkResult:
    """Benchmark: Cross-conversation memory retrieval.

    Measure latency of adding, searching, and injecting memory facts.
    """
    try:
        from .memory import MemoryManager
    except ImportError:
        return BenchmarkResult(name="memory", error="Module not available")

    tmpdir = tempfile.mkdtemp(prefix="bench_mem_")
    db_path = Path(tmpdir) / "bench_mem.db"

    try:
        mm = MemoryManager(db_path=db_path)

        # Pre-remplir avec des faits
        facts = [
            ("User works in bioinformatics", "preference"),
            ("User prefers R for statistical analysis", "preference"),
            ("User is doing PhD at Rennes", "preference"),
            ("User uses Kubuntu with Ollama", "tool"),
            ("User's project is called Opti-Oignon", "project"),
            ("User likes qwen3-coder:30b for coding tasks", "preference"),
            ("User researches biodiversity with metabarcoding", "project"),
            ("User lives in Panama for fieldwork", "preference"),
        ]
        for text, cat in facts:
            mm.add_fact(text, category=cat)

        # Benchmark: get all facts
        get_result = _measure(
            lambda: mm.get_all_facts(),
            iterations=iterations,
        )

        # Benchmark: filtered by category
        filter_result = _measure(
            lambda: mm.get_all_facts(category="preference"),
            iterations=iterations,
        )

        # Benchmark: build injection prompt
        inject_result = _measure(
            lambda: mm.format_for_prompt(),
            iterations=iterations,
        )

        get_result.name = "memory"
        get_result.metadata["get_all_mean_ms"] = get_result.mean_ms
        get_result.metadata["filter_mean_ms"] = filter_result.mean_ms
        get_result.metadata["inject_mean_ms"] = inject_result.mean_ms
        get_result.metadata["fact_count"] = len(facts)
        return get_result

    finally:
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def bench_token_budget(iterations: int = 500) -> BenchmarkResult:
    """Benchmark: Token budget calculation speed.

    Measure latency of token budget computation per model.
    """
    try:
        from .context_window import TokenBudgetManager
    except ImportError:
        return BenchmarkResult(name="token_budget", error="Module not available")

    tbm = TokenBudgetManager()

    models = [
        "qwen3:32b", "qwen3-coder:30b", "deepseek-r1:32b",
        "nemotron-3-nano:30b", "llama3.1:8b", "unknown_model",
    ]

    model_idx = [0]
    def _budget_op():
        model_idx[0] = (model_idx[0] + 1) % len(models)
        tbm.get_budget(models[model_idx[0]])

    result = _measure(_budget_op, iterations=iterations)
    result.name = "token_budget"
    result.metadata["models_tested"] = len(models)
    result.metadata["profiles_available"] = len(tbm.known_models) if hasattr(tbm, 'known_models') else 0
    return result


def bench_model_warmup_status(iterations: int = 200) -> BenchmarkResult:
    """Benchmark: Model warmup status queries.

    Measure latency of VRAM status queries (without Ollama).
    """
    try:
        from .model_warmup import ModelWarmup
    except ImportError:
        return BenchmarkResult(name="model_warmup_status", error="Module not available")

    mw = ModelWarmup()

    # Benchmark: get_stats (pas d'appel Ollama)
    stats_result = _measure(
        lambda: mw.get_stats(),
        iterations=iterations,
    )

    # Benchmark: get_warmup_report
    report_result = _measure(
        lambda: mw.get_warmup_report(),
        iterations=iterations,
    )

    # Benchmark: get_vram_summary
    vram_result = _measure(
        lambda: mw.get_vram_summary(),
        iterations=iterations,
    )

    stats_result.name = "model_warmup_status"
    stats_result.metadata["report_mean_ms"] = report_result.mean_ms
    stats_result.metadata["vram_summary_mean_ms"] = vram_result.mean_ms
    return stats_result


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

# Liste des benchmarks disponibles
AVAILABLE_BENCHMARKS: dict[str, Callable] = {
    "response_cache": bench_response_cache,
    "semantic_cache": bench_semantic_cache,
    "artifact_detection": bench_artifact_detection,
    "context_window": bench_context_window,
    "conversation_db": bench_conversation_db,
    "memory": bench_memory,
    "token_budget": bench_token_budget,
    "model_warmup_status": bench_model_warmup_status,
}


class BenchmarkRunner:
    """Orchestrateur de benchmarks de performance.

    Allows executing, combining, and exporting results
    de tous les benchmarks disponibles.
    """

    def __init__(self):
        """Initialize benchmark runner."""
        self._results: dict[str, BenchmarkResult] = {}
        self._last_suite: BenchmarkSuite | None = None

    def run(self, name: str, iterations: int = 200) -> BenchmarkResult:
        """Run a single benchmark by name.

        Args:
            name: Nom du benchmark (cle dans AVAILABLE_BENCHMARKS)
            iterations: Nombre d'iterations

        Returns:
            BenchmarkResult avec statistiques
        """
        if name not in AVAILABLE_BENCHMARKS:
            return BenchmarkResult(
                name=name,
                error=f"Unknown benchmark: {name}. Available: {list(AVAILABLE_BENCHMARKS.keys())}",
            )

        try:
            result = AVAILABLE_BENCHMARKS[name](iterations=iterations)
            self._results[name] = result
            return result
        except Exception as e:
            logger.error(f"Benchmark {name} failed: {e}")
            result = BenchmarkResult(name=name, error=str(e))
            self._results[name] = result
            return result

    def run_all(self, iterations: int = 200) -> BenchmarkSuite:
        """Run all available benchmarks.

        Args:
            iterations: Nombre d'iterations par benchmark

        Returns:
            BenchmarkSuite with all results
        """
        from datetime import datetime

        suite = BenchmarkSuite(
            timestamp=datetime.now().isoformat(),
            version=self._get_version(),
        )

        suite_start = time.perf_counter()

        for name in AVAILABLE_BENCHMARKS:
            logger.info(f"Running benchmark: {name}")
            result = self.run(name, iterations=iterations)
            suite.results[name] = result

        suite.total_time_ms = (time.perf_counter() - suite_start) * 1000
        self._last_suite = suite
        return suite

    def get_results(self) -> dict[str, BenchmarkResult]:
        """Get all collected results.

        Returns:
            Dict mapping benchmark name to BenchmarkResult
        """
        return dict(self._results)

    def get_report(self) -> str:
        """Generate a text report of all results.

        Returns:
            Formatted text report
        """
        if not self._results:
            return "No benchmarks have been run yet."

        lines = [
            "=" * 70,
            "  OPTI-OIGNON PERFORMANCE BENCHMARKS",
            "=" * 70,
            "",
        ]

        for name, result in sorted(self._results.items()):
            if result.error:
                lines.append(f"  {name}: ERROR - {result.error}")
                lines.append("")
                continue

            lines.append(f"  {name}")
            lines.append(f"  {'-' * 50}")
            lines.append(f"    Iterations: {result.iterations}")
            lines.append(f"    Mean:       {result.mean_ms:.3f} ms")
            lines.append(f"    Median:     {result.median_ms:.3f} ms")
            lines.append(f"    Min/Max:    {result.min_ms:.3f} / {result.max_ms:.3f} ms")
            lines.append(f"    Stddev:     {result.stddev_ms:.3f} ms")
            lines.append(f"    P95:        {result.p95_ms:.3f} ms")
            lines.append(f"    P99:        {result.p99_ms:.3f} ms")
            lines.append(f"    Throughput: {result.throughput_ops:.0f} ops/s")

            if result.metadata:
                lines.append("    Metadata:")
                for k, v in result.metadata.items():
                    if isinstance(v, float):
                        lines.append(f"      {k}: {v:.3f}")
                    else:
                        lines.append(f"      {k}: {v}")
            lines.append("")

        if self._last_suite:
            lines.append(f"  Total suite time: {self._last_suite.total_time_ms:.1f} ms")
            lines.append(f"  Version: {self._last_suite.version}")
            lines.append(f"  Timestamp: {self._last_suite.timestamp}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def export_json(self, filepath: str) -> str:
        """Export results to JSON file.

        Args:
            filepath: File path de sortie

        Returns:
            Path of the created file
        """
        data = {}
        if self._last_suite:
            data["timestamp"] = self._last_suite.timestamp
            data["version"] = self._last_suite.version
            data["total_time_ms"] = self._last_suite.total_time_ms

        data["benchmarks"] = {}
        for name, result in self._results.items():
            data["benchmarks"][name] = asdict(result)

        path = Path(filepath)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return str(path)

    def export_dict(self) -> dict[str, Any]:
        """Export results as a dictionary.

        Returns:
            Dict with all benchmark data
        """
        data = {}
        if self._last_suite:
            data["timestamp"] = self._last_suite.timestamp
            data["version"] = self._last_suite.version
            data["total_time_ms"] = self._last_suite.total_time_ms

        data["benchmarks"] = {}
        for name, result in self._results.items():
            data["benchmarks"][name] = asdict(result)
        return data

    def get_summary_md(self) -> str:
        """Generate a Markdown summary for the health dashboard.

        Returns:
            Markdown-formatted summary string
        """
        if not self._results:
            return "*No benchmarks run yet*"

        lines = ["**Performance Benchmarks**\n"]
        for name, result in sorted(self._results.items()):
            if result.error:
                status = "ERR"
                detail = result.error[:40]
            else:
                status = "OK"
                detail = f"{result.mean_ms:.2f}ms avg, {result.p95_ms:.2f}ms p95"
            lines.append(f"- `{name}` [{status}]: {detail}")

        return "\n".join(lines)

    @staticmethod
    def _get_version() -> str:
        """Get current Opti-Oignon version."""
        try:
            from . import __version__
            return __version__
        except ImportError:
            return "unknown"

    @staticmethod
    def list_benchmarks() -> list[str]:
        """List available benchmark names.

        Returns:
            List of benchmark name strings
        """
        return list(AVAILABLE_BENCHMARKS.keys())


# =============================================================================
# SINGLETON ET FONCTIONS DE COMMODITE
# =============================================================================

benchmark_runner = BenchmarkRunner()


def run_all(iterations: int = 200) -> BenchmarkSuite:
    """Convenience function to run all benchmarks."""
    return benchmark_runner.run_all(iterations=iterations)


# Indicateur de disponibilite
BENCHMARK_AVAILABLE = True


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Opti-Oignon Performance Benchmarks")
    parser.add_argument("--iterations", "-n", type=int, default=200, help="Iterations per benchmark")
    parser.add_argument("--benchmark", "-b", type=str, default=None, help="Run specific benchmark")
    parser.add_argument("--json", "-j", type=str, default=None, help="Export results to JSON")
    parser.add_argument("--list", "-l", action="store_true", help="List available benchmarks")
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name in AVAILABLE_BENCHMARKS:
            print(f"  - {name}")
        exit(0)

    if args.benchmark:
        result = benchmark_runner.run(args.benchmark, iterations=args.iterations)
        print(benchmark_runner.get_report())
    else:
        benchmark_runner.run_all(iterations=args.iterations)
        print(benchmark_runner.get_report())

    if args.json:
        benchmark_runner.export_json(args.json)
        print(f"\nResults exported to: {args.json}")
