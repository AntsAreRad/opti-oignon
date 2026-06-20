#!/usr/bin/env python3
"""
Benchmark script for S159 streaming and connection improvements.

Measures:
- SSE/WebSocket backpressure buffer throughput and latency
- SQLite connection pool checkout/checkin under concurrency
- Chunked JSON response generator performance
- Before/after comparison mode

Usage::

    python scripts/bench_streaming.py
    python scripts/bench_streaming.py --json
    python scripts/bench_streaming.py --compare before.json after.json

Output is human-readable by default; use ``--json`` for CI integration.
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import tempfile
import threading
import time

# Hardcoded, never overridable
checkpoint_before_apply = True

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_module(name: str, filepath: str):
    """Load a single module by file path, bypassing __init__.py chains."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filepath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_modules():
    """Pre-load the S159 modules we need without triggering the full
    opti_oignon import chain (which requires ollama)."""
    import types
    pkg = os.path.join(_PROJECT_ROOT, "opti_oignon")

    # Create a minimal package stub so sub-module imports resolve
    if "opti_oignon" not in sys.modules:
        stub = types.ModuleType("opti_oignon")
        stub.__path__ = [pkg]
        sys.modules["opti_oignon"] = stub

    # db_utils (needed by connection_pool; has try/except for db_encryption)
    if "opti_oignon.db_utils" not in sys.modules:
        _load_module("opti_oignon.db_utils", os.path.join(pkg, "db_utils.py"))

    if "opti_oignon.sse_backpressure" not in sys.modules:
        _load_module(
            "opti_oignon.sse_backpressure",
            os.path.join(pkg, "sse_backpressure.py"),
        )
    if "opti_oignon.connection_pool" not in sys.modules:
        _load_module(
            "opti_oignon.connection_pool",
            os.path.join(pkg, "connection_pool.py"),
        )
    if "opti_oignon.chunked_response" not in sys.modules:
        _load_module(
            "opti_oignon.chunked_response",
            os.path.join(pkg, "chunked_response.py"),
        )


# ---------------------------------------------------------------------------
# Benchmark: Backpressure buffer
# ---------------------------------------------------------------------------

def bench_backpressure(iterations: int = 5000, buffer_size: int = 100) -> dict:
    """Measure push/pop throughput and drop rate of BackpressureBuffer."""
    from opti_oignon.sse_backpressure import BackpressureBuffer

    buf = BackpressureBuffer(max_size=buffer_size, slow_threshold=0.9, idle_timeout=30.0)

    # -- Push throughput (sync) --
    push_times = []
    for i in range(iterations):
        t0 = time.perf_counter()
        buf.push({"type": "token", "content": f"tok_{i}", "seq": i})
        push_times.append((time.perf_counter() - t0) * 1e6)  # microseconds

    push_stats = {
        "count": iterations,
        "mean_us": round(statistics.mean(push_times), 2),
        "median_us": round(statistics.median(push_times), 2),
        "p99_us": round(sorted(push_times)[int(len(push_times) * 0.99)], 2),
        "dropped": buf.stats.dropped,
    }

    # -- Pop throughput (async) --
    buf2 = BackpressureBuffer(max_size=buffer_size, slow_threshold=0.9, idle_timeout=30.0)
    # Pre-fill
    for i in range(min(iterations, buffer_size)):
        buf2.push({"type": "token", "content": f"pre_{i}"})

    async def _pop_all():
        pop_times = []
        popped = 0
        while popped < min(iterations, buffer_size):
            t0 = time.perf_counter()
            ev = await buf2.pop(timeout=1.0)
            elapsed = (time.perf_counter() - t0) * 1e6
            if ev is None:
                break
            pop_times.append(elapsed)
            popped += 1
        return pop_times

    pop_times = asyncio.run(_pop_all())
    pop_stats = {
        "count": len(pop_times),
        "mean_us": round(statistics.mean(pop_times), 2) if pop_times else 0,
        "median_us": round(statistics.median(pop_times), 2) if pop_times else 0,
    }

    # -- Producer-consumer simulation --
    buf3 = BackpressureBuffer(max_size=buffer_size, slow_threshold=0.8, idle_timeout=30.0)
    produced = 0
    consumed = 0
    sim_duration = 1.0  # seconds
    sim_start = time.monotonic()

    def _producer():
        nonlocal produced
        while time.monotonic() - sim_start < sim_duration:
            buf3.push({"seq": produced})
            produced += 1
            # Simulate fast producer
            time.sleep(0.0001)

    async def _consumer():
        nonlocal consumed
        while time.monotonic() - sim_start < sim_duration + 0.5:
            ev = await buf3.pop(timeout=0.1)
            if ev is not None:
                consumed += 1
            if buf3.closed:
                break

    t = threading.Thread(target=_producer, daemon=True)
    t.start()
    asyncio.run(_consumer())
    buf3.close()
    t.join(timeout=2.0)

    sim_stats = {
        "duration_s": sim_duration,
        "produced": produced,
        "consumed": consumed,
        "dropped": buf3.stats.dropped,
        "producer_rate_hz": round(produced / sim_duration),
        "consumer_rate_hz": round(consumed / sim_duration),
    }

    return {
        "benchmark": "backpressure_buffer",
        "push": push_stats,
        "pop": pop_stats,
        "simulation": sim_stats,
    }


# ---------------------------------------------------------------------------
# Benchmark: Connection pool
# ---------------------------------------------------------------------------

def bench_connection_pool(
    pool_size: int = 5,
    operations: int = 200,
    concurrency: int = 10,
) -> dict:
    """Measure connection pool throughput under concurrent access."""
    from opti_oignon.connection_pool import ConnectionPool

    # Create a temp database
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name

    try:
        pool = ConnectionPool(
            db_path=db_path,
            pool_size=pool_size,
            health_check=False,  # Skip for pure throughput test
            wal_mode=True,
        )

        # Initialize table
        with pool.connection() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS bench "
                "(id INTEGER PRIMARY KEY, val TEXT)"
            )
            conn.commit()

        # -- Sequential checkout/checkin --
        seq_times = []
        for i in range(operations):
            t0 = time.perf_counter()
            with pool.connection() as conn:
                conn.execute("SELECT 1")
            seq_times.append((time.perf_counter() - t0) * 1e6)

        seq_stats = {
            "count": operations,
            "mean_us": round(statistics.mean(seq_times), 2),
            "median_us": round(statistics.median(seq_times), 2),
            "p99_us": round(sorted(seq_times)[int(len(seq_times) * 0.99)], 2),
        }

        # -- Concurrent access --
        barrier = threading.Barrier(concurrency)
        concurrent_times = []
        errors = []
        lock = threading.Lock()

        def _worker(ops_per_thread):
            try:
                barrier.wait(timeout=5.0)
            except threading.BrokenBarrierError:
                return
            for _ in range(ops_per_thread):
                t0 = time.perf_counter()
                try:
                    with pool.connection() as conn:
                        conn.execute(
                            "INSERT INTO bench (val) VALUES (?)",
                            (f"t_{threading.current_thread().name}",),
                        )
                        conn.commit()
                    elapsed = (time.perf_counter() - t0) * 1e6
                    with lock:
                        concurrent_times.append(elapsed)
                except Exception as exc:
                    with lock:
                        errors.append(str(exc))

        ops_per = operations // concurrency
        threads = []
        for i in range(concurrency):
            t = threading.Thread(target=_worker, args=(ops_per,), daemon=True)
            threads.append(t)

        t_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
        wall_time_ms = (time.perf_counter() - t_start) * 1000

        conc_stats = {
            "threads": concurrency,
            "ops_per_thread": ops_per,
            "total_ops": len(concurrent_times),
            "errors": len(errors),
            "wall_time_ms": round(wall_time_ms, 1),
            "throughput_ops_s": round(len(concurrent_times) / (wall_time_ms / 1000)) if wall_time_ms > 0 else 0,
            "mean_us": round(statistics.mean(concurrent_times), 2) if concurrent_times else 0,
            "p99_us": round(sorted(concurrent_times)[int(len(concurrent_times) * 0.99)], 2) if concurrent_times else 0,
        }

        pool_status = pool.get_status()
        pool.close()

        return {
            "benchmark": "connection_pool",
            "pool_size": pool_size,
            "sequential": seq_stats,
            "concurrent": conc_stats,
            "pool_stats": pool_status.get("stats", {}),
        }

    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Benchmark: Chunked response generator
# ---------------------------------------------------------------------------

def bench_chunked_response(
    payload_sizes: list[int] | None = None,
    chunk_size: int = 4096,
) -> dict:
    """Measure chunked JSON generator throughput at various payload sizes."""
    from opti_oignon.chunked_response import chunked_json_generator

    if payload_sizes is None:
        payload_sizes = [1_000, 10_000, 100_000, 500_000]

    results = []
    for size in payload_sizes:
        # Build a payload of approximately `size` bytes
        item = {"id": 0, "content": "x" * 80, "score": 0.95, "source": "bench.txt"}
        items_needed = max(1, size // 100)
        payload = {"results": [dict(item, id=i) for i in range(items_needed)]}

        # Measure generation
        t0 = time.perf_counter()
        chunks = list(chunked_json_generator(payload, chunk_size=chunk_size))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_bytes = sum(len(c) for c in chunks)

        results.append({
            "target_size": size,
            "actual_bytes": total_bytes,
            "chunks": len(chunks),
            "chunk_size": chunk_size,
            "elapsed_ms": round(elapsed_ms, 2),
            "throughput_mb_s": round(total_bytes / (elapsed_ms / 1000) / 1e6, 1) if elapsed_ms > 0 else 0,
        })

    return {
        "benchmark": "chunked_response",
        "results": results,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_results(before_path: str, after_path: str) -> dict:
    """Compare two benchmark JSON files and compute deltas."""
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    comparisons = []
    for b_bench in before.get("benchmarks", []):
        name = b_bench.get("benchmark", "")
        a_bench = None
        for ab in after.get("benchmarks", []):
            if ab.get("benchmark") == name:
                a_bench = ab
                break
        if a_bench is None:
            continue

        comp = {"benchmark": name, "metrics": []}

        # Extract comparable numeric values recursively
        def _extract_metrics(d, prefix=""):
            metrics = {}
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (int, float)) and k != "benchmark":
                    metrics[key] = v
                elif isinstance(v, dict):
                    metrics.update(_extract_metrics(v, key))
            return metrics

        b_metrics = _extract_metrics(b_bench)
        a_metrics = _extract_metrics(a_bench)

        for key in sorted(set(b_metrics) | set(a_metrics)):
            bv = b_metrics.get(key)
            av = a_metrics.get(key)
            if bv is not None and av is not None and bv != 0:
                delta_pct = round((av - bv) / abs(bv) * 100, 1)
                comp["metrics"].append({
                    "metric": key,
                    "before": bv,
                    "after": av,
                    "delta_pct": delta_pct,
                })

        comparisons.append(comp)

    return {"comparison": comparisons}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_benchmarks(verbose: bool = True) -> dict:
    """Run all benchmarks and return combined results."""
    _ensure_modules()

    # Suppress noisy per-event warnings during benchmarks
    logging.getLogger("opti_oignon.sse_backpressure").setLevel(logging.ERROR)
    logging.getLogger("opti_oignon.connection_pool").setLevel(logging.ERROR)
    logging.getLogger("opti_oignon.chunked_response").setLevel(logging.ERROR)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": [],
    }

    if verbose:
        print("Running backpressure buffer benchmark...")
    results["benchmarks"].append(bench_backpressure())

    if verbose:
        print("Running connection pool benchmark...")
    results["benchmarks"].append(bench_connection_pool())

    if verbose:
        print("Running chunked response benchmark...")
    results["benchmarks"].append(bench_chunked_response())

    return results


def _format_human(results: dict) -> str:
    """Format benchmark results for human reading."""
    lines = []
    lines.append(f"Benchmark results -- {results.get('timestamp', '')}")
    lines.append("")

    for bench in results.get("benchmarks", []):
        name = bench.get("benchmark", "unknown")
        lines.append(f"[{name}]")

        def _print_dict(d, indent=2):
            for k, v in d.items():
                if k == "benchmark":
                    continue
                if isinstance(v, dict):
                    lines.append(f"{' ' * indent}{k}:")
                    _print_dict(v, indent + 2)
                elif isinstance(v, list):
                    lines.append(f"{' ' * indent}{k}:")
                    for item in v:
                        if isinstance(item, dict):
                            _print_dict(item, indent + 2)
                            lines.append("")
                        else:
                            lines.append(f"{' ' * (indent + 2)}- {item}")
                else:
                    lines.append(f"{' ' * indent}{k}: {v}")

        _print_dict(bench)
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="S159 streaming benchmarks")
    parser.add_argument("--json", action="store_true", help="Output JSON for CI")
    parser.add_argument("--output", "-o", help="Write results to file")
    parser.add_argument(
        "--compare", nargs=2, metavar=("BEFORE", "AFTER"),
        help="Compare two benchmark JSON files",
    )
    args = parser.parse_args()

    if args.compare:
        result = compare_results(args.compare[0], args.compare[1])
        if args.json:
            output = json.dumps(result, indent=2)
        else:
            lines = ["Comparison results", ""]
            for comp in result.get("comparison", []):
                lines.append(f"[{comp['benchmark']}]")
                for m in comp.get("metrics", []):
                    direction = "+" if m["delta_pct"] > 0 else ""
                    lines.append(
                        f"  {m['metric']}: {m['before']} -> {m['after']} "
                        f"({direction}{m['delta_pct']}%)"
                    )
                lines.append("")
            output = "\n".join(lines)
    else:
        results = run_all_benchmarks(verbose=not args.json)
        if args.json:
            output = json.dumps(results, indent=2)
        else:
            output = _format_human(results)

    print(output)

    if args.output:
        data = results if not args.compare else result
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
