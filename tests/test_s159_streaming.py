"""
tests/test_s159_streaming.py -- S159 streaming and connection improvements tests.

Verifies:
- Goal 1: SSE backpressure buffer (push, pop, drop oldest, slow detection, idle timeout)
- Goal 2: SQLite connection pool (checkout/checkin, health check, exhaustion, WAL)
- Goal 3: Chunked transfer for RAG responses (generator, UTF-8 safety, progress)
- Goal 4: Benchmark script (AST validity, importlib isolation)
- Goal 5: Module structure (checkpoint_before_apply sentinels, integration points)
"""

import ast
import asyncio
import importlib.util
import os
import queue
import sqlite3
import sys
import tempfile
import threading
import time
import types
from unittest.mock import MagicMock, patch

# -- Isolation stubs (standard pattern) --
for mod_name in [
    "opti_oignon",
    "opti_oignon.db_utils",
    "opti_oignon.db_encryption",
    "opti_oignon.config",
    "opti_oignon.auth",
    "opti_oignon.middleware",
    "opti_oignon.security_mode",
]:
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        if mod_name == "opti_oignon":
            stub.__path__ = [
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "opti_oignon",
                )
            ]
        sys.modules[mod_name] = stub

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKPRESSURE_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "sse_backpressure.py"
)
POOL_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "connection_pool.py"
)
CHUNKED_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "chunked_response.py"
)
BENCH_PATH = os.path.join(PROJECT_ROOT, "scripts", "bench_streaming.py")
ROUTES_CHAT_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "api", "routes_chat.py"
)
ROUTES_RAG_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "api", "routes_rag.py"
)


# -- Helpers --


def _load_module(name, path):
    """Load a module by file path without triggering the full import chain."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, f"Cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ast_parse(path):
    """Parse a Python file and return the AST."""
    with open(path) as f:
        return ast.parse(f.read(), filename=path)


# -- Load modules --

bp_mod = _load_module("opti_oignon.sse_backpressure", BACKPRESSURE_PATH)
BackpressureBuffer = bp_mod.BackpressureBuffer
BufferStats = bp_mod.BufferStats

pool_mod = _load_module("opti_oignon.connection_pool", POOL_PATH)
ConnectionPool = pool_mod.ConnectionPool
PooledConnection = pool_mod.PooledConnection
get_pool = pool_mod.get_pool
close_all_pools = pool_mod.close_all_pools
list_pools = pool_mod.list_pools

chunked_mod = _load_module("opti_oignon.chunked_response", CHUNKED_PATH)
chunked_json_generator = chunked_mod.chunked_json_generator
chunked_text_generator = chunked_mod.chunked_text_generator


# =========================================================================
# CLASS 1: BackpressureBuffer -- basic push/pop
# =========================================================================


class TestBackpressureBasic:
    def test_push_and_pop(self):
        buf = BackpressureBuffer(max_size=10)
        buf.push({"type": "token", "content": "hello"})
        assert buf.size == 1
        result = asyncio.run(buf.pop(timeout=1.0))
        assert result is not None
        assert result["content"] == "hello"
        assert buf.size == 0

    def test_push_returns_true(self):
        buf = BackpressureBuffer(max_size=5)
        assert buf.push({"seq": 1}) is True

    def test_push_returns_false_when_closed(self):
        buf = BackpressureBuffer(max_size=5)
        buf.close()
        assert buf.push({"seq": 1}) is False

    def test_pop_returns_none_when_closed_and_empty(self):
        buf = BackpressureBuffer(max_size=5)
        buf.close()
        result = asyncio.run(buf.pop(timeout=0.1))
        assert result is None

    def test_pop_returns_none_on_timeout(self):
        buf = BackpressureBuffer(max_size=5, idle_timeout=60.0)
        result = asyncio.run(buf.pop(timeout=0.05))
        assert result is None

    def test_fifo_order(self):
        buf = BackpressureBuffer(max_size=10)
        for i in range(5):
            buf.push({"seq": i})
        results = []
        for _ in range(5):
            ev = asyncio.run(buf.pop(timeout=0.1))
            results.append(ev["seq"])
        assert results == [0, 1, 2, 3, 4]

    def test_push_many(self):
        buf = BackpressureBuffer(max_size=10)
        count = buf.push_many([{"seq": i} for i in range(5)])
        assert count == 5
        assert buf.size == 5


# =========================================================================
# CLASS 2: BackpressureBuffer -- drop oldest / overflow
# =========================================================================


class TestBackpressureOverflow:
    def test_drop_oldest_when_full(self):
        buf = BackpressureBuffer(max_size=3)
        for i in range(5):
            buf.push({"seq": i})
        assert buf.size == 3
        assert buf.stats.dropped == 2
        # Remaining should be 2, 3, 4 (oldest 0, 1 dropped)
        results = []
        for _ in range(3):
            ev = asyncio.run(buf.pop(timeout=0.1))
            results.append(ev["seq"])
        assert results == [2, 3, 4]

    def test_stats_count_drops(self):
        buf = BackpressureBuffer(max_size=2)
        for i in range(10):
            buf.push({"seq": i})
        assert buf.stats.dropped == 8
        assert buf.stats.pushed == 10

    def test_push_many_stops_on_close(self):
        buf = BackpressureBuffer(max_size=100)
        buf.close()
        count = buf.push_many([{"seq": i} for i in range(10)])
        assert count == 0


# =========================================================================
# CLASS 3: BackpressureBuffer -- slow client detection
# =========================================================================


class TestBackpressureSlow:
    def test_is_slow_false_when_empty(self):
        buf = BackpressureBuffer(max_size=10, slow_threshold=0.8)
        assert buf.is_slow is False

    def test_is_slow_true_when_above_threshold(self):
        buf = BackpressureBuffer(max_size=10, slow_threshold=0.5)
        for i in range(6):
            buf.push({"seq": i})
        assert buf.is_slow is True

    def test_slow_warnings_incremented(self):
        buf = BackpressureBuffer(max_size=10, slow_threshold=0.5)
        for i in range(8):
            buf.push({"seq": i})
        assert buf.stats.slow_warnings > 0


# =========================================================================
# CLASS 4: BackpressureBuffer -- idle timeout
# =========================================================================


class TestBackpressureIdleTimeout:
    def test_not_timed_out_initially(self):
        buf = BackpressureBuffer(max_size=10, idle_timeout=60.0)
        assert buf.is_idle_timed_out is False

    def test_timed_out_after_idle(self):
        buf = BackpressureBuffer(max_size=10, idle_timeout=0.05)
        time.sleep(0.1)
        assert buf.is_idle_timed_out is True

    def test_pop_resets_idle_timer(self):
        buf = BackpressureBuffer(max_size=10, idle_timeout=0.5)
        buf.push({"seq": 0})
        asyncio.run(buf.pop(timeout=0.1))
        assert buf.is_idle_timed_out is False


# =========================================================================
# CLASS 5: BackpressureBuffer -- lifecycle
# =========================================================================


class TestBackpressureLifecycle:
    def test_close(self):
        buf = BackpressureBuffer(max_size=10)
        buf.push({"seq": 0})
        buf.close()
        assert buf.closed is True
        assert buf.push({"seq": 1}) is False

    def test_reset(self):
        buf = BackpressureBuffer(max_size=10)
        for i in range(5):
            buf.push({"seq": i})
        buf.close()
        buf.reset()
        assert buf.closed is False
        assert buf.size == 0
        assert buf.stats.pushed == 0

    def test_get_status(self):
        buf = BackpressureBuffer(max_size=10)
        buf.push({"seq": 0})
        status = buf.get_status()
        assert status["max_size"] == 10
        assert status["current_size"] == 1
        assert "stats" in status
        assert status["closed"] is False

    def test_drain_returns_all(self):
        buf = BackpressureBuffer(max_size=10)
        for i in range(5):
            buf.push({"seq": i})
        items = asyncio.run(buf.drain())
        assert len(items) == 5
        assert buf.size == 0
        assert buf.stats.popped == 5


# =========================================================================
# CLASS 6: BackpressureBuffer -- validation
# =========================================================================


class TestBackpressureValidation:
    def test_max_size_must_be_positive(self):
        with pytest.raises(ValueError, match="max_size"):
            BackpressureBuffer(max_size=0)

    def test_slow_threshold_range(self):
        with pytest.raises(ValueError, match="slow_threshold"):
            BackpressureBuffer(max_size=10, slow_threshold=0.0)
        with pytest.raises(ValueError, match="slow_threshold"):
            BackpressureBuffer(max_size=10, slow_threshold=1.5)

    def test_idle_timeout_must_be_positive(self):
        with pytest.raises(ValueError, match="idle_timeout"):
            BackpressureBuffer(max_size=10, idle_timeout=-1)


# =========================================================================
# CLASS 7: ConnectionPool -- basic checkout/checkin
# =========================================================================


class TestPoolBasic:
    def test_checkout_and_checkin(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path=db_path, pool_size=3, health_check=False)
            conn = pool.checkout()
            assert conn is not None
            conn.execute("SELECT 1")
            pool.checkin(conn)
            assert pool.available >= 1
            pool.close()
        finally:
            os.unlink(db_path)

    def test_context_manager(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path=db_path, pool_size=2, health_check=False)
            with pool.connection() as conn:
                conn.execute("SELECT 1")
            assert pool.available >= 1
            pool.close()
        finally:
            os.unlink(db_path)

    def test_multiple_checkouts(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path=db_path, pool_size=3, health_check=False)
            conns = []
            for _ in range(3):
                conns.append(pool.checkout())
            assert pool.in_use == 3
            for c in conns:
                pool.checkin(c)
            assert pool.available == 3
            pool.close()
        finally:
            os.unlink(db_path)


# =========================================================================
# CLASS 8: ConnectionPool -- WAL mode
# =========================================================================


class TestPoolWAL:
    def test_wal_mode_enforced(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path=db_path, pool_size=1, health_check=False, wal_mode=True)
            with pool.connection() as conn:
                result = conn.execute("PRAGMA journal_mode").fetchone()
                assert result[0] == "wal"
            pool.close()
        finally:
            os.unlink(db_path)

    def test_wal_mode_disabled(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path=db_path, pool_size=1, health_check=False, wal_mode=False)
            with pool.connection() as conn:
                result = conn.execute("PRAGMA journal_mode").fetchone()
                # Default is "delete" for fresh databases without WAL
                assert result[0] in ("delete", "memory")
            pool.close()
        finally:
            os.unlink(db_path)


# =========================================================================
# CLASS 9: ConnectionPool -- health check
# =========================================================================


class TestPoolHealthCheck:
    def test_health_check_passes(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path=db_path, pool_size=1, health_check=True)
            with pool.connection() as conn:
                conn.execute("SELECT 1")
            assert pool.stats.failed_health_checks == 0
            pool.close()
        finally:
            os.unlink(db_path)

    def test_health_check_skip(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path=db_path, pool_size=1, health_check=False)
            # The internal _check_health always returns True when disabled
            pc = PooledConnection(conn=MagicMock())
            assert pool._check_health(pc) is True
            pool.close()
        finally:
            os.unlink(db_path)


# =========================================================================
# CLASS 10: ConnectionPool -- exhaustion
# =========================================================================


class TestPoolExhaustion:
    def test_timeout_when_exhausted(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(
                db_path=db_path, pool_size=1,
                checkout_timeout=0.1, health_check=False,
            )
            c1 = pool.checkout()
            with pytest.raises(TimeoutError, match="exhausted"):
                pool.checkout()
            assert pool.stats.wait_timeouts == 1
            pool.checkin(c1)
            pool.close()
        finally:
            os.unlink(db_path)

    def test_checkout_after_closed_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path=db_path, pool_size=1, health_check=False)
            pool.close()
            with pytest.raises(RuntimeError, match="closed"):
                pool.checkout()
        finally:
            os.unlink(db_path)


# =========================================================================
# CLASS 11: ConnectionPool -- stats and status
# =========================================================================


class TestPoolStats:
    def test_stats_tracking(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path=db_path, pool_size=2, health_check=False)
            with pool.connection() as conn:
                conn.execute("SELECT 1")
            with pool.connection() as conn:
                conn.execute("SELECT 1")
            assert pool.stats.checkouts == 2
            assert pool.stats.checkins == 2
            pool.close()
        finally:
            os.unlink(db_path)

    def test_get_status(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path=db_path, pool_size=3, health_check=False)
            status = pool.get_status()
            assert status["pool_size"] == 3
            assert status["closed"] is False
            assert "stats" in status
            pool.close()
        finally:
            os.unlink(db_path)

    def test_pool_stats_to_dict(self):
        from opti_oignon.connection_pool import PoolStats
        ps = PoolStats(checkouts=10, checkins=8, created=3)
        d = ps.to_dict()
        assert d["checkouts"] == 10
        assert d["checkins"] == 8
        assert "avg_wait_ms" in d


# =========================================================================
# CLASS 12: ConnectionPool -- validation
# =========================================================================


class TestPoolValidation:
    def test_pool_size_must_be_positive(self):
        with pytest.raises(ValueError, match="pool_size"):
            ConnectionPool(db_path=":memory:", pool_size=0)

    def test_connect_timeout_must_be_positive(self):
        with pytest.raises(ValueError, match="connect_timeout"):
            ConnectionPool(db_path=":memory:", connect_timeout=0)


# =========================================================================
# CLASS 13: ConnectionPool -- registry (get_pool / close_all)
# =========================================================================


class TestPoolRegistry:
    def test_get_pool_singleton(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            # Clear registry
            pool_mod._pools.clear()
            p1 = get_pool(db_path, pool_size=2, health_check=False)
            p2 = get_pool(db_path, pool_size=5, health_check=False)
            assert p1 is p2
            assert p1.pool_size == 2  # First call wins
            close_all_pools()
        finally:
            os.unlink(db_path)

    def test_list_pools(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool_mod._pools.clear()
            get_pool(db_path, pool_size=2, health_check=False)
            pools = list_pools()
            assert len(pools) == 1
            assert pools[0]["pool_size"] == 2
            close_all_pools()
        finally:
            os.unlink(db_path)

    def test_close_all_pools(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool_mod._pools.clear()
            p = get_pool(db_path, pool_size=1, health_check=False)
            close_all_pools()
            assert p.closed is True
            assert len(pool_mod._pools) == 0
        finally:
            os.unlink(db_path)


# =========================================================================
# CLASS 14: Chunked JSON generator
# =========================================================================


class TestChunkedJson:
    def test_small_payload_single_chunk(self):
        payload = {"key": "value"}
        chunks = list(chunked_json_generator(payload, chunk_size=4096))
        assert len(chunks) == 1
        import json
        result = json.loads(b"".join(chunks))
        assert result["key"] == "value"

    def test_large_payload_multiple_chunks(self):
        payload = {"items": [{"id": i, "data": "x" * 200} for i in range(100)]}
        chunks = list(chunked_json_generator(payload, chunk_size=512))
        assert len(chunks) > 1
        import json
        result = json.loads(b"".join(chunks))
        assert len(result["items"]) == 100

    def test_chunk_size_respected(self):
        payload = {"data": "a" * 10000}
        chunks = list(chunked_json_generator(payload, chunk_size=1024))
        for chunk in chunks[:-1]:  # All but last should be near chunk_size
            assert len(chunk) <= 1024

    def test_empty_payload(self):
        chunks = list(chunked_json_generator({}, chunk_size=4096))
        import json
        result = json.loads(b"".join(chunks))
        assert result == {}

    def test_progress_callback(self):
        payload = {"data": "x" * 5000}
        progress_calls = []
        def on_progress(sent, total):
            progress_calls.append((sent, total))
        list(chunked_json_generator(payload, chunk_size=512, on_progress=on_progress))
        assert len(progress_calls) > 0
        # Last call should have sent == total
        assert progress_calls[-1][0] == progress_calls[-1][1]

    def test_utf8_boundary_safety(self):
        # Payload with multi-byte characters
        payload = {"text": "\u00e9\u00e8\u00ea" * 500}
        chunks = list(chunked_json_generator(payload, chunk_size=128))
        # Must reassemble to valid JSON
        import json
        result = json.loads(b"".join(chunks))
        assert "\u00e9" in result["text"]

    def test_chunk_size_minimum(self):
        with pytest.raises(ValueError, match="chunk_size"):
            list(chunked_json_generator({"a": 1}, chunk_size=10))


# =========================================================================
# CLASS 15: Chunked text generator
# =========================================================================


class TestChunkedText:
    def test_basic_text(self):
        text = "Hello world, this is a test string for chunked transfer."
        chunks = list(chunked_text_generator(text, chunk_size=64))
        assert "".join(chunks) == text

    def test_empty_text(self):
        chunks = list(chunked_text_generator("", chunk_size=64))
        assert chunks == []

    def test_progress_callback(self):
        text = "x" * 500
        calls = []
        list(chunked_text_generator(text, chunk_size=100, on_progress=lambda s, t: calls.append(s)))
        assert len(calls) == 5
        assert calls[-1] == 500

    def test_chunk_size_minimum(self):
        with pytest.raises(ValueError, match="chunk_size"):
            list(chunked_text_generator("hello", chunk_size=32))


# =========================================================================
# CLASS 16: AST validity of all new/modified files
# =========================================================================


class TestASTValidity:
    def test_sse_backpressure_ast(self):
        _ast_parse(BACKPRESSURE_PATH)

    def test_connection_pool_ast(self):
        _ast_parse(POOL_PATH)

    def test_chunked_response_ast(self):
        _ast_parse(CHUNKED_PATH)

    def test_bench_streaming_ast(self):
        _ast_parse(BENCH_PATH)

    def test_routes_chat_ast(self):
        _ast_parse(ROUTES_CHAT_PATH)

    def test_routes_rag_ast(self):
        _ast_parse(ROUTES_RAG_PATH)


# =========================================================================
# CLASS 17: checkpoint_before_apply sentinels
# =========================================================================


class TestCheckpointSentinels:
    def _has_sentinel(self, path):
        with open(path) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "checkpoint_before_apply":
                        if isinstance(node.value, ast.Constant) and node.value.value is True:
                            return True
        return False

    def test_backpressure_sentinel(self):
        assert self._has_sentinel(BACKPRESSURE_PATH)

    def test_connection_pool_sentinel(self):
        assert self._has_sentinel(POOL_PATH)

    def test_chunked_response_sentinel(self):
        assert self._has_sentinel(CHUNKED_PATH)

    def test_bench_streaming_sentinel(self):
        assert self._has_sentinel(BENCH_PATH)


# =========================================================================
# CLASS 18: Module availability flags
# =========================================================================


class TestModuleFlags:
    def test_backpressure_available(self):
        assert bp_mod.SSE_BACKPRESSURE_AVAILABLE is True

    def test_connection_pool_available(self):
        assert pool_mod.CONNECTION_POOL_AVAILABLE is True

    def test_chunked_response_available(self):
        assert chunked_mod.CHUNKED_RESPONSE_AVAILABLE is True


# =========================================================================
# CLASS 19: Integration -- routes_chat.py backpressure constants
# =========================================================================


class TestRoutesChatIntegration:
    def test_backpressure_import_present(self):
        with open(ROUTES_CHAT_PATH) as f:
            source = f.read()
        assert "BackpressureBuffer" in source
        assert "BACKPRESSURE_AVAILABLE" in source

    def test_backpressure_constants_present(self):
        with open(ROUTES_CHAT_PATH) as f:
            source = f.read()
        assert "_BP_MAX_SIZE" in source
        assert "_BP_SLOW_THRESHOLD" in source
        assert "_BP_IDLE_TIMEOUT" in source

    def test_backpressure_critical_events(self):
        with open(ROUTES_CHAT_PATH) as f:
            source = f.read()
        assert "_BP_CRITICAL_EVENTS" in source
        assert '"error"' in source

    def test_done_metadata_backpressure(self):
        with open(ROUTES_CHAT_PATH) as f:
            source = f.read()
        assert '"backpressure"' in source
        assert '"events_dropped"' in source


# =========================================================================
# CLASS 20: Integration -- routes_rag.py chunked endpoint
# =========================================================================


class TestRoutesRagIntegration:
    def test_stream_endpoint_present(self):
        with open(ROUTES_RAG_PATH) as f:
            source = f.read()
        assert "/query/stream" in source

    def test_streaming_response_import(self):
        with open(ROUTES_RAG_PATH) as f:
            source = f.read()
        assert "StreamingResponse" in source

    def test_chunked_response_import(self):
        with open(ROUTES_RAG_PATH) as f:
            source = f.read()
        assert "chunked_json_generator" in source
        assert "CHUNKED_RESPONSE_AVAILABLE" in source

    def test_chunk_size_field_in_query_request(self):
        with open(ROUTES_RAG_PATH) as f:
            source = f.read()
        assert "chunk_size" in source

    def test_transfer_encoding_header(self):
        with open(ROUTES_RAG_PATH) as f:
            source = f.read()
        assert "Transfer-Encoding" in source


# =========================================================================
# CLASS 21: ConnectionPool -- concurrent checkout/checkin
# =========================================================================


class TestPoolConcurrency:
    def test_concurrent_checkout_checkin(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(
                db_path=db_path, pool_size=3,
                checkout_timeout=5.0, health_check=False,
            )
            with pool.connection() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)"
                )
                conn.commit()

            errors = []
            lock = threading.Lock()

            def worker(n):
                try:
                    for i in range(10):
                        with pool.connection() as conn:
                            conn.execute("INSERT INTO t (v) VALUES (?)", (f"w{n}_{i}",))
                            conn.commit()
                except Exception as exc:
                    with lock:
                        errors.append(str(exc))

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)

            assert errors == [], f"Errors during concurrent access: {errors}"
            assert pool.stats.checkouts >= 50  # 5 threads * 10 ops + 1 init

            with pool.connection() as conn:
                count = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
            assert count == 50
            pool.close()
        finally:
            os.unlink(db_path)


# =========================================================================
# CLASS 22: BackpressureBuffer -- async drain with timeout
# =========================================================================


class TestBackpressureDrain:
    def test_drain_empty_with_timeout(self):
        buf = BackpressureBuffer(max_size=10)
        items = asyncio.run(buf.drain(timeout=0.05))
        assert items == []

    def test_drain_empty_no_timeout(self):
        buf = BackpressureBuffer(max_size=10)
        items = asyncio.run(buf.drain(timeout=0.0))
        assert items == []

    def test_drain_after_close(self):
        buf = BackpressureBuffer(max_size=10)
        buf.push({"seq": 0})
        buf.close()
        items = asyncio.run(buf.drain())
        assert len(items) == 1


# =========================================================================
# CLASS 23: BufferStats dataclass
# =========================================================================


class TestBufferStats:
    def test_default_values(self):
        s = BufferStats()
        assert s.pushed == 0
        assert s.popped == 0
        assert s.dropped == 0
        assert s.slow_warnings == 0

    def test_to_dict(self):
        s = BufferStats(pushed=10, popped=8, dropped=2, slow_warnings=1)
        d = s.to_dict()
        assert d["pushed"] == 10
        assert d["dropped"] == 2
        assert "created_at" in d


# =========================================================================
# CLASS 24: Benchmark script structure
# =========================================================================


class TestBenchmarkScript:
    def test_has_main_function(self):
        tree = _ast_parse(BENCH_PATH)
        func_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        assert "main" in func_names
        assert "bench_backpressure" in func_names
        assert "bench_connection_pool" in func_names
        assert "bench_chunked_response" in func_names
        assert "compare_results" in func_names

    def test_has_argparse(self):
        with open(BENCH_PATH) as f:
            source = f.read()
        assert "argparse" in source
        assert "--json" in source
        assert "--compare" in source

    def test_importlib_isolation(self):
        with open(BENCH_PATH) as f:
            source = f.read()
        assert "importlib.util.spec_from_file_location" in source
        assert "_ensure_modules" in source


# =========================================================================
# CLASS 25: Chunked JSON -- reassembly correctness
# =========================================================================


class TestChunkedReassembly:
    def test_nested_payload(self):
        import json
        payload = {
            "query": "test query",
            "results": [
                {"content": "chunk " * 100, "score": 0.95, "nested": {"a": [1, 2, 3]}}
                for _ in range(10)
            ],
            "metadata": {"total": 10, "collection": "default"},
        }
        chunks = list(chunked_json_generator(payload, chunk_size=256))
        result = json.loads(b"".join(chunks))
        assert result["query"] == "test query"
        assert len(result["results"]) == 10
        assert result["results"][0]["nested"]["a"] == [1, 2, 3]

    def test_unicode_heavy_payload(self):
        import json
        payload = {"text": "\U0001f600\U0001f601\U0001f602" * 200}
        chunks = list(chunked_json_generator(payload, chunk_size=128))
        result = json.loads(b"".join(chunks))
        assert "\U0001f600" in result["text"]

    def test_exact_chunk_boundary(self):
        import json
        # Payload that is exactly chunk_size bytes
        data = "a" * 100
        payload = {"d": data}
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        chunks = list(chunked_json_generator(payload, chunk_size=len(raw)))
        assert len(chunks) == 1
        assert b"".join(chunks) == raw
