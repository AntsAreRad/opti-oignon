"""
tests/test_s160_async_plugins.py -- S160 async plugins + parallel RAG tests.

Verifies:
- Goal 1: Async plugin subprocess (lifecycle, IPC, timeout, shutdown escalation)
- Goal 2: Parallel RAG ingestion (concurrent chunks, error isolation, progress)
- Goal 3: Batch embedding manager + connection pool integration
- Goal 4: Module structure (checkpoint_before_apply, FEATURE_AVAILABLE, AST)
"""

import ast
import asyncio
import importlib.util
import json
import os
import sqlite3
import struct
import sys
import tempfile
import threading
import time
import types
from unittest.mock import AsyncMock, MagicMock, patch

# -- Isolation stubs (standard pattern) --
for mod_name in [
    "opti_oignon",
    "opti_oignon.db_utils",
    "opti_oignon.db_encryption",
    "opti_oignon.config",
    "opti_oignon.auth",
    "opti_oignon.middleware",
    "opti_oignon.security_mode",
    "opti_oignon.connection_pool",
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
        if mod_name == "opti_oignon.connection_pool":
            # Provide a minimal ConnectionPool stub for pool_integration
            class _StubConnectionPool:
                def __init__(self, db_path, **kw):
                    self._db_path = db_path
                    self._kw = kw
                    self._closed = False
                import contextlib
                @contextlib.contextmanager
                def connection(self):
                    conn = sqlite3.connect(":memory:")
                    try:
                        yield conn
                    finally:
                        conn.close()
                def close(self):
                    self._closed = True
            stub.ConnectionPool = _StubConnectionPool
        sys.modules[mod_name] = stub

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASYNC_PLUGIN_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "async_plugin_subprocess.py"
)
PARALLEL_INGEST_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "rag", "parallel_ingest.py"
)
EMBEDDINGS_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "rag", "embeddings.py"
)
POOL_INTEGRATION_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "rag", "pool_integration.py"
)


# -- Helpers -----------------------------------------------------------------

def _load_module(name, path):
    """Load a module via importlib isolation."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_async_plugin():
    return _load_module("async_plugin_subprocess", ASYNC_PLUGIN_PATH)


def _load_parallel_ingest():
    return _load_module("parallel_ingest", PARALLEL_INGEST_PATH)


def _load_embeddings():
    # Need to stub RAG config dependencies
    rag_config_stub = types.ModuleType("opti_oignon.rag.config")

    class _EmbConf:
        model = "test-embed"
        fast_model = "test-fast"
        ollama_url = "http://localhost:11434"
        dimension = 128
        batch_size = 32
        timeout = 10

    class _RAGConf:
        embedding = _EmbConf()

    rag_config_stub.EmbeddingConfig = _EmbConf
    rag_config_stub.get_config = lambda: _RAGConf()

    if "opti_oignon.rag" not in sys.modules:
        rag_pkg = types.ModuleType("opti_oignon.rag")
        rag_pkg.__path__ = [
            os.path.join(PROJECT_ROOT, "opti_oignon", "rag")
        ]
        sys.modules["opti_oignon.rag"] = rag_pkg
    sys.modules["opti_oignon.rag.config"] = rag_config_stub

    # Stub numpy, requests, tqdm
    if "numpy" not in sys.modules:
        np_stub = types.ModuleType("numpy")
        np_stub.array = lambda x, **kw: x
        np_stub.ndarray = list  # type stub for annotations
        np_stub.float32 = float
        class _linalg:
            @staticmethod
            def norm(arr, **kw):
                return [[1.0]] * len(arr)
        np_stub.linalg = _linalg()
        np_stub.where = lambda cond, a, b: a
        sys.modules["numpy"] = np_stub
    else:
        # Ensure ndarray exists on previously created stub
        if not hasattr(sys.modules["numpy"], "ndarray"):
            sys.modules["numpy"].ndarray = list

    if "requests" not in sys.modules:
        req_stub = types.ModuleType("requests")
        class _ReqExc:
            class ConnectionError(Exception): pass  # noqa: E701
            class Timeout(Exception): pass  # noqa: E701
            class RequestException(Exception): pass  # noqa: E701
        req_stub.exceptions = _ReqExc()
        req_stub.get = MagicMock()
        req_stub.post = MagicMock()
        sys.modules["requests"] = req_stub

    if "tqdm" not in sys.modules:
        tqdm_stub = types.ModuleType("tqdm")
        tqdm_stub.tqdm = lambda x, **kw: x
        sys.modules["tqdm"] = tqdm_stub

    # Load as a proper submodule so relative imports work
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.rag.embeddings", EMBEDDINGS_PATH,
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "opti_oignon.rag"
    spec.loader.exec_module(mod)
    return mod


def _load_pool_integration():
    return _load_module("pool_integration", POOL_INTEGRATION_PATH)


# ============================================================================
# Goal 1: Async Plugin Subprocess
# ============================================================================

class TestAsyncPluginModuleStructure:
    """Module-level checks for async_plugin_subprocess.py."""

    def test_ast_validity(self):
        with open(ASYNC_PLUGIN_PATH) as f:
            tree = ast.parse(f.read())
        assert tree is not None

    def test_checkpoint_before_apply(self):
        mod = _load_async_plugin()
        assert hasattr(mod, "checkpoint_before_apply")
        assert mod.checkpoint_before_apply is True

    def test_feature_available(self):
        mod = _load_async_plugin()
        assert mod.FEATURE_AVAILABLE is True

    def test_no_emoji_in_source(self):
        with open(ASYNC_PLUGIN_PATH, encoding="utf-8") as f:
            content = f.read()
        for ch in content:
            assert ord(ch) < 0x1F600 or ord(ch) > 0x1F9FF

    def test_all_comments_english(self):
        with open(ASYNC_PLUGIN_PATH, encoding="utf-8") as f:
            content = f.read()
        french_words = ["fonction", "initialiser", "retourne", "parametres"]
        lower = content.lower()
        for word in french_words:
            # Allow 'parametres' only in identifiers, not comments
            assert f"# {word}" not in lower


class TestEncodeDecodeMessage:
    """Wire protocol: encode/decode length-prefixed JSON."""

    def test_roundtrip_simple(self):
        mod = _load_async_plugin()
        payload = {"method": "ping", "params": {}}
        data = mod.encode_message(payload)
        # First 4 bytes are length
        length = struct.unpack("!I", data[:4])[0]
        assert length > 0
        decoded = mod.decode_message(data)
        assert decoded == payload

    def test_roundtrip_nested(self):
        mod = _load_async_plugin()
        payload = {"method": "hook", "params": {"data": [1, 2, 3], "nested": {"a": True}}}
        data = mod.encode_message(payload)
        decoded = mod.decode_message(data)
        assert decoded == payload

    def test_empty_payload(self):
        mod = _load_async_plugin()
        data = mod.encode_message({})
        decoded = mod.decode_message(data)
        assert decoded == {}

    def test_decode_too_short_raises(self):
        mod = _load_async_plugin()
        with pytest.raises(mod.AsyncPluginIPCError, match="too short"):
            mod.decode_message(b"\x00\x00")

    def test_decode_length_mismatch_raises(self):
        mod = _load_async_plugin()
        # Header says 100 bytes but only 5 bytes follow
        header = struct.pack("!I", 100)
        with pytest.raises(mod.AsyncPluginIPCError, match="mismatch"):
            mod.decode_message(header + b"hello")

    def test_decode_invalid_json_raises(self):
        mod = _load_async_plugin()
        raw = b"not json at all"
        header = struct.pack("!I", len(raw))
        with pytest.raises(mod.AsyncPluginIPCError, match="JSON"):
            mod.decode_message(header + raw)

    def test_encode_oversized_raises(self):
        mod = _load_async_plugin()
        payload = {"data": "x" * (mod.MAX_ASYNC_MESSAGE_SIZE + 1)}
        with pytest.raises(mod.AsyncPluginIPCError, match="exceeds"):
            mod.encode_message(payload)

    def test_encode_with_non_serializable_uses_default_str(self):
        mod = _load_async_plugin()
        from pathlib import Path
        payload = {"path": Path("/tmp/test")}
        data = mod.encode_message(payload)
        decoded = mod.decode_message(data)
        assert decoded["path"] == "/tmp/test"


class TestAsyncPluginProcess:
    """AsyncPluginProcess dataclass."""

    def test_creation(self):
        mod = _load_async_plugin()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.pid = 12345
        app = mod.AsyncPluginProcess(
            plugin_name="test-plugin",
            process=mock_proc,
            worker_script="/tmp/worker.py",
        )
        assert app.plugin_name == "test-plugin"
        assert app.is_alive is True
        assert app.pid == 12345
        assert app.call_count == 0

    def test_is_alive_false_when_exited(self):
        mod = _load_async_plugin()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        app = mod.AsyncPluginProcess(
            plugin_name="dead",
            process=mock_proc,
            worker_script="/tmp/w.py",
        )
        assert app.is_alive is False
        assert app.pid is None

    def test_to_dict(self):
        mod = _load_async_plugin()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.pid = 999
        app = mod.AsyncPluginProcess(
            plugin_name="info",
            process=mock_proc,
            worker_script="/tmp/w.py",
            call_timeout=15.0,
        )
        d = app.to_dict()
        assert d["plugin_name"] == "info"
        assert d["pid"] == 999
        assert d["alive"] is True
        assert d["call_timeout"] == 15.0
        assert "uptime_s" in d

    def test_default_timeout(self):
        mod = _load_async_plugin()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        app = mod.AsyncPluginProcess(
            plugin_name="t",
            process=mock_proc,
            worker_script="/tmp/w.py",
        )
        assert app.call_timeout == mod.DEFAULT_CALL_TIMEOUT_S


class TestAsyncPluginManagerInit:
    """AsyncPluginSubprocessManager initialization."""

    def test_default_init(self):
        mod = _load_async_plugin()
        mgr = mod.AsyncPluginSubprocessManager()
        assert mgr.running_plugins == []
        assert mgr.default_call_timeout == mod.DEFAULT_CALL_TIMEOUT_S

    def test_custom_timeouts(self):
        mod = _load_async_plugin()
        mgr = mod.AsyncPluginSubprocessManager(
            default_call_timeout=5.0,
            startup_timeout=3.0,
            shutdown_grace=2.0,
        )
        assert mgr.default_call_timeout == 5.0

    def test_custom_worker_script(self):
        mod = _load_async_plugin()
        mgr = mod.AsyncPluginSubprocessManager(worker_script="/custom/worker.py")
        assert mgr._worker_script == "/custom/worker.py"


class TestAsyncPluginManagerLifecycle:
    """Manager start/stop/call (mocked subprocess)."""

    def _make_ready_response(self, mod):
        """Create bytes for a ready message."""
        return mod.encode_message({"status": "ready"})

    @pytest.mark.asyncio
    async def test_start_plugin_success(self):
        mod = _load_async_plugin()
        mgr = mod.AsyncPluginSubprocessManager(worker_script="/tmp/w.py")

        ready_data = self._make_ready_response(mod)

        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.pid = 5555
        mock_proc.stdout = AsyncMock()
        mock_proc.stderr = AsyncMock()

        # Configure stdout to return the ready message
        mock_proc.stdout.readexactly = AsyncMock(side_effect=[
            ready_data[:4],  # length header
            ready_data[4:],  # payload
        ])
        mock_proc.stderr.readline = AsyncMock(return_value=b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            app = await mgr.start_plugin("plug1", "/tmp", "entry.py")

        assert app.plugin_name == "plug1"
        assert "plug1" in mgr.running_plugins

    @pytest.mark.asyncio
    async def test_stop_plugin_not_running(self):
        mod = _load_async_plugin()
        mgr = mod.AsyncPluginSubprocessManager()
        result = await mgr.stop_plugin("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_is_running_false_when_empty(self):
        mod = _load_async_plugin()
        mgr = mod.AsyncPluginSubprocessManager()
        assert await mgr.is_running("nope") is False

    @pytest.mark.asyncio
    async def test_list_plugins_empty(self):
        mod = _load_async_plugin()
        mgr = mod.AsyncPluginSubprocessManager()
        result = await mgr.list_plugins()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_status_none_when_missing(self):
        mod = _load_async_plugin()
        mgr = mod.AsyncPluginSubprocessManager()
        assert await mgr.get_status("missing") is None

    @pytest.mark.asyncio
    async def test_call_plugin_not_running_raises(self):
        mod = _load_async_plugin()
        mgr = mod.AsyncPluginSubprocessManager()
        with pytest.raises(mod.AsyncPluginNotRunning):
            await mgr.call_plugin("missing", "test", {})

    @pytest.mark.asyncio
    async def test_ping_not_running_returns_false(self):
        mod = _load_async_plugin()
        mgr = mod.AsyncPluginSubprocessManager()
        result = await mgr.ping("missing")
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_all_empty(self):
        mod = _load_async_plugin()
        mgr = mod.AsyncPluginSubprocessManager()
        count = await mgr.stop_all()
        assert count == 0


class TestAsyncPluginSingleton:
    """Module-level singleton management."""

    def test_get_and_reset(self):
        mod = _load_async_plugin()
        mod.reset_async_plugin_manager()
        mgr1 = mod.get_async_plugin_manager()
        mgr2 = mod.get_async_plugin_manager()
        assert mgr1 is mgr2
        mod.reset_async_plugin_manager()
        mgr3 = mod.get_async_plugin_manager()
        assert mgr3 is not mgr1


class TestAsyncPluginExceptions:
    """Exception hierarchy."""

    def test_hierarchy(self):
        mod = _load_async_plugin()
        assert issubclass(mod.AsyncPluginTimeout, mod.AsyncPluginError)
        assert issubclass(mod.AsyncPluginIPCError, mod.AsyncPluginError)
        assert issubclass(mod.AsyncPluginNotRunning, mod.AsyncPluginError)

    def test_base_is_exception(self):
        mod = _load_async_plugin()
        assert issubclass(mod.AsyncPluginError, Exception)


# ============================================================================
# Goal 2: Parallel RAG Ingestion
# ============================================================================

class TestParallelIngestModuleStructure:
    """Module-level checks for parallel_ingest.py."""

    def test_ast_validity(self):
        with open(PARALLEL_INGEST_PATH) as f:
            tree = ast.parse(f.read())
        assert tree is not None

    def test_checkpoint_before_apply(self):
        mod = _load_parallel_ingest()
        assert mod.checkpoint_before_apply is True

    def test_feature_available(self):
        mod = _load_parallel_ingest()
        assert mod.FEATURE_AVAILABLE is True


class TestChunkResult:
    """ChunkResult dataclass."""

    def test_to_dict_success(self):
        mod = _load_parallel_ingest()
        r = mod.ChunkResult(
            chunk_index=0,
            chunk_id="c1",
            status=mod.ChunkStatus.SUCCESS,
            elapsed_s=0.5,
            embedding_dim=128,
        )
        d = r.to_dict()
        assert d["status"] == "success"
        assert d["embedding_dim"] == 128
        assert "error_message" not in d

    def test_to_dict_failed(self):
        mod = _load_parallel_ingest()
        r = mod.ChunkResult(
            chunk_index=1,
            chunk_id="c2",
            status=mod.ChunkStatus.FAILED,
            error_message="boom",
        )
        d = r.to_dict()
        assert d["status"] == "failed"
        assert d["error_message"] == "boom"


class TestIngestBatchResult:
    """IngestBatchResult aggregation."""

    def test_summary_empty(self):
        mod = _load_parallel_ingest()
        r = mod.IngestBatchResult()
        s = r.summary()
        assert s["total"] == 0
        assert s["success_rate"] == 0.0

    def test_all_succeeded(self):
        mod = _load_parallel_ingest()
        r = mod.IngestBatchResult(total=2, success=2)
        assert r.all_succeeded is True

    def test_not_all_succeeded(self):
        mod = _load_parallel_ingest()
        r = mod.IngestBatchResult(total=3, success=2, failed=1)
        assert r.all_succeeded is False

    def test_failed_chunks_filter(self):
        mod = _load_parallel_ingest()
        results = [
            mod.ChunkResult(0, "a", mod.ChunkStatus.SUCCESS),
            mod.ChunkResult(1, "b", mod.ChunkStatus.FAILED, error_message="err"),
            mod.ChunkResult(2, "c", mod.ChunkStatus.SUCCESS),
        ]
        r = mod.IngestBatchResult(total=3, success=2, failed=1, chunk_results=results)
        failed = r.failed_chunks()
        assert len(failed) == 1
        assert failed[0].chunk_id == "b"


class TestProgressTracker:
    """Thread-safe progress tracking."""

    def test_increment(self):
        mod = _load_parallel_ingest()
        tracker = mod.ProgressTracker(10)
        assert tracker.done == 0
        assert tracker.total == 10
        tracker.increment()
        assert tracker.done == 1

    def test_callback_called(self):
        mod = _load_parallel_ingest()
        calls = []
        tracker = mod.ProgressTracker(5, callback=lambda d, t: calls.append((d, t)))
        tracker.increment()
        tracker.increment()
        assert calls == [(1, 5), (2, 5)]

    def test_callback_error_does_not_propagate(self):
        mod = _load_parallel_ingest()
        def bad_cb(d, t):
            raise ValueError("boom")
        tracker = mod.ProgressTracker(3, callback=bad_cb)
        # Should not raise
        tracker.increment()
        assert tracker.done == 1

    def test_thread_safety(self):
        mod = _load_parallel_ingest()
        tracker = mod.ProgressTracker(1000)
        def inc_many():
            for _ in range(100):
                tracker.increment()
        threads = [threading.Thread(target=inc_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert tracker.done == 1000


class TestParallelIngestWorker:
    """Core parallel ingestion logic."""

    def test_init_defaults(self):
        mod = _load_parallel_ingest()
        w = mod.ParallelIngestWorker()
        assert w.max_workers == mod.DEFAULT_MAX_WORKERS
        assert w.chunk_timeout == mod.DEFAULT_CHUNK_TIMEOUT_S

    def test_init_invalid_workers(self):
        mod = _load_parallel_ingest()
        with pytest.raises(ValueError, match="must be >= 1"):
            mod.ParallelIngestWorker(max_workers=0)

    def test_empty_chunks(self):
        mod = _load_parallel_ingest()
        w = mod.ParallelIngestWorker()
        result = w.ingest_chunks([], lambda t: [0.1], lambda *a: None)
        assert result.total == 0

    def test_all_success(self):
        mod = _load_parallel_ingest()
        w = mod.ParallelIngestWorker(max_workers=2)
        chunks = [{"chunk_id": f"c{i}", "text": f"hello {i}"} for i in range(5)]
        embed_fn = lambda t: [0.1, 0.2, 0.3]
        stored = []
        store_fn = lambda cid, emb, meta: stored.append(cid)

        result = w.ingest_chunks(chunks, embed_fn, store_fn)
        assert result.total == 5
        assert result.success == 5
        assert result.failed == 0
        assert result.all_succeeded
        assert len(stored) == 5

    def test_error_isolation(self):
        mod = _load_parallel_ingest()
        w = mod.ParallelIngestWorker(max_workers=2)
        chunks = [
            {"chunk_id": "ok1", "text": "good"},
            {"chunk_id": "bad", "text": "fail"},
            {"chunk_id": "ok2", "text": "good"},
        ]

        def embed_fn(t):
            if t == "fail":
                raise RuntimeError("embedding crash")
            return [0.1]

        result = w.ingest_chunks(chunks, embed_fn, lambda *a: None)
        assert result.total == 3
        assert result.success == 2
        assert result.failed == 1
        failed = result.failed_chunks()
        assert len(failed) == 1
        assert "embedding crash" in failed[0].error_message

    def test_skip_empty(self):
        mod = _load_parallel_ingest()
        w = mod.ParallelIngestWorker(max_workers=1)
        chunks = [
            {"chunk_id": "c1", "text": "hello"},
            {"chunk_id": "c2", "text": ""},
            {"chunk_id": "c3", "text": "   "},
        ]
        result = w.ingest_chunks(chunks, lambda t: [0.1], lambda *a: None)
        assert result.success == 1
        assert result.skipped == 2

    def test_no_skip_empty_when_disabled(self):
        mod = _load_parallel_ingest()
        w = mod.ParallelIngestWorker(max_workers=1)
        chunks = [{"chunk_id": "c1", "text": ""}]
        result = w.ingest_chunks(
            chunks, lambda t: [0.1], lambda *a: None, skip_empty=False,
        )
        assert result.success == 1
        assert result.skipped == 0

    def test_embed_returns_none_is_failure(self):
        mod = _load_parallel_ingest()
        w = mod.ParallelIngestWorker(max_workers=1)
        chunks = [{"chunk_id": "c1", "text": "test"}]
        result = w.ingest_chunks(chunks, lambda t: None, lambda *a: None)
        assert result.failed == 1
        assert "None" in result.chunk_results[0].error_message

    def test_progress_callback(self):
        mod = _load_parallel_ingest()
        w = mod.ParallelIngestWorker(max_workers=1)
        chunks = [{"chunk_id": f"c{i}", "text": f"t{i}"} for i in range(3)]
        progress = []
        result = w.ingest_chunks(
            chunks, lambda t: [0.1], lambda *a: None,
            progress_cb=lambda d, t: progress.append((d, t)),
        )
        assert result.success == 3
        assert len(progress) == 3
        assert progress[-1] == (3, 3)

    def test_plain_string_chunks(self):
        mod = _load_parallel_ingest()
        w = mod.ParallelIngestWorker(max_workers=1)
        chunks = ["hello", "world"]
        result = w.ingest_chunks(chunks, lambda t: [0.5], lambda *a: None)
        assert result.success == 2

    def test_object_chunks_with_metadata(self):
        mod = _load_parallel_ingest()
        w = mod.ParallelIngestWorker(max_workers=1)
        chunk = mod.SimpleChunk(chunk_id="sc1", text="data", metadata={"src": "test"})
        result = w.ingest_chunks([chunk], lambda t: [0.1], lambda *a: None)
        assert result.success == 1
        assert result.chunk_results[0].metadata.get("src") == "test"

    def test_store_fn_error_is_failure(self):
        mod = _load_parallel_ingest()
        w = mod.ParallelIngestWorker(max_workers=1)
        chunks = [{"chunk_id": "c1", "text": "data"}]
        def bad_store(*a):
            raise OSError("disk full")
        result = w.ingest_chunks(chunks, lambda t: [0.1], bad_store)
        assert result.failed == 1
        assert "disk full" in result.chunk_results[0].error_message

    def test_elapsed_time_tracked(self):
        mod = _load_parallel_ingest()
        w = mod.ParallelIngestWorker(max_workers=1)
        chunks = [{"chunk_id": "c1", "text": "data"}]
        result = w.ingest_chunks(chunks, lambda t: [0.1], lambda *a: None)
        assert result.elapsed_s >= 0
        assert result.chunk_results[0].elapsed_s >= 0

    def test_concurrent_workers(self):
        mod = _load_parallel_ingest()
        w = mod.ParallelIngestWorker(max_workers=4)
        active_threads = []
        lock = threading.Lock()

        def slow_embed(t):
            with lock:
                active_threads.append(threading.current_thread().name)
            time.sleep(0.05)
            return [0.1]

        chunks = [{"chunk_id": f"c{i}", "text": f"t{i}"} for i in range(8)]
        result = w.ingest_chunks(chunks, slow_embed, lambda *a: None)
        assert result.success == 8
        # Verify multiple threads were used
        unique = set(active_threads)
        assert len(unique) >= 2


class TestParallelIngestConvenience:
    """parallel_ingest() convenience function."""

    def test_basic_call(self):
        mod = _load_parallel_ingest()
        result = mod.parallel_ingest(
            ["a", "b", "c"],
            lambda t: [0.1],
            lambda *a: None,
            max_workers=2,
        )
        assert result.success == 3

    def test_with_progress(self):
        mod = _load_parallel_ingest()
        events = []
        result = mod.parallel_ingest(
            ["x"],
            lambda t: [0.1],
            lambda *a: None,
            progress_cb=lambda d, t: events.append(d),
        )
        assert events == [1]


class TestChunkStatusEnum:
    """ChunkStatus enum values."""

    def test_values(self):
        mod = _load_parallel_ingest()
        assert mod.ChunkStatus.PENDING.value == "pending"
        assert mod.ChunkStatus.SUCCESS.value == "success"
        assert mod.ChunkStatus.FAILED.value == "failed"
        assert mod.ChunkStatus.SKIPPED.value == "skipped"


# ============================================================================
# Goal 3: Batch Embedding Manager
# ============================================================================

class TestBatchEmbeddingManagerStructure:
    """Module-level checks for BatchEmbeddingManager in embeddings.py."""

    def test_ast_validity(self):
        with open(EMBEDDINGS_PATH) as f:
            tree = ast.parse(f.read())
        assert tree is not None

    def test_checkpoint_before_apply_in_source(self):
        with open(EMBEDDINGS_PATH) as f:
            content = f.read()
        assert "checkpoint_before_apply = True" in content

    def test_class_exists(self):
        mod = _load_embeddings()
        assert hasattr(mod, "BatchEmbeddingManager")


class TestBatchEmbeddingManager:
    """BatchEmbeddingManager functionality."""

    def _make_embedder(self, mod):
        """Create a mock embedder."""
        embedder = MagicMock(spec=mod.OllamaEmbeddings)
        embedder.embed_batch = MagicMock(
            side_effect=lambda texts: [[0.1] * 3 for _ in texts]
        )
        return embedder

    def test_init_defaults(self):
        mod = _load_embeddings()
        embedder = self._make_embedder(mod)
        mgr = mod.BatchEmbeddingManager(embedder)
        assert mgr.batch_size == mod.DEFAULT_EMBEDDING_BATCH_SIZE
        assert mgr.pending_count == 0

    def test_init_invalid_batch_size(self):
        mod = _load_embeddings()
        embedder = self._make_embedder(mod)
        with pytest.raises(ValueError, match="must be >= 1"):
            mod.BatchEmbeddingManager(embedder, batch_size=0)

    def test_add_below_threshold_returns_none(self):
        mod = _load_embeddings()
        embedder = self._make_embedder(mod)
        mgr = mod.BatchEmbeddingManager(embedder, batch_size=5)
        result = mgr.add("hello")
        assert result is None
        assert mgr.pending_count == 1

    def test_add_triggers_flush_at_batch_size(self):
        mod = _load_embeddings()
        embedder = self._make_embedder(mod)
        mgr = mod.BatchEmbeddingManager(embedder, batch_size=3)
        mgr.add("a")
        mgr.add("b")
        result = mgr.add("c")  # triggers flush
        assert result is not None
        assert len(result) == 3
        assert mgr.pending_count == 0
        embedder.embed_batch.assert_called_once()

    def test_flush_empties_pending(self):
        mod = _load_embeddings()
        embedder = self._make_embedder(mod)
        mgr = mod.BatchEmbeddingManager(embedder, batch_size=10)
        mgr.add("x")
        mgr.add("y")
        assert mgr.pending_count == 2
        result = mgr.flush()
        assert len(result) == 2
        assert mgr.pending_count == 0

    def test_flush_empty_returns_empty(self):
        mod = _load_embeddings()
        embedder = self._make_embedder(mod)
        mgr = mod.BatchEmbeddingManager(embedder, batch_size=5)
        result = mgr.flush()
        assert result == []

    def test_callback_called_per_text(self):
        mod = _load_embeddings()
        embedder = self._make_embedder(mod)
        mgr = mod.BatchEmbeddingManager(embedder, batch_size=2)
        results = []
        mgr.add("a", callback=lambda emb: results.append(("a", emb)))
        mgr.add("b", callback=lambda emb: results.append(("b", emb)))
        assert len(results) == 2
        assert results[0][0] == "a"
        assert results[1][0] == "b"

    def test_stats_tracking(self):
        mod = _load_embeddings()
        embedder = self._make_embedder(mod)
        mgr = mod.BatchEmbeddingManager(embedder, batch_size=2)
        mgr.add("a")
        mgr.add("b")  # triggers flush
        stats = mgr.stats
        assert stats["total_batches"] == 1
        assert stats["total_texts"] == 2
        assert stats["total_errors"] == 0

    def test_reset_stats(self):
        mod = _load_embeddings()
        embedder = self._make_embedder(mod)
        mgr = mod.BatchEmbeddingManager(embedder, batch_size=1)
        mgr.add("x")
        assert mgr.stats["total_batches"] == 1
        mgr.reset_stats()
        assert mgr.stats["total_batches"] == 0

    def test_embed_many_basic(self):
        mod = _load_embeddings()
        embedder = self._make_embedder(mod)
        mgr = mod.BatchEmbeddingManager(embedder, batch_size=3)
        result = mgr.embed_many(["a", "b", "c", "d", "e"])
        assert len(result) == 5
        # Should have been called twice: batch of 3 + batch of 2
        assert embedder.embed_batch.call_count == 2

    def test_embed_many_empty(self):
        mod = _load_embeddings()
        embedder = self._make_embedder(mod)
        mgr = mod.BatchEmbeddingManager(embedder, batch_size=5)
        result = mgr.embed_many([])
        assert result == []

    def test_embed_batch_error_returns_nones(self):
        mod = _load_embeddings()
        embedder = self._make_embedder(mod)
        embedder.embed_batch.side_effect = RuntimeError("API down")
        mgr = mod.BatchEmbeddingManager(embedder, batch_size=2)
        mgr.add("a")
        result = mgr.add("b")  # triggers flush
        assert result == [None, None]
        assert mgr.stats["total_errors"] == 1

    def test_callback_error_does_not_propagate(self):
        mod = _load_embeddings()
        embedder = self._make_embedder(mod)
        mgr = mod.BatchEmbeddingManager(embedder, batch_size=1)
        def bad_cb(emb):
            raise ValueError("cb crash")
        # Should not raise
        mgr.add("x", callback=bad_cb)
        assert mgr.stats["total_batches"] == 1


# ============================================================================
# Goal 3 (continued): Connection Pool Integration
# ============================================================================

class TestPoolIntegrationModuleStructure:
    """Module-level checks for pool_integration.py."""

    def test_ast_validity(self):
        with open(POOL_INTEGRATION_PATH) as f:
            tree = ast.parse(f.read())
        assert tree is not None

    def test_checkpoint_before_apply(self):
        mod = _load_pool_integration()
        assert mod.checkpoint_before_apply is True

    def test_feature_available(self):
        mod = _load_pool_integration()
        assert mod.FEATURE_AVAILABLE is True


class TestFallbackPool:
    """FallbackPool when ConnectionPool is unavailable."""

    def test_creation(self):
        mod = _load_pool_integration()
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            pool = mod.FallbackPool(f.name)
            assert pool.db_path == f.name
            assert pool.checkout_count == 0

    def test_connection_context_manager(self):
        mod = _load_pool_integration()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            pool = mod.FallbackPool(path)
            with pool.connection() as conn:
                conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
                conn.execute("INSERT INTO t VALUES (1)")
                conn.commit()
            assert pool.checkout_count == 1

            # Verify data persists
            with pool.connection() as conn:
                rows = conn.execute("SELECT * FROM t").fetchall()
                assert len(rows) == 1
            assert pool.checkout_count == 2
        finally:
            os.unlink(path)

    def test_wal_mode_enforced(self):
        mod = _load_pool_integration()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            pool = mod.FallbackPool(path)
            with pool.connection() as conn:
                mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
                assert mode == "wal"
        finally:
            os.unlink(path)

    def test_close_is_noop(self):
        mod = _load_pool_integration()
        pool = mod.FallbackPool(":memory:")
        pool.close()  # should not raise

    def test_stats(self):
        mod = _load_pool_integration()
        pool = mod.FallbackPool("/tmp/test.db")
        s = pool.stats()
        assert s["type"] == "fallback"
        assert s["checkout_count"] == 0


class TestPoolRegistry:
    """get_rag_pool / close / list / reset."""

    def test_get_creates_pool(self):
        mod = _load_pool_integration()
        mod.reset_rag_pools()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            pool = mod.get_rag_pool(path)
            assert pool is not None
            assert len(mod.list_rag_pools()) == 1
        finally:
            mod.reset_rag_pools()
            os.unlink(path)

    def test_same_path_returns_same_pool(self):
        mod = _load_pool_integration()
        mod.reset_rag_pools()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            p1 = mod.get_rag_pool(path)
            p2 = mod.get_rag_pool(path)
            assert p1 is p2
        finally:
            mod.reset_rag_pools()
            os.unlink(path)

    def test_close_rag_pool(self):
        mod = _load_pool_integration()
        mod.reset_rag_pools()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            mod.get_rag_pool(path)
            assert mod.close_rag_pool(path) is True
            assert mod.close_rag_pool(path) is False  # already closed
            assert len(mod.list_rag_pools()) == 0
        finally:
            mod.reset_rag_pools()
            os.unlink(path)

    def test_close_all_rag_pools(self):
        mod = _load_pool_integration()
        mod.reset_rag_pools()
        with tempfile.TemporaryDirectory() as td:
            for i in range(3):
                p = os.path.join(td, f"db{i}.db")
                open(p, "w").close()
                mod.get_rag_pool(p)
            assert len(mod.list_rag_pools()) == 3
            count = mod.close_all_rag_pools()
            assert count == 3
            assert len(mod.list_rag_pools()) == 0

    def test_reset_rag_pools(self):
        mod = _load_pool_integration()
        mod.reset_rag_pools()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            mod.get_rag_pool(path)
            mod.reset_rag_pools()
            assert len(mod.list_rag_pools()) == 0
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


class TestPooledConnection:
    """pooled_connection and rag_connection context managers."""

    def test_pooled_connection_with_fallback(self):
        mod = _load_pool_integration()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            pool = mod.FallbackPool(path)
            with mod.pooled_connection(pool) as conn:
                conn.execute("SELECT 1")
        finally:
            os.unlink(path)

    def test_rag_connection_shortcut(self):
        mod = _load_pool_integration()
        mod.reset_rag_pools()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            with mod.rag_connection(path) as conn:
                conn.execute("CREATE TABLE x (id INTEGER)")
                conn.commit()
            # Pool should have been created
            assert len(mod.list_rag_pools()) == 1
        finally:
            mod.reset_rag_pools()
            os.unlink(path)

    def test_pooled_connection_unsupported_raises(self):
        mod = _load_pool_integration()
        with pytest.raises(TypeError, match="Unsupported"):
            with mod.pooled_connection("not_a_pool") as conn:
                pass


# ============================================================================
# Goal 4: Cross-cutting checks
# ============================================================================

class TestAllModulesAST:
    """AST validity for all new/modified files."""

    @pytest.mark.parametrize("path", [
        ASYNC_PLUGIN_PATH,
        PARALLEL_INGEST_PATH,
        EMBEDDINGS_PATH,
        POOL_INTEGRATION_PATH,
    ])
    def test_ast_parse(self, path):
        with open(path) as f:
            tree = ast.parse(f.read())
        assert tree is not None

    @pytest.mark.parametrize("path", [
        ASYNC_PLUGIN_PATH,
        PARALLEL_INGEST_PATH,
        POOL_INTEGRATION_PATH,
    ])
    def test_no_hardcoded_hex_colors(self, path):
        import re
        with open(path) as f:
            content = f.read()
        # Match #rgb, #rrggbb, #rrggbbaa patterns (not in comments about HMAC/hex)
        hex_colors = re.findall(r'["\']#[0-9a-fA-F]{3,8}["\']', content)
        assert hex_colors == [], f"Hardcoded colors found: {hex_colors}"


class TestCheckpointSentinels:
    """Verify checkpoint_before_apply in all new modules."""

    @pytest.mark.parametrize("loader,name", [
        (_load_async_plugin, "async_plugin_subprocess"),
        (_load_parallel_ingest, "parallel_ingest"),
        (_load_pool_integration, "pool_integration"),
    ])
    def test_checkpoint_sentinel(self, loader, name):
        mod = loader()
        assert hasattr(mod, "checkpoint_before_apply"), f"{name} missing sentinel"
        assert mod.checkpoint_before_apply is True, f"{name} sentinel not True"


class TestFeatureFlags:
    """Verify FEATURE_AVAILABLE in all new modules."""

    @pytest.mark.parametrize("loader,name", [
        (_load_async_plugin, "async_plugin_subprocess"),
        (_load_parallel_ingest, "parallel_ingest"),
        (_load_pool_integration, "pool_integration"),
    ])
    def test_feature_flag(self, loader, name):
        mod = loader()
        assert hasattr(mod, "FEATURE_AVAILABLE"), f"{name} missing flag"
        assert mod.FEATURE_AVAILABLE is True, f"{name} flag not True"
