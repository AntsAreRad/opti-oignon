#!/usr/bin/env python3
"""
S193 F6e — caching layer fixes.

Covers:
  - TC-04: context-fingerprint gating in the semantic cache (exact tier,
    semantic tier, legacy S23 path) + executor wiring (source assertions)
  - PCH-01: pre-cache handles both dict-form and object-form ollama responses
  - CPL-01: pool bound holds (reserved-slot creation, failure releases slot)
  - CPL-02: rollback on checkin (no dirty transaction returns to the pool)
  - CPL-03: per-checkout health is a liveness probe, not an integrity scan
"""

import importlib.util
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_PROJECT = Path(__file__).resolve().parent.parent


def _load_module(name: str, rel_path: str):
    full = _PROJECT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(full))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_pkg_module(pkg: str, sub: str, rel_path: str):
    """Load a module that does relative imports (e.g. ``from .config import``).

    Pre-seeds a stub parent package and a stub ``config`` submodule exposing
    DATA_DIR, per the established S193 loader idiom.
    """
    import types
    if pkg not in sys.modules:
        pkg_mod = types.ModuleType(pkg)
        pkg_mod.__path__ = [str(_PROJECT / "opti_oignon")]
        sys.modules[pkg] = pkg_mod
        cfg = types.ModuleType(f"{pkg}.config")
        cfg.DATA_DIR = Path(tempfile.mkdtemp(prefix="s193e_data_"))
        sys.modules[f"{pkg}.config"] = cfg
    full = _PROJECT / rel_path
    name = f"{pkg}.{sub}"
    spec = importlib.util.spec_from_file_location(name, str(full))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_sc_mod = _load_pkg_module("s193e_pkg", "semantic_cache", "opti_oignon/semantic_cache.py")
_pc_mod = _load_module("s193e_pre_cache", "opti_oignon/pre_cache.py")
_cp_mod = _load_module("s193e_connection_pool", "opti_oignon/connection_pool.py")


def _cache(tmp):
    c = _sc_mod.SemanticCache(
        db_path=Path(tmp) / "sem.db",
        config_path=Path(tmp) / "none.yaml",
    )
    c.enabled = True
    return c


# ---------------------------------------------------------------------------
# TC-04 — exact tier fingerprint gating
# ---------------------------------------------------------------------------

class TestTC04ExactTier:
    def test_same_fingerprint_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = _cache(tmp)
            c.put("q1", "r1", model="m", conversation_id="cv",
                  context_fingerprint="FP_A")
            e = c.get("q1", conversation_id="cv", model="m",
                      context_fingerprint="FP_A")
            assert e is not None and e.response == "r1"

    def test_different_fingerprint_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = _cache(tmp)
            c.put("q1", "r1", model="m", conversation_id="cv",
                  context_fingerprint="FP_A")
            # Identical query, same conversation, same model -- but the
            # generation context changed: must MISS (this was the stale-serve).
            assert c.get("q1", conversation_id="cv", model="m",
                         context_fingerprint="FP_B") is None

    def test_legacy_entry_never_matches_fingerprinted_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = _cache(tmp)
            c.put("q1", "r1", model="m", conversation_id="cv")  # no fp stored
            assert c.get("q1", conversation_id="cv", model="m",
                         context_fingerprint="FP_A") is None

    def test_unfingerprinted_request_keeps_legacy_behaviour(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = _cache(tmp)
            c.put("q1", "r1", model="m", conversation_id="cv",
                  context_fingerprint="FP_A")
            # Callers that pass no fingerprint behave exactly as before.
            e = c.get("q1", conversation_id="cv", model="m")
            assert e is not None and e.response == "r1"


# ---------------------------------------------------------------------------
# TC-04 — semantic tier fingerprint gating (stubbed embeddings)
# ---------------------------------------------------------------------------

class TestTC04SemanticTier:
    def _stub_embeddings(self, monkeypatch):
        monkeypatch.setattr(
            _sc_mod, "_get_embedding", lambda text, model="": [1.0, 0.0, 0.0]
        )

    def test_semantic_match_respects_fingerprint(self, monkeypatch):
        self._stub_embeddings(monkeypatch)
        with tempfile.TemporaryDirectory() as tmp:
            c = _cache(tmp)
            c.embeddings_available = True
            c.put("alpha beta gamma", "resp-A", model="m",
                  conversation_id="cv", context_fingerprint="FP_A")
            # Different wording -> exact tier misses, semantic tier evaluates.
            hit = c.get("alpha beta gamma please", conversation_id="cv",
                        model="m", context_fingerprint="FP_A")
            assert hit is not None and hit.match_type == "semantic"
            miss = c.get("alpha beta gamma please", conversation_id="cv",
                         model="m", context_fingerprint="FP_B")
            assert miss is None


# ---------------------------------------------------------------------------
# TC-04 — legacy S23 path fingerprint gating
# ---------------------------------------------------------------------------

class TestTC04LegacyPath:
    def test_find_similar_by_embedding_filters_on_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = _cache(tmp)
            ok = c.store_embedding(
                cache_key="k1", model="m", query_text="q",
                embedding=[1.0, 0.0], context_fingerprint="FP_A",
            )
            assert ok
            m_same = c.find_similar_by_embedding(
                [1.0, 0.0], "m", context_fingerprint="FP_A",
            )
            assert m_same is not None and m_same.cache_key == "k1"
            m_other = c.find_similar_by_embedding(
                [1.0, 0.0], "m", context_fingerprint="FP_B",
            )
            assert m_other is None
            # Unfingerprinted request keeps the legacy behaviour.
            m_any = c.find_similar_by_embedding([1.0, 0.0], "m")
            assert m_any is not None

    def test_legacy_rows_excluded_when_fingerprint_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = _cache(tmp)
            c.store_embedding(
                cache_key="k0", model="m", query_text="q", embedding=[1.0, 0.0],
            )  # stored without fingerprint ('' in column)
            assert c.find_similar_by_embedding(
                [1.0, 0.0], "m", context_fingerprint="FP_A",
            ) is None


# ---------------------------------------------------------------------------
# TC-04 — executor wiring (source assertions; executor imports ollama)
# ---------------------------------------------------------------------------

class TestTC04ExecutorWiring:
    def test_executor_computes_and_passes_fingerprint(self):
        src = (_PROJECT / "opti_oignon/executor.py").read_text()
        assert "import hashlib" in src
        assert '_ctx_fp = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()' in src
        assert src.count("context_fingerprint=_ctx_fp") == 4
        assert src.count("context_fingerprint=_CTX_FP_NOCTX") == 4
        assert "_CTX_FP_NOCTX = hashlib.sha256" in src


# ---------------------------------------------------------------------------
# PCH-01 — pre-cache both-form response parse
# ---------------------------------------------------------------------------

class _FakeSemCache:
    def __init__(self):
        self.puts = []

    def get(self, query, model=""):
        return None

    def put(self, query, response, model="", metadata=None,
            conversation_id=None, context_fingerprint=""):
        self.puts.append((query, response, model))
        return "hash"


def _precache_with_one_query(tmp, cache):
    cfg = Path(tmp) / "pc.yaml"
    cfg.write_text(
        "enabled: true\n"
        "queries:\n"
        "  - {query: 'Hello there', task_type: general, model: 'm1'}\n",
        encoding="utf-8",
    )
    return _pc_mod.PreCache(config_path=cfg, cache=cache)


class TestPCH01BothFormParse:
    def test_object_form_response(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _FakeSemCache()
            pc = _precache_with_one_query(tmp, fake)
            stub = SimpleNamespace(chat=lambda **kw: SimpleNamespace(
                message=SimpleNamespace(content="obj-resp")
            ))
            monkeypatch.setattr(_pc_mod, "_ollama_module", stub)
            monkeypatch.setattr(_pc_mod, "OLLAMA_AVAILABLE", True)
            res = pc.warm_common_queries()
            assert res.cached == 1 and res.failed == 0
            assert fake.puts[0][1] == "obj-resp"

    def test_dict_form_response(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _FakeSemCache()
            pc = _precache_with_one_query(tmp, fake)
            stub = SimpleNamespace(
                chat=lambda **kw: {"message": {"content": "dict-resp"}}
            )
            monkeypatch.setattr(_pc_mod, "_ollama_module", stub)
            monkeypatch.setattr(_pc_mod, "OLLAMA_AVAILABLE", True)
            res = pc.warm_common_queries()
            assert res.cached == 1 and res.failed == 0
            assert fake.puts[0][1] == "dict-resp"


# ---------------------------------------------------------------------------
# CPL — connection pool
# ---------------------------------------------------------------------------

class TestCPL01PoolBound:
    def test_bound_holds_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = _cp_mod.ConnectionPool(
                os.path.join(tmp, "p.db"), pool_size=1,
                checkout_timeout=0.1, health_check=True,
            )
            c1 = pool.checkout()
            with pytest.raises(TimeoutError):
                pool.checkout()
            assert pool.stats.created == 1
            pool.checkin(c1)
            pool.close()

    def test_creation_failure_releases_slot(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            pool = _cp_mod.ConnectionPool(
                os.path.join(tmp, "p.db"), pool_size=1, checkout_timeout=0.1,
            )
            def boom(*a, **kw):
                raise RuntimeError("factory down")
            monkeypatch.setattr(_cp_mod, "_create_connection", boom)
            with pytest.raises(RuntimeError):
                pool.checkout()
            monkeypatch.undo()
            # Slot released: a later checkout succeeds.
            conn = pool.checkout()
            assert conn is not None
            pool.checkin(conn)
            pool.close()

    def test_concurrent_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = _cp_mod.ConnectionPool(
                os.path.join(tmp, "p.db"), pool_size=2,
                checkout_timeout=5.0, health_check=False,
            )
            errors = []

            def worker():
                try:
                    for _ in range(20):
                        with pool.connection() as conn:
                            conn.execute("SELECT 1")
                except Exception as e:  # pragma: no cover
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert not errors
            # The reserve-under-lock fix makes this a hard invariant.
            assert pool.stats.created <= 2
            pool.close()


class TestCPL02RollbackOnCheckin:
    def test_open_transaction_rolled_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = _cp_mod.ConnectionPool(
                os.path.join(tmp, "p.db"), pool_size=1, health_check=False,
            )
            c = pool.checkout()
            c.execute("CREATE TABLE t (x INTEGER)")
            c.commit()
            c.execute("INSERT INTO t VALUES (1)")  # opens a transaction
            assert c.in_transaction
            pool.checkin(c)
            c2 = pool.checkout()  # same pooled connection
            assert not c2.in_transaction
            row = c2.execute("SELECT COUNT(*) FROM t").fetchone()
            assert row[0] == 0  # the dangling insert was rolled back
            pool.checkin(c2)
            pool.close()


class TestCPL03LivenessProbe:
    def test_health_is_select_one(self):
        src = (_PROJECT / "opti_oignon/connection_pool.py").read_text()
        assert 'pc.conn.execute("SELECT 1")' in src
        assert 'execute("PRAGMA integrity_check")' not in src

    def test_checkout_with_health_check_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = _cp_mod.ConnectionPool(
                os.path.join(tmp, "p.db"), pool_size=1, health_check=True,
            )
            conn = pool.checkout()
            assert conn.execute("SELECT 1").fetchone()[0] == 1
            pool.checkin(conn)
            pool.close()
