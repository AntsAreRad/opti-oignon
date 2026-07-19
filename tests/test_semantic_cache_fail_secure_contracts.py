#!/usr/bin/env python3
"""What the semantic cache promises when its storage layer is not there.

Every byte the cache persists is a user query or a model response, so the
cache is only allowed to talk to disk through the project's hardened
connection layer. These contracts pin the whole posture around that rule,
from both directions.

No plaintext fallback, ever. With the connection layer genuinely absent
the module still imports, but the cache is honestly unavailable: lookups
miss, writes report failure, and NOTHING is created on disk -- no database
file, no sidecar, no plaintext copy of a query or a response anywhere
under the data directory. A cache that quietly downgrades to a bare
connection when the hardened layer is missing is a data leak wearing a
cache's name, and an abstract-syntax census closes the refactor door: the
module may not contain a single call site that connects through the
standard sqlite3 module directly.

Refusal is survivable. When the layer refuses the connection -- the
enforced-encryption posture without a working cipher, a key that does not
verify, a file the layer cannot read -- importing the module still
succeeds, constructing an instance still succeeds, and every public
operation comes back with its benign value instead of an exception: a
lookup is a miss and is counted as one, a write reports nothing stored,
counts are zero, statistics carry the in-memory session figures with an
empty store. The failure is logged loudly once per instance and quietly
after that, so an operator sees it without the log drowning in it. The
isolation posture keeps its own refusals: an unscoped lookup or write in
the isolated mode fails closed before the storage layer is ever asked.

The file outlives the failure. A database the layer cannot open is left
byte-for-byte intact -- recovery is an explicit operator decision, never a
side effect of a failed lookup. And the gate does not castrate the cache:
with a working layer seeded, a stored answer round-trips, so the degraded
paths above are degradations and not the only behaviour there is.

Loaded through the shared isolation window. The connection layer is the
seam: per contract it is blocked, seeded to refuse, seeded broken at the
statement level, or seeded working; the configuration module is stood in
with a throwaway data directory. No real database under the repository,
no model backend and no network is ever reached.
"""

import hashlib
import logging
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.semantic_cache"
_MODULE_PATH = source("semantic_cache.py")


class _LayerSeam:
    """A stand-in connection layer whose behaviour a contract can steer.

    ``behaviour`` is called with the requested path and keyword arguments and
    may return a connection or raise; every call is counted so a contract can
    assert the layer was (or was not) consulted.
    """

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = 0

    def safe_connect(self, path, **kwargs):
        self.calls += 1
        return self.behaviour(path, **kwargs)


class _BrokenConnection:
    """A connection that fails at the first statement, like an unreadable file."""

    def __init__(self):
        self.row_factory = None

    def execute(self, *args, **kwargs):
        raise sqlite3.DatabaseError("file is not a database")

    def commit(self):  # pragma: no cover - never reached after a failed execute
        return None

    def close(self):
        return None


def _refusing(exc_type=RuntimeError, message="encrypted connection required"):
    def _behaviour(path, **kwargs):
        raise exc_type(message)

    return _behaviour


def _working(path, **kwargs):
    kwargs.pop("check_same_thread", None)
    return sqlite3.connect(str(path), check_same_thread=False)


def _db_utils_stub(seam):
    mod = types.ModuleType("opti_oignon.db_utils")
    mod.safe_connect = seam.safe_connect
    return mod


def _load(*, seam=None, layer_absent=False):
    """Load the real module with the connection layer per the posture.

    seam         -- a ``_LayerSeam`` seeded behind ``safe_connect``.
    layer_absent -- block the layer entirely and PROVE it unreachable.
    """
    data_dir = Path(tempfile.mkdtemp(prefix="cache_data_"))
    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = data_dir

    seeded = {"opti_oignon.config": cfg}
    blocked = []
    if layer_absent:
        blocked.append("opti_oignon.db_utils")
    elif seam is not None:
        seeded["opti_oignon.db_utils"] = _db_utils_stub(seam)

    loaded, restore = isolate(
        targets={_TARGET: _MODULE_PATH},
        blocked=blocked,
        seeded=seeded,
    )
    return loaded[_TARGET], data_dir, restore


def _fresh_cache(module, db_dir, **switches):
    """Build an enabled cache on a private database with explicit switches."""
    cache = module.SemanticCache(
        db_path=db_dir / "cache_under_contract.db",
        config_path=db_dir / "absent.yaml",
        ttl_seconds=switches.pop("ttl", 3600),
        max_entries=switches.pop("max_entries", 100),
        scope=switches.pop("scope", "conversation"),
    )
    cache._config["enabled"] = True
    cache._config["exact_match_enabled"] = switches.pop("exact", True)
    cache._config["semantic_match_enabled"] = switches.pop("semantic", False)
    cache.embeddings_available = switches.pop("embeddings", False)
    return cache


# ---------------------------------------------------------------------------
# d1 -- with the layer absent the module imports and the cache is unavailable
# ---------------------------------------------------------------------------

def test_d1_layer_absent_module_imports_and_operations_are_benign():
    module, data_dir, restore = _load(layer_absent=True)
    try:
        module._is_bulbe = lambda: False
        db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
        cache = _fresh_cache(module, db_dir)

        assert cache.put("who wrote this?", "SECRET ANSWER", conversation_id="c1") == ""
        assert cache.get("who wrote this?", conversation_id="c1") is None
        assert cache.entry_count() == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# d2 -- with the layer absent nothing at all lands on disk
# ---------------------------------------------------------------------------

def test_d2_layer_absent_nothing_is_written_to_disk():
    module, data_dir, restore = _load(layer_absent=True)
    try:
        module._is_bulbe = lambda: False
        db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
        cache = _fresh_cache(module, db_dir)
        cache.put("what is the launch code?", "TOPSECRETMARKER", conversation_id="c1")
        cache.store_embedding("key-1", "model-a", "TOPSECRETMARKER", [1.0, 0.0])

        created = [p for p in db_dir.rglob("*") if p.is_file() and p.name != "absent.yaml"]
        assert created == [], "an unavailable cache must not create files"
        for root in (db_dir, data_dir):
            for path in root.rglob("*"):
                if path.is_file():
                    assert b"TOPSECRETMARKER" not in path.read_bytes(), (
                        "cached content must never be written outside the layer"
                    )
    finally:
        restore()


# ---------------------------------------------------------------------------
# d3 -- a layer that refuses the connection cannot break the import
# ---------------------------------------------------------------------------

def test_d3_refusing_layer_leaves_the_import_and_singleton_alive():
    seam = _LayerSeam(_refusing())
    module, data_dir, restore = _load(seam=seam)
    try:
        # The module executed to the end: the shared instance exists and the
        # construction consulted the layer instead of assuming it works.
        assert module.semantic_cache is not None
        assert seam.calls >= 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# d4 -- every read and count is benign while the layer refuses
# ---------------------------------------------------------------------------

def test_d4_reads_and_counts_are_benign_while_the_layer_refuses():
    seam = _LayerSeam(_refusing())
    module, data_dir, restore = _load(seam=seam)
    try:
        module._is_bulbe = lambda: False
        db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
        cache = _fresh_cache(module, db_dir)

        assert cache.get("anything cached?", conversation_id="c1") is None
        assert cache.entry_count() == 0
        assert cache.embedding_count() == 0
        assert cache.find_similar_by_embedding([1.0, 0.0], "model-a") is None
        stats = cache.get_stats()
        assert stats.total_entries == 0
        assert stats.enabled is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# d5 -- every write and maintenance operation is benign while the layer refuses
# ---------------------------------------------------------------------------

def test_d5_writes_and_maintenance_are_benign_while_the_layer_refuses():
    seam = _LayerSeam(_refusing())
    module, data_dir, restore = _load(seam=seam)
    try:
        module._is_bulbe = lambda: False
        db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
        cache = _fresh_cache(module, db_dir)

        assert cache.put("q", "a", conversation_id="c1") == ""
        assert cache.invalidate("c1") == 0
        assert cache.invalidate() == 0
        assert cache.expire_stale() == 0
        assert cache.clear() == 0
        assert cache.store_embedding("key-1", "model-a", "q", [1.0, 0.0]) is False
        assert cache.remove_embedding("key-1") is False
        assert cache.remove_embeddings_for_model("model-a") == 0
        orphan_source = types.SimpleNamespace(get=lambda key: None)
        assert cache.cleanup_orphans(orphan_source) == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# d6 -- a refused lookup is a counted miss, not a silent nothing
# ---------------------------------------------------------------------------

def test_d6_refused_lookup_is_counted_as_a_miss():
    seam = _LayerSeam(_refusing())
    module, data_dir, restore = _load(seam=seam)
    try:
        module._is_bulbe = lambda: False
        db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
        cache = _fresh_cache(module, db_dir)

        before = cache.get_stats().total_misses
        assert cache.get("was this cached?", conversation_id="c1") is None
        after = cache.get_stats().total_misses
        assert after == before + 1, "a degraded lookup must still count as a miss"
    finally:
        restore()


# ---------------------------------------------------------------------------
# d7 -- the failure is logged loudly once per instance, quietly afterwards
# ---------------------------------------------------------------------------

def test_d7_layer_failure_logs_one_warning_then_stays_quiet(caplog):
    seam = _LayerSeam(_refusing())
    module, data_dir, restore = _load(seam=seam)
    try:
        module._is_bulbe = lambda: False
        db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
        # The module-level shared instance already warned for itself during
        # the load above; the once-per-instance promise is pinned on a fresh
        # instance, so earlier records are cleared first.
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=module.logger.name):
            cache = _fresh_cache(module, db_dir)
            for _ in range(3):
                cache.get("still there?", conversation_id="c1")
                cache.put("still there?", "answer", conversation_id="c1")
        warnings = [
            rec for rec in caplog.records
            if rec.levelno == logging.WARNING and "unavailable" in rec.getMessage()
        ]
        assert len(warnings) == 1, "the layer failure must be warned exactly once"
        quiet = [
            rec for rec in caplog.records
            if rec.levelno == logging.DEBUG and "still unavailable" in rec.getMessage()
        ]
        assert quiet, "later failures must drop to the quiet level"
    finally:
        restore()


# ---------------------------------------------------------------------------
# d8 -- the isolated mode still fails closed before the layer is consulted
# ---------------------------------------------------------------------------

def test_d8_isolated_mode_refuses_unscoped_traffic_before_the_layer():
    seam = _LayerSeam(_refusing())
    module, data_dir, restore = _load(seam=seam)
    try:
        db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
        cache = _fresh_cache(module, db_dir, scope="global")
        module._is_bulbe = lambda: True

        consulted = seam.calls
        assert cache.get("unscoped question", conversation_id=None) is None
        assert cache.put("unscoped question", "answer", conversation_id=None) == ""
        assert seam.calls == consulted, (
            "an unscoped request in the isolated mode must be refused before "
            "the storage layer is asked"
        )
        # A scoped request may reach the layer; the refusal there stays benign.
        assert cache.get("scoped question", conversation_id="c1") is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# d9 -- a database the layer cannot open is left byte-for-byte intact
# ---------------------------------------------------------------------------

def test_d9_an_unreadable_database_file_is_never_deleted_or_rewritten():
    seam = _LayerSeam(_refusing(message="key verification failed"))
    module, data_dir, restore = _load(seam=seam)
    try:
        module._is_bulbe = lambda: False
        db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
        db_path = db_dir / "cache_under_contract.db"

        plain = sqlite3.connect(str(db_path))
        plain.execute("CREATE TABLE pre_existing (payload TEXT)")
        plain.execute("INSERT INTO pre_existing VALUES ('kept')")
        plain.commit()
        plain.close()
        before = hashlib.md5(db_path.read_bytes()).hexdigest()

        cache = module.SemanticCache(
            db_path=db_path, config_path=db_dir / "absent.yaml"
        )
        cache._config["enabled"] = True
        cache.get("anything?", conversation_id="c1")
        cache.put("anything?", "answer", conversation_id="c1")
        cache.expire_stale()
        cache.clear()

        assert db_path.exists(), "a degraded cache must never delete its file"
        after = hashlib.md5(db_path.read_bytes()).hexdigest()
        assert after == before, "a degraded cache must never rewrite its file"
    finally:
        restore()


# ---------------------------------------------------------------------------
# d10 -- a connection that fails at the statement level degrades to a miss
# ---------------------------------------------------------------------------

def test_d10_statement_level_failure_degrades_to_a_miss():
    seam = _LayerSeam(lambda path, **kw: _BrokenConnection())
    module, data_dir, restore = _load(seam=seam)
    try:
        module._is_bulbe = lambda: False
        db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
        cache = _fresh_cache(module, db_dir)

        before = cache.get_stats().total_misses
        assert cache.get("readable?", conversation_id="c1") is None
        assert cache.get_stats().total_misses == before + 1
        assert cache.put("readable?", "answer", conversation_id="c1") == ""
        assert cache.entry_count() == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# d11 -- control: with a working layer a stored answer still round-trips
# ---------------------------------------------------------------------------

def test_d11_working_layer_round_trips_a_stored_answer():
    seam = _LayerSeam(_working)
    module, data_dir, restore = _load(seam=seam)
    try:
        module._is_bulbe = lambda: False
        db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
        cache = _fresh_cache(module, db_dir)

        key = cache.put("what round-trips?", "THE ANSWER", conversation_id="c1")
        assert key != "", "a working layer must store the entry"
        served = cache.get("what round-trips?", conversation_id="c1")
        assert served is not None and served.response == "THE ANSWER"
        assert seam.calls >= 2, "storage traffic must cross the seeded layer"
    finally:
        restore()


# ---------------------------------------------------------------------------
# d12 -- a refused construction still yields a usable degraded instance
# ---------------------------------------------------------------------------

def test_d12_refused_construction_yields_a_usable_degraded_instance():
    seam = _LayerSeam(_refusing())
    module, data_dir, restore = _load(seam=seam)
    try:
        module._is_bulbe = lambda: False
        db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
        cache = _fresh_cache(module, db_dir)

        # No half-initialised crash on any later call.
        assert cache.get("usable?", conversation_id="c1") is None
        assert cache.put("usable?", "answer", conversation_id="c1") == ""
        assert cache.get_stats().total_entries == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# d13 -- refusal, filesystem and database errors are all contained
# ---------------------------------------------------------------------------

def test_d13_runtime_os_and_database_errors_are_all_contained():
    seam = _LayerSeam(_refusing(RuntimeError))
    module, data_dir, restore = _load(seam=seam)
    try:
        module._is_bulbe = lambda: False
        db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
        cache = _fresh_cache(module, db_dir)

        assert cache.get("q", conversation_id="c1") is None

        seam.behaviour = _refusing(OSError, "read-only file system")
        assert cache.get("q", conversation_id="c1") is None
        assert cache.put("q", "a", conversation_id="c1") == ""

        seam.behaviour = _refusing(sqlite3.DatabaseError, "malformed")
        assert cache.get("q", conversation_id="c1") is None
        assert cache.invalidate() == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# d14 -- census: no call site connects through the standard sqlite3 module
# ---------------------------------------------------------------------------

def test_d14_no_direct_sqlite3_connect_call_site_in_the_module():
    import ast

    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    sqlite_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sqlite3":
                    sqlite_aliases.add(alias.asname or alias.name)

    direct_connects = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in sqlite_aliases
    ]
    assert direct_connects == [], (
        "the cache must never open a connection outside the hardened layer"
    )
