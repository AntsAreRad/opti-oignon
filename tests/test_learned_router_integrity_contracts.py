#!/usr/bin/env python3
"""Contracts for the learned router integrity and fallback invariants.

The learned router accumulates real usage as training data, persists a fitted
classifier to disk, and may override the heuristic task type. Feeding it from
usage (and, later, from feedback) is only safe if the persisted model can
never be loaded blind, the store is opened and written safely, the fallback is
deterministic, and the store cannot grow without bound. These contracts pin
those guard-rails without pinning anything the model learns.

  * L1 -- a persisted model is loaded only behind a valid keyed MAC. With no
    key, no sidecar, or a tampered artifact the verification fails safe and
    the artifact is never deserialized; a matching MAC lets the load proceed.
  * L2 -- the training store is opened through the encrypted-connection helper
    and written with parameterized queries: a label carrying a SQL-control
    payload is stored verbatim and the table survives.
  * L3 -- the fallback is deterministic: disabled, untrained, or below the
    confidence threshold all yield the heuristic task type; only an enabled,
    trained, above-threshold prediction overrides it.
  * L4 -- the store is bounded: once it passes the configured ceiling, older
    samples are pruned so accumulated logging cannot grow it without bound.

Local-only (the public distribution ships no tests). Runs under pytest or the
__main__ runner. Loading follows the sibling-harness idiom: the real module is
loaded under a stand-in package with a counting encrypted-connection helper
and a controllable key source, so the integrity paths can be driven without a
real key store and without training a real classifier.
"""

import importlib.util
import sqlite3
import sys
import tempfile
import traceback
import types
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

# Counts calls to the stand-in encrypted-connection helper.
_SAFE_CONNECT_CALLS = {"n": 0}


# ---------------------------------------------------------------------------
# Isolated loading (sibling-harness idiom)
# ---------------------------------------------------------------------------
def _load():
    """Load the real learned_router module under a stand-in package.

    Returns (module, encryption_stub, restore). The encryption stub's ``_key``
    attribute drives the key-present and no-key paths; the connection helper is
    a real sqlite connection wrapped so its use can be observed.
    """
    keys = (
        "opti_oignon", "opti_oignon.db_utils", "opti_oignon.encryption",
        "opti_oignon.learned_router",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    _SAFE_CONNECT_CALLS["n"] = 0
    dbu = types.ModuleType("opti_oignon.db_utils")

    def _counting_connect(path, **kw):
        _SAFE_CONNECT_CALLS["n"] += 1
        return sqlite3.connect(str(path), **kw)

    dbu.safe_connect = _counting_connect
    sys.modules["opti_oignon.db_utils"] = dbu
    pkg.db_utils = dbu

    enc = types.ModuleType("opti_oignon.encryption")
    enc._key = None
    enc.get_encryption_key = lambda: enc._key
    sys.modules["opti_oignon.encryption"] = enc
    pkg.encryption = enc

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.learned_router", _OO / "learned_router.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.learned_router"] = mod
    spec.loader.exec_module(mod)
    pkg.learned_router = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, enc, restore


# ---------------------------------------------------------------------------
# Local material
# ---------------------------------------------------------------------------
class _Key:
    """A master key that exposes raw bytes, as the real key type does."""

    def __init__(self, raw):
        self._raw = raw

    def as_bytes(self):
        return self._raw


class _FakePipeline:
    """A stand-in fitted classifier returning a fixed high-confidence class."""

    classes_ = np.array(["code_python", "debug"])

    def predict_proba(self, queries):
        return np.array([[0.9, 0.1]])


class _FakeJoblib:
    """A stand-in joblib whose load is observable and never runs real pickle."""

    def __init__(self):
        self.load_called = False
        self.payload = _FakePipeline()

    def load(self, path):
        self.load_called = True
        return self.payload

    def dump(self, obj, path):
        Path(path).write_bytes(b"fake-model-bytes")


def _tmp():
    return Path(tempfile.mkdtemp())


# ---------------------------------------------------------------------------
# L1 -- persisted model loaded only behind a valid keyed MAC
# ---------------------------------------------------------------------------
def test_l1_persisted_model_requires_a_valid_mac():
    mod, enc, restore = _load()
    tmp = _tmp()
    model_path = tmp / "learned_router.pkl"
    mac_path = mod._model_mac_path(model_path)
    try:
        model_path.write_bytes(b"pickle-payload")
        key = _Key(b"k" * 32)

        # No key -> verification fails safe.
        enc._key = None
        assert mod.verify_model_mac(
            model_path, mac_path, mod._router_master_key(),
        ) is False

        # Key present but no sidecar -> fails safe.
        enc._key = key
        assert not mac_path.exists()
        assert mod.verify_model_mac(
            model_path, mac_path, mod._router_master_key(),
        ) is False

        # Write a MAC, then tamper the artifact -> mismatch -> fails safe.
        assert mod.write_model_mac(
            model_path, mac_path, mod._router_master_key(),
        ) is True
        assert mod.verify_model_mac(
            model_path, mac_path, mod._router_master_key(),
        ) is True
        model_path.write_bytes(b"pickle-payload-tampered")
        assert mod.verify_model_mac(
            model_path, mac_path, mod._router_master_key(),
        ) is False

        # Content matching the MAC again -> valid.
        model_path.write_bytes(b"pickle-payload")
        assert mod.verify_model_mac(
            model_path, mac_path, mod._router_master_key(),
        ) is True
    finally:
        restore()


def test_l1b_try_load_never_deserializes_without_a_valid_mac():
    mod, enc, restore = _load()
    tmp = _tmp()
    model_path = tmp / "learned_router.pkl"
    mac_path = mod._model_mac_path(model_path)
    try:
        fake = _FakeJoblib()
        mod.SKLEARN_AVAILABLE = True
        mod.joblib = fake

        router = mod.LearnedRouter(
            config_path=tmp / "absent.yaml",
            db_path=tmp / "lr.db",
            model_path=model_path,
        )

        # A model file exists but has no valid MAC and there is no key: the
        # load must be refused and the artifact never handed to joblib.
        enc._key = None
        model_path.write_bytes(b"pickle-payload")
        fake.load_called = False
        assert router._try_load_model() is False
        assert fake.load_called is False, (
            "a model must never be deserialized without a valid MAC"
        )

        # With a key and a matching MAC, the load proceeds.
        enc._key = _Key(b"k" * 32)
        assert mod.write_model_mac(
            model_path, mac_path, mod._router_master_key(),
        ) is True
        fake.load_called = False
        assert router._try_load_model() is True
        assert fake.load_called is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# L2 -- store opened safely and written with parameterized queries
# ---------------------------------------------------------------------------
def test_l2_persistence_uses_safe_connect_and_parameterized_sql():
    mod, enc, restore = _load()
    tmp = _tmp()
    try:
        mod.SKLEARN_AVAILABLE = False  # keep init light, no model load

        _SAFE_CONNECT_CALLS["n"] = 0
        router = mod.LearnedRouter(
            config_path=tmp / "absent.yaml",
            db_path=tmp / "lr.db",
            model_path=tmp / "lr.pkl",
        )
        assert _SAFE_CONNECT_CALLS["n"] > 0, (
            "the training store must be opened via the encrypted helper"
        )

        # A SQL-control payload in the label is stored verbatim, not executed:
        # parameterized queries neutralize it and the table survives.
        payload = "'; DROP TABLE training_samples; --"
        router.log_sample("some query text", payload, source="router")
        assert router.get_sample_count() == 1, (
            "the table must survive a SQL-control payload -- queries are bound"
        )
        dist = router.get_class_distribution()
        assert dist.get(payload) == 1, (
            "the label is persisted literally, proving the query is parameterized"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# L3 -- fallback is deterministic
# ---------------------------------------------------------------------------
def test_l3_fallback_is_deterministic():
    mod, enc, restore = _load()
    tmp = _tmp()
    try:
        mod.SKLEARN_AVAILABLE = True
        router = mod.LearnedRouter(
            config_path=tmp / "absent.yaml",
            db_path=tmp / "lr.db",
            model_path=tmp / "lr.pkl",
        )
        heuristic = "scientific_writing"

        # Disabled -> heuristic, even with a model present.
        router._config["enabled"] = False
        router._pipeline = _FakePipeline()
        pred = router.classify_with_fallback("q", heuristic)
        assert pred.fallback_used is True
        assert pred.task_type == heuristic

        # Enabled but untrained -> heuristic.
        router._config["enabled"] = True
        router._pipeline = None
        pred = router.classify_with_fallback("q", heuristic)
        assert pred.fallback_used is True
        assert pred.task_type == heuristic

        # Enabled and trained but below threshold -> heuristic kept.
        router._pipeline = _FakePipeline()
        router._config["confidence_threshold"] = 0.99
        pred = router.classify_with_fallback("q", heuristic)
        assert pred.fallback_used is True
        assert pred.task_type == heuristic

        # Enabled, trained, above threshold -> the model label wins.
        router._config["confidence_threshold"] = 0.5
        pred = router.classify_with_fallback("q", heuristic)
        assert pred.fallback_used is False
        assert pred.task_type == "code_python"
    finally:
        restore()


# ---------------------------------------------------------------------------
# L4 -- store growth is bounded by prune
# ---------------------------------------------------------------------------
def test_l4_store_growth_is_bounded_by_prune():
    mod, enc, restore = _load()
    tmp = _tmp()
    try:
        mod.SKLEARN_AVAILABLE = False
        router = mod.LearnedRouter(
            config_path=tmp / "absent.yaml",
            db_path=tmp / "lr.db",
            model_path=tmp / "lr.pkl",
        )
        router._config["max_stored_samples"] = 5
        for i in range(20):
            router.log_sample(f"query number {i}", "code_python", source="router")
        count = router.get_sample_count()
        assert count == 5, (
            "the store must be pruned to the configured ceiling"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("L1 model requires valid MAC", test_l1_persisted_model_requires_a_valid_mac),
        ("L1b try_load never blind-loads", test_l1b_try_load_never_deserializes_without_a_valid_mac),
        ("L2 safe_connect + parameterized SQL", test_l2_persistence_uses_safe_connect_and_parameterized_sql),
        ("L3 fallback is deterministic", test_l3_fallback_is_deterministic),
        ("L4 store growth bounded by prune", test_l4_store_growth_is_bounded_by_prune),
    ]
    passed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
