#!/usr/bin/env python3
"""What the deferred-import module promises on its own account.

Two proxies live here and they do not fail the same way, which is the sharpest
thing this suite pins. The module-level proxy wraps any import failure in a
fresh ImportError that names the module, keeps the ORIGINAL exception as its
cause and as the recorded error, and then keeps trying: repair the module and
the next touch loads it. The attribute-level proxy remembers an import or
attribute failure forever and re-raises the very same exception object on
every later touch -- but it only remembers those two kinds; any other error
raised while resolving escapes unrecorded, and the next touch simply tries
again. An attribute that resolves to None is never recorded either, so every
touch resolves it anew and each delegation fails afresh.

Around the proxies, the module keeps one registry: the factory hands back the
same proxy for the same name and a different one for a different name, the
compatibility alias IS the factory, the statistics snapshot reports load
state, load time and the error text per registered name, eager preloading maps
each name to a plain success flag, and the human report names each state and
totals the loads -- or says plainly that nothing is registered. The roster of
modules considered heavy is a frozen list and is pinned as such. Background
preloading runs in a daemon thread with a fixed name and shields itself from a
callback that blows up.

Every module a proxy is pointed at is seeded or declared unreachable and
proven so under the shared window; nothing outside the window is ever
imported.
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.lazy_loader"
_TARGET_SOURCE = source("lazy_loader.py")

_LEAF = "opti_oignon.sample_leaf"
_OTHER = "opti_oignon.other_leaf"

_HEAVY_CENSUS = (
    "opti_oignon.rag_store", "opti_oignon.rag_hybrid_search", "opti_oignon.rag_external",
    "opti_oignon.rag_dashboard", "opti_oignon.coding_agent", "opti_oignon.coding_history",
    "opti_oignon.telemetry", "opti_oignon.telemetry_history", "opti_oignon.inference_profiler",
    "opti_oignon.performance_benchmark", "opti_oignon.benchmark_evaluator",
    "opti_oignon.benchmark_runner", "opti_oignon.benchmark_judge",
    "opti_oignon.benchmark_recommendations", "opti_oignon.benchmark_auto_trigger",
    "opti_oignon.benchmark_custom_profiles", "opti_oignon.fine_tune_export",
    "opti_oignon.fine_tune_tracker", "opti_oignon.plugin_index", "opti_oignon.plugin_installer",
    "opti_oignon.plugin_reviews", "opti_oignon.plugin_template", "opti_oignon.pipeline_manager",
    "opti_oignon.speculative_decoding", "opti_oignon.auto_tuner",
)


def _mod(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _window(*, blocked=(), seeded=None):
    loaded, restore = isolate(
        targets={_TARGET: _TARGET_SOURCE},
        blocked=blocked,
        seeded=dict(seeded or {}),
    )
    return loaded[_TARGET], restore


# ---------------------------------------------------------------------------
# The module-level proxy.


def test_l1_a_module_proxy_defers_the_import_until_first_touch():
    lazy, restore = _window(seeded={_LEAF: _mod(_LEAF, marker="ready")})
    try:
        proxy = lazy.LazyModule(_LEAF)
        assert proxy.is_loaded is False
        assert proxy.load_time == 0.0
        assert proxy.load_error is None
        assert "not loaded" in repr(proxy)
    finally:
        restore()


def test_l2_the_first_attribute_touch_imports_and_delegates():
    lazy, restore = _window(seeded={_LEAF: _mod(_LEAF, marker="ready")})
    try:
        proxy = lazy.LazyModule(_LEAF)
        assert proxy.marker == "ready"
        assert proxy.is_loaded is True
        assert "loaded in" in repr(proxy)
    finally:
        restore()


def test_l3_a_failed_import_is_wrapped_and_the_original_error_is_kept():
    lazy, restore = _window(blocked=(_LEAF,))
    try:
        proxy = lazy.LazyModule(_LEAF)
        with pytest.raises(ImportError) as caught:
            proxy.marker
        assert str(caught.value).startswith(f"Lazy import failed for {_LEAF}")
        assert proxy.load_error is not None
        assert caught.value.__cause__ is proxy.load_error
        assert proxy.is_loaded is False
        assert "FAILED" in repr(proxy)
    finally:
        restore()


def test_l4_a_module_proxy_tries_again_after_a_failure():
    lazy, restore = _window(blocked=(_LEAF,))
    try:
        proxy = lazy.LazyModule(_LEAF)
        with pytest.raises(ImportError):
            proxy.marker
        sys.modules[_LEAF] = _mod(_LEAF, marker="repaired")
        assert proxy.marker == "repaired"
        assert proxy.is_loaded is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# The registry around the proxies.


def test_l5_the_registry_hands_back_one_proxy_per_name_and_the_alias_is_the_factory():
    lazy, restore = _window()
    try:
        first = lazy.lazy_import(_LEAF)
        again = lazy.lazy_import(_LEAF)
        other = lazy.lazy_import(_OTHER)
        assert again is first
        assert other is not first
        assert lazy.get_lazy_module is lazy.lazy_import
    finally:
        restore()


def test_l6_the_statistics_snapshot_reports_state_per_registered_name():
    lazy, restore = _window(seeded={_LEAF: _mod(_LEAF, marker=1)}, blocked=(_OTHER,))
    try:
        lazy.lazy_import(_LEAF).marker
        with pytest.raises(ImportError):
            lazy.lazy_import(_OTHER).marker
        stats = lazy.get_lazy_stats()
        assert set(stats) == {_LEAF, _OTHER}
        assert stats[_LEAF]["loaded"] is True
        assert isinstance(stats[_LEAF]["load_time"], float)
        assert stats[_LEAF]["error"] is None
        assert stats[_OTHER]["loaded"] is False
        assert isinstance(stats[_OTHER]["error"], str)
    finally:
        restore()


def test_l7_eager_preloading_maps_each_name_to_its_outcome():
    lazy, restore = _window(seeded={_LEAF: _mod(_LEAF)}, blocked=(_OTHER,))
    try:
        results = lazy.preload(_LEAF, _OTHER)
        assert results == {_LEAF: True, _OTHER: False}
        assert lazy.lazy_import(_LEAF).is_loaded is True
    finally:
        restore()


def test_l8_the_report_names_each_state_and_totals_the_loads():
    lazy, restore = _window(seeded={_LEAF: _mod(_LEAF, marker=1)}, blocked=(_OTHER,))
    try:
        assert lazy.get_startup_report() == "No lazy modules registered"
        lazy.lazy_import(_LEAF).marker
        with pytest.raises(ImportError):
            lazy.lazy_import(_OTHER).marker
        report = lazy.get_startup_report()
        assert f"{_LEAF}: loaded" in report
        assert f"{_OTHER}: FAILED" in report
        assert "Total: 1/2 loaded" in report
    finally:
        restore()


def test_l9_the_heavy_module_roster_is_exactly_the_frozen_list():
    lazy, restore = _window()
    try:
        assert tuple(lazy.HEAVY_MODULES) == _HEAVY_CENSUS
    finally:
        restore()


def test_l10_background_preload_is_a_named_daemon_thread_that_shields_its_callback():
    lazy, restore = _window(seeded={_LEAF: _mod(_LEAF)}, blocked=(_OTHER,))
    try:
        seen = []
        thread = lazy.preload_in_background(_LEAF, _OTHER, callback=seen.append, delay=0)
        assert thread.daemon is True
        assert thread.name == "lazy-preload"
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert seen == [{_LEAF: True, _OTHER: False}]

        def explode(results):
            raise RuntimeError("callback down")

        second = lazy.preload_in_background(_LEAF, callback=explode, delay=0)
        second.join(timeout=10)
        assert not second.is_alive()
    finally:
        restore()


# ---------------------------------------------------------------------------
# The attribute-level proxy: what it remembers and what it does not.


def test_l11_only_import_and_attribute_failures_are_remembered():
    settled = SimpleNamespace(value="ok")
    state = {"calls": 0}

    def flaky(name):
        state["calls"] += 1
        if state["calls"] == 1:
            raise ValueError("transient")
        return settled

    module = _mod(_LEAF)
    module.__getattr__ = flaky
    lazy, restore = _window(seeded={_LEAF: module})
    try:
        proxy = lazy.LazyAttr(_LEAF, "resource")
        with pytest.raises(ValueError):
            proxy.value
        assert proxy.value == "ok"
        assert state["calls"] == 2
    finally:
        restore()


def test_l12_an_attribute_that_resolves_to_none_is_resolved_again_on_every_touch():
    state = {"calls": 0}

    def always_none(name):
        state["calls"] += 1
        return None

    module = _mod(_LEAF)
    module.__getattr__ = always_none
    lazy, restore = _window(seeded={_LEAF: module})
    try:
        proxy = lazy.LazyAttr(_LEAF, "resource")
        with pytest.raises(AttributeError):
            proxy.value
        with pytest.raises(AttributeError):
            proxy.value
        assert state["calls"] == 2
    finally:
        restore()
