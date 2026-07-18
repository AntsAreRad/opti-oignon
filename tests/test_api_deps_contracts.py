#!/usr/bin/env python3
"""What the dependency-binding layer promises to every router that imports it.

The module under contract binds ninety-five availability flags and their
singletons through two very different mechanisms, and the difference is the
whole point of this suite. The historical family raises a flag only when a
guarded import actually succeeds, so a raised flag there means live code. The
deferred family raises its flag when the module's SPEC can be found, without
importing anything, and binds a proxy that will not try the import until the
first attribute is touched. A flag of the second family is therefore a claim
about existence, not importability -- and the proxy is always truthy, before
resolution and even after a failed one, so neither the flag nor a truthiness
check can tell a live singleton from a dead one. Both arms of that gap are
pinned here at their source: a module whose spec was found but which cannot
import fails at the first attribute touch with the import error, and a module
that imports but lacks the expected name fails the same way with the attribute
error; in both cases the failure is remembered and the very same exception
object is raised on every later touch.

The spec check itself has three distinct roads to a lowered flag and the suite
walks each one deliberately: a name whose cache entry carries no spec, a name
whose cache entry was neutralised, and a name the finder chain refuses. Under
the shared window each road is manufactured, never hoped for.

The historical family has its own corners. A re-exported flag follows the
dependency's own value while the singleton stays bound, so a consumer that
tests the object instead of the flag sees a different world than one that
tests the flag. Two names are bound at load time to the RESULT of a factory
call, and only an import failure is shielded there: any other factory error
escapes and takes the whole module load down with it. The agent group layers
the two mechanisms and can leave the class bound while the flag is down and
the instance is gone. The two runner names, one per module, are kept distinct
on purpose after an earlier shadowing, and each resolves through its own
module with its own flag.

The one function in the module consults the backend registry first and only
falls through to the direct client when the registry yields no backend or
raises; empty and missing answers are normalised to an empty list, and every
answer shape of the direct client is folded to a plain list.

Everything the module reaches for is seeded or declared unreachable and proven
so; the loader module is loaded REAL from its own source, first in the window,
because the module under contract cannot even be imported without it.
"""

import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_LOADER = "opti_oignon.lazy_loader"
_TARGET = "opti_oignon.api.deps"
_LOADER_SOURCE = source("lazy_loader.py")
_TARGET_SOURCE = source("api", "deps.py")

_ENGINE_MODULE = "opti_oignon.benchmark_runner"
_MICRO_MODULE = "opti_oignon.performance_benchmark"
_JUDGE_MODULE = "opti_oignon.benchmark_judge"

_FLAG_CENSUS = (
    "ADAPTIVE_ROUTING_AVAILABLE", "ADMIN_AUDIT_AVAILABLE", "ANALYTICS_AVAILABLE",
    "ANALYZER_AVAILABLE", "ARTIFACT_AVAILABLE", "AUTH_AVAILABLE", "AUTO_TRIGGER_AVAILABLE",
    "AUTO_TUNER_AVAILABLE", "BACKUP_AVAILABLE", "BENCHMARK_AVAILABLE",
    "BENCHMARK_HISTORY_AVAILABLE", "BENCHMARK_JUDGE_AVAILABLE",
    "BENCHMARK_RECOMMENDATIONS_AVAILABLE", "BENCHMARK_RUNNER_AVAILABLE",
    "BENCHMARK_V2_AVAILABLE", "BRANCHES_AVAILABLE", "CASCADING_AVAILABLE",
    "CODE_EXECUTOR_AVAILABLE", "CODING_AGENT_AVAILABLE", "CODING_HISTORY_AVAILABLE",
    "CONFIG_AVAILABLE", "CONSENSUS_AVAILABLE", "CONTEXT_MANAGER_AVAILABLE",
    "CONTEXT_OPTIMIZER_AVAILABLE", "CONTEXT_WINDOW_AVAILABLE", "CONVERSATION_AVAILABLE",
    "CONVERSATION_COMPRESSOR_AVAILABLE", "CUSTOM_PROFILES_AVAILABLE",
    "DB_ENCRYPTION_AVAILABLE", "EXECUTOR_AVAILABLE", "EXTERNAL_STORES_AVAILABLE",
    "FEEDBACK_AVAILABLE", "FILE_TOOLS_AVAILABLE", "FINE_TUNE_EXPORT_AVAILABLE",
    "FINE_TUNE_TRACKER_AVAILABLE", "FINGERPRINT_AVAILABLE", "HUMANIZER_AVAILABLE",
    "HYBRID_SEARCH_AVAILABLE", "INFERENCE_BACKEND_AVAILABLE", "INFERENCE_PROFILER_AVAILABLE",
    "LEARNED_ROUTER_AVAILABLE", "LIVE_METRICS_AVAILABLE", "LLAMA_CPP_AVAILABLE",
    "MEMORY_AVAILABLE", "MODEL_HEALTH_AVAILABLE", "MODEL_LIFECYCLE_AVAILABLE",
    "MODEL_MANAGER_AVAILABLE", "MODEL_WARMUP_AVAILABLE", "NETWORK_MANAGER_AVAILABLE",
    "OLLAMA_LIB_AVAILABLE", "PERFORMANCE_MONITOR_AVAILABLE", "PII_SANITIZER_AVAILABLE",
    "PIPELINE_AVAILABLE", "PLUGIN_ALLOWLIST_AVAILABLE", "PLUGIN_HOOKS_AVAILABLE",
    "PLUGIN_INDEX_AVAILABLE", "PLUGIN_INSTALLER_AVAILABLE", "PLUGIN_LOADER_AVAILABLE",
    "PLUGIN_REGISTRY_AVAILABLE", "PLUGIN_REVIEWS_AVAILABLE", "PLUGIN_TEMPLATE_AVAILABLE",
    "PLUGIN_USER_CONFIG_AVAILABLE", "PRESET_AVAILABLE", "PRE_CACHE_AVAILABLE",
    "PROFILE_AVAILABLE", "PROJECTS_AVAILABLE", "PROJECT_CONTEXT_AVAILABLE",
    "PROJECT_TRIGGERS_AVAILABLE", "PROMPT_OPTIMIZATION_AVAILABLE", "RAG_CHUNKER_AVAILABLE",
    "RAG_DASHBOARD_AVAILABLE", "RAG_STORE_AVAILABLE", "RBAC_ENFORCEMENT_AVAILABLE",
    "RESPONSE_CACHE_AVAILABLE", "ROUTER_AVAILABLE", "SANDBOX_AVAILABLE",
    "SANDBOX_TOOLS_AVAILABLE", "SEARCH_KILLSWITCH_AVAILABLE", "SECURE_BYTES_AVAILABLE",
    "SECURITY_MODE_AVAILABLE", "SEMANTIC_CACHE_AVAILABLE", "SMART_ROUTER_AVAILABLE",
    "SPECULATIVE_AVAILABLE", "SPECULATIVE_DECODING_AVAILABLE", "SQLCIPHER_AVAILABLE",
    "SYNC_QUEUE_AVAILABLE", "SYSTEM_PRESETS_AVAILABLE", "TELEMETRY_AVAILABLE",
    "TELEMETRY_HISTORY_AVAILABLE", "TWO_FA_AVAILABLE", "USER_DATA_MANAGER_AVAILABLE",
    "USER_KEY_MANAGER_AVAILABLE", "USER_SETTINGS_AVAILABLE", "VISION_CONFIG_AVAILABLE",
    "WEB_SEARCH_AVAILABLE",
)

_PROXY_CENSUS = (
    "CodingHistoryStore", "auto_trigger", "benchmark_evaluator", "benchmark_judge",
    "benchmark_recommender", "benchmark_runner", "coding_history_store",
    "custom_profile_store", "fine_tune_exporter", "fine_tune_tracker", "get_auto_refresh",
    "get_auto_tuner_manager", "get_external_manager", "get_history_store",
    "get_hybrid_engine", "get_profiler", "get_rag_dashboard", "get_rag_store",
    "get_speculative_decoding_manager", "get_telemetry", "judge_store",
    "perf_benchmark_runner", "plugin_index_instance", "plugin_review_store_instance",
    "plugin_template_instance", "remote_installer_instance", "reset_history_store",
    "reset_profiler", "reset_telemetry",
)

_REPRESENTATIVE_SINGLETONS = (
    "conversation_manager", "learned_router", "web_searcher", "CodingAgent",
    "coding_agent_instance", "context_manager_instance", "get_backend_registry",
    "pipeline_manager",
)


# ---------------------------------------------------------------------------
# Rigging: stand-in modules and one loader for the two-module window.


def _mod(name, *, spec=False, **attrs):
    """Build a stand-in module; a spec is attached only when asked for."""
    module = ModuleType(name)
    if spec:
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load(*, blocked=(), seeded=None, ollama=None):
    """Load the loader module REAL, then the module under contract.

    The direct inference client is unreachable by default; a test that
    exercises the direct path hands in its own stand-in instead.
    """
    seeds = dict(seeded or {})
    blocks = list(blocked)
    if ollama is None:
        blocks.append("ollama")
    else:
        seeds["ollama"] = ollama
    loaded, restore = isolate(
        targets={_LOADER: _LOADER_SOURCE, _TARGET: _TARGET_SOURCE},
        blocked=tuple(blocks),
        seeded=seeds,
        packages=("opti_oignon.api",),
    )
    return loaded[_TARGET], restore


class _Registry:
    """Backend registry stand-in that answers with one prepared backend."""

    def __init__(self, backend):
        self._backend = backend
        self.asked = []

    def get_backend(self, name):
        self.asked.append(name)
        return self._backend


class _RaisingRegistry:
    def get_backend(self, name):
        raise RuntimeError("registry down")


class _Backend:
    def __init__(self, models):
        self._models = models

    def list_models(self):
        return self._models


def _inference_seed(registry):
    return _mod(
        "opti_oignon.inference_backend",
        LLAMA_CPP_AVAILABLE=False,
        OLLAMA_AVAILABLE=False,
        get_backend_registry=lambda: registry,
        init_backends_from_config=lambda: None,
    )


class _Agent:
    """Sample agent class handed to the layered agent group."""


# ---------------------------------------------------------------------------
# The spec check: one road to a raised flag, three roads to a lowered one.


def test_d1_a_cached_module_carrying_a_spec_raises_the_deferred_flag():
    engine = SimpleNamespace(tag="engine")
    deps, restore = _load(
        seeded={_ENGINE_MODULE: _mod(_ENGINE_MODULE, spec=True, benchmark_runner=engine)},
    )
    try:
        assert deps.BENCHMARK_RUNNER_AVAILABLE is True
        assert deps.benchmark_runner is not None
    finally:
        restore()


def test_d2_a_cached_module_without_a_spec_reads_as_absent():
    deps, restore = _load(
        seeded={_ENGINE_MODULE: _mod(_ENGINE_MODULE, benchmark_runner=object())},
    )
    try:
        assert deps.BENCHMARK_RUNNER_AVAILABLE is False
        assert deps.benchmark_runner is None
    finally:
        restore()


def test_d3_a_neutralised_module_reads_as_absent():
    deps, restore = _load(blocked=("opti_oignon.rag_store",))
    try:
        assert deps.RAG_STORE_AVAILABLE is False
        assert deps.get_rag_store is None
    finally:
        restore()


def test_d4_a_module_the_finder_chain_refuses_reads_as_absent():
    deps, restore = _load()
    try:
        assert deps.BENCHMARK_RUNNER_AVAILABLE is False
        assert deps.benchmark_runner is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# The deferred proxies, observed from the consumer side.


def test_d5_the_loader_module_is_a_hard_dependency_of_the_layer():
    with pytest.raises(ImportError):
        isolate(
            targets={_TARGET: _TARGET_SOURCE},
            blocked=("ollama", _LOADER),
            packages=("opti_oignon.api",),
        )


def test_d6_an_unresolved_proxy_is_truthy_and_reports_itself_deferred():
    deps, restore = _load(
        seeded={_ENGINE_MODULE: _mod(_ENGINE_MODULE, spec=True, benchmark_runner=object())},
    )
    try:
        proxy = deps.benchmark_runner
        assert bool(proxy) is True
        assert "deferred" in repr(proxy)
    finally:
        restore()


def test_d7_the_first_attribute_touch_resolves_through_the_cached_module():
    marker = object()
    engine = SimpleNamespace(handle=marker)
    deps, restore = _load(
        seeded={_ENGINE_MODULE: _mod(_ENGINE_MODULE, spec=True, benchmark_runner=engine)},
    )
    try:
        assert deps.benchmark_runner.handle is marker
    finally:
        restore()


def test_d8_a_raised_flag_with_a_dead_module_fails_at_first_touch_and_stays_failed():
    deps, restore = _load(
        seeded={
            _ENGINE_MODULE: _mod(
                _ENGINE_MODULE, spec=True, benchmark_runner=SimpleNamespace(is_busy=False)
            ),
        },
    )
    try:
        assert deps.BENCHMARK_RUNNER_AVAILABLE is True
        sys.modules[_ENGINE_MODULE] = None
        with pytest.raises(ImportError) as first:
            deps.benchmark_runner.is_busy
        with pytest.raises(ImportError) as second:
            deps.benchmark_runner.is_busy
        assert second.value is first.value
        assert bool(deps.benchmark_runner) is True
    finally:
        restore()


def test_d9_a_raised_flag_with_a_missing_name_fails_and_the_failure_is_remembered():
    deps, restore = _load(seeded={_JUDGE_MODULE: _mod(_JUDGE_MODULE, spec=True)})
    try:
        assert deps.BENCHMARK_JUDGE_AVAILABLE is True
        with pytest.raises(AttributeError) as first:
            deps.benchmark_judge.mark
        with pytest.raises(AttributeError) as second:
            deps.benchmark_judge.grade
        assert second.value is first.value
        with pytest.raises(AttributeError):
            deps.judge_store.load
        assert bool(deps.benchmark_judge) is True
    finally:
        restore()


def test_d10_calling_a_proxy_invokes_the_resolved_callable():
    store = object()
    deps, restore = _load(
        seeded={
            "opti_oignon.rag_store": _mod(
                "opti_oignon.rag_store", spec=True, get_rag_store=lambda: store
            ),
        },
    )
    try:
        assert deps.get_rag_store() is store
    finally:
        restore()


def test_d11_the_two_runner_names_bind_independent_modules():
    micro = SimpleNamespace(tag="micro")
    engine = SimpleNamespace(tag="engine")
    deps, restore = _load(
        seeded={
            _MICRO_MODULE: _mod(_MICRO_MODULE, spec=True, benchmark_runner=micro),
            _ENGINE_MODULE: _mod(_ENGINE_MODULE, spec=True, benchmark_runner=engine),
        },
    )
    try:
        assert deps.BENCHMARK_AVAILABLE is True
        assert deps.BENCHMARK_RUNNER_AVAILABLE is True
        assert deps.perf_benchmark_runner.tag == "micro"
        assert deps.benchmark_runner.tag == "engine"
    finally:
        restore()
    deps, restore = _load(
        seeded={_MICRO_MODULE: _mod(_MICRO_MODULE, spec=True, benchmark_runner=micro)},
    )
    try:
        assert deps.BENCHMARK_AVAILABLE is True
        assert deps.BENCHMARK_RUNNER_AVAILABLE is False
        assert deps.benchmark_runner is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# The guarded-import family: plain groups.


def test_d12_a_guarded_import_that_succeeds_raises_the_flag_and_binds_the_singleton():
    manager = object()
    deps, restore = _load(
        seeded={
            "opti_oignon.conversation": _mod(
                "opti_oignon.conversation", conversation_manager=manager
            ),
        },
    )
    try:
        assert deps.CONVERSATION_AVAILABLE is True
        assert deps.conversation_manager is manager
    finally:
        restore()


def test_d13_a_guarded_import_that_fails_lowers_the_flag_and_binds_none():
    deps, restore = _load(blocked=("opti_oignon.response_cache",))
    try:
        assert deps.RESPONSE_CACHE_AVAILABLE is False
        assert deps.response_cache is None
    finally:
        restore()


def test_d14_a_present_module_missing_the_requested_name_counts_as_a_failed_import():
    deps, restore = _load(seeded={"opti_oignon.memory": _mod("opti_oignon.memory")})
    try:
        assert deps.MEMORY_AVAILABLE is False
        assert deps.memory_manager is None
    finally:
        restore()


def test_d15_a_two_name_group_binds_all_or_nothing():
    sliding = object()
    budget = object()
    deps, restore = _load(
        seeded={
            "opti_oignon.context_window": _mod(
                "opti_oignon.context_window",
                sliding_window_manager=sliding,
                token_budget_manager=budget,
            ),
        },
    )
    try:
        assert deps.CONTEXT_WINDOW_AVAILABLE is True
        assert deps.sliding_window_manager is sliding
        assert deps.token_budget_manager is budget
    finally:
        restore()
    deps, restore = _load(
        seeded={
            "opti_oignon.context_window": _mod(
                "opti_oignon.context_window", sliding_window_manager=sliding
            ),
        },
    )
    try:
        assert deps.CONTEXT_WINDOW_AVAILABLE is False
        assert deps.sliding_window_manager is None
        assert deps.token_budget_manager is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# The guarded-import family: factories called at load time.


def test_d16_the_factory_result_is_bound_at_load_time_not_the_factory():
    calls = []
    pipeline = object()

    def factory():
        calls.append(1)
        return pipeline

    context = object()
    deps, restore = _load(
        seeded={
            "opti_oignon.pipeline_manager": _mod(
                "opti_oignon.pipeline_manager", get_pipeline_manager=factory
            ),
            "opti_oignon.context_manager": _mod(
                "opti_oignon.context_manager", get_context_manager=lambda: context
            ),
        },
    )
    try:
        assert deps.PIPELINE_AVAILABLE is True
        assert deps.pipeline_manager is pipeline
        assert deps.get_pipeline_manager is factory
        assert calls == [1]
        assert deps.CONTEXT_MANAGER_AVAILABLE is True
        assert deps.context_manager_instance is context
    finally:
        restore()


def test_d17_a_factory_import_failure_degrades_the_group_cleanly():
    def factory():
        raise ImportError("inner refusal")

    deps, restore = _load(
        seeded={
            "opti_oignon.pipeline_manager": _mod(
                "opti_oignon.pipeline_manager", get_pipeline_manager=factory
            ),
        },
    )
    try:
        assert deps.PIPELINE_AVAILABLE is False
        assert deps.pipeline_manager is None
        assert deps.get_pipeline_manager is None
    finally:
        restore()


def test_d18_only_import_failures_are_shielded_any_other_factory_error_escapes():
    def factory():
        raise RuntimeError("factory down")

    with pytest.raises(RuntimeError):
        isolate(
            targets={_LOADER: _LOADER_SOURCE, _TARGET: _TARGET_SOURCE},
            blocked=("ollama",),
            seeded={
                "opti_oignon.pipeline_manager": _mod(
                    "opti_oignon.pipeline_manager", get_pipeline_manager=factory
                ),
            },
            packages=("opti_oignon.api",),
        )


# ---------------------------------------------------------------------------
# The guarded-import family: re-exported flags.


def test_d19_a_re_exported_flag_follows_the_dependency_while_the_singleton_stays_bound():
    router_obj = object()
    deps, restore = _load(
        seeded={
            "opti_oignon.learned_router": _mod(
                "opti_oignon.learned_router",
                LEARNED_ROUTER_AVAILABLE=False,
                learned_router=router_obj,
                LearnedRouterMetrics=int,
            ),
        },
    )
    try:
        assert deps.LEARNED_ROUTER_AVAILABLE is False
        assert deps.learned_router is router_obj
        assert deps.LearnedRouterMetrics is int
    finally:
        restore()


def test_d20_the_search_flag_mirrors_the_dependency_flag_under_its_own_name():
    searcher = object()
    for value in (True, False):
        deps, restore = _load(
            seeded={
                "opti_oignon.web_search": _mod(
                    "opti_oignon.web_search", DDGS_AVAILABLE=value, web_searcher=searcher
                ),
            },
        )
        try:
            assert deps.WEB_SEARCH_AVAILABLE is value
            assert deps.web_searcher is searcher
        finally:
            restore()


def test_d21_a_dependency_missing_its_flag_name_degrades_like_a_failed_import():
    deps, restore = _load(
        seeded={"opti_oignon.auth": _mod("opti_oignon.auth", auth_manager=object())},
    )
    try:
        assert deps.AUTH_AVAILABLE is False
        assert deps.auth_manager is None
    finally:
        restore()


def test_d22_flag_only_re_exports_pass_the_dependency_value_through():
    deps, restore = _load(
        blocked=("opti_oignon.sandbox_tools",),
        seeded={
            "opti_oignon.file_tools": _mod(
                "opti_oignon.file_tools", FILE_TOOLS_AVAILABLE=True
            ),
            "opti_oignon.rbac_enforcement": _mod(
                "opti_oignon.rbac_enforcement", RBAC_ENFORCEMENT_AVAILABLE=True
            ),
        },
    )
    try:
        assert deps.FILE_TOOLS_AVAILABLE is True
        assert deps.RBAC_ENFORCEMENT_AVAILABLE is True
        assert deps.SANDBOX_TOOLS_AVAILABLE is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# The layered agent group.


def _agent_seed(**attrs):
    return _mod("opti_oignon.coding_agent", spec=True, **attrs)


def test_d23_the_agent_is_instantiated_at_load_time_when_the_inner_flag_is_up():
    deps, restore = _load(
        seeded={
            "opti_oignon.coding_agent": _agent_seed(
                CODING_AGENT_AVAILABLE=True, CodingAgent=_Agent, coding_agent_config=object()
            ),
        },
    )
    try:
        assert deps.CODING_AGENT_AVAILABLE is True
        assert isinstance(deps.coding_agent_instance, _Agent)
        assert deps.CodingAgent is _Agent
    finally:
        restore()


def test_d24_an_inner_flag_down_leaves_the_class_bound_but_no_instance():
    deps, restore = _load(
        seeded={
            "opti_oignon.coding_agent": _agent_seed(
                CODING_AGENT_AVAILABLE=False, CodingAgent=_Agent, coding_agent_config=object()
            ),
        },
    )
    try:
        assert deps.CODING_AGENT_AVAILABLE is False
        assert deps.coding_agent_instance is None
        assert deps.CodingAgent is _Agent
    finally:
        restore()


def test_d25_a_spec_bearing_module_missing_the_agent_names_degrades():
    deps, restore = _load(seeded={"opti_oignon.coding_agent": _agent_seed()})
    try:
        assert deps.CODING_AGENT_AVAILABLE is False
        assert deps.CodingAgent is None
        assert deps.coding_agent_instance is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# The model-listing function.


def test_d26_the_backend_registry_answers_first_and_empty_answers_become_lists():
    for models, expected in ((["alpha"], ["alpha"]), ([], []), (None, [])):
        registry = _Registry(_Backend(models))
        deps, restore = _load(
            seeded={"opti_oignon.inference_backend": _inference_seed(registry)},
        )
        try:
            assert deps.get_ollama_models() == expected
            assert registry.asked == ["ollama"]
        finally:
            restore()


def test_d27_a_missing_or_failing_backend_falls_through_to_the_direct_client():
    for registry in (_Registry(None), _RaisingRegistry()):
        direct = _mod("ollama", list=lambda: {"models": ["direct"]})
        deps, restore = _load(
            seeded={"opti_oignon.inference_backend": _inference_seed(registry)},
            ollama=direct,
        )
        try:
            assert deps.get_ollama_models() == ["direct"]
        finally:
            restore()


def test_d28_every_direct_client_answer_shape_is_folded_to_a_plain_list():
    class _Listing:
        models = ["held"]

    cases = (
        (_mod("ollama", list=lambda: _Listing()), ["held"]),
        (_mod("ollama", list=lambda: {"models": ["mapped"]}), ["mapped"]),
        (_mod("ollama", list=lambda: []), []),
    )
    for client, expected in cases:
        deps, restore = _load(ollama=client)
        try:
            assert deps.get_ollama_models() == expected
        finally:
            restore()


# ---------------------------------------------------------------------------
# The whole surface, with everything absent.


def test_d29_with_every_dependency_absent_every_flag_is_a_lowered_boolean():
    deps, restore = _load()
    try:
        flags = tuple(sorted(name for name in vars(deps) if name.endswith("_AVAILABLE")))
        assert flags == _FLAG_CENSUS
        assert all(getattr(deps, name) is False for name in _FLAG_CENSUS)
    finally:
        restore()


def test_d30_with_every_dependency_absent_every_proxy_and_singleton_is_none():
    deps, restore = _load()
    try:
        for name in _PROXY_CENSUS:
            assert getattr(deps, name) is None
        for name in _REPRESENTATIVE_SINGLETONS:
            assert getattr(deps, name) is None
        assert deps.get_ollama_models() == []
    finally:
        restore()
