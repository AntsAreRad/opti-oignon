#!/usr/bin/env python3
"""
Dependency helpers for the FastAPI API layer.

Provides singleton access to existing Opti-Oignon components
via conditional imports with availability flags.
"""

import importlib.util
import logging

from opti_oignon.lazy_loader import LazyAttr as _LazyAttr

logger = logging.getLogger(__name__)


def _module_exists(name: str) -> bool:
    """Check if a module file exists without importing it (S134)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


# -- Conversation Manager --
try:
    from opti_oignon.conversation import conversation_manager
    CONVERSATION_AVAILABLE = True
except ImportError:
    CONVERSATION_AVAILABLE = False
    conversation_manager = None

# -- Response Cache --
try:
    from opti_oignon.response_cache import response_cache
    RESPONSE_CACHE_AVAILABLE = True
except ImportError:
    RESPONSE_CACHE_AVAILABLE = False
    response_cache = None

# -- Semantic Cache --
try:
    from opti_oignon.semantic_cache import semantic_cache
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False
    semantic_cache = None

# -- Memory Store --
try:
    from opti_oignon.memory import memory_manager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    memory_manager = None

# -- Artifact Manager --
try:
    from opti_oignon.artifacts import artifact_manager
    ARTIFACT_AVAILABLE = True
except ImportError:
    ARTIFACT_AVAILABLE = False
    artifact_manager = None

# -- Code Executor --
try:
    from opti_oignon.code_executor import code_executor
    CODE_EXECUTOR_AVAILABLE = True
except ImportError:
    CODE_EXECUTOR_AVAILABLE = False
    code_executor = None

# -- Context Window --
try:
    from opti_oignon.context_window import sliding_window_manager, token_budget_manager
    CONTEXT_WINDOW_AVAILABLE = True
except ImportError:
    CONTEXT_WINDOW_AVAILABLE = False
    sliding_window_manager = None
    token_budget_manager = None

# -- Model Warmup --
try:
    from opti_oignon.model_warmup import model_warmup
    MODEL_WARMUP_AVAILABLE = True
except ImportError:
    MODEL_WARMUP_AVAILABLE = False
    model_warmup = None

# -- Benchmark Runner (lazy S134) --
BENCHMARK_AVAILABLE = _module_exists("opti_oignon.performance_benchmark")
# S193 PRF-01: this is the performance micro-benchmark runner. It was exported
# as `benchmark_runner` and silently shadowed by the S88 BenchmarkRunner export
# further down (same name), so routes_health's run_all()/run() calls hit the S88
# runner and raised AttributeError. Renamed to keep both reachable.
perf_benchmark_runner = _LazyAttr("opti_oignon.performance_benchmark", "benchmark_runner") if BENCHMARK_AVAILABLE else None

# -- Preset Manager --
try:
    from opti_oignon.presets import preset_manager
    PRESET_AVAILABLE = True
except ImportError:
    PRESET_AVAILABLE = False
    preset_manager = None

# -- Pipeline Manager --
try:
    from opti_oignon.pipeline_manager import get_pipeline_manager
    pipeline_manager = get_pipeline_manager()
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    pipeline_manager = None
    get_pipeline_manager = None

# -- Executor --
try:
    from opti_oignon.executor import executor
    EXECUTOR_AVAILABLE = True
except ImportError:
    EXECUTOR_AVAILABLE = False
    executor = None

# -- Analyzer --
try:
    from opti_oignon.analyzer import analyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    analyzer = None

# -- Router --
try:
    from opti_oignon.router import router
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False
    router = None


# -- Config --
try:
    from opti_oignon.config import config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    config = None

# -- Model Profiles (S46) --
try:
    from opti_oignon.model_profiles import profile_manager
    PROFILE_AVAILABLE = True
except ImportError:
    PROFILE_AVAILABLE = False
    profile_manager = None

# -- Context Manager (S47) --
try:
    from opti_oignon.context_manager import get_context_manager
    context_manager_instance = get_context_manager()
    CONTEXT_MANAGER_AVAILABLE = True
except ImportError:
    CONTEXT_MANAGER_AVAILABLE = False
    context_manager_instance = None
    get_context_manager = None

# -- Consensus Engine (S50) --
try:
    from opti_oignon.consensus import consensus_engine
    CONSENSUS_AVAILABLE = True
except ImportError:
    CONSENSUS_AVAILABLE = False
    consensus_engine = None

# -- Smart Router (S54) --
try:
    from opti_oignon.smart_router import smart_router
    SMART_ROUTER_AVAILABLE = True
except ImportError:
    SMART_ROUTER_AVAILABLE = False
    smart_router = None

# -- Feedback Store (S55) --
try:
    from opti_oignon.feedback import feedback_store
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False
    feedback_store = None

# -- Adaptive Routing (S62) --
try:
    from opti_oignon.adaptive_routing import feedback_routing_adapter
    ADAPTIVE_ROUTING_AVAILABLE = True
except ImportError:
    ADAPTIVE_ROUTING_AVAILABLE = False
    feedback_routing_adapter = None

# -- Model Health Monitor (S63) --
try:
    from opti_oignon.model_health import model_health_monitor
    MODEL_HEALTH_AVAILABLE = True
except ImportError:
    MODEL_HEALTH_AVAILABLE = False
    model_health_monitor = None

# -- Analytics Engine (S55) --
try:
    from opti_oignon.analytics import analytics_engine, performance_tracker
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    analytics_engine = None
    performance_tracker = None

# -- Project Store (S57) --
try:
    from opti_oignon.projects import project_store
    PROJECTS_AVAILABLE = True
except ImportError:
    PROJECTS_AVAILABLE = False
    project_store = None

# -- Project Context (S58) --
try:
    from opti_oignon.project_context import project_context_builder, project_indexer
    PROJECT_CONTEXT_AVAILABLE = True
except ImportError:
    PROJECT_CONTEXT_AVAILABLE = False
    project_indexer = None
    project_context_builder = None

# -- Project Triggers (S58) --
try:
    from opti_oignon.project_triggers import trigger_detector
    PROJECT_TRIGGERS_AVAILABLE = True
except ImportError:
    PROJECT_TRIGGERS_AVAILABLE = False
    trigger_detector = None

# -- Prompt Optimization (S65) --
try:
    from opti_oignon.prompt_optimization import prompt_budget_manager, prompt_template_engine
    PROMPT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PROMPT_OPTIMIZATION_AVAILABLE = False
    prompt_budget_manager = None
    prompt_template_engine = None

# -- Benchmark History (S60) --
try:
    from opti_oignon.benchmark_history import benchmark_history as benchmark_history_store
    BENCHMARK_HISTORY_AVAILABLE = True
except ImportError:
    BENCHMARK_HISTORY_AVAILABLE = False
    benchmark_history_store = None
# -- Conversation Compressor (S66) --
try:
    from opti_oignon.conversation_compressor import conversation_compressor
    CONVERSATION_COMPRESSOR_AVAILABLE = True
except ImportError:
    CONVERSATION_COMPRESSOR_AVAILABLE = False
    conversation_compressor = None

# -- Learned Router (S67) --
try:
    from opti_oignon.learned_router import (
        LEARNED_ROUTER_AVAILABLE,
        LearnedRouterMetrics,
        learned_router,
    )
    LEARNED_ROUTER_AVAILABLE = LEARNED_ROUTER_AVAILABLE  # re-export
except ImportError:
    LEARNED_ROUTER_AVAILABLE = False
    learned_router = None
    LearnedRouterMetrics = None

# -- Cascading Inference (S69) --
try:
    from opti_oignon.cascading import cascading_inference
    CASCADING_AVAILABLE = True
except ImportError:
    CASCADING_AVAILABLE = False
    cascading_inference = None

# -- Speculative Generation (S70) --
try:
    from opti_oignon.speculative import speculative_generator
    SPECULATIVE_AVAILABLE = True
except ImportError:
    SPECULATIVE_AVAILABLE = False
    speculative_generator = None

# -- Network Manager (S71) --
try:
    from opti_oignon.network_manager import network_manager
    NETWORK_MANAGER_AVAILABLE = True
except ImportError:
    NETWORK_MANAGER_AVAILABLE = False
    network_manager = None

# -- Sync Queue (S71) --
try:
    from opti_oignon.sync_queue import sync_queue
    SYNC_QUEUE_AVAILABLE = True
except ImportError:
    SYNC_QUEUE_AVAILABLE = False
    sync_queue = None

# -- Pre-Cache (S71) --
try:
    from opti_oignon.pre_cache import pre_cache
    PRE_CACHE_AVAILABLE = True
except ImportError:
    PRE_CACHE_AVAILABLE = False
    pre_cache = None

# -- Performance Monitor (S72) --
try:
    from opti_oignon.performance_monitor import performance_monitor
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False
    performance_monitor = None

# -- Sandbox Manager (S73) --
try:
    from opti_oignon.sandbox_manager import (
        SANDBOX_AVAILABLE,
        sandbox_manager,
    )
    SANDBOX_AVAILABLE = SANDBOX_AVAILABLE  # re-export
except ImportError:
    SANDBOX_AVAILABLE = False
    sandbox_manager = None

# -- File Tools (S73) --
try:
    from opti_oignon.file_tools import FILE_TOOLS_AVAILABLE
    FILE_TOOLS_AVAILABLE = FILE_TOOLS_AVAILABLE  # re-export
except ImportError:
    FILE_TOOLS_AVAILABLE = False

# -- Sandbox Tools (S73) --
try:
    from opti_oignon.sandbox_tools import SANDBOX_TOOLS_AVAILABLE
    SANDBOX_TOOLS_AVAILABLE = SANDBOX_TOOLS_AVAILABLE  # re-export
except ImportError:
    SANDBOX_TOOLS_AVAILABLE = False

# -- Coding Agent (S74 / lazy S134) --
# CodingAgent requires instantiation so we use conditional import
# but defer until _module_exists confirms availability.
CODING_AGENT_AVAILABLE = _module_exists("opti_oignon.coding_agent")
CodingAgent = None
coding_agent_config = None
coding_agent_instance = None
if CODING_AGENT_AVAILABLE:
    try:
        from opti_oignon.coding_agent import (
            CODING_AGENT_AVAILABLE as _ca_avail,
        )
        from opti_oignon.coding_agent import (
            CodingAgent,
            coding_agent_config,  # noqa: F401
        )
        CODING_AGENT_AVAILABLE = _ca_avail
        coding_agent_instance = CodingAgent() if CODING_AGENT_AVAILABLE else None
    except ImportError:
        CODING_AGENT_AVAILABLE = False

# -- Session Fingerprint (S75) --
try:
    from opti_oignon.session_fingerprint import (
        FINGERPRINT_AVAILABLE,
        FingerprintManager,
        fingerprint_config,
    )
    FINGERPRINT_AVAILABLE = FINGERPRINT_AVAILABLE  # re-export
except ImportError:
    FINGERPRINT_AVAILABLE = False
    FingerprintManager = None
    fingerprint_config = None

# -- Coding History (S76 / lazy S134) --
CODING_HISTORY_AVAILABLE = _module_exists("opti_oignon.coding_history")
CodingHistoryStore = _LazyAttr("opti_oignon.coding_history", "CodingHistoryStore") if CODING_HISTORY_AVAILABLE else None
coding_history_store = _LazyAttr("opti_oignon.coding_history", "coding_history_store") if CODING_HISTORY_AVAILABLE else None

# -- Web Search (S82) --
try:
    from opti_oignon.web_search import (
        DDGS_AVAILABLE as _DDGS_AVAIL,
    )
    from opti_oignon.web_search import (
        web_searcher,
    )
    WEB_SEARCH_AVAILABLE = _DDGS_AVAIL
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    web_searcher = None

# -- PII Sanitizer (S82) --
try:
    from opti_oignon.pii_sanitizer import pii_sanitizer
    PII_SANITIZER_AVAILABLE = True
except ImportError:
    PII_SANITIZER_AVAILABLE = False
    pii_sanitizer = None

# -- System Presets (S84) --
try:
    from opti_oignon.system_presets import (
        SYSTEM_PRESETS_AVAILABLE,
        system_presets_manager,
    )
    SYSTEM_PRESETS_AVAILABLE = SYSTEM_PRESETS_AVAILABLE  # re-export
except ImportError:
    SYSTEM_PRESETS_AVAILABLE = False
    system_presets_manager = None

# -- Humanizer (S86) --
try:
    from opti_oignon.humanizer import HUMANIZER_AVAILABLE, humanizer_engine
    HUMANIZER_AVAILABLE = HUMANIZER_AVAILABLE  # re-export
except ImportError:
    HUMANIZER_AVAILABLE = False
    humanizer_engine = None

# -- Benchmark Evaluator V2 (S88 / lazy S134) --
BENCHMARK_V2_AVAILABLE = _module_exists("opti_oignon.benchmark_evaluator")
benchmark_evaluator = _LazyAttr("opti_oignon.benchmark_evaluator", "benchmark_evaluator") if BENCHMARK_V2_AVAILABLE else None

# -- Benchmark Runner V2 (S88 / lazy S134) --
BENCHMARK_RUNNER_AVAILABLE = _module_exists("opti_oignon.benchmark_runner")
benchmark_runner = _LazyAttr("opti_oignon.benchmark_runner", "benchmark_runner") if BENCHMARK_RUNNER_AVAILABLE else None

# -- Benchmark Judge (S89 / lazy S134) --
BENCHMARK_JUDGE_AVAILABLE = _module_exists("opti_oignon.benchmark_judge")
benchmark_judge = _LazyAttr("opti_oignon.benchmark_judge", "benchmark_judge") if BENCHMARK_JUDGE_AVAILABLE else None
judge_store = _LazyAttr("opti_oignon.benchmark_judge", "judge_store") if BENCHMARK_JUDGE_AVAILABLE else None

# -- Benchmark Recommendations (S89 / lazy S134) --
BENCHMARK_RECOMMENDATIONS_AVAILABLE = _module_exists("opti_oignon.benchmark_recommendations")
benchmark_recommender = _LazyAttr("opti_oignon.benchmark_recommendations", "benchmark_recommender") if BENCHMARK_RECOMMENDATIONS_AVAILABLE else None

# -- Benchmark Auto-Trigger (S90 / lazy S134) --
AUTO_TRIGGER_AVAILABLE = _module_exists("opti_oignon.benchmark_auto_trigger")
auto_trigger = _LazyAttr("opti_oignon.benchmark_auto_trigger", "auto_trigger") if AUTO_TRIGGER_AVAILABLE else None

# -- Benchmark Custom Profiles (S90 / lazy S134) --
CUSTOM_PROFILES_AVAILABLE = _module_exists("opti_oignon.benchmark_custom_profiles")
custom_profile_store = _LazyAttr("opti_oignon.benchmark_custom_profiles", "custom_profile_store") if CUSTOM_PROFILES_AVAILABLE else None

# -- Vision Config (S94) --
try:
    from opti_oignon.vision_config import VISION_CONFIG_AVAILABLE, vision_config
    VISION_CONFIG_AVAILABLE = VISION_CONFIG_AVAILABLE  # re-export
except ImportError:
    VISION_CONFIG_AVAILABLE = False
    vision_config = None

# -- Fine-Tune Export (S96 / lazy S134) --
FINE_TUNE_EXPORT_AVAILABLE = _module_exists("opti_oignon.fine_tune_export")
fine_tune_exporter = _LazyAttr("opti_oignon.fine_tune_export", "fine_tune_exporter") if FINE_TUNE_EXPORT_AVAILABLE else None

# -- Fine-Tune Tracker (S96 / lazy S134) --
FINE_TUNE_TRACKER_AVAILABLE = _module_exists("opti_oignon.fine_tune_tracker")
fine_tune_tracker = _LazyAttr("opti_oignon.fine_tune_tracker", "fine_tune_tracker") if FINE_TUNE_TRACKER_AVAILABLE else None

# -- Conversation Branches (S97) --
try:
    from opti_oignon.conversation_branches import (
        BRANCHES_AVAILABLE,
        branch_manager,
    )
except ImportError:
    BRANCHES_AVAILABLE = False
    branch_manager = None

# -- Auth Manager (S98) --
try:
    from opti_oignon.auth import (
        AUTH_AVAILABLE,
        auth_manager,
    )
except ImportError:
    AUTH_AVAILABLE = False
    auth_manager = None

# -- User Settings Store (S98) --
try:
    from opti_oignon.user_isolation import (
        USER_SETTINGS_AVAILABLE,
        user_settings_store,
    )
except ImportError:
    USER_SETTINGS_AVAILABLE = False
    user_settings_store = None

# -- RAG Vector Store (S99 / lazy S134) --
RAG_STORE_AVAILABLE = _module_exists("opti_oignon.rag_store")
get_rag_store = _LazyAttr("opti_oignon.rag_store", "get_rag_store") if RAG_STORE_AVAILABLE else None

# -- RAG Chunker (S99 / lazy S134) --
RAG_CHUNKER_AVAILABLE = _module_exists("opti_oignon.rag_chunker")

# -- RAG Hybrid Search (S100 / lazy S134) --
HYBRID_SEARCH_AVAILABLE = _module_exists("opti_oignon.rag_hybrid_search")
get_hybrid_engine = _LazyAttr("opti_oignon.rag_hybrid_search", "get_hybrid_engine") if HYBRID_SEARCH_AVAILABLE else None
# Presence is not reach. The flag above only says the module resolves; it says
# nothing about whether any product surface routes a query through the engine.
# Reporting presence as a capability would advertise a retrieval path an
# install cannot take, so capability reporting consults this statement
# instead. It is checked against the tree by contract: wiring a caller
# without flipping it fails, and flipping it without a caller fails too.
HYBRID_SEARCH_ROUTED = False

# -- RAG External Stores (S100 / lazy S134) --
EXTERNAL_STORES_AVAILABLE = _module_exists("opti_oignon.rag_external")
get_external_manager = _LazyAttr("opti_oignon.rag_external", "get_external_manager") if EXTERNAL_STORES_AVAILABLE else None

# -- RAG Dashboard (S100 / lazy S134) --
RAG_DASHBOARD_AVAILABLE = _module_exists("opti_oignon.rag_dashboard")
get_rag_dashboard = _LazyAttr("opti_oignon.rag_dashboard", "get_rag_dashboard") if RAG_DASHBOARD_AVAILABLE else None
get_auto_refresh = _LazyAttr("opti_oignon.rag_dashboard", "get_auto_refresh") if RAG_DASHBOARD_AVAILABLE else None


# -- Plugin Registry (S101) --
try:
    from opti_oignon.plugin_manifest import (
        PLUGIN_MANIFEST_AVAILABLE as PLUGIN_REGISTRY_AVAILABLE,
    )
    from opti_oignon.plugin_manifest import (
        plugin_registry as plugin_registry_instance,
    )
except ImportError:
    PLUGIN_REGISTRY_AVAILABLE = False
    plugin_registry_instance = None

# -- Plugin Loader (S101) --
try:
    from opti_oignon.plugin_loader import (
        PLUGIN_LOADER_AVAILABLE,
    )
    from opti_oignon.plugin_loader import (
        plugin_loader as plugin_loader_instance,
    )
except ImportError:
    PLUGIN_LOADER_AVAILABLE = False
    plugin_loader_instance = None

# -- Plugin Hooks (S101) --
try:
    from opti_oignon.plugin_hooks import (
        PLUGIN_HOOKS_AVAILABLE,
        hook_manager,
    )
except ImportError:
    PLUGIN_HOOKS_AVAILABLE = False
    hook_manager = None

# -- Plugin Index (S102 / lazy S134) --
PLUGIN_INDEX_AVAILABLE = _module_exists("opti_oignon.plugin_index")
plugin_index_instance = _LazyAttr("opti_oignon.plugin_index", "plugin_index") if PLUGIN_INDEX_AVAILABLE else None

# -- Plugin Installer (S102 / lazy S134) --
PLUGIN_INSTALLER_AVAILABLE = _module_exists("opti_oignon.plugin_installer")
remote_installer_instance = _LazyAttr("opti_oignon.plugin_installer", "remote_installer") if PLUGIN_INSTALLER_AVAILABLE else None

# -- Plugin Reviews (S102 / lazy S134) --
PLUGIN_REVIEWS_AVAILABLE = _module_exists("opti_oignon.plugin_reviews")
plugin_review_store_instance = _LazyAttr("opti_oignon.plugin_reviews", "plugin_review_store") if PLUGIN_REVIEWS_AVAILABLE else None

# -- Plugin Template (S102 / lazy S134) --
PLUGIN_TEMPLATE_AVAILABLE = _module_exists("opti_oignon.plugin_template")
plugin_template_instance = _LazyAttr("opti_oignon.plugin_template", "plugin_template_generator") if PLUGIN_TEMPLATE_AVAILABLE else None

# -- Inference Backend (S105) --
try:
    from opti_oignon.inference_backend import (
        LLAMA_CPP_AVAILABLE,
        get_backend_registry,
        init_backends_from_config,
    )
    from opti_oignon.inference_backend import (
        OLLAMA_AVAILABLE as OLLAMA_LIB_AVAILABLE,
    )
    INFERENCE_BACKEND_AVAILABLE = True
except ImportError:
    INFERENCE_BACKEND_AVAILABLE = False
    get_backend_registry = None
    init_backends_from_config = None
    OLLAMA_LIB_AVAILABLE = False
    LLAMA_CPP_AVAILABLE = False

# -- GGUF Model Manager (S105) --
try:
    from opti_oignon.model_manager import (
        get_model_manager,
        init_model_manager,
    )
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    get_model_manager = None
    init_model_manager = None

# -- Speculative Decoding Manager (S110 / lazy S134) --
SPECULATIVE_DECODING_AVAILABLE = _module_exists("opti_oignon.speculative_decoding")
get_speculative_decoding_manager = _LazyAttr("opti_oignon.speculative_decoding", "get_speculative_decoding_manager") if SPECULATIVE_DECODING_AVAILABLE else None

# -- Auto-Tuner Manager (S110 / lazy S134) --
AUTO_TUNER_AVAILABLE = _module_exists("opti_oignon.auto_tuner")
get_auto_tuner_manager = _LazyAttr("opti_oignon.auto_tuner", "get_auto_tuner_manager") if AUTO_TUNER_AVAILABLE else None

# -- Live Metrics Collector (S111) --
try:
    from opti_oignon.live_metrics import (
        get_live_metrics,
        reset_live_metrics,
    )
    LIVE_METRICS_AVAILABLE = True
except ImportError:
    LIVE_METRICS_AVAILABLE = False
    get_live_metrics = None
    reset_live_metrics = None

# -- Telemetry Collector (S112 / lazy S134) --
TELEMETRY_AVAILABLE = _module_exists("opti_oignon.telemetry")
get_telemetry = _LazyAttr("opti_oignon.telemetry", "get_telemetry") if TELEMETRY_AVAILABLE else None  # type: ignore[assignment]
reset_telemetry = _LazyAttr("opti_oignon.telemetry", "reset_telemetry") if TELEMETRY_AVAILABLE else None  # type: ignore[assignment]

# -- Inference Profiler (S113 / lazy S134) --
INFERENCE_PROFILER_AVAILABLE = _module_exists("opti_oignon.inference_profiler")
get_profiler = _LazyAttr("opti_oignon.inference_profiler", "get_profiler") if INFERENCE_PROFILER_AVAILABLE else None  # type: ignore[assignment]
reset_profiler = _LazyAttr("opti_oignon.inference_profiler", "reset_profiler") if INFERENCE_PROFILER_AVAILABLE else None  # type: ignore[assignment]

# -- Telemetry History Store (S114 / lazy S134) --
TELEMETRY_HISTORY_AVAILABLE = _module_exists("opti_oignon.telemetry_history")
get_history_store = _LazyAttr("opti_oignon.telemetry_history", "get_history_store") if TELEMETRY_HISTORY_AVAILABLE else None  # type: ignore[assignment]
reset_history_store = _LazyAttr("opti_oignon.telemetry_history", "reset_history_store") if TELEMETRY_HISTORY_AVAILABLE else None  # type: ignore[assignment]

try:
    from opti_oignon.model_lifecycle import (
        get_lifecycle_manager,
    )
    from opti_oignon.model_lifecycle import (
        reset_manager as reset_lifecycle_manager,
    )
    MODEL_LIFECYCLE_AVAILABLE = True
except ImportError:
    MODEL_LIFECYCLE_AVAILABLE = False
    get_lifecycle_manager = None  # type: ignore[assignment]
    reset_lifecycle_manager = None  # type: ignore[assignment]

# -- Backup Manager (S121) --
try:
    from opti_oignon.backup_manager import (
        BACKUP_AVAILABLE,
        backup_manager,
    )
    BACKUP_AVAILABLE = BACKUP_AVAILABLE  # re-export
except ImportError:
    BACKUP_AVAILABLE = False
    backup_manager = None


# -- Context Optimizer (S123) --
try:
    from opti_oignon.context_optimizer import (
        ContextOptimizer,
    )
    from opti_oignon.context_optimizer import (
        get_optimizer as get_context_optimizer,
    )
    from opti_oignon.context_optimizer import (
        init_optimizer as init_context_optimizer,
    )
    CONTEXT_OPTIMIZER_AVAILABLE = True
except ImportError:
    CONTEXT_OPTIMIZER_AVAILABLE = False
    get_context_optimizer = None
    init_context_optimizer = None
    ContextOptimizer = None


def get_ollama_models() -> list:
    """Retrieve the list of available Ollama models.

    Uses the backend registry if available (S105), otherwise falls back
    to direct ollama library call.

    Returns:
        List of model objects, or empty list if unavailable.
    """
    # Try backend abstraction first (S105)
    if INFERENCE_BACKEND_AVAILABLE and get_backend_registry is not None:
        try:
            registry = get_backend_registry()
            backend = registry.get_backend("ollama")
            if backend is not None:
                models = backend.list_models()
                return models or []
        except Exception as e:
            logger.debug(f"Backend registry Ollama list failed: {e}")

    # Fallback: direct ollama library call
    try:
        import ollama
        response = ollama.list()
        # ollama-python >= 0.4: ListResponse with .models attribute
        if hasattr(response, "models"):
            return response.models or []
        # Older versions: dict with "models" key
        if isinstance(response, dict):
            return response.get("models", [])
        return list(response) if response else []
    except Exception as e:
        logger.debug(f"Ollama unavailable: {e}")
        return []


# -- Security Mode (S126) --
try:
    from opti_oignon.security_mode import security_mode_manager
    SECURITY_MODE_AVAILABLE = True
except ImportError:
    SECURITY_MODE_AVAILABLE = False
    security_mode_manager = None


# -- Plugin Allowlist (S126) --
try:
    from opti_oignon.plugin_allowlist import plugin_allowlist_manager
    PLUGIN_ALLOWLIST_AVAILABLE = True
except ImportError:
    PLUGIN_ALLOWLIST_AVAILABLE = False
    plugin_allowlist_manager = None


# -- DB Encryption / SQLCipher (S126) --
try:
    from opti_oignon.db_encryption import SQLCIPHER_AVAILABLE, get_encrypted_connection
    DB_ENCRYPTION_AVAILABLE = True
except ImportError:
    DB_ENCRYPTION_AVAILABLE = False
    SQLCIPHER_AVAILABLE = False
    get_encrypted_connection = None


# -- Search Kill Switch (S126) --
try:
    from opti_oignon.search_killswitch import search_killswitch
    SEARCH_KILLSWITCH_AVAILABLE = True
except ImportError:
    SEARCH_KILLSWITCH_AVAILABLE = False
    search_killswitch = None


# -- Two-Factor Authentication (S126) --
try:
    from opti_oignon.auth_2fa import two_factor_manager
    TWO_FA_AVAILABLE = True
except ImportError:
    TWO_FA_AVAILABLE = False
    two_factor_manager = None


# -- Secure Bytes / Key Memory Protection (S126) --
try:
    from opti_oignon.secure_bytes import SecureBytes
    from opti_oignon.secure_bytes import get_platform_info as secure_bytes_info
    SECURE_BYTES_AVAILABLE = True
except ImportError:
    SECURE_BYTES_AVAILABLE = False
    SecureBytes = None
    secure_bytes_info = None


# -- Per-User Key Manager (S142) --
try:
    from opti_oignon.user_key_manager import (
        UserKeyManager,
        get_user_key_manager,
    )
    USER_KEY_MANAGER_AVAILABLE = True
except ImportError:
    USER_KEY_MANAGER_AVAILABLE = False
    get_user_key_manager = None
    UserKeyManager = None


# -- RBAC Enforcement (S142) --
try:
    from opti_oignon.rbac_enforcement import RBAC_ENFORCEMENT_AVAILABLE
except ImportError:
    RBAC_ENFORCEMENT_AVAILABLE = False


# -- Admin Audit (S142) --
try:
    from opti_oignon.admin_audit import (
        ADMIN_AUDIT_AVAILABLE,
        get_admin_audit_store,
    )
except ImportError:
    ADMIN_AUDIT_AVAILABLE = False
    get_admin_audit_store = None


# -- User Data Manager (S142) --
try:
    from opti_oignon.user_data_manager import (
        USER_DATA_MANAGER_AVAILABLE,
        get_user_data_deleter,
        get_user_data_exporter,
    )
except ImportError:
    USER_DATA_MANAGER_AVAILABLE = False
    get_user_data_exporter = None
    get_user_data_deleter = None


# -- Plugin User Config (S142) --
try:
    from opti_oignon.plugin_user_config import (
        PLUGIN_USER_CONFIG_AVAILABLE,
        get_plugin_user_config_store,
    )
except ImportError:
    PLUGIN_USER_CONFIG_AVAILABLE = False
    get_plugin_user_config_store = None
