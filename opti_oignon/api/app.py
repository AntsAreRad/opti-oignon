#!/usr/bin/env python3
"""
Main FastAPI application for Opti-Oignon.

Provides a REST API for the local LLM optimization suite,
served via uvicorn (default port 8001).

S134: Heavy module imports are deferred via lazy_loader in deps.py.
Route modules are thin wrappers and load eagerly so all endpoints are
available immediately. The actual heavy dependencies (chromadb, etc.)
only import on first access to the relevant API endpoint.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from opti_oignon.__version__ import __version__

from .routes_agent import router as agent_router
from .routes_agent_eval import router as agent_eval_router
from .routes_answer_verification import answer_verification_router
from .routes_artifacts import router as artifacts_router
from .routes_auth import router as auth_router
from .routes_backends import router as backends_router
from .routes_backup import router as backup_router
from .routes_benchmark import router as benchmark_dashboard_router
from .routes_benchmark_v2 import router as benchmark_v2_router
from .routes_branches import router as branches_router
from .routes_cache import router as cache_router
from .routes_cascading import router as cascading_router
from .routes_chat import router as chat_router
from .routes_citation_verification import citation_verification_router
from .routes_claim_verification import claim_verification_router
from .routes_code import router as code_router
from .routes_coding import router as coding_router
from .routes_compression import router as compression_router
from .routes_context import router as context_router
from .routes_context_optimizer import router as context_optimizer_router
from .routes_conversations import router as conversations_router
from .routes_exec_pipelines import router as exec_pipelines_router
from .routes_export import router as export_router
from .routes_feedback import router as feedback_router
from .routes_files import router as files_router
from .routes_fine_tune import router as fine_tune_router
from .routes_governor import router as governor_router
from .routes_health import router as health_router
from .routes_humanizer import router as humanizer_router
from .routes_learned_routing import router as learned_routing_router
from .routes_live_metrics import router as live_metrics_router
from .routes_memory import memories_router
from .routes_memory import router as memory_router
from .routes_model_lifecycle import router as model_lifecycle_router
from .routes_models import router as models_router
from .routes_network import router as network_router
from .routes_note_actions import note_actions_router
from .routes_note_updates import note_updates_router
from .routes_notes import notes_router
from .routes_notes_attachments import notes_attachments_router
from .routes_notes_caption import notes_caption_router
from .routes_notes_transcription import notes_transcription_router
from .routes_performance import router as performance_router
from .routes_pipelines import router as pipelines_router
from .routes_plugin_marketplace import router as plugin_marketplace_router
from .routes_plugins import router as plugins_router
from .routes_presets import router as presets_router
from .routes_profiler import router as profiler_router
from .routes_projects import router as projects_router
from .routes_prompt import router as prompt_router
from .routes_rag import router as rag_router
from .routes_rag_dashboard import router as rag_dashboard_router
from .routes_sandbox import router as sandbox_router
from .routes_search import router as search_router
from .routes_security import router as security_router
from .routes_settings import router as settings_router
from .routes_smart_routing import router as smart_routing_router
from .routes_speculative import router as speculative_router
from .routes_speculative_decoding import router as speculative_decoding_router
from .routes_sync import router as sync_router
from .routes_system_presets import router as system_presets_router
from .routes_telemetry import router as telemetry_router
from .routes_tuner import router as tuner_router
from .routes_users import router as users_router
from .routes_vision import router as vision_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup: singletons initialize on import; heavy deps lazy-loaded (S134)
    logger.info("Opti-Oignon API started")
    # Veilid sync auto-driver: armed only when explicitly opted in
    # (OPTI_SYNC_AUTORUN); a no-op otherwise, and never breaks startup. The
    # mode boundary (Bulbe hard-stop) is enforced on every pass inside the
    # driver, not here.
    from opti_oignon.veilid.sync_service import arm_if_enabled
    arm_if_enabled()
    # One-shot legacy -> store memory migration (M3a-startup). Idempotent and
    # marker-guarded (a no-op after the first successful pass), and it never
    # raises -- a migration problem must not break the boot.
    try:
        from ..memory.migration import run_boot_migration
        run_boot_migration()
    except Exception:  # noqa: BLE001 - startup must not break on import/call
        logger.warning("boot: legacy memory migration call failed", exc_info=True)
    # Load all enabled plugins and register their hooks, so plugin effects apply
    # during inference AFTER a restart -- not only right after enable_plugin().
    # (The registry persists "enabled" state; the loaded instances do not, so
    # without this the hooks are never re-registered on boot.) Guarded: a plugin
    # load failure must not break the boot.
    try:
        from opti_oignon.plugin_loader import plugin_loader
        if plugin_loader is not None:
            _loaded = plugin_loader.load_all_enabled()
            logger.info("plugins: loaded %d enabled plugin(s) at startup", len(_loaded))
    except Exception:  # noqa: BLE001 - startup must not break on plugin loading
        logger.warning("plugins: load_all_enabled at startup failed", exc_info=True)
    yield
    # Shutdown: stop the sync driver if it was armed (a no-op otherwise).
    from opti_oignon.veilid.sync_service import reset_sync_service
    reset_sync_service()
    # Shut down loaded plugins (unload + stop the subprocess watchdog). Defensive.
    try:
        from opti_oignon.plugin_loader import plugin_loader
        if plugin_loader is not None:
            plugin_loader.shutdown_all()
    except Exception:  # noqa: BLE001 - shutdown is defensive
        logger.debug("plugins: shutdown_all failed", exc_info=True)
    logger.info("Opti-Oignon API stopped")


app = FastAPI(
    title="Opti-Oignon API",
    description="REST API for the Opti-Oignon local LLM optimization suite",
    version=__version__,
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Security headers middleware (S124 Phase 4)
#
# Registered BEFORE CORS so that security headers are added to every
# response without overwriting CORS headers.
# ---------------------------------------------------------------------------
try:
    from .security_middleware import SecurityHeadersMiddleware
    app.add_middleware(SecurityHeadersMiddleware)
    logger.info("Security headers middleware registered")
except Exception as _sec_exc:
    logger.warning("Failed to register security headers middleware: %s", _sec_exc)

# ---------------------------------------------------------------------------
# Content Security Policy middleware (S155)
#
# Nonce-based strict CSP with report-only mode (default).
# Generates a unique nonce per request, restricts connect-src to localhost.
# CSP violation reporting via /api/csp-report endpoint.
# ---------------------------------------------------------------------------
try:
    from opti_oignon.middleware.csp import CSPMiddleware as _CSPMiddleware
    app.add_middleware(_CSPMiddleware)
    logger.info("CSP middleware registered")
except Exception as _csp_exc:
    logger.warning("Failed to register CSP middleware: %s", _csp_exc)

# ---------------------------------------------------------------------------
# Security mode middleware (S127)
#
# Enforces Daily/Bulbe mode restrictions: blocks search when kill switch
# is engaged, rejects Bearer tokens in Bulbe, enforces plugin allowlist,
# sets SameSite=Strict cookies.  Degrades gracefully if security_mode
# module is unavailable.
# ---------------------------------------------------------------------------
try:
    from .security_mode_middleware import SecurityModeMiddleware
    app.add_middleware(SecurityModeMiddleware)
    logger.info("Security mode middleware registered")
except Exception as _mode_exc:
    logger.warning("Failed to register security mode middleware: %s", _mode_exc)

# ---------------------------------------------------------------------------
# CSRF middleware (S136 audit fix)
#
# Validates double-submit cookie on all POST/PUT/DELETE/PATCH requests.
# Previously _validate_csrf() was defined but never called -- this
# middleware enforces it globally.
# ---------------------------------------------------------------------------
try:
    from .csrf_middleware import CSRFMiddleware
    app.add_middleware(CSRFMiddleware)
    logger.info("CSRF middleware registered")
except Exception as _csrf_exc:
    logger.warning("Failed to register CSRF middleware: %s", _csrf_exc)

# ---------------------------------------------------------------------------
# Global auth middleware (S136 audit fix)
#
# Deny-by-default: all endpoints require authentication except an
# explicit allowlist (login, register, health, OpenAPI).  This replaces
# the fragile per-router Depends() approach that left 37+ routers
# unprotected.
# ---------------------------------------------------------------------------
try:
    from .auth_middleware import AuthMiddleware
    app.add_middleware(AuthMiddleware)
    logger.info("Global auth middleware registered (deny-by-default)")
except Exception as _auth_mw_exc:
    logger.warning("Failed to register auth middleware: %s", _auth_mw_exc)

# ---------------------------------------------------------------------------
# CORS configuration (S124 — hardened)
#
# Default: localhost-only origins. Credentials are allowed ONLY when origins
# are explicitly listed (never with wildcard).  The env var
# OPTI_CORS_ORIGINS overrides; set to "*" for wide-open (credentials will
# be forced OFF to prevent cookie/token theft from arbitrary sites).
# ---------------------------------------------------------------------------
import os as _os
import re as _re

_LOCALHOST_ORIGINS: list[str] = [
    "http://localhost",
    "http://127.0.0.1",
    "http://[::1]",
]

# Match localhost with any port (e.g. http://localhost:5173)
_LOCALHOST_RE = _re.compile(
    r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
)


def _resolve_cors_origins() -> tuple[list[str], bool]:
    """Determine CORS origins and whether credentials should be allowed.

    Returns (origins_list, allow_credentials).

    Security rules:
    - Empty / unset env var -> localhost-only, credentials ON
    - Explicit origins list -> those origins, credentials ON
    - "*" -> wildcard, credentials OFF (browser will not send tokens)
    - security.yaml cors.origins takes precedence over env var
    """
    # Try security.yaml first
    yaml_origins: list[str] | None = None
    try:
        import yaml as _yaml
        _sec_path = _os.path.join(
            _os.path.dirname(__file__), "..", "config", "security.yaml"
        )
        if _os.path.isfile(_sec_path):
            with open(_sec_path, encoding="utf-8") as _fh:
                _sec_cfg = _yaml.safe_load(_fh) or {}
            cors_cfg = _sec_cfg.get("cors", {})
            if isinstance(cors_cfg, dict) and "origins" in cors_cfg:
                raw = cors_cfg["origins"]
                if isinstance(raw, list):
                    yaml_origins = [str(o).strip() for o in raw if str(o).strip()]
                elif isinstance(raw, str):
                    raw = raw.strip()
                    if raw == "*":
                        yaml_origins = ["*"]
                    elif raw:
                        yaml_origins = [o.strip() for o in raw.split(",") if o.strip()]
    except Exception:
        pass  # YAML loading is best-effort

    # Env var (lower priority)
    env_val = _os.environ.get("OPTI_CORS_ORIGINS", "").strip()

    # Resolve: yaml (if non-empty) > env > default (localhost)
    if yaml_origins:
        origins = yaml_origins
    elif env_val:
        if env_val == "*":
            origins = ["*"]
        else:
            origins = [o.strip() for o in env_val.split(",") if o.strip()]
    else:
        origins = []  # Will use localhost defaults

    # Wildcard handling: credentials MUST be OFF
    if "*" in origins:
        logger.warning(
            "CORS origins set to wildcard (*). Credentials are DISABLED to "
            "prevent token theft. Set explicit origins for credential support."
        )
        return ["*"], False

    # No origins specified -> default to localhost
    if not origins:
        origins = list(_LOCALHOST_ORIGINS)

    # Always include localhost variants for dev convenience
    for lo in _LOCALHOST_ORIGINS:
        if lo not in origins:
            origins.append(lo)

    return origins, True


_cors_origins, _cors_credentials = _resolve_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=_LOCALHOST_RE.pattern if _cors_credentials else None,
)

# Router registration
app.include_router(conversations_router)
app.include_router(models_router)
app.include_router(chat_router)
app.include_router(artifacts_router)
app.include_router(code_router)
app.include_router(memory_router)
app.include_router(memories_router)
app.include_router(notes_router)
app.include_router(note_actions_router)
app.include_router(claim_verification_router)
app.include_router(answer_verification_router)
app.include_router(citation_verification_router)
app.include_router(notes_attachments_router)
app.include_router(notes_transcription_router)
app.include_router(notes_caption_router)
app.include_router(note_updates_router)
app.include_router(cache_router)
app.include_router(health_router)
app.include_router(files_router)
app.include_router(export_router)
app.include_router(presets_router)
app.include_router(pipelines_router)
app.include_router(settings_router)
app.include_router(context_router)
app.include_router(exec_pipelines_router)
app.include_router(smart_routing_router)
app.include_router(feedback_router)
app.include_router(projects_router)
app.include_router(prompt_router)
app.include_router(compression_router)
app.include_router(learned_routing_router)
app.include_router(cascading_router)
app.include_router(speculative_router)
app.include_router(benchmark_dashboard_router)
app.include_router(network_router)
app.include_router(performance_router)
app.include_router(sandbox_router)
app.include_router(coding_router)
app.include_router(search_router)
app.include_router(system_presets_router)
app.include_router(humanizer_router)
app.include_router(benchmark_v2_router)
app.include_router(vision_router)
app.include_router(fine_tune_router)
app.include_router(branches_router)
app.include_router(auth_router)
app.include_router(rag_router)
app.include_router(rag_dashboard_router)
app.include_router(plugin_marketplace_router)
app.include_router(plugins_router)
app.include_router(backends_router)
app.include_router(speculative_decoding_router)
app.include_router(tuner_router)
app.include_router(live_metrics_router)
app.include_router(model_lifecycle_router)
app.include_router(telemetry_router)
app.include_router(profiler_router)
app.include_router(backup_router)
app.include_router(context_optimizer_router)
# Sandboxed agent loop: status / cancel / run / WebSocket event stream (S177,
# Theme 3 / Odysseus Core). Guarded so a partial agent build cannot block app
# startup; Bulbe approvals reuse /api/security/tool-approval/*.
if agent_router is not None:
    app.include_router(agent_router)
# Veilid sync: list / status / watermark / run a pull round (S180, Theme 4 /
# Veilid Sync). Guarded so a partial veilid build cannot block app startup; the
# run surface is Daily-only and refuses under Bulbe via the binding-layer gate,
# and a sensitive apply reuses /api/security/tool-approval/* through the engine.
if sync_router is not None:
    app.include_router(sync_router)
# Resource governor: status / admissions / evict / config (S227, Resource
# Governor cycle Bloc 4). Guarded so a constrained build cannot block app
# startup; mode-free (it behaves identically in Daily and Bulbe).
if governor_router is not None:
    app.include_router(governor_router)
# Agent eval harness: run / status / results / history / cancel (S230, AGT
# cycle Lot 3). Guarded so a partial agent_eval build cannot block app
# startup; the runner consumes the Lot 1/2 agent surface read-only and the
# admission contract rides the resource governor when present.
if agent_eval_router is not None:
    app.include_router(agent_eval_router)
app.include_router(security_router)
app.include_router(users_router)

# CSP violation report endpoint (S155)
try:
    from opti_oignon.middleware.csp import csp_router as _csp_router
    if _csp_router is not None:
        app.include_router(_csp_router)
        logger.info("CSP report endpoint registered")
except Exception as _csp_route_exc:
    logger.warning("Failed to register CSP report endpoint: %s", _csp_route_exc)


@app.get("/api/health", tags=["health"])
def health_check():
    """Basic health check endpoint."""
    from .deps import (
        ANALYTICS_AVAILABLE,
        ARTIFACT_AVAILABLE,
        AUTH_AVAILABLE,
        AUTO_TRIGGER_AVAILABLE,
        BENCHMARK_AVAILABLE,
        BENCHMARK_HISTORY_AVAILABLE,
        BENCHMARK_JUDGE_AVAILABLE,
        BENCHMARK_RECOMMENDATIONS_AVAILABLE,
        BENCHMARK_V2_AVAILABLE,
        BRANCHES_AVAILABLE,
        CASCADING_AVAILABLE,
        CODE_EXECUTOR_AVAILABLE,
        CODING_AGENT_AVAILABLE,
        CONFIG_AVAILABLE,
        CONTEXT_OPTIMIZER_AVAILABLE,
        CONTEXT_WINDOW_AVAILABLE,
        CONVERSATION_AVAILABLE,
        CONVERSATION_COMPRESSOR_AVAILABLE,
        CUSTOM_PROFILES_AVAILABLE,
        EXTERNAL_STORES_AVAILABLE,
        FEEDBACK_AVAILABLE,
        FILE_TOOLS_AVAILABLE,
        FINE_TUNE_EXPORT_AVAILABLE,
        FINE_TUNE_TRACKER_AVAILABLE,
        FINGERPRINT_AVAILABLE,
        HUMANIZER_AVAILABLE,
        HYBRID_SEARCH_AVAILABLE,
        INFERENCE_BACKEND_AVAILABLE,
        INFERENCE_PROFILER_AVAILABLE,
        LEARNED_ROUTER_AVAILABLE,
        MEMORY_AVAILABLE,
        MODEL_HEALTH_AVAILABLE,
        MODEL_MANAGER_AVAILABLE,
        MODEL_WARMUP_AVAILABLE,
        NETWORK_MANAGER_AVAILABLE,
        PERFORMANCE_MONITOR_AVAILABLE,
        PII_SANITIZER_AVAILABLE,
        PIPELINE_AVAILABLE,
        PLUGIN_HOOKS_AVAILABLE,
        PLUGIN_INDEX_AVAILABLE,
        PLUGIN_INSTALLER_AVAILABLE,
        PLUGIN_LOADER_AVAILABLE,
        PLUGIN_REGISTRY_AVAILABLE,
        PLUGIN_REVIEWS_AVAILABLE,
        PLUGIN_TEMPLATE_AVAILABLE,
        PRE_CACHE_AVAILABLE,
        PRESET_AVAILABLE,
        PROFILE_AVAILABLE,
        PROJECT_CONTEXT_AVAILABLE,
        PROJECT_TRIGGERS_AVAILABLE,
        PROJECTS_AVAILABLE,
        PROMPT_OPTIMIZATION_AVAILABLE,
        RAG_CHUNKER_AVAILABLE,
        RAG_DASHBOARD_AVAILABLE,
        RAG_STORE_AVAILABLE,
        RESPONSE_CACHE_AVAILABLE,
        SANDBOX_AVAILABLE,
        SANDBOX_TOOLS_AVAILABLE,
        SEMANTIC_CACHE_AVAILABLE,
        SMART_ROUTER_AVAILABLE,
        SPECULATIVE_AVAILABLE,
        SYNC_QUEUE_AVAILABLE,
        SYSTEM_PRESETS_AVAILABLE,
        TELEMETRY_AVAILABLE,
        USER_SETTINGS_AVAILABLE,
        WEB_SEARCH_AVAILABLE,
    )

    return {
        "status": "ok",
        "version": __version__,
        "modules": {
            "conversation": CONVERSATION_AVAILABLE,
            "presets": PRESET_AVAILABLE,
            "system_presets": SYSTEM_PRESETS_AVAILABLE,
            "memory": MEMORY_AVAILABLE,
            "artifacts": ARTIFACT_AVAILABLE,
            "code_executor": CODE_EXECUTOR_AVAILABLE,
            "response_cache": RESPONSE_CACHE_AVAILABLE,
            "semantic_cache": SEMANTIC_CACHE_AVAILABLE,
            "pipelines": PIPELINE_AVAILABLE,
            "benchmarks": BENCHMARK_AVAILABLE,
            "model_warmup": MODEL_WARMUP_AVAILABLE,
            "config": CONFIG_AVAILABLE,
            "model_profiles": PROFILE_AVAILABLE,
            "model_health": MODEL_HEALTH_AVAILABLE,
            "context_window": CONTEXT_WINDOW_AVAILABLE,
            "smart_router": SMART_ROUTER_AVAILABLE,
            "feedback": FEEDBACK_AVAILABLE,
            "analytics": ANALYTICS_AVAILABLE,
            "projects": PROJECTS_AVAILABLE,
            "project_context": PROJECT_CONTEXT_AVAILABLE,
            "project_triggers": PROJECT_TRIGGERS_AVAILABLE,
            "benchmark_history": BENCHMARK_HISTORY_AVAILABLE,
            "prompt_optimization": PROMPT_OPTIMIZATION_AVAILABLE,
            "conversation_compressor": CONVERSATION_COMPRESSOR_AVAILABLE,
            "learned_router": LEARNED_ROUTER_AVAILABLE,
            "cascading": CASCADING_AVAILABLE,
            "speculative": SPECULATIVE_AVAILABLE,
            "network_manager": NETWORK_MANAGER_AVAILABLE,
            "sync_queue": SYNC_QUEUE_AVAILABLE,
            "pre_cache": PRE_CACHE_AVAILABLE,
            "performance_monitor": PERFORMANCE_MONITOR_AVAILABLE,
            "sandbox": SANDBOX_AVAILABLE,
            "file_tools": FILE_TOOLS_AVAILABLE,
            "sandbox_tools": SANDBOX_TOOLS_AVAILABLE,
            "coding_agent": CODING_AGENT_AVAILABLE,
            "fingerprint": FINGERPRINT_AVAILABLE,
            "web_search": WEB_SEARCH_AVAILABLE,
            "pii_sanitizer": PII_SANITIZER_AVAILABLE,
            "humanizer": HUMANIZER_AVAILABLE,
            "benchmark_v2": BENCHMARK_V2_AVAILABLE,
            "benchmark_judge": BENCHMARK_JUDGE_AVAILABLE,
            "benchmark_recommendations": BENCHMARK_RECOMMENDATIONS_AVAILABLE,
            "auto_trigger": AUTO_TRIGGER_AVAILABLE,
            "custom_profiles": CUSTOM_PROFILES_AVAILABLE,
            "fine_tune_export": FINE_TUNE_EXPORT_AVAILABLE,
            "fine_tune_tracker": FINE_TUNE_TRACKER_AVAILABLE,
            "branches": BRANCHES_AVAILABLE,
            "auth": AUTH_AVAILABLE,
            "user_settings": USER_SETTINGS_AVAILABLE,
            "rag_store": RAG_STORE_AVAILABLE,
            "rag_chunker": RAG_CHUNKER_AVAILABLE,
            "hybrid_search": HYBRID_SEARCH_AVAILABLE,
            "external_stores": EXTERNAL_STORES_AVAILABLE,
            "rag_dashboard": RAG_DASHBOARD_AVAILABLE,
            "plugin_registry": PLUGIN_REGISTRY_AVAILABLE,
            "plugin_loader": PLUGIN_LOADER_AVAILABLE,
            "plugin_hooks": PLUGIN_HOOKS_AVAILABLE,
            "plugin_index": PLUGIN_INDEX_AVAILABLE,
            "plugin_installer": PLUGIN_INSTALLER_AVAILABLE,
            "plugin_reviews": PLUGIN_REVIEWS_AVAILABLE,
            "plugin_template": PLUGIN_TEMPLATE_AVAILABLE,
            "inference_backend": INFERENCE_BACKEND_AVAILABLE,
            "model_manager": MODEL_MANAGER_AVAILABLE,
            "telemetry": TELEMETRY_AVAILABLE,
            "inference_profiler": INFERENCE_PROFILER_AVAILABLE,
            "context_optimizer": CONTEXT_OPTIMIZER_AVAILABLE,
        },
        # S124: Security and sandbox isolation status
        "security": _get_health_security_info(),
    }


def _get_health_security_info() -> dict:
    """Build security section for /api/health response (S124)."""
    info: dict = {}

    # Sandbox isolation status
    try:
        from .deps import SANDBOX_AVAILABLE
        if SANDBOX_AVAILABLE:
            from .deps import sandbox_manager
            if sandbox_manager is not None:
                info["sandbox"] = sandbox_manager.get_isolation_status()
    except Exception:
        info["sandbox"] = {"isolation_level": "unknown"}

    # CORS config summary
    info["cors"] = {
        "origins_count": len(_cors_origins),
        "wildcard": "*" in _cors_origins,
        "credentials": _cors_credentials,
    }

    # Security headers
    try:
        from .security_middleware import get_security_headers_config
        hdr_cfg = get_security_headers_config()
        info["headers_enabled"] = not hdr_cfg.get("_disabled", False)
    except Exception:
        info["headers_enabled"] = False

    # Rate limiting
    try:
        from opti_oignon.auth import login_rate_limiter
        info["rate_limiting_enabled"] = login_rate_limiter.enabled
    except Exception:
        info["rate_limiting_enabled"] = False

    # S157: Red team resistance summary
    try:
        from .routes_security import _redteam_report_store
        if _redteam_report_store:
            latest_id = max(_redteam_report_store.keys())
            latest = _redteam_report_store[latest_id]
            latest_score = latest.get("score", {})
            info["redteam"] = {
                "last_run_id": latest_id,
                "last_run_timestamp": latest.get("timestamp", ""),
                "bypass_rate": latest_score.get("overall_bypass_rate", 0),
                "detection_rate": latest_score.get("overall_detection_rate", 0),
                "total_reports": len(_redteam_report_store),
            }
        else:
            info["redteam"] = {
                "last_run_id": None,
                "warning": "No red team run recorded",
            }
    except Exception:
        info["redteam"] = {"available": False}

    # S158: Security scheduler summary
    try:
        from opti_oignon.security_scheduler import get_scheduler
        scheduler = get_scheduler()
        sched_status = scheduler.get_status()
        info["scheduler"] = {
            "enabled": sched_status.get("enabled", False),
            "running": sched_status.get("running", False),
            "redteam_interval": sched_status.get("redteam", {}).get("interval"),
            "redteam_run_count": sched_status.get("redteam", {}).get("run_count", 0),
            "dep_audit_count": sched_status.get("dep_audit", {}).get("audit_count", 0),
            "alerts_total": sched_status.get("alerts_total", 0),
            "quiet_hours_active": sched_status.get("quiet_hours", {}).get("active", False),
        }
    except Exception:
        info["scheduler"] = {"available": False}

    return info
