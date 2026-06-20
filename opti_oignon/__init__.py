#!/usr/bin/env python3
"""
OPTI-OIGNON - Local LLM Optimization Suite
==========================================

A comprehensive framework for optimizing local LLMs running on Ollama,
featuring intelligent routing, RAG, and multi-agent orchestration.

Features:
    - Intelligent task detection and model routing
    - Customizable system prompts per task type
    - RAG (Retrieval-Augmented Generation) integration
    - Multi-agent pipeline orchestration
    - Dynamic pipeline planning
    - Custom pipeline management
    - Conversation history and export
    - FastAPI REST backend + SvelteKit frontend

Usage:
    # Start the API
    python -m opti_oignon api

    # Or import components
    from opti_oignon import analyzer, router, executor

Author: Léon
Version: see __version__.py
License: MIT
"""

from .__version__ import __version__

__author__ = "Léon"
__license__ = "MIT"

# Core components
from .analyzer import AnalysisResult, analyze, analyzer
from .config import CONFIG_DIR, DATA_DIR, config
from .executor import execute, executor, get_prompt
from .history import history
from .presets import Preset, preset_manager
from .router import RoutingResult, router

# Pipeline Manager (new in 1.2.0)
try:
    from .pipeline_manager import (
        Pipeline,
        PipelineStep,
        get_pipeline_manager,
    )
    PIPELINE_MANAGER_AVAILABLE = True
except ImportError:
    PIPELINE_MANAGER_AVAILABLE = False
    get_pipeline_manager = None
    Pipeline = None
    PipelineStep = None

# S216 (PIP-06 lot): the Gradio-era dynamic_pipeline_ui shim and the unwired
# dynamic_planning module (DPL-01) are retired. Execution pipelines run via
# opti_oignon.pipelines.PipelineRunner, wired into the chat path.

# Context Summarization (v1.4.0 — F2)
try:
    from .context_summary import ContextSummarizer, context_summarizer
    CONTEXT_SUMMARY_AVAILABLE = True
except ImportError:
    CONTEXT_SUMMARY_AVAILABLE = False
    context_summarizer = None
    ContextSummarizer = None

# Cross-Conversation Memory (v1.4.0 — F1)
try:
    from .memory import MemoryFact, MemoryManager, memory_manager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    memory_manager = None
    MemoryManager = None
    MemoryFact = None

# Code Execution (v1.4.0 — F3)
try:
    from .code_executor import (
        CodeBlock,
        CodeExecutor,
        ExecutionResult,
        code_executor,
    )
    CODE_EXECUTOR_AVAILABLE = True
except ImportError:
    CODE_EXECUTOR_AVAILABLE = False
    code_executor = None
    CodeExecutor = None
    CodeBlock = None
    ExecutionResult = None

# Response Cache (v1.4.0 — S18/C3)
try:
    from .response_cache import (
        CacheEntry,
        CacheStats,
        ResponseCache,
        response_cache,
    )
    RESPONSE_CACHE_AVAILABLE = True
except ImportError:
    RESPONSE_CACHE_AVAILABLE = False
    response_cache = None
    ResponseCache = None
    CacheEntry = None
    CacheStats = None

# Semantic Similarity Cache (v1.4.0 — S23 G1)
try:
    from .semantic_cache import (
        SemanticCache,
        SemanticCacheStats,
        SemanticMatch,
        cosine_similarity,
        semantic_cache,
    )
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False
    semantic_cache = None
    SemanticCache = None
    SemanticMatch = None
    SemanticCacheStats = None

# Lazy Loader (v1.4.0 — S23 F1)
try:
    from .lazy_loader import (
        LazyModule,
        get_lazy_stats,
        lazy_import,
        preload,
    )
    LAZY_LOADER_AVAILABLE = True
except ImportError:
    LAZY_LOADER_AVAILABLE = False
    lazy_import = None
    LazyModule = None
    get_lazy_stats = None

# Model Warm-up / Keepalive (v1.4.0 — S24 F2)
try:
    from .model_warmup import (
        MODEL_WARMUP_AVAILABLE,
        LoadedModel,
        ModelWarmup,
        WarmupResult,
        WarmupStats,
        model_warmup,
    )
except ImportError:
    MODEL_WARMUP_AVAILABLE = False
    model_warmup = None
    ModelWarmup = None
    WarmupResult = None
    WarmupStats = None
    LoadedModel = None

# Performance Benchmarks (v1.4.0 — S25 H2)
try:
    from .performance_benchmark import (
        BENCHMARK_AVAILABLE,
        BenchmarkRunner,
        BenchmarkSuite,
        benchmark_runner,
    )
    from .performance_benchmark import (
        BenchmarkResult as BenchmarkResultClass,
    )
    from .performance_benchmark import (
        run_all as run_benchmarks,
    )
except ImportError:
    BENCHMARK_AVAILABLE = False
    benchmark_runner = None
    BenchmarkRunner = None
    BenchmarkResultClass = None
    BenchmarkSuite = None
    run_benchmarks = None

# Inference backend abstraction (v2.0 -- S105)
try:
    from .inference_backend import (
        BackendRegistry,
        InferenceBackend,
        get_backend_registry,
        init_backends_from_config,
    )
    INFERENCE_BACKEND_AVAILABLE = True
except ImportError:
    INFERENCE_BACKEND_AVAILABLE = False
    BackendRegistry = None
    InferenceBackend = None
    get_backend_registry = None
    init_backends_from_config = None

# GGUF model manager (v2.0 -- S105)
try:
    from .model_manager import (
        ModelManager,
        get_model_manager,
        init_model_manager,
        parse_gguf_header,
    )
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    ModelManager = None
    get_model_manager = None
    init_model_manager = None
    parse_gguf_header = None

# FastAPI API (v1.4.0 -- S26)
try:
    from .api.app import app as api_app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    api_app = None

# Context Optimizer (v2.4.0 -- S123)
try:
    from .context_optimizer import (
        ContextOptimizer,
        OptimizedContext,
        OptimizationReport,
        get_optimizer as get_context_optimizer,
        init_optimizer as init_context_optimizer,
    )
    CONTEXT_OPTIMIZER_AVAILABLE = True
except ImportError:
    CONTEXT_OPTIMIZER_AVAILABLE = False
    ContextOptimizer = None
    OptimizedContext = None
    OptimizationReport = None
    get_context_optimizer = None
    init_context_optimizer = None

# Convenience exports
__all__ = [
    # Version info
    "__version__",
    "__author__",

    # Configuration
    "config",
    "DATA_DIR",
    "CONFIG_DIR",

    # Core components
    "analyzer",
    "analyze",
    "AnalysisResult",
    "router",
    "RoutingResult",
    "executor",
    "execute",
    "get_prompt",
    "preset_manager",
    "Preset",
    "history",

    # Pipeline Manager
    "PIPELINE_MANAGER_AVAILABLE",
    "get_pipeline_manager",
    "Pipeline",
    "PipelineStep",

    # Context Summary
    "CONTEXT_SUMMARY_AVAILABLE",
    "context_summarizer",
    "ContextSummarizer",

    # Memory
    "MEMORY_AVAILABLE",
    "memory_manager",
    "MemoryManager",
    "MemoryFact",

    # Code Execution
    "CODE_EXECUTOR_AVAILABLE",
    "code_executor",
    "CodeExecutor",
    "CodeBlock",
    "ExecutionResult",

    # Response Cache
    "RESPONSE_CACHE_AVAILABLE",
    "response_cache",
    "ResponseCache",
    "CacheEntry",
    "CacheStats",

    # Semantic Cache (S23 G1)
    "SEMANTIC_CACHE_AVAILABLE",
    "semantic_cache",
    "SemanticCache",
    "SemanticMatch",
    "SemanticCacheStats",
    "cosine_similarity",

    # Lazy Loader (S23 F1)
    "LAZY_LOADER_AVAILABLE",
    "lazy_import",
    "LazyModule",
    "get_lazy_stats",

    # Model Warm-up (S24 F2)
    "MODEL_WARMUP_AVAILABLE",
    "model_warmup",
    "ModelWarmup",
    "WarmupResult",
    "WarmupStats",
    "LoadedModel",

    # Performance Benchmarks (S25 H2)
    "BENCHMARK_AVAILABLE",
    "benchmark_runner",
    "BenchmarkRunner",
    "BenchmarkResultClass",
    "BenchmarkSuite",
    "run_benchmarks",

    # Inference backend (S105)
    "INFERENCE_BACKEND_AVAILABLE",
    "InferenceBackend",
    "BackendRegistry",
    "get_backend_registry",
    "init_backends_from_config",

    # GGUF model manager (S105)
    "MODEL_MANAGER_AVAILABLE",
    "ModelManager",
    "get_model_manager",
    "init_model_manager",
    "parse_gguf_header",

    # FastAPI API (S26)
    "API_AVAILABLE",
    "api_app",
]


def main():
    """Main entry point - launches the API server."""
    from .ui import launch
    launch()
