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

Every name this package exports is imported when it is first asked for,
not when the package is. Importing the package used to pull the API
application and, through it, most of the tree; a name now costs exactly
what it needs, and only the code that asks for it pays. The surface is the
same: the names in ``__all__``, the availability flags, ``main()``.

Author: Léon
Version: see __version__.py
License: MIT
"""

import importlib
import sys
import types
from pathlib import Path as _Path

from .__version__ import __version__

__author__ = "Léon"
__license__ = "MIT"

# Exported name -> (module, attribute it is imported from). Read by
# __getattr__ below on first access; nothing here imports anything.
_EXPORTS = {
    # Core components
    "analyzer": (".analyzer", "analyzer"),
    "analyze": (".analyzer", "analyze"),
    "AnalysisResult": (".analyzer", "AnalysisResult"),
    "config": (".config", "config"),
    "DATA_DIR": (".config", "DATA_DIR"),
    "CONFIG_DIR": (".config", "CONFIG_DIR"),
    "executor": (".executor", "executor"),
    "execute": (".executor", "execute"),
    "get_prompt": (".executor", "get_prompt"),
    "history": (".history", "history"),
    "preset_manager": (".presets", "preset_manager"),
    "Preset": (".presets", "Preset"),
    "router": (".router", "router"),
    "RoutingResult": (".router", "RoutingResult"),
    # Pipeline manager
    "get_pipeline_manager": (".pipeline_manager", "get_pipeline_manager"),
    "Pipeline": (".pipeline_manager", "Pipeline"),
    "PipelineStep": (".pipeline_manager", "PipelineStep"),
    # Context summary
    "context_summarizer": (".context_summary", "context_summarizer"),
    "ContextSummarizer": (".context_summary", "ContextSummarizer"),
    # Memory
    "memory_manager": (".memory", "memory_manager"),
    "MemoryManager": (".memory", "MemoryManager"),
    "MemoryFact": (".memory", "MemoryFact"),
    # Code executor
    "code_executor": (".code_executor", "code_executor"),
    "CodeExecutor": (".code_executor", "CodeExecutor"),
    "CodeBlock": (".code_executor", "CodeBlock"),
    "ExecutionResult": (".code_executor", "ExecutionResult"),
    # Response cache
    "response_cache": (".response_cache", "response_cache"),
    "ResponseCache": (".response_cache", "ResponseCache"),
    "CacheEntry": (".response_cache", "CacheEntry"),
    "CacheStats": (".response_cache", "CacheStats"),
    # Semantic cache
    "semantic_cache": (".semantic_cache", "semantic_cache"),
    "SemanticCache": (".semantic_cache", "SemanticCache"),
    "SemanticMatch": (".semantic_cache", "SemanticMatch"),
    "SemanticCacheStats": (".semantic_cache", "SemanticCacheStats"),
    "cosine_similarity": (".semantic_cache", "cosine_similarity"),
    # Lazy loader
    "lazy_import": (".lazy_loader", "lazy_import"),
    "LazyModule": (".lazy_loader", "LazyModule"),
    "get_lazy_stats": (".lazy_loader", "get_lazy_stats"),
    "preload": (".lazy_loader", "preload"),
    # Model warmup
    "model_warmup": (".model_warmup", "model_warmup"),
    "ModelWarmup": (".model_warmup", "ModelWarmup"),
    "WarmupResult": (".model_warmup", "WarmupResult"),
    "WarmupStats": (".model_warmup", "WarmupStats"),
    "LoadedModel": (".model_warmup", "LoadedModel"),
    "MODEL_WARMUP_AVAILABLE": (".model_warmup", "MODEL_WARMUP_AVAILABLE"),
    # Performance benchmark
    "benchmark_runner": (".performance_benchmark", "benchmark_runner"),
    "BenchmarkRunner": (".performance_benchmark", "BenchmarkRunner"),
    "BenchmarkSuite": (".performance_benchmark", "BenchmarkSuite"),
    "BenchmarkResultClass": (".performance_benchmark", "BenchmarkResult"),
    "run_benchmarks": (".performance_benchmark", "run_all"),
    "BENCHMARK_AVAILABLE": (".performance_benchmark", "BENCHMARK_AVAILABLE"),
    # Inference backend
    "InferenceBackend": (".inference_backend", "InferenceBackend"),
    "BackendRegistry": (".inference_backend", "BackendRegistry"),
    "get_backend_registry": (".inference_backend", "get_backend_registry"),
    "init_backends_from_config": (".inference_backend", "init_backends_from_config"),
    # Model manager
    "ModelManager": (".model_manager", "ModelManager"),
    "get_model_manager": (".model_manager", "get_model_manager"),
    "init_model_manager": (".model_manager", "init_model_manager"),
    "parse_gguf_header": (".model_manager", "parse_gguf_header"),
    # FastAPI API
    "api_app": (".api.app", "app"),
    # Context optimizer
    "ContextOptimizer": (".context_optimizer", "ContextOptimizer"),
    "OptimizedContext": (".context_optimizer", "OptimizedContext"),
    "OptimizationReport": (".context_optimizer", "OptimizationReport"),
    "get_context_optimizer": (".context_optimizer", "get_optimizer"),
    "init_context_optimizer": (".context_optimizer", "init_optimizer"),
}

# Availability flag -> the module whose import decides it. True when the
# module imports, False when it does not; computed when asked, like the
# guarded imports used to compute it at package import.
_FLAGS = {
    "PIPELINE_MANAGER_AVAILABLE": ".pipeline_manager",
    "CONTEXT_SUMMARY_AVAILABLE": ".context_summary",
    "MEMORY_AVAILABLE": ".memory",
    "CODE_EXECUTOR_AVAILABLE": ".code_executor",
    "RESPONSE_CACHE_AVAILABLE": ".response_cache",
    "SEMANTIC_CACHE_AVAILABLE": ".semantic_cache",
    "LAZY_LOADER_AVAILABLE": ".lazy_loader",
    "INFERENCE_BACKEND_AVAILABLE": ".inference_backend",
    "MODEL_MANAGER_AVAILABLE": ".model_manager",
    "API_AVAILABLE": ".api.app",
    "CONTEXT_OPTIMIZER_AVAILABLE": ".context_optimizer",
}

# Names whose module may be absent: the guarded groups of the former
# facade resolved them to None then. They still do.
_OPTIONAL = {name for name, (module, _attr) in _EXPORTS.items() if module not in (
    ".analyzer", ".config", ".executor", ".history", ".presets", ".router",
)}


# Exported names that are also submodule names: ``config`` the object and
# ``opti_oignon.config`` the module. Importing the submodule binds it on the
# package, ahead of any accessor; the former facade overwrote that binding
# with the object, and the module class below keeps doing so.
_COLLIDING = frozenset(
    name for name in _EXPORTS if (_Path(__file__).parent / f"{name}.py").exists()
)


def _import(module):
    return importlib.import_module(module, __name__)


def _resolve(name):
    """Import what ``name`` needs, bind the value on the package, return it."""
    if name in _FLAGS:
        try:
            _import(_FLAGS[name])
            value = True
        except ImportError:
            value = False
    elif name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        try:
            value = getattr(_import(module), attribute)
        except ImportError:
            if name not in _OPTIONAL:
                raise
            value = None
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


class _Facade(types.ModuleType):
    """The package's module type: exports resolve on first access.

    ``__getattr__`` serves a name that is not bound yet. ``__getattribute__``
    steps in for the colliding names only, when the binding found is the
    submodule the import system put there rather than the exported object.
    """

    def __getattribute__(self, name):
        if name in _COLLIDING:
            bound = types.ModuleType.__getattribute__(self, "__dict__").get(name)
            if isinstance(bound, types.ModuleType):
                return _resolve(name)
        return types.ModuleType.__getattribute__(self, name)

    def __getattr__(self, name):
        return _resolve(name)

    def __dir__(self):
        return sorted(set(types.ModuleType.__getattribute__(self, "__dict__")) | set(__all__))


sys.modules[__name__].__class__ = _Facade


# Convenience exports
__all__ = [
    # Version
    "__version__",
    "__author__",

    # Core
    "config",
    "DATA_DIR",
    "CONFIG_DIR",
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

    # Code Executor
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

    # Semantic Cache
    "SEMANTIC_CACHE_AVAILABLE",
    "semantic_cache",
    "SemanticCache",
    "SemanticMatch",
    "SemanticCacheStats",
    "cosine_similarity",

    # Lazy Loader
    "LAZY_LOADER_AVAILABLE",
    "lazy_import",
    "LazyModule",
    "get_lazy_stats",

    # Model Warmup
    "MODEL_WARMUP_AVAILABLE",
    "model_warmup",
    "ModelWarmup",
    "WarmupResult",
    "WarmupStats",
    "LoadedModel",

    # Performance Benchmark
    "BENCHMARK_AVAILABLE",
    "benchmark_runner",
    "BenchmarkRunner",
    "BenchmarkResultClass",
    "BenchmarkSuite",
    "run_benchmarks",

    # Inference Backend
    "INFERENCE_BACKEND_AVAILABLE",
    "InferenceBackend",
    "BackendRegistry",
    "get_backend_registry",
    "init_backends_from_config",

    # Model Manager
    "MODEL_MANAGER_AVAILABLE",
    "ModelManager",
    "get_model_manager",
    "init_model_manager",
    "parse_gguf_header",

    # FastAPI API
    "API_AVAILABLE",
    "api_app",
]


def main():
    """Main entry point - launches the API server."""
    from .ui import launch
    launch()
