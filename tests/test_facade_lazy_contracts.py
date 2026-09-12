#!/usr/bin/env python3
"""Contracts for the package facade: the same names, loaded when asked.

Importing the package pulled two hundred and thirty-two modules, because
its own __init__ imported the API application. The facade now exports the
same names through a module __getattr__: nothing of the project is
imported until a name is asked for, and asking for one imports only what
that name needs. The surface does not change; the moment it costs does.

  * LF1 -- importing the package in a fresh interpreter loads no project
    module beyond the version, and far fewer modules than it did.
  * LF2 -- every exported name is still there, in the same order, and
    resolves to the object its module defines; availability flags are
    computed when asked; an unknown name is an AttributeError; asking for
    one name does not import the others.
  * LF3 -- the static census agrees: the facade's eager closure is the
    version module alone; its runtime imports by name are invisible to a
    static census, and the contract says so rather than pretend.

Local-only (the public distribution ships no tests). LF1 runs the import
in a subprocess from an empty directory, so whatever the import creates
lands there and not in the tree.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO  # noqa: E402

# Below what the import used to cost (1634 modules measured before the
# rewrite) and above what it costs now, so the ceiling forbids a return.
MODULE_CEILING = 400

_EXPORTS = """__version__ __author__ config DATA_DIR CONFIG_DIR analyzer analyze AnalysisResult router
RoutingResult executor execute get_prompt preset_manager Preset history PIPELINE_MANAGER_AVAILABLE
get_pipeline_manager Pipeline PipelineStep CONTEXT_SUMMARY_AVAILABLE context_summarizer ContextSummarizer
MEMORY_AVAILABLE memory_manager MemoryManager MemoryFact CODE_EXECUTOR_AVAILABLE code_executor CodeExecutor
CodeBlock ExecutionResult RESPONSE_CACHE_AVAILABLE response_cache ResponseCache CacheEntry CacheStats
SEMANTIC_CACHE_AVAILABLE semantic_cache SemanticCache SemanticMatch SemanticCacheStats cosine_similarity
LAZY_LOADER_AVAILABLE lazy_import LazyModule get_lazy_stats MODEL_WARMUP_AVAILABLE model_warmup ModelWarmup
WarmupResult WarmupStats LoadedModel BENCHMARK_AVAILABLE benchmark_runner BenchmarkRunner BenchmarkResultClass
BenchmarkSuite run_benchmarks INFERENCE_BACKEND_AVAILABLE InferenceBackend BackendRegistry get_backend_registry
init_backends_from_config MODEL_MANAGER_AVAILABLE ModelManager get_model_manager init_model_manager
parse_gguf_header API_AVAILABLE api_app""".split()


# ---------------------------------------------------------------------------
# LF1 -- a fresh import loads nothing of the project
# ---------------------------------------------------------------------------
def test_lf1_a_fresh_import_loads_no_project_module_beyond_the_version(tmp_path):
    code = (
        "import sys, json; import opti_oignon; "
        "print(json.dumps({'project': sorted(k for k in sys.modules if k.startswith('opti_oignon')), "
        "'count': len(sys.modules), 'version': opti_oignon.__version__}))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-800:]
    import json
    seen = json.loads(result.stdout.strip().splitlines()[-1])
    assert seen["project"] == ["opti_oignon", "opti_oignon.__version__"], seen["project"]
    assert seen["count"] <= MODULE_CEILING, f"{seen['count']} modules at import, ceiling {MODULE_CEILING}"
    assert seen["count"] >= 20, "control: the interpreter itself loads modules; a count this low is a broken measurement"
    assert isinstance(seen["version"], str) and seen["version"]
    assert not list(tmp_path.iterdir()), "the import creates nothing in the working directory"


# ---------------------------------------------------------------------------
# LF2 -- the surface is intact and lazy per name
# ---------------------------------------------------------------------------
def test_lf2_every_export_resolves_when_asked_and_only_then():
    import opti_oignon as oo

    assert list(oo.__all__) == _EXPORTS, "the same seventy-one names, in the same order"
    assert len(oo.__all__) == 71
    for name in oo.__all__:
        assert name in dir(oo), f"{name} is listed by dir()"
    with pytest.raises(AttributeError):
        oo.this_name_was_never_exported
    for key in ("opti_oignon.api.app", "opti_oignon.performance_benchmark"):
        sys.modules.pop(key, None)
    import importlib
    router_module = importlib.import_module("opti_oignon.router")
    assert oo.RoutingResult is router_module.RoutingResult
    backend_module = importlib.import_module("opti_oignon.inference_backend")
    assert oo.get_backend_registry is backend_module.get_backend_registry
    assert oo.INFERENCE_BACKEND_AVAILABLE is True and oo.MEMORY_AVAILABLE is True
    assert oo.config is sys.modules["opti_oignon.config"].config
    assert "opti_oignon.api.app" not in sys.modules, "asking for the router did not import the API application"
    assert "opti_oignon.performance_benchmark" not in sys.modules, "nor the benchmark runner"
    assert oo.__author__ and oo.__license__ == "MIT"
    assert callable(oo.main)


# ---------------------------------------------------------------------------
# LF3 -- the census agrees
# ---------------------------------------------------------------------------
def test_lf3_the_static_census_sees_a_facade_that_pulls_only_the_version():
    import importlib.util

    spec = importlib.util.spec_from_file_location("core_census", REPO / "scripts" / "core_census.py")
    core_census = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core_census)
    report = core_census.census(REPO / "opti_oignon")
    facade = report["modules"]["opti_oignon"]
    assert facade["closure_eager"] == ["opti_oignon.__version__"]
    assert facade["imports_eager"] == ["opti_oignon.__version__"]
    # The exports are imported by name at runtime, which a static census
    # cannot see: the lazy edges of the facade are invisible to it by
    # construction, and that is recorded here rather than hidden.
    assert facade["imports_lazy"] == ["opti_oignon.ui"], "main() names the UI inside its body, and nothing else is literal"
    assert "opti_oignon.api.app" not in facade["closure_all"]
    importers = [n for n, m in report["modules"].items() if "opti_oignon" in m["imports_eager"]]
    for name in importers:
        assert len(report["modules"][name]["closure_eager"]) < 232, f"{name} no longer inherits the whole tree through the facade"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
