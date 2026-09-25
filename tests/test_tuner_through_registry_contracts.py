#!/usr/bin/env python3
"""Contracts that the tuner reaches a real backend, and only through the registry.

Four call sites asked the registry for ``get_backend``, a method it does not
have. Each sat inside a broad ``except``, so the tuner route always fell to
the simulated benchmark, the llama.cpp benchmark always answered "not
available", and the speculative-decoding listing was always empty -- in
silence. The one real path left, the Ollama benchmark, posted to the
server's HTTP endpoint directly, so fixing the name alone would have sent
every sweep point to the model without the governor's admission.

  * TU1 -- the tuner route picks the Ollama benchmark when the registry's
    Ollama backend is healthy, and a run of it reaches that backend.
  * TU2 -- with Ollama unhealthy it picks llama.cpp when that is healthy;
    with neither it falls to the simulated benchmark, which says so.
  * TU3 -- the llama.cpp benchmark handed no backend finds it in the registry.
  * TU4 -- the speculative-decoding listing reads the llama.cpp backend from
    the registry.
  * TU5 -- the Ollama benchmark asks its backend with the tuner's options and
    a timeout; a backend that refuses (the governor's admission) is an error
    by name, and no backend is an error by name, never a run.
  * TU6 -- the tuner module holds no HTTP transport of its own: no
    ``requests``, no endpoint path of the inference server.
  * TU7 -- the benchmark's token budget and timeout come from
    ``auto_tuner.yaml`` through the route to the run, on Ollama and on
    llama.cpp.
  * TU8 -- an unreadable budget or timeout falls back to its own default,
    says so, and leaves the other settings as written.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window; the registry is a stand-in that has ``get`` and
nothing else, the way the real one does.
"""

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_TUNER = "opti_oignon.auto_tuner"


class _Registry:
    """The registry's lookup surface: ``get(name)``, as the real one exposes it."""

    def __init__(self, **backends):
        self._backends = backends
        self.asked = []

    def get(self, name):
        self.asked.append(name)
        return self._backends.get(name)


class _Backend:
    def __init__(self, *, healthy=True, extra=None, models=None, refuse=None):
        self.healthy = healthy
        self.extra = extra if extra is not None else {
            "eval_count": 64, "eval_duration": 2_000_000_000,
            "prompt_eval_count": 16, "prompt_eval_duration": 250_000_000,
        }
        self.models = models
        self.refuse = refuse
        self.calls = []

    def health_check(self):
        return self.healthy

    def generate(self, model=None, messages=None, options=None, **kwargs):
        self.calls.append({"model": model, "messages": messages, "options": dict(options or {})})
        if self.refuse:
            raise RuntimeError(self.refuse)
        return SimpleNamespace(content="stand-in reply", extra=dict(self.extra))

    def list_models(self):
        return self.models


def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _open(registry, *, route=None, manager=None):
    backend_mod = _stub("opti_oignon.inference_backend", get_backend_registry=lambda: registry)
    deps = _stub(
        "opti_oignon.api.deps",
        AUTO_TUNER_AVAILABLE=True, INFERENCE_BACKEND_AVAILABLE=True, SPECULATIVE_DECODING_AVAILABLE=True,
        get_auto_tuner_manager=manager or (lambda: None), get_speculative_decoding_manager=lambda: None,
        get_backend_registry=lambda: registry,
    )
    seeded = {"opti_oignon.inference_backend": backend_mod, "opti_oignon.api.deps": deps}
    targets = {_TUNER: source("auto_tuner.py")}
    if route is not None:
        targets["opti_oignon.api.schemas"] = source("api", "schemas.py")
    if route == "tuner":
        targets["opti_oignon.api.routes_tuner"] = source("api", "routes_tuner.py")
    elif route == "speculative":
        targets["opti_oignon.api.routes_speculative_decoding"] = source("api", "routes_speculative_decoding.py")
    loaded, restore = isolate(targets=targets, blocked=("requests",), seeded=seeded, packages=("opti_oignon.api",))
    return loaded, restore


# ---------------------------------------------------------------------------
# TU1 -- the route reaches a healthy Ollama backend
# ---------------------------------------------------------------------------
def test_tu1_the_tuner_route_picks_a_healthy_ollama_backend_and_a_run_reaches_it():
    ollama = _Backend()
    registry = _Registry(ollama=ollama, llama_cpp=_Backend())
    loaded, restore = _open(registry, route="tuner")
    try:
        tuner = loaded[_TUNER]
        bench = loaded["opti_oignon.api.routes_tuner"]._resolve_benchmark_fn("stand-in-model")
        result = bench({"threads": 6, "batch_size": 512})
        assert len(ollama.calls) == 1, "the run reached the registry's Ollama backend"
        assert ollama.calls[0]["model"] == "stand-in-model"
        assert result.error == "" and result.source == tuner.SOURCE_MEASURED, "a real run, labelled by what it read"
        assert result.source != tuner.SOURCE_SIMULATED
        assert "ollama" in registry.asked
    finally:
        restore()


# ---------------------------------------------------------------------------
# TU2 -- llama.cpp second, the simulation last, and it says so
# ---------------------------------------------------------------------------
def test_tu2_llama_cpp_is_next_and_the_simulation_is_last_and_labelled():
    llama = _Backend()
    registry = _Registry(ollama=_Backend(healthy=False), llama_cpp=llama)
    loaded, restore = _open(registry, route="tuner")
    try:
        tuner = loaded[_TUNER]
        routes = loaded["opti_oignon.api.routes_tuner"]
        result = routes._resolve_benchmark_fn("stand-in-model")({"threads": 6, "batch_size": 512})
        assert len(llama.calls) == 1, "an unhealthy Ollama hands over to a healthy llama.cpp"
        assert result.source != tuner.SOURCE_SIMULATED

        registry._backends = {"ollama": _Backend(healthy=False), "llama_cpp": _Backend(healthy=False)}
        fallback = routes._resolve_benchmark_fn("stand-in-model")({"threads": 6, "batch_size": 512})
        assert fallback.source == tuner.SOURCE_SIMULATED, "with nothing healthy the run is simulated, and says so"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TU3 -- the llama.cpp benchmark finds its backend in the registry
# ---------------------------------------------------------------------------
def test_tu3_the_llamacpp_benchmark_without_a_backend_finds_it_in_the_registry():
    llama = _Backend()
    registry = _Registry(llama_cpp=llama)
    loaded, restore = _open(registry)
    try:
        result = loaded[_TUNER].create_llamacpp_benchmark_fn("stand-in-model")({"threads": 6, "batch_size": 512})
        assert result.error == "", result.error
        assert len(llama.calls) == 1 and registry.asked == ["llama_cpp"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# TU4 -- the speculative listing reads llama.cpp from the registry
# ---------------------------------------------------------------------------
def test_tu4_the_speculative_listing_reads_the_llamacpp_backend_from_the_registry():
    model = SimpleNamespace(name="draft-0.5b.gguf", family="qwen", parameter_size="0.5B",
                            quantization_level="Q8_0", path="/models/draft-0.5b.gguf")
    registry = _Registry(llama_cpp=_Backend(models=[model]))
    loaded, restore = _open(registry, route="speculative")
    try:
        listed = loaded["opti_oignon.api.routes_speculative_decoding"]._get_llama_cpp_models()
        assert [m["name"] for m in listed] == ["draft-0.5b.gguf"], "the backend's catalogue, not an empty list"
        assert listed[0]["quantization"] == "Q8_0" and registry.asked == ["llama_cpp"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# TU5 -- the Ollama benchmark asks its backend, and refusals are named
# ---------------------------------------------------------------------------
def test_tu5_the_ollama_benchmark_asks_its_backend_and_names_every_refusal():
    registry = _Registry()
    loaded, restore = _open(registry)
    try:
        tuner = loaded[_TUNER]
        backend = _Backend()
        tuner.create_ollama_benchmark_fn("stand-in-model", benchmark_tokens=77, backend=backend)(
            {"threads": 6, "batch_size": 512, "flash_attention": True})
        options = backend.calls[0]["options"]
        assert options["num_thread"] == 6 and options["num_batch"] == 512 and options["flash_attn"] is True
        assert options["num_predict"] == 77
        assert isinstance(options.get("timeout"), (int, float)) and options["timeout"] > 0, "the request is bounded"

        refused = tuner.create_ollama_benchmark_fn(
            "stand-in-model", backend=_Backend(refuse="admission refused: VRAM budget exceeded"))({"threads": 6})
        assert "admission refused: VRAM budget exceeded" in refused.error, "the governor's refusal, by name"
        assert refused.source != tuner.SOURCE_MEASURED

        orphan = tuner.create_ollama_benchmark_fn("stand-in-model")({"threads": 6})
        assert registry.asked == ["ollama"], "no backend handed in: the registry is asked"
        assert "no ollama backend" in orphan.error.lower(), "no backend is an error by name, never a run"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TU6 -- no HTTP transport of the tuner's own
# ---------------------------------------------------------------------------
def test_tu6_the_tuner_holds_no_http_transport_of_its_own():
    path = REPO / "opti_oignon" / "auto_tuner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "time" in imported, "control: the census reads the module's imports"
    assert not imported & {"requests", "httpx", "urllib", "http", "aiohttp"}, imported
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert any("/" in s for s in literals), "control: the census reads string literals"
    endpoints = [s for s in literals if "/api/" in s]
    assert endpoints == [], f"no endpoint of the inference server is spelled here: {endpoints}"


def _manager_from(tuner, tmp_path, text):
    """The real manager, loaded from an auto_tuner.yaml written here; its results stay here too."""
    tuner._RESULTS_PATH = tmp_path / "tuner_results.json"
    config = tmp_path / "auto_tuner.yaml"
    config.write_text(text, encoding="utf-8")
    return tuner.AutoTunerManager(config_path=str(config))


# ---------------------------------------------------------------------------
# TU7 -- the budget and the timeout come from the YAML
# ---------------------------------------------------------------------------
def test_tu7_the_benchmark_budget_and_timeout_come_from_auto_tuner_yaml(tmp_path):
    ollama, llama = _Backend(), _Backend()
    registry = _Registry(ollama=ollama, llama_cpp=llama)
    holder = {}
    loaded, restore = _open(registry, route="tuner", manager=lambda: holder["manager"])
    try:
        tuner = loaded[_TUNER]
        routes = loaded["opti_oignon.api.routes_tuner"]
        manager = _manager_from(tuner, tmp_path, "auto_tuner:\n  benchmark_tokens: 64\n  benchmark_timeout_s: 30\n")
        assert (manager.config.benchmark_tokens, manager.config.benchmark_timeout_s) == (64, 30.0)
        started = []

        def start_tuning(model_name, benchmark_fn):
            started.append(benchmark_fn)
            return tuner.TunerJob(job_id="stand-in", model_name=model_name, status="pending")

        holder["manager"] = SimpleNamespace(config=manager.config, start_tuning=start_tuning)
        routes.start_tuning(loaded["opti_oignon.api.schemas"].TunerRunRequest(model_name="stand-in-model"))
        assert len(started) == 1, "the route started one tuning run"
        started[0]({"threads": 6, "batch_size": 512})
        options = ollama.calls[-1]["options"]
        assert options["num_predict"] == 64 and options["timeout"] == 30.0, options
        registry._backends["ollama"] = _Backend(healthy=False)
        routes._resolve_benchmark_fn("stand-in-model", manager.config)({"threads": 6, "batch_size": 512})
        assert llama.calls[-1]["options"]["num_predict"] == 64, "llama.cpp takes the same budget"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TU8 -- an unreadable value falls back alone
# ---------------------------------------------------------------------------
def test_tu8_an_unreadable_budget_or_timeout_falls_back_to_its_own_default(tmp_path, caplog):
    loaded, restore = _open(_Registry())
    try:
        tuner = loaded[_TUNER]
        defaults = tuner.TunerConfig()
        cases = (
            ("benchmark_tokens: lots\n  benchmark_timeout_s: -5\n", "benchmark_tokens"),
            ("benchmark_tokens: true\n  benchmark_timeout_s: soon\n", "benchmark_timeout_s"),
            ("benchmark_tokens: 0\n  benchmark_timeout_s: 0\n", "benchmark_tokens"),
        )
        for text, named in cases:
            caplog.clear()
            manager = _manager_from(tuner, tmp_path, "auto_tuner:\n  warmup_runs: 1\n  " + text)
            config = manager.config
            assert (config.benchmark_tokens, config.benchmark_timeout_s) == (defaults.benchmark_tokens, defaults.benchmark_timeout_s), text
            assert config.warmup_runs == 1, "the other settings are read as written"
            assert any(named in record.getMessage() for record in caplog.records), f"the fallback is said: {text!r}"
        manager = _manager_from(tuner, tmp_path, "auto_tuner:\n  benchmark_tokens: '96'\n  benchmark_timeout_s: '12.5'\n")
        assert (manager.config.benchmark_tokens, manager.config.benchmark_timeout_s) == (96, 12.5)
    finally:
        restore()
