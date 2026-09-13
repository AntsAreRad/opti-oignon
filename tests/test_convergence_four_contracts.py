#!/usr/bin/env python3
"""Contracts that the model catalogue is read through the inference registry.

Three convergence blocks sent every request method through the registry and
emptied the funnel guard's ledger. The fourth found what the census could
not see: a module that binds the client module to an attribute at
construction and requests through the attribute -- one of them a ``chat``
with images -- two request methods handed on uncalled, and the catalogue
reads nobody counted, ``list()`` and ``show()``, at twenty-two sites in
sixteen modules. The catalogue is two heads the backend contract already
had, ``list_models`` and ``model_info``; the Ollama backend now carries in
``extra`` what the client reported beyond the typed fields, and three
readers on the shared registry clients answer what every module used to
fold on its own. The distinction kept on purpose: ``None`` is "no backend
is registered, nobody looked", an empty list is a backend that looked and
found nothing.

  * CQ1 -- the three catalogue readers answer through the registry and say
    ``None`` without a backend, never an empty list.
  * CQ2 -- the vision pipeline lists through ``list_models`` and describes
    through ``generate`` with the images, and refuses by name without a
    backend; the client is never reached.
  * CQ3 -- the extraction and curation passes hand back a registry-backed
    chat callable in the shape they always read, and none without a
    backend; the extraction picks its model from the registry's listing.
  * CQ4 -- the router lists through the registry and keeps its last list
    when no backend is registered.
  * CQ5 -- the context window and the prompt budget read the context length
    from ``model_info``, the budget falling back to the modelfile parameter
    text the backend carried, and both say unknown by their own convention
    without a backend.
  * CQ6 -- model profiles auto-detect from ``model_info`` and vision
    detection reads the family list the backend carried.
  * CQ7 -- the network manager's reachability is the backend's health check
    and its embedding check matches the listing with or without a tag.
  * CQ8 -- the system presets read the size in bytes and the parameter
    size from the record; the CLI prints the listing from records and exits
    non-zero when no backend is registered.
  * CQ9 -- the benchmark trigger's default lister snapshots names and
    digests from records.
  * CQ10 -- the health monitor probes through ``model_info``, names the
    backend when a model is not served, and discovers names through the
    listing.
  * CQ11 -- the lifecycle manager lists, shows and reads the digest through
    the registry while the client module it still holds is asked nothing.
  * CQ12 -- the summariser picks its model from the active backend's
    listing and has none without a backend.
  * CQ13 -- CQ10 word for word, except that discovery without a backend
    is unknown rather than empty, the doctrine the listing head gained one
    block later (CQ10 is deselected by name).

Local-only (the public distribution ships no tests). Every module is loaded
through the shared isolation window over a scripted client that records
what it is asked; ``list`` and ``show`` on it raise, so a module that still
reached the client would fail by name.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import StubRegistry, seed_registry  # noqa: E402

_REGISTRY = "opti_oignon.inference_backend"
_CLIENTS = "opti_oignon.registry_clients"


class _Scripted:
    """A scripted client: chat answers, a listing, a show; every call recorded."""

    def __init__(self, reply="answer", models=None, show=None):
        self.calls = []
        self.reply = reply
        self._models = models if models is not None else [
            {"model": "qwen3:8b-q4", "size": 4_700_000_000, "digest": "sha256:aa", "modified_at": "2026-09-01T00:00:00Z",
             "details": {"family": "qwen3", "families": ["qwen3"], "parameter_size": "8B", "quantization_level": "Q4"}},
            {"model": "llava:7b", "size": 4_000_000_000, "digest": "sha256:bb",
             "details": {"family": "llama", "families": ["llama", "clip"], "parameter_size": "7B"}},
        ]
        self._show = show if show is not None else {
            "details": {"family": "qwen3", "families": ["qwen3"], "parameter_size": "8B", "quantization_level": "Q4"},
            "model_info": {"qwen3.context_length": 40960},
            "parameters": "num_ctx 8192\nstop <|im_end|>",
            "digest": "sha256:aa", "template": "T", "modelfile": "FROM x",
        }

    def chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        return {"message": {"content": self.reply}}

    def list(self):
        self.calls.append(("list",))
        return {"models": list(self._models)}

    def show(self, model):
        self.calls.append(("show", model))
        return dict(self._show)


class _Forbidden(types.ModuleType):
    """A client module on which any catalogue read is a failure by name."""

    def __init__(self):
        super().__init__("ollama")
        self.calls = []

    def list(self):
        self.calls.append("list")
        raise AssertionError("direct client list()")

    def show(self, model):
        self.calls.append(("show", model))
        raise AssertionError("direct client show()")

    def chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        raise AssertionError("direct client chat()")


def _empty_registry_module():
    module = types.ModuleType(_REGISTRY)
    module.get_backend_registry = lambda: StubRegistry()
    return module


def _window(rel, name, *, scripted=None, empty=False, seeded=None, packages=("opti_oignon",), with_clients=True):
    """Load ``rel`` as ``name`` over a registry seeded on ``scripted`` or seeded empty.

    The shared clients module is loaded first when the module reads the
    catalogue through it; the scripted registry is the only backend source.
    """
    seeds = dict(seeded or {})
    if scripted is not None:
        seed_registry(seeds, scripted)
    elif empty:
        seeds[_REGISTRY] = _empty_registry_module()
    targets = {}
    if with_clients:
        targets[_CLIENTS] = source("registry_clients.py")
    targets[name] = source(*rel.split("/"))
    loaded, restore = isolate(targets=targets, seeded=seeds, packages=packages)
    return loaded[name], restore


def _forbid_client():
    """Mount a client that refuses every read; returns (module, undo)."""
    had = "ollama" in sys.modules
    prev = sys.modules.get("ollama")
    client = _Forbidden()
    sys.modules["ollama"] = client

    def undo():
        if had:
            sys.modules["ollama"] = prev
        else:
            sys.modules.pop("ollama", None)

    return client, undo


def _reads(scripted, kind):
    return [c for c in scripted.calls if c[0] == kind]


# ---------------------------------------------------------------------------
# CQ1 -- the three readers
# ---------------------------------------------------------------------------
def test_cq1_the_catalogue_readers_answer_through_the_registry_and_unknown_is_none():
    scripted = _Scripted()
    mod, restore = _window("registry_clients.py", _CLIENTS, scripted=scripted, with_clients=False)
    try:
        assert mod.installed_model_names() == ["qwen3:8b-q4", "llava:7b"]
        records = mod.installed_models()
        assert [r.name for r in records] == ["qwen3:8b-q4", "llava:7b"]
        assert records[0].extra["size_bytes"] == 4_700_000_000 and records[0].extra["digest"] == "sha256:aa"
        info = mod.describe_model("qwen3:8b-q4")
        assert info.context_length == 40960 and info.family == "qwen3"
        assert info.extra["parameters"].startswith("num_ctx") and info.extra["families"] == ["qwen3"]
        assert _reads(scripted, "list") == [("list",), ("list",)] and _reads(scripted, "show") == [("show", "qwen3:8b-q4")]
        assert mod.backend_for() is not None and mod.backend_for("llava:7b") is not None
    finally:
        restore()

    mod, restore = _window("registry_clients.py", _CLIENTS, empty=True, with_clients=False)
    try:
        assert mod.installed_model_names() is None, "no backend is unknown, not an empty list"
        assert mod.installed_models() is None
        assert mod.describe_model("qwen3:8b-q4") is None
        assert mod.backend_for() is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# CQ2 -- the vision pipeline
# ---------------------------------------------------------------------------
def test_cq2_the_vision_pipeline_lists_and_describes_through_the_registry():
    client, undo = _forbid_client()
    scripted = _Scripted(reply="a cat on a mat")
    mod, restore = _window("vision_pipeline.py", "opti_oignon.vision_pipeline", scripted=scripted, with_clients=False)
    try:
        pipeline = mod.VisionPipeline(vision_config=None)
        assert pipeline._list_available_models() == ["qwen3:8b-q4", "llava:7b"]
        out = pipeline.describe_image(["aW1n"], "what is this?", vision_model="llava:7b")
        assert out == "a cat on a mat"
        (kind, asked), = _reads(scripted, "chat")
        assert asked["model"] == "llava:7b"
        assert asked["messages"] == [{"role": "user", "content": asked["messages"][0]["content"]}]
        assert "what is this?" in asked["messages"][0]["content"]
        assert asked["options"] == {"num_predict": pipeline.max_description_tokens}
        assert asked["images"] == ["aW1n"], "the images travel on the request, not in the message"
        assert client.calls == [], "the client module was never reached"
    finally:
        restore()
        undo()

    mod, restore = _window("vision_pipeline.py", "opti_oignon.vision_pipeline", empty=True, with_clients=False)
    try:
        pipeline = mod.VisionPipeline(vision_config=None)
        assert pipeline._list_available_models() == []
        assert pipeline.describe_image(["aW1n"], "q", vision_model="llava:7b") == "", (
            "no backend means no description, refused by name"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# CQ3 -- extraction and curation
# ---------------------------------------------------------------------------
def test_cq3_the_memory_passes_hand_back_a_registry_backed_chat_fn_and_pick_from_the_listing():
    client, undo = _forbid_client()
    try:
        for rel, name, cls in (
            ("memory/extraction.py", "opti_oignon.memory.extraction", "FactExtractor"),
            ("memory/curation.py", "opti_oignon.memory.curation", "MemoryCurator"),
        ):
            scripted = _Scripted(reply="- fact: x")
            mod, restore = _window(rel, name, scripted=scripted, packages=("opti_oignon", "opti_oignon.memory"), with_clients=False)
            try:
                worker = getattr(mod, cls)(fallback_models=["qwen3:8b", "other"])
                chat_fn = worker._get_chat_fn()
                assert callable(chat_fn)
                reply = chat_fn(model="qwen3:8b", messages=[{"role": "user", "content": "hi"}], options={"temperature": 0.1})
                assert reply == {"message": {"content": "- fact: x"}}, "the shape the passes read"
                (kind, asked), = _reads(scripted, "chat")
                assert asked["model"] == "qwen3:8b" and asked["messages"] == [{"role": "user", "content": "hi"}]
                assert asked["options"] == {"temperature": 0.1}
                if cls == "FactExtractor":
                    assert worker._resolve_model() == "qwen3:8b-q4", "the prefix match on the registry's listing"
                    assert _reads(scripted, "list") == [("list",)]
                assert client.calls == []
            finally:
                restore()

            mod, restore = _window(rel, name, empty=True, packages=("opti_oignon", "opti_oignon.memory"), with_clients=False)
            try:
                worker = getattr(mod, cls)(fallback_models=["qwen3:8b"])
                assert worker._get_chat_fn() is None, "no backend means no model pass"
                assert worker._resolve_model() == "qwen3:8b", "the first fallback stands in"
            finally:
                restore()
    finally:
        undo()


# ---------------------------------------------------------------------------
# CQ4 -- the router
# ---------------------------------------------------------------------------
def _router_seeds():
    analyzer = types.ModuleType("opti_oignon.analyzer")
    analyzer.AnalysisResult = object
    analyzer.TaskType = SimpleNamespace(PLANNING_DEEP="planning_deep", SIMPLE_QUESTION="simple_question")
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(get_temperature=lambda *a, **k: 0.3, get_timeout=lambda *a, **k: 30, get_model=lambda *a, **k: "m")
    return {"opti_oignon.analyzer": analyzer, "opti_oignon.config": cfg}


def test_cq4_the_router_lists_through_the_registry_and_keeps_its_last_list_without_a_backend():
    client, undo = _forbid_client()
    scripted = _Scripted()
    mod, restore = _window("router.py", "opti_oignon.router", scripted=scripted, seeded=_router_seeds())
    try:
        router = mod.ModelRouter()
        assert router.get_available_models(force_refresh=True) == ["qwen3:8b-q4", "llava:7b"]
        assert router.is_model_available("llava:7b") is True
        assert _reads(scripted, "list") == [("list",)] and client.calls == []
    finally:
        restore()

    mod, restore = _window("router.py", "opti_oignon.router", empty=True, seeded=_router_seeds())
    try:
        router = mod.ModelRouter()
        router._available_models = ["kept"]
        assert router.get_available_models(force_refresh=True) == ["kept"], (
            "no backend keeps the last list rather than erasing it"
        )
    finally:
        restore()
        undo()


# ---------------------------------------------------------------------------
# CQ5 -- context window and prompt budget
# ---------------------------------------------------------------------------
def test_cq5_the_context_window_and_the_prompt_budget_read_the_length_from_model_info():
    client, undo = _forbid_client()
    scripted = _Scripted()
    mod, restore = _window("context_window.py", "opti_oignon.context_window", scripted=scripted)
    try:
        manager = mod.TokenBudgetManager()
        assert manager._fetch_ollama_context_window("qwen3:8b-q4") == 40960
        assert manager._profiles["qwen3:8b-q4"]["context_window"] == 40960, "cached for next time"
        assert _reads(scripted, "show") == [("show", "qwen3:8b-q4")]
    finally:
        restore()

    no_length = _Scripted(show={"details": {"family": "x"}, "parameters": "num_ctx 16384"})
    mod, restore = _window("prompt_optimization.py", "opti_oignon.prompt_optimization", scripted=no_length)
    try:
        budget = mod.PromptTokenBudgetManager()
        assert budget._query_ollama_show("m") == 16384, "the modelfile parameter text the backend carried"
    finally:
        restore()
    mod, restore = _window("prompt_optimization.py", "opti_oignon.prompt_optimization", scripted=_Scripted())
    try:
        assert mod.PromptTokenBudgetManager()._query_ollama_show("m") == 40960, "the typed field first"
    finally:
        restore()

    mod, restore = _window("context_window.py", "opti_oignon.context_window", empty=True)
    try:
        assert mod.TokenBudgetManager()._fetch_ollama_context_window("m") == 0, "unknown by this reader's convention"
    finally:
        restore()
    mod, restore = _window("prompt_optimization.py", "opti_oignon.prompt_optimization", empty=True)
    try:
        assert mod.PromptTokenBudgetManager()._query_ollama_show("m") is None
    finally:
        restore()
        undo()
    assert client.calls == []


# ---------------------------------------------------------------------------
# CQ6 -- model profiles and vision detection
# ---------------------------------------------------------------------------
def test_cq6_profiles_auto_detect_and_vision_detection_read_model_info(tmp_path):
    client, undo = _forbid_client()
    scripted = _Scripted()
    mod, restore = _window("model_profiles.py", "opti_oignon.model_profiles", scripted=scripted)
    try:
        manager = mod.ModelProfileManager(profiles_path=tmp_path / "profiles.yaml")
        profile = manager.auto_detect("qwen3:8b-q4")
        assert profile is not None and profile.auto_detected is True
        assert profile.context_window == 40960 and profile.parameter_count == "8B"
        assert profile.quantization == "Q4" and profile.family == "qwen3"
        assert _reads(scripted, "show") == [("show", "qwen3:8b-q4")]
    finally:
        restore()
    mod, restore = _window("model_profiles.py", "opti_oignon.model_profiles", empty=True)
    try:
        assert mod.ModelProfileManager(profiles_path=tmp_path / "p.yaml").auto_detect("m") is None
    finally:
        restore()

    vision = _Scripted(show={"details": {"family": "llama", "families": ["llama", "clip"]}})
    mod, restore = _window("vision_config.py", "opti_oignon.vision_config", scripted=vision)
    try:
        config = mod.VisionConfig(config_path=tmp_path / "vision.yaml")
        assert config._probe_model_capabilities("llava:7b") is True, "clip in the family list the backend carried"
        assert _reads(vision, "show") == [("show", "llava:7b")]
    finally:
        restore()
    text_only = _Scripted(show={"details": {"family": "qwen3", "families": ["qwen3"]}})
    mod, restore = _window("vision_config.py", "opti_oignon.vision_config", scripted=text_only)
    try:
        assert mod.VisionConfig(config_path=tmp_path / "vision.yaml")._probe_model_capabilities("qwen3:8b") is False
    finally:
        restore()
        undo()
    assert client.calls == []


# ---------------------------------------------------------------------------
# CQ7 -- the network manager
# ---------------------------------------------------------------------------
def test_cq7_reachability_is_the_backend_health_check_and_the_embedding_check_matches_the_listing(tmp_path):
    client, undo = _forbid_client()
    scripted = _Scripted(models=[{"model": "mxbai-embed-large:latest"}, {"model": "qwen3:8b"}])
    mod, restore = _window("network_manager.py", "opti_oignon.network_manager", scripted=scripted)
    try:
        manager = mod.NetworkManager(config_path=tmp_path / "net.yaml")
        assert manager.check_ollama() is True, "the bridge backend's health check answers True"
        assert manager.check_embedding() is True, "matched with a tag the config does not name"
        manager._config["embedding_model"] = "nomic-embed"
        assert manager.check_embedding() is False
        assert client.calls == []
    finally:
        restore()
    mod, restore = _window("network_manager.py", "opti_oignon.network_manager", empty=True)
    try:
        manager = mod.NetworkManager(config_path=tmp_path / "net.yaml")
        assert manager.check_ollama() is False and manager.check_embedding() is False
    finally:
        restore()
        undo()


# ---------------------------------------------------------------------------
# CQ8 -- system presets and the CLI listing
# ---------------------------------------------------------------------------
def _presets_seeds():
    cfg = types.ModuleType("opti_oignon.config")
    cfg.CONFIG_DIR = Path("/nonexistent")
    cfg.DATA_DIR = Path("/nonexistent")
    cfg.load_yaml = lambda *a, **k: {}
    cfg.save_yaml = lambda *a, **k: True
    return {"opti_oignon.config": cfg}


def test_cq8_presets_read_the_record_and_the_cli_prints_records_or_exits(capsys):
    client, undo = _forbid_client()
    scripted = _Scripted()
    mod, restore = _window("system_presets.py", "opti_oignon.system_presets", scripted=scripted, seeded=_presets_seeds())
    try:
        found = mod.detect_ollama_models()
        assert [m.name for m in found] == ["qwen3:8b-q4", "llava:7b"]
        assert found[0].size_bytes == 4_700_000_000 and found[0].parameter_count_b == 8.0
        assert client.calls == []
    finally:
        restore()
    mod, restore = _window("system_presets.py", "opti_oignon.system_presets", empty=True, seeded=_presets_seeds())
    try:
        assert mod.detect_ollama_models() == []
    finally:
        restore()

    version = types.ModuleType("opti_oignon.__version__")
    version.__version__ = "0.0"
    mod, restore = _window("main.py", "opti_oignon.main", scripted=_Scripted(), seeded={"opti_oignon.__version__": version})
    try:
        mod.cmd_info(SimpleNamespace(info_type="models"))
        out = capsys.readouterr().out
        assert "qwen3:8b-q4" in out and "4.7 GB" in out and "2026-09-01" in out and "Total: 2 models" in out
    finally:
        restore()
    mod, restore = _window("main.py", "opti_oignon.main", empty=True, seeded={"opti_oignon.__version__": version})
    try:
        with pytest.raises(SystemExit) as raised:
            mod.cmd_info(SimpleNamespace(info_type="models"))
        assert raised.value.code == 1
        assert "No inference backend is registered" in capsys.readouterr().out
    finally:
        restore()
        undo()


# ---------------------------------------------------------------------------
# CQ9 -- the benchmark trigger's default lister
# ---------------------------------------------------------------------------
def test_cq9_the_trigger_snapshots_names_and_digests_from_records(tmp_path):
    client, undo = _forbid_client()
    scripted = _Scripted()
    mod, restore = _window("benchmark_auto_trigger.py", "opti_oignon.benchmark_auto_trigger", scripted=scripted)
    try:
        trigger = mod.AutoTrigger(config_path=tmp_path / "trigger.yaml")
        snapshot = trigger.take_snapshot()
        assert snapshot.models == {"qwen3:8b-q4": "sha256:aa", "llava:7b": "sha256:bb"}
        assert _reads(scripted, "list") == [("list",)] and client.calls == []
    finally:
        restore()
    mod, restore = _window("benchmark_auto_trigger.py", "opti_oignon.benchmark_auto_trigger", empty=True)
    try:
        assert mod._default_ollama_list() == []
        assert mod.AutoTrigger(config_path=tmp_path / "t.yaml").take_snapshot().models == {}
    finally:
        restore()
        undo()


# ---------------------------------------------------------------------------
# CQ10 -- the health monitor
# ---------------------------------------------------------------------------
def test_cq10_the_health_monitor_probes_through_model_info_and_names_the_backend(tmp_path):
    client, undo = _forbid_client()
    scripted = _Scripted()
    mod, restore = _window("model_health.py", "opti_oignon.model_health", scripted=scripted)
    try:
        monitor = mod.ModelHealthMonitor(config_path=tmp_path / "health.yaml")
        assert monitor._discover_models() == ["qwen3:8b-q4", "llava:7b"]
        record = monitor.check_model("qwen3:8b-q4")
        assert record.consecutive_failures == 0 and record.last_error == "" and record.latency_ms >= 0
        assert _reads(scripted, "show") == [("show", "qwen3:8b-q4")]
        assert monitor.get_config()["ollama_available"] is True
        assert client.calls == []
    finally:
        restore()

    class _Unknown(_Scripted):
        def show(self, model):
            self.calls.append(("show", model))
            raise RuntimeError("model not found")

    mod, restore = _window("model_health.py", "opti_oignon.model_health", scripted=_Unknown())
    try:
        record = mod.ModelHealthMonitor(config_path=tmp_path / "health.yaml").check_model("ghost")
        assert record.consecutive_failures == 1 and "model not found" in record.last_error
    finally:
        restore()

    mod, restore = _window("model_health.py", "opti_oignon.model_health", empty=True)
    try:
        monitor = mod.ModelHealthMonitor(config_path=tmp_path / "health.yaml")
        assert monitor._discover_models() == []
        record = monitor.check_model("m")
        assert record.last_error == "No inference backend is registered"
        assert monitor.get_config()["ollama_available"] is False
    finally:
        restore()
        undo()


# ---------------------------------------------------------------------------
# CQ11 -- the lifecycle manager
# ---------------------------------------------------------------------------
def test_cq11_the_lifecycle_manager_reads_the_catalogue_through_the_registry_only(tmp_path):
    client, undo = _forbid_client()
    scripted = _Scripted()
    mod, restore = _window("model_lifecycle.py", "opti_oignon.model_lifecycle", scripted=scripted)
    try:
        manager = mod.ModelLifecycleManager(
            config=mod.LifecycleConfig(enabled=True), aliases_path=tmp_path / "aliases.json", ollama_module=client,
        )
        listed = manager.list_models()
        assert [m["name"] for m in listed] == ["qwen3:8b-q4", "llava:7b"]
        assert listed[0]["size"] == 4_700_000_000 and listed[0]["digest"] == "sha256:aa"[:16]
        assert listed[0]["details"] == {"family": "qwen3", "parameter_size": "8B", "quantization_level": "Q4", "families": ["qwen3"]}
        assert listed[0]["modified_at"] > 0
        shown = manager.get_model_info("qwen3:8b-q4")
        assert shown["modelfile"] == "FROM x" and shown["parameters"].startswith("num_ctx")
        assert shown["template"] == "T" and shown["model_info"] == {"qwen3.context_length": 40960}
        assert shown["details"]["family"] == "qwen3"
        assert manager._get_local_digest("qwen3:8b-q4") == "sha256:aa"
        assert client.calls == [], "the client module it still holds for pull and delete was asked nothing"
        assert manager._ollama is client, "and it is still held, for the heads the contract does not have"
    finally:
        restore()
    mod, restore = _window("model_lifecycle.py", "opti_oignon.model_lifecycle", empty=True)
    try:
        manager = mod.ModelLifecycleManager(
            config=mod.LifecycleConfig(enabled=True), aliases_path=tmp_path / "aliases.json", ollama_module=client,
        )
        assert manager.list_models() == [] and manager.get_model_info("m") is None and manager._get_local_digest("m") == ""
        assert client.calls == []
    finally:
        restore()
        undo()


# ---------------------------------------------------------------------------
# CQ12 -- the summariser's model pick
# ---------------------------------------------------------------------------
def test_cq12_the_summariser_picks_its_model_from_the_active_backend_listing():
    client, undo = _forbid_client()
    scripted = _Scripted()
    mod, restore = _window("context_summary.py", "opti_oignon.context_summary", scripted=scripted, with_clients=False)
    try:
        summariser = mod.ContextSummarizer()
        summariser.FALLBACK_MODELS = ["qwen3:8b", "other"]
        assert summariser._find_available_model() == "qwen3:8b-q4", "the prefix match on the listing"
        assert _reads(scripted, "list") == [("list",)] and client.calls == []
    finally:
        restore()
    mod, restore = _window("context_summary.py", "opti_oignon.context_summary", empty=True, with_clients=False)
    try:
        assert mod.ContextSummarizer()._find_available_model() is None
    finally:
        restore()
        undo()


# ---------------------------------------------------------------------------
# CQ13 -- CQ10 under the unknown doctrine
# ---------------------------------------------------------------------------
def test_cq13_the_health_monitor_probes_through_model_info_and_says_unknown_without_a_backend(tmp_path):
    client, undo = _forbid_client()
    scripted = _Scripted()
    mod, restore = _window("model_health.py", "opti_oignon.model_health", scripted=scripted)
    try:
        monitor = mod.ModelHealthMonitor(config_path=tmp_path / "health.yaml")
        assert monitor._discover_models() == ["qwen3:8b-q4", "llava:7b"]
        record = monitor.check_model("qwen3:8b-q4")
        assert record.consecutive_failures == 0 and record.last_error == "" and record.latency_ms >= 0
        assert _reads(scripted, "show") == [("show", "qwen3:8b-q4")]
        assert monitor.get_config()["ollama_available"] is True
        assert client.calls == []
    finally:
        restore()

    class _Unknown(_Scripted):
        def show(self, model):
            self.calls.append(("show", model))
            raise RuntimeError("model not found")

    mod, restore = _window("model_health.py", "opti_oignon.model_health", scripted=_Unknown())
    try:
        record = mod.ModelHealthMonitor(config_path=tmp_path / "health.yaml").check_model("ghost")
        assert record.consecutive_failures == 1 and "model not found" in record.last_error
    finally:
        restore()

    mod, restore = _window("model_health.py", "opti_oignon.model_health", empty=True)
    try:
        monitor = mod.ModelHealthMonitor(config_path=tmp_path / "health.yaml")
        assert monitor._discover_models() is None, "no backend registered is unknown, not an empty discovery"
        record = monitor.check_model("m")
        assert record.last_error == "No inference backend is registered"
        assert monitor.get_config()["ollama_available"] is False
    finally:
        restore()
        undo()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
