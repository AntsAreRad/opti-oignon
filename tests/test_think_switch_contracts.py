#!/usr/bin/env python3
"""Contracts for the think switch: what a request tells Ollama about thinking.

The registry declared ``think: bool = False`` and sent ``think`` to Ollama
only when it was true, so a False never left the process and a model that
thinks by default thought anyway -- through the agentic pipelines that say
``think=False``, and through the drift A/B and the librarian, whose small
token budgets went to the thinking. ``think`` now has three states: None,
the default, sends nothing and leaves the model's own way, exactly as
before; True is sent; False is sent, but only to a model whose capabilities,
as Ollama reports them, include thinking -- so a model that does not think
never receives the switch at all.

  * TH1 -- an explicit False reaches a model that declares thinking, on both
    heads, and True is still sent.
  * TH2 -- nothing reaches a model that does not declare thinking, or whose
    capabilities cannot be read; None sends nothing to any model; and the
    capabilities are read once per model.
  * TH3 -- the drift A/B, on its trial and its turns, and the librarian's
    summariser ask for no thinking.
  * TH4 -- the remote core carries the three states over its wire, and a
    request that says nothing leaves the model's default.

Local-only (the public distribution ships no tests). The backend module is
loaded through the shared isolation window with the client library faked.
"""

import importlib.util
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_BACKEND = "opti_oignon.inference_backend"
_MSGS = [{"role": "user", "content": "hi"}]
_ONION = ("probes", "drift", "core_store", "receipts", "composer", "peels", "librarian")


class _Ollama:
    """The client library, recording each chat and each capability read."""

    def __init__(self, capabilities):
        self._capabilities = capabilities
        self.chats = []
        self.shows = []

    def show(self, name):
        self.shows.append(name)
        value = self._capabilities.get(name)
        if isinstance(value, Exception):
            raise value
        info = {"details": {}, "model_info": {}}
        if value is not None:
            info["capabilities"] = list(value)
        return info

    def chat(self, **kwargs):
        self.chats.append(dict(kwargs))
        if kwargs.get("stream"):
            return iter([{"message": {"content": "x"}, "done": True}])
        return {"message": {"content": "ok"}}


def _open(capabilities, monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    loaded, restore = isolate(targets={_BACKEND: source("inference_backend.py")}, packages=("opti_oignon",))
    mod = loaded[_BACKEND]
    fake = _Ollama(capabilities)
    mod.OLLAMA_AVAILABLE = True
    mod._ollama_module = fake
    return mod, fake, restore


# ---------------------------------------------------------------------------
# TH1 -- an explicit False reaches a model that thinks
# ---------------------------------------------------------------------------
def test_th1_an_explicit_false_reaches_a_model_that_declares_thinking_on_both_heads(monkeypatch):
    mod, fake, restore = _open({"thinker": ["completion", "thinking"]}, monkeypatch)
    try:
        backend = mod.OllamaBackend()
        backend.generate("thinker", _MSGS, think=False)
        list(backend.stream("thinker", _MSGS, think=False))
        backend.generate("thinker", _MSGS, think=True)
        assert [chat.get("think", "absent") for chat in fake.chats] == [False, False, True], fake.chats
    finally:
        restore()


# ---------------------------------------------------------------------------
# TH2 -- nothing reaches a model that does not think; None sends nothing
# ---------------------------------------------------------------------------
def test_th2_nothing_reaches_a_model_that_does_not_think_and_none_sends_nothing(monkeypatch):
    capabilities = {
        "plain": ["completion", "tools"],
        "silent": None,
        "broken": RuntimeError("show failed"),
        "thinker": ["completion", "thinking"],
    }
    mod, fake, restore = _open(capabilities, monkeypatch)
    try:
        backend = mod.OllamaBackend()
        for model in ("plain", "silent", "broken"):
            backend.generate(model, _MSGS, think=False)
            list(backend.stream(model, _MSGS, think=False))
        backend.generate("thinker", _MSGS)
        list(backend.stream("thinker", _MSGS))
        assert all("think" not in chat for chat in fake.chats), fake.chats
        assert "thinker" not in fake.shows, "None asks nothing, not even the capabilities"
        backend.generate("thinker", _MSGS, think=False)
        backend.generate("thinker", _MSGS, think=False)
        assert fake.chats[-1].get("think") is False, "control: the model that thinks does receive it"
        assert fake.shows.count("thinker") == 1 and fake.shows.count("plain") == 1, fake.shows
    finally:
        restore()


# ---------------------------------------------------------------------------
# TH3 -- the drift A/B and the librarian ask for no thinking
# ---------------------------------------------------------------------------
class _Recording:
    """A backend that lists two models, records every request, and breaks after three."""

    def __init__(self):
        self.requests = []

    def list_models(self):
        return [type("_Listed", (), {"name": name})() for name in ("answer:latest", "keeper:latest")]

    def generate(self, model, messages, **kwargs):
        self.requests.append((model, kwargs.get("think", "absent")))
        if len(self.requests) > 3:
            raise RuntimeError("enough")
        return type("_Reply", (), {"content": "Noted."})()


def test_th3_the_drift_ab_and_the_librarian_ask_for_no_thinking(capsys):
    targets = {f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _ONION}
    targets["opti_oignon.agent.untrusted_context"] = source("agent", "untrusted_context.py")
    loaded, restore = isolate(targets=targets, packages=("opti_oignon.memory", "opti_oignon.agent"))
    try:
        spec = importlib.util.spec_from_file_location("drift_ab_under_contract", REPO / "scripts" / "drift_ab.py")
        ab = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ab)
        backend = _Recording()
        ab.main(["--model", "answer", "--librarian-model", "keeper"], resolve=lambda model: backend)
        capsys.readouterr()
        assert backend.requests[:3] == [("answer", False), ("keeper", False), ("answer", False)], backend.requests
        librarian = loaded["opti_oignon.memory.librarian"]
        keeper = _Recording()
        summarize = librarian.registry_summarizer(librarian.load_config(), resolve=lambda model: keeper)
        summarize([{"turn_id": "t1", "role": "user", "text": "Alice lives in Berlin."}])
        assert [think for _model, think in keeper.requests] == [False], keeper.requests
    finally:
        restore()


# ---------------------------------------------------------------------------
# TH4 -- the remote core carries the three states
# ---------------------------------------------------------------------------
def test_th4_the_remote_core_carries_the_three_states_over_its_wire():
    loaded, restore = isolate(
        targets={
            _BACKEND: source("inference_backend.py"),
            "opti_oignon.core_daemon": source("core_daemon.py"),
            "opti_oignon.core_client": source("core_client.py"),
        },
        packages=("opti_oignon",),
    )
    try:
        client = loaded["opti_oignon.core_client"].RemoteCoreBackend
        daemon = loaded["opti_oignon.core_daemon"].CoreService
        for think in (None, False, True):
            payload = json.loads(json.dumps(client._payload("m", _MSGS, None, "30m", think)))
            assert payload["think"] is think, (think, payload)
            request, refusal = daemon._request(payload)
            assert refusal is None and request["think"] is think, (think, request)
        request, _refusal = daemon._request({"model": "m", "messages": _MSGS})
        assert request["think"] is None, "a request that says nothing leaves the model's default"
        assert inspect.signature(client.generate).parameters["think"].default is None
    finally:
        restore()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
