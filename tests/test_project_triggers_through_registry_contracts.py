#!/usr/bin/env python3
"""Contracts for the project trigger detector's third level, through the registry.

The third level asks a model whether a query concerns a project, under a
hard budget of half a second and five tokens. It posted to the inference
server's generate endpoint itself, so the classification was never
admitted by the governor and the funnel guard could not see it. It now
asks the registry's backend for the configured model.

  * LT1 -- the classification is one request to the registry's backend: the
    prompt as one user message, temperature zero, five tokens, the budget
    as the request timeout; YES is relevant, NO is not, anything else is
    no answer.
  * LT2 -- no registry, no backend for the model, or a backend that fails
    or times out: no answer, never a guess, and nothing else is tried.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window with the HTTP transport declared unreachable.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


class _Backend:
    def __init__(self, answer="YES", fail=None):
        self.answer, self.fail = answer, fail
        self.calls = []

    def generate(self, model=None, messages=None, options=None, **kwargs):
        self.calls.append({"model": model, "messages": messages, "options": dict(options or {})})
        if self.fail:
            raise self.fail
        return SimpleNamespace(content=self.answer)


class _Registry:
    def __init__(self, backend):
        self.backend = backend
        self.asked = []

    def resolve_backend(self, model):
        self.asked.append(model)
        return self.backend


class _Store:
    def get_project(self, project_id):
        return SimpleNamespace(name="Onion garden", description="Tending onions at dawn", settings={})

    def list_files(self, project_id):
        return [SimpleNamespace(filename="beds.md")]


def _open(registry_factory):
    backend_mod = types.ModuleType("opti_oignon.inference_backend")
    backend_mod.get_backend_registry = registry_factory
    loaded, restore = isolate(
        targets={"opti_oignon.project_triggers": source("project_triggers.py")},
        blocked=("requests", "opti_oignon.projects"),
        seeded={"opti_oignon.inference_backend": backend_mod},
    )
    mod = loaded["opti_oignon.project_triggers"]
    detector = mod.ProjectTriggerDetector(store=_Store())
    detector._config.update({"level3_model": "clf:1b", "level3_timeout_ms": 500})
    return detector, restore


# ---------------------------------------------------------------------------
# LT1 -- one request to the registry's backend, answered by the model
# ---------------------------------------------------------------------------
def test_lt1_the_third_level_asks_the_registrys_backend_within_its_budget():
    backend = _Backend("YES")
    registry = _Registry(backend)
    detector, restore = _open(lambda: registry)
    try:
        assert detector._check_level3("How deep do I plant the sets?", "p1") is True
        assert registry.asked == ["clf:1b"], "the configured model, resolved by the registry"
        call = backend.calls[0]
        assert [m["role"] for m in call["messages"]] == ["user"], "the prompt travels as one user message"
        assert "How deep do I plant the sets?" in call["messages"][0]["content"]
        assert "Onion garden" in call["messages"][0]["content"]
        options = call["options"]
        assert options["temperature"] == 0.0 and options["num_predict"] == 5
        assert options["timeout"] == 0.5, "the budget of the level is the request's timeout"
        backend.answer = "NO"
        assert detector._check_level3("What is the capital of Peru?", "p1") is False
        backend.answer = "PERHAPS"
        assert detector._check_level3("Anything?", "p1") is None, "an ambiguous answer is no answer"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LT2 -- no backend, a failure or a timeout: no answer
# ---------------------------------------------------------------------------
def test_lt2_without_a_backend_or_on_failure_the_third_level_answers_nothing():
    detector, restore = _open(lambda: _Registry(None))
    try:
        assert detector._check_level3("How deep?", "p1") is None, "no backend for the model: no answer"
    finally:
        restore()

    def broken():
        raise RuntimeError("registry down")

    detector, restore = _open(broken)
    try:
        assert detector._check_level3("How deep?", "p1") is None, "no registry: no answer"
    finally:
        restore()

    for failure in (TimeoutError("timed out"), RuntimeError("admission refused")):
        backend = _Backend(fail=failure)
        detector, restore = _open(lambda b=backend: _Registry(b))
        try:
            assert detector._check_level3("How deep?", "p1") is None, f"{failure!r}: no answer, never a guess"
            assert len(backend.calls) == 1, "one attempt, nothing else tried"
        finally:
            restore()
