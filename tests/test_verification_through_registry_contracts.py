#!/usr/bin/env python3
"""Contract for the verification engine's fix request through the registry.

The generate-execute-verify-fix loop asked the client for its fixes
directly. It now asks the registry's backend, and with no backend it
produces no fix and says so, rather than reaching for the client.

  * VR1 -- the fix request reaches the registry's backend with the fix
    messages, the fixed code comes back from the reply, and no backend
    means no fix and no exception.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window; the structured engine and the code executor are
declared unreachable, so only the fix head is under test.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import seed_registry  # noqa: E402


class _Scripted:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": self.reply}}


def _open(reply="Here you go:\n```python\nprint('fixed')\n```"):
    seeded = {}
    scripted = _Scripted(reply)
    seed_registry(seeded, scripted)
    loaded, restore = isolate(
        targets={"opti_oignon.verification": source("verification.py")},
        blocked=("opti_oignon.structured_output", "opti_oignon.code_executor"),
        seeded=seeded,
        packages=("opti_oignon",),
    )
    return loaded["opti_oignon.verification"], scripted, restore


def test_vr1_the_fix_request_goes_through_the_registry_and_no_backend_means_no_fix():
    mod, scripted, restore = _open()
    try:
        engine = mod.VerificationEngine(structured_engine=None, code_exec=None, max_iterations=1)
        fixed = engine._attempt_fix("print(1", "SyntaxError", "python", "print one", "m", 1)
        assert fixed == "print('fixed')"
        assert len(scripted.calls) == 1
        call = scripted.calls[0]
        assert call["model"] == "m" and call["options"]["temperature"] == 0.0
        assert any("print(1" in m["content"] for m in call["messages"]), "the broken code is in the fix request"
        assert not hasattr(mod, "ollama") and not hasattr(mod, "_ollama")
        text = source("verification.py").read_text(encoding="utf-8")
        assert "import ollama" not in text, "the module names no client"
    finally:
        restore()
    seeded = {}
    loaded, restore = isolate(
        targets={"opti_oignon.verification": source("verification.py")},
        blocked=("opti_oignon.structured_output", "opti_oignon.code_executor", "opti_oignon.inference_backend"),
        seeded=seeded,
        packages=("opti_oignon",),
    )
    try:
        mod = loaded["opti_oignon.verification"]
        engine = mod.VerificationEngine(structured_engine=None, code_exec=None, max_iterations=1)
        assert engine._attempt_fix("print(1", "SyntaxError", "python", "print one", "m", 1) is None
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
