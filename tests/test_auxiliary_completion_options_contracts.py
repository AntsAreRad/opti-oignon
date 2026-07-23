#!/usr/bin/env python3
"""What the completion paths that carry no conversation are allowed to send.

The external llama-server keeps one prompt-KV cache per slot. Told which
slot to decode on, it honours the instruction; told nothing, it picks one
itself, and the one it picks may be the slot a conversation is holding.
The per-conversation slot decision exists so that two conversations never
land on the same slot -- but it is taken in one place only, the streaming
hub, because it needs an identity to key on and the hub is where an
identity exists.

Two other paths reach the same server: question refinement and simple
execution. Neither is given a conversation, so neither has anything to
key a slot on, and the invariant the hub upholds says nothing about
them. What they must not do is participate in the slot cache at all: a
request that names no slot but asks the server to reuse a cached prefix
is a request to be decoded on whatever attention state the chosen slot
already holds, which is some conversation's, and a request that names a
slot it did not earn takes one from a conversation that did.

Both paths build their options literally, from a temperature and nothing
else. That is the property pinned here, and it is pinned as a property
rather than left as an accident of how the code reads today: these
contracts are already satisfied by the tree they were written against,
so they are witnesses, and the whole of their proof is that a change on
either path reddens them.

The last contract pins the asymmetry itself. The hub does ask for prompt
reuse, deliberately, because it also names the slot the reuse applies
to. Reading the two facts side by side is what makes the difference
between them intentional: making the auxiliary paths symmetric with the
hub, without giving them a slot to be symmetric on, would turn a request
that merely displaces a cached prefix into one decoded on top of it.

Loaded through the shared isolation window with a scripted backend; no
server, no model, no network is ever reached.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_AFFINITY = "opti_oignon.slot_affinity"
_OPTIMIZER = "opti_oignon.context_optimizer"
_EXECUTOR = "opti_oignon.executor"

_MODEL = "test-model:1b"
_CACHE_SWITCH = "cache_prompt"
_SLOT = "id_slot"


class _Recorder:
    """A backend that answers immediately and remembers what it was sent."""

    def __init__(self, name="llama_server", slot_count=4):
        self.name = name
        self.calls = []
        self.slot_reads = 0
        self._listing = [
            {"id": i, "id_task": -1} for i in range(slot_count)
        ]

    def slots(self):
        self.slot_reads += 1
        return list(self._listing)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content="ok", thinking=None)

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        yield SimpleNamespace(content="ok", thinking=None)


def _load(*, prefix_on, affinity_on):
    """Open the window on the executor with a scripted backend behind it."""
    recorder = _Recorder()
    registry = SimpleNamespace(
        active=recorder, resolve_backend=lambda model: recorder,
    )

    ollama_stub = types.ModuleType("ollama")
    ollama_stub.chat = lambda **k: iter([{"message": {"content": "ok"}}])

    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(
        get_model=lambda *a, **k: _MODEL,
        get_temperature=lambda *a, **k: 0.2,
    )
    router = types.ModuleType("opti_oignon.router")
    router.RoutingResult = type("RoutingResult", (), {})
    backends = types.ModuleType("opti_oignon.inference_backend")
    backends.get_backend_registry = lambda: registry

    seeded = {
        "opti_oignon.config": cfg,
        "opti_oignon.router": router,
        "opti_oignon.inference_backend": backends,
    }
    # The affinity module must enter the window BEFORE the executor is
    # executed, or the executor's guarded import answers unavailable and
    # the contract passes on an absence the test built for itself.
    targets = {
        _AFFINITY: source("slot_affinity.py"),
        _OPTIMIZER: source("context_optimizer.py"),
        # The deduplicator rides along as a real target: the hub imports
        # it plainly, and it is pure and standard-library only.
        "opti_oignon.context_dedup": source("context_dedup.py"),
        _EXECUTOR: source("executor.py"),
    }
    had = "ollama" in sys.modules
    previous = sys.modules.get("ollama")
    sys.modules["ollama"] = ollama_stub
    loaded, win_restore = isolate(
        targets=targets, seeded=seeded, packages=("opti_oignon",),
    )
    loaded[_OPTIMIZER].init_optimizer(
        config={
            "enabled": False,
            "stable_prefix": {"enabled": bool(prefix_on)},
            "slot_affinity": {"enabled": bool(affinity_on)},
        }
    )
    assert loaded[_EXECUTOR].SLOT_AFFINITY_AVAILABLE, (
        "the window must carry the affinity module, not its absence"
    )

    def restore():
        win_restore()
        if had:
            sys.modules["ollama"] = previous
        else:
            sys.modules.pop("ollama", None)

    return loaded[_EXECUTOR], recorder, restore


def _refine(mod, recorder):
    """Drive question refinement and return the options that reached the wire."""
    mod.Executor().refine_question("q", model=_MODEL)
    assert recorder.calls, "the refinement never reached the scripted backend"
    return recorder.calls[-1].get("options") or {}


def _simple(mod, recorder):
    """Drive simple execution and return the options that reached the wire."""
    mod.Executor().execute_simple("q", _MODEL, "you are a test")
    assert recorder.calls, "the execution never reached the scripted backend"
    return recorder.calls[-1].get("options") or {}


def _hub(mod, recorder, conversation_id="conv-a"):
    """Drive the streaming hub and return the options that reached the wire."""
    routing = SimpleNamespace(
        model=_MODEL, task_type="general", temperature=0.2,
        prompt_variant="standard", timeout=30,
    )
    generator = mod.Executor().execute(
        "q", routing, refine=False,
        conversation_id=conversation_id, persist=False,
    )
    try:
        while True:
            next(generator)
    except StopIteration:
        pass
    assert recorder.calls, "the hub never reached the scripted backend"
    return recorder.calls[-1].get("options") or {}


# ---------------------------------------------------------------------------
# x1 -- refinement never asks the server to reuse a cached prefix
# ---------------------------------------------------------------------------

def test_x1_refinement_asks_for_no_prompt_reuse():
    mod, recorder, restore = _load(prefix_on=True, affinity_on=True)
    try:
        options = _refine(mod, recorder)
        assert _CACHE_SWITCH not in options, (
            "question refinement asked the server to reuse a cached prefix "
            "while naming no slot: the prefix it is decoded on belongs to "
            "whichever conversation the server happens to pick"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# x2 -- refinement never names a slot
# ---------------------------------------------------------------------------

def test_x2_refinement_names_no_slot():
    mod, recorder, restore = _load(prefix_on=True, affinity_on=True)
    try:
        options = _refine(mod, recorder)
        assert _SLOT not in options, (
            "question refinement named a slot it has no conversation to "
            "have earned: the slot it names is one a conversation holds"
        )
        assert recorder.slot_reads == 0, (
            "the refinement read the slot listing; it has no decision to take"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# x3 -- simple execution never asks for prompt reuse
# ---------------------------------------------------------------------------

def test_x3_simple_execution_asks_for_no_prompt_reuse():
    mod, recorder, restore = _load(prefix_on=True, affinity_on=True)
    try:
        options = _simple(mod, recorder)
        assert _CACHE_SWITCH not in options, (
            "simple execution asked the server to reuse a cached prefix "
            "while naming no slot"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# x4 -- simple execution never names a slot
# ---------------------------------------------------------------------------

def test_x4_simple_execution_names_no_slot():
    mod, recorder, restore = _load(prefix_on=True, affinity_on=True)
    try:
        options = _simple(mod, recorder)
        assert _SLOT not in options, (
            "simple execution named a slot it has no conversation to have "
            "earned"
        )
        assert recorder.slot_reads == 0, (
            "simple execution read the slot listing; it has no decision to "
            "take"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# x5 -- the asymmetry is deliberate: the hub does both, and says so
# ---------------------------------------------------------------------------

def test_x5_the_hub_pairs_prompt_reuse_with_a_named_slot():
    mod, recorder, restore = _load(prefix_on=True, affinity_on=True)
    try:
        options = _hub(mod, recorder)
        assert options.get(_CACHE_SWITCH) is True, (
            "the hub stopped asking for prompt reuse; the asymmetry the "
            "auxiliary contracts describe no longer has two sides"
        )
        assert _SLOT in options, (
            "the hub asked for prompt reuse without naming the slot the "
            "reuse applies to: reuse and a named slot travel together or "
            "neither travels"
        )
    finally:
        restore()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
