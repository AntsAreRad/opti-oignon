#!/usr/bin/env python3
"""The llama.cpp load seam: where the integrity gate is wired in.

``LlamaCppBackend._get_or_load`` is the one place in the codebase that hands a
GGUF file to native code. Two guards already stood in front of it and both
answer the same question: ``_resolve_model_path`` proves the file sits inside
a configured model directory, and the SSRF defence proves it was fetched from
a routable public host. Neither proves the bytes are the bytes we pinned, and
the docstring of ``_resolve_model_path`` names the threat it does not close --
a trojaned model exploiting the native parser.

This suite pins the gate that closes it, and in particular pins the posture
that distinguishes it from the resource governor sitting beside it:

  * B1 a refusal is LOUD and EARLY -- it propagates out of the load, and the
    model is never constructed. A gate that runs after the parser has already
    read the file is not a gate.
  * B2 the gate is handed the path containment RESOLVED, not the string the
    caller supplied -- a check pointed at the wrong object is not a check --
    and a verified model then loads exactly as before.
  * B3 an unresolvable provenance module is a REFUSAL when the mode enforces.
    This is the whole argument. The governor gate above fails open on purpose,
    because an absent resource governor means an unguarded but otherwise
    correct load. An absent integrity proof does not mean "load unverified" --
    it means no proof exists. Swallowing that import would put a silent
    fail-open on the one seam that feeds raw bytes to a native parser.
  * B4 the same unresolvable module in Daily does NOT block: an installation
    that has not enrolled its models yet keeps working, which is what makes
    B3 affordable rather than a brick.
  * B5 an unresolvable model name is refused by containment BEFORE the gate is
    ever consulted -- the graft does not reorder or weaken the path guard, and
    a traversal never reaches the hashing code.

Isolation follows the house idiom. ``inference_backend`` imports stdlib only
at module top (the Ollama and llama.cpp SDKs load lazily inside methods), so
the backend is exercised with a recording stand-in for the Llama constructor
and with the provenance module either seeded or deliberately absent -- the
absence being the very condition B3 and B4 are about.

Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


class _IsolationGuard:
    """Refuse every project submodule the test did not seed.

    A stand-in package whose ``__path__`` is empty isolates the tree only
    while the parent path is the sole way to resolve a submodule. That
    assumption breaks wherever the project is installed in editable mode:
    such an install registers a finder that answers on the module NAME and
    ignores the parent path, so a real submodule resolves behind the test's
    back -- silently importing live code. This guard sits ahead of every
    finder and refuses the names that were not seeded, so a load behaves
    identically whether the project is installed or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


class _ProvenanceRefusalStub(Exception):
    """Stands in for model_provenance.ProvenanceRefusal."""


class _LlamaRecorder:
    """Stand-in for the Llama constructor; counts how often it ran.

    The count is the observable that separates "refused" from "refused too
    late": a gate that raises only after the native parser has already read
    the file has not protected anything.
    """

    def __init__(self):
        self.constructions = 0

    def __call__(self, **kwargs):
        self.constructions += 1
        return object()


# Every project module inference_backend reaches for lazily. Each one is either
# seeded by the test or blocked outright; nothing is left to whatever an
# earlier test happened to leave in the module cache.
_LAZY = (
    "opti_oignon.model_provenance",
    "opti_oignon.security_mode",
    "opti_oignon.resource_governor",
    "opti_oignon.telemetry",
)


def _load(*, provenance=None, mode=None):
    """Load inference_backend in isolation; returns (module, restore).

    ``provenance`` seeds a stand-in provenance module; leaving it None makes
    the lazy import fail, which is the condition B3 and B4 are about.
    ``mode`` seeds a stand-in security_mode; leaving it None makes the mode
    unresolvable, which the seam must read as the fortress.

    Absence has to be MANUFACTURED, not merely hoped for. Python consults
    sys.modules before any finder, so a module some earlier test already
    imported for real would resolve out of the cache and the meta-path guard
    would never be asked -- and B3, whose whole subject is what happens when
    the provenance module cannot be reached, would silently test nothing.
    Blocking each unseeded name with None raises ImportError ahead of every
    finder and makes the absence real.
    """
    keys = ("opti_oignon", "opti_oignon.inference_backend") + _LAZY
    saved = {k: sys.modules.get(k) for k in keys}

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    for name in _LAZY:
        sys.modules[name] = None

    if provenance is not None:
        sys.modules["opti_oignon.model_provenance"] = provenance
        pkg.model_provenance = provenance
    if mode is not None:
        sys.modules["opti_oignon.security_mode"] = mode
        pkg.security_mode = mode

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.inference_backend", _OO / "inference_backend.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.inference_backend"] = mod
    spec.loader.exec_module(mod)
    pkg.inference_backend = mod

    def restore():
        if guard in sys.meta_path:
            sys.meta_path.remove(guard)
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


def _provenance_stub(*, refuse=False):
    """A stand-in provenance module whose gate either passes or refuses."""
    stub = types.ModuleType("opti_oignon.model_provenance")
    stub.ProvenanceRefusal = _ProvenanceRefusalStub
    stub.calls = []

    def guard_model_load(path, **kwargs):
        stub.calls.append(Path(path))
        if refuse:
            raise _ProvenanceRefusalStub(f"digest mismatch: {Path(path).name}")
        return True

    stub.guard_model_load = guard_model_load
    return stub


def _mode_stub(name):
    stub = types.ModuleType("opti_oignon.security_mode")
    stub.get_current_mode = lambda: name
    return stub


_NAME = "pinned-model-Q4_K_M.gguf"


def _backend_with_model(mod, tmp_path, recorder):
    """A LlamaCppBackend over a real model dir, with the loader stubbed out."""
    (tmp_path / _NAME).write_bytes(b"GGUF" + b"\x00" * 64)
    mod.LLAMA_CPP_AVAILABLE = True
    mod._LlamaCpp = recorder
    return mod.LlamaCppBackend(model_dirs=[str(tmp_path)])


# ---------------------------------------------------------------------------
# B1-B2: the gate on the happy and the refused path
# ---------------------------------------------------------------------------


def test_b1_refusal_is_loud_and_the_model_is_never_constructed(tmp_path):
    prov = _provenance_stub(refuse=True)
    mod, restore = _load(provenance=prov, mode=_mode_stub("daily"))
    try:
        recorder = _LlamaRecorder()
        backend = _backend_with_model(mod, tmp_path, recorder)

        try:
            backend._get_or_load(_NAME)
        except _ProvenanceRefusalStub:
            pass
        else:
            raise AssertionError("a refused model was loaded anyway")

        # The refusal propagated as itself -- not swallowed into a log line --
        # AND it arrived before the parser did. A gate that raises only after
        # the native constructor has already read the file has protected
        # nothing, so the construction count is the assertion that matters.
        assert recorder.constructions == 0
        assert _NAME not in backend._loaded_models
    finally:
        restore()


def test_b2_gate_is_handed_the_contained_path(tmp_path):
    prov = _provenance_stub(refuse=False)
    mod, restore = _load(provenance=prov, mode=_mode_stub("daily"))
    try:
        recorder = _LlamaRecorder()
        backend = _backend_with_model(mod, tmp_path, recorder)

        backend._get_or_load(_NAME)

        # The gate must hash the file containment RESOLVED, not the string the
        # caller supplied. Handed the raw name it would hash some other file or
        # none at all, and the pin would verify something the parser never
        # reads -- a check pointed at the wrong object is not a check.
        assert prov.calls == [tmp_path / _NAME]
        assert recorder.constructions == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# B3-B4: the fail-open that must not exist, and the escape hatch that must
# ---------------------------------------------------------------------------


def test_b3_unavailable_provenance_refuses_when_the_mode_enforces(tmp_path):
    # No provenance module seeded and no security module either: the mode is
    # undeterminable, which is the fortress. This is a broken or partial
    # installation, and it is exactly the case where a silent fallback would
    # hand unverified bytes to the native parser.
    mod, restore = _load(provenance=None, mode=None)
    try:
        recorder = _LlamaRecorder()
        backend = _backend_with_model(mod, tmp_path, recorder)

        assert mod._provenance_mode() == "bulbe"

        try:
            backend._get_or_load(_NAME)
        except RuntimeError as exc:
            assert "provenance" in str(exc).lower()
        else:
            raise AssertionError(
                "an unverifiable model loaded while the mode enforced"
            )
    finally:
        restore()


def test_b4_unavailable_provenance_does_not_block_daily(tmp_path):
    # Same unresolvable module, but the mode is Daily. Nothing blocks: an
    # installation that has not enrolled its models keeps working, which is
    # what makes B3 an affordable posture rather than a brick.
    mod, restore = _load(provenance=None, mode=_mode_stub("daily"))
    try:
        recorder = _LlamaRecorder()
        backend = _backend_with_model(mod, tmp_path, recorder)

        backend._get_or_load(_NAME)

        assert recorder.constructions == 1
        assert _NAME in backend._loaded_models
    finally:
        restore()


# ---------------------------------------------------------------------------
# B5: containment still comes first
# ---------------------------------------------------------------------------


def test_b5_unresolvable_model_is_refused_before_the_gate(tmp_path):
    prov = _provenance_stub(refuse=False)
    mod, restore = _load(provenance=prov, mode=_mode_stub("daily"))
    try:
        recorder = _LlamaRecorder()
        backend = _backend_with_model(mod, tmp_path, recorder)

        # A traversal name and an absolute path both fail containment. They
        # must be refused there and never reach the gate: hashing a file the
        # path guard already rejected would mean the graft had reordered the
        # defences it was supposed to sit behind.
        for hostile in ("../../../etc/passwd.gguf", "/etc/passwd.gguf"):
            try:
                backend._get_or_load(hostile)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError(f"containment did not refuse: {hostile}")

        assert prov.calls == []
        assert recorder.constructions == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _main() -> int:
    import tempfile

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(Path(tmp))
                print(f"PASS {fn.__name__}")
            except Exception:
                failed += 1
                print(f"FAIL {fn.__name__}")
                traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
