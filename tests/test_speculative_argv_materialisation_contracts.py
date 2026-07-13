#!/usr/bin/env python3
"""Contracts for the llama-server argv materialisation.

build_llama_server_command turns a SpeculativeConfig into the argv an
external llama-server is launched with. The function is pure -- no reads, no
spawning -- and the argv is a LIST, so there is no shell to inject into. What
was never pinned is that it stays that way: that validation is loud rather
than guessed, that numeric fields are materialised as integers, and above all
that the draft posture is exact -- an enabled config with a draft model emits
the draft flag quad, while an enabled config WITHOUT a draft model emits no
draft flags at all (the self-drafting posture, where the draft lives inside
the model file and the server applies it natively). A regression that emitted
an empty ``-md`` flag, or dropped validation, would misconfigure a launch
silently.

The clauses pin distinct mechanisms so that one probe reddens exactly one
clause:

  * Contract Q1 -- an empty or whitespace model path raises. Validation is
    loud, never a bare command with a missing model.
  * Contract Q2 -- a config whose own validate() reports errors raises. A
    distinct mechanism from Q1 (the path check), so a probe on one does not
    redden the other even though both converge on ValueError.
  * Contract Q3 -- the draft posture is exact: enabled + draft model emits
    the -md / --draft-max / --draft-min / -ngld quad; enabled WITHOUT a draft
    model emits none of them; disabled emits none of them.
  * Contract Q4 -- numeric fields (port, n_ctx) are materialised as integers,
    so a float slipping through the type hint never reaches the argv verbatim.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the house idiom: canonical
dotted names, an empty-path package stand-in, and a meta-path guard sealing
the isolation window.
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


# ---------------------------------------------------------------------------
# Isolated loading of the speculative decoding module. Its module-top imports
# are stdlib plus yaml; no sibling project module needs seeding.
# ---------------------------------------------------------------------------
def _load():
    """Load speculative_decoding in isolation; returns (module, restore)."""
    keys = ("opti_oignon", "opti_oignon.speculative_decoding")
    saved = {k: sys.modules.get(k) for k in keys}

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.speculative_decoding", _OO / "speculative_decoding.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.speculative_decoding"] = mod
    spec.loader.exec_module(mod)
    pkg.speculative_decoding = mod

    def restore():
        if guard in sys.meta_path:
            sys.meta_path.remove(guard)
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


_MODEL = "/models/main.gguf"
_DRAFT = "/models/draft.gguf"


# ---------------------------------------------------------------------------
# Contract Q1 -- an empty or whitespace model path raises
# ---------------------------------------------------------------------------
def test_q1_empty_model_path_raises():
    mod, restore = _load()
    try:
        cfg = mod.SpeculativeConfig(enabled=False)
        for bad in ("", "   ", "\t"):
            try:
                mod.build_llama_server_command(bad, cfg)
            except ValueError:
                continue
            raise AssertionError(f"model_path {bad!r} must raise ValueError")

        # A valid path with the same config does NOT raise: isolates the
        # refusal to the path check.
        cmd = mod.build_llama_server_command(_MODEL, cfg)
        assert cmd[:2] == ["llama-server", "-m"] and cmd[2] == _MODEL
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract Q2 -- a config whose validate() reports errors raises
# ---------------------------------------------------------------------------
def test_q2_invalid_config_raises():
    mod, restore = _load()
    try:
        # draft_min > draft_max is a validate() error; the path is valid, so
        # this isolates the refusal to the config-validation mechanism.
        bad_cfg = mod.SpeculativeConfig(
            enabled=True, draft_model=_DRAFT, draft_max=4, draft_min=9,
        )
        assert bad_cfg.validate(), "precondition: this config must be invalid"
        try:
            mod.build_llama_server_command(_MODEL, bad_cfg)
        except ValueError:
            pass
        else:
            raise AssertionError("an invalid config must raise ValueError")

        # A valid config with the same valid path does NOT raise.
        good_cfg = mod.SpeculativeConfig(
            enabled=True, draft_model=_DRAFT, draft_max=16, draft_min=5,
        )
        assert not good_cfg.validate()
        cmd = mod.build_llama_server_command(_MODEL, good_cfg)
        assert "llama-server" in cmd[0]
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract Q3 -- the draft posture is exact
# ---------------------------------------------------------------------------
def test_q3_draft_posture_is_exact():
    mod, restore = _load()
    try:
        quad = {"-md", "--draft-max", "--draft-min", "-ngld"}

        # Enabled WITH a draft model: the full quad is present, and -md
        # carries the draft path.
        cfg_on = mod.SpeculativeConfig(
            enabled=True, draft_model=_DRAFT, draft_max=16, draft_min=5,
            draft_gpu_layers=99,
        )
        cmd_on = mod.build_llama_server_command(_MODEL, cfg_on)
        assert quad.issubset(set(cmd_on)), (
            f"enabled+draft must emit the draft quad; got {cmd_on}"
        )
        assert cmd_on[cmd_on.index("-md") + 1] == _DRAFT, (
            "-md must carry the draft model path"
        )

        # Enabled WITHOUT a draft model: the self-draft posture. No draft
        # flags at all -- crucially no empty -md.
        cfg_mtp = mod.SpeculativeConfig(enabled=True, draft_model="")
        cmd_mtp = mod.build_llama_server_command(_MODEL, cfg_mtp)
        assert quad.isdisjoint(set(cmd_mtp)), (
            f"enabled without a draft model must emit NO draft flags; got {cmd_mtp}"
        )

        # Disabled: likewise no draft flags.
        cfg_off = mod.SpeculativeConfig(enabled=False, draft_model=_DRAFT)
        cmd_off = mod.build_llama_server_command(_MODEL, cfg_off)
        assert quad.isdisjoint(set(cmd_off)), (
            f"a disabled config must emit NO draft flags; got {cmd_off}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract Q4 -- numeric fields are materialised as integers
# ---------------------------------------------------------------------------
def test_q4_numeric_fields_are_integers():
    mod, restore = _load()
    try:
        cfg = mod.SpeculativeConfig(enabled=False)
        # Floats slipped past the type hints must not reach the argv verbatim.
        cmd = mod.build_llama_server_command(
            _MODEL, cfg, host="127.0.0.1", port=8080.9, n_ctx=4096.7,
        )
        port_val = cmd[cmd.index("--port") + 1]
        assert port_val == "8080", (
            f"port must materialise as an integer string, got {port_val!r}"
        )
        ctx_val = cmd[cmd.index("-c") + 1]
        assert ctx_val == "4096", (
            f"n_ctx must materialise as an integer string, got {ctx_val!r}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner (pytest picks up the test_ functions; direct execution works too)
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    names = sorted(n for n in globals() if n.startswith("test_"))
    selected = [
        n for n in names if not argv or any(fragment in n for fragment in argv)
    ]
    failures = 0
    for name in selected:
        try:
            globals()[name]()
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"{len(selected) - failures}/{len(selected)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
