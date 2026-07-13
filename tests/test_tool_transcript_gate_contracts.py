#!/usr/bin/env python3
"""Contracts for the per-model allowlist gating native tool transcripts.

The executor already carries two transcript shapes (flat and native) and
their own contracts. What ships here is the POSITION of the switch: the
configuration alone must never be able to turn the native replay on for
an arbitrary model. These contracts pin who decides, with which model,
and under what default:

  * Contract G1 -- a configured ``tool_transcript: native`` with the
    ``tool_transcript_models`` allowlist absent, or with a malformed
    (non-list) value, keeps the flat shapes byte-exact at both
    consumption sites (decision rebuild and final generation).
  * Contract G2 -- an allowlist that does not name the run's model
    (empty, or naming only other models) keeps flat despite the global
    native flag.
  * Contract G3 -- the gate actually opens: with the run's model listed
    (whitespace-tolerant exact name), the decision rounds carry the
    trained assistant/tool roles and the final generation ends on the
    last tool message. The deep native shape stays owned by the
    transcript contracts; this clause only pins that listing the model
    is what opens the path.
  * Contract G4 -- a configuration that raises while reading the
    allowlist keeps flat (fail-secure on unreadable configuration).
  * Contract G5 -- a flat request never enters the gate: with the
    preference flat and a permissive allowlist, the shapes stay
    byte-exact flat and the allowlist key is never even read.

An explicitly constructed mode (the eval-harness channel) is not gated;
that channel is pinned inert-by-default by the existing transcript
contracts, which every probe here must keep green.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling transcript
contracts: canonical dotted names so package and relative imports resolve
against the loaded copies, a minimal pydantic stand-in only when the real
package is absent, and a meta-path guard sealing the isolation window.
"""

import copy
import importlib.util
import json
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
    ignores the parent path, so a real submodule resolves behind the
    test's back -- silently importing live code. This guard sits ahead of
    every finder and refuses the names that were not seeded, so a load
    behaves identically whether the project is installed or not.
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
# Minimal pydantic stand-in (only when the real package is absent)
# ---------------------------------------------------------------------------
def _pydantic_shim() -> types.ModuleType:
    mod = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for name in getattr(self.__class__, "__annotations__", {}):
                default = getattr(self.__class__, name, None)
                if isinstance(default, (list, dict)):
                    default = type(default)(default)
                setattr(self, name, default)
            for key, value in kwargs.items():
                setattr(self, key, value)

    mod.BaseModel = BaseModel
    return mod


# ---------------------------------------------------------------------------
# Deterministic configuration stand-in (recorder)
# ---------------------------------------------------------------------------
class _ConfigStub:
    """User-preference seam with scripted values and a read recorder."""

    def __init__(self, prefs=None, raise_on=()):
        self.prefs = dict(prefs or {})
        self.raise_on = set(raise_on)
        self.reads: list[tuple] = []

    def get_user_preference(self, key, default=None):
        self.reads.append((key, default))
        if key in self.raise_on:
            raise RuntimeError("scripted unreadable configuration")
        return self.prefs.get(key, default)


def _config_module(stub: _ConfigStub) -> types.ModuleType:
    mod = types.ModuleType("opti_oignon.config")
    mod.config = stub
    return mod


# ---------------------------------------------------------------------------
# Isolated loading (sibling-transcript idiom plus the meta-path guard)
# ---------------------------------------------------------------------------
def _load(prefs=None, raise_on=()):
    keys = (
        "pydantic", "ollama", "opti_oignon", "opti_oignon.tool_calling",
        "opti_oignon.tool_registry", "opti_oignon.structured_output",
        "opti_oignon.response_hygiene", "opti_oignon.tool_executor",
        "opti_oignon.config", "opti_oignon.agent_eval",
        "opti_oignon.agent_eval.tasks", "opti_oignon.agent_eval.chat_runner",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    try:
        import pydantic  # noqa: F401
    except ImportError:
        sys.modules["pydantic"] = _pydantic_shim()

    ollama_stub = types.ModuleType("ollama")
    ollama_stub.chat = lambda **kw: None
    sys.modules["ollama"] = ollama_stub

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    stub = _ConfigStub(prefs=prefs, raise_on=raise_on)
    cfg = _config_module(stub)
    sys.modules["opti_oignon.config"] = cfg
    pkg.config = cfg

    def _real(dotted: str, path: Path):
        spec = importlib.util.spec_from_file_location(dotted, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    pkg.tool_calling = _real(
        "opti_oignon.tool_calling", _OO / "tool_calling.py",
    )
    pkg.tool_registry = _real(
        "opti_oignon.tool_registry", _OO / "tool_registry.py",
    )

    so = types.ModuleType("opti_oignon.structured_output")
    so.StructuredOutputEngine = object
    so.ToolCallRequest = object
    so.structured_output_engine = None
    so.STRUCTURED_OUTPUT_AVAILABLE = False
    sys.modules["opti_oignon.structured_output"] = so
    pkg.structured_output = so

    pkg.response_hygiene = _real(
        "opti_oignon.response_hygiene", _OO / "response_hygiene.py",
    )
    te = _real("opti_oignon.tool_executor", _OO / "tool_executor.py")
    pkg.tool_executor = te

    ae = types.ModuleType("opti_oignon.agent_eval")
    ae.__path__ = []
    sys.modules["opti_oignon.agent_eval"] = ae
    pkg.agent_eval = ae
    ae.tasks = _real(
        "opti_oignon.agent_eval.tasks", _OO / "agent_eval" / "tasks.py",
    )
    cr = _real(
        "opti_oignon.agent_eval.chat_runner",
        _OO / "agent_eval" / "chat_runner.py",
    )
    ae.chat_runner = cr

    if not cr.FEATURE_AVAILABLE:
        raise RuntimeError("harness reports FEATURE_AVAILABLE False")

    def restore():
        if guard in sys.meta_path:
            sys.meta_path.remove(guard)
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return cr, te, stub, restore


# ---------------------------------------------------------------------------
# Local material
# ---------------------------------------------------------------------------
MSG = "Read notes.txt and report."

FINAL_TEXT = "I read notes.txt with my file tool; the source said alpha."

_FLAT_ENTRY = (
    "[environment] tool call by assistant: read_file\n"
    "Arguments: {'filename': 'notes.txt'}\n"
    "Result: alpha\n"
)


def _flat_decision(te) -> list[dict]:
    return [{
        "role": "user",
        "content": (
            f"{MSG}\n\n{te.ENV_RESULTS_HEADER}\n{_FLAT_ENTRY}\n\n"
            "Call the next tool if needed; otherwise answer directly. "
            "Never attribute these tool actions to the user."
        ),
    }]


def _flat_final(te) -> list[dict]:
    return [
        {"role": "system", "content": te.FINAL_ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{MSG}\n\n{te.ENV_RESULTS_HEADER}\n{_FLAT_ENTRY}\n\n"
                "Write the final user-facing answer, reporting in "
                "first person what you did and what the results were."
            ),
        },
    ]


class _RecordingClient:
    """ScriptedChatClient wrapper capturing the full messages of each call."""

    def __init__(self, inner):
        self._inner = inner
        self.captured: list[dict] = []

    def chat(self, model=None, messages=None, tools=None, options=None,
             stream=False, **kwargs):
        self.captured.append({
            "stream": bool(stream),
            "tools": tools is not None,
            "messages": copy.deepcopy(list(messages or [])),
        })
        return self._inner.chat(
            model=model, messages=messages, tools=tools, options=options,
            stream=stream, **kwargs,
        )

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _run_flow(cr, te):
    """Drive one read-then-answer scripted flow through the executor.

    The executor is built WITHOUT an explicit transcript argument, so its
    mode comes from the configuration channel -- the channel under test.
    Returns ``(executor, captured_calls, result)``.
    """
    workspace = cr.VirtualWorkspace({"notes.txt": "alpha\n"})
    registry = cr.build_scripted_registry(workspace)
    turns = [
        cr.ScriptTurn(tool_calls=[
            {"name": "read_file", "arguments": {"filename": "notes.txt"}},
        ]),
        cr.ScriptTurn(content=""),
        cr.ScriptTurn(content=FINAL_TEXT),
    ]
    client = _RecordingClient(cr.ScriptedChatClient(turns))
    executor = te.ToolExecutor(
        registry=registry,
        structured_engine=cr._NoDecisionEngine(),
        max_tool_calls=6,
        default_model=cr.SCRIPTED_MODEL,
    )
    with cr.scripted_chat_backend(client):
        result = executor.execute_with_tools(message=MSG)
    assert client.leftover == 0 and client.overrun == 0, (
        client.leftover, client.overrun,
    )
    return executor, client.captured, result


def _assert_flat_shapes(te, executor, calls, result, requested):
    """The both-site flat verdict shared by the flat-position clauses."""
    assert executor.tool_transcript == requested, executor.tool_transcript
    assert executor._transcript_explicit is False
    assert len(calls) == 3, [(c["stream"], c["tools"]) for c in calls]
    assert calls[1]["messages"] == _flat_decision(te), calls[1]["messages"]
    assert calls[2]["messages"] == _flat_final(te), calls[2]["messages"]
    joined = json.dumps([c["messages"] for c in calls])
    assert '"role": "tool"' not in joined
    assert result.response == FINAL_TEXT, result.response


# ---------------------------------------------------------------------------
# Contract G1 -- absent or malformed allowlist keeps flat
# ---------------------------------------------------------------------------
def test_g1_absent_or_malformed_allowlist_keeps_flat():
    # Face 1: the allowlist key is absent entirely.
    cr, te, _stub, restore = _load(prefs={"tool_transcript": "native"})
    try:
        executor, calls, result = _run_flow(cr, te)
        _assert_flat_shapes(
            te, executor, calls, result, te.TOOL_TRANSCRIPT_NATIVE,
        )
    finally:
        restore()

    # Face 2: the allowlist is present but malformed (not a list).
    cr, te, _stub, restore = _load(prefs={
        "tool_transcript": "native",
        "tool_transcript_models": "qwen3-scripted",
    })
    try:
        executor, calls, result = _run_flow(cr, te)
        _assert_flat_shapes(
            te, executor, calls, result, te.TOOL_TRANSCRIPT_NATIVE,
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract G2 -- an allowlist not naming the model keeps flat
# ---------------------------------------------------------------------------
def test_g2_unlisted_model_keeps_flat_despite_global_flag():
    # Face 1: the allowlist is an empty list (the shipped default).
    cr, te, _stub, restore = _load(prefs={
        "tool_transcript": "native",
        "tool_transcript_models": [],
    })
    try:
        executor, calls, result = _run_flow(cr, te)
        _assert_flat_shapes(
            te, executor, calls, result, te.TOOL_TRANSCRIPT_NATIVE,
        )
    finally:
        restore()

    # Face 2: the allowlist names only other models (and junk entries).
    cr, te, _stub, restore = _load(prefs={
        "tool_transcript": "native",
        "tool_transcript_models": ["other-model:7b", 7, None],
    })
    try:
        executor, calls, result = _run_flow(cr, te)
        _assert_flat_shapes(
            te, executor, calls, result, te.TOOL_TRANSCRIPT_NATIVE,
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract G3 -- listing the run's model is what opens the native path
# ---------------------------------------------------------------------------
def test_g3_listed_model_opens_native():
    cr, te, _stub, restore = _load(prefs={
        "tool_transcript": "native",
        "tool_transcript_models": ["  qwen3-scripted  ", "other-model:7b"],
    })
    try:
        executor, calls, result = _run_flow(cr, te)
        assert executor.tool_transcript == te.TOOL_TRANSCRIPT_NATIVE
        assert executor._transcript_explicit is False
        assert len(calls) == 3, [(c["stream"], c["tools"]) for c in calls]

        # Decision round 2 replays the trained shape.
        m = calls[1]["messages"]
        assert [x["role"] for x in m] == ["user", "assistant", "tool"], m
        assert m[1]["tool_calls"][0]["function"]["name"] == "read_file"
        assert m[2]["content"].startswith(
            "[environment] tool result (untrusted content): read_file",
        ), m[2]

        # The final generation is led by the attribution system message
        # and ends on the last tool message; no flat header anywhere.
        m = calls[2]["messages"]
        assert m[0] == {
            "role": "system", "content": te.FINAL_ANSWER_SYSTEM_PROMPT,
        }
        assert m[-1]["role"] == "tool", m
        joined = json.dumps([c["messages"] for c in calls])
        assert te.ENV_RESULTS_HEADER not in joined

        assert result.response == FINAL_TEXT, result.response
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract G4 -- an unreadable configuration keeps flat
# ---------------------------------------------------------------------------
def test_g4_unreadable_configuration_keeps_flat():
    cr, te, _stub, restore = _load(
        prefs={"tool_transcript": "native"},
        raise_on={"tool_transcript_models"},
    )
    try:
        executor, calls, result = _run_flow(cr, te)
        _assert_flat_shapes(
            te, executor, calls, result, te.TOOL_TRANSCRIPT_NATIVE,
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract G5 -- a flat request never enters the gate
# ---------------------------------------------------------------------------
def test_g5_flat_request_never_consults_the_gate():
    cr, te, stub, restore = _load(prefs={
        "tool_transcript": "flat",
        "tool_transcript_models": ["qwen3-scripted"],
    })
    try:
        executor, calls, result = _run_flow(cr, te)
        _assert_flat_shapes(
            te, executor, calls, result, te.TOOL_TRANSCRIPT_FLAT,
        )
        touched = [key for key, _default in stub.reads]
        assert "tool_transcript_models" not in touched, touched
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
