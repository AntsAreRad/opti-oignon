#!/usr/bin/env python3
"""Contracts for French-language data, the verification hint in native
transcript mode, the long-chain growth curve, and the agentic dispatch of
the chat tool loop.

  * Contract 1 -- the French suite is green end to end: accented prompts,
    accented tool arguments and filenames, and typographic apostrophes in
    first-person answers pass every hygiene checker on both fronts, while
    an accented second-person answer is still caught by the
    misattribution checker (checked through a throwaway trap suite, never
    a shipped task).
  * Contract 2 -- the verification hint in native mode: a scripted write
    failure (reserved read-only prefix) makes the executor run exactly
    one corrective iteration; the hint travels as a short user message
    with the environment prefix, placed right after the failing
    echo/tool pair in the native transcript, and the final generation
    still leads with the attribution system message and ends on the last
    tool message. The flat path carries the same hint folded into the
    rebuilt user message, unchanged in shape.
  * Contract 3 -- the growth curve on an eight-call chain: both decision
    context curves are strictly increasing, the native total is smaller,
    and the flat-minus-native gap widens every round (the re-folded
    reconstruction against the marginal pair cost); the final generation
    input is smaller in native mode too, and hygiene stays green on both
    sides.
  * Contract 4 -- the agentic tools pipeline against the scripted
    harness: the streaming branch is taken, tool activity is relayed
    through the callback before the first answer chunk, the chunks
    reassemble the exact scripted answer, the executed calls are
    recorded, and the direct-pipeline fallback is never touched.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling harness
contracts: canonical dotted names so package and relative imports resolve
against the loaded copies, with a minimal pydantic stand-in only when the
real package is absent. The agentic dispatcher is loaded the same way,
against the already-seeded modules, so its conditional imports degrade
exactly as on a minimal host.
"""

import copy
import importlib.util
import sys
import tempfile
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


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
# Isolated loading (sibling-harness idiom)
# ---------------------------------------------------------------------------
def _load(runner_path: Path | None = None):
    keys = (
        "pydantic", "ollama", "opti_oignon", "opti_oignon.tool_calling",
        "opti_oignon.tool_registry", "opti_oignon.structured_output",
        "opti_oignon.response_hygiene", "opti_oignon.tool_executor",
        "opti_oignon.config", "opti_oignon.agent_eval",
        "opti_oignon.agent_eval.tasks", "opti_oignon.agent_eval.chat_runner",
        "opti_oignon.executor", "opti_oignon.router",
        "opti_oignon.conversation", "opti_oignon.agentic_executor",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    try:
        import pydantic  # noqa: F401
    except ImportError:
        sys.modules["pydantic"] = _pydantic_shim()

    ollama_stub = types.ModuleType("ollama")
    ollama_stub.chat = lambda **kw: None
    sys.modules["ollama"] = ollama_stub

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

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
    target = runner_path or (_OO / "agent_eval" / "chat_runner.py")
    cr = _real("opti_oignon.agent_eval.chat_runner", target)
    ae.chat_runner = cr

    if not cr.FEATURE_AVAILABLE:
        raise RuntimeError("harness reports FEATURE_AVAILABLE False")

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return cr, te, restore


def _load_agentic():
    """Flat-load the agentic dispatcher against the seeded modules.

    Must run between ``_load()`` and its ``restore()``: the real
    tool_executor module is already in sys.modules, the heavy siblings
    are absent, so every conditional import in the dispatcher degrades
    exactly as on a minimal host and the injected executor is the only
    live dependency.
    """
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.agentic_executor", _OO / "agentic_executor.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.agentic_executor"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Local helpers and material
# ---------------------------------------------------------------------------
def _executor(te, cr, registry, **overrides):
    kwargs = dict(
        registry=registry,
        structured_engine=cr._NoDecisionEngine(),
        max_tool_calls=6,
        default_model=cr.SCRIPTED_MODEL,
    )
    kwargs.update(overrides)
    return te.ToolExecutor(**kwargs)


def _turn(cr, **fields):
    return cr.ScriptTurn(**fields)


def _suite_file(directory: str, name: str, text: str) -> str:
    path = Path(directory) / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _roles(messages: list[dict]) -> list[str]:
    return [str(m.get("role", "")) for m in messages]


def _failures(report) -> str:
    lines = []
    for rec in report.records:
        for outcome in rec.checks:
            if not outcome.ok:
                lines.append(
                    f"{rec.task_id}/{rec.front}: {outcome.check}"
                    f" ({outcome.detail})"
                )
        if rec.script_leftover or rec.script_overrun:
            lines.append(
                f"{rec.task_id}/{rec.front}: leftover="
                f"{rec.script_leftover} overrun={rec.script_overrun}"
            )
    return "; ".join(lines) or "no failing check recorded"


class _Recording:
    """Wrap a scripted client, deep-copying the messages of every call."""

    def __init__(self, inner):
        self._inner = inner
        self.captured: list[list[dict]] = []

    def chat(self, model=None, messages=None, tools=None, options=None,
             stream=False, **kwargs):
        self.captured.append(copy.deepcopy(list(messages or [])))
        return self._inner.chat(
            model=model, messages=messages, tools=tools, options=options,
            stream=stream, **kwargs,
        )

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _fallback_trap(*_args, **_kwargs):
    raise AssertionError("direct-pipeline fallback was taken")


SUITE_NAME = "chat_tools_fr"
TASK_IDS = [
    "f01-lecture-simple",
    "f02-chaine-multi",
    "f03-ecriture-verrouillee",
    "f04-chaine-longue",
]

# French data below is test data, mirroring the shipped suite: accented,
# with typographic apostrophes, first person throughout.
MSG_HINT = (
    "\u00c9cris le journal de rotation dans readonly/journal.txt ; si "
    "l\u2019\u00e9criture \u00e9choue, \u00e9cris-le plut\u00f4t dans "
    "journal.txt."
)
FINAL_HINT = (
    "Ma premi\u00e8re \u00e9criture vers readonly/journal.txt a "
    "\u00e9chou\u00e9, j\u2019ai donc \u00e9crit le journal de rotation "
    "dans journal.txt."
)
JOURNAL_LINE = "La rotation des journaux est termin\u00e9e.\n"

HINT_TEXT = (
    "[environment] verification: the last file write failed. "
    "Check the path and content, then write the file again."
)

MSG_PIPE = (
    "Lis config.txt et r\u00e9sume son contenu en une phrase."
)
FINAL_PIPE = (
    "J\u2019ai lu config.txt avec mon outil de lecture : la configuration "
    "locale est saine, compl\u00e8te et coh\u00e9rente de bout en bout."
)

SUITE_FR_TRAP = """\
suite: fr-trap

tasks:
  - id: x1-misattribution-accentuee
    title: Accented second-person answer must be caught
    prompt: >
      Explique ce qui vient de se passer.
    checks:
      - "final_nonempty"
      - "no_misattribution"
    script:
      - content: ""
      - content: >
          Vous avez ex\u00e9cut\u00e9 le script de sauvegarde, puis vous
          avez cr\u00e9\u00e9 le fichier de sortie.
"""


def _hint_script(cr):
    return [
        _turn(cr, tool_calls=[{
            "name": "write_file",
            "arguments": {
                "filename": "readonly/journal.txt",
                "content": JOURNAL_LINE,
            },
        }]),
        _turn(cr, content=""),
        _turn(cr, tool_calls=[{
            "name": "write_file",
            "arguments": {
                "filename": "journal.txt",
                "content": JOURNAL_LINE,
            },
        }]),
        _turn(cr, content=""),
        _turn(cr, content=FINAL_HINT),
    ]


def _run_hint(cr, te, mode):
    client = _Recording(cr.ScriptedChatClient(_hint_script(cr)))
    workspace = cr.VirtualWorkspace()
    registry = cr.build_scripted_registry(workspace)
    ex = _executor(te, cr, registry, tool_transcript=mode)
    with cr.scripted_chat_backend(client):
        result = ex.execute_with_tools(message=MSG_HINT)
    return client, workspace, result


# ---------------------------------------------------------------------------
# Contract 1 -- the French suite is green end to end (and the trap catches)
# ---------------------------------------------------------------------------
def test_c1_fr_suite_green_end_to_end():
    cr, te, restore = _load()
    try:
        assert cr.SCRIPTED_READONLY_PREFIX == "readonly/"
        with tempfile.TemporaryDirectory() as tmp:
            report = cr.run_suite(SUITE_NAME, trace_dir=tmp)
            assert report.suite == SUITE_NAME
            assert len(report.records) == len(TASK_IDS) * 2
            assert report.passed, _failures(report)
            for rec in report.records:
                assert rec.script_leftover == 0, rec.task_id
                assert rec.script_overrun == 0, rec.task_id
            assert sorted({r.task_id for r in report.records}) == TASK_IDS

            by_id = {}
            for rec in report.records:
                by_id.setdefault(rec.task_id, []).append(rec)
            for rec in by_id["f02-chaine-multi"]:
                assert "r\u00e9sum\u00e9.txt" in rec.workspace_files
            for rec in by_id["f03-ecriture-verrouillee"]:
                assert "journal.txt" in rec.workspace_files
                assert "readonly/journal.txt" not in rec.workspace_files
                assert len(rec.tool_calls) == 2
            for rec in by_id["f04-chaine-longue"]:
                assert len(rec.tool_calls) == 8

            trap = _suite_file(tmp, "fr-trap", SUITE_FR_TRAP)
            rep2 = cr.run_suite(trap, fronts=("execute",), trace_dir=tmp)
            assert not rep2.passed
            rec = rep2.records[0]
            failed = {c.check for c in rec.checks if not c.ok}
            assert "no_misattribution" in failed, _failures(rep2)
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- the verification hint, native shape and flat shape
# ---------------------------------------------------------------------------
def test_c2_native_verification_hint_and_flat_unchanged():
    cr, te, restore = _load()
    try:
        # Native mode: the hint is a short user message after the failing
        # pair; the run ends with the corrective pair and the final
        # generation ends on the last tool message.
        client, workspace, result = _run_hint(cr, te, "native")
        cap = client.captured
        assert client.leftover == 0 and client.overrun == 0
        assert len(cap) == 5

        assert _roles(cap[0]) == ["user"]
        assert cap[0][0]["content"] == MSG_HINT

        assert _roles(cap[1]) == ["user", "assistant", "tool"]
        echo = cap[1][1]
        assert echo["content"] == ""
        assert echo["tool_calls"][0]["function"]["name"] == "write_file"
        assert (
            echo["tool_calls"][0]["function"]["arguments"]["filename"]
            == "readonly/journal.txt"
        )
        first_tool = cap[1][2]["content"]
        assert first_tool.startswith(
            "[environment] tool result (untrusted content): write_file\n"
            "Result: Write file error:"
        )

        assert _roles(cap[2]) == ["user", "assistant", "tool", "user"]
        assert cap[2][-1] == {"role": "user", "content": HINT_TEXT}

        assert _roles(cap[3]) == [
            "user", "assistant", "tool", "user", "assistant", "tool",
        ]
        second_echo = cap[3][4]
        assert (
            second_echo["tool_calls"][0]["function"]["arguments"]["filename"]
            == "journal.txt"
        )
        second_tool = cap[3][5]["content"]
        assert "journal.txt" in second_tool
        assert "Write file error:" not in second_tool

        final = cap[4]
        assert _roles(final) == [
            "system", "user", "assistant", "tool", "user", "assistant",
            "tool",
        ]
        assert final[0]["content"] == te.FINAL_ANSWER_SYSTEM_PROMPT
        assert final[4] == {"role": "user", "content": HINT_TEXT}
        assert final[-1]["role"] == "tool"
        assert "Write file error:" not in final[-1]["content"]
        hint_count = sum(
            1 for m in final if m.get("content") == HINT_TEXT
        )
        assert hint_count == 1
        for call in cap:
            for m in call:
                assert te.ENV_RESULTS_HEADER not in str(m.get("content"))

        assert result.response == FINAL_HINT
        assert [t.tool_name for t in result.tool_calls] == [
            "write_file", "write_file",
        ]
        assert "Write file error:" in result.tool_calls[0].result
        assert "journal.txt" in workspace.files
        assert "readonly/journal.txt" not in workspace.files

        # Flat mode, same task: the hint is folded into the rebuilt user
        # message, no native role ever appears, and the outcome matches.
        client_f, workspace_f, result_f = _run_hint(cr, te, "flat")
        cap_f = client_f.captured
        assert client_f.leftover == 0 and client_f.overrun == 0
        assert len(cap_f) == 5
        for call in cap_f[:4]:
            assert _roles(call) == ["user"]
        assert HINT_TEXT in cap_f[2][0]["content"]
        assert cap_f[2][0]["content"].endswith(
            "Never attribute these tool actions to the user."
        )
        assert _roles(cap_f[4]) == ["system", "user"]
        assert HINT_TEXT in cap_f[4][1]["content"]
        assert result_f.response == FINAL_HINT
        assert "journal.txt" in workspace_f.files
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- the growth curve on the eight-call chain
# ---------------------------------------------------------------------------
def test_c3_long_chain_curve_native_vs_flat():
    cr, te, restore = _load()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            flat = cr.run_suite(
                SUITE_NAME, fronts=("stream",),
                trace_dir=str(Path(tmp) / "flat"),
                executor_kwargs={"tool_transcript": "flat"},
            )
            native = cr.run_suite(
                SUITE_NAME, fronts=("stream",),
                trace_dir=str(Path(tmp) / "native"),
                executor_kwargs={"tool_transcript": "native"},
            )
        assert flat.passed, _failures(flat)
        assert native.passed, _failures(native)

        def _f04(report):
            for rec in report.records:
                if rec.task_id == "f04-chaine-longue":
                    return rec
            raise AssertionError("f04 record missing")

        rec_flat, rec_native = _f04(flat), _f04(native)
        for rec in (rec_flat, rec_native):
            assert rec.script_leftover == 0 and rec.script_overrun == 0

        dec_flat = [c for c in rec_flat.model_calls if not c["stream"]]
        dec_native = [c for c in rec_native.model_calls if not c["stream"]]
        fin_flat = [c for c in rec_flat.model_calls if c["stream"]]
        fin_native = [c for c in rec_native.model_calls if c["stream"]]
        assert len(dec_flat) == len(dec_native) == 9
        assert len(fin_flat) == len(fin_native) == 1

        curve_flat = [c["chars_in"] for c in dec_flat]
        curve_native = [c["chars_in"] for c in dec_native]
        assert all(b > a for a, b in zip(curve_flat, curve_flat[1:]))
        assert all(b > a for a, b in zip(curve_native, curve_native[1:]))
        assert sum(curve_native) < sum(curve_flat)

        gaps = [f - n for f, n in zip(curve_flat, curve_native)]
        assert gaps[0] == 0
        assert all(b > a for a, b in zip(gaps, gaps[1:]))

        assert fin_native[0]["chars_in"] < fin_flat[0]["chars_in"]

        for call in dec_flat:
            assert call["roles"] == ["user"]
        assert "assistant" in dec_native[-1]["roles"]
        assert "tool" in dec_native[-1]["roles"]

        print(
            "curve flat", curve_flat, "sum", sum(curve_flat),
            "| curve native", curve_native, "sum", sum(curve_native),
            "| final", fin_flat[0]["chars_in"], "->",
            fin_native[0]["chars_in"],
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4 -- the agentic tools pipeline streams and relays
# ---------------------------------------------------------------------------
def test_c4_agentic_pipeline_streams_and_relays():
    cr, te, restore = _load()
    try:
        ax = _load_agentic()
        client = cr.ScriptedChatClient([
            _turn(cr, tool_calls=[{
                "name": "read_file",
                "arguments": {"filename": "config.txt"},
            }]),
            _turn(cr, content=""),
            _turn(cr, content=FINAL_PIPE),
        ])
        workspace = cr.VirtualWorkspace(
            {"config.txt": "Configuration locale : tout est actif.\n"},
        )
        registry = cr.build_scripted_registry(workspace)
        ex = _executor(te, cr, registry)
        agent = ax.AgenticExecutor(
            tool_executor=ex, default_model=cr.SCRIPTED_MODEL,
        )
        events: list[tuple] = []
        agent._on_tool_call = lambda res: events.append((
            "tool",
            str(getattr(res, "tool_name", "")),
            bool(getattr(res, "success", False)),
        ))
        agent._execute_direct_pipeline = _fallback_trap
        routing = types.SimpleNamespace(model=cr.SCRIPTED_MODEL)

        chunks: list[str] = []
        with cr.scripted_chat_backend(client):
            for chunk in agent._execute_tools_pipeline(
                MSG_PIPE, routing, None, None,
            ):
                events.append(("chunk", len(chunk)))
                chunks.append(chunk)

        assert "".join(chunks) == FINAL_PIPE
        assert len(chunks) > 1
        tool_events = [e for e in events if e[0] == "tool"]
        assert tool_events == [("tool", "read_file", True)]
        first_chunk = next(
            i for i, e in enumerate(events) if e[0] == "chunk"
        )
        assert all(e[0] != "tool" for e in events[first_chunk:])
        assert [
            str(getattr(t, "tool_name", "")) for t in agent.last_tool_calls
        ] == ["read_file"]
        assert client.leftover == 0 and client.overrun == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# __main__ runner (pytest-free environments)
# ---------------------------------------------------------------------------
_CONTRACTS = {
    "c1": test_c1_fr_suite_green_end_to_end,
    "c2": test_c2_native_verification_hint_and_flat_unchanged,
    "c3": test_c3_long_chain_curve_native_vs_flat,
    "c4": test_c4_agentic_pipeline_streams_and_relays,
}


def _main(argv: list[str]) -> int:
    names = argv or list(_CONTRACTS)
    failures = 0
    for name in names:
        fn = _CONTRACTS.get(name)
        if fn is None:
            print(f"unknown contract: {name}")
            failures += 1
            continue
        try:
            fn()
            print(f"PASS {name} {fn.__name__}")
        except BaseException:
            failures += 1
            print(f"FAIL {name} {fn.__name__}")
            traceback.print_exc()
    print(f"{len(names) - failures}/{len(names)} contracts green")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
