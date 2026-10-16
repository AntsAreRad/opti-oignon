#!/usr/bin/env python3
"""Contracts for an answer an engine cut after its first words: never told as a whole one.

A head that loses its stream mid-answer raises once it streamed words. What
stands between it and the user must not turn that raise back into a clean
end: the tools pipeline -- the default chat path -- must not complete the
cut answer with a fresh one and save the two glued together, and the core
daemon must not close the stream as if the answer had ended.

  * XC1 -- the tools pipeline: the tool executor's final answer, cut once
    it produced words, raises its cut answer error rather than asking the
    engine again; the pipeline keeps the record of the tools that ran, ends
    the turn on its error after what streamed, saves nothing and asks no
    other pipeline -- also when the marker filter still holds every word the
    engine produced, through the executor's own streaming. A final answer
    that fails before any word still falls back -- to a single-shot
    generation in the executor, to the direct pipeline in the agentic one.
  * XC2 -- the core daemon and its client: a backend that breaks once its
    stream began ends the daemon's stream with a line naming the failure,
    before the stream's end; the client raises on that line, after what
    streamed, and on a stream that ends before its done chunk.

Local-only (the public distribution ships no tests). The tool executor,
the agentic executor, the daemon and its client are loaded through the
shared isolation window; the engines are stand-ins, and the daemon runs in
a thread on an ephemeral loopback port.
"""

import json
import sys
import threading
import types
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import StubRegistry  # noqa: E402

BUDGET_S = {
    "test_xc1_the_tools_pipeline_ends_a_cut_answer_on_its_error_and_saves_nothing": 3.0,
    "test_xc2_the_core_daemon_and_its_client_never_tell_a_cut_answer_as_a_whole_one": 5.0,
}

_MESSAGES = [{"role": "user", "content": "write x"}]
_BROKE = "the engine went away"
# The loopback needs no proxy and this suite's own opener, not the global urlopen another suite may have replaced.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ---------------------------------------------------------------------------
# XC1 -- the tools pipeline
# ---------------------------------------------------------------------------
class _Engine:
    """A backend whose stream gives ``first`` and then breaks, and whose single shot answers whole."""

    def __init__(self, first):
        self.first, self.generated = first, []

    def stream(self, **kwargs):
        if self.first:
            yield SimpleNamespace(content=self.first, thinking="", done=False)
        raise RuntimeError(_BROKE)

    def generate(self, **kwargs):
        self.generated.append(kwargs)
        return SimpleNamespace(content="a single shot", tool_calls=[])


def _tool_executor(engine, *, agentic=False):
    """The tool executor from its file, its registry's backend ``engine``, and the agentic executor beside it when
    asked; ``(loaded, restore)``."""
    registry = SimpleNamespace(active=engine, resolve_backend=lambda model: engine)
    backends = types.ModuleType("opti_oignon.inference_backend")
    backends.get_backend_registry = lambda: registry
    config = types.ModuleType("opti_oignon.config")
    config.config = SimpleNamespace(get_model=lambda *a, **k: "stand-in", get_temperature=lambda *a, **k: 0.3,
                                    get=lambda *a, **k: None)
    targets = {
        "opti_oignon.tool_calling": source("tool_calling.py"),
        "opti_oignon.tool_registry": source("tool_registry.py"),
        "opti_oignon.response_hygiene": source("response_hygiene.py"),
        "opti_oignon.structured_output": source("structured_output.py"),
        "opti_oignon.tool_executor": source("tool_executor.py"),
    }
    if agentic:
        targets["opti_oignon.agentic_executor"] = source("agentic_executor.py")
    loaded, restore = isolate(targets=targets,
                              seeded={"opti_oignon.inference_backend": backends, "opti_oignon.config": config},
                              packages=("opti_oignon",))
    return loaded, restore


class _Stage:
    """A tools stage: its tool runs, then its answer streams ``first`` and breaks."""

    def __init__(self, first):
        self.first = first
        self.call = SimpleNamespace(tool_name="write_file", arguments={"path": "x.txt"}, result="written",
                                    success=True, execution_time=0, reasoning="")

    def stream_with_tools(self, **kwargs):
        kwargs["on_tool_call"](self.call)
        if self.first:
            yield self.first
        raise RuntimeError(_BROKE)


def _drained(stream):
    """Every piece ``stream`` yields before it raises, and what it raised."""
    pieces = []
    with pytest.raises(RuntimeError) as raised:
        for piece in stream:
            pieces.append(piece)
    return pieces, str(raised.value)


def test_xc1_the_tools_pipeline_ends_a_cut_answer_on_its_error_and_saves_nothing():
    # The tool executor: a cut answer raises, the engine is not asked again; one with no word falls back.
    for first, falls_back in (("the answer begins", False), ("", True)):
        engine = _Engine(first)
        loaded, restore = _tool_executor(engine)
        try:
            module = loaded["opti_oignon.tool_executor"]
            executor = module.ToolExecutor(default_model="stand-in")
            if falls_back:
                assert list(executor._stream_final_response(_MESSAGES, "stand-in")) == ["a single shot"]
                assert len(engine.generated) == 1, "a stream that gave no word falls back to a single shot"
            else:
                with pytest.raises(module.CutAnswerError) as cut:
                    pieces = []
                    for piece in executor._stream_final_response(_MESSAGES, "stand-in"):
                        pieces.append(piece)
                assert pieces == [first] and str(cut.value) == _BROKE, (pieces, str(cut.value))
                assert isinstance(cut.value, RuntimeError), "a cut answer is a failed request to every handler"
                assert engine.generated == [], "a cut answer is never completed with a single shot"
        finally:
            restore()

    # The agentic executor's tools pipeline: the turn ends on its error after what streamed.
    loaded, restore = isolate(targets={"opti_oignon.agentic_executor": source("agentic_executor.py")}, seeded={})
    try:
        agentic = loaded["opti_oignon.agentic_executor"]
        agentic.TOOL_EXECUTOR_AVAILABLE = True
        for first in ("the answer begins", ""):
            stage = _Stage(first)
            agent = agentic.AgenticExecutor(executor=SimpleNamespace(), tool_executor=stage, default_model="m")
            saved, direct = [], []
            agent._save_to_conversation = lambda *args, **kwargs: saved.append(args)
            agent._get_conversation_context = lambda conversation_id: []

            def direct_pipeline(*args, direct=direct, **kwargs):
                direct.append(args)
                yield "a direct answer"

            agent._execute_direct_pipeline = direct_pipeline
            turn = agent._legacy_turn()
            pieces = list(agent._execute_tools_pipeline("write x", SimpleNamespace(model="m"), "conv-1", None,
                                                        turn=turn))
            if first:
                assert pieces == [first, f"\n\n[ERR] Error: {_BROKE}"], pieces
                assert saved == [] and direct == [], "nothing saved, and no other pipeline asked"
                assert turn.results["tool_calls"] == [stage.call], "the record of the tools that ran is kept"
                assert agent.get_tool_history("conv-1") == [stage.call]
            else:
                assert pieces == ["a direct answer"] and len(direct) == 1, (
                    "control: an answer that failed before any word still falls back", pieces)
    finally:
        restore()

    # Through the executor's own streaming: the engine's first word is one the marker filter holds, a line that
    # may still turn out to be scaffold, and the stream breaks before the filter lets it go.
    engine = _Engine("The")
    loaded, restore = _tool_executor(engine, agentic=True)
    try:
        module, agentic = loaded["opti_oignon.tool_executor"], loaded["opti_oignon.agentic_executor"]
        hygiene = loaded["opti_oignon.response_hygiene"]
        assert hygiene.StreamMarkerFilter().feed("The") == "", "control: the filter holds the line's start"
        executor = module.ToolExecutor(default_model="stand-in")
        call = module.ToolCallResult(tool_name="write_file", arguments={"path": "x.txt"}, result="written")

        def tool_loop(message, model, conversation_messages, tool_history, approval_fn, on_tool_call, **kwargs):
            on_tool_call(call)
            return [call], ["write_file: written"], None, [], None, 0

        executor._run_tool_loop = tool_loop
        agent = agentic.AgenticExecutor(executor=SimpleNamespace(), tool_executor=executor, default_model="m")
        saved, direct = [], []
        agent._save_to_conversation = lambda *args, **kwargs: saved.append(args)
        agent._get_conversation_context = lambda conversation_id: []

        def direct_pipeline(*args, **kwargs):
            direct.append(args)
            yield "a direct answer"

        agent._execute_direct_pipeline = direct_pipeline
        turn = agent._legacy_turn()
        pieces = list(agent._execute_tools_pipeline("write x", SimpleNamespace(model="stand-in"), "conv-1", None,
                                                    turn=turn))
        assert engine.generated == [], "the engine was not asked again"
        assert pieces == [f"\n\n[ERR] Error: {_BROKE}"], pieces
        assert saved == [] and direct == [], "nothing saved, and no other pipeline asked, though nothing streamed"
        assert turn.results["tool_calls"] == [call] and agent.get_tool_history("conv-1") == [call]
    finally:
        restore()


# ---------------------------------------------------------------------------
# XC2 -- the core daemon and its client
# ---------------------------------------------------------------------------
class _Chunk:
    def __init__(self, content="", thinking="", done=False, model=""):
        self.content, self.thinking, self.done, self.model = content, thinking, done, model


class _Base:
    """The registry's abstract backend, as a stand-in: the names the client subclasses."""


class _Breaking:
    """A daemon's backend whose stream gives its first words, then breaks or ends with no done chunk."""

    name = "breaking"

    def __init__(self):
        self.ending = "raise"

    def health_check(self):
        return True

    def stream(self, model, messages, options=None, keep_alive="30m", think=None, images=None):
        yield _Chunk(content="the answer begins", model=model)
        if self.ending == "raise":
            raise RuntimeError(_BROKE)


def _daemon(backend):
    registry = StubRegistry()
    registry.register(backend)
    registry.activate(backend.name)
    module = types.ModuleType("opti_oignon.inference_backend")
    module.get_backend_registry = lambda: registry
    module._governor_admission = lambda model, options: None
    module.InferenceBackend = _Base
    module.ChatResponse = object
    module.StreamChunk = _Chunk
    module.BackendModelInfo = object
    loaded, restore = isolate(
        targets={"opti_oignon.core_daemon": source("core_daemon.py"),
                 "opti_oignon.core_client": source("core_client.py")},
        seeded={"opti_oignon.inference_backend": module},
        packages=("opti_oignon",),
    )
    return loaded["opti_oignon.core_daemon"], loaded["opti_oignon.core_client"], restore


def test_xc2_the_core_daemon_and_its_client_never_tell_a_cut_answer_as_a_whole_one():
    backend = _Breaking()
    daemon, client, restore = _daemon(backend)
    try:
        config = daemon.CoreConfig(enabled=True, host="127.0.0.1", port=0, token="", timeout_s=5.0)
        server = daemon.make_server(config, daemon.CoreService(token=""))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            # The wire: the words that streamed, then a line naming the failure, and the stream's end.
            request = urllib.request.Request(base + "/inference/stream",
                                             data=json.dumps({"model": "m", "messages": _MESSAGES}).encode(),
                                             headers={"Content-Type": "application/json"})
            with _OPENER.open(request, timeout=5) as response:
                lines = [json.loads(raw) for raw in response.read().decode("utf-8").splitlines() if raw.strip()]
            assert [line.get("content") for line in lines[:-1]] == ["the answer begins"], lines
            assert lines[-1] == {"error": f"backend failed mid-stream: {_BROKE}"}, lines[-1]
            # The client: what streamed, then the failure raised by name.
            remote = client.RemoteCoreBackend(base, timeout_s=5.0)
            pieces, why = _drained(remote.stream("m", _MESSAGES))
            assert [piece.content for piece in pieces] == ["the answer begins"], why
            assert why == f"core daemon: backend failed mid-stream: {_BROKE}", why
            # A stream that ends before its done chunk is no whole answer either.
            backend.ending = "silent"
            pieces, why = _drained(remote.stream("m", _MESSAGES))
            assert [piece.content for piece in pieces] == ["the answer begins"] and not pieces[-1].done
            assert why == "core daemon: the stream ended before its done chunk", why
        finally:
            server.shutdown()
            server.server_close()
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
