#!/usr/bin/env python3
"""Contracts for a user turn saved with no claim: it is no one's, never typed.

The chat route makes a claim for every turn -- its text, who wrote it, where
each part lies -- and hands it to every path that saves the user's words.
Only typed words may later endorse a write or make a decision probe, so a
path that saves a user turn with no claim cannot vouch for it: such a turn is
legacy, the least trusted origin. These contracts supersede the ones that
pinned the opposite (a turn saved with no claim was typed), and keep
everything else those held:

  * Contract UO1 to UO4 -- each agentic pipeline that saves its own turn
    (cascading, speculative, think and tools, tools) saves the question as
    legacy when the turn carries no claim, and as the claim says when it
    carries one; the answer is flagged tool. Supersede OT17 to OT20.
  * Contract UO5 -- the coding agent saves its turn the same way. Supersedes
    OT21.
  * Contract UO6 -- the agentic executor saves the claimed text with the
    claim's parts, a text composed from it as legacy, and a turn run with no
    claim as legacy. Supersedes UT7.

The agentic executor and the coding agent are loaded through the shared
isolation window over a store that keeps what it is handed. Local-only. Runs
under pytest or the __main__ runner.
"""

import sys
import threading
import traceback
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_AGENTIC = "opti_oignon.agentic_executor"
_CODING = "opti_oignon.chat_coding_agent"
_CONV = "conv-agentic"
_HEAD = "\n\n---\nDocument provided:"
_QUESTION = "Where does Alice meet Bob on 2024-03-15?"
_DOCS = (("venue.txt", "Contoso opens the venue to 40 guests."),
         ("notes.md", "Bob arrives at noon.\nAlice brings the keys."))


class _Claim:
    """A turn's claim as the chat route makes it: it vouches for its own text and nothing else."""

    def __init__(self, content, origin="typed", segments=()):
        self.content = content
        self.origin = origin
        self.segments = tuple(tuple(s) for s in segments)

    def parts_for(self, text):
        if text == self.content:
            return self.origin, [list(s) for s in self.segments]
        return "legacy", []


def _composed_claim(question=_QUESTION, documents=_DOCS):
    """The claim of a question with documents attached, composed as the executor composes it."""
    content, segments = question, [[0, len(question), "typed"]]
    for name, text in documents:
        content += f"{_HEAD} {name}\n"
        segments.append([len(content), len(content) + len(text), "document"])
        content += text
    return _Claim(content, "typed", segments)


class _AgentStore:
    """Mirrors the store's keyword surface, origin included."""

    def __init__(self):
        self.saved = []

    def add_message(self, conv_id=None, role=None, content=None, model=None, metadata=None, *, origin="legacy",
                    segments=()):
        self.saved.append({"conv_id": conv_id, "role": role, "content": content, "model": model,
                           "origin": origin, "segments": [list(s) for s in segments]})
        return {"ok": True}


class _PipelineResult:
    def __init__(self, text):
        self.final_response = text
        self.model = "fake-model"
        self.tier_index = 0
        self.tier_name = "fast"
        self.score = 0.9
        self.total_latency_ms = 1.0
        self.draft_accepted = True
        self.iterations = 1
        self.convergence_score = 1.0


class _Cascade:
    enabled = True

    def __init__(self, text="cascade answer"):
        self.text = text

    def cascade(self, query, task_type=None):
        return _PipelineResult(self.text)


class _Speculative:
    enabled = True

    def generate(self, query, task_type=None):
        return _PipelineResult("speculative answer")


class _Reasoner:
    """Streams like Executor.execute and, like a persist=False call, saves nothing."""

    def __init__(self):
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        yield "reasoning part. "
        yield "more reasoning."


class _ToolResult:
    def __init__(self):
        self.tool_calls = ["write_file"]
        self.response = "TOOL_OUTPUT_BLOCK"


class _Tools:
    def should_use_tools(self, message, model):
        return True

    def execute_with_tools(self, **kwargs):
        return _ToolResult()


def _agentic(store):
    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = store
    loaded, restore = isolate(targets={_AGENTIC: source("agentic_executor.py")},
                              seeded={"opti_oignon.conversation": conv})
    return loaded[_AGENTIC], restore


def _routing():
    return SimpleNamespace(model="m", task_type=None)


def _quiet(agent):
    agent._get_conversation_context = lambda cid: []
    agent.get_tool_history = lambda cid: []
    agent._record_tool_calls = lambda cid, tc: None
    agent._emit_tool_call = lambda tc: None
    return agent


def _origins(store):
    return [(m["role"], m["origin"], m["segments"]) for m in store.saved]


def _both(run_pipeline):
    """Run a pipeline once with no claim and once with a claim for its question; the two stores."""
    stores = []
    for claim in (None, _Claim("q")):
        store = _AgentStore()
        mod, restore = _agentic(store)
        try:
            run_pipeline(mod, None if claim is None else mod._Turn(_run(claim)))
        finally:
            restore()
        stores.append(store)
    return stores


_UNCLAIMED = [("user", "legacy", []), ("assistant", "assistant+tool", [])]
_CLAIMED = [("user", "typed", []), ("assistant", "assistant+tool", [])]


# ---------------------------------------------------------------------------
# Contracts UO1 to UO4 -- the agentic pipelines (supersede OT17 to OT20)
# ---------------------------------------------------------------------------
def test_uo1_the_cascading_pipeline_saves_an_unclaimed_question_as_legacy_and_a_claimed_one_typed():
    def run(mod, turn):
        mod.CASCADING_INFERENCE_AVAILABLE = True
        agent = mod.AgenticExecutor(cascading_inference=_Cascade())
        list(agent._execute_cascading_pipeline("q", _routing(), _CONV, None, turn=turn))

    for store, expected in zip(_both(run), (_UNCLAIMED, _CLAIMED)):
        assert len(store.saved) == 2, "the cascading pipeline persists user and assistant"
        assert store.saved[1]["content"] == "cascade answer" and store.saved[1]["model"] == "fake-model"
        assert _origins(store) == expected, _origins(store)


def test_uo2_the_speculative_pipeline_saves_an_unclaimed_question_as_legacy_and_a_claimed_one_typed():
    def run(mod, turn):
        mod.SPECULATIVE_GENERATION_AVAILABLE = True
        agent = mod.AgenticExecutor(speculative_generator=_Speculative())
        list(agent._execute_speculative_pipeline("q", _routing(), _CONV, None, turn=turn))

    for store, expected in zip(_both(run), (_UNCLAIMED, _CLAIMED)):
        assert len(store.saved) == 2, "the speculative pipeline persists user and assistant"
        assert store.saved[1]["content"] == "speculative answer" and store.saved[1]["model"] == "fake-model"
        assert _origins(store) == expected, _origins(store)


def test_uo3_the_think_and_tools_pipeline_saves_an_unclaimed_question_as_legacy_and_a_claimed_one_typed():
    reasoners = []

    def run(mod, turn):
        mod.TOOL_EXECUTOR_AVAILABLE = True
        reasoner = _Reasoner()
        reasoners.append(reasoner)
        agent = _quiet(mod.AgenticExecutor(executor=reasoner, tool_executor=_Tools()))
        list(agent._execute_think_tools_pipeline("q", _routing(), _CONV, None, turn=turn))

    for store, expected, reasoner in zip(_both(run), (_UNCLAIMED, _CLAIMED), reasoners):
        assert reasoner.calls and reasoner.calls[0].get("persist") is False, "the executor's own save is suppressed"
        assert len(store.saved) == 2, "user and assistant persisted exactly once"
        answer = store.saved[1]["content"]
        assert "reasoning" in answer and "TOOL_OUTPUT_BLOCK" in answer, "reasoning and tool output both persist"
        assert _origins(store) == expected, _origins(store)


def test_uo4_the_tools_pipeline_saves_an_unclaimed_question_as_legacy_and_a_claimed_one_typed():
    def run(mod, turn):
        mod.TOOL_EXECUTOR_AVAILABLE = True
        agent = _quiet(mod.AgenticExecutor(tool_executor=_Tools()))
        list(agent._execute_tools_pipeline("q", _routing(), _CONV, None, turn=turn))

    for store, expected in zip(_both(run), (_UNCLAIMED, _CLAIMED)):
        assert len(store.saved) == 2, "the tools pipeline persists user and assistant"
        assert store.saved[1]["content"] == "TOOL_OUTPUT_BLOCK"
        assert _origins(store) == expected, _origins(store)


# ---------------------------------------------------------------------------
# Contract UO5 -- the coding agent (supersedes OT21)
# ---------------------------------------------------------------------------
class _CodingStore:
    def __init__(self):
        self.saved = []

    def add_message(self, conv_id, role, content, model=None, metadata=None, *, origin="legacy", segments=()):
        self.saved.append({"conv_id": conv_id, "role": role, "content": content, "origin": origin,
                           "segments": list(segments)})


def test_uo5_the_coding_agent_saves_an_unclaimed_turn_as_legacy_and_a_claimed_one_by_its_claim():
    results = []
    for claim in (None, _Claim("Fix the failing test.")):
        store = _CodingStore()
        conv = types.ModuleType("opti_oignon.conversation")
        conv.conversation_manager = store
        loaded, restore = isolate(targets={_CODING: source("chat_coding_agent.py")},
                                  seeded={"opti_oignon.conversation": conv})
        try:
            mod = loaded[_CODING]
            assert mod.CONVERSATION_AVAILABLE is True, "control: the agent reaches the store"
            session = object.__new__(mod.ChatCodingSession)
            session._conversation_id = "conv-code"
            if claim is not None:
                session._turn_user_turn = claim
            session._save_turn_to_conversation("Fix the failing test.", "Fixed: the import was wrong.", "coder:7b")
        finally:
            restore()
        results.append(store)
    unclaimed, claimed = results
    for store in results:
        assert len(store.saved) == 2, "control: the coding turn was saved"
    assert [(m["role"], m["origin"], m["segments"]) for m in unclaimed.saved] == [
        ("user", "legacy", []), ("assistant", "assistant+tool", [])], unclaimed.saved
    assert [(m["role"], m["origin"], m["segments"]) for m in claimed.saved] == [
        ("user", "typed", []), ("assistant", "assistant+tool", [])], claimed.saved


# ---------------------------------------------------------------------------
# Contract UO6 -- the agentic executor's runs (supersedes UT7)
# ---------------------------------------------------------------------------
def _run(claim):
    """A run as the chat route hands it on: its stop, its results, the turn's claim."""
    return SimpleNamespace(stop=threading.Event(), results={}, steps=None, user_turn=claim)


def test_uo6_the_agentic_executor_saves_by_the_runs_claim_and_a_run_with_none_as_legacy():
    claim = _composed_claim()
    step = "Original question: " + claim.content
    store = _AgentStore()
    mod, restore = _agentic(store)
    try:
        mod.CASCADING_INFERENCE_AVAILABLE = True
        agent = mod.AgenticExecutor(executor=SimpleNamespace(), cascading_inference=_Cascade("At the venue."),
                                    default_model="m")
        for message, run in ((claim.content, _run(claim)), (step, _run(claim)), (_QUESTION, None)):
            list(agent.execute(message=message, routing=SimpleNamespace(model="m", task_type=None),
                               conversation_id="conv-a", cascading=True, **({"run": run} if run else {})))
    finally:
        restore()
    users = [(m["content"], m["origin"], m["segments"]) for m in store.saved if m["role"] == "user"]
    assert users == [
        (claim.content, "typed", [list(s) for s in claim.segments]),
        (step, "legacy", []),
        (_QUESTION, "legacy", []),
    ], users


def _run_all():
    cases = (
        ("UO1 cascading", test_uo1_the_cascading_pipeline_saves_an_unclaimed_question_as_legacy_and_a_claimed_one_typed),
        ("UO2 speculative",
         test_uo2_the_speculative_pipeline_saves_an_unclaimed_question_as_legacy_and_a_claimed_one_typed),
        ("UO3 think and tools",
         test_uo3_the_think_and_tools_pipeline_saves_an_unclaimed_question_as_legacy_and_a_claimed_one_typed),
        ("UO4 tools", test_uo4_the_tools_pipeline_saves_an_unclaimed_question_as_legacy_and_a_claimed_one_typed),
        ("UO5 coding agent",
         test_uo5_the_coding_agent_saves_an_unclaimed_turn_as_legacy_and_a_claimed_one_by_its_claim),
        ("UO6 agentic runs", test_uo6_the_agentic_executor_saves_by_the_runs_claim_and_a_run_with_none_as_legacy),
    )
    failed = 0
    for name, case in cases:
        try:
            case()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
