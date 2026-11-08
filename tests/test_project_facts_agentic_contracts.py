#!/usr/bin/env python3
"""Contracts for the answers the agentic pipelines save themselves: they keep their project's context too.

The executor saves an answer of a project turn with the facts of its project's context (``project_facts``, block
268). The think-and-tools and self-correction pipelines call the executor without its save (``persist=False``) and
save the answer themselves (``AgenticExecutor._save_to_conversation``), with its model's name only: the ``done`` frame
said the facts, a reload did not.

  * PAG1 -- THE PIPELINE'S SAVE KEEPS THEM: the real ``_save_to_conversation``, for a turn whose run carries the facts
    the executor wrote, saves the answer with them beside its model's name; for a turn without them, as before (its
    model's name alone); the user's row carries none.
  * PAG2 -- THINK AND TOOLS KEEPS THEM, END TO END: the pipeline, its executor call writing the facts on the turn's
    run, saves its one answer -- the thinking and the tools' output -- with them.
  * PAG3 -- THE TEXTS: the CHANGELOG says it, and no longer says the reload misses them.

Local-only (the public distribution ships no tests). The agentic executor runs in the shared isolation window through
the unclaimed turn suite's own world (tests/test_unclaimed_turn_origin_contracts.py), over a stand-in store that takes
the real store's keywords and keeps what each row is saved with; no contract reaches an engine or a store on disk.
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_unclaimed_turn_origin_contracts as uo  # noqa: E402
from _isolation import REPO  # noqa: E402

BUDGET_S = {
    "test_pag1_the_pipelines_save_keeps_them": 4.0,
    "test_pag2_think_and_tools_keeps_them_end_to_end": 6.0,
    "test_pag3_the_texts": 2.0,
}

CONV = "conv-pag"
FACTS = {"source": "mycelium+rag", "passages": 3, "said": {"form": "left", "files": 2}}


class _Store:
    """The store's keyword surface, keeping each row's role, text, model and metadata."""

    def __init__(self):
        self.saved = []

    def add_message(self, conv_id=None, role=None, content=None, model=None, metadata=None, **kwargs):
        self.saved.append({"role": role, "content": content, "model": model, "metadata": metadata})
        return {"ok": True}

    def get_messages(self, conv_id):
        return []

    def get_labelled_context_messages(self, conv_id, **kwargs):
        return []


def _turn(mod, facts):
    return mod._Turn(SimpleNamespace(stop=threading.Event(), results={} if facts is None else
                                     {"project_context": dict(facts)}))


def test_pag1_the_pipelines_save_keeps_them():
    saved = {}
    for name, facts in (("with", FACTS), ("without", None)):
        store = _Store()
        mod, restore = uo._agentic(store)
        try:
            agen = uo._quiet(mod.AgenticExecutor(executor=uo._Reasoner()))
            agen._save_to_conversation(CONV, "do a thing", "An answer.", "m", turn=_turn(mod, facts))
            saved[name] = [(m["role"], m["metadata"]) for m in store.saved]
        finally:
            restore()
    assert saved["with"][-1] == ("assistant", {"model": "m", "project_context": FACTS}), saved["with"]
    assert saved["without"][-1] == ("assistant", {"model": "m"}), "a turn without them is saved as before"
    assert [role for role, _m in saved["with"]] == ["user", "assistant"], saved["with"]
    assert not (saved["with"][0][1] or {}).get("project_context"), "the user's row carries none"


class _FactsExecutor(uo._Reasoner):
    """The executor's call, writing the facts of its project's context on the run, as the real one does."""

    def execute(self, **kwargs):
        run = kwargs.get("run")
        if run is not None:
            run.results["project_context"] = dict(FACTS)
        yield from super().execute(**kwargs)


def test_pag2_think_and_tools_keeps_them_end_to_end():
    store = _Store()
    mod, restore = uo._agentic(store)
    try:
        mod.TOOL_EXECUTOR_AVAILABLE = True
        executor = _FactsExecutor()
        agen = uo._quiet(mod.AgenticExecutor(executor=executor, tool_executor=uo._Tools()))
        turn = _turn(mod, None)
        list(agen._execute_think_tools_pipeline("do a thing", uo._routing(), CONV, None, turn=turn))
        msgs = store.saved
    finally:
        restore()
    assert executor.calls and executor.calls[0].get("run") is turn, "control: the executor's call runs on the turn"
    assert [m["role"] for m in msgs] == ["user", "assistant"], msgs
    assert "TOOL_OUTPUT_BLOCK" in msgs[1]["content"], "control: the one answer holds the tools' output"
    assert msgs[1]["metadata"] == {"model": msgs[1]["model"], "project_context": FACTS}, msgs[1]["metadata"]


def test_pag3_the_texts():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    flat = " ".join(changelog.split())
    entry = " ".join(next((e for e in changelog.split("\n- ") if "PAG1-PAG3" in e), "").split())
    assert "think-and-tools" in entry and "self-correction" in entry and "reload" in entry, entry[:300]
    assert "the reload does not" not in flat, "the CHANGELOG no longer says the reload misses them"
