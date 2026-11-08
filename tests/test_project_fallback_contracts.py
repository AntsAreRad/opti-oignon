#!/usr/bin/env python3
"""Contracts for a project turn that never loses its passages without saying so.

With the context optimizer on (the shipped default), the executor left the project's context to the optimizer and
skipped its own composition. Three turns then answered without the project's passages, and only a log line knew: an
optimizer that raised handed the turn to the manual pipeline, which built the window without them; the optimizer was
handed the turn's project only from inside the prompt templates' branch, so a template module that could not load
took the project with it; and a conversation turn whose conversation manager was missing went the single-turn way,
where nothing composed the project at all.

  * PZ1 -- AN OPTIMIZER THAT FAILS KEEPS THE PROJECT: the optimizer raises; the request the fallback sends carries
    the project's passages, wrapped, in the place the manual path gives them in the turn's tail (after the memory,
    before the web's results), labelled by the digest of their text, and their line is said; the request, its label
    and the response cache's key are, byte for byte, those of the manual path for the same turn. Limit (BACKLOG 267
    a): the archive's snippets are matched against the tail before the optimizer is asked, so a snippet that repeats
    the project's text is not dropped on the optimizer's path, its fallback included.
  * PZ2 -- THE OPTIMIZER IS HANDED THE PROJECT WITHOUT THE TEMPLATES: the prompt templates absent (their module
    cannot load), the optimizer is still handed the turn's project; a store that cannot say it hands none, and the
    turn goes on.
  * PZ3 -- A SINGLE TURN COMPOSES ITS PROJECT: the conversation manager missing, the optimizer on, the turn goes the
    single-turn way: the request carries the project's passages and the optimizer is never asked.
  * PZ4 -- ONE RETRIEVAL A TURN: the optimizer that answers composes the project itself (the executor asks nothing);
    the optimizer that fails, or none, asks once (an optimizer that composed the project before it failed asked too:
    BACKLOG 267 b); a turn without a project's text sends what the manual path sends, carries no file and logs no
    withheld context.
  * PZ5 -- THE TEXTS: the CHANGELOG says it.
  * PZ6 -- A WITHHELD PROJECT SAYS SO (review 267-1): nothing can wrap the project's text as data; on the manual
    path, the failed optimizer's and the single turn's, the request carries none of it, and after the passages'
    line, which no longer holds, the turn says it withheld them, in the loader's words (both of the garden's nets,
    printable ASCII).

Local-only (the public distribution ships no tests). The executor runs in the shared isolation window through the
turn label suite's own world (tests/test_turn_label_contracts.py): the model client is scripted, the conversation
store is a stand-in, and the project's composer is stood in on the executor itself, its own retrieval being the
project context suites' (LD24, LD29): no contract reaches an engine or a store on disk.
"""

import json
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_chat_loader_contracts as _ld  # noqa: E402
import test_turn_label_contracts as tl  # noqa: E402
from _isolation import REPO  # noqa: E402

BUDGET_S = {
    "test_pz1_an_optimizer_that_fails_keeps_the_project": 8.0,
    "test_pz2_the_optimizer_is_handed_the_project_without_the_templates": 6.0,
    "test_pz3_a_single_turn_composes_its_project": 6.0,
    "test_pz4_one_retrieval_a_turn": 8.0,
    "test_pz5_the_texts": 2.0,
    "test_pz6_a_withheld_project_says_so": 10.0,
}

_NOTES = "Project notes: the venue is Hall B."
_MEMORY = ("Relevant memories:\n- Alice lives in Lyon.", [("f1", True)])
_LINE = "[OK] Project context injected: 1 chunks, ~9 tokens, from the project's Mycelium store"
_WITHHELD = "[!] Project context withheld: the untrusted-data wrapper is unavailable"


class _Failing:
    """An optimizer that raises when asked."""

    enabled = True

    def __init__(self):
        self.asked = []

    def optimize(self, **kwargs):
        self.asked.append(kwargs)
        raise RuntimeError("the optimizer fails")


class _Keys:
    """The response cache, keeping the arguments of each key it is asked for and holding no answer."""

    enabled = True

    def __init__(self):
        self.asked = []

    def make_cache_key(self, *args):
        self.asked.append(("single", args))
        return "k" * 16

    def make_conversation_cache_key(self, *args):
        self.asked.append(("conversation", args))
        return "k" * 16

    def get(self, key):
        return None

    def put(self, *args, **kwargs):
        return None


class _Answering(tl._TrimmingOptimizer):
    """An optimizer that answers, and keeps what it was handed."""

    def __init__(self):
        self.asked = []

    def optimize(self, **kwargs):
        self.asked.append(kwargs)
        return super().optimize(**kwargs)


def _turn(optimizer, *, notes=_NOTES, web=True, memory=None, cache=None, conversation="conv-1", setup=None):
    """One turn of the real executor: the optimizer given (None: off), the project's composer stood in, a memory block
    placed and a response cache asked when given. Returns the messages sent, the request's label, the composer's calls
    and the lines said."""
    mod, wrapper, scripted, store, librarian, restore = tl._executor(web_results=[tl._WEB_RESULT] if web else None,
                                                                     memory=memory, cache=cache)
    try:
        if optimizer is None:
            mod.CONTEXT_OPTIMIZER_AVAILABLE = False
        else:
            mod.CONTEXT_OPTIMIZER_AVAILABLE = True
            mod._get_context_optimizer = lambda: optimizer
        if setup is not None:
            setup(mod)
        ex = mod.Executor()
        ex._memory_enabled = memory is not None
        calls, said = [], []

        def compose(question, conversation_id, on_status=None, budget_tokens=None, fit=None, facts=None):
            calls.append(conversation_id)
            if notes and on_status is not None:
                on_status(_LINE)
            return notes

        ex._compose_project_context = compose
        run = SimpleNamespace(stop=threading.Event(), results={})
        tl._drive(ex.execute(tl._QUESTION, tl._routing(), refine=False, web_search=web,
                             conversation_id=conversation, run=run, on_status=said.append))
        return tl._sent(scripted), run.results.get("context_label"), calls, said
    finally:
        restore()


def test_pz1_an_optimizer_that_fails_keeps_the_project():
    manual_keys, keys = _Keys(), _Keys()
    manual_sent, manual_label, manual_calls, manual_said = _turn(None, memory=_MEMORY, cache=manual_keys)
    failing = _Failing()
    sent, label, calls, said = _turn(failing, memory=_MEMORY, cache=keys)
    assert failing.asked, "control: the optimizer was asked, and raised"
    text = json.dumps(manual_sent)
    assert all(w in text for w in ("Alice lives in Lyon.", "Hall B", "The hall opens on Monday.")), (
        "control: the manual path sends the memory, the project and the web"
    )
    text = json.dumps(sent)
    assert "Hall B" in text, "the fallback's request carries the project's passages"
    assert text.index("Alice lives in Lyon.") < text.index("Hall B") < text.index("The hall opens on Monday."), (
        "the project's passages take the place the manual path gives them, after the memory, before the web"
    )
    assert sent == manual_sent, "the fallback sends, byte for byte, what the manual path sends for the same turn"
    assert label == manual_label and "file:" + tl._sha(_NOTES) in label[1], (label, manual_label)
    assert _LINE in said, "the passages' line is said"
    assert manual_keys.asked and "Hall B" in repr(manual_keys.asked), "control: the manual path's key holds the project"
    assert keys.asked == manual_keys.asked, (
        "the response cache's key is the manual path's: the identity view holds the project's passages"
    )


def test_pz2_the_optimizer_is_handed_the_project_without_the_templates():
    def with_project(mod):
        assert not mod.PROMPT_OPTIMIZATION_AVAILABLE, "control: the templates' module cannot load in this window"
        mod.PROJECT_CONTEXT_AVAILABLE = True
        mod._project_store = SimpleNamespace(
            get_project_for_conversation=lambda cid: "proj-1" if cid == "conv-1" else None)

    answering = _Answering()
    _turn(answering, web=False, setup=with_project)
    assert answering.asked, "control: the optimizer was asked"
    asked = answering.asked[-1]
    assert asked["project_id"] == "proj-1" and asked["project_active"] is True, (
        f"the optimizer is handed the turn's project ({asked['project_id']!r}, {asked['project_active']!r})"
    )

    def unreadable(mod):
        mod.PROJECT_CONTEXT_AVAILABLE = True

        def get(cid):
            raise OSError("the projects' store cannot be read")

        mod._project_store = SimpleNamespace(get_project_for_conversation=get)

    answering = _Answering()
    sent, label, calls, said = _turn(answering, web=False, setup=unreadable)
    assert answering.asked and answering.asked[-1]["project_id"] is None, "a store that cannot say hands no project"
    assert sent, "control: the turn went on"


def test_pz3_a_single_turn_composes_its_project():
    def no_manager(mod):
        mod.conversation_manager = None

    answering = _Answering()
    sent, label, calls, said = _turn(answering, web=False, setup=no_manager)
    assert not answering.asked, "the single turn never asks the optimizer"
    assert "Hall B" in json.dumps(sent), "the single turn's request carries the project's passages"
    assert calls == ["conv-1"] and _LINE in said, (calls, said)
    assert "file:" + tl._sha(_NOTES) in label[1], label


def test_pz4_one_retrieval_a_turn(caplog):
    answering = _Answering()
    sent, label, calls, said = _turn(answering, web=False)
    assert answering.asked, "control: the optimizer answered"
    assert calls == [], "the optimizer that answers composes the project itself: the executor asks nothing"
    for optimizer in (_Failing(), None):
        sent, label, calls, said = _turn(optimizer, web=False)
        assert calls == ["conv-1"], f"{type(optimizer).__name__}: the project is asked once ({calls})"
    caplog.set_level(logging.WARNING)
    caplog.clear()
    manual_sent, manual_label, _calls, _said = _turn(None, notes="")
    sent, label, calls, said = _turn(_Failing(), notes="")
    assert "withheld" not in caplog.text, "a turn without a project's text withholds nothing and says nothing of it"
    assert calls == ["conv-1"] and sent == manual_sent and label == manual_label, (
        "a turn without a project's text sends what the manual path sends"
    )
    assert "file" not in label[0], f"a turn without a project's text carries no file ({label})"
    assert "Hall B" not in json.dumps(sent) and not any("Project context" in line for line in said)


def test_pz5_the_texts():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = " ".join(next((e for e in changelog.split("\n- ") if "PZ1-PZ6" in e), "").split())
    assert "optimizer" in entry and "project" in entry and "passages" in entry, entry[:300]


def test_pz6_a_withheld_project_says_so():
    def no_wrapper(mod):
        mod.UNTRUSTED_WRAP_AVAILABLE = False

    def no_manager(mod):
        no_wrapper(mod)
        mod.conversation_manager = None

    turns = {"manual": _turn(None, web=False, setup=no_wrapper),
             "failed optimizer": _turn(_Failing(), web=False, setup=no_wrapper),
             "single turn": _turn(_Answering(), web=False, setup=no_manager)}
    for name, (sent, label, calls, said) in turns.items():
        assert calls == ["conv-1"] and _LINE in said, f"{name}: control: the passages were composed and said"
        assert "Hall B" not in json.dumps(sent) and "file" not in label[0], f"{name}: nothing of them is sent"
        project = [line for line in said if "Project context" in line]
        assert project[-1] == _WITHHELD, f"{name}: the turn says, after the passages' line, it withheld them: {said}"
    out = _ld._node("words", {"sample": _ld._SAMPLE, "ranAs": [], "statuses": [_LINE, _WITHHELD], "replays": {}})
    words = out["statuses"][1]
    assert words and "withheld" in words and "passage" in words, f"the loader says it: {out['statuses']}"
    ethics, restore = _ld._nets()
    try:
        assert ethics.class_hits("Thinking"), "control: the class net refuses a governed word"
        assert not ethics.class_hits(words) and not ethics.banned_hits(words), words
    finally:
        restore()
    assert all(" " <= c <= "~" for c in words), words
