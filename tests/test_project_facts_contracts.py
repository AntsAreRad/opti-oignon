#!/usr/bin/env python3
"""Contracts for what an answer keeps of its project's context: closed forms, never a name.

A turn's project context was said in passing only (status lines the chat's loader replaces one by one); nothing of
it stayed with the answer. Now the turn writes its facts -- the source of the passages kept, their number, the form of
what the turn said of its context -- in its results on every path, the answer is saved with them, the ``done`` frame
carries them and the messages' read-back gives them again. What is kept never names a file: what a turn says can.

  * PQ1 -- THE COMPOSER WRITES THE FACTS ITS LINES SAY: the real ``_compose_project_context`` over a stand-in
    builder: the passages' source and count (the three sources; a text fitted to its share of the window is "cut";
    instructions with no passage "none"), the form of what the turn said with its numbers (files counted, seconds),
    a form the table does not know "unknown", a retrieval the window had no room for "left_out"; no name in them.
    A "cut" text keeps the count its line says, the builder's before the fit (BACKLOG 266 e).
  * PQ2 -- EVERY PATH OF THE TURN KEEPS THEM: the manual path, the failed optimizer's fallback, an answer served
    from the cache and the single turn write the composer's facts in the turn's results and save them with the
    answer (the single turn saves nothing); a text nothing could wrap is
    kept "withheld", with no passage; on the optimizer's path the facts are its project zone's; a turn without a
    project writes nothing and is saved as before.
  * PQ3 -- THE DONE FRAME AND THE READ-BACK GIVE THEM: the chat route's ``done`` frame carries the turn's facts (none
    when it has none); ``GET .../messages`` gives each answer's facts in their closed shape only, and nothing else of
    a message's metadata.
  * PQ4 -- THE TWO TABLES AGREE: the forms this server keeps (``project_facts.SAID``) and the loader's
    (``loaderWords.ts``, ``PROJECT_SAID``) are the same patterns, in the same order, each under the same key; each
    form the real turn writes is read by both.
  * PQ5 -- NO NAME IS KEPT: a turn that names the project's files keeps their count; stored facts read back keep
    only their closed shape (a name smuggled in, an unknown source or form, a wrong number: nothing).
  * PQ6 -- THE TEXTS: the CHANGELOG says it.
  * PQ8 -- A CALL KEEPS ITS OWN FACTS (review 268-1): two calls on one run (a pipeline's steps share it), the second
    with no project passage (its builder fails): the second answer is saved, and its run carries, no fact of the
    first's.
  * PQ7 -- EVERY MESSAGE WRITE SITE IS ACCOUNTED FOR (supersedes ot25 of tests/test_turn_origin_contracts.py, its
    assertions kept): the executor's two sites that save an answer are each two calls now (with the facts, and as
    before), every one naming its origin: six sites in the executor, none forwarding its keywords.

Local-only (the public distribution ships no tests). The executor runs in the shared isolation window through the
project context words suite's and the turn label suite's own worlds, the routes in windows of their own; the facts'
module is loaded from its file wherever a window does not carry it: no contract reaches an engine or a store on disk.
"""

import asyncio
import importlib.util
import json
import re
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_chat_loader_contracts as _ld  # noqa: E402
import test_pipeline_step_relay_contracts as relay  # noqa: E402
import test_project_context_words_contracts as ldw  # noqa: E402
import test_project_fallback_contracts as pz  # noqa: E402
import test_turn_label_contracts as tl  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_pq1_the_composer_writes_the_facts_its_lines_say": 6.0,
    "test_pq2_every_path_of_the_turn_keeps_them": 10.0,
    "test_pq3_the_done_frame_and_the_read_back_give_them": 8.0,
    "test_pq4_the_two_tables_agree": 8.0,
    "test_pq5_no_name_is_kept": 4.0,
    "test_pq6_the_texts": 2.0,
    "test_pq7_every_message_write_site_is_accounted_for": 4.0,
    "test_pq8_a_call_keeps_its_own_facts": 6.0,
}

_LEFT = ("the project holds 2 file(s) its Mycelium store does not (scan.pdf, minutes-2024.docx): those are left to "
         "the knowledge base")
_NAMES = ("scan.pdf", "minutes-2024", "notes.md")


def _facts_module():
    """``opti_oignon/project_facts.py`` loaded from its file (stdlib only), never through the package."""
    spec = importlib.util.spec_from_file_location("project_facts_under_contract",
                                                  REPO / "opti_oignon" / "project_facts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PF = _facts_module()


def _composed(executor, ctx, budget_tokens=None, fit=None):
    """The facts and the lines the executor's own project path writes for a builder that answers ``ctx``."""
    ldw._executor_lines(executor, ctx)          # seeds the project's store, trigger and builder
    lines, facts = [], {}
    extra = {} if fit is None else {"fit": fit}
    executor.Executor._compose_project_context(None, "where is the plan", "c1", on_status=lines.append,
                                               budget_tokens=budget_tokens, facts=facts, **extra)
    return facts, lines


def test_pq1_the_composer_writes_the_facts_its_lines_say():
    ctx = ldw._context
    cases = {
        "mycelium": (ctx(context_text="[a (lines 1-4)] plan", chunks_used=3, total_tokens_estimate=40,
                         source="mycelium"), {"source": "mycelium", "passages": 3, "said": None}),
        "both": (ctx(context_text="[a (lines 1-4)] plan", chunks_used=4, total_tokens_estimate=50,
                     source="mycelium+rag", said=_LEFT),
                 {"source": "mycelium+rag", "passages": 4, "said": {"form": "left", "files": 2}}),
        "rag": (ctx(context_text="from the knowledge base", chunks_used=2, total_tokens_estimate=30, source="rag",
                    said="the project holds files its Mycelium store does not: the knowledge base serves it"),
                {"source": "rag", "passages": 2, "said": {"form": "rag_serves"}}),
        "unreadable": (ctx(said="the project's Mycelium store cannot be read (DatabaseError): no project passage"),
                       {"source": "none", "passages": 0, "said": {"form": "unreadable"}}),
        "instructions": (ctx(context_text="Answer in English.", chunks_used=0, total_tokens_estimate=5,
                             source="mycelium"), {"source": "none", "passages": 0, "said": None}),
        "lexical": (ctx(context_text="[a (lines 1-4)] plan", chunks_used=1, total_tokens_estimate=9,
                        source="mycelium", said="the search took longer than 2.5 s: the lexical stages answered"),
                    {"source": "mycelium", "passages": 1, "said": {"form": "lexical", "s": 2.5}}),
        "unknown": (ctx(context_text="[a (lines 1-4)] plan", chunks_used=1, total_tokens_estimate=9,
                        source="mycelium", said="a new form that names notes.md"),
                    {"source": "mycelium", "passages": 1, "said": {"form": "unknown"}}),
    }
    executor, restore = ldw._load_executor()
    try:
        executor._project_facts = PF
        got = {name: _composed(executor, c) for name, (c, _want) in cases.items()}
        cut = _composed(executor, cases["both"][0], fit=lambda text: text[:5])
        floor = _composed(executor, cases["mycelium"][0], budget_tokens=executor._RAG_FLOOR - 1)
    finally:
        restore()
    for name, (_c, want) in cases.items():
        assert got[name][0] == want, f"{name}: {got[name][0]} (said: {got[name][1]})"
    assert any(line.endswith("from the project's Mycelium store") for line in got["mycelium"][1]), (
        "control: the passages' line names the source the facts keep")
    assert cut[0] == {"source": "cut", "passages": 4, "said": {"form": "left", "files": 2}}, cut
    assert floor[0] == {"source": "left_out", "passages": 0, "said": None}, floor
    every = json.dumps([g[0] for g in got.values()] + [cut[0], floor[0]])
    assert not [n for n in _NAMES if n in every], f"no name is kept: {every}"


class _Metadata:
    """The turn label world's store, keeping what each answer is saved with beside its text."""

    def __init__(self, store):
        self.saved, add = [], store.add_message

        def add_message(conversation_id, role, content, model=None, metadata=None, **kwargs):
            self.saved.append((role, metadata))
            return add(conversation_id, role, content, model=model, metadata=metadata, **kwargs)

        store.add_message = add_message


def _zone_report(strategy, passages, said):
    zone = SimpleNamespace(zone="project", strategy=strategy, passages=passages, said=said, actual_tokens=40,
                           as_dict=lambda: {"zone": "project"})
    return SimpleNamespace(total_trimmed=0, overflow=False, preset_used="p", duration_ms=1.0, zones=[zone])


class _Zoned(pz._Answering):
    """An optimizer that answers with a project zone."""

    def __init__(self, report):
        super().__init__()
        self.report = report

    def optimize(self, **kwargs):
        result = super().optimize(**kwargs)
        result.report = self.report
        return result


def _path(optimizer, *, wrapper=True, manager=True, project=True, facts=None, cache=None):
    """One turn of the real executor; returns its results' facts and the metadata its answer was saved with."""
    facts = PF.facts("mycelium", 2) if facts is None else facts
    mod, wrapper_mod, scripted, store, librarian, restore = tl._executor(cache=cache)
    try:
        mod._project_facts = PF
        kept = _Metadata(store)
        mod.CONTEXT_OPTIMIZER_AVAILABLE = optimizer is not None
        if optimizer is not None:
            mod._get_context_optimizer = lambda: optimizer
        if not wrapper:
            mod.UNTRUSTED_WRAP_AVAILABLE = False
        if not manager:
            mod.conversation_manager = None
        if project:
            mod.PROJECT_CONTEXT_AVAILABLE = True
            mod._project_store = SimpleNamespace(get_project_for_conversation=lambda cid: "proj-1")
        ex = mod.Executor()

        def compose(question, conversation_id, on_status=None, budget_tokens=None, fit=None, facts_out=None,
                    **kwargs):
            sink = kwargs.get("facts", facts_out)
            if project and sink is not None:
                sink.update(facts)
            return pz._NOTES if project else ""

        ex._compose_project_context = compose
        run = SimpleNamespace(stop=threading.Event(), results={})
        tl._drive(ex.execute(tl._QUESTION, tl._routing(), refine=False, conversation_id="conv-1", run=run))
        answers = [metadata for role, metadata in kept.saved if role == "assistant"]
        return run.results.get("project_context"), answers
    finally:
        restore()


def test_pq2_every_path_of_the_turn_keeps_them():
    facts = PF.facts("mycelium", 2, _LEFT)
    for name, optimizer, extra in (("manual", None, {}), ("failed optimizer", pz._Failing(), {})):
        got, answers = _path(optimizer, facts=facts, **extra)
        assert got == facts, f"{name}: the turn's results keep the composer's facts ({got})"
        assert answers == [{"project_context": facts}], f"{name}: the answer is saved with them ({answers})"
    got, answers = _path(None, facts=facts, cache=tl._Cache())
    assert got == facts and answers == [{"project_context": facts}], (
        f"an answer served from the cache is saved with the turn's facts ({got}, {answers})")
    got, answers = _path(pz._Answering(), manager=False, facts=facts)
    assert got == facts and answers == [], f"single turn: kept in its results, nothing saved ({got}, {answers})"
    got, answers = _path(None, wrapper=False, facts=facts)
    assert got == {"source": "withheld", "passages": 0, "said": facts["said"]}, f"withheld: {got}"
    zoned = _Zoned(_zone_report("mycelium+rag", 3, _LEFT))
    got, answers = _path(zoned)
    want = {"source": "mycelium+rag", "passages": 3, "said": {"form": "left", "files": 2}}
    assert zoned.asked and got == want and answers == [{"project_context": want}], (
        f"the optimizer's path keeps its project zone's facts ({got}, {answers})")
    got, answers = _path(_Zoned(_zone_report("withheld", 0, "")))
    assert got == {"source": "withheld", "passages": 0, "said": None}, got
    got, answers = _path(_Zoned(_zone_report("mycelium", 0, "")))
    assert got == {"source": "none", "passages": 0, "said": None}, f"a zone that kept no passage names none: {got}"
    unread = "the project's Mycelium store cannot be read (DatabaseError): no project passage"
    for strategy, passages, said, want in (
            ("unreadable", 0, unread, {"source": "none", "passages": 0, "said": {"form": "unreadable"}}),
            ("rag", 2, "", {"source": "rag", "passages": 2, "said": None}),
            ("error", 0, "", None), ("unified", 4, "", None)):
        got, answers = _path(_Zoned(_zone_report(strategy, passages, said)))
        assert got == want, f"zone {strategy}: {got}"
    for optimizer in (None, _Zoned(_zone_report("none", 0, ""))):
        got, answers = _path(optimizer, project=False)
        assert got is None and answers == [None], (
            f"a turn without a project writes nothing and is saved as before ({got}, {answers})")


def _done(facts):
    def script(run, on_status):
        if facts is not None:
            run.results["project_context"] = facts
        yield "An answer."

    sent = relay._stream(relay._Runner(script))
    return next(f for f in sent if f.get("type") == "done").get("metadata") or {}


def _messages(stored):
    """``GET /api/conversations/c1/messages`` through the real route over a stand-in store holding ``stored``."""
    rows = [SimpleNamespace(id=n, role="assistant", content="An answer.", timestamp=None, model="m",
                            token_estimate=3, metadata=metadata) for n, metadata in enumerate(stored, 1)]
    manager = SimpleNamespace(get_conversation=lambda cid: SimpleNamespace(id=cid), get_messages=lambda cid: rows)
    deps = types.ModuleType("opti_oignon.api.deps")
    deps.CONVERSATION_AVAILABLE = True
    deps.conversation_manager = manager
    loaded, restore = isolate(
        targets={"opti_oignon.api.schemas": source("api", "schemas.py"),
                 "opti_oignon.project_facts": source("project_facts.py"),
                 "opti_oignon.api.routes_conversations": source("api", "routes_conversations.py")},
        seeded={"opti_oignon.api.deps": deps}, packages=("opti_oignon.api",))
    try:
        routes = loaded["opti_oignon.api.routes_conversations"]
        routes.conversation_manager = manager
        if hasattr(routes, "_check_available"):
            routes._check_available = lambda: None
        return routes.get_messages("c1")
    finally:
        restore()


def _plain_done(facts):
    """The done frame of a plain chat turn (no pipeline, no agentic executor) through the real route."""
    def execute(*args, run=None, **kwargs):
        if facts is not None:
            run.results["project_context"] = facts
        yield "An answer."

    world = relay._load(relay._Runner(lambda run, on_status: iter(())))
    world.rc.AGENTIC_EXECUTOR_AVAILABLE = False
    world.rc.executor = SimpleNamespace(execute=execute)
    ws = relay._Socket()
    try:
        request = world.schemas.ChatRequest(conversation_id="c1", message="Where is the plan?")
        asyncio.run(world.rc._stream_response(ws, "c1", request.message, request))
    finally:
        world.close()
    return next(f for f in ws.sent if f.get("type") == "done").get("metadata") or {}


def test_pq3_the_done_frame_and_the_read_back_give_them():
    facts = PF.facts("rag", 2, "the project holds files its Mycelium store does not: the knowledge base serves it")
    assert _done(facts).get("project_context") == facts, "the done frame carries the turn's facts"
    assert _plain_done(facts).get("project_context") == facts, "a plain chat turn's done frame carries them too"
    assert "project_context" not in _done(None), "a turn without them sends none"
    read = _messages([{"model": "m", "project_context": facts}, {"model": "m"}, None,
                      {"project_context": {**facts, "source": "elsewhere"}}])
    assert [m["project_context"] for m in read] == [facts, None, None, None], read
    assert all(set(m) == {"id", "role", "content", "timestamp", "model", "token_estimate", "project_context"}
               for m in read), "nothing else of a message's metadata leaves"


def test_pq4_the_two_tables_agree():
    written = (REPO / "frontend" / "src" / "lib" / "chat" / "loaderWords.ts").read_text(encoding="utf-8")
    block = written.split("const PROJECT_SAID", 1)[1].split("];", 1)[0]
    theirs = re.findall(r"\[\s*/\^(.+?)\$/,\s*'project_(\w+)'", block, re.S)
    ours = [(pattern.pattern.removeprefix("\\A").removesuffix("\\Z"), form) for pattern, form, _names in PF.SAID]
    assert all(p.pattern.startswith("\\A") and p.pattern.endswith("\\Z") for p, _f, _n in PF.SAID), (
        "each of ours is anchored at both ends, as the loader's are")
    assert len(theirs) == len(ours) == 6, (theirs, ours)
    assert [form for _p, form in theirs] == [form for _p, form in ours], "the same forms, in the same order"
    assert [p.replace("\\/", "/") for p, _f in theirs] == [p for p, _f in ours], "the same patterns"
    assert all(p.flags == int(re.ASCII) for p, _f, _n in PF.SAID), (
        "ours read as the loader's, which has no flag: \\w and \\d are ASCII there, and nothing else is set")
    assert PF.said_form("the project's Mycelium store cannot be read (\u00c9chec): no project passage") == {
        "form": "unknown"}, "a non-ASCII name the loader does not read is not a form here either"
    samples = {
        "unreadable": "the project's Mycelium store cannot be read (DatabaseError): no project passage",
        "search_failed": "the project's Mycelium search failed (TimeoutError)",
        "rag_serves": "the project holds files its Mycelium store does not: the knowledge base serves it",
        "lexical": "the search took longer than 10 s: the lexical stages answered",
        "left": _LEFT,
        "lexical_left": "the search took longer than 2.5 s: the lexical stages answered; " + _LEFT,
    }
    assert [PF.said_form(s)["form"] for s in samples.values()] == list(samples), "ours reads each form"
    out = _ld._node("words", {"sample": _ld._SAMPLE, "ranAs": [], "replays": {},
                              "statuses": [f"[!] Project context: {s}" for s in samples.values()]})
    assert all(out["statuses"]), f"the loader reads each form: {out['statuses']}"


def test_pq5_no_name_is_kept():
    facts = PF.facts("mycelium+rag", 4, _LEFT)
    assert facts == {"source": "mycelium+rag", "passages": 4, "said": {"form": "left", "files": 2}}, facts
    assert PF.clean(facts) == facts, "control: kept facts read back whole"
    for smuggled in ({**facts, "said": {"form": "left", "files": 2, "names": "scan.pdf"}},
                     {**facts, "names": ["scan.pdf"]}):
        assert "scan.pdf" not in json.dumps(PF.clean(smuggled)), smuggled
    lexical = PF.facts("mycelium", 1, "the search took longer than 2.5 s: the lexical stages answered")
    for number in (float("nan"), float("inf"), 1e999):
        assert PF.clean({**lexical, "said": {"form": "lexical", "s": number}}) is None, number
    for said in ({"form": "left", "files": 2, "names": "scan.pdf"}, {"form": "a form of /home/x"}):
        try:
            PF.facts("mycelium", 1, said)
        except ValueError:
            pass
        else:
            raise AssertionError(f"a said form outside the table is refused by name: {said}")
    for wrong in ({**facts, "source": "the store at /home/x"}, {**facts, "passages": -1},
                  {**facts, "passages": True}, {**facts, "said": {"form": "a form of /home/x"}},
                  {**facts, "said": {"form": "left", "files": "2"}}, {**facts, "said": {"form": "left", "files": 2.5}},
                  {**facts, "said": "scan.pdf"}, "scan.pdf"):
        assert PF.clean(wrong) is None, wrong
    try:
        PF.facts("the store at /home/x", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("a source outside the table is refused by name")


def test_pq6_the_texts():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = " ".join(next((e for e in changelog.split("\n- ") if "PQ1-PQ8" in e), "").split())
    assert "project" in entry and "answer" in entry and "name" in entry, entry[:300]


def test_pq7_every_message_write_site_is_accounted_for():
    import test_turn_origin_contracts as ot

    named, defaulted, forwarded, selftest, inserts = ot._census()
    expected = {**ot._NAMED, "opti_oignon/executor.py": 6}
    assert sum(named.values()) >= 10, f"the census finds the sites that name an origin: {dict(named)}"
    assert dict(named) == expected, f"the sites that name an origin: {dict(named)}"
    assert set(defaulted) == ot._DEFAULTED, f"only benches and evaluations keep the legacy default: {dict(defaulted)}"
    assert dict(forwarded) == {"opti_oignon/conversation.py": 1}, "one envelope forwards its keywords"
    assert len(inserts) >= 5, f"the census finds the insert statements: {inserts}"
    assert all(has for _rel, _line, has in inserts), (
        f"every insert into a message table names the origin: {[i for i in inserts if not i[2]]}"
    )


def test_pq8_a_call_keeps_its_own_facts():
    mod, wrapper_mod, scripted, store, librarian, restore = tl._executor()
    try:
        mod._project_facts = PF
        kept = _Metadata(store)
        mod.CONTEXT_OPTIMIZER_AVAILABLE = False
        ex = mod.Executor()
        calls = []

        def compose(question, conversation_id, on_status=None, budget_tokens=None, fit=None, **kwargs):
            calls.append(question)
            if len(calls) == 1:
                kwargs["facts"].update(PF.facts("mycelium", 3))
                return pz._NOTES
            return ""                    # the second call's builder failed: no fact, no text

        ex._compose_project_context = compose
        run = SimpleNamespace(stop=threading.Event(), results={})
        tl._drive(ex.execute(tl._QUESTION, tl._routing(), refine=False, conversation_id="conv-1", run=run))
        first = run.results.get("project_context")
        tl._drive(ex.execute("And the venue?", tl._routing(), refine=False, conversation_id="conv-1", run=run))
        answers = [metadata for role, metadata in kept.saved if role == "assistant"]
    finally:
        restore()
    assert first == PF.facts("mycelium", 3) and len(calls) == 2, "control: the first call kept its facts"
    assert run.results.get("project_context") is None, "the second call's run carries none of the first's facts"
    assert answers == [{"project_context": first}, None], f"the second answer is saved without them: {answers}"
