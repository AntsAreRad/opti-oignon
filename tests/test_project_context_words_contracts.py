#!/usr/bin/env python3
"""Contracts for the words that say where a project's context came from, and what the Mycelium was not given.

The chat's loader shows a server status only when its closed table maps it (``lib/chat/loaderWords.ts``). The source
of a project's passages went in a line of its own the table did not map, and what the turn says of its context (a
store that cannot be read, the knowledge base serving, the lexical stages answering, files left to the knowledge
base) went BEFORE the passages' line, which took its place at once. The settings' optimizer panel had the project
zone's source and detail and showed neither; the projects panel read neither index answer's ``mycelium``.

  * LD24 -- THE PASSAGES SAY THEIR SOURCE, WHAT THE TURN SAYS FOLLOWS: the executor's own project lines (the real
    ``_compose_project_context`` over a stand-in builder) are, in order, the trigger, the passages with their source
    (only when there are passages; one passage said as one), then what the turn says; each is said in the table's
    words, and replayed through the loader the project's last line is the turn's; the window's two "left out" lines
    are said too.
  * LD25 -- EVERY FORM THE TURN SAYS, IN WORDS, NO FILE NAMED: each form the real turn writes (the chat's search over
    a real project store: the lexical stages past the deadline, files left to the knowledge base, both, a store that
    cannot be read, one that holds none of the files, a search that fails; a deadline in whole seconds as in tenths)
    is said in the table's words, the files counted and never named, a name broken by any line separator too; a near
    form is not said; every word passes both of the garden's nets, printable ASCII.
  * LD26 -- THE OPTIMIZER PANEL SAYS THE SOURCE (review 266-1, 3): the optimizer's project zone names a source only
    when passages were kept (no passage, or no source asked: ``none``, the panel silent; the real builder over a
    budget that holds no chunk and over a missing project too); the strategies it reports are the panel's table, a
    store that could not be read or searched said as such, and the panel shows the zone's detail.
  * LD27 -- THE PROJECTS PANEL READS THE INDEX ANSWERS: the fields the frontend types for ``POST .../files/{id}/index``
    and ``POST .../reindex`` are the keys the routes answer (the real routes over a real projects' store), the
    count read from ``chunk_count``; both panels say what the Mycelium was not given when there is any.
  * LD28 -- THE TEXTS: the CHANGELOG says it.
  * LD29 -- THE SOURCE IS WHERE THE KEPT PASSAGES CAME FROM (review 266-1, 266-2): a project the store serves in part
    whose knowledge base gives none of the other files is said served by its store alone; one whose store finds
    nothing while the knowledge base serves the other files, by the knowledge base alone; both only when both gave
    passages kept within the budget (a knowledge base's chunk cut to fit counts, one the budget leaves out does not).
  * LD30 -- THE OPTIMIZER'S PATH SAYS THE PROJECT'S LINES (review 266-2): with the optimizer on (the shipped default),
    the executor says the project zone's passages with their source and what the turn said, after the optimizer's own
    line, in the words the executor's own path uses; a zone whose text was withheld says no passage and names no
    source (the panel says it withheld), a zone of the unified layer nothing.
  * LD32 -- THE BUDGET DECIDES WHAT WAS KEPT (review 266-2): LD29's budget clause, on the real builder: the store's
    passages fit, the knowledge base's long chunk cut to fit (both) or left out by a budget with no room left (the
    store alone).
  * LD31 -- THE JOURNAL'S FIT IS SAID (review 266-2): on the onion journal's path the passages' line is said after the
    text is fitted to its share of the window: cut, it names no source (the passages the fit drops are the knowledge
    base's, last) and says it was cut; fitted to nothing, no passages' line; untouched, the source as before.

Local-only (the public distribution ships no tests). The table runs under Node through the loader contracts' own
driver (tests/test_chat_loader_contracts.py), the nets through ``allium/ethics.py`` loaded by the shared isolation
window, and the turn's forms through the project turn suite's world: no contract reaches the engine.
"""

import re
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_chat_loader_contracts as _ld  # noqa: E402
import test_project_turn_contracts as tm  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402
from test_project_mycelium_contracts import NOTES  # noqa: E402

BUDGET_S = {
    "test_ld24_the_passages_say_their_source_and_the_turns_words_follow": 8.0,
    "test_ld25_every_form_the_turn_says_in_words_no_file_named": 10.0,
    "test_ld26_the_optimizer_panel_says_the_source": 4.0,
    "test_ld27_the_projects_panel_reads_the_index_answers": 6.0,
    "test_ld28_the_texts": 2.0,
    "test_ld29_the_source_is_where_the_kept_passages_came_from": 6.0,
    "test_ld32_the_budget_decides_what_was_kept": 6.0,
    "test_ld30_the_optimizers_path_says_the_projects_lines": 8.0,
    "test_ld31_the_journals_fit_is_said": 6.0,
}

# The project turn suite's fixtures.
no_engine = tm.no_engine
owned = tm.owned
world = tm.world
chat = tm.chat

_FRONT = REPO / "frontend" / "src" / "lib"
_PDF = b"%PDF-1.4\x00\x01\x02" * 64

# What a turn says, in the table's words, for each closed form (LD24 hands the executor these; LD25 makes each one
# with the real turn).
_UNREADABLE = "Project context: its Mycelium store cannot be read, no project passage"
_FAILED = "Project context: its Mycelium search failed, no project passage"
_RAG = "Project context: its Mycelium store holds none of its files, the knowledge base serves it"
_LEFT_OUT = "Project context left out: no room in the window"


def _words(statuses, replays=None):
    return _ld._node("words", {"sample": _ld._SAMPLE, "ranAs": [], "statuses": list(statuses),
                               "replays": replays or {}})


def _clean(words):
    """The words that fail a net or are not printable ASCII (none expected)."""
    ethics, restore = _ld._nets()
    try:
        assert ethics.class_hits("Thinking"), "control: the class net refuses a governed word"
        refused = {w: (ethics.class_hits(w), ethics.banned_hits(w)) for w in words
                   if ethics.class_hits(w) or ethics.banned_hits(w)}
    finally:
        restore()
    return refused, [w for w in words if re.search(r"[^ -~]", w)]


class _ForbiddenClient(types.ModuleType):
    """The model client, refusing: no contract here reaches the engine."""

    def __init__(self):
        super().__init__("ollama")

    def chat(self, **kwargs):
        raise AssertionError("direct client call")


def _load_executor():
    """The executor in the shared isolation window, as its neighbours load it (the funnel refusal suite): its config,
    router and backends seeded, the model client refusing."""
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = types.SimpleNamespace(get_model=lambda *a, **k: "test-model:1b",
                                       get_temperature=lambda *a, **k: 0.2)
    router = types.ModuleType("opti_oignon.router")
    router.RoutingResult = type("RoutingResult", (), {})
    backends = types.ModuleType("opti_oignon.inference_backend")
    backends.get_backend_registry = lambda: types.SimpleNamespace(active=None, resolve_backend=lambda model: None)
    had, prev = "ollama" in sys.modules, sys.modules.get("ollama")
    sys.modules["ollama"] = _ForbiddenClient()
    loaded, win_restore = isolate(
        targets={"opti_oignon.context_optimizer": source("context_optimizer.py"),
                 "opti_oignon.context_dedup": source("context_dedup.py"),
                 "opti_oignon.executor": source("executor.py")},
        seeded={"opti_oignon.config": cfg, "opti_oignon.router": router, "opti_oignon.inference_backend": backends},
        packages=("opti_oignon",),
    )

    def restore():
        win_restore()
        if had:
            sys.modules["ollama"] = prev
        else:
            sys.modules.pop("ollama", None)

    return loaded["opti_oignon.executor"], restore


def _executor_lines(executor, ctx, budget_tokens=None, fit=None):
    """The status lines the executor's own project path says for a builder that answers ``ctx``."""
    executor.PROJECT_CONTEXT_AVAILABLE = True
    executor._project_store = types.SimpleNamespace(get_project_for_conversation=lambda conversation_id: "p1")
    executor._trigger_detector = types.SimpleNamespace(
        detect=lambda question, project_id, skip_l3=False: types.SimpleNamespace(
            relevant=True, trigger_level=2, confidence=0.81))
    executor._project_context_builder = types.SimpleNamespace(
        available=True, build_context=lambda project_id, question, budget_tokens=None: ctx,
        build_system_instructions_only=lambda project_id: ctx)
    lines = []
    extra = {} if fit is None else {"fit": fit}
    text = executor.Executor._compose_project_context(None, "where is the plan", "c1", on_status=lines.append,
                                                      budget_tokens=budget_tokens, **extra)
    return text, lines


def _context(context_text="", chunks_used=0, total_tokens_estimate=0, source="", said=""):
    """A built project context as the executor and the optimizer read it."""
    return types.SimpleNamespace(context_text=context_text, chunks_used=chunks_used,
                                 total_tokens_estimate=total_tokens_estimate, source=source, said=said)


def test_ld24_the_passages_say_their_source_and_the_turns_words_follow():
    both_said = ("the project holds 1 file(s) its Mycelium store does not (scan.pdf): those are left to the "
                 "knowledge base")
    cases = {
        "mycelium": _context(context_text="[notes.md (lines 1-4)] plan", chunks_used=3, total_tokens_estimate=40,
                                   source="mycelium"),
        "both": _context(context_text="[notes.md (lines 1-4)] plan", chunks_used=4, total_tokens_estimate=50,
                               source="mycelium+rag", said=both_said),
        "rag": _context(context_text="from the knowledge base", chunks_used=2, total_tokens_estimate=30,
                              source="rag", said="the project holds files its Mycelium store does not: the knowledge "
                                                 "base serves it"),
        "unreadable": _context(source="", said="the project's Mycelium store cannot be read (DatabaseError): "
                                                     "no project passage"),
        "instructions": _context(context_text="Answer in English.", chunks_used=0, total_tokens_estimate=5,
                                       source="mycelium"),
        "single": _context(context_text="from the knowledge base", chunks_used=1, total_tokens_estimate=9,
                           source="rag"),
    }
    executor, restore = _load_executor()
    try:
        lines = {name: _executor_lines(executor, ctx) for name, ctx in cases.items()}
        _floor_text, floor = _executor_lines(executor, cases["mycelium"], budget_tokens=executor._RAG_FLOOR - 1)
    finally:
        restore()
    assert lines["unreadable"][0] == "" and lines["mycelium"][0] == cases["mycelium"].context_text, (
        "control: the text the executor hands on is unchanged")
    written = (REPO / "opti_oignon" / "executor.py").read_text(encoding="utf-8")
    onion = re.findall(r'(?<![\w])status\("(\[!\] Project context left out: [^"]+)"\)', written)
    assert len(onion) == 1 and floor == ["[!] Project context left out: its share of the window holds no chunk"], (
        onion, floor)
    for name, (_text, said) in lines.items():
        assert said[0].startswith("[>] Project context: L2 trigger"), (name, said)
        assert not [line for line in said if line.startswith("[>] Project context from")], (
            f"{name}: the source rides on the passages' line, never a line of its own: {said}")
        oks = [i for i, line in enumerate(said) if line.startswith("[OK] Project context injected:")]
        tells = [i for i, line in enumerate(said) if line.startswith("[!] Project context: ")]
        if cases[name].said:
            assert len(tells) == 1 and tells[0] == len(said) - 1 and all(i < tells[0] for i in oks), (
                f"{name}: what the turn says is the project's last line: {said}")
    flat = [line for name in cases for line in lines[name][1]] + floor + onion
    sent = _ld._SENT
    replays = {name: [["start", 0], ["frame", 10, sent]]
               + [["frame", 20 + 10 * i, _ld._frame("status", message=line)] for i, line in enumerate(said)]
               for name, (_text, said) in lines.items()}
    out = _words(flat, replays)
    said = dict(zip(flat, out["statuses"]))
    by_name = {name: [said[line] for line in lines[name][1]] for name in cases}
    assert by_name["mycelium"] == ["Reading project context", "Project context: 3 passages from its Mycelium store"], (
        by_name["mycelium"])
    assert by_name["both"] == ["Reading project context",
                               "Project context: 4 passages from its Mycelium store and the knowledge base",
                               "Project context: 1 file left to the knowledge base"], by_name["both"]
    assert by_name["rag"] == ["Reading project context", "Project context: 2 passages from the knowledge base", _RAG], (
        by_name["rag"])
    assert by_name["unreadable"] == ["Reading project context", _UNREADABLE], by_name["unreadable"]
    assert by_name["instructions"] == ["Reading project context", "Project context: 0 passages"], (
        "no passage, no source said", by_name["instructions"])
    assert by_name["single"] == ["Reading project context", "Project context: 1 passage from the knowledge base"], (
        "one passage said as one", by_name["single"])
    assert [said[line] for line in floor + onion] == [_LEFT_OUT, _LEFT_OUT], [said[line] for line in floor + onion]
    for name in ("both", "rag", "unreadable"):
        shown = [snap["line"] for snap in out["replays"][name]]
        assert shown[-1] == by_name[name][-1], (f"{name}: the project's last line is what the turn says", shown)
    refused, odd = _clean([w for w in out["statuses"] if w])
    assert not refused and not odd, ("the words pass both nets, printable ASCII", refused, odd)


def _join_turns():
    for thread in threading.enumerate():
        if thread.name == "mycelium-turn":
            thread.join(10)


def _past_the_deadline(chat, monkeypatch, deadline=0.3, slow=1.0):
    """The full search slower than a short deadline: the lexical stages answer."""
    from opti_oignon.mycelium import search

    real = search.Searcher.find

    def find(self, query, k=10, stages=("symbols", "lexical", "dense"), rerank=False):
        if "dense" in stages:
            time.sleep(slow)
        return real(self, query, k, stages, rerank)

    monkeypatch.setattr(search.Searcher, "find", find)
    monkeypatch.setattr(chat.turn, "DEADLINE_S", deadline)


def test_ld25_every_form_the_turn_says_in_words_no_file_named(chat, monkeypatch):
    pdfs = {f"scan{i}.pdf": _PDF for i in range(6)}
    alone, _ = tm._indexed(chat, {"notes.md": NOTES.encode()}, name="alone")
    one, one_added = tm._indexed(chat, {"notes.md": NOTES.encode(), "scan.pdf": _PDF}, name="one")
    six, _ = tm._indexed(chat, {"notes.md": NOTES.encode(), **pdfs}, name="six")
    query = "zanzibar_quartz plan"
    forms = {"one": chat.builder.build_context(one.id, query).said,
             "six": chat.builder.build_context(six.id, query).said,
             "none_held": chat.turn.passages(one.id, [one_added["scan.pdf"].file_path], query).said}
    with monkeypatch.context() as patched:
        _past_the_deadline(chat, patched)
        forms["lexical"] = chat.builder.build_context(alone.id, query).said
        forms["lexical_one"] = chat.builder.build_context(one.id, query).said
        _join_turns()
    with monkeypatch.context() as patched:
        _past_the_deadline(chat, patched, deadline=1.0, slow=1.6)
        forms["lexical_whole"] = chat.builder.build_context(alone.id, query).said
        _join_turns()
    with monkeypatch.context() as patched:
        def failing(project_id, paths, query, k=None):
            raise RuntimeError("the search broke")

        patched.setattr(chat.turn, "passages", failing)
        forms["failed"] = chat.builder.build_context(alone.id, query).said
    (chat.place.folder() / f"project_{alone.id}.db").write_bytes(b"not a store at all " * 512)
    forms["unreadable"] = chat.builder.build_context(alone.id, query).said
    shown_names = sum(f"scan{i}.pdf" in forms["six"] for i in range(6))     # which five: the stored paths' order
    assert all(forms.values()) and shown_names == 5 and "and 1 more" in forms["six"], (
        "control: the real turn said every form, the files named", forms)
    assert "longer than 1 s:" in forms["lexical_whole"], ("control: a whole second is written whole", forms)
    # A name the projects keep may hold any line separator; the form is said all the same.
    for mark, separator in (("lf", "\n"), ("cr", "\r"), ("ls", "\u2028"), ("ps", "\u2029")):
        forms[f"name_{mark}"] = forms["one"].replace("(scan.pdf)", f"(two{separator}lines.pdf)")
    near = ["the project's Mycelium store cannot be read (DatabaseError): no project passage, again",
            "the search took longer than 0.3 s: the lexical stages answered; and something else",
            "the project holds files its Mycelium store does not",
            "the project's Mycelium search failed (RuntimeError) twice",
            "the project holds files its Mycelium store does not: the knowledge base serves it, and more",
            "the project holds 1 file(s) its Mycelium store does not (a.pdf): those are left to the knowledge base, too",
            "a form no turn writes"]
    messages = [f"[!] Project context: {form}" for form in list(forms.values()) + near]
    out = _words(messages)
    said = dict(zip(list(forms) + near, out["statuses"]))
    assert said["one"] == "Project context: 1 file left to the knowledge base", said["one"]
    assert said["six"] == "Project context: 6 files left to the knowledge base", said["six"]
    assert said["none_held"] == _RAG, said["none_held"]
    assert said["lexical"] == "Project context: the search ran past 0.3 s, names and full text answered", (
        said["lexical"])
    assert said["lexical_one"] == ("Project context: the search ran past 0.3 s, names and full text answered, "
                                   "1 file left to the knowledge base"), said["lexical_one"]
    assert said["failed"] == _FAILED and said["unreadable"] == _UNREADABLE, (said["failed"], said["unreadable"])
    assert said["lexical_whole"] == "Project context: the search ran past 1 s, names and full text answered", (
        said["lexical_whole"])
    assert [said[f"name_{mark}"] for mark in ("lf", "cr", "ls", "ps")] == [
        "Project context: 1 file left to the knowledge base"] * 4, said
    assert [said[form] for form in near] == [None] * len(near), ("a near form is not said", said)
    shown = [w for w in out["statuses"] if w]
    assert not [w for w in shown if re.search(r"scan\d?\.pdf|notes\.md|lines\.pdf|\(", w)], (
        "no file named in the words", shown)
    refused, odd = _clean(shown)
    assert not refused and not odd, ("the words pass both nets, printable ASCII", refused, odd)


def test_ld26_the_optimizer_panel_says_the_source(chat):
    from opti_oignon.context_optimizer import ContextOptimizer

    def zone_of(builder, project_id="p1", budget_tokens=4096):
        holder = types.SimpleNamespace(_project_builder=builder, _estimate_tokens=lambda text, model: len(text) // 4)
        return ContextOptimizer._inject_project_context(holder, project_id=project_id, query="zanzibar_quartz plan",
                                                        budget_tokens=budget_tokens, model="m")[1]

    cases = {("mycelium", "", 1): "mycelium", ("mycelium+rag", "files left", 2): "mycelium+rag", ("rag", "", 1): "rag",
             ("", "the project's Mycelium search failed (RuntimeError)", 0): "unreadable",
             ("", "", 0): "none", ("rag", "", 0): "none", ("mycelium", "", 0): "none"}
    reported = {}
    for (origin, said, n), want in cases.items():
        ctx = _context(context_text="text" if n else "", chunks_used=n, source=origin, said=said)
        reported[(origin, said, n)] = zone_of(types.SimpleNamespace(
            build_context=lambda project_id, query, budget_tokens=None, c=ctx: c)).strategy
    assert reported == cases, reported
    project, _added = tm._indexed(chat, {"notes.md": NOTES.encode()})
    assert zone_of(chat.builder, project.id).strategy == "mycelium", "control: the real builder names its source"
    assert zone_of(chat.builder, project.id, budget_tokens=10).strategy == "none", "a budget that holds no chunk"
    assert zone_of(chat.builder, "no-such-project").strategy == "none", "a project that is not there"
    panel = (_FRONT / "components" / "settings" / "ContextOptimizerPanel.svelte").read_text(encoding="utf-8")
    table = re.search(r"const projectSources: Record<string, string> = \{(.*?)\n\t\};", panel, re.S)
    assert table, "the panel keeps a table of the project's sources"
    keys = set(re.findall(r"^\s*'?([\w+]+)'?:", table.group(1), re.M))
    assert keys == set(cases.values()) - {"none"} | {"withheld"}, (keys, set(cases.values()))
    withheld = re.search(r"^\s*withheld: '([^']+)',$", table.group(1), re.M)
    assert withheld and "withheld" in withheld.group(1) and "from" not in withheld.group(1), "a withheld text"
    unreadable = re.search(r"^\s*unreadable: \"([^\"]+)\",$", table.group(1), re.M)
    assert unreadable and "read or searched" in unreadable.group(1), "a search that failed is not said unreadable"
    shown = re.search(r"\{#each lastReport\.zones\.filter\(\(z\) => z\.zone === 'project' && projectSources\[z\.strategy\]\)"
                      r" as zone\}(.*?)\{/each\}", panel, re.S)
    assert shown and "projectSources[zone.strategy]" in shown.group(1) and "zone.detail" in shown.group(1), (
        "the panel shows the zone's source and its detail")


def _fields(source, name):
    body = re.search(rf"export interface {name} \{{(.*?)\n\}}", source, re.S)
    assert body, f"{name} is typed"
    return set(re.findall(r"^\t(\w+)\??:", body.group(1), re.M))


def test_ld27_the_projects_panel_reads_the_index_answers(chat, monkeypatch):
    from opti_oignon.api import routes_projects as rp
    from opti_oignon.project_context import IndexResult

    project, added = tm._project(chat, {"notes.md": NOTES.encode(), "scan.pdf": _PDF})
    monkeypatch.setattr(rp, "project_store", chat.store)
    monkeypatch.setattr(rp, "project_indexer", chat.indexer)
    monkeypatch.setattr(rp, "PROJECTS_AVAILABLE", True)
    monkeypatch.setattr(rp, "PROJECT_CONTEXT_AVAILABLE", True)
    monkeypatch.setattr(chat.indexer, "index_mycelium", lambda project_id, vectors=True: ["scan.pdf: not given"])
    monkeypatch.setattr(chat.indexer, "_index_file", lambda project_id, file_id: IndexResult(
        file_id=file_id, filename="notes.md", success=True, chunk_count=2))
    one = rp.index_project_file(project.id, added["notes.md"].id)
    whole = rp.reindex_project(project.id)
    api = (_FRONT / "api" / "projects.ts").read_text(encoding="utf-8")
    assert _fields(api, "FileIndexResult") == set(one) and "mycelium" in one, (sorted(one), _fields(api, "FileIndexResult"))
    assert _fields(api, "ReindexResult") == set(whole) and "mycelium" in whole, (sorted(whole), _fields(api, "ReindexResult"))
    assert re.search(r"indexProjectFile\(projectId: string, fileId: string\): Promise<FileIndexResult>", api), api[:200]
    assert re.search(r"reindexProject\(projectId: string\): Promise<ReindexResult>", api), "the reindex is typed"
    store = (_FRONT / "stores" / "projects.ts").read_text(encoding="utf-8")
    index_body = store.split("export async function indexFile(", 1)[1].split("\n}\n", 1)[0]
    reindex_body = store.split("export async function reindexAll(", 1)[1].split("\n}\n", 1)[0]
    assert "result.chunk_count" in index_body and "result.chunks" not in index_body, "the count read where it is"
    assert "mycelium: result.mycelium" in index_body and "mycelium: result.mycelium" in reindex_body, (
        "both passes hand on what the Mycelium was not given")
    manager = (_FRONT / "components" / "panels" / "FileManager.svelte").read_text(encoding="utf-8")
    detail = (_FRONT / "components" / "panels" / "ProjectDetail.svelte").read_text(encoding="utf-8")
    assert re.search(r"if \(result\.mycelium\.length > 0\) toastWarning\(myceliumLeft\(result\.mycelium\)", manager), (
        "the file's index says it")
    assert re.search(r"lines\.join\('; '\)", manager), "each line the server gave"
    assert re.search(r"if \(result\.mycelium\.length > 0\) \{\s*toastWarning\(\s*`[^`]*\$\{result\.mycelium\.join\('; '\)\}`",
                     detail), "the reindex says it"


def test_ld28_the_texts():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = " ".join(next((e for e in changelog.split("\n- ") if "LD24-LD32" in e), "").split())
    assert "source" in entry and "Mycelium" in entry and "loader" in entry, entry[:300]


def _two_sources(chat, name, store_finds=True, rag_gives=True, long_rag=False):
    """A project of a note its store holds and a PDF it does not, the knowledge base giving the PDF's chunk or not, the
    store's search finding the note or nothing."""
    project, added = tm._indexed(chat, {"notes.md": NOTES.encode(), "scan.pdf": _PDF}, name=name)
    scan = added["scan.pdf"].file_path

    def knowledge_base(project_id, query, *args, **kwargs):
        chat.asked.append(project_id)
        own = [chat.pc.RetrievedChunk(content="from the note, a second copy", source_file=added["notes.md"].file_path)]
        text = "from the scan" + (" and more of it" * 2000 if long_rag else "")
        return own + ([chat.pc.RetrievedChunk(content=text, source_file=scan)] if rag_gives else [])

    chat.builder.retrieve_chunks = knowledge_base
    if not store_finds:
        real = chat.turn.passages

        def nothing_found(project_id, paths, query, k=None):
            turn = real(project_id, paths, query)
            return chat.turn.Turn(turn.state, passages=[], said=turn.said, uncovered=turn.uncovered)

        chat.turn.passages = nothing_found
    return project


def test_ld29_the_source_is_where_the_kept_passages_came_from(chat, monkeypatch):
    project = _two_sources(chat, "both")
    ctx = chat.builder.build_context(project.id, "zanzibar_quartz plan")
    assert ctx.source == "mycelium+rag" and "from the scan" in ctx.context_text, ("control: both gave", ctx.source)
    project = _two_sources(chat, "store", rag_gives=False)
    ctx = chat.builder.build_context(project.id, "zanzibar_quartz plan")
    assert ctx.chunks_used >= 1 and "from the scan" not in ctx.context_text and "second copy" not in ctx.context_text
    assert ctx.source == "mycelium", ("the knowledge base gave nothing: the store alone", ctx.source, ctx.context_text)
    assert "scan.pdf" in ctx.said, "the turn still says the file it left to the knowledge base"
    monkeypatch.setattr(chat.turn, "passages", chat.turn.passages)
    project = _two_sources(chat, "rag", store_finds=False)
    ctx = chat.builder.build_context(project.id, "zanzibar_quartz plan")
    assert ctx.context_text.count("from the scan") == 1 and "zanzibar_quartz" not in ctx.context_text, ctx.context_text
    assert ctx.source == "rag", ("the store found nothing: the knowledge base alone", ctx.source)


def test_ld32_the_budget_decides_what_was_kept(chat, monkeypatch):
    """LD32, LD29's budget clause (review 266-2): the store's passages fit; the knowledge base's long chunk is cut to fit (kept: both) or
    left out by a budget with no room past the store's passages (the store alone)."""
    project = _two_sources(chat, "budget", long_rag=True)
    builder, pc = chat.builder, chat.pc
    probe = pc.ProjectContext(project_id=project.id)
    chunks = builder._project_chunks(probe, project.id, "zanzibar_quartz plan")
    held = [c for c in chunks if (c.metadata or {}).get(pc._SERVED_BY) != "rag"]
    assert held and len(held) < len(chunks), ("control: the store's passages, then the knowledge base's", chunks)
    header = builder._config.get("context_header", "--- Project Context ---")
    footer = builder._config.get("context_footer", "--- End Project Context ---")
    room = (sum(pc._estimate_tokens(c.content) for c in held) + pc._estimate_tokens(header + footer) + 20
            + pc._estimate_tokens(project.system_instructions or ""))
    cut = builder.build_context(project.id, "zanzibar_quartz plan", budget_tokens=room + 200)
    assert cut.chunks_used == len(held) + 1 and "from the scan" in cut.context_text, (cut.chunks_used, len(held))
    assert cut.source == "mycelium+rag", ("a chunk cut to fit was kept", cut.source)
    left = builder.build_context(project.id, "zanzibar_quartz plan", budget_tokens=room + 30)
    assert left.chunks_used == len(held) and "from the scan" not in left.context_text, (left.chunks_used, len(held))
    assert left.source == "mycelium", ("the budget left the knowledge base's chunk out", left.source)


def test_ld30_the_optimizers_path_says_the_projects_lines(chat):
    from opti_oignon.context_optimizer import ContextOptimizer

    project = _two_sources(chat, "optimizer")
    holder = types.SimpleNamespace(_project_builder=chat.builder, _estimate_tokens=lambda text, model: len(text) // 4)
    _text, zone = ContextOptimizer._inject_project_context(holder, project_id=project.id, query="zanzibar_quartz plan",
                                                           budget_tokens=4096, model="m")
    ctx = chat.builder.build_context(project.id, "zanzibar_quartz plan")
    assert zone.strategy == "mycelium+rag" and zone.passages == ctx.chunks_used >= 2, (zone.strategy, zone.passages)
    assert zone.said == ctx.said and "scan.pdf" in zone.said, zone.said
    assert "said" not in zone.as_dict() and "passages" not in zone.as_dict(), "the API's report keeps its shape"
    withheld = types.SimpleNamespace(**{**vars(zone), "passages": 0})
    unified = types.SimpleNamespace(**{**vars(zone), "strategy": "unified"})
    executor, restore = _load_executor()
    try:
        said = executor._project_zone_lines(types.SimpleNamespace(zones=[zone]))
        quiet = executor._project_zone_lines(types.SimpleNamespace(zones=[withheld]))
        none = executor._project_zone_lines(types.SimpleNamespace(zones=[unified]))
    finally:
        restore()
    assert said == [f"[OK] Project context injected: {zone.passages} chunks, ~{zone.actual_tokens} tokens, from the "
                    "project's Mycelium store and the knowledge base", f"[!] Project context: {zone.said}"], said
    assert quiet == [f"[!] Project context: {zone.said}"] and none == [], (quiet, none)
    out = _words(said)
    assert out["statuses"] == [f"Project context: {zone.passages} passages from its Mycelium store and the knowledge "
                               "base", "Project context: 1 file left to the knowledge base"], out["statuses"]
    written = (REPO / "opti_oignon" / "executor.py").read_text(encoding="utf-8")
    branch = written.split("opt_result = optimizer.optimize(", 1)[1].split("except Exception as e:", 1)[0]
    assert re.search(r'status\(\s*f"\[>\] Optimizer: .*?\n\s*\)\n\s*for line in _project_zone_lines\(rpt\):\n'
                     r"\s*status\(line\)", branch, re.S), "the optimizer's branch says them after its own line"
    optimizer = (REPO / "opti_oignon" / "context_optimizer.py").read_text(encoding="utf-8")
    assert re.search(r'" \| withheld: no wrapper"\n\s*project_zone\.passages, project_zone\.strategy = 0, "withheld"',
                     optimizer), "a withheld text says no passage and names no source"


def test_ld31_the_journals_fit_is_said():
    both = _context(context_text="[notes.md (lines 1-4)]\nzanzibar plan\n\n[scan.pdf]\nfrom the scan", chunks_used=2,
                    total_tokens_estimate=20, source="mycelium+rag",
                    said="the project holds 1 file(s) its Mycelium store does not (scan.pdf): those are left to the "
                         "knowledge base")
    executor, restore = _load_executor()
    try:
        whole = _executor_lines(executor, both, budget_tokens=4096, fit=lambda text: text)
        cut = _executor_lines(executor, both, budget_tokens=4096, fit=lambda text: text.split("\n\n[scan.pdf]")[0])
        gone = _executor_lines(executor, both, budget_tokens=4096, fit=lambda text: "")
    finally:
        restore()
    assert whole[0] == both.context_text and whole[1][1] == ("[OK] Project context injected: 2 chunks, ~20 tokens, from "
                                                             "the project's Mycelium store and the knowledge base"), whole
    assert cut[0] == both.context_text.split("\n\n[scan.pdf]")[0], ("the text handed on is the fitted one", cut[0])
    assert cut[1][1] == "[OK] Project context injected: 2 chunks, ~20 tokens, cut to its share of the window", cut[1]
    assert gone[0] == "" and not [line for line in gone[1] if line.startswith("[OK]")], gone[1]
    for _text, said in (whole, cut, gone):
        assert said[-1] == f"[!] Project context: {both.said}", ("what the turn says, last, whatever the fit", said)
    out = _words([cut[1][1]])
    assert out["statuses"] == ["Project context: 2 passages, cut to fit the window"], out["statuses"]
    written = (REPO / "opti_oignon" / "executor.py").read_text(encoding="utf-8")
    journal = written.split("_rag_fit = (", 1)[1].split("_onion_tail.append", 1)[0]
    composed, after = journal.split("_rag_text = self._compose_project_context(", 1)
    assert "self._onion_fit(" in composed and "fit=_rag_fit" in after, "the journal's path hands its fit to the composer"
    assert "_onion_fit(" not in after, "the text is fitted once, before its lines are said"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
