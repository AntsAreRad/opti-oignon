"""Contracts of the Mycelium as the agent's tool: ``mycelium_search`` and the card of its corpus.

  * MY1 -- THE TOOL IS THE AGENT'S, IN BOTH MODES: its schema (a query, k from 1 to 10), class none, in both modes'
    lists and tool sets; in Bulbe each call is asked, in Daily none; outside a run it says no workspace is bound.
  * MY2 -- PV10 of tests/test_provenance_gate_contracts.py, superseded: every agent and chat tool has a class from
    the closed set of six, the agent's tools now fourteen.
  * MY3 -- PV13 of tests/test_provenance_gate_contracts.py, superseded: the agent's allowlists are the class table
    read per mode, now fourteen in Daily and ten in Bulbe.
  * MY4 -- THE CORPUS OF THE RUN'S WORKSPACE: a run searches the sources rooted where its workspace was cloned from,
    never another source of their store (review 1), never another store, never a secret source; no workspace, an
    uploaded one, or a folder no store holds, is said with the owner's route.
  * MY5 -- PASSAGES AS THE RECIPE WIDENED THEM: each cited path:first-last, 20 lines at least around its part, those
    of one file that meet merged until none do, with the state of its lines; k held from 1 to 10; the answer bounded
    in characters and lines (MY11 for a passage too long); a query missing or a k that is no number said.
  * MY6 -- THE JUDGE'S VERDICT SAID: when the judge finds no passage answers, the answer says so first, and that the
    judge errs; it never tells the model to say so rather than guess; a passage judged yes comes first.
  * MY7 -- THE CARD AND THE RUN: the card names the corpus and its public and private sources, never a secret one;
    a run started for a workspace carries the card in its tools section and a handler of its own corpus.
  * MY8 -- THE EVALUATION NEVER READS THE OWNER'S CORPORA: the tool is neutralized on the eval path.
  * MY9 -- A STORE WIPED IS NO LONGER READ: a store the handler had open and the account wipe then removed is not
    searched again, and its connection is closed; one opened under another security mode is judged again by the
    platform's opener (review 1); indexed anew, it is searched.
  * MY10 -- SEARCHES AT ONCE: handlers of one store called from several threads at once each get their passages
    (review 1).
  * MY11 -- A PASSAGE TOO LONG IS CUT, NOT DROPPED: the answer gives its first lines and names the passages it had
    no room for, within the agent loop's lines; merges chain until none is left (review 1).
"""
import importlib
import sqlite3
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_provenance_gate_contracts as pv  # noqa: E402
from test_mycelium_place_contracts import GATE, _folder  # noqa: E402

BUDGET_S = {
    "test_my1_the_tool_is_the_agents_in_both_modes": 3.0,
    "test_my2_every_agent_and_chat_tool_has_a_class_from_the_closed_set_of_six_now_fourteen": 3.0,
    "test_my3_the_agent_allowlists_are_the_class_table_read_per_mode_now_fourteen_and_ten": 3.0,
    "test_my4_the_corpus_of_the_runs_workspace": 4.0,
    "test_my5_passages_as_the_recipe_widened_them": 4.0,
    "test_my6_the_judges_verdict_said": 4.0,
    "test_my7_the_card_and_the_run": 5.0,
    "test_my8_the_evaluation_never_reads_the_owners_corpora": 2.0,
    "test_my9_a_store_wiped_is_no_longer_read": 4.0,
    "test_my10_searches_at_once": 5.0,
    "test_my11_a_passage_too_long_is_cut_not_dropped": 4.0,
}

NOTES = "# Notes\n\nThe zanzibar inventory of the gate.\n"


def _quick(path, read_only=False):
    if read_only:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False, timeout=0.2)
    return sqlite3.connect(path, check_same_thread=False, timeout=0.2)


@pytest.fixture(autouse=True)
def no_engine(monkeypatch):
    """No contract reaches the machine's engine or governor."""
    from opti_oignon.mycelium import vectors

    def refused():
        raise RuntimeError("a contract asked the real inference registry")

    monkeypatch.setattr(vectors, "_registry", refused)
    monkeypatch.setattr(vectors, "_governor", lambda: None)


@pytest.fixture
def owned(tmp_path, monkeypatch):
    """The data folder a test folder, the store's opener a quick plain one, the machine's mode a fixed one, the
    tool's open stores and the workspace manifests emptied after the contract."""
    config = importlib.import_module("opti_oignon.config")
    from opti_oignon import sandbox_workspace
    from opti_oignon.mycelium import place, tool
    from opti_oignon.mycelium import store as store_mod

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(store_mod, "_safe_connect", _quick)
    monkeypatch.setattr(tool, "_mode", lambda: "daily")     # the machine's mode file is never read by a contract
    yield types.SimpleNamespace(place=place, tool=tool, data=tmp_path / "data")
    for held in list(tool._open.values()):
        held[-1].close()
    tool._open.clear()
    sandbox_workspace.reset_workspace_manifests()


class Sandbox:
    """A run's sandbox session as the tool reads it: an id and a manager that knows its workspace's path."""

    def __init__(self, session_id, workspace):
        self.session_id = session_id
        self.sandbox_manager = types.SimpleNamespace(
            get_workspace_path=lambda sid: workspace if sid == session_id else None)


def _cloned(tmp_path, name, root):
    """A sandbox whose workspace was cloned from ``root``, recorded as the copy-in records it."""
    from opti_oignon import sandbox_workspace

    workspace = str(tmp_path / f"ws-{name}")
    sandbox_workspace.get_workspace_manifests().record(name, {}, cloned_root=str(root), workspace=workspace)
    return Sandbox(name, workspace)


def _index(owned, corpus, root, source=None, sensitivity=None):
    from opti_oignon.mycelium.index import Indexer

    store = owned.place.open_to_write(corpus)
    try:
        Indexer(store).index(source or corpus, root, sensitivity=sensitivity)
    finally:
        store.close()


def test_my1_the_tool_is_the_agents_in_both_modes():
    """MY1 -- ``mycelium_search`` is one of the agent's schemas, not sandboxed, a required query and an optional k;
    class none; in both allowlists and both modes' tool sets, with a handler; in Bulbe the gate asks for each call
    (denied: refused), in Daily it asks nothing; the process-level handler says that no workspace is bound. A tool
    run unasked in Bulbe, or missing from a mode -> RED."""
    from opti_oignon import provenance
    from opti_oignon.agent import allowlists, tools

    [schema] = [s for s in tools.ALL_SCHEMAS if s.name == "mycelium_search"]
    assert not schema.sandboxed and schema.required_names() == ["query"]
    assert [p.name for p in schema.parameters] == ["query", "k"]
    assert provenance.TOOL_EFFECTS["mycelium_search"] == "none"
    assert "mycelium_search" in allowlists.DAILY_ALLOWLIST and "mycelium_search" in allowlists.BULBE_ALLOWLIST
    for mode in ("daily", "bulbe"):
        tool_set = tools.build_tool_set(mode)
        assert "mycelium_search" in tool_set.names and "mycelium_search" in tool_set.tool_handlers, mode
    asked = []

    def deny(conversation_id, name, arguments):
        asked.append(name)
        return False

    bulbe = allowlists.evaluate("mycelium_search", {"query": "x"}, mode="bulbe", approval_fn=deny,
                                conversation_id="c")
    assert not bulbe.allowed and asked == ["mycelium_search"], (bulbe, asked)
    daily = allowlists.evaluate("mycelium_search", {"query": "x"}, mode="daily", approval_fn=deny,
                                conversation_id="c")
    assert daily.allowed and asked == ["mycelium_search"], (daily, asked)
    said = tools.get_tool_registry().handler("mycelium_search")({"query": "the gate"})
    assert said.startswith("No workspace is bound to this run"), said


def test_my2_every_agent_and_chat_tool_has_a_class_from_the_closed_set_of_six_now_fourteen():
    """MY2 -- PV10, superseded (deselected by name in the canonical selection rule): the same census, the agent's
    tools fourteen with the Mycelium's search. A tool with no class from the closed set -> RED."""
    loaded, restore = pv._agent_window()
    try:
        prov, tools = loaded[pv._PROV], loaded[pv._TOOLS]
        effects, table = prov.EFFECTS, dict(prov.TOOL_EFFECTS)
        agent = [schema.name for schema in tools.ALL_SCHEMAS]
    finally:
        restore()
    chat = sorted(set(pv._declared_chat_tool_names()))
    assert effects == ("none", "session", "sandbox", "deferred", "network", "approved"), effects
    assert len(agent) == 14 and len(chat) >= 9, f"the census of tools is too small to be trusted: {agent} {chat}"
    unclassed = sorted({name for name in agent + chat if table.get(name) not in effects})
    assert unclassed == [], f"tools with no class from the closed set: {unclassed}"
    strangers = sorted(set(table) - set(agent) - set(chat))
    assert strangers == [], f"the table names tools nothing declares: {strangers}"


def test_my3_the_agent_allowlists_are_the_class_table_read_per_mode_now_fourteen_and_ten():
    """MY3 -- PV13, superseded (deselected by name in the canonical selection rule): the allowlists are the class
    table read per mode, fourteen tools in Daily and ten in Bulbe with the Mycelium's search; a mode the table does
    not know is read as Bulbe. An allowlist apart from the table -> RED."""
    loaded, restore = pv._agent_window()
    try:
        prov, tools, allow = loaded[pv._PROV], loaded[pv._TOOLS], loaded[pv._ALLOW]
        names = [schema.name for schema in tools.ALL_SCHEMAS]
        daily = {n for n in names if prov.permitted(prov.TOOL_EFFECTS[n], "daily")}
        bulbe = {n for n in names if prov.permitted(prov.TOOL_EFFECTS[n], "bulbe")}
        lists = (set(allow.DAILY_ALLOWLIST), set(allow.BULBE_ALLOWLIST))
        unknown_mode = {n for n in names if prov.permitted(prov.TOOL_EFFECTS[n], "fortress")}
    finally:
        restore()
    assert (daily, bulbe) == lists, (daily ^ lists[0], bulbe ^ lists[1])
    assert (len(daily), len(bulbe)) == (14, 10), (len(daily), len(bulbe))
    assert unknown_mode == bulbe


def test_my4_the_corpus_of_the_runs_workspace(owned, tmp_path):
    """MY4 -- two corpora: oo holds the folder a workspace was cloned from and a secret source; zz holds another
    folder. The run's handler finds the gate in oo; it never gives zz's words nor the secret source's. A run with no
    workspace says none is bound; a workspace cloned from a folder no store holds names it and the route. A search of
    another store, or of a secret source -> RED."""
    root = _folder(tmp_path / "src", {"gate.py": GATE, "notes.md": NOTES})
    other = _folder(tmp_path / "other", {"far.md": "# Far\n\nThe quokka ledger.\n"})
    vault = _folder(tmp_path / "vault", {"keys.md": "# Keys\n\nThe wombat plan.\n"})
    _index(owned, "oo", root)
    _index(owned, "oo", vault, source="vault", sensitivity="secret")
    _index(owned, "zz", other)
    handle = owned.tool.make_handler(_cloned(tmp_path, "s1", root))
    found = handle({"query": "check_entity", "k": 3})
    assert "[gate.py:" in found and "(fresh)" in found, found
    assert "quokka" not in handle({"query": "quokka ledger"})
    assert "wombat" not in handle({"query": "wombat plan"})
    assert owned.tool.make_handler(None)({"query": "gate"}).startswith("No workspace is bound")
    lone = _folder(tmp_path / "lone", {"a.md": NOTES})
    said = owned.tool.make_handler(_cloned(tmp_path, "s2", lone))({"query": "gate"})
    assert str(lone) in said and "mycelium index" in said, said
    notes = _folder(tmp_path / "home-notes", {"diary.md": "# Diary\n\nThe quokka ledger of my bank.\n"})
    _index(owned, "oo", notes, source="notes")
    assert "diary" not in handle({"query": "quokka ledger bank"}), "a source of the store rooted elsewhere was read"
    _index(owned, "oo", root, source="again")
    assert handle({"query": "check_entity", "k": 3}).count("[gate.py:") == 1, "one folder under two sources twice"
    from opti_oignon.mycelium.store import StoreError

    real_open = owned.place.open_to_read

    def refusing(corpus):
        raise StoreError("the store is in plaintext and the machine is in Bulbe: refused")

    owned.place.open_to_read = refusing
    real_mode = owned.tool._mode
    owned.tool._mode = lambda: "another mode"
    try:
        said = handle({"query": "check_entity"})
    finally:
        owned.place.open_to_read = real_open
        owned.tool._mode = real_mode
    assert "could not be opened" in said and "refused" in said and "mycelium index" not in said, said
    hidden = _folder(tmp_path / "hidden", {"keys.md": "# Keys\n\nThe wombat plan.\n"})
    _index(owned, "sec", hidden, sensitivity="secret")
    said = owned.tool.make_handler(_cloned(tmp_path, "s4", hidden))({"query": "wombat plan"})
    assert "wombat" not in said and "No Mycelium corpus is indexed" in said, "a secret source was searched"
    from opti_oignon import sandbox_workspace

    uploaded = Sandbox("s3", str(tmp_path / "ws-s3"))
    sandbox_workspace.get_workspace_manifests().record("s3", {"a.md": "0" * 64}, workspace=str(tmp_path / "ws-s3"))
    said = owned.tool.make_handler(uploaded)({"query": "gate"})
    assert said.startswith("This run's workspace was not cloned"), said


def _long_module():
    """A module of twenty functions of seven lines each -- parts of their own, too long to be merged and shorter than
    a passage --, the eleventh naming a rare word."""
    blocks = ['"""A long module."""\n']
    for i in range(20):
        word = "zanzibarish" if i == 10 else f"plain{i}"
        blocks.append(f"\n\ndef step{i}(x):\n    \"\"\"Step {i}: {word}.\"\"\"\n    a = x + {i}\n    b = a * 2\n"
                      f"    c = b - 1\n    d = c + a\n    return d\n")
    return "".join(blocks)


def test_my5_passages_as_the_recipe_widened_them(owned, tmp_path, monkeypatch):
    """MY5 -- a long module: the part that names the rare word comes cited path:first-last over 20 lines at least,
    around it; two parts of one file found together come as one passage; k 0 gives one passage, 50 at most ten, and
    a k that is text is said; no query is said; the answer stays within its bound, said; a file changed since is
    'modified', a file removed 'gone'. A passage cut to its part, or an unbounded answer -> RED."""
    import re

    many = {f"walk{i:02d}.py": f'"""Walker {i}."""\n\n\ndef walk{i:02d}(x):\n    """Walk the steps {i}."""\n'
                                f"    return x + {i}\n" for i in range(14)}
    root = _folder(tmp_path / "src", {"long.py": _long_module(), "gate.py": GATE, **many})
    _index(owned, "oo", root)
    handle = owned.tool.make_handler(_cloned(tmp_path, "s1", root))
    found = handle({"query": "zanzibarish", "k": 1})
    [(path, first, last)] = re.findall(r"\[(long\.py):(\d+)-(\d+)\]", found)
    first, last = int(first), int(last)
    rare = _long_module().split("\n").index('    """Step 10: zanzibarish."""') + 1
    assert last - first + 1 >= 20 and first <= rare <= last, (first, last, rare)
    both = handle({"query": "step9 step10", "k": 2})
    assert len(re.findall(r"\[long\.py:", both)) == 1, both
    assert len(re.findall(r"^\[", handle({"query": "walk the steps", "k": 0}), re.M)) == 1
    assert len(re.findall(r"^\[", handle({"query": "walk the steps", "k": 50}), re.M)) == 10
    assert "k is a whole number" in handle({"query": "step", "k": "3"})
    assert "needs a query" in handle({})
    monkeypatch.setattr(owned.tool, "MAX_BYTES", 4000)
    bounded = handle({"query": "walk the steps", "k": 10})
    assert len(bounded.encode("utf-8")) <= 4000 and "the answer's bound" in bounded, len(bounded.encode("utf-8"))
    (root / "gate.py").write_text(GATE.replace("Check one", "Changed"), encoding="utf-8")
    assert "(modified)" in handle({"query": "check_entity", "k": 1})
    (root / "gate.py").unlink()
    assert "(gone)" in handle({"query": "check_entity", "k": 1})


def test_my11_a_passage_too_long_is_cut_not_dropped(owned, tmp_path):
    """MY11 (review 1) -- a section of thirty long lines, past the answer's bound alone, and a small note: the answer
    gives the section's first lines, its citation saying which, and names the note it had no room for; a function of
    four hundred lines gives lines within the agent loop's bound, never an answer without text. Merges chain: three
    parts of one file, the third bridging the first two, give one passage. An answer emptied by one long passage, or
    past the loop's lines -> RED."""
    import re

    long_lines = "\n".join(f"Line {i} " + "word " * 100 for i in range(30))
    big = "def big(x):\n" + "".join(f"    x = x + {i}\n" for i in range(400)) + "    return x\n"
    root = _folder(tmp_path / "src", {"guide.md": "# Guide\n\n" + long_lines + "\n", "small.md": NOTES,
                                       "big.py": big})
    _index(owned, "oo", root)
    handle = owned.tool.make_handler(_cloned(tmp_path, "s1", root))
    said = handle({"query": "guide word zanzibar inventory", "k": 2})
    assert "Line 0 word" in said and "the answer's bound" in said, said[:300]
    assert "small.md" in said, said[-300:]
    said = handle({"query": "big", "k": 10})
    assert "x = x + 0" in said and said.count("\n") <= owned.tool.MAX_LINES + 10, said.count("\n")
    from opti_oignon.mycelium.search import Searcher

    store = owned.tool.corpus_for(owned.tool.workspace_root(_cloned(tmp_path, "s2", root)))[1]
    searcher = Searcher(store)
    hit = next(h for h in searcher.search("big return x", 10, stages=("lexical",)) if h["path"] == "big.py")
    spans = owned.tool.widened(searcher, [dict(hit, first=1, last=20), dict(hit, first=41, last=60),
                                          dict(hit, first=15, last=45)])
    assert [(s["first"], s["last"]) for s in spans] == [(1, 60)], [(s["first"], s["last"]) for s in spans]
    assert re.search(r"\[guide\.md:\d+-\d+\]", handle({"query": "guide word", "k": 1}))
    # Review 2: one line longer than the bound, then a small note; bytes, not characters; lines in all; a file that
    # shrank since.
    wide = _folder(tmp_path / "wide", {"wide.py": "WIDE = '" + "zanzibar " * 3000 + "'\n", "small.md": NOTES,
                                       "box.md": "# Box\n\n" + "\n".join(chr(0x2500) * 300 for _ in range(60)) + "\n",
                                       "mod.py": "def mod(x):\n" + "".join(f"    x = x + {i}\n" for i in range(30))
                                       + "    return x\n"})
    _index(owned, "wide", wide)
    wide_handle = owned.tool.make_handler(_cloned(tmp_path, "s9", wide))
    said = wide_handle({"query": "zanzibar inventory", "k": 2})
    assert "zanzibar inventory of the gate" in said, said[:400]
    assert len(said.encode("utf-8")) <= owned.tool.MAX_BYTES
    real_bytes = owned.tool.MAX_BYTES
    owned.tool.MAX_BYTES = 3400         # room for the note's lines, not for the wide line's first
    try:
        said = wide_handle({"query": "zanzibar", "k": 2})       # the wide line first, the note second
    finally:
        owned.tool.MAX_BYTES = real_bytes
    assert "zanzibar inventory of the gate" in said and "wide.py" in said, said[:600]
    said = wide_handle({"query": "Box", "k": 1})
    assert "[box.md:" in said and chr(0x2500) in said, said[:300]
    assert len(said.encode("utf-8")) <= owned.tool.MAX_BYTES and said.count("\n") + 1 <= owned.tool.MAX_LINES
    said = handle({"query": "big return x walk", "k": 10})
    assert said.count("\n") + 1 <= owned.tool.MAX_LINES, said.count("\n")
    (wide / "mod.py").write_text("def mod(x):\n" + "".join(f"    x = x + {i}\n" for i in range(18)) + "    return x\n",
                                 encoding="utf-8")
    cited = re.findall(r"\[mod\.py:(\d+)-(\d+)\]", wide_handle({"query": "mod", "k": 1}))
    held = len((wide / "mod.py").read_text(encoding="utf-8").split("\n"))     # the lines as the index reads them
    assert cited and all(int(last) <= held for _first, last in cited), (cited, held)


class Judge:
    """The reranker's model as a stand-in: every passage gets ``word``."""

    def __init__(self, word):
        self.word = word

    def ask(self, prompt, num_predict, system=None, timeout_s=None):
        return self.word


def _judged(owned, monkeypatch, word):
    from opti_oignon.mycelium.rerank import Reranker, RerankSettings
    from opti_oignon.mycelium.search import Searcher

    def searcher(store, names):
        return Searcher(store, allow=owned.tool.ALLOW, sources=names,
                        reranker=Reranker(RerankSettings("judge", 20, 2, 3000, 5.0), asker=Judge(word)))

    monkeypatch.setattr(owned.tool, "_searcher", searcher)


def test_my6_the_judges_verdict_said(owned, tmp_path, monkeypatch):
    """MY6 -- the judge says no to every passage: the answer opens with the judge's verdict and its error, then the
    closest passages; it never says 'say so rather than guess'. The judge says yes: no such verdict. A verdict of
    none hidden, or the old order to the model -> RED."""
    root = _folder(tmp_path / "src", {"gate.py": GATE, "notes.md": NOTES})
    _index(owned, "oo", root)
    _judged(owned, monkeypatch, "no")
    handle = owned.tool.make_handler(_cloned(tmp_path, "s1", root))
    said = handle({"query": "check_entity", "k": 2})
    assert said.startswith(owned.tool.JUDGED_NONE) and "[gate.py:" in said, said
    assert "say so rather than guess" not in said.lower()
    _judged(owned, monkeypatch, "yes")
    said = handle({"query": "check_entity", "k": 2})
    assert not said.startswith(owned.tool.JUDGED_NONE) and "[gate.py:" in said, said


def test_my7_the_card_and_the_run(owned, tmp_path, monkeypatch):
    """MY7 -- the card of a workspace's corpus names it and its public and private sources with their files, never
    the secret one; a workspace no store holds is said. A run started for that workspace (its loop a recorder) is
    handed the card in its tools section and a handler that searches its corpus. A card naming a secret source, or a
    run left with the process-level handler -> RED."""
    root = _folder(tmp_path / "src", {"gate.py": GATE, "notes.md": NOTES})
    vault = _folder(tmp_path / "vault", {"keys.md": "# Keys\n\nThe wombat plan.\n"})
    _index(owned, "oo", root)
    _index(owned, "oo", vault, source="vault", sensitivity="secret")
    sandbox = _cloned(tmp_path, "s1", root)
    card = owned.tool.card(sandbox)
    assert "`oo`" in card and "- oo (folder, 2 files" in card and "vault" not in card, card
    lone = _folder(tmp_path / "lone", {"a.md": NOTES})
    assert "No Mycelium corpus is indexed" in owned.tool.card(_cloned(tmp_path, "s2", lone))

    from opti_oignon.api import routes_agent

    calls = []

    def loop(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(stop_reason="done", rounds=0, final="", events=[])

    monkeypatch.setattr(routes_agent.agent_loop, "run", loop)
    manager = routes_agent.AgentRunManager()
    started = manager.start("find the gate", model_client=object(), mode="daily", sandbox=sandbox,
                            include_memory=False, consult=False)
    assert started.get("started") is True, started
    manager.join(timeout=10.0)
    [kwargs] = calls
    assert card in kwargs["system_prompt"], kwargs["system_prompt"][-400:]
    handler = kwargs["tool_handlers"]["mycelium_search"]
    assert "[gate.py:" in handler({"query": "check_entity", "k": 1})


def test_my8_the_evaluation_never_reads_the_owners_corpora():
    """MY8 -- the agent's evaluation neutralizes the Mycelium's search with the tools that write the user's state
    and the network. An evaluation that reads the owner's corpora -> RED."""
    from opti_oignon.agent_eval import runner

    assert "mycelium_search" in runner._DISABLED_TOOLS


def test_my9_a_store_wiped_is_no_longer_read(owned, tmp_path, monkeypatch):
    """MY9 -- the handler searched a corpus (its store kept open); the account wipe then removes every store: the
    next search finds no corpus -- never the removed file's passages through the connection still open. Indexed
    again, the corpus is searched again. A wiped store still read -> RED."""
    root = _folder(tmp_path / "src", {"gate.py": GATE})
    _index(owned, "oo", root)
    handle = owned.tool.make_handler(_cloned(tmp_path, "s1", root))
    assert "[gate.py:" in handle({"query": "check_entity"})
    owned.place.remove_every_store()
    said = handle({"query": "check_entity"})
    assert "[gate.py:" not in said and "No Mycelium corpus is indexed" in said, said
    assert owned.tool._open == {}, "a wiped store's connection was kept open"
    _index(owned, "oo", root)
    assert "[gate.py:" in handle({"query": "check_entity"})
    from opti_oignon.mycelium.store import StoreError

    def refused(corpus):
        raise StoreError("the store is in plaintext and the machine is in Bulbe: refused")

    monkeypatch.setattr(owned.place, "open_to_read", refused)
    monkeypatch.setattr(owned.tool, "_mode", lambda: "a mode the store was not opened under")
    said = handle({"query": "check_entity"})
    assert "[gate.py:" not in said, "a store opened under another mode was read without its opener"



def test_my10_searches_at_once(owned, tmp_path):
    """MY10 (review 1) -- eight threads, each calling the handler of one store ten times at once: every answer holds
    its passage. Searches that mix their statements on the store's one connection -> RED."""
    import threading

    root = _folder(tmp_path / "src", {"gate.py": GATE, "notes.md": NOTES})
    _index(owned, "oo", root)
    handle = owned.tool.make_handler(_cloned(tmp_path, "s1", root))
    answers = []

    def run():
        for _ in range(10):
            answers.append(handle({"query": "check_entity", "k": 2}))

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    bad = [a for a in answers if "[gate.py:" not in a]
    assert len(answers) == 80 and bad == [], (len(answers), bad[:2])
    # Review 2: the mode changes while a search is inside its judge; the store is closed only once the search ends.
    import time

    from opti_oignon.mycelium.rerank import Reranker, RerankSettings
    from opti_oignon.mycelium.search import Searcher

    inside, release = threading.Event(), threading.Event()

    class SlowJudge:
        def ask(self, prompt, num_predict, system=None, timeout_s=None):
            inside.set()
            release.wait(5)
            return "yes"

    def searcher(store, names):
        return Searcher(store, allow=owned.tool.ALLOW, sources=names,
                        reranker=Reranker(RerankSettings("judge", 20, 1, 3000, 10.0), asker=SlowJudge()))

    owned.tool._searcher, real_searcher = searcher, owned.tool._searcher
    slow = []
    worker = threading.Thread(target=lambda: slow.append(handle({"query": "check_entity", "k": 1})))
    try:
        worker.start()
        assert inside.wait(5)
        real_mode = owned.tool._mode
        owned.tool._mode = lambda: "a mode changed meanwhile"
        closer = threading.Thread(target=lambda: owned.tool.corpus_for(owned.tool.workspace_root(
            _cloned(tmp_path, "s7", root))))
        closer.start()
        time.sleep(0.2)
        release.set()
        worker.join(10)
        closer.join(10)
        owned.tool._mode = real_mode
    finally:
        owned.tool._searcher = real_searcher
    assert slow and "[gate.py:" in slow[0], slow
