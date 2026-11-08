"""Contracts of the Mycelium's map (P20, phase 2, stage 1, M8): a corpus walked as the tree of its folders and files,
each node said in a few lines drawn from the store alone -- no model writes any of it -- and the agent's tool that
walks it, ``mycelium_map``, beside ``mycelium_search``.

  * MA1 -- A FOLDER: its sub-folders and its files, biggest first, with how many files, parts and which languages, a
    file's first sentence; nothing indexed there, said; a folder whose name holds a LIKE wildcard, or differs from
    another only by case, lists its own files only.
  * MA2 -- A FILE: its parts with their lines, what it defines, the settings it reads (only names that are surely
    settings), the tests, the docs and the other files that name what it defines, each with how many parts do.
  * MA3 -- FROM THE STORE ALONE, ONE SOURCE: the map changes when the corpus is indexed again, and nothing is written;
    a file's users, settings and the counts that bound them are its own source's -- another source of the same store,
    a secret one included, changes nothing of what is said; a source the store does not hold, or holds as secret, is
    refused by name.
  * MA4 -- THE TOOL: a run walks the corpus of its workspace, one of its own public or private sources (named when it
    has several; a secret source of the store, or one rooted elsewhere, refused by name), held to the search's bounds
    (lines, bytes, a line too long cut); no workspace or an uploaded one said; a path that is not text, or too long,
    said; a failure said, never raised; outside a run, no corpus.
  * MA5 -- THE TOOL DECLARED: its schema (path, source, both optional), class none, a read tool in both modes, on the
    eval path neutralized like the search; the corpus card names it, and says it is not something to cite;
    SECURITY.md and the CHANGELOG say it.
  * MA6 -- MY2 of tests/test_mycelium_agent_contracts.py, superseded: every agent and chat tool has a class from the
    closed set of six, the agent's tools now fifteen.
  * MA7 -- MY3 there, superseded: the agent's allowlists are the class table read per mode, now fifteen in Daily and
    eleven in Bulbe.
  * MA8 -- A RUN THAT WALKED ITS MAP IS CHECKED (review 26x-1): the map names a file's parts and their lines; a run that
    walked it and no search ends with its answer's citations checked all the same -- a line the map named but no
    search gave said unseen --; a run on the same manager that walks nothing is left as the loop gave it.
MA1-MA5 run the real package on corpora the suite writes and indexes (the store's opener a quick plain one, the data
folder the test's own), as the Mycelium agent suite does; MA6 and MA7 load the agent's tables on the provenance gate
suite's window, as MY2 and MY3 did; MA8 runs the agent route's manager with a stand-in loop, as the citation gate
suite does. No engine is reached.
"""
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_citation_gate_contracts as ci  # noqa: E402
import test_mycelium_agent_contracts as my_t  # noqa: E402
import test_provenance_gate_contracts as pv  # noqa: E402

# The Mycelium agent suite's fixtures: no engine reached; the data folder, the store's opener and the mode the test's.
no_engine = my_t.no_engine
owned = my_t.owned

BUDGET_S = {
    "test_ma1_a_folder": 4.0,
    "test_ma2_a_file": 4.0,
    "test_ma3_from_the_store_alone_one_source": 6.0,
    "test_ma4_the_tool": 6.0,
    "test_ma5_the_tool_declared": 3.0,
    "test_ma6_every_agent_and_chat_tool_has_a_class_from_the_closed_set_of_six_now_fifteen": 2.0,
    "test_ma7_the_agent_allowlists_are_the_class_table_read_per_mode_now_fifteen_and_eleven": 2.0,
    "test_ma8_a_run_that_walked_its_map_is_checked": 5.0,
}


def _corpus(root: Path) -> Path:
    """A small project: a package of three modules (one in a sub-folder), a setting file, a test, a doc."""
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text('"""Computes the frobnication of widgets."""\n\n\n'
                                       "def frobnicate(settings):\n"
                                       '    return settings.get("call_timeout_s", 5) * 2\n\n\n'
                                       "def helper():\n    return 1\n")
    (root / "pkg" / "b.py").write_text('"""Uses the frobnication."""\nfrom pkg.a import frobnicate\n\n\n'
                                       "def run():\n    return frobnicate({})\n")
    (root / "pkg" / "sub" / "c.py").write_text('"""A leaf."""\n\n\ndef leaf():\n    return 0\n')
    (root / "config").mkdir()
    (root / "config" / "app.yaml").write_text("call_timeout_s: 5\ncode: x\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_a.py").write_text("from pkg.a import frobnicate\n\n\ndef test_frobnicate():\n"
                                              "    assert frobnicate({}) == 10\n")
    (root / "docs").mkdir()
    (root / "docs" / "guide.md").write_text("# The guide\n\nCall `frobnicate` to frobnicate.\n")
    return root


def _store(owned, corpus):
    store = owned.place.open_to_read(corpus)
    return store


def test_ma1_a_folder(owned, tmp_path):
    from opti_oignon.mycelium import map as mp

    root = _corpus(tmp_path / "proj")
    (root / "a_b").mkdir()
    (root / "a_b" / "in_ab.py").write_text('"""In a_b."""\n')
    (root / "aXb").mkdir()
    (root / "aXb" / "in_axb.py").write_text('"""In aXb."""\n')
    (root / "Docs").mkdir()
    (root / "Docs" / "Upper.md").write_text("# Upper\n")
    my_t._index(owned, "proj", root)
    store = _store(owned, "proj")
    try:
        subs, files = mp.folder(store, "proj", "")
        names = [s[0] for s in subs]
        assert names[0] == "pkg" and set(names) == {"pkg", "config", "tests", "docs", "Docs", "a_b", "aXb"}, subs
        assert subs[0][1] == 3 and "python 3" in subs[0][3], subs[0]
        said = mp.branch(store, "proj", "pkg")
        assert said.splitlines()[0] == "proj:pkg", said
        assert "sub/  1 files" in said and "a.py" in said and "Computes the frobnication of widgets." in said, said
        assert mp.branch(store, "proj", "nowhere") == "proj:nowhere: nothing indexed here."
        assert [f[0] for f in mp.folder(store, "proj", "a_b")[1]] == ["in_ab.py"], "a LIKE wildcard in a folder's name"
        assert [f[0] for f in mp.folder(store, "proj", "docs")[1]] == ["guide.md"], "a folder of another case"
        assert [f[0] for f in mp.folder(store, "proj", "Docs")[1]] == ["Upper.md"], "a folder of another case"
    finally:
        store.close()


def test_ma2_a_file(owned, tmp_path):
    from opti_oignon.mycelium import map as mp

    root = _corpus(tmp_path / "proj")
    my_t._index(owned, "proj", root)
    store = _store(owned, "proj")
    try:
        card = mp.file_card(store, "proj", "pkg/a.py")
        assert card["sentence"] == "Computes the frobnication of widgets.", card
        assert {"frobnicate", "helper"} <= set(card["defines"]), card["defines"]
        assert card["reads_settings"] == ["call_timeout_s"], card["reads_settings"]
        assert [p for p, _n in card["tested_by"]] == ["tests/test_a.py"], card
        assert [p for p, _n in card["documented_by"]] == ["docs/guide.md"], card
        assert [p for p, _n in card["used_by"]] == ["pkg/b.py"], card
        assert card["parts"] and card["parts"][0][2] == 1 and card["parts"][-1][3] >= 9, card["parts"]
        said = mp.branch(store, "proj", "pkg/a.py")
        for line in ("pkg/a.py: Computes the frobnication of widgets.", "reads settings: call_timeout_s",
                     "tested by: tests/test_a.py (1)", "documented by: docs/guide.md (1)", "used by: pkg/b.py (1)"):
            assert line in said, (line, said)
        assert mp.file_card(store, "proj", "pkg/none.py") is None
    finally:
        store.close()


def test_ma3_from_the_store_alone_one_source(owned, tmp_path):
    from opti_oignon.mycelium import map as mp

    root = _corpus(tmp_path / "proj")
    other = tmp_path / "other"
    other.mkdir()
    (other / "uses.py").write_text("from pkg.a import frobnicate\n\n\ndef again():\n    return frobnicate({})\n")
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "keys.yaml").write_text("vault_api_endpoint: x\n")
    for n in range(6):          # six definitions of a name the run's file defines, one a file
        (vault / f"m{n}.py").write_text(f"def frobnicate():\n    return {n}\n")
    (vault / "uses.py").write_text('def reads():\n    return {}.get("vault_api_endpoint")\n')
    (root / "pkg" / "a.py").write_text((root / "pkg" / "a.py").read_text()
                                       + '\n\ndef endpoint(s):\n    return s.get("vault_api_endpoint")\n')
    my_t._index(owned, "proj", root, source="proj")
    my_t._index(owned, "proj", other, source="elsewhere")
    my_t._index(owned, "proj", vault, source="vault", sensitivity="secret")
    path = owned.place.path_for("proj")
    before = (path.stat().st_mtime_ns, path.stat().st_size)
    store = _store(owned, "proj")
    try:
        card = mp.file_card(store, "proj", "pkg/a.py")
        assert [p for p, _n in card["used_by"]] == ["pkg/b.py"], "a file of another source of the store named"
        assert card["reads_settings"] == ["call_timeout_s"], ("a setting of another source said", card)
        with pytest.raises(mp.NoSource):
            mp.branch(store, "vault", "")
        with pytest.raises(mp.NoSource):
            mp.branch(store, "nobody", "")
    finally:
        store.close()
    assert (path.stat().st_mtime_ns, path.stat().st_size) == before, "the map wrote to the store"
    (root / "pkg" / "d.py").write_text('"""A new one."""\n')
    my_t._index(owned, "proj", root, source="proj")
    store = _store(owned, "proj")
    try:
        assert "d.py" in mp.branch(store, "proj", "pkg")
    finally:
        store.close()


def test_ma4_the_tool(owned, tmp_path, monkeypatch):
    root = _corpus(tmp_path / "proj")
    my_t._index(owned, "proj", root)
    run = my_t._cloned(tmp_path, "run1", os.path.realpath(root))
    walk = owned.tool.make_map_handler(run)
    said = walk({"path": "pkg"})
    assert said.startswith("proj:pkg") and "a.py" in said, said
    assert walk({"path": "pkg/a.py", "source": "proj"}).startswith("pkg/a.py: Computes")
    assert walk({"path": "", "source": "nobody"}) == 'mycelium_map: no source named "nobody" among this run\'s (proj).'
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "keys.yaml").write_text("vault_api_endpoint: x\n")
    my_t._index(owned, "proj", vault, source="vault", sensitivity="secret")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "z.py").write_text('"""Z."""\n')
    my_t._index(owned, "proj", elsewhere, source="elsewhere")
    for name in ("vault", "elsewhere"):
        assert walk({"path": "", "source": name}) == (f'mycelium_map: no source named "{name}" among this run\'s '
                                                      "(proj)."), name
    assert walk({"path": 12}).startswith("mycelium_map: path is a folder or a file")
    assert walk({"path": "x" * 1001}).startswith("mycelium_map: path is a folder or a file")
    assert walk({"source": 3}).startswith("mycelium_map: source is the name")
    # Two sources: the source must be named.
    my_t._index(owned, "proj", os.path.realpath(root), source="twin")
    assert walk({"path": ""}).startswith("mycelium_map needs a source: this run's are "), walk({"path": ""})
    # Held to the search's bounds.
    lines = owned.tool.MAX_LINES
    owned.tool.MAX_LINES = 4
    try:
        bounded = walk({"path": "", "source": "proj"})
    finally:
        owned.tool.MAX_LINES = lines
    assert len(bounded.splitlines()) <= 4 and bounded.splitlines()[-1].endswith("more lines not shown"), bounded
    held_bytes, held_chars = owned.tool.MAX_BYTES, owned.tool.MAX_LINE_CHARS
    owned.tool.MAX_BYTES, owned.tool.MAX_LINE_CHARS = 150, 30
    try:
        small = walk({"path": "pkg", "source": "proj"})
    finally:
        owned.tool.MAX_BYTES, owned.tool.MAX_LINE_CHARS = held_bytes, held_chars
    assert len(small.encode()) <= 150 and small.splitlines()[-1].endswith("more lines not shown"), small
    assert any(line.endswith(" [line cut]") for line in small.splitlines()), small
    # No workspace, an uploaded one, outside a run, a failure: said, never raised.
    assert owned.tool.make_map_handler(None)({"path": ""}) == owned.tool.NO_WORKSPACE
    from opti_oignon import sandbox_workspace

    uploaded = my_t.Sandbox("s3", str(tmp_path / "ws-s3"))
    sandbox_workspace.get_workspace_manifests().record("s3", {"a.md": "0" * 64}, workspace=str(tmp_path / "ws-s3"))
    assert owned.tool.make_map_handler(uploaded)({"path": ""}) == owned.tool.UPLOADED
    from opti_oignon.agent import tools as agent_tools

    assert agent_tools._unbound_mycelium_map({"path": ""}) == owned.tool.NO_WORKSPACE

    def broken(*args):
        raise RuntimeError("the index cannot be read")

    from opti_oignon.mycelium import map as mp

    monkeypatch.setattr(mp, "branch", broken)
    assert walk({"path": "", "source": "proj"}) == "The Mycelium could not walk its map this time (RuntimeError)."


def test_ma5_the_tool_declared(owned, tmp_path):
    from opti_oignon import provenance
    from opti_oignon.agent import allowlists
    from opti_oignon.agent import tools as agent_tools
    from opti_oignon.agent_eval import runner

    schema = next(s for s in agent_tools.ALL_SCHEMAS if s.name == "mycelium_map")
    assert [(p.name, p.required) for p in schema.parameters] == [("path", False), ("source", False)]
    assert schema.sandboxed is False and "mycelium_map" in agent_tools.HANDLER_TOOL_NAMES
    assert provenance.TOOL_EFFECTS["mycelium_map"] == provenance.NONE
    assert "mycelium_map" in allowlists.READ_TOOLS and "mycelium_map" in runner._DISABLED_TOOLS
    root = _corpus(tmp_path / "proj")
    my_t._index(owned, "proj", root)
    run = my_t._cloned(tmp_path, "run1", os.path.realpath(root))
    card = owned.tool.card(run)
    assert "mycelium_map walks the same corpus's map without reading any file" in card, card
    assert "not something to cite" in card, card
    repo = Path(__file__).resolve().parents[1]
    security = " ".join((repo / "SECURITY.md").read_text(encoding="utf-8").split())
    assert "Its map, `mycelium_map`, reads the same sources the same way, from the store's index alone" in security
    changelog = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    added = changelog.split("### Added\n", 1)[1].split("\n### ", 1)[0]
    entry = " ".join(next((e for e in added.split("\n- ") if "MA1-MA8" in e), "").split())
    assert "mycelium_map(path?, source?)" in entry, entry[:200]


def test_ma6_every_agent_and_chat_tool_has_a_class_from_the_closed_set_of_six_now_fifteen():
    loaded, restore = pv._agent_window()
    try:
        prov, tools = loaded[pv._PROV], loaded[pv._TOOLS]
        effects, table = prov.EFFECTS, dict(prov.TOOL_EFFECTS)
        agent = [schema.name for schema in tools.ALL_SCHEMAS]
    finally:
        restore()
    chat = sorted(set(pv._declared_chat_tool_names()))
    assert effects == ("none", "session", "sandbox", "deferred", "network", "approved"), effects
    assert len(agent) == 15 and len(chat) >= 9, f"the census of tools is too small to be trusted: {agent} {chat}"
    unclassed = sorted({name for name in agent + chat if table.get(name) not in effects})
    assert unclassed == [], f"tools with no class from the closed set: {unclassed}"
    strangers = sorted(set(table) - set(agent) - set(chat))
    assert strangers == [], f"the table names tools nothing declares: {strangers}"


def test_ma7_the_agent_allowlists_are_the_class_table_read_per_mode_now_fifteen_and_eleven():
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
    assert (len(daily), len(bulbe)) == (15, 11), (len(daily), len(bulbe))
    assert unknown_mode == bulbe


def test_ma8_a_run_that_walked_its_map_is_checked(owned, tmp_path, monkeypatch):
    from opti_oignon.api import routes_agent

    root = ci._folder(tmp_path / "src", {"budget.py": ci.BUDGET})
    ci._index(owned, "oo", root)
    sandbox = ci._cloned(tmp_path, "s1", root)
    cited = "The limit is set in [budget.py:6]."
    walked, walk = [], [True]

    def loop(**kwargs):
        if walk[0]:
            walked.append(kwargs["tool_handlers"]["mycelium_map"]({"path": "budget.py"}))
        kwargs["on_event"](routes_agent.agent_loop.AgentEvent(kind="done", round=1, data={"final_text": cited}))
        return types.SimpleNamespace(stop_reason="done", rounds=1, final_text=cited)

    monkeypatch.setattr(routes_agent.agent_loop, "run", loop)
    manager = routes_agent.AgentRunManager()
    recorder = ci._Recorder()
    manager.subscribe(recorder)
    for _ in range(2):
        started = manager.start("find the budget", model_client=object(), mode="daily", sandbox=sandbox,
                                include_memory=False, consult=False)
        assert started.get("started") is True, started
        manager.join(timeout=10.0)
        walk[0] = False
    first, second = recorder.done()
    assert walked and walked[0].startswith("budget.py"), walked
    final = first["data"]["final_text"]
    assert final.startswith(cited + "\n\n") and len(final) > len(cited) + 2, final
    assert [c["verdict"] for c in first["data"]["citations"]["checked"]] == ["unseen"], first["data"]
    assert second["data"] == {"final_text": cited}, ("a run that walked nothing was checked", second["data"])
