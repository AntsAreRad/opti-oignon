"""Contracts for the images a code run copies out of its sandbox: kept by conversation, gone with it.

A run whose sandbox goes with it copies its new images to the code output directory (``data/exec_outputs``) and names
them in its result. They were kept flat, by no conversation: a deleted conversation's charts stayed there for good,
and the account wipe did not reach them. Now a run of a conversation copies them into the conversation's own folder,
named by a digest of its id; a run of none, into the directory itself. A conversation's deletion asks for them before
it is committed, as it asks every store that keeps something drawn from the conversation beyond this process: its
folder and every file at the top -- of no conversation, or left by an older version with none kept -- overwritten,
then removed; one that stays refuses the deletion. The wipe, for a single-user install's one user, empties the whole
directory.

  * XO1 -- BY CONVERSATION: through the real executor and sandbox, a run of a conversation copies its image into the
    conversation's folder (a digest, never the id as a path), another conversation's into its own, a run of none to
    the top.
  * XO2 -- GONE WITH IT: forgetting a conversation overwrites and removes its folder and the files at the top, and
    leaves another conversation's; from then on this process copies out nothing of it, until the deletion is refused
    and it is remembered again -- after any failure, a directory it cannot list too (round 1); a file that stays is
    raised, never passed over; forgetting a conversation that had no code sandbox asks nothing of the sandbox
    manager.
  * XO3 -- THROUGH A REAL DELETION: the conversation's deletion forgets its outputs before it is committed; one that
    cannot be removed refuses the deletion by name, and the conversation stays.
  * XO4 -- THE WIPE: for a single-user install's one user, the wipe empties the output directory whole and names it
    removed; one that stays is named by the inventory's label; the inventory and the wipe's lists say so.
  * XO5 -- dl6 of tests/test_deletion_leaves_nothing_contracts.py, whose stand-in executor had no outputs to forget:
    its assertions unchanged, the stand-in given its outputs' forgetting, which the deletion asks too.
  * XO6 -- sk11 of tests/test_skill_forgetting_contracts.py, which counted five copies of the overwrite: re-asserted
    unchanged on those five, and the sixth, the code executor's, the same text.
  * XO7 -- THE CODE PANEL KEEPS ITS RUNS BY ITS CONVERSATION (round 1): the panel, the one shipped caller of the
    execution route, passes the conversation it is open on; before, every run of it kept none.
  * XO8 -- THE ROUTE KEEPS A RUN BY A CONVERSATION THE STORE HOLDS (round 2): an id the store does not hold (deleted
    elsewhere, typed by hand), or a store that cannot answer, keeps the run by none, so the next deletion takes its
    outputs; one it holds is passed on to the executor.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import ast
import hashlib
import logging
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_code_execution_in_the_sandbox_contracts as ce_t  # noqa: E402
import test_deletion_leaves_nothing_contracts as dl  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_xo1_by_conversation": 8.0,
    "test_xo2_gone_with_it": 3.0,
    "test_xo3_through_a_real_deletion": 4.0,
    "test_xo4_the_wipe": 5.0,
    "test_xo5_the_code_sandbox_the_workspace_binding_and_the_code_outputs_let_go": 4.0,
    "test_xo6_the_overwrite_is_one_text_in_its_six_copies": 4.0,
    "test_xo7_the_code_panel_keeps_its_runs_by_its_conversation": 2.0,
    "test_xo8_the_route_keeps_a_run_by_a_conversation_the_store_holds": 3.0,
}

_CE = "opti_oignon.code_executor"


def _image(words):
    """A run that writes ``words`` to an image file of its workspace."""
    return f"open('chart.png', 'wb').write({words!r})\n"


def _digest(conv_id):
    return "c-" + hashlib.sha256(conv_id.encode("utf-8")).hexdigest()[:32]


def _executor(data, asked=None):
    """code_executor alone, its output directory under ``data``; with ``asked``, a sandbox manager module that
    notes each name asked of it. The closer restores."""
    config = types.ModuleType("opti_oignon.config")
    config.DATA_DIR = data
    seeded = {"opti_oignon.config": config}
    if asked is not None:
        manager = types.ModuleType("opti_oignon.sandbox_manager")
        manager.__getattr__ = lambda name: asked.append(name)
        seeded["opti_oignon.sandbox_manager"] = manager
    loaded, restore = isolate(targets={_CE: source("code_executor.py")}, seeded=seeded)
    return loaded[_CE], restore


def _plant(folder, name, words):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(words)
    return path


def test_xo1_by_conversation(tmp_path):
    sm, ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path)
        executor = ce.CodeExecutor(sandbox_mgr=manager)
        runs = {conv: executor.execute(_image(conv.encode()), "python", conv_id=conv)
                for conv in ("conv-zanzibar", "conv-lamu")}
        alone = executor.execute(_image(b"no-conversation"), "python")
    finally:
        restore()
    outputs = tmp_path / "data" / "exec_outputs"
    for conv, result in runs.items():
        assert result.success, (result.error_message, result.stderr[-300:])
        assert len(result.output_files) == 1, result.output_files
        copied = Path(result.output_files[0])
        assert copied.parent == outputs / _digest(conv), (conv, copied)
        assert conv not in str(copied), "the conversation's id is a path"
        assert copied.read_bytes() == conv.encode()
    assert alone.success and len(alone.output_files) == 1
    assert Path(alone.output_files[0]).parent == outputs, "a run of no conversation left the top"


def test_xo2_gone_with_it(tmp_path):
    data = tmp_path / "data"
    outputs = data / "exec_outputs"
    gone = _plant(outputs / _digest("conv-zanzibar"), "1_chart.png", b"zanzibar-chart" * 64)
    kept = _plant(outputs / _digest("conv-lamu"), "1_chart.png", b"lamu-chart")
    top = _plant(outputs, "1_old.png", b"an older version's chart")
    asked = []
    ce, restore = _executor(data, asked)
    try:
        # A conversation with no code sandbox: the sandbox manager is not asked to learn it.
        assert ce.forget_conversation("conv-zanzibar") == 0 and asked == [], asked
        held = open(gone, "rb")  # noqa: SIM115 - a reader that still holds the file sees its bytes overwritten
        try:
            assert ce.forget_outputs("conv-zanzibar") == 2
            assert held.read() == b"\0" * len(b"zanzibar-chart" * 64), "a removed output was not overwritten first"
        finally:
            held.close()
        assert not gone.parent.exists() and not top.exists(), "the conversation's outputs, or the top's, stayed"
        assert kept.read_bytes() == b"lamu-chart", "another conversation's outputs went"
        image = tmp_path / "chart.png"
        image.write_bytes(b"after")
        assert ce.CodeExecutor._copy_out([str(image)], "conv-zanzibar") == [], "a forgotten conversation copied out"
        assert not (outputs / _digest("conv-zanzibar")).exists()
        ce.unforget_conversation("conv-zanzibar")
        again = ce.CodeExecutor._copy_out([str(image)], "conv-zanzibar")
        assert len(again) == 1 and Path(again[0]).parent == outputs / _digest("conv-zanzibar")

        stuck = _plant(outputs / _digest("conv-pemba") / "sealed", "kept.png", b"x")
        stuck.parent.chmod(0o500)
        try:
            with pytest.raises(OSError, match="code outputs that stayed"):
                ce.forget_outputs("conv-pemba")
        finally:
            stuck.parent.chmod(0o700)

        # Any failure refuses the deletion and leaves the conversation remembered (round 1): a directory it cannot
        # list, here.
        outputs.chmod(0o300)
        try:
            with pytest.raises(PermissionError):
                ce.forget_outputs("conv-lamu")
        finally:
            outputs.chmod(0o700)
        assert ce.CodeExecutor._copy_out([str(image)], "conv-lamu"), "a refused forgetting left its conversation marked"
    finally:
        restore()


def test_xo3_through_a_real_deletion(tmp_path):
    data = tmp_path / "data"
    ce, close = _executor(data)
    try:
        world = dl._world(tmp_path, stores={_CE: ce})
        try:
            gone = dl._conversation(world, "Zanzibar")
            kept = dl._conversation(world, "Lamu")
            outputs = data / "exec_outputs"
            mine = _plant(outputs / _digest(gone), "1_chart.png", b"zanzibar-chart")
            theirs = _plant(outputs / _digest(kept), "1_chart.png", b"lamu-chart")
            assert world.manager.delete_conversation(gone)
            assert not mine.exists() and theirs.exists(), "the deletion left the conversation's outputs"

            stuck = _plant(outputs / _digest(kept) / "sealed", "kept.png", b"x")
            stuck.parent.chmod(0o500)
            try:
                with pytest.raises(world.conv.ConversationNotForgotten, match="code outputs"):
                    world.manager.delete_conversation(kept)
            finally:
                stuck.parent.chmod(0o700)
            assert world.manager.get_conversation(kept) is not None, "a refused deletion went through"
            image = tmp_path / "chart.png"
            image.write_bytes(b"after")
            assert ce.CodeExecutor._copy_out([str(image)], kept), "a conversation that stays copies out no more"
        finally:
            world.restore()
    finally:
        close()


def test_xo4_the_wipe(tmp_path):
    data = tmp_path / "data"
    ce, close = _executor(data)
    try:
        world = dl._world(tmp_path, stores={_CE: ce}, wipe=True)
        try:
            udm = world.mod("user_data_manager")
            outputs = data / "exec_outputs"
            for folder in (_digest("conv-a"), _digest("conv-b")):
                _plant(outputs / folder, "1_chart.png", b"chart")
            _plant(outputs, "1_none.png", b"none")
            assert udm.UserDataDeleter._empty_copies() == []
            assert list(outputs.iterdir()) == [], "the wipe left code outputs"

            stuck = _plant(outputs / _digest("conv-c") / "sealed", "kept.png", b"x")
            stuck.parent.chmod(0o500)
            try:
                failed = udm.UserDataDeleter._empty_copies()
            finally:
                stuck.parent.chmod(0o700)
            assert [line for line in failed if line.startswith("code execution outputs (not emptied")], failed
            label = next(label for label in udm.WIPE_SINGLE_USER if label.startswith("code execution outputs"))
            assert label in udm.WIPE_NOT_COVERED, "the outputs are removed for one user only, named otherwise"
        finally:
            world.restore()
    finally:
        close()
    loaded, restore = isolate(targets={"opti_oignon.store_inventory": source("store_inventory.py")})
    try:
        inventory = loaded["opti_oignon.store_inventory"]
        entry = next(e for e in inventory.STORES if e.names == ("exec_outputs",))
        assert (entry.label, entry.holds, entry.wipe, entry.failures) == (
            label, inventory.DRAWN, inventory.ONE_USER, ("code execution outputs",)), entry
        assert "code_executor.py" in entry.modules
    finally:
        restore()


def test_xo5_the_code_sandbox_the_workspace_binding_and_the_code_outputs_let_go(tmp_path, caplog):
    # dl6 of tests/test_deletion_leaves_nothing_contracts.py, its assertions unchanged; its stand-in executor
    # forgets its outputs too, which the deletion now asks.
    asked = []
    stand_ins = {name: dl._module(name, forget_conversation=lambda cid, name=name: asked.append((name, cid)) or 1)
                 for name in ("opti_oignon.code_executor", "opti_oignon.sandbox_workspace")}
    # The stand-in executor's outputs, which the deletion asks too, whether this process ran code or not (block 200).
    stand_ins[_CE].forget_outputs = lambda cid: asked.append(("code outputs", cid)) or 0
    stand_ins[_CE].unforget_conversation = lambda cid: None
    world = dl._world(tmp_path, stores=stand_ins)
    try:
        gone = dl._conversation(world, "Zanzibar")
        assert world.manager.delete_conversation(gone)
        assert sorted(asked) == [("code outputs", gone), ("opti_oignon.code_executor", gone),
                                 ("opti_oignon.sandbox_workspace", gone)], (
            f"the code sandbox, the workspace binding or the code outputs were not asked: {asked}")
    finally:
        world.restore()
    world = dl._world(tmp_path / "unloaded")
    try:
        caplog.set_level(logging.WARNING)
        gone = dl._conversation(world, "Pemba")
        assert world.manager.delete_conversation(gone)
        said = caplog.text
        assert "no code sandbox in this tree" not in said and "no workspace binding in this tree" not in said, (
            "a store of this process's memory was loaded only to be asked")
        assert "no sandbox audit in this tree" not in said, "the sandbox manager was loaded only to be asked"
        # The outputs are files on disk: asked whether loaded or not; this window has no executor, which is said.
        assert "no code outputs in this tree" in said, said
    finally:
        world.restore()

    tree = ast.parse(source("chat_coding_agent.py").read_text(encoding="utf-8"))
    ensure = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                   and node.name == "_ensure_sandbox"), None)
    assert ensure is not None, "the census reads no chat coding sandbox's creation"
    body = ast.unparse(ensure)
    assert body.index("create_sandbox(self._session_id)") < body.index(
        "attribute(self._session_id, self._conversation_id)"), (
        "a chat coding session's sandbox does not say its conversation, so its audit rows keep none")

    sandboxes = dl._Sandboxes(tmp_path)
    loaded, restore = isolate(targets={"opti_oignon.code_executor": source("code_executor.py")})
    try:
        ce = loaded["opti_oignon.code_executor"]
        executor = ce.code_executor
        executor._sandbox_mgr = sandboxes
        sid = executor._conversation_session(sandboxes, "conv-zanzibar")
        assert sandboxes.given == [(sid, "conv-zanzibar")], "the code sandbox does not say its conversation"
        assert ce.forget_conversation("conv-zanzibar") == 1 and sandboxes.destroyed == [sid], (
            "the deleted conversation's code sandbox was not destroyed")
        assert ce.forget_conversation("conv-zanzibar") == 0, "a sandbox already gone was destroyed again"
    finally:
        restore()

    sandboxes = dl._Sandboxes(tmp_path)
    loaded, restore = isolate(targets={"opti_oignon.sandbox_workspace": source("sandbox_workspace.py")},
                              seeded={"opti_oignon.sandbox_manager": dl._module(
                                  "opti_oignon.sandbox_manager", sandbox_manager=sandboxes)})
    try:
        sw = loaded["opti_oignon.sandbox_workspace"]
        assert sw.forget_conversation("conv-zanzibar") == 0, "a binding store never made held a binding"
        bindings = sw.get_workspace_bindings()
        bindings.bind("conv-zanzibar", "ws-1")
        bindings.bind("conv-lamu", "ws-2")
        assert sw.forget_conversation("conv-zanzibar") == 1, "the deleted conversation's workspace stayed bound"
        assert bindings.get_conversation_for("ws-1") is None and bindings.get_conversation_for("ws-2") == "conv-lamu"
        assert sandboxes.bindings[-1] == ("ws-1", None), "the workspace's session still names the conversation"
    finally:
        restore()


def test_xo6_the_overwrite_is_one_text_in_its_six_copies(tmp_path, monkeypatch):
    # sk11 of tests/test_skill_forgetting_contracts.py, which counted five copies, re-asserted unchanged on them; the
    # sixth, the code executor's, is the same text.
    import test_project_forgetting_contracts as pf
    import test_skill_forgetting_contracts as sk

    copies = pf._copies()
    assert sorted(copies) == ["agent/skills.py", "code_executor.py", "history.py", "learned_router.py",
                              "notes/blob_store.py", "projects.py"], sorted(copies)
    assert len(set(copies.values())) == 1, "the copies of the overwrite differ"
    five = {path: text for path, text in copies.items() if path != "code_executor.py"}
    monkeypatch.setattr(pf, "_copies", lambda: dict(five))
    sk.test_sk11_the_overwrite_is_one_text_in_its_five_copies(tmp_path, monkeypatch)
    tree = ast.parse(source("code_executor.py").read_text(encoding="utf-8"))
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"forget_outputs", "empty_outputs", "_overwrite_and_remove", "_made_writable"} <= defined


def test_xo7_the_code_panel_keeps_its_runs_by_its_conversation():
    panel = (source().parent / "frontend" / "src" / "lib" / "components" / "panels" / "CodePanel.svelte")
    text = panel.read_text(encoding="utf-8")
    calls = [line.strip() for line in text.splitlines() if "executeCode(" in line and "import" not in line]
    assert len(calls) == 1, calls
    assert "conv_id: $activeConversationId" in calls[0], calls
    imports = [line for line in text.splitlines() if "from '$lib/stores/conversations'" in line]
    assert imports and "activeConversationId" in imports[0], imports
    api = (source().parent / "frontend" / "src" / "lib" / "api" / "code.ts").read_text(encoding="utf-8")
    assert "conv_id?: string;" in api and "'/api/code/execute', params" in api, "the route no longer takes it"


def test_xo8_the_route_keeps_a_run_by_a_conversation_the_store_holds():
    given = []

    def execute(**kwargs):
        given.append(kwargs["conv_id"])
        return types.SimpleNamespace(success=True, stdout="", stderr="", return_code=0, execution_time=0.0,
                                     language="python", truncated=False, error_message="", output_files=[])

    class Store:
        broken = False

        def get_conversation(self, conv_id):
            if self.broken:
                raise RuntimeError("the store cannot be read")
            return object() if conv_id == "conv-held" else None

    store = Store()
    deps = types.ModuleType("opti_oignon.api.deps")
    deps.CODE_EXECUTOR_AVAILABLE, deps.CONVERSATION_AVAILABLE = True, True
    deps.code_executor = types.SimpleNamespace(execute=execute)
    deps.conversation_manager = store
    loaded, restore = isolate(targets={"opti_oignon.api.schemas": source("api", "schemas.py"),
                                       "opti_oignon.api.routes_code": source("api", "routes_code.py")},
                              seeded={"opti_oignon.api.deps": deps}, packages=("opti_oignon.api",))
    try:
        route = loaded["opti_oignon.api.routes_code"].execute_code
        for asked in ("conv-held", "conv-gone", None, ""):
            route(types.SimpleNamespace(code="print(1)", language="python", timeout=None, conv_id=asked))
        store.broken = True
        route(types.SimpleNamespace(code="print(1)", language="python", timeout=None, conv_id="conv-held"))
    finally:
        restore()
    assert given == ["conv-held", None, None, None, None], given
