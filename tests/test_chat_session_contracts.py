#!/usr/bin/env python3
"""Contracts for the chat session behind ``oo chat``.

The session turns a line into a turn or a command. A turn goes through the
real executor, over a scripted registry, under the session's conversation
id; a command is a user action on the onion or the skill registry, and
every refusal comes back as an event carrying its reason, never as an
exception into the loop that prints.

  * CH1 -- a line is a turn: analysed, routed, streamed through the
    executor under one conversation id created on the first turn and kept
    after it; the tokens arrive in order and the request reaches the
    registry.
  * CH2 -- ``/skill`` runs its arguments as the turn with a published
    skill's body in the system prompt; a draft, an unknown name, a name
    published twice and a missing request are refused by name, and none
    of them sends a request.
  * CH3 -- the onion commands over the real librarian: ``/pin`` lands in
    the Core, ``/close`` empties the Flesh through the gate, saves and
    ends the conversation, ``/open`` after a restart finds it again with
    the same Core root, ``/recall`` hands the verbatim span back and says
    the receipt is resolved.
  * CH4 -- every refusal is an event by name: the onion switched off, a
    command without its conversation or its argument, an unknown command,
    an unknown receipt, a span the gate refuses on ``/close``, a turn whose
    executor raises; the session goes on after each, and ends only on
    ``/quit``.
  * CH5 -- the module imports nothing from the package at load: every seam
    is resolved when first used.
  * CH6 -- ``oo chat`` through the click runner: the options reach the
    session, stdin drives it line by line, the stream goes to stdout, a
    refusal to stderr by name, and ``/quit`` ends it before the next line.

Local-only (the public distribution ships no tests). The executor window is
the one of the onion wiring suite; the librarian, its stores and the skill
registry are loaded from source beside it.
"""

import ast
import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402
from _registry_bridge import seed_registry  # noqa: E402

_SESSION = "opti_oignon.cli.session"
_LIBRARIAN = "opti_oignon.memory.librarian"
_SKILLS = "opti_oignon.agent.skills"
_ONION = ("probes", "core_store", "receipts", "composer", "peels", "onion_store", "librarian")


class _Scripted:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("backend down")
        return iter([{"message": {"content": "Hello"}}, {"message": {"content": " world"}, "done": True}])


class _Conversations:
    def __init__(self):
        self.messages = {}

    def add_message(self, conversation_id, role, content, **kwargs):
        self.messages.setdefault(conversation_id, []).append({"role": role, "content": content})

    def get_context_messages(self, conversation_id, **kwargs):
        return list(self.messages.get(conversation_id, []))

    def get_conversation(self, conversation_id):
        return SimpleNamespace(id=conversation_id, messages=self.get_context_messages(conversation_id), metadata={})

    def update_conversation_metadata(self, conversation_id, *args, **kwargs):
        return None


def _routing():
    return SimpleNamespace(model="test-model:1b", task_type="general", temperature=0.2, prompt_variant="standard", timeout=30)


def _load(*, fail=False, cli=False):
    scripted = _Scripted(fail=fail)
    ollama_stub = types.ModuleType("ollama")
    ollama_stub.chat = scripted.chat
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(get_model=lambda *a, **k: "test-model:1b", get_temperature=lambda *a, **k: 0.2)
    router = types.ModuleType("opti_oignon.router")
    router.RoutingResult = type("RoutingResult", (), {})
    retrieval = types.ModuleType("opti_oignon.memory.retrieval")
    retrieval.build_memory_block = lambda *a, **k: ""
    retrieval.working_memory_block = lambda *a, **k: ""
    conversations = _Conversations()
    conversation = types.ModuleType("opti_oignon.conversation")
    conversation.conversation_manager = conversations
    seeded = {
        "opti_oignon.config": cfg,
        "opti_oignon.router": router,
        "opti_oignon.memory.retrieval": retrieval,
        "opti_oignon.conversation": conversation,
    }
    targets = {f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _ONION}
    targets.update({
        "opti_oignon.agent.untrusted_context": source("agent", "untrusted_context.py"),
        _SKILLS: source("agent", "skills.py"),
        "opti_oignon.context_dedup": source("context_dedup.py"),
        "opti_oignon.executor": source("executor.py"),
        _SESSION: source("cli", "session.py"),
    })
    if cli:
        targets.update({f"opti_oignon.cli.{m}": source("cli", f"{m}.py") for m in ("config", "client", "output", "main")})
    seed_registry(seeded, scripted)
    had, prev = "ollama" in sys.modules, sys.modules.get("ollama")
    sys.modules["ollama"] = ollama_stub
    loaded, win_restore = isolate(targets=targets, seeded=seeded,
                                  packages=("opti_oignon.agent", "opti_oignon.memory", "opti_oignon.cli"))
    loaded[_LIBRARIAN].reset_librarian()

    def restore():
        win_restore()
        if had:
            sys.modules["ollama"] = prev
        else:
            sys.modules.pop("ollama", None)

    return loaded, scripted, conversations, restore


def _session(loaded, **kw):
    ids = iter(f"conv-{i}" for i in range(1, 100))
    kw.setdefault("executor", loaded["opti_oignon.executor"].Executor())
    kw.setdefault("analyze", lambda q: SimpleNamespace(task_type="general"))
    kw.setdefault("route", lambda analysis, priority, model: _routing())
    kw.setdefault("new_conversation", lambda title, model: next(ids))
    return loaded[_SESSION].ChatSession(**kw)


def _run(session, line):
    return list(session.handle(line))


def _kinds(events, kind):
    return [e.text for e in events if e.kind == kind]


def _faithful(turns):
    return " ".join(t["text"] for t in turns)


def _onion_seam(loaded, tmp_path, *, summarize=_faithful, enabled=True):
    lib = loaded[_LIBRARIAN]
    peels = loaded["opti_oignon.memory.peels"]
    composer = loaded["opti_oignon.memory.composer"]
    store_mod = loaded["opti_oignon.memory.onion_store"]
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "onion.db"
    cfg = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=1,
                              temperature=0.1, num_predict=64, persist_path=str(path), require_encryption=False)
    gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
    budget = composer.Budget(window=2000, reserve=200, core=300, receipts=300, peels=800, flesh=200, turn=200)

    def install_store():
        lib._store[(str(path), False)] = store_mod.OnionStore(
            path, connect=lambda p: sqlite3.connect(str(p)), require_encryption=False)

    install_store()
    seam = SimpleNamespace(
        onion_enabled=lambda: enabled,
        open_onion=lambda cid: lib.open_onion(cid, config=cfg, budget=budget),
        close_onion=lambda cid: lib.close_onion(cid, config=cfg, summarize=summarize, gate=gate),
        pin=lambda cid, text, actor: lib.pin(cid, text, actor=actor, config=cfg, budget=budget),
        recall=lambda cid, key: lib.recall(cid, key, config=cfg),
    )
    return seam, lib, cfg, install_store


def _mirror(lib, cfg, conversations, cid):
    """What the executor's write seam does when the onion is on."""
    assert lib.maybe_curate(cid, conversations.get_context_messages(cid), config=cfg, runner=lambda c: None)


def _skill_registry(loaded, root):
    skills = loaded[_SKILLS]
    body = "## When to Use\nReviewing a diff.\n\n## Procedure\nName every changed function before judging it."

    def put(base, category, name, status):
        d = base / category / name
        d.mkdir(parents=True)
        d.joinpath("SKILL.md").write_text(
            skills.Skill(name=name, category=category, status=status, body=body).to_markdown(), encoding="utf-8")

    put(root, "code", "review", skills.STATUS_PUBLISHED)
    put(root, "code", "twice", skills.STATUS_PUBLISHED)
    put(root, "notes", "twice", skills.STATUS_PUBLISHED)
    put(root / ".drafts", "code", "unvetted", skills.STATUS_DRAFT)
    return skills.SkillRegistry(root)


def _system(call):
    return [m for m in call["messages"] if m.get("role") == "system"][0]["content"]


# ---------------------------------------------------------------------------
# CH1 -- a line is a turn
# ---------------------------------------------------------------------------
def test_ch1_a_line_is_a_turn_streamed_through_the_executor_under_one_conversation():
    loaded, scripted, conversations, restore = _load()
    try:
        session = _session(loaded)
        first = _run(session, "What is an onion?")
        assert _kinds(first, "info") == ["conversation conv-1"], "the conversation is created on the first turn, and said"
        assert "".join(_kinds(first, "token")) == "Hello world", "the stream, in order"
        assert _kinds(first, "refusal") == []
        second = _run(session, "And a shallot?")
        assert _kinds(second, "info") == [] and session.conversation_id == "conv-1", "the same conversation carries on"
        assert len(scripted.calls) == 2, "each turn reached the registry once"
        user = [m for m in scripted.calls[1]["messages"] if m.get("role") == "user"]
        assert user[-1]["content"].startswith("And a shallot?")
        assert [m["role"] for m in conversations.messages["conv-1"]] == ["user", "assistant", "user", "assistant"]
        assert _run(session, "   ") == [], "an empty line is nothing"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CH2 -- /skill
# ---------------------------------------------------------------------------
def test_ch2_a_published_skill_rides_the_system_prompt_and_every_other_is_refused(tmp_path):
    loaded, scripted, conversations, restore = _load()
    try:
        session = _session(loaded, skills=_skill_registry(loaded, tmp_path / "skills"))
        events = _run(session, "/skill review check the parser change")
        assert "skill code/review v1" in _kinds(events, "info")
        assert "".join(_kinds(events, "token")) == "Hello world"
        assert len(scripted.calls) == 1
        assert "Name every changed function before judging it." in _system(scripted.calls[0]), "the body is in the system prompt"
        user = [m for m in scripted.calls[0]["messages"] if m.get("role") == "user"]
        assert user[-1]["content"].startswith("check the parser change"), "the arguments are the turn"
        assert _kinds(_run(session, "/skill code/review and again"), "refusal") == [], "category/name addresses it too"

        refused = {
            "/skill unvetted do it": "draft, not published",
            "/skill code/unvetted do it": "draft, not published",
            "/skill nowhere do it": "no published skill nowhere",
            "/skill twice do it": "more than one category",
            "/skill review": "needs a request",
            "/skill": "needs a skill name",
        }
        sent = len(scripted.calls)
        for line, reason in refused.items():
            events = _run(session, line)
            assert len(_kinds(events, "refusal")) == 1 and reason in _kinds(events, "refusal")[0], line
            assert _kinds(events, "token") == [], line
        assert len(scripted.calls) == sent, "no refused skill sent a request"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CH3 -- the onion commands over the real librarian
# ---------------------------------------------------------------------------
def test_ch3_pin_close_open_and_recall_act_on_the_real_onion(tmp_path):
    loaded, scripted, conversations, restore = _load()
    try:
        seam, lib, cfg, install_store = _onion_seam(loaded, tmp_path)
        session = _session(loaded, librarian=seam)
        for question in ("Alice runs service 1 on 2026-03-01.", "Service 2 stays on the new cluster.", "Bob owns service 3."):
            _run(session, question)
        cid = session.conversation_id
        _mirror(lib, cfg, conversations, cid)

        pinned = _run(session, "/pin The user is called Alice.")
        assert _kinds(pinned, "refusal") == [] and _kinds(pinned, "info")[0].startswith("pinned ")
        assert [e.text for e in lib.core_entries(cid, config=cfg)] == ["The user is called Alice."]

        closed = _run(session, "/close")
        assert _kinds(closed, "refusal") == []
        assert "3 span(s) evicted, 0 turn(s) left verbatim" in closed[0].text and "saved" in closed[0].text
        root = lib.peek_state(cid, cfg).core.root()
        assert root[:12] in closed[0].text, "the Core root it leaves is printed"
        assert session.conversation_id is None, "a closed conversation is no longer the session's"

        lib.reset_librarian()
        install_store()
        opened = _run(session, f"/open {cid}")
        assert _kinds(opened, "refusal") == [] and session.conversation_id == cid
        assert f"opened {cid}: 0 turn(s) in the Flesh, 3 peel(s), Core root {root[:12]}" == opened[0].text
        key = lib.open_receipts(cid, config=cfg)[0].key
        assert key[:12] in opened[1].text, "the open receipts are listed"

        recalled = _run(session, f"/recall {key}")
        assert _kinds(recalled, "refusal") == []
        assert "Alice runs service 1 on 2026-03-01." in recalled[0].text, "the verbatim span"
        assert "resolved" in recalled[0].text, "the side effect is said"
        assert key not in [r.key for r in lib.open_receipts(cid, config=cfg)]
    finally:
        restore()


# ---------------------------------------------------------------------------
# CH4 -- every refusal is an event by name
# ---------------------------------------------------------------------------
def test_ch4_every_refusal_is_an_event_by_name_and_the_session_goes_on(tmp_path):
    loaded, scripted, conversations, restore = _load()
    try:
        off, *_ = _onion_seam(loaded, tmp_path / "off", enabled=False)
        session = _session(loaded, librarian=off)
        _run(session, "Hello there.")
        assert "switched off" in _kinds(_run(session, "/pin anything"), "refusal")[0]

        blank = lambda turns: "nothing of note"  # noqa: E731
        seam, lib, cfg, _ = _onion_seam(loaded, tmp_path / "on", summarize=blank)
        session = _session(loaded, librarian=seam, new_conversation=lambda title, model: "conv-on")
        cases = {
            "/pin anything": "needs a conversation",
            "/close": "needs a conversation",
            "/recall abc": "needs a conversation",
            "/open": "needs a conversation id",
            "/open nobody": "nothing persisted",
            "/frobnicate": "unknown command /frobnicate",
        }
        for line, reason in cases.items():
            refusals = _kinds(_run(session, line), "refusal")
            assert len(refusals) == 1 and reason in refusals[0], line

        _run(session, "Alice runs service 1 on 2026-03-01.")
        cid = session.conversation_id
        _mirror(lib, cfg, conversations, cid)
        assert "needs the text" in _kinds(_run(session, "/pin"), "refusal")[0]
        assert "not in the ledger" in _kinds(_run(session, "/recall " + "0" * 64), "refusal")[0]
        closed = _run(session, "/close")
        refusals = _kinds(closed, "refusal")
        assert len(refusals) == 1 and "the gate refused" in refusals[0] and "failed:" in refusals[0], (
            "the span the gate refused is named, with its failed probes"
        )
        assert "0 span(s) evicted, 2 turn(s) left verbatim" in closed[0].text
        assert len(lib.peek_state(cid, cfg).flesh.turns()) == 2, "no override: the span stays"

        assert "".join(_kinds(_run(session, "Still here?"), "token")) == "Hello world", "the session went on"
        quit_events = _run(session, "/quit")
        assert [e.kind for e in quit_events] == ["quit"]
        assert "ended" in _kinds(_run(session, "Anyone?"), "refusal")[0]
    finally:
        restore()

    loaded, scripted, conversations, restore = _load(fail=True)
    try:
        session = _session(loaded, executor=SimpleNamespace(execute=_raising))
        refusals = _kinds(_run(session, "Hello?"), "refusal")
        assert refusals == ["turn refused: RuntimeError: executor down"], "a failed turn is said by name"
        assert "/help" in _kinds(_run(session, "/help"), "info")[0], "and the session goes on"
    finally:
        restore()


def _raising(*args, **kwargs):
    raise RuntimeError("executor down")


# ---------------------------------------------------------------------------
# CH5 -- nothing from the package at load
# ---------------------------------------------------------------------------
def test_ch5_the_session_imports_nothing_from_the_package_at_load():
    tree = ast.parse((REPO / "opti_oignon" / "cli" / "session.py").read_text(encoding="utf-8"))
    top = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top.append("." * node.level + (node.module or ""))
    assert top, "control: the module has a module-level import to read"
    assert all(not name.startswith((".", "opti_oignon")) for name in top), top
    inner = [
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("opti_oignon")
    ]
    assert len(inner) >= 6, "control: the seams import the package, inside functions"


# ---------------------------------------------------------------------------
# CH6 -- oo chat through the click runner
# ---------------------------------------------------------------------------
def test_ch6_oo_chat_drives_the_session_over_stdin_and_prints_refusals_to_stderr(tmp_path):
    from click.testing import CliRunner

    loaded, scripted, conversations, restore = _load(cli=True)
    try:
        off, *_ = _onion_seam(loaded, tmp_path, enabled=False)
        made = []

        def factory(model, conversation_id):
            made.append((model, conversation_id))
            return _session(loaded, librarian=off, conversation_id=conversation_id)

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            loaded["opti_oignon.cli.main"].cli,
            ["--no-color", "chat", "-m", "test-model:1b", "--conversation", "conv-9"],
            input="What is an onion?\n/pin The user is called Alice.\n/quit\nNever read.\n",
            obj={"chat_session": factory},
        )
        assert result.exit_code == 0, result.output
        assert made == [("test-model:1b", "conv-9")], "the options reach the session"
        assert "Hello world\n" in result.stdout, "the stream, then a line break"
        assert "Error: /pin refused" not in result.stdout
        assert "switched off" in result.stderr and "Error:" in result.stderr, "a refusal goes to stderr by name"
        assert len(scripted.calls) == 1, "the line after /quit was never sent"
        assert [m["content"] for m in conversations.messages["conv-9"] if m["role"] == "user"][0].startswith("What is an onion?")
    finally:
        restore()
