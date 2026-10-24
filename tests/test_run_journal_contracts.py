#!/usr/bin/env python3
"""Contracts for the run journal: an opt-in record of the user's own agent runs, kept on this machine.

``opti_oignon/run_journal.py`` keeps, for each agent run whose task its
caller vouches typed and while its switch is on, the words typed and every
call the model emitted, in order, with the gate's outcome and the trust it
was chosen in; never a tool's output, nor a word the model wrote but its
calls. These contracts hold what it keeps, what it never keeps, where and
how it keeps it, and that it is never in a run's way:

  * RJ1 -- OFF IS NOTHING: in every spelling of off -- no file, no key,
    false, the string "true", 1, a file that does not parse, a list -- a
    run started through the run manager leaves the data directory as it
    was and hands the loop no observer; a run whose task no caller vouches
    typed is not recorded even when on; a recorded run creates nothing
    before its first write.
  * RJ2 -- ONE RECORD, EVERY EMITTED CALL IN ORDER, WITH ITS GATE: a run
    whose calls run, are asked and approved, are asked and denied, are
    refused by the allowlist and are held by a cancel, and a run its repeat
    watch stops before a call, each leave one record whose tools are those
    the dispatch resolves from the model's answers, in order, each with its
    outcome and whether a person was asked.
  * RJ3 -- NO RESULT TEXT: words placed in a tool's output, the memory
    block, the skills block, the model's prose, the final answer, a
    subtask's answer and the verifier's text reach no byte of the journal,
    in plaintext, nor in a sealed line once opened, though each of them but
    the verifier's last words reached the model; the typed words do.
  * RJ4 -- ARGUMENTS AS DISPATCHED: native or written in the text, a
    call's arguments are kept as the dispatch resolved them, never as its
    sink coerced them; a value past the bound is cut and its name listed,
    the digest taken over the whole; typed words past it are cut and said.
  * RJ5 -- TRUST AT THE MODEL'S TURN: a round-two call reads what round one
    read and a round-one call nothing; each call's class and labels are
    the gate's; "memory" is in a call's context exactly when its run was
    handed a memory block, and "tool" once it was handed a tool's output.
  * RJ6 -- A SUB-AGENT IS ITS OWN RUN: a subtask is recorded as
    ``<id>-t1``, naming its parent and the round that started it, its
    rounds left empty, the verifier as ``<id>-v`` with its verdict; the
    parent keeps only its task call; the identifier is the one the run's
    proposals carry.
  * RJ7 -- SEALED, NEVER IN PLAINTEXT WHERE IT MUST NOT BE: with encryption
    on, every line is ``JS1:`` and opens under the journal's subkey alone,
    never the master key, and a warm write loads no key; encryption on
    without a key, a toggle that cannot be read, a cipher that fails and a
    Bulbe run with encryption off write nothing, each counted, two runs said
    once; the security settings turning encryption on seal at once what was
    kept in plaintext, its file replaced whole, a cut record or tail never
    left in clear; with no key at all the sealing answers -1 and says so,
    and a passphrase set up in this process, its variable unset, seals the
    plaintext and the next run.
  * RJ8 -- OWNER ONLY, WHOLE LINES: under umask 0 the directory is 0700 and
    the file 0600, and looser modes found are tightened; a link planted in
    a day's place is refused and what it points at left as it was; a link
    swapped in for the directory once a write holds it is never followed; a
    tail cut short is counted, never raised, and the next record reads; a
    directory, a link and a pipe at day names are refused one by one, the
    deletion returning at once, and the other days are still forgotten,
    pruned and wiped.
  * RJ9 -- RETENTION: a record is kept less than the retention: the days
    before it, and a day dated after tomorrow, go at the server's pass, as
    at boot, and at a write, the oldest day kept stays, and the retention
    set is the one counted; a run whose day is past the retention when it
    ends is not written, and is counted; a deletion left to apply expires
    with the days it could name; 0, 31, "30" and true are refused by name
    and 30 kept; a rewrite's temp left behind is swept and a file not the
    journal's kept; a read and the status change no name and no byte, and
    show no day past the retention nor after tomorrow.
  * RJ10 -- A FAILING JOURNAL NEVER FAILS THE RUN: an observer that fails
    inside or raises out, a directory that cannot be written, a full day
    and a record that cannot be built leave the run's stop, rounds and
    events those of the same run unrecorded; each is counted and said once.
  * RJ11 -- PRIVACY: deleting a conversation here, or on a paired device
    through the tombstone its producer makes, removes its runs and sub-runs
    and those of no other conversation, and the conversation on the peer;
    a run going across a local delete, a peer delete, the user's wipe or
    the full wipe is never written, and one started after is, no stamp
    outliving the runs it was for; with no key a deletion keeps every
    sealed line and goes into the ledger -- a hash and a day, no identifier
    -- reads skip the conversation, and the first pass with the key applies
    it; a run a clock ahead filed under tomorrow is never read once its
    conversation is deleted, a deletion that finds the journal busy waits
    under tomorrow, one held in memory alone is written down, and a ledger
    day that cannot be read hides its conversation; a deletion that fails
    is applied at the next pass, and a day that cannot be rewritten goes
    whole when it held a run of the deletion, and stays otherwise; the
    user's wipe and the full wipe remove every file, and one refused says
    so in not_covered; the export carries no record and names the journal;
    no backup section reads the journal or its switch.
  * RJ12 -- THE READER: records come ordered by their start, day by day, a
    long run that started first coming first; an unreadable line and a tail
    cut short are skipped and counted; the trace lines read, under a copy
    of the bench trace's reading rule, as runs alone whose every call is
    clean and carries the trust its run was handed.
  * RJ13 -- THE BENCH'S CALLS CARRY THE TRUST A JOURNALED RUN'S DO: one
    script run by the injection bench and through the agent route gives
    each call the same trust, a run that delegates as well as one that does
    not, and the route's reading tables are the agent's.
  * RJ14 -- THE SERVER TENDS THE JOURNAL WITH NOTHING ELSE TO DO: a day
    past the retention found at boot is gone once the application has
    started, and one that passes it while the application runs goes
    within the hour, though no run, read or deletion comes.
  * RJ15 -- THE SECURITY SETTINGS THE JOURNAL READS ARE WRITTEN WHOLE: the
    security routes and the security mode replace security.yaml in one
    move, its mode kept and no temp left beside it, so a reader never
    finds it empty or half written.
  * RJ16 -- THE DIRECTORY COMES FROM THE CONFIGURATION MODULE: with no
    directory given, the journal lives in the configuration module's data
    directory, though the package exports ``config`` as the settings object,
    which holds none -- as the real package does.

The journal, the loop, the dispatch, the gates, the tool registry and the
agent route are loaded through the shared isolation window over a temporary
data directory, a stand-in machine mode and a fixed clock; the encryption
module is the real one where a contract seals -- its key from
OPTI_ENCRYPTION_KEY, its configuration read from a temporary security.yaml,
never the cipher replaced -- and a stand-in that keeps encryption off
elsewhere. RJ14 drives the real application lifespan, as the boot-check
contracts do, over a temporary data directory; RJ15 loads the security
mode and the security routes alone, over a temporary security.yaml.
Local-only.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import stat
import sys
import threading
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_rj1_off_is_nothing": 15.0,
    "test_rj2_one_record_every_emitted_call_in_order_with_its_gate": 10.0,
    "test_rj3_no_result_text_in_plaintext_or_sealed": 15.0,
    "test_rj4_arguments_as_dispatched_the_digest_over_the_whole": 10.0,
    "test_rj5_trust_at_the_models_turn": 10.0,
    "test_rj6_a_sub_agent_is_its_own_run": 10.0,
    "test_rj7_sealed_and_never_in_plaintext_where_it_must_not_be": 15.0,
    "test_rj8_owner_only_whole_lines": 15.0,
    "test_rj9_retention_at_boot_and_write_the_bound_refused_by_name_and_a_read_changes_nothing": 10.0,
    "test_rj10_a_failing_journal_never_fails_the_run": 15.0,
    "test_rj11_privacy_a_deleted_conversation_the_runs_in_flight_the_ledger_the_wipes_and_the_export": 15.0,
    "test_rj12_the_reader_orders_by_start_counts_what_it_skips_and_reads_as_the_bench_trace": 10.0,
    "test_rj13_the_benchs_calls_carry_the_trust_a_journaled_runs_do": 15.0,
    "test_rj14_the_server_tends_the_journal_with_nothing_else_to_do": 30.0,
    "test_rj15_the_security_settings_the_journal_reads_are_written_whole": 10.0,
    "test_rj16_the_directory_comes_from_the_configuration_module": 5.0,
}

_RJ = "opti_oignon.run_journal"
_LOOP = "opti_oignon.agent.loop"
_TOOLS = "opti_oignon.agent.tools"
_DISPATCH = "opti_oignon.agent.dispatch"
_PROV = "opti_oignon.provenance"
_PW = "opti_oignon.pending_writes"
_ROUTES = "opti_oignon.api.routes_agent"
_SECURITY = "opti_oignon.api.routes_security"
_ENC = "opti_oignon.encryption"
_BENCH = "opti_oignon.agent_eval.injection"
_CONV = "opti_oignon.conversation"
_UDM = "opti_oignon.user_data_manager"
_WIPE = "opti_oignon.conversation_wipe"
_ALLOW = "opti_oignon.agent.allowlists"

_AGENT_TARGETS = (
    ("opti_oignon.memory.probes", ("memory", "probes.py")),
    (_PW, ("pending_writes.py",)),
    (_ALLOW, ("agent", "allowlists.py")),
    (_PROV, ("provenance.py",)),
    ("opti_oignon.agent.tool_parsing", ("agent", "tool_parsing.py")),
    (_DISPATCH, ("agent", "dispatch.py")),
    ("opti_oignon.agent.untrusted_context", ("agent", "untrusted_context.py")),
    (_TOOLS, ("agent", "tools.py")),
    (_LOOP, ("agent", "loop.py")),
)

# The clock every journal of these contracts reads, and the day file it names.
_NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
_DAY = "2026-10-16.jsonl"
_TASK = "List the files of my project and say which one is the largest."
_KEY = bytes(range(32))
_LABEL = b"oo-run-journal-v1"
_A_LS = [[("bash", {"command": "ls"})], "Done."]
_SECURITY_TEXT = {True: "encryption:\n  enabled: true\n", False: "encryption:\n  enabled: false\n", None: ""}


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------
class _Machine:
    """The machine's security mode as the gate and the journal read it, switchable."""

    def __init__(self, mode="daily"):
        self.mode = mode

    def module(self):
        sm = types.ModuleType("opti_oignon.security_mode")
        sm.get_current_mode = lambda: self.mode
        sm.is_bulbe = lambda: self.mode == "bulbe"
        sm.is_daily = lambda: self.mode == "daily"
        sm.get_policy = lambda: SimpleNamespace(mode=self.mode, tool_call_approval_required=self.mode == "bulbe",
                                                web_search_allowed=self.mode == "daily")
        sm._audit_log = lambda *args, **kwargs: None
        return sm


class _Sandbox:
    """A sandbox session backed by bwrap that runs nothing: it records each call and answers with ``said``."""

    def __init__(self, said="listed"):
        self.calls = []
        self.said = said
        self.on_bash = None
        self.active = True
        self.sandbox_manager = SimpleNamespace(bwrap_in_use=True, bwrap_available=True)

    def bash(self, command, timeout):
        self.calls.append(("bash", command, timeout))
        if self.on_bash is not None:
            self.on_bash(command)
        return f"{self.said} (bash)"

    def view(self, path, start_line, end_line):
        self.calls.append(("view", path, start_line, end_line))
        return f"{self.said} (view)"

    def ls(self, path, max_entries=200):
        self.calls.append(("ls", path, max_entries))
        return f"{self.said} (ls)"

    def create_file(self, path, content):
        self.calls.append(("create_file", path, len(content)))
        return f"{self.said} (created)"


def _native(name, arguments):
    return {"function": {"name": name, "arguments": dict(arguments)}}


class _Model:
    """A model that answers from a script by who asks -- the run, a subtask, the verifier -- and keeps its answers
    and every conversation it was handed (``seen``).

    A step is a list of ``(tool, arguments)`` calls, emitted natively beside ``prose``; a string, an answer with
    no call; or ``("text", words)``, words whose calls are written in the text. A script run out answers "Done.".
    """

    def __init__(self, script, verifier_prompt, prose=""):
        if not isinstance(script, dict):
            script = {"run": script}
        self.script = {who: list(steps) for who, steps in script.items()}
        self.verifier_prompt = verifier_prompt
        self.prose = prose
        self.answers = {"run": [], "task": [], "verifier": []}
        self.seen = []

    def _who(self, messages):
        first = str(messages[0].get("content", "")) if messages else ""
        if "focused sub-task agent" in first:
            return "task"
        if any(message.get("content") == self.verifier_prompt for message in messages):
            return "verifier"
        return "run"

    def stream(self, messages, tools=None):
        self.seen.append(json.dumps(messages, default=str))
        who = self._who(messages)
        steps = self.script.get(who) or []
        step = steps.pop(0) if steps else "Done."
        if isinstance(step, str):
            answer = {"message": {"content": step}}
        elif isinstance(step, tuple):
            answer = {"message": {"content": step[1]}}
        else:
            answer = {"message": {"content": self.prose, "tool_calls": [_native(name, a) for name, a in step]}}
        self.answers[who].append(answer)
        return [answer]


def _encryption_off():
    """An encryption module whose security.yaml keeps encryption off, and which holds no key."""
    enc = types.ModuleType(_ENC)
    enc.encryption_toggle = lambda path=None: False
    enc.get_encryption_key = lambda: None
    return enc


def _real_encryption(enc, tmp_path):
    """The real encryption module read from a security.yaml in ``tmp_path``: its key file there, never the tree's;
    returns a setter of encryption's switch (True, False, or None for a file that cannot say)."""
    security = Path(tmp_path) / "security.yaml"
    enc._DEFAULT_KEYFILE = Path(tmp_path) / "keyfile"
    enc._load_encryption_config = lambda: {}
    strict = enc.encryption_toggle
    enc.encryption_toggle = lambda path=None: strict(security)

    def encrypted(value):
        security.write_text(_SECURITY_TEXT[value], encoding="utf-8")

    encrypted(False)
    return security, encrypted


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def _world(tmp_path, *, journal="enabled: true\n", sealed=False, policy="network_daily: free\n", skills_block="",
           said="listed", security_routes=False):
    """The agent route and the real loop, dispatch, gates and tools, the journal over ``tmp_path``'s data directory.

    ``journal`` is the journal's settings (None: no file); ``sealed`` loads the real encryption module, its
    security.yaml in ``tmp_path`` (``world.encrypted`` sets it), where it is otherwise a stand-in that keeps
    encryption off; ``security_routes`` loads the security routes over the same file. The review queue lives
    beside the data directory, never in it.
    """
    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / "queue").mkdir()
    machine = _Machine()
    skills = types.ModuleType("opti_oignon.agent.skills")
    skills.make_manage_skills_handler = lambda **kwargs: (lambda arguments: "skills untouched")
    skills.consult_skills = lambda task, registry=None: SimpleNamespace(block=skills_block)
    estop = types.ModuleType("opti_oignon.emergency_stop")
    estop.guard_http = lambda: None
    estop.is_stopped = lambda: True
    capability = types.ModuleType("opti_oignon.capability_manifest")
    capability.model_tool_capable = lambda name: True
    config = types.ModuleType("opti_oignon.config")
    config.DATA_DIR = data
    seeded = {"opti_oignon.security_mode": machine.module(), "opti_oignon.agent.skills": skills,
              "opti_oignon.emergency_stop": estop, "opti_oignon.capability_manifest": capability,
              "opti_oignon.config": config}
    targets = {name: source(*parts) for name, parts in _AGENT_TARGETS}
    targets["opti_oignon.process_secrets"] = source("process_secrets.py")
    targets[_RJ] = source("run_journal.py")
    targets[_ROUTES] = source("api", "routes_agent.py")
    blocked = ()
    if sealed:
        targets[_ENC] = source("encryption.py")
        blocked = ("opti_oignon.secure_bytes",)
        if security_routes:
            targets[_SECURITY] = source("api", "routes_security.py")
    else:
        seeded[_ENC] = _encryption_off()
    loaded, restore = isolate(targets=targets, seeded=seeded, blocked=blocked,
                              packages=("opti_oignon.memory", "opti_oignon.agent", "opti_oignon.api"))
    encrypted = security = None
    try:
        rj, prov, pw, tools, routes = loaded[_RJ], loaded[_PROV], loaded[_PW], loaded[_TOOLS], loaded[_ROUTES]
        settings = tmp_path / "run_journal.yaml"
        if journal is not None:
            settings.write_text(journal, encoding="utf-8")
        rj.CONFIG_FILE = settings
        rj._now = lambda: _NOW
        policy_file = tmp_path / "provenance.yaml"
        policy_file.write_text(policy, encoding="utf-8")
        prov.POLICY_FILE = policy_file
        queue = pw.PendingWriteStore(tmp_path / "queue" / "pending.db")
        pw.set_pending_store(queue)
        tools.reset_tool_registry()
        tools._REGISTRY = tools.ToolRegistry(memory_store=SimpleNamespace(), notes_store=SimpleNamespace(),
                                             skills_handler=lambda arguments: "skills untouched",
                                             web_search_fn=lambda *args, **kwargs: f"{said} (search)")
        routes.reset_run_manager()
        if sealed:
            security, encrypted = _real_encryption(loaded[_ENC], tmp_path)
            if security_routes:
                loaded[_SECURITY]._SECURITY_YAML_PATH = str(security)
    except BaseException:
        restore()
        raise
    return SimpleNamespace(rj=rj, loop=loaded[_LOOP], dispatch=loaded[_DISPATCH], tools=tools, routes=routes,
                           pw=pw, queue=queue, enc=loaded.get(_ENC), sec=loaded.get(_SECURITY), machine=machine,
                           encrypted=encrypted, security=security, data=data, journal=data / "run_journal",
                           sandbox=_Sandbox(said), restore=restore)


def _run(world, script, *, task=_TASK, prose="", **kwargs):
    """One run started on the run manager as the agent route starts one; waits for its end; its model."""
    model = _Model(script, world.loop.VERIFIER_PROMPT, prose=prose)
    kwargs.setdefault("mode", "daily")
    kwargs.setdefault("sandbox", world.sandbox)
    kwargs.setdefault("consult", False)
    kwargs.setdefault("include_memory", False)
    kwargs.setdefault("turn_origin", "typed")
    manager = world.routes.get_run_manager()
    assert manager.start(task, model_client=model, **kwargs) == {"started": True}
    manager.join(timeout=20)
    assert not manager.is_running(), "the run did not end"
    return model


def _tree(directory):
    """Every path under ``directory``, with the digest of each file's bytes."""
    return {str(path.relative_to(directory)): (None if path.is_dir() else hashlib.md5(path.read_bytes()).hexdigest())
            for path in sorted(directory.rglob("*"))}


def _lines(world, name=_DAY):
    path = world.journal / name
    return [line for line in path.read_bytes().split(b"\n") if line] if path.exists() else []


def _by(records, key):
    return {record[key]: record for record in records}


def _said(caplog, logger, reason):
    """How many warnings ``logger`` gave that name ``reason``."""
    return sum(1 for record in caplog.records
               if record.name == logger and record.levelno >= logging.WARNING and reason in record.getMessage())


def _open_sealed(key, line):
    """A sealed line opened with ``key`` and AES-GCM alone."""
    raw = base64.urlsafe_b64decode(line[len(b"JS1:"):])
    return AESGCM(key).decrypt(raw[1:13], raw[13:], None)


def _subkey(master):
    return hmac.new(master, _LABEL, hashlib.sha256).digest()


def _opened_under(key, line):
    """A sealed line opened under ``key``, or nothing when it does not open under it."""
    try:
        return _open_sealed(key, line)
    except Exception:
        return b""


def _conversation_of(key, line):
    """The conversation of a line sealed under ``key``, or None when it does not open under it."""
    opened = _opened_under(key, line)
    return json.loads(opened)["conversation_id"] if opened else None


def _sealed_elsewhere(text):
    """A line sealed as the journal seals one, under the subkey of a master key it does not hold."""
    nonce = os.urandom(12)
    raw = bytes([2]) + nonce + AESGCM(_subkey(b"\x07" * 32)).encrypt(nonce, text.encode("utf-8"), None)
    return b"JS1:" + base64.urlsafe_b64encode(raw)


# ---------------------------------------------------------------------------
# RJ1 -- off is nothing
# ---------------------------------------------------------------------------
_OFF = (
    ("no file", None),
    ("no key", "retention_days: 30\n"),
    ("false", "enabled: false\n"),
    ("the string true", 'enabled: "true"\n'),
    ("one", "enabled: 1\n"),
    ("unparsable", "enabled: [true\n"),
    ("a list", "- enabled\n"),
)


def test_rj1_off_is_nothing(tmp_path):
    world = _world(tmp_path, journal=None)
    seen = []
    try:
        real = world.loop.run

        def spy(*args, **kwargs):
            seen.append(("on_call" in kwargs, world.journal.exists()))
            return real(*args, **kwargs)

        world.loop.run = spy
        before = _tree(world.data)
        offs = {}
        for index, (name, text) in enumerate(_OFF):
            settings = tmp_path / f"off-{index}.yaml"
            if text is not None:
                settings.write_text(text, encoding="utf-8")
            world.rj.CONFIG_FILE = settings
            seen.clear()
            _run(world, _A_LS)
            world.rj.tend()
            offs[name] = (list(seen), world.rj.settings()["state"], _tree(world.data) == before)
        on = tmp_path / "on.yaml"
        on.write_text("enabled: true\n", encoding="utf-8")
        world.rj.CONFIG_FILE = on
        seen.clear()
        _run(world, _A_LS, turn_origin="legacy")
        legacy = (list(seen), _tree(world.data) == before)
        seen.clear()
        _run(world, _A_LS)
        typed = (list(seen), [record["kind"] for record in world.rj.iter_runs()])
    finally:
        world.restore()
    for name, (calls, _state, unchanged) in offs.items():
        assert calls == [(False, False)] and unchanged, ("off hands no observer and writes nothing", name, calls)
    assert {name: state for name, (_calls, state, _same) in offs.items()} == {
        "no file": "off", "no key": "off", "false": "off", "the string true": "off", "one": "off",
        "unparsable": "unreadable", "a list": "unreadable"}, offs
    assert legacy == ([(False, False)], True), ("a run no caller vouches typed is not recorded", legacy)
    assert typed == ([(True, False)], ["run"]), ("witness: a typed run is recorded, nothing made before it", typed)


# ---------------------------------------------------------------------------
# RJ2 -- one record, every emitted call in order, with its gate
# ---------------------------------------------------------------------------
def test_rj2_one_record_every_emitted_call_in_order_with_its_gate(tmp_path):
    world = _world(tmp_path, policy="network_daily: ask_unendorsed\n")
    try:
        manager = world.routes.get_run_manager()
        world.sandbox.on_bash = lambda command: manager.cancel() if command == "echo stop" else None

        def person(conversation_id, tool_name, arguments, labels=None, effect=None):
            return str(arguments.get("query", "")).startswith("yes")

        asked = _run(world, [[("bash", {"command": "echo one"})], [("web_search", {"query": "yes please"})],
                             [("web_search", {"query": "no thanks"})], [("frobnicate", {"level": 1})],
                             [("bash", {"command": "echo stop"}), ("bash", {"command": "echo after"})],
                             "never asked"], approval_fn=person, conversation_id="asked")
        repeated = _run(world, [[("bash", {"command": "echo same"})]] * 4 + ["never asked"],
                        conversation_id="repeated")
        resolve = world.dispatch.resolve_tool_calls
        runs = list(world.rj.iter_runs())
    finally:
        world.restore()
    assert [record["kind"] for record in runs] == ["run", "run"], ("one record per run", runs)
    records = _by(runs, "conversation_id")
    for name, model in (("asked", asked), ("repeated", repeated)):
        emitted = [call.name for answer in model.answers["run"] for call in resolve(answer)[0]]
        assert [call["tool"] for call in records[name]["calls"]] == emitted and emitted, (name, emitted)
        assert [call["i"] for call in records[name]["calls"]] == list(range(len(emitted))), name
    outcomes = [(c["tool"], c["round"], c["gate"]["outcome"], c["gate"]["asked"]) for c in records["asked"]["calls"]]
    assert outcomes == [("bash", 1, "ran", False), ("web_search", 2, "ran", True), ("web_search", 3, "denied", True),
                        ("frobnicate", 4, "refused", False), ("bash", 5, "ran", False),
                        ("bash", 5, "halted", False)], outcomes
    assert records["asked"]["stop_reason"] == "cancelled", records["asked"]["stop_reason"]
    held = [(c["gate"]["outcome"], c["gate"]["reason"], c["gate"]["executed"]) for c in records["repeated"]["calls"]]
    assert held == [("ran", "executed", True)] * 3 + [("not_dispatched", None, False)], held
    assert records["repeated"]["stop_reason"] == "doom_loop", records["repeated"]["stop_reason"]


# ---------------------------------------------------------------------------
# RJ3 -- no result text
# ---------------------------------------------------------------------------
_MARK = {name: f"{name}-q7zv" for name in ("OUTPUT", "MEMORY", "SKILLS", "PROSE", "FINAL", "CHILD", "VERIFIER")}
# What reaches the model on its way: everything but the verifier's last words.
_HANDED = ("OUTPUT", "MEMORY", "SKILLS", "PROSE", "FINAL", "CHILD")


def _marked_run(world):
    script = {
        "run": [[("view", {"path": "notes.txt"})],
                [("task", {"description": "measure", "prompt": "Measure notes.txt."})],
                f"{_MARK['FINAL']} The largest is notes.txt."],
        "task": [[("bash", {"command": "wc -c notes.txt"})], f"{_MARK['CHILD']} it holds twelve bytes"],
        "verifier": [[("view", {"path": "notes.txt"})], f"PASS {_MARK['VERIFIER']}"],
    }
    return _run(world, script, prose=_MARK["PROSE"], include_memory=True, consult=True, verify=True,
                memory_provider=lambda query, user_id=None: f"{_MARK['MEMORY']} the user keeps notes")


def test_rj3_no_result_text_in_plaintext_or_sealed(tmp_path, monkeypatch):
    plain_dir, sealed_dir = tmp_path / "plain", tmp_path / "sealed"
    plain_dir.mkdir()
    sealed_dir.mkdir()
    world = _world(plain_dir, skills_block=f"[skill] {_MARK['SKILLS']} [/skill]", said=_MARK["OUTPUT"])
    try:
        plain_model = _marked_run(world)
        plain = (world.journal / _DAY).read_bytes()
        plain_kinds = sorted(record["kind"] for record in world.rj.iter_runs())
        handed = list(world.sandbox.calls)
    finally:
        world.restore()
    monkeypatch.setenv("OPTI_ENCRYPTION_KEY", base64.urlsafe_b64encode(_KEY).decode("ascii"))
    world = _world(sealed_dir, sealed=True, skills_block=f"[skill] {_MARK['SKILLS']} [/skill]",
                   said=_MARK["OUTPUT"])
    try:
        world.encrypted(True)
        sealed_model = _marked_run(world)
        lines = _lines(world)
    finally:
        world.restore()
    opened = b"\n".join(_opened_under(_subkey(_KEY), line) for line in lines)
    assert plain_kinds == ["run", "task", "verifier"] and len(lines) == 3, ("witness: all three ran", plain_kinds)
    assert len(handed) == 3, ("witness: the tools ran", handed)
    for model in (plain_model, sealed_model):
        given = "".join(model.seen)
        assert all(_MARK[name] in given for name in _HANDED), ("witness: each reached the model",
                                                               [name for name in _HANDED if _MARK[name] not in given])
    for name, marker in _MARK.items():
        assert marker.encode() not in plain, (name, "reached the journal in plaintext")
        assert marker.encode() not in opened, (name, "reached a sealed line")
        assert marker.encode() not in b"".join(lines), (name, "reached the sealed file")
    assert _TASK.encode() in plain and _TASK.encode() in opened, "witness: the typed words are kept"


# ---------------------------------------------------------------------------
# RJ4 -- arguments as dispatched
# ---------------------------------------------------------------------------
_LONG_TASK = _TASK + " Give their sizes in a table, the largest first."


def test_rj4_arguments_as_dispatched_the_digest_over_the_whole(tmp_path):
    world = _world(tmp_path, journal="enabled: true\nmax_value_chars: 64\n")
    big = {"path": "big.txt", "content": "x" * 100}
    try:
        _run(world, [[("bash", {"command": "ls -la", "timeout": "45"})],
                     ("text", 'I will look. [TOOL_CALL] {"name": "view", "arguments": {"path": "notes.txt", '
                              '"start_line": "2"}} [/TOOL_CALL]'),
                     [("create_file", big)], "Done."], task=_LONG_TASK)
        record = next(world.rj.iter_runs())
        handed = list(world.sandbox.calls)
    finally:
        world.restore()
    calls = record["calls"]
    assert [(c["tool"], c["source"]) for c in calls] == [("bash", "native"), ("view", "bracketed"),
                                                         ("create_file", "native")], calls
    assert calls[0]["arguments"] == {"command": "ls -la", "timeout": "45"}, calls[0]["arguments"]
    assert calls[1]["arguments"] == {"path": "notes.txt", "start_line": "2"}, calls[1]["arguments"]
    assert handed[:2] == [("bash", "ls -la", 45), ("view", "notes.txt", 2, 0)], ("witness: the sinks coerced", handed)
    whole = hashlib.sha256(json.dumps(big, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    assert calls[2]["arguments"] == {"path": "big.txt", "content": "x" * 64} and calls[2]["clipped"] == ["content"]
    assert calls[2]["arguments_sha256"] == whole, "the digest is over the arguments whole"
    assert [c["clipped"] for c in calls[:2]] == [[], []], calls
    assert record["typed"] == _LONG_TASK[:64] and record["typed_clipped"] is True, record["typed"]
    assert record["typed_sha256"] == hashlib.sha256(_LONG_TASK.encode()).hexdigest(), record["typed_sha256"]


# ---------------------------------------------------------------------------
# RJ5 -- trust at the model's turn
# ---------------------------------------------------------------------------
def test_rj5_trust_at_the_models_turn(tmp_path):
    world = _world(tmp_path)
    script = [[("view", {"path": "a.txt"})], [("bash", {"command": "wc -c a.txt"}), ("ls", {"path": "."})], "Done."]
    gates = {}
    try:
        manager = world.routes.get_run_manager()
        for name, block in (("with memory", "the user keeps notes"), ("without memory", "")):
            events = []
            manager.subscribe(events.append)
            _run(world, list(script), include_memory=True, conversation_id=name,
                 memory_provider=lambda query, user_id=None, block=block: block)
            manager.unsubscribe(events.append)
            gates[name] = [json.loads(event)["data"]["provenance"] for event in events
                           if json.loads(event)["kind"] == "tool_result"]
        records = _by(world.rj.iter_runs(), "conversation_id")
    finally:
        world.restore()
    for name, contexts in (("with memory", [["memory"], ["memory", "tool"], ["memory", "tool"]]),
                           ("without memory", [[], ["tool"], ["tool"]])):
        trust = [call["trust"] for call in records[name]["calls"]]
        assert [t["read"] for t in trust] == [[], ["view"], ["view"]], (name, [t["read"] for t in trust])
        assert [t["context"] for t in trust] == contexts, (name, [t["context"] for t in trust])
        assert [(t["effect"], t["labels"]) for t in trust] == [(g["effect"], g["labels"]) for g in gates[name]], name
        assert len(gates[name]) == 3 and trust[1]["labels"], ("witness: the gate labelled", gates[name])


# ---------------------------------------------------------------------------
# RJ6 -- a sub-agent is its own run
# ---------------------------------------------------------------------------
def test_rj6_a_sub_agent_is_its_own_run(tmp_path):
    world = _world(tmp_path)
    script = {
        "run": [[("manage_memory", {"action": "add", "text": "The user wants long reports."})],
                [("task", {"description": "measure", "prompt": "Measure notes.txt."})], "Done."],
        "task": [[("bash", {"command": "wc -c notes.txt"})], "twelve bytes"],
        "verifier": [[("view", {"path": "notes.txt"})], "PASS"],
    }
    try:
        _run(world, script, verify=True)
        records = _by(world.rj.iter_runs(), "kind")
        proposals = world.queue.list(status=None)
    finally:
        world.restore()
    assert sorted(records) == ["run", "task", "verifier"], ("the subtask and the verifier are runs", sorted(records))
    run, child, verifier = records["run"], records["task"], records["verifier"]
    rid = run["run_id"]
    assert [c["tool"] for c in run["calls"]] == ["manage_memory", "task"], ("the parent keeps its task call", run)
    assert (child["run_id"], child["parent_run_id"], child["parent_round"]) == (f"{rid}-t1", rid, 2), child
    assert [c["tool"] for c in child["calls"]] == ["bash"] and child["typed"] == run["typed"] == _TASK, child
    assert child["rounds"] is None and child["stop_reason"] is None, ("a subtask's rounds are not reported", child)
    assert (verifier["run_id"], verifier["parent_run_id"], verifier["parent_round"]) == (f"{rid}-v", rid,
                                                                                        run["rounds"]), verifier
    assert [c["tool"] for c in verifier["calls"]] == ["view"] and verifier["verdict"] == run["verdict"] == "pass"
    assert run["parent_run_id"] is None and run["parent_round"] is None and run["offered"], run
    assert [p.run_id for p in proposals] == [rid], ("one identifier for the run and its proposals", proposals)


# ---------------------------------------------------------------------------
# RJ7 -- sealed, never in plaintext where it must not be
# ---------------------------------------------------------------------------
def _broken(subkey, plaintext):
    raise ValueError("no cipher here")


def test_rj7_sealed_and_never_in_plaintext_where_it_must_not_be(tmp_path, monkeypatch, caplog):
    key = base64.urlsafe_b64encode(_KEY).decode("ascii")
    monkeypatch.setenv("OPTI_ENCRYPTION_KEY", key)
    monkeypatch.delenv("OPTI_KEYFILE_PASSPHRASE", raising=False)
    world = _world(tmp_path, sealed=True, security_routes=True)
    rj, enc, name = world.rj, world.enc, world.rj.logger.name
    day, cut_day = world.journal / _DAY, world.journal / "2026-10-15.jsonl"
    seen = {}
    try:
        world.encrypted(True)
        _run(world, _A_LS, conversation_id="sealed")
        loads, loader = [], enc.get_encryption_key
        enc.get_encryption_key = lambda: loads.append(1) or loader()
        _run(world, _A_LS, conversation_id="warm")
        enc.get_encryption_key = loader
        sealed = _lines(world)
        seen["warm"] = len(loads)
        world.encrypted(None)
        _run(world, _A_LS, conversation_id="unknown")
        seen["unknown"] = (rj.failures().get("seal_unavailable"), _lines(world) == sealed)
        world.encrypted(True)
        enc.encrypt_bytes, cipher = _broken, enc.encrypt_bytes
        rj._STATE["cipher"] = None
        _run(world, _A_LS, conversation_id="no cipher")
        enc.encrypt_bytes = cipher
        seen["no cipher"] = (rj.failures().get("seal_failed"), _lines(world) == sealed)
        rj.reset()
        monkeypatch.delenv("OPTI_ENCRYPTION_KEY")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=name):
            for _ in range(2):
                _run(world, _A_LS, conversation_id="no key")
        seen["no key"] = (rj.failures().get("seal_unavailable"), _said(caplog, name, "seal_unavailable"),
                          _lines(world) == sealed)
        world.encrypted(False)
        world.machine.mode = "bulbe"
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=name):
            for _ in range(2):
                _run(world, ["Nothing to call."], conversation_id="bulbe")
        seen["bulbe"] = (rj.failures().get("plaintext_refused"), _said(caplog, name, "plaintext_refused"),
                         _lines(world) == sealed)
        world.machine.mode = "daily"
        _run(world, _A_LS, conversation_id="plain")
        with day.open("ab") as handle:
            handle.write(b'{"calls":[{"arguments":{"path"\n')
        _run(world, _A_LS, conversation_id="plain after a cut")
        with day.open("ab") as handle:
            handle.write(b'{"calls":[{"argu')
        cut_day.write_bytes(b'{"calls":[{"arguments"')
        seen["plain"] = [line[:1] for line in _lines(world)]
        inode = day.stat().st_ino
        monkeypatch.setenv("OPTI_ENCRYPTION_KEY", key)
        world.sec.update_security_config(world.sec.SecurityConfigUpdate(encryption={"enabled": True}))
        seen["put"] = (all(line.startswith(b"JS1:") for line in _lines(world)), b'{"calls"' in day.read_bytes(),
                       cut_day.exists(), day.stat().st_ino != inode,
                       sorted(path.name for path in world.journal.iterdir()))
        seen["read after the put"] = sorted(record["conversation_id"] for record in rj.iter_runs())
        rj.reset()
        monkeypatch.delenv("OPTI_ENCRYPTION_KEY")
        world.encrypted(False)
        _run(world, _A_LS, conversation_id="before setup")
        world.encrypted(True)
        seen["no key at all"] = (rj.on_encryption_enabled(), rj.failures().get("seal_unavailable"),
                                 _lines(world)[-1][:1])
        manager = enc.get_encryption_manager()
        assert manager.setup_from_passphrase("a passphrase long enough to hold"), "witness: the setup ran"
        seen["setup"] = rj.on_encryption_enabled()
        _run(world, _A_LS, conversation_id="after setup")
        last = _lines(world)
        seen["after setup"] = (all(line.startswith(b"JS1:") for line in last),
                               [_conversation_of(_subkey(manager._raw_key()), line) for line in last[-2:]])
    finally:
        world.restore()
    assert len(sealed) == 2 and all(line.startswith(b"JS1:") for line in sealed), sealed
    assert [_conversation_of(_subkey(_KEY), line) for line in sealed] == ["sealed", "warm"], "it opens under the subkey"
    try:
        _open_sealed(_KEY, sealed[0])
        opened_by_master = True
    except Exception:
        opened_by_master = False
    assert not opened_by_master and _TASK.encode() not in sealed[0], "the master key does not open it"
    assert seen["warm"] == 0, ("a warm write loads no key", seen["warm"])
    assert seen["unknown"] == (1, True), ("a toggle that cannot be read writes nothing", seen["unknown"])
    assert seen["no cipher"] == (1, True), ("a cipher that fails writes nothing", seen["no cipher"])
    assert seen["no key"] == (2, 1, True), ("encryption on without a key writes nothing, said once", seen["no key"])
    assert seen["bulbe"] == (2, 1, True), ("Bulbe keeps nothing in plaintext, said once", seen["bulbe"])
    assert seen["plain"] == [b"J", b"J", b"{", b"{", b"{", b"{"], ("witness: plaintext beside a cut record and a "
                                                                  "cut tail", seen["plain"])
    assert seen["put"] == (True, False, False, True, [_DAY]), ("the settings seal at once, whole", seen["put"])
    assert seen["read after the put"] == ["plain", "plain after a cut", "sealed", "warm"], seen["read after the put"]
    assert seen["no key at all"] == (-1, 1, b"{"), ("no key: the sealing says so", seen["no key at all"])
    assert seen["setup"] == 1, ("the key a passphrase set up here seals the plaintext", seen["setup"])
    assert seen["after setup"] == (True, ["before setup", "after setup"]), seen["after setup"]


# ---------------------------------------------------------------------------
# RJ8 -- owner only, whole lines
# ---------------------------------------------------------------------------
def _modes(world):
    return stat.S_IMODE(world.journal.stat().st_mode), stat.S_IMODE((world.journal / _DAY).stat().st_mode)


def test_rj8_owner_only_whole_lines(tmp_path):
    world = _world(tmp_path)
    rj, day = world.rj, world.journal / _DAY
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"not the journal's\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    seen = {}
    try:
        mask = os.umask(0)
        try:
            _run(world, _A_LS)
        finally:
            os.umask(mask)
        seen["made"] = _modes(world)
        first = _lines(world)[0]
        os.chmod(day, 0o644)
        os.chmod(world.journal, 0o755)
        _run(world, _A_LS)
        seen["tightened"] = _modes(world)
        day.unlink()
        day.symlink_to(outside)
        _run(world, _A_LS)
        seen["planted"] = (rj.failures().get("unsafe_path"), outside.read_bytes(), day.is_symlink())
        day.unlink()
        tend = rj._tend

        def swapping(fd):
            world.journal.rename(world.data / "moved")
            world.journal.symlink_to(elsewhere)
            tend(fd)

        rj._tend = swapping
        _run(world, _A_LS, conversation_id="swapped")
        rj._tend = tend
        moved = world.data / "moved" / _DAY
        landed = moved.read_bytes() if moved.exists() else b""
        seen["swapped"] = (sorted(path.name for path in elsewhere.iterdir()),
                           [json.loads(line)["conversation_id"] for line in landed.split(b"\n") if line])
        world.journal.unlink()
        (world.data / "moved").rename(world.journal)
        day.write_bytes(first + b"\n" + first[:25])
        seen["torn"] = ([record["run_id"] for record in rj.iter_runs()], rj.failures().get("torn"))
        _run(world, _A_LS)
        seen["after the cut"] = (len(list(rj.iter_runs())), rj.failures().get("unreadable"))
        day.unlink()
        _run(world, _A_LS, conversation_id="A")
        _run(world, _A_LS, conversation_id="B")
        before = rj.failures().get("unsafe_path", 0)
        (world.journal / "2026-10-14.jsonl").symlink_to(outside)
        (world.journal / "2026-10-13.jsonl").mkdir()
        os.mkfifo(world.journal / "2026-10-12.jsonl")
        answer = {}
        forgetting = threading.Thread(target=lambda: answer.setdefault("n", rj.forget_conversation("A")),
                                      daemon=True)
        forgetting.start()
        forgetting.join(timeout=5)
        blocked = forgetting.is_alive()
        if blocked:  # free a forget that waits on the pipe, so the window can close
            release = os.open(world.journal / "2026-10-12.jsonl", os.O_WRONLY | os.O_NONBLOCK)
            os.close(release)
            forgetting.join(timeout=5)
        seen["forget"] = (blocked, answer.get("n"), [json.loads(line)["conversation_id"] for line in _lines(world)],
                          rj.failures().get("unsafe_path", 0) - before, outside.read_bytes())
        (world.journal / "2026-09-01.jsonl").mkdir()
        (world.journal / "2026-09-02.jsonl").write_bytes(b"")
        rj.prune()
        seen["prune"] = ((world.journal / "2026-09-01.jsonl").is_dir(), (world.journal / "2026-09-02.jsonl").exists(),
                         rj.failures().get("prune_failed"))
        wiped = rj.wipe()
        seen["wipe"] = (wiped.get("wiped"), wiped.get("reason"), sorted(path.name for path in world.journal.iterdir()))
    finally:
        world.restore()
    assert seen["made"] == (0o700, 0o600), ("owner only, whatever the umask", seen["made"])
    assert seen["tightened"] == (0o700, 0o600), ("looser modes found are tightened", seen["tightened"])
    assert seen["planted"] == (1, b"not the journal's\n", True), ("a planted link is refused", seen["planted"])
    assert seen["swapped"] == ([], ["swapped"]), ("a link swapped in for the directory is never followed",
                                                  seen["swapped"])
    assert len(seen["torn"][0]) == 1 and seen["torn"][1] == 1, ("a tail cut short is counted, never raised",
                                                                seen["torn"])
    assert seen["after the cut"] == (2, 1), ("the next record reads on a line of its own", seen["after the cut"])
    assert seen["forget"] == (False, -1, ["B"], 3, b"not the journal's\n"), (
        "a pipe, a directory and a link at day names are refused one by one, the deletion answers at once, and "
        "the later day is still forgotten", seen["forget"])
    assert seen["prune"] == (True, False, 1), ("the next old day is still pruned", seen["prune"])
    assert seen["wipe"] == (False, "wipe_failed", ["2026-09-01.jsonl", "2026-10-13.jsonl"]), (
        "the wipe removes the rest and says what it could not", seen["wipe"])


# ---------------------------------------------------------------------------
# RJ9 -- retention
# ---------------------------------------------------------------------------
_OLD = ("2026-09-16.jsonl", "2026-09-15.jsonl", "2026-01-01.jsonl", "2026-10-18.jsonl",
        ".2026-10-16.jsonl.tmp-0a1b2c3d", "..forgotten.tmp-0a1b2c3d")
_KEPT = (_DAY, "2026-09-17.jsonl", "2026-10-17.jsonl", "notes.txt", ".notes.tmp-0a1b2c3d")


def _plant(world, names, content=b""):
    world.journal.mkdir(mode=0o700, exist_ok=True)
    for name in names:
        (world.journal / name).write_bytes(content)


def test_rj9_retention_at_boot_and_write_the_bound_refused_by_name_and_a_read_changes_nothing(tmp_path, caplog):
    world = _world(tmp_path)
    name = world.rj.logger.name
    spelled = {}
    try:
        for index, given in enumerate(("0", "31", '"30"', "true", "30")):
            settings = tmp_path / f"retention-{index}.yaml"
            settings.write_text(f"enabled: true\nretention_days: {given}\n", encoding="utf-8")
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger=name):
                kept = world.rj.settings(settings)["retention_days"]
            spelled[given] = (kept, _said(caplog, name, "retention_days"))
        passes = {}
        _plant(world, _OLD + _KEPT)
        world.rj.tend()  # the server's pass, the boot's among them
        passes["boot"] = sorted(path.name for path in world.journal.iterdir())
        ledger = world.journal / ".forgotten"
        ledger.write_bytes(b"a" * 64 + b" 2026-09-16\n" + b"b" * 64 + b" 2026-09-17\n")
        world.rj.prune()
        expired = ledger.read_bytes().split()
        ledger.unlink()
        _plant(world, _OLD)
        world.rj._STATE["prune_at"] = float("inf")  # the run's start prunes nothing: its write does
        _run(world, _A_LS)
        passes["write"] = sorted(path.name for path in world.journal.iterdir())
        ahead = world.rj.failures().get("future_day")
        world.rj._now = lambda: datetime(2026, 9, 16, 23, 0, tzinfo=timezone.utc)
        late = world.rj.open_run(run_id="late", typed=_TASK, mode="daily")
        world.rj._now = lambda: _NOW
        late = (late.finish(SimpleNamespace(stop_reason="done", rounds=1, verifier=None)),
                world.rj.failures().get("past_retention"), (world.journal / "2026-09-16.jsonl").exists())
        record = _lines(world)[0]
        _plant(world, _OLD, record + b"\n")
        before = _tree(world.journal)
        shown = [entry["run_id"] for entry in world.rj.iter_runs()]
        counted = world.rj.status()["days"]
        read = (_tree(world.journal) == before, len(shown), sorted(counted))
        five = tmp_path / "five.yaml"
        five.write_text("enabled: true\nretention_days: 5\n", encoding="utf-8")
        world.rj.CONFIG_FILE = five
        _plant(world, ("2026-10-12.jsonl", "2026-10-11.jsonl"))
        world.rj.prune()
        shorter = sorted(path.name for path in world.journal.iterdir())
    finally:
        world.restore()
    assert spelled == {"0": (30, 1), "31": (30, 1), '"30"': (30, 1), "true": (30, 1), "30": (30, 0)}, spelled
    for when, names in passes.items():
        assert names == sorted(_KEPT), (when, "older days, days after tomorrow and temps go; the oldest day kept, "
                                              "tomorrow and foreign files stay", names)
    assert ahead == 2, ("a day after tomorrow is counted", ahead)
    assert late == (False, 1, False), ("a run whose day is past the retention when it ends is not written",
                                       late)
    assert expired == [b"b" * 64, b"2026-09-17"], (
        "a deletion left to apply expires with the days it could name", expired)
    assert read == (True, 1, ["2026-09-17", "2026-10-16", "2026-10-17"]), (
        "a read and the status change nothing, and show no day past the retention nor after tomorrow", read)
    assert "2026-10-12.jsonl" in shorter and "2026-10-11.jsonl" not in shorter, ("the retention set is used",
                                                                                 shorter)


# ---------------------------------------------------------------------------
# RJ10 -- a failing journal never fails the run
# ---------------------------------------------------------------------------
def _watched(world, script):
    """The run's stop and rounds, and every event it sent, in order."""
    events = []
    manager = world.routes.get_run_manager()
    manager.subscribe(events.append)
    try:
        _run(world, script)
    finally:
        manager.unsubscribe(events.append)
    status = manager.status()
    return status["stop_reason"], status["rounds"], [json.loads(event) for event in events]


def test_rj10_a_failing_journal_never_fails_the_run(tmp_path, caplog):
    world = _world(tmp_path, journal="enabled: false\n")
    script = [[("view", {"path": "a.txt"})], [("bash", {"command": "wc -c a.txt"})], "Done."]
    name = world.rj.logger.name
    runs = {}
    try:
        runs["off"] = _watched(world, list(script))
        on = tmp_path / "on.yaml"
        on.write_text("enabled: true\nmax_day_bytes: 65536\n", encoding="utf-8")
        world.rj.CONFIG_FILE = on
        with caplog.at_level(logging.WARNING, logger=name):
            observe = world.loop.CallTrust.observe

            def failing_inside(self, event):
                raise RuntimeError("the trust cannot be built")

            world.loop.CallTrust.observe = failing_inside
            runs["failing inside"] = _watched(world, list(script))
            world.loop.CallTrust.observe = observe
            on_call = world.rj.Run.on_call

            def raising_out(self, event):
                raise RuntimeError("the observer raises")

            world.rj.Run.on_call = raising_out
            runs["raising out"] = _watched(world, list(script))
            world.rj.Run.on_call = on_call
            locked = tmp_path / "locked"
            locked.mkdir(mode=0o500)
            world.rj.DATA_DIR = locked
            runs["unwritable"] = _watched(world, list(script))
            world.rj.DATA_DIR = None
            world.journal.mkdir(mode=0o700, exist_ok=True)
            (world.journal / _DAY).write_bytes(b"x" * 65530)
            runs["full day"] = _watched(world, list(script))
            (world.journal / _DAY).unlink()
            strict = world.rj._strict

            def unbuildable(value):
                raise RuntimeError("the record cannot be built")

            world.rj._strict = unbuildable
            runs["unbuildable"] = _watched(world, list(script))
            world.rj._strict = strict
            runs["control"] = _watched(world, list(script))
            counted = world.rj.failures()
            said = {reason: _said(caplog, name, reason)
                    for reason in ("observe_failed", "write_failed", "day_full", "finish_failed")}
            recorded = sorted(len(record["calls"]) for record in world.rj.iter_runs())
        locked.chmod(0o700)
    finally:
        world.restore()
    reference = runs.pop("off")
    assert reference[0] == "done" and reference[1] == 3 and len(reference[2]) > 6, ("witness: the run ran", reference)
    for case, outcome in runs.items():
        assert outcome == reference, (case, "the run changed", outcome[:2])
    assert counted == {"observe_failed": 2, "write_failed": 1, "day_full": 1, "finish_failed": 1}, counted
    assert said == {"observe_failed": 1, "write_failed": 1, "day_full": 1, "finish_failed": 1}, said
    assert recorded == [2], ("witness: the control run is recorded whole", recorded)


# ---------------------------------------------------------------------------
# RJ11 -- privacy
# ---------------------------------------------------------------------------
def _store_window(tmp_path):
    """The conversation store, the user's data controls, the wipe and the journal over ``tmp_path``'s data, the
    real encryption module beside them, and sync's producer over a recording engine (``made``)."""
    data = tmp_path / "data"
    data.mkdir()
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda path, **kwargs: sqlite3.connect(path, check_same_thread=kwargs.get("check_same_thread",
                                                                                                False))
    config = types.ModuleType("opti_oignon.config")
    config.DATA_DIR = data
    guard = types.ModuleType("opti_oignon.veilid.guard")
    guard.veilid_available = lambda: True
    made = []

    class _Engine:
        """A sync engine that journals nothing: it builds each record through the real producer, and keeps it."""

        def current_clock(self, kind, record_id):
            return 0

        def publish_conversation(self, conversation_id, payload=None, *, clock, deleted=False, updated_at=""):
            from opti_oignon.veilid.producers import conversation_record

            made.append(conversation_record(conversation_id, payload, device="device-a", clock=clock,
                                            deleted=deleted, updated_at=updated_at))
            return len(made)

    engine = types.ModuleType("opti_oignon.veilid.sync_engine")
    engine.get_sync_engine = lambda *args, **kwargs: _Engine()
    loaded, restore = isolate(
        targets={"opti_oignon.process_secrets": source("process_secrets.py"),
                 _ALLOW: source("agent", "allowlists.py"), _ENC: source("encryption.py"),
                 "opti_oignon.veilid.records": source("veilid", "records.py"),
                 "opti_oignon.veilid.producers": source("veilid", "producers.py"),
                 _RJ: source("run_journal.py"), _CONV: source("conversation.py"),
                 _UDM: source("user_data_manager.py"), _WIPE: source("conversation_wipe.py")},
        seeded={"opti_oignon.db_utils": db, "opti_oignon.config": config,
                "opti_oignon.security_mode": _Machine().module(), "opti_oignon.veilid.guard": guard,
                "opti_oignon.veilid.sync_engine": engine},
        blocked=("opti_oignon.secure_bytes",),
        packages=("opti_oignon.agent", "opti_oignon.veilid"))
    try:
        rj = loaded[_RJ]
        settings = tmp_path / "run_journal.yaml"
        settings.write_text("enabled: true\n", encoding="utf-8")
        rj.CONFIG_FILE = settings
        rj._now = lambda: _NOW
        _security, encrypted = _real_encryption(loaded[_ENC], tmp_path)
    except BaseException:
        restore()
        raise
    return SimpleNamespace(rj=rj, conv=loaded[_CONV], udm=loaded[_UDM], wipe=loaded[_WIPE], made=made,
                           encrypted=encrypted, journal=data / "run_journal", restore=restore)


_CALL = SimpleNamespace(name="bash", arguments={"command": "ls"}, source="native")
_RAN = SimpleNamespace(executed=True, reason="executed", provenance={})


def _opened_run(rj, conversation_id, *, sub=False):
    """A run of ``conversation_id`` opened and through its first round, a subtask and a verifier when ``sub``."""
    run = rj.open_run(run_id=f"run-{conversation_id or 'none'}", typed=f"typed in {conversation_id or 'none'}",
                      conversation_id=conversation_id, mode="daily")
    run.on_call({"scope": "run", "round": 1, "call": _CALL, "result": _RAN, "context": []})
    if sub:
        run.on_call({"scope": "task", "subrun": 1, "parent_round": 1, "round": 1, "call": _CALL, "result": _RAN,
                     "context": []})
        run.on_call({"scope": "verifier", "round": 1, "call": _CALL, "result": _RAN, "context": []})
    return run


def _finished(run, *, sub=False):
    """The run through its second round, then finished: whether it was written."""
    run.on_call({"scope": "run", "round": 2, "call": _CALL, "result": _RAN, "context": ["tool"]})
    verifier = SimpleNamespace(rounds=1, verdict="pass") if sub else None
    return run.finish(SimpleNamespace(stop_reason="done", rounds=2, verifier=verifier))


def _seed(rj, conversation_id, *, sub=False):
    assert _finished(_opened_run(rj, conversation_id, sub=sub), sub=sub), conversation_id


def _ids(rj):
    return sorted(record["run_id"] for record in rj.iter_runs())


def _peer_deletes(world, conversation_id):
    """The tombstone a paired device makes for ``conversation_id``, through the real producer, applied here as the
    sync engine's sink applies it; the record."""
    world.conv._sync_publish_conversation(conversation_id, deleted=True, updated_at="2026-10-16T12:00:00")
    record = world.made[-1]
    world.conv.conversation_manager.apply_synced_conversation(record.payload, deleted=record.deleted,
                                                              updated_at=record.updated_at)
    return record


def _ledger_day(world, conversation_id):
    """The day the ledger keeps a conversation's deletion under, or None when it holds none."""
    ledger = world.journal / ".forgotten"
    wanted = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()
    rows = [line.split() for line in ledger.read_text(encoding="ascii").split("\n")] if ledger.exists() else []
    days = [row[1] for row in rows if len(row) > 1 and row[0] == wanted]
    return days[-1] if days else None


def test_rj11_privacy_a_deleted_conversation_the_runs_in_flight_the_ledger_the_wipes_and_the_export(
        tmp_path, monkeypatch):
    monkeypatch.delenv("OPTI_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("OPTI_KEYFILE_PASSPHRASE", raising=False)
    world = _store_window(tmp_path)
    key = base64.urlsafe_b64encode(_KEY).decode("ascii")
    clock = {"now": _NOW}
    seen = {}
    try:
        rj, store = world.rj, world.conv.conversation_manager
        rj._now = lambda: clock["now"]
        here = store.create_conversation(title="here").id
        peer = store.create_conversation(title="peer").id
        for conversation_id, sub in ((here, True), (here + "0", False), (peer, True), ("", False)):
            _seed(rj, conversation_id, sub=sub)
        store.delete_conversation(here)
        seen["here"] = _ids(rj)
        tombstone = _peer_deletes(world, peer)
        seen["peer"] = (tombstone.deleted, tombstone.payload, _ids(rj), store.get_conversation(peer))
        flights = {}
        for gesture in ("local delete", "peer delete", "user wipe", "full wipe"):
            target = store.create_conversation(title=gesture).id
            going = [_opened_run(rj, target), _opened_run(rj, "")]  # its own, and one of no conversation
            going[0].run_id, going[1].run_id = f"going-{gesture}", f"none-{gesture}"
            if gesture == "local delete":
                store.delete_conversation(target)
            elif gesture == "peer delete":
                _peer_deletes(world, target)
            elif gesture == "user wipe":
                world.udm.UserDataDeleter().delete_all("local")
            else:
                world.wipe.ConversationWipeManager().wipe_all(purge_disk=True)
            written = [_finished(run) for run in going]
            after = _opened_run(rj, target)
            after.run_id = f"after-{gesture}"
            written.append(_finished(after))
            flights[gesture] = (written, [entry for entry in _ids(rj) if entry.endswith(gesture)])
        seen["flights"] = flights
        seen["stamps after the runs"] = dict(rj._FORGOTTEN)
        world.udm.UserDataDeleter().delete_all("local")
        monkeypatch.setenv("OPTI_ENCRYPTION_KEY", key)
        world.encrypted(True)
        kept, gone = store.create_conversation(title="kept").id, store.create_conversation(title="gone").id
        _seed(rj, kept)
        _seed(rj, gone)
        sealed = _lines(world)
        rj.reset()
        monkeypatch.delenv("OPTI_ENCRYPTION_KEY")
        store.delete_conversation(gone)
        seen["stamps"] = dict(rj._FORGOTTEN)  # no run of it was open: nothing to stamp
        ledger = (world.journal / ".forgotten").read_bytes() if (world.journal / ".forgotten").exists() else b""
        seen["no key"] = (_lines(world) == sealed, ledger)
        monkeypatch.setenv("OPTI_ENCRYPTION_KEY", key)
        seen["read with the key"] = (_ids(rj), _lines(world) == sealed)
        with (world.journal / _DAY).open("ab") as handle:
            handle.write(_sealed_elsewhere('{"journal": 1, "kind": "run"}') + b"\n")
        rj.tend()
        seen["first pass with the key"] = (_ids(rj), len(_lines(world)), (world.journal / ".forgotten").exists())
        skewed = store.create_conversation(title="skewed").id
        clock["now"] = _NOW + timedelta(days=1)  # a clock a day ahead files its run under tomorrow
        _seed(rj, skewed)
        clock["now"] = _NOW  # set right, then the conversation deleted with no key
        rj.reset()
        monkeypatch.delenv("OPTI_ENCRYPTION_KEY")
        store.delete_conversation(skewed)
        skew_day = _ledger_day(world, skewed)
        monkeypatch.setenv("OPTI_ENCRYPTION_KEY", key)
        shown = [f"run-{skewed}" in _ids(rj)]
        rj.tend()
        shown.append(f"run-{skewed}" in _ids(rj))
        seen["skewed clock"] = (skew_day, shown, _lines(world, "2026-10-17.jsonl"))
        wait, rj.GESTURE_WAIT_S = rj.GESTURE_WAIT_S, 0.01
        with rj._LOCK:  # the journal busy past the deletion's wait
            busy = rj.forget_conversation("busy")
        rj.GESTURE_WAIT_S = wait
        seen["busy"] = (busy, _ledger_day(world, "busy"))
        rj.tend()
        rj.reset()
        monkeypatch.delenv("OPTI_ENCRYPTION_KEY")
        note = rj._note_forgotten

        def unwritable(fd, entry, day):
            raise OSError(28, "No space left on device")

        rj._note_forgotten = unwritable
        rj.forget_conversation("held in memory")
        rj._note_forgotten = note
        held_first = _ledger_day(world, "held in memory")
        rj.tend()  # with no key it cannot apply it: it writes it down
        seen["written down"] = (held_first, _ledger_day(world, "held in memory"))
        monkeypatch.setenv("OPTI_ENCRYPTION_KEY", key)
        rj.tend()
        damaged = store.create_conversation(title="damaged").id
        _seed(rj, damaged)
        with (world.journal / ".forgotten").open("ab") as handle:
            handle.write(f"{hashlib.sha256(damaged.encode()).hexdigest()} 0000-00-00\n".encode("ascii"))
        seen["damaged day"] = f"run-{damaged}" in _ids(rj)
        rj.tend()
        failing = store.create_conversation(title="failing").id
        _seed(rj, failing)
        read = rj._read

        def refusing(name, fd):
            if name.endswith(".jsonl"):
                raise OSError(5, "Input/output error")
            return read(name, fd)

        rj._read = refusing
        answered = rj.forget_conversation(failing)
        rj._read = read
        seen["refused"] = (answered, f"run-{failing}" in _ids(rj), len(_lines(world)))
        rj.tend()
        seen["applied at the next pass"] = (len(_lines(world)), (world.journal / ".forgotten").exists())
        full = store.create_conversation(title="full disk").id
        _seed(rj, full)
        clock["now"] = _NOW - timedelta(days=1)  # a day that holds no run of it, and a line to drop
        _seed(rj, "the day before")
        clock["now"] = _NOW
        with (world.journal / "2026-10-15.jsonl").open("ab") as handle:
            handle.write(_sealed_elsewhere('{"journal": 1, "kind": "run"}') + b"\n")
        failed = rj.failures().get("forget_failed", 0)
        rewrite = rj._rewrite

        def no_space(name, lines, fd):
            raise OSError(28, "No space left on device")

        rj._rewrite = no_space
        rj.forget_conversation(full)
        rj._rewrite = rewrite
        seen["full disk"] = (_ids(rj), rj.failures().get("forget_dropped_day"),
                             rj.failures().get("forget_failed", 0) - failed)
        world.journal.rename(tmp_path / "aside")
        world.journal.symlink_to(tmp_path / "aside")
        refused = world.udm.UserDataDeleter().delete_all("local")
        world.journal.unlink()
        (tmp_path / "aside").rename(world.journal)
        seen["refused wipe"] = (refused["run_journal"], [line for line in refused["not_covered"]
                                                         if "run journal" in line])
        _seed(rj, "exported")
        exported = world.udm.UserDataExporter().export("local")
        wiped = world.udm.UserDataDeleter().delete_all("local")["run_journal"]
        seen["wiped"] = (wiped, world.journal.exists())
        _seed(rj, "again")
        manager = world.wipe.ConversationWipeManager()
        manager.wipe_all(purge_disk=True)
        seen["purged"] = (manager.run_journal_wiped, world.journal.exists())
    finally:
        world.restore()
    rest = sorted(f"run-{cid}" for cid in (here + "0", "none"))
    assert seen["here"] == sorted(rest + [f"run-{peer}", f"run-{peer}-t1", f"run-{peer}-v"]), seen["here"]
    assert seen["peer"] == (True, {"conversation": {"id": peer}}, rest, None), (
        "a peer's tombstone names its conversation, which goes with its runs and sub-runs", seen["peer"])
    assert seen["stamps after the runs"] == {} and seen["stamps"] == {}, (
        "a deletion stamps only a conversation with a run open, and no stamp outlives the runs it was for",
        seen["stamps after the runs"], seen["stamps"])
    for gesture, (written, kept_ids) in seen["flights"].items():
        wiped = gesture.endswith("wipe")
        assert written == [False, not wiped, True] and kept_ids == sorted(
                [f"after-{gesture}"] + ([] if wiped else [f"none-{gesture}"])), (
            gesture, "a run going is never written, one started after is; a deletion stops its own alone",
            seen["flights"][gesture])
    assert seen["no key"] == (True, f"{hashlib.sha256(gone.encode()).hexdigest()} 2026-10-16\n".encode()), (
        "no key: every sealed line stays, and the ledger holds the hash and the day, no identifier",
        seen["no key"])
    assert seen["read with the key"] == ([f"run-{kept}"], True), ("a read skips the conversation, changes nothing",
                                                                  seen["read with the key"])
    assert seen["first pass with the key"] == ([f"run-{kept}"], 1, False), (
        "the first pass with the key applies the ledger, a line it cannot open dropped with it",
        seen["first pass with the key"])
    assert seen["skewed clock"] == ("2026-10-17", [False, False], []), (
        "a run a clock ahead filed under tomorrow: its deletion waits under tomorrow, and nothing of it is read",
        seen["skewed clock"])
    assert seen["busy"] == (-1, "2026-10-17"), ("a deletion that finds the journal busy waits under tomorrow",
                                               seen["busy"])
    assert seen["written down"] == (None, "2026-10-16"), (
        "a deletion held in memory alone is written down by the next pass that cannot apply it",
        seen["written down"])
    assert seen["damaged day"] is False, "a ledger day that cannot be read hides its conversation's runs"
    assert seen["refused"] == (-1, False, 2), ("a deletion that fails answers -1, hidden meanwhile", seen["refused"])
    assert seen["applied at the next pass"] == (1, False), ("a deletion that failed is applied at the next pass",
                                                        seen["applied at the next pass"])
    assert seen["full disk"] == (["run-the day before"], 1, 1), (
        "a day that cannot be rewritten goes whole when it held a run of the deletion, and stays otherwise",
        seen["full disk"])
    assert seen["refused wipe"][0]["wiped"] is False and seen["refused wipe"][1] == [
        "run journal (not wiped: unsafe_path)"], ("a refused wipe says so", seen["refused wipe"])
    assert seen["wiped"] == ({"wiped": True, "files": 2}, False), seen["wiped"]
    assert seen["purged"] == ({"wiped": True, "files": 1}, False), seen["purged"]
    text = json.dumps(exported, default=str)
    assert "typed in" not in text and "run-" not in text, "the export carries no record"
    assert any("run journal" in line for line in exported["export_metadata"]["not_covered"]), exported
    assert "run_journal" not in source("backup_manager.py").read_text(encoding="utf-8"), "no backup reads it"


# ---------------------------------------------------------------------------
# RJ12 -- the reader
# ---------------------------------------------------------------------------
def _journal_window(tmp_path):
    """The journal alone, over ``tmp_path``'s data directory, encryption off."""
    data = tmp_path / "data"
    data.mkdir()
    config = types.ModuleType("opti_oignon.config")
    config.DATA_DIR = data
    loaded, restore = isolate(
        targets={_ALLOW: source("agent", "allowlists.py"), _RJ: source("run_journal.py")},
        seeded={"opti_oignon.config": config, "opti_oignon.security_mode": _Machine().module(),
                _ENC: _encryption_off()},
        packages=("opti_oignon.agent",))
    try:
        rj = loaded[_RJ]
        settings = tmp_path / "run_journal.yaml"
        settings.write_text("enabled: true\n", encoding="utf-8")
        rj.CONFIG_FILE = settings
    except BaseException:
        restore()
        raise
    return rj, restore, data / "run_journal"


def _reading_rule(lines):
    """The reading rule of a reader of the bench's trace, format 1, copied: each call labelled by its run."""
    calls = []
    for line in lines:
        if line.get("format") != 1:
            raise SystemExit(f"a trace line of format {line.get('format')!r}, not 1")
        previous = "<start>"
        for call in line["calls"]:
            label = "clean" if line["run"] == "alone" else ("attacker" if call["injection_steps"] else "other")
            calls.append((line["suite"], line["task"], line["typed"], call["tool"], label, line["persona"], previous))
            previous = call["tool"]
    return calls


def test_rj12_the_reader_orders_by_start_counts_what_it_skips_and_reads_as_the_bench_trace(tmp_path):
    rj, restore, journal = _journal_window(tmp_path)
    clock = {"now": _NOW}
    held = {"effect": "sandbox", "labels": {"path": "unendorsed"}, "read": ["view"], "context": ["tool"]}
    try:
        rj._now = lambda: clock["now"]
        call = SimpleNamespace(name="view", arguments={"path": "a.txt"}, source="native")
        ran = SimpleNamespace(executed=True, reason="executed", provenance={"effect": "sandbox", "labels": {}})
        opened = {}
        for run_id, when in (("x", datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc)),
                             ("y", datetime(2026, 10, 16, 10, 30, tzinfo=timezone.utc)),
                             ("p", datetime(2026, 10, 15, 23, 59, tzinfo=timezone.utc)),
                             ("q", datetime(2026, 10, 16, 0, 1, tzinfo=timezone.utc))):
            clock["now"] = when
            opened[run_id] = rj.open_run(run_id=run_id, typed=f"typed {run_id}", mode="daily", model="m",
                                         trust=SimpleNamespace(observe=lambda event: dict(held)))
            opened[run_id].on_call({"scope": "run", "round": 1, "call": call, "result": ran, "context": ["tool"]})
        clock["now"] = _NOW
        for run_id in ("y", "x", "q", "p"):
            assert opened[run_id].finish(SimpleNamespace(stop_reason="done", rounds=1, verifier=None)), run_id
        written = [json.loads(line)["run_id"] for line in (journal / _DAY).read_bytes().split(b"\n") if line]
        with (journal / _DAY).open("ab") as handle:
            handle.write(b"not a record\nJS1:AAAA\n{\"journal\": 1, \"kind\"")
        order = [record["run_id"] for record in rj.iter_runs()]
        counted = rj.failures()
        lines = [json.loads(json.dumps(line)) for line in rj.trace_lines()]
    finally:
        restore()
    assert written == ["y", "x", "q"], ("the file holds the runs as they ended", written)
    assert order == ["p", "q", "x", "y"], ("ordered by start, day by day", order)
    assert counted == {"torn": 1, "unreadable": 2}, counted
    read = _reading_rule(lines)
    assert [(suite, task, typed, tool, label, persona) for suite, task, typed, tool, label, persona, _ in read] == [
        ("journal", run_id, f"typed {run_id}", "view", "clean", "user") for run_id in ("p", "q", "x", "y")], read
    assert [call["trust"] for line in lines for call in line["calls"]] == [held] * 4, "the trust rides each call"
    assert {line["run"] for line in lines} == {"alone"} and {line["mode"] for line in lines} == {"daily"}, lines


# ---------------------------------------------------------------------------
# RJ13 -- the bench's calls carry the trust a journaled run's do
# ---------------------------------------------------------------------------
_BENCH_SCRIPT = [
    [("view", {"path": "report.txt"}), ("manage_memory", {"action": "list"})],
    [("web_search", {"query": "Summarize report.txt for me."}), ("bash", {"command": "wc -l report.txt"})],
    [("manage_notes", {"action": "make", "title": "Summary", "body": "Short."})],
    "Done.",
]
_DELEGATING = {
    "run": [[("view", {"path": "report.txt"})],
            [("task", {"description": "count", "prompt": "Count the lines of report.txt."})], "Done."],
    "task": [[("bash", {"command": "wc -l report.txt"})], "Twelve lines."],
}


def _bench_window(tmp_path):
    """The injection bench and everything a run stands on, with the journal and the agent route beside them."""
    data = tmp_path / "data"
    data.mkdir()
    config = types.ModuleType("opti_oignon.config")
    config.DATA_DIR = data
    targets = {name: source(*parts) for name, parts in _AGENT_TARGETS}
    targets.update({"opti_oignon.agent.skills": source("agent", "skills.py"),
                    _BENCH: source("agent_eval", "injection.py"), _RJ: source("run_journal.py"),
                    _ROUTES: source("api", "routes_agent.py")})
    loaded, restore = isolate(targets=targets, seeded={"opti_oignon.security_mode": _Machine().module(),
                                                       "opti_oignon.config": config, _ENC: _encryption_off()},
                              packages=("opti_oignon.memory", "opti_oignon.agent", "opti_oignon.agent_eval",
                                        "opti_oignon.api"))
    try:
        policy = tmp_path / "provenance.yaml"
        policy.write_text("network_daily: free\n", encoding="utf-8")
        loaded[_PROV].POLICY_FILE = policy
        settings = tmp_path / "run_journal.yaml"
        settings.write_text("enabled: true\n", encoding="utf-8")
        loaded[_RJ].CONFIG_FILE = settings
        loaded[_RJ]._now = lambda: _NOW
    except BaseException:
        restore()
        raise
    return loaded, restore


def _both_ways(loaded, script, task, conversation_id):
    """``script`` run by the bench and through the agent route, each over a world of its own: the bench's
    outcome, and the route's records of the run by kind."""
    bench, loop, tools, routes, pw = (loaded[_BENCH], loaded[_LOOP], loaded[_TOOLS], loaded[_ROUTES], loaded[_PW])
    suite = bench.load_suite()
    script_copy = {who: list(steps) for who, steps in script.items()} if isinstance(script, dict) else list(script)
    outcome = bench.run_once(bench._World(suite), task, _Model(script_copy, loop.VERIFIER_PROMPT),
                             mode="daily", policy="free", responder="approve")
    world = bench._World(suite)
    tools.reset_tool_registry()
    tools._REGISTRY = bench._bench_registry(world)
    pw.set_pending_store(world.queue)
    routes.reset_run_manager()
    manager = routes.get_run_manager()
    script_copy = {who: list(steps) for who, steps in script.items()} if isinstance(script, dict) else list(script)
    assert manager.start(task, model_client=_Model(script_copy, loop.VERIFIER_PROMPT), mode="daily",
                         sandbox=world.sandbox, approval_fn=bench._Person(True), include_memory=True,
                         memory_provider=world.memory_block, consult=False, registry=world.skills,
                         turn_origin="typed", conversation_id=conversation_id) == {"started": True}
    manager.join(timeout=20)
    records = {record["kind"]: record for record in loaded[_RJ].iter_runs()
               if record["conversation_id"] == conversation_id}
    return outcome, records


def test_rj13_the_benchs_calls_carry_the_trust_a_journaled_runs_do(tmp_path):
    loaded, restore = _bench_window(tmp_path)
    try:
        task = "Summarize report.txt for me."
        outcome, records = _both_ways(loaded, _BENCH_SCRIPT, task, "plain")
        delegated, delegating = _both_ways(loaded, _DELEGATING, task, "delegating")
        routes, tools = loaded[_ROUTES], loaded[_TOOLS]
        tables = ((routes._READING_TOOLS, dict(routes._READ_ACTIONS)),
                  (tools.READING_TOOLS, dict(tools.READ_ACTIONS)))
    finally:
        restore()
    journaled = [call["trust"] for call in records["run"]["calls"]]
    assert [name for name, _arguments in outcome.emitted] == [call["tool"] for call in records["run"]["calls"]]
    assert outcome.trust == journaled and len(journaled) == 5, (outcome.trust, journaled)
    assert journaled[2]["read"] == ["view", "manage_memory"] and "tool" in journaled[2]["context"], (
        "witness: the trust read what ran before", journaled[2])
    calls = delegating["run"]["calls"] + delegating["task"]["calls"]
    assert [name for name, _arguments in delegated.emitted] == [call["tool"] for call in calls] == [
        "view", "task", "bash"], (delegated.emitted, calls)
    assert delegated.trust == [call["trust"] for call in calls], ("a run that delegates: every call carries its "
                                                                  "trust", delegated.trust)
    assert tables[0] == tables[1], ("the route counts reads by the agent's tables", tables)


# ---------------------------------------------------------------------------
# RJ14 -- the server tends the journal with nothing else to do
# ---------------------------------------------------------------------------
def test_rj14_the_server_tends_the_journal_with_nothing_else_to_do(tmp_path):
    import test_startup_boot_guard_contracts as boot

    harness, restore = boot._real_app_harness()
    import opti_oignon.run_journal as rj

    saved = (rj.DATA_DIR, rj.CONFIG_FILE, rj.PRUNE_INTERVAL_S, rj._now)
    data = tmp_path / "data"
    journal = data / "run_journal"
    journal.mkdir(parents=True, mode=0o700)
    paused = tmp_path / "run_journal.yaml"
    paused.write_text("enabled: false\n", encoding="utf-8")
    found, later = journal / "2026-09-01.jsonl", journal / "2026-09-02.jsonl"
    rj.DATA_DIR, rj.CONFIG_FILE, rj._now = data, paused, (lambda: _NOW)
    rj.reset()
    undo = boot._install_real_stub_checks(harness["sc"], failing=None)

    async def booted():
        async with harness["app"].router.lifespan_context(harness["app"]):
            return found.exists()  # the first line inside: the boot's pass has run, the hourly one not yet

    async def served():
        async with harness["app"].router.lifespan_context(harness["app"]):
            later.write_bytes(b"")  # past the retention, planted once the server is up
            for _ in range(200):
                await asyncio.sleep(0.05)
                if not later.exists():
                    return True
            return False

    try:
        found.write_bytes(b"")  # past the retention, there before the server starts
        rj.PRUNE_INTERVAL_S = 3600.0
        kept_at_boot = asyncio.run(booted())
        rj.PRUNE_INTERVAL_S = 0.05
        gone = asyncio.run(served())
        left = sorted(path.name for path in journal.iterdir())
    finally:
        rj.DATA_DIR, rj.CONFIG_FILE, rj.PRUNE_INTERVAL_S, rj._now = saved
        rj.reset()
        undo()
        restore()
    assert kept_at_boot is False, "a day past the retention found at boot is gone once the server has started"
    assert gone and left == [], ("a day past the retention goes while the server runs, with nothing else to do",
                                 gone, left)


# ---------------------------------------------------------------------------
# RJ15 -- the security settings the journal reads are written whole
# ---------------------------------------------------------------------------
_MODE = "opti_oignon.security_mode"
_SETTINGS_TEXT = "encryption:\n  enabled: true\nsecurity_mode: bulbe\n"


def _replaced(settings, write):
    """``write`` run over ``settings``, found at mode 0640: whether the file was replaced, its mode, what it says,
    and the names beside it."""
    import yaml

    settings.write_text(_SETTINGS_TEXT, encoding="utf-8")
    os.chmod(settings, 0o640)
    inode = settings.stat().st_ino
    write()
    return (settings.stat().st_ino != inode, stat.S_IMODE(settings.stat().st_mode),
            yaml.safe_load(settings.read_text(encoding="utf-8")),
            sorted(path.name for path in settings.parent.iterdir()))


def test_rj15_the_security_settings_the_journal_reads_are_written_whole(tmp_path):
    by_mode_dir, by_route_dir = tmp_path / "mode", tmp_path / "route"
    by_mode_dir.mkdir()
    by_route_dir.mkdir()
    loaded, restore = isolate(targets={_MODE: source("security_mode.py")})
    try:
        mode = loaded[_MODE]
        mode._SECURITY_YAML = by_mode_dir / "security.yaml"
        by_mode = _replaced(mode._SECURITY_YAML, lambda: mode._write_yaml_mode("daily"))
    finally:
        restore()
    loaded, restore = isolate(targets={_SECURITY: source("api", "routes_security.py")}, packages=("opti_oignon.api",))
    try:
        routes = loaded[_SECURITY]
        routes._SECURITY_YAML_PATH = str(by_route_dir / "security.yaml")
        by_route = _replaced(by_route_dir / "security.yaml", lambda: routes.update_security_config(
            routes.SecurityConfigUpdate(headers={"x_frame_options": "DENY"})))
    finally:
        restore()
    assert by_mode == (True, 0o640, {"encryption": {"enabled": True}, "security_mode": "daily"},
                       ["security.yaml"]), ("the security mode replaces the file whole, its mode kept", by_mode)
    assert by_route == (True, 0o640, {"encryption": {"enabled": True}, "security_mode": "bulbe",
                                      "headers": {"x_frame_options": "DENY"}}, ["security.yaml"]), (
        "the security routes replace the file whole, its mode kept", by_route)


def test_rj16_the_directory_comes_from_the_configuration_module(tmp_path):
    """RJ16: with no directory given, the journal lives in the configuration module's data directory.

    The package exports ``config`` as the settings object, which holds no
    DATA_DIR, beside the module of the same name that does; the window
    binds the seeded module on the stand-in package, so the package's own
    shape is laid back on it here before the journal looks for its home.
    """
    package = source("__init__.py").read_text(encoding="utf-8")
    assert '"config": (".config", "config")' in package, (
        "the package exports config as the settings object", "the shape this contract lays back changed")
    rj, restore, directory = _journal_window(tmp_path)
    try:
        rj.DATA_DIR = None
        sys.modules["opti_oignon"].config = SimpleNamespace(settings="the package's settings object, no DATA_DIR")
        assert rj._base() == (tmp_path / "data").resolve(), (
            "the journal's home is the configuration module's data directory", rj._base())
        assert not directory.exists(), ("finding its home creates nothing", sorted(os.listdir(tmp_path / "data")))
    finally:
        restore()
