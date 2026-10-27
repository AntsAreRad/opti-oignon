#!/usr/bin/env python3
"""The chat's tool path with the onion memory off, captured case by case.

With the onion memory off, what the chat's agentic path hands every seam on
its way through the tools -- each request to the backend with its tool
schemas, each admission the backend's gate asks of the resource governor,
each question put to the approval hook, each row the conversation store is
asked to keep, the calls it remembers for the next turn, every chunk it
yields and what it writes into its run -- is held byte for byte by the
golden contracts (``tests/test_tool_path_off_golden_contracts.py``). This
module captures it, from a matrix of cases, each in a window of its own
opened through the shared isolation window:

  * the agentic executor, the tool executor and the chat executor as they
    are, with the structured-output engine, the tool-calling primitives, the
    response hygiene, the capability manifest, the provenance gate and the
    typed units it endorses with, the untrusted-data envelope and the
    deduplicator, the execution pipelines' runner and the step recorder;
  * the tool registry's own class, holding the built-in definitions with
    their names, descriptions and parameters as they are, each handler
    scripted, its requirements cleared and the tool enabled;
  * the Ollama backend's own heads over a scripted client, behind a scripted
    registry: each request is recorded as the path hands it to a head, and
    the head then asks the governor's gate as it always does;
  * where a case seats a governor, the resource governor's own module -- its
    gate, its ticket helpers and its typed refusal as they are -- serving a
    recording governor as its own. Its decisions carry no ``load_expected``,
    ``num_ctx`` or ``keep_alive``: the gate's load accounting never runs, and
    no admitted context or keep-alive reaches a request, so what the
    contracts read of budgets holds for this governor only;
  * every other module the path reaches as a stand-in that answers fixed
    inputs -- the configuration, the machine's mode (Daily), the
    conversation store, the caches, the network check, the ledger, the
    capture, the librarian (off, and every name only an onion turn reaches a
    tripwire), the web search engine where a case searches, the reasoning,
    consensus and self-correction engines -- and a clock, a uuid source and
    a calendar that stand still.

The turn is the chat route's: the keywords it hands the agentic executor or
the pipeline runner, a turn carrying the claim of what the user typed, its
way to ask (``demand``) and a step recorder whose every frame is traced, and
a status line traced wherever one is said. In Daily the route hands no
per-call approval hook and asks only through the turn; one case arms Bulbe's
per-call seam (``approval_fn``) with a callable of its own, so a swap of the
two channels shows. The calls the path makes at its own seams
-- ``stream_with_tools``, ``execute_with_tools`` and the chat executor's
``execute`` -- are traced by the names of the keywords they are handed.

Each case records, turn by turn: every request the backend received (the
digest of the whole request, its small fields as they are, the names and
digest of the tool schemas it offered, the digest of its schema, its think
flag, each message as its role, length and digest, and the digest of what
the head sent the engine), the gate's admissions, the approval hook's calls,
the store's writes (role, origin, content, context and the digest of the
lineage), the tool calls the executor remembers after the turn, the
non-empty yields, what the turn wrote into its run, and the ordered trace of
the seams it reached. Long strings are kept as their length and digest. A
case also records the facts that show its trigger fired.

The capture runs in child processes under two hash seeds, each covered by
the data firewall the way every child of the suite is. Bytecode the children
compile goes to a directory under the system's temporary directory, never
into the tree. The golden file is written only on purpose:
``python3 tests/_tool_path_off_golden.py --write`` refuses a tree whose
sources differ from HEAD and seeds that disagree, writes the file and prints
the digest the contracts pin; ``--check`` compares and writes nothing, and a
missing file is a difference; ``--explain NAME`` prints every field of the
matching cases, each request whole, to read a difference the digests only
point at: run it on the change and on the tree before it to read both
sides. A test never writes the file.

Local-only (the public distribution ships no tests). Stdlib-only apart from
the windows it loads.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid as _uuid
from datetime import datetime as _datetime
from pathlib import Path
from types import SimpleNamespace

if __name__ == "__main__":
    # Run as a script, nothing it imports, nor any child it starts, leaves bytecode beside the sources.
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from _isolation import REPO, isolate, source  # noqa: E402
from _onion_off_golden import (  # noqa: E402
    StillClock,
    _caches,
    _memory,
    _module,
    _network,
    _plain,
    _reads_datetime,
    _web,
    canonical,
    render,
    sha,
    shape,
    short,
)
from _registry_bridge import StubRegistry  # noqa: E402

FIXTURE = TESTS / "executor_golden" / "v2" / "tool_path_off.json"
SEEDS = ("0", "1")
FORMAT = 1

# A model both the native tool-calling and the thinking heuristics name, one neither names (the format= path), and
# a smaller one of the first kind, the fast tool model a turn that asks to optimize routes its tool work to.
NATIVE, PLAIN, FAST = "qwen3:8b", "test-model:1b", "qwen3:1.7b"
# What each model tells the Ollama heads it can do, read before they tell a model not to think.
_CAPABILITIES = {NATIVE: ("completion", "tools", "thinking"), PLAIN: ("completion",),
                 FAST: ("completion", "tools", "thinking")}

_WRAPPER = "opti_oignon.agent.untrusted_context"
_LIBRARIAN = "opti_oignon.memory.librarian"
_GOVERNOR = "opti_oignon.resource_governor"
_BACKEND = "opti_oignon.inference_backend"
_STEPS = "opti_oignon.pipeline_step"

# The modules as they are, in the order they load: each may import one loaded before it. The governor's own module
# joins after the backend in a case that seats a governor, so the backend and the executor bind its gate and
# helpers.
_TARGETS = (
    (_WRAPPER, ("agent", "untrusted_context.py")),
    ("opti_oignon.context_dedup", ("context_dedup.py",)),
    ("opti_oignon.memory.probes", ("memory", "probes.py")),
    ("opti_oignon.pending_writes", ("pending_writes.py",)),
    ("opti_oignon.provenance", ("provenance.py",)),
    (_BACKEND, ("inference_backend.py",)),
    ("opti_oignon.tool_calling", ("tool_calling.py",)),
    ("opti_oignon.response_hygiene", ("response_hygiene.py",)),
    ("opti_oignon.tool_registry", ("tool_registry.py",)),
    ("opti_oignon.capability_manifest", ("capability_manifest.py",)),
    ("opti_oignon.structured_output", ("structured_output.py",)),
    ("opti_oignon.tool_executor", ("tool_executor.py",)),
    ("opti_oignon.executor", ("executor.py",)),
    ("opti_oignon.agentic_executor", ("agentic_executor.py",)),
    (_STEPS, ("pipeline_step.py",)),
    ("opti_oignon.pipelines", ("pipelines.py",)),
)
# The modules that read the clock on the path: each is handed the still clock.
_STILLED = ("opti_oignon.executor", "opti_oignon.agentic_executor", "opti_oignon.tool_executor",
            "opti_oignon.structured_output", _BACKEND, _STEPS)

# The built-in tools, in the order the registry declares them: the set every turn offers.
_TOOLS = ("web_search", "execute_code", "read_file", "write_file", "list_files")
# What each scripted handler returns unless its case queues other results.
_RESULTS = {
    "web_search": "No results.",
    "execute_code": "1.0",
    "read_file": "Beds: onions in the east beds, garlic by the wall, shallots in the west bed.",
    "write_file": "Wrote sow.py.",
    "list_files": "notes.txt\nsow.py",
}
_TRACEBACK = ("Traceback (most recent call last):\n  File \"<sandbox>\", line 1, in <module>\n"
              "ZeroDivisionError: division by zero")
# The pipelines that save their own answer, never through the chat executor: what the capture and the librarian
# must not be offered with the onion off.
PIPELINE_OWNED = ("tools", "think_tools", "reasoning", "consensus", "self_correct")

# How long the ledger waits for a held ticket's release before it records anyway.
_RELEASE_WAIT_S = 2.0


# ---------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------
# The fields of an event kept as they are, beside the digest of all of them:
# enough to read the order and the outcome of a turn at a glance.
_NAMED = ("admitted", "caller", "chunk", "effect", "hit", "keys", "message", "online", "origin", "outcome",
          "positional", "request", "requested_ctx", "role", "state", "success", "think", "tool")


class Trace:
    """The seams a turn reached, in order: one list per turn.

    Each event is its kind, the fields of ``_NAMED`` it carries, and the
    digest of all its fields as ``shape`` keeps them; ``explain`` keeps the
    fields themselves instead. Events come from the turn's thread and from
    the executor's stream thread, which hands its chunks on through a queue
    after it has said everything it says, so their order is the code's.
    ``released`` is set while no admission ticket is held: the ledger
    stand-in and the caller's yields wait for it before they are traced,
    since the stream thread releases its ticket after its last chunk is
    queued.
    """

    def __init__(self, explain=False):
        self.turns = [[]]
        self._explain = explain
        self._lock = threading.Lock()
        self.released = threading.Event()
        self.released.set()

    def add(self, kind, **fields):
        if not fields:
            event = [kind]
        else:
            shaped = shape(fields)
            if self._explain:
                event = [kind, shaped]
            else:
                kept = {name: shaped[name] for name in _NAMED if name in shaped}
                if len(kept) < len(shaped):
                    kept["fields"] = sha(canonical(shaped))[:16]
                event = [kind, kept]
        with self._lock:
            self.turns[-1].append(event)

    def next_turn(self):
        with self._lock:
            self.turns.append([])


# ---------------------------------------------------------------------------
# A world: what one case's window holds, and what it records turn by turn
# ---------------------------------------------------------------------------
class World:
    """One case's seams and records: each record is a list per turn, the current turn's last."""

    def __init__(self, case, explain=False):
        self.case, self.options, self.explain = case, case["options"], explain
        self.trace = Trace(explain)
        self.requests, self.admissions, self.approvals, self.writes, self.engine = [[]], [[]], [[]], [[]], [[]]
        self.run, self.turn, self.yields = None, 0, []
        self._queued = {name: list(results) for name, results in self.options.get("results", {}).items()}

    def next_turn(self):
        for kept in (self.requests, self.admissions, self.approvals, self.writes, self.engine):
            kept.append([])
        self.trace.next_turn()

    def stop_at(self, where):
        """Raise the turn's stop where the stand-in calling this stands, on the turn the case names."""
        if self.options.get("stop", {}).get(where) == self.turn:
            self.trace.add("stop")
            self.run.stop.set()

    def record(self, call, model, messages, options, given):
        """One request as the path hands it to a head: whole in its digest, its small fields as they are."""
        whole = {"call": call, "model": model, "messages": [dict(m) for m in messages or []], "options": options,
                 "given": given}
        digest = sha(canonical(_plain(whole)))
        rest = dict(options or {})
        tools, schema = rest.pop("tools", None), rest.pop("schema", None)
        entry = {
            "call": call, "sha256": digest, "model": model, "given": sorted(given),
            "keep_alive": given.get("keep_alive"), "think": given.get("think"), "options": shape(rest),
            "tools": None if tools is None else [[_schema_name(item) for item in tools],
                                                 sha(canonical(_plain(tools)))[:16]],
            "schema": None if schema is None else short(canonical(_plain(schema))),
            "messages": [_message_form(message) for message in messages or []],
            "engine": None,
        }
        if self.explain:
            entry["whole"] = _plain(whole)
        self.requests[-1].append(entry)
        self.trace.add(call, request=digest[:16], think=given.get("think"))

    def seam(self, name, method):
        """``method`` as the path calls it, each call traced by the names of its keywords and its positional
        count before it runs; what it returns, a generator included, is handed back as it is."""

        def call(*args, **kwargs):
            self.trace.add("call." + name, keys=sorted(kwargs), positional=len(args))
            return method(*args, **kwargs)

        return call

    def handler(self, name):
        def run(**arguments):
            self.trace.add("tool", tool=name, arguments=arguments)
            queued = self._queued.get(name)
            result = queued.pop(0) if queued else _RESULTS[name]
            self.stop_at("tool")
            return result

        run.__name__ = f"scripted_{name}"
        return run

    def _asked(self, channel, tool_name, arguments, labels, effect):
        self.approvals[-1].append({"channel": channel, "tool": tool_name, "arguments": shape(dict(arguments)),
                                   "labels": dict(labels or {}), "effect": effect})
        self.trace.add("ask." + channel, tool=tool_name, arguments=dict(arguments), labels=dict(labels or {}),
                       effect=effect)
        return True

    def demand(self, tool_name, arguments, labels=None, effect=None):
        """The turn's way to ask the user about one call, as the chat route sets it in every mode (``run.demand``):
        asked with each argument's label and the call's class; it allows."""
        return self._asked("demand", tool_name, arguments, labels, effect)

    def bulbe_approval(self, tool_name, arguments, labels=None, effect=None):
        """Bulbe's per-call approval seam (``approval_fn``), which the route arms when its security policy asks for
        every call: a callable of its own, traced apart from ``demand``; it allows."""
        return self._asked("approval", tool_name, arguments, labels, effect)

    def capture(self, conversation_id, messages):
        self.trace.add("capture", conversation=conversation_id, messages=messages)

    def step(self, frame):
        """The sink of the turn's step recorder: each frame as the recorder sends it."""
        self.trace.add("step", state=frame.get("state"), frame=frame)

    # The callbacks the chat route hands the path, each traced as it is called.
    def on_status(self, message):
        self.trace.add("status", message=message)

    def on_tool_call(self, call):
        self.trace.add("hook.tool_call", tool=call.tool_name, success=call.success, call=vars(call))

    def on_reasoning_step(self, step):
        self.trace.add("hook.reasoning_step", step=step)

    def on_consensus_model(self, response):
        self.trace.add("hook.consensus_model", response=response)


def _schema_name(item):
    function = item.get("function") if isinstance(item, dict) else None
    return function.get("name") if isinstance(function, dict) else None


def _message_form(message):
    """A message as a record keeps it: its role, its content's length and digest, and its calls' when it has any."""
    form = [message.get("role"), *short(message.get("content", ""))]
    if "tool_calls" in message:
        form += short(canonical(_plain(message["tool_calls"])))
    return form


# ---------------------------------------------------------------------------
# The engine: a scripted client behind the Ollama heads
# ---------------------------------------------------------------------------
_PAST_SCRIPT = "[past the script]"


class ScriptedClient:
    """The client library the Ollama heads call, each request answered by the next reply of its case's script.

    A reply is a mapping: its ``content``, its ``calls`` (name and arguments,
    in order) and its ``thinking``. A stream answers its thinking first, then
    its content cut in two at a space past the middle, the last chunk marked
    done. ``show`` answers a model's capabilities. A request past the script
    is answered with fixed words and counted, so a path that asks once more
    shows by name instead of stopping the capture. Each request is kept whole
    for the facts, and its digest is written on the request the head serves.
    """

    def __init__(self, world, script):
        self._world, self.script, self.past = world, list(script), 0

    def module(self):
        return _module("ollama", chat=self.chat, show=self.show)

    def show(self, model):
        self._world.trace.add("engine.show", model=model)
        return {"capabilities": list(_CAPABILITIES.get(model, ("completion",)))}

    def chat(self, **kwargs):
        world = self._world
        world.engine[-1].append(kwargs)
        if world.requests[-1]:
            world.requests[-1][-1]["engine"] = sha(canonical(_plain(kwargs)))[:16]
        if self.script:
            reply = self.script.pop(0)
        else:
            self.past += 1
            reply = {"content": _PAST_SCRIPT}
        if kwargs.get("stream"):
            return iter(_chunks(reply))
        message = {"role": "assistant", "content": reply.get("content", "")}
        calls = [{"function": {"name": name, "arguments": dict(arguments)}} for name, arguments in reply.get("calls", ())]
        if calls:
            message["tool_calls"] = calls
        return {"message": message, "done": True}


def _chunks(reply):
    text = reply.get("content", "")
    cut = text.find(" ", len(text) // 2)
    pieces = [text] if cut <= 0 else [text[:cut], text[cut:]]
    chunks = [{"message": {"thinking": reply["thinking"], "content": ""}}] if reply.get("thinking") else []
    chunks += [{"message": {"content": piece}} for piece in pieces]
    chunks[-1]["done"] = True
    return chunks


def _recording_ollama(backend_module, world):
    """The Ollama backend's own heads, each request recorded as the path hands it, before the head asks the gate."""

    class RecordingOllama(backend_module.OllamaBackend):
        def generate(self, model, messages, options=None, **given):
            world.record("generate", model, messages, options, given)
            return super().generate(model, messages, options, **given)

        def stream(self, model, messages, options=None, **given):
            world.record("stream", model, messages, options, given)
            yield from super().stream(model, messages, options, **given)

    return RecordingOllama()


# ---------------------------------------------------------------------------
# Stand-ins, one per seam
# ---------------------------------------------------------------------------
def _config(model, preferences):
    """The configuration: the case's model for every task, and the user preferences it sets."""
    held = dict(preferences)
    settings = SimpleNamespace(get_model=lambda *a, **k: model, get_temperature=lambda *a, **k: 0.2,
                               get_user_preference=lambda key, default=None: held.get(key, default))
    return _module("opti_oignon.config", config=settings, get_model=lambda *a, **k: model)


class Store:
    """The conversation store as the agentic path and the executor read it, and what it was asked to keep.

    A row carries its role, content, origin and segments, and the context and
    lineage it was written with; a user row's label is derived from its parts,
    as the application's store derives it, and the readers of the turns hand
    the user's turns and the answers alone, as the application's do.
    ``get_messages`` hands every row, as the application's does: it is the
    read the tools pipeline makes for its context, where a stop can be raised.
    """

    def __init__(self, world, history=()):
        self._world = world
        self.rows = {conversation_id: [dict(row) for row in rows] for conversation_id, rows in history}

    def add_message(self, conv_id, role, content, model=None, metadata=None, *, origin="legacy", segments=(),
                    context=None, lineage=None):
        world = self._world
        parts = [list(segment) for segment in segments or ()]
        write = {"role": role, "origin": origin, "content": shape(content), "segments": parts, "model": model,
                 "context": None if context is None else list(context),
                 "lineage": None if lineage is None else short(canonical(list(lineage)))}
        if world.explain:
            write["lineage"] = None if lineage is None else list(lineage)
        world.writes[-1].append(write)
        world.trace.add("store.add", conversation=conv_id, role=role, content=content, model=model,
                        metadata=metadata, origin=origin, segments=parts, context=context, lineage=lineage)
        self.rows.setdefault(conv_id, []).append(
            {"role": role, "content": content, "origin": origin, "segments": parts,
             "context": None if context is None else list(context),
             "lineage": None if lineage is None else list(lineage)})
        return SimpleNamespace(role=role, content=content)

    def get_messages(self, conv_id):
        self._world.stop_at("read")
        return [SimpleNamespace(id=index + 1, role=row["role"], content=row["content"])
                for index, row in enumerate(self.rows.get(conv_id, []))]

    def _turns(self, conv_id):
        return [row for row in self.rows.get(conv_id, []) if row["role"] in ("user", "assistant")]

    def get_context_messages(self, conv_id, **kwargs):
        return [{"role": row["role"], "content": row["content"]} for row in self._turns(conv_id)]

    def get_labelled_context_messages(self, conv_id):
        wrapper = sys.modules[_WRAPPER]
        out = []
        for row in self._turns(conv_id):
            if row["role"] == "user":
                context, lineage = wrapper.user_turn_label(row["content"], row["origin"], row["segments"])
            elif row["context"] is not None:
                context, lineage = row["context"], row["lineage"] or []
            else:
                context, lineage = ["legacy"], []
            out.append(wrapper.labelled({"role": row["role"], "content": row["content"]}, context, lineage))
        return out

    def get_mirror_messages(self, conv_id):
        return [{"role": row["role"], "content": row["content"], "origin": row["origin"],
                 "segments": list(row["segments"])} for row in self._turns(conv_id)]

    def get_conversation(self, conv_id):
        return SimpleNamespace(id=conv_id, messages=self.get_context_messages(conv_id), metadata={})

    def update_conversation_metadata(self, conv_id, *args, **kwargs):
        self._world.trace.add("store.metadata", conversation=conv_id, **kwargs)

    def withdrawn_sources(self):
        return []


def _librarian(world):
    """The librarian with the onion off: its switch reads off and its write side is traced; each name only an onion
    turn reaches is a tripwire that says it was reached."""

    def tripwire(name):
        def reached(*args, **kwargs):
            world.trace.add("trip." + name)

        return reached

    def maybe_curate(conversation_id, messages, **kwargs):
        world.trace.add("curate", conversation=conversation_id, messages=messages, **kwargs)
        return True

    return _module(_LIBRARIAN, onion_enabled=lambda path=None: False, maybe_curate=maybe_curate,
                   memory_block=tripwire("memory_block"), plan_window=tripwire("plan_window"),
                   compose_window=tripwire("compose_window"), tail_room=tripwire("tail_room"))


def _ledger(world):
    """The context ledger: each record traced with every field the executor writes into it -- its request id, its
    model, its caller, its outcome, its duration and whatever else the exit adds -- once any held ticket is
    released, so the order of the two does not depend on the scheduler."""

    def record(**fields):
        world.trace.released.wait(_RELEASE_WAIT_S)
        world.trace.add("ledger", **fields)

    return {"opti_oignon.context_ledger": _module("opti_oignon.context_ledger",
                                                  get_context_ledger=lambda: SimpleNamespace(record=record))}


class Governor:
    """The governor the resource governor's own module serves as its own (its ``_governor``).

    The module's gate, its ticket helpers and its typed refusal run as they
    are: a ticket held for the model stands the gate down, and a ticketless
    call is admitted by its backstop, ``caller="direct"``, with the context
    its options request. Every admission they and the funnels ask is traced
    and kept, with the real ``admit``'s every argument; each decision is the
    module's own ``AdmissionDecision``; it admits the first ``admit_first``
    (every one when None) and refuses the rest. A ticket held or released is
    traced where the module tells its governor, ``note_held``.

    The caveats. Its decisions carry no ``load_expected``, ``num_ctx`` or
    ``keep_alive``, so the gate's load accounting never runs and no admitted
    context or keep-alive is sent: the contracts' no-budget rule holds for
    this governor only. ``invalidate_on_load`` is what the accounting calls
    for every admitted load, off the onion path too, so it is traced as the
    governor's own call, never as a tripwire. ``end_pending_load`` is reached
    only by an onion turn's early exit or by a split an engine cannot place,
    neither of which this window's Ollama heads make: it is a tripwire.
    """

    def __init__(self, world, admit_first=None):
        self._world, self._admit_first, self._count = world, admit_first, 0
        self.config = SimpleNamespace(enabled=True)
        self.decision = None

    def bind(self, module):
        """Serve ``module`` as its governor: its decisions are the module's own."""
        self.decision = module.AdmissionDecision
        module._governor = self

    def admit(self, model, requested_ctx=None, caller="chat", extra_models=None, digest=None, engine=None):
        self._count += 1
        admitted = self._admit_first is None or self._count <= self._admit_first
        self._world.admissions[-1].append({"caller": caller, "model": model, "requested_ctx": requested_ctx,
                                           "admitted": admitted, "extra_models": extra_models, "digest": digest,
                                           "engine": engine})
        self._world.trace.add("admit", model=model, requested_ctx=requested_ctx, caller=caller, admitted=admitted,
                              extra_models=extra_models, digest=digest, engine=engine)
        return self.decision(admitted=admitted, model=model, action="admit" if admitted else "refuse",
                             reason="" if admitted else "not enough memory", ticket_id=f"ticket-{self._count}",
                             caller=caller, requested_ctx=requested_ctx, shortfall_gb=None if admitted else 2.0)

    def note_held(self, decision):
        trace = self._world.trace
        if decision is None:
            trace.add("ticket.clear")
            trace.released.set()
        else:
            trace.released.clear()
            trace.add("ticket.set", action=decision.action, caller=decision.caller)

    def end_pending_load(self, *args, **kwargs):
        self._world.trace.add("trip.end_pending_load")

    def invalidate_on_load(self, model, num_ctx=None):
        self._world.trace.add("governor.invalidate_on_load", model=model, num_ctx=num_ctx)


# What the three engines the agentic path names as outside say, in their own protocols.
_ENGINE_STEPS = (("Count the beds", "There are three beds."), ("Match each crop", "Onions take the east beds."))
_REASONED = "Give the east beds to the onions and the wall bed to the garlic."
_ANSWERS = (("qwen3:8b", "The west bed."), ("gemma3:4b", "The west bed, by the wall."))
_SELECTED = "The west bed, by the wall."
_CORRECTED = "The three beds are the onion bed in the east, the garlic bed by the wall, and the shallot bed."


def _engines(world):
    """The reasoning, consensus and self-correction engines: each call traced, each answer fixed, each spoken in the
    engine's own protocol (its progress events and live callbacks first, then its steps, its result and its text)."""
    trace = world.trace

    class ReasoningEngine:
        available = True

        def execute_reasoning(self, question, strategy=None, model=None, on_step=None, should_stop=None,
                              on_progress=None):
            trace.add("reasoning", question=question, strategy=strategy, model=model,
                      progress=on_progress is not None)
            steps = [SimpleNamespace(index=index, label=label, content=content)
                     for index, (label, content) in enumerate(_ENGINE_STEPS)]
            if on_progress is not None:
                on_progress("run", name="decompose")
                on_progress("plan", total=len(steps), labels=[step.label for step in steps])
            for step in steps:
                if on_progress is not None:
                    on_progress("begin", index=step.index)
                if on_step is not None:
                    on_step(step)
                if on_progress is not None:
                    on_progress("end", index=step.index, state="done")
            result = SimpleNamespace(final_answer=_REASONED, steps=steps, strategy="decompose", metadata={})
            for step in steps:
                yield ("reasoning_step", step)
            yield ("reasoning_done", result)
            yield result.final_answer

    class ConsensusEngine:
        available = True

        def execute_consensus(self, query, models=None, strategy=None, system_prompt=None, on_model_done=None,
                              should_stop=None, on_progress=None):
            trace.add("consensus", query=query, models=models, strategy=strategy, system_prompt=system_prompt,
                      progress=on_progress is not None)
            answers = [SimpleNamespace(model=model, response=text, success=True) for model, text in _ANSWERS]
            if on_progress is not None:
                on_progress("query", total=len(answers))
            for answer in answers:
                if on_model_done is not None:
                    on_model_done(answer)
                if on_progress is not None:
                    on_progress("model_end", model=answer.model, ok=True)
            if on_progress is not None:
                on_progress("compare")
                on_progress("compared")
            result = SimpleNamespace(selected_response=_SELECTED, individual_responses=answers,
                                     strategy="majority_vote", metadata={})
            for answer in answers:
                yield ("consensus_model_done", answer)
            yield ("consensus_done", result)
            yield result.selected_response

    class SelfCorrectionEngine:
        available = True

        def execute_self_correction(self, user_message, response, model=None, should_stop=None):
            trace.add("self_correction", user_message=user_message, response=response, model=model)
            yield ("correction_step", {"iteration": 1, "compliance_score": 0.6, "quality_score": 0.8,
                                       "improvements": ["name the shallot bed"], "duration_ms": 0.0})
            for start in range(0, len(_CORRECTED), 50):
                yield _CORRECTED[start:start + 50]
            yield ("correction_done", SimpleNamespace(corrected_response=_CORRECTED, iterations=1))

    def named(name):
        return type(name, (), {})

    return {
        "opti_oignon.reasoning": _module(
            "opti_oignon.reasoning", ReasoningConfig=named("ReasoningConfig"), ReasoningEngine=ReasoningEngine,
            ReasoningResult=named("ReasoningResult"), ReasoningStep=named("ReasoningStep"),
            reasoning_engine=ReasoningEngine()),
        "opti_oignon.consensus": _module(
            "opti_oignon.consensus", ConsensusConfig=named("ConsensusConfig"), ConsensusEngine=ConsensusEngine,
            ConsensusResult=named("ConsensusResult"), ModelResponse=named("ModelResponse"),
            consensus_engine=ConsensusEngine()),
        "opti_oignon.self_correction": _module(
            "opti_oignon.self_correction", SelfCorrectionConfig=named("SelfCorrectionConfig"),
            SelfCorrectionEngine=SelfCorrectionEngine, SelfCorrectionResult=named("SelfCorrectionResult"),
            self_correction_engine=SelfCorrectionEngine()),
    }


class StillUuid:
    """``uuid`` as the executor and the governor's module read it: one fixed identifier, and any other name fails
    loudly."""

    def uuid4(self):
        return _uuid.UUID(int=1)

    def __getattr__(self, name):
        raise AttributeError(f"the still uuid does not answer uuid.{name}: still it here before it is read")


class StillDate(_datetime):
    """``datetime`` as the execution pipelines and the governor's module read it: the still clock's instant, as a
    naive date."""

    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1)


def _scripted_registry(registry_module, world):
    """The registry's own class holding the built-in definitions, their names, descriptions and parameters as they
    are, each handler scripted, its requirements cleared and the tool enabled; set where the capability manifest
    reads the registry."""
    registry = registry_module.ToolRegistry()
    for name in _TOOLS:
        built = registry_module.tool_registry.get(name)
        registry.register(dataclasses.replace(built, handler=world.handler(name), requires=[], enabled=True))
    registry_module.tool_registry = registry
    return registry


def _third_party(modules):
    """Seat third-party modules in the cache for one window; returns what puts the cache back."""
    saved = {name: sys.modules.get(name, _ABSENT) for name in modules}
    sys.modules.update(modules)

    def restore():
        for name, value in saved.items():
            if value is _ABSENT:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    return restore


_ABSENT = object()


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def _turn(question, conversation="conv-1", kind="agentic", **kwargs):
    return {"question": question, "conversation": conversation, "kind": kind, "kwargs": kwargs}


def _say(text, thinking=None):
    return {"content": text, **({"thinking": thinking} if thinking else {})}


def _call(*calls, content=""):
    return {"content": content, "calls": [(name, dict(arguments)) for name, arguments in calls]}


def _decision(**fields):
    return {"content": canonical(fields)}


def _case(name, turns, *, checks=(), **options):
    scripted = [_fact("past_script", 0), _fact("script_left", 0)]
    return {"name": name, "turns": tuple(turns), "checks": tuple(scripted + list(checks)), "options": options}


def _events(record, kind):
    return [event for turn in record["turns"] for event in turn["trace"] if event[0] == kind]


def _fired(kind, least=1):
    return (f"{kind} reached at least {least} time(s)", lambda record: len(_events(record, kind)) >= least)


def _absent(kind):
    return (f"{kind} never reached", lambda record: not _events(record, kind))


def _fact(key, expected):
    return (f"{key} is {expected!r}", lambda record: record["facts"].get(key) == expected)


def _calls(*turns):
    expected = [list(turn) for turn in turns]
    return (f"the requests are {expected}",
            lambda record: [[request["call"] for request in turn["requests"]] for turn in record["turns"]] == expected)


def _models(*models):
    return (f"the requests go to {list(models)}",
            lambda record: [request["model"] for turn in record["turns"] for request in turn["requests"]]
            == list(models))


def _ran(*tools):
    return (f"the handlers that ran are {list(tools)}",
            lambda record: [event[1]["tool"] for event in _events(record, "tool")] == list(tools))


def _yields(index, expected):
    return (f"turn {index + 1} yields {expected!r}",
            lambda record: record["turns"][index]["yields"] == shape(list(expected)))


def _saved(*rows):
    expected = [list(row) for row in rows]
    return (f"the rows written are {expected}",
            lambda record: [[write["role"], write["origin"]] for turn in record["turns"] for write in turn["writes"]]
            == expected)


def _nothing_saved():
    return ("nothing is written", lambda record: not any(turn["writes"] for turn in record["turns"]))


def _history(*calls):
    expected = [list(call) for call in calls]
    return (f"the remembered calls are {expected}",
            lambda record: [[call["tool_name"], call["success"]] for call in record["turns"][-1]["history"]]
            == expected)


def _result(index, key, expected):
    return (f"turn {index + 1} wrote {key}={expected!r} into its run",
            lambda record: record["turns"][index]["results"].get(key) == expected)


def _admitted(*admissions):
    expected = [list(item) for item in admissions]
    return (f"the admissions are {expected}",
            lambda record: [[item["caller"], item["requested_ctx"], item["admitted"]] for turn in record["turns"]
                            for item in turn["admissions"]] == expected)


def _callers(*admissions):
    expected = [list(item) for item in admissions]
    return (f"the admissions are, by caller and outcome, {expected}",
            lambda record: [[item["caller"], item["admitted"]] for turn in record["turns"]
                            for item in turn["admissions"]] == expected)


def _think_flags(*flags):
    return (f"the think flags handed are {list(flags)}",
            lambda record: [request["think"] for turn in record["turns"] for request in turn["requests"]]
            == list(flags))


def _offered(kind):
    """Every decision request offers the tools the way ``kind`` names: native schemas, or a format= schema."""
    def holds(record):
        decisions = [request for turn in record["turns"] for request in turn["requests"]
                     if request["options"].get("temperature") == 0.0]
        if kind == "tools":
            return bool(decisions) and all(r["tools"] and r["schema"] is None for r in decisions)
        return bool(decisions) and all(r["schema"] and r["tools"] is None for r in decisions)

    return (f"every decision offers the tools as {kind}", holds)


def _seams(*names):
    """The path's own seams called, in order, by name, each with the keywords the case's record holds."""
    return (f"the seams called are {list(names)}",
            lambda record: [event[0][len("call."):] for turn in record["turns"] for event in turn["trace"]
                            if event[0].startswith("call.")] == list(names))


_SAVED_PAIR = (("user", "typed"), ("assistant", "assistant+tool"))
_QUESTION = "What do the notes in notes.txt say about the east beds?"
_READ = ("read_file", {"path": "notes.txt"})
_LIST = ("list_files", {"path": "."})
_DONE = "That is all I need."
_FINAL = "I read notes.txt: the east beds hold the onions, and the garlic grows by the wall."
_ONE_CALL = (_call(_READ), _say(_DONE), _say(_FINAL))
_DIRECT = "Sow onions in late winter, as soon as the soil can be worked."
_NATIVE_TRANSCRIPT = {"tool_transcript": "native", "tool_transcript_models": [NATIVE]}
_NARRATION = ("I would keep it in `sow.py`:\n\n```python\nfor month in (\"February\", \"March\"):\n"
              "    print(month)\n```\n")
_THOUGHT = "Weighing the beds before I look at the notes."
_PHASE_ONE = "The sunniest bed should take the onions."
_TOOL_ANSWER = "The notes agree: the east beds hold the onions."
_WEIGH = "Weigh the beds for me, then check notes.txt before you answer."
_THINK_TOOLS = (_say(_PHASE_ONE, thinking=_THOUGHT), _call(_READ), _say(_DONE), _say(_TOOL_ANSWER))
_DRAFT = "Onion bed, garlic bed."
_NO_BACKEND = "Cannot generate response: no inference backend is registered."
_SECOND_DIRECT = "The onions are in the east beds, as the notes said."
_WEB_ANSWER = "Onions are sown from February, the results say."
_ELEVENTH = "Welcome back to the garden notebook."
_SYSTEM_ROW = "Synchronized from a peer device."
_PASTED = "Which bed suits shallots? Pasted note: shallots like the wall bed."
_TYPED_END = len("Which bed suits shallots?")
_FALLBACK = "I encountered an error generating the final response, but here are the tool results:"


def _row(role, content, fact=None):
    """A row of the eleven-row history: a typed user turn, or an answer written in sight of one memory."""
    if role == "user":
        return {"role": role, "content": content, "origin": "typed", "segments": [], "context": None,
                "lineage": None}
    return {"role": role, "content": content, "origin": "assistant", "segments": [], "context": ["memory"],
            "lineage": [f"memory:fact-{fact}"]}


# Eleven rows of conversation and one system row a peer's copy carried. The cut takes the first (an answer naming a
# memory no later row names); the first row kept is a turn with a pasted document part, which the label of the saved
# answer names; the system row stands among the last ten rows, so the turns are read before they are cut.
_ROWS = (
    _row("assistant", _ELEVENTH, 0),
    {"role": "user", "content": _PASTED, "origin": "typed", "context": None, "lineage": None,
     "segments": [[0, _TYPED_END, "typed"], [_TYPED_END, len(_PASTED), "document"]]},
    _row("assistant", "The wall bed suits them.", 2),
    _row("user", "And the onions?"),
    _row("assistant", "The east beds.", 4),
    _row("user", "And the garlic?"),
    _row("assistant", "By the wall, beside the shallots.", 6),
    _row("user", "When do we lift the garlic?"),
    _row("assistant", "In early summer.", 8),
    {"role": "system", "content": _SYSTEM_ROW, "origin": "legacy", "segments": [], "context": None, "lineage": None},
    _row("user", "Thank you."),
    _row("assistant", "Glad to help.", 10),
)

CASES = (
    # The tools pipeline, every turn of the default chat: the decisions, the tools, the final answer, the save.
    _case("tools/zero-call-reuse", [_turn("Good morning! When should onions be sown?")], script=[_say(_DIRECT)],
          checks=[_calls(["generate"]), _yields(0, [_DIRECT]), _offered("tools"), _absent("tool"),
                  _saved(*_SAVED_PAIR), _seams("stream_with_tools")]),
    _case("tools/one-call-then-final", [_turn(_QUESTION)], script=_ONE_CALL,
          checks=[_calls(["generate", "generate", "stream"]), _ran("read_file"), _absent("ask.approval"),
                  _absent("ask.demand"), _history(["read_file", True]), _saved(*_SAVED_PAIR)]),
    # Bulbe's per-call seam armed, its own callable: the gate asks it for each call, and never the turn's demand.
    _case("tools/approval-seam", [_turn(_QUESTION)], script=_ONE_CALL, bulbe=True,
          checks=[_calls(["generate", "generate", "stream"]), _ran("read_file"), _fired("ask.approval"),
                  _absent("ask.demand"), _history(["read_file", True]), _saved(*_SAVED_PAIR)]),
    _case("tools/native-two-calls", [_turn("Which files sit in the workspace, and what does notes.txt hold?")],
          prefs=_NATIVE_TRANSCRIPT,
          script=[_call(("list_files", {}), _READ), _say(_DONE), _say(_FINAL)],
          checks=[_calls(["generate", "generate", "stream"]), _ran("list_files", "read_file"),
                  _fact("second_decision_ends_on_a_tool_message", True),
                  _fact("final_ends_on_a_tool_message", True)]),
    _case("tools/retryable-failure-then-success", [_turn("Open notes.txt for me and tell me about the west bed.")],
          script=[_call(("read_file", {"file": "notes.txt"})), _call(_READ), _say(_DONE), _say(_FINAL)],
          checks=[_calls(["generate", "generate", "generate", "stream"]), _ran("read_file"),
                  _history(["read_file", False], ["read_file", True])]),
    _case("tools/no-progress-guard", [_turn("Which files sit in the workspace?")],
          script=[_call(_LIST), _call(_LIST), _say("The workspace holds notes.txt and sow.py.")],
          checks=[_calls(["generate", "generate", "stream"]), _ran("list_files"), _history(["list_files", True]),
                  _absent("ask.approval")]),
    _case("tools/verification-hint", [_turn("Divide one by zero in the sandbox and show me the output.")],
          results={"execute_code": [_TRACEBACK, "1.0"]},
          script=[_call(("execute_code", {"code": "print(1 / 0)"})), _say("It raised."),
                  _call(("execute_code", {"code": "print(1 / 1)"})), _say(_DONE),
                  _say("The division by zero raised; divided by one it printed 1.0.")],
          checks=[_calls(["generate"] * 4 + ["stream"]), _ran("execute_code", "execute_code"),
                  _result(0, "verification_hints", 1)]),
    # Today's defect, pinned as it stands: the salvage synthesizes write_file with a ``filename`` key, the built-in
    # write_file's parameter is ``path``, and no repair maps the one onto the other, so every salvage on the
    # built-in registry ends in a refused call and the final answers over the refusal. A fix of that key changes
    # this case: read it as the change it is, never as a regression.
    _case("tools/salvage-from-narration", [_turn("Write the sowing months to sow.py so it prints them.")],
          script=[_say(_NARRATION), _say("I could not save sow.py: the call lacked its path.")],
          checks=[_calls(["generate", "stream"]), _ran(), _history(["write_file", False]),
                  _fact("salvage_reasoning", ["intent-transpiled"])]),
    _case("tools/format-fallback", [_turn(_QUESTION)], model=PLAIN,
          script=[_decision(tool_name="read_file", arguments={"path": "notes.txt"}, reasoning="The notes hold it."),
                  _decision(tool_name="none", arguments={}, reasoning="The notes answered it."), _say(_FINAL)],
          checks=[_calls(["generate", "generate", "stream"]), _offered("schema"), _fired("engine.show"),
                  _ran("read_file"), _absent("ask.approval")]),
    _case("tools/format-zero-call", [_turn("Good morning! When should onions be sown?")], model=PLAIN,
          script=[_decision(tool_name="none", arguments={}, reasoning="No tool helps here."), _say(_DIRECT)],
          checks=[_calls(["generate", "generate"]), _offered("schema"), _yields(0, [_DIRECT]), _absent("tool"),
                  _saved(*_SAVED_PAIR)]),
    _case("tools/prior-calls-second-turn",
          [_turn("What does notes.txt say about the garlic?"), _turn("And which bed holds the onions, then?")],
          script=[*_ONE_CALL, _say(_SECOND_DIRECT)],
          checks=[_calls(["generate", "generate", "stream"], ["generate"]),
                  _fact("earlier_call_in_the_second_turn", True), _yields(1, [_SECOND_DIRECT]),
                  _saved(*_SAVED_PAIR, *_SAVED_PAIR)]),
    _case("tools/prior-calls-native",
          [_turn("What does notes.txt say about the garlic?"), _turn("And which bed holds the onions, then?")],
          prefs=_NATIVE_TRANSCRIPT, script=[*_ONE_CALL, _say(_SECOND_DIRECT)],
          checks=[_calls(["generate", "generate", "stream"], ["generate"]),
                  _fact("earlier_echo_in_the_second_turn", True), _yields(1, [_SECOND_DIRECT]),
                  _saved(*_SAVED_PAIR, *_SAVED_PAIR)]),
    _case("tools/fast-tool-model",
          [_turn("List files in the workspace and tell me which one holds the notes.", optimize=True)],
          fast_tool_model=FAST, script=[_call(("list_files", {})), _say(_DONE), _say("notes.txt holds the notes.")],
          checks=[_calls(["generate", "generate", "stream"]), _models(FAST, FAST, FAST),
                  _result(0, "pipeline", "tools"), _ran("list_files"), _saved(*_SAVED_PAIR)]),
    _case("tools/eleven-rows-last-ten", [_turn("Which bed did we settle on for the shallots?")],
          history=(("conv-1", _ROWS),), script=_ONE_CALL,
          checks=[_calls(["generate", "generate", "stream"]), _fact("first_decision_rows", 10),
                  _fact("eleventh_row_sent", False), _fact("system_row_sent", False),
                  _fact("label_names_the_pasted_document", True)]),
    # The other pipelines a chat turn can take with tools armed, each saving its own answer. Today's behavior,
    # pinned as it stands: the reasoning pass of think+tools and the draft of self-correct run with persist=False
    # and still reach cache.put, the cache store reading no persist. A fix shows here as the change it is.
    _case("think-tools/one-save", [_turn(_WEIGH, think=True)], script=_THINK_TOOLS,
          checks=[_calls(["stream", "generate", "generate", "generate"]), _think_flags(True, None, None, None),
                  _result(0, "pipeline", "think_tools"), _saved(*_SAVED_PAIR), _fact("joined_save", True),
                  _seams("executor.execute", "execute_with_tools")]),
    _case("think-tools/zero-call", [_turn(_WEIGH, think=True)],
          script=[_say(_PHASE_ONE, thinking=_THOUGHT), _say("No tool is needed for that.")],
          checks=[_calls(["stream", "generate"]), _absent("tool"), _saved(*_SAVED_PAIR),
                  _fact("phase_one_saved", True)]),
    _case("think-tools/governed", [_turn(_WEIGH, think=True)], governor={}, script=_THINK_TOOLS,
          checks=[_calls(["stream", "generate", "generate", "generate"]),
                  _callers(["chat", True], *[["direct", True]] * 3), _fired("ticket.set"), _fired("ticket.clear"),
                  _fact("joined_save", True)]),
    # The final of the tools phase refused: the tool executor's own fallback answers with the raw results.
    _case("think-tools/final-refused", [_turn(_WEIGH, think=True)], governor={"admit_first": 3},
          script=_THINK_TOOLS[:3],
          checks=[_calls(["stream", "generate", "generate", "generate"]),
                  _callers(["chat", True], ["direct", True], ["direct", True], ["direct", False]),
                  _fact("engine_requests", 3), _fact("tool_results_fallback_saved", True)]),
    _case("think-tools/fast-tool-model",
          [_turn("List files in the workspace and weigh which bed the shallots should take.", think=True,
                 optimize=True)], fast_tool_model=FAST,
          script=[_say(_PHASE_ONE, thinking=_THOUGHT), _call(("list_files", {})), _say(_DONE), _say(_TOOL_ANSWER)],
          checks=[_calls(["stream", "generate", "generate", "generate"]), _models(NATIVE, FAST, FAST, FAST),
                  _result(0, "pipeline", "think_tools"), _ran("list_files"), _saved(*_SAVED_PAIR)]),
    _case("reasoning/save", [_turn("Walk me step by step through sharing the beds between onions and garlic.")],
          checks=[_calls([]), _fired("reasoning"), _fired("hook.reasoning_step", 2), _fired("step", 4),
                  _result(0, "pipeline", "reasoning"), _saved(*_SAVED_PAIR)]),
    _case("consensus/save", [_turn("Which bed suits the shallots best?", consensus=True)],
          checks=[_calls([]), _fired("consensus"), _fired("hook.consensus_model", 2), _fired("step", 4),
                  _result(0, "pipeline", "consensus"), _saved(*_SAVED_PAIR)]),
    # Today's behavior, pinned as it stands: the draft runs with persist=False and still reaches cache.put.
    _case("self-correct/save", [_turn("Name the three beds of the garden.", self_correct=True)],
          script=[_say(_DRAFT)],
          checks=[_calls(["stream"]), _think_flags(False), _fired("self_correction"), _fired("status", 2),
                  _fired("step", 4), _result(0, "pipeline", "self_correct"), _saved(*_SAVED_PAIR),
                  _fact("corrected_save", True)]),
    # A turn the web search routes past the tools: the chat executor saves it, and offers it to the capture.
    _case("web/executor-save", [_turn("Search the web for when onions are sown.", web_search=True)], web=True,
          script=[_say(_WEB_ANSWER)],
          checks=[_calls(["stream"]), _fired("web.search"), _fired("capture"), _result(0, "pipeline", "web_search"),
                  _saved(("user", "typed"), ("assistant", "assistant+web")), _seams("executor.execute")]),
    _case("exec-pipeline/one-step", [_turn(_QUESTION, kind="pipeline")], governor={},
          script=_ONE_CALL,
          checks=[_calls(["generate", "generate", "stream"]),
                  _admitted(["pipeline", None, True], *[["direct", None, True]] * 3), _absent("ask.approval"),
                  _fired("status"), _fired("step", 3), _saved(*_SAVED_PAIR)]),
    # The governor the backend's gate asks on every call, and the stops a turn can meet.
    _case("governor/admits-every-call", [_turn(_QUESTION)], governor={}, script=_ONE_CALL,
          checks=[_calls(["generate", "generate", "stream"]), _admitted(*[["direct", None, True]] * 3),
                  _absent("ticket.set")]),
    _case("governor/refuses-mid-loop", [_turn(_QUESTION)], governor={"admit_first": 1},
          script=[_call(_READ)],
          checks=[_calls(["generate", "generate", "generate", "stream", "generate"]),
                  _admitted(["direct", None, True], *[["direct", None, False]] * 4),
                  _fact("error_text_yielded", True), _fact("engine_requests", 1), _saved(*_SAVED_PAIR)]),
    _case("no-backend", [_turn(_QUESTION)], backend=False,
          checks=[_calls([]), _yields(0, [_NO_BACKEND]), _absent("tool"), _saved(*_SAVED_PAIR)]),
    _case("stop/before-the-first-decision", [_turn(_QUESTION)], stop={"read": 0},
          checks=[_calls([]), _fired("stop"), _yields(0, []), _nothing_saved(), _result(0, "pipeline", "tools")]),
    _case("stop/mid-loop", [_turn(_QUESTION)], stop={"tool": 0}, script=[_call(_READ)],
          checks=[_calls(["generate"]), _fired("stop"), _ran("read_file"), _yields(0, []), _nothing_saved(),
                  _history(["read_file", True])]),
    _case("stop/final-closed-by-caller", [_turn(_QUESTION)], close_after=1, script=_ONE_CALL,
          checks=[_calls(["generate", "generate", "stream"]), _fired("close"), _nothing_saved(),
                  _fact("first_turn_yields", 1), _history(["read_file", True])]),
)


# ---------------------------------------------------------------------------
# A case, run
# ---------------------------------------------------------------------------
def _routing(model):
    return SimpleNamespace(model=model, task_type="general", temperature=0.2, prompt_variant="standard", timeout=30)


def _open(case, explain=False):
    """Seat a case's stand-ins and open its window; returns the world it runs in and how to close it."""
    world = World(case, explain)
    options = case["options"]
    model = options.get("model", NATIVE)
    world.client = ScriptedClient(world, options.get("script", ()))
    world.store = Store(world, options.get("history", ()))
    seeded = {
        "opti_oignon.config": _config(model, options.get("prefs", {})),
        "opti_oignon.router": _module("opti_oignon.router", RoutingResult=type("RoutingResult", (), {})),
        "opti_oignon.conversation": _module("opti_oignon.conversation", conversation_manager=world.store),
        "opti_oignon.security_mode": _module("opti_oignon.security_mode", is_bulbe=lambda: False,
                                             get_current_mode=lambda: "daily"),
        "opti_oignon.agent.allowlists": _module("opti_oignon.agent.allowlists", current_mode=lambda: "daily"),
        _LIBRARIAN: _librarian(world),
        "opti_oignon.memory.auto_capture": _module("opti_oignon.memory.auto_capture", maybe_capture=world.capture),
    }
    seeded.update(_memory(world.trace, placed=False))
    seeded.update(_caches(world.trace))
    seeded.update(_network(world.trace))
    seeded.update(_ledger(world))
    seeded.update(_engines(world))
    if options.get("web"):
        seeded.update(_web(world.trace, gated=False))
    targets = {name: source(*parts) for name, parts in _TARGETS}
    governor = None
    if "governor" in options:
        governor = Governor(world, **options["governor"])
        ordered = list(targets.items())
        at = [name for name, _path in ordered].index(_BACKEND) + 1
        targets = dict(ordered[:at] + [(_GOVERNOR, source("resource_governor.py"))] + ordered[at:])
    # The client library is the scripted one; the in-process engine is made unreachable, so its native library is
    # never loaded whatever the machine has installed.
    restore = _third_party({"ollama": world.client.module(), "llama_cpp": None})
    try:
        loaded, close_window = isolate(targets=targets, seeded=seeded,
                                       packages=("opti_oignon.agent", "opti_oignon.memory"))
    except BaseException:
        restore()
        raise

    def close():
        close_window()
        restore()

    try:
        stilled = [loaded[name] for name in _STILLED]
        if any(_reads_datetime(module) for module in stilled):
            raise RuntimeError("a module the still clock is set on reads datetime, which it does not still")
        clock = StillClock()
        for module in stilled:
            module.time = clock
        loaded["opti_oignon.executor"].uuid = StillUuid()
        loaded["opti_oignon.pipelines"].datetime = StillDate
        if governor is not None:
            governed = loaded[_GOVERNOR]
            governed.time, governed.uuid, governed.datetime = clock, StillUuid(), StillDate
            governor.bind(governed)
        backend_module = loaded[_BACKEND]
        registry = StubRegistry()
        if options.get("backend", True):
            backend = _recording_ollama(backend_module, world)
            registry.register(backend)
            registry.activate(backend.name)
        # The accessor every module asks reads this one: the registry the case built, never one probed for.
        backend_module._registry_instance = registry
        world.modules = loaded
        tools = loaded["opti_oignon.tool_executor"].ToolExecutor(
            registry=_scripted_registry(loaded["opti_oignon.tool_registry"], world))
        executor = loaded["opti_oignon.executor"].Executor()
        # The path's own seams, traced by the keywords each call hands them.
        tools.stream_with_tools = world.seam("stream_with_tools", tools.stream_with_tools)
        tools.execute_with_tools = world.seam("execute_with_tools", tools.execute_with_tools)
        executor.execute = world.seam("executor.execute", executor.execute)
        world.agent = loaded["opti_oignon.agentic_executor"].AgenticExecutor(
            executor=executor, tool_executor=tools, fast_tool_model=options.get("fast_tool_model"))
    except BaseException:
        close()
        raise
    return world, close


def _start(world, turn, run):
    """The turn's call, as the chat route makes it: the agentic executor, or the execution pipeline's runner, each
    handed the keywords the route hands it."""
    options = world.case["options"]
    model = options.get("model", NATIVE)
    approval = world.bulbe_approval if options.get("bulbe") else None
    kwargs = turn["kwargs"]
    if turn["kind"] == "pipeline":
        pipelines = world.modules["opti_oignon.pipelines"]
        step = pipelines.ExecutionStep(step_type="tools", label="Read the notes")
        pipeline = pipelines.ExecutionPipeline(id="garden-notes", name="Garden notes", steps=[step],
                                               created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00")
        return pipelines.PipelineRunner(agentic_executor=world.agent).execute(
            pipeline=pipeline, message=turn["question"], routing=_routing(model),
            conversation_id=turn["conversation"], on_status=world.on_status, on_tool_call=world.on_tool_call,
            on_reasoning_step=world.on_reasoning_step, on_consensus_model=world.on_consensus_model,
            approval_fn=approval, run=run)
    return world.agent.execute(
        message=turn["question"], routing=_routing(model), conversation_id=turn["conversation"],
        think=kwargs.get("think"), web_search=kwargs.get("web_search"), consensus=kwargs.get("consensus"),
        consensus_models=None, consensus_strategy=None, self_correct=kwargs.get("self_correct"),
        optimize=kwargs.get("optimize"), on_tool_call=world.on_tool_call, on_reasoning_step=world.on_reasoning_step,
        on_consensus_model=world.on_consensus_model, on_status=world.on_status, approval_fn=approval, run=run)


def _drive(world, generator, close_after=None):
    """Every non-empty yield, as it reaches the caller and into the trace; the caller closes after ``close_after``."""
    yields = []
    while True:
        try:
            item = next(generator)
        except StopIteration:
            return yields
        if item == "" or item is None:
            continue
        kept = list(item) if isinstance(item, tuple) else item
        yields.append(kept)
        # Traced once any held ticket is released, as the ledger is: the stream thread releases its ticket after
        # its last chunk is queued, and the caller may read the first before that, so waiting keeps the order of
        # the two the code's, never the scheduler's.
        world.trace.released.wait(_RELEASE_WAIT_S)
        world.trace.add("yield", chunk=kept)
        if close_after is not None and len(yields) >= close_after:
            world.trace.add("close")
            generator.close()
            return yields


def run_case(case, explain=False):
    """One case's record: its turns, each with what its seams were handed, in order; its facts."""
    world, close = _open(case, explain)
    turns = []
    try:
        executor_module = world.modules["opti_oignon.executor"]
        steps_module = world.modules[_STEPS]
        for index, turn in enumerate(case["turns"]):
            if index:
                world.next_turn()
            # The chat route's turn: its conversation, its stop, its results, its step recorder, the claim of what
            # the user typed, and its way to ask.
            run = SimpleNamespace(conversation_id=turn["conversation"], stop=threading.Event(), results={},
                                  steps=steps_module.StepRecorder(world.step), demand=world.demand,
                                  user_turn=executor_module.user_turn(turn["question"], turn["question"], ()))
            world.run, world.turn = run, index
            close_after = case["options"].get("close_after") if index == 0 else None
            yields = _drive(world, _start(world, turn, run), close_after)
            world.yields.append(yields)
            turns.append({
                "requests": world.requests[-1],
                "admissions": world.admissions[-1],
                "approvals": world.approvals[-1],
                "writes": world.writes[-1],
                "history": shape([vars(call) for call in world.agent.get_tool_history(turn["conversation"])]),
                "yields": shape(yields),
                "results": shape({key: run.results[key] for key in sorted(run.results)}),
                "trace": world.trace.turns[-1],
            })
        facts = _facts(world, turns)
    finally:
        close()
    return {"turns": turns, "facts": facts}


def _echo_then_result(messages):
    """Whether ``messages`` replay an earlier call natively: the assistant's echo of it, then its tool message."""
    for echo, result in zip(messages, messages[1:]):
        calls = echo.get("tool_calls") or []
        if (echo.get("role") == "assistant" and calls and calls[0].get("function", {}).get("name") == "read_file"
                and result.get("role") == "tool"
                and str(result.get("content", "")).startswith("[environment] earlier tool result")):
            return True
    return False


def _facts(world, turns):
    """What shows each case's trigger fired, read off what the engine received and what the store holds."""
    engine = world.engine
    first = engine[0][0]["messages"] if engine and engine[0] else []
    sent = canonical(_plain([request["messages"] for turn in engine for request in turn]))
    rows = world.store.rows.get("conv-1", [])
    saved = rows[-1] if rows and rows[-1]["role"] == "assistant" else None
    second = engine[1][0]["messages"] if len(engine) > 1 and engine[1] else []
    last = world.yields[0][-1] if world.yields and world.yields[0] else None
    return {
        "past_script": world.client.past,
        "script_left": len(world.client.script),
        "engine_requests": sum(len(turn) for turn in engine),
        "second_decision_ends_on_a_tool_message": bool(
            engine and len(engine[0]) > 1 and engine[0][1]["messages"][-1].get("role") == "tool"),
        "final_ends_on_a_tool_message": bool(
            engine and engine[0] and engine[0][-1]["messages"][-1].get("role") == "tool"),
        "earlier_call_in_the_second_turn": bool(
            second and "earlier tool call by assistant" in str(second[-1].get("content", ""))),
        "earlier_echo_in_the_second_turn": _echo_then_result(second),
        "first_decision_rows": len([message for message in first[:-1] if message.get("role") != "system"]),
        "eleventh_row_sent": _ELEVENTH in sent,
        "system_row_sent": _SYSTEM_ROW in sent,
        "label_names_the_pasted_document": bool(
            saved and "document" in (saved["context"] or [])
            and any(entry.startswith("document:") for entry in saved["lineage"] or [])),
        "joined_save": bool(saved and saved["content"] == _PHASE_ONE + "\n\n" + _TOOL_ANSWER),
        "phase_one_saved": bool(saved and saved["content"] == _PHASE_ONE),
        "tool_results_fallback_saved": bool(saved and saved["content"].startswith(_PHASE_ONE + "\n\n" + _FALLBACK)),
        "corrected_save": bool(saved and saved["content"] == _CORRECTED),
        "error_text_yielded": isinstance(last, str) and last.startswith("Error generating response: Not enough"),
        "salvage_reasoning": [call.reasoning for call in world.agent.get_tool_history("conv-1")],
        "first_turn_yields": len(turns[0]["yields"]) if turns else 0,
    }


def capture_all():
    """Every case's record, by name."""
    return {case["name"]: run_case(case) for case in CASES}


def golden_of(records):
    """What the golden file holds of the records: every case's turns. The facts are checked on each capture."""
    return {"format": FORMAT, "cases": {name: {"turns": record["turns"]} for name, record in records.items()}}


def capture_all_json():
    """The capture as a child prints it: the golden content, then every case's facts, as canonical JSON."""
    records = capture_all()
    return canonical({"golden": golden_of(records), "facts": {name: r["facts"] for name, r in records.items()}})


# ---------------------------------------------------------------------------
# Two hash seeds, in children the data firewall covers
# ---------------------------------------------------------------------------
_CHILD = (
    "import sys\n"
    "sys.dont_write_bytecode = False\n"
    "sys.pycache_prefix = sys.argv[2]\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "import _tool_path_off_golden as golden\n"
    "sys.stdout.write(golden.capture_all_json())\n"
)


def capture_in_children(seeds=SEEDS, timeout=180):
    """The capture as each seed's child prints it, by seed. The environment is extended, never replaced."""
    out = {}
    with tempfile.TemporaryDirectory(prefix="tool-path-off-golden-") as empty, \
            tempfile.TemporaryDirectory(prefix="tool-path-off-golden-bytecode-") as bytecode:
        for seed in seeds:
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
            done = subprocess.run([sys.executable, "-c", _CHILD, str(TESTS), bytecode], cwd=empty, env=env,
                                  capture_output=True, text=True, timeout=timeout)
            if done.returncode != 0:
                raise RuntimeError(f"the capture under seed {seed} failed:\n{done.stderr[-4000:]}")
            if os.listdir(empty):
                raise RuntimeError(f"the capture under seed {seed} left files in its working directory")
            out[seed] = done.stdout
    return out


def fixture_text(captured):
    """The golden file's text for a capture as a child printed it: its golden part, rendered, one trailing newline."""
    return render(json.loads(captured)["golden"]) + "\n"


def compare(text, fixture=None):
    """``same`` when the golden file holds ``text`` exactly, ``differs`` when it holds another, ``absent`` when there
    is none: a missing file is never read as agreeing."""
    path = Path(fixture) if fixture is not None else FIXTURE
    if not path.is_file():
        return "absent"
    return "same" if path.read_text(encoding="ascii") == text else "differs"


# ---------------------------------------------------------------------------
# The command line: --check and --write, never anything else
# ---------------------------------------------------------------------------
# What the capture runs through: the package, and the tests/ files it loads.
_SOURCES = ("opti_oignon", "tests/_isolation.py", "tests/_registry_bridge.py", "tests/_onion_off_golden.py",
            "tests/_tool_path_off_golden.py")


def _sources_differ():
    done = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", *_SOURCES],
                          capture_output=True)
    return done.returncode != 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="capture and compare with the golden file")
    action.add_argument("--write", action="store_true", help="capture and write the golden file")
    action.add_argument("--explain", metavar="NAME", help="print every field of the cases whose name holds NAME")
    parser.add_argument("--allow-dirty", action="store_true", help="write even when the sources differ from HEAD")
    args = parser.parse_args(argv)
    # Run in this process (--explain), the windows would otherwise leave bytecode beside the sources.
    sys.dont_write_bytecode = True

    from _data_firewall import REFUSED, DataFirewall

    firewall = DataFirewall(REPO)
    firewall.install()
    try:
        uncovered = firewall.cover_children()
        if uncovered:
            print(f"the children cannot be covered by the data firewall ({uncovered}); refusing", file=sys.stderr)
            return REFUSED
        if args.explain:
            for case in CASES:
                if args.explain in case["name"]:
                    print(render({case["name"]: run_case(case, explain=True)}, open_to=5))
            return 0
        if args.write and _sources_differ() and not args.allow_dirty:
            print(f"a source the capture runs through differs from HEAD ({', '.join(_SOURCES)}): the golden is "
                  "written from a clean tree, or with --allow-dirty", file=sys.stderr)
            return 1
        captured = capture_in_children()
        texts = {seed: fixture_text(text) for seed, text in captured.items()}
        if len(set(texts.values())) != 1:
            print("the hash seeds disagree: nothing written", file=sys.stderr)
            return 1
        text = next(iter(texts.values()))
        if args.check:
            verdict = compare(text)
            print({"same": "golden: same", "differs": "golden: DIFFERS", "absent": "golden: ABSENT"}[verdict])
            return 0 if verdict == "same" else 1
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(text, encoding="ascii")
        print(f"golden written: {FIXTURE.relative_to(REPO)} sha256 {sha(text)}")
        return 0
    finally:
        firewall.uncover_children()
        firewall.uninstall()


if __name__ == "__main__":
    raise SystemExit(main())
