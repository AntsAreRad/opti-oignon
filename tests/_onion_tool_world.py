#!/usr/bin/env python3
"""The chat's tools pipeline with the onion's journal laying its window, in a window of its own.

Built on the onion window's world (``tests/_onion_window_world.py``) and on the
tool path's off golden (``tests/_tool_path_off_golden.py``), both reused and
never edited. The window holds, as they are, the agentic executor, the tool
executor and the chat executor, with the structured-output engine, the
tool-calling primitives, the response hygiene, the capability manifest, the
provenance gate, the untrusted-data envelope, the tool registry's own class
over the built-in definitions (each handler scripted) and the Ollama heads
over a scripted client behind a scripted registry. Added here:

* ``ToolJournal`` -- the window world's journal librarian, its window
  carrying what a tool turn's rounds are held to (the schemas' charge, the
  transcript's room, the most one result may take, the limit its rounds may
  count to, the overhead and the margin) and the names a tool turn's
  transcript is laid with, the composer's own, which the window loads.
* ``Engine`` -- the scripted client: each request answered by the next reply
  of the script, a reply's ``extra`` reported on its response or its last
  chunk, its ``calls`` made, its stream broken before or after its first
  words when told.
* ``Store`` -- the conversation store as the executor and the agentic path
  read it: the last rows' read a tripwire of its own (``read.rows``), the
  mirror handing each turn with its origin, segments, context and lineage.
* ``open_tools`` -- the window: the journal librarian, or the onion's own
  modules with its switch read on and nothing written (``real``); the window
  world's traced governor (``gov``), or the resource governor's own module
  serving the off golden's recording governor (``real_gov``), its gate and
  ticket run as they are.

``run_turns`` drives the agentic executor turn by turn, as the chat route
hands it each turn, and keeps every request whole, every admission, every
write, every non-empty yield, the status lines, the run's results and the
trace.

Local-only (the public distribution ships no tests). Stdlib-only apart from
the windows it loads.
"""

import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _onion_off_golden as golden  # noqa: E402
import _onion_window_world as w  # noqa: E402
import _tool_path_off_golden as tp  # noqa: E402
from _isolation import isolate, source  # noqa: E402

NATIVE, PLAIN, FAST = tp.NATIVE, tp.PLAIN, tp.FAST
QUESTION = tp._QUESTION
READ = tp._READ
FINAL = tp._FINAL
DONE = tp._DONE
TOOLS = tp._TOOLS
_COMPOSER = "opti_oignon.memory.composer"
_LIBRARIAN = "opti_oignon.memory.librarian"
_GOVERNOR = "opti_oignon.resource_governor"
_ONION = ("probes", "core_store", "receipts", "composer", "peels", "librarian")
# What the journal librarian's window holds a tool turn's rounds to, as the shipped journal section reads.
FLOOR, MARGIN, OVERHEAD, ROOM, CAP = 1024, 0.15, 4, 3072, 1536


def count(text):
    """The turn's counter, as the executor counts with no context manager seated: four characters a token."""
    return len(str(text)) // 4


def limit(num_ctx, margin=MARGIN, floor=FLOOR):
    """The most a window and its transcript may count sent with ``num_ctx``, the answer keeping its floor."""
    room = num_ctx - floor
    cap = math.floor(room / (1 + margin))
    while cap > 0 and math.ceil(cap * (1 + margin)) > room:
        cap -= 1
    return cap


# ---------------------------------------------------------------------------
# The journal librarian, for a tool turn
# ---------------------------------------------------------------------------
class ToolJournal(w.JournalLibrarian):
    """The window world's journal librarian, its window holding a tool turn's rounds.

    A plan handed ``tools`` makes a tool window: the heaviest schema's count
    charged on the whole, the window's answer room the context less what it
    is accounted, and the figures its rounds are held to: ``tool_room``,
    ``result_cap``, the limit -- the context's own (``limit``), or the
    window's count and ``spare`` more when that is lower -- the overhead and
    the margin. The names a tool turn's transcript is laid with are the
    composer's, loaded in the window.
    """

    def __init__(self, trace, *, tool_room=ROOM, result_cap=CAP, spare=None, **kwargs):
        super().__init__(trace, **kwargs)
        self.tool_room, self.result_cap, self.spare = tool_room, result_cap, spare

    def module(self):
        module = super().module()
        module.lay_tool_result = self.lay_tool_result
        module.tool_call_line = self.tool_call_line
        module.bare_text = self.bare_text
        return module

    def lay_tool_result(self, name, place, text, **kwargs):
        return sys.modules[_COMPOSER].tool_result(name, place, text, **kwargs)

    def tool_call_line(self, name, arguments):
        return sys.modules[_COMPOSER].tool_call_line(name, arguments)

    def bare_text(self, text):
        return sys.modules[_COMPOSER].model_text(text)

    def compose_window(self, plan, *, num_ctx=None):
        window = super().compose_window(plan, num_ctx=num_ctx)
        tools = getattr(plan, "inputs", {}).get("kwargs", {}).get("tools")
        if getattr(window, "refusal", None) is None and tools is not None:
            charge = max(count(text) for text in tools)
            window.tokens += charge
            window.tools = charge
            window.accounted = self.accounted if self.accounted is not None else math.ceil(window.tokens * 1.15)
            window.num_predict = (self.num_predict if self.num_predict is not None
                                  else window.num_ctx - window.accounted)
            window.tool_room, window.result_cap, window.overhead, window.margin = (
                self.tool_room, self.result_cap, OVERHEAD, MARGIN)
            window.limit = limit(window.num_ctx)
            if self.spare is not None:
                window.limit = min(window.limit, window.tokens + self.spare)
        return window


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class Engine:
    """The client library the Ollama heads call, each request answered by the next reply of the script.

    A reply is a mapping: ``content``, ``calls`` (name and arguments, in
    order), ``extra`` (what the engine reports beside it: on a response, or
    on a stream's last chunk), and ``fail`` (a stream's ``before`` or
    ``after`` its first words; any value fails a whole answer) with the words
    of its error (``error``). A
    stream answers its content cut in two at a space past the middle, its
    calls on its last chunk. A request past the script is answered with fixed
    words and counted.
    """

    def __init__(self, world, script):
        self._world, self.script, self.past = world, list(script), 0

    def module(self):
        return golden._module("ollama", chat=self.chat, show=self.show)

    def show(self, model):
        return {"capabilities": list(tp._CAPABILITIES.get(model, ("completion",)))}

    def chat(self, **kwargs):
        self._world.engine.append(kwargs)
        if self.script:
            reply = self.script.pop(0)
        else:
            self.past += 1
            reply = {"content": tp._PAST_SCRIPT}
        if kwargs.get("stream"):
            self._world.stop_at("final")
            return self._stream(reply)
        self._world.stop_at("decision")
        if reply.get("fail"):
            raise RuntimeError(reply.get("error", "the engine broke"))
        message = {"role": "assistant", "content": reply.get("content", "")}
        calls = _client_calls(reply)
        if calls:
            message["tool_calls"] = calls
        return {"message": message, "done": True, **dict(reply.get("extra") or {})}

    @staticmethod
    def _stream(reply):
        if reply.get("fail") == "before":
            raise RuntimeError(reply.get("error", "the engine broke"))
        text = reply.get("content", "")
        cut = text.find(" ", len(text) // 2)
        pieces = [text] if cut <= 0 else [text[:cut], text[cut:]]
        for place, piece in enumerate(pieces):
            if reply.get("fail") == "after" and place == 1:
                raise RuntimeError(reply.get("error", "the engine broke"))
            yield {"message": {"content": piece}}
        last = {"message": {"content": ""}, "done": True, **dict(reply.get("extra") or {})}
        calls = _client_calls(reply)
        if calls:
            last["message"]["tool_calls"] = calls
        yield last


def _client_calls(reply):
    return [{"function": {"name": name, "arguments": dict(arguments)}} for name, arguments in reply.get("calls", ())]


def say(text, **extra):
    """A reply of words, with what the engine reports beside it."""
    return {"content": text, **({"extra": extra} if extra else {})}


def call(*calls, content="", **extra):
    """A reply calling ``calls``, each ``(name, arguments)``."""
    return {"content": content, "calls": [(name, dict(arguments)) for name, arguments in calls],
            **({"extra": extra} if extra else {})}


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
class Store:
    """The conversation store as the executor and the agentic path read it, and what it was asked to keep.

    A row carries its role, content, origin and segments and the context and
    lineage it was written with. The mirror hands each turn with its origin,
    segments, context and lineage, as the application's store does: a user
    turn's label read from its own parts, an answer's the one it was written
    with, legacy when it has none. ``get_messages`` -- the read the tools
    pipeline makes of the last rows off the journal -- is traced
    (``read.rows``), and is where a stop can be raised.
    """

    def __init__(self, world, history=()):
        self._world = world
        self.rows = {conversation_id: [dict(row) for row in rows] for conversation_id, rows in history}
        self.withdrawn = []

    def add_message(self, conv_id, role, content, model=None, metadata=None, *, origin="legacy", segments=(),
                    context=None, lineage=None):
        parts = [list(segment) for segment in segments or ()]
        row = {"role": role, "content": content, "origin": origin, "segments": parts,
               "context": None if context is None else list(context),
               "lineage": None if lineage is None else list(lineage)}
        self._world.writes[-1].append(dict(row, model=model))
        self._world.trace.add("store.add", role=role, origin=origin)
        self.rows.setdefault(conv_id, []).append(row)
        return SimpleNamespace(role=role, content=content)

    def get_messages(self, conv_id):
        self._world.trace.add("read.rows", conversation=conv_id)
        self._world.stop_at("read")
        return [SimpleNamespace(id=index + 1, role=row["role"], content=row["content"])
                for index, row in enumerate(self.rows.get(conv_id, []))]

    def _turns(self, conv_id):
        return [row for row in self.rows.get(conv_id, []) if row["role"] in ("user", "assistant")]

    def _label(self, row):
        wrapper = sys.modules[w._WRAPPER]
        if row["role"] == "user":
            return wrapper.user_turn_label(row["content"], row["origin"], row["segments"])
        if row["context"] is not None:
            return list(row["context"]), list(row["lineage"] or [])
        return ["legacy"], []

    def get_context_messages(self, conv_id, **kwargs):
        return [{"role": row["role"], "content": row["content"]} for row in self._turns(conv_id)]

    def get_labelled_context_messages(self, conv_id):
        wrapper = sys.modules[w._WRAPPER]
        return [wrapper.labelled({"role": row["role"], "content": row["content"]}, *self._label(row))
                for row in self._turns(conv_id)]

    def get_mirror_messages(self, conv_id):
        out = []
        for row in self._turns(conv_id):
            context, lineage = self._label(row)
            out.append({"role": row["role"], "content": row["content"], "origin": row["origin"],
                        "segments": [list(segment) for segment in row["segments"]], "context": list(context),
                        "lineage": list(lineage)})
        return out

    def get_conversation(self, conv_id):
        return SimpleNamespace(id=conv_id, messages=self.get_context_messages(conv_id), metadata={})

    def update_conversation_metadata(self, conv_id, *args, **kwargs):
        self._world.trace.add("store.metadata", conversation=conv_id)

    def withdrawn_sources(self):
        return list(self.withdrawn)


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------
class World:
    """One window's seams and what they saw, a list per turn for each, the current turn's last."""

    def __init__(self, *, script, results=None, stop=None):
        self.trace = golden.Trace(explain=True)
        self.requests, self.admissions, self.approvals, self.writes, self.engine = [[]], [[]], [[]], [[]], []
        self.statuses, self.calls = [[]], [[]]
        self.run, self.turn = None, 0
        self.stops = {where: set(turns) for where, turns in (stop or {}).items()}
        self._queued = {name: list(queued) for name, queued in (results or {}).items()}
        self.client = Engine(self, script)

    def next_turn(self):
        for kept in (self.requests, self.admissions, self.approvals, self.writes, self.statuses, self.calls):
            kept.append([])
        self.trace.next_turn()

    def stop_at(self, where):
        """Raise the turn's stop where the stand-in calling this stands, on the turns its world names."""
        if self.turn in self.stops.get(where, ()):
            self.trace.add("stop", where=where)
            self.run.stop.set()

    def hook(self, where):
        return lambda: self.stop_at(where)

    def record(self, call, model, messages, options, given):
        """One request as the path hands it to a head, whole."""
        self.requests[-1].append({"call": call, "model": model, "messages": [dict(m) for m in messages or []],
                                  "options": dict(options or {}), "given": dict(given)})
        self.trace.add(call, model=model)

    def seam(self, name, method):
        def traced(*args, **kwargs):
            self.trace.add("call." + name, keys=sorted(kwargs), positional=len(args))
            return method(*args, **kwargs)

        return traced

    def judged(self, method):
        """The tool executor's one way to run a call, traced with what the gate judges it by: the typed parts of
        the provenance it is handed (None for none) and the names offered."""
        def traced(tool_name, arguments, *args, **kwargs):
            provenance = kwargs.get("provenance")
            self.trace.add("judged", tool=tool_name, units=None if provenance is None else tuple(provenance.units),
                           offered=None if kwargs.get("offered") is None else sorted(kwargs["offered"]))
            return method(tool_name, arguments, *args, **kwargs)

        return traced

    def handler(self, name):
        def run(**arguments):
            self.trace.add("tool", tool=name, arguments=arguments)
            queued = self._queued.get(name)
            result = queued.pop(0) if queued else tp._RESULTS[name]
            self.stop_at("tool")
            if isinstance(result, BaseException):
                raise result
            return result

        run.__name__ = f"scripted_{name}"
        return run

    def demand(self, tool_name, arguments, labels=None, effect=None):
        self.approvals[-1].append({"tool": tool_name, "arguments": dict(arguments)})
        self.trace.add("ask.demand", tool=tool_name)
        return True

    def capture(self, conversation_id, messages):
        self.trace.add("capture", conversation=conversation_id, turns=len(messages or []))

    def step(self, frame):
        self.trace.add("step", state=frame.get("state"))

    def on_status(self, message):
        self.statuses[-1].append(message)
        self.trace.add("status", message=message)

    def on_tool_call(self, call):
        self.calls[-1].append(call)
        self.trace.add("hook.tool_call", tool=call.tool_name, success=call.success)


def open_tools(*, script=(), journal=None, enabled=True, real=False, gov=None, real_gov=None, model=NATIVE,
               prefs=None, history=(), offline=False, hit=False, backend=True, results=None, stop=None,
               fast_tool_model=None, llama=False):
    """Seat the agentic path with the stand-ins asked and open its window; ``(world, close)``.

    ``journal`` is the ``ToolJournal``'s keywords, its switch ``enabled``;
    ``real`` seats the onion's own modules instead, its switch read on (or
    ``enabled``), no persistence, no counts file, no burst thread. ``gov``
    seats the window world's traced governor with its keywords (its state in
    ``world.gov_state``); ``real_gov`` the resource governor's own module
    serving the off golden's recording governor (``admit_first``). ``stop``
    maps a place -- ``plan``, ``admission``, ``thread``, ``read``,
    ``decision``, ``final``, ``tool`` -- to the turns whose stop is raised
    there. ``results`` queues each tool's results; ``backend`` False seats
    no backend; ``llama`` names the Ollama head ``llama_server``, its one
    slot holding 16384 tokens, with slot affinity switched on.
    """
    world = World(script=script, results=results, stop=stop)
    world.model = model
    world.store = Store(world, history)
    trace = world.trace
    seeded = {
        "opti_oignon.config": tp._config(model, prefs or {}),
        "opti_oignon.router": golden._module("opti_oignon.router", RoutingResult=type("RoutingResult", (), {})),
        "opti_oignon.conversation": golden._module("opti_oignon.conversation", conversation_manager=world.store),
        "opti_oignon.security_mode": golden._module("opti_oignon.security_mode", is_bulbe=lambda: False,
                                                    get_current_mode=lambda: "daily"),
        "opti_oignon.agent.allowlists": golden._module("opti_oignon.agent.allowlists", current_mode=lambda: "daily"),
        "opti_oignon.memory.auto_capture": golden._module("opti_oignon.memory.auto_capture",
                                                          maybe_capture=world.capture),
    }
    seeded.update(golden._memory(trace, placed=False))
    seeded.update(golden._caches(trace, hit_keys=("*",) if hit else ()))
    seeded.update(golden._network(trace, online=not offline))
    ledger_seeded, world.rows = w.ledger(trace)
    seeded.update(ledger_seeded)
    world.journal, world.gov_state, world.governor = None, None, None
    if not real:
        world.journal = ToolJournal(trace, **dict(journal or {}, enabled=enabled), on_plan=world.hook("plan"))
        seeded[_LIBRARIAN] = world.journal.module()
    if gov is not None:
        gov_seeded, world.gov_state = w.governor(trace, **gov, on_admit=world.hook("admission"),
                                                 on_ticket=world.hook("thread"))
        seeded.update(gov_seeded)
    if llama:
        seeded.update(golden._slots(trace))
        affinity = SimpleNamespace(enabled=False, config={"slot_affinity": {"enabled": True},
                                                          "stable_prefix": {"enabled": False}})
        seeded["opti_oignon.context_optimizer"] = golden._module("opti_oignon.context_optimizer",
                                                                 get_optimizer=lambda: affinity)
    ordered = [(name, source(*parts)) for name, parts in tp._TARGETS]
    at = [name for name, _path in ordered].index("opti_oignon.memory.probes") + 1
    onion = [(f"opti_oignon.memory.{name}", source("memory", f"{name}.py")) for name in _ONION if name != "probes"]
    ordered[at:at] = onion if real else [(_COMPOSER, source("memory", "composer.py"))]
    if real_gov is not None:
        at = [name for name, _path in ordered].index(tp._BACKEND) + 1
        ordered[at:at] = [(_GOVERNOR, source("resource_governor.py"))]
    restore = tp._third_party({"ollama": world.client.module(), "llama_cpp": None})
    try:
        loaded, close_window = isolate(targets=dict(ordered), seeded=seeded,
                                       packages=("opti_oignon.agent", "opti_oignon.memory"))
    except BaseException:
        restore()
        raise

    def close():
        close_window()
        restore()

    try:
        clock = golden.StillClock()
        for name in tp._STILLED:
            loaded[name].time = clock
        loaded["opti_oignon.executor"].uuid = tp.StillUuid()
        loaded["opti_oignon.pipelines"].datetime = tp.StillDate
        if real_gov is not None:
            governed = loaded[_GOVERNOR]
            governed.time, governed.uuid, governed.datetime = clock, tp.StillUuid(), tp.StillDate
            world.governor = tp.Governor(world, **real_gov)
            world.governor.bind(governed)
        if real:
            lib = loaded[_LIBRARIAN]
            lib.reset_librarian()
            loading = lib.load_config
            lib.load_config = lambda path=None: _replace(loading(path), enabled=enabled, persist_path="",
                                                         counters_path="")
            lib._default_runner = lambda conversation_id: trace.add("burst", conversation=conversation_id)
            world.librarian = lib
        backend_module = loaded[tp._BACKEND]
        registry = tp.StubRegistry()
        if backend:
            head = tp._recording_ollama(backend_module, world)
            if llama:
                head = _named_llama(type(head))()
            registry.register(head)
            registry.activate(head.name)
        backend_module._registry_instance = registry
        world.loaded = loaded
        world.executor_module = loaded["opti_oignon.executor"]
        world.tool_module = loaded["opti_oignon.tool_executor"]
        world.composer = loaded.get(_COMPOSER)
        world.wrapper = loaded[w._WRAPPER]
        tools = world.tool_module.ToolExecutor(registry=tp._scripted_registry(loaded["opti_oignon.tool_registry"],
                                                                              world))
        executor = world.executor_module.Executor()
        tools.stream_with_tools = world.seam("stream_with_tools", tools.stream_with_tools)
        tools.execute_with_tools = world.seam("execute_with_tools", tools.execute_with_tools)
        tools._execute_tool = world.judged(tools._execute_tool)
        executor.execute = world.seam("executor.execute", executor.execute)
        world.tools, world.executor = tools, executor
        world.agent = loaded["opti_oignon.agentic_executor"].AgenticExecutor(
            executor=executor, tool_executor=tools, fast_tool_model=fast_tool_model)
    except BaseException:
        close()
        raise
    return world, close


def _named_llama(recording):
    """The recording Ollama head under the llama-server engine's name, its one slot holding 16384 tokens."""

    class Named(recording):
        @property
        def name(self):
            return "llama_server"

        def slots(self):
            return [{"id": 1, "n_ctx": 16384}]

    return Named


def _replace(config, **fields):
    from dataclasses import replace

    return replace(config, **fields)


def routing(model=NATIVE):
    return SimpleNamespace(model=model, task_type="general", temperature=0.2, prompt_variant="standard", timeout=30)


def run_turns(world, turns, *, close_after=None):
    """Drive each turn of ``turns`` -- ``(conversation_id, question, keywords)`` -- as the chat route hands it the
    agentic executor; the caller closes the first turn after ``close_after`` yields when given. Each turn's record:
    its requests, admissions, writes, yields, status lines, the run's results, its ledger rows and its trace."""
    executor_module = world.executor_module
    steps = world.loaded[tp._STEPS]
    records = []
    for index, (conversation_id, question, kwargs) in enumerate(turns):
        if index:
            world.next_turn()
        rows_before = len(world.rows)
        run = SimpleNamespace(conversation_id=conversation_id, stop=threading.Event(), results={},
                              steps=steps.StepRecorder(world.step), demand=world.demand,
                              user_turn=executor_module.user_turn(question, question, ()))
        world.run, world.turn = run, index
        kwargs = dict(kwargs)
        model = kwargs.pop("model", world.model)
        generator = world.agent.execute(
            message=question, routing=routing(model), conversation_id=conversation_id,
            think=kwargs.get("think"), web_search=kwargs.get("web_search"), consensus=kwargs.get("consensus"),
            self_correct=kwargs.get("self_correct"), optimize=kwargs.get("optimize"),
            on_tool_call=world.on_tool_call, on_status=world.on_status, approval_fn=kwargs.get("approval_fn"),
            run=run)
        yields = tp._drive(world, generator, close_after if index == 0 else None)
        records.append({"requests": world.requests[-1], "admissions": world.admissions[-1],
                        "writes": world.writes[-1], "yields": yields, "statuses": world.statuses[-1],
                        "results": run.results, "rows": world.rows[rows_before:], "trace": world.trace.turns[-1],
                        "calls": world.calls[-1]})
    return records


def kinds(record):
    """The kinds of a turn's trace, in order."""
    return [event[0] for event in record["trace"]]


def events(record, kind):
    """The fields of each event of ``kind`` in a turn's trace."""
    return [event[1] if len(event) > 1 else {} for event in record["trace"] if event[0] == kind]


def transcript(request, window):
    """What a request sent past the window's messages."""
    return request["messages"][len(window.messages):]


def as_sent(window):
    """A composed window's messages as a request carries them."""
    return [{"role": role, "content": content} for role, content, _label in window.messages]


def cost(message):
    """A message's count as the journal counts one: its content, its calls when it carries any, and its overhead."""
    import json

    total = count(message.get("content") or "") + OVERHEAD
    if message.get("tool_calls"):
        total += count(json.dumps(message["tool_calls"], sort_keys=True, ensure_ascii=False, default=str))
    return total
