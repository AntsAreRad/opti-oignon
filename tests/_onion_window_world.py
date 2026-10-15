#!/usr/bin/env python3
"""The executor with the onion's journal laying a turn's window, in a window of its own.

The stand-ins are the onion-off golden's (``tests/_onion_off_golden.py``),
reused and never edited. Five are added:

* ``JournalLibrarian`` -- the librarian with the journal: its plan and its
  composition scripted, each call traced and kept; the read side the
  executor used to ask (``memory_block``, ``use_counter``) a tripwire that
  says it was reached.
* ``ThreeNames`` -- a librarian of three names, as the onion stood before
  its journal: its block traced, so a call to it is seen.
* ``governor`` -- the governor as the journal reaches it: each admission
  traced and held, as the real one holds an interactive admission and the
  load it claims, until a ticket is taken for it or it is given back
  through a real ``ticket_scope`` and ``end_pending_load``.
* ``real_governor`` -- the governor itself, its config set by hand, the
  model it is asked for loaded nowhere.
* ``open_real`` -- the onion's own modules beside the hub, its switch read
  on, nothing written: no persistence, no counts file, no burst thread.

The store is the golden's, read as the application's store hands the onion
its turns (``Store``): each with its origin, its segments, its context and
its lineage.

``run_turns`` drives the hub turn by turn, writes each non-empty yield into
the trace as it reaches the caller, and keeps the status lines each turn
said. Every ledger row is kept whole.

Local-only (the public distribution ships no tests). Stdlib-only apart from
the windows it loads.
"""

import math
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _onion_off_golden as golden  # noqa: E402
from _isolation import isolate, source  # noqa: E402

MODEL = golden.MODEL
FIRST, SECOND = golden.FIRST, golden.SECOND
CORE_HEAD = "Core 0123456789abcdef, pinned by the user:"
CORE_TEXT = "The user is called Alice."
_EXECUTOR, _WRAPPER, _DEDUP, _LIBRARIAN = golden._EXECUTOR, golden._WRAPPER, golden._DEDUP, golden._LIBRARIAN
_OPTIMIZER, _TIERS, _BACKEND = golden._OPTIMIZER, golden._TIERS, golden._BACKEND_MODULE
_GOVERNOR = "opti_oignon.resource_governor"
_ONION = ("probes", "core_store", "receipts", "composer", "peels", "librarian")
# What the journal librarian says each kind of source may take.
ROOM = {"memory": 400, "rag": 600, "web": 500, "document": 500}


# ---------------------------------------------------------------------------
# The librarians
# ---------------------------------------------------------------------------
class JournalLibrarian:
    """A librarian that plans and composes the journal as told, each call traced and kept.

    ``plan_refusal`` and ``compose_refusal`` are ``(motive, figures)`` the
    plan or the composition answers with; ``raises`` names the seams that
    raise (``plan``, ``compose``, ``curate``); ``accounted``, ``num_predict``
    and ``sent`` (the context the window says it was composed for) force the
    window's figures, to reach the hub's guard; ``cut`` is what the window
    says it let go, by kind; ``labels`` replaces each message's label, in
    order. The window it lays is short: the system
    messages, the Core's message (the conversation's first user turn joined
    to it), the conversation in its roles, and one last user message of the
    sources the turn brings, each wrapped, and the turn.
    """

    def __init__(self, trace, *, enabled=True, rung=16384, plan_refusal=None, compose_refusal=None,
                 accounted=None, num_predict=None, sent=None, cut=(), raises=(), labels=None, on_plan=None):
        self._trace = trace
        self.cut = tuple(cut)
        self.enabled, self.rung = enabled, rung
        self.plan_refusal, self.compose_refusal = plan_refusal, compose_refusal
        self.accounted, self.num_predict, self.sent = accounted, num_predict, sent
        self.raises, self.labels, self.on_plan = set(raises), labels, on_plan
        self.plans, self.composed = [], []

    def module(self):
        return golden._module(_LIBRARIAN, onion_enabled=self.onion_enabled, memory_block=self.memory_block,
                              maybe_curate=self.maybe_curate, use_counter=self.use_counter,
                              plan_window=self.plan_window, compose_window=self.compose_window,
                              tail_room=self.tail_room)

    def onion_enabled(self, path=None):
        return self.enabled

    def memory_block(self, conversation_id, question=None, **kwargs):
        self._trace.add("trip.memory_block", conversation=conversation_id)
        return "Core: the block of before."

    def use_counter(self, conversation_id, count, model):
        self._trace.add("trip.use_counter", conversation=conversation_id, model=model)

    def maybe_curate(self, conversation_id, messages, **kwargs):
        self._trace.add("curate", conversation=conversation_id, messages=messages, **kwargs)
        if "curate" in self.raises:
            raise RuntimeError("the curation's own words, never to be shown")
        return True

    def tail_room(self, model_window=None, **kwargs):
        self._trace.add("tail_room", model_window=model_window)
        return dict(ROOM)

    def plan_window(self, conversation_id, *, model, count, system, turn, wrap, tail=(), model_window=None,
                    messages=None, withdrawn=(), **kwargs):
        if "plan" in self.raises:
            raise RuntimeError("the plan's own words, never to be shown")
        inputs = {"conversation": conversation_id, "model": model, "count": count, "system": tuple(system),
                  "turn": dict(turn), "wrap": wrap, "tail": [dict(part) for part in tail],
                  "model_window": model_window, "messages": [dict(m) for m in messages or []],
                  "withdrawn": tuple(withdrawn), "kwargs": dict(kwargs)}
        self.plans.append(inputs)
        self._trace.add("plan", model=model, model_window=model_window, withdrawn=list(withdrawn),
                        tail=[part.get("kind") for part in tail], turns=len(inputs["messages"]),
                        images=turn.get("images"))
        if self.on_plan is not None:
            self.on_plan()
        if self.plan_refusal is not None:
            motive, figures = self.plan_refusal
            return SimpleNamespace(rung=0, refusal=SimpleNamespace(motive=motive, figures=tuple(figures), remedy=""))
        return SimpleNamespace(rung=self.rung, refusal=None, inputs=inputs)

    def compose_window(self, plan, *, num_ctx=None):
        self._trace.add("compose", num_ctx=num_ctx)
        if "compose" in self.raises:
            raise RuntimeError("the composition's own words, never to be shown")
        if self.compose_refusal is not None:
            motive, figures = self.compose_refusal
            return SimpleNamespace(rung=plan.rung, refusal=SimpleNamespace(
                motive=motive, figures=tuple(figures), remedy=""))
        messages = self.lay(plan.inputs)
        tokens = sum(len(content) // 4 + 4 for _role, content, _label in messages)
        sent = self.sent if self.sent is not None else (num_ctx if num_ctx is not None else plan.rung)
        accounted = self.accounted if self.accounted is not None else math.ceil(tokens * 1.15)
        room = self.num_predict if self.num_predict is not None else sent - accounted
        system_tokens = sum(len(content) // 4 + 4 for role, content, _label in messages if role == "system")
        first = not self.composed
        window = SimpleNamespace(
            messages=tuple(messages), rung=sent, num_ctx=sent, num_predict=room, accounted=accounted,
            tokens=tokens, layers=(("system", system_tokens), ("core", 12), ("peels", 0), ("receipts", 0),
                                   ("flesh", tokens - system_tokens - 12), ("tail", 0), ("turn", 0), ("images", 0)),
            cut=self.cut, epoch=len(self.composed) + 1, motive="first" if first else None,
            first_diff=None if first else 1, chain="0" * 64, cold=first, refusal=None)
        self.composed.append(window)
        return window

    def lay(self, inputs):
        wrap = inputs["wrap"]
        messages = [("system", text, ((), ())) for text in inputs["system"]]
        core = f"{CORE_HEAD}\n{CORE_TEXT}"
        flesh = [(m["role"], m["content"], ((), ())) for m in inputs["messages"] if m.get("role") in ("user", "assistant")]
        if flesh and flesh[0][0] == "user":
            core = f"{core}\n\n{flesh.pop(0)[1]}"
        messages.append(("user", core, ((), ())))
        messages.extend(flesh)
        last, context, lineage = [], set(), set()
        for part in inputs["tail"]:
            text = "\n".join(item[0] for item in part["items"])
            last.append(wrap(text, source="file" if part["kind"] == "rag" else "web"))
            for _text, label in part["items"]:
                context.update(label[0])
                lineage.update(label[1])
            if part.get("suffix"):
                last.append(part["suffix"])
        last.append(inputs["turn"]["content"])
        if messages[-1][0] == "user":
            _role, content, label = messages.pop()
            last.insert(0, content)
            context.update(label[0])
            lineage.update(label[1])
        messages.append(("user", "\n\n".join(last), (tuple(sorted(context)), tuple(sorted(lineage)))))
        if self.labels is not None:
            messages = [(role, content, self.labels[index] if index < len(self.labels) else label)
                        for index, (role, content, label) in enumerate(messages)]
        return messages


class ThreeNames:
    """A librarian of three names, as the onion stood before its journal: on, with a block it would hand over."""

    def __init__(self, trace, *, enabled=True, block=golden.ONION_BLOCK):
        self._trace, self.enabled, self.block = trace, enabled, block

    def module(self):
        return golden._module(_LIBRARIAN, onion_enabled=self.onion_enabled, memory_block=self.memory_block,
                              maybe_curate=self.maybe_curate)

    def onion_enabled(self, path=None):
        return self.enabled

    def memory_block(self, conversation_id, question=None, **kwargs):
        self._trace.add("trip.memory_block", conversation=conversation_id)
        return self.block

    def maybe_curate(self, conversation_id, messages, **kwargs):
        self._trace.add("curate", conversation=conversation_id, messages=messages, **kwargs)
        return True


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
class Store(golden.Store):
    """The golden's store, its mirror read as the application's store hands the onion its turns.

    Each turn comes with its origin, its segments, its context and its
    lineage: a user turn's label read from its own parts, an answer's the
    one it was saved with, legacy otherwise. A turn of the history given
    with its origin, segments, context or lineage keeps them.
    """

    def __init__(self, trace, *, history=None, **kwargs):
        super().__init__(trace, history={cid: [{"role": m["role"], "content": m["content"]} for m in turns]
                                         for cid, turns in (history or {}).items()}, **kwargs)
        for cid, turns in (history or {}).items():
            self.kept[cid] = [(m["role"], m.get("origin"), m.get("segments"),
                               (m.get("context"), m.get("lineage")) if "context" in m else None) for m in turns]

    def _mirror_messages(self, conversation_id):
        wrapper = sys.modules[_WRAPPER]
        out = []
        for message, (role, origin, segments, label) in zip(self.messages.get(conversation_id, []),
                                                             self.kept.get(conversation_id, [])):
            if role == "user" and origin is not None:
                context, lineage = wrapper.user_turn_label(message["content"], origin, segments or [])
            elif role == "assistant" and label is not None and label[0] is not None:
                context, lineage = label[0], label[1] or []
            else:
                context, lineage = ["legacy"], []
            out.append(dict(message, origin=origin or "legacy", segments=[list(s) for s in segments or []],
                            context=list(context), lineage=list(lineage)))
        return out


# ---------------------------------------------------------------------------
# The governors
# ---------------------------------------------------------------------------
def governor(trace, *, admitted=True, action="admit", num_ctx=16384, keep_alive=None, on_admit=None,
             on_ticket=None):
    """The governor as the journal reaches it: ``(seeded modules, state)``.

    Each admission is traced, and an admitted one held in ``state.admitted``
    and ``state.pending``, as the real governor holds an interactive
    admission and the load it claims. A ticket held ends the admitted-only
    hold; ``ticket_scope`` (a real scope) and ``end_pending_load`` give an
    admission back, both traced. ``num_ctx`` is the context admitted, or one
    for each admission in turn.
    """
    state = SimpleNamespace(admitted=set(), pending=set(), decisions=[])

    def admit(model, requested_ctx=None, caller="chat", extra_models=None):
        trace.add("admit", model=model, requested_ctx=requested_ctx, caller=caller)
        given = num_ctx[len(state.decisions)] if isinstance(num_ctx, (tuple, list)) else num_ctx
        decision = SimpleNamespace(
            admitted=admitted, action=action, reason="" if admitted else "not enough memory",
            num_ctx=given if admitted else None, keep_alive=keep_alive, conditional_on_eviction=False,
            load_expected=admitted, ticket_id=f"ticket-{len(state.decisions) + 1}",
            refusal_payload=lambda: {"message": "refused: not enough memory"},
        )
        state.decisions.append(decision)
        if admitted:
            state.admitted.add(decision.ticket_id)
            state.pending.add(decision.ticket_id)
        if on_admit is not None:
            on_admit()
        return decision

    def set_active_ticket(held):
        trace.released.clear()
        trace.add("ticket.set", action=getattr(held, "action", None))
        if held is not None:
            state.admitted.discard(held.ticket_id)
        if on_ticket is not None:
            on_ticket()

    def clear_active_ticket():
        trace.add("ticket.clear")
        trace.released.set()

    @contextmanager
    def ticket_scope(decision):
        trace.add("forgo.hold", ticket=getattr(decision, "ticket_id", None))
        if decision is not None:
            state.admitted.discard(decision.ticket_id)
        try:
            yield
        finally:
            trace.add("forgo.release")

    def end_pending_load(ticket_id):
        trace.add("forgo.end", ticket=ticket_id)
        state.pending.discard(ticket_id)

    real = SimpleNamespace(config=SimpleNamespace(enabled=True), admit=admit, end_pending_load=end_pending_load,
                           invalidate_on_load=lambda *a, **k: trace.add("trip.invalidate_on_load"))
    return {_GOVERNOR: golden._module(
        _GOVERNOR, get_resource_governor=lambda: real, set_active_ticket=set_active_ticket,
        clear_active_ticket=clear_active_ticket, ticket_scope=ticket_scope,
    )}, state


def _db_utils():
    """A db_utils stand-in whose safe_connect is plain sqlite, as the governor's class contracts seat it."""
    import sqlite3

    return golden._module("opti_oignon.db_utils", safe_connect=lambda p, **kw: sqlite3.connect(
        str(p), check_same_thread=kw.get("check_same_thread", False)))


def real_governor(rg, *, capacity=24.0, weights=None, ram_mb=64000.0):
    """The governor itself, its config set by hand, a snapshot that shows no model loaded: ``(governor, clock)``.

    The recipe of the governor's class contracts: a clock that stands
    still, no hardware, no warm-up, no registry, a database of its own;
    ``ram_mb`` of memory available.
    """
    clock = SimpleNamespace(t=1000.0)
    config = rg.GovernorConfig(total_vram_gb=capacity, safety_margin_gb=1.5, kv_coefficient=0.5)
    config.weights_override_models = dict(weights or {MODEL: 1.0})
    gov = rg.ResourceGovernor(
        config_path="/nonexistent-resource-governor-config", db_path=str(Path(tempfile.mkdtemp()) / "governor.db"),
        warmup=None, registry=None, clock=lambda: clock.t, meminfo_path="/nonexistent-meminfo", hardware=None,
    )
    gov._config = config
    gov._snapshot = rg.ResourceSnapshot(taken_at=clock.t, ttl_s=9999.0, loaded=[], capacity_gb=capacity,
                                        vram_in_use_gb=0.0, ram_available_mb=ram_mb)
    return gov, clock


# ---------------------------------------------------------------------------
# The ledger, every row kept whole
# ---------------------------------------------------------------------------
def ledger(trace):
    rows = []

    def record(**fields):
        trace.released.wait(golden._RELEASE_WAIT_S)
        rows.append(dict(fields))
        trace.add("ledger", outcome=fields.get("outcome"))

    return {"opti_oignon.context_ledger": golden._module(
        "opti_oignon.context_ledger", get_context_ledger=lambda: SimpleNamespace(record=record))}, rows


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class Engine(golden.Scripted):
    """The scripted client, whose last chunk reports what an engine reports when told (``extra``: its counts)."""

    def __init__(self, extra=None):
        super().__init__()
        self.extra = dict(extra or {})

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return iter([{"message": {"content": "Hello"}},
                     {"message": {"content": " world"}, "done": True, **self.extra}])


# ---------------------------------------------------------------------------
# The worlds
# ---------------------------------------------------------------------------
def open_world(*, journal=None, three_names=None, gov=None, real_gov=None, offline=False, hit=False,
               projects=False, web=False, backend="ollama", slots=None, engine_context=None, limits=None,
               withdrawn=None, history=None, stop_at=None, integrated=False, optimizer=True, engine_extra=None):
    """Seat the hub with the stand-ins asked and open its window; ``(world, close)``.

    ``journal`` is the ``JournalLibrarian``'s keywords (the librarian by
    default); ``three_names`` seats a ``ThreeNames`` with its keywords
    instead. ``gov`` is the traced governor's keywords, ``real_gov`` the
    real governor's (``capacity``, ``weights``); neither seats none.
    ``slots`` is the llama-server engine's listing (without it, it lists
    none); ``engine_context`` the context an in-process engine loads at
    when no admission names one. ``limits`` seats the model's limits (window, output); without
    them the hub reads 32768. ``stop_at`` maps ``plan``, ``admission``,
    ``lookup``, ``thread`` and ``generating`` to the turns whose stop is
    raised there. ``engine_extra`` is what the engine reports on its last
    chunk. ``integrated`` adds the optimizer, the summary tiers, the
    compressor and the sliding window, as the golden's integrated window
    seats them, the optimizer on unless ``optimizer`` is false.
    """
    trace, requests = golden.Trace(explain=True), [[]]
    stops = dict(stop_at or {})
    world = SimpleNamespace(trace=trace, requests=requests, run=None, turn=0, gov_state=None, journal=None,
                            three_names=None, stops=stops, statuses=[], governor=None)
    world.scripted = Engine(engine_extra)
    recording = golden.RecordingBackend(world.scripted, trace, requests, name=backend)
    if backend == "llama_server" and slots is not None:
        recording.slots = lambda: [dict(entry) for entry in slots]
    if engine_context is not None:
        recording._n_ctx = engine_context
    world.backend = recording
    store = Store(trace, history=history or {}, labels=integrated, archive=False)
    if withdrawn is not None:
        store.withdrawn_sources = lambda: list(withdrawn)
    world.store = store
    seeded = {
        "opti_oignon.config": golden._module("opti_oignon.config", config=SimpleNamespace(
            get_model=lambda *a, **k: MODEL, get_temperature=lambda *a, **k: 0.2)),
        "opti_oignon.router": golden._module("opti_oignon.router", RoutingResult=type("RoutingResult", (), {})),
        "opti_oignon.conversation": golden._module("opti_oignon.conversation", conversation_manager=store),
        _BACKEND: golden._registry_module(world.backend),
    }
    seeded.update(golden._memory(trace, placed=integrated))
    seeded.update(golden._caches(trace, hit_keys=("*",) if hit else (), on_get=_stop_at(world, stops.get("lookup"))))
    seeded.update(golden._network(trace, online=not offline))
    if gov is not None:
        gov_seeded, world.gov_state = governor(trace, **gov, on_admit=_stop_at(world, stops.get("admission")),
                                               on_ticket=_stop_at(world, stops.get("thread")))
        seeded.update(gov_seeded)
    if projects:
        project_seeded, world.projects = golden._projects(trace)
        seeded.update(project_seeded)
    if web:
        seeded.update(golden._web(trace, gated=integrated))
    if limits is not None:
        cm_seeded, world.context_manager = golden._context_manager(trace, window=limits[0], output=limits[1])
        seeded.update(cm_seeded)
    targets = {_WRAPPER: source("agent", "untrusted_context.py"), _DEDUP: source("context_dedup.py")}
    if real_gov is not None:
        targets[_GOVERNOR] = source("resource_governor.py")
        seeded["opti_oignon.db_utils"] = _db_utils()
    if three_names is not None:
        world.three_names = ThreeNames(trace, **three_names)
        seeded[_LIBRARIAN] = world.three_names.module()
    else:
        world.journal = JournalLibrarian(trace, **(journal or {}), on_plan=_stop_at(world, stops.get("plan")))
        seeded[_LIBRARIAN] = world.journal.module()
    if integrated:
        seeded.update(golden._integrated_base(trace))
        if offline:
            seeded.update(golden._network(trace, online=False))
        if limits is None:
            cm_seeded, world.context_manager = golden._context_manager(trace, window=32768, output=4096)
            seeded.update(cm_seeded)
        po_seeded, world.budget_manager = golden._prompt_optimization(trace, history_tokens=3200)
        seeded.update(po_seeded)
        co_seeded, world.compressor = golden._compressor(trace, enabled=True, trigger=True,
                                                         archive=golden._ARCHIVE_RESULTS)
        seeded.update(co_seeded)
        seeded.update(golden._summarizer(trace, available=True))
        sw_seeded, world.sliding = golden._sliding(trace, keep_last=2)
        seeded.update(sw_seeded)
        targets[_TIERS] = source("context_summary_tiers.py")
        targets[_OPTIMIZER] = source("context_optimizer.py")
    ledger_seeded, world.rows = ledger(trace)
    seeded.update(ledger_seeded)
    targets[_EXECUTOR] = source("executor.py")
    world, close = _seat(world, targets, seeded, integrated=integrated, optimizer=optimizer)
    if real_gov is not None:
        rg = world.loaded[_GOVERNOR]
        world.governor, world.governor_clock = real_governor(rg, **real_gov)
        rg._governor = world.governor
    return world, close


def open_real(*, gov=None, hit=False, projects=False, web=False, history=None, withdrawn=None, limits=(32768, 4096),
              engine_extra=None, project_text=None, web_results=None, estimate=None):
    """The onion's own modules beside the hub, its switch read on, nothing written: ``(world, close)``.

    The librarian keeps no state on disk (no persistence, no counts file)
    and starts no burst thread; everything else as ``open_world``, the
    retrieval answering ``project_text`` and the search ``web_results`` when
    they are given, and the turn counted with ``estimate`` when it is.
    """
    trace, requests = golden.Trace(explain=True), [[]]
    world = SimpleNamespace(trace=trace, requests=requests, run=None, turn=0, gov_state=None, journal=None,
                            three_names=None, stops={}, statuses=[], governor=None)
    world.scripted = Engine(engine_extra)
    world.backend = golden.RecordingBackend(world.scripted, trace, requests, name="ollama")
    store = Store(trace, history=history or {}, labels=True, archive=False)
    if withdrawn is not None:
        store.withdrawn_sources = lambda: list(withdrawn)
    world.store = store
    seeded = {
        "opti_oignon.config": golden._module("opti_oignon.config", config=SimpleNamespace(
            get_model=lambda *a, **k: MODEL, get_temperature=lambda *a, **k: 0.2)),
        "opti_oignon.router": golden._module("opti_oignon.router", RoutingResult=type("RoutingResult", (), {})),
        "opti_oignon.conversation": golden._module("opti_oignon.conversation", conversation_manager=store),
        _BACKEND: golden._registry_module(world.backend),
    }
    seeded.update(golden._memory(trace, placed=False))
    seeded.update(golden._caches(trace, hit_keys=("*",) if hit else ()))
    seeded.update(golden._network(trace))
    if gov is not None:
        gov_seeded, world.gov_state = governor(trace, **gov)
        seeded.update(gov_seeded)
    if projects:
        project_seeded, world.projects = golden._projects(trace)
        if project_text is not None:
            world.projects.build_context = lambda *a, **k: SimpleNamespace(
                context_text=project_text, chunks_used=1, total_tokens_estimate=len(project_text) // 3)
        seeded.update(project_seeded)
    if web:
        web_seeded = golden._web(trace, gated=False)
        if web_results is not None:
            web_seeded["opti_oignon.web_search"].web_search_engine.search = (
                lambda query, max_results=5: [dict(result) for result in web_results])
        seeded.update(web_seeded)
    cm_seeded, world.context_manager = golden._context_manager(trace, window=limits[0], output=limits[1])
    if estimate is not None:
        cm_seeded["opti_oignon.context_manager"].estimate_tokens = estimate
        world.context_manager.estimate_tokens = estimate
    seeded.update(cm_seeded)
    ledger_seeded, world.rows = ledger(trace)
    seeded.update(ledger_seeded)
    targets = {f"opti_oignon.memory.{name}": source("memory", f"{name}.py") for name in _ONION}
    targets.update({_WRAPPER: source("agent", "untrusted_context.py"), _DEDUP: source("context_dedup.py"),
                    _EXECUTOR: source("executor.py")})
    world, close = _seat(world, targets, seeded)
    lib = world.loaded[_LIBRARIAN]
    lib.reset_librarian()
    loading = lib.load_config
    lib.load_config = lambda path=None: replace(loading(path), enabled=True, persist_path="", counters_path="")
    lib._default_runner = lambda conversation_id: None
    world.librarian = lib
    return world, close


def _stop_at(world, turns):
    """A hook that raises the turn's stop where the stand-in calling it stands, on each turn of ``turns``."""
    if not turns:
        return None

    def raise_stop():
        if world.turn in turns:
            world.trace.add("stop")
            world.run.stop.set()

    return raise_stop


def _seat(world, targets, seeded, *, integrated=False, optimizer=True):
    had, previous = "ollama" in sys.modules, sys.modules.get("ollama")
    sys.modules["ollama"] = golden._module("ollama", chat=world.scripted.chat)
    try:
        loaded, close_window = isolate(targets=targets, seeded=seeded,
                                       packages=("opti_oignon.agent", "opti_oignon.memory"))
    except BaseException:
        golden._restore_ollama(had, previous)
        raise
    world.loaded = loaded
    world.executor_module, world.wrapper = loaded[_EXECUTOR], loaded[_WRAPPER]
    clock = golden.StillClock()
    world.executor_module.time = clock
    if integrated:
        loaded[_TIERS].time = clock
        optimizer_module = loaded[_OPTIMIZER]
        optimizer_module.time = clock
        optimizer_module.init_optimizer(
            config={"enabled": optimizer, "compression": {"strategy": "auto"}, "stable_prefix": {"enabled": False},
                    "slot_affinity": {"enabled": False},
                    "emergency": {"enabled": True, "min_recent_messages": 2, "max_block_chars": 2000},
                    "report": {"max_retained": 10}},
            budget_manager=world.budget_manager, project_context_builder=getattr(world, "projects", None),
            conversation_compressor=world.compressor, sliding_window_manager=world.sliding,
            context_manager=world.context_manager,
        )
        world.optimizer = optimizer_module.get_optimizer()

    def close():
        close_window()
        golden._restore_ollama(had, previous)

    return world, close


# ---------------------------------------------------------------------------
# Driving the hub
# ---------------------------------------------------------------------------
def routing(model=MODEL):
    return SimpleNamespace(model=model, task_type="general", temperature=0.2, prompt_variant="standard", timeout=30)


def run_turns(world, turns, *, executor=None):
    """Drive each turn of ``turns`` -- ``(conversation_id, question, keywords)`` -- through one executor.

    A turn's keywords go to ``execute``, but for ``refine`` (False unless
    given), ``claim`` (the run carries the user's claim on the question
    unless false), ``typed`` (the words the claim says the user typed, when
    the question sent is a rewrite of them), ``pasted`` (the ranges of the
    typed words the claim says were pasted), ``model`` (the model the turn
    is routed to), ``routed_images`` (images the routing carries) and
    ``raise_at`` (the head of a status line the caller fails on). Each
    turn's record: what was sent, what was yielded and returned, the run's
    results, the status lines and the trace.
    """
    executor = executor or world.executor_module.Executor()
    records = []
    for index, (conversation_id, question, kwargs) in enumerate(turns):
        if index:
            world.trace.next_turn()
            world.requests.append([])
        run = SimpleNamespace(stop=threading.Event(), results={})
        world.run, world.turn = run, index
        kwargs = dict(kwargs)
        refine = kwargs.pop("refine", False)
        claimed = kwargs.pop("claim", True)
        pasted = kwargs.pop("pasted", ())
        typed = kwargs.pop("typed", question)
        model = kwargs.pop("model", MODEL)
        routed = routing(model)
        routed_images = kwargs.pop("routed_images", None)
        if routed_images:
            routed.images = list(routed_images)
        if claimed:
            run.user_turn = world.executor_module.user_turn(typed, question, kwargs.get("documents") or (), pasted)
        said = []
        stop_here = index in (world.stops.get("generating") or ())
        raise_at = kwargs.pop("raise_at", None)

        def on_status(line, said=said, run=run, stop_here=stop_here, raise_at=raise_at):
            said.append(line)
            if stop_here and line.startswith("[>] Generating"):
                world.trace.add("stop")
                run.stop.set()
            if raise_at and line.startswith(raise_at):
                raise RuntimeError("the caller's status line failed")

        generator = executor.execute(question, routed, refine=refine, conversation_id=conversation_id,
                                     run=run, on_status=on_status, **kwargs)
        yields = []
        try:
            while True:
                item = next(generator)
                if item != "":
                    world.trace.add("yield", text=str(item)[:80])
                    yields.append(item)
        except StopIteration as stop:
            returned = stop.value
        records.append({"requests": world.requests[-1], "yields": yields, "returned": returned,
                        "results": dict(run.results), "statuses": said, "trace": world.trace.turns[-1]})
    world.executor = executor
    return records


def kinds(record):
    """The kinds of a turn's trace, in order."""
    return [event[0] for event in record["trace"]]


def sent(world):
    """Every request the engine was sent, whole, as the client received it."""
    return list(world.scripted.calls)
