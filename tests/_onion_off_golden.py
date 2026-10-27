#!/usr/bin/env python3
"""The chat executor's requests with the onion memory off, captured case by case.

With the onion memory off, what the chat executor hands every seam it works
through -- the backend, the response and semantic caches, the network check
and its queue, the resource governor and its ticket, the conversation store,
the capture and the librarian's write side, and each collaborator it asks for
context -- is held byte for byte by the golden contracts
(``tests/test_executor_onion_off_golden_contracts.py``). This module captures
it, from a matrix of cases in two windows, both opened through the shared
isolation window:

  * the isolated window of the onion wiring contracts: the hub over a
    scripted registry, with one stand-in per case beside it, so the
    executor's own fallbacks run (its token estimate, its context limits);
  * an integrated window: the context optimizer, the summary tiers, the
    untrusted-data envelope and the deduplicator as they are, every other
    module the executor imports as a stand-in that answers fixed inputs,
    and a clock that stands still.

Each case records, turn by turn: every request the backend received (its
digest over the canonical JSON of the whole request, its small fields as
they are, and each message as its role, length and digest), the non-empty
yields, the value the call returned, what it wrote into its run, and the
ordered trace of the seams it reached. Long strings are kept as their length
and digest. A case also records the facts that show its trigger fired.

The capture runs in child processes under two hash seeds, each covered by
the data firewall the way every child of the suite is. Bytecode the children
compile goes to a directory under the system's temporary directory, never
into the tree. The golden file is written only on purpose:
``python3 tests/_onion_off_golden.py --write`` refuses a tree whose sources
differ from HEAD and seeds that disagree, writes the file and prints the
digest the contracts pin; ``--check`` compares and writes nothing;
``--explain NAME`` prints every field of the matching cases, to read a
difference the digests only point at: run it on the change and on the
tree before it (a checkout of the base) to read both sides. A test never
writes the file.

Local-only (the public distribution ships no tests). Stdlib-only apart from
the windows it loads.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time as _time
import types
from pathlib import Path
from types import SimpleNamespace

if __name__ == "__main__":
    # Run as a script, nothing it imports, nor any child it starts, leaves bytecode beside the sources.
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from _isolation import REPO, isolate, source  # noqa: E402
from _registry_bridge import ScriptedBackend, StubRegistry  # noqa: E402

FIXTURE = TESTS / "executor_golden" / "v2" / "onion_off.json"
SEEDS = ("0", "1")
FORMAT = 1

MODEL = "test-model:1b"
LEGACY_BLOCK = "Remembered facts:\n- the user tends their onion garden at dawn"
ONION_BLOCK = "[data layer=core]\nThe user is called Alice.\n[/data]"
FIRST, SECOND = "What is my name?", "And where does the garden grow?"

_EXECUTOR = "opti_oignon.executor"
_WRAPPER = "opti_oignon.agent.untrusted_context"
_DEDUP = "opti_oignon.context_dedup"
_OPTIMIZER = "opti_oignon.context_optimizer"
_TIERS = "opti_oignon.context_summary_tiers"
_LIBRARIAN = "opti_oignon.memory.librarian"
_BACKEND_MODULE = "opti_oignon.inference_backend"

# The longest string kept as it is in a record; longer ones are kept as
# their length and the head of their digest.
_VERBATIM = 48


# ---------------------------------------------------------------------------
# Canonical forms
# ---------------------------------------------------------------------------
def sha(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def short(text):
    """A string as its length and the head of its digest."""
    text = str(text)
    return [len(text), sha(text)[:16]]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def shape(value):
    """``value`` as a record keeps it: short strings as they are, long ones digested, objects by their fields."""
    if isinstance(value, str):
        return value if len(value) <= _VERBATIM else short(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): shape(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, dict) and "role" in item for item in value):
            # A list of messages: how many, and the digest of all of them, labels and fields included.
            return {"messages": len(value), "sha256": sha(canonical(_plain(value)))[:16]}
        return [shape(item) for item in value]
    if isinstance(value, SimpleNamespace):
        return shape(vars(value))
    if callable(value):
        return "<callable>"
    return shape(getattr(value, "__dict__", None) or type(value).__name__)


def _plain(value):
    """``value`` with every object made of plain JSON types, so it can be digested whole."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return _plain(getattr(value, "__dict__", None) or type(value).__name__)


# The fields of an event kept as they are, beside the digest of all of them:
# enough to read the order and the outcome of a turn at a glance.
_NAMED = ("action", "caller", "hit", "online", "origin", "outcome", "request", "requested_ctx", "role")


class Trace:
    """The seams a turn reached, in order: one list per turn.

    Each event is its kind, the fields of ``_NAMED`` it carries, and the
    digest of all its fields as ``shape`` keeps them; ``explain`` keeps the
    fields themselves instead, to read a difference.

    ``released`` is set while no admission ticket is held. The hub releases
    a ticket on its stream's thread and may write its ledger on the other
    before that thread has run again, after a cancel: the ledger waits for
    the release, at most ``_RELEASE_WAIT_S``, so the order of the two does
    not depend on the scheduler. A ticket never released still shows: its
    turn's trace has no release.
    """

    def __init__(self, explain=False):
        self.turns = [[]]
        self._explain = explain
        self.released = threading.Event()
        self.released.set()

    def add(self, kind, **fields):
        if not fields:
            self.turns[-1].append([kind])
            return
        shaped = shape(fields)
        if self._explain:
            self.turns[-1].append([kind, shaped])
            return
        kept = {name: shaped[name] for name in _NAMED if name in shaped}
        if len(kept) < len(shaped):
            kept["fields"] = sha(canonical(shaped))[:16]
        self.turns[-1].append([kind, kept])

    def next_turn(self):
        self.turns.append([])

    def kinds(self):
        return [kind for turn in self.turns for kind, *_rest in turn]


# ---------------------------------------------------------------------------
# The backend: the scripted client of the wiring contracts, recorded
# ---------------------------------------------------------------------------
class Scripted:
    """The scripted client of the wiring contracts: every request answered "Hello world", after ``thinking``."""

    def __init__(self, thinking=""):
        self.calls = []
        self._thinking = thinking

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        chunks = [{"message": {"thinking": self._thinking, "content": ""}}] if self._thinking else []
        return iter(chunks + [{"message": {"content": "Hello"}}, {"message": {"content": " world"}, "done": True}])


class RecordingBackend(ScriptedBackend):
    """The bridge's backend, which also keeps every request whole, with any argument the hub adds."""

    def __init__(self, scripted, trace, requests, name="ollama", on_generate=None):
        super().__init__(scripted)
        self.name = name
        self._trace, self._requests, self._on_generate = trace, requests, on_generate

    def _record(self, call, model, messages, options, keep_alive, think, images, extra):
        whole = {
            "call": call, "model": model, "messages": [dict(m) for m in messages or []], "options": options,
            "keep_alive": keep_alive, "think": bool(think), "images": list(images) if images else None,
            "extra": {key: shape(value) for key, value in extra.items()},
        }
        digest = sha(canonical(whole))
        self._requests[-1].append({
            "call": call, "sha256": digest, "model": model, "options": shape(options), "keep_alive": keep_alive,
            "think": bool(think), "images": len(images) if images else None, "extra": sorted(extra),
            "messages": [[m.get("role"), *short(m.get("content", ""))] for m in messages or []],
        })
        self._trace.add(call, request=digest[:16])

    def stream(self, model, messages, options=None, keep_alive="30m", think=False, images=None, **extra):
        self._record("stream", model, messages, options, keep_alive, think, images, extra)
        return super().stream(model, messages, options, keep_alive, think, images)

    def generate(self, model, messages, options=None, keep_alive="30m", think=False, images=None, **extra):
        self._record("generate", model, messages, options, keep_alive, think, images, extra)
        if self._on_generate is not None:
            self._on_generate()
        return super().generate(model, messages, options, keep_alive, think, images)


def _registry_module(backend):
    registry = StubRegistry()
    registry.register(backend)
    registry.activate(backend.name)
    module = types.ModuleType(_BACKEND_MODULE)
    module.get_backend_registry = lambda: registry
    return module


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


# ---------------------------------------------------------------------------
# A clock that stands still
# ---------------------------------------------------------------------------
class StillClock:
    """``time`` as the executor, the optimizer and the summary tiers read it, standing still at one instant (UTC).

    Every reading of the clock those modules make is stilled -- ``time``,
    ``monotonic``, ``perf_counter`` and their ``_ns`` forms, ``strftime``
    (with a ``struct_time`` or none), ``localtime`` and ``gmtime`` -- and any
    other name fails loudly, so a module that starts reading the clock
    another way stops the capture instead of moving it. None of the three
    imports ``datetime``; ``_open`` refuses a window where one does.
    """

    NOW = 1767225600.0  # 2026-01-01T00:00:00Z

    def time(self):
        return self.NOW

    def time_ns(self):
        return int(self.NOW) * 1_000_000_000

    def monotonic(self):
        return 1000.0

    def monotonic_ns(self):
        return 1000 * 1_000_000_000

    def perf_counter(self):
        return 1000.0

    def perf_counter_ns(self):
        return 1000 * 1_000_000_000

    def gmtime(self, seconds=None):
        return _time.gmtime(self.NOW if seconds is None else seconds)

    def localtime(self, seconds=None):
        return _time.gmtime(self.NOW if seconds is None else seconds)

    def strftime(self, fmt, moment=None):
        return _time.strftime(fmt, self.gmtime() if moment is None else moment)

    def __getattr__(self, name):
        raise AttributeError(f"the still clock does not answer time.{name}: still it here before it is read")


# ---------------------------------------------------------------------------
# The conversation store
# ---------------------------------------------------------------------------
class Store:
    """A conversation store that remembers what it was handed and says what it was asked to keep.

    It hands back the very messages it holds, as the wiring contracts' store
    does. ``labels`` adds the labelled reader the application's store has,
    each turn under the label it was written with; ``archive`` adds the read
    of the archive with ids the summary tiers verify against.
    """

    def __init__(self, trace, *, history=None, labels=False, archive=False, metadata=None, mirror=True):
        self._trace = trace
        self.messages = {cid: [dict(m) for m in turns] for cid, turns in (history or {}).items()}
        self.kept = {cid: [(m["role"], None, None, None) for m in turns] for cid, turns in (history or {}).items()}
        self.metadata = {cid: dict(value) for cid, value in (metadata or {}).items()}
        self._labels, self._archive, self._mirror = labels, archive, mirror

    def add_message(self, conversation_id, role, content, **kwargs):
        self._trace.add("store.add", conversation=conversation_id, role=role, content=content, **kwargs)
        self.messages.setdefault(conversation_id, []).append({"role": role, "content": content})
        self.kept.setdefault(conversation_id, []).append(
            (role, kwargs.get("origin"), kwargs.get("segments"), (kwargs.get("context"), kwargs.get("lineage"))))

    def get_context_messages(self, conversation_id, **kwargs):
        return list(self.messages.get(conversation_id, []))

    def get_conversation(self, conversation_id):
        return SimpleNamespace(id=conversation_id, messages=self.get_context_messages(conversation_id),
                               metadata=dict(self.metadata.get(conversation_id, {})))

    def update_conversation_metadata(self, conversation_id, *args, **kwargs):
        self._trace.add("store.metadata", conversation=conversation_id, **kwargs)
        if isinstance(kwargs.get("metadata"), dict):
            self.metadata.setdefault(conversation_id, {}).update(kwargs["metadata"])

    def __getattr__(self, name):
        if name == "get_mirror_messages" and self._mirror:
            return self._mirror_messages
        if name == "get_labelled_context_messages" and self._labels:
            return self._labelled_messages
        if name == "get_messages" and self._archive:
            return self._archive_messages
        raise AttributeError(name)

    def _mirror_messages(self, conversation_id):
        out = []
        for message, (role, origin, segments, _label) in zip(self.messages.get(conversation_id, []),
                                                              self.kept.get(conversation_id, [])):
            out.append(dict(message, origin=origin or "legacy", segments=list(segments or [])))
        return out

    def _labelled_messages(self, conversation_id):
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
            out.append(wrapper.labelled(message, context, lineage))
        return out

    def _archive_messages(self, conversation_id):
        return [SimpleNamespace(id=index + 1, role=m["role"], content=m["content"])
                for index, m in enumerate(self.messages.get(conversation_id, []))]


def history_of(pairs, *, size):
    """A deterministic history of ``pairs`` exchanges, each turn about ``size`` characters of plain words."""
    words = ("onion", "garden", "bulb", "spring", "harvest", "soil", "water", "shade", "frost", "seed")
    out = []
    for index in range(pairs):
        for role in ("user", "assistant"):
            text = f"{role} turn {index}: " + " ".join(words[(index + k) % len(words)] for k in range(size // 6))
            out.append({"role": role, "content": text[:size]})
    return out


# ---------------------------------------------------------------------------
# The librarian, as each posture presents it
# ---------------------------------------------------------------------------
class _Librarian:
    """A librarian fake: the read side answers as told and is never recorded; the write side is."""

    def __init__(self, trace, *, enabled, block, counter):
        self._trace, self.enabled, self.block, self._counter = trace, enabled, block, counter

    def module(self):
        module = _module(_LIBRARIAN, onion_enabled=self.onion_enabled, memory_block=self.memory_block,
                         maybe_curate=self.maybe_curate)
        if self._counter:
            module.use_counter = self.use_counter
        return module

    def onion_enabled(self, path=None):
        return self.enabled

    def memory_block(self, conversation_id, question=None, **kwargs):
        return self.block

    def use_counter(self, conversation_id, count, model):
        return None

    def maybe_curate(self, conversation_id, messages, **kwargs):
        self._trace.add("curate", conversation=conversation_id, messages=messages, **kwargs)
        return True


# The postures of the librarian with the onion off, and the one on with
# nothing to say: the request is the same under each.
POSTURES = ("counting-off", "three-names-off", "three-names-on", "real-off", "blocked")


# ---------------------------------------------------------------------------
# Stand-ins, one group per seam
# ---------------------------------------------------------------------------
def _caches(trace, hit_keys=(), on_get=None):
    """The response and semantic caches: every key the hub computes, every get and put, a hit on ``hit_keys``."""
    stored = {}

    def make_cache_key(model, system_prompt, user_content):
        key = "single-" + sha(canonical([model, system_prompt, user_content]))[:16]
        trace.add("cache.key", model=model, system=system_prompt, user=user_content, key=key)
        return key

    def make_conversation_cache_key(model, system_prompt, history, user_content):
        key = "turns-" + sha(canonical([model, system_prompt, history, user_content]))[:16]
        trace.add("cache.key", model=model, system=system_prompt, history=history, user=user_content, key=key)
        return key

    def get(key):
        if on_get is not None:
            on_get()
        entry = stored.get(key)
        trace.add("cache.get", key=key, hit=entry is not None)
        return entry

    def put(**kwargs):
        trace.add("cache.put", **kwargs)
        if kwargs.get("explicit_key") in hit_keys or "*" in hit_keys:
            stored[kwargs["explicit_key"]] = SimpleNamespace(response=kwargs["response"])

    response = SimpleNamespace(enabled=True, make_cache_key=make_cache_key,
                               make_conversation_cache_key=make_conversation_cache_key, get=get, put=put)

    def semantic_get(query, **kwargs):
        trace.add("semantic.get", query=query, **kwargs)
        return None

    def get_with_fallback(response_cache, cache_key, model, query, **kwargs):
        trace.add("semantic.fallback", key=cache_key, model=model, query=query, **kwargs)
        return None, 0.0, "miss"

    def semantic_put(**kwargs):
        trace.add("semantic.put", **kwargs)
        return "semantic-key"

    def store_embedding(**kwargs):
        trace.add("semantic.embed", **kwargs)

    semantic = SimpleNamespace(enabled=True, get=semantic_get, get_with_fallback=get_with_fallback,
                               put=semantic_put, store_embedding=store_embedding)
    return {
        "opti_oignon.response_cache": _module("opti_oignon.response_cache", response_cache=response),
        "opti_oignon.semantic_cache": _module("opti_oignon.semantic_cache", semantic_cache=semantic),
    }


class _Network:
    def __init__(self, trace, online):
        self._trace, self._online = trace, online

    @property
    def is_online(self):
        self._trace.add("network.online", online=self._online)
        return self._online


def _network(trace, online=True):
    def enqueue(query, task_type, model):
        trace.add("network.enqueue", query=query, task_type=task_type, model=model)
        return SimpleNamespace(id="queued-1")

    return {
        "opti_oignon.network_manager": _module("opti_oignon.network_manager", network_manager=_Network(trace, online)),
        "opti_oignon.sync_queue": _module("opti_oignon.sync_queue", sync_queue=SimpleNamespace(enqueue=enqueue)),
    }


# How long the ledger waits for a held ticket's release before it records anyway.
_RELEASE_WAIT_S = 2.0


def _ledger(trace):
    def record(**fields):
        trace.released.wait(_RELEASE_WAIT_S)
        trace.add("ledger", outcome=fields.get("outcome"))

    return {"opti_oignon.context_ledger": _module("opti_oignon.context_ledger",
                                                  get_context_ledger=lambda: SimpleNamespace(record=record))}


def _governor(trace, *, admitted=True, action="admit", num_ctx=None, keep_alive=None, on_admit=None,
              on_ticket=None):
    decision = SimpleNamespace(
        admitted=admitted, action=action, reason="" if admitted else "not enough memory", num_ctx=num_ctx,
        keep_alive=keep_alive, conditional_on_eviction=False, load_expected=False,
        refusal_payload=lambda: {"message": "refused: not enough memory"},
    )

    def admit(model, requested_ctx=None, caller="chat", extra_models=None):
        trace.add("admit", model=model, requested_ctx=requested_ctx, caller=caller, extra_models=extra_models)
        if on_admit is not None:
            on_admit()
        return decision

    def set_active_ticket(held):
        trace.released.clear()
        trace.add("ticket.set", action=held.action)
        if on_ticket is not None:
            on_ticket()

    def clear_active_ticket():
        trace.add("ticket.clear")
        trace.released.set()

    governor = SimpleNamespace(
        config=SimpleNamespace(enabled=True), admit=admit,
        # What an onion turn that ends early would call: never reached with the onion off.
        invalidate_on_load=lambda *a, **k: trace.add("trip.invalidate_on_load"),
        end_pending_load=lambda *a, **k: trace.add("trip.end_pending_load"),
    )
    return {"opti_oignon.resource_governor": _module(
        "opti_oignon.resource_governor",
        get_resource_governor=lambda: governor,
        set_active_ticket=set_active_ticket,
        clear_active_ticket=clear_active_ticket,
        ticket_scope=lambda *a, **k: trace.add("trip.ticket_scope"),
    )}


def _stop_at(world, turns):
    """A hook that, on each turn of ``turns``, raises the turn's stop where the stand-in calling it stands, and
    marks the place in the trace; None for no turn."""
    if not turns:
        return None

    def raise_stop():
        if world.turn in turns:
            world.trace.add("stop")
            world.run.stop.set()

    return raise_stop


def _slots(trace):
    """Slot affinity for the named llama-server backend: the slot it would name, recorded."""

    def choose(**kwargs):
        trace.add("slot.choose", **kwargs)
        return 1

    return {"opti_oignon.slot_affinity": _module("opti_oignon.slot_affinity", ENVELOPE_NONE="none",
                                                 get_slot_affinity=lambda: SimpleNamespace(choose=choose))}


_PROJECT_TEXT = "Project notes: the garden plan keeps onions in the east beds, garlic by the wall."


def _projects(trace, *, project_id="p-1"):
    def get_project_for_conversation(conversation_id):
        trace.add("project.lookup", conversation=conversation_id)
        return project_id

    def detect(question, project, **kwargs):
        trace.add("project.detect", question=question, project=project, kwargs=kwargs)
        return SimpleNamespace(relevant=True, trigger_level=1, confidence=0.9)

    def build_context(*args, **kwargs):
        trace.add("project.build", args=list(args), kwargs=kwargs)
        return SimpleNamespace(context_text=_PROJECT_TEXT, chunks_used=2, total_tokens_estimate=24)

    def build_system_instructions_only(project):
        trace.add("project.instructions", project=project)
        return SimpleNamespace(context_text="", chunks_used=0, total_tokens_estimate=0)

    builder = SimpleNamespace(available=True, build_context=build_context,
                              build_system_instructions_only=build_system_instructions_only)
    return {
        "opti_oignon.project_context": _module("opti_oignon.project_context", project_context_builder=builder),
        "opti_oignon.project_triggers": _module("opti_oignon.project_triggers",
                                                trigger_detector=SimpleNamespace(detect=detect)),
        "opti_oignon.projects": _module("opti_oignon.projects", project_store=SimpleNamespace(
            get_project_for_conversation=get_project_for_conversation)),
    }, builder


_WEB_RESULTS = (
    {"title": "Onion care", "snippet": "Onions like full sun and loose soil.", "url": "https://example.org/onions"},
    {"title": "Garden rows", "snippet": "Plant rows a hand apart.", "url": "https://example.org/rows"},
)


def _web(trace, *, gated):
    def search(query, max_results=5):
        trace.add("web.search", query=query, max_results=max_results)
        return [dict(result) for result in _WEB_RESULTS]

    seeded = {"opti_oignon.web_search": _module("opti_oignon.web_search",
                                                web_search_engine=SimpleNamespace(search=search))}
    if gated:
        class TurnProvenance:
            def __init__(self, demand=None):
                self.demand = demand

            @classmethod
            def of_user_turn(cls, claim, demand=None):
                return cls(demand)

        def check(tool, arguments, turn, defaults=None, network=False):
            trace.add("provenance.check", tool=tool, arguments=arguments, defaults=defaults, network=network)
            return SimpleNamespace(allowed=True, arguments=dict(arguments), message="")

        seeded["opti_oignon.search_killswitch"] = _module(
            "opti_oignon.search_killswitch", search_killswitch=SimpleNamespace(is_killed=lambda: False))
        seeded["opti_oignon.provenance"] = _module("opti_oignon.provenance", TurnProvenance=TurnProvenance,
                                                   check=check)
    return seeded


def _memory(trace, *, placed):
    """The legacy composer; with ``placed``, the composer that also says which facts it placed."""

    def build(question, **kwargs):
        trace.add("memory.build", question=question, kwargs=kwargs)
        return LEGACY_BLOCK

    module = _module("opti_oignon.memory.retrieval", build_memory_block=build, working_memory_block=build)
    if placed:
        def compose(question, **kwargs):
            trace.add("memory.compose", question=question, kwargs=kwargs)
            return LEGACY_BLOCK, [("fact-1", True), ("fact-2", False)]

        module.compose_memory_block = compose
    return {"opti_oignon.memory.retrieval": module}


# ---------------------------------------------------------------------------
# The integrated window's collaborators, answering fixed inputs
# ---------------------------------------------------------------------------
def _context_manager(trace, *, window, output):
    limits = SimpleNamespace(context_window=window, max_output=output)

    def estimate_tokens(text, model=None):
        return max(1, len(str(text)) // 3)

    def check_context(prompt, document, system_prompt, model):
        trace.add("context.check", prompt=prompt, document=document, system_prompt=system_prompt, model=model)
        total = estimate_tokens(prompt) + estimate_tokens(document) + estimate_tokens(system_prompt)
        return SimpleNamespace(
            exceeds_limit=False, truncation_needed=False, warning_message=None, total_tokens=total,
            available_for_input=window - output, prompt_tokens=estimate_tokens(prompt),
            system_tokens=estimate_tokens(system_prompt), document_tokens=estimate_tokens(document),
        )

    manager = SimpleNamespace(estimate_tokens=estimate_tokens, smart_truncate=lambda **k: (k.get("text", ""), 0))
    module = _module(
        "opti_oignon.context_manager", ContextCheck=SimpleNamespace, get_context_manager=lambda: manager,
        check_context=check_context, estimate_tokens=estimate_tokens, get_model_limits=lambda model: limits,
        smart_truncate=lambda *a, **k: ("", 0),
    )
    return {"opti_oignon.context_manager": module}, manager


_TEMPLATE_PROMPT = "You are a careful assistant for an onion grower. Answer in the user's language."


def _prompt_optimization(trace, *, history_tokens, total_window=8192, reserve=1200):
    template = SimpleNamespace(task_type="general", source="builtin", temperature_override=None)

    def get_template(task_type, project_id=None):
        trace.add("template.get", task_type=task_type, project=project_id)
        return template

    def interpolate(template, context=None):
        return _TEMPLATE_PROMPT

    def calculate_budget(**kwargs):
        trace.add("budget", **kwargs)
        return SimpleNamespace(total_window=total_window, system_tokens=800, project_tokens=1600,
                               history_tokens=history_tokens, user_tokens=800, reserve_tokens=reserve,
                               model=kwargs.get("model", ""), fingerprint_tokens=0)

    budget_manager = SimpleNamespace(calculate_budget=calculate_budget)
    return {"opti_oignon.prompt_optimization": _module(
        "opti_oignon.prompt_optimization", prompt_budget_manager=budget_manager,
        prompt_template_engine=SimpleNamespace(get_template=get_template, interpolate=interpolate),
    )}, budget_manager


_COMPRESSED_SUMMARY = "Earlier the user and the assistant spoke of the onion beds and of the spring planting."


def _compressor(trace, *, enabled, trigger=False, archive=()):
    class CompressedContext:  # the type the hub imports; the results below carry its fields
        pass

    def compress(messages, budget_tokens, model, **kwargs):
        trace.add("compress", messages=messages, budget_tokens=budget_tokens, model=model, kwargs=kwargs)
        recent = list(messages[-2:])
        return SimpleNamespace(summary=_COMPRESSED_SUMMARY, compressed_count=len(messages) - len(recent),
                               recent_messages=recent, strategy_used="rule", original_count=len(messages),
                               tokens_saved=100)

    def retrieve_from_archive(conversation_id, question):
        trace.add("archive.retrieve", conversation=conversation_id, question=question)
        return [SimpleNamespace(**result) for result in archive]

    def check_retrieval_trigger(question, min_confidence=0.6):
        trace.add("archive.trigger", question=question, min_confidence=min_confidence)
        return trigger

    compressor = SimpleNamespace(enabled=enabled, compress=compress, retrieve_from_archive=retrieve_from_archive,
                                 get_config=lambda: {"retrieval_trigger_min_confidence": 0.6})
    return {"opti_oignon.conversation_compressor": _module(
        "opti_oignon.conversation_compressor", CompressedContext=CompressedContext,
        check_retrieval_trigger=check_retrieval_trigger, conversation_compressor=compressor,
    )}, compressor


_SUMMARY = "The user asked about the onion beds; the assistant described spring planting."


def _summarizer(trace, *, available, text=_SUMMARY):
    def summarize_messages(messages=None, **kwargs):
        trace.add("summarize", messages=messages, **kwargs)
        return text

    summarizer = SimpleNamespace(available=available, SUMMARY_THRESHOLD=4, MAX_SUMMARY_TOKENS=60,
                                 summarize_messages=summarize_messages)
    return {"opti_oignon.context_summary": _module("opti_oignon.context_summary", context_summarizer=summarizer)}


def _sliding(trace, *, keep_last):
    def prepare_messages(history, model, system_tokens=0):
        trace.add("slide", history=history, model=model, system_tokens=system_tokens)
        kept = list(history[-keep_last:])
        return kept, {"strategy": "importance", "kept": len(kept), "dropped": len(history) - len(kept),
                      "total_tokens": sum(max(1, len(m["content"]) // 3) for m in kept)}

    manager = SimpleNamespace(prepare_messages=prepare_messages)
    return {"opti_oignon.context_window": _module("opti_oignon.context_window", sliding_window_manager=manager,
                                                  token_budget_manager=None)}, manager


def _integrated_base(trace):
    """What every integrated case seats: the capture, the warm-up's keep-alive, the ledger, the network."""

    def capture(conversation_id, messages):
        trace.add("capture", conversation=conversation_id, messages=messages)

    seeded = {
        "opti_oignon.memory.auto_capture": _module("opti_oignon.memory.auto_capture", maybe_capture=capture),
        "opti_oignon.model_warmup": _module("opti_oignon.model_warmup", MODEL_WARMUP_AVAILABLE=True,
                                            model_warmup=SimpleNamespace(keep_alive="45m")),
    }
    seeded.update(_ledger(trace))
    seeded.update(_network(trace))
    return seeded


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def _turns(*pairs):
    return tuple((conversation_id, question, dict(kwargs)) for conversation_id, question, kwargs in pairs)


_TWO = _turns(("conv-1", FIRST, {}), ("conv-1", SECOND, {}))
_SINGLE = _turns((None, FIRST, {}), (None, SECOND, {}))
_DOCUMENTS = _turns(
    ("conv-1", FIRST, {"documents": (("notes.txt", "Beds: east, west.\nOnions in the east."),
                                     ("plan.txt", "Spring: sow. Summer: water. Autumn: lift."))}),
    ("conv-1", SECOND, {}),
)
_ARCHIVE_RESULTS = (
    {"role": "user", "snippet": "user turn 0: onion garden bulb spring harvest soil water", "score": 0.8},
    {"role": "assistant", "snippet": LEGACY_BLOCK, "score": 0.7},
)


def _case(name, env, turns=_TWO, *, posture="counting-off", groups=(), checks=(), **options):
    return {"name": f"{env}/{name}", "env": env, "turns": turns, "posture": posture, "groups": tuple(groups),
            "checks": tuple(checks), "options": options}


def _kinds_in(record, kind):
    return sum(1 for turn in record["turns"] for event in turn["trace"] if event[0] == kind)


def _fired(kind, least=1):
    return (f"{kind} reached at least {least} time(s)", lambda record: _kinds_in(record, kind) >= least)


def _absent(kind):
    return (f"{kind} never reached", lambda record: _kinds_in(record, kind) == 0)


def _ended_offline():
    """The offline trigger: each turn ends there, said by the executor, and nothing keeps it for later."""
    return ("every turn ended offline, said by the executor",
            lambda record: all(turn["window"].get("offline_queued") for turn in record["turns"]))


def _seams(*kinds):
    """The seams a plain case reaches, and no other: what the cases built on it add shows as a difference."""
    return (f"the seams reached are exactly {sorted(kinds)}",
            lambda record: {event[0] for turn in record["turns"] for event in turn["trace"]} == set(kinds))


def _fact(key, expected):
    return (f"{key} is {expected!r}", lambda record: record["facts"].get(key) == expected)


def _no_stream_on(turn_index):
    return (f"no request on turn {turn_index + 1}", lambda record: not record["turns"][turn_index]["requests"])


def _first_options(key, expected):
    return (f"the first request's options carry {key}={expected!r}",
            lambda record: bool(record["turns"][0]["requests"])
            and record["turns"][0]["requests"][0]["options"].get(key) == expected)


def _thought():
    return ("a thinking pair yielded, and every request asked to think",
            lambda record: any(isinstance(item, list) and item[:1] == ["thinking"]
                               for item in record["turns"][0]["yields"])
            and all(request["think"] for turn in record["turns"] for request in turn["requests"]))


def _stops_after(turn, before):
    kinds = [event[0] for event in turn["trace"]]
    return ("stop" in kinds and kinds[kinds.index("stop") - 1:kinds.index("stop")] == [before]
            and not any(request["call"] == "stream" for request in turn["requests"])
            and turn["results"].get("cancelled") is True)


def _stopped_after(turn_index, before):
    return (f"turn {turn_index + 1} stopped right after {before}, sent no stream, and told its run it was cancelled",
            lambda record: _stops_after(record["turns"][turn_index], before))


def _refined_first():
    return ("the refinement is asked before the cache, on every turn",
            lambda record: all("generate" in kinds and "cache.key" in kinds
                               and kinds.index("generate") < kinds.index("cache.key")
                               for kinds in ([event[0] for event in turn["trace"]] for turn in record["turns"])))


def _slot_inside_ticket():
    return ("the slot is named after the ticket is taken and before the stream, on every turn",
            lambda record: all("ticket.set" in kinds and kinds[kinds.index("ticket.set"):kinds.index("ticket.set") + 3]
                               == ["ticket.set", "slot.choose", "stream"]
                               for kinds in ([event[0] for event in turn["trace"]] for turn in record["turns"])))


_SEATED = ("caches", "ledger", "network")
_THINKING = "Weighing the beds first."
_THINK = _turns(("conv-1", FIRST, {"think": True}), ("conv-1", SECOND, {"think": True}))
_REFINE = _turns(("conv-1", FIRST, {"refine": True}), ("conv-1", SECOND, {"refine": True}))
_MANIFEST = _turns(("conv-1", FIRST, {"capability_block": "Tools: none."}),
                   ("conv-1", SECOND, {"capability_block": "Tools: none."}))
_ADMITS = {"num_ctx": 16384, "keep_alive": "12m"}

# What each posture of the librarian is, read off the module the window seated in its place (never off the names
# the hub binds, which its read side may stop importing), and what it does on the write side.
_POSTURE_CHECKS = {
    "counting-off": [_fact("librarian_names", 4), _absent("curate"), _fact("librarian_is_the_real_one", False)],
    "three-names-off": [_fact("librarian_names", 3), _absent("curate")],
    "three-names-on": [_fact("librarian_names", 3), _fired("curate", 2)],
    "real-off": [_fact("librarian_names", 4), _fact("librarian_config_off", True),
                 _fact("librarian_is_the_real_one", True)],
    "blocked": [_fact("librarian_names", 0)],
}
_WIRING = _fact("scripted_digest_is_the_wiring_contracts", True)
_PLAIN_ISOLATED = ("cache.get", "cache.key", "cache.put", "ledger", "memory.build", "network.online", "store.add",
                   "store.metadata", "stream")
_PLAIN_INTEGRATED = ("budget", "cache.get", "cache.key", "cache.put", "capture", "ledger", "memory.compose",
                     "network.online", "store.add", "store.metadata", "stream", "template.get")
_UNPLACED = [_fact("capability_block_given", True), _fact("capability_block_placed", False)]

CASES = (
    # The isolated window: the wiring contracts' own, the onion off.
    _case("base", "isolated", checks=[_WIRING, *_POSTURE_CHECKS["counting-off"],
                                      _seams("memory.build", "store.add", "store.metadata", "stream")]),
    _case("single-turn", "isolated", _SINGLE, groups=_SEATED,
          checks=[_fired("cache.key", 2), _fired("semantic.get", 2)]),
    _case("manual", "isolated", groups=_SEATED,
          checks=[_fired("cache.key", 2), _fired("network.online", 2), _fact("window_strategy", "keep_all"),
                  _seams(*_PLAIN_ISOLATED)]),
    _case("drop-legacy", "isolated", groups=_SEATED, history=("conv-1", 10, 4500),
          checks=[_fact("fallback_legacy", True)]),
    _case("project-manual", "isolated", groups=_SEATED + ("projects",), checks=[_fired("project.build", 2)]),
    _case("web", "isolated", turns=_turns(("conv-1", FIRST, {"web_search": True}), ("conv-1", SECOND, {})),
          groups=_SEATED + ("web",), checks=[_fired("web.search")]),
    _case("documents", "isolated", _DOCUMENTS, groups=_SEATED, checks=[_fact("documents_in_first_request", 2)]),
    _case("manifest", "isolated", _MANIFEST, groups=_SEATED, checks=_UNPLACED),
    _case("governor-admits", "isolated", groups=_SEATED, governor=_ADMITS,
          checks=[_fired("admit", 2), _fired("ticket.set", 2), _fired("stream", 2), _first_options("num_ctx", 16384)]),
    _case("governor-steps-down", "isolated", groups=_SEATED,
          governor={"action": "downsize", "num_ctx": 4096, "keep_alive": "12m"},
          checks=[_fired("admit", 2), _first_options("num_ctx", 4096)]),
    _case("governor-refuses", "isolated", groups=_SEATED, governor={"admitted": False, "action": "refuse"},
          checks=[_fired("admit", 2), _no_stream_on(0), _no_stream_on(1)]),
    _case("cache-hit", "isolated", _turns(("conv-1", FIRST, {}), ("conv-2", FIRST, {})), groups=_SEATED,
          hit=True, checks=[_no_stream_on(1), _fact("second_turn_hit", True)]),
    _case("offline", "isolated", groups=("caches", "ledger"), offline=True,
          checks=[_ended_offline(), _absent("network.enqueue"), _no_stream_on(0)]),
    _case("think", "isolated", _THINK, groups=_SEATED, thinking=_THINKING, checks=[_thought()]),
    # A stop on the second turn, raised where each check of the hub stands: during the cache lookup, during the
    # admission, at the start of the stream's thread (where its ticket is taken), during the refinement.
    _case("stop-in-cache-lookup", "isolated", groups=_SEATED, stop_at={"lookup": (1,)},
          checks=[_stopped_after(1, "cache.key"), _fired("stream")]),
    _case("stop-in-admission", "isolated", groups=_SEATED, governor=_ADMITS, stop_at={"admission": (1,)},
          checks=[_stopped_after(1, "admit"), _fired("stream")]),
    _case("stop-at-thread-start", "isolated", groups=_SEATED, governor=_ADMITS, stop_at={"thread": (1,)},
          checks=[_stopped_after(1, "ticket.set"), _fired("stream")]),
    _case("stop-in-refinement", "isolated", _REFINE, groups=_SEATED, governor=_ADMITS,
          stop_at={"refinement": (1,)}, checks=[_stopped_after(1, "generate"), _fired("stream")]),
    _case("refine", "isolated", _REFINE, groups=_SEATED, governor=_ADMITS,
          checks=[_refined_first(), _fired("generate", 2), _fired("stream", 2)]),
    *(_case(f"librarian-{posture}", "isolated", posture=posture, checks=[_WIRING, *_POSTURE_CHECKS[posture]])
      for posture in POSTURES if posture != "counting-off"),
    # The integrated window: the optimizer, the tiers and the envelope as they are.
    _case("single-turn", "integrated", _SINGLE, checks=[_fired("semantic.get", 2), _fired("semantic.put", 2)]),
    _case("manual", "integrated", checks=[_fired("budget", 2), _fact("window_strategy", "keep_all"),
                                          _seams(*_PLAIN_INTEGRATED)]),
    _case("optimizer", "integrated", optimizer=True, checks=[_fact("window_strategy", "optimizer")]),
    _case("optimizer-raises", "integrated", optimizer=True, budget_raises=True,
          checks=[_fact("window_strategy", "keep_all"), _fired("budget.raises", 2)]),
    _case("compressor", "integrated", history=("conv-1", 6, 300), compressor=True, history_budget=40,
          checks=[_fired("compress", 2), _fact("summary_leads_the_first_request", True)]),
    _case("summary", "integrated", history=("conv-1", 12, 400), limits=(3000, 500),
          summarizer={"available": True},
          checks=[_fired("summarize"), _fired("store.metadata", 3), _fact("tiers_kept", False)]),
    _case("summary-tiers", "integrated", history=("conv-1", 12, 400), limits=(3000, 500),
          summarizer={"available": True}, archive=True, checks=[_fired("summarize"), _fact("tiers_kept", True)]),
    _case("sliding-window", "integrated", history=("conv-1", 12, 400), limits=(3000, 500),
          summarizer={"available": False}, sliding=4, checks=[_fired("slide"), _fact("window_dropped", True)]),
    # A summary too long for the window: the hard limit cuts the turns after it, then the summary itself.
    _case("hard-truncation", "integrated", history=("conv-1", 12, 400), limits=(3000, 500),
          summarizer={"available": True, "text": "summary " * 900},
          checks=[_fired("summarize"), _fact("first_history_count", 2),
                  _fact("summary_leads_the_first_request", False)]),
    _case("optimizer-mechanisms", "integrated", history=("conv-1", 12, 1500), optimizer=True, compressor=True,
          sliding=2, history_budget=30, window_budget=(1200, 200),
          checks=[_fact("history_zone", "compressed:rule+sliding_window+emergency")]),
    _case("project-optimizer-manifest", "integrated", _MANIFEST, optimizer=True, groups=("projects",),
          checks=[_fired("project.build", 2), _fact("manifest_second", True)]),
    _case("archive", "integrated", history=("conv-1", 2, 120), compressor=True, trigger=True,
          checks=[_fired("archive.retrieve", 2), _fact("archive_duplicate_dropped", True)]),
    _case("web", "integrated", turns=_turns(("conv-1", FIRST, {"web_search": True}), ("conv-1", SECOND, {})),
          groups=("web",), checks=[_fired("web.search"), _fired("provenance.check")]),
    # Every part the tail can take, in the order the hub joins them: memory, project, web results, archive.
    _case("every-tail-part", "integrated", history=("conv-1", 2, 120), compressor=True, trigger=True,
          groups=("projects", "web"),
          turns=_turns(("conv-1", FIRST, {"web_search": True}), ("conv-1", SECOND, {"web_search": True})),
          checks=[_fired("project.build", 2), _fired("web.search", 2), _fired("archive.retrieve", 2),
                  _fact("archive_listed", True)]),
    _case("documents", "integrated", _DOCUMENTS, checks=[_fired("context.check"),
                                                         _fact("documents_in_first_request", 2)]),
    _case("manifest", "integrated", _MANIFEST, checks=_UNPLACED),
    _case("governor-admits", "integrated", governor=_ADMITS,
          checks=[_fired("admit", 2), _fired("ticket.set", 2), _fired("stream", 2), _first_options("num_ctx", 16384)]),
    _case("governor-steps-down", "integrated", governor={"action": "downsize", "num_ctx": 4096},
          checks=[_fired("admit", 2), _first_options("num_ctx", 4096)]),
    _case("governor-refuses", "integrated", governor={"admitted": False, "action": "refuse"},
          checks=[_fired("admit", 2), _no_stream_on(0), _no_stream_on(1)]),
    _case("cache-hit", "integrated", _turns(("conv-1", FIRST, {}), ("conv-2", FIRST, {})), hit=True,
          checks=[_no_stream_on(1), _fact("second_turn_hit", True)]),
    _case("offline", "integrated", offline=True,
          checks=[_ended_offline(), _absent("network.enqueue"), _no_stream_on(0)]),
    # The engine named llama-server, with the prompt cache reused and the slot named: the off bytes carry the
    # switch and the slot, and no length cap.
    _case("named-llama-server", "integrated", llama_server=True, stable_prefix=True, slot_affinity=True,
          checks=[_fired("slot.choose", 2), _first_options("cache_prompt", True), _first_options("id_slot", 1)]),
    # The same under an admitting governor: the slot is named between the ticket and the stream.
    _case("named-llama-server-admitted", "integrated", llama_server=True, stable_prefix=True, slot_affinity=True,
          governor=_ADMITS, checks=[_slot_inside_ticket(), _first_options("num_ctx", 16384)]),
    _case("think", "integrated", _THINK, thinking=_THINKING, checks=[_thought()]),
    _case("stop-in-cache-lookup", "integrated", stop_at={"lookup": (1,)},
          checks=[_stopped_after(1, "cache.key"), _fired("stream")]),
    _case("refine", "integrated", _REFINE, governor=_ADMITS,
          checks=[_refined_first(), _fired("generate", 2), _fired("stream", 2)]),
    *(_case(f"librarian-{posture}", "integrated", posture=posture,
            checks=_POSTURE_CHECKS[posture] + ([_seams(*_PLAIN_INTEGRATED)] if posture == "counting-off" else []))
      for posture in POSTURES),
)


# ---------------------------------------------------------------------------
# A case, run
# ---------------------------------------------------------------------------
def _routing():
    return SimpleNamespace(model=MODEL, task_type="general", temperature=0.2, prompt_variant="standard", timeout=30)


def _open(case, trace, requests):
    """Seat a case's stand-ins and open its window; returns the world it runs in and how to close it."""
    options, groups, env = case["options"], set(case["groups"]), case["env"]
    integrated = env == "integrated"
    scripted = Scripted(thinking=options.get("thinking", ""))
    stops = options.get("stop_at", {})
    world = SimpleNamespace(scripted=scripted, trace=trace, requests=requests, run=None, turn=0, librarian=None)
    backend = RecordingBackend(scripted, trace, requests,
                               name="llama_server" if options.get("llama_server") else "ollama",
                               on_generate=_stop_at(world, stops.get("refinement", ())))
    world.backend = backend
    config = _module("opti_oignon.config", config=SimpleNamespace(get_model=lambda *a, **k: MODEL,
                                                                  get_temperature=lambda *a, **k: 0.2))
    router = _module("opti_oignon.router", RoutingResult=type("RoutingResult", (), {}))
    history = {}
    if "history" in options:
        conversation_id, pairs, size = options["history"]
        history[conversation_id] = history_of(pairs, size=size)
    store = Store(trace, history=history, labels=integrated, archive=bool(options.get("archive")))
    seeded = {
        "opti_oignon.config": config,
        "opti_oignon.router": router,
        "opti_oignon.conversation": _module("opti_oignon.conversation", conversation_manager=store),
        _BACKEND_MODULE: _registry_module(backend),
    }
    seeded.update(_memory(trace, placed=integrated))
    blocked, targets = [], {_WRAPPER: source("agent", "untrusted_context.py"), _DEDUP: source("context_dedup.py")}
    world.store = store
    posture = case["posture"]
    if posture == "blocked":
        blocked.append(_LIBRARIAN)
    elif posture == "real-off":
        targets[_LIBRARIAN] = source("memory", "librarian.py")
    else:
        seeded[_LIBRARIAN] = _Librarian(trace, enabled=posture == "three-names-on",
                                        block=ONION_BLOCK if posture == "counting-off" else "",
                                        counter=posture == "counting-off").module()
    if "caches" in groups or integrated:
        seeded.update(_caches(trace, hit_keys=("*",) if options.get("hit") else (),
                              on_get=_stop_at(world, stops.get("lookup", ()))))
    if "ledger" in groups:
        seeded.update(_ledger(trace))
    if "network" in groups:
        seeded.update(_network(trace))
    if options.get("offline"):
        seeded.update(_network(trace, online=False))
    if "governor" in options:
        seeded.update(_governor(trace, **options["governor"], on_admit=_stop_at(world, stops.get("admission", ())),
                                on_ticket=_stop_at(world, stops.get("thread", ()))))
    if options.get("slot_affinity"):
        seeded.update(_slots(trace))
    if "projects" in groups:
        project_seeded, world.projects = _projects(trace)
        seeded.update(project_seeded)
    if "web" in groups:
        seeded.update(_web(trace, gated=integrated))
    if integrated:
        seeded.update(_integrated_base(trace))
        if options.get("offline"):
            seeded.update(_network(trace, online=False))
        window, output = options.get("limits", (32768, 4096))
        cm_seeded, world.context_manager = _context_manager(trace, window=window, output=output)
        seeded.update(cm_seeded)
        total_window, reserve = options.get("window_budget", (8192, 1200))
        po_seeded, world.budget_manager = _prompt_optimization(
            trace, history_tokens=options.get("history_budget", 3200), total_window=total_window, reserve=reserve)
        seeded.update(po_seeded)
        co_seeded, world.compressor = _compressor(trace, enabled=bool(options.get("compressor")),
                                                  trigger=bool(options.get("trigger")),
                                                  archive=_ARCHIVE_RESULTS if options.get("trigger") else ())
        seeded.update(co_seeded)
        summarizer = options.get("summarizer")
        if summarizer is not None:
            seeded.update(_summarizer(trace, **summarizer))
        world.sliding = None
        if options.get("sliding"):
            sw_seeded, world.sliding = _sliding(trace, keep_last=options["sliding"])
            seeded.update(sw_seeded)
        targets[_TIERS] = source("context_summary_tiers.py")
        targets[_OPTIMIZER] = source("context_optimizer.py")
    targets[_EXECUTOR] = source("executor.py")
    had, previous = "ollama" in sys.modules, sys.modules.get("ollama")
    sys.modules["ollama"] = _module("ollama", chat=scripted.chat)
    try:
        loaded, close_window = isolate(targets=targets, blocked=blocked, seeded=seeded,
                                       packages=("opti_oignon.agent", "opti_oignon.memory"))
    except BaseException:
        _restore_ollama(had, previous)
        raise
    world.executor_module, world.wrapper = loaded[_EXECUTOR], loaded[_WRAPPER]
    world.librarian = loaded.get(_LIBRARIAN)
    stilled = [world.executor_module] + ([loaded[_TIERS], loaded[_OPTIMIZER]] if integrated else [])
    if any(_reads_datetime(module) for module in stilled):
        close_window()
        _restore_ollama(had, previous)
        raise RuntimeError("a module the still clock is set on reads datetime, which it does not still")
    clock = StillClock()
    world.executor_module.time = clock
    if integrated:
        loaded[_TIERS].time = clock
        optimizer_module = loaded[_OPTIMIZER]
        optimizer_module.time = clock
        budget_manager = world.budget_manager
        if options.get("budget_raises"):
            def refusing(**kwargs):
                trace.add("budget.raises", **kwargs)
                raise RuntimeError("budget engine unavailable")

            budget_manager = SimpleNamespace(calculate_budget=refusing)
        optimizer_module.init_optimizer(
            config={"enabled": bool(options.get("optimizer")), "compression": {"strategy": "auto"},
                    "stable_prefix": {"enabled": bool(options.get("stable_prefix"))},
                    "slot_affinity": {"enabled": bool(options.get("slot_affinity"))},
                    "emergency": {"enabled": True, "min_recent_messages": 2, "max_block_chars": 2000},
                    "report": {"max_retained": 10}},
            budget_manager=budget_manager,
            project_context_builder=getattr(world, "projects", None),
            conversation_compressor=world.compressor if options.get("compressor") else None,
            sliding_window_manager=world.sliding,
            context_manager=world.context_manager,
        )

    def close():
        close_window()
        _restore_ollama(had, previous)

    return world, close


def _reads_datetime(module):
    """Whether ``module`` holds ``datetime``, the module or one of its names, under any name."""
    return any(getattr(value, "__name__", None) == "datetime" or getattr(value, "__module__", None) == "datetime"
               for value in vars(module).values())


def _restore_ollama(had, previous):
    if had:
        sys.modules["ollama"] = previous
    else:
        sys.modules.pop("ollama", None)


def _drive(generator):
    yields = []
    try:
        while True:
            item = next(generator)
            if item != "":
                yields.append(list(item) if isinstance(item, tuple) else item)
    except StopIteration as stop:
        return yields, stop.value


def run_case(case, explain=False):
    """One case's record: its turns, each with its requests, yields, return value, run results and trace; its facts."""
    trace, requests = Trace(explain), [[]]
    world, close = _open(case, trace, requests)
    turns = []
    try:
        executor = world.executor_module.Executor()
        for index, (conversation_id, question, kwargs) in enumerate(case["turns"]):
            if index:
                trace.next_turn()
                requests.append([])
            run = SimpleNamespace(stop=threading.Event(), results={})
            world.run, world.turn = run, index
            kwargs = dict(kwargs)
            refine = kwargs.pop("refine", False)
            if case["env"] == "integrated":
                run.user_turn = world.executor_module.user_turn(question, question, kwargs.get("documents") or ())
            yields, returned = _drive(executor.execute(question, _routing(), refine=refine,
                                                       conversation_id=conversation_id, run=run, **kwargs))
            turns.append({
                "requests": requests[-1], "yields": shape(yields), "returned": shape(list(returned or ())),
                "results": shape({key: run.results[key] for key in sorted(run.results)
                                  if key in ("admission_refused", "cancelled", "context_label", "step_error")}),
                "window": _window(executor),
                "trace": trace.turns[-1],
            })
        facts = _facts(case, world, executor, turns)
    finally:
        close()
    return {"turns": turns, "facts": facts}


_WIRING_DIGEST = "d1780f443e749215b4957c5281c645c9025d050a6f18c21687bdcbc9e574742a"


def _window(executor):
    """The window figures a turn leaves on the executor: its strategy, what it dropped, the optimizer's zones."""
    stats = executor.last_window_stats or {}
    report = executor.last_optimization_report
    window = {
        "strategy": stats.get("strategy"), "kept": stats.get("kept"), "dropped": stats.get("dropped", 0),
        "fallback_legacy": bool(stats.get("fallback_legacy")), "history_count": stats.get("history_count"),
        "zones": {zone.zone: zone.strategy for zone in getattr(report, "zones", ())},
        "cache_hit": bool(executor.last_cache_hit), "offline_queued": bool(executor.last_offline_queued),
    }
    return {key: value for key, value in window.items() if value not in (None, False, 0, {})}


def _facts(case, world, executor, turns):
    """What shows each case's trigger fired, read off the run itself and its first request."""
    window = turns[0]["window"] if turns else {}
    first = world.scripted.calls[0]["messages"] if world.scripted.calls else []
    joined = "\n".join(str(m.get("content", "")) for m in first)
    facts = {
        "window_strategy": window.get("strategy"),
        "window_dropped": bool(window.get("dropped")),
        "fallback_legacy": bool(window.get("fallback_legacy")),
        "first_history_count": max(0, len(first) - 2),
        "second_turn_hit": bool(
            len(turns) > 1 and not turns[1]["requests"] and turns[1]["window"].get("cache_hit")),
        "documents_in_first_request": joined.count("Document provided:"),
        "summary_leads_the_first_request": len(first) > 1 and world.wrapper.is_summary_message(first[1]),
        "manifest_second": len(first) > 1 and first[1].get("role") == "system" and first[1].get(
            "content") == "Tools: none.",
        "history_zone": window.get("zones", {}).get("history"),
        "tiers_kept": "context_summary_tiers" in world.store.metadata.get("conv-1", {}),
        "archive_listed": "[user] user turn 0" in joined,
        "archive_duplicate_dropped": "[user] user turn 0" in joined and "[assistant] Remembered facts" not in joined,
    }
    seated = sys.modules.get(_LIBRARIAN)
    facts["librarian_names"] = 0 if seated is None else sum(
        hasattr(seated, name) for name in ("onion_enabled", "memory_block", "maybe_curate", "use_counter"))
    facts["librarian_config_off"] = _config_off(world.librarian) if case["posture"] == "real-off" else None
    facts["librarian_is_the_real_one"] = world.librarian is not None
    facts["capability_block_given"] = any("capability_block" in kwargs for _cid, _q, kwargs in case["turns"])
    facts["capability_block_placed"] = any("Tools: none." in str(message.get("content", ""))
                                           for call in world.scripted.calls for message in call.get("messages", []))
    if case["env"] == "isolated" and case["turns"] == _TWO and not case["groups"] and not case["options"]:
        calls = json.dumps(world.scripted.calls, sort_keys=True, ensure_ascii=False)
        facts["scripted_digest_is_the_wiring_contracts"] = sha(calls) == _WIRING_DIGEST
    return facts


def _config_off(librarian):
    """True when the real librarian's own configuration reads, and says off: an unreadable one is not "off"."""
    try:
        return librarian.load_config().enabled is False
    except Exception:  # noqa: BLE001 - a configuration that cannot be read is not one that says off
        return False


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
    "import _onion_off_golden as golden\n"
    "sys.stdout.write(golden.capture_all_json())\n"
)


def capture_in_children(seeds=SEEDS, timeout=180):
    """The capture as each seed's child prints it, by seed. The environment is extended, never replaced."""
    out = {}
    with tempfile.TemporaryDirectory(prefix="onion-off-golden-") as empty, \
            tempfile.TemporaryDirectory(prefix="onion-off-golden-bytecode-") as bytecode:
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


def render(value, depth=0, open_to=6):
    """JSON in ASCII, keys sorted, each container opened onto its own lines down to ``open_to`` levels and
    compact below, so a difference reads line by line: one line per request, per event, per figure."""
    pad = " " * depth
    if depth >= open_to or not isinstance(value, (dict, list)) or not value:
        return canonical(value)
    inner = " " * (depth + 1)
    if isinstance(value, dict):
        items = [f"{inner}{canonical(key)}: {render(value[key], depth + 1, open_to)}" for key in sorted(value)]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    items = [inner + render(item, depth + 1, open_to) for item in value]
    return "[\n" + ",\n".join(items) + "\n" + pad + "]"


def fixture_text(captured):
    """The golden file's text for a capture as a child printed it: its golden part, rendered, one trailing newline."""
    return render(json.loads(captured)["golden"]) + "\n"


# ---------------------------------------------------------------------------
# The command line: --check and --write, never anything else
# ---------------------------------------------------------------------------
# What the capture runs through: the package, and the tests/ files it loads.
_SOURCES = ("opti_oignon", "tests/_isolation.py", "tests/_registry_bridge.py", "tests/_onion_off_golden.py")


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
            same = FIXTURE.is_file() and FIXTURE.read_text(encoding="ascii") == text
            print("golden: same" if same else "golden: DIFFERS")
            return 0 if same else 1
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(text, encoding="ascii")
        print(f"golden written: {FIXTURE.relative_to(REPO)} sha256 {sha(text)}")
        return 0
    finally:
        firewall.uncover_children()
        firewall.uninstall()


if __name__ == "__main__":
    raise SystemExit(main())
