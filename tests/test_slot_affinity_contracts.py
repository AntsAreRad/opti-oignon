#!/usr/bin/env python3
"""What per-conversation slot affinity promises about the prompt-KV cache.

The external llama-server keeps one prompt-KV cache per slot and, told
nothing, picks the slot itself. Two conversations then share one routinely:
the second either evicts the first's cached prefix or is decoded on attention
state computed over the first's content. These contracts pin the decision that
replaces the server's, and they pin it at all three levels it passes through.

The decision. A conversation is keyed on its identity, the fingerprint of the
invariant head, and the trust-envelope state. The current turn is deliberately
NOT part of the key: a key that moved every turn would hand out a fresh slot
every turn and defeat the cache it exists to protect. Distinct keys never hold
the same slot at the same time, and a caller with no conversation identity is
given no slot at all rather than one some conversation would lose.

The degrading. The server does not serve its slot listing unless it was
started with the flag that enables it, so an empty listing is what an ordinary
host reports -- the degraded shape is the common one, not the edge case. No
listing, no identity, a listing that is not a list, unreadable entries, every
slot decoding: each answers no slot, and none of them raises.

The seam and the hub. The slot number rides the option whitelist verbatim on
both completion paths, and is absent from the wire when the caller did not ask
for it. The execution hub names a slot only when its own switch is on AND the
resolved backend is the llama-server one; every other combination leaves the
options exactly as they were, and never reads the listing at all.

Loaded through the shared isolation window with scripted transports and a
scripted backend; no server, no model, no network is ever reached.
"""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_AFFINITY = "opti_oignon.slot_affinity"
_BACKEND = "opti_oignon.inference_backend"
_OPTIMIZER = "opti_oignon.context_optimizer"
_EXECUTOR = "opti_oignon.executor"


def _listing(count, busy=(), start=0):
    """A slot listing shaped like the server's, with chosen slots decoding."""
    return [
        {"id": i, "id_task": 7 if i in busy else -1}
        for i in range(start, start + count)
    ]


def _load_affinity():
    loaded, restore = isolate(
        targets={_AFFINITY: source("slot_affinity.py")},
        seeded={},
        packages=("opti_oignon",),
    )
    return loaded[_AFFINITY], restore


# ---------------------------------------------------------------------------
# sa1 -- the reason the module exists: two conversations never share a slot
# ---------------------------------------------------------------------------

def test_sa1_two_conversations_never_hold_the_same_slot():
    mod, restore = _load_affinity()
    try:
        router = mod.SlotAffinity()
        seen = []
        for index in range(4):
            seen.append(
                router.choose(
                    conversation_id=f"conv-{index}",
                    prefix_fingerprint="head",
                    slots=_listing(4),
                )
            )
        assert None not in seen, "four idle slots must serve four conversations"
        assert len(set(seen)) == len(seen), "a slot was handed to two conversations"
        held = router.assignments
        assert len(set(held.values())) == len(held)
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa2 -- a turn with no conversation identity never claims a slot
# ---------------------------------------------------------------------------

def test_sa2_a_call_without_identity_is_given_no_slot():
    mod, restore = _load_affinity()
    try:
        router = mod.SlotAffinity()
        owner = router.choose(
            conversation_id="conv-a", prefix_fingerprint="head", slots=_listing(4)
        )
        assert owner is not None
        for anonymous in (None, "", 0):
            assert (
                router.choose(
                    conversation_id=anonymous,
                    prefix_fingerprint="head",
                    slots=_listing(4),
                )
                is None
            )
        assert list(router.assignments.values()) == [owner], (
            "an anonymous call took a slot from the conversation that held it"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa3 -- the current turn is not part of the key
# ---------------------------------------------------------------------------

def test_sa3_the_turn_does_not_enter_the_key():
    mod, restore = _load_affinity()
    try:
        first = mod.routing_key(conversation_id="conv-a", prefix_fingerprint="head")
        again = mod.routing_key(conversation_id="conv-a", prefix_fingerprint="head")
        assert first == again
        router = mod.SlotAffinity()
        turns = [
            router.choose(
                conversation_id="conv-a", prefix_fingerprint="head", slots=_listing(4)
            )
            for _ in range(5)
        ]
        assert len(set(turns)) == 1, "the slot moved between turns of one conversation"
        assert turns[0] is not None
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa4 -- a different trust envelope is a different key
# ---------------------------------------------------------------------------

def test_sa4_the_envelope_state_separates_keys():
    mod, restore = _load_affinity()
    try:
        bare = mod.routing_key(conversation_id="conv-a", prefix_fingerprint="head")
        with_web = mod.routing_key(
            conversation_id="conv-a", prefix_fingerprint="head", envelope="web"
        )
        assert bare != with_web
        assert mod.envelope_state(["web", "memory", "web"]) == "memory+web"
        assert mod.envelope_state(None) == mod.ENVELOPE_NONE
        assert mod.envelope_state("web") == mod.ENVELOPE_NONE
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa5 -- one conversation, one slot, even when its key moves
# ---------------------------------------------------------------------------

def test_sa5_a_conversation_never_strands_a_second_slot():
    mod, restore = _load_affinity()
    try:
        router = mod.SlotAffinity()
        router.choose(
            conversation_id="conv-a", prefix_fingerprint="head", slots=_listing(4)
        )
        router.choose(
            conversation_id="conv-a", prefix_fingerprint="moved", slots=_listing(4)
        )
        router.choose(
            conversation_id="conv-a",
            prefix_fingerprint="moved",
            envelope="web",
            slots=_listing(4),
        )
        assert len(router.assignments) == 1, "the conversation is holding two slots"
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa6 -- an empty listing names no slot (the ordinary host shape)
# ---------------------------------------------------------------------------

def test_sa6_an_empty_listing_names_no_slot():
    mod, restore = _load_affinity()
    try:
        router = mod.SlotAffinity()
        assert (
            router.choose(
                conversation_id="conv-a", prefix_fingerprint="head", slots=[]
            )
            is None
        )
        assert router.assignments == {}
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa7 -- a listing that is not a list names no slot, and does not raise
# ---------------------------------------------------------------------------

def test_sa7_a_listing_that_is_not_a_list_names_no_slot():
    mod, restore = _load_affinity()
    try:
        router = mod.SlotAffinity()
        for shape in (None, {"slots": 4}, "four", 4, object()):
            assert (
                router.choose(
                    conversation_id="conv-a",
                    prefix_fingerprint="head",
                    slots=shape,
                )
                is None
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa8 -- unreadable entries are skipped, never guessed at
# ---------------------------------------------------------------------------

def test_sa8_unreadable_entries_yield_no_slot():
    mod, restore = _load_affinity()
    try:
        router = mod.SlotAffinity()
        junk = [None, "x", 5, {"id": "two"}, {"id": True}, {"id": -1}, {}]
        assert (
            router.choose(
                conversation_id="conv-a", prefix_fingerprint="head", slots=junk
            )
            is None
        )
        mixed = junk + [{"id": 3, "id_task": -1}]
        assert (
            router.choose(
                conversation_id="conv-a", prefix_fingerprint="head", slots=mixed
            )
            == 3
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa9 -- a decoding slot is never taken for a fresh assignment
# ---------------------------------------------------------------------------

def test_sa9_a_busy_slot_is_never_freshly_assigned():
    mod, restore = _load_affinity()
    try:
        router = mod.SlotAffinity()
        assert (
            router.choose(
                conversation_id="conv-a",
                prefix_fingerprint="head",
                slots=_listing(2, busy=(0, 1)),
            )
            is None
        )
        assert router.assignments == {}
        assert (
            router.choose(
                conversation_id="conv-a",
                prefix_fingerprint="head",
                slots=_listing(2, busy=(0,)),
            )
            == 1
        )
        processing = [{"id": 0, "is_processing": True}, {"id": 1, "is_processing": True}]
        assert (
            router.choose(
                conversation_id="conv-b",
                prefix_fingerprint="head",
                slots=processing,
            )
            is None
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa10 -- eviction is explicit and least recently used
# ---------------------------------------------------------------------------

def test_sa10_eviction_is_least_recently_used():
    mod, restore = _load_affinity()
    try:
        router = mod.SlotAffinity()
        first = router.choose(
            conversation_id="conv-a", prefix_fingerprint="head", slots=_listing(2)
        )
        second = router.choose(
            conversation_id="conv-b", prefix_fingerprint="head", slots=_listing(2)
        )
        # Touching conv-a makes conv-b the least recently used.
        router.choose(
            conversation_id="conv-a", prefix_fingerprint="head", slots=_listing(2)
        )
        taken = router.choose(
            conversation_id="conv-c", prefix_fingerprint="head", slots=_listing(2)
        )
        assert taken == second, "eviction did not take the least recently used slot"
        assert (
            router.choose(
                conversation_id="conv-a",
                prefix_fingerprint="head",
                slots=_listing(2),
            )
            == first
        ), "the recently used conversation lost its slot"
        assert len(router.assignments) == 2
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa11 -- eviction never leaves two keys on one slot
# ---------------------------------------------------------------------------

def test_sa11_eviction_never_produces_a_shared_slot():
    mod, restore = _load_affinity()
    try:
        router = mod.SlotAffinity()
        for index in range(12):
            router.choose(
                conversation_id=f"conv-{index}",
                prefix_fingerprint="head",
                slots=_listing(3),
            )
            held = router.assignments
            assert len(set(held.values())) == len(held), (
                "eviction handed one slot to two conversations"
            )
        assert len(router.assignments) == 3
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa12 -- a slot that leaves the listing takes its assignment with it
# ---------------------------------------------------------------------------

def test_sa12_a_vanished_slot_drops_its_assignment():
    mod, restore = _load_affinity()
    try:
        router = mod.SlotAffinity()
        held = router.choose(
            conversation_id="conv-a", prefix_fingerprint="head", slots=_listing(4)
        )
        assert held == 0
        # The server comes back with a different, smaller set of slot numbers.
        moved = router.choose(
            conversation_id="conv-a",
            prefix_fingerprint="head",
            slots=_listing(2, start=7),
        )
        assert moved == 7
        assert list(router.assignments.values()) == [7]
    finally:
        restore()


# ---------------------------------------------------------------------------
# Scripted transport for the seam contracts
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, body, lines=None):
        self._body = body
        self._lines = list(lines or [])

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Transport:
    def __init__(self):
        self.requests = []

    def urlopen(self, req, timeout=None):
        payload = json.loads(req.data.decode("utf-8")) if req.data else None
        self.requests.append({"url": req.full_url, "payload": payload})
        return _FakeResponse(
            json.dumps({"choices": [{"message": {"content": "hi"}}], "model": "m"}
                       ).encode("utf-8"),
            lines=[
                b'data: {"choices": [{"delta": {"content": "hi"}}]}\n',
                b"data: [DONE]\n",
            ],
        )


def _load_backend():
    loaded, win_restore = isolate(
        targets={_BACKEND: source("inference_backend.py")},
        seeded={},
        packages=("opti_oignon",),
    )
    mod = loaded[_BACKEND]
    transport = _Transport()
    real = mod.urllib.request.urlopen
    mod.urllib.request.urlopen = transport.urlopen
    backend = mod.LlamaServerBackend(host="http://fake:8080")

    def restore():
        mod.urllib.request.urlopen = real
        win_restore()

    return backend, transport, restore


# ---------------------------------------------------------------------------
# sa13 -- the slot number rides the whitelist on the non-streaming path
# ---------------------------------------------------------------------------

def test_sa13_generate_forwards_the_slot_number():
    backend, transport, restore = _load_backend()
    try:
        backend.generate(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            options={"id_slot": 3, "cache_prompt": True},
        )
        payload = transport.requests[-1]["payload"]
        assert payload.get("id_slot") == 3
        assert payload.get("cache_prompt") is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa14 -- and on the streaming path
# ---------------------------------------------------------------------------

def test_sa14_stream_forwards_the_slot_number():
    backend, transport, restore = _load_backend()
    try:
        list(
            backend.stream(
                model="m",
                messages=[{"role": "user", "content": "q"}],
                options={"id_slot": 2},
            )
        )
        assert transport.requests[-1]["payload"].get("id_slot") == 2
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa15 -- control: not asked for, not on the wire, on either path
# ---------------------------------------------------------------------------

def test_sa15_the_slot_number_is_absent_when_not_asked_for():
    backend, transport, restore = _load_backend()
    try:
        backend.generate(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            options={"temperature": 0.3},
        )
        assert "id_slot" not in transport.requests[-1]["payload"]
        list(
            backend.stream(
                model="m",
                messages=[{"role": "user", "content": "q"}],
                options={"temperature": 0.3},
            )
        )
        assert "id_slot" not in transport.requests[-1]["payload"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# Scripted backend and window for the hub contracts
# ---------------------------------------------------------------------------

class _RecorderBackend:
    def __init__(self, name, listing):
        self.name = name
        self.calls = []
        self.listing = list(listing)
        self.slot_reads = 0

    def slots(self):
        self.slot_reads += 1
        return list(self.listing)

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        yield SimpleNamespace(content="ok", thinking=None)


def _load_hub(*, flag_on, backend_name, listing):
    recorder = _RecorderBackend(backend_name, listing)
    registry = SimpleNamespace(
        active=recorder, resolve_backend=lambda model: recorder
    )

    ollama_stub = types.ModuleType("ollama")
    ollama_stub.chat = lambda **k: iter([{"message": {"content": "ok"}}])

    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(
        get_model=lambda *a, **k: "test-model:1b",
        get_temperature=lambda *a, **k: 0.2,
    )
    router = types.ModuleType("opti_oignon.router")
    router.RoutingResult = type("RoutingResult", (), {})
    backends = types.ModuleType("opti_oignon.inference_backend")
    backends.get_backend_registry = lambda: registry

    seeded = {
        "opti_oignon.config": cfg,
        "opti_oignon.router": router,
        "opti_oignon.inference_backend": backends,
    }
    # The affinity module has to be in the window BEFORE the executor is
    # executed, or the executor's guarded import answers unavailable and the
    # contract would pass on an absence the test created itself.
    targets = {
        _AFFINITY: source("slot_affinity.py"),
        _OPTIMIZER: source("context_optimizer.py"),
        # The deduplicator rides along as a real target: the hub imports
        # it plainly, and it is pure and standard-library only.
        "opti_oignon.context_dedup": source("context_dedup.py"),
        _EXECUTOR: source("executor.py"),
    }
    had = "ollama" in sys.modules
    previous = sys.modules.get("ollama")
    sys.modules["ollama"] = ollama_stub
    loaded, win_restore = isolate(
        targets=targets, seeded=seeded, packages=("opti_oignon",)
    )
    loaded[_OPTIMIZER].init_optimizer(
        config={
            "enabled": False,
            "stable_prefix": {"enabled": False},
            "slot_affinity": {"enabled": bool(flag_on)},
        }
    )
    assert loaded[_EXECUTOR].SLOT_AFFINITY_AVAILABLE, (
        "the window must carry the affinity module, not its absence"
    )

    def restore():
        win_restore()
        if had:
            sys.modules["ollama"] = previous
        else:
            sys.modules.pop("ollama", None)

    return loaded[_EXECUTOR], recorder, restore


def _drive(mod, recorder, conversation_id):
    executor = mod.Executor()
    routing = SimpleNamespace(
        model="test-model:1b",
        task_type="general",
        temperature=0.2,
        prompt_variant="standard",
        timeout=30,
    )
    generator = executor.execute(
        "q", routing, refine=False,
        conversation_id=conversation_id, persist=False,
    )
    try:
        while True:
            next(generator)
    except StopIteration:
        pass
    assert recorder.calls, "the request must reach the scripted backend"
    return recorder.calls[-1]["options"]


# ---------------------------------------------------------------------------
# sa16 -- flag on and llama-server resolved: the hub names the slot
# ---------------------------------------------------------------------------

def test_sa16_the_hub_names_a_slot_under_its_own_flag():
    mod, recorder, restore = _load_hub(
        flag_on=True, backend_name="llama_server", listing=_listing(4)
    )
    try:
        options = _drive(mod, recorder, "conv-a")
        assert options.get("id_slot") == 0
        assert recorder.slot_reads >= 1, "the listing must be read, not assumed"
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa17 -- two conversations through one hub land on different slots
# ---------------------------------------------------------------------------

def test_sa17_the_hub_keeps_two_conversations_apart():
    mod, recorder, restore = _load_hub(
        flag_on=True, backend_name="llama_server", listing=_listing(4)
    )
    try:
        first = _drive(mod, recorder, "conv-a").get("id_slot")
        second = _drive(mod, recorder, "conv-b").get("id_slot")
        again = _drive(mod, recorder, "conv-a").get("id_slot")
        assert first is not None and second is not None
        assert first != second, "two conversations shared one prompt cache"
        assert again == first, "a conversation lost its slot between turns"
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa18 -- no conversation identity: the hub names no slot
# ---------------------------------------------------------------------------

def test_sa18_the_hub_names_no_slot_without_a_conversation():
    mod, recorder, restore = _load_hub(
        flag_on=True, backend_name="llama_server", listing=_listing(4)
    )
    try:
        options = _drive(mod, recorder, None)
        assert "id_slot" not in options
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa19 -- control: the flag off leaves the options alone and reads nothing
# ---------------------------------------------------------------------------

def test_sa19_the_flag_off_names_no_slot_and_reads_no_listing():
    mod, recorder, restore = _load_hub(
        flag_on=False, backend_name="llama_server", listing=_listing(4)
    )
    try:
        options = _drive(mod, recorder, "conv-a")
        assert "id_slot" not in options
        assert recorder.slot_reads == 0, "the listing was read with the flag off"
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa20 -- control: another backend never sees a slot number
# ---------------------------------------------------------------------------

def test_sa20_another_backend_never_sees_a_slot_number():
    mod, recorder, restore = _load_hub(
        flag_on=True, backend_name="ollama", listing=_listing(4)
    )
    try:
        options = _drive(mod, recorder, "conv-a")
        assert "id_slot" not in options
        assert recorder.slot_reads == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# sa21 -- an empty listing at the hub names no slot and does not raise
# ---------------------------------------------------------------------------

def test_sa21_an_empty_listing_at_the_hub_names_no_slot():
    mod, recorder, restore = _load_hub(
        flag_on=True, backend_name="llama_server", listing=[]
    )
    try:
        options = _drive(mod, recorder, "conv-a")
        assert "id_slot" not in options
        assert recorder.slot_reads >= 1
    finally:
        restore()
