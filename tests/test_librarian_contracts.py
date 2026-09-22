#!/usr/bin/env python3
"""Contracts for the librarian: the loop that grows the onion behind the chat path.

The librarian mirrors a conversation into a Flesh, evicts through the
probe gate with a summariser that asks the inference registry (never the
client behind it, and with the keep-alive of ``onion.yaml`` so the model
is released after each burst), and composes the memory block the executor
places in the prompt: Core, receipts digest, selected Peels, every recalled
segment framed as data. It is gated by the YAML, throttled by a watermark,
dispatched through an injectable runner, and never raises into a turn.

  * LB1 -- the configuration is the YAML's, the onion is off by default,
    and an out-of-range configuration is refused.
  * LB2 -- the summariser asks the registry: model and keep-alive from the
    configuration, the turns quoted with their ids; no backend, no
    summariser -- never the client.
  * LB3 -- the dispatch is gated and throttled: disabled fires nothing,
    growth below the threshold fires nothing, growth at the threshold
    fires once, a failing runner is swallowed.
  * LB4 -- a curation step evicts through the gate while the Flesh exceeds
    its cap and stops when it fits; a refused summary leaves the Flesh.
  * LB5 -- the memory block carries Core, receipts and the Peels for the
    question, framed as data with provenance; an unknown conversation
    yields nothing; a tampered Core yields nothing, and no exception.
  * LB6 -- the memory block is within the sum of its layer caps.
  * LB7 -- the mirror is exact and idempotent: sequential turn ids, roles
    and text untouched, no duplicate on a second mirror.

With a persistence path in ``onion.yaml`` the state goes through the onion
store and comes back across a restart:

  * LB8 -- the mirror and every accepted eviction are saved; after the
    process forgets everything, the next sight of the conversation loads
    the same state, the cursor included so nothing is mirrored twice, and
    the memory block is the one from before.
  * LB9 -- a store that refuses (the connection seam unreachable here)
    leaves the state absent: no dispatch, no block, no file, and the
    refusal is logged by name once; the same configuration without a path
    keeps the onion in the process as before.
  * LB10 -- the persistence section is read from the YAML with encryption
    required by default, a relative path resolves under the data directory,
    an absolute one stands, and an absent section means no store.

The user's surface on the Core and the Cellar, the only path to either:

  * LB11 -- a pin lands in the conversation's Core with the user as actor,
    is content-addressed and idempotent, is read back by ``core_entries``,
    and composes into the memory block; any other actor is refused by name
    before the store is touched.
  * LB12 -- a pin that would push the Core over its cap is refused by name
    at pin time, before it lands, and the block it would have blanked is
    still composed from what was there.
  * LB13 -- a supersession links the old entry to the new one, the old text
    stays, the block carries the successor only; superseding an unknown or
    an already superseded entry is refused by name.
  * LB14 -- a recall hands back the verbatim span behind an open receipt,
    marks the receipt resolved so the digest no longer shows it, refuses
    an unknown key by name, and every mutation of LB11-14 is saved through
    the store when one is configured.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source; the registry is blocked, so a
summariser can only come from an injected resolver.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_ONION_YAML = REPO / "opti_oignon" / "config" / "onion.yaml"
_MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian")
_PERSISTED = _MODULES + ("onion_store",)


def _open(*, persisted=False, seeded=None):
    loaded, restore = isolate(
        targets={f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in (_PERSISTED if persisted else _MODULES)},
        blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils"),
        seeded=seeded,
        packages=("opti_oignon.memory",),
    )
    lib = loaded["opti_oignon.memory.librarian"]
    lib.reset_librarian()
    return lib, loaded, restore


def _messages(n):
    out = []
    for i in range(1, n + 1):
        role = "user" if i % 2 else "assistant"
        out.append({"role": role, "content": f"Turn {i}: Alice reviewed service {i} on 2026-03-{i:02d} and we agreed that service {i} stays on the new cluster."})
    return out


def _faithful(turns):
    return " ".join(t["text"] for t in turns)


def _config(lib, **over):
    fields = dict(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=4, temperature=0.1, num_predict=128)
    fields.update(over)
    return lib.LibrarianConfig(**fields)


def _small_budget(loaded):
    composer = loaded["opti_oignon.memory.composer"]
    return composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=140, turn=60)


class _Backend:
    def __init__(self):
        self.calls = []

    def generate(self, model, messages, options=None, keep_alive="30m", think=False, images=None):
        self.calls.append(dict(model=model, messages=messages, options=options, keep_alive=keep_alive))
        user = [m for m in messages if m["role"] == "user"][0]["content"]
        return SimpleNamespace(content=user)


# ---------------------------------------------------------------------------
# LB1 -- configuration
# ---------------------------------------------------------------------------
def test_lb1_the_configuration_is_the_yamls_and_the_onion_is_off_by_default():
    import yaml

    lib, loaded, restore = _open()
    try:
        raw = yaml.safe_load(_ONION_YAML.read_text(encoding="utf-8"))
        cfg = lib.load_config()
        assert raw["enabled"] is False and cfg.enabled is False, "the maintainer turns the onion on"
        assert lib.onion_enabled() is False
        assert cfg.model == str(raw["librarian"]["model"])
        assert cfg.keep_alive == str(raw["librarian"]["keep_alive"]) == "0", "unloadable by default"
        assert cfg.min_new_turns == int(raw["librarian"]["min_new_turns"]) >= 1
        assert cfg.validate() == []
        assert _config(lib, min_new_turns=0).validate() != []
        assert _config(lib, num_predict=0).validate() != []
        assert lib.onion_enabled(path=REPO / "does-not-exist.yaml") is False, "an unreadable file is off, not on"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LB2 -- the summariser asks the registry
# ---------------------------------------------------------------------------
def test_lb2_the_summariser_asks_the_registry_with_the_configured_keep_alive():
    lib, loaded, restore = _open()
    try:
        backend = _Backend()
        cfg = _config(lib, model="librarian:3b", keep_alive="0", num_predict=77)
        summarize = lib.registry_summarizer(cfg, resolve=lambda model: backend)
        assert summarize is not None
        turns = [{"turn_id": "t0001", "role": "user", "text": "Alice lives in Berlin."},
                 {"turn_id": "t0002", "role": "assistant", "text": "Noted."}]
        out = summarize(turns)
        assert len(backend.calls) == 1
        call = backend.calls[0]
        assert call["model"] == "librarian:3b"
        assert call["keep_alive"] == "0"
        assert call["options"]["num_predict"] == 77
        user = [m for m in call["messages"] if m["role"] == "user"][0]["content"]
        assert "t0001" in user and "Alice lives in Berlin." in user, "turns are quoted with their ids"
        assert isinstance(out, str) and "Alice lives in Berlin." in out
        assert lib.registry_summarizer(cfg, resolve=lambda model: None) is None, "no backend, no summariser"
        assert lib.registry_summarizer(cfg) is None, "the registry is blocked in this window: nothing else is tried"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LB3 -- gated and throttled dispatch
# ---------------------------------------------------------------------------
def test_lb3_the_dispatch_is_gated_throttled_and_never_raises():
    lib, loaded, restore = _open()
    try:
        fired = []
        runner = lambda cid: fired.append(cid)  # noqa: E731
        assert lib.maybe_curate("c1", _messages(8), config=_config(lib, enabled=False), runner=runner) is False
        assert fired == [] and lib.peek_state("c1") is None
        cfg = _config(lib, min_new_turns=4)
        assert lib.maybe_curate("c1", _messages(3), config=cfg, runner=runner) is False
        assert fired == []
        assert lib.maybe_curate("c1", _messages(4), config=cfg, runner=runner) is True
        assert fired == ["c1"]
        assert lib.maybe_curate("c1", _messages(4), config=cfg, runner=runner) is False, "no growth, no fire"
        assert lib.maybe_curate("c1", _messages(8), config=cfg, runner=runner) is True
        assert fired == ["c1", "c1"]
        assert len(lib.peek_state("c1").flesh.turns()) == 8, "the mirror followed the growth"

        def boom(cid):
            raise RuntimeError("runner down")
        assert lib.maybe_curate("c2", _messages(4), config=cfg, runner=boom) is True
        assert lib.maybe_curate(None, _messages(4), config=cfg, runner=runner) is False
        assert lib.maybe_curate("c3", [], config=cfg, runner=runner) is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# LB4 -- a curation step
# ---------------------------------------------------------------------------
def test_lb4_curation_evicts_through_the_gate_until_the_flesh_fits():
    lib, loaded, restore = _open()
    try:
        peels = loaded["opti_oignon.memory.peels"]
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        budget = _small_budget(loaded)
        state = lib.state_for("c1")
        state.mirror(_messages(12))
        assert state.flesh.tokens(lib.estimate_tokens) > budget.flesh, "control: the Flesh overflows"
        steps = []
        while True:
            outcome = lib.curate(state, _faithful, gate=gate, budget=budget)
            steps.append(outcome)
            if not outcome.evicted:
                break
        assert len(steps) >= 3
        assert all(o.evicted for o in steps[:-1]) and "fits" in steps[-1].reason
        assert state.flesh.tokens(lib.estimate_tokens) <= budget.flesh
        assert len(state.ledger.open()) == len(steps) - 1
        assert len(state.tree.all()) == len(steps) - 1
        named = sorted(tid for r in state.ledger.open() for tid in r.turn_ids)
        assert named == [f"t{i:04d}" for i in range(1, 2 * (len(steps) - 1) + 1)]

        refused = lib.state_for("c2")
        refused.mirror(_messages(12))
        before = refused.flesh.turns()
        outcome = lib.curate(refused, lambda turns: "nothing of note", gate=gate, budget=budget)
        assert outcome.evicted is False and refused.flesh.turns() == before
        assert refused.ledger.all() == [] and refused.tree.all() == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# LB5 -- the memory block
# ---------------------------------------------------------------------------
def test_lb5_the_memory_block_carries_core_receipts_and_peels_as_data():
    lib, loaded, restore = _open()
    try:
        peels = loaded["opti_oignon.memory.peels"]
        core_store = loaded["opti_oignon.memory.core_store"]
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        budget = _small_budget(loaded)
        assert lib.memory_block("nobody", "anything") == ""
        state = lib.state_for("c1")
        state.core.add("The user is called Alice.", actor=core_store.USER)
        state.mirror(_messages(12))
        while lib.curate(state, _faithful, gate=gate, budget=budget).evicted:
            pass
        block = lib.memory_block("c1", "what happened with service 3", budget=budget)
        assert "The user is called Alice." in block
        for receipt in state.ledger.open():
            assert receipt.key[:12] in block, "every open receipt is in the digest"
        assert "service 3" in block
        assert "layer=peels" in block and "layer=receipts" in block and "provenance=" in block
        assert "layer=core" not in block, "the Core bears instruction; it is not framed as data"
        entry = state.core.active()[0]
        object.__setattr__(entry, "text", "The user is called Mallory.")
        assert lib.memory_block("c1", "what happened with service 3", budget=budget) == "", (
            "a tampered Core yields no block at all: fail closed, no exception"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# LB6 -- within the layer caps
# ---------------------------------------------------------------------------
def test_lb6_the_memory_block_is_within_the_sum_of_its_layer_caps():
    lib, loaded, restore = _open()
    try:
        peels = loaded["opti_oignon.memory.peels"]
        core_store = loaded["opti_oignon.memory.core_store"]
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        budget = _small_budget(loaded)
        state = lib.state_for("c1")
        state.core.add("Answers are concise.", actor=core_store.USER)
        state.mirror(_messages(24))
        while lib.curate(state, _faithful, gate=gate, budget=budget).evicted:
            pass
        assert len(state.tree.all()) >= 5, "control: enough peels to overflow the peels cap"
        block = lib.memory_block("c1", "service reviewed by alice on the new cluster", budget=budget)
        assert block
        assert lib.estimate_tokens(block) <= budget.core + budget.receipts + budget.peels + 40, (
            "the payload fits the memory layers; the framing lines are the only overhead"
        )
        payload = [line for line in block.splitlines() if not line.startswith("[")]
        assert lib.estimate_tokens("\n".join(payload)) <= budget.core + budget.receipts + budget.peels
        peel_payloads, current = [], None
        for line in block.splitlines():
            if line.startswith("[data layer=peels"):
                current = []
            elif line == "[/data]" and current is not None:
                peel_payloads.append("\n".join(current))
                current = None
            elif current is not None:
                current.append(line)
        assert len(peel_payloads) >= 1, "control: peels are in the block"
        assert sum(lib.estimate_tokens(t) for t in peel_payloads) <= budget.peels, (
            "the peels layer is within its own cap, not only within the sum"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# LB7 -- the mirror
# ---------------------------------------------------------------------------
def test_lb7_the_mirror_is_exact_and_idempotent():
    lib, loaded, restore = _open()
    try:
        state = lib.state_for("c1")
        msgs = _messages(5)
        assert state.mirror(msgs) == 5
        turns = state.flesh.turns()
        assert [t["turn_id"] for t in turns] == [f"t{i:04d}" for i in range(1, 6)]
        assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant", "user"]
        assert [t["text"] for t in turns] == [m["content"] for m in msgs]
        assert state.mirror(msgs) == 0 and len(state.flesh.turns()) == 5
        assert state.mirror(msgs + _messages(7)[5:]) == 2 and len(state.flesh.turns()) == 7
        assert state.mirror([{"role": "user"}, {"role": "user", "content": ""}] + msgs) == 0, (
            "a shorter or malformed history never rewinds the mirror"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# LB8 -- the state survives a restart through the store
# ---------------------------------------------------------------------------
def test_lb8_the_state_is_saved_and_comes_back_across_a_restart_with_its_cursor(tmp_path):
    import sqlite3

    lib, loaded, restore = _open(persisted=True)
    try:
        peels = loaded["opti_oignon.memory.peels"]
        core_store = loaded["opti_oignon.memory.core_store"]
        composer = loaded["opti_oignon.memory.composer"]
        budget = composer.Budget(window=2000, reserve=200, core=300, receipts=300, peels=800, flesh=200, turn=200)
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        path = tmp_path / "onion.db"
        cfg = _config(lib, persist_path=str(path), require_encryption=False, min_new_turns=4)
        store_mod = loaded["opti_oignon.memory.onion_store"]
        store_mod.OnionStore(path, connect=lambda p: sqlite3.connect(str(p)), require_encryption=False)
        lib._store[(str(path), False)] = store_mod.OnionStore(path, connect=lambda p: sqlite3.connect(str(p)), require_encryption=False)

        def synchronous(cid):
            state = lib.peek_state(cid, cfg)
            while lib.curate(state, _faithful, gate=gate, budget=budget).evicted:
                lib._save_state(cid, state, cfg)

        assert lib.maybe_curate("c1", _messages(12), config=cfg, runner=synchronous) is True
        state = lib.peek_state("c1", cfg)
        state.core.add("The user is Alice.", actor=core_store.USER)
        lib._save_state("c1", state, cfg)
        assert len(state.tree.all()) >= 2, "control: the burst made peels"
        before = lib.memory_block("c1", "service reviewed by alice", budget=budget, config=cfg)
        assert before, "control: a block before the restart"
        turns_before, seen_before = state.flesh.turns(), state.seen

        lib.reset_librarian()
        lib._store[(str(path), False)] = store_mod.OnionStore(path, connect=lambda p: sqlite3.connect(str(p)), require_encryption=False)
        assert lib._states == {}, "control: the process forgot everything"
        after = lib.memory_block("c1", "service reviewed by alice", budget=budget, config=cfg)
        assert after == before, "the block after the restart is the block from before"
        state = lib.peek_state("c1", cfg)
        assert state.flesh.turns() == turns_before and state.seen == seen_before
        assert state.mirror(_messages(12)) == 0, "the cursor came back: nothing is mirrored twice"
        assert [e.text for e in state.core.active()] == ["The user is Alice."]
        assert lib.maybe_curate("c1", _messages(12), config=cfg, runner=lambda cid: None) is False, (
            "no growth since the save, no dispatch"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# LB9 -- a refusing store leaves the state absent, by name
# ---------------------------------------------------------------------------
def test_lb9_a_refusing_store_leaves_the_state_absent_and_says_so_once(tmp_path, caplog):
    import logging

    lib, loaded, restore = _open(persisted=True)
    try:
        path = tmp_path / "refused.db"
        cfg = _config(lib, persist_path=str(path))
        fired = []
        with caplog.at_level(logging.WARNING, logger="opti_oignon.memory.librarian"):
            assert lib.maybe_curate("c1", _messages(8), config=cfg, runner=fired.append) is False
            assert lib.maybe_curate("c1", _messages(8), config=cfg, runner=fired.append) is False
            assert lib.memory_block("c1", "anything", config=cfg) == ""
        assert fired == [] and lib._states == {}, "no state was fabricated in place of the refused one"
        assert not path.exists(), "no file was created"
        refusals = [r for r in caplog.records if "refused, not replaced" in r.getMessage() and "c1" in r.getMessage()]
        assert len(refusals) == 1, "the refusal is logged by name, once per conversation"

        plain = _config(lib, min_new_turns=4)
        assert lib.maybe_curate("c2", _messages(4), config=plain, runner=fired.append) is True
        assert fired == ["c2"] and lib.peek_state("c2", plain) is not None, "without a path the onion lives in the process"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LB10 -- the persistence section
# ---------------------------------------------------------------------------
def test_lb10_the_persistence_section_is_read_with_encryption_required_by_default(tmp_path):
    import types

    import yaml

    data_dir = tmp_path / "data"
    config_mod = types.ModuleType("opti_oignon.config")
    config_mod.DATA_DIR = data_dir
    lib, loaded, restore = _open(seeded={"opti_oignon.config": config_mod})
    try:
        raw = yaml.safe_load(_ONION_YAML.read_text(encoding="utf-8"))
        shipped = lib.load_config()
        assert raw["persistence"] == {"path": "", "require_encryption": True}, "the shipped YAML: no path, encryption required"
        assert shipped.persist_path == "" and shipped.require_encryption is True
        assert lib.onion_store(shipped) is None, "no path, no store"

        relative = dict(raw, persistence={"path": "onion.db"})
        p = tmp_path / "relative.yaml"
        p.write_text(yaml.safe_dump(relative), encoding="utf-8")
        cfg = lib.load_config(p)
        assert cfg.require_encryption is True, "required unless the file says otherwise"
        assert lib._persistence_path(cfg) == data_dir / "onion.db", "a relative path resolves under the data directory"

        absolute = dict(raw, persistence={"path": str(tmp_path / "abs.db"), "require_encryption": False})
        p.write_text(yaml.safe_dump(absolute), encoding="utf-8")
        cfg = lib.load_config(p)
        assert lib._persistence_path(cfg) == tmp_path / "abs.db" and cfg.require_encryption is False

        absent = {k: v for k, v in raw.items() if k != "persistence"}
        p.write_text(yaml.safe_dump(absent), encoding="utf-8")
        assert lib._persistence_path(lib.load_config(p)) is None, "an absent section is no store"

        bad = dict(raw, persistence={"path": "x.db", "require_encryption": "yes"})
        p.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(lib.LibrarianError, match="require_encryption"):
            lib.load_config(p)
    finally:
        restore()


# ---------------------------------------------------------------------------
# LB11 -- a pin is the user's, content-addressed, and composes
# ---------------------------------------------------------------------------
def test_lb11_a_pin_is_the_users_alone_content_addressed_and_composes_into_the_block():
    lib, loaded, restore = _open()
    try:
        cfg = _config(lib)
        budget = _small_budget(loaded)
        entry_id = lib.pin("c1", "Answers cite their source.", actor="user", config=cfg)
        assert lib.pin("c1", "Answers cite their source.", actor="user", config=cfg) == entry_id, "the same bytes are one entry"
        entries = lib.core_entries("c1", config=cfg)
        assert [(e.id, e.text, e.status) for e in entries] == [(entry_id, "Answers cite their source.", "active")]
        block = lib.memory_block("c1", "anything", budget=budget, config=cfg)
        assert "Answers cite their source." in block
        for actor in ("model", "librarian", "", None):
            with pytest.raises(PermissionError, match="explicit user action"):
                lib.pin("c1", "The model pins.", actor=actor, config=cfg)
        assert [e.text for e in lib.core_entries("c1", config=cfg)] == ["Answers cite their source."], "nothing landed"
        with pytest.raises(ValueError, match="empty"):
            lib.pin("c1", "   ", actor="user", config=cfg)
        assert lib.core_entries("nobody", config=cfg) == [], "an unknown conversation has no entries, and none is created"
        assert lib.peek_state("nobody", cfg) is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# LB12 -- the cap is checked at pin time
# ---------------------------------------------------------------------------
def test_lb12_a_pin_over_the_core_cap_is_refused_before_it_lands_and_the_block_survives():
    lib, loaded, restore = _open()
    try:
        cfg = _config(lib)
        budget = _small_budget(loaded)
        lib.pin("c1", "Answers cite their source.", actor="user", config=cfg)
        before = lib.memory_block("c1", "anything", budget=budget, config=cfg)
        assert before, "control: a block before the oversized pin"
        oversized = " ".join(["word"] * (budget.core * 2))
        with pytest.raises(lib.LibrarianError, match="cap") as raised:
            lib.pin("c1", oversized, actor="user", config=cfg, budget=budget)
        assert str(budget.core) in str(raised.value), "the refusal names the cap"
        assert [e.text for e in lib.core_entries("c1", config=cfg)] == ["Answers cite their source."], "nothing landed"
        assert lib.memory_block("c1", "anything", budget=budget, config=cfg) == before, "the block is what it was, not blank"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LB13 -- supersession
# ---------------------------------------------------------------------------
def test_lb13_a_supersession_links_old_to_new_and_the_block_carries_the_successor_only():
    lib, loaded, restore = _open()
    try:
        cfg = _config(lib)
        budget = _small_budget(loaded)
        old_id = lib.pin("c1", "Answers are concise.", actor="user", config=cfg)
        new_id = lib.supersede("c1", old_id, "Answers are concise and cite their source.", actor="user", config=cfg)
        entries = {e.id: e for e in lib.core_entries("c1", config=cfg)}
        assert entries[old_id].superseded_by == new_id and entries[old_id].text == "Answers are concise."
        assert entries[new_id].status == "active"
        block = lib.memory_block("c1", "anything", budget=budget, config=cfg)
        assert "cite their source" in block and "Answers are concise.\n" not in block + "\n"
        with pytest.raises(ValueError, match="already superseded"):
            lib.supersede("c1", old_id, "A third text.", actor="user", config=cfg)
        with pytest.raises(KeyError):
            lib.supersede("c1", "0" * 64, "A third text.", actor="user", config=cfg)
        with pytest.raises(PermissionError, match="explicit user action"):
            lib.supersede("c1", new_id, "The model supersedes.", actor="model", config=cfg)
    finally:
        restore()


# ---------------------------------------------------------------------------
# LB14 -- recall, and every mutation saved
# ---------------------------------------------------------------------------
def test_lb14_a_recall_hands_back_the_span_marks_the_receipt_and_every_mutation_is_saved(tmp_path):
    import sqlite3

    lib, loaded, restore = _open(persisted=True)
    try:
        peels = loaded["opti_oignon.memory.peels"]
        composer = loaded["opti_oignon.memory.composer"]
        store_mod = loaded["opti_oignon.memory.onion_store"]
        budget = composer.Budget(window=2000, reserve=200, core=300, receipts=300, peels=800, flesh=200, turn=200)
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        path = tmp_path / "onion.db"
        cfg = _config(lib, persist_path=str(path), require_encryption=False)
        opener = lambda p: sqlite3.connect(str(p))  # noqa: E731
        lib._store[(str(path), False)] = store_mod.OnionStore(path, connect=opener, require_encryption=False)

        state = lib.state_for("c1", cfg)
        state.mirror(_messages(12))
        while lib.curate(state, _faithful, gate=gate, budget=budget).evicted:
            pass
        receipts = lib.open_receipts("c1", config=cfg)
        assert len(receipts) >= 2, "control: open receipts to recall"
        key = receipts[0].key
        span = lib.recall("c1", key, config=cfg)
        assert [t["turn_id"] for t in span] == list(receipts[0].turn_ids)
        assert span[0]["text"].startswith("Turn 1:"), "the verbatim span, not a summary"
        assert key not in [r.key for r in lib.open_receipts("c1", config=cfg)], "the receipt is resolved"
        assert receipts[0].stub not in lib.memory_block("c1", "service", budget=budget, config=cfg), "the digest no longer shows it"
        with pytest.raises(KeyError, match="not in the ledger"):
            lib.recall("c1", "0" * 64, config=cfg)
        just_recalled = store_mod.OnionStore(path, connect=opener, require_encryption=False).load("c1", lib.OnionState())
        assert [r.resolved for r in just_recalled.ledger.all() if r.key == key] == [True], (
            "the recall itself was saved, before any later mutation could save it"
        )

        entry_id = lib.pin("c1", "The user is Alice.", actor="user", config=cfg)
        lib.supersede("c1", entry_id, "The user is Alice, in Lyon.", actor="user", config=cfg)
        reader = store_mod.OnionStore(path, connect=opener, require_encryption=False)
        fresh = reader.load("c1", lib.OnionState())
        assert [e.text for e in fresh.core.active()] == ["The user is Alice, in Lyon."], "pin and supersession were saved"
        assert [r.resolved for r in fresh.ledger.all()][0] is True, "the recall was saved"
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
