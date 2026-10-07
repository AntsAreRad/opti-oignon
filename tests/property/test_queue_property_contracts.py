#!/usr/bin/env python3
"""Property contracts for the onion's queue, under seeded generated conversations.

A contract here states one property and holds it over conversations drawn
from a generator of the standard library, seeded: the same conversations on
every run, so a red names the seed and the case that broke it. The user
types names, places, dates, numbers and decisions; the assistant reports
back; the summariser drops sentences at random and sometimes invents one.

  * QP1 -- every step of the queue evicts its span, and every peel it makes
    passes both faces of the gate.
  * QP2 -- what a repair stitches and what a held span keeps are words the
    user typed, at their place in the Cellar, and nothing else.
  * QP3 -- a span that comes back with the refusal it had is never asked for
    a third summary.
  * QP4 -- through any interleaving of new turns and steps, every turn is in
    the Flesh or under exactly one receipt.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import random
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _isolation import isolate, source  # noqa: E402

_SEEDS = (11, 23, 47)
_MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian")
_NAMES = ("Alice", "Carol", "Dave", "Erin")
_TOOLS = ("Docker", "Podman", "Nginx", "Redis", "Kafka")
_PLACES = ("Berlin", "Lisbon", "Oslo", "Turin")


def _open():
    loaded, restore = isolate(
        targets={f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _MODULES},
        blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils"),
        packages=("opti_oignon.memory",),
    )
    loaded["opti_oignon.memory.probes"]._native = lambda: None
    return loaded["opti_oignon.memory.librarian"], loaded, restore


def _cases(n):
    for seed in _SEEDS:
        rng = random.Random(seed)
        for case in range(n):
            yield seed, case, rng


def _typed(rng):
    said = []
    for _ in range(rng.randint(1, 3)):
        kind = rng.randrange(4)
        if kind == 0:
            said.append(f"Then {rng.choice(_NAMES)} moved the build to {rng.choice(_PLACES)} on "
                        f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}.")
        elif kind == 1:
            said.append(f"We keep {rng.choice(_TOOLS)} on the {rng.choice(_PLACES)} server.")
        elif kind == 2:
            said.append(f"The queue holds {rng.randint(2, 900)} jobs tonight.")
        else:
            said.append("Thanks, that helps a lot.")
    return " ".join(said)


def _assistant(rng):
    return rng.choice((
        f"Noted: the {rng.choice(_PLACES)} build runs {rng.randint(2, 90)} jobs a day, a sensible load for it.",
        "Understood. I will keep that in mind for the next steps of the plan.",
        f"The logs from {rng.choice(_PLACES)} look clean since {rng.randint(2, 30)} March.",
    ))


def _conversation(rng, turns):
    out = []
    for i in range(turns):
        if i % 2 == 0:
            out.append({"role": "user", "origin": "typed", "segments": [], "content": _typed(rng)})
        else:
            out.append({"role": "assistant", "origin": "assistant", "segments": [], "content": _assistant(rng)})
    return out


def _lossy(rng, probes):
    """A summariser that keeps each sentence by chance and sometimes invents one."""

    def summarize(turns):
        kept = [s for t in turns for s in probes.sentences(t["text"]) if rng.random() < 0.6]
        if rng.random() < 0.3:
            kept.append(f"Zoe approved {rng.randint(1000, 9999)} euros.")
        return " ".join(kept)

    return summarize


def _tiny(loaded):
    composer = loaded["opti_oignon.memory.composer"]
    return composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=1, turn=60)


def _emptied(lib, loaded, rng, gate):
    """A generated conversation, mirrored and evicted to the last span; the state and every step's outcome."""
    probes = loaded["opti_oignon.memory.probes"]
    lib.reset_librarian()
    state = lib.state_for("c")
    state.mirror(_conversation(rng, rng.randint(2, 8)))
    summarize, steps = _lossy(rng, probes), []
    for _ in range(16):
        if not state.flesh.turns():
            break
        steps.append(lib.curate(state, summarize, gate=gate, budget=_tiny(loaded)))
    return state, steps


def test_qp1_every_step_evicts_its_span_and_every_peel_it_makes_passes_both_faces_of_the_gate():
    lib, loaded, restore = _open()
    try:
        peels, probes = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.probes"]
        gate = replace(peels.load_gate(), span_turns=2)
        made = repaired = 0
        for seed, case, rng in _cases(12):
            state, steps = _emptied(lib, loaded, rng, gate)
            assert all(step.evicted for step in steps), (seed, case, [s.rung for s in steps])
            assert state.flesh.turns() == [], (seed, case, "the queue stopped short")
            for peel in state.tree.all():
                span = state.cellar.get(peel.sources[0])
                judged = peels.decide(span, probes.generate_probes(span, gate.lexicon), peel.text, gate)
                assert judged.accepted, (seed, case, peel.rung, judged.reason[:80])
                made += 1
                repaired += peel.rung == "repaired"
        assert made >= 20, "control: peels were made to judge"
        assert repaired >= 5, "control: repaired peels were among them, each judged"
    finally:
        restore()


def test_qp2_what_a_repair_stitches_and_a_held_span_keeps_are_words_the_user_typed():
    lib, loaded, restore = _open()
    try:
        peels, probes = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.probes"]
        gate = replace(peels.load_gate(), span_turns=2)
        places = 0
        for seed, case, rng in _cases(12):
            state, _steps = _emptied(lib, loaded, rng, gate)
            kept = [(p.sources[0], unit) for p in state.tree.all() for unit in p.stitched]
            kept += [(r.key, unit) for r in state.ledger.all() for unit in r.anchors]
            for key, (turn_id, start, stop) in kept:
                (turn,) = [t for t in state.cellar.get(key) if t["turn_id"] == turn_id]
                assert turn["origin"] == "typed", (seed, case, turn_id)
                assert turn["text"][start:stop] in probes.sentences(turn["text"]), (seed, case, turn_id)
                places += 1
        assert places >= 10, "control: units were stitched or kept"
    finally:
        restore()


def test_qp3_a_span_that_comes_back_with_the_refusal_it_had_is_never_asked_for_a_third_summary():
    lib, loaded, restore = _open()
    try:
        peels, probes = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.probes"]
        gate = replace(peels.load_gate(), span_turns=2)
        ladder = peels.load_ladder()
        marked = 0
        for seed, case, rng in _cases(15):
            messages, refusals, asked = _conversation(rng, 2), {}, []
            lossy = _lossy(rng, probes)
            text = lossy([{"turn_id": "t", "text": m["content"]} for m in messages])

            def reask(turns, missing, text=text):
                asked.append(missing)
                return text

            for _ in range(3):
                state = lib.OnionState()
                state.mirror(messages)
                peels.advance(flesh=state.flesh, cellar=state.cellar, ledger=state.ledger, tree=state.tree, gate=gate,
                              ladder=ladder, summarize=lambda turns, text=text: text, reask=reask, refusals=refusals)
            assert len(asked) <= 1, (seed, case, len(asked))
            marked += len(refusals)
        assert marked >= 5, "control: refusals were marked"
    finally:
        restore()


def test_qp4_through_any_interleaving_every_turn_is_in_the_flesh_or_under_exactly_one_receipt():
    lib, loaded, restore = _open()
    try:
        peels, probes = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.probes"]
        gate = replace(peels.load_gate(), span_turns=2)
        checked = 0
        for seed, case, rng in _cases(10):
            lib.reset_librarian()
            state = lib.state_for("c")
            messages, summarize = _conversation(rng, rng.randint(4, 10)), _lossy(rng, probes)
            shown = 0
            for _ in range(12):
                if shown < len(messages) and rng.random() < 0.5:
                    shown = min(len(messages), shown + rng.randint(1, 3))
                    state.mirror(messages[:shown])
                else:
                    lib.curate(state, summarize, gate=gate, budget=_tiny(loaded))
                with state.lock:
                    flesh = [t["turn_id"] for t in state.flesh.turns()]
                    receipted = [tid for r in state.ledger.all() for tid in r.turn_ids]
                assert len(receipted) == len(set(receipted)), (seed, case, "a turn under two receipts")
                assert sorted(flesh + receipted) == [f"t{i:04d}" for i in range(1, state.seen + 1)], (seed, case)
                checked += 1
        assert checked >= 300, "control: every step was checked"
    finally:
        restore()
