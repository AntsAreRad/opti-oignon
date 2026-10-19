#!/usr/bin/env python3
"""Host A/B: drift with the onion memory against drift without it.

The design measures drift as contradictions per 1000 turns against a
no-onion baseline, and only a real model can give the answers. So the
number comes from the host, and everything around the model is here, held
by contracts: a scripted conversation whose facts change on known turns,
the rule that decides what an answer contradicts, the two ways of building
the model's context, and the driver that runs both over the same turns.

The two arms differ the way the chat path does. With the onion switched
on, the chat executor sends each turn of a conversation it answers as the
onion's journal lays it: the librarian plans the turn's window, then
composes it -- the system prompt, the Core, the peels and the receipts
digest, the Flesh, the sources the turn brings and the turn, each data
part wrapped as untrusted data -- and that window takes the place of the
history window. So both arms send the same system prompt first and the
same turn last, to the same model; between them the plain arm sends its
history window, the onion arm its journal -- planned within the model's
window and counted with the model's counter, as the executor's -- with the
context it was composed for. The librarian runs in this process with
persistence off and its counts kept in the process, so none of the user's
memories enters either arm and none of the bench's counts joins the
user's: no memory store is read or written. It curates in one of two
modes. ``launcher``, the
default, is the chat path's rhythm: the librarian's launcher fires a burst
once the conversation grew by ``min_new_turns``, on a thread of its own,
while the next turns are answered, and the arm ends only once its bursts
have. ``fast`` is the bench as it was: curation after every turn, on the
answering thread, so the reading never depends on timing, though it
engages the onion more than the chat path does. Both ask for the second
summary, and the registry's summariser runs as on the chat path: each
burst, or each measured turn in the fast mode, is a run of its own, with
its budget of time on the model. The requests go through the inference
registry as the application's do, so the resource governor admits each of
them and records its decision in its own store, as it does for any
request; the answering requests hold no interactive ticket, so the
background gate does not hold a burst behind them as it would behind a
chat turn, and a turn's latency during a burst is the machine's to
measure.

The report names the mode, and carries what the onion did -- bursts,
evictions by rung, refusals by motive, the window's tokens at each turn,
its layers summed (``block_tokens``), what the memory laid of them
(``memory_tokens``) and the Flesh's (``flesh_tokens``) -- with the
version of the probe generator and the fingerprint of the decision
lexicon, so that two readings are compared only under the same. It also
carries what the gate weighed for each summary the librarian handed it
(``gate``): the span's turns and words, the summary's, the words the peel
shows, the length ratio, the novelty and the probes' coverage, whether it
was accepted and the motives of a refusal, each decision and their
spread -- numbers and motive names, never a word.

Each arm's reading also scores its probes on their own (``probes``):
how many asked, how many answered with the fact expected, contradicted,
or left undecided, and the share recalled; the report gives the
difference of the two arms' recall. Most undecided sentences are answers
to the long turns of talk, which no template reads: they weigh nothing in
the recall.

With ``--answers PATH``, once the run has measured, every turn's question
(with the fact a probe expects) and both arms' answers go to PATH as JSON
lines, each answer's sentences with their verdicts (contradicts, agrees,
decided, undecided): what a rate and its undecided share were read from.
The conversation is the bench's own, synthetic; the file is written
whole, so a run that breaks leaves an earlier one as it was.

With ``--home-rung R`` the onion arm is planned under ``onion.yaml`` with
``home_rung`` set to R, in this process alone (``pinned_rung``): the
journal's tool room and result cap, set at the home rung and held to
another in proportion, are scaled so each rung keeps the value the shipped
file gives it, the machine's ``local.yaml`` read beside the copy as beside
the file, and no tracked file changes. With ``--states DIR``, a new
directory, at each probe turn the onion arm's state before its window is
planned -- the bench's own conversation, synthetic, in a store file in
plaintext -- the turn, the conversation as stored and the window it sent
go under DIR, for ``scripts/drift_pairs.py`` to compose that window again
under another tree; the user's store is still never read or written.

A sentence the deterministic templates cannot parse is counted undecided,
never as agreement: the host's model judge is the remedy, and the report
says how much was left to it. The templates read English, and so does the
conversation.

Run on the host, never in CI::

    python3 scripts/drift_ab.py                                # both arms, one JSON report
    python3 scripts/drift_ab.py --mode fast                    # curation after every turn, as before
    python3 scripts/drift_ab.py --model llama3:8b              # another answering model
    python3 scripts/drift_ab.py --librarian-model qwen3:4b     # another librarian
    python3 scripts/drift_ab.py --history-tokens 2048
    python3 scripts/drift_ab.py --answers /tmp/drift_answers.jsonl  # the answers and their verdicts too

Without a backend for the answering model or for the librarian's, or when
the backend lists either model as not installed, it asks nothing, names
the models the backend does serve, prints nothing that looks like a result
and exits 2. Before the first turn it then tries the pair in the run's
order -- one minimal request to the answering model, at the context the
onion arm sends its turns with, then one to the librarian's while the
first is still loaded, which is the run's peak -- so
a pair that cannot run together, the governor refusing to load the second
model beside the first, is refused in seconds and not after the plain arm.
A request that fails once the run has started ends the run the same way: a
half-run is not a measurement.
"""

import argparse
import contextlib
import importlib
import importlib.util
import inspect
import json
import math
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

checkpoint_before_apply = True

_REPO = Path(__file__).resolve().parent.parent
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# The plain arm's history window, in estimated tokens. Small on purpose: the
# conversation has to outgrow it, as it outgrows the onion's own window, for
# the arms to differ in what they remember.
HISTORY_TOKENS = 1024
MAX_CURATION_STEPS = 16
# How the onion arm curates: at the chat path's rhythm, or after every turn.
MODES = ("launcher", "fast")
# How both arms ask the answering model: greedy, seeded, the same room for the answer.
SAMPLING = {"temperature": 0.0, "seed": 0, "num_predict": 128}

SYSTEM_PROMPT = (
    "You are a helpful assistant in a long working conversation. Answer briefly, "
    "in English. When asked for a fact in a given sentence form, answer with that "
    "sentence and nothing else."
)


@dataclass(frozen=True)
class Turn:
    """One user turn: its text, the statement it makes true from here on, the one it retires."""

    text: str
    fact: str = ""
    replaces: str = ""
    probe: bool = False
    expects: str = ""


def estimate_tokens(text):
    """The composer's fallback estimate, restated so the windows are comparable."""
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))


_TOPICS = (
    "the search index", "the billing export", "the login service",
    "the image cache", "the mail relay", "the metrics pipeline",
)


def _filler(n):
    topic = _TOPICS[n % len(_TOPICS)]
    return (
        f"Let us go over {topic} for step {n}. We replayed the same recorded traffic on both "
        f"clusters, and the median latency went from {90 + n} to {60 + n} milliseconds while "
        f"the error count stayed flat over the whole replay. The cutover window we have in mind "
        f"is a quiet evening after the weekly report, the rollback path keeps the old cluster "
        f"warm for a full week, and the dashboards and alerts that must exist before the switch "
        f"are listed in the plan we wrote together last time. The owners of the upstream "
        f"services have read the plan and asked only for a heads-up an hour before the switch, "
        f"which the rota already covers. Is there anything in that plan you would change before "
        f"we commit to it, and which signal would you watch first during the first hour?"
    )


def _probe(question, form, expects):
    return Turn(f"Quick check: {question} Answer with one sentence of the form '{form}'.", probe=True, expects=expects)


def _conversation():
    turns = []
    counter = iter(range(1, 10_000))

    def fill(count):
        turns.extend(Turn(_filler(next(counter))) for _ in range(count))

    turns += [
        Turn("Some context before we start: Alice lives in Berlin.", fact="Alice lives in Berlin"),
        Turn("Also useful: Alice works at Contoso.", fact="Alice works at Contoso"),
        Turn("Our date to keep in mind: the Harvest release is on 2026-05-12.", fact="The Harvest release is on 2026-05-12"),
        Turn("Bob lives in Lyon.", fact="Bob lives in Lyon"),
        Turn("The budget review is on 2026-04-30.", fact="The budget review is on 2026-04-30"),
        Turn("Carol works at Fabrikam.", fact="Carol works at Fabrikam"),
    ]
    fill(9)
    turns.append(_probe("where does Bob live?", "Bob lives in <city>.", "Bob lives in Lyon"))
    fill(9)
    turns.append(_probe("where does Alice work?", "Alice works at <company>.", "Alice works at Contoso"))
    fill(9)
    turns.append(_probe("when is the budget review?", "The budget review is on <YYYY-MM-DD>.", "The budget review is on 2026-04-30"))
    turns += [
        Turn("News: the move is done, Alice lives in Oslo.", fact="Alice lives in Oslo", replaces="Alice lives in Berlin"),
        Turn("The Harvest release moved: the Harvest release is on 2026-06-02.",
             fact="The Harvest release is on 2026-06-02", replaces="The Harvest release is on 2026-05-12"),
        Turn("Carol changed jobs: Carol works at Northwind.", fact="Carol works at Northwind", replaces="Carol works at Fabrikam"),
    ]
    fill(9)
    turns.append(_probe("where does Alice live?", "Alice lives in <city>.", "Alice lives in Oslo"))
    fill(9)
    turns.append(_probe("when is the Harvest release?", "The Harvest release is on <YYYY-MM-DD>.", "The Harvest release is on 2026-06-02"))
    fill(9)
    turns.append(_probe("where does Carol work?", "Carol works at <company>.", "Carol works at Northwind"))
    fill(9)
    turns.append(_probe("where does Bob live?", "Bob lives in <city>.", "Bob lives in Lyon"))
    fill(9)
    return tuple(turns)


TURNS = _conversation()


def held_facts(turns, upto):
    """The statements holding after the first ``upto`` turns, as ``(id, statement)`` pairs."""
    held = {}
    for index, turn in enumerate(turns[:upto]):
        if turn.replaces:
            held.pop(turn.replaces, None)
        if turn.fact:
            held[turn.fact] = f"f{index + 1:03d}"
    return [(fact_id, statement) for statement, fact_id in held.items()]


def _templates():
    """The drift templates, loaded from their file: standard library only, no package import."""
    spec = importlib.util.spec_from_file_location("drift_ab_templates", _REPO / "opti_oignon" / "memory" / "drift.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _claim_key(claim):
    return (claim.subject, claim.predicate, claim.obj, claim.negated)


def reading(turns, answers, *, arm, source, templates=None, verdicts=None):
    """What each answer contradicts, and agrees with, among the facts holding at its turn.

    With ``verdicts``, a list, each answer's sentences are appended to it as
    read, one list per answer: each sentence with its verdict, "contradicts"
    a fact holding at its turn, else "agrees" with one, else "decided"
    (parsed, neither agreeing nor contradicting), or "undecided" (no
    template parses it).

    The probes are read on their own (``probes``): each answer to a turn
    that asks for a fact is "agrees" when one of its sentences states the
    fact expected and none contradicts that fact -- an answer that gives the
    retired value beside the current one recalls nothing -- else
    "contradicts" when one contradicts a fact holding, else "decided" or
    "undecided"; ``recall`` is the share of probes that agree. The undecided
    sentences of the other turns -- answers to talk the templates were never
    meant to read -- weigh nothing there.
    """
    if len(answers) > len(turns):
        raise ValueError(f"{len(answers)} answers for {len(turns)} turns")
    drift = templates or _templates()
    sentences = decided = agreements = 0
    at = []
    probes = dict.fromkeys(("asked", "agrees", "contradicts", "decided", "undecided"), 0)
    for index, answer in enumerate(answers):
        held = held_facts(turns, index + 1)
        held_claims = {_claim_key(c) for c in (drift._parse(s) for _id, s in held) if c is not None}
        turn = turns[index]
        wanted = drift._parse(turn.expects) if turn.probe and turn.expects else None
        expected = _claim_key(wanted) if wanted is not None else None
        probed = next((fact_id for fact_id, statement in held if statement == turn.expects), None)
        recalled = hedged = False
        said = []
        for sentence in (s.strip() for s in _SENTENCE_SPLIT.split(str(answer or "")) if s.strip()):
            sentences += 1
            claim = drift._parse(sentence)
            if claim is None:
                said.append({"text": sentence, "verdict": "undecided"})
                continue
            decided += 1
            agrees = _claim_key(claim) in held_claims
            if agrees:
                agreements += 1
            if expected is not None and _claim_key(claim) == expected:
                recalled = True
            found = list(drift.find_contradictions(held, sentence))
            for contradiction in found:
                at.append({"turn": index + 1, "sentence": sentence, "fact": contradiction.fact_id,
                           "template": contradiction.template})
                hedged = hedged or (turn.probe and contradiction.fact_id == probed)
            said.append({"text": sentence, "verdict": "contradicts" if found else "agrees" if agrees else "decided"})
        if verdicts is not None:
            verdicts.append(said)
        if turn.probe:
            probes["asked"] += 1
            kinds = {s["verdict"] for s in said}
            probes["agrees" if recalled and not hedged else "contradicts" if "contradicts" in kinds
                   else "undecided" if kinds <= {"undecided"} else "decided"] += 1
    turns_judged = len(answers)
    return {
        "arm": arm,
        "source": source,
        "turns": turns_judged,
        "sentences": sentences,
        "decided": decided,
        "undecided": sentences - decided,
        "agreements": agreements,
        "contradictions": len(at),
        "rate_per_1000_turns": None if turns_judged == 0 else round(1000.0 * len(at) / turns_judged, 3),
        "at": at,
        "probes": {**probes, "recall": None if not probes["asked"] else round(probes["agrees"] / probes["asked"], 3)},
    }


def history_window(history, tokens, estimate=estimate_tokens):
    """The newest whole messages that fit in ``tokens``, oldest first."""
    kept, used = [], 0
    for message in reversed(history):
        cost = estimate(message["content"])
        if used + cost > tokens:
            break
        kept.append(message)
        used += cost
    return list(reversed(kept))


def plain_arm(ask, *, history_tokens=HISTORY_TOKENS, system=SYSTEM_PROMPT):
    """The chat without the onion: the system prompt, the history window, the turn."""
    history = []

    def answer(text):
        messages = [{"role": "system", "content": system}, *history_window(history, history_tokens),
                    {"role": "user", "content": text}]
        reply = str(ask(messages))
        history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": reply}])
        return reply

    return answer


def _takes(call, name):
    """Whether ``call`` takes a keyword argument ``name``."""
    try:
        parameters = inspect.signature(call).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def _envelope(wrap):
    """``wrap`` as the journal calls an envelope, ``wrap(content, *, source, frames)``; a wrap of one argument
    takes every part, whatever its source."""
    if _takes(wrap, "source"):
        return wrap

    def enveloped(content, *, source=None, frames=False):
        return wrap(content)

    return enveloped


def _turns_stored(history):
    """The conversation as the store hands it to the librarian: each turn with the origin the executor gives it,
    the user's words typed, the answer the assistant's."""
    return [dict(m, origin="typed" if m["role"] == "user" else "assistant") for m in history]


_HOME_RUNG = re.compile(r"^home_rung:[ \t]*([0-9]+)[ \t]*(?:#.*)?$", re.M)


class RungRefused(ValueError):
    """A rung the onion's configuration cannot be pinned to, by name."""


def pinned_text(text, rung):
    """``onion.yaml``'s ``text`` with ``home_rung`` set to ``rung``, and the journal's tool room and result cap --
    set at the home rung and held to another in proportion -- scaled so each rung keeps the value the shipped file
    gives it (unscaled, 4096 is refused: the tool room and the answer floor would leave nothing of it). Refused when
    the file does not name ``home_rung`` once at its top level."""
    found = _HOME_RUNG.findall(text)
    if len(found) != 1:
        raise RungRefused("onion.yaml does not name home_rung once at its top level")
    home = int(found[0])
    text = _HOME_RUNG.sub(f"home_rung: {rung}", text)
    for key in ("tool_room", "tool_result_cap"):
        line = re.compile(rf"^(  {key}:[ \t]*)([0-9]+)", re.M)
        held = line.findall(text)
        if len(held) == 1:
            value = max(1, int(held[0][1]) * rung // home)
            text = line.sub(lambda match, value=value: f"{match.group(1)}{value}", text)
    return text


@contextlib.contextmanager
def pinned_rung(rung):
    """The onion read with ``home_rung`` set to ``rung``, in this process, for the block: a copy of ``onion.yaml``
    (``pinned_text``) that the package's own loaders read first, and the three readers that hold its path as a
    module constant -- the composer, the librarian and the peels -- pointed at it, then back, whatever ends the
    block. The machine's ``local.yaml``, read beside the file the librarian's configuration comes from, is read
    beside the copy. Nothing tracked changes. None pins nothing; a copy the loaders refuse, or whose home is not
    ``rung``, is refused by name, the constants untouched."""
    if rung is None:
        yield
        return
    from opti_oignon.memory import composer, librarian, peels

    modules = (composer, librarian, peels)
    shipped = [(module, module._CONFIG) for module in modules]
    workdir = Path(tempfile.mkdtemp(prefix="oo-drift-rung-"))
    try:
        copy = workdir / "onion.yaml"
        copy.write_text(pinned_text(Path(composer._CONFIG).read_text(encoding="utf-8"), rung), encoding="utf-8")
        local = Path(librarian._CONFIG).with_name("local.yaml")
        if local.is_file():
            shutil.copyfile(local, workdir / "local.yaml")
        try:
            budgets = composer.load_budgets(copy)
            composer.load_journal(copy)
            librarian.load_config(copy)
            peels.load_gate(copy)
            peels.load_ladder(copy)
        except Exception as exc:  # noqa: BLE001 - the pinned copy refused, by name
            raise RungRefused(f"the pinned onion.yaml does not read ({type(exc).__name__}: {exc})") from None
        if budgets.home != rung:
            raise RungRefused(f"the pinned copy's home is {budgets.home}, not {rung}")
        for module in modules:
            module._CONFIG = copy
        yield
    finally:
        for module, saved in shipped:
            module._CONFIG = saved
        shutil.rmtree(workdir, ignore_errors=True)


def _kept(states, turn, conversation_id, state, meta):
    """The onion arm's state before turn ``turn``'s window is planned, under ``states``: the store, in plaintext --
    the bench's own conversation -- saved under the state's lock, as every save is; what the window is planned
    from; the home rung it is planned under; and whether a burst held the conversation (``held``), when the window
    may lay over a shadow a composition with no burst in flight does not."""
    store = importlib.import_module("opti_oignon.memory.onion_store")
    composer = importlib.import_module("opti_oignon.memory.composer")
    directory = Path(states)
    directory.mkdir(parents=True, exist_ok=True)
    with state.lock:
        held = state.slot.locked()
        store.OnionStore(directory / f"t{turn:03d}.db", connect=sqlite3.connect, require_encryption=False).save(
            conversation_id, state)
    (directory / f"t{turn:03d}.json").write_text(json.dumps(dict(
        meta, turn=turn, cid=conversation_id, home_rung=composer.load_budgets().home, held=held)), encoding="utf-8")


def _sent(states, turn, reply, messages, window):
    """The window turn ``turn`` sent, and its answer, beside its state."""
    Path(states, f"t{turn:03d}.window.json").write_text(json.dumps(
        {"reply": reply, "messages": messages, "layers": dict(window.layers), "cut": [list(c) for c in window.cut],
         "rung": window.rung, "num_ctx": int(window.num_ctx)}), encoding="utf-8")


def onion_arm(ask, *, librarian, config, summarize, wrap, reask=None, mode="fast", resolve=None,
              system=SYSTEM_PROMPT, conversation_id="drift-ab", gate=None, budget=None, model="bench",
              model_window=None, count=estimate_tokens, states=None, record=()):
    """The chat with the onion: each turn sent as the window the onion's journal composed for it.

    As on the chat path: the librarian plans the turn's window -- the
    conversation as stored before it, the turn as typed, counted with
    ``count`` for ``model`` and planned within ``model_window`` (the host
    command hands both as the executor reads them: the model's counter
    and the model's window; None plans under ``home_rung``) -- and
    composes it under the plan's rung; the messages composed are sent,
    their labels left behind. A window refused by name ends the run: a
    half-run is not a measurement.
    ``ask`` is handed the window too when it takes a ``window`` keyword, so
    the host command sends it with the context it was composed for.
    ``wrap`` is the envelope every data part rides in, ``wrap(content, *,
    source, frames)``; a wrap of one argument takes every part whatever its
    source.

    The librarian runs in this process and never writes a store: a
    configuration with a persistence path is refused, so a measurement can
    never touch the user's onion, and so is one with a counters path, so
    the bench's counts never join the user's. ``mode`` is one of ``MODES``:
    the launcher's rhythm, or curation after every turn (the default here,
    the bench as it was; the host command's default is the launcher's).
    ``summarize`` and ``reask`` None are the registry's, through
    ``resolve``, each burst or measured turn a run of its own; ``gate``
    and ``budget`` are what it curates under, the shipped ones when None.
    The arm keeps the window's tokens at each turn, its layers summed, in
    ``block_tokens``, what the memory laid of them -- the Core, the
    receipts digest, the peels and the tail -- in ``memory_tokens``, the
    Flesh's in ``flesh_tokens``, and its ``settle`` waits for the bursts it
    launched. With ``states``, a directory, each turn numbered in
    ``record`` (from 1) leaves there its state before its window is planned,
    the turn and the conversation as stored, and the window it sent with
    its answer (``_kept``, ``_sent``).
    """
    if getattr(config, "persist_path", ""):
        raise ValueError("the drift A/B runs the librarian in process: a persistence path would write the user's store")
    if getattr(config, "counters_path", ""):
        raise ValueError("the drift A/B keeps its counts in the process: a counters path would add them to the user's")
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {', '.join(MODES)}")
    if states is not None and not str(states):
        raise ValueError("the drift A/B keeps states only under a directory it is named")
    history, launched, tokens, memory, flesh = [], [], [], [], []
    envelope, sized = _envelope(wrap), _takes(ask, "window")

    def callers():
        """The summariser and the second asking for one run: those handed in, or the registry's under a run of their own."""
        if summarize is not None:
            return summarize, reask, None
        run = librarian._Run(config)
        return (librarian.registry_summarizer(config, resolve=resolve, run=run),
                librarian.registry_reasker(config, resolve=resolve, run=run), run)

    def burst(cid):
        summariser, again, run = callers()
        librarian._curation_burst(cid, config=config, summarize=summariser, reask=again, gate=gate, budget=budget,
                                  run=run)

    def runner(cid):
        # The launcher's own way: a burst on a thread of its own, while the
        # next turns are answered.
        thread = threading.Thread(target=burst, args=(cid,), name="oo-drift-burst", daemon=True)
        launched.append(thread)
        thread.start()

    def curate_now(state):
        summariser, again, run = callers()
        # As a burst aims once a journal laid the conversation: at the journal's
        # mark of the Flesh's cap at its rung, read at the cost the journal lays a
        # turn at, so the plan finds no Flesh to catch up; a budget handed in wins.
        aim, cost, record = budget, None, getattr(state, "journal", None)
        if aim is None and getattr(record, "burst", None) is not None:
            sticky, mark, cost = record.burst
            aim = replace(sticky, flesh=math.floor(mark * sticky.flesh))
        if run is not None:
            run.hold()
        try:
            for _ in range(MAX_CURATION_STEPS):
                outcome = librarian.curate(state, summariser, gate=gate, budget=aim, reask=again, cost=cost)
                # As a burst does: a model that failed is not asked again this turn.
                summariser, again = librarian.broken(outcome, summariser, again)
                if not outcome.evicted:
                    break
        finally:
            if run is not None:
                run.release()

    def answer(text):
        turn = len(history) // 2 + 1
        keep = states is not None and turn in record
        if keep:
            _kept(states, turn, conversation_id, librarian.state_for(conversation_id, config),
                  {"question": text, "model": model, "model_window": model_window, "system": system,
                   "history": _turns_stored(history)})
        plan = librarian.plan_window(
            conversation_id, model=model, count=count, system=system,
            turn={"role": "user", "content": text, "origin": "typed", "segments": []}, wrap=envelope,
            model_window=model_window, messages=_turns_stored(history), config=config, gate=gate,
        )
        window = librarian.compose_window(plan)
        refusal = getattr(window, "refusal", None)
        if refusal is not None:
            raise RuntimeError(f"the onion window was refused ({refusal.motive}): {list(refusal.figures)}")
        layers = dict(window.layers)
        tokens.append(sum(layers.values()))
        # What the memory laid, no retrieval or web riding on this bench, and the Flesh, beside the whole.
        memory.append(sum(layers.get(name, 0) for name in ("core", "receipts", "peels", "tail")))
        flesh.append(layers.get("flesh", 0))
        messages = [{"role": role, "content": content} for role, content, _label in window.messages]
        reply = str(ask(messages, window=window) if sized else ask(messages))
        if keep:
            _sent(states, turn, reply, messages, window)
        history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": reply}])
        # The librarian reads each turn with the origin the executor gives
        # it: the user's words typed, the answer the assistant's.
        mirrored = _turns_stored(history)
        if mode == "launcher":
            librarian.maybe_curate(conversation_id, mirrored, config=config, runner=runner)
        else:
            state = librarian.state_for(conversation_id, config)
            state.mirror(mirrored)
            curate_now(state)
        return reply

    def settle():
        for thread in launched:
            thread.join()

    answer.block_tokens = tokens
    answer.memory_tokens = memory
    answer.flesh_tokens = flesh
    answer.settle = settle
    return answer


def run_arm(turns, answer):
    """Every turn through one arm, in order, and the bursts it launched ended; the answers."""
    answers = [answer(turn.text) for turn in turns]
    settle = getattr(answer, "settle", None)
    if settle is not None:
        settle()
    return answers


def _stats(values):
    """The spread of ``values``, None left out: how many, the least, the median, the ninth decile, the most, the
    mean; None throughout for none."""
    kept = sorted(float(v) for v in values if v is not None)
    if not kept:
        return {"n": 0, "min": None, "median": None, "p90": None, "max": None, "mean": None}

    def at(share):
        return kept[min(len(kept) - 1, int(math.ceil(share * len(kept))) - 1)]

    return {"n": len(kept), "min": round(kept[0], 3), "median": round(at(0.5), 3), "p90": round(at(0.9), 3),
            "max": round(kept[-1], 3), "mean": round(sum(kept) / len(kept), 3)}


def recording_gate(peels, probes, figures, unread):
    """``peels.decide`` with each decision's figures appended to ``figures``: numbers and motive names, never a word.

    Each figure is what the gate weighed: the span's turns and words, the
    summary's words, the words the peel shows (the summary and the user's
    words it references), the length ratio and the novelty it was held to,
    the share of the span's facts the probes ask for, whether it was
    accepted, and the motives of a refusal. Reading the figures never
    breaks the decision it reads: a decision whose figures could not be
    read is counted in ``unread``, by the name of what stopped it.
    """
    decide = peels.decide

    def recorded(span, probes_, text, gate, refs=()):
        decision = decide(span, probes_, text, gate, refs)
        try:
            figures.append({
                "span_turns": len(span),
                "span_words": sum(probes.word_count(str(turn.get("text", "") or "")) for turn in span),
                "summary_words": probes.word_count(str(text or "")),
                "shown_words": probes.word_count(peels.joined(text, peels.shown_words(span, refs))[0]),
                "length_ratio": decision.length_ratio,
                "novelty": decision.novelty,
                "probe_coverage": decision.probe_coverage,
                "accepted": bool(decision.accepted),
                "motives": list(peels.refusal_motives(decision, gate)),
            })
        except Exception as exc:  # noqa: BLE001 - the bench's instrument never breaks the run it measures
            unread.append(type(exc).__name__)
        return decision

    return recorded


def gate_report(figures, max_length_ratio, unread=()):
    """What the gate weighed over a run, decision by decision and summed: numbers and motive names only; and how
    many decisions were taken whose figures could not be read."""
    motives = {}
    for figure in figures:
        for motive in figure["motives"]:
            motives[motive] = motives.get(motive, 0) + 1
    over = [f for f in figures if f["length_ratio"] is not None and max_length_ratio is not None
            and f["length_ratio"] > max_length_ratio]
    return {
        "decisions": len(figures),
        "unread": len(unread),
        "accepted": sum(1 for f in figures if f["accepted"]),
        "motives": dict(sorted(motives.items())),
        "max_length_ratio": max_length_ratio,
        "over_length": len(over),
        "length_ratio": _stats(f["length_ratio"] for f in figures),
        "span_words": _stats(f["span_words"] for f in figures),
        "summary_words": _stats(f["summary_words"] for f in figures),
        "reference_words": _stats(max(0, f["shown_words"] - f["summary_words"]) for f in figures),
        "each": [
            {key: (round(value, 4) if isinstance(value, float) else value) for key, value in figure.items()}
            for figure in figures
        ],
    }


def write_answers(path, turns, arms):
    """Every turn's question and each arm's answer, with its sentences' verdicts, one JSON line per arm and turn.

    ``arms`` maps an arm's name to ``(answers, verdicts)``. The file is
    written whole beside its place and moved into it, so a run that breaks
    leaves an earlier file as it was. The conversation is the bench's own,
    synthetic: no word of anyone's.
    """
    target = Path(path)
    staged = target.with_name(target.name + ".partial")
    try:
        with staged.open("w", encoding="utf-8") as handle:
            for arm, (answers, verdicts) in arms.items():
                for index, (turn, answer, said) in enumerate(zip(turns, answers, verdicts), start=1):
                    handle.write(json.dumps({"arm": arm, "turn": index, "question": turn.text,
                                             "expects": turn.expects if turn.probe else None, "answer": answer,
                                             "sentences": said}, ensure_ascii=False) + "\n")
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)


def compare(onion, plain):
    """Both readings and the difference of their rates, onion minus plain."""
    a, b = onion["rate_per_1000_turns"], plain["rate_per_1000_turns"]
    ra, rb = onion["probes"]["recall"], plain["probes"]["recall"]
    return {"onion": onion, "plain": plain,
            "delta_rate_per_1000_turns": None if a is None or b is None else round(a - b, 3),
            "delta_probe_recall": None if ra is None or rb is None else round(ra - rb, 3)}


def _registry_resolver():
    from opti_oignon.inference_backend import get_backend_registry, init_backends_from_config

    try:
        init_backends_from_config()
    except Exception as exc:  # noqa: BLE001 - reported, the resolution below decides
        print(f"backend configuration not applied: {exc!r}", file=sys.stderr)

    def resolve(model):
        try:
            return get_backend_registry().resolve_backend(model)
        except Exception:  # noqa: BLE001 - a broken registry is absence
            return None

    return resolve


def _unserved(name, backend):
    """Why ``name`` cannot be measured on ``backend``, or None when it is served or nobody can tell.

    A name without a tag is the backend's latest tag of it. A listing the
    backend cannot give (None) or that fails says nothing either way, and
    the run is left to find out.
    """
    try:
        listing = backend.list_models()
    except Exception:  # noqa: BLE001 - a failed listing says nothing
        return None
    if listing is None:
        return None
    names = {str(getattr(entry, "name", entry)) for entry in listing}
    if name in names or (":" not in name and f"{name}:latest" in names):
        return None
    return f"{name!r} is not installed (the backend serves: {', '.join(sorted(names)) or 'no model'})"


_TRIAL = [{"role": "user", "content": "Reply with one word."}]


def _try_pair(backend, model, librarian_backend, config, num_ctx=None):
    """Why the pair cannot run together, or None: one minimal request each, in the run's order, the answering
    model's at ``num_ctx``, the context the onion arm sends its turns with, when it is known."""
    options = {"temperature": 0.0, "num_predict": 1, **({"num_ctx": num_ctx} if num_ctx else {})}
    try:
        backend.generate(model=model, messages=_TRIAL, options=options, think=False)
    except Exception as exc:  # noqa: BLE001 - reported by name, nothing measured
        return f"the answering model {model!r} could not answer ({exc!r})"
    try:
        librarian_backend.generate(
            model=config.model,
            messages=_TRIAL,
            options={"temperature": config.temperature, "num_predict": 1},
            keep_alive=config.keep_alive,
            think=False,
        )
    except Exception as exc:  # noqa: BLE001 - reported by name, nothing measured
        return f"the librarian's model {config.model!r} could not answer beside {model!r} ({exc!r})"
    return None


def _default_model():
    from opti_oignon.config import config

    return config.get_model("general")


def _governor_on():
    """Whether a resource governor admits this process's requests: present and switched on; unreadable is off."""
    try:
        from opti_oignon.resource_governor import get_resource_governor

        return bool(get_resource_governor().config.enabled)
    except Exception:  # noqa: BLE001 - an unreadable governor is off, as the executor reads it
        return False


def _model_window(model, backend):
    """The model's window as the executor plans the journal under it (``Executor._onion_model_window``): the
    context manager's, 32768 when it cannot be read, as the executor's fallback; then bounded, either way, on a
    llama-server engine by the smallest context of its slots, and on an in-process engine with no governor to admit
    it by the context it loads at."""
    window = 32768
    try:
        from opti_oignon.context_manager import get_model_limits

        window = int(get_model_limits(model).context_window)
    except Exception:  # noqa: BLE001 - unread: the executor's own fallback, still bounded below
        pass
    name = getattr(backend, "name", "")
    if name == "llama_server":
        try:
            contexts = [entry.get("n_ctx") for entry in (backend.slots() or []) if isinstance(entry, dict)]
            contexts = [value for value in contexts if type(value) is int and value > 0]
            if contexts:
                window = min(window, min(contexts))
        except Exception:  # noqa: BLE001 - the slot bound unread, as on the chat path
            pass
    elif name == "llama_cpp":
        held = getattr(backend, "_n_ctx", None)
        if type(held) is int and held > 0 and not _governor_on():
            window = min(window, held)
    return window


def _model_counter(model):
    """The counter the executor hands the journal for ``model``; the bench's estimate when it cannot be reached."""
    try:
        from opti_oignon.context_manager import estimate_tokens as counted
    except Exception:  # noqa: BLE001 - unreachable: the bench's own estimate
        return estimate_tokens
    return lambda text: counted(text, model)


def _trial_ctx(model_window=None):
    """The rung the onion arm's turns are sent at -- the largest of the ladder within ``home_rung`` and the model's
    window -- so the pair is tried at the run's peak; None when the budgets do not read."""
    try:
        from opti_oignon.memory import composer

        budgets = composer.load_budgets()
        bound = min(budgets.home, model_window) if model_window else budgets.home
        return max((rung for rung in budgets.rungs if rung <= bound), default=None)
    except Exception:  # noqa: BLE001 - unread: the trial as before
        return None


def _spread(values):
    """A figure the report carries per turn, with its mean and its largest."""
    return {"per_turn": values, "mean": round(sum(values) / len(values), 1) if values else 0,
            "max": max(values, default=0)}


def asker(backend, model, *, cold=False):
    """How both arms ask ``model`` on ``backend``: the same sampling and the same room for the answer. A window the
    journal composed goes with the context it was composed for, as on the chat path: an engine left at its own
    default context would cut it. ``cold`` asks each window of a model loaded for it alone (Ollama's keep-alive
    of 0): an engine reuses the prefix a request shares with the one before it, computed in other batches, and
    an answer then depends on the requests before it."""

    def ask(messages, window=None):
        options = dict(SAMPLING)
        if window is not None:
            options["num_ctx"] = int(window.num_ctx)
        extra = {"keep_alive": 0} if cold else {}
        response = backend.generate(model=model, messages=messages, options=options, think=False, **extra)
        return str(getattr(response, "content", "") or "")

    return ask


def main(argv=None, *, resolve=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="the answering model (default: the general route's)")
    parser.add_argument("--librarian-model", default=None, help="the librarian's model (default: onion.yaml's)")
    parser.add_argument("--history-tokens", type=int, default=HISTORY_TOKENS,
                        help="the plain arm's history window, in estimated tokens (the onion arm sends the journal's "
                             "window)")
    parser.add_argument("--mode", choices=MODES, default="launcher",
                        help="launcher: bursts at the chat path's rhythm, off the answering thread (default); "
                             "fast: curation after every turn, on the answering thread")
    parser.add_argument("--answers", default=None, metavar="PATH",
                        help="write every turn's question and both arms' answers, each sentence with its verdict, "
                             "as JSON lines to PATH once the run has measured (the bench's own conversation, "
                             "synthetic)")
    parser.add_argument("--home-rung", type=int, default=None, metavar="R",
                        help="plan the onion arm under onion.yaml with home_rung set to R, in this process (the tool "
                             "room and result cap scaled to keep their value at each rung, local.yaml read as beside "
                             "the file)")
    parser.add_argument("--states", default=None, metavar="DIR",
                        help="keep, at each probe turn, the onion arm's state before its window is planned and the "
                             "window it sent, under DIR, a new directory (plaintext: the bench's own conversation, "
                             "synthetic), for scripts/drift_pairs.py")
    args = parser.parse_args(argv)
    if args.states is not None:
        # A new directory, made here and by this run alone: one run's states never mix with another's.
        target = Path(args.states)
        try:
            if not args.states:
                raise FileExistsError(args.states)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.mkdir()
        except OSError as exc:
            print(f"refused: {args.states!r} is not a new directory this run can make ({type(exc).__name__}): one "
                  "run's states never mix with another's", file=sys.stderr)
            return 2
    if resolve is None:
        # The package the run measures is this tree's, the rung's pin included.
        sys.path.insert(0, str(_REPO))
    try:
        with pinned_rung(args.home_rung):
            return _measured(args, resolve)
    except RungRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


def _measured(args, resolve):
    """The run ``main`` was asked for, its arguments parsed, under the rung it pinned."""
    if resolve is None:
        resolve = _registry_resolver()
    model = args.model or _default_model()
    backend = resolve(model)
    if backend is None:
        print(f"no backend resolves {model!r}: nothing measured", file=sys.stderr)
        return 2
    from opti_oignon.agent import untrusted_context
    from opti_oignon.memory import librarian

    config = replace(librarian.load_config(), enabled=True, persist_path="", counters_path="")
    if args.librarian_model:
        config = replace(config, model=args.librarian_model)
    summarize = librarian.registry_summarizer(config, resolve=resolve)
    if summarize is None:
        print(f"no backend resolves the librarian's {config.model!r}: nothing measured", file=sys.stderr)
        return 2
    checks = (
        ("answering model", model, backend, "--model"),
        ("librarian's model", config.model, resolve(config.model), "--librarian-model"),
    )
    for role, name, serving, option in checks:
        problem = _unserved(name, serving) if serving is not None else None
        if problem is not None:
            print(f"the {role} {problem}: nothing measured; choose another with {option}", file=sys.stderr)
            return 2
    # The model's window as the executor reads it: the onion arm is planned within it, and the pair is tried at the
    # rung its turns are sent at.
    window = _model_window(model, backend)
    problem = _try_pair(backend, model, checks[1][2] or backend, config, _trial_ctx(window))
    if problem is not None:
        print(f"{problem}: nothing measured, no turn asked", file=sys.stderr)
        return 2

    ask = asker(backend, model)

    from opti_oignon.memory import peels, probes

    started = time.perf_counter()
    # The journal wraps each data part under its own source, as the chat path hands it the envelope, and counts
    # them with the model's counter, as the executor hands it.
    onion_answer = onion_arm(ask, librarian=librarian, config=config, summarize=None, resolve=resolve,
                             wrap=untrusted_context.wrap, mode=args.mode, model=model, model_window=window,
                             count=_model_counter(model), states=args.states,
                             record={number for number, turn in enumerate(TURNS, 1) if turn.probe})
    # What the gate weighed at each decision of the run, read where the queue asks it, and put back as it was.
    figures, unread, decide = [], [], peels.decide
    peels.decide = recording_gate(peels, probes, figures, unread)
    try:
        plain_answers = run_arm(TURNS, plain_arm(ask, history_tokens=args.history_tokens))
        onion_answers = run_arm(TURNS, onion_answer)
    except Exception as exc:  # noqa: BLE001 - a half-run is not a measurement
        print(f"the run broke before its end ({exc!r}): nothing measured", file=sys.stderr)
        return 2
    finally:
        peels.decide = decide
    said = {"plain": [], "onion": []}
    plain = reading(TURNS, plain_answers, arm="plain", source="measured", verdicts=said["plain"])
    onion = reading(TURNS, onion_answers, arm="onion", source="measured", verdicts=said["onion"])
    report = {
        "source": "measured",
        "mode": args.mode,
        "model": model,
        "librarian_model": config.model,
        "generator": probes.GENERATOR_VERSION,
        "lexicon": peels.load_gate().lexicon.fingerprint,
        "taken_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "history_tokens": args.history_tokens,
        "home_rung": args.home_rung,
        "states": args.states,
        "turns": len(TURNS),
        "seconds": round(time.perf_counter() - started, 1),
        # What the onion did, so a rate reads with the engagement behind it.
        "engagement": {
            "counts": librarian.counters(),
            "block_tokens": _spread(onion_answer.block_tokens),
            "memory_tokens": _spread(onion_answer.memory_tokens),
            "flesh_tokens": _spread(onion_answer.flesh_tokens),
        },
        # What the gate weighed for each summary the librarian handed it: numbers and motive names only.
        "gate": gate_report(figures, peels.load_gate().max_length_ratio, unread),
        **compare(onion, plain),
    }
    if args.answers:
        try:
            write_answers(args.answers, TURNS, {"plain": (plain_answers, said["plain"]),
                                                "onion": (onion_answers, said["onion"])})
        except (OSError, TypeError, ValueError) as exc:
            print(f"the answers could not be written to {args.answers} ({exc!r}); the report stands", file=sys.stderr)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
