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
lexicon, so that two readings are compared only under the same.

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
import importlib.util
import inspect
import json
import math
import re
import sys
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


def reading(turns, answers, *, arm, source, templates=None):
    """What each answer contradicts, and agrees with, among the facts holding at its turn."""
    if len(answers) > len(turns):
        raise ValueError(f"{len(answers)} answers for {len(turns)} turns")
    drift = templates or _templates()
    sentences = decided = agreements = 0
    at = []
    for index, answer in enumerate(answers):
        held = held_facts(turns, index + 1)
        held_claims = {_claim_key(c) for c in (drift._parse(s) for _id, s in held) if c is not None}
        for sentence in (s.strip() for s in _SENTENCE_SPLIT.split(str(answer or "")) if s.strip()):
            sentences += 1
            claim = drift._parse(sentence)
            if claim is None:
                continue
            decided += 1
            if _claim_key(claim) in held_claims:
                agreements += 1
            for found in drift.find_contradictions(held, sentence):
                at.append({"turn": index + 1, "sentence": sentence, "fact": found.fact_id, "template": found.template})
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


def onion_arm(ask, *, librarian, config, summarize, wrap, reask=None, mode="fast", resolve=None,
              system=SYSTEM_PROMPT, conversation_id="drift-ab", gate=None, budget=None, model="bench",
              model_window=None, count=estimate_tokens):
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
    launched.
    """
    if getattr(config, "persist_path", ""):
        raise ValueError("the drift A/B runs the librarian in process: a persistence path would write the user's store")
    if getattr(config, "counters_path", ""):
        raise ValueError("the drift A/B keeps its counts in the process: a counters path would add them to the user's")
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {', '.join(MODES)}")
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


def compare(onion, plain):
    """Both readings and the difference of their rates, onion minus plain."""
    a, b = onion["rate_per_1000_turns"], plain["rate_per_1000_turns"]
    return {"onion": onion, "plain": plain, "delta_rate_per_1000_turns": None if a is None or b is None else round(a - b, 3)}


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


def asker(backend, model):
    """How both arms ask ``model`` on ``backend``: the same sampling and the same room for the answer. A window the
    journal composed goes with the context it was composed for, as on the chat path: an engine left at its own
    default context would cut it."""

    def ask(messages, window=None):
        options = {"temperature": 0.0, "seed": 0, "num_predict": 128}
        if window is not None:
            options["num_ctx"] = int(window.num_ctx)
        response = backend.generate(model=model, messages=messages, options=options, think=False)
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
    args = parser.parse_args(argv)
    if resolve is None:
        sys.path.insert(0, str(_REPO))
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
                             count=_model_counter(model))
    try:
        plain_answers = run_arm(TURNS, plain_arm(ask, history_tokens=args.history_tokens))
        onion_answers = run_arm(TURNS, onion_answer)
    except Exception as exc:  # noqa: BLE001 - a half-run is not a measurement
        print(f"the run broke before its end ({exc!r}): nothing measured", file=sys.stderr)
        return 2
    plain = reading(TURNS, plain_answers, arm="plain", source="measured")
    onion = reading(TURNS, onion_answers, arm="onion", source="measured")
    report = {
        "source": "measured",
        "mode": args.mode,
        "model": model,
        "librarian_model": config.model,
        "generator": probes.GENERATOR_VERSION,
        "lexicon": peels.load_gate().lexicon.fingerprint,
        "taken_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "history_tokens": args.history_tokens,
        "turns": len(TURNS),
        "seconds": round(time.perf_counter() - started, 1),
        # What the onion did, so a rate reads with the engagement behind it.
        "engagement": {
            "counts": librarian.counters(),
            "block_tokens": _spread(onion_answer.block_tokens),
            "memory_tokens": _spread(onion_answer.memory_tokens),
            "flesh_tokens": _spread(onion_answer.flesh_tokens),
        },
        **compare(onion, plain),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
