#!/usr/bin/env python3
"""Host A/B: drift with the onion memory against drift without it.

The design measures drift as contradictions per 1000 turns against a
no-onion baseline, and only a real model can give the answers. So the
number comes from the host, and everything around the model is here, held
by contracts: a scripted conversation whose facts change on known turns,
the rule that decides what an answer contradicts, the two ways of building
the model's context, and the driver that runs both over the same turns.

The two arms differ the way the chat path does. With the onion switched
on, the executor adds one thing to the prompt: the onion's memory block --
Core, receipts digest, the Peels selected for the turn -- wrapped as
untrusted data after the system prompt. So both arms send the same system
prompt and the same history window to the same model, and the onion arm
adds its block. The librarian runs in this process with persistence off
and curates after every turn, not on a thread, so the reading never
depends on timing; nothing reads or writes the data directory, and none of
the user's memories enters either arm.

A sentence the deterministic templates cannot parse is counted undecided,
never as agreement: the host's model judge is the remedy, and the report
says how much was left to it. The templates read English, and so does the
conversation.

Run on the host, never in CI::

    python3 scripts/drift_ab.py                      # both arms, one JSON report
    python3 scripts/drift_ab.py --model llama3:8b    # another answering model
    python3 scripts/drift_ab.py --history-tokens 2048

Without a backend for the answering model or for the librarian's, it
prints nothing that looks like a result and exits 2.
"""

import argparse
import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

checkpoint_before_apply = True

_REPO = Path(__file__).resolve().parent.parent
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# The history window both arms share, in estimated tokens. Small on purpose:
# the conversation has to outgrow it for the onion to have anything to do.
HISTORY_TOKENS = 1024
MAX_CURATION_STEPS = 16

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


def onion_arm(ask, *, librarian, config, summarize, wrap, history_tokens=HISTORY_TOKENS, system=SYSTEM_PROMPT,
              conversation_id="drift-ab", gate=None, budget=None):
    """The chat with the onion: the same window, plus the onion's block after the system prompt.

    The librarian runs in this process and never writes a store: a
    configuration with a persistence path is refused, so a measurement can
    never touch the user's onion.
    """
    if getattr(config, "persist_path", ""):
        raise ValueError("the drift A/B runs the librarian in process: a persistence path would write the user's store")
    history = []

    def answer(text):
        block = librarian.memory_block(conversation_id, text, budget=budget, config=config)
        head = system + "\n\n" + wrap(block) if block else system
        messages = [{"role": "system", "content": head}, *history_window(history, history_tokens),
                    {"role": "user", "content": text}]
        reply = str(ask(messages))
        history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": reply}])
        state = librarian.state_for(conversation_id, config)
        state.mirror(history)
        for _ in range(MAX_CURATION_STEPS):
            if not librarian.curate(state, summarize, gate=gate, budget=budget).evicted:
                break
        return reply

    return answer


def run_arm(turns, answer):
    """Every turn through one arm, in order; the answers."""
    return [answer(turn.text) for turn in turns]


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


def _default_model():
    from opti_oignon.config import config

    return config.get_model("general")


def main(argv=None, *, resolve=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="the answering model (default: the general route's)")
    parser.add_argument("--history-tokens", type=int, default=HISTORY_TOKENS, help="the history window both arms share")
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

    config = replace(librarian.load_config(), enabled=True, persist_path="")
    summarize = librarian.registry_summarizer(config, resolve=resolve)
    if summarize is None:
        print(f"no backend resolves the librarian's {config.model!r}: nothing measured", file=sys.stderr)
        return 2

    def ask(messages):
        response = backend.generate(model=model, messages=messages, options={"temperature": 0.0, "seed": 0, "num_predict": 128})
        return str(getattr(response, "content", "") or "")

    def wrap(block):
        return untrusted_context.wrap(block, source=untrusted_context.SOURCE_MEMORY)

    started = time.perf_counter()
    plain = reading(TURNS, run_arm(TURNS, plain_arm(ask, history_tokens=args.history_tokens)), arm="plain", source="measured")
    onion_answers = run_arm(TURNS, onion_arm(ask, librarian=librarian, config=config, summarize=summarize, wrap=wrap,
                                             history_tokens=args.history_tokens))
    onion = reading(TURNS, onion_answers, arm="onion", source="measured")
    report = {
        "source": "measured",
        "model": model,
        "librarian_model": config.model,
        "taken_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "history_tokens": args.history_tokens,
        "turns": len(TURNS),
        "seconds": round(time.perf_counter() - started, 1),
        **compare(onion, plain),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
