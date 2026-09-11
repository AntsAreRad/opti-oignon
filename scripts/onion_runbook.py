#!/usr/bin/env python3
"""Host runbook for the onion memory: the measurements only a real model can give.

Everything the contracts prove runs over injected summarisers and says so.
This script asks the real one -- the librarian's model through the inference
registry, released after every call -- over the two corpora the baseline
was taken on, and prints what it finds with ``source: measured``, the model
name and the time. It refuses to print a number it did not measure: with no
backend it exits non-zero and prints nothing that looks like a result.

Run on the host, never in CI::

    python3 scripts/onion_runbook.py                # the measurements
    python3 scripts/onion_runbook.py --migrate-ledger  # memory_facts -> memory_ledger

What is measured:

* gate acceptance at the configured thresholds, and at every threshold of
  a sweep, so the arbitration of the thresholds has a table;
* compression fidelity, by resampling every accepted peel against its
  Cellar spans;
* the effective-context multiplier of the accepted peels;
* wall-clock seconds per summariser call, as a latency reading.

The migration reads the canonical facts table through its own store and
writes the ledger table beside it; both paths resolve under the data
directory, which this script never prints.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _corpus_turns(corpus):
    return [
        {"turn_id": f"t{i + 1:04d}", "role": t["role"], "text": t["content"]}
        for i, t in enumerate(corpus.turns)
    ]


def _summariser():
    from opti_oignon.inference_backend import init_backends_from_config
    from opti_oignon.memory.librarian import load_config, registry_summarizer

    try:
        init_backends_from_config()
    except Exception as exc:  # noqa: BLE001 - reported below as absence
        print(f"backend configuration not applied: {exc!r}", file=sys.stderr)
    config = load_config()
    summarize = registry_summarizer(config)
    if summarize is None:
        print(f"no backend resolves {config.model!r}: nothing measured", file=sys.stderr)
        return None, config
    return summarize, config


def _measure(summarize, config):
    from opti_oignon.memory.baseline import CORPORA
    from opti_oignon.memory.peels import (
        Gate,
        PeelTree,
        context_multiplier,
        evict_gated,
        fidelity,
        load_gate,
    )
    from opti_oignon.memory.receipts import Cellar, Flesh, ReceiptLedger

    gate = load_gate()
    timings = []

    def timed(turns):
        start = time.perf_counter()
        try:
            return summarize(turns)
        finally:
            timings.append(time.perf_counter() - start)

    report = {
        "source": "measured",
        "model": config.model,
        "keep_alive": config.keep_alive,
        "taken_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gate": {"decision_threshold": gate.decision_threshold, "episodic_threshold": gate.episodic_threshold, "span_turns": gate.span_turns},
        "corpora": {},
    }
    for name, corpus in CORPORA.items():
        turns = _corpus_turns(corpus)
        entry = {"turns": len(turns), "at_configured_gate": {}, "sweep": {}}
        cellar, ledger, tree = Cellar(), ReceiptLedger(), PeelTree()
        flesh = Flesh(turns)
        accepted = refused = 0
        while len(flesh.turns()) >= gate.span_turns:
            outcome = evict_gated(flesh=flesh, cellar=cellar, ledger=ledger, tree=tree, gate=gate, summarize=timed)
            if outcome.evicted:
                accepted += 1
            else:
                refused += 1
                break
        entry["at_configured_gate"] = {
            "spans_accepted": accepted, "spans_refused": refused,
            "fidelity": fidelity(tree, cellar) | {"source": "measured"},
            "context_multiplier": context_multiplier(tree, cellar) | {"source": "measured"},
        }
        # The sweep re-summarises each span once and judges the same text at
        # every threshold, so the table compares thresholds, not samples.
        from opti_oignon.memory.peels import judge
        from opti_oignon.memory.probes import generate_probes

        spans = [turns[i:i + gate.span_turns] for i in range(0, len(turns), gate.span_turns)]
        judged = []
        for span in spans:
            text = timed(span)
            judged.append((generate_probes(span), text))
        for threshold in SWEEP:
            g = Gate(decision_threshold=max(threshold, gate.decision_threshold), episodic_threshold=threshold, span_turns=gate.span_turns)
            passed = sum(1 for probes, text in judged if judge(probes, text, g).accepted)
            entry["sweep"][str(threshold)] = {"spans": len(judged), "accepted": passed}
        report["corpora"][name] = entry
    if timings:
        report["summariser_seconds"] = {
            "calls": len(timings), "mean": round(sum(timings) / len(timings), 3),
            "max": round(max(timings), 3),
        }
    return report


def _migrate():
    from opti_oignon.memory.canonical_store import get_canonical_store
    from opti_oignon.memory.ledger_store import LedgerStore, migrate_facts

    store = get_canonical_store()
    rows = [r.__dict__ if hasattr(r, "__dict__") else r for r in store.list(active_only=False)]
    facts = migrate_facts(rows)
    ledger = LedgerStore(Path(store.db_path).with_name("memory_ledger.db"))
    held = skipped = 0
    for fact in facts:
        try:
            ledger.add(fact)
            held += 1
        except ValueError:
            skipped += 1  # already held: the migration is re-runnable
    print(json.dumps({"source": "measured", "rows": len(rows), "held": held, "already_held": skipped}, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--migrate-ledger", action="store_true", help="migrate memory_facts into the ledger table")
    args = parser.parse_args(argv)
    if args.migrate_ledger:
        return _migrate()
    summarize, config = _summariser()
    if summarize is None:
        return 2
    print(json.dumps(_measure(summarize, config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
