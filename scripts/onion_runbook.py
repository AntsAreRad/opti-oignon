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
    python3 scripts/onion_runbook.py --probe-recall SET  # the probe recall on a labelled set

What is measured:

* gate acceptance at the configured thresholds, and at every threshold of
  a sweep, so the arbitration of the thresholds has a table;
* compression fidelity, by resampling every accepted peel against its
  Cellar spans;
* the effective-context multiplier of the accepted peels;
* wall-clock seconds per summariser call, as a latency reading;
* the share of the probes' draws and scores the native core served, and
  why the reference served the rest.

Each rate stands beside the probe recall the gate states for its probes,
or None and why: a rate counts the probes a summary answers, never the
facts no probe was drawn for. The second face reports, span by span, the
share of the facts the span holds that its probes asked for, against the
floor, and the facts left unasked, counted kind by kind and never written
out. ``--probe-recall`` reads the probe recall on a
labelled set of real conversations, kept off the repository: it needs no
model, and it prints counts only, never a writing or a turn of the set.

The migration reads the canonical facts table through its own store and
writes the ledger table beside it; both paths resolve under the data
directory, which this script never prints.
"""

import argparse
import json
import math
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


def _probe_recall_entry(gate):
    """The probe recall that holds for the gate's probes, as the report states it beside a rate, or None and why."""
    from opti_oignon.memory.peels import stated_probe_recall

    held, note = stated_probe_recall(gate.probe_recall, gate.lexicon)
    return {"probe_recall": None if held is None else held.entry(), "probe_recall_note": note}


def _gate_entry(gate):
    """What the report says of the gate it measured with."""
    return {"decision_threshold": gate.decision_threshold, "episodic_threshold": gate.episodic_threshold,
            "code_threshold": gate.code_threshold, "span_turns": gate.span_turns,
            "lexicon": gate.lexicon.fingerprint, "reporters": sorted(gate.reporters),
            "max_novelty": gate.max_novelty, "max_length_ratio": gate.max_length_ratio,
            "probe_floor": gate.probe_floor, **_probe_recall_entry(gate)}


def _at_gate(accepted, refused, tree, cellar, gate):
    """The rates at the configured gate, each beside the recall that holds for its probes."""
    from opti_oignon.memory.peels import context_multiplier, fidelity

    return {
        "spans_accepted": accepted, "spans_refused": refused, **_probe_recall_entry(gate),
        "fidelity": fidelity(tree, cellar, gate.lexicon, gate.probe_recall) | {"source": "measured"},
        "context_multiplier": context_multiplier(tree, cellar) | {"source": "measured"},
    }


def _native_entry():
    """What the native core served of the probes' work in this run, call by call, and why the reference served the rest."""
    from opti_oignon.memory.probes import native_share

    return native_share() | {"source": "measured"}


def _sweep_row(spans, accepted, gate):
    """One threshold of the sweep: the spans judged, those accepted, and the recall beside them."""
    return {"spans": spans, "accepted": accepted, **_probe_recall_entry(gate)}


def _probe_recall_report(path):
    """The probe recall of the generator on the labelled set at ``path``: counts only, never a text of the set."""
    from opti_oignon.memory.peels import load_gate
    from opti_oignon.memory.probes import load_labelled, measure_recall

    gate = load_gate()
    spans = load_labelled(path)
    figure = measure_recall(spans, gate.lexicon, source="labelled").probe_recall
    missed = sum(size - recalled for _name, recalled, size in figure.classes)
    entry = figure.entry()
    return {
        "source": "labelled", "taken_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": entry["generator"], "lexicon": entry["lexicon"], "spans": len(spans),
        "facts": sum(size for _name, _recalled, size in figure.classes), "missed": missed,
        "classes": entry["classes"], "stated": _probe_recall_entry(gate),
    }


def _spread(figures, bound, under=False):
    """Where a bound's figures fall: how many were measured, how many not, how many are over the bound -- or under
    it, for a floor -- and the nearest-rank minimum, median, ninetieth percentile and maximum, to four places."""
    known = sorted(f for f in figures if f is not None)
    entry = {"measured": len(known), "unmeasured": len(figures) - len(known)}
    if under:
        entry["under"] = sum(1 for f in known if bound is not None and f < bound)
    else:
        entry["over"] = sum(1 for f in known if bound is not None and f > bound)
    if known:
        def rank(q):
            return round(known[math.ceil(q * len(known)) - 1], 4)
        entry.update(min=round(known[0], 4), p50=rank(0.5), p90=rank(0.9), max=round(known[-1], 4))
    return entry


def _second_face(judged, gate):
    """What the second face refuses, where its bounds' figures fall, and what share of each span's facts its probes
    asked for against the floor, with the facts no probe asked for kind by kind, each span judged once at the gate."""
    from opti_oignon.memory.peels import decide

    decisions = [decide(span, probes, text, gate) for span, probes, text in judged]
    claims, unasked = {}, {}
    for decision in decisions:
        for kind, _what, _turn in decision.unsupported:
            claims[kind] = claims.get(kind, 0) + 1
        for kind, _what, _turn in decision.unasked:
            unasked[kind] = unasked.get(kind, 0) + 1
    return {"spans": len(judged), "spans_refused": sum(1 for d in decisions if d.unsupported),
            "claims_refused": claims,
            "novelty": _spread([d.novelty for d in decisions], gate.max_novelty),
            "length_ratio": _spread([d.length_ratio for d in decisions], gate.max_length_ratio),
            "probe_coverage": _spread([d.probe_coverage for d in decisions], gate.probe_floor, under=True),
            "unasked": unasked,
            "source": "measured"}


def _measure(summarize, config):
    from dataclasses import replace

    from opti_oignon.memory.baseline import CORPORA
    from opti_oignon.memory.peels import PeelTree, evict_gated, load_gate
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
        "gate": _gate_entry(gate),
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
        entry["at_configured_gate"] = _at_gate(accepted, refused, tree, cellar, gate)
        # The sweep re-summarises each span once and judges the same text at
        # every threshold, with both faces, so the table compares thresholds,
        # not samples. What the second face refuses does not move with the
        # thresholds, nor do the figures of its bounds: they are reported
        # once, claims kind by kind and each bound by the spread of its
        # figures, those its calibration is owed on.
        from opti_oignon.memory.peels import decide
        from opti_oignon.memory.probes import generate_probes

        spans = [turns[i:i + gate.span_turns] for i in range(0, len(turns), gate.span_turns)]
        judged = []
        for span in spans:
            text = timed(span)
            judged.append((span, generate_probes(span, gate.lexicon), text))
        entry["second_face"] = _second_face(judged, gate)
        for threshold in SWEEP:
            g = replace(gate, decision_threshold=max(threshold, gate.decision_threshold), episodic_threshold=threshold)
            passed = sum(1 for span, probes, text in judged if decide(span, probes, text, g).accepted)
            entry["sweep"][str(threshold)] = _sweep_row(len(judged), passed, gate)
        report["corpora"][name] = entry
    report["native"] = _native_entry()
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
    parser.add_argument("--probe-recall", metavar="SET", help="read the probe recall on a labelled set: counts only")
    args = parser.parse_args(argv)
    if args.migrate_ledger:
        return _migrate()
    if args.probe_recall:
        from opti_oignon.memory.probes import LabelledSetError

        try:
            print(json.dumps(_probe_recall_report(args.probe_recall), indent=2))
        except LabelledSetError as exc:
            print(f"labelled set refused: {exc}", file=sys.stderr)
            return 2
        return 0
    summarize, config = _summariser()
    if summarize is None:
        return 2
    print(json.dumps(_measure(summarize, config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
