#!/usr/bin/env python3
"""Paired windows: a change of the onion's composition measured on the same states.

A drift A/B run (``scripts/drift_ab.py``) with ``--states DIR`` keeps, at
each probe turn, the onion arm's state before its window was planned, the
turn, the conversation as stored, the home rung and system prompt it was
planned under, and the window it sent. A run's later states follow its own
answers -- a composition that changes one answer changes the conversation,
and every state after it -- so two trees measured end to end differ by
their course as much as by their composition, and one run's verdict on a
probe proves nothing. ``compose`` loads each kept state again, from a copy
of its store, and plans and composes its probe's window under this tree's
package, at the home rung it was kept under, as the bench's onion arm plans
it; unless ``--no-ask``, it asks the answering model with the bench's
sampling and reads the answer with this tree's reading. Two trees that
compose the same states differ in their windows by their composition
alone; each reads the answers with its own bench, and a kept turn that is
not this tree's probe of that number is refused by name. Under the tree
that kept them a state composes again to the window it sent, unless a
burst held the conversation when the run planned it (``held``): no burst
runs here.

``compare`` reads two files ``compose`` wrote over the same states, probe
by probe: whether the windows differ, and, when both were asked, the
probes each answers right and what the second gains and loses.

Run on the host, never in CI::

    python3 scripts/drift_ab.py --mode fast --home-rung 4096 --states /tmp/states   # once
    python3 scripts/drift_pairs.py compose /tmp/states --out before.jsonl  # in one tree
    python3 scripts/drift_pairs.py compose /tmp/states --out after.jsonl   # in the other
    python3 scripts/drift_pairs.py compare before.jsonl after.jsonl

``--model`` names the answering model (the recording run's by default);
``--rung``, when given, must be the home rung the states were kept under,
and names it for states kept before they named it;
``--no-ask`` composes the windows only, to see whether two trees differ at
all. A state kept under a newer schema of the onion's store than this
tree's is refused by name, and one under an older schema is brought up in
the copy, as the store brings up its own; keep states with the older of
two trees. A refusal writes nothing and exits 2.
"""

import argparse
import importlib.util
import json
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
VERDICTS = ("agrees", "contradicts", "decided", "undecided")


class Refused(ValueError):
    """What ``compose`` and ``compare`` refuse, by name."""


def _drift_ab():
    """The bench's own script, from this tree: its conversation, sampling, envelope, counters, rung pin and
    reading."""
    loaded = sys.modules.get("drift_ab")
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location("drift_ab", _REPO / "scripts" / "drift_ab.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["drift_ab"] = module
    spec.loader.exec_module(module)
    return module


def recorded(directory):
    """The probe turns kept under ``directory``, in order: each one's state and turn files present."""
    directory = Path(directory)
    return [int(meta.stem[1:]) for meta in sorted(directory.glob("t[0-9][0-9][0-9].json"))
            if meta.with_suffix(".db").is_file()]


def compose_recorded(librarian, onion_store, directory, turn, *, wrap, counter, system, gate=None):
    """``(meta, window)``: the window the onion arm plans for the probe kept as ``turn`` under ``directory``,
    composed again under ``librarian``. The state is loaded from a copy of its store -- the record is never
    opened -- into the conversation's new state, the librarian in process with persistence and counters off;
    the turn is planned as the arm plans it: typed, counted with ``counter(model)`` for the model it was planned
    for, within the window the run planned it in, from the conversation as stored before it, under ``system``
    and ``gate`` (the shipped one when None, as the run's)."""
    directory = Path(directory)
    meta = json.loads((directory / f"t{turn:03d}.json").read_text(encoding="utf-8"))
    work = Path(tempfile.mkdtemp(prefix="oo-drift-pairs-"))
    try:
        copy = work / "state.db"
        shutil.copyfile(directory / f"t{turn:03d}.db", copy)
        librarian.reset_librarian()
        config = replace(librarian.load_config(), enabled=True, persist_path="", counters_path="")
        state = librarian.state_for(meta["cid"], config)
        if onion_store.OnionStore(copy, connect=sqlite3.connect, require_encryption=False).load(
                meta["cid"], state) is None:
            raise Refused(f"the store kept for turn {turn} does not know conversation {meta['cid']!r}")
        plan = librarian.plan_window(
            meta["cid"], model=meta["model"], count=counter(meta["model"]), system=system,
            turn={"role": "user", "content": meta["question"], "origin": "typed", "segments": []}, wrap=wrap,
            model_window=meta["model_window"], messages=meta["history"], config=config, gate=gate,
        )
        return meta, librarian.compose_window(plan)
    finally:
        librarian.reset_librarian()
        shutil.rmtree(work, ignore_errors=True)


def _kept_rung(directory, turns, rung):
    """The one home rung the kept ``turns`` were planned under, ``rung`` when given and the same; ``rung`` for
    states that name none, kept before they did. Refused when the turns name several, or another than ``rung``, or
    none and no ``rung`` is given."""
    kept = {json.loads(Path(directory, f"t{turn:03d}.json").read_text(encoding="utf-8")).get("home_rung")
            for turn in turns}
    if kept == {None}:
        if rung is None:
            raise Refused(f"the states under {directory} name no home rung: name it with --rung")
        return rung
    if len(kept) != 1 or None in kept:
        raise Refused(f"the states under {directory} name no single home rung ({sorted(map(str, kept))})")
    home = kept.pop()
    if rung is not None and rung != home:
        raise Refused(f"the states were kept under home_rung {home}, not {rung}")
    return home


def _windowed(window):
    """What a window is compared by: its messages, the rung it was composed under and the context it is sent
    with."""
    messages = [{"role": role, "content": content} for role, content, _label in window.messages]
    return messages, window.rung, int(window.num_ctx)


def compose(directory, out, *, rung=None, model=None, ask=True, turns=None, resolve=None):
    """Every probe kept under ``directory`` (``turns`` alone when given) composed again under this tree at the home
    rung it was kept under and, with ``ask``, asked and read; one JSON line per probe to ``out``, written whole.
    0; 2 refused by name, nothing written."""
    if resolve is None and str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    ab = _drift_ab()
    from opti_oignon.agent import untrusted_context
    from opti_oignon.memory import librarian, onion_store

    # The tree whose package composed: what a file is read against.
    tree = str(Path(librarian.__file__).resolve().parents[2])
    partial = Path(f"{out}.partial")
    try:
        wanted = [turn for turn in recorded(directory) if turns is None or turn in turns]
        if not wanted:
            raise Refused(f"no probe state kept under {directory}")
        home = _kept_rung(directory, wanted, rung)
        asked = {}
        with ab.pinned_rung(home), open(partial, "w", encoding="utf-8") as handle:
            for turn in wanted:
                meta = json.loads(Path(directory, f"t{turn:03d}.json").read_text(encoding="utf-8"))
                probe = ab.TURNS[turn - 1] if 0 < turn <= len(ab.TURNS) else None
                if probe is None or not probe.probe or probe.text != meta["question"]:
                    raise Refused(f"turn {turn} kept is not this tree's probe {turn}: its reading would be another's")
                meta, window = compose_recorded(librarian, onion_store, directory, turn,
                                                wrap=ab._envelope(untrusted_context.wrap), counter=ab._model_counter,
                                                system=meta.get("system", ab.SYSTEM_PROMPT))
                row = {"turn": turn, "question": meta["question"], "rung": home, "tree": tree,
                       "held": meta.get("held"),
                       "refusal": None if window.refusal is None else str(window.refusal.motive)}
                if window.refusal is None:
                    messages, window_rung, num_ctx = _windowed(window)
                    row.update(messages=messages, layers=dict(window.layers), cut=[list(c) for c in window.cut],
                               window_rung=window_rung, num_ctx=num_ctx)
                    sent = Path(directory, f"t{turn:03d}.window.json")
                    if sent.is_file():
                        kept = json.loads(sent.read_text(encoding="utf-8"))
                        row.update(same_as_recorded=(kept["messages"], kept["rung"], kept["num_ctx"]) == (
                            messages, window_rung, num_ctx), recorded_reply=kept["reply"])
                    if ask:
                        name = model or meta["model"]
                        if name not in asked:
                            backend = (resolve or ab._registry_resolver())(name)
                            if backend is None:
                                raise Refused(f"no backend resolves {name!r}: nothing asked")
                            asked[name] = ab.asker(backend, name)
                        reply = str(asked[name](messages, window=window))
                        read = ab.reading(ab.TURNS, [""] * (turn - 1) + [reply], arm="onion", source="paired")
                        row.update(model=name, reply=reply,
                                   verdict=next((kind for kind in VERDICTS if read["probes"][kind]), None))
                handle.write(json.dumps(row) + "\n")
        partial.replace(out)
    except (Refused, ab.RungRefused, OSError, ValueError, KeyError) as exc:
        partial.unlink(missing_ok=True)
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


def compare(before, after):
    """Two files ``compose`` wrote over the same states, probe by probe: ``{"probes", "windows_differ",
    "refused", "asked", "before", "after", "gains", "losses"}`` -- the probes both composed; those whose windows
    differ (messages, rung or context); those refused, with their motives; and, when every probe of both was
    asked, how many each answers right and what the second gains and loses (None otherwise). Refused when a probe
    of one is not the other's: another question or another home rung."""
    def rows(path):
        return {row["turn"]: row for row in (json.loads(line) for line in Path(path).read_text(
            encoding="utf-8").splitlines() if line.strip())}

    first, second = rows(before), rows(after)
    shared = sorted(set(first) & set(second))
    for turn in shared:
        if (first[turn]["question"], first[turn]["rung"]) != (second[turn]["question"], second[turn]["rung"]):
            raise Refused(f"probe {turn} is not the same in both files: another question or another home rung")

    def window(row):
        return row.get("refusal"), row.get("messages"), row.get("window_rung"), row.get("num_ctx")

    asked = bool(shared) and all("verdict" in first[t] and "verdict" in second[t] for t in shared)
    right = {name: [t for t in shared if rows_[t].get("verdict") == "agrees"]
             for name, rows_ in (("before", first), ("after", second))}
    return {
        "probes": shared,
        "windows_differ": [t for t in shared if window(first[t]) != window(second[t])],
        "refused": {name: {t: rows_[t]["refusal"] for t in shared if rows_[t].get("refusal")}
                    for name, rows_ in (("before", first), ("after", second))},
        "asked": asked,
        "before": len(right["before"]) if asked else None,
        "after": len(right["after"]) if asked else None,
        "gains": [t for t in right["after"] if t not in right["before"]] if asked else None,
        "losses": [t for t in right["before"] if t not in right["after"]] if asked else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    composing = sub.add_parser("compose", help="compose the kept states' windows again under this tree")
    composing.add_argument("states", help="the directory a drift A/B run kept its states in (--states)")
    composing.add_argument("--out", required=True, help="the JSON lines written, one per probe")
    composing.add_argument("--rung", type=int, default=None,
                           help="the home rung the states were kept under (read from them when left out)")
    composing.add_argument("--model", default=None, help="the answering model (default: the run's)")
    composing.add_argument("--no-ask", action="store_true", help="compose the windows only")
    composing.add_argument("--turns", type=int, nargs="*", default=None, help="only these probe turns")
    comparing = sub.add_parser("compare", help="read two composed files over the same states")
    comparing.add_argument("before")
    comparing.add_argument("after")
    args = parser.parse_args(argv)
    if args.command == "compare":
        try:
            print(json.dumps(compare(args.before, args.after), indent=2))
        except (Refused, OSError, ValueError, KeyError) as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        return 0
    return compose(args.states, args.out, rung=args.rung, model=args.model, ask=not args.no_ask, turns=args.turns)


if __name__ == "__main__":
    raise SystemExit(main())
