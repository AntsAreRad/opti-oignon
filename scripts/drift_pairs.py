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
probes each answers right and what the second gains and loses -- and, when
both read margins, how each probe's margin moved.

A verdict is one greedy draw: a small model can flip it on a window that
gives it no more and no less to go on. With ``--margins K`` the model is
asked through Ollama's own API on this machine, each token of its reply
with its K likeliest alternatives, and each probe carries its margin: at
the reply's token where the statement the probe expects and its rival
part -- the stale statement it retired, or, with none, the likeliest other
token -- the log-probability of going the expected way less that of going
the rival's, in nats. A margin moves with the window where a verdict only
flips. Each probe also says the token its margin is read against, and
``compare`` which probes moved and against what: a margin lost to another
subject's value (" Oslo" for Bob) is not one lost to the copy of the
sentence form ("<").

Run on the host, never in CI::

    python3 scripts/drift_ab.py --mode fast --home-rung 4096 --states /tmp/states   # once
    python3 scripts/drift_pairs.py compose /tmp/states --out before.jsonl  # in one tree
    python3 scripts/drift_pairs.py compose /tmp/states --out after.jsonl   # in the other
    python3 scripts/drift_pairs.py compare before.jsonl after.jsonl

``--model`` names the answering model (the recording run's by default);
``--rung``, when given, must be the home rung the states were kept under,
and names it for states kept before they named it;
``--no-ask`` composes the windows only, to see whether two trees differ at
all; ``--margins K`` reads the margins (K from 1 to 20). A state kept under
a newer schema of the onion's store than this tree's is refused by name,
and one under an older schema is brought up in the copy, as the store
brings up its own; keep states with the older of two trees. A refusal
writes nothing and exits 2.
"""

import argparse
import importlib.util
import ipaddress
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
VERDICTS = ("agrees", "contradicts", "decided", "undecided")
# A margin that moved by at least this, in nats -- e times the odds -- moved, for ``compare``.
MOVED = 1.0
# The most alternatives Ollama reads at a token.
MOST_ALTERNATIVES = 20
# What a model's own file may set, held off when margins are read: a penalty weighs on a token the window holds
# near its end -- the value a window lays nearest the question -- after the log-probabilities are read; and a token
# a draft model proposed carries no log-probability.
AS_READ = {"presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0, "draft_num_predict": 0}
# A window of a 27B model loaded cold on the maintainer's machine is asked in about 12 s: one asked for fifteen
# minutes is wedged.
TIMEOUT_S = 900
# Straight to the address given: a proxy the environment names never sees a window.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


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


def probe_values(turns, turn):
    """``(expected, stale, prefix)`` for the probe at ``turn`` of ``turns``: the statement it expects, the one a
    later turn retired for it (None when none did), and the words its sentence form lays before the value."""
    probe = turns[turn - 1]
    stale = next((t.replaces for t in reversed(turns[:turn]) if t.fact == probe.expects and t.replaces), None)
    form = re.search(r"of the form '([^']*)'", probe.text)
    return probe.expects, stale, form.group(1).split("<", 1)[0] if form else ""


def margin(tokens, expected, rival=None, prefix="", reply=None):
    """``(margin, bound, why, token, took)``: how much likelier the reply went ``expected``'s way, in nats, read from
    ``tokens`` -- each ``{"token", "logprob", "top_logprobs"}`` as Ollama gives them, the token itself counted
    among its alternatives.

    An alternative goes on with a statement when the statement's rest
    starts with it, or when it runs past the statement's end by marks only,
    never by more of a word; one that goes on with both statements decides
    nothing. With a ``rival`` -- the stale statement the expected one
    retired -- the margin is read once, at the reply's first token to reach
    past the characters both statements share: the log-probability of the
    likeliest alternative that goes on with ``expected`` less that of the
    likeliest that goes on with ``rival``. With none, at each token of the
    value, from the first to reach past ``prefix`` (the words the sentence
    form lays before it) to the end of ``expected`` or the first token the
    reply took off it: the likeliest alternative that goes on with
    ``expected`` less the likeliest that does not, the least of them -- a
    value is held as firmly as its weakest token. ``bound`` is "exact", or
    "at least" when the rival's side is not among the alternatives read and
    "at most" when the expected's is not, the least likely read standing in;
    ``token`` is the token read, ``took`` the way the reply went there:
    "expected", "rival" or "other". None, with ``why``, when the tokens are
    not ``reply``'s (when given), there is no form, the reply leaves the
    form before the value, ends before it, or reads neither side."""
    return _read_margin(tokens, expected, rival, prefix, reply)[:5]


def margin_against(tokens, expected, rival=None, prefix="", reply=None):
    """The token ``margin`` reads its margin against, where it reads it: the likeliest alternative that goes on with
    ``rival`` and not with ``expected`` -- with one -- or that does not go on with ``expected`` -- with none, at the
    weakest token of the value: " Berlin" against " Oslo", or "<" where the reply would copy its sentence form; the
    first listed of two equally likely. None where that side is not among the alternatives read (a margin of at
    least), the least likely read standing in, or where no margin is read."""
    return _read_margin(tokens, expected, rival, prefix, reply)[5]


def _read_margin(tokens, expected, rival, prefix, reply):
    """``margin``'s reading, the token it is read against last."""
    texts = [str(token.get("token", "")) for token in tokens]
    if reply is not None and "".join(texts) != reply:
        whole = b"".join(bytes(token.get("bytes") or b"") for token in tokens).decode("utf-8", "replace")
        why = "a character split across tokens" if whole == reply else "the tokens read are not the reply's"
        return None, None, why, None, None, None
    if not rival and not prefix:
        return None, None, "no sentence form to read the value at", None, None, None
    part = len(os.path.commonprefix([expected, rival])) if rival else len(prefix)
    start = 0
    for index, text in enumerate(texts):
        if start + len(text) > part:
            break
        start += len(text)
    else:
        return None, None, "the reply ends before the value", None, None, None
    if "".join(texts)[:start] != expected[:start]:
        return None, None, "the reply leaves the form before the value", None, None, None

    def goes_on(text, rest):
        return rest.startswith(text) or (text.startswith(rest) and not any(c.isalnum() for c in text[len(rest):]))

    def read(at):
        found = [(str(c.get("token", "")), c.get("logprob")) for c in tokens[at].get("top_logprobs") or []]
        found.append((texts[at], tokens[at].get("logprob")))
        return [(text, float(value)) for text, value in found if text and isinstance(value, (int, float))]

    def step(at, ours, theirs):
        alternatives = read(at) if tokens[at].get("top_logprobs") else []
        if not alternatives:
            return "no alternative read at the value"
        mine = [v for text, v in alternatives
                if goes_on(text, ours) and not (theirs is not None and goes_on(text, theirs))]
        if theirs is not None:
            other = [(text, v) for text, v in alternatives if goes_on(text, theirs) and not goes_on(text, ours)]
        else:
            other = [(text, v) for text, v in alternatives if not goes_on(text, ours)]
        if not mine and not other:
            return "neither side among the alternatives read"
        floor = min(v for _text, v in alternatives)
        bound = "exact" if mine and other else "at most" if not mine else "at least"
        took = ("expected" if goes_on(texts[at], ours) else
                "rival" if theirs is not None and goes_on(texts[at], theirs) else "other")
        against = max(other, key=lambda pair: pair[1]) if other else (None, floor)
        return round((max(mine) if mine else floor) - against[1], 4), bound, texts[at], took, against[0]

    if rival:
        found = step(index, expected[start:], rival[start:])
        if isinstance(found, str):
            return None, None, found, None, None, None
        value, bound, token, took, against = found
        return value, bound, None, token, took, against
    least, at = None, index
    while at < len(texts) and start < len(expected):
        found = step(at, expected[start:], None)
        if isinstance(found, str):
            return None, None, found, None, None, None
        if least is None or found[0] < least[0]:
            least = found
        if found[3] != "expected":
            break
        start += len(texts[at])
        at += 1
    value, bound, token, took, against = least
    return value, bound, None, token, took, against


def _on_this_machine(endpoint):
    """True when ``endpoint`` addresses this machine, as the backends judge it: localhost, a loopback or an
    unspecified address; no name is resolved."""
    host = (urllib.parse.urlparse(str(endpoint or "")).hostname or "").strip()
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _post(url, body, timeout=TIMEOUT_S):
    """Ollama's answer to ``body`` posted at ``url``: straight to it, whatever proxy the environment names, within
    ``timeout`` seconds; a refusal, an unreachable server, a timeout or an answer that is no JSON object refused by
    name."""
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            answer = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        said = exc.read().decode("utf-8", "replace")[:300]
        raise Refused(f"Ollama refused the request ({exc.code}): {said}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise Refused(f"Ollama did not answer at {url}: {getattr(exc, 'reason', exc)}") from None
    if not isinstance(answer, dict):
        raise Refused(f"Ollama answered no JSON object at {url}")
    return answer


def logprobs_asker(backend, model, top, fetch=None):
    """How ``compose`` asks ``model`` when it reads margins: Ollama's chat API at the backend's endpoint, on this
    machine only, with the bench's sampling, the model's own penalties and draft held off (``AS_READ``) and the
    context the window was composed for, from a model loaded for the window alone (a keep-alive of 0), each token
    of the reply with its ``top`` likeliest alternatives; ``ask(messages, window=None)`` gives ``(reply, tokens)``.
    A backend that is not Ollama's, or whose endpoint is not on this machine, is refused. The backend tells
    whether the model is told not to think, as the bench's own requests are; the governor's thread and layer
    options are not sent. ``fetch(url, body)`` posts the request (``_post`` when None)."""
    endpoint = backend.endpoint() if callable(getattr(backend, "endpoint", None)) else None
    if getattr(backend, "name", None) != "ollama" or not _on_this_machine(endpoint):
        raise Refused(f"margins are read from Ollama's API on this machine: {model!r} is served by "
                      f"{getattr(backend, 'name', None)!r} at {endpoint!r}")
    think = backend._think_flag(model, False) if callable(getattr(backend, "_think_flag", None)) else None
    ab = _drift_ab()

    def ask(messages, window=None):
        options = dict(ab.SAMPLING, **AS_READ)
        if window is not None:
            options["num_ctx"] = int(window.num_ctx)
        body = {"model": model, "messages": messages, "stream": False, "logprobs": True, "top_logprobs": top,
                "keep_alive": 0, "options": options}
        if think is not None:
            body["think"] = think
        answer = (fetch or _post)(f"{str(endpoint).rstrip('/')}/api/chat", body)
        return str((answer.get("message") or {}).get("content", "")), list(answer.get("logprobs") or [])

    return ask


def _cold(backend):
    """Whether ``backend`` asks each window of a model loaded for it alone: Ollama's keep-alive of 0; another
    engine's prompt cache is its own."""
    return getattr(backend, "name", None) == "ollama"


def compose(directory, out, *, rung=None, model=None, ask=True, turns=None, resolve=None, margins=0, fetch=None):
    """Every probe kept under ``directory`` (``turns`` alone when given) composed again under this tree at the home
    rung it was kept under and, with ``ask``, asked and read; one JSON line per probe to ``out``, written whole.
    On Ollama each window is asked of a model loaded for it alone (``cold``), with the bench's sampling -- the
    bench's trial line asked first, so that a model left loaded before compose answers no window warm. With
    ``margins``, the number of alternatives read at each token (at most ``MOST_ALTERNATIVES``), the model is asked
    through ``logprobs_asker`` (``fetch`` its post), its own penalties and draft held off (``sampling`` "raw"), and
    each probe carries its ``margin`` (``margin``): the stale statement the probe's expected one retired as its
    rival, or every other token when none was, the token read and the way the reply went there, and the token it
    is read against (``margin_against``). 0; 2 refused by name, nothing written."""
    if resolve is None and str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    ab = _drift_ab()
    from opti_oignon.agent import untrusted_context
    from opti_oignon.memory import librarian, onion_store

    # The tree whose package composed: what a file is read against.
    tree = str(Path(librarian.__file__).resolve().parents[2])
    partial = Path(f"{out}.partial")
    try:
        if margins and not (ask and 0 < margins <= MOST_ALTERNATIVES):
            raise Refused(f"margins read 1 to {MOST_ALTERNATIVES} alternatives of an asked model, not {margins!r}")
        wanted = [turn for turn in recorded(directory) if turns is None or turn in turns]
        if not wanted:
            raise Refused(f"no probe state kept under {directory}")
        home = _kept_rung(directory, wanted, rung)
        asked, how = {}, {}
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
                            asked[name] = (logprobs_asker(backend, name, margins, fetch) if margins
                                           else ab.asker(backend, name, cold=_cold(backend)))
                            how[name] = (True, "raw") if margins else (_cold(backend), "bench")
                            if how[name][0]:
                                # A model left loaded answers its next window warm: the trial line unloads it first.
                                asked[name](ab._TRIAL)
                        if margins:
                            reply, tokens = asked[name](messages, window)
                        else:
                            reply = str(asked[name](messages, window=window))
                        read = ab.reading(ab.TURNS, [""] * (turn - 1) + [reply], arm="onion", source="paired")
                        row.update(model=name, reply=reply, cold=how[name][0], sampling=how[name][1],
                                   verdict=next((kind for kind in VERDICTS if read["probes"][kind]), None))
                        if margins:
                            expected, stale, prefix = probe_values(ab.TURNS, turn)
                            value, bound, why, token, took, against = _read_margin(tokens, expected, stale,
                                                                                   prefix, reply)
                            row.update(margin=value, margin_bound=bound, margin_rival=stale, margin_why=why,
                                       margin_token=token, margin_took=took, margin_against=against)
                handle.write(json.dumps(row) + "\n")
        partial.replace(out)
    except (Refused, ab.RungRefused, OSError, ValueError, KeyError) as exc:
        partial.unlink(missing_ok=True)
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


def _paired(before, after):
    """``(first, second, shared)``: the rows of two files ``compose`` wrote, by turn, and the probes both hold;
    refused when a probe of one is not the other's -- another question or another home rung -- or was asked
    otherwise in each (cold or not, the bench's sampling or the raw one)."""
    def rows(path):
        return {row["turn"]: row for row in (json.loads(line) for line in Path(path).read_text(
            encoding="utf-8").splitlines() if line.strip())}

    first, second = rows(before), rows(after)
    shared = sorted(set(first) & set(second))
    for turn in shared:
        if (first[turn]["question"], first[turn]["rung"]) != (second[turn]["question"], second[turn]["rung"]):
            raise Refused(f"probe {turn} is not the same in both files: another question or another home rung")
        mine, theirs = _asked_how(first[turn]), _asked_how(second[turn])
        if None not in (mine, theirs) and mine != theirs:
            raise Refused(f"probe {turn} was asked otherwise in each file: {mine} against {theirs} (cold, sampling)")
    return first, second, shared


def _asked_how(row):
    """How a probe was asked, ``(cold, sampling)``: nothing for a probe composed only."""
    return (row.get("cold", False), row.get("sampling", "bench")) if "verdict" in row else None


def compare(before, after):
    """Two files ``compose`` wrote over the same states, probe by probe: ``{"probes", "windows_differ",
    "refused", "asked", "before", "after", "gains", "losses"}`` -- the probes both composed; those whose windows
    differ (messages, rung or context); those refused, with their motives; and, when every probe of both was
    asked, how many each answers right and what the second gains and loses (None otherwise). Refused when a probe
    of one is not the other's -- another question or another home rung -- or was asked otherwise in each (cold or
    not, the bench's sampling or the raw one): a file written before windows were asked cold is compared with
    another such file only."""
    first, second, shared = _paired(before, after)

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


def compare_margins(before, after):
    """The margins of two files ``compose --margins`` wrote over the same states, probe by probe: ``{"probes",
    "before", "after", "deltas", "up", "down", "bounded", "unread"}`` -- the probes asked in both and read on both
    sides, their means, each probe's change, those read exactly on both sides that moved by ``MOVED`` or more
    either way, those read on a bound on either side, and the probes asked in both but not read on both sides; a
    window refused or composed only is none of them. None when either file read no margin. Refused as
    ``compare`` refuses."""
    first, second, shared = _paired(before, after)
    asked = [t for t in shared if "verdict" in first[t] and "verdict" in second[t]]
    if not asked or not all("margin" in first[t] and "margin" in second[t] for t in asked):
        return None
    both = [t for t in asked if first[t]["margin"] is not None and second[t]["margin"] is not None]
    deltas = {t: round(second[t]["margin"] - first[t]["margin"], 4) for t in both}
    bounded = [t for t in both if (first[t]["margin_bound"], second[t]["margin_bound"]) != ("exact", "exact")]

    def mean(rows):
        return round(sum(rows[t]["margin"] for t in both) / len(both), 4) if both else None

    return {"probes": both, "before": mean(first), "after": mean(second), "deltas": deltas,
            "up": [t for t in both if t not in bounded and deltas[t] >= MOVED],
            "down": [t for t in both if t not in bounded and deltas[t] <= -MOVED],
            "bounded": bounded, "unread": [t for t in asked if t not in both]}


def compare_against(before, after):
    """What the margins of two files ``compose --margins`` wrote over the same states are read against, where they
    moved: ``{turn: {"before", "after", "delta"}}`` for each probe read exactly on both sides whose margin moved by
    ``MOVED`` or more, or whose token read against changed -- " Oslo" on one side, "<" on the other, a city of
    another subject giving way to the copy of the sentence form. None when either file does not say what its margins
    are read against. Refused as ``compare`` refuses."""
    first, second, shared = _paired(before, after)
    asked = [t for t in shared if "verdict" in first[t] and "verdict" in second[t]]
    if not asked or not all("margin_against" in first[t] and "margin_against" in second[t] for t in asked):
        return None
    moved = {}
    for t in asked:
        old, new = first[t], second[t]
        if old["margin"] is None or new["margin"] is None or (old["margin_bound"], new["margin_bound"]) != (
                "exact", "exact"):
            continue
        delta = round(new["margin"] - old["margin"], 4)
        if abs(delta) >= MOVED or old["margin_against"] != new["margin_against"]:
            moved[t] = {"before": old["margin_against"], "after": new["margin_against"], "delta": delta}
    return moved


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
    composing.add_argument("--margins", type=int, default=0,
                           help="read each probe's margin with this many alternatives a token (1 to 20; Ollama on "
                                "this machine)")
    comparing = sub.add_parser("compare", help="read two composed files over the same states")
    comparing.add_argument("before")
    comparing.add_argument("after")
    args = parser.parse_args(argv)
    if args.command == "compare":
        try:
            print(json.dumps(dict(compare(args.before, args.after),
                                  margins=compare_margins(args.before, args.after),
                                  against=compare_against(args.before, args.after)), indent=2))
        except (Refused, OSError, ValueError, KeyError) as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        return 0
    return compose(args.states, args.out, rung=args.rung, model=args.model, ask=not args.no_ask, turns=args.turns,
                   margins=args.margins)


if __name__ == "__main__":
    raise SystemExit(main())
