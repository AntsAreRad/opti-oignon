#!/usr/bin/env python3
"""Contracts for the margins of the paired windows, and for asking each window cold.

A verdict is one greedy draw, and a small model flips it on a window that
gives it no more and no less to go on; and an engine that reuses the prefix
a request shares with the one before it, computed in other batches, answers
a window otherwise after another than alone. ``scripts/drift_pairs.py
compose`` asks each window of a model loaded for it alone on Ollama, and
with ``--margins K`` asks through Ollama's own API on this machine, the
model's own penalties and draft model held off, each token of its reply
with its K likeliest alternatives, reading each probe's margin.

  * MG1 -- THE MARGIN WHERE THE VALUES PART: with a stale statement, at the
    reply's token past the characters both statements share, the
    log-probability of the likeliest alternative that goes on with the
    expected statement less that of the likeliest that goes on with the
    stale one -- one going on with both deciding nothing, one running past
    a statement's end by marks only going on with it, by more of a word
    not -- negative when the reply went the stale way; with none, at each
    token of the value, the least: a date is held as firmly as its weakest
    digit, a wrong one read where the reply left it; the token read and the
    way the reply went there said.
  * MG2 -- A BOUND, NOT A GUESS: the rival's side not among the alternatives
    read gives a margin of at least, the expected's at most, the least
    likely read standing in; neither, no margin; the reply's own token is
    always among its alternatives.
  * MG3 -- NO MARGIN OFF THE FORM: a reply that leaves the form before the
    value, ends before it, reads no alternative there, a probe with no
    sentence form, tokens that are not the reply's -- a draft model's carry
    none -- and a character split across tokens have none, and say why.
  * MG4 -- COMPOSE WITH MARGINS, END TO END: the bench's trial line asked
    first, then each kept probe once, through the chat API at an Ollama
    endpoint on this machine (localhost, a loopback), of a model loaded for
    the window alone, the model's own penalties and draft held off, the
    window's context, the alternatives asked for, the model told not to
    think when the backend says so and nothing said when it says nothing;
    read for its verdict and its margin, the stale statement as rival where
    the probe retired one; refused, writing nothing and sending nothing:
    an endpoint off this machine, none, an engine that is not Ollama, more
    alternatives than Ollama reads, margins with nothing asked.
  * MG5 -- COMPARE READS THE MARGINS: the probes asked in both and read on
    both sides, their means, each change, those read exactly that moved by
    a nat or more, those read on a bound, those asked but not read on both
    sides, a refused window none of them; nothing when a file read none;
    two files whose probes were asked otherwise -- cold or not, the bench's
    sampling or the raw one, a file written before windows were asked cold
    -- refused, a probe composed only comparing with any.
  * MG6 -- EACH WINDOW ASKED COLD ON OLLAMA: without margins, the trial line
    first, then each probe of a model loaded for it alone on an Ollama
    backend, with the bench's sampling and the window's context; on another
    engine as it always was; each row says which.
  * MG7 -- THE REQUEST STAYS ON THIS MACHINE AND ENDS: posted straight to
    the address whatever proxy the environment names; Ollama's refusal said
    with what it said; no answer within the time given, and an answer that
    is no JSON object, refused by name.

Local-only (the public distribution ships no tests). The scripts load as
the paired windows' suite loads them (tests/test_drift_pairs_contracts.py);
Ollama's API is a recording seam, and a server on the loopback for MG7.
"""

import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_drift_ab_contracts as da  # noqa: E402
import test_drift_pairs_contracts as dp  # noqa: E402

BUDGET_S = {
    "test_mg1_the_margin_where_the_values_part": 2.0,
    "test_mg2_a_bound_not_a_guess": 2.0,
    "test_mg3_no_margin_off_the_form": 2.0,
    "test_mg4_compose_with_margins_end_to_end": 90.0,
    "test_mg5_compare_reads_the_margins": 2.0,
    "test_mg6_each_window_asked_cold_on_ollama": 20.0,
    "test_mg7_the_request_stays_on_this_machine_and_ends": 10.0,
}


def _tokens(*pieces):
    """A reply as Ollama's tokens: each ``(text, alternatives)``, the alternatives ``{token: logprob}`` with the
    token itself first."""
    out = []
    for text, alternatives in pieces:
        read = [{"token": token, "logprob": value} for token, value in alternatives.items()]
        out.append({"token": text, "logprob": read[0]["logprob"], "top_logprobs": read})
    return out


_ALICE = ("Alice lives in Oslo", "Alice lives in Berlin", "Alice lives in ")
_REVIEW = "The budget review is on 2026-04-30"


def _alice(value_alternatives):
    return _tokens(("Alice", {"Alice": -0.01}), (" lives", {" lives": -0.01}), (" in", {" in": -0.01}),
                   (next(iter(value_alternatives)), value_alternatives), (".", {".": -0.01}))


def _review(month=("4", {"4": -0.3, "5": -2.3})):
    """``_REVIEW``'s reply digit by digit, the weakest step at the day's first digit (1.0), ``month`` its second."""
    return _tokens(("The budget review is on", {"The budget review is on": -0.01}),
                   (" 2", {" 2": -0.01, " 1": -6.0}), ("0", {"0": -0.01, "1": -7.0}), ("2", {"2": -0.01, "3": -6.0}),
                   ("6", {"6": -0.02, "5": -5.0}), ("-", {"-": -0.01, "/": -9.0}), ("0", {"0": -0.01, "1": -3.0}),
                   month, ("-", {"-": -0.01, "/": -9.0}), ("3", {"3": -0.4, "1": -1.4}), ("0", {"0": -0.01, "1": -4.0}),
                   (".", {".": -0.01, "!": -5.0}))


# ---------------------------------------------------------------------------
# MG1 -- the margin where the values part
# ---------------------------------------------------------------------------
def test_mg1_the_margin_where_the_values_part():
    """MG1: the expected's alternative less the stale one's where they part; both-ways alternatives decide nothing;
    run-past by marks only; the stale way negative; with no stale statement, the weakest token of the value."""
    pairs = dp._pairs()
    expected, stale, prefix = _ALICE
    read = pairs.margin(_alice({" Oslo": -0.2, " Berlin": -3.5, " ": -0.1, " Os": -4.0}), expected, stale, prefix)
    assert read == (3.3, "exact", None, " Oslo", "expected"), ("Oslo's -0.2 less Berlin's -3.5", read)
    wrong = pairs.margin(_alice({" Berlin": -0.3, " Oslo": -1.6}), expected, stale, prefix)
    assert wrong == (-1.3, "exact", None, " Berlin", "rival"), ("the reply went the stale way", wrong)
    longer = pairs.margin(_alice({" Oslo": -0.2, " Oslo-Akershus": -0.1, " Berlin": -3.0}), expected, stale, prefix)
    assert longer == (2.8, "exact", None, " Oslo", "expected"), ("more of a word goes on with neither", longer)
    date = ("The Harvest release is on 2026-06-02", "The Harvest release is on 2026-05-12",
            "The Harvest release is on ")
    digits = _tokens(("The Harvest release is on 2026-0", {"The Harvest release is on 2026-0": -0.01}),
                     ("6", {"6": -0.5, "5": -1.25, "7": -6.0}), ("-02.", {"-02.": -0.01}))
    assert pairs.margin(digits, *date) == (0.75, "exact", None, "6", "expected"), "at the month, where the dates part"
    whole = _tokens(("The Harvest release is on", {"The Harvest release is on": -0.01}),
                    (" 2026-06-02.", {" 2026-06-02.": -0.4, " 2026-05-12.": -2.4}))
    assert pairs.margin(whole, *date) == (2.0, "exact", None, " 2026-06-02.", "expected"), "run past by a mark"
    alone = pairs.margin(_tokens(("Bob lives in", {"Bob lives in": -0.01}),
                                 (" Lyon", {" Lyon": -0.1, " Lyons": -1.1, " Paris": -2.6}), (".", {".": -0.01})),
                         "Bob lives in Lyon", None, "Bob lives in ")
    assert alone == (1.0, "exact", None, " Lyon", "expected"), ("no stale statement: against every other", alone)
    review = pairs.margin(_review(), _REVIEW, None, "The budget review is on ")
    assert review == (1.0, "exact", None, "3", "expected"), ("a date held as firmly as its weakest digit", review)
    wrong_date = _tokens(("The budget review is on", {"The budget review is on": -0.01}),
                         (" 2", {" 2": -0.01, " 1": -6.0}), ("0", {"0": -0.01, "1": -7.0}),
                         ("2", {"2": -0.01, "3": -6.0}), ("6", {"6": -0.02, "5": -5.0}), ("-", {"-": -0.01, "/": -9.0}),
                         ("0", {"0": -0.01, "1": -3.0}), ("5", {"5": -0.2, "4": -1.9}), ("-", {"-": -0.01, "/": -9.0}),
                         ("1", {"1": -0.05, "3": -4.0}), ("2", {"2": -0.1, "0": -3.0}), (".", {".": -0.01}))
    off = pairs.margin(wrong_date, _REVIEW, None, "The budget review is on ")
    assert off == (-1.7, "exact", None, "5", "other"), ("a wrong date read where the reply left it, never past", off)
    turns = da._script().TURNS
    probes = {turn.expects: n for n, turn in enumerate(turns, 1) if turn.probe}
    assert pairs.probe_values(turns, probes["Alice lives in Oslo"]) == _ALICE, "the stale statement it retired"
    assert pairs.probe_values(turns, probes["Bob lives in Lyon"]) == ("Bob lives in Lyon", None, "Bob lives in "), (
        "none retired")


# ---------------------------------------------------------------------------
# MG2 -- a bound, not a guess
# ---------------------------------------------------------------------------
def test_mg2_a_bound_not_a_guess():
    """MG2: the rival's side not read, at least; the expected's not read, at most; neither, none; the reply's own
    token among its alternatives."""
    pairs = dp._pairs()
    expected, stale, prefix = _ALICE
    least = pairs.margin(_alice({" Oslo": -0.1, " Paris": -4.1}), expected, stale, prefix)
    assert least == (4.0, "at least", None, " Oslo", "expected"), ("Berlin not read: at least", least)
    most = pairs.margin(_alice({" Berlin": -0.1, " Paris": -2.6}), expected, stale, prefix)
    assert most == (-2.5, "at most", None, " Berlin", "rival"), ("Oslo not read: at most", most)
    neither = pairs.margin(_alice({" Paris": -0.1, " Rome": -2.0}), expected, stale, prefix)
    assert neither == (None, None, "neither side among the alternatives read", None, None), neither
    unlisted = _alice({" Berlin": -3.5})
    unlisted[3]["token"], unlisted[3]["logprob"] = " Oslo", -0.2
    assert pairs.margin(unlisted, expected, stale, prefix) == (3.3, "exact", None, " Oslo", "expected"), (
        "the reply's own token counted though its alternatives leave it out")


# ---------------------------------------------------------------------------
# MG3 -- no margin off the form
# ---------------------------------------------------------------------------
def test_mg3_no_margin_off_the_form():
    """MG3: off the form, short, nothing read, no form, not the reply's tokens, a character split across tokens."""
    pairs = dp._pairs()
    expected, stale, prefix = _ALICE

    def why(*args, **kwargs):
        found = pairs.margin(*args, **kwargs)
        assert found[0] is None and found[1] is None and found[3:] == (None, None), found
        return found[2]

    she = _tokens(("She", {"She": -0.1}), (" lives", {" lives": -0.1}), (" in", {" in": -0.1}),
                  (" Oslo", {" Oslo": -0.1, " Berlin": -3.0}))
    assert why(she, expected, stale, prefix) == "the reply leaves the form before the value"
    short = _tokens(("Alice", {"Alice": -0.1}), (" lives", {" lives": -0.1}))
    assert why(short, expected, stale, prefix) == "the reply ends before the value"
    bare = [{"token": text, "logprob": -0.1} for text in ("Alice", " lives", " in", " Oslo")]
    assert why(bare, expected, stale, prefix) == "no alternative read at the value"
    assert why(_alice({" Oslo": -0.2}), "Alice lives in Oslo", None, "") == "no sentence form to read the value at"
    first = _alice({" Oslo": -0.2, " Berlin": -3.5})[:1]
    assert why(first, expected, stale, prefix, reply="Alice lives in Oslo.") == "the tokens read are not the reply's"
    assert pairs.margin(_alice({" Oslo": -0.2, " Berlin": -3.5}), expected, stale, prefix,
                        reply="Alice lives in Oslo.")[0] == 3.3, "the reply's own tokens read"
    split = [{"token": "Alice lives in Caf\ufffd", "logprob": -0.1, "bytes": list(b"Alice lives in Caf\xc3")},
             {"token": "\ufffd", "logprob": -0.1, "bytes": list(b"\xa9")}]
    assert why(split, "Alice lives in Caf\u00e9", None, "Alice lives in ", reply="Alice lives in Caf\u00e9") == (
        "a character split across tokens")


# ---------------------------------------------------------------------------
# MG4 -- compose with margins, end to end
# ---------------------------------------------------------------------------
_KEPT = (16, 36, 49)


def _kept_run(ab, lib, loaded, states, turns=50):
    """``test_drift_pairs_contracts._kept_run`` up to the first update probe: probes 16, 36 and 49 kept."""
    composer = loaded["opti_oignon.memory.composer"]

    def ask(messages):
        return "Noted."

    budget = composer.Budget(window=2000, reserve=200, core=300, receipts=300, peels=800, flesh=200, turn=200)
    arm = ab.onion_arm(ask, librarian=lib, config=dp._config(lib), summarize=da._faithful,
                       wrap=loaded[dp._WRAP].wrap, budget=budget, states=states, record=set(_KEPT))
    ab.run_arm(ab.TURNS[:turns], arm)


class _Ollama:
    """An Ollama backend and its chat API: each probe answered with the fact it expects, word by word and digit by
    digit, every token read beside an alternative no statement goes on with, the stale statement's at the token
    where the probe's statements part; the requests kept."""

    def __init__(self, ab, pairs, endpoint="http://127.0.0.1:11434", name="ollama", think=False):
        self._ab, self._pairs, self._endpoint, self.name, self._think = ab, pairs, endpoint, name, think
        self.requests = []

    def endpoint(self):
        return self._endpoint

    def _think_flag(self, model, think):
        return self._think

    def fetch(self, url, body):
        self.requests.append((url, body))
        asked = body["messages"][-1]["content"]
        turn = next((n for n, t in enumerate(self._ab.TURNS, 1) if t.probe and asked.endswith(t.text)), None)
        if turn is None:
            return {"message": {"role": "assistant", "content": "Noted."}, "logprobs": []}
        expected, stale, _prefix = self._pairs.probe_values(self._ab.TURNS, turn)
        part = len(self._pairs.os.path.commonprefix([expected, stale])) if stale else None
        tokens, start = [], 0
        for text in re.findall(r" ?[A-Za-z]+|\d|[^A-Za-z\d]", f"{expected}."):
            alternatives = {text: -0.25, "~": -2.25}
            if part is not None and start <= part < start + len(text):
                alternatives[stale[start:start + len(text)]] = -1.25
            tokens.append({"token": text, "logprob": -0.25,
                           "top_logprobs": [{"token": t, "logprob": v} for t, v in alternatives.items()]})
            start += len(text)
        return {"message": {"role": "assistant", "content": f"{expected}."}, "logprobs": tokens}


def test_mg4_compose_with_margins_end_to_end(tmp_path):
    """MG4: the trial line, then each probe once through the chat API on this machine, read for its verdict and
    margin; refusals that write and send nothing."""
    ab, pairs = da._script(), dp._pairs()
    loaded, restore = dp._window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        states = tmp_path / "states"
        _kept_run(ab, lib, loaded, states)

        def composed(ollama, name, **kwargs):
            lib.reset_librarian()
            out = tmp_path / f"{name}.jsonl"
            code = pairs.compose(states, out, resolve=lambda model: ollama, fetch=ollama.fetch, **kwargs)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] if code == 0 else []
            return code, rows, out

        ollama = _Ollama(ab, pairs)
        code, rows, _out = composed(ollama, "margins", margins=5)
        assert code == 0 and [r["turn"] for r in rows] == list(_KEPT), (code, "every kept probe, in order")
        read = {r["turn"]: (r["verdict"], r["margin"], r["margin_bound"], r["margin_rival"], r["margin_token"],
                            r["margin_took"]) for r in rows}
        assert read == {16: ("agrees", 2.0, "exact", None, " Lyon", "expected"),
                        36: ("agrees", 2.0, "exact", None, "2", "expected"),
                        49: ("agrees", 1.0, "exact", "Alice lives in Berlin", " Oslo", "expected")}, read
        assert all((r["cold"], r["sampling"]) == (True, "raw") for r in rows), "each row says how it was asked"
        assert [body["messages"] for _url, body in ollama.requests] == [ab._TRIAL] + [r["messages"] for r in rows], (
            "the trial line first, then each window once")
        url, body = ollama.requests[1]
        assert url == "http://127.0.0.1:11434/api/chat", url
        assert body["options"] == dict(ab.SAMPLING, num_ctx=rows[0]["num_ctx"], **pairs.AS_READ), body["options"]
        assert pairs.AS_READ == {"presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0,
                                 "draft_num_predict": 0}, pairs.AS_READ
        assert all((b["logprobs"], b["top_logprobs"], b["stream"], b["think"], b["keep_alive"]) == (
            True, 5, False, False, 0) for _u, b in ollama.requests), "the alternatives, of a model loaded alone"
        assert "num_ctx" not in ollama.requests[0][1]["options"], "the trial line at the model's own context"
        quiet = _Ollama(ab, pairs, think=None)
        assert composed(quiet, "quiet", margins=5)[0] == 0 and all("think" not in b for _u, b in quiet.requests), (
            "nothing said of thinking when the backend says nothing")
        for host in ("http://localhost:11434", "http://[::1]:11434"):
            assert composed(_Ollama(ab, pairs, endpoint=host), "host", margins=5)[0] == 0, host
        refused = {}
        for key, backend in (("far", _Ollama(ab, pairs, endpoint="http://10.0.0.7:11434")),
                             ("none", _Ollama(ab, pairs, endpoint=None)),
                             ("engine", _Ollama(ab, pairs, name="llama_server"))):
            refused[key] = composed(backend, key, margins=5)
            assert not backend.requests, ("nothing sent", key)
        refused["many"] = composed(ollama, "many", margins=21)
        refused["unasked"] = composed(ollama, "unasked", ask=False, margins=5)
        assert {key: code for key, (code, _rows, _out) in refused.items()} == dict.fromkeys(refused, 2), refused
        assert not any(any(tmp_path.glob(f"{key}.jsonl*")) for key in refused), "a refusal writes nothing"
    finally:
        lib.reset_librarian()
        restore()


# ---------------------------------------------------------------------------
# MG5 -- compare reads the margins
# ---------------------------------------------------------------------------
def test_mg5_compare_reads_the_margins(tmp_path):
    """MG5: the probes asked in both and read on both sides, their means and changes, those that moved, on a bound,
    unread; a refused window none of them; files asked otherwise refused."""
    pairs = dp._pairs()
    window = [{"role": "user", "content": "w"}]

    def row(turn, value, bound="exact", **extra):
        return dict({"turn": turn, "question": f"q{turn}", "rung": 4096, "refusal": None, "messages": window,
                     "window_rung": 4096, "num_ctx": 4096, "verdict": "agrees", "cold": True, "sampling": "raw",
                     "margin": value, "margin_bound": bound}, **extra)

    def refused_row(turn):
        return {"turn": turn, "question": f"q{turn}", "rung": 4096, "refusal": "count"}

    def written(name, rows):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return path

    before = written("before", [row(16, 3.0), row(26, 2.0), row(36, 1.0), row(49, None), row(59, 4.0, "at least"),
                                row(69, 1.0)])
    after = written("after", [row(16, 4.5), row(26, 1.5), row(36, -0.5), row(49, 2.0), row(59, 6.0),
                              refused_row(69)])
    assert pairs.compare_margins(before, after) == {
        "probes": [16, 26, 36, 59], "before": 2.5, "after": 2.875,
        "deltas": {16: 1.5, 26: -0.5, 36: -1.5, 59: 2.0}, "up": [16], "down": [36], "bounded": [59],
        "unread": [49]}, "probe by probe; a change on a bound is not a move"
    plain = written("plain", [{k: v for k, v in r.items() if not k.startswith("margin")}
                              for r in (row(16, 0.0), row(26, 0.0))])
    assert pairs.compare_margins(before, plain) is None, "a file that read none: nothing"
    with pytest.raises(pairs.Refused, match="probe 16"):
        pairs.compare_margins(before, written("other", [row(16, 1.0, question="another")]))
    raw = written("raw", [row(16, 1.0)])
    old = written("old", [{k: v for k, v in row(16, 1.0).items() if k not in ("cold", "sampling")}])
    for name, other in (("warm", [row(16, 1.0, cold=False)]), ("bench", [row(16, 1.0, sampling="bench")]),
                        ("old", None)):
        path = old if other is None else written(name, other)
        with pytest.raises(pairs.Refused, match="asked otherwise"):
            pairs.compare(raw, path)
        with pytest.raises(pairs.Refused, match="asked otherwise"):
            pairs.compare_margins(raw, path)
    composed = written("composed", [{"turn": 16, "question": "q16", "rung": 4096, "refusal": None, "messages": window,
                                     "window_rung": 4096, "num_ctx": 4096}])
    assert pairs.compare(raw, composed)["probes"] == [16], "a probe composed only compares with any"


# ---------------------------------------------------------------------------
# MG6 -- each window asked cold on Ollama
# ---------------------------------------------------------------------------
class _Generating(dp._Answering):
    """The paired suite's answering backend, named, keeping what each request carried beside its messages."""

    def __init__(self, ab, name):
        super().__init__(ab)
        self.name, self.asked = name, []

    def generate(self, model, messages, **kwargs):
        self.asked.append((messages, kwargs))
        if messages == self._ab._TRIAL:
            return dp.SimpleNamespace(content="Noted.")
        return super().generate(model, messages, **kwargs)


def test_mg6_each_window_asked_cold_on_ollama(tmp_path):
    """MG6: on Ollama, the trial line, then each probe of a model loaded for it alone, with the bench's sampling and
    the window's context; on another engine as before; each row says which."""
    ab, pairs = da._script(), dp._pairs()
    loaded, restore = dp._window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        states = tmp_path / "states"
        dp._kept_run(ab, lib, loaded, states)
        for name, cold in (("ollama", True), ("llama_server", False)):
            lib.reset_librarian()
            backend = _Generating(ab, name)
            out = tmp_path / f"{name}.jsonl"
            assert pairs.compose(states, out, resolve=lambda model: backend) == 0, name
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            assert all(r["verdict"] == "agrees" for r in rows), ("asked and read", name)
            assert all((r["cold"], r["sampling"]) == (cold, "bench") for r in rows), ("each row says how", name)
            windows = [(r["messages"], {"options": dict(ab.SAMPLING, num_ctx=r["num_ctx"]), "think": False,
                                        **({"keep_alive": 0} if cold else {})}) for r in rows]
            trial = [(ab._TRIAL, {"options": dict(ab.SAMPLING), "think": False, "keep_alive": 0})] if cold else []
            assert backend.asked == trial + windows, ("the trial first on Ollama alone, then each window", name)
    finally:
        lib.reset_librarian()
        restore()


# ---------------------------------------------------------------------------
# MG7 -- the request stays on this machine and ends
# ---------------------------------------------------------------------------
class _Server(BaseHTTPRequestHandler):
    """A server on the loopback answering by path: a JSON object, a refusal with its words, a list, a slow answer."""

    def do_POST(self):  # noqa: N802 - the handler's own name
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/slow":
            time.sleep(1.5)
        code, body = {"/refuse": (404, b'{"error": "model not found"}'), "/list": (200, b"[]")}.get(
            self.path, (200, b'{"message": {"role": "assistant", "content": "Noted."}}'))
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_mg7_the_request_stays_on_this_machine_and_ends(monkeypatch):
    """MG7: straight to the loopback whatever proxy is named; a refusal said; a timeout and a non-object refused."""
    pairs = dp._pairs()
    for variable in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(variable, "http://127.0.0.1:9")
    for variable in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(variable, raising=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        assert pairs._post(f"{base}/ok", {"model": "m"})["message"]["content"] == "Noted.", "past the proxy named"
        with pytest.raises(pairs.Refused, match=r"404.*model not found"):
            pairs._post(f"{base}/refuse", {"model": "m"})
        with pytest.raises(pairs.Refused, match="no JSON object"):
            pairs._post(f"{base}/list", {"model": "m"})
        with pytest.raises(pairs.Refused, match="did not answer"):
            pairs._post(f"{base}/slow", {"model": "m"}, timeout=0.3)
    finally:
        server.shutdown()
        server.server_close()
