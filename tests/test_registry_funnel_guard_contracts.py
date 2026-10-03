#!/usr/bin/env python3
"""Contracts for the guard that keeps inference inside the registry.

Four sessions made BackendRegistry worth routing through: admission on every
head, a provenance label on every figure, a schema and a tool list as engine
options, a probed VRAM capacity, a sealed recipe. None of it applies to a
request that reaches the client behind the registry's back -- and at the time
this guard was written, twenty-nine modules did, at fifty-one sites.

This guard is a RATCHET in the shape of the isolation-seal guard, and for the
same reason: a ratchet that only counts is a ratchet on the count. Every owed
module carries the digest of its text as the debt was enumerated. An owed
module that changes while still calling the client directly no longer matches
its seal and becomes a violation: touch it, and you migrate it. The debt is
frozen as found, it can be paid, and it cannot grow -- not in modules, and
not in lines.

The census is taken on the syntax tree, never on the text. A docstring that
says "passed straight to ollama.chat" is not a request, and a guard that
charged for prose would be green and red for reasons unrelated to what leaves
the process.

  * RF1 -- every owed name carries a full seal.
  * RF2 -- the census counts calls, not prose, and knows every spelling: the
    bare module, an alias, a name imported from the module, and a client
    constructed from it.
  * RF3 -- a direct caller nobody owes for is a violation.
  * RF4 -- an owed module that has not moved is tolerated.
  * RF5 -- an owed module that gained a line while still calling directly is
    a broken seal.
  * RF6 -- paying the debt is the way out: a module that no longer calls the
    client directly is not a violation, and its entry becomes stale.
  * RF7 -- an owed module that vanished is stale.
  * RF8 -- the estate satisfies its own guard as it stands.
  * RF9 -- the entry point refuses a violation and accepts the estate.
  * RF10 -- the probe is proven able to count: on the real tree it finds the
    debt the ledger records, and that debt is not zero.

The ledger reached zero in the third convergence block. RF1, RF5, RF6, RF7
and RF10 read the real ledger for an owed name and are deselected by name;
their successors keep every property on a synthetic ledger, where the
guard's helpers are exercised against entries the contract writes itself:

  * RF11 -- the real ledger is empty, and a seal is a full digest.
  * RF12 -- an owed module that gained a line while still calling directly
    is a broken seal.
  * RF13 -- paying the debt is the way out: the entry becomes stale.
  * RF14 -- an owed module that vanished is stale.
  * RF15 -- the probe is proven able to count on the real tree: the funnel
    itself carries the calls, nothing outside it does, and a ``ps()`` read
    counts as a site since the loaded set became a head.
  * RF16 -- the entry point refuses an empty estate, and its green names
    the number of modules it scanned.

The fourth convergence block found a request the census could not see: a
module that binds the client module to an attribute at construction and
requests through the attribute. A receiver is not a disguise either:

  * RF17 -- the census follows the client into a bound attribute and a
    bound name, counts a reference to a request method that is handed on
    uncalled, and on the real tree nothing outside the funnel carries one.
  * RF18 -- a catalogue read, ``list()`` or ``show()``, is a site since the
    catalogue became two heads on the backend contract; model management
    (``pull``, ``delete``) is not, by decision; and on the real tree only
    the funnel reads the catalogue from the client.

The census could not see a request that never touches the client library
at all: a module that posts to the inference server's endpoint with its own
HTTP transport. Six modules did, at nine sites; one was paid in the block
that widened the census, the other five are sealed on a ledger of their own
with the reason each needs a decision:

  * RF19 -- a raw site is an endpoint literal of the inference server, in a
    string or an f-string, in a module that imports an HTTP transport; the
    application's own route, a docstring, a model-management endpoint and
    a path that only begins with an endpoint are not; the two censuses do
    not count each other's sites.
  * RF20 -- a raw site nobody owes for is a violation; an owed module that
    has not moved is tolerated, and one that grew while still posting is a
    broken seal.
  * RF21 -- paying the raw debt makes the entry stale, and so does vanishing.
  * RF22 -- on the real tree the raw ledger names exactly the modules that
    post, each sealed on its current text and each carrying a site, and
    nothing outside it posts.
  * RF23 -- the entry point names the raw debt with its site count, and
    refuses an unowed raw site by name.

The RAG embedder was the first raw debt paid, once the batch had a head on
the backend contract. RF22 and RF23 named the five modules the widening
found and are deselected by name; their successors keep every assertion
over the four that remain:

  * RF24 -- the raw ledger names exactly the four modules that still post,
    each sealed and carrying a site, nothing outside it posts, and the
    embedder posts no more.
  * RF25 -- the entry point names the four with their site count, and
    refuses an unowed raw site by name.

The red team's three entry points went through the registry next, with the
loopback property moved onto the backend's real endpoint, and the
launcher's liveness probe was exempted by name, by decision: probing that
a process answers is not an inference request. RF24 and RF25 named the
four and are deselected by name:

  * RF26 -- the raw ledger is empty; the one exemption is the launcher,
    with its reason, and it still spells a site, so the exemption is not
    decoration; nothing else posts, the red team included.
  * RF27 -- an exempt module that stops posting is a stale exemption, an
    exempt module that posts is not a violation, the entry point names the
    exemption beside its empty ledger, and an unowed raw site is refused.

Routing every Ollama head through one host-aware transport hid the funnel's
own calls from the census: a request made on what a function returns was
not seen, so a module that wrapped the client in a helper would have
passed as clean.

  * RF28 -- the census follows the client through a function or a method
    that returns it: a request on its result is a site, and on the real
    tree the funnel's heads are seen again while nothing outside it has one.

The client also carries Ollama's cloud search and fetch, which post a query
to ollama.com under an account key and have no head on the backend contract.
They are counted like a request, in every spelling the census knows:

  * RF29 -- ``web_search`` and ``web_fetch`` are sites on the module, a
    client, a bound receiver or an imported name, through a submodule
    import, ``import_module`` or ``__import__`` with a constant, and
    ``getattr``; on the real tree nothing outside the funnel reaches them,
    and the funnel itself uses neither. The spellings a review found
    unseen are sites too: a star import, the import functions reached or
    renamed otherwise, a ``sys.modules`` lookup, every binding form (an
    unpacking, a walrus, a loop or ``with`` target, a default, an argument,
    a container, a class attribute, a lambda), and a chain rooted at an
    import call or at what ``getattr`` hands back.
  * RF30 -- ``/api/web_search`` and ``/api/web_fetch`` are raw sites, and
    they and the cloud host count even in a module that imports no HTTP
    transport; a path is read without its fragment, surrounding spaces or
    percent encoding, and as a server routes it -- bytes, dot segments,
    repeated slashes, a host written with a Unicode full stop; the cloud
    host counts as a place to post from, never as a link to a page; an
    exemption or a raw ledger entry excuses local endpoints only; the
    funnel spells none of them, and the entry point refuses both kinds by
    name.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window.
"""

import ast
import hashlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate  # noqa: E402

_GUARD = REPO / ".github" / "scripts" / "registry_funnel_guard.py"
_PACKAGE = REPO / "opti_oignon"

_DIRECT = "import ollama\n\ndef ask():\n    return ollama.chat(model='m', messages=[])\n"
_ALIASED = "import ollama as _o\n\ndef ask():\n    return _o.generate(model='m', prompt='p')\n"
_BARE = "from ollama import chat\n\ndef ask():\n    return chat(model='m', messages=[])\n"
_CLIENT = "import ollama\n\ndef ask():\n    return ollama.Client(host='h').chat(model='m', messages=[])\n"
_ATTRIBUTE = (
    "import ollama as _m\n\nclass P:\n    def __init__(self, m=None):\n"
    "        self._c = m or _m\n\n    def ask(self):\n"
    "        return self._c.chat(model='m', messages=[])\n"
)
_BOUND = (
    "import ollama\n\ndef ask():\n    c = ollama.Client(host='h')\n"
    "    return c.chat(model='m', messages=[])\n"
)
_UNCALLED = "import ollama\n\ndef pick():\n    return ollama.chat\n"
_LIST = "import ollama\n\ndef names():\n    return [m.model for m in ollama.list().models]\n"
_SHOW = "import ollama as _o\n\ndef ctx(m):\n    return _o.show(m).modelinfo\n"
_PULL = "import ollama\n\ndef fetch(m):\n    ollama.pull(m)\n    ollama.delete(m)\n"
_PROSE = '"""This module used to call ollama.chat directly."""\n\ndef ask():\n    return None\n'
_ROUTED = (
    "from opti_oignon.inference_backend import get_backend_registry\n\n"
    "def ask():\n    return get_backend_registry().resolve_backend('m').generate('m', [])\n"
)


def _load():
    loaded, restore = isolate(targets={"registry_funnel_guard": _GUARD})
    return loaded["registry_funnel_guard"], restore


def _real_files():
    """The estate as the guard reads it: repo-relative posix paths and text."""
    out = []
    for p in sorted(_PACKAGE.rglob("*.py")):
        rel = p.relative_to(REPO).as_posix()
        out.append((rel, p.read_text(encoding="utf-8", errors="ignore")))
    return out


def _an_owed_name(guard):
    return sorted(guard.LEDGER)[0]


# ---------------------------------------------------------------------------
# RF1 -- every owed name carries a full seal
# ---------------------------------------------------------------------------
def test_rf1_every_owed_name_carries_a_seal():
    guard, restore = _load()
    try:
        assert guard.LEDGER, "the ledger is not empty: the debt is real"
        for name, seal in guard.LEDGER.items():
            assert name.startswith("opti_oignon/") and name.endswith(".py"), (
                f"an owed name is a repo-relative module path: {name!r}"
            )
            assert len(seal) == 64 and int(seal, 16) >= 0, (
                f"the seal of {name} is a full sha256, not a prefix or a count"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF2 -- the census counts calls, not prose, in every spelling
# ---------------------------------------------------------------------------
def test_rf2_the_census_counts_calls_in_every_spelling_and_never_prose():
    guard, restore = _load()
    try:
        assert guard.count_sites(_DIRECT) == 1, "the bare module form is one site"
        assert guard.count_sites(_ALIASED) == 1, "an alias is not a disguise"
        assert guard.count_sites(_BARE) == 1, (
            "a name imported from the module is still the module"
        )
        assert guard.count_sites(_CLIENT) >= 1, (
            "a client constructed from the module is a direct route too"
        )
        assert guard.count_sites(_PROSE) == 0, (
            "a docstring that names the client is not a request"
        )
        assert guard.count_sites(_ROUTED) == 0, (
            "a request through the registry is what the guard exists to allow"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF3 -- a direct caller nobody owes for is a violation
# ---------------------------------------------------------------------------
def test_rf3_a_direct_caller_nobody_owes_for_is_a_violation():
    guard, restore = _load()
    try:
        files = [("opti_oignon/newcomer.py", _DIRECT)]
        assert guard.find_violations(files) == ["opti_oignon/newcomer.py"]
        assert guard.find_violations([("opti_oignon/newcomer.py", _ROUTED)]) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF4 -- an owed module that has not moved is tolerated
# ---------------------------------------------------------------------------
def test_rf4_an_owed_module_that_has_not_moved_is_tolerated():
    guard, restore = _load()
    try:
        files = _real_files()
        assert guard.find_violations(files) == [], (
            "the debt as enumerated is carried, not charged"
        )
        assert guard.find_broken_seals(files) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF5 -- an owed module that gained a line is a broken seal
# ---------------------------------------------------------------------------
def test_rf5_an_owed_module_that_gained_a_line_is_a_broken_seal():
    guard, restore = _load()
    try:
        name = _an_owed_name(guard)
        files = [
            (n, t + "\n# one more line\n" if n == name else t)
            for n, t in _real_files()
        ]
        assert guard.find_broken_seals(files) == [name], (
            "an owed module that changed while still calling directly no "
            "longer matches its seal: touch it, and you migrate it"
        )
        assert name not in guard.find_violations(files), (
            "and it is charged in exactly one place, not two"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF6 -- paying the debt is the way out
# ---------------------------------------------------------------------------
def test_rf6_paying_the_debt_is_not_a_violation_and_makes_the_entry_stale():
    guard, restore = _load()
    try:
        name = _an_owed_name(guard)
        files = [(n, _ROUTED if n == name else t) for n, t in _real_files()]
        assert guard.find_broken_seals(files) == [], (
            "a module that migrated is not broken; charging it would leave "
            "no way to pay"
        )
        assert guard.find_violations(files) == []
        assert guard.find_stale_ledger_entries(files) == [name], (
            "it migrated, so it must come off the ledger, or the debt count "
            "stops meaning anything"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF7 -- an owed module that vanished is stale
# ---------------------------------------------------------------------------
def test_rf7_an_owed_module_that_vanished_is_stale():
    guard, restore = _load()
    try:
        name = _an_owed_name(guard)
        files = [(n, t) for n, t in _real_files() if n != name]
        assert guard.find_stale_ledger_entries(files) == [name]
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF8 -- the estate satisfies its own guard
# ---------------------------------------------------------------------------
def test_rf8_the_estate_satisfies_its_own_guard():
    guard, restore = _load()
    try:
        files = _real_files()
        assert guard.find_violations(files) == []
        assert guard.find_broken_seals(files) == []
        assert guard.find_stale_ledger_entries(files) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF9 -- the entry point refuses a violation and accepts the estate
# ---------------------------------------------------------------------------
def test_rf9_the_entry_point_refuses_a_violation_and_accepts_the_estate(
    tmp_path, capsys,
):
    guard, restore = _load()
    try:
        bad = tmp_path / "opti_oignon"
        bad.mkdir()
        (bad / "newcomer.py").write_text(_DIRECT, encoding="utf-8")
        assert guard.main(["guard", str(tmp_path)]) == 1, (
            "a tree with an unowed direct caller is refused"
        )
        assert "newcomer.py" in capsys.readouterr().out

        assert guard.main(["guard", str(REPO)]) == 0, (
            "the estate as it stands is accepted"
        )
        out = capsys.readouterr().out
        assert "owed" in out and "may only shrink" in out, (
            "and the acceptance names the debt it is carrying"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF10 -- the probe is proven able to count
# ---------------------------------------------------------------------------
def test_rf10_the_probe_finds_the_debt_the_ledger_records():
    guard, restore = _load()
    try:
        files = dict(_real_files())
        total = 0
        for name in guard.LEDGER:
            n = guard.count_sites(files[name])
            assert n > 0, (
                f"{name} is owed for, so the census must find at least one "
                "site in it -- a zero here would mean the ledger and the "
                "probe disagree about what a direct call is"
            )
            total += n
        assert total >= len(guard.LEDGER), "the debt is not zero"
        assert hashlib.sha256(files[_an_owed_name(guard)].encode("utf-8")).hexdigest() \
            == guard.LEDGER[_an_owed_name(guard)], (
            "and the seal is taken on the same text the census reads"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF11-RF16 -- the same properties on a synthetic ledger, and the zero's
# denominator
# ---------------------------------------------------------------------------
_PS = "import ollama as _o\n\ndef loaded():\n    return _o.ps()\n"


def _with_ledger(guard, ledger):
    guard.LEDGER = dict(ledger)


def test_rf11_the_real_ledger_is_empty_and_a_seal_is_a_full_digest():
    guard, restore = _load()
    try:
        assert guard.LEDGER == {}, "the debt is paid: nothing is owed"
        seal = guard.digest(_DIRECT)
        assert len(seal) == 64 and int(seal, 16) >= 0
        assert seal == hashlib.sha256(_DIRECT.encode("utf-8")).hexdigest()
    finally:
        restore()


def test_rf12_an_owed_module_that_gained_a_line_is_a_broken_seal():
    guard, restore = _load()
    try:
        name = "opti_oignon/owed.py"
        _with_ledger(guard, {name: guard.digest(_DIRECT)})
        assert guard.find_broken_seals([(name, _DIRECT)]) == []
        grown = _DIRECT + "\nx = 1\n"
        assert guard.find_broken_seals([(name, grown)]) == [name], (
            "one more line while still calling directly breaks the seal"
        )
        assert name not in guard.find_violations([(name, grown)]), (
            "an owed name is answered for by the seal, not the violation list"
        )
    finally:
        restore()


def test_rf13_paying_the_debt_is_not_a_violation_and_makes_the_entry_stale():
    guard, restore = _load()
    try:
        name = "opti_oignon/owed.py"
        _with_ledger(guard, {name: guard.digest(_DIRECT)})
        files = [(name, _ROUTED)]
        assert guard.find_broken_seals(files) == []
        assert guard.find_violations(files) == []
        assert guard.find_stale_ledger_entries(files) == [name]
    finally:
        restore()


def test_rf14_an_owed_module_that_vanished_is_stale():
    guard, restore = _load()
    try:
        name = "opti_oignon/owed.py"
        _with_ledger(guard, {name: guard.digest(_DIRECT)})
        assert guard.find_stale_ledger_entries([("opti_oignon/other.py", _ROUTED)]) == [name]
    finally:
        restore()


def test_rf15_the_probe_counts_on_the_real_tree_and_a_ps_read_is_a_site():
    guard, restore = _load()
    try:
        files = dict(_real_files())
        assert guard.count_sites(files["opti_oignon/inference_backend.py"]) >= 4, (
            "control: the funnel itself carries the calls, and the probe sees them"
        )
        outside = {name: guard.count_sites(text) for name, text in files.items() if name != "opti_oignon/inference_backend.py"}
        assert sum(outside.values()) == 0, {k: v for k, v in outside.items() if v}
        assert guard.count_sites(_PS) == 1, "a ps() read bypasses the loaded-models head"
        assert guard.count_sites(_DIRECT) == 1
    finally:
        restore()


def test_rf16_the_entry_point_refuses_an_empty_estate_and_names_its_denominator(tmp_path, capsys):
    guard, restore = _load()
    try:
        (tmp_path / "opti_oignon").mkdir()
        assert guard.main(["guard", str(tmp_path)]) == 1, "an estate with nothing to scan is a refusal"
        assert "nothing was scanned" in capsys.readouterr().out
        assert guard.main(["guard", str(REPO)]) == 0
        out = capsys.readouterr().out
        scanned = len(_real_files())
        assert f"{scanned} module(s) scanned" in out, out
        assert "0 module(s) owed" in out
    finally:
        restore()


def test_rf17_the_census_follows_the_client_into_a_bound_receiver_and_an_uncalled_reference():
    guard, restore = _load()
    try:
        assert guard.count_sites(_ATTRIBUTE) == 1, (
            "a request through an attribute the client module was bound to is a site"
        )
        assert guard.count_sites(_BOUND) == 2, (
            "a client bound to a name is a site, and a request through that name is another"
        )
        assert guard.count_sites(_UNCALLED) == 1, (
            "a request method handed on uncalled is a route to the client"
        )
        assert guard.count_sites(_ROUTED) == 0
        files = dict(_real_files())
        outside = {name: guard.count_sites(text) for name, text in files.items() if name != "opti_oignon/inference_backend.py"}
        assert sum(outside.values()) == 0, {k: v for k, v in outside.items() if v}
    finally:
        restore()


def test_rf18_a_catalogue_read_is_a_site_and_model_management_is_not():
    guard, restore = _load()
    try:
        assert guard.count_sites(_LIST) == 1, "a list() read bypasses the list_models head"
        assert guard.count_sites(_SHOW) == 1, "a show() read bypasses the model_info head"
        assert guard.count_sites(_PULL) == 0, (
            "pull and delete have no head on the contract and are not counted, by decision"
        )
        assert {"list", "show"} <= set(guard._CLIENT_CALLS)
        files = dict(_real_files())
        assert guard.count_sites(files["opti_oignon/inference_backend.py"]) >= 6, (
            "control: the funnel reads the catalogue from the client, and the probe sees it"
        )
        outside = {name: guard.count_sites(text) for name, text in files.items() if name != "opti_oignon/inference_backend.py"}
        assert sum(outside.values()) == 0, {k: v for k, v in outside.items() if v}
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF19-RF23 -- raw HTTP to the inference server
# ---------------------------------------------------------------------------
_RAW_REQUESTS = "import requests\n\ndef ask(url):\n    return requests.post(f'{url}/api/generate', json={})\n"
_RAW_URLLIB = (
    "def up():\n    import urllib.request\n"
    "    return urllib.request.urlopen('http://localhost:11434/api/tags')\n"
)
_RAW_ROUTE = "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n@router.post('/api/chat')\ndef chat():\n    return {}\n"
_RAW_PROSE = 'import requests\n\ndef f():\n    """Posts to /api/chat"""\n    return requests.get("https://example.org")\n'
_RAW_PULL = "import requests\n\ndef fetch(u):\n    return requests.post(u + '/api/pull', json={})\n"
_RAW_APP = "import httpx\n\ndef s(u):\n    return httpx.get(f'{u}/api/chat/stream')\n"
_RAW_OWED = (
    "opti_oignon/rag/embeddings.py",
    "opti_oignon/redteam/generator.py",
    "opti_oignon/redteam/strategies.py",
    "opti_oignon/redteam/targets.py",
    "opti_oignon/ui.py",
)


def _with_raw_ledger(guard, ledger):
    guard.RAW_LEDGER = dict(ledger)


def test_rf19_a_raw_site_is_an_endpoint_literal_in_a_module_with_an_http_transport():
    guard, restore = _load()
    try:
        assert guard.count_raw_sites(_RAW_REQUESTS) == 1, "an f-string endpoint behind requests is a site"
        assert guard.count_raw_sites(_RAW_URLLIB) == 1, "a transport imported inside a function counts too"
        assert guard.count_raw_sites(_RAW_ROUTE) == 0, "the application's own route is not a request"
        assert guard.count_raw_sites(_RAW_PROSE) == 0, "a docstring is prose"
        assert guard.count_raw_sites(_RAW_PULL) == 0, "model management is not counted, by the same decision as the client"
        assert guard.count_raw_sites(_RAW_APP) == 0, "a path that only begins with an endpoint is another route"
        assert guard.count_sites(_RAW_REQUESTS) == 0 and guard.count_raw_sites(_DIRECT) == 0, (
            "the two censuses do not count each other's sites"
        )
    finally:
        restore()


def test_rf20_an_unowed_raw_site_is_a_violation_and_a_grown_owed_one_a_broken_seal():
    guard, restore = _load()
    try:
        name = "opti_oignon/poster.py"
        _with_raw_ledger(guard, {})
        assert guard.find_raw_violations([(name, _RAW_REQUESTS)]) == [name]
        _with_raw_ledger(guard, {name: guard.digest(_RAW_REQUESTS)})
        assert guard.find_raw_violations([(name, _RAW_REQUESTS)]) == []
        assert guard.find_raw_broken_seals([(name, _RAW_REQUESTS)]) == [], "an owed module that has not moved is tolerated"
        grown = _RAW_REQUESTS + "\nx = 1\n"
        assert guard.find_raw_broken_seals([(name, grown)]) == [name], "one more line while still posting breaks the seal"
        assert guard.find_raw_violations([(name, grown)]) == [], "an owed name is answered for by its seal"
    finally:
        restore()


def test_rf21_paying_or_losing_raw_debt_makes_the_entry_stale():
    guard, restore = _load()
    try:
        name = "opti_oignon/poster.py"
        _with_raw_ledger(guard, {name: guard.digest(_RAW_REQUESTS)})
        assert guard.find_stale_raw_entries([(name, _ROUTED)]) == [name], "paid"
        assert guard.find_raw_broken_seals([(name, _ROUTED)]) == []
        assert guard.find_stale_raw_entries([("opti_oignon/other.py", _ROUTED)]) == [name], "vanished"
        assert guard.find_stale_raw_entries([(name, _RAW_REQUESTS)]) == []
    finally:
        restore()


def test_rf22_on_the_real_tree_the_raw_ledger_names_exactly_the_modules_that_post():
    guard, restore = _load()
    try:
        files = dict(_real_files())
        assert sorted(guard.RAW_LEDGER) == sorted(_RAW_OWED)
        total = 0
        for name in _RAW_OWED:
            assert guard.RAW_LEDGER[name] == hashlib.sha256(files[name].encode("utf-8")).hexdigest(), (
                f"{name} is sealed on the text the census reads"
            )
            n = guard.count_raw_sites(files[name])
            assert n > 0, f"{name} is owed for, so the census finds a site in it"
            total += n
        assert total >= 9, "the debt the widening found, not zero"
        outside = {n: guard.count_raw_sites(t) for n, t in files.items() if n not in guard.RAW_LEDGER}
        assert sum(outside.values()) == 0, {k: v for k, v in outside.items() if v}
        assert guard.count_raw_sites(files["opti_oignon/project_triggers.py"]) == 0, "the paid module posts no more"
    finally:
        restore()


def test_rf23_the_entry_point_names_the_raw_debt_and_refuses_an_unowed_raw_site(tmp_path, capsys):
    guard, restore = _load()
    try:
        assert guard.main(["guard", str(REPO)]) == 0
        out = capsys.readouterr().out
        sites = sum(guard.count_raw_sites(t) for n, t in _real_files() if n in guard.RAW_LEDGER)
        assert f"{len(_RAW_OWED)} module(s) owed" in out and f"{sites} raw site(s)" in out, out
        pkg = tmp_path / "opti_oignon"
        pkg.mkdir()
        (pkg / "poster.py").write_text(_RAW_REQUESTS, encoding="utf-8")
        assert guard.main(["guard", str(tmp_path)]) == 1
        refused = capsys.readouterr().out
        assert "opti_oignon/poster.py" in refused and "HTTP" in refused, refused
    finally:
        restore()


_RAW_OWED_AFTER_EMBEDDINGS = (
    "opti_oignon/redteam/generator.py",
    "opti_oignon/redteam/strategies.py",
    "opti_oignon/redteam/targets.py",
    "opti_oignon/ui.py",
)


def test_rf24_the_raw_ledger_names_the_four_modules_that_still_post():
    guard, restore = _load()
    try:
        files = dict(_real_files())
        assert sorted(guard.RAW_LEDGER) == sorted(_RAW_OWED_AFTER_EMBEDDINGS)
        total = 0
        for name in _RAW_OWED_AFTER_EMBEDDINGS:
            assert guard.RAW_LEDGER[name] == hashlib.sha256(files[name].encode("utf-8")).hexdigest(), (
                f"{name} is sealed on the text the census reads"
            )
            n = guard.count_raw_sites(files[name])
            assert n > 0, f"{name} is owed for, so the census finds a site in it"
            total += n
        assert total >= 5, "the debt that remains, not zero"
        outside = {n: guard.count_raw_sites(t) for n, t in files.items() if n not in guard.RAW_LEDGER}
        assert sum(outside.values()) == 0, {k: v for k, v in outside.items() if v}
        assert guard.count_raw_sites(files["opti_oignon/project_triggers.py"]) == 0, "the paid module posts no more"
        assert guard.count_raw_sites(files["opti_oignon/rag/embeddings.py"]) == 0, "the embedder posts no more"
    finally:
        restore()


def test_rf25_the_entry_point_names_the_four_and_refuses_an_unowed_raw_site(tmp_path, capsys):
    guard, restore = _load()
    try:
        assert guard.main(["guard", str(REPO)]) == 0
        out = capsys.readouterr().out
        sites = sum(guard.count_raw_sites(t) for n, t in _real_files() if n in guard.RAW_LEDGER)
        assert f"{len(_RAW_OWED_AFTER_EMBEDDINGS)} module(s) owed" in out and f"{sites} raw site(s)" in out, out
        pkg = tmp_path / "opti_oignon"
        pkg.mkdir()
        (pkg / "poster.py").write_text(_RAW_REQUESTS, encoding="utf-8")
        assert guard.main(["guard", str(tmp_path)]) == 1
        refused = capsys.readouterr().out
        assert "opti_oignon/poster.py" in refused and "HTTP" in refused, refused
    finally:
        restore()


def test_rf26_the_raw_ledger_is_empty_and_the_launcher_is_the_one_named_exemption():
    guard, restore = _load()
    try:
        files = dict(_real_files())
        assert guard.RAW_LEDGER == {}, "the raw debt is paid"
        assert sorted(guard.RAW_EXEMPT) == ["opti_oignon/ui.py"]
        reason = guard.RAW_EXEMPT["opti_oignon/ui.py"]
        assert isinstance(reason, str) and "not an inference request" in reason, "an exemption carries its reason"
        assert guard.count_raw_sites(files["opti_oignon/ui.py"]) > 0, (
            "control: the exempted module still spells a site, so the exemption is not decoration"
        )
        outside = {n: guard.count_raw_sites(t) for n, t in files.items() if n not in guard.RAW_EXEMPT}
        assert sum(outside.values()) == 0, {k: v for k, v in outside.items() if v}
        for name in ("opti_oignon/redteam/generator.py", "opti_oignon/redteam/strategies.py", "opti_oignon/redteam/targets.py"):
            assert guard.count_raw_sites(files[name]) == 0, f"{name} posts no more"
    finally:
        restore()


def test_rf27_an_exemption_that_stops_posting_is_stale_and_the_entry_point_names_it(tmp_path, capsys):
    guard, restore = _load()
    try:
        name = "opti_oignon/probe.py"
        guard.RAW_EXEMPT = {name: "a liveness probe, not an inference request"}
        assert guard.find_raw_violations([(name, _RAW_URLLIB)]) == [], "an exempt module that posts is not a violation"
        assert guard.find_stale_raw_exemptions([(name, _RAW_URLLIB)]) == []
        assert guard.find_stale_raw_exemptions([(name, _ROUTED)]) == [name], "an exemption that no longer posts is stale"
        assert guard.find_stale_raw_exemptions([]) == [name], "and so is one whose module vanished"
    finally:
        restore()

    guard, restore = _load()
    try:
        assert guard.main(["guard", str(REPO)]) == 0
        out = capsys.readouterr().out
        assert "0 module(s) owed" in out and "1 exempt" in out and "opti_oignon/ui.py" in out, out
        pkg = tmp_path / "opti_oignon"
        pkg.mkdir()
        (pkg / "poster.py").write_text(_RAW_REQUESTS, encoding="utf-8")
        (pkg / "ui.py").write_text(_RAW_URLLIB, encoding="utf-8")
        assert guard.main(["guard", str(tmp_path)]) == 1
        refused = capsys.readouterr().out
        assert "opti_oignon/poster.py" in refused and "HTTP" in refused, refused
    finally:
        restore()


_RETURNED = "import ollama\n\ndef _t():\n    return ollama\n\ndef ask():\n    return _t().chat(model='m', messages=[])\n"
_RETURNED_METHOD = (
    "import ollama as _o\n\nclass B:\n    def _transport(self, timeout=None):\n"
    "        return _o if timeout is None else _o.Client(timeout=timeout)\n\n"
    "    def names(self):\n        return self._transport().list()\n"
)


def test_rf28_the_census_follows_the_client_through_a_function_that_returns_it():
    guard, restore = _load()
    try:
        assert guard.count_sites(_RETURNED) == 1, "a request on what a helper returns is a site"
        assert guard.count_sites(_RETURNED_METHOD) == 2, "a method returning the client, and the client it builds"
        assert guard.count_sites(_ROUTED) == 0
        files = dict(_real_files())
        assert guard.count_sites(files["opti_oignon/inference_backend.py"]) >= 9, (
            "control: the funnel's heads, reached through its transport, are seen"
        )
        outside = {n: guard.count_sites(t) for n, t in files.items() if n != "opti_oignon/inference_backend.py"}
        assert sum(outside.values()) == 0, {k: v for k, v in outside.items() if v}
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF29-RF30 -- Ollama's cloud search and fetch
# ---------------------------------------------------------------------------
_FUNNEL_NAME = "opti_oignon/inference_backend.py"


def _modules():
    """The estate as the guard reads it, walked without listing the package's data directory.

    The data directory holds no module, and a walk into it is a path the
    test session's firewall has to keep off the maintainer's data.
    """
    data = _PACKAGE / "data"
    out = []
    for dirpath, dirnames, filenames in os.walk(_PACKAGE):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if here / d != data and d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                path = here / name
                out.append((path.relative_to(REPO).as_posix(), path.read_text(encoding="utf-8", errors="ignore")))
    return out
_CLOUD_CALL = "import ollama\n\ndef look(q):\n    return ollama.web_search(q)\n"
_CLOUD_CLIENT = "import ollama\n\ndef fetch(u):\n    c = ollama.Client()\n    return c.web_fetch(u)\n"
_CLOUD_UNCALLED = "from ollama import web_search as ws\n\nHANDLER = ws\n"
_CLOUD_ATTRIBUTE = (
    "import ollama\n\nclass C:\n    def __init__(self, c=None):\n        self._c = c or ollama\n\n"
    "    def fetch(self, u):\n        return self._c.web_fetch(u)\n"
)
_CLOUD_COLLISION = "from opti_oignon import web_search\n\ndef look(q):\n    return web_search.search(q)\n"
_CLOUD_LOCAL_DEF = "def web_search(q):\n    return []\n\ndef look(q):\n    return web_search(q)\n"
_CLOUD_PROSE = '"""Calls ollama.web_search and ollama.web_fetch."""\n\ndef look(q):\n    return None\n'
_SUB_FROM = "from ollama._client import Client\n\ndef look(q):\n    return Client().web_search(q)\n"
_SUB_AS = "import ollama._client as oc\n\ndef fetch(u):\n    return oc.Client().web_fetch(u)\n"
_SUB_DOTTED = "import ollama._client\n\ndef make():\n    return ollama._client.Client()\n"
_SUB_NAME = "from ollama import _client\n\ndef make():\n    return _client.Client()\n"
_SUB_ALONE = "import ollama._types\n\nX = 1\n"
_DYN_IMPORT_MODULE = (
    "import importlib\n\ndef look(q):\n    o = importlib.import_module('ollama')\n    return o.web_search(q)\n"
)
_DYN_DUNDER = "def fetch(u):\n    return __import__('ollama').web_fetch(u)\n"
_DYN_GETATTR = "import ollama\n\ndef look(q):\n    return getattr(ollama, 'web_search')(q)\n"
_DYN_GETATTR_NAME = "import ollama\n\ndef look(name, q):\n    return getattr(ollama, name)(q)\n"
_DYN_GETATTR_PULL = "import ollama\n\ndef fetch(m):\n    return getattr(ollama, 'pull')(m)\n"
_DYN_GETATTR_OTHER = "def look(obj, q):\n    return getattr(obj, 'web_search')(q)\n"
_IMPORT_SPELLINGS = {
    "a star import": "from ollama import *  # noqa: F403\n\ndef look(q):\n    return Client().web_search(q)\n",
    "a star import, the async client": "from ollama import *  # noqa: F403\n\ndef f(u):\n    return AsyncClient().web_fetch(u)\n",
    "importlib.__import__": "import importlib\n\nimportlib.__import__('ollama').web_search('q')\n",
    "builtins.__import__": "import builtins\n\nbuiltins.__import__('ollama').web_fetch('u')\n",
    "a renamed import_module": "from importlib import import_module as load\n\nload('ollama').web_search('q')\n",
    "a renamed __import__": "from builtins import __import__ as load\n\nload('ollama').web_fetch('u')\n",
    "a keyword argument": "import importlib\n\nimportlib.import_module(name='ollama').web_search('q')\n",
    "a sys.modules lookup": "import sys\n\nsys.modules['ollama'].web_search('q')\n",
    "a sys.modules get": "import sys\n\nsys.modules.get('ollama').web_fetch('u')\n",
}
_IMPORT_OTHER_PACKAGE = "import importlib\n\nimportlib.import_module('json').dumps({})\n"
_BINDING_SPELLINGS = {
    "an unpacking": "import ollama\n\na, b = ollama, 1\na.web_search('q')\n",
    "a walrus": "import ollama\n\n(c := ollama).web_search('q')\n",
    "a for target": "import ollama\n\nfor c in (ollama,):\n    c.web_search('q')\n",
    "a comprehension target": "import ollama\n\n[c.web_fetch('u') for c in [ollama]]\n",
    "a with target": (
        "import ollama\nfrom contextlib import nullcontext\n\n"
        "with nullcontext(ollama) as c:\n    c.web_search('q')\n"
    ),
    "a parameter default": "import ollama\n\ndef f(c=ollama):\n    return c.web_search('q')\n",
    "a lambda that returns it": "import ollama\n\nf = lambda: ollama\nf().web_search('q')\n",
    "a lambda parameter": "import ollama\n\n(lambda m: m.web_search('q'))(ollama)\n",
    "an argument to a function of the module": (
        "import ollama\n\ndef f(m):\n    return m.web_search('q')\n\nf(ollama)\n"
    ),
    "an argument to a method of the module": (
        "import ollama\n\nclass K:\n    def f(self, m):\n        return m.web_fetch('u')\n\nK().f(ollama)\n"
    ),
    "a list element": "import ollama\n\nd = [ollama]\nd[0].web_fetch('u')\n",
    "a mapping value": "import ollama\n\nd = {'c': ollama}\nd['c'].web_search('q')\n",
    "a class attribute": "import ollama\n\nclass H:\n    client = ollama\n\nH.client.web_search('q')\n",
    "a class attribute read on self": (
        "import ollama\n\nclass K:\n    c = ollama\n\n    def f(self):\n        return self.c.web_search('q')\n"
    ),
    "a class of the module given the client": (
        "import ollama\n\nclass W:\n    def __init__(self, c):\n        self.c = c\n\n"
        "    def ask(self, q):\n        return self.c.web_search(q)\n\nW(ollama).ask('q')\n"
    ),
    "an instance wrapping the client": "import ollama\n\nclass W:\n    pass\n\nw = W(ollama)\nw.web_search('q')\n",
    "a cast": "import ollama\nfrom typing import Any, cast\n\nc = cast(Any, ollama)\nc.web_search('q')\n",
    "a comprehension that yields it": "import ollama\n\nclients = [c for c in [ollama]]\nclients[0].web_fetch('u')\n",
}
_NOT_THE_CLIENT = {
    "what a request returns": (
        "import ollama\n\ndef f(m, key):\n    for chunk in ollama.pull(m, stream=True):\n"
        "        getattr(chunk, key)\n"
    ),
    "a context manager given something else": (
        "from contextlib import nullcontext\n\nwith nullcontext(1) as c:\n    c.web_search('q')\n"
    ),
}
_CHAIN_SPELLINGS = {
    "getattr of a private name": "import ollama\n\ngetattr(ollama, '_client').web_search('q')\n",
    "a chain on import_module": "import importlib\n\nimportlib.import_module('ollama')._client.web_search('q')\n",
    "a chain on __import__": "__import__('ollama')._client.web_fetch('u')\n",
    "a chain on a name bound to an import": (
        "import importlib\n\no = importlib.import_module('ollama')\no._client.web_search('q')\n"
    ),
    "a chain on a name bound to the module": "import ollama\n\nc = ollama\nc._client.web_search('q')\n",
}


def test_rf29_the_cloud_search_and_fetch_are_sites_in_every_spelling_the_census_knows():
    guard, restore = _load()
    try:
        # c1 -- the module, a client, an imported name, a bound receiver.
        assert guard.count_sites(_CLOUD_CALL) == 1, "a cloud search on the module is a site"
        assert guard.count_sites(_CLOUD_CLIENT) == 2, "the client class, and the fetch made on it"
        assert guard.count_sites(_CLOUD_UNCALLED) == 1, "an imported search handed on uncalled"
        assert guard.count_sites(_CLOUD_ATTRIBUTE) == 1, "a fetch through a bound receiver"
        assert guard.count_sites(_CLOUD_COLLISION) == 0, "the application's own web_search module is not the client"
        assert guard.count_sites(_CLOUD_LOCAL_DEF) == 0, "a local function of the same name is not the client"
        assert guard.count_sites(_CLOUD_PROSE) == 0, "prose never counts"

        # c2 -- through the client's submodules.
        assert guard.count_sites(_SUB_FROM) == 1, "a class imported from a submodule"
        assert guard.count_sites(_SUB_AS) == 1, "a submodule bound to an alias"
        assert guard.count_sites(_SUB_DOTTED) == 1, "a dotted submodule import binds the package"
        assert guard.count_sites(_SUB_NAME) == 1, "a submodule imported by name from the package"
        assert guard.count_sites(_SUB_ALONE) == 0, "a submodule import that reaches nothing counted is not a site"

        # c3 -- dynamic access with a constant.
        assert guard.count_sites(_DYN_IMPORT_MODULE) == 1, "import_module with a constant returns the client"
        assert guard.count_sites(_DYN_DUNDER) == 1, "__import__ with a constant returns the client"
        assert guard.count_sites(_DYN_GETATTR) == 1, "getattr with a counted name"
        assert guard.count_sites(_DYN_GETATTR_NAME) == 1, "getattr with a name that cannot be read is charged"
        assert guard.count_sites(_DYN_GETATTR_PULL) == 0, "model management stays uncounted through getattr"
        assert guard.count_sites(_DYN_GETATTR_OTHER) == 0, "getattr on an unrelated object is not the client"

        # c4 -- the sets, and a violation by name.
        assert {"web_search", "web_fetch"} <= set(guard._CLIENT_CALLS)
        assert guard.find_violations([("opti_oignon/cloud.py", _CLOUD_CALL)]) == ["opti_oignon/cloud.py"]

        # c5 -- the real tree: nothing outside the funnel reaches them, and
        # the funnel's own count owes nothing to the two names.
        files = dict(_modules())
        outside = {n: guard.count_sites(t) for n, t in files.items() if n != _FUNNEL_NAME}
        assert len(outside) > 300, len(outside)
        assert sum(outside.values()) == 0, {k: v for k, v in outside.items() if v}
        funnel = files[_FUNNEL_NAME]
        full = guard.count_sites(funnel)
        assert full >= 9, "control: the funnel's heads are seen"
        guard._CLIENT_CALLS = frozenset(guard._CLIENT_CALLS) - {"web_search", "web_fetch"}
        assert guard.count_sites(funnel) == full, "the funnel reaches neither cloud method"
        spelled = [
            node.attr for node in ast.walk(ast.parse(funnel))
            if isinstance(node, ast.Attribute) and node.attr in ("web_search", "web_fetch")
        ]
        assert spelled == [], spelled
    finally:
        restore()

    # A fresh guard: the clause above narrowed the counted names.
    guard, restore = _load()
    try:
        # c6 -- every way the package is imported by name.
        for label, text in _IMPORT_SPELLINGS.items():
            assert guard.count_sites(text) >= 1, label
        assert guard.count_sites(_IMPORT_OTHER_PACKAGE) == 0, "import_module of another package is not the client"

        # c7 -- every way a name is bound to the client.
        for label, text in _BINDING_SPELLINGS.items():
            assert guard.count_sites(text) >= 1, label
        for label, text in _NOT_THE_CLIENT.items():
            assert guard.count_sites(text) == 0, label

        # c8 -- an attribute chain rooted at an import call, at a name
        # assigned from one, or at what getattr hands back.
        for label, text in _CHAIN_SPELLINGS.items():
            assert guard.count_sites(text) >= 1, label
    finally:
        restore()


_RAW_CLOUD = (
    "import httpx\n\ndef look(q):\n"
    "    return httpx.post('https://ollama.com/api/web_search', json={'query': q})\n"
)
_RAW_CLOUD_FSTRING = (
    "import requests\n\ndef fetch(host, u):\n"
    "    return requests.post(f'{host}/api/web_fetch', json={'url': u})\n"
)
_RAW_CLOUD_APP_ROUTE = (
    "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n"
    "@router.get('/api/search/config')\ndef config():\n    return {}\n"
)
_RAW_CLOUD_LONGER = "import httpx\n\ndef h(u):\n    return httpx.get(f'{u}/api/web_search/help')\n"
_RAW_NO_TRANSPORT_HOST = "URL = 'https://ollama.com/api/web_fetch'\n"
_RAW_NO_TRANSPORT_PATH = "def target(host):\n    return f'{host}/api/web_fetch'\n"
_RAW_URLLIB3 = "import urllib3\n\nURL = 'https://ollama.com/api/web_fetch'\n"
_RAW_NO_TRANSPORT_LOCAL = "PATH = '/api/chat'\n"
_RAW_FRAGMENT = "U = 'http://127.0.0.1:11434/api/web_search#x'\n"
_RAW_SPACE = "U = 'http://127.0.0.1:11434/api/web_fetch '\n"
_RAW_PERCENT = "U = 'http://127.0.0.1:11434/api/web%5Fsearch'\n"
_RAW_HOST_PIECES = "BASE = 'https://ollama.com/api/' + 'web_search'\n"
_RAW_HOST_PROSE = '"""Talks to ollama.com when asked."""\n\nX = 1\n'
# Non-ASCII characters are built, never typed.
_FULL_STOP = chr(0x3002)
_RAW_ROUTED_FORMS = {
    "a bytes cloud path": "import requests\n\nrequests.post(b'http://127.0.0.1:11434/api/web_search')\n",
    "a bytes cloud host": "U = b'https://ollama.com/api/'\n",
    "dot segments": "import requests\n\nrequests.post('http://127.0.0.1:11434/api/x/../web_search')\n",
    "a dot segment": "import httpx\n\nhttpx.post('https://example.org/api/./web_fetch')\n",
    "a repeated slash": "U = '/api//web_search'\n",
    "a Unicode full stop in the host": "import httpx\n\nhttpx.post('https://ollama" + _FULL_STOP + "com/api/x')\n",
}
_RAW_BYTES_LOCAL = "import requests\n\nrequests.post(b'http://127.0.0.1:11434/api/chat')\n"
_RAW_LOCAL_DOTS = "import requests\n\nrequests.post('http://127.0.0.1:11434/api/x/../chat')\n"
_RAW_BYTES_NO_TRANSPORT = "P = b'/api/chat'\n"
_RAW_CLOUD_BASES = {
    "a base URL": "BASE = 'https://ollama.com'\n",
    "the host alone": "HOST = 'ollama.com'\n",
    "a subdomain under /api": "B = 'https://www.ollama.com/api/'\n",
    "the compatible API": "B = 'https://ollama.com/v1/chat/completions'\n",
    "a capitalised host": "import requests\n\nrequests.post('https://OLLAMA.COM/api/x')\n",
    "a URL inside a sentence": "MSG = 'posting to https://ollama.com/api/web_search now'\n",
}
_RAW_CLOUD_LINKS = {
    "a download link in a message": "MSG = 'Install Ollama from https://ollama.com/download'\n",
    "a library page": "L = 'https://ollama.com/library/llama3'\n",
    "the host named in prose": "MSG = 'see ollama.com for details'\n",
    "another host's search path": "import httpx\n\nhttpx.get('https://example.org/api/web_search/help')\n",
}
_RAW_CLOUD_IN_LAUNCHER = (
    "\n\ndef cloud(q):\n    import requests\n"
    "    return requests.post('https://ollama.com/api/web_search', json={'query': q})\n"
)


def test_rf30_the_cloud_paths_and_host_are_raw_sites_and_the_entry_point_refuses_them(tmp_path, capsys):
    guard, restore = _load()
    try:
        # c1 -- the cloud endpoints, spelled with a transport.
        assert guard.count_raw_sites(_RAW_CLOUD) == 1, "a cloud search posted with httpx"
        assert guard.count_raw_sites(_RAW_CLOUD_FSTRING) == 1, "a cloud fetch in an f-string behind requests"
        assert guard.count_raw_sites(_RAW_CLOUD_APP_ROUTE) == 0, "the application's search route is another path"
        assert guard.count_raw_sites(_RAW_CLOUD_LONGER) == 0, "a path that only begins with the endpoint"
        assert {"/api/web_search", "/api/web_fetch"} <= set(guard._RAW_ENDPOINTS)

        # c2 -- without a transport the cloud paths still count; the local
        # rule is unchanged.
        assert guard.count_raw_sites(_RAW_NO_TRANSPORT_HOST) == 1, "the client posts with a transport of its own"
        assert guard.count_raw_sites(_RAW_NO_TRANSPORT_PATH) == 1, "a cloud path needs no transport to count"
        assert guard.count_raw_sites(_RAW_URLLIB3) == 1, "a transport the list does not name"
        assert guard.count_raw_sites(_RAW_NO_TRANSPORT_LOCAL) == 0, "a local endpoint still needs a transport"

        # c3 -- a path is read without its fragment, spaces or percent encoding.
        assert guard.count_raw_sites(_RAW_FRAGMENT) == 1, "a fragment does not hide the path"
        assert guard.count_raw_sites(_RAW_SPACE) == 1, "a trailing space does not hide the path"
        assert guard.count_raw_sites(_RAW_PERCENT) == 1, "percent encoding does not hide the path"

        # c4 -- the cloud host counts in pieces, never in prose.
        assert guard.count_raw_sites(_RAW_HOST_PIECES) >= 1, "the host names the cloud"
        assert guard.count_raw_sites(_RAW_HOST_PROSE) == 0, "a docstring is prose"

        # c5 -- the funnel spells none of them.
        assert set(guard._CLOUD_ENDPOINTS) == {"/api/web_search", "/api/web_fetch"}
        assert guard._CLOUD_HOST == "ollama.com"
        funnel = _PACKAGE.joinpath("inference_backend.py").read_text(encoding="utf-8")
        tree = ast.parse(funnel)
        prose = guard._docstring_nodes(tree)
        constants = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose
        ]
        assert len(constants) > 50, len(constants)
        cloud = [
            c for c in constants
            if "ollama.com" in c or c.split("?")[0].split("#")[0].strip().rstrip("/").endswith(
                ("/api/web_search", "/api/web_fetch")
            )
        ]
        assert cloud == [], cloud
    finally:
        restore()

    # c6 -- the entry point refuses both kinds by name.
    guard, restore = _load()
    try:
        control = tmp_path / "control" / "opti_oignon"
        control.mkdir(parents=True)
        (control / "routed.py").write_text(_ROUTED, encoding="utf-8")
        (control / "ui.py").write_text(_RAW_URLLIB, encoding="utf-8")
        assert guard.main(["guard", str(control.parent)]) == 0, "control: the launcher alone is exempt"
        capsys.readouterr()

        client_tree = tmp_path / "client" / "opti_oignon"
        client_tree.mkdir(parents=True)
        (client_tree / "cloud.py").write_text(_CLOUD_CALL, encoding="utf-8")
        (client_tree / "ui.py").write_text(_RAW_URLLIB, encoding="utf-8")
        assert guard.main(["guard", str(client_tree.parent)]) == 1
        refused = capsys.readouterr().out
        heading = refused.find("reach the client")
        assert heading >= 0 and refused.find("opti_oignon/cloud.py", heading) > heading, refused

        raw_tree = tmp_path / "raw" / "opti_oignon"
        raw_tree.mkdir(parents=True)
        (raw_tree / "poster.py").write_text(_RAW_CLOUD, encoding="utf-8")
        (raw_tree / "ui.py").write_text(_RAW_URLLIB, encoding="utf-8")
        assert guard.main(["guard", str(raw_tree.parent)]) == 1
        refused = capsys.readouterr().out
        assert "opti_oignon/poster.py" in refused and "HTTP" in refused, refused

        # The repository, read as the guard reads it but without listing the
        # package's data directory, which holds no module.
        guard._estate = lambda root: _modules()
        assert guard.main(["guard", str(REPO)]) == 0
        assert "0 module(s) owed" in capsys.readouterr().out
    finally:
        restore()

    # c7 -- a path is read as a server would route it: bytes, dot segments,
    # repeated slashes, and a host written with a Unicode full stop.
    guard, restore = _load()
    try:
        for label, text in _RAW_ROUTED_FORMS.items():
            assert guard.count_raw_sites(text) == 1, label
        assert guard.count_raw_sites(_RAW_BYTES_LOCAL) == 1, "a bytes local endpoint behind a transport"
        assert guard.count_raw_sites(_RAW_LOCAL_DOTS) == 1, "a local endpoint behind dot segments"
        assert guard.count_raw_sites(_RAW_BYTES_NO_TRANSPORT) == 0, "a local endpoint still needs a transport"

        # c8 -- the cloud host counts as a place to post from, never as a
        # link to a page on it.
        for label, text in _RAW_CLOUD_BASES.items():
            assert guard.count_raw_sites(text) == 1, label
        for label, text in _RAW_CLOUD_LINKS.items():
            assert guard.count_raw_sites(text) == 0, label

        # c9 -- an exemption or a raw ledger entry excuses local endpoints
        # only: a cloud site in the excused module is refused by name, and
        # an exemption is stale on its local endpoints alone.
        exempt = "opti_oignon/ui.py"
        assert sorted(guard.RAW_EXEMPT) == [exempt]
        launcher = _RAW_URLLIB + _RAW_CLOUD_IN_LAUNCHER
        assert guard.find_raw_violations([(exempt, _RAW_URLLIB)]) == [], "control: the probe alone is exempt"
        assert guard.find_cloud_violations([(exempt, _RAW_URLLIB)]) == [], "control: no cloud site, nothing refused"
        assert guard.find_cloud_violations([(exempt, launcher)]) == [exempt], "an exemption does not cover the cloud"
        assert guard.find_stale_raw_exemptions([(exempt, _RAW_CLOUD_IN_LAUNCHER)]) == [exempt], (
            "an exemption that spells only the cloud no longer spells what it was granted for"
        )
        owed = "opti_oignon/owed.py"
        guard.RAW_LEDGER = {owed: guard.digest(_RAW_CLOUD)}
        assert guard.find_raw_violations([(owed, _RAW_CLOUD)]) == [], "the ledger answers for its local debt"
        assert guard.find_cloud_violations([(owed, _RAW_CLOUD)]) == [owed], "a ledger entry does not cover the cloud"
        guard.RAW_LEDGER = {}
        tree = tmp_path / "launcher" / "opti_oignon"
        tree.mkdir(parents=True)
        (tree / "ui.py").write_text(launcher, encoding="utf-8")
        (tree / "routed.py").write_text(_ROUTED, encoding="utf-8")
        capsys.readouterr()
        assert guard.main(["guard", str(tree.parent)]) == 1
        refused = capsys.readouterr().out
        heading = refused.find("Cloud search and fetch")
        assert heading >= 0 and refused.find(exempt, heading) > heading, refused
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF31-RF34 -- what the guard could not read fails it by name
# ---------------------------------------------------------------------------
def _fixture_estate(guard, tmp_path):
    """A one-module estate under ``tmp_path`` and ledgers that owe nothing."""
    _with_ledger(guard, {})
    _with_raw_ledger(guard, {})
    guard.RAW_EXEMPT = {}
    tree = tmp_path / "opti_oignon"
    tree.mkdir()
    (tree / "routed.py").write_text(_ROUTED, encoding="utf-8")
    return tree


def test_rf31_a_module_that_does_not_parse_fails_the_guard_by_name(tmp_path, capsys):
    guard, restore = _load()
    try:
        tree = _fixture_estate(guard, tmp_path)
        assert guard.main(["guard", str(tmp_path)]) == 0, "control: the estate parses and passes"
        capsys.readouterr()
        (tree / "broken.py").write_text("def ask(:\n" + _DIRECT, encoding="utf-8")
        assert guard.main(["guard", str(tmp_path)]) == 1, (
            "a module whose sites cannot be counted is not a module without sites"
        )
        assert "opti_oignon/broken.py" in capsys.readouterr().out
    finally:
        restore()


def test_rf32_a_module_that_is_not_utf8_fails_the_guard_by_name(tmp_path, capsys):
    guard, restore = _load()
    try:
        tree = _fixture_estate(guard, tmp_path)
        (tree / "latin.py").write_bytes(b"# caf\xe9\nVALUE = 1\n")
        assert guard.main(["guard", str(tmp_path)]) == 1, (
            "a module read with its bytes dropped is not the module on disk"
        )
        assert "opti_oignon/latin.py" in capsys.readouterr().out
    finally:
        restore()


def test_rf33_a_directory_the_walk_cannot_list_fails_the_guard_by_name(tmp_path, capsys):
    guard, restore = _load()
    locked = None
    try:
        tree = _fixture_estate(guard, tmp_path)
        locked = tree / "sub"
        locked.mkdir()
        (locked / "hidden.py").write_text(_DIRECT, encoding="utf-8")
        os.chmod(locked, 0)
        assert guard.main(["guard", str(tmp_path)]) == 1, (
            "a directory the walk cannot list is not a directory without modules"
        )
        assert "opti_oignon/sub" in capsys.readouterr().out
    finally:
        if locked is not None:
            os.chmod(locked, 0o755)
        restore()


def test_rf34_the_package_data_directory_is_never_listed(tmp_path, capsys):
    guard, restore = _load()
    data = None
    try:
        tree = _fixture_estate(guard, tmp_path)
        data = tree / "data"
        data.mkdir()
        (data / "stray.py").write_text(_DIRECT, encoding="utf-8")
        os.chmod(data, 0)
        assert guard.main(["guard", str(tmp_path)]) == 0, (
            "the maintainer's data directory holds no module and is pruned before listing"
        )
        assert "1 module(s) scanned" in capsys.readouterr().out
    finally:
        if data is not None:
            os.chmod(data, 0o755)
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
