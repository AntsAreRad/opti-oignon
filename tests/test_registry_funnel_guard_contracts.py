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

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window.
"""

import hashlib
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
