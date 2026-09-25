#!/usr/bin/env python3
"""Contracts for the native core: the onion's integrity primitives and its
composer assembly in Rust, behind the Python surface, by strangling.

Python stays the reference and the fallback. The native module is asked
for lazily, at the call, never at import; when it is present its answers
must be the reference's, byte for byte on every hash and field for field
on every assembled prompt; when it is absent nothing changes. The recipe
that builds it is pinned in the tree so the artefact is reproducible, and
the artefact itself is never tracked.

  * NC1 -- the native module builds and loads through the package's loader
    and exposes the surface the Python modules ask for.
  * NC2 -- the four hashes are byte-equal to the reference on fixtures with
    Unicode, escapes, unordered keys and non-string values.
  * NC3 -- the composer assembles the same prompt with the native module
    as without it, and refuses with the same words.
  * NC4 -- with the native module unreachable the reference stands, and no
    onion module reaches for it at import.
  * NC5 -- the recipe is pinned: the crate, its lock, the build script and
    the ignore rule are in the tree, and the loader does not load at import.
  * NC6 -- the native probe generator draws, probe for probe, what the
    reference draws, and the Python surface goes through it.
  * NC7 -- the native scorer fails the probes the reference fails, in the
    same order; a probe shape it does not take goes to the reference,
    which raises what it raises.
  * NC8 -- the probes' scope is exact: every code point the core accepts
    is classed as Python classes it, the accepted set holds ASCII and
    French, and a text carrying any other code point is refused by the
    core and answered by the reference.
  * NC9 -- a regular expression the core does not implement sends the call
    to the reference: the patterns travel with every call.
  * NC10 -- the native probes answer as the reference on French decisions
    and negations, on the typographic apostrophe, and on the two letters
    Python folds beyond ASCII when case is ignored: the dotless i and the
    long s.
  * NC11 -- the build script installs the artefact by rename: a process
    that holds the old one keeps its bytes, and no staging file is left.
  * NC12 -- the build script checks the artefact of its own tree, whatever
    directory it is run from.
  * NC13 -- the native probes draw French names and keep French words whole
    as the reference does, and every accepted code point at the head of a
    name is a capital for the core exactly when Python calls it one.

NC1 to NC3, NC6 to NC10 and NC13 need the built artefact: the CI job that
carries a Rust toolchain builds it and runs this file by name, and a local
sweep needs ``scripts/build_oo_core.sh`` run once. NC4, NC5, NC11 and NC12
always run: the last two drive a copy of the script with a stand-in cargo
and a stand-in loader.

Local-only (the public distribution ships no tests).
"""

import ast
import collections
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_ONION = ("composer", "core_store", "receipts", "peels")
_CRATE = REPO / "rust" / "oo_core"


def _open(*, native):
    targets = {f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _ONION}
    targets["opti_oignon.memory.probes"] = source("memory", "probes.py")
    if native:
        targets["opti_oignon.native"] = source("native", "__init__.py")
        blocked = ()
    else:
        blocked = ("opti_oignon.native",)
    loaded, restore = isolate(targets=targets, blocked=blocked, packages=("opti_oignon.memory",))
    return loaded, restore


def _fixtures():
    return {
        "texts": ["plain", "Ünïcode — ça va? 日本語", 'quotes " and \\ backslash', "line\nbreak\ttab\x01ctl", "", "  spaces  "],
        "spans": [
            [{"turn_id": "t0001", "role": "user", "text": "Ünïcode — ça"}, {"text": "b", "role": "assistant", "turn_id": "t0002"}],
            [{"z": 1, "a": True, "m": None, "nested": {"y": [1, 2, "x"], "b": "é"}}],
            [],
            [{"text": 'esc "q" \\ \n \x1f \x7f'}],
        ],
        "sources": [("k1", "k2"), (), ("only",), ("Ünï", 'q"uote')],
    }


def _state(loaded, *, peels=3):
    composer = loaded["opti_oignon.memory.composer"]
    core_store = loaded["opti_oignon.memory.core_store"]
    receipts = loaded["opti_oignon.memory.receipts"]
    core = core_store.CoreStore()
    core.add("The user is called Alice.", actor=core_store.USER)
    core.add("Answers are concise and in English.", actor=core_store.USER)
    cellar, ledger = receipts.Cellar(), receipts.ReceiptLedger()
    flesh = receipts.Flesh([{"turn_id": f"t{i:02d}", "role": "user", "text": f"Turn {i} says that service {i} moved to the new cluster."} for i in range(1, 7)])
    flesh.evict_oldest(cellar, ledger)
    flesh.evict_oldest(cellar, ledger)
    retrieval = [composer.Peel(text=f"Episode {i}: the migration of service {i} was reviewed.", provenance=f"peel:{i}") for i in range(1, peels + 1)]
    budget = composer.Budget(window=600, reserve=60, core=60, receipts=60, peels=120, flesh=200, turn=100)
    return dict(core=core, ledger=ledger, cellar=cellar, retrieval=retrieval, flesh=flesh, turn="Which service moved last?", budget=budget)


# ---------------------------------------------------------------------------
# NC1 -- builds and loads
# ---------------------------------------------------------------------------
def test_nc1_the_native_module_loads_and_exposes_the_surface():
    loaded, restore = _open(native=True)
    try:
        native = loaded["opti_oignon.native"]
        core = native.load()
        assert core is not None, "oo_core is not built: run scripts/build_oo_core.sh on this machine"
        for name in ("entry_hash", "span_key", "peel_id", "digest_spans", "compose_segments", "VERSION"):
            assert hasattr(core, name), f"the native surface carries {name}"
        assert isinstance(core.VERSION, str) and core.VERSION
        assert native.available() is True
        assert native.load() is core, "loaded once, then cached"
    finally:
        restore()


# ---------------------------------------------------------------------------
# NC2 -- hashes byte-equal
# ---------------------------------------------------------------------------
def test_nc2_the_four_hashes_are_byte_equal_to_the_reference():
    loaded, restore = _open(native=True)
    try:
        core = loaded["opti_oignon.native"].load()
        assert core is not None, "oo_core is not built: run scripts/build_oo_core.sh on this machine"
        fx = _fixtures()
        for text in fx["texts"]:
            assert core.entry_hash(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()
        for span in fx["spans"]:
            expected = hashlib.sha256(json.dumps(list(span), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
            assert core.span_key(span) == expected, f"span_key on {span!r}"
        for text in fx["texts"]:
            for sources in fx["sources"]:
                expected = hashlib.sha256(json.dumps([text, list(sources)], ensure_ascii=False).encode("utf-8")).hexdigest()
                assert core.peel_id(text, list(sources)) == expected, f"peel_id on {text!r} {sources!r}"
        expected = hashlib.sha256(json.dumps([list(s) for s in fx["spans"]], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        assert core.digest_spans(fx["spans"]) == expected
        assert core.digest_spans([]) == hashlib.sha256(b"[]").hexdigest()
        with pytest.raises(Exception):
            core.span_key([{"x": 1.5}])
    finally:
        restore()


# ---------------------------------------------------------------------------
# NC3 -- the composer agrees with itself
# ---------------------------------------------------------------------------
def test_nc3_the_composer_assembles_the_same_prompt_with_and_without_the_native_module():
    with_native, restore_native = _open(native=True)
    try:
        assert with_native["opti_oignon.native"].load() is not None, "oo_core is not built: run scripts/build_oo_core.sh on this machine"
        composer = with_native["opti_oignon.memory.composer"]
        assert composer._native_compose() is not None, "the composer sees the native module"
        native_prompt = composer.compose(**_state(with_native, peels=40))
        state = _state(with_native)
        b = state["budget"]
        refusals = {}
        for name, budget in (
            ("core", composer.Budget(window=b.window, reserve=b.reserve, core=1, receipts=b.receipts, peels=b.peels, flesh=b.flesh, turn=b.turn + b.core - 1)),
            ("flesh", composer.Budget(window=b.window, reserve=b.reserve, core=b.core, receipts=b.receipts, peels=b.peels + b.flesh - 1, flesh=1, turn=b.turn)),
        ):
            with pytest.raises(composer.BudgetError) as caught:
                composer.compose(**{**state, "budget": budget})
            refusals[name] = str(caught.value)
    finally:
        restore_native()
    without, restore = _open(native=False)
    try:
        composer = without["opti_oignon.memory.composer"]
        assert composer._native_compose() is None, "control: the reference path"
        reference = composer.compose(**_state(without, peels=40))
        plain = lambda prompt: [(s.layer, s.text, s.provenance, s.tokens, s.instruction_bearing) for s in prompt.segments]  # noqa: E731
        assert plain(native_prompt) == plain(reference), "field for field, across two windows"
        assert len(plain(reference)) >= 5
        assert native_prompt.tokens == reference.tokens and native_prompt.dropped_peels == reference.dropped_peels
        assert native_prompt.core_root == reference.core_root
        assert native_prompt.render() == reference.render()
        state = _state(without)
        b = state["budget"]
        for name, budget in (
            ("core", composer.Budget(window=b.window, reserve=b.reserve, core=1, receipts=b.receipts, peels=b.peels, flesh=b.flesh, turn=b.turn + b.core - 1)),
            ("flesh", composer.Budget(window=b.window, reserve=b.reserve, core=b.core, receipts=b.receipts, peels=b.peels + b.flesh - 1, flesh=1, turn=b.turn)),
        ):
            with pytest.raises(composer.BudgetError) as caught:
                composer.compose(**{**state, "budget": budget})
            assert str(caught.value) == refusals[name], f"{name}: the refusal is the same sentence"
    finally:
        restore()


# ---------------------------------------------------------------------------
# NC4 -- the reference stands without the native module
# ---------------------------------------------------------------------------
def test_nc4_without_the_native_module_the_reference_stands_and_nothing_imports_it_at_scope():
    loaded, restore = _open(native=False)
    try:
        core_store = loaded["opti_oignon.memory.core_store"]
        receipts = loaded["opti_oignon.memory.receipts"]
        peels = loaded["opti_oignon.memory.peels"]
        composer = loaded["opti_oignon.memory.composer"]
        fx = _fixtures()
        assert core_store.entry_hash(fx["texts"][1]) == hashlib.sha256(fx["texts"][1].encode("utf-8")).hexdigest()
        assert receipts.span_key(fx["spans"][0]) == hashlib.sha256(json.dumps(fx["spans"][0], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        assert peels.peel_id("t", ["a"]) == hashlib.sha256(json.dumps(["t", ["a"]], ensure_ascii=False).encode("utf-8")).hexdigest()
        assert composer._native_compose() is None
        prompt = composer.compose(**_state(loaded))
        assert [s.layer for s in prompt.segments][0] == "core" and prompt.tokens >= 1
    finally:
        restore()
    for name in _ONION:
        tree = ast.parse(source("memory", f"{name}.py").read_text(encoding="utf-8"))
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {a.name for n in top if isinstance(n, ast.Import) for a in n.names} | {("." * n.level) + (n.module or "") for n in top if isinstance(n, ast.ImportFrom)}
        assert not any("native" in n for n in names), f"{name}.py asks for the native module at the call, never at import: {names}"


# ---------------------------------------------------------------------------
# NC5 -- the recipe is pinned
# ---------------------------------------------------------------------------
def test_nc5_the_recipe_is_pinned_and_the_loader_loads_nothing_at_import():
    manifest = (_CRATE / "Cargo.toml").read_text(encoding="utf-8")
    assert re.search(r'^name\s*=\s*"oo_core"', manifest, re.M)
    assert re.search(r'crate-type\s*=\s*\["cdylib"\]', manifest)
    assert re.search(r'pyo3.*"=0\.22\.6"', manifest) or re.search(r'version\s*=\s*"=0\.22\.6"', manifest), "pyo3 is pinned exactly"
    lock = (_CRATE / "Cargo.lock").read_text(encoding="utf-8")
    assert 'name = "pyo3"\nversion = "0.22.6"' in lock, "the lock names the pinned pyo3"
    script = REPO / "scripts" / "build_oo_core.sh"
    assert script.is_file() and script.stat().st_mode & 0o111, "the build script is in the tree and executable"
    body = script.read_text(encoding="utf-8")
    assert "cargo build --release" in body and "opti_oignon/native" in body
    ignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "opti_oignon/native/*.so" in ignore, "the artefact is never tracked"
    loader = source("native", "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(loader)
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any(("oo_core" in (a.name for a in n.names)) for n in top if isinstance(n, ast.Import))
    assert not any(n.module and "oo_core" in n.module for n in top if isinstance(n, ast.ImportFrom))
    assert not any(isinstance(n, ast.ImportFrom) and any(a.name == "oo_core" for a in n.names) for n in top), "the loader imports the artefact inside load(), never at module scope"
    sys.modules.pop("opti_oignon.native.oo_core", None)
    loaded, restore = isolate(targets={"opti_oignon.native": source("native", "__init__.py")}, packages=())
    try:
        assert "opti_oignon.native.oo_core" not in sys.modules, "importing the loader loads no artefact"
        assert loaded["opti_oignon.native"]._loaded is loaded["opti_oignon.native"]._UNSET, "nothing asked, nothing loaded"
        assert hasattr(loaded["opti_oignon.native"], "load") and hasattr(loaded["opti_oignon.native"], "available")
    finally:
        restore()


# ---------------------------------------------------------------------------
# The recall probes, natively (NC6 to NC9)
# ---------------------------------------------------------------------------
# The scope the native probes claim: ASCII, Latin-1 Supplement, Latin
# Extended-A without the dotted capital I (it lowercases to an ASCII "i"
# and a combining dot), General Punctuation and the euro sign.
_SCOPE = ((0x00, 0x12F), (0x131, 0x17F), (0x2000, 0x206F), (0x20AC, 0x20AC))

# French letters and typography, as escapes: e acute and grave, a grave,
# c cedilla, o circumflex, the oe ligature in both cases, the capital E
# acute, the guillemets, the typographic apostrophe and quotes, the
# ellipsis, the em and en dashes, the euro sign, the two no-break spaces.
_FRENCH = "\u00e9\u00e8\u00e0\u00e7\u00f4\u0153\u0152\u00c9\u00ab\u00bb\u2019\u201c\u201d\u2026\u2014\u2013\u20ac\u00a0\u202f"

_EDGE = (
    "We decided on 2026-09-24 to ship. Alice will not use Docker! Bob can't come? Carol never agreed.",
    "The budget is 1,200.50 euros, 3.14abc is not a number, -5 and x-7 and 7- are not, 12,5% is.",
    "Version 2.0.1 ships; 1.2.3.4 splits; 10.5, 20,5. And 3. Then 4.",
    "NOT this. Never that! No way? CANNOT be. Cannot't. don't won't isn't. no-one. not_now. n't",
    "We will meet at Harvest-HQ with McDonald and USA reps; A B Cd EFg hIJ.",
    "Nous avons d\u00e9cid\u00e9 le 2026-09-24 : \u00c9lodie viendra \u00e0 Paris, pas \u00e0 Lyon. On a dit 12\u202f000 \u20ac.",
    "\u00ab L\u00e9on \u00bb a dit : on n\u2019utilisera pas Docker\u2026 \u2014 le 24/09, \u00e0 7 h 30. \u0152uvre, c\u0153ur, \u00c9T\u00c9, \u00c7a.",
    "Tabs\tand\nnewlines. Here!\u00a0Next.\u2003Em-space? \u2028Line-sep.\u205fMath.  Double.\x1cSeparated.",
    "x2026-09-24 2026-09-24x 2026-09-24- -2026-09-24 2026-9-24 20260-09-24 2026-09-245 2026-09-24.",
    "Numbers: 0 00 007 1e5 1_000 \u00bd \u00b23 3\u00b2 \u00b9 5\u00bd 1.5.6 ,5 5, .5 5. 1,2,3 4-5 -6 7-",
    "The plan to ship must be agreed. We shall see. I chose red. They agreed not to, and never will not.",
    "We decided not to go and never looked back. We will not, cannot, and don't.",
    "",
    "   ",
    "Single",
    "A.B.C. D. E! F? G",
)

_PIECES = (
    "1", "12", "2026", "09", "-", ".", ",", " ", "  ", "\t", "\n", "\u00a0", "\u2009", "!", "?", "'", "\u2019",
    "not", "Not", "NEVER", "no", "cannot", "n't", "will ", "decided", "agreed", "must", "plan to", "shall",
    "Alice", "McD", "A", "b", "x", "_", "the", "The", "We", "Harvest", "2026-09-24", "3.14", "1,000",
    "\u00e9", "\u00c9", "\u00df", "\u00b2", "\u00bd", "\u0153", "\u2026", "\u2014", "\u20ac", "\u00b5", "\u00aa",
)


def _in_scope():
    return [chr(cp) for lo, hi in _SCOPE for cp in range(lo, hi + 1)]


def _generated(count, seed):
    rng = random.Random(seed)
    return ["".join(rng.choice(_PIECES) for _ in range(rng.randint(1, 40))) for _ in range(count)]


def _around(c):
    """Texts that put one code point where each class decision of the probes is taken."""
    return (
        f"A{c}12{c}B. x{c}Yz{c}.{c}Next{c}sentence{c}not{c}here{c}2026-01-02{c}end",
        f"{c}Abc {c}9 not{c} {c}never can't{c} 1{c}2 3.{c}4 5,{c}6 7{c}.8",
        f"We decided{c}to use Kafka. Will{c}not. {c}No{c}. Mc{c}Donald 2026-01-0{c}3 {c}ot ca{c}not n{c}t",
        f"{c}",
        f"Z{c}Z{c}.{c}{c}? {c}! 7-{c} {c}-7 {c}2026-03-04{c} We will {c}",
    )


def _spans(texts, size=3):
    return [
        [{"turn_id": f"t{i + j:04d}", "role": "user", "text": text} for j, text in enumerate(texts[i:i + size])]
        for i in range(0, len(texts), size)
    ]


def _runs(points):
    runs = []
    for cp in points:
        if runs and cp == runs[-1][1] + 1:
            runs[-1][1] = cp
        else:
            runs.append([cp, cp])
    return runs


def _probe_window():
    targets = {
        "opti_oignon.memory.probes": source("memory", "probes.py"),
        "opti_oignon.memory.baseline": source("memory", "baseline.py"),
        "opti_oignon.native": source("native", "__init__.py"),
    }
    return isolate(targets=targets, packages=("opti_oignon.memory",))


def _baseline_texts(loaded):
    return [t["content"] for corpus in loaded["opti_oignon.memory.baseline"].CORPORA.values() for t in corpus.turns]


class _Counting:
    """The native core behind a counter, so a contract sees the surface go through it."""

    _COUNTED = ("probe_generate", "probe_score")

    def __init__(self, core):
        self.core = core
        self.calls = collections.Counter()
        self.answered = collections.Counter()

    def __getattr__(self, name):
        target = getattr(self.core, name)
        if name not in self._COUNTED:
            return target

        def counted(*args, **kwargs):
            self.calls[name] += 1
            result = target(*args, **kwargs)
            if result is not None:
                self.answered[name] += 1
            return result

        return counted


def _reference(probes, fn, *args):
    """``fn`` with the native core out of reach: the reference path."""
    saved = probes._native
    probes._native = lambda: None
    try:
        return fn(*args)
    finally:
        probes._native = saved


# ---------------------------------------------------------------------------
# NC6 -- the generator draws what the reference draws
# ---------------------------------------------------------------------------
def test_nc6_the_native_probe_generator_draws_what_the_reference_draws():
    loaded, restore = _probe_window()
    try:
        probes = loaded["opti_oignon.memory.probes"]
        core = loaded["opti_oignon.native"].load()
        assert core is not None, "oo_core is not built: run scripts/build_oo_core.sh on this machine"
        counting = _Counting(core)
        probes._native = lambda: counting
        corpus = _baseline_texts(loaded) + list(_EDGE) + [t for c in _in_scope() for t in _around(c)] + _generated(1200, seed=20260924)
        spans = _spans(corpus)
        kinds = collections.Counter()
        for span in spans:
            drawn = probes.generate_probes(span)
            assert drawn == _reference(probes, probes.generate_probes, span), f"probe for probe on {span!r}"
            kinds.update(p.kind for p in drawn)
            kinds.update(f"negations={p.negations}" for p in drawn if p.kind == "decision")
        assert counting.calls["probe_generate"] == counting.answered["probe_generate"] == len(spans), (
            "every span went through the core, and the core answered every one"
        )
        for kind in ("date", "number", "entity", "decision", "negations=0", "negations=1", "negations=2"):
            assert kinds[kind] >= 1, f"the corpus draws {kind}: an equivalence over nothing proves nothing"
        assert probes.generate_probes([]) == [] and counting.answered["probe_generate"] == len(spans) + 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# NC7 -- the scorer fails what the reference fails
# ---------------------------------------------------------------------------
def test_nc7_the_native_scorer_fails_the_probes_the_reference_fails():
    loaded, restore = _probe_window()
    try:
        probes = loaded["opti_oignon.memory.probes"]
        core = loaded["opti_oignon.native"].load()
        assert core is not None, "oo_core is not built: run scripts/build_oo_core.sh on this machine"
        counting = _Counting(core)
        probes._native = lambda: counting
        rng = random.Random(20260925)
        texts = _baseline_texts(loaded) + list(_EDGE) + _generated(300, seed=7)
        verdicts = collections.Counter()
        scored = 0
        for span in _spans(texts):
            drawn = _reference(probes, probes.generate_probes, span)
            source_text = " ".join(t["text"] for t in span)
            lossy = " ".join(w for w in source_text.split(" ") if rng.random() > 0.3)
            inverted = source_text.replace(" not ", " ").replace("will ", "will not ")
            for candidate in (source_text, lossy, inverted, rng.choice(texts), ""):
                got = probes.score(drawn, candidate)
                assert got == _reference(probes, probes.score, drawn, candidate), f"scoring {candidate!r}"
                verdicts.update((p.kind, p in got.failures) for p in drawn)
                scored += 1
        Probe = probes.Probe
        hand = [
            Probe("entity", "q", "Alice", "t1"),
            Probe("entity", "q", "ALICE", "t1"),
            Probe("entity", "q", "", "t1"),
            Probe("entity", "q", "\u00c9lodie", "t1"),
            Probe("number", "q", "1,200.50", "t1"),
            Probe("number", "q", "", "t1"),
            Probe("date", "q", "2026-09-24", "t1"),
            Probe("date", "q", "1.5", "t1"),
            Probe("custom", "q", "docker", "t1"),
            Probe("decision", "q", "x", "t1", key=frozenset({"docker", "use"}), negated=True, negations=1),
            Probe("decision", "q", "x", "t1", key=frozenset({"docker", "use"}), negations=0),
            Probe("decision", "q", "x", "t1", key=frozenset({"Docker"}), negations=0),
            Probe("decision", "q", "x", "t1", key=frozenset({"ship", "docker", "alice", "use"}), negations=True),
        ]
        for candidate in _EDGE + ("Alice will not use Docker.", "alice uses docker, 1,200.50 on 2026-09-24", "Use Docker. Not docker use. Never."):
            got = probes.score(hand, candidate)
            assert got == _reference(probes, probes.score, hand, candidate), f"hand-built probes on {candidate!r}"
            scored += 1
        assert counting.calls["probe_score"] == counting.answered["probe_score"] == scored, (
            "every scoring went through the core, and the core answered every one"
        )
        for kind in ("date", "number", "entity", "decision"):
            assert verdicts[(kind, True)] >= 1 and verdicts[(kind, False)] >= 1, f"{kind} probes both pass and fail in the corpus"
        answered = counting.answered["probe_score"]
        for shape, raised in (
            (Probe("decision", "q", "x", "t1", key=frozenset(), negations=0), ZeroDivisionError),
            (Probe("decision", "q", "x", "t1", key=["docker"], negations=0), TypeError),
            (Probe("entity", "q", None, "t1"), AttributeError),
        ):
            with pytest.raises(raised):
                probes.score([shape], "One sentence.")
        assert counting.answered["probe_score"] == answered, "a probe shape the core does not take never reaches it"
    finally:
        restore()


# ---------------------------------------------------------------------------
# NC8 -- the scope is exact
# ---------------------------------------------------------------------------
def test_nc8_the_probes_scope_is_exact_and_everything_outside_it_goes_to_the_reference():
    loaded, restore = _probe_window()
    try:
        probes = loaded["opti_oignon.memory.probes"]
        core = loaded["opti_oignon.native"].load()
        assert core is not None, "oo_core is not built: run scripts/build_oo_core.sh on this machine"
        table = core.probe_text_classes()
        accepted = {row[0] for row in table}
        assert {ord(c) for c in _in_scope()} <= accepted, "ASCII, Latin-1, Latin Extended-A but the dotted I, General Punctuation, the euro"
        assert {ord(c) for c in _FRENCH} <= accepted, "French letters and typography take the native path"
        for cp, space, word, digit, lower in table:
            c = chr(cp)
            assert space == (re.fullmatch(r"\s", c) is not None), f"U+{cp:04X} whitespace"
            assert space == (c.strip() == ""), f"U+{cp:04X} strip"
            assert word == (re.fullmatch(r"\w", c) is not None), f"U+{cp:04X} word"
            assert digit == (re.fullmatch(r"\d", c) is not None), f"U+{cp:04X} digit"
            assert lower == c.lower(), f"U+{cp:04X} lowercase"
            for letter in "notevrca":
                expected = c.isascii() and c.lower() == letter
                assert (re.fullmatch(letter, c, re.IGNORECASE) is not None) == expected, f"U+{cp:04X} against {letter}, ignoring case"
        counting = _Counting(core)
        probes._native = lambda: counting
        runs = _runs(sorted(accepted))
        outside = {lo - 1 for lo, _ in runs if lo > 0} | {hi + 1 for _, hi in runs if hi < 0x10FFFF}
        outside |= {0x130, 0x212A, 0x3000, 0x1F9C5, 0x10FFFF}
        outside = sorted(cp for cp in outside - accepted if not 0xD800 <= cp <= 0xDFFF)
        assert {0x130, 0x212A, 0x3000} <= set(outside), "the dotted I, the Kelvin sign and the ideographic space stay outside"
        for cp in outside:
            text = f"Alice will not use 12 on 2026-09-24 {chr(cp)}K. The {chr(cp)}Kelvin agreed."
            span = [{"turn_id": "t1", "text": text}]
            before = dict(counting.answered)
            drawn = probes.generate_probes(span)
            assert drawn == _reference(probes, probes.generate_probes, span), f"U+{cp:04X}: the reference answers"
            assert probes.score(drawn, text) == _reference(probes, probes.score, drawn, text), f"U+{cp:04X}: the reference scores"
            assert dict(counting.answered) == before, f"U+{cp:04X}: the core refuses what it cannot class"
        assert counting.calls["probe_generate"] == counting.calls["probe_score"] == len(outside), "the core was asked each time"
    finally:
        restore()


# ---------------------------------------------------------------------------
# NC9 -- the patterns travel with the call
# ---------------------------------------------------------------------------
def test_nc9_a_pattern_the_core_does_not_implement_sends_the_call_to_the_reference():
    loaded, restore = _probe_window()
    try:
        probes = loaded["opti_oignon.memory.probes"]
        core = loaded["opti_oignon.native"].load()
        assert core is not None, "oo_core is not built: run scripts/build_oo_core.sh on this machine"
        counting = _Counting(core)
        probes._native = lambda: counting
        text = "Alice will ship without Docker on 2026-09-24. Bob counted 12 hours."
        span = [{"turn_id": "t1", "text": text}]
        drawn = probes.generate_probes(span)
        probes.score(drawn, text)
        assert counting.answered["probe_generate"] == counting.answered["probe_score"] == 1, "control: the core answers the module's own patterns"
        variants = []
        for name in ("_SENTENCE_SPLIT", "_DATE", "_NUMBER", "_WORD", "_CAPITALISED", "_NEGATION"):
            original = getattr(probes, name)
            variants.append((name, re.compile(original.pattern + "(?:)", original.flags)))
            variants.append((name, re.compile(original.pattern, original.flags ^ re.IGNORECASE)))
        for name in ("_ANSWER_BEFORE", "_ANSWER_AFTER"):
            variants.append((name, getattr(probes, name) + "(?:)"))
        for name, altered in variants:
            original = getattr(probes, name)
            setattr(probes, name, altered)
            try:
                answered = dict(counting.answered)
                assert probes.generate_probes(span) == _reference(probes, probes.generate_probes, span), name
                assert probes.score(drawn, text) == _reference(probes, probes.score, drawn, text), name
                assert dict(counting.answered) == answered, f"{name} altered: the core refuses a pattern it does not implement"
            finally:
                setattr(probes, name, original)
        saved = probes._NEGATION
        probes._NEGATION = re.compile(saved.pattern.replace("cannot", "cannot|without"), saved.flags)
        try:
            assert [p.negations for p in probes.generate_probes(span) if p.kind == "decision"] == [1], "the changed pattern takes effect"
        finally:
            probes._NEGATION = saved
        assert [p.negations for p in drawn if p.kind == "decision"] == [0], "control: the module's own pattern does not count it"
    finally:
        restore()



# ---------------------------------------------------------------------------
# NC10 -- French, the typographic apostrophe, the dotless i and the long s
# ---------------------------------------------------------------------------
_NEGATIONS = (
    "Nous avons d\u00e9cid\u00e9 de ne pas utiliser Docker. On a d\u00e9cid\u00e9 : on utilisera pas Docker.",
    "Il faut qu'on n'utilise plus Docker, et on ne l'a jamais fait. Aucun serveur, aucune base, rien.",
    "Nous n\u2019utiliserons jamais Kafka. Je vais le dire : il n\u2019y a rien \u00e0 changer.",
    "We decided we don\u2019t ship. We won't. We can\u2019t, WE CAN'T, we don'T.",
    "Nous avons convenu que Bob ne m\u00e8ne pas la revue. Il doit la mener, ne pas oublier.",
    "Ne PAS toucher. JAMAIS. Rien. N\u2019importe. n' existe. rock n' roll. N'2026 et n\u2019_x.",
    "Jama\u0131s ! Pa\u017f question. R\u0131en. Nous avons opt\u00e9 pour Oslo le 2026-10-01, pa\u017f Lyon.",
    "Pr\u00e9vu de livrer le 2026-11-02 ; nous allons livrer, on va livrer, je vais livrer, il faut livrer.",
    "La d\u00e9cision est prise : Carol doit mener la revue, nous devons finir, vous devez lire, ils doivent signer.",
)


def test_nc10_the_native_probes_read_french_the_typographic_apostrophe_and_the_folded_letters():
    loaded, restore = _probe_window()
    try:
        probes = loaded["opti_oignon.memory.probes"]
        core = loaded["opti_oignon.native"].load()
        assert core is not None, "oo_core is not built: run scripts/build_oo_core.sh on this machine"
        counting = _Counting(core)
        probes._native = lambda: counting
        negations = 0
        for text in _NEGATIONS:
            span = [{"turn_id": "t1", "text": text}]
            drawn = probes.generate_probes(span)
            assert drawn == _reference(probes, probes.generate_probes, span), f"probe for probe on {text!r}"
            negations += sum(p.negations for p in drawn if p.kind == "decision")
            for candidate in (text, text.replace("ne ", "").replace("pas ", ""), text.replace("\u2019", "'"), ""):
                assert probes.score(drawn, candidate) == _reference(probes, probes.score, drawn, candidate), candidate
        assert counting.calls["probe_generate"] == counting.answered["probe_generate"] == len(_NEGATIONS)
        assert counting.answered["probe_score"] == counting.calls["probe_score"] == 4 * len(_NEGATIONS)
        assert negations >= 10, "the corpus holds negations to count, French and English"
        assert any(p.kind == "decision" for text in _NEGATIONS[:3] for p in probes.generate_probes([{"turn_id": "t", "text": text}]))
        words = re.search(r"\(([a-z|]+)\)", probes._NEGATION.pattern).group(1).split("|")
        letters = "".join(sorted(set("".join(words) + "nt")))
        folds = {cp: set(matched) for cp, matched in core.probe_letter_folds(letters)}
        for lo, hi in _SCOPE:
            for cp in range(lo, hi + 1):
                python = {letter for letter in letters if re.fullmatch(letter, chr(cp), re.IGNORECASE)}
                assert folds.get(cp, set()) == python, f"U+{cp:04X} folds to {sorted(python)} in Python"
        assert {0x131, 0x17F} <= set(folds), "the dotless i and the long s fold, as Python folds them"
    finally:
        restore()



# ---------------------------------------------------------------------------
# The build script (NC11 and NC12)
# ---------------------------------------------------------------------------
_STAND_IN_CARGO = """#!/usr/bin/env bash
while [ $# -gt 0 ]; do
  if [ "$1" = "--manifest-path" ]; then manifest="$2"; shift; fi
  shift
done
mkdir -p "$(dirname "$manifest")/target/release"
echo "new artefact" > "$(dirname "$manifest")/target/release/liboo_core.so"
"""

_STAND_IN_LOADER = """from pathlib import Path
from types import SimpleNamespace


def load():
    path = Path(__file__).resolve().parent / "oo_core.so"
    return SimpleNamespace(VERSION=path.read_text().strip()) if path.is_file() else None
"""

_OTHER_TREE_LOADER = """from types import SimpleNamespace


def load():
    return SimpleNamespace(VERSION="another tree")
"""


def _package(root, loader):
    native = root / "opti_oignon" / "native"
    native.mkdir(parents=True)
    (root / "opti_oignon" / "__init__.py").write_text("", encoding="utf-8")
    (native / "__init__.py").write_text(loader, encoding="utf-8")
    return native


def _script_tree(tmp_path):
    """A tree the build script can run in: its own copy, a stand-in cargo, a stand-in loader."""
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    script = root / "scripts" / "build_oo_core.sh"
    shutil.copy2(REPO / "scripts" / "build_oo_core.sh", script)
    (root / "rust" / "oo_core").mkdir(parents=True)
    (root / "rust" / "oo_core" / "Cargo.toml").write_text('[package]\nname = "oo_core"\n', encoding="utf-8")
    native = _package(root, _STAND_IN_LOADER)
    (native / "oo_core.so").write_text("old artefact\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cargo = bin_dir / "cargo"
    cargo.write_text(_STAND_IN_CARGO, encoding="utf-8")
    cargo.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    return root, script, native, env


def test_nc11_the_build_script_installs_by_rename_and_leaves_nothing_behind(tmp_path):
    root, script, native, env = _script_tree(tmp_path)
    installed = native / "oo_core.so"
    before = installed.stat().st_ino
    with installed.open("rb") as held:
        run = subprocess.run(["bash", str(script)], cwd=root, env=env, capture_output=True, text=True, timeout=120)
        assert run.returncode == 0, run.stderr
        held.seek(0)
        assert held.read() == b"old artefact\n", "a process holding the old artefact keeps its bytes"
    assert installed.read_bytes() == b"new artefact\n", "the new artefact is in place"
    assert installed.stat().st_ino != before, "a new file, renamed over the old one"
    assert sorted(p.name for p in native.iterdir()) == ["__init__.py", "oo_core.so"], "no staging file is left behind"


def test_nc12_the_build_script_checks_its_own_trees_artefact_from_any_directory(tmp_path):
    root, script, native, env = _script_tree(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    _package(elsewhere, _OTHER_TREE_LOADER)
    run = subprocess.run(["bash", str(script)], cwd=elsewhere, env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines()[-1] == "oo_core new artefact", run.stdout


# ---------------------------------------------------------------------------
# NC13 -- French names and whole French words, natively
# ---------------------------------------------------------------------------
_NAMES = (
    "Nous avons invit\u00e9 \u00c9lodie, H\u00e9l\u00e8ne et Chlo\u00e9 \u00e0 la revue avec Bob.",
    "Chlo\u00e9 et Andr\u00e9 ont d\u00e9cid\u00e9 que Z\u00f6e m\u00e8ne la d\u00e9mo \u00e0 Lyon.",
    "\u00c9T\u00c9 comme \u00c7a, \u0152dipe et \u0178vonne ; \u00c0 bient\u00f4t, \u00d8ystein ! O\u00f9 ? L\u00e0. D\u00e9j\u00e0.",
    "Km\u00b2 et \u00b5Service, \u00aaBc, O\u2019Neil, d\u2019Artagnan et Mc\u00c9lodie d\u00e9cideront le 2026-09-24.",
)


def test_nc13_the_native_probes_draw_french_names_and_whole_words_as_the_reference():
    loaded, restore = _probe_window()
    try:
        probes = loaded["opti_oignon.memory.probes"]
        core = loaded["opti_oignon.native"].load()
        assert core is not None, "oo_core is not built: run scripts/build_oo_core.sh on this machine"
        counting = _Counting(core)
        probes._native = lambda: counting
        accented = 0
        for text in _NAMES:
            span = [{"turn_id": "t1", "text": text}]
            drawn = probes.generate_probes(span)
            assert drawn == _reference(probes, probes.generate_probes, span), f"probe for probe on {text!r}"
            accented += sum(1 for p in drawn if p.kind == "entity" and not p.answer.isascii())
            for candidate in (text, text.replace("\u00e9", "e"), text.lower(), ""):
                assert probes.score(drawn, candidate) == _reference(probes, probes.score, drawn, candidate), candidate
        assert accented >= 6, "names with accents are drawn"
        capitals = 0
        for c in _in_scope():
            text = f"{c}bc {c}. x{c}y {c}Abc"
            span = [{"turn_id": "t1", "text": text}]
            drawn = probes.generate_probes(span)
            assert drawn == _reference(probes, probes.generate_probes, span), f"U+{ord(c):04X}"
            assert probes.score(drawn, text.lower()) == _reference(probes, probes.score, drawn, text.lower())
            capitals += c.isupper()
        assert capitals >= 100, "the scope holds the capitals of ASCII, Latin-1 and Latin Extended-A"
        assert counting.calls["probe_generate"] == counting.answered["probe_generate"] == len(_NAMES) + len(_in_scope())
        assert counting.calls["probe_score"] == counting.answered["probe_score"] == 4 * len(_NAMES) + len(_in_scope())
    finally:
        restore()

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
