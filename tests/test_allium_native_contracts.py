#!/usr/bin/env python3
"""Contracts for the componion engine's native side and for what its sources may not contain.

The engine runs natively when the native core answers for the same world as
the reference, and the reference answers otherwise. Its sources are held to
the rules that keep two engines equal: no float, no hashed iteration order,
no unsafe code, nothing but ASCII, and no internal planning vocabulary.

  * AN1 -- the handshake: a native core whose identity differs from the
    reference's (a stale build, other law files) is not used, the
    reference answers, and that is said once; a matching one is used.
  * AN2 -- the Rust engine's sources hold no float type, no ``std::``, no
    ``unsafe`` beyond the attribute that forbids it, no ``HashMap``, no
    byte outside ASCII and no planning code, and every source file is read.
  * AN3 -- the clean guard's patterns, run over both native crates while
    the guards do not scan ``rust/``, find nothing, and every file is read.
  * AN4 -- the locks agree: every package the engine's lock pins is pinned
    at the same version in the native core's lock, which links the engine.
  * AN5 -- the native call releases the GIL: a Python thread counts while a
    long native call runs, and does not while Python itself is busy.
  * AN6 -- the Python reference iterates no ``set`` or ``frozenset`` and
    calls no ``hash()``, ``id()`` or ``popitem()``, and its reference
    modules divide only by floor division.
  * AN7 -- the golden vectors come out the same under two different string
    hash seeds, and equal the committed ones.
  * AN8 -- the committed golden vectors are hexadecimal and the clean
    guard's patterns find nothing in them.
  * AN9 -- the engine forbids unsafe code and denies arithmetic with side
    effects; ``cargo clippy`` enforces the second, and the ladder runs it
    (owed where clippy is not installed).
  * AN10 -- the organs, which read a genome a peer may have sent, hold no
    construct that can panic: no unwrap, no expect, no panic or unreachable
    macro, no ``abs`` (which panics on the signed minimum) and no indexing;
    their module denies the matching lints.
  * AN11 -- the journal twin, which checks the facts a peer may send, holds
    none of those constructs either, and denies the same lints at its head.
  * AN12 -- outside its tests, the journal twin holds no assertion and no
    placeholder macro, and none of the slice calls that panic on a length
    (``split_at``, ``copy_from_slice``); it denies ``todo`` and
    ``unimplemented`` at its head, and every ban can fire.
  * AN13 -- the platform files (settings, mode, chain, membrane, anchors,
    store) import only the standard library at module level, and SQLite's
    module only inside the store's default plain connect.

Local-only (the public distribution ships no tests). The modules load
through the shared isolation window.
"""

import ast
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_window import native_module, open_allium  # noqa: E402
from _isolation import REPO  # noqa: E402

ENGINE_SRC = REPO / "rust" / "allium" / "src"
CORE_SRC = REPO / "rust" / "oo_core" / "src"
PACKAGE = REPO / "opti_oignon" / "allium"
GOLDEN_DIR = REPO / "tests" / "allium_golden"

BUDGET_S = {
    "test_an1_a_native_core_that_answers_for_another_world_is_not_used_and_that_is_said_once": 2.0,
    "test_an2_the_engine_sources_hold_nothing_that_would_split_the_two_engines": 2.0,
    "test_an3_the_clean_guard_patterns_find_nothing_in_either_native_crate": 2.0,
    "test_an4_the_two_locks_agree_on_every_package_the_engine_pins": 2.0,
    "test_an5_a_long_native_call_lets_another_python_thread_run": 2.0,
    "test_an6_the_python_reference_holds_no_hashed_order_and_no_true_division": 2.0,
    "test_an7_the_golden_vectors_are_the_same_under_two_string_hash_seeds": 2.0,
    "test_an8_the_committed_golden_vectors_are_hexadecimal_and_clean": 2.0,
    "test_an9_the_engine_forbids_unsafe_code_and_denies_side_effect_arithmetic": 2.0,
    "test_an10_the_organs_hold_no_construct_that_can_panic_and_deny_them": 2.0,
    "test_an11_the_journal_twin_holds_no_construct_that_can_panic_and_denies_them": 2.0,
    "test_an12_the_journal_twin_outside_its_tests_holds_no_assertion_or_placeholder_that_can_panic": 2.0,
    "test_an13_the_platform_files_import_the_standard_library_alone_and_sqlite_in_one_place": 2.0,
}


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


def _clean_guard():
    """The public clean guard, loaded by path; ``.github/scripts`` never joins ``sys.path``."""
    spec = importlib.util.spec_from_file_location("_clean_guard_for_allium", REPO / ".github" / "scripts" / "public_clean_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# AN1 -- the handshake
# ---------------------------------------------------------------------------
class _FakeCore:
    def __init__(self, identity):
        self._identity = identity

    def allium_engine(self):
        return self._identity

    def allium_call(self, _request):
        return b"answered by the fake core"


def test_an1_a_native_core_that_answers_for_another_world_is_not_used_and_that_is_said_once(caplog):
    loaded, restore = open_allium()
    try:
        engine = loaded["opti_oignon.allium.engine"]
        ref = loaded["opti_oignon.allium.ref.protocol"]
        request = b'{"name":"fixture","op":"law","v":1}'
        stale = ref.engine_info().replace(b'"engine":"0.1.0"', b'"engine":"0.0.9"')
        assert stale != ref.engine_info()
        engine.reset()
        engine._load_native = lambda: _FakeCore(stale)
        with caplog.at_level(logging.WARNING, logger=engine.logger.name):
            answers = [engine.call(request) for _ in range(3)]
        assert answers == [ref.call(request)] * 3, "the reference answers"
        warnings = [r for r in caplog.records if "does not answer for the reference" in r.getMessage()]
        assert len(warnings) == 1, "said once"
        assert engine.native_in_use() is False
        engine.reset()
        engine._load_native = lambda: _FakeCore(ref.engine_info())
        assert engine.call(request) == b"answered by the fake core" and engine.native_in_use() is True
        engine.reset()
        engine._load_native = lambda: None
        assert engine.call(request) == ref.call(request), "no native core: the reference answers"
        real = native_module(loaded)
        assert engine.handshake(real) is True, "the built core answers for the reference's world"
    finally:
        restore()


# ---------------------------------------------------------------------------
# AN2 -- what the engine's sources may not hold
# ---------------------------------------------------------------------------
_BANS = (
    ("float type", re.compile(r"\bf(?:32|64)\b")),
    ("std path", re.compile(r"\bstd::")),
    ("hashed map", re.compile(r"\bHash(?:Map|Set)\b")),
    ("unsafe", re.compile(r"\bunsafe\b(?!_code\))")),
)


def _bans(text, clean):
    found = []
    for name, pattern in _BANS:
        if pattern.search(text):
            found.append(name)
    if any(ord(char) > 0x7E or (ord(char) < 0x20 and char not in "\n\t") for char in text):
        found.append("non-ascii")
    if clean.find_violations(text.splitlines()):
        found.append("planning code")
    return found


def test_an2_the_engine_sources_hold_nothing_that_would_split_the_two_engines():
    clean = _clean_guard()
    files = sorted(ENGINE_SRC.rglob("*.rs"))
    assert len(files) >= 6, files
    scanned = []
    for path in files:
        text = path.read_bytes().decode("ascii", errors="replace")
        assert _bans(text, clean) == [], (path, _bans(text, clean))
        scanned.append(path)
    assert scanned == sorted(ENGINE_SRC.rglob("*.rs")), "every source file is read"
    canary = "let x: f64 = 1.0;\nuse std::collections::HashMap;\nunsafe { go() }\nlet s = \"caf" + chr(0xE9) + "\";\n// " + "S" + "123 note\n"
    assert _bans(canary, clean) == ["float type", "std path", "hashed map", "unsafe", "non-ascii", "planning code"]


# ---------------------------------------------------------------------------
# AN3 -- the clean guard over both crates
# ---------------------------------------------------------------------------
def test_an3_the_clean_guard_patterns_find_nothing_in_either_native_crate():
    clean = _clean_guard()
    read = []
    for root in (REPO / "rust" / "allium", REPO / "rust" / "oo_core"):
        for path in sorted(root.rglob("*")):
            if "target" in path.relative_to(root).parts or not path.is_file():
                continue
            text = path.read_bytes().decode("ascii", errors="replace")
            assert clean.find_violations(text.splitlines()) == [], path
            read.append(path.relative_to(REPO).as_posix())
    for expected in ("rust/allium/src/lib.rs", "rust/allium/Cargo.lock", "rust/oo_core/src/allium.rs", "rust/oo_core/src/lib.rs"):
        assert expected in read, expected
    assert clean.find_violations(["x = '" + "S" + "123'"]), "the patterns can find something"


# ---------------------------------------------------------------------------
# AN4 -- the locks agree
# ---------------------------------------------------------------------------
def _lock(path):
    packages = {}
    name = None
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("name = "):
            name = line.split('"')[1]
        elif line.startswith("version = ") and name is not None:
            packages[name] = line.split('"')[1]
            name = None
    return packages


def test_an4_the_two_locks_agree_on_every_package_the_engine_pins():
    engine_lock = _lock(REPO / "rust" / "allium" / "Cargo.lock")
    core_lock = _lock(REPO / "rust" / "oo_core" / "Cargo.lock")
    assert "allium" in engine_lock and "sha2" in engine_lock, engine_lock
    for name, version in engine_lock.items():
        assert core_lock.get(name) == version, (name, version, core_lock.get(name))
    core_text = (REPO / "rust" / "oo_core" / "Cargo.lock").read_text(encoding="ascii")
    block = core_text.split('name = "oo_core"', 1)[1].split("[[package]]", 1)[0]
    assert '"allium"' in block, "the native core depends on the engine"
    manifest = (REPO / "rust" / "oo_core" / "Cargo.toml").read_text(encoding="ascii")
    assert 'allium = { path = "../allium" }' in manifest
    engine_manifest = (REPO / "rust" / "allium" / "Cargo.toml").read_text(encoding="ascii")
    assert 'sha2 = { version = "=0.10.9", default-features = false }' in engine_manifest


# ---------------------------------------------------------------------------
# AN5 -- the GIL is released
# ---------------------------------------------------------------------------
def test_an5_a_long_native_call_lets_another_python_thread_run():
    source = (CORE_SRC / "allium.rs").read_text(encoding="ascii")
    assert "py.allow_threads(move || allium::call(&owned))" in source
    loaded, restore = open_allium()
    counter = [0]
    stop = threading.Event()

    def count():
        while not stop.is_set():
            counter[0] += 1

    thread = threading.Thread(target=count, name="an5-counter")
    interval = sys.getswitchinterval()
    try:
        native = native_module(loaded)
        rows = ",".join("[%d,524288,4]" % (1000 + i) for i in range(45000))
        request = ('{"args":[' + rows + '],"fn":"hill_up","op":"fx","v":1}').encode("ascii")
        thread.start()
        time.sleep(0.05)
        assert counter[0] > 0, "the counter runs"
        # Each window opens with a short sleep: the counter takes the GIL, the
        # main thread takes it back after a full switch interval, and the
        # counter then starts a fresh wait of 0.1 s. A window shorter than that
        # can only see the counter move if the GIL is released inside it; a
        # switch requested earlier would otherwise land just after the call
        # returns and be counted as if it had happened during it.
        sys.setswitchinterval(0.1)
        time.sleep(0.001)
        before = counter[0]
        for _ in range(300000):
            pass
        busy = counter[0] - before
        time.sleep(0.001)
        before = counter[0]
        native.allium_call(request)
        during = counter[0] - before
    finally:
        sys.setswitchinterval(interval)
        stop.set()
        if thread.is_alive():
            thread.join()
        restore()
    assert busy == 0, "control: Python holding the GIL lets no other thread count"
    assert during >= 100, f"the native call held the GIL: the counter moved by {during}"


# ---------------------------------------------------------------------------
# AN6 -- no hashed order in the reference
# ---------------------------------------------------------------------------
def _order_findings(tree, reference):
    findings = []
    for node in ast.walk(tree):
        iters = []
        if isinstance(node, (ast.For, ast.AsyncFor)):
            iters.append(node.iter)
        if isinstance(node, ast.comprehension):
            iters.append(node.iter)
        for it in iters:
            if isinstance(it, (ast.Set, ast.SetComp)) or (
                isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id in ("set", "frozenset")
            ):
                findings.append("iterates a set")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("hash", "id"):
                findings.append(f"calls {func.id}()")
            if isinstance(func, ast.Attribute) and func.attr == "popitem":
                findings.append("calls popitem()")
            if reference and isinstance(func, ast.Name) and func.id == "float":
                findings.append("calls float()")
        if reference and isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            findings.append("true division")
    return findings


def test_an6_the_python_reference_holds_no_hashed_order_and_no_true_division():
    files = sorted(PACKAGE.rglob("*.py"))
    assert len(files) >= 8, files
    for path in files:
        tree = ast.parse(path.read_text(encoding="ascii"))
        assert _order_findings(tree, reference=True) == [], (path, _order_findings(tree, reference=True))
    planted = "for x in {'a', 'b'}:\n    pass\n[y for y in frozenset(z)]\nhash(k)\nid(k)\nd.popitem()\nfloat(3)\nq = 1 / 2\n"
    assert sorted(set(_order_findings(ast.parse(planted), reference=True))) == sorted({
        "iterates a set", "calls hash()", "calls id()", "calls popitem()", "calls float()", "true division",
    })


# ---------------------------------------------------------------------------
# AN7 -- the golden vectors under two string hash seeds
# ---------------------------------------------------------------------------
_REPLAY = r"""
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
from opti_oignon.allium.ref import protocol as p
from opti_oignon.allium import wire
seed = bytes(range(32)).hex()
def ask(obj):
    return wire.parse(p.call(wire.emit(obj)))
out = {
    "engine": {"sha256": hashlib.sha256(p.engine_info()).hexdigest()},
    "fact_id": ask({"fact": {"being": "0f" * 16, "body": {"note": "a first word", "turn": 3}, "kind": "sow", "laws": 0, "origin": "a1" * 8, "oseq": 0, "t": 0}, "op": "fact_id", "v": 1}),
    "rng": {
        "below": ask({"bound": 1000, "domain": "golden", "index": 2, "kind": "below", "n": 8, "op": "rng", "seed": seed, "v": 1})["out"],
        "key": ask({"domain": "golden", "index": 0, "kind": "key", "n": 1, "op": "rng", "seed": seed, "v": 1})["out"],
        "noise": ask({"domain": "golden", "index": 1, "kind": "noise", "n": 4, "op": "rng", "seed": seed, "v": 1})["out"],
        "seed": seed,
        "stream": ask({"domain": "golden", "index": 0, "kind": "stream", "n": 8, "op": "rng", "seed": seed, "v": 1})["out"],
        "unit": ask({"domain": "golden", "index": 3, "kind": "unit", "n": 8, "op": "rng", "seed": seed, "v": 1})["out"],
    },
}
print(json.dumps(out, sort_keys=True))
"""


def test_an7_the_golden_vectors_are_the_same_under_two_string_hash_seeds(tmp_path):
    outputs = []
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONDONTWRITEBYTECODE="1")
        run = subprocess.run([sys.executable, "-c", _REPLAY, str(REPO)], cwd=tmp_path, env=env,
                             capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stderr[-2000:]
        outputs.append(json.loads(run.stdout))
    golden = json.loads((GOLDEN_DIR / "v1" / "chassis.json").read_text(encoding="ascii"))
    assert outputs[0] == outputs[1] == golden, "the reference draws the same under any string hash seed"


# ---------------------------------------------------------------------------
# AN8 -- committed vectors are hexadecimal and clean
# ---------------------------------------------------------------------------
def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def test_an8_the_committed_golden_vectors_are_hexadecimal_and_clean():
    clean = _clean_guard()
    files = sorted(GOLDEN_DIR.rglob("*.json"))
    assert files, "there are committed vectors"
    count = 0
    for path in files:
        text = path.read_text(encoding="ascii")
        assert clean.find_violations(text.splitlines()) == [], path
        for value in _strings(json.loads(text)):
            assert re.fullmatch(r"[0-9a-f]+", value), (path, value)
            count += 1
    assert count >= 10, count
    assert clean.find_violations(['{"vector": "' + "S" + '27"}']), "the patterns can find something"


# ---------------------------------------------------------------------------
# AN9 -- the engine's own attributes, and clippy in the ladder
# ---------------------------------------------------------------------------
def test_an9_the_engine_forbids_unsafe_code_and_denies_side_effect_arithmetic():
    lib = (ENGINE_SRC / "lib.rs").read_text(encoding="ascii")
    assert "#![forbid(unsafe_code)]" in lib
    assert "#![deny(clippy::arithmetic_side_effects)]" in lib
    assert "#![cfg_attr(not(test), no_std)]" in lib
    ladder = (REPO / "scripts" / "ladder.sh").read_text(encoding="ascii")
    assert "cargo clippy" in ladder and "rust/allium" in ladder, "the ladder runs clippy over the engine"
    assert "OWED" in ladder, "clippy missing is owed, never a pass"


# ---------------------------------------------------------------------------
# AN10 -- the organs cannot panic on hostile input
# ---------------------------------------------------------------------------
_ORGAN_BANS = (
    ("unwrap", r"\.unwrap\(\)"),
    ("expect", r"\.expect\("),
    ("panic", r"panic!"),
    ("unreachable", r"unreachable!"),
    ("abs", r"\.abs\(\)"),
    ("indexing", r"[A-Za-z0-9_)\]]\["),
)
_ORGAN_DENY = "#![deny(clippy::indexing_slicing, clippy::unwrap_used, clippy::expect_used, clippy::panic)]"


def _organ_findings(text):
    return [name for name, pattern in _ORGAN_BANS if re.search(pattern, text)]


def test_an10_the_organs_hold_no_construct_that_can_panic_and_deny_them():
    organs = ENGINE_SRC / "organs"
    files = sorted(organs.glob("*.rs"))
    assert len(files) >= 3, files
    for path in files:
        assert _organ_findings(path.read_text(encoding="ascii")) == [], path.name
    assert _ORGAN_DENY in (organs / "mod.rs").read_text(encoding="ascii")
    lib = (ENGINE_SRC / "lib.rs").read_text(encoding="ascii")
    assert "pub mod organs;" in lib
    canary = "a.unwrap() b.expect(1) panic!() unreachable!() x.abs() values[0]"
    assert _organ_findings(canary) == [name for name, _ in _ORGAN_BANS], "every ban can fire"


# ---------------------------------------------------------------------------
# AN11 -- the journal twin cannot panic on hostile input
# ---------------------------------------------------------------------------
def test_an11_the_journal_twin_holds_no_construct_that_can_panic_and_denies_them():
    text = (ENGINE_SRC / "journal.rs").read_text(encoding="ascii")
    for name in ("check_envelope", "body_digest", "eid", "fact_id", "fact_envelope"):
        assert f"pub fn {name}(" in text, f"the scanned file is the journal twin: {name}"
    assert _organ_findings(text) == [], _organ_findings(text)
    assert _ORGAN_DENY in text.splitlines(), "the journal module denies the lints itself"
    lib = (ENGINE_SRC / "lib.rs").read_text(encoding="ascii")
    assert "pub mod journal;" in lib
    canary = "a.unwrap() b.expect(1) panic!() unreachable!() x.abs() values[0]"
    assert _organ_findings(canary) == [name for name, _ in _ORGAN_BANS], "every ban can fire"

# ---------------------------------------------------------------------------
# AN12 -- the journal twin, outside its tests: no assertion, no placeholder
# ---------------------------------------------------------------------------
_JOURNAL_BANS = (
    ("assert", r"\bassert!"),
    ("assert_eq", r"\bassert_eq!"),
    ("todo", r"\btodo!"),
    ("unimplemented", r"\bunimplemented!"),
    ("split_at", r"\.split_at\("),
    ("copy_from_slice", r"\.copy_from_slice\("),
)
_JOURNAL_DENY = "#![deny(clippy::todo, clippy::unimplemented)]"


def _journal_findings(text):
    return [name for name, pattern in _JOURNAL_BANS if re.search(pattern, text)]


def test_an12_the_journal_twin_outside_its_tests_holds_no_assertion_or_placeholder_that_can_panic():
    text = (ENGINE_SRC / "journal.rs").read_text(encoding="ascii")
    marker = "#[cfg(test)]"
    assert text.count(marker) == 1, "the scan stops where the test module starts"
    twin, tests = text.split(marker)
    assert "pub fn fact_envelope(" in twin and "pub fn check_envelope(" in twin, "the scanned part is the twin"
    assert "assert_eq!(" in tests, "witness: the part left out is the one that asserts"
    assert _journal_findings(twin) == [], _journal_findings(twin)
    assert _JOURNAL_DENY in twin.splitlines()[:3], "the module denies the placeholders at its head"
    planted = ("assert!(ok)", "assert_eq!(a, b)", "todo!()", "unimplemented!()", "v.split_at(3)",
               "d.copy_from_slice(s)")
    for (name, _pattern), canary in zip(_JOURNAL_BANS, planted):
        assert _journal_findings(canary) == [name], (name, _journal_findings(canary))
    assert _journal_findings("debug_assert!(x) assert_ne!(a, b) v.split_first()") == [], "only what is named"


# ---------------------------------------------------------------------------
# AN13 -- the platform files: the standard library at module level, SQLite in one place
# ---------------------------------------------------------------------------
_PLATFORM = ("settings", "mode", "chain", "membrane", "anchors", "store")


def _module_level_imports(tree):
    """The import statements run when the module is imported: not those inside a function."""
    found = []

    def visit(nodes):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                found.append(node)
            for field in ("body", "orelse", "finalbody", "handlers"):
                inner = getattr(node, field, None)
                if isinstance(inner, list):
                    visit(inner)

    visit(tree.body)
    return found


def _platform_import_findings(tree, allowed_sqlite):
    findings = []
    for node in _module_level_imports(tree):
        if isinstance(node, ast.ImportFrom):
            names = ["." * node.level + (node.module or "")]
        else:
            names = [alias.name for alias in node.names]
        for name in names:
            if name.startswith(".") or name.split(".")[0] not in sys.stdlib_module_names:
                findings.append("module level: " + name)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        else:
            continue
        if not any(name.split(".")[0] == "sqlite3" for name in names):
            continue
        owner = parents.get(node)
        while owner is not None and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = parents.get(owner)
        where = "<module>" if owner is None else owner.name
        if where != allowed_sqlite:
            findings.append("sqlite3 in " + where)
    return findings


def test_an13_the_platform_files_import_the_standard_library_alone_and_sqlite_in_one_place():
    read = []
    for name in _PLATFORM:
        tree = ast.parse((PACKAGE / f"{name}.py").read_text(encoding="ascii"))
        allowed = "_plain_connect" if name == "store" else None
        assert _platform_import_findings(tree, allowed) == [], (name, _platform_import_findings(tree, allowed))
        read.append(name)
        if name == "store":
            store_tree = tree
    assert read == list(_PLATFORM), "every platform file is read"
    assert _platform_import_findings(store_tree, None) == ["sqlite3 in _plain_connect"], \
        "witness: the one import allowed is there, and the check sees it"
    canary = ("import os\nimport sqlite3\nimport yaml\nfrom . import wire\nfrom opti_oignon import config\n"
              "try:\n    import requests\nexcept ImportError:\n    pass\n"
              "def _plain_connect():\n    import sqlite3\n"
              "def elsewhere():\n    import sqlite3\n    from opti_oignon import db_utils\n")
    assert _platform_import_findings(ast.parse(canary), "_plain_connect") == [
        "module level: yaml", "module level: .", "module level: opti_oignon", "module level: requests",
        "sqlite3 in <module>", "sqlite3 in elsewhere"], "every rule can fire"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
