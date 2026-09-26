#!/usr/bin/env python3
"""Contracts for the firewall between the test process and the maintainer's data.

Real personal data lives in ``data/`` (keys, the security mode, the signed
audit chain), in ``opti_oignon/data/`` (conversations, plugins, the
governor, the change feed) and in database files inside the tree. For the
whole test session ``tests/_data_firewall.py`` redirects every path in
those places to a mirror that starts as a fresh checkout would: the
tracked files as HEAD holds them, nothing else.

  * DF1 -- a path inside a data place, or a database file inside the tree,
    maps to its place in the mirror; any other path, a descriptor, and a
    path relative to a directory descriptor pass untouched.
  * DF2 -- a session run under the firewall sees a planted key and a
    planted database as a fresh install would, its writes land in the
    mirror, and the planted files and the real places are left exactly as
    they were.
  * DF3 -- the tracked files of the data places are served as HEAD holds
    them, not as the working copy has them, and nothing untracked shows.
  * DF4 -- a SQLite or SQLCipher connect by ``file:`` URI that names a file
    in a data place opens the mirror's file, its query kept (a read-only
    open stays read-only); a URI elsewhere, and an in-memory one, pass
    untouched.

Local-only (the public distribution ships no tests). DF2 and DF3 run a
real pytest session on a fake tree carrying a copy of the conftest and of
the firewall, and read that session's own junit file.
"""

import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS))

import _data_firewall  # noqa: E402

BUDGET_S = {
    "test_df4_a_uri_connect_to_a_data_place_opens_the_mirror_and_keeps_its_query": 2.0,
}


def _module():
    return _data_firewall


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _tree(tmp_path, canary):
    """A fake tree with the conftest, the firewall and one canary suite."""
    tree = tmp_path / "tree"
    (tree / "tests").mkdir(parents=True)
    for name in ("conftest.py", "_data_firewall.py"):
        shutil.copy2(_TESTS / name, tree / "tests" / name)
    (tree / "tests" / "test_canary.py").write_text(canary, encoding="utf-8")
    (tree / "data").mkdir()
    (tree / "opti_oignon" / "data").mkdir(parents=True)
    return tree


def _session(tree):
    """Run the canary suite in its tree; name -> list of failure messages, and the output."""
    junit = tree.parent / "junit.xml"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTEST_ADDOPTS", None)
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
         "--rootdir", str(tree), f"--junitxml={junit}", str(tree / "tests" / "test_canary.py")],
        cwd=tree, env=env, capture_output=True, text=True, timeout=180,
    )
    outcomes = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        reds = outcomes.setdefault(case.get("name"), [])
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is not None:
                reds.append(node.get("message") or "")
    return outcomes, run.stdout


# ---------------------------------------------------------------------------
# DF1 -- which paths go to the mirror
# ---------------------------------------------------------------------------
def test_df1_a_data_path_or_a_database_in_the_tree_maps_to_the_mirror_and_nothing_else(tmp_path, monkeypatch):
    root, mirror = tmp_path / "tree", tmp_path / "mirror"
    firewall = _module().DataFirewall(root, mirror=mirror, seed=False)
    inside = {
        root / "data": "data",
        root / "data" / ".keyfile": "data/.keyfile",
        root / "opti_oignon" / "data" / "conversations.db": "opti_oignon/data/conversations.db",
        root / "opti_oignon" / "data" / "rag" / "cache": "opti_oignon/data/rag/cache",
        root / "opti_oignon" / "config" / "fingerprint.db": "opti_oignon/config/fingerprint.db",
        root / "stray.sqlite3": "stray.sqlite3",
    }
    for path, relative in inside.items():
        assert firewall.target(str(path)) == str(mirror / relative), path
        assert firewall.target(path) == str(mirror / relative), "a path object maps too"
    assert firewall.target(os.fsencode(root / "data" / "x")) == os.fsencode(mirror / "data" / "x"), "bytes stay bytes"
    outside = [
        str(root / "opti_oignon" / "memory" / "probes.py"),
        str(root / "tests" / "conftest.py"),
        str(root / "data_notes.txt"),
        str(root / "database.py"),
        str(tmp_path / "elsewhere.db"),
        str(root),
    ]
    for path in outside:
        assert firewall.target(path) == path, path
    assert firewall.target(3) == 3, "a descriptor is not a path"
    monkeypatch.chdir(tmp_path)
    assert firewall.target("tree/data/.keyfile") == str(mirror / "data" / ".keyfile"), "relative to the working directory"
    seen = []
    wrapped = firewall._one_path(lambda *args, **kwargs: seen.append((args, kwargs)))
    wrapped("tree/data/.keyfile")
    wrapped("tree/data/.keyfile", dir_fd=7)
    assert seen == [((str(mirror / "data" / ".keyfile"),), {}), (("tree/data/.keyfile",), {"dir_fd": 7})], seen
    assert firewall.redirected, "each redirection is counted"


# ---------------------------------------------------------------------------
# DF2 -- a session sees a fresh install and leaves the real places as they were
# ---------------------------------------------------------------------------
_FRESH = '''import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_the_private_places_read_as_a_fresh_install():
    assert not (ROOT / "data" / ".keyfile").exists()
    assert os.listdir(ROOT / "data") == []
    database = sqlite3.connect(str(ROOT / "opti_oignon" / "data" / "conversations.db"))
    assert database.execute("select name from sqlite_master").fetchall() == []
    database.execute("create table written (x)")
    database.commit()
    database.close()
    (ROOT / "data" / "written.txt").write_text("in the mirror")
    assert (ROOT / "data" / "written.txt").read_text() == "in the mirror"
    assert sorted(os.listdir(ROOT / "data")) == ["written.txt"]


def test_elsewhere_is_the_real_filesystem(tmp_path):
    database = sqlite3.connect(str(tmp_path / "scratch.db"))
    database.execute("create table t (x)")
    database.close()
    assert (tmp_path / "scratch.db").exists()
'''


def test_df2_a_session_sees_a_fresh_install_and_leaves_the_real_places_as_they_were(tmp_path):
    tree = _tree(tmp_path, _FRESH)
    key = tree / "data" / ".keyfile"
    key.write_bytes(b"the maintainer's master key")
    conversations = tree / "opti_oignon" / "data" / "conversations.db"
    database = sqlite3.connect(str(conversations))
    database.execute("create table conversation (text)")
    database.execute("insert into conversation values ('a private conversation')")
    database.commit()
    database.close()
    before = {path: _md5(path) for path in (key, conversations)}
    outcomes, output = _session(tree)
    assert set(outcomes) == {"test_the_private_places_read_as_a_fresh_install", "test_elsewhere_is_the_real_filesystem"}, output
    assert not any(outcomes.values()), outcomes
    assert {path: _md5(path) for path in before} == before, "the planted files are untouched"
    assert sorted(p.name for p in (tree / "data").iterdir()) == [".keyfile"], "nothing was written into the real place"
    assert sorted(p.name for p in (tree / "opti_oignon" / "data").iterdir()) == ["conversations.db"]
    assert "data firewall:" in output, output


# ---------------------------------------------------------------------------
# DF3 -- tracked files as HEAD holds them
# ---------------------------------------------------------------------------
_TRACKED = '''import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_the_tracked_files_are_served_as_head_holds_them():
    presets = ROOT / "opti_oignon" / "data" / "system_presets.yaml"
    assert presets.read_text() == "committed: true\\n"
    assert sorted(os.listdir(ROOT / "opti_oignon" / "data")) == ["system_presets.yaml"]
    assert not (ROOT / "opti_oignon" / "data" / "private.db").exists()
'''


def test_df3_tracked_files_are_served_as_head_holds_them_and_nothing_untracked_shows(tmp_path):
    tree = _tree(tmp_path, _TRACKED)
    presets = tree / "opti_oignon" / "data" / "system_presets.yaml"
    presets.write_text("committed: true\n", encoding="utf-8")
    git = ["git", "-C", str(tree), "-c", "user.name=contract", "-c", "user.email=contract@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True, capture_output=True)
    subprocess.run([*git, "add", "opti_oignon/data/system_presets.yaml"], check=True, capture_output=True)
    subprocess.run([*git, "commit", "-q", "-m", "presets"], check=True, capture_output=True)
    presets.write_text("changed in the working copy\n", encoding="utf-8")
    (tree / "opti_oignon" / "data" / "private.db").write_bytes(b"not a tracked file")
    outcomes, output = _session(tree)
    assert outcomes == {"test_the_tracked_files_are_served_as_head_holds_them": []}, (outcomes, output)
    assert presets.read_text(encoding="utf-8") == "changed in the working copy\n", "the working copy is left as it was"


# ---------------------------------------------------------------------------
# DF4 -- a connect by URI is redirected too
# ---------------------------------------------------------------------------
def _uri_modules():
    """The connect functions the firewall covers: SQLite's, and SQLCipher's when it is installed."""
    modules = [("sqlite3", sqlite3)]
    try:
        import sqlcipher3.dbapi2 as cipher
    except Exception:  # noqa: BLE001 - no SQLCipher here: the standard library alone
        cipher = None
    if cipher is not None:
        modules.append(("sqlcipher3", cipher))
    return modules


def test_df4_a_uri_connect_to_a_data_place_opens_the_mirror_and_keeps_its_query(tmp_path):
    root, mirror = tmp_path / "tree", tmp_path / "mirror"
    (root / "data").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere.sqlite"
    firewall = _module().DataFirewall(root, mirror=mirror, seed=False)
    read, refused, outside, memory = {}, {}, {}, {}
    firewall.install()
    try:
        for name, module in _uri_modules():
            real = root / "data" / f"{name} store.db"
            conn = module.connect(str(real))
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO users VALUES (1)")
            conn.commit()
            conn.close()
            conn = module.connect(real.as_uri() + "?mode=ro", uri=True)
            try:
                read[name] = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
                try:
                    conn.execute("INSERT INTO users VALUES (2)")
                    refused[name] = False
                except module.OperationalError:
                    refused[name] = True
            finally:
                conn.close()
            conn = module.connect(elsewhere.as_uri() + "?mode=rwc", uri=True)
            conn.execute("CREATE TABLE IF NOT EXISTS t (x)")
            conn.close()
            outside[name] = elsewhere.exists()
            conn = module.connect("file::memory:?cache=shared", uri=True)
            memory[name] = conn.execute("SELECT 1").fetchone()[0]
            conn.close()
    finally:
        firewall.uninstall()
    names = [name for name, _module_ in _uri_modules()]
    assert "sqlite3" in names
    assert read == {name: 1 for name in names}, "a read-only URI opens the store the firewall wrote: the mirror's"
    assert refused == {name: True for name in names}, "and its query is kept: it stays read-only"
    assert outside == {name: True for name in names}, "a URI outside the data places is the real filesystem"
    assert memory == {name: 1 for name in names}, "an in-memory URI passes"
    assert sorted(path.name for path in (root / "data").iterdir()) == [], "nothing reached the real place"
    assert sorted(path.name for path in (mirror / "data").iterdir()) == sorted(f"{name} store.db" for name in names)
    counted = set().union(*firewall.redirected.values())
    assert {f"data/{name} store.db" for name in names} <= counted, counted


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
