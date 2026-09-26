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
  * DF5 -- a Python child a contract starts with the environment it
    inherits checks, connects, writes, makes a directory and lists in both
    data places and finds and writes only the mirror; the real places are
    left exactly as they were.
  * DF6 -- the same for a child whose environment is a copy extended with a
    variable, and for one built by the componion suites' ``child_env``.
  * DF7 -- the child's firewall loads no module the interpreter had not
    loaded without it, and a SQLite module the child imports afterwards is
    still redirected.
  * DF8 -- in the test process, a listing through the globber every
    ``pathlib`` glob and rglob goes through, and its existence check, see
    the mirror and name what they find under the tree; uninstalled, the
    globber is as it was.
  * DF9 -- the summary line counts what the children kept off.
  * DF10 -- a child that cannot be covered does not run: it exits 70 and
    names why.
  * DF11 -- in the test process, every connect the firewall covers -- each
    package's and each DB-API module's, SQLite's and SQLCipher's -- opens
    the mirror.
  * DF12 -- a covered child that imports those modules after it started
    opens the mirror with each connect.
  * DF13 -- a path spelled through ``..``, with doubled slashes, with a
    leading ``//`` or climbing out of the working directory and back maps to
    the mirror.
  * DF14 -- a listing with no path, whose working directory is a data
    place, lists the mirror, in the test process and in a covered child.
  * DF15 -- the opens ``tarfile``, ``bz2`` and ``tokenize`` bound at import,
    and the calls that truncate, link, read a link, change an owner and make
    a pipe or a node, reach only the mirror; uninstalled, the bound opens
    are as they were.
  * DF16 -- a covered child started from an empty directory imports the
    package of the tree under test, and one whose path does not reach that
    tree is refused the package an install finds elsewhere, whose code never
    runs.
  * DF17 -- a session whose children cannot be covered stops before any
    suite, with status 70 and the reason; one whose own interpreter skips
    the user site while its children do not is measured covered and runs.
  * DF18 -- the summary counts every process that installed the firewall,
    one that kept nothing off and a fork of the test process included, and
    every launch the firewall does not reach, by reason and once each, a
    launch that goes on through ``os.posix_spawn`` included.

Local-only (the public distribution ships no tests). DF2, DF3, DF5, DF6, DF7,
DF9 and DF17 run a real pytest session on a fake tree carrying a copy of the
conftest and of the firewall -- and, from DF5 on, of its child hook -- and
read that session's own junit file. DF12, DF14, DF16 and DF18 start their
children from this process, under a firewall of their own on a scratch tree
that covers them as the session covers its own.
"""

import contextlib
import hashlib
import json
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
    "test_df1_a_data_path_or_a_database_in_the_tree_maps_to_the_mirror_and_nothing_else": 2.0,
    "test_df2_a_session_sees_a_fresh_install_and_leaves_the_real_places_as_they_were": 2.0,
    "test_df3_tracked_files_are_served_as_head_holds_them_and_nothing_untracked_shows": 2.0,
    "test_df4_a_uri_connect_to_a_data_place_opens_the_mirror_and_keeps_its_query": 2.0,
    "test_df5_a_child_started_with_the_inherited_environment_reaches_only_the_mirror": 2.0,
    "test_df6_a_child_with_a_copied_environment_or_the_componion_child_env_reaches_only_the_mirror": 2.0,
    "test_df7_the_child_hook_loads_nothing_new_and_a_later_sqlite_import_is_still_redirected": 2.0,
    "test_df8_a_pathlib_listing_in_this_process_reaches_the_mirror_and_names_the_tree": 2.0,
    "test_df9_the_summary_counts_what_the_children_kept_off": 2.0,
    "test_df10_a_child_that_cannot_be_covered_refuses_to_run_and_says_why": 2.0,
    "test_df11_every_connect_in_this_process_each_package_and_each_dbapi_module_opens_the_mirror": 2.0,
    "test_df12_a_covered_child_that_imports_each_connect_late_opens_the_mirror_with_each": 2.0,
    "test_df13_a_path_spelled_with_dots_doubled_slashes_or_a_climb_maps_to_the_mirror": 2.0,
    "test_df14_a_listing_with_no_path_whose_working_directory_is_a_data_place_lists_the_mirror": 2.0,
    "test_df15_the_opens_bound_at_import_and_the_link_pipe_node_calls_reach_only_the_mirror": 2.0,
    "test_df16_a_covered_child_imports_the_package_of_its_tree_and_never_one_an_install_finds_elsewhere": 2.0,
    "test_df17_a_session_whose_children_cannot_be_covered_stops_and_one_whose_children_can_runs": 2.0,
    "test_df18_the_summary_counts_every_covered_process_and_every_launch_the_firewall_does_not_reach": 2.0,
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


# ---------------------------------------------------------------------------
# DF5 to DF10 -- child processes, and the globber of the test process
# ---------------------------------------------------------------------------
_PLACES = ("data", "opti_oignon/data")
_KEY = b"the maintainer's master key"


def _covered_tree(tmp_path, canary):
    """A fake tree as ``_tree`` makes it, with the child hook, and a key and a store planted in each place."""
    tree = _tree(tmp_path, canary)
    site = _TESTS / "_firewall_site"
    if site.is_dir():
        shutil.copytree(site, tree / "tests" / "_firewall_site")
    for place in _PLACES:
        (tree / place / "planted.key").write_bytes(_KEY)
        store = sqlite3.connect(str(tree / place / "planted.db"))
        store.execute("create table private (text)")
        store.execute("insert into private values ('a private record')")
        store.commit()
        store.close()
    return tree


def _planted(tree):
    return {tree / place / name: _md5(tree / place / name) for place in _PLACES for name in ("planted.key", "planted.db")}


def _left_as_planted(tree, planted):
    for place in _PLACES:
        found = sorted(p.name for p in (tree / place).iterdir())
        assert found == ["planted.db", "planted.key"], f"a new entry in the real {place}: {found}"
    assert {path: _md5(path) for path in planted} == planted, "the planted files are byte-identical"


# What a child does in each data place, under a label of its own: an
# existence check of the planted key, a connect to the planted store, a
# connect that writes, a file written, a directory made, and four listings.
_CHILD = r"""import json
import os
import sqlite3
import sys
from pathlib import Path

root, label = Path(sys.argv[1]), sys.argv[2]
seen = {}
for place in (root / "data", root / "opti_oignon" / "data"):
    view = {"key": (place / "planted.key").exists()}
    store = sqlite3.connect(str(place / "planted.db"))
    view["tables"] = store.execute("select name from sqlite_master").fetchall()
    store.close()
    store = sqlite3.connect(str(place / (label + ".db")))
    store.execute("create table written (x)")
    store.commit()
    store.close()
    (place / ("written-" + label + ".txt")).write_text("from " + label)
    os.mkdir(place / ("made-" + label))
    view["listdir"] = sorted(os.listdir(place))
    view["scandir"] = sorted(entry.name for entry in os.scandir(place))
    view["glob"] = sorted(path.name for path in place.glob("*"))
    view["rglob"] = sorted(path.name for path in place.rglob("*"))
    seen[place.relative_to(root).as_posix()] = view
print(json.dumps(seen))
"""

# The canary of DF5 and DF6: it runs the child once per launch, in order,
# and checks each view against the mirror as the launches so far left it.
_CHILDREN_CANARY = r"""import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = {tests!r}
CHILD = {child!r}

{launches}

def test_each_child_finds_and_writes_the_mirror(tmp_path):
    labels = []
    for label, env in launches(tmp_path):
        labels.append(label)
        run = subprocess.run([sys.executable, "-c", CHILD, str(ROOT), label], env=env, cwd=tmp_path,
                             capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, (label, run.stderr[-2000:])
        seen = json.loads(run.stdout)
        written = {{"planted.db"}} | {{name for done in labels
                                     for name in (done + ".db", "made-" + done, "written-" + done + ".txt")}}
        expected = {{"key": False, "tables": [], "listdir": sorted(written), "scandir": sorted(written),
                    "glob": sorted(written), "rglob": sorted(written)}}
        assert seen == {{"data": expected, "opti_oignon/data": expected}}, (label, seen)
    for place in ("data", "opti_oignon/data"):
        for label in labels:
            assert (ROOT / place / ("written-" + label + ".txt")).read_text() == "from " + label, (place, label)
"""


def _children_canary(launches):
    return _CHILDREN_CANARY.format(tests=str(_TESTS), child=_CHILD, launches=launches)


# ---------------------------------------------------------------------------
# DF5 -- a child with the inherited environment reaches only the mirror
# ---------------------------------------------------------------------------
_INHERITED = """def launches(tmp_path):
    return [("inherited", None)]
"""


def test_df5_a_child_started_with_the_inherited_environment_reaches_only_the_mirror(tmp_path):
    tree = _covered_tree(tmp_path, _children_canary(_INHERITED))
    planted = _planted(tree)
    outcomes, output = _session(tree)
    assert outcomes == {"test_each_child_finds_and_writes_the_mirror": []}, (outcomes, output[-3000:])
    _left_as_planted(tree, planted)


# ---------------------------------------------------------------------------
# DF6 -- a copied environment, and the componion suites' child_env
# ---------------------------------------------------------------------------
_COPIED_AND_GARDEN = """def launches(tmp_path):
    sys.path.insert(0, TESTS)
    from _allium_garden_support import child_env

    garden = child_env(tmp_path)
    assert garden is not os.environ and "PYTHONPATH" in garden, "child_env builds an environment of its own"
    return [("copied", dict(os.environ, DF_SHAPE="copied")), ("garden", garden)]
"""


def test_df6_a_child_with_a_copied_environment_or_the_componion_child_env_reaches_only_the_mirror(tmp_path):
    tree = _covered_tree(tmp_path, _children_canary(_COPIED_AND_GARDEN))
    planted = _planted(tree)
    outcomes, output = _session(tree)
    assert outcomes == {"test_each_child_finds_and_writes_the_mirror": []}, (outcomes, output[-3000:])
    _left_as_planted(tree, planted)


# ---------------------------------------------------------------------------
# DF7 -- the child hook loads nothing new; a later SQLite import is redirected
# ---------------------------------------------------------------------------
_FOOTPRINT = r"""import sys

loaded = sorted(sys.modules)
import json
from pathlib import Path

if sys.argv[2] == "write":
    import sqlite3

    store = sqlite3.connect(str(Path(sys.argv[1]) / "opti_oignon" / "data" / "late.db"))
    store.execute("create table late (x)")
    store.commit()
    store.close()
print(json.dumps(loaded))
"""

_FOOTPRINT_CANARY = r"""import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHILD = {child!r}
HOOK = ("OO_TEST_FIREWALL_ROOT", "OO_TEST_FIREWALL_MIRROR")


def _loaded(mode, env):
    run = subprocess.run([sys.executable, "-c", CHILD, str(ROOT), mode], env=env,
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, (mode, run.stderr[-2000:])
    return set(json.loads(run.stdout))


def test_the_hook_loads_nothing_new_and_a_later_sqlite_import_is_redirected():
    bare = {{name: value for name, value in os.environ.items() if name not in HOOK}}
    bare["PYTHONPATH"] = os.pathsep.join(entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                                         if entry and os.path.basename(entry.rstrip(os.sep)) != "_firewall_site")
    covered = _loaded("write", None)
    uncovered = _loaded("measure", bare)
    assert "sqlite3" not in uncovered, "control: SQLite is not loaded at startup, so the import below is a late one"
    assert (covered - uncovered, uncovered - covered) == ({{"usercustomize"}}, set()), (
        sorted(covered - uncovered), sorted(uncovered - covered))
    store = sqlite3.connect(str(ROOT / "opti_oignon" / "data" / "late.db"))
    tables = store.execute("select name from sqlite_master").fetchall()
    store.close()
    assert tables == [("late",)], "the late import wrote the mirror, read back through this session's firewall"
"""


def test_df7_the_child_hook_loads_nothing_new_and_a_later_sqlite_import_is_still_redirected(tmp_path):
    tree = _covered_tree(tmp_path, _FOOTPRINT_CANARY.format(child=_FOOTPRINT))
    planted = _planted(tree)
    outcomes, output = _session(tree)
    assert outcomes == {"test_the_hook_loads_nothing_new_and_a_later_sqlite_import_is_redirected": []}, (
        outcomes, output[-3000:])
    _left_as_planted(tree, planted)


# ---------------------------------------------------------------------------
# DF8 -- a pathlib listing in this process reaches the mirror
# ---------------------------------------------------------------------------
def test_df8_a_pathlib_listing_in_this_process_reaches_the_mirror_and_names_the_tree(tmp_path):
    import glob

    root, mirror = tmp_path / "tree", tmp_path / "mirror"
    for place in _PLACES:
        (root / place / "projects" / "a private project").mkdir(parents=True)
        (root / place / "planted.key").write_bytes(_KEY)
    real = {place: sorted(p.relative_to(root).as_posix() for p in (root / place).rglob("*")) for place in _PLACES}
    globber = getattr(glob, "_StringGlobber", None)
    bound = {name: globber.__dict__[name] for name in ("scandir", "lstat")} if globber else {}
    firewall = _module().DataFirewall(root, mirror=mirror, seed=False)
    firewall.install()
    try:
        for place in _PLACES:
            (root / place / "witness.txt").write_text("in the mirror", encoding="utf-8")
        seen = {
            place: {
                "glob": sorted(p.name for p in (root / place).glob("*")),
                "rglob": sorted(p.relative_to(root).as_posix() for p in (root / place).rglob("*")),
                "exists": [p.name for p in (root / place).glob("planted.key")],
                "string": sorted(os.path.basename(p) for p in glob.glob(str(root / place / "*"))),
            }
            for place in _PLACES
        }
        tree = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    finally:
        firewall.uninstall()
    assert seen == {
        place: {"glob": ["witness.txt"], "rglob": [f"{place}/witness.txt"], "exists": [], "string": ["witness.txt"]}
        for place in _PLACES
    }, seen
    assert tree == ["data", "data/witness.txt", "opti_oignon", "opti_oignon/data", "opti_oignon/data/witness.txt"], (
        "a walk of the whole tree lists the mirror inside the data places, named under the tree", tree)
    after = {place: sorted(p.relative_to(root).as_posix() for p in (root / place).rglob("*")) for place in _PLACES}
    assert after == real, "the real places are left as they were"
    assert real["data"] == ["data/planted.key", "data/projects", "data/projects/a private project"], "control"
    if globber is not None:
        assert {name: globber.__dict__[name] for name in bound} == bound, "uninstalled, the globber is as it was"
    assert set(_PLACES) <= set().union(*firewall.redirected.values()), "each listing is counted"


# ---------------------------------------------------------------------------
# DF9 -- the summary line counts what the children kept off
# ---------------------------------------------------------------------------
_COUNTED = r'''import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHILD = """import sys
from pathlib import Path
root = Path(sys.argv[1])
(root / "data" / "one.txt").write_text("one")
(root / "opti_oignon" / "data" / "two.txt").write_text("two")
"""


def test_a_child_writes_two_paths():
    run = subprocess.run([sys.executable, "-c", CHILD, str(ROOT)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-2000:]
    assert (ROOT / "data" / "one.txt").read_text() == "one"
'''


def test_df9_the_summary_counts_what_the_children_kept_off(tmp_path):
    import re

    tree = _covered_tree(tmp_path, _COUNTED)
    planted = _planted(tree)
    outcomes, output = _session(tree)
    assert outcomes == {"test_a_child_writes_two_paths": []}, (outcomes, output[-3000:])
    counted = re.findall(r"child processes: (\d+) path\(s\) kept off in (\d+) process\(es\)", output)
    assert counted == [("2", "1")], output[-1500:]
    _left_as_planted(tree, planted)


# ---------------------------------------------------------------------------
# DF10 -- a child that cannot be covered refuses to run
# ---------------------------------------------------------------------------
def test_df10_a_child_that_cannot_be_covered_refuses_to_run_and_says_why(tmp_path):
    root, mirror = tmp_path / "tree", tmp_path / "mirror"
    (root / "data").mkdir(parents=True)
    site = _TESTS / "_firewall_site"
    broken = tmp_path / "broken"
    (broken / "_firewall_site").mkdir(parents=True)
    if site.is_dir():
        shutil.copy2(site / "usercustomize.py", broken / "_firewall_site" / "usercustomize.py")
    (broken / "_data_firewall.py").write_text("raise RuntimeError('a firewall that cannot load')\n", encoding="utf-8")
    code = "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('ran'); print('ran')"
    cases = {
        "unpaired": (site, f"{root}{os.pathsep}{root}", str(mirror)),
        "relative": (site, "tree", str(mirror)),
        "unloadable": (broken / "_firewall_site", str(root), str(mirror)),
    }
    seen, reasons = {}, {}
    for case, (hook, roots, mirrors) in cases.items():
        env = dict(os.environ, OO_TEST_FIREWALL_ROOT=roots, OO_TEST_FIREWALL_MIRROR=mirrors, PYTHONPATH=str(hook))
        run = subprocess.run([sys.executable, "-c", code, str(root / "data" / case)], env=env, cwd=tmp_path,
                             capture_output=True, text=True, timeout=60)
        seen[case] = (run.returncode, run.stdout, "refuses to run uncovered" in run.stderr)
        reasons[case] = run.stderr
    assert seen == {case: (70, "", True) for case in cases}, (seen, reasons)
    assert "a firewall that cannot load" in reasons["unloadable"], reasons["unloadable"]
    assert sorted(p.name for p in (root / "data").iterdir()) == [], "no refused child ran a line of its own"
    control = subprocess.run([sys.executable, "-c", "print('ran')"], cwd=tmp_path, capture_output=True, text=True,
                             env=dict(os.environ, OO_TEST_FIREWALL_ROOT="", OO_TEST_FIREWALL_MIRROR="",
                                      PYTHONPATH=str(site)), timeout=60)
    assert (control.returncode, control.stdout, control.stderr) == (0, "ran\n", ""), "control: no variables, no hook"


# ---------------------------------------------------------------------------
# DF11 to DF18 -- what the review of DF5 to DF10 found open
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _covering(root, mirror):
    """A firewall on ``root``, installed in this process and carried to the children started inside."""
    firewall = _module().DataFirewall(root, mirror=mirror, seed=False)
    firewall.install()
    try:
        uncovered = firewall.cover_children()
        assert uncovered is None, f"the scratch tree's children are covered: {uncovered}"
        yield firewall
    finally:
        firewall.uncover_children()
        firewall.uninstall()


def _entries(folder):
    return sorted(path.name for path in Path(folder).iterdir())


def _connect_modules():
    """Every module whose ``connect`` the firewall covers: SQLite's package and DB-API module, and SQLCipher's when installed."""
    import sqlite3.dbapi2

    modules = {"sqlite3": sqlite3, "sqlite3.dbapi2": sqlite3.dbapi2}
    try:
        import sqlcipher3
        import sqlcipher3.dbapi2
    except Exception:  # noqa: BLE001 - no SQLCipher here: the standard library alone
        return modules
    modules.update({"sqlcipher3": sqlcipher3, "sqlcipher3.dbapi2": sqlcipher3.dbapi2})
    return modules


# ---------------------------------------------------------------------------
# DF11 -- every connect of the test process opens the mirror
# ---------------------------------------------------------------------------
def test_df11_every_connect_in_this_process_each_package_and_each_dbapi_module_opens_the_mirror(tmp_path):
    root, mirror = tmp_path / "tree", tmp_path / "mirror"
    (root / "data").mkdir(parents=True)
    modules = _connect_modules()
    firewall = _module().DataFirewall(root, mirror=mirror, seed=False)
    firewall.install()
    try:
        for name, module in modules.items():
            store = module.connect(str(root / "data" / f"{name}.db"))
            store.execute("create table written (x)")
            store.commit()
            store.close()
    finally:
        firewall.uninstall()
    assert {"sqlite3", "sqlite3.dbapi2"} <= set(modules), "control: both SQLite names were exercised"
    assert _entries(root / "data") == [], "no connect reached the real place"
    assert _entries(mirror / "data") == sorted(f"{name}.db" for name in modules), "each connect opened the mirror"


# ---------------------------------------------------------------------------
# DF12 -- in a covered child, every connect imported late opens the mirror
# ---------------------------------------------------------------------------
_LATE_CONNECTS = r"""import json
import os
import sys

root = sys.argv[1]
late = not any(name.split(".")[0] in ("sqlite3", "sqlcipher3") for name in sys.modules)
import sqlite3.dbapi2

modules = {"sqlite3": sqlite3, "sqlite3.dbapi2": sqlite3.dbapi2}
try:
    import sqlcipher3.dbapi2

    modules.update({"sqlcipher3": sqlcipher3, "sqlcipher3.dbapi2": sqlcipher3.dbapi2})
except Exception:
    pass
for name, module in modules.items():
    store = module.connect(os.path.join(root, "data", name + ".db"))
    store.execute("create table written (x)")
    store.commit()
    store.close()
print(json.dumps({"late": late, "names": sorted(modules)}))
"""


def test_df12_a_covered_child_that_imports_each_connect_late_opens_the_mirror_with_each(tmp_path):
    root, mirror = tmp_path / "tree", tmp_path / "mirror"
    (root / "data").mkdir(parents=True)
    with _covering(root, mirror):
        run = subprocess.run([sys.executable, "-c", _LATE_CONNECTS, str(root)], cwd=tmp_path,
                             capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-2000:]
    seen = json.loads(run.stdout)
    assert seen["late"] and {"sqlite3", "sqlite3.dbapi2"} <= set(seen["names"]), (
        "control: the child loaded no SQLite before its own import, and used both SQLite names", seen)
    assert set(_connect_modules()) == set(seen["names"]), "the child exercised every connect this interpreter has"
    assert _entries(root / "data") == [], "no connect of the child reached the real place"
    assert _entries(mirror / "data") == sorted(f"{name}.db" for name in seen["names"]), "each opened the mirror"


# ---------------------------------------------------------------------------
# DF13 -- a path is judged by the place it reaches, however it is spelled
# ---------------------------------------------------------------------------
def test_df13_a_path_spelled_with_dots_doubled_slashes_or_a_climb_maps_to_the_mirror(tmp_path, monkeypatch):
    base = tmp_path / "base"
    root, mirror, other = base / "tree", tmp_path / "mirror", base / "other"
    (root / "data").mkdir(parents=True)
    other.mkdir()
    spellings = {
        "dotdot": f"{other}/../tree/data/dotdot.txt",
        "doubled": f"{base}//tree//data/doubled.txt",
        "leading": "/" + str(root / "data" / "leading.txt"),
        "relative": "../tree/data/relative.txt",
    }
    assert spellings["leading"].startswith("//") and not spellings["leading"].startswith("///"), "control"
    monkeypatch.chdir(other)
    firewall = _module().DataFirewall(root, mirror=mirror, seed=False)
    mapped = {name: firewall.target(path) for name, path in spellings.items()}
    assert mapped == {name: str(mirror / "data" / f"{name}.txt") for name in spellings}, mapped
    firewall.install()
    try:
        for name, path in spellings.items():
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(name)
    finally:
        firewall.uninstall()
    assert _entries(root / "data") == [], "no spelling reached the real place"
    assert _entries(mirror / "data") == sorted(f"{name}.txt" for name in spellings)


# ---------------------------------------------------------------------------
# DF14 -- a listing with no path, in a data place, lists the mirror
# ---------------------------------------------------------------------------
_NO_PATH_LISTING = r"""import json
import os

with open("child.txt", "w", encoding="utf-8") as handle:
    handle.write("from the child")
print(json.dumps({"listdir": sorted(os.listdir()), "none": sorted(os.listdir(None)),
                  "scandir": sorted(entry.name for entry in os.scandir())}))
"""


def test_df14_a_listing_with_no_path_whose_working_directory_is_a_data_place_lists_the_mirror(tmp_path, monkeypatch):
    root, mirror = tmp_path / "tree", tmp_path / "mirror"
    place = root / "data"
    place.mkdir(parents=True)
    (place / "planted.key").write_bytes(_KEY)
    with _covering(root, mirror):
        monkeypatch.chdir(place)
        with open("witness.txt", "w", encoding="utf-8") as handle:
            handle.write("in the mirror")
        here = {"listdir": sorted(os.listdir()), "none": sorted(os.listdir(None)),
                "scandir": sorted(entry.name for entry in os.scandir())}
        run = subprocess.run([sys.executable, "-c", _NO_PATH_LISTING], cwd=place,
                             capture_output=True, text=True, timeout=60)
        monkeypatch.chdir(tmp_path)
    assert here == {"listdir": ["witness.txt"], "none": ["witness.txt"], "scandir": ["witness.txt"]}, here
    assert run.returncode == 0, run.stderr[-2000:]
    child = json.loads(run.stdout)
    assert child == {key: ["child.txt", "witness.txt"] for key in ("listdir", "none", "scandir")}, child
    assert _entries(place) == ["planted.key"] and (place / "planted.key").read_bytes() == _KEY, "the real place as planted"


# ---------------------------------------------------------------------------
# DF15 -- the opens bound at import, and the calls that make links, pipes and nodes
# ---------------------------------------------------------------------------
def _raised(call):
    try:
        call()
    except OSError as error:
        return type(error).__name__
    return "no error"


def test_df15_the_opens_bound_at_import_and_the_link_pipe_node_calls_reach_only_the_mirror(tmp_path):
    import bz2
    import stat
    import tarfile
    import tokenize

    root, mirror = tmp_path / "tree", tmp_path / "mirror"
    place = root / "data"
    place.mkdir(parents=True)
    (place / "planted.key").write_bytes(_KEY)
    (place / "planted.py").write_text("PLANTED = True\n", encoding="utf-8")
    planted = {path: _md5(path) for path in place.iterdir()}
    member = tmp_path / "member.txt"
    member.write_text("a member", encoding="utf-8")
    outside = tmp_path / "outside.tar"
    with tarfile.open(outside, "w") as tar:
        tar.add(member, arcname="extracted.txt")
    bound = {"tarfile": tarfile.bltn_open, "bz2": bz2._builtin_open, "tokenize": tokenize._builtin_open}
    firewall = _module().DataFirewall(root, mirror=mirror, seed=False)
    firewall.install()
    seen = {}
    try:
        with tarfile.open(place / "archive.tar", "w") as tar:
            tar.add(member, arcname="member.txt")
        with tarfile.open(outside) as tar:
            tar.extractall(place, filter="data")
        with bz2.open(place / "packed.bz2", "wb") as packed:
            packed.write(b"packed")
        seen["tokenize planted"] = _raised(lambda: tokenize.open(place / "planted.py").close())
        (place / "cut.txt").write_text("abcdef", encoding="utf-8")
        os.truncate(place / "cut.txt", 2)
        seen["truncate planted"] = _raised(lambda: os.truncate(place / "planted.key", 1))
        os.link(place / "cut.txt", place / "hard.txt")
        os.symlink("cut.txt", place / "soft")
        seen["readlink"] = os.readlink(place / "soft")
        seen["readlink planted"] = _raised(lambda: os.readlink(place / "planted.key"))
        os.mkfifo(place / "fifo")
        os.mknod(place / "node", 0o600 | stat.S_IFREG)
        seen["chown planted"] = _raised(lambda: os.chown(place / "planted.key", os.getuid(), os.getgid()))
        seen["lchown planted"] = _raised(lambda: os.lchown(place / "planted.key", os.getuid(), os.getgid()))
    finally:
        firewall.uninstall()
    after = {"tarfile": tarfile.bltn_open, "bz2": bz2._builtin_open, "tokenize": tokenize._builtin_open}
    assert seen == {
        "tokenize planted": "FileNotFoundError", "truncate planted": "FileNotFoundError", "readlink": "cut.txt",
        "readlink planted": "FileNotFoundError", "chown planted": "FileNotFoundError",
        "lchown planted": "FileNotFoundError",
    }, ("a planted file is absent from the mirror, as from a fresh checkout", seen)
    assert _entries(place) == ["planted.key", "planted.py"], "nothing new in the real place"
    assert {path: _md5(path) for path in planted} == planted, "the planted files are byte-identical"
    assert _entries(mirror / "data") == [
        "archive.tar", "cut.txt", "extracted.txt", "fifo", "hard.txt", "node", "packed.bz2", "soft"]
    assert (mirror / "data" / "cut.txt").read_text(encoding="utf-8") == "ab", "the truncation reached the mirror's file"
    assert after == bound, "uninstalled, the bound opens are as they were"


# ---------------------------------------------------------------------------
# DF16 -- a covered child imports the package of its own tree
# ---------------------------------------------------------------------------
_INSTALLED_ELSEWHERE = r'''import importlib.util
import json
import sys

elsewhere = sys.argv[1]
# The editable install of this interpreter maps the package to one checkout; point it at another, as a second
# worktree finds it. Nothing runs when there is not exactly one such install to point.
installs = [module for name, module in list(sys.modules.items())
            if name.startswith("__editable__") and "opti_oignon" in getattr(module, "MAPPING", {})]
if len(installs) != 1:
    print(json.dumps({"installs": len(installs)}))
    raise SystemExit(0)
installs[0].MAPPING["opti_oignon"] = elsewhere + "/opti_oignon"
'''

_RESOLVE = _INSTALLED_ELSEWHERE + r'''
print(json.dumps({"installs": 1, "origin": importlib.util.find_spec("opti_oignon").origin}))
'''

_IMPORT = _INSTALLED_ELSEWHERE + r'''
try:
    import opti_oignon  # noqa: F401
    refused = None
except ImportError as error:
    refused = str(error)
print(json.dumps({"installs": 1, "refused": refused}))
'''


def test_df16_a_covered_child_imports_the_package_of_its_tree_and_never_one_an_install_finds_elsewhere(tmp_path):
    root, mirror, elsewhere, empty = (tmp_path / name for name in ("tree", "mirror", "elsewhere", "empty"))
    (root / "opti_oignon" / "data").mkdir(parents=True)
    (root / "opti_oignon" / "__init__.py").write_text("", encoding="utf-8")
    ran = elsewhere / "ran.txt"
    (elsewhere / "opti_oignon").mkdir(parents=True)
    (elsewhere / "opti_oignon" / "__init__.py").write_text(f"open({str(ran)!r}, 'w').write('ran')\n", encoding="utf-8")
    empty.mkdir()
    with _covering(root, mirror):
        resolved = subprocess.run([sys.executable, "-c", _RESOLVE, str(elsewhere)], cwd=empty,
                                  capture_output=True, text=True, timeout=60)
        sites = os.pathsep.join(entry for entry in os.environ["PYTHONPATH"].split(os.pathsep)
                                if os.path.basename(entry) == "_firewall_site")
        imported = subprocess.run([sys.executable, "-c", _IMPORT, str(elsewhere)], cwd=empty,
                                  env=dict(os.environ, PYTHONPATH=sites), capture_output=True, text=True, timeout=60)
    assert resolved.returncode == 0, resolved.stderr[-2000:]
    resolution = json.loads(resolved.stdout)
    assert resolution["installs"] == 1, "control: this interpreter has one editable install of the package to point"
    assert resolution["origin"] == str(root / "opti_oignon" / "__init__.py"), (
        "from an empty directory, the package of the tree under test", resolved.stdout)
    assert imported.returncode == 0, imported.stderr[-2000:]
    refused = json.loads(imported.stdout)["refused"]
    assert refused and "data firewall" in refused and str(elsewhere) in refused, refused
    assert not ran.exists(), "the package the install found elsewhere never ran"


# ---------------------------------------------------------------------------
# DF17 -- a session whose children cannot be covered does not run
# ---------------------------------------------------------------------------
_ONE_CHILD = r'''import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_a_child_writes_a_data_place():
    run = subprocess.run([sys.executable, "-c", "import sys; open(sys.argv[1], 'w').write('child')",
                          str(ROOT / "data" / "from-child.txt")], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-2000:]
    assert (ROOT / "data" / "from-child.txt").read_text() == "child"
'''


def _session_run(tree, *, flags=(), extra=None):
    """Run the canary suite in its tree; the process, and name -> failure messages, or None without a junit file."""
    junit = tree.parent / "junit.xml"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(extra or {}))
    env.pop("PYTEST_ADDOPTS", None)
    run = subprocess.run(
        [sys.executable, *flags, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
         "--rootdir", str(tree), f"--junitxml={junit}", str(tree / "tests" / "test_canary.py")],
        cwd=tree, env=env, capture_output=True, text=True, timeout=180,
    )
    if not junit.exists():
        return run, None
    outcomes = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        reds = outcomes.setdefault(case.get("name"), [])
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is not None:
                reds.append(node.get("message") or "")
    return run, outcomes


def test_df17_a_session_whose_children_cannot_be_covered_stops_and_one_whose_children_can_runs(tmp_path):
    stopped_tree = _covered_tree(tmp_path / "stopped", _ONE_CHILD)
    running_tree = _covered_tree(tmp_path / "running", _ONE_CHILD)
    planted = {tree: _planted(tree) for tree in (stopped_tree, running_tree)}
    stopped, stopped_outcomes = _session_run(stopped_tree, extra={"PYTHONNOUSERSITE": "1"})
    running, running_outcomes = _session_run(running_tree, flags=("-s",))
    assert (stopped.returncode, stopped_outcomes) == (70, None), (
        "no child of this session runs the hook: it stops before any suite", stopped.returncode, stopped_outcomes,
        stopped.stdout[-1500:], stopped.stderr[-1500:])
    assert "refuses to run uncovered" in stopped.stderr and "user site" in stopped.stderr, stopped.stderr[-1500:]
    assert (running.returncode, running_outcomes) == (0, {"test_a_child_writes_a_data_place": []}), (
        "its own interpreter skips the user site, its children do not: measured covered, it runs",
        running.returncode, running_outcomes, running.stdout[-1500:], running.stderr[-1500:])
    for tree, files in planted.items():
        _left_as_planted(tree, files)


# ---------------------------------------------------------------------------
# DF18 -- the summary counts every covered process and every launch not reached
# ---------------------------------------------------------------------------
def test_df18_the_summary_counts_every_covered_process_and_every_launch_the_firewall_does_not_reach(tmp_path):
    root, mirror, empty = tmp_path / "tree", tmp_path / "mirror", tmp_path / "empty"
    place = root / "data"
    place.mkdir(parents=True)
    empty.mkdir()
    variables = (_module().ROOTS_VARIABLE, _module().MIRRORS_VARIABLE)
    write = "import sys; open(sys.argv[1], 'w').write('one')"
    with _covering(root, mirror) as firewall:
        bare = {name: value for name, value in os.environ.items() if name not in variables}
        launches = [
            ([sys.executable, "-c", "pass"], None, empty),
            ([sys.executable, "-c", write, str(place / "one.txt")], None, empty),
            ([sys.executable, "-I", "-c", "pass"], None, empty),
            ([sys.executable, "-s", "-c", "pass"], None, empty),
            ([sys.executable, "-E", "-c", "pass"], None, empty),
            ([sys.executable, "-S", "-c", "pass"], None, empty),
            ([sys.executable, "-c", "pass"], dict(os.environ, PYTHONPATH=str(empty)), empty),
            ([sys.executable, "-c", "pass"], dict(os.environ, PYTHONNOUSERSITE="1"), empty),
            ([sys.executable, "-c", "pass"], bare, empty),
            (["/bin/sh", "-c", "true"], None, place),
            (["/bin/sh", "-c", "true"], None, empty),
        ]
        codes = [subprocess.run(argv, env=env, cwd=cwd, capture_output=True, timeout=60).returncode
                 for argv, env, cwd in launches]
        pid = os.fork()
        if pid == 0:
            try:
                with open(place / "forked.txt", "w", encoding="utf-8") as handle:
                    handle.write("forked")
            finally:
                os._exit(0)
        os.waitpid(pid, 0)
        seen = firewall.children_seen()
        line = firewall.summary()
    assert codes == [0] * len(launches), codes
    assert (seen["installed"], seen["paths"], seen["processes"]) == (3, 2, 2), (
        "installed: the child that kept nothing off, the one that wrote, and the fork; paths: one each for the last two",
        seen)
    assert seen["launches"] == {
        "-I": 1, "-s": 1, "-E": 1, "-S": 1, "PYTHONPATH without the site": 1, "PYTHONNOUSERSITE": 1,
        "without the firewall's variables": 1, "not Python, in a data place": 1,
    }, seen["launches"]
    assert "child processes: 2 path(s) kept off in 2 process(es), 3 installed the firewall" in line, line
    assert "; 8 launch(es) it does not reach (" in line, line
    assert _entries(place) == [], "nothing reached the real place"
    assert _entries(mirror / "data") == ["forked.txt", "one.txt"]
    spawned = tmp_path / "spawned"
    (spawned / "data").mkdir(parents=True)
    with _covering(spawned, tmp_path / "spawned-mirror") as firewall:
        run = subprocess.run([sys.executable, "-I", "-c", "pass"], close_fds=False, capture_output=True, timeout=60)
        once = firewall.children_seen()["launches"]
    assert (run.returncode, once) == (0, {"-I": 1}), ("a launch that goes on through os.posix_spawn is counted once",
                                                      run.returncode, once)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
