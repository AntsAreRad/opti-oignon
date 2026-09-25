#!/usr/bin/env python3
"""Contracts for the ladder's guard runner: a guard sees the data places as CI does.

The ladder's guard tier ran each guard as ``python3 guard.py`` on the
maintainer's tree. A guard that imports the application -- the published-prose
guard builds the OpenAPI schema from it -- runs the application's module-level
singletons, and those open the stores of the data places, the branch store
with the master key among them. In CI the tree is a fresh checkout and nothing
real is there; on the maintainer's machine every local run of the tier opened
the real stores. ``tests/_guard_mirrored.py`` runs a guard in its own process
under the firewall the test session installs, the mirror seeded from HEAD, and
the ladder's guard tier goes through it.

  * GM1 -- a guard run through the runner that reads and writes the data
    places sees the mirror: a planted key is absent, the guard reads back what
    it wrote, and the real places are left exactly as they were.
  * GM2 -- the runner cannot be told apart from a direct run: the guard runs
    as ``__main__`` with its own directory first on ``sys.path`` and its
    arguments on ``sys.argv``, and a clean end, an integer status and a
    refusal message all come out the same.
  * GM3 -- end to end, the ladder's guard tier on a fake tree passes its
    guards and leaves that tree's data places as they were.

Local-only (the public distribution ships no tests). Each contract works on a
fake tree carrying copies of the runner and of the firewall.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_REPO = _TESTS.parent
_KEY = b"the maintainer's master key"

_TOUCH = '''import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
print("key seen" if (ROOT / "data" / ".keyfile").exists() else "key absent")
(ROOT / "data" / "written.txt").write_text("from the guard")
print("reread:", (ROOT / "data" / "written.txt").read_text())
store = sqlite3.connect(str(ROOT / "opti_oignon" / "data" / "store.db"))
store.execute("create table t (x)")
store.commit()
store.close()
'''

_PROBE = '''import json
import sys

import sibling_helper

print(json.dumps({"name": __name__, "path0": sys.path[0], "argv": sys.argv[1:],
                  "helper": sibling_helper.VALUE}))
wanted = sys.argv[1] if len(sys.argv) > 1 else None
if wanted == "message":
    raise SystemExit("refused: a message")
if wanted is not None:
    raise SystemExit(int(wanted))
'''


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _tree(tmp_path, guards):
    """A fake tree with the runner, the firewall, a planted key and ``guards``."""
    tree = tmp_path / "tree"
    (tree / "tests").mkdir(parents=True)
    for name in ("_guard_mirrored.py", "_data_firewall.py"):
        shutil.copy2(_TESTS / name, tree / "tests" / name)
    scripts = tree / ".github" / "scripts"
    scripts.mkdir(parents=True)
    for name, text in guards.items():
        (scripts / name).write_text(text, encoding="utf-8")
    (tree / "data").mkdir()
    (tree / "data" / ".keyfile").write_bytes(_KEY)
    (tree / "opti_oignon" / "data").mkdir(parents=True)
    return tree


def _run(argv, cwd):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)


def _left_as_they_were(tree):
    assert sorted(p.name for p in (tree / "data").iterdir()) == [".keyfile"], "nothing written in the real place"
    assert (tree / "data" / ".keyfile").read_bytes() == _KEY, "the planted key is untouched"
    assert list((tree / "opti_oignon" / "data").iterdir()) == [], "no store created in the real place"


# ---------------------------------------------------------------------------
# GM1 -- a guard reads and writes the mirror, never the real places
# ---------------------------------------------------------------------------
def test_gm1_a_guard_run_through_the_runner_sees_the_mirror_and_leaves_the_real_places(tmp_path):
    tree = _tree(tmp_path, {"touch_guard.py": _TOUCH})
    before = _md5(tree / "data" / ".keyfile")
    run = _run([sys.executable, str(tree / "tests" / "_guard_mirrored.py"),
                str(tree / ".github" / "scripts" / "touch_guard.py")], cwd=tree)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.splitlines()[:2] == ["key absent", "reread: from the guard"], run.stdout
    _left_as_they_were(tree)
    assert _md5(tree / "data" / ".keyfile") == before


# ---------------------------------------------------------------------------
# GM2 -- the runner cannot be told apart from a direct run
# ---------------------------------------------------------------------------
def test_gm2_the_runner_cannot_be_told_apart_from_a_direct_run(tmp_path):
    tree = _tree(tmp_path, {"probe_guard.py": _PROBE, "sibling_helper.py": "VALUE = 42\n"})
    guard = tree / ".github" / "scripts" / "probe_guard.py"
    for args in ([], ["0"], ["1"], ["2"], ["message"]):
        direct = _run([sys.executable, str(guard), *args], cwd=tree)
        mirrored = _run([sys.executable, str(tree / "tests" / "_guard_mirrored.py"), str(guard), *args], cwd=tree)
        assert mirrored.returncode == direct.returncode, (args, direct.stderr, mirrored.stderr)
        seen_direct = json.loads(direct.stdout.splitlines()[0])
        seen_mirrored = json.loads(mirrored.stdout.splitlines()[0])
        assert Path(seen_mirrored.pop("path0")).resolve() == Path(seen_direct.pop("path0")).resolve(), args
        assert seen_mirrored == seen_direct == {"name": "__main__", "argv": args, "helper": 42}, args
        assert mirrored.stderr.strip().splitlines()[-1:] == direct.stderr.strip().splitlines()[-1:], args
    assert direct.returncode == 1 and "refused: a message" in direct.stderr, "control: the message case refuses"


# ---------------------------------------------------------------------------
# GM3 -- the ladder's guard tier, end to end on a fake tree
# ---------------------------------------------------------------------------
def test_gm3_the_ladder_guard_tier_leaves_the_fake_trees_data_places_as_they_were(tmp_path):
    tree = _tree(tmp_path, {"touch_guard.py": _TOUCH, "quiet_guard.py": "raise SystemExit(0)\n"})
    (tree / "scripts").mkdir()
    shutil.copy2(_REPO / "scripts" / "ladder.sh", tree / "scripts" / "ladder.sh")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CLAUDE_PROJECT_DIR=str(tree))
    run = subprocess.run(["bash", str(tree / "scripts" / "ladder.sh"), "t2"], cwd=tree, env=env,
                         capture_output=True, text=True, timeout=300)
    assert "PASS  touch_guard.py" in run.stdout and "PASS  quiet_guard.py" in run.stdout, run.stdout
    assert "2 guard(s) run" in run.stdout, run.stdout
    _left_as_they_were(tree)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
