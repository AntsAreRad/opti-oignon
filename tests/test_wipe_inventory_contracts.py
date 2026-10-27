#!/usr/bin/env python3
"""Contracts for the inventory that keeps the account wipe true: every store classified, the copies of words emptied.

The account wipe answered with three hand-written lists, said to be "completed against the full at-rest inventory".
Nothing tied them to the code: the package spells some sixty database names and a dozen folders, and the lists
missed the knowledge base -- which the wipe deleted in name only, looking for collections nothing ever names so --,
the coding agent's decision preferences, the command line's history, the skills, the note attachments, the project
files; three of their labels were wrong. Every store is now classified in one inventory (``store_inventory``), a
census keeps it complete against the code, and the wipe's lists are held equal to it. In single-user mode, for the
install's one user, the wipe also empties the stores that only keep copies of the user's words, overwritten.

  * WI1 -- EVERY DATABASE NAME THE PACKAGE SPELLS IS CLASSIFIED: a string of one of its modules, a value of one of
    its YAML files; every name the inventory gives a store is still spelled by a module it names, and every name it
    calls no store is still spelled somewhere.
  * WI2 -- EVERY NAME JOINED TO A DATA FOLDER IS CLASSIFIED.
  * WI3 -- THE INVENTORY'S WORDS ARE CLOSED: what a store holds and what the wipe does with it come from their
    lists; a store of words is never "not the user's"; no name is a store and not one; every label is unique, and
    every module named is there.
  * WI4 -- THE WIPE'S LISTS ARE THE INVENTORY'S, in its order: not covered, removed for the one user, kept by design.
  * WI5 -- THE CODING AGENT'S DECISION PREFERENCES ARE EMPTIED, OVERWRITTEN, their summary with them; a store that is
    not there is not created, and one that cannot be emptied raises.
  * WI6 -- THE HUMANIZER'S FEEDBACK IS EMPTIED, OVERWRITTEN: the texts it compared and their ratings, through the
    engine or, when it did not start, the store its configuration names, whatever its name.
  * WI7 -- THE LEARNED ROUTER FORGETS EVERY QUERY AND THE MODEL LEARNED FROM THEM: its samples and decisions
    overwritten, its log emptied, the model file and its MAC overwritten whole then removed, the model in memory
    dropped; a training under way never puts its model back; a link in the model's place is refused, never followed.
  * WI8 -- THE COMMAND LINE'S HISTORY IS REMOVED, EACH FILE OVERWRITTEN FIRST: importing the module makes no folder;
    a file that is not the history's stays, and the folder with it; one that cannot be removed raises, after the
    others went.
  * WI9 -- THE FEEDBACK STORE IS EMPTIED WHOLE, the entries given outside any conversation included, overwritten, its
    log emptied; a store that did not start raises.
  * WI10 -- THE KNOWLEDGE BASE'S SEARCH LOG IS EMPTIED, OVERWRITTEN, AND ITS COLLECTIONS STAY; a store that is not
    there is not created.
  * WI11 -- THE SINGLE-USER WIPE EMPTIES THE COPIES THROUGH THEIR REAL MODULES, for the install's one user, and names
    one it cannot empty; a multi-user wipe, one asked for another name, or one that cannot read the mode empties
    none and names them.
  * WI12 -- THE CHAT'S KNOWLEDGE BASE SEARCH LOGS NO CITATION.
  * WI13 -- EVERY STORAGE STATEMENT OF THE DECISION PREFERENCES IS A CONSTANT, AND ONE IS BOUND (supersedes b33 of
    tests/test_session_fingerprint_heuristics_seam_contracts.py, which froze them at six): ten now, the wipe's
    emptying four of them.
  * WI14 -- A STORE THE WIPE REMOVES FOR THE USER THAT CANNOT BE REMOVED IS NAMED, in either mode, and stops none of
    the others: the facts, their log, their vectors, the pending writes, the plugin settings and reviews, the
    settings, the key salts, the knowledge base's collections -- the store failing to open as to delete; the
    answer's schema carries the pending writes.
  * WI15 -- THE WIPE MAKES NO KNOWLEDGE BASE: an install that has none keeps none, in either mode; one that has one
    is asked through the store there is; one that cannot be opened is named, and fails no export.
  * WI16 -- EVERY FAILURE THE WIPE NAMES IS A STORE OF THE INVENTORY: each label it gives a store it could not
    remove or empty is one of an entry's.

Local-only (the public distribution ships no tests). Every store real, loaded from a copy of its source in tmp, so
the paths it draws from its own file land there, in isolation windows, on connections that do not overwrite deleted
pages by default, as the cipher's driver does not (sqlcipher3: secure_delete 0); the file's bytes read after each
erasure.
"""

import ast
import os
import re
import sqlite3
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_census import modules as package_modules  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402

PACKAGE = REPO / "opti_oignon"
BUDGET_S = {
    "test_wi1_every_database_name_the_package_spells_is_classified": 5.0,
    "test_wi2_every_name_joined_to_a_data_folder_is_classified": 5.0,
    "test_wi3_the_inventorys_words_are_closed": 3.0,
    "test_wi4_the_wipes_lists_are_the_inventorys": 3.0,
    "test_wi5_the_coding_agents_decision_preferences_are_emptied_overwritten": 5.0,
    "test_wi6_the_humanizers_feedback_is_emptied_overwritten": 5.0,
    "test_wi7_the_learned_router_forgets_every_query_and_the_model_learned_from_them": 5.0,
    "test_wi8_the_command_lines_history_is_removed_each_file_overwritten_first": 5.0,
    "test_wi9_the_feedback_store_is_emptied_whole": 5.0,
    "test_wi10_the_knowledge_bases_search_log_is_emptied_and_its_collections_stay": 5.0,
    "test_wi11_the_single_user_wipe_empties_the_copies_through_their_real_modules": 10.0,
    "test_wi12_the_chats_knowledge_base_search_logs_no_citation": 3.0,
    "test_wi13_every_storage_statement_of_the_decision_preferences_is_a_constant_and_one_is_bound": 3.0,
    "test_wi14_a_store_the_wipe_removes_for_the_user_that_cannot_be_removed_is_named": 5.0,
    "test_wi15_the_wipe_makes_no_knowledge_base": 5.0,
    "test_wi16_every_failure_the_wipe_names_is_a_store_of_the_inventory": 3.0,
}
# The stores of the first wave, loaded from copies of their sources.
_WAVE = ("session_fingerprint.py", "humanizer.py", "learned_router.py", "history.py", "feedback.py", "rag_store.py")
_COPIES_LABELS = (
    "the coding agent's decision preferences (not user-scoped)",
    "humanizer feedback store",
    "learned routing store (the queries, word for word, and the model learned from them)",
    "the command line's history (questions and answers, not user-scoped)",
    "feedback store (unscoped, FBK-01)",
    "the knowledge base's search log (the queries, not user-scoped)",
)


def _plain(path, *, check_same_thread=False, **_options):
    """A plain connection that, like many builds, does not overwrite deleted pages unless asked to."""
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread, timeout=0.3)
    conn.execute("PRAGMA secure_delete = OFF")
    return conn


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _bytes(path):
    path = Path(path)
    return b"".join(p.read_bytes() for p in path.parent.glob(path.name + "*") if p.is_file())


def _inventory():
    """The inventory, run from its source: data only, nothing of the package imported."""
    path = source("store_inventory.py")
    module = types.ModuleType("_wi_store_inventory")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    try:
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    return module


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------
_NAME = re.compile(r"[\w./{}~-]*\.(?:db|sqlite3?)")
_QUOTED = re.compile(r"[\"'`]([\w./{}~-]*\.(?:db|sqlite3?))[\"'`]")
_YAML_NAME = re.compile(r"[\w./{}~-]*\.(?:db|sqlite3?)\b")
_DATA_FOLDER = re.compile(r"(?i)data_dir|data_root|[\"']data[\"']\)?$")
# The install's key file is named by the loader, the mode and the old recovery codes only, which ky7 of
# tests/test_signing_key_contracts.py holds: the inventory does not spell it, so its joins are left to that census.
_KEY_FILE = ".keyfile"


def _sources():
    """Each module of the package but the inventory itself: ``{relative path: tree}``."""
    trees = {}
    for path in sorted({Path(p) for p in package_modules().values()}):
        relative = path.relative_to(PACKAGE).as_posix()
        if relative != "store_inventory.py":
            trees[relative] = ast.parse(path.read_text(encoding="utf-8"))
    return trees


def _strings(tree):
    """Each string of a module, with its line: the literal parts of its f-strings included."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value, getattr(node, "lineno", 0)


def _yaml_files():
    """Each YAML file of the package; no data folder is walked."""
    for folder, dirs, files in os.walk(PACKAGE):
        dirs[:] = sorted(d for d in dirs if d not in ("data", "__pycache__", "node_modules"))
        for name in sorted(files):
            if name.endswith((".yaml", ".yml")):
                yield Path(folder, name)


def _spelled(trees):
    """Each database name the package spells, with the module or YAML file that spells it: ``{(name, where): [line]}``."""
    found = {}
    for relative, tree in trees.items():
        for value, line in _strings(tree):
            text = value.strip()
            names = {text} if _NAME.fullmatch(text) else set(_QUOTED.findall(value))
            for name in names:
                found.setdefault((name, relative), []).append(line)
    for path in _yaml_files():
        relative = path.relative_to(PACKAGE).as_posix()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            content = re.split(r"(?:^|\s)#", line, maxsplit=1)[0]
            for name in _YAML_NAME.findall(content):
                found.setdefault((name, relative), []).append(number)
    return found


class _Bindings:
    """What each module binds at its top level -- a string, or an expression -- and what it imports from the package,
    so a name joined to a data folder is read through the constants that spell it, in its module or another."""

    def __init__(self, trees):
        self.trees = trees
        self.bound = {module: {} for module in trees}
        self.imported = {module: {} for module in trees}
        for module, tree in trees.items():
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    self.bound[module][node.targets[0].id] = node.value
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                    self.bound[module][node.target.id] = node.value
                elif isinstance(node, ast.ImportFrom):
                    origin = self._origin(module, node)
                    if origin is not None:
                        for alias in node.names:
                            self.imported[module][alias.asname or alias.name] = (origin, alias.name)

    def _origin(self, module, node):
        """The package module a ``from`` import reads, relative to the package; None when it is not the package's."""
        if node.level:
            parts = module.split("/")[:-1]
            parts = parts[:len(parts) - (node.level - 1)] if node.level > 1 else parts
            name = "/".join([*parts, *(node.module.split(".") if node.module else [])])
        elif node.module and node.module.startswith("opti_oignon."):
            name = node.module[len("opti_oignon."):].replace(".", "/")
        elif node.module == "opti_oignon":
            name = ""
        else:
            return None
        for candidate in (f"{name}.py", f"{name}/__init__.py" if name else "__init__.py"):
            if candidate in self.trees:
                return candidate
        return None

    def _binding(self, name, module, depth):
        """The expression ``name`` is bound to, with its module: in the module itself, or where it was imported from."""
        if depth > 6:
            return None
        if name in self.bound.get(module, {}):
            return self.bound[module][name], module
        if name in self.imported.get(module, {}):
            origin, original = self.imported[module][name]
            return self._binding(original, origin, depth + 1)
        return None

    def folder(self, node, module, depth=0):
        """Whether ``node`` is a data folder: spelled like one, bound to one, or a folder inside one."""
        if depth > 6:
            return False
        if _DATA_FOLDER.search(ast.unparse(node)):
            return True
        if isinstance(node, ast.Name):
            found = self._binding(node.id, module, depth)
            return found is not None and self.folder(found[0], found[1], depth + 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return self.folder(node.left, module, depth + 1)
        if isinstance(node, ast.Call):
            callee = ast.unparse(node.func)
            if callee in _PATH_CALLS and node.args:
                return self.folder(node.args[0], module, depth + 1)
            if callee in ("os.environ.get", "os.getenv") and len(node.args) >= 2:
                # A folder a setting may move: its default is where it lies.
                return self.folder(node.args[1], module, depth + 1)
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("joinpath", "resolve", "expanduser"):
                return self.folder(node.func.value, module, depth + 1)
        return False

    def text(self, node, module, depth=0):
        """The path ``node`` spells, when it can be read: a constant, a name bound to one, a sum of them; an
        f-string's literal head, ``*`` standing for what it fills in; None when it cannot be read."""
        if depth > 6:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            found = self._binding(node.id, module, depth)
            return None if found is None else self.text(found[0], found[1], depth + 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self.text(node.left, module, depth + 1), self.text(node.right, module, depth + 1)
            return None if left is None or right is None else left + right
        if isinstance(node, ast.JoinedStr):
            head = ""
            for value in node.values:
                if isinstance(value, ast.Constant):
                    head += str(value.value)
                else:
                    return head + "*"
            return head
        return None


_PATH_CALLS = ("Path", "PurePath", "pathlib.Path", "os.path.join", "os.path.abspath", "os.path.realpath",
               "os.path.normpath")


def _named(spelled):
    """The name a join gives, from the path spelled after the data folder: ``*`` when it cannot be read."""
    if spelled is None:
        return "*"
    spelled = spelled.lstrip("/")
    while spelled.startswith("./"):
        spelled = spelled[2:]
    return spelled or "*"


def _joined(trees):
    """Each name the code joins to a data folder, with the module: ``{(name, module): [line]}``. A join is
    ``folder / name``, ``folder.joinpath(name, ...)``, ``Path(folder, name, ...)`` or ``os.path.join(folder, name,
    ...)`` -- the data folder also spelled as a ``"data"`` part anywhere among them -- or ``f"{folder}/name"``; a
    folder is one spelled like one, a name bound to one in its module or another, or a folder inside one; a name is
    read through the constants that spell it, an f-string's literal head standing for it (``head*``), ``*`` for one
    that cannot be read."""
    bindings = _Bindings(trees)
    found = {}
    for relative, tree in trees.items():
        for node in ast.walk(tree):
            spelled = missing = object()
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and bindings.folder(node.left, relative):
                spelled = bindings.text(node.right, relative)
            elif isinstance(node, ast.Call):
                callee = ast.unparse(node.func)
                args = node.args
                if (isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath" and args
                        and bindings.folder(node.func.value, relative)):
                    spelled = bindings.text(args[0], relative)
                elif callee in ("os.path.join", "Path", "PurePath", "pathlib.Path") and len(args) >= 2:
                    if bindings.folder(args[0], relative):
                        spelled = bindings.text(args[1], relative)
                    else:
                        for at in range(len(args) - 1):
                            if bindings.text(args[at], relative) == "data":
                                spelled = bindings.text(args[at + 1], relative)
                                break
            elif (isinstance(node, ast.JoinedStr) and len(node.values) >= 2
                  and isinstance(node.values[0], ast.FormattedValue)
                  and bindings.folder(node.values[0].value, relative)):
                rest = ast.JoinedStr(values=node.values[1:])
                head = bindings.text(rest, relative)
                if head is not None and head.startswith("/"):
                    spelled = head
            if spelled is missing:
                continue
            name = _named(spelled)
            if name != "data":
                found.setdefault((name, relative), []).append(node.lineno)
    return found


def _allowed(inventory):
    """Each classified name, with the modules allowed to spell it: those of its stores, or of its not-store entry."""
    allowed = {}
    for store in inventory.STORES:
        for name in store.names:
            allowed.setdefault(name, set()).update(store.modules)
    for name, (_why, modules) in inventory.NOT_STORES.items():
        allowed.setdefault(name, set()).update(modules)
    return allowed


def _unclassified(found, allowed):
    """The spellings no entry accounts for: a module not among its name's, a YAML file naming a name never
    classified."""
    return {f"{name} @ {where}": lines for (name, where), lines in found.items()
            if (where.endswith(".py") and where not in allowed.get(name, set()))
            or (not where.endswith(".py") and name not in allowed)}


# ---------------------------------------------------------------------------
# WI1 -- every database name the package spells is classified
# ---------------------------------------------------------------------------
def test_wi1_every_database_name_the_package_spells_is_classified():
    inventory = _inventory()
    trees = _sources()
    spelled = _spelled(trees)
    names = {name for name, _where in spelled}
    # The census finds what it must, or it proves nothing: a name in a module, in a tuple of suffixes, in a YAML file.
    assert {"conversations.db", "humanizer_feedback.db", ".sowing.glass.db"} <= names, sorted(names)
    assert ("fingerprint.db", "config/fingerprint.yaml") in spelled, "the census reads no YAML file"
    unclassified = _unclassified(spelled, _allowed(inventory))
    assert unclassified == {}, f"database names, or the modules spelling them, the inventory does not classify: " \
                               f"{unclassified}"
    # A module spells a name as a string of its own, or quoted inside one, as the census reads it.
    strings = {relative: {value for value, _line in _strings(tree)} for relative, tree in trees.items()}
    quoted = {relative: {name for value in found for name in _QUOTED.findall(value)}
              for relative, found in strings.items()}
    # A name read only as a join -- ``*``, or an f-string's ``head*`` -- is still made by a join of its modules.
    joined = set(_joined(trees))
    stale = {}
    for store in inventory.STORES:
        for name in store.names:
            if "*" in name:
                spelled = any((name, module) in joined for module in store.modules)
            else:
                spelled = any(name in strings.get(module, set()) | quoted.get(module, set())
                              for module in store.modules)
            if not spelled:
                stale.setdefault(store.label, []).append(name)
    for name, (_why, modules) in inventory.NOT_STORES.items():
        if not any(name in strings.get(module, set()) | quoted.get(module, set()) for module in modules):
            stale.setdefault("names that are no store", []).append(name)
    assert stale == {}, f"names no module of their entry spells any more: {stale}"


# ---------------------------------------------------------------------------
# WI2 -- every name joined to a data folder is classified
# ---------------------------------------------------------------------------
def test_wi2_every_name_joined_to_a_data_folder_is_classified():
    inventory = _inventory()
    joined = _joined(_sources())
    names = {name for name, _module in joined}
    # The census finds what it must: a join to a folder constant, to a data folder spelled out, through os.path.
    assert {"history", "skills", "plugin_logs", "learned_router.pkl"} <= names, sorted(names)
    # And each other spelling of a join, planted: through a constant of the module or of another, a folder inside a
    # data folder, a "data" part anywhere, an f-string's literal head, a name that cannot be read.
    planted = {
        "planted.py": ast.parse(
            'import os\nfrom pathlib import Path\nfrom .places import BASE, LEAF\nSUB = "sub"\n'
            'CACHE = DATA_DIR / "cache"\nA = DATA_DIR.joinpath("alpha.db")\nB = Path(DATA_DIR, "beta")\n'
            'C = f"{DATA_DIR}/gamma/x.db"\nD = os.path.join(_DATA_DIR, "delta")\n'
            'E = Path(__file__).parent / "data" / "epsilon"\nF = os.path.join(ROOT, "data", "zeta")\n'
            'G = DATA_DIR / f"eta_{kind}.json"\nH = f"{DATA_DIR}/{sub}/theta.json"\nI = DATA_DIR.joinpath(SUB, "iota")\n'
            'J = Path(ROOT, "data", "kappa")\nK = CACHE / "lambda.json"\nL = BASE / LEAF\nM = DATA_DIR / name\n'),
        "places.py": ast.parse('BASE = DATA_DIR / "base"\nLEAF = "mu.json"\n'),
    }
    assert {name for name, module in _joined(planted) if module == "planted.py"} == {
        "cache", "alpha.db", "beta", "gamma/x.db", "delta", "epsilon", "zeta", "eta_*", "*", "sub", "kappa",
        "lambda.json", "mu.json"}, f"a spelling of a join the census does not read: {sorted(_joined(planted))}"
    joined = {(name, module): lines for (name, module), lines in joined.items() if name != _KEY_FILE}
    unclassified = _unclassified(joined, _allowed(inventory))
    assert unclassified == {}, f"names joined to a data folder, or the modules joining them, the inventory does not " \
                               f"classify: {unclassified}"


# ---------------------------------------------------------------------------
# WI3 -- the inventory's words are closed
# ---------------------------------------------------------------------------
def test_wi3_the_inventorys_words_are_closed():
    inventory = _inventory()
    for store in inventory.STORES:
        assert store.holds in inventory.HOLDS, f"{store.label}: holds {store.holds!r}"
        assert store.wipe in inventory.WIPES, f"{store.label}: the wipe {store.wipe!r}"
        assert store.label, f"a store with no label: {store}"
        if store.holds in (inventory.WORDS, inventory.DRAWN):
            assert store.wipe != inventory.NOT_THE_USERS, f"{store.label}: a store of words called not the user's"
        for module in store.modules:
            assert (PACKAGE / module).is_file(), f"{store.label}: no module {module}"
    labels = [store.label for store in inventory.STORES]
    assert len(labels) == len(set(labels)), f"a label twice: {sorted(l for l in labels if labels.count(l) > 1)}"
    stores = {name for store in inventory.STORES for name in store.names}
    assert sorted(stores & set(inventory.NOT_STORES)) == [], (
        f"names called a store and not one: {sorted(stores & set(inventory.NOT_STORES))}")
    for name, (why, modules) in inventory.NOT_STORES.items():
        assert why and modules, f"{name}: a name called no store without why, or without the modules spelling it"
        for module in modules:
            assert (PACKAGE / module).is_file(), f"{name}: no module {module}"
    assert _KEY_FILE not in stores, "the inventory spells the key file, which ky7 keeps to three modules"


# ---------------------------------------------------------------------------
# WI4 -- the wipe's lists are the inventory's
# ---------------------------------------------------------------------------
def _lists():
    """The wipe's three lists, read from its source."""
    found = {}
    for node in ast.parse(source("user_data_manager.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("WIPE_NOT_COVERED", "WIPE_SINGLE_USER", "WIPE_RETAINED_BY_DESIGN"):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    return found


def test_wi4_the_wipes_lists_are_the_inventorys():
    inventory = _inventory()
    lists = _lists()

    def of(*wipes):
        return tuple(store.label for store in inventory.STORES if store.wipe in wipes)

    assert lists["WIPE_NOT_COVERED"] == of(inventory.ONE_USER, inventory.NOT_COVERED), (
        "the stores a wipe leaves are not the inventory's: "
        f"missing {sorted(set(of(inventory.ONE_USER, inventory.NOT_COVERED)) - set(lists['WIPE_NOT_COVERED']))}, "
        f"extra {sorted(set(lists['WIPE_NOT_COVERED']) - set(of(inventory.ONE_USER, inventory.NOT_COVERED)))}")
    assert lists["WIPE_SINGLE_USER"] == of(inventory.ONE_USER), "the stores removed for the one user are not the inventory's"
    assert lists["WIPE_RETAINED_BY_DESIGN"] == of(inventory.KEPT), "the stores kept by design are not the inventory's"
    assert "Completed against the full at-rest inventory" not in source("user_data_manager.py").read_text(
        encoding="utf-8"), "the wipe still claims an inventory no census held"


# ---------------------------------------------------------------------------
# The first wave: each store loaded from a copy of its source in tmp
# ---------------------------------------------------------------------------
def _copies(tmp_path, *relatives):
    """Each source copied under ``tmp_path/pkg/opti_oignon``, so the paths it draws from its own file land in tmp."""
    root = tmp_path / "pkg" / "opti_oignon"
    targets = {}
    for relative in relatives:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source(*relative.split("/")).read_bytes())
        targets["opti_oignon." + relative[:-3].replace("/", ".")] = target
    return targets


def _wave(tmp_path, *, wipe=False, single=True, auth=True, knowledge=True, stores=None):
    """The six stores of the wave over ``tmp_path``; with ``wipe``, the account wipe, the auth manager in single-user
    mode or not (``auth=False``: not readable), one account, ``u-1``; ``stores`` seeds stand-ins; with ``knowledge``,
    a stand-in for the knowledge base's store, which would build its vector layer."""
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    seeded = {
        "opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain),
        "opti_oignon.config": _module("opti_oignon.config", DATA_DIR=data),
    }
    targets = _copies(tmp_path, *_WAVE)
    if wipe:
        seeded["opti_oignon.run_journal"] = _module("opti_oignon.run_journal", wipe=lambda: {"wiped": True})
        if auth:
            seeded["opti_oignon.auth"] = _module("opti_oignon.auth", auth_manager=SimpleNamespace(
                single_user_mode=single, count_users=lambda: 1,
                get_user=lambda uid: SimpleNamespace(id=uid) if uid == "u-1" else None))
        targets["opti_oignon.user_data_manager"] = source("user_data_manager.py")
    seeded.update(stores or {})
    loaded, restore = isolate(targets=targets, seeded=seeded)
    if knowledge:
        # The knowledge base itself, which the wipe lists for collections of a user, stands in: its store would build
        # its vector layer. Its search log is the real file.
        loaded["opti_oignon.rag_store"]._store_instance = SimpleNamespace(
            data_dir=data / "rag", list_collections=lambda: [], delete_collection=lambda name: None)
    # The package's data folder, which a tree always has: the decision preferences make no folder of their own.
    pkg_data = tmp_path / "pkg" / "opti_oignon" / "data"
    pkg_data.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(loaded=loaded, restore=restore, data=data, pkg_data=pkg_data,
                           root_data=tmp_path / "pkg" / "data", mod=lambda name: loaded[f"opti_oignon.{name}"])


def _kept_open(path):
    """A connection that has read ``path`` and stays open, as a server's would: the last connection's close no longer
    removes the write-ahead log, so a store that does not empty it leaves its older frames."""
    conn = sqlite3.connect(str(path), timeout=0.3)
    conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    return conn


def _locked(path):
    """A connection holding ``path`` exclusively: a store asked to empty it cannot."""
    conn = sqlite3.connect(str(path), timeout=0.3)
    conn.execute("BEGIN EXCLUSIVE")
    return conn


# ---------------------------------------------------------------------------
# WI5 -- the coding agent's decision preferences
# ---------------------------------------------------------------------------
def _decisions(world, *contexts):
    fp = world.mod("session_fingerprint")
    path = Path(fp._anchor(fp.fingerprint_config.sqlite_path))
    store = fp.UserPreferencesStore(db_path=str(path))
    for number, context in enumerate(contexts):
        store.record(("approve", "abort")[number % 2], "plan", context)
    if contexts:
        conn = _plain(path)
        conn.execute("INSERT OR REPLACE INTO preference_summary (key, value) VALUES ('last', ?)", (contexts[-1],))
        conn.commit()
        conn.close()
    return path, store


def test_wi5_the_coding_agents_decision_preferences_are_emptied_overwritten(tmp_path):
    world = _wave(tmp_path)
    try:
        fp = world.mod("session_fingerprint")
        path = Path(fp._anchor(fp.fingerprint_config.sqlite_path))
        assert path.parent == world.pkg_data, f"the store is not the copy's own: {path}"
        assert fp.empty_preferences() == 0 and not path.exists(), "a store not there was created to be emptied"
        path, store = _decisions(world, "Zanzibar billing refactor, approved", "Lamu migration, aborted")
        assert b"Zanzibar" in _bytes(path), "the decisions were not recorded: the contract proves nothing"
        assert fp.empty_preferences() == 2, "not every decision was deleted"
        held = _bytes(path)
        for word in (b"Zanzibar", b"Lamu"):
            assert word not in held, f"{word!r} stayed in the decision preferences' file"
        assert store.total_decisions == 0 and store.get_ratios() == {"approve": 0.0, "modify": 0.0, "abort": 0.0}
        _decisions(world, "Pemba decision")
        holder = _locked(path)
        try:
            with pytest.raises(sqlite3.OperationalError):
                fp.empty_preferences()
        finally:
            holder.rollback()
            holder.close()
        assert b"Pemba" in _bytes(path), "a store that could not be emptied lost its rows: the lock proved nothing"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# WI6 -- the humanizer's feedback
# ---------------------------------------------------------------------------
def _compared(engine, number, original, humanized):
    db = engine._feedback_db
    assert db.store_comparison(f"cmp-{number}", original, humanized, ["contractions"], None, "light", "rewrite")
    assert db.store_rating(f"cmp-{number}", "humanized")
    return Path(db._db_path)


def test_wi6_the_humanizers_feedback_is_emptied_overwritten(tmp_path):
    world = _wave(tmp_path)
    try:
        hm = world.mod("humanizer")
        engine = hm.humanizer_engine
        assert engine is not None, "the engine did not start: the contract proves nothing"
        path = Path(engine._feedback_db._db_path)
        assert path.parent == world.root_data, f"the store is not the copy's own: {path}"
        assert hm.empty_feedback() == 0 and not path.exists(), "a store not there was created to be emptied"
        _compared(engine, 1, "Zanzibar original sentence", "Lamu humanized sentence")
        _compared(engine, 2, "Pemba original", "Kilwa humanized")
        assert hm.empty_feedback() == 2, "not every comparison was deleted"
        held = _bytes(path)
        for word in (b"Zanzibar", b"Lamu", b"Pemba", b"Kilwa"):
            assert word not in held, f"{word!r} stayed in the humanizer's feedback"
        assert engine.get_stats().total_ratings == 0
        # An engine that did not start: the store its configuration names.
        _compared(engine, 3, "Mafia original", "Mafia humanized")
        hm.humanizer_engine = None
        assert hm.empty_feedback() == 1 and b"Mafia" not in _bytes(path), (
            "with no engine, the configured store was not emptied")
        # Whatever name the configuration gives it.
        config = Path(hm._DEFAULT_CONFIG_PATH)
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("feedback_db: humanizer_ratings.db\n", encoding="utf-8")
        renamed = SimpleNamespace(_feedback_db=hm.HumanizerFeedbackDB(world.root_data / "humanizer_ratings.db"))
        _compared(renamed, 4, "Chumbe original", "Chumbe humanized")
        assert hm.empty_feedback() == 1 and b"Chumbe" not in _bytes(world.root_data / "humanizer_ratings.db"), (
            "with no engine, the store the configuration names was not the one emptied")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# WI7 -- the learned router
# ---------------------------------------------------------------------------
def _routed(router, *queries):
    for query in queries:
        router.log_sample(query, "debug")
        router.log_routing_decision(query_text=query, ml_task_type="debug", ml_confidence=0.9,
                                    yaml_task_type="general", routing_source="learned")


def test_wi7_the_learned_router_forgets_every_query_and_the_model_learned_from_them(tmp_path, monkeypatch):
    world = _wave(tmp_path)
    try:
        lr = world.mod("learned_router")
        router = lr.learned_router
        assert router is not None and Path(router._db_path).parent == world.pkg_data, "not the copy's own router"
        assert lr.empty() == 0 and not Path(router._db_path).exists(), "a store not there was created to be emptied"
        _routed(router, "Zanzibar invoice query", "Lamu stack trace query")
        model, mac = Path(router._model_path), lr._model_mac_path(router._model_path)
        # Larger than one block of the overwrite, as a real model is.
        model.write_bytes(b"vocabulary: Pemba Kilwa " * 8192)
        size = model.stat().st_size
        mac.write_bytes(b"mac of Pemba")
        router._pipeline = object()
        holder = os.open(model, os.O_RDONLY)
        # Another connection open, as a server's: the log is not removed at the last close, so it is emptied.
        keeper = _kept_open(router._db_path)
        try:
            assert lr.empty() == 4, "not every sample and decision was deleted"
            assert os.pread(holder, size, 0).strip(b"\0") == b"", "the model file was removed, not overwritten whole"
            held = _bytes(router._db_path)
            for word in (b"Zanzibar", b"Lamu"):
                assert word not in held, f"{word!r} stayed in the router's store or its log"
        finally:
            os.close(holder)
            keeper.close()
        assert not model.exists() and not mac.exists(), "the model learned from the queries stayed"
        assert router._pipeline is None and not router.is_trained, "the model in memory stayed"
        # A link in the model's place is refused, never followed: what it points at stays whole.
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"Mafia outside the store")
        model.symlink_to(outside)
        with pytest.raises(OSError):
            lr.empty()
        assert outside.read_bytes() == b"Mafia outside the store", "a link was followed and its target overwritten"
        model.unlink()
        # A training under way when the queries are forgotten never puts its model back.
        _routed(router, "Unguja query")
        trained = []

        class Pipeline:
            def fit(self, texts, labels):
                trained.append(list(texts))
                lr.empty()  # the wipe, between the samples read and the model put in place

        monkeypatch.setitem(sys.modules, "sklearn", _module("sklearn"))
        monkeypatch.setitem(sys.modules, "sklearn.model_selection", _module(
            "sklearn.model_selection", cross_val_score=lambda *_a, **_k: SimpleNamespace(mean=lambda: 0.9)))
        monkeypatch.setattr(lr, "SKLEARN_AVAILABLE", True)
        monkeypatch.setattr(lr, "joblib", _module("joblib", dump=lambda pipe, where: Path(where).write_bytes(
            b"model learned from Unguja")))
        monkeypatch.setattr(lr, "write_model_mac", lambda *_a: True)
        monkeypatch.setattr(router, "_build_pipeline", Pipeline)
        result = router.train(min_samples=1)
        assert trained and "Unguja query" in trained[0], "the training did not run: the case proves nothing"
        assert not result.success and router._pipeline is None, "a training under way put its model back in memory"
        assert not model.exists(), "a training under way wrote its model back after the queries were forgotten"
        # A forgetting right after the samples were read, before the fit: the training still drops its model.
        _routed(router, "Kojani query")
        read = router._load_training_data

        def read_then_forget():
            samples = read()
            lr.empty()
            return samples

        monkeypatch.setattr(router, "_load_training_data", read_then_forget)
        monkeypatch.setattr(router, "_build_pipeline", lambda: SimpleNamespace(fit=lambda texts, labels: None))
        result = router.train(min_samples=1)
        assert not result.success and router._pipeline is None and not model.exists(), (
            "a training that read the samples before a forgetting put its model back")
        monkeypatch.setattr(router, "_load_training_data", read)
        # A forgetting from another thread while the model is written waits for it, then removes it.
        _routed(router, "Misali query")
        forgetting = []

        def dump(pipe, where):
            forgetting.append(threading.Thread(target=lr.empty))
            forgetting[0].start()
            time.sleep(0.2)
            Path(where).write_bytes(b"model learned from Misali")

        monkeypatch.setattr(lr, "joblib", _module("joblib", dump=dump))
        result = router.train(min_samples=1)
        forgetting[0].join(timeout=5)
        assert result.success and not forgetting[0].is_alive(), "the training did not run its course: proves nothing"
        assert router._pipeline is None and not model.exists() and router.last_training_result is None, (
            "a forgetting during the model's writing left the model in memory or on disk")
        # A router that did not start: the default files.
        _routed(router, "Chumbe query")
        lr.learned_router = None
        assert lr.empty() == 2 and b"Chumbe" not in _bytes(router._db_path), "with no router, nothing was forgotten"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# WI8 -- the command line's history
# ---------------------------------------------------------------------------
def _asked(history, *questions):
    for question in questions:
        history.add(question=question, refined_question=question, response=f"answer to {question}",
                    task_type="general", model="m")


def test_wi8_the_command_lines_history_is_removed_each_file_overwritten_first(tmp_path):
    world = _wave(tmp_path)
    try:
        hi = world.mod("history")
        folder = world.data / "history"
        assert not folder.exists(), "importing the history made its folder"
        assert hi.forget_every_entry() == 0 and not folder.exists(), "a folder not there was made to be emptied"
        _asked(hi.history, "Zanzibar question at the command line", "Lamu question")
        days = sorted(folder.glob("history_*.json"))
        assert days and (folder / "index.json").exists(), "the history wrote nothing: the contract proves nothing"
        holder = os.open(days[0], os.O_RDONLY)
        try:
            assert hi.forget_every_entry() == 2, "not every file of the history was removed"
            assert os.pread(holder, 4096, 0).strip(b"\0") == b"", "an entry's file was removed, not overwritten first"
        finally:
            os.close(holder)
        assert not folder.exists(), "the emptied folder stayed"
        assert hi.history.get_recent(10) == [] and hi.history.search("Zanzibar") == []
        assert hi.history._index == {}, "the index in memory still names the removed entries"
        # A file that is not the history's stays, and the folder with it.
        _asked(hi.history, "Pemba question")
        (folder / "notes.txt").write_text("Kilwa, not the history's", encoding="utf-8")
        assert hi.forget_every_entry() == 2 and (folder / "notes.txt").exists(), "a file not the history's went"
        # One that cannot be removed raises, after the others went.
        _asked(hi.history, "Mafia question")
        outside = tmp_path / "outside.json"
        outside.write_text("[]", encoding="utf-8")
        (folder / "history_1999-01-01.json").symlink_to(outside)
        with pytest.raises(OSError):
            hi.forget_every_entry()
        assert not any(b"Mafia" in p.read_bytes() for p in folder.glob("*.json") if not p.is_symlink()), (
            "a failure kept the other files of the history")
        assert outside.read_text(encoding="utf-8") == "[]", "a link was followed and its target overwritten"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# WI9 -- the feedback store, whole
# ---------------------------------------------------------------------------
def _rated(fb, *pairs):
    for conversation, words in pairs:
        fb.feedback_store.add_feedback(fb.FeedbackEntry(conversation_id=conversation, feedback_text=words))


def test_wi9_the_feedback_store_is_emptied_whole(tmp_path):
    world = _wave(tmp_path)
    try:
        fb = world.mod("feedback")
        path = Path(fb.feedback_store._db_path)
        assert path.parent == world.pkg_data, f"the store is not the copy's own: {path}"
        assert fb.empty() == 0 and not path.exists(), "a store not there was created to be emptied"
        _rated(fb, ("c-1", "Zanzibar answer was wrong"), ("", "Lamu, given outside any conversation"))
        # Another connection open, as a server's: the log is not removed at the last close, so it is emptied.
        keeper = _kept_open(path)
        try:
            assert fb.empty() == 2, "not every entry was deleted"
            held = _bytes(path)
            for word in (b"Zanzibar", b"Lamu"):
                assert word not in held, f"{word!r} stayed in the feedback store or its log"
        finally:
            keeper.close()
        fb.feedback_store = None
        with pytest.raises(RuntimeError):
            fb.empty()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# WI10 -- the knowledge base's search log
# ---------------------------------------------------------------------------
def _searched(rs, folder, *queries):
    db = rs._RAGDatabase(folder / "rag_documents.db")
    db.create_collection("papers", "Pemba field notes collection")
    for number, query in enumerate(queries):
        db.insert_citation(rs.CitationRecord(
            citation_id=f"cit-{number}", query=query, collection_name="papers", chunk_id=f"chunk-{number}",
            parent_doc_id="doc-1", source_file="report.pdf", section="intro", score=0.5, timestamp=time.time()))
    return folder / "rag_documents.db"


def test_wi10_the_knowledge_bases_search_log_is_emptied_and_its_collections_stay(tmp_path):
    world = _wave(tmp_path)
    try:
        rs = world.mod("rag_store")
        folder = world.data / "rag"
        assert rs.forget_every_query() == 0 and not folder.exists(), "a store not there was created to be emptied"
        folder.mkdir(parents=True)
        path = _searched(rs, folder, "Zanzibar search words", "Lamu search words")
        assert rs.forget_every_query() == 2, "not every search was deleted"
        held = _bytes(path)
        for word in (b"Zanzibar", b"Lamu"):
            assert word not in held, f"{word!r} stayed in the knowledge base's search log or its log"
        assert b"Pemba field notes collection" in held, "the collections went with the searches"
        assert rs.forget_every_query(tmp_path / "elsewhere") == 0 and not (tmp_path / "elsewhere").exists()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# WI11 -- the single-user wipe, through the real modules
# ---------------------------------------------------------------------------
def _filled(world):
    """Each store of the wave given words; the files they land in."""
    _, _store = _decisions(world, "Zanzibar decision")
    _compared(world.mod("humanizer").humanizer_engine, 1, "Lamu original", "Lamu humanized")
    router = world.mod("learned_router").learned_router
    _routed(router, "Pemba query")
    Path(router._model_path).write_bytes(b"vocabulary: Pemba")
    _asked(world.mod("history").history, "Kilwa question")
    _rated(world.mod("feedback"), ("", "Mafia feedback"))
    (world.data / "rag").mkdir(parents=True, exist_ok=True)
    _searched(world.mod("rag_store"), world.data / "rag", "Chumbe search")
    fp = world.mod("session_fingerprint")
    return {
        b"Zanzibar": Path(fp._anchor(fp.fingerprint_config.sqlite_path)),
        b"Lamu": world.root_data / "humanizer_feedback.db",
        b"Pemba": Path(router._db_path),
        b"Mafia": world.pkg_data / "feedback.db",
        b"Chumbe": world.data / "rag" / "rag_documents.db",
    }


def _history_words(world):
    folder = world.data / "history"
    return b"".join(p.read_bytes() for p in folder.glob("*.json")) if folder.is_dir() else b""


def test_wi11_the_single_user_wipe_empties_the_copies_through_their_real_modules(tmp_path):
    world = _wave(tmp_path / "single", wipe=True)
    try:
        udm = world.mod("user_data_manager")
        files = _filled(world)
        answer = udm.UserDataDeleter().delete_all("local")
        for word, path in files.items():
            assert word not in _bytes(path), f"{word!r} stayed in {path.name} after the single-user wipe"
        assert b"Kilwa" not in _history_words(world), "the command line's history stayed"
        assert not Path(world.mod("learned_router").learned_router._model_path).exists(), "the router's model stayed"
        assert not set(_COPIES_LABELS) & set(answer["not_covered"]), "the answer names a store the wipe emptied"
        assert not [line for line in answer["not_covered"] if "not emptied" in line], answer["not_covered"]
        assert tuple(answer["retained_by_design"]) == udm.WIPE_RETAINED_BY_DESIGN
        # A store that is there and cannot be emptied is named.
        _decisions(world, "Unguja decision")
        holder = _locked(files[b"Zanzibar"])
        try:
            answer = udm.UserDataDeleter().delete_all("u-1")
        finally:
            holder.rollback()
            holder.close()
        assert "decision preferences (not emptied: OperationalError)" in answer["not_covered"], answer["not_covered"]
    finally:
        world.restore()
    for case, options, asked in (("multi-user", {"single": False}, "local"), ("unreadable", {"auth": False}, "local"),
                                 ("not the one user", {}, "no-such-user")):
        world = _wave(tmp_path / case, wipe=True, **options)
        try:
            udm = world.mod("user_data_manager")
            files = _filled(world)
            answer = udm.UserDataDeleter().delete_all(asked)
            for word, path in files.items():
                assert word in _bytes(path), f"{case}: {path.name} was emptied"
            assert b"Kilwa" in _history_words(world), f"{case}: the command line's history was removed"
            assert set(udm.WIPE_NOT_COVERED) <= set(answer["not_covered"]), (
                f"{case}: the answer does not name what it left: "
                f"{sorted(set(udm.WIPE_NOT_COVERED) - set(answer['not_covered']))}")
        finally:
            world.restore()


# ---------------------------------------------------------------------------
# WI12 -- the chat's knowledge base search logs no citation
# ---------------------------------------------------------------------------
def test_wi12_the_chats_knowledge_base_search_logs_no_citation():
    asked = []
    engine = SimpleNamespace(search=lambda query, **options: asked.append((query, options))
                             or SimpleNamespace(results=[]))
    seeded = {
        "opti_oignon.context_dedup": _module("opti_oignon.context_dedup", drop_duplicates=lambda items, **_k: items),
        "opti_oignon.rag_hybrid_search": _module("opti_oignon.rag_hybrid_search", get_hybrid_engine=lambda: engine),
    }
    loaded, restore = isolate(targets={"opti_oignon.unified_retrieval": source("unified_retrieval.py")},
                              seeded=seeded)
    try:
        sources = loaded["opti_oignon.unified_retrieval"]._default_sources("c-1", "p-1")
        assert sources["rag"]("Zanzibar turn words", 3) == []
    finally:
        restore()
    assert asked and asked[0][0] == "Zanzibar turn words", "the chat's retrieval did not search: proves nothing"
    assert asked[0][1].get("track_citations") is False, (
        f"the chat's search logs its query as a citation, where no deletion reaches it: {asked[0][1]}")


# ---------------------------------------------------------------------------
# WI13 -- the decision preferences' storage statements (supersedes b33)
# ---------------------------------------------------------------------------
_FINGERPRINT_SQL_SITES = 10
_FINGERPRINT_SQL_BOUND = 1


def test_wi13_every_storage_statement_of_the_decision_preferences_is_a_constant_and_one_is_bound():
    tree = ast.parse(source("session_fingerprint.py").read_text())
    sites = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr in ("execute", "executescript", "executemany")]
    assert len(sites) == _FINGERPRINT_SQL_SITES, (
        f"SQL site census moved: expected {_FINGERPRINT_SQL_SITES}, found {len(sites)}"
    )
    for node in sites:
        assert node.args, "an SQL site must receive its statement"
        first = node.args[0]
        assert isinstance(first, ast.Constant) and isinstance(
            first.value, str
        ), (
            "every SQL statement must be a constant string; a dynamically "
            "built statement reddens this census"
        )
    bound = sum(1 for node in sites if len(node.args) >= 2)
    assert bound == _FINGERPRINT_SQL_BOUND, (
        f"expected exactly {_FINGERPRINT_SQL_BOUND} parameterised site, "
        f"found {bound}"
    )


# ---------------------------------------------------------------------------
# WI14 -- a store the wipe removes for the user that cannot be removed is named
# ---------------------------------------------------------------------------
def _refusing(*_args, **_kwargs):
    raise sqlite3.OperationalError("database is locked")


def _refusing_stores(canonical, vectors, **overrides):
    """Every store the wipe removes for the user, each refusing; ``overrides`` replace a getter."""
    def getter(store):
        return lambda: store

    stores = {
        "opti_oignon.memory": _module("opti_oignon.memory", get_canonical_store=getter(canonical),
                                      get_vector_store=getter(vectors)),
        "opti_oignon.pending_writes": _module("opti_oignon.pending_writes",
                                              get_pending_store=getter(SimpleNamespace(delete_user=_refusing))),
        "opti_oignon.plugin_user_config": _module("opti_oignon.plugin_user_config", get_plugin_user_config_store=getter(
            SimpleNamespace(delete_all_configs=_refusing))),
        "opti_oignon.plugin_reviews": _module("opti_oignon.plugin_reviews",
                                              plugin_review_store=SimpleNamespace(delete_reviews_for_user=_refusing)),
        "opti_oignon.user_isolation": _module("opti_oignon.user_isolation",
                                              user_settings_store=SimpleNamespace(delete_settings=_refusing)),
        "opti_oignon.user_key_manager": _module("opti_oignon.user_key_manager", get_user_key_manager=getter(
            SimpleNamespace(delete_user_keys=_refusing))),
    }
    for name, attributes in overrides.items():
        for key, value in attributes.items():
            setattr(stores[name], key, value)
    return stores


def _wiped(tmp_path, single, stores):
    """The answer of a wipe asked for ``u-1``, the install's one account, with ``stores``; the world after it."""
    world = _wave(tmp_path, wipe=True, single=single, stores=stores)
    rs = world.mod("rag_store")
    rs._store_instance = SimpleNamespace(data_dir=world.data / "rag", list_collections=_refusing,
                                         delete_collection=_refusing)
    return world, world.mod("user_data_manager").UserDataDeleter().delete_all("u-1")


def test_wi14_a_store_the_wipe_removes_for_the_user_that_cannot_be_removed_is_named(tmp_path):
    for mode, single in (("multi-user", False), ("single-user", True)):
        cleared = []
        vectors = SimpleNamespace(clear=lambda **kwargs: cleared.append(kwargs))
        world, answer = _wiped(tmp_path / mode / "refused", single, _refusing_stores(
            SimpleNamespace(clear=_refusing), vectors))
        world.restore()
        for label in ("facts", "pending writes", "plugin settings", "plugin reviews", "settings",
                      "encryption key salts", "knowledge base collections named for the user"):
            assert f"{label} (not deleted: OperationalError)" in answer["not_covered"], (
                f"{mode}: {label} could not be removed and the answer does not say so: {answer['not_covered']}")
        assert answer["pending_writes"] == 0 and answer["memories"] == 0 and cleared, (
            f"{mode}: the facts' vectors were not cleared when the facts could not be")
        # A store that cannot be opened at all: named the same, and the steps after it still run.
        cleared = []
        opening = {"opti_oignon.memory": {"get_canonical_store": _refusing},
                   "opti_oignon.pending_writes": {"get_pending_store": _refusing},
                   "opti_oignon.plugin_user_config": {"get_plugin_user_config_store": _refusing},
                   "opti_oignon.user_key_manager": {"get_user_key_manager": _refusing}}
        stores = _refusing_stores(None, SimpleNamespace(clear=lambda **kwargs: cleared.append(kwargs)), **opening)
        # The conversation store failing at its import, as a store that cannot be opened does: past its step's own
        # handling, caught by the wipe's.
        conversation = _module("opti_oignon.conversation")
        conversation.__getattr__ = _refusing
        stores["opti_oignon.conversation"] = conversation
        world, answer = _wiped(tmp_path / mode / "unopened", single, stores)
        try:
            for label in ("conversations", "facts", "pending writes", "plugin settings", "encryption key salts"):
                assert f"{label} (not deleted: OperationalError)" in answer["not_covered"], (
                    f"{mode}: a store that could not be opened is named nowhere: {label}: {answer['not_covered']}")
            assert cleared, f"{mode}: the facts' vectors were not cleared when their store could not be opened"
            # The steps after them ran: the last ones of each mode said their part.
            assert "settings (not deleted: OperationalError)" in answer["not_covered"], (
                f"{mode}: a store that could not be opened stopped the steps after it")
            if single:
                assert "sandbox audit's words (this process did not load the sandbox)" in answer["not_covered"], (
                    "a store that could not be opened stopped the single-user steps")
        finally:
            world.restore()
        # The facts deleted, their log not emptied: each said for what it is, the vectors cleared, the count kept.
        cleared = []
        canonical = SimpleNamespace(clear=lambda **_kwargs: 3, empty_log=_refusing)
        world, answer = _wiped(tmp_path / mode / "partial", single, _refusing_stores(
            canonical, SimpleNamespace(clear=lambda **kwargs: cleared.append(kwargs))))
        world.restore()
        assert answer["memories"] == 3, f"{mode}: the facts removed were not counted: {answer['memories']}"
        assert "facts' log (not emptied: OperationalError)" in answer["not_covered"], answer["not_covered"]
        assert "facts (not deleted: OperationalError)" not in answer["not_covered"], (
            f"{mode}: facts that were removed are said to stay")
        assert cleared, f"{mode}: the facts' vectors were skipped after the log failed"
    schema = ast.parse(source("api", "routes_users.py").read_text(encoding="utf-8"))
    fields = {node.target.id for cls in schema.body if isinstance(cls, ast.ClassDef) and cls.name == "DeleteDataResponse"
              for node in cls.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    assert "pending_writes" in fields and "not_covered" in fields, f"the answer's schema drops a count: {sorted(fields)}"


# ---------------------------------------------------------------------------
# WI15 -- the wipe makes no knowledge base
# ---------------------------------------------------------------------------
def test_wi15_the_wipe_makes_no_knowledge_base(tmp_path):
    for case, single in (("single-user", True), ("multi-user", False)):
        world = _wave(tmp_path / case, wipe=True, single=single, knowledge=False)
        try:
            answer = world.mod("user_data_manager").UserDataDeleter().delete_all("local")
            assert answer["rag_collections"] == 0
            assert not (world.data / "rag").exists(), f"{case}: the wipe made the knowledge base it asked"
            rs = world.mod("rag_store")
            assert rs.existing_rag_store() is None and rs._store_instance is None, (
                f"{case}: a knowledge base that is not there was built to be asked")
        finally:
            world.restore()
    # One that is there is asked through the store there is.
    world = _wave(tmp_path / "built", wipe=True, knowledge=True)
    try:
        rs = world.mod("rag_store")
        assert rs.existing_rag_store() is rs._store_instance, "the store this process built was not the one asked"
    finally:
        world.restore()
    # One whose file is there and that cannot be opened: named by the wipe, and the export still answers.
    world = _wave(tmp_path / "unopened", wipe=True, knowledge=False)
    try:
        rs = world.mod("rag_store")
        (world.data / "rag").mkdir(parents=True)
        _searched(rs, world.data / "rag", "Unguja search")
        rs.get_rag_store = _refusing
        udm = world.mod("user_data_manager")
        answer = udm.UserDataDeleter().delete_all("u-1")
        assert "knowledge base collections named for the user (not deleted: OperationalError)" in answer[
            "not_covered"], answer["not_covered"]
        exported = udm.UserDataExporter().export("u-1")
        assert exported["rag_collections"] == [] and "settings" in exported, "a knowledge base that could not be " \
                                                                            "opened failed the whole export"
    finally:
        world.restore()




# ---------------------------------------------------------------------------
# WI16 -- every failure the wipe names is a store of the inventory
# ---------------------------------------------------------------------------
def _failure_labels():
    """Each label the wipe gives a store it could not remove or empty, read from its source."""
    tree = ast.parse(source("user_data_manager.py").read_text(encoding="utf-8"))
    labels = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("_failed", "_guarded")
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)):
            labels.add(node.args[1].value)
        elif isinstance(node, ast.FunctionDef) and node.name in ("_empty_requests", "_empty_copies"):
            for loop in ast.walk(node):
                if isinstance(loop, ast.For) and isinstance(loop.iter, ast.Tuple):
                    labels.update(item.elts[0].value for item in loop.iter.elts if isinstance(item, ast.Tuple))
        elif (isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant)
              and " (" in str(node.values[0].value)):
            labels.add(str(node.values[0].value).split(" (", 1)[0])
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and re.fullmatch(r"[a-z][\w' ]+ \((?:this process|the conversation store)[^)]*\)", node.value)):
            labels.add(node.value.split(" (", 1)[0])
    return labels


def test_wi16_every_failure_the_wipe_names_is_a_store_of_the_inventory():
    inventory = _inventory()
    labels = _failure_labels()
    # The reading finds what it must: a store removed, a store emptied, a group, a sentence of its own.
    assert {"facts", "pending writes", "decision preferences", "response cache", "conversations", "run journal",
            "old memory store", "sandbox audit's words"} <= labels, sorted(labels)
    known = {name for store in inventory.STORES for name in store.failures}
    assert sorted(labels - known) == [], f"failures the wipe names that no store of the inventory is named by: " \
                                         f"{sorted(labels - known)}"
    assert sorted(known - labels) == [], f"failure labels of the inventory the wipe never gives: {sorted(known - labels)}"
