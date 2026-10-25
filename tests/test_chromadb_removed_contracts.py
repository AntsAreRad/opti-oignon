#!/usr/bin/env python3
"""Contracts for chromadb's departure: no module of the platform reads, imports or probes it.

``db_encryption migrate`` reads ChromaDB's folders with the standard library
(``chroma_migration``, block 181): nothing of the package, its scripts or its
guards imports chromadb any more, by any spelling.

  * RM2 -- NO MODULE IMPORTS CHROMADB BY ANY SPELLING: no ``import`` and no
    ``from ... import`` of chromadb or of a module below it, no call handed a
    constant that names it, by position or by name, whatever is called -- an
    import, a lookup, a probe, a read of its installation --, and no
    subscript or membership test of the modules with such a constant, in the
    package's modules, ``scripts/`` or ``.github/scripts/``. A name built at
    run time is no spelling this census can see.

Local-only (the public distribution ships no tests). It reads source files only.
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_census import modules as package_modules  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
BUDGET_S = {
    "test_rm2_no_module_imports_chromadb_by_any_spelling": 8.0,
}


def _names_chromadb(value):
    return isinstance(value, str) and (value == "chromadb" or value.startswith("chromadb."))


def _spellings(tree):
    """Each way ``tree`` reaches chromadb, unparsed: an import of it or below it; a call handed a constant that names
    it, whatever is called (an import, a lookup, a probe, a read of its installation); a subscript, or a membership
    test, of the modules by such a constant."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(_names_chromadb(alias.name) for alias in node.names):
            found.append(ast.unparse(node))
        elif isinstance(node, ast.ImportFrom) and _names_chromadb(node.module or ""):
            found.append(ast.unparse(node))
        elif isinstance(node, ast.Call) and any(
                isinstance(arg, ast.Constant) and _names_chromadb(arg.value)
                for arg in [*node.args, *(keyword.value for keyword in node.keywords)]):
            found.append(ast.unparse(node))
        elif isinstance(node, ast.Subscript) and ast.unparse(node.value).endswith("modules") and (
                isinstance(node.slice, ast.Constant) and _names_chromadb(node.slice.value)):
            found.append(ast.unparse(node))
        elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant) and _names_chromadb(
                node.left.value) and any(ast.unparse(right).endswith(("modules", "modules.keys()"))
                                         for right in node.comparators):
            found.append(ast.unparse(node))
    return found


def test_rm2_no_module_imports_chromadb_by_any_spelling():
    files = sorted({Path(path) for path in package_modules().values()}
                   | set((_REPO / "scripts").glob("*.py")) | set((_REPO / ".github" / "scripts").glob("*.py")))
    assert len(files) > 100 and _REPO / "opti_oignon" / "chroma_migration.py" in files, ("control: the tree", len(files))
    reached = {}
    for path in files:
        spellings = _spellings(ast.parse(path.read_text(encoding="utf-8")))
        if spellings:
            reached[path.relative_to(_REPO).as_posix()] = spellings
    assert reached == {}, ("chromadb reached", reached)
    # Control: every spelling the census looks for is found where it is written.
    planted = ("import chromadb\nimport chromadb.config as c\nfrom chromadb import PersistentClient\n"
               "from chromadb.config import Settings\nimport importlib\nimportlib.import_module('chromadb')\n"
               "__import__('chromadb.api')\nimportlib.util.find_spec('chromadb')\nsys.modules['chromadb']\n"
               "sys.modules.get('chromadb')\n__import__(name='chromadb')\nimportlib.import_module(name='chromadb')\n"
               "importlib.util.find_spec(name='chromadb')\n'chromadb' in sys.modules\n"
               "importlib.metadata.version('chromadb')\nversion('chromadb')\nmd.version('chromadb')\n"
               "'chromadb' in sys.modules.keys()\nsys.modules.__contains__('chromadb')\n"
               "sys.modules.setdefault('chromadb', None)\nsys.modules.pop('chromadb')\n"
               "importlib.metadata.distribution('chromadb')\n")
    assert sorted(_spellings(ast.parse(planted))) == sorted([
        "import chromadb", "import chromadb.config as c", "from chromadb import PersistentClient",
        "from chromadb.config import Settings", "importlib.import_module('chromadb')", "__import__('chromadb.api')",
        "importlib.util.find_spec('chromadb')", "sys.modules['chromadb']", "sys.modules.get('chromadb')",
        "__import__(name='chromadb')", "importlib.import_module(name='chromadb')",
        "importlib.util.find_spec(name='chromadb')", "'chromadb' in sys.modules",
        "importlib.metadata.version('chromadb')", "version('chromadb')", "md.version('chromadb')",
        "'chromadb' in sys.modules.keys()", "sys.modules.__contains__('chromadb')",
        "sys.modules.setdefault('chromadb', None)", "sys.modules.pop('chromadb')",
        "importlib.metadata.distribution('chromadb')"]), _spellings(ast.parse(planted))
    assert _spellings(ast.parse("x = 'chromadb is gone'\nimport chromadb_like\nfrom chroma import y\n")) == [], (
        "a string or another module is no spelling")
