"""Contracts of the Mycelium's network, its first kind of edge: a file that names, resolved through its imports, what
another file of the same source defines (the ``link`` table of the store's schema 4), computed again by every index
pass that changed something, shown by a file's map card both ways, and walked by the agent from the files it has
opened.

  * RS1 -- CA1 of tests/test_mycelium_cards_contracts.py, superseded (it held the schema at 3): a new store is at 4
    with its ``card`` and ``link`` tables; a store of schema 3 is read as it is -- searched, mapped, no link -- and
    opened to write moves to 4 in one transaction, the table added and nothing else touched (no part cut again), a
    failed move leaving it at 3 with no table; a store of schema 2 moves through 3 to 4; a store at 5 is refused by
    name, to read and to write.
  * RS2 -- CA6 there, superseded: a store of schema 1 opens at schema 4 and is cut again.
  * RS3 -- CA7 there, superseded: each move from schema 1 (to 2, to 3, to 4) is a transaction of its own, a failed one
    leaving the store as it was; opened again, the store is at 4.
  * RS4 -- CA8 there, superseded (its newer schema was 4): reading writes nothing, and a store at a newer schema (5),
    one of schema 1, a database of another kind and an empty file are refused by name, each left as it was.
  * RS5 -- CA9 there, superseded: an opener reads schema 1, another moves the store first: the first opens it at 4,
    its column, its card table and its link table there once.
  * RS6 -- THE EDGES: on a project written by hand -- absolute and relative imports (``from ..`` too), an alias, a
    module imported as a whole, a chain of attributes through an imported alias, a class's base through one, a
    package's ``__init__`` that defines nothing (no edge) or re-exports a name (followed to its module), a ``src/``
    layout, a file naming itself, an import of something outside the project, a call by attribute on no import -- the
    edges, each with how many parts of its file resolve to it, are exactly the ones written here by hand; a ``src/``
    that is a package keeps its name, and a ``lib/`` module named as the standard library's never takes its imports;
    an edge holds at least one part.
  * RS7 -- FORGETTING, REMOVING, RENAMING: a removed file takes its edges both ways, and what named it resolves again
    to what is left; a renamed file keeps its id and its edges are computed again (its module changed); a forgotten
    source takes all of its edges and their version (a source given its id later computes its own); no edge joins two
    sources, though a file of one imports a module of the other's.
  * RS8 -- A SECRET SOURCE: no edge is computed for it, by its pass or asked directly; a source marked secret loses
    its edges at once, a rewrite owed; marked private again, its next pass computes them though no file changed.
  * RS9 -- WHAT A PASS WRITES: a pass that changed nothing computes no edge and writes nothing to the ``link`` table; a
    new version of the resolution computes every edge again at the next pass, and one dropped there owes a rewrite.
  * RS10 -- A FAILED PASS: a pass that fails leaves the edges as they were, none of its own kept.
  * RS11 -- THE FILE'S CARD: a file's card lists the files it links to and the files that link to it, most parts
    first, then by path; ``branch`` says both lines, bounded; a store of schema 3 read as it is shows none, and the
    lists MA2 and MA3 hold (tests, docs, users by name) are unchanged.
  * RS12 -- THE WALK: from the files the agent has opened, two lists of files to open next -- linked (each edge
    counted once each way, between an opened file and another) and same names (the join by name the map's users
    use, both ways: a name with more than six definitions in the source counts for nothing, six count) --, the opened
    left out, a file the first list shows never in the second (one it leaves past its eight may be), at most eight
    each, most first then by path; a path the source does not hold is said, and how many past eight; none opened, or
    none the source holds, gives nothing; another source's files never come.
  * RS13 -- MA5 of tests/test_mycelium_map_contracts.py, superseded (it held the tool's parameters at two): the tool
    declared with ``path``, ``source`` and ``opened``, all optional, class none, a read tool in both modes, on the
    eval path neutralized like the search; the corpus card names it and says it is not something to cite;
    SECURITY.md and the CHANGELOG say it. The run's handler walks from ``opened``, refusing a value that is not a
    list of at most 25 paths, and a call that gives a path and ``opened`` both.
  * RS14 -- THE CUTTER: version 6 records, besides what it recorded, each chain of attributes read on an imported
    name (kind ``dotted``, the whole chain as written, once, the head's import), never one written, and binds the head
    of a plain ``import a.b`` to ``a``; on the package's own code every other record is the one a cutter without the
    chains makes; the readers by name (a file's card, the value head's traits) give exactly what they gave without
    those rows; a store cut by version 5 is cut again at its next pass, keeps every part's embedded text, and owes no
    rewrite.
  * RS15 -- THE TEXTS: the CHANGELOG says the network, its schema, its walk and the contracts; SECURITY.md says
    what the store keeps of it and what the map shows.
RS1-RS11 and RS14 run the real store, index, links and map modules on stores in the test's folder through a plain
SQLite opener; no engine is reached (the vector suite's guard). RS12 and RS13 run the real tool on the Mycelium agent
suite's fixtures, as MA4 and MA5 do.
"""
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_mycelium_agent_contracts as my_t  # noqa: E402
import test_mycelium_place_contracts as pl  # noqa: E402
import test_mycelium_vector_contracts as mv  # noqa: E402

from opti_oignon.mycelium import index as index_mod  # noqa: E402
from opti_oignon.mycelium import links  # noqa: E402
from opti_oignon.mycelium import map as mp  # noqa: E402
from opti_oignon.mycelium import store as store_mod  # noqa: E402
from opti_oignon.mycelium.index import Indexer  # noqa: E402
from opti_oignon.mycelium.search import Searcher  # noqa: E402
from opti_oignon.mycelium.store import Store, StoreError  # noqa: E402

BUDGET_S = {
    "test_rs1_the_schema": 3.0,
    "test_rs2_a_schema_1_store_opens_at_schema_4_and_is_cut_again": 1.5,
    "test_rs3_each_move_from_schema_1_is_whole": 1.5,
    "test_rs4_reading_writes_nothing": 3.0,
    "test_rs5_two_openers_of_a_schema_1_store_move_it_once": 1.5,
    "test_rs6_the_edges": 2.0,
    "test_rs7_forgetting_removing_renaming": 3.0,
    "test_rs8_a_secret_source": 2.0,
    "test_rs9_what_a_pass_writes": 2.0,
    "test_rs10_a_failed_pass": 2.0,
    "test_rs11_the_files_card": 2.0,
    "test_rs12_the_walk": 5.0,
    "test_rs13_the_tool_declared": 4.0,
    "test_rs14_the_cutter": 20.0,
    "test_rs15_the_texts": 1.0,
}

# The vector suite's guard: no contract reaches the machine's engine or governor.
no_engine = mv.no_engine
seams = mv.seams
place = pl.place
owned = my_t.owned

def _fn(signature, body):
    """A function of seven lines: the cutter keeps a unit of six lines or more as a part of its own (``MIN_LINES``)."""
    return f'\n\ndef {signature}:\n    """{signature}.\n\n    Seven lines, so it is\n    a part of its own.\n    """\n    {body}\n'


def _class(name, base):
    """A class of seven lines, a part of its own, on ``base`` when one is named."""
    on = f"({base})" if base else ""
    return f'\n\nclass {name}{on}:\n    """{name}.\n\n    Seven lines, so it is\n    a part of its own.\n    """\n    flag = 0\n'


# The project the edges are written for, by hand (RS6): each function a part of its own, the docstring and the imports
# above them one part.
PROJECT = {
    "app/__init__.py": "",
    "app/a.py": '"""A."""\n' + _fn("f()", "return 1"),
    "app/b.py": '"""B."""\nfrom app.a import f\n' + _fn("g()", "return f()"),
    "app/c.py": '"""C."""\nimport app.a as alias\nfrom app import b\n' + _fn("h()", "return alias.f() + b.g()"),
    "app/d.py": '"""D."""\nfrom app.a import f\n' + _fn("f2()", "return 2"),
    "app/e.py": '"""E."""\nimport app.e\n' + _fn("k()", "return app.e.k()"),
    "app/u.py": '"""U."""\n' + _fn("use(obj)", "return obj.f()"),
    "app/long.py": '"""Long."""\n' + _fn("frobnicate_widgets()", "return 0"),
    "other/x.py": '"""X."""\nimport json\n' + _fn("x()", "return json.dumps({})"),
    "src/lib2/m.py": '"""M."""\nfrom . import n\n' + _fn("mm()", "return n.nn()"),
    "src/lib2/n.py": '"""N."""\n' + _fn("nn()", "return 0"),
    "docs/guide.md": "# Guide\n\nCall `f` from `app.a`.\n",
    "app/pkg/__init__.py": '"""Pkg."""\nfrom .impl import thing\n' + _fn("pk()", "return 4"),
    "app/pkg/impl.py": '"""Impl."""\n' + _fn("thing()", "return 3") + _class("Base", ""),
    "app/v.py": ('"""V."""\nimport app.pkg as p\nfrom app.pkg import thing\n'
                 + _fn("viaduct()", "return p.impl.thing() + thing() + p.pk()") + _class("Vv", "p.impl.Base")),
    "app/deep/__init__.py": "",
    "app/deep/k.py": '"""K."""\nfrom .. import a\nfrom ..a import f as ff\n' + _fn("k1()", "return a.f() + ff()"),
    "app/s.py": '"""S."""\nimport app.pkg as pp\n' + _fn("setter()", "pp.flag = 1\n    return pp"),
    "app/r.py": '"""R."""\n' + _fn("ring(o)", "return o.use(o)"),
    "src/lib2/n2.py": '"""N2."""\nfrom lib2.n import nn\n' + _fn("nn2()", "return nn()"),
    # A name re-exported through three modules, and through four (followed three steps, then its module).
    "app/hop/__init__.py": "",
    **{f"app/hop/h{n}.py": f"from app.hop.h{n + 1} import deep\n" for n in range(4)},
    "app/hop/h4.py": '"""H4."""\n' + _fn("deep()", "return 5"),
    "app/hop/user1.py": "from app.hop.h1 import deep\n" + _fn("u1()", "return deep()"),
    "app/hop/user0.py": "from app.hop.h0 import deep\n" + _fn("u0()", "return deep()"),
    # Two modules that re-export a name from each other: the walk ends.
    "app/cyc/__init__.py": "",
    "app/cyc/ca.py": "from app.cyc.cb import loop\n",
    "app/cyc/cb.py": "from app.cyc.ca import loop\n",
    "app/cyc/cu.py": "from app.cyc.ca import loop\n",
}
# Each edge, src -> dst, with the parts of src that resolve to dst (written by hand from the project above, by the
# resolution's rule: links.py's docstring).
EDGES = {
    ("app/b.py", "app/a.py"): 2,            # its imports (from app.a import f); g calls f
    ("app/c.py", "app/a.py"): 2,            # its imports (import app.a as alias); h calls alias.f
    ("app/c.py", "app/b.py"): 2,            # its imports (from app import b: the submodule); h calls b.g -- the
    #                                         empty app/__init__.py holds nothing: no edge to it
    ("app/d.py", "app/a.py"): 1,            # its imports alone
    ("src/lib2/m.py", "src/lib2/n.py"): 2,  # its imports (from . import n, src.lib2.n); mm calls n.nn
    ("app/pkg/__init__.py", "app/pkg/impl.py"): 1,  # it imports thing from impl
    ("app/v.py", "app/pkg/impl.py"): 3,     # its imports (thing, bound in the package to impl's); viaduct
    #                                         (p.impl.thing, thing); the class Vv on p.impl.Base
    ("app/v.py", "app/pkg/__init__.py"): 2,  # its imports (import app.pkg as p: the package, which holds pk);
    #                                          viaduct calls p.pk, which the package's __init__ defines
    ("app/deep/k.py", "app/a.py"): 2,       # its imports (from .. import a, from ..a import f as ff); k1
    ("app/s.py", "app/pkg/__init__.py"): 2,  # its imports (import app.pkg as pp); setter reads pp, writing on it
    ("src/lib2/n2.py", "src/lib2/n.py"): 2,  # an absolute import from src/ (lib2.n: the layout installed); nn2
    # A name bound through four modules: three steps reach where it is defined; four stop at the third, which holds it.
    ("app/hop/h0.py", "app/hop/h4.py"): 1, ("app/hop/h1.py", "app/hop/h4.py"): 1,
    ("app/hop/h2.py", "app/hop/h4.py"): 1, ("app/hop/h3.py", "app/hop/h4.py"): 1,
    ("app/hop/user1.py", "app/hop/h4.py"): 2, ("app/hop/user0.py", "app/hop/h3.py"): 2,
    # A cycle of bindings ends: ca and cb each come back to themselves (no edge), cu stops at cb.
    ("app/cyc/cu.py", "app/cyc/cb.py"): 1,
}

# The two reviews' cases, each a layout the interpreter resolves one way (RS6's third project).
REVIEWED = {
    "mypkg/__init__.py": '__version__ = "1"\n',
    "mypkg/mypkg.py": '"""M."""\n' + _fn("main()", "return 0"),
    "mypkg/utils.py": '"""U."""\n' + _fn("u()", "return 1"),
    "mypkg/cli.py": ('"""C."""\nfrom mypkg import __version__\nimport mypkg.utils\n'
                     + _fn("cli()", "return mypkg.utils.u() + len(__version__)")),
    "src/utils.py": '"""U."""\n' + _fn("helper()", "return 1"),
    "main.py": '"""M."""\nfrom src.utils import helper\n' + _fn("m2()", "return helper()"),
    "tests/test_u.py": '"""T."""\nfrom src import utils\n' + _fn("t()", "return utils.helper()"),
    "lib/helpers.py": '"""H."""\n' + _fn("hx()", "return 2"),
    "run.py": '"""R."""\nimport lib.helpers\n' + _fn("rr()", "return lib.helpers.hx()"),
    "app2/__init__.py": "app = object()\ndb = object()\n",
    "app2/routes.py": '"""R."""\nfrom app2 import app, db\n' + _fn("r()", "return app, db"),
    "rpkg/__init__.py": "from ._impl import Core as Engine\n",
    "rpkg/_impl.py": '"""I."""\n' + _class("Core", ""),
    "ruse.py": '"""R."""\nfrom rpkg import Engine\n' + _fn("ru()", "return Engine()"),
    "shadow/__init__.py": "from .core import sub\n",
    "shadow/core.py": '"""C."""\n' + _fn("sub()", "return 1"),
    "shadow/sub.py": '"""S."""\n' + _fn("other()", "return 2"),
    "suse.py": '"""S."""\nfrom shadow import sub\n' + _fn("su()", "return sub()"),
    "dup.py": '"""D."""\n' + _fn("g()", "return 1"),
    "dup/__init__.py": '"""D."""\n' + _fn("g()", "return 2"),
    "duse.py": '"""D."""\nfrom dup import g\n' + _fn("du()", "return g()"),
    "src/pkg2/__init__.py": "from .core import run\n",
    "src/pkg2/core.py": '"""C."""\n' + _fn("run()", "return 1"),
    "tests/test_cli.py": '"""T."""\nimport pkg2\n' + _fn("tc()", "return pkg2.run()"),
    "star/__init__.py": "from .impl2 import *\n",
    "star/impl2.py": '"""I."""\n' + _fn("starred()", "return 1"),
    "stuse.py": '"""S."""\nfrom star import starred\n' + _fn("st()", "return starred()"),
    "emp/__init__.py": "",
    "emp/m.py": '"""M."""\n' + _fn("em()", "return 1"),
    "euse.py": '"""E."""\nimport emp\nimport emp.m\n' + _fn("eu()", "return emp.m.em()"),
    "simp.py": '"""S."""\nimport shadow.sub\n',
    "lazy.py": '"""L."""\n' + _fn("load()", "from mypkg import utils as late\n    import mypkg.mypkg as late2\n    return late, late2"),
    "luse.py": '"""U."""\nfrom lazy import late, late2\n' + _fn("lu()", "return late, late2"),
    "cond.py": "if True:\n    def inner():\n        from mypkg import utils as hid\n        return hid\n",
    "cuse.py": '"""C."""\nfrom cond import hid\n' + _fn("cu()", "return hid"),
    # A package that imports its own submodules (networkx, matplotlib): the binding names a module, entered.
    "selfpkg/__init__.py": "from selfpkg import core\nfrom . import extra\n",
    "selfpkg/core.py": '"""C."""\n' + _fn("cc()", "return 1"),
    "selfpkg/extra.py": '"""E."""\n' + _fn("ex()", "return 2"),
    "suser.py": '"""S."""\nfrom selfpkg.core import cc\nimport selfpkg\n' + _fn("su2()", "return cc() + selfpkg.extra.ex()"),
    # A module named as the class the package re-exports (sentence-transformers): `from m import n` imports the
    # MODULE m by its path, then reads n on it.
    "evalpkg/__init__.py": "from .SentenceEvaluator import SentenceEvaluator\n",
    "evalpkg/SentenceEvaluator.py": '"""E."""\n' + _class("SentenceEvaluator", ""),
    "evuser.py": ('"""E."""\nfrom evalpkg.SentenceEvaluator import SentenceEvaluator\n'
                  + _fn("ev()", "return SentenceEvaluator()")),
    # Imports with a star, chained: three steps reach the name, four do not; a first star that misses; a cycle.
    "st/__init__.py": "",
    "st/sx.py": "from st.st0 import *\n",
    **{f"st/st{n}.py": f"from st.st{n + 1} import *\n" for n in range(3)},
    "st/st3.py": '"""S."""\n' + _fn("zz()", "return 1"),
    "st/empty_star.py": '"""E."""\n' + _fn("other()", "return 0"),
    "stw.py": "from st.empty_star import *\nfrom st.st3 import *\n",
    "user3.py": '"""U."""\nfrom st.st1 import zz\n' + _fn("u3()", "return zz()"),
    "user4.py": '"""U."""\nfrom st.st0 import zz\n' + _fn("u4()", "return zz()"),
    "user5.py": '"""U."""\nfrom st.sx import zz\n' + _fn("u5()", "return zz()"),
    "user6.py": '"""U."""\nfrom stw import zz\n' + _fn("u6()", "return zz()"),
    "cyc2/__init__.py": "",
    "cyc2/a.py": "from cyc2.b import *\n",
    "cyc2/b.py": "from cyc2.a import *\n",
    "user7.py": '"""U."""\nfrom cyc2.a import nothing\n' + _fn("u7()", "return nothing"),
}
REVIEWED_EDGES = {
    ("mypkg/cli.py", "mypkg/__init__.py"): 2,   # __version__, assigned in the package (never mypkg/mypkg.py)
    ("mypkg/cli.py", "mypkg/utils.py"): 2,      # import mypkg.utils; mypkg.utils.u
    ("main.py", "src/utils.py"): 2,             # src/ no package: importable as src.utils (a namespace) too
    ("tests/test_u.py", "src/utils.py"): 2,     # from src import utils; utils.helper
    ("run.py", "lib/helpers.py"): 2,            # import lib.helpers; lib.helpers.hx
    ("app2/routes.py", "app2/__init__.py"): 2,  # from app2 import app, db: names the package assigns
    ("rpkg/__init__.py", "rpkg/_impl.py"): 1,   # it imports Core
    ("ruse.py", "rpkg/_impl.py"): 2,            # Engine, bound in the package to _impl's Core (an alias)
    ("shadow/__init__.py", "shadow/core.py"): 1,
    ("suse.py", "shadow/core.py"): 2,           # from shadow import sub: the package's attribute before its submodule
    ("duse.py", "dup/__init__.py"): 2,          # the package dup before the module dup.py
    ("src/pkg2/__init__.py", "src/pkg2/core.py"): 1,
    ("tests/test_cli.py", "src/pkg2/__init__.py"): 1,   # import pkg2: the package installed from src/, which holds run
    ("tests/test_cli.py", "src/pkg2/core.py"): 1,       # pkg2.run, bound in the package to core's
    ("star/__init__.py", "star/impl2.py"): 1,   # from .impl2 import *
    ("stuse.py", "star/impl2.py"): 2,           # starred, through the package's import with a star
    ("euse.py", "emp/m.py"): 2,                 # import emp.m; emp.m.em -- the empty emp/__init__.py: no edge
    ("simp.py", "shadow/sub.py"): 1,            # import shadow.sub imports the submodule, whatever the package binds
    ("lazy.py", "mypkg/utils.py"): 1,           # a function's own imports; the module binds no `late`, no `late2`:
    ("lazy.py", "mypkg/mypkg.py"): 1,           #   luse, nothing
    ("cond.py", "mypkg/utils.py"): 1,           # an import inside a function inside an `if`: cuse, nothing
    ("selfpkg/__init__.py", "selfpkg/core.py"): 1, ("selfpkg/__init__.py", "selfpkg/extra.py"): 1,
    ("suser.py", "selfpkg/core.py"): 2,         # selfpkg.core.cc, the package's own binding of core entered; cc
    ("suser.py", "selfpkg/__init__.py"): 1,     # import selfpkg: the package, which holds its bindings
    ("suser.py", "selfpkg/extra.py"): 1,        # selfpkg.extra.ex, through the package's binding of extra
    ("evalpkg/__init__.py", "evalpkg/SentenceEvaluator.py"): 1,
    ("evuser.py", "evalpkg/SentenceEvaluator.py"): 2,   # the module by its path, then its class
    ("st/sx.py", "st/st0.py"): 1, ("st/st0.py", "st/st1.py"): 1, ("st/st1.py", "st/st2.py"): 1,
    ("st/st2.py", "st/st3.py"): 1,              # each `from m import *` imports its module
    ("user3.py", "st/st3.py"): 2,               # two stars to zz
    ("user4.py", "st/st3.py"): 2,               # three stars to zz; user5's four reach nothing
    ("stw.py", "st/empty_star.py"): 1, ("stw.py", "st/st3.py"): 1,
    ("user6.py", "st/st3.py"): 2,               # the first star misses, the second holds zz
    ("cyc2/a.py", "cyc2/b.py"): 1, ("cyc2/b.py", "cyc2/a.py"): 1,   # user7: the cycle ends, nothing
}


def _write(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _plain(path):
    return Store(path, lambda p: sqlite3.connect(p, check_same_thread=False))


def _project(tmp_path, name="proj", files=None):
    root = _write(tmp_path / name, files or PROJECT)
    store = _plain(tmp_path / f"{name}.myc")
    Indexer(store).index("oo", root)
    return store, root


def _edges(store, source="oo"):
    return {(a, b): n for a, b, n in store.db.execute(
        "SELECT fa.path, fb.path, l.evidence FROM link l JOIN file fa ON fa.id = l.src JOIN file fb ON fb.id = l.dst "
        "JOIN source s ON s.id = fa.source WHERE s.name = ?", (source,))}


def _link_writes(store):
    """The statements that write the ``link`` table, recorded while the returned list is held."""
    seen = []

    def trace(sql):
        low = " ".join(sql.split()).lower()
        if any(w in low for w in ("insert into link", "delete from link", "update link", "replace into link")):
            seen.append(low)
    store.db.set_trace_callback(trace)
    return seen


def test_rs1_the_schema(tmp_path):
    store, root = _project(tmp_path)
    assert store.meta("schema") == "4" and store_mod.SCHEMA_VERSION == 4
    tables = {r[0] for r in store.db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"card", "link"} <= tables, tables
    assert _edges(store), "a new store computed no edge"
    # A store of schema 3, as 275 made it: no link table, its stamp 3; files, parts and their texts in it.
    store.db.execute("DROP TABLE link")
    store.db.execute("UPDATE meta SET value = '3' WHERE key = 'schema'")
    store.db.commit()
    parts = store.db.execute("SELECT id, embed_sha, sha256 FROM part ORDER BY id").fetchall()
    store.close()
    path = tmp_path / "proj.myc"
    reader = Store(path, lambda p: sqlite3.connect(f"file:{p}?mode=ro", uri=True), read_only=True)
    try:
        assert Searcher(reader).search("frobnicate_widgets", 3, stages=("symbols",)), "a store of 3 not searched"
        card = mp.file_card(reader, "oo", "app/a.py")
        assert card is not None and card["links_to"] == [] and card["linked_from"] == [], card
        said = mp.branch(reader, "oo", "app/a.py")
        assert "linked from:" not in said and "links to:" not in said, said
        assert mp.next_files(reader, "oo", ["app/a.py"])[0] == [], "a walk on a store of 3 read a link table"
    finally:
        reader.close()
    assert sqlite3.connect(path).execute("SELECT value FROM meta WHERE key = 'schema'").fetchone() == ("3",), (
        "reading moved the store")
    writer = _plain(path)
    assert writer.meta("schema") == "4"
    assert writer.db.execute("SELECT name FROM sqlite_master WHERE name = 'link'").fetchone() == ("link",)
    assert writer.db.execute("SELECT id, embed_sha, sha256 FROM part ORDER BY id").fetchall() == parts
    counts = Indexer(writer).index("oo", root)
    assert counts["recut"] == 0, "the move to 4 cut the parts again"
    assert _edges(writer) == EDGES, "the first pass after the move did not compute the edges"
    writer.close()

    class FailingStamp:
        def __init__(self, path):
            self.db = sqlite3.connect(path)

        def execute(self, sql, *args):
            if sql.startswith("UPDATE meta SET value = '4'"):
                raise sqlite3.OperationalError("disk I/O error")
            return self.db.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self.db, name)

    again = tmp_path / "again.myc"
    first = _plain(again)
    Indexer(first).index("oo", root)
    first.db.execute("DROP TABLE link")
    first.db.execute("UPDATE meta SET value = '3' WHERE key = 'schema'")
    first.db.commit()
    first.close()
    failing = FailingStamp(again)
    with pytest.raises(sqlite3.OperationalError):
        Store(again, lambda _p: failing)
    failing.db.close()
    held = sqlite3.connect(again)
    assert held.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone() == ("3",)
    assert held.execute("SELECT name FROM sqlite_master WHERE name = 'link'").fetchone() is None, (
        "a failed move left its table behind")
    held.close()
    # A store of schema 2: no card and no link table: it moves through 3 to 4.
    two = tmp_path / "two.myc"
    old = _plain(two)
    Indexer(old).index("oo", root)
    old.db.execute("DROP TABLE link")
    old.db.execute("DROP TABLE card")
    old.db.execute("UPDATE meta SET value = '2' WHERE key = 'schema'")
    old.db.commit()
    old.close()
    moved = _plain(two)
    assert moved.meta("schema") == "4"
    assert {r[0] for r in moved.db.execute("SELECT name FROM sqlite_master WHERE name IN ('card', 'link')")} == {
        "card", "link"}
    moved.close()
    # The move's table comes in the move's own transaction: an opening that fails after it leaves the store at 4
    # with its table.

    class FailingScript:
        def __init__(self, path):
            self.db = sqlite3.connect(path)

        def executescript(self, *_a):
            raise sqlite3.OperationalError("disk I/O error")

        def __getattr__(self, name):
            return getattr(self.db, name)

    three = tmp_path / "three.myc"
    held = _plain(three)
    Indexer(held).index("oo", root)
    held.db.execute("DROP TABLE link")
    held.db.execute("UPDATE meta SET value = '3' WHERE key = 'schema'")
    held.db.commit()
    held.close()
    failing = FailingScript(three)
    with pytest.raises(sqlite3.OperationalError):
        Store(three, lambda _p: failing)
    failing.db.close()
    after = sqlite3.connect(three)
    assert after.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone() == ("4",)
    assert after.execute("SELECT name FROM sqlite_master WHERE name = 'link'").fetchone() == ("link",)
    after.close()
    # A store at 4 that lacks a table (a process ended after its last move's commit) gets it at its next opening.
    lacking = _plain(two)
    lacking.db.execute("DROP TABLE vector")
    lacking.db.commit()
    lacking.close()
    mended = _plain(two)
    assert mended.db.execute("SELECT name FROM sqlite_master WHERE name = 'vector'").fetchone() == ("vector",)
    mended.close()
    # A whole store at 4 opened to write takes no lock: another connection writing it does not stop the opening.
    other = sqlite3.connect(two)
    other.execute("BEGIN IMMEDIATE")
    try:
        quick = Store(two, lambda p: sqlite3.connect(p, timeout=0.1, check_same_thread=False))
        quick.close()
    finally:
        other.rollback()
        other.close()
    moved = _plain(two)
    moved.db.execute("UPDATE meta SET value = '5' WHERE key = 'schema'")
    moved.db.commit()
    moved.close()
    for read_only in (False, True):
        with pytest.raises(StoreError) as refused:
            Store(two, sqlite3.connect, read_only=read_only)
        assert "schema 5" in str(refused.value), refused.value


def test_rs2_a_schema_1_store_opens_at_schema_4_and_is_cut_again(tmp_path, seams):
    root, path = mv.schema_1_store(tmp_path)
    store = Store(path, sqlite3.connect)
    assert store.meta("schema") == str(store_mod.SCHEMA_VERSION) == "4"
    assert {r[0] for r in store.db.execute("SELECT name FROM sqlite_master WHERE name IN ('card', 'link')")} == {
        "card", "link"}
    assert store.db.execute("SELECT embed_sha FROM part WHERE id = 1").fetchone() == ("",)
    counts = Indexer(store).index("oo", root)
    assert counts["recut"] == 1
    assert store.db.execute("SELECT COUNT(*) FROM part WHERE embed_sha = ''").fetchone()[0] == 0
    seams(mv.Registry({"enc-a": mv.Engine()}), mv.Governor())
    assert Indexer(store).embed(mv.Embedder(mv.A))["embedded"] > 0 and mv.texts_without_vector(store, mv.A) == 0
    store.close()


def test_rs3_each_move_from_schema_1_is_whole(tmp_path):
    for stamp in ("2", "3", "4"):
        (tmp_path / f"at{stamp}").mkdir()
        root, path = mv.schema_1_store(tmp_path / f"at{stamp}")

        class FailingMove:
            def __init__(self, path):
                self.db = sqlite3.connect(path)

            def execute(self, sql, *args, _stamp=stamp):
                if sql.startswith(f"UPDATE meta SET value = '{_stamp}'"):
                    raise sqlite3.OperationalError("disk I/O error")
                return self.db.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self.db, name)

        failing = FailingMove(path)
        with pytest.raises(sqlite3.OperationalError):
            Store(path, lambda _p, failing=failing: failing)
        failing.db.close()
        held = sqlite3.connect(path)
        before = str(int(stamp) - 1)
        assert held.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone() == (before,), stamp
        tables = {r[0] for r in held.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert ("card" in tables) == (int(before) >= 3) and ("link" in tables) == (int(before) >= 4), (stamp, tables)
        held.close()
        store = Store(path, sqlite3.connect)
        assert store.meta("schema") == "4" and store.meta("cutter:1") is None
        assert Indexer(store).index("oo", root)["recut"] == 1
        store.close()


def test_rs4_reading_writes_nothing(place, tmp_path):
    mod = place.mod
    root = pl._folder(tmp_path / "src", {"gate.py": pl.GATE})
    writer = mod.open_to_write("oo")
    Indexer(writer).index("oo", root)
    writer.close()
    before = pl._snapshot(place.folder)
    reader = mod.open_to_read("oo")
    assert [s["name"] for s in reader.sources()] == ["oo"]
    reader.generation()
    assert Searcher(reader).search("check_entity", 3, stages=("symbols",))
    assert mp.file_card(reader, "oo", "gate.py")["linked_from"] == []
    with pytest.raises(Exception):
        reader.meta("vectors", 7)
        reader.db.commit()
    reader.close()
    assert pl._snapshot(place.folder) == before, "reading wrote to the store"

    def planted(name, statements):
        db = sqlite3.connect(place.folder / f"{name}.db")
        for statement in statements:
            db.execute(statement)
        db.commit()
        db.close()
        return (place.folder / f"{name}.db").read_bytes()

    cases = {
        "newer": (["CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)",
                   "INSERT INTO meta VALUES ('schema', '5')"], "schema 5"),
        "old": (["CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)",
                 "INSERT INTO meta VALUES ('schema', '1')"], "schema 1"),
        "foreign": (["CREATE TABLE notes(body TEXT)", "INSERT INTO notes VALUES ('mine')"], "not a Mycelium store"),
    }
    for name, (statements, said) in cases.items():
        held = planted(name, statements)
        with pytest.raises(place.store.StoreError) as refused:
            mod.open_to_read(name)
        assert said in str(refused.value), (name, str(refused.value))
        assert (place.folder / f"{name}.db").read_bytes() == held, f"{name}: the refused file was changed"
    (place.folder / "empty.db").write_bytes(b"")
    with pytest.raises(place.store.StoreError) as refused:
        mod.open_to_read("empty")
    assert "not a Mycelium store" in str(refused.value)
    assert (place.folder / "empty.db").read_bytes() == b""
    assert sorted(p.name for p in place.folder.iterdir()) == ["empty.db", "foreign.db", "newer.db", "old.db",
                                                              "oo.db"], "a reader left a file behind"


def test_rs5_two_openers_of_a_schema_1_store_move_it_once(tmp_path):
    root, path = mv.schema_1_store(tmp_path)
    second = mv.SecondOpener(path)
    store = Store(path, lambda _p: second)
    assert second.done and store.meta("schema") == "4"
    columns = [r[1] for r in store.db.execute("PRAGMA table_info(part)")]
    assert columns.count("embed_sha") == 1
    assert store.db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'card'").fetchone() == (1,)
    assert store.db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'link'").fetchone() == (1,)
    assert Indexer(store).index("oo", root)["recut"] == 1
    store.close()


def test_rs6_the_edges(tmp_path):
    store, _root = _project(tmp_path)
    try:
        assert _edges(store) == EDGES
        assert store.db.execute("SELECT COUNT(*) FROM link WHERE src = dst").fetchone() == (0,)
        with pytest.raises(sqlite3.IntegrityError):
            store.db.execute("INSERT INTO link(src, dst, evidence) VALUES (1, 2, 0)")
        store.db.rollback()
        names = links.module_names
        assert names("src/lib2/m.py") == ["src.lib2.m", "lib2.m"] and names("app/__init__.py") == ["app"]
        assert names("lib/pkg/x.py") == ["lib.pkg.x", "pkg.x"] and names("tools/run.py") == ["tools.run"]
        assert names("src.py") == ["src"] and names("src/x.py") == ["src.x", "x"]
        assert names("src/x.py", frozenset({"src"})) == ["src.x"]      # src/ a package: its own name only
        assert names("lib/json.py") == ["lib.json"]                     # never the standard library's name
    finally:
        store.close()
    # A src/ that is a package keeps its name; a lib/ module named as the standard library never takes its imports.
    other, _r = _project(tmp_path, "pk", {
        "src/__init__.py": "",
        "src/utils.py": '"""U."""\n' + _fn("u()", "return 1"),
        "main.py": '"""Main."""\nfrom src.utils import u\nimport helpers\n' + _fn("go()", "return u() + helpers.hx()"),
        "helpers.py": '"""H."""\n' + _fn("hx()", "return 2"),
        "lib/json.py": '"""J."""\n' + _fn("dumps(x)", "return x"),
        "tools/run.py": '"""R."""\nimport json\n' + _fn("r()", "return json.dumps({})"),
        "alt.py": '"""A."""\nimport utils\n' + _fn("al()", "return utils.u()"),     # src/ a package: no `utils`
    })
    try:
        assert _edges(other) == {("main.py", "src/utils.py"): 2, ("main.py", "helpers.py"): 2}, _edges(other)
    finally:
        other.close()
    reviewed, _r = _project(tmp_path, "rv", REVIEWED)
    try:
        assert _edges(reviewed) == REVIEWED_EDGES, _edges(reviewed)
    finally:
        reviewed.close()


def test_rs7_forgetting_removing_renaming(tmp_path):
    store, root = _project(tmp_path)
    try:
        ids = dict(store.db.execute("SELECT path, id FROM file"))
        # Removed: app/b.py takes its edges both ways; what named it resolves to nothing left (the package app
        # defines no b).
        (root / "app" / "b.py").unlink()
        Indexer(store).index("oo", root)
        assert _edges(store) == {k: v for k, v in EDGES.items() if "app/b.py" not in k}
        assert store.db.execute("SELECT COUNT(*) FROM link WHERE src = ? OR dst = ?",
                                (ids["app/b.py"], ids["app/b.py"])).fetchone() == (0,)
        # Renamed: app/a.py -> app/z.py: its id kept; what imported app.a no longer resolves to it.
        (root / "app" / "a.py").rename(root / "app" / "z.py")
        counts = Indexer(store).index("oo", root)
        assert counts["renamed"] == 1
        assert dict(store.db.execute("SELECT path, id FROM file"))["app/z.py"] == ids["app/a.py"]
        assert not [k for k in _edges(store) if "app/z.py" in k], _edges(store)
        assert ("src/lib2/m.py", "src/lib2/n.py") in _edges(store)
        # Another source imports app.a's twin module: no edge crosses two sources.
        other = _write(tmp_path / "other", {"app/a.py": PROJECT["app/a.py"],
                                            "use.py": "from app.a import f\n" + _fn("w()", "return f()"),
                                            "far.py": "import lib2.n\n" + _fn("w2()", "return lib2.n.nn()")})
        Indexer(store).index("elsewhere", other)
        assert _edges(store, "elsewhere") == {("use.py", "app/a.py"): 2}
        crossing = store.db.execute(
            "SELECT COUNT(*) FROM link l JOIN file fa ON fa.id = l.src JOIN file fb ON fb.id = l.dst "
            "WHERE fa.source != fb.source").fetchone()
        assert crossing == (0,)
        # Forgotten: a source takes all of its edges, and the version of them with its settings: a new source given
        # its id later, first secret, then private, computes its edges at its next pass though no file changed.
        sid = store.db.execute("SELECT id FROM source WHERE name = 'elsewhere'").fetchone()[0]
        store.forget("elsewhere")
        assert _edges(store, "elsewhere") == {} and store.meta(f"links:{sid}") is None
        fresh = _write(tmp_path / "fresh", {"app/a.py": PROJECT["app/a.py"],
                                            "use.py": "from app.a import f\n" + _fn("w()", "return f()")})
        Indexer(store).index("fresh", fresh, sensitivity="secret")
        assert store.db.execute("SELECT id FROM source WHERE name = 'fresh'").fetchone()[0] == sid
        store.source("fresh", fresh, sensitivity="private")
        counts = Indexer(store).index("fresh", fresh)
        assert counts["added"] == counts["changed"] == 0
        assert _edges(store, "fresh") == {("use.py", "app/a.py"): 2}
        twin = _write(tmp_path / "twin", {"lib/x.py": '"""X."""\n' + _fn("xa()", "return 1"),
                                          "x.py": '"""X."""\n' + _fn("xa()", "return 2"),
                                          "user.py": "import x\n" + _fn("ux()", "return x.xa()")})
        Indexer(store).index("twin", twin)
        assert _edges(store, "twin") == {("user.py", "lib/x.py"): 2}     # lib/ no package: x, the first by path
        (twin / "lib" / "x.py").unlink()
        counts = Indexer(store).index("twin", twin)
        assert counts["removed"] == 1 and counts["changed"] == counts["added"] == 0
        assert _edges(store, "twin") == {("user.py", "x.py"): 2}, "a pass that only removed computed no edge"
        store.forget("oo")
        assert _edges(store) == {} and _edges(store, "fresh") == {("use.py", "app/a.py"): 2}
    finally:
        store.close()


def test_rs8_a_secret_source(tmp_path):
    root = _write(tmp_path / "proj", PROJECT)
    store = _plain(tmp_path / "s.myc")
    try:
        Indexer(store).index("vault", root, sensitivity="secret")
        assert store.db.execute("SELECT COUNT(*) FROM part").fetchone()[0] > 0
        assert _edges(store, "vault") == {}, "a secret source's edges were computed"
        vault = store.db.execute("SELECT id FROM source WHERE name = 'vault'").fetchone()[0]
        assert links.recompute(store, vault, moved=True) == {"links": 0, "dropped": 0}
        assert _edges(store, "vault") == {} and store.meta(f"links:{vault}") is None, "asked, a secret source linked"
        store.db.commit()
        Indexer(store).index("oo", _write(tmp_path / "open", PROJECT))
        assert _edges(store) == EDGES
        store.meta("rewrite") is None or store.rewrite()
        assert store.meta("rewrite") is None
        store.source("oo", tmp_path / "open", sensitivity="secret")
        assert _edges(store) == {} and store.meta("rewrite") is not None, "marked secret, its edges stayed"
        store.rewrite()
        store.source("oo", tmp_path / "open", sensitivity="private")
        counts = Indexer(store).index("oo", tmp_path / "open")
        assert counts["added"] == counts["changed"] == counts["removed"] == 0
        assert _edges(store) == EDGES, "marked private again, its next pass computed no edge"
    finally:
        store.close()


def test_rs9_what_a_pass_writes(tmp_path, monkeypatch):
    store, root = _project(tmp_path)
    computed = []
    real = links._edges

    def counted(db, source):
        computed.append(source)
        return real(db, source)

    monkeypatch.setattr(links, "_edges", counted)
    try:
        seen = _link_writes(store)
        counts = Indexer(store).index("oo", root)
        assert counts["added"] == counts["changed"] == counts["removed"] == counts["recut"] == 0
        assert seen == [] and computed == [], (seen, computed)
        # The probes see a write and a computation: a changed file computes the edges again and writes them.
        (root / "app" / "d.py").write_text(PROJECT["app/d.py"] + _fn("more()", "return f()"))
        Indexer(store).index("oo", root)
        assert seen and computed, "the probes saw nothing"
        assert _edges(store)[("app/d.py", "app/a.py")] == 2
        store.db.set_trace_callback(None)
        if store.meta("rewrite"):
            store.rewrite()
        # A new version of the resolution: every edge computed again at the next pass, though no file changed; an
        # edge it drops owes a rewrite.
        ids = dict(store.db.execute("SELECT path, id FROM file"))
        store.db.execute("INSERT INTO link(src, dst, evidence) VALUES (?, ?, 1)", (ids["app/u.py"], ids["app/e.py"]))
        store.db.commit()
        monkeypatch.setattr(links, "LINKS_VERSION", links.LINKS_VERSION + 1)
        seen = _link_writes(store)
        Indexer(store).index("oo", root)
        assert seen and ("app/u.py", "app/e.py") not in _edges(store)
        assert store.meta("rewrite") is not None, "a dropped edge owed no rewrite"
        store.rewrite()
        seen, computed[:] = _link_writes(store), []
        Indexer(store).index("oo", root)
        assert seen == [] and computed == [] and store.meta("rewrite") is None
    finally:
        store.close()


def test_rs10_a_failed_pass(tmp_path, monkeypatch):
    store, root = _project(tmp_path)
    try:
        before = _edges(store)
        (root / "app" / "new.py").write_text("from app.a import f\n" + _fn("n()", "return f()"))
        (root / "app" / "d.py").unlink()

        def broken(*_a, **_k):
            raise RuntimeError("the episode cannot be written")

        monkeypatch.setattr(store, "episode", broken)
        with pytest.raises(RuntimeError):
            Indexer(store).index("oo", root)
        monkeypatch.undo()
        assert _edges(store) == before, "a failed pass kept edges of its own"
        assert store.db.execute("SELECT COUNT(*) FROM file WHERE path = 'app/new.py'").fetchone() == (0,)
    finally:
        store.close()


def test_rs11_the_files_card(tmp_path):
    store, root = _project(tmp_path)
    try:
        card = mp.file_card(store, "oo", "app/a.py")
        assert card["links_to"] == []
        assert card["linked_from"] == [("app/b.py", 2), ("app/c.py", 2), ("app/deep/k.py", 2), ("app/d.py", 1)], (
            card["linked_from"])
        card = mp.file_card(store, "oo", "app/c.py")
        assert card["links_to"] == [("app/a.py", 2), ("app/b.py", 2)], card["links_to"]
        said = mp.branch(store, "oo", "app/c.py")
        assert "links to: app/a.py (2), app/b.py (2)" in said and "linked from:" not in said, said
        said = mp.branch(store, "oo", "app/a.py")
        assert "linked from: app/b.py (2), app/c.py (2), app/deep/k.py (2), app/d.py (1)" in said, said
        assert "links to:" not in said
        _write(root, {f"app/m{n:02d}.py": "from app.a import f\n" for n in range(12)})
        Indexer(store).index("oo", root)
        said = mp.branch(store, "oo", "app/a.py")
        line = next(x for x in said.splitlines() if x.startswith("linked from: "))
        assert line.count(" (") == 8 and line.endswith(", ... 8 more"), line
    finally:
        store.close()


def test_rs12_the_walk(owned, tmp_path):
    root = _write(tmp_path / "proj", PROJECT)
    my_t._index(owned, "proj", root)
    other = _write(tmp_path / "other", {"app/a.py": PROJECT["app/a.py"], "near.py": "from app.a import f\n"})
    my_t._index(owned, "proj", other, source="elsewhere")
    store = owned.place.open_to_read("proj")

    def again():
        nonlocal store
        store.close()
        my_t._index(owned, "proj", root)
        store = owned.place.open_to_read("proj")

    try:
        linked, named = mp.next_files(store, "proj", ["app/a.py"])     # linked to it: incoming edges
        assert linked == [("app/b.py", 1), ("app/c.py", 1), ("app/d.py", 1), ("app/deep/k.py", 1)], linked
        assert named == [("app/u.py", 1), ("docs/guide.md", 1)], named  # u calls f by attribute; the guide names it
        assert mp.next_files(store, "proj", ["app/b.py"]) == ([("app/a.py", 1), ("app/c.py", 1)], [])   # both ways
        # Same names both ways: u calls f, which a defines; r calls use, which u defines.
        assert mp.next_files(store, "proj", ["app/u.py"]) == ([], [("app/a.py", 1), ("app/r.py", 1)])
        linked, named = mp.next_files(store, "proj", ["app/a.py", "app/b.py"])
        assert linked == [("app/c.py", 2), ("app/d.py", 1), ("app/deep/k.py", 1)], linked
        assert named == [("app/u.py", 1), ("docs/guide.md", 1)], named      # never an opened file
        assert mp.next_files(store, "proj", ["src/lib2/n.py"]) == ([("src/lib2/m.py", 1), ("src/lib2/n2.py", 1)], [])
        assert mp.next_files(store, "proj", []) == ([], []) == mp.next_files(store, "proj", ["nowhere.py"])
        # A name with more than six definitions counts for nothing in the second list; six count.
        _write(root, {**{f"app/w{n}.py": "def common():\n    return 0\n" for n in range(7)},
                      "app/caller.py": "def call(o):\n    return o.common()\n",
                      **{f"app/six{n}.py": "def sixth():\n    return 0\n" for n in range(6)},
                      "app/caller6.py": "def call6(o):\n    return o.sixth()\n"})
        again()
        assert mp.next_files(store, "proj", ["app/caller.py"]) == ([], [])
        assert mp.next_files(store, "proj", ["app/caller6.py"]) == ([], [(f"app/six{n}.py", 1) for n in range(6)])
        _write(root, {**{f"app/k{n}.py": "def kinds():\n    return 0\n" for n in range(4)},
                      **{f"app/x{n}/kinds.py": '"""K."""\n' for n in range(3)},
                      "app/callk.py": "def callk(o):\n    return o.kinds()\n"})
        again()
        assert mp.next_files(store, "proj", ["app/callk.py"]) == ([], []), "a module's name not counted"
        # At most eight each, most first then by path; a file the linked list leaves past its eight is in the second.
        _write(root, {f"app/m{n:02d}.py": "from app.a import f\n" for n in range(12)})
        again()
        linked, named = mp.next_files(store, "proj", ["app/a.py"])
        assert linked == [("app/b.py", 1), ("app/c.py", 1), ("app/d.py", 1), ("app/deep/k.py", 1)] + [
            (f"app/m{n:02d}.py", 1) for n in range(4)], linked
        assert named == [(f"app/m{n:02d}.py", 1) for n in range(4, 12)], named
        said = mp.walk(store, "proj", ["app/a.py", "nowhere.py"])
        assert said.splitlines()[0] == "proj: next from 2 opened files", said
        assert "not indexed in proj: nowhere.py" in said and "linked (resolved, both ways): app/b.py (1)" in said
        assert "near.py" not in said, "another source's file came"
        assert "same names: app/m04.py (1)" in said, said
        alone = mp.walk(store, "proj", ["src/lib2/n.py"])
        assert alone.splitlines() == ["proj: next from 1 opened file",
                                      "linked (resolved, both ways): src/lib2/m.py (1), src/lib2/n2.py (1)",
                                      "same names: none"], alone
        lost = mp.walk(store, "proj", [f"gone{n}.py" for n in range(10)])
        assert "not indexed in proj: gone0.py" in lost and lost.splitlines()[1].endswith(", ... 2 more"), lost
        assert lost.endswith("linked (resolved, both ways): none\nsame names: none"), lost
        eight = mp.walk(store, "proj", [f"gone{n}.py" for n in range(8)])
        assert eight.splitlines()[1].endswith("gone7.py"), eight
    finally:
        store.close()


def test_rs13_the_tool_declared(owned, tmp_path):
    from opti_oignon import provenance
    from opti_oignon.agent import allowlists
    from opti_oignon.agent import tools as agent_tools
    from opti_oignon.agent_eval import runner

    schema = next(s for s in agent_tools.ALL_SCHEMAS if s.name == "mycelium_map")
    assert [(p.name, p.required) for p in schema.parameters] == [("path", False), ("source", False),
                                                                ("opened", False)]
    assert next(p for p in schema.parameters if p.name == "opened").json_type() == "array"
    assert schema.sandboxed is False and "mycelium_map" in agent_tools.HANDLER_TOOL_NAMES
    assert provenance.TOOL_EFFECTS["mycelium_map"] == provenance.NONE
    assert "mycelium_map" in allowlists.READ_TOOLS and "mycelium_map" in runner._DISABLED_TOOLS
    root = _write(tmp_path / "proj", PROJECT)
    my_t._index(owned, "proj", root)
    run = my_t._cloned(tmp_path, "run1", os.path.realpath(root))
    card = owned.tool.card(run)
    assert "mycelium_map walks the same corpus's map without reading any file" in card, card
    assert "not something to cite" in card, card
    repo = Path(__file__).resolve().parents[1]
    security = " ".join((repo / "SECURITY.md").read_text(encoding="utf-8").split())
    assert "Its map, `mycelium_map`, reads the same sources the same way, from the store's index alone" in security
    walk = owned.tool.make_map_handler(run)
    said = walk({"opened": ["app/a.py"]})
    assert said.splitlines()[0] == "proj: next from 1 opened file" and "app/b.py (1)" in said, said
    for bad in ("app/a.py", [3], ["x" * 1001], [f"f{n}.py" for n in range(26)]):
        assert walk({"opened": bad}).startswith("mycelium_map: opened is a list of at most 25 paths"), bad
    for fine in ([f"f{n}.py" for n in range(25)], ["x" * 1000], ["y" * 100]):
        assert walk({"opened": fine}).startswith("proj: next from "), fine
    assert walk({"path": "app", "opened": ["app/a.py"]}).startswith("mycelium_map: give a path or opened, not both")
    assert walk({"opened": []}).startswith("proj:/"), "an empty opened is no walk: the root's map"


def test_rs14_the_cutter(tmp_path, monkeypatch):
    store, root = _project(tmp_path)
    try:
        dotted = sorted(store.db.execute(
            "SELECT f.path, r.name, r.module FROM ref r JOIN part p ON p.id = r.part JOIN file f ON f.id = p.file "
            "WHERE r.kind = 'dotted'").fetchall())
        # A head an import of a module bound: the module's path; a head a `from m import n` bound: `m:n`.
        assert ("app/c.py", "alias.f", "app.a") in dotted and ("app/c.py", "b.g", "app:b") in dotted, dotted
        assert ("app/e.py", "app.e.k", "app") in dotted and ("src/lib2/m.py", "n.nn", "src.lib2:n") in dotted
        assert ("other/x.py", "json.dumps", "json") in dotted and ("app/v.py", "p.impl.Base", "app.pkg") in dotted
        assert ("app/e.py", "app.e", "app") not in dotted, "a chain recorded with its shorter head"
        assert not [r for r in dotted if r[0] in ("app/u.py", "app/s.py")], "a chain on no import, or written"
        # One part's whole record, written by hand: c's h.
        h = set(store.db.execute(
            "SELECT r.name, r.kind, r.module FROM ref r JOIN part p ON p.id = r.part JOIN file f ON f.id = p.file "
            "WHERE f.path = 'app/c.py' AND p.symbol = 'h'"))
        assert h == {("f", "call", ""), ("alias.f", "dotted", "app.a"), ("alias", "use", "app.a"),
                     ("g", "call", ""), ("b.g", "dotted", "app:b"), ("b", "use", "app")}, h
        # The head of a plain `import app.e` is bound to `app` (it was the module named last).
        assert ("app", "use", "app") in set(store.db.execute(
            "SELECT r.name, r.kind, r.module FROM ref r JOIN part p ON p.id = r.part JOIN file f ON f.id = p.file "
            "WHERE f.path = 'app/e.py' AND p.symbol = 'k'"))
        from opti_oignon.mycelium import traits

        def readers():
            cards = {p: mp.file_card(store, "oo", p) for (p,) in store.db.execute("SELECT path FROM file")}
            for c in cards.values():
                c.pop("links_to"), c.pop("linked_from")
            rows = traits.rows(Searcher(store), "where is f called", depth=10)
            return cards, (rows.files, [list(map(float, r)) for r in rows.rows])

        with_dotted = readers()
        store.db.execute("DELETE FROM ref WHERE kind = 'dotted'")
        store.db.commit()
        store.matrices.clear()
        assert readers() == with_dotted, "a dotted chain changed what a reader by name gives"
        # A store cut by version 5: cut again, its embedded texts kept, no rewrite owed.
        texts = dict(store.db.execute("SELECT f.path || ':' || p.first, p.embed_sha FROM part p "
                                      "JOIN file f ON f.id = p.file"))
        store.meta("cutter:1", "5")
        store.db.commit()
        if store.meta("rewrite"):
            store.rewrite()
        counts = Indexer(store).index("oo", root)
        assert counts["recut"] == store.db.execute("SELECT COUNT(*) FROM file").fetchone()[0], counts
        assert dict(store.db.execute("SELECT f.path || ':' || p.first, p.embed_sha FROM part p "
                                     "JOIN file f ON f.id = p.file")) == texts
        assert store.meta("rewrite") is None, "a recut owed a rewrite"
        assert index_mod.CHUNKER_VERSION == 6
    finally:
        store.close()
    spaces, _r = _project(tmp_path, "ns2", {
        "nsmod.py": ("import os\na, (b, *c) = 1, (2, 3)\n[d] = [4]\nX: int = 5\nY = 0\nY += 1\nos.sep2 = 'x'\n"
                     "for K in range(2):\n    pass\nwith open(os.devnull) as W:\n    pass\n"
                     "try:\n    import json as J2\nexcept ImportError:\n    H = None\nelse:\n    E = 1\n"
                     "finally:\n    F = 2\nif True:\n    def fin():\n        local_x = 1\n        return local_x\n"
                     "    class Cin:\n        inner = 1\nwhile False:\n    Wh = 3\n"),
        "loc.py": '"""L."""\n' + _fn("lo()", "import os.path as op\n    return op"),
        "app/__init__.py": "", "app/a.py": PROJECT["app/a.py"], "app/b.py": PROJECT["app/b.py"],
        "seq.py": '"""S."""\nfrom app import b\nimport app.a as alias\n' + _fn("sq()", "return alias.f(b)"),
    })
    try:
        def records(path, kinds):
            return set(spaces.db.execute(
                "SELECT r.name, r.kind, r.module FROM ref r JOIN part p ON p.id = r.part JOIN file f ON f.id = p.file "
                f"WHERE f.path = ? AND r.kind IN ({','.join('?' * len(kinds))})", (path, *kinds)))

        assert records("nsmod.py", ("assign",)) == {(f"={n}", "assign", "") for n in (
            "a", "b", "c", "d", "X", "Y", "K", "W", "H", "E", "F", "fin", "Cin", "Wh")}, records("nsmod.py", ("assign",))
        assert records("nsmod.py", ("binds",)) == {("=os", "binds", "os"), ("=J2", "binds", "json")}
        assert records("loc.py", ("binds", "assign")) == set(), "a function's own import in the module's namespace"
        assert ("=b", "bound", "app:b") in records("seq.py", ("bound",)), "a bare name after a chain lost its record"
    finally:
        spaces.close()
    # On the package's own code, the chains are the only records the attribute visitor adds: every other record is
    # the one a cutter without it makes.
    from opti_oignon.mycelium import chunk

    package = Path(__file__).resolve().parents[1] / "opti_oignon"
    sources = []
    for folder, dirs, names in os.walk(package):
        dirs[:] = sorted(d for d in dirs if d not in ("data", "__pycache__"))
        sources += [Path(folder) / n for n in sorted(names) if n.endswith(".py")]

    def records():
        out = {}
        for path in sources:
            rel = path.relative_to(package.parent).as_posix()
            for part in chunk.cut(rel, path.read_text(encoding="utf-8").split("\n")):
                out.setdefault(rel, []).append(sorted(part.refs))
        return out

    network = {"dotted", "imodule", "ifrom", "bound", "binds", "assign"}
    whole = records()
    monkeypatch.setattr(chunk, "NETWORK", False)
    bare = records()
    rows = [r for parts in whole.values() for p in parts for r in p if r[1] in network]
    assert len(whole) > 300 and {r[1] for r in rows} == network and len(rows) > 5000, len(rows)
    assert all(r[0].startswith("=") or "." in r[0] for r in rows), "a network record that could join by name"
    assert not [r for parts in bare.values() for p in parts for r in p if r[1] in network]
    for rel, parts in whole.items():
        assert [[r for r in p if r[1] not in network] for p in parts] == bare[rel], rel


def test_rs15_the_texts():
    repo = Path(__file__).resolve().parents[1]
    changelog = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = " ".join(next((e for e in changelog.split("\n- ") if "RS1-RS15" in e), "").split())
    for said in ("schema 4", "`link`", "mycelium_map(path?, source?, opened?)", "both ways", "not a secret one",
                 "cutter", "a Python file links", "a source of documents has none", "within ten openings for 39%"):
        assert said in entry, (said, entry[:300])
    security = " ".join((repo / "SECURITY.md").read_text(encoding="utf-8").split())
    for said in ("a source marked secret loses its links at once, a rewrite owed",
                 "the files it links to and that link to it", "the files to open next"):
        assert said in security, said
