"""Contracts of the traits of a search's candidate files (``opti_oignon/mycelium/traits.py``) and of the search's lists
before their fusion (``Searcher.lists``, ``search.fuse``).

The traits are what the value head (``value.py``) orders a question's candidate files by. They are read from a real
store, indexed by the real ``Indexer``, its vectors those of a stand-in encoder on this machine (the seams of
``test_mycelium_vector_contracts``).

  * TR1 -- THE CANDIDATES AND THEIR ROWS: the files of the fused list's parts, in its order, each once, ``depth`` at
    most, each named by its source and its path; a row of the named traits for each, finite.
  * TR2 -- EACH TRAIT SAYS WHAT IT SAYS: the file that defines the question's identifier, the one three others name,
    the test, the markdown, the path that holds the question's words; its parts counted; its vectors' best cosine and
    the cosines of the means of its parts and of its folder's, as the matrix and the question's vector give them; its
    rank in each list and its best scores, as the lists give them.
  * TR3 -- THE SEARCH IS THE SAME: ``find`` cites the parts of the fusion of the lists ``lists`` gives, in order,
    the highest fused score first, parts of equal scores by their ids (review 1); the vectors' list is ``dense``'s, the
    closest first; the part ids ``lists`` hands over are the caller's own.
  * TR4 -- WHAT THE SEARCHER MAY NOT READ IS NOT READ: a secret source gives no candidate, no link and no vector to a
    mean (its files and folders absent from the tables, review 1); a searcher allowed to read it reads it.
  * TR5 -- A SILENT LIST: with no encoder active, the vectors' traits are 0 and the silence is said; past the turn's
    deadline, the store's tables are not read (said); the matrix not read (its store held) is said and not kept
    (review 1); the search's ids not the tables' ones: said, the vectors' traits at 0 (review 2).
  * TR6 -- READ ONCE PER GENERATION: the store's tables are read once while its generation holds, again once it moves,
    whichever opening of the store's file a searcher reads (the turn opens one per search, review 1); another store's
    bytes in the same file, at the same generation, are another store's tables, and a store's file gone takes its
    tables with it (review 2).
  * TR7 -- THE CHANGELOG SAYS IT.
  * TR8 -- A NAME DEFINED TOO OFTEN MAKES NO LINK, AND A FILE'S KIND IS ITS LANGUAGE (review 1): a name more than the
    search's own bound defines links no file, one the bound defines links its files (review 2); a .yml file is yaml,
    a .js file the store reads as ts, a .MD markdown.
  * TR9 -- THE TABLES ARE ONE READING OF THE STORE (review 2): a pass of another connection that commits while they
    are read waits for them; they are never read half before it and half after.
"""
import math
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_mycelium_vector_contracts import A, Engine, Governor, Registry  # noqa: E402

from opti_oignon.mycelium import search, traits, vectors  # noqa: E402
from opti_oignon.mycelium.index import Indexer  # noqa: E402
from opti_oignon.mycelium.search import Searcher  # noqa: E402
from opti_oignon.mycelium.store import Store  # noqa: E402
from opti_oignon.mycelium.vectors import Embedder  # noqa: E402

BUDGET_S = {
    "test_tr1_the_candidates_are_the_fused_lists_files_each_with_its_row": 2.0,
    "test_tr2_each_trait_says_what_it_says": 2.0,
    "test_tr3_the_search_cites_the_fusion_of_its_lists": 2.0,
    "test_tr4_what_the_searcher_may_not_read_is_not_read": 3.0,
    "test_tr5_a_silent_list_leaves_its_traits_at_nought_and_says_why": 2.0,
    "test_tr5b_past_the_deadline_and_with_the_matrix_held": 2.0,
    "test_tr6_the_stores_tables_are_read_once_per_generation": 3.0,
    "test_tr6b_another_stores_bytes_in_the_same_file_are_another_store": 3.0,
    "test_tr7_the_changelog_says_the_traits": 1.0,
    "test_tr8_a_name_defined_too_often_makes_no_link_and_a_kind_is_a_language": 2.0,
    "test_tr9_the_tables_are_one_reading_of_the_store": 6.0,
}

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "select_peels share of terms"
FILES = {
    "pkg/gate.py": 'def select_peels(conversation_id, query):\n    """Choose the peels of a conversation by their share '
                   'of terms."""\n    chosen = []\n    for peel in conversation_id:\n        if peel:\n'
                   '            chosen.append(peel)\n    return chosen\n\n\ndef share_of(a, b):\n'
                   '    """The share of terms two texts hold."""\n    left = set(a.split())\n    right = set(b.split())\n'
                   '    if not left:\n        return 0.0\n    return len(left & right) / len(left)\n',
    "pkg/peels.py": "from pkg.gate import select_peels\n\n\ndef held(conversation_id):\n"
                    '    return select_peels(conversation_id, "")\n',
    "pkg/other.py": 'from pkg.gate import select_peels\n\n\ndef again():\n    return select_peels("a", "b")\n',
    "tests/test_gate.py": "from pkg.gate import select_peels\n\n\ndef test_select_peels_is_empty():\n"
                          '    assert select_peels("c", "q") == []\n',
    "docs/guide.md": "# Guide\n\nHow the peels of a conversation are selected by their share of terms.\n",
    "config/settings.yaml": "peels:\n  share: 0.5\n",
}
SECRET = {
    "vault/use.py": "from pkg.gate import select_peels\n\n\ndef hidden():\n    return select_peels(\"x\", \"y\")\n",
    "vault/peels_notes.md": "# Notes\n\nThe share of terms of the peels, select peels.\n",
}


@pytest.fixture(autouse=True)
def seams(monkeypatch):
    """The stand-in encoder on this machine; no contract reaches the machine's engine or governor."""
    monkeypatch.setattr(vectors, "_registry", lambda: Registry({"enc-a": Engine()}))
    monkeypatch.setattr(vectors, "_governor", lambda: Governor())


def _write(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def store(tmp_path):
    held = Store(tmp_path / "corpus.myc", sqlite3.connect)
    Indexer(held).index("oo", _write(tmp_path / "src", FILES))
    Indexer(held).embed(Embedder(A))
    yield held
    held.close()


def _paths_of(store):
    return {part: (source, path) for part, source, path in store.db.execute(
        "SELECT p.id, s.name, f.path FROM part p JOIN file f ON f.id = p.file JOIN source s ON s.id = f.source")}


def _row(found, path, source="oo"):
    return dict(zip(traits.TRAITS, found.rows[found.files.index((source, path))]))


def test_tr1_the_candidates_are_the_fused_lists_files_each_with_its_row(store):
    searcher = Searcher(store)
    found = traits.rows(searcher, QUESTION)
    assert len(traits.TRAITS) == len(set(traits.TRAITS)) == 19
    listed = searcher.lists(QUESTION)
    order, _fused, _why = search.fuse(listed.lists)
    paths = _paths_of(store)
    expected = []
    for part in order:
        if paths[part] not in expected:
            expected.append(paths[part])
    assert found.files == expected[:traits.DEPTH] and len(found.files) >= 4, (found.files, expected)
    assert len(found.rows) == len(found.files)
    assert all(len(row) == 19 and all(math.isfinite(x) for x in row) for row in found.rows)
    assert traits.rows(searcher, QUESTION, depth=2).files == expected[:2]
    assert traits.rows(searcher, QUESTION, depth=0) == traits.Candidates([], [], traits.rows(searcher, QUESTION).silent)


def test_tr2_each_trait_says_what_it_says(store):
    import numpy as np

    searcher = Searcher(store)
    found = traits.rows(searcher, QUESTION)
    gate, peels = _row(found, "pkg/gate.py"), _row(found, "pkg/peels.py")
    assert gate["defines_named"] == 1.0 and peels["defines_named"] == 0.0
    assert gate["links"] == pytest.approx(math.log1p(3)), "three other files name select_peels"
    assert peels["links"] == 0.0
    assert gate["kind_py"] == 1.0 and sum(gate[f"kind_{k}"] for k in traits.KINDS) == 1.0
    assert _row(found, "tests/test_gate.py")["test"] == 1.0 and gate["test"] == 0.0
    if ("oo", "docs/guide.md") in found.files:
        assert _row(found, "docs/guide.md")["kind_md"] == 1.0
    assert peels["path_words"] == pytest.approx(1 / 5) and gate["path_words"] == 0.0
    counted = dict(store.db.execute("SELECT f.path, COUNT(*) FROM part p JOIN file f ON f.id = p.file GROUP BY f.path"))
    assert gate["parts"] == pytest.approx(math.log1p(counted["pkg/gate.py"]))
    _active, ids, matrix = searcher._matrix()
    question = np.asarray(Embedder(A).query(QUESTION), dtype=np.float64)
    question /= np.linalg.norm(question)
    paths = _paths_of(store)
    for path in ("pkg/gate.py", "pkg/peels.py"):
        rows = np.array([matrix[j] for j, i in enumerate(ids) if paths[i] == ("oo", path)], dtype=np.float64)
        folder = np.array([matrix[j] for j, i in enumerate(ids) if paths[i][1].startswith("pkg/")], dtype=np.float64)
        row = _row(found, path)
        assert row["dense_best"] == pytest.approx(float((rows @ question).max()), abs=1e-5)
        mean = rows.sum(axis=0)
        assert row["centroid"] == pytest.approx(float(mean @ question / np.linalg.norm(mean)), abs=1e-5)
        mean = folder.sum(axis=0)
        assert row["folder"] == pytest.approx(float(mean @ question / np.linalg.norm(mean)), abs=1e-5)
    listed = searcher.lists(QUESTION)
    for stage, trait in (("lexical", "lexical_rank"), ("dense", "dense_rank")):
        files = []
        for part, _score in listed.lists[stage]:
            if paths[part] not in files:
                files.append(paths[part])
        for named in found.files:
            want = 1 / (1 + files.index(named)) if named in files else 0.0
            assert _row(found, named[1])[trait] == pytest.approx(want), (stage, named)
    assert counted["pkg/gate.py"] >= 2, "a file of more than one part"
    best = max(score for part, score in listed.lists["lexical"] if paths[part] == ("oo", "pkg/gate.py"))
    assert gate["lexical_best"] == pytest.approx(best) and gate["lexical_best"] > 0
    _order, fused, _why = search.fuse(listed.lists)
    assert gate["fused_best"] == pytest.approx(max(s for p, s in fused.items() if paths[p] == ("oo", "pkg/gate.py")))
    assert gate["rewritten_rank"] == 0.0, "no bridge: no rewritten list"


def test_tr3_the_search_cites_the_fusion_of_its_lists(store):
    searcher = Searcher(store)
    listed = searcher.lists(QUESTION)
    order, _fused, _why = search.fuse(listed.lists)
    hits = searcher.find(QUESTION, k=5).hits
    assert [h["part"] for h in hits] == order[:5] and len(hits) == 5
    assert listed.worded == QUESTION and listed.scores is not None and len(listed.scores[0]) == len(listed.scores[1])
    dense = searcher.dense(QUESTION)
    assert dense == listed.lists["dense"]
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True), "the fusion orders by the reciprocal ranks, the highest first"
    cosines = [score for _part, score in dense]
    assert cosines == sorted(cosines, reverse=True) and len(cosines) >= 4, "the vectors' list, the closest first"
    tied = {"one": [(5, 1.0), (3, 0.9)], "two": [(3, 1.0), (5, 0.9)], "three": [(9, 1.0)]}
    order, fused, _why = search.fuse(tied)
    assert order == [3, 5, 9] and fused[3] == fused[5] == 1 / 61 + 1 / 62, "equal fused scores: the lower id first"
    listed.scores[0].reverse()
    assert searcher.dense(QUESTION) == dense, "the ids lists hands over are not the store's own"


def test_tr4_what_the_searcher_may_not_read_is_not_read(store, tmp_path):
    Indexer(store).index("vault", _write(tmp_path / "secret", SECRET), sensitivity="secret")
    Indexer(store).embed(Embedder(A))
    closed = traits.rows(Searcher(store), QUESTION)
    assert all(source == "oo" for source, _path in closed.files), closed.files
    assert _row(closed, "pkg/gate.py")["links"] == pytest.approx(math.log1p(3)), "the secret namer is not counted"
    opened = traits.rows(Searcher(store, allow=("public", "private", "secret")), QUESTION)
    assert any(source == "vault" for source, _path in opened.files), opened.files
    assert _row(opened, "pkg/gate.py")["links"] == pytest.approx(math.log1p(4)), "read, the secret namer counts"
    tables = traits._held(Searcher(store))
    assert all(source == "oo" for source, _path, _lang in tables["named"].values())
    assert all(source == "oo" for source, _folder in tables["folder_norms"]) and tables["folder_norms"]
    opened_tables = traits._held(Searcher(store, allow=("public", "private", "secret")))
    assert any(source == "vault" for source, _folder in opened_tables["folder_norms"])
    named = traits.rows(Searcher(store, allow=("public", "private", "secret"), sources=("oo",)), QUESTION)
    assert all(source == "oo" for source, _path in named.files)
    assert _row(named, "pkg/gate.py")["links"] == pytest.approx(math.log1p(3))
    assert _row(named, "pkg/gate.py")["folder"] == pytest.approx(_row(closed, "pkg/gate.py")["folder"])


def test_tr5_a_silent_list_leaves_its_traits_at_nought_and_says_why(tmp_path):
    held = Store(tmp_path / "bare.myc", sqlite3.connect)
    try:
        Indexer(held).index("oo", _write(tmp_path / "src", FILES))
        found = traits.rows(Searcher(held), QUESTION)
        assert found.files and "no encoder is active" in found.silent["dense"]
        for row in found.rows:
            named = dict(zip(traits.TRAITS, row))
            assert named["dense_best"] == named["dense_rank"] == named["centroid"] == named["folder"] == 0.0
        assert _row(found, "pkg/gate.py")["defines_named"] == 1.0, "the other lists still answer"
    finally:
        held.close()


def test_tr5b_past_the_deadline_and_with_the_matrix_held(store, monkeypatch):
    import threading

    stop = threading.Event()
    stop.set()
    late = traits.rows(Searcher(store, stop=stop), QUESTION)
    assert "deadline" in late.silent["traits"] and late.files, late.silent
    assert all(dict(zip(traits.TRAITS, row))["links"] == 0.0 for row in late.rows), "the tables were not read"
    searcher = Searcher(store)
    real = Searcher._matrix
    monkeypatch.setattr(Searcher, "_matrix", lambda self: None)
    blind = traits.rows(searcher, QUESTION)
    assert "could not be read" in blind.silent["traits"], blind.silent
    monkeypatch.setattr(Searcher, "_matrix", real)
    seen = traits.rows(searcher, QUESTION)
    assert "traits" not in seen.silent and _row(seen, "pkg/gate.py")["centroid"] > 0, "a matrix not read is not kept"
    real_lists = Searcher.lists

    def moved(self, query, stages=search.STAGES):
        listed = real_lists(self, query, stages)
        return listed._replace(scores=(list(reversed(listed.scores[0])), listed.scores[1]))

    monkeypatch.setattr(Searcher, "lists", moved)
    shifted = traits.rows(searcher, QUESTION)
    assert "moved" in shifted.silent["traits"] and _row(shifted, "pkg/gate.py")["centroid"] == 0.0, shifted.silent


def test_tr6_the_stores_tables_are_read_once_per_generation(store, tmp_path, monkeypatch):
    calls = []
    real = traits._tables

    def counted(searcher):
        calls.append(1)
        return real(searcher)

    monkeypatch.setattr(traits, "_tables", counted)
    searcher = Searcher(store)
    first = traits.rows(searcher, QUESTION)
    traits.rows(Searcher(store), QUESTION)
    assert len(calls) == 1, "the generation held: read once, for every searcher of the store"
    other = Store(store.path, sqlite3.connect)
    try:
        traits.rows(Searcher(other), QUESTION)
    finally:
        other.close()
    assert len(calls) == 1, "another opening of the same store's file: read once all the same"
    _write(tmp_path / "src", {"pkg/more.py": FILES["pkg/other.py"].replace("again", "more")})
    Indexer(store).index("oo", tmp_path / "src")
    again = traits.rows(searcher, QUESTION)
    assert len(calls) == 2, "the generation moved: read again"
    assert _row(first, "pkg/gate.py")["links"] == pytest.approx(math.log1p(3))
    assert _row(again, "pkg/gate.py")["links"] == pytest.approx(math.log1p(4)), "the new namer is counted"


def test_tr6b_another_stores_bytes_in_the_same_file_are_another_store(tmp_path):
    def made(path, src, files):
        held = Store(path, sqlite3.connect)
        Indexer(held).index("oo", _write(src, files))
        Indexer(held).embed(Embedder(A))
        return held

    other = {k.replace("pkg/", "zzz/").replace("gate", "porte"): v for k, v in FILES.items()}
    first, second = made(tmp_path / "a.myc", tmp_path / "srca", FILES), made(tmp_path / "b.myc", tmp_path / "srcb", other)
    assert first.generation() == second.generation(), "the same generation: the trap the cache must not fall into"
    seen = traits.rows(Searcher(first), QUESTION)
    first.close()
    second.close()
    inode = (tmp_path / "a.myc").stat().st_ino
    data = (tmp_path / "b.myc").read_bytes()
    with open(tmp_path / "a.myc", "r+b") as handle:
        handle.truncate(0)
        handle.write(data)
    assert (tmp_path / "a.myc").stat().st_ino == inode
    again = Store(tmp_path / "a.myc", sqlite3.connect)
    try:
        found = traits.rows(Searcher(again), QUESTION)
    finally:
        again.close()
    assert ("oo", "pkg/gate.py") in seen.files and not any(p.startswith("pkg/") for _s, p in found.files), found.files
    (tmp_path / "a.myc").unlink()
    elsewhere = made(tmp_path / "c.myc", tmp_path / "srcc", FILES)
    try:
        traits.rows(Searcher(elsewhere), QUESTION)
    finally:
        elsewhere.close()
    assert not any(key[0] == str((tmp_path / "a.myc").resolve()) for key in traits._HELD), "a store gone: its tables too"


def test_tr7_the_changelog_says_the_traits():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    added = changelog.split("### Added\n", 1)[1].split("\n### ", 1)[0]
    entry = " ".join(next((e for e in added.split("\n- ") if "TR1-TR9" in e), "").split())
    for said in ("Mycelium", "traits", "candidate files", "value head"):
        assert said in entry, (said, entry[:300])


def test_tr8_a_name_defined_too_often_makes_no_link_and_a_kind_is_a_language(tmp_path):
    files = {f"many/m{n}.py": f"def common_name():\n    return {n}\n" for n in range(search.MAX_DEFINITIONS + 1)}
    files.update({
        "many/caller.py": "from many.m0 import common_name\n\n\ndef select_peels():\n    return common_name()\n",
        "few/defines.py": "def rare_name():\n    return 1\n",
        **{f"six/s{n}.py": f"def bound_name():\n    return {n}\n" for n in range(search.MAX_DEFINITIONS)},
        "six/caller.py": "from six.s0 import bound_name\n\n\ndef select_six():\n    return bound_name()\n",
        "few/caller.py": "from few.defines import rare_name\n\n\ndef select_peels_rare():\n    return rare_name()\n",
        "deploy/select_peels.yml": "select_peels:\n  share: terms\n",
        "web/select_peels.js": "function select_peels() {\n  return 'share of terms';\n}\n",
        "docs/SELECT_PEELS.MD": "# Select peels\n\nThe share of terms.\n",
    })
    held = Store(tmp_path / "many.myc", sqlite3.connect)
    try:
        Indexer(held).index("oo", _write(tmp_path / "many_src", files))
        Indexer(held).embed(Embedder(A))
        tables = traits._held(Searcher(held))
        links = {tables["named"][f][1]: n for f, n in tables["links"].items()}
        assert links.get("few/defines.py") == 1, "a name defined once makes its link"
        assert not any(path.startswith("many/m") for path in links), links
        assert all(links.get(f"six/s{n}.py") == 1 for n in range(search.MAX_DEFINITIONS)), links
        found = traits.rows(Searcher(held), QUESTION + " select peels")
        for path, kind in (("deploy/select_peels.yml", "yaml"), ("web/select_peels.js", "ts"),
                           ("docs/SELECT_PEELS.MD", "md")):
            assert _row(found, path)[f"kind_{kind}"] == 1.0, (path, kind)
    finally:
        held.close()


def test_tr9_the_tables_are_one_reading_of_the_store(tmp_path):
    import threading

    path = tmp_path / "c.myc"
    made = Store(path, sqlite3.connect)
    Indexer(made).index("oo", _write(tmp_path / "src", FILES))
    Indexer(made).embed(Embedder(A))
    made.close()
    reader = Store(path, sqlite3.connect)
    committed, failed = threading.Event(), []

    def a_pass():
        writer = Store(path, sqlite3.connect)
        try:
            _write(tmp_path / "src", {"pkg/late.py": "def brand_new_name():\n    return 1\n"})
            Indexer(writer).index("oo", tmp_path / "src")
        except Exception as exc:  # noqa: BLE001 - said below
            failed.append(exc)
        finally:
            writer.close()
            committed.set()

    class Between:
        """The reader's connection, a pass of another connection let in just before the symbols are read."""

        def __init__(self, db):
            self.db, self.worker = db, None

        def execute(self, sql, *args):
            if self.worker is None and sql.startswith("SELECT y.name, y.part FROM symbol y"):
                self.worker = threading.Thread(target=a_pass)
                self.worker.start()
                committed.wait(2.0)
            return self.db.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self.db, name)

    between = Between(reader.db)
    reader.db = between
    try:
        found = traits.rows(Searcher(reader), QUESTION)
    finally:
        reader.db = between.db
    assert between.worker is not None, "the pass was let in"
    between.worker.join(10.0)
    try:
        assert found.files and not failed, failed
        assert ("oo", "pkg/late.py") not in found.files, "the tables were read before the pass, whole"
        assert any(p == "pkg/late.py" for _s, p in traits.rows(Searcher(reader), "brand_new_name").files)
    finally:
        reader.close()
