"""Contracts of the Mycelium's first layer: the store, the cutter, the full-text index and the symbol table.

MC1 every non-empty line of a module is in a part; its docstring, its head and a class's constants are parts
MC2 a YAML key takes the comment block above it and defines its nested keys; Markdown goes by heading path
MC3 a second pass writes nothing; a changed file alone is cut again; a touched one is not; a renamed one keeps its row
MC4 a removed file, or one that is no longer text, leaves no part, no token, no symbol, no reference
MC5 the store does not copy the sources' text; under SQLCipher it holds neither a SQLite header nor a token
MC6 the words of an identifier find it; an exact name goes through the symbol table; what a part raises finds it
MC7 a citation says fresh, modified or gone, by the sha256 of its own lines
MC8 a secret source is found only when allowed; a source keeps its root; a sensitivity is named
MC9 a newer schema is refused by name; a store cut by another cutter is cut again
MC10 each pass is an episode that says what it did, and the status says what changed without writing
MC11 a repository's files are what git's index tracks, read without a process: versions 2 and 4, a worktree's .git
     file; an untracked file, a submodule, a symbolic link never; a split index or a broken checksum refused by name
MC12 a symbolic link, or a path whose directory resolves outside the source's root, is never indexed
"""

import os
import sqlite3
import subprocess
import time

import pytest

from opti_oignon.mycelium.chunk import cut
from opti_oignon.mycelium.index import Indexer
from opti_oignon.mycelium.lexis import identifiers, question_terms, subwords
from opti_oignon.mycelium.search import Searcher, verify
from opti_oignon.mycelium.store import Store, StoreError
from opti_oignon.mycelium.tracked import IndexUnreadable, tracked_files

# Each contract's time on this machine, a margin above what it measured (all under 0.05 s on 5 November).
BUDGET_S = {
    "test_mc1_every_line_of_a_module_is_in_a_part_and_its_docstring_head_and_class_constants_are_parts": 1.5,
    "test_mc2_a_yaml_key_takes_its_comments_and_markdown_goes_by_heading_path": 1.5,
    "test_mc3_a_second_pass_writes_nothing_and_only_a_changed_file_is_cut_again": 1.5,
    "test_mc4_a_removed_file_and_a_file_no_longer_text_leave_nothing_behind": 1.5,
    "test_mc5_the_store_copies_no_text_and_under_sqlcipher_shows_nothing": 1.5,
    "test_mc6_identifiers_meet_their_words_names_route_exactly_and_messages_find_their_code": 1.5,
    "test_mc7_a_citation_says_fresh_modified_or_gone": 1.5,
    "test_mc8_a_secret_source_is_found_only_when_allowed_and_a_source_keeps_its_root": 1.5,
    "test_mc9_a_newer_schema_is_refused_and_another_cutter_cuts_again": 1.5,
    "test_mc10_each_pass_is_an_episode_and_the_status_writes_nothing": 1.5,
    "test_mc11_a_repositorys_files_are_read_from_its_git_index_without_a_process": 3.0,
    "test_mc12_a_symbolic_link_or_a_path_resolving_outside_the_root_is_never_indexed": 1.5,
}

LIBRARIAN = '''"""The librarian curates the onion.

One burst runs per conversation; a hold keeps anchors in the Cellar."""
import time

from .peels import select_peels

LIMIT = 4


# The burst's own clock: one per conversation.
def run_burst(conversation_id, config):
    """Run one burst."""
    if config["span_turns"] > LIMIT:
        raise RuntimeError("Cannot connect to the engine")
    return select_peels(conversation_id)


class Librarian:
    """Holds the states."""

    timeout_s = 30

    def plan(self):
        return run_burst("x", {})

    RETRIES = (
        "connection refused",
        "broken pipe",
        "reset by peer",
        "timed out",
    )

    def compose(self):
        return self.plan()
'''

PEELS = '''"""Peels: summaries of evicted spans."""


def select_peels(conversation_id):
    """A peel scores the share of the query's terms."""
    chosen = []
    for peel in []:
        chosen.append(peel)
    return chosen
'''

TEST = '''from opti_oignon.memory.librarian import run_burst


def test_burst_refuses_a_long_span():
    limit = 9
    assert run_burst
    assert limit > 4
    assert callable(run_burst)
    assert run_burst is not None
'''

ONION_YAML = '''# Onion memory
# =============

reserve: 2048

# The span of a peel, in turns: four turns leave the Flesh together.
peels:
  span_turns: 4
  floor: 0.5
  reask: true
  anchors: 768
'''

DOC = '''# The onion

It keeps the conversation.

## The librarian

`run_burst` runs one burst; see `select_peels`.
It curates the onion while the person reads.
A burst never runs during an answer.
Its calls are bounded by the governor.
'''

FILES = {"opti_oignon/memory/librarian.py": LIBRARIAN, "opti_oignon/memory/peels.py": PEELS,
         "tests/test_librarian.py": TEST, "opti_oignon/config/onion.yaml": ONION_YAML, "docs/onion.md": DOC}


def _write(root):
    for rel, text in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _later(path):
    os.utime(path, ns=(time.time_ns(), time.time_ns() + 10 ** 9))


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "repo"
    _write(root)
    store = Store(tmp_path / "corpus.myc", sqlite3.connect)
    indexer = Indexer(store)
    counts = indexer.index("oo", root)
    yield {"root": root, "store": store, "indexer": indexer, "counts": counts, "tmp": tmp_path}
    store.close()


def test_mc1_every_line_of_a_module_is_in_a_part_and_its_docstring_head_and_class_constants_are_parts():
    """MC1 -- every non-empty line of a module is in a part; the head opens the parts and defines the module; a
    comment above a definition goes with it; a class's constants between its methods are a part; a docstring long
    enough is a part of its own. A head or a constant left out, or a comment cut away -> RED."""
    lines = LIBRARIAN.split("\n")
    parts = cut("opti_oignon/memory/librarian.py", lines)
    covered = {n for p in parts for n in range(p.first, p.last + 1)}
    assert all(n in covered for n, line in enumerate(lines, 1) if line.strip())
    head = parts[0]
    assert head.first == 1 and "LIMIT = 4" in "\n".join(lines[head.first - 1:head.last])
    assert ("librarian", "opti_oignon.memory.librarian", "module") in head.defines
    burst = next(p for p in parts if p.symbol == "run_burst")
    assert lines[burst.first - 1].startswith("# The burst's own clock")
    retries = next(p for p in parts if p.kind == "code" and p.symbol.startswith("Librarian:"))
    assert "connection refused" in "\n".join(lines[retries.first - 1:retries.last])
    doc = '"""' + "\n".join(f"Line {n} of the design of this module." for n in range(12)) + '\n"""'
    text = doc + "\nimport os\nimport sys\nimport json\nimport time\nimport re\nLIMIT = 4\nNAME = 'x'\n\n\n" \
        + "def f():\n" + "\n".join(f"    a{n} = {n}" for n in range(8)) + "\n    return a0\n"
    long = cut("opti_oignon/thing.py", text.split("\n"))
    assert long[0].kind == "doc" and (long[0].first, long[0].last) == (1, 13) and long[1].kind == "module"


def test_mc2_a_yaml_key_takes_its_comments_and_markdown_goes_by_heading_path():
    """MC2 -- a YAML key's part starts at the comment block above it and defines its nested keys; a Markdown
    section's symbol is its heading path and it names its backticked identifiers. The comment block lost, a
    nested key undefined, a heading path flattened -> RED."""
    lines = ONION_YAML.split("\n")
    peels = next(p for p in cut("opti_oignon/config/onion.yaml", lines) if p.symbol == "onion.peels")
    assert lines[peels.first - 1].startswith("# The span of a peel")
    assert ("span_turns", "onion.peels.span_turns", "setting") in peels.defines
    section = next(p for p in cut("docs/onion.md", DOC.split("\n")) if p.symbol == "The onion > The librarian")
    assert ("run_burst", "mention", "") in section.refs


def test_mc3_a_second_pass_writes_nothing_and_only_a_changed_file_is_cut_again(corpus):
    """MC3 -- a second pass over an unchanged source writes no part; a changed file is cut again alone; a file
    whose times moved but not its content is not cut again; a renamed file keeps its row and is found under its
    new path. Any re-cut beyond the changed file, or a rename treated as a removal and an addition -> RED."""
    assert corpus["counts"]["added"] == 5
    again = corpus["indexer"].index("oo", corpus["root"])
    assert again["unchanged"] == 5 and again["parts"] == 0
    peels = corpus["root"] / "opti_oignon/memory/peels.py"
    peels.write_text(PEELS + "\n\ndef drop_peel(peel):\n    return None\n", encoding="utf-8")
    _later(peels)
    _later(corpus["root"] / "docs/onion.md")
    again = corpus["indexer"].index("oo", corpus["root"])
    assert again["changed"] == 1 and again["touched"] == 1 and again["unchanged"] == 3
    db = corpus["store"].db
    file_id = db.execute("SELECT id FROM file WHERE path = 'docs/onion.md'").fetchone()[0]
    (corpus["root"] / "docs/onion.md").rename(corpus["root"] / "docs/memory.md")
    again = corpus["indexer"].index("oo", corpus["root"])
    assert again["renamed"] == 1 and again["removed"] == 0 and again["added"] == 0
    assert db.execute("SELECT id FROM file WHERE path = 'docs/memory.md'").fetchone()[0] == file_id


def test_mc4_a_removed_file_and_a_file_no_longer_text_leave_nothing_behind(corpus):
    """MC4 -- a file removed, and a file whose bytes are no longer text, leave no part, no row of tokens, no
    symbol and no reference; the index holds one row of tokens per living part. A stale part or token -> RED."""
    (corpus["root"] / "opti_oignon/memory/peels.py").unlink()
    (corpus["root"] / "docs/onion.md").write_bytes(b"\x00binary now\x00")
    again = corpus["indexer"].index("oo", corpus["root"])
    assert again["removed"] == 2
    db = corpus["store"].db
    assert db.execute("SELECT COUNT(*) FROM symbol WHERE name = 'select_peels'").fetchone()[0] == 0
    alive = {r[0] for r in db.execute("SELECT id FROM part")}
    assert db.execute("SELECT COUNT(*) FROM fts").fetchone()[0] == len(alive)
    assert db.execute("SELECT COUNT(*) FROM ref WHERE part NOT IN (SELECT id FROM part)").fetchone()[0] == 0
    found = [i for i, _s in Searcher(corpus["store"]).lexical("peel share query terms conversation onion")]
    assert set(found) <= alive


def test_mc5_the_store_copies_no_text_and_under_sqlcipher_shows_nothing(corpus, tmp_path):
    """MC5 -- the index keeps tokens, never the sources' text (a sentence of a source is not in the file's bytes);
    under SQLCipher the file holds neither the SQLite header nor an identifier of the corpus. Content kept in the
    index, or a plaintext store -> RED."""
    corpus["store"].db.commit()
    raw = (tmp_path / "corpus.myc").read_bytes()
    assert b"A peel scores the share of the query's terms" not in raw
    sqlcipher = pytest.importorskip("sqlcipher3.dbapi2")

    def connect(path):
        db = sqlcipher.connect(path)
        db.execute("PRAGMA key = \"x'" + "ab" * 32 + "'\"")
        return db

    store = Store(tmp_path / "sealed.myc", connect)
    Indexer(store).index("oo", corpus["root"])
    assert Searcher(store).lexical("select_peels")
    store.close()
    sealed = (tmp_path / "sealed.myc").read_bytes()
    assert not sealed.startswith(b"SQLite format 3") and b"select_peels" not in sealed


def test_mc6_identifiers_meet_their_words_names_route_exactly_and_messages_find_their_code(corpus):
    """MC6 -- `get user name` meets `getUserName` and `get_user_name`; an exact identifier goes through the
    symbol table; what a part raises is found by those words. An identifier left whole only, a symbol table not
    asked, or the messages left out of the index -> RED."""
    assert subwords("getUserName get_user_name HTTPServer") == "get User Name get user name HTTP Server"
    assert question_terms("Ou est get_user_name ?")[:2] == ["get_user_name", "get"]
    assert identifiers("why does run_burst call selectPeels") == ["run_burst", "selectPeels"]
    s = Searcher(corpus["store"])
    assert s.search("which function selects the peels", 5, stages=("lexical",))[0]["symbol"] == "select_peels"
    routed = s.search("run_burst", 3, stages=("symbols",))
    assert routed[0]["symbol"] == "run_burst" and routed[0]["why"] == ["symbols#1"]
    assert s.search("cannot connect to the engine", 1, stages=("lexical",))[0]["symbol"] == "run_burst"


def test_mc7_a_citation_says_fresh_modified_or_gone(corpus):
    """MC7 -- a citation is checked against the sha256 of its own lines: fresh, then modified once one of its
    lines changes, gone once the file is not there; a source of documents is checked through its reader, and is
    unchecked (never fresh) without one. A citation said fresh after a change -> RED."""
    hit = Searcher(corpus["store"]).search("select_peels", 1, stages=("symbols",))[0]
    assert verify(hit) == "fresh"
    path = corpus["root"] / hit["path"]
    path.write_text(path.read_text(encoding="utf-8").replace("share of the query's terms", "share"), encoding="utf-8")
    assert verify(hit) == "modified"
    path.unlink()
    assert verify(hit) == "gone"
    corpus["indexer"].index_documents("cellar", {"span-1": "Gina lost her job this week."})
    found = Searcher(corpus["store"]).search("Gina lost her job", 1, stages=("lexical",))[0]
    assert verify(found) == "unchecked" and verify(found, corpus["indexer"].readers) == "fresh"


def test_mc8_a_secret_source_is_found_only_when_allowed_and_a_source_keeps_its_root(corpus, tmp_path):
    """MC8 -- a source marked secret is never in a search that does not allow it; a known source name with
    another root is refused by name; a sensitivity outside the three is refused by name. A secret part in a
    default search -> RED."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Vault\n\nthe zanzibar passphrase lives here\n", encoding="utf-8")
    corpus["indexer"].index("vault", vault, sensitivity="secret")
    assert not Searcher(corpus["store"]).lexical("zanzibar passphrase")
    assert Searcher(corpus["store"], allow=("public", "private", "secret")).lexical("zanzibar passphrase")
    with pytest.raises(StoreError):
        corpus["store"].source("oo", "/elsewhere")
    with pytest.raises(StoreError):
        corpus["store"].source("other", "/x", sensitivity="semi")


def test_mc9_a_newer_schema_is_refused_and_another_cutter_cuts_again(corpus, tmp_path):
    """MC9 -- a store written by a newer schema is refused by name; a store cut by another version of the cutter
    cuts every file again at the next pass, and then no more. A newer store opened, or an old cut kept -> RED."""
    db = corpus["store"].db
    db.execute("UPDATE meta SET value = 'old' WHERE key LIKE 'cutter:%'")
    db.commit()
    assert corpus["indexer"].index("oo", corpus["root"])["recut"] == 5
    assert corpus["indexer"].index("oo", corpus["root"])["recut"] == 0
    db.execute("UPDATE meta SET value = '99' WHERE key = 'schema'")
    db.commit()
    corpus["store"].close()
    with pytest.raises(StoreError):
        Store(tmp_path / "corpus.myc", sqlite3.connect)
    corpus["store"] = Store(tmp_path / "other.myc", sqlite3.connect)


def test_mc10_each_pass_is_an_episode_and_the_status_writes_nothing(corpus):
    """MC10 -- each pass is an episode that says what it added, changed, renamed and removed, and its time; the
    status reads sizes and times alone and changes nothing in the store. An empty episode, or a status that
    writes -> RED."""
    assert corpus["indexer"].status("oo", corpus["root"]) == {"new": 0, "touched": 0, "gone": 0,
                                                              "estimate_seconds": 0.0}
    (corpus["root"] / "docs/new.md").write_text("# New\n\nwords\n", encoding="utf-8")
    (corpus["root"] / "opti_oignon/memory/peels.py").unlink()
    before = corpus["store"].stats()
    said = corpus["indexer"].status("oo", corpus["root"])
    assert said["new"] == 1 and said["gone"] == 1 and said["estimate_seconds"] is not None
    assert corpus["store"].stats() == before
    corpus["indexer"].index("oo", corpus["root"])
    latest, first = corpus["store"].episodes()[:2]
    assert first["detail"]["added"] == 5 and latest["detail"]["added"] == 1 and latest["detail"]["removed"] == 1
    assert latest["detail"]["seconds"] >= 0


def git(cwd, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                          cwd=cwd, env=env, check=True, capture_output=True).stdout


def test_mc11_a_repositorys_files_are_read_from_its_git_index_without_a_process(tmp_path, monkeypatch):
    """MC11 -- in a real repository (git writes the index), the listing is ``git ls-files`` less what is not a
    regular file, at index versions 2 and 4 alike, and through a worktree's ``.git`` file; an untracked file, a
    submodule and a symbolic link are not listed; no process is started. A split index and an index whose checksum
    fails are refused by name. A listing that runs git, guesses from the working tree, or lists a link -> RED."""
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    (repo / "pkg" / "deep").mkdir(parents=True)
    (repo / "pkg" / "mod.py").write_text(PEELS, encoding="utf-8")
    (repo / "pkg" / "deep" / ("long_" + "n" * 70 + ".py")).write_text("X = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("# Title\n\nText.\n", encoding="utf-8")
    git(repo, "init", "-q")
    os.symlink("/etc/hostname", repo / "link.txt")
    sub = tmp_path / "sub"
    sub.mkdir()
    git(sub, "init", "-q")
    (sub / "s.py").write_text("Y = 2\n", encoding="utf-8")
    git(sub, "add", "-A")
    git(sub, "commit", "-q", "-m", "s")
    git(repo, "add", "-A")
    git(repo, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "vendored")
    git(repo, "commit", "-q", "-m", "c")
    (repo / "untracked.py").write_text("Z = 3\n", encoding="utf-8")
    regular = sorted(p for p in git(repo, "ls-files", "-z", "-s").decode().split("\0") if p.startswith("100"))
    want = sorted(line.split("\t", 1)[1] for line in regular)
    assert len(want) == 4 and ".gitmodules" in want and "link.txt" not in want and "vendored" not in want

    def no_process(*_a, **_k):
        raise AssertionError("a process was started")

    for version in ("2", "4"):
        git(repo, "update-index", "--index-version", version)
        with monkeypatch.context() as m:
            m.setattr(subprocess, "run", no_process)
            m.setattr(subprocess, "Popen", no_process)
            assert sorted(tracked_files(repo)) == want
            assert sorted(index_listing(repo)) == want
    git(repo, "worktree", "add", "-q", str(tmp_path / "wt"))
    assert (tmp_path / "wt" / ".git").is_file() and sorted(tracked_files(tmp_path / "wt")) == want
    index = repo / ".git" / "index"
    good = index.read_bytes()
    index.write_bytes(good[:40] + bytes([good[40] ^ 1]) + good[41:])
    with pytest.raises(IndexUnreadable, match="checksum"):
        tracked_files(repo)
    index.write_bytes(good)
    git(repo, "-c", "core.splitIndex=true", "update-index", "--split-index")
    with pytest.raises(IndexUnreadable, match="split in two"):
        tracked_files(repo)


def index_listing(root):
    from opti_oignon.mycelium.index import listing

    return listing(root)


def test_mc12_a_symbolic_link_or_a_path_resolving_outside_the_root_is_never_indexed(tmp_path):
    """MC12 -- a link to a file outside the root, a link inside it, and a tracked path whose directory became a
    link to another folder leave no part; the regular files are indexed. A link followed out of the root -> RED."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text('KEY = "do not index me"\n', encoding="utf-8")
    root = tmp_path / "src"
    root.mkdir()
    (root / "peels.py").write_text(PEELS, encoding="utf-8")
    os.symlink(outside / "secret.py", root / "secret_link.py")
    os.symlink(root / "peels.py", root / "peels_again.py")
    os.symlink(outside, root / "elsewhere")
    store = Store(tmp_path / "c.myc", sqlite3.connect)
    indexer = Indexer(store)
    counts = indexer.index("oo", root, paths=["peels.py", "secret_link.py", "peels_again.py",
                                              "elsewhere/secret.py"])
    paths = {r[0] for r in store.db.execute("SELECT path FROM file")}
    assert paths == {"peels.py"} and counts["skipped"] == 3
    assert not Searcher(store).lexical("do not index me KEY")
    counts = indexer.index("walked", root)
    assert {r[0] for r in store.db.execute("SELECT f.path FROM file f JOIN source s ON s.id = f.source "
                                           "WHERE s.name = 'walked'")} == {"peels.py"}
    store.close()
