"""Contracts written from an independent review of the Mycelium (each one red on the code it reviewed), for the
modules of its first block: the cutter, the store, the indexer and the search. The review's contracts on the vectors,
the map and the tool come with the blocks that bring them.

MX1 a source marked again with another sensitivity takes it at once
MX3 a file with Windows line ends is fresh and read again as the index read it
MX5 a source of documents is cut again when the cutter changes
MX6 a partial pass over documents leaves the unchanged ones readable
MX7 one file that cannot be read, named in bytes that are not UTF-8, or too deep for the parser never stops a pass
MX8 a pass that fails leaves nothing of itself behind
MX9 a walk skips dotfiles and key files; a root inside a repository is listed from the repository's index
MX10 a citation is read only from a regular file of bounded size: never a FIFO, never a file grown past the bound
MX12 a worktree whose git directory is gone is refused by name, never read as empty
MX14 a store whose schema cannot be read is refused, never stamped

From the review of block 234 (5 November):
MX16 a source indexed again without naming a sensitivity keeps the one it has
MX17 a root that cannot be read never empties its source, and a folder that cannot be read keeps its files
MX18 one pass at a time on a store: another pass waits, and a pass that fails leaves nothing another committed
MX19 a reader of documents is a mapping, not a chain; a new indexer's partial pass leaves the others unchecked
MX20 a partial pass never marks the source cut by a new cutter
MX21 a file is read for a pass as a citation is: never through a symbolic link put in its place
MX22 the paths a caller names go through the listing's filters
MX23 credentials, client secrets, Terraform states, password files and a key folder's files are never listed
MX24 a database that is not a Mycelium store is refused, never stamped
MX25 the sensitivity filter comes before the limit, in both stages: a secret source neither hides nor shows a part
MX26 a removed file's words are gone from the store's bytes
MX27 an episode holds counts and the meta table closed keys: neither takes a source's text
MX28 a document that cannot be kept is skipped; a pass that fails leaves the reader it found
MX29 a folder source's root is kept absolute
MX30 a file rewritten at the same size within the clock's tick, its time put back, is read again

From the second review of block 234 (5 November):
MX31 a listing that fails, a root gone under named paths, or a folder found empty where the source held files never
     empties the source
MX32 the merge after a pass that removed files never undoes the pass: a merge that fails is said, the pass stands
MX33 a pass that could not read a folder never marks the source cut by its cutter
MX34 a tracked file whose folder cannot be entered is kept, not removed
MX35 a source whose first pass was partial is marked cut, so its unchanged documents are not cut again and again
MX36 a source's row changed while a pass runs waits for it: the pass's half-done writes are never committed
MX37 a file is never read through a folder swapped for a link
"""

import os
import sqlite3
import subprocess
import threading
from pathlib import Path

import pytest

from opti_oignon.mycelium import index as index_module
from opti_oignon.mycelium.index import MAX_BYTES, Indexer, listing
from opti_oignon.mycelium.search import Searcher, source_lines, verify
from opti_oignon.mycelium.store import Store, StoreError
from opti_oignon.mycelium.tracked import IndexUnreadable, tracked_files

# Each contract's time on this machine, a margin above what it measured (all under 0.05 s on 5 November).
BUDGET_S = {
    "test_mx1_a_source_marked_again_takes_its_new_sensitivity_at_once": 1.5,
    "test_mx3_a_file_with_windows_line_ends_is_fresh_and_readable": 1.5,
    "test_mx5_a_source_of_documents_is_cut_again_when_the_cutter_changes": 1.5,
    "test_mx6_a_partial_pass_over_documents_leaves_the_unchanged_ones_readable": 1.5,
    "test_mx7_one_bad_file_never_stops_a_pass": 1.5,
    "test_mx8_a_pass_that_fails_leaves_nothing_of_itself_behind": 1.5,
    "test_mx9_a_walk_skips_secrets_and_a_root_inside_a_repository_uses_its_index": 3.0,
    "test_mx10_a_citation_is_read_only_from_a_bounded_regular_file": 1.5,
    "test_mx12_a_worktree_whose_git_directory_is_gone_is_refused": 3.0,
    "test_mx14_a_store_whose_schema_cannot_be_read_is_refused_never_stamped": 1.5,
    "test_mx16_a_source_indexed_again_keeps_its_sensitivity": 1.5,
    "test_mx17_a_root_that_cannot_be_read_never_empties_its_source": 1.5,
    "test_mx18_one_pass_at_a_time_and_a_failed_pass_leaves_nothing_committed": 6.0,
    "test_mx19_a_reader_of_documents_is_a_mapping_and_a_new_indexer_leaves_others_unchecked": 8.0,
    "test_mx20_a_partial_pass_never_marks_the_source_cut_by_a_new_cutter": 1.5,
    "test_mx21_a_file_is_never_read_through_a_link_put_in_its_place": 1.5,
    "test_mx22_named_paths_go_through_the_listings_filters": 1.5,
    "test_mx23_more_secret_files_are_never_listed": 3.0,
    "test_mx24_a_database_that_is_not_a_store_is_refused": 1.5,
    "test_mx25_the_sensitivity_filter_comes_before_the_limit": 4.0,
    "test_mx26_a_removed_files_words_leave_the_stores_bytes": 1.5,
    "test_mx27_episodes_hold_counts_and_meta_takes_closed_keys": 1.5,
    "test_mx28_a_bad_document_is_skipped_and_a_failed_pass_keeps_the_reader": 1.5,
    "test_mx29_a_folder_sources_root_is_kept_absolute": 1.5,
    "test_mx30_a_file_rewritten_at_the_same_size_and_time_is_read_again": 1.5,
    "test_mx31_a_listing_that_fails_or_finds_nothing_never_empties_the_source": 1.5,
    "test_mx32_a_merge_that_fails_never_undoes_the_pass": 1.5,
    "test_mx33_a_pass_that_could_not_read_a_folder_never_marks_the_cutter": 1.5,
    "test_mx34_a_tracked_file_whose_folder_cannot_be_entered_is_kept": 3.0,
    "test_mx35_a_source_first_indexed_in_part_is_marked_cut": 1.5,
    "test_mx36_a_sources_row_changed_during_a_pass_waits_for_it": 6.0,
    "test_mx37_a_file_is_never_read_through_a_folder_swapped_for_a_link": 1.5,
}

GATE = '''"""The eviction gate: a summary is checked against its sources before it replaces them."""


def probe_summary(summary, sources):
    """Entity, number and date probes: each one the sources name must be in the summary."""
    missing = [name for name in sources if name not in summary]
    if missing:
        raise ValueError("the summary drops what its sources name")
    return True
'''


KEYS = '''"""The master key's loader."""


def load_master_key(path):
    """Read the master key from its file."""
    with open(path, "rb") as handle:
        held = handle.read()
    kept = held
    return kept
'''


def git(cwd, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                          cwd=cwd, env=env, check=True, capture_output=True).stdout


class Locked:
    """A connection whose first read of the schema finds the file locked."""

    def __init__(self, path):
        self.db, self.first = sqlite3.connect(path), True

    def execute(self, sql, *args):
        if self.first and "FROM meta WHERE key = 'schema'" in sql:
            self.first = False
            raise sqlite3.OperationalError("database is locked")
        return self.db.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self.db, name)


def folder(tmp_path, name, files):
    root = tmp_path / name
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def test_mx1_a_source_marked_again_takes_its_new_sensitivity_at_once(tmp_path):
    """MX1 -- a private corpus indexed again as secret is no longer searched by a caller that may not read secrets,
    and still by one that may. A sensitivity kept from the first pass -> RED. (The corpus card and the vectors'
    cache take it too: their contracts come with the blocks that bring them.)"""
    root = folder(tmp_path, "notes", {"gate.py": GATE})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("notes", root)
    searcher = Searcher(store)
    assert searcher.lexical("probe_summary"), "control: found while private"
    Indexer(store).index("notes", root, sensitivity="secret")
    assert searcher.lexical("probe_summary") == []
    assert Searcher(store, allow=("public", "private", "secret")).lexical("probe_summary")
    store.close()


def test_mx3_a_file_with_windows_line_ends_is_fresh_and_readable(tmp_path):
    """MX3 -- a CRLF file: its citations are fresh and its lines are read again as the index read them. Every CRLF
    file stale for ever -> RED. (Showing its lines through the tool comes with the tool's block.)"""
    root = tmp_path / "src"
    root.mkdir()
    (root / "gate.py").write_bytes(GATE.replace("\n", "\r\n").encode("utf-8"))
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("oo", root)
    hits = Searcher(store).search("probe_summary", stages=("symbols", "lexical"))
    assert hits and all(verify(h) == "fresh" for h in hits)
    assert all(source_lines(h) is not None for h in hits)
    store.close()


def test_mx5_a_source_of_documents_is_cut_again_when_the_cutter_changes(tmp_path, monkeypatch):
    """MX5 -- documents indexed, the cutter's version changed: the same documents are cut again (their parts
    written anew). Documents kept under an old cutter for ever -> RED."""
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    docs = {"a.md": "# A\n\nThe first note.\n", "b.md": "# B\n\nThe second note.\n"}
    Indexer(store).index_documents("cellar", docs)
    monkeypatch.setattr(index_module, "CHUNKER_VERSION", "another")
    counts = Indexer(store).index_documents("cellar", docs)
    assert counts.get("recut") == 2
    store.close()


def test_mx6_a_partial_pass_over_documents_leaves_the_unchanged_ones_readable(tmp_path):
    """MX6 -- two documents indexed, then one changed in a pass that is not complete: the unchanged one still
    verifies fresh and is read. Every unchanged document gone after a partial pass -> RED."""
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    indexer = Indexer(store)
    indexer.index_documents("cellar", {"a.md": "# A\n\nThe probe of the first note.\n", "b.md": "# B\n\nOther.\n"})
    indexer.index_documents("cellar", {"b.md": "# B\n\nChanged.\n"}, complete=False)
    hits = Searcher(store).search("probe first note", stages=("lexical",))
    a = [h for h in hits if h["path"] == "a.md"]
    assert a and verify(a[0], indexer.readers) == "fresh"
    assert source_lines(a[0], indexer.readers) is not None
    store.close()


def test_mx7_one_bad_file_never_stops_a_pass(tmp_path):
    """MX7 -- a file that cannot be read, a file named in bytes that are not UTF-8, and a Python file too deep for
    the parser: the pass indexes the others and counts them as skipped (the deep one is cut as plain text). A pass
    stopped by one file -> RED."""
    root = folder(tmp_path, "src", {"gate.py": GATE, "locked.py": "X = 1\n"})
    deep = "VALUE = " + " + ".join(["1"] * 3000) + "\n"
    (root / "deep.py").write_text(deep, encoding="utf-8")
    os.chmod(root / "locked.py", 0)
    with open(os.path.join(os.fsencode(root), b"bad\xff.py"), "w") as handle:
        handle.write("Y = 2\n")
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    try:
        counts = Indexer(store).index("oo", root)
    finally:
        os.chmod(root / "locked.py", 0o644)
    paths = {r[0] for r in store.db.execute("SELECT path FROM file")}
    assert {"gate.py", "deep.py"} <= paths and "locked.py" not in paths or os.geteuid() == 0
    assert not any("bad" in p for p in paths) and counts["skipped"] >= 1
    store.close()


def test_mx8_a_pass_that_fails_leaves_nothing_of_itself_behind(tmp_path, monkeypatch):
    """MX8 -- two files changed, the pass fails while cutting the second: nothing it did stays (the first file
    keeps its old parts and its old sha), and the next pass cuts both again. A file left with its new sha and no
    parts, 'unchanged' from then on -> RED."""
    root = folder(tmp_path, "src", {"a.py": GATE, "b.py": KEYS})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    indexer = Indexer(store)
    indexer.index("oo", root)
    before = {r[0]: r[1] for r in store.db.execute("SELECT path, sha256 FROM file")}
    (root / "a.py").write_text(GATE.replace("Entity", "Every entity"), encoding="utf-8")
    (root / "b.py").write_text(KEYS.replace("master key", "main key"), encoding="utf-8")
    original = indexer._write_parts

    def failing(file_id, rel, lines):
        if rel == "b.py":
            raise RuntimeError("cut failed")
        return original(file_id, rel, lines)

    monkeypatch.setattr(indexer, "_write_parts", failing)
    with pytest.raises(RuntimeError):
        indexer.index("oo", root)
    store.db.commit()
    assert {r[0]: r[1] for r in store.db.execute("SELECT path, sha256 FROM file")} == before
    monkeypatch.setattr(indexer, "_write_parts", original)
    counts = indexer.index("oo", root)
    assert counts["changed"] == 2
    assert store.db.execute("SELECT COUNT(*) FROM part p JOIN file f ON f.id = p.file WHERE f.path = 'a.py'"
                            ).fetchone()[0] > 0
    store.close()


def test_mx9_a_walk_skips_secrets_and_a_root_inside_a_repository_uses_its_index(tmp_path):
    """MX9 -- a walk lists no dotfile and no key file (.env, .netrc, .pem, .key, .p12); a root that is a folder of
    a repository lists what the repository tracks under it, never an ignored file. A secret file tokenised -> RED."""
    # ``.notes.md`` is a dotfile and no secret: the walk skips it for being one.
    root = folder(tmp_path, "plain", {"gate.py": GATE, ".env": "TOKEN=1\n", ".netrc": "machine x\n",
                                      "id.pem": "-----BEGIN-----\n", "server.key": "k\n", "cert.p12": "p\n",
                                      ".notes.md": "# Notes\n"})
    assert listing(root) == ["gate.py"]
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git is not installed")
    repo = folder(tmp_path, "repo", {"sub/gate.py": GATE, "sub/.gitignore": "secret.txt\n", "top.py": "Z = 1\n"})
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    (repo / "sub" / "secret.txt").write_text("password\n", encoding="utf-8")
    assert sorted(listing(repo / "sub")) == [".gitignore", "gate.py"]


def test_mx10_a_citation_is_read_only_from_a_bounded_regular_file(tmp_path):
    """MX10 -- an indexed file replaced by a FIFO, or grown past the bound: the citation is gone, the search does
    not block and reads nothing of it. A chat turn blocked on a FIFO, or a huge file read whole -> RED."""
    root = folder(tmp_path, "src", {"gate.py": GATE, "keys.py": KEYS})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("oo", root)
    searcher = Searcher(store)
    gate = searcher.search("probe_summary", stages=("symbols",))[0]
    keys = searcher.search("load_master_key", stages=("symbols",))[0]
    (root / "gate.py").unlink()
    os.mkfifo(root / "gate.py")
    with open(root / "keys.py", "a", encoding="utf-8") as handle:
        handle.write("#" * (MAX_BYTES + 10) + "\n")

    found = {}

    def probe():
        found["gate"] = (verify(gate), source_lines(gate))

    reader = threading.Thread(target=probe, daemon=True)
    reader.start()
    reader.join(5)
    if reader.is_alive():
        os.close(os.open(root / "gate.py", os.O_WRONLY | os.O_NONBLOCK))     # let the blocked read go
        reader.join(5)
        pytest.fail("reading a citation blocked on a FIFO")
    assert found["gate"] == ("gone", None)
    assert verify(keys) == "gone" and source_lines(keys) is None
    store.close()


def test_mx12_a_worktree_whose_git_directory_is_gone_is_refused(tmp_path):
    """MX12 -- a ``.git`` file naming a directory that does not exist: refused by name, never an empty listing
    (which would remove every file of the source). -> RED otherwise."""
    root = folder(tmp_path, "wt", {"gate.py": GATE})
    (root / ".git").write_text(f"gitdir: {tmp_path / 'gone' / 'worktrees' / 'wt'}\n", encoding="utf-8")
    with pytest.raises(IndexUnreadable, match="gone|does not exist"):
        tracked_files(root)


def test_mx14_a_store_whose_schema_cannot_be_read_is_refused_never_stamped(tmp_path):
    """MX14 -- a store at a newer schema, read while locked: the open fails by name and the file keeps its schema.
    A locked read taken for a new file, and the store stamped at this code's schema -> RED."""
    path = tmp_path / "s.myc"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("INSERT INTO meta VALUES ('schema', '9')")
    db.commit()
    db.close()
    with pytest.raises((StoreError, sqlite3.OperationalError)):
        Store(path, Locked)
    assert sqlite3.connect(path).execute("SELECT value FROM meta WHERE key = 'schema'").fetchone() == ("9",)


def _rows(store, source=None):
    return sorted(r[0] for r in store.db.execute(
        "SELECT f.path FROM file f JOIN source s ON s.id = f.source WHERE ? IS NULL OR s.name = ?", (source, source)))


def test_mx16_a_source_indexed_again_keeps_its_sensitivity(tmp_path):
    """MX16 -- a secret folder and a secret set of documents indexed again with no sensitivity named stay secret: a
    caller that may not read secrets finds nothing of them. A routine refresh turning a secret source private ->
    RED."""
    root = folder(tmp_path, "vault", {"gate.py": GATE})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("vault", root, sensitivity="secret")
    Indexer(store).index("vault", root)
    Indexer(store).index_documents("notes", {"n.md": "# N\n\nzanzibar inventory.\n"}, sensitivity="secret")
    Indexer(store).index_documents("notes", {"n.md": "# N\n\nzanzibar inventory.\n"})
    assert {s["name"]: s["sensitivity"] for s in store.sources()} == {"vault": "secret", "notes": "secret"}
    assert Searcher(store).lexical("probe_summary") == [] and Searcher(store).lexical("zanzibar") == []
    store.close()


def test_mx17_a_root_that_cannot_be_read_never_empties_its_source(tmp_path):
    """MX17 -- the folder renamed away (a drive unmounted), then indexed again: the pass refuses or removes nothing;
    a subfolder that cannot be read keeps its files. Every file removed because nothing could be listed -> RED."""
    root = folder(tmp_path, "src", {"gate.py": GATE, "keys.py": KEYS})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("oo", root)
    root.rename(tmp_path / "elsewhere")
    try:
        Indexer(store).index("oo", root)
    except Exception:  # noqa: BLE001 - refusing is one right answer; removing everything is the wrong one
        pass
    assert _rows(store, "oo") == ["gate.py", "keys.py"]
    if os.geteuid() != 0:
        sub = folder(tmp_path, "tree", {"top.py": "TOP = 1\n", "inner/deep.py": "DEEP = 2\n"})
        Indexer(store).index("tree", sub)
        (sub / "inner").chmod(0)
        try:
            try:
                Indexer(store).index("tree", sub)
            except Exception:  # noqa: BLE001 - a refusal keeps every file too
                pass
            assert _rows(store, "tree") == ["inner/deep.py", "top.py"]
        finally:
            (sub / "inner").chmod(0o755)
    store.close()


def test_mx18_one_pass_at_a_time_and_a_failed_pass_leaves_nothing_committed(tmp_path):
    """MX18 -- a pass holds the store while it runs: a pass of another source started meanwhile waits, so it never
    commits the first one's half-done writes, and the first one, failing, rolls them all back. A file left with its
    new sha and no parts, unchanged for ever -> RED."""
    root = folder(tmp_path, "src", {"a.py": "A = 1\n", "b.py": "def old_name():\n    return 1\n"})
    # One connection shared by the threads, as the platform opens it (``safe_connect``, check_same_thread off).
    store = Store(tmp_path / "s.myc", lambda path: sqlite3.connect(path, check_same_thread=False))
    Indexer(store).index("oo", root)
    old_sha = store.db.execute("SELECT sha256 FROM file WHERE path = 'b.py'").fetchone()[0]
    (root / "b.py").write_text("def a_new_name():\n    return 22\n", encoding="utf-8")     # another size
    entered, release = threading.Event(), threading.Event()
    first = Indexer(store)
    own_write = first._write_parts

    def paused(file_id, rel, lines):
        if rel == "b.py":
            entered.set()
            release.wait(10)
            raise RuntimeError("the pass fails half-way")
        return own_write(file_id, rel, lines)

    first._write_parts = paused
    failed = []

    def run_first():
        try:
            first.index("oo", root)
        except RuntimeError as exc:
            failed.append(exc)

    a = threading.Thread(target=run_first)
    a.start()
    assert entered.wait(10), "control: the first pass reached b.py"
    b = threading.Thread(target=lambda: Indexer(store).index_documents("cellar", {"d.md": "# D\n\nText.\n"}))
    b.start()
    b.join(0.5)
    waited = b.is_alive()
    release.set()
    a.join(10)
    b.join(10)
    assert waited, "the other pass waited for the first"
    assert failed and not a.is_alive() and not b.is_alive()
    assert store.db.execute("SELECT sha256 FROM file WHERE path = 'b.py'").fetchone()[0] == old_sha
    assert _rows(store, "cellar") == ["d.md"], "the other pass ran once the first had ended"
    store.close()


def test_mx19_a_reader_of_documents_is_a_mapping_and_a_new_indexer_leaves_others_unchecked(tmp_path):
    """MX19 -- a thousand partial passes keep the first document verifiable (the reader is no chain of closures);
    a partial pass made by a new indexer, which never held the whole source, leaves the documents it was not given
    unchecked, never gone. A RecursionError, or a living document reported gone -> RED."""
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    indexer = Indexer(store)
    indexer.index_documents("cellar", {"d0.md": "# D0\n\nThe probe of the first note.\n"})
    for n in range(1, 1050):
        indexer.index_documents("cellar", {f"d{n}.md": f"# D{n}\n\nNote {n}.\n"}, complete=False)
    first = [h for h in Searcher(store).search("probe first note", stages=("lexical",)) if h["path"] == "d0.md"]
    assert first and verify(first[0], indexer.readers) == "fresh"
    second = Indexer(store)
    second.index_documents("cellar", {"d1.md": "# D1\n\nNote one, changed.\n"}, complete=False)
    assert verify(first[0], second.readers) in ("unchecked", "fresh")
    store.close()


def test_mx20_a_partial_pass_never_marks_the_source_cut_by_a_new_cutter(tmp_path, monkeypatch):
    """MX20 -- the cutter changed, then a partial pass of one document: the next complete pass still cuts the
    others again. A partial pass stamping the new cutter, the documents it was not given cut by the old one for
    ever -> RED."""
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    docs = {"a.md": "# A\n\nThe first note.\n", "b.md": "# B\n\nThe second note.\n"}
    indexer = Indexer(store)
    indexer.index_documents("cellar", docs)
    monkeypatch.setattr(index_module, "CHUNKER_VERSION", "another")
    indexer.index_documents("cellar", {"b.md": docs["b.md"]}, complete=False)
    counts = indexer.index_documents("cellar", docs)
    assert counts.get("recut", 0) >= 1, counts
    store.close()


def test_mx21_a_file_is_never_read_through_a_link_put_in_its_place(tmp_path, monkeypatch):
    """MX21 -- a file swapped for a symbolic link to a file outside the source, after the pass checked it and
    before it read it: nothing of the outside file is indexed. The outside file's words in the index -> RED."""
    outside = tmp_path / "outside.txt"
    outside.write_text("zanzibar ledger\n", encoding="utf-8")
    root = folder(tmp_path, "src", {"a.py": "A = 1\n", "b.py": "B = 2\n"})
    real_inside = index_module.inside
    swapped, calls = [], []

    def inside(base, full):
        ok = real_inside(base, full)
        if ok and Path(full).name == "a.py":
            calls.append(full)
            if len(calls) == 2 and not swapped:     # after the read's own check, before it opens the file
                os.remove(full)
                os.symlink(outside, full)
                swapped.append(full)
        return ok

    monkeypatch.setattr(index_module, "inside", inside)
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("oo", root)
    assert swapped, "control: the file was swapped"
    assert Searcher(store, allow=("public", "private", "secret")).lexical("zanzibar") == []
    store.close()


def test_mx22_named_paths_go_through_the_listings_filters(tmp_path):
    """MX22 -- a pass over named paths keeps only what a listing would: no secret, no data folder, no key file.
    A secret file indexed because its path was named -> RED."""
    root = folder(tmp_path, "src", {"a.py": "A = 1\n", ".env": "TOKEN=1\n", "data/n.md": "# N\n", "k.pem": "x\n"})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("oo", root, paths=[".env", "data/n.md", "k.pem", "a.py"])
    assert _rows(store, "oo") == ["a.py"]
    store.close()


def test_mx23_more_secret_files_are_never_listed(tmp_path):
    """MX23 -- cloud credentials, an OAuth client secret, Terraform states, an old SSH key, a password file and the
    files of a key folder are never listed, walked or tracked. One of them tokenised -> RED."""
    names = {"gate.py": GATE, "credentials.json": "{}\n", "client_secret_123.json": "{}\n",
             "prod.tfstate": "{}\n", "prod.tfstate.backup": "{}\n", "id_dsa": "k\n", "secrets.yml": "a: 1\n",
             "secrets.yaml": "a: 1\n", "prod.tfvars": "token = 1\n"}
    assert listing(folder(tmp_path, "plain", names)) == ["gate.py"]
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git is not installed")
    repo = folder(tmp_path, "repo", {"gate.py": GATE, ".envrc": "export X=1\n", ".htpasswd": "u:x\n",
                                     ".aws/credentials": "[default]\n", ".ssh/config": "Host x\n",
                                     ".kube/config": "token: x\n", ".docker/config.json": "{}\n"})
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    assert listing(repo) == ["gate.py"]


def test_mx24_a_database_that_is_not_a_store_is_refused(tmp_path):
    """MX24 -- a SQLite file of another kind (tables, no meta) opened as a store: refused by name, its tables
    untouched. The store's tables written into another program's database -> RED."""
    path = tmp_path / "other.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE facts(x)")
    db.commit()
    db.close()
    with pytest.raises(StoreError):
        Store(path, sqlite3.connect)
    tables = [r[0] for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    assert tables == ["facts"]
    browser = tmp_path / "browser.db"         # its own meta(key, value), as a browser's databases have
    db = sqlite3.connect(browser)
    db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    db.execute("INSERT INTO meta VALUES ('version', '1')")
    db.execute("CREATE TABLE urls(x)")
    db.commit()
    db.close()
    with pytest.raises(StoreError):
        Store(browser, sqlite3.connect)
    assert sorted(r[0] for r in sqlite3.connect(browser).execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")) == ["meta", "urls"]


def test_mx25_the_sensitivity_filter_comes_before_the_limit(tmp_path):
    """MX25 -- three hundred secret parts that match better, and one private part: a caller that may not read
    secrets still finds the private one; a name defined in a secret source neither shows it nor, counted, hides a
    private definition of the same name. A private match lost behind secret ones, or a secret definition found by
    its name -> RED."""
    secret = folder(tmp_path, "vault", {f"s{n}.md": "# S\n\nalpha alpha alpha alpha.\n" for n in range(300)}
                    | {"keys.py": "def vault_key_loader():\n    return 1\n"}
                    | {f"t{n}.py": "def shared_name():\n    return 1\n" for n in range(7)})
    plain = folder(tmp_path, "notes", {"p.md": "# P\n\nalpha once, among many other words here.\n",
                                       "q.py": "def shared_name():\n    return 2\n"})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("vault", secret, sensitivity="secret")
    Indexer(store).index("notes", plain)
    searcher = Searcher(store)
    paths = {store.db.execute("SELECT f.path FROM part p JOIN file f ON f.id = p.file WHERE p.id = ?",
                              (i,)).fetchone()[0] for i, _s in searcher.lexical("alpha")}
    assert paths == {"p.md"}
    assert searcher.search("vault_key_loader", stages=("symbols",)) == []
    assert [h["path"] for h in searcher.search("shared_name", stages=("symbols",))] == ["q.py"]
    store.close()


def test_mx26_a_removed_files_words_leave_the_stores_bytes(tmp_path):
    """MX26 -- a file holding a rare word, indexed, then removed: once the pass is committed and the store closed,
    the word is nowhere in the store's file. A removed file's words left in the pages -> RED."""
    root = folder(tmp_path, "src", {"a.py": "A = 1\n", "rare.md": "# Rare\n\nquizzaciously quizzacious.\n"})
    path = tmp_path / "s.myc"
    store = Store(path, sqlite3.connect)
    Indexer(store).index("oo", root)
    assert b"quizzac" in path.read_bytes(), "control: the word was indexed"
    (root / "rare.md").unlink()
    Indexer(store).index("oo", root)
    store.close()
    assert b"quizzac" not in path.read_bytes()


def test_mx27_episodes_hold_counts_and_meta_takes_closed_keys(tmp_path):
    """MX27 -- an episode's detail is counts (numbers), a meta key one the store knows: text handed to either is
    refused. A source's text written through the quiet writes -> RED."""
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    with pytest.raises(StoreError):
        store.episode("index", None, {"note": "the vault's password is zanzibar"})
    with pytest.raises(StoreError):
        store.meta("note", "the vault's password is zanzibar")
    store.meta("cutter:1", 3)
    assert store.meta("cutter:1") == "3"
    store.close()


def test_mx28_a_bad_document_is_skipped_and_a_failed_pass_keeps_the_reader(tmp_path):
    """MX28 -- a document whose text cannot be encoded is skipped, the others kept; a pass that fails half-way leaves
    the reader of the pass before it, so the documents still verify fresh. One bad text stopping the pass, or a
    reader of a pass that never landed -> RED."""
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    indexer = Indexer(store)
    counts = indexer.index_documents("cellar", {"a.md": "# A\n\nThe probe of the first note.\n",
                                                "b.md": "x\udcff", 7: "seven"})
    assert counts.get("skipped") == 2 and _rows(store, "cellar") == ["a.md"]
    hit = [h for h in Searcher(store).search("probe first note", stages=("lexical",)) if h["path"] == "a.md"][0]
    own_write = indexer._write_parts

    def failing(file_id, rel, lines):
        raise RuntimeError("the pass fails")

    indexer._write_parts = failing
    with pytest.raises(RuntimeError):
        indexer.index_documents("cellar", {"a.md": "# A\n\nChanged.\n"})
    indexer._write_parts = own_write
    assert verify(hit, indexer.readers) == "fresh"
    store.close()


def test_mx29_a_folder_sources_root_is_kept_absolute(tmp_path, monkeypatch):
    """MX29 -- a folder named by a relative path is kept by its absolute root, so a citation verifies from any
    working directory. A root that means another folder once the directory changes -> RED."""
    folder(tmp_path, "src", {"gate.py": GATE})
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path / "sub")
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("oo", "../src")
    assert Path(store.sources()[0]["root"]).is_absolute() and ".." not in store.sources()[0]["root"]
    monkeypatch.chdir(tmp_path)
    Indexer(store).index("oo", "src")     # the same folder named another way, from another directory
    store.close()


def test_mx30_a_file_rewritten_at_the_same_size_and_time_is_read_again(tmp_path):
    """MX30 -- a file rewritten at the same size right after a pass, its modification time put back: the next pass
    reads it again (a time that is not older than the pass before by RACY_NS proves nothing), and finds its new name.
    The old text kept for ever, the file taken as unchanged -> RED."""
    root = folder(tmp_path, "src", {"b.py": "def old_name():\n    return 1\n"})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("oo", root)
    held = (root / "b.py").stat()
    (root / "b.py").write_text("def new_name():\n    return 2\n", encoding="utf-8")    # the same size
    os.utime(root / "b.py", ns=(held.st_atime_ns, held.st_mtime_ns))
    Indexer(store).index("oo", root)
    searcher = Searcher(store)
    assert searcher.search("new_name", stages=("symbols",)) and not searcher.search("old_name", stages=("symbols",))
    store.close()


def test_mx31_a_listing_that_fails_or_finds_nothing_never_empties_the_source(tmp_path, monkeypatch):
    """MX31 -- the root's own listing failing after its access check (a network root, a policy that denies the
    read), named paths under a root gone, and the root found empty where the source held files (a mount point left
    after the drive is unmounted): each pass refuses or removes nothing. Every file removed -> RED."""
    root = folder(tmp_path, "src", {"gate.py": GATE, "keys.py": KEYS})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("oo", root)
    real_scandir = os.scandir

    def scandir(path="."):
        if os.fspath(path) == str(root):
            raise PermissionError(13, "denied", str(root))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    try:
        Indexer(store).index("oo", root)
    except Exception:  # noqa: BLE001 - a refusal is right
        pass
    monkeypatch.setattr(os, "scandir", real_scandir)
    assert _rows(store, "oo") == ["gate.py", "keys.py"], "a root whose listing failed"
    root.rename(tmp_path / "away")
    try:
        Indexer(store).index("oo", root, paths=["gate.py", "keys.py"])
    except Exception:  # noqa: BLE001
        pass
    assert _rows(store, "oo") == ["gate.py", "keys.py"], "named paths under a root gone"
    (tmp_path / "away").rename(root)
    for name in ("gate.py", "keys.py"):
        (root / name).unlink()
    try:
        Indexer(store).index("oo", root)
    except Exception:  # noqa: BLE001
        pass
    assert _rows(store, "oo") == ["gate.py", "keys.py"], "an empty folder where the source held files"
    store.close()


def test_mx32_a_merge_that_fails_never_undoes_the_pass(tmp_path, monkeypatch):
    """MX32 -- a pass that removed a file and added one, its merge of the index failing (a full disk): the pass
    stands, its removal and its addition kept, and the failure is said. Every pass that removes anything undone for
    ever -> RED."""
    root = folder(tmp_path, "src", {"a.py": "A = 1\n", "b.py": "B = 2\n"})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("oo", root)
    (root / "b.py").unlink()
    (root / "c.py").write_text("C = 3\n", encoding="utf-8")

    def full(self, pages=None):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(Store, "purge", full)
    Indexer(store).index("oo", root)
    assert _rows(store, "oo") == ["a.py", "c.py"]
    store.close()


def test_mx33_a_pass_that_could_not_read_a_folder_never_marks_the_cutter(tmp_path, monkeypatch):
    """MX33 -- the cutter changed, a folder unreadable during the next pass: that pass does not mark the source cut
    by the new cutter, so the pass after it cuts the folder's files again. A file kept under the old cut for ever
    -> RED."""
    if os.geteuid() == 0:
        pytest.skip("root reads any folder")
    sub = folder(tmp_path, "tree", {"top.py": "TOP = 1\n", "inner/deep.py": "DEEP = 2\n"})
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("tree", sub)
    monkeypatch.setattr(index_module, "CHUNKER_VERSION", "another")
    (sub / "inner").chmod(0)
    try:
        try:
            Indexer(store).index("tree", sub)
        except Exception:  # noqa: BLE001 - a refusal leaves the mark too
            pass
    finally:
        (sub / "inner").chmod(0o755)
    counts = Indexer(store).index("tree", sub)
    assert counts.get("recut", 0) >= 1, counts
    store.close()


def test_mx34_a_tracked_file_whose_folder_cannot_be_entered_is_kept(tmp_path):
    """MX34 -- a repository's tracked file under a folder that cannot be entered: its row is kept (it could not be
    looked at), not removed. A file removed because its folder was closed for a moment -> RED."""
    if os.geteuid() == 0:
        pytest.skip("root enters any folder")
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git is not installed")
    repo = folder(tmp_path, "repo", {"top.py": "TOP = 1\n", "inner/deep.py": "DEEP = 2\n"})
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("repo", repo)
    (repo / "inner").chmod(0)
    try:
        Indexer(store).index("repo", repo)
    finally:
        (repo / "inner").chmod(0o755)
    assert _rows(store, "repo") == ["inner/deep.py", "top.py"]
    store.close()


def test_mx35_a_source_first_indexed_in_part_is_marked_cut(tmp_path):
    """MX35 -- a source of documents whose first pass was partial, sent the same document twice: the second pass
    finds it unchanged. Every unchanged document dropped and cut again at each pass -> RED."""
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    indexer = Indexer(store)
    indexer.index_documents("cellar", {"a.md": "# A\n\nThe note.\n"}, complete=False)
    counts = indexer.index_documents("cellar", {"a.md": "# A\n\nThe note.\n"}, complete=False)
    assert counts.get("unchanged") == 1 and counts.get("recut") == 0, counts
    store.close()


def test_mx36_a_sources_row_changed_during_a_pass_waits_for_it(tmp_path):
    """MX36 -- a source marked secret through the store while a pass of it runs: the change waits for the pass, so
    it never commits the pass's half-done writes, which the pass, failing, rolls back. A file left with its new sha
    and no parts -> RED."""
    root = folder(tmp_path, "src", {"a.py": "A = 1\n", "b.py": "def old_name():\n    return 1\n"})
    store = Store(tmp_path / "s.myc", lambda path: sqlite3.connect(path, check_same_thread=False))
    Indexer(store).index("oo", root)
    old_sha = store.db.execute("SELECT sha256 FROM file WHERE path = 'b.py'").fetchone()[0]
    (root / "b.py").write_text("def a_new_name():\n    return 22\n", encoding="utf-8")
    entered, release = threading.Event(), threading.Event()
    first = Indexer(store)
    own_write = first._write_parts

    def paused(file_id, rel, lines):
        if rel == "b.py":
            entered.set()
            release.wait(10)
            raise RuntimeError("the pass fails half-way")
        return own_write(file_id, rel, lines)

    first._write_parts = paused
    failed = []

    def run_first():
        try:
            first.index("oo", root)
        except RuntimeError as exc:
            failed.append(exc)

    a = threading.Thread(target=run_first)
    a.start()
    assert entered.wait(10), "control: the pass reached b.py"
    b = threading.Thread(target=lambda: store.source("oo", root.absolute(), "folder", "secret"))
    b.start()
    b.join(0.5)
    waited = b.is_alive()
    release.set()
    a.join(10)
    b.join(10)
    assert waited and failed
    assert store.db.execute("SELECT sha256 FROM file WHERE path = 'b.py'").fetchone()[0] == old_sha
    assert store.sources()[0]["sensitivity"] == "secret"
    store.close()


def test_mx37_a_file_is_never_read_through_a_folder_swapped_for_a_link(tmp_path, monkeypatch):
    """MX37 -- a folder of the source swapped for a link to a folder outside it, after a file under it was checked
    and before it was opened: nothing of the outside file is indexed. The outside file's words in the index ->
    RED."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.py").write_text("ZANZIBAR = 'ledger'\n", encoding="utf-8")
    root = folder(tmp_path, "src", {"d/a.py": "A = 1\n", "b.py": "B = 2\n"})
    real_inside = index_module.inside
    calls, swapped = [], []

    def inside(base, full):
        ok = real_inside(base, full)
        if ok and Path(full).name == "a.py":
            calls.append(full)
            if len(calls) == 2 and not swapped:     # after the read's own check, before it opens the file
                kept = tmp_path / "kept_d"
                (root / "d").rename(kept)
                os.symlink(outside, root / "d")
                swapped.append(kept)
        return ok

    monkeypatch.setattr(index_module, "inside", inside)
    store = Store(tmp_path / "s.myc", sqlite3.connect)
    Indexer(store).index("oo", root)
    assert swapped, "control: the folder was swapped"
    assert Searcher(store, allow=("public", "private", "secret")).lexical("zanzibar") == []
    store.close()
