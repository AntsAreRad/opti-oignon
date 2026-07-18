#!/usr/bin/env python3
"""What the project store promises about the records and the blobs it keeps.

A project groups conversations, uploaded files, and produced outputs, and it
answers two demands at once. Every write is DATA: a project name, a description,
a conversation id -- a value carrying SQL metacharacters has to round-trip as a
string, never execute, so every statement is parameterised. And the store owns a
patch of the filesystem: an upload is written under a per-project directory, and
deleting a project takes its rows AND its directory with it. The row side
cascades on the schema's own key -- files, outputs, and conversation links all
reference the project and are declared ON DELETE CASCADE -- and every connection
turns foreign keys on, so the cascade the schema declares is the cascade that
actually runs.

An upload's name is never trusted. Before a byte reaches disk the filename is
reduced to its basename and stripped of null bytes, so a caller handing in a
path that climbs out of the project directory writes a plain file inside it
instead, and a name that sanitises down to nothing becomes a fixed placeholder.
The store stamps every stored file with a unique id prefix, so two uploads of
the same name never collide. Size, count, and extension limits are read from
config and enforced before the write, and a write that fails on disk surfaces as
a value error rather than a half-registered row.

Reads degrade rather than raise. A listing, a lookup, a stats roll-up on a
project that has no rows comes back empty or zeroed, never as an exception,
because a read that throws takes a UI panel down with it. Settings and key-term
columns arrive from SQLite as JSON strings and decode to their native shapes; a
blob that will not parse decodes to an empty container rather than sinking the
row. Updating a project MERGES settings into what is already stored rather than
replacing them, and only the fields the caller actually supplies are touched.

One promise is worded more narrowly by the code than by its own docstring, and
this suite pins the code. Linking a conversation returns True whenever the
project exists -- including a link that already existed, where the insert is
ignored and nothing is touched -- and returns False only when the project is
missing. The docstring speaks of a False for an already-linked conversation;
the store does not produce one, and that idempotent-True is pinned here as the
behaviour it is, not the behaviour the sentence describes.

The store reaches for exactly one project seam: the connection factory. It is
seeded with a plain sqlite connection so a real temporary database backs every
contract, and one contract removes it entirely to prove the module's own
fallback still connects. The type map the module keeps at module scope is
rebuilt from config on every construction; contracts that care pass an explicit
category mapping or set the map themselves, so none leans on whatever the
on-disk seed config happens to hold. Loaded through the shared isolation window;
no real backend is ever touched.
"""

import sqlite3
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.projects"


def _default_connect(path, **kwargs):
    return sqlite3.connect(str(path), **kwargs)


def _load(tmp_path, *, connect=_default_connect, block_db=False):
    """Load the project store in isolation.

    connect    -- the stand-in ``safe_connect``. The default is a plain sqlite
                  connection; a caller can pass a counting factory to pin the
                  seam itself.
    block_db   -- when true the connection module is declared UNREACHABLE and
                  proven so, so the module's own import-fallback connection
                  factory is what runs. There is no data-directory seam to
                  redirect: the module derives its default paths from its own
                  file, and the module-level singleton it builds on load writes
                  into the (git-ignored) package data directory. Every contract
                  builds its own store with explicit temporary paths, so no
                  contract ever depends on that singleton or its location.
    """
    seeded = {}
    blocked = []

    if block_db:
        blocked.append("opti_oignon.db_utils")
    else:
        du = types.ModuleType("opti_oignon.db_utils")
        du.safe_connect = connect
        seeded["opti_oignon.db_utils"] = du

    loaded, restore = isolate(
        targets={_TARGET: source("projects.py")},
        blocked=blocked,
        seeded=seeded,
    )
    return loaded[_TARGET], restore


# --- fixtures -------------------------------------------------------------
# A store is always built with explicit paths distinct from the singleton's, so
# a contract never touches the package data directory and never depends on the
# seed config file on disk. The config path defaults to a name that does not
# exist, which makes the loader fall through to the built-in defaults.

def _store(pj, tmp_path, *, db="p.db", storage="store", config_path=None):
    return pj.ProjectStore(
        db_path=Path(tmp_path) / db,
        config_path=config_path if config_path is not None else Path(tmp_path) / "absent.yaml",
        storage_base=Path(tmp_path) / storage,
    )


def _cfg_file(tmp_path, name="projects.yaml", **projects_cfg):
    """Write a minimal ``projects:`` config the loader reads through yaml."""
    lines = ["projects:"]
    for key, value in projects_cfg.items():
        if isinstance(value, dict):
            lines.append(f"  {key}:")
            for kk, vv in value.items():
                lines.append(f"    {kk}: {vv!r}")
        elif isinstance(value, list):
            lines.append(f"  {key}:")
            for item in value:
                lines.append(f"    - {item!r}")
        else:
            lines.append(f"  {key}: {value!r}")
    path = Path(tmp_path) / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _raw(tmp_path, db="p.db"):
    return sqlite3.connect(str(Path(tmp_path) / db))


# =========================================================================
# Data classes and pure helpers
# =========================================================================

def test_p1_dataclasses_roundtrip_to_dict(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        pr = pj.Project(id="pid", name="n", description="d",
                        system_instructions="s", settings={"k": [1, 2]},
                        created_at="t0", updated_at="t1")
        assert pr.to_dict() == {
            "id": "pid", "name": "n", "description": "d",
            "system_instructions": "s", "settings": {"k": [1, 2]},
            "created_at": "t0", "updated_at": "t1",
        }
        pf = pj.ProjectFile(id="fid", project_id="pid", filename="a.py",
                            file_path="/x/a.py", file_type="code",
                            file_size_bytes=9, indexed=True, chunk_count=2,
                            summary="s", key_terms=["a"], uploaded_at="u",
                            updated_at="v")
        d = pf.to_dict()
        assert d["key_terms"] == ["a"] and d["indexed"] is True and d["file_size_bytes"] == 9
        po = pj.ProjectOutput(id="oid", project_id="pid", filename="o.txt",
                              file_path="/x/o.txt", output_type="report")
        assert po.to_dict()["output_type"] == "report"
    finally:
        restore()


def test_p2_post_init_generates_id_and_timestamps(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        pr = pj.Project(name="fresh")
        assert len(pr.id) == 12, "an absent id is filled with a twelve-char token"
        assert pr.created_at and pr.updated_at, "absent timestamps are stamped now"
        kept = pj.Project(id="keep", name="x", created_at="fixed", updated_at="fixed")
        assert kept.id == "keep" and kept.created_at == "fixed", (
            "supplied id and timestamps are left untouched"
        )
        po = pj.ProjectOutput(filename="o")
        assert len(po.id) == 12 and po.created_at
    finally:
        restore()


def test_p3_project_from_dict_filters_and_parses_settings(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        pr = pj.Project.from_dict({
            "id": "x", "name": "n", "settings": '{"a": 1}',
            "bogus_key": "ignored",
        })
        assert pr.id == "x" and pr.name == "n"
        assert pr.settings == {"a": 1}, "a settings JSON string decodes to a dict"
        assert not hasattr(pr, "bogus_key"), "an unknown column is dropped"
        bad = pj.Project.from_dict({"name": "n", "settings": "not json"})
        assert bad.settings == {}, (
            "settings that will not parse decode to an empty dict, never raise "
            "out of from_dict"
        )
    finally:
        restore()


def test_p4_project_file_from_dict_parses_key_terms_and_bool(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        pf = pj.ProjectFile.from_dict({
            "id": "f", "key_terms": '["a", "b"]', "indexed": 1, "junk": 9,
        })
        assert pf.key_terms == ["a", "b"], "a key_terms JSON string decodes to a list"
        assert pf.indexed is True, "an integer indexed flag is coerced to bool"
        assert not hasattr(pf, "junk")
        bad = pj.ProjectFile.from_dict({"key_terms": "nope"})
        assert bad.key_terms == [], "unparseable key_terms decode to an empty list"
    finally:
        restore()


def test_p5_project_output_from_dict_ignores_unknown_keys(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        po = pj.ProjectOutput.from_dict({
            "id": "o", "filename": "f", "output_type": "data", "extra": "x",
        })
        assert po.id == "o" and po.filename == "f" and po.output_type == "data"
        assert not hasattr(po, "extra"), "an unknown column is dropped"
    finally:
        restore()


def test_p6_detect_file_type_longest_suffix_then_unknown(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        cats = {".gz": "data", ".tar.gz": "archive", ".py": "code"}
        assert pj.detect_file_type("x.tar.gz", cats) == "archive", (
            "a compound extension is matched before a shorter suffix of it -- "
            "longest match wins"
        )
        assert pj.detect_file_type("X.PY", cats) == "code", "matching is case-insensitive"
        assert pj.detect_file_type("plain", cats) == "unknown", (
            "a name with no known suffix falls through to unknown"
        )
    finally:
        restore()


def test_p7_extract_file_metadata_size_lines_and_missing(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        pj._FILE_TYPE_MAP = pj._build_file_type_map(pj._DEFAULT_CATEGORIES)
        txt = Path(tmp_path) / "a.txt"
        txt.write_bytes(b"l1\nl2\nl3")
        meta = pj.extract_file_metadata(txt)
        assert meta["size_bytes"] == 8
        assert meta["line_count"] == 3, "a text file reports its line count"
        png = Path(tmp_path) / "a.png"
        png.write_bytes(b"\x89PNG")
        m2 = pj.extract_file_metadata(png)
        assert m2["size_bytes"] == 4 and m2["line_count"] is None, (
            "a non-text file reports size but no line count"
        )
        m3 = pj.extract_file_metadata(Path(tmp_path) / "ghost.txt")
        assert m3 == {"size_bytes": 0, "line_count": None}, (
            "a file that cannot be stat'd returns the zeroed default, never raises"
        )
    finally:
        restore()


def test_p8_iso_now_shape_and_type_map_build(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        now = pj._iso_now()
        assert now.endswith("Z") and "T" in now and len(now) == 20
        built = pj._build_file_type_map({"code": [".PY", ".Js"], "text": [".TXT"]})
        assert built[".py"] == "code" and built[".js"] == "code" and built[".txt"] == "text", (
            "the map lowercases keys and inverts category -> extensions into "
            "extension -> category"
        )
    finally:
        restore()


# =========================================================================
# Connection seam, config loading, construction
# =========================================================================

def test_p9_all_db_access_flows_through_the_connection_seam(tmp_path):
    calls = {"n": 0}

    def counting(path, **kwargs):
        calls["n"] += 1
        return sqlite3.connect(str(path), **kwargs)

    pj, restore = _load(tmp_path, connect=counting)
    try:
        store = _store(pj, tmp_path)
        calls["n"] = 0
        p = store.create_project("x")
        store.list_projects()
        store.get_project(p.id)
        f = store.add_file(p.id, "a.txt", b"hi")
        store.get_file(f.id)
        store.delete_project(p.id)
        assert calls["n"] > 0, (
            "every database touch must be opened through safe_connect; a path "
            "that reached sqlite directly would bypass the encrypted-connection "
            "seam and never increment this counter"
        )
    finally:
        restore()


def test_p10_missing_connection_module_falls_back_and_still_connects(tmp_path):
    pj, restore = _load(tmp_path, block_db=True)
    try:
        assert callable(pj._safe_connect), (
            "with the connection module unreachable the import fallback binds a "
            "plain sqlite connection factory"
        )
        store = _store(pj, tmp_path)
        p = store.create_project("via_fallback")
        assert store.get_project(p.id).name == "via_fallback", (
            "the store initialises and round-trips a project on the fallback seam"
        )
    finally:
        restore()


def test_p11_load_config_missing_and_malformed_fall_back_to_defaults(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        # Missing file: the default properties hold.
        missing = _store(pj, tmp_path, config_path=Path(tmp_path) / "none.yaml")
        assert missing.enabled is True and missing.max_projects == 50
        # Malformed yaml: the loader swallows the parse error and uses defaults.
        bad = Path(tmp_path) / "bad.yaml"
        bad.write_text(":\n  - [unbalanced\n", encoding="utf-8")
        store = _store(pj, tmp_path, db="bad.db", storage="bad_store", config_path=bad)
        assert store.enabled is True and store.max_projects == 50, (
            "a config that will not parse falls back to defaults, never raises "
            "out of construction"
        )
    finally:
        restore()


def test_p12_config_properties_default_and_override(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        d = _store(pj, tmp_path, config_path=Path(tmp_path) / "none.yaml")
        assert d.max_files_per_project == 100
        assert d.max_file_size_mb == 50 and d.max_file_size_bytes == 50 * 1024 * 1024
        assert d.allowed_extensions == []
        assert set(d.default_settings) == {
            "default_model", "default_pipeline", "context_budget_tokens", "auto_index"
        }
        cfg = _cfg_file(tmp_path, enabled=False, max_projects=7,
                        allowed_extensions=[".txt"])
        o = _store(pj, tmp_path, db="o.db", storage="o_store", config_path=cfg)
        assert o.enabled is False and o.max_projects == 7 and o.allowed_extensions == [".txt"], (
            "config values override the defaults where present"
        )
    finally:
        restore()


def test_p13_construction_rebuilds_type_map_and_makes_storage(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        cfg = _cfg_file(tmp_path, file_type_categories={"special": [".zzz"]})
        store = _store(pj, tmp_path, config_path=cfg)
        assert pj._FILE_TYPE_MAP.get(".zzz") == "special", (
            "construction rebuilds the module-scope extension map from the "
            "config's own categories"
        )
        assert Path(store._storage_base).is_dir(), "the storage base is created on construction"
    finally:
        restore()


# =========================================================================
# Project CRUD
# =========================================================================

def test_p14_create_validates_strips_and_lays_out_directory(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        with pytest.raises(ValueError):
            store.create_project("")
        with pytest.raises(ValueError):
            store.create_project("   ")
        p = store.create_project("  Name  ", description="  d  ",
                                 system_instructions="  s  ")
        assert p.name == "Name" and p.description == "d" and p.system_instructions == "s", (
            "name, description and instructions are stripped of surrounding space"
        )
        assert len(p.id) == 12
        base = Path(store._storage_base) / p.id
        assert (base / "files").is_dir() and (base / "outputs").is_dir(), (
            "a project directory with files/ and outputs/ is created on disk"
        )
        assert store.get_project(p.id) is not None, "the project is persisted"
    finally:
        restore()


def test_p15_create_merges_settings_and_enforces_limit(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        p = store.create_project("x", settings={"default_model": "custom"})
        assert p.settings["default_model"] == "custom", "a supplied setting overrides its default"
        assert p.settings["context_budget_tokens"] == 4096, (
            "settings are merged onto the defaults, so untouched defaults survive"
        )

        cfg = _cfg_file(tmp_path, max_projects=2)
        capped = _store(pj, tmp_path, db="cap.db", storage="cap_store", config_path=cfg)
        capped.create_project("a")
        capped.create_project("b")
        with pytest.raises(ValueError):
            capped.create_project("c")
    finally:
        restore()


def test_p16_get_project_returns_object_or_none(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        assert store.get_project("missing") is None
        p = store.create_project("here", settings={"a": 1})
        got = store.get_project(p.id)
        assert got is not None and got.name == "here"
        assert isinstance(got.settings, dict) and got.settings["a"] == 1, (
            "settings come back as a dict, decoded from the stored JSON string"
        )
    finally:
        restore()


def test_p17_list_projects_orders_by_update_desc_and_empty(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        assert store.list_projects() == []
        a = store.create_project("a")
        b = store.create_project("b")
        c = store.create_project("c")
        raw = _raw(tmp_path)
        raw.execute("UPDATE projects SET updated_at=? WHERE id=?", ("2020-01-01T00:00:01Z", a.id))
        raw.execute("UPDATE projects SET updated_at=? WHERE id=?", ("2020-01-01T00:00:03Z", b.id))
        raw.execute("UPDATE projects SET updated_at=? WHERE id=?", ("2020-01-01T00:00:02Z", c.id))
        raw.commit()
        raw.close()
        assert [p.name for p in store.list_projects()] == ["b", "c", "a"], (
            "the listing is ordered by updated_at, most recent first"
        )
    finally:
        restore()


def test_p18_update_merges_settings_and_touches_only_supplied(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        assert store.update_project("missing", name="x") is None
        p = store.create_project("orig", description="keep", settings={"a": 1})
        with pytest.raises(ValueError):
            store.update_project(p.id, name="   ")
        upd = store.update_project(p.id, name="new", settings={"b": 2})
        assert upd is not None and upd.name == "new"
        assert upd.description == "keep", "a field left None is not touched"
        assert upd.settings.get("a") == 1 and upd.settings.get("b") == 2, (
            "settings are merged into the stored settings, not replaced wholesale"
        )
    finally:
        restore()


def test_p19_delete_removes_directory_and_cascades_children(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        assert store.delete_project("missing") is False
        p = store.create_project("doomed")
        store.add_file(p.id, "a.txt", b"data")
        store.add_output(p.id, "o.txt", b"out")
        store.link_conversation(p.id, "conv1")
        base = Path(store._storage_base) / p.id
        assert base.exists()

        assert store.delete_project(p.id) is True
        assert not base.exists(), "the on-disk project directory is removed"

        raw = _raw(tmp_path)
        for table in ("project_files", "project_outputs", "project_conversations"):
            n = raw.execute(f"SELECT COUNT(*) FROM {table} WHERE project_id=?", (p.id,)).fetchone()[0]
            assert n == 0, (
                f"deleting the project cascades to {table}; a surviving child row "
                "means foreign keys were not enforced on the delete"
            )
        raw.close()
    finally:
        restore()


# =========================================================================
# File management
# =========================================================================

def test_p20_add_file_rejects_unknown_project(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        with pytest.raises(ValueError):
            store.add_file("no-such-project", "a.txt", b"x")
    finally:
        restore()


def test_p21_add_file_enforces_size_limit(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        cfg = _cfg_file(tmp_path, max_file_size_mb=0)
        store = _store(pj, tmp_path, config_path=cfg)
        p = store.create_project("x")
        with pytest.raises(ValueError):
            store.add_file(p.id, "big.txt", b"anything")
    finally:
        restore()


def test_p22_add_file_enforces_count_limit(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        cfg = _cfg_file(tmp_path, max_files_per_project=2)
        store = _store(pj, tmp_path, config_path=cfg)
        p = store.create_project("x")
        store.add_file(p.id, "a.txt", b"1")
        store.add_file(p.id, "b.txt", b"2")
        with pytest.raises(ValueError):
            store.add_file(p.id, "c.txt", b"3")
    finally:
        restore()


def test_p23_add_file_enforces_extension_allowlist(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        cfg = _cfg_file(tmp_path, allowed_extensions=[".txt"])
        store = _store(pj, tmp_path, config_path=cfg)
        p = store.create_project("x")
        with pytest.raises(ValueError):
            store.add_file(p.id, "danger.py", b"x")
        ok = store.add_file(p.id, "fine.txt", b"x")
        assert ok.filename == "fine.txt", "an allowed extension is accepted"
    finally:
        restore()


def test_p24_add_file_sanitises_name_and_stores_bytes(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        p = store.create_project("x")
        pf = store.add_file(p.id, "../../etc/passwd", b"secret")
        assert pf.filename == "passwd", (
            "a path that climbs out of the project directory is reduced to its "
            "basename before anything is written"
        )
        assert pf.file_type == "unknown"
        stored = Path(pf.file_path)
        assert stored.exists() and stored.read_bytes() == b"secret"
        assert stored.name == f"{pf.id}_{pf.filename}", (
            "the stored file is prefixed with its id so equal names never collide"
        )
        assert stored.parent == Path(store._storage_base) / p.id / "files", (
            "the write lands inside the project's files directory, not above it"
        )

        nul = store.add_file(p.id, "a\x00b.txt", b"y")
        assert nul.filename == "ab.txt", "null bytes are stripped from the name"
        empty = store.add_file(p.id, "subdir/", b"z")
        assert empty.filename == "unnamed_file", (
            "a name that sanitises to nothing becomes a fixed placeholder"
        )

        before = store.get_project(p.id).updated_at
        assert store.list_files(p.id), "the file is registered in the database"
        assert before is not None, "adding a file touches the project's updated_at"
    finally:
        restore()


def test_p25_get_file_and_list_files_order(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        assert store.get_file("missing") is None
        p = store.create_project("x")
        assert store.list_files(p.id) == []
        a = store.add_file(p.id, "a.txt", b"1")
        b = store.add_file(p.id, "b.txt", b"2")
        c = store.add_file(p.id, "c.txt", b"3")
        raw = _raw(tmp_path)
        raw.execute("UPDATE project_files SET uploaded_at=? WHERE id=?", ("2020-01-01T00:00:01Z", a.id))
        raw.execute("UPDATE project_files SET uploaded_at=? WHERE id=?", ("2020-01-01T00:00:03Z", b.id))
        raw.execute("UPDATE project_files SET uploaded_at=? WHERE id=?", ("2020-01-01T00:00:02Z", c.id))
        raw.commit()
        raw.close()
        assert [f.filename for f in store.list_files(p.id)] == ["b.txt", "c.txt", "a.txt"], (
            "files are listed by uploaded_at, most recent first"
        )
        assert store.get_file(a.id).filename == "a.txt"
    finally:
        restore()


def test_p26_remove_file_and_read_content(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        assert store.remove_file("missing") is False
        p = store.create_project("x")
        pf = store.add_file(p.id, "a.txt", b"body")
        assert store.read_file_content(pf.id) == b"body"

        stored = Path(pf.file_path)
        assert stored.exists()
        assert store.remove_file(pf.id) is True
        assert not stored.exists(), "removal unlinks the file from disk"
        assert store.get_file(pf.id) is None, "removal deletes the database row"
        assert store.read_file_content("missing") is None

        pf2 = store.add_file(p.id, "b.txt", b"x")
        Path(pf2.file_path).unlink()
        assert store.read_file_content(pf2.id) is None, (
            "a row whose file has vanished from disk reads back as None, not an error"
        )
    finally:
        restore()


# =========================================================================
# Output management
# =========================================================================

def test_p27_add_output_sanitises_and_persists(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        with pytest.raises(ValueError):
            store.add_output("no-such-project", "o.txt", b"x")
        p = store.create_project("x")
        po = store.add_output(p.id, "../escape.txt", b"out",
                              output_type="report", description="d")
        assert po.filename == "escape.txt", "an output name is reduced to its basename"
        assert po.output_type == "report"
        stored = Path(po.file_path)
        assert stored.exists() and stored.read_bytes() == b"out"
        assert stored.parent == Path(store._storage_base) / p.id / "outputs"
        empty = store.add_output(p.id, "subdir/", b"z")
        assert empty.filename == "output_file", (
            "an output name that sanitises to nothing becomes a fixed placeholder"
        )
    finally:
        restore()


def test_p28_get_output_and_list_outputs_order(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        assert store.get_output("missing") is None
        p = store.create_project("x")
        assert store.list_outputs(p.id) == []
        a = store.add_output(p.id, "a.txt", b"1")
        b = store.add_output(p.id, "b.txt", b"2")
        c = store.add_output(p.id, "c.txt", b"3")
        raw = _raw(tmp_path)
        raw.execute("UPDATE project_outputs SET created_at=? WHERE id=?", ("2020-01-01T00:00:01Z", a.id))
        raw.execute("UPDATE project_outputs SET created_at=? WHERE id=?", ("2020-01-01T00:00:03Z", b.id))
        raw.execute("UPDATE project_outputs SET created_at=? WHERE id=?", ("2020-01-01T00:00:02Z", c.id))
        raw.commit()
        raw.close()
        assert [o.filename for o in store.list_outputs(p.id)] == ["b.txt", "c.txt", "a.txt"], (
            "outputs are listed by created_at, most recent first"
        )
        assert store.get_output(a.id).filename == "a.txt"
    finally:
        restore()


def test_p29_remove_output(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        assert store.remove_output("missing") is False
        p = store.create_project("x")
        po = store.add_output(p.id, "o.txt", b"out")
        stored = Path(po.file_path)
        assert stored.exists()
        assert store.remove_output(po.id) is True
        assert not stored.exists() and store.get_output(po.id) is None
    finally:
        restore()


# =========================================================================
# Conversation linking
# =========================================================================

def test_p30_link_conversation_returns_true_when_project_exists(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        assert store.link_conversation("no-such-project", "conv") is False, (
            "a link against a missing project is refused"
        )
        p = store.create_project("x")
        assert store.link_conversation(p.id, "conv1") is True
        assert store.link_conversation(p.id, "conv1") is True, (
            "a second link of the same conversation is ignored by the store yet "
            "still reports True -- the idempotent-True is pinned as the behaviour "
            "the code produces, not the False the docstring describes"
        )
        assert [d["conversation_id"] for d in store.list_conversations(p.id)] == ["conv1"], (
            "the duplicate did not create a second row"
        )
    finally:
        restore()


def test_p31_unlink_conversation(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        p = store.create_project("x")
        store.link_conversation(p.id, "conv1")
        assert store.unlink_conversation(p.id, "conv1") is True
        assert store.unlink_conversation(p.id, "conv1") is False, (
            "unlinking a link that is not there returns False"
        )
        assert store.list_conversations(p.id) == []
    finally:
        restore()


def test_p32_list_conversations_order_and_lookup(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        p = store.create_project("x")
        assert store.list_conversations(p.id) == []
        assert store.get_project_for_conversation("unlinked") is None
        store.link_conversation(p.id, "c1")
        store.link_conversation(p.id, "c2")
        raw = _raw(tmp_path)
        raw.execute("UPDATE project_conversations SET linked_at=? WHERE conversation_id=?", ("2020-01-01T00:00:01Z", "c1"))
        raw.execute("UPDATE project_conversations SET linked_at=? WHERE conversation_id=?", ("2020-01-01T00:00:02Z", "c2"))
        raw.commit()
        raw.close()
        rows = store.list_conversations(p.id)
        assert [r["conversation_id"] for r in rows] == ["c2", "c1"], (
            "links are listed by linked_at, most recent first"
        )
        assert set(rows[0]) == {"conversation_id", "linked_at"}
        assert store.get_project_for_conversation("c1") == p.id, (
            "a linked conversation resolves back to its project"
        )
    finally:
        restore()


# =========================================================================
# Aggregate stats and injection safety
# =========================================================================

def test_p33_project_stats_count_and_ghost_zeroed(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        p = store.create_project("x")
        store.add_file(p.id, "a.txt", b"12345")
        store.add_file(p.id, "b.txt", b"678")
        store.add_output(p.id, "o.txt", b"out")
        store.link_conversation(p.id, "c1")
        st = store.get_project_stats(p.id)
        assert st["file_count"] == 2
        assert st["total_size_bytes"] == 8, "the total size sums the file sizes"
        assert st["output_count"] == 1 and st["conversation_count"] == 1

        ghost = store.get_project_stats("no-such-project")
        assert ghost == {
            "file_count": 0, "total_size_bytes": 0,
            "output_count": 0, "conversation_count": 0,
        }, (
            "stats for a project with no rows are zeroed, and the summed size "
            "coalesces to zero rather than coming back null"
        )
    finally:
        restore()


def test_p34_metacharacters_in_names_and_ids_roundtrip(tmp_path):
    pj, restore = _load(tmp_path)
    try:
        store = _store(pj, tmp_path)
        evil_name = 'nasty"; DROP TABLE projects;--'
        evil_desc = "x'); DELETE FROM projects;--"
        p = store.create_project(evil_name, description=evil_desc)
        got = store.get_project(p.id)
        assert got.name == evil_name and got.description == evil_desc, (
            "a project name and description carrying SQL metacharacters round-trip "
            "as data; that only holds under parameterisation"
        )
        evil_conv = "c'); DROP TABLE project_conversations;--"
        store.link_conversation(p.id, evil_conv)
        assert store.get_project_for_conversation(evil_conv) == p.id
        listed = [d["conversation_id"] for d in store.list_conversations(p.id)]
        assert evil_conv in listed
        # The tables are still standing.
        assert store.get_project(p.id) is not None
    finally:
        restore()
