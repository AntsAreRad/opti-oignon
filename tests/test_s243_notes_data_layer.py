"""S243 -- the Notes feature's N.1 data layer: the SQLCipher metadata/text store
with per-user isolation and the two-layer per-attachment AES-256-GCM blob store.

This is a real source feature lot, not a host-assured runbook. It adds the
container-provable data layer that unblocks the Mobile cycle's gated notes/vault
bloc (N.8 the Veilid record type, N.9 the phone-app sync contract). It is genuinely
net-new: before this lot no notes module, ``manage_notes`` tool, ``class Note`` or
``notes_store`` existed anywhere in source. N.1 is the data layer ONLY -- the notes
UI (N.2+), the LLM-from-note / LLM-from-chat surfaces (N.3 / N.4, the gated
``manage_notes`` STATE_MUTATION tool), and the Veilid record type (N.8) are later
blocs.

The new subpackage ``opti_oignon/notes/`` carries two focused modules, mirroring
the codebase's sibling-module shape (peers.py / deferred_ledger.py under veilid;
canonical_store.py / vector_store.py under memory):

 - ``notes_store.py``: the SQLCipher metadata/text store. ``note`` and
   ``attachment`` tables with a ``user_id`` column, per-user isolation via the
   ``effective_user_id`` pattern (the memory store is the reference, NOT the
   user_data_manager prefix bug UD-01), opened through ``safe_connect`` with the
   S136 ImportError plaintext fallback, WAL, parameterized SQL with no f-strings,
   and the allowlist idiom (a frozenset + ``str.format``) for the one interpolated
   identifier (ORDER BY) -- the sanctioned alternative to f-string SQL. Tombstone
   deletes; the body CRDT and the OR-Set tags are stored opaque (the backend stays
   CRDT-agnostic).
 - ``blob_store.py``: the per-attachment AES-256-GCM blob store. Each media blob is
   sealed under a PER-ATTACHMENT subkey derived (N1-D1) by the codebase's
   domain-separated HMAC-SHA256 construction -- the idiom signing.py / db_encryption
   .py / auth_2fa.py already use, what db_encryption calls "HKDF-like construction
   (HMAC-SHA256)" -- with the exact roadmap domain string
   ``b"oo-notes-attachment-" + attachment_id``, a fresh nonce per blob (via
   ``encrypt_bytes``' ``os.urandom``), keys in SecureBytes, no plaintext temp on
   disk (the encrypted blob is written directly; decryption is in memory only). The
   master key is injectable (defaulting to ``get_encryption_key``) and the seal
   refuses without one (``NotesBlobUnavailable``), so no plaintext blob ever lands
   on disk -- the signing.py SigningUnavailable posture.

Families:

 1. Source / structure (raw source pins) -- the subpackage and both modules exist,
    notes_store imports ``safe_connect`` with the S136 fallback, the
    ``checkpoint_before_apply = True`` sentinel and the ``FEATURE_AVAILABLE`` flag
    are present in both, the ORDER BY allowlist frozenset is present, pure ASCII /
    no decoration.
 2. Schema / parameterization (raw source pins) -- ``CREATE TABLE note`` and
    ``CREATE TABLE attachment`` carry ``user_id``; the attachment-kind allowlist and
    the ``oo-notes-attachment-`` domain string are present; SQL is parameterized
    (``?`` placeholders) with no f-string SQL; ``effective_user_id`` scopes the
    queries.
 3. Behavioural (isolated module load; green only after the implementation) -- the
    store opens against a temp dir under WAL with an empty count, a note added for
    one user is invisible to another (per-user isolation), a tombstone delete hides
    the note, attachments round-trip and stay scoped; the blob store round-trips a
    payload under an injected master key (the on-disk bytes are ciphertext, the
    reopened bytes are the plaintext), a blob is bound to its attachment id (a blob
    sealed for one id does not open under another), the nonce is fresh per blob, no
    plaintext temp lands on disk, and the seal refuses without a master key.
 4. Premise guards (raw source pins; green before AND after) -- the surfaces this
    lot rests on are intact: ``db_utils.safe_connect``, ``encryption.encrypt_bytes``
    / ``decrypt_bytes``, ``secure_bytes.SecureBytes``, ``user_isolation
    .effective_user_id``, the memory canonical_store isolation reference, and the
    N.4 boundary (``manage_notes`` is NOT yet in STATE_MUTATION_TOOLS).
 5. Doc rolls (flattened doc pins; green after the additive rolls) --
    NOTES_FEATURE_ROADMAP.md N.1 rolled to landed at S243; ATREST_INVENTORY.md
    carries an additive design note for the new notes stores.
 6. AST validity of the two new modules, the package __init__, and this suite.

Red-before discipline on the pristine S242 tree (no opti_oignon/notes/, docs not
rolled): families 1, 2, 3, 5 FAIL (the read helpers return empty strings so absence
is a failure, never a collection error; the behavioural isolated loader raises
inside the test body when the source files are absent), family 6's module-parse
pins FAIL (empty source is rejected before ``ast.parse``), while family 4 and the
suite-parse pin PASS by design (they pin pre-existing invariants this lot relies
on). Document pins read through a whitespace-flattening helper so reflow that does
not change wording cannot break them; source pins stay raw. The behavioural family
loads the modules in isolation via ``spec_from_file_location`` under flat names
(robust to an earlier suite stubbing ``sys.modules["opti_oignon"]`` as a
non-package), so collection itself touches no import chain and the full regression
sweep is order-independent.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "opti_oignon"

NOTES_DIR = PKG / "notes"
NOTES_INIT_PATH = NOTES_DIR / "__init__.py"
NOTES_STORE_PATH = NOTES_DIR / "notes_store.py"
BLOB_STORE_PATH = NOTES_DIR / "blob_store.py"

DB_UTILS_PATH = PKG / "db_utils.py"
ENCRYPTION_PATH = PKG / "encryption.py"
SECURE_BYTES_PATH = PKG / "secure_bytes.py"
USER_ISO_PATH = PKG / "user_isolation.py"
CANONICAL_STORE_PATH = PKG / "memory" / "canonical_store.py"
ALLOWLISTS_PATH = PKG / "agent" / "allowlists.py"

ROADMAP_NOTES_PATH = REPO / "NOTES_FEATURE_ROADMAP.md"
ATREST_PATH = REPO / "ATREST_INVENTORY.md"


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _flat(text: str) -> str:
    """Collapse all whitespace runs to single spaces (reflow-immune pins)."""
    return re.sub(r"\s+", " ", text)


def _notes_store_src() -> str:
    return _read(NOTES_STORE_PATH)


def _blob_store_src() -> str:
    return _read(BLOB_STORE_PATH)


# The behavioural family loads the two modules in ISOLATION via
# spec_from_file_location under flat names. This is robust to an earlier test in
# the chain having replaced sys.modules["opti_oignon"] with a non-package stub
# (the importlib-isolation idiom several suites use), which would otherwise break
# a plain ``from opti_oignon.notes...`` import in the full regression sweep. Under
# a flat name the modules' guarded relative imports fall back: notes_store
# degrades to a plaintext sqlite connection (the documented in-container posture,
# identical to safe_connect with SQLCipher absent), and blob_store's AES-256-GCM
# primitives are injected from an isolated load of encryption.py so the crypto
# round-trip is exercised with the REAL primitive, not a stub.
_ISOLATED: dict = {}


def _isolated(mod_name: str, rel: str):
    if mod_name not in _ISOLATED:
        spec = importlib.util.spec_from_file_location(mod_name, str(PKG / rel))
        if spec is None or spec.loader is None:
            raise ImportError(mod_name)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        _ISOLATED[mod_name] = mod
    return _ISOLATED[mod_name]


def _notes_modules():
    """Return (notes_store, blob_store) modules loaded in isolation, with
    blob_store's AES-256-GCM primitives injected from a real isolated encryption
    load. Raises if the source files are absent (the red-before failure shape)."""
    ns = _isolated("s243_notes_store_iso", "notes/notes_store.py")
    bs = _isolated("s243_blob_store_iso", "notes/blob_store.py")
    if getattr(bs, "encrypt_bytes", None) is None:
        enc = _isolated("s243_encryption_iso", "encryption.py")
        bs.encrypt_bytes = enc.encrypt_bytes
        bs.decrypt_bytes = enc.decrypt_bytes
    return ns, bs


# ---------------------------------------------------------------------------
# Family 1 -- source / structure
# ---------------------------------------------------------------------------


class TestNotesPackageExists:
    def test_subpackage_and_modules_exist(self):
        assert NOTES_DIR.is_dir(), "opti_oignon/notes/ missing"
        assert NOTES_INIT_PATH.exists(), "opti_oignon/notes/__init__.py missing"
        assert NOTES_STORE_PATH.exists(), "notes_store.py missing"
        assert BLOB_STORE_PATH.exists(), "blob_store.py missing"

    def test_pure_ascii_no_decoration(self):
        for path in (NOTES_STORE_PATH, BLOB_STORE_PATH):
            raw = _read(path)
            assert raw != "", path.name
            assert all(ord(c) < 128 for c in raw), path.name
            assert "====" not in raw, path.name


class TestNotesStoreStructure:
    def test_safe_connect_import_with_s136_fallback(self):
        src = _notes_store_src()
        assert "from ..db_utils import safe_connect" in src
        # The S136 ImportError plaintext fallback names the degradation and keeps
        # a residual sqlite3.connect for the fallback path.
        assert "except" in src
        assert "sqlite3.connect" in src

    def test_checkpoint_sentinel_present(self):
        assert "checkpoint_before_apply = True" in _notes_store_src()
        assert "checkpoint_before_apply = True" in _blob_store_src()

    def test_feature_available_flag_present(self):
        assert "FEATURE_AVAILABLE = True" in _notes_store_src()
        assert "FEATURE_AVAILABLE = True" in _blob_store_src()

    def test_orderby_allowlist_frozenset_present(self):
        src = _notes_store_src()
        assert "_ORDERABLE_COLUMNS" in src
        assert "frozenset" in src

    def test_singleton_reset_hooks_present(self):
        assert "def reset_notes_store(" in _notes_store_src()
        assert "def reset_notes_blob_store(" in _blob_store_src()


# ---------------------------------------------------------------------------
# Family 2 -- schema / parameterization
# ---------------------------------------------------------------------------


class TestSchema:
    def test_note_table_has_user_id(self):
        src = _notes_store_src()
        assert "CREATE TABLE IF NOT EXISTS note" in src
        # tombstone + opaque body + opaque tags columns
        assert "body_crdt" in src
        assert "tags" in src
        assert "deleted" in src

    def test_attachment_table_has_user_id_and_note_id(self):
        src = _notes_store_src()
        assert "CREATE TABLE IF NOT EXISTS attachment" in src
        assert "note_id" in src
        assert "blob_ref" in src
        # opt-in derived text columns
        assert "transcript_text" in src
        assert "caption_text" in src
        assert "ocr_text" in src

    def test_user_id_column_in_both_tables(self):
        src = _notes_store_src()
        # both CREATE TABLE blocks must declare user_id
        assert src.count("user_id TEXT NOT NULL") >= 2

    def test_attachment_kind_allowlist_present(self):
        src = _notes_store_src()
        assert "ATTACHMENT_KINDS" in src
        for kind in ("audio", "image", "drawing"):
            assert kind in src, kind

    def test_no_fstring_sql(self):
        src = _notes_store_src()
        for needle in ('f"INSERT', "f'INSERT", 'f"SELECT', "f'SELECT",
                       'f"UPDATE', "f'UPDATE", 'f"DELETE', "f'DELETE",
                       'f"CREATE', "f'CREATE"):
            assert needle not in src, needle

    def test_parameterized_placeholders_present(self):
        src = _notes_store_src()
        assert "VALUES (?" in src
        assert "WHERE id = ? AND user_id = ?" in src

    def test_effective_user_id_scoping(self):
        assert "effective_user_id(" in _notes_store_src()


class TestBlobCryptoSource:
    def test_attachment_domain_string_present(self):
        src = _blob_store_src()
        assert 'b"oo-notes-attachment-"' in src

    def test_hmac_sha256_subkey_idiom(self):
        src = _blob_store_src()
        assert "hmac" in src
        assert "sha256" in src

    def test_aesgcm_primitive_reused(self):
        src = _blob_store_src()
        assert "encrypt_bytes" in src
        assert "decrypt_bytes" in src

    def test_unavailable_without_master_key(self):
        src = _blob_store_src()
        assert "class NotesBlobUnavailable" in src

    def test_securebytes_for_keys(self):
        assert "SecureBytes" in _blob_store_src() or "secure_key_from_bytes" in _blob_store_src()


# ---------------------------------------------------------------------------
# Family 3 -- behavioural (isolated module load; green only after the implementation)
# ---------------------------------------------------------------------------


class TestNotesStoreBehaviour:
    def test_opens_empty_under_wal(self, tmp_path):
        ns, _ = _notes_modules()
        store = ns.NotesStore(root=tmp_path)
        try:
            assert store.count_notes() == 0
            assert store.journal_mode() == "wal"
            assert store.db_path.name == "notes.db"
        finally:
            store.close()

    def test_per_user_isolation(self, tmp_path):
        ns, _ = _notes_modules()
        store = ns.NotesStore(root=tmp_path, single_user_mode=False)
        try:
            rec = store.add_note("alice note", user_id="alice")
            assert rec.user_id == "alice"
            assert len(store.list_notes(user_id="alice")) == 1
            # bob sees nothing; the row is scoped by user_id, not a prefix (UD-01)
            assert store.list_notes(user_id="bob") == []
            assert store.get_note(rec.id, user_id="bob") is None
            assert store.get_note(rec.id, user_id="alice") is not None
        finally:
            store.close()

    def test_tombstone_delete_hides_note(self, tmp_path):
        ns, _ = _notes_modules()
        store = ns.NotesStore(root=tmp_path)
        try:
            rec = store.add_note("doomed")
            assert store.count_notes() == 1
            assert store.delete_note(rec.id) is True
            assert store.count_notes() == 0
            assert store.list_notes() == []
            # the row survives as a tombstone (sync-correct deletion)
            assert len(store.list_notes(include_deleted=True)) == 1
        finally:
            store.close()

    def test_attachments_roundtrip_and_scope(self, tmp_path):
        ns, _ = _notes_modules()
        store = ns.NotesStore(root=tmp_path, single_user_mode=False)
        try:
            note = store.add_note("with media", user_id="alice")
            att = store.add_attachment(
                note.id, "image", blob_ref=note.id + ":0", mime="image/png",
                user_id="alice",
            )
            assert att.kind == "image"
            assert len(store.list_attachments(note.id, user_id="alice")) == 1
            assert store.list_attachments(note.id, user_id="bob") == []
        finally:
            store.close()

    def test_attachment_kind_validated(self, tmp_path):
        ns, _ = _notes_modules()
        store = ns.NotesStore(root=tmp_path)
        try:
            note = store.add_note("n")
            with pytest.raises(ValueError):
                store.add_attachment(note.id, "video", blob_ref="x", mime="video/mp4")
        finally:
            store.close()


class TestBlobStoreBehaviour:
    def test_seal_open_roundtrip_with_injected_key(self, tmp_path):
        _, bs = _notes_modules()
        master = os.urandom(32)
        blob = bs.NotesBlobStore(root=tmp_path, master_key=master)
        payload = b"the-decrypted-attachment-bytes"
        path = blob.seal("att-1", payload)
        assert path.exists()
        # on-disk is ciphertext, not the plaintext
        on_disk = path.read_bytes()
        assert payload not in on_disk
        assert on_disk != payload
        # reopen yields the plaintext
        assert blob.open("att-1") == payload

    def test_blob_bound_to_attachment_id(self, tmp_path):
        _, bs = _notes_modules()
        master = os.urandom(32)
        blob = bs.NotesBlobStore(root=tmp_path, master_key=master)
        p1 = blob.seal("att-1", b"secret-payload")
        # copy att-1's ciphertext under att-2's path: the att-2 subkey must not open it
        p2 = blob._blob_path("att-2")
        shutil.copyfile(p1, p2)
        with pytest.raises(Exception):
            blob.open("att-2")

    def test_fresh_nonce_per_blob(self, tmp_path):
        _, bs = _notes_modules()
        master = os.urandom(32)
        blob = bs.NotesBlobStore(root=tmp_path, master_key=master)
        pa = blob.seal("att-a", b"same-bytes").read_bytes()
        pb = blob.seal("att-b", b"same-bytes").read_bytes()
        assert pa != pb
        # nonce is bytes [1:13] of the version||nonce||ct||tag format
        assert pa[1:13] != pb[1:13]

    def test_no_plaintext_temp_on_disk(self, tmp_path):
        _, bs = _notes_modules()
        master = os.urandom(32)
        blob = bs.NotesBlobStore(root=tmp_path, master_key=master)
        marker = b"PLAINTEXT-MARKER-7f3a9c"
        blob.seal("att-x", marker)
        for f in blob.blob_dir.iterdir():
            assert marker not in f.read_bytes(), f.name

    def test_seal_refuses_without_master_key(self, tmp_path):
        _, bs = _notes_modules()
        # in-container get_encryption_key() is None and no key injected
        blob = bs.NotesBlobStore(root=tmp_path)
        with pytest.raises(bs.NotesBlobUnavailable):
            blob.seal("att-z", b"data")


# ---------------------------------------------------------------------------
# Family 4 -- premise guards (green before AND after)
# ---------------------------------------------------------------------------


class TestPremiseGuards:
    def test_safe_connect_exists(self):
        assert "def safe_connect(" in _read(DB_UTILS_PATH)

    def test_aesgcm_primitives_exist(self):
        src = _read(ENCRYPTION_PATH)
        assert "def encrypt_bytes(" in src
        assert "def decrypt_bytes(" in src

    def test_secure_bytes_exists(self):
        assert "class SecureBytes" in _read(SECURE_BYTES_PATH)

    def test_effective_user_id_exists(self):
        assert "def effective_user_id(" in _read(USER_ISO_PATH)

    def test_memory_store_is_the_isolation_reference(self):
        assert "effective_user_id(" in _read(CANONICAL_STORE_PATH)

    def test_manage_notes_not_yet_a_state_mutation_tool(self):
        # The N.4 boundary: manage_notes is a later bloc, not this data-layer lot.
        src = _read(ALLOWLISTS_PATH)
        assert "STATE_MUTATION_TOOLS = frozenset(" in src
        assert "manage_notes" not in src


# ---------------------------------------------------------------------------
# Family 5 -- doc rolls (additive)
# ---------------------------------------------------------------------------


class TestDocRolls:
    def test_roadmap_n1_rolled_to_landed(self):
        text = _flat(_read(ROADMAP_NOTES_PATH))
        assert "N.1" in text
        assert "LANDED at S243" in text

    def test_atrest_notes_design_note(self):
        text = _flat(_read(ATREST_PATH))
        assert "Notes data layer" in text
        assert "oo-notes-attachment-" in text


# ---------------------------------------------------------------------------
# Family 6 -- AST validity
# ---------------------------------------------------------------------------


class TestAST:
    def test_new_modules_parse(self):
        for path in (NOTES_INIT_PATH, NOTES_STORE_PATH, BLOB_STORE_PATH):
            src = _read(path)
            assert src != "", path.name
            ast.parse(src, filename=str(path))

    def test_this_suite_parses(self):
        src = _read(Path(__file__))
        assert src != ""
        ast.parse(src, filename=__file__)
