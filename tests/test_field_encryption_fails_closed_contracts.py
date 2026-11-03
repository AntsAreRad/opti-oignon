#!/usr/bin/env python3
"""Contracts: encryption that is on never writes a field in clear.

Field encryption (``encryption.encrypt_field``, the manager behind it) seals conversation titles and messages, legacy
memory's facts and TOTP secrets. It wrote them in clear while security.yaml said encryption was on: a security.yaml
it could not read or parse, or whose section held no mapping, read as encryption off; encryption on with no key that
loads turned the manager off with one warning; a cipher that raised handed the value back as it was; and a TOTP
secret was stored as base64 whenever the manager was not on, whatever the switch said.

  * EF1 -- the manager reads its state from security.yaml, strictly, and the key: on (the switch on and a key) seals;
    off (no file, no section, the switch off) hands the value back; refusing (the switch on and no key, or a file
    that cannot say: empty, no YAML, no mapping, a section holding no mapping, unreadable) raises
    ``EncryptionUnavailable`` with its reason and hands nothing back, says the reason in its status and once in the
    log, and still reads what a key that loads sealed; the empty value and a sealed one pass as they are.
  * EF2 -- a manager told on with no key, with no AES library, or with a cipher that does not seal a fixed value,
    refuses; a cipher that raises is refused, never answered with the value, and the manager refuses from then on
    while a fixed value does not seal either; a value that is no text (a lone surrogate) raises its own error and
    the manager refuses nothing more for it; the three setup methods (a random key, a passphrase, the key that
    loads) leave the refusing state, the key and the switch set before the reason is cleared, and the key file a
    setup writes is no change for the manager to follow.
  * EF3 -- over a refusing manager, each conversation write (a titled conversation, a rename, a message, the title
    given back to the rule, the titles tended at start, a refined title, a synced conversation) and legacy memory's
    facts (added, updated) write nothing in clear and change nothing; the conversation routes answer 503 with the
    reason. Over a manager that is on, the same writes seal; off, a title is written as before.
  * EF4 -- a TOTP secret is sealed when encryption is on, base64 only when it is off, and refused with nothing stored
    when the manager refuses, cannot be built or its cipher fails; a sealed secret is read while the manager refuses
    with a key that loads; the setup route answers 503 with the reason.
  * EF5 -- the chat sockets (``/stream`` and ``/retry``) refuse a turn with the reason before reading it when the
    conversation store is there and its field encryption refuses (no key, a file that cannot say, no AES library, a
    cipher that failed on an earlier save) or cannot be asked; on and off, and with no store while it refuses, the
    turn is read.
  * EF6 -- the manager the stores use follows security.yaml and the key file while the platform runs: the switch
    turned on with no key refuses at the next write and says so, a key file put back seals, a file broken refuses
    and mended seals with the key held; a cipher refusal kept through an unrelated save while the cipher still
    fails, and ended once it seals again; the key file the held key came from removed, or holding another key,
    refuses, and put back seals; the switch turned off writes as told; a FIFO at security.yaml's name or at the key
    file's is never waited on; a manager told its state does not follow; one that cannot be built refuses.

Local-only (the public distribution ships no tests). The real ``encryption.py`` is loaded from its source through
the shared isolation window, beside the real conversation store, its titles and routes, legacy memory, the security
routes and the chat routes (each window loads what its clause needs); ``secure_bytes`` is unreachable (it installs
signal handlers when imported). The module's ``__file__`` is moved under the test's temporary directory, so its own
reader reads a security.yaml there; its key file is there too, OPTI_ENCRYPTION_KEY and OPTI_KEYFILE_PASSPHRASE are
cleared unless a clause sets the key. The TOTP clauses use the two-factor suite's window, the real encryption module
put in place of its stand-in; the chat clauses use the user-turn suite's stand-ins for the deps, the session and the
store. Nothing reads the tree's data or keys.
"""

import asyncio
import base64
import importlib.util
import logging
import os
import sqlite3
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_auth_2fa_secret_at_rest_contracts as sa_t  # noqa: E402
import test_user_turn_contracts as ut_t  # noqa: E402
from _isolation import isolate, source  # noqa: E402

# Seconds per junit case, the parametrized ids included: the ladder reads this literal alone.
BUDGET_S = {
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[no_file-a_key_loads]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[no_file-no_key]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[no_section-a_key_loads]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[no_section-no_key]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear"
    "[switched_off-a_key_loads]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[switched_off-no_key]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[switched_on-a_key_loads]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[switched_on-no_key]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[empty-a_key_loads]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[empty-no_key]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[not_YAML-a_key_loads]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[not_YAML-no_key]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[no_mapping-a_key_loads]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[no_mapping-no_key]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear"
    "[a_section_with_no_mapping-a_key_loads]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear"
    "[a_section_with_no_mapping-no_key]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[unreadable-a_key_loads]": 2.0,
    "test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear[unreadable-no_key]": 2.0,
    "test_ef2_a_manager_told_on_without_a_key_or_a_failing_cipher_refuses_and_setup_leaves_the_refusal": 3.0,
    "test_ef3_the_stores_write_nothing_in_clear_over_a_refusing_manager_and_the_routes_say_why[no_key]": 3.0,
    "test_ef3_the_stores_write_nothing_in_clear_over_a_refusing_manager_and_the_routes_say_why[cannot_say]": 3.0,
    "test_ef4_a_totp_secret_is_sealed_when_on_base64_only_when_off_and_refused_when_the_manager_refuses[no_key]": 2.0,
    "test_ef4_a_totp_secret_is_sealed_when_on_base64_only_when_off_and_refused_when_the_manager_refuses"
    "[cannot_say]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[stream-no_key]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[stream-cannot_say]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[stream-no_cipher]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[stream-cipher_failed]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[stream-cannot_be_asked]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[stream-on]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[stream-off]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[stream-no_store]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[retry-no_key]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[retry-cannot_say]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[retry-no_cipher]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[retry-cipher_failed]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[retry-cannot_be_asked]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[retry-on]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[retry-off]": 2.0,
    "test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it[retry-no_store]": 2.0,
    "test_ef6_the_manager_follows_security_yaml_and_the_key_file_while_the_platform_runs": 3.0,
}

_KEY = bytes(range(32))
_ENV_KEY = base64.urlsafe_b64encode(_KEY).decode("ascii")
_VARIABLES = ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE")
_ENC = "opti_oignon.encryption"
_SECRETS = "opti_oignon.process_secrets"
_CONV = "opti_oignon.conversation"
_TITLES = "opti_oignon.conversation_titles"
_LEGACY = "opti_oignon.memory.legacy"
_ROUTES = "opti_oignon.api.routes_conversations"
_SECRET = "the owner's words, never in clear"

# security.yaml as a clause leaves it: None is no file.
_FILES = {
    "no_file": None,
    "no_section": "theme: dark\n",
    "switched_off": "encryption:\n  enabled: false\n",
    "switched_on": "encryption:\n  enabled: true\n",
    "empty": "",
    "not_YAML": "encryption: [enabled\n",
    "no_mapping": "- encryption\n",
    "a_section_with_no_mapping": "encryption: true\n",
    "unreadable": "encryption:\n  enabled: false\n",
}
_OFF = ("no_file", "no_section", "switched_off")
_REFUSALS = {"no_key": ("switched_on", None), "cannot_say": ("not_YAML", _ENV_KEY)}


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def _home(enc, tmp_path):
    """Point the real module at ``tmp_path``: its own reader reads ``pkg/config/security.yaml`` there, its key file is
    there, its manager is built again; returns the security.yaml path."""
    package = tmp_path / "pkg"
    (package / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "keys").mkdir(exist_ok=True)
    enc.__file__ = str(package / "encryption.py")
    enc._DEFAULT_KEYFILE = tmp_path / "keys" / ".keyfile"
    enc._encryption_manager = None
    return package / "config" / "security.yaml"


def _write(config, case):
    """Leave security.yaml as ``case`` says."""
    if config.exists():
        config.chmod(0o600)
        config.unlink()
    text = _FILES[case]
    if text is None:
        return
    config.write_text(text, encoding="utf-8")
    if case == "unreadable":
        config.chmod(0)


def _state(enc, config, monkeypatch, case, key):
    """security.yaml as ``case``, OPTI_ENCRYPTION_KEY set to ``key`` (None: unset), the manager built again."""
    _write(config, case)
    if key is None:
        monkeypatch.delenv("OPTI_ENCRYPTION_KEY", raising=False)
    else:
        monkeypatch.setenv("OPTI_ENCRYPTION_KEY", key)
    enc._encryption_manager = None


def _release(fifo):
    """Open the FIFO for writing without blocking, so that a reader it holds goes on (with no reader, nothing)."""
    try:
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
    except OSError:
        pass


def _closed(config):
    if config.exists():
        config.chmod(0o600)


class _NoSuchRefusal(Exception):
    """Stands for a refusal the module under test does not have: never raised."""


def _unavailable(enc):
    """The module's refusal, so that code with none fails on what it did rather than on a missing name."""
    return getattr(enc, "EncryptionUnavailable", _NoSuchRefusal)


def _reason(enc, manager=None):
    """The reason the manager (the module's own when not given) refuses, or None."""
    if manager is None:
        refusal = getattr(enc, "field_refusal", None)
        return refusal() if refusal is not None else None
    return getattr(manager, "refusing", None)


def _sealing(enc, manager, value=_SECRET):
    """What ``encrypt`` makes of ``value``: ("refused", reason), ("in clear", value) or ("sealed", value)."""
    try:
        made = manager.encrypt(value)
    except _unavailable(enc) as exc:
        return "refused", str(exc)
    return ("in clear" if made == value else "sealed"), made


def _encryption(tmp_path, monkeypatch):
    """The real encryption module alone; returns it, its security.yaml path and ``restore``."""
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)
    loaded, restore = isolate(
        targets={_SECRETS: source("process_secrets.py"), _ENC: source("encryption.py")},
        blocked=("opti_oignon.secure_bytes",),
    )
    try:
        config = _home(loaded[_ENC], tmp_path)
    except BaseException:
        restore()
        raise
    return loaded[_ENC], config, restore


class _Engine:
    """A sync engine that keeps nothing."""

    def current_clock(self, kind, record_id):
        return 0

    def publish_conversation(self, conversation_id, payload=None, *, clock, deleted=False, updated_at=""):
        return 1


def _stores(tmp_path, monkeypatch):
    """The conversation store with its titles and routes, and legacy memory, over ``tmp_path``'s data directory, the
    real encryption module beside them."""
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)
    data = tmp_path / "data"
    data.mkdir()
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda path, **kwargs: sqlite3.connect(path, check_same_thread=kwargs.get("check_same_thread",
                                                                                                False))
    config = types.ModuleType("opti_oignon.config")
    config.DATA_DIR = data
    guard = types.ModuleType("opti_oignon.veilid.guard")
    guard.veilid_available = lambda: True
    engine = types.ModuleType("opti_oignon.veilid.sync_engine")
    engine.get_sync_engine = lambda *args, **kwargs: _Engine()
    deps = ut_t._deps_stub()
    loaded, restore = isolate(
        targets={_SECRETS: source("process_secrets.py"), _ENC: source("encryption.py"),
                 "opti_oignon.veilid.records": source("veilid", "records.py"),
                 "opti_oignon.veilid.producers": source("veilid", "producers.py"),
                 _TITLES: source("conversation_titles.py"), _CONV: source("conversation.py"),
                 _LEGACY: source("memory", "legacy.py"),
                 "opti_oignon.api.schemas": source("api", "schemas.py"),
                 _ROUTES: source("api", "routes_conversations.py")},
        seeded={"opti_oignon.db_utils": db, "opti_oignon.config": config, "opti_oignon.veilid.guard": guard,
                "opti_oignon.veilid.sync_engine": engine, "opti_oignon.api.deps": deps},
        blocked=("opti_oignon.secure_bytes",),
        packages=("opti_oignon.veilid", "opti_oignon.memory", "opti_oignon.api"),
    )
    try:
        enc = loaded[_ENC]
        security = _home(enc, tmp_path)
        store = loaded[_CONV].conversation_manager
        routes = loaded[_ROUTES]
        routes.CONVERSATION_AVAILABLE = True
        routes.conversation_manager = store
    except BaseException:
        restore()
        raise
    return SimpleNamespace(enc=enc, config=security, data=data, store=store, routes=routes,
                           memory=loaded[_LEGACY].memory_manager, restore=restore)


def _in_clear(data, words):
    """The words found in any file under ``data``, whatever their case."""
    found = set()
    for path in data.rglob("*"):
        if path.is_file():
            blob = path.read_bytes().lower()
            found |= {word for word in words if word.lower().encode("utf-8") in blob}
    return found


def _rows(data):
    """Every row of every table of every database under ``data``."""
    rows = {}
    for path in sorted(data.rglob("*.db")):
        conn = sqlite3.connect(path)
        try:
            tables = [name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
            for table in tables:
                rows[(path.name, table)] = sorted(map(repr, conn.execute(f"SELECT * FROM {table}").fetchall()))
        finally:
            conn.close()
    return rows


def _load_source(name, *parts):
    """Load ``name`` from its source into the open window, bound on its stand-in parent."""
    spec = importlib.util.spec_from_file_location(name, source(*parts))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    setattr(sys.modules["opti_oignon"], name.rpartition(".")[2], module)
    spec.loader.exec_module(module)
    return module


def _two_factor(tmp_path, monkeypatch):
    """The two-factor suite's window with the real encryption module in place of its stand-in."""
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)
    ctx = sa_t._load()
    try:
        _load_source(_SECRETS, "process_secrets.py")
        enc = _load_source(_ENC, "encryption.py")
        config = _home(enc, tmp_path)
    except BaseException:
        ctx.restore()
        raise
    return ctx, enc, config


def _stored_secrets(recorder, user_id):
    return [params[1] for sql, params in recorder.statements
            if "INSERT" in sql and "totp_config" in sql and params and params[0] == user_id]


class _Socket(ut_t._Socket):
    """The user-turn suite's socket, counting the reads of its payload."""

    def __init__(self, payload=None):
        super().__init__(payload)
        self.read = 0

    async def receive_json(self):
        self.read += 1
        return await super().receive_json()


def _chat(tmp_path, monkeypatch):
    """The chat routes beside the real encryption module, the deps, the session and the store stood in for."""
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)
    loaded, restore = isolate(
        targets={_SECRETS: source("process_secrets.py"), _ENC: source("encryption.py"),
                 "opti_oignon.api.schemas": source("api", "schemas.py"),
                 "opti_oignon.api.routes_chat": source("api", "routes_chat.py")},
        seeded={"opti_oignon.api.deps": ut_t._deps_stub(), "opti_oignon.api.routes_auth": ut_t._auth_stub()},
        blocked=("opti_oignon.secure_bytes",),
        packages=("opti_oignon.api",),
    )
    try:
        config = _home(loaded[_ENC], tmp_path)
    except BaseException:
        restore()
        raise
    return loaded["opti_oignon.api.routes_chat"], loaded[_ENC], config, restore


# ---------------------------------------------------------------------------
# EF1 -- the manager's state, read strictly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["a_key_loads", "no_key"])
@pytest.mark.parametrize("case", list(_FILES))
def test_ef1_the_manager_reads_its_state_strictly_and_never_hands_back_a_value_in_clear(tmp_path, monkeypatch,
                                                                                       caplog, case, key):
    if case == "unreadable" and os.geteuid() == 0:
        pytest.skip("root reads a file of mode 0")
    enc, config, restore = _encryption(tmp_path, monkeypatch)
    try:
        _state(enc, config, monkeypatch, case, _ENV_KEY if key == "a_key_loads" else None)
        with caplog.at_level(logging.ERROR, logger=_ENC):
            manager = enc.EncryptionManager()
        said = [record.getMessage() for record in caplog.records if record.name == _ENC
                and record.levelno >= logging.ERROR]
        status = manager.get_status()
        toggle = enc.encryption_toggle(config)
        sealed_elsewhere = enc.EncryptionManager(key=_KEY, enabled=True).encrypt(_SECRET)
        if case in _OFF:
            assert _sealing(enc, manager) == ("in clear", _SECRET), "off: the value as it is"
            assert _reason(enc, manager) is None and manager.enabled is False and said == [], (case, key)
            assert status.get("refusing") is None and status["config_enabled"] is False and toggle is False
            assert _reason(enc) is None and enc.encrypt_field(_SECRET) == _SECRET
        elif case == "switched_on" and key == "a_key_loads":
            kind, sealed = _sealing(enc, manager)
            assert kind == "sealed" and sealed.startswith("ENC2:") and manager.decrypt(sealed) == _SECRET
            assert _reason(enc, manager) is None and manager.enabled is True and said == []
            assert status.get("refusing") is None and status["config_enabled"] is True and toggle is True
            assert _reason(enc) is None and enc.encrypt_field(_SECRET).startswith("ENC2:")
        else:
            kind, made = _sealing(enc, manager)
            assert kind == "refused", (case, key, "handed back", kind, made)
            reason = _reason(enc, manager)
            words = "no key loads" if case == "switched_on" else "cannot say whether encryption is on"
            assert reason and words in reason and "nothing is written in clear" in reason, (case, key, reason)
            assert made == reason and _SECRET not in made
            assert manager.enabled is False and manager.algorithm == "none"
            assert manager.has_key is (key == "a_key_loads" and case != "switched_on"), (case, key)
            assert manager.encrypt("") == "" and manager.encrypt(sealed_elsewhere) == sealed_elsewhere
            assert len(said) == 1 and reason in said[0], ("said once when built", said)
            assert status.get("refusing") == reason and status["enabled"] is False
            assert status["config_enabled"] is (case == "switched_on") and toggle is (True if case == "switched_on"
                                                                                       else None)
            assert _reason(enc) == reason
            with pytest.raises(_unavailable(enc)):
                enc.encrypt_field(_SECRET)
            if manager.has_key:
                assert manager.decrypt(sealed_elsewhere) == _SECRET, "a key that loads still reads"
    finally:
        _closed(config)
        restore()


# ---------------------------------------------------------------------------
# EF2 -- told on, a failing cipher, and the setup methods
# ---------------------------------------------------------------------------
def test_ef2_a_manager_told_on_without_a_key_or_a_failing_cipher_refuses_and_setup_leaves_the_refusal(
        tmp_path, monkeypatch):
    enc, config, restore = _encryption(tmp_path, monkeypatch)
    try:
        _state(enc, config, monkeypatch, "switched_on", None)
        told = enc.EncryptionManager(enabled=True)
        kind, made = _sealing(enc, told)
        assert kind == "refused" and "no key loads" in made, ("told on with no key", kind)
        off = enc.EncryptionManager(enabled=False)
        assert _sealing(enc, off) == ("in clear", _SECRET), "told off: off, whatever the file says"
        given = enc.EncryptionManager(key=_KEY, enabled=True)
        assert _reason(enc, given) is None and given.decrypt(given.encrypt(_SECRET)) == _SECRET

        backend = enc._CRYPTO_BACKEND
        enc._CRYPTO_BACKEND = None
        try:
            bare = (enc.EncryptionManager(key=_KEY, enabled=True), enc.EncryptionManager(key=_KEY))
        finally:
            enc._CRYPTO_BACKEND = backend
        for manager in bare:
            kind, made = _sealing(enc, manager)
            assert kind == "refused" and "no AES library is installed" in made, ("no cipher library", kind, made)

        real = enc.encrypt_bytes

        def broken(key, plaintext):
            raise ValueError("the cipher is down")

        enc.encrypt_bytes = broken
        try:
            kind, made = _sealing(enc, given)
        finally:
            enc.encrypt_bytes = real
        assert kind == "refused", ("a failing cipher handed back", kind)
        assert "the cipher failed (ValueError)" in made and _SECRET not in made
        assert _reason(enc, given) == made and _sealing(enc, given) == ("refused", made), "refusing from then on"
        enc.encrypt_bytes = broken
        try:
            dead = enc.EncryptionManager(key=_KEY, enabled=True)
        finally:
            enc.encrypt_bytes = real
        assert _sealing(enc, dead)[0] == "refused" and "the cipher failed (ValueError)" in (_reason(enc, dead) or ""), \
            "a cipher that does not seal a fixed value refuses from the start"
        fresh = enc.EncryptionManager(key=_KEY, enabled=True)
        with pytest.raises(UnicodeEncodeError):
            fresh.encrypt("a lone surrogate \ud800 in a turn")
        assert _reason(enc, fresh) is None and _sealing(enc, fresh)[0] == "sealed", "no text is no cipher failure"

        order = []

        class _Traced(enc.EncryptionManager):
            def __setattr__(self, name, value):
                if name in ("_key", "_enabled", "_refusing"):
                    order.append(name)
                super().__setattr__(name, value)

        for way in ("random", "passphrase", "existing"):
            enc._DEFAULT_KEYFILE.unlink(missing_ok=True)
            monkeypatch.delenv("OPTI_ENCRYPTION_KEY", raising=False)
            manager = _Traced()
            assert _sealing(enc, manager)[0] == "refused", way
            order.clear()
            if way == "random":
                assert manager.setup_random_key() is True
            elif way == "passphrase":
                assert manager.setup_from_passphrase("a passphrase typed into setup") is True
            else:
                monkeypatch.setenv("OPTI_ENCRYPTION_KEY", _ENV_KEY)
                assert manager.enable_with_existing_key() is True
            assert order == ["_key", "_enabled", "_refusing"], (way, "the reason cleared last", order)
            sealed = manager.encrypt(_SECRET)
            assert _reason(enc, manager) is None and manager.enabled is True
            assert manager.get_status()["refusing"] is None
            assert sealed.startswith("ENC2:") and manager.decrypt(sealed) == _SECRET, way

        enc._DEFAULT_KEYFILE.unlink(missing_ok=True)
        _write(config, "switched_off")
        manager = enc.EncryptionManager()
        assert manager.setup_random_key() is True
        manager.follow()
        assert _sealing(enc, manager)[0] == "sealed", "the key file setup wrote is no change to follow"
    finally:
        _closed(config)
        restore()


# ---------------------------------------------------------------------------
# EF3 -- the stores over a refusing manager
# ---------------------------------------------------------------------------
_ON_WORDS = ("quokka", "wombat", "axolotl", "capybara", "pangolin", "narwhal", "okapi", "aardvark")
_REFUSED_WORDS = ("tapir", "dugong", "manatee", "kinkajou", "fossa", "gharial", "saola", "caracal", "jerboa")


def _synced(conversation_id, title, words):
    return {"conversation": {"id": conversation_id, "title": title, "title_source": "manual",
                             "messages": [{"role": "user", "content": words, "timestamp": "2026-11-02T05:00:00"}]}}


@pytest.mark.parametrize("refusal", list(_REFUSALS))
def test_ef3_the_stores_write_nothing_in_clear_over_a_refusing_manager_and_the_routes_say_why(
        tmp_path, monkeypatch, refusal):
    world = _stores(tmp_path, monkeypatch)
    try:
        enc, store, memory, routes = world.enc, world.store, world.memory, world.routes
        _state(enc, world.config, monkeypatch, "switched_on", _ENV_KEY)
        titled = store.create_conversation(title="Quokka notes").id
        assert store.add_message(titled, "user", "capybara wallows daily", origin="typed") is not None
        drawn = store.create_conversation().id
        assert store.add_message(drawn, "user", "wombat burrow digging tips", origin="typed") is not None
        assert store.rename_conversation(titled, "Axolotl pond") is True
        fact = memory.add_fact("pangolin scales are keratin", "context")
        assert fact is not None
        assert store.apply_synced_conversation(_synced("peer-on", "Narwhal tusk", "narwhal tusk facts")) is True
        refined = store.create_conversation().id
        assert store.add_message(refined, "user", "okapi forest stripes", origin="typed") is not None
        witness = store.title_state(refined)
        held = store.title_state(drawn)
        assert witness is not None and held is not None, "witness: the rule's titles are there to refine"
        assert store.commit_refined_title(refined, witness[0], witness[1], "Okapi forest") is True
        pending = store.create_conversation().id
        assert store.add_message(pending, "user", "aardvark termite feast", origin="typed") is not None
        conn = sqlite3.connect(store._db_path)
        try:
            # A conversation older than the rule: its placeholder not yet settled, for the titles tended at start.
            conn.execute("UPDATE conversations SET title = ?, title_source = 'default', title_basis = '', "
                         "title_turn = '' WHERE id = ?", ("New conversation", pending))
            conn.commit()
        finally:
            conn.close()
        assert _in_clear(world.data, _ON_WORDS) == set(), "on: every field sealed"

        case, key = _REFUSALS[refusal]
        _state(enc, world.config, monkeypatch, case, key)
        before = _rows(world.data)
        with pytest.raises(_unavailable(enc)):
            store.create_conversation(title="Tapir trail")
        with pytest.raises(_unavailable(enc)):
            store.rename_conversation(titled, "Dugong meadow")
        assert store.add_message(drawn, "user", "manatee seagrass grazing", origin="typed") is None
        assert store.commit_refined_title(drawn, held[0], held[1], "Kinkajou honey") is False
        assert store.apply_synced_conversation(_synced("peer-refused", "Fossa hunt", "fossa hunts lemurs")) is False
        assert store.tend_titles() == 0
        assert memory.add_fact("gharial snout is narrow", "context") is None
        with pytest.raises(_unavailable(enc)):
            memory.update_fact(fact.id, new_fact="saola mountain sightings")
        with pytest.raises(_unavailable(enc)):
            store.use_automatic_title(titled)
        assert _rows(world.data) == before, "a refused write changes nothing"
        reason = _reason(enc)
        assert reason, refusal
        for call in (lambda: routes.create_conversation(routes.ConversationCreate(title="Caracal ears")),
                     lambda: routes.rename_conversation(titled, routes.ConversationRename(title="Caracal ears")),
                     lambda: routes.use_automatic_title(titled)):
            with pytest.raises(routes.HTTPException) as answered:
                call()
            assert answered.value.status_code == 503 and answered.value.detail == f"Not saved: {reason}"
        building = enc.get_encryption_manager

        def unbuildable():
            raise RuntimeError("a held secret was wiped")

        enc.get_encryption_manager = unbuildable
        try:
            with pytest.raises(_unavailable(enc)) as raised:
                store.use_automatic_title(titled)
        finally:
            enc.get_encryption_manager = building
        assert "cannot be built (RuntimeError)" in str(raised.value)
        assert _rows(world.data) == before, "a refused route changes nothing"
        assert _in_clear(world.data, _ON_WORDS + _REFUSED_WORDS) == set(), (refusal, "nothing in clear")

        _state(enc, world.config, monkeypatch, "switched_off", None)
        store.create_conversation(title="Jerboa leaps")
        assert _in_clear(world.data, _ON_WORDS + _REFUSED_WORDS) == {"jerboa"}, "off: written as before"
    finally:
        _closed(world.config)
        world.restore()


# ---------------------------------------------------------------------------
# EF4 -- the TOTP secret
# ---------------------------------------------------------------------------
def _secret_at_rest(tmp_path, monkeypatch, refusal):
    """The two-factor store's part of EF4: the secret stored as the manager's state says, or refused."""
    ctx, enc, config = _two_factor(tmp_path, monkeypatch)
    try:
        two_factor = ctx.mod.two_factor_manager
        _state(enc, config, monkeypatch, "switched_off", None)
        off = two_factor.totp_setup("uid-off")
        assert off["success"] and _stored_secrets(ctx.recorder, "uid-off")[0].startswith("B64:"), "off: as before"
        _state(enc, config, monkeypatch, "switched_on", _ENV_KEY)
        on = two_factor.totp_setup("uid-on")
        (stored,) = _stored_secrets(ctx.recorder, "uid-on")
        assert on["success"] and stored.startswith("ENC2:") and on["secret"] not in stored
        assert ctx.mod._decrypt_secret(stored) == on["secret"]

        case, key = _REFUSALS[refusal]
        _state(enc, config, monkeypatch, case, key)
        try:
            two_factor.totp_setup("uid-refused")
        except _unavailable(enc) as exc:
            raised = str(exc)
        else:
            raised = None
        assert _stored_secrets(ctx.recorder, "uid-refused") == [], ("stored", _stored_secrets(ctx.recorder,
                                                                                                "uid-refused"))
        assert raised == _reason(enc, enc.EncryptionManager()) and raised
        if key is not None:
            assert ctx.mod._decrypt_secret(stored) == on["secret"], "a key that loads still reads the secret"
        for broken in ("the cipher", "the manager"):
            _state(enc, config, monkeypatch, "switched_on", _ENV_KEY)
            real = (enc.encrypt_bytes, enc.get_encryption_key)

            def fails(*args, _broken=broken, **kwargs):
                raise RuntimeError(f"{_broken} is down")

            if broken == "the cipher":
                enc.encrypt_bytes = fails
            else:
                enc.get_encryption_key = fails
            try:
                with pytest.raises(_unavailable(enc)):
                    two_factor.totp_setup("uid-broken")
            finally:
                enc.encrypt_bytes, enc.get_encryption_key = real
            assert _stored_secrets(ctx.recorder, "uid-broken") == [], (broken, "nothing stored")
    finally:
        _closed(config)
        ctx.restore()


def _setup_route(tmp_path, monkeypatch, refusal):
    """The route's part of EF4: 503 with the reason where the manager refuses, the secret where it is on."""
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)
    loaded, restore = isolate(
        targets={_SECRETS: source("process_secrets.py"), _ENC: source("encryption.py"),
                 "opti_oignon.api.routes_security": source("api", "routes_security.py")},
        blocked=("opti_oignon.secure_bytes",),
        packages=("opti_oignon.api",),
    )
    enc, routes = loaded[_ENC], loaded["opti_oignon.api.routes_security"]
    config = _home(enc, tmp_path / "route")
    try:
        routes.TWO_FA_AVAILABLE = True
        routes._account_password = lambda *args: "uid-owner"
        routes.two_factor_manager = SimpleNamespace(
            get_status=lambda user_id: SimpleNamespace(totp_verified=False),
            totp_setup=lambda user_id, label=None: {"success": True,
                                                    "secret": enc.EncryptionManager().encrypt("JBSWY3DPEHPK3PXP")})

        def answer():
            call = routes.totp_setup(routes.StepUp(password="the owner's password"), SimpleNamespace(),
                                     current={"user_id": "uid-owner"}, account=SimpleNamespace(username="owner"))
            try:
                return 200, asyncio.run(call)
            except routes.HTTPException as exc:
                return exc.status_code, exc.detail

        _state(enc, config, monkeypatch, "switched_on", _ENV_KEY)
        status, body = answer()
        assert status == 200 and body["secret"].startswith("ENC2:"), "witness: the route answers on"
        _state(enc, config, monkeypatch, *_REFUSALS[refusal])
        assert answer() == (503, f"TOTP not set up: {_reason(enc, enc.EncryptionManager())}"), refusal
    finally:
        _closed(config)
        restore()


@pytest.mark.parametrize("refusal", list(_REFUSALS))
def test_ef4_a_totp_secret_is_sealed_when_on_base64_only_when_off_and_refused_when_the_manager_refuses(
        tmp_path, monkeypatch, refusal):
    _secret_at_rest(tmp_path, monkeypatch, refusal)
    _setup_route(tmp_path, monkeypatch, refusal)


# ---------------------------------------------------------------------------
# EF5 -- the chat says it before it answers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state", ["no_key", "cannot_say", "no_cipher", "cipher_failed", "cannot_be_asked", "on",
                                   "off", "no_store"])
@pytest.mark.parametrize("socket", ["stream", "retry"])
def test_ef5_the_chat_sockets_refuse_a_turn_the_store_could_not_save_before_reading_it(tmp_path, monkeypatch,
                                                                                      socket, state):
    rc, enc, config, restore = _chat(tmp_path, monkeypatch)
    try:
        if state in _REFUSALS:
            _state(enc, config, monkeypatch, *_REFUSALS[state])
        elif state in ("on", "no_cipher", "cipher_failed"):
            _state(enc, config, monkeypatch, "switched_on", _ENV_KEY)
        elif state == "no_store":
            _state(enc, config, monkeypatch, *_REFUSALS["no_key"])
        else:
            _state(enc, config, monkeypatch, "switched_off", None)
        if state == "no_cipher":
            monkeypatch.setattr(enc, "_CRYPTO_BACKEND", None)
        if state == "cipher_failed":
            def failing(key, plaintext):
                raise RuntimeError("the cipher is down")

            assert enc.encrypt_field("a first answer").startswith("ENC2:"), "witness: built on, the cipher sealing"
            monkeypatch.setattr(enc, "encrypt_bytes", failing)
            # An earlier turn's save, refused after its answer.
            with pytest.raises(_unavailable(enc)):
                enc.encrypt_field("an earlier answer")
        if state == "cannot_be_asked":
            def unbuildable():
                raise RuntimeError("the held key was wiped")

            enc.get_encryption_manager = unbuildable
        store = ut_t._Store()
        rc.CONVERSATION_AVAILABLE = state != "no_store"
        rc.conversation_manager = store if state != "no_store" else None
        rc._emergency_stop = None
        streamed = []

        async def spy(websocket, conversation_id, message, request):
            streamed.append(message)

        rc._stream_response = spy
        ws = _Socket({"message": "a typed turn", "conversation_id": "conv-1"} if socket == "stream"
                     else {"conversation_id": "conv-1"})
        asyncio.run((rc.chat_stream if socket == "stream" else rc.chat_retry)(ws))
        errors = [frame.get("content") for frame in ws.sent if frame.get("type") == "error"]
        if state in ("no_key", "cannot_say", "no_cipher", "cipher_failed", "cannot_be_asked"):
            assert ws.read == 0 and streamed == [] and store.saved == [], (socket, state, "the turn was read", ws.sent)
            expected = (f"Not saved, so not run: {_reason(enc)}" if state != "cannot_be_asked"
                        else "Not saved, so not run: the field encryption cannot be asked (RuntimeError)")
            assert errors == [expected], (socket, state, ws.sent)
        else:
            assert ws.read == 1 and not any("Not saved" in str(error) for error in errors), (socket, state, ws.sent)
    finally:
        _closed(config)
        restore()


# ---------------------------------------------------------------------------
# EF6 -- the manager follows its files while the platform runs
# ---------------------------------------------------------------------------
def _bounded(call):
    """``call``'s outcome, on a thread given ten seconds: ("raised", exc), ("gave", value) or ("hung", None)."""
    outcome = []

    def run():
        try:
            outcome.append(("gave", call()))
        except BaseException as exc:  # noqa: BLE001 - the outcome is what is checked
            outcome.append(("raised", exc))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(10)
    return outcome[0] if outcome else ("hung", None)


def test_ef6_the_manager_follows_security_yaml_and_the_key_file_while_the_platform_runs(tmp_path, monkeypatch):
    enc, config, restore = _encryption(tmp_path, monkeypatch)
    try:
        _state(enc, config, monkeypatch, "switched_off", None)
        assert enc.encrypt_field(_SECRET) == _SECRET, "built off"
        held = enc._encryption_manager

        _write(config, "switched_on")
        with pytest.raises(_unavailable(enc)) as raised:
            enc.encrypt_field(_SECRET)
        reason = _reason(enc)
        assert reason and "no key loads" in reason and str(raised.value) == reason, ("turned on, no key", reason)
        assert enc.get_encryption_manager().get_status()["refusing"] == reason

        enc.save_keyfile(_KEY, kdf_name="random", allow_unprotected=True)
        sealed = enc.encrypt_field(_SECRET)
        assert sealed.startswith("ENC2:") and enc.decrypt_field(sealed) == _SECRET and _reason(enc) is None, \
            "a key file put back"

        _write(config, "not_YAML")
        with pytest.raises(_unavailable(enc)):
            enc.encrypt_field(_SECRET)
        assert "cannot say" in _reason(enc)
        _write(config, "switched_on")
        assert enc.encrypt_field(_SECRET).startswith("ENC2:") and _reason(enc) is None, "mended, the key held"

        real = enc.encrypt_bytes

        def down(key, plaintext):
            raise RuntimeError("the cipher is down")

        enc.encrypt_bytes = down
        try:
            with pytest.raises(_unavailable(enc)):
                enc.encrypt_field(_SECRET)
            kept = _reason(enc)
            assert kept and "the cipher failed (RuntimeError)" in kept, kept
            _write(config, "switched_on")
            with pytest.raises(_unavailable(enc)):
                enc.encrypt_field(_SECRET)
            assert _reason(enc) == kept, "an unrelated save does not end a refusal while the cipher still fails"
        finally:
            enc.encrypt_bytes = real
        _write(config, "switched_on")
        assert enc.encrypt_field(_SECRET).startswith("ENC2:") and _reason(enc) is None, "the cipher sealing again"

        keyfile = enc._DEFAULT_KEYFILE
        _write(config, "switched_off")
        assert enc.encrypt_field(_SECRET) == _SECRET
        keyfile.unlink()
        _write(config, "switched_on")
        with pytest.raises(_unavailable(enc)):
            enc.encrypt_field(_SECRET)
        assert "the key in use came from is gone" in (_reason(enc) or ""), ("the key file gone", _reason(enc))
        enc.save_keyfile(_KEY, kdf_name="random", allow_unprotected=True)
        assert enc.encrypt_field(_SECRET).startswith("ENC2:") and _reason(enc) is None, "the key file put back"
        keyfile.unlink()
        enc.save_keyfile(bytes(range(1, 33)), kdf_name="random", allow_unprotected=True)
        with pytest.raises(_unavailable(enc)):
            enc.encrypt_field(_SECRET)
        assert "holds another key" in (_reason(enc) or ""), ("another key", _reason(enc))
        keyfile.unlink()
        enc.save_keyfile(_KEY, kdf_name="random", allow_unprotected=True)
        assert enc.encrypt_field(_SECRET).startswith("ENC2:") and _reason(enc) is None, "its own key back"

        keyfile.unlink()
        os.mkfifo(keyfile)
        try:
            loaded = _bounded(enc.get_encryption_key)
            sealing = _bounded(lambda: enc.encrypt_field(_SECRET))
        finally:
            _release(keyfile)
            keyfile.unlink()
        assert loaded == ("gave", None), ("a FIFO at the key file's name", loaded)
        assert sealing[0] == "gave" and sealing[1].startswith("ENC2:"), ("the held key stands", sealing)
        enc.save_keyfile(_KEY, kdf_name="random", allow_unprotected=True)

        _write(config, "switched_off")
        assert enc.encrypt_field(_SECRET) == _SECRET, "turned off: as told"

        config.unlink()
        os.mkfifo(config)
        try:
            kind, value = _bounded(lambda: enc.encrypt_field(_SECRET))
        finally:
            _release(config)
            config.unlink()
        assert kind == "raised" and isinstance(value, _unavailable(enc)) and "cannot say" in str(value), \
            ("a FIFO in the file's place", kind, value)
        assert enc.get_encryption_manager() is held, "one manager, followed"

        _write(config, "switched_off")
        told = enc.EncryptionManager(enabled=False)
        _write(config, "switched_on")
        told.follow()
        assert _sealing(enc, told) == ("in clear", _SECRET), "a manager told its state does not follow"

        enc._encryption_manager = None

        def wiped():
            raise RuntimeError("a held secret was wiped")

        enc.get_encryption_key = wiped
        with pytest.raises(_unavailable(enc)) as raised:
            enc.encrypt_field(_SECRET)
        assert "cannot be built (RuntimeError)" in str(raised.value)
    finally:
        _closed(config)
        restore()
