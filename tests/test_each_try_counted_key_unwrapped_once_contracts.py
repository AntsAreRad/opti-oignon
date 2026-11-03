#!/usr/bin/env python3
"""Contracts for the tries a password or a code is checked under, and the key file the platform signs with.

The login asked its limiter, ran bcrypt, then counted the failure: tries sent together all passed the limiter before
the first failure was counted, as many as the thread pool runs. The second step read whether its challenge was locked,
checked the code, then counted the try, and the account's TOTP and recovery windows were read and written back
unguarded: tries sent together were each checked. And every load of a wrapped key file derived its wrapping key again
(Argon2id over 64 MiB, or PBKDF2) -- every status of the security mode, every mode read, every downgrade and approval,
most of them on the event loop, which stopped every request meanwhile.

  * TQ1 -- a login that finds a password being checked for its address or its username is refused (429,
    ``Retry-After: 1``) before its password is read, and is not counted; a login of another address and another
    account is checked meanwhile.
  * TQ2 -- the login's claim is let go however the check ends: a right password, a wrong one, a check that raises,
    the limiter's refusal; each try asks the limiter, then the password, then counts its failure, one after the
    other to the limiter's refusal.
  * TQ3 -- a second-step try is counted before its code is read: of six sent together at one challenge, five are
    read, each already counted, and the sixth is refused unread; the try that reaches the cap is still read, and a
    right one consumes the challenge; an enrolment's tries are counted the same way; an answer that is not there
    (an empty code, an offered security key's missing answer) is no try.
  * TQ4 -- the account's TOTP and recovery windows are read, checked and written back under one lock: two tries that
    meet at the window's read are counted one after the other, and the one past the window reads no code; the
    windows are forgotten under their own locks when the account's factors are purged.
  * TQ5 -- ``encryption.load_keyfile`` unwraps a wrapped key file once per process for each text and the passphrase it
    holds, keeping one text at a time and no digest of a passphrase: each call has a key of its own; a passphrase
    given as an argument is never kept; another text or held passphrase derives again; the held passphrase refused is
    refused again without a derivation; the slot a new text takes is wiped; a slot wiped is unwrapped again; a load
    during the slot's replacement is served what it held; a file removed is refused as before.
  * TQ6 -- once the key file is unwrapped, the security mode's state and status, its mode read, its key state and
    the allowlist's key derive nothing.
  * TQ7 -- the texts say it.
  * TQ8 -- of two right TOTP codes sent together, one signs in: the step a code consumes is written only over an
    earlier one, so the same code twice signs in once, and an earlier step -- a code read together, or the setup's
    confirmation of an active TOTP -- never brings the later code back.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_auth_2fa_routes_seam_contracts as seam  # noqa: E402
import test_signing_key_contracts as sk  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_tq1_a_login_that_finds_another_being_checked_is_refused_unread": 5.0,
    "test_tq2_the_login_s_claim_is_let_go_however_the_check_ends": 5.0,
    "test_tq3_a_second_step_try_is_counted_before_its_code_is_read": 5.0,
    "test_tq4_the_account_s_windows_count_tries_that_meet_one_after_the_other": 5.0,
    "test_tq5_a_wrapped_key_file_is_unwrapped_once_for_each_text_and_passphrase": 15.0,
    "test_tq6_the_mode_and_the_allowlist_derive_nothing_once_the_key_is_unwrapped": 15.0,
    "test_tq7_the_texts_say_it": 5.0,
    "test_tq8_of_two_right_codes_sent_together_one_signs_in": 5.0,
}

_BUSY = "A login for this account or from this address is being checked. Try again."
_LIMITED = "Too many login attempts. Please try again later."
_LOCKED = "Too many 2FA attempts. Please log in again."
K2 = bytes(range(101, 133))


class _Limiter:
    """The login limiter's calls, in one log with the password checks; it refuses an address or an account once
    ``cap`` failures of it are in."""

    def __init__(self, cap=5):
        self.log, self.cap, self.lock = [], cap, threading.Lock()

    def check_rate_limit(self, ip, username):
        with self.lock:
            self.log.append(("check", ip, username))
            failed = [e for e in self.log if e[0] == "failure" and (e[1] == ip or e[2] == username)]
            return (True, 0) if len(failed) < self.cap else (False, 60)

    def record_failure(self, ip, username):
        with self.lock:
            self.log.append(("failure", ip, username))

    def record_success(self, ip, username):
        with self.lock:
            self.log.append(("success", ip, username))


def _login(ctx, username, password, ip):
    """``(status, body, headers)`` of one login from ``ip``."""
    from fastapi import HTTPException

    try:
        out = ctx.routes.login(ctx.routes.LoginRequest(username=username, password=password),
                               seam._Request(host=ip), seam._Response())
    except HTTPException as exc:
        return exc.status_code, {"detail": exc.detail}, {}
    if hasattr(out, "body"):
        return out.status_code, json.loads(out.body), dict(out.headers)
    return 200, out.model_dump() if hasattr(out, "model_dump") else out, {}


def _second_step(ctx, challenge_id, code, method="totp"):
    from fastapi import HTTPException

    try:
        out = ctx.routes.login_2fa(ctx.routes.TwoFALoginRequest(challenge_id=challenge_id, code=code, method=method),
                                   seam._Request(), seam._Response())
    except HTTPException as exc:
        return exc.status_code, exc.detail
    return 200, out


def _enrolment_step(ctx, challenge_id, code):
    from fastapi import HTTPException

    try:
        out = ctx.routes.login_2fa_enrol_complete(
            ctx.routes.EnrolCompleteRequest(challenge_id=challenge_id, code=code), seam._Request(), seam._Response())
    except HTTPException as exc:
        return exc.status_code, exc.detail
    return 200, out


def _together(calls, inside, count, go):
    """Run ``calls`` together; once ``count`` of them are inside the check (``inside()``) or done, let them go."""
    with ThreadPoolExecutor(len(calls)) as pool:
        futures = [pool.submit(call) for call in calls]
        deadline = time.monotonic() + 10
        while inside() + sum(f.done() for f in futures) < count and time.monotonic() < deadline:
            time.sleep(0.005)
        go.set()
        return [f.result(10) for f in futures]


def test_tq1_a_login_that_finds_another_being_checked_is_refused_unread():
    ctx = seam._load()
    limiter = _Limiter()
    ctx.routes._get_rate_limiter = lambda: limiter
    entered, release = threading.Event(), threading.Event()
    held = {"guess-1", "guess-3"}

    def authenticate(username, password):
        with limiter.lock:
            limiter.log.append(("password", username, password))
        if password in held:
            entered.set()
            assert release.wait(10), "the held check was never let go"
        return None

    ctx.manager.authenticate = authenticate
    pool = ThreadPoolExecutor(2)
    try:
        first = pool.submit(_login, ctx, "alice", "guess-1", "10.0.0.1")
        assert entered.wait(10)
        # The same address for another account, the same account from another address: refused before the limiter
        # is asked or the password read, and counted nowhere.
        for username, ip in (("bob", "10.0.0.1"), ("alice", "10.0.0.2")):
            status, body, headers = _login(ctx, username, "guess-2", ip)
            assert (status, body, headers.get("retry-after")) == (429, {"detail": _BUSY}, "1"), (username, ip)
        assert limiter.log == [("check", "10.0.0.1", "alice"), ("password", "alice", "guess-1")], limiter.log
        # Another address and another account: checked while the first is.
        entered.clear()
        other = pool.submit(_login, ctx, "carol", "guess-3", "10.0.0.3")
        assert entered.wait(10)
        assert ("password", "carol", "guess-3") in limiter.log and not first.done()
        release.set()
        assert first.result(10)[0] == 401 and other.result(10)[0] == 401
        assert sorted(limiter.log) == sorted([
            ("check", "10.0.0.1", "alice"), ("password", "alice", "guess-1"), ("failure", "10.0.0.1", "alice"),
            ("check", "10.0.0.3", "carol"), ("password", "carol", "guess-3"), ("failure", "10.0.0.3", "carol")])
        assert ctx.routes._logging_in == set()
    finally:
        release.set()
        pool.shutdown(wait=True)
        ctx.restore()


def test_tq2_the_login_s_claim_is_let_go_however_the_check_ends():
    ctx = seam._load()
    limiter = _Limiter(cap=3)
    ctx.routes._get_rate_limiter = lambda: limiter
    real = ctx.manager.authenticate

    def authenticate(username, password):
        with limiter.lock:
            limiter.log.append(("password", username, password))
        if password == "raises":
            raise RuntimeError("the account store cannot be read")
        return real(username, password)

    ctx.manager.authenticate = authenticate
    try:
        assert _login(ctx, seam._RESIDENT, seam._RIGHT_PASSWORD, "10.0.0.1")[0] == 200
        assert ctx.routes._logging_in == set(), "a right password"
        assert _login(ctx, seam._RESIDENT, "wrong", "10.0.0.1")[0] == 401
        assert ctx.routes._logging_in == set(), "a wrong password"
        with pytest.raises(RuntimeError):
            _login(ctx, seam._RESIDENT, "raises", "10.0.0.1")
        assert ctx.routes._logging_in == set(), "a check that raises"
        # One try after the other, each asks the limiter, then the password, then counts its failure, to the
        # limiter's refusal, which reads no password.
        limiter.log.clear()
        for n in range(3):
            assert _login(ctx, "dora", f"guess-{n}", "10.0.0.4")[0] == 401
        assert limiter.log == [entry for n in range(3) for entry in (
            ("check", "10.0.0.4", "dora"), ("password", "dora", f"guess-{n}"), ("failure", "10.0.0.4", "dora"))]
        status, body, _headers = _login(ctx, "dora", "guess-3", "10.0.0.4")
        assert (status, body) == (429, {"detail": _LIMITED})
        assert limiter.log[-1] == ("check", "10.0.0.4", "dora")
        assert ctx.routes._logging_in == set(), "the limiter's refusal"
        assert _login(ctx, seam._RESIDENT, seam._RIGHT_PASSWORD, "10.0.0.5")[0] == 200
    finally:
        ctx.restore()


def _meeting_at_the_read(store, parties):
    """The first ``parties`` reads of a challenge wait a moment for each other, once read: tries sent together all
    find it unlocked before any of them counts."""
    real_get, barrier, lock, calls = store.get, threading.Barrier(parties, timeout=2), threading.Lock(), [0]

    def get(challenge_id):
        value = real_get(challenge_id)
        with lock:
            calls[0] += 1
            meet = calls[0] <= parties
        if meet:
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
        return value

    store.get = get


def test_tq3_a_second_step_try_is_counted_before_its_code_is_read():
    ctx = seam._load()
    try:
        secret = seam._enroll(ctx)
        mgr = ctx.tfa.two_factor_manager
        store = ctx.routes._challenge_store
        challenge_id = seam._password_step(ctx)["challenge_id"]
        real_validate = mgr.validate_2fa
        go, lock, read = threading.Event(), threading.Lock(), []

        def validate(user_id, code, method):
            with lock:
                read.append(store.get(challenge_id)["attempts"])
            assert go.wait(10)
            return {"success": False}

        mgr.validate_2fa = validate
        _meeting_at_the_read(store, 6)
        answers = _together([lambda: _second_step(ctx, challenge_id, "111111")] * 6, lambda: len(read), 6, go)
        assert len(read) == 5 and min(read) >= 1, read
        assert sorted(detail for _status, detail in answers) == sorted(
            [f"Invalid 2FA code. {n} attempt(s) remaining." for n in (4, 3, 2, 1)] + [_LOCKED, _LOCKED])
        assert {status for status, _detail in answers} == {403}
        assert _second_step(ctx, challenge_id, "222222") == (403, _LOCKED) and len(read) == 5
        del store.get

        # The try that reaches the cap is still read, and a right one consumes the challenge.
        mgr.validate_2fa = real_validate
        ctx.clock.advance(30)
        challenge_id = seam._password_step(ctx)["challenge_id"]
        for n in (4, 3, 2, 1):
            assert _second_step(ctx, challenge_id, "111111") == (403, f"Invalid 2FA code. {n} attempt(s) remaining.")
        right = seam._totp_code(secret, int(ctx.clock.time() // 30))
        assert _second_step(ctx, challenge_id, right)[0] == 200
        assert store.get(challenge_id) is None

        # An answer that is not there is no try: empty codes, an offered security key's missing answer leave the
        # challenge open and its count as it was.
        challenge_id = store.create(seam._RESIDENT_ID, ["totp", "webauthn"])
        for _ in range(5):
            assert _second_step(ctx, challenge_id, " ") == (400, "A 2FA code is required for this method.")
            assert _second_step(ctx, challenge_id, "", "webauthn") == (
                400, "A security key's answer is required for this method.")
        assert store.get(challenge_id)["attempts"] == 0
        assert _second_step(ctx, challenge_id, "111111") == (403, "Invalid 2FA code. 4 attempt(s) remaining.")

        # An enrolment's tries, counted the same way (the account's state read without the seam's one connection).
        mgr.get_status = lambda user_id: types.SimpleNamespace(any_method_active=False)
        enrol = store.create("uid-enrolling", ["totp"], enrol=True)
        go, read = threading.Event(), []

        def verify(user_id, code):
            with lock:
                read.append(store.get(enrol)["attempts"])
            assert go.wait(10)
            return {"success": False, "error": "invalid_code"}

        mgr.totp_verify = verify
        _meeting_at_the_read(store, 6)
        answers = _together([lambda: _enrolment_step(ctx, enrol, "111111")] * 6, lambda: len(read), 6, go)
        assert len(read) == 5 and min(read) >= 1, read
        assert sorted(detail for _status, detail in answers).count(_LOCKED) == 2, answers
        del store.get
        enrol = store.create("uid-enrolling-again", ["totp"], enrol=True)
        for _ in range(5):
            assert _enrolment_step(ctx, enrol, "") == (400, "A 2FA code is required for this method.")
        assert store.get(enrol)["attempts"] == 0
    finally:
        ctx.restore()


class _MeetingWindow(dict):
    """A window whose read, once read, waits a moment for a second reader: two tries that read it unguarded both hold
    what it was before either writes it back."""

    def __init__(self, *args, barrier):
        super().__init__(*args)
        self.barrier = barrier

    def get(self, key, default=None):
        value = super().get(key, default)
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return value


def test_tq4_the_account_s_windows_count_tries_that_meet_one_after_the_other():
    ctx = seam._load()
    try:
        tfa = ctx.tfa
        mgr = tfa.two_factor_manager
        opened = []
        real_conn = tfa._get_2fa_conn

        def conn():
            opened.append(threading.get_ident())
            return real_conn()

        tfa._get_2fa_conn = conn
        now = ctx.clock.time()
        for window, room, check in (("_totp_attempts", tfa.MAX_TOTP_ATTEMPTS_PER_WINDOW, mgr.totp_validate),
                                    ("_recovery_attempts", tfa.MAX_RECOVERY_ATTEMPTS_PER_HOUR,
                                     mgr.validate_recovery_code)):
            setattr(mgr, window, _MeetingWindow({"uid-w": [now] * (room - 1)},
                                                barrier=threading.Barrier(2, timeout=0.5)))
            opened.clear()
            with ThreadPoolExecutor(2) as pool:
                answers = list(pool.map(lambda _n: check("uid-w", "000000"), range(2)))
            assert answers == [False, False], window
            # One try took the last place and read its code; the other found the window full and read none.
            assert len(set(opened)) == 1, (window, len(set(opened)))
            assert len(dict.get(getattr(mgr, window), "uid-w")) == room, window
        # The windows are forgotten under their own locks when the account's factors are purged.
        mgr._totp_attempts["uid-w"] = [now]
        with mgr._attempts_lock:
            purge = threading.Thread(target=mgr.purge_account, args=("uid-w",))
            purge.start()
            purge.join(0.3)
            assert purge.is_alive() and "uid-w" in mgr._totp_attempts, "the purge waits for the windows' lock"
        purge.join(10)
        assert not purge.is_alive() and "uid-w" not in mgr._totp_attempts
        mgr._totp_confirm_failures["uid-w"] = [now]
        with mgr._confirm_lock:
            purge = threading.Thread(target=mgr.purge_account, args=("uid-w",))
            purge.start()
            purge.join(0.3)
            assert purge.is_alive() and "uid-w" in mgr._totp_confirm_failures, "the purge waits for the lock"
        purge.join(10)
        assert not purge.is_alive() and "uid-w" not in mgr._totp_confirm_failures
    finally:
        ctx.restore()


def _key_world(tmp_path, monkeypatch):
    """encryption, the security mode and the allowlist in a window with the real SecureBytes -- which zeroes the bytes
    it is made from --, every path they reach under ``tmp_path``."""
    pytest.importorskip("cryptography")
    for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE", "RAG_HOME"):
        monkeypatch.delenv(name, raising=False)
    names = ("process_secrets", "secure_bytes", "encryption", "security_mode", "plugin_allowlist")
    loaded, restore = isolate(targets={f"opti_oignon.{name}": source(f"{name}.py") for name in names})
    try:
        enc, sm, pa = (loaded[f"opti_oignon.{name}"] for name in ("encryption", "security_mode", "plugin_allowlist"))
        assert enc._SECURE_BYTES_AVAILABLE is True
        data = tmp_path / "data"
        data.mkdir(parents=True)
        keyfile = data / ".keyfile"
        enc._DEFAULT_KEYFILE = sm._DEFAULT_KEYFILE = keyfile
        sm._DATA_DIR, sm._LOCKFILE_PATH, sm._SECURITY_YAML = (data, data / ".security_mode_lock",
                                                              tmp_path / "security.yaml")
        sm._audit_log = lambda event, severity="INFO", **details: None
        pa._DATA_DIR, pa._ALLOWLIST_PATH = data, data / "plugin_allowlist.json"
        return types.SimpleNamespace(enc=enc, sm=sm, pa=pa, sb=loaded["opti_oignon.secure_bytes"], keyfile=keyfile,
                                     restore=restore)
    except BaseException:
        restore()
        raise


def _counting(enc, monkeypatch):
    derived = []
    real = enc.derive_key_from_passphrase

    def derive(*args, **kwargs):
        derived.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(enc, "derive_key_from_passphrase", derive)
    return derived


def _held(key):
    """A key's bytes, or what stands for a key wiped."""
    try:
        return key.as_bytes()
    except RuntimeError:
        return "wiped"


def test_tq5_a_wrapped_key_file_is_unwrapped_once_for_each_text_and_passphrase(tmp_path, monkeypatch):
    w = _key_world(tmp_path, monkeypatch)
    try:
        enc = w.enc
        sk._key_file(w, wrapped=True)
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", sk.PHRASE)
        derived = _counting(enc, monkeypatch)
        first = enc.load_keyfile()[0]
        second = enc.load_keyfile()[0]
        assert first.as_bytes() == second.as_bytes() == sk.K1 and len(derived) == 1
        # Each call has a key of its own: one wiped wipes no other.
        assert first is not second
        first.wipe()
        assert _held(second) == _held(enc.load_keyfile()[0]) == sk.K1 and len(derived) == 1
        second.wipe()
        assert _held(enc.load_keyfile()[0]) == sk.K1 and len(derived) == 1
        # A passphrase given as an argument is never kept: derived at each call, the right one as a mistyped one,
        # and the held passphrase still served from what it unwrapped.
        assert _held(enc.load_keyfile(passphrase=sk.PHRASE)[0]) == sk.K1
        for _ in range(2):
            with pytest.raises(enc.PassphraseRefused):
                enc.load_keyfile(passphrase="a typo of the phrase")
        assert len(derived) == 4
        assert _held(enc.load_keyfile()[0]) == sk.K1 and len(derived) == 4
        # The held passphrase refused is refused again without a derivation.
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", "a typo of the phrase")
        for _ in range(2):
            with pytest.raises(enc.PassphraseRefused):
                enc.load_keyfile()
        assert len(derived) == 5
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", sk.PHRASE)
        assert _held(enc.load_keyfile()[0]) == sk.K1 and len(derived) == 6
        # A byte changed in the blob is another text, derived again and refused.
        text = w.keyfile.read_text(encoding="ascii")
        payload = json.loads(text)
        blob = bytearray(base64.urlsafe_b64decode(payload["blob"]))
        blob[-1] ^= 1
        payload["blob"] = base64.urlsafe_b64encode(bytes(blob)).decode("ascii")
        w.keyfile.write_text(json.dumps(payload) + "\n", encoding="ascii")
        with pytest.raises(enc.PassphraseRefused):
            enc.load_keyfile()
        assert len(derived) == 7
        w.keyfile.write_text(text, encoding="ascii")
        assert _held(enc.load_keyfile()[0]) == sk.K1 and len(derived) == 8
        # A key file wrapped again takes the place of the one it replaced, whose key and passphrase are wiped here.
        replaced = enc._unwrapped_slot
        w.keyfile.unlink()
        enc.save_keyfile(K2, passphrase=sk.PHRASE)
        derived.clear()
        assert _held(enc.load_keyfile()[0]) == K2 and len(derived) == 1
        assert replaced[1].is_wiped and replaced[2].is_wiped
        # What is kept: the digest of the text alone -- never a digest of the passphrase, which would test guesses
        # at a hash's speed --, the passphrase held as a key is, the key.
        digest, held, outcome = enc._unwrapped_slot
        assert digest == hashlib.sha256(w.keyfile.read_text(encoding="ascii").strip().encode("ascii")).digest()
        assert isinstance(held, enc.SecureBytes) and held.as_bytes() == sk.PHRASE.encode("utf-8")
        assert outcome.as_bytes() == K2
        # Every key wiped (at exit, on a signal): unwrapped again.
        w.sb._wipe_all_keys()
        try:
            again = _held(enc.load_keyfile()[0])
        except (RuntimeError, ValueError) as exc:
            again = type(exc).__name__
        assert again == K2 and len(derived) == 2
        # A kept key zeroed and not yet marked wiped -- every key's wipe at exit or on a signal, met mid-copy -- is
        # no key: unwrapped again.
        enc._unwrapped_slot[2]._buf[:] = bytes(32)
        derived.clear()
        assert _held(enc.load_keyfile()[0]) == K2 and len(derived) == 1
        # A load during the slot's replacement is served what the slot held, never a key the replacement zeroes:
        # the passphrase is compared and the key copied under the lock the slot is replaced under.
        paused, release, real_hmac = threading.Event(), threading.Event(), enc.hmac

        def compare(a, b):
            if threading.current_thread().name == "reader":
                paused.set()
                release.wait(5)
            return real_hmac.compare_digest(a, b)

        monkeypatch.setattr(enc, "hmac", types.SimpleNamespace(compare_digest=compare))
        derived.clear()
        served = {}
        reader = threading.Thread(target=lambda: served.update(key=_held(enc.load_keyfile()[0])), name="reader")
        reader.start()
        assert paused.wait(5)
        other = threading.Thread(target=enc._remember_unwrapped,
                                 args=(b"\x00" * 32, sk.PHRASE, enc.SecureBytes(bytearray(sk.K1))))
        other.start()
        other.join(0.3)
        replaced_while_read = not other.is_alive()
        release.set()
        reader.join(5)
        other.join(5)
        assert (served, derived, replaced_while_read) == ({"key": K2}, [], False)
        # A key file removed is refused, whatever was unwrapped.
        w.keyfile.unlink()
        with pytest.raises(FileNotFoundError):
            enc.load_keyfile()
    finally:
        w.restore()


def test_tq6_the_mode_and_the_allowlist_derive_nothing_once_the_key_is_unwrapped(tmp_path, monkeypatch):
    w = _key_world(tmp_path, monkeypatch)
    try:
        sm, manager = w.sm, w.sm.security_mode_manager
        sk._key_file(w, wrapped=True)
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", sk.PHRASE)
        assert manager.initialize_mode(sm.MODE_DAILY, "tester") is True
        derived = _counting(w.enc, monkeypatch)
        for _ in range(3):
            state = manager.get_state()
            assert (state.mode, state.hmac_valid) == (sm.MODE_DAILY, True)
        assert manager.status()["mode"] == sm.MODE_DAILY
        manager.invalidate_cache()
        assert sm.get_current_mode() == sm.MODE_DAILY
        key, unavailable = sm._key_state()
        assert unavailable is None and key.as_bytes() == sk.K1
        key, why = w.pa._key_or_why()
        assert why is None and key.as_bytes() == sk.K1
        assert derived == []
    finally:
        w.restore()


def test_tq8_of_two_right_codes_sent_together_one_signs_in():
    ctx = seam._load()
    try:
        tfa, mgr = ctx.tfa, ctx.tfa.two_factor_manager
        real_decrypt = tfa._decrypt_secret
        meeting = {}

        def decrypt(value):
            """Each try waits here for the other once both have read the last step used, then for those before it
            in ``order`` to end: both read it before either writes it, and they write one after the other."""
            if "barrier" in meeting:
                meeting["read"][threading.current_thread().name].set()
                try:
                    meeting["barrier"].wait()
                except threading.BrokenBarrierError:
                    pass
                order = meeting["order"]
                for before in order[:order.index(threading.current_thread().name)]:
                    meeting["done"][before].wait(5)
            return real_decrypt(value)

        def together(user_id, codes):
            """``codes``, a name for each, sent together in that order; what each answered. Each is started once
            the one before it has read the step (the seam's one connection runs a statement for one thread at a
            time, where the store opens a connection per call)."""
            meeting.update(barrier=threading.Barrier(len(codes), timeout=5), order=[name for name, _ in codes],
                           done={name: threading.Event() for name, _ in codes},
                           read={name: threading.Event() for name, _ in codes})
            results = {}

            def run(name, value):
                results[name] = mgr.totp_validate(user_id, value)
                meeting["done"][name].set()

            threads = [threading.Thread(target=run, args=pair, name=pair[0]) for pair in codes]
            for thread, (name, _code) in zip(threads, codes):
                thread.start()
                assert meeting["read"][name].wait(5), name
            for thread in threads:
                thread.join(10)
            meeting.clear()
            return results

        tfa._decrypt_secret = decrypt
        # The same live code sent twice together.
        secret = seam._enroll(ctx, "uid-twice")
        ctx.clock.advance(30)
        code = seam._totp_code(secret, int(ctx.clock.time() // 30))
        assert together("uid-twice", [("first", code), ("second", code)]) == {"first": True, "second": False}
        assert mgr.totp_validate("uid-twice", code) is False

        # Two right codes, the later step's written first: the earlier one never writes the step back below it,
        # and the later code is not taken again.
        secret = seam._enroll(ctx, "uid-order")
        ctx.clock.advance(60)
        step = int(ctx.clock.time() // 30)
        later, earlier = seam._totp_code(secret, step), seam._totp_code(secret, step - 1)
        assert together("uid-order", [("later", later), ("earlier", earlier)]) == {"later": True, "earlier": False}
        assert mgr.totp_validate("uid-order", later) is False

        # The setup's confirmation of an active TOTP never writes the step back below one a sign-in consumed.
        ctx.clock.advance(30)
        step = int(ctx.clock.time() // 30)
        current, before = seam._totp_code(secret, step), seam._totp_code(secret, step - 1)
        assert mgr.totp_validate("uid-order", current) is True
        assert mgr.totp_verify("uid-order", before)["success"] is True
        assert mgr.totp_validate("uid-order", current) is False
    finally:
        ctx.restore()


def test_tq7_the_texts_say_it():
    security = (REPO / "SECURITY.md").read_text(encoding="utf-8")
    assert ("a login checks one password at a time for an address and for an account, so that tries sent together "
            "are each counted") in " ".join(security.split())
    assert "a code sent at the second step is counted before it is read" in " ".join(security.split())
    encryption = " ".join((REPO / "docs" / "security" / "encryption.md").read_text(encoding="utf-8").split())
    assert ("The backend unwraps a wrapped key file once for each text and the passphrase it was started with, and "
            "keeps that key and that passphrase in memory") in encryption
    assert "A passphrase given to a command is never kept." in encryption
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    fixed = changelog.split("### Fixed\n\n", 1)[1].split("\n### ", 1)[0]
    assert "\n- [SECURITY] Tries sent together are each counted. A login asked the\n" in fixed
    routes = " ".join((REPO / "opti_oignon" / "api" / "routes_auth.py").read_text(encoding="utf-8").split())
    assert "one password checked at a time for each" in routes
    loader = " ".join((REPO / "opti_oignon" / "encryption.py").read_text(encoding="utf-8").split())
    assert ("an envelope opened with the passphrase the process holds is unwrapped once per process for each text "
            "and passphrase") in loader
    tfa = " ".join((REPO / "opti_oignon" / "auth_2fa.py").read_text(encoding="utf-8").split())
    assert ("the step is written only over an earlier one, here and at the setup's confirmation, so of two right "
            "codes sent together one signs in, and no confirmation brings a used code back") in tfa
