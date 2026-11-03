#!/usr/bin/env python3
"""Contracts for the login that, in Bulbe, never opens a session on a password alone (block 225).

``TwoFactorAuthManager.is_2fa_required`` -- every account in Bulbe, in Daily one with a factor active -- had no
caller: the login asked a second factor only of an account that had one, so in Bulbe an account with none signed in
with its password alone, and the registration gave a new account its tokens at once. Refusing those logins would
leave such an account no way to enrol -- the routes that add a factor need a session -- and Bulbe is reached without
the UI too (the configuration, the emergency stop, a mode that cannot be read). So the password gives the challenge
of an enrolment instead, and the account's first session comes with its first factor.

  * EN1 -- in Bulbe, an account with no factor gets no token from its password: the enrolment challenge (``totp``),
    no session made, no cookie, no login audited; in Daily the same account gets its tokens at once, and in Bulbe an
    account with a factor active is asked it, as before.
  * EN2 -- the enrolment completes: its first step begins a TOTP for that account alone, named by its username, and
    may be begun again; a right code activates it, gives the tokens and the recovery codes once, audited
    ``login_2fa_enrolled``; the challenge is spent, the code that confirmed it does not pass the next login's second
    step, and the next login asks the factor. In cookie mode the confirmation sets both session cookies, the
    enrolment answer none.
  * EN3 -- the enrolment challenge is bounded: an unknown or expired one, or one of a second step, never begins or
    completes an enrolment; it is never answered through ``/login/2fa``; wrong codes are counted, the remaining tries
    named, then the challenge is locked and a right code no longer passes; an account allows five wrong confirmation
    codes in five minutes across its logins (429, then a right code passes once the window ended); an account that
    enrolled through its other login since is sent back to log in, its challenge spent and its recovery codes kept --
    and so is a begin whose status was read before that other login's confirmation landed: the confirmed TOTP is
    never replaced, through the settings either (409); a right code checked against a secret another login set up
    again meanwhile activates nothing and sends back to log in; wrong codes sent while one is checked are each counted
    before their code is read; and the five minutes run from the latest begin.
  * EN4 -- the mode that cannot be read is Bulbe: an ``is_bulbe`` that raises, or a mode module without one, asks the
    enrolment; a factor that cannot be set up here (the TOTP library missing) is 503, named, no token, no challenge,
    and so is a secret the field encryption cannot seal; and ``is_2fa_required`` reads the caller's status when given
    one, the mode alone in Bulbe.
  * EN5 -- the registration in Bulbe answers the enrolment, no token, the account made and enrolled through it; in
    Daily it gives tokens at once; a two-factor module that cannot be reached, or a factor that cannot be set up in
    Bulbe, refuses it (503) with no account left behind.
  * EN7 -- in place of vb1 and vb3 of the verification bounds suite: the code that confirms a TOTP is consumed with
    it, never a first use; an unverified enrolment and an unknown user refuse a right code, the next step's code
    validates once verified -- the refusal was the verification flag --, once only inside its window, and the step
    after it again.

The routes run through the second-factor suite's window (``test_two_factor_account_contracts._world``): the real
second-factor module, the real login and security routes behind a FastAPI application, the seam suite's in-memory
store, clock and TOTP library. Nothing reads or writes the platform's data.

Local-only (the public distribution ships no tests).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_two_factor_account_contracts as tf  # noqa: E402

A, B, PW_A, PW_B = tf.A, tf.B, tf.PW_A, tf.PW_B
LOGIN, SECOND, BEGIN, COMPLETE = ("/api/auth/login", "/api/auth/login/2fa", "/api/auth/login/2fa/enrol/begin",
                                  "/api/auth/login/2fa/enrol/complete")
REGISTER = "/api/auth/register"

BUDGET_S = {
    "test_en1_in_bulbe_an_account_without_a_factor_gets_no_token_from_its_password": 4.0,
    "test_en2_the_enrolment_completes_with_the_first_session_and_the_recovery_codes": 4.0,
    "test_en3_the_enrolment_challenge_is_bounded": 4.0,
    "test_en4_a_mode_that_cannot_be_read_is_bulbe_and_a_factor_that_cannot_be_set_up_refuses": 4.0,
    "test_en5_the_registration_in_bulbe_enrols_before_its_first_session": 4.0,
    "test_en7_the_code_that_confirms_a_totp_never_signs_in_again": 2.0,
}


@pytest.fixture
def world():
    """A window of the routes, in Bulbe unless told, two accounts with no factor. One window at a time: making the
    next closes the last, and the last closes at teardown."""
    made = []

    def make(**kwargs):
        while made:
            made.pop().close()
        kwargs.setdefault("bulbe", True)
        made.append(tf._world(**kwargs))
        return made[-1]

    yield make
    while made:
        made.pop().close()


def _login(w, username="alice", password=PW_A):
    return tf._post(w, LOGIN, {"username": username, "password": password})


def _enrolment(w, username="alice", password=PW_A):
    answer = _login(w, username, password)
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body.get("requires_2fa_enrolment") is True, body
    return body["challenge_id"]


def _begin(w, challenge_id):
    return tf._post(w, BEGIN, {"challenge_id": challenge_id})


def _complete(w, challenge_id, code):
    return tf._post(w, COMPLETE, {"challenge_id": challenge_id, "code": code})


def _enrolled(w, username="alice", password=PW_A):
    """Enrol through the login: the secret, and the answer that completed it."""
    challenge_id = _enrolment(w, username, password)
    secret = _begin(w, challenge_id).json()["secret"]
    done = _complete(w, challenge_id, tf._code(w, secret))
    assert done.status_code == 200, done.text
    w.clock.advance(31)  # past the step the confirmation used
    return secret, done


def _totp(w, user_id):
    """Whether the account's TOTP is there and whether it is active: ``None``, 0 (begun) or 1 (active)."""
    conn = w.tfa._get_2fa_conn()
    try:
        row = conn.execute("SELECT verified FROM totp_config WHERE user_id = ?", (user_id,)).fetchone()
    finally:
        conn.close()
    return None if row is None else row[0]


def _secrets_in_turn(w, *secrets):
    """The TOTP library's new secrets, one per setup, in turn (the seam's library gives one alone)."""
    given = iter(secrets)
    w.tfa.pyotp.random_base32 = lambda: next(given)


def _cookies(answer):
    """The names of the cookies an answer sets."""
    return {header.split("=", 1)[0] for header in answer.headers.get_list("set-cookie")}


def _audited(w, event):
    return [entry for entry in w.accounts.audit if len(entry) > 1 and entry[1] == event]


def _with_create_user(w):
    """The account store's ``create_user``, as the registration calls it."""
    made = []

    def create_user(username, password, email=""):
        user_id = f"uid-{username}"
        if any(u.username == username for u in w.accounts.users.values()):
            return None
        w.accounts.add(user_id, username, password)
        made.append(user_id)
        return w.accounts.users[user_id]

    w.accounts.create_user = create_user
    return made


# --------------------------------------------------------------------------- EN1


def test_en1_in_bulbe_an_account_without_a_factor_gets_no_token_from_its_password(world):
    w = world()
    answer = _login(w)
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert set(body) == {"requires_2fa_enrolment", "challenge_id", "methods"}, body
    assert body["requires_2fa_enrolment"] is True and body["methods"] == ["totp"] and body["challenge_id"]
    assert "set-cookie" not in answer.headers, "no cookie"
    assert w.accounts.tokens == {}, "no session made"
    assert _audited(w, "login") == [], "no login audited"

    daily = world(bulbe=False)
    given = _login(daily)
    assert given.status_code == 200 and given.json().get("access_token"), given.text
    assert "requires_2fa_enrolment" not in given.json()

    asked = world()
    tf._enroll_totp(asked, A, PW_A)
    asked.current = {}
    challenge = _login(asked)
    assert challenge.status_code == 200, challenge.text
    assert challenge.json().get("requires_2fa") is True and "requires_2fa_enrolment" not in challenge.json()
    assert asked.accounts.tokens == {}


# --------------------------------------------------------------------------- EN2


def test_en2_the_enrolment_completes_with_the_first_session_and_the_recovery_codes(world):
    w = world()
    _secrets_in_turn(w, "FIRSTSECRETAAAAA", "SECONDSECRETBBBB")
    challenge_id = _enrolment(w)
    first = _begin(w, challenge_id)
    assert first.status_code == 200, first.text
    again = _begin(w, challenge_id)
    assert again.status_code == 200 and again.json()["secret"] != first.json()["secret"], "begun again"
    begun = again.json()
    assert {"secret", "uri"} <= set(begun) and "alice" in begun["uri"], begun
    assert _totp(w, A) == 0, "begun, not active"
    assert _totp(w, B) is None, "for that account alone"

    stale = _complete(w, challenge_id, tf._code(w, first.json()["secret"]))
    assert stale.status_code == 403 and "Invalid 2FA code" in stale.json()["detail"], "the latest begun is checked"
    done = _complete(w, challenge_id, tf._code(w, begun["secret"]))
    assert done.status_code == 200, done.text
    body = done.json()
    assert body["access_token"] == "tok-" + A and body["user_id"] == A
    codes = body["recovery_codes"]
    assert len(codes) == w.tfa.RECOVERY_CODE_COUNT and len(set(codes)) == len(codes), codes
    assert all(w.tfa.TwoFactorAuthManager.recovery_shaped(code) for code in codes)
    assert _totp(w, A) == 1, "active"
    assert w.tfa.two_factor_manager.get_status(A).recovery_codes_remaining == len(codes)
    assert len(_audited(w, "login_2fa_enrolled")) == 1

    spent = _complete(w, challenge_id, tf._code(w, begun["secret"]))
    assert spent.status_code == 403 and spent.json()["detail"] == (
        "Invalid or expired enrolment. Please log in again."), ("spent", spent.text)
    w.accounts.tokens.clear()
    replayed = _login(w).json()
    again_code = tf._post(w, SECOND, {"challenge_id": replayed["challenge_id"], "method": "totp",
                                       "code": tf._code(w, begun["secret"])})
    assert again_code.status_code == 403 and w.accounts.tokens == {}, ("the confirming code replayed", again_code.text)

    w.clock.advance(31)
    w.accounts.tokens.clear()
    asked = _login(w)
    assert asked.status_code == 200 and asked.json().get("requires_2fa") is True, asked.text
    assert {"totp", "recovery"} <= set(asked.json()["methods"])
    assert w.accounts.tokens == {}, "the next login asks the factor"

    w = world()
    w.routes._is_cookie_mode = lambda: True
    challenge_id = _enrolment(w)
    assert not _cookies(_login(w)), "the enrolment answer sets no cookie"
    secret = _begin(w, challenge_id).json()["secret"]
    done = _complete(w, challenge_id, tf._code(w, secret))
    assert done.status_code == 200, done.text
    assert {w.routes._ACCESS_COOKIE, w.routes._REFRESH_COOKIE} <= _cookies(done), done.headers


# --------------------------------------------------------------------------- EN3


def test_en3_the_enrolment_challenge_is_bounded(world):
    w = world()
    for path, body in ((BEGIN, {"challenge_id": "no-such-challenge"}),
                       (COMPLETE, {"challenge_id": "no-such-challenge", "code": "123456"})):
        unknown = tf._post(w, path, body)
        assert unknown.status_code == 403 and "log in again" in unknown.json()["detail"].lower(), (path, unknown.text)

    bob_secret = tf._enroll_totp(w, B, PW_B)
    w.current = {}
    second = _login(w, "bob", PW_B).json()
    assert second.get("requires_2fa") is True, second
    for path, body in ((BEGIN, {"challenge_id": second["challenge_id"]}),
                       (COMPLETE, {"challenge_id": second["challenge_id"], "code": "123456"})):
        refused = tf._post(w, path, body)
        assert refused.status_code == 403 and refused.json()["detail"] == (
            "Invalid or expired enrolment. Please log in again."), (path, refused.text)
    kept = tf._post(w, SECOND, {"challenge_id": second["challenge_id"], "method": "auto",
                                "code": tf._code(w, bob_secret)})
    assert kept.status_code == 200 and kept.json()["user_id"] == B, "the second step's challenge left as it was"

    challenge_id = _enrolment(w)
    through = tf._post(w, SECOND, {"challenge_id": challenge_id, "method": "auto", "code": "123456"})
    assert through.status_code == 403 and "log in again" in through.json()["detail"].lower(), through.text
    secret = _begin(w, challenge_id).json()["secret"]
    right = tf._code(w, secret)
    wrong = str((int(right) + 1) % 1_000_000).zfill(6)
    for remaining in (4, 3, 2, 1):
        miss = _complete(w, challenge_id, wrong)
        assert miss.status_code == 403 and miss.json()["detail"] == (
            f"Invalid 2FA code. {remaining} attempt(s) remaining."), miss.text
    locked = _complete(w, challenge_id, wrong)
    assert locked.json()["detail"] == "Too many 2FA attempts. Please log in again.", locked.text
    late = _complete(w, challenge_id, right)
    assert late.status_code == 403 and "log in again" in late.json()["detail"].lower(), late.text
    assert _totp(w, A) == 0 and "tok-" + A not in w.accounts.tokens, "nothing passed"

    expired = _enrolment(w)
    w.clock.advance(w.routes._ChallengeStore.CHALLENGE_TTL_SECONDS + 1)
    gone = _begin(w, expired)
    assert gone.status_code == 403 and "log in again" in gone.json()["detail"].lower(), gone.text

    one, other = _enrolment(w), _enrolment(w)
    secret = _begin(w, one).json()["secret"]
    assert _complete(w, one, tf._code(w, secret)).status_code == 200
    codes = tf._of(tf._rows(w), A)["recovery_codes"]
    since = _complete(w, other, tf._code(w, secret))
    assert since.status_code == 403 and since.json()["detail"] == (
        "This account has a second factor now. Please log in again."), since.text
    assert w.routes._challenge_store.get(other) is None, "its challenge spent"
    assert tf._of(tf._rows(w), A)["recovery_codes"] == codes, "the recovery codes kept"

    w = world()
    owner, thief = _enrolment(w), _enrolment(w)
    secret = _begin(w, owner).json()["secret"]
    manager = w.tfa.two_factor_manager
    setup = manager.totp_setup

    def confirmed_meanwhile(user_id, **kwargs):
        # The owner's confirmation lands between the begin's status read and its setup.
        assert _complete(w, owner, tf._code(w, secret)).status_code == 200
        return setup(user_id, **kwargs)

    manager.totp_setup = lambda user_id, **kwargs: (setattr(manager, "totp_setup", setup)
                                                    or confirmed_meanwhile(user_id, **kwargs))
    raced = _begin(w, thief)
    assert raced.status_code == 403 and raced.json()["detail"] == (
        "This account has a second factor now. Please log in again."), raced.text
    assert _totp(w, A) == 1 and w.routes._challenge_store.get(thief) is None, "the confirmed TOTP kept"
    tf._as(w, B)
    bob = tf._post(w, "/api/security/2fa/totp/setup", {"password": PW_B}).json()["secret"]
    manager.totp_setup = lambda user_id, **kwargs: (setattr(manager, "totp_setup", setup)
                                                    or manager.totp_verify(B, tf._code(w, bob))
                                                    and setup(user_id, **kwargs))
    again = tf._post(w, "/api/security/2fa/totp/setup", {"password": PW_B})
    assert again.status_code == 409 and _totp(w, B) == 1, ("the settings never replace it either", again.text)

    w = world()
    first = _enrolment(w)
    secret = _begin(w, first).json()["secret"]
    right = tf._code(w, secret)
    wrong = str((int(right) + 1) % 1_000_000).zfill(6)
    for challenge_id, misses in ((first, 3), (_enrolment(w), 2)):
        for _ in range(misses):
            assert _complete(w, challenge_id, wrong).status_code == 403
    third = _enrolment(w)
    held = _complete(w, third, right)
    assert held.status_code == 429 and held.json()["detail"] == (
        "Too many wrong codes for this account. Wait a few minutes, then log in again."), held.text
    assert _totp(w, A) == 0 and w.accounts.tokens == {}, "five wrong codes across logins hold the account"
    w.clock.advance(w.tfa.TOTP_WINDOW_SECONDS + 1)
    after = _enrolment(w)
    passed = _complete(w, after, tf._code(w, secret))
    assert passed.status_code == 200, ("the window ended", passed.text)

    w = world()
    _secrets_in_turn(w, "OWNERSECRETAAAAA", "OTHERSECRETBBBBB")
    owner, other = _enrolment(w), _enrolment(w)
    code = tf._code(w, _begin(w, owner).json()["secret"])
    decrypt, again = w.tfa._decrypt_secret, []

    def set_up_again_meanwhile(ciphertext):
        # Another login's begin lands between the owner's read of the secret and its activation.
        w.tfa._decrypt_secret = decrypt
        again.append(_begin(w, other).json()["secret"])
        return decrypt(ciphertext)

    w.tfa._decrypt_secret = set_up_again_meanwhile
    void = _complete(w, owner, code)
    assert again == ["OTHERSECRETBBBBB"], again
    assert void.status_code == 403 and void.json()["detail"] == (
        "Another login set this enrolment's secret up again. Please log in again."), void.text
    assert _totp(w, A) == 0 and w.accounts.tokens == {}, "the secret set up again is never activated"

    w = world()
    challenge_id = _enrolment(w)
    secret = _begin(w, challenge_id).json()["secret"]
    right = tf._code(w, secret)
    wrong = str((int(right) + 1) % 1_000_000).zfill(6)
    manager, decrypt, together = w.tfa.two_factor_manager, w.tfa._decrypt_secret, []

    def sent_together(ciphertext):
        # Six more wrong codes arrive while the first is read: each is counted before its code is.
        w.tfa._decrypt_secret = decrypt
        together.extend(manager.totp_verify(A, wrong).get("error") for _ in range(6))
        return decrypt(ciphertext)

    w.tfa._decrypt_secret = sent_together
    assert _complete(w, challenge_id, wrong).status_code == 403
    assert together == ["invalid_code"] * 4 + ["rate_limited"] * 2, together
    assert _complete(w, _enrolment(w), right).status_code == 429, "the seven are the account's"

    w = world()
    challenge_id = _enrolment(w)
    w.clock.advance(200)
    secret = _begin(w, challenge_id).json()["secret"]
    w.clock.advance(200)
    slow = _complete(w, challenge_id, tf._code(w, secret))
    assert slow.status_code == 200, ("five minutes from the latest begin", slow.text)


# --------------------------------------------------------------------------- EN4


def test_en4_a_mode_that_cannot_be_read_is_bulbe_and_a_factor_that_cannot_be_set_up_refuses(world):
    w = world(bulbe=False)
    mode_module = sys.modules["opti_oignon.security_mode"]

    def unreadable():
        raise OSError("the mode file cannot be read")

    mode_module.is_bulbe = unreadable
    assert _login(w).json().get("requires_2fa_enrolment") is True, "an is_bulbe that raises is Bulbe"
    del mode_module.is_bulbe
    assert _login(w).json().get("requires_2fa_enrolment") is True, "a mode module without one is Bulbe"
    assert w.accounts.tokens == {}

    w = world()
    manager = w.tfa.two_factor_manager
    held = manager.get_status(A)
    statements = w.db.calls
    assert manager.is_2fa_required(A, status=held) is True and manager.is_2fa_required(A) is True
    assert w.db.calls == statements, "in Bulbe the mode alone answers"
    w.mode.bulbe = False
    assert manager.is_2fa_required(A, status=held) is False and w.db.calls == statements, "the status given is read"

    w = world()
    w.tfa.TOTP_AVAILABLE = False
    refused = _login(w)
    assert refused.status_code == 503, refused.text
    assert refused.json()["detail"] == "A second factor cannot be set up here: login refused."
    assert w.accounts.tokens == {} and w.routes._challenge_store._store == {}, "no token, no challenge"

    w = world()
    challenge_id = _enrolment(w)
    w.tfa.TOTP_AVAILABLE = False
    late = _begin(w, challenge_id)
    assert late.status_code == 503 and late.json()["detail"] == "A second factor cannot be set up here: login refused."
    w.tfa.TOTP_AVAILABLE = True
    unsealed = type("EncryptionUnavailable", (Exception,), {"__module__": "opti_oignon.encryption"})

    def cannot_seal(user_id, **kwargs):
        raise unsealed("the master key is not loaded")

    w.tfa.two_factor_manager.totp_setup = cannot_seal
    sealed = _begin(w, challenge_id)
    assert sealed.status_code == 503 and sealed.json()["detail"] == (
        "TOTP not set up: the master key is not loaded"), sealed.text


# --------------------------------------------------------------------------- EN5


def test_en5_the_registration_in_bulbe_enrols_before_its_first_session(world):
    w = world()
    made = _with_create_user(w)
    answer = tf._post(w, REGISTER, {"username": "carol", "password": "carol-own-password"})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body.get("requires_2fa_enrolment") is True and "access_token" not in body, body
    assert made == ["uid-carol"] and w.accounts.tokens == {}, "the account made, no session"
    secret = _begin(w, body["challenge_id"]).json()["secret"]
    done = _complete(w, body["challenge_id"], tf._code(w, secret))
    assert done.status_code == 200 and done.json()["user_id"] == "uid-carol", done.text

    daily = world(bulbe=False)
    _with_create_user(daily)
    given = tf._post(daily, REGISTER, {"username": "dave", "password": "dave-own-password"})
    assert given.status_code == 200 and given.json().get("access_token") == "tok-uid-dave", given.text

    for broken in ("module", "totp"):
        w = world()
        made = _with_create_user(w)
        if broken == "module":
            w.routes._get_2fa_manager = lambda: None
        else:
            w.tfa.TOTP_AVAILABLE = False
        refused = tf._post(w, REGISTER, {"username": "erin", "password": "erin-own-password"})
        assert refused.status_code == 503 and "registration refused" in refused.json()["detail"], (broken,
                                                                                                    refused.text)
        assert made == [] and w.accounts.tokens == {}, (broken, "no account left behind")


# --------------------------------------------------------------------------- EN7


def test_en7_the_code_that_confirms_a_totp_never_signs_in_again(world):
    w = world(bulbe=False)
    manager = w.tfa.two_factor_manager
    seed = manager.totp_setup(A)["secret"]
    confirming = tf._code(w, seed)
    assert manager.totp_validate(A, confirming) is False, "a right code against an unverified enrolment is refused"
    assert manager.totp_validate("uid-ghost", confirming) is False, "an unknown user is refused"
    assert manager.totp_verify(A, confirming)["success"]
    assert manager.totp_validate(A, confirming) is False, "the code that confirmed the TOTP is consumed with it"
    w.clock.advance(30)
    following = tf._code(w, seed)
    assert manager.totp_validate(A, following) is True, "once verified, the next step validates"
    assert manager.totp_validate(A, following) is False, "once only inside its window"
    w.clock.advance(30)
    assert manager.totp_validate(A, tf._code(w, seed)) is True, "the step after it again: no lockout"
