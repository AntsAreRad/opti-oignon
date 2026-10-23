"""The password a security ceremony asks for, checked so that nothing but the right password of the platform's owner,
asking through their own session, passes -- right as the installed bcrypt reads it: bcrypt 5 refuses a password of
more than 72 bytes, bcrypt 4 compares its first 72.

A ceremony -- approving a batch of plugins, leaving Bulbe mode, re-enabling search -- asks for the password of the
owner whose session asks for it: an admin's account, or -- while no account is an admin's, as on an install whose
accounts all registered with the default role -- the first registered (``AuthManager.owners``). ``check_password``
answers a ``Verdict``: passed, with the account's id, or refused, by name, with the status the route answers and a
sentence that says why:

* ``no_account`` (403): the session names no account, or one that is not there; the synthetic session of single-user
  mode stands for the owner's account when exactly one account is an owner's, and for none otherwise;
* ``not_owner`` (403): the account is not the owner's -- an account registered after it approves nothing with its
  own password;
* ``password_busy`` (429): a password for the same account is being checked already -- one check at a time per
  account, so that tries sent together cannot all pass the limiter before the first failure is counted;
* ``password_locked`` (429): the login limiter holds the account or the address back -- asked before the password is,
  so that a locked account is no oracle;
* ``wrong_password`` (403, never 401: the session is valid, a factor is wrong): recorded by the login limiter as a
  failed try for that account and address;
* ``password_unverifiable`` (503): anything that raised while the account or the password was being checked -- the
  account store that does not load or cannot be read, a stored hash bcrypt refuses to read, a password it refuses
  (bcrypt 5 refuses one of more than 72 bytes). A hash in a scheme the platform does not know reads as a wrong
  password, as the login reads it.

Nothing it cannot check passes. It writes nothing but the limiter's count of a failed try, and says no password.

``check_own_password`` is the same check without the owner: the password of the account the session stands for,
whoever it is -- the step-up an account gives before a second factor of its own is added or removed. It refuses as
``check_password`` does, ``not_owner`` aside. ``session_account`` says which account a session stands for, read from
the account store, and whether it is the owner's.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

_LOCAL_SESSION = "local"

# The accounts whose password is being checked: one check at a time per account, refused rather than queued, so
# that tries sent together meet the limiter one after the other and no worker thread waits on another.
_checking: set[str] = set()
_checking_lock = threading.Lock()


def _claim(account_id: str) -> bool:
    with _checking_lock:
        if account_id in _checking:
            return False
        _checking.add(account_id)
        return True


def _release(account_id: str) -> None:
    with _checking_lock:
        _checking.discard(account_id)


@dataclass(frozen=True)
class SessionAccount:
    """The account a session stands for, as the account store holds it now."""

    account_id: str
    username: str
    owner: bool


@dataclass(frozen=True)
class Verdict:
    """What the check found: the account whose password it is, or why it refused (``error``, ``status``,
    ``message``)."""

    account_id: str = ""
    status: int = 200
    error: str = ""
    message: str = ""

    @property
    def passed(self) -> bool:
        return not self.error


def _refused(status: int, error: str, message: str) -> Verdict:
    return Verdict(status=status, error=error, message=message)


def _account(session: Any, auth: Any) -> Any:
    """The account the session stands for, or None: the one its ``sub`` names, or, for the synthetic session of
    single-user mode, the owner's when exactly one account is an owner's."""
    if not isinstance(session, dict):
        return None
    user_id = session.get("sub") or session.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        return None
    if user_id == _LOCAL_SESSION:
        owners = auth.owners()
        return auth.get_user(owners[0]) if len(owners) == 1 else None
    return auth.get_user(user_id)


def session_account(session: Any) -> SessionAccount | None:
    """The account ``session`` stands for -- the one its ``sub`` names, or for the synthetic session of single-user
    mode the owner's when exactly one account is an owner's -- read from the account store now, with whether it is
    the owner's; None when it stands for none. Raises when the store did not load or cannot be read."""
    from opti_oignon.auth import auth_manager

    if auth_manager is None:
        raise RuntimeError("the account store did not load")
    account = _account(session, auth_manager)
    if account is None:
        return None
    return SessionAccount(account.user_id, account.username, bool(auth_manager.is_owner(account.user_id)))


def check_password(session: Any, password: str, *, client_ip: str = "unknown") -> Verdict:
    """Whether ``password`` is the password of the owner's account ``session`` stands for; see the module's
    docstring."""
    return _check(session, password, client_ip, owner_only=True)


def check_own_password(session: Any, password: str, *, client_ip: str = "unknown") -> Verdict:
    """Whether ``password`` is the password of the account ``session`` stands for, owner or not: the step-up before
    a second factor of its own is added or removed. Refuses as ``check_password`` does, ``not_owner`` aside."""
    return _check(session, password, client_ip, owner_only=False)


def _check(session: Any, password: str, client_ip: str, *, owner_only: bool) -> Verdict:
    try:
        from opti_oignon.auth import auth_manager, login_rate_limiter, verify_password

        if auth_manager is None:
            raise RuntimeError("the account store did not load")
        account = _account(session, auth_manager)
        if account is None:
            return _refused(403, "no_account", "No account stands behind this session: its password is needed.")
        if owner_only and not auth_manager.is_owner(account.user_id):
            return _refused(403, "not_owner", "Only the owner's password approves this ceremony: nothing was done.")
        if not _claim(account.user_id):
            return _refused(429, "password_busy",
                            "A password for this account is being checked already: try again in a moment.")
        try:
            allowed, retry_after = login_rate_limiter.check_rate_limit(client_ip, account.username)
            if not allowed:
                return _refused(429, "password_locked",
                                f"Too many failed tries for this account: try again in {retry_after} s.")
            if verify_password(password, account.password_hash):
                return Verdict(account_id=account.user_id)
            login_rate_limiter.record_failure(client_ip, account.username)
            return _refused(403, "wrong_password", "That password is not this account's: nothing was done.")
        finally:
            _release(account.user_id)
    except Exception as exc:  # noqa: BLE001 - a password that cannot be checked is refused, never taken
        return _refused(503, "password_unverifiable",
                        f"The password cannot be checked here ({type(exc).__name__}): nothing was done.")
