"""Contracts of the Ollama engine's models folder and this account (MP-01, after 256): the background check of the
pinned layers runs under the account that may write the engine's folder, so whatever rewrites a blob can fight it -- a
tripwire. The way out is a folder this account cannot write; the startup checklist looks for the ways it could. A
DETECTOR, not a certificate (Leon's choice after review 2): it names the ways in it finds, or says it found none among
those it knows, naming what it does not judge -- never that the folder is out of reach.

  * OF1 -- THIS ACCOUNT'S FOLDER: a folder the account made is writable (an entry on the way to it can be replaced);
    inside the folder the kernel's own answer decides (a file anyone may write is writable, one no one may write gives
    none); root, or a capability that writes any file (read from the effective set), may write anything. The startup
    check warns (never critical, so it never blocks Bulbe), names the folder, says the layer check is a tripwire, and
    gives the way out.
  * OF2 -- NONE FOUND: an account that owns nothing there and may write none of it, a read-only mount whose files'
    bits give it nothing, a sticky folder on the way whose entry is not its own (nor the folder): none found. The
    startup check passes as info, names what it does not judge, never says guarantee or reach, and says which folder
    the layer check reads (and how to point it at this one).
  * OF3 -- EACH WAY IN: a blob it may write, a blob it owns (the right given back; on a read-only mount too, never
    counted as protection), a folder on the way, a folder inside, a blob its group may write through a group the
    account belongs to that the process does not carry, the folder holding a link reached through another link or
    through a target that climbs with '..', a link that leads to a file it may write: each one is named; at most three
    said, and the walk stops there.
  * OF4 -- CANNOT LOOK: a folder it may enter but not list, a folder (the folder or one inside) it cannot enter, a
    folder on the way it cannot look into, links that do not end, a link inside, more entries than the bound, a folder
    named relative or through '..', groups or an account that cannot be read, an access list where the bits do not
    decide (a read-only mount, groups the process does not carry), a reader that raises or does not end (no second
    walk while one runs): not checked, never a verdict; the startup check passes as info and says why. An access
    list's reader: an error is not "none". The account's groups are read from the account. No folder at all is not
    checked. A pipe in the folder is never waited on.
  * OF5 -- THE FOLDERS AND THE TEXTS: the folders looked at are OLLAMA_MODELS as named (or ~/.ollama/models, the
    folder the layer check reads) and the Linux system install's, once each; the runner calls the check, never
    critical; SECURITY.md and the CHANGELOG say it is a detector, what it does not judge, and the way out.
  * OF6 -- UB5 over the roster with the models folder check (supersedes it: the roster it listed lacks the tenth
    check): each check runs on its own, one that raises is reported not checked, the others still run.
  * OF7 -- UB7 over that roster (supersedes it): a check that returns no CheckItem is reported not checked.
OF1-OF5 run the real package on folders the suite makes, the account seen as another one through the module's seams
(``_euid``, ``_access``, ``_capabilities``, ``_carried_groups``, ``_missing_groups``, ``_read_only_mount``,
``_listxattr``) where a contract needs rights a test cannot take; OF6 and OF7 load the checklist on the shared
isolation window, as UB5 and UB7 did. No engine is reached, no file's bytes read.
"""
import ast
import errno
import os
import stat
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_bulbe_never_boots_unchecked_contracts as ub_t  # noqa: E402
import test_startup_boot_guard_contracts as bg_t  # noqa: E402

_ROSTER = ub_t._RUNNER_CHECKS + (("_check_ollama_models_writable", "ollama_models_writable"),)

_OF6_CASES = [
    pytest.param(function, name, "bulbe", id=f"{function[len('_check_'):]}-bulbe") for function, name in _ROSTER
] + [pytest.param("_check_ollama_bind", "ollama_bind", "daily", id="ollama_bind-daily")]

BUDGET_S = {
    "test_of1_this_accounts_folder": 3.0,
    "test_of2_none_found": 3.0,
    "test_of3_each_way_in": 3.0,
    "test_of4_cannot_look": 6.0,
    "test_of5_the_folders_and_the_texts": 3.0,
    "test_of6_ub5_over_the_roster_with_the_models_folder[code_signing_scripts-bulbe]": 2.0,
    "test_of6_ub5_over_the_roster_with_the_models_folder[ollama_bind-bulbe]": 2.0,
    "test_of6_ub5_over_the_roster_with_the_models_folder[luks-bulbe]": 2.0,
    "test_of6_ub5_over_the_roster_with_the_models_folder[security_mode-bulbe]": 2.0,
    "test_of6_ub5_over_the_roster_with_the_models_folder[encrypted_swap-bulbe]": 2.0,
    "test_of6_ub5_over_the_roster_with_the_models_folder[governor_ollama_limits-bulbe]": 2.0,
    "test_of6_ub5_over_the_roster_with_the_models_folder[pqc_primitive-bulbe]": 2.0,
    "test_of6_ub5_over_the_roster_with_the_models_folder[backend_provenance_coverage-bulbe]": 2.0,
    "test_of6_ub5_over_the_roster_with_the_models_folder[process_secrets-bulbe]": 2.0,
    "test_of6_ub5_over_the_roster_with_the_models_folder[ollama_models_writable-bulbe]": 2.0,
    "test_of6_ub5_over_the_roster_with_the_models_folder[ollama_bind-daily]": 2.0,
    "test_of7_ub7_over_the_roster_with_the_models_folder[bulbe]": 2.0,
    "test_of7_ub7_over_the_roster_with_the_models_folder[daily]": 2.0,
}

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def real():
    """The real model_provenance and startup_checks, imported when a contract runs, never at collection."""
    from opti_oignon import model_provenance, startup_checks
    return model_provenance, startup_checks


def _store(base: Path) -> Path:
    """A models folder as the engine lays it out: blobs, and a manifest under manifests."""
    folder = base / "models"
    (folder / "blobs").mkdir(parents=True)
    tag = folder / "manifests" / "registry.ollama.ai" / "library" / "tiny"
    tag.mkdir(parents=True)
    (tag / "latest").write_text("{}")
    for name in ("sha256-" + "a" * 64, "sha256-" + "b" * 64):
        (folder / "blobs" / name).write_bytes(b"layer")
    return folder


def _as_another(monkeypatch, mp, writable=(), closed=(), uid=None, missing=()):
    """This process seen as an account that owns nothing the suite made (or ``uid``), with no capability: it may write
    only the paths in ``writable``, enter every folder but those in ``closed``, and belongs to the groups ``missing``
    beside the ones it carries; every other right as the kernel answers it."""
    monkeypatch.setattr(mp, "_euid", lambda: os.geteuid() + 1 if uid is None else uid)
    monkeypatch.setattr(mp, "_capabilities", lambda: set())
    monkeypatch.setattr(mp, "_missing_groups", lambda euid: (set(missing), None))
    monkeypatch.setattr(mp, "_carried_groups", lambda: set())
    allowed = {str(Path(p)) for p in writable}
    shut = {str(Path(p)) for p in closed}

    def access(path, mode):
        path = str(Path(path))
        if mode & os.W_OK:
            return path in allowed
        if mode & os.X_OK and path in shut:
            return False
        return os.access(path, mode)

    monkeypatch.setattr(mp, "_access", access)


def _checked(monkeypatch, tmp_path, folder, mp, sc):
    """The real startup check, the engine's folder named by OLLAMA_MODELS, the system install's absent."""
    monkeypatch.setenv("OLLAMA_MODELS", str(folder))
    monkeypatch.setattr(mp, "OLLAMA_SYSTEM_MODELS_DIR", tmp_path / "no-system-install")
    monkeypatch.setattr(sc, "_reach_walk", [])
    item = sc._check_ollama_models_writable()
    assert item.name == "ollama_models_writable" and item.severity != "critical", item
    return item


def test_of1_this_accounts_folder(real, monkeypatch, tmp_path):
    mp, sc = real
    folder = _store(tmp_path)
    state, why = mp.ollama_folder_reach(folder)
    assert state == "writable" and why and len(why) <= 3, (state, why)
    assert any("can be moved away and another put in its place" in way for way in why), why

    item = _checked(monkeypatch, tmp_path, folder, mp, sc)
    assert item.passed is False and item.severity == "warning" and item.score_impact == -3, item
    assert str(folder) in item.detail and "tripwire" in item.detail, item.detail
    tips = " ".join(item.tips)
    for advice in ("its own account", "OLLAMA_MODELS", "read-only mount is not counted", "SECURITY.md"):
        assert advice in tips, (advice, item.tips)

    # Inside the folder the kernel's own answer decides: a file no one may write is passed, one anyone may write is not.
    original = mp._access
    _as_another(monkeypatch, mp)
    monkeypatch.setattr(mp, "_access", lambda path, mode: original(path, mode) if (
        str(path).startswith(str(folder)) or not mode & os.W_OK) else False)
    blob = folder / "blobs" / ("sha256-" + "a" * 64)
    shut = [p for p in (folder, *folder.rglob("*")) if p.is_dir()]
    try:
        for path in folder.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        for path in shut:
            path.chmod(0o555)
        assert mp.ollama_folder_reach(folder) == ("none_found", [])
        blob.chmod(0o666)
        assert mp.ollama_folder_reach(folder) == ("writable", [f"{blob} may be written by this account"])
    finally:
        for path in shut:
            path.chmod(0o755)

    # Root, or a capability that writes any file.
    _as_another(monkeypatch, mp, uid=0)
    assert mp.ollama_folder_reach(folder) == ("writable", ["this process runs as root: it may write any folder"])
    _as_another(monkeypatch, mp)
    monkeypatch.setattr(mp, "_capabilities", lambda: {"CAP_FOWNER", "CAP_CHOWN"})
    assert mp.ollama_folder_reach(folder) == (
        "writable", ["this process holds CAP_CHOWN, CAP_FOWNER: it may write any folder"])
    monkeypatch.undo()
    for bits, held in (("0000000000000008", {"CAP_FOWNER"}), ("0000000000200000", {"CAP_SYS_ADMIN"}),
                       ("0000000000000003", {"CAP_CHOWN", "CAP_DAC_OVERRIDE"}), ("0000000000000000", set()),
                       ("0000000000000004", set())):
        assert mp._capabilities(f"Name:\tpython3\nCapInh:\t0000000000000000\nCapEff:\t{bits}\n") == held, bits
    assert mp._capabilities("Name:\tpython3\n") is None and mp._capabilities("CapEff:\tzz\n") is None


def test_of2_none_found(real, monkeypatch, tmp_path):
    mp, sc = real
    folder = _store(tmp_path)
    _as_another(monkeypatch, mp)
    assert mp.ollama_folder_reach(folder) == ("none_found", [])
    item = _checked(monkeypatch, tmp_path, folder, mp, sc)
    assert item.passed is True and item.severity == "info" and item.score_impact == 0 and not item.tips, item
    assert item.detail.startswith("No way for this account to write it found: " + str(folder)), item.detail
    for said in ("Not judged: mounts this account controls", "other accounts", "can become root",
                 "The background check of the layers reads it."):
        assert said in item.detail, (said, item.detail)
    assert "guarantee" not in item.detail and "reach" not in item.detail, item.detail

    # None found, but not the folder the layer check reads: which one it reads said, and how to point it here.
    named = tmp_path / "named-but-absent"
    monkeypatch.setenv("OLLAMA_MODELS", str(named))
    monkeypatch.setattr(mp, "OLLAMA_SYSTEM_MODELS_DIR", folder)
    item = sc._check_ollama_models_writable()
    assert item.passed is True and f"The background check of the layers reads {named}, not this folder" in (
        item.detail), item.detail
    assert any("OLLAMA_MODELS" in tip for tip in item.tips), item.tips

    # A read-only mount whose files' bits give this account nothing: none found.
    monkeypatch.setattr(mp, "_read_only_mount", lambda path: True)
    monkeypatch.setattr(mp, "_listxattr", lambda path: [])
    assert mp.ollama_folder_reach(folder) == ("none_found", [])
    monkeypatch.undo()

    # A sticky folder keeps from this account only an entry neither of the two is its own.
    def st(mode, uid):
        return os.stat_result((mode, 0, 0, 1, uid, 0, 0, 0, 0, 0))

    sticky, plain = stat.S_IFDIR | 0o1777, stat.S_IFDIR | 0o777
    assert mp._sticky_keeps(st(sticky, 7), st(stat.S_IFREG, 8), 9) is True
    assert mp._sticky_keeps(st(sticky, 7), st(stat.S_IFLNK, 9), 9) is False, "an entry of its own"
    assert mp._sticky_keeps(st(sticky, 9), st(stat.S_IFREG, 8), 9) is False, "a folder of its own"
    assert mp._sticky_keeps(st(plain, 7), st(stat.S_IFREG, 8), 9) is False, "a folder that is not sticky"
    held = tmp_path / "sticky"
    held.mkdir()
    held.chmod(0o1777)
    inside = _store(held)
    _as_another(monkeypatch, mp, writable=[held])
    assert mp.ollama_folder_reach(inside) == ("none_found", [])
    held.chmod(0o0777)
    _as_another(monkeypatch, mp, writable=[held])
    state, why = mp.ollama_folder_reach(inside)
    assert state == "writable" and why == [f"{held} may be written by this account: {inside} in it can be moved away "
                                           "and another put in its place"], why


def test_of3_each_way_in(real, monkeypatch, tmp_path):
    mp, sc = real
    folder = _store(tmp_path)
    blob = folder / "blobs" / ("sha256-" + "a" * 64)
    inner = folder / "manifests" / "registry.ollama.ai"
    for path, said in ((blob, f"{blob} may be written by this account"),
                       (tmp_path, f"{tmp_path} may be written by this account: {folder} in it can be moved away"),
                       (inner, f"{inner} may be written by this account")):
        _as_another(monkeypatch, mp, writable=[path])
        state, why = mp.ollama_folder_reach(folder)
        assert state == "writable" and why and why[0].startswith(said), (path, state, why)

    # A blob of its own, though no one may write it: the owner gives itself the right.
    owned = folder / "blobs" / ("sha256-" + "c" * 64)
    owned.write_bytes(b"layer")
    owned.chmod(0o444)
    _as_another(monkeypatch, mp)
    monkeypatch.setattr(mp, "_euid", lambda: owned.stat().st_uid)
    monkeypatch.setattr(mp, "_access", lambda path, mode: False if mode & os.W_OK else os.access(path, mode))
    monkeypatch.setattr(mp, "_REACH_SAID", 1000)      # every way said: the suite owns the folders above it too
    state, why = mp.ollama_folder_reach(folder)
    assert state == "writable" and f"{owned} is this account's, which can give itself the right to write it" in why, (
        why)
    # On a read-only mount, never counted as protection: the same owner found by the bits.
    monkeypatch.setattr(mp, "_read_only_mount", lambda path: True)
    state, why = mp.ollama_folder_reach(folder)
    assert state == "writable" and (f"{owned} lies on a read-only mount, never counted as protection: its rights let "
                                    "this account write it") in why, why
    monkeypatch.undo()
    owned.unlink()

    # A blob its group may write, a group this account belongs to that the process does not carry.
    group = next((gid for gid in os.getgroups() if gid != os.getegid()), None)
    assert group is not None, "this contract needs a group of the account's beside its primary one"
    os.chown(blob, -1, group)
    blob.chmod(0o464)
    _as_another(monkeypatch, mp, missing=[group])
    state, why = mp.ollama_folder_reach(folder)
    assert state == "writable" and why == [f"{blob} may be written by its group {group}, which this account belongs to "
                                           "(this process does not carry it)"], why
    blob.chmod(0o644)

    # A link reached through another link, and a link whose target climbs with '..': the folders on the way are seen.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "l1").symlink_to(tmp_path / "b" / "l2")
    (tmp_path / "b" / "l2").symlink_to(tmp_path)
    _as_another(monkeypatch, mp, writable=[tmp_path / "b"])
    state, why = mp.ollama_folder_reach(tmp_path / "a" / "l1" / "models")
    assert state == "writable" and why == [f"{tmp_path / 'b'} may be written by this account: {tmp_path / 'b' / 'l2'} "
                                           "in it can be moved away and another put in its place"], why
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "store").mkdir()
    _store(tmp_path / "c" / "store")
    (tmp_path / "a" / "up").symlink_to(Path("..") / "c" / "store")
    _as_another(monkeypatch, mp, writable=[tmp_path / "c"])
    state, why = mp.ollama_folder_reach(tmp_path / "a" / "up" / "models")
    assert state == "writable" and why == [f"{tmp_path / 'c'} may be written by this account: "
                                           f"{tmp_path / 'c' / 'store'} in it can be moved away and another put in "
                                           "its place"], why

    # A link that leads to a file it may write.
    target = tmp_path / "target"
    target.write_bytes(b"layer")
    link = folder / "blobs" / ("sha256-" + "d" * 64)
    link.symlink_to(target)
    _as_another(monkeypatch, mp, writable=[link])
    assert mp.ollama_folder_reach(folder) == ("writable", [f"{link} may be written by this account"])
    link.unlink()

    # Three ways said at most, and the walk stops there.
    many = [folder / "blobs" / f"sha256-{n:064x}" for n in range(6)]
    for path in many:
        path.write_bytes(b"layer")
    _as_another(monkeypatch, mp, writable=many)
    state, why = mp.ollama_folder_reach(folder)
    assert state == "writable" and len(why) == 3, why


def test_of4_cannot_look(real, monkeypatch, tmp_path):
    mp, sc = real
    folder = _store(tmp_path)
    # A folder it may enter but not list: not checked, the folders where none was found said beside.
    shut = folder / "manifests"
    shut.chmod(0o300)
    try:
        _as_another(monkeypatch, mp)
        state, why = mp.ollama_folder_reach(folder)
        assert state == "unchecked" and why == [f"{shut} cannot be listed (PermissionError)"], (state, why)
        beside = _store(tmp_path / "beside")
        monkeypatch.setenv("OLLAMA_MODELS", str(folder))
        monkeypatch.setattr(mp, "OLLAMA_SYSTEM_MODELS_DIR", beside)
        monkeypatch.setattr(sc, "_reach_walk", [])
        item = sc._check_ollama_models_writable()
        assert item.passed is True and item.severity == "info" and item.score_impact == 0, item
        assert item.detail == (f"Not checked: {folder} ({shut} cannot be listed (PermissionError)). "
                               f"No way for this account to write it found: {beside}"), item.detail
    finally:
        shut.chmod(0o700)

    # A folder it cannot enter, the folder itself or one it holds: what it holds was not looked at.
    for closed in (folder, folder / "blobs"):
        _as_another(monkeypatch, mp, closed=[closed])
        assert mp.ollama_folder_reach(folder) == (
            "unchecked", [f"{closed} cannot be entered by this account: what it holds was not looked at"]), closed

    # A folder on the way it cannot look into; links that do not end.
    parent = tmp_path / "parent"
    deep = _store(parent)
    parent.chmod(0o000)
    try:
        _as_another(monkeypatch, mp)
        assert mp.ollama_folder_reach(deep) == ("unchecked", [f"{deep} cannot be looked at (PermissionError)"])
    finally:
        parent.chmod(0o755)
    (tmp_path / "loop1").symlink_to(tmp_path / "loop2")
    (tmp_path / "loop2").symlink_to(tmp_path / "loop1")
    assert mp.ollama_folder_reach(tmp_path / "loop1" / "models") == (
        "unchecked", [f"{tmp_path / 'loop1' / 'models'} cannot be looked at (OSError)"])

    # A link inside: where it leads is not walked.
    link = folder / "blobs" / "away"
    link.symlink_to(tmp_path)
    _as_another(monkeypatch, mp)
    assert mp.ollama_folder_reach(folder) == ("unchecked", [f"{link} is a link (where it leads is not walked)"])
    link.unlink()

    # More entries than the bound; a folder named relative or through '..'; no groups, no account to read.
    state, why = mp.ollama_folder_reach(folder, bound=3)
    assert state == "unchecked" and why == ["it holds more than 3 entries: the first 3 looked at"], why
    for name in (Path("models"), tmp_path / "beside" / ".." / "models"):
        assert mp.ollama_folder_reach(name) == ("unchecked", [f"{name} is named relative or through '..': the engine "
                                                              "and this check may resolve it differently"])
    monkeypatch.setattr(mp, "_missing_groups", lambda euid: (set(), "this account's groups cannot be read (KeyError)"))
    assert mp.ollama_folder_reach(folder) == ("unchecked", ["this account's groups cannot be read (KeyError)"])
    monkeypatch.setattr(mp, "_euid", lambda: None)
    assert mp.ollama_folder_reach(folder) == ("unchecked", ["this platform's file rights are not read here"])

    # An access list where the bits do not decide: on a read-only mount, or beside groups the process does not carry.
    _as_another(monkeypatch, mp)
    monkeypatch.setattr(mp, "_read_only_mount", lambda path: True)
    monkeypatch.setattr(mp, "_listxattr", lambda path: ["system.nfs4_acl"] if str(path) == str(folder) else [])
    assert mp.ollama_folder_reach(folder) == (
        "unchecked", [f"{folder} lies on a read-only mount and carries an access list, which is not read"])
    monkeypatch.undo()
    _as_another(monkeypatch, mp, missing=[4242])
    monkeypatch.setattr(mp, "_listxattr", lambda path: ["system.posix_acl_access"] if str(path) == str(
        folder / "blobs") else ["user.mime_type"])
    assert mp.ollama_folder_reach(folder) == (
        "unchecked", [f"{folder / 'blobs'} carries an access list, and this account belongs to groups this process does "
                      "not carry"])

    # What an access list's reader answers: an error is not "none".
    def answers(result):
        def listxattr(path):
            if isinstance(result, OSError):
                raise result
            return result
        return listxattr

    for result, said in ((["system.posix_acl_access"], True), (["system.nfs4_acl"], True), (["user.x"], False),
                         (OSError(errno.ENOTSUP, "no"), False), (OSError(errno.EACCES, "no"), None),
                         (OSError(errno.EIO, "no"), None)):
        monkeypatch.setattr(mp, "_listxattr", answers(result))
        if said is None:
            with pytest.raises(mp._Unseen):
                mp._has_acl(folder)
        else:
            assert mp._has_acl(folder) is said, result
    monkeypatch.setattr(mp, "_listxattr", None)
    with pytest.raises(mp._Unseen):
        mp._has_acl(folder)
    monkeypatch.undo()

    # The account's groups the process does not carry, read from the account; none when they cannot be read.
    import pwd
    entry = pwd.getpwuid(os.geteuid())
    expected = set(os.getgrouplist(entry.pw_name, entry.pw_gid)) - {os.getegid()}
    assert expected, "this contract needs a group of the account's beside its primary one"
    monkeypatch.setattr(mp, "_carried_groups", lambda: {os.getegid()})
    assert mp._missing_groups(os.geteuid()) == (expected, None)
    monkeypatch.undo()
    assert mp._carried_groups() == set(os.getgroups()) | {os.getegid()}
    monkeypatch.undo()

    def unknown(uid):
        raise KeyError(uid)

    monkeypatch.setattr(pwd, "getpwuid", unknown)
    assert mp._missing_groups(os.geteuid()) == (set(), "this account's groups cannot be read (KeyError)")
    monkeypatch.undo()

    # A reader that raises, and one that does not end: not checked, named, the boot goes on; while it has not ended
    # no other walk is started.
    def raising():
        raise RuntimeError("the rights cannot be read")

    monkeypatch.setattr(mp, "ollama_models_reach", raising)
    item = _checked(monkeypatch, tmp_path, folder, mp, sc)
    assert item.passed is True and item.detail == ("Not checked: the engine's models folder could not be looked at "
                                                   "(RuntimeError: the rights cannot be read)"), item.detail
    release = threading.Event()
    monkeypatch.setattr(mp, "ollama_models_reach", lambda: release.wait(30) and [])
    monkeypatch.setattr(sc, "_REACH_DEADLINE_S", 0.2)
    try:
        item = _checked(monkeypatch, tmp_path, folder, mp, sc)
        assert item.passed is True and item.detail == ("Not checked: the engine's models folder could not be looked "
                                                       "at (TimeoutError: the walk did not end within 0.2 s)"), item
        walkers = [t for t in threading.enumerate() if t.name == "ollama-models-reach" and t.is_alive()]
        again = sc._check_ollama_models_writable()
        assert again.detail == ("Not checked: the engine's models folder could not be looked at (TimeoutError: an "
                                "earlier walk has not ended)"), again.detail
        assert [t for t in threading.enumerate() if t.name == "ollama-models-reach" and t.is_alive()] == walkers
    finally:
        release.set()
    monkeypatch.undo()

    # No folder at all: not checked, saying where an engine started otherwise keeps them.
    gone = tmp_path / "gone"
    assert mp.ollama_folder_reach(gone) == ("absent", [])
    item = _checked(monkeypatch, tmp_path, gone, mp, sc)
    assert item.passed is True and item.detail.startswith(
        f"Not checked: no folder of the engine's models at {gone}, {tmp_path / 'no-system-install'}"), item.detail
    assert "another OLLAMA_MODELS" in item.detail, item.detail

    # A pipe in the folder is never opened, so never waited on.
    os.mkfifo(folder / "blobs" / "pipe")
    _as_another(monkeypatch, mp)
    seen = []
    walker = threading.Thread(target=lambda: seen.append(mp.ollama_folder_reach(folder)), daemon=True)
    walker.start()
    walker.join(5)
    assert not walker.is_alive() and seen == [("none_found", [])], seen


def test_of5_the_folders_and_the_texts(real, monkeypatch, tmp_path):
    mp, sc = real
    system = tmp_path / "system"
    monkeypatch.setattr(mp, "OLLAMA_SYSTEM_MODELS_DIR", system)
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "named"))
    assert mp.ollama_models_folders() == [tmp_path / "named", system]
    assert mp.ollama_models_folders()[0] == Path(mp.ollama_models_dir())
    monkeypatch.setenv("OLLAMA_MODELS", str(system))
    assert mp.ollama_models_folders() == [system]
    monkeypatch.delenv("OLLAMA_MODELS")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert mp.ollama_models_folders() == [tmp_path / "home" / ".ollama" / "models", system]

    assert "_check_ollama_models_writable" in ub_t._called_by_runner()
    tree = ast.parse((_ROOT / "opti_oignon" / "startup_checks.py").read_text(encoding="utf-8"))
    check = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "_check_ollama_models_writable")
    assert not any(isinstance(node, ast.Constant) and node.value == "critical" for node in ast.walk(check)), (
        "the models folder check is advisory: never critical, so it never blocks the boot"
    )

    security = " ".join((_ROOT / "SECURITY.md").read_text(encoding="utf-8").split())
    for said in ("`ollama_models_writable`", "a detector, not a certificate", "never that the folder is out of reach",
                 "under its own account", "A read-only mount is never counted as protection", "OLLAMA_MODELS",
                 "never blocks", "Not judged, and said: mounts this account controls"):
        assert said in security, said
    changelog = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    added = changelog.split("### Added\n", 1)[1].split("\n### ", 1)[0]
    entry = " ".join(next((entry for entry in added.split("\n- ") if "OF1-OF7" in entry), "").split())
    for said in ("ollama_models_writable", "a detector, not a certificate", "never blocks", "costs no score",
                 "never counted as protection"):
        assert said in entry, (said, entry[:300])


@pytest.mark.parametrize(("function", "name", "mode"), _OF6_CASES)
def test_of6_ub5_over_the_roster_with_the_models_folder(function, name, mode):
    assert {f for f, _ in _ROSTER} == ub_t._called_by_runner(), (
        "the checks this contract lists are not the ones the runner calls"
    )
    mod, restore = ub_t._checklist(mode)
    try:
        for other, _ in _ROSTER:
            setattr(mod, other, lambda _n=other: bg_t._passing_item(mod, _n))
        if function == "_check_ollama_bind":
            error, error_name = ub_t._Unprintable("unused"), "_Unprintable"
        else:
            # A swap file whose name is not UTF-8, as /proc/swaps prints it.
            error = UnicodeDecodeError("utf-8", b"/swap\xe9", 5, 6, "invalid continuation byte")
            error_name = "UnicodeDecodeError"
        setattr(mod, function, ub_t._raising(error))
        mod.clear_cache()
        blocks = name == "ollama_bind" and mode == "bulbe"
        if blocks:
            with pytest.raises(mod.StartupBlockedError) as raised:
                mod.enforce_boot_checks()
            assert "ollama_bind" in str(raised.value), str(raised.value)
            result = mod.get_cached_result()
        else:
            result = mod.enforce_boot_checks()
            assert result is not None and result is mod.get_cached_result(), (
                f"a {name} check that raises stopped the checklist: the boot has no result to serve"
            )
        assert len(result.checks) == len(_ROSTER), (
            f"the checks after {name} did not run: {[c.name for c in result.checks]}"
        )
        item = {c.name: c for c in result.checks}[name]
        assert item.passed is False and "not checked" in item.detail and error_name in item.detail, (
            f"the {name} check that raised must be reported not checked, named by its error: {item.detail!r}"
        )
        assert item.severity == ("critical" if blocks else "warning"), (
            f"mode {mode!r}: the {name} check that raised is rated {item.severity!r}"
        )
        assert result.blocked is blocks, (
            f"mode {mode!r}: a {name} check that raised gives blocked={result.blocked} ({result.block_reason!r})"
        )
        assert all(c.passed for c in result.checks if c.name != name)
    finally:
        restore()


@pytest.mark.parametrize("mode", ("bulbe", "daily"))
def test_of7_ub7_over_the_roster_with_the_models_folder(mode):
    mod, restore = ub_t._checklist(mode)
    try:
        for other, _ in _ROSTER:
            setattr(mod, other, lambda _n=other: bg_t._passing_item(mod, _n))
        mod._check_ollama_bind = lambda: {"passed": True}
        mod._check_luks = lambda: None
        mod.clear_cache()
        if mode == "bulbe":
            with pytest.raises(mod.StartupBlockedError) as raised:
                mod.enforce_boot_checks()
            assert "ollama_bind" in str(raised.value), str(raised.value)
            result = mod.get_cached_result()
        else:
            result = mod.enforce_boot_checks()
            assert result is not None and result.blocked is False
        assert len(result.checks) == len(_ROSTER), [c.name for c in result.checks]
        items = {c.name: c for c in result.checks}
        for name, returned in (("ollama_bind", "dict"), ("luks_encryption", "NoneType")):
            item = items[name]
            assert item.passed is False and "not checked" in item.detail and returned in item.detail, (
                f"a {name} check that returned {returned} must be reported not checked, saying what it returned: "
                f"{item.detail!r}"
            )
        assert items["ollama_bind"].severity == ("critical" if mode == "bulbe" else "warning")
        assert items["luks_encryption"].severity == "warning"
    finally:
        restore()
