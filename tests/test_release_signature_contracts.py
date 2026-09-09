"""Contracts for the release signature verifier.

The verifier is the last thing standing between a user and an archive that
someone else wrote. It is driven here as a program, not imported as a
module: the thing under test is the script's verdict, and a verdict is
whatever the process returns to a shell. Every case runs against a
throwaway keyring created inside the test's own temporary directory, so
the developer's real keys are never consulted and never at risk.

Two properties are worth naming, because they are the ones a string-matching
verifier gets wrong:

* The verdict must come from the machine-readable status channel, never from
  human-readable prose. A signer picks their own user id, so any check that
  greps the rendered output lets the signer write part of the sentence that
  judges them.
* An expected-key argument must match a key, not a substring of the output.
  A fingerprint that also happens to appear in a file name is not a match.

A refusing verifier is not automatically a correct one, so the happy path is
pinned alongside the refusals: an honestly signed archive must still pass.
"""

import os
import shutil
import subprocess

import pytest

_VERIFY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "verify_release.sh",
)

_ORDINARY_UID = "Release Tester <tester@example.invalid>"

# A user id that contains the exact prose a rendered-output check looks for.
# Nothing stops a signer choosing this, which is the point.
_DECEPTIVE_UID = "Good signature <impostor@example.invalid>"


pytestmark = pytest.mark.skipif(
    shutil.which("gpg") is None, reason="gpg is not installed"
)


def _gpg(home, *args):
    """Run gpg against a throwaway keyring."""
    env = dict(os.environ, GNUPGHOME=home)
    return subprocess.run(
        ["gpg", "--batch", "--yes", *args],
        env=env,
        capture_output=True,
        text=True,
    )


def _keyring(tmp_path, uid):
    """Create an isolated keyring holding exactly one signing key."""
    home = tmp_path / "gnupg"
    home.mkdir(mode=0o700, exist_ok=True)
    home_str = str(home)
    result = _gpg(
        home_str, "--passphrase", "", "--quick-generate-key",
        uid, "rsa2048", "sign", "0",
    )
    assert result.returncode == 0, result.stderr
    listed = _gpg(home_str, "--list-keys", "--with-colons")
    fingerprint = ""
    for line in listed.stdout.splitlines():
        if line.startswith("fpr:"):
            fingerprint = line.split(":")[9]
            break
    assert fingerprint, listed.stdout
    return home_str, fingerprint


def _archive(tmp_path, name="release.zip", body="payload"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _sign(home, archive):
    result = _gpg(
        home, "--armor", "--detach-sign",
        "--output", f"{archive}.sig", str(archive),
    )
    assert result.returncode == 0, result.stderr


def _checksum(archive):
    result = subprocess.run(
        ["sha256sum", archive.name],
        cwd=str(archive.parent), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    (archive.parent / f"{archive.name}.sha256").write_text(
        result.stdout, encoding="utf-8"
    )


def _verify(home, archive, *args):
    env = dict(os.environ, GNUPGHOME=home)
    return subprocess.run(
        ["bash", _VERIFY, archive.name, *args],
        cwd=str(archive.parent), env=env, capture_output=True, text=True,
    )


def test_rs1_an_archive_without_a_signature_is_refused(tmp_path):
    """No signature file at all is a refusal, not a warning."""
    home, _ = _keyring(tmp_path, _ORDINARY_UID)
    archive = _archive(tmp_path)
    result = _verify(home, archive)
    assert result.returncode != 0, result.stdout


def test_rs2_a_tampered_archive_is_refused(tmp_path):
    """The signature covers the bytes; changed bytes end the matter."""
    home, _ = _keyring(tmp_path, _ORDINARY_UID)
    archive = _archive(tmp_path)
    _sign(home, archive)
    archive.write_text("tampered", encoding="utf-8")
    result = _verify(home, archive)
    assert result.returncode != 0, result.stdout


def test_rs3_a_signer_cannot_write_the_verdict_in_their_own_user_id(tmp_path):
    """A deceptive user id must not turn a bad signature into a good one.

    The archive here is tampered with, exactly as in the previous contract.
    The only thing that changes is the text the signer chose for themselves.
    If that text can move the verdict, the verifier is reading prose.
    """
    home, _ = _keyring(tmp_path, _DECEPTIVE_UID)
    archive = _archive(tmp_path)
    _sign(home, archive)
    archive.write_text("tampered", encoding="utf-8")
    result = _verify(home, archive)
    assert result.returncode != 0, result.stdout


def test_rs4_an_expected_key_must_match_a_key_not_a_substring(tmp_path):
    """A key argument that only appears in the file name is not a match.

    The archive is named after the key that is being demanded, so any check
    that searches the rendered output for that string will find it and be
    satisfied. The signature is genuine but comes from a different key, so
    the correct answer is refusal.
    """
    home, _ = _keyring(tmp_path, _ORDINARY_UID)
    absent_key = "DEADBEEFDEADBEEF"
    archive = _archive(tmp_path, name=f"release-{absent_key}.zip")
    _sign(home, archive)
    result = _verify(home, archive, "--key", absent_key)
    assert result.returncode != 0, result.stdout


def test_rs5_strict_mode_refuses_a_missing_checksum(tmp_path):
    home, _ = _keyring(tmp_path, _ORDINARY_UID)
    archive = _archive(tmp_path)
    _sign(home, archive)
    result = _verify(home, archive, "--strict")
    assert result.returncode != 0, result.stdout


def test_rs6_a_checksum_that_disagrees_is_refused(tmp_path):
    home, _ = _keyring(tmp_path, _ORDINARY_UID)
    archive = _archive(tmp_path)
    _sign(home, archive)
    _checksum(archive)
    (tmp_path / "release.zip.sha256").write_text(
        "0" * 64 + "  release.zip\n", encoding="utf-8"
    )
    result = _verify(home, archive)
    assert result.returncode != 0, result.stdout


def test_rs7_an_honestly_signed_archive_still_passes(tmp_path):
    """Scope control: a verifier that refuses everything is not a verifier."""
    home, fingerprint = _keyring(tmp_path, _ORDINARY_UID)
    archive = _archive(tmp_path)
    _sign(home, archive)
    _checksum(archive)
    result = _verify(home, archive, "--strict", "--key", fingerprint)
    assert result.returncode == 0, result.stdout + result.stderr


def test_rs8_exit_codes_match_the_documented_map(tmp_path):
    """A missing checksum file exits with the missing-file code, cleanly.

    The script's own header documents one code for a checksum that
    disagrees and another for a file that is absent; a caller that
    distinguishes the two must be able to trust that map. The diagnostic
    must also end at its message: an exit code leaking into the printed
    line is an argument passed one slot too far.
    """
    home, fingerprint = _keyring(tmp_path, _ORDINARY_UID)
    archive = _archive(tmp_path)
    _sign(home, archive)
    # The caller names its key. Without that the script reads the project
    # pin ahead of everything else and refuses on identity long before it
    # reaches the checksum, so this clause would report the refusal code
    # for a throwaway signer instead of the code for the absent file it
    # was written to pin. The sibling clause above already had it right.
    result = _verify(home, archive, "--strict", "--key", fingerprint)
    assert result.returncode == 3, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    for line in combined.splitlines():
        if "strict mode" in line:
            assert line.rstrip().endswith(")"), (
                f"diagnostic must end at its message, got: {line!r}"
            )


def test_rs9_a_pinned_project_key_refuses_any_other_signer(tmp_path):
    """A fingerprint recorded next to the script is the expected key.

    Without a pin, a valid signature only proves the archive matches SOME
    key in the local keyring -- and the documentation tells users to import
    the key shipped with the archive, so that proof is integrity, never
    identity. With a pin present, a genuine signature from any other key
    must be refused, and the honest signer must still pass: both blades,
    or the pin is decoration.
    """
    home, fingerprint = _keyring(tmp_path, _ORDINARY_UID)
    archive = _archive(tmp_path)
    _sign(home, archive)

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    script_copy = scripts_dir / "verify_release.sh"
    script_copy.write_text(
        open(_VERIFY, encoding="utf-8").read(), encoding="utf-8"
    )
    pin = scripts_dir / "release_key.fpr"

    def _verify_pinned():
        env = dict(os.environ, GNUPGHOME=home)
        return subprocess.run(
            ["bash", str(script_copy), archive.name],
            cwd=str(archive.parent), env=env, capture_output=True, text=True,
        )

    pin.write_text("A1B2" * 10 + "\n", encoding="utf-8")
    refused = _verify_pinned()
    assert refused.returncode != 0, (
        "a genuine signature from a key other than the pinned one "
        "must be refused:\n" + refused.stdout + refused.stderr
    )

    pin.write_text(fingerprint + "\n", encoding="utf-8")
    accepted = _verify_pinned()
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr


def test_rs10_a_named_key_takes_precedence_over_the_project_pin(tmp_path):
    """An explicit key must reach the checks the pin would pre-empt.

    The pin is read before anything else and answers on identity, which
    is right when nobody said which key to expect and wrong when someone
    did. Unpinned, that ordering is invisible: every clause that omits a
    key passes for as long as no pin exists, and the day one is recorded
    they all report the refusal code in place of whatever they were
    written to pin. This holds the precedence itself, so the ordering is
    a contract rather than a property of whether a file happens to exist.
    """
    home, fingerprint = _keyring(tmp_path, _ORDINARY_UID)
    archive = _archive(tmp_path)
    _sign(home, archive)

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    script_copy = scripts_dir / "verify_release.sh"
    script_copy.write_text(
        open(_VERIFY, encoding="utf-8").read(), encoding="utf-8"
    )
    (scripts_dir / "release_key.fpr").write_text(
        "A1B2" * 10 + "\n", encoding="utf-8"
    )

    def _run(*args):
        env = dict(os.environ, GNUPGHOME=home)
        return subprocess.run(
            ["bash", str(script_copy), archive.name, *args],
            cwd=str(archive.parent), env=env, capture_output=True, text=True,
        )

    silent = _run("--strict")
    assert silent.returncode == 1, (
        "with a pin recorded and no key named, the refusal is on identity"
    )

    named = _run("--strict", "--key", fingerprint)
    assert named.returncode == 3, (
        "a named key must carry past the pin and reach the documented "
        "missing-file code:\n" + named.stdout + named.stderr
    )

    _checksum(archive)
    complete = _run("--strict", "--key", fingerprint)
    assert complete.returncode == 0, complete.stdout + complete.stderr
