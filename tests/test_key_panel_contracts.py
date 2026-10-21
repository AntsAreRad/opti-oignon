#!/usr/bin/env python3
"""Contracts for the key setup of the Security screen: it says what the server did.

Once the stores are encrypted, the key is in place, and the server's setup
enables it as it is: it makes no key and uses no passphrase it is sent. The
panel used to answer that setup with words of its own -- "configured with a
random key", "configured with your passphrase. Remember it" -- and showed a
refusal as "API error 409", the server's reason dropped. These contracts pin
what it says now:

  * KP1 -- WHERE THE SETUP STARTS: a key here -- one that loads, a key
    file, OPTI_ENCRYPTION_KEY -- is enabled with one action and no
    passphrase, the server's refusal saying why when it cannot open it;
    the status the real server gives once the offline command has run --
    encryption off, so no key loaded, and the key file there -- starts
    there; with no key, the choice says the key file must be kept either
    way, and never that a passphrase recovers the data without it (the
    key file holds its salt).
  * KP2 -- WHAT A SUCCESS AND A REFUSAL SAY: a success says the key the
    server's answer names, for each value the server's setup gives -- the
    key already here, no key made, a passphrase sent not used; a key made,
    where, whether it is wrapped and under which passphrase, and that a
    copy of it is kept --, the databases still kept in plaintext and the
    command that encrypts them, and only the server's own words when its
    answer names no key; a refusal says the server's reason, else what the
    error says, never a bare status.
  * KP3 -- THE PANEL SAYS THEM: the panel starts from ``keySetupStart``,
    says ``KEY_SETUP_TEXT``, says each success through ``keySetupSucceeded``
    -- shown once encryption is active, which the setup just made it --
    and each failure through ``keySetupFailed`` over the API client's reader
    of refusals; no promise of its own is left, nor in the settings catalog.

The pure module runs under Node (``_frontend.run_ts``); the panel and the
catalog are read as text; the server's values are read from
``api/routes_security.py`` with ``ast``; KP1 takes the real server's status
through the shared isolation window of the store encryption's contracts.
Local-only. Runs under pytest; KP2 and KP3 under the __main__ runner too
(KP1 needs pytest's temporary directory).
"""

import ast
import json
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _frontend import read, run_ts  # noqa: E402
from _isolation import source  # noqa: E402
from test_store_encryption_contracts import _store, _window  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder
# from the junit file.
BUDGET_S = {
    "test_kp1_the_setup_starts_from_the_key_here_as_the_real_server_says_it": 8.0,
    "test_kp2_a_success_says_the_key_the_server_named_and_a_refusal_its_reason": 6.0,
    "test_kp3_the_panel_says_what_the_module_says_and_no_promise_of_its_own": 3.0,
}

_PURE = "frontend/src/lib/settings/keySetup.ts"
_PANEL = "frontend/src/lib/components/settings/KeyCeremonyPanel.svelte"
_CATALOG = "frontend/src/lib/settings/catalog.ts"

_DRIVER = r"""
const mod = await import(process.env.OO_KEY_SETUP);
const clause = process.argv[2];
const input = JSON.parse(process.env.OO_INPUT);
let result;
if (clause === 'start') {
  result = {
    starts: [...input.statuses.map((status) => mod.keySetupStart(status)), mod.keySetupStart(null),
             mod.keySetupStart(undefined)],
    text: mod.KEY_SETUP_TEXT,
  };
} else {
  result = {
    succeeded: input.answers.map(([mode, answer]) => mod.keySetupSucceeded(mode, answer)),
    failed: input.failures.map(([refusal, error]) => mod.keySetupFailed(refusal, error)),
  };
}
console.log('RESULT ' + JSON.stringify(result));
console.log('PASS ' + clause);
"""


def _run(clause, payload):
    out = run_ts({"OO_KEY_SETUP": _PURE}, _DRIVER, clause, env={"OO_INPUT": json.dumps(payload)})
    lines = [line for line in out.splitlines() if line.startswith("RESULT ")]
    assert len(lines) == 1, out
    return json.loads(lines[0][len("RESULT "):])


def _setup_answers():
    """The values the server's setup gives its answer's ``key``, read from its source; and its refusal of a key file
    that does not open here."""
    tree = ast.parse(source("api", "routes_security.py").read_text(encoding="utf-8"))
    setup = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                 and node.name == "setup_encryption")
    keys = [node.value.value for node in ast.walk(setup) if isinstance(node, ast.Assign)
            and [ast.unparse(target) for target in node.targets] == ["key"]
            and isinstance(node.value, ast.Constant)]
    refusal = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and [ast.unparse(target) for target in node.targets] == ["_KEYFILE_REFUSAL"])
    return keys, ast.literal_eval(refusal)


# ---------------------------------------------------------------------------
# KP1 -- where the setup starts
# ---------------------------------------------------------------------------
def test_kp1_the_setup_starts_from_the_key_here_as_the_real_server_says_it(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch, routes=True)
    try:
        bare = world.routes.get_encryption_status()
        _store(world.places[0] / "auth.db", 2)
        world.dbe.encrypt_all_stores()
        encrypted = world.routes.get_encryption_status()
    finally:
        world.restore()
    assert (encrypted["enabled"], encrypted["has_key"], encrypted["keyfile_exists"]) == (False, False, True), (
        "control: once the command has run, the server loads no key while encryption is off", encrypted)
    statuses = [encrypted, bare, {"has_key": True}, {"env_key_set": True}, {"keyfile_exists": True}, {}]
    got = _run("start", {"statuses": statuses})
    assert got["starts"] == ["existing", "new", "existing", "existing", "existing", "new", "new", "new"], got["starts"]
    text = got["text"]
    assert sorted(text) == ["existing", "new"], sorted(text)
    assert "OPTI_ENCRYPTION_KEY" in text["existing"] and "makes no key" in text["existing"] and (
        "no passphrase" in text["existing"]) and "says why" in text["existing"], text["existing"]
    assert "keep a copy" in text["new"] and "only with that passphrase" in text["new"] and (
        "OPTI_KEYFILE_PASSPHRASE" in text["new"]), text["new"]
    assert not [start for start, words in text.items() if re.search(r"recover|if the key ?file is lost|encrypted with",
                                                                     words, re.I)], (
        "no start promises the data back without the key file, nor that the databases are encrypted", text)


# ---------------------------------------------------------------------------
# KP2 -- what a success and a refusal say
# ---------------------------------------------------------------------------
def test_kp2_a_success_says_the_key_the_server_named_and_a_refusal_its_reason():
    keys, refusal = _setup_answers()
    assert sorted(keys) == ["existing", "made"], ("control: the server's setup names two keys", keys)
    existing, made = "existing", "made"
    status = {"keyfile_path": "/srv/oo/data/.keyfile"}
    command = "python3 -m opti_oignon.db_encryption encrypt"
    answers = [
        ["random", {"setup": True, "key": existing, "plaintext_stores": 0, "wrapped": None, "status": status}],
        ["passphrase", {"setup": True, "key": existing, "plaintext_stores": 0, "status": status}],
        ["random", {"setup": True, "key": existing, "plaintext_stores": 2, "status": status}],
        ["random", {"setup": True, "key": made, "plaintext_stores": 0, "wrapped": False, "status": status}],
        ["random", {"setup": True, "key": made, "plaintext_stores": 0, "wrapped": True, "status": status}],
        ["passphrase", {"setup": True, "key": made, "plaintext_stores": 0, "wrapped": True, "status": status}],
        ["random", {"setup": True, "detail": "Encryption configured successfully"}],
    ]
    failures = [
        [{"status": 409, "detail": refusal},
         {"status": 409, "message": "API error 409", "detail": f"Conflict updating security/encryption: {refusal}."}],
        [None, {"status": 0, "message": "Connection failed", "detail": "Unable to reach the backend."}],
        [None, {"message": "boom"}],
        [None, "a string thrown"],
        [None, None],
    ]
    got = _run("said", {"answers": answers, "failures": failures})
    taken, taken_typed, taken_plain, random_made, random_wrapped, phrase_made, unnamed = got["succeeded"]
    for line in (taken, taken_typed, taken_plain):
        assert line.startswith("Encryption enabled with the key already here: no key was made."), line
        assert "encrypted with" not in line and "was made and saved" not in line, line
    assert "The passphrase you typed was not used." in taken_typed and "passphrase" not in taken, (taken, taken_typed)
    assert command not in taken and "2 database(s) of the platform are still kept in plaintext" in taken_plain and (
        command in taken_plain), taken_plain
    assert random_made.startswith("A random key was made") and status["keyfile_path"] in random_made and (
        "Keep a copy of it" in random_made) and "wrapped" not in random_made, random_made
    assert "wrapped under the server's OPTI_KEYFILE_PASSPHRASE" in random_wrapped and (
        "Keep a copy of the key file and of that passphrase" in random_wrapped), random_wrapped
    assert "wrapped under your passphrase" in phrase_made and (
        "OPTI_KEYFILE_PASSPHRASE set to that passphrase" in phrase_made) and (
        "the passphrase alone does not open your data" in phrase_made), phrase_made
    assert unnamed == "Encryption enabled: Encryption configured successfully.", (
        "an answer that names no key: the server's words, nothing more", unnamed)
    assert got["failed"] == [refusal, "Unable to reach the backend.", "boom", "Setup failed", "Setup failed"], (
        got["failed"])


# ---------------------------------------------------------------------------
# KP3 -- the panel says them
# ---------------------------------------------------------------------------
def test_kp3_the_panel_says_what_the_module_says_and_no_promise_of_its_own():
    panel = read(_PANEL)
    script = panel.split("</script>", 1)[0]
    assert "import { KEY_SETUP_TEXT, keySetupFailed, keySetupStart, keySetupSucceeded } from '$lib/settings/keySetup';" in (
        script) and "import { httpRefusal } from '$lib/switches/serverSwitch';" in script, "the module and the reader"
    assert "$: start = keySetupStart(encStatus);" in script, "the start follows the status read"
    assert "{#if wizardStep === 'choose' && start === 'existing'}" in panel and "start === 'locked'" not in (
        panel), "two starts: the key here, or none"
    for start in ("existing", "new"):
        assert f"{{KEY_SETUP_TEXT.{start}}}" in panel, start
    existing = panel.split("start === 'existing'}", 1)[1].split("{:else", 1)[0]
    assert "on:click={startRandom}" in existing and "<input" not in existing, (
        "the key in place is enabled with one action and no passphrase", existing)
    assert "{#if !encStatus?.enabled || wizardStep === 'success'}" in panel, (
        "what a setup just did stays shown once encryption is active")
    said = re.findall(r"successMessage = ([^;]+);", script)
    assert set(said) == {"''", "keySetupSucceeded('passphrase', result)", "keySetupSucceeded('random', result)"}, said
    assert script.count("setupError = keySetupFailed(httpRefusal(e), e);") == 2 and (
            "setupError = e instanceof Error" not in script), (
        "each failure says the server's reason")
    gone = ("lets you recover", "if the keyfile is lost", "configured with your passphrase", "Remember it",
            "configured with a random key")
    assert not [words for words in gone if words in panel], [words for words in gone if words in panel]
    entry = next(line for line in read(_CATALOG).splitlines() if "id: 'key-ceremony'" in line)
    assert "rotation" not in entry and "enable the key in place" in entry, ("the catalog says what the panel does",
                                                                           entry)


def _run_all():
    cases = (
        ("KP2 what a success and a refusal say",
         test_kp2_a_success_says_the_key_the_server_named_and_a_refusal_its_reason),
        ("KP3 the panel says them", test_kp3_the_panel_says_what_the_module_says_and_no_promise_of_its_own),
    )
    failed = 0
    for name, case in cases:
        try:
            case()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
