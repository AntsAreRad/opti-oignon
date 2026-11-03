#!/usr/bin/env python3
"""Contracts for the page's side of the enrolment at login (block 225).

In Bulbe the login (and the registration) of an account with no second factor answers the challenge of an
enrolment, not tokens (``test_bulbe_login_enrols_a_factor_contracts.py``): the page must answer it, or such an
account could not sign in through the page at all.

  * EN6 -- the page answers the enrolment: the store returns it rather than tokens, from the login before any
    challenge is read and from the registration; the client calls the two enrolment routes the server serves; the
    server's middleware lets both through without a session and the client never sends a refusal of theirs to the
    login; the step begins the TOTP when it shows, shows the QR code and the secret -- saying it replaces any shown
    before --, sends the code typed, and shows
    the recovery codes the answer carries -- the confirmation stores no token and sets no user: the root layout sends
    a signed-in user on the login page away at once, so the session begins only when the codes were kept, and then
    the page goes on; a refusal whose words say "log in again" --
    each of the server's sentences that ends an enrolment says it -- sends back to the password, on the login page
    with the server's reason, and to the login from the registration.

EN6 reads the sources: it proves a form, never a browser.

Local-only (the public distribution ships no tests).
"""

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _frontend import read  # noqa: E402

BUDGET_S = {
    "test_en6_the_page_answers_the_enrolment": 2.0,
}

BEGIN, COMPLETE = "/api/auth/login/2fa/enrol/begin", "/api/auth/login/2fa/enrol/complete"
STEP = "frontend/src/lib/components/auth/EnrolSecondFactor.svelte"

# The server's sentences that end an enrolment: each sends the page back to the password.
_ENDS_ENROLMENT = (
    "Invalid or expired enrolment. Please log in again.",
    "This login enrols a second factor first. Please log in again.",
    "This account has a second factor now. Please log in again.",
    "Too many 2FA attempts. Please log in again.",
    "This login has ended. Please log in again.",
    "This account no longer exists. Please log in again.",
)


def _function(text, name):
    """The exported async function ``name``, from its declaration to its closing brace."""
    start = text.index(f"export async function {name}(")
    return text[start:text.index("\n}\n", start) + 2]


def _public_prefixes():
    """The middleware's public prefixes, as the module declares them."""
    tree = ast.parse(read("opti_oignon/api/auth_middleware.py"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_PUBLIC_PREFIXES" for t in node.targets):
            return [ast.literal_eval(element) for element in node.value.elts]
    raise AssertionError("no _PUBLIC_PREFIXES in the middleware")


def test_en6_the_page_answers_the_enrolment():
    store = read("frontend/src/lib/stores/auth.ts")
    enrols = (r"if \('requires_2fa_enrolment' in answer && answer\.requires_2fa_enrolment\) \{\s*"
              r"return \{ enrolment: answer \};")
    login, register = _function(store, "doLogin"), _function(store, "doRegister")
    for body in (login, register):
        assert re.search(enrols, body), "the store returns the enrolment rather than tokens"
    assert re.search(enrols, login).start() < login.index("'requires_2fa' in answer"), "read before any challenge"
    complete = _function(store, "completeEnrolment")
    assert "authApi.enrolComplete({ challenge_id: challengeId, code })" in complete
    assert "return { tokens: answer, recoveryCodes: answer.recovery_codes ?? [] };" in complete
    assert not re.search(r"finishLogin|storeTokens|currentUser|scheduleRefresh", complete), (
        "the confirmation begins no session")
    assert "return finishLogin(tokens);" in _function(store, "finishEnrolment")
    layout = read("frontend/src/routes/+layout.svelte")
    assert re.search(r"\$currentUser && isPublicRoute", layout), "the layout that would leave the codes unseen"
    assert "return authApi.enrolBegin({ challenge_id: challengeId });" in _function(store, "beginEnrolment")

    api = read("frontend/src/lib/api/auth.ts")
    assert "apiPost<TwoFactorEnrolmentSecret>(`${BASE}" + BEGIN.removeprefix("/api/auth") + "`, req)" in api
    assert "apiPost<EnrolledTokens>(`${BASE}" + COMPLETE.removeprefix("/api/auth") + "`, req)" in api
    server = read("opti_oignon/api/routes_auth.py")
    for path in (BEGIN, COMPLETE):
        assert re.search(r'@router\.post\("' + re.escape(path.removeprefix("/api/auth")) + r'"', server), path
    for sentence in _ENDS_ENROLMENT:
        assert sentence in server and "log in again" in sentence.lower(), sentence

    prefixes = _public_prefixes()
    client = read("frontend/src/lib/api/client.ts")
    listed = client[client.index("const AUTH_PATHS_NO_REDIRECT"):client.index("];", client.index(
        "const AUTH_PATHS_NO_REDIRECT"))]
    no_redirect = re.findall(r"'([^']+)'", listed)
    assert "if (AUTH_PATHS_NO_REDIRECT.some((p) => path.startsWith(p))) return;" in client
    for path in (BEGIN, COMPLETE):
        assert any(path.startswith(prefix) for prefix in prefixes), (path, "reached without a session")
        assert any(path.startswith(prefix) for prefix in no_redirect), (path, "never sent to the login")

    step = read(STEP)
    assert "onMount(begin);" in step and "begun = await beginEnrolment(enrolment.challenge_id);" in step
    assert "<img src={begun.qr_code_base64}" in step and "{begun.secret}" in step, "the QR code and the secret"
    assert "This secret replaces any shown before for this account" in step, "a new secret voids an earlier scan"
    assert "const given = code.trim();" in step
    assert re.search(r"const confirmed = await completeEnrolment\(enrolment\.challenge_id, given\);\s*"
                     r"session = confirmed\.tokens;\s*recoveryCodes = confirmed\.recoveryCodes;", step)
    assert re.search(r"await finishEnrolment\(session\);\s*dispatch\('done', null\);", step), (
        "the session begins once the codes were kept, then the page goes on")
    assert "on:click={proceed}>" in step and step.count("finishEnrolment(") == 1
    assert "{recoveryCodes.join('\\n')}" in step, "the recovery codes shown"
    assert re.search(r"if \(message\.toLowerCase\(\)\.includes\('log in again'\)\) \{\s*dispatch\('restart', "
                     r"\{ reason: message \}\);", step), "an ended enrolment restarts"

    page = read("frontend/src/routes/login/+page.svelte")
    assert "import EnrolSecondFactor from '$lib/components/auth/EnrolSecondFactor.svelte';" in page
    assert re.search(r"if \('enrolment' in outcome\) \{\s*enrolment = outcome\.enrolment;", page)
    assert page.index("if ('enrolment' in outcome)") < page.index("if ('challenge' in outcome)")
    assert re.search(r"\{#if enrolment\}\s*<EnrolSecondFactor \{enrolment\} on:done=\{\(\) => goto\('/chat'\)\} "
                     r"on:restart=\{restartEnrolment\} />\s*\{:else if challenge\}", page)
    assert re.search(r"function restartEnrolment\(event: CustomEvent<\{ reason: string \}>\) \{\s*enrolment = null;"
                     r"\s*errorMsg = event\.detail\.reason;", page), "back to the password, with why"

    registration = read("frontend/src/routes/register/+page.svelte")
    assert "import EnrolSecondFactor from '$lib/components/auth/EnrolSecondFactor.svelte';" in registration
    assert re.search(r"const outcome = await doRegister\(username\.trim\(\), password, email\.trim\(\)\);\s*"
                     r"if \('enrolment' in outcome\) \{\s*enrolment = outcome\.enrolment;", registration)
    assert ("<EnrolSecondFactor {enrolment} on:done={() => goto('/chat')} on:restart={() => goto('/login')} />"
            in registration)
