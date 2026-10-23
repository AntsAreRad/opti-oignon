#!/usr/bin/env python3
"""Contracts for the page's side of the second factor that belongs to the account (block 177).

Once the factors enrolled from the settings belong to the account, the login asks them: the page must answer its
second step, or an account that enrolled from the settings could not sign in through the page at all. And every
route that adds or removes a credential now asks the account's password: the panels must send it.

  * TF21 -- the WebAuthn transport, run under Node: the server's options in python-fido2's shape (``{publicKey:
    {...}}``, binary fields in base64url) become the browser's buffers for ``navigator.credentials.get`` and
    ``create``, options without the ``publicKey`` member too, and an assertion and an attestation go back as
    base64url in the shape the server reads.
  * TF22 -- the page answers the second step: the store returns the challenge rather than tokens when the first
    step asks a second factor, the login page offers a code when the first step offered a TOTP or a recovery code
    and the security key when it offered one with its options, sends a code (``auto``) or the key's assertion built
    from those options, and a refusal that ends the challenge -- each of the server's sentences for one says "log in
    again", the words the page reads -- sends it back to the first step, the client names
    ``/api/auth/login/2fa`` among the paths that never redirect to the login, the page reads the server's own reason
    (``httpRefusal``, never the client's generic text), and each panel that adds or removes a factor asks the
    password through ``StepUpPrompt`` -- a password field that sends what was typed -- and sends it in the field the
    server reads, showing the server's reason wherever a step-up is refused; the security key is registered under the
    name typed for it, in the field the server reads.
  * TF24 -- the client's DELETE, run under Node with ``fetch`` stood in for: a body goes as JSON with its content
    type, and without one neither is sent.

TF21 runs the module itself under Node's type stripping (``_frontend.run_ts``); TF22 reads the sources. They prove a
form and a transport, never a browser.

Local-only (the public distribution ships no tests).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _frontend import read, run_ts  # noqa: E402

BUDGET_S = {
    "test_tf21_the_webauthn_transport_between_the_server_and_the_browser": 6.0,
    "test_tf22_the_page_answers_the_second_step_and_sends_the_step_up": 2.0,
    "test_tf24_the_clients_delete_sends_its_body_as_json": 6.0,
}

_DRIVER = r"""
const clause = process.argv[2];
const { requestOptions, creationOptions, serializeAssertion, serializeAttestation, bufferToBase64url } =
  await import(process.env.OO_WEBAUTHN);
function fail(why) { console.log('FAIL ' + why); process.exit(1); }
function bytes(buffer) { return Array.from(new Uint8Array(buffer)).join(','); }
function buffer(list) { return new Uint8Array(list).buffer; }

const wrapped = requestOptions({ publicKey: {
  challenge: 'AQID', rpId: 'localhost', userVerification: 'preferred',
  allowCredentials: [{ type: 'public-key', id: 'BAUG' }] } });
if (!(wrapped.challenge instanceof ArrayBuffer) || bytes(wrapped.challenge) !== '1,2,3') fail('request challenge');
if (bytes(wrapped.allowCredentials[0].id) !== '4,5,6' || wrapped.allowCredentials[0].type !== 'public-key') {
  fail('request credential id');
}
if (wrapped.rpId !== 'localhost' || 'publicKey' in wrapped) fail('request options not unwrapped');

const flat = requestOptions({ challenge: 'BwgJ' });
if (bytes(flat.challenge) !== '7,8,9') fail('flat request options');

const created = creationOptions({ publicKey: {
  challenge: 'CgsM', rp: { id: 'localhost', name: 'Opti-Oignon' },
  user: { id: 'dWlk', name: 'alice', displayName: 'alice' },
  excludeCredentials: [{ type: 'public-key', id: 'DQ4P' }] } });
if (bytes(created.challenge) !== '10,11,12' || bytes(created.user.id) !== '117,105,100') fail('creation fields');
if (created.user.name !== 'alice' || bytes(created.excludeCredentials[0].id) !== '13,14,15') fail('creation rest');
if ('publicKey' in created) fail('creation options not unwrapped');

const assertion = serializeAssertion({ id: 'key-1', rawId: buffer([1, 2]), type: 'public-key', response: {
  clientDataJSON: buffer([3]), authenticatorData: buffer([4, 5]), signature: buffer([255, 254]),
  userHandle: buffer([117, 105, 100]) } });
const expected = { id: 'key-1', rawId: 'AQI', type: 'public-key', response: {
  clientDataJSON: 'Aw', authenticatorData: 'BAU', signature: '__4', userHandle: 'dWlk' } };
if (JSON.stringify(assertion) !== JSON.stringify(expected)) fail('assertion ' + JSON.stringify(assertion));
const anonymous = serializeAssertion({ id: 'k', rawId: buffer([1]), type: 'public-key', response: {
  clientDataJSON: buffer([1]), authenticatorData: buffer([1]), signature: buffer([1]), userHandle: null } });
if (anonymous.response.userHandle !== null) fail('a missing user handle');

const attestation = serializeAttestation({ id: 'key-2', rawId: buffer([6]), type: 'public-key', response: {
  clientDataJSON: buffer([7]), attestationObject: buffer([8, 9]) } });
if (attestation.response.attestationObject !== 'CAk' || attestation.rawId !== 'Bg') fail('attestation');
if (bufferToBase64url(buffer([251, 255])) !== '-_8') fail('base64url alphabet');
console.log('PASS ' + clause);
"""


def test_tf21_the_webauthn_transport_between_the_server_and_the_browser():
    run_ts({"OO_WEBAUTHN": "frontend/src/lib/webauthn.ts"}, _DRIVER, "transport")


_PANELS = {
    "frontend/src/lib/components/settings/TOTPSetup.svelte": (
        ("/api/security/2fa/totp/setup", "{ password }"),
        ("/api/security/2fa/totp'", "{ password }")),
    "frontend/src/lib/components/settings/WebAuthnSetup.svelte": (
        ("/api/security/2fa/webauthn/register/begin", "{ password: event.detail.password }"),
        ("/api/security/2fa/webauthn/register/complete", "credential_name: newKeyName || 'Security Key',"),
        ("/api/security/2fa/webauthn/credentials/", "{ password: event.detail.password }")),
    "frontend/src/lib/components/settings/RecoveryCodesPanel.svelte": (
        ("/api/security/2fa/recovery-codes/generate", "{ password: event.detail.password }"),),
    "frontend/src/lib/components/settings/AppPasswordsPanel.svelte": (
        ("/api/security/2fa/app-passwords',", "{ name: newLabel.trim(), password: event.detail.password }"),
        ("/api/security/2fa/app-passwords/", "{ password: event.detail.password }")),
}
# Where a step-up's refusal is shown, each reading the server's reason: TOTPSetup answers both its acts in one place.
_REFUSAL_SITES = {
    "frontend/src/lib/components/settings/TOTPSetup.svelte": 1,
    "frontend/src/lib/components/settings/WebAuthnSetup.svelte": 2,
    "frontend/src/lib/components/settings/RecoveryCodesPanel.svelte": 1,
    "frontend/src/lib/components/settings/AppPasswordsPanel.svelte": 2,
}
# The server's sentences that end a login's challenge: each sends the page back to the first step.
_ENDS_CHALLENGE = (
    "Invalid or expired 2FA challenge. Please log in again.",
    "Too many 2FA attempts. Please log in again.",
    "This login has ended. Please log in again.",
    "This account no longer exists. Please log in again.",
)


def _call(text, path):
    """The API call that names ``path`` as a quoted or templated literal, up to its closing parenthesis."""
    at = next(m.start() for m in re.finditer(re.escape(path), text) if text[m.start() - 1] in "'`")
    start = text.rindex("await api", 0, at)
    depth, i = 0, text.index("(", start)
    while True:
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return text[start:i + 1]
        i += 1


def test_tf22_the_page_answers_the_second_step_and_sends_the_step_up():
    store = read("frontend/src/lib/stores/auth.ts")
    assert re.search(r"if \('requires_2fa' in answer && answer\.requires_2fa\) \{\s*return \{ challenge: answer \};",
                     store), "the store returns the challenge rather than tokens"
    assert "return finishLogin(await authApi.login2fa(answer));" in store
    api = read("frontend/src/lib/api/auth.ts")
    assert "apiPost<AuthTokens>(`${BASE}/login/2fa`, req)" in api
    page = read("frontend/src/routes/login/+page.svelte")
    assert "const given = code.trim();" in page, "the code typed is the code sent"
    assert "doLoginSecondStep({ challenge_id: challengeId, method: 'auto', code: given })" in page
    assert re.search(r"const refusal = httpRefusal\(err\);\s*if \(refusal && refusal\.detail\) return refusal\.detail;",
                     page), "the page reads the server's own reason"
    assert "navigator.credentials.get({ publicKey: requestOptions(options) })" in page
    assert "webauthn_response: serializeAssertion(credential as PublicKeyCredential)" in page
    assert re.search(r"if \(endsChallenge\(message\)\) \{\s*challenge = null;", page), "a dead challenge restarts"
    assert ("$: offersCode = !!challenge && (challenge.methods.includes('totp') || "
            "challenge.methods.includes('recovery'));") in page, "a code is offered when the first step offered one"
    assert ("$: offersKey = !!challenge && challenge.methods.includes('webauthn') && "
            "!!challenge.webauthn_options;") in page, "the key is offered when the first step offered it"
    assert "return message.toLowerCase().includes('log in again');" in page, "the page reads the server's words"
    server = read("opti_oignon/api/routes_auth.py")
    for sentence in _ENDS_CHALLENGE:
        assert sentence in server and "log in again" in sentence.lower(), sentence
    client = read("frontend/src/lib/api/client.ts")
    no_redirect = client[client.index("const AUTH_PATHS_NO_REDIRECT"):client.index("];", client.index(
        "const AUTH_PATHS_NO_REDIRECT"))]
    assert "'/api/auth/login/2fa'" in no_redirect and "2fa-challenge" not in no_redirect
    assert "export async function apiDelete<T>(path: string, body?: unknown): Promise<T>" in client
    assert "...(body !== undefined ? { body: JSON.stringify(body) } : {})" in client
    prompt = read("frontend/src/lib/components/settings/StepUpPrompt.svelte")
    assert re.search(r'<Input\s+type="password"', prompt), "the password is typed in a password field"
    assert "const given = password;" in prompt and "dispatch('confirm', { password: given });" in prompt
    for panel, calls in _PANELS.items():
        text = read(panel)
        assert "import StepUpPrompt from './StepUpPrompt.svelte';" in text, panel
        assert "<StepUpPrompt" in text and "on:confirm=" in text, panel
        assert text.count("httpRefusal(e)?.detail") == _REFUSAL_SITES[panel], (panel, "the server's own reason")
        for path, payload in calls:
            call = _call(text, path)
            assert payload in call, (panel, path, call)


_DELETE_DRIVER = r"""
const clause = process.argv[2];
const calls = [];
globalThis.document = { cookie: 'oo_csrf_token=csrf-value' };
globalThis.window = { location: { origin: 'http://localhost:5173', pathname: '/settings' } };
globalThis.fetch = async (url, init) => {
  calls.push({ url, init });
  return new Response(JSON.stringify({ success: true }), { status: 200, headers: { 'Content-Type': 'application/json' } });
};
const { apiDelete } = await import(process.env.OO_CLIENT);
function fail(why) { console.log('FAIL ' + why); process.exit(1); }
const answer = await apiDelete('/api/security/2fa/totp', { password: 'typed' });
if (!answer || answer.success !== true) fail('the answer');
const withBody = calls[0].init;
if (withBody.method !== 'DELETE') fail('method ' + withBody.method);
if (withBody.headers['Content-Type'] !== 'application/json') fail('content type ' + JSON.stringify(withBody.headers));
if (withBody.body !== JSON.stringify({ password: 'typed' })) fail('body ' + withBody.body);
if (withBody.credentials !== 'include') fail('credentials');
if (withBody.headers['X-CSRF-Token'] !== 'csrf-value') fail('the CSRF header');
if (calls[0].url !== 'http://localhost:5173/api/security/2fa/totp') fail('url ' + calls[0].url);
await apiDelete('/api/conversations/x');
const without = calls[1].init;
if ('body' in without || 'Content-Type' in without.headers) fail('no body, no content type ' + JSON.stringify(without));
console.log('PASS ' + clause);
"""


def test_tf24_the_clients_delete_sends_its_body_as_json():
    run_ts({"OO_CLIENT": "frontend/src/lib/api/client.ts"}, _DELETE_DRIVER, "delete")
