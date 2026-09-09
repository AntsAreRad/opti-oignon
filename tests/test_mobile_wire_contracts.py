#!/usr/bin/env python3
"""Contracts for the wire the mobile client and the desktop responder share.

The phone does not speak the HTTP surface. It speaks JSON envelopes over a
Veilid app_call, and the two ends of that conversation are written in
different languages, in different trees, with nothing between them: the
Kotlin client declares the field names and the refusal vocabulary it
understands, the Python responder emits its own, and until these contracts
existed the two agreed only by coincidence. Either side could add a field,
rename a reason or move the protocol version and the whole suite stayed
green while the phone stopped working.

The coincidence is load-bearing in one direction more than the other. The
Kotlin codec is configured to reject unknown keys, so a single field added
to a reply does not degrade the phone -- it raises inside the decoder. That
makes the reply shape the sharpest edge on the wire and MW1 the contract
that matters most.

  * MW1 -- the reply fields the client declares are exactly the keys the
    responder emits, measured by calling the responder rather than by
    reading it.
  * MW2 -- the refusal vocabulary is the same set on both sides.
  * MW3 -- the request envelope types are the same pair on both sides.
  * MW4 -- the protocol version is the same integer on both sides.
  * MW5 -- the client codec is strict, which is what makes MW1 matter.
  * MW6 -- the request envelopes the client emits are well-formed to the
    responder, again by calling it.
  * MW7 -- the notes client offers no way to opt a note into phone sync.

MW2 reads the responder with the syntax tree, never with a line-oriented
pattern: one of the eleven reasons is emitted through a local alias with
its argument on the following line, and a pattern narrow enough to be
readable is narrow enough to miss it and report a disagreement that does
not exist.

Local-only. Runs under pytest; imports the application the way the other
published-surface contracts do.
"""

import ast
import re
from pathlib import Path

from opti_oignon.veilid import protocol, remote_inference

REPO = Path(__file__).resolve().parent.parent
MOBILE = REPO / "android" / "app" / "src" / "main" / "kotlin" / "org" / "optioignon" / "mobile"
ENVELOPES = MOBILE / "wire" / "Envelopes.kt"
NOTES_SYNC = MOBILE / "sync" / "NotesSyncContract.kt"
RESPONDER = REPO / "opti_oignon" / "veilid" / "remote_inference.py"

# Helpers that build a refusal, mapped to the position of the reason in
# their argument list. A helper added here without its position is the one
# way this contract can go quiet, so the mapping is asserted non-empty.
_REFUSAL_HELPERS = {"_refusal": 1, "_ref": 0}

# Names that would let the phone opt a note into sync. The desktop owns
# that decision; the phone is a consumer and must carry no such verb.
_OPT_IN_NAMES = ("optIn", "setMobileAllowed", "setPhoneSync", "allowOnPhone", "grant")


def _kotlin_source(path):
    return path.read_text(encoding="utf-8")


def _serial_names(source, data_class):
    """Return the wire names the given Kotlin data class declares."""
    match = re.search(
        r"data class " + data_class + r"\((.*?)\n\)", source, re.S
    )
    assert match, f"{data_class} not found in {ENVELOPES.name}"
    return set(re.findall(r'@SerialName\("([^"]+)"\)', match.group(1)))


def _kotlin_reasons(source):
    """Return the refusal vocabulary the client declares."""
    tail = source.split("object Reason")
    assert len(tail) == 2, "the Reason object is not declared exactly once"
    return set(re.findall(r'const val \w+ = "([^"]+)"', tail[1]))


def _python_reasons():
    """Return the refusal vocabulary the responder emits.

    Walks the syntax tree rather than the lines: a refusal whose reason sits
    on the line after the call is invisible to a pattern and would read as a
    vocabulary the client invented.
    """
    tree = ast.parse(RESPONDER.read_text(encoding="utf-8"))
    reasons = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        position = _REFUSAL_HELPERS.get(node.func.id)
        if position is None or len(node.args) <= position:
            continue
        argument = node.args[position]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            reasons.add(argument.value)
    return reasons


def test_mw1_the_client_declares_exactly_the_reply_keys_the_responder_emits():
    emitted = set(remote_inference._refusal("r", "rate_limited", "d"))
    emitted |= set(
        remote_inference._stream_reply(
            protocol.MSG_REMOTE_INFER, "r", "text", 0, True
        )
    )
    declared = _serial_names(_kotlin_source(ENVELOPES), "InferReply")
    unknown = sorted(emitted - declared)
    assert not unknown, (
        f"the responder emits {unknown} which the client does not declare; "
        "its codec rejects unknown keys, so this raises on the phone"
    )
    unused = sorted(declared - emitted)
    assert not unused, (
        f"the client declares {unused} which the responder never emits"
    )


def test_mw2_the_refusal_vocabulary_is_the_same_set_on_both_sides():
    assert _REFUSAL_HELPERS, "no refusal helper is being read"
    client = _kotlin_reasons(_kotlin_source(ENVELOPES))
    server = _python_reasons()
    assert client == server, (
        f"client-only reasons {sorted(client - server)}; "
        f"server-only reasons {sorted(server - client)}"
    )


def test_mw3_the_request_envelope_types_are_the_same_pair():
    source = _kotlin_source(ENVELOPES)
    client = set(re.findall(r'const val TYPE_\w+ = "([^"]+)"', source))
    server = {protocol.MSG_REMOTE_INFER, protocol.MSG_REMOTE_INFER_CONT}
    assert client == server, (
        f"client types {sorted(client)} against server types {sorted(server)}"
    )


def test_mw4_the_protocol_version_is_the_same_integer():
    source = _kotlin_source(ENVELOPES)
    match = re.search(r"const val PROTOCOL_VERSION = (\d+)", source)
    assert match, "the client does not pin a protocol version"
    assert int(match.group(1)) == protocol.PROTOCOL_VERSION


def test_mw5_the_client_codec_rejects_unknown_keys():
    source = _kotlin_source(ENVELOPES)
    assert "ignoreUnknownKeys = false" in source, (
        "the client codec tolerates unknown keys, which silently retires the "
        "guarantee the reply-shape contract exists to hold"
    )
    assert "explicitNulls = false" in source, (
        "the client codec would put an explicit null on the wire"
    )


def test_mw6_the_request_envelopes_the_client_emits_are_well_formed():
    source = _kotlin_source(ENVELOPES)
    initial = {
        name: value
        for name, value in (
            ("v", protocol.PROTOCOL_VERSION),
            ("type", protocol.MSG_REMOTE_INFER),
            ("device", "a-device"),
            ("request_id", "a-request"),
            ("prompt", "a prompt"),
        )
        if name in _serial_names(source, "InferRequest")
    }
    continuation = {
        name: value
        for name, value in (
            ("v", protocol.PROTOCOL_VERSION),
            ("type", protocol.MSG_REMOTE_INFER_CONT),
            ("device", "a-device"),
            ("request_id", "a-request"),
            ("cursor", 0),
        )
        if name in _serial_names(source, "ContRequest")
    }
    served = (
        remote_inference.serve_remote_inference(initial),
        remote_inference.serve_remote_inference_continuation(continuation),
    )
    for reply in served:
        assert reply.get("reason") != "malformed", (
            "the responder reads an envelope the client emits as malformed: "
            f"{reply.get('detail')!r}"
        )


def test_mw7_the_notes_client_cannot_opt_a_note_into_phone_sync():
    source = _kotlin_source(NOTES_SYNC)
    declarations = re.findall(r"\bfun\s+(\w+)", source)
    offending = [
        name
        for name in declarations
        if any(verb.lower() in name.lower() for verb in _OPT_IN_NAMES)
    ]
    assert not offending, (
        f"the phone declares {offending}; opting a note into phone sync is a "
        "desktop-side trust decision and must not exist on the consumer"
    )
