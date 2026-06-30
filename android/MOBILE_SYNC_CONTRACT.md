# Mobile Sync Contract

This document is the implementer's contract for the Android client. It is the
exact wire behaviour the desktop responder already enforces and proves, restated
from the responder's point of view so the phone can be coded against it. Every
field name, message type, and refusal reason below is the literal value the
desktop emits; the phone MUST speak exactly this.

The desktop side of every guarantee here is exercised in-process by the backend
test suite. This client code is **not** validated in that environment (there is
no Android toolchain there); it is validated host-side. See `BUILD_RUNBOOK.md`.

## Transport

The phone reaches a paired desktop over a Veilid private route -- the same route
it uses for note/vault sync. All three request kinds ride one `app_call`
transport and are discriminated by the envelope `type`:

- `delta_request` / `record_batch` -- sync (notes, vault).
- `remote_infer` -- borrow the desktop's local model for one chat turn.
- `remote_infer_cont` -- pull the next chunk of a streamed reply.

The desktop responder decodes the request, reads `type`, and routes it. The peer
identity is the **route-authenticated** peer, never a field in the request
payload. A request payload that claims a different device than the route is
refused (`provenance_mismatch`).

## Network isolation (Bulbe)

Bulbe is a physical, binding-layer state on the desktop, not a policy flag: the
desktop node does not bind while Bulbe is active, so a request cannot arrive at
all. If a request is somehow processed under Bulbe, the responder **raises and
sends no reply** -- the phone observes a transport timeout / no answer, NOT a
refusal envelope. The client must treat "no reply" as a terminal, non-retryable
state for that route until sync is otherwise available; it must never interpret
silence as success.

## Remote inference -- request envelopes

Initial request (phone -> desktop):

```json
{ "v": 1, "type": "remote_infer", "device": "<this device id>",
  "request_id": "<fresh unique id>", "prompt": "<user prompt>" }
```

The optional `rag` field requests RAG read-only scope. It is a **separate
per-device sub-grant**, off by default. A request carrying `rag` is refused
(`rag_not_granted`) unless the desktop has turned that device's sub-grant on.

Continuation request (phone -> desktop), to pull the next chunk:

```json
{ "v": 1, "type": "remote_infer_cont", "device": "<this device id>",
  "request_id": "<the same id>", "cursor": <int from the previous reply> }
```

The surface is closed. The initial request may carry ONLY
`{v, type, device, request_id, prompt, rag}`; the continuation may carry ONLY
`{v, type, device, request_id, cursor}`. Any other field -- including any tool,
sandbox, filesystem, shell, config, or pipeline field -- is **refused, not
silently dropped** (`out_of_surface`). This is the tier 1 bounded surface.

## Remote inference -- reply envelopes

Success (one chunk of the stream):

```json
{ "v": 1, "type": "remote_infer" | "remote_infer_cont", "ok": true,
  "request_id": "<id>", "content": "<chunk text>",
  "cursor": <next cursor>, "done": <bool> }
```

Refusal (always structured, never a silent failure):

```json
{ "v": 1, "type": "remote_infer" | "remote_infer_cont", "ok": false,
  "refused": true, "request_id": "<id>", "reason": "<code>", "detail": "<text>" }
```

The reply `type` echoes the request kind, so a refusal is parsed by the same
branch as the matching success reply.

## The pull loop

1. Send the initial `remote_infer`. The reply is the first chunk:
   `{content, cursor, done}`.
2. While `done` is `false`, send a `remote_infer_cont` with `cursor` set to the
   `cursor` from the previous reply. Append each `content`.
3. Stop when a reply has `done: true`. That chunk is the last one; the
   desktop-side buffer is consumed on that pull.

A single-chunk reply is `done: true` immediately -- the first reply is the whole
answer.

Continuations are cheap buffer reads and do **not** consume the device's rate
budget. Only the initial `remote_infer` is rate-limited.

## Refusal reasons (the complete set)

| reason | meaning | client action |
|---|---|---|
| `malformed` | envelope/version/prompt/cursor invalid | fix the request; do not retry as-is |
| `provenance_mismatch` | `device` field != route peer | send the route's own device id |
| `no_authenticated_identity` | no route peer and no device | re-establish the route |
| `unknown_device` | not a registered peer | (re)pair the device |
| `peer_not_confirmed` | pairing awaits mutual confirmation | complete pairing confirmation |
| `remote_chat_disabled` | this device's remote-chat grant is off | ask the desktop to enable it |
| `rag_not_granted` | `rag` requested without the sub-grant | drop `rag`, or ask the desktop to grant it |
| `out_of_surface` | a field outside the bounded surface | remove the field |
| `rate_limited` | over the fixed per-device window | back off, retry later |
| `execution_error` | the desktop could not complete the turn | retry later |
| `buffer_mismatch` | no in-flight stream for this device + id + cursor | restart with a fresh `remote_infer` |

`buffer_mismatch` is the streaming-specific case: the stream was already
consumed, evicted under load, revoked, or the `(device, request_id)` did not
match a live buffer. A device can therefore never read another device's stream.
The only recovery is a new `remote_infer` with a new `request_id`.

## Desktop-side bounds the client should expect

These are enforced on the desktop; the phone does not set them, but should be
built to tolerate them:

- A very long reply may be truncated at the desktop's per-reply byte cap. The
  client must treat a `done: true` as authoritative regardless of length.
- The desktop bounds concurrent in-flight streams; a long-abandoned stream may
  be evicted (a later pull then returns `buffer_mismatch`).
- A revoke or unpair at the desktop kills this device's in-flight streams at
  once; the next pull returns `buffer_mismatch`. The grant is also flipped off
  durably, so a fresh `remote_infer` is refused (`remote_chat_disabled`).

## Notes sync -- the phone is a consumer, never a controller

The per-note phone-sync opt-in is a human trust decision made **only at the
desktop**. The phone:

- NEVER sets or sees the opt-in flag. It is not on the sync wire (the outbound
  payload omits it); the phone cannot read it and has no message that flips it.
- Receives only notes the desktop has opted in, filtered at serve time on the
  desktop against a live lookup -- not by anything the phone sends.
- When the desktop newly opts a note in, that note is republished full-state, so
  a phone already past its sync watermark still receives the newly allowed note
  on its next pull.

The client must not implement, request, or assume any path that sets the opt-in.
Sync is watermark-based pull, identical to the desktop-to-desktop contract; the
phone-class serve filter is the only difference, and it lives on the desktop.

## What the client must implement (checklist)

- [ ] Encode/decode the three envelopes above, exactly these field names.
- [ ] Send the route's own device id in `device`; never spoof another.
- [ ] The pull loop, terminating on `done: true`.
- [ ] Map every refusal `reason` to a user-meaningful state per the table.
- [ ] Treat "no reply" (Bulbe / route down) as terminal, never as success.
- [ ] Never request `rag` unless the desktop has granted the sub-grant.
- [ ] Never attempt to set the notes opt-in; sync is pull-only.
