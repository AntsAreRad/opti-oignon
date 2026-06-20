# Remote Inference

Remote inference lets a paired, lower-trust device -- the phone -- borrow this
desktop's local models over a private [Veilid](https://veilid.com) route. The
phone sends a chat request over the route it already uses for sync; the desktop
runs it through the same executor and admission path a local chat uses; the
response streams back. Neither the model, the prompt, nor the response ever leaves
your own two machines, and there is no server in the middle.

This is a Daily-mode capability. Like the rest of sync, it is refused under Bulbe
at the binding layer, not by a policy flag: a node cannot come up while Bulbe is
active, so a remote request in Bulbe cannot arrive at all.

## The bounded surface (tier 1)

What a remote device may ask for is deliberately narrow. The served surface is the
tier 1 bounded surface: inference and RAG read-only, and nothing else -- no
state-mutation tools, no sandbox, no filesystem, no shell, no config. The bound is
enforced before any generation begins. Fuller, elevated access (tier 2) belongs to
the Mobile app cycle's per-device ceremony, not here.

A remote request is authenticated twice over: by the private route to a paired
peer, and by the per-record post-quantum signature (VL-01) on any record the
request references. It is then built into an ordinary chat request and submitted to
the executor, so it traverses the same resource-governor admission funnel as a
local request. The handler never calls the backend directly and adds no admission
bypass.

## Per-device grants and the RAG sub-grant

Each paired device carries its own remote-chat grant, stored on the peer registry.
The default is tier 1: remote chat is enabled, and a separate RAG read-only
sub-grant is off. A request that asks for a RAG scope is refused unless the asking
device's RAG sub-grant has been turned on. The two controls are independent, so you
can let a device chat against your models without letting it read your RAG
collections.

## Revocation

Revoking a device is immediate and total. It flips the device's grant off (the
durable half) and kills its in-flight streaming sessions at once (the live half),
so a response already in flight stops being pullable. Unpairing a device detaches
it the same way -- it is wired to the same kill, with no new primitive. Both ride
the emergency-stop posture: nothing about the remote surface survives a revoke.

## Streaming (Option A, pull)

A chat response is many chunks produced over time, which a single reply cannot
carry. The channel uses Option A -- chunked app_calls, pull. The desktop begins
generating and buffers the response keyed by the request id; the device then issues
successive app_calls, each carrying the request id and a cursor, that each return
the next chunk until a done marker. A single-chunk reply is done at once. The
device's pull rate is the backpressure, and the server-side buffer is bounded per
in-flight request and bound to the device and the request id -- any mismatch is a
refusal. A stalled or hostile peer surfaces as a timeout and never wedges the
desktop.

## The rate limit

Each device is bounded by a fixed-window rate limit. A device that exceeds its
window gets a structured refusal rather than the channel absorbing unbounded work,
and an alert is recorded in the channel telemetry for the desktop to surface.

## Audit

Everything is audit-chained on the same hash-chain trail the sync exchange already
writes to: the grant, every remote request served, every refusal, and every
revocation. The trail is retained by design and never travels on the remote
surface.

## The desktop control surface

The desktop owns the controls. Under the sync router, at SYN-06 router
authentication parity (every endpoint authenticated), the surface exposes a
device's remote-chat grant, a setter for the grant and the RAG sub-grant, a revoke,
and the channel's rate/telemetry state. These are surfaced in a RemoteChannelPanel
mounted in the Sync settings panel, where you manage per-device grants, the RAG
sub-grant, revocation, and see the live sessions and rate alerts.

The four endpoints, under `/api/sync`:

- `GET /peers/{peer_id}/remote-chat` -- a device's grant state.
- `POST /peers/{peer_id}/remote-chat` -- enable or disable remote chat, set the RAG sub-grant.
- `POST /peers/{peer_id}/remote-chat/revoke` -- revoke: flip the grant and kill live sessions.
- `GET /remote-chat/telemetry` -- the per-device rate/telemetry and live-session state.

## Host-assured behaviour

The wire logic is fully proven in the container. What only a real machine can
confirm -- the live private route between two devices, the streamed per-chunk
round-trip latency, and end-to-end remote chat -- is host-assured and never
simulated in-container. The live walk is directed by `HOST_SHAKEDOWN_S236.md`. The
phone client itself is the Mobile app cycle's work; here the wire is validated by a
desktop-to-desktop exercise over a real route.
