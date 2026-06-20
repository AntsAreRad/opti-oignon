# Veilid Sync

Veilid sync is an optional capability that lets you carry conversations, memory,
and skills across your own devices over [Veilid](https://veilid.com), a
privacy-first, end-to-end encrypted peer-to-peer overlay. There is no server in
the middle: devices reach each other through private routes, addressed by keys
you hold.

Sync is a Daily-mode capability. It is disabled under Bulbe at the binding layer,
not by a policy flag, so a node cannot come up while Bulbe is active. It is also
fully optional: the bindings and the headless server are installed only when you
ask for them, and nothing about sync runs unless you start it.

This page describes the foundation -- the node lifecycle, the client wrapper, the
packaging script, and how to install the optional dependency -- the sync protocol
that moves records between your devices, and the live route, transport, and
sync-status surface that drive a real exchange. The pairing interface arrives in a
later session.

## The node lifecycle

A Veilid node moves through a small set of states. It is **stopped** until you
start it; **started** once it is up and connected to the local veilid-server but
not yet on the network; **attached** once it has joined the network; and back to
**started** when it detaches. Stopping returns it to **stopped**.

Four operations drive these transitions: start, attach, detach, and stop. Start
and attach are refused under Bulbe. Detach and stop are never refused, so a node
can always leave the network and shut down, even immediately after a mode change.
Every transition is fail-secure: if an underlying step fails, the node settles in
a truthful state (it reverts to started after a failed attach, and a failed stop
still ends stopped) and surfaces a typed error rather than leaking an arbitrary
one.

## The client wrapper

Veilid's API is asyncio-native; most of Opti-Oignon is not. The client wrapper
owns a single event loop on a dedicated background thread and submits Veilid
work to it, so the framework never runs on, and never blocks, the application's
main event loop. A synchronous surface drives the node from its own thread, and
an async surface lets a future route await the same work without blocking its
loop. Every operation is bounded by a timeout, so a stalled peer can never wedge
a caller, and only typed errors ever escape.

## Staging the server

Veilid talks to a headless veilid-server. The script
`scripts/fetch_veilid_server.py` stages that binary for your platform: it
downloads the pinned release artifact over HTTPS, verifies it against a pinned
SHA-256 (and an optional signature), and only then places it under the data
directory with an exec bit. The pinned checksum ships empty on purpose: you paste
the checksum Veilid publishes for the release, verified out of band, before
staging. With no pinned or supplied checksum the script refuses to stage rather
than trust an unverified binary, and on any mismatch it removes the download and
stages nothing. The script performs no network at import and never runs the
binary it stages.

Preview the plan without touching the network:

```
python scripts/fetch_veilid_server.py --print-plan
```

## Installing

The Veilid Python bindings are an optional dependency, kept out of the blanket
install just like the llama and SQLCipher extras, because the bindings and the
server are platform-specific. Install them explicitly:

```
pip install opti-oignon[veilid]
```

Then stage the server with the script above. Sync remains Daily-only; on a Bulbe
deployment, leave the extra uninstalled.

## The sync protocol

Sync is convergent: every device holds the full set of records and reconciles
incoming changes, with no primary. What syncs is your conversations, your two-tier
memory (a canonical tier and an archive tier), and your skills registry, between
your own devices.

A record is the unit that moves. Each one carries a stable identity, a logical
clock that acts as its version, a content hash, the device that produced it, and a
kind that namespaces the identity so the two memory tiers never collide; a record
can also be a tombstone, so a deletion converges like any other change. The
encoding round-trips and is defensive: an object that is not a well-formed record,
or whose hash does not match its content, is rejected rather than trusted, so a
tampered record is dropped before it is reconciled.

Reconciliation is last-writer-wins per record by the logical clock, with a
deterministic tie-break on content hash and then device id. A version superseded by
a higher clock is a clean update. When two versions tie on the clock with different
content -- a concurrent edit on two devices -- the tie-break picks one and the loser
is kept in a conflict log rather than dropped, so nothing is lost where an automatic
merge would be unsafe. The merge is idempotent and order-independent, so two devices
reach the same set regardless of who reconciles first.

A device journals every change it makes in a per-device change feed, an
append-only SQLite log under the data directory. A peer asks for the delta since a
watermark and receives the records changed after it, plus a new watermark to advance
to. Journalling is a local operation, so it works in any mode; only moving a delta
over the wire is Daily-only.

An exchange is a pull. A device builds a delta request from its watermark, the peer
answers with a batch of records from its feed, and the device applies the batch by
reconciling it into its own set, journalling only what changed, and advancing its
watermark. Every step that acts over the wire -- building a request or a batch,
responding, applying a batch -- refuses under Bulbe at the binding layer; parsing an
incoming message and reconciling are pure and run in any mode, because reading data
is not acting on the wire.

## The sync route and the per-peer watermark

The protocol is driven by an engine behind a thin HTTP route, the same shape as the
agent route. Three pieces make it up.

A per-peer store records which peers this device is paired with and how far it has
consumed each one. Per peer it holds the pairing identity -- a stable peer id and
the peer's public routing key, plus an optional label -- and a watermark, the last
peer-feed sequence this device has applied. The store is SQLite in WAL mode under
the data directory. Re-pairing a peer (a rotated route) refreshes its routing key
and label but keeps its watermark and its original pairing time, so a device never
loses track of how far it has synced, and the watermark only ever moves forward.
Managing the paired set is a local operation, allowed in any mode; what is
Daily-only is running a round.

The sync engine runs a round: it reads the peer's watermark, pulls the delta from
the peer through the protocol envelope in bounded chunks, verifies each record's
signature against its origin device's registered key, reconciles what verifies into
the local set (journalling only what changed, so a round is idempotent), and
advances the watermark past every consumed chunk. Applying a skill is a sensitive
action and passes the same human approval gate as the agent's memory and skills
writes; a record that is not approved is not applied -- it is quarantined to a
per-record pending-approvals ledger (full envelope, provenance shown in the panel,
never the body), so a record awaiting your decision never blocks the rest of sync
and is never re-prompted. Approving it later re-verifies against the trust state at
that moment -- a signing key that changed in the meantime refuses honestly; refusing
removes it without applying. Conversation and memory records are user data and apply
without a gate. Every round, every peer change, and every approval decision is
recorded in the hash-chain audit log.

The route exposes the read and run halves a panel will use: list the paired peers,
read one peer's status or its watermark, and run a pull round. A round is
Daily-only, so under Bulbe the route refuses through the binding-layer gate; an
unpaired peer is a clean not-found. The run handler resolves a live peer over the
Veilid transport and runs the round against it.

## The live transport

A round reaches a peer over a Veilid private route. A live peer satisfies the same
pull contract the protocol defines, but instead of answering from a local feed it
sends the delta request to the peer's public routing key and waits for the reply.
The send is one request/response over the route, bounded by the client's timeout:
a stalled or hostile peer surfaces as a timeout rather than hanging the caller, and
the route maps that to a clear status (a gateway timeout). The peer's answer is
parsed defensively, so a garbled reply degrades to an empty round that holds the
watermark rather than an error. Resolving a live peer needs the framework installed
and the node attached; without an attached transport the route reports that live
sync is not available, while the round contract itself is exercised against a fake
peer in tests. The transport gates under Bulbe like every wire-acting step, and the
node will not bind under Bulbe in any case.

## Serving a peer

Sync is bidirectional: a paired device both pulls from its peers and serves them.
The responder answers an inbound delta request with a batch drawn from this
device's change feed, the mirror of the pull. It refuses under Bulbe at the same
binding-layer gate, so a device under Bulbe neither reaches peers nor answers them,
and every served answer is recorded in the hash-chain audit log. An unparseable
request gets a benign empty answer rather than an over-send.

What moves over a round is real data: each syncable domain -- a conversation, the
two memory tiers, a skill -- has a producer that turns it into a record, and a
device journals a produced record locally in any mode so a peer can pull it.

## Sync status

The route reports sync status at `GET /api/sync/status`: whether sync is running
(from the node's live state), the most recent round, and, per peer, when it last
synced and how its last round went. Each peer's individual status is enriched the
same way. Sync status is runtime information -- it resets when the process
restarts, since a fresh process has not run a round yet -- while the durable fact,
how far this device has synced each peer, is the watermark kept in the peer store.
A round that fails (a timeout, an unavailable transport) is recorded too, so the
surface shows the last attempt, not only the last success.



Sync derives its security from keys and correct implementation, in the open
(Kerckhoffs): the sync identity and the route secrets live in keys you hold,
while the protocol and the code are public. Traffic is end-to-end encrypted and
travels over private routes, so peers are reached without exposing addresses, and
participation is limited to explicit, key-addressed devices rather than arbitrary
outbound requests. This is the same boundary that retired webhooks: sync never
makes open-ended outbound calls.

Under Bulbe, the node refuses to bind. The refusal lives at the binding layer and
reads the live security mode fail-secure, so an indeterminable mode is treated as
Bulbe. The sensitive sync operations follow the same Daily-only, approval-aware
discipline as the agent's memory and skills writes. Every lifecycle transition is
recorded in the hash-chain audit log.

## Pairing and the sharing-control panel

Pairing introduces two of your own devices to each other. A device generates a
pairing payload that carries its identity, its public routing key, its public
signing key, and an integrity check over that public material; another device scans
or pastes it to pair. The integrity check is a SHA-256 over the public fields,
recomputed and compared when a payload is accepted, so a garbled or tampered
payload (a mistyped code, an altered key) is rejected before it is ever stored.
Accepting a payload registers the peer as pending: nothing syncs against a pending
peer. Both devices then show a short confirmation code derived from both payloads'
public material (identical on both screens when the exchange was not tampered
with); compare them and confirm on each device to activate the pairing. A later
re-pair whose signing key differs demotes the peer back to pending -- a changed
trust root is a new trust decision you confirm again. There is no secret in a
pairing payload: a peer is addressed by public keys you hold, and the security of a
pairing lives in those keys, the integrity check, and the human comparison, never
in the shape of the exchange (Kerckhoffs). Pairing is exchanged over the route
surface `POST /api/sync/pairing/accept`, with this device's own payload at
`GET /api/sync/pairing/self`, pending confirmations at
`GET /api/sync/pairing/pending` and their confirm/reject verbs; relabelling and
unpairing a device are `POST /api/sync/peers/{peer_id}/label` and
`DELETE /api/sync/peers/{peer_id}`.

Pairing management -- generating and accepting a payload, relabelling, unpairing --
is local-disk and works in any mode, like the peer store it populates. Accepting a
payload registers the device through the watermark-preserving upsert, so a re-pair
with a rotated route never resets how far the device has synced, and the
registration is recorded in the hash-chain audit log. Generating this device's
payload reads a live routing key from an attached node, so it reports the key is
not yet available rather than fabricating one when the node is not attached (under
Bulbe the node never attaches).

The sharing-control panel is the settings surface for all of this: it pairs your
devices (generate this device's code, scan or paste a peer's), lists, labels, and
removes paired peers, watches sync status (running, last sync, per-peer outcome),
controls what is shared across your devices, and runs a pull round per peer. Sync
stays Daily-only: the panel surfaces the Bulbe refusal honestly -- the run and
generate actions are disabled under Bulbe -- rather than offering a round that
cannot run, while pairing management stays available in any mode.

## Record signing and the 3.7.0 upgrade order

Every record a device publishes is signed with the project's post-quantum
signature suite (ML-DSA-65) over its canonical bytes -- clock and provenance
bound, so re-clocking or re-attributing a record breaks the signature -- and
every record a round receives is verified against the key registered for the
record's origin device. A record that does not verify is refused: counted,
surfaced in the panel toast, never applied, and never holding the watermark.
Since 3.7.0 this includes unsigned records from a device with no registered
signing key; the migration grace that admitted them during the rollout is
closed, as a hard constant rather than a setting.

Upgrading a fleet to 3.7.0, in order: upgrade every device; re-pair each peer
pair (one confirmation per peer -- the changed-key demotion drives the
ceremony); on each device press "Republish signed records" in the sync panel
(or `POST /api/sync/republish`) once, which re-journals that device's own
records with signatures at their existing clocks; then run rounds and watch
the refused count fall to zero. Honest interim: a device that upgrades before
its peers republish refuses their unsigned records; those records re-arrive
signed after the peer's republish, so nothing is lost and convergence is
merely delayed. The republish is local-disk and works in any mode; a device
that cannot sign answers 503 rather than republishing unsigned. A device with
no signature backend at all still accepts records as `unverified` with a
warning -- refusing what it cannot check would partition the fleet, not
protect it.

## Related

The remote-inference channel rides this same sync surface: a paired, lower-trust
device borrows this desktop's local models over the private route. See
[Remote Inference](remote-inference.md) for the bounded surface, the per-device
grant and the RAG read-only sub-grant, revocation, the rate limit, and the
desktop control surface.
