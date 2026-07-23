/**
 * API client for Veilid sync (Theme 4 / Veilid Sync).
 *
 * Defines the contract the sharing-control panel (SyncPanel.svelte) consumes to
 * pair a user's own devices and watch sync over the backend route
 * (opti_oignon/api/routes_sync.py): the read surfaces (sync status and the paired
 * peers, each with its watermark and last-round outcome), the pairing ceremony
 * (generate this device's payload, accept a peer's payload, and -- PAIR-02 --
 * compare and confirm the mutual confirmation code before the entry
 * activates), and the management
 * and run actions (relabel, unpair, run a pull round).
 *
 * Two boundaries hold and the panel surfaces them honestly. Sync is Daily-only:
 * under Bulbe the node refuses to bind, so a round cannot run -- the status carries
 * a bulbe_disabled flag and the run endpoint answers 403, which the panel shows
 * rather than offering a round that cannot run. Pairing management (generating and
 * accepting a payload, relabelling, unpairing) is local-disk and permitted in any
 * mode. A pairing payload carries only public material (an identity, a public
 * routing key) plus an integrity check over it; there is no secret in it
 * (Kerckhoffs).
 */

import { apiGet, apiPost, apiDelete } from './client';

const BASE = '/api/sync';

/** One paired peer, mirroring routes_sync._peer_to_dict enriched with last-sync. */
export interface SyncPeer {
	peer_id: string;
	routing_key: string;
	label: string;
	watermark: number;
	added_at: string;
	updated_at: string;
	/** PAIR-02: true while the pairing awaits mutual confirmation; it gates nothing. */
	pending?: boolean;
	/** PAIR-02: true when pending because a re-pair carried a different signing key. */
	key_changed?: boolean;
	/**
	 * N.9 / the recorded device class ("phone" | "desktop"), or null for
	 * the grandfathered class-less row. Phone-class peers are served a note
	 * only when its per-item mobile-allowed flag affirmatively permits it
	 * (filter-at-serve); set or cleared only through the control surface.
	 */
	device_class?: string | null;
	/** Timestamp of the last successful round; empty when none or the last failed. */
	last_sync?: string;
	/** The full outcome of the peer's last round, or null when none yet. */
	last_round?: RoundOutcome | null;
}

/** A pairing awaiting confirmation, mirroring routes_sync.pending_pairings_payload. */
export interface PendingPairing extends SyncPeer {
	/**
	 * The PAIR-02 mutual confirmation code ("1234 5678"), identical on both
	 * devices; null until this device generates its own pairing code (the
	 * missing half of the derivation).
	 */
	confirmation_code: string | null;
}

/** The accept response: the registered (pending) peer plus its confirmation code. */
export interface PairingAccepted extends SyncPeer {
	confirmation_code?: string | null;
}

/** The outcome of one round (or a failed attempt), mirroring sync_status.RoundOutcome. */
export interface RoundOutcome {
	peer_id: string;
	applied: number;
	deferred: number;
	conflicts: number;
	rejected: number;
	refused: number;
	unverified: number;
	previous_watermark: number;
	new_watermark: number;
	advanced: boolean;
	at: string;
	ok: boolean;
	error: string;
}

/** The sync-status surface, mirroring routes_sync.status_payload. */
export interface SyncStatus {
	running: boolean;
	attached: boolean;
	attachment: string;
	bulbe_disabled: boolean;
	veilid_available: boolean;
	last_round: RoundOutcome | null;
	peers: SyncPeer[];
}

/** The summary of a pull round, mirroring routes_sync.round_result_to_dict. */
export interface RoundResult {
	peer_id: string;
	applied: number;
	deferred: number;
	conflicts: number;
	rejected: number;
	refused: number;
	unverified: number;
	previous_watermark: number;
	new_watermark: number;
	advanced: boolean;
}

/**
 * One pending content approval (SYN-05), mirroring
 * routes_sync.deferred_entry_to_dict: a sensitive record the round quarantined
 * instead of applying. Provenance only -- the record body never reaches the
 * panel; on approval it enters the local set through the engine's
 * verify -> gate -> apply seam against the CURRENT trust state.
 */
export interface DeferredRecord {
	kind: string;
	record_id: string;
	origin_device: string;
	peer_id: string;
	clock: number;
	deferred_at: string;
	last_offered_at: string;
}

/** The outcome of approving a deferred record, mirroring engine.approve_deferred. */
export interface DeferredApproveResult {
	kind: string;
	record_id: string;
	approved: boolean;
	/** True when re-verification against the current trust state refused. */
	refused: boolean;
	reason: string;
	applied: number;
	conflicts: number;
	rejected: number;
	unverified: number;
}

/** This device's pairing payload, mirroring routes_sync.self_pairing_payload. */
export interface PairingSelf {
	peer_id: string;
	routing_key: string;
	/** The structured payload (identity, routing key, integrity check). */
	payload: Record<string, unknown>;
	/** The compact JSON form -- the text a QR encodes or a peer transcribes. */
	text: string;
}

/** The sync-status surface: whether sync is running, the last round, per-peer last-sync. */
export async function getSyncStatus(): Promise<SyncStatus> {
	return apiGet<SyncStatus>(`${BASE}/status`);
}

/** List the paired peers. */
export async function listSyncPeers(): Promise<SyncPeer[]> {
	const res = await apiGet<{ peers: SyncPeer[] }>(`${BASE}/peers`);
	return res?.peers ?? [];
}

/** One paired peer's status, enriched with its last-sync and last round. */
export async function getSyncPeer(peerId: string): Promise<SyncPeer> {
	return apiGet<SyncPeer>(`${BASE}/peers/${encodeURIComponent(peerId)}`);
}

/**
 * Generate this device's pairing payload: its identity, its public routing key,
 * and an integrity check over them. The node must be attached (Daily mode) to
 * supply a live routing key; otherwise the backend answers 503.
 */
export async function getPairingSelf(): Promise<PairingSelf> {
	return apiGet<PairingSelf>(`${BASE}/pairing/self`);
}

/**
 * Accept a peer's pairing payload and register it. The payload is validated
 * defensively: a tampered or malformed one is rejected (400). Permitted in any
 * mode -- registering a peer is local-disk, not a wire action. `text` is the
 * scanned or transcribed JSON payload; `label` is the local name to assign.
 * PAIR-02: the peer registers pending and the response carries the
 * confirmation code to compare on both devices (null until this device has
 * generated its own pairing code).
 */
export async function acceptPairing(text: string, label = ''): Promise<PairingAccepted> {
	let payload: unknown = text;
	try {
		payload = JSON.parse(text);
	} catch {
		payload = text;
	}
	return apiPost<PairingAccepted>(`${BASE}/pairing/accept`, { payload, label });
}

/**
 * The pairings awaiting mutual confirmation (PAIR-02), each with its code.
 * `self_ready` is false until this device has generated its own pairing code,
 * the missing half of every confirmation code. Local-disk, any mode.
 */
export async function listPendingPairings(): Promise<{
	pending: PendingPairing[];
	self_ready: boolean;
}> {
	return apiGet<{ pending: PendingPairing[]; self_ready: boolean }>(
		`${BASE}/pairing/pending`
	);
}

/** Confirm a pending pairing (PAIR-02): the human compared the codes. */
export async function confirmPairing(peerId: string): Promise<SyncPeer> {
	return apiPost<SyncPeer>(
		`${BASE}/pairing/pending/${encodeURIComponent(peerId)}/confirm`
	);
}

/** Reject a pending pairing (PAIR-02): the entry is removed, nothing trusted. */
export async function rejectPairing(
	peerId: string
): Promise<{ peer_id: string; rejected: boolean }> {
	return apiPost<{ peer_id: string; rejected: boolean }>(
		`${BASE}/pairing/pending/${encodeURIComponent(peerId)}/reject`
	);
}

/** Relabel a paired peer; the watermark and pairing time are preserved. */
export async function relabelPeer(peerId: string, label: string): Promise<SyncPeer> {
	return apiPost<SyncPeer>(`${BASE}/peers/${encodeURIComponent(peerId)}/label`, { label });
}

/**
 * Set or clear a paired peer's device class; the human-confirmed path.
 * Only 'phone' | 'desktop' | null leave this client; the route refuses free
 * text with a 400 before the store is touched, 404 for an unpaired peer, and
 * the write goes through the engine's audited setter. The returned peer is
 * the registry's fresh truth (confirmed posture).
 */
export async function setDeviceClass(
	peerId: string,
	deviceClass: 'phone' | 'desktop' | null
): Promise<SyncPeer> {
	return apiPost<SyncPeer>(`${BASE}/peers/${encodeURIComponent(peerId)}/device-class`, {
		device_class: deviceClass
	});
}

/** Unpair a peer. */
export async function unpairPeer(peerId: string): Promise<{ peer_id: string; removed: boolean }> {
	return apiDelete<{ peer_id: string; removed: boolean }>(
		`${BASE}/peers/${encodeURIComponent(peerId)}`
	);
}

/**
 * Run one pull round against a paired peer over the live transport. Refused under
 * Bulbe (403); 404 for an unpaired peer; 503 when the transport is unavailable;
 * 504 when the peer stalls.
 */
export async function runSync(peerId: string): Promise<RoundResult> {
	return apiPost<RoundResult>(`${BASE}/peers/${encodeURIComponent(peerId)}/run`);
}

/**
 * The one-time VL-01 fleet-order step: re-sign and re-journal this
 * device's own unsigned records at the same clocks, so peers adopt the signed
 * bytes on their next pull. Local-disk signing, available in any mode; the
 * server answers 503 honestly when this device cannot sign.
 */
export async function republishSigned(): Promise<{ republished: number }> {
	return apiPost<{ republished: number }>(`${BASE}/republish`);
}

/**
 * The pending content approvals (SYN-05): sensitive records quarantined by a
 * round, awaiting approve/refuse. Local-disk read, available in any mode.
 */
export async function listDeferredRecords(): Promise<{
	deferred: DeferredRecord[];
	count: number;
}> {
	return apiGet<{ deferred: DeferredRecord[]; count: number }>(`${BASE}/deferred`);
}

/**
 * Approve a pending record: it re-enters the engine's verify -> gate -> apply
 * seam against the current trust state. The result's refused flag is honest --
 * a changed key or demoted origin since deferral refuses instead of applying,
 * and the entry is removed either way. Available in any mode.
 */
export async function approveDeferredRecord(
	kind: string,
	recordId: string
): Promise<DeferredApproveResult> {
	return apiPost<DeferredApproveResult>(`${BASE}/deferred/approve`, {
		kind,
		record_id: recordId
	});
}

/** Refuse a pending record: its entry is removed and nothing applies. Any mode. */
export async function refuseDeferredRecord(
	kind: string,
	recordId: string
): Promise<DeferredRecord & { removed: boolean }> {
	return apiPost<DeferredRecord & { removed: boolean }>(`${BASE}/deferred/refuse`, {
		kind,
		record_id: recordId
	});
}

/**
 * A device's remote-inference grant state (cas 7 Lot 2), mirroring
 * routes_sync.remote_chat_grant_payload. Remote chat is the tier-1 bounded
 * surface (inference, optionally RAG read-only via the sub-grant); enabled by
 * default for a confirmed peer, instantly revocable.
 */
export interface RemoteChatGrant {
	peer_id: string;
	remote_chat_enabled: boolean;
	rag_subgrant: boolean;
	pending?: boolean;
}

/** The remote channel rate/telemetry state (cas 7 Lot 2). */
export interface RemoteChatTelemetry {
	devices: Record<
		string,
		{ requests_in_window: number; window_started_at: number; alerts: number }
	>;
	active_sessions: number;
}

/** A device's remote-inference grant (enabled, RAG sub-grant). */
export async function getRemoteChatGrant(peerId: string): Promise<RemoteChatGrant> {
	return apiGet<RemoteChatGrant>(
		`${BASE}/peers/${encodeURIComponent(peerId)}/remote-chat`
	);
}

/** Enable/disable remote chat and/or set the RAG read-only sub-grant for a device. */
export async function setRemoteChatGrant(
	peerId: string,
	opts: { enabled?: boolean; rag?: boolean }
): Promise<RemoteChatGrant> {
	return apiPost<RemoteChatGrant>(
		`${BASE}/peers/${encodeURIComponent(peerId)}/remote-chat`,
		opts
	);
}

/** Revoke a device's remote-chat grant and kill its live streaming sessions. */
export async function revokeRemoteChat(peerId: string): Promise<{
	peer_id: string;
	revoked: boolean;
	killed_sessions: number;
	remote_chat_enabled: boolean;
}> {
	return apiPost<{
		peer_id: string;
		revoked: boolean;
		killed_sessions: number;
		remote_chat_enabled: boolean;
	}>(`${BASE}/peers/${encodeURIComponent(peerId)}/remote-chat/revoke`);
}

/** The remote channel rate/telemetry state for the panel (per-device, live sessions). */
export async function getRemoteChatTelemetry(): Promise<RemoteChatTelemetry> {
	return apiGet<RemoteChatTelemetry>(`${BASE}/remote-chat/telemetry`);
}

/** A short, display-safe form of a routing key (it is public, but long). */
export function shortRoutingKey(key: string): string {
	if (!key) return '';
	if (key.length <= 16) return key;
	return `${key.slice(0, 8)}...${key.slice(-6)}`;
}
