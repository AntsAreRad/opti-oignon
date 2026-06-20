<!--
  SyncPanel.svelte (S182, Theme 4 / Veilid Sync)
  The sharing-control panel for optional peer-to-peer sync over Veilid, built on
  the S166 lib/ds primitives (Card, Button, Icon, EmptyState, InlineError). It
  pairs a user's own devices (generate this device's pairing code, scan or paste a
  peer's), lists and labels and removes paired peers, watches sync status (running,
  last sync, per-peer outcome over /api/sync/status and /api/sync/peers), and runs a
  pull round. A pairing payload carries only public material (an identity, a public
  routing key, since S205 the signing public key) plus an integrity check over it
  -- no secret (Kerckhoffs).

  PAIR-02 (S206): pairing is completed by a mutual confirmation. Accepting a
  payload registers the peer PENDING -- it gates nothing (no round, no serving,
  no trusted key) -- and an "Awaiting confirmation" section shows the short
  comparison code derived from both devices' public material, identical on both
  screens. The humans compare, then confirm on both devices (or reject; a
  re-pair that changed the signing key is surfaced distinctly). The code needs
  this device's own generated payload as its other half, so the panel hints to
  show the pairing code first when it is missing.

  SYN-05 (S207): a second, deliberately distinct waiting list. "Awaiting
  confirmation" is DEVICE trust (a pairing); "Pending record approvals" is
  CONTENT approval -- sensitive records (skills) a round quarantined to the
  deferred ledger instead of applying. Each entry shows provenance only (kind,
  id, origin device, serving peer, when deferred), never the record body.
  Approve re-enters the engine's verify -> gate -> apply seam against the
  current trust state (a changed key refuses honestly); Refuse removes the
  entry and applies nothing. Both are local-disk decisions, available in any
  mode; rounds no longer re-prompt for a record sitting here.

  Sync is Daily-only: under Bulbe the node refuses to bind, so a round cannot run.
  The panel surfaces that refusal honestly (a banner, disabled run and generate
  actions) rather than offering a round that cannot run. Pairing management
  (accepting a payload, confirming or rejecting a pending pairing, relabelling,
  unpairing) is local-disk and stays available in
  any mode. Updates announce through an aria-live region. Design-system tokens only
  (--oo-*); lucide icons through Icon. Registered in FRONTEND_REDESIGN_SPEC.md.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { Button, Card, Icon, EmptyState, InlineError } from '$lib/ds';
	import {
		getSyncStatus,
		listSyncPeers,
		listPendingPairings,
		getPairingSelf,
		acceptPairing,
		confirmPairing,
		rejectPairing,
		relabelPeer,
		setDeviceClass,
		unpairPeer,
		runSync,
		republishSigned,
		listDeferredRecords,
		approveDeferredRecord,
		refuseDeferredRecord,
		shortRoutingKey,
		type SyncStatus,
		type SyncPeer,
		type PendingPairing,
		type PairingSelf,
		type DeferredRecord
	} from '$lib/api/sync';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import RemoteChannelPanel from './RemoteChannelPanel.svelte';

	let status: SyncStatus | null = null;
	let peers: SyncPeer[] = [];
	let loading = false;
	let error: string | null = null;
	let busyPeer: string | null = null;

	// Pairing-out: this device's payload (the code a peer scans or transcribes).
	let selfPairing: PairingSelf | null = null;
	let selfError: string | null = null;
	let selfBusy = false;

	// Pairing-in: a scanned or pasted payload to accept.
	let pasteText = '';
	let pasteLabel = '';
	let accepting = false;

	// PAIR-02: pairings awaiting the mutual confirmation. Each entry carries
	// the short code both humans compare; selfReady is false until this device
	// has generated its own pairing code (the missing half of the derivation).
	let pendingPairings: PendingPairing[] = [];
	let selfReady = false;
	let busyPending: string | null = null;

	// SYN-05 (S207): pending content approvals -- sensitive records a round
	// quarantined to the deferred ledger instead of applying. Distinct from
	// the device-trust list above; provenance only, never the record body.
	let deferredRecords: DeferredRecord[] = [];
	// A failed deferred-list load must be distinguishable from an empty queue
	// (a silent catch would hide pending approvals); surfaced inline below.
	let deferredError: string | null = null;
	let republishing = false;
	let busyDeferred: string | null = null;

	$: bulbeDisabled = status?.bulbe_disabled ?? false;
	$: veilidAvailable = status?.veilid_available ?? false;
	$: running = status?.running ?? false;
	// A pending entry gates nothing; the active list shows confirmed peers only.
	$: confirmedPeers = peers.filter((p) => !p.pending);

	async function load() {
		loading = true;
		error = null;
		try {
			const [s, p, pp, dl] = await Promise.all([
				getSyncStatus(),
				listSyncPeers(),
				listPendingPairings().catch(() => ({ pending: [], self_ready: false })),
				listDeferredRecords().then(
					(d) => ({ ok: true as const, ...d }),
					(e: unknown) => ({
						ok: false as const,
						deferred: [] as DeferredRecord[],
						count: 0,
						reason: e instanceof Error ? e.message : 'Failed to load pending approvals'
					})
				)
			]);
			status = s;
			peers = s.peers && s.peers.length > 0 ? s.peers : p;
			pendingPairings = pp.pending ?? [];
			selfReady = pp.self_ready ?? false;
			deferredRecords = dl.deferred ?? [];
			deferredError = dl.ok ? null : dl.reason;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load sync status';
		} finally {
			loading = false;
		}
	}

	function deferredKey(d: DeferredRecord): string {
		return `${d.kind}/${d.record_id}`;
	}

	async function approveDeferred(d: DeferredRecord) {
		busyDeferred = deferredKey(d);
		try {
			const result = await approveDeferredRecord(d.kind, d.record_id);
			if (result.refused) {
				// Honest fail-secure outcome: the trust state changed since the
				// record was deferred (a new key, a demoted origin); nothing
				// applied and the entry is gone.
				toastError(
					`Not applied: ${d.kind} ${d.record_id} no longer verifies against ` +
						`the current trust state; it was removed`
				);
			} else {
				toastSuccess(
					`Approved ${d.kind} ${d.record_id}: ${result.applied} applied` +
						(result.conflicts > 0 ? `, ${result.conflicts} conflict(s) retained` : '')
				);
			}
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to approve record');
		} finally {
			busyDeferred = null;
		}
	}

	async function refuseDeferred(d: DeferredRecord) {
		busyDeferred = deferredKey(d);
		try {
			await refuseDeferredRecord(d.kind, d.record_id);
			toastSuccess(`Refused ${d.kind} ${d.record_id}; nothing was applied`);
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to refuse record');
		} finally {
			busyDeferred = null;
		}
	}

	async function showPairingCode() {
		selfBusy = true;
		selfError = null;
		try {
			selfPairing = await getPairingSelf();
			// Generating the payload pins this device's half of the
			// confirmation-code material; refresh so pending codes appear.
			await load();
		} catch (e) {
			selfError =
				e instanceof Error ? e.message : 'Routing key not available (node not attached)';
		} finally {
			selfBusy = false;
		}
	}

	async function copyCode() {
		if (!selfPairing) return;
		try {
			await navigator.clipboard.writeText(selfPairing.text);
			toastSuccess('Pairing code copied');
		} catch {
			toastError('Could not copy to clipboard');
		}
	}

	// The one-time VL-01 fleet-order step (S208): re-sign this device's own
	// records so peers adopt the signed bytes. Local-disk, any mode; a count
	// of 0 is honest idempotence (nothing unsigned remained).
	async function republishNow() {
		republishing = true;
		try {
			const result = await republishSigned();
			toastSuccess(
				result.republished > 0
					? `Republished ${result.republished} record${result.republished > 1 ? 's' : ''} with signatures`
					: 'Nothing to republish; all local records are already signed'
			);
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Republish failed; signing unavailable');
		} finally {
			republishing = false;
		}
	}

	async function pairDevice() {
		if (!pasteText.trim()) return;
		accepting = true;
		try {
			const peer = await acceptPairing(pasteText.trim(), pasteLabel.trim());
			toastSuccess(
				`Pairing with ${peer.label || peer.peer_id} awaits confirmation: ` +
					`compare the code shown on both devices, then confirm`
			);
			pasteText = '';
			pasteLabel = '';
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Invalid pairing code');
		} finally {
			accepting = false;
		}
	}

	async function confirmPending(p: PendingPairing) {
		busyPending = p.peer_id;
		try {
			await confirmPairing(p.peer_id);
			toastSuccess(`Confirmed ${p.label || p.peer_id}`);
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to confirm pairing');
		} finally {
			busyPending = null;
		}
	}

	async function rejectPending(p: PendingPairing) {
		busyPending = p.peer_id;
		try {
			await rejectPairing(p.peer_id);
			toastSuccess(`Rejected ${p.label || p.peer_id}; nothing was trusted`);
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to reject pairing');
		} finally {
			busyPending = null;
		}
	}

	async function syncNow(peer: SyncPeer) {
		busyPeer = peer.peer_id;
		try {
			const result = await runSync(peer.peer_id);
			toastSuccess(
				`Synced ${peer.label || peer.peer_id}: ${result.applied} applied` +
					(result.deferred > 0 ? `, ${result.deferred} awaiting approval below` : '') +
					(result.refused > 0 ? `, ${result.refused} refused (signature)` : '') +
					(result.unverified > 0 ? `, ${result.unverified} unverified (no signature check)` : '')
			);
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Sync round failed');
		} finally {
			busyPeer = null;
		}
	}

	async function rename(peer: SyncPeer) {
		const next = window.prompt('New label for this device', peer.label);
		if (next === null) return;
		busyPeer = peer.peer_id;
		try {
			await relabelPeer(peer.peer_id, next.trim());
			toastSuccess('Label updated');
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to relabel');
		} finally {
			busyPeer = null;
		}
	}

	// N.9 (S260): the human-confirmed class setter the S258 accept seam
	// deferred to. CONFIRMED posture: the row is reloaded from the registry
	// after the audited write; nothing is shown that the backend has not
	// recorded. Only the two allowlisted values ever leave this panel.
	async function markClass(peer: SyncPeer, deviceClass: 'phone' | 'desktop') {
		busyPeer = peer.peer_id;
		try {
			await setDeviceClass(peer.peer_id, deviceClass);
			toastSuccess(`${peer.label || peer.peer_id} is now treated as ${deviceClass}`);
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to set device class');
		} finally {
			busyPeer = null;
		}
	}

	async function remove(peer: SyncPeer) {
		busyPeer = peer.peer_id;
		try {
			await unpairPeer(peer.peer_id);
			toastSuccess(`Unpaired ${peer.label || peer.peer_id}`);
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to unpair');
		} finally {
			busyPeer = null;
		}
	}

	onMount(load);
</script>

<section class="sync-panel">
	<header class="sync-header">
		<div class="sync-title">
			<Icon name="share-2" />
			<h2>Sync (Veilid)</h2>
		</div>
		<Button variant="ghost" on:click={load} disabled={loading}>
			<Icon name="refresh-cw" />
			Refresh
		</Button>
	</header>

	{#if bulbeDisabled}
		<p class="sync-bulbe-note" role="note">
			<Icon name="wifi-off" />
			Sync is disabled in Bulbe mode: peer-to-peer networking is refused at the binding
			layer. You can still manage paired devices here; switch to Daily mode to run a round.
		</p>
	{:else if !veilidAvailable}
		<p class="sync-bulbe-note" role="note">
			<Icon name="wifi-off" />
			The Veilid framework is not installed. Install opti-oignon[veilid] to enable sync.
		</p>
	{:else}
		<p class="sync-state" role="status">
			<Icon name="radio" />
			{running ? 'Sync node is running' : 'Sync node is not running'}
			{#if status?.attachment}<span class="sync-sub"> ({status.attachment})</span>{/if}
			{#if deferredRecords.length > 0}
				<span class="sync-sub">
					· {deferredRecords.length} record{deferredRecords.length > 1 ? 's' : ''} awaiting approval
				</span>
			{/if}
		</p>
	{/if}

	{#if error}
		<InlineError message={error} onRetry={load} />
	{/if}

	<Card>
		<div class="sync-section">
			<div class="sync-section-title">
				<Icon name="qr-code" />
				<h3>This device</h3>
			</div>
			<p class="sync-hint">
				Show this device's pairing code on another of your devices to link them. The code
				carries public routing and signing keys plus an integrity check, never a secret.
			</p>
			<Button variant="secondary" on:click={showPairingCode} disabled={selfBusy}>
				<Icon name="qr-code" />
				{selfPairing ? 'Refresh pairing code' : 'Show pairing code'}
			</Button>
			{#if selfError}
				<p class="sync-inline-error" role="alert">{selfError}</p>
			{/if}
			{#if selfPairing}
				<div class="sync-qr" aria-label="Pairing code">
					<code class="sync-code">{selfPairing.text}</code>
					<Button variant="ghost" on:click={copyCode}>
						<Icon name="copy" />
						Copy
					</Button>
				</div>
			{/if}
			<p class="sync-hint">
				After upgrading and re-pairing your devices, republish once on each device so
				every record carries a signature. Running it again is harmless.
			</p>
			<Button variant="secondary" on:click={republishNow} disabled={republishing}>
				<Icon name="pen-line" />
				{republishing ? 'Republishing...' : 'Republish signed records'}
			</Button>
		</div>
	</Card>

	<Card>
		<div class="sync-section">
			<div class="sync-section-title">
				<Icon name="plus" />
				<h3>Add a device</h3>
			</div>
			<p class="sync-hint">
				Scan or paste the pairing code from another of your devices. Pairing is local; it
				works in any mode.
			</p>
			<label class="sync-field">
				<span>Pairing code</span>
				<textarea
					class="sync-textarea"
					rows="3"
					bind:value={pasteText}
					placeholder="Paste the scanned pairing code here"
				></textarea>
			</label>
			<label class="sync-field">
				<span>Label (optional)</span>
				<input class="sync-input" type="text" bind:value={pasteLabel} placeholder="e.g. Laptop" />
			</label>
			<Button variant="primary" on:click={pairDevice} disabled={accepting || !pasteText.trim()}>
				<Icon name="plus" />
				Pair device
			</Button>
		</div>
	</Card>

	{#if pendingPairings.length > 0}
		<div class="sync-pending" role="status" aria-live="polite">
			<h3 class="sync-peers-title">Awaiting confirmation</h3>
			<p class="sync-hint">
				Compare the code below with the one shown on the other device. Confirm on both
				devices only when the codes match; reject if they differ or if you did not start
				this pairing.
			</p>
			{#each pendingPairings as p (p.peer_id)}
				<Card>
					<div class="pending-row">
						<div class="pending-main">
							<Icon name="smartphone" />
							<div class="pending-meta">
								<span class="peer-name">{p.label || p.peer_id}</span>
								<span class="peer-sub">
									<Icon name="link-2" size="sm" />
									{shortRoutingKey(p.routing_key)}
								</span>
								{#if p.key_changed}
									<span class="pending-key-changed">
										<Icon name="shield-alert" size="sm" />
										Signing key changed: re-confirm this device. If you did not re-pair
										it yourself, reject it.
									</span>
								{/if}
								<span class="peer-sub">
									<Icon name="smartphone" size="sm" />
									Recorded class: {p.device_class ?? 'unspecified'}
									{#if p.device_class === 'phone'}
										<span class="peer-class-note">
											(will be served mobile-allowed notes only)
										</span>
									{/if}
								</span>
								{#if p.confirmation_code}
									<code class="confirm-code" aria-label="Confirmation code"
										>{p.confirmation_code}</code
									>
								{:else}
									<span class="pending-code-hint">
										{selfReady
											? 'Confirmation code unavailable; show this device pairing code again to refresh it'
											: 'Show this device pairing code first to display the confirmation code'}
									</span>
								{/if}
							</div>
						</div>
						<div class="pending-actions">
							<Button
								variant="primary"
								on:click={() => confirmPending(p)}
								disabled={busyPending === p.peer_id || !p.confirmation_code}
								title={p.confirmation_code
									? 'Confirm: the codes match on both devices'
									: 'The confirmation code must be visible before confirming'}
							>
								<Icon name="check" />
								Confirm
							</Button>
							<Button
								variant="danger"
								on:click={() => rejectPending(p)}
								disabled={busyPending === p.peer_id}
							>
								<Icon name="x" />
								Reject
							</Button>
						</div>
					</div>
				</Card>
			{/each}
		</div>
	{/if}

	{#if deferredError}
		<p class="sync-inline-error" role="alert">
			<Icon name="alert-triangle" size="sm" />
			Pending approvals could not be loaded: {deferredError}. Records held for
			approval are not shown until this succeeds.
		</p>
	{/if}

	{#if deferredRecords.length > 0}
		<div class="sync-deferred" role="status" aria-live="polite">
			<h3 class="sync-peers-title">Pending record approvals</h3>
			<p class="sync-hint">
				Content held for your approval: sensitive records received from your paired
				devices that were not applied. Approving re-verifies against the current trust
				state before applying; refusing removes the record and applies nothing.
			</p>
			{#each deferredRecords as d (deferredKey(d))}
				<Card>
					<div class="pending-row">
						<div class="pending-main">
							<Icon name="file-lock-2" />
							<div class="pending-meta">
								<span class="peer-name">{d.kind}: {d.record_id}</span>
								<span class="peer-sub">
									<Icon name="smartphone" size="sm" />
									from {d.origin_device}
									{#if d.peer_id !== d.origin_device}
										(via {d.peer_id})
									{/if}
									· version {d.clock}
								</span>
								<span class="peer-sub">
									<Icon name="clock" size="sm" />
									deferred {d.deferred_at}
								</span>
							</div>
						</div>
						<div class="pending-actions">
							<Button
								variant="primary"
								on:click={() => approveDeferred(d)}
								disabled={busyDeferred === deferredKey(d)}
								title="Apply this record after re-verifying it against the current trust state"
							>
								<Icon name="check" />
								Approve
							</Button>
							<Button
								variant="danger"
								on:click={() => refuseDeferred(d)}
								disabled={busyDeferred === deferredKey(d)}
								title="Remove this record without applying it"
							>
								<Icon name="x" />
								Refuse
							</Button>
						</div>
					</div>
				</Card>
			{/each}
		</div>
	{/if}

	<div class="sync-peers" role="status" aria-live="polite">
		<h3 class="sync-peers-title">Paired devices</h3>
		{#if loading && confirmedPeers.length === 0}
			<p class="sync-loading">Loading peers...</p>
		{:else if confirmedPeers.length === 0}
			<EmptyState
				icon="smartphone"
				title="No devices paired"
				description="Pair another of your devices to sync conversations, memory, and skills."
			/>
		{:else}
			{#each confirmedPeers as peer (peer.peer_id)}
				<Card>
					<div class="peer-row">
						<div class="peer-main">
							<Icon name="smartphone" />
							<div class="peer-meta">
								<span class="peer-name">{peer.label || peer.peer_id}</span>
								<span class="peer-sub">
									<Icon name="link-2" size="sm" />
									{shortRoutingKey(peer.routing_key)} · watermark {peer.watermark}
								</span>
								<span class="peer-sub">
									<Icon name="smartphone" size="sm" />
									Class: {peer.device_class ?? 'desktop (grandfathered)'}
									{#if peer.device_class === 'phone'}
										<span class="peer-class-note">served mobile-allowed notes only</span>
									{/if}
								</span>
								<span class="peer-sub">
									<Icon name="clock" size="sm" />
									{#if peer.last_sync}
										Last sync {peer.last_sync}
									{:else if peer.last_round && !peer.last_round.ok}
										Last attempt failed: {peer.last_round.error}
									{:else}
										Never synced
									{/if}
								</span>
							</div>
						</div>
						<div class="peer-actions">
							<Button
								variant="primary"
								on:click={() => syncNow(peer)}
								disabled={busyPeer === peer.peer_id || bulbeDisabled}
								title={bulbeDisabled ? 'Sync is disabled in Bulbe mode' : 'Run a pull round'}
							>
								<Icon name="radio" />
								Sync now
							</Button>
							<Button
								variant="ghost"
								on:click={() => rename(peer)}
								disabled={busyPeer === peer.peer_id}
							>
								<Icon name="pencil" />
								Rename
							</Button>
							{#if peer.device_class === 'phone'}
								<Button
									variant="ghost"
									on:click={() => markClass(peer, 'desktop')}
									disabled={busyPeer === peer.peer_id}
									title="Escalate to desktop class: every note is served to this device. Human-only; never happens at the pairing ceremony."
								>
									<Icon name="monitor" />
									Treat as desktop
								</Button>
							{:else}
								<Button
									variant="ghost"
									on:click={() => markClass(peer, 'phone')}
									disabled={busyPeer === peer.peer_id}
									title="Restrict to phone class: only mobile-allowed notes are served to this device."
								>
									<Icon name="smartphone" />
									Treat as phone
								</Button>
							{/if}
							<Button
								variant="danger"
								on:click={() => remove(peer)}
								disabled={busyPeer === peer.peer_id}
							>
								<Icon name="trash-2" />
								Remove
							</Button>
						</div>
					</div>
				</Card>
			{/each}
		{/if}
	</div>

	<RemoteChannelPanel {peers} />
</section>

<style>
	.sync-panel {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-4);
		padding: var(--oo-space-4);
	}
	.sync-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}
	.sync-title {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
	}
	.sync-title h2 {
		margin: 0;
		font-size: var(--oo-text-lg);
		color: var(--oo-fg-primary);
	}
	.sync-bulbe-note {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		margin: 0;
		padding: var(--oo-space-2) var(--oo-space-3);
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-warning);
		background: var(--oo-warning-bg);
		border: 1px solid var(--oo-warning-bd);
		border-radius: var(--oo-radius-md);
	}
	.sync-state {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		margin: 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
	}
	.sync-sub {
		color: var(--oo-fg-muted);
	}
	.sync-section {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}
	.sync-section-title {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
	}
	.sync-section-title h3 {
		margin: 0;
		font-size: var(--oo-text-base);
		color: var(--oo-fg-primary);
	}
	.sync-hint {
		margin: 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}
	.sync-inline-error {
		margin: 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-error);
	}
	.sync-qr {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		padding: var(--oo-space-3);
		background: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
	}
	.sync-code {
		flex: 1;
		overflow-wrap: anywhere;
		font-family: var(--oo-font-mono);
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-secondary);
	}
	.sync-field {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
	}
	.sync-textarea,
	.sync-input {
		width: 100%;
		padding: var(--oo-space-2) var(--oo-space-3);
		font-family: var(--oo-font-sans);
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-primary);
		background: var(--oo-bg-base);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
	}
	.sync-textarea {
		font-family: var(--oo-font-mono);
		resize: vertical;
	}
	.sync-peers {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}
	.sync-peers-title {
		margin: 0;
		font-size: var(--oo-text-base);
		color: var(--oo-fg-primary);
	}
	.sync-loading {
		margin: 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}
	.peer-row {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: var(--oo-space-3);
		flex-wrap: wrap;
	}
	.peer-main {
		display: flex;
		align-items: flex-start;
		gap: var(--oo-space-2);
	}
	.peer-meta {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
	}
	.peer-name {
		font-size: var(--oo-text-base);
		color: var(--oo-fg-primary);
	}
	.peer-sub {
		display: flex;
		align-items: center;
		gap: var(--oo-space-1);
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
	}
	.peer-class-note {
		color: var(--oo-fg-muted);
		font-style: italic;
	}
	.peer-actions {
		display: flex;
		gap: var(--oo-space-2);
		flex-wrap: wrap;
	}
	.sync-pending,
	.sync-deferred {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}
	.pending-row {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: var(--oo-space-3);
		flex-wrap: wrap;
	}
	.pending-main {
		display: flex;
		align-items: flex-start;
		gap: var(--oo-space-2);
		min-width: 0;
	}
	.pending-meta {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		min-width: 0;
	}
	.pending-actions {
		display: flex;
		gap: var(--oo-space-2);
		flex-wrap: wrap;
	}
	.confirm-code {
		font-family: var(--oo-font-mono);
		font-size: var(--oo-text-xl);
		font-weight: 600;
		letter-spacing: 0.12em;
		color: var(--oo-fg-primary);
		background: var(--oo-bg-base);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
		padding: var(--oo-space-2) var(--oo-space-3);
		align-self: flex-start;
		user-select: all;
	}
	.pending-key-changed {
		display: flex;
		align-items: center;
		gap: var(--oo-space-1);
		font-size: var(--oo-text-sm);
		font-weight: 600;
		color: var(--oo-error);
	}
	.pending-code-hint {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}
</style>
