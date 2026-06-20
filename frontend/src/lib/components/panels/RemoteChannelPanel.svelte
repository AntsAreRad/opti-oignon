<!--
  RemoteChannelPanel.svelte (cas 7 Lot 2, S235, REMOTE_INFERENCE_SPEC section 11)
  The desktop control surface for the remote-inference channel, a child of the
  SyncPanel family (Settings > Network & Privacy), built on the S166 lib/ds
  primitives (Card, Button, Icon, EmptyState, InlineError).

  It lets a paired device borrow this desktop's models over the private Veilid
  route. The remote surface is the tier-1 bounded surface -- inference, and
  optionally read-only RAG via a separate per-device sub-grant -- Daily-only and
  instantly revocable per device; nothing else (state-mutation, sandbox,
  filesystem, shell, config) is ever reachable. Per device the panel shows the
  remote-chat grant (enabled by default for a confirmed peer), the RAG read-only
  sub-grant (off by default), and a revoke action that disables the grant AND
  drops the device's in-flight streaming sessions in one gesture. The channel's
  live-session count is surfaced from the rate/telemetry endpoint.

  All controls go through /api/sync (routes_sync), which carries router-level
  authentication (SYN-06). The grant and the revocation are audit-chained on the
  server. Design-system tokens only (--oo-*); lucide icons through Icon. The peer
  list is supplied by the parent SyncPanel. Registered in
  FRONTEND_REDESIGN_SPEC.md.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { Button, Card, Icon, EmptyState, InlineError } from '$lib/ds';
	import {
		getRemoteChatGrant,
		setRemoteChatGrant,
		revokeRemoteChat,
		getRemoteChatTelemetry,
		shortRoutingKey,
		type SyncPeer,
		type RemoteChatGrant,
		type RemoteChatTelemetry
	} from '$lib/api/sync';

	/** The paired peers, supplied by the parent SyncPanel. */
	export let peers: SyncPeer[] = [];

	let grants: Record<string, RemoteChatGrant> = {};
	let telemetry: RemoteChatTelemetry | null = null;
	let busy: Record<string, boolean> = {};
	let error = '';

	// Only confirmed peers can hold a grant; a pending pairing grants nothing.
	$: confirmedPeers = (peers || []).filter((p) => !p.pending);

	async function loadTelemetry(): Promise<void> {
		try {
			telemetry = await getRemoteChatTelemetry();
		} catch (e) {
			// telemetry is best-effort; never block the panel on it
			telemetry = null;
		}
	}

	async function loadGrant(peerId: string): Promise<void> {
		try {
			grants[peerId] = await getRemoteChatGrant(peerId);
			grants = { ...grants };
		} catch (e) {
			error = `Could not load the remote-chat grant for ${peerId}.`;
		}
	}

	async function setEnabled(peerId: string, enabled: boolean): Promise<void> {
		busy[peerId] = true;
		busy = { ...busy };
		error = '';
		try {
			grants[peerId] = await setRemoteChatGrant(peerId, { enabled });
			grants = { ...grants };
			await loadTelemetry();
		} catch (e) {
			error = `Could not update remote chat for ${peerId}.`;
		} finally {
			busy[peerId] = false;
			busy = { ...busy };
		}
	}

	async function setRag(peerId: string, rag: boolean): Promise<void> {
		busy[peerId] = true;
		busy = { ...busy };
		error = '';
		try {
			grants[peerId] = await setRemoteChatGrant(peerId, { rag });
			grants = { ...grants };
		} catch (e) {
			error = `Could not update the RAG sub-grant for ${peerId}.`;
		} finally {
			busy[peerId] = false;
			busy = { ...busy };
		}
	}

	async function revoke(peerId: string): Promise<void> {
		busy[peerId] = true;
		busy = { ...busy };
		error = '';
		try {
			await revokeRemoteChat(peerId);
			await loadGrant(peerId);
			await loadTelemetry();
		} catch (e) {
			error = `Could not revoke remote chat for ${peerId}.`;
		} finally {
			busy[peerId] = false;
			busy = { ...busy };
		}
	}

	onMount(() => {
		loadTelemetry();
		for (const peer of confirmedPeers) {
			loadGrant(peer.peer_id);
		}
	});
</script>

<Card>
	<h3 class="rc-title"><Icon name="radio" /> Remote chat</h3>
	<p class="rc-intro">
		Let a paired device borrow this desktop's models over the private route. The
		remote surface is inference only -- optionally read-only RAG via a separate
		sub-grant -- available in Daily mode and instantly revocable per device.
	</p>

	{#if error}
		<InlineError message={error} />
	{/if}

	{#if confirmedPeers.length === 0}
		<EmptyState
			title="No paired devices"
			description="Pair and confirm a device to grant it remote chat."
		/>
	{:else}
		<ul class="rc-list">
			{#each confirmedPeers as peer (peer.peer_id)}
				<li class="rc-item">
					<div class="rc-peer">
						<Icon name="smartphone" />
						<span class="rc-label">{peer.label || peer.peer_id}</span>
						<span class="rc-key">{shortRoutingKey(peer.routing_key)}</span>
					</div>
					{#if grants[peer.peer_id]}
						<div class="rc-controls">
							<Button
								variant={grants[peer.peer_id].remote_chat_enabled ? 'secondary' : 'primary'}
								disabled={busy[peer.peer_id]}
								on:click={() =>
									setEnabled(peer.peer_id, !grants[peer.peer_id].remote_chat_enabled)}
							>
								{grants[peer.peer_id].remote_chat_enabled
									? 'Disable remote chat'
									: 'Enable remote chat'}
							</Button>
							<Button
								variant="secondary"
								disabled={busy[peer.peer_id] || !grants[peer.peer_id].remote_chat_enabled}
								on:click={() => setRag(peer.peer_id, !grants[peer.peer_id].rag_subgrant)}
							>
								{grants[peer.peer_id].rag_subgrant ? 'RAG read: on' : 'RAG read: off'}
							</Button>
							<Button
								variant="danger"
								disabled={busy[peer.peer_id]}
								on:click={() => revoke(peer.peer_id)}
							>
								<Icon name="x" /> Revoke
							</Button>
						</div>
					{:else}
						<span class="rc-loading">Loading grant&hellip;</span>
					{/if}
				</li>
			{/each}
		</ul>
	{/if}

	{#if telemetry}
		<p class="rc-telemetry">Live remote sessions: {telemetry.active_sessions}</p>
	{/if}
</Card>

<style>
	.rc-title {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2, 0.5rem);
		margin: 0 0 var(--oo-space-2, 0.5rem);
		font-size: var(--oo-font-size-lg, 1.125rem);
		color: var(--oo-fg-primary);
	}

	.rc-intro {
		margin: 0 0 var(--oo-space-3, 0.75rem);
		color: var(--oo-fg-secondary);
		font-size: var(--oo-font-size-sm, 0.875rem);
	}

	.rc-list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3, 0.75rem);
	}

	.rc-item {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2, 0.5rem);
		padding: var(--oo-space-3, 0.75rem);
		border: 1px solid var(--oo-border-subtle, rgba(127, 127, 127, 0.2));
		border-radius: var(--oo-radius-md, 0.5rem);
	}

	.rc-peer {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2, 0.5rem);
	}

	.rc-label {
		font-weight: 600;
		color: var(--oo-fg-primary);
	}

	.rc-key {
		color: var(--oo-fg-secondary);
		font-family: var(--oo-font-mono, monospace);
		font-size: var(--oo-font-size-xs, 0.75rem);
	}

	.rc-controls {
		display: flex;
		flex-wrap: wrap;
		gap: var(--oo-space-2, 0.5rem);
	}

	.rc-loading {
		color: var(--oo-fg-secondary);
		font-size: var(--oo-font-size-sm, 0.875rem);
	}

	.rc-telemetry {
		margin: var(--oo-space-3, 0.75rem) 0 0;
		color: var(--oo-fg-secondary);
		font-size: var(--oo-font-size-xs, 0.75rem);
	}
</style>
