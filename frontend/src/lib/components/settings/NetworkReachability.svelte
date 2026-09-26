<!--
  NetworkReachability.svelte
  The server's reachability, at the head of the Workshop's network page:
  whether the inference server answers, how long it took, what waits in the
  offline queue, and the last error the server met. The status card in the
  sidebar says whether the backend is up; the detail lives here.

  It reads the network manager's status when the page is shown, and asks
  the server to check again on request (Refresh); it never polls.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import Button from '$lib/ds/Button.svelte';
	import { getNetworkStatus, pollNow, type NetworkStatusInfo } from '$lib/api/network';

	let status: NetworkStatusInfo | null = null;
	let failed = false;
	let busy = false;

	async function read(fresh: boolean) {
		busy = true;
		try {
			status = fresh ? await pollNow() : await getNetworkStatus();
			failed = false;
		} catch {
			failed = true;
		} finally {
			busy = false;
		}
	}

	$: headline = failed
		? 'The server did not answer'
		: !status
			? 'Checking the server'
			: !status.available
				? 'The network watch is off on this server'
				: status.online
					? 'The inference server is reachable'
					: 'The inference server is not reachable';

	onMount(() => {
		void read(false);
	});
</script>

<section class="oo-reach" aria-labelledby="oo-reach-title">
	<div class="oo-reach-head">
		<h2 class="oo-reach-title" id="oo-reach-title">Reachability</h2>
		<Button variant="secondary" size="sm" shape="pill" iconLeft="retry" loading={busy} on:click={() => read(true)}>
			Refresh
		</Button>
	</div>
	<p class="oo-reach-headline" role="status">{headline}</p>
	{#if status && status.available && !failed}
		<dl class="oo-reach-facts">
			<div>
				<dt>Latency</dt>
				<dd>{status.online && status.latency_ms > 0 ? `${Math.round(status.latency_ms)} ms` : 'Not measured'}</dd>
			</div>
			<div>
				<dt>Embeddings</dt>
				<dd>{status.embedding_reachable ? 'Reachable' : 'Not reachable'}</dd>
			</div>
			<div>
				<dt>Offline queue</dt>
				<dd>{status.queue_size === 1 ? '1 request' : `${status.queue_size} requests`}</dd>
			</div>
			{#if status.last_error}
				<div>
					<dt>Last error</dt>
					<dd class="oo-reach-error">{status.last_error}</dd>
				</div>
			{/if}
		</dl>
	{/if}
</section>

<style>
	.oo-reach {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
		margin-bottom: var(--oo-space-4);
		padding: var(--oo-space-4) var(--oo-space-5);
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-2xl);
		background-color: var(--oo-bg-surface);
		color: var(--oo-fg-primary);
	}
	.oo-reach-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-3);
	}
	.oo-reach-title {
		margin: 0;
		font-family: var(--oo-font-serif);
		font-size: var(--oo-text-lg);
		font-weight: 400;
	}
	.oo-reach-headline {
		margin: 0;
		font-size: var(--oo-text-sm);
		font-weight: 500;
	}
	.oo-reach-facts {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
		gap: var(--oo-space-3) var(--oo-space-5);
		margin: 0;
	}
	.oo-reach-facts dt {
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-xs);
	}
	.oo-reach-facts dd {
		margin: 0;
		font-size: var(--oo-text-sm);
	}
	.oo-reach-error {
		color: var(--oo-error);
		overflow-wrap: anywhere;
	}
</style>
