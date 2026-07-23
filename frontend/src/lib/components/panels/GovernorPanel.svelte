<!--
  GovernorPanel.svelte (Resource Governor cycle Bloc 4)
  The status card for the Resource Governor, built on the lib/ds primitives
  (Card, Button, Icon, EmptyState, InlineError). It surfaces what the governor
  already measures over /api/governor: capacity and in-use VRAM with honest
  provenance (which read paths contributed), the learned ceiling, the pressure
  level, the queue depth, the external-Ollama advisory, and the last admission
  decisions with refusals highlighted.

  The governor is mode-free: it reports identically in Daily and Bulbe (a local
  resource control with no egress, no secrets, no state mutation). When capacity
  is unknown the snapshot says so honestly rather than guessing, and the advisory
  degrades to "unknown" for an externally-managed Ollama instead of claiming
  enforcement. Read-only this card: eviction and config writes are API surfaces;
  the card does not act.

  On-demand fetch (onMount plus a manual Refresh), the SyncPanel idiom; live
  polling is a host concern. Updates announce through an aria-live region.
  Design-system tokens only (--oo-*); lucide icons through Icon.
  Registered in FRONTEND_REDESIGN_SPEC.md.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { Button, Card, Icon, EmptyState, InlineError } from '$lib/ds';
	import {
		getGovernorStatus,
		getGovernorAdmissions,
		type GovernorStatus,
		type AdmissionRecord
	} from '$lib/api/governor';

	let status: GovernorStatus | null = null;
	let admissions: AdmissionRecord[] = [];
	let loading = false;
	let error = '';

	function fmtGb(value: number | null | undefined): string {
		if (value === null || value === undefined) {
			return 'unknown';
		}
		return `${value.toFixed(1)} GB`;
	}

	function pressureLabel(level: string): string {
		if (level === 'hard') {
			return 'High';
		}
		if (level === 'soft') {
			return 'Elevated';
		}
		return 'Normal';
	}

	function isRefusal(decision: string): boolean {
		return decision === 'refuse' || decision === 'queue_timeout';
	}

	async function load() {
		loading = true;
		error = '';
		try {
			const [s, a] = await Promise.all([
				getGovernorStatus(),
				getGovernorAdmissions(20)
			]);
			status = s;
			admissions = a.admissions;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to read governor status';
		} finally {
			loading = false;
		}
	}

	onMount(load);
</script>

<section class="governor-panel">
	<header class="governor-header">
		<div class="governor-title">
			<Icon name="gauge" />
			<h2>Resource governor</h2>
		</div>
		<Button variant="ghost" on:click={load} disabled={loading}>
			<Icon name="refresh-cw" />
			Refresh
		</Button>
	</header>

	<p class="governor-aria" aria-live="polite">
		{#if loading}Reading governor status.{:else if status}Governor status updated.{/if}
	</p>

	{#if error}
		<InlineError message={error} onRetry={load} />
	{/if}

	{#if status}
		<Card>
			<div class="governor-grid">
				<div class="governor-metric">
					<span class="governor-metric-label">Capacity</span>
					<span class="governor-metric-value">{fmtGb(status.snapshot.capacity_gb)}</span>
					<span class="governor-metric-sub">source: {status.snapshot.capacity_source}</span>
				</div>
				<div class="governor-metric">
					<span class="governor-metric-label">In use</span>
					<span class="governor-metric-value">{fmtGb(status.snapshot.vram_in_use_gb)}</span>
					<span class="governor-metric-sub">available: {fmtGb(status.snapshot.vram_available_gb)}</span>
				</div>
				<div class="governor-metric">
					<span class="governor-metric-label">Pressure</span>
					<span class="governor-metric-value" data-level={status.pressure.level}>
						{pressureLabel(status.pressure.level)}
					</span>
					<span class="governor-metric-sub">
						{#if status.pressure.ratio !== null}{(status.pressure.ratio * 100).toFixed(0)}% of capacity{:else}ratio unknown{/if}
					</span>
				</div>
				<div class="governor-metric">
					<span class="governor-metric-label">Queue depth</span>
					<span class="governor-metric-value">{status.queue_depth}</span>
					<span class="governor-metric-sub">
						{#if status.pressure.keep_alive_overridden}keep_alive shortened{:else}keep_alive nominal{/if}
					</span>
				</div>
			</div>

			<p class="governor-provenance">
				<Icon name="info" />
				Measurement sources: {status.snapshot.sources.join(', ') || 'none'}.
				{#if status.snapshot.capacity_gb === null}
					Capacity is unknown; admission stays fail-open.
				{/if}
				{#if status.learned_ceiling_gb !== null}
					Learned ceiling: {fmtGb(status.learned_ceiling_gb)}.
				{/if}
			</p>

			<p class="governor-advisory">
				<Icon name="server" />
				External Ollama limits: {status.ollama_limits.status}.
			</p>
		</Card>

		<Card>
			<div class="governor-section-head">
				<Icon name="list" />
				<h3>Recent admissions</h3>
			</div>
			{#if admissions.length === 0}
				<EmptyState
					icon="inbox"
					title="No admission decisions yet"
					description="The governor records a decision on each model load attempt."
				/>
			{:else}
				<ul class="governor-decisions">
					{#each admissions as record (record.id)}
						<li class="governor-decision" class:governor-refusal={isRefusal(record.decision)}>
							<span class="governor-decision-model">{record.model}</span>
							<span class="governor-decision-meta">
								{record.caller} &middot; {record.decision}
								{#if record.reason}&middot; {record.reason}{/if}
							</span>
						</li>
					{/each}
				</ul>
			{/if}
		</Card>
	{:else if !error}
		<EmptyState
			icon="gauge"
			title="Loading governor status"
			description="Reading capacity, pressure and recent admissions."
		/>
	{/if}
</section>

<style>
	.governor-panel {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-4);
	}
	.governor-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}
	.governor-title {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
	}
	.governor-title h2 {
		margin: 0;
		font-size: var(--oo-text-lg);
		color: var(--oo-fg-primary);
	}
	.governor-aria {
		margin: 0;
		height: 0;
		overflow: hidden;
		color: var(--oo-fg-muted);
	}
	.governor-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
		gap: var(--oo-space-4);
	}
	.governor-metric {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
	}
	.governor-metric-label {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}
	.governor-metric-value {
		font-size: var(--oo-text-xl);
		color: var(--oo-fg-primary);
	}
	.governor-metric-value[data-level='hard'] {
		color: var(--oo-error);
	}
	.governor-metric-value[data-level='soft'] {
		color: var(--oo-warning);
	}
	.governor-metric-sub {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
	}
	.governor-provenance,
	.governor-advisory {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		margin: var(--oo-space-3) 0 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
	}
	.governor-section-head {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		margin-bottom: var(--oo-space-2);
	}
	.governor-section-head h3 {
		margin: 0;
		font-size: var(--oo-text-base);
		color: var(--oo-fg-primary);
	}
	.governor-decisions {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}
	.governor-decision {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
		padding: var(--oo-space-2);
		border-left: 2px solid var(--oo-bd-default);
	}
	.governor-decision.governor-refusal {
		border-left-color: var(--oo-error);
	}
	.governor-decision-model {
		color: var(--oo-fg-primary);
	}
	.governor-decision-meta {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
	}
</style>
