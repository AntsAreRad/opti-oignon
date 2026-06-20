<!--
  ObservabilityPanel.svelte -- S114 Combined Observability View.

  Unified panel linking Telemetry, Profiler, and Performance dashboards.
  Features:
  1. Quick status overview widget showing all three subsystem statuses
  2. Sub-tab navigation between the three dashboards
  3. Cross-linking: click a model in profiler to filter telemetry history by model
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import TelemetryDashboard from './TelemetryDashboard.svelte';
	import ProfilerDashboard from './ProfilerDashboard.svelte';
	import PerformanceDashboard from './PerformanceDashboard.svelte';
	import TelemetryHistoryPanel from './TelemetryHistoryPanel.svelte';
	import { getTelemetryStats } from '$lib/api/telemetry';
	import { getProfilerSummary } from '$lib/api/profiler';
	import { getHistoryStats } from '$lib/api/telemetry';
	import type { TelemetryStats, TelemetryHistoryStats } from '$lib/api/telemetry';
	import type { ProfilerSummaryResponse } from '$lib/api/profiler';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	type SubTab = 'overview' | 'telemetry' | 'profiler' | 'performance' | 'history';
	let activeSubTab: SubTab = 'overview';

	// Status overview
	let statusLoading = true;
	let telemetryStatus: TelemetryStats | null = null;
	let profilerStatus: ProfilerSummaryResponse | null = null;
	let historyStatus: TelemetryHistoryStats | null = null;

	// Cross-linking: model selected from profiler
	let linkedModel = '';

	// -------------------------------------------------------------------------
	// Lifecycle
	// -------------------------------------------------------------------------

	onMount(loadOverview);

	async function loadOverview() {
		statusLoading = true;
		try {
			const [ts, ps, hs] = await Promise.all([
				getTelemetryStats().catch(() => null),
				getProfilerSummary().catch(() => null),
				getHistoryStats().catch(() => null),
			]);
			telemetryStatus = ts;
			profilerStatus = ps;
			historyStatus = hs;
		} catch {
			// Silently degrade
		} finally {
			statusLoading = false;
		}
	}

	// -------------------------------------------------------------------------
	// Cross-linking
	// -------------------------------------------------------------------------

	function handleProfilerModelSelect(e: CustomEvent<string>) {
		linkedModel = e.detail;
		if (linkedModel) {
			activeSubTab = 'history';
		}
	}

	function clearLinkedModel() {
		linkedModel = '';
	}

	// -------------------------------------------------------------------------
	// Helpers
	// -------------------------------------------------------------------------

	const subTabs: { id: SubTab; label: string }[] = [
		{ id: 'overview', label: 'Overview' },
		{ id: 'telemetry', label: 'Telemetry' },
		{ id: 'profiler', label: 'Profiler' },
		{ id: 'performance', label: 'Performance' },
		{ id: 'history', label: 'History' },
	];
</script>

<div class="observability-panel">
	<!-- Sub-tab navigation -->
	<div class="sub-tabs">
		{#each subTabs as tab}
			<button
				class="sub-tab"
				class:active={activeSubTab === tab.id}
				on:click={() => activeSubTab = tab.id}
			>
				{tab.label}
			</button>
		{/each}
	</div>

	<!-- Overview tab -->
	{#if activeSubTab === 'overview'}
		<div class="overview-grid">
			<!-- Telemetry status -->
			<button class="status-card" on:click={() => activeSubTab = 'telemetry'}>
				<div class="status-header">
					<span class="status-dot" class:green={telemetryStatus?.enabled} class:gray={!telemetryStatus?.enabled}></span>
					<h4>Telemetry Pipeline</h4>
				</div>
				{#if statusLoading}
					<div class="status-loading">Loading...</div>
				{:else if telemetryStatus}
					<div class="status-metrics">
						<div class="sm-row">
							<span class="sm-label">Events</span>
							<span class="sm-value">{telemetryStatus.total_events.toLocaleString()}</span>
						</div>
						<div class="sm-row">
							<span class="sm-label">Requests</span>
							<span class="sm-value">{telemetryStatus.total_requests.toLocaleString()}</span>
						</div>
						<div class="sm-row">
							<span class="sm-label">Tokens</span>
							<span class="sm-value">{telemetryStatus.total_tokens.toLocaleString()}</span>
						</div>
						<div class="sm-row">
							<span class="sm-label">Consumers</span>
							<span class="sm-value">{telemetryStatus.consumer_count}</span>
						</div>
					</div>
				{:else}
					<div class="status-unavailable">Unavailable</div>
				{/if}
			</button>

			<!-- Profiler status -->
			<button class="status-card" on:click={() => activeSubTab = 'profiler'}>
				<div class="status-header">
					<span class="status-dot" class:green={profilerStatus && profilerStatus.total_profiled_requests > 0} class:gray={!profilerStatus || profilerStatus.total_profiled_requests === 0}></span>
					<h4>Inference Profiler</h4>
				</div>
				{#if statusLoading}
					<div class="status-loading">Loading...</div>
				{:else if profilerStatus}
					<div class="status-metrics">
						<div class="sm-row">
							<span class="sm-label">Profiled</span>
							<span class="sm-value">{profilerStatus.total_profiled_requests.toLocaleString()}</span>
						</div>
						<div class="sm-row">
							<span class="sm-label">Models</span>
							<span class="sm-value">{profilerStatus.models.length}</span>
						</div>
						{#if profilerStatus.models.length > 0}
							{@const topModel = profilerStatus.models.reduce((a, b) => a.request_count > b.request_count ? a : b)}
							<div class="sm-row">
								<span class="sm-label">Top model</span>
								<span class="sm-value sm-truncate" title={topModel.model}>{topModel.model}</span>
							</div>
							<div class="sm-row">
								<span class="sm-label">Avg latency</span>
								<span class="sm-value">{Math.round(topModel.avg_total_ms)}ms</span>
							</div>
						{/if}
					</div>
				{:else}
					<div class="status-unavailable">Unavailable</div>
				{/if}
			</button>

			<!-- History status -->
			<button class="status-card" on:click={() => activeSubTab = 'history'}>
				<div class="status-header">
					<span class="status-dot" class:green={historyStatus?.available} class:gray={!historyStatus?.available}></span>
					<h4>Event History</h4>
				</div>
				{#if statusLoading}
					<div class="status-loading">Loading...</div>
				{:else if historyStatus && historyStatus.available}
					<div class="status-metrics">
						<div class="sm-row">
							<span class="sm-label">Stored</span>
							<span class="sm-value">{historyStatus.total_stored.toLocaleString()}</span>
						</div>
						<div class="sm-row">
							<span class="sm-label">Retention</span>
							<span class="sm-value">{historyStatus.retention_days}d</span>
						</div>
						<div class="sm-row">
							<span class="sm-label">Max</span>
							<span class="sm-value">{historyStatus.max_events.toLocaleString()}</span>
						</div>
					</div>
				{:else}
					<div class="status-unavailable">Unavailable</div>
				{/if}
			</button>
		</div>

		<div class="overview-hint">
			Click a card to open the corresponding dashboard, or use the tabs above.
		</div>
	{/if}

	<!-- Telemetry tab -->
	{#if activeSubTab === 'telemetry'}
		<TelemetryDashboard />
	{/if}

	<!-- Profiler tab -->
	{#if activeSubTab === 'profiler'}
		<ProfilerDashboard on:selectModel={handleProfilerModelSelect} />
	{/if}

	<!-- Performance tab -->
	{#if activeSubTab === 'performance'}
		<PerformanceDashboard />
	{/if}

	<!-- History tab -->
	{#if activeSubTab === 'history'}
		{#if linkedModel}
			<div class="linked-model-banner">
				Filtered by model: <strong>{linkedModel}</strong>
				<button class="btn-clear-link" on:click={clearLinkedModel}>Clear filter</button>
			</div>
		{/if}
		<TelemetryHistoryPanel initialModelFilter={linkedModel} />
	{/if}
</div>

<style>
	.observability-panel {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	/* Sub-tab navigation */
	.sub-tabs {
		display: flex;
		gap: 0.25rem;
		padding-bottom: 0.5rem;
		border-bottom: 1px solid var(--oo-bd-subtle);
	}

	.sub-tab {
		padding: 0.4rem 0.85rem;
		border-radius: 6px;
		font-size: 0.82rem;
		cursor: pointer;
		border: 1px solid transparent;
		background: transparent;
		color: var(--oo-text-secondary);
		transition: background 0.15s, color 0.15s;
	}

	.sub-tab:hover {
		background: var(--oo-bg-elevated);
		color: var(--oo-text-primary);
	}

	.sub-tab.active {
		background: var(--oo-accent-primary);
		color: var(--oo-fg-on-accent);
		border-color: var(--oo-accent-primary);
		font-weight: 600;
	}

	/* Overview grid */
	.overview-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
		gap: 0.75rem;
	}

	.status-card {
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-subtle);
		border-radius: 8px;
		padding: 1rem;
		cursor: pointer;
		transition: border-color 0.15s, box-shadow 0.15s;
		text-align: left;
		width: 100%;
		font-family: inherit;
		color: inherit;
	}

	.status-card:hover {
		border-color: var(--oo-accent-primary);
		box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
	}

	.status-header {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.75rem;
	}

	.status-header h4 {
		margin: 0;
		font-size: 0.88rem;
		font-weight: 600;
		color: var(--oo-text-primary);
	}

	.status-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.status-dot.green {
		background: var(--oo-success);
	}

	.status-dot.gray {
		background: var(--oo-text-tertiary);
		opacity: 0.5;
	}

	.status-metrics {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
	}

	.sm-row {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
	}

	.sm-label {
		font-size: 0.75rem;
		color: var(--oo-text-tertiary);
	}

	.sm-value {
		font-size: 0.82rem;
		font-weight: 600;
		color: var(--oo-text-primary);
		font-variant-numeric: tabular-nums;
	}

	.sm-truncate {
		max-width: 120px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.status-loading {
		font-size: 0.78rem;
		color: var(--oo-text-tertiary);
	}

	.status-unavailable {
		font-size: 0.78rem;
		color: var(--oo-text-tertiary);
		font-style: italic;
	}

	.overview-hint {
		font-size: 0.75rem;
		color: var(--oo-text-tertiary);
		text-align: center;
		padding: 0.5rem;
	}

	/* Cross-link banner */
	.linked-model-banner {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.85rem;
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-accent-primary);
		border-radius: 6px;
		font-size: 0.82rem;
		color: var(--oo-text-primary);
	}

	.btn-clear-link {
		padding: 0.2rem 0.5rem;
		border-radius: 4px;
		font-size: 0.72rem;
		cursor: pointer;
		border: 1px solid var(--oo-bd-subtle);
		background: var(--oo-bg-elevated);
		color: var(--oo-text-secondary);
		margin-left: auto;
	}

	.btn-clear-link:hover {
		background: var(--oo-bg-overlay);
	}
</style>
