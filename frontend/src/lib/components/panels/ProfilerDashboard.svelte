<!--
  ProfilerDashboard.svelte -- Inference Profiler Dashboard.

  Shows:
  1. Per-model summary cards (avg/p50/p95/p99 latency, avg tok/s)
  2. Time breakdown visualization (stacked bar: prompt_eval / token_gen / overhead)
  3. Recent requests table (request_id, model, total_ms, tokens_in/out, tok/s)
  4. Auto-refresh toggle
-->
<script lang="ts">
	import { onMount, onDestroy, createEventDispatcher } from 'svelte';
	import {
		getProfilerSummary,
		getRecentProfiles,
	} from '$lib/api/profiler';
	import type {
		ProfilerSummary,
		ProfilerSummaryResponse,
		InferenceProfile,
		ProfilerRecentResponse,
	} from '$lib/api/profiler';

	const dispatch = createEventDispatcher<{ selectModel: string }>();

	// Allow parent to set initial model filter
	export let initialModel: string = '';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	let loading = true;
	let error = '';

	let summaryData: ProfilerSummaryResponse | null = null;
	let models: ProfilerSummary[] = [];
	let recentProfiles: InferenceProfile[] = [];
	let totalProfiled = 0;

	// Auto-refresh
	let autoRefresh = false;
	let refreshInterval: ReturnType<typeof setInterval> | null = null;
	let refreshSeconds = 5;
	let lastRefresh = '';

	// Selected model filter for recent table
	let selectedModel = initialModel;

	// -------------------------------------------------------------------------
	// Lifecycle
	// -------------------------------------------------------------------------

	onMount(loadAll);
	onDestroy(stopAutoRefresh);

	// React to external initialModel changes
	$: if (initialModel !== undefined) selectedModel = initialModel;

	async function loadAll() {
		loading = true;
		error = '';
		try {
			const [sumResp, recResp] = await Promise.all([
				getProfilerSummary().catch(() => ({ models: [], total_profiled_requests: 0 })),
				getRecentProfiles(50).catch(() => ({ profiles: [], count: 0 })),
			]);
			summaryData = sumResp;
			models = sumResp.models || [];
			totalProfiled = sumResp.total_profiled_requests || 0;
			recentProfiles = recResp.profiles || [];
			lastRefresh = new Date().toLocaleTimeString();
		} catch (e: any) {
			error = e?.detail || e?.message || 'Failed to load profiler data';
		} finally {
			loading = false;
		}
	}

	// -------------------------------------------------------------------------
	// Actions
	// -------------------------------------------------------------------------

	function toggleAutoRefresh() {
		autoRefresh = !autoRefresh;
		if (autoRefresh) {
			refreshInterval = setInterval(loadAll, refreshSeconds * 1000);
		} else {
			stopAutoRefresh();
		}
	}

	function stopAutoRefresh() {
		if (refreshInterval) {
			clearInterval(refreshInterval);
			refreshInterval = null;
		}
	}

	function filterByModel(model: string) {
		selectedModel = selectedModel === model ? '' : model;
		dispatch('selectModel', selectedModel);
	}

	// -------------------------------------------------------------------------
	// Computed
	// -------------------------------------------------------------------------

	$: filteredRecent = selectedModel
		? recentProfiles.filter(p => p.model === selectedModel)
		: recentProfiles;

	function fmtMs(ms: number): string {
		if (ms >= 1000) return (ms / 1000).toFixed(1) + 's';
		return Math.round(ms) + 'ms';
	}

	function fmtTokS(t: number): string {
		return t.toFixed(1);
	}

	function breakdownPct(m: ProfilerSummary): { prompt: number; gen: number; overhead: number } {
		const total = m.avg_prompt_eval_ms + m.avg_token_gen_ms + m.avg_overhead_ms;
		if (total <= 0) return { prompt: 33, gen: 34, overhead: 33 };
		return {
			prompt: Math.round((m.avg_prompt_eval_ms / total) * 100),
			gen: Math.round((m.avg_token_gen_ms / total) * 100),
			overhead: Math.round((m.avg_overhead_ms / total) * 100),
		};
	}

	function timeAgo(ts: number): string {
		const diff = Math.max(0, Math.floor((Date.now() / 1000) - ts));
		if (diff < 60) return diff + 's ago';
		if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
		if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
		return Math.floor(diff / 86400) + 'd ago';
	}
</script>

<div class="profiler-dashboard">
	<div class="dashboard-header">
		<h2>Inference Profiler</h2>
		<div class="header-actions">
			<button
				class="btn-secondary"
				on:click={loadAll}
				disabled={loading}
			>
				{loading ? 'Loading...' : 'Refresh'}
			</button>
			<button
				class="btn-secondary"
				class:active={autoRefresh}
				on:click={toggleAutoRefresh}
			>
				Auto ({refreshSeconds}s) {autoRefresh ? 'ON' : 'OFF'}
			</button>
		</div>
	</div>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	{#if loading && !summaryData}
		<div class="loading-state">Loading profiler data...</div>
	{:else}
		<!-- Overview stats -->
		<div class="status-row">
			<span class="stat-pill">
				{totalProfiled.toLocaleString()} total profiled requests
			</span>
			<span class="stat-pill">
				{models.length} model{models.length !== 1 ? 's' : ''}
			</span>
			{#if lastRefresh}
				<span class="last-refresh">Last update: {lastRefresh}</span>
			{/if}
		</div>

		<!-- Per-model summary cards -->
		{#if models.length === 0}
			<div class="empty-state">No profiling data yet. Send some inference requests to populate.</div>
		{:else}
			<div class="section">
				<h3>Per-Model Latency</h3>
				<div class="model-cards">
					{#each models as m}
						{@const pct = breakdownPct(m)}
						<button
							class="model-card"
							class:selected={selectedModel === m.model}
							on:click={() => filterByModel(m.model)}
						>
							<div class="model-name" title={m.model}>{m.model}</div>
							<div class="model-stats">
								<div class="stat-row">
									<span class="stat-label">avg</span>
									<span class="stat-value">{fmtMs(m.avg_total_ms)}</span>
								</div>
								<div class="stat-row">
									<span class="stat-label">p50</span>
									<span class="stat-value">{fmtMs(m.p50_total_ms)}</span>
								</div>
								<div class="stat-row">
									<span class="stat-label">p95</span>
									<span class="stat-value">{fmtMs(m.p95_total_ms)}</span>
								</div>
								<div class="stat-row">
									<span class="stat-label">p99</span>
									<span class="stat-value">{fmtMs(m.p99_total_ms)}</span>
								</div>
							</div>
							<div class="stat-row tok-row">
								<span class="stat-label">tok/s</span>
								<span class="stat-value accent">{fmtTokS(m.avg_tok_per_sec)}</span>
							</div>
							<div class="stat-row">
								<span class="stat-label">reqs</span>
								<span class="stat-value">{m.request_count}</span>
							</div>

							<!-- Stacked time breakdown bar -->
							<div class="breakdown-bar" title="Prompt eval: {pct.prompt}% | Token gen: {pct.gen}% | Overhead: {pct.overhead}%">
								<div class="bar-prompt" style="width: {pct.prompt}%"></div>
								<div class="bar-gen" style="width: {pct.gen}%"></div>
								<div class="bar-overhead" style="width: {pct.overhead}%"></div>
							</div>
							<div class="breakdown-legend">
								<span class="legend-item"><span class="dot dot-prompt"></span>Prompt</span>
								<span class="legend-item"><span class="dot dot-gen"></span>Generate</span>
								<span class="legend-item"><span class="dot dot-overhead"></span>Overhead</span>
							</div>
						</button>
					{/each}
				</div>
			</div>

			<!-- Recent requests table -->
			<div class="section">
				<div class="section-header">
					<h3>Recent Requests{selectedModel ? ` (${selectedModel})` : ''}</h3>
					{#if selectedModel}
						<button class="btn-clear" on:click={() => selectedModel = ''}>Clear filter</button>
					{/if}
				</div>
				{#if filteredRecent.length === 0}
					<div class="empty-state">No recent profiles{selectedModel ? ' for this model' : ''}</div>
				{:else}
					<div class="table-wrap">
						<table class="profiler-table">
							<thead>
								<tr>
									<th>Time</th>
									<th>Model</th>
									<th>Total</th>
									<th>Prompt</th>
									<th>Gen</th>
									<th>Overhead</th>
									<th>In</th>
									<th>Out</th>
									<th>tok/s</th>
								</tr>
							</thead>
							<tbody>
								{#each filteredRecent.slice(0, 30) as p}
									<tr>
										<td class="cell-time" title={new Date(p.timestamp * 1000).toLocaleString()}>
											{timeAgo(p.timestamp)}
										</td>
										<td class="cell-model" title={p.model}>
											{p.model.length > 20 ? p.model.slice(0, 18) + '...' : p.model}
										</td>
										<td class="cell-num">{fmtMs(p.total_ms)}</td>
										<td class="cell-num cell-prompt">{fmtMs(p.prompt_eval_ms)}</td>
										<td class="cell-num cell-gen">{fmtMs(p.token_gen_ms)}</td>
										<td class="cell-num cell-overhead">{fmtMs(p.overhead_ms)}</td>
										<td class="cell-num">{p.tokens_in}</td>
										<td class="cell-num">{p.tokens_out}</td>
										<td class="cell-num cell-tok">{fmtTokS(p.tok_per_sec)}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
					{#if filteredRecent.length > 30}
						<div class="table-footnote">Showing 30 of {filteredRecent.length} profiles</div>
					{/if}
				{/if}
			</div>
		{/if}
	{/if}
</div>

<style>
	.profiler-dashboard {
		padding: 1.5rem;
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.dashboard-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 0.75rem;
	}

	.dashboard-header h2 {
		margin: 0;
		font-size: 1.3rem;
		color: var(--oo-text-primary);
	}

	.header-actions {
		display: flex;
		gap: 0.5rem;
		flex-wrap: wrap;
	}

	.btn-secondary {
		padding: 0.4rem 0.85rem;
		border-radius: 6px;
		font-size: 0.82rem;
		cursor: pointer;
		border: 1px solid var(--oo-bd-subtle);
		background: var(--oo-bg-elevated);
		color: var(--oo-text-primary);
		transition: background 0.15s, opacity 0.15s;
	}

	.btn-secondary:hover {
		background: var(--oo-bg-overlay);
	}

	.btn-secondary.active {
		background: var(--oo-accent-primary);
		color: var(--oo-fg-on-accent);
		border-color: var(--oo-accent-primary);
	}

	.btn-secondary:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn-clear {
		padding: 0.2rem 0.6rem;
		border-radius: 4px;
		font-size: 0.75rem;
		cursor: pointer;
		border: 1px solid var(--oo-bd-subtle);
		background: var(--oo-bg-elevated);
		color: var(--oo-text-secondary);
	}

	.btn-clear:hover {
		background: var(--oo-bg-overlay);
	}

	.error-banner {
		background: var(--oo-danger-bg, rgba(200, 80, 80, 0.12));
		color: var(--oo-error);
		border: 1px solid var(--oo-danger-bd, rgba(200, 80, 80, 0.2));
		border-radius: 6px;
		padding: 0.6rem 1rem;
		font-size: 0.85rem;
	}

	.loading-state,
	.empty-state {
		color: var(--oo-text-secondary);
		font-size: 0.9rem;
		padding: 2rem;
		text-align: center;
	}

	.status-row {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		flex-wrap: wrap;
	}

	.stat-pill {
		display: inline-block;
		padding: 0.25rem 0.7rem;
		border-radius: 12px;
		font-size: 0.78rem;
		font-weight: 600;
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-subtle);
		color: var(--oo-text-primary);
	}

	.last-refresh {
		color: var(--oo-text-tertiary);
		font-size: 0.78rem;
	}

	/* Sections */
	.section {
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-subtle);
		border-radius: 8px;
		padding: 1rem;
	}

	.section h3 {
		margin: 0 0 0.75rem 0;
		font-size: 0.95rem;
		color: var(--oo-text-primary);
	}

	.section-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.75rem;
	}

	.section-header h3 {
		margin: 0;
	}

	/* Model cards */
	.model-cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
		gap: 0.75rem;
	}

	.model-card {
		background: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-subtle);
		border-radius: 8px;
		padding: 0.85rem;
		cursor: pointer;
		transition: border-color 0.15s, box-shadow 0.15s;
		text-align: left;
		width: 100%;
		font-family: inherit;
		color: inherit;
	}

	.model-card:hover {
		border-color: var(--oo-acc-400);
	}

	.model-card.selected {
		border-color: var(--oo-accent-primary);
		box-shadow: 0 0 0 1px var(--oo-accent-primary);
	}

	.model-name {
		font-size: 0.82rem;
		font-weight: 600;
		color: var(--oo-text-primary);
		margin-bottom: 0.5rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.model-stats {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.2rem 0.5rem;
		margin-bottom: 0.4rem;
	}

	.stat-row {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 0.3rem;
	}

	.stat-label {
		font-size: 0.72rem;
		color: var(--oo-text-tertiary);
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}

	.stat-value {
		font-size: 0.82rem;
		font-weight: 600;
		color: var(--oo-text-primary);
		font-variant-numeric: tabular-nums;
	}

	.stat-value.accent {
		color: var(--oo-accent-primary);
	}

	.tok-row {
		margin-top: 0.2rem;
		padding-top: 0.3rem;
		border-top: 1px solid var(--oo-bd-subtle);
	}

	/* Breakdown stacked bar */
	.breakdown-bar {
		display: flex;
		height: 6px;
		border-radius: 3px;
		overflow: hidden;
		margin-top: 0.5rem;
	}

	.bar-prompt {
		background: var(--oo-accent-primary);
		height: 100%;
	}

	.bar-gen {
		background: var(--oo-acc-400);
		height: 100%;
	}

	.bar-overhead {
		background: var(--oo-warning);
		height: 100%;
		opacity: 0.6;
	}

	.breakdown-legend {
		display: flex;
		gap: 0.6rem;
		margin-top: 0.3rem;
	}

	.legend-item {
		display: inline-flex;
		align-items: center;
		gap: 0.2rem;
		font-size: 0.65rem;
		color: var(--oo-text-tertiary);
	}

	.dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		display: inline-block;
	}

	.dot-prompt { background: var(--oo-accent-primary); }
	.dot-gen { background: var(--oo-acc-400); }
	.dot-overhead { background: var(--oo-warning); opacity: 0.6; }

	/* Recent requests table */
	.table-wrap {
		overflow-x: auto;
	}

	.profiler-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.78rem;
	}

	.profiler-table th {
		text-align: left;
		padding: 0.4rem 0.5rem;
		font-weight: 600;
		color: var(--oo-text-secondary);
		border-bottom: 1px solid var(--oo-bd-subtle);
		white-space: nowrap;
		font-size: 0.72rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}

	.profiler-table td {
		padding: 0.35rem 0.5rem;
		border-bottom: 1px solid var(--oo-bd-subtle);
		color: var(--oo-text-primary);
	}

	.profiler-table tbody tr:hover {
		background: var(--oo-bg-elevated);
	}

	.cell-time {
		white-space: nowrap;
		color: var(--oo-text-tertiary);
	}

	.cell-model {
		max-width: 140px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-weight: 500;
	}

	.cell-num {
		text-align: right;
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
	}

	.cell-prompt { color: var(--oo-accent-primary); }
	.cell-gen { color: var(--oo-acc-400); }
	.cell-overhead { color: var(--oo-warning); }
	.cell-tok { font-weight: 600; }

	.table-footnote {
		font-size: 0.72rem;
		color: var(--oo-text-tertiary);
		text-align: center;
		padding: 0.4rem 0;
	}
</style>
