<!--
  TelemetryDashboard.svelte -- Inference Telemetry Dashboard.

  Shows:
  1. Event counters (total events, requests, tokens)
  2. Consumer health badges
  3. Buffer utilization gauge
  4. Active requests indicator
  5. Manual flush button
  6. Auto-refresh toggle
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		getTelemetryStats,
		getTelemetryConsumers,
		flushTelemetry,
	} from '$lib/api/telemetry';
	import type {
		TelemetryStats,
		TelemetryConsumerInfo,
	} from '$lib/api/telemetry';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	let loading = true;
	let error = '';

	let stats: TelemetryStats | null = null;
	let consumers: TelemetryConsumerInfo[] = [];
	let flushing = false;

	// Auto-refresh
	let autoRefresh = false;
	let refreshInterval: ReturnType<typeof setInterval> | null = null;
	let refreshSeconds = 5;
	let lastRefresh = '';

	// -------------------------------------------------------------------------
	// Lifecycle
	// -------------------------------------------------------------------------

	onMount(loadAll);
	onDestroy(stopAutoRefresh);

	async function loadAll() {
		loading = true;
		error = '';
		try {
			const [s, c] = await Promise.all([
				getTelemetryStats(),
				getTelemetryConsumers().catch(() => ({ consumers: [], count: 0 })),
			]);
			stats = s;
			consumers = c.consumers || [];
			lastRefresh = new Date().toLocaleTimeString();
		} catch (e: any) {
			error = e?.detail || e?.message || 'Failed to load telemetry data';
		} finally {
			loading = false;
		}
	}

	// -------------------------------------------------------------------------
	// Actions
	// -------------------------------------------------------------------------

	async function handleFlush() {
		flushing = true;
		try {
			const result = await flushTelemetry();
			await loadAll();
		} catch (e: any) {
			error = e?.detail || e?.message || 'Flush failed';
		} finally {
			flushing = false;
		}
	}

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

	// -------------------------------------------------------------------------
	// Computed
	// -------------------------------------------------------------------------

	$: bufferPct = stats
		? Math.min(100, Math.round((stats.buffer_size / Math.max(stats.buffer_max_size, 1)) * 100))
		: 0;
</script>

<div class="telemetry-dashboard">
	<div class="dashboard-header">
		<h2>Inference Telemetry</h2>
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
			<button
				class="btn-primary"
				on:click={handleFlush}
				disabled={flushing || !stats?.enabled}
			>
				{flushing ? 'Flushing...' : 'Flush Buffer'}
			</button>
		</div>
	</div>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	{#if loading && !stats}
		<div class="loading-state">Loading telemetry data...</div>
	{:else if stats}
		<!-- Status badge -->
		<div class="status-row">
			<span class="status-badge" class:enabled={stats.enabled} class:disabled={!stats.enabled}>
				{stats.enabled ? 'Pipeline Active' : 'Pipeline Disabled'}
			</span>
			{#if lastRefresh}
				<span class="last-refresh">Last update: {lastRefresh}</span>
			{/if}
		</div>

		<!-- Metric cards -->
		<div class="metric-grid">
			<div class="metric-card">
				<div class="metric-value">{stats.total_events.toLocaleString()}</div>
				<div class="metric-label">Total Events</div>
			</div>
			<div class="metric-card">
				<div class="metric-value">{stats.total_requests.toLocaleString()}</div>
				<div class="metric-label">Total Requests</div>
			</div>
			<div class="metric-card">
				<div class="metric-value">{stats.total_tokens.toLocaleString()}</div>
				<div class="metric-label">Total Tokens</div>
			</div>
			<div class="metric-card">
				<div class="metric-value">{stats.active_requests}</div>
				<div class="metric-label">Active Requests</div>
			</div>
		</div>

		<!-- Buffer gauge -->
		<div class="section">
			<h3>Buffer Utilization</h3>
			<div class="buffer-gauge">
				<div class="gauge-bar">
					<div
						class="gauge-fill"
						class:gauge-low={bufferPct < 50}
						class:gauge-mid={bufferPct >= 50 && bufferPct < 80}
						class:gauge-high={bufferPct >= 80}
						style="width: {bufferPct}%"
					></div>
				</div>
				<div class="gauge-label">
					{stats.buffer_size} / {stats.buffer_max_size} events ({bufferPct}%)
				</div>
			</div>
		</div>

		<!-- Consumer health -->
		<div class="section">
			<h3>Consumers ({stats.consumer_count})</h3>
			{#if consumers.length === 0}
				<div class="empty-state">No consumers registered</div>
			{:else}
				<div class="consumer-list">
					{#each consumers as consumer}
						<div class="consumer-badge" class:healthy={consumer.healthy} class:unhealthy={!consumer.healthy}>
							<span class="health-dot"></span>
							<span class="consumer-name">{consumer.name}</span>
						</div>
					{/each}
				</div>
			{/if}
		</div>
	{:else}
		<div class="empty-state">Telemetry data unavailable</div>
	{/if}
</div>

<style>
	.telemetry-dashboard {
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

	.btn-secondary,
	.btn-primary {
		padding: 0.4rem 0.85rem;
		border-radius: 6px;
		font-size: 0.82rem;
		cursor: pointer;
		border: 1px solid var(--oo-bd-subtle);
		transition: background 0.15s, opacity 0.15s;
	}

	.btn-secondary {
		background: var(--oo-bg-elevated);
		color: var(--oo-text-primary);
	}

	.btn-secondary:hover {
		background: var(--oo-bg-overlay);
	}

	.btn-secondary.active {
		background: var(--oo-accent-primary);
		color: var(--oo-fg-on-accent);
		border-color: var(--oo-accent-primary);
	}

	.btn-primary {
		background: var(--oo-accent-primary);
		color: var(--oo-fg-on-accent);
		border-color: var(--oo-accent-primary);
	}

	.btn-primary:hover {
		opacity: 0.9;
	}

	.btn-primary:disabled,
	.btn-secondary:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.error-banner {
		background: var(--oo-danger-bg, rgba(200, 80, 80, 0.12));
		color: var(--oo-danger);
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
		gap: 1rem;
	}

	.status-badge {
		display: inline-block;
		padding: 0.25rem 0.7rem;
		border-radius: 12px;
		font-size: 0.78rem;
		font-weight: 600;
	}

	.status-badge.enabled {
		background: var(--oo-success-bg);
		color: var(--oo-success);
		border: 1px solid var(--oo-success-bd);
	}

	.status-badge.disabled {
		background: var(--oo-warning-bg);
		color: var(--oo-warning);
		border: 1px solid var(--oo-warning-bd);
	}

	.last-refresh {
		color: var(--oo-text-tertiary);
		font-size: 0.78rem;
	}

	/* Metric cards */
	.metric-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
		gap: 0.75rem;
	}

	.metric-card {
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-subtle);
		border-radius: 8px;
		padding: 1rem;
		text-align: center;
	}

	.metric-value {
		font-size: 1.5rem;
		font-weight: 700;
		color: var(--oo-text-primary);
		font-variant-numeric: tabular-nums;
	}

	.metric-label {
		font-size: 0.78rem;
		color: var(--oo-text-secondary);
		margin-top: 0.25rem;
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

	/* Buffer gauge */
	.buffer-gauge {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
	}

	.gauge-bar {
		height: 10px;
		background: var(--oo-bg-elevated);
		border-radius: 5px;
		overflow: hidden;
	}

	.gauge-fill {
		height: 100%;
		border-radius: 5px;
		transition: width 0.3s ease;
	}

	.gauge-fill.gauge-low {
		background: var(--oo-success);
	}

	.gauge-fill.gauge-mid {
		background: var(--oo-warning);
	}

	.gauge-fill.gauge-high {
		background: var(--oo-danger);
	}

	.gauge-label {
		font-size: 0.78rem;
		color: var(--oo-text-secondary);
	}

	/* Consumer badges */
	.consumer-list {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
	}

	.consumer-badge {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
		padding: 0.3rem 0.7rem;
		border-radius: 6px;
		font-size: 0.8rem;
		background: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-subtle);
	}

	.health-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
	}

	.consumer-badge.healthy .health-dot {
		background: var(--oo-success);
	}

	.consumer-badge.unhealthy .health-dot {
		background: var(--oo-danger);
	}

	.consumer-name {
		color: var(--oo-text-primary);
	}
</style>
