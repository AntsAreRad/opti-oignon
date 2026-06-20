<!--
  TelemetryHistoryPanel.svelte -- S115 Telemetry History Dashboard.

  Browsing and visualization of SQLite-backed telemetry event history.
  Features:
  1. Paginated event table with model filter dropdown
  2. Hourly trend chart (latency + throughput sparklines via CSS bars)
  3. Per-model breakdown cards with event counts and avg latency
  4. History stats display (total stored, retention, oldest event)
  5. Purge controls (purge by age, purge all with confirmation)
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		getTelemetryHistory,
		getTelemetryTrends,
		getHistoryModelBreakdown,
		getHistoryStats,
		purgeHistory,
		updateHistorySettings,
		getHistoryExportUrl,
	} from '$lib/api/telemetry';
	import type {
		HistoryEvent,
		TrendBucket,
		ModelBreakdown,
		TelemetryHistoryStats,
	} from '$lib/api/telemetry';
	import EventTimeline from './EventTimeline.svelte';

	// -------------------------------------------------------------------------
	// Props
	// -------------------------------------------------------------------------

	/** Optional model filter set externally (e.g. cross-link from profiler). */
	export let initialModelFilter: string = '';

	// -------------------------------------------------------------------------
	// State — history table
	// -------------------------------------------------------------------------

	let loading = true;
	let error = '';

	let events: HistoryEvent[] = [];
	let totalEvents = 0;
	let pageSize = 25;
	let currentPage = 0;
	let modelFilter = '';

	// Available models for dropdown
	let availableModels: string[] = [];

	// -------------------------------------------------------------------------
	// State — trends
	// -------------------------------------------------------------------------

	let trendBuckets: TrendBucket[] = [];
	let trendHours = 24;
	let trendLoading = false;

	// -------------------------------------------------------------------------
	// State — model breakdown
	// -------------------------------------------------------------------------

	let modelBreakdown: ModelBreakdown[] = [];

	// -------------------------------------------------------------------------
	// State — stats & purge
	// -------------------------------------------------------------------------

	let historyStats: TelemetryHistoryStats | null = null;
	let purging = false;
	let purgeConfirmOpen = false;
	let purgeDays = 7;

	// Retention settings (S115)
	let retentionDays = 7;
	let autoPurgeEnabled = false;
	let savingSettings = false;

	// Auto-refresh
	let autoRefresh = false;
	let refreshTimer: ReturnType<typeof setInterval> | null = null;

	// -------------------------------------------------------------------------
	// Lifecycle
	// -------------------------------------------------------------------------

	onMount(() => {
		if (initialModelFilter) {
			modelFilter = initialModelFilter;
		}
		loadAll();
	});

	onDestroy(stopAutoRefresh);

	// -------------------------------------------------------------------------
	// Data loading
	// -------------------------------------------------------------------------

	async function loadAll() {
		loading = true;
		error = '';
		try {
			await Promise.all([
				loadHistory(),
				loadTrends(),
				loadModelBreakdown(),
				loadStats(),
			]);
		} catch (e: any) {
			error = e?.message || 'Failed to load history data';
		} finally {
			loading = false;
		}
	}

	async function loadHistory() {
		try {
			const res = await getTelemetryHistory(pageSize, currentPage * pageSize, modelFilter);
			events = res.events || [];
			totalEvents = res.total || 0;
		} catch {
			events = [];
			totalEvents = 0;
		}
	}

	async function loadTrends() {
		trendLoading = true;
		try {
			const res = await getTelemetryTrends(trendHours, modelFilter);
			trendBuckets = res.buckets || [];
		} catch {
			trendBuckets = [];
		} finally {
			trendLoading = false;
		}
	}

	async function loadModelBreakdown() {
		try {
			const res = await getHistoryModelBreakdown();
			modelBreakdown = res.models || [];
			availableModels = modelBreakdown.map((m) => m.model).filter(Boolean);
		} catch {
			modelBreakdown = [];
		}
	}

	async function loadStats() {
		try {
			historyStats = await getHistoryStats();
			if (historyStats) {
				retentionDays = historyStats.retention_days;
				autoPurgeEnabled = historyStats.auto_purge_enabled;
			}
		} catch {
			historyStats = null;
		}
	}

	// -------------------------------------------------------------------------
	// Retention settings (S115)
	// -------------------------------------------------------------------------

	async function saveSettings() {
		savingSettings = true;
		try {
			await updateHistorySettings({
				retention_days: retentionDays,
				auto_purge_enabled: autoPurgeEnabled,
			});
			await loadStats();
		} catch (e: any) {
			error = e?.message || 'Failed to save settings';
		} finally {
			savingSettings = false;
		}
	}

	function handleExportCsv() {
		const url = getHistoryExportUrl(modelFilter);
		window.open(url, '_blank');
	}

	// -------------------------------------------------------------------------
	// Storage usage
	// -------------------------------------------------------------------------

	$: storageUsagePct = historyStats
		? Math.min(100, Math.round((historyStats.total_stored / Math.max(historyStats.max_events, 1)) * 100))
		: 0;

	// -------------------------------------------------------------------------
	// Pagination
	// -------------------------------------------------------------------------

	$: totalPages = Math.max(1, Math.ceil(totalEvents / pageSize));

	function goToPage(page: number) {
		if (page < 0 || page >= totalPages) return;
		currentPage = page;
		loadHistory();
	}

	function handleModelFilterChange() {
		currentPage = 0;
		loadHistory();
		loadTrends();
	}

	function clearModelFilter() {
		modelFilter = '';
		handleModelFilterChange();
	}

	// -------------------------------------------------------------------------
	// Trend chart helpers
	// -------------------------------------------------------------------------

	$: maxLatency = trendBuckets.length
		? Math.max(...trendBuckets.map((b) => b.avg_latency_ms), 1)
		: 1;
	$: maxThroughput = trendBuckets.length
		? Math.max(...trendBuckets.map((b) => b.avg_tok_per_sec), 1)
		: 1;
	$: maxEventCount = trendBuckets.length
		? Math.max(...trendBuckets.map((b) => b.event_count), 1)
		: 1;

	function trendBarHeight(value: number, max: number): number {
		return max > 0 ? Math.max(2, (value / max) * 100) : 2;
	}

	function latencyColor(ms: number): string {
		if (ms < 500) return 'var(--oo-success)';
		if (ms < 2000) return 'var(--oo-warning)';
		return 'var(--oo-danger)';
	}

	function changeTrendHours(h: number) {
		trendHours = h;
		loadTrends();
	}

	// -------------------------------------------------------------------------
	// Purge
	// -------------------------------------------------------------------------

	async function handlePurgeByAge() {
		if (purgeDays < 1) return;
		purging = true;
		try {
			await purgeHistory(purgeDays);
			await loadAll();
		} catch (e: any) {
			error = e?.message || 'Purge failed';
		} finally {
			purging = false;
		}
	}

	async function handlePurgeAll() {
		purging = true;
		purgeConfirmOpen = false;
		try {
			await purgeHistory(0);
			await loadAll();
		} catch (e: any) {
			error = e?.message || 'Purge failed';
		} finally {
			purging = false;
		}
	}

	// -------------------------------------------------------------------------
	// Auto-refresh
	// -------------------------------------------------------------------------

	function toggleAutoRefresh() {
		autoRefresh = !autoRefresh;
		if (autoRefresh) {
			refreshTimer = setInterval(loadAll, 10_000);
		} else {
			stopAutoRefresh();
		}
	}

	function stopAutoRefresh() {
		if (refreshTimer) {
			clearInterval(refreshTimer);
			refreshTimer = null;
		}
	}

	// -------------------------------------------------------------------------
	// Formatting helpers
	// -------------------------------------------------------------------------

	function fmtTs(ts: number): string {
		if (!ts) return '—';
		return new Date(ts * 1000).toLocaleString();
	}

	function fmtDuration(ms: number): string {
		if (ms < 1000) return `${Math.round(ms)}ms`;
		return `${(ms / 1000).toFixed(1)}s`;
	}

	function fmtOldest(ts: number): string {
		if (!ts) return 'No data';
		const age = (Date.now() / 1000 - ts) / 86400;
		if (age < 1) return 'Today';
		return `${Math.round(age)}d ago`;
	}
</script>

<div class="history-panel">
	<!-- Header -->
	<div class="panel-header">
		<h2>Telemetry History</h2>
		<div class="header-actions">
			<button class="btn-sm" on:click={loadAll} disabled={loading}>
				{loading ? 'Loading...' : 'Refresh'}
			</button>
			<button class="btn-sm" class:active={autoRefresh} on:click={toggleAutoRefresh}>
				Auto {autoRefresh ? 'ON' : 'OFF'}
			</button>
		</div>
	</div>

	{#if error}
		<div class="error-banner">{error}</div>
	{/if}

	<!-- Stats cards -->
	{#if historyStats}
		<div class="stats-grid">
			<div class="stat-card">
				<div class="stat-value">{historyStats.total_stored.toLocaleString()}</div>
				<div class="stat-label">Stored Events</div>
			</div>
			<div class="stat-card">
				<div class="stat-value">{historyStats.retention_days}d</div>
				<div class="stat-label">Retention</div>
			</div>
			<div class="stat-card">
				<div class="stat-value">{fmtOldest(historyStats.oldest_event_ts)}</div>
				<div class="stat-label">Oldest Event</div>
			</div>
			<div class="stat-card">
				<div class="stat-value">{historyStats.max_events.toLocaleString()}</div>
				<div class="stat-label">Max Capacity</div>
			</div>
		</div>
	{:else if !loading}
		<div class="empty-state">History store unavailable</div>
	{/if}

	<!-- Hourly Trend Chart -->
	<div class="section">
		<div class="section-header">
			<h3>Hourly Trends</h3>
			<div class="trend-range-btns">
				{#each [6, 12, 24, 48, 168] as h}
					<button
						class="btn-xs"
						class:active={trendHours === h}
						on:click={() => changeTrendHours(h)}
					>
						{h < 48 ? `${h}h` : `${h / 24}d`}
					</button>
				{/each}
			</div>
		</div>

		{#if trendBuckets.length === 0}
			<div class="empty-state-sm">No trend data for the selected range</div>
		{:else}
			<!-- Latency bars -->
			<div class="chart-label">Avg Latency</div>
			<div class="trend-chart">
				{#each trendBuckets as bucket}
					<div class="trend-col" title="{bucket.bucket_label}: {Math.round(bucket.avg_latency_ms)}ms avg, {bucket.event_count} events">
						<div
							class="trend-bar"
							style="height: {trendBarHeight(bucket.avg_latency_ms, maxLatency)}%; background: {latencyColor(bucket.avg_latency_ms)}"
						></div>
					</div>
				{/each}
			</div>

			<!-- Throughput bars -->
			<div class="chart-label">Avg tok/s</div>
			<div class="trend-chart">
				{#each trendBuckets as bucket}
					<div class="trend-col" title="{bucket.bucket_label}: {bucket.avg_tok_per_sec.toFixed(1)} tok/s">
						<div
							class="trend-bar bar-throughput"
							style="height: {trendBarHeight(bucket.avg_tok_per_sec, maxThroughput)}%"
						></div>
					</div>
				{/each}
			</div>

			<!-- Event count bars -->
			<div class="chart-label">Events/h</div>
			<div class="trend-chart trend-chart-short">
				{#each trendBuckets as bucket}
					<div class="trend-col" title="{bucket.bucket_label}: {bucket.event_count} events">
						<div
							class="trend-bar bar-count"
							style="height: {trendBarHeight(bucket.event_count, maxEventCount)}%"
						></div>
					</div>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Event Timeline -->
	<div class="section">
		<EventTimeline modelFilter={modelFilter} />
	</div>

	<!-- Per-model Breakdown -->
	{#if modelBreakdown.length > 0}
		<div class="section">
			<h3>Model Breakdown</h3>
			<div class="model-cards">
				{#each modelBreakdown as m}
					<button
						class="model-card"
						class:selected={modelFilter === m.model}
						on:click={() => { modelFilter = m.model; handleModelFilterChange(); }}
					>
						<div class="mc-name" title={m.model}>{m.model}</div>
						<div class="mc-stats">
							<span>{m.event_count} events</span>
							<span>{fmtDuration(m.avg_latency_ms)} avg</span>
							<span>{m.avg_tok_per_sec.toFixed(1)} tok/s</span>
						</div>
					</button>
				{/each}
			</div>
		</div>
	{/if}

	<!-- Event History Table -->
	<div class="section">
		<div class="section-header">
			<h3>Event History ({totalEvents.toLocaleString()} total)</h3>
			<div class="filter-row">
				{#if availableModels.length > 0}
					<select
						class="model-select"
						bind:value={modelFilter}
						on:change={handleModelFilterChange}
					>
						<option value="">All models</option>
						{#each availableModels as m}
							<option value={m}>{m}</option>
						{/each}
					</select>
				{/if}
				{#if modelFilter}
					<button class="btn-xs" on:click={clearModelFilter}>Clear filter</button>
				{/if}
			</div>
		</div>

		{#if events.length === 0}
			<div class="empty-state-sm">No events found</div>
		{:else}
			<div class="table-wrap">
				<table class="event-table">
					<thead>
						<tr>
							<th>Time</th>
							<th>Model</th>
							<th>Latency</th>
							<th>In</th>
							<th>Out</th>
							<th>tok/s</th>
						</tr>
					</thead>
					<tbody>
						{#each events as ev}
							<tr>
								<td class="td-time">{fmtTs(ev.timestamp)}</td>
								<td class="td-model" title={ev.model}>{ev.model}</td>
								<td class="td-num">{fmtDuration(ev.latency_ms)}</td>
								<td class="td-num">{ev.tokens_in}</td>
								<td class="td-num">{ev.tokens_out}</td>
								<td class="td-num">{ev.tok_per_sec.toFixed(1)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>

			<!-- Pagination -->
			<div class="pagination">
				<button class="btn-xs" disabled={currentPage === 0} on:click={() => goToPage(0)}>
					First
				</button>
				<button class="btn-xs" disabled={currentPage === 0} on:click={() => goToPage(currentPage - 1)}>
					Prev
				</button>
				<span class="page-info">
					Page {currentPage + 1} of {totalPages}
				</span>
				<button class="btn-xs" disabled={currentPage >= totalPages - 1} on:click={() => goToPage(currentPage + 1)}>
					Next
				</button>
				<button class="btn-xs" disabled={currentPage >= totalPages - 1} on:click={() => goToPage(totalPages - 1)}>
					Last
				</button>
			</div>
		{/if}
	</div>

	<!-- Retention & Configuration (S115) -->
	<div class="section config-section">
		<h3>Retention & Configuration</h3>

		<!-- Retention slider -->
		<div class="config-row">
			<label class="config-label">
				Retention period:
				<strong>{retentionDays} day{retentionDays !== 1 ? 's' : ''}</strong>
			</label>
			<div class="slider-row">
				<input
					type="range"
					class="retention-slider"
					min="1"
					max="90"
					bind:value={retentionDays}
				/>
				<span class="slider-minmax">1d — 90d</span>
			</div>
		</div>

		<!-- Auto-purge toggle -->
		<div class="config-row">
			<label class="config-label toggle-label">
				<input
					type="checkbox"
					bind:checked={autoPurgeEnabled}
				/>
				Auto-purge (daily cleanup of events older than retention)
			</label>
		</div>

		<!-- Save settings -->
		<div class="config-row">
			<button class="btn-sm" on:click={saveSettings} disabled={savingSettings}>
				{savingSettings ? 'Saving...' : 'Save Settings'}
			</button>
		</div>

		<!-- Storage usage -->
		{#if historyStats}
			<div class="storage-section">
				<div class="storage-header">
					<span class="config-label">Storage usage</span>
					<span class="storage-numbers">
						{historyStats.total_stored.toLocaleString()} / {historyStats.max_events.toLocaleString()} ({storageUsagePct}%)
					</span>
				</div>
				<div class="storage-bar">
					<div
						class="storage-fill"
						class:fill-ok={storageUsagePct < 60}
						class:fill-warn={storageUsagePct >= 60 && storageUsagePct < 85}
						class:fill-danger={storageUsagePct >= 85}
						style="width: {storageUsagePct}%"
					></div>
				</div>
			</div>
		{/if}

		<!-- Export -->
		<div class="config-row">
			<button class="btn-sm" on:click={handleExportCsv}>
				Export to CSV{modelFilter ? ` (${modelFilter})` : ''}
			</button>
		</div>
	</div>

	<!-- Purge Controls -->
	<div class="section purge-section">
		<h3>Data Management</h3>
		<div class="purge-row">
			<label class="purge-label">
				Purge events older than
				<input
					type="number"
					class="purge-input"
					bind:value={purgeDays}
					min="1"
					max="365"
				/>
				days
			</label>
			<button class="btn-sm btn-warn" on:click={handlePurgeByAge} disabled={purging || purgeDays < 1}>
				{purging ? 'Purging...' : 'Purge Old'}
			</button>
		</div>
		<div class="purge-row">
			{#if purgeConfirmOpen}
				<span class="purge-confirm-text">Delete ALL events? This cannot be undone.</span>
				<button class="btn-sm btn-danger" on:click={handlePurgeAll} disabled={purging}>
					Confirm Delete All
				</button>
				<button class="btn-xs" on:click={() => purgeConfirmOpen = false}>Cancel</button>
			{:else}
				<button class="btn-sm btn-danger" on:click={() => purgeConfirmOpen = true} disabled={purging}>
					Purge All Events
				</button>
			{/if}
		</div>
	</div>
</div>

<style>
	.history-panel {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		padding: 1.5rem;
	}

	/* Header */
	.panel-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 0.75rem;
	}

	.panel-header h2 {
		margin: 0;
		font-size: 1.3rem;
		color: var(--oo-text-primary);
	}

	.header-actions {
		display: flex;
		gap: 0.5rem;
	}

	/* Buttons */
	.btn-sm {
		padding: 0.35rem 0.75rem;
		border-radius: 6px;
		font-size: 0.82rem;
		cursor: pointer;
		border: 1px solid var(--oo-bd-subtle);
		background: var(--oo-bg-elevated);
		color: var(--oo-text-primary);
		transition: background 0.15s;
	}

	.btn-sm:hover { background: var(--oo-bg-overlay); }
	.btn-sm:disabled { opacity: 0.5; cursor: not-allowed; }
	.btn-sm.active {
		background: var(--oo-accent-primary);
		color: var(--oo-fg-on-accent);
		border-color: var(--oo-accent-primary);
	}

	.btn-xs {
		padding: 0.2rem 0.5rem;
		border-radius: 4px;
		font-size: 0.75rem;
		cursor: pointer;
		border: 1px solid var(--oo-bd-subtle);
		background: var(--oo-bg-elevated);
		color: var(--oo-text-secondary);
		transition: background 0.15s;
	}

	.btn-xs:hover { background: var(--oo-bg-overlay); }
	.btn-xs:disabled { opacity: 0.4; cursor: not-allowed; }
	.btn-xs.active {
		background: var(--oo-accent-primary);
		color: var(--oo-fg-on-accent);
		border-color: var(--oo-accent-primary);
	}

	.btn-warn {
		background: var(--oo-warning-bg, rgba(200, 170, 50, 0.12));
		border-color: var(--oo-warning-bd, rgba(200, 170, 50, 0.25));
		color: var(--oo-warning);
	}
	.btn-warn:hover { opacity: 0.85; }

	.btn-danger {
		background: var(--oo-danger-bg, rgba(200, 80, 80, 0.12));
		border-color: var(--oo-danger-bd, rgba(200, 80, 80, 0.25));
		color: var(--oo-danger);
	}
	.btn-danger:hover { opacity: 0.85; }

	/* Error / empty */
	.error-banner {
		background: var(--oo-danger-bg, rgba(200, 80, 80, 0.12));
		color: var(--oo-danger);
		border: 1px solid var(--oo-danger-bd, rgba(200, 80, 80, 0.2));
		border-radius: 6px;
		padding: 0.5rem 0.85rem;
		font-size: 0.85rem;
	}

	.empty-state {
		color: var(--oo-text-secondary);
		font-size: 0.9rem;
		padding: 2rem;
		text-align: center;
	}

	.empty-state-sm {
		color: var(--oo-text-tertiary);
		font-size: 0.82rem;
		padding: 1rem;
		text-align: center;
	}

	/* Stats cards */
	.stats-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
		gap: 0.6rem;
	}

	.stat-card {
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-subtle);
		border-radius: 8px;
		padding: 0.85rem;
		text-align: center;
	}

	.stat-value {
		font-size: 1.3rem;
		font-weight: 700;
		color: var(--oo-text-primary);
		font-variant-numeric: tabular-nums;
	}

	.stat-label {
		font-size: 0.72rem;
		color: var(--oo-text-secondary);
		margin-top: 0.15rem;
	}

	/* Sections */
	.section {
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-subtle);
		border-radius: 8px;
		padding: 1rem;
	}

	.section h3 {
		margin: 0 0 0.6rem 0;
		font-size: 0.92rem;
		color: var(--oo-text-primary);
	}

	.section-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 0.5rem;
		margin-bottom: 0.6rem;
	}

	.section-header h3 { margin-bottom: 0; }

	/* Trend chart */
	.trend-range-btns {
		display: flex;
		gap: 0.25rem;
	}

	.chart-label {
		font-size: 0.7rem;
		color: var(--oo-text-tertiary);
		margin-bottom: 0.2rem;
		margin-top: 0.5rem;
	}

	.trend-chart {
		display: flex;
		align-items: flex-end;
		gap: 2px;
		height: 60px;
		padding: 0;
		overflow-x: auto;
	}

	.trend-chart-short { height: 36px; }

	.trend-col {
		flex: 1 1 0;
		min-width: 4px;
		max-width: 18px;
		height: 100%;
		display: flex;
		align-items: flex-end;
		cursor: default;
	}

	.trend-bar {
		width: 100%;
		border-radius: 2px 2px 0 0;
		min-height: 2px;
		transition: height 0.2s ease;
	}

	.bar-throughput {
		background: var(--oo-accent-primary);
	}

	.bar-count {
		background: var(--oo-text-tertiary);
		opacity: 0.5;
	}

	/* Model breakdown cards */
	.model-cards {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
	}

	.model-card {
		background: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-subtle);
		border-radius: 6px;
		padding: 0.6rem 0.8rem;
		cursor: pointer;
		text-align: left;
		font-family: inherit;
		color: inherit;
		transition: border-color 0.15s;
		min-width: 140px;
	}

	.model-card:hover { border-color: var(--oo-accent-primary); }
	.model-card.selected {
		border-color: var(--oo-accent-primary);
		background: var(--oo-accent-bg, rgba(130, 150, 100, 0.08));
	}

	.mc-name {
		font-size: 0.82rem;
		font-weight: 600;
		color: var(--oo-text-primary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		max-width: 180px;
	}

	.mc-stats {
		display: flex;
		flex-direction: column;
		gap: 0.1rem;
		margin-top: 0.3rem;
		font-size: 0.72rem;
		color: var(--oo-text-tertiary);
	}

	/* Filter row */
	.filter-row {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.model-select {
		padding: 0.25rem 0.5rem;
		border-radius: 4px;
		border: 1px solid var(--oo-bd-subtle);
		background: var(--oo-bg-elevated);
		color: var(--oo-text-primary);
		font-size: 0.78rem;
	}

	/* Event table */
	.table-wrap {
		overflow-x: auto;
	}

	.event-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.8rem;
	}

	.event-table th {
		text-align: left;
		padding: 0.4rem 0.5rem;
		border-bottom: 1px solid var(--oo-bd-subtle);
		color: var(--oo-text-secondary);
		font-weight: 600;
		font-size: 0.75rem;
		white-space: nowrap;
	}

	.event-table td {
		padding: 0.35rem 0.5rem;
		border-bottom: 1px solid var(--oo-bg-elevated);
		color: var(--oo-text-primary);
	}

	.event-table tbody tr:hover {
		background: var(--oo-bg-elevated);
	}

	.td-time {
		font-size: 0.75rem;
		color: var(--oo-text-secondary);
		white-space: nowrap;
	}

	.td-model {
		max-width: 180px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.td-num {
		font-variant-numeric: tabular-nums;
		text-align: right;
		white-space: nowrap;
	}

	/* Pagination */
	.pagination {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.5rem;
		padding-top: 0.6rem;
	}

	.page-info {
		font-size: 0.78rem;
		color: var(--oo-text-secondary);
	}

	/* Purge section */
	.purge-section {
		border-color: var(--oo-warning-bd, rgba(200, 170, 50, 0.25));
	}

	.purge-row {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		flex-wrap: wrap;
		margin-bottom: 0.5rem;
	}

	.purge-row:last-child { margin-bottom: 0; }

	.purge-label {
		font-size: 0.82rem;
		color: var(--oo-text-secondary);
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}

	.purge-input {
		width: 60px;
		padding: 0.2rem 0.4rem;
		border-radius: 4px;
		border: 1px solid var(--oo-bd-subtle);
		background: var(--oo-bg-elevated);
		color: var(--oo-text-primary);
		font-size: 0.82rem;
		text-align: center;
	}

	/* Config section (S115) */
	.config-section {
		border-color: var(--oo-accent-primary);
	}

	.config-row {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		flex-wrap: wrap;
		margin-bottom: 0.6rem;
	}

	.config-row:last-child { margin-bottom: 0; }

	.config-label {
		font-size: 0.82rem;
		color: var(--oo-text-secondary);
	}

	.toggle-label {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
		cursor: pointer;
	}

	.toggle-label input[type="checkbox"] {
		accent-color: var(--oo-accent-primary);
	}

	.slider-row {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex: 1;
		min-width: 180px;
	}

	.retention-slider {
		flex: 1;
		accent-color: var(--oo-accent-primary);
		height: 4px;
	}

	.slider-minmax {
		font-size: 0.68rem;
		color: var(--oo-text-tertiary);
		white-space: nowrap;
	}

	/* Storage usage */
	.storage-section {
		margin-bottom: 0.6rem;
	}

	.storage-header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		margin-bottom: 0.3rem;
	}

	.storage-numbers {
		font-size: 0.75rem;
		color: var(--oo-text-secondary);
		font-variant-numeric: tabular-nums;
	}

	.storage-bar {
		height: 8px;
		background: var(--oo-bg-elevated);
		border-radius: 4px;
		overflow: hidden;
	}

	.storage-fill {
		height: 100%;
		border-radius: 4px;
		transition: width 0.3s ease;
	}

	.storage-fill.fill-ok { background: var(--oo-success); }
	.storage-fill.fill-warn { background: var(--oo-warning); }
	.storage-fill.fill-danger { background: var(--oo-danger); }

	.purge-confirm-text {
		font-size: 0.82rem;
		color: var(--oo-danger);
		font-weight: 600;
	}
</style>
