<!--
  PerformanceDashboard.svelte -- S72 Real-Time Performance Dashboard.

  Sections:
  1. Throughput overview (tokens/sec, request count)
  2. Latency distribution (p50/p95/p99 per model, bar chart)
  3. Model utilization (donut-style horizontal bars)
  4. Drift alerts (color-coded warnings)
  5. Recommendation cards
  6. Auto-refresh toggle
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		getPerformanceSummary,
		getLatencyStats,
		getDriftResults,
		getRecommendations,
		getUtilization,
		getPerformanceHistory,
		cleanupMetrics,
	} from '$lib/api/performance';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type {
		PerformanceSummary,
		PerformanceLatencyStats,
		PerformanceDriftEntry,
		PerformanceRecommendation,
		PerformanceHistoryRecord,
	} from '$lib/types';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	let loading = true;
	let error = '';

	let summary: PerformanceSummary | null = null;
	let drifts: PerformanceDriftEntry[] = [];
	let recommendations: PerformanceRecommendation[] = [];
	let utilization: Record<string, number> = {};
	let modelLatencies: Map<string, PerformanceLatencyStats> = new Map();

	// Auto-refresh
	let autoRefresh = false;
	let refreshInterval: ReturnType<typeof setInterval> | null = null;
	let refreshSeconds = 10;
	let lastRefresh = '';

	// -------------------------------------------------------------------------
	// Load
	// -------------------------------------------------------------------------

	onMount(loadAll);
	onDestroy(stopAutoRefresh);

	async function loadAll() {
		loading = true;
		error = '';
		try {
			const [summaryData, driftData, recData] = await Promise.all([
				getPerformanceSummary(),
				getDriftResults(),
				getRecommendations(),
			]);

			summary = summaryData;
			drifts = driftData.drifts || [];
			recommendations = recData.recommendations || [];
			utilization = summaryData.utilization || {};

			// Load per-model latency stats
			modelLatencies = new Map();
			const models = Object.keys(utilization);
			for (const model of models.slice(0, 10)) {
				try {
					const stats = await getLatencyStats(model);
					if (stats.count > 0) {
						modelLatencies.set(model, stats);
					}
				} catch {
					// Skip models with errors
				}
			}
			// Trigger reactivity
			modelLatencies = modelLatencies;

			lastRefresh = new Date().toLocaleTimeString();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load performance data';
		} finally {
			loading = false;
		}
	}

	function toggleAutoRefresh() {
		autoRefresh = !autoRefresh;
		if (autoRefresh) {
			startAutoRefresh();
		} else {
			stopAutoRefresh();
		}
	}

	function startAutoRefresh() {
		stopAutoRefresh();
		refreshInterval = setInterval(loadAll, refreshSeconds * 1000);
	}

	function stopAutoRefresh() {
		if (refreshInterval) {
			clearInterval(refreshInterval);
			refreshInterval = null;
		}
	}

	async function handleCleanup() {
		try {
			const result = await cleanupMetrics();
			toastSuccess(`Cleaned up ${result.deleted} old records`);
			await loadAll();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Cleanup failed');
		}
	}

	// -------------------------------------------------------------------------
	// Helpers
	// -------------------------------------------------------------------------

	function formatMs(ms: number): string {
		if (ms < 1000) return `${ms.toFixed(0)}ms`;
		return `${(ms / 1000).toFixed(1)}s`;
	}

	function formatTps(tps: number): string {
		if (tps < 1) return tps.toFixed(2);
		if (tps < 100) return tps.toFixed(1);
		return tps.toFixed(0);
	}

	function severityColor(severity: string): string {
		switch (severity) {
			case 'critical': return 'var(--oo-error)';
			case 'warning': return 'var(--oo-warning)';
			default: return 'var(--oo-info)';
		}
	}

	function driftColor(drift: PerformanceDriftEntry): string {
		const ratio = Math.abs(drift.change_ratio);
		if (ratio > 0.5) return 'var(--oo-error)';
		if (ratio > 0.3) return 'var(--oo-warning)';
		return 'var(--oo-info)';
	}

	function utilizationWidth(fraction: number): string {
		return `${Math.max(2, fraction * 100)}%`;
	}

	function latencyBarWidth(ms: number, maxMs: number): string {
		if (maxMs === 0) return '0%';
		return `${Math.max(2, (ms / maxMs) * 100)}%`;
	}

	$: maxLatency = Math.max(
		...[...modelLatencies.values()].map(s => s.p99),
		1
	);
</script>

<div class="space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between flex-wrap gap-3">
		<div>
			<h2 class="text-base font-medium" style="color: var(--oo-fg-primary);">
				Performance Dashboard
			</h2>
			<p class="text-xs mt-0.5" style="color: var(--oo-fg-tertiary);">
				Real-time metrics, drift detection, and optimization recommendations
			</p>
		</div>
		<div class="flex items-center gap-2">
			{#if lastRefresh}
				<span class="text-xs" style="color: var(--oo-fg-tertiary);">
					Last: {lastRefresh}
				</span>
			{/if}
			<button
				on:click={loadAll}
				disabled={loading}
				class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
				style="background-color: var(--oo-btn-secondary-bg);
					   color: var(--oo-btn-secondary-fg);"
			>
				{loading ? 'Loading...' : 'Refresh'}
			</button>
			<button
				on:click={toggleAutoRefresh}
				class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
				style="background-color: {autoRefresh ? 'var(--oo-acc-500)' : 'var(--oo-btn-secondary-bg)'};
					   color: {autoRefresh ? 'var(--oo-btn-primary-fg)' : 'var(--oo-btn-secondary-fg)'};"
			>
				Auto {autoRefresh ? 'ON' : 'OFF'}
			</button>
		</div>
	</div>

	{#if loading && !summary}
		<div class="flex items-center gap-2 py-8 justify-center text-sm"
			style="color: var(--oo-fg-tertiary);">
			<div class="w-5 h-5 border-2 rounded-full animate-spin"
				style="border-color: var(--oo-bd-default); border-top-color: var(--oo-acc-500);" />
			Loading performance data...
		</div>
	{:else if error}
		<div class="px-3 py-2 rounded text-sm"
			style="background-color: var(--oo-error-bg);
				   border: 1px solid var(--oo-error-bd);
				   color: var(--oo-error);">
			{error}
			<button on:click={loadAll} class="ml-2 underline hover:no-underline">Retry</button>
		</div>
	{:else if summary}
		<!-- Throughput Overview Cards -->
		<div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated);
				border: 1px solid var(--oo-bd-default);">
				<div class="text-xs font-medium mb-1" style="color: var(--oo-fg-tertiary);">
					Tokens In/s
				</div>
				<div class="text-lg font-semibold" style="color: var(--oo-acc-400);">
					{formatTps(summary.throughput.tokens_in_per_sec)}
				</div>
			</div>
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated);
				border: 1px solid var(--oo-bd-default);">
				<div class="text-xs font-medium mb-1" style="color: var(--oo-fg-tertiary);">
					Tokens Out/s
				</div>
				<div class="text-lg font-semibold" style="color: var(--oo-acc-400);">
					{formatTps(summary.throughput.tokens_out_per_sec)}
				</div>
			</div>
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated);
				border: 1px solid var(--oo-bd-default);">
				<div class="text-xs font-medium mb-1" style="color: var(--oo-fg-tertiary);">
					Requests
				</div>
				<div class="text-lg font-semibold" style="color: var(--oo-fg-primary);">
					{summary.throughput.request_count}
				</div>
			</div>
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated);
				border: 1px solid var(--oo-bd-default);">
				<div class="text-xs font-medium mb-1" style="color: var(--oo-fg-tertiary);">
					Avg Latency
				</div>
				<div class="text-lg font-semibold" style="color: var(--oo-fg-primary);">
					{formatMs(summary.latency.mean)}
				</div>
			</div>
		</div>

		<!-- Latency Distribution -->
		{#if modelLatencies.size > 0}
			<div class="p-4 rounded-lg" style="background-color: var(--oo-bg-elevated);
				border: 1px solid var(--oo-bd-default);">
				<h3 class="text-sm font-medium mb-3" style="color: var(--oo-fg-primary);">
					Latency by Model
				</h3>
				<div class="space-y-2">
					{#each [...modelLatencies.entries()] as [model, stats]}
						<div class="flex items-center gap-2">
							<span class="text-xs w-32 truncate shrink-0" style="color: var(--oo-fg-secondary);"
								title={model}>
								{model}
							</span>
							<div class="flex-1 flex items-center gap-1 h-5">
								<!-- p50 bar -->
								<div class="h-3 rounded-sm"
									style="width: {latencyBarWidth(stats.p50, maxLatency)};
										   background-color: var(--oo-acc-400);
										   opacity: 0.6;"
									title="p50: {formatMs(stats.p50)}" />
								<!-- p95 extension -->
								<div class="h-3 rounded-sm"
									style="width: {latencyBarWidth(stats.p95 - stats.p50, maxLatency)};
										   background-color: var(--oo-warning);
										   opacity: 0.4;"
									title="p95: {formatMs(stats.p95)}" />
								<!-- p99 extension -->
								<div class="h-3 rounded-sm"
									style="width: {latencyBarWidth(stats.p99 - stats.p95, maxLatency)};
										   background-color: var(--oo-error);
										   opacity: 0.3;"
									title="p99: {formatMs(stats.p99)}" />
							</div>
							<span class="text-xs w-16 text-right shrink-0" style="color: var(--oo-fg-tertiary);">
								{formatMs(stats.p50)}
							</span>
						</div>
					{/each}
				</div>
				<div class="flex gap-4 mt-2 text-xs" style="color: var(--oo-fg-tertiary);">
					<span class="flex items-center gap-1">
						<span class="w-3 h-2 rounded-sm inline-block" style="background-color: var(--oo-acc-400); opacity: 0.6;" />
						p50
					</span>
					<span class="flex items-center gap-1">
						<span class="w-3 h-2 rounded-sm inline-block" style="background-color: var(--oo-warning); opacity: 0.4;" />
						p95
					</span>
					<span class="flex items-center gap-1">
						<span class="w-3 h-2 rounded-sm inline-block" style="background-color: var(--oo-error); opacity: 0.3;" />
						p99
					</span>
				</div>
			</div>
		{/if}

		<!-- Model Utilization -->
		{#if Object.keys(utilization).length > 0}
			<div class="p-4 rounded-lg" style="background-color: var(--oo-bg-elevated);
				border: 1px solid var(--oo-bd-default);">
				<h3 class="text-sm font-medium mb-3" style="color: var(--oo-fg-primary);">
					Model Utilization
				</h3>
				<div class="space-y-2">
					{#each Object.entries(utilization).sort((a, b) => b[1] - a[1]) as [model, fraction]}
						<div class="flex items-center gap-2">
							<span class="text-xs w-32 truncate shrink-0" style="color: var(--oo-fg-secondary);"
								title={model}>
								{model}
							</span>
							<div class="flex-1 h-4 rounded-full overflow-hidden"
								style="background-color: var(--oo-bg-surface);">
								<div class="h-full rounded-full transition-all duration-300"
									style="width: {utilizationWidth(fraction)};
										   background-color: var(--oo-acc-500);" />
							</div>
							<span class="text-xs w-12 text-right shrink-0" style="color: var(--oo-fg-tertiary);">
								{(fraction * 100).toFixed(0)}%
							</span>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Drift Alerts -->
		{#if drifts.length > 0}
			<div class="p-4 rounded-lg" style="background-color: var(--oo-bg-elevated);
				border: 1px solid var(--oo-bd-default);">
				<h3 class="text-sm font-medium mb-3" style="color: var(--oo-fg-primary);">
					Drift Alerts
				</h3>
				<div class="space-y-2">
					{#each drifts as drift}
						<div class="flex items-start gap-2 p-2 rounded text-sm"
							style="background-color: var(--oo-bg-overlay);
								   border-left: 3px solid {driftColor(drift)};">
							<span class="text-xs font-mono shrink-0 mt-0.5"
								style="color: {driftColor(drift)};">
								{drift.direction === 'up' ? '+' : ''}{(drift.change_ratio * 100).toFixed(0)}%
							</span>
							<div class="flex-1">
								<span class="text-xs font-medium" style="color: var(--oo-fg-secondary);">
									{drift.model}
								</span>
								<span class="text-xs" style="color: var(--oo-fg-tertiary);">
									{drift.metric} {drift.direction === 'up' ? 'increased' : 'decreased'}
									(baseline: {drift.metric === 'latency' ? formatMs(drift.baseline_value) : drift.baseline_value.toFixed(2)}
									 &rarr;
									 {drift.metric === 'latency' ? formatMs(drift.recent_value) : drift.recent_value.toFixed(2)})
								</span>
							</div>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Recommendations -->
		{#if recommendations.length > 0}
			<div class="p-4 rounded-lg" style="background-color: var(--oo-bg-elevated);
				border: 1px solid var(--oo-bd-default);">
				<h3 class="text-sm font-medium mb-3" style="color: var(--oo-fg-primary);">
					Recommendations
				</h3>
				<div class="space-y-2">
					{#each recommendations as rec}
						<div class="p-2 rounded text-sm"
							style="background-color: var(--oo-bg-overlay);
								   border-left: 3px solid {severityColor(rec.severity)};">
							<div class="flex items-center gap-2 mb-0.5">
								<span class="text-xs font-medium px-1.5 py-0.5 rounded"
									style="background-color: {severityColor(rec.severity)}20;
										   color: {severityColor(rec.severity)};">
									{rec.severity}
								</span>
								<span class="text-xs font-medium" style="color: var(--oo-fg-secondary);">
									{rec.model}
								</span>
							</div>
							<p class="text-xs" style="color: var(--oo-fg-tertiary);">
								{rec.message}
							</p>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Empty State -->
		{#if summary.throughput.request_count === 0}
			<div class="text-center py-8">
				<p class="text-sm" style="color: var(--oo-fg-tertiary);">
					No performance data yet. Metrics will appear after LLM calls are made.
				</p>
			</div>
		{/if}

		<!-- Cleanup -->
		<div class="flex justify-end">
			<button
				on:click={handleCleanup}
				class="px-3 py-1.5 rounded text-xs transition-colors"
				style="background-color: var(--oo-btn-secondary-bg);
					   color: var(--oo-btn-secondary-fg);"
			>
				Cleanup Old Records
			</button>
		</div>
	{/if}
</div>
