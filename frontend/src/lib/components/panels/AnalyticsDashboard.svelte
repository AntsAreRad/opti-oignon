<!--
  AnalyticsDashboard.svelte (S55)
  Analytics overview panel displaying:
  - Summary stats (requests, success rate, avg response time, tokens/s)
  - Feedback stats (thumbs up/down, average score)
  - Model usage distribution (horizontal bar chart)
  - Pipeline usage distribution (horizontal bar chart)
  - Per-model performance table
  - Response time trend (mini bar chart)
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getAnalyticsOverview,
		getAnalyticsTrends,
		getFeedbackStats,
		getRoutingAccuracy,
		cleanupAnalytics,
	} from '$lib/api/feedback';
	import type {
		AnalyticsOverviewInfo,
		TrendsInfo,
		FeedbackStatsInfo,
		RoutingAccuracyInfo,
	} from '$lib/types';

	let overview: AnalyticsOverviewInfo | null = null;
	let trends: TrendsInfo | null = null;
	let feedbackStats: FeedbackStatsInfo | null = null;
	let routingAccuracy: RoutingAccuracyInfo | null = null;
	let loading = true;
	let error = '';

	// Time window for trends
	let trendWindow = '24h';
	const windowOptions = [
		{ value: '1h', label: '1 hour' },
		{ value: '24h', label: '24 hours' },
		{ value: '7d', label: '7 days' },
		{ value: '30d', label: '30 days' },
	];

	async function loadData() {
		loading = true;
		error = '';
		try {
			const [ov, tr, fb, ra] = await Promise.all([
				getAnalyticsOverview(),
				getAnalyticsTrends(trendWindow, 12),
				getFeedbackStats(),
				getRoutingAccuracy(),
			]);
			overview = ov;
			trends = tr;
			feedbackStats = fb;
			routingAccuracy = ra;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load analytics';
		} finally {
			loading = false;
		}
	}

	async function handleWindowChange() {
		try {
			trends = await getAnalyticsTrends(trendWindow, 12);
		} catch (e) {
			// keep old data
		}
	}

	function formatMs(ms: number): string {
		if (ms < 1000) return `${Math.round(ms)}ms`;
		return `${(ms / 1000).toFixed(1)}s`;
	}

	function formatNumber(n: number): string {
		if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
		if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
		return String(n);
	}

	function barWidth(value: number, max: number): number {
		if (max <= 0) return 0;
		return Math.max(2, Math.round((value / max) * 100));
	}

	// Trend bar height as percentage of max count
	function trendBarHeight(count: number, maxCount: number): number {
		if (maxCount <= 0) return 0;
		return Math.max(4, Math.round((count / maxCount) * 100));
	}

	onMount(loadData);
</script>

<div class="space-y-5">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<h2 class="text-base font-medium" style="color: var(--oo-fg-primary);">
			Analytics & Feedback
		</h2>
		<button
			on:click={loadData}
			class="px-2.5 py-1 text-xs rounded transition-colors"
			style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-secondary);
				border: 1px solid var(--oo-bd-default);"
		>
			Refresh
		</button>
	</div>

	{#if loading}
		<div class="flex items-center gap-2 text-sm py-8 justify-center" style="color: var(--oo-fg-muted);">
			<div class="w-5 h-5 border-2 rounded-full animate-spin"
				style="border-color: var(--oo-bd-default); border-top-color: var(--oo-acc-500);" />
			Loading analytics...
		</div>
	{:else if error}
		<div class="px-3 py-2 rounded text-sm" style="background-color: var(--oo-error-bg); border: 1px solid var(--oo-error-bd); color: var(--oo-error);">
			{error}
			<button on:click={loadData} class="ml-2 underline hover:no-underline">Retry</button>
		</div>
	{:else if overview}
		<!-- Summary cards -->
		<div class="grid grid-cols-2 gap-3">
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
				<div class="text-xs" style="color: var(--oo-fg-muted);">Total Requests</div>
				<div class="text-xl font-semibold mt-0.5" style="color: var(--oo-fg-primary);">
					{formatNumber(overview.total_requests)}
				</div>
			</div>
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
				<div class="text-xs" style="color: var(--oo-fg-muted);">Success Rate</div>
				<div class="text-xl font-semibold mt-0.5" style="color: {overview.success_rate >= 0.95 ? 'var(--oo-success)' : overview.success_rate >= 0.8 ? 'var(--oo-warning)' : 'var(--oo-error)'};">
					{(overview.success_rate * 100).toFixed(1)}%
				</div>
			</div>
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
				<div class="text-xs" style="color: var(--oo-fg-muted);">Avg Response Time</div>
				<div class="text-xl font-semibold mt-0.5" style="color: var(--oo-fg-primary);">
					{formatMs(overview.avg_response_time_ms)}
				</div>
			</div>
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
				<div class="text-xs" style="color: var(--oo-fg-muted);">Avg Tokens/s</div>
				<div class="text-xl font-semibold mt-0.5" style="color: var(--oo-fg-primary);">
					{overview.avg_tokens_per_second.toFixed(1)}
				</div>
			</div>
		</div>

		<!-- Feedback summary -->
		{#if feedbackStats && feedbackStats.total_count > 0}
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
				<div class="text-xs font-medium mb-2" style="color: var(--oo-fg-secondary);">Feedback</div>
				<div class="flex items-center gap-4 text-sm">
					<div class="flex items-center gap-1.5">
						<svg class="w-4 h-4" style="color: var(--oo-success);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
							<path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3H14zM7 22H4a2 2 0 01-2-2v-7a2 2 0 012-2h3" />
						</svg>
						<span style="color: var(--oo-fg-primary);">{feedbackStats.thumbs_up}</span>
					</div>
					<div class="flex items-center gap-1.5">
						<svg class="w-4 h-4" style="color: var(--oo-error);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
							<path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3H10zM17 2h2.67A2.31 2.31 0 0122 4v7a2.31 2.31 0 01-2.33 2H17" />
						</svg>
						<span style="color: var(--oo-fg-primary);">{feedbackStats.thumbs_down}</span>
					</div>
					<div class="text-xs ml-auto" style="color: var(--oo-fg-muted);">
						Score: {(feedbackStats.average_score * 100).toFixed(0)}%
					</div>
				</div>

				<!-- Per-model feedback breakdown -->
				{#if Object.keys(feedbackStats.by_model).length > 0}
					<div class="mt-3 space-y-1.5">
						<div class="text-xs" style="color: var(--oo-fg-muted);">By Model</div>
						{#each Object.entries(feedbackStats.by_model) as [model, stats]}
							<div class="flex items-center justify-between text-xs">
								<span class="font-mono truncate max-w-[60%]" style="color: var(--oo-fg-secondary);">{model}</span>
								<span style="color: var(--oo-fg-muted);">
									{stats.thumbs_up} up / {stats.thumbs_down} down
								</span>
							</div>
						{/each}
					</div>
				{/if}
			</div>
		{/if}

		<!-- Response time trend -->
		{#if trends && trends.data.length > 0}
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
				<div class="flex items-center justify-between mb-2">
					<div class="text-xs font-medium" style="color: var(--oo-fg-secondary);">Request Trend</div>
					<select
						bind:value={trendWindow}
						on:change={handleWindowChange}
						class="text-xs px-1.5 py-0.5 rounded"
						style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd);
							color: var(--oo-fg-secondary);"
					>
						{#each windowOptions as opt}
							<option value={opt.value}>{opt.label}</option>
						{/each}
					</select>
				</div>
				<!-- Mini bar chart -->
				{#each [Math.max(...trends.data.map(d => d.count), 1)] as maxCount}
					<div class="flex items-end gap-0.5 h-16">
						{#each trends.data as point}
							<div
								class="flex-1 rounded-t-sm transition-all"
								style="height: {trendBarHeight(point.count, maxCount)}%;
									background-color: {point.count > 0 ? 'var(--oo-acc-500)' : 'var(--oo-bd-default)'};"
								title="{point.count} requests, avg {formatMs(point.avg_response_time_ms)}"
							/>
						{/each}
					</div>
				{/each}
				<div class="text-xs mt-1 text-center" style="color: var(--oo-fg-faint);">
					{trends.data.reduce((s, d) => s + d.count, 0)} total requests in {trendWindow}
				</div>
			</div>
		{/if}

		<!-- Model distribution -->
		{#if Object.keys(overview.model_distribution).length > 0}
			{@const maxModelCount = Math.max(...Object.values(overview.model_distribution))}
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
				<div class="text-xs font-medium mb-2" style="color: var(--oo-fg-secondary);">Model Usage</div>
				<div class="space-y-1.5">
					{#each Object.entries(overview.model_distribution) as [model, count]}
						<div>
							<div class="flex justify-between text-xs mb-0.5">
								<span class="font-mono truncate max-w-[70%]" style="color: var(--oo-fg-secondary);">{model}</span>
								<span style="color: var(--oo-fg-muted);">{count}</span>
							</div>
							<div class="h-1.5 rounded-full overflow-hidden" style="background-color: var(--oo-bg-base);">
								<div
									class="h-full rounded-full transition-all"
									style="width: {barWidth(count, maxModelCount)}%; background-color: var(--oo-acc-500);"
								/>
							</div>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Pipeline distribution -->
		{#if Object.keys(overview.pipeline_distribution).length > 0}
			{@const maxPipeCount = Math.max(...Object.values(overview.pipeline_distribution))}
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
				<div class="text-xs font-medium mb-2" style="color: var(--oo-fg-secondary);">Pipeline Usage</div>
				<div class="space-y-1.5">
					{#each Object.entries(overview.pipeline_distribution) as [pipeline, count]}
						<div>
							<div class="flex justify-between text-xs mb-0.5">
								<span style="color: var(--oo-fg-secondary);">{pipeline}</span>
								<span style="color: var(--oo-fg-muted);">{count}</span>
							</div>
							<div class="h-1.5 rounded-full overflow-hidden" style="background-color: var(--oo-bg-base);">
								<div
									class="h-full rounded-full transition-all"
									style="width: {barWidth(count, maxPipeCount)}%; background-color: var(--oo-pipe-consensus);"
								/>
							</div>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Per-model performance table -->
		{#if Object.keys(overview.model_performance).length > 0}
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
				<div class="text-xs font-medium mb-2" style="color: var(--oo-fg-secondary);">Model Performance</div>
				<div class="overflow-x-auto">
					<table class="w-full text-xs">
						<thead>
							<tr style="color: var(--oo-fg-muted);">
								<th class="text-left py-1 pr-2">Model</th>
								<th class="text-right py-1 px-1">Reqs</th>
								<th class="text-right py-1 px-1">Avg RT</th>
								<th class="text-right py-1 px-1">Tok/s</th>
								<th class="text-right py-1 pl-1">Success</th>
							</tr>
						</thead>
						<tbody>
							{#each Object.entries(overview.model_performance) as [model, perf]}
								<tr style="border-top: 1px solid var(--oo-bd-default);">
									<td class="py-1 pr-2 font-mono truncate max-w-[120px]" style="color: var(--oo-fg-secondary);">{model}</td>
									<td class="py-1 px-1 text-right" style="color: var(--oo-fg-primary);">{perf.count}</td>
									<td class="py-1 px-1 text-right" style="color: var(--oo-fg-primary);">{formatMs(perf.avg_response_time_ms)}</td>
									<td class="py-1 px-1 text-right" style="color: var(--oo-fg-primary);">{perf.avg_tokens_per_second?.toFixed(1) ?? '-'}</td>
									<td class="py-1 pl-1 text-right" style="color: {perf.success_rate >= 0.95 ? 'var(--oo-success)' : 'var(--oo-warning)'};">
										{(perf.success_rate * 100).toFixed(0)}%
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		{/if}

		<!-- Routing accuracy -->
		{#if routingAccuracy && (routingAccuracy.routed.count > 0 || routingAccuracy.unrouted.count > 0)}
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
				<div class="text-xs font-medium mb-2" style="color: var(--oo-fg-secondary);">Routing Accuracy</div>
				<div class="grid grid-cols-2 gap-3 text-xs">
					<div>
						<div style="color: var(--oo-fg-muted);">Smart Routed</div>
						<div style="color: var(--oo-fg-primary);">{routingAccuracy.routed.count} requests</div>
						{#if routingAccuracy.routed.count > 0}
							<div style="color: var(--oo-fg-muted);">
								{formatMs(routingAccuracy.routed.avg_response_time_ms)} avg,
								{(routingAccuracy.routed.success_rate * 100).toFixed(0)}% success
							</div>
						{/if}
					</div>
					<div>
						<div style="color: var(--oo-fg-muted);">Default Routing</div>
						<div style="color: var(--oo-fg-primary);">{routingAccuracy.unrouted.count} requests</div>
						{#if routingAccuracy.unrouted.count > 0}
							<div style="color: var(--oo-fg-muted);">
								{formatMs(routingAccuracy.unrouted.avg_response_time_ms)} avg,
								{(routingAccuracy.unrouted.success_rate * 100).toFixed(0)}% success
							</div>
						{/if}
					</div>
				</div>
			</div>
		{/if}

		<!-- Empty state -->
		{#if overview.total_requests === 0 && (!feedbackStats || feedbackStats.total_count === 0)}
			<div class="text-center py-6 text-sm" style="color: var(--oo-fg-muted);">
				No data yet. Analytics will appear as you use the chat.
			</div>
		{/if}
	{/if}
</div>
