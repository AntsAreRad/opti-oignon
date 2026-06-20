<!--
  RAGDashboardPanel.svelte -- S100 Knowledge Base Dashboard.

  Sub-sections:
  1. Overview: key metrics cards
  2. Usage chart: queries/citations per day (sparkline bars)
  3. Top cited sources
  4. Collection health
  5. External connector status
  6. Auto-refresh controls
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getDashboardStats,
		getUsageOverTime,
		getSourceReliability,
		getCollectionHealth,
		triggerRefresh,
		getConnectors,
		getBackends,
	} from '$lib/api/ragDashboard';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type {
		RAGDashboardStats,
		RAGUsageDataPoint,
		RAGSourceReliability,
		RAGCollectionHealth,
		RAGConnectorStatus,
	} from '$lib/types';

	let loading = true;
	let stats: RAGDashboardStats | null = null;
	let usageData: RAGUsageDataPoint[] = [];
	let sources: RAGSourceReliability[] = [];
	let health: RAGCollectionHealth[] = [];
	let connectors: RAGConnectorStatus[] = [];
	let backends: Record<string, boolean> = {};
	let refreshing = false;

	async function loadAll() {
		loading = true;
		try {
			const [statsRes, usageRes, srcRes, healthRes, connRes, backRes] =
				await Promise.allSettled([
					getDashboardStats(),
					getUsageOverTime(30),
					getSourceReliability(10),
					getCollectionHealth(),
					getConnectors(),
					getBackends(),
				]);

			if (statsRes.status === 'fulfilled') stats = statsRes.value;
			if (usageRes.status === 'fulfilled') usageData = usageRes.value.data;
			if (srcRes.status === 'fulfilled') sources = srcRes.value.sources;
			if (healthRes.status === 'fulfilled') health = healthRes.value.collections;
			if (connRes.status === 'fulfilled') connectors = connRes.value.connectors;
			if (backRes.status === 'fulfilled') backends = backRes.value.backends;
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to load dashboard');
		} finally {
			loading = false;
		}
	}

	async function handleRefresh() {
		refreshing = true;
		try {
			const result = await triggerRefresh();
			if (result.sources_refreshed > 0) {
				toastSuccess(
					`Refreshed ${result.sources_refreshed} of ${result.sources_checked} sources`
				);
			} else {
				toastSuccess(`Checked ${result.sources_checked} sources, all up to date`);
			}
			if (result.errors.length > 0) {
				toastError(`${result.errors.length} refresh errors`);
			}
			await loadAll();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Refresh failed');
		} finally {
			refreshing = false;
		}
	}

	function formatBytes(bytes: number): string {
		if (bytes === 0) return '0 B';
		const units = ['B', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(1024));
		return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
	}

	function formatTime(ts: number): string {
		if (!ts) return '-';
		return new Date(ts * 1000).toLocaleDateString();
	}

	function shortName(path: string): string {
		const parts = path.split('/');
		return parts[parts.length - 1] || path;
	}

	function scoreColor(score: number): string {
		if (score >= 0.7) return 'var(--oo-success)';
		if (score >= 0.4) return 'var(--oo-warning)';
		return 'var(--oo-fg-muted)';
	}

	function barHeight(value: number, max: number): number {
		if (max === 0) return 0;
		return Math.max(2, Math.round((value / max) * 48));
	}

	// Reactive sparkline maximum (BUG-01: was @const in template, invalid outside Svelte block)
	$: maxQueries = usageData.length > 0 ? Math.max(...usageData.map((d) => d.query_count), 1) : 1;

	onMount(loadAll);
</script>

{#if loading}
	<p class="text-sm py-4" style="color: var(--oo-fg-muted);">Loading dashboard...</p>
{:else}
	<div class="space-y-5">

		<!-- ==================== OVERVIEW CARDS ==================== -->
		{#if stats}
			<div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
				{#each [
					{ label: 'Collections', value: stats.total_collections },
					{ label: 'Documents', value: stats.total_documents },
					{ label: 'Chunks', value: stats.total_chunks },
					{ label: 'Citations', value: stats.total_citations },
				] as card}
					<div
						class="rounded-xl p-3 text-center"
						style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);"
					>
						<div
							class="text-xl font-semibold"
							style="color: var(--oo-fg-primary);"
						>
							{card.value.toLocaleString()}
						</div>
						<div class="text-xs mt-0.5" style="color: var(--oo-fg-muted);">
							{card.label}
						</div>
					</div>
				{/each}
			</div>

			<!-- Second row: queries + storage -->
			<div class="grid grid-cols-3 gap-3">
				<div
					class="rounded-xl p-3 text-center"
					style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);"
				>
					<div class="text-lg font-semibold" style="color: var(--oo-fg-primary);">
						{stats.total_queries_today}
					</div>
					<div class="text-xs" style="color: var(--oo-fg-muted);">Queries today</div>
				</div>
				<div
					class="rounded-xl p-3 text-center"
					style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);"
				>
					<div class="text-lg font-semibold" style="color: var(--oo-fg-primary);">
						{stats.total_queries_week}
					</div>
					<div class="text-xs" style="color: var(--oo-fg-muted);">This week</div>
				</div>
				<div
					class="rounded-xl p-3 text-center"
					style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);"
				>
					<div class="text-lg font-semibold" style="color: var(--oo-fg-primary);">
						{formatBytes(stats.storage_bytes)}
					</div>
					<div class="text-xs" style="color: var(--oo-fg-muted);">Storage</div>
				</div>
			</div>
		{/if}

		<!-- ==================== USAGE CHART ==================== -->
		{#if usageData.length > 0}
			<div
				class="rounded-xl p-4"
				style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);"
			>
				<div class="flex items-center justify-between mb-3">
					<h3 class="text-sm font-medium" style="color: var(--oo-fg-primary);">
						Query Activity (30 days)
					</h3>
					<span class="text-xs" style="color: var(--oo-fg-muted);">
						{stats ? stats.total_queries_all : 0} total queries
					</span>
				</div>

				<!-- Sparkline bar chart -->
				<div class="flex items-end gap-px" style="height: 52px;">
					{#each usageData as point}
						{@const h = barHeight(point.query_count, maxQueries)}
						<div
							class="flex-1 rounded-t-sm transition-all"
							style="height: {h}px; background-color: {point.query_count > 0
								? 'var(--oo-acc-600)'
								: 'var(--oo-bd-subtle)'}; min-width: 2px; opacity: {point.query_count > 0 ? 0.8 : 0.3};"
							title="{point.date}: {point.query_count} queries"
						/>
					{/each}
				</div>
				<div class="flex justify-between mt-1">
					<span class="text-xs" style="color: var(--oo-fg-muted);">
						{usageData[0]?.date ?? ''}
					</span>
					<span class="text-xs" style="color: var(--oo-fg-muted);">
						{usageData[usageData.length - 1]?.date ?? ''}
					</span>
				</div>
			</div>
		{/if}

		<!-- ==================== TOP CITED SOURCES ==================== -->
		{#if sources.length > 0}
			<div
				class="rounded-xl p-4"
				style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);"
			>
				<h3 class="text-sm font-medium mb-3" style="color: var(--oo-fg-primary);">
					Top Sources by Reliability
				</h3>
				<div class="space-y-2">
					{#each sources as src, i}
						{@const maxCite = Math.max(...sources.map((s) => s.citation_count), 1)}
						{@const barW = Math.max(4, Math.round((src.citation_count / maxCite) * 100))}
						<div class="flex items-center gap-3">
							<span
								class="text-xs font-mono w-5 text-right shrink-0"
								style="color: var(--oo-fg-muted);"
							>
								{i + 1}
							</span>
							<div class="flex-1 min-w-0">
								<div class="flex items-center justify-between mb-0.5">
									<span
										class="text-xs font-medium truncate"
										style="color: var(--oo-fg-primary);"
										title={src.source_file}
									>
										{shortName(src.source_file)}
									</span>
									<span
										class="text-xs font-mono shrink-0 ml-2"
										style="color: {scoreColor(src.reliability_score)};"
									>
										{(src.reliability_score * 100).toFixed(0)}%
									</span>
								</div>
								<div
									class="rounded-full"
									style="height: 4px; background-color: var(--oo-bd-subtle);"
								>
									<div
										class="rounded-full transition-all"
										style="height: 4px; width: {barW}%;
											background-color: var(--oo-acc-600); opacity: 0.7;"
									/>
								</div>
								<div class="flex gap-3 mt-0.5">
									<span class="text-xs" style="color: var(--oo-fg-muted);">
										{src.citation_count} citations
									</span>
									<span class="text-xs" style="color: var(--oo-fg-muted);">
										avg {(src.avg_score * 100).toFixed(0)}%
									</span>
								</div>
							</div>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<!-- ==================== COLLECTION HEALTH ==================== -->
		{#if health.length > 0}
			<div
				class="rounded-xl p-4"
				style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);"
			>
				<h3 class="text-sm font-medium mb-3" style="color: var(--oo-fg-primary);">
					Collection Health
				</h3>
				<div class="space-y-2">
					{#each health as coll}
						<div
							class="rounded-lg p-3"
							style="border: 1px solid var(--oo-bd-subtle);"
						>
							<div class="flex items-center justify-between mb-1">
								<span
									class="text-sm font-medium"
									style="color: var(--oo-fg-primary);"
								>
									{coll.name}
								</span>
								<span
									class="text-xs font-mono px-2 py-0.5 rounded-full"
									style="color: {scoreColor(coll.freshness_score)};
										background-color: var(--oo-bg-overlay);"
								>
									{(coll.freshness_score * 100).toFixed(0)}% fresh
								</span>
							</div>
							<div
								class="flex flex-wrap gap-x-4 gap-y-0.5 text-xs"
								style="color: var(--oo-fg-muted);"
							>
								<span>{coll.document_count} docs</span>
								<span>{coll.chunk_count} chunks</span>
								<span>{coll.citation_count} citations</span>
								<span>avg {Math.round(coll.avg_chunk_size)} chars/chunk</span>
								{#if coll.file_types.length > 0}
									<span>{coll.file_types.join(', ')}</span>
								{/if}
							</div>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<!-- ==================== EXTERNAL CONNECTORS ==================== -->
		<div
			class="rounded-xl p-4"
			style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);"
		>
			<h3 class="text-sm font-medium mb-3" style="color: var(--oo-fg-primary);">
				External Connectors
			</h3>

			<!-- Available backends -->
			<div class="flex gap-3 mb-3">
				{#each Object.entries(backends) as [name, available]}
					<span
						class="text-xs px-2 py-1 rounded-lg"
						style="background-color: var(--oo-bg-overlay);
							color: {available ? 'var(--oo-success)' : 'var(--oo-fg-muted)'};"
					>
						{name}: {available ? 'installed' : 'not installed'}
					</span>
				{/each}
			</div>

			{#if connectors.length === 0}
				<p class="text-xs" style="color: var(--oo-fg-muted);">
					No external connectors configured. Add them in config/rag.yaml.
				</p>
			{:else}
				<div class="space-y-2">
					{#each connectors as conn}
						<div
							class="flex items-center justify-between rounded-lg p-2"
							style="border: 1px solid var(--oo-bd-subtle);"
						>
							<div class="flex items-center gap-2">
								<span
									class="w-2 h-2 rounded-full"
									style="background-color: {conn.connected
										? 'var(--oo-success)'
										: 'var(--oo-error)'};"
								/>
								<span class="text-sm" style="color: var(--oo-fg-primary);">
									{conn.name}
								</span>
								<span class="text-xs" style="color: var(--oo-fg-muted);">
									({conn.connector_type})
								</span>
							</div>
							<div class="flex items-center gap-3 text-xs" style="color: var(--oo-fg-muted);">
								<span>{conn.document_count.toLocaleString()} vectors</span>
								{#if conn.last_query_time_ms > 0}
									<span>{conn.last_query_time_ms.toFixed(0)}ms</span>
								{/if}
								{#if conn.error}
									<span style="color: var(--oo-error);" title={conn.error}>
										error
									</span>
								{/if}
							</div>
						</div>
					{/each}
				</div>
			{/if}
		</div>

		<!-- ==================== REFRESH BUTTON ==================== -->
		<div class="flex justify-end">
			<button
				on:click={handleRefresh}
				disabled={refreshing}
				class="px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
				style="background-color: var(--oo-bg-overlay); color: var(--oo-fg-secondary);
					border: 1px solid var(--oo-bd-default);"
			>
				{refreshing ? 'Refreshing...' : 'Check for stale sources'}
			</button>
		</div>
	</div>
{/if}
