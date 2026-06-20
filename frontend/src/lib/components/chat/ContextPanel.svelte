<!--
  ContextPanel.svelte
  Side panel showing context pipeline health:
  token usage bar, counters, trimming, budget allocation.
  Refreshes after each message exchange.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { activeConversationId } from '$lib/stores/conversations';
	import { selectedModel } from '$lib/stores/chatOptions';
	import { isStreaming } from '$lib/stores/chat';
	import { getContextHealth } from '$lib/api/context';
	import type { ContextHealthResponse } from '$lib/types';

	let data: ContextHealthResponse | null = null;
	let loading = false;
	let error: string | null = null;
	let refreshTimer: ReturnType<typeof setInterval> | null = null;

	// Load context data
	async function loadData() {
		loading = true;
		error = null;
		try {
			data = await getContextHealth(
				$activeConversationId || undefined,
				$selectedModel || undefined
			);
		} catch (e: any) {
			error = e?.detail || e?.message || 'Failed to load context health';
			data = null;
		} finally {
			loading = false;
		}
	}

	// Refresh after streaming ends
	$: if (!$isStreaming && $activeConversationId) {
		loadData();
	}

	// Refresh when the conversation changes
	$: if ($activeConversationId) {
		loadData();
	}

	// Refresh when the model changes
	$: if ($selectedModel) {
		loadData();
	}

	// Periodic refresh (every 30s)
	onMount(() => {
		loadData();
		refreshTimer = setInterval(loadData, 30000);
	});

	onDestroy(() => {
		if (refreshTimer) clearInterval(refreshTimer);
	});

	// Data access shortcuts
	$: conv = data?.current_conversation;
	$: budget = data?.budget_allocation;
	$: contextStatus = data?.status || 'unknown';

	// Bar color based on usage
	$: usagePercent = conv?.usage_percent ?? 0;
	$: barColor =
		usagePercent > 90 ? 'bg-[var(--oo-error)]' :
		usagePercent > 70 ? 'bg-[var(--oo-warning)]' :
		usagePercent > 50 ? 'bg-[var(--oo-warning)]' :
		'bg-[var(--oo-success)]';

	// Status badge color
	$: statusColor =
		contextStatus === 'healthy' ? 'text-[var(--oo-success)]' :
		contextStatus === 'trimming' ? 'text-[var(--oo-warning)]' :
		contextStatus === 'warning' ? 'text-[var(--oo-error)]' :
		contextStatus === 'degraded' ? 'text-surface-500' :
		'text-surface-600';

	// Formatage compact
	function formatTokens(n: number): string {
		if (n < 1000) return String(n);
		if (n < 100000) return `${(n / 1000).toFixed(1)}k`;
		return `${(n / 1000).toFixed(0)}k`;
	}

	function formatPercent(n: number): string {
		return `${n.toFixed(1)}%`;
	}
</script>

<div class="h-full flex flex-col text-sm">
	<!-- Panel header -->
	<div class="px-4 py-3 border-b border-surface-800 flex items-center justify-between">
		<div class="flex items-center gap-2">
			<svg class="w-4 h-4 text-surface-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
			</svg>
			<span class="font-medium text-surface-300">Context Health</span>
		</div>
		<button
			on:click={loadData}
			class="p-1 rounded text-surface-500 hover:text-surface-300 hover:bg-surface-800 transition-colors"
			title="Refresh"
			aria-label="Refresh context data"
			disabled={loading}
		>
			<svg class="w-3.5 h-3.5 {loading ? 'animate-spin' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
			</svg>
		</button>
	</div>

	<!-- Content -->
	<div class="flex-1 overflow-y-auto px-4 py-3 space-y-4">
		{#if loading && !data}
			<div class="text-center text-surface-500 py-8">
				<svg class="w-5 h-5 mx-auto animate-spin mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
				</svg>
				Loading context data...
			</div>
		{:else if error}
			<div class="text-center text-[var(--oo-error)]/80 py-4">
				<p class="text-xs">{error}</p>
				<button on:click={loadData} class="mt-2 text-xs text-accent-400 hover:underline">Retry</button>
			</div>
		{:else if data}
			<!-- Status badge -->
			<div class="flex items-center gap-2">
				<span class="inline-block w-2 h-2 rounded-full {statusColor.replace('text-', 'bg-')}" />
				<span class="text-xs font-medium {statusColor} uppercase tracking-wider">{contextStatus}</span>
				{#if !data.context_window_available}
					<span class="text-xs text-surface-600">(module unavailable)</span>
				{/if}
			</div>

			<!-- Token usage bar -->
			{#if conv}
				<div class="space-y-1.5">
					<div class="flex items-center justify-between text-xs">
						<span class="text-surface-400">Token Usage</span>
						<span class="text-surface-300 tabular-nums">
							{formatTokens(conv.estimated_tokens)} / {formatTokens(conv.model_context_window || 0)}
						</span>
					</div>
					<div class="w-full h-2 rounded-full bg-surface-800 overflow-hidden">
						<div
							class="h-full rounded-full transition-all duration-500 {barColor}"
							style="width: {Math.min(usagePercent, 100)}%"
						/>
					</div>
					<div class="text-right text-xs text-surface-500 tabular-nums">
						{formatPercent(usagePercent)}
					</div>
				</div>

				<!-- Compteurs -->
				<div class="grid grid-cols-2 gap-2">
					<div class="bg-surface-800/50 rounded-lg px-3 py-2">
						<div class="text-xs text-surface-500">Messages</div>
						<div class="text-base font-medium text-surface-200 tabular-nums">{conv.messages_count}</div>
					</div>
					<div class="bg-surface-800/50 rounded-lg px-3 py-2">
						<div class="text-xs text-surface-500">Est. Tokens</div>
						<div class="text-base font-medium text-surface-200 tabular-nums">{formatTokens(conv.estimated_tokens)}</div>
					</div>
				</div>

				<!-- Active model -->
				{#if conv.model}
					<div class="bg-surface-800/50 rounded-lg px-3 py-2">
						<div class="text-xs text-surface-500 mb-0.5">Active Model</div>
						<div class="text-sm text-surface-300 font-mono">{conv.model}</div>
						<div class="text-xs text-surface-500 mt-0.5">
							Context window: {formatTokens(conv.model_context_window)} tokens
						</div>
					</div>
				{/if}

				<!-- Indicateur de trimming -->
				<div class="flex items-center gap-2 px-3 py-2 rounded-lg {conv.trimming_active ? 'bg-[var(--oo-warning-bg)] border border-[var(--oo-warning-bd)]' : 'bg-surface-800/50'}">
					<span class="inline-block w-2 h-2 rounded-full {conv.trimming_active ? 'bg-[var(--oo-warning)]' : 'bg-[var(--oo-success)]'}" />
					<span class="text-xs {conv.trimming_active ? 'text-[var(--oo-warning)]' : 'text-surface-400'}">
						{conv.trimming_active ? 'Trimming active' : 'No trimming needed'}
					</span>
				</div>

				<!-- Stats fenetre glissante -->
				{#if conv.last_window_stats && Object.keys(conv.last_window_stats).length > 0}
					<div class="space-y-1">
						<div class="text-xs text-surface-500 font-medium">Last Window Stats</div>
						<div class="bg-surface-800/30 rounded-lg px-3 py-2 space-y-1">
							{#if conv.last_window_stats.strategy}
								<div class="flex justify-between text-xs">
									<span class="text-surface-500">Strategy</span>
									<span class="text-surface-300">{conv.last_window_stats.strategy}</span>
								</div>
							{/if}
							{#if conv.last_window_stats.kept !== undefined}
								<div class="flex justify-between text-xs">
									<span class="text-surface-500">Kept</span>
									<span class="text-surface-300 tabular-nums">{conv.last_window_stats.kept}</span>
								</div>
							{/if}
							{#if conv.last_window_stats.dropped !== undefined}
								<div class="flex justify-between text-xs">
									<span class="text-surface-500">Dropped</span>
									<span class="text-surface-300 tabular-nums {conv.last_window_stats.dropped > 0 ? 'text-[var(--oo-warning)]' : ''}">{conv.last_window_stats.dropped}</span>
								</div>
							{/if}
							{#if conv.last_window_stats.total_tokens !== undefined}
								<div class="flex justify-between text-xs">
									<span class="text-surface-500">Total Tokens</span>
									<span class="text-surface-300 tabular-nums">{formatTokens(conv.last_window_stats.total_tokens)}</span>
								</div>
							{/if}
						</div>
					</div>
				{/if}
			{:else}
				<div class="text-center text-surface-500 py-4 text-xs">
					No active conversation
				</div>
			{/if}

			<!-- Budget allocation -->
			{#if budget && budget.context_window > 0}
				<div class="space-y-1.5">
					<div class="text-xs text-surface-500 font-medium">Budget Allocation</div>

					<!-- Segmented bar -->
					<div class="w-full h-3 rounded-full bg-surface-800 overflow-hidden flex">
						<div
							class="h-full bg-[var(--oo-info)] transition-all"
							style="width: {budget.system_ratio * 100}%"
							title="System: {formatTokens(budget.system_prompt)}"
						/>
						<div
							class="h-full bg-[var(--oo-success)] transition-all"
							style="width: {budget.history_ratio * 100}%"
							title="History: {formatTokens(budget.history)}"
						/>
						<div
							class="h-full bg-[var(--oo-cat-purple)] transition-all"
							style="width: {budget.generation_ratio * 100}%"
							title="Response: {formatTokens(budget.reserved_for_response)}"
						/>
					</div>

					<!-- Legende -->
					<div class="space-y-1 text-xs">
						<div class="flex items-center justify-between">
							<span class="flex items-center gap-1.5">
								<span class="inline-block w-2 h-2 rounded-sm bg-[var(--oo-info)]" />
								<span class="text-surface-400">System</span>
							</span>
							<span class="text-surface-300 tabular-nums">{formatTokens(budget.system_prompt)} ({(budget.system_ratio * 100).toFixed(0)}%)</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="flex items-center gap-1.5">
								<span class="inline-block w-2 h-2 rounded-sm bg-[var(--oo-success)]" />
								<span class="text-surface-400">History</span>
							</span>
							<span class="text-surface-300 tabular-nums">{formatTokens(budget.history)} ({(budget.history_ratio * 100).toFixed(0)}%)</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="flex items-center gap-1.5">
								<span class="inline-block w-2 h-2 rounded-sm bg-[var(--oo-cat-purple)]" />
								<span class="text-surface-400">Response</span>
							</span>
							<span class="text-surface-300 tabular-nums">{formatTokens(budget.reserved_for_response)} ({(budget.generation_ratio * 100).toFixed(0)}%)</span>
						</div>
						<div class="flex items-center justify-between border-t border-surface-800 pt-1 mt-1">
							<span class="text-surface-400">Total allocated</span>
							<span class="text-surface-300 tabular-nums">{formatTokens(budget.total_allocated)}</span>
						</div>
					</div>
				</div>
			{/if}
		{/if}
	</div>
</div>
