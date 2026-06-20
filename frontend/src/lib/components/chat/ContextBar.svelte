<!--
  ContextBar.svelte
  Compact bar showing tokens, active model and routing info.
  Always visible below presets when a conversation is active.
  BUG-08 S108: Synced with context health API (same source as ContextPanel).
-->
<script lang="ts">
	import { activeConversation, activeConversationId } from '$lib/stores/conversations';
	import { selectedModel } from '$lib/stores/chatOptions';
	import { isStreaming } from '$lib/stores/chat';
	import { getContextHealth } from '$lib/api/context';
	import ProjectContextBadge from './ProjectContextBadge.svelte';
	import ProjectLinker from './ProjectLinker.svelte';
	import type { ContextHealthResponse } from '$lib/types';

	let data: ContextHealthResponse | null = null;

	// Reload when conversation, model, or streaming state changes
	$: if ($activeConversationId || $selectedModel || !$isStreaming) {
		loadContextData();
	}

	async function loadContextData() {
		if (!$activeConversationId) { data = null; return; }
		try {
			data = await getContextHealth(
				$activeConversationId || undefined,
				$selectedModel || undefined
			);
		} catch {
			data = null;
		}
	}

	$: conv = $activeConversation;
	$: taskType = conv?.task_type ?? null;
	$: preset = conv?.preset ?? null;

	// Real token data from context health API
	$: totalTokens = data?.current_conversation?.estimated_tokens ?? 0;
	$: budget = data?.current_conversation?.model_context_window ?? 0;
	$: modelName = data?.current_conversation?.model || $selectedModel || conv?.model || 'default';

	// Usage percentage
	$: usagePercent = budget > 0 ? Math.min((totalTokens / budget) * 100, 100) : 0;

	// Bar color based on usage
	$: barColor =
		usagePercent > 90 ? 'bg-[var(--oo-error)]' :
		usagePercent > 70 ? 'bg-[var(--oo-warning)]' :
		'bg-accent-500';

	// Compact token formatting
	function formatTokens(n: number): string {
		if (n < 1000) return String(n);
		if (n < 10000) return `${(n / 1000).toFixed(1)}k`;
		return `${(n / 1000).toFixed(0)}k`;
	}

	// Short model name
	function shortModelName(name: string): string {
		const base = name.split(':')[0];
		if (base.length > 16) return base.substring(0, 14) + '...';
		return name;
	}
</script>

{#if $activeConversationId}
	<div class="flex items-center gap-3 px-4 py-1 text-[11px]"
		style="border-bottom: 1px solid var(--oo-bd-subtle); color: var(--oo-fg-muted);">
		<!-- Token counter + budget bar -->
		<div class="flex items-center gap-1.5 shrink-0" title="{totalTokens} / {budget} tokens">
			<svg class="w-3 h-3 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />
			</svg>
			<span class="tabular-nums">{formatTokens(totalTokens)}</span>
			<span style="color: var(--oo-fg-faint);">/</span>
			<span class="tabular-nums">{formatTokens(budget)}</span>

			<!-- Mini progress bar -->
			<div class="w-12 h-1 rounded-full overflow-hidden" style="background-color: var(--oo-bg-elevated);">
				<div
					class="h-full rounded-full transition-all duration-300 {barColor}"
					style="width: {usagePercent}%"
				/>
			</div>
		</div>

		<!-- Separator -->
		<div class="w-px h-3 shrink-0" style="background-color: var(--oo-bd-subtle);" />

		<!-- Active model -->
		<div class="flex items-center gap-1 shrink-0 min-w-0" title="Model: {modelName}">
			<svg class="w-3 h-3 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
			</svg>
			<span class="truncate">{shortModelName(modelName)}</span>
		</div>

		<!-- Task type (if detected) -->
		{#if taskType}
			<div class="w-px h-3 shrink-0 hidden sm:block" style="background-color: var(--oo-bd-subtle);" />
			<span class="hidden sm:inline-flex items-center gap-1 truncate" title="Detected task: {taskType}">
				<svg class="w-3 h-3 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M13 10V3L4 14h7v7l9-11h-7z" />
				</svg>
				{taskType}
			</span>
		{/if}

		<!-- Preset (if active) -->
		{#if preset}
			<div class="w-px h-3 shrink-0 hidden sm:block" style="background-color: var(--oo-bd-subtle);" />
			<span class="hidden sm:inline-flex items-center gap-1 truncate" title="Preset: {preset}">
				<svg class="w-3 h-3 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
				</svg>
				{preset}
			</span>
		{/if}

		<!-- Project context badge + linker (S59) -->
		<div class="w-px h-3 shrink-0 hidden sm:block" style="background-color: var(--oo-bd-subtle);" />
		<ProjectContextBadge on:openProject />
		<ProjectLinker />
	</div>
{/if}
