<!--
  ReasoningDisplay.svelte
  Shows multi-step reasoning (CoT, ToT, Self-Consistency).
  Collapsible block with progress and final synthesis.
-->
<script lang="ts">
	import type { ReasoningStepInfo, ReasoningMetaInfo } from '$lib/types';
	import { Card } from '$lib/ds';

	export let steps: ReasoningStepInfo[] = [];
	export let meta: ReasoningMetaInfo | null = null;
	export let isStreaming: boolean = false;

	let expanded = false;

	$: hasSteps = steps.length > 0;
	$: totalDuration = meta?.total_duration_ms ?? steps.reduce((sum, s) => sum + s.duration_ms, 0);
	$: confidencePercent = meta ? Math.round(meta.confidence * 100) : 0;
	$: strategyLabel = meta?.strategy === 'tree_of_thought'
		? 'Tree of Thought'
		: meta?.strategy === 'self_consistency'
			? 'Self-Consistency'
			: 'Step-by-Step';
</script>

{#if hasSteps}
	<Card variant="flat" padding="none" class="mb-2 overflow-hidden">
		<!-- Header -->
		<button
			class="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-surface-400
				hover:text-surface-300 transition-colors cursor-pointer select-none"
			on:click={() => expanded = !expanded}
			aria-expanded={expanded}
		>
			<svg
				class="w-3 h-3 transition-transform flex-shrink-0 {expanded ? 'rotate-90' : ''}"
				fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"
			>
				<path d="M9 5l7 7-7 7" />
			</svg>

			<span class="font-medium">{strategyLabel}</span>

			<!-- Progress -->
			<span class="text-surface-500">
				{steps.length} step{steps.length !== 1 ? 's' : ''}
			</span>

			{#if isStreaming}
				<span class="inline-block w-1 h-3 bg-accent-500/50 animate-pulse" />
			{/if}

			<!-- Confidence indicator -->
			{#if meta && !isStreaming}
				<span class="ml-auto flex items-center gap-1.5 text-surface-500">
					<span class="inline-block w-8 h-1 rounded-full bg-surface-700 overflow-hidden">
						<span
							class="block h-full rounded-full transition-all duration-300
								{confidencePercent >= 70 ? 'bg-[var(--oo-success)]' : confidencePercent >= 40 ? 'bg-[var(--oo-warning)]' : 'bg-[var(--oo-error)]'}"
							style="width: {confidencePercent}%"
						/>
					</span>
					{confidencePercent}%
				</span>
			{/if}

			{#if totalDuration > 0 && !isStreaming}
				<span class="text-surface-600 ml-1">
					{(totalDuration / 1000).toFixed(1)}s
				</span>
			{/if}
		</button>

		<!-- Step content (collapsible) -->
		{#if expanded}
			<div class="border-t border-surface-700/30">
				{#each steps as step, i}
					<div class="px-3 py-1.5 {i > 0 ? 'border-t border-surface-800/50' : ''}">
						<div class="flex items-center gap-2 mb-0.5">
							<!-- Step number -->
							<span class="flex-shrink-0 w-4 h-4 rounded-full bg-surface-700
								text-[10px] font-mono text-surface-400 flex items-center justify-center">
								{step.step_number}
							</span>
							<span class="text-xs font-medium text-surface-300">{step.title}</span>
							{#if step.duration_ms > 0}
								<span class="ml-auto text-[10px] text-surface-600">
									{(step.duration_ms / 1000).toFixed(1)}s
								</span>
							{/if}
						</div>
						<div class="ml-6 text-xs text-surface-500 leading-relaxed whitespace-pre-wrap
							max-h-32 overflow-y-auto">
							{step.content.length > 500 ? step.content.substring(0, 500) + '...' : step.content}
						</div>
					</div>
				{/each}
			</div>
		{/if}
	</Card>
{/if}
