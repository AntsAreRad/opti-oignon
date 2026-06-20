<!--
  CodingAgentProgress.svelte (S118)
  Live progress display during coding agent execution.
  Shows plan steps, implementation progress, test results, fix attempts
  as they stream in real-time via WebSocket coding events.
-->
<script lang="ts">
	import { streamingCodingEvents, isCodingStream } from '$lib/stores/chat';
	import { Card } from '$lib/ds';
	import type { CodingEventEntry } from '$lib/stores/chat';

	$: events = $streamingCodingEvents;
	$: active = $isCodingStream;

	// Extract structured state from events
	$: planSteps = (() => {
		const planEvent = events.find((e) => e.eventType === 'coding_plan');
		return (planEvent?.data?.steps as string[]) || [];
	})();

	$: currentPhase = (() => {
		if (events.length === 0) return '';
		const last = events[events.length - 1];
		switch (last.eventType) {
			case 'coding_status': return last.content || 'Starting...';
			case 'coding_plan': return 'Plan ready';
			case 'coding_step': return last.content || 'Implementing...';
			case 'coding_test': return last.content || 'Testing...';
			case 'coding_fix': return last.content || 'Fixing...';
			case 'coding_done': return 'Done';
			default: return last.content || '';
		}
	})();

	$: lastTest = (() => {
		const testEvents = events.filter((e) => e.eventType === 'coding_test');
		if (testEvents.length === 0) return null;
		return testEvents[testEvents.length - 1];
	})();

	$: fixAttempts = events.filter((e) => e.eventType === 'coding_fix').length;

	$: implementedFiles = events
		.filter((e) => e.eventType === 'coding_step' && e.data?.action === 'write_file')
		.map((e) => e.data?.file as string)
		.filter(Boolean);

	$: isDone = events.some((e) => e.eventType === 'coding_done');
</script>

{#if active && events.length > 0 && !isDone}
	<Card variant="flat" padding="none" class="overflow-hidden text-xs mt-2 mb-1">

		<!-- Header with animated indicator -->
		<div class="flex items-center gap-2 px-3 py-1.5"
			style="border-bottom: 1px solid var(--oo-bd-subtle);">
			<div class="w-2 h-2 rounded-full animate-pulse"
				style="background-color: var(--oo-sage);" />
			<span style="color: var(--oo-sage); font-weight: 500;">
				Code Agent
			</span>
			<span style="color: var(--oo-fg-muted);">{currentPhase}</span>
		</div>

		<!-- Plan steps (if available) -->
		{#if planSteps.length > 0}
			<div class="px-3 py-1.5 space-y-0.5"
				style="border-bottom: 1px solid var(--oo-bd-subtle);">
				{#each planSteps as step, i}
					<div class="flex items-start gap-1.5">
						{#if i < implementedFiles.length}
							<svg class="w-3 h-3 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24"
								stroke="currentColor" stroke-width="2.5"
								style="color: var(--oo-success);">
								<path d="M5 13l4 4L19 7" />
							</svg>
						{:else if i === implementedFiles.length}
							<div class="w-3 h-3 mt-0.5 flex-shrink-0 rounded-full animate-pulse"
								style="background-color: var(--oo-sage); opacity: 0.6;" />
						{:else}
							<div class="w-3 h-3 mt-0.5 flex-shrink-0 rounded-full"
								style="background-color: var(--oo-bg-base); border: 1px solid var(--oo-bd-subtle);" />
						{/if}
						<span style="color: {i <= implementedFiles.length ? 'var(--oo-fg-secondary)' : 'var(--oo-fg-muted)'};">
							{step}
						</span>
					</div>
				{/each}
			</div>
		{/if}

		<!-- Files being written -->
		{#if implementedFiles.length > 0}
			<div class="px-3 py-1 flex flex-wrap gap-1"
				style="border-bottom: 1px solid var(--oo-bd-subtle);">
				{#each implementedFiles as file}
					<span class="px-1.5 py-0.5 rounded font-mono"
						style="background-color: var(--oo-bg-base); color: var(--oo-fg-secondary);">
						{file}
					</span>
				{/each}
			</div>
		{/if}

		<!-- Test result (if running) -->
		{#if lastTest}
			<div class="px-3 py-1 flex items-center gap-2">
				{#if lastTest.data?.passed}
					<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"
						style="color: var(--oo-success);">
						<path d="M5 13l4 4L19 7" />
					</svg>
					<span style="color: var(--oo-success);">Tests passed</span>
				{:else}
					<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"
						style="color: var(--oo-error);">
						<path d="M6 18L18 6M6 6l12 12" />
					</svg>
					<span style="color: var(--oo-error);">Tests failed</span>
					{#if fixAttempts > 0}
						<span style="color: var(--oo-fg-muted);">
							(fix attempt {fixAttempts}...)
						</span>
					{/if}
				{/if}
			</div>
		{/if}
	</Card>
{/if}
