<!--
  CodingAgentInline.svelte (S118)
  Displays inline when a chat message was produced by the coding agent.
  Shows: plan steps with progress, test results, fix attempts,
  and embedded SandboxFileManager for preview/download.
-->
<script lang="ts">
	import SandboxFileManager from '$lib/components/panels/SandboxFileManager.svelte';
	import SandboxIsolationBadge from '$lib/components/chat/SandboxIsolationBadge.svelte';
	import { Card } from '$lib/ds';

	/** Coding result from done metadata. */
	export let codingResult: {
		turn?: number;
		plan?: string;
		files_written?: string[];
		test_passed?: boolean | null;
		fix_attempts?: number;
		summary?: string;
		vision_meta?: Record<string, unknown>;
		tool_calls?: { tool_name: string; result: string; success: boolean }[];
		plugin_annotations?: { plugin: string; data: unknown }[];
	} = {};

	/** Sandbox session ID for file management. */
	export let sandboxSessionId: string = '';

	/** Number of files in sandbox. */
	export let sandboxFiles: unknown[] = [];

	/** Turn count in the coding session. */
	export let turnCount: number = 0;

	let planOpen = false;

	$: planSteps = (codingResult.plan || '')
		.split('\n')
		.filter((l) => l.trim())
		.filter((l) => /^\d+[.)]/.test(l.trim()));

	$: hasPlan = planSteps.length > 0;
	$: hasFiles = (codingResult.files_written || []).length > 0;
	$: testStatus = codingResult.test_passed;
	$: fixAttempts = codingResult.fix_attempts || 0;
	$: hasVision = !!codingResult.vision_meta;
	$: hasToolCalls = (codingResult.tool_calls || []).length > 0;
	$: hasPlugins = (codingResult.plugin_annotations || []).length > 0;
</script>

<Card variant="flat" padding="none" class="mt-2 overflow-hidden text-xs">
	<!-- Header -->
	<div class="flex items-center gap-2 px-3 py-1.5"
		style="background-color: var(--oo-sage-bg); border-bottom: 1px solid var(--oo-sage-bd);">
		<svg class="w-3.5 h-3.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.8"
			style="color: var(--oo-sage);">
			<path d="M16 18l2-2-2-2" /><path d="M8 6L6 8l2 2" />
			<path d="M14.5 4l-5 16" />
		</svg>
		<span style="color: var(--oo-sage); font-weight: 500;">
			Code Agent
		</span>
		{#if turnCount > 0}
			<span style="color: var(--oo-fg-muted);">Turn {turnCount}</span>
		{/if}

		<!-- Feature badges -->
		{#if hasVision}
			<span class="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded"
				style="background-color: var(--oo-bg-base); color: var(--oo-fg-muted);">
				<svg class="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
					<path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
				</svg>
				Vision
			</span>
		{/if}
		{#if hasToolCalls}
			<span class="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded"
				style="background-color: var(--oo-bg-base); color: var(--oo-fg-muted);">
				{codingResult.tool_calls?.length} tool(s)
			</span>
		{/if}
		{#if hasPlugins}
			<span class="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded"
				style="background-color: var(--oo-bg-base); color: var(--oo-fg-muted);">
				{codingResult.plugin_annotations?.length} plugin(s)
			</span>
		{/if}

		<!-- S125: Sandbox isolation indicator -->
		<SandboxIsolationBadge />

		<div class="flex-1" />

		<!-- Test result badge -->
		{#if testStatus === true}
			<span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded"
				style="background-color: var(--oo-success-bg); color: var(--oo-success); border: 1px solid var(--oo-success-bd);">
				<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
					<path d="M5 13l4 4L19 7" />
				</svg>
				Tests passed
			</span>
		{:else if testStatus === false}
			<span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded"
				style="background-color: var(--oo-error-bg); color: var(--oo-error); border: 1px solid var(--oo-error-bd);">
				<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
					<path d="M6 18L18 6M6 6l12 12" />
				</svg>
				Tests failed
				{#if fixAttempts > 0}
					({fixAttempts} fix attempts)
				{/if}
			</span>
		{/if}
	</div>

	<!-- Plan (collapsible) -->
	{#if hasPlan}
		<details class="group" bind:open={planOpen}>
			<summary class="cursor-pointer px-3 py-1 flex items-center gap-1.5 select-none"
				style="color: var(--oo-fg-muted); border-bottom: 1px solid var(--oo-bd-subtle);">
				<svg class="w-3 h-3 transition-transform {planOpen ? 'rotate-90' : ''}"
					fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M9 5l7 7-7 7" />
				</svg>
				Plan ({planSteps.length} steps)
			</summary>
			<div class="px-3 py-1.5 space-y-0.5"
				style="color: var(--oo-fg-secondary); border-bottom: 1px solid var(--oo-bd-subtle);">
				{#each planSteps as step, i}
					<div class="flex items-start gap-1.5">
						<svg class="w-3 h-3 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24"
							stroke="currentColor" stroke-width="2"
							style="color: var(--oo-success);">
							<path d="M5 13l4 4L19 7" />
						</svg>
						<span>{step.replace(/^\d+[.)]\s*/, '')}</span>
					</div>
				{/each}
			</div>
		</details>
	{/if}

	<!-- Files written -->
	{#if hasFiles}
		<div class="px-3 py-1.5 flex flex-wrap gap-1"
			style="border-bottom: 1px solid var(--oo-bd-subtle);">
			<span style="color: var(--oo-fg-muted);">Files:</span>
			{#each codingResult.files_written || [] as file}
				<span class="px-1.5 py-0.5 rounded font-mono"
					style="background-color: var(--oo-bg-base); color: var(--oo-fg-secondary);">
					{file}
				</span>
			{/each}
		</div>
	{/if}

	<!-- Sandbox file manager -->
	{#if sandboxSessionId}
		<SandboxFileManager sessionId={sandboxSessionId} />
	{/if}
</Card>
