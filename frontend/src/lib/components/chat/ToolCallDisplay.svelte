<!--
  ToolCallDisplay.svelte
  Displays tool calls inline in assistant messages.
  Shows tool name, arguments, result (collapsible),
  execution time, and a colored status indicator.
  S44: Tool Calling Framework
  S62: Multi-turn tool call history timeline
-->
<script lang="ts">
	import type { ToolCallInfo } from '$lib/types';
	import PluginPermissionBadge from './PluginPermissionBadge.svelte';
	import { Card } from '$lib/ds';

	export let toolCalls: ToolCallInfo[] = [];
	export let toolHistory: ToolCallInfo[] = [];

	let expandedIndex: number | null = null;
	let historyExpanded = false;

	// S169: A tool call may carry optional plugin metadata (plugin_name +
	// permissions) when it was invoked through a plugin; surface it inline.
	function pluginPerms(call: ToolCallInfo): string[] {
		const p = (call as Record<string, unknown>).permissions;
		return Array.isArray(p) ? (p as string[]) : [];
	}

	function pluginName(call: ToolCallInfo): string {
		const n = (call as Record<string, unknown>).plugin_name;
		return typeof n === 'string' ? n : call.tool_name;
	}

	function toggleExpand(index: number) {
		expandedIndex = expandedIndex === index ? null : index;
	}

	function formatTime(seconds: number): string {
		if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
		return `${seconds.toFixed(1)}s`;
	}

	function truncateResult(result: string, maxLen: number = 120): string {
		if (result.length <= maxLen) return result;
		return result.substring(0, maxLen) + '...';
	}
</script>

<!-- S62: Prior tool history timeline -->
{#if toolHistory.length > 0}
	<div class="mb-2">
		<button
			class="flex items-center gap-1.5 text-xs text-surface-500 hover:text-surface-300
				transition-colors px-2 py-1 rounded-md hover:bg-surface-800/30"
			on:click={() => historyExpanded = !historyExpanded}
			aria-expanded={historyExpanded}
		>
			<svg class="w-3 h-3 transition-transform {historyExpanded ? 'rotate-90' : ''}"
				fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M9 5l7 7-7 7" />
			</svg>
			<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
			</svg>
			<span>{toolHistory.length} prior tool call{toolHistory.length !== 1 ? 's' : ''}</span>
		</button>

		{#if historyExpanded}
			<div class="ml-3 mt-1 border-l-2 border-surface-700/50 pl-3 flex flex-col gap-1">
				{#each toolHistory as priorCall, idx}
					<div class="flex items-center gap-2 text-xs text-surface-500 py-0.5">
						<span class="flex-shrink-0 w-1 h-1 rounded-full
							{priorCall.success ? 'bg-[var(--oo-success)]/60' : 'bg-[var(--oo-error)]/60'}" />
						<span class="font-mono text-surface-400">{priorCall.tool_name}</span>
						{#if priorCall.execution_time !== undefined && priorCall.execution_time > 0}
							<span class="text-surface-600">{formatTime(priorCall.execution_time)}</span>
						{/if}
						{#if priorCall.result_preview}
							<span class="truncate text-surface-600 max-w-[200px]">
								{truncateResult(priorCall.result_preview, 60)}
							</span>
						{/if}
					</div>
				{/each}
			</div>
		{/if}
	</div>
{/if}

<!-- Current turn tool calls -->
{#if toolCalls.length > 0}
	<div class="flex flex-col gap-1.5 mt-1.5 mb-1">
		{#each toolCalls as call, idx}
			<Card variant="flat" padding="none" class="text-xs overflow-hidden">
				<div class={call.success ? 'bg-[var(--oo-success-bg)]/20' : 'bg-[var(--oo-error-bg)]/20'}>
					<!-- Clickable header -->
				<button
					class="w-full flex items-center gap-2 px-2.5 py-1.5 text-left
						hover:bg-surface-800/30 transition-colors rounded-lg"
					on:click={() => toggleExpand(idx)}
					aria-expanded={expandedIndex === idx}
				>
					<!-- Status indicator -->
					<span class="flex-shrink-0 w-1.5 h-1.5 rounded-full
						{call.success ? 'bg-[var(--oo-success)]' : 'bg-[var(--oo-error)]'}" />

					<!-- Tool icon -->
					<svg class="w-3 h-3 flex-shrink-0
						{call.success ? 'text-[var(--oo-success)]' : 'text-[var(--oo-error)]'}"
						fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						{#if call.tool_name === 'web_search'}
							<circle cx="11" cy="11" r="8" />
							<path d="M21 21l-4.35-4.35" />
						{:else if call.tool_name === 'execute_code'}
							<path d="M16 18l6-6-6-6" />
							<path d="M8 6l-6 6 6 6" />
						{:else if call.tool_name === 'read_file' || call.tool_name === 'write_file'}
							<path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
							<path d="M14 2v6h6" />
						{:else if call.tool_name === 'list_files'}
							<path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
						{:else}
							<path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z" />
						{/if}
					</svg>

					<!-- Tool name -->
					<span class="font-mono font-medium
						{call.success ? 'text-[var(--oo-success)]' : 'text-[var(--oo-error)]'}">
						{call.tool_name}
					</span>

					<!-- S169: plugin permission badge (only when plugin metadata present) -->
					{#if pluginPerms(call).length > 0}
						<PluginPermissionBadge pluginName={pluginName(call)} permissions={pluginPerms(call)} />
					{/if}

					<!-- Running status -->
					{#if call.status === 'executing'}
						<span class="text-surface-500 flex items-center gap-1">
							<span class="inline-block w-1 h-3 bg-accent-500/50 animate-pulse" />
							running
						</span>
					{/if}

					<!-- Execution time -->
					{#if call.execution_time !== undefined && call.execution_time > 0}
						<span class="text-surface-500 ml-auto">
							{formatTime(call.execution_time)}
						</span>
					{/if}

					<!-- Expand arrow -->
					<svg class="w-3 h-3 text-surface-500 transition-transform flex-shrink-0
						{expandedIndex === idx ? 'rotate-90' : ''}"
						fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path d="M9 5l7 7-7 7" />
					</svg>
				</button>

				<!-- Details (expandable) -->
				{#if expandedIndex === idx}
					<div class="px-2.5 pb-2 border-t
						{call.success ? 'border-[var(--oo-success-bd)]/20' : 'border-[var(--oo-error-bd)]/20'}">
						<!-- Arguments -->
						{#if call.arguments && Object.keys(call.arguments).length > 0}
							<div class="mt-1.5">
								<span class="text-surface-500 font-medium">Args:</span>
								<pre class="mt-0.5 text-surface-400 bg-surface-900/50 rounded px-2 py-1
									overflow-x-auto max-h-24 whitespace-pre-wrap">{JSON.stringify(call.arguments, null, 2)}</pre>
							</div>
						{/if}

						<!-- Reasoning -->
						{#if call.reasoning}
							<div class="mt-1.5">
								<span class="text-surface-500 font-medium">Reasoning:</span>
								<span class="text-surface-400 ml-1">{call.reasoning}</span>
							</div>
						{/if}

						<!-- Result -->
						{#if call.result_preview}
							<div class="mt-1.5">
								<span class="text-surface-500 font-medium">Result:</span>
								<pre class="mt-0.5 text-surface-400 bg-surface-900/50 rounded px-2 py-1
									overflow-x-auto max-h-40 whitespace-pre-wrap">{call.result_preview}</pre>
							</div>
						{/if}
					</div>
				{:else if call.result_preview}
					<!-- Compact preview -->
					<div class="px-2.5 pb-1.5 text-surface-500 truncate">
						{truncateResult(call.result_preview)}
					</div>
				{/if}
				</div>
			</Card>
		{/each}
	</div>
{/if}
