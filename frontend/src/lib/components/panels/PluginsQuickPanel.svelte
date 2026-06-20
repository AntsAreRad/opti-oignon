<!--
  PluginsQuickPanel.svelte (S108)
  Lightweight panel for the right sidebar: lists installed plugins,
  toggle enable/disable, and link to full Settings > Plugins page.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { closePanel } from '$lib/stores/panels';
	import { listPlugins, enablePlugin, disablePlugin } from '$lib/api/plugins';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type { PluginInfo } from '$lib/types';

	let plugins: PluginInfo[] = [];
	let loading = true;
	let toggling: Record<string, boolean> = {};

	// S109: Known slash commands per plugin
	const SLASH_COMMANDS: Record<string, string> = {
		'scratchpad': '/note, /notes, /note search',
		'task-extractor': '/tasks, /tasks done <id>',
		'github-connector': '/gh issues, /gh pr, /gh repo',
		'session-summarizer': '/summary',
	};

	function usageHint(plugin: PluginInfo): string {
		const cmds = SLASH_COMMANDS[plugin.name];
		if (cmds) return cmds;
		const hasToolCall = plugin.hooks.includes('tool_call');
		const hasPostInference = plugin.hooks.includes('post_inference');
		if (hasPostInference && !hasToolCall) return 'Runs automatically';
		if (hasToolCall) return 'Slash command';
		return '';
	}

	onMount(loadPlugins);

	async function loadPlugins() {
		loading = true;
		try {
			const resp = await listPlugins();
			plugins = resp.plugins;
		} catch {
			toastError('Failed to load plugins');
		} finally {
			loading = false;
		}
	}

	async function handleToggle(plugin: PluginInfo) {
		toggling = { ...toggling, [plugin.name]: true };
		try {
			if (plugin.state === 'enabled') {
				await disablePlugin(plugin.name);
				toastSuccess(`${plugin.name} disabled`);
			} else {
				await enablePlugin(plugin.name);
				toastSuccess(`${plugin.name} enabled`);
			}
			await loadPlugins();
		} catch {
			toastError(`Failed to toggle ${plugin.name}`);
		} finally {
			toggling = { ...toggling, [plugin.name]: false };
		}
	}

	function goToSettings() {
		closePanel();
		goto('/settings?tab=plugins');
	}
</script>

<div class="flex flex-col h-full" style="background-color: var(--oo-panel-bg);">
	<!-- Header -->
	<div class="flex items-center justify-between px-4 py-3 shrink-0"
		style="border-bottom: 1px solid var(--oo-bd-subtle);">
		<div class="flex items-center gap-2">
			<svg class="w-4 h-4" style="color: var(--oo-fg-tertiary);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
			</svg>
			<span class="text-sm font-medium" style="color: var(--oo-fg-secondary);">Plugins</span>
		</div>
		<button
			on:click={closePanel}
			class="p-1 rounded transition-colors"
			style="color: var(--oo-fg-muted);"
			title="Close panel"
			aria-label="Close plugins panel"
		>
			<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M6 18L18 6M6 6l12 12" />
			</svg>
		</button>
	</div>

	<!-- Plugin list -->
	<div class="flex-1 overflow-y-auto px-4 py-3 space-y-2">
		{#if loading}
			<p class="text-xs py-4 text-center" style="color: var(--oo-fg-muted);">Loading plugins...</p>
		{:else if plugins.length === 0}
			<p class="text-xs py-4 text-center" style="color: var(--oo-fg-muted);">No plugins installed.</p>
		{:else}
			{#each plugins as plugin (plugin.name)}
				<div class="rounded-lg px-3 py-2"
					style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);">
					<div class="flex items-center justify-between">
						<div class="min-w-0 flex-1">
							<div class="text-sm font-medium truncate" style="color: var(--oo-fg-primary);">
								{plugin.name}
							</div>
							<div class="text-xs truncate mt-0.5" style="color: var(--oo-fg-muted);">
								{plugin.description}
							</div>
							{#if usageHint(plugin)}
								<div class="text-[10px] mt-1 font-mono truncate"
									style="color: var(--oo-fg-tertiary);">
									{usageHint(plugin)}
								</div>
							{/if}
						</div>
						<!-- Toggle button -->
						<button
							on:click={() => handleToggle(plugin)}
							disabled={!!toggling[plugin.name]}
							class="shrink-0 ml-2 w-9 h-5 rounded-full transition-colors relative"
							style="background-color: {plugin.state === 'enabled' ? 'var(--oo-success)' : 'var(--oo-bg-overlay)'};"
							title="{plugin.state === 'enabled' ? 'Disable' : 'Enable'} {plugin.name}"
						>
							<span
								class="absolute top-0.5 w-4 h-4 rounded-full transition-all"
								style="background-color: var(--oo-toggle-knob);
									left: {plugin.state === 'enabled' ? '18px' : '2px'};"
							/>
						</button>
					</div>
					{#if plugin.hooks.length > 0}
						<div class="flex flex-wrap gap-1 mt-1.5">
							{#each plugin.hooks as hook}
								<span class="text-[10px] px-1.5 py-0.5 rounded"
									style="background-color: var(--oo-bg-overlay); color: var(--oo-fg-muted);">
									{hook}
								</span>
							{/each}
						</div>
					{/if}
				</div>
			{/each}
		{/if}
	</div>

	<!-- Footer: link to full settings -->
	<div class="px-4 py-3 shrink-0" style="border-top: 1px solid var(--oo-bd-subtle);">
		<button
			on:click={goToSettings}
			class="w-full text-xs py-2 rounded-lg transition-colors"
			style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-secondary);
				border: 1px solid var(--oo-bd-default);"
		>
			Open Plugins Settings
		</button>
	</div>
</div>
