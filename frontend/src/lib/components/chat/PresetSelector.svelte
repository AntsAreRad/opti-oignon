<!--
  PresetSelector.svelte
  Preset picker rendered as horizontal chips with a toggle for automatic
  preset detection. Uses the ds Icon primitive for the auto toggle.
-->
<script lang="ts">
	import Icon from '$lib/ds/Icon.svelte';
	import {
		availablePresets,
		selectedPreset,
		usePresets,
		optionsLoading,
	} from '$lib/stores/chatOptions';
	import type { PresetInfo } from '$lib/types';

	let expanded = false;

	$: presetList = $availablePresets;
	$: visiblePresets = expanded ? presetList : presetList.slice(0, 5);
	$: hasMore = presetList.length > 5;
	$: activePreset = presetList.find((p) => p.id === $selectedPreset) ?? null;

	function togglePreset(id: string) {
		if ($selectedPreset === id) {
			selectedPreset.set(null);
		} else {
			selectedPreset.set(id);
		}
	}

	function toggleAutoPresets() {
		usePresets.update((v) => !v);
	}
</script>

{#if presetList.length > 0}
	<div class="flex items-center gap-1.5 flex-wrap">
		<!-- Toggle auto-detection -->
		<button
			on:click={toggleAutoPresets}
			class="flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-medium transition-colors
				{$usePresets
					? 'bg-accent-600/15 text-accent-400'
					: 'bg-surface-800/60 text-surface-500 line-through'}"
			title="{$usePresets ? 'Disable' : 'Enable'} auto preset detection"
			aria-pressed={$usePresets}
		>
			{#if $usePresets}
				<Icon name="lightbulb" size="sm" />
			{:else}
				<Icon name="lightbulb-off" size="sm" />
			{/if}
			Auto
		</button>

		<!-- Separator -->
		<div class="w-px h-4 bg-surface-700" />

		<!-- Preset pills -->
		{#each visiblePresets as preset (preset.id)}
			<button
				on:click={() => togglePreset(preset.id)}
				class="flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-medium transition-colors
					{$selectedPreset === preset.id
						? 'bg-accent-600/20 text-accent-400 ring-1 ring-accent-600/30'
						: 'bg-surface-800/50 text-surface-400 hover:bg-surface-800 hover:text-surface-300'}"
				title="{preset.name}: {preset.description}"
			>
				{#if preset.icon}
					<span class="text-xs">{preset.icon}</span>
				{/if}
				<span class="truncate max-w-[80px]">{preset.name}</span>
			</button>
		{/each}

		<!-- Expand / collapse -->
		{#if hasMore}
			<button
				on:click={() => {
					expanded = !expanded;
				}}
				class="px-1.5 py-0.5 rounded-md text-[10px] text-surface-500 hover:text-surface-300
					hover:bg-surface-800 transition-colors"
			>
				{expanded ? 'less' : `+${presetList.length - 5}`}
			</button>
		{/if}
	</div>
{:else if $optionsLoading}
	<div class="text-[10px] text-surface-500">Loading presets...</div>
{/if}
