<!--
  ModelSelector.svelte
  Dropdown to pick an Ollama model. Shows the resolved effective model
  under "Auto", groups models by family, and surfaces an approximate
  memory footprint per model (E1 RAM hint, from the on-disk size). Uses
  the ds Icon primitive.
-->
<script lang="ts">
	import Icon from '$lib/ds/Icon.svelte';
	import {
		availableModels,
		selectedModel,
		effectiveModel,
		effectiveModelSource,
		optionsLoading,
	} from '$lib/stores/chatOptions';

	let open = false;
	let dropdown: HTMLDivElement;

	$: displayName = $selectedModel ?? ($effectiveModel || 'Auto');
	$: shortName = displayName.length > 24 ? displayName.slice(0, 22) + '...' : displayName;
	$: isAuto = !$selectedModel;
	// E1: approximate footprint of the resolved model when on Auto.
	$: effectiveSize = $availableModels.find((m) => m.name === $effectiveModel)?.size ?? null;

	// Group by family.
	$: grouped = (() => {
		const groups: Record<string, typeof $availableModels> = {};
		for (const m of $availableModels) {
			const family = m.family || 'other';
			if (!groups[family]) groups[family] = [];
			groups[family].push(m);
		}
		// Sort each group by name.
		for (const key of Object.keys(groups)) {
			groups[key].sort((a, b) => a.name.localeCompare(b.name));
		}
		return groups;
	})();

	$: familyKeys = Object.keys(grouped).sort();

	function selectModel(name: string | null) {
		selectedModel.set(name);
		open = false;
	}

	function handleClickOutside(event: MouseEvent) {
		if (dropdown && !dropdown.contains(event.target as Node)) {
			open = false;
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			open = false;
		}
	}
</script>

<svelte:window on:click={handleClickOutside} on:keydown={handleKeydown} />

<div class="relative" bind:this={dropdown}>
	<button
		on:click|stopPropagation={() => {
			open = !open;
		}}
		disabled={$optionsLoading}
		class="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono transition-colors
			{isAuto
				? 'bg-surface-800/60 text-surface-400 hover:bg-surface-800 hover:text-surface-300'
				: 'bg-accent-600/15 text-accent-400 hover:bg-accent-600/25'}
			disabled:opacity-40 disabled:cursor-not-allowed"
		title="Select model ({$effectiveModelSource})"
	>
		<Icon name="cpu" size="sm" />
		<span class="truncate max-w-[120px] sm:max-w-[160px]">{shortName}</span>
		<span class="transition-transform {open ? 'rotate-180' : ''}"><Icon name="chevron-down" size="sm" /></span>
	</button>

	{#if open}
		<div
			class="absolute top-full left-0 mt-1 w-64 max-h-72 overflow-y-auto z-40
			border rounded-lg shadow-xl"
			style="background-color: var(--oo-bg-elevated); border-color: var(--oo-bd-subtle);"
		>
			<!-- Auto option -->
			<button
				on:click|stopPropagation={() => selectModel(null)}
				class="w-full text-left px-3 py-2 text-xs transition-colors
					{isAuto ? 'bg-accent-600/15 text-accent-400' : 'text-surface-300 hover:bg-surface-800'}"
			>
				<div class="flex items-center justify-between">
					<span class="font-medium">Auto</span>
					{#if isAuto}
						<span class="text-surface-500 font-mono text-[10px]">{$effectiveModelSource}</span>
					{/if}
				</div>
				{#if $effectiveModel}
					<div class="text-surface-500 font-mono text-[10px] mt-0.5 flex items-center justify-between gap-2">
						<span class="truncate">{$effectiveModel}</span>
						{#if effectiveSize}
							<span class="shrink-0" title="Approx. memory footprint">~{effectiveSize}</span>
						{/if}
					</div>
				{/if}
			</button>

			<div class="border-t" style="border-color: var(--oo-bd-subtle);" />

			{#if $availableModels.length === 0}
				<div class="px-3 py-3 text-xs text-surface-500 text-center">
					{$optionsLoading ? 'Loading models...' : 'No models available'}
				</div>
			{:else}
				{#each familyKeys as family}
					{#if familyKeys.length > 1}
						<div class="px-3 pt-2 pb-1 text-[10px] font-semibold uppercase tracking-wider text-surface-500">
							{family}
						</div>
					{/if}
					{#each grouped[family] as model}
						<button
							on:click|stopPropagation={() => selectModel(model.name)}
							class="w-full text-left px-3 py-1.5 text-xs transition-colors
								{$selectedModel === model.name
									? 'bg-accent-600/15 text-accent-400'
									: 'text-surface-300 hover:bg-surface-800'}"
						>
							<div class="flex items-center justify-between gap-2">
								<span class="font-mono truncate">{model.name}</span>
								<span class="flex items-center gap-1.5 shrink-0 text-surface-500 text-[10px]">
									{#if model.parameter_size}
										<span>{model.parameter_size}</span>
									{/if}
									{#if model.size}
										<span title="Approx. memory footprint">~{model.size}</span>
									{/if}
								</span>
							</div>
						</button>
					{/each}
				{/each}
			{/if}
		</div>
	{/if}
</div>
