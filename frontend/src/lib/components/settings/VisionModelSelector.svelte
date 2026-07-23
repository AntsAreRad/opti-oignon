<!--
  VisionModelSelector.svelte -- Vision Model settings panel.

  Sections:
  1. Selected vision model dropdown (auto or explicit)
  2. Detected vision models list with detection method badges
  3. Known vision models manual override input
  4. Refresh cache button
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getVisionConfig,
		updateVisionConfig,
		clearVisionCache,
	} from '$lib/api/vision';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type { VisionConfig } from '$lib/api/vision';

	let loading = true;
	let error = '';
	let saving = false;
	let refreshing = false;

	let config: VisionConfig | null = null;
	let selectedModel = 'auto';
	let knownModelsInput = '';

	// All models for the dropdown (fetched from parent or vision endpoint)
	let allModels: string[] = [];

	onMount(loadConfig);

	async function loadConfig() {
		loading = true;
		error = '';
		try {
			config = await getVisionConfig();
			selectedModel = config.vision_model;
			knownModelsInput = config.known_vision_models.join(', ');

			// Build full dropdown: "auto" + detected vision models + all available
			// We get available_vision_models from the config response
			allModels = config.available_vision_models;
		} catch (e) {
			error = `Failed to load vision config: ${e}`;
		} finally {
			loading = false;
		}
	}

	async function handleModelChange() {
		saving = true;
		try {
			config = await updateVisionConfig({ vision_model: selectedModel });
			toastSuccess('Vision model updated');
		} catch (e) {
			toastError(`Failed to update vision model: ${e}`);
		} finally {
			saving = false;
		}
	}

	async function handleKnownModelsUpdate() {
		saving = true;
		try {
			const models = knownModelsInput
				.split(',')
				.map((m) => m.trim())
				.filter((m) => m.length > 0);
			config = await updateVisionConfig({ known_vision_models: models });
			knownModelsInput = config.known_vision_models.join(', ');
			allModels = config.available_vision_models;
			toastSuccess('Known vision models updated');
		} catch (e) {
			toastError(`Failed to update known models: ${e}`);
		} finally {
			saving = false;
		}
	}

	async function handleRefreshCache() {
		refreshing = true;
		try {
			await clearVisionCache();
			await loadConfig();
			toastSuccess('Vision cache cleared, models re-detected');
		} catch (e) {
			toastError(`Failed to refresh: ${e}`);
		} finally {
			refreshing = false;
		}
	}

	function badgeStyle(method: string): string {
		if (method === 'capability') return 'background-color: var(--oo-success-bg); color: var(--oo-success);';
		if (method === 'manual') return 'background-color: var(--oo-tobacco-bg); color: var(--oo-tobacco);';
		return 'background-color: var(--oo-bg-elevated); color: var(--oo-fg-muted);';
	}

	function badgeLabel(method: string): string {
		if (method === 'capability') return 'ollama';
		if (method === 'manual') return 'manual';
		if (method === 'pattern') return 'name';
		return method;
	}
</script>

<div class="space-y-4">
	<div class="flex items-center justify-between">
		<div>
			<h3 class="text-sm font-medium" style="color: var(--oo-fg-primary);">Vision Model</h3>
			<p class="text-xs mt-0.5" style="color: var(--oo-fg-muted);">
				Select which model handles image analysis. Auto mode detects vision-capable models
				via Ollama capabilities and name patterns.
			</p>
		</div>
		<button
			on:click={handleRefreshCache}
			disabled={refreshing || loading}
			class="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs transition-colors"
			style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default); color: var(--oo-fg-secondary);"
			title="Re-detect vision models"
		>
			<svg class="w-3.5 h-3.5 {refreshing ? 'animate-spin' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M1 4v6h6" /><path d="M3.51 15a9 9 0 102.13-9.36L1 10" />
			</svg>
			{refreshing ? 'Detecting...' : 'Refresh'}
		</button>
	</div>

	{#if loading}
		<div class="text-xs py-4 text-center" style="color: var(--oo-fg-muted);">Loading vision config...</div>
	{:else if error}
		<div class="text-xs py-2 rounded-lg px-3" style="background-color: var(--oo-error-bg); color: var(--oo-error);">
			{error}
		</div>
	{:else if config}
		<!-- Model selector -->
		<div class="flex items-center gap-3">
			<label for="vision-model-select" class="text-xs font-medium shrink-0" style="color: var(--oo-fg-secondary);">
				Active model
			</label>
			<select
				id="vision-model-select"
				bind:value={selectedModel}
				on:change={handleModelChange}
				disabled={saving}
				class="flex-1 text-xs rounded-lg px-3 py-1.5"
				style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default); color: var(--oo-fg-primary);"
			>
				<option value="auto">Auto-detect</option>
				{#each allModels as model}
					<option value={model}>{model}</option>
				{/each}
			</select>
		</div>

		<!-- Effective model display -->
		{#if config.effective_model}
			<div class="flex items-center gap-2 text-xs" style="color: var(--oo-fg-muted);">
				<span>Currently using:</span>
				<span class="font-mono font-medium" style="color: var(--oo-tobacco);">{config.effective_model}</span>
			</div>
		{:else}
			<div class="text-xs" style="color: var(--oo-warning);">
				No vision model detected. Attach a vision-capable model or add one to the known list below.
			</div>
		{/if}

		<!-- Detected vision models -->
		{#if config.available_vision_models.length > 0}
			<div>
				<div class="text-xs font-medium mb-1.5" style="color: var(--oo-fg-secondary);">
					Detected vision models ({config.available_vision_models.length})
				</div>
				<div class="flex flex-wrap gap-1.5">
					{#each config.available_vision_models as model}
						<span
							class="inline-flex items-center gap-1.5 text-xs px-2 py-1 rounded-lg font-mono"
							style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle); color: var(--oo-fg-primary);"
						>
							{model}
						</span>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Known vision models manual input -->
		<div>
			<label for="known-vision-models" class="text-xs font-medium block mb-1" style="color: var(--oo-fg-secondary);">
				Manual vision models
			</label>
			<p class="text-xs mb-1.5" style="color: var(--oo-fg-muted);">
				Comma-separated list of models to always treat as vision-capable (e.g. qwen3.5:32b).
			</p>
			<div class="flex gap-2">
				<input
					id="known-vision-models"
					type="text"
					bind:value={knownModelsInput}
					placeholder="qwen3.5:32b, my-model:latest"
					class="flex-1 text-xs rounded-lg px-3 py-1.5"
					style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default); color: var(--oo-fg-primary);"
				/>
				<button
					on:click={handleKnownModelsUpdate}
					disabled={saving}
					class="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
					style="background-color: var(--oo-tobacco-bg); border: 1px solid var(--oo-tobacco-bd); color: var(--oo-tobacco);"
				>
					{saving ? 'Saving...' : 'Save'}
				</button>
			</div>
		</div>

		<!-- Detection info -->
		<div class="text-xs pt-1" style="color: var(--oo-fg-faint);">
			Strategy: {config.detection_strategy} |
			Families: {config.vision_families.join(', ')} |
			Patterns: {config.auto_detect_patterns.join(', ')}
		</div>
	{/if}
</div>
