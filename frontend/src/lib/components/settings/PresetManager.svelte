<!--
  PresetManager.svelte
  Gestion complete des presets: liste, creation, edition, duplication, suppression.
  Inclut recherche/filtre, test de keywords, et formulaire inline.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import type { PresetInfo } from '$lib/types';
	import {
		listPresets,
		createPreset,
		updatePreset,
		deletePreset,
		duplicatePreset,
		searchPresets,
		matchPreset
	} from '$lib/api/presets';

	let presets: PresetInfo[] = [];
	let filtered: PresetInfo[] = [];
	let searchQuery = '';
	let loading = true;
	let error = '';

	// Formulaire d'edition/creation
	let editing: PresetInfo | null = null;
	let creating = false;
	let formData: Partial<PresetInfo> = {};

	// Test de keywords
	let keywordTestInput = '';
	let keywordTestResult: PresetInfo | null = null;
	let keywordTesting = false;

	// Confirmation de suppression
	let confirmDeleteId: string | null = null;

	const defaultForm: Partial<PresetInfo> = {
		id: '',
		name: '',
		description: '',
		task: 'general',
		model: '',
		temperature: 0.7,
		prompt_variant: 'standard',
		icon: '',
		tags: [],
		keywords: [],
		detection_weight: 1.0,
		custom_prompt: null
	};

	async function load() {
		loading = true;
		error = '';
		try {
			presets = await listPresets();
			applyFilter();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load presets';
		} finally {
			loading = false;
		}
	}

	function applyFilter() {
		if (!searchQuery.trim()) {
			filtered = presets;
		} else {
			const q = searchQuery.toLowerCase();
			filtered = presets.filter(
				(p) =>
					p.name.toLowerCase().includes(q) ||
					p.task.toLowerCase().includes(q) ||
					p.description.toLowerCase().includes(q) ||
					p.tags.some((t) => t.toLowerCase().includes(q))
			);
		}
	}

	$: searchQuery, applyFilter();

	function startCreate() {
		editing = null;
		creating = true;
		formData = { ...defaultForm };
	}

	function startEdit(preset: PresetInfo) {
		creating = false;
		editing = preset;
		formData = { ...preset, tags: [...preset.tags], keywords: [...preset.keywords] };
	}

	function cancelEdit() {
		editing = null;
		creating = false;
		formData = {};
	}

	async function savePreset() {
		error = '';
		try {
			if (creating) {
				await createPreset(formData);
			} else if (editing) {
				await updatePreset(editing.id, formData);
			}
			cancelEdit();
			await load();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to save preset';
		}
	}

	async function handleDelete(id: string) {
		error = '';
		try {
			await deletePreset(id);
			confirmDeleteId = null;
			await load();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to delete preset';
		}
	}

	async function handleDuplicate(id: string) {
		error = '';
		try {
			await duplicatePreset(id);
			await load();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to duplicate preset';
		}
	}

	async function testKeywords() {
		if (!keywordTestInput.trim()) return;
		keywordTesting = true;
		keywordTestResult = null;
		try {
			keywordTestResult = await matchPreset(keywordTestInput);
		} catch {
			keywordTestResult = null;
		} finally {
			keywordTesting = false;
		}
	}

	function handleTagsInput(e: Event) {
		const val = (e.target as HTMLInputElement).value;
		formData.tags = val
			.split(',')
			.map((s) => s.trim())
			.filter(Boolean);
	}

	function handleKeywordsInput(e: Event) {
		const val = (e.target as HTMLInputElement).value;
		formData.keywords = val
			.split(',')
			.map((s) => s.trim())
			.filter(Boolean);
	}

	onMount(load);
</script>

<div class="space-y-4">
	<!-- Toolbar: search + create -->
	<div class="flex flex-col sm:flex-row gap-2">
		<input
			type="text"
			bind:value={searchQuery}
			placeholder="Filter presets..."
			class="flex-1 px-3 py-1.5 rounded bg-surface-800 border border-surface-700 text-surface-200 text-sm
				focus:outline-none focus:border-accent-500"
		/>
		<button
			on:click={startCreate}
			class="px-3 py-1.5 rounded bg-accent-600 hover:bg-accent-500 text-[var(--oo-btn-primary-fg)] text-sm font-medium shrink-0"
		>
			+ New Preset
		</button>
	</div>

	<!-- Keyword test -->
	<div class="flex gap-2 items-center">
		<input
			type="text"
			bind:value={keywordTestInput}
			placeholder="Test keyword matching..."
			class="flex-1 px-3 py-1.5 rounded bg-surface-800 border border-surface-700 text-surface-200 text-sm
				focus:outline-none focus:border-accent-500"
			on:keydown={(e) => e.key === 'Enter' && testKeywords()}
		/>
		<button
			on:click={testKeywords}
			disabled={keywordTesting}
			class="px-3 py-1.5 rounded bg-surface-700 hover:bg-surface-600 text-surface-200 text-sm shrink-0
				disabled:opacity-50"
		>
			{keywordTesting ? 'Testing...' : 'Test'}
		</button>
		{#if keywordTestResult}
			<span class="text-xs text-accent-400">
				Match: {keywordTestResult.icon} {keywordTestResult.name}
			</span>
		{:else if keywordTestInput && !keywordTesting && keywordTestResult === null}
			<span class="text-xs text-surface-500">No match</span>
		{/if}
	</div>

	<!-- Error display -->
	{#if error}
		<div class="px-3 py-2 rounded bg-[var(--oo-error-bg)] border border-[var(--oo-error-bd)] text-[var(--oo-error)] text-sm">
			{error}
		</div>
	{/if}

	<!-- Create/Edit form -->
	{#if creating || editing}
		<div class="p-4 rounded-lg bg-surface-800 border border-surface-700 space-y-3">
			<h3 class="text-sm font-medium text-surface-200">
				{creating ? 'New Preset' : `Edit: ${editing?.name}`}
			</h3>

			<div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
				{#if creating}
					<div>
						<label class="block text-xs text-surface-400 mb-1" for="preset-id">ID</label>
						<input id="preset-id"
							type="text"
							bind:value={formData.id}
							class="w-full px-2 py-1.5 rounded bg-surface-900 border border-surface-700 text-surface-200 text-sm
								focus:outline-none focus:border-accent-500"
							placeholder="unique-id"
						/>
					</div>
				{/if}
				<div>
					<label class="block text-xs text-surface-400 mb-1" for="preset-name">Name</label>
					<input id="preset-name"
						type="text"
						bind:value={formData.name}
						class="w-full px-2 py-1.5 rounded bg-surface-900 border border-surface-700 text-surface-200 text-sm
							focus:outline-none focus:border-accent-500"
					/>
				</div>
				<div>
					<label class="block text-xs text-surface-400 mb-1" for="preset-task">Task</label>
					<input id="preset-task"
						type="text"
						bind:value={formData.task}
						class="w-full px-2 py-1.5 rounded bg-surface-900 border border-surface-700 text-surface-200 text-sm
							focus:outline-none focus:border-accent-500"
					/>
				</div>
				<div>
					<label class="block text-xs text-surface-400 mb-1" for="preset-model">Model</label>
					<input id="preset-model"
						type="text"
						bind:value={formData.model}
						class="w-full px-2 py-1.5 rounded bg-surface-900 border border-surface-700 text-surface-200 text-sm
							focus:outline-none focus:border-accent-500"
					/>
				</div>
				<div>
					<label class="block text-xs text-surface-400 mb-1" for="preset-temperature">Temperature</label>
					<input id="preset-temperature"
						type="number"
						step="0.1"
						min="0"
						max="2"
						bind:value={formData.temperature}
						class="w-full px-2 py-1.5 rounded bg-surface-900 border border-surface-700 text-surface-200 text-sm
							focus:outline-none focus:border-accent-500"
					/>
				</div>
				<div>
					<label class="block text-xs text-surface-400 mb-1" for="preset-icon">Icon</label>
					<input id="preset-icon"
						type="text"
						bind:value={formData.icon}
						class="w-full px-2 py-1.5 rounded bg-surface-900 border border-surface-700 text-surface-200 text-sm
							focus:outline-none focus:border-accent-500"
						placeholder="emoji"
					/>
				</div>
				<div>
					<label class="block text-xs text-surface-400 mb-1" for="preset-prompt-variant">Prompt Variant</label>
					<input id="preset-prompt-variant"
						type="text"
						bind:value={formData.prompt_variant}
						class="w-full px-2 py-1.5 rounded bg-surface-900 border border-surface-700 text-surface-200 text-sm
							focus:outline-none focus:border-accent-500"
					/>
				</div>
				<div>
					<label class="block text-xs text-surface-400 mb-1" for="preset-detection-weight">Detection Weight</label>
					<input id="preset-detection-weight"
						type="number"
						step="0.1"
						min="0"
						max="10"
						bind:value={formData.detection_weight}
						class="w-full px-2 py-1.5 rounded bg-surface-900 border border-surface-700 text-surface-200 text-sm
							focus:outline-none focus:border-accent-500"
					/>
				</div>
			</div>

			<div>
				<label class="block text-xs text-surface-400 mb-1" for="preset-description">Description</label>
				<textarea id="preset-description"
					bind:value={formData.description}
					rows="2"
					class="w-full px-2 py-1.5 rounded bg-surface-900 border border-surface-700 text-surface-200 text-sm
						resize-none focus:outline-none focus:border-accent-500"
				/>
			</div>

			<div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
				<div>
					<label class="block text-xs text-surface-400 mb-1" for="preset-tags">Tags (comma separated)</label>
					<input id="preset-tags"
						type="text"
						value={formData.tags?.join(', ') ?? ''}
						on:input={handleTagsInput}
						class="w-full px-2 py-1.5 rounded bg-surface-900 border border-surface-700 text-surface-200 text-sm
							focus:outline-none focus:border-accent-500"
					/>
				</div>
				<div>
					<label class="block text-xs text-surface-400 mb-1" for="preset-keywords">Keywords (comma separated)</label>
					<input id="preset-keywords"
						type="text"
						value={formData.keywords?.join(', ') ?? ''}
						on:input={handleKeywordsInput}
						class="w-full px-2 py-1.5 rounded bg-surface-900 border border-surface-700 text-surface-200 text-sm
							focus:outline-none focus:border-accent-500"
					/>
				</div>
			</div>

			<div>
				<label class="block text-xs text-surface-400 mb-1" for="preset-custom-prompt">Custom Prompt</label>
				<textarea id="preset-custom-prompt"
					bind:value={formData.custom_prompt}
					rows="3"
					class="w-full px-2 py-1.5 rounded bg-surface-900 border border-surface-700 text-surface-200 text-sm
						resize-none font-mono focus:outline-none focus:border-accent-500"
					placeholder="Optional custom system prompt..."
				/>
			</div>

			<div class="flex gap-2 justify-end">
				<button
					on:click={cancelEdit}
					class="px-3 py-1.5 rounded bg-surface-700 hover:bg-surface-600 text-surface-300 text-sm"
				>
					Cancel
				</button>
				<button
					on:click={savePreset}
					class="px-3 py-1.5 rounded bg-accent-600 hover:bg-accent-500 text-[var(--oo-btn-primary-fg)] text-sm font-medium"
				>
					{creating ? 'Create' : 'Save'}
				</button>
			</div>
		</div>
	{/if}

	<!-- Presets list -->
	{#if loading}
		<div class="flex items-center gap-2 text-surface-400 text-sm py-4">
			<div class="w-4 h-4 border-2 border-surface-600 border-t-accent-500 rounded-full animate-spin" />
			Loading presets...
		</div>
	{:else if filtered.length === 0}
		<p class="text-surface-500 text-sm py-4">
			{searchQuery ? 'No presets match your filter.' : 'No presets configured.'}
		</p>
	{:else}
		<div class="space-y-2">
			{#each filtered as preset (preset.id)}
				<div class="p-3 rounded-lg bg-surface-800/60 border border-surface-700/50 hover:border-surface-600 transition-colors">
					<div class="flex items-start gap-3">
						<!-- Icon + info -->
						<span class="text-lg shrink-0">{preset.icon || '⚙'}</span>
						<div class="flex-1 min-w-0">
							<div class="flex items-center gap-2 flex-wrap">
								<span class="text-sm font-medium text-surface-200">{preset.name}</span>
								<span class="text-xs px-1.5 py-0.5 rounded bg-surface-700 text-surface-400 font-mono">
									{preset.id}
								</span>
								<span class="text-xs px-1.5 py-0.5 rounded bg-accent-900/40 text-accent-400">
									{preset.task}
								</span>
							</div>
							{#if preset.description}
								<p class="text-xs text-surface-500 mt-0.5 line-clamp-1">{preset.description}</p>
							{/if}
							<div class="flex items-center gap-3 mt-1 text-xs text-surface-500">
								{#if preset.model}
									<span>Model: {preset.model}</span>
								{/if}
								<span>Temp: {preset.temperature}</span>
								{#if preset.keywords.length > 0}
									<span>Keywords: {preset.keywords.length}</span>
								{/if}
							</div>
						</div>

						<!-- Actions -->
						<div class="flex items-center gap-1 shrink-0">
							<button
								on:click={() => startEdit(preset)}
								class="p-1.5 rounded text-surface-400 hover:text-surface-200 hover:bg-surface-700"
								title="Edit"
							>
								<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
									<path d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
								</svg>
							</button>
							<button
								on:click={() => handleDuplicate(preset.id)}
								class="p-1.5 rounded text-surface-400 hover:text-surface-200 hover:bg-surface-700"
								title="Duplicate"
							>
								<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
									<path d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
								</svg>
							</button>
							{#if confirmDeleteId === preset.id}
								<button
									on:click={() => handleDelete(preset.id)}
									class="px-2 py-1 rounded bg-[var(--oo-error-bg)] hover:bg-[var(--oo-error)] text-[var(--oo-error)] text-xs"
								>
									Confirm
								</button>
								<button
									on:click={() => (confirmDeleteId = null)}
									class="px-2 py-1 rounded bg-surface-700 hover:bg-surface-600 text-surface-300 text-xs"
								>
									Cancel
								</button>
							{:else}
								<button
									on:click={() => (confirmDeleteId = preset.id)}
									class="p-1.5 rounded text-surface-400 hover:text-[var(--oo-error)] hover:bg-surface-700"
									title="Delete"
								>
									<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
										<path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
									</svg>
								</button>
							{/if}
						</div>
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>
