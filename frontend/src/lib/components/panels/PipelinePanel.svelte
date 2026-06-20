<!--
  PipelinePanel.svelte
  Panneau de gestion des pipelines multi-agents.
  Liste, creation, edition, duplication, suppression, export.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import PipelineStepEditor from './PipelineStepEditor.svelte';
	import {
		listPipelines,
		getPipeline,
		createPipeline,
		updatePipeline,
		deletePipeline,
		duplicatePipeline,
		exportPipelines,
		listAgents
	} from '$lib/api/pipelines';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type { PipelineInfo, PipelineStepInfo, PipelineCreate } from '$lib/types';

	// -- Etat --
	let pipelines: PipelineInfo[] = [];
	let loading = true;
	let error: string | null = null;
	let agents: string[] = [];

	// Vue: 'list' | 'detail' | 'create'
	let view: 'list' | 'detail' | 'create' = 'list';
	let selectedPipeline: PipelineInfo | null = null;
	let editMode = false;

	// Formulaire creation/edition
	let formId = '';
	let formName = '';
	let formDescription = '';
	let formPattern = 'chain';
	let formEmoji = '';
	let formKeywords = '';
	let formSteps: PipelineStepInfo[] = [];
	let saving = false;

	// Confirmation suppression
	let deleteConfirm: string | null = null;

	const PATTERNS = ['chain', 'parallel', 'conditional', 'loop', 'map_reduce'];

	// -- Chargement --
	async function loadPipelines() {
		loading = true;
		error = null;
		try {
			pipelines = await listPipelines();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load pipelines';
		} finally {
			loading = false;
		}
	}

	async function loadAgents() {
		try {
			agents = await listAgents();
		} catch {
			agents = [];
		}
	}

	onMount(() => {
		loadPipelines();
		loadAgents();
	});

	// -- Navigation --
	function showList() {
		view = 'list';
		selectedPipeline = null;
		editMode = false;
	}

	function showDetail(pipeline: PipelineInfo) {
		selectedPipeline = pipeline;
		view = 'detail';
		editMode = false;
	}

	function showCreate() {
		formId = '';
		formName = '';
		formDescription = '';
		formPattern = 'chain';
		formEmoji = '';
		formKeywords = '';
		formSteps = [{ name: 'Step 1', agent: 'generate', prompt_template: null, description: '', system_prompt: null, model: null }];
		view = 'create';
	}

	function startEdit() {
		if (!selectedPipeline) return;
		formId = selectedPipeline.id;
		formName = selectedPipeline.name;
		formDescription = selectedPipeline.description;
		formPattern = selectedPipeline.pattern ?? 'chain';
		formEmoji = selectedPipeline.emoji;
		formKeywords = selectedPipeline.keywords.join(', ');
		formSteps = selectedPipeline.steps.map(s => ({ ...s }));
		editMode = true;
	}

	// -- Actions CRUD --
	async function handleCreate() {
		if (!formId.trim() || !formName.trim()) {
			toastError('ID and name are required');
			return;
		}
		saving = true;
		try {
			const config: PipelineCreate = {
				id: formId.trim(),
				name: formName.trim(),
				description: formDescription.trim(),
				pattern: formPattern,
				emoji: formEmoji.trim(),
				steps: formSteps,
				keywords: formKeywords.split(',').map(k => k.trim()).filter(Boolean)
			};
			const created = await createPipeline(config);
			toastSuccess(`Pipeline "${created.name}" created`);
			await loadPipelines();
			showDetail(created);
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to create pipeline');
		} finally {
			saving = false;
		}
	}

	async function handleUpdate() {
		if (!selectedPipeline) return;
		saving = true;
		try {
			const updated = await updatePipeline(selectedPipeline.id, {
				name: formName.trim(),
				description: formDescription.trim(),
				pattern: formPattern,
				emoji: formEmoji.trim(),
				steps: formSteps,
				keywords: formKeywords.split(',').map(k => k.trim()).filter(Boolean)
			});
			toastSuccess(`Pipeline "${updated.name}" updated`);
			await loadPipelines();
			selectedPipeline = updated;
			editMode = false;
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to update pipeline');
		} finally {
			saving = false;
		}
	}

	async function handleDelete(id: string) {
		try {
			await deletePipeline(id);
			toastSuccess('Pipeline deleted');
			await loadPipelines();
			if (selectedPipeline?.id === id) showList();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to delete pipeline');
		}
		deleteConfirm = null;
	}

	async function handleDuplicate(pipeline: PipelineInfo) {
		try {
			const newId = `${pipeline.id}_copy_${Date.now().toString(36)}`;
			const dup = await duplicatePipeline(pipeline.id, newId);
			toastSuccess(`Pipeline duplicated as "${dup.name}"`);
			await loadPipelines();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to duplicate pipeline');
		}
	}

	async function handleExport() {
		try {
			const data = await exportPipelines(false);
			const blob = new Blob([data.yaml_content || JSON.stringify(data, null, 2)], { type: 'text/yaml' });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = 'pipelines-export.yaml';
			a.click();
			URL.revokeObjectURL(url);
			toastSuccess('Pipelines exported');
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to export');
		}
	}

	// -- Steps management --
	function addStep() {
		formSteps = [...formSteps, {
			name: `Step ${formSteps.length + 1}`,
			agent: agents[0] || 'generate',
			prompt_template: null,
			description: '',
			system_prompt: null,
			model: null
		}];
	}

	function updateStep(e: CustomEvent<{ index: number; step: PipelineStepInfo }>) {
		const { index, step } = e.detail;
		formSteps = formSteps.map((s, i) => i === index ? step : s);
	}

	function removeStep(e: CustomEvent<{ index: number }>) {
		formSteps = formSteps.filter((_, i) => i !== e.detail.index);
	}
</script>

<div class="h-full flex flex-col">
	<!-- Header -->
	<div class="px-4 py-3 border-b border-surface-700/50 flex items-center gap-2">
		{#if view !== 'list'}
			<button
				on:click={showList}
				class="p-1 rounded-md text-surface-400 hover:text-surface-200 hover:bg-surface-800 transition-colors"
				title="Back to list"
			aria-label="Back to pipeline list"
			>
				<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M15 19l-7-7 7-7" />
				</svg>
			</button>
		{/if}

		<h2 class="text-sm font-medium text-surface-200 flex-1">
			{#if view === 'list'}Pipelines
			{:else if view === 'create'}New Pipeline
			{:else if editMode}Edit Pipeline
			{:else}{selectedPipeline?.emoji} {selectedPipeline?.name || 'Pipeline'}
			{/if}
		</h2>

		{#if view === 'list'}
			<button
				on:click={handleExport}
				class="p-1.5 rounded-md text-surface-400 hover:text-surface-200 hover:bg-surface-800 transition-colors"
				title="Export all pipelines"
			aria-label="Export all pipelines"
			>
				<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
				</svg>
			</button>
			<button
				on:click={showCreate}
				class="p-1.5 rounded-md text-accent-400 hover:text-accent-300 bg-accent-600/10 hover:bg-accent-600/20 transition-colors"
				title="Create new pipeline"
			aria-label="Create new pipeline"
			>
				<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M12 4v16m8-8H4" />
				</svg>
			</button>
		{/if}
	</div>

	<!-- Content -->
	<div class="flex-1 overflow-y-auto">
		{#if loading}
			<div class="flex items-center justify-center h-32">
				<div class="animate-spin rounded-full h-6 w-6 border-2 border-accent-500 border-t-transparent" />
			</div>
		{:else if error}
			<div class="p-4">
				<p class="text-xs text-[var(--oo-error)]">{error}</p>
				<button on:click={loadPipelines} class="text-xs text-accent-400 hover:text-accent-300 mt-2">
					Retry
				</button>
			</div>

		<!-- LIST VIEW -->
		{:else if view === 'list'}
			{#if pipelines.length === 0}
				<div class="p-4 text-center">
					<p class="text-sm text-surface-400">No pipelines configured</p>
					<button on:click={showCreate} class="text-xs text-accent-400 hover:text-accent-300 mt-2">
						Create your first pipeline
					</button>
				</div>
			{:else}
				<div class="p-2 space-y-1">
					{#each pipelines as pipeline}
						<div class="group flex items-center rounded-lg hover:bg-surface-800/50 transition-colors">
							<button
								on:click={() => showDetail(pipeline)}
								class="flex-1 flex items-center gap-2.5 px-3 py-2.5 text-left min-w-0"
							>
								<!-- Emoji -->
								<span class="text-base shrink-0">{pipeline.emoji || '🔗'}</span>

								<div class="flex-1 min-w-0">
									<div class="text-xs font-medium text-surface-200 truncate">{pipeline.name}</div>
									<div class="text-xs text-surface-500 truncate">
										{pipeline.step_count} step{pipeline.step_count !== 1 ? 's' : ''}
										{#if pipeline.pattern} &middot; {pipeline.pattern}{/if}
										{#if pipeline.is_builtin}
											<span class="text-accent-500/60">builtin</span>
										{/if}
									</div>
								</div>
							</button>

							<!-- Actions (hover) -->
							<div class="shrink-0 pr-2 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
								<button
									on:click|stopPropagation={() => handleDuplicate(pipeline)}
									class="p-1 rounded text-surface-500 hover:text-surface-300"
									title="Duplicate"
								>
									<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
										<rect x="9" y="9" width="13" height="13" rx="2" />
										<path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
									</svg>
								</button>
								{#if !pipeline.is_builtin}
									{#if deleteConfirm === pipeline.id}
										<button
											on:click|stopPropagation={() => handleDelete(pipeline.id)}
											class="p-1 rounded text-[var(--oo-error)] hover:text-[var(--oo-error)]"
											title="Confirm delete"
										>
											<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
												<path d="M5 13l4 4L19 7" />
											</svg>
										</button>
									{:else}
										<button
											on:click|stopPropagation={() => { deleteConfirm = pipeline.id; }}
											class="p-1 rounded text-surface-500 hover:text-[var(--oo-error)]"
											title="Delete"
										>
											<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
												<path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
											</svg>
										</button>
									{/if}
								{/if}
							</div>
						</div>
					{/each}
				</div>
			{/if}

		<!-- DETAIL VIEW -->
		{:else if view === 'detail' && selectedPipeline}
			<div class="p-4 space-y-4">
				{#if !editMode}
					<!-- Read-only detail -->
					<div>
						<div class="text-xs text-surface-500">ID</div>
						<div class="text-sm text-surface-300 font-mono">{selectedPipeline.id}</div>
					</div>

					{#if selectedPipeline.description}
						<div>
							<div class="text-xs text-surface-500">Description</div>
							<div class="text-sm text-surface-300">{selectedPipeline.description}</div>
						</div>
					{/if}

					<div class="flex gap-4">
						<div>
							<div class="text-xs text-surface-500">Pattern</div>
							<div class="text-sm text-surface-300">{selectedPipeline.pattern}</div>
						</div>
						<div>
							<div class="text-xs text-surface-500">Steps</div>
							<div class="text-sm text-surface-300">{selectedPipeline.step_count}</div>
						</div>
						{#if selectedPipeline.is_builtin}
							<div>
								<div class="text-xs text-surface-500">Type</div>
								<div class="text-xs text-accent-400">Built-in</div>
							</div>
						{/if}
					</div>

					{#if selectedPipeline.keywords.length > 0}
						<div>
							<div class="text-xs text-surface-500 mb-1">Keywords</div>
							<div class="flex flex-wrap gap-1">
								{#each selectedPipeline.keywords as kw}
									<span class="px-1.5 py-0.5 text-xs bg-surface-800 text-surface-400 rounded">{kw}</span>
								{/each}
							</div>
						</div>
					{/if}

					<!-- Steps (read-only) -->
					{#if selectedPipeline.steps.length > 0}
						<div>
							<div class="text-xs text-surface-500 mb-2">Steps</div>
							<div class="space-y-1.5">
								{#each selectedPipeline.steps as step, i}
									<div class="flex items-center gap-2 px-3 py-2 bg-surface-800/50 rounded-lg">
										<span class="text-xs text-accent-400 font-mono w-5 text-center">{i + 1}</span>
										<div class="flex-1 min-w-0">
											<div class="text-xs text-surface-200">{step.name}</div>
											<div class="text-xs text-surface-500">{step.agent}{step.model ? ` (${step.model})` : ''}</div>
										</div>
									</div>
								{/each}
							</div>
						</div>
					{/if}

					<!-- Actions -->
					<div class="flex gap-2 pt-2">
						{#if !selectedPipeline.is_builtin}
							<button
								on:click={startEdit}
								class="flex-1 px-3 py-1.5 text-xs font-medium rounded-md
									bg-accent-600/20 text-accent-400 hover:bg-accent-600/30 transition-colors"
							>
								Edit
							</button>
						{/if}
						<button
							on:click={() => { if (selectedPipeline) handleDuplicate(selectedPipeline); }}
							class="flex-1 px-3 py-1.5 text-xs font-medium rounded-md
								bg-surface-800 text-surface-300 hover:bg-surface-700 transition-colors"
						>
							Duplicate
						</button>
					</div>

				{:else}
					<!-- Edit form (same as create but with update) -->
					<div class="space-y-3">
						<div>
							<label class="block text-xs text-surface-400 mb-1" for="edit-name">Name</label>
							<input id="edit-name" type="text" bind:value={formName}
								class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
									text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500" />
						</div>
						<div>
							<label class="block text-xs text-surface-400 mb-1" for="edit-desc">Description</label>
							<textarea id="edit-desc" bind:value={formDescription} rows="2"
								class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
									text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500 resize-y" />
						</div>
						<div class="flex gap-2">
							<div class="flex-1">
								<label class="block text-xs text-surface-400 mb-1" for="edit-pattern">Pattern</label>
								<select id="edit-pattern" bind:value={formPattern}
									class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
										text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500">
									{#each PATTERNS as p}
										<option value={p}>{p}</option>
									{/each}
								</select>
							</div>
							<div class="w-16">
								<label class="block text-xs text-surface-400 mb-1" for="edit-emoji">Emoji</label>
								<input id="edit-emoji" type="text" bind:value={formEmoji} maxlength="4"
									class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
										text-xs text-center focus:outline-none focus:ring-1 focus:ring-accent-500" />
							</div>
						</div>
						<div>
							<label class="block text-xs text-surface-400 mb-1" for="edit-kw">Keywords (comma-separated)</label>
							<input id="edit-kw" type="text" bind:value={formKeywords}
								class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
									text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500"
								placeholder="keyword1, keyword2" />
						</div>

						<!-- Steps editor -->
						<div>
							<div class="flex items-center justify-between mb-2">
								<span class="text-xs text-surface-400">Steps ({formSteps.length})</span>
								<button on:click={addStep}
									class="text-xs text-accent-400 hover:text-accent-300 transition-colors">
									+ Add step
								</button>
							</div>
							<div class="space-y-2">
								{#each formSteps as step, i (i)}
									<PipelineStepEditor
										{step}
										index={i}
										{agents}
										removable={formSteps.length > 1}
										on:update={updateStep}
										on:remove={removeStep}
									/>
								{/each}
							</div>
						</div>

						<!-- Save / Cancel -->
						<div class="flex gap-2 pt-2">
							<button
								on:click={handleUpdate}
								disabled={saving}
								class="flex-1 px-3 py-1.5 text-xs font-medium rounded-md
									bg-accent-600 text-[var(--oo-btn-primary-fg)] hover:bg-accent-500 disabled:opacity-50 transition-colors"
							>
								{saving ? 'Saving...' : 'Save Changes'}
							</button>
							<button
								on:click={() => { editMode = false; }}
								class="px-3 py-1.5 text-xs font-medium rounded-md
									bg-surface-800 text-surface-300 hover:bg-surface-700 transition-colors"
							>
								Cancel
							</button>
						</div>
					</div>
				{/if}
			</div>

		<!-- CREATE VIEW -->
		{:else if view === 'create'}
			<div class="p-4 space-y-3">
				<div>
					<label class="block text-xs text-surface-400 mb-1" for="create-id">Pipeline ID</label>
					<input id="create-id" type="text" bind:value={formId}
						class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
							text-xs text-surface-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent-500"
						placeholder="my_custom_pipeline" />
				</div>
				<div>
					<label class="block text-xs text-surface-400 mb-1" for="create-name">Name</label>
					<input id="create-name" type="text" bind:value={formName}
						class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
							text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500"
						placeholder="My Custom Pipeline" />
				</div>
				<div>
					<label class="block text-xs text-surface-400 mb-1" for="create-desc">Description</label>
					<textarea id="create-desc" bind:value={formDescription} rows="2"
						class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
							text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500 resize-y"
						placeholder="Pipeline description" />
				</div>
				<div class="flex gap-2">
					<div class="flex-1">
						<label class="block text-xs text-surface-400 mb-1" for="create-pattern">Pattern</label>
						<select id="create-pattern" bind:value={formPattern}
							class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
								text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500">
							{#each PATTERNS as p}
								<option value={p}>{p}</option>
							{/each}
						</select>
					</div>
					<div class="w-16">
						<label class="block text-xs text-surface-400 mb-1" for="create-emoji">Emoji</label>
						<input id="create-emoji" type="text" bind:value={formEmoji} maxlength="4"
							class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
								text-xs text-center focus:outline-none focus:ring-1 focus:ring-accent-500"
							placeholder="🔗" />
					</div>
				</div>
				<div>
					<label class="block text-xs text-surface-400 mb-1" for="create-kw">Keywords</label>
					<input id="create-kw" type="text" bind:value={formKeywords}
						class="w-full bg-surface-800 border border-surface-700 rounded-md px-2.5 py-1.5
							text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-accent-500"
						placeholder="keyword1, keyword2" />
				</div>

				<!-- Steps editor -->
				<div>
					<div class="flex items-center justify-between mb-2">
						<span class="text-xs text-surface-400">Steps ({formSteps.length})</span>
						<button on:click={addStep}
							class="text-xs text-accent-400 hover:text-accent-300 transition-colors">
							+ Add step
						</button>
					</div>
					<div class="space-y-2">
						{#each formSteps as step, i (i)}
							<PipelineStepEditor
								{step}
								index={i}
								{agents}
								removable={formSteps.length > 1}
								on:update={updateStep}
								on:remove={removeStep}
							/>
						{/each}
					</div>
				</div>

				<!-- Create button -->
				<button
					on:click={handleCreate}
					disabled={saving || !formId.trim() || !formName.trim()}
					class="w-full px-3 py-2 text-xs font-medium rounded-md
						bg-accent-600 text-[var(--oo-btn-primary-fg)] hover:bg-accent-500 disabled:opacity-50 transition-colors"
				>
					{saving ? 'Creating...' : 'Create Pipeline'}
				</button>
			</div>
		{/if}
	</div>
</div>
