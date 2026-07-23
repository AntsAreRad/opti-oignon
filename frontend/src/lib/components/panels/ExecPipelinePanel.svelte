<!--
  ExecPipelinePanel.svelte
  Panneau de gestion des pipelines d execution.
-->
<script lang="ts">
	import { onMount } from "svelte";
	import PipelineEditor from "./PipelineEditor.svelte";
	import { listExecPipelines, createExecPipeline, updateExecPipeline, deleteExecPipeline, duplicateExecPipeline, listStepTypes } from "$lib/api/execPipelines";
	import { selectedExecPipeline } from "$lib/stores/chatOptions";
	import { toastSuccess, toastError } from "$lib/stores/notifications";
	import type { ExecPipelineInfo, ExecStepInfo, ExecStepTypeInfo } from "$lib/types";

	let pipelines: ExecPipelineInfo[] = [];
	let stepTypes: ExecStepTypeInfo[] = [];
	let loading = true;
	let error: string | null = null;
	let view: "list" | "detail" | "create" = "list";
	let selectedPipeline: ExecPipelineInfo | null = null;
	let editMode = false;
	let formId = "";
	let formName = "";
	let formDescription = "";
	let formSteps: ExecStepInfo[] = [];
	let saving = false;
	let deleteConfirm: string | null = null;

	async function loadPipelines() {
		loading = true; error = null;
		try { pipelines = await listExecPipelines(); }
		catch (e) { error = e instanceof Error ? e.message : 'Failed to load'; }
		finally { loading = false; }
	}

	async function loadStepTypes() {
		try { stepTypes = await listStepTypes(); }
		catch { stepTypes = [
			{ type: 'direct', description: 'Direct response' },
			{ type: 'tools', description: 'Tool execution' },
			{ type: 'think', description: 'Chain-of-thought' },
			{ type: 'think_tools', description: 'Think + Tools' },
			{ type: 'web_search', description: 'Web search' },
			{ type: 'code_verify', description: 'Code verification' },
			{ type: 'reasoning', description: 'Structured reasoning' },
			{ type: 'consensus', description: 'Multi-model consensus' },
			{ type: 'self_correct', description: 'Self-correction' },
		]; }
	}

	onMount(() => { loadPipelines(); loadStepTypes(); });

	function showList() { view = 'list'; selectedPipeline = null; editMode = false; deleteConfirm = null; }
	function showDetail(p: ExecPipelineInfo) { selectedPipeline = p; view = 'detail'; editMode = false; }
	function showCreate() {
		formId = ''; formName = ''; formDescription = '';
		formSteps = [{ step_type: 'direct', label: 'Step 1', model_override: null, parameters: {}, condition: null, pass_previous_output: true }];
		view = 'create';
	}
	function startEdit() {
		if (!selectedPipeline) return;
		formId = selectedPipeline.id; formName = selectedPipeline.name;
		formDescription = selectedPipeline.description;
		formSteps = selectedPipeline.steps.map(s => ({ ...s }));
		editMode = true;
	}

	async function handleCreate() {
		if (!formId.trim() || !formName.trim()) { toastError('ID and name required'); return; }
		if (formSteps.length === 0) { toastError('At least one step required'); return; }
		saving = true;
		try {
			const c = await createExecPipeline({ id: formId.trim(), name: formName.trim(), description: formDescription.trim(), steps: formSteps });
			toastSuccess('Pipeline created'); await loadPipelines(); showDetail(c);
		} catch (e) { toastError(e instanceof Error ? e.message : 'Failed'); }
		finally { saving = false; }
	}

	async function handleUpdate() {
		if (!selectedPipeline) return; saving = true;
		try {
			const u = await updateExecPipeline(selectedPipeline.id, { name: formName.trim(), description: formDescription.trim(), steps: formSteps });
			toastSuccess('Updated'); await loadPipelines(); selectedPipeline = u; editMode = false;
		} catch (e) { toastError(e instanceof Error ? e.message : 'Failed'); }
		finally { saving = false; }
	}

	async function handleDelete(id: string) {
		try { await deleteExecPipeline(id); toastSuccess('Deleted'); await loadPipelines();
			if (selectedPipeline?.id === id) showList();
			if ($selectedExecPipeline === id) selectedExecPipeline.set(null);
		} catch (e) { toastError(e instanceof Error ? e.message : 'Failed'); }
		deleteConfirm = null;
	}

	async function handleDuplicate(p: ExecPipelineInfo) {
		try { await duplicateExecPipeline(p.id, p.id + '_copy_' + Date.now().toString(36));
			toastSuccess('Duplicated'); await loadPipelines();
		} catch (e) { toastError(e instanceof Error ? e.message : 'Failed'); }
	}

	function selectForChat(p: ExecPipelineInfo) {
		selectedExecPipeline.set($selectedExecPipeline === p.id ? null : p.id);
	}

	function handleStepsChange(e: CustomEvent<{ steps: ExecStepInfo[] }>) { formSteps = e.detail.steps; }
</script>

<div class="h-full flex flex-col">
	<div class="px-4 py-3 flex items-center gap-2" style="border-bottom: 1px solid var(--oo-bd-subtle);">
		{#if view !== 'list'}
			<button on:click={showList} class="p-1 rounded-md" style="color: var(--oo-fg-tertiary);" title="Back" aria-label="Back to pipeline list">
				<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M15 19l-7-7 7-7" /></svg>
			</button>
		{/if}
		<h2 class="text-sm font-medium flex-1" style="color: var(--oo-fg-secondary);">
			{#if view === 'list'}Execution Pipelines{:else if view === 'create'}New Pipeline{:else if editMode}Edit Pipeline{:else}{selectedPipeline?.name || 'Pipeline'}{/if}
		</h2>
		{#if view === 'list'}
			<button on:click={showCreate} class="p-1.5 rounded-md" style="color: var(--oo-acc-400); background-color: var(--oo-warning-bg);" title="New" aria-label="Create new pipeline execution">
				<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 4v16m8-8H4" /></svg>
			</button>
		{/if}
	</div>

	<div class="flex-1 overflow-y-auto">
		{#if loading}
			<div class="flex items-center justify-center h-32">
				<div class="animate-spin rounded-full h-6 w-6" style="border: 2px solid var(--oo-acc-400); border-top-color: transparent;" />
			</div>
		{:else if error}
			<div class="p-4">
				<p class="text-xs" style="color: var(--oo-error);">{error}</p>
				<button on:click={loadPipelines} class="text-xs mt-2" style="color: var(--oo-acc-400);">Retry</button>
			</div>

		{:else if view === 'list'}
			{#if pipelines.length === 0}
				<div class="p-4 text-center">
					<p class="text-sm" style="color: var(--oo-fg-muted);">No execution pipelines</p>
					<button on:click={showCreate} class="text-xs mt-2" style="color: var(--oo-acc-400);">Create your first pipeline</button>
				</div>
			{:else}
				{#if $selectedExecPipeline}
					<div class="mx-2 mt-2 px-3 py-2 rounded-lg text-xs flex items-center gap-2" style="background-color: var(--oo-warning-bg); border: 1px solid var(--oo-warning-bd);">
						<svg class="w-3.5 h-3.5 shrink-0" style="color: var(--oo-acc-400);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
						<span style="color: var(--oo-acc-300);">Active: {pipelines.find(p => p.id === $selectedExecPipeline)?.name || $selectedExecPipeline}</span>
						<button on:click={() => selectedExecPipeline.set(null)} class="ml-auto p-0.5 rounded" style="color: var(--oo-fg-muted);"><svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M6 18L18 6M6 6l12 12" /></svg></button>
					</div>
				{/if}
				<div class="p-2 space-y-1">
					{#each pipelines as pipeline}
						<div class="group flex items-center rounded-lg transition-colors" style="background-color: {$selectedExecPipeline === pipeline.id ? 'var(--oo-warning-bg)' : 'transparent'};">
							<button on:click={() => selectForChat(pipeline)} class="p-1.5 ml-1 rounded shrink-0" style="color: {$selectedExecPipeline === pipeline.id ? 'var(--oo-acc-400)' : 'var(--oo-fg-faint)'};" title={$selectedExecPipeline === pipeline.id ? 'Deselect' : 'Use'}>
								<svg class="w-3.5 h-3.5" fill={$selectedExecPipeline === pipeline.id ? 'currentColor' : 'none'} viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
							</button>
							<button on:click={() => showDetail(pipeline)} class="flex-1 flex items-center gap-2 px-2 py-2.5 text-left min-w-0">
								<div class="flex-1 min-w-0">
									<div class="text-xs font-medium truncate" style="color: var(--oo-fg-primary);">{pipeline.name}</div>
									<div class="text-xs truncate" style="color: var(--oo-fg-muted);">{pipeline.step_types_summary}{#if pipeline.is_builtin}<span style="color: var(--oo-acc-500); opacity: 0.6;"> builtin</span>{/if}</div>
								</div>
							</button>
							<div class="shrink-0 pr-2 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
								<button on:click|stopPropagation={() => handleDuplicate(pipeline)} class="p-1 rounded" style="color: var(--oo-fg-muted);" title="Duplicate">
									<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" /></svg>
								</button>
								{#if !pipeline.is_builtin}
									{#if deleteConfirm === pipeline.id}
										<button on:click|stopPropagation={() => handleDelete(pipeline.id)} class="p-1 rounded" style="color: var(--oo-error);"><svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M5 13l4 4L19 7" /></svg></button>
									{:else}
										<button on:click|stopPropagation={() => { deleteConfirm = pipeline.id; }} class="p-1 rounded" style="color: var(--oo-fg-muted);"><svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg></button>
									{/if}
								{/if}
							</div>
						</div>
					{/each}
				</div>
			{/if}

		{:else if view === 'detail' && selectedPipeline}
			<div class="p-4 space-y-4">
				{#if !editMode}
					<div><div class="text-xs" style="color: var(--oo-fg-muted);">ID</div><div class="text-sm font-mono" style="color: var(--oo-fg-tertiary);">{selectedPipeline.id}</div></div>
					{#if selectedPipeline.description}<div><div class="text-xs" style="color: var(--oo-fg-muted);">Description</div><div class="text-sm" style="color: var(--oo-fg-secondary);">{selectedPipeline.description}</div></div>{/if}
					<div class="flex gap-4">
						<div><div class="text-xs" style="color: var(--oo-fg-muted);">Steps</div><div class="text-sm" style="color: var(--oo-fg-secondary);">{selectedPipeline.step_count}</div></div>
						{#if selectedPipeline.is_builtin}<div><div class="text-xs" style="color: var(--oo-fg-muted);">Type</div><div class="text-xs" style="color: var(--oo-acc-400);">Built-in</div></div>{/if}
					</div>
					<div><div class="text-xs mb-2" style="color: var(--oo-fg-muted);">Pipeline Flow</div><PipelineEditor steps={selectedPipeline.steps} {stepTypes} readOnly={true} /></div>
					<div class="flex gap-2 pt-2">
						<button on:click={() => { if (selectedPipeline) selectForChat(selectedPipeline); }} class="flex-1 px-3 py-1.5 text-xs font-medium rounded-md" style="{$selectedExecPipeline === selectedPipeline?.id ? 'background-color: var(--oo-warning-bg); color: var(--oo-acc-300); border: 1px solid var(--oo-warning-bd);' : 'background-color: var(--oo-bg-elevated); color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-default);'}">{$selectedExecPipeline === selectedPipeline?.id ? 'Active' : 'Use for Chat'}</button>
						{#if !selectedPipeline.is_builtin}<button on:click={startEdit} class="flex-1 px-3 py-1.5 text-xs font-medium rounded-md" style="background-color: var(--oo-warning-bg); color: var(--oo-acc-400);">Edit</button>{/if}
						<button on:click={() => { if (selectedPipeline) handleDuplicate(selectedPipeline); }} class="px-3 py-1.5 text-xs font-medium rounded-md" style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-secondary);">Duplicate</button>
					</div>
				{:else}
					<div class="space-y-3">
						<div><label class="block text-xs mb-1" style="color: var(--oo-fg-tertiary);">Name</label><input type="text" bind:value={formName} class="w-full rounded-md px-2.5 py-1.5 text-xs outline-none" style="background-color: var(--oo-input-bg); color: var(--oo-fg-secondary); border: 1px solid var(--oo-input-bd);" /></div>
						<div><label class="block text-xs mb-1" style="color: var(--oo-fg-tertiary);">Description</label><textarea bind:value={formDescription} rows="2" class="w-full rounded-md px-2.5 py-1.5 text-xs outline-none resize-y" style="background-color: var(--oo-input-bg); color: var(--oo-fg-secondary); border: 1px solid var(--oo-input-bd);" /></div>
						<div><div class="text-xs mb-2" style="color: var(--oo-fg-tertiary);">Steps ({formSteps.length})</div><PipelineEditor steps={formSteps} {stepTypes} on:change={handleStepsChange} /></div>
						<div class="flex gap-2 pt-2">
							<button on:click={handleUpdate} disabled={saving} class="flex-1 px-3 py-1.5 text-xs font-medium rounded-md disabled:opacity-50" style="background-color: var(--oo-acc-500); color: white;">{saving ? 'Saving...' : 'Save'}</button>
							<button on:click={() => { editMode = false; }} class="px-3 py-1.5 text-xs font-medium rounded-md" style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-secondary);">Cancel</button>
						</div>
					</div>
				{/if}
			</div>

		{:else if view === 'create'}
			<div class="p-4 space-y-3">
				<div><label class="block text-xs mb-1" style="color: var(--oo-fg-tertiary);">Pipeline ID</label><input type="text" bind:value={formId} class="w-full rounded-md px-2.5 py-1.5 text-xs font-mono outline-none" style="background-color: var(--oo-input-bg); color: var(--oo-fg-secondary); border: 1px solid var(--oo-input-bd);" placeholder="my_pipeline" /></div>
				<div><label class="block text-xs mb-1" style="color: var(--oo-fg-tertiary);">Name</label><input type="text" bind:value={formName} class="w-full rounded-md px-2.5 py-1.5 text-xs outline-none" style="background-color: var(--oo-input-bg); color: var(--oo-fg-secondary); border: 1px solid var(--oo-input-bd);" placeholder="My Pipeline" /></div>
				<div><label class="block text-xs mb-1" style="color: var(--oo-fg-tertiary);">Description</label><textarea bind:value={formDescription} rows="2" class="w-full rounded-md px-2.5 py-1.5 text-xs outline-none resize-y" style="background-color: var(--oo-input-bg); color: var(--oo-fg-secondary); border: 1px solid var(--oo-input-bd);" placeholder="What does this pipeline do?" /></div>
				<div><div class="text-xs mb-2" style="color: var(--oo-fg-tertiary);">Steps ({formSteps.length})</div><PipelineEditor steps={formSteps} {stepTypes} on:change={handleStepsChange} /></div>
				<button on:click={handleCreate} disabled={saving || !formId.trim() || !formName.trim() || formSteps.length === 0} class="w-full px-3 py-2 text-xs font-medium rounded-md disabled:opacity-50" style="background-color: var(--oo-acc-500); color: white;">{saving ? 'Creating...' : 'Create Pipeline'}</button>
			</div>
		{/if}
	</div>
</div>
