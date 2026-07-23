<!--
  PromptConfigPanel — Prompt Intelligence settings.

  Displays:
  - Current model context window + budget allocation bar
  - Template list with view/edit per task type
  - Enable/disable toggle
  - Cache stats
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getBudget,
		listTemplates,
		getTemplate,
		setTemplateOverride,
		clearTemplateOverride,
		clearAllOverrides,
		getCacheStats,
		clearCache,
		reloadPromptConfig,
		type TokenBudget,
		type TemplateSummary,
		type PromptTemplate,
		type CacheStats
	} from '$lib/api/promptConfig';
	import { toastSuccess, toastError } from '$lib/stores/notifications';

	// State
	let loading = true;
	let error = '';
	let budget: TokenBudget | null = null;
	let templates: TemplateSummary[] = [];
	let cacheStats: CacheStats | null = null;

	// Budget model selector
	let budgetModel = 'qwen3-coder:30b';
	let projectActive = false;

	// Template viewer/editor
	let selectedTemplate: PromptTemplate | null = null;
	let editingTemplate = false;
	let editPrompt = '';
	let editTemp: string = '';

	async function loadData() {
		loading = true;
		error = '';
		try {
			const [tpls, stats] = await Promise.all([listTemplates(), getCacheStats()]);
			templates = tpls;
			cacheStats = stats;
			await loadBudget();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load prompt config';
		} finally {
			loading = false;
		}
	}

	async function loadBudget() {
		try {
			budget = await getBudget(budgetModel, projectActive);
		} catch (e) {
			budget = null;
		}
	}

	async function handleModelChange() {
		await loadBudget();
	}

	async function handleProjectToggle() {
		await loadBudget();
	}

	async function viewTemplate(taskType: string) {
		try {
			selectedTemplate = await getTemplate(taskType);
			editingTemplate = false;
			editPrompt = selectedTemplate.system_prompt;
			editTemp = selectedTemplate.temperature_override !== null
				? String(selectedTemplate.temperature_override)
				: '';
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to load template');
		}
	}

	async function saveOverride() {
		if (!selectedTemplate) return;
		try {
			const body: { system_prompt: string; temperature_override?: number | null } = {
				system_prompt: editPrompt
			};
			if (editTemp.trim() !== '') {
				body.temperature_override = parseFloat(editTemp);
			}
			const updated = await setTemplateOverride(selectedTemplate.task_type, body);
			selectedTemplate = updated;
			editingTemplate = false;
			toastSuccess(`Override saved for "${selectedTemplate.task_type}"`);
			templates = await listTemplates();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to save override');
		}
	}

	async function removeOverride() {
		if (!selectedTemplate) return;
		try {
			await clearTemplateOverride(selectedTemplate.task_type);
			toastSuccess(`Override cleared for "${selectedTemplate.task_type}"`);
			selectedTemplate = await getTemplate(selectedTemplate.task_type);
			editingTemplate = false;
			templates = await listTemplates();
		} catch {
			toastError('No override to clear');
		}
	}

	async function handleClearAllOverrides() {
		try {
			const res = await clearAllOverrides();
			toastSuccess(`Cleared ${res.cleared} override(s)`);
			templates = await listTemplates();
			if (selectedTemplate) {
				selectedTemplate = await getTemplate(selectedTemplate.task_type);
			}
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to clear overrides');
		}
	}

	async function handleClearCache() {
		try {
			const res = await clearCache();
			toastSuccess(`Cleared ${res.cleared} cached entries`);
			cacheStats = await getCacheStats();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to clear cache');
		}
	}

	async function handleReload() {
		try {
			await reloadPromptConfig();
			toastSuccess('Prompt config reloaded from disk');
			await loadData();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to reload config');
		}
	}

	// Budget bar segment helper
	function pct(value: number, total: number): string {
		if (total <= 0) return '0%';
		return Math.max(0, Math.min(100, (value / total) * 100)).toFixed(1) + '%';
	}

	// Source badge color
	function sourceColor(source: string): string {
		switch (source) {
			case 'runtime': return 'color: var(--oo-acc-400);';
			case 'project': return 'color: var(--oo-pipe-tools);';
			case 'fallback': return 'color: var(--oo-fg-tertiary);';
			default: return 'color: var(--oo-acc-400);';
		}
	}

	onMount(loadData);
</script>

<div class="space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<h2 class="text-base font-medium" style="color: var(--oo-fg-primary);">
			Prompt Intelligence
		</h2>
		<button
			on:click={handleReload}
			class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
			style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-default);"
		>
			Reload Config
		</button>
	</div>

	{#if loading}
		<div class="flex items-center gap-2 text-sm py-8 justify-center" style="color: var(--oo-fg-tertiary);">
			<div class="w-5 h-5 border-2 rounded-full animate-spin"
				style="border-color: var(--oo-bd-default); border-top-color: var(--oo-acc-500);" />
			Loading prompt config...
		</div>
	{:else if error}
		<div class="px-3 py-2 rounded text-sm mb-4"
			style="background-color: var(--oo-error-bg); border: 1px solid var(--oo-error-bd); color: var(--oo-error);">
			{error}
			<button on:click={loadData} class="ml-2 underline hover:no-underline">Retry</button>
		</div>
	{:else}
		<!-- Token Budget Section -->
		<div class="p-4 rounded-lg" style="background-color: var(--oo-bg-secondary); border: 1px solid var(--oo-bd-default);">
			<h3 class="text-sm font-medium mb-3" style="color: var(--oo-fg-primary);">Token Budget</h3>

			<!-- Model selector -->
			<div class="flex items-center gap-3 mb-3">
				<input
					type="text"
					bind:value={budgetModel}
					on:change={handleModelChange}
					class="flex-1 px-3 py-1.5 rounded text-sm"
					style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
					placeholder="Model name (e.g. qwen3:32b)"
				/>
				<label class="flex items-center gap-2 text-xs cursor-pointer" style="color: var(--oo-fg-secondary);">
					<input
						type="checkbox"
						bind:checked={projectActive}
						on:change={handleProjectToggle}
						class="rounded"
					/>
					Project active
				</label>
			</div>

			{#if budget}
				<!-- Context window info -->
				<div class="flex items-center justify-between text-xs mb-2" style="color: var(--oo-fg-tertiary);">
					<span>Context window: <strong style="color: var(--oo-fg-primary);">{budget.total_window.toLocaleString()}</strong> tokens</span>
					<span>Utilization: {(budget.utilization * 100).toFixed(1)}%</span>
				</div>

				<!-- Stacked budget bar -->
				<div class="w-full h-6 rounded overflow-hidden flex" style="background-color: var(--oo-bg-tertiary);">
					<div
						class="h-full flex items-center justify-center text-[10px] font-medium"
						style="width: {pct(budget.system_tokens, budget.total_window)}; background-color: var(--oo-pipe-direct); color: white;"
						title="System: {budget.system_tokens.toLocaleString()} tokens"
					>
						{#if budget.system_tokens / budget.total_window > 0.06}Sys{/if}
					</div>
					{#if budget.project_tokens > 0}
						<div
							class="h-full flex items-center justify-center text-[10px] font-medium"
							style="width: {pct(budget.project_tokens, budget.total_window)}; background-color: var(--oo-pipe-tools); color: white;"
							title="Project: {budget.project_tokens.toLocaleString()} tokens"
						>
							{#if budget.project_tokens / budget.total_window > 0.06}Proj{/if}
						</div>
					{/if}
					<div
						class="h-full flex items-center justify-center text-[10px] font-medium"
						style="width: {pct(budget.history_tokens, budget.total_window)}; background-color: var(--oo-success); color: white;"
						title="History: {budget.history_tokens.toLocaleString()} tokens"
					>
						{#if budget.history_tokens / budget.total_window > 0.06}History{/if}
					</div>
					<div
						class="h-full flex items-center justify-center text-[10px] font-medium"
						style="width: {pct(budget.user_tokens, budget.total_window)}; background-color: var(--oo-acc-400); color: white;"
						title="User: {budget.user_tokens.toLocaleString()} tokens"
					>
						{#if budget.user_tokens / budget.total_window > 0.06}User{/if}
					</div>
					<div
						class="h-full flex items-center justify-center text-[10px] font-medium"
						style="width: {pct(budget.reserve_tokens, budget.total_window)}; background-color: var(--oo-fg-tertiary); color: white;"
						title="Reserve: {budget.reserve_tokens.toLocaleString()} tokens"
					>
						{#if budget.reserve_tokens / budget.total_window > 0.06}Res{/if}
					</div>
				</div>

				<!-- Legend -->
				<div class="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-[11px]" style="color: var(--oo-fg-tertiary);">
					<span><span class="inline-block w-2.5 h-2.5 rounded-sm mr-1" style="background-color: var(--oo-pipe-direct);"></span>System {budget.system_tokens.toLocaleString()}</span>
					{#if budget.project_tokens > 0}
						<span><span class="inline-block w-2.5 h-2.5 rounded-sm mr-1" style="background-color: var(--oo-pipe-tools);"></span>Project {budget.project_tokens.toLocaleString()}</span>
					{/if}
					<span><span class="inline-block w-2.5 h-2.5 rounded-sm mr-1" style="background-color: var(--oo-success);"></span>History {budget.history_tokens.toLocaleString()}</span>
					<span><span class="inline-block w-2.5 h-2.5 rounded-sm mr-1" style="background-color: var(--oo-acc-400);"></span>User {budget.user_tokens.toLocaleString()}</span>
					<span><span class="inline-block w-2.5 h-2.5 rounded-sm mr-1" style="background-color: var(--oo-fg-tertiary);"></span>Reserve {budget.reserve_tokens.toLocaleString()}</span>
				</div>
			{/if}
		</div>

		<!-- Templates Section -->
		<div class="p-4 rounded-lg" style="background-color: var(--oo-bg-secondary); border: 1px solid var(--oo-bd-default);">
			<div class="flex items-center justify-between mb-3">
				<h3 class="text-sm font-medium" style="color: var(--oo-fg-primary);">Prompt Templates</h3>
				<button
					on:click={handleClearAllOverrides}
					class="px-2 py-1 rounded text-[11px] transition-colors"
					style="color: var(--oo-fg-tertiary); border: 1px solid var(--oo-bd-default);"
					title="Clear all runtime overrides"
				>
					Reset All Overrides
				</button>
			</div>

			<!-- Template list -->
			<div class="grid grid-cols-2 sm:grid-cols-3 gap-2 mb-4">
				{#each templates as tpl}
					<button
						on:click={() => viewTemplate(tpl.task_type)}
						class="px-3 py-2 rounded text-left text-xs transition-colors"
						style="background-color: {selectedTemplate?.task_type === tpl.task_type ? 'var(--oo-acc-500/0.15)' : 'var(--oo-bg-tertiary)'}; border: 1px solid {selectedTemplate?.task_type === tpl.task_type ? 'var(--oo-acc-500)' : 'var(--oo-bd-default)'}; color: var(--oo-fg-primary);"
					>
						<div class="font-medium truncate">{tpl.task_type}</div>
						<div class="flex items-center gap-1 mt-0.5" style="color: var(--oo-fg-tertiary);">
							<span style={sourceColor(tpl.source)}>{tpl.source}</span>
							{#if tpl.temperature_override !== null}
								<span>· t={tpl.temperature_override}</span>
							{/if}
						</div>
					</button>
				{/each}
			</div>

			<!-- Template detail/editor -->
			{#if selectedTemplate}
				<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-primary); border: 1px solid var(--oo-bd-default);">
					<div class="flex items-center justify-between mb-2">
						<div class="flex items-center gap-2">
							<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">
								{selectedTemplate.task_type}
							</span>
							<span class="text-[11px] px-1.5 py-0.5 rounded" style={sourceColor(selectedTemplate.source) + ' background-color: var(--oo-bg-tertiary);'}>
								{selectedTemplate.source}
							</span>
						</div>
						<div class="flex gap-1">
							{#if !editingTemplate}
								<button
									on:click={() => { editingTemplate = true; }}
									class="px-2 py-1 rounded text-[11px] transition-colors"
									style="background-color: var(--oo-acc-500); color: white;"
								>
									Edit
								</button>
							{:else}
								<button
									on:click={saveOverride}
									class="px-2 py-1 rounded text-[11px] transition-colors"
									style="background-color: var(--oo-success); color: white;"
								>
									Save Override
								</button>
								<button
									on:click={() => { editingTemplate = false; editPrompt = selectedTemplate?.system_prompt ?? ''; }}
									class="px-2 py-1 rounded text-[11px] transition-colors"
									style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-default);"
								>
									Cancel
								</button>
							{/if}
							{#if selectedTemplate.source === 'runtime'}
								<button
									on:click={removeOverride}
									class="px-2 py-1 rounded text-[11px] transition-colors"
									style="color: var(--oo-error); border: 1px solid var(--oo-error-bd);"
								>
									Remove Override
								</button>
							{/if}
						</div>
					</div>

					{#if selectedTemplate.temperature_override !== null}
						<div class="text-[11px] mb-2" style="color: var(--oo-fg-tertiary);">
							Temperature override: <strong style="color: var(--oo-fg-primary);">{selectedTemplate.temperature_override}</strong>
						</div>
					{/if}

					{#if editingTemplate}
						<textarea
							bind:value={editPrompt}
							rows="10"
							class="w-full px-3 py-2 rounded text-xs font-mono resize-y"
							style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
						></textarea>
						<div class="flex items-center gap-2 mt-2">
							<label class="text-[11px]" style="color: var(--oo-fg-tertiary);">Temperature override:</label>
							<input
								type="text"
								bind:value={editTemp}
								class="w-20 px-2 py-1 rounded text-xs"
								style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
								placeholder="e.g. 0.3"
							/>
						</div>
					{:else}
						<pre class="whitespace-pre-wrap text-xs p-2 rounded overflow-auto max-h-64"
							style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-secondary);"
						>{selectedTemplate.system_prompt}</pre>
					{/if}
				</div>
			{/if}
		</div>

		<!-- Cache Section -->
		{#if cacheStats}
			<div class="p-4 rounded-lg" style="background-color: var(--oo-bg-secondary); border: 1px solid var(--oo-bd-default);">
				<div class="flex items-center justify-between mb-2">
					<h3 class="text-sm font-medium" style="color: var(--oo-fg-primary);">Context Window Cache</h3>
					<button
						on:click={handleClearCache}
						class="px-2 py-1 rounded text-[11px] transition-colors"
						style="color: var(--oo-fg-tertiary); border: 1px solid var(--oo-bd-default);"
					>
						Clear Cache
					</button>
				</div>
				<div class="flex gap-4 text-xs" style="color: var(--oo-fg-tertiary);">
					<span>Entries: <strong style="color: var(--oo-fg-primary);">{cacheStats.entries}</strong> / {cacheStats.max_entries}</span>
					<span>TTL: {cacheStats.ttl_seconds}s</span>
				</div>
				{#if cacheStats.models.length > 0}
					<div class="mt-1 text-[11px]" style="color: var(--oo-fg-tertiary);">
						Cached: {cacheStats.models.join(', ')}
					</div>
				{/if}
			</div>
		{/if}
	{/if}
</div>
