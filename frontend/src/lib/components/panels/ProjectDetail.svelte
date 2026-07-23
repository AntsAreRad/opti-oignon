<!--
  ProjectDetail.svelte
  Full project detail on the ds primitives, promoted into the /projects/[id]
  route. Editable name/description/system_instructions, a settings group
  (default model, context budget, auto-index), a stats bar, and a tabbed
  interface (Files | Outputs | Conversations | Context). All behaviour and the
  project APIs are preserved; only the presentation moves to primitives.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { Button, Card, Input, Switch, Tabs, Icon } from '$lib/ds';
	import type { TabItem } from '$lib/ds';
	import {
		activeProjectDetail,
		detailLoading,
		updateProject,
		reindexAll,
		getContextPreview,
		formatFileSize,
	} from '$lib/stores/projects';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type { ProjectContextPreview } from '$lib/types';
	import FileManager from './FileManager.svelte';

	const dispatch = createEventDispatcher<{ back: void }>();

	let activeTab = 'files';

	// Editing state
	let editing = false;
	let editName = '';
	let editDescription = '';
	let editInstructions = '';
	let saving = false;

	// Settings editing
	let editingSettings = false;
	let settingsModel = '';
	let settingsBudget = 4096;
	let settingsAutoIndex = true;
	let savingSettings = false;

	// Reindex state
	let reindexing = false;

	// Context preview state
	let contextQuery = '';
	let contextPreview: ProjectContextPreview | null = null;
	let contextLoading = false;

	// Sync edit fields when detail changes
	$: if ($activeProjectDetail && !editing) {
		editName = $activeProjectDetail.name;
		editDescription = $activeProjectDetail.description;
		editInstructions = $activeProjectDetail.system_instructions;
	}

	$: if ($activeProjectDetail && !editingSettings) {
		const s = $activeProjectDetail.settings || {};
		settingsModel = (s.default_model as string) ?? '';
		settingsBudget = (s.context_budget_tokens as number) ?? 4096;
		settingsAutoIndex = (s.auto_index as boolean) ?? true;
	}

	$: detail = $activeProjectDetail;
	$: stats = detail?.stats ?? null;

	let tabItems: TabItem[] = [];
	$: tabItems = [
		{ id: 'files', label: `Files (${detail?.files.length ?? 0})` },
		{ id: 'outputs', label: `Outputs (${detail?.outputs.length ?? 0})` },
		{ id: 'conversations', label: `Conversations (${detail?.conversations.length ?? 0})` },
		{ id: 'context', label: 'Context' },
	];

	function startEdit() {
		if (!detail) return;
		editName = detail.name;
		editDescription = detail.description;
		editInstructions = detail.system_instructions;
		editing = true;
	}

	async function saveEdit() {
		if (!detail || !editName.trim()) return;
		saving = true;
		try {
			await updateProject(detail.id, {
				name: editName.trim(),
				description: editDescription.trim(),
				system_instructions: editInstructions.trim(),
			});
			toastSuccess('Project updated');
			editing = false;
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to update');
		} finally {
			saving = false;
		}
	}

	async function saveSettings() {
		if (!detail) return;
		savingSettings = true;
		try {
			await updateProject(detail.id, {
				settings: {
					...detail.settings,
					default_model: settingsModel.trim() || undefined,
					context_budget_tokens: settingsBudget,
					auto_index: settingsAutoIndex,
				},
			});
			toastSuccess('Settings saved');
			editingSettings = false;
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to save settings');
		} finally {
			savingSettings = false;
		}
	}

	async function handleReindex() {
		if (!detail) return;
		reindexing = true;
		try {
			const result = await reindexAll(detail.id);
			toastSuccess(`Reindexed: ${result.indexed} files (${result.failed} failed)`);
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Reindex failed');
		} finally {
			reindexing = false;
		}
	}

	async function handleContextPreview() {
		if (!detail || !contextQuery.trim()) return;
		contextLoading = true;
		contextPreview = null;
		try {
			contextPreview = await getContextPreview(detail.id, contextQuery.trim());
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Context preview failed');
		} finally {
			contextLoading = false;
		}
	}

	function formatDate(dateStr: string): string {
		if (!dateStr) return 'N/A';
		try {
			return new Date(dateStr).toLocaleString();
		} catch {
			return dateStr;
		}
	}
</script>

<div class="h-full overflow-y-auto">
	<div class="max-w-3xl mx-auto px-4 py-6">
		{#if $detailLoading}
			<div class="flex items-center gap-2 justify-center py-12 text-sm" style="color: var(--oo-fg-muted);">
				<span class="oo-spin" aria-hidden="true"></span>
				Loading project...
			</div>
		{:else if !detail}
			<div class="text-center py-12">
				<p class="text-sm" style="color: var(--oo-fg-muted);">Project not found</p>
				<Button variant="link" size="sm" on:click={() => dispatch('back')}>Back to projects</Button>
			</div>
		{:else}
			<!-- Header -->
			<div class="flex items-center gap-3 mb-4">
				<Button
					variant="ghost"
					size="sm"
					iconOnly="arrow-left"
					ariaLabel="Back to projects"
					on:click={() => dispatch('back')}
				/>

				{#if editing}
					<div class="flex-1 min-w-0 flex flex-col gap-2">
						<Input label="Project name" hideLabel bind:value={editName} placeholder="Project name" />
						<Input label="Description" hideLabel bind:value={editDescription} placeholder="Description" />
						<Input
							type="textarea"
							label="System instructions"
							hideLabel
							rows={3}
							bind:value={editInstructions}
							placeholder="System instructions..."
						/>
						<div class="flex gap-2">
							<Button variant="primary" size="sm" loading={saving} disabled={!editName.trim()} on:click={saveEdit}>
								Save
							</Button>
							<Button variant="ghost" size="sm" on:click={() => (editing = false)}>Cancel</Button>
						</div>
					</div>
				{:else}
					<div class="flex-1 min-w-0">
						<h1 class="text-lg font-semibold truncate" style="color: var(--oo-fg-primary);">
							{detail.name}
						</h1>
						{#if detail.description}
							<p class="text-xs mt-0.5" style="color: var(--oo-fg-muted);">{detail.description}</p>
						{/if}
					</div>
					<Button variant="ghost" size="sm" iconOnly="pencil" ariaLabel="Edit project" on:click={startEdit} />
				{/if}
			</div>

			<!-- System instructions preview (read mode) -->
			{#if !editing && detail.system_instructions}
				<div
					class="mb-4 px-3 py-2 rounded text-xs font-mono max-h-24 overflow-y-auto"
					style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle); color: var(--oo-fg-muted);"
				>
					{detail.system_instructions}
				</div>
			{/if}

			<!-- Stats bar -->
			<Card variant="flat" padding="sm" class="mb-4">
				<div class="flex flex-wrap items-center gap-4 text-xs" style="color: var(--oo-fg-muted);">
					<span class="flex items-center gap-1" title="Files">
						<Icon name="file" size="sm" />
						{stats?.file_count ?? detail.files.length} files
					</span>
					<span class="flex items-center gap-1" title="Indexed files">
						<Icon name="search" size="sm" />
						{stats?.indexed_file_count ?? 0} indexed
					</span>
					<span class="flex items-center gap-1" title="Total chunks">
						<Icon name="layers" size="sm" />
						{stats?.total_chunk_count ?? 0} chunks
					</span>
					<span class="flex items-center gap-1" title="Conversations">
						<Icon name="message-square" size="sm" />
						{stats?.conversation_count ?? detail.conversations.length}
					</span>
					<span class="flex items-center gap-1" title="Total size">
						<Icon name="database" size="sm" />
						{formatFileSize(stats?.total_file_size_bytes ?? 0)}
					</span>
				</div>
			</Card>

			<!-- Settings group -->
			<details class="mb-4 oo-settings">
				<summary class="oo-settings-summary">Settings</summary>
				<Card variant="flat" padding="md" class="mt-2">
					<div class="flex flex-col gap-3">
						<Input
							label="Default model"
							size="sm"
							bind:value={settingsModel}
							on:focus={() => (editingSettings = true)}
							placeholder="e.g. qwen3-coder:30b"
						/>
						<div>
							<label class="block text-xs mb-1" style="color: var(--oo-fg-muted);" for="set-budget">
								Context budget (tokens): {settingsBudget}
							</label>
							<input
								id="set-budget"
								type="range"
								min="512"
								max="16384"
								step="512"
								bind:value={settingsBudget}
								on:input={() => (editingSettings = true)}
								class="w-full oo-range"
							/>
							<div class="flex justify-between text-[10px]" style="color: var(--oo-fg-faint);">
								<span>512</span><span>16384</span>
							</div>
						</div>
						<Switch
							label="Auto-index new files on upload"
							size="sm"
							bind:checked={settingsAutoIndex}
							on:change={() => (editingSettings = true)}
						/>
						{#if editingSettings}
							<div class="flex gap-2 pt-1">
								<Button variant="primary" size="sm" loading={savingSettings} on:click={saveSettings}>
									Save settings
								</Button>
								<Button variant="ghost" size="sm" on:click={() => (editingSettings = false)}>Cancel</Button>
							</div>
						{/if}
					</div>
				</Card>
			</details>

			<!-- Tabs -->
			<Tabs bind:value={activeTab} tabs={tabItems} variant="underline" size="sm">
				{#if activeTab === 'files'}
					<FileManager projectId={detail.id} files={detail.files} />
				{:else if activeTab === 'outputs'}
					{#if detail.outputs.length === 0}
						<p class="text-xs py-6 text-center" style="color: var(--oo-fg-faint);">
							No outputs yet. Outputs are saved from conversations linked to this project.
						</p>
					{:else}
						<div class="flex flex-col gap-1.5">
							{#each detail.outputs as output (output.id)}
								<div
									class="flex items-center gap-3 px-3 py-2 rounded text-xs"
									style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);"
								>
									<span style="color: var(--oo-acc-400);"><Icon name="file-text" size="sm" /></span>
									<div class="min-w-0 flex-1">
										<span class="font-medium" style="color: var(--oo-fg-primary);">{output.filename}</span>
										{#if output.description}
											<span class="ml-2" style="color: var(--oo-fg-faint);">{output.description}</span>
										{/if}
									</div>
									<span class="shrink-0 px-1.5 py-0.5 rounded" style="background-color: var(--oo-bg-base); color: var(--oo-fg-faint);">
										{output.output_type}
									</span>
									<span style="color: var(--oo-fg-faint);">{formatDate(output.created_at)}</span>
								</div>
							{/each}
						</div>
					{/if}
				{:else if activeTab === 'conversations'}
					{#if detail.conversations.length === 0}
						<p class="text-xs py-6 text-center" style="color: var(--oo-fg-faint);">
							No conversations linked. Link a conversation from the chat header.
						</p>
					{:else}
						<div class="flex flex-col gap-1.5">
							{#each detail.conversations as conv}
								<div
									class="flex items-center gap-3 px-3 py-2 rounded text-xs"
									style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);"
								>
									<span style="color: var(--oo-fg-tertiary);"><Icon name="message-square" size="sm" /></span>
									<span class="font-mono min-w-0 flex-1 truncate" style="color: var(--oo-fg-primary);">
										{conv.conversation_id}
									</span>
									<span style="color: var(--oo-fg-faint);">{formatDate(conv.linked_at)}</span>
								</div>
							{/each}
						</div>
					{/if}
				{:else if activeTab === 'context'}
					<div class="flex flex-col gap-4">
						<div class="flex items-center gap-3">
							<Button variant="secondary" size="sm" iconLeft="refresh-cw" loading={reindexing} on:click={handleReindex}>
								Reindex all files
							</Button>
							<span class="text-[11px]" style="color: var(--oo-fg-faint);">
								Rebuild ChromaDB index for all project files
							</span>
						</div>

						<Card variant="flat" padding="md">
							<h3 class="text-xs font-medium mb-2" style="color: var(--oo-fg-primary);">Context injection preview</h3>
							<p class="text-[11px] mb-2" style="color: var(--oo-fg-faint);">
								Test what context would be injected for a given query.
							</p>
							<div class="flex gap-2 items-end">
								<div class="flex-1">
									<Input
										label="Test query"
										hideLabel
										size="sm"
										bind:value={contextQuery}
										placeholder="Enter a test query..."
										on:keydown={(e) => {
											if (e.key === 'Enter') handleContextPreview();
										}}
									/>
								</div>
								<Button
									variant="primary"
									size="sm"
									loading={contextLoading}
									disabled={!contextQuery.trim()}
									on:click={handleContextPreview}
								>
									Preview
								</Button>
							</div>

							{#if contextPreview}
								<div class="mt-3 flex flex-col gap-2">
									<div class="flex items-center gap-2 text-xs">
										<span class="font-medium" style="color: var(--oo-fg-primary);">Trigger:</span>
										{#if contextPreview.trigger.relevant}
											<span class="px-1.5 py-0.5 rounded" style="background-color: var(--oo-success-bg); color: var(--oo-success);">
												Relevant
											</span>
										{:else}
											<span class="px-1.5 py-0.5 rounded" style="background-color: var(--oo-error-bg); color: var(--oo-error);">
												Not relevant
											</span>
										{/if}
										<span style="color: var(--oo-fg-faint);">
											Level {contextPreview.trigger.level} | {(contextPreview.trigger.confidence * 100).toFixed(0)}% | {contextPreview.trigger.duration_ms.toFixed(0)}ms
										</span>
									</div>
									{#if contextPreview.trigger.reason}
										<p class="text-[11px] pl-2" style="color: var(--oo-fg-muted);">{contextPreview.trigger.reason}</p>
									{/if}

									{#if contextPreview.context}
										<div class="text-xs flex flex-col gap-1 pt-1" style="border-top: 1px solid var(--oo-bd-subtle);">
											<div class="flex items-center gap-3" style="color: var(--oo-fg-muted);">
												<span>{contextPreview.context.chunks_retrieved} chunks</span>
												<span>{contextPreview.context.total_tokens} tokens</span>
												<span>{contextPreview.context.source_files.length} source files</span>
											</div>
											{#if contextPreview.context.source_files.length > 0}
												<div class="flex flex-wrap gap-1">
													{#each contextPreview.context.source_files as sf}
														<span class="px-1.5 py-0.5 rounded text-[10px]" style="background-color: var(--oo-bg-base); color: var(--oo-fg-faint);">
															{sf}
														</span>
													{/each}
												</div>
											{/if}
											{#if contextPreview.context.content_preview}
												<pre class="mt-1 p-2 rounded text-[10px] max-h-32 overflow-y-auto whitespace-pre-wrap" style="background-color: var(--oo-bg-base); color: var(--oo-fg-muted);">{contextPreview.context.content_preview}</pre>
											{/if}
										</div>
									{/if}
								</div>
							{/if}
						</Card>
					</div>
				{/if}
			</Tabs>
		{/if}
	</div>
</div>

<style>
	.oo-spin {
		width: 1.25rem;
		height: 1.25rem;
		border: 2px solid var(--oo-bd-default);
		border-top-color: var(--oo-acc-500);
		border-radius: var(--oo-radius-full);
		display: inline-block;
		animation: oo-spin 0.7s linear infinite;
	}
	@keyframes oo-spin {
		to {
			transform: rotate(360deg);
		}
	}
	.oo-settings-summary {
		cursor: pointer;
		font-size: var(--oo-text-xs);
		font-weight: 500;
		color: var(--oo-fg-tertiary);
		padding: var(--oo-space-1) var(--oo-space-1);
	}
	.oo-range {
		accent-color: var(--oo-acc-500);
	}
</style>
