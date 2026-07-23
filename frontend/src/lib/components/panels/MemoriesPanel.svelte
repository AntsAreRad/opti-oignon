<!--
  MemoriesPanel.svelte (Theme 3 / Odysseus Core)
  The memories panel for the two-tier MemoryStore, built on the lib/ds
  primitives (Card, Tabs, Modal, Select, Icon). Lists memories grouped by
  category under an Active / Archived tab, with soft-delete, restore, and edit
  actions wired to the /api/memories surface. Distinct from the legacy
  MemoryPanel.svelte, which stays in place. Design-system tokens only (--oo-*);
  lucide-svelte icons through Icon.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { Button, Card, Tabs, Modal, Select, Icon, EmptyState, InlineError } from '$lib/ds';
	import type { TabItem } from '$lib/ds';
	import {
		listMemories,
		editMemory,
		softDeleteMemory,
		restoreMemory,
		MEMORY_CATEGORIES,
		type MemoryRecord
	} from '$lib/api/memories';
	import { toastSuccess, toastError } from '$lib/stores/notifications';

	let memories: MemoryRecord[] = [];
	let loading = false;
	let error: string | null = null;
	let statusTab = 'active';
	let busyId: string | null = null;

	// Edit modal state
	let editing: MemoryRecord | null = null;
	let editText = '';
	let editCategory = 'fact';
	let saving = false;

	const CATEGORY_ORDER = MEMORY_CATEGORIES;
	const CATEGORY_LABELS: Record<string, string> = {
		identity: 'Identity',
		preference: 'Preference',
		fact: 'Fact',
		contact: 'Contact',
		project: 'Project',
		goal: 'Goal'
	};

	const tabs: TabItem[] = [
		{ id: 'active', label: 'Active', icon: 'brain' },
		{ id: 'archived', label: 'Archived', icon: 'archive' }
	];

	const categoryOptions = CATEGORY_ORDER.map((c) => ({
		value: c,
		label: CATEGORY_LABELS[c] ?? c
	}));

	function label(category: string): string {
		return CATEGORY_LABELS[category] ?? category;
	}

	async function loadMemories() {
		loading = true;
		error = null;
		try {
			memories = await listMemories({ active_only: false });
		} catch {
			error = 'Could not load memories.';
			memories = [];
		} finally {
			loading = false;
		}
	}

	$: activeMemories = memories.filter((m) => m.active);
	$: archivedMemories = memories.filter((m) => !m.active);
	$: current = statusTab === 'active' ? activeMemories : archivedMemories;

	type Group = { category: string; label: string; items: MemoryRecord[] };

	$: grouped = CATEGORY_ORDER.map((category) => ({
		category,
		label: label(category),
		items: current.filter((m) => m.category === category)
	})).filter((g: Group) => g.items.length > 0);

	function replace(updated: MemoryRecord) {
		memories = memories.map((m) => (m.id === updated.id ? updated : m));
	}

	function openEdit(memory: MemoryRecord) {
		editing = memory;
		editText = memory.text;
		editCategory = memory.category;
	}

	function closeEdit() {
		editing = null;
	}

	function onCategoryChange(event: CustomEvent<string | string[]>) {
		const value = event.detail;
		editCategory = typeof value === 'string' ? value : (value[0] ?? 'fact');
	}

	async function saveEdit() {
		if (!editing || saving) return;
		const text = editText.trim();
		if (!text) {
			toastError('Memory text cannot be empty');
			return;
		}
		saving = true;
		try {
			const updated = await editMemory(editing.id, { text, category: editCategory });
			replace(updated);
			toastSuccess('Memory updated');
			closeEdit();
		} catch {
			toastError('Failed to update memory');
		} finally {
			saving = false;
		}
	}

	async function doSoftDelete(memory: MemoryRecord) {
		if (busyId) return;
		busyId = memory.id;
		try {
			await softDeleteMemory(memory.id);
			replace({ ...memory, active: false });
			toastSuccess('Memory archived');
		} catch {
			toastError('Failed to archive memory');
		} finally {
			busyId = null;
		}
	}

	async function doRestore(memory: MemoryRecord) {
		if (busyId) return;
		busyId = memory.id;
		try {
			await restoreMemory(memory.id);
			replace({ ...memory, active: true });
			toastSuccess('Memory restored');
		} catch {
			toastError('Failed to restore memory');
		} finally {
			busyId = null;
		}
	}

	onMount(loadMemories);
</script>

<section class="memories-panel">
	<header class="memories-header">
		<div class="memories-title">
			<Icon name="brain" size="md" />
			<h2>Memories</h2>
		</div>
		<Button
			variant="ghost"
			size="sm"
			iconOnly="refresh-cw"
			ariaLabel="Reload memories"
			loading={loading}
			on:click={loadMemories}
		/>
	</header>

	<Tabs bind:value={statusTab} {tabs} />

	{#if error}
		<InlineError message={error} onRetry={loadMemories} retrying={loading} />
	{:else if loading && memories.length === 0}
		<p class="memories-status">Loading memories...</p>
	{:else if current.length === 0}
		<EmptyState
			icon="inbox"
			title={statusTab === 'active' ? 'No active memories' : 'No archived memories'}
			description={statusTab === 'active'
				? 'Memories extracted from your conversations appear here.'
				: 'Archived memories can be restored from here.'}
		/>
	{:else}
		{#each grouped as group (group.category)}
			<div class="memories-group">
				<div class="memories-group-head">
					<Icon name="tag" size="sm" />
					<span class="memories-group-label">{group.label}</span>
					<span class="memories-group-count">{group.items.length}</span>
				</div>

				{#each group.items as memory (memory.id)}
					<Card variant="flat" padding="sm">
						<div class="memory-row">
							<p class="memory-text">{memory.text}</p>
							<div class="memory-actions">
								{#if memory.active}
									<Button
										variant="ghost"
										size="sm"
										iconLeft="pencil"
										on:click={() => openEdit(memory)}
									>
										Edit
									</Button>
									<Button
										variant="danger"
										size="sm"
										iconLeft="trash-2"
										loading={busyId === memory.id}
										on:click={() => doSoftDelete(memory)}
									>
										Archive
									</Button>
								{:else}
									<Button
										variant="secondary"
										size="sm"
										iconLeft="rotate-ccw"
										loading={busyId === memory.id}
										on:click={() => doRestore(memory)}
									>
										Restore
									</Button>
								{/if}
							</div>
						</div>
					</Card>
				{/each}
			</div>
		{/each}
	{/if}
</section>

<Modal open={editing !== null} title="Edit memory" size="md" onClose={closeEdit}>
	<div class="memory-edit">
		<label class="memory-edit-label" for="memory-edit-text">Memory text</label>
		<textarea
			id="memory-edit-text"
			class="memory-edit-text"
			rows="3"
			bind:value={editText}
		></textarea>
		<Select
			label="Category"
			value={editCategory}
			options={categoryOptions}
			on:change={onCategoryChange}
		/>
	</div>
	<svelte:fragment slot="footer">
		<Button variant="ghost" on:click={closeEdit}>Cancel</Button>
		<Button variant="primary" loading={saving} on:click={saveEdit}>Save</Button>
	</svelte:fragment>
</Modal>

<style>
	.memories-panel {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
		padding: var(--oo-space-3);
	}

	.memories-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.memories-title {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-primary);
	}

	.memories-title h2 {
		margin: 0;
		font-size: var(--oo-text-lg);
		font-weight: 600;
	}

	.memories-status {
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
		margin: 0;
	}

	.memories-group {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}

	.memories-group-head {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-tertiary);
	}

	.memories-group-label {
		font-size: var(--oo-text-sm);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.memories-group-count {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-faint);
		background: var(--oo-bg-elevated);
		border-radius: var(--oo-radius-full);
		padding: 0 var(--oo-space-2);
	}

	.memory-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-3);
	}

	.memory-text {
		margin: 0;
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
		line-height: 1.4;
	}

	.memory-actions {
		display: flex;
		align-items: center;
		gap: var(--oo-space-1);
		flex-shrink: 0;
	}

	.memory-edit {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
	}

	.memory-edit-label {
		font-size: var(--oo-text-sm);
		font-weight: 500;
		color: var(--oo-fg-secondary);
	}

	.memory-edit-text {
		width: 100%;
		resize: vertical;
		font: inherit;
		color: var(--oo-fg-primary);
		background: var(--oo-bg-base);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
		padding: var(--oo-space-2);
	}

	.memory-edit-text:focus {
		outline: none;
		border-color: var(--oo-acc-500);
	}
</style>
