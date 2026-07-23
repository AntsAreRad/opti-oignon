<!--
  ProjectList.svelte
  Project index on the ds primitives. Sort (last updated / name / created),
  list/cards view toggle, and a per-project secondary action menu (Open,
  Settings, Star, Duplicate, Archive, Delete). Starred/archived are stored in
  the existing project `settings` blob and toggled through updateProject, so no
  project API changes are introduced. Navigation uses real /projects/[id] links.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { Button, Card, Input, Modal, Select, Icon, EmptyState, InlineError } from '$lib/ds';
	import {
		projects,
		projectsLoading,
		projectError,
		loadProjects,
		createProject,
		updateProject,
		deleteProject,
	} from '$lib/stores/projects';
	import type { ProjectInfo } from '$lib/types';
	import { toastSuccess, toastError } from '$lib/stores/notifications';

	/** When true (driven by ?new=1), open the create modal on load. */
	export let openCreate = false;

	type SortKey = 'updated' | 'name' | 'created';
	type View = 'cards' | 'list';

	let view: View = 'cards';
	let sortBy: SortKey = 'updated';
	let showArchived = false;

	// Create modal state
	let showCreate = false;
	let newName = '';
	let newDescription = '';
	let newInstructions = '';
	let creating = false;

	// Per-row action menu + delete confirmation
	let openMenuId: string | null = null;
	let deleteConfirmId: string | null = null;
	let busyId: string | null = null;

	const sortOptions = [
		{ value: 'updated', label: 'Last updated' },
		{ value: 'name', label: 'Name' },
		{ value: 'created', label: 'Created' },
	];

	$: if (openCreate && !showCreate) {
		showCreate = true;
	}

	function isStarred(p: ProjectInfo): boolean {
		return Boolean((p.settings || {}).starred);
	}
	function isArchived(p: ProjectInfo): boolean {
		return Boolean((p.settings || {}).archived);
	}

	$: archivedCount = $projects.filter(isArchived).length;

	$: visible = $projects
		.filter((p) => (showArchived ? true : !isArchived(p)))
		.slice()
		.sort((a, b) => {
			if (sortBy === 'name') return a.name.localeCompare(b.name);
			const key = sortBy === 'created' ? 'created_at' : 'updated_at';
			return (b[key] || '').localeCompare(a[key] || '');
		});

	onMount(() => {
		loadProjects();
		if (typeof window !== 'undefined') {
			window.addEventListener('click', closeMenu);
		}
	});

	onDestroy(() => {
		if (typeof window !== 'undefined') {
			window.removeEventListener('click', closeMenu);
		}
	});

	function closeMenu() {
		openMenuId = null;
	}

	function onSortChange(e: CustomEvent<string | string[]>) {
		if (typeof e.detail === 'string') sortBy = e.detail as SortKey;
	}

	function toggleMenu(id: string) {
		openMenuId = openMenuId === id ? null : id;
		deleteConfirmId = null;
	}

	function closeCreate() {
		showCreate = false;
		newName = '';
		newDescription = '';
		newInstructions = '';
	}

	async function handleCreate() {
		if (!newName.trim()) return;
		creating = true;
		try {
			const project = await createProject({
				name: newName.trim(),
				description: newDescription.trim(),
				system_instructions: newInstructions.trim(),
			});
			toastSuccess(`Project "${project.name}" created`);
			closeCreate();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to create project');
		} finally {
			creating = false;
		}
	}

	async function toggleStar(p: ProjectInfo) {
		busyId = p.id;
		try {
			await updateProject(p.id, { settings: { ...p.settings, starred: !isStarred(p) } });
			toastSuccess(isStarred(p) ? 'Unstarred' : 'Starred');
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to update project');
		} finally {
			busyId = null;
			openMenuId = null;
		}
	}

	async function toggleArchive(p: ProjectInfo) {
		busyId = p.id;
		try {
			await updateProject(p.id, { settings: { ...p.settings, archived: !isArchived(p) } });
			toastSuccess(isArchived(p) ? 'Unarchived' : 'Archived');
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to update project');
		} finally {
			busyId = null;
			openMenuId = null;
		}
	}

	async function duplicate(p: ProjectInfo) {
		busyId = p.id;
		try {
			const copy = await createProject({
				name: `${p.name} (copy)`,
				description: p.description,
				system_instructions: p.system_instructions,
				settings: { ...p.settings, starred: false, archived: false },
			});
			toastSuccess(`Duplicated to "${copy.name}"`);
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to duplicate project');
		} finally {
			busyId = null;
			openMenuId = null;
		}
	}

	async function handleDelete(p: ProjectInfo) {
		busyId = p.id;
		try {
			await deleteProject(p.id);
			toastSuccess(`Project "${p.name}" deleted`);
			deleteConfirmId = null;
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to delete project');
		} finally {
			busyId = null;
			openMenuId = null;
		}
	}

	function formatDate(dateStr: string): string {
		if (!dateStr) return 'N/A';
		try {
			const d = new Date(dateStr);
			const now = new Date();
			const diffMins = Math.floor((now.getTime() - d.getTime()) / 60000);
			if (diffMins < 1) return 'Just now';
			if (diffMins < 60) return `${diffMins}m ago`;
			const diffHrs = Math.floor(diffMins / 60);
			if (diffHrs < 24) return `${diffHrs}h ago`;
			const diffDays = Math.floor(diffHrs / 24);
			if (diffDays < 30) return `${diffDays}d ago`;
			return d.toLocaleDateString();
		} catch {
			return dateStr;
		}
	}
</script>

<div class="h-full overflow-y-auto">
	<div class="max-w-4xl mx-auto px-4 py-6">
		<!-- Header -->
		<div class="flex items-start justify-between gap-3 mb-5">
			<div>
				<h1 class="text-lg font-semibold" style="color: var(--oo-fg-primary);">Projects</h1>
				<p class="text-xs mt-0.5" style="color: var(--oo-fg-muted);">
					Manage project files, context injection, and conversation linking.
				</p>
			</div>
			<Button variant="primary" iconLeft="plus" on:click={() => (showCreate = true)}>
				New project
			</Button>
		</div>

		<!-- Toolbar: count + sort + view toggle -->
		{#if $projects.length > 0}
			<div class="flex items-center justify-between gap-3 mb-4 flex-wrap">
				<span class="text-xs" style="color: var(--oo-fg-muted);">
					{visible.length}
					{visible.length === 1 ? 'project' : 'projects'}
				</span>
				<div class="flex items-center gap-3">
					{#if archivedCount > 0}
						<button
							class="text-xs underline-offset-2 hover:underline"
							style="color: var(--oo-fg-tertiary);"
							on:click={() => (showArchived = !showArchived)}
						>
							{showArchived ? 'Hide archived' : `Show archived (${archivedCount})`}
						</button>
					{/if}
					<div class="w-40">
						<Select
							label="Sort by"
							hideLabel
							size="sm"
							value={sortBy}
							options={sortOptions}
							on:change={onSortChange}
						/>
					</div>
					<div class="flex items-center gap-0.5 rounded-md p-0.5" style="border: 1px solid var(--oo-bd-default);">
						<Button
							variant={view === 'list' ? 'secondary' : 'ghost'}
							size="sm"
							iconOnly="list"
							ariaLabel="List view"
							on:click={() => (view = 'list')}
						/>
						<Button
							variant={view === 'cards' ? 'secondary' : 'ghost'}
							size="sm"
							iconOnly="layout-grid"
							ariaLabel="Card view"
							on:click={() => (view = 'cards')}
						/>
					</div>
				</div>
			</div>
		{/if}

		<!-- States -->
		{#if $projectsLoading}
			<div class="flex items-center gap-2 justify-center py-12 text-sm" style="color: var(--oo-fg-muted);">
				<span class="oo-spin" aria-hidden="true"></span>
				Loading projects...
			</div>
		{:else if $projectError}
			<div class="mb-4">
				<InlineError message={$projectError} onRetry={() => loadProjects()} />
			</div>
		{:else if $projects.length === 0}
			<EmptyState
				icon="folder"
				title="No projects yet"
				description="Create a project to organize files and inject context into conversations."
			>
				<Button variant="primary" iconLeft="plus" on:click={() => (showCreate = true)}>
					New project
				</Button>
			</EmptyState>
		{:else}
			<div class={view === 'cards' ? 'grid grid-cols-1 sm:grid-cols-2 gap-3' : 'flex flex-col gap-2'}>
				{#each visible as project (project.id)}
					<Card variant="flat" padding="md" class="oo-project-card">
						<div class="flex items-start justify-between gap-2">
							<div class="min-w-0 flex-1">
								<div class="flex items-center gap-1.5 min-w-0">
									{#if isStarred(project)}
										<span style="color: var(--oo-warning);" title="Starred"><Icon name="star" size="sm" /></span>
									{/if}
									<a
										href={`/projects/${project.id}`}
										class="text-sm font-medium truncate hover:underline"
										style="color: var(--oo-fg-primary);"
									>
										{project.name}
									</a>
									{#if isArchived(project)}
										<span
											class="text-[10px] px-1.5 py-0.5 rounded shrink-0"
											style="background-color: var(--oo-bg-base); color: var(--oo-fg-faint);"
										>archived</span>
									{/if}
								</div>
								{#if project.description}
									<p class="text-xs mt-1 line-clamp-2" style="color: var(--oo-fg-muted);">
										{project.description}
									</p>
								{/if}
								<div class="flex items-center gap-3 mt-2 text-[11px]" style="color: var(--oo-fg-faint);">
									<span class="flex items-center gap-1">
										<Icon name="clock" size="sm" />
										{formatDate(project.updated_at)}
									</span>
									{#if project.system_instructions}
										<span class="flex items-center gap-1" title="Has system instructions">
											<Icon name="file-text" size="sm" />
											Instructions
										</span>
									{/if}
								</div>
							</div>

							<!-- Secondary action menu -->
							<div class="relative shrink-0">
								<Button
									variant="ghost"
									size="sm"
									iconOnly="more-horizontal"
									ariaLabel="Project actions"
									on:click={(e) => {
										e.stopPropagation();
										toggleMenu(project.id);
									}}
								/>
								{#if openMenuId === project.id}
									<div
										class="oo-menu-list"
										role="menu"
										on:click={(e) => e.stopPropagation()}
										on:keydown={() => {}}
									>
										<a class="oo-menu-item" role="menuitem" href={`/projects/${project.id}`}>
											<Icon name="folder-open" size="sm" /> Open
										</a>
										<a class="oo-menu-item" role="menuitem" href={`/projects/${project.id}`}>
											<Icon name="settings" size="sm" /> Settings
										</a>
										<button class="oo-menu-item" role="menuitem" on:click={() => toggleStar(project)}>
											<Icon name="star" size="sm" /> {isStarred(project) ? 'Unstar' : 'Star'}
										</button>
										<button class="oo-menu-item" role="menuitem" on:click={() => duplicate(project)}>
											<Icon name="copy" size="sm" /> Duplicate
										</button>
										<button class="oo-menu-item" role="menuitem" on:click={() => toggleArchive(project)}>
											<Icon name="archive" size="sm" /> {isArchived(project) ? 'Unarchive' : 'Archive'}
										</button>
										<div class="oo-menu-sep"></div>
										{#if deleteConfirmId === project.id}
											<button
												class="oo-menu-item oo-menu-item-danger"
												role="menuitem"
												disabled={busyId === project.id}
												on:click={() => handleDelete(project)}
											>
												<Icon name="trash-2" size="sm" /> Confirm delete
											</button>
										{:else}
											<button
												class="oo-menu-item oo-menu-item-danger"
												role="menuitem"
												on:click={() => (deleteConfirmId = project.id)}
											>
												<Icon name="trash-2" size="sm" /> Delete
											</button>
										{/if}
									</div>
								{/if}
							</div>
						</div>

						<div class="flex items-center gap-2 mt-3">
							<Button variant="secondary" size="sm" href={`/projects/${project.id}`}>Open</Button>
						</div>
					</Card>
				{/each}
			</div>
		{/if}
	</div>
</div>

<!-- Create project modal -->
<Modal open={showCreate} title="New project" size="md" onClose={closeCreate}>
	<div class="flex flex-col gap-4">
		<Input
			label="Name"
			required
			bind:value={newName}
			placeholder="e.g. BCI Bioacoustics Analysis"
		/>
		<Input
			label="Description"
			bind:value={newDescription}
			placeholder="Short description of the project"
		/>
		<Input
			type="textarea"
			label="System instructions"
			rows={3}
			bind:value={newInstructions}
			placeholder="Instructions injected into every conversation linked to this project..."
		/>
	</div>
	<svelte:fragment slot="footer">
		<Button variant="ghost" on:click={closeCreate}>Cancel</Button>
		<Button variant="primary" loading={creating} disabled={!newName.trim()} on:click={handleCreate}>
			Create
		</Button>
	</svelte:fragment>
</Modal>

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
	:global(.oo-error-card) {
		background-color: var(--oo-error-bg);
		border-color: var(--oo-error-bd);
		color: var(--oo-error);
	}
	:global(.oo-project-card) {
		transition: border-color var(--oo-motion-fast) var(--oo-ease-default);
	}
	:global(.oo-project-card:hover) {
		border-color: var(--oo-bd-strong);
	}
	.oo-menu-list {
		position: absolute;
		right: 0;
		top: calc(100% + 4px);
		z-index: 20;
		min-width: 168px;
		display: flex;
		flex-direction: column;
		padding: var(--oo-space-1);
		background-color: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
		box-shadow: var(--oo-shadow-md);
	}
	.oo-menu-item {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		width: 100%;
		padding: var(--oo-space-2) var(--oo-space-3);
		border-radius: var(--oo-radius-sm);
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
		text-align: left;
		transition: background-color var(--oo-motion-fast) var(--oo-ease-default);
	}
	.oo-menu-item:hover {
		background-color: var(--oo-bg-elevated);
		color: var(--oo-fg-primary);
	}
	.oo-menu-item:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}
	.oo-menu-item-danger {
		color: var(--oo-error);
	}
	.oo-menu-item-danger:hover {
		background-color: var(--oo-error-bg);
		color: var(--oo-error);
	}
	.oo-menu-sep {
		height: 1px;
		margin: var(--oo-space-1) 0;
		background-color: var(--oo-bd-subtle);
	}
</style>
