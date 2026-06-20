<!--
  ModelAssignment.svelte (S60)
  Visual editor for model-to-role routing configuration.
  Displays roles (task types) with primary/fast/quality dropdowns
  populated from installed Ollama models.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		roles,
		installedModels,
		benchmarkError,
		loadRoles,
		saveRole,
	} from '$lib/stores/benchmark';
	import type { ModelRoleInfo } from '$lib/types';

	let editingRole: string | null = null;
	let editPrimary = '';
	let editFast = '';
	let editQuality = '';
	let saving = false;
	let rolesLoading = true;

	onMount(async () => {
		rolesLoading = true;
		await loadRoles();
		rolesLoading = false;
	});

	function startEdit(role: ModelRoleInfo) {
		editingRole = role.role;
		editPrimary = role.primary;
		editFast = role.fast;
		editQuality = role.quality;
	}

	function cancelEdit() {
		editingRole = null;
	}

	async function handleSave() {
		if (!editingRole) return;
		saving = true;
		await saveRole(editingRole, {
			primary: editPrimary || undefined,
			fast: editFast || undefined,
			quality: editQuality || undefined,
		});
		saving = false;
		editingRole = null;
	}

	function isInstalled(model: string): boolean {
		return !model || $installedModels.includes(model);
	}
</script>

<div class="assignment">
	<div class="assign-header">
		<h3 class="section-title">Model Assignment</h3>
		<p class="assign-desc">
			Configure which models handle each task type. 
			Each role has three priorities: primary (default), fast (low latency), and quality (best output).
		</p>
	</div>

	{#if rolesLoading}
		<div class="flex items-center gap-2 py-6 justify-center" style="color: var(--oo-fg-muted);">
			<div class="w-5 h-5 border-2 rounded-full animate-spin"
				style="border-color: var(--oo-bd-default); border-top-color: var(--oo-acc-400);" />
			<span class="text-sm">Loading roles...</span>
		</div>
	{:else if $roles.length === 0}
		<div class="empty-state">
			<p>No role assignments found.</p>
			<p class="empty-hint">Model config may not be loaded, or no routing rules are defined.</p>
		</div>
	{:else}
		<div class="roles-list">
			{#each $roles as role}
				<div class="role-card" class:editing={editingRole === role.role}>
					<div class="role-header">
						<span class="role-name">{role.role}</span>
						{#if editingRole !== role.role}
							<button class="edit-btn" on:click={() => startEdit(role)}>Edit</button>
						{/if}
					</div>

					{#if editingRole === role.role}
						<!-- Edit mode -->
						<div class="edit-grid">
							<div class="edit-field">
								<label class="edit-label" for={`edit-primary-${role.role}`}>Primary</label>
								<select id={`edit-primary-${role.role}`} class="edit-select" bind:value={editPrimary}>
									<option value="">— none —</option>
									{#each $installedModels as model}
										<option value={model}>{model}</option>
									{/each}
								</select>
							</div>
							<div class="edit-field">
								<label class="edit-label" for={`edit-fast-${role.role}`}>Fast</label>
								<select id={`edit-fast-${role.role}`} class="edit-select" bind:value={editFast}>
									<option value="">— none —</option>
									{#each $installedModels as model}
										<option value={model}>{model}</option>
									{/each}
								</select>
							</div>
							<div class="edit-field">
								<label class="edit-label" for={`edit-quality-${role.role}`}>Quality</label>
								<select id={`edit-quality-${role.role}`} class="edit-select" bind:value={editQuality}>
									<option value="">— none —</option>
									{#each $installedModels as model}
										<option value={model}>{model}</option>
									{/each}
								</select>
							</div>
						</div>
						<div class="edit-actions">
							<button class="btn-save" on:click={handleSave} disabled={saving}>
								{saving ? 'Saving...' : 'Save'}
							</button>
							<button class="btn-cancel" on:click={cancelEdit}>Cancel</button>
						</div>
					{:else}
						<!-- Display mode -->
						<div class="role-models">
							<div class="model-slot">
								<span class="slot-label">Primary</span>
								<span class="slot-value" class:missing={!isInstalled(role.primary)} class:empty={!role.primary}>
									{role.primary || '—'}
								</span>
							</div>
							<div class="model-slot">
								<span class="slot-label">Fast</span>
								<span class="slot-value" class:missing={!isInstalled(role.fast)} class:empty={!role.fast}>
									{role.fast || '—'}
								</span>
							</div>
							<div class="model-slot">
								<span class="slot-label">Quality</span>
								<span class="slot-value" class:missing={!isInstalled(role.quality)} class:empty={!role.quality}>
									{role.quality || '—'}
								</span>
							</div>
						</div>
					{/if}
				</div>
			{/each}
		</div>
	{/if}

	<!-- Installed models overview -->
	{#if $installedModels.length > 0}
		<div class="installed-section">
			<h4 class="sub-title">Installed Models ({$installedModels.length})</h4>
			<div class="installed-grid">
				{#each $installedModels as model}
					<span class="installed-chip">{model}</span>
				{/each}
			</div>
		</div>
	{/if}
</div>

<style>
	.assignment {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.section-title {
		font-size: 0.9rem;
		font-weight: 600;
		margin: 0 0 0.25rem 0;
		color: var(--oo-fg-primary);
	}

	.sub-title {
		font-size: 0.8rem;
		font-weight: 600;
		margin: 0 0 0.5rem 0;
		color: var(--oo-fg-secondary);
	}

	.assign-desc {
		font-size: 0.75rem;
		color: var(--oo-fg-tertiary);
		margin: 0;
		line-height: 1.4;
	}

	.empty-state {
		text-align: center;
		padding: 2rem;
		color: var(--oo-fg-tertiary);
		font-size: 0.85rem;
	}

	.empty-hint {
		font-size: 0.72rem;
		color: var(--oo-fg-muted);
	}

	/* -- Roles list -- */

	.roles-list {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.role-card {
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-default);
		border-radius: 6px;
		padding: 0.875rem 1rem;
		transition: border-color 0.15s;
	}

	.role-card.editing {
		border-color: var(--oo-acc-400);
	}

	.role-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 0.5rem;
	}

	.role-name {
		font-size: 0.85rem;
		font-weight: 600;
		color: var(--oo-fg-primary);
		text-transform: capitalize;
	}

	.edit-btn {
		background: none;
		border: none;
		color: var(--oo-acc-400);
		font-size: 0.72rem;
		cursor: pointer;
		padding: 0.125rem 0.375rem;
	}

	.edit-btn:hover {
		text-decoration: underline;
	}

	/* Display mode */

	.role-models {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 0.5rem;
	}

	.model-slot {
		display: flex;
		flex-direction: column;
		gap: 0.125rem;
	}

	.slot-label {
		font-size: 0.62rem;
		color: var(--oo-fg-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.slot-value {
		font-family: monospace;
		font-size: 0.72rem;
		color: var(--oo-acc-400);
		padding: 0.25rem 0.375rem;
		background: var(--oo-bg-elevated);
		border-radius: 3px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.slot-value.empty {
		color: var(--oo-fg-muted);
	}

	.slot-value.missing {
		color: var(--oo-error);
		background: var(--oo-error-bg, rgba(239, 68, 68, 0.08));
	}

	/* Edit mode */

	.edit-grid {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 0.5rem;
		margin-bottom: 0.75rem;
	}

	.edit-field {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.edit-label {
		font-size: 0.62rem;
		color: var(--oo-fg-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.edit-select {
		padding: 0.375rem 0.5rem;
		background: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-default);
		border-radius: 4px;
		color: var(--oo-fg-primary);
		font-size: 0.72rem;
		font-family: monospace;
	}

	.edit-actions {
		display: flex;
		gap: 0.5rem;
	}

	.btn-save {
		padding: 0.35rem 0.75rem;
		background: var(--oo-acc-400);
		color: var(--oo-fg-on-accent);
		border: none;
		border-radius: 4px;
		font-size: 0.72rem;
		font-weight: 600;
		cursor: pointer;
	}

	.btn-save:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}

	.btn-save:hover:not(:disabled) {
		background: var(--oo-acc-300);
	}

	.btn-cancel {
		padding: 0.35rem 0.75rem;
		background: transparent;
		border: 1px solid var(--oo-bd-default);
		color: var(--oo-fg-tertiary);
		border-radius: 4px;
		font-size: 0.72rem;
		cursor: pointer;
	}

	.btn-cancel:hover {
		border-color: var(--oo-bd-strong);
		color: var(--oo-fg-secondary);
	}

	/* -- Installed models -- */

	.installed-section {
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-default);
		border-radius: 6px;
		padding: 0.875rem 1rem;
	}

	.installed-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 0.375rem;
	}

	.installed-chip {
		display: inline-block;
		padding: 0.2rem 0.5rem;
		background: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-default);
		border-radius: 3px;
		font-family: monospace;
		font-size: 0.68rem;
		color: var(--oo-fg-secondary);
	}
</style>
