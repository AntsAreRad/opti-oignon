<!--
  ModelAssignment.svelte
  Visual editor for model-to-role routing configuration.
  Displays roles (task types) with primary/fast/quality dropdowns
  populated from installed Ollama models.

  It renders from the roles store (lib/stores/modelRoles.ts): a read under
  way, a read that failed (with its reason and a retry), a read that found
  no role, and a save the server refused, shown under its own role with the
  editor left open on the reader's choices. A retry reads quietly: the
  failure and its retry stay on screen until the answer comes, so the retry
  keeps focus, and roles read at last take it.
-->
<script lang="ts">
	import { onMount, tick } from 'svelte';
	import Button from '$lib/ds/Button.svelte';
	import InlineError from '$lib/ds/InlineError.svelte';
	import {
		roles,
		installedModels,
		rolesRead,
		saveErrors,
		loadRoles,
		saveRole,
		forgetSaveError,
	} from '$lib/stores/modelRoles';
	import type { ModelRoleInfo } from '$lib/types';

	let editingRole: string | null = null;
	let editPrimary = '';
	let editFast = '';
	let editQuality = '';
	let saving = false;
	let retrying = false;
	let panel: HTMLElement | undefined;

	onMount(() => {
		loadRoles();
	});

	function startEdit(role: ModelRoleInfo) {
		editingRole = role.role;
		editPrimary = role.primary;
		editFast = role.fast;
		editQuality = role.quality;
	}

	function cancelEdit() {
		if (editingRole) forgetSaveError(editingRole);
		editingRole = null;
	}

	async function handleSave() {
		if (!editingRole) return;
		saving = true;
		const saved = await saveRole(editingRole, {
			primary: editPrimary || undefined,
			fast: editFast || undefined,
			quality: editQuality || undefined,
		});
		saving = false;
		if (saved) editingRole = null;
	}

	/** Reads the roles again, the failure kept until the answer; then focus goes to the roles. */
	async function retryRead() {
		if (retrying) return;
		retrying = true;
		await loadRoles(true);
		retrying = false;
		if ($rolesRead.state === 'ok') {
			await tick();
			panel?.focus();
		}
	}

	function isInstalled(model: string): boolean {
		return !model || $installedModels.includes(model);
	}
</script>

<div class="assignment" tabindex="-1" bind:this={panel}>
	<div class="assign-header">
		<p class="assign-desc">
			Configure which models handle each task type. 
			Each role has three priorities: primary (default), fast (low latency), and quality (best output).
		</p>
	</div>

	{#if $rolesRead.state === 'idle' || $rolesRead.state === 'loading'}
		<p class="assign-status">Loading roles</p>
	{:else if $rolesRead.state === 'error'}
		<div class="assign-failure" role="alert">
			<p>Could not read the roles: {$rolesRead.reason ?? 'the server did not answer'}</p>
			<Button variant="secondary" size="sm" iconLeft="retry" on:click={retryRead}>Retry</Button>
		</div>
	{:else if $roles.length === 0}
		<div class="empty-state">
			<p>No role assignments found.</p>
			<p class="empty-hint">Model config may not be loaded, or no routing rules are defined.</p>
		</div>
	{:else}
		<div class="roles-list">
			{#each $roles as role}
				<div class="role-card" class:editing={editingRole === role.role} data-role={role.role}>
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
					{#if $saveErrors[role.role]}
						<div class="role-error">
							<InlineError message={`The ${role.role} role was not saved: ${$saveErrors[role.role]}`} />
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

	.assign-status {
		margin: 0;
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
	}

	.assign-failure {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--oo-space-3);
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-sm);
	}

	.assign-failure p {
		margin: 0;
	}

	.role-error {
		margin-top: var(--oo-space-3);
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
		background: var(--oo-acc-fill);
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
		background: var(--oo-acc-fill-hover);
		color: var(--oo-fg-on-accent);
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
