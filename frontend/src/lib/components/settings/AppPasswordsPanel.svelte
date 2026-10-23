<!--
  AppPasswordsPanel.svelte
  App-specific password management in security settings.

  Features:
  - Create new app password with label
  - Show generated password ONCE with copy button
  - List existing app passwords (name, created, last used, status)
  - Revoke button per password with confirmation
  - Creating and revoking ask the account's password first (StepUpPrompt)
  - An app password is never a second factor at login, and no client signs in with one yet
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { apiGet, apiPost, apiDelete } from '$lib/api/client';
	import StepUpPrompt from './StepUpPrompt.svelte';
	import { httpRefusal } from '$lib/switches/serverSwitch';

	interface AppPassword {
		password_id: string;
		name: string;
		created_at: number;
		last_used: number | null;
		revoked: number;
	}

	let passwords: AppPassword[] = [];
	let loading = true;
	let error = '';

	// Create flow
	let showCreate = false;
	let newLabel = '';
	let creating = false;
	let createdPassword = '';
	let createdName = '';
	let copied = false;

	// Revoke
	let confirmRevokeId = '';
	let revoking = '';
	let createError = '';
	let revokeError = '';
	let accountRequired = '';

	onMount(async () => {
		await loadPasswords();
		loading = false;
	});

	async function loadPasswords() {
		try {
			const data = await apiGet<{ passwords: AppPassword[]; account_required?: boolean; message?: string }>(
				'/api/security/2fa/app-passwords'
			);
			passwords = (data.passwords || []).filter(p => !p.revoked);
			accountRequired = data.account_required ? data.message || 'Create the account first.' : '';
		} catch (e: any) {
			error = e.detail || 'Failed to load app passwords';
		}
	}

	async function createPassword(event: CustomEvent<{ password: string }>) {
		if (!newLabel.trim()) {
			createError = 'Give the password a label first.';
			return;
		}
		creating = true;
		createError = '';

		try {
			const result = await apiPost<Record<string, any>>(
				'/api/security/2fa/app-passwords',
				{ name: newLabel.trim(), password: event.detail.password }
			);
			if (result.success) {
				createdPassword = result.password || '';
				createdName = result.name || newLabel.trim();
				newLabel = '';
				showCreate = false;
				await loadPasswords();
			} else {
				createError = result.message || 'Failed to create app password.';
			}
		} catch (e: any) {
			// The server's reason: a wrong password, a wait.
			createError = httpRefusal(e)?.detail || 'Failed to create app password.';
		} finally {
			creating = false;
		}
	}

	async function revokePassword(passwordId: string, event: CustomEvent<{ password: string }>) {
		revoking = passwordId;
		revokeError = '';
		try {
			await apiDelete<{ success: boolean }>(
				`/api/security/2fa/app-passwords/${encodeURIComponent(passwordId)}`,
				{ password: event.detail.password }
			);
			confirmRevokeId = '';
			await loadPasswords();
		} catch (e: any) {
			revokeError = httpRefusal(e)?.detail || 'Failed to revoke password.';
		} finally {
			revoking = '';
		}
	}

	async function copyPassword() {
		try {
			await navigator.clipboard.writeText(createdPassword);
			copied = true;
			setTimeout(() => { copied = false; }, 2000);
		} catch {
			error = 'Clipboard access denied. Please copy manually.';
		}
	}

	function dismissCreated() {
		createdPassword = '';
		createdName = '';
	}

	function formatDate(ts: number | null): string {
		if (!ts) return 'Never';
		return new Date(ts * 1000).toLocaleDateString(undefined, {
			year: 'numeric', month: 'short', day: 'numeric',
		});
	}

</script>

<div class="space-y-4">
	{#if accountRequired}
		<p class="text-sm oo-account-required">{accountRequired}</p>
	{/if}
	<div class="flex items-center justify-end">
		{#if !showCreate && !createdPassword && !accountRequired}
			<button
				class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
				style="background-color: var(--oo-acc-fill); color: var(--oo-fg-on-accent);"
				on:click={() => { showCreate = true; error = ''; }}
			>
				Create App Password
			</button>
		{/if}
	</div>

	<p class="text-xs" style="color: var(--oo-fg-muted);">
		An app password is meant for a CLI tool or a script, but no client signs in with one yet. Each one can be revoked.
	</p>

	{#if error}
		<p class="text-sm" style="color: var(--oo-fg-error);">{error}</p>
	{/if}

	<!-- Newly created password (shown once) -->
	{#if createdPassword}
		<div class="rounded-lg p-4 space-y-3"
			style="background-color: var(--oo-bg-tertiary); border: 2px solid var(--oo-fg-warning);">
			<div class="flex items-start gap-2">
				<svg class="w-5 h-5 shrink-0" style="color: var(--oo-fg-warning);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
					<path stroke-linecap="round" stroke-linejoin="round"
						d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
					/>
				</svg>
				<p class="text-sm font-medium" style="color: var(--oo-fg-warning);">
					Copy this password now. It will not be shown again.
				</p>
			</div>

			<div class="space-y-1">
				<p class="text-xs" style="color: var(--oo-fg-muted);">Label: {createdName}</p>
				<div class="flex gap-2">
					<code
						class="flex-1 px-3 py-2 rounded font-mono text-sm select-all break-all"
						style="background-color: var(--oo-card-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
					>
						{createdPassword}
					</code>
					<button
						class="shrink-0 px-3 py-2 rounded text-xs font-medium transition-colors"
						style="background-color: var(--oo-acc-fill); color: var(--oo-fg-on-accent);"
						on:click={copyPassword}
					>
						{copied ? 'Copied!' : 'Copy'}
					</button>
				</div>
			</div>

			<button
				class="px-3 py-1.5 rounded text-xs transition-colors"
				style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
				on:click={dismissCreated}
			>
				I have saved this password
			</button>
		</div>
	{/if}

	<!-- Create form -->
	{#if showCreate}
		<div class="rounded-lg p-4 space-y-3"
			style="background-color: var(--oo-bg-tertiary); border: 1px solid var(--oo-bd-subtle);">
			<p class="text-sm" style="color: var(--oo-fg-secondary);">
				Give this password a descriptive label so you can identify it later.
			</p>
			<div class="flex gap-2">
				<input
					type="text"
					bind:value={newLabel}
					placeholder="e.g. CLI on workstation, backup script"
					aria-label="App password description"
					class="flex-1 px-3 py-2 rounded text-sm"
					style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
				/>
			</div>
			<StepUpPrompt
				action={creating ? 'Creating...' : 'Create'}
				busy={creating}
				error={createError}
				on:confirm={createPassword}
				on:cancel={() => { showCreate = false; newLabel = ''; createError = ''; }}
			/>
		</div>
	{/if}

	<!-- Password list -->
	{#if loading}
		<p class="text-sm" style="color: var(--oo-fg-muted);">Loading app passwords...</p>
	{:else if passwords.length === 0 && !createdPassword}
		<p class="text-sm" style="color: var(--oo-fg-muted);">
			No app passwords created yet.
		</p>
	{:else if passwords.length > 0}
		<div class="space-y-2">
			{#each passwords as pw (pw.password_id)}
				<div class="space-y-2">
				<div
					class="flex items-center justify-between rounded-lg p-3"
					style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);"
				>
					<div class="min-w-0 flex-1">
						<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">
							{pw.name}
						</span>
						<div class="flex gap-3 mt-0.5 text-xs" style="color: var(--oo-fg-muted);">
							<span>Created {formatDate(pw.created_at)}</span>
							<span>Last used: {formatDate(pw.last_used)}</span>
						</div>
					</div>
					<div class="shrink-0 ml-3">
						{#if confirmRevokeId !== pw.password_id}
							<button
								class="px-2 py-1 rounded text-xs transition-colors"
								style="color: var(--oo-fg-error);"
								on:click={() => { confirmRevokeId = pw.password_id; revokeError = ''; }}
								title="Revoke this password"
							>
								Revoke
							</button>
						{/if}
					</div>
				</div>
				{#if confirmRevokeId === pw.password_id}
					<StepUpPrompt
						action={revoking === pw.password_id ? 'Revoking...' : 'Revoke'}
						busy={revoking === pw.password_id}
						error={revokeError}
						on:confirm={(e) => revokePassword(pw.password_id, e)}
						on:cancel={() => { confirmRevokeId = ''; }}
					/>
				{/if}
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.oo-account-required {
		color: var(--oo-fg-secondary);
	}
</style>
