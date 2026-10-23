<!--
  WebAuthnSetup.svelte
  Manage WebAuthn/FIDO2 security keys in the settings panel.

  Features:
  - List registered security keys (name, last used timestamp)
  - "Register new key" button that begins the WebAuthn ceremony
  - Calls navigator.credentials.create() with server-provided options
  - Delete button per credential with confirmation
  - Registering and removing a key ask the account's password first (StepUpPrompt)
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { apiGet, apiPost, apiDelete } from '$lib/api/client';
	import { creationOptions, serializeAttestation } from '$lib/webauthn';
	import StepUpPrompt from './StepUpPrompt.svelte';
	import { httpRefusal } from '$lib/switches/serverSwitch';

	interface WebAuthnCredential {
		credential_id: string;
		name: string;
		created_at: number;
		last_used: number | null;
	}

	let credentials: WebAuthnCredential[] = [];
	let loading = true;
	let error = '';
	let registerLoading = false;
	let registerError = '';
	let registerSuccess = '';
	let deletingId = '';
	let deleteError = '';
	let confirmDeleteId = '';
	let newKeyName = '';
	let showRegisterForm = false;
	let webauthnAvailable = false;
	let accountRequired = '';

	onMount(async () => {
		// Check browser WebAuthn support
		webauthnAvailable = !!(window.PublicKeyCredential);
		await loadCredentials();
		loading = false;
	});

	async function loadCredentials() {
		try {
			const data = await apiGet<{ credentials: WebAuthnCredential[]; account_required?: boolean; message?: string }>(
				'/api/security/2fa/webauthn/credentials'
			);
			credentials = data.credentials || [];
			accountRequired = data.account_required ? data.message || 'Create the account first.' : '';
		} catch (e: any) {
			error = e.detail || 'Failed to load security keys';
		}
	}

	async function beginRegistration(event: CustomEvent<{ password: string }>) {
		registerError = '';
		registerSuccess = '';
		registerLoading = true;

		try {
			// Step 1: the account's password, then the registration options from the server
			const answer = await apiPost<Record<string, any>>(
				'/api/security/2fa/webauthn/register/begin',
				{ password: event.detail.password }
			);

			if (!answer.success || !answer.options) {
				registerError = answer.message || 'Server rejected registration request';
				return;
			}

			// Step 2: the server's base64url fields as the browser's buffers
			const publicKey = creationOptions(answer.options);

			// Step 3: Call browser WebAuthn API
			const credential = await navigator.credentials.create({ publicKey });

			if (!credential) {
				registerError = 'Registration cancelled or timed out.';
				return;
			}

			// Step 4: Send response back to server
			const result = await apiPost<Record<string, any>>(
				'/api/security/2fa/webauthn/register/complete',
				{
					response: serializeAttestation(credential as PublicKeyCredential),
					credential_name: newKeyName || 'Security Key',
				}
			);

			if (result.success) {
				registerSuccess = 'Security key registered successfully.';
				showRegisterForm = false;
				newKeyName = '';
				await loadCredentials();
			} else {
				registerError = result.message || 'Registration failed.';
			}
		} catch (e: any) {
			if (e.name === 'NotAllowedError') {
				registerError = 'Registration was cancelled or timed out. Please try again.';
			} else if (e.name === 'SecurityError') {
				registerError = 'Security error. Ensure you are using HTTPS or localhost.';
			} else {
				registerError = httpRefusal(e)?.detail || e.message || 'Registration failed.';
			}
		} finally {
			registerLoading = false;
		}
	}

	async function deleteCredential(credentialId: string, event: CustomEvent<{ password: string }>) {
		deletingId = credentialId;
		deleteError = '';
		try {
			const result = await apiDelete<{ success: boolean }>(
				`/api/security/2fa/webauthn/credentials/${encodeURIComponent(credentialId)}`,
				{ password: event.detail.password }
			);
			if (result.success) {
				credentials = credentials.filter(c => c.credential_id !== credentialId);
				confirmDeleteId = '';
			}
		} catch (e: any) {
			// The server's reason: a wrong password, a wait.
			deleteError = httpRefusal(e)?.detail || 'Failed to delete key';
		} finally {
			deletingId = '';
		}
	}

	function formatDate(ts: number | null): string {
		if (!ts) return 'Never';
		return new Date(ts * 1000).toLocaleDateString(undefined, {
			year: 'numeric', month: 'short', day: 'numeric',
			hour: '2-digit', minute: '2-digit',
		});
	}
</script>

<div class="space-y-4">
	<div class="flex items-center justify-end">
		{#if webauthnAvailable && !showRegisterForm && !accountRequired}
			<button
				class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
				style="background-color: var(--oo-acc-fill); color: var(--oo-fg-on-accent);"
				on:click={() => { showRegisterForm = true; registerError = ''; registerSuccess = ''; }}
			>
				Register New Key
			</button>
		{/if}
	</div>

	{#if accountRequired}
		<p class="text-sm oo-account-required">{accountRequired}</p>
	{/if}

	{#if !webauthnAvailable}
		<p class="text-xs" style="color: var(--oo-fg-warning);">
			WebAuthn is not supported in this browser. Use a modern browser with HTTPS or localhost.
		</p>
	{/if}

	{#if registerSuccess}
		<div class="rounded p-3 text-sm" style="background-color: var(--oo-bg-success, rgba(34,197,94,0.1)); color: var(--oo-sage);">
			{registerSuccess}
		</div>
	{/if}

	{#if showRegisterForm}
		<div class="rounded-lg p-4 space-y-3" style="background-color: var(--oo-bg-tertiary); border: 1px solid var(--oo-bd-subtle);">
			<p class="text-sm" style="color: var(--oo-fg-secondary);">
				Give your key a name, then tap or insert your security key when prompted.
			</p>
			<div class="flex gap-2">
				<input
					type="text"
					bind:value={newKeyName}
					placeholder="Key name (e.g. YubiKey Blue)"
					aria-label="Security key nickname"
					class="flex-1 px-3 py-2 rounded text-sm"
					style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
				/>
			</div>
			<StepUpPrompt
				action="Register"
				busy={registerLoading}
				error={registerError}
				on:confirm={beginRegistration}
				on:cancel={() => { showRegisterForm = false; }}
			/>
			{#if registerLoading}
				<div class="flex items-center gap-2 text-sm" style="color: var(--oo-tobacco);">
					<svg class="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
						<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" opacity="0.3"/>
						<path d="M12 2a10 10 0 019.95 9" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
					</svg>
					Tap your security key or follow the browser prompt...
				</div>
			{/if}
		</div>
	{/if}

	{#if loading}
		<p class="text-sm" style="color: var(--oo-fg-muted);">Loading security keys...</p>
	{:else if error}
		<p class="text-sm" style="color: var(--oo-fg-error);">{error}</p>
	{:else if credentials.length === 0}
		<p class="text-sm" style="color: var(--oo-fg-muted);">
			No security keys registered. Add a YubiKey, Google Titan, or platform authenticator.
		</p>
	{:else}
		<div class="space-y-2">
			{#each credentials as cred (cred.credential_id)}
				<div class="space-y-2">
				<div
					class="flex items-center justify-between rounded-lg p-3"
					style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);"
				>
					<div class="min-w-0 flex-1">
						<div class="flex items-center gap-2">
							<svg class="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" style="color: var(--oo-tobacco);">
								<path stroke-linecap="round" stroke-linejoin="round" d="M15.75 5.25a3 3 0 013 3m3 0a6 6 0 01-7.029 5.912c-.563-.097-1.159.026-1.563.43L10.5 17.25H8.25v2.25H6v2.25H2.25v-2.818c0-.597.237-1.17.659-1.591l6.499-6.499c.404-.404.527-1 .43-1.563A6 6 0 1121.75 8.25z" />
							</svg>
							<span class="text-sm font-medium truncate" style="color: var(--oo-fg-primary);">
								{cred.name || 'Unnamed Key'}
							</span>
						</div>
						<div class="flex gap-3 mt-1 text-xs" style="color: var(--oo-fg-muted);">
							<span>Added {formatDate(cred.created_at)}</span>
							<span>Last used: {formatDate(cred.last_used)}</span>
						</div>
					</div>
					<div class="shrink-0 ml-3">
						{#if confirmDeleteId !== cred.credential_id}
							<button
								class="px-2 py-1 rounded text-xs transition-colors"
								style="color: var(--oo-fg-error); border: 1px solid transparent;"
								on:click={() => { confirmDeleteId = cred.credential_id; deleteError = ''; }}
								title="Remove this key"
							>
								Remove
							</button>
						{/if}
					</div>
				</div>
				{#if confirmDeleteId === cred.credential_id}
					<StepUpPrompt
						action="Remove key"
						busy={deletingId === cred.credential_id}
						error={deleteError}
						on:confirm={(e) => deleteCredential(cred.credential_id, e)}
						on:cancel={() => { confirmDeleteId = ''; }}
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
