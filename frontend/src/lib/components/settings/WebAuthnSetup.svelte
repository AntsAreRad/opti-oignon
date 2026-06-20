<!--
  WebAuthnSetup.svelte (S127)
  Manage WebAuthn/FIDO2 security keys in the settings panel.

  Features:
  - List registered security keys (name, last used timestamp)
  - "Register new key" button that begins the WebAuthn ceremony
  - Calls navigator.credentials.create() with server-provided options
  - Delete button per credential with confirmation
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { apiGet, apiPost, apiDelete } from '$lib/api/client';

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
	let confirmDeleteId = '';
	let newKeyName = '';
	let showRegisterForm = false;
	let webauthnAvailable = false;

	onMount(async () => {
		// Check browser WebAuthn support
		webauthnAvailable = !!(window.PublicKeyCredential);
		await loadCredentials();
		loading = false;
	});

	async function loadCredentials() {
		try {
			const data = await apiGet<{ credentials: WebAuthnCredential[] }>(
				'/api/security/2fa/webauthn/credentials'
			);
			credentials = data.credentials || [];
		} catch (e: any) {
			error = e.detail || 'Failed to load security keys';
		}
	}

	async function beginRegistration() {
		registerError = '';
		registerSuccess = '';
		registerLoading = true;

		try {
			// Step 1: Get registration options from server
			const options = await apiPost<Record<string, any>>(
				'/api/security/2fa/webauthn/register/begin'
			);

			if (!options.success && options.success !== undefined) {
				registerError = options.message || 'Server rejected registration request';
				registerLoading = false;
				return;
			}

			// Step 2: Convert base64url fields to ArrayBuffer for WebAuthn API
			const publicKey = prepareCreateOptions(options);

			// Step 3: Call browser WebAuthn API
			const credential = await navigator.credentials.create({ publicKey });

			if (!credential) {
				registerError = 'Registration cancelled or timed out.';
				registerLoading = false;
				return;
			}

			// Step 4: Send response back to server
			const attestation = serializeCredential(credential as PublicKeyCredential);
			const result = await apiPost<Record<string, any>>(
				'/api/security/2fa/webauthn/register/complete',
				{
					response: attestation,
					name: newKeyName || 'Security Key',
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
				registerError = e.detail || e.message || 'Registration failed.';
			}
		} finally {
			registerLoading = false;
		}
	}

	async function deleteCredential(credentialId: string) {
		deletingId = credentialId;
		try {
			const result = await apiDelete<{ success: boolean }>(
				`/api/security/2fa/webauthn/credentials/${encodeURIComponent(credentialId)}`
			);
			if (result.success) {
				credentials = credentials.filter(c => c.credential_id !== credentialId);
			}
		} catch (e: any) {
			error = e.detail || 'Failed to delete key';
		} finally {
			deletingId = '';
			confirmDeleteId = '';
		}
	}

	function formatDate(ts: number | null): string {
		if (!ts) return 'Never';
		return new Date(ts * 1000).toLocaleDateString(undefined, {
			year: 'numeric', month: 'short', day: 'numeric',
			hour: '2-digit', minute: '2-digit',
		});
	}

	// -- WebAuthn helpers -------------------------------------------------------

	function base64urlToBuffer(base64url: string): ArrayBuffer {
		let base64 = base64url.replace(/-/g, '+').replace(/_/g, '/');
		while (base64.length % 4 !== 0) base64 += '=';
		const binary = atob(base64);
		const bytes = new Uint8Array(binary.length);
		for (let i = 0; i < binary.length; i++) {
			bytes[i] = binary.charCodeAt(i);
		}
		return bytes.buffer;
	}

	function bufferToBase64url(buffer: ArrayBuffer): string {
		const bytes = new Uint8Array(buffer);
		let binary = '';
		for (const byte of bytes) {
			binary += String.fromCharCode(byte);
		}
		return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
	}

	function prepareCreateOptions(serverOptions: Record<string, any>): PublicKeyCredentialCreationOptions {
		const opts: any = { ...serverOptions };

		// Convert challenge
		if (typeof opts.challenge === 'string') {
			opts.challenge = base64urlToBuffer(opts.challenge);
		}

		// Convert user.id
		if (opts.user && typeof opts.user.id === 'string') {
			opts.user = { ...opts.user, id: base64urlToBuffer(opts.user.id) };
		}

		// Convert excludeCredentials[].id
		if (opts.excludeCredentials) {
			opts.excludeCredentials = opts.excludeCredentials.map((c: any) => ({
				...c,
				id: typeof c.id === 'string' ? base64urlToBuffer(c.id) : c.id,
			}));
		}

		return opts as PublicKeyCredentialCreationOptions;
	}

	function serializeCredential(cred: PublicKeyCredential): Record<string, any> {
		const response = cred.response as AuthenticatorAttestationResponse;
		return {
			id: cred.id,
			rawId: bufferToBase64url(cred.rawId),
			type: cred.type,
			response: {
				clientDataJSON: bufferToBase64url(response.clientDataJSON),
				attestationObject: bufferToBase64url(response.attestationObject),
			},
		};
	}
</script>

<div class="space-y-4">
	<div class="flex items-center justify-between">
		<h4 class="text-sm font-semibold" style="color: var(--oo-fg-primary);">
			Security Keys (WebAuthn/FIDO2)
		</h4>
		{#if webauthnAvailable && !showRegisterForm}
			<button
				class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
				style="background-color: var(--oo-tobacco); color: var(--oo-fg-on-accent);"
				on:click={() => { showRegisterForm = true; registerError = ''; registerSuccess = ''; }}
			>
				Register New Key
			</button>
		{/if}
	</div>

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
				<button
					class="px-4 py-2 rounded text-sm font-medium transition-colors"
					style="background-color: var(--oo-tobacco); color: var(--oo-fg-on-accent);"
					disabled={registerLoading}
					on:click={beginRegistration}
				>
					{#if registerLoading}
						Waiting...
					{:else}
						Register
					{/if}
				</button>
				<button
					class="px-3 py-2 rounded text-sm"
					style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
					on:click={() => { showRegisterForm = false; }}
					disabled={registerLoading}
				>
					Cancel
				</button>
			</div>
			{#if registerLoading}
				<div class="flex items-center gap-2 text-sm" style="color: var(--oo-tobacco);">
					<svg class="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
						<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" opacity="0.3"/>
						<path d="M12 2a10 10 0 019.95 9" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
					</svg>
					Tap your security key or follow the browser prompt...
				</div>
			{/if}
			{#if registerError}
				<p class="text-xs" style="color: var(--oo-fg-error);">{registerError}</p>
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
						{#if confirmDeleteId === cred.credential_id}
							<div class="flex gap-1">
								<button
									class="px-2 py-1 rounded text-xs"
									style="background-color: var(--oo-fg-error); color: var(--oo-fg-on-accent);"
									disabled={deletingId === cred.credential_id}
									on:click={() => deleteCredential(cred.credential_id)}
								>
									{deletingId === cred.credential_id ? '...' : 'Confirm'}
								</button>
								<button
									class="px-2 py-1 rounded text-xs"
									style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
									on:click={() => { confirmDeleteId = ''; }}
								>
									No
								</button>
							</div>
						{:else}
							<button
								class="px-2 py-1 rounded text-xs transition-colors"
								style="color: var(--oo-fg-error); border: 1px solid transparent;"
								on:click={() => { confirmDeleteId = cred.credential_id; }}
								title="Remove this key"
							>
								Remove
							</button>
						{/if}
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>
