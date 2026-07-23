<!--
  RemoteAccessPanel.svelte
  Remote access configuration panel for settings/security.

  Features:
    - Enable/disable remote access (Daily mode only)
    - Generate client certificates per device
    - List active certs with revoke buttons
    - TLS status (cert expiry, CA fingerprint)
    - IP allowlist display
    - Greyed out with message in Bulbe mode

  CSS: uses --oo-* variables only.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getRemoteAccessStatus,
		enableRemoteAccess,
		disableRemoteAccess,
		generateClientCert,
		revokeClientCert,
		type RemoteAccessStatus,
		type ClientCertInfo,
	} from '$lib/api/remoteAccess';

	let status: RemoteAccessStatus | null = null;
	let loading = true;
	let error = '';
	let isBulbe = false;

	// Enable form
	let passphrase = '';
	let enabling = false;
	let enableError = '';
	let enableSuccess = '';

	// Generate cert form
	let deviceName = '';
	let certPassphrase = '';
	let generating = false;
	let generateError = '';
	let generateSuccess = '';

	// Revoke state
	let revoking = '';
	let revokeError = '';

	// Disable state
	let disabling = false;

	onMount(async () => {
		await loadStatus();
		loading = false;
	});

	async function loadStatus() {
		try {
			status = await getRemoteAccessStatus();
			isBulbe = false;
			error = '';
		} catch (e: any) {
			if (e?.status === 403) {
				isBulbe = true;
				error = '';
			} else {
				error = e?.message || 'Failed to load remote access status';
			}
		}
	}

	async function handleEnable() {
		if (passphrase.length < 12) {
			enableError = 'Passphrase must be at least 12 characters.';
			return;
		}
		enabling = true;
		enableError = '';
		enableSuccess = '';
		try {
			const result = await enableRemoteAccess(passphrase);
			if (result.success) {
				enableSuccess = result.message || 'Remote access enabled.';
				passphrase = '';
				await loadStatus();
			} else {
				enableError = result.message || result.error || 'Failed to enable.';
			}
		} catch (e: any) {
			enableError = e?.message || 'Failed to enable remote access.';
		} finally {
			enabling = false;
		}
	}

	async function handleDisable() {
		if (!confirm('Disable remote access? The server will bind to localhost on next restart.')) return;
		disabling = true;
		try {
			await disableRemoteAccess();
			await loadStatus();
		} catch (e: any) {
			error = e?.message || 'Failed to disable.';
		} finally {
			disabling = false;
		}
	}

	async function handleGenerateCert() {
		if (!deviceName.trim()) {
			generateError = 'Device name is required.';
			return;
		}
		if (certPassphrase.length < 8) {
			generateError = 'Passphrase must be at least 8 characters.';
			return;
		}
		generating = true;
		generateError = '';
		generateSuccess = '';
		try {
			const result = await generateClientCert(deviceName.trim(), certPassphrase);
			if (result.success) {
				generateSuccess = `Certificate generated for "${result.device_name}". Fingerprint: ${result.fingerprint?.substring(0, 16)}...`;
				deviceName = '';
				certPassphrase = '';
				await loadStatus();
			} else {
				generateError = result.message || result.error || 'Generation failed.';
			}
		} catch (e: any) {
			generateError = e?.message || 'Failed to generate certificate.';
		} finally {
			generating = false;
		}
	}

	async function handleRevoke(name: string) {
		if (!confirm(`Revoke certificate for "${name}"? This takes effect immediately.`)) return;
		revoking = name;
		revokeError = '';
		try {
			await revokeClientCert(name);
			await loadStatus();
		} catch (e: any) {
			revokeError = e?.message || 'Revocation failed.';
		} finally {
			revoking = '';
		}
	}

	function formatDate(ts: number): string {
		if (!ts) return '-';
		return new Date(ts * 1000).toLocaleDateString(undefined, {
			year: 'numeric',
			month: 'short',
			day: 'numeric',
		});
	}
</script>

<div class="space-y-4">
	<!-- Bulbe mode: disabled state -->
	{#if isBulbe}
		<div
			class="rounded-lg p-6 text-center"
			style="background-color: var(--oo-bg-tertiary); border: 1px solid var(--oo-bd-subtle); opacity: 0.6;"
		>
			<svg class="w-8 h-8 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" style="color: var(--oo-fg-muted);">
				<path stroke-linecap="round" stroke-linejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
			</svg>
			<p class="text-sm font-medium" style="color: var(--oo-fg-muted);">
				Remote access is disabled in Bulbe mode
			</p>
			<p class="text-xs mt-1" style="color: var(--oo-fg-muted);">
				This is a physical constraint enforced at the socket level. The server is bound to 127.0.0.1 and cannot serve remote connections.
			</p>
		</div>

	<!-- Loading -->
	{:else if loading}
		<p class="text-sm" style="color: var(--oo-fg-muted);">Loading remote access status...</p>

	<!-- Error -->
	{:else if error}
		<p class="text-sm" style="color: var(--oo-fg-error);">{error}</p>

	<!-- Active content (Daily mode) -->
	{:else if status}
		<!-- Status Card -->
		<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
			<div class="flex items-center justify-between mb-3">
				<div class="flex items-center gap-2">
					<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" style="color: var(--oo-tobacco);">
						<path stroke-linecap="round" stroke-linejoin="round" d="M12.75 3.03v.568c0 .334.148.65.405.864l1.068.89c.442.369.535 1.01.216 1.49l-.51.766a2.25 2.25 0 01-1.161.886l-.143.048a1.107 1.107 0 00-.57 1.664c.369.555.169 1.307-.427 1.605L9 13.125l.423 1.059a.956.956 0 01-1.652.928l-.679-.906a1.125 1.125 0 00-1.906.172L4.5 15.75l-.612.153M12.75 3.031a9 9 0 00-8.862 12.872M12.75 3.031a9 9 0 016.69 14.036m0 0l-.177-.529A2.25 2.25 0 0017.128 15H16.5l-.324-.324a1.453 1.453 0 00-2.328.377l-.036.073a1.586 1.586 0 01-.982.816l-.99.282c-.55.157-.894.702-.8 1.267l.073.438a2.25 2.25 0 01-1.228 2.39" />
					</svg>
					<h4 class="text-sm font-semibold" style="color: var(--oo-fg-primary);">Remote Access</h4>
				</div>
				{#if status.remote_access_allowed}
					<span class="px-2 py-0.5 rounded text-xs" style="background-color: var(--oo-sage-bg, rgba(120,150,120,0.15)); color: var(--oo-sage);">Enabled</span>
				{:else}
					<span class="px-2 py-0.5 rounded text-xs" style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-muted);">Disabled</span>
				{/if}
			</div>

			{#if status.remote_access_allowed}
				<button
					class="px-3 py-1.5 rounded text-xs transition-colors"
					style="background-color: var(--oo-fg-error); color: white;"
					on:click={handleDisable}
					disabled={disabling}
				>
					{disabling ? 'Disabling...' : 'Disable Remote Access'}
				</button>
			{:else}
				<!-- Enable form -->
				<div class="space-y-2 mt-2">
					<p class="text-xs" style="color: var(--oo-fg-muted);">
						Enable remote access with TLS mutual authentication. You will need to generate client certificates for each device.
					</p>
					<input
						type="password"
						bind:value={passphrase}
						placeholder="CA passphrase (min 12 chars)"
						aria-label="Certificate authority passphrase"
						class="w-full px-3 py-1.5 rounded text-sm"
						style="background-color: var(--oo-bg-subtle); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
					/>
					<button
						class="px-3 py-1.5 rounded text-xs transition-colors"
						style="background-color: var(--oo-tobacco); color: white;"
						on:click={handleEnable}
						disabled={enabling || passphrase.length < 12}
					>
						{enabling ? 'Setting up TLS...' : 'Enable Remote Access'}
					</button>
					{#if enableError}
						<p class="text-xs" style="color: var(--oo-fg-error);">{enableError}</p>
					{/if}
					{#if enableSuccess}
						<p class="text-xs" style="color: var(--oo-sage);">{enableSuccess}</p>
					{/if}
				</div>
			{/if}
		</div>

		<!-- TLS Status -->
		{#if status.tls && status.tls.ca_exists}
			<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
				<h4 class="text-sm font-semibold mb-3" style="color: var(--oo-fg-primary);">TLS Certificate Status</h4>
				<div class="space-y-2 text-xs" style="color: var(--oo-fg-secondary);">
					<div class="flex justify-between">
						<span>CA Fingerprint</span>
						<span class="font-mono" style="color: var(--oo-fg-muted);">{status.tls.ca_fingerprint.substring(0, 24)}...</span>
					</div>
					{#if status.tls.server_cert_expiry}
						<div class="flex justify-between">
							<span>Server Cert Expires</span>
							<span>{status.tls.server_cert_expiry.substring(0, 10)}</span>
						</div>
						<div class="flex justify-between">
							<span>Days Until Expiry</span>
							<span
								style="color: {status.tls.days_until_expiry <= 30 ? 'var(--oo-fg-error)' : 'var(--oo-sage)'};"
							>
								{status.tls.days_until_expiry}
							</span>
						</div>
					{/if}
					{#if status.tls.warning}
						<p class="mt-2 px-2 py-1 rounded" style="background-color: rgba(220,38,38,0.1); color: var(--oo-fg-error);">
							{status.tls.warning}
						</p>
					{/if}
				</div>
			</div>
		{/if}

		<!-- Client Certificates -->
		{#if status.remote_access_allowed}
			<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
				<h4 class="text-sm font-semibold mb-3" style="color: var(--oo-fg-primary);">Client Certificates</h4>

				<!-- Generate new cert -->
				<div class="space-y-2 mb-4 pb-4" style="border-bottom: 1px solid var(--oo-bd-subtle);">
					<p class="text-xs" style="color: var(--oo-fg-muted);">
						Generate a certificate for a new device. You must be at the server (localhost) to do this.
					</p>
					<div class="flex gap-2">
						<input
							type="text"
							bind:value={deviceName}
							placeholder="Device name (e.g. iPhone-Leon)"
							aria-label="Client device name"
							class="flex-1 px-3 py-1.5 rounded text-sm"
							style="background-color: var(--oo-bg-subtle); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
						/>
						<input
							type="password"
							bind:value={certPassphrase}
							placeholder="P12 passphrase (8+ chars)"
							aria-label="P12 export passphrase"
							class="flex-1 px-3 py-1.5 rounded text-sm"
							style="background-color: var(--oo-bg-subtle); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
						/>
					</div>
					<button
						class="px-3 py-1.5 rounded text-xs transition-colors"
						style="background-color: var(--oo-tobacco); color: white;"
						on:click={handleGenerateCert}
						disabled={generating || !deviceName.trim() || certPassphrase.length < 8}
					>
						{generating ? 'Generating...' : 'Generate Certificate'}
					</button>
					{#if generateError}
						<p class="text-xs" style="color: var(--oo-fg-error);">{generateError}</p>
					{/if}
					{#if generateSuccess}
						<p class="text-xs" style="color: var(--oo-sage);">{generateSuccess}</p>
					{/if}
				</div>

				<!-- Cert list -->
				{#if status.tls?.client_certs && status.tls.client_certs.length > 0}
					<div class="space-y-2">
						{#each status.tls.client_certs as cert}
							<div
								class="flex items-center justify-between p-2 rounded"
								style="background-color: var(--oo-bg-subtle); {cert.revoked ? 'opacity: 0.5;' : ''}"
							>
								<div class="text-xs">
									<span class="font-medium" style="color: var(--oo-fg-primary);">{cert.device_name}</span>
									<span class="font-mono ml-2" style="color: var(--oo-fg-muted);">
										{cert.fingerprint.substring(0, 12)}...
									</span>
									<span class="ml-2" style="color: var(--oo-fg-muted);">
										Expires {formatDate(cert.expires_at)}
									</span>
									{#if cert.revoked}
										<span class="ml-2 px-1.5 py-0.5 rounded" style="background-color: rgba(220,38,38,0.1); color: var(--oo-fg-error);">Revoked</span>
									{/if}
								</div>
								{#if !cert.revoked}
									<button
										class="px-2 py-1 rounded text-xs transition-colors"
										style="color: var(--oo-fg-error); border: 1px solid var(--oo-fg-error);"
										on:click={() => handleRevoke(cert.device_name)}
										disabled={revoking === cert.device_name}
									>
										{revoking === cert.device_name ? '...' : 'Revoke'}
									</button>
								{/if}
							</div>
						{/each}
					</div>
				{:else}
					<p class="text-xs" style="color: var(--oo-fg-muted);">No client certificates generated yet.</p>
				{/if}

				{#if revokeError}
					<p class="text-xs mt-2" style="color: var(--oo-fg-error);">{revokeError}</p>
				{/if}
			</div>
		{/if}

		<!-- Security Info -->
		<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
			<h4 class="text-sm font-semibold mb-2" style="color: var(--oo-fg-primary);">Security Notes</h4>
			<div class="text-xs space-y-1" style="color: var(--oo-fg-muted);">
				<p>mTLS (mutual TLS) ensures both server and client authenticate each other via certificates.</p>
				<p>Remote JWT tokens expire in 5 minutes (vs 60 minutes for local sessions).</p>
				<p>Certificate revocations take effect immediately (no caching).</p>
				<p>Three failed auth attempts will revoke all remote sessions automatically.</p>
				<p>Client cert provisioning requires physical access to the server (localhost only).</p>
			</div>
		</div>
	{/if}
</div>
