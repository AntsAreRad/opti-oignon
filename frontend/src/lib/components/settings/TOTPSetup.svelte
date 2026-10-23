<!--
  TOTPSetup.svelte
  TOTP setup and management in the settings panel.

  Features:
  - Shows QR code from /api/security/2fa/totp/setup
  - Manual secret display (toggle)
  - 6-digit verification input with auto-submit
  - Success confirmation
  - Disable TOTP option
  - Setting up and disabling ask the account's password first (StepUpPrompt)
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { apiGet, apiPost, apiDelete } from '$lib/api/client';
	import StepUpPrompt from './StepUpPrompt.svelte';
	import { httpRefusal } from '$lib/switches/serverSwitch';

	interface RetiredFactors {
		at: number;
		totp: number;
		webauthn: number;
		recovery_codes: number;
		app_passwords: number;
	}

	interface TwoFAStatus {
		totp_available: boolean;
		totp_enabled: boolean;
		totp_verified: boolean;
		/** Single-user mode with no account to give a factor to. */
		account_required?: boolean;
		message?: string;
		/** False while single-user mode lets every request through: nothing asks the factor yet. */
		asked_at_login?: boolean;
		/** The factors set up before they belonged to an account, removed: never asked at login. */
		retired_unbound?: RetiredFactors;
	}

	let status: TwoFAStatus | null = null;
	let loading = true;
	let error = '';

	// Setup flow state
	let setupActive = false;
	let setupLoading = false;
	let qrCodeBase64 = '';
	let secret = '';
	let uri = '';
	let showSecret = false;

	// Verification
	let verifyCode = '';
	let verifyLoading = false;
	let verifyError = '';
	let verifySuccess = '';

	// Disable
	let confirmDisable = false;
	let disableLoading = false;

	// The password asked before setting up or disabling
	let stepUpFor: 'setup' | 'disable' | null = null;
	let stepUpBusy = false;
	let stepUpError = '';

	onMount(async () => {
		await loadStatus();
		loading = false;
	});

	async function loadStatus() {
		try {
			const data = await apiGet<TwoFAStatus>('/api/security/2fa/status');
			status = data;
		} catch (e: any) {
			error = e.detail || 'Failed to load 2FA status';
		}
	}

	function askStepUp(action: 'setup' | 'disable') {
		stepUpFor = action;
		stepUpError = '';
	}

	async function onStepUp(event: CustomEvent<{ password: string }>) {
		const { password } = event.detail;
		stepUpBusy = true;
		stepUpError = '';
		try {
			if (stepUpFor === 'setup') await beginSetup(password);
			else if (stepUpFor === 'disable') await disableTotp(password);
			stepUpFor = null;
		} catch (e: any) {
			// The server's reason: a wrong password, a wait, TOTP already active.
			stepUpError = httpRefusal(e)?.detail || e.message || 'Refused.';
		} finally {
			stepUpBusy = false;
		}
	}

	async function beginSetup(password: string) {
		verifyError = '';
		verifySuccess = '';
		const result = await apiPost<Record<string, any>>(
			'/api/security/2fa/totp/setup',
			{ password }
		);
		if (result.success) {
			setupActive = true;
			qrCodeBase64 = result.qr_code_base64 || '';
			secret = result.secret || '';
			uri = result.uri || '';
		} else {
			throw new Error(result.message || 'Failed to generate TOTP secret.');
		}
	}

	async function verifyTotp() {
		if (verifyCode.length !== 6) return;

		verifyLoading = true;
		verifyError = '';

		try {
			const result = await apiPost<Record<string, any>>(
				'/api/security/2fa/totp/verify',
				{ code: verifyCode }
			);
			if (result.success) {
				verifySuccess = 'Authenticator app verified and activated.';
				setupActive = false;
				qrCodeBase64 = '';
				secret = '';
				verifyCode = '';
				await loadStatus();
			} else {
				verifyError = result.message || 'Invalid code. Please try again.';
				verifyCode = '';
			}
		} catch (e: any) {
			verifyError = e.detail || 'Verification failed.';
			verifyCode = '';
		} finally {
			verifyLoading = false;
		}
	}

	async function disableTotp(password: string) {
		disableLoading = true;
		try {
			await apiDelete<{ success: boolean }>('/api/security/2fa/totp', { password });
			confirmDisable = false;
			verifySuccess = '';
			await loadStatus();
		} finally {
			disableLoading = false;
		}
	}

	function handleCodeInput(event: Event) {
		const input = event.target as HTMLInputElement;
		// Strip non-digits
		input.value = input.value.replace(/\D/g, '').slice(0, 6);
		verifyCode = input.value;
		// Auto-submit on 6 digits
		if (verifyCode.length === 6) {
			verifyTotp();
		}
	}

	function cancelSetup() {
		setupActive = false;
		qrCodeBase64 = '';
		secret = '';
		verifyCode = '';
		verifyError = '';
	}
	/** A count with its noun, singular for one. */
	function counted(n: number, one: string, many: string): string {
		return `${n} ${n === 1 ? one : many}`;
	}
</script>

<div class="space-y-4">
	<div class="flex items-center justify-end">
		{#if status?.totp_verified}
			<span class="px-2 py-0.5 rounded text-xs font-medium" style="background-color: var(--oo-bg-success, rgba(34,197,94,0.1)); color: var(--oo-sage);">
				Active
			</span>
		{/if}
	</div>

	{#if loading}
		<p class="text-sm" style="color: var(--oo-fg-muted);">Loading TOTP status...</p>
	{:else if error}
		<p class="text-sm" style="color: var(--oo-fg-error);">{error}</p>
	{:else if verifySuccess && !setupActive}
		<div class="rounded p-3 text-sm" style="background-color: var(--oo-bg-success, rgba(34,197,94,0.1)); color: var(--oo-sage);">
			{verifySuccess}
		</div>
	{/if}

	{#if !loading && status?.account_required}
		<p class="text-sm oo-account-required">{status.message}</p>
	{:else if !loading && status}
		{#if status.retired_unbound}
			<div class="rounded p-3 text-xs" style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-subtle);" role="status">
				Second factors set up here before factors belonged to an account were never asked at login, and were
				removed ({counted(status.retired_unbound.totp, 'authenticator app', 'authenticator apps')},
				{counted(status.retired_unbound.webauthn, 'security key', 'security keys')},
				{counted(status.retired_unbound.recovery_codes, 'recovery code', 'recovery codes')},
				{counted(status.retired_unbound.app_passwords, 'app password', 'app passwords')}). Set up again the
				ones you use.
			</div>
		{/if}
		{#if status.asked_at_login === false}
			<p class="text-xs" style="color: var(--oo-fg-muted);">
				Single-user mode: nothing asks a second factor at login until authentication is on.
			</p>
		{/if}
		{#if status.totp_verified && !setupActive}
			<!-- TOTP is active -->
			<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
				<div class="flex items-center gap-2 mb-2">
					<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" style="color: var(--oo-sage);">
						<path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
					</svg>
					<span class="text-sm" style="color: var(--oo-fg-secondary);">
						Authenticator app is configured and active.
					</span>
				</div>
				<p class="text-xs mb-3" style="color: var(--oo-fg-muted);">
					Use your authenticator app (Google Authenticator, Authy, etc.) to generate codes during login.
				</p>
				{#if confirmDisable}
					<StepUpPrompt
						action="Disable TOTP"
						busy={stepUpBusy || disableLoading}
						error={stepUpError}
						on:confirm={onStepUp}
						on:cancel={() => { confirmDisable = false; stepUpFor = null; }}
					/>
				{:else}
					<button
						class="px-3 py-1.5 rounded text-xs transition-colors"
						style="color: var(--oo-fg-error); border: 1px solid var(--oo-bd-subtle);"
						on:click={() => { confirmDisable = true; askStepUp('disable'); }}
					>
						Disable TOTP
					</button>
				{/if}
			</div>
		{:else if setupActive}
			<!-- Setup flow -->
			<div class="rounded-lg p-4 space-y-4" style="background-color: var(--oo-bg-tertiary); border: 1px solid var(--oo-bd-subtle);">
				{#if setupLoading}
					<p class="text-sm" style="color: var(--oo-fg-muted);">Generating secret...</p>
				{:else}
					<!-- Step 1: QR Code -->
					<div class="space-y-2">
						<p class="text-sm font-medium" style="color: var(--oo-fg-primary);">
							Step 1: Scan this QR code with your authenticator app
						</p>
						{#if qrCodeBase64}
							<div class="flex justify-center p-4 rounded" style="background-color: var(--oo-qr-bg);">
								<img
									src={qrCodeBase64}
									alt="TOTP QR Code"
									class="w-40 h-40"
									style="image-rendering: pixelated;"
								/>
							</div>
						{:else}
							<p class="text-xs" style="color: var(--oo-fg-warning);">
								QR code not available. Use the manual entry below.
							</p>
						{/if}

						<!-- Manual secret toggle -->
						<div>
							<button
								class="text-xs underline"
								style="color: var(--oo-fg-muted);"
								on:click={() => { showSecret = !showSecret; }}
							>
								{showSecret ? 'Hide manual secret' : 'Show manual secret'}
							</button>
							{#if showSecret && secret}
								<div class="mt-2 p-2 rounded font-mono text-sm select-all break-all"
									style="background-color: var(--oo-card-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
								>
									{secret}
								</div>
								<p class="text-xs mt-1" style="color: var(--oo-fg-muted);">
									Enter this key manually in your authenticator app.
								</p>
							{/if}
						</div>
					</div>

					<!-- Step 2: Verification -->
					<div class="space-y-2">
						<p class="text-sm font-medium" style="color: var(--oo-fg-primary);">
							Step 2: Enter the 6-digit code from your app
						</p>
						<div class="flex gap-2">
							<input
								type="text"
								inputmode="numeric"
								autocomplete="one-time-code"
								maxlength="6"
								placeholder="000000"
								aria-label="Enter TOTP verification code"
								value={verifyCode}
								on:input={handleCodeInput}
								disabled={verifyLoading}
								class="w-32 px-3 py-2 rounded text-center text-lg font-mono tracking-widest"
								style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
							/>
							{#if verifyLoading}
								<div class="flex items-center">
									<svg class="w-5 h-5 animate-spin" style="color: var(--oo-tobacco);" viewBox="0 0 24 24" fill="none">
										<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" opacity="0.3"/>
										<path d="M12 2a10 10 0 019.95 9" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
									</svg>
								</div>
							{/if}
						</div>
						<p class="text-xs" style="color: var(--oo-fg-muted);">
							Code auto-submits when 6 digits are entered.
						</p>
						{#if verifyError}
							<p class="text-xs" style="color: var(--oo-fg-error);">{verifyError}</p>
						{/if}
					</div>

					<!-- Cancel -->
					<button
						class="px-3 py-1.5 rounded text-xs"
						style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
						on:click={cancelSetup}
					>
						Cancel Setup
					</button>
				{/if}
			</div>
		{:else}
			<!-- Not configured yet -->
			<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
				<p class="text-sm mb-3" style="color: var(--oo-fg-secondary);">
					Add an authenticator app as a backup 2FA method. Works with Google Authenticator, Authy, and other TOTP-compatible apps.
				</p>
				{#if stepUpFor === 'setup'}
					<StepUpPrompt
						action="Set up"
						busy={stepUpBusy}
						error={stepUpError}
						on:confirm={onStepUp}
						on:cancel={() => { stepUpFor = null; }}
					/>
				{:else}
					<button
						class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
						style="background-color: var(--oo-acc-fill); color: var(--oo-fg-on-accent);"
						on:click={() => askStepUp('setup')}
					>
						Set Up Authenticator
					</button>
				{/if}
			</div>
		{/if}
	{/if}
</div>

<style>
	.oo-account-required {
		color: var(--oo-fg-secondary);
	}
</style>
