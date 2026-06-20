<!--
  KeyCeremonyPanel.svelte (S129)
  Encryption key ceremony UI for first-time setup.

  Features:
    - Step-by-step wizard: choose method, set passphrase, confirm, success
    - Passphrase strength meter (client-side, zxcvbn-style)
    - Key status display (algorithm, KDF, SecureBytes, mlock)
    - PQC backup signature status and key generation
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getEncryptionStatus,
		setupEncryptionPassphrase,
		setupEncryptionRandom,
		getPqcStatus,
		generatePqcKeys,
		deletePqcKeys,
		scorePassphrase,
		type EncryptionStatus,
		type PqcStatus,
		type StrengthResult,
	} from '../../api/keyCeremony';

	// -- State ---------------------------------------------------------------

	let encStatus: EncryptionStatus | null = null;
	let pqcStatus: PqcStatus | null = null;
	let loading = true;
	let error = '';
	let successMessage = '';

	// Wizard state
	type WizardStep = 'choose' | 'passphrase' | 'confirm' | 'success';
	let wizardStep: WizardStep = 'choose';
	let setupMode: 'passphrase' | 'random' = 'passphrase';

	// Passphrase fields
	let passphrase = '';
	let passphraseConfirm = '';
	let showPassphrase = false;
	let strength: StrengthResult = scorePassphrase('');
	let setupInProgress = false;
	let setupError = '';

	// PQC
	let pqcGenerating = false;
	let pqcError = '';

	// -- Lifecycle ------------------------------------------------------------

	onMount(async () => {
		await loadStatus();
		loading = false;
	});

	async function loadStatus() {
		try {
			encStatus = await getEncryptionStatus();
		} catch {
			error = 'Failed to load encryption status';
		}
		try {
			pqcStatus = await getPqcStatus();
		} catch {
			// PQC may not be available
		}
	}

	// -- Passphrase scoring ---------------------------------------------------

	$: strength = scorePassphrase(passphrase);
	$: canProceedPassphrase = strength.score >= 1 && passphrase.length >= 8;
	$: passphraseMatch = passphrase === passphraseConfirm && passphraseConfirm.length > 0;

	// -- Wizard actions -------------------------------------------------------

	function startPassphrase() {
		setupMode = 'passphrase';
		wizardStep = 'passphrase';
		passphrase = '';
		passphraseConfirm = '';
		setupError = '';
	}

	async function startRandom() {
		setupMode = 'random';
		setupError = '';
		setupInProgress = true;
		try {
			const result = await setupEncryptionRandom();
			if (result.setup) {
				encStatus = result.status;
				wizardStep = 'success';
				successMessage = 'Encryption configured with a random key. The key is stored in your keyfile.';
			} else {
				setupError = result.detail || 'Setup failed';
			}
		} catch (e: unknown) {
			setupError = e instanceof Error ? e.message : 'Setup failed';
		} finally {
			setupInProgress = false;
		}
	}

	function goToConfirm() {
		wizardStep = 'confirm';
		passphraseConfirm = '';
	}

	async function confirmSetup() {
		if (!passphraseMatch) return;
		setupInProgress = true;
		setupError = '';
		try {
			const result = await setupEncryptionPassphrase(passphrase);
			if (result.setup) {
				encStatus = result.status;
				wizardStep = 'success';
				successMessage = 'Encryption configured with your passphrase. Remember it - you will need it to access your data.';
			} else {
				setupError = result.detail || 'Setup failed';
			}
		} catch (e: unknown) {
			setupError = e instanceof Error ? e.message : 'Setup failed';
		} finally {
			setupInProgress = false;
			passphrase = '';
			passphraseConfirm = '';
		}
	}

	function resetWizard() {
		wizardStep = 'choose';
		passphrase = '';
		passphraseConfirm = '';
		setupError = '';
		successMessage = '';
	}

	// -- PQC actions ----------------------------------------------------------

	async function handleGeneratePqcKeys() {
		pqcGenerating = true;
		pqcError = '';
		try {
			const result = await generatePqcKeys();
			if (result.success) {
				pqcStatus = result.status;
			}
		} catch (e: unknown) {
			pqcError = e instanceof Error ? e.message : 'Key generation failed';
		} finally {
			pqcGenerating = false;
		}
	}

	async function handleDeletePqcKeys() {
		pqcError = '';
		try {
			const result = await deletePqcKeys();
			pqcStatus = result.status;
		} catch (e: unknown) {
			pqcError = e instanceof Error ? e.message : 'Key deletion failed';
		}
	}
</script>

<div class="space-y-4">

	{#if loading}
		<p class="text-sm" style="color: var(--oo-fg-muted);">Loading encryption status...</p>

	{:else if error}
		<p class="text-sm" style="color: var(--oo-fg-error);">{error}</p>

	{:else}

		<!-- Current Key Status -->
		<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
			<h4 class="text-sm font-semibold mb-3" style="color: var(--oo-fg-primary);">Encryption Status</h4>

			{#if encStatus}
				<div class="grid grid-cols-2 gap-2 text-xs">
					<div style="color: var(--oo-fg-muted);">State</div>
					<div style="color: {encStatus.enabled ? 'var(--oo-sage)' : 'var(--oo-fg-warning)'};">
						{encStatus.enabled ? 'Active' : 'Not configured'}
					</div>

					{#if encStatus.has_key}
						<div style="color: var(--oo-fg-muted);">Algorithm</div>
						<div class="font-mono" style="color: var(--oo-fg-secondary);">{encStatus.algorithm}</div>

						<div style="color: var(--oo-fg-muted);">Key Derivation</div>
						<div class="font-mono" style="color: var(--oo-fg-secondary);">{encStatus.kdf}</div>

						<div style="color: var(--oo-fg-muted);">Crypto Backend</div>
						<div class="font-mono" style="color: var(--oo-fg-secondary);">{encStatus.crypto_backend}</div>

						<div style="color: var(--oo-fg-muted);">Memory Protection</div>
						<div style="color: {encStatus.secure_bytes_active ? 'var(--oo-sage)' : 'var(--oo-fg-muted)'};">
							{encStatus.secure_bytes_active ? 'SecureBytes active' : 'Standard'}
							{#if encStatus.key_mlocked}
								<span class="ml-1" style="color: var(--oo-sage);">(mlock)</span>
							{/if}
						</div>

						<div style="color: var(--oo-fg-muted);">Key Source</div>
						<div style="color: var(--oo-fg-secondary);">
							{#if encStatus.env_key_set}
								Environment variable
							{:else if encStatus.keyfile_exists}
								Keyfile
							{:else}
								Unknown
							{/if}
						</div>
					{/if}
				</div>
			{/if}
		</div>

		<!-- Setup Wizard (only if not yet configured) -->
		{#if !encStatus?.enabled}
			<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
				<h4 class="text-sm font-semibold mb-3" style="color: var(--oo-fg-primary);">Key Setup Ceremony</h4>

				{#if wizardStep === 'choose'}
					<p class="text-xs mb-4" style="color: var(--oo-fg-muted);">
						Choose how to generate your encryption key. A passphrase-derived key lets you recover
						your data if the keyfile is lost. A random key is stronger but requires a keyfile backup.
					</p>
					<div class="flex gap-3">
						<button
							class="flex-1 px-3 py-2 rounded text-xs font-medium transition-colors"
							style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
							on:click={startPassphrase}
						>
							Use Passphrase
						</button>
						<button
							class="flex-1 px-3 py-2 rounded text-xs font-medium transition-colors"
							style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
							on:click={startRandom}
							disabled={setupInProgress}
						>
							{setupInProgress ? 'Generating...' : 'Random Key'}
						</button>
					</div>

				{:else if wizardStep === 'passphrase'}
					<div class="space-y-3">
						<div>
							<label class="block text-xs mb-1" style="color: var(--oo-fg-muted);" for="ceremony-passphrase">
								Enter your passphrase
							</label>
							<div class="relative">
								<input
									id="ceremony-passphrase"
									type={showPassphrase ? 'text' : 'password'}
									value={passphrase}
									on:input={(event) => (passphrase = event.currentTarget.value)}
									placeholder="Minimum 8 characters, 14+ recommended"
									class="w-full px-3 py-2 rounded text-sm font-mono"
									style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
								/>
								<button
									class="absolute right-2 top-1/2 -translate-y-1/2 text-xs px-1"
									style="color: var(--oo-fg-muted);"
									type="button"
									on:click={() => showPassphrase = !showPassphrase}
								>
									{showPassphrase ? 'Hide' : 'Show'}
								</button>
							</div>
						</div>

						<!-- Strength meter -->
						<div>
							<div class="flex items-center justify-between mb-1">
								<span class="text-xs" style="color: var(--oo-fg-muted);">Strength</span>
								<span class="text-xs font-medium" style="color: {strength.color};">{strength.label}</span>
							</div>
							<div class="w-full h-1.5 rounded-full" style="background-color: var(--oo-bg-tertiary);">
								<div
									class="h-1.5 rounded-full transition-all duration-300"
									style="width: {strength.percent}%; background-color: {strength.color};"
								></div>
							</div>
							{#if strength.feedback}
								<p class="text-xs mt-1" style="color: var(--oo-fg-muted);">{strength.feedback}</p>
							{/if}
						</div>

						<div class="flex gap-2">
							<button
								class="px-3 py-1.5 rounded text-xs"
								style="color: var(--oo-fg-muted);"
								on:click={resetWizard}
							>
								Back
							</button>
							<button
								class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
								style="background-color: {canProceedPassphrase ? 'var(--oo-tobacco)' : 'var(--oo-bg-tertiary)'}; color: {canProceedPassphrase ? 'var(--oo-fg-on-accent)' : 'var(--oo-fg-muted)'};"
								disabled={!canProceedPassphrase}
								on:click={goToConfirm}
							>
								Continue
							</button>
						</div>
					</div>

				{:else if wizardStep === 'confirm'}
					<div class="space-y-3">
						<div>
							<label class="block text-xs mb-1" style="color: var(--oo-fg-muted);" for="ceremony-confirm">
								Confirm your passphrase
							</label>
							<input
								id="ceremony-confirm"
								type="password"
								bind:value={passphraseConfirm}
								placeholder="Re-type your passphrase"
								class="w-full px-3 py-2 rounded text-sm font-mono"
								style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-subtle);"
							/>
							{#if passphraseConfirm.length > 0 && !passphraseMatch}
								<p class="text-xs mt-1" style="color: var(--oo-fg-error);">Passphrases do not match</p>
							{/if}
						</div>

						{#if setupError}
							<p class="text-xs" style="color: var(--oo-fg-error);">{setupError}</p>
						{/if}

						<div class="flex gap-2">
							<button
								class="px-3 py-1.5 rounded text-xs"
								style="color: var(--oo-fg-muted);"
								on:click={() => { wizardStep = 'passphrase'; passphraseConfirm = ''; }}
							>
								Back
							</button>
							<button
								class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
								style="background-color: {passphraseMatch ? 'var(--oo-tobacco)' : 'var(--oo-bg-tertiary)'}; color: {passphraseMatch ? 'var(--oo-fg-on-accent)' : 'var(--oo-fg-muted)'};"
								disabled={!passphraseMatch || setupInProgress}
								on:click={confirmSetup}
							>
								{setupInProgress ? 'Configuring...' : 'Confirm & Encrypt'}
							</button>
						</div>
					</div>

				{:else if wizardStep === 'success'}
					<div class="space-y-3">
						<div class="flex items-center gap-2">
							<span style="color: var(--oo-sage);">&#10003;</span>
							<span class="text-sm font-medium" style="color: var(--oo-sage);">Encryption Active</span>
						</div>
						<p class="text-xs" style="color: var(--oo-fg-secondary);">{successMessage}</p>
						<button
							class="px-3 py-1.5 rounded text-xs"
							style="color: var(--oo-fg-muted); background-color: var(--oo-bg-tertiary);"
							on:click={() => loadStatus()}
						>
							Refresh Status
						</button>
					</div>
				{/if}

				{#if setupError && wizardStep === 'choose'}
					<p class="text-xs mt-2" style="color: var(--oo-fg-error);">{setupError}</p>
				{/if}
			</div>
		{/if}

		<!-- PQC Backup Signatures -->
		{#if pqcStatus}
			<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
				<h4 class="text-sm font-semibold mb-3" style="color: var(--oo-fg-primary);">Post-Quantum Backup Signatures</h4>

				<div class="grid grid-cols-2 gap-2 text-xs mb-3">
					<div style="color: var(--oo-fg-muted);">Library (liboqs)</div>
					<div style="color: {pqcStatus.available ? 'var(--oo-sage)' : 'var(--oo-fg-muted)'};">
						{pqcStatus.available ? 'Installed' : 'Not installed'}
					</div>

					<div style="color: var(--oo-fg-muted);">Algorithm</div>
					<div class="font-mono" style="color: var(--oo-fg-secondary);">{pqcStatus.algorithm}</div>

					<div style="color: var(--oo-fg-muted);">Config Enabled</div>
					<div style="color: {pqcStatus.config_enabled ? 'var(--oo-sage)' : 'var(--oo-fg-muted)'};">
						{pqcStatus.config_enabled ? 'Yes' : 'No'}
					</div>

					<div style="color: var(--oo-fg-muted);">Signing Active</div>
					<div style="color: {pqcStatus.effective_enabled ? 'var(--oo-sage)' : 'var(--oo-fg-muted)'};">
						{pqcStatus.effective_enabled ? 'Yes' : 'No'}
					</div>

					<div style="color: var(--oo-fg-muted);">Keypair</div>
					<div style="color: {pqcStatus.keypair_exists ? 'var(--oo-sage)' : 'var(--oo-fg-muted)'};">
						{pqcStatus.keypair_exists ? 'Available' : 'Not generated'}
						{#if pqcStatus.public_key_size}
							<span class="font-mono ml-1" style="color: var(--oo-fg-faint);">
								({pqcStatus.public_key_size}B)
							</span>
						{/if}
					</div>
				</div>

				{#if pqcStatus.available}
					<div class="flex gap-2">
						{#if !pqcStatus.keypair_exists}
							<button
								class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
								style="background-color: var(--oo-tobacco); color: var(--oo-fg-on-accent);"
								disabled={pqcGenerating}
								on:click={handleGeneratePqcKeys}
							>
								{pqcGenerating ? 'Generating...' : 'Generate PQC Keypair'}
							</button>
						{:else}
							<button
								class="px-3 py-1.5 rounded text-xs transition-colors"
								style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
								on:click={handleGeneratePqcKeys}
								disabled={pqcGenerating}
							>
								{pqcGenerating ? 'Regenerating...' : 'Regenerate Keys'}
							</button>
							<button
								class="px-3 py-1.5 rounded text-xs transition-colors"
								style="color: var(--oo-fg-error);"
								on:click={handleDeletePqcKeys}
							>
								Delete Keys
							</button>
						{/if}
					</div>
				{:else}
					<p class="text-xs" style="color: var(--oo-fg-muted);">
						Install liboqs-python to enable post-quantum signatures:
						<code class="font-mono px-1 py-0.5 rounded" style="background-color: var(--oo-bg-tertiary);">
							pip install liboqs-python
						</code>
					</p>
				{/if}

				{#if pqcError}
					<p class="text-xs mt-2" style="color: var(--oo-fg-error);">{pqcError}</p>
				{/if}
			</div>
		{/if}

	{/if}

</div>
