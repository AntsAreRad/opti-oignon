<!--
  RecoveryCodesPanel.svelte (S127)
  Recovery codes management in the security settings.

  Features:
  - Generate recovery codes (shown ONCE)
  - Copy all codes to clipboard
  - Print-friendly display
  - Warning when remaining codes < 3
  - Remaining count indicator
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { apiGet, apiPost } from '$lib/api/client';

	let remaining = 0;
	let loading = true;
	let error = '';

	// Generated codes (shown once)
	let generatedCodes: string[] = [];
	let showCodes = false;
	let generating = false;
	let copied = false;
	let confirmGenerate = false;

	onMount(async () => {
		await loadStatus();
		loading = false;
	});

	async function loadStatus() {
		try {
			const data = await apiGet<Record<string, any>>('/api/security/2fa/status');
			remaining = data.recovery_codes_remaining ?? 0;
		} catch (e: any) {
			error = e.detail || 'Failed to load recovery code status';
		}
	}

	async function generateCodes() {
		generating = true;
		error = '';

		try {
			const result = await apiPost<{ codes: string[] }>(
				'/api/security/2fa/recovery-codes/generate'
			);
			generatedCodes = result.codes || [];
			showCodes = true;
			confirmGenerate = false;
			await loadStatus();
		} catch (e: any) {
			error = e.detail || 'Failed to generate recovery codes.';
		} finally {
			generating = false;
		}
	}

	async function copyAllCodes() {
		try {
			const text = generatedCodes.join('\n');
			await navigator.clipboard.writeText(text);
			copied = true;
			setTimeout(() => { copied = false; }, 2000);
		} catch {
			// Fallback: select a textarea
			error = 'Clipboard access denied. Please copy manually.';
		}
	}

	function printCodes() {
		const content = generatedCodes.map((c, i) => `${i + 1}. ${c}`).join('\n');
		const printWindow = window.open('', '_blank');
		if (printWindow) {
			printWindow.document.write(`
				<html><head><title>Opti-Oignon Recovery Codes</title>
				<style>body{font-family:monospace;padding:2em;} h2{margin-bottom:0.5em;}
				.codes{font-size:1.2em;line-height:2;} .warning{color:darkred;margin-top:1em;font-size:0.9em;}</style>
				</head><body>
				<h2>Opti-Oignon Recovery Codes</h2>
				<pre class="codes">${content}</pre>
				<p class="warning">Store these codes in a safe place. Each code can only be used once.</p>
				</body></html>
			`);
			printWindow.document.close();
			printWindow.print();
		}
	}

	function dismissCodes() {
		generatedCodes = [];
		showCodes = false;
	}
</script>

<div class="space-y-4">
	<div class="flex items-center justify-between">
		<h4 class="text-sm font-semibold" style="color: var(--oo-fg-primary);">
			Recovery Codes
		</h4>
		{#if !loading && remaining > 0}
			<span
				class="px-2 py-0.5 rounded text-xs font-mono"
				style="color: {remaining < 3 ? 'var(--oo-fg-error)' : 'var(--oo-fg-muted)'}; background-color: var(--oo-bg-tertiary);"
			>
				{remaining} remaining
			</span>
		{/if}
	</div>

	{#if loading}
		<p class="text-sm" style="color: var(--oo-fg-muted);">Loading...</p>
	{:else if error}
		<p class="text-sm" style="color: var(--oo-fg-error);">{error}</p>
	{/if}

	<!-- Warning when low -->
	{#if !loading && remaining > 0 && remaining < 3}
		<div class="rounded p-3 flex items-start gap-2"
			style="background-color: var(--oo-bg-warning, rgba(217,119,6,0.1)); border: 1px solid var(--oo-fg-warning);">
			<svg class="w-4 h-4 shrink-0 mt-0.5" style="color: var(--oo-fg-warning);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
				<path stroke-linecap="round" stroke-linejoin="round"
					d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
				/>
			</svg>
			<p class="text-xs" style="color: var(--oo-fg-warning);">
				You have only {remaining} recovery code{remaining === 1 ? '' : 's'} left. Generate new codes to avoid losing access.
			</p>
		</div>
	{/if}

	<!-- Generated codes display (shown once) -->
	{#if showCodes && generatedCodes.length > 0}
		<div class="rounded-lg p-4 space-y-3"
			style="background-color: var(--oo-bg-tertiary); border: 2px solid var(--oo-fg-warning);">
			<div class="flex items-start gap-2">
				<svg class="w-5 h-5 shrink-0" style="color: var(--oo-fg-warning);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
					<path stroke-linecap="round" stroke-linejoin="round"
						d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
					/>
				</svg>
				<p class="text-sm font-medium" style="color: var(--oo-fg-warning);">
					Save these codes now. They will not be shown again.
				</p>
			</div>

			<div class="grid grid-cols-2 gap-1 p-3 rounded font-mono text-sm"
				style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
				{#each generatedCodes as code, i}
					<div class="py-0.5" style="color: var(--oo-fg-primary);">
						<span class="text-xs" style="color: var(--oo-fg-muted);">{i + 1}.</span>
						{code}
					</div>
				{/each}
			</div>

			<div class="flex gap-2 flex-wrap">
				<button
					class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
					style="background-color: var(--oo-tobacco); color: var(--oo-fg-on-accent);"
					on:click={copyAllCodes}
				>
					{copied ? 'Copied!' : 'Copy All'}
				</button>
				<button
					class="px-3 py-1.5 rounded text-xs transition-colors"
					style="color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-subtle);"
					on:click={printCodes}
				>
					Print
				</button>
				<button
					class="px-3 py-1.5 rounded text-xs transition-colors"
					style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
					on:click={dismissCodes}
				>
					I have saved my codes
				</button>
			</div>
		</div>
	{/if}

	<!-- Generate / regenerate button -->
	{#if !showCodes && !loading}
		<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
			<p class="text-sm mb-3" style="color: var(--oo-fg-secondary);">
				{#if remaining === 0}
					No recovery codes configured. Generate codes as a backup method in case you lose access to your authenticator.
				{:else}
					You have {remaining} recovery codes remaining. Generating new codes will replace all existing ones.
				{/if}
			</p>

			{#if confirmGenerate}
				<div class="flex items-center gap-2">
					{#if remaining > 0}
						<span class="text-xs" style="color: var(--oo-fg-warning);">
							This will replace your existing {remaining} codes. Continue?
						</span>
					{/if}
					<button
						class="px-3 py-1.5 rounded text-xs font-medium"
						style="background-color: var(--oo-tobacco); color: var(--oo-fg-on-accent);"
						disabled={generating}
						on:click={generateCodes}
					>
						{generating ? 'Generating...' : 'Yes, Generate'}
					</button>
					<button
						class="px-2 py-1.5 rounded text-xs"
						style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
						on:click={() => { confirmGenerate = false; }}
					>
						Cancel
					</button>
				</div>
			{:else}
				<button
					class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
					style="background-color: var(--oo-tobacco); color: var(--oo-fg-on-accent);"
					on:click={() => { confirmGenerate = remaining > 0; if (!confirmGenerate) generateCodes(); }}
				>
					{remaining > 0 ? 'Regenerate Codes' : 'Generate Recovery Codes'}
				</button>
			{/if}
		</div>
	{/if}
</div>
