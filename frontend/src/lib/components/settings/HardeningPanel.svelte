<!--
  HardeningPanel.svelte (S131)
  System hardening status and controls.

  Sections:
    - Conversation RAM Wipe: manual wipe, auto-wipe toggle status
    - Ollama Log Status: current level, recommendations, verbose warning
    - Swap/Hibernation: encrypted badge or warning
    - Network Hardening: DNS, proxy, ports checklist
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getHardeningStatus,
		wipeAllConversations,
		type HardeningStatus,
		type WipeAllResult,
	} from '$lib/api/hardening';

	let status: HardeningStatus | null = null;
	let loading = true;
	let error = '';

	// Wipe state
	let wiping = false;
	let wipeResult: WipeAllResult | null = null;
	let wipeError = '';

	onMount(async () => {
		await loadStatus();
		loading = false;
	});

	async function loadStatus() {
		try {
			status = await getHardeningStatus();
			error = '';
		} catch (e: any) {
			error = e?.message || 'Failed to load hardening status';
		}
	}

	async function handleWipeAll() {
		if (!confirm('Wipe ALL conversation buffers from RAM? This cannot be undone.')) return;
		wiping = true;
		wipeError = '';
		wipeResult = null;
		try {
			wipeResult = await wipeAllConversations();
			await loadStatus();
		} catch (e: any) {
			wipeError = e?.message || 'Wipe failed';
		} finally {
			wiping = false;
		}
	}
</script>

<div class="space-y-4">
	{#if loading}
		<p class="text-sm" style="color: var(--oo-fg-muted);">Loading hardening status...</p>
	{:else if error}
		<p class="text-sm" style="color: var(--oo-fg-error);">{error}</p>
	{:else if status}

		<!-- Conversation RAM Wipe -->
		<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
			<div class="flex items-center justify-between mb-3">
				<div class="flex items-center gap-2">
					<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" style="color: var(--oo-tobacco);">
						<path stroke-linecap="round" stroke-linejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
					</svg>
					<h4 class="text-sm font-semibold" style="color: var(--oo-fg-primary);">Conversation RAM Wipe</h4>
				</div>
				{#if status.conversation_wipe.available}
					<span class="px-2 py-0.5 rounded text-xs" style="background-color: var(--oo-sage-bg, rgba(120,150,120,0.15)); color: var(--oo-sage);">Active</span>
				{:else}
					<span class="px-2 py-0.5 rounded text-xs" style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-muted);">Unavailable</span>
				{/if}
			</div>

			{#if status.conversation_wipe.available}
				<div class="space-y-2 text-xs" style="color: var(--oo-fg-secondary);">
					<div class="flex justify-between">
						<span>Auto-wipe on close</span>
						<span style="color: {status.conversation_wipe.auto_wipe_on_close ? 'var(--oo-sage)' : 'var(--oo-fg-muted)'};">
							{status.conversation_wipe.auto_wipe_on_close ? 'Enabled' : 'Disabled'}
						</span>
					</div>
					<div class="flex justify-between">
						<span>Bulbe per-turn wipe</span>
						<span style="color: {status.conversation_wipe.bulbe_wipe_per_turn ? 'var(--oo-sage)' : 'var(--oo-fg-muted)'};">
							{status.conversation_wipe.bulbe_wipe_per_turn ? 'Enabled' : 'Disabled'}
						</span>
					</div>
					<div class="flex justify-between">
						<span>Active conversations</span>
						<span class="font-mono">{status.conversation_wipe.active_conversations}</span>
					</div>
					<div class="flex justify-between">
						<span>Registered buffers</span>
						<span class="font-mono">{status.conversation_wipe.total_registered_buffers}</span>
					</div>
					<div class="flex justify-between">
						<span>memset() available</span>
						<span style="color: {status.conversation_wipe.memset_available ? 'var(--oo-sage)' : 'var(--oo-fg-warning)'};">
							{status.conversation_wipe.memset_available ? 'Yes' : 'No (best-effort only)'}
						</span>
					</div>
				</div>

				<div class="mt-3 flex items-center gap-2">
					<button
						class="px-3 py-1 rounded text-xs font-medium transition-colors"
						style="background-color: var(--oo-fg-error); color: white;"
						on:click={handleWipeAll}
						disabled={wiping}
					>
						{wiping ? 'Wiping...' : 'Wipe All Conversations'}
					</button>
					{#if wipeResult}
						<span class="text-xs" style="color: var(--oo-sage);">
							Wiped {wipeResult.conversations_wiped} conversations ({wipeResult.total_fields_zeroed} fields zeroed)
						</span>
					{/if}
					{#if wipeError}
						<span class="text-xs" style="color: var(--oo-fg-error);">{wipeError}</span>
					{/if}
				</div>

				<p class="mt-2 text-xs" style="color: var(--oo-fg-faint);">
					Note: RAM wipe is best-effort. Python GC may retain copies of short/interned strings.
				</p>
			{/if}
		</div>

		<!-- Ollama Log Status -->
		<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
			<div class="flex items-center justify-between mb-3">
				<div class="flex items-center gap-2">
					<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" style="color: var(--oo-tobacco);">
						<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
					</svg>
					<h4 class="text-sm font-semibold" style="color: var(--oo-fg-primary);">Ollama Logging</h4>
				</div>
				{#if status.ollama_log.available}
					{#if status.ollama_log.log_level && (status.ollama_log.log_level.includes('debug') || status.ollama_log.log_level.includes('trace'))}
						<span class="px-2 py-0.5 rounded text-xs" style="background-color: rgba(217,119,6,0.15); color: var(--oo-fg-warning);">Verbose</span>
					{:else}
						<span class="px-2 py-0.5 rounded text-xs" style="background-color: var(--oo-sage-bg, rgba(120,150,120,0.15)); color: var(--oo-sage);">OK</span>
					{/if}
				{:else}
					<span class="px-2 py-0.5 rounded text-xs" style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-muted);">Unavailable</span>
				{/if}
			</div>

			{#if status.ollama_log.available}
				<div class="space-y-2 text-xs" style="color: var(--oo-fg-secondary);">
					<div class="flex justify-between">
						<span>Current log level</span>
						<span class="font-mono">{status.ollama_log.log_level || 'unknown'}</span>
					</div>
					<div class="flex justify-between">
						<span>Log sanitization</span>
						<span style="color: {status.ollama_log.sanitization_enabled ? 'var(--oo-sage)' : 'var(--oo-fg-muted)'};">
							{status.ollama_log.sanitization_enabled ? 'Enabled' : 'Disabled'}
						</span>
					</div>
				</div>
				{#if status.ollama_log.recommendations && Object.keys(status.ollama_log.recommendations).length > 0}
					<div class="mt-3 rounded p-2" style="background-color: var(--oo-bg-subtle); border: 1px solid var(--oo-bd-subtle);">
						<p class="text-xs font-medium mb-1" style="color: var(--oo-fg-muted);">Recommended environment variables:</p>
						<div class="space-y-0.5">
							{#each Object.entries(status.ollama_log.recommendations) as [key, val]}
								<p class="text-xs font-mono" style="color: var(--oo-fg-secondary);">{key}={val}</p>
							{/each}
						</div>
					</div>
				{/if}
			{/if}
		</div>

		<!-- Swap / Hibernation -->
		<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
			<div class="flex items-center justify-between mb-3">
				<div class="flex items-center gap-2">
					<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" style="color: var(--oo-tobacco);">
						<path stroke-linecap="round" stroke-linejoin="round" d="M5.25 14.25h13.5m-13.5 0a3 3 0 01-3-3m3 3a3 3 0 100 6h13.5a3 3 0 100-6m-16.5-3a3 3 0 013-3h13.5a3 3 0 013 3m-19.5 0a4.5 4.5 0 01.9-2.7L5.737 5.1a3.375 3.375 0 012.7-1.35h7.126c1.062 0 2.062.5 2.7 1.35l2.587 3.45a4.5 4.5 0 01.9 2.7m0 0a3 3 0 01-3 3m0 3h.008v.008h-.008v-.008zm0-6h.008v.008h-.008v-.008z" />
					</svg>
					<h4 class="text-sm font-semibold" style="color: var(--oo-fg-primary);">Swap Protection</h4>
				</div>
				{#if status.swap.available !== false}
					{#if status.swap.safe}
						<span class="px-2 py-0.5 rounded text-xs" style="background-color: var(--oo-sage-bg, rgba(120,150,120,0.15)); color: var(--oo-sage);">
							{status.swap.swap_enabled ? 'Encrypted' : 'No Swap'}
						</span>
					{:else}
						<span class="px-2 py-0.5 rounded text-xs" style="background-color: rgba(220,38,38,0.15); color: var(--oo-fg-error);">Unencrypted</span>
					{/if}
				{:else}
					<span class="px-2 py-0.5 rounded text-xs" style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-muted);">Unavailable</span>
				{/if}
			</div>

			{#if status.swap.available !== false}
				<div class="space-y-2 text-xs" style="color: var(--oo-fg-secondary);">
					<div class="flex justify-between">
						<span>Swap enabled</span>
						<span>{status.swap.swap_enabled ? 'Yes' : 'No'}</span>
					</div>
					{#if status.swap.swap_enabled}
						<div class="flex justify-between">
							<span>All devices encrypted</span>
							<span style="color: {status.swap.encrypted ? 'var(--oo-sage)' : 'var(--oo-fg-error)'};">
								{status.swap.encrypted ? 'Yes' : 'No'}
							</span>
						</div>
						{#if status.swap.devices && status.swap.devices.length > 0}
							<div class="mt-1 rounded p-2" style="background-color: var(--oo-bg-subtle);">
								{#each status.swap.devices as dev}
									<div class="flex justify-between">
										<span class="font-mono">{dev.device}</span>
										<span style="color: {dev.encrypted ? 'var(--oo-sage)' : 'var(--oo-fg-error)'};">
											{dev.encrypted ? 'encrypted' : 'PLAIN'}
										</span>
									</div>
								{/each}
							</div>
						{/if}
					{/if}
					{#if !status.swap.platform_supported}
						<p style="color: var(--oo-fg-faint);">Swap check is Linux-only. Skipped on this platform.</p>
					{/if}
				</div>
				{#if !status.swap.safe && status.swap.swap_enabled}
					<p class="mt-2 text-xs" style="color: var(--oo-fg-error);">
						Unencrypted swap detected. Sensitive data may be written to disk.
						Use encrypted swap, zram, or disable swap. See INSTALL.md.
					</p>
				{/if}
			{/if}
		</div>

		<!-- Network Hardening -->
		<div class="rounded-lg p-4" style="background-color: var(--oo-card-bg); border: 1px solid var(--oo-bd-subtle);">
			<div class="flex items-center gap-2 mb-3">
				<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" style="color: var(--oo-tobacco);">
					<path stroke-linecap="round" stroke-linejoin="round" d="M12 21a9.004 9.004 0 008.716-6.747M12 21a9.004 9.004 0 01-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 017.843 4.582M12 3a8.997 8.997 0 00-7.843 4.582m15.686 0A11.953 11.953 0 0112 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0121 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0112 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 013 12c0-1.605.42-3.113 1.157-4.418" />
				</svg>
				<h4 class="text-sm font-semibold" style="color: var(--oo-fg-primary);">Network Hardening</h4>
			</div>

			{#if status.network.available}
				<div class="space-y-3">
					<!-- DNS -->
					{#if status.network.dns}
						<div class="flex items-center justify-between text-xs">
							<span style="color: var(--oo-fg-secondary);">DNS Encryption</span>
							<div class="flex items-center gap-1.5">
								<span class="font-mono" style="color: var(--oo-fg-muted);">
									{status.network.dns.protocol}
								</span>
								{#if status.network.dns.encrypted}
									<span class="w-2 h-2 rounded-full" style="background-color: var(--oo-sage);"></span>
								{:else}
									<span class="w-2 h-2 rounded-full" style="background-color: var(--oo-fg-warning);"></span>
								{/if}
							</div>
						</div>
					{/if}

					<!-- Proxy -->
					{#if status.network.proxy}
						<div class="flex items-center justify-between text-xs">
							<span style="color: var(--oo-fg-secondary);">SOCKS Proxy</span>
							<div class="flex items-center gap-1.5">
								{#if status.network.proxy.configured}
									<span class="font-mono" style="color: var(--oo-fg-muted);">
										{status.network.proxy.reachable ? 'Connected' : 'Unreachable'}
									</span>
									<span class="w-2 h-2 rounded-full" style="background-color: {status.network.proxy.reachable ? 'var(--oo-sage)' : 'var(--oo-fg-error)'};"></span>
								{:else}
									<span style="color: var(--oo-fg-muted);">Not configured</span>
									<span class="w-2 h-2 rounded-full" style="background-color: var(--oo-fg-muted);"></span>
								{/if}
							</div>
						</div>
					{/if}

					<!-- Ports -->
					{#if status.network.ports}
						<div class="flex items-center justify-between text-xs">
							<span style="color: var(--oo-fg-secondary);">Open Ports</span>
							<div class="flex items-center gap-1.5">
								<span class="font-mono" style="color: var(--oo-fg-muted);">
									{status.network.ports.total} total, {status.network.ports.unexpected} unexpected
								</span>
								{#if status.network.ports.unexpected === 0}
									<span class="w-2 h-2 rounded-full" style="background-color: var(--oo-sage);"></span>
								{:else}
									<span class="w-2 h-2 rounded-full" style="background-color: var(--oo-fg-warning);"></span>
								{/if}
							</div>
						</div>
					{/if}
				</div>

				<!-- Warnings -->
				{#if status.network.warnings && status.network.warnings.length > 0}
					<div class="mt-3 space-y-1">
						{#each status.network.warnings as warning}
							<p class="text-xs" style="color: var(--oo-fg-warning);">{warning}</p>
						{/each}
					</div>
				{/if}
			{:else}
				<p class="text-xs" style="color: var(--oo-fg-muted);">Network hardening checks not available.</p>
			{/if}
		</div>

	{/if}
</div>
