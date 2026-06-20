<!--
  BackendStatus.svelte (S106, merged S167)
  Single consolidated health indicator for the header status cluster.
  Shows one colored dot + label; the popover combines backend health
  (status, version, modules) with Ollama connectivity (online / latency /
  queue), folding in the former inline NetworkIndicator (S167 merge).
  Uses --oo-* CSS variables exclusively.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		backendStatus,
		backendVersion,
		backendError,
		backendModules,
		startHealthPolling,
		stopHealthPolling,
		checkHealthNow,
	} from '$lib/stores/health';

	let expanded = false;
	let moduleCount = 0;
	let moduleTotal = 0;

	// Ollama network status (merged from NetworkIndicator, S167).
	let netAvailable = false;
	let netOnline = false;
	let netLatencyMs = 0;
	let netQueueSize = 0;
	let netError = '';
	let netPoll: ReturnType<typeof setInterval> | null = null;
	const NET_POLL_MS = 15000;

	$: {
		const mods = $backendModules;
		const keys = Object.keys(mods);
		moduleTotal = keys.length;
		moduleCount = Object.values(mods).filter(Boolean).length;
	}

	$: statusLabel =
		$backendStatus === 'connected'
			? 'Connected'
			: $backendStatus === 'degraded'
				? 'Degraded'
				: 'Disconnected';

	// Ollama-only label for the popover row.
	$: ollamaLabel = !netAvailable
		? 'N/A'
		: !netOnline
			? 'Offline'
			: netLatencyMs > 3000
				? 'Slow'
				: 'Online';

	// Combined dot color: backend health first, then Ollama connectivity.
	$: dotColor = (() => {
		if ($backendStatus === 'disconnected') return 'var(--oo-error)';
		if (netAvailable && !netOnline) return 'var(--oo-error)';
		if ($backendStatus === 'degraded' || (netAvailable && netLatencyMs > 3000))
			return 'var(--oo-warning)';
		return 'var(--oo-success)';
	})();

	function toggleExpanded() {
		expanded = !expanded;
	}

	function handleClickOutside(event: MouseEvent) {
		const target = event.target as HTMLElement;
		if (expanded && !target.closest('.backend-status-wrapper')) {
			expanded = false;
		}
	}

	async function fetchNetwork() {
		try {
			const resp = await fetch('/api/network/status');
			if (resp.ok) {
				const data = await resp.json();
				netAvailable = data.available ?? false;
				netOnline = data.online ?? false;
				netLatencyMs = data.latency_ms ?? 0;
				netQueueSize = data.queue_size ?? 0;
				netError = data.last_error ?? '';
			}
		} catch {
			netAvailable = false;
			netOnline = false;
		}
	}

	onMount(() => {
		startHealthPolling();
		fetchNetwork();
		netPoll = setInterval(fetchNetwork, NET_POLL_MS);
		document.addEventListener('click', handleClickOutside, true);
	});

	onDestroy(() => {
		stopHealthPolling();
		if (netPoll) clearInterval(netPoll);
		document.removeEventListener('click', handleClickOutside, true);
	});

	function refreshAll() {
		checkHealthNow();
		fetchNetwork();
	}
</script>

<div class="backend-status-wrapper relative">
	<button
		class="status-btn"
		on:click={toggleExpanded}
		title="{statusLabel}{$backendVersion ? ` v${$backendVersion}` : ''} | Ollama: {ollamaLabel}"
		aria-label="Backend status: {statusLabel}, Ollama: {ollamaLabel}"
	>
		<!-- Animated dot -->
		<span class="status-dot" style="background-color: {dotColor};">
			{#if $backendStatus === 'connected'}
				<span class="dot-pulse" style="background-color: {dotColor};" />
			{/if}
		</span>
		<span class="status-label">{statusLabel}</span>
	</button>

	<!-- Expanded popup -->
	{#if expanded}
		<div class="status-popup">
			<div class="popup-header">
				<span class="popup-dot" style="background-color: {dotColor};" />
				<span class="popup-title">{statusLabel}</span>
				{#if $backendVersion}
					<span class="popup-version">v{$backendVersion}</span>
				{/if}
			</div>

			{#if $backendError}
				<div class="popup-error">{$backendError}</div>
			{/if}

			{#if moduleTotal > 0}
				<div class="popup-modules">
					{moduleCount}/{moduleTotal} modules active
				</div>
			{/if}

			<!-- Ollama connectivity (merged from NetworkIndicator) -->
			<div class="popup-net">
				<span class="popup-net-dot" style="background-color: {dotColor};" />
				<span class="popup-net-label">Ollama</span>
				<span class="popup-net-value">{ollamaLabel}</span>
				{#if netAvailable && netOnline}
					<span class="popup-net-latency">{Math.round(netLatencyMs)}ms</span>
				{/if}
			</div>
			{#if netQueueSize > 0}
				<div class="popup-modules">{netQueueSize} queued</div>
			{/if}
			{#if netError}
				<div class="popup-error">{netError}</div>
			{/if}

			<button class="popup-retry" on:click={refreshAll}>
				Refresh
			</button>
		</div>
	{/if}
</div>

<style>
	.backend-status-wrapper {
		display: inline-flex;
		align-items: center;
	}

	.status-btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 4px 8px;
		border-radius: 6px;
		border: none;
		background: transparent;
		cursor: pointer;
		transition: background-color 0.15s ease;
		color: var(--oo-fg-tertiary);
		font-size: 0.75rem;
	}

	.status-btn:hover {
		background-color: var(--oo-bg-elevated);
	}

	.status-dot {
		position: relative;
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.dot-pulse {
		position: absolute;
		inset: -2px;
		border-radius: 50%;
		opacity: 0;
		animation: pulse 2.5s ease-in-out infinite;
	}

	@keyframes pulse {
		0%, 100% { opacity: 0; transform: scale(1); }
		50% { opacity: 0.3; transform: scale(1.6); }
	}

	.status-label {
		display: none;
	}

	/* Show label on wider screens */
	@media (min-width: 640px) {
		.status-label {
			display: inline;
		}
	}

	.status-popup {
		position: absolute;
		top: calc(100% + 6px);
		right: 0;
		z-index: 50;
		min-width: 200px;
		padding: 12px;
		border-radius: 8px;
		background-color: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-subtle);
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.popup-header {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.popup-dot {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.popup-title {
		font-weight: 600;
		font-size: 0.85rem;
		color: var(--oo-fg-primary);
	}

	.popup-version {
		margin-left: auto;
		font-size: 0.75rem;
		color: var(--oo-fg-muted);
		font-family: monospace;
	}

	.popup-error {
		font-size: 0.75rem;
		color: var(--oo-error);
		padding: 4px 8px;
		border-radius: 4px;
		background-color: var(--oo-error-bg);
	}

	.popup-modules {
		font-size: 0.75rem;
		color: var(--oo-fg-muted);
	}

	.popup-net {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 0.75rem;
		color: var(--oo-fg-secondary);
		padding-top: 8px;
		border-top: 1px solid var(--oo-bd-subtle);
	}

	.popup-net-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.popup-net-label {
		font-weight: 500;
	}

	.popup-net-value {
		margin-left: auto;
		color: var(--oo-fg-muted);
	}

	.popup-net-latency {
		color: var(--oo-fg-muted);
		font-family: monospace;
	}

	.popup-retry {
		align-self: flex-start;
		padding: 4px 12px;
		border-radius: 4px;
		border: 1px solid var(--oo-bd-subtle);
		background: transparent;
		color: var(--oo-fg-secondary);
		font-size: 0.75rem;
		cursor: pointer;
		transition: background-color 0.15s ease;
	}

	.popup-retry:hover {
		background-color: var(--oo-bg-overlay);
	}
</style>
