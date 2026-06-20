<!--
  LiveMetricsOverlay.svelte -- S111
  Floating mini-dashboard showing real-time performance metrics
  during active inference: tok/s, latency, GPU %, memory usage.
  Auto-hides when no inference is active. Compact bar chart shows
  tok/s over the last 30 seconds.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { isStreaming } from '$lib/stores/chat';

	/** Whether the overlay is visible (auto-controlled). */
	let visible = false;

	/** Current metrics snapshot. */
	let metrics: Record<string, any> = {};

	/** tok/s sparkline data (last 30 seconds). */
	let sparkline: number[] = [];

	/** Maximum tok/s seen (for scaling the sparkline). */
	let maxTokS = 1;

	/** Polling interval handle. */
	let pollTimer: ReturnType<typeof setInterval> | null = null;

	/** Fade-out delay handle. */
	let hideTimer: ReturnType<typeof setTimeout> | null = null;

	const POLL_MS = 600;
	const HIDE_DELAY_MS = 3000;
	const SPARKLINE_MAX = 60; // ~30s at 500ms sampling

	/** Whether GPU data is available. */
	$: hasGpu = metrics.gpu_utilization_pct >= 0;

	/** Formatted tok/s. */
	$: tokS = metrics.tokens_per_second != null
		? metrics.tokens_per_second.toFixed(1)
		: '0.0';

	/** Formatted GPU %. */
	$: gpuPct = hasGpu
		? Math.round(metrics.gpu_utilization_pct)
		: null;

	/** Formatted memory. */
	$: memUsed = metrics.system_memory_used_mb > 0
		? (metrics.system_memory_used_mb / 1024).toFixed(1)
		: null;

	$: memTotal = metrics.system_memory_total_mb > 0
		? (metrics.system_memory_total_mb / 1024).toFixed(1)
		: null;

	/** GPU memory. */
	$: gpuMemUsed = metrics.gpu_memory_used_mb > 0
		? (metrics.gpu_memory_used_mb / 1024).toFixed(1)
		: null;

	$: gpuMemTotal = metrics.gpu_memory_total_mb > 0
		? (metrics.gpu_memory_total_mb / 1024).toFixed(1)
		: null;

	/** GPU temp. */
	$: gpuTemp = metrics.gpu_temperature_c > 0
		? Math.round(metrics.gpu_temperature_c)
		: null;

	async function fetchMetrics() {
		try {
			const res = await fetch('/api/metrics/live');
			if (!res.ok) return;
			metrics = await res.json();

			// Update sparkline.
			const val = metrics.tokens_per_second ?? 0;
			sparkline = [...sparkline.slice(-(SPARKLINE_MAX - 1)), val];
			maxTokS = Math.max(1, ...sparkline);
		} catch {
			// Silently ignore fetch errors.
		}
	}

	function startPolling() {
		if (pollTimer) return;
		fetchMetrics();
		pollTimer = setInterval(fetchMetrics, POLL_MS);
	}

	function stopPolling() {
		if (pollTimer) {
			clearInterval(pollTimer);
			pollTimer = null;
		}
	}

	// React to streaming state changes.
	const unsubStreaming = isStreaming.subscribe((streaming) => {
		if (streaming) {
			// Show overlay and start polling.
			visible = true;
			sparkline = [];
			if (hideTimer) {
				clearTimeout(hideTimer);
				hideTimer = null;
			}
			startPolling();
		} else {
			// Fade out after a delay so user can see final metrics.
			hideTimer = setTimeout(() => {
				visible = false;
				stopPolling();
			}, HIDE_DELAY_MS);
		}
	});

	onMount(() => {
		// If already streaming on mount, start.
		if ($isStreaming) {
			visible = true;
			startPolling();
		}
	});

	onDestroy(() => {
		unsubStreaming();
		stopPolling();
		if (hideTimer) clearTimeout(hideTimer);
	});
</script>

{#if visible}
	<div class="live-metrics-overlay" class:fading={!$isStreaming}>
		<!-- Header -->
		<div class="overlay-header">
			<span class="overlay-dot" class:active={$isStreaming}></span>
			<span class="overlay-title">
				{$isStreaming ? 'Generating...' : 'Done'}
			</span>
			{#if metrics.active_model}
				<span class="overlay-model">{metrics.active_model}</span>
			{/if}
		</div>

		<!-- Main stat: tok/s -->
		<div class="stat-main">
			<span class="stat-value">{tokS}</span>
			<span class="stat-unit">tok/s</span>
		</div>

		<!-- Sparkline chart -->
		{#if sparkline.length > 1}
			<div class="sparkline-container">
				<svg viewBox="0 0 {sparkline.length} 20" preserveAspectRatio="none"
					class="sparkline-svg">
					{#each sparkline as val, i}
						<rect
							x={i}
							y={20 - (val / maxTokS) * 18}
							width="0.8"
							height={(val / maxTokS) * 18}
							class="sparkline-bar"
						/>
					{/each}
				</svg>
			</div>
		{/if}

		<!-- Secondary stats -->
		<div class="stat-grid">
			{#if metrics.eval_time_ms > 0}
				<div class="stat-item">
					<span class="stat-label">Latency</span>
					<span class="stat-val">{Math.round(metrics.eval_time_ms)}ms</span>
				</div>
			{/if}

			{#if gpuPct !== null}
				<div class="stat-item">
					<span class="stat-label">GPU</span>
					<span class="stat-val">{gpuPct}%</span>
				</div>
			{/if}

			{#if gpuTemp !== null}
				<div class="stat-item">
					<span class="stat-label">Temp</span>
					<span class="stat-val">{gpuTemp}&deg;C</span>
				</div>
			{/if}

			{#if gpuMemUsed !== null && gpuMemTotal !== null}
				<div class="stat-item">
					<span class="stat-label">VRAM</span>
					<span class="stat-val">{gpuMemUsed}/{gpuMemTotal}G</span>
				</div>
			{/if}

			{#if memUsed !== null && memTotal !== null}
				<div class="stat-item">
					<span class="stat-label">RAM</span>
					<span class="stat-val">{memUsed}/{memTotal}G</span>
				</div>
			{/if}

			{#if metrics.total_tokens > 0}
				<div class="stat-item">
					<span class="stat-label">Tokens</span>
					<span class="stat-val">{metrics.total_tokens}</span>
				</div>
			{/if}
		</div>
	</div>
{/if}

<style>
	.live-metrics-overlay {
		position: absolute;
		bottom: 5rem;
		right: 1.25rem;
		z-index: var(--oo-z-overlay);
		min-width: 11rem;
		max-width: 14rem;
		padding: 0.625rem 0.75rem;
		border-radius: 0.75rem;
		background-color: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-subtle);
		box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
		font-size: 0.75rem;
		line-height: 1.4;
		color: var(--oo-fg-primary);
		opacity: 1;
		transition: opacity 0.5s ease-out;
		pointer-events: auto;
	}

	.live-metrics-overlay.fading {
		opacity: 0.7;
	}

	.overlay-header {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		margin-bottom: 0.375rem;
	}

	.overlay-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background-color: var(--oo-fg-muted);
		flex-shrink: 0;
	}

	.overlay-dot.active {
		background-color: var(--oo-success);
		animation: pulse-dot 1.5s ease-in-out infinite;
	}

	@keyframes pulse-dot {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.4; }
	}

	.overlay-title {
		font-weight: 600;
		color: var(--oo-fg-secondary);
		font-size: 0.6875rem;
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}

	.overlay-model {
		margin-left: auto;
		color: var(--oo-fg-muted);
		font-size: 0.625rem;
		max-width: 5rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.stat-main {
		display: flex;
		align-items: baseline;
		gap: 0.25rem;
		margin-bottom: 0.25rem;
	}

	.stat-value {
		font-size: 1.375rem;
		font-weight: 700;
		color: var(--oo-acc-400);
		line-height: 1;
	}

	.stat-unit {
		font-size: 0.625rem;
		color: var(--oo-fg-muted);
		font-weight: 500;
	}

	.sparkline-container {
		height: 1.25rem;
		margin-bottom: 0.375rem;
		overflow: hidden;
		border-radius: 0.25rem;
	}

	.sparkline-svg {
		width: 100%;
		height: 100%;
		display: block;
	}

	.sparkline-bar {
		fill: var(--oo-acc-400);
		opacity: 0.6;
	}

	.stat-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.125rem 0.5rem;
	}

	.stat-item {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 0.25rem;
	}

	.stat-label {
		color: var(--oo-fg-muted);
		font-size: 0.625rem;
	}

	.stat-val {
		color: var(--oo-fg-secondary);
		font-weight: 600;
		font-size: 0.6875rem;
		font-variant-numeric: tabular-nums;
	}
</style>
