<!--
  EventTimeline.svelte -- Event Timeline Visualization.

  Interactive horizontal timeline showing inference events over time.
  Features:
  1. Events rendered as dots/ticks on a horizontal axis
  2. Color-coded by latency percentile (green < p50, yellow < p95, red > p95)
  3. Hover tooltips with event details (model, latency, tokens)
  4. Zoom controls (1h / 6h / 24h / 7d)
  5. Click-to-inspect: clicking a dot shows full event details
-->
<script lang="ts">
	import { onMount, onDestroy, createEventDispatcher } from 'svelte';
	import { getTelemetryHistory } from '$lib/api/telemetry';
	import type { HistoryEvent } from '$lib/api/telemetry';

	const dispatch = createEventDispatcher<{ selectEvent: HistoryEvent }>();

	// -------------------------------------------------------------------------
	// Props
	// -------------------------------------------------------------------------

	/** Optional model filter. */
	export let modelFilter: string = '';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	let events: HistoryEvent[] = [];
	let loading = false;
	let error = '';

	// Zoom
	type ZoomLevel = { label: string; hours: number };
	const zoomLevels: ZoomLevel[] = [
		{ label: '1h', hours: 1 },
		{ label: '6h', hours: 6 },
		{ label: '24h', hours: 24 },
		{ label: '7d', hours: 168 },
	];
	let activeZoom: ZoomLevel = zoomLevels[2]; // default 24h

	// Percentile thresholds (computed from data)
	let p50 = 0;
	let p95 = 0;

	// Tooltip
	let tooltipEvent: HistoryEvent | null = null;
	let tooltipX = 0;
	let tooltipY = 0;
	let tooltipVisible = false;

	// Selected event detail
	let selectedEvent: HistoryEvent | null = null;

	// Timeline container ref
	let timelineEl: HTMLDivElement;

	// Auto-refresh
	let refreshTimer: ReturnType<typeof setInterval> | null = null;

	// -------------------------------------------------------------------------
	// Lifecycle
	// -------------------------------------------------------------------------

	onMount(() => {
		loadEvents();
		refreshTimer = setInterval(loadEvents, 15_000);
	});

	onDestroy(() => {
		if (refreshTimer) clearInterval(refreshTimer);
	});

	// -------------------------------------------------------------------------
	// Data
	// -------------------------------------------------------------------------

	async function loadEvents() {
		loading = true;
		error = '';
		try {
			// Fetch up to 500 events within the zoom window
			const cutoffTs = Date.now() / 1000 - activeZoom.hours * 3600;
			const res = await getTelemetryHistory(500, 0, modelFilter);
			// Filter client-side by time range
			events = (res.events || []).filter((e) => e.timestamp >= cutoffTs);
			computePercentiles();
		} catch (e: any) {
			error = e?.message || 'Failed to load timeline events';
			events = [];
		} finally {
			loading = false;
		}
	}

	function computePercentiles() {
		if (events.length === 0) {
			p50 = 0;
			p95 = 0;
			return;
		}
		const sorted = [...events].map((e) => e.latency_ms).sort((a, b) => a - b);
		const idx50 = Math.floor(sorted.length * 0.5);
		const idx95 = Math.floor(sorted.length * 0.95);
		p50 = sorted[idx50] || 0;
		p95 = sorted[Math.min(idx95, sorted.length - 1)] || 0;
	}

	function changeZoom(z: ZoomLevel) {
		activeZoom = z;
		selectedEvent = null;
		tooltipVisible = false;
		loadEvents();
	}

	// -------------------------------------------------------------------------
	// Timeline layout
	// -------------------------------------------------------------------------

	$: timeRange = (() => {
		const now = Date.now() / 1000;
		const start = now - activeZoom.hours * 3600;
		return { start, end: now, span: activeZoom.hours * 3600 };
	})();

	function eventX(ev: HistoryEvent): number {
		const pct = (ev.timestamp - timeRange.start) / timeRange.span;
		return Math.max(0, Math.min(100, pct * 100));
	}

	function eventY(ev: HistoryEvent): number {
		// Vertical position based on latency (lower = faster, higher = slower)
		if (p95 <= 0) return 50;
		const pct = Math.min(ev.latency_ms / (p95 * 1.5), 1);
		return 10 + pct * 75; // 10% to 85% of height
	}

	function dotColor(ev: HistoryEvent): string {
		if (ev.latency_ms <= p50) return 'var(--oo-success)';
		if (ev.latency_ms <= p95) return 'var(--oo-warning)';
		return 'var(--oo-danger)';
	}

	function dotSize(ev: HistoryEvent): number {
		// Scale by token count — more tokens = bigger dot
		const tokens = ev.tokens_in + ev.tokens_out;
		if (tokens < 100) return 5;
		if (tokens < 500) return 7;
		return 9;
	}

	// -------------------------------------------------------------------------
	// Tooltip
	// -------------------------------------------------------------------------

	function handleDotEnter(ev: HistoryEvent, e: MouseEvent) {
		tooltipEvent = ev;
		tooltipVisible = true;
		updateTooltipPos(e);
	}

	function handleDotMove(e: MouseEvent) {
		if (tooltipVisible) updateTooltipPos(e);
	}

	function handleDotLeave() {
		tooltipVisible = false;
		tooltipEvent = null;
	}

	function updateTooltipPos(e: MouseEvent) {
		if (!timelineEl) return;
		const rect = timelineEl.getBoundingClientRect();
		tooltipX = e.clientX - rect.left + 12;
		tooltipY = e.clientY - rect.top - 10;
		// Keep tooltip within bounds
		if (tooltipX + 220 > rect.width) tooltipX = e.clientX - rect.left - 230;
		if (tooltipY < 0) tooltipY = 10;
	}

	// -------------------------------------------------------------------------
	// Selection
	// -------------------------------------------------------------------------

	function handleDotClick(ev: HistoryEvent) {
		selectedEvent = selectedEvent?.id === ev.id ? null : ev;
		dispatch('selectEvent', ev);
	}

	function clearSelection() {
		selectedEvent = null;
	}

	// -------------------------------------------------------------------------
	// Formatting
	// -------------------------------------------------------------------------

	function fmtTs(ts: number): string {
		if (!ts) return '—';
		return new Date(ts * 1000).toLocaleString();
	}

	function fmtShortTime(ts: number): string {
		if (!ts) return '';
		const d = new Date(ts * 1000);
		return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
	}

	function fmtDuration(ms: number): string {
		if (ms < 1000) return `${Math.round(ms)}ms`;
		return `${(ms / 1000).toFixed(1)}s`;
	}

	// Time axis labels
	$: axisLabels = (() => {
		const count = Math.min(6, Math.max(2, Math.floor(activeZoom.hours / 2)));
		const labels: { pct: number; text: string }[] = [];
		for (let i = 0; i <= count; i++) {
			const ts = timeRange.start + (timeRange.span * i) / count;
			const pct = (i / count) * 100;
			labels.push({ pct, text: fmtShortTime(ts) });
		}
		return labels;
	})();
</script>

<div class="event-timeline">
	<!-- Header -->
	<div class="tl-header">
		<h3>Event Timeline</h3>
		<div class="tl-controls">
			{#each zoomLevels as z}
				<button
					class="btn-zoom"
					class:active={activeZoom.label === z.label}
					on:click={() => changeZoom(z)}
				>
					{z.label}
				</button>
			{/each}
			<button class="btn-zoom" on:click={loadEvents} disabled={loading}>
				{loading ? '...' : '↻'}
			</button>
		</div>
	</div>

	<!-- Legend -->
	<div class="tl-legend">
		<span class="legend-item">
			<span class="legend-dot dot-green"></span>
			&lt; p50 ({fmtDuration(p50)})
		</span>
		<span class="legend-item">
			<span class="legend-dot dot-yellow"></span>
			&lt; p95 ({fmtDuration(p95)})
		</span>
		<span class="legend-item">
			<span class="legend-dot dot-red"></span>
			&gt; p95
		</span>
		<span class="legend-count">{events.length} events</span>
	</div>

	{#if error}
		<div class="tl-error">{error}</div>
	{/if}

	{#if events.length === 0 && !loading}
		<div class="tl-empty">No events in the selected time range</div>
	{:else}
		<!-- Timeline canvas -->
		<div class="tl-canvas" bind:this={timelineEl}>
			<!-- Y-axis label -->
			<div class="y-label y-top">Slow</div>
			<div class="y-label y-bottom">Fast</div>

			<!-- Event dots -->
			{#each events as ev (ev.id)}
				<button
					class="event-dot"
					class:selected={selectedEvent?.id === ev.id}
					style="
						left: {eventX(ev)}%;
						top: {eventY(ev)}%;
						width: {dotSize(ev)}px;
						height: {dotSize(ev)}px;
						background: {dotColor(ev)};
					"
					on:mouseenter={(e) => handleDotEnter(ev, e)}
					on:mousemove={handleDotMove}
					on:mouseleave={handleDotLeave}
					on:click={() => handleDotClick(ev)}
					aria-label="Event {ev.request_id}"
				></button>
			{/each}

			<!-- Tooltip -->
			{#if tooltipVisible && tooltipEvent}
				<div
					class="tl-tooltip"
					style="left: {tooltipX}px; top: {tooltipY}px"
				>
					<div class="tt-model">{tooltipEvent.model || 'unknown'}</div>
					<div class="tt-row">
						<span class="tt-label">Time</span>
						<span class="tt-val">{fmtTs(tooltipEvent.timestamp)}</span>
					</div>
					<div class="tt-row">
						<span class="tt-label">Latency</span>
						<span class="tt-val">{fmtDuration(tooltipEvent.latency_ms)}</span>
					</div>
					<div class="tt-row">
						<span class="tt-label">Tokens</span>
						<span class="tt-val">{tooltipEvent.tokens_in}→{tooltipEvent.tokens_out}</span>
					</div>
					<div class="tt-row">
						<span class="tt-label">tok/s</span>
						<span class="tt-val">{tooltipEvent.tok_per_sec.toFixed(1)}</span>
					</div>
				</div>
			{/if}

			<!-- X-axis -->
			<div class="tl-axis">
				{#each axisLabels as lbl}
					<span class="axis-label" style="left: {lbl.pct}%">{lbl.text}</span>
				{/each}
			</div>
		</div>
	{/if}

	<!-- Selected event detail -->
	{#if selectedEvent}
		<div class="detail-card">
			<div class="detail-header">
				<h4>Event Detail</h4>
				<button class="btn-close" on:click={clearSelection} aria-label="Close event detail"><span aria-hidden="true">✕</span></button>
			</div>
			<div class="detail-grid">
				<div class="dg-row">
					<span class="dg-label">Request ID</span>
					<span class="dg-value dg-mono">{selectedEvent.request_id}</span>
				</div>
				<div class="dg-row">
					<span class="dg-label">Model</span>
					<span class="dg-value">{selectedEvent.model}</span>
				</div>
				<div class="dg-row">
					<span class="dg-label">Time</span>
					<span class="dg-value">{fmtTs(selectedEvent.timestamp)}</span>
				</div>
				<div class="dg-row">
					<span class="dg-label">Total Latency</span>
					<span class="dg-value">{fmtDuration(selectedEvent.latency_ms)}</span>
				</div>
				<div class="dg-row">
					<span class="dg-label">Prompt Eval</span>
					<span class="dg-value">{fmtDuration(selectedEvent.prompt_eval_ms)}</span>
				</div>
				<div class="dg-row">
					<span class="dg-label">Token Gen</span>
					<span class="dg-value">{fmtDuration(selectedEvent.token_gen_ms)}</span>
				</div>
				<div class="dg-row">
					<span class="dg-label">Tokens In / Out</span>
					<span class="dg-value">{selectedEvent.tokens_in} / {selectedEvent.tokens_out}</span>
				</div>
				<div class="dg-row">
					<span class="dg-label">Throughput</span>
					<span class="dg-value">{selectedEvent.tok_per_sec.toFixed(1)} tok/s</span>
				</div>
			</div>
		</div>
	{/if}
</div>

<style>
	.event-timeline {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
	}

	/* Header */
	.tl-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 0.5rem;
	}

	.tl-header h3 {
		margin: 0;
		font-size: 0.92rem;
		color: var(--oo-text-primary);
	}

	.tl-controls {
		display: flex;
		gap: 0.25rem;
	}

	.btn-zoom {
		padding: 0.2rem 0.55rem;
		border-radius: 4px;
		font-size: 0.75rem;
		cursor: pointer;
		border: 1px solid var(--oo-bd-subtle);
		background: var(--oo-bg-elevated);
		color: var(--oo-text-secondary);
		transition: background 0.15s;
	}

	.btn-zoom:hover { background: var(--oo-bg-overlay); }
	.btn-zoom:disabled { opacity: 0.4; cursor: not-allowed; }
	.btn-zoom.active {
		background: var(--oo-accent-primary);
		color: var(--oo-fg-on-accent);
		border-color: var(--oo-accent-primary);
	}

	/* Legend */
	.tl-legend {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		flex-wrap: wrap;
		font-size: 0.72rem;
		color: var(--oo-text-tertiary);
	}

	.legend-item {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
	}

	.legend-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.dot-green { background: var(--oo-success); }
	.dot-yellow { background: var(--oo-warning); }
	.dot-red { background: var(--oo-danger); }

	.legend-count {
		margin-left: auto;
		font-variant-numeric: tabular-nums;
	}

	/* Error / empty */
	.tl-error {
		font-size: 0.82rem;
		color: var(--oo-danger);
		padding: 0.4rem;
	}

	.tl-empty {
		font-size: 0.82rem;
		color: var(--oo-text-tertiary);
		text-align: center;
		padding: 2rem 1rem;
	}

	/* Timeline canvas */
	.tl-canvas {
		position: relative;
		height: 180px;
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-subtle);
		border-radius: 8px;
		overflow: hidden;
		padding-bottom: 24px; /* space for axis */
	}

	/* Y-axis labels */
	.y-label {
		position: absolute;
		left: 4px;
		font-size: 0.62rem;
		color: var(--oo-text-tertiary);
		opacity: 0.6;
		pointer-events: none;
		z-index: 1;
	}

	.y-top { top: 6px; }
	.y-bottom { bottom: 28px; }

	/* Event dots */
	.event-dot {
		position: absolute;
		border-radius: 50%;
		border: none;
		cursor: pointer;
		transform: translate(-50%, -50%);
		opacity: 0.75;
		transition: opacity 0.15s, transform 0.15s, box-shadow 0.15s;
		z-index: 2;
		padding: 0;
	}

	.event-dot:hover {
		opacity: 1;
		transform: translate(-50%, -50%) scale(1.6);
		z-index: 10;
	}

	.event-dot.selected {
		opacity: 1;
		transform: translate(-50%, -50%) scale(1.8);
		box-shadow: 0 0 0 3px var(--oo-accent-primary);
		z-index: 11;
	}

	/* Tooltip */
	.tl-tooltip {
		position: absolute;
		background: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-subtle);
		border-radius: 6px;
		padding: 0.5rem 0.65rem;
		pointer-events: none;
		z-index: 20;
		min-width: 160px;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
	}

	.tt-model {
		font-size: 0.78rem;
		font-weight: 600;
		color: var(--oo-text-primary);
		margin-bottom: 0.3rem;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		max-width: 200px;
	}

	.tt-row {
		display: flex;
		justify-content: space-between;
		gap: 0.5rem;
		font-size: 0.7rem;
		line-height: 1.5;
	}

	.tt-label { color: var(--oo-text-tertiary); }
	.tt-val {
		color: var(--oo-text-primary);
		font-variant-numeric: tabular-nums;
	}

	/* X-axis */
	.tl-axis {
		position: absolute;
		bottom: 0;
		left: 0;
		right: 0;
		height: 22px;
		border-top: 1px solid var(--oo-bd-subtle);
		background: var(--oo-bg-elevated);
	}

	.axis-label {
		position: absolute;
		transform: translateX(-50%);
		font-size: 0.62rem;
		color: var(--oo-text-tertiary);
		top: 4px;
		white-space: nowrap;
	}

	/* Detail card */
	.detail-card {
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-accent-primary);
		border-radius: 8px;
		padding: 0.85rem;
	}

	.detail-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.5rem;
	}

	.detail-header h4 {
		margin: 0;
		font-size: 0.85rem;
		color: var(--oo-text-primary);
	}

	.btn-close {
		padding: 0.15rem 0.4rem;
		border-radius: 4px;
		border: 1px solid var(--oo-bd-subtle);
		background: var(--oo-bg-elevated);
		color: var(--oo-text-secondary);
		font-size: 0.75rem;
		cursor: pointer;
	}

	.btn-close:hover { background: var(--oo-bg-overlay); }

	.detail-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
		gap: 0.3rem;
	}

	.dg-row {
		display: flex;
		justify-content: space-between;
		gap: 0.5rem;
		padding: 0.2rem 0;
		font-size: 0.78rem;
	}

	.dg-label { color: var(--oo-text-tertiary); }
	.dg-value {
		color: var(--oo-text-primary);
		font-variant-numeric: tabular-nums;
		text-align: right;
	}

	.dg-mono {
		font-family: monospace;
		font-size: 0.72rem;
		max-width: 160px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
</style>
