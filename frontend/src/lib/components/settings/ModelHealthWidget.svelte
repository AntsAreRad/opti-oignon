<!--
  ModelHealthWidget.svelte
  Compact model health status grid showing each model's health
  (green/yellow/red dots), latency bar, last check timestamp,
  and a "Check All" button for manual refresh.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		getAllModelHealth,
		forceHealthCheck,
		type ModelHealthRecord,
		type AllModelHealthResponse,
	} from '$lib/api/smartRouting';

	let data: AllModelHealthResponse | null = null;
	let loading = true;
	let checking = false;
	let error = '';
	let pollTimer: ReturnType<typeof setInterval> | null = null;

	// Sort records: unavailable first, then degraded, then healthy, then unknown
	const STATUS_ORDER: Record<string, number> = {
		unavailable: 0,
		degraded: 1,
		healthy: 2,
		unknown: 3,
	};

	$: sortedRecords = data
		? Object.values(data.records).sort(
				(a, b) => (STATUS_ORDER[a.status] ?? 4) - (STATUS_ORDER[b.status] ?? 4)
			)
		: [];

	$: summaryText = data?.summary
		? `${data.summary.healthy ?? 0} healthy, ${data.summary.degraded ?? 0} degraded, ${data.summary.unavailable ?? 0} unavailable`
		: '';

	function statusColor(status: string): string {
		switch (status) {
			case 'healthy':
				return 'var(--oo-success)';
			case 'degraded':
				return 'var(--oo-warning)';
			case 'unavailable':
				return 'var(--oo-error)';
			default:
				return 'var(--oo-fg-tertiary)';
		}
	}

	function statusLabel(status: string): string {
		switch (status) {
			case 'healthy':
				return 'Healthy';
			case 'degraded':
				return 'Degraded';
			case 'unavailable':
				return 'Unavailable';
			default:
				return 'Unknown';
		}
	}

	function latencyPercent(ms: number): number {
		// Scale: 0-5000ms maps to 0-100%
		return Math.min(100, (ms / 5000) * 100);
	}

	function latencyColor(ms: number): string {
		if (ms <= 0) return 'var(--oo-fg-tertiary)';
		if (ms < 1000) return 'var(--oo-success)';
		if (ms < 3000) return 'var(--oo-warning)';
		return 'var(--oo-error)';
	}

	function formatTimestamp(ts: number): string {
		if (!ts || ts <= 0) return 'Never';
		const d = new Date(ts * 1000);
		const now = Date.now();
		const diff = now - d.getTime();
		if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`;
		if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
		return d.toLocaleTimeString();
	}

	async function load() {
		loading = true;
		error = '';
		try {
			data = await getAllModelHealth();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load health data';
		} finally {
			loading = false;
		}
	}

	async function handleCheckAll() {
		checking = true;
		error = '';
		try {
			const result = await forceHealthCheck();
			// Reload full data after check
			await load();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Health check failed';
		} finally {
			checking = false;
		}
	}

	onMount(() => {
		load();
		// Poll every 30 seconds
		pollTimer = setInterval(load, 30000);
	});

	onDestroy(() => {
		if (pollTimer) clearInterval(pollTimer);
	});
</script>

<div class="health-widget">
	<div class="health-header">
		<h3 class="health-title">Model Health</h3>
		<button
			class="check-btn"
			on:click={handleCheckAll}
			disabled={checking}
			title="Force health check on all models"
		>
			{checking ? 'Checking...' : 'Check All'}
		</button>
	</div>

	{#if loading && !data}
		<div class="health-loading">
			<div class="spinner" />
			Loading health data...
		</div>
	{:else if error && !data}
		<div class="health-error">
			{error}
			<button on:click={load} class="retry-link">Retry</button>
		</div>
	{:else if sortedRecords.length === 0}
		<div class="health-empty">
			No models tracked yet. Run a health check to discover models.
		</div>
	{:else}
		<div class="health-summary">{summaryText}</div>
		<div class="health-grid">
			{#each sortedRecords as record (record.model)}
				<div class="health-row">
					<div class="health-dot-wrap" title={statusLabel(record.status)}>
						<span
							class="health-dot"
							style="background: {statusColor(record.status)}"
						/>
					</div>
					<span class="health-model" title={record.model}>
						{record.model}
					</span>
					<div class="health-latency" title="{record.latency_ms.toFixed(0)}ms">
						<div class="latency-track">
							<div
								class="latency-fill"
								style="width: {latencyPercent(record.latency_ms)}%; background: {latencyColor(record.latency_ms)}"
							/>
						</div>
						<span class="latency-text">
							{record.latency_ms > 0 ? `${record.latency_ms.toFixed(0)}ms` : '-'}
						</span>
					</div>
					<span class="health-time" title="Last check">
						{formatTimestamp(record.last_check)}
					</span>
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.health-widget {
		border: 1px solid var(--oo-bd-default);
		border-radius: 8px;
		padding: 12px;
		background: var(--oo-bg-elevated);
	}

	.health-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 8px;
	}

	.health-title {
		font-size: 0.8rem;
		font-weight: 600;
		color: var(--oo-fg-primary);
		margin: 0;
	}

	.check-btn {
		padding: 3px 10px;
		font-size: 0.7rem;
		border-radius: 4px;
		border: 1px solid var(--oo-bd-default);
		background: var(--oo-bg-surface);
		color: var(--oo-fg-secondary);
		cursor: pointer;
		transition: background 0.15s, border-color 0.15s;
	}

	.check-btn:hover:not(:disabled) {
		background: var(--oo-bg-overlay);
		border-color: var(--oo-acc-500);
	}

	.check-btn:disabled {
		opacity: 0.5;
		cursor: default;
	}

	.health-loading,
	.health-error,
	.health-empty {
		font-size: 0.75rem;
		color: var(--oo-fg-tertiary);
		padding: 8px 0;
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.health-error {
		color: var(--oo-error);
	}

	.retry-link {
		background: none;
		border: none;
		color: var(--oo-acc-400);
		cursor: pointer;
		text-decoration: underline;
		font-size: 0.75rem;
		padding: 0;
	}

	.spinner {
		width: 14px;
		height: 14px;
		border: 2px solid var(--oo-bd-default);
		border-top-color: var(--oo-acc-500);
		border-radius: 50%;
		animation: spin 0.6s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	.health-summary {
		font-size: 0.7rem;
		color: var(--oo-fg-tertiary);
		margin-bottom: 8px;
	}

	.health-grid {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.health-row {
		display: grid;
		grid-template-columns: 16px 1fr 100px 55px;
		align-items: center;
		gap: 8px;
		padding: 3px 0;
	}

	.health-dot-wrap {
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.health-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		display: inline-block;
		flex-shrink: 0;
	}

	.health-model {
		font-family: 'JetBrains Mono', 'Fira Code', monospace;
		font-size: 0.7rem;
		color: var(--oo-fg-secondary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.health-latency {
		display: flex;
		align-items: center;
		gap: 4px;
	}

	.latency-track {
		flex: 1;
		height: 4px;
		background: var(--oo-bg-surface);
		border-radius: 2px;
		overflow: hidden;
	}

	.latency-fill {
		height: 100%;
		border-radius: 2px;
		transition: width 0.3s ease;
	}

	.latency-text {
		font-size: 0.6rem;
		color: var(--oo-fg-tertiary);
		font-family: 'JetBrains Mono', 'Fira Code', monospace;
		min-width: 32px;
		text-align: right;
	}

	.health-time {
		font-size: 0.6rem;
		color: var(--oo-fg-tertiary);
		text-align: right;
		white-space: nowrap;
	}
</style>
