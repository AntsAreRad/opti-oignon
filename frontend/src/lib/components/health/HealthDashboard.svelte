<!--
  HealthDashboard.svelte
  System Status content: an overview row, the six module groups (Backend,
  Inference & models, RAG & memory, Plugins & tools, Network, Security) built
  from the /api/health/dashboard module map, a live network line from
  /api/network/status, the model warmup state, the benchmark runner, and a
  derived "Recent alerts" list. All presentation is on the ds primitives and
  --oo-* tokens; the health and network APIs are unchanged.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { Card, Button, Icon } from '$lib/ds';
	import type { HealthDashboard as DashboardType, BenchmarkResultSchema } from '$lib/types';
	import { getHealthDashboard, runBenchmarks, runBenchmark } from '$lib/api/health';
	import { getNetworkStatus, type NetworkStatusInfo } from '$lib/api/network';
	import { currentMode } from '$lib/stores/securityMode';

	let dashboard: DashboardType | null = null;
	let net: NetworkStatusInfo | null = null;
	let loading = true;
	let error = '';

	// Benchmarks
	let benchmarkResults: Record<string, BenchmarkResultSchema> = {};
	let benchmarkRunning = false;
	let benchmarkError = '';
	let expandedBenchmark: string | null = null;

	// The six module groups (spec 9.4). Members are health module keys; any
	// module not listed here is collected into a trailing "Other" group.
	const MODULE_GROUPS: { id: string; label: string; icon: string; members: string[] }[] = [
		{
			id: 'backend',
			label: 'Backend',
			icon: 'server',
			members: ['conversation', 'config', 'benchmarks', 'benchmark_history', 'performance_monitor', 'analytics', 'feedback'],
		},
		{
			id: 'models',
			label: 'Inference & models',
			icon: 'cpu',
			members: ['model_warmup', 'model_profiles', 'model_health', 'smart_router', 'learned_router', 'cascading', 'speculative', 'pipelines', 'prompt_optimization', 'presets', 'system_presets', 'context_window', 'conversation_compressor'],
		},
		{
			id: 'rag',
			label: 'RAG & memory',
			icon: 'book-open',
			members: ['projects', 'project_context', 'project_triggers', 'memory', 'artifacts', 'web_search', 'response_cache', 'semantic_cache'],
		},
		{
			id: 'plugins',
			label: 'Plugins & tools',
			icon: 'plug',
			members: ['code_executor', 'sandbox', 'sandbox_tools', 'file_tools', 'coding_agent'],
		},
		{
			id: 'network',
			label: 'Network',
			icon: 'globe',
			members: ['network_manager', 'sync_queue', 'pre_cache'],
		},
		{
			id: 'security',
			label: 'Security',
			icon: 'shield-check',
			members: ['fingerprint', 'pii_sanitizer'],
		},
	];

	async function load() {
		loading = true;
		error = '';
		try {
			dashboard = await getHealthDashboard();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load system status';
		} finally {
			loading = false;
		}
		// Network status is best-effort; a failure should not blank the page.
		try {
			net = await getNetworkStatus();
		} catch {
			net = null;
		}
	}

	async function handleRunBenchmarks() {
		benchmarkRunning = true;
		benchmarkError = '';
		try {
			benchmarkResults = await runBenchmarks();
		} catch (e) {
			benchmarkError = e instanceof Error ? e.message : 'Benchmark execution failed';
		} finally {
			benchmarkRunning = false;
		}
	}

	function formatMs(ms: number): string {
		if (ms < 1) return `${(ms * 1000).toFixed(0)}us`;
		if (ms < 1000) return `${ms.toFixed(1)}ms`;
		return `${(ms / 1000).toFixed(2)}s`;
	}

	$: modules = dashboard?.modules ?? {};
	$: moduleEntries = Object.entries(modules);
	$: upCount = moduleEntries.filter(([, v]) => v).length;
	$: classified = new Set(MODULE_GROUPS.flatMap((g) => g.members));
	$: otherEntries = moduleEntries.filter(([k]) => !classified.has(k));
	$: benchmarkEntries = Object.entries(benchmarkResults);

	// Recent alerts derived from current state: degraded status + each down module.
	$: alerts = [
		...(dashboard && dashboard.status !== 'ok'
			? [{ level: 'warn', text: 'Backend reports a degraded status' }]
			: []),
		...moduleEntries
			.filter(([, v]) => !v)
			.map(([k]) => ({ level: 'warn', text: `Module ${k.replace(/_/g, ' ')} unavailable` })),
	];

	function groupEntries(members: string[]): [string, boolean][] {
		return members.filter((m) => m in modules).map((m) => [m, modules[m]]);
	}

	onMount(load);
</script>

<div class="flex flex-col gap-6">
	{#if loading}
		<div class="flex items-center gap-2 text-sm py-8 justify-center" style="color: var(--oo-fg-muted);">
			<span class="oo-spin" aria-hidden="true"></span>
			Loading system status...
		</div>
	{:else if error}
		<Card variant="flat" padding="sm" class="oo-status-error">
			<div class="flex items-center justify-between gap-3 text-sm">
				<span>{error}</span>
				<Button variant="ghost" size="sm" on:click={load}>Retry</Button>
			</div>
		</Card>
	{:else if dashboard}
		<!-- Status header -->
		<div class="flex items-center gap-3">
			<span class="oo-dot" data-state={dashboard.status === 'ok' ? 'up' : 'down'}></span>
			<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">
				System {dashboard.status === 'ok' ? 'healthy' : 'degraded'}
			</span>
			<span class="text-xs font-mono" style="color: var(--oo-fg-muted);">v{dashboard.version}</span>
			<div class="ml-auto">
				<Button variant="ghost" size="sm" iconOnly="refresh-cw" ariaLabel="Refresh system status" on:click={load} />
			</div>
		</div>

		<!-- Overview -->
		<div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
			<Card variant="flat" padding="sm">
				<div class="text-xs" style="color: var(--oo-fg-muted);">Backend</div>
				<div class="text-sm font-medium" style="color: var(--oo-fg-primary);">
					{net?.online ? 'Online' : dashboard.status === 'ok' ? 'Online' : 'Degraded'}
					{#if net && net.latency_ms > 0}
						<span class="font-mono text-xs" style="color: var(--oo-fg-muted);">({net.latency_ms.toFixed(0)}ms)</span>
					{/if}
				</div>
			</Card>
			<Card variant="flat" padding="sm">
				<div class="text-xs" style="color: var(--oo-fg-muted);">Ollama</div>
				<div class="text-sm font-medium" style="color: var(--oo-fg-primary);">
					{net ? (net.ollama_reachable ? 'Reachable' : 'Unreachable') : 'Unknown'}
				</div>
			</Card>
			<Card variant="flat" padding="sm">
				<div class="text-xs" style="color: var(--oo-fg-muted);">Modules</div>
				<div class="text-sm font-medium" style="color: var(--oo-fg-primary);">{upCount}/{moduleEntries.length}</div>
			</Card>
			<Card variant="flat" padding="sm">
				<div class="text-xs" style="color: var(--oo-fg-muted);">Mode</div>
				<div class="text-sm font-medium capitalize" style="color: var(--oo-fg-primary);">{$currentMode}</div>
			</Card>
		</div>

		<!-- Module groups -->
		<div>
			<h3 class="text-sm font-medium mb-2" style="color: var(--oo-fg-secondary);">Module status</h3>
			<div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
				{#each MODULE_GROUPS as group (group.id)}
					{@const entries = groupEntries(group.members)}
					<Card variant="flat" padding="md">
						<div class="flex items-center gap-2 mb-2">
							<span style="color: var(--oo-fg-tertiary);"><Icon name={group.icon} size="sm" /></span>
							<span class="text-xs font-medium" style="color: var(--oo-fg-primary);">{group.label}</span>
							{#if group.id === 'network' && net}
								<span class="text-[10px] ml-auto font-mono" style="color: var(--oo-fg-faint);">
									{net.online ? 'online' : 'offline'} | queue {net.queue_size}
								</span>
							{:else}
								<span class="text-[10px] ml-auto" style="color: var(--oo-fg-faint);">
									{entries.filter(([, v]) => v).length}/{entries.length}
								</span>
							{/if}
						</div>
						{#if entries.length === 0}
							<p class="text-[11px]" style="color: var(--oo-fg-faint);">No modules reported</p>
						{:else}
							<div class="flex flex-col gap-1">
								{#each entries as [name, available]}
									<div class="flex items-center gap-2 text-xs">
										<span class="oo-dot oo-dot-sm" data-state={available ? 'up' : 'down'}></span>
										<span class="truncate" style="color: {available ? 'var(--oo-fg-secondary)' : 'var(--oo-fg-faint)'};">
											{name.replace(/_/g, ' ')}
										</span>
									</div>
								{/each}
							</div>
						{/if}
						{#if group.id === 'security'}
							<div class="flex items-center gap-2 text-xs mt-1 pt-1" style="border-top: 1px solid var(--oo-bd-subtle);">
								<span class="oo-dot oo-dot-sm" data-state="up"></span>
								<span style="color: var(--oo-fg-secondary);">Security mode: <span class="capitalize">{$currentMode}</span></span>
							</div>
						{/if}
					</Card>
				{/each}

				{#if otherEntries.length > 0}
					<Card variant="flat" padding="md">
						<div class="flex items-center gap-2 mb-2">
							<span style="color: var(--oo-fg-tertiary);"><Icon name="layers" size="sm" /></span>
							<span class="text-xs font-medium" style="color: var(--oo-fg-primary);">Other</span>
							<span class="text-[10px] ml-auto" style="color: var(--oo-fg-faint);">
								{otherEntries.filter(([, v]) => v).length}/{otherEntries.length}
							</span>
						</div>
						<div class="flex flex-col gap-1">
							{#each otherEntries as [name, available]}
								<div class="flex items-center gap-2 text-xs">
									<span class="oo-dot oo-dot-sm" data-state={available ? 'up' : 'down'}></span>
									<span class="truncate" style="color: {available ? 'var(--oo-fg-secondary)' : 'var(--oo-fg-faint)'};">
										{name.replace(/_/g, ' ')}
									</span>
								</div>
							{/each}
						</div>
					</Card>
				{/if}
			</div>
		</div>

		<!-- Warmup -->
		{#if dashboard.warmup_status}
			<div>
				<h3 class="text-sm font-medium mb-2" style="color: var(--oo-fg-secondary);">Model warmup</h3>
				<Card variant="flat" padding="md">
					<div class="flex items-center gap-2 text-sm" style="color: var(--oo-fg-secondary);">
						<span style="color: var(--oo-fg-muted);">Status:</span>
						<span style="color: var(--oo-fg-primary);">
							{dashboard.warmup_status.is_warming ? 'Warming up...' : 'Idle'}
						</span>
					</div>
					{#if Array.isArray(dashboard.warmup_status.warmed_models) && dashboard.warmup_status.warmed_models.length > 0}
						<div class="flex items-center gap-2 mt-1 flex-wrap">
							<span class="text-sm" style="color: var(--oo-fg-muted);">Warmed:</span>
							{#each dashboard.warmup_status.warmed_models as model}
								<span class="text-xs px-1.5 py-0.5 rounded font-mono" style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-secondary);">
									{model}
								</span>
							{/each}
						</div>
					{/if}
				</Card>
			</div>
		{/if}

		<!-- Benchmarks -->
		<div>
			<div class="flex items-center gap-3 mb-2">
				<h3 class="text-sm font-medium" style="color: var(--oo-fg-secondary);">Benchmarks</h3>
				<Button variant="secondary" size="sm" loading={benchmarkRunning} on:click={handleRunBenchmarks}>
					Run all
				</Button>
			</div>

			{#if benchmarkError}
				<Card variant="flat" padding="sm" class="oo-status-error mb-2">
					<span class="text-xs">{benchmarkError}</span>
				</Card>
			{/if}

			{#if benchmarkEntries.length > 0}
				<div class="flex flex-col gap-2">
					{#each benchmarkEntries as [name, result]}
						<Card variant="flat" padding="sm">
							<button
								class="w-full flex items-center gap-3 text-left"
								on:click={() => (expandedBenchmark = expandedBenchmark === name ? null : name)}
							>
								<span class="oo-dot oo-dot-sm" data-state={result.error ? 'down' : 'up'}></span>
								<span class="text-sm flex-1" style="color: var(--oo-fg-primary);">{result.name}</span>
								<span class="text-xs font-mono" style="color: var(--oo-fg-muted);">{formatMs(result.mean_ms)} avg</span>
								<span class="oo-chev" class:oo-chev-open={expandedBenchmark === name} style="color: var(--oo-fg-faint);">
									<Icon name="chevron-down" size="sm" />
								</span>
							</button>

							{#if expandedBenchmark === name}
								<div class="mt-2 pt-2" style="border-top: 1px solid var(--oo-bd-subtle);">
									{#if result.error}
										<p class="text-xs mb-2" style="color: var(--oo-error);">{result.error}</p>
									{/if}
									<div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs" style="color: var(--oo-fg-secondary);">
										<div><span style="color: var(--oo-fg-faint);">Iterations:</span> {result.iterations}</div>
										<div><span style="color: var(--oo-fg-faint);">Mean:</span> {formatMs(result.mean_ms)}</div>
										<div><span style="color: var(--oo-fg-faint);">Median:</span> {formatMs(result.median_ms)}</div>
										<div><span style="color: var(--oo-fg-faint);">Min:</span> {formatMs(result.min_ms)}</div>
										<div><span style="color: var(--oo-fg-faint);">Max:</span> {formatMs(result.max_ms)}</div>
										<div><span style="color: var(--oo-fg-faint);">Stddev:</span> {formatMs(result.stddev_ms)}</div>
										<div><span style="color: var(--oo-fg-faint);">P95:</span> {formatMs(result.p95_ms)}</div>
										<div><span style="color: var(--oo-fg-faint);">P99:</span> {formatMs(result.p99_ms)}</div>
									</div>
									{#if result.throughput_ops > 0}
										<div class="mt-2 text-xs" style="color: var(--oo-fg-secondary);">
											<span style="color: var(--oo-fg-faint);">Throughput:</span> {result.throughput_ops.toFixed(0)} ops/s
										</div>
									{/if}
								</div>
							{/if}
						</Card>
					{/each}
				</div>
			{:else}
				<p class="text-xs" style="color: var(--oo-fg-faint);">Run benchmarks to see latency metrics.</p>
			{/if}
		</div>

		<!-- Recent alerts -->
		<div>
			<h3 class="text-sm font-medium mb-2" style="color: var(--oo-fg-secondary);">Recent alerts</h3>
			{#if alerts.length === 0}
				<Card variant="flat" padding="sm">
					<div class="flex items-center gap-2 text-xs" style="color: var(--oo-fg-muted);">
						<span class="oo-dot oo-dot-sm" data-state="up"></span>
						No active alerts.
					</div>
				</Card>
			{:else}
				<div class="flex flex-col gap-1.5">
					{#each alerts as alert}
						<div class="flex items-center gap-2 px-3 py-2 rounded text-xs" style="background-color: var(--oo-warning-bg); color: var(--oo-warning);">
							<Icon name="alert-triangle" size="sm" />
							<span>{alert.text}</span>
						</div>
					{/each}
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.oo-spin {
		width: 1.25rem;
		height: 1.25rem;
		border: 2px solid var(--oo-bd-default);
		border-top-color: var(--oo-acc-500);
		border-radius: var(--oo-radius-full);
		display: inline-block;
		animation: oo-spin 0.7s linear infinite;
	}
	@keyframes oo-spin {
		to {
			transform: rotate(360deg);
		}
	}
	.oo-dot {
		width: 0.6rem;
		height: 0.6rem;
		border-radius: var(--oo-radius-full);
		display: inline-block;
		flex-shrink: 0;
		background-color: var(--oo-fg-faint);
	}
	.oo-dot-sm {
		width: 0.5rem;
		height: 0.5rem;
	}
	.oo-dot[data-state='up'] {
		background-color: var(--oo-success);
	}
	.oo-dot[data-state='down'] {
		background-color: var(--oo-error);
	}
	.oo-chev {
		display: inline-flex;
		transition: transform var(--oo-motion-fast) var(--oo-ease-default);
	}
	.oo-chev-open {
		transform: rotate(180deg);
	}
	:global(.oo-status-error) {
		background-color: var(--oo-error-bg);
		border-color: var(--oo-error-bd);
		color: var(--oo-error);
	}
</style>
