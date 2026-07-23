<!--
  BenchmarkPage.svelte
  Main benchmark dashboard with three tabs:
  - Run: Configure and execute benchmarks with live progress
  - History: Browse past runs, view details, compare
  - Model Assignment: Edit model-to-role routing config
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		initBenchmarkStore,
		disconnectWs,
		benchmarkError,
	} from '$lib/stores/benchmark';
	import BenchmarkRunner from '$lib/components/panels/BenchmarkRunner.svelte';
	import BenchmarkHistory from '$lib/components/panels/BenchmarkHistory.svelte';
	import ModelAssignment from '$lib/components/panels/ModelAssignment.svelte';
	import BenchmarkV2Panel from '$lib/components/panels/BenchmarkV2Panel.svelte';

	type Tab = 'run' | 'history' | 'models' | 'evaluation';
	let activeTab: Tab = 'evaluation';

	const tabs: { id: Tab; label: string; icon: string }[] = [
		{ id: 'evaluation', label: 'Quality Evaluation', icon: '◆' },
		{ id: 'run', label: 'Run Benchmark', icon: '▶' },
		{ id: 'history', label: 'History', icon: '☰' },
		{ id: 'models', label: 'Model Assignment', icon: '⚙' },
	];

	let storeLoading = true;

	onMount(async () => {
		await initBenchmarkStore();
		storeLoading = false;
	});

	onDestroy(() => {
		disconnectWs();
	});

	function dismissError() {
		benchmarkError.set(null);
	}
</script>

<div class="benchmark-page">
	<header class="benchmark-header">
		<h1>Benchmark Dashboard</h1>
		<p class="subtitle">Evaluate, compare, and optimize your local LLM models</p>
	</header>

	{#if storeLoading}
		<div class="flex items-center gap-2 py-8 justify-center" style="color: var(--oo-fg-muted);">
			<div class="w-5 h-5 border-2 rounded-full animate-spin"
				style="border-color: var(--oo-bd-default); border-top-color: var(--oo-acc-400);" />
			<span class="text-sm">Loading benchmark data...</span>
		</div>
	{:else}

	{#if $benchmarkError}
		<div class="error-banner" role="alert">
			<span class="error-icon">!</span>
			<span class="error-text">{$benchmarkError}</span>
			<button class="error-dismiss" on:click={dismissError} aria-label="Dismiss">&times;</button>
		</div>
	{/if}

	<nav class="tab-bar" role="tablist">
		{#each tabs as tab}
			<button
				class="tab-btn"
				class:active={activeTab === tab.id}
				role="tab"
				aria-selected={activeTab === tab.id}
				on:click={() => (activeTab = tab.id)}
			>
				<span class="tab-icon">{tab.icon}</span>
				{tab.label}
			</button>
		{/each}
	</nav>

	<div class="tab-content">
		{#if activeTab === 'evaluation'}
			<BenchmarkV2Panel />
		{:else if activeTab === 'run'}
			<BenchmarkRunner />
		{:else if activeTab === 'history'}
			<BenchmarkHistory />
		{:else if activeTab === 'models'}
			<ModelAssignment />
		{/if}
	</div>
	{/if}
</div>

<style>
	.benchmark-page {
		max-width: 1200px;
		margin: 0 auto;
		padding: 1.5rem;
		color: var(--oo-fg-primary);
	}

	.benchmark-header {
		margin-bottom: 1.5rem;
	}

	.benchmark-header h1 {
		font-size: 1.5rem;
		font-weight: 600;
		margin: 0 0 0.25rem 0;
		color: var(--oo-fg-primary);
	}

	.subtitle {
		font-size: 0.85rem;
		color: var(--oo-fg-tertiary);
		margin: 0;
	}

	.error-banner {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.625rem 0.875rem;
		margin-bottom: 1rem;
		background: var(--oo-error-bg);
		border: 1px solid var(--oo-error-bd);
		border-radius: 6px;
		font-size: 0.8rem;
		color: var(--oo-error);
	}

	.error-icon {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 18px;
		height: 18px;
		border-radius: 50%;
		background: var(--oo-error);
		color: var(--oo-fg-on-semantic);
		font-size: 0.65rem;
		font-weight: 700;
		flex-shrink: 0;
	}

	.error-text {
		flex: 1;
	}

	.error-dismiss {
		background: none;
		border: none;
		color: var(--oo-error);
		cursor: pointer;
		font-size: 1rem;
		padding: 0 0.25rem;
		opacity: 0.7;
	}

	.error-dismiss:hover {
		opacity: 1;
	}

	.tab-bar {
		display: flex;
		gap: 0.25rem;
		border-bottom: 1px solid var(--oo-bd-default);
		margin-bottom: 1.25rem;
	}

	.tab-btn {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		padding: 0.5rem 1rem;
		background: none;
		border: none;
		border-bottom: 2px solid transparent;
		color: var(--oo-fg-tertiary);
		font-size: 0.8rem;
		cursor: pointer;
		transition: color 0.15s, border-color 0.15s;
		margin-bottom: -1px;
	}

	.tab-btn:hover {
		color: var(--oo-fg-secondary);
	}

	.tab-btn.active {
		color: var(--oo-acc-400);
		border-bottom-color: var(--oo-acc-400);
	}

	.tab-icon {
		font-size: 0.75rem;
	}

	.tab-content {
		min-height: 400px;
	}
</style>
