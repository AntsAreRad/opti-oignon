<!--
  BenchmarkPage.svelte
  The benchmarks page: the quality evaluation, in one row of the design
  system's tabs over its seven sections (Run, Leaderboard, Head-to-head,
  Trends, Compare, History, Profiles). The row scrolls within itself on a
  phone, never the page.

  The tab is the page's to keep in its address (lib/benchmark/tab.ts): it
  is given here as `tab`, and a change is dispatched as `change`. The
  sections' shared styles are loaded once, here, from
  ./benchmark/benchmark.css.

  Model roles are assigned at Workshop > Models and inference > Model
  assignment; the page runs one engine, the evaluation's.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { Tabs } from '$lib/ds';
	import type { TabItem } from '$lib/ds';
	import type { BenchmarkTab } from '$lib/benchmark/tab';
	import './benchmark/benchmark.css';
	import BenchmarkRunSection from './benchmark/BenchmarkRunSection.svelte';
	import BenchmarkLeaderboard from './benchmark/BenchmarkLeaderboard.svelte';
	import BenchmarkHeadToHead from './benchmark/BenchmarkHeadToHead.svelte';
	import BenchmarkTrends from './benchmark/BenchmarkTrends.svelte';
	import BenchmarkCompareSection from './benchmark/BenchmarkCompareSection.svelte';
	import BenchmarkHistorySection from './benchmark/BenchmarkHistorySection.svelte';
	import BenchmarkProfiles from './benchmark/BenchmarkProfiles.svelte';

	/** The section shown. */
	export let tab: BenchmarkTab = 'run';

	const dispatch = createEventDispatcher<{ change: BenchmarkTab }>();

	function choose(event: CustomEvent<string>) {
		dispatch('change', event.detail as BenchmarkTab);
	}

	const tabs: TabItem[] = [
		{ id: 'run', label: 'Run' },
		{ id: 'leaderboard', label: 'Leaderboard' },
		{ id: 'h2h', label: 'Head-to-head' },
		{ id: 'trends', label: 'Trends' },
		{ id: 'compare', label: 'Compare' },
		{ id: 'history', label: 'History' },
		{ id: 'profiles', label: 'Profiles' }
	];
</script>

<div class="oo-bench">
	<header class="oo-bench-head">
		<h1 class="oo-bench-title">Benchmarks</h1>
		<p class="oo-bench-desc">
			The quality evaluation: run a profile over your models, then rank, compare and follow them.
		</p>
	</header>

	<Tabs
		value={tab}
		{tabs}
		variant="underline"
		size="sm"
		on:change={choose}
	>
		{#if tab === 'run'}
			<BenchmarkRunSection />
		{:else if tab === 'leaderboard'}
			<BenchmarkLeaderboard />
		{:else if tab === 'h2h'}
			<BenchmarkHeadToHead />
		{:else if tab === 'trends'}
			<BenchmarkTrends />
		{:else if tab === 'compare'}
			<BenchmarkCompareSection />
		{:else if tab === 'history'}
			<BenchmarkHistorySection />
		{:else if tab === 'profiles'}
			<BenchmarkProfiles />
		{/if}
	</Tabs>
</div>

<style>
	.oo-bench {
		box-sizing: border-box;
		max-width: 1200px;
		margin: 0 auto;
		padding: var(--oo-space-6) var(--oo-space-5) var(--oo-space-9);
		color: var(--oo-fg-primary);
	}

	.oo-bench-head {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		margin-bottom: var(--oo-space-5);
	}
	.oo-bench-title {
		margin: 0;
		color: var(--oo-fg-primary);
		font-family: var(--oo-font-serif);
		font-size: var(--oo-text-3xl);
		font-weight: 400;
		line-height: var(--oo-leading-tight);
	}
	.oo-bench-desc {
		margin: 0;
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
	}
</style>
