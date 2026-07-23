<!--
  BenchmarkV2Panel.svelte
  Thin orchestrator for the Quality Evaluation engine. The 2101-line monolith
  was split into self-contained per-section components under ./benchmark
  (Run, Leaderboard, Head-to-Head, Trends, Compare, History, Profiles); the
  shared formatting/chart helpers live in ./benchmark/format and the section
  styles are loaded once from ./benchmark/benchmark.css. This file only owns
  the tab selection and renders the active section. What the engine measures
  and reports is unchanged.
-->
<script lang="ts">
	import { Tabs } from '$lib/ds';
	import type { TabItem } from '$lib/ds';
	import './benchmark/benchmark.css';
	import BenchmarkRunSection from './benchmark/BenchmarkRunSection.svelte';
	import BenchmarkLeaderboard from './benchmark/BenchmarkLeaderboard.svelte';
	import BenchmarkHeadToHead from './benchmark/BenchmarkHeadToHead.svelte';
	import BenchmarkTrends from './benchmark/BenchmarkTrends.svelte';
	import BenchmarkCompareSection from './benchmark/BenchmarkCompareSection.svelte';
	import BenchmarkHistorySection from './benchmark/BenchmarkHistorySection.svelte';
	import BenchmarkProfiles from './benchmark/BenchmarkProfiles.svelte';

	let activeTab = 'run';

	const tabs: TabItem[] = [
		{ id: 'run', label: 'Run' },
		{ id: 'leaderboard', label: 'Leaderboard' },
		{ id: 'h2h', label: 'Head-to-Head' },
		{ id: 'trends', label: 'Trends' },
		{ id: 'compare', label: 'Compare' },
		{ id: 'history', label: 'History' },
		{ id: 'profiles', label: 'Profiles' },
	];
</script>

<div class="bv2-panel">
	<Tabs bind:value={activeTab} {tabs} variant="underline" size="sm">
		{#if activeTab === 'run'}
			<BenchmarkRunSection />
		{:else if activeTab === 'leaderboard'}
			<BenchmarkLeaderboard />
		{:else if activeTab === 'h2h'}
			<BenchmarkHeadToHead />
		{:else if activeTab === 'trends'}
			<BenchmarkTrends />
		{:else if activeTab === 'compare'}
			<BenchmarkCompareSection />
		{:else if activeTab === 'history'}
			<BenchmarkHistorySection />
		{:else if activeTab === 'profiles'}
			<BenchmarkProfiles />
		{/if}
	</Tabs>
</div>
