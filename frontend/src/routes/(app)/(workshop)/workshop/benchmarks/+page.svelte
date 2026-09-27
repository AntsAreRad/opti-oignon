<!--
  Benchmark page.
  The quality evaluation, one tab at a time: the tab lives in the address
  (?tab=, read and written by lib/benchmark/tab.ts), so a reload, Back and
  a shared link land on it. A ?run=<id> query (each run of the History tab
  links it, with tab=history) opens that run's detail in a drawer; closing
  it removes run and keeps the rest of the address, so it lands where the
  run was opened from.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import BenchmarkPage from '$lib/components/panels/BenchmarkPage.svelte';
	import BenchmarkRunDrawer from '$lib/components/panels/benchmark/BenchmarkRunDrawer.svelte';
	import { benchmarkTab, tabAddress, type BenchmarkTab } from '$lib/benchmark/tab';

	$: runId = $page.url.searchParams.get('run') ?? '';
	$: tab = benchmarkTab($page.url);

	function showTab(next: BenchmarkTab) {
		goto(tabAddress($page.url, next), { replaceState: true, keepFocus: true, noScroll: true });
	}

	function closeDrawer() {
		const url = new URL($page.url);
		url.searchParams.delete('run');
		goto(`${url.pathname}${url.search}`, { replaceState: true, keepFocus: true, noScroll: true });
	}
</script>

<div class="oo-bench-page">
	<BenchmarkPage {tab} on:change={(event) => showTab(event.detail)} />
</div>

<BenchmarkRunDrawer {runId} open={!!runId} onClose={closeDrawer} />

<style>
	.oo-bench-page {
		height: 100%;
		overflow-y: auto;
	}
</style>
