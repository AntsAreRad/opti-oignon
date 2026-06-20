<!--
  Benchmark page (S169).
  Renders the benchmark dashboard inside the shared AppShell (from +layout).
  A ?run=<id> query (set by the sidebar runs list) opens the per-run detail
  drawer (spec 9.5: per-run detail in a drawer-right).
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import BenchmarkPage from '$lib/components/panels/BenchmarkPage.svelte';
	import BenchmarkRunDrawer from '$lib/components/panels/benchmark/BenchmarkRunDrawer.svelte';

	$: runId = $page.url.searchParams.get('run') ?? '';

	function closeDrawer() {
		const url = new URL($page.url);
		url.searchParams.delete('run');
		goto(`${url.pathname}${url.search}`, { replaceState: true, keepFocus: true, noScroll: true });
	}
</script>

<div class="h-full overflow-y-auto">
	<BenchmarkPage />
</div>

<BenchmarkRunDrawer {runId} open={!!runId} onClose={closeDrawer} />
