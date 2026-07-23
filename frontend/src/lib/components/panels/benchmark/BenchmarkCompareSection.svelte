<!--
  BenchmarkCompareSection.svelte
  The "Compare" section extracted from BenchmarkV2Panel: aggregated
  multi-model comparison table. Self-contained; behaviour unchanged.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { compareModels } from '$lib/api/benchmarkV2';
	import { scoreColor, pct } from './format';

	let compareData: Record<string, unknown>[] = [];
	let compareLoading = false;

	onMount(loadCompare);

	async function loadCompare() {
		compareLoading = true;
		try {
			const data = await compareModels(undefined, undefined, 15);
			compareData = data.models;
		} catch {
			// silent
		} finally {
			compareLoading = false;
		}
	}
</script>

		<div class="bv2-section">
			{#if compareLoading}
				<p class="bv2-hint">Loading comparison data...</p>
			{:else if compareData.length === 0}
				<p class="bv2-hint">No benchmark data available yet. Run some benchmarks first.</p>
			{:else}
				<h3 class="bv2-subtitle">Model Comparison (Aggregated)</h3>
				<div class="bv2-table-wrap">
					<table class="bv2-table">
						<thead>
							<tr>
								<th>Model</th>
								<th>Avg Accuracy</th>
								<th>Avg Code</th>
								<th>Avg Structure</th>
								<th>Avg Speed</th>
								<th>Avg Composite</th>
								<th>Runs</th>
							</tr>
						</thead>
						<tbody>
							{#each compareData as row}
								<tr>
									<td class="bv2-model-name">{row.model ?? '-'}</td>
									<td style="color: {scoreColor(Number(row.avg_accuracy ?? 0))}">{pct(Number(row.avg_accuracy ?? 0))}</td>
									<td style="color: {scoreColor(Number(row.avg_code ?? 0))}">{pct(Number(row.avg_code ?? 0))}</td>
									<td style="color: {scoreColor(Number(row.avg_structure ?? 0))}">{pct(Number(row.avg_structure ?? 0))}</td>
									<td style="color: {scoreColor(Number(row.avg_speed ?? 0))}">{pct(Number(row.avg_speed ?? 0))}</td>
									<td class="bv2-composite" style="color: {scoreColor(Number(row.avg_composite ?? 0))}">{pct(Number(row.avg_composite ?? 0))}</td>
									<td>{row.run_count ?? 0}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</div>
