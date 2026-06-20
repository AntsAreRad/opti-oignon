<!--
  BenchmarkTrends.svelte (S169)
  The "Trends" section extracted from BenchmarkV2Panel: composite score trend
  over time for a selected model. Self-contained; behaviour unchanged.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import type { BenchmarkV2TrendResponse } from '$lib/types';
	import { getTrends } from '$lib/api/benchmarkV2';
	import { scoreColor, pct, formatDate, trendPath } from './format';
	import { EmptyState } from '$lib/ds';

	let availableModels: string[] = [];
	let trendsModel = '';
	let trendsData: BenchmarkV2TrendResponse | null = null;
	let trendsLoading = false;

	onMount(loadModels);

	async function loadModels() {
		try {
			const resp = await fetch('/api/models');
			if (resp.ok) {
				const data = await resp.json();
				availableModels = (data.models || []).map((m: { name?: string; model?: string }) => m.name || m.model || '');
			}
		} catch {
			// Models endpoint may not be available
		}
	}

	async function loadTrends() {
		if (!trendsModel) return;
		trendsLoading = true;
		trendsData = null;
		try {
			trendsData = await getTrends(trendsModel);
		} catch {
			// silent
		} finally {
			trendsLoading = false;
		}
	}
</script>

		<div class="bv2-section">
			<h3 class="bv2-subtitle">Performance Trends</h3>
			<div class="bv2-trends-selector">
				<div class="bv2-field">
					<label class="bv2-label" for="trends-model">Model</label>
					<select id="trends-model" class="bv2-select" bind:value={trendsModel}>
						<option value="" disabled>Select model</option>
						{#each availableModels as m}
							<option value={m}>{m}</option>
						{/each}
					</select>
				</div>
				<button
					class="bv2-run-btn"
					disabled={trendsLoading || !trendsModel}
					on:click={loadTrends}
				>
					Load Trends
				</button>
			</div>

			{#if trendsLoading}
				<p class="bv2-hint">Loading trends...</p>
			{:else if !trendsData}
				<EmptyState
					size="sm"
					icon="trending-up"
					title="No trend loaded"
					description="Pick a model and load its trend to see composite score over time."
				/>
			{:else if trendsData.points.length === 0}
				<EmptyState
					size="sm"
					icon="trending-up"
					title="No data points yet"
					description="This model has no recorded benchmark runs to chart."
				/>
			{:else}
				<div class="bv2-trends-info">
					<span class="bv2-trends-direction">
						Trend: <strong>{trendsData.trend_direction}</strong>
					</span>
					{#if trendsData.regression_detected}
						<span class="bv2-trends-regression">Regression detected</span>
					{/if}
					<span class="bv2-hint">{trendsData.points.length} data points</span>
				</div>

				{#if trendsData.points.length >= 2}
					<div class="bv2-trends-chart">
						<svg viewBox="0 0 500 120" class="bv2-sparkline-svg" preserveAspectRatio="none">
							<path
								d={trendPath(trendsData.points, 500, 120)}
								fill="none"
								stroke="var(--oo-acc-400)"
								stroke-width="2"
							/>
						</svg>
					</div>
				{/if}

				<div class="bv2-table-wrap" style="margin-top: 0.75rem;">
					<table class="bv2-table">
						<thead>
							<tr>
								<th>Date</th>
								<th>Profile</th>
								<th>Composite</th>
								<th>Accuracy</th>
								<th>Code</th>
								<th>Structure</th>
								<th>Speed</th>
							</tr>
						</thead>
						<tbody>
							{#each trendsData.points.slice().reverse().slice(0, 15) as pt}
								<tr>
									<td>{formatDate(pt.timestamp)}</td>
									<td>{pt.profile}</td>
									<td class="bv2-composite" style="color: {scoreColor(pt.composite)}">{pct(pt.composite)}</td>
									<td style="color: {scoreColor(pt.accuracy)}">{pct(pt.accuracy)}</td>
									<td style="color: {scoreColor(pt.code)}">{pct(pt.code)}</td>
									<td style="color: {scoreColor(pt.structure)}">{pct(pt.structure)}</td>
									<td style="color: {scoreColor(pt.speed)}">{pct(pt.speed)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</div>
