<!--
  BenchmarkLeaderboard.svelte
  The "Leaderboard" section extracted from BenchmarkV2Panel: ranked model
  scores plus role recommendations. Self-contained; behaviour unchanged.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import type { BenchmarkV2LeaderboardEntry, BenchmarkV2RecommendationEntry } from '$lib/types';
	import { getLeaderboard, getRecommendations, applyRecommendations } from '$lib/api/benchmarkV2';
	import { scoreColor, pct, roleLabel } from './format';
	import { EmptyState, InlineError } from '$lib/ds';
	import { parseApiError } from '$lib/api/errorHandler';

	let loadError: string | null = null;

	let leaderboardEntries: BenchmarkV2LeaderboardEntry[] = [];
	let leaderboardLoading = false;
	let recommendations: BenchmarkV2RecommendationEntry[] = [];
	let recApplied = false;
	let recApplying = false;

	onMount(loadLeaderboard);

	async function loadLeaderboard() {
		leaderboardLoading = true;
		loadError = null;
		try {
			const data = await getLeaderboard(undefined, 20);
			leaderboardEntries = data.entries;
		} catch (e) {
			loadError = parseApiError(e, 'loading the leaderboard').message;
		} finally {
			leaderboardLoading = false;
		}
		try {
			const rec = await getRecommendations();
			recommendations = rec.recommendations;
			recApplied = rec.applied;
		} catch {
			// silent
		}
	}

	async function handleApplyRecommendations() {
		recApplying = true;
		try {
			const res = await applyRecommendations();
			recApplied = res.applied;
		} catch {
			// silent
		} finally {
			recApplying = false;
		}
	}
</script>

		<div class="bv2-section">
			{#if loadError}
				<InlineError message={loadError} onRetry={loadLeaderboard} />
			{:else if leaderboardLoading}
				<p class="bv2-hint">Loading leaderboard...</p>
			{:else if leaderboardEntries.length === 0}
				<EmptyState
					size="sm"
					icon="trophy"
					title="No benchmark data yet"
					description="Run evaluations to populate the leaderboard."
				/>
			{:else}
				<h3 class="bv2-subtitle">Model Leaderboard</h3>
				<div class="bv2-table-wrap">
					<table class="bv2-table">
						<thead>
							<tr>
								<th>#</th>
								<th>Model</th>
								<th>Composite</th>
								<th>Accuracy</th>
								<th>Code</th>
								<th>Structure</th>
								<th>Speed</th>
								<th>Runs</th>
							</tr>
						</thead>
						<tbody>
							{#each leaderboardEntries as entry}
								<tr>
									<td class="bv2-rank">{entry.rank}</td>
									<td class="bv2-model-name">{entry.model}</td>
									<td class="bv2-composite" style="color: {scoreColor(entry.composite)}">{pct(entry.composite)}</td>
									<td style="color: {scoreColor(entry.accuracy_avg)}">{pct(entry.accuracy_avg)}</td>
									<td style="color: {scoreColor(entry.code_avg)}">{pct(entry.code_avg)}</td>
									<td style="color: {scoreColor(entry.structure_avg)}">{pct(entry.structure_avg)}</td>
									<td style="color: {scoreColor(entry.speed_avg)}">{pct(entry.speed_avg)}</td>
									<td>{entry.run_count}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>

				<!-- Recommendations -->
				{#if recommendations.length > 0}
					<h3 class="bv2-subtitle" style="margin-top: 1.5rem;">Recommendations</h3>
					<div class="bv2-rec-grid">
						{#each recommendations as rec}
							<div class="bv2-rec-card">
								<div class="bv2-rec-role">{roleLabel(rec.role)}</div>
								<div class="bv2-rec-model">{rec.model}</div>
								<div class="bv2-rec-score" style="color: {scoreColor(rec.composite_score)}">
									{pct(rec.composite_score)}
								</div>
								<div class="bv2-rec-reason">{rec.reason}</div>
							</div>
						{/each}
					</div>
					<button
						class="bv2-btn-sm"
						style="margin-top: 0.75rem;"
						disabled={recApplied || recApplying}
						on:click={handleApplyRecommendations}
					>
						{#if recApplied}
							Applied
						{:else if recApplying}
							Applying...
						{:else}
							Apply to Smart Router
						{/if}
					</button>
				{/if}
			{/if}
		</div>
