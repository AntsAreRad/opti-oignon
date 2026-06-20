<!--
  BenchmarkHeadToHead.svelte (S169)
  The "Head-to-Head" section extracted from BenchmarkV2Panel: pick two models
  and compare them directly. Self-contained; behaviour unchanged.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import type { BenchmarkV2HeadToHeadResponse } from '$lib/types';
	import { getHeadToHead } from '$lib/api/benchmarkV2';
	import { pct, winnerClass } from './format';

	let availableModels: string[] = [];
	let h2hModelA = '';
	let h2hModelB = '';
	let h2hResult: BenchmarkV2HeadToHeadResponse | null = null;
	let h2hLoading = false;

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

	async function loadH2H() {
		if (!h2hModelA || !h2hModelB || h2hModelA === h2hModelB) return;
		h2hLoading = true;
		h2hResult = null;
		try {
			h2hResult = await getHeadToHead(h2hModelA, h2hModelB);
		} catch {
			// silent
		} finally {
			h2hLoading = false;
		}
	}
</script>

		<div class="bv2-section">
			<h3 class="bv2-subtitle">Head-to-Head Comparison</h3>
			<div class="bv2-h2h-selectors">
				<div class="bv2-field">
					<label class="bv2-label" for="h2h-a">Model A</label>
					<select id="h2h-a" class="bv2-select" bind:value={h2hModelA}>
						<option value="" disabled>Select model</option>
						{#each availableModels as m}
							<option value={m}>{m}</option>
						{/each}
					</select>
				</div>
				<span class="bv2-h2h-vs">vs</span>
				<div class="bv2-field">
					<label class="bv2-label" for="h2h-b">Model B</label>
					<select id="h2h-b" class="bv2-select" bind:value={h2hModelB}>
						<option value="" disabled>Select model</option>
						{#each availableModels as m}
							<option value={m}>{m}</option>
						{/each}
					</select>
				</div>
				<button
					class="bv2-run-btn"
					disabled={h2hLoading || !h2hModelA || !h2hModelB || h2hModelA === h2hModelB}
					on:click={loadH2H}
				>
					Compare
				</button>
			</div>

			{#if h2hLoading}
				<p class="bv2-hint">Loading comparison...</p>
			{/if}

			{#if h2hResult}
				<div class="bv2-h2h-results">
					<div class="bv2-h2h-overall">
						<span class="bv2-h2h-winner">
							{#if h2hResult.overall_winner === 'tie'}
								Tie
							{:else}
								Winner: {h2hResult.overall_winner}
							{/if}
						</span>
						<span class="bv2-hint">
							{h2hResult.model_a_wins}–{h2hResult.model_b_wins}
							{#if h2hResult.ties > 0}({h2hResult.ties} ties){/if}
						</span>
					</div>
					<div class="bv2-h2h-metrics">
						{#each h2hResult.metrics as metric}
							<div class="bv2-h2h-row {winnerClass(metric.winner, h2hResult.model_a)}">
								<span class="bv2-h2h-val-a">{pct(metric.model_a_value)}</span>
								<span class="bv2-h2h-metric-name">{metric.metric}</span>
								<span class="bv2-h2h-val-b">{pct(metric.model_b_value)}</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		</div>
