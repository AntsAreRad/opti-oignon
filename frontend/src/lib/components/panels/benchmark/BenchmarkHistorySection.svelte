<!--
  BenchmarkHistorySection.svelte
  The "History" section of the benchmarks page: the fifty latest
  runs, filtered by a run's id, profile or models. Each run's profile links
  to the same page with ?tab=history&run=<id>, which opens the run's detail
  drawer over History, and closes onto it
  (every model's accuracy, code, structure and speed): the history is how
  the interface reaches it.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import type { BenchmarkV2HistoryEntry } from '$lib/types';
	import { getHistory } from '$lib/api/benchmarkV2';
	import { scoreColor, pct, formatDuration, formatDate } from './format';
	import { EmptyState, Input } from '$lib/ds';

	let historyEntries: BenchmarkV2HistoryEntry[] = [];
	let historyLoading = false;
	let filterWords = '';

	$: needle = filterWords.trim().toLowerCase();
	$: shownEntries = needle
		? historyEntries.filter((entry) =>
				`${entry.run_id} ${entry.profile} ${entry.models.join(' ')}`.toLowerCase().includes(needle)
			)
		: historyEntries;

	function runHref(entry: BenchmarkV2HistoryEntry): string {
		return `/workshop/benchmarks?tab=history&run=${encodeURIComponent(entry.run_id)}`;
	}

	onMount(loadHistory);

	async function loadHistory() {
		historyLoading = true;
		try {
			const data = await getHistory(50);
			historyEntries = data.runs;
		} catch {
			// silent
		} finally {
			historyLoading = false;
		}
	}
</script>

		<div class="bv2-section">
			{#if historyLoading}
				<p class="bv2-hint">Loading history...</p>
			{:else if historyEntries.length === 0}
				<EmptyState
					size="sm"
					icon="history"
					title="No benchmark runs recorded yet"
					description="Completed runs from the Run tab will appear here."
				/>
			{:else}
				<div class="bv2-history-filter">
					<Input label="Filter runs" hideLabel placeholder="Filter runs by id, profile or model" iconLeft="search" bind:value={filterWords} />
				</div>
				{#if shownEntries.length === 0}
					<p class="bv2-hint">No run matches "{filterWords.trim()}".</p>
				{/if}
				<div class="bv2-history-list">
					{#each shownEntries as entry (entry.run_id)}
						<div class="bv2-history-card">
							<div class="bv2-history-header">
								<a class="bv2-history-profile bv2-history-open" href={runHref(entry)}>
									{entry.profile}<span class="oo-sr-only">: open this run's scores</span>
								</a>
								<span class="bv2-history-status" class:completed={entry.status === 'completed'} class:failed={entry.status === 'failed'}>
									{entry.status}
								</span>
								<span class="bv2-history-date">{formatDate(entry.started_at)}</span>
								<span class="bv2-history-duration">{formatDuration(entry.duration_ms)}</span>
							</div>
							<div class="bv2-history-models">
								{#each entry.models as model}
									{@const ms = entry.model_scores[model]}
									<div class="bv2-history-model-row">
										<span class="bv2-model-name">{model}</span>
										{#if ms}
											<span style="color: {scoreColor(ms.composite)}">{pct(ms.composite)}</span>
										{/if}
									</div>
								{/each}
							</div>
						</div>
					{/each}
				</div>
			{/if}
		</div>

<style>
	.bv2-history-filter {
		max-width: 22rem;
		margin-bottom: var(--oo-space-3);
	}
	.bv2-history-open {
		color: var(--oo-fg-primary);
		text-decoration: underline;
		text-underline-offset: 2px;
	}
	.bv2-history-open:hover {
		color: var(--oo-acc-ink);
	}
</style>
