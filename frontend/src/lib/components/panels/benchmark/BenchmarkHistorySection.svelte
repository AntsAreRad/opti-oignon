<!--
  BenchmarkHistorySection.svelte
  The "History" section extracted from BenchmarkV2Panel: list of past runs.
  Self-contained; behaviour unchanged.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import type { BenchmarkV2HistoryEntry } from '$lib/types';
	import { getHistory } from '$lib/api/benchmarkV2';
	import { scoreColor, pct, formatDuration, formatDate } from './format';
	import { EmptyState } from '$lib/ds';

	let historyEntries: BenchmarkV2HistoryEntry[] = [];
	let historyLoading = false;

	onMount(loadHistory);

	async function loadHistory() {
		historyLoading = true;
		try {
			const data = await getHistory(20);
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
				<div class="bv2-history-list">
					{#each historyEntries as entry}
						<div class="bv2-history-card">
							<div class="bv2-history-header">
								<span class="bv2-history-profile">{entry.profile}</span>
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
