<!--
  BenchmarkRunDrawer.svelte
  Per-run detail surface (spec 9.5): a drawer-right that shows a single run's
  summary and per-model scores. Sourced from the existing benchmark history
  (getHistory) so no new endpoint is introduced.
-->
<script lang="ts">
	import { Modal } from '$lib/ds';
	import type { BenchmarkV2HistoryEntry } from '$lib/types';
	import { getHistory } from '$lib/api/benchmarkV2';
	import { scoreColor, pct, formatDuration, formatDate } from './format';

	export let runId = '';
	export let open = false;
	export let onClose: () => void;

	let loading = false;
	let run: BenchmarkV2HistoryEntry | null = null;
	let notFound = false;
	let lastLoaded = '';

	// Load the run from history whenever the drawer opens for a new id.
	$: if (open && runId && runId !== lastLoaded) {
		lastLoaded = runId;
		loadRun(runId);
	}

	async function loadRun(id: string) {
		loading = true;
		run = null;
		notFound = false;
		try {
			const data = await getHistory(50);
			run = data.runs.find((r) => r.run_id === id) ?? null;
			notFound = run === null;
		} catch {
			notFound = true;
		} finally {
			loading = false;
		}
	}

	$: modelScores = run ? Object.entries(run.model_scores) : [];
</script>

<Modal {open} variant="drawer-right" size="lg" title="Run detail" {onClose}>
	{#if loading}
		<p class="text-sm" style="color: var(--oo-fg-muted);">Loading run...</p>
	{:else if notFound || !run}
		<p class="text-sm" style="color: var(--oo-fg-muted);">Run not found in recent history.</p>
	{:else}
		<div class="flex flex-col gap-4">
			<div>
				<div class="text-xs font-mono" style="color: var(--oo-fg-faint);">{run.run_id}</div>
				<div class="flex flex-wrap items-center gap-3 mt-1 text-xs" style="color: var(--oo-fg-muted);">
					<span>Profile: <span style="color: var(--oo-fg-secondary);">{run.profile}</span></span>
					<span>Status: <span style="color: var(--oo-fg-secondary);">{run.status}</span></span>
					<span>{formatDate(run.started_at)}</span>
					<span>{formatDuration(run.duration_ms)}</span>
				</div>
			</div>

			<div>
				<h3 class="text-sm font-medium mb-2" style="color: var(--oo-fg-secondary);">Model scores</h3>
				{#if modelScores.length === 0}
					<p class="text-xs" style="color: var(--oo-fg-faint);">No scores recorded for this run.</p>
				{:else}
					<div class="flex flex-col gap-2">
						{#each modelScores as [model, ms]}
							<div class="rounded p-3" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);">
								<div class="text-sm font-medium mb-2" style="color: var(--oo-fg-primary);">{model}</div>
								<div class="grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs">
									<div>
										<span style="color: var(--oo-fg-faint);">Accuracy</span>
										<div style="color: {scoreColor(ms.accuracy_avg)};">{pct(ms.accuracy_avg)}</div>
									</div>
									<div>
										<span style="color: var(--oo-fg-faint);">Code</span>
										<div style="color: {scoreColor(ms.code_avg)};">{pct(ms.code_avg)}</div>
									</div>
									<div>
										<span style="color: var(--oo-fg-faint);">Structure</span>
										<div style="color: {scoreColor(ms.structure_avg)};">{pct(ms.structure_avg)}</div>
									</div>
									<div>
										<span style="color: var(--oo-fg-faint);">Speed</span>
										<div style="color: {scoreColor(ms.speed_avg)};">{pct(ms.speed_avg)}</div>
									</div>
									<div>
										<span style="color: var(--oo-fg-faint);">Composite</span>
										<div class="font-semibold" style="color: {scoreColor(ms.composite)};">{pct(ms.composite)}</div>
									</div>
								</div>
							</div>
						{/each}
					</div>
				{/if}
			</div>
		</div>
	{/if}
</Modal>
