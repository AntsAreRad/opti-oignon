<!--
  BenchmarkHistory.svelte (S60)
  Browse past benchmark runs, view detailed results, and compare.
  Shows: run list, detail panel with results table + ranking, delete.
-->
<script lang="ts">
	import {
		runs,
		runsTotal,
		selectedRun,
		benchmarkLoading,
		loadRuns,
		loadRunDetail,
		removeRun,
	} from '$lib/stores/benchmark';
	import type { BenchmarkResultItem } from '$lib/types';
	import { Button } from '$lib/ds';

	let confirmDeleteId: string | null = null;

	function selectRun(runId: string) {
		loadRunDetail(runId);
	}

	function goBack() {
		selectedRun.set(null);
	}

	function handleDelete(runId: string) {
		if (confirmDeleteId === runId) {
			removeRun(runId);
			confirmDeleteId = null;
		} else {
			confirmDeleteId = runId;
			setTimeout(() => { confirmDeleteId = null; }, 3000);
		}
	}

	function formatDate(iso: string): string {
		if (!iso) return '—';
		try {
			const d = new Date(iso);
			return d.toLocaleDateString(undefined, {
				month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
			});
		} catch {
			return iso;
		}
	}

	function formatDuration(sec: number | null): string {
		if (sec === null || sec === undefined) return '—';
		if (sec < 60) return `${Math.round(sec)}s`;
		const m = Math.floor(sec / 60);
		const s = Math.round(sec % 60);
		return `${m}m ${s}s`;
	}

	function scoreColor(score: number): string {
		if (score >= 8) return 'var(--oo-success)';
		if (score >= 5) return 'var(--oo-acc-400)';
		if (score >= 3) return 'var(--oo-acc-600)';
		return 'var(--oo-error)';
	}

	function statusIcon(status: string): string {
		switch (status) {
			case 'completed': return '✓';
			case 'cancelled': return '✗';
			case 'error': return '!';
			case 'running': return '●';
			default: return '?';
		}
	}

	function statusCls(status: string): string {
		switch (status) {
			case 'completed': return 'st-ok';
			case 'cancelled': return 'st-warn';
			case 'error': return 'st-err';
			case 'running': return 'st-run';
			default: return 'st-neutral';
		}
	}
</script>

<div class="history">
	{#if $selectedRun}
		<!-- DETAIL VIEW -->
		<div class="detail">
			<Button variant="ghost" size="sm" iconLeft="arrow-left" on:click={goBack}>Back to list</Button>

			<div class="detail-header">
				<div class="detail-meta">
					<h3>Run {$selectedRun.id.slice(0, 8)}</h3>
					<span class="detail-status {statusCls($selectedRun.status)}">
						{@html statusIcon($selectedRun.status)} {$selectedRun.status}
					</span>
				</div>
				<div class="detail-stats">
					<div class="stat">
						<span class="stat-val">{$selectedRun.models.length}</span>
						<span class="stat-lbl">Models</span>
					</div>
					<div class="stat">
						<span class="stat-val">{$selectedRun.results.length}</span>
						<span class="stat-lbl">Tests</span>
					</div>
					<div class="stat">
						<span class="stat-val" style="color: {$selectedRun.avg_score ? scoreColor($selectedRun.avg_score) : 'inherit'}">
							{$selectedRun.avg_score?.toFixed(1) ?? '—'}
						</span>
						<span class="stat-lbl">Avg Score</span>
					</div>
					<div class="stat">
						<span class="stat-val">{formatDuration($selectedRun.duration_sec)}</span>
						<span class="stat-lbl">Duration</span>
					</div>
				</div>
			</div>

			<!-- Ranking -->
			{#if $selectedRun.global_ranking.length > 0}
				<div class="ranking-section">
					<h4 class="sub-title">Model Ranking</h4>
					<div class="ranking-list">
						{#each $selectedRun.global_ranking as entry}
							<div class="rank-row">
								<span class="rank-pos">#{entry.rank}</span>
								<span class="rank-model">{entry.model}</span>
								<span class="rank-score" style="color: {scoreColor(entry.avg_score)}">
									{entry.avg_score.toFixed(1)}
								</span>
								<span class="rank-time">{entry.avg_time.toFixed(1)}s avg</span>
								<span class="rank-tests">{entry.tests} tests</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}

			<!-- Best by category -->
			{#if Object.keys($selectedRun.best_by_category).length > 0}
				<div class="best-section">
					<h4 class="sub-title">Best by Category</h4>
					<div class="best-grid">
						{#each Object.entries($selectedRun.best_by_category) as [cat, model]}
							<div class="best-item">
								<span class="best-cat">{cat}</span>
								<span class="best-model">{model}</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}

			<!-- Results table -->
			{#if $selectedRun.results.length > 0}
				<div class="detail-results">
					<h4 class="sub-title">All Results</h4>
					<div class="table-wrapper">
						<table class="results-table">
							<thead>
								<tr>
									<th>Model</th>
									<th>Task</th>
									<th>Score</th>
									<th>Auto</th>
									<th>User</th>
									<th>Time</th>
									<th>Status</th>
								</tr>
							</thead>
							<tbody>
								{#each $selectedRun.results as r}
									<tr>
										<td class="cell-model">{r.model}</td>
										<td>{r.task_name || r.task}</td>
										<td style="color: {scoreColor(r.score)}; font-weight: 600;">
											{r.score.toFixed(1)}
										</td>
										<td class="cell-dim">{r.auto_score.toFixed(1)}</td>
										<td class="cell-dim">{r.user_score?.toFixed(1) ?? '—'}</td>
										<td class="cell-mono">{r.time_seconds.toFixed(1)}s</td>
										<td>
											<span class="mini-badge {statusCls(r.status)}">{r.status}</span>
										</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				</div>
			{/if}
		</div>
	{:else}
		<!-- RUN LIST -->
		<div class="list-section">
			<div class="list-header">
				<h3 class="section-title">Benchmark History</h3>
				<span class="run-count">{$runsTotal} run{$runsTotal !== 1 ? 's' : ''}</span>
			</div>

			{#if $benchmarkLoading}
				<p class="loading-msg">Loading...</p>
			{:else if $runs.length === 0}
				<div class="empty-state">
					<p>No benchmark runs yet.</p>
					<p class="empty-hint">Start a benchmark from the Run tab to see results here.</p>
				</div>
			{:else}
				<div class="run-list">
					{#each $runs as run}
						<div class="run-card" role="button" tabindex="0" on:click={() => selectRun(run.id)} on:keydown={(e) => e.key === 'Enter' && selectRun(run.id)}>
							<div class="run-card-top">
								<div class="run-card-info">
									<span class="run-id">{run.id.slice(0, 8)}</span>
									<span class="run-status {statusCls(run.status)}">{@html statusIcon(run.status)}</span>
								</div>
								<span class="run-date">{formatDate(run.started_at)}</span>
							</div>
							<div class="run-card-stats">
								<span class="mini-stat">
									<span class="mini-val" style="color: {run.avg_score ? scoreColor(run.avg_score) : 'inherit'}">
										{run.avg_score?.toFixed(1) ?? '—'}
									</span> avg
								</span>
								<span class="mini-stat">{run.total_tests} tests</span>
								<span class="mini-stat">{run.models_tested?.length ?? 0} models</span>
								{#if run.best_model}
									<span class="mini-stat best">&#9733; {run.best_model}</span>
								{/if}
							</div>
							<div class="run-card-actions">
								<button
									class="delete-btn"
									class:confirm={confirmDeleteId === run.id}
									on:click|stopPropagation={() => handleDelete(run.id)}
								>
									{confirmDeleteId === run.id ? 'Confirm?' : 'Delete'}
								</button>
							</div>
						</div>
					{/each}
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.history {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.section-title {
		font-size: 0.9rem;
		font-weight: 600;
		margin: 0;
		color: var(--oo-fg-primary);
	}

	.sub-title {
		font-size: 0.8rem;
		font-weight: 600;
		margin: 0 0 0.5rem 0;
		color: var(--oo-fg-secondary);
	}

	/* -- List -- */

	.list-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 0.75rem;
	}

	.run-count {
		font-size: 0.72rem;
		color: var(--oo-fg-muted);
	}

	.loading-msg {
		font-size: 0.8rem;
		color: var(--oo-fg-muted);
		font-style: italic;
	}

	.empty-state {
		text-align: center;
		padding: 2rem 1rem;
		color: var(--oo-fg-tertiary);
		font-size: 0.85rem;
	}

	.empty-hint {
		font-size: 0.75rem;
		color: var(--oo-fg-muted);
	}

	.run-list {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.run-card {
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-default);
		border-radius: 6px;
		padding: 0.75rem 1rem;
		cursor: pointer;
		transition: border-color 0.15s, background 0.15s;
	}

	.run-card:hover {
		border-color: var(--oo-bd-strong);
		background: var(--oo-bg-elevated);
	}

	.run-card-top {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.375rem;
	}

	.run-card-info {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.run-id {
		font-family: monospace;
		font-size: 0.78rem;
		color: var(--oo-fg-primary);
	}

	.run-status {
		font-size: 0.72rem;
	}

	.run-date {
		font-size: 0.68rem;
		color: var(--oo-fg-muted);
	}

	.run-card-stats {
		display: flex;
		gap: 1rem;
		font-size: 0.72rem;
		color: var(--oo-fg-tertiary);
		margin-bottom: 0.25rem;
	}

	.mini-val {
		font-weight: 600;
	}

	.mini-stat.best {
		color: var(--oo-acc-400);
	}

	.run-card-actions {
		display: flex;
		justify-content: flex-end;
	}

	.delete-btn {
		background: none;
		border: none;
		color: var(--oo-fg-muted);
		font-size: 0.68rem;
		cursor: pointer;
		padding: 0.125rem 0.375rem;
	}

	.delete-btn:hover {
		color: var(--oo-error);
	}

	.delete-btn.confirm {
		color: var(--oo-error);
		font-weight: 600;
	}

	/* Status colors */
	.st-ok { color: var(--oo-success); }
	.st-warn { color: var(--oo-acc-400); }
	.st-err { color: var(--oo-error); }
	.st-run { color: var(--oo-acc-400); }
	.st-neutral { color: var(--oo-fg-muted); }

	/* -- Detail -- */

	.detail-header {
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-default);
		border-radius: 8px;
		padding: 1rem 1.25rem;
		margin-bottom: 1rem;
	}

	.detail-meta {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		margin-bottom: 0.75rem;
	}

	.detail-meta h3 {
		font-size: 1rem;
		font-weight: 600;
		margin: 0;
		font-family: monospace;
	}

	.detail-status {
		font-size: 0.72rem;
		font-weight: 500;
	}

	.detail-stats {
		display: flex;
		gap: 1.5rem;
	}

	.stat {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.125rem;
	}

	.stat-val {
		font-size: 1.1rem;
		font-weight: 700;
		color: var(--oo-fg-primary);
	}

	.stat-lbl {
		font-size: 0.62rem;
		color: var(--oo-fg-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	/* -- Ranking -- */

	.ranking-section, .best-section, .detail-results {
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-default);
		border-radius: 8px;
		padding: 1rem 1.25rem;
		margin-bottom: 1rem;
	}

	.ranking-list {
		display: flex;
		flex-direction: column;
		gap: 0.375rem;
	}

	.rank-row {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.375rem 0.5rem;
		background: var(--oo-bg-elevated);
		border-radius: 4px;
		font-size: 0.78rem;
	}

	.rank-pos {
		font-weight: 700;
		color: var(--oo-acc-400);
		min-width: 2rem;
	}

	.rank-model {
		flex: 1;
		font-family: monospace;
		font-size: 0.72rem;
		color: var(--oo-acc-400);
	}

	.rank-score {
		font-weight: 700;
	}

	.rank-time, .rank-tests {
		font-size: 0.68rem;
		color: var(--oo-fg-muted);
	}

	/* -- Best by category -- */

	.best-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
	}

	.best-item {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		padding: 0.3rem 0.625rem;
		background: var(--oo-bg-elevated);
		border-radius: 4px;
		font-size: 0.72rem;
	}

	.best-cat {
		color: var(--oo-fg-tertiary);
		text-transform: capitalize;
	}

	.best-model {
		color: var(--oo-acc-400);
		font-family: monospace;
	}

	/* -- Results table -- */

	.table-wrapper {
		overflow-x: auto;
	}

	.results-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.75rem;
	}

	.results-table th {
		text-align: left;
		padding: 0.5rem 0.625rem;
		color: var(--oo-fg-tertiary);
		font-weight: 500;
		border-bottom: 1px solid var(--oo-bd-default);
	}

	.results-table td {
		padding: 0.4rem 0.625rem;
		border-bottom: 1px solid var(--oo-bd-subtle);
		color: var(--oo-fg-secondary);
	}

	.results-table tbody tr:hover {
		background: var(--oo-bg-elevated);
	}

	.cell-model {
		font-family: monospace;
		font-size: 0.72rem;
		color: var(--oo-acc-400);
	}

	.cell-dim {
		color: var(--oo-fg-muted);
	}

	.cell-mono {
		font-family: monospace;
		font-size: 0.72rem;
	}

	.mini-badge {
		font-size: 0.62rem;
		font-weight: 500;
		text-transform: uppercase;
	}
</style>
