<!--
  BenchmarkRunner.svelte
  Configure and execute LLM benchmarks with live progress tracking.
  Shows: model/suite selector, progress bar, live results table.
-->
<script lang="ts">
	import {
		suites,
		installedModels,
		isRunning,
		progress,
		liveResults,
		runError,
		benchmarkError,
		launchBenchmark,
		stopBenchmark,
	} from '$lib/stores/benchmark';
	import type { BenchmarkResultItem } from '$lib/types';
	import OnionLoader from '$lib/components/OnionLoader.svelte';
	import { Button } from '$lib/ds';

	// -- Config form state --
	let selectedModels: string[] = [];
	let selectedSuite = '';
	let temperature = 0.7;
	let timeout = 300;
	let maxTokens = 1000;
	let showAdvanced = false;

	// -- Model selection --
	function toggleModel(model: string) {
		if (selectedModels.includes(model)) {
			selectedModels = selectedModels.filter((m) => m !== model);
		} else {
			selectedModels = [...selectedModels, model];
		}
	}

	function selectAllModels() {
		selectedModels = [...$installedModels];
	}

	function clearModels() {
		selectedModels = [];
	}

	// -- Launch --
	async function handleStart() {
		const models = selectedModels.length > 0 ? selectedModels : [...$installedModels];
		await launchBenchmark({
			models,
			suiteId: selectedSuite || undefined,
			temperature,
			timeout,
			maxTokens,
		});
	}

	// -- Helpers --
	function scoreColor(score: number): string {
		if (score >= 8) return 'var(--oo-success)';
		if (score >= 5) return 'var(--oo-acc-400)';
		if (score >= 3) return 'var(--oo-acc-600)';
		return 'var(--oo-error)';
	}

	function statusBadge(status: string): { label: string; cls: string } {
		switch (status) {
			case 'success': return { label: 'OK', cls: 'badge-success' };
			case 'error': return { label: 'ERR', cls: 'badge-error' };
			case 'timeout': return { label: 'TIMEOUT', cls: 'badge-error' };
			case 'refused': return { label: 'REFUSED', cls: 'badge-warn' };
			default: return { label: status, cls: 'badge-neutral' };
		}
	}

	function formatTime(sec: number): string {
		if (sec < 60) return `${sec}s`;
		const m = Math.floor(sec / 60);
		const s = sec % 60;
		return `${m}m ${s}s`;
	}
</script>

<div class="runner">
	{#if !$isRunning}
		<!-- CONFIG FORM -->
		<div class="config-section">
			<h3 class="section-title">Configuration</h3>

			<!-- Suite selector -->
			<div class="field">
				<label class="field-label" for="suite-select">Benchmark Suite</label>
				<select id="suite-select" class="field-select" bind:value={selectedSuite}>
					<option value="">All tasks (custom)</option>
					{#each $suites as suite}
						<option value={suite.id}>
							{suite.name} ({suite.task_count} tasks)
						</option>
					{/each}
				</select>
			</div>

			<!-- Model selector -->
			<div class="field">
				<div class="field-label-row">
					<label class="field-label">Models</label>
					<div class="field-actions">
						<Button variant="link" size="sm" on:click={selectAllModels}>Select all</Button>
						<Button variant="link" size="sm" on:click={clearModels}>Clear</Button>
					</div>
				</div>
				{#if $installedModels.length === 0}
					<p class="empty-hint">No Ollama models detected. Ensure Ollama is running.</p>
				{:else}
					<div class="model-grid">
						{#each $installedModels as model}
							<button
								class="model-chip"
								class:selected={selectedModels.includes(model)}
								on:click={() => toggleModel(model)}
							>
								<span class="chip-check">{selectedModels.includes(model) ? '✓' : ''}</span>
								{model}
							</button>
						{/each}
					</div>
					{#if selectedModels.length === 0}
						<p class="empty-hint">No models selected — all installed models will be tested.</p>
					{/if}
				{/if}
			</div>

			<!-- Advanced options -->
			<button class="toggle-advanced" on:click={() => (showAdvanced = !showAdvanced)} aria-expanded={showAdvanced}>
				{showAdvanced ? '▼' : '▶'} Advanced options
			</button>
			{#if showAdvanced}
				<div class="advanced-grid">
					<div class="field-inline">
						<label class="field-label" for="temp">Temperature</label>
						<input id="temp" type="number" class="field-input" bind:value={temperature} min="0" max="2" step="0.1" />
					</div>
					<div class="field-inline">
						<label class="field-label" for="to">Timeout (s)</label>
						<input id="to" type="number" class="field-input" bind:value={timeout} min="30" max="1800" step="30" />
					</div>
					<div class="field-inline">
						<label class="field-label" for="mt">Max tokens</label>
						<input id="mt" type="number" class="field-input" bind:value={maxTokens} min="100" max="4096" step="100" />
					</div>
				</div>
			{/if}

			<!-- Start button -->
			<Button variant="primary" iconLeft="play" on:click={handleStart}>Start Benchmark</Button>
		</div>
	{:else}
		<!-- LIVE PROGRESS -->
		<div class="progress-section">
			<div class="progress-header">
				<div style="display: flex; align-items: center; gap: 8px;">
					<OnionLoader size={22} color="var(--oo-acc-400)" />
					<h3 class="section-title">Benchmark Running</h3>
				</div>
				<Button variant="secondary" size="sm" on:click={stopBenchmark}>Cancel</Button>
			</div>

			{#if $progress}
				<div class="progress-info">
					<div class="progress-bar-wrapper">
						<div class="progress-bar" style="width: {$progress.percent}%"></div>
					</div>
					<div class="progress-stats">
						<span>{$progress.completed_tests} / {$progress.total_tests} tests</span>
						<span class="progress-pct">{$progress.percent}%</span>
					</div>
					<div class="progress-detail">
						<span class="detail-label">Current:</span>
						<span class="detail-model">{$progress.current_model}</span>
						<span class="detail-sep">&rarr;</span>
						<span>{$progress.current_task_name}</span>
					</div>
					<div class="progress-times">
						<span>Elapsed: {formatTime($progress.elapsed_sec)}</span>
						<span>Remaining: ~{formatTime($progress.estimated_remaining_sec)}</span>
					</div>
				</div>
			{:else}
				<p class="waiting-msg">Waiting for first result...</p>
			{/if}
		</div>
	{/if}

	{#if $runError}
		<div class="run-error" role="alert">
			<span class="error-icon">!</span> {$runError}
		</div>
	{/if}

	<!-- LIVE RESULTS TABLE -->
	{#if $liveResults.length > 0}
		<div class="results-section">
			<h3 class="section-title">Results ({$liveResults.length})</h3>
			<div class="results-table-wrapper">
				<table class="results-table">
					<thead>
						<tr>
							<th>Model</th>
							<th>Task</th>
							<th>Score</th>
							<th>Time</th>
							<th>Status</th>
							<th>Keywords</th>
						</tr>
					</thead>
					<tbody>
						{#each $liveResults as result}
							{@const badge = statusBadge(result.status)}
							<tr>
								<td class="cell-model">{result.model}</td>
								<td class="cell-task">{result.task_name || result.task}</td>
								<td>
									<span class="score-pill" style="color: {scoreColor(result.score)}">
										{result.score.toFixed(1)}
									</span>
								</td>
								<td class="cell-time">{result.time_seconds.toFixed(1)}s</td>
								<td><span class="status-badge {badge.cls}">{badge.label}</span></td>
								<td class="cell-kw">
									{#if result.keywords_found.length > 0 || result.keywords_missing.length > 0}
										<span class="kw-found">{result.keywords_found.length}</span>
										/
										<span class="kw-total">{result.keywords_found.length + result.keywords_missing.length}</span>
									{:else}
										<span class="kw-na">—</span>
									{/if}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>
	{/if}
</div>

<style>
	.runner {
		display: flex;
		flex-direction: column;
		gap: 1.25rem;
	}

	.section-title {
		font-size: 0.9rem;
		font-weight: 600;
		margin: 0 0 0.75rem 0;
		color: var(--oo-fg-primary);
	}

	/* -- Config form -- */

	.config-section {
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-default);
		border-radius: 8px;
		padding: 1.25rem;
	}

	.field {
		margin-bottom: 1rem;
	}

	.field-label {
		display: block;
		font-size: 0.78rem;
		color: var(--oo-fg-secondary);
		margin-bottom: 0.375rem;
	}

	.field-label-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 0.375rem;
	}

	.field-actions {
		display: flex;
		gap: 0.5rem;
	}

	.field-select {
		width: 100%;
		padding: 0.5rem 0.625rem;
		background: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-default);
		border-radius: 5px;
		color: var(--oo-fg-primary);
		font-size: 0.8rem;
	}

	.model-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 0.375rem;
	}

	.model-chip {
		display: inline-flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.3rem 0.625rem;
		background: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-default);
		border-radius: 4px;
		color: var(--oo-fg-secondary);
		font-size: 0.72rem;
		cursor: pointer;
		transition: border-color 0.15s, background 0.15s;
		font-family: monospace;
	}

	.model-chip:hover {
		border-color: var(--oo-bd-strong);
	}

	.model-chip.selected {
		background: var(--oo-warning-bg);
		border-color: var(--oo-acc-400);
		color: var(--oo-acc-300);
	}

	.chip-check {
		font-size: 0.65rem;
		width: 0.75rem;
		text-align: center;
	}

	.empty-hint {
		font-size: 0.72rem;
		color: var(--oo-fg-muted);
		margin: 0.375rem 0 0 0;
		font-style: italic;
	}

	.toggle-advanced {
		background: none;
		border: none;
		color: var(--oo-fg-tertiary);
		font-size: 0.75rem;
		cursor: pointer;
		padding: 0.25rem 0;
		margin-bottom: 0.5rem;
	}

	.toggle-advanced:hover {
		color: var(--oo-fg-secondary);
	}

	.advanced-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
		gap: 0.75rem;
		margin-bottom: 1rem;
		padding: 0.75rem;
		background: var(--oo-bg-elevated);
		border-radius: 5px;
	}

	.field-inline {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.field-input {
		padding: 0.375rem 0.5rem;
		background: var(--oo-bg-base);
		border: 1px solid var(--oo-bd-default);
		border-radius: 4px;
		color: var(--oo-fg-primary);
		font-size: 0.78rem;
		width: 100%;
	}

	/* -- Progress -- */

	.progress-section {
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-default);
		border-radius: 8px;
		padding: 1.25rem;
	}

	.progress-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.progress-bar-wrapper {
		height: 6px;
		background: var(--oo-bg-elevated);
		border-radius: 3px;
		overflow: hidden;
		margin-bottom: 0.5rem;
	}

	.progress-bar {
		height: 100%;
		background: var(--oo-acc-400);
		border-radius: 3px;
		transition: width 0.3s ease;
	}

	.progress-stats {
		display: flex;
		justify-content: space-between;
		font-size: 0.78rem;
		color: var(--oo-fg-secondary);
		margin-bottom: 0.5rem;
	}

	.progress-pct {
		color: var(--oo-acc-400);
		font-weight: 600;
	}

	.progress-detail {
		font-size: 0.75rem;
		color: var(--oo-fg-tertiary);
		margin-bottom: 0.25rem;
	}

	.detail-label {
		color: var(--oo-fg-muted);
	}

	.detail-model {
		color: var(--oo-acc-400);
		font-family: monospace;
	}

	.detail-sep {
		margin: 0 0.25rem;
	}

	.progress-times {
		display: flex;
		gap: 1.5rem;
		font-size: 0.72rem;
		color: var(--oo-fg-muted);
	}

	.waiting-msg {
		font-size: 0.8rem;
		color: var(--oo-fg-muted);
		font-style: italic;
	}

	/* -- Error -- */

	.run-error {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.75rem;
		background: var(--oo-error-bg);
		border: 1px solid var(--oo-error-bd, rgba(239, 68, 68, 0.25));
		border-radius: 5px;
		color: var(--oo-error);
		font-size: 0.78rem;
	}

	.error-icon {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: var(--oo-error);
		color: var(--oo-fg-on-semantic);
		font-size: 0.6rem;
		font-weight: 700;
	}

	/* -- Results table -- */

	.results-section {
		background: var(--oo-bg-surface);
		border: 1px solid var(--oo-bd-default);
		border-radius: 8px;
		padding: 1.25rem;
	}

	.results-table-wrapper {
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
		white-space: nowrap;
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
		max-width: 200px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.cell-task {
		max-width: 160px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.score-pill {
		font-weight: 700;
		font-size: 0.78rem;
	}

	.cell-time {
		font-family: monospace;
		font-size: 0.72rem;
	}

	.status-badge {
		display: inline-block;
		padding: 0.125rem 0.375rem;
		border-radius: 3px;
		font-size: 0.62rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}

	.badge-success {
		background: var(--oo-success-bg);
		color: var(--oo-success);
	}

	.badge-error {
		background: var(--oo-error-bg);
		color: var(--oo-error);
	}

	.badge-warn {
		background: var(--oo-warning-bg);
		color: var(--oo-acc-400);
	}

	.badge-neutral {
		background: var(--oo-bg-overlay);
		color: var(--oo-fg-tertiary);
	}

	.kw-found {
		color: var(--oo-success);
	}

	.kw-total {
		color: var(--oo-fg-tertiary);
	}

	.kw-na {
		color: var(--oo-fg-muted);
	}
</style>
