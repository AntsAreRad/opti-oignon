<!--
  BenchmarkRunSection.svelte
  The "Run" section extracted from BenchmarkV2Panel: profile + model
  selection, LLM-as-Judge toggle, run execution with live progress, the
  results table + radar comparison + export, and the auto-trigger controls
  as well. Self-contained; markup and API calls are unchanged.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import type {
		BenchmarkV2Profile,
		BenchmarkV2Progress,
		BenchmarkV2Results,
		BenchmarkV2CustomProfile,
		BenchmarkV2AutoTriggerStatus,
		BenchmarkV2AutoTriggerEvent,
		BenchmarkV2AutoTriggerTestPollResponse,
	} from '$lib/types';
	import {
		getProfiles,
		startRun,
		pollUntilDone,
		getCustomProfiles,
		exportJson,
		exportCsv,
		downloadBlob,
		getAutoTriggerStatus,
		getAutoTriggerEvents,
		enableAutoTrigger,
		disableAutoTrigger,
		updateAutoTriggerConfig,
		testPollAutoTrigger,
	} from '$lib/api/benchmarkV2';
	import { scoreColor, pct, formatDuration, radarPoints, radarLabels, radarColors, radarLabelPos, formatCooldown, formatEventTime } from './format';

	let profiles: BenchmarkV2Profile[] = [];
	let selectedProfile = '';
	let availableModels: string[] = [];
	let selectedModels: string[] = [];
	let customProfiles: BenchmarkV2CustomProfile[] = [];

	let useJudge = false;
	let judgeModel = '';

	let running = false;
	let runError = '';
	let progress: BenchmarkV2Progress | null = null;
	let results: BenchmarkV2Results | null = null;

	// Auto-Trigger
	let autoTriggerStatus: BenchmarkV2AutoTriggerStatus | null = null;
	let autoTriggerToggling = false;
	let autoTriggerEvents: BenchmarkV2AutoTriggerEvent[] = [];
	let eventLogOpen = false;
	let eventsLoading = false;
	let cooldownDisplay = 0;
	let cooldownInterval: ReturnType<typeof setInterval> | null = null;
	let testPollRunning = false;
	let testPollResult: BenchmarkV2AutoTriggerTestPollResponse | null = null;

	onMount(async () => {
		await loadProfiles();
		await loadModels();
		await loadCustomProfiles();
		await loadAutoTriggerStatus();
		startCooldownTimer();
	});

	onDestroy(() => {
		stopCooldownTimer();
	});

	async function loadProfiles() {
		try {
			const data = await getProfiles();
			profiles = data.profiles;
			if (profiles.length > 0 && !selectedProfile) {
				selectedProfile = profiles[0].id;
			}
		} catch (e) {
			runError = `Failed to load profiles: ${e}`;
		}
	}

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

	async function loadCustomProfiles() {
		try {
			const data = await getCustomProfiles();
			customProfiles = data.profiles;
		} catch {
			// silent
		}
	}

	function toggleModel(model: string) {
		if (selectedModels.includes(model)) {
			selectedModels = selectedModels.filter((m) => m !== model);
		} else {
			selectedModels = [...selectedModels, model];
		}
	}

	async function handleStartRun() {
		if (!selectedProfile || selectedModels.length === 0) return;
		running = true;
		runError = '';
		results = null;
		progress = null;
		try {
			const started = await startRun({
				profile: selectedProfile,
				models: selectedModels,
				use_judge: useJudge,
				judge_model: useJudge ? judgeModel : undefined,
			});
			results = await pollUntilDone(started.run_id, (p) => {
				progress = p;
			});
			if (results.error) {
				runError = results.error;
			}
		} catch (e) {
			runError = `Run failed: ${e}`;
		} finally {
			running = false;
		}
	}

	async function handleExportJson() {
		if (!results?.run_id) return;
		const blob = await exportJson(results.run_id);
		downloadBlob(blob, `benchmark_${results.run_id}.json`);
	}

	async function handleExportCsv() {
		if (!results?.run_id) return;
		const blob = await exportCsv(results.run_id);
		downloadBlob(blob, `benchmark_${results.run_id}.csv`);
	}

	async function loadAutoTriggerStatus() {
		try {
			autoTriggerStatus = await getAutoTriggerStatus();
			cooldownDisplay = autoTriggerStatus?.cooldown_remaining ?? 0;
		} catch {
			// Auto-trigger may not be available
		}
	}

	async function handleAutoTriggerToggle() {
		autoTriggerToggling = true;
		try {
			if (autoTriggerStatus?.enabled) {
				await disableAutoTrigger();
			} else {
				await enableAutoTrigger();
			}
			await loadAutoTriggerStatus();
		} catch {
			// silent
		} finally {
			autoTriggerToggling = false;
		}
	}

	async function loadAutoTriggerEvents() {
		eventsLoading = true;
		try {
			const data = await getAutoTriggerEvents();
			autoTriggerEvents = data.events.slice(-10);
		} catch {
			// silent
		} finally {
			eventsLoading = false;
		}
	}

	function toggleEventLog() {
		eventLogOpen = !eventLogOpen;
		if (eventLogOpen && autoTriggerEvents.length === 0) {
			loadAutoTriggerEvents();
		}
	}

	async function handleTestPoll() {
		testPollRunning = true;
		testPollResult = null;
		try {
			testPollResult = await testPollAutoTrigger();
		} catch (e) {
			testPollResult = {
				ok: false,
				error: String(e),
				snapshot_models: 0,
				model_names: [],
				diff: null,
			};
		} finally {
			testPollRunning = false;
		}
	}

	async function handleAutoTriggerProfileChange(e: Event) {
		const target = e.target as HTMLSelectElement;
		const newProfile = target.value;
		try {
			await updateAutoTriggerConfig({ trigger_profile: newProfile });
			await loadAutoTriggerStatus();
		} catch {
			// silent
		}
	}

	function startCooldownTimer() {
		stopCooldownTimer();
		cooldownInterval = setInterval(() => {
			if (cooldownDisplay > 0) {
				cooldownDisplay = Math.max(0, cooldownDisplay - 1);
			}
		}, 1000);
	}

	function stopCooldownTimer() {
		if (cooldownInterval !== null) {
			clearInterval(cooldownInterval);
			cooldownInterval = null;
		}
	}

	$: currentProfile = profiles.find((p) => p.id === selectedProfile);
	$: progressPct = progress && progress.total_questions > 0
		? Math.round((progress.completed_questions / progress.total_questions) * 100)
		: 0;
</script>

		<div class="bv2-section">
			<!-- Profile selector -->
			<div class="bv2-field">
				<label class="bv2-label" for="profile-select">Profile</label>
				<select id="profile-select" class="bv2-select" bind:value={selectedProfile}>
					{#each profiles as p}
						<option value={p.id}>{p.name}</option>
					{/each}
				</select>
				{#if currentProfile}
					<p class="bv2-hint">{currentProfile.description}</p>
					<div class="bv2-tags">
						{#each currentProfile.categories as cat}
							<span class="bv2-tag">{cat}</span>
						{/each}
						<span class="bv2-tag accent">weights: {currentProfile.weight_preset}</span>
					</div>
				{/if}
			</div>

			<!-- Model multi-select -->
			<div class="bv2-field">
				<label class="bv2-label">Models ({selectedModels.length} selected)</label>
				<div class="bv2-model-grid">
					{#each availableModels as model}
						<button
							class="bv2-model-chip"
							class:selected={selectedModels.includes(model)}
							on:click={() => toggleModel(model)}
						>
							{model}
						</button>
					{/each}
					{#if availableModels.length === 0}
						<p class="bv2-hint">No models found. Is Ollama running?</p>
					{/if}
				</div>
			</div>

			<!-- Judge toggle -->
			<div class="bv2-field bv2-judge-field">
				<label class="bv2-toggle-row">
					<input type="checkbox" bind:checked={useJudge} />
					<span class="bv2-label-inline">Enable LLM-as-Judge evaluation</span>
				</label>
				{#if useJudge}
					<div class="bv2-judge-select">
						<label class="bv2-label" for="judge-model-select">Judge model (strongest recommended)</label>
						<select id="judge-model-select" class="bv2-select" bind:value={judgeModel}>
							<option value="" disabled>Select judge model</option>
							{#each availableModels as model}
								<option value={model}>{model}</option>
							{/each}
						</select>
					</div>
				{/if}
			</div>

			<!-- Auto-Trigger toggle -->
			{#if autoTriggerStatus !== null}
				<div class="bv2-field bv2-autotrigger-field">
					<label class="bv2-toggle-row">
						<input
							type="checkbox"
							checked={autoTriggerStatus.enabled}
							disabled={autoTriggerToggling}
							on:change={handleAutoTriggerToggle}
						/>
						<span class="bv2-label-inline">Auto-trigger on new models</span>
					</label>
					<p class="bv2-warning-text">
						Benchmarks will run automatically when new models are detected. This uses significant GPU/RAM resources.
					</p>
					{#if autoTriggerStatus.enabled}
						<div class="bv2-autotrigger-info">
							<span class="bv2-badge bv2-badge-active">Active</span>
							<span class="bv2-autotrigger-detail">{autoTriggerStatus.known_models} models tracked</span>
							{#if autoTriggerStatus.recent_events > 0}
								<span class="bv2-autotrigger-detail">{autoTriggerStatus.recent_events} recent events</span>
							{/if}
						</div>

						<!-- Profile selector for auto-triggered runs -->
						<div class="bv2-autotrigger-profile-row">
							<label class="bv2-label-sm" for="at-profile-select">Profile for auto-runs:</label>
							<select
								id="at-profile-select"
								class="bv2-select-sm"
								value={autoTriggerStatus.trigger_profile}
								on:change={handleAutoTriggerProfileChange}
							>
								{#each profiles as p}
									<option value={p.id}>{p.name}</option>
								{/each}
								{#each customProfiles as cp}
									<option value={cp.profile_id}>{cp.name}</option>
								{/each}
							</select>
						</div>

						<!-- Cooldown countdown -->
						{#if cooldownDisplay > 0}
							<div class="bv2-cooldown-bar">
								<span class="bv2-cooldown-icon">&#9716;</span>
								<span class="bv2-cooldown-text">Cooldown: {formatCooldown(cooldownDisplay)}</span>
							</div>
						{/if}

						<!-- Resource guard indicator -->
						{#if autoTriggerStatus.resource_guard_active}
							<div class="bv2-resource-guard">
								<span class="bv2-badge bv2-badge-guard">Resource Guard</span>
								<span class="bv2-autotrigger-detail">Max load: {autoTriggerStatus.resource_guard_load_max.toFixed(1)}</span>
							</div>
						{/if}
					{/if}

					<!-- Test Connection button -->
					<div class="bv2-autotrigger-actions">
						<button
							class="bv2-btn-sm"
							disabled={testPollRunning}
							on:click={handleTestPoll}
						>
							{testPollRunning ? 'Testing...' : 'Test Connection'}
						</button>
						{#if autoTriggerStatus.enabled && autoTriggerStatus.recent_events > 0}
							<button
								class="bv2-btn-sm"
								disabled={eventsLoading}
								on:click={toggleEventLog}
							>
								{eventLogOpen ? 'Hide Events' : 'Show Events'}
							</button>
						{/if}
					</div>

					<!-- Test poll result -->
					{#if testPollResult}
						<div class="bv2-test-poll-result" class:bv2-ok={testPollResult.ok} class:bv2-err={!testPollResult.ok}>
							{#if testPollResult.ok}
								<span>Connected - {testPollResult.snapshot_models} models found</span>
								{#if testPollResult.diff}
									{#if testPollResult.diff.has_changes}
										<span class="bv2-test-diff">Changes: +{testPollResult.diff.added.length} ~{testPollResult.diff.updated.length} -{testPollResult.diff.removed.length}</span>
									{:else}
										<span class="bv2-test-diff">No changes detected</span>
									{/if}
								{/if}
							{:else}
								<span>Connection failed: {testPollResult.error}</span>
							{/if}
						</div>
					{/if}

					<!-- Event log -->
					{#if eventLogOpen}
						<div class="bv2-event-log">
							<div class="bv2-event-log-header">
								<span class="bv2-label-sm">Recent Events</span>
								<button class="bv2-btn-xs" on:click={loadAutoTriggerEvents} disabled={eventsLoading}>
									{eventsLoading ? '...' : 'Refresh'}
								</button>
							</div>
							{#if autoTriggerEvents.length === 0}
								<p class="bv2-muted">No events recorded yet.</p>
							{:else}
								<div class="bv2-event-list">
									{#each autoTriggerEvents as evt}
										<div class="bv2-event-item" class:bv2-event-skipped={evt.skipped}>
											<div class="bv2-event-top">
												<span class="bv2-event-time">{formatEventTime(evt.timestamp)}</span>
												<span class="bv2-event-type">{evt.trigger_type}</span>
												{#if evt.skipped}
													<span class="bv2-badge bv2-badge-skip">Skipped</span>
												{:else if evt.run_id}
													<span class="bv2-badge bv2-badge-active">Run</span>
												{/if}
											</div>
											<div class="bv2-event-models">{evt.models.join(', ')}</div>
											{#if evt.skipped && evt.skip_reason}
												<div class="bv2-event-reason">{evt.skip_reason}</div>
											{/if}
											{#if evt.run_id}
												<div class="bv2-event-runid">Run: {evt.run_id}</div>
											{/if}
										</div>
									{/each}
								</div>
							{/if}
						</div>
					{/if}
				</div>
			{/if}

			<!-- Run button -->
			<button
				class="bv2-run-btn"
				disabled={running || selectedModels.length === 0 || !selectedProfile || (useJudge && !judgeModel)}
				on:click={handleStartRun}
			>
				{#if running}
					Running...
				{:else}
					Run Evaluation
				{/if}
			</button>

			<!-- Progress -->
			{#if running && progress}
				<div class="bv2-progress">
					<div class="bv2-progress-bar-bg">
						<div class="bv2-progress-bar-fill" style="width: {progressPct}%"></div>
					</div>
					<div class="bv2-progress-info">
						<span>{progress.completed_questions}/{progress.total_questions} questions</span>
						<span>{progress.current_model} / {progress.current_question}</span>
						<span>{formatDuration(progress.elapsed_ms)}</span>
					</div>
				</div>
			{/if}

			<!-- Error -->
			{#if runError}
				<div class="bv2-error">{runError}</div>
			{/if}

			<!-- Results table -->
			{#if results && results.status === 'completed'}
				<div class="bv2-results">
					<div class="bv2-results-header">
						<h3 class="bv2-subtitle">Results — {results.profile} ({formatDuration(results.duration_ms)})</h3>
						<div class="bv2-export-btns">
							<button class="bv2-btn-sm" on:click={handleExportJson}>Export JSON</button>
							<button class="bv2-btn-sm" on:click={handleExportCsv}>Export CSV</button>
						</div>
					</div>
					<div class="bv2-table-wrap">
						<table class="bv2-table">
							<thead>
								<tr>
									<th>Model</th>
									<th>Accuracy</th>
									<th>Code</th>
									<th>Structure</th>
									<th>Speed</th>
									<th>Composite</th>
									<th>Q</th>
								</tr>
							</thead>
							<tbody>
								{#each Object.entries(results.model_scores) as [model, ms]}
									<tr>
										<td class="bv2-model-name">{model}</td>
										<td style="color: {scoreColor(ms.accuracy_avg)}">{pct(ms.accuracy_avg)}</td>
										<td style="color: {scoreColor(ms.code_avg)}">{pct(ms.code_avg)}</td>
										<td style="color: {scoreColor(ms.structure_avg)}">{pct(ms.structure_avg)}</td>
										<td style="color: {scoreColor(ms.speed_avg)}">{pct(ms.speed_avg)}</td>
										<td class="bv2-composite" style="color: {scoreColor(ms.composite)}">{pct(ms.composite)}</td>
										<td>{ms.questions_evaluated}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>

					<!-- Judge summary -->
					{#if results.judge_scores && results.judge_scores.length > 0}
						<h3 class="bv2-subtitle">Judge Evaluation ({results.judge_scores.length} scores)</h3>
						<div class="bv2-judge-summary">
							{#if results.judge_summary?.models}
								{#each Object.entries(results.judge_summary.models) as [jmodel, jdata]}
									<div class="bv2-judge-card">
										<span class="bv2-model-name">{jmodel}</span>
										<span class="bv2-judge-score" style="color: {scoreColor(Number(jdata?.avg_score ?? 0))}">
											{pct(Number(jdata?.avg_score ?? 0))}
										</span>
										<span class="bv2-hint">{jdata?.evaluations ?? 0} evals, {jdata?.errors ?? 0} errors</span>
									</div>
								{/each}
							{/if}
						</div>
					{/if}

					<!-- Radar chart -->
					{#if Object.keys(results.model_scores).length > 0}
						<h3 class="bv2-subtitle">Radar Comparison</h3>
						<div class="bv2-radar-wrap">
							<svg viewBox="0 0 260 260" class="bv2-radar-svg">
								{#each [0.25, 0.5, 0.75, 1.0] as ring}
									<circle
										cx="130" cy="130"
										r={ring * 120}
										fill="none"
										stroke="var(--oo-bd-subtle)"
										stroke-width="0.5"
										stroke-dasharray="3,3"
									/>
								{/each}
								{#each [0, 1, 2, 3] as i}
									{@const angle = (Math.PI * 2 * i) / 4 - Math.PI / 2}
									<line
										x1="130" y1="130"
										x2={130 + 120 * Math.cos(angle)}
										y2={130 + 120 * Math.sin(angle)}
										stroke="var(--oo-bd-subtle)"
										stroke-width="0.5"
									/>
								{/each}
								{#each radarLabels as label, i}
									{@const pos = radarLabelPos(i, 120)}
									<text
										x={pos.x} y={pos.y}
										text-anchor="middle"
										dominant-baseline="middle"
										fill="var(--oo-fg-tertiary)"
										font-size="10"
									>{label}</text>
								{/each}
								{#each Object.entries(results.model_scores) as [model, ms], idx}
									<polygon
										points={radarPoints(ms, 130)}
										style="fill: {radarColors[idx % radarColors.length]}; fill-opacity: 0.15; stroke: {radarColors[idx % radarColors.length]}; stroke-width: 1.5;"
									/>
								{/each}
							</svg>
							<div class="bv2-radar-legend">
								{#each Object.keys(results.model_scores) as model, idx}
									<div class="bv2-legend-item">
										<span class="bv2-legend-dot" style="background: {radarColors[idx % radarColors.length]}"></span>
										{model}
									</div>
								{/each}
							</div>
						</div>
					{/if}
				</div>
			{/if}
		</div>
