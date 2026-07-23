<!--
  PerformanceTunerPanel.svelte -- Inference Auto-Tuner panel.

  Collapsible panel in Settings > Performance tab.
  Sections:
  1. Status overview and feature description
  2. Model input + "Run Tuner" button (estimated time ~2-5 min)
  3. Progress bar during tuning (polled via job endpoint)
  4. Results table: parameter, value, tokens/sec before/after
  5. "Apply" button to save optimal config
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		getTunerStatus,
		startTuning,
		listTunerResults,
		applyTunerResult,
		deleteTunerResult,
		cancelTuning,
		getTunerJob,
		getTunerRecommendations,
	} from '$lib/api/tuner';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type { TunerStatus, TunerJob, TunerProfile, TunerRecommendation } from '$lib/types';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	let loading = true;
	let error = '';
	let open = false;

	let available = false;
	let savedProfiles: string[] = [];
	let results: Record<string, TunerProfile> = {};

	// Tuning session
	let modelInput = '';
	let running = false;
	let currentJob: TunerJob | null = null;
	let pollTimer: ReturnType<typeof setInterval> | null = null;

	// Selected result to view
	let selectedModel = '';

	// Recommendations per model
	let recommendations: Record<string, TunerRecommendation[]> = {};
	let loadingRecs = '';

	// -------------------------------------------------------------------------
	// Lifecycle
	// -------------------------------------------------------------------------

	onMount(loadData);

	onDestroy(() => {
		if (pollTimer) clearInterval(pollTimer);
	});

	async function loadData() {
		loading = true;
		error = '';
		try {
			const [status, resultsResp] = await Promise.all([
				getTunerStatus(),
				listTunerResults(),
			]);
			available = status.available;
			savedProfiles = status.saved_profiles;
			results = resultsResp.results as Record<string, TunerProfile>;

			// Check for active jobs
			const jobKeys = Object.keys(status.active_jobs);
			if (jobKeys.length > 0) {
				const activeModel = jobKeys[0];
				const job = status.active_jobs[activeModel] as TunerJob;
				if (job.status === 'running' || job.status === 'pending') {
					currentJob = job;
					modelInput = activeModel;
					running = true;
					startPolling(activeModel);
				}
			}
		} catch (e) {
			error = `Failed to load tuner status: ${e}`;
		} finally {
			loading = false;
		}
	}

	// -------------------------------------------------------------------------
	// Actions
	// -------------------------------------------------------------------------

	async function handleRunTuner() {
		if (!modelInput.trim()) {
			toastError('Please enter a model name');
			return;
		}

		running = true;
		currentJob = null;
		try {
			const job = await startTuning(modelInput.trim());
			currentJob = job;
			startPolling(modelInput.trim());
			toastSuccess(`Tuning started for ${modelInput}`);
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to start tuning');
			running = false;
		}
	}

	async function handleCancel() {
		if (!modelInput.trim()) return;
		try {
			await cancelTuning(modelInput.trim());
			running = false;
			currentJob = null;
			stopPolling();
			toastSuccess('Tuning cancelled');
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to cancel');
		}
	}

	async function handleApply(model: string) {
		try {
			const resp = await applyTunerResult(model);
			toastSuccess(`Applied optimal parameters for ${model}`);
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to apply');
		}
	}

	async function handleDelete(model: string) {
		try {
			await deleteTunerResult(model);
			toastSuccess(`Cleared results for ${model}`);
			delete results[model];
			delete recommendations[model];
			results = results;
			recommendations = recommendations;
			savedProfiles = savedProfiles.filter((m) => m !== model);
			if (selectedModel === model) selectedModel = '';
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to delete');
		}
	}

	// Load recommendations for a model
	async function loadRecommendations(model: string) {
		if (recommendations[model]) return; // Already loaded
		loadingRecs = model;
		try {
			const resp = await getTunerRecommendations(model);
			recommendations[model] = resp.recommendations || [];
			recommendations = recommendations;
		} catch {
			recommendations[model] = [];
			recommendations = recommendations;
		} finally {
			loadingRecs = '';
		}
	}

	async function toggleModel(model: string) {
		if (selectedModel === model) {
			selectedModel = '';
		} else {
			selectedModel = model;
			await loadRecommendations(model);
		}
	}

	function confidenceColor(confidence: string): string {
		switch (confidence) {
			case 'high': return 'var(--oo-status-success)';
			case 'medium': return 'var(--oo-status-warning)';
			default: return 'var(--oo-fg-muted)';
		}
	}

	// -------------------------------------------------------------------------
	// Polling
	// -------------------------------------------------------------------------

	function startPolling(model: string) {
		stopPolling();
		pollTimer = setInterval(async () => {
			try {
				const job = await getTunerJob(model);
				currentJob = job;
				if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
					running = false;
					stopPolling();
					if (job.status === 'completed') {
						toastSuccess(`Tuning completed for ${model}!`);
						await loadData();
					} else if (job.status === 'failed') {
						toastError(`Tuning failed: ${job.error}`);
					}
				}
			} catch {
				// Job may not exist yet, keep polling
			}
		}, 1500);
	}

	function stopPolling() {
		if (pollTimer) {
			clearInterval(pollTimer);
			pollTimer = null;
		}
	}

	// -------------------------------------------------------------------------
	// Formatting helpers
	// -------------------------------------------------------------------------

	function formatSpeed(val: number): string {
		return val.toFixed(1);
	}

	function formatSpeedup(val: number): string {
		return val.toFixed(2) + 'x';
	}

	function formatDate(ts: number): string {
		if (!ts) return '';
		return new Date(ts * 1000).toLocaleString();
	}

	function paramLabel(key: string): string {
		const labels: Record<string, string> = {
			batch_size: 'Batch Size',
			ubatch_size: 'Micro Batch',
			threads: 'Threads',
			flash_attention: 'Flash Attention',
			gpu_layers: 'GPU Layers',
		};
		return labels[key] || key;
	}
</script>

<!-- Collapsible wrapper -->
<div class="rounded-lg overflow-hidden" style="border: 1px solid var(--oo-bd-subtle);">
	<button
		on:click={() => { open = !open; }}
		class="w-full flex items-center justify-between px-4 py-3 text-left transition-colors"
		style="background-color: var(--oo-bg-elevated);"
	>
		<div class="flex items-center gap-2">
			<svg class="w-4 h-4" style="color: var(--oo-fg-tertiary);" fill="none"
				viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
					stroke-linecap="round" stroke-linejoin="round" />
			</svg>
			<div>
				<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">
					Performance Tuner
				</span>
				<span class="text-xs ml-2" style="color: var(--oo-fg-muted);">
					{savedProfiles.length > 0 ? `${savedProfiles.length} model(s) tuned` : 'Find optimal inference params'}
				</span>
			</div>
		</div>
		<svg class="w-4 h-4 transition-transform {open ? 'rotate-180' : ''}"
			style="color: var(--oo-fg-muted);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
			<path d="M19 9l-7 7-7-7" />
		</svg>
	</button>

	{#if open}
		<div class="px-4 py-4 space-y-4" style="border-top: 1px solid var(--oo-bd-subtle);">
			{#if loading}
				<div class="flex items-center gap-2 text-sm py-2" style="color: var(--oo-fg-muted);">
					<div class="w-4 h-4 border-2 rounded-full animate-spin"
						style="border-color: var(--oo-bd-default); border-top-color: var(--oo-acc-400);" />
					Loading...
				</div>
			{:else if error}
				<div class="px-3 py-2 rounded-lg text-xs"
					style="background-color: var(--oo-error-bg); border: 1px solid var(--oo-error-bd); color: var(--oo-error);">
					{error}
					<button on:click={loadData} class="ml-2 underline">Retry</button>
				</div>
			{:else}
				<!-- Description -->
				<p class="text-xs" style="color: var(--oo-fg-muted);">
					Benchmarks different inference parameters to find the fastest configuration
					for your hardware. Estimated time: 2-5 minutes per model.
				</p>

				<!-- Run tuner -->
				<div class="flex gap-2">
					<input
						type="text"
						bind:value={modelInput}
						class="flex-1 px-3 py-2 rounded-lg text-sm font-mono"
						style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
						placeholder="Model name (e.g. llama3:8b)"
						disabled={running}
					/>
					{#if running}
						<button
							on:click={handleCancel}
							class="px-4 py-2 rounded-lg text-sm font-medium shrink-0 transition-colors"
							style="background-color: var(--oo-error-bg); color: var(--oo-error); border: 1px solid var(--oo-error-bd);"
						>
							Cancel
						</button>
					{:else}
						<button
							on:click={handleRunTuner}
							class="px-4 py-2 rounded-lg text-sm font-medium shrink-0 transition-colors"
							style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
						>
							Run Tuner
						</button>
					{/if}
				</div>

				<!-- Progress bar -->
				{#if currentJob && (currentJob.status === 'running' || currentJob.status === 'pending')}
					<div class="space-y-2">
						<div class="flex items-center justify-between text-xs">
							<span style="color: var(--oo-fg-secondary);">{currentJob.current_step || 'Starting...'}</span>
							<span class="font-mono" style="color: var(--oo-fg-muted);">
								{(currentJob.progress * 100).toFixed(0)}%
							</span>
						</div>
						<div class="w-full h-2 rounded-full overflow-hidden"
							style="background-color: var(--oo-bg-overlay);">
							<div
								class="h-full rounded-full transition-all duration-300"
								style="width: {currentJob.progress * 100}%; background-color: var(--oo-acc-500);"
							/>
						</div>
						<p class="text-[10px]" style="color: var(--oo-fg-muted);">
							Step {currentJob.completed_steps}/{currentJob.total_steps}
						</p>
					</div>
				{/if}

				<!-- Completed job summary -->
				{#if currentJob && currentJob.status === 'completed' && currentJob.result}
					<div class="rounded-lg px-3 py-3"
						style="background-color: var(--oo-success-bg); border: 1px solid var(--oo-success-bd);">
						<div class="text-xs font-medium mb-1" style="color: var(--oo-success);">
							Tuning Complete
						</div>
						<p class="text-xs" style="color: var(--oo-fg-secondary);">
							{currentJob.current_step}
						</p>
					</div>
				{/if}

				<!-- Saved results -->
				{#if savedProfiles.length > 0}
					<div>
						<div class="text-xs font-medium mb-2" style="color: var(--oo-fg-secondary);">
							Tuned Models
						</div>
						<div class="space-y-2">
							{#each savedProfiles as model (model)}
								{@const profile = results[model]}
								<div class="rounded-lg px-3 py-3"
									style="background-color: var(--oo-bg-overlay); border: 1px solid var(--oo-bd-subtle);">
									<div class="flex items-center justify-between mb-2">
										<span class="text-sm font-mono font-medium" style="color: var(--oo-fg-primary);">
											{model}
										</span>
										<div class="flex items-center gap-2">
											{#if profile}
												<span class="text-xs font-mono px-1.5 py-0.5 rounded"
													style="background-color: var(--oo-acc-900); color: var(--oo-acc-400);">
													{formatSpeedup(profile.speedup_factor)}
												</span>
											{/if}
											<button
												on:click={() => toggleModel(model)}
												class="text-xs underline"
												style="color: var(--oo-fg-muted);"
											>
												{selectedModel === model ? 'Hide' : 'Details'}
											</button>
										</div>
									</div>

									{#if profile}
										<!-- Speed summary -->
										<div class="grid grid-cols-2 gap-2 text-xs mb-2">
											<div>
												<span style="color: var(--oo-fg-muted);">Baseline:</span>
												<span class="font-mono" style="color: var(--oo-fg-primary);">
													{formatSpeed(profile.baseline_tg_speed)} tok/s
												</span>
											</div>
											<div>
												<span style="color: var(--oo-fg-muted);">Optimized:</span>
												<span class="font-mono font-medium" style="color: var(--oo-acc-400);">
													{formatSpeed(profile.best_tg_speed)} tok/s
												</span>
											</div>
										</div>

										<!-- Expanded details -->
										{#if selectedModel === model}
											<div class="mt-2 pt-2 space-y-2" style="border-top: 1px solid var(--oo-bd-subtle);">
												<div class="text-xs font-medium" style="color: var(--oo-fg-secondary);">
													Optimal Parameters
												</div>
												<div class="grid grid-cols-2 gap-1 text-xs">
													{#each Object.entries(profile.best_params) as [key, value]}
														<div class="flex justify-between px-2 py-1 rounded"
															style="background-color: var(--oo-bg-elevated);">
															<span style="color: var(--oo-fg-muted);">{paramLabel(key)}</span>
															<span class="font-mono" style="color: var(--oo-fg-primary);">{value}</span>
														</div>
													{/each}
												</div>
												<div class="text-[10px]" style="color: var(--oo-fg-muted);">
													Hardware: {profile.hardware_fingerprint} &middot; {formatDate(profile.timestamp)}
												</div>

												<!-- Recommendation cards -->
												{#if loadingRecs === model}
													<div class="text-xs py-2" style="color: var(--oo-fg-muted);">
														Loading recommendations...
													</div>
												{:else if recommendations[model] && recommendations[model].length > 0}
													<div class="mt-3 pt-2" style="border-top: 1px solid var(--oo-bd-subtle);">
														<div class="text-xs font-medium mb-2" style="color: var(--oo-fg-secondary);">
															Recommendations
														</div>
														<div class="space-y-2">
															{#each recommendations[model] as rec}
																<div class="rounded px-3 py-2"
																	style="background-color: var(--oo-bg-elevated);
																		border-left: 3px solid {confidenceColor(rec.confidence)};">
																	<div class="flex items-center justify-between mb-1">
																		<span class="text-xs font-medium" style="color: var(--oo-fg-primary);">
																			{rec.title}
																		</span>
																		<span class="text-[10px] font-mono px-1.5 py-0.5 rounded"
																			style="background-color: var(--oo-acc-900); color: var(--oo-acc-400);">
																			{rec.estimated_speedup.toFixed(1)}x
																		</span>
																	</div>
																	<p class="text-[11px] leading-snug" style="color: var(--oo-fg-muted);">
																		{rec.description}
																	</p>
																	{#if rec.parameter !== 'all'}
																		<div class="flex items-center justify-between mt-1.5">
																			<span class="text-[10px]" style="color: var(--oo-fg-muted);">
																				{rec.parameter}: {rec.current_value} &rarr; {rec.recommended_value}
																			</span>
																			<button
																				on:click={() => handleApply(model)}
																				class="text-[10px] px-2 py-0.5 rounded transition-colors"
																				style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
																			>
																				Apply
																			</button>
																		</div>
																	{/if}
																</div>
															{/each}
														</div>
													</div>
												{/if}
											</div>
										{/if}
									{/if}

									<!-- Action buttons -->
									<div class="flex gap-2 mt-2">
										<button
											on:click={() => handleApply(model)}
											class="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
											style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
										>
											Apply
										</button>
										<button
											on:click={() => handleDelete(model)}
											class="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
											style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-tertiary);
												border: 1px solid var(--oo-bd-subtle);"
										>
											Clear
										</button>
									</div>
								</div>
							{/each}
						</div>
					</div>
				{/if}
			{/if}
		</div>
	{/if}
</div>
