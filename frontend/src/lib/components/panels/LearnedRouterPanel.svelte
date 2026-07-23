<!--
  LearnedRouterPanel — ML-based routing panel.

  Sections:
  1. Status card — training state, sample count, last accuracy
  2. Training controls — Train button, model type selector, threshold slider
  3. Live classifier test — query input, ML vs YAML side-by-side
  4. A/B metrics — learned vs yaml usage ratio, confidence, agreement rate
  5. Confidence histogram — bar chart of ML confidence distribution
  6. Top disagreements — table of ML vs YAML divergences
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getLearnedRouterStatus,
		triggerTraining,
		getLearnedRouterConfig,
		updateLearnedRouterConfig,
		classifyQuery,
		getABMetrics,
		type LearnedRouterStatus,
		type LearnedRouterConfig,
		type TrainingResult,
		type ClassifyResponse,
		type ABMetrics,
		type HistogramBucket
	} from '$lib/api/learnedRouting';
	import { toastSuccess, toastError } from '$lib/stores/notifications';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	let loading = true;
	let error = '';

	let status: LearnedRouterStatus | null = null;
	let config: LearnedRouterConfig | null = null;
	let metrics: ABMetrics | null = null;

	// Training
	let training = false;
	let lastTrainResult: TrainingResult | null = null;

	// Config edits
	let localEnabled = false;
	let localModelType: 'logistic' | 'random_forest' = 'logistic';
	let localThreshold = 0.70;
	let savingConfig = false;

	// Live classifier
	let testQuery = '';
	let testYamlTask = 'general';
	let classifying = false;
	let classifyResult: ClassifyResponse | null = null;

	// Metrics window
	let metricsWindow = 24;
	let loadingMetrics = false;

	// -------------------------------------------------------------------------
	// Load
	// -------------------------------------------------------------------------

	async function load() {
		loading = true;
		error = '';
		try {
			[status, config] = await Promise.all([
				getLearnedRouterStatus(),
				getLearnedRouterConfig()
			]);
			if (status && config) syncLocalFromConfig(config, status);
			await loadMetrics();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load learned router data';
		} finally {
			loading = false;
		}
	}

	function syncLocalFromConfig(cfg: LearnedRouterConfig, st: LearnedRouterStatus) {
		localEnabled = cfg.enabled;
		localModelType = (cfg.model_type as 'logistic' | 'random_forest') ?? 'logistic';
		localThreshold = cfg.confidence_threshold;
	}

	async function loadMetrics() {
		loadingMetrics = true;
		try {
			metrics = await getABMetrics(metricsWindow);
		} catch {
			metrics = null;
		} finally {
			loadingMetrics = false;
		}
	}

	// -------------------------------------------------------------------------
	// Training
	// -------------------------------------------------------------------------

	async function handleTrain() {
		training = true;
		lastTrainResult = null;
		try {
			const result = await triggerTraining();
			lastTrainResult = result;
			if (result.success) {
				toastSuccess(`Training complete — accuracy: ${(result.accuracy * 100).toFixed(1)}%`);
				await load();
			} else {
				toastError(`Training failed: ${result.error}`);
			}
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Training failed');
		} finally {
			training = false;
		}
	}

	// -------------------------------------------------------------------------
	// Config save
	// -------------------------------------------------------------------------

	async function saveConfig() {
		savingConfig = true;
		try {
			await updateLearnedRouterConfig({
				model_type: localModelType,
				confidence_threshold: localThreshold
			});
			toastSuccess('Routing config updated');
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to save config');
		} finally {
			savingConfig = false;
		}
	}

	async function toggleEnabled() {
		savingConfig = true;
		try {
			await updateLearnedRouterConfig({ enabled: !localEnabled });
			localEnabled = !localEnabled;
			toastSuccess(localEnabled ? 'Learned router enabled' : 'Learned router disabled');
			await load();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to toggle learned router');
		} finally {
			savingConfig = false;
		}
	}

	// -------------------------------------------------------------------------
	// Live classify
	// -------------------------------------------------------------------------

	async function handleClassify() {
		if (!testQuery.trim()) return;
		classifying = true;
		classifyResult = null;
		try {
			classifyResult = await classifyQuery({
				query: testQuery,
				yaml_task_type: testYamlTask || 'general'
			});
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Classification failed');
		} finally {
			classifying = false;
		}
	}

	// -------------------------------------------------------------------------
	// Helpers
	// -------------------------------------------------------------------------

	function formatDate(ts: number): string {
		if (!ts) return '—';
		return new Date(ts * 1000).toLocaleString();
	}

	function pct(v: number): string {
		return `${(v * 100).toFixed(1)}%`;
	}

	/** Histogram bar height as percentage of tallest bucket. */
	function histMax(buckets: HistogramBucket[]): number {
		return Math.max(...buckets.map((b) => b.count), 1);
	}

	onMount(load);
</script>

<!-- ============================================================ -->
<!-- Root container                                               -->
<!-- ============================================================ -->
<div class="space-y-6">

	<div class="flex items-center justify-between">
		<h2 class="text-base font-medium" style="color: var(--oo-fg-primary);">
			Learned Router
			<span class="ml-2 text-xs font-normal" style="color: var(--oo-fg-tertiary);">S67 · ML-based routing</span>
		</h2>
		<button
			on:click={load}
			disabled={loading}
			class="text-xs px-2 py-1 rounded transition-colors"
			style="background-color: var(--oo-btn-ghost-bg); color: var(--oo-fg-secondary);"
		>
			{loading ? 'Loading…' : 'Refresh'}
		</button>
	</div>

	{#if loading}
		<div class="flex items-center gap-2 py-8 justify-center" style="color: var(--oo-fg-tertiary);">
			<div class="w-4 h-4 border-2 border-surface-600 border-t-accent-500 rounded-full animate-spin" />
			<span class="text-sm">Loading…</span>
		</div>
	{:else if error}
		<div class="px-3 py-2 rounded text-sm" style="background-color: var(--oo-err-bg); border: 1px solid var(--oo-err-bd); color: var(--oo-err-fg);">
			{error}
			<button on:click={load} class="ml-2 underline hover:no-underline">Retry</button>
		</div>
	{:else if status}

		<!-- ============================================================ -->
		<!-- 1. Status card                                               -->
		<!-- ============================================================ -->
		<div class="rounded-lg p-4 space-y-3"
			style="background-color: var(--oo-surface-800); border: 1px solid var(--oo-bd-default);">

			<div class="flex items-center justify-between">
				<h3 class="text-sm font-medium" style="color: var(--oo-fg-secondary);">Status</h3>
				<div class="flex items-center gap-2">
					<!-- Trained badge -->
					<span
						class="text-xs px-2 py-0.5 rounded-full font-medium"
						style="{status.trained
							? 'background-color: var(--oo-acc-900); color: var(--oo-acc-300);'
							: 'background-color: var(--oo-surface-700); color: var(--oo-fg-tertiary);'}"
					>
						{status.trained ? 'Trained' : 'Untrained'}
					</span>
					<!-- Enabled toggle -->
					<button
						on:click={toggleEnabled}
						disabled={savingConfig || (!status.trained && !localEnabled)}
						title={!status.trained && !localEnabled ? 'Train the model first' : ''}
						aria-label={localEnabled ? 'Disable learned router' : 'Enable learned router'}
						class="relative inline-flex h-5 w-9 items-center rounded-full transition-colors disabled:opacity-40"
						style="{localEnabled
							? 'background-color: var(--oo-acc-500);'
							: 'background-color: var(--oo-surface-600);'}"
					>
						<span
							class="inline-block h-3.5 w-3.5 transform rounded-full bg-[var(--oo-toggle-knob)] shadow transition-transform"
							style="transform: translateX({localEnabled ? '18px' : '2px'})"
						/>
					</button>
					<span class="text-xs" style="color: var(--oo-fg-tertiary);">
						{localEnabled ? 'Enabled' : 'Disabled'}
					</span>
				</div>
			</div>

			<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
				<div>
					<div class="text-xs" style="color: var(--oo-fg-tertiary);">Samples</div>
					<div class="text-sm font-mono font-medium" style="color: var(--oo-fg-primary);">
						{status.sample_count.toLocaleString()}
					</div>
				</div>
				<div>
					<div class="text-xs" style="color: var(--oo-fg-tertiary);">Min. required</div>
					<div class="text-sm font-mono font-medium" style="color: var(--oo-fg-primary);">
						{status.min_training_samples}
					</div>
				</div>
				<div>
					<div class="text-xs" style="color: var(--oo-fg-tertiary);">Accuracy</div>
					<div class="text-sm font-mono font-medium" style="color: var(--oo-fg-primary);">
						{status.last_training ? pct(status.last_training.accuracy) : '—'}
					</div>
				</div>
				<div>
					<div class="text-xs" style="color: var(--oo-fg-tertiary);">Last trained</div>
					<div class="text-xs" style="color: var(--oo-fg-secondary);">
						{status.last_training ? formatDate(status.last_training.trained_at) : '—'}
					</div>
				</div>
			</div>

			<!-- Sample progress bar -->
			{#if !status.trained}
				<div>
					<div class="flex justify-between text-xs mb-1" style="color: var(--oo-fg-tertiary);">
						<span>Samples collected</span>
						<span>{status.sample_count} / {status.min_training_samples}</span>
					</div>
					<div class="h-1.5 rounded-full" style="background-color: var(--oo-surface-600);">
						<div
							class="h-1.5 rounded-full transition-all"
							style="background-color: var(--oo-acc-500); width: {Math.min(100, (status.sample_count / status.min_training_samples) * 100)}%"
						/>
					</div>
					<p class="text-xs mt-1" style="color: var(--oo-fg-tertiary);">
						Routing decisions are automatically logged. Once {status.min_training_samples} samples are collected, training becomes available.
					</p>
				</div>
			{/if}

			<!-- Class distribution -->
			{#if Object.keys(status.class_distribution).length > 0}
				<div>
					<div class="text-xs mb-2" style="color: var(--oo-fg-tertiary);">Sample distribution</div>
					<div class="flex flex-wrap gap-1.5">
						{#each Object.entries(status.class_distribution).sort((a, b) => b[1] - a[1]) as [cls, cnt]}
							<span class="text-xs px-2 py-0.5 rounded"
								style="background-color: var(--oo-surface-700); color: var(--oo-fg-secondary);">
								{cls} <span style="color: var(--oo-fg-tertiary);">·</span> {cnt}
							</span>
						{/each}
					</div>
				</div>
			{/if}
		</div>

		<!-- ============================================================ -->
		<!-- 2. Training controls                                         -->
		<!-- ============================================================ -->
		<div class="rounded-lg p-4 space-y-4"
			style="background-color: var(--oo-surface-800); border: 1px solid var(--oo-bd-default);">

			<h3 class="text-sm font-medium" style="color: var(--oo-fg-secondary);">Training & Configuration</h3>

			<div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
				<!-- Model type -->
				<div>
					<label class="block text-xs mb-1" style="color: var(--oo-fg-tertiary);">Classifier</label>
					<select
						bind:value={localModelType}
						class="w-full px-3 py-1.5 rounded text-sm"
						style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
					>
						<option value="logistic">Logistic Regression</option>
						<option value="random_forest">Random Forest</option>
					</select>
				</div>

				<!-- Confidence threshold -->
				<div>
					<label class="block text-xs mb-1" style="color: var(--oo-fg-tertiary);">
						Confidence threshold: {localThreshold.toFixed(2)}
					</label>
					<input
						type="range"
						min="0.5"
						max="0.99"
						step="0.01"
						bind:value={localThreshold}
						class="w-full"
					/>
					<p class="text-xs mt-0.5" style="color: var(--oo-fg-tertiary);">
						Below this, YAML router is used as fallback.
					</p>
				</div>
			</div>

			<div class="flex gap-2 flex-wrap">
				<button
					on:click={saveConfig}
					disabled={savingConfig}
					class="px-3 py-1.5 rounded text-sm transition-colors disabled:opacity-50"
					style="background-color: var(--oo-surface-700); color: var(--oo-fg-secondary);"
				>
					{savingConfig ? 'Saving…' : 'Save config'}
				</button>

				<button
					on:click={handleTrain}
					disabled={training || status.sample_count < status.min_training_samples}
					class="px-4 py-1.5 rounded text-sm font-medium transition-colors disabled:opacity-50"
					style="background-color: var(--oo-acc-600); color: white;"
					title={status.sample_count < status.min_training_samples
						? `Need ${status.min_training_samples - status.sample_count} more samples`
						: 'Retrain classifier on all stored samples'}
				>
					{#if training}
						<span class="flex items-center gap-1.5">
							<span class="w-3.5 h-3.5 border-2 border-[var(--oo-fg-primary)]/40 border-t-[var(--oo-fg-primary)] rounded-full animate-spin inline-block" />
							Training…
						</span>
					{:else}
						{status.trained ? 'Retrain' : 'Train'}
					{/if}
				</button>
			</div>

			<!-- Last training result -->
			{#if lastTrainResult}
				<div class="text-xs px-3 py-2 rounded"
					style="{lastTrainResult.success
						? 'background-color: var(--oo-acc-900); color: var(--oo-acc-300);'
						: 'background-color: var(--oo-err-bg); color: var(--oo-err-fg);'}">
					{#if lastTrainResult.success}
						Training complete — accuracy: {pct(lastTrainResult.accuracy)},
						{lastTrainResult.n_samples} samples, {lastTrainResult.n_classes} classes
					{:else}
						{lastTrainResult.error}
					{/if}
				</div>
			{/if}
		</div>

		<!-- ============================================================ -->
		<!-- 3. Live classifier test                                      -->
		<!-- ============================================================ -->
		<div class="rounded-lg p-4 space-y-3"
			style="background-color: var(--oo-surface-800); border: 1px solid var(--oo-bd-default);">

			<h3 class="text-sm font-medium" style="color: var(--oo-fg-secondary);">Live Classify Test</h3>
			<p class="text-xs" style="color: var(--oo-fg-tertiary);">
				Enter a query to preview what the ML model and YAML heuristic would each predict.
			</p>

			<div class="flex gap-2">
				<input
					type="text"
					bind:value={testQuery}
					placeholder="e.g. write a python function to parse JSON…"
					class="flex-1 px-3 py-2 rounded text-sm"
					style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
					on:keydown={(e) => e.key === 'Enter' && handleClassify()}
				/>
				<input
					type="text"
					bind:value={testYamlTask}
					placeholder="YAML task"
					class="w-28 px-3 py-2 rounded text-sm"
					style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
					title="YAML router baseline task type for comparison"
				/>
				<button
					on:click={handleClassify}
					disabled={classifying || !testQuery.trim()}
					class="px-4 py-2 rounded text-sm font-medium transition-colors disabled:opacity-50"
					style="background-color: var(--oo-acc-600); color: white;"
				>
					{classifying ? '…' : 'Classify'}
				</button>
			</div>

			{#if classifyResult}
				<div class="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-2">
					<!-- ML prediction -->
					<div class="rounded px-3 py-2"
						style="background-color: var(--oo-surface-700); border: 1px solid var(--oo-bd-default);">
						<div class="text-xs mb-1" style="color: var(--oo-fg-tertiary);">ML prediction</div>
						<div class="text-sm font-medium font-mono" style="color: var(--oo-acc-300);">
							{classifyResult.ml_prediction.task_type}
						</div>
						<div class="text-xs mt-0.5" style="color: var(--oo-fg-tertiary);">
							confidence: {pct(classifyResult.ml_prediction.confidence)}
						</div>
						{#if classifyResult.ml_prediction.top_classes.length > 0}
							<div class="mt-1.5 space-y-0.5">
								{#each classifyResult.ml_prediction.top_classes.slice(0, 3) as cls}
									<div class="flex justify-between text-xs" style="color: var(--oo-fg-tertiary);">
										<span>{cls.task_type}</span>
										<span class="font-mono">{pct(cls.confidence)}</span>
									</div>
								{/each}
							</div>
						{/if}
					</div>

					<!-- YAML + decision -->
					<div class="rounded px-3 py-2"
						style="background-color: var(--oo-surface-700); border: 1px solid var(--oo-bd-default);">
						<div class="text-xs mb-1" style="color: var(--oo-fg-tertiary);">YAML heuristic</div>
						<div class="text-sm font-medium font-mono" style="color: var(--oo-fg-secondary);">
							{classifyResult.yaml_task_type}
						</div>
						<div class="mt-2 pt-2" style="border-top: 1px solid var(--oo-bd-default);">
							<div class="text-xs" style="color: var(--oo-fg-tertiary);">Would use</div>
							<div class="flex items-center gap-2 mt-0.5">
								<span class="text-sm font-medium font-mono" style="color: var(--oo-fg-primary);">
									{classifyResult.final_task_type}
								</span>
								<span class="text-xs px-1.5 py-0.5 rounded"
									style="{classifyResult.routing_source === 'learned'
										? 'background-color: var(--oo-acc-900); color: var(--oo-acc-300);'
										: 'background-color: var(--oo-surface-600); color: var(--oo-fg-tertiary);'}">
									{classifyResult.routing_source}
								</span>
							</div>
						</div>
					</div>
				</div>
			{/if}
		</div>

		<!-- ============================================================ -->
		<!-- 4 + 5 + 6. A/B Metrics                                      -->
		<!-- ============================================================ -->
		<div class="rounded-lg p-4 space-y-4"
			style="background-color: var(--oo-surface-800); border: 1px solid var(--oo-bd-default);">

			<div class="flex items-center justify-between">
				<h3 class="text-sm font-medium" style="color: var(--oo-fg-secondary);">A/B Metrics</h3>
				<div class="flex items-center gap-2">
					<select
						bind:value={metricsWindow}
						on:change={loadMetrics}
						class="text-xs px-2 py-1 rounded"
						style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-secondary);"
					>
						<option value={1}>Last 1h</option>
						<option value={6}>Last 6h</option>
						<option value={24}>Last 24h</option>
						<option value={168}>Last 7d</option>
					</select>
					<button
						on:click={loadMetrics}
						disabled={loadingMetrics}
						class="text-xs px-2 py-1 rounded transition-colors"
						style="background-color: var(--oo-btn-ghost-bg); color: var(--oo-fg-tertiary);"
					>
						{loadingMetrics ? '…' : 'Refresh'}
					</button>
				</div>
			</div>

			{#if metrics}
				{#if metrics.total_decisions === 0}
					<p class="text-sm py-4 text-center" style="color: var(--oo-fg-tertiary);">
						No routing decisions logged in this time window yet.
					</p>
				{:else}
					<!-- Summary stats -->
					<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
						<div>
							<div class="text-xs" style="color: var(--oo-fg-tertiary);">Total decisions</div>
							<div class="text-sm font-mono font-medium" style="color: var(--oo-fg-primary);">
								{metrics.total_decisions}
							</div>
						</div>
						<div>
							<div class="text-xs" style="color: var(--oo-fg-tertiary);">ML used</div>
							<div class="text-sm font-mono font-medium" style="color: var(--oo-acc-300);">
								{pct(metrics.learned_ratio)}
							</div>
						</div>
						<div>
							<div class="text-xs" style="color: var(--oo-fg-tertiary);">Avg confidence</div>
							<div class="text-sm font-mono font-medium" style="color: var(--oo-fg-primary);">
								{pct(metrics.avg_ml_confidence)}
							</div>
						</div>
						<div>
							<div class="text-xs" style="color: var(--oo-fg-tertiary);">Agreement rate</div>
							<div class="text-sm font-mono font-medium" style="color: var(--oo-fg-primary);">
								{pct(metrics.class_agreement_rate)}
							</div>
						</div>
					</div>

					<!-- Routing source bar -->
					{#if metrics.total_decisions > 0}
						<div>
							<div class="text-xs mb-1" style="color: var(--oo-fg-tertiary);">
								Routing source — ML: {metrics.learned_count} · YAML: {metrics.yaml_count}
							</div>
							<div class="h-2 rounded-full overflow-hidden" style="background-color: var(--oo-surface-600);">
								<div
									class="h-full rounded-full transition-all"
									style="background-color: var(--oo-acc-500); width: {pct(metrics.learned_ratio)}"
								/>
							</div>
						</div>
					{/if}

					<!-- Confidence histogram -->
					{#if metrics.confidence_histogram.length > 0}
						<div>
							<div class="text-xs mb-2" style="color: var(--oo-fg-tertiary);">ML confidence distribution</div>
							<div class="flex items-end gap-0.5 h-16">
								{#each metrics.confidence_histogram as bucket}
									{@const maxVal = histMax(metrics.confidence_histogram)}
									<div
										class="flex-1 rounded-t transition-all"
										style="background-color: var(--oo-acc-600); opacity: {0.4 + 0.6 * (bucket.count / maxVal)}; height: {Math.max(4, (bucket.count / maxVal) * 100)}%;"
										title="{bucket.bucket_min.toFixed(1)}–{bucket.bucket_max.toFixed(1)}: {bucket.count}"
									/>
								{/each}
							</div>
							<div class="flex justify-between text-xs mt-1" style="color: var(--oo-fg-tertiary);">
								<span>0%</span>
								<span>50%</span>
								<span>100%</span>
							</div>
						</div>
					{/if}

					<!-- Top disagreements -->
					{#if metrics.top_disagreements.length > 0}
						<div>
							<div class="text-xs mb-2" style="color: var(--oo-fg-tertiary);">Top ML vs YAML disagreements</div>
							<div class="space-y-1">
								{#each metrics.top_disagreements as d}
									<div class="flex items-center justify-between text-xs px-2 py-1.5 rounded"
										style="background-color: var(--oo-surface-700);">
										<span>
											<span class="font-mono" style="color: var(--oo-acc-300);">{d.ml_task_type}</span>
											<span style="color: var(--oo-fg-tertiary);"> vs </span>
											<span class="font-mono" style="color: var(--oo-fg-secondary);">{d.yaml_task_type}</span>
										</span>
										<span class="font-mono" style="color: var(--oo-fg-tertiary);">{d.count}×</span>
									</div>
								{/each}
							</div>
						</div>
					{/if}
				{/if}
			{:else if loadingMetrics}
				<div class="flex items-center gap-2 py-4 justify-center" style="color: var(--oo-fg-tertiary);">
					<div class="w-4 h-4 border-2 border-surface-600 border-t-accent-500 rounded-full animate-spin" />
					<span class="text-xs">Loading metrics…</span>
				</div>
			{/if}
		</div>

	{/if}
</div>
