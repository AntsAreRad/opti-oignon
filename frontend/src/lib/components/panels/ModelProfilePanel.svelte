<!--
  ModelProfilePanel.svelte
  View and edit model capability profiles for smart routing.
  Shows task scores as horizontal bars, router config,
  and per-step model selection preview.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getProfiles,
		saveProfile,
		deleteProfile,
		updateTaskScores,
		autoDetectModel,
		saveAllProfiles,
		getRouterConfig,
		updateRouterConfig,
		saveRouterConfig,
		selectForPipeline,
	} from '$lib/api/smartRouting';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type { ModelProfileInfo, SmartRouterConfig, SmartRoutingResult } from '$lib/types';

	// State
	let profiles: Record<string, ModelProfileInfo> = {};
	let config: SmartRouterConfig | null = null;
	let loading = true;
	let error = '';
	let saving = false;
	let detecting = '';

	// Selected profile for detail view
	let selectedModel: string | null = null;
	let editingScores: Record<string, number> = {};
	let newTaskType = '';
	let newTaskScore = 0.5;

	// Pipeline preview
	let pipelinePreview: Record<string, SmartRoutingResult> = {};
	let previewLoading = false;

	// Known step types for pipeline preview
	const STEP_TYPES = [
		'direct', 'tools', 'code_verify', 'think',
		'web_search', 'reasoning', 'consensus', 'self_correct'
	];

	// Derived state
	$: profileList = Object.values(profiles).sort((a, b) =>
		a.display_name.localeCompare(b.display_name)
	);
	$: selectedProfile = selectedModel ? profiles[selectedModel] ?? null : null;

	// ------------------------------------------------------------------
	// Data loading
	// ------------------------------------------------------------------

	async function loadAll() {
		loading = true;
		error = '';
		try {
			const [profData, cfgData] = await Promise.all([
				getProfiles(),
				getRouterConfig().catch(() => null),
			]);
			profiles = profData.profiles;
			config = cfgData;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load profiles';
		} finally {
			loading = false;
		}
	}

	async function loadPipelinePreview() {
		if (!config?.operational) return;
		previewLoading = true;
		try {
			const result = await selectForPipeline(STEP_TYPES);
			pipelinePreview = result.selections;
		} catch {
			// Non-critical, silent fail
		} finally {
			previewLoading = false;
		}
	}

	// ------------------------------------------------------------------
	// Config actions
	// ------------------------------------------------------------------

	async function toggleEnabled() {
		if (!config) return;
		try {
			const result = await updateRouterConfig({ enabled: !config.enabled });
			config = result.config;
			toastSuccess(config.enabled ? 'Smart routing enabled' : 'Smart routing disabled');
			if (config.enabled) loadPipelinePreview();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to toggle');
		}
	}

	async function setSpeedPreference(pref: string) {
		try {
			const result = await updateRouterConfig({ speed_preference: pref });
			config = result.config;
			toastSuccess(`Speed preference: ${pref}`);
			loadPipelinePreview();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to update');
		}
	}

	async function handleSaveConfig() {
		saving = true;
		try {
			await saveRouterConfig();
			toastSuccess('Router config saved to disk');
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to save config');
		} finally {
			saving = false;
		}
	}

	// ------------------------------------------------------------------
	// Profile actions
	// ------------------------------------------------------------------

	function selectProfileForEdit(name: string) {
		selectedModel = name;
		const p = profiles[name];
		editingScores = p?.task_scores ? { ...p.task_scores } : {};
	}

	function closeDetail() {
		selectedModel = null;
		editingScores = {};
		newTaskType = '';
	}

	async function handleUpdateScores() {
		if (!selectedModel) return;
		saving = true;
		try {
			await updateTaskScores(selectedModel, editingScores);
			// Refresh profile
			const data = await getProfiles();
			profiles = data.profiles;
			toastSuccess(`Task scores updated for ${selectedModel}`);
			loadPipelinePreview();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to update scores');
		} finally {
			saving = false;
		}
	}

	function addTaskScore() {
		const key = newTaskType.trim().toLowerCase().replace(/\s+/g, '_');
		if (!key) return;
		editingScores = { ...editingScores, [key]: newTaskScore };
		newTaskType = '';
		newTaskScore = 0.5;
	}

	function removeTaskScore(key: string) {
		const updated = { ...editingScores };
		delete updated[key];
		editingScores = updated;
	}

	async function handleAutoDetect(modelName: string) {
		detecting = modelName;
		try {
			await autoDetectModel(modelName);
			const data = await getProfiles();
			profiles = data.profiles;
			toastSuccess(`Auto-detected: ${modelName}`);
		} catch (e) {
			toastError(e instanceof Error ? e.message : `Detection failed for ${modelName}`);
		} finally {
			detecting = '';
		}
	}

	async function handleDeleteProfile(modelName: string) {
		try {
			await deleteProfile(modelName);
			const data = await getProfiles();
			profiles = data.profiles;
			if (selectedModel === modelName) closeDetail();
			toastSuccess(`Profile removed: ${modelName}`);
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to delete');
		}
	}

	async function handleSaveAllProfiles() {
		saving = true;
		try {
			await saveAllProfiles();
			toastSuccess('All profiles saved to disk');
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to save profiles');
		} finally {
			saving = false;
		}
	}

	// ------------------------------------------------------------------
	// Helpers
	// ------------------------------------------------------------------

	function tierBadgeStyle(tier: string): string {
		switch (tier) {
			case 'fast': return 'background-color: var(--oo-success-bg); color: var(--oo-success);';
			case 'slow': return 'background-color: var(--oo-error-bg); color: var(--oo-error);';
			case 'high': return 'background-color: var(--oo-info-bg); color: var(--oo-info);';
			case 'low': return 'background-color: var(--oo-warning-bg); color: var(--oo-warning);';
			default: return 'background-color: var(--oo-bg-elevated); color: var(--oo-fg-secondary);';
		}
	}

	function scoreColor(score: number): string {
		if (score >= 0.8) return 'var(--oo-success)';
		if (score >= 0.6) return 'var(--oo-warning)';
		if (score >= 0.4) return 'var(--oo-acc-400)';
		return 'var(--oo-error)';
	}

	function scoreBarWidth(score: number): string {
		return `${Math.round(score * 100)}%`;
	}

	function formatCtx(tokens: number): string {
		if (tokens >= 1048576) return `${(tokens / 1048576).toFixed(0)}M`;
		if (tokens >= 1024) return `${Math.round(tokens / 1024)}K`;
		return String(tokens);
	}

	// MTP detection by model name/family pattern matching.
	const _MTP_PATTERNS = ['deepseek-v3', 'deepseek-r1', 'deepseek-v2.5', 'qwen3', 'glm-4', 'glm4'];
	function _isMtpCapable(name: string, family?: string): boolean {
		const n = (name || '').toLowerCase();
		const f = (family || '').toLowerCase();
		return _MTP_PATTERNS.some((p) => n.includes(p) || f.includes(p));
	}

	onMount(() => {
		loadAll().then(loadPipelinePreview);
	});
</script>

<div class="space-y-6">
	<!-- Loading / Error -->
	{#if loading}
		<div class="flex items-center gap-2 text-sm py-8 justify-center" style="color: var(--oo-fg-tertiary);">
			<div class="w-5 h-5 border-2 rounded-full animate-spin"
				style="border-color: var(--oo-bd-default); border-top-color: var(--oo-acc-500);" />
			Loading model profiles...
		</div>
	{:else if error}
		<div class="px-3 py-2 rounded text-sm mb-4"
			style="background-color: var(--oo-error-bg); border: 1px solid var(--oo-error-bd); color: var(--oo-error);">
			{error}
			<button on:click={loadAll} class="ml-2 underline hover:no-underline">Retry</button>
		</div>
	{:else}

		<!-- ============================================================ -->
		<!-- ROUTER CONFIG SECTION                                        -->
		<!-- ============================================================ -->
		<div class="p-4 rounded-lg" style="background-color: var(--oo-panel-bg); border: 1px solid var(--oo-bd-default);">
			<div class="flex items-center justify-between mb-3">
				<h3 class="text-sm font-medium" style="color: var(--oo-fg-primary);">Smart Routing</h3>
				<label class="flex items-center gap-2 cursor-pointer">
					<span class="text-xs" style="color: var(--oo-fg-tertiary);">
						{config?.operational ? 'Active' : 'Inactive'}
					</span>
					<button
						on:click={toggleEnabled}
						class="relative w-9 h-5 rounded-full transition-colors"
						style="background-color: {config?.enabled ? 'var(--oo-acc-500)' : 'var(--oo-bd-default)'};" aria-label="Toggle model profile"
					>
						<span
							class="absolute top-0.5 left-0.5 w-4 h-4 rounded-full transition-transform"
							style="background-color: var(--oo-toggle-knob); transform: translateX({config?.enabled ? '16px' : '0'});"
						/>
					</button>
				</label>
			</div>

			{#if config}
				<div class="flex flex-wrap gap-3 items-center text-xs" style="color: var(--oo-fg-secondary);">
					<span>Profiles: <strong>{config.profile_count}</strong></span>
					<span>Default: <strong>{config.default_model}</strong></span>
					<span>Speed:</span>
					{#each ['fast', 'balanced', 'quality'] as pref}
						<button
							on:click={() => setSpeedPreference(pref)}
							class="px-2 py-0.5 rounded text-xs transition-colors"
							style="{config.speed_preference === pref
								? 'background-color: var(--oo-acc-500); color: white;'
								: 'background-color: var(--oo-bd-default); color: var(--oo-fg-tertiary);'}"
						>
							{pref}
						</button>
					{/each}
					<button
						on:click={handleSaveConfig}
						disabled={saving}
						class="ml-auto px-2 py-0.5 rounded text-xs transition-colors"
						style="background-color: var(--oo-bd-default); color: var(--oo-fg-tertiary);"
					>
						{saving ? 'Saving...' : 'Save config'}
					</button>
				</div>
			{/if}
		</div>

		<!-- ============================================================ -->
		<!-- PIPELINE PREVIEW                                             -->
		<!-- ============================================================ -->
		{#if config?.operational && Object.keys(pipelinePreview).length > 0}
			<div class="p-4 rounded-lg" style="background-color: var(--oo-panel-bg); border: 1px solid var(--oo-bd-default);">
				<h3 class="text-sm font-medium mb-3" style="color: var(--oo-fg-primary);">
					Per-Step Model Selection
					{#if previewLoading}
						<span class="text-xs font-normal" style="color: var(--oo-fg-tertiary);">(refreshing...)</span>
					{/if}
				</h3>
				<div class="grid grid-cols-2 gap-2">
					{#each STEP_TYPES as step}
						{@const r = pipelinePreview[step]}
						{#if r}
							<div class="flex items-center gap-2 px-2 py-1.5 rounded text-xs"
								style="background-color: var(--oo-bg-elevated);">
								<span class="font-mono w-24 shrink-0" style="color: var(--oo-fg-tertiary);">{step}</span>
								<span class="truncate" style="color: var(--oo-fg-primary);">
									{r.model}
								</span>
								{#if !r.fallback}
									<span class="ml-auto shrink-0 tabular-nums" style="color: {scoreColor(r.score)};">
										{(r.score).toFixed(2)}
									</span>
								{:else}
									<span class="ml-auto text-xs italic" style="color: var(--oo-fg-tertiary);">fallback</span>
								{/if}
							</div>
						{/if}
					{/each}
				</div>
			</div>
		{/if}

		<!-- ============================================================ -->
		<!-- MODEL PROFILES LIST                                          -->
		<!-- ============================================================ -->
		<div>
			<div class="flex items-center justify-between mb-3">
				<h3 class="text-sm font-medium" style="color: var(--oo-fg-primary);">
					Model Profiles ({profileList.length})
				</h3>
				<button
					on:click={handleSaveAllProfiles}
					disabled={saving}
					class="px-3 py-1 rounded text-xs transition-colors"
					style="background-color: var(--oo-btn-primary-bg); color: var(--oo-btn-primary-fg);"
				>
					{saving ? 'Saving...' : 'Save all to disk'}
				</button>
			</div>

			<div class="space-y-2">
				{#each profileList as profile (profile.name)}
					<button
						on:click={() => selectProfileForEdit(profile.name)}
						class="w-full text-left p-3 rounded-lg transition-colors"
						style="background-color: {selectedModel === profile.name
							? 'var(--oo-info-bg)'
							: 'var(--oo-panel-bg)'};
							border: 1px solid {selectedModel === profile.name
							? 'var(--oo-acc-500)'
							: 'var(--oo-bd-default)'};"
					>
						<div class="flex items-center gap-2 mb-1">
							<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">
								{profile.display_name}
							</span>
							<span class="text-xs font-mono" style="color: var(--oo-fg-tertiary);">{profile.name}</span>
							<span class="px-1.5 py-0.5 rounded text-xs" style={tierBadgeStyle(profile.speed_tier)}>
								{profile.speed_tier}
							</span>
							<span class="px-1.5 py-0.5 rounded text-xs" style={tierBadgeStyle(profile.quality_tier)}>
								{profile.quality_tier}
							</span>
							<span class="ml-auto text-xs" style="color: var(--oo-fg-tertiary);">
								{formatCtx(profile.context_window)} ctx
							</span>
						</div>

						<!-- Task score mini bars (top 5) -->
						{#if profile.task_scores && Object.keys(profile.task_scores).length > 0}
							{@const sortedScores = Object.entries(profile.task_scores).sort((a, b) => b[1] - a[1]).slice(0, 5)}
							<div class="flex gap-1 mt-1.5">
								{#each sortedScores as [task, score]}
									<div class="flex items-center gap-1">
										<span class="text-xs truncate max-w-16" style="color: var(--oo-fg-tertiary);">{task}</span>
										<div class="w-12 h-1.5 rounded-full overflow-hidden" style="background-color: var(--oo-bd-default);">
											<div class="h-full rounded-full" style="width: {scoreBarWidth(score)}; background-color: {scoreColor(score)};" />
										</div>
									</div>
								{/each}
							</div>
						{:else}
							<div class="text-xs italic mt-1" style="color: var(--oo-fg-tertiary);">
								No task scores (uses heuristic scoring)
							</div>
						{/if}
					</button>
				{/each}
			</div>
		</div>

		<!-- ============================================================ -->
		<!-- PROFILE DETAIL / EDITOR                                      -->
		<!-- ============================================================ -->
		{#if selectedProfile}
			<div class="p-4 rounded-lg" style="background-color: var(--oo-panel-bg); border: 1px solid var(--oo-acc-500);">
				<div class="flex items-center justify-between mb-4">
					<h3 class="text-sm font-medium" style="color: var(--oo-fg-primary);">
						{selectedProfile.display_name}
					</h3>
					<div class="flex gap-2">
						<button
							on:click={() => handleAutoDetect(selectedProfile.name)}
							disabled={detecting === selectedProfile.name}
							class="px-2 py-1 rounded text-xs"
							style="background-color: var(--oo-bd-default); color: var(--oo-fg-tertiary);"
						>
							{detecting === selectedProfile.name ? 'Detecting...' : 'Auto-detect'}
						</button>
						<button
							on:click={() => handleDeleteProfile(selectedProfile.name)}
							class="px-2 py-1 rounded text-xs"
							style="background-color: var(--oo-error-bg); color: var(--oo-error);"
						>
							Delete
						</button>
						<button
							on:click={closeDetail}
							class="px-2 py-1 rounded text-xs"
							style="background-color: var(--oo-bd-default); color: var(--oo-fg-tertiary);"
						>
							Close
						</button>
					</div>
				</div>

				<!-- Profile metadata -->
				<div class="grid grid-cols-2 gap-x-4 gap-y-1 text-xs mb-4" style="color: var(--oo-fg-secondary);">
					<span>Context window: <strong>{formatCtx(selectedProfile.context_window)}</strong></span>
					<span>Speed: <strong>{selectedProfile.speed_tier}</strong></span>
					<span>Quality: <strong>{selectedProfile.quality_tier}</strong></span>
					{#if selectedProfile.parameter_count}
						<span>Parameters: <strong>{selectedProfile.parameter_count}</strong></span>
					{/if}
					{#if selectedProfile.family}
						<span>Family: <strong>{selectedProfile.family}</strong></span>
					{/if}
					{#if _isMtpCapable(selectedProfile.name, selectedProfile.family)}
						<span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium"
							style="background-color: var(--oo-acc-900); color: var(--oo-acc-400);"
							title="This model supports Multi-Token Prediction. Support coming in a future update.">
							MTP
						</span>
					{/if}
					{#if selectedProfile.auto_detected}
						<span class="italic" style="color: var(--oo-fg-tertiary);">(auto-detected)</span>
					{/if}
				</div>

				<!-- Capabilities -->
				{#if selectedProfile.capabilities.length > 0}
					<div class="mb-3">
						<div class="text-xs mb-1" style="color: var(--oo-fg-tertiary);">Capabilities</div>
						<div class="flex flex-wrap gap-1">
							{#each selectedProfile.capabilities as cap}
								<span class="px-1.5 py-0.5 rounded text-xs"
									style="background-color: var(--oo-info-bg); color: var(--oo-info);">
									{cap}
								</span>
							{/each}
						</div>
					</div>
				{/if}

				<!-- Task Scores Editor -->
				<div class="mb-3">
					<div class="text-xs mb-2" style="color: var(--oo-fg-tertiary);">
						Task Scores
					</div>
					{#if Object.keys(editingScores).length > 0}
						<div class="space-y-1.5">
							{#each Object.entries(editingScores).sort((a, b) => b[1] - a[1]) as [task, score]}
								<div class="flex items-center gap-2">
									<span class="text-xs font-mono w-28 truncate" style="color: var(--oo-fg-secondary);">
										{task}
									</span>
									<div class="flex-1 h-3 rounded-full overflow-hidden"
										style="background-color: var(--oo-bd-default);">
										<div class="h-full rounded-full transition-all"
											style="width: {scoreBarWidth(score)}; background-color: {scoreColor(score)};" />
									</div>
									<input
										type="range" min="0" max="1" step="0.05"
										bind:value={editingScores[task]}
										class="w-20"
									/>
									<span class="text-xs tabular-nums w-8 text-right" style="color: {scoreColor(score)};">
										{score.toFixed(2)}
									</span>
									<button
										on:click={() => removeTaskScore(task)}
										class="text-xs px-1 rounded"
										style="color: var(--oo-error);"
										title="Remove"
									>x</button>
								</div>
							{/each}
						</div>
					{:else}
						<div class="text-xs italic" style="color: var(--oo-fg-tertiary);">
							No task scores defined. Add some below.
						</div>
					{/if}

					<!-- Add new task score -->
					<div class="flex items-center gap-2 mt-3">
						<input
							type="text"
							bind:value={newTaskType}
							placeholder="task type (e.g. code_python)"
							class="flex-1 px-2 py-1 rounded text-xs"
							style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
							on:keydown={(e) => { if (e.key === 'Enter') addTaskScore(); }}
						/>
						<input
							type="range" min="0" max="1" step="0.05"
							bind:value={newTaskScore}
							class="w-16"
						/>
						<span class="text-xs tabular-nums w-8" style="color: var(--oo-fg-tertiary);">
							{newTaskScore.toFixed(2)}
						</span>
						<button
							on:click={addTaskScore}
							class="px-2 py-1 rounded text-xs"
							style="background-color: var(--oo-btn-primary-bg); color: var(--oo-btn-primary-fg);"
						>
							Add
						</button>
					</div>
				</div>

				<!-- Save scores button -->
				<button
					on:click={handleUpdateScores}
					disabled={saving}
					class="w-full py-2 rounded text-sm font-medium transition-colors"
					style="background-color: var(--oo-btn-primary-bg); color: var(--oo-btn-primary-fg);"
				>
					{saving ? 'Saving...' : 'Save task scores'}
				</button>
			</div>
		{/if}
	{/if}
</div>
