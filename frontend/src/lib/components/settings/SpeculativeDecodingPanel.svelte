<!--
  SpeculativeDecodingPanel.svelte -- S110 + S111 Speculative Decoding settings panel.

  Collapsible panel for configuring llama.cpp native speculative decoding.
  Sections:
  1. Enable/disable toggle with backend requirement warning
  2. Draft model selector (filtered to compatible models)
  3. Draft parameter controls (draft_max, draft_min, GPU layers)
  4. VRAM budget indicator
  5. Acceptance rate display with per-request mini-chart (S111)
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		getSpeculativeDecodingStatus,
		updateSpeculativeDecodingConfig,
		resetSpeculativeStats,
	} from '$lib/api/speculativeDecoding';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type {
		SpeculativeDecodingStatus,
		SpeculativeDecodingConfig,
		SpeculativeDecodingStats,
	} from '$lib/types';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	let loading = true;
	let error = '';
	let open = false;

	let available = false;
	let backendRequired = 'llama_cpp';
	let stats: SpeculativeDecodingStats | null = null;

	// Editable config fields
	let localEnabled = false;
	let localDraftModel = '';
	let localDraftMax = 16;
	let localDraftMin = 5;
	let localDraftGpuLayers = 99;
	let localAutoSelect = true;
	let saving = false;

	// S111: Acceptance rate history for mini-chart
	interface AcceptanceRecord {
		timestamp: number;
		draft_tokens: number;
		accepted_tokens: number;
		acceptance_rate: number;
		speedup_factor: number;
		request_id: string;
	}

	let acceptanceHistory: AcceptanceRecord[] = [];
	let rollingRate = 0;
	let historyPollTimer: ReturnType<typeof setInterval> | null = null;

	/** Maximum acceptance rate in history (for chart scaling). */
	$: maxRate = Math.max(0.1, ...acceptanceHistory.map(r => r.acceptance_rate));

	// -------------------------------------------------------------------------
	// Load
	// -------------------------------------------------------------------------

	onMount(loadData);

	onDestroy(() => {
		if (historyPollTimer) clearInterval(historyPollTimer);
	});

	async function loadData() {
		loading = true;
		error = '';
		try {
			const status: SpeculativeDecodingStatus = await getSpeculativeDecodingStatus();
			available = status.available;
			backendRequired = status.backend_required;
			stats = status.stats;

			localEnabled = status.config.enabled;
			localDraftModel = status.config.draft_model;
			localDraftMax = status.config.draft_max;
			localDraftMin = status.config.draft_min;
			localDraftGpuLayers = status.config.draft_gpu_layers;
			localAutoSelect = status.config.auto_select_draft;

			// S111: Fetch acceptance history if enabled
			if (localEnabled && stats && stats.total_runs > 0) {
				await fetchAcceptanceHistory();
				startHistoryPolling();
			}
		} catch (e) {
			error = `Failed to load speculative decoding status: ${e}`;
		} finally {
			loading = false;
		}
	}

	// -------------------------------------------------------------------------
	// S111: Acceptance history
	// -------------------------------------------------------------------------

	async function fetchAcceptanceHistory() {
		try {
			const res = await fetch('/api/speculative-decoding/acceptance-history?last_n=50');
			if (!res.ok) return;
			const data = await res.json();
			acceptanceHistory = data.history || [];
			rollingRate = data.rolling_acceptance_rate || 0;
		} catch {
			// Silently ignore fetch errors
		}
	}

	function startHistoryPolling() {
		if (historyPollTimer) return;
		historyPollTimer = setInterval(fetchAcceptanceHistory, 5000);
	}

	// -------------------------------------------------------------------------
	// Actions
	// -------------------------------------------------------------------------

	async function handleSave() {
		saving = true;
		try {
			const update: Partial<SpeculativeDecodingConfig> = {
				enabled: localEnabled,
				draft_model: localDraftModel,
				draft_max: localDraftMax,
				draft_min: localDraftMin,
				draft_gpu_layers: localDraftGpuLayers,
				auto_select_draft: localAutoSelect,
			};
			await updateSpeculativeDecodingConfig(update);
			toastSuccess('Speculative decoding configuration saved');
			await loadData();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to save config');
		} finally {
			saving = false;
		}
	}

	async function handleResetStats() {
		try {
			await resetSpeculativeStats();
			toastSuccess('Acceptance stats cleared');
			await loadData();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to reset stats');
		}
	}

	function formatRate(rate: number): string {
		return (rate * 100).toFixed(1) + '%';
	}

	function formatSpeedup(factor: number): string {
		return factor.toFixed(1) + 'x';
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
				<path d="M13 10V3L4 14h7v7l9-11h-7z" stroke-linecap="round" stroke-linejoin="round" />
			</svg>
			<div>
				<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">
					Speculative Decoding
				</span>
				<span class="text-xs ml-2" style="color: var(--oo-fg-muted);">
					{#if localEnabled}
						Enabled
					{:else}
						Disabled
					{/if}
					&mdash; llama.cpp only
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
				<!-- Backend warning -->
				<div class="px-3 py-2 rounded-lg text-xs flex items-start gap-2"
					style="background-color: var(--oo-warning-bg); border: 1px solid var(--oo-warning-bd); color: var(--oo-warning);">
					<svg class="w-3.5 h-3.5 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
					</svg>
					<span>
						Requires the llama.cpp backend. Not available with Ollama.
						Pair a small draft model with your main model for 2-5x token generation speedup with zero quality loss.
					</span>
				</div>

				<!-- Enable toggle -->
				<div class="flex items-center justify-between">
					<div>
						<span class="text-sm" style="color: var(--oo-fg-primary);">Enable speculative decoding</span>
						<p class="text-xs mt-0.5" style="color: var(--oo-fg-muted);">
							Draft model generates candidate tokens verified by the main model.
						</p>
					</div>
					<button
						on:click={() => { localEnabled = !localEnabled; }}
						class="shrink-0 ml-4 w-11 h-6 rounded-full transition-colors relative"
						style="background-color: {localEnabled ? 'var(--oo-success)' : 'var(--oo-bg-overlay)'};"
					>
						<span
							class="absolute top-0.5 w-5 h-5 rounded-full transition-all"
							style="background-color: var(--oo-toggle-knob);
								left: {localEnabled ? '22px' : '2px'};"
						/>
					</button>
				</div>

				{#if localEnabled}
					<!-- Draft model -->
					<div>
						<label class="block text-xs font-medium mb-1" style="color: var(--oo-fg-secondary);">
							Draft Model
						</label>
						<input
							type="text"
							bind:value={localDraftModel}
							class="w-full px-3 py-2 rounded-lg text-sm font-mono"
							style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
							placeholder="Path to draft GGUF (e.g. llama-3.2-1b-Q4_K_M.gguf)"
						/>
						<p class="text-xs mt-1" style="color: var(--oo-fg-muted);">
							Use a small model from the same family as your main model (e.g. 1B or 3B variant).
						</p>
					</div>

					<!-- Draft parameters -->
					<div class="grid grid-cols-3 gap-3">
						<div>
							<label class="block text-xs font-medium mb-1" style="color: var(--oo-fg-secondary);">
								Draft Max
							</label>
							<input
								type="number"
								bind:value={localDraftMax}
								min="1"
								max="64"
								class="w-full px-3 py-2 rounded-lg text-sm"
								style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
							/>
							<p class="text-[10px] mt-0.5" style="color: var(--oo-fg-muted);">Max draft tokens</p>
						</div>
						<div>
							<label class="block text-xs font-medium mb-1" style="color: var(--oo-fg-secondary);">
								Draft Min
							</label>
							<input
								type="number"
								bind:value={localDraftMin}
								min="1"
								max="32"
								class="w-full px-3 py-2 rounded-lg text-sm"
								style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
							/>
							<p class="text-[10px] mt-0.5" style="color: var(--oo-fg-muted);">Min draft tokens</p>
						</div>
						<div>
							<label class="block text-xs font-medium mb-1" style="color: var(--oo-fg-secondary);">
								GPU Layers
							</label>
							<input
								type="number"
								bind:value={localDraftGpuLayers}
								min="-1"
								max="999"
								class="w-full px-3 py-2 rounded-lg text-sm"
								style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
							/>
							<p class="text-[10px] mt-0.5" style="color: var(--oo-fg-muted);">Draft model GPU layers</p>
						</div>
					</div>

					<!-- Auto-select toggle -->
					<label class="flex items-center gap-3 cursor-pointer">
						<input type="checkbox" bind:checked={localAutoSelect} class="rounded" />
						<div>
							<span class="text-sm" style="color: var(--oo-fg-primary);">Auto-select draft model</span>
							<p class="text-xs" style="color: var(--oo-fg-muted);">
								Automatically pick the best compatible draft from installed models.
							</p>
						</div>
					</label>

					<!-- Acceptance stats -->
					{#if stats && stats.total_runs > 0}
						<div class="rounded-lg px-3 py-3"
							style="background-color: var(--oo-bg-overlay); border: 1px solid var(--oo-bd-subtle);">
							<div class="text-xs font-medium mb-2" style="color: var(--oo-fg-secondary);">
								Acceptance Statistics
							</div>
							<div class="grid grid-cols-2 gap-2 text-xs">
								<div>
									<span style="color: var(--oo-fg-muted);">Overall rate:</span>
									<span class="font-mono font-medium" style="color: var(--oo-fg-primary);">
										{formatRate(stats.overall_acceptance_rate)}
									</span>
								</div>
								<div>
									<span style="color: var(--oo-fg-muted);">Last run rate:</span>
									<span class="font-mono font-medium" style="color: var(--oo-fg-primary);">
										{formatRate(stats.last_acceptance_rate)}
									</span>
								</div>
								<div>
									<span style="color: var(--oo-fg-muted);">Last speedup:</span>
									<span class="font-mono font-medium" style="color: var(--oo-acc-400);">
										{formatSpeedup(stats.last_speedup_factor)}
									</span>
								</div>
								<div>
									<span style="color: var(--oo-fg-muted);">Total runs:</span>
									<span class="font-mono font-medium" style="color: var(--oo-fg-primary);">
										{stats.total_runs}
									</span>
								</div>
							</div>
								{#if rollingRate > 0}
									<div class="mt-1">
										<span style="color: var(--oo-fg-muted);">Rolling rate (10):</span>
										<span class="font-mono font-medium" style="color: var(--oo-acc-400);">
											{formatRate(rollingRate)}
										</span>
									</div>
								{/if}

							<!-- S111: Acceptance rate mini-chart -->
							{#if acceptanceHistory.length > 1}
								<div class="mt-3">
									<div class="text-[10px] font-medium mb-1" style="color: var(--oo-fg-muted);">
										Per-request acceptance rate (last {acceptanceHistory.length} requests)
									</div>
									<div style="height: 2rem; overflow: hidden; border-radius: 0.25rem;">
										<svg
											viewBox="0 0 {acceptanceHistory.length} 30"
											preserveAspectRatio="none"
											style="width: 100%; height: 100%; display: block;"
										>
											<!-- Reference line at 100% -->
											<line x1="0" y1="2" x2={acceptanceHistory.length} y2="2"
												stroke="var(--oo-bd-subtle)" stroke-width="0.3"
												stroke-dasharray="2,2" />
											<!-- Reference line at 50% -->
											<line x1="0" y1="16" x2={acceptanceHistory.length} y2="16"
												stroke="var(--oo-bd-subtle)" stroke-width="0.2"
												stroke-dasharray="1,2" />
											<!-- Bars -->
											{#each acceptanceHistory as record, i}
												{@const barHeight = (record.acceptance_rate / maxRate) * 26}
												{@const barColor = record.acceptance_rate >= 0.7
													? 'var(--oo-success)'
													: record.acceptance_rate >= 0.4
														? 'var(--oo-warning)'
														: 'var(--oo-error)'}
												<rect
													x={i}
													y={30 - barHeight}
													width="0.75"
													height={barHeight}
													fill={barColor}
													opacity="0.7"
												/>
											{/each}
										</svg>
									</div>
									<div class="flex justify-between text-[9px] mt-0.5" style="color: var(--oo-fg-muted);">
										<span>Oldest</span>
										<span>Latest</span>
									</div>
								</div>
							{/if}

							<button
								on:click={handleResetStats}
								class="mt-2 text-xs underline"
								style="color: var(--oo-fg-muted);"
							>
								Clear stats
							</button>
						</div>
					{/if}
				{/if}

				<!-- Save button -->
				<div class="flex justify-end pt-2">
					<button
						on:click={handleSave}
						disabled={saving}
						class="px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
						style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
					>
						{saving ? 'Saving...' : 'Save'}
					</button>
				</div>
			{/if}
		</div>
	{/if}
</div>
