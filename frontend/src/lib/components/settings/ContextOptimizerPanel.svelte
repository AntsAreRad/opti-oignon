<!--
  ContextOptimizerPanel.svelte — S123 Context Window Optimizer settings panel.

  Sections:
  1. Enable/disable toggle
  2. Priority preset selector (balanced / rag_heavy / history_heavy)
  3. Zone weight sliders with live preview
  4. Last optimization report display (stacked bar + trim summary)
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getOptimizerConfig,
		updateOptimizerConfig,
		getOptimizerReports,
		getOptimizerPresets,
	} from '$lib/api/contextOptimizer';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type {
		OptimizerConfigResponse,
		OptimizationReport,
		ZoneReport,
	} from '$lib/api/contextOptimizer';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	let loading = true;
	let error = '';
	let saving = false;

	let available = false;
	let enabled = false;
	let activePreset = 'balanced';
	let presets: Record<string, Record<string, number>> = {};

	// Zone weight sliders (0-100 range for UX, converted to 0-1 for API)
	let systemWeight = 10;
	let projectWeight = 25;
	let historyWeight = 40;
	let userWeight = 10;
	let reserveWeight = 15;

	// Last report
	let lastReport: OptimizationReport | null = null;

	// Collapsible
	let reportOpen = false;

	// Zone display labels and colors (CSS variable names)
	const zoneConfig: Record<string, { label: string; color: string }> = {
		system: { label: 'System', color: 'var(--oo-acc-600)' },
		project: { label: 'RAG / Project', color: 'var(--oo-success)' },
		history: { label: 'History', color: 'var(--oo-warning)' },
		user: { label: 'User', color: 'var(--oo-info)' },
		reserve: { label: 'Reserve', color: 'var(--oo-fg-muted)' },
	};

	// -------------------------------------------------------------------------
	// Lifecycle
	// -------------------------------------------------------------------------

	onMount(loadData);

	async function loadData() {
		loading = true;
		error = '';
		try {
			const [configResp, presetsResp, reportsResp] = await Promise.all([
				getOptimizerConfig(),
				getOptimizerPresets(),
				getOptimizerReports(1),
			]);
			available = configResp.available;
			enabled = configResp.enabled;
			activePreset = configResp.active_preset || 'balanced';
			presets = presetsResp.presets || {};

			// Load slider values from active preset or config
			applyPresetToSliders(activePreset);

			// Load last report
			if (reportsResp.reports && reportsResp.reports.length > 0) {
				lastReport = reportsResp.reports[reportsResp.reports.length - 1];
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load optimizer config';
		} finally {
			loading = false;
		}
	}

	function applyPresetToSliders(presetName: string) {
		const ratios = presets[presetName];
		if (ratios) {
			systemWeight = Math.round((ratios.system_ratio || 0.10) * 100);
			projectWeight = Math.round((ratios.project_ratio || 0.25) * 100);
			historyWeight = Math.round((ratios.history_ratio || 0.40) * 100);
			userWeight = Math.round((ratios.user_ratio || 0.10) * 100);
			reserveWeight = Math.round((ratios.reserve_ratio || 0.15) * 100);
		}
	}

	// -------------------------------------------------------------------------
	// Actions
	// -------------------------------------------------------------------------

	async function toggleEnabled() {
		saving = true;
		try {
			const resp = await updateOptimizerConfig({ enabled: !enabled });
			enabled = resp.enabled;
			toastSuccess(enabled ? 'Context optimizer enabled' : 'Context optimizer disabled');
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to toggle optimizer');
		} finally {
			saving = false;
		}
	}

	async function selectPreset(presetName: string) {
		saving = true;
		try {
			const resp = await updateOptimizerConfig({ active_preset: presetName });
			activePreset = resp.active_preset;
			applyPresetToSliders(presetName);
			toastSuccess(`Priority preset: ${presetName}`);
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to set preset');
		} finally {
			saving = false;
		}
	}

	async function saveCustomRatios() {
		saving = true;
		try {
			const ratios = {
				system_ratio: systemWeight / 100,
				project_ratio: projectWeight / 100,
				history_ratio: historyWeight / 100,
				user_ratio: userWeight / 100,
				reserve_ratio: reserveWeight / 100,
			};
			await updateOptimizerConfig({
				active_preset: 'custom',
				custom_ratios: ratios,
			});
			activePreset = 'custom';
			toastSuccess('Custom zone weights saved');
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to save custom ratios');
		} finally {
			saving = false;
		}
	}

	async function refreshReport() {
		try {
			const resp = await getOptimizerReports(1);
			if (resp.reports && resp.reports.length > 0) {
				lastReport = resp.reports[resp.reports.length - 1];
			}
		} catch {
			// Silent fail
		}
	}

	// -------------------------------------------------------------------------
	// Computed
	// -------------------------------------------------------------------------

	$: totalWeight = systemWeight + projectWeight + historyWeight + userWeight + reserveWeight;
	$: weightBalanced = totalWeight === 100;
	$: weightWarning = totalWeight !== 100
		? `Total is ${totalWeight}% (should be 100%)`
		: '';

	// Stacked bar widths for report
	function zoneBarWidth(zone: ZoneReport, totalWindow: number): string {
		if (totalWindow <= 0) return '0%';
		const pct = (zone.budgeted_tokens / totalWindow) * 100;
		return `${Math.max(1, Math.round(pct))}%`;
	}

	function zoneActualWidth(zone: ZoneReport, totalWindow: number): string {
		if (totalWindow <= 0) return '0%';
		const pct = (zone.actual_tokens / totalWindow) * 100;
		return `${Math.max(0, Math.round(pct))}%`;
	}

	function formatTokens(n: number): string {
		if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
		return String(n);
	}

	function presetLabel(name: string): string {
		switch (name) {
			case 'balanced': return 'Balanced';
			case 'rag_heavy': return 'RAG-Heavy';
			case 'history_heavy': return 'History-Heavy';
			case 'custom': return 'Custom';
			default: return name;
		}
	}

	function presetDescription(name: string): string {
		switch (name) {
			case 'balanced': return 'Default allocation for general use';
			case 'rag_heavy': return 'More space for RAG chunks, less history';
			case 'history_heavy': return 'Prioritize conversation history over RAG';
			default: return '';
		}
	}
</script>

<div class="space-y-5">
	{#if loading}
		<div class="flex items-center gap-2 text-sm py-4" style="color: var(--oo-fg-muted);">
			<div class="w-4 h-4 border-2 rounded-full animate-spin"
				style="border-color: var(--oo-bd-default); border-top-color: var(--oo-acc-400);"></div>
			Loading optimizer...
		</div>
	{:else if error}
		<div class="px-3 py-2 rounded-lg text-sm"
			style="background-color: var(--oo-error-bg); border: 1px solid var(--oo-error-bd); color: var(--oo-error);">
			{error}
			<button on:click={loadData} class="ml-2 underline">Retry</button>
		</div>
	{:else}
		<!-- Section: Enable/Disable -->
		<div class="flex items-center justify-between px-4 py-3 rounded-lg"
			style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);">
			<div>
				<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">
					Context Optimizer
				</span>
				<p class="text-xs mt-0.5" style="color: var(--oo-fg-muted);">
					Unified context budget management with zone priorities
				</p>
			</div>
			<button
				on:click={toggleEnabled}
				disabled={saving || !available}
				class="shrink-0 ml-4 w-11 h-6 rounded-full transition-colors relative disabled:opacity-40"
				style="background-color: {enabled ? 'var(--oo-success)' : 'var(--oo-bg-overlay)'};" aria-label="Toggle context optimizer"
				title="{enabled ? 'Disable' : 'Enable'} context optimizer"
			>
				<span
					class="absolute top-0.5 w-5 h-5 rounded-full transition-all"
					style="background-color: var(--oo-toggle-knob);
						left: {enabled ? '22px' : '2px'};"
				/>
			</button>
		</div>

		{#if !available}
			<p class="text-xs px-1" style="color: var(--oo-warning);">
				Context optimizer module not available. Check backend logs.
			</p>
		{/if}

		{#if available}
			<!-- Section: Priority Presets -->
			<div>
				<h3 class="text-sm font-medium mb-2" style="color: var(--oo-fg-primary);">
					Priority Preset
				</h3>
				<div class="grid grid-cols-3 gap-2">
					{#each Object.keys(presets) as pName}
						{@const isActive = activePreset === pName}
						<button
							on:click={() => selectPreset(pName)}
							disabled={saving}
							class="px-3 py-2.5 rounded-lg text-left transition-all disabled:opacity-50"
							style="background-color: {isActive ? 'var(--oo-acc-900)' : 'var(--oo-bg-elevated)'};
								border: 1.5px solid {isActive ? 'var(--oo-acc-500)' : 'var(--oo-bd-subtle)'};"
						>
							<span class="text-xs font-medium block" style="color: var(--oo-fg-primary);">
								{presetLabel(pName)}
							</span>
							<span class="text-[10px] block mt-0.5" style="color: var(--oo-fg-muted);">
								{presetDescription(pName)}
							</span>
						</button>
					{/each}
				</div>
			</div>

			<!-- Section: Zone Weight Sliders -->
			<div>
				<div class="flex items-center justify-between mb-2">
					<h3 class="text-sm font-medium" style="color: var(--oo-fg-primary);">
						Zone Weights
					</h3>
					{#if !weightBalanced}
						<span class="text-[10px] px-2 py-0.5 rounded-full"
							style="background-color: var(--oo-warning-bg); color: var(--oo-warning);
								border: 1px solid var(--oo-warning-bd);">
							{weightWarning}
						</span>
					{/if}
				</div>

				<!-- Live preview bar -->
				<div class="flex h-3 rounded-full overflow-hidden mb-3"
					style="background-color: var(--oo-bg-overlay);">
					<div style="width: {systemWeight}%; background-color: {zoneConfig.system.color};" title="System: {systemWeight}%"></div>
					<div style="width: {projectWeight}%; background-color: {zoneConfig.project.color};" title="RAG: {projectWeight}%"></div>
					<div style="width: {historyWeight}%; background-color: {zoneConfig.history.color};" title="History: {historyWeight}%"></div>
					<div style="width: {userWeight}%; background-color: {zoneConfig.user.color};" title="User: {userWeight}%"></div>
					<div style="width: {reserveWeight}%; background-color: {zoneConfig.reserve.color};" title="Reserve: {reserveWeight}%"></div>
				</div>

				<div class="space-y-2.5">
					<!-- System -->
					<div class="flex items-center gap-3">
						<span class="w-2.5 h-2.5 rounded-sm shrink-0" style="background-color: {zoneConfig.system.color};"></span>
						<span class="text-xs w-20 shrink-0" style="color: var(--oo-fg-secondary);">System</span>
						<input type="range" min="5" max="30" step="1" bind:value={systemWeight} class="flex-1" />
						<span class="text-xs font-mono w-8 text-right" style="color: var(--oo-fg-tertiary);">{systemWeight}%</span>
					</div>
					<!-- RAG / Project -->
					<div class="flex items-center gap-3">
						<span class="w-2.5 h-2.5 rounded-sm shrink-0" style="background-color: {zoneConfig.project.color};"></span>
						<span class="text-xs w-20 shrink-0" style="color: var(--oo-fg-secondary);">RAG</span>
						<input type="range" min="0" max="50" step="1" bind:value={projectWeight} class="flex-1" />
						<span class="text-xs font-mono w-8 text-right" style="color: var(--oo-fg-tertiary);">{projectWeight}%</span>
					</div>
					<!-- History -->
					<div class="flex items-center gap-3">
						<span class="w-2.5 h-2.5 rounded-sm shrink-0" style="background-color: {zoneConfig.history.color};"></span>
						<span class="text-xs w-20 shrink-0" style="color: var(--oo-fg-secondary);">History</span>
						<input type="range" min="10" max="70" step="1" bind:value={historyWeight} class="flex-1" />
						<span class="text-xs font-mono w-8 text-right" style="color: var(--oo-fg-tertiary);">{historyWeight}%</span>
					</div>
					<!-- User -->
					<div class="flex items-center gap-3">
						<span class="w-2.5 h-2.5 rounded-sm shrink-0" style="background-color: {zoneConfig.user.color};"></span>
						<span class="text-xs w-20 shrink-0" style="color: var(--oo-fg-secondary);">User</span>
						<input type="range" min="5" max="30" step="1" bind:value={userWeight} class="flex-1" />
						<span class="text-xs font-mono w-8 text-right" style="color: var(--oo-fg-tertiary);">{userWeight}%</span>
					</div>
					<!-- Reserve -->
					<div class="flex items-center gap-3">
						<span class="w-2.5 h-2.5 rounded-sm shrink-0" style="background-color: {zoneConfig.reserve.color};"></span>
						<span class="text-xs w-20 shrink-0" style="color: var(--oo-fg-secondary);">Reserve</span>
						<input type="range" min="5" max="40" step="1" bind:value={reserveWeight} class="flex-1" />
						<span class="text-xs font-mono w-8 text-right" style="color: var(--oo-fg-tertiary);">{reserveWeight}%</span>
					</div>
				</div>

				<div class="flex items-center justify-end gap-2 mt-3">
					<button
						on:click={saveCustomRatios}
						disabled={saving || !weightBalanced}
						class="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors disabled:opacity-40"
						style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
					>
						{saving ? 'Saving...' : 'Save Custom Weights'}
					</button>
				</div>
			</div>

			<!-- Section: Last Optimization Report -->
			<div class="rounded-lg overflow-hidden"
				style="border: 1px solid var(--oo-bd-subtle);">
				<button
					on:click={() => { reportOpen = !reportOpen; refreshReport(); }}
					class="w-full flex items-center justify-between px-4 py-3 text-left transition-colors"
					style="background-color: var(--oo-bg-elevated);"
				>
					<div>
						<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">
							Last Optimization Report
						</span>
						{#if lastReport}
							<span class="text-xs ml-2" style="color: var(--oo-fg-muted);">
								{lastReport.model} — {formatTokens(lastReport.total_actual)} / {formatTokens(lastReport.total_window)}
							</span>
						{/if}
					</div>
					<svg class="w-4 h-4 transition-transform {reportOpen ? 'rotate-180' : ''}"
						style="color: var(--oo-fg-muted);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path d="M19 9l-7 7-7-7" />
					</svg>
				</button>

				{#if reportOpen}
					<div class="px-4 py-4 space-y-4" style="border-top: 1px solid var(--oo-bd-subtle);">
						{#if lastReport}
							<!-- Stacked budget bar -->
							<div>
								<p class="text-xs mb-1.5 font-medium" style="color: var(--oo-fg-secondary);">
									Budget allocation ({formatTokens(lastReport.total_window)} window)
								</p>
								<div class="flex h-5 rounded overflow-hidden"
									style="background-color: var(--oo-bg-overlay);">
									{#each lastReport.zones as zone}
										{@const cfg = zoneConfig[zone.zone]}
										{#if cfg}
											<div
												style="width: {zoneBarWidth(zone, lastReport.total_window)};
													background-color: {cfg.color}; opacity: 0.35;"
												title="{cfg.label}: budgeted {formatTokens(zone.budgeted_tokens)}"
											></div>
										{/if}
									{/each}
								</div>
								<!-- Actual usage overlay -->
								<div class="flex h-2 rounded overflow-hidden mt-1"
									style="background-color: var(--oo-bg-overlay);">
									{#each lastReport.zones as zone}
										{@const cfg = zoneConfig[zone.zone]}
										{#if cfg}
											<div
												style="width: {zoneActualWidth(zone, lastReport.total_window)};
													background-color: {cfg.color};"
												title="{cfg.label}: actual {formatTokens(zone.actual_tokens)}"
											></div>
										{/if}
									{/each}
								</div>
								<div class="flex items-center gap-3 mt-1.5 flex-wrap">
									{#each lastReport.zones as zone}
										{@const cfg = zoneConfig[zone.zone]}
										{#if cfg}
											<span class="flex items-center gap-1 text-[10px]" style="color: var(--oo-fg-muted);">
												<span class="w-2 h-2 rounded-sm inline-block" style="background-color: {cfg.color};"></span>
												{cfg.label}: {formatTokens(zone.actual_tokens)}/{formatTokens(zone.budgeted_tokens)}
											</span>
										{/if}
									{/each}
								</div>
							</div>

							<!-- Trim summary -->
							{#if lastReport.total_trimmed > 0}
								<div class="px-3 py-2 rounded-lg text-xs"
									style="background-color: var(--oo-warning-bg); border: 1px solid var(--oo-warning-bd); color: var(--oo-warning);">
									Trimmed {formatTokens(lastReport.total_trimmed)} tokens total.
									{#each lastReport.zones.filter(z => z.trimmed_tokens > 0) as zone}
										{@const cfg = zoneConfig[zone.zone]}
										{cfg?.label || zone.zone}: {formatTokens(zone.trimmed_tokens)}t ({zone.strategy}).
									{/each}
								</div>
							{/if}

							{#if lastReport.overflow}
								<div class="px-3 py-2 rounded-lg text-xs"
									style="background-color: var(--oo-error-bg); border: 1px solid var(--oo-error-bd); color: var(--oo-error);">
									Emergency truncation was needed for this inference.
								</div>
							{/if}

							<!-- Metadata -->
							<div class="flex items-center gap-4 text-[10px]" style="color: var(--oo-fg-muted);">
								<span>Preset: {lastReport.preset_used}</span>
								<span>Duration: {lastReport.duration_ms.toFixed(1)}ms</span>
								<span>Model: {lastReport.model}</span>
							</div>
						{:else}
							<p class="text-xs" style="color: var(--oo-fg-muted);">
								No optimization report yet. Reports are generated after each inference when the optimizer is enabled.
							</p>
						{/if}
					</div>
				{/if}
			</div>
		{/if}
	{/if}
</div>
