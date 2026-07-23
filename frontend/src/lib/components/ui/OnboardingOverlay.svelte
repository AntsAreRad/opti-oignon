<!--
  OnboardingOverlay.svelte
  Full-screen overlay shown on first run (user_initialized === false).
  Detects installed Ollama models, recommends a system preset,
  and allows one-click apply. Dismissible after apply or skip.
  Migrated to the shared <Modal> primitive (native dialog focus
  trap + Escape). Backdrop click is disabled; Escape maps to Skip.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { Modal } from '$lib/ds';
	import type {
		SystemPresetInfo,
		SystemPresetDetectResponse,
	} from '$lib/types';
	import {
		getOnboardingState,
		listSystemPresets,
		detectAndRecommend,
		applySystemPreset,
	} from '$lib/api/systemPresets';

	let visible = false;
	let step: 'loading' | 'ready' | 'applying' | 'done' | 'error' = 'loading';

	let presets: SystemPresetInfo[] = [];
	let detection: SystemPresetDetectResponse | null = null;
	let selectedPresetId = '';
	let applyResult: { preset_name: string; selected_model: string | null; warnings: string[] } | null = null;
	let errorMsg = '';

	const MAX_RETRIES = 3;
	const RETRY_DELAY_MS = 2000;

	onMount(async () => {
		for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
			try {
				const state = await getOnboardingState();
				if (state.user_initialized) {
					visible = false;
					return;
				}
				visible = true;
				await loadData();
				return;
			} catch {
			// BUG-10: Backend may not be ready yet (404 / connection error).
				// Retry with delay before giving up.
				if (attempt < MAX_RETRIES - 1) {
					await new Promise((r) => setTimeout(r, RETRY_DELAY_MS));
				} else {
					// All retries exhausted — backend not available
					visible = false;
				}
			}
		}
	});

	async function loadData() {
		step = 'loading';
		try {
			const [presetsResp, detectResp] = await Promise.all([
				listSystemPresets(),
				detectAndRecommend(),
			]);
			presets = presetsResp.presets;
			detection = detectResp;
			selectedPresetId = detectResp.recommended_preset;
			step = 'ready';
		} catch (e) {
			errorMsg = e instanceof Error ? e.message : 'Failed to load system data';
			step = 'error';
		}
	}

	async function handleApply() {
		if (!selectedPresetId) return;
		step = 'applying';
		errorMsg = '';
		try {
			const result = await applySystemPreset(selectedPresetId);
			if (result.applied) {
				applyResult = {
					preset_name: result.preset_name,
					selected_model: result.selected_model,
					warnings: result.warnings,
				};
				step = 'done';
			} else {
				errorMsg = result.error || 'Failed to apply preset';
				step = 'error';
			}
		} catch (e) {
			errorMsg = e instanceof Error ? e.message : 'Failed to apply preset';
			step = 'error';
		}
	}

	function handleSkip() {
		visible = false;
	}

	function handleClose() {
		visible = false;
		// Reload page to pick up new configs
		window.location.reload();
	}

	function presetIconSvg(icon: string): string {
		switch (icon) {
			case 'leaf': return 'M17 8C8 10 5.9 16.17 3.82 21.34l1.89.66L7 18.5C9 15 12 12 17 8z M20.5 3.5C17 7 13 9 10 10l1 2c3-1 6.5-3.5 9.5-6.5';
			case 'scale': return 'M12 3v18m-7-4l7-10 7 10M5 17h14';
			case 'zap': return 'M13 2L3 14h9l-1 10 10-12h-9l1-10z';
			default: return 'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5';
		}
	}
</script>

<Modal
	open={visible}
	variant="center"
	size="lg"
	title="Welcome to Opti-Oignon"
	closeOnBackdrop={false}
	onClose={handleSkip}
>
	<!-- Branded intro -->
	<div class="ob-intro">
		<div class="ob-logo-ring">
			<img src="/bousier-oignon.png" alt="Opti-Oignon" class="ob-logo oo-logo-adaptive" />
		</div>
		<p class="ob-tagline">Let's configure your setup in one click.</p>
	</div>

	<div class="ob-body">
		{#if step === 'loading'}
			<div class="flex flex-col items-center gap-3 py-6">
				<div class="w-6 h-6 border-2 rounded-full animate-spin"
					style="border-color: var(--oo-bd-default); border-top-color: var(--oo-acc-400);"></div>
				<span class="text-sm" style="color: var(--oo-fg-tertiary);">
					Scanning installed models...
				</span>
			</div>
		{:else if step === 'error'}
			<div class="px-4 py-3 rounded-lg text-sm"
				style="background-color: var(--oo-error-bg); border: 1px solid var(--oo-error-bd); color: var(--oo-error);">
				{errorMsg}
			</div>
			<div class="flex justify-center gap-3">
				<button on:click={loadData}
					class="px-4 py-2 rounded-lg text-sm font-medium"
					style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-secondary);
						border: 1px solid var(--oo-bd-default);">
					Retry
				</button>
				<button on:click={handleSkip}
					class="px-4 py-2 rounded-lg text-sm"
					style="color: var(--oo-fg-muted);">
					Skip for now
				</button>
			</div>
		{:else if step === 'ready'}
			<!-- Detected models summary -->
			{#if detection}
				<div class="rounded-lg px-4 py-3"
					style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);">
					<div class="flex items-center gap-2 mb-2">
						<svg class="w-4 h-4" style="color: var(--oo-acc-400);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
							<circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
						</svg>
						<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">
							{detection.models.length} model{detection.models.length !== 1 ? 's' : ''} detected
						</span>
					</div>
					{#if detection.models.length > 0}
						<div class="flex flex-wrap gap-1.5 max-h-28 overflow-y-auto">
							{#each detection.models as m}
								<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs"
									style="background-color: var(--oo-bg-overlay); color: var(--oo-fg-secondary);
										border: 1px solid var(--oo-bd-subtle);">
									<span class="font-mono">{m.name}</span>
									{#if m.parameter_count_b > 0}
										<span style="color: var(--oo-fg-muted);">{m.parameter_count_b}B</span>
									{/if}
								</span>
							{/each}
						</div>
					{:else}
						<p class="text-xs" style="color: var(--oo-fg-muted);">
							No models found. Install models with <code class="font-mono px-1 py-0.5 rounded"
								style="background-color: var(--oo-bg-overlay);">ollama pull</code> first, or pick Minimal.
						</p>
					{/if}
				</div>
			{/if}

			<!-- Preset selector -->
			<div class="space-y-2">
				<p class="text-xs font-medium uppercase tracking-wide"
					style="color: var(--oo-fg-muted);">
					Choose a configuration preset
				</p>
				{#each presets as preset (preset.id)}
					<button
						on:click={() => { selectedPresetId = preset.id; }}
						class="w-full text-left px-4 py-3 rounded-lg transition-all"
						style="background-color: {selectedPresetId === preset.id ? 'var(--oo-acc-900)' : 'var(--oo-bg-elevated)'};
							border: 1.5px solid {selectedPresetId === preset.id ? 'var(--oo-acc-500)' : 'var(--oo-bd-subtle)'};
							{selectedPresetId === preset.id ? 'box-shadow: 0 0 12px var(--oo-msg-user-bg);' : ''}"
					>
						<div class="flex items-center gap-3">
							<!-- Icon -->
							<div class="w-9 h-9 rounded-lg flex items-center justify-center shrink-0"
								style="background-color: {selectedPresetId === preset.id ? 'var(--oo-acc-600)' : 'var(--oo-bg-overlay)'};">
								<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke-width="1.8"
									stroke="{selectedPresetId === preset.id ? 'var(--oo-acc-50)' : 'var(--oo-fg-tertiary)'}">
									<path d="{presetIconSvg(preset.icon)}" />
								</svg>
							</div>
							<div class="flex-1 min-w-0">
								<div class="flex items-center gap-2">
									<span class="text-sm font-medium"
										style="color: var(--oo-fg-primary);">
										{preset.name}
									</span>
									{#if detection?.recommended_preset === preset.id}
										<span class="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
											style="background-color: var(--oo-success-bg); color: var(--oo-success);
												border: 1px solid var(--oo-success-bd);">
											Recommended
										</span>
									{/if}
									<span class="text-[10px] ml-auto"
										style="color: var(--oo-fg-muted);">
										{preset.recommended_ram_gb}+ GB RAM
									</span>
								</div>
								<p class="text-xs mt-0.5 line-clamp-2"
									style="color: var(--oo-fg-tertiary);">
									{preset.description}
								</p>
							</div>
						</div>
					</button>
				{/each}
			</div>

			{#if detection?.reason}
				<p class="text-xs italic" style="color: var(--oo-fg-muted);">
					{detection.reason}
				</p>
			{/if}

		{:else if step === 'applying'}
			<div class="flex flex-col items-center gap-3 py-6">
				<div class="w-6 h-6 border-2 rounded-full animate-spin"
					style="border-color: var(--oo-bd-default); border-top-color: var(--oo-acc-400);"></div>
				<span class="text-sm" style="color: var(--oo-fg-tertiary);">
					Applying configuration...
				</span>
			</div>

		{:else if step === 'done'}
			<div class="text-center py-4 space-y-3">
				<div class="mx-auto w-12 h-12 rounded-full flex items-center justify-center"
					style="background-color: var(--oo-success-bg); border: 1px solid var(--oo-success-bd);">
					<svg class="w-6 h-6" style="color: var(--oo-success);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
						<path d="M5 13l4 4L19 7" stroke-linecap="round" stroke-linejoin="round" />
					</svg>
				</div>
				<div>
					<p class="text-sm font-medium" style="color: var(--oo-fg-primary);">
						{applyResult?.preset_name} preset applied
					</p>
					{#if applyResult?.selected_model}
						<p class="text-xs mt-1" style="color: var(--oo-fg-tertiary);">
							Default model: <span class="font-mono">{applyResult.selected_model}</span>
						</p>
					{/if}
				</div>
				{#if applyResult?.warnings && applyResult.warnings.length > 0}
					<div class="text-left px-4 py-2 rounded-lg text-xs"
						style="background-color: var(--oo-warning-bg); border: 1px solid var(--oo-warning-bd); color: var(--oo-warning);">
						{#each applyResult.warnings as w}
							<p>{w}</p>
						{/each}
					</div>
				{/if}
			</div>
		{/if}
	</div>

	<svelte:fragment slot="footer">
		{#if step === 'ready'}
			<button on:click={handleSkip}
				class="text-sm px-3 py-1.5 rounded-lg transition-colors"
				style="color: var(--oo-fg-muted); margin-right: auto;"
				title="Skip and configure manually later in Settings">
				Skip
			</button>
			<button on:click={handleApply}
				disabled={!selectedPresetId}
				class="px-5 py-2 rounded-lg text-sm font-medium transition-all disabled:opacity-40"
				style="background-color: var(--oo-acc-500); color: var(--oo-acc-50);
					{selectedPresetId ? 'box-shadow: 0 2px 12px var(--oo-msg-user-bd);' : ''}">
				Apply {presets.find(p => p.id === selectedPresetId)?.name ?? ''} Preset
			</button>
		{:else if step === 'done'}
			<button on:click={handleClose}
				class="px-5 py-2 rounded-lg text-sm font-medium"
				style="background-color: var(--oo-acc-500); color: var(--oo-acc-50);
					box-shadow: 0 2px 12px var(--oo-msg-user-bd);">
				Get Started
			</button>
		{:else if step === 'applying'}
			<span class="text-xs" style="color: var(--oo-fg-muted);">Please wait...</span>
		{/if}
	</svelte:fragment>
</Modal>

<style>
	.ob-intro {
		text-align: center;
		margin-bottom: var(--oo-space-5);
	}
	.ob-logo-ring {
		margin: 0 auto var(--oo-space-3);
		width: 6rem;
		height: 6rem;
		border-radius: var(--oo-radius-full);
		display: flex;
		align-items: center;
		justify-content: center;
		background: radial-gradient(circle, var(--oo-acc-500) 0%, var(--oo-acc-700) 100%);
		box-shadow: 0 0 24px var(--oo-acc-400);
	}
	.ob-logo {
		width: 4rem;
		height: 4rem;
		object-fit: contain;
	}
	.ob-tagline {
		margin: 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-tertiary);
	}
	.ob-body {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-4);
	}
</style>
