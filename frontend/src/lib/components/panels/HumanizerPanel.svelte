<!--
  HumanizerPanel.svelte -- S86 Humanizer settings panel.

  Sections:
  1. Enable/disable toggle
  2. Mode selector (rewrite / logprobs / hybrid)
  3. Intensity selector (light / moderate / heavy)
  4. Formality selector (casual / neutral / formal)
  5. Rewrite model override
  6. Banned phrases editor
  7. Feedback statistics summary
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getHumanizerConfig,
		updateHumanizerConfig,
		getHumanizerStats,
	} from '$lib/api/humanizer';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type { HumanizerConfigResponse, HumanizerStatsResponse } from '$lib/types';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	let loading = true;
	let error = '';

	let config: HumanizerConfigResponse | null = null;
	let localEnabled = false;
	let localMode = 'rewrite';
	let localIntensity = 'moderate';
	let localFormality = 'neutral';
	let localRewriteModel = '';
	let localMaxLength = 8000;
	let localBannedPhrases = '';
	let saving = false;

	// Stats
	let stats: HumanizerStatsResponse | null = null;
	let statsLoading = false;

	const modes = [
		{ value: 'rewrite', label: 'LLM Rewrite', desc: 'Prompt-based naturalness rewrite' },
		{ value: 'logprobs', label: 'Rule-based', desc: 'Token-level vocabulary and filler fixes' },
		{ value: 'hybrid', label: 'Hybrid', desc: 'Rule-based cleanup then LLM rewrite' },
	];

	const intensities = [
		{ value: 'light', label: 'Light', desc: 'Minimal changes' },
		{ value: 'moderate', label: 'Moderate', desc: 'Balanced naturalness' },
		{ value: 'heavy', label: 'Heavy', desc: 'Substantial rewrite' },
	];

	const formalities = [
		{ value: 'casual', label: 'Casual', desc: 'Contractions everywhere' },
		{ value: 'neutral', label: 'Neutral', desc: 'Natural mix' },
		{ value: 'formal', label: 'Formal', desc: 'No contractions' },
	];

	// -------------------------------------------------------------------------
	// Load
	// -------------------------------------------------------------------------

	onMount(loadData);

	async function loadData() {
		loading = true;
		error = '';
		try {
			config = await getHumanizerConfig();
			localEnabled = config.enabled;
			localMode = config.mode;
			localIntensity = config.intensity;
			localFormality = config.formality;
			localRewriteModel = config.rewrite_model || '';
			localMaxLength = config.max_input_length;
			localBannedPhrases = (config.banned_phrases || []).join('\n');
		} catch (e) {
			error = `Failed to load humanizer config: ${e}`;
		} finally {
			loading = false;
		}
		loadStats();
	}

	async function loadStats() {
		statsLoading = true;
		try {
			stats = await getHumanizerStats();
		} catch {
			// Stats may not be available yet
		} finally {
			statsLoading = false;
		}
	}

	// -------------------------------------------------------------------------
	// Save
	// -------------------------------------------------------------------------

	async function save() {
		saving = true;
		try {
			const phrases = localBannedPhrases
				.split('\n')
				.map((p) => p.trim())
				.filter((p) => p.length > 0);

			config = await updateHumanizerConfig({
				enabled: localEnabled,
				mode: localMode,
				intensity: localIntensity,
				formality: localFormality,
				rewrite_model: localRewriteModel || null,
				max_input_length: localMaxLength,
				banned_phrases: phrases,
			});
			toastSuccess('Humanizer configuration saved');
		} catch (e) {
			toastError(`Failed to save: ${e}`);
		} finally {
			saving = false;
		}
	}

	async function toggleEnabled() {
		localEnabled = !localEnabled;
		saving = true;
		try {
			config = await updateHumanizerConfig({ enabled: localEnabled });
			toastSuccess(localEnabled ? 'Humanizer enabled' : 'Humanizer disabled');
		} catch (e) {
			localEnabled = !localEnabled;
			toastError(`Failed to toggle: ${e}`);
		} finally {
			saving = false;
		}
	}
</script>

{#if loading}
	<div class="flex items-center gap-2 py-4">
		<div class="w-4 h-4 rounded-full animate-pulse" style="background-color: var(--oo-fg-muted);"></div>
		<span class="text-sm" style="color: var(--oo-fg-muted);">Loading humanizer config...</span>
	</div>
{:else if error}
	<div class="rounded-lg px-4 py-3 text-sm" style="background-color: var(--oo-error-bg); color: var(--oo-error);">
		{error}
	</div>
{:else}
	<div class="space-y-5">

		<!-- Enable toggle -->
		<div class="flex items-center justify-between">
			<div>
				<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">Enable Humanizer</span>
				<p class="text-xs" style="color: var(--oo-fg-muted);">Post-process LLM output for more natural language</p>
			</div>
			<button
				on:click={toggleEnabled}
				class="relative w-10 h-5 rounded-full transition-colors"
				style="background-color: {localEnabled ? 'var(--oo-acc-500)' : 'var(--oo-bg-overlay)'};"
				aria-label="Toggle humanizer"
				aria-pressed={localEnabled}
			>
				<span
					class="absolute top-0.5 left-0.5 w-4 h-4 rounded-full transition-transform"
					style="background-color: var(--oo-toggle-knob); transform: translateX({localEnabled ? '20px' : '0'});"
				></span>
			</button>
		</div>

		<!-- Mode -->
		<div>
			<label class="block text-sm font-medium mb-1.5" style="color: var(--oo-fg-primary);">Mode</label>
			<div class="flex gap-2">
				{#each modes as m}
					<button
						on:click={() => { localMode = m.value; }}
						class="flex-1 px-3 py-2 rounded-lg text-xs text-center transition-colors"
						style="{localMode === m.value
							? 'background-color: var(--oo-msg-user-bg); color: var(--oo-acc-300); border: 1px solid var(--oo-acc-400);'
							: 'background-color: var(--oo-bg-elevated); color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);'}"
						title={m.desc}
					>
						{m.label}
					</button>
				{/each}
			</div>
		</div>

		<!-- Intensity -->
		<div>
			<label class="block text-sm font-medium mb-1.5" style="color: var(--oo-fg-primary);">Intensity</label>
			<div class="flex gap-2">
				{#each intensities as lvl}
					<button
						on:click={() => { localIntensity = lvl.value; }}
						class="flex-1 px-3 py-2 rounded-lg text-xs text-center transition-colors"
						style="{localIntensity === lvl.value
							? 'background-color: var(--oo-msg-user-bg); color: var(--oo-acc-300); border: 1px solid var(--oo-acc-400);'
							: 'background-color: var(--oo-bg-elevated); color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);'}"
						title={lvl.desc}
					>
						{lvl.label}
					</button>
				{/each}
			</div>
		</div>

		<!-- Formality -->
		<div>
			<label class="block text-sm font-medium mb-1.5" style="color: var(--oo-fg-primary);">Formality</label>
			<div class="flex gap-2">
				{#each formalities as f}
					<button
						on:click={() => { localFormality = f.value; }}
						class="flex-1 px-3 py-2 rounded-lg text-xs text-center transition-colors"
						style="{localFormality === f.value
							? 'background-color: var(--oo-msg-user-bg); color: var(--oo-acc-300); border: 1px solid var(--oo-acc-400);'
							: 'background-color: var(--oo-bg-elevated); color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);'}"
						title={f.desc}
					>
						{f.label}
					</button>
				{/each}
			</div>
		</div>

		<!-- Rewrite model -->
		<div>
			<label class="block text-sm font-medium mb-1.5" style="color: var(--oo-fg-primary);"
				>Rewrite Model</label
			>
			<input
				type="text"
				bind:value={localRewriteModel}
				class="w-full px-3 py-2 rounded-lg text-sm font-mono"
				style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
				placeholder="Leave empty to use current model"
			/>
			<p class="text-xs mt-1" style="color: var(--oo-fg-muted);">
				Model for LLM rewrite pass. Empty uses the active conversation model.
			</p>
		</div>

		<!-- Max input length -->
		<div>
			<label class="block text-sm font-medium mb-1.5" style="color: var(--oo-fg-primary);"
				>Max Input Length</label
			>
			<input
				type="number"
				bind:value={localMaxLength}
				min={100}
				max={50000}
				step={500}
				class="w-32 px-3 py-2 rounded-lg text-sm"
				style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
			/>
			<span class="text-xs ml-2" style="color: var(--oo-fg-muted);">characters</span>
		</div>

		<!-- Banned phrases -->
		<div>
			<label class="block text-sm font-medium mb-1.5" style="color: var(--oo-fg-primary);"
				>Banned Phrases</label
			>
			<textarea
				bind:value={localBannedPhrases}
				rows="5"
				class="w-full px-3 py-2 rounded-lg text-sm font-mono resize-y"
				style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
				placeholder="One phrase per line"
			></textarea>
			<p class="text-xs mt-1" style="color: var(--oo-fg-muted);">
				Formulaic phrases to strip from LLM output (one per line).
			</p>
		</div>

		<!-- Save button -->
		<div>
			<button
				on:click={save}
				disabled={saving}
				class="px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
				style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
			>
				{saving ? 'Saving...' : 'Save Configuration'}
			</button>
		</div>

		<!-- Stats summary -->
		{#if stats && stats.total_ratings > 0}
			<div class="mt-4 rounded-lg p-4" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);">
				<h3 class="text-sm font-medium mb-2" style="color: var(--oo-fg-primary);">Feedback Statistics</h3>
				<div class="grid grid-cols-2 gap-3 text-xs">
					<div>
						<span style="color: var(--oo-fg-muted);">Total ratings</span>
						<span class="block font-medium" style="color: var(--oo-fg-primary);">{stats.total_ratings}</span>
					</div>
					<div>
						<span style="color: var(--oo-fg-muted);">Win rate</span>
						<span class="block font-medium" style="color: var(--oo-acc-400);">{(stats.win_rate * 100).toFixed(1)}%</span>
					</div>
					<div>
						<span style="color: var(--oo-fg-muted);">Humanized wins</span>
						<span class="block font-medium" style="color: var(--oo-fg-primary);">{stats.humanized_wins}</span>
					</div>
					<div>
						<span style="color: var(--oo-fg-muted);">Original wins</span>
						<span class="block font-medium" style="color: var(--oo-fg-primary);">{stats.original_wins}</span>
					</div>
				</div>

				{#if Object.keys(stats.by_strategy).length > 0}
					<div class="mt-3 pt-3" style="border-top: 1px solid var(--oo-bd-subtle);">
						<span class="text-xs font-medium" style="color: var(--oo-fg-secondary);">By strategy</span>
						<div class="mt-1 space-y-1">
							{#each Object.entries(stats.by_strategy) as [strategy, counts]}
								<div class="flex items-center justify-between text-xs">
									<span class="font-mono" style="color: var(--oo-fg-muted);">{strategy}</span>
									<span style="color: var(--oo-fg-secondary);">
										{counts.humanized}W / {counts.original}L / {counts.tie}T
									</span>
								</div>
							{/each}
						</div>
					</div>
				{/if}
			</div>
		{:else if !statsLoading}
			<p class="text-xs" style="color: var(--oo-fg-muted);">
				No feedback data yet. Use the A/B comparison in chat messages to rate humanized output.
			</p>
		{/if}

	</div>
{/if}
