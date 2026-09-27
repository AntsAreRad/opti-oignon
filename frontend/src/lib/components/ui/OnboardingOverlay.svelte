<!--
  OnboardingOverlay.svelte
  The first-run dialog, shown while the install is not set up
  (user_initialized false). It lists the models it finds installed,
  recommends a system preset, and applies the one the reader chooses.

  The presets are one choice, a radio group named "System preset": each
  card says its name, the memory it wants and what it does; the chosen one
  carries a check, the recommended one says "Recommended" in a word, and
  the dialog opens on the chosen card. Once a preset is applied, the dialog
  closes on "Get started", Escape or its close button alike, and reads
  again what the preset changed (the chat's models and default model, the
  feature map, the backends, the control bar's switches:
  lib/stores/configRefresh.ts) instead of reloading the page. While the
  preset is being applied, the dialog cannot be closed.

  Built on the ds Modal (a native modal dialog): Escape and the close
  button mean Skip, a click on the backdrop does nothing. Stop all sits in
  its head, since the page under it is inert, and never takes its first
  focus. Each step that replaces the control just pressed hands focus to
  its own: the note while the models are scanned or the preset applied,
  "Get started" once applied, the retry after a failure. Its props are its
  state, set by the component itself once mounted, so each step can be
  drawn on the server.
-->
<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { Button, Icon, InlineError, Modal } from '$lib/ds';
	import StopAllButton from '$lib/components/layout/StopAllButton.svelte';
	import { isPhone } from '$lib/stores/ui';
	import { refreshAfterConfigChange } from '$lib/stores/configRefresh';
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

	/** Whether the dialog is shown. */
	export let visible = false;
	/** Where the dialog is. */
	export let step: 'loading' | 'ready' | 'applying' | 'done' | 'error' = 'loading';
	/** The presets offered. */
	export let presets: SystemPresetInfo[] = [];
	/** The models found and the preset they recommend. */
	export let detection: SystemPresetDetectResponse | null = null;
	/** The preset chosen. */
	export let selectedPresetId = '';

	let applyResult: { preset_name: string; selected_model: string | null; warnings: string[] } | null = null;
	let errorMsg = '';
	let choices: HTMLElement | undefined;
	let body: HTMLElement | undefined;
	let actions: HTMLElement | undefined;

	const MAX_RETRIES = 3;
	const RETRY_DELAY_MS = 2000;

	$: chosen = presets.find((preset) => preset.id === selectedPresetId);

	onMount(async () => {
		try {
			await resolveOnboarding();
		} finally {
			// The browser specs cannot tell "the dialog has not appeared yet"
			// from "the dialog will never appear" by looking at the page: both
			// are an absence. Marking the decision is what makes them
			// distinguishable, and a finally is the only place that covers all
			// three ways out -- configured, not configured, backend silent.
			document.documentElement.setAttribute('data-onboarding', 'resolved');
		}
	});

	async function resolveOnboarding() {
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
				// The backend may not be ready yet (404 or no connection):
				// wait, then ask again before giving up.
				if (attempt < MAX_RETRIES - 1) {
					await new Promise((r) => setTimeout(r, RETRY_DELAY_MS));
				} else {
					// Every attempt failed: the backend is not there.
					visible = false;
				}
			}
		}
	}

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
			await tick();
			choices?.querySelector<HTMLElement>('[data-autofocus]')?.focus();
		} catch (e) {
			errorMsg = e instanceof Error ? e.message : 'Failed to load system data';
			step = 'error';
			focusStep();
		}
	}

	/** A retry after a failure: the note says the models are being scanned. */
	function retryLoad() {
		loadData();
		focusStep();
	}

	/** The control just pressed is gone: focus goes to what the step shows instead. */
	async function focusStep() {
		await tick();
		const target =
			step === 'done'
				? actions?.querySelector<HTMLElement>('button')
				: step === 'error'
					? body?.querySelector<HTMLElement>('.oo-inline-error button')
					: step === 'loading' || step === 'applying'
						? body?.querySelector<HTMLElement>('.ob-step-note')
						: null;
		target?.focus();
	}

	async function handleApply() {
		if (!selectedPresetId) return;
		step = 'applying';
		errorMsg = '';
		focusStep();
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
		focusStep();
	}

	/** Skip, Escape and the close button: once a preset was applied, what it changed is read again. */
	function handleSkip() {
		visible = false;
		if (applyResult) refreshAfterConfigChange();
	}

	/** Closes the dialog, then reads again what the preset changed. */
	function handleGetStarted() {
		visible = false;
		refreshAfterConfigChange();
	}

	/** The arrows move the choice through the presets, as a radio group does. */
	async function onChoiceKey(event: KeyboardEvent, index: number) {
		const forward = event.key === 'ArrowDown' || event.key === 'ArrowRight';
		const back = event.key === 'ArrowUp' || event.key === 'ArrowLeft';
		if (!forward && !back) return;
		event.preventDefault();
		const count = presets.length;
		const next = (index + (forward ? 1 : -1) + count) % count;
		selectedPresetId = presets[next].id;
		await tick();
		choices?.querySelectorAll<HTMLElement>('[role="radio"]')[next]?.focus();
	}
</script>

<Modal
	open={visible}
	variant="center"
	size="lg"
	title="Welcome to Opti-Oignon"
	closeOnBackdrop={false}
	closable={step !== 'applying'}
	closeOnEsc={step !== 'applying'}
	onClose={handleSkip}
>
	<svelte:fragment slot="actions">
		<StopAllButton placement="dialog-head" large={$isPhone} />
	</svelte:fragment>

	<div class="ob-intro">
		<img src="/bousier-oignon.png" alt="Opti-Oignon" class="ob-logo oo-logo-adaptive" />
		<p class="ob-tagline">Let's configure your setup in one click.</p>
	</div>

	<div class="ob-body" bind:this={body}>
		{#if step === 'loading'}
			<p class="ob-note ob-step-note" tabindex="-1">Scanning installed models</p>
		{:else if step === 'error'}
			<InlineError message={errorMsg} onRetry={retryLoad} />
		{:else if step === 'ready'}
			{#if detection}
				<div class="ob-found">
					<p class="ob-found-count">
						{detection.models.length}
						{detection.models.length === 1 ? 'model' : 'models'} detected
					</p>
					{#if detection.models.length > 0}
						<ul class="ob-models">
							{#each detection.models as m}
								<li>
									<span class="ob-model-name">{m.name}</span>
									{#if m.parameter_count_b > 0}
										<span class="ob-model-size">{m.parameter_count_b}B</span>
									{/if}
								</li>
							{/each}
						</ul>
					{:else}
						<p class="ob-note">
							No models found. Install models with <code>ollama pull</code> first, or pick Minimal.
						</p>
					{/if}
				</div>
			{/if}

			<div class="ob-choices" role="radiogroup" aria-label="System preset" bind:this={choices}>
				{#each presets as preset, index (preset.id)}
					{@const picked = preset.id === selectedPresetId}
					<button
						type="button"
						class="ob-choice"
						role="radio"
						aria-checked={picked}
						tabindex={picked ? 0 : -1}
						data-autofocus={picked || undefined}
						on:click={() => (selectedPresetId = preset.id)}
						on:keydown={(event) => onChoiceKey(event, index)}
					>
						<span class="ob-choice-head">
							<span class="ob-choice-name">{preset.name}</span>
							{#if detection?.recommended_preset === preset.id}
								<span class="ob-choice-word">Recommended</span>
							{/if}
							<span class="ob-choice-ram">{preset.recommended_ram_gb}+ GB RAM</span>
							{#if picked}
								<span class="ob-choice-check"><Icon name="check" size="sm" /></span>
							{/if}
						</span>
						<span class="ob-choice-desc">{preset.description}</span>
					</button>
				{/each}
			</div>

			{#if detection?.reason}
				<p class="ob-note">{detection.reason}</p>
			{/if}
		{:else if step === 'applying'}
			<p class="ob-note ob-step-note" tabindex="-1">Applying configuration</p>
		{:else if step === 'done'}
			<div class="ob-done">
				<p class="ob-done-title">
					<span class="ob-choice-check"><Icon name="check" size="sm" /></span>
					{applyResult?.preset_name} preset applied
				</p>
				{#if applyResult?.selected_model}
					<p class="ob-note">Default model: <code>{applyResult.selected_model}</code></p>
				{/if}
				{#if applyResult?.warnings && applyResult.warnings.length > 0}
					<div class="ob-warnings">
						<p class="ob-warnings-title">
							<Icon name="alert-triangle" size="sm" />
							Warnings
						</p>
						<ul>
							{#each applyResult.warnings as w}
								<li>{w}</li>
							{/each}
						</ul>
					</div>
				{/if}
			</div>
		{/if}
	</div>

	<svelte:fragment slot="footer">
		<span class="ob-actions" bind:this={actions}>
			{#if step === 'ready'}
				<span class="ob-skip">
					<Button variant="ghost" on:click={handleSkip}>Skip</Button>
				</span>
				<Button variant="primary" disabled={!selectedPresetId} on:click={handleApply}>
					Apply {chosen?.name ?? ''} preset
				</Button>
			{:else if step === 'error'}
				<Button variant="ghost" on:click={handleSkip}>Skip for now</Button>
			{:else if step === 'done'}
				<Button variant="primary" on:click={handleGetStarted}>Get started</Button>
			{:else if step === 'applying'}
				<span class="ob-note">Please wait</span>
			{/if}
		</span>
	</svelte:fragment>
</Modal>

<style>
	.ob-intro {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--oo-space-3);
		margin-bottom: var(--oo-space-5);
		text-align: center;
	}
	.ob-logo {
		width: 4rem;
		height: 4rem;
		object-fit: contain;
	}
	.ob-tagline {
		margin: 0;
		color: var(--oo-fg-secondary);
		font-family: var(--oo-font-serif);
		font-size: var(--oo-text-base);
	}

	.ob-body {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-4);
	}
	.ob-note {
		margin: 0;
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-sm);
	}

	.ob-found {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}
	.ob-found-count {
		margin: 0;
		color: var(--oo-fg-primary);
		font-size: var(--oo-text-sm);
	}
	.ob-models {
		display: flex;
		flex-wrap: wrap;
		gap: var(--oo-space-1) var(--oo-space-3);
		max-height: 7rem;
		margin: 0;
		padding: 0;
		overflow-y: auto;
		list-style: none;
		font-size: var(--oo-text-xs);
	}
	.ob-model-name {
		color: var(--oo-fg-secondary);
		font-family: var(--oo-font-mono);
	}
	.ob-model-size {
		margin-left: var(--oo-space-1);
		color: var(--oo-fg-muted);
	}

	.ob-choices {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}
	/* Each preset is a quiet card on the sunken ground; the chosen one
	   carries a check and its name at weight 600. */
	.ob-choice {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
		width: 100%;
		padding: var(--oo-space-3) var(--oo-space-4);
		border: 1px solid var(--oo-edge);
		border-radius: var(--oo-radius-md);
		background-color: var(--oo-bg-subtle);
		color: var(--oo-fg-primary);
		font: inherit;
		text-align: left;
		cursor: pointer;
		transition: background-color var(--oo-motion-fast) var(--oo-ease-default);
	}
	.ob-choice:hover {
		background-color: color-mix(in srgb, var(--oo-fg-primary) 4%, var(--oo-bg-subtle));
	}
	.ob-choice-head {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: var(--oo-space-1) var(--oo-space-3);
	}
	.ob-choice-name {
		font-size: var(--oo-text-sm);
	}
	.ob-choice[aria-checked='true'] .ob-choice-name {
		font-weight: 600;
	}
	.ob-choice-word {
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-xs);
	}
	.ob-choice-ram {
		margin-left: auto;
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-xs);
	}
	.ob-choice-check {
		display: inline-flex;
		align-self: center;
		color: var(--oo-fg-primary);
	}
	.ob-choice-desc {
		color: var(--oo-fg-muted);
		font-size: var(--oo-text-xs);
		line-height: var(--oo-leading-snug);
	}

	.ob-done {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--oo-space-2);
		text-align: center;
	}
	.ob-done-title {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-2);
		margin: 0;
		color: var(--oo-fg-primary);
		font-size: var(--oo-text-sm);
	}
	.ob-warnings {
		align-self: stretch;
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-xs);
		text-align: left;
	}
	.ob-warnings-title {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-1);
		margin: 0 0 var(--oo-space-1);
	}
	.ob-warnings ul {
		margin: 0;
		padding-left: var(--oo-space-5);
	}

	/* The footer's buttons lay out as the footer's own; the span only finds them. */
	.ob-actions {
		display: contents;
	}
	.ob-skip {
		margin-right: auto;
	}

	@media (prefers-reduced-motion: reduce) {
		.ob-choice {
			transition: none;
		}
	}
</style>
