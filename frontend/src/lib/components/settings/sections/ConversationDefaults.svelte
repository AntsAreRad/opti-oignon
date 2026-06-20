<!--
  ConversationDefaults.svelte (S168)
  Conversation & Chat > Defaults group of the consolidated /settings hub.

  This is where the legacy "Quick" tab content now lives (spec 5.5: the
  default model, temperature, code execution, memory injection and the
  one-click system presets were scattered across quick / prompt / presets and
  the runtime ChatControlBar). Each toggle here is the effective application
  default that the per-conversation ChatControlBar toggles start from.

  Controls apply immediately with a toast (spec 5.9). Built on the ds
  primitives (Switch, Input, Button) inside SettingsGroup wrappers.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import SettingsGroup from '$lib/components/settings/SettingsGroup.svelte';
	import Switch from '$lib/ds/Switch.svelte';
	import Input from '$lib/ds/Input.svelte';
	import Button from '$lib/ds/Button.svelte';
	import { getSettings, updateSetting, reloadSettings } from '$lib/api/settings';
	import {
		listSystemPresets,
		detectAndRecommend,
		applySystemPreset,
		getOnboardingState,
		resetOnboarding
	} from '$lib/api/systemPresets';
	import { handleApiError } from '$lib/api/errorHandler';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type { SystemPresetInfo, SystemPresetDetectResponse } from '$lib/types';

	let loading = true;
	let error = '';

	let defaultModel = '';
	let defaultTemperature = 0.7;
	let codeExecutionEnabled = true;
	let memoryInjectionEnabled = true;
	let persistentDir = '';

	let systemPresets: SystemPresetInfo[] = [];
	let detection: SystemPresetDetectResponse | null = null;
	let currentAppliedPreset: string | null = null;
	let applyingPreset = false;
	let detectLoading = false;
	let reloading = false;
	let resettingOnboarding = false;

	async function load() {
		loading = true;
		error = '';
		try {
			const data = await getSettings();
			defaultModel = (data.user?.default_model as string) ?? (data.models?.default as string) ?? '';
			defaultTemperature = (data.user?.temperature as number) ?? 0.7;
			codeExecutionEnabled = (data.user?.code_execution as boolean) ?? true;
			memoryInjectionEnabled = (data.user?.memory_injection as boolean) ?? true;
			persistentDir = (data.user?.persistent_dir as string) ?? '';
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load settings';
		} finally {
			loading = false;
		}
	}

	async function loadSystemPresets() {
		detectLoading = true;
		try {
			const [presetsResp, detectResp, onbState] = await Promise.all([
				listSystemPresets(),
				detectAndRecommend(),
				getOnboardingState()
			]);
			systemPresets = presetsResp.presets;
			detection = detectResp;
			currentAppliedPreset = onbState.applied_preset;
		} catch {
			// System presets module may not be available.
		} finally {
			detectLoading = false;
		}
	}

	async function saveSetting(key: string, value: unknown, label: string) {
		try {
			await updateSetting(key, value);
			toastSuccess(`${label} updated`);
		} catch (e) {
			handleApiError(e, `updating ${key}`);
		}
	}

	async function handleApplySystemPreset(presetId: string) {
		applyingPreset = true;
		try {
			const result = await applySystemPreset(presetId);
			if (result.applied) {
				currentAppliedPreset = result.preset_id;
				toastSuccess(`Applied "${result.preset_name}" preset. Model: ${result.selected_model || 'none'}`);
				if (result.warnings.length > 0) toastError(result.warnings.join('; '));
				await load();
			} else {
				toastError(result.error || 'Failed to apply preset');
			}
		} catch (e) {
			handleApiError(e, 'applying preset');
		} finally {
			applyingPreset = false;
		}
	}

	async function handleReload() {
		reloading = true;
		try {
			await reloadSettings();
			toastSuccess('Configuration reloaded from disk');
			await load();
		} catch (e) {
			handleApiError(e, 'reloading configuration');
		} finally {
			reloading = false;
		}
	}

	async function handleResetOnboarding() {
		resettingOnboarding = true;
		try {
			await resetOnboarding();
			currentAppliedPreset = null;
			toastSuccess('Onboarding reset. The setup overlay will appear on next page load.');
		} catch (e) {
			handleApiError(e, 'resetting onboarding');
		} finally {
			resettingOnboarding = false;
		}
	}

	async function resetDefaults() {
		defaultTemperature = 0.7;
		codeExecutionEnabled = true;
		memoryInjectionEnabled = true;
		persistentDir = '';
		try {
			await Promise.all([
				updateSetting('temperature', 0.7),
				updateSetting('code_execution', true),
				updateSetting('memory_injection', true),
				updateSetting('persistent_dir', '')
			]);
			toastSuccess('Conversation defaults reset');
		} catch (e) {
			handleApiError(e, 'resetting defaults');
		}
	}

	function presetIconSvg(icon: string): string {
		switch (icon) {
			case 'leaf':
				return 'M17 8C8 10 5.9 16.17 3.82 21.34l1.89.66L7 18.5C9 15 12 12 17 8z M20.5 3.5C17 7 13 9 10 10l1 2c3-1 6.5-3.5 9.5-6.5';
			case 'scale':
				return 'M12 3v18m-7-4l7-10 7 10M5 17h14';
			case 'zap':
				return 'M13 2L3 14h9l-1 10 10-12h-9l1-10z';
			default:
				return 'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5';
		}
	}

	onMount(() => {
		load();
		loadSystemPresets();
	});
</script>

<div class="oo-conv-defaults">
	{#if error}
		<div class="oo-conv-error" role="alert">
			{error}
			<button type="button" class="oo-conv-retry" on:click={load}>Retry</button>
		</div>
	{/if}

	<SettingsGroup
		id="conversation-system-preset"
		title="System preset"
		description="Infrastructure-level configuration. Applies caching, cascading, routing and token budgets in one click."
	>
		{#if detectLoading}
			<p class="oo-conv-muted">Detecting models...</p>
		{:else}
			{#if detection && detection.models.length > 0}
				<div class="oo-conv-models">
					{#each detection.models as m}
						<span class="oo-conv-chip">
							<span class="oo-conv-chip-name">{m.name}</span>
							{#if m.parameter_count_b > 0}
								<span class="oo-conv-chip-size">{m.parameter_count_b}B</span>
							{/if}
						</span>
					{/each}
				</div>
			{/if}

			<div class="oo-preset-grid">
				{#each systemPresets as preset (preset.id)}
					{@const isCurrent = currentAppliedPreset === preset.id}
					{@const isRecommended = detection?.recommended_preset === preset.id}
					<div class="oo-preset" class:oo-preset-current={isCurrent}>
						<div class="oo-preset-icon" class:oo-preset-icon-current={isCurrent}>
							<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke="currentColor">
								<path d={presetIconSvg(preset.icon)} />
							</svg>
						</div>
						<div class="oo-preset-info">
							<div class="oo-preset-head">
								<span class="oo-preset-name">{preset.name}</span>
								{#if isCurrent}
									<span class="oo-preset-tag oo-preset-tag-active">Active</span>
								{:else if isRecommended}
									<span class="oo-preset-tag oo-preset-tag-rec">Recommended</span>
								{/if}
								<span class="oo-preset-ram">{preset.recommended_ram_gb}+ GB</span>
							</div>
							<p class="oo-preset-desc">{preset.description}</p>
						</div>
						<Button
							size="sm"
							variant={isCurrent ? 'ghost' : 'primary'}
							loading={applyingPreset}
							disabled={applyingPreset || isCurrent}
							on:click={() => handleApplySystemPreset(preset.id)}
						>
							{isCurrent ? 'Applied' : 'Apply'}
						</Button>
					</div>
				{/each}
			</div>

			{#if detection?.reason}
				<p class="oo-conv-reason">{detection.reason}</p>
			{/if}
		{/if}
	</SettingsGroup>

	<SettingsGroup
		id="conversation-defaults"
		title="Defaults for new conversations"
		description="The starting values every new conversation inherits. Per-conversation overrides live in the chat control bar."
		onReset={resetDefaults}
	>
		{#if loading}
			<p class="oo-conv-muted">Loading settings...</p>
		{:else}
			<Input
				label="Default model"
				bind:value={defaultModel}
				placeholder="e.g. qwen3:8b"
				hint="Used when a conversation does not pin a model."
				on:change={() => saveSetting('default_model', defaultModel, 'Default model')}
			/>
			<Input
				type="number"
				label="Default temperature"
				bind:value={defaultTemperature}
				hint="0 is deterministic; higher is more varied."
				on:change={() => saveSetting('temperature', defaultTemperature, 'Default temperature')}
			/>
			<Switch
				label="Code execution"
				description="Allow the assistant to run code in the sandbox by default."
				bind:checked={codeExecutionEnabled}
				on:change={() => saveSetting('code_execution', codeExecutionEnabled, 'Code execution')}
			/>
			<Switch
				label="Memory injection"
				description="Inject relevant memory into the prompt by default."
				bind:checked={memoryInjectionEnabled}
				on:change={() => saveSetting('memory_injection', memoryInjectionEnabled, 'Memory injection')}
			/>
			<Input
				label="Persistent directory"
				bind:value={persistentDir}
				placeholder="Optional path"
				hint="Working directory persisted across sandbox runs."
				on:change={() => saveSetting('persistent_dir', persistentDir, 'Persistent directory')}
			/>
		{/if}
	</SettingsGroup>

	<SettingsGroup
		id="conversation-config-maintenance"
		title="Configuration"
		description="Reload configuration from disk or re-run the first-time setup."
	>
		<div class="oo-conv-actions">
			<Button variant="secondary" size="sm" iconLeft="refresh-cw" loading={reloading} on:click={handleReload}>
				Reload from disk
			</Button>
			<Button variant="ghost" size="sm" iconLeft="rotate-ccw" loading={resettingOnboarding} on:click={handleResetOnboarding}>
				Reset onboarding
			</Button>
		</div>
	</SettingsGroup>
</div>

<style>
	.oo-conv-defaults {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-4);
	}

	.oo-conv-error {
		padding: var(--oo-space-2) var(--oo-space-3);
		border-radius: var(--oo-radius-md);
		background-color: var(--oo-error-bg);
		border: 1px solid var(--oo-error-bd);
		color: var(--oo-error);
		font-size: var(--oo-text-sm);
	}

	.oo-conv-retry {
		margin-left: var(--oo-space-2);
		text-decoration: underline;
		color: inherit;
	}

	.oo-conv-muted {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
	}

	.oo-conv-models {
		display: flex;
		flex-wrap: wrap;
		gap: var(--oo-space-2);
	}

	.oo-conv-chip {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-1);
		padding: 2px var(--oo-space-2);
		border-radius: var(--oo-radius-sm);
		background-color: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-subtle);
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-secondary);
	}

	.oo-conv-chip-name {
		font-family: var(--oo-font-mono);
	}

	.oo-conv-chip-size {
		color: var(--oo-fg-muted);
	}

	.oo-preset-grid {
		display: grid;
		gap: var(--oo-space-2);
	}

	.oo-preset {
		display: flex;
		align-items: center;
		gap: var(--oo-space-3);
		padding: var(--oo-space-3) var(--oo-space-4);
		border-radius: var(--oo-radius-md);
		background-color: var(--oo-bg-elevated);
		border: 1.5px solid var(--oo-bd-subtle);
	}

	.oo-preset-current {
		background-color: var(--oo-accent-bg);
		border-color: var(--oo-accent);
	}

	.oo-preset-icon {
		width: 36px;
		height: 36px;
		border-radius: var(--oo-radius-md);
		display: flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		background-color: var(--oo-bg-overlay);
		color: var(--oo-fg-tertiary);
	}

	.oo-preset-icon svg {
		width: 20px;
		height: 20px;
	}

	.oo-preset-icon-current {
		background-color: var(--oo-accent);
		color: var(--oo-fg-on-accent);
	}

	.oo-preset-info {
		flex: 1;
		min-width: 0;
	}

	.oo-preset-head {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
	}

	.oo-preset-name {
		font-size: var(--oo-text-sm);
		font-weight: 500;
		color: var(--oo-fg-primary);
	}

	.oo-preset-tag {
		font-size: var(--oo-text-2xs);
		padding: 1px var(--oo-space-2);
		border-radius: var(--oo-radius-full);
		font-weight: 500;
	}

	.oo-preset-tag-active {
		background-color: var(--oo-accent);
		color: var(--oo-fg-on-accent);
	}

	.oo-preset-tag-rec {
		background-color: var(--oo-success-bg);
		color: var(--oo-success);
		border: 1px solid var(--oo-success-bd);
	}

	.oo-preset-ram {
		margin-left: auto;
		font-size: var(--oo-text-2xs);
		color: var(--oo-fg-muted);
	}

	.oo-preset-desc {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-tertiary);
		margin: 2px 0 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.oo-conv-reason {
		font-size: var(--oo-text-xs);
		font-style: italic;
		color: var(--oo-fg-muted);
	}

	.oo-conv-actions {
		display: flex;
		flex-wrap: wrap;
		gap: var(--oo-space-2);
	}
</style>
