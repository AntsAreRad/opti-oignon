<!--
  CascadingPanel.svelte -- Cascading Inference settings panel.

  Sections:
  1. Enable/disable toggle
  2. Tier configuration table (model, threshold, max_tokens per tier)
  3. Add/remove/reorder tiers
  4. Test cascade button with result visualization
  5. Last cascade summary (model used, tier reached, score, latency)
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getCascadingStatus,
		updateCascadingConfig,
		testCascade,
	} from '$lib/api/cascading';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type { CascadeStatus, CascadeTier, CascadeResult } from '$lib/types';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	let loading = true;
	let error = '';

	let status: CascadeStatus | null = null;
	let localEnabled = false;
	let localTiers: CascadeTier[] = [];
	let localRetries = 1;
	let localTimeout = 30;
	let saving = false;

	// Test
	let testQuery = 'Explain what a hash table is and how it works.';
	let testing = false;
	let testResult: CascadeResult | null = null;

	// -------------------------------------------------------------------------
	// Load
	// -------------------------------------------------------------------------

	onMount(loadData);

	async function loadData() {
		loading = true;
		error = '';
		try {
			status = await getCascadingStatus();
			localEnabled = status.enabled;
			localTiers = status.tiers.map((t) => ({ ...t }));
			const cfg = status.config || {};
			localRetries = (cfg.max_retries_per_tier as number) ?? 1;
			localTimeout = (cfg.timeout_per_tier_seconds as number) ?? 30;
		} catch (e) {
			error = `Failed to load cascading status: ${e}`;
		} finally {
			loading = false;
		}
	}

	// -------------------------------------------------------------------------
	// Actions
	// -------------------------------------------------------------------------

	async function handleSave() {
		saving = true;
		try {
			const tiers = localTiers.map((t) => ({
				name: t.name,
				model: t.model,
				threshold: t.threshold,
				max_tokens: t.max_tokens,
				temperature: t.temperature,
			}));
			status = await updateCascadingConfig({
				enabled: localEnabled,
				tiers,
				max_retries_per_tier: localRetries,
				timeout_per_tier_seconds: localTimeout,
			});
			toastSuccess('Cascading config saved');
		} catch (e) {
			toastError(`Save failed: ${e}`);
		} finally {
			saving = false;
		}
	}

	function addTier() {
		localTiers = [
			...localTiers,
			{
				name: `tier_${localTiers.length}`,
				model: '',
				threshold: 0.0,
				max_tokens: 4096,
				temperature: 0.5,
			},
		];
	}

	function removeTier(index: number) {
		localTiers = localTiers.filter((_, i) => i !== index);
	}

	function moveTier(index: number, direction: -1 | 1) {
		const target = index + direction;
		if (target < 0 || target >= localTiers.length) return;
		const copy = [...localTiers];
		[copy[index], copy[target]] = [copy[target], copy[index]];
		localTiers = copy;
	}

	async function handleTest() {
		if (!testQuery.trim()) return;
		testing = true;
		testResult = null;
		try {
			const resp = await testCascade(testQuery);
			testResult = resp.result;
			toastSuccess(`Cascade resolved at tier "${testResult.tier_name}"`);
		} catch (e) {
			toastError(`Cascade test failed: ${e}`);
		} finally {
			testing = false;
		}
	}

	// -------------------------------------------------------------------------
	// Helpers
	// -------------------------------------------------------------------------

	function tierColor(tierName: string): string {
		if (tierName === 'fast') return 'var(--oo-success)';
		if (tierName === 'standard' || tierName === 'medium') return 'var(--oo-warning)';
		return 'var(--oo-error)';
	}

	function formatMs(ms: number): string {
		if (ms < 1000) return `${Math.round(ms)}ms`;
		return `${(ms / 1000).toFixed(1)}s`;
	}
</script>

<div class="space-y-4">
	<div class="flex items-center justify-between">
		<h3 class="text-sm font-medium" style="color: var(--oo-fg-primary);">
			Cascading Inference
		</h3>
		{#if !loading}
			<span class="text-xs px-2 py-0.5 rounded-full"
				style="background-color: {status?.available ? 'var(--oo-success-bg)' : 'var(--oo-error-bg)'};
					color: {status?.available ? 'var(--oo-success)' : 'var(--oo-error)'};">
				{status?.available ? 'Available' : 'Unavailable'}
			</span>
		{/if}
	</div>

	{#if loading}
		<p class="text-xs" style="color: var(--oo-fg-muted);">Loading...</p>
	{:else if error}
		<p class="text-xs" style="color: var(--oo-error);">{error}</p>
	{:else}
		<!-- Toggle -->
		<label class="flex items-center gap-2 cursor-pointer">
			<input type="checkbox" bind:checked={localEnabled}
				class="rounded" style="accent-color: var(--oo-acc-400);" />
			<span class="text-xs" style="color: var(--oo-fg-secondary);">
				Enable cascading inference
			</span>
		</label>

		<p class="text-xs" style="color: var(--oo-fg-muted);">
			Routes queries through progressively larger models, stopping at the first
			whose response meets the quality threshold.
		</p>

		<!-- Tier table -->
		<div class="space-y-2">
			<div class="flex items-center justify-between">
				<span class="text-xs font-medium" style="color: var(--oo-fg-secondary);">
					Tiers ({localTiers.length})
				</span>
				<button on:click={addTier}
					class="text-xs px-2 py-0.5 rounded"
					style="background-color: var(--oo-bg-elevated); color: var(--oo-acc-400);
						border: 1px solid var(--oo-bd-default);">
					+ Add Tier
				</button>
			</div>

			{#each localTiers as tier, idx}
				<div class="rounded-lg p-3 space-y-2"
					style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class="w-2 h-2 rounded-full inline-block"
								style="background-color: {tierColor(tier.name)};" />
							<input type="text" bind:value={tier.name}
								class="text-xs font-medium w-24 px-1 py-0.5 rounded"
								style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary);
									border: 1px solid var(--oo-input-bd);"
								placeholder="Tier name" />
						</div>
						<div class="flex items-center gap-1">
							<button on:click={() => moveTier(idx, -1)}
								class="text-xs px-1 py-0.5 rounded"
								style="color: var(--oo-fg-muted);"
								disabled={idx === 0}
								title="Move up">
								&#9650;
							</button>
							<button on:click={() => moveTier(idx, 1)}
								class="text-xs px-1 py-0.5 rounded"
								style="color: var(--oo-fg-muted);"
								disabled={idx === localTiers.length - 1}
								title="Move down">
								&#9660;
							</button>
							<button on:click={() => removeTier(idx)}
								class="text-xs px-1 py-0.5 rounded"
								style="color: var(--oo-error);"
								title="Remove tier">
								&#10005;
							</button>
						</div>
					</div>

					<div class="grid grid-cols-2 gap-2">
						<div>
							<label class="text-xs block mb-0.5" style="color: var(--oo-fg-muted);">Model</label>
							<input type="text" bind:value={tier.model}
								class="text-xs w-full px-2 py-1 rounded"
								style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary);
									border: 1px solid var(--oo-input-bd);"
								placeholder="e.g. qwen3:8b" />
						</div>
						<div>
							<label class="text-xs block mb-0.5" style="color: var(--oo-fg-muted);">
								Threshold ({tier.threshold.toFixed(2)})
							</label>
							<input type="range" bind:value={tier.threshold}
								min="0" max="1" step="0.05"
								class="w-full"
								style="accent-color: var(--oo-acc-400);" />
						</div>
						<div>
							<label class="text-xs block mb-0.5" style="color: var(--oo-fg-muted);">Max tokens</label>
							<input type="number" bind:value={tier.max_tokens}
								class="text-xs w-full px-2 py-1 rounded"
								style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary);
									border: 1px solid var(--oo-input-bd);"
								min="256" max="32768" step="256" />
						</div>
						<div>
							<label class="text-xs block mb-0.5" style="color: var(--oo-fg-muted);">
								Temperature ({tier.temperature.toFixed(2)})
							</label>
							<input type="range" bind:value={tier.temperature}
								min="0" max="2" step="0.05"
								class="w-full"
								style="accent-color: var(--oo-acc-400);" />
						</div>
					</div>
				</div>
			{/each}
		</div>

		<!-- Global settings -->
		<div class="grid grid-cols-2 gap-3">
			<div>
				<label class="text-xs block mb-0.5" style="color: var(--oo-fg-muted);">Retries per tier</label>
				<input type="number" bind:value={localRetries}
					class="text-xs w-full px-2 py-1 rounded"
					style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary);
						border: 1px solid var(--oo-input-bd);"
					min="0" max="5" />
			</div>
			<div>
				<label class="text-xs block mb-0.5" style="color: var(--oo-fg-muted);">Timeout per tier (s)</label>
				<input type="number" bind:value={localTimeout}
					class="text-xs w-full px-2 py-1 rounded"
					style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary);
						border: 1px solid var(--oo-input-bd);"
					min="5" max="300" />
			</div>
		</div>

		<!-- Save button -->
		<button on:click={handleSave}
			class="text-xs px-3 py-1.5 rounded-lg font-medium transition-colors"
			style="background-color: var(--oo-acc-400); color: var(--oo-bg-primary);"
			disabled={saving}>
			{saving ? 'Saving...' : 'Save Configuration'}
		</button>

		<!-- Test cascade -->
		<div class="space-y-2 pt-2" style="border-top: 1px solid var(--oo-bd-default);">
			<span class="text-xs font-medium" style="color: var(--oo-fg-secondary);">Test Cascade</span>
			<div class="flex gap-2">
				<input type="text" bind:value={testQuery}
					class="text-xs flex-1 px-2 py-1.5 rounded"
					style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary);
						border: 1px solid var(--oo-input-bd);"
					placeholder="Enter a test query..." />
				<button on:click={handleTest}
					class="text-xs px-3 py-1.5 rounded font-medium"
					style="background-color: var(--oo-bg-elevated); color: var(--oo-acc-400);
						border: 1px solid var(--oo-bd-default);"
					disabled={testing || !localEnabled}>
					{testing ? 'Running...' : 'Run Test'}
				</button>
			</div>

			{#if testResult}
				<div class="rounded-lg p-3 space-y-1.5"
					style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
					<div class="flex items-center gap-2">
						<span class="w-2 h-2 rounded-full inline-block"
							style="background-color: {tierColor(testResult.tier_name)};" />
						<span class="text-xs font-medium" style="color: var(--oo-fg-primary);">
							Resolved at "{testResult.tier_name}" tier
						</span>
					</div>
					<div class="grid grid-cols-3 gap-2 text-xs" style="color: var(--oo-fg-muted);">
						<div>Model: <span style="color: var(--oo-fg-secondary);">{testResult.model_used}</span></div>
						<div>Score: <span style="color: var(--oo-fg-secondary);">{testResult.score.toFixed(3)}</span></div>
						<div>Latency: <span style="color: var(--oo-fg-secondary);">{formatMs(testResult.total_latency_ms)}</span></div>
					</div>
					{#if testResult.attempts.length > 1}
						<div class="text-xs" style="color: var(--oo-fg-muted);">
							Tiers attempted: {testResult.attempts.length}
							{#if testResult.escalation_reasons.length > 0}
								<span> -- Escalations: {testResult.escalation_reasons.length}</span>
							{/if}
						</div>
					{/if}
				</div>
			{/if}
		</div>

		<!-- Last cascade summary -->
		{#if status?.last_result}
			<div class="space-y-1 pt-2" style="border-top: 1px solid var(--oo-bd-default);">
				<span class="text-xs font-medium" style="color: var(--oo-fg-secondary);">Last Cascade</span>
				<div class="text-xs grid grid-cols-2 gap-1" style="color: var(--oo-fg-muted);">
					<div>Model: <span style="color: var(--oo-fg-secondary);">{status.last_result.model_used}</span></div>
					<div>Tier: <span style="color: var(--oo-fg-secondary);">{status.last_result.tier_name}</span></div>
					<div>Score: <span style="color: var(--oo-fg-secondary);">{Number(status.last_result.score).toFixed(3)}</span></div>
					<div>Latency: <span style="color: var(--oo-fg-secondary);">{formatMs(Number(status.last_result.total_latency_ms))}</span></div>
				</div>
			</div>
		{/if}
	{/if}
</div>
