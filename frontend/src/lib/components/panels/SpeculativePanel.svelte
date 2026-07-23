<!--
  SpeculativePanel.svelte -- Speculative Generation settings panel.

  Sections:
  1. Enable/disable toggle (mutual exclusion with cascading)
  2. Draft/verify model configuration
  3. Convergence threshold and iteration settings
  4. Test speculative button with result visualization
  5. Last generation summary (draft accepted, convergence, latency)
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getSpeculativeStatus,
		updateSpeculativeConfig,
		testSpeculative,
	} from '$lib/api/speculative';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type { SpeculativeStatus, SpeculativeResult } from '$lib/types';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	let loading = true;
	let error = '';

	let status: SpeculativeStatus | null = null;
	let localEnabled = false;
	let localDraftModel = 'qwen3:8b';
	let localVerifyModel = 'qwen3:32b';
	let localMaxIterations = 2;
	let localConvergenceThreshold = 0.85;
	let localDraftMaxTokens = 2048;
	let localVerifyMaxTokens = 4096;
	let localDraftTemperature = 0.5;
	let localVerifyTemperature = 0.3;
	let saving = false;

	// Test
	let testQuery = 'Explain what a hash table is and how it works.';
	let testing = false;
	let testResult: SpeculativeResult | null = null;

	// -------------------------------------------------------------------------
	// Load
	// -------------------------------------------------------------------------

	onMount(loadData);

	async function loadData() {
		loading = true;
		error = '';
		try {
			status = await getSpeculativeStatus();
			localEnabled = status.enabled;
			localDraftModel = status.draft_model || 'qwen3:8b';
			localVerifyModel = status.verify_model || 'qwen3:32b';
			localMaxIterations = status.max_iterations ?? 2;
			localConvergenceThreshold = status.convergence_threshold ?? 0.85;
			const cfg = status.config || {};
			localDraftMaxTokens = (cfg.draft_max_tokens as number) ?? 2048;
			localVerifyMaxTokens = (cfg.verify_max_tokens as number) ?? 4096;
			localDraftTemperature = (cfg.draft_temperature as number) ?? 0.5;
			localVerifyTemperature = (cfg.verify_temperature as number) ?? 0.3;
		} catch (e) {
			error = `Failed to load speculative status: ${e}`;
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
			status = await updateSpeculativeConfig({
				enabled: localEnabled,
				draft_model: localDraftModel,
				verify_model: localVerifyModel,
				max_iterations: localMaxIterations,
				convergence_threshold: localConvergenceThreshold,
				draft_max_tokens: localDraftMaxTokens,
				verify_max_tokens: localVerifyMaxTokens,
				draft_temperature: localDraftTemperature,
				verify_temperature: localVerifyTemperature,
			});
			toastSuccess('Speculative config saved');
		} catch (e) {
			toastError(`Save failed: ${e}`);
		} finally {
			saving = false;
		}
	}

	async function handleTest() {
		if (!testQuery.trim()) return;
		testing = true;
		testResult = null;
		try {
			const resp = await testSpeculative(testQuery);
			testResult = resp.result;
			toastSuccess('Speculative test complete');
		} catch (e) {
			toastError(`Test failed: ${e}`);
		} finally {
			testing = false;
		}
	}
</script>

<!-- ===================================================================== -->
<!-- Template -->
<!-- ===================================================================== -->

<div class="panel" style="background: var(--oo-bg-surface); border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-lg); padding: 1.25rem;">
	<h3 style="margin: 0 0 1rem; font-size: 1rem; font-weight: 600; color: var(--oo-fg-primary);">
		Speculative Generation
	</h3>

	{#if loading}
		<p style="color: var(--oo-fg-secondary); font-size: 0.875rem;">Loading...</p>
	{:else if error}
		<p style="color: var(--oo-status-error); font-size: 0.875rem;">{error}</p>
	{:else}
		<!-- Enable toggle -->
		<div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
			<label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer; font-size: 0.875rem; color: var(--oo-fg-primary);">
				<input
					type="checkbox"
					bind:checked={localEnabled}
					style="accent-color: var(--oo-accent-primary);"
				/>
				Enable speculative generation
			</label>
			<span style="font-size: 0.75rem; color: var(--oo-fg-tertiary);">
				(draft-verify pattern; mutually exclusive with cascading)
			</span>
		</div>

		<!-- Model config -->
		<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem;">
			<div>
				<label style="display: block; font-size: 0.75rem; color: var(--oo-fg-secondary); margin-bottom: 0.25rem;">
					Draft model (fast)
				</label>
				<input
					type="text"
					bind:value={localDraftModel}
					style="width: 100%; padding: 0.375rem 0.5rem; font-size: 0.8125rem; border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-sm); background: var(--oo-bg-input); color: var(--oo-fg-primary);"
				/>
			</div>
			<div>
				<label style="display: block; font-size: 0.75rem; color: var(--oo-fg-secondary); margin-bottom: 0.25rem;">
					Verify model (powerful)
				</label>
				<input
					type="text"
					bind:value={localVerifyModel}
					style="width: 100%; padding: 0.375rem 0.5rem; font-size: 0.8125rem; border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-sm); background: var(--oo-bg-input); color: var(--oo-fg-primary);"
				/>
			</div>
		</div>

		<!-- Parameters -->
		<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 1rem;">
			<div>
				<label style="display: block; font-size: 0.75rem; color: var(--oo-fg-secondary); margin-bottom: 0.25rem;">
					Max iterations
				</label>
				<input
					type="number"
					bind:value={localMaxIterations}
					min="1"
					max="5"
					style="width: 100%; padding: 0.375rem 0.5rem; font-size: 0.8125rem; border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-sm); background: var(--oo-bg-input); color: var(--oo-fg-primary);"
				/>
			</div>
			<div>
				<label style="display: block; font-size: 0.75rem; color: var(--oo-fg-secondary); margin-bottom: 0.25rem;">
					Convergence threshold
				</label>
				<input
					type="number"
					bind:value={localConvergenceThreshold}
					min="0"
					max="1"
					step="0.05"
					style="width: 100%; padding: 0.375rem 0.5rem; font-size: 0.8125rem; border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-sm); background: var(--oo-bg-input); color: var(--oo-fg-primary);"
				/>
			</div>
			<div>
				<label style="display: block; font-size: 0.75rem; color: var(--oo-fg-secondary); margin-bottom: 0.25rem;">
					Draft temp
				</label>
				<input
					type="number"
					bind:value={localDraftTemperature}
					min="0"
					max="2"
					step="0.1"
					style="width: 100%; padding: 0.375rem 0.5rem; font-size: 0.8125rem; border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-sm); background: var(--oo-bg-input); color: var(--oo-fg-primary);"
				/>
			</div>
			<div>
				<label style="display: block; font-size: 0.75rem; color: var(--oo-fg-secondary); margin-bottom: 0.25rem;">
					Verify temp
				</label>
				<input
					type="number"
					bind:value={localVerifyTemperature}
					min="0"
					max="2"
					step="0.1"
					style="width: 100%; padding: 0.375rem 0.5rem; font-size: 0.8125rem; border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-sm); background: var(--oo-bg-input); color: var(--oo-fg-primary);"
				/>
			</div>
		</div>

		<!-- Token limits -->
		<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem;">
			<div>
				<label style="display: block; font-size: 0.75rem; color: var(--oo-fg-secondary); margin-bottom: 0.25rem;">
					Draft max tokens
				</label>
				<input
					type="number"
					bind:value={localDraftMaxTokens}
					min="256"
					max="16384"
					step="256"
					style="width: 100%; padding: 0.375rem 0.5rem; font-size: 0.8125rem; border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-sm); background: var(--oo-bg-input); color: var(--oo-fg-primary);"
				/>
			</div>
			<div>
				<label style="display: block; font-size: 0.75rem; color: var(--oo-fg-secondary); margin-bottom: 0.25rem;">
					Verify max tokens
				</label>
				<input
					type="number"
					bind:value={localVerifyMaxTokens}
					min="256"
					max="16384"
					step="256"
					style="width: 100%; padding: 0.375rem 0.5rem; font-size: 0.8125rem; border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-sm); background: var(--oo-bg-input); color: var(--oo-fg-primary);"
				/>
			</div>
		</div>

		<!-- Save button -->
		<button
			on:click={handleSave}
			disabled={saving}
			style="padding: 0.5rem 1rem; font-size: 0.8125rem; font-weight: 500; background: var(--oo-accent-primary); color: var(--oo-fg-on-accent); border: none; border-radius: var(--oo-radius-sm); cursor: pointer; opacity: {saving ? 0.6 : 1};"
		>
			{saving ? 'Saving...' : 'Save configuration'}
		</button>

		<!-- Test section -->
		<div style="margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid var(--oo-bd-default);">
			<h4 style="margin: 0 0 0.75rem; font-size: 0.875rem; font-weight: 600; color: var(--oo-fg-primary);">
				Test speculative generation
			</h4>
			<div style="display: flex; gap: 0.5rem; margin-bottom: 0.75rem;">
				<input
					type="text"
					bind:value={testQuery}
					placeholder="Enter a test query..."
					style="flex: 1; padding: 0.375rem 0.5rem; font-size: 0.8125rem; border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-sm); background: var(--oo-bg-input); color: var(--oo-fg-primary);"
				/>
				<button
					on:click={handleTest}
					disabled={testing || !localEnabled}
					style="padding: 0.375rem 0.75rem; font-size: 0.8125rem; background: var(--oo-bg-surface-raised); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-sm); cursor: pointer; opacity: {testing || !localEnabled ? 0.5 : 1};"
				>
					{testing ? 'Running...' : 'Run test'}
				</button>
			</div>

			{#if testResult}
				<div style="background: var(--oo-bg-surface-raised); border: 1px solid var(--oo-bd-default); border-radius: var(--oo-radius-sm); padding: 0.75rem; font-size: 0.8125rem;">
					<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-bottom: 0.75rem;">
						<div>
							<span style="color: var(--oo-fg-tertiary);">Draft accepted:</span>
							<span style="color: {testResult.draft_accepted ? 'var(--oo-status-success)' : 'var(--oo-status-warning)'}; font-weight: 500;">
								{testResult.draft_accepted ? 'Yes' : 'No'}
							</span>
						</div>
						<div>
							<span style="color: var(--oo-fg-tertiary);">Convergence:</span>
							<span style="font-weight: 500; color: var(--oo-fg-primary);">{testResult.convergence_score.toFixed(3)}</span>
						</div>
						<div>
							<span style="color: var(--oo-fg-tertiary);">Iterations:</span>
							<span style="font-weight: 500; color: var(--oo-fg-primary);">{testResult.iterations}</span>
						</div>
					</div>
					<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-bottom: 0.75rem;">
						<div>
							<span style="color: var(--oo-fg-tertiary);">Total:</span>
							<span style="color: var(--oo-fg-primary);">{testResult.total_latency_ms.toFixed(0)}ms</span>
						</div>
						<div>
							<span style="color: var(--oo-fg-tertiary);">Draft:</span>
							<span style="color: var(--oo-fg-primary);">{testResult.draft_latency_ms.toFixed(0)}ms</span>
						</div>
						<div>
							<span style="color: var(--oo-fg-tertiary);">Verify:</span>
							<span style="color: var(--oo-fg-primary);">{testResult.verify_latency_ms.toFixed(0)}ms</span>
						</div>
					</div>
					<div style="color: var(--oo-fg-secondary); max-height: 6rem; overflow-y: auto; white-space: pre-wrap; font-size: 0.75rem; line-height: 1.4;">
						{testResult.final_response.slice(0, 500)}{testResult.final_response.length > 500 ? '...' : ''}
					</div>
				</div>
			{/if}
		</div>

		<!-- Last result summary -->
		{#if status?.last_result}
			<div style="margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid var(--oo-bd-default);">
				<p style="font-size: 0.75rem; color: var(--oo-fg-tertiary); margin: 0;">
					Last run:
					draft {status.last_result.draft_accepted ? 'accepted' : 'rejected'},
					convergence {Number(status.last_result.convergence_score ?? 0).toFixed(3)},
					{Number(status.last_result.total_latency_ms ?? 0).toFixed(0)}ms
				</p>
			</div>
		{/if}
	{/if}
</div>
