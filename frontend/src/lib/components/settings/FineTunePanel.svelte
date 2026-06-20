<!--
  FineTunePanel.svelte -- S96 Fine-Tuning Data Export & Variant Management.

  Three sub-tabs:
  1. Export: format selector, filters, preview, download
  2. Variants: list, register, compare
  3. Quality: conversation scoring overview
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		exportTrainingData,
		previewExport,
		getQualityScores,
		listVariants,
		registerVariant,
		unregisterVariant,
		runComparison,
	} from '$lib/api/fineTune';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type {
		FineTuneExportResponse,
		FineTunePreviewResponse,
		FineTuneQualityScore,
		FineTuneVariant,
		FineTuneCompareResponse,
	} from '$lib/types';

	type SubTab = 'export' | 'variants' | 'quality';
	let activeSubTab: SubTab = 'export';

	// -- Export state --
	let exportFormat = 'sharegpt';
	let exportModel = '';
	let exportMinQuality = 0;
	let exportMinTurns = 1;
	let exporting = false;
	let exportResult: FineTuneExportResponse | null = null;
	let previewing = false;
	let preview: FineTunePreviewResponse | null = null;

	// -- Variants state --
	let variants: FineTuneVariant[] = [];
	let variantsLoading = true;
	let showRegisterForm = false;
	let registering = false;
	let newVariant = {
		name: '',
		base_model: '',
		variant_model: '',
		description: '',
		dataset_size: 0,
		epochs: 0,
		learning_rate: 0,
		loss: 0,
		training_duration_seconds: 0,
	};

	// -- Compare state --
	let compareVariantId = '';
	let comparePrompts = '';
	let comparing = false;
	let compareResult: FineTuneCompareResponse | null = null;

	// -- Quality state --
	let qualityScores: FineTuneQualityScore[] = [];
	let qualityLoading = false;

	onMount(loadAll);

	async function loadAll() {
		await Promise.all([loadVariants(), loadQuality()]);
	}

	// -- Export functions --
	async function handlePreview() {
		previewing = true;
		preview = null;
		try {
			preview = await previewExport({
				format: exportFormat,
				model: exportModel || undefined,
				min_quality: exportMinQuality,
				min_turns: exportMinTurns,
			});
		} catch (e) {
			toastError(`Preview failed: ${e}`);
		} finally {
			previewing = false;
		}
	}

	async function handleExport() {
		exporting = true;
		exportResult = null;
		try {
			const result = await exportTrainingData({
				format: exportFormat,
				model: exportModel || undefined,
				min_quality: exportMinQuality,
				min_turns: exportMinTurns,
			});
			exportResult = result;
			toastSuccess(`Exported ${result.conversation_count} conversations`);
		} catch (e) {
			toastError(`Export failed: ${e}`);
		} finally {
			exporting = false;
		}
	}

	function downloadExport() {
		if (!exportResult?.data) return;
		const ext = exportFormat === 'jsonl' ? 'jsonl' : 'json';
		const mime = exportFormat === 'jsonl' ? 'application/jsonl' : 'application/json';
		const blob = new Blob([exportResult.data], { type: mime });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `training_data_${exportFormat}.${ext}`;
		a.click();
		URL.revokeObjectURL(url);
	}

	// -- Variant functions --
	async function loadVariants() {
		variantsLoading = true;
		try {
			const resp = await listVariants();
			variants = resp.variants;
		} catch (e) {
			toastError(`Failed to load variants: ${e}`);
		} finally {
			variantsLoading = false;
		}
	}

	async function handleRegister() {
		if (!newVariant.name || !newVariant.base_model || !newVariant.variant_model) {
			toastError('Name, base model, and variant model are required');
			return;
		}
		registering = true;
		try {
			await registerVariant(newVariant);
			toastSuccess(`Variant "${newVariant.name}" registered`);
			showRegisterForm = false;
			newVariant = { name: '', base_model: '', variant_model: '', description: '', dataset_size: 0, epochs: 0, learning_rate: 0, loss: 0, training_duration_seconds: 0 };
			await loadVariants();
		} catch (e) {
			toastError(`Registration failed: ${e}`);
		} finally {
			registering = false;
		}
	}

	async function handleUnregister(id: string, name: string) {
		if (!confirm(`Unregister variant "${name}"? This also deletes comparison history.`)) return;
		try {
			await unregisterVariant(id);
			toastSuccess(`Variant "${name}" unregistered`);
			await loadVariants();
		} catch (e) {
			toastError(`Failed to unregister: ${e}`);
		}
	}

	async function handleCompare() {
		if (!compareVariantId || !comparePrompts.trim()) {
			toastError('Select a variant and enter at least one prompt');
			return;
		}
		comparing = true;
		compareResult = null;
		const prompts = comparePrompts.split('\n').map(p => p.trim()).filter(Boolean);
		try {
			compareResult = await runComparison({ variant_id: compareVariantId, prompts });
			toastSuccess('Comparison completed');
		} catch (e) {
			toastError(`Comparison failed: ${e}`);
		} finally {
			comparing = false;
		}
	}

	// -- Quality functions --
	async function loadQuality() {
		qualityLoading = true;
		try {
			const resp = await getQualityScores({ limit: 50 });
			qualityScores = resp.scores;
		} catch {
			/* silent -- quality may have no data */
		} finally {
			qualityLoading = false;
		}
	}

	function formatScore(score: number): string {
		return (score * 100).toFixed(1) + '%';
	}
</script>

<div class="space-y-4">
	<!-- Sub-tab navigation -->
	<div class="flex gap-1 rounded-lg p-1" style="background-color: var(--oo-bg-sunken);">
		{#each [
			{ key: 'export', label: 'Export' },
			{ key: 'variants', label: 'Variants' },
			{ key: 'quality', label: 'Quality' },
		] as tab}
			<button
				class="flex-1 px-3 py-1.5 text-xs font-medium rounded-md transition-colors"
				style="background-color: {activeSubTab === tab.key ? 'var(--oo-bg-elevated)' : 'transparent'}; color: {activeSubTab === tab.key ? 'var(--oo-fg-primary)' : 'var(--oo-fg-muted)'};"
				on:click={() => (activeSubTab = tab.key)}
			>
				{tab.label}
			</button>
		{/each}
	</div>

	<!-- EXPORT TAB -->
	{#if activeSubTab === 'export'}
		<div class="space-y-3">
			<!-- FT-05 (S194): unencrypted-export warning recommended by FT-01 -->
			<p
				class="rounded-md px-3 py-2 text-xs"
				style="background-color: var(--oo-warning-bg); color: var(--oo-warning); border: 1px solid var(--oo-warning-bd);"
			>
				This export is unencrypted and contains your conversation content in
				clear text. The dataset leaves the encrypted store; handle the file
				accordingly.
			</p>
			<!-- Format selector -->
			<div class="space-y-1">
				<label class="text-xs font-medium" style="color: var(--oo-fg-secondary);">Format</label>
				<select
					bind:value={exportFormat}
					class="w-full rounded-md px-2 py-1.5 text-sm"
					style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default);"
				>
					<option value="sharegpt">ShareGPT JSON</option>
					<option value="alpaca">Alpaca JSON</option>
					<option value="jsonl">JSONL</option>
				</select>
			</div>

			<!-- Filters -->
			<div class="grid grid-cols-2 gap-2">
				<div class="space-y-1">
					<label class="text-xs font-medium" style="color: var(--oo-fg-secondary);">Model filter</label>
					<input
						type="text"
						bind:value={exportModel}
						placeholder="All models"
						class="w-full rounded-md px-2 py-1.5 text-sm"
						style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default);"
					/>
				</div>
				<div class="space-y-1">
					<label class="text-xs font-medium" style="color: var(--oo-fg-secondary);">Min turns</label>
					<input
						type="number"
						bind:value={exportMinTurns}
						min="1"
						class="w-full rounded-md px-2 py-1.5 text-sm"
						style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default);"
					/>
				</div>
			</div>

			<div class="space-y-1">
				<label class="text-xs font-medium" style="color: var(--oo-fg-secondary);">
					Min quality: {formatScore(exportMinQuality)}
				</label>
				<input
					type="range"
					bind:value={exportMinQuality}
					min="0"
					max="1"
					step="0.05"
					class="w-full"
				/>
			</div>

			<!-- Action buttons -->
			<div class="flex gap-2">
				<button
					class="flex-1 px-3 py-1.5 text-xs font-medium rounded-md transition-colors"
					style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default);"
					disabled={previewing}
					on:click={handlePreview}
				>
					{previewing ? 'Loading...' : 'Preview'}
				</button>
				<button
					class="flex-1 px-3 py-1.5 text-xs font-medium rounded-md transition-colors"
					style="background-color: var(--oo-accent); color: var(--oo-fg-on-accent);"
					disabled={exporting}
					on:click={handleExport}
				>
					{exporting ? 'Exporting...' : 'Export'}
				</button>
			</div>

			<!-- Preview results -->
			{#if preview}
				<div class="rounded-lg p-3 space-y-2" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
					<div class="text-xs font-medium" style="color: var(--oo-fg-secondary);">
						Preview: {preview.total_conversations} conversations, {preview.total_messages} messages
					</div>
					{#if preview.sample_data}
						<pre class="text-xs rounded-md p-2 overflow-x-auto max-h-48" style="background-color: var(--oo-bg-sunken); color: var(--oo-fg-muted);">{preview.sample_data.slice(0, 2000)}{preview.sample_data.length > 2000 ? '\n...(truncated)' : ''}</pre>
					{/if}
				</div>
			{/if}

			<!-- Export result -->
			{#if exportResult}
				<div class="rounded-lg p-3 space-y-2" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
					<div class="flex items-center justify-between">
						<span class="text-xs font-medium" style="color: var(--oo-fg-secondary);">
							Exported {exportResult.conversation_count} conversations ({exportResult.message_count} messages)
						</span>
						<button
							class="px-3 py-1 text-xs rounded-md"
							style="background-color: var(--oo-accent); color: var(--oo-fg-on-accent);"
							on:click={downloadExport}
						>
							Download
						</button>
					</div>
				</div>
			{/if}
		</div>
	{/if}

	<!-- VARIANTS TAB -->
	{#if activeSubTab === 'variants'}
		<div class="space-y-3">
			<!-- Register button -->
			<button
				class="w-full px-3 py-1.5 text-xs font-medium rounded-md transition-colors"
				style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default);"
				on:click={() => (showRegisterForm = !showRegisterForm)}
			>
				{showRegisterForm ? 'Cancel' : 'Register New Variant'}
			</button>

			<!-- Register form -->
			{#if showRegisterForm}
				<div class="rounded-lg p-3 space-y-2" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
					<div class="grid grid-cols-2 gap-2">
						<div class="space-y-1">
							<label class="text-xs" style="color: var(--oo-fg-muted);">Name</label>
							<input type="text" bind:value={newVariant.name} placeholder="My fine-tuned model"
								class="w-full rounded-md px-2 py-1 text-xs"
								style="background-color: var(--oo-bg-sunken); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default);" />
						</div>
						<div class="space-y-1">
							<label class="text-xs" style="color: var(--oo-fg-muted);">Base model</label>
							<input type="text" bind:value={newVariant.base_model} placeholder="qwen3:32b"
								class="w-full rounded-md px-2 py-1 text-xs"
								style="background-color: var(--oo-bg-sunken); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default);" />
						</div>
						<div class="space-y-1">
							<label class="text-xs" style="color: var(--oo-fg-muted);">Variant model</label>
							<input type="text" bind:value={newVariant.variant_model} placeholder="qwen3:32b-finetuned"
								class="w-full rounded-md px-2 py-1 text-xs"
								style="background-color: var(--oo-bg-sunken); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default);" />
						</div>
						<div class="space-y-1">
							<label class="text-xs" style="color: var(--oo-fg-muted);">Epochs</label>
							<input type="number" bind:value={newVariant.epochs} min="0"
								class="w-full rounded-md px-2 py-1 text-xs"
								style="background-color: var(--oo-bg-sunken); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default);" />
						</div>
					</div>
					<div class="space-y-1">
						<label class="text-xs" style="color: var(--oo-fg-muted);">Description</label>
						<input type="text" bind:value={newVariant.description} placeholder="Optional description"
							class="w-full rounded-md px-2 py-1 text-xs"
							style="background-color: var(--oo-bg-sunken); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default);" />
					</div>
					<button
						class="w-full px-3 py-1.5 text-xs font-medium rounded-md"
						style="background-color: var(--oo-accent); color: var(--oo-fg-on-accent);"
						disabled={registering}
						on:click={handleRegister}
					>
						{registering ? 'Registering...' : 'Register'}
					</button>
				</div>
			{/if}

			<!-- Variant list -->
			{#if variantsLoading}
				<div class="text-xs py-4 text-center" style="color: var(--oo-fg-muted);">Loading variants...</div>
			{:else if variants.length === 0}
				<div class="text-xs py-4 text-center" style="color: var(--oo-fg-muted);">No fine-tuned variants registered yet.</div>
			{:else}
				{#each variants as v}
					<div class="rounded-lg p-3 space-y-1.5" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
						<div class="flex items-center justify-between">
							<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">{v.name}</span>
							<div class="flex items-center gap-2">
								<span class="text-xs px-1.5 py-0.5 rounded"
									style="background-color: var(--oo-bg-sunken); color: {v.status === 'active' ? 'var(--oo-fg-success)' : 'var(--oo-fg-muted)'};">
									{v.status}
								</span>
								<button
									class="text-xs px-2 py-0.5 rounded"
									style="color: var(--oo-fg-error);"
									on:click={() => handleUnregister(v.variant_id, v.name)}
								>
									Remove
								</button>
							</div>
						</div>
						<div class="grid grid-cols-2 gap-1 text-xs" style="color: var(--oo-fg-muted);">
							<div>Base: <span style="color: var(--oo-fg-secondary);">{v.base_model}</span></div>
							<div>Variant: <span style="color: var(--oo-fg-secondary);">{v.variant_model}</span></div>
							{#if v.epochs > 0}
								<div>Epochs: <span style="color: var(--oo-fg-secondary);">{v.epochs}</span></div>
							{/if}
							{#if v.dataset_size > 0}
								<div>Dataset: <span style="color: var(--oo-fg-secondary);">{v.dataset_size} samples</span></div>
							{/if}
							{#if v.loss > 0}
								<div>Loss: <span style="color: var(--oo-fg-secondary);">{v.loss.toFixed(4)}</span></div>
							{/if}
						</div>
						{#if v.description}
							<div class="text-xs" style="color: var(--oo-fg-muted);">{v.description}</div>
						{/if}
					</div>
				{/each}
			{/if}

			<!-- A/B Comparison section -->
			{#if variants.length > 0}
				<div class="space-y-2 pt-2" style="border-top: 1px solid var(--oo-bd-default);">
					<span class="text-xs font-medium" style="color: var(--oo-fg-secondary);">A/B Comparison</span>
					<select
						bind:value={compareVariantId}
						class="w-full rounded-md px-2 py-1.5 text-xs"
						style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default);"
					>
						<option value="">Select variant...</option>
						{#each variants as v}
							<option value={v.variant_id}>{v.name} ({v.base_model} vs {v.variant_model})</option>
						{/each}
					</select>
					<textarea
						bind:value={comparePrompts}
						placeholder="Enter prompts (one per line)"
						rows="3"
						class="w-full rounded-md px-2 py-1.5 text-xs"
						style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default); resize: vertical;"
					/>
					<button
						class="w-full px-3 py-1.5 text-xs font-medium rounded-md"
						style="background-color: var(--oo-accent); color: var(--oo-fg-on-accent);"
						disabled={comparing}
						on:click={handleCompare}
					>
						{comparing ? 'Comparing...' : 'Run Comparison'}
					</button>

					{#if compareResult}
						<div class="rounded-lg p-3 space-y-2" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
							<div class="text-xs font-medium" style="color: var(--oo-fg-secondary);">Results</div>
							<div class="grid grid-cols-3 gap-2 text-xs text-center">
								<div class="rounded-md p-2" style="background-color: var(--oo-bg-sunken);">
									<div style="color: var(--oo-fg-muted);">Base wins</div>
									<div class="text-lg font-bold" style="color: var(--oo-fg-primary);">{compareResult.base_wins}</div>
								</div>
								<div class="rounded-md p-2" style="background-color: var(--oo-bg-sunken);">
									<div style="color: var(--oo-fg-muted);">Ties</div>
									<div class="text-lg font-bold" style="color: var(--oo-fg-primary);">{compareResult.ties}</div>
								</div>
								<div class="rounded-md p-2" style="background-color: var(--oo-bg-sunken);">
									<div style="color: var(--oo-fg-muted);">Variant wins</div>
									<div class="text-lg font-bold" style="color: var(--oo-fg-primary);">{compareResult.variant_wins}</div>
								</div>
							</div>
							{#if compareResult.summary}
								<div class="text-xs" style="color: var(--oo-fg-muted);">{compareResult.summary}</div>
							{/if}
						</div>
					{/if}
				</div>
			{/if}
		</div>
	{/if}

	<!-- QUALITY TAB -->
	{#if activeSubTab === 'quality'}
		<div class="space-y-3">
			<div class="flex items-center justify-between">
				<span class="text-xs font-medium" style="color: var(--oo-fg-secondary);">Conversation Quality Scores</span>
				<button
					class="px-2 py-1 text-xs rounded-md"
					style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-primary); border: 1px solid var(--oo-bd-default);"
					disabled={qualityLoading}
					on:click={loadQuality}
				>
					{qualityLoading ? 'Loading...' : 'Refresh'}
				</button>
			</div>

			{#if qualityScores.length === 0}
				<div class="text-xs py-4 text-center" style="color: var(--oo-fg-muted);">
					No quality scores available. Start conversations and provide feedback to generate scores.
				</div>
			{:else}
				{#each qualityScores as qs}
					<div class="rounded-lg p-3 space-y-1" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
						<div class="flex items-center justify-between">
							<span class="text-xs font-mono" style="color: var(--oo-fg-secondary);">
								{qs.conversation_id.slice(0, 12)}...
							</span>
							<span class="text-sm font-bold" style="color: var(--oo-accent);">
								{formatScore(qs.combined_score)}
							</span>
						</div>
						<div class="grid grid-cols-3 gap-1 text-xs" style="color: var(--oo-fg-muted);">
							<div>Feedback: {formatScore(qs.feedback_score)}</div>
							<div>Benchmark: {formatScore(qs.benchmark_score)}</div>
							<div>Entries: {qs.feedback_count}</div>
						</div>
						<div class="flex gap-2 text-xs">
							{#if qs.has_feedback}
								<span class="px-1.5 py-0.5 rounded" style="background-color: var(--oo-bg-sunken); color: var(--oo-fg-success);">Feedback</span>
							{/if}
							{#if qs.has_benchmarks}
								<span class="px-1.5 py-0.5 rounded" style="background-color: var(--oo-bg-sunken); color: var(--oo-fg-success);">Benchmarks</span>
							{/if}
							{#if !qs.has_feedback && !qs.has_benchmarks}
								<span class="px-1.5 py-0.5 rounded" style="background-color: var(--oo-bg-sunken); color: var(--oo-fg-muted);">Default score</span>
							{/if}
						</div>
					</div>
				{/each}
			{/if}
		</div>
	{/if}
</div>
