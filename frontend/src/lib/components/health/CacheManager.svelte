<!--
  CacheManager.svelte
  Response + semantic cache statistics and management, lifted into System
  Status. Presentation moves to the ds primitives and --oo-* tokens; the cache
  API and all behaviour (refresh, clear all, clear by model) are unchanged.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { Card, Button, Select } from '$lib/ds';
	import type { CacheCombinedStats } from '$lib/types';
	import { getCacheStats, clearAllCache, clearModelCache } from '$lib/api/cache';

	let stats: CacheCombinedStats | null = null;
	let loading = true;
	let error = '';

	// Clear state
	let confirmClearAll = false;
	let clearingAll = false;
	let clearModelName = '';
	let clearingModel = false;
	let clearMessage = '';

	async function load() {
		loading = true;
		error = '';
		try {
			stats = await getCacheStats();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load cache stats';
		} finally {
			loading = false;
		}
	}

	async function handleClearAll() {
		clearingAll = true;
		clearMessage = '';
		try {
			const result = await clearAllCache();
			clearMessage = `Cleared ${result.entries_removed} entries`;
			confirmClearAll = false;
			await load();
		} catch (e) {
			clearMessage = e instanceof Error ? e.message : 'Failed to clear cache';
		} finally {
			clearingAll = false;
		}
	}

	async function handleClearModel() {
		if (!clearModelName.trim()) return;
		clearingModel = true;
		clearMessage = '';
		try {
			const result = await clearModelCache(clearModelName);
			clearMessage = `Cleared ${result.entries_removed} entries for ${clearModelName}`;
			clearModelName = '';
			await load();
		} catch (e) {
			clearMessage = e instanceof Error ? e.message : 'Failed to clear model cache';
		} finally {
			clearingModel = false;
		}
	}

	function formatBytes(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
	}

	function formatPercent(rate: number): string {
		return `${(rate * 100).toFixed(1)}%`;
	}

	function onModelChange(e: CustomEvent<string | string[]>) {
		if (typeof e.detail === 'string') clearModelName = e.detail;
	}

	$: hitRate = stats?.response_cache?.hit_rate ?? 0;
	$: hitRateState = hitRate > 0.7 ? 'good' : hitRate > 0.3 ? 'mid' : 'low';
	$: modelEntries = stats?.response_cache?.entries_by_model
		? Object.entries(stats.response_cache.entries_by_model)
		: [];
	$: modelOptions = [
		{ value: '', label: 'Select model...' },
		...modelEntries.map(([model]) => ({ value: model, label: model })),
	];

	onMount(load);
</script>

<div class="flex flex-col gap-5">
	{#if loading}
		<div class="flex items-center gap-2 text-sm py-4 justify-center" style="color: var(--oo-fg-muted);">
			<span class="oo-spin" aria-hidden="true"></span>
			Loading cache stats...
		</div>
	{:else if error}
		<Card variant="flat" padding="sm" class="oo-cache-error">
			<div class="flex items-center justify-between gap-3 text-sm">
				<span>{error}</span>
				<Button variant="ghost" size="sm" on:click={load}>Retry</Button>
			</div>
		</Card>
	{:else}
		<!-- Response cache -->
		<div>
			<div class="flex items-center justify-between mb-2">
				<h3 class="text-sm font-medium" style="color: var(--oo-fg-secondary);">Response cache</h3>
				<Button variant="ghost" size="sm" iconOnly="refresh-cw" ariaLabel="Refresh cache stats" on:click={load} />
			</div>

			{#if stats?.response_cache}
				<Card variant="flat" padding="md">
					<div class="flex flex-col gap-3">
						<div>
							<div class="flex justify-between text-xs mb-1">
								<span style="color: var(--oo-fg-muted);">Hit rate</span>
								<span class="font-mono" style="color: var(--oo-fg-primary);">{formatPercent(hitRate)}</span>
							</div>
							<div class="oo-bar">
								<div class="oo-bar-fill" data-state={hitRateState} style="width: {Math.min(hitRate * 100, 100)}%"></div>
							</div>
						</div>

						<div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
							<div>
								<span style="color: var(--oo-fg-faint);">Entries</span>
								<div class="font-mono" style="color: var(--oo-fg-primary);">{stats.response_cache.total_entries}</div>
							</div>
							<div>
								<span style="color: var(--oo-fg-faint);">Hits</span>
								<div class="font-mono" style="color: var(--oo-success);">{stats.response_cache.total_hits}</div>
							</div>
							<div>
								<span style="color: var(--oo-fg-faint);">Misses</span>
								<div class="font-mono" style="color: var(--oo-fg-muted);">{stats.response_cache.total_misses}</div>
							</div>
							<div>
								<span style="color: var(--oo-fg-faint);">Size</span>
								<div class="font-mono" style="color: var(--oo-fg-primary);">{formatBytes(stats.response_cache.total_size_bytes)}</div>
							</div>
						</div>

						{#if modelEntries.length > 0}
							<div>
								<span class="text-xs" style="color: var(--oo-fg-faint);">By model</span>
								<div class="flex flex-wrap gap-1 mt-1">
									{#each modelEntries as [model, count]}
										<span class="text-xs px-1.5 py-0.5 rounded font-mono" style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-secondary);">
											{model}: {count}
										</span>
									{/each}
								</div>
							</div>
						{/if}
					</div>
				</Card>
			{:else}
				<p class="text-xs" style="color: var(--oo-fg-faint);">Response cache not available.</p>
			{/if}
		</div>

		<!-- Semantic cache -->
		<div>
			<h3 class="text-sm font-medium mb-2" style="color: var(--oo-fg-secondary);">Semantic cache</h3>
			{#if stats?.semantic_cache}
				<Card variant="flat" padding="md">
					<div class="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
						<div>
							<span style="color: var(--oo-fg-faint);">Embeddings</span>
							<div class="font-mono" style="color: var(--oo-fg-primary);">{stats.semantic_cache.total_embeddings}</div>
						</div>
						<div>
							<span style="color: var(--oo-fg-faint);">Hits</span>
							<div class="font-mono" style="color: var(--oo-success);">{stats.semantic_cache.semantic_hits}</div>
						</div>
						<div>
							<span style="color: var(--oo-fg-faint);">Misses</span>
							<div class="font-mono" style="color: var(--oo-fg-muted);">{stats.semantic_cache.semantic_misses}</div>
						</div>
						<div>
							<span style="color: var(--oo-fg-faint);">Avg similarity</span>
							<div class="font-mono" style="color: var(--oo-fg-primary);">{stats.semantic_cache.avg_similarity.toFixed(3)}</div>
						</div>
						<div>
							<span style="color: var(--oo-fg-faint);">Threshold</span>
							<div class="font-mono" style="color: var(--oo-fg-primary);">{stats.semantic_cache.threshold.toFixed(2)}</div>
						</div>
						{#if stats.semantic_cache.embedding_model}
							<div>
								<span style="color: var(--oo-fg-faint);">Model</span>
								<div class="font-mono text-[10px]" style="color: var(--oo-fg-primary);">{stats.semantic_cache.embedding_model}</div>
							</div>
						{/if}
					</div>
				</Card>
			{:else}
				<p class="text-xs" style="color: var(--oo-fg-faint);">Semantic cache not available.</p>
			{/if}
		</div>

		<!-- Clear actions -->
		<div>
			<h3 class="text-sm font-medium mb-2" style="color: var(--oo-fg-secondary);">Cache management</h3>
			<div class="flex flex-col gap-3">
				<div class="flex items-center gap-2">
					{#if confirmClearAll}
						<span class="text-xs" style="color: var(--oo-error);">Clear all cached responses?</span>
						<Button variant="danger" size="sm" loading={clearingAll} on:click={handleClearAll}>Confirm</Button>
						<Button variant="ghost" size="sm" on:click={() => (confirmClearAll = false)}>Cancel</Button>
					{:else}
						<Button variant="danger" size="sm" on:click={() => (confirmClearAll = true)}>Clear all cache</Button>
					{/if}
				</div>

				<div class="flex items-end gap-2">
					<div class="flex-1">
						<Select
							label="Clear by model"
							hideLabel
							size="sm"
							value={clearModelName}
							options={modelOptions}
							on:change={onModelChange}
						/>
					</div>
					<Button variant="secondary" size="sm" disabled={!clearModelName} loading={clearingModel} on:click={handleClearModel}>
						Clear model
					</Button>
				</div>

				{#if clearMessage}
					<p class="text-xs" style="color: var(--oo-accent);">{clearMessage}</p>
				{/if}
			</div>
		</div>
	{/if}
</div>

<style>
	.oo-spin {
		width: 1rem;
		height: 1rem;
		border: 2px solid var(--oo-bd-default);
		border-top-color: var(--oo-acc-500);
		border-radius: var(--oo-radius-full);
		display: inline-block;
		animation: oo-spin 0.7s linear infinite;
	}
	@keyframes oo-spin {
		to {
			transform: rotate(360deg);
		}
	}
	.oo-bar {
		height: 0.5rem;
		border-radius: var(--oo-radius-full);
		background-color: var(--oo-bg-elevated);
		overflow: hidden;
	}
	.oo-bar-fill {
		height: 100%;
		border-radius: var(--oo-radius-full);
		background-color: var(--oo-fg-muted);
		transition: width var(--oo-motion-default) var(--oo-ease-default);
	}
	.oo-bar-fill[data-state='good'] {
		background-color: var(--oo-success);
	}
	.oo-bar-fill[data-state='mid'] {
		background-color: var(--oo-warning);
	}
	.oo-bar-fill[data-state='low'] {
		background-color: var(--oo-error);
	}
	:global(.oo-cache-error) {
		background-color: var(--oo-error-bg);
		border-color: var(--oo-error-bd);
		color: var(--oo-error);
	}
</style>
