<!--
  CacheStatsPanel — S68 Semantic Cache settings panel.

  Sections:
  1. Enable/disable toggle + embeddings availability
  2. Hit rate gauge (exact vs semantic breakdown)
  3. Tokens saved counter + entry count bar
  4. Config sliders (TTL, threshold, max entries)
  5. Scope selector + clear cache button
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getS68CacheStatus,
		toggleS68Cache,
		updateS68CacheConfig,
		clearS68Cache,
		expireS68Cache,
	} from '$lib/api/semanticCache';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type { S68CacheStats, S68CacheStatus } from '$lib/types';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	let loading = true;
	let error = '';

	let status: S68CacheStatus | null = null;
	let stats: S68CacheStats | null = null;

	// Config edits
	let localEnabled = false;
	let localThreshold = 0.92;
	let localTtl = 3600;
	let localMaxEntries = 1000;
	let localScope: string = 'global';
	let localExactEnabled = true;
	let localSemanticEnabled = true;
	let savingConfig = false;

	// Clear
	let clearing = false;
	let expiring = false;

	// -------------------------------------------------------------------------
	// Load
	// -------------------------------------------------------------------------

	onMount(loadData);

	async function loadData() {
		loading = true;
		error = '';
		try {
			status = await getS68CacheStatus();
			stats = status.stats ?? null;
			if (status.config) {
				localEnabled = (status.config.enabled as boolean) ?? false;
				localThreshold = (status.config.similarity_threshold as number) ?? 0.92;
				localTtl = (status.config.ttl_seconds as number) ?? 3600;
				localMaxEntries = (status.config.max_entries as number) ?? 1000;
				localScope = (status.config.scope as string) ?? 'global';
				localExactEnabled = (status.config.exact_match_enabled as boolean) ?? true;
				localSemanticEnabled = (status.config.semantic_match_enabled as boolean) ?? true;
			}
		} catch (e) {
			error = `Failed to load cache status: ${e}`;
		} finally {
			loading = false;
		}
	}

	// -------------------------------------------------------------------------
	// Actions
	// -------------------------------------------------------------------------

	async function handleToggle() {
		try {
			status = await toggleS68Cache();
			localEnabled = status.enabled;
			stats = status.stats ?? null;
			toastSuccess(`Cache ${localEnabled ? 'enabled' : 'disabled'}`);
		} catch (e) {
			toastError(`Toggle failed: ${e}`);
		}
	}

	async function handleSaveConfig() {
		savingConfig = true;
		try {
			status = await updateS68CacheConfig({
				enabled: localEnabled,
				similarity_threshold: localThreshold,
				ttl_seconds: localTtl,
				max_entries: localMaxEntries,
				scope: localScope,
				exact_match_enabled: localExactEnabled,
				semantic_match_enabled: localSemanticEnabled,
			});
			stats = status.stats ?? null;
			toastSuccess('Cache configuration saved');
		} catch (e) {
			toastError(`Save failed: ${e}`);
		} finally {
			savingConfig = false;
		}
	}

	async function handleClear() {
		clearing = true;
		try {
			const result = await clearS68Cache();
			toastSuccess(`Cleared ${result.entries_removed} entries`);
			await loadData();
		} catch (e) {
			toastError(`Clear failed: ${e}`);
		} finally {
			clearing = false;
		}
	}

	async function handleExpire() {
		expiring = true;
		try {
			const result = await expireS68Cache();
			toastSuccess(`Expired ${result.entries_removed} stale entries`);
			await loadData();
		} catch (e) {
			toastError(`Expire failed: ${e}`);
		} finally {
			expiring = false;
		}
	}

	// -------------------------------------------------------------------------
	// Helpers
	// -------------------------------------------------------------------------

	function pct(v: number): string {
		return (v * 100).toFixed(1) + '%';
	}

	function formatBytes(b: number): string {
		if (b < 1024) return b + ' B';
		if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
		return (b / 1048576).toFixed(1) + ' MB';
	}

	function formatTtl(seconds: number): string {
		if (seconds < 60) return seconds + 's';
		if (seconds < 3600) return Math.round(seconds / 60) + 'min';
		return (seconds / 3600).toFixed(1) + 'h';
	}
</script>

<div class="space-y-4">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<h3 class="text-sm font-medium" style="color: var(--oo-fg-primary);">
			Semantic Cache
		</h3>
		{#if !loading}
			<span class="text-xs px-2 py-0.5 rounded-full"
				style="{status?.available
					? 'background-color: var(--oo-success-bg); color: var(--oo-success);'
					: 'background-color: var(--oo-error-bg); color: var(--oo-error);'}">
				{status?.available ? 'Available' : 'Unavailable'}
			</span>
		{/if}
	</div>

	{#if loading}
		<p class="text-xs" style="color: var(--oo-fg-muted);">Loading cache status...</p>
	{:else if error}
		<p class="text-xs" style="color: var(--oo-error);">{error}</p>
	{:else}
		<!-- 1. Enable toggle + embedding status -->
		<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
			<div class="flex items-center justify-between mb-2">
				<span class="text-xs" style="color: var(--oo-fg-secondary);">Cache Enabled</span>
				<button
					on:click={handleToggle}
					class="relative w-9 h-5 rounded-full transition-colors"
					style="{localEnabled
						? 'background-color: var(--oo-acc-500);'
						: 'background-color: var(--oo-bg-tertiary);'}"
					aria-label="Toggle cache"
				>
					<span
						class="absolute top-0.5 w-4 h-4 rounded-full transition-transform"
						style="background-color: var(--oo-toggle-knob); {localEnabled ? 'left: 1.125rem;' : 'left: 0.125rem;'}"
					/>
				</button>
			</div>
			<div class="flex items-center gap-2 text-xs" style="color: var(--oo-fg-muted);">
				<span class="inline-block w-2 h-2 rounded-full"
					style="{stats?.embeddings_available
						? 'background-color: var(--oo-success);'
						: 'background-color: var(--oo-error);'}"></span>
				Embeddings: {stats?.embeddings_available ? stats.embedding_model : 'unavailable (exact-only mode)'}
			</div>
		</div>

		<!-- 2. Hit rate breakdown -->
		{#if stats}
			<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
				<div class="text-xs font-medium mb-2" style="color: var(--oo-fg-secondary);">Hit Rate</div>
				<div class="flex items-center gap-3 mb-2">
					<span class="text-2xl font-bold" style="color: var(--oo-acc-300);">
						{pct(stats.hit_rate)}
					</span>
					<div class="flex-1 text-xs space-y-1" style="color: var(--oo-fg-muted);">
						<div class="flex justify-between">
							<span>Exact</span>
							<span>{stats.exact_hits} ({pct(stats.exact_hit_rate)})</span>
						</div>
						<div class="flex justify-between">
							<span>Semantic</span>
							<span>{stats.semantic_hits} ({pct(stats.semantic_hit_rate)})</span>
						</div>
						<div class="flex justify-between">
							<span>Misses</span>
							<span>{stats.total_misses}</span>
						</div>
					</div>
				</div>
				<!-- Hit rate bar -->
				<div class="h-2 rounded-full overflow-hidden flex" style="background-color: var(--oo-bg-tertiary);">
					{#if stats.exact_hit_rate > 0}
						<div class="h-full" style="width: {stats.exact_hit_rate * 100}%; background-color: var(--oo-acc-500);"></div>
					{/if}
					{#if stats.semantic_hit_rate > 0}
						<div class="h-full" style="width: {stats.semantic_hit_rate * 100}%; background-color: var(--oo-acc-300);"></div>
					{/if}
				</div>
			</div>

			<!-- 3. Tokens saved + entries -->
			<div class="grid grid-cols-2 gap-3">
				<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
					<div class="text-xs" style="color: var(--oo-fg-muted);">Tokens Saved</div>
					<div class="text-lg font-bold" style="color: var(--oo-acc-300);">
						{stats.tokens_saved.toLocaleString()}
					</div>
				</div>
				<div class="p-3 rounded-lg" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
					<div class="text-xs" style="color: var(--oo-fg-muted);">Entries</div>
					<div class="text-lg font-bold" style="color: var(--oo-fg-primary);">
						{stats.total_entries}
						<span class="text-xs font-normal" style="color: var(--oo-fg-muted);">/ {stats.max_entries}</span>
					</div>
					<!-- Entry count bar -->
					<div class="h-1.5 rounded-full mt-1" style="background-color: var(--oo-bg-tertiary);">
						<div class="h-full rounded-full" style="width: {Math.min(100, (stats.total_entries / stats.max_entries) * 100)}%; background-color: var(--oo-acc-500);"></div>
					</div>
				</div>
			</div>

			<!-- DB size -->
			<div class="text-xs" style="color: var(--oo-fg-muted);">
				Database size: {formatBytes(stats.size_bytes)}
			</div>
		{/if}

		<!-- 4. Config sliders -->
		<div class="p-3 rounded-lg space-y-3" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default);">
			<div class="text-xs font-medium" style="color: var(--oo-fg-secondary);">Configuration</div>

			<!-- Similarity threshold -->
			<div>
				<div class="flex justify-between text-xs mb-1">
					<label for="cache-threshold" style="color: var(--oo-fg-muted);">Similarity Threshold</label>
					<span style="color: var(--oo-fg-primary);">{localThreshold.toFixed(2)}</span>
				</div>
				<input id="cache-threshold" type="range" min="0.5" max="0.99" step="0.01"
					bind:value={localThreshold}
					class="w-full h-1.5 rounded-full appearance-none cursor-pointer"
					style="background-color: var(--oo-bg-tertiary);" />
			</div>

			<!-- TTL -->
			<div>
				<div class="flex justify-between text-xs mb-1">
					<label for="cache-ttl" style="color: var(--oo-fg-muted);">TTL</label>
					<span style="color: var(--oo-fg-primary);">{formatTtl(localTtl)}</span>
				</div>
				<input id="cache-ttl" type="range" min="60" max="86400" step="60"
					bind:value={localTtl}
					class="w-full h-1.5 rounded-full appearance-none cursor-pointer"
					style="background-color: var(--oo-bg-tertiary);" />
			</div>

			<!-- Max entries -->
			<div>
				<div class="flex justify-between text-xs mb-1">
					<label for="cache-max" style="color: var(--oo-fg-muted);">Max Entries</label>
					<span style="color: var(--oo-fg-primary);">{localMaxEntries}</span>
				</div>
				<input id="cache-max" type="range" min="50" max="10000" step="50"
					bind:value={localMaxEntries}
					class="w-full h-1.5 rounded-full appearance-none cursor-pointer"
					style="background-color: var(--oo-bg-tertiary);" />
			</div>

			<!-- Match type toggles -->
			<div class="flex gap-4 text-xs" style="color: var(--oo-fg-muted);">
				<label class="flex items-center gap-1.5 cursor-pointer">
					<input type="checkbox" bind:checked={localExactEnabled} class="rounded" />
					Exact match
				</label>
				<label class="flex items-center gap-1.5 cursor-pointer">
					<input type="checkbox" bind:checked={localSemanticEnabled} class="rounded" />
					Semantic match
				</label>
			</div>

			<!-- Scope selector -->
			<div>
				<label for="cache-scope" class="text-xs" style="color: var(--oo-fg-muted);">Scope</label>
				<select id="cache-scope"
					bind:value={localScope}
					class="ml-2 text-xs rounded px-2 py-1"
					style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary);
						border: 1px solid var(--oo-input-bd);">
					<option value="global">Global</option>
					<option value="conversation">Per-Conversation</option>
				</select>
			</div>

			<!-- Save config button -->
			<button
				on:click={handleSaveConfig}
				disabled={savingConfig}
				class="w-full px-3 py-1.5 rounded text-xs font-medium transition-colors"
				style="background-color: var(--oo-btn-primary-bg); color: var(--oo-btn-primary-fg);
					opacity: {savingConfig ? '0.5' : '1'};"
			>
				{savingConfig ? 'Saving...' : 'Save Configuration'}
			</button>
		</div>

		<!-- 5. Actions -->
		<div class="flex gap-2">
			<button
				on:click={handleClear}
				disabled={clearing}
				class="flex-1 px-3 py-1.5 rounded text-xs font-medium transition-colors"
				style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-secondary);
					border: 1px solid var(--oo-bd-default); opacity: {clearing ? '0.5' : '1'};"
			>
				{clearing ? 'Clearing...' : 'Clear Cache'}
			</button>
			<button
				on:click={handleExpire}
				disabled={expiring}
				class="flex-1 px-3 py-1.5 rounded text-xs font-medium transition-colors"
				style="background-color: var(--oo-bg-elevated); color: var(--oo-fg-secondary);
					border: 1px solid var(--oo-bd-default); opacity: {expiring ? '0.5' : '1'};"
			>
				{expiring ? 'Expiring...' : 'Expire Stale'}
			</button>
		</div>
	{/if}
</div>
