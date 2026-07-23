<!--
  CompressionSettings — Conversation Compressor settings panel.

  Displays:
  - Enable/disable toggle with strategy selector
  - Recent messages keep slider
  - Archive retrieval trigger settings
  - Last compression stats (when available)
  - Archive search UI for manual inspection
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getCompressionConfig,
		updateCompressionConfig,
		reloadCompressionConfig,
		getCompressionStats,
		searchArchive,
		type CompressionConfig,
		type CompressionStats,
		type ArchiveSearchResultItem
	} from '$lib/api/compression';
	import { toastSuccess, toastError } from '$lib/stores/notifications';

	// -------------------------------------------------------------------------
	// State
	// -------------------------------------------------------------------------

	let loading = true;
	let error = '';
	let saving = false;

	let config: CompressionConfig | null = null;

	// Editable local copies (updated on load, saved on change)
	let enabled = true;
	let strategy: 'rule' | 'llm' | 'hybrid' = 'hybrid';
	let recentKeep = 6;
	let triggerEnabled = true;
	let triggerMinConfidence = 0.6;

	// Stats
	let stats: CompressionStats | null = null;
	let statsConvId = '';
	let loadingStats = false;

	// Archive search
	let searchConvId = '';
	let searchQuery = '';
	let searchResults: ArchiveSearchResultItem[] = [];
	let searching = false;
	let searchDone = false;

	// -------------------------------------------------------------------------
	// Load
	// -------------------------------------------------------------------------

	async function loadConfig() {
		loading = true;
		error = '';
		try {
			config = await getCompressionConfig();
			syncLocalFromConfig(config);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load compression config';
		} finally {
			loading = false;
		}
	}

	function syncLocalFromConfig(cfg: CompressionConfig) {
		enabled = cfg.enabled;
		strategy = cfg.strategy;
		recentKeep = cfg.recent_messages_keep;
		triggerEnabled = cfg.retrieval_trigger_enabled;
		triggerMinConfidence = cfg.retrieval_trigger_min_confidence;
	}

	// -------------------------------------------------------------------------
	// Save helpers
	// -------------------------------------------------------------------------

	async function saveField(updates: Record<string, unknown>) {
		saving = true;
		try {
			config = await updateCompressionConfig(updates as Parameters<typeof updateCompressionConfig>[0]);
			syncLocalFromConfig(config);
			toastSuccess('Compression setting updated');
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to update setting');
		} finally {
			saving = false;
		}
	}

	async function handleToggleEnabled() {
		await saveField({ enabled });
	}

	async function handleStrategyChange() {
		await saveField({ strategy });
	}

	async function handleRecentKeepChange() {
		await saveField({ recent_messages_keep: recentKeep });
	}

	async function handleTriggerEnabledChange() {
		await saveField({ retrieval_trigger_enabled: triggerEnabled });
	}

	async function handleConfidenceChange() {
		await saveField({ retrieval_trigger_min_confidence: triggerMinConfidence });
	}

	async function handleReload() {
		try {
			await reloadCompressionConfig();
			toastSuccess('Compression config reloaded from disk');
			await loadConfig();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to reload config');
		}
	}

	// -------------------------------------------------------------------------
	// Stats
	// -------------------------------------------------------------------------

	async function loadStats() {
		if (!statsConvId.trim()) return;
		loadingStats = true;
		try {
			stats = await getCompressionStats(statsConvId.trim());
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to load stats');
			stats = null;
		} finally {
			loadingStats = false;
		}
	}

	// -------------------------------------------------------------------------
	// Archive search
	// -------------------------------------------------------------------------

	async function handleSearch() {
		if (!searchConvId.trim() || !searchQuery.trim()) return;
		searching = true;
		searchDone = false;
		searchResults = [];
		try {
			const resp = await searchArchive(searchConvId.trim(), searchQuery.trim());
			searchResults = resp.results;
			searchDone = true;
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Archive search failed');
		} finally {
			searching = false;
		}
	}

	// -------------------------------------------------------------------------
	// Helpers
	// -------------------------------------------------------------------------

	function strategyLabel(s: string): string {
		switch (s) {
			case 'rule': return 'Rule-based';
			case 'llm': return 'LLM-based';
			case 'hybrid': return 'Hybrid';
			default: return s;
		}
	}

	function strategyDescription(s: string): string {
		switch (s) {
			case 'rule': return 'Fast heuristic extraction — no LLM call needed';
			case 'llm': return 'LLM summarization — higher quality, costs one inference';
			case 'hybrid': return 'Rule first pass, LLM refinement if needed (recommended)';
			default: return '';
		}
	}

	function roleColor(role: string): string {
		return role === 'user' ? 'color: var(--oo-acc-400);' : 'color: var(--oo-success);';
	}

	onMount(loadConfig);
</script>

<div class="space-y-5">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h3 class="text-sm font-medium" style="color: var(--oo-fg-primary);">
				Conversation Compressor
			</h3>
			<p class="text-xs mt-0.5" style="color: var(--oo-fg-tertiary);">
				Compresses history to fit the token budget while keeping the full archive searchable.
			</p>
		</div>
		<button
			on:click={handleReload}
			class="px-3 py-1.5 rounded text-xs font-medium transition-colors shrink-0"
			style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-default);"
		>
			Reload Config
		</button>
	</div>

	{#if loading}
		<div class="flex items-center gap-2 text-sm py-6 justify-center" style="color: var(--oo-fg-tertiary);">
			<div class="w-4 h-4 border-2 rounded-full animate-spin"
				style="border-color: var(--oo-bd-default); border-top-color: var(--oo-acc-500);" />
			Loading...
		</div>
	{:else if error}
		<div class="px-3 py-2 rounded text-sm"
			style="background-color: var(--oo-error-bg); border: 1px solid var(--oo-error-bd); color: var(--oo-error);">
			{error}
			<button on:click={loadConfig} class="ml-2 underline hover:no-underline">Retry</button>
		</div>
	{:else if config}

		<!-- Enable / Strategy -->
		<div class="p-4 rounded-lg space-y-4" style="background-color: var(--oo-bg-secondary); border: 1px solid var(--oo-bd-default);">

			<!-- Enable toggle -->
			<label class="flex items-center justify-between cursor-pointer">
				<div>
					<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">Enable compression</span>
					<p class="text-xs mt-0.5" style="color: var(--oo-fg-tertiary);">
						Compress older messages when history exceeds the token budget.
					</p>
				</div>
				<div class="relative ml-4">
					<input
						type="checkbox"
						bind:checked={enabled}
						on:change={handleToggleEnabled}
						class="sr-only"
						id="compression-toggle"
						disabled={saving}
					/>
					<label
						for="compression-toggle"
						class="block w-10 h-6 rounded-full cursor-pointer transition-colors"
						style="background-color: {enabled ? 'var(--oo-acc-500)' : 'var(--oo-bg-tertiary)'}; border: 1px solid {enabled ? 'var(--oo-acc-500)' : 'var(--oo-bd-default)'};"
					>
						<span
							class="absolute top-1 left-1 w-4 h-4 rounded-full bg-[var(--oo-toggle-knob)] transition-transform"
							style="transform: translateX({enabled ? '16px' : '0px'});"
						/>
					</label>
				</div>
			</label>

			<!-- Strategy selector -->
			{#if enabled}
				<div>
					<label class="block text-xs font-medium mb-2" style="color: var(--oo-fg-secondary);">
						Compression strategy
					</label>
					<div class="grid grid-cols-3 gap-2">
						{#each ['rule', 'llm', 'hybrid'] as s}
							<button
								on:click={() => { strategy = s; handleStrategyChange(); }}
								class="px-3 py-2 rounded text-xs text-left transition-colors"
								style="background-color: {strategy === s ? 'var(--oo-msg-user-bg)' : 'var(--oo-bg-tertiary)'}; border: 1px solid {strategy === s ? 'var(--oo-acc-500)' : 'var(--oo-bd-default)'}; color: {strategy === s ? 'var(--oo-acc-400)' : 'var(--oo-fg-secondary)'};"
								disabled={saving}
							>
								<div class="font-medium">{strategyLabel(s)}</div>
							</button>
						{/each}
					</div>
					<p class="text-[11px] mt-1.5" style="color: var(--oo-fg-tertiary);">
						{strategyDescription(strategy)}
					</p>
				</div>

				<!-- Recent keep slider -->
				<div>
					<div class="flex items-center justify-between mb-1">
						<label class="text-xs font-medium" style="color: var(--oo-fg-secondary);">
							Recent messages kept verbatim
						</label>
						<span class="text-xs font-mono px-2 py-0.5 rounded" style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-primary);">
							{recentKeep}
						</span>
					</div>
					<input
						type="range"
						min="2"
						max="20"
						step="2"
						bind:value={recentKeep}
						on:change={handleRecentKeepChange}
						class="w-full"
						disabled={saving}
					/>
					<div class="flex justify-between text-[10px] mt-0.5" style="color: var(--oo-fg-tertiary);">
						<span>2 (aggressive)</span>
						<span>20 (conservative)</span>
					</div>
				</div>
			{/if}
		</div>

		<!-- Archive retrieval trigger -->
		{#if enabled}
			<div class="p-4 rounded-lg space-y-3" style="background-color: var(--oo-bg-secondary); border: 1px solid var(--oo-bd-default);">
				<div>
					<h4 class="text-sm font-medium" style="color: var(--oo-fg-primary);">Archive Retrieval</h4>
					<p class="text-xs mt-0.5" style="color: var(--oo-fg-tertiary);">
						Automatically inject relevant older messages when user references past context
						("you said…", "we discussed…").
					</p>
				</div>

				<label class="flex items-center gap-3 cursor-pointer">
					<input
						type="checkbox"
						bind:checked={triggerEnabled}
						on:change={handleTriggerEnabledChange}
						class="rounded"
						disabled={saving}
					/>
					<span class="text-sm" style="color: var(--oo-fg-secondary);">Enable trigger detection</span>
				</label>

				{#if triggerEnabled}
					<div>
						<div class="flex items-center justify-between mb-1">
							<label class="text-xs" style="color: var(--oo-fg-secondary);">
								Minimum trigger confidence
							</label>
							<span class="text-xs font-mono px-2 py-0.5 rounded" style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-primary);">
								{triggerMinConfidence.toFixed(2)}
							</span>
						</div>
						<input
							type="range"
							min="0.3"
							max="0.95"
							step="0.05"
							bind:value={triggerMinConfidence}
							on:change={handleConfidenceChange}
							class="w-full"
							disabled={saving}
						/>
						<div class="flex justify-between text-[10px] mt-0.5" style="color: var(--oo-fg-tertiary);">
							<span>0.3 (permissive)</span>
							<span>0.95 (strict)</span>
						</div>
					</div>
				{/if}
			</div>
		{/if}

		<!-- Compression stats -->
		<div class="p-4 rounded-lg space-y-3" style="background-color: var(--oo-bg-secondary); border: 1px solid var(--oo-bd-default);">
			<h4 class="text-sm font-medium" style="color: var(--oo-fg-primary);">Last Compression Stats</h4>

			<div class="flex gap-2">
				<input
					type="text"
					bind:value={statsConvId}
					placeholder="Conversation ID (UUID)"
					class="flex-1 px-3 py-1.5 rounded text-xs font-mono"
					style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
				/>
				<button
					on:click={loadStats}
					disabled={loadingStats || !statsConvId.trim()}
					class="px-3 py-1.5 rounded text-xs font-medium transition-colors disabled:opacity-50"
					style="background-color: var(--oo-acc-500); color: white;"
				>
					{loadingStats ? 'Loading…' : 'Load'}
				</button>
			</div>

			{#if stats}
				{#if !stats.last_compression_available}
					<p class="text-xs" style="color: var(--oo-fg-tertiary);">
						No compression recorded for this conversation yet.
					</p>
				{:else}
					<div class="grid grid-cols-2 gap-2 text-xs">
						<div class="px-3 py-2 rounded" style="background-color: var(--oo-bg-tertiary);">
							<div style="color: var(--oo-fg-tertiary);">Strategy</div>
							<div class="font-medium mt-0.5" style="color: var(--oo-fg-primary);">{stats.strategy_used ?? '—'}</div>
						</div>
						<div class="px-3 py-2 rounded" style="background-color: var(--oo-bg-tertiary);">
							<div style="color: var(--oo-fg-tertiary);">Tokens saved</div>
							<div class="font-medium mt-0.5" style="color: var(--oo-acc-400);">{stats.tokens_saved?.toLocaleString() ?? '—'}</div>
						</div>
						<div class="px-3 py-2 rounded" style="background-color: var(--oo-bg-tertiary);">
							<div style="color: var(--oo-fg-tertiary);">Messages</div>
							<div class="font-medium mt-0.5" style="color: var(--oo-fg-primary);">
								{stats.original_count ?? '—'} → {stats.compressed_count ?? '—'} compressed
							</div>
						</div>
						<div class="px-3 py-2 rounded" style="background-color: var(--oo-bg-tertiary);">
							<div style="color: var(--oo-fg-tertiary);">Ratio</div>
							<div class="font-medium mt-0.5" style="color: var(--oo-fg-primary);">
								{stats.compression_ratio !== null ? (stats.compression_ratio * 100).toFixed(1) + '%' : '—'}
							</div>
						</div>
					</div>
					{#if stats.summary}
						<div>
							<div class="text-[11px] mb-1" style="color: var(--oo-fg-tertiary);">Summary injected into prompt:</div>
							<pre class="text-[11px] p-2 rounded whitespace-pre-wrap overflow-auto max-h-32"
								style="background-color: var(--oo-bg-tertiary); color: var(--oo-fg-secondary);"
							>{stats.summary}</pre>
						</div>
					{/if}
				{/if}
			{/if}
		</div>

		<!-- Archive search -->
		<div class="p-4 rounded-lg space-y-3" style="background-color: var(--oo-bg-secondary); border: 1px solid var(--oo-bd-default);">
			<div>
				<h4 class="text-sm font-medium" style="color: var(--oo-fg-primary);">Archive Search</h4>
				<p class="text-xs mt-0.5" style="color: var(--oo-fg-tertiary);">
					Search the full uncompressed conversation history. Results show messages that
					would be retrieved if the archive trigger fired.
				</p>
			</div>

			<div class="space-y-2">
				<input
					type="text"
					bind:value={searchConvId}
					placeholder="Conversation ID (UUID)"
					class="w-full px-3 py-1.5 rounded text-xs font-mono"
					style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
				/>
				<div class="flex gap-2">
					<input
						type="text"
						bind:value={searchQuery}
						placeholder="Search query (e.g. 'NMDS ordination')"
						class="flex-1 px-3 py-1.5 rounded text-xs"
						style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
						on:keydown={(e) => e.key === 'Enter' && handleSearch()}
					/>
					<button
						on:click={handleSearch}
						disabled={searching || !searchQuery.trim() || !searchConvId.trim()}
						class="px-3 py-1.5 rounded text-xs font-medium transition-colors disabled:opacity-50 shrink-0"
						style="background-color: var(--oo-acc-500); color: white;"
					>
						{searching ? 'Searching…' : 'Search'}
					</button>
				</div>
			</div>

			{#if searchDone}
				{#if searchResults.length === 0}
					<p class="text-xs py-2 text-center" style="color: var(--oo-fg-tertiary);">
						No matching messages found in archive.
					</p>
				{:else}
					<div class="space-y-2">
						<div class="text-[11px]" style="color: var(--oo-fg-tertiary);">
							{searchResults.length} result{searchResults.length !== 1 ? 's' : ''} found
						</div>
						{#each searchResults as result}
							<div class="p-3 rounded text-xs" style="background-color: var(--oo-bg-tertiary); border: 1px solid var(--oo-bd-default);">
								<div class="flex items-center justify-between mb-1">
									<span class="font-medium" style={roleColor(result.role)}>{result.role}</span>
									<span style="color: var(--oo-fg-tertiary);">
										score: {result.score.toFixed(3)}
									</span>
								</div>
								<p class="leading-relaxed" style="color: var(--oo-fg-secondary);">{result.snippet}</p>
								{#if result.timestamp}
									<p class="text-[10px] mt-1" style="color: var(--oo-fg-tertiary);">{result.timestamp}</p>
								{/if}
							</div>
						{/each}
					</div>
				{/if}
			{/if}
		</div>

	{/if}
</div>
