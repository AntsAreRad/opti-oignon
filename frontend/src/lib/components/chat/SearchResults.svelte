<!--
  SearchResults.svelte
  Display inline web search results in a chat message.
  Numbered citations, source links with favicon, expandable snippets.
-->
<script lang="ts">
	import type { SearchResult } from '$lib/types';
	import { Card } from '$lib/ds';

	export let results: SearchResult[] = [];
	export let query: string = '';
	export let engine: string = '';
	export let citations: string[] = [];

	let expandedIndex: number | null = null;

	function toggleExpand(index: number) {
		expandedIndex = expandedIndex === index ? null : index;
	}

	function getDomain(url: string): string {
		try {
			return new URL(url).hostname.replace(/^www\./, '');
		} catch {
			return url;
		}
	}

	function hideFavicon(e: Event) {
		const el = e.currentTarget;
		if (el instanceof HTMLElement) el.style.display = 'none';
	}

	function getFaviconUrl(url: string): string {
		try {
			const domain = new URL(url).origin;
			return `${domain}/favicon.ico`;
		} catch {
			return '';
		}
	}
</script>

{#if results.length > 0}
	<Card variant="flat" padding="sm" class="mt-3">
		<!-- Header: search performed -->
		<div class="flex items-center gap-2 mb-2">
			<svg class="w-3.5 h-3.5 text-accent-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<circle cx="11" cy="11" r="8" />
				<path d="M21 21l-4.35-4.35" />
			</svg>
			<span class="text-xs text-surface-400">
				{results.length} result{results.length !== 1 ? 's' : ''} for "<span class="text-surface-300">{query}</span>"
			</span>
			{#if engine}
				<span class="text-xs text-surface-600">via {engine}</span>
			{/if}
		</div>

		<!-- Results -->
		<div class="space-y-1.5">
			{#each results as result, i}
				<div class="group rounded-lg bg-surface-900/50 border border-surface-700/30 hover:border-surface-600/50 transition-colors">
					<!-- Primary row -->
					<button
						on:click={() => toggleExpand(i)}
						aria-expanded={expandedIndex === i}
						class="w-full text-left px-3 py-2 flex items-start gap-2"
					>
						<!-- Numbered citation badge -->
						<span class="inline-flex items-center justify-center w-5 h-5 rounded-full bg-accent-600/20 text-accent-400 text-xs font-mono shrink-0 mt-0.5">
							{i + 1}
						</span>

						<div class="flex-1 min-w-0">
							<!-- Title -->
							<div class="text-xs font-medium text-surface-200 truncate">
								{result.title}
							</div>

							<!-- Source with favicon -->
							<div class="flex items-center gap-1.5 mt-0.5">
								{#if getFaviconUrl(result.url)}
									<img
										src={getFaviconUrl(result.url)}
										alt=""
										class="w-3 h-3 rounded-sm"
										on:error={hideFavicon}
									/>
								{/if}
								<span class="text-xs text-surface-500 truncate">{getDomain(result.url)}</span>
								{#if result.relevance_score > 0}
									<span class="text-xs text-surface-600 font-mono">
										{Math.round(result.relevance_score * 100)}%
									</span>
								{/if}
							</div>
						</div>

						<!-- Expand chevron -->
						<svg
							class="w-3.5 h-3.5 text-surface-500 shrink-0 mt-0.5 transition-transform"
							class:rotate-180={expandedIndex === i}
							fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"
						>
							<path d="M19 9l-7 7-7-7" />
						</svg>
					</button>

					<!-- Expandable snippet -->
					{#if expandedIndex === i}
						<div class="px-3 pb-2 pl-10">
							<p class="text-xs text-surface-400 leading-relaxed">{result.snippet}</p>
							<a
								href={result.url}
								target="_blank"
								rel="noopener noreferrer"
								class="inline-flex items-center gap-1 text-xs text-accent-400 hover:text-accent-300 mt-1.5"
							>
								Open source
								<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
									<path d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
								</svg>
							</a>
						</div>
					{/if}
				</div>
			{/each}
		</div>

		<!-- Inline citations -->
		{#if citations.length > 0}
			<div class="mt-2 flex flex-wrap gap-1">
				{#each citations as cite, i}
					<span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-surface-800 text-xs text-surface-400">
						<span class="text-accent-400 font-mono">[{i + 1}]</span>
						<span class="truncate max-w-[200px]">{cite}</span>
					</span>
				{/each}
			</div>
		{/if}
	</Card>
{/if}
