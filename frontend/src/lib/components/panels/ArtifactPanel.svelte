<!--
  ArtifactPanel.svelte
  Panneau lateral droit pour afficher les artifacts d'une conversation.
  Artifact list, content view, download, deletion.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { activeConversationId } from '$lib/stores/conversations';
	import { closePanel } from '$lib/stores/panels';
	import { toastError, toastSuccess } from '$lib/stores/notifications';
	import { listArtifacts, getArtifact, deleteArtifact, downloadArtifact } from '$lib/api/artifacts';
	import type { ArtifactInfo, ArtifactContent } from '$lib/types';

	let artifacts: ArtifactInfo[] = [];
	let selectedArtifact: ArtifactContent | null = null;
	let loading = false;
	let loadingContent = false;
	let deleteConfirmId: string | null = null;

	// Recharge quand la conversation change
	$: if ($activeConversationId) {
		loadArtifacts($activeConversationId);
		selectedArtifact = null;
	}

	async function loadArtifacts(convId: string) {
		loading = true;
		try {
			artifacts = await listArtifacts(convId);
		} catch {
			artifacts = [];
		} finally {
			loading = false;
		}
	}

	async function viewArtifact(id: string) {
		loadingContent = true;
		try {
			selectedArtifact = await getArtifact(id);
		} catch {
			toastError('Failed to load artifact content');
		} finally {
			loadingContent = false;
		}
	}

	async function handleDownload(id: string, title: string) {
		try {
			const blob = await downloadArtifact(id);
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = title || 'artifact';
			a.click();
			URL.revokeObjectURL(url);
		} catch {
			toastError('Failed to download artifact');
		}
	}

	async function handleDelete(id: string) {
		try {
			await deleteArtifact(id);
			artifacts = artifacts.filter((a) => a.id !== id);
			if (selectedArtifact?.id === id) selectedArtifact = null;
			deleteConfirmId = null;
			toastSuccess('Artifact deleted');
		} catch {
			toastError('Failed to delete artifact');
		}
	}

	function backToList() {
		selectedArtifact = null;
	}

	/** Detecte un langage pour le badge de couleur. */
	function langColor(lang: string): string {
		const colors: Record<string, string> = {
			python: 'text-[var(--oo-info)]',
			javascript: 'text-[var(--oo-warning)]',
			typescript: 'text-[var(--oo-info)]',
			html: 'text-[var(--oo-cat-orange)]',
			css: 'text-[var(--oo-cat-purple)]',
			markdown: 'text-surface-300',
			json: 'text-[var(--oo-success)]',
			sql: 'text-[var(--oo-error)]',
		};
		return colors[lang.toLowerCase()] || 'text-surface-400';
	}
</script>

<div class="flex flex-col h-full" style="background-color: var(--oo-panel-bg);">
	<!-- Header -->
	<div class="flex items-center justify-between px-3 py-2 shrink-0" style="border-bottom: 1px solid var(--oo-bd-subtle);">
		<div class="flex items-center gap-2">
			{#if selectedArtifact}
				<button
					on:click={backToList}
					class="p-1 rounded text-surface-400 hover:text-surface-200 hover:bg-surface-800"
					title="Back to list"
				aria-label="Back to artifact list"
				>
					<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path d="M15 19l-7-7 7-7" />
					</svg>
				</button>
			{/if}
			<h2 class="text-sm font-medium text-surface-300">
				{selectedArtifact ? selectedArtifact.title : 'Artifacts'}
			</h2>
			{#if !selectedArtifact}
				<span class="text-xs text-surface-500 font-mono">{artifacts.length}</span>
			{/if}
		</div>
		<button
			on:click={closePanel}
			class="p-1 rounded text-surface-400 hover:text-surface-200 hover:bg-surface-800"
			title="Close panel"
		aria-label="Close artifact panel"
		>
			<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
				<path d="M6 18L18 6M6 6l12 12" />
			</svg>
		</button>
	</div>

	<!-- Content -->
	<div class="flex-1 overflow-y-auto min-h-0">
		{#if loading}
			<div class="flex items-center justify-center py-12">
				<div class="w-5 h-5 border-2 border-surface-600 border-t-accent-400 rounded-full animate-spin" />
			</div>
		{:else if selectedArtifact}
			<!-- Artifact detail view -->
			<div class="p-3">
				<!-- Metadata bar -->
				<div class="flex items-center gap-2 mb-3 text-xs text-surface-500">
					<span class="{langColor(selectedArtifact.language)} font-mono">
						{selectedArtifact.language || selectedArtifact.artifact_type}
					</span>
					<span>v{selectedArtifact.version}</span>
					{#if selectedArtifact.line_count > 0}
						<span>{selectedArtifact.line_count} lines</span>
					{/if}
				</div>

				<!-- Action buttons -->
				<div class="flex gap-2 mb-3">
					<button
						on:click={() => { if (selectedArtifact) handleDownload(selectedArtifact.id, selectedArtifact.title); }}
						class="px-2 py-1 text-xs rounded bg-surface-800 text-surface-300 hover:bg-surface-700 transition-colors"
					>
						Download
					</button>
					<button
						on:click={() => navigator.clipboard.writeText(selectedArtifact?.content ?? '')}
						class="px-2 py-1 text-xs rounded bg-surface-800 text-surface-300 hover:bg-surface-700 transition-colors"
					>
						Copy
					</button>
				</div>

				<!-- Content -->
				{#if loadingContent}
					<div class="flex items-center justify-center py-8">
						<div class="w-5 h-5 border-2 border-surface-600 border-t-accent-400 rounded-full animate-spin" />
					</div>
				{:else}
					<pre class="text-xs font-mono text-surface-300 bg-surface-900 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap break-words border border-surface-800"><code>{selectedArtifact.content}</code></pre>
				{/if}
			</div>
		{:else if artifacts.length === 0}
			<div class="flex flex-col items-center justify-center py-12 px-4 text-center">
				<svg class="w-8 h-8 text-surface-600 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
					<path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
				</svg>
				<p class="text-sm text-surface-500">No artifacts yet</p>
				<p class="text-xs text-surface-600 mt-1">Artifacts appear when the LLM generates code or structured content.</p>
			</div>
		{:else}
			<!-- Artifact list -->
			<div class="py-1">
				{#each artifacts as artifact (artifact.id)}
					<div class="group flex items-center gap-2 px-3 py-2 hover:bg-surface-900/50 cursor-pointer transition-colors">
						<button
							class="flex-1 min-w-0 text-left"
							on:click={() => viewArtifact(artifact.id)}
						>
							<div class="text-sm text-surface-300 truncate">{artifact.title}</div>
							<div class="flex items-center gap-2 mt-0.5 text-xs text-surface-500">
								<span class="{langColor(artifact.language)} font-mono">
									{artifact.language || artifact.artifact_type}
								</span>
								<span>v{artifact.version}</span>
								{#if artifact.line_count > 0}
									<span>{artifact.line_count}L</span>
								{/if}
							</div>
						</button>

						<!-- Actions (visible au hover) -->
						<div class="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
							<button
								on:click|stopPropagation={() => handleDownload(artifact.id, artifact.title)}
								class="p-1 rounded text-surface-500 hover:text-surface-300 hover:bg-surface-800"
								title="Download"
							>
								<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
									<path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
								</svg>
							</button>
							{#if deleteConfirmId === artifact.id}
								<button
									on:click|stopPropagation={() => handleDelete(artifact.id)}
									class="px-1.5 py-0.5 rounded text-xs bg-[var(--oo-error)] text-[var(--oo-fg-on-semantic)] hover:bg-[var(--oo-error)]"
								>
									Confirm
								</button>
								<button
									on:click|stopPropagation={() => (deleteConfirmId = null)}
									class="px-1.5 py-0.5 rounded text-xs text-surface-400 hover:text-surface-200"
								>
									Cancel
								</button>
							{:else}
								<button
									on:click|stopPropagation={() => (deleteConfirmId = artifact.id)}
									class="p-1 rounded text-surface-500 hover:text-[var(--oo-error)] hover:bg-surface-800"
									title="Delete"
								>
									<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
										<path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
									</svg>
								</button>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		{/if}
	</div>
</div>
