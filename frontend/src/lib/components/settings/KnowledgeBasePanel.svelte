<!--
  KnowledgeBasePanel.svelte -- S99 RAG v2 Knowledge Base Management.
  Updated S120: Batch upload, document manager, folder scan integration.

  Sub-sections:
  1. Collections: create/delete, stats overview
  2. Batch Upload: drag-and-drop multi-file with progress tracking
  3. Documents: search/filter/paginated manager with bulk delete
  4. Folder Scan: local directory ingestion
  5. Query: test interface with citation display
  6. Dashboard: RAG stats
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		listCollections,
		createCollection,
		deleteCollection,
		ingestURL,
		queryKnowledgeBase,
	} from '$lib/api/rag';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import RAGDashboardPanel from './RAGDashboardPanel.svelte';
	import BatchUpload from '$lib/components/rag/BatchUpload.svelte';
	import IngestProgress from '$lib/components/rag/IngestProgress.svelte';
	import DocumentManager from '$lib/components/rag/DocumentManager.svelte';
	import FolderScan from '$lib/components/rag/FolderScan.svelte';
	import type {
		RAGCollection,
		RAGQueryResponse,
		RAGIngestJob,
	} from '$lib/types';

	type SubTab = 'collections' | 'batch-upload' | 'documents' | 'folder-scan' | 'query' | 'dashboard';
	let activeSubTab: SubTab = 'collections';

	// -- Collections state --
	let collections: RAGCollection[] = [];
	let collectionsLoading = true;
	let newCollName = '';
	let newCollDesc = '';
	let creating = false;

	// -- Batch upload job tracking (S120) --
	let activeJobs: RAGIngestJob[] = [];

	// -- Document manager ref (S120) --
	let docManagerRef: DocumentManager;

	// -- URL ingestion (kept from S99, shown in batch-upload tab) --
	let urlInput = '';
	let urlIngesting = false;
	let selectedCollection = 'default';

	// -- Query state --
	let queryText = '';
	let queryCollection = 'default';
	let querying = false;
	let queryResponse: RAGQueryResponse | null = null;

	// ---------------------------------------------------------------
	// COLLECTIONS
	// ---------------------------------------------------------------

	async function loadCollections() {
		collectionsLoading = true;
		try {
			const resp = await listCollections();
			collections = resp.collections;
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to load collections');
		} finally {
			collectionsLoading = false;
		}
	}

	async function handleCreate() {
		if (!newCollName.trim()) return;
		creating = true;
		try {
			await createCollection({ name: newCollName.trim(), description: newCollDesc.trim() });
			toastSuccess(`Collection "${newCollName}" created`);
			newCollName = '';
			newCollDesc = '';
			await loadCollections();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to create collection');
		} finally {
			creating = false;
		}
	}

	async function handleDelete(name: string) {
		if (!confirm(`Delete collection "${name}" and all its documents?`)) return;
		try {
			await deleteCollection(name);
			toastSuccess(`Collection "${name}" deleted`);
			await loadCollections();
			if (docManagerRef) docManagerRef.refresh();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to delete collection');
		}
	}

	// ---------------------------------------------------------------
	// DOCUMENTS (S120: now handled by DocumentManager component)
	// ---------------------------------------------------------------

	// ---------------------------------------------------------------
	// BATCH UPLOAD JOB TRACKING (S120)
	// ---------------------------------------------------------------

	function handleBatchJobStarted(event: CustomEvent<RAGIngestJob>) {
		activeJobs = [...activeJobs, event.detail];
	}

	function handleJobCompleted(job: RAGIngestJob) {
		// Refresh document manager and collections after job finishes
		if (docManagerRef) docManagerRef.refresh();
		loadCollections();
	}

	function removeJob(jobId: string) {
		activeJobs = activeJobs.filter(j => j.job_id !== jobId);
	}

	// ---------------------------------------------------------------
	// URL INGESTION (kept from S99, available in batch-upload tab)
	// ---------------------------------------------------------------

	async function handleURLIngest() {
		if (!urlInput.trim()) return;
		urlIngesting = true;
		try {
			const result = await ingestURL({ url: urlInput.trim(), collection: selectedCollection });
			toastSuccess(`Ingested URL: ${result.chunk_count} chunks`);
			urlInput = '';
			if (docManagerRef) docManagerRef.refresh();
			await loadCollections();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'URL ingestion failed');
		} finally {
			urlIngesting = false;
		}
	}

	// ---------------------------------------------------------------
	// QUERY
	// ---------------------------------------------------------------

	async function handleQuery() {
		if (!queryText.trim()) return;
		querying = true;
		queryResponse = null;
		try {
			queryResponse = await queryKnowledgeBase({
				query: queryText.trim(),
				collection: queryCollection,
				n_results: 5,
			});
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Query failed');
		} finally {
			querying = false;
		}
	}

	// ---------------------------------------------------------------
	// HELPERS
	// ---------------------------------------------------------------

	function shortName(path: string): string {
		const parts = path.split('/');
		return parts[parts.length - 1] || path;
	}

	function scoreColor(score: number): string {
		if (score >= 0.8) return 'var(--oo-success)';
		if (score >= 0.5) return 'var(--oo-warning)';
		return 'var(--oo-fg-muted)';
	}

	onMount(() => {
		loadCollections();
	});
</script>

<!-- Sub-tab navigation -->
<div class="flex gap-2 mb-4" style="border-bottom: 1px solid var(--oo-bd-subtle);">
	{#each [
		{ id: 'collections', label: 'Collections' },
		{ id: 'batch-upload', label: 'Batch Upload' },
		{ id: 'documents', label: 'Documents' },
		{ id: 'folder-scan', label: 'Folder Scan' },
		{ id: 'query', label: 'Query Test' },
		{ id: 'dashboard', label: 'Dashboard' },
	] as tab}
		<button
			on:click={() => { activeSubTab = tab.id; }}
			class="px-3 py-2 text-sm font-medium transition-colors"
			style="color: {activeSubTab === tab.id ? 'var(--oo-acc-600)' : 'var(--oo-fg-muted)'};
				border-bottom: 2px solid {activeSubTab === tab.id ? 'var(--oo-acc-600)' : 'transparent'};"
		>
			{tab.label}
		</button>
	{/each}
</div>

<!-- ==================== COLLECTIONS ==================== -->
{#if activeSubTab === 'collections'}
	<div class="space-y-4">
		<!-- Create form -->
		<div class="rounded-lg p-4" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);">
			<h3 class="text-sm font-medium mb-3" style="color: var(--oo-fg-primary);">Create Collection</h3>
			<div class="flex flex-col gap-2">
				<input
					type="text"
					bind:value={newCollName}
					placeholder="Collection name"
					class="px-3 py-2 rounded-lg text-sm"
					style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
				/>
				<input
					type="text"
					bind:value={newCollDesc}
					placeholder="Description (optional)"
					class="px-3 py-2 rounded-lg text-sm"
					style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
				/>
				<button
					on:click={handleCreate}
					disabled={creating || !newCollName.trim()}
					class="self-start px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
					style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
				>
					{creating ? 'Creating...' : 'Create'}
				</button>
			</div>
		</div>

		<!-- List -->
		{#if collectionsLoading}
			<p class="text-sm" style="color: var(--oo-fg-muted);">Loading collections...</p>
		{:else if collections.length === 0}
			<p class="text-sm" style="color: var(--oo-fg-muted);">No collections yet. Create one above.</p>
		{:else}
			<div class="space-y-2">
				{#each collections as coll}
					<div class="rounded-lg p-3 flex items-center justify-between"
						style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);">
						<div>
							<span class="text-sm font-medium" style="color: var(--oo-fg-primary);">{coll.name}</span>
							{#if coll.description}
								<span class="text-xs ml-2" style="color: var(--oo-fg-muted);">{coll.description}</span>
							{/if}
							<div class="text-xs mt-1" style="color: var(--oo-fg-muted);">
								{coll.document_count} docs / {coll.chunk_count} chunks
							</div>
						</div>
						<button
							on:click={() => handleDelete(coll.name)}
							class="px-2 py-1 rounded text-xs"
							style="color: var(--oo-error); border: 1px solid var(--oo-error-bd);"
						>
							Delete
						</button>
					</div>
				{/each}
			</div>
		{/if}
	</div>
{/if}

<!-- ==================== BATCH UPLOAD (S120) ==================== -->
{#if activeSubTab === 'batch-upload'}
	<div class="space-y-6">
		<!-- Batch file upload component -->
		<BatchUpload on:jobStarted={handleBatchJobStarted} />

		<!-- URL ingestion (kept from S99) -->
		<div class="rounded-lg p-4" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);">
			<h4 class="text-sm font-medium mb-3" style="color: var(--oo-fg-secondary);">Ingest from URL</h4>
			<div class="flex gap-2">
				<input
					type="url"
					bind:value={urlInput}
					placeholder="https://example.com/article"
					class="flex-1 px-3 py-2 rounded-lg text-sm"
					style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
					on:keydown={(e) => { if (e.key === 'Enter') handleURLIngest(); }}
				/>
				<button
					on:click={handleURLIngest}
					disabled={urlIngesting || !urlInput.trim()}
					class="px-4 py-2 rounded-lg text-sm font-medium shrink-0 disabled:opacity-50"
					style="background-color: var(--oo-bg-overlay); color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-default);"
				>
					{urlIngesting ? 'Ingesting...' : 'Ingest URL'}
				</button>
			</div>
		</div>

		<!-- Active job progress trackers -->
		{#if activeJobs.length > 0}
			<div class="space-y-3">
				<h4 class="text-sm font-medium" style="color: var(--oo-fg-secondary);">Active Jobs</h4>
				{#each activeJobs as job (job.job_id)}
					<div class="relative">
						<IngestProgress
							jobId={job.job_id}
							initialJob={job}
							on:completed={(e) => handleJobCompleted(e.detail)}
							on:cancelled={() => removeJob(job.job_id)}
						/>
						{#if job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled'}
							<button
								on:click={() => removeJob(job.job_id)}
								class="absolute top-2 right-12 text-xs px-2 py-0.5 rounded"
								style="color: var(--oo-fg-muted);"
								title="Dismiss"
							>
								&#10005;
							</button>
						{/if}
					</div>
				{/each}
			</div>
		{/if}
	</div>
{/if}

<!-- ==================== DOCUMENTS (S120) ==================== -->
{#if activeSubTab === 'documents'}
	<DocumentManager bind:this={docManagerRef} />
{/if}

<!-- ==================== FOLDER SCAN (S120) ==================== -->
{#if activeSubTab === 'folder-scan'}
	<FolderScan on:jobStarted={handleBatchJobStarted} />
{/if}

<!-- ==================== QUERY TEST ==================== -->
{#if activeSubTab === 'query'}
	<div class="space-y-4">
		<div class="flex gap-2">
			<select
				bind:value={queryCollection}
				class="px-3 py-2 rounded-lg text-sm shrink-0"
				style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
			>
				<option value="default">default</option>
				{#each collections.filter(c => c.name !== 'default') as coll}
					<option value={coll.name}>{coll.name}</option>
				{/each}
			</select>
			<input
				type="text"
				bind:value={queryText}
				placeholder="Search your knowledge base..."
				class="flex-1 px-3 py-2 rounded-lg text-sm"
				style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
				on:keydown={(e) => { if (e.key === 'Enter') handleQuery(); }}
			/>
			<button
				on:click={handleQuery}
				disabled={querying || !queryText.trim()}
				class="px-4 py-2 rounded-lg text-sm font-medium shrink-0 disabled:opacity-50"
				style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
			>
				{querying ? 'Searching...' : 'Search'}
			</button>
		</div>

		<!-- Results -->
		{#if queryResponse}
			<div class="text-xs mb-2" style="color: var(--oo-fg-muted);">
				{queryResponse.total_results} results for "{queryResponse.query}"
			</div>

			{#if queryResponse.results.length === 0}
				<p class="text-sm" style="color: var(--oo-fg-muted);">No results found. Try a different query or ingest more documents.</p>
			{:else}
				<div class="space-y-3">
					{#each queryResponse.results as result, i}
						<div class="rounded-lg p-3" style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);">
							<!-- Header -->
							<div class="flex items-center justify-between mb-2">
								<div class="flex items-center gap-2">
									<span class="text-xs font-mono px-1.5 py-0.5 rounded"
										style="background-color: var(--oo-bg-overlay); color: var(--oo-fg-muted);">
										#{i + 1}
									</span>
									<span class="text-xs font-medium" style="color: var(--oo-fg-secondary);">
										{shortName(result.source_file)}
									</span>
									{#if result.section}
										<span class="text-xs" style="color: var(--oo-fg-muted);">{result.section}</span>
									{/if}
									{#if result.page}
										<span class="text-xs" style="color: var(--oo-fg-muted);">p.{result.page}</span>
									{/if}
								</div>
								<span class="text-xs font-mono font-medium" style="color: {scoreColor(result.score)};">
									{(result.score * 100).toFixed(1)}%
								</span>
							</div>
							<!-- Content preview -->
							<pre class="text-xs whitespace-pre-wrap leading-relaxed max-h-32 overflow-y-auto"
								style="color: var(--oo-fg-primary); font-family: inherit;"
							>{result.content.slice(0, 500)}{result.content.length > 500 ? '...' : ''}</pre>
						</div>
					{/each}
				</div>
			{/if}

			<!-- Citations -->
			{#if queryResponse.citations.length > 0}
				<div class="mt-4">
					<h4 class="text-xs font-medium mb-2" style="color: var(--oo-fg-muted);">Citations</h4>
					<div class="space-y-1">
						{#each queryResponse.citations as cit}
							<div class="text-xs flex gap-2" style="color: var(--oo-fg-muted);">
								<span class="font-mono">[{cit.chunk_id.slice(0, 8)}]</span>
								<span>{shortName(cit.source_file)}</span>
								{#if cit.section}
									<span>({cit.section})</span>
								{/if}
								<span style="color: {scoreColor(cit.score)};">{(cit.score * 100).toFixed(1)}%</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		{/if}
	</div>
{/if}

<!-- ==================== DASHBOARD ==================== -->
{#if activeSubTab === 'dashboard'}
	<RAGDashboardPanel />
{/if}
