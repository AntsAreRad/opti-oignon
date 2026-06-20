<!--
  DocumentManager.svelte -- S120 RAG Document Manager Panel.

  Provides search/filter/browse for ingested documents:
  - Search bar filtering by filename (server-side via search query param)
  - File type filter dropdown (server-side via file_type query param)
  - Collection filter dropdown
  - Paginated document table: filename, type, chunks, size, date
  - Delete button per document with confirmation
  - Bulk delete with checkbox selection
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { listDocuments, listCollections, deleteDocument } from '$lib/api/rag';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type { RAGCollection, RAGDocument } from '$lib/types';

	/** Externally trigger a refresh (e.g. after batch ingest completes). */
	export function refresh() {
		loadDocuments();
	}

	// Filter state
	let collections: RAGCollection[] = [];
	let filterCollection = '';
	let filterSearch = '';
	let filterFileType = '';
	let searchDebounceTimer: ReturnType<typeof setTimeout> | null = null;

	// Pagination state
	const PAGE_SIZE = 20;
	let currentPage = 0;
	let totalDocs = 0;

	// Documents
	let documents: RAGDocument[] = [];
	let loading = true;

	// Bulk selection
	let selectedIds: Set<string> = new Set();
	let bulkDeleting = false;

	// Known file types for filter dropdown
	const FILE_TYPES = [
		'', 'pdf', 'docx', 'xlsx', 'csv', 'txt', 'markdown', 'html', 'code', 'json', 'yaml',
	];

	const FILE_TYPE_LABELS: Record<string, string> = {
		'': 'All types',
		'pdf': 'PDF',
		'docx': 'Word (DOCX)',
		'xlsx': 'Excel (XLSX)',
		'csv': 'CSV/TSV',
		'txt': 'Plain text',
		'markdown': 'Markdown',
		'html': 'HTML',
		'code': 'Code',
		'json': 'JSON',
		'yaml': 'YAML',
	};

	// Computed
	$: totalPages = Math.max(1, Math.ceil(totalDocs / PAGE_SIZE));
	$: allSelected = documents.length > 0 && documents.every(d => selectedIds.has(d.doc_id));

	async function loadCollections() {
		try {
			const resp = await listCollections();
			collections = resp.collections;
		} catch { /* ignore */ }
	}

	async function loadDocuments() {
		loading = true;
		selectedIds = new Set();
		try {
			const resp = await listDocuments({
				collection: filterCollection || undefined,
				search: filterSearch || undefined,
				file_type: filterFileType || undefined,
				limit: PAGE_SIZE,
				offset: currentPage * PAGE_SIZE,
			});
			documents = resp.documents;
			totalDocs = resp.total;
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to load documents');
		} finally {
			loading = false;
		}
	}

	/** Debounced search input handler. */
	function handleSearchInput() {
		if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
		searchDebounceTimer = setTimeout(() => {
			currentPage = 0;
			loadDocuments();
		}, 400);
	}

	function handleFilterChange() {
		currentPage = 0;
		loadDocuments();
	}

	function goToPage(page: number) {
		if (page < 0 || page >= totalPages) return;
		currentPage = page;
		loadDocuments();
	}

	// Selection helpers
	function toggleSelect(docId: string) {
		const next = new Set(selectedIds);
		if (next.has(docId)) {
			next.delete(docId);
		} else {
			next.add(docId);
		}
		selectedIds = next;
	}

	function toggleSelectAll() {
		if (allSelected) {
			selectedIds = new Set();
		} else {
			selectedIds = new Set(documents.map(d => d.doc_id));
		}
	}

	/** Delete a single document with confirmation. */
	async function handleDelete(doc: RAGDocument) {
		const name = shortName(doc.source_file);
		if (!confirm(`Delete document "${name}" and its ${doc.chunk_count} chunks?`)) return;
		try {
			await deleteDocument(doc.doc_id);
			toastSuccess(`Deleted "${name}"`);
			await loadDocuments();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to delete document');
		}
	}

	/** Bulk delete selected documents. */
	async function handleBulkDelete() {
		const count = selectedIds.size;
		if (count === 0) return;
		if (!confirm(`Delete ${count} selected document(s) and all their chunks?`)) return;

		bulkDeleting = true;
		let success = 0;
		let failed = 0;

		for (const docId of selectedIds) {
			try {
				await deleteDocument(docId);
				success++;
			} catch {
				failed++;
			}
		}

		if (success > 0) toastSuccess(`Deleted ${success} document(s)`);
		if (failed > 0) toastError(`Failed to delete ${failed} document(s)`);

		bulkDeleting = false;
		selectedIds = new Set();
		await loadDocuments();
		await loadCollections();
	}

	function shortName(path: string): string {
		const parts = path.split('/');
		return parts[parts.length - 1] || path;
	}

	function formatTime(ts: number): string {
		if (!ts) return '-';
		return new Date(ts * 1000).toLocaleDateString(undefined, {
			year: 'numeric', month: 'short', day: 'numeric',
		});
	}

	function formatSize(bytes: number): string {
		if (bytes < 1024) return `${bytes}B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
	}

	onMount(() => {
		loadCollections();
		loadDocuments();
	});
</script>

<div class="space-y-4" data-testid="document-manager">
	<!-- Filters row -->
	<div class="flex items-center gap-2 flex-wrap">
		<!-- Search -->
		<div class="flex-1 min-w-48">
			<input
				type="text"
				bind:value={filterSearch}
				on:input={handleSearchInput}
				placeholder="Search by filename..."
				class="w-full px-3 py-1.5 rounded-lg text-sm"
				style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
				data-testid="doc-search-input"
			/>
		</div>

		<!-- File type filter -->
		<select
			bind:value={filterFileType}
			on:change={handleFilterChange}
			class="px-3 py-1.5 rounded-lg text-sm"
			style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
			data-testid="doc-filetype-filter"
		>
			{#each FILE_TYPES as ft}
				<option value={ft}>{FILE_TYPE_LABELS[ft] || ft}</option>
			{/each}
		</select>

		<!-- Collection filter -->
		<select
			bind:value={filterCollection}
			on:change={handleFilterChange}
			class="px-3 py-1.5 rounded-lg text-sm"
			style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
			data-testid="doc-collection-filter"
		>
			<option value="">All collections</option>
			{#each collections as coll}
				<option value={coll.name}>{coll.name}</option>
			{/each}
		</select>

		<!-- Bulk delete button -->
		{#if selectedIds.size > 0}
			<button
				on:click={handleBulkDelete}
				disabled={bulkDeleting}
				class="px-3 py-1.5 rounded-lg text-xs font-medium disabled:opacity-50"
				style="color: var(--oo-error); border: 1px solid var(--oo-error-bd);"
				data-testid="doc-bulk-delete-btn"
			>
				{bulkDeleting ? 'Deleting...' : `Delete ${selectedIds.size} selected`}
			</button>
		{/if}
	</div>

	<!-- Results count -->
	<div class="flex items-center justify-between">
		<span class="text-xs" style="color: var(--oo-fg-muted);" data-testid="doc-total-count">
			{totalDocs} document{totalDocs !== 1 ? 's' : ''}{filterSearch ? ` matching "${filterSearch}"` : ''}
		</span>
		{#if totalPages > 1}
			<span class="text-xs" style="color: var(--oo-fg-muted);">
				Page {currentPage + 1} of {totalPages}
			</span>
		{/if}
	</div>

	<!-- Documents table -->
	{#if loading}
		<p class="text-sm py-4 text-center" style="color: var(--oo-fg-muted);">Loading documents...</p>
	{:else if documents.length === 0}
		<p class="text-sm py-4 text-center" style="color: var(--oo-fg-muted);">
			{filterSearch || filterFileType || filterCollection
				? 'No documents match the current filters.'
				: 'No documents ingested yet.'}
		</p>
	{:else}
		<div class="rounded-lg overflow-hidden" style="border: 1px solid var(--oo-bd-subtle);"
			data-testid="doc-table">
			<!-- Table header -->
			<div class="grid items-center gap-2 px-3 py-2 text-xs font-medium"
				style="grid-template-columns: 2rem 1fr 5rem 4rem 5rem 5rem 3rem;
					background-color: var(--oo-bg-overlay); color: var(--oo-fg-muted);">
				<div>
					<input
						type="checkbox"
						checked={allSelected}
						on:change={toggleSelectAll}
						class="cursor-pointer"
						data-testid="doc-select-all"
					/>
				</div>
				<div>Filename</div>
				<div>Type</div>
				<div>Chunks</div>
				<div>Size</div>
				<div>Date</div>
				<div></div>
			</div>

			<!-- Table rows -->
			<div class="divide-y" style="border-color: var(--oo-bd-subtle);">
				{#each documents as doc}
					<div class="grid items-center gap-2 px-3 py-2"
						style="grid-template-columns: 2rem 1fr 5rem 4rem 5rem 5rem 3rem;
							background-color: var(--oo-bg-elevated);"
						data-testid="doc-row">
						<div>
							<input
								type="checkbox"
								checked={selectedIds.has(doc.doc_id)}
								on:change={() => toggleSelect(doc.doc_id)}
								class="cursor-pointer"
								data-testid="doc-row-checkbox"
							/>
						</div>
						<div class="text-sm truncate" style="color: var(--oo-fg-primary);"
							title={doc.source_file}
							data-testid="doc-row-name">
							{shortName(doc.source_file)}
						</div>
						<div>
							<span class="text-xs font-mono px-1.5 py-0.5 rounded"
								style="background-color: var(--oo-bg-overlay); color: var(--oo-fg-muted);">
								{doc.file_type}
							</span>
						</div>
						<div class="text-xs" style="color: var(--oo-fg-muted);">
							{doc.chunk_count}
						</div>
						<div class="text-xs" style="color: var(--oo-fg-muted);">
							{formatSize(doc.raw_text_length)}
						</div>
						<div class="text-xs" style="color: var(--oo-fg-muted);">
							{formatTime(doc.ingested_at)}
						</div>
						<div>
							<button
								on:click={() => handleDelete(doc)}
								class="text-xs px-1.5 py-0.5 rounded"
								style="color: var(--oo-error);"
								title="Delete document"
								data-testid="doc-row-delete-btn"
							>
								&#10005;
							</button>
						</div>
					</div>
				{/each}
			</div>
		</div>

		<!-- Pagination -->
		{#if totalPages > 1}
			<div class="flex items-center justify-center gap-1" data-testid="doc-pagination">
				<button
					on:click={() => goToPage(0)}
					disabled={currentPage === 0}
					class="px-2 py-1 rounded text-xs disabled:opacity-30"
					style="color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-subtle);"
				>
					&#171;
				</button>
				<button
					on:click={() => goToPage(currentPage - 1)}
					disabled={currentPage === 0}
					class="px-2 py-1 rounded text-xs disabled:opacity-30"
					style="color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-subtle);"
				>
					&#8249;
				</button>

				{#each Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
					const start = Math.max(0, Math.min(currentPage - 3, totalPages - 7));
					return start + i;
				}).filter(p => p < totalPages) as page}
					<button
						on:click={() => goToPage(page)}
						class="px-2.5 py-1 rounded text-xs font-medium"
						style="color: {page === currentPage ? 'var(--oo-acc-50)' : 'var(--oo-fg-secondary)'};
							background-color: {page === currentPage ? 'var(--oo-acc-600)' : 'transparent'};
							border: 1px solid {page === currentPage ? 'var(--oo-acc-600)' : 'var(--oo-bd-subtle)'};"
					>
						{page + 1}
					</button>
				{/each}

				<button
					on:click={() => goToPage(currentPage + 1)}
					disabled={currentPage >= totalPages - 1}
					class="px-2 py-1 rounded text-xs disabled:opacity-30"
					style="color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-subtle);"
				>
					&#8250;
				</button>
				<button
					on:click={() => goToPage(totalPages - 1)}
					disabled={currentPage >= totalPages - 1}
					class="px-2 py-1 rounded text-xs disabled:opacity-30"
					style="color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-subtle);"
				>
					&#187;
				</button>
			</div>
		{/if}
	{/if}
</div>
