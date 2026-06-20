<!--
  BatchUpload.svelte -- S120 RAG Batch File Upload.

  Drag-and-drop zone for multiple files with:
  - File type validation (PDF, TXT, MD, HTML, DOCX, CSV, XLSX, etc.)
  - Collection selector (existing or create new)
  - Queued file list with individual remove buttons
  - Upload button triggers POST /api/rag/ingest/batch
  - Emits job_id on successful submission for progress tracking
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { listCollections, ingestBatch, createCollection } from '$lib/api/rag';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type { RAGCollection, RAGIngestJob } from '$lib/types';

	const dispatch = createEventDispatcher<{
		jobStarted: RAGIngestJob;
	}>();

	// Supported file extensions matching backend SUPPORTED_EXTENSIONS
	const SUPPORTED_EXTENSIONS = new Set([
		'.pdf', '.txt', '.md', '.html', '.htm', '.docx', '.doc',
		'.csv', '.tsv', '.xlsx', '.xls',
		'.py', '.r', '.rmd',
		'.json', '.yaml', '.yml', '.toml',
		'.js', '.ts', '.css', '.sql', '.sh',
	]);

	const MAX_FILE_SIZE_MB = 50;

	// State
	let collections: RAGCollection[] = [];
	let selectedCollection = 'default';
	let newCollectionName = '';
	let creatingCollection = false;
	let queuedFiles: File[] = [];
	let dragOver = false;
	let uploading = false;

	// Load collections on mount
	loadCollections();

	async function loadCollections() {
		try {
			const resp = await listCollections();
			collections = resp.collections;
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to load collections');
		}
	}

	/** Validate a file against supported extensions and size limit. */
	function validateFile(file: File): string | null {
		const ext = '.' + file.name.split('.').pop()?.toLowerCase();
		if (!SUPPORTED_EXTENSIONS.has(ext)) {
			return `Unsupported file type: ${ext}`;
		}
		if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
			return `File too large: ${(file.size / (1024 * 1024)).toFixed(1)}MB (max ${MAX_FILE_SIZE_MB}MB)`;
		}
		return null;
	}

	/** Add files to the queue, skipping duplicates and invalid files. */
	function addFiles(fileList: FileList | File[]) {
		const files = Array.from(fileList);
		const existingNames = new Set(queuedFiles.map(f => f.name));
		let skipped = 0;

		for (const file of files) {
			if (existingNames.has(file.name)) {
				skipped++;
				continue;
			}
			const error = validateFile(file);
			if (error) {
				toastError(`${file.name}: ${error}`);
				continue;
			}
			queuedFiles = [...queuedFiles, file];
			existingNames.add(file.name);
		}

		if (skipped > 0) {
			toastError(`Skipped ${skipped} duplicate file(s)`);
		}
	}

	/** Remove a file from the queue by index. */
	function removeFile(index: number) {
		queuedFiles = queuedFiles.filter((_, i) => i !== index);
	}

	/** Clear the entire queue. */
	function clearQueue() {
		queuedFiles = [];
	}

	// Drag-and-drop handlers
	function handleDrop(e: DragEvent) {
		e.preventDefault();
		dragOver = false;
		if (e.dataTransfer?.files) {
			addFiles(e.dataTransfer.files);
		}
	}

	function handleDragOver(e: DragEvent) {
		e.preventDefault();
		dragOver = true;
	}

	function handleDragLeave() {
		dragOver = false;
	}

	/** File input change handler. */
	function handleFileInput(e: Event) {
		const input = e.currentTarget as HTMLInputElement;
		if (input.files) {
			addFiles(input.files);
			input.value = ''; // Reset so same file can be re-added
		}
	}

	/** Create a new collection inline. */
	async function handleCreateCollection() {
		const name = newCollectionName.trim();
		if (!name) return;
		creatingCollection = true;
		try {
			await createCollection({ name, description: '' });
			toastSuccess(`Collection "${name}" created`);
			await loadCollections();
			selectedCollection = name;
			newCollectionName = '';
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to create collection');
		} finally {
			creatingCollection = false;
		}
	}

	/** Start batch upload. */
	async function handleUpload() {
		if (queuedFiles.length === 0) return;
		uploading = true;
		try {
			const job = await ingestBatch(queuedFiles, selectedCollection);
			toastSuccess(`Batch job started: ${job.total_files} files`);
			dispatch('jobStarted', job);
			queuedFiles = [];
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Batch upload failed');
		} finally {
			uploading = false;
		}
	}

	/** Format file size for display. */
	function formatSize(bytes: number): string {
		if (bytes < 1024) return `${bytes}B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
	}

	/** Get file extension for display. */
	function fileExt(name: string): string {
		const parts = name.split('.');
		return parts.length > 1 ? parts.pop()!.toUpperCase() : '?';
	}

	$: totalSize = queuedFiles.reduce((sum, f) => sum + f.size, 0);
</script>

<div class="space-y-4" data-testid="batch-upload">
	<!-- Collection selector -->
	<div class="flex items-center gap-2 flex-wrap">
		<label class="text-sm" style="color: var(--oo-fg-secondary);">Collection:</label>
		<select
			bind:value={selectedCollection}
			class="px-3 py-1.5 rounded-lg text-sm"
			style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
			data-testid="batch-collection-select"
		>
			<option value="default">default</option>
			{#each collections.filter(c => c.name !== 'default') as coll}
				<option value={coll.name}>{coll.name}</option>
			{/each}
		</select>

		<span class="text-xs" style="color: var(--oo-fg-muted);">or</span>

		<input
			type="text"
			bind:value={newCollectionName}
			placeholder="New collection name"
			class="px-2 py-1.5 rounded-lg text-sm"
			style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
			on:keydown={(e) => { if (e.key === 'Enter') handleCreateCollection(); }}
			data-testid="batch-new-collection-input"
		/>
		<button
			on:click={handleCreateCollection}
			disabled={creatingCollection || !newCollectionName.trim()}
			class="px-3 py-1.5 rounded-lg text-xs font-medium disabled:opacity-50"
			style="background-color: var(--oo-bg-overlay); color: var(--oo-fg-secondary); border: 1px solid var(--oo-bd-default);"
			data-testid="batch-create-collection-btn"
		>
			{creatingCollection ? 'Creating...' : 'Create'}
		</button>
	</div>

	<!-- Drag-and-drop zone -->
	<div
		role="button"
		tabindex="0"
		on:drop={handleDrop}
		on:dragover={handleDragOver}
		on:dragleave={handleDragLeave}
		on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') document.getElementById('batch-file-input')?.click(); }}
		class="rounded-lg p-8 text-center cursor-pointer transition-colors"
		style="border: 2px dashed {dragOver ? 'var(--oo-acc-600)' : 'var(--oo-bd-default)'};
			background-color: {dragOver ? 'var(--oo-bg-overlay)' : 'var(--oo-bg-elevated)'};"
		data-testid="batch-drop-zone"
	>
		<div class="text-2xl mb-2" style="color: var(--oo-fg-muted);">&#128193;</div>
		<p class="text-sm" style="color: var(--oo-fg-secondary);">
			Drag and drop files here or
			<label class="underline cursor-pointer" style="color: var(--oo-acc-600);">
				browse
				<input
					id="batch-file-input"
					type="file"
					multiple
					accept=".pdf,.txt,.md,.html,.htm,.docx,.doc,.csv,.tsv,.xlsx,.xls,.py,.r,.rmd,.json,.yaml,.yml,.toml,.js,.ts,.css,.sql,.sh"
					class="hidden"
					on:change={handleFileInput}
					data-testid="batch-file-input"
				/>
			</label>
		</p>
		<p class="text-xs mt-2" style="color: var(--oo-fg-muted);">
			PDF, DOCX, XLSX, CSV, TXT, MD, HTML, code files &mdash; max {MAX_FILE_SIZE_MB}MB per file
		</p>
	</div>

	<!-- Queued file list -->
	{#if queuedFiles.length > 0}
		<div class="rounded-lg overflow-hidden" style="border: 1px solid var(--oo-bd-subtle);" data-testid="batch-file-queue">
			<!-- Header -->
			<div class="flex items-center justify-between px-3 py-2"
				style="background-color: var(--oo-bg-overlay);">
				<span class="text-xs font-medium" style="color: var(--oo-fg-secondary);">
					{queuedFiles.length} file{queuedFiles.length > 1 ? 's' : ''} queued ({formatSize(totalSize)})
				</span>
				<button
					on:click={clearQueue}
					class="text-xs px-2 py-0.5 rounded"
					style="color: var(--oo-fg-muted);"
					data-testid="batch-clear-queue-btn"
				>
					Clear all
				</button>
			</div>

			<!-- File rows -->
			<div class="divide-y" style="border-color: var(--oo-bd-subtle);">
				{#each queuedFiles as file, i}
					<div class="flex items-center justify-between px-3 py-2"
						style="background-color: var(--oo-bg-elevated);">
						<div class="flex items-center gap-2 min-w-0 flex-1">
							<span class="text-xs font-mono px-1.5 py-0.5 rounded shrink-0"
								style="background-color: var(--oo-bg-overlay); color: var(--oo-fg-muted);">
								{fileExt(file.name)}
							</span>
							<span class="text-sm truncate" style="color: var(--oo-fg-primary);"
								data-testid="batch-file-name">{file.name}</span>
							<span class="text-xs shrink-0" style="color: var(--oo-fg-muted);">{formatSize(file.size)}</span>
						</div>
						<button
							on:click={() => removeFile(i)}
							class="text-xs px-2 py-0.5 rounded shrink-0 ml-2"
							style="color: var(--oo-error);"
							data-testid="batch-remove-file-btn"
							title="Remove from queue"
						>
							&#10005;
						</button>
					</div>
				{/each}
			</div>
		</div>

		<!-- Upload button -->
		<div class="flex justify-end">
			<button
				on:click={handleUpload}
				disabled={uploading || queuedFiles.length === 0}
				class="px-5 py-2.5 rounded-lg text-sm font-medium disabled:opacity-50 transition-opacity"
				style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
				data-testid="batch-upload-btn"
			>
				{#if uploading}
					Uploading {queuedFiles.length} file{queuedFiles.length > 1 ? 's' : ''}...
				{:else}
					Upload {queuedFiles.length} file{queuedFiles.length > 1 ? 's' : ''} to "{selectedCollection}"
				{/if}
			</button>
		</div>
	{/if}
</div>
