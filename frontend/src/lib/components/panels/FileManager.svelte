<!--
  FileManager.svelte
  Manages project files: upload (drag-and-drop + file picker),
  list with type icon/size/indexed status/chunk count,
  index/reindex individual files, view summary + key terms, delete.
-->
<script lang="ts">
	import {
		uploadFile,
		deleteFile,
		indexFile,
		getFileSummary,
		formatFileSize,
	} from '$lib/stores/projects';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import { Modal } from '$lib/ds';
	import type { ProjectFileInfo, ProjectFileSummary } from '$lib/types';

	export let projectId: string;
	export let files: ProjectFileInfo[] = [];

	// Upload state
	let dragging = false;
	let uploading = false;
	let uploadProgress = '';
	let fileInput: HTMLInputElement;

	// Per-file action state
	let indexingId: string | null = null;
	let deletingId: string | null = null;
	let deleteConfirmId: string | null = null;

	// Summary modal state
	let summaryData: ProjectFileSummary | null = null;
	let summaryLoading = false;
	let showSummary = false;

	// File type icon mapping
	function fileTypeIcon(filename: string): string {
		const ext = filename.split('.').pop()?.toLowerCase() ?? '';
		const codeExts = ['py', 'js', 'ts', 'jsx', 'tsx', 'r', 'sh', 'c', 'cpp', 'h', 'java', 'go', 'rs', 'lua', 'rb', 'pl'];
		const dataExts = ['json', 'yaml', 'yml', 'csv', 'tsv', 'xml', 'toml', 'ini', 'cfg'];
		const docExts = ['md', 'txt', 'tex', 'bib', 'log', 'html', 'css'];
		if (codeExts.includes(ext)) return 'code';
		if (dataExts.includes(ext)) return 'data';
		if (docExts.includes(ext)) return 'doc';
		return 'file';
	}

	// Handle drag events
	function handleDragEnter(e: DragEvent) {
		e.preventDefault();
		dragging = true;
	}

	function handleDragOver(e: DragEvent) {
		e.preventDefault();
		dragging = true;
	}

	function handleDragLeave(e: DragEvent) {
		e.preventDefault();
		// Only reset if leaving the drop zone itself
		const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
		const x = e.clientX;
		const y = e.clientY;
		if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
			dragging = false;
		}
	}

	async function handleDrop(e: DragEvent) {
		e.preventDefault();
		dragging = false;
		const droppedFiles = e.dataTransfer?.files;
		if (droppedFiles && droppedFiles.length > 0) {
			await uploadFiles(Array.from(droppedFiles));
		}
	}

	function handleFileSelect() {
		const selected = fileInput?.files;
		if (selected && selected.length > 0) {
			uploadFiles(Array.from(selected));
		}
	}

	async function uploadFiles(fileList: File[]) {
		uploading = true;
		let successCount = 0;
		let failCount = 0;

		for (let i = 0; i < fileList.length; i++) {
			uploadProgress = `Uploading ${i + 1}/${fileList.length}: ${fileList[i].name}`;
			try {
				await uploadFile(projectId, fileList[i]);
				successCount++;
			} catch (e) {
				failCount++;
				toastError(`Failed to upload ${fileList[i].name}: ${e instanceof Error ? e.message : 'Unknown error'}`);
			}
		}

		uploading = false;
		uploadProgress = '';

		if (successCount > 0) {
			toastSuccess(`Uploaded ${successCount} file${successCount > 1 ? 's' : ''}${failCount > 0 ? ` (${failCount} failed)` : ''}`);
		}

		// Reset the file input
		if (fileInput) fileInput.value = '';
	}

	async function handleIndex(fileId: string, filename: string) {
		indexingId = fileId;
		try {
			const result = await indexFile(projectId, fileId);
			toastSuccess(`Indexed "${filename}": ${result.chunks} chunks`);
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Indexing failed');
		} finally {
			indexingId = null;
		}
	}

	async function handleDelete(fileId: string, filename: string) {
		deletingId = fileId;
		try {
			await deleteFile(projectId, fileId);
			toastSuccess(`Deleted "${filename}"`);
			deleteConfirmId = null;
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Delete failed');
		} finally {
			deletingId = null;
		}
	}

	async function handleShowSummary(fileId: string) {
		showSummary = true;
		summaryLoading = true;
		summaryData = null;
		try {
			summaryData = await getFileSummary(projectId, fileId);
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to load summary');
			showSummary = false;
		} finally {
			summaryLoading = false;
		}
	}
</script>

<div class="space-y-3">
	<!-- Drop zone / upload area -->
	<div
		class="relative rounded-lg border-2 border-dashed transition-colors text-center py-6 px-4"
		style="border-color: {dragging ? 'var(--oo-acc-500)' : 'var(--oo-bd-subtle)'};
			background-color: {dragging ? 'var(--oo-warning-bg)' : 'transparent'};"
		on:dragenter={handleDragEnter}
		on:dragover={handleDragOver}
		on:dragleave={handleDragLeave}
		on:drop={handleDrop}
		role="region"
		aria-label="File upload drop zone"
	>
		{#if uploading}
			<div class="flex flex-col items-center gap-2">
				<div class="w-6 h-6 border-2 rounded-full animate-spin"
					style="border-color: var(--oo-bd-default); border-top-color: var(--oo-acc-500);" />
				<span class="text-xs" style="color: var(--oo-fg-muted);">{uploadProgress}</span>
			</div>
		{:else}
			<svg class="w-8 h-8 mx-auto mb-2" style="color: {dragging ? 'var(--oo-acc-400)' : 'var(--oo-fg-faint)'};"
				fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
				<path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
			</svg>
			<p class="text-xs mb-1" style="color: var(--oo-fg-muted);">
				{dragging ? 'Drop files here' : 'Drag and drop files here'}
			</p>
			<p class="text-[11px] mb-2" style="color: var(--oo-fg-faint);">
				or
			</p>
			<button
				on:click={() => fileInput?.click()}
				class="px-3 py-1.5 rounded text-xs font-medium transition-colors"
				style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-default); color: var(--oo-fg-primary);"
			>
				Browse Files
			</button>
			<input
				bind:this={fileInput}
				type="file"
				multiple
				class="hidden"
				on:change={handleFileSelect}
			/>
		{/if}
	</div>

	<!-- File list -->
	{#if files.length === 0}
		<p class="text-xs text-center py-4" style="color: var(--oo-fg-faint);">
			No files uploaded yet.
		</p>
	{:else}
		<div class="space-y-1">
			{#each files as f (f.id)}
				<div class="group flex items-center gap-2.5 px-3 py-2 rounded transition-colors"
					style="background-color: var(--oo-bg-elevated); border: 1px solid var(--oo-bd-subtle);">
					<!-- Type icon -->
					<div class="shrink-0 w-7 h-7 flex items-center justify-center rounded"
						style="background-color: var(--oo-bg-base);">
						{#if fileTypeIcon(f.filename) === 'code'}
							<svg class="w-4 h-4" style="color: var(--oo-acc-400);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								<path d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
							</svg>
						{:else if fileTypeIcon(f.filename) === 'data'}
							<svg class="w-4 h-4" style="color: var(--oo-info);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								<path d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
							</svg>
						{:else if fileTypeIcon(f.filename) === 'doc'}
							<svg class="w-4 h-4" style="color: var(--oo-success);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								<path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
							</svg>
						{:else}
							<svg class="w-4 h-4" style="color: var(--oo-fg-faint);" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								<path d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
							</svg>
						{/if}
					</div>

					<!-- File info -->
					<div class="min-w-0 flex-1">
						<div class="flex items-center gap-2">
							<span class="text-xs font-medium truncate" style="color: var(--oo-fg-primary);">
								{f.filename}
							</span>
							<!-- Indexed badge -->
							{#if f.indexed}
								<span class="shrink-0 px-1.5 py-0.5 rounded text-[10px]"
									style="background-color: var(--oo-success-bg); color: var(--oo-success);">
									Indexed
								</span>
							{:else}
								<span class="shrink-0 px-1.5 py-0.5 rounded text-[10px]"
									style="background-color: var(--oo-warning-bg); color: var(--oo-warning);">
									Not indexed
								</span>
							{/if}
						</div>
						<div class="flex items-center gap-3 mt-0.5 text-[10px]" style="color: var(--oo-fg-faint);">
							<span>{formatFileSize(f.file_size_bytes)}</span>
							{#if f.file_type}
								<span>{f.file_type}</span>
							{/if}
							{#if f.chunk_count > 0}
								<span>{f.chunk_count} chunks</span>
							{/if}
						</div>
					</div>

					<!-- Action buttons (visible on hover) -->
					<div class="flex items-center gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
						<!-- Index / Reindex button -->
						<button
							on:click={() => handleIndex(f.id, f.filename)}
							disabled={indexingId === f.id}
							class="p-1 rounded-md transition-colors disabled:opacity-50"
							style="color: var(--oo-fg-tertiary);"
							title={f.indexed ? 'Reindex file' : 'Index file'}
						>
							{#if indexingId === f.id}
								<div class="w-4 h-4 border-2 rounded-full animate-spin"
									style="border-color: var(--oo-bd-default); border-top-color: var(--oo-acc-500);" />
							{:else}
								<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
									<path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
								</svg>
							{/if}
						</button>

						<!-- Summary button -->
						{#if f.indexed}
							<button
								on:click={() => handleShowSummary(f.id)}
								class="p-1 rounded-md transition-colors"
								style="color: var(--oo-fg-tertiary);"
								title="View summary and key terms"
							>
								<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
									<path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
								</svg>
							</button>
						{/if}

						<!-- Delete button -->
						{#if deleteConfirmId === f.id}
							<button
								on:click={() => handleDelete(f.id, f.filename)}
								disabled={deletingId === f.id}
								class="px-2 py-0.5 rounded text-[10px] font-medium transition-colors"
								style="background-color: var(--oo-error-bg); color: var(--oo-error);"
							>
								{deletingId === f.id ? '...' : 'Confirm'}
							</button>
							<button
								on:click={() => (deleteConfirmId = null)}
								class="px-1.5 py-0.5 rounded text-[10px]"
								style="color: var(--oo-fg-muted);"
							>
								No
							</button>
						{:else}
							<button
								on:click={() => (deleteConfirmId = f.id)}
								class="p-1 rounded-md transition-colors"
								style="color: var(--oo-fg-faint);"
								title="Delete file"
							>
								<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
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

<!-- Summary modal -->
<Modal
	open={showSummary}
	variant="center"
	size="md"
	title={summaryData?.filename ?? 'File Summary'}
	onClose={() => (showSummary = false)}
>
	{#if summaryLoading}
		<div class="flex items-center gap-2 justify-center py-6 text-xs" style="color: var(--oo-fg-muted);">
			<div class="w-4 h-4 border-2 rounded-full animate-spin"
				style="border-color: var(--oo-bd-default); border-top-color: var(--oo-acc-500);"></div>
			Loading summary...
		</div>
	{:else if summaryData}
		<!-- Status -->
		<div class="flex items-center gap-2 mb-3 text-xs">
			{#if summaryData.indexed}
				<span class="px-1.5 py-0.5 rounded"
					style="background-color: var(--oo-success-bg); color: var(--oo-success);">
					Indexed
				</span>
				<span style="color: var(--oo-fg-faint);">{summaryData.chunk_count} chunks</span>
			{:else}
				<span class="px-1.5 py-0.5 rounded"
					style="background-color: var(--oo-warning-bg); color: var(--oo-warning);">
					Not indexed
				</span>
			{/if}
		</div>

		<!-- Summary text -->
		{#if summaryData.summary}
			<div class="mb-3">
				<h4 class="text-[11px] font-medium mb-1" style="color: var(--oo-fg-tertiary);">Summary</h4>
				<p class="text-xs leading-relaxed" style="color: var(--oo-fg-muted);">
					{summaryData.summary}
				</p>
			</div>
		{/if}

		<!-- Key terms -->
		{#if summaryData.key_terms && summaryData.key_terms.length > 0}
			<div>
				<h4 class="text-[11px] font-medium mb-1" style="color: var(--oo-fg-tertiary);">Key Terms</h4>
				<div class="flex flex-wrap gap-1">
					{#each summaryData.key_terms as term}
						<span class="px-1.5 py-0.5 rounded text-[10px]"
							style="background-color: var(--oo-bg-base); color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);">
							{term}
						</span>
					{/each}
				</div>
			</div>
		{/if}

		{#if !summaryData.summary && (!summaryData.key_terms || summaryData.key_terms.length === 0)}
			<p class="text-xs py-4 text-center" style="color: var(--oo-fg-faint);">
				No summary available. Index the file first to generate summary and key terms.
			</p>
		{/if}
	{/if}
</Modal>
