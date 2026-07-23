<!--
  FileUpload.svelte
  Drag-and-drop file upload with attached file badges.
  Integrates above ChatInput.
  Full-window drop zone overlay, per-file progress, batch upload,
  configurable max file size, file type icons, rejection feedback.
-->
<script lang="ts">
	import { createEventDispatcher, onMount, onDestroy } from 'svelte';
	import { uploadFile, validateFile, MAX_FILE_SIZE } from '$lib/api/files';
	import { toastError } from '$lib/stores/notifications';
	import type { AttachedFile } from '$lib/types';

	export let attachedFiles: AttachedFile[] = [];
	export let disabled: boolean = false;
	// Configurable max file size (bytes), defaults to value from files.ts
	export let maxFileSize: number = MAX_FILE_SIZE;
	// Max batch size
	export let maxBatchSize: number = 10;

	const dispatch = createEventDispatcher<{
		attach: AttachedFile;
		remove: number;
	}>();

	let isDragOver = false;
	let fileInput: HTMLInputElement;
	// Per-file upload progress tracking
	interface UploadProgress {
		filename: string;
		status: 'uploading' | 'done' | 'error';
		errorMsg?: string;
	}
	let uploadQueue: UploadProgress[] = [];
	$: isUploading = uploadQueue.some(q => q.status === 'uploading');
	// Track drag enter/leave depth for full-window overlay
	let dragDepth = 0;

	async function handleFiles(files: FileList | null) {
		if (!files || files.length === 0 || disabled) return;

		const fileArray = Array.from(files);

		// Batch size validation
		if (fileArray.length > maxBatchSize) {
			toastError(`Too many files: ${fileArray.length} (max ${maxBatchSize})`);
			return;
		}

		// Initialize progress entries
		const newEntries: UploadProgress[] = fileArray.map(f => ({
			filename: f.name,
			status: 'uploading' as const,
		}));
		uploadQueue = [...uploadQueue, ...newEntries];

		for (let i = 0; i < fileArray.length; i++) {
			const file = fileArray[i];
			const queueIndex = uploadQueue.length - fileArray.length + i;

			// Use configurable max size for validation
			const error = validateFileWithSize(file);
			if (error) {
				toastError(`${file.name}: ${error}`);
				uploadQueue = uploadQueue.map((q, idx) =>
					idx === queueIndex ? { ...q, status: 'error', errorMsg: error } : q
				);
				continue;
			}

			try {
				const result = await uploadFile(file);
				const attached: AttachedFile = {
					filename: result.filename,
					content: result.content,
					size_bytes: result.size_bytes,
					extension: result.extension,
				};
				dispatch('attach', attached);
				uploadQueue = uploadQueue.map((q, idx) =>
					idx === queueIndex ? { ...q, status: 'done' } : q
				);
			} catch (err) {
				const msg = err instanceof Error ? err.message : 'Upload failed';
				toastError(`${file.name}: ${msg}`);
				uploadQueue = uploadQueue.map((q, idx) =>
					idx === queueIndex ? { ...q, status: 'error', errorMsg: msg } : q
				);
			}
		}

		// Clear completed entries after a short delay
		setTimeout(() => {
			uploadQueue = uploadQueue.filter(q => q.status === 'uploading');
		}, 2000);

		// Reset file input
		if (fileInput) fileInput.value = '';
	}

	// Validate with configurable max size
	function validateFileWithSize(file: File): string | null {
		// First run standard validation (extension check)
		const baseError = validateFile(file);
		if (baseError) {
			// If the base error is about size, override with configurable limit
			if (baseError.includes('too large') && file.size <= maxFileSize) {
				return null;
			}
			if (!baseError.includes('too large')) {
				return baseError;
			}
		}
		// Check against configurable limit
		if (file.size > maxFileSize) {
			const sizeKB = (file.size / 1024).toFixed(0);
			const maxKB = (maxFileSize / 1024).toFixed(0);
			return `File too large: ${sizeKB}KB (max ${maxKB}KB)`;
		}
		return null;
	}

	// Full-window drag handlers (depth tracking avoids flicker on child elements)
	function handleWindowDragEnter(event: DragEvent) {
		event.preventDefault();
		if (disabled) return;
		dragDepth++;
		if (dragDepth === 1) {
			isDragOver = true;
		}
	}

	function handleWindowDragOver(event: DragEvent) {
		event.preventDefault();
	}

	function handleWindowDragLeave(_event: DragEvent) {
		dragDepth--;
		if (dragDepth <= 0) {
			dragDepth = 0;
			isDragOver = false;
		}
	}

	function handleWindowDrop(event: DragEvent) {
		event.preventDefault();
		dragDepth = 0;
		isDragOver = false;
		if (!disabled) handleFiles(event.dataTransfer?.files ?? null);
	}

	function handleClick() {
		if (!disabled && fileInput) fileInput.click();
	}

	function handleInputChange(event: Event) {
		const input = event.target as HTMLInputElement;
		handleFiles(input.files);
	}

	function removeFile(index: number) {
		dispatch('remove', index);
	}

	function formatSize(bytes: number): string {
		if (bytes < 1024) return `${bytes}B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
	}

	function extensionColor(ext: string): string {
		const colors: Record<string, string> = {
			'.py': 'bg-[var(--oo-info)]/20 text-[var(--oo-info)]',
			'.r': 'bg-[var(--oo-success)]/20 text-[var(--oo-success)]',
			'.R': 'bg-[var(--oo-success)]/20 text-[var(--oo-success)]',
			'.js': 'bg-[var(--oo-warning)]/20 text-[var(--oo-warning)]',
			'.ts': 'bg-[var(--oo-info)]/20 text-[var(--oo-info)]',
			'.json': 'bg-[var(--oo-cat-orange)]/20 text-[var(--oo-cat-orange)]',
			'.md': 'bg-[var(--oo-cat-purple)]/20 text-[var(--oo-cat-purple)]',
			'.csv': 'bg-[var(--oo-success)]/20 text-[var(--oo-success)]',
			'.sh': 'bg-[var(--oo-fg-muted)]/20 text-[var(--oo-fg-tertiary)]',
			'.pdf': 'bg-[var(--oo-error)]/20 text-[var(--oo-error)]',
		};
		return colors[ext] || 'bg-surface-700 text-surface-300';
	}

	// Register/unregister window-level drag listeners
	onMount(() => {
		window.addEventListener('dragenter', handleWindowDragEnter);
		window.addEventListener('dragover', handleWindowDragOver);
		window.addEventListener('dragleave', handleWindowDragLeave);
		window.addEventListener('drop', handleWindowDrop);
	});

	onDestroy(() => {
		window.removeEventListener('dragenter', handleWindowDragEnter);
		window.removeEventListener('dragover', handleWindowDragOver);
		window.removeEventListener('dragleave', handleWindowDragLeave);
		window.removeEventListener('drop', handleWindowDrop);
	});
</script>

<!-- Hidden file input -->
<input
	bind:this={fileInput}
	type="file"
	class="hidden"
	multiple
	on:change={handleInputChange}
	{disabled}
/>

<!-- Full-window drop overlay -->
{#if isDragOver}
	<div class="drop-overlay" aria-hidden="true">
		<div class="drop-overlay-content">
			<svg class="drop-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
				<path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
			</svg>
			<span class="drop-text">Drop files to upload</span>
			<span class="drop-hint">
				Max {formatSize(maxFileSize)} per file, up to {maxBatchSize} files
			</span>
		</div>
	</div>
{/if}

<!-- Main wrapper -->
<div class="relative">
	<!-- Upload progress indicators -->
	{#if uploadQueue.length > 0}
		<div class="upload-progress-list">
			{#each uploadQueue as item}
				<div class="upload-progress-item"
					class:done={item.status === 'done'}
					class:error={item.status === 'error'}>
					{#if item.status === 'uploading'}
						<svg class="w-3 h-3 animate-spin flex-shrink-0" fill="none" viewBox="0 0 24 24">
							<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
							<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
						</svg>
					{:else if item.status === 'done'}
						<svg class="w-3 h-3 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
							<path d="M5 13l4 4L19 7" />
						</svg>
					{:else}
						<svg class="w-3 h-3 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
							<path d="M6 18L18 6M6 6l12 12" />
						</svg>
					{/if}
					<span class="upload-filename">{item.filename}</span>
					{#if item.errorMsg}
						<span class="upload-error-msg">{item.errorMsg}</span>
					{/if}
				</div>
			{/each}
		</div>
	{/if}

	<!-- Attached files badges -->
	{#if attachedFiles.length > 0}
		<div class="flex flex-wrap gap-1.5 mb-2">
			{#each attachedFiles as file, i}
				<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs
					{extensionColor(file.extension)}">
					<!-- File icon -->
					<svg class="w-3 h-3 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
						<polyline points="14 2 14 8 20 8" />
					</svg>
					<span class="truncate max-w-[120px]">{file.filename}</span>
					<span class="text-[10px] opacity-70">{formatSize(file.size_bytes)}</span>
					<!-- Remove button -->
					<button
						on:click|stopPropagation={() => removeFile(i)}
						class="ml-0.5 p-0.5 rounded hover:bg-surface-800/30 transition-colors"
						title="Remove file"
					>
						<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
							<path d="M6 18L18 6M6 6l12 12" />
						</svg>
					</button>
				</span>
			{/each}
		</div>
	{/if}

	<!-- Upload button + uploading indicator -->
	<div class="flex items-center gap-2">
		<button
			on:click={handleClick}
			disabled={disabled || isUploading}
			class="p-1.5 rounded-lg transition-colors
				disabled:opacity-30 disabled:cursor-not-allowed"
			style="color: var(--oo-fg-muted); background: none;"
			title="Attach file (max {formatSize(maxFileSize)})"
		>
			{#if isUploading}
				<svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
					<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
					<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
				</svg>
			{:else}
				<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
				</svg>
			{/if}
		</button>

		<!-- Slot for ChatInput textarea -->
		<div class="flex-1 min-w-0">
			<slot />
		</div>
	</div>
</div>

<style>
	/* Full-window drop overlay */
	.drop-overlay {
		position: fixed;
		inset: 0;
		z-index: 9999;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--oo-overlay-bg, rgba(0, 0, 0, 0.6));
		backdrop-filter: blur(4px);
		pointer-events: none;
	}

	.drop-overlay-content {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.75rem;
		padding: 2.5rem 3rem;
		border: 2px dashed var(--oo-accent);
		border-radius: 16px;
		background: var(--oo-surface);
	}

	.drop-icon {
		width: 2.5rem;
		height: 2.5rem;
		color: var(--oo-accent);
	}

	.drop-text {
		font-size: 1rem;
		font-weight: 600;
		color: var(--oo-text-primary);
	}

	.drop-hint {
		font-size: 0.75rem;
		color: var(--oo-text-tertiary);
	}

	/* Upload progress list */
	.upload-progress-list {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		margin-bottom: 0.5rem;
	}

	.upload-progress-item {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		padding: 0.2rem 0.5rem;
		border-radius: 4px;
		font-size: 0.6875rem;
		color: var(--oo-text-secondary);
		background: var(--oo-surface-hover);
	}

	.upload-progress-item.done {
		color: var(--oo-success);
	}

	.upload-progress-item.error {
		color: var(--oo-error);
	}

	.upload-filename {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		max-width: 200px;
	}

	.upload-error-msg {
		font-size: 0.625rem;
		opacity: 0.8;
		margin-left: auto;
	}
</style>
