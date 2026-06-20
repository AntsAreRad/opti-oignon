<!--
  ChatInput.svelte
  Text input with auto-resize, send, cancel, retry, and image upload.
  Integrates ChatControlBar above the input area (S42).
  S48: Image upload button with thumbnails.
  S107: Ctrl+Enter global send shortcut support.
  S132: Mobile responsive — full-width, 44px touch targets, enterkeyhint, safe-area.
-->
<script lang="ts">
	import { createEventDispatcher, onMount, onDestroy, tick } from 'svelte';
	import ChatControlBar from './ChatControlBar.svelte';
	import { validateImageFile, isImageFile, imageToBase64 } from '$lib/api/files';
	import { toastError } from '$lib/stores/notifications';
	import Icon from '$lib/ds/Icon.svelte';
	import type { AttachedImage } from '$lib/types';

	export let disabled: boolean = false;
	export let isStreaming: boolean = false;
	export let canRetry: boolean = false;

	const dispatch = createEventDispatcher<{
		send: { text: string; images: string[] };
		cancel: void;
		retry: void;
		editLast: void;
	}>();

	let inputText = '';
	let textarea: HTMLTextAreaElement;
	let imageInput: HTMLInputElement;

	// S118: Detect /code slash command for visual feedback
	$: isCodeCommand = inputText.trimStart().startsWith('/code ')
		|| inputText.trimStart() === '/code';

	// S48: Attached images
	let attachedImages: AttachedImage[] = [];

	const MIN_ROWS = 1;
	const MAX_ROWS = 6;
	const LINE_HEIGHT = 24;

	$: canSend = (inputText.trim().length > 0 || attachedImages.length > 0) && !disabled && !isStreaming;

	function autoResize() {
		if (!textarea) return;
		textarea.style.height = 'auto';
		const maxHeight = MAX_ROWS * LINE_HEIGHT;
		const minHeight = MIN_ROWS * LINE_HEIGHT;
		const newHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);
		textarea.style.height = `${newHeight}px`;
	}

	async function handleInput() {
		await tick();
		autoResize();
	}

	function handleKeydown(event: KeyboardEvent) {
		// Ctrl+Enter or Cmd+Enter: send message
		if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
			event.preventDefault();
			handleSend();
			return;
		}
		// Enter (without Shift): send message
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			handleSend();
			return;
		}
		// S135: Up arrow with empty input: edit last user message
		if (event.key === 'ArrowUp' && inputText.trim() === '' && attachedImages.length === 0) {
			event.preventDefault();
			dispatch('editLast');
			return;
		}
	}

	function handleSend() {
		if (!canSend) return;
		const text = inputText.trim();
		const images = attachedImages.map((img) => img.base64_data);
		inputText = '';
		clearAttachedImages();
		if (textarea) textarea.style.height = 'auto';
		dispatch('send', { text, images });
	}

	function handleCancel() {
		dispatch('cancel');
	}

	function handleRetry() {
		dispatch('retry');
	}

	// S48: Image handling
	function handleImageClick() {
		if (!disabled && !isStreaming && imageInput) {
			imageInput.click();
		}
	}

	async function handleImageChange(event: Event) {
		const input = event.target as HTMLInputElement;
		if (!input.files) return;
		await processImageFiles(input.files);
		input.value = '';
	}

	async function processImageFiles(files: FileList) {
		const skippedNonImage: string[] = [];
		for (const file of Array.from(files)) {
			if (!isImageFile(file)) {
				skippedNonImage.push(file.name);
				continue;
			}

			const error = validateImageFile(file);
			if (error) {
				toastError(`${file.name}: ${error}`);
				continue;
			}

			try {
				const base64 = await imageToBase64(file);
				const previewUrl = URL.createObjectURL(file);

				const attached: AttachedImage = {
					filename: file.name,
					base64_data: base64,
					mime_type: file.type || 'image/png',
					size_bytes: file.size,
					preview_url: previewUrl,
				};

				attachedImages = [...attachedImages, attached];
			} catch (err) {
				const msg = err instanceof Error ? err.message : 'Unknown error';
				toastError(`Failed to process "${file.name}": ${msg}. Try a different image format.`);
			}
		}
		// S94: Inform user about skipped non-image files
		if (skippedNonImage.length > 0) {
			const names = skippedNonImage.length <= 3
				? skippedNonImage.join(', ')
				: `${skippedNonImage.slice(0, 2).join(', ')} and ${skippedNonImage.length - 2} more`;
			toastError(`Skipped non-image files: ${names}. Only images are accepted here.`);
		}
	}

	function removeImage(index: number) {
		const removed = attachedImages[index];
		if (removed?.preview_url) {
			URL.revokeObjectURL(removed.preview_url);
		}
		attachedImages = attachedImages.filter((_, i) => i !== index);
	}

	function clearAttachedImages() {
		for (const img of attachedImages) {
			if (img.preview_url) URL.revokeObjectURL(img.preview_url);
		}
		attachedImages = [];
	}

	function formatSize(bytes: number): string {
		if (bytes < 1024) return `${bytes}B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
	}

	// Handle image paste (Ctrl+V)
	function handlePaste(event: ClipboardEvent) {
		const items = event.clipboardData?.items;
		if (!items) return;

		const imageFiles: File[] = [];
		for (const item of Array.from(items)) {
			if (item.type.startsWith('image/')) {
				const file = item.getAsFile();
				if (file) imageFiles.push(file);
			}
		}

		if (imageFiles.length > 0) {
			event.preventDefault();
			const dt = new DataTransfer();
			for (const f of imageFiles) dt.items.add(f);
			processImageFiles(dt.files);
		}
	}

	// Handle image drag-and-drop
	let isDragOver = false;

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
		if (!disabled && !isStreaming) isDragOver = true;
	}

	function handleDragLeave() {
		isDragOver = false;
	}

	function handleDrop(event: DragEvent) {
		event.preventDefault();
		isDragOver = false;
		if (disabled || isStreaming) return;
		const files = event.dataTransfer?.files;
		if (files) {
			const dt = new DataTransfer();
			for (const f of Array.from(files)) {
				if (isImageFile(f)) dt.items.add(f);
			}
			if (dt.files.length > 0) {
				processImageFiles(dt.files);
			}
		}
	}

	// Listen for global Ctrl+Enter send event (from KeyboardShortcuts)
	function handleGlobalSend() {
		handleSend();
	}

	onMount(() => {
		if (textarea) textarea.focus();
		window.addEventListener('opti-send-message', handleGlobalSend);
	});

	onDestroy(() => {
		clearAttachedImages();
		window.removeEventListener('opti-send-message', handleGlobalSend);
	});
</script>

<!-- Hidden image input -->
<input
	bind:this={imageInput}
	type="file"
	class="hidden"
	accept="image/png,image/jpeg,image/gif,image/webp,image/bmp"
	multiple
	on:change={handleImageChange}
/>

<!-- svelte-ignore a11y-no-static-element-interactions -->
<div
	class="flex flex-col gap-1 relative"
	on:dragover={handleDragOver}
	on:dragleave={handleDragLeave}
	on:drop={handleDrop}
>
	<!-- Drag overlay -->
	{#if isDragOver}
		<div class="absolute inset-0 z-10 flex items-center justify-center
			rounded-xl pointer-events-none"
			style="background-color: var(--oo-warning-bg); border: 2px dashed var(--oo-warning-bd);">
			<div class="flex items-center gap-2 text-sm font-medium" style="color: var(--oo-acc-400);">
				<svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
				</svg>
				Drop images here
			</div>
		</div>
	{/if}

	<!-- S42: Control bar -->
	<ChatControlBar />

	<!-- S48/S94: Image preview thumbnails (improved size and quality) -->
	{#if attachedImages.length > 0}
		<div class="flex flex-wrap gap-2 px-1">
			{#each attachedImages as img, i}
				<div class="relative group">
					<img
						src={img.preview_url}
						alt={img.filename}
						class="w-20 h-20 object-cover rounded-xl"
						style="border: 1px solid var(--oo-bd-default);"
					/>
					<span class="absolute bottom-0 left-0 right-0 text-center text-[9px]
						bg-black/60 text-surface-300 rounded-b-xl py-0.5 truncate px-0.5">
						{formatSize(img.size_bytes)}
					</span>
					<button
						on:click|stopPropagation={() => removeImage(i)}
						class="absolute -top-1.5 -right-1.5 w-5 h-5 flex items-center justify-center
							rounded-full bg-[var(--oo-error)]/80 text-[var(--oo-fg-on-semantic)] text-xs
							opacity-0 group-hover:opacity-100 transition-opacity"
						title="Remove image"
					>
						<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="3">
							<path d="M6 18L18 6M6 6l12 12" />
						</svg>
					</button>
				</div>
			{/each}
		</div>
	{/if}

	<div class="flex items-end gap-1 sm:gap-2">
		<!-- S48: Image upload button — S132: touch-friendly -->
		<button
			on:click={handleImageClick}
			disabled={disabled || isStreaming}
			class="shrink-0 p-2 rounded-lg transition-colors touch-target
				disabled:opacity-30 disabled:cursor-not-allowed"
			style="color: var(--oo-fg-muted);"
			title="Attach image (or paste/drag)"
			aria-label="Attach image"
		>
			<Icon name="image" size="sm" />
		</button>

		<!-- Textarea — S132: full-width mobile, enterkeyhint for mobile keyboard -->
		<div class="flex-1 relative">
			<!-- S118: /code slash command indicator -->
			{#if isCodeCommand}
				<div class="absolute -top-5 left-1 flex items-center gap-1 text-xs px-1.5 py-0.5 rounded z-10"
					style="background-color: var(--oo-sage-bg); color: var(--oo-sage); border: 1px solid var(--oo-sage-bd);">
					<svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
						<path d="M16 18l2-2-2-2" /><path d="M8 6L6 8l2 2" />
						<path d="M14.5 4l-5 16" />
					</svg>
					Code Agent
				</div>
			{/if}
			<textarea
				bind:this={textarea}
				bind:value={inputText}
				on:input={handleInput}
				on:keydown={handleKeydown}
				on:paste={handlePaste}
				enterkeyhint="send"
				placeholder={isStreaming ? 'Generating...' : attachedImages.length > 0 ? 'Describe the image...' : 'Type a message...'}
				disabled={disabled || isStreaming}
				rows={MIN_ROWS}
				class="w-full text-sm rounded-xl outline-none resize-none
					disabled:opacity-50 disabled:cursor-not-allowed"
				style="background-color: var(--oo-input-bg); color: var(--oo-fg-primary);
					padding: 0.625rem 1rem; border: 1px solid var(--oo-input-bd);
					line-height: {LINE_HEIGHT}px; max-height: {MAX_ROWS * LINE_HEIGHT}px;
					font-family: var(--oo-font-sans); font-size: 16px;"
			/>
		</div>

		<!-- S132: Send/stop button — 44x44px touch target -->
		{#if isStreaming}
			<button
				on:click={handleCancel}
				class="shrink-0 rounded-xl transition-colors touch-target"
				style="background-color: var(--oo-error-bg); color: var(--oo-error);
					width: 44px; height: 44px; display: flex; align-items: center; justify-content: center;"
				title="Stop generation"
				aria-label="Stop generation"
			>
				<Icon name="square" size="md" />
			</button>
		{:else}
			<button
				on:click={handleSend}
				disabled={!canSend}
				class="shrink-0 rounded-xl transition-colors touch-target
					disabled:opacity-30 disabled:cursor-not-allowed"
				style="background-color: var(--oo-btn-primary-bg); color: var(--oo-btn-primary-fg);
					width: 44px; height: 44px; display: flex; align-items: center; justify-content: center;"
				title="Send (Enter or Ctrl+Enter)"
				aria-label="Send message"
			>
				<Icon name="send" size="md" />
			</button>
		{/if}
	</div>

	<!-- Retry button -->
	{#if canRetry && !isStreaming}
		<button
			on:click={handleRetry}
			class="self-center inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs
				transition-colors touch-target"
			style="color: var(--oo-fg-muted);"
			title="Regenerate response"
			aria-label="Regenerate response"
		>
			<Icon name="refresh-cw" size="sm" />
			Regenerate
		</button>
	{/if}
</div>
