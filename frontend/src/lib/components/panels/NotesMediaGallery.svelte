<!--
  NotesMediaGallery.svelte (Notes feature N.6 UI half)
  The encrypted image gallery for the active note: a picker uploads images as
  encrypted attachments over the route (sealed server-side under a
  per-attachment subkey; nothing plaintext touches disk), thumbnails decrypt
  in memory through short-lived object URLs (revoked on removal and destroy),
  and each image offers the opt-in caption / OCR as preview-then-approve:
  the first run returns the text for review without persisting, the explicit
  approval writes it back. A structured refusal (the fail-secure sandbox
  gate, the absent opt-in vision extra) is shown with its reason, never
  silently dropped. Design-system tokens only (--oo-*); lucide-svelte icons
  through Icon.
-->
<script lang="ts">
	import { onDestroy } from 'svelte';
	import { Button, Card, Icon, EmptyState, InlineError } from '$lib/ds';
	import {
		imageAttachments,
		mediaLoading,
		mediaError,
		loadAttachments,
		uploadNoteAttachment,
		removeAttachment,
		captionAttachment
	} from '$lib/stores/attachments';
	import { fetchAttachmentBlob, type AttachmentRecord } from '$lib/api/attachments';
	import type { CaptionResult } from '$lib/api/caption';
	import { toastSuccess, toastError } from '$lib/stores/notifications';

	/** The note whose image vault this gallery shows. */
	export let noteId: string;

	// Load the note's attachments (idempotent in the store, so the voice
	// control's identical call is a no-op for the same note).
	$: if (noteId) {
		void loadAttachments(noteId);
	}

	// -- Upload --

	let fileInput: HTMLInputElement | null = null;
	let uploading = false;

	function pickImages(): void {
		fileInput?.click();
	}

	async function uploadImages(): Promise<void> {
		const files = fileInput?.files;
		if (!files || files.length === 0) return;
		uploading = true;
		try {
			for (const file of Array.from(files)) {
				await uploadNoteAttachment(noteId, 'image', file, file.name);
			}
			toastSuccess(files.length > 1 ? 'Images uploaded' : 'Image uploaded');
		} catch {
			toastError('Failed to upload image');
		} finally {
			uploading = false;
			if (fileInput) fileInput.value = '';
		}
	}

	// -- Thumbnails (in-memory object URLs, revoked on removal / destroy) --

	let thumbs: Record<string, string> = {};
	const thumbPending = new Set<string>();

	async function ensureThumb(item: AttachmentRecord): Promise<void> {
		if (thumbs[item.id] || thumbPending.has(item.id)) return;
		thumbPending.add(item.id);
		try {
			const blob = await fetchAttachmentBlob(item.id);
			thumbs = { ...thumbs, [item.id]: URL.createObjectURL(blob) };
		} catch {
			// The grid shows a placeholder; the meta row still renders.
		} finally {
			thumbPending.delete(item.id);
		}
	}

	$: $imageAttachments.forEach((item) => {
		void ensureThumb(item);
	});

	function revokeThumb(id: string): void {
		const url = thumbs[id];
		if (url) {
			URL.revokeObjectURL(url);
			const { [id]: _gone, ...rest } = thumbs;
			thumbs = rest;
		}
	}

	onDestroy(() => {
		for (const url of Object.values(thumbs)) {
			URL.revokeObjectURL(url);
		}
		thumbs = {};
	});

	// -- Caption / OCR: preview-then-approve --

	let busyId: string | null = null;
	let previews: Record<string, CaptionResult> = {};

	async function describe(item: AttachmentRecord): Promise<void> {
		busyId = item.id;
		try {
			const result = await captionAttachment(item.id, false);
			previews = { ...previews, [item.id]: result };
			if (result.refused) {
				toastError(result.reason || 'Caption refused');
			}
		} catch {
			toastError('Caption request failed');
		} finally {
			busyId = null;
		}
	}

	async function approveCaption(item: AttachmentRecord): Promise<void> {
		busyId = item.id;
		try {
			const result = await captionAttachment(item.id, true);
			previews = { ...previews, [item.id]: result };
			if (result.refused) {
				toastError(result.reason || 'Caption refused');
			} else if (result.ok && result.written_back) {
				toastSuccess('Caption saved');
			}
		} catch {
			toastError('Caption request failed');
		} finally {
			busyId = null;
		}
	}

	function discardPreview(id: string): void {
		const { [id]: _gone, ...rest } = previews;
		previews = rest;
	}

	async function remove(item: AttachmentRecord): Promise<void> {
		busyId = item.id;
		try {
			revokeThumb(item.id);
			await removeAttachment(item.id);
			discardPreview(item.id);
		} catch {
			toastError('Failed to delete image');
		} finally {
			busyId = null;
		}
	}

	function savedText(item: AttachmentRecord): string {
		const parts: string[] = [];
		if (item.caption_text) parts.push(item.caption_text);
		if (item.ocr_text) parts.push(item.ocr_text);
		return parts.join('\n');
	}

	function previewText(result: CaptionResult): string {
		const parts: string[] = [];
		if (result.caption_text) parts.push(result.caption_text);
		if (result.ocr_text) parts.push(result.ocr_text);
		return parts.join('\n');
	}
</script>

<Card padding="md">
	<div class="mg-head">
		<div class="mg-title">
			<Icon name="image" size="sm" />
			<h3>Image gallery</h3>
		</div>
		<Button
			variant="secondary"
			size="sm"
			iconLeft="upload"
			loading={uploading}
			on:click={pickImages}
		>
			Add images
		</Button>
		<input
			class="mg-file-input"
			type="file"
			accept="image/*"
			multiple
			bind:this={fileInput}
			on:change={uploadImages}
		/>
	</div>

	{#if $mediaError}
		<InlineError
			message={$mediaError}
			onRetry={() => loadAttachments(noteId, true)}
			retrying={$mediaLoading}
		/>
	{:else if $imageAttachments.length === 0}
		{#if !$mediaLoading}
			<EmptyState
				icon="image"
				title="No images yet"
				description="Add images to keep them encrypted with this note."
			/>
		{/if}
	{:else}
		<ul class="mg-grid">
			{#each $imageAttachments as item (item.id)}
				<li class="mg-item">
					<div class="mg-thumb">
						{#if thumbs[item.id]}
							<img src={thumbs[item.id]} alt={item.caption_text || 'Note image'} />
						{:else}
							<span class="mg-thumb-placeholder">
								<Icon name="image" size="md" />
							</span>
						{/if}
					</div>

					<div class="mg-actions">
						<Button
							variant="ghost"
							size="sm"
							iconLeft="eye"
							loading={busyId === item.id}
							on:click={() => describe(item)}
						>
							Describe
						</Button>
						<Button
							variant="ghost"
							size="sm"
							iconLeft="trash-2"
							loading={busyId === item.id}
							on:click={() => remove(item)}
						>
							Delete
						</Button>
					</div>

					{#if previews[item.id] && previews[item.id].refused}
						<p class="mg-refused">
							Caption refused: {previews[item.id].reason || 'unavailable'}
						</p>
					{:else if previews[item.id] && previews[item.id].ok && !previews[item.id].written_back}
						<div class="mg-preview">
							<p class="mg-preview-label">Caption / OCR preview (not saved)</p>
							<p class="mg-text">{previewText(previews[item.id])}</p>
							<div class="mg-preview-actions">
								<Button
									variant="primary"
									size="sm"
									iconLeft="check"
									loading={busyId === item.id}
									on:click={() => approveCaption(item)}
								>
									Approve and save
								</Button>
								<Button variant="ghost" size="sm" on:click={() => discardPreview(item.id)}>
									Discard
								</Button>
							</div>
						</div>
					{:else if savedText(item)}
						<p class="mg-text mg-saved">{savedText(item)}</p>
					{/if}
				</li>
			{/each}
		</ul>
	{/if}
</Card>

<style>
	.mg-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-3);
		margin-bottom: var(--oo-space-3);
	}

	.mg-title {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
	}

	.mg-title h3 {
		margin: 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-primary);
	}

	.mg-file-input {
		display: none;
	}

	.mg-grid {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
		gap: var(--oo-space-3);
	}

	.mg-item {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
		padding: var(--oo-space-2);
		background: var(--oo-bg-elevated);
	}

	.mg-thumb {
		width: 100%;
		aspect-ratio: 4 / 3;
		overflow: hidden;
		border-radius: var(--oo-radius-md);
		background: var(--oo-bg-base);
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.mg-thumb img {
		width: 100%;
		height: 100%;
		object-fit: cover;
	}

	.mg-thumb-placeholder {
		color: var(--oo-fg-faint);
	}

	.mg-actions {
		display: flex;
		align-items: center;
		gap: var(--oo-space-1);
		flex-wrap: wrap;
	}

	.mg-refused {
		margin: 0;
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
	}

	.mg-preview {
		padding: var(--oo-space-2);
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md);
		background: var(--oo-bg-base);
	}

	.mg-preview-label {
		margin: 0 0 var(--oo-space-1);
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-tertiary);
	}

	.mg-text {
		margin: 0;
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-primary);
		white-space: pre-wrap;
	}

	.mg-saved {
		color: var(--oo-fg-secondary);
	}

	.mg-preview-actions {
		display: flex;
		gap: var(--oo-space-2);
		margin-top: var(--oo-space-2);
	}
</style>
