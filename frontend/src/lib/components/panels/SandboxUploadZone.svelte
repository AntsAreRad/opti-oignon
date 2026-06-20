<!--
  SandboxUploadZone.svelte (S211, Sandbox Workspace cycle, Bloc 2)
  Drag-and-drop / file-picker upload zone (spec 5.1): an EXPLICIT user
  action targeting the selected workspace -- the model can trigger no
  upload (S73/S74), and no host path is ever read by the server on this
  path (the browser supplies the bytes as multipart). Caps are surfaced
  honestly: an exceeded cap (file count, per-file bytes, workspace quota)
  refuses the whole request with 413; invalid names and destination
  collisions come back as per-file refusals and are listed, never
  overwritten. Successful uploads record the section 6.1 baseline manifest
  server-side. Design-system tokens only (--oo-*); lucide icons through
  Icon. Registered in FRONTEND_REDESIGN_SPEC.md.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { Button, Icon } from '$lib/ds';
	import { uploadFiles } from '$lib/api/sandbox';
	import type { SandboxUploadRefused } from '$lib/types';

	export let sessionId: string | null = null;
	export let disabled = false;

	const dispatch = createEventDispatcher<{
		uploaded: { sessionId: string; count: number; bytes: number };
	}>();

	let dragOver = false;
	let uploading = false;
	let error: string | null = null;
	let refusals: SandboxUploadRefused[] = [];
	let lastSummary = '';
	let fileInput: HTMLInputElement | null = null;

	async function send(files: File[]) {
		if (!sessionId || files.length === 0 || disabled) return;
		uploading = true;
		error = null;
		refusals = [];
		lastSummary = '';
		try {
			const result = await uploadFiles(sessionId, files);
			refusals = result.refused;
			lastSummary = `${result.uploaded_paths.length} file(s) uploaded (${result.uploaded_bytes} bytes)`;
			dispatch('uploaded', {
				sessionId,
				count: result.uploaded_paths.length,
				bytes: result.uploaded_bytes
			});
		} catch (e) {
			error = e instanceof Error ? e.message : 'Upload failed';
		} finally {
			uploading = false;
		}
	}

	function handleDrop(event: DragEvent) {
		event.preventDefault();
		dragOver = false;
		const files = Array.from(event.dataTransfer?.files ?? []);
		void send(files);
	}

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
		dragOver = true;
	}

	function handleDragLeave() {
		dragOver = false;
	}

	function handlePick(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const files = Array.from(input.files ?? []);
		input.value = '';
		void send(files);
	}

	function handleZoneKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			fileInput?.click();
		}
	}
</script>

<div class="upload-zone-wrap">
	<div
		class="upload-zone"
		class:drag-over={dragOver}
		class:zone-disabled={disabled || !sessionId}
		role="button"
		tabindex="0"
		aria-label="Upload files into the selected workspace"
		on:drop={handleDrop}
		on:dragover={handleDragOver}
		on:dragleave={handleDragLeave}
		on:keydown={handleZoneKeydown}
	>
		<Icon name="upload" />
		<p class="upload-hint">
			{#if !sessionId}
				Select a workspace to upload files into it.
			{:else}
				Drop files here, or pick them. They land in the workspace only;
				nothing returns to the host without your approval.
			{/if}
		</p>
		<Button
			variant="secondary"
			size="sm"
			iconLeft="file-plus"
			loading={uploading}
			disabled={disabled || !sessionId || uploading}
			ariaLabel="Pick files to upload"
			on:click={() => fileInput?.click()}
		>
			Pick files
		</Button>
		<input
			class="upload-input"
			type="file"
			multiple
			bind:this={fileInput}
			on:change={handlePick}
			disabled={disabled || !sessionId || uploading}
			aria-hidden="true"
			tabindex="-1"
		/>
	</div>

	{#if lastSummary}
		<p class="upload-summary" role="status">{lastSummary}</p>
	{/if}
	{#if error}
		<p class="upload-error" role="alert">{error}</p>
	{/if}
	{#if refusals.length > 0}
		<ul class="upload-refusals" aria-label="Refused files">
			{#each refusals as r (r.name + r.reason)}
				<li><span class="refusal-name">{r.name}</span> {r.reason}</li>
			{/each}
		</ul>
	{/if}
</div>

<style>
	.upload-zone-wrap {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
	}
	.upload-zone {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.5rem;
		padding: 0.9rem;
		border: 1px dashed var(--oo-bd-default);
		border-radius: var(--oo-radius-md, 8px);
		color: var(--oo-fg-secondary);
		background: var(--oo-bg-surface);
		text-align: center;
		cursor: pointer;
	}
	.upload-zone:focus-visible {
		outline: 2px solid var(--oo-acc-500);
		outline-offset: 2px;
	}
	.drag-over {
		border-color: var(--oo-acc-500);
		color: var(--oo-fg-primary);
	}
	.zone-disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}
	.upload-hint {
		margin: 0;
		font-size: 0.75rem;
	}
	.upload-input {
		display: none;
	}
	.upload-summary {
		margin: 0;
		font-size: 0.75rem;
		color: var(--oo-success);
	}
	.upload-error {
		margin: 0;
		font-size: 0.75rem;
		color: var(--oo-error);
	}
	.upload-refusals {
		margin: 0;
		padding-left: 1rem;
		font-size: 0.75rem;
		color: var(--oo-warning);
	}
	.refusal-name {
		font-weight: 600;
		color: var(--oo-fg-primary);
	}
</style>
