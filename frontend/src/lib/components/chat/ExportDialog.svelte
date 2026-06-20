<!--
  ExportDialog.svelte
  Modal dialog for exporting a conversation (Markdown, JSON, HTML).
  Actions: preview, copy to clipboard, download.
  S166: migrated to the shared <Modal> primitive, which provides the
  native <dialog> focus trap plus Escape and backdrop handling.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { exportConversation, downloadExport, type ExportFormat } from '$lib/api/export';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import { Modal, Button } from '$lib/ds';

	export let conversationId: string;
	export let conversationTitle: string = 'conversation';

	const dispatch = createEventDispatcher<{ close: void }>();

	let open = true;
	let format: ExportFormat = 'markdown';
	let content = '';
	let filename = '';
	let loading = false;
	let error = '';

	const formats: { id: ExportFormat; label: string; icon: string }[] = [
		{ id: 'markdown', label: 'Markdown', icon: 'M3 5h2l3 10h1l3-10h2v14H12V9l-3 10H7L4 9v10H3V5z' },
		{ id: 'json', label: 'JSON', icon: 'M8 3a2 2 0 00-2 2v1.5A1.5 1.5 0 014.5 8H3v2h1.5A1.5 1.5 0 016 11.5V14a2 2 0 002 2m8-13a2 2 0 012 2v1.5A1.5 1.5 0 0019.5 8H21v2h-1.5a1.5 1.5 0 00-1.5 1.5V14a2 2 0 01-2 2' },
		{ id: 'html', label: 'HTML', icon: 'M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4' }
	];

	async function loadExport() {
		loading = true;
		error = '';
		content = '';
		try {
			const data = await exportConversation(conversationId, format);
			content = data.content;
			filename = data.filename ?? `export-${conversationId}.${format}`;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Export failed';
		} finally {
			loading = false;
		}
	}

	function handleDownload() {
		if (!content) return;
		downloadExport(content, filename, format);
		toastSuccess(`Exported as ${filename}`);
	}

	async function handleCopy() {
		if (!content) return;
		try {
			await navigator.clipboard.writeText(content);
			toastSuccess('Copied to clipboard');
		} catch {
			toastError('Failed to copy to clipboard');
		}
	}

	function close() {
		open = false;
		dispatch('close');
	}

	// Load on mount and when the format changes.
	$: if (conversationId && format) {
		loadExport();
	}

	$: preview = content.length > 2000 ? content.slice(0, 2000) + '\n\n... (truncated)' : content;
</script>

<Modal {open} variant="center" size="lg" title="Export Conversation" onClose={close}>
	<p class="oo-export-subtitle">{conversationTitle}</p>

	<!-- Format selector -->
	<div class="oo-export-formats">
		{#each formats as f}
			<button
				type="button"
				on:click={() => (format = f.id)}
				class="oo-export-fmt"
				class:active={format === f.id}
			>
				<svg class="oo-export-fmt-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path stroke-linecap="round" stroke-linejoin="round" d={f.icon} />
				</svg>
				{f.label}
			</button>
		{/each}
	</div>

	<!-- Preview -->
	<div class="oo-export-preview">
		{#if loading}
			<div class="oo-export-loading">
				<span class="oo-export-spinner" aria-hidden="true"></span>
				<span>Generating export...</span>
			</div>
		{:else if error}
			<div class="oo-export-error">{error}</div>
		{:else if content}
			<pre class="oo-export-pre">{preview}</pre>
		{/if}
	</div>

	<svelte:fragment slot="footer">
		<span class="oo-export-filename">{filename}</span>
		<Button variant="secondary" size="sm" iconLeft="copy" disabled={!content || loading} on:click={handleCopy}>
			Copy
		</Button>
		<Button variant="primary" size="sm" iconLeft="download" disabled={!content || loading} on:click={handleDownload}>
			Download
		</Button>
	</svelte:fragment>
</Modal>

<style>
	.oo-export-subtitle {
		margin: 0 0 var(--oo-space-4);
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.oo-export-formats {
		display: flex;
		gap: var(--oo-space-2);
		margin-bottom: var(--oo-space-4);
	}
	.oo-export-fmt {
		display: inline-flex;
		align-items: center;
		gap: var(--oo-space-2);
		padding: var(--oo-space-2) var(--oo-space-3);
		border: 1px solid transparent;
		border-radius: var(--oo-radius-md);
		background: transparent;
		color: var(--oo-fg-secondary);
		font-size: var(--oo-text-xs);
		font-weight: 500;
		cursor: pointer;
	}
	.oo-export-fmt:hover {
		background-color: var(--oo-bg-hover);
		color: var(--oo-fg-primary);
	}
	.oo-export-fmt.active {
		background-color: var(--oo-acc-100);
		color: var(--oo-acc-700);
		border-color: var(--oo-acc-300);
	}
	.oo-export-fmt-icon {
		width: 14px;
		height: 14px;
	}
	.oo-export-preview {
		min-height: 8rem;
		max-height: 50vh;
		overflow-y: auto;
	}
	.oo-export-loading {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: var(--oo-space-3);
		padding: var(--oo-space-8) 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
	}
	.oo-export-spinner {
		width: 18px;
		height: 18px;
		border: 2px solid var(--oo-bd-strong);
		border-top-color: var(--oo-acc-500);
		border-radius: var(--oo-radius-full);
		animation: oo-export-spin 0.7s linear infinite;
	}
	@keyframes oo-export-spin {
		to {
			transform: rotate(360deg);
		}
	}
	.oo-export-error {
		padding: var(--oo-space-3) var(--oo-space-4);
		border-radius: var(--oo-radius-md);
		background-color: var(--oo-error-bg);
		border: 1px solid var(--oo-error-bd);
		color: var(--oo-error);
		font-size: var(--oo-text-sm);
	}
	.oo-export-pre {
		margin: 0;
		padding: var(--oo-space-3);
		background-color: var(--oo-bg-subtle);
		border-radius: var(--oo-radius-md);
		font-family: var(--oo-font-mono);
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-secondary);
		white-space: pre-wrap;
		word-break: break-word;
	}
	.oo-export-filename {
		flex: 1;
		font-family: var(--oo-font-mono);
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-muted);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	@media (prefers-reduced-motion: reduce) {
		.oo-export-spinner {
			animation-duration: 1.2s;
		}
	}
</style>
