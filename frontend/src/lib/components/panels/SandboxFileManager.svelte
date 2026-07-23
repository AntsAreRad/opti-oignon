<script lang="ts">
	/**
	 * SandboxFileManager - Sandbox File Copy-Out + Human Approval UI.
	 *
	 * Displays files inside an active sandbox session, allows preview,
	 * checkbox selection, approve & download, and reject all.
	 * No file is ever copied out without explicit user action.
	 */
	import { onDestroy } from 'svelte';
	import { SandboxFilesPoller } from '$lib/sandbox/filesPolling';
	import type { PollState } from '$lib/sandbox/filesPolling';
	import { ApiError } from '$lib/api/client';
	import {
		listSandboxFiles,
		previewSandboxFile,
		approveSandboxFiles,
		copyOutSandboxFiles,
		rejectSandboxFiles,
		getSandboxDownloadUrl,
		getApprovalInfo
	} from '$lib/api/sandbox';
	import type {
		SandboxFileEntry,
		SandboxFilesResponse,
		SandboxPreviewResponse,
		SandboxApprovalInfoResponse
	} from '$lib/types';

	// -- Props --
	export let sessionId: string = '';
	/** Snapshot from the done metadata: shown at once, before any fetch. */
	export let initialFiles: SandboxFileEntry[] | null = null;

	// -- State --
	let files: SandboxFileEntry[] = [];
	let approvalState: string = 'pending';
	let loading = false;
	let error = '';
	let successMsg = '';

	// Selection
	let selectedPaths: Set<string> = new Set();
	$: allSelected = files.length > 0 && files.every(f => selectedPaths.has(f.path));

	// Preview
	let previewFile: SandboxPreviewResponse | null = null;
	let previewLoading = false;

	// Polling, delegated to the dependency-free scheduler: a 404 is
	// terminal for a given id (absent when never listed, expired after
	// life) instead of an endless request spam, transient failures back
	// off, and switching ids re-arms the machine.
	const poller = new SandboxFilesPoller();
	let lifecycle: PollState = 'idle';
	let pollTimer: ReturnType<typeof setTimeout> | null = null;
	let trackedSession = '';

	function stopPolling() {
		if (pollTimer !== null) {
			clearTimeout(pollTimer);
			pollTimer = null;
		}
	}

	function schedule(delayMs: number | null) {
		stopPolling();
		if (delayMs === null) return;
		pollTimer = setTimeout(() => {
			void fetchFiles();
		}, delayMs);
	}

	function seedFiles(entries: SandboxFileEntry[] | null): SandboxFileEntry[] {
		if (!entries) return [];
		return entries
			.filter((entry) => entry && typeof entry.path === 'string')
			.map((entry) => ({
				path: entry.path,
				size: Number(entry.size ?? 0),
				modified: Number(entry.modified ?? 0),
				approved: Boolean(entry.approved ?? false)
			}));
	}

	// -- Lifecycle --
	onDestroy(stopPolling);

	$: if (sessionId !== trackedSession) {
		trackedSession = sessionId;
		stopPolling();
		if (sessionId) {
			files = seedFiles(initialFiles);
			selectedPaths = new Set();
			const decision = poller.onSessionChange();
			lifecycle = decision.state;
			schedule(decision.nextDelayMs);
		} else {
			files = [];
			lifecycle = 'idle';
		}
	}

	// -- Data fetching --
	async function fetchFiles() {
		if (!sessionId) return;
		try {
			const res: SandboxFilesResponse = await listSandboxFiles(sessionId);
			files = res.files;
			approvalState = res.approval_state;
			const decision = poller.onSuccess();
			lifecycle = decision.state;
			schedule(decision.nextDelayMs);
		} catch (e: unknown) {
			const notFound =
				(e instanceof ApiError && e.status === 404) ||
				(e instanceof Error && e.message.includes('not found'));
			const decision = notFound
				? poller.onNotFound()
				: poller.onTransientError();
			lifecycle = decision.state;
			if (decision.state === 'expired') {
				// The workspace existed and is gone: the stale list would lie.
				files = [];
			}
			schedule(decision.nextDelayMs);
		}
	}

	// -- Preview --
	async function handlePreview(path: string) {
		previewLoading = true;
		previewFile = null;
		error = '';
		try {
			previewFile = await previewSandboxFile(sessionId, path);
		} catch (e: any) {
			error = e?.message || 'Preview failed';
		} finally {
			previewLoading = false;
		}
	}

	function closePreview() {
		previewFile = null;
	}

	// -- Selection --
	function toggleSelect(path: string) {
		const next = new Set(selectedPaths);
		if (next.has(path)) {
			next.delete(path);
		} else {
			next.add(path);
		}
		selectedPaths = next;
	}

	function toggleSelectAll() {
		if (allSelected) {
			selectedPaths = new Set();
		} else {
			selectedPaths = new Set(files.map(f => f.path));
		}
	}

	// -- Approve & Download --
	async function handleApproveAndDownload() {
		if (selectedPaths.size === 0) return;
		loading = true;
		error = '';
		successMsg = '';
		try {
			const paths = Array.from(selectedPaths);

			// Step 1: Approve
			await approveSandboxFiles(sessionId, { paths });

			// Step 2: Copy out
			const result = await copyOutSandboxFiles(sessionId, { paths });

			successMsg = `${result.copied_count} file(s) copied to ${result.dest_dir}`;

			// Refresh file list to update approval badges
			await fetchFiles();
		} catch (e: any) {
			error = e?.message || 'Approve & download failed';
		} finally {
			loading = false;
		}
	}

	// -- Download single approved file --
	function handleDownloadFile(path: string) {
		const url = getSandboxDownloadUrl(sessionId, path);
		const a = document.createElement('a');
		a.href = url;
		a.download = path.split('/').pop() || path;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
	}

	// -- Reject All --
	async function handleRejectAll() {
		loading = true;
		error = '';
		successMsg = '';
		try {
			await rejectSandboxFiles(sessionId);
			approvalState = 'rejected';
			successMsg = 'All files rejected. No copy-out possible.';
			selectedPaths = new Set();
			await fetchFiles();
		} catch (e: any) {
			error = e?.message || 'Reject failed';
		} finally {
			loading = false;
		}
	}

	// -- Helpers --
	function formatSize(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
	}

	function formatTime(ts: number): string {
		if (!ts) return '';
		return new Date(ts * 1000).toLocaleTimeString();
	}

	function fileExtension(path: string): string {
		const parts = path.split('.');
		return parts.length > 1 ? parts[parts.length - 1].toLowerCase() : '';
	}
</script>

<div class="sfm-root">
	{#if !sessionId}
		<p class="sfm-empty">No active sandbox session.</p>
	{:else if lifecycle === 'expired'}
		<div class="sfm-expired">
			<span class="sfm-badge sfm-badge-expired">EXPIRED</span>
			<p>This sandbox workspace existed and has been destroyed.</p>
		</div>
	{:else if lifecycle === 'absent'}
		<div class="sfm-expired">
			<span class="sfm-badge sfm-badge-absent">NO WORKSPACE</span>
			<p>No sandbox workspace exists under this id (none was created).</p>
		</div>
	{:else}
		<!-- Header -->
		<div class="sfm-header">
			<div class="sfm-header-left">
				<h4 class="sfm-title">Sandbox Files</h4>
				<span class="sfm-session-id" title={sessionId}>
					{sessionId.slice(0, 12)}{sessionId.length > 12 ? '...' : ''}
				</span>
				<span class="sfm-badge sfm-badge-{approvalState}">{approvalState.toUpperCase()}</span>
			</div>
			<div class="sfm-header-right">
				<span class="sfm-count">{files.length} file{files.length !== 1 ? 's' : ''}</span>
			</div>
		</div>

		<!-- Messages -->
		{#if error}
			<p class="sfm-error">{error}</p>
		{/if}
		{#if successMsg}
			<p class="sfm-success">{successMsg}</p>
		{/if}

		<!-- File list -->
		{#if files.length === 0}
			<p class="sfm-empty">No files in sandbox workspace.</p>
		{:else}
			<!-- Toolbar -->
			<div class="sfm-toolbar">
				<label class="sfm-select-all">
					<input type="checkbox" checked={allSelected} on:change={toggleSelectAll} />
					Select all
				</label>
				<div class="sfm-toolbar-actions">
					<button
						class="sfm-btn sfm-btn-approve"
						disabled={selectedPaths.size === 0 || loading || approvalState === 'rejected'}
						on:click={handleApproveAndDownload}
					>
						{loading ? 'Processing...' : `Approve & Download (${selectedPaths.size})`}
					</button>
					<button
						class="sfm-btn sfm-btn-reject"
						disabled={loading || approvalState === 'rejected'}
						on:click={handleRejectAll}
					>
						Reject All
					</button>
				</div>
			</div>

			<!-- File entries -->
			<div class="sfm-file-list">
				{#each files as file (file.path)}
					<div class="sfm-file-item" class:sfm-file-approved={file.approved} class:sfm-file-selected={selectedPaths.has(file.path)}>
						<label class="sfm-file-checkbox">
							<input
								type="checkbox"
								checked={selectedPaths.has(file.path)}
								on:change={() => toggleSelect(file.path)}
								disabled={approvalState === 'rejected'}
							/>
						</label>
						<div class="sfm-file-info">
							<span class="sfm-file-path" title={file.path}>{file.path}</span>
							<span class="sfm-file-meta">
								{formatSize(file.size)}
								{#if file.modified}
									&middot; {formatTime(file.modified)}
								{/if}
							</span>
						</div>
						<div class="sfm-file-actions">
							{#if file.approved}
								<span class="sfm-badge sfm-badge-approved" title="Approved for copy-out">OK</span>
								<button class="sfm-btn-icon" title="Download" on:click={() => handleDownloadFile(file.path)}>
									&#8681;
								</button>
							{/if}
							<button class="sfm-btn-icon" title="Preview" on:click={() => handlePreview(file.path)}>
								&#128065;
							</button>
						</div>
					</div>
				{/each}
			</div>
		{/if}

		<!-- Preview modal -->
		{#if previewFile || previewLoading}
			<div class="sfm-preview-overlay" on:click={closePreview} on:keydown={e => e.key === 'Escape' && closePreview()}>
				<div class="sfm-preview-panel" on:click|stopPropagation on:keydown|stopPropagation>
					<div class="sfm-preview-header">
						<span class="sfm-preview-path">{previewFile?.path || 'Loading...'}</span>
						<button class="sfm-btn-icon" on:click={closePreview}>&#10005;</button>
					</div>
					{#if previewLoading}
						<div class="sfm-preview-loading">Loading preview...</div>
					{:else if previewFile}
						<div class="sfm-preview-meta">
							<span>{formatSize(previewFile.size)}</span>
							{#if previewFile.is_binary}
								<span class="sfm-badge sfm-badge-binary">BINARY</span>
							{/if}
							{#if previewFile.truncated}
								<span class="sfm-badge sfm-badge-truncated">TRUNCATED</span>
							{/if}
						</div>
						<pre class="sfm-preview-content">{previewFile.content}</pre>
					{/if}
				</div>
			</div>
		{/if}
	{/if}
</div>

<style>
	.sfm-root {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.sfm-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.sfm-header-left {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.sfm-title {
		margin: 0;
		font-size: 0.9rem;
		font-weight: 600;
		color: var(--oo-fg-primary);
	}

	.sfm-session-id {
		font-size: 0.7rem;
		font-family: monospace;
		color: var(--oo-fg-tertiary);
	}

	.sfm-count {
		font-size: 0.75rem;
		color: var(--oo-fg-tertiary);
	}

	/* Badges */
	.sfm-badge {
		display: inline-block;
		font-size: 0.6rem;
		font-weight: 700;
		letter-spacing: 0.05em;
		padding: 0.1rem 0.35rem;
		border-radius: 3px;
	}

	.sfm-badge-pending {
		background: var(--oo-warning-bg, rgba(234, 179, 8, 0.15));
		color: var(--oo-warning);
	}

	.sfm-badge-approved {
		background: var(--oo-success-bg, rgba(34, 197, 94, 0.15));
		color: var(--oo-success);
	}

	.sfm-badge-rejected {
		background: var(--oo-error-bg, rgba(239, 68, 68, 0.15));
		color: var(--oo-error);
	}

	.sfm-badge-expired {
		background: var(--oo-error-bg, rgba(239, 68, 68, 0.15));
		color: var(--oo-fg-tertiary);
	}

	.sfm-badge-absent {
		background: var(--oo-bg-elevated);
		color: var(--oo-fg-tertiary);
	}

	.sfm-badge-binary {
		background: var(--oo-info-bg, rgba(59, 130, 246, 0.15));
		color: var(--oo-pipe-direct);
	}

	.sfm-badge-truncated {
		background: var(--oo-warning-bg, rgba(234, 179, 8, 0.15));
		color: var(--oo-warning);
	}

	/* Toolbar */
	.sfm-toolbar {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.35rem 0;
		border-bottom: 1px solid var(--oo-bd-soft, rgba(255, 255, 255, 0.08));
	}

	.sfm-select-all {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		font-size: 0.75rem;
		color: var(--oo-fg-tertiary);
		cursor: pointer;
	}

	.sfm-select-all input { margin: 0; cursor: pointer; }

	.sfm-toolbar-actions {
		display: flex;
		gap: 0.4rem;
	}

	/* Buttons */
	.sfm-btn {
		font-size: 0.7rem;
		padding: 0.25rem 0.6rem;
		border-radius: 4px;
		cursor: pointer;
		border: 1px solid var(--oo-bd-soft, rgba(255, 255, 255, 0.15));
		background: transparent;
		transition: background 0.15s, border-color 0.15s;
	}

	.sfm-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.sfm-btn-approve {
		color: var(--oo-success);
		border-color: var(--oo-success);
	}

	.sfm-btn-approve:not(:disabled):hover {
		background: var(--oo-success-bg, rgba(34, 197, 94, 0.15));
	}

	.sfm-btn-reject {
		color: var(--oo-error);
		border-color: var(--oo-error);
	}

	.sfm-btn-reject:not(:disabled):hover {
		background: var(--oo-error-bg, rgba(239, 68, 68, 0.15));
	}

	.sfm-btn-icon {
		background: none;
		border: none;
		cursor: pointer;
		font-size: 0.85rem;
		padding: 0.15rem;
		color: var(--oo-fg-secondary);
		line-height: 1;
	}

	.sfm-btn-icon:hover { color: var(--oo-fg-primary); }

	/* File list */
	.sfm-file-list {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		max-height: 300px;
		overflow-y: auto;
		padding-right: 0.15rem;
	}

	.sfm-file-item {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		padding: 0.35rem 0.5rem;
		border-radius: 4px;
		background: var(--oo-bg-secondary, rgba(255, 255, 255, 0.03));
		border: 1px solid var(--oo-bd-soft, rgba(255, 255, 255, 0.06));
		transition: border-color 0.15s;
	}

	.sfm-file-item.sfm-file-selected {
		border-color: var(--oo-acc-400);
	}

	.sfm-file-item.sfm-file-approved {
		border-left: 3px solid var(--oo-success);
	}

	.sfm-file-checkbox { display: flex; align-items: center; }
	.sfm-file-checkbox input { margin: 0; cursor: pointer; }

	.sfm-file-info {
		flex: 1;
		display: flex;
		flex-direction: column;
		min-width: 0;
	}

	.sfm-file-path {
		font-size: 0.8rem;
		font-family: monospace;
		color: var(--oo-fg-primary);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.sfm-file-meta {
		font-size: 0.65rem;
		color: var(--oo-fg-tertiary);
	}

	.sfm-file-actions {
		display: flex;
		align-items: center;
		gap: 0.3rem;
	}

	/* Messages */
	.sfm-error {
		font-size: 0.8rem;
		color: var(--oo-error);
		margin: 0;
		padding: 0.3rem 0.5rem;
		background: var(--oo-error-bg, rgba(239, 68, 68, 0.1));
		border-radius: 4px;
	}

	.sfm-success {
		font-size: 0.8rem;
		color: var(--oo-success);
		margin: 0;
		padding: 0.3rem 0.5rem;
		background: var(--oo-success-bg, rgba(34, 197, 94, 0.1));
		border-radius: 4px;
	}

	.sfm-empty {
		font-size: 0.8rem;
		color: var(--oo-fg-tertiary);
		font-style: italic;
		margin: 0;
	}

	.sfm-expired {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.sfm-expired p {
		font-size: 0.8rem;
		color: var(--oo-fg-tertiary);
		margin: 0;
	}

	/* Preview overlay */
	.sfm-preview-overlay {
		position: fixed;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		background: var(--oo-overlay-bg, rgba(0, 0, 0, 0.6));
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 1000;
	}

	.sfm-preview-panel {
		background: var(--oo-bg-elevated);
		border: 1px solid var(--oo-bd-default, rgba(255, 255, 255, 0.12));
		border-radius: 8px;
		width: min(90vw, 700px);
		max-height: 80vh;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.sfm-preview-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.5rem 0.75rem;
		border-bottom: 1px solid var(--oo-bd-soft, rgba(255, 255, 255, 0.08));
	}

	.sfm-preview-path {
		font-family: monospace;
		font-size: 0.8rem;
		color: var(--oo-fg-primary);
	}

	.sfm-preview-meta {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.3rem 0.75rem;
		font-size: 0.7rem;
		color: var(--oo-fg-tertiary);
		border-bottom: 1px solid var(--oo-bd-soft, rgba(255, 255, 255, 0.05));
	}

	.sfm-preview-content {
		padding: 0.75rem;
		margin: 0;
		font-size: 0.75rem;
		font-family: monospace;
		color: var(--oo-fg-secondary);
		white-space: pre-wrap;
		word-break: break-all;
		overflow-y: auto;
		flex: 1;
		max-height: 60vh;
	}

	.sfm-preview-loading {
		padding: 2rem;
		text-align: center;
		font-size: 0.85rem;
		color: var(--oo-fg-tertiary);
	}
</style>
