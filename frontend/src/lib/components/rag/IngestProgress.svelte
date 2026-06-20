<!--
  IngestProgress.svelte -- S120 RAG Batch Ingestion Progress Tracker.

  Polls GET /api/rag/ingest/jobs/{job_id} every 2s while running.
  Shows per-file status indicators, overall progress bar, cancel button.
  Auto-stops polling when job completes/fails/cancelled.
  Emits 'completed' event when job finishes.
-->
<script lang="ts">
	import { onMount, onDestroy, createEventDispatcher } from 'svelte';
	import { getIngestJob, deleteIngestJob } from '$lib/api/rag';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import type { RAGIngestJob, RAGIngestFileStatus } from '$lib/types';

	/** The job ID to track. */
	export let jobId: string;

	/** Optional: initial job data to avoid first fetch delay. */
	export let initialJob: RAGIngestJob | null = null;

	const dispatch = createEventDispatcher<{
		completed: RAGIngestJob;
		cancelled: string;
	}>();

	const POLL_INTERVAL_MS = 2000;
	const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled']);

	let job: RAGIngestJob | null = initialJob;
	let pollTimer: ReturnType<typeof setInterval> | null = null;
	let cancelling = false;
	let error: string | null = null;

	// Computed
	$: isTerminal = job ? TERMINAL_STATUSES.has(job.status) : false;
	$: progressPct = job ? Math.round(job.progress * 100) : 0;
	$: statusLabel = job ? formatStatus(job.status) : 'Loading...';

	function formatStatus(status: string): string {
		switch (status) {
			case 'pending': return 'Pending';
			case 'running': return 'Processing';
			case 'completed': return 'Completed';
			case 'failed': return 'Failed';
			case 'cancelled': return 'Cancelled';
			default: return status;
		}
	}

	function statusColor(status: string): string {
		switch (status) {
			case 'pending': return 'var(--oo-fg-muted)';
			case 'running': return 'var(--oo-warning)';
			case 'completed': return 'var(--oo-success)';
			case 'failed': return 'var(--oo-error)';
			case 'cancelled': return 'var(--oo-fg-muted)';
			default: return 'var(--oo-fg-muted)';
		}
	}

	function fileStatusColor(status: string): string {
		switch (status) {
			case 'queued': return 'var(--oo-fg-muted)';
			case 'processing': return 'var(--oo-warning)';
			case 'done': return 'var(--oo-success)';
			case 'error': return 'var(--oo-error)';
			case 'skipped': return 'var(--oo-fg-muted)';
			default: return 'var(--oo-fg-muted)';
		}
	}

	function fileStatusIcon(status: string): string {
		switch (status) {
			case 'queued': return '\u25CB';      // open circle
			case 'processing': return '\u25D4';   // circle with upper right quadrant
			case 'done': return '\u2713';         // check mark
			case 'error': return '\u2717';        // cross mark
			case 'skipped': return '\u2014';      // em dash
			default: return '?';
		}
	}

	function formatSize(bytes: number): string {
		if (bytes < 1024) return `${bytes}B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
	}

	function formatDuration(startSec: number | null, endSec: number | null): string {
		if (!startSec) return '-';
		const end = endSec || Date.now() / 1000;
		const elapsed = Math.max(0, end - startSec);
		if (elapsed < 60) return `${elapsed.toFixed(1)}s`;
		return `${Math.floor(elapsed / 60)}m ${Math.round(elapsed % 60)}s`;
	}

	async function fetchJob() {
		try {
			job = await getIngestJob(jobId);
			error = null;

			if (isTerminal) {
				stopPolling();
				if (job.status === 'completed' || job.status === 'failed') {
					dispatch('completed', job);
				}
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to fetch job status';
		}
	}

	function startPolling() {
		if (pollTimer) return;
		pollTimer = setInterval(fetchJob, POLL_INTERVAL_MS);
	}

	function stopPolling() {
		if (pollTimer) {
			clearInterval(pollTimer);
			pollTimer = null;
		}
	}

	async function handleCancel() {
		if (cancelling) return;
		cancelling = true;
		try {
			await deleteIngestJob(jobId);
			toastSuccess('Ingestion job cancelled');
			dispatch('cancelled', jobId);
			stopPolling();
			// Fetch one more time to get final status
			await fetchJob();
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Failed to cancel job');
		} finally {
			cancelling = false;
		}
	}

	onMount(() => {
		fetchJob();
		if (!isTerminal) {
			startPolling();
		}
	});

	onDestroy(() => {
		stopPolling();
	});
</script>

<div class="rounded-lg overflow-hidden" style="border: 1px solid var(--oo-bd-subtle);" data-testid="ingest-progress">
	<!-- Header with status -->
	<div class="flex items-center justify-between px-4 py-3"
		style="background-color: var(--oo-bg-overlay);">
		<div class="flex items-center gap-3">
			<span class="text-sm font-medium" style="color: {job ? statusColor(job.status) : 'var(--oo-fg-muted)'};"
				data-testid="ingest-progress-status">
				{statusLabel}
			</span>
			{#if job}
				<span class="text-xs" style="color: var(--oo-fg-muted);">
					{job.completed_files + job.failed_files + job.skipped_files} / {job.total_files} files
				</span>
				{#if job.started_at}
					<span class="text-xs" style="color: var(--oo-fg-muted);">
						{formatDuration(job.started_at, job.completed_at)}
					</span>
				{/if}
			{/if}
		</div>
		<div class="flex items-center gap-2">
			{#if job && job.total_chunks > 0}
				<span class="text-xs" style="color: var(--oo-fg-muted);">
					{job.total_chunks} chunks
				</span>
			{/if}
			{#if job && !isTerminal}
				<button
					on:click={handleCancel}
					disabled={cancelling}
					class="px-3 py-1 rounded text-xs font-medium disabled:opacity-50"
					style="color: var(--oo-error); border: 1px solid var(--oo-error-bd);"
					data-testid="ingest-cancel-btn"
				>
					{cancelling ? 'Cancelling...' : 'Cancel'}
				</button>
			{/if}
		</div>
	</div>

	<!-- Progress bar -->
	{#if job}
		<div class="px-4 py-2" style="background-color: var(--oo-bg-elevated);">
			<div class="w-full rounded-full h-2 overflow-hidden"
				style="background-color: var(--oo-bg-overlay);">
				<div
					class="h-full rounded-full transition-all duration-500"
					style="width: {progressPct}%;
						background-color: {job.status === 'failed' ? 'var(--oo-error)' : job.status === 'completed' ? 'var(--oo-success)' : 'var(--oo-acc-600)'};"
					data-testid="ingest-progress-bar"
				></div>
			</div>
			<div class="flex items-center justify-between mt-1">
				<span class="text-xs" style="color: var(--oo-fg-muted);">
					{progressPct}%
				</span>
				{#if job.failed_files > 0}
					<span class="text-xs" style="color: var(--oo-error);">
						{job.failed_files} failed
					</span>
				{/if}
				{#if job.skipped_files > 0}
					<span class="text-xs" style="color: var(--oo-fg-muted);">
						{job.skipped_files} skipped
					</span>
				{/if}
			</div>
		</div>
	{/if}

	<!-- Error message -->
	{#if error}
		<div class="px-4 py-2 text-xs" style="color: var(--oo-error); background-color: var(--oo-bg-elevated);">
			{error}
		</div>
	{/if}
	{#if job?.error_message}
		<div class="px-4 py-2 text-xs" style="color: var(--oo-error); background-color: var(--oo-bg-elevated);">
			{job.error_message}
		</div>
	{/if}

	<!-- Per-file status list -->
	{#if job && job.files.length > 0}
		<div class="divide-y" style="border-color: var(--oo-bd-subtle);" data-testid="ingest-file-list">
			{#each job.files as file}
				<div class="flex items-center gap-3 px-4 py-2"
					style="background-color: var(--oo-bg-elevated);">
					<!-- Status icon -->
					<span class="text-sm font-mono w-5 text-center shrink-0"
						style="color: {fileStatusColor(file.status)};"
						data-testid="ingest-file-status-icon"
						title={file.status}>
						{fileStatusIcon(file.status)}
					</span>

					<!-- Filename -->
					<span class="text-sm truncate min-w-0 flex-1"
						style="color: var(--oo-fg-primary);"
						data-testid="ingest-file-name">
						{file.filename}
					</span>

					<!-- Size -->
					<span class="text-xs shrink-0" style="color: var(--oo-fg-muted);">
						{formatSize(file.file_size)}
					</span>

					<!-- Chunks (if done) -->
					{#if file.status === 'done' && file.chunk_count > 0}
						<span class="text-xs shrink-0" style="color: var(--oo-success);">
							{file.chunk_count} chunks
						</span>
					{/if}

					<!-- Error (if error) -->
					{#if file.status === 'error' && file.error_message}
						<span class="text-xs shrink-0 truncate max-w-48" style="color: var(--oo-error);"
							title={file.error_message}>
							{file.error_message}
						</span>
					{/if}

					<!-- Processing spinner -->
					{#if file.status === 'processing'}
						<span class="text-xs shrink-0 animate-pulse" style="color: var(--oo-warning);">
							processing...
						</span>
					{/if}
				</div>
			{/each}
		</div>
	{/if}
</div>
