<!--
  FolderScan.svelte -- RAG Folder Scan & Ingest UI.

  Text input for local directory path, recursive toggle,
  collection selector, scan+ingest button.
  Reuses IngestProgress for job tracking after submission.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { listCollections, ingestFolder } from '$lib/api/rag';
	import { toastSuccess, toastError } from '$lib/stores/notifications';
	import IngestProgress from './IngestProgress.svelte';
	import type { RAGCollection, RAGIngestJob } from '$lib/types';

	const dispatch = createEventDispatcher<{
		jobStarted: RAGIngestJob;
	}>();

	// State
	let collections: RAGCollection[] = [];
	let selectedCollection = 'default';
	let directoryPath = '';
	let recursive = true;
	let scanning = false;

	// Active job tracking
	let activeJob: RAGIngestJob | null = null;

	// Load collections on init
	loadCollections();

	async function loadCollections() {
		try {
			const resp = await listCollections();
			collections = resp.collections;
		} catch { /* ignore */ }
	}

	async function handleScan() {
		const dir = directoryPath.trim();
		if (!dir) return;

		scanning = true;
		try {
			const job = await ingestFolder({
				directory: dir,
				collection: selectedCollection,
				recursive,
			});
			toastSuccess(`Folder scan started: ${job.total_files} files found`);
			activeJob = job;
			dispatch('jobStarted', job);
		} catch (e) {
			toastError(e instanceof Error ? e.message : 'Folder scan failed');
		} finally {
			scanning = false;
		}
	}

	function handleJobCompleted(event: CustomEvent<RAGIngestJob>) {
		const job = event.detail;
		if (job.status === 'completed') {
			toastSuccess(`Folder ingestion complete: ${job.total_chunks} chunks from ${job.completed_files} files`);
		}
	}

	function handleJobCancelled() {
		activeJob = null;
	}

	function clearJob() {
		activeJob = null;
	}
</script>

<div class="space-y-4" data-testid="folder-scan">
	<!-- Directory path input -->
	<div class="space-y-2">
		<label class="text-sm font-medium" style="color: var(--oo-fg-secondary);">
			Local directory path
		</label>
		<input
			type="text"
			bind:value={directoryPath}
			placeholder="/home/user/documents/research"
			class="w-full px-3 py-2 rounded-lg text-sm font-mono"
			style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
			on:keydown={(e) => { if (e.key === 'Enter') handleScan(); }}
			disabled={scanning}
			data-testid="folder-path-input"
		/>
		<p class="text-xs" style="color: var(--oo-fg-muted);">
			Absolute path to a directory on the server. Supported files will be discovered and ingested.
		</p>
	</div>

	<!-- Options row -->
	<div class="flex items-center gap-4 flex-wrap">
		<!-- Recursive toggle -->
		<label class="flex items-center gap-2 cursor-pointer">
			<input
				type="checkbox"
				bind:checked={recursive}
				class="cursor-pointer"
				data-testid="folder-recursive-toggle"
			/>
			<span class="text-sm" style="color: var(--oo-fg-secondary);">Recursive scan</span>
		</label>

		<!-- Collection selector -->
		<div class="flex items-center gap-2">
			<label class="text-sm" style="color: var(--oo-fg-secondary);">Collection:</label>
			<select
				bind:value={selectedCollection}
				class="px-3 py-1.5 rounded-lg text-sm"
				style="background-color: var(--oo-input-bg); border: 1px solid var(--oo-input-bd); color: var(--oo-fg-primary);"
				data-testid="folder-collection-select"
			>
				<option value="default">default</option>
				{#each collections.filter(c => c.name !== 'default') as coll}
					<option value={coll.name}>{coll.name}</option>
				{/each}
			</select>
		</div>
	</div>

	<!-- Scan button -->
	<div class="flex items-center gap-3">
		<button
			on:click={handleScan}
			disabled={scanning || !directoryPath.trim()}
			class="px-5 py-2.5 rounded-lg text-sm font-medium disabled:opacity-50 transition-opacity"
			style="background-color: var(--oo-acc-600); color: var(--oo-acc-50);"
			data-testid="folder-scan-btn"
		>
			{scanning ? 'Scanning...' : 'Scan & Ingest'}
		</button>

		{#if activeJob && (activeJob.status === 'completed' || activeJob.status === 'failed' || activeJob.status === 'cancelled')}
			<button
				on:click={clearJob}
				class="text-xs px-3 py-1.5 rounded"
				style="color: var(--oo-fg-muted); border: 1px solid var(--oo-bd-subtle);"
				data-testid="folder-clear-job-btn"
			>
				Clear
			</button>
		{/if}
	</div>

	<!-- Progress tracker (reuses IngestProgress) -->
	{#if activeJob}
		<IngestProgress
			jobId={activeJob.job_id}
			initialJob={activeJob}
			on:completed={handleJobCompleted}
			on:cancelled={handleJobCancelled}
		/>
	{/if}
</div>
