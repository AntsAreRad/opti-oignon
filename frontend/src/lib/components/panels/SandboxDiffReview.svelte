<!--
  SandboxDiffReview.svelte (S212, Sandbox Workspace cycle, Bloc 3)
  The diff review + apply flow (spec sections 6 and 10): load the live
  workspace diff against the recorded baseline (added/modified/deleted,
  hash-driven), preview files, approve changes per file, confirm deletions
  in their OWN explicit step (visually distinct, never inside an
  approve-all), and apply only the approved set back to the host. The apply
  echoes the reviewed diff_hash; if the workspace changed since the review
  the server answers 409 and the diff must be re-run. Upload-only
  workspaces (no cloned root) require an explicit target directory under
  the host share-root allowlist. Cap and refusal errors are surfaced
  honestly, per file. An EXPLICIT user action: the model can trigger
  neither the review nor the apply (S73/S74). Design-system tokens only
  (--oo-*); lucide icons through Icon. Registered in
  FRONTEND_REDESIGN_SPEC.md.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { Button, Icon, InlineError, Input } from '$lib/ds';
	import {
		getDiff,
		confirmDeletions,
		applyChanges,
		approveSandboxFiles,
		previewSandboxFile
	} from '$lib/api/sandbox';
	import type {
		SandboxDiffResponse,
		SandboxDiffEntry,
		SandboxApplyResponse
	} from '$lib/types';

	export let sessionId: string | null = null;
	export let disabled = false;

	const dispatch = createEventDispatcher<{
		applied: { sessionId: string; applied: number; deleted: number };
	}>();

	let diff: SandboxDiffResponse | null = null;
	let loading = false;
	let approving = false;
	let confirming = false;
	let applying = false;
	let error: string | null = null;
	let applyResult: SandboxApplyResponse | null = null;
	let targetDir = '';

	let selectedWrites: Set<string> = new Set();
	let selectedDeletions: Set<string> = new Set();

	let previewPath: string | null = null;
	let previewContent = '';
	let previewBinary = false;
	let previewTruncated = false;

	$: addedEntries = diff ? diff.entries.filter((e) => e.kind === 'added') : [];
	$: modifiedEntries = diff
		? diff.entries.filter((e) => e.kind === 'modified')
		: [];
	$: deletedEntries = diff
		? diff.entries.filter((e) => e.kind === 'deleted')
		: [];
	$: approvedSet = new Set(diff ? diff.approved_paths : []);
	$: confirmedSet = new Set(diff ? diff.confirmed_deletions : []);
	$: needsExplicitTarget = diff !== null && !diff.cloned_root;
	$: applyReady =
		diff !== null &&
		(diff.approved_paths.length > 0 || diff.confirmed_deletions.length > 0) &&
		(!needsExplicitTarget || targetDir.trim() !== '');

	async function loadDiff() {
		if (!sessionId) return;
		loading = true;
		error = null;
		applyResult = null;
		previewPath = null;
		try {
			diff = await getDiff(sessionId);
			selectedWrites = new Set();
			selectedDeletions = new Set();
		} catch (e) {
			diff = null;
			error = e instanceof Error ? e.message : 'Diff failed';
		} finally {
			loading = false;
		}
	}

	function toggleWrite(path: string) {
		if (selectedWrites.has(path)) {
			selectedWrites.delete(path);
		} else {
			selectedWrites.add(path);
		}
		selectedWrites = new Set(selectedWrites);
	}

	function toggleDeletion(path: string) {
		if (selectedDeletions.has(path)) {
			selectedDeletions.delete(path);
		} else {
			selectedDeletions.add(path);
		}
		selectedDeletions = new Set(selectedDeletions);
	}

	async function handleApprove() {
		if (!sessionId || selectedWrites.size === 0) return;
		approving = true;
		error = null;
		try {
			await approveSandboxFiles(sessionId, {
				paths: Array.from(selectedWrites)
			});
			await loadDiff();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Approval failed';
		} finally {
			approving = false;
		}
	}

	async function handleConfirmDeletions() {
		if (!sessionId || selectedDeletions.size === 0) return;
		confirming = true;
		error = null;
		try {
			const result = await confirmDeletions(
				sessionId,
				Array.from(selectedDeletions)
			);
			if (result.refused.length > 0) {
				error = result.refused
					.map((r) => `${r.path}: ${r.reason}`)
					.join('; ');
			}
			await loadDiff();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Deletion confirmation failed';
		} finally {
			confirming = false;
		}
	}

	async function handleApply() {
		if (!sessionId || !diff || !applyReady) return;
		applying = true;
		error = null;
		applyResult = null;
		try {
			const result = await applyChanges(sessionId, {
				diff_hash: diff.diff_hash,
				target_dir: needsExplicitTarget ? targetDir.trim() : undefined
			});
			applyResult = result;
			dispatch('applied', {
				sessionId,
				applied: result.applied.length,
				deleted: result.deleted.length
			});
			await loadDiff();
			applyResult = result;
		} catch (e) {
			const message = e instanceof Error ? e.message : 'Apply failed';
			error = message.includes('changed since')
				? `${message} Reload the diff and review again.`
				: message;
		} finally {
			applying = false;
		}
	}

	async function togglePreview(entry: SandboxDiffEntry) {
		if (!sessionId) return;
		if (previewPath === entry.path) {
			previewPath = null;
			return;
		}
		error = null;
		try {
			const result = await previewSandboxFile(sessionId, entry.path);
			previewPath = entry.path;
			previewContent = result.content;
			previewBinary = result.is_binary;
			previewTruncated = result.truncated;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Preview failed';
		}
	}

	function formatBytes(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
	}
</script>

<div class="diff-review">
	<div class="review-bar">
		<Button
			variant="secondary"
			size="sm"
			iconLeft="file-diff"
			loading={loading}
			disabled={disabled || !sessionId}
			ariaLabel="Compute the workspace diff against its baseline"
			on:click={loadDiff}
		>
			{diff === null ? 'Review changes' : 'Reload diff'}
		</Button>
		{#if diff}
			<span class="review-summary">
				{addedEntries.length} added, {modifiedEntries.length} modified,
				{deletedEntries.length} deleted, {diff.unchanged} unchanged
			</span>
		{/if}
	</div>

	{#if !sessionId}
		<p class="review-note" role="note">Select a workspace to review it.</p>
	{/if}

	{#if error}
		<InlineError message={error} />
	{/if}

	{#if diff}
		{#if !diff.baseline_present}
			<p class="review-warn" role="note">
				No baseline is recorded for this workspace (upload-only or after a
				restart): everything is classified as added and there is no implicit
				write-back target. Review everything and choose an explicit target
				directory.
			</p>
		{/if}
		{#if diff.skipped_symlinks > 0 || diff.skipped_special > 0}
			<p class="review-note" role="note">
				Skipped during the walk (never followed, never applied):
				{diff.skipped_symlinks} symlink(s), {diff.skipped_special} special
				file(s).
			</p>
		{/if}

		{#if diff.entries.length === 0}
			<p class="review-note" role="note">
				The workspace matches its baseline; nothing to apply.
			</p>
		{/if}

		{#if addedEntries.length > 0 || modifiedEntries.length > 0}
			<ul class="review-list" aria-label="Added and modified files">
				{#each [...addedEntries, ...modifiedEntries] as entry (entry.path)}
					<li class="review-row">
						<label class="row-main">
							<input
								type="checkbox"
								checked={selectedWrites.has(entry.path)}
								disabled={approving || applying}
								on:change={() => toggleWrite(entry.path)}
							/>
							<span class="row-kind kind-{entry.kind}">{entry.kind}</span>
							<span class="row-path" title={entry.path}>{entry.path}</span>
							<span class="row-size">{formatBytes(entry.size)}</span>
							{#if approvedSet.has(entry.path)}
								<span class="row-approved">approved</span>
							{/if}
						</label>
						<Button
							variant="ghost"
							size="sm"
							iconOnly="eye"
							ariaLabel={`Preview ${entry.path}`}
							on:click={() => togglePreview(entry)}
						/>
						{#if previewPath === entry.path}
							<pre class="row-preview">{previewBinary
									? `binary (hex): ${previewContent}`
									: previewContent}{previewTruncated
									? '\n[truncated preview]'
									: ''}</pre>
						{/if}
					</li>
				{/each}
			</ul>
			<Button
				variant="secondary"
				size="sm"
				iconLeft="check"
				loading={approving}
				disabled={approving || applying || selectedWrites.size === 0}
				ariaLabel="Approve the selected files for apply"
				on:click={handleApprove}
			>
				Approve selected ({selectedWrites.size})
			</Button>
		{/if}

		{#if deletedEntries.length > 0}
			<div class="deletion-block">
				<p class="deletion-title">
					<Icon name="trash-2" />
					Deletions require their own confirmation
				</p>
				<p class="review-warn" role="note">
					Applying a confirmed deletion removes the file from your disk.
					Deletions are never included in an approve-all.
				</p>
				<ul class="review-list" aria-label="Deleted files">
					{#each deletedEntries as entry (entry.path)}
						<li class="review-row">
							<label class="row-main">
								<input
									type="checkbox"
									checked={selectedDeletions.has(entry.path)}
									disabled={confirming || applying}
									on:change={() => toggleDeletion(entry.path)}
								/>
								<span class="row-kind kind-deleted">deleted</span>
								<span class="row-path" title={entry.path}>{entry.path}</span>
								{#if confirmedSet.has(entry.path)}
									<span class="row-confirmed">confirmed</span>
								{/if}
							</label>
						</li>
					{/each}
				</ul>
				<Button
					variant="danger"
					size="sm"
					iconLeft="trash-2"
					loading={confirming}
					disabled={confirming || applying || selectedDeletions.size === 0}
					ariaLabel="Confirm the selected deletions for apply"
					on:click={handleConfirmDeletions}
				>
					Confirm selected deletions ({selectedDeletions.size})
				</Button>
			</div>
		{/if}

		{#if needsExplicitTarget}
			<Input
				label="Target directory"
				placeholder="Explicit host directory (inside the share roots)"
				bind:value={targetDir}
				disabled={applying}
			/>
		{:else if diff.cloned_root}
			<p class="review-note" role="note">
				Apply target: the originally-cloned root {diff.cloned_root}
				{#if diff.cloned_mount}(workspace subtree {diff.cloned_mount}/
				round-trips onto it; other paths need an explicit target){/if}
			</p>
		{/if}

		<Button
			variant="primary"
			iconLeft="upload"
			loading={applying}
			disabled={applying || disabled || !applyReady}
			ariaLabel="Apply the approved changes to the host"
			on:click={handleApply}
		>
			Apply approved changes
		</Button>
		{#if !applyReady && diff.entries.length > 0}
			<p class="review-note" role="note">
				{needsExplicitTarget && targetDir.trim() === ''
					? 'Choose an explicit target directory before applying.'
					: 'Approve files or confirm deletions before applying.'}
			</p>
		{/if}
	{/if}

	{#if applyResult}
		<div class="apply-result" role="status">
			<p class="apply-line">
				Applied to {applyResult.target}: {applyResult.applied.length}
				write(s), {applyResult.deleted.length} deletion(s);
				{applyResult.skipped_unapproved} unapproved and
				{applyResult.skipped_unconfirmed} unconfirmed change(s) skipped.
			</p>
			{#if applyResult.refused.length > 0}
				<ul class="apply-refused" aria-label="Refused apply paths">
					{#each applyResult.refused as entry (entry.path)}
						<li>{entry.path}: {entry.error}</li>
					{/each}
				</ul>
			{/if}
		</div>
	{/if}
</div>

<style>
	.diff-review {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
	}
	.review-bar {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex-wrap: wrap;
	}
	.review-summary {
		font-size: 0.75rem;
		color: var(--oo-fg-secondary);
	}
	.review-list {
		margin: 0;
		padding: 0;
		list-style: none;
		max-height: 14rem;
		overflow-y: auto;
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md, 8px);
	}
	.review-row {
		border-bottom: 1px solid var(--oo-bd-subtle);
		padding: 0.25rem 0.5rem;
	}
	.review-row:last-child {
		border-bottom: none;
	}
	.row-main {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.8rem;
		color: var(--oo-fg-primary);
		cursor: pointer;
	}
	.row-kind {
		font-size: 0.7rem;
		text-transform: uppercase;
	}
	.kind-added {
		color: var(--oo-success);
	}
	.kind-modified {
		color: var(--oo-warning);
	}
	.kind-deleted {
		color: var(--oo-error);
	}
	.row-path {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.row-size {
		font-size: 0.7rem;
		color: var(--oo-fg-tertiary);
	}
	.row-approved,
	.row-confirmed {
		font-size: 0.7rem;
		color: var(--oo-success);
	}
	.row-preview {
		margin: 0.25rem 0 0;
		padding: 0.4rem;
		max-height: 8rem;
		overflow: auto;
		font-size: 0.7rem;
		white-space: pre-wrap;
		word-break: break-all;
		background: var(--oo-bg-elevated);
		border-radius: var(--oo-radius-sm, 4px);
		color: var(--oo-fg-secondary);
	}
	.deletion-block {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
		padding: 0.5rem;
		border: 1px solid var(--oo-error);
		border-radius: var(--oo-radius-md, 8px);
	}
	.deletion-title {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin: 0;
		font-size: 0.8rem;
		font-weight: 600;
		color: var(--oo-error);
	}
	.review-note {
		margin: 0;
		font-size: 0.75rem;
		color: var(--oo-fg-tertiary);
	}
	.review-warn {
		margin: 0;
		font-size: 0.75rem;
		color: var(--oo-warning);
	}
	.apply-result {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}
	.apply-line {
		margin: 0;
		font-size: 0.75rem;
		color: var(--oo-success);
	}
	.apply-refused {
		margin: 0;
		padding-left: 1rem;
		font-size: 0.75rem;
		color: var(--oo-error);
	}
</style>
