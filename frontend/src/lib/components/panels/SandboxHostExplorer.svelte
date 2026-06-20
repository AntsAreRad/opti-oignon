<!--
  SandboxHostExplorer.svelte (S211, Sandbox Workspace cycle, Bloc 2)
  The host explorer + clone flow (spec 5.2): browse the allowlisted share
  roots (the server confines every request -- outside the roots it answers
  403 before any existence check), pick a directory, and clone it into the
  selected workspace. The clone is symlink-safe (links skipped and counted,
  targets never exposed), skips device/special files, and is capped (bytes
  and file count, plus the per-workspace quota) by an exact pre-walk that
  refuses with 413 before any copy; a destination collision answers 409.
  Cap errors and skip counts are surfaced honestly. Cloning records the
  section 6.1 baseline manifest server-side (Bloc 3's diff consumes it).
  An EXPLICIT user action: the model can trigger neither a browse nor a
  clone (S73/S74). Hidden entries are shown dimmed rather than hidden --
  a clone copies them, so hiding them would lie about what is shared.
  Design-system tokens only (--oo-*); lucide icons through Icon.
  Registered in FRONTEND_REDESIGN_SPEC.md.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { Button, Icon, InlineError } from '$lib/ds';
	import { browseHost, cloneDirectory } from '$lib/api/sandbox';
	import type { HostBrowseResponse, SandboxCloneResponse } from '$lib/types';

	export let sessionId: string | null = null;
	export let disabled = false;

	const dispatch = createEventDispatcher<{
		cloned: { sessionId: string; dest: string; files: number };
	}>();

	let listing: HostBrowseResponse | null = null;
	let loading = false;
	let cloning = false;
	let error: string | null = null;
	let cloneResult: SandboxCloneResponse | null = null;

	async function open(path?: string) {
		loading = true;
		error = null;
		try {
			listing = await browseHost(path);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Browse failed';
		} finally {
			loading = false;
		}
	}

	function enter(name: string) {
		if (!listing) return;
		const next = listing.path === '' ? name : `${listing.path}/${name}`;
		void open(next);
	}

	function upOne() {
		if (!listing || listing.path === '') return;
		const isRoot = listing.roots.includes(listing.path);
		if (isRoot) {
			void open();
			return;
		}
		const parent = listing.path.split('/').slice(0, -1).join('/') || '/';
		void open(parent);
	}

	async function handleClone() {
		if (!sessionId || !listing || listing.path === '' || disabled) return;
		cloning = true;
		error = null;
		cloneResult = null;
		try {
			const result = await cloneDirectory(sessionId, { src_path: listing.path });
			cloneResult = result;
			dispatch('cloned', {
				sessionId,
				dest: result.dest,
				files: result.copied_files
			});
		} catch (e) {
			error = e instanceof Error ? e.message : 'Clone failed';
		} finally {
			cloning = false;
		}
	}

	function formatBytes(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
		if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
		return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
	}
</script>

<div class="host-explorer">
	{#if listing === null}
		<Button
			variant="secondary"
			size="sm"
			iconLeft="folder-open"
			loading={loading}
			{disabled}
			ariaLabel="Browse the allowlisted host folders"
			on:click={() => open()}
		>
			Browse host folders
		</Button>
		<p class="explorer-note" role="note">
			Browsing is confined to the configured share roots; cloning a folder
			copies it into the workspace (symlinks and special files are skipped).
		</p>
	{:else}
		<div class="explorer-bar">
			<Button
				variant="ghost"
				size="sm"
				iconOnly="arrow-up"
				ariaLabel="Go up one directory"
				disabled={loading || listing.path === ''}
				on:click={upOne}
			/>
			<span class="explorer-path" title={listing.path || 'Share roots'}>
				{listing.path === '' ? 'Share roots' : listing.path}
			</span>
			<Button
				variant="primary"
				size="sm"
				iconLeft="copy"
				loading={cloning}
				disabled={cloning || loading || disabled || !sessionId || listing.path === ''}
				ariaLabel="Clone this directory into the selected workspace"
				on:click={handleClone}
			>
				Clone here
			</Button>
		</div>

		{#if !sessionId}
			<p class="explorer-note" role="note">
				Select a workspace to enable cloning.
			</p>
		{/if}

		{#if error}
			<InlineError message={error} />
		{/if}

		{#if listing.entries.length === 0}
			<p class="explorer-note" role="note">
				{listing.path === ''
					? 'No share roots are configured (host_share_roots).'
					: 'This directory is empty.'}
			</p>
		{:else}
			<ul class="explorer-list" aria-label="Directory entries">
				{#each listing.entries as entry (entry.name)}
					<li class="explorer-row" class:row-hidden={entry.hidden}>
						{#if entry.type === 'dir'}
							<button
								type="button"
								class="explorer-dir"
								disabled={loading}
								on:click={() => enter(entry.name)}
							>
								<Icon name="folder" />
								<span class="entry-name">{entry.name}</span>
							</button>
						{:else}
							<span class="explorer-file">
								<Icon
									name={entry.type === 'symlink' ? 'link' : 'file'}
								/>
								<span class="entry-name">{entry.name}</span>
								{#if entry.type === 'file'}
									<span class="entry-size">{formatBytes(entry.size)}</span>
								{:else}
									<span class="entry-size">{entry.type}</span>
								{/if}
							</span>
						{/if}
					</li>
				{/each}
			</ul>
		{/if}

		{#if cloneResult}
			<p class="clone-result" role="status">
				Cloned into {cloneResult.dest}: {cloneResult.copied_files} file(s),
				{formatBytes(cloneResult.copied_bytes)}; skipped
				{cloneResult.skipped_symlinks} symlink(s) and
				{cloneResult.skipped_special} special file(s). Baseline manifest:
				{cloneResult.manifest_files} file(s).
			</p>
		{/if}
	{/if}
</div>

<style>
	.host-explorer {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
	}
	.explorer-bar {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	.explorer-path {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: 0.75rem;
		color: var(--oo-fg-secondary);
	}
	.explorer-list {
		margin: 0;
		padding: 0;
		list-style: none;
		max-height: 14rem;
		overflow-y: auto;
		border: 1px solid var(--oo-bd-default);
		border-radius: var(--oo-radius-md, 8px);
	}
	.explorer-row {
		border-bottom: 1px solid var(--oo-bd-subtle);
	}
	.explorer-row:last-child {
		border-bottom: none;
	}
	.row-hidden {
		opacity: 0.55;
	}
	.explorer-dir,
	.explorer-file {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		width: 100%;
		padding: 0.35rem 0.5rem;
		font-size: 0.8rem;
		color: var(--oo-fg-primary);
	}
	.explorer-dir {
		background: none;
		border: none;
		text-align: left;
		cursor: pointer;
	}
	.explorer-dir:hover {
		background: var(--oo-bg-elevated);
	}
	.explorer-dir:focus-visible {
		outline: 2px solid var(--oo-acc-500);
		outline-offset: -2px;
	}
	.entry-name {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.entry-size {
		font-size: 0.7rem;
		color: var(--oo-fg-tertiary);
	}
	.explorer-note {
		margin: 0;
		font-size: 0.75rem;
		color: var(--oo-fg-tertiary);
	}
	.clone-result {
		margin: 0;
		font-size: 0.75rem;
		color: var(--oo-success);
	}
</style>
