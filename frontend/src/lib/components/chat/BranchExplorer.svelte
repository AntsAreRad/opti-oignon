<!--
  BranchExplorer.svelte
  Branch management panel for conversation exploration.
  Shows visual branch tree (collapsible, recursive), switcher, fork controls,
  comparison view, and merge.
  Replaced flat list with recursive BranchTreeNodeItem component.
-->
<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import type { Branch, BranchTreeNode, BranchComparison } from '$lib/types';
	import {
		listBranches,
		forkBranch,
		deleteBranch,
		updateBranch,
		compareBranches,
		mergeBranches,
		getBranchTree,
	} from '$lib/api/branches';
	import BranchTreeNodeItem from './BranchTreeNodeItem.svelte';

	export let conversationId: string;
	export let currentMessageId: number | null = null;

	const dispatch = createEventDispatcher<{
		switchBranch: { branchId: string | null };
		fork: { branchId: string };
	}>();

	let branches: Branch[] = [];
	let tree: BranchTreeNode | null = null;
	let activeBranchId: string | null = null;
	let loading = false;
	let error: string | null = null;

	// UI state
	let showTree = false;
	let showCompare = false;
	let comparison: BranchComparison | null = null;
	let compareAId: string | null = null;
	let compareBId: string | null = null;
	let comparing = false;

	// Rename state
	let renamingId: string | null = null;
	let renameValue = '';

	// Fork state
	let forking = false;
	let forkName = '';

	$: hasBranches = branches.length > 0;

	async function loadBranches() {
		if (!conversationId) return;
		loading = true;
		error = null;
		try {
			branches = await listBranches(conversationId);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load branches';
		} finally {
			loading = false;
		}
	}

	async function loadTree() {
		if (!conversationId) return;
		try {
			tree = await getBranchTree(conversationId);
		} catch (e) {
			tree = null;
		}
	}

	async function handleFork() {
		if (!conversationId || currentMessageId == null) return;
		forking = true;
		error = null;
		try {
			const branch = await forkBranch({
				conversation_id: conversationId,
				fork_message_id: currentMessageId,
				name: forkName.trim() || undefined,
				parent_branch_id: activeBranchId ?? undefined,
			});
			forkName = '';
			await loadBranches();
			await loadTree();
			dispatch('fork', { branchId: branch.branch_id });
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to fork';
		} finally {
			forking = false;
		}
	}

	function handleSwitch(branchId: string | null) {
		activeBranchId = branchId;
		dispatch('switchBranch', { branchId });
	}

	async function handleDelete(branchId: string) {
		try {
			await deleteBranch(branchId);
			if (activeBranchId === branchId) {
				activeBranchId = null;
				dispatch('switchBranch', { branchId: null });
			}
			await loadBranches();
			await loadTree();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to delete';
		}
	}

	function startRename(branch: Branch) {
		renamingId = branch.branch_id;
		renameValue = branch.name;
	}

	async function confirmRename() {
		if (!renamingId) return;
		try {
			await updateBranch(renamingId, { name: renameValue.trim() || undefined });
			renamingId = null;
			renameValue = '';
			await loadBranches();
			await loadTree();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to rename';
		}
	}

	function cancelRename() {
		renamingId = null;
		renameValue = '';
	}

	async function handleCompare() {
		if (!conversationId) return;
		comparing = true;
		comparison = null;
		try {
			comparison = await compareBranches({
				conversation_id: conversationId,
				branch_a_id: compareAId,
				branch_b_id: compareBId,
			});
			showCompare = true;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to compare';
		} finally {
			comparing = false;
		}
	}

	async function handleMerge(sourceId: string, targetId: string) {
		try {
			await mergeBranches({
				source_branch_id: sourceId,
				target_branch_id: targetId,
			});
			await loadBranches();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to merge';
		}
	}

	onMount(() => {
		loadBranches();
		loadTree();
	});
</script>

<div class="branch-explorer">
	<!-- Header -->
	<div class="branch-header">
		<button
			class="branch-toggle"
			on:click={() => (showTree = !showTree)}
			aria-expanded={showTree}
			title="Toggle branch explorer"
		>
			<svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
				<path
					d="M4 2v4h2v2H4v6M10 8v6M10 8h2a2 2 0 0 0 0-4h-2v4Z"
					style="stroke: var(--oo-text-secondary); stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; fill: none;"
				/>
			</svg>
			<span class="branch-label">
				{#if activeBranchId}
					{branches.find((b) => b.branch_id === activeBranchId)?.name ?? 'Branch'}
				{:else}
					Main
				{/if}
			</span>
			{#if hasBranches}
				<span class="branch-count">{branches.length}</span>
			{/if}
		</button>
	</div>

	<!-- Expanded panel -->
	{#if showTree}
		<div class="branch-panel">
			<!-- Error display -->
			{#if error}
				<div class="branch-error">{error}</div>
			{/if}

			<!-- Fork controls -->
			<div class="branch-section">
				<div class="section-title">Fork at current message</div>
				<div class="fork-controls">
					<input
						type="text"
						class="fork-input"
						aria-label="Branch name"
						placeholder="Branch name (optional)"
						bind:value={forkName}
						on:keydown={(e) => e.key === 'Enter' && handleFork()}
						disabled={forking || currentMessageId == null}
					/>
					<button
						class="fork-btn"
						on:click={handleFork}
						disabled={forking || currentMessageId == null}
						title={currentMessageId == null
							? 'Select a message to fork from'
							: 'Create branch at this message'}
					>
						{forking ? 'Forking...' : 'Fork'}
					</button>
				</div>
				{#if currentMessageId == null}
					<div class="fork-hint">Click a message to select the fork point</div>
				{/if}
			</div>

			<!-- Branch tree (recursive) -->
			<div class="branch-section">
				<div class="section-title">Branches</div>

				{#if loading}
					<div class="branch-loading">Loading branches...</div>
				{:else if tree}
					<div class="tree-container" role="tree" aria-label="Branch tree">
						<BranchTreeNodeItem
							branchId={null}
							name="Main"
							color="var(--oo-accent)"
							messageCount={tree.message_count || 0}
							lastActivity={tree.last_activity || ''}
							forkMessageId={null}
							depth={0}
							children={tree.children || []}
							{activeBranchId}
							isRoot={true}
							on:switchBranch={(e) => handleSwitch(e.detail.branchId)}
						/>
					</div>
				{:else}
					<!-- Fallback: simple list if tree not available -->
					<button
						class="branch-item"
						class:active={activeBranchId === null}
						on:click={() => handleSwitch(null)}
					>
						<span
							class="branch-dot"
							style="background-color: var(--oo-accent);"
						></span>
						<span class="branch-name">Main</span>
					</button>

					{#each branches as branch (branch.branch_id)}
						<div
							class="branch-item"
							class:active={activeBranchId === branch.branch_id}
						>
							{#if renamingId === branch.branch_id}
								<input
									class="rename-input"
									type="text"
									aria-label="Rename branch"
									bind:value={renameValue}
									on:keydown={(e) => {
										if (e.key === 'Enter') confirmRename();
										if (e.key === 'Escape') cancelRename();
									}}
								/>
								<button class="icon-btn" on:click={confirmRename} title="Confirm" aria-label="Confirm rename">
									<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
										<path
											d="M2 6l3 3 5-5"
											style="stroke: var(--oo-success); stroke-width: 2; fill: none; stroke-linecap: round; stroke-linejoin: round;"
										/>
									</svg>
								</button>
								<button class="icon-btn" on:click={cancelRename} title="Cancel" aria-label="Cancel rename">
									<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
										<path
											d="M3 3l6 6M9 3l-6 6"
											style="stroke: var(--oo-error); stroke-width: 2; fill: none; stroke-linecap: round;"
										/>
									</svg>
								</button>
							{:else}
								<button
									class="branch-switch"
									on:click={() => handleSwitch(branch.branch_id)}
								>
									<span
										class="branch-dot"
										style="background-color: {branch.color};"
									></span>
									<span class="branch-name">{branch.name}</span>
									{#if branch.stats}
										<span class="branch-meta">
											{branch.stats.message_count} msg
										</span>
									{/if}
								</button>
								<button
									class="icon-btn"
									on:click={() => startRename(branch)}
									title="Rename branch"
								>
									<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
										<path
											d="M8.5 1.5l2 2L4 10H2v-2l6.5-6.5z"
											style="stroke: var(--oo-text-secondary); stroke-width: 1.2; fill: none; stroke-linecap: round; stroke-linejoin: round;"
										/>
									</svg>
								</button>
								<button
									class="icon-btn danger"
									on:click={() => handleDelete(branch.branch_id)}
									title="Delete branch"
								>
									<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
										<path
											d="M3 3l6 6M9 3l-6 6"
											style="stroke: var(--oo-error); stroke-width: 1.5; fill: none; stroke-linecap: round;"
										/>
									</svg>
								</button>
							{/if}
						</div>
					{/each}
				{/if}
			</div>

			<!-- Compare section -->
			{#if branches.length >= 1}
				<div class="branch-section">
					<div class="section-title">Compare</div>
					<div class="compare-controls">
						<select
							class="compare-select"
							bind:value={compareAId}
						>
							<option value={null}>Main</option>
							{#each branches as b (b.branch_id)}
								<option value={b.branch_id}>{b.name}</option>
							{/each}
						</select>
						<span class="compare-vs">vs</span>
						<select
							class="compare-select"
							bind:value={compareBId}
						>
							<option value={null}>Main</option>
							{#each branches as b (b.branch_id)}
								<option value={b.branch_id}>{b.name}</option>
							{/each}
						</select>
						<button
							class="compare-btn"
							on:click={handleCompare}
							disabled={comparing || (compareAId === null && compareBId === null)}
						>
							{comparing ? 'Comparing...' : 'Compare'}
						</button>
					</div>
				</div>
			{/if}

			<!-- Comparison results -->
			{#if showCompare && comparison}
				<div class="branch-section">
					<div class="section-title">
						Comparison: {comparison.branch_a_name} vs {comparison.branch_b_name}
						<button
							class="icon-btn"
							on:click={() => { showCompare = false; comparison = null; }}
							title="Close comparison"
						>
							<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
								<path
									d="M3 3l6 6M9 3l-6 6"
									style="stroke: var(--oo-text-secondary); stroke-width: 1.5; fill: none; stroke-linecap: round;"
								/>
							</svg>
						</button>
					</div>
					<div class="compare-result">
						<div class="compare-shared">
							<div class="compare-label">Shared messages</div>
							<div class="compare-count">{comparison.shared_messages.length}</div>
						</div>
						<div class="compare-columns">
							<div class="compare-col">
								<div class="compare-col-header" style="border-color: var(--oo-accent);">
									{comparison.branch_a_name}
								</div>
								{#each comparison.branch_a_messages as msg}
									<div class="compare-msg" class:user={msg.role === 'user'}>
										<span class="msg-role">{msg.role}</span>
										<span class="msg-content">{msg.content}</span>
									</div>
								{/each}
								{#if comparison.branch_a_messages.length === 0}
									<div class="compare-empty">No divergent messages</div>
								{/if}
							</div>
							<div class="compare-col">
								<div class="compare-col-header" style="border-color: var(--oo-success);">
									{comparison.branch_b_name}
								</div>
								{#each comparison.branch_b_messages as msg}
									<div class="compare-msg" class:user={msg.role === 'user'}>
										<span class="msg-role">{msg.role}</span>
										<span class="msg-content">{msg.content}</span>
									</div>
								{/each}
								{#if comparison.branch_b_messages.length === 0}
									<div class="compare-empty">No divergent messages</div>
								{/if}
							</div>
						</div>
					</div>

					<!-- Merge button -->
					{#if comparison.branch_a_id && comparison.branch_b_id}
						<div class="merge-controls">
							<button
								class="merge-btn"
								on:click={() => {
									if (comparison?.branch_a_id && comparison?.branch_b_id) {
										handleMerge(comparison.branch_a_id, comparison.branch_b_id);
									}
								}}
								title="Copy messages from {comparison.branch_a_name} into {comparison.branch_b_name}"
							>
								Merge {comparison.branch_a_name} into {comparison.branch_b_name}
							</button>
						</div>
					{/if}
				</div>
			{/if}

			</div>
	{/if}
</div>

<style>
	.branch-explorer {
		font-size: 0.8125rem;
	}

	.branch-header {
		display: flex;
		align-items: center;
	}

	.branch-toggle {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		padding: 0.25rem 0.5rem;
		border-radius: 6px;
		border: 1px solid var(--oo-border);
		background: var(--oo-surface);
		color: var(--oo-text-secondary);
		cursor: pointer;
		transition: background-color 0.15s ease, border-color 0.15s ease;
	}

	.branch-toggle:hover {
		background: var(--oo-surface-hover);
		border-color: var(--oo-border-hover);
	}

	.branch-label {
		font-weight: 500;
		color: var(--oo-text-primary);
	}

	.branch-count {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		min-width: 1.125rem;
		height: 1.125rem;
		padding: 0 0.25rem;
		border-radius: 999px;
		background: var(--oo-accent);
		color: var(--oo-text-on-accent);
		font-size: 0.6875rem;
		font-weight: 600;
	}

	.branch-panel {
		margin-top: 0.5rem;
		padding: 0.75rem;
		border: 1px solid var(--oo-border);
		border-radius: 10px;
		background: var(--oo-surface);
		max-height: 70vh;
		overflow-y: auto;
	}

	.branch-error {
		padding: 0.375rem 0.5rem;
		margin-bottom: 0.5rem;
		border-radius: 6px;
		background: var(--oo-error-bg);
		color: var(--oo-error);
		font-size: 0.75rem;
	}

	.branch-section {
		margin-bottom: 0.75rem;
		padding-bottom: 0.75rem;
		border-bottom: 1px solid var(--oo-border);
	}

	.branch-section:last-child {
		border-bottom: none;
		margin-bottom: 0;
		padding-bottom: 0;
	}

	.section-title {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		font-size: 0.6875rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--oo-text-tertiary);
		margin-bottom: 0.5rem;
	}

	/* Fork controls */
	.fork-controls {
		display: flex;
		gap: 0.375rem;
	}

	.fork-input {
		flex: 1;
		padding: 0.3rem 0.5rem;
		border-radius: 6px;
		border: 1px solid var(--oo-border);
		background: var(--oo-bg);
		color: var(--oo-text-primary);
		font-size: 0.8125rem;
	}

	.fork-input:focus {
		outline: none;
		border-color: var(--oo-accent);
	}

	.fork-input:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.fork-btn {
		padding: 0.3rem 0.75rem;
		border-radius: 6px;
		border: 1px solid var(--oo-accent);
		background: var(--oo-accent);
		color: var(--oo-text-on-accent);
		font-size: 0.8125rem;
		font-weight: 500;
		cursor: pointer;
		transition: opacity 0.15s ease;
	}

	.fork-btn:hover:not(:disabled) {
		opacity: 0.85;
	}

	.fork-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.fork-hint {
		margin-top: 0.25rem;
		font-size: 0.6875rem;
		color: var(--oo-text-tertiary);
	}

	/* Branch list */
	.branch-item {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		width: 100%;
		padding: 0.3rem 0.5rem;
		border-radius: 6px;
		border: 1px solid transparent;
		background: none;
		color: var(--oo-text-primary);
		cursor: pointer;
		font-size: 0.8125rem;
		text-align: left;
		transition: background-color 0.15s ease;
	}

	.branch-item:hover {
		background: var(--oo-surface-hover);
	}

	.branch-item.active {
		background: var(--oo-surface-hover);
		border-color: var(--oo-accent);
	}

	.branch-switch {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		flex: 1;
		border: none;
		background: none;
		color: var(--oo-text-primary);
		cursor: pointer;
		font-size: 0.8125rem;
		text-align: left;
		padding: 0;
	}

	.branch-dot {
		display: inline-block;
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.branch-name {
		flex: 1;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.branch-meta {
		font-size: 0.6875rem;
		color: var(--oo-text-tertiary);
		flex-shrink: 0;
	}

	.branch-loading {
		padding: 0.5rem;
		font-size: 0.75rem;
		color: var(--oo-text-tertiary);
	}

	.rename-input {
		flex: 1;
		padding: 0.15rem 0.375rem;
		border-radius: 4px;
		border: 1px solid var(--oo-accent);
		background: var(--oo-bg);
		color: var(--oo-text-primary);
		font-size: 0.8125rem;
	}

	.icon-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 22px;
		height: 22px;
		border-radius: 4px;
		border: none;
		background: none;
		cursor: pointer;
		opacity: 0.5;
		transition: opacity 0.15s ease;
		flex-shrink: 0;
	}

	.icon-btn:hover {
		opacity: 1;
		background: var(--oo-surface-hover);
	}

	/* Compare */
	.compare-controls {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		flex-wrap: wrap;
	}

	.compare-select {
		flex: 1;
		min-width: 5rem;
		padding: 0.25rem 0.375rem;
		border-radius: 6px;
		border: 1px solid var(--oo-border);
		background: var(--oo-bg);
		color: var(--oo-text-primary);
		font-size: 0.75rem;
	}

	.compare-vs {
		font-size: 0.6875rem;
		color: var(--oo-text-tertiary);
		font-weight: 500;
	}

	.compare-btn {
		padding: 0.25rem 0.625rem;
		border-radius: 6px;
		border: 1px solid var(--oo-border);
		background: var(--oo-surface);
		color: var(--oo-text-primary);
		font-size: 0.75rem;
		cursor: pointer;
		transition: border-color 0.15s ease;
	}

	.compare-btn:hover:not(:disabled) {
		border-color: var(--oo-accent);
	}

	.compare-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	/* Comparison result */
	.compare-result {
		margin-top: 0.5rem;
	}

	.compare-shared {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		margin-bottom: 0.5rem;
		font-size: 0.75rem;
		color: var(--oo-text-secondary);
	}

	.compare-label {
		font-weight: 500;
	}

	.compare-count {
		padding: 0.1rem 0.375rem;
		border-radius: 4px;
		background: var(--oo-surface-hover);
		font-size: 0.6875rem;
	}

	.compare-columns {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.5rem;
	}

	.compare-col {
		border: 1px solid var(--oo-border);
		border-radius: 6px;
		overflow: hidden;
	}

	.compare-col-header {
		padding: 0.3rem 0.5rem;
		font-size: 0.6875rem;
		font-weight: 600;
		border-bottom: 2px solid;
		background: var(--oo-surface-hover);
		color: var(--oo-text-primary);
	}

	.compare-msg {
		padding: 0.3rem 0.5rem;
		border-bottom: 1px solid var(--oo-border);
		font-size: 0.75rem;
	}

	.compare-msg:last-child {
		border-bottom: none;
	}

	.compare-msg.user {
		background: var(--oo-surface-hover);
	}

	.msg-role {
		font-weight: 600;
		font-size: 0.625rem;
		text-transform: uppercase;
		color: var(--oo-text-tertiary);
		margin-right: 0.25rem;
	}

	.msg-content {
		color: var(--oo-text-primary);
		word-break: break-word;
		display: -webkit-box;
		-webkit-line-clamp: 3;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}

	.compare-empty {
		padding: 0.75rem;
		text-align: center;
		font-size: 0.75rem;
		color: var(--oo-text-tertiary);
	}

	/* Merge */
	.merge-controls {
		margin-top: 0.5rem;
	}

	.merge-btn {
		width: 100%;
		padding: 0.3rem 0.5rem;
		border-radius: 6px;
		border: 1px solid var(--oo-border);
		background: var(--oo-surface);
		color: var(--oo-text-primary);
		font-size: 0.75rem;
		cursor: pointer;
		transition: border-color 0.15s ease;
	}

	.merge-btn:hover {
		border-color: var(--oo-accent);
	}

	/* Tree container */
	.tree-container {
		padding: 0.125rem 0;
	}
</style>
