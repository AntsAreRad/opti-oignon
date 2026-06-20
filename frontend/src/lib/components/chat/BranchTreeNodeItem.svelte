<!--
  BranchTreeNodeItem.svelte (S154)
  Recursive tree node component for branch visualization.
  Renders a single node with collapse toggle, metadata, and children.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	export let branchId: string | null = null;
	export let name: string = 'Main';
	export let color: string = 'var(--oo-accent)';
	export let messageCount: number = 0;
	export let lastActivity: string = '';
	export let forkMessageId: number | null = null;
	export let depth: number = 0;
	export let children: any[] = [];
	export let activeBranchId: string | null = null;
	export let isRoot: boolean = false;

	const dispatch = createEventDispatcher<{
		switchBranch: { branchId: string | null };
		fork: { branchId: string | null };
	}>();

	let collapsed = depth > 2;
	$: hasChildren = children && children.length > 0;
	$: isActive = branchId === activeBranchId;

	function handleClick() {
		dispatch('switchBranch', { branchId });
	}

	function toggleCollapse(e: MouseEvent) {
		e.stopPropagation();
		collapsed = !collapsed;
	}

	function formatActivity(ts: string): string {
		if (!ts) return '';
		try {
			const d = new Date(ts);
			const now = new Date();
			const diff = now.getTime() - d.getTime();
			const mins = Math.floor(diff / 60000);
			if (mins < 1) return 'now';
			if (mins < 60) return `${mins}m ago`;
			const hours = Math.floor(mins / 60);
			if (hours < 24) return `${hours}h ago`;
			const days = Math.floor(hours / 24);
			if (days < 30) return `${days}d ago`;
			return d.toLocaleDateString();
		} catch {
			return '';
		}
	}

	function handleChildSwitch(e: CustomEvent<{ branchId: string | null }>) {
		dispatch('switchBranch', e.detail);
	}

	function handleChildFork(e: CustomEvent<{ branchId: string | null }>) {
		dispatch('fork', e.detail);
	}
</script>

<div class="tree-node-wrapper" style="--node-depth: {depth};">
	<div
		class="tree-node-row"
		class:active={isActive}
		class:root={isRoot}
		role="treeitem"
		aria-expanded={hasChildren ? !collapsed : undefined}
		aria-level={depth + 1}
		tabindex="0"
		on:click={handleClick}
		on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleClick(); } }}
	>
		<!-- Collapse toggle -->
		{#if hasChildren}
			<button
				class="collapse-toggle"
				on:click={toggleCollapse}
				aria-label={collapsed ? 'Expand branch' : 'Collapse branch'}
			>
				<svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
					<path
						d={collapsed ? 'M3 1l4 4-4 4' : 'M1 3l4 4 4-4'}
						style="stroke: var(--oo-text-tertiary); stroke-width: 1.5; fill: none; stroke-linecap: round; stroke-linejoin: round;"
					/>
				</svg>
			</button>
		{:else}
			<span class="collapse-spacer"></span>
		{/if}

		<!-- Connector line (not on root) -->
		{#if !isRoot}
			<span class="tree-connector-line"></span>
		{/if}

		<!-- Dot -->
		<span class="node-dot" style="background-color: {color};"></span>

		<!-- Name -->
		<span class="node-name" title={name}>{name}</span>

		<!-- Metadata -->
		{#if messageCount > 0}
			<span class="node-meta">{messageCount} msg</span>
		{/if}
		{#if lastActivity}
			<span class="node-meta node-time">{formatActivity(lastActivity)}</span>
		{/if}
		{#if forkMessageId !== null && !isRoot}
			<span class="node-fork-badge" title="Forked at message #{forkMessageId}">
				#{forkMessageId}
			</span>
		{/if}

		<!-- Children count badge -->
		{#if hasChildren && collapsed}
			<span class="node-children-badge">{children.length}</span>
		{/if}
	</div>

	<!-- Children (recursive) -->
	{#if hasChildren && !collapsed}
		<div class="tree-children" role="group">
			{#each children as child (child.branch_id || child.name)}
				<svelte:self
					branchId={child.branch_id}
					name={child.name}
					color={child.color || 'var(--oo-text-tertiary)'}
					messageCount={child.message_count || 0}
					lastActivity={child.last_activity || ''}
					forkMessageId={child.fork_message_id ?? null}
					depth={depth + 1}
					children={child.children || []}
					{activeBranchId}
					isRoot={false}
					on:switchBranch={handleChildSwitch}
					on:fork={handleChildFork}
				/>
			{/each}
		</div>
	{/if}
</div>

<style>
	.tree-node-wrapper {
		position: relative;
	}

	.tree-children {
		margin-left: 1.125rem;
		border-left: 1px solid var(--oo-border);
	}

	.tree-node-row {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		padding: 0.25rem 0.4rem;
		border-radius: 6px;
		cursor: pointer;
		font-size: 0.8125rem;
		transition: background-color 0.12s ease;
		position: relative;
	}

	.tree-node-row:hover {
		background: var(--oo-surface-hover);
	}

	.tree-node-row:focus-visible {
		outline: 2px solid var(--oo-accent);
		outline-offset: -2px;
	}

	.tree-node-row.active {
		background: var(--oo-surface-hover);
		border: 1px solid var(--oo-accent);
		padding: calc(0.25rem - 1px) calc(0.4rem - 1px);
	}

	.tree-node-row:not(.active) {
		border: 1px solid transparent;
	}

	.collapse-toggle {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 16px;
		height: 16px;
		border: none;
		background: none;
		cursor: pointer;
		border-radius: 3px;
		flex-shrink: 0;
		padding: 0;
	}

	.collapse-toggle:hover {
		background: var(--oo-surface-hover);
	}

	.collapse-spacer {
		display: inline-block;
		width: 16px;
		flex-shrink: 0;
	}

	.tree-connector-line {
		position: absolute;
		left: -0.5625rem;
		top: 50%;
		width: 0.5625rem;
		height: 0;
		border-top: 1px solid var(--oo-border);
	}

	.node-dot {
		display: inline-block;
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.node-name {
		flex: 1;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-weight: 500;
		color: var(--oo-text-primary);
	}

	.node-meta {
		font-size: 0.6875rem;
		color: var(--oo-text-tertiary);
		flex-shrink: 0;
		white-space: nowrap;
	}

	.node-time {
		font-style: italic;
	}

	.node-fork-badge {
		font-size: 0.625rem;
		color: var(--oo-text-tertiary);
		background: var(--oo-surface-hover);
		padding: 0 0.25rem;
		border-radius: 3px;
		flex-shrink: 0;
	}

	.node-children-badge {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		min-width: 1rem;
		height: 1rem;
		padding: 0 0.2rem;
		border-radius: 999px;
		background: var(--oo-surface-hover);
		color: var(--oo-text-tertiary);
		font-size: 0.625rem;
		font-weight: 600;
		flex-shrink: 0;
	}
</style>
