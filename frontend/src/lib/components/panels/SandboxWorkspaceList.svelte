<!--
  SandboxWorkspaceList.svelte (Sandbox Workspace cycle, Bloc 1)
  The decomposed workspace list of the sandbox manager (spec section 4.2),
  built on the lib/ds primitives. Each row surfaces the manager fields: id,
  optional label, bound conversation, age, approximate disk use, running/idle,
  network on/off (off this cycle; Bloc 4 flips it) and last activity, with the
  stop / delete / select actions. Stop SIGKILLs the running command and keeps
  the workspace (files persist for inspection); delete destroys it; select
  binds it to the active conversation. Design-system tokens only (--oo-*);
  lucide icons through Icon. Registered in FRONTEND_REDESIGN_SPEC.md.
-->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { Button, Icon, EmptyState } from '$lib/ds';
	import type { SandboxSessionInfo } from '$lib/types';

	export let sessions: SandboxSessionInfo[] = [];
	export let activeConversationId: string | null = null;
	export let busyId: string | null = null;

	const dispatch = createEventDispatcher<{
		stop: { sessionId: string };
		destroy: { sessionId: string };
		select: { sessionId: string };
		unbind: { sessionId: string; conversationId: string };
	}>();

	function formatAge(seconds: number): string {
		if (seconds < 60) return `${Math.floor(seconds)}s`;
		if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
		if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
		return `${Math.floor(seconds / 86400)}d`;
	}

	function formatBytes(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
		if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
		return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
	}

	function formatLastActivity(epochSeconds: number): string {
		if (!epochSeconds) return 'never';
		const deltaSec = Math.max(0, Date.now() / 1000 - epochSeconds);
		return `${formatAge(deltaSec)} ago`;
	}

	function boundToActive(session: SandboxSessionInfo): boolean {
		return (
			activeConversationId !== null &&
			session.bound_conversation_id === activeConversationId
		);
	}

	function isQuickAuto(session: SandboxSessionInfo): boolean {
		// Quick sessions are keyed by the conversation that spawned them,
		// so their manager id IS that conversation id. Their binding is
		// implicit (the chat lifecycle owns it): the explicit unbind route
		// is a no-op for them, so neither Unbind nor Select is offered.
		return (
			session.bound_conversation_id !== null &&
			session.session_id === session.bound_conversation_id
		);
	}
</script>

{#if sessions.length === 0}
	<EmptyState
		icon="box"
		title="No workspaces"
		description="Create a workspace to give this conversation an isolated sandbox."
	/>
{:else}
	<ul class="workspace-list">
		{#each sessions as session (session.session_id)}
			<li class="workspace-row" class:is-bound-active={boundToActive(session)}>
				<div class="workspace-head">
					<span class="workspace-name" title={session.session_id}>
						<span class="workspace-state-icon" class:is-running={session.running}>
							<Icon name={session.running ? 'loader' : 'box'} size="sm" />
						</span>
						<span class="workspace-label">{session.label || session.session_id}</span>
						{#if session.label}
							<span class="workspace-id">{session.session_id}</span>
						{/if}
					</span>
					<span class="workspace-status" class:is-running={session.running}>
						{session.running ? 'running' : 'idle'}
					</span>
				</div>

				<div class="workspace-meta">
					<span title="Workspace age">age {formatAge(session.age_seconds)}</span>
					<span title="Approximate disk use (bounded walk)">
						disk {formatBytes(session.disk_use_bytes)}
					</span>
					<span title="Network stays off this cycle (Bloc 4 flips it)">
						network {session.network_enabled ? 'on' : 'off'}
					</span>
					<span title="Last activity">active {formatLastActivity(session.last_activity)}</span>
					{#if session.bound_conversation_id}
						<span
							class="workspace-bound"
							class:is-bound-active={boundToActive(session)}
							title={isQuickAuto(session)
								? 'Auto-created by the chat for this conversation; the chat lifecycle manages it'
								: 'Bound conversation'}
						>
							bound {boundToActive(session)
								? 'to this conversation'
								: session.bound_conversation_id.slice(0, 8)}{isQuickAuto(session) ? ' (auto)' : ''}
						</span>
					{/if}
				</div>

				<div class="workspace-actions">
					{#if isQuickAuto(session)}
						<!-- Implicit chat-owned binding: no explicit bind action applies. -->
					{:else if boundToActive(session)}
						<Button
							size="sm"
							variant="ghost"
							iconLeft="unlink"
							disabled={busyId === session.session_id}
							ariaLabel="Unbind this workspace from the active conversation"
							on:click={() =>
								dispatch('unbind', {
									sessionId: session.session_id,
									conversationId: session.bound_conversation_id ?? ''
								})}
						>
							Unbind
						</Button>
					{:else}
						<Button
							size="sm"
							variant="ghost"
							iconLeft="link"
							disabled={busyId === session.session_id ||
								activeConversationId === null ||
								session.bound_conversation_id !== null}
							ariaLabel="Bind this workspace to the active conversation"
							on:click={() => dispatch('select', { sessionId: session.session_id })}
						>
							Select
						</Button>
					{/if}
					<Button
						size="sm"
						variant="ghost"
						iconLeft="square"
						disabled={!session.running || busyId === session.session_id}
						ariaLabel="Stop the running command; the workspace and its files persist"
						on:click={() => dispatch('stop', { sessionId: session.session_id })}
					>
						Stop
					</Button>
					<Button
						size="sm"
						variant="danger"
						iconLeft="trash-2"
						disabled={busyId === session.session_id}
						ariaLabel="Destroy the workspace and all its files"
						on:click={() => dispatch('destroy', { sessionId: session.session_id })}
					>
						Delete
					</Button>
				</div>
			</li>
		{/each}
	</ul>
{/if}

<style>
	.workspace-list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.workspace-row {
		border: 1px solid var(--oo-bd-subtle);
		border-radius: 0.375rem;
		background-color: var(--oo-bg-surface);
		padding: 0.75rem;
	}
	.workspace-row.is-bound-active {
		border-color: var(--oo-acc-400);
	}
	.workspace-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
	}
	.workspace-name {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		min-width: 0;
		color: var(--oo-fg-primary);
	}
	.workspace-state-icon {
		display: inline-flex;
		color: var(--oo-fg-tertiary);
	}
	.workspace-state-icon.is-running {
		color: var(--oo-acc-400);
	}
	.workspace-label {
		font-weight: 500;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.workspace-id {
		font-size: 0.75rem;
		color: var(--oo-fg-tertiary);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.workspace-status {
		font-size: 0.75rem;
		padding: 0.125rem 0.375rem;
		border-radius: 0.25rem;
		color: var(--oo-fg-tertiary);
		background-color: var(--oo-bg-elevated);
		white-space: nowrap;
	}
	.workspace-status.is-running {
		color: var(--oo-acc-400);
	}
	.workspace-meta {
		margin-top: 0.375rem;
		display: flex;
		flex-wrap: wrap;
		column-gap: 0.75rem;
		row-gap: 0.125rem;
		font-size: 0.75rem;
		color: var(--oo-fg-secondary);
	}
	.workspace-bound.is-bound-active {
		color: var(--oo-acc-400);
	}
	.workspace-actions {
		margin-top: 0.5rem;
		display: flex;
		align-items: center;
		gap: 0.375rem;
		flex-wrap: wrap;
	}
</style>
