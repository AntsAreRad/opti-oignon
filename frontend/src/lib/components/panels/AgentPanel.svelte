<!--
  AgentPanel.svelte (S176, Theme 3 / Odysseus Core)
  The agent panel for the sandboxed agent loop, built on the S166 lib/ds
  primitives (Card, Button, Icon, EmptyState, InlineError). It shows the live
  tool-call stream (consuming the loop's AgentEvents over $lib/api/agent), a
  round / step display, a cancel control, and the Bulbe approval prompts
  (approve / deny) wired to the existing tool-call approval API. Streaming
  updates use aria-live regions. Design-system tokens only (--oo-*); lucide
  icons through Icon. Registered in FRONTEND_REDESIGN_SPEC.md.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { Button, Card, Icon, EmptyState, InlineError } from '$lib/ds';
	import type { IconName } from '$lib/ds';
	import {
		connectAgentStream,
		cancelAgentRun,
		getPendingApprovals,
		approveToolCall,
		denyToolCall,
		type AgentEvent,
		type AgentToolResult,
		type PendingApproval
	} from '$lib/api/agent';
	import type { ReconnectingWebSocket } from '$lib/api/client';
	import { toastSuccess, toastError } from '$lib/stores/notifications';

	const MAX_EVENTS = 200;

	type StreamItem = { id: number; event: AgentEvent };

	let items: StreamItem[] = [];
	let pending: PendingApproval[] = [];
	let running = false;
	let currentRound = 0;
	let stopReason = '';
	let error: string | null = null;
	let busyApprovalId: string | null = null;
	let cancelling = false;
	let nextId = 0;

	let socket: ReconnectingWebSocket | null = null;
	let pollTimer: ReturnType<typeof setInterval> | null = null;

	const KIND_ICON: Record<string, IconName> = {
		round_start: 'circle-dot',
		model_output: 'message-square',
		tool_result: 'wrench',
		done: 'check-circle',
		error: 'alert-triangle',
		verifier_output: 'badge-check'
	};

	const TOOL_ICON: Record<string, IconName> = {
		bash: 'terminal',
		view: 'eye',
		create_file: 'file-plus',
		str_replace: 'replace',
		web_search: 'search',
		manage_memory: 'brain'
	};

	function toolResult(event: AgentEvent): AgentToolResult | null {
		if (event.kind !== 'tool_result') return null;
		return event.data as unknown as AgentToolResult;
	}

	function iconFor(event: AgentEvent): IconName {
		const tr = toolResult(event);
		if (tr) return TOOL_ICON[tr.tool_name] ?? 'wrench';
		return KIND_ICON[event.kind] ?? 'activity';
	}

	function summary(event: AgentEvent): string {
		const tr = toolResult(event);
		if (tr) return tr.observation || tr.reason;
		if (event.kind === 'model_output' || event.kind === 'verifier_output') {
			return String(event.data.content ?? '');
		}
		if (event.kind === 'error') return String(event.data.error ?? 'error');
		if (event.kind === 'done') return String(event.data.final_text ?? 'Done.');
		return '';
	}

	function label(event: AgentEvent): string {
		const tr = toolResult(event);
		if (tr) return tr.tool_name;
		return event.kind.replace(/_/g, ' ');
	}

	function pushEvent(event: AgentEvent) {
		const item = { id: nextId++, event };
		const next = [...items, item];
		items = next.length > MAX_EVENTS ? next.slice(next.length - MAX_EVENTS) : next;

		if (event.kind === 'round_start') {
			currentRound = event.round;
			running = true;
		} else if (event.kind === 'done') {
			running = false;
			stopReason = 'done';
		} else if (event.kind === 'error') {
			running = false;
			stopReason = 'error';
		}
	}

	async function refreshPending() {
		try {
			const res = await getPendingApprovals();
			pending = res.available ? res.pending : [];
		} catch {
			// transient; keep the last known list
		}
	}

	async function handleApprove(id: string) {
		if (busyApprovalId) return;
		busyApprovalId = id;
		try {
			await approveToolCall(id);
			pending = pending.filter((p) => p.approval_id !== id);
			toastSuccess('Tool call approved');
		} catch {
			toastError('Failed to approve tool call');
		} finally {
			busyApprovalId = null;
		}
	}

	async function handleDeny(id: string) {
		if (busyApprovalId) return;
		busyApprovalId = id;
		try {
			await denyToolCall(id);
			pending = pending.filter((p) => p.approval_id !== id);
			toastSuccess('Tool call denied');
		} catch {
			toastError('Failed to deny tool call');
		} finally {
			busyApprovalId = null;
		}
	}

	async function handleCancel() {
		if (cancelling) return;
		cancelling = true;
		try {
			await cancelAgentRun();
			running = false;
			stopReason = 'cancelled';
			toastSuccess('Agent run cancelled');
		} catch {
			toastError('Failed to cancel agent run');
		} finally {
			cancelling = false;
		}
	}

	onMount(() => {
		socket = connectAgentStream(
			(event) => pushEvent(event),
			() => {
				error = 'Lost the agent event stream; reconnecting.';
			}
		);
		refreshPending();
		pollTimer = setInterval(refreshPending, 2000);
	});

	onDestroy(() => {
		if (socket) socket.close();
		if (pollTimer) clearInterval(pollTimer);
	});
</script>

<section class="agent-panel">
	<header class="agent-header">
		<div class="agent-title">
			<Icon name="bot" size="md" />
			<h2>Agent</h2>
		</div>
		<div class="agent-controls">
			<span class="agent-round" role="status" aria-live="polite">
				{#if running}
					Round {currentRound}
				{:else if stopReason}
					Stopped ({stopReason})
				{:else}
					Idle
				{/if}
			</span>
			<Button
				variant="danger"
				size="sm"
				iconLeft="square"
				disabled={!running}
				loading={cancelling}
				on:click={handleCancel}
			>
				Cancel
			</Button>
		</div>
	</header>

	{#if error}
		<InlineError message={error} />
	{/if}

	{#if pending.length > 0}
		<div class="agent-approvals" aria-live="polite">
			<div class="agent-approvals-head">
				<Icon name="shield-alert" size="sm" />
				<span>Approvals required</span>
			</div>
			{#each pending as approval (approval.approval_id)}
				<Card variant="raised" padding="sm">
					<div class="approval-row">
						<div class="approval-info">
							<div class="approval-name">
								<Icon name={TOOL_ICON[approval.tool_name] ?? 'wrench'} size="sm" />
								<span>{approval.tool_name}</span>
								<span class="approval-risk approval-risk-{approval.risk_level}">
									{approval.risk_level}
								</span>
							</div>
							<p class="approval-args">{approval.arguments_summary}</p>
						</div>
						<div class="approval-actions">
							<Button
								variant="primary"
								size="sm"
								iconLeft="check"
								loading={busyApprovalId === approval.approval_id}
								on:click={() => handleApprove(approval.approval_id)}
							>
								Approve
							</Button>
							<Button
								variant="danger"
								size="sm"
								iconLeft="x"
								loading={busyApprovalId === approval.approval_id}
								on:click={() => handleDeny(approval.approval_id)}
							>
								Deny
							</Button>
						</div>
					</div>
				</Card>
			{/each}
		</div>
	{/if}

	<div class="agent-stream" aria-live="polite" aria-label="Agent tool-call stream">
		{#if items.length === 0}
			<EmptyState
				icon="bot"
				title="No agent activity"
				description="Tool calls, model output, and verifier checks stream here as the agent runs in the sandbox."
			/>
		{:else}
			{#each items as item (item.id)}
				<Card variant="flat" padding="sm">
					<div class="stream-row">
						<span class="stream-icon stream-kind-{item.event.kind}">
							<Icon name={iconFor(item.event)} size="sm" />
						</span>
						<div class="stream-body">
							<div class="stream-meta">
								<span class="stream-label">{label(item.event)}</span>
								<span class="stream-round">r{item.event.round}</span>
								{#if toolResult(item.event)}
									<span
										class="stream-status"
										class:stream-ok={toolResult(item.event)?.executed}
										class:stream-bad={!toolResult(item.event)?.executed}
									>
										{toolResult(item.event)?.executed ? 'ran' : 'refused'}
									</span>
								{/if}
							</div>
							{#if summary(item.event)}
								<p class="stream-text">{summary(item.event)}</p>
							{/if}
						</div>
					</div>
				</Card>
			{/each}
		{/if}
	</div>
</section>

<style>
	.agent-panel {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-3);
		padding: var(--oo-space-3);
	}

	.agent-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.agent-title {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-primary);
	}

	.agent-title h2 {
		margin: 0;
		font-size: var(--oo-text-lg);
		font-weight: 600;
	}

	.agent-controls {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
	}

	.agent-round {
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-muted);
		font-variant-numeric: tabular-nums;
	}

	.agent-approvals {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
	}

	.agent-approvals-head {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-acc-600);
		font-size: var(--oo-text-sm);
		font-weight: 600;
	}

	.approval-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--oo-space-3);
	}

	.approval-info {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
		min-width: 0;
	}

	.approval-name {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		color: var(--oo-fg-primary);
		font-weight: 600;
		font-size: var(--oo-text-sm);
	}

	.approval-risk {
		font-size: var(--oo-text-xs);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		border-radius: var(--oo-radius-full);
		padding: 0 var(--oo-space-2);
		background: var(--oo-bg-elevated);
		color: var(--oo-fg-tertiary);
	}

	.approval-risk-high {
		background: var(--oo-danger-bg);
		color: var(--oo-danger);
	}

	.approval-risk-medium {
		background: var(--oo-warning-bg);
		color: var(--oo-warning);
	}

	.approval-args {
		margin: 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
		overflow-wrap: anywhere;
	}

	.approval-actions {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
		flex-shrink: 0;
	}

	.agent-stream {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-2);
		max-height: 60vh;
		overflow-y: auto;
	}

	.stream-row {
		display: flex;
		align-items: flex-start;
		gap: var(--oo-space-2);
	}

	.stream-icon {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		color: var(--oo-fg-tertiary);
	}

	.stream-kind-error {
		color: var(--oo-danger);
	}

	.stream-kind-done {
		color: var(--oo-success);
	}

	.stream-body {
		display: flex;
		flex-direction: column;
		gap: var(--oo-space-1);
		min-width: 0;
	}

	.stream-meta {
		display: flex;
		align-items: center;
		gap: var(--oo-space-2);
	}

	.stream-label {
		font-size: var(--oo-text-sm);
		font-weight: 600;
		color: var(--oo-fg-primary);
		text-transform: capitalize;
	}

	.stream-round {
		font-size: var(--oo-text-xs);
		color: var(--oo-fg-faint);
		font-variant-numeric: tabular-nums;
	}

	.stream-status {
		font-size: var(--oo-text-xs);
		border-radius: var(--oo-radius-full);
		padding: 0 var(--oo-space-2);
		background: var(--oo-bg-elevated);
		color: var(--oo-fg-tertiary);
	}

	.stream-ok {
		background: var(--oo-success-bg);
		color: var(--oo-success);
	}

	.stream-bad {
		background: var(--oo-danger-bg);
		color: var(--oo-danger);
	}

	.stream-text {
		margin: 0;
		font-size: var(--oo-text-sm);
		color: var(--oo-fg-secondary);
		line-height: 1.4;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
</style>
