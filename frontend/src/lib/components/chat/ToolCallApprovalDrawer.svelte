<!--
  ToolCallApprovalDrawer.svelte (S169)
  Standalone, self-contained approvals surface extracted from the inline
  ToolCallApproval card. Polls for pending tool-call approvals and presents
  them in a ds Modal drawer-right with per-request Allow / Deny actions and a
  risk badge. A small anchored pill appears only while approvals are pending.
  Pure --oo-* tokens; English only. The original inline card is unchanged.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { Modal, Button } from '$lib/ds';
	import {
		getPendingApprovals,
		approveToolCall,
		denyToolCall,
		type PendingApproval,
	} from '$lib/api/toolCallApproval';

	let pending: PendingApproval[] = [];
	let open = false;
	let actioningId = '';
	let error = '';
	let pollTimer: ReturnType<typeof setInterval> | null = null;

	const POLL_MS = 5000;

	onMount(() => {
		refresh();
		pollTimer = setInterval(refresh, POLL_MS);
	});

	onDestroy(() => {
		if (pollTimer) clearInterval(pollTimer);
	});

	async function refresh() {
		try {
			const data = await getPendingApprovals();
			pending = data.available ? data.pending : [];
		} catch {
			// Approval endpoint may be unavailable; treat as no pending.
			pending = [];
		}
	}

	async function handleApprove(id: string) {
		actioningId = id;
		error = '';
		try {
			const result = await approveToolCall(id);
			if (!result.success) error = 'Approval failed';
		} catch {
			error = 'Approval failed';
		} finally {
			actioningId = '';
			await refresh();
		}
	}

	async function handleDeny(id: string) {
		actioningId = id;
		error = '';
		try {
			const result = await denyToolCall(id);
			if (!result.success) error = 'Denial failed';
		} catch {
			error = 'Denial failed';
		} finally {
			actioningId = '';
			await refresh();
		}
	}

	function riskColor(level: string): string {
		if (level === 'high') return 'var(--oo-error)';
		if (level === 'medium') return 'var(--oo-warning)';
		return 'var(--oo-success)';
	}

	function closeDrawer() {
		open = false;
	}

	$: count = pending.length;
</script>

{#if count > 0}
	<button class="oo-approval-pill" on:click={() => (open = true)}>
		<span class="oo-approval-dot"></span>
		{count} pending approval{count !== 1 ? 's' : ''}
	</button>
{/if}

<Modal {open} variant="drawer-right" size="md" title="Tool call approvals" onClose={closeDrawer}>
	{#if count === 0}
		<p class="text-sm" style="color: var(--oo-fg-muted);">No pending approvals.</p>
	{:else}
		{#if error}
			<div class="text-xs mb-3 px-3 py-2 rounded" style="background-color: var(--oo-error-bg); color: var(--oo-error);">
				{error}
			</div>
		{/if}
		<div class="flex flex-col gap-3">
			{#each pending as req (req.approval_id)}
				<div class="rounded-lg p-3" style="background-color: var(--oo-bg-elevated); border: 1px solid {riskColor(req.risk_level)};">
					<div class="flex items-center gap-2 mb-2">
						<span class="text-xs font-mono px-2 py-0.5 rounded" style="background-color: var(--oo-bg-subtle); color: var(--oo-fg-primary);">
							{req.tool_name}
						</span>
						<span class="text-xs px-1.5 py-0.5 rounded font-medium capitalize" style="background-color: {riskColor(req.risk_level)}; color: var(--oo-fg-on-accent);">
							{req.risk_level}
						</span>
						{#if req.timeout_remaining > 0}
							<span class="text-xs font-mono ml-auto" style="color: var(--oo-fg-muted);">{req.timeout_remaining}s</span>
						{/if}
					</div>

					{#if req.arguments_summary}
						<pre class="text-xs mb-3 p-2 rounded whitespace-pre-wrap" style="background-color: var(--oo-bg-subtle); color: var(--oo-fg-muted); word-break: break-all;">{req.arguments_summary}</pre>
					{/if}

					<div class="flex gap-2">
						<Button variant="primary" size="sm" block loading={actioningId === req.approval_id} on:click={() => handleApprove(req.approval_id)}>
							Allow
						</Button>
						<Button variant="danger" size="sm" block loading={actioningId === req.approval_id} on:click={() => handleDeny(req.approval_id)}>
							Deny
						</Button>
					</div>
				</div>
			{/each}
		</div>
		<p class="text-xs mt-3" style="color: var(--oo-fg-faint);">
			Requests auto-deny on timeout (fail-secure).
		</p>
	{/if}
</Modal>

<style>
	.oo-approval-pill {
		position: fixed;
		right: 1.25rem;
		bottom: 5rem;
		z-index: 40;
		display: inline-flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.875rem;
		border-radius: var(--oo-radius-full);
		font-size: var(--oo-text-xs);
		font-weight: 500;
		color: var(--oo-fg-on-accent);
		background-color: var(--oo-warning);
		border: none;
		cursor: pointer;
		box-shadow: var(--oo-shadow-md);
	}
	.oo-approval-dot {
		width: 0.5rem;
		height: 0.5rem;
		border-radius: var(--oo-radius-full);
		background-color: var(--oo-fg-on-accent);
		animation: oo-approval-pulse 1.4s ease-in-out infinite;
	}
	@keyframes oo-approval-pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.4; }
	}
</style>
