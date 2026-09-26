<!--
  ToolCallApprovalDrawer.svelte
  Standalone, self-contained approvals surface extracted from the inline
  ToolCallApproval card. Mounted once, by the shell's layout, on every page.
  Polls for pending tool-call approvals and presents them in a ds Modal
  drawer-right with per-request Allow / Deny actions and a risk badge. The
  count goes to the approvals store, whose pill (in the status card and the
  phone header) appears only while approvals are pending and opens this
  drawer. Pure --oo-* tokens; English only. The original inline card is
  unchanged.

  The drawer is a modal dialog, which makes the rest of the page inert, the
  shell's Stop all included; so its foot holds Stop all of its own, one tap
  away while a risky call is being read.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { Modal, Button } from '$lib/ds';
	import StopAllButton from '$lib/components/layout/StopAllButton.svelte';
	import { isPhone } from '$lib/stores/ui';
	import { pendingApprovals, approvalsOpen, closeApprovals } from '$lib/stores/approvals';
	import {
		getPendingApprovals,
		approveToolCall,
		denyToolCall,
		type PendingApproval,
	} from '$lib/api/toolCallApproval';

	let pending: PendingApproval[] = [];
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

	$: count = pending.length;
	$: pendingApprovals.set(count);
</script>

<Modal open={$approvalsOpen} variant="drawer-right" size="md" title="Tool call approvals" onClose={closeApprovals}>
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
						<span class="text-xs px-1.5 py-0.5 rounded font-medium capitalize" style="background-color: {riskColor(req.risk_level)}; color: var(--oo-fg-on-semantic);">
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
	<svelte:fragment slot="footer">
		<StopAllButton placement="dialog" large={$isPhone} />
	</svelte:fragment>
</Modal>
