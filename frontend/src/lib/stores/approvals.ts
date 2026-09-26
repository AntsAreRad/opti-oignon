/**
 * The tool-call approvals the agent waits on, as the shell shows them.
 *
 * The approvals drawer (components/chat/ToolCallApprovalDrawer.svelte),
 * mounted once by the shell, reads the pending requests and writes their
 * count here; the pill in the status card and in the phone header reads the
 * count, and opens the drawer through `openApprovals()`. The pill appears
 * only while approvals are pending.
 */

import { writable } from 'svelte/store';

/** How many tool calls wait for an approval. */
export const pendingApprovals = writable<number>(0);

/** Whether the approvals drawer is open. */
export const approvalsOpen = writable<boolean>(false);

/** Opens the approvals drawer. */
export function openApprovals(): void {
	approvalsOpen.set(true);
}

/** Closes the approvals drawer. */
export function closeApprovals(): void {
	approvalsOpen.set(false);
}
