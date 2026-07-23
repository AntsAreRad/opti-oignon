/**
 * API client for the sandboxed agent loop (Theme 3 / Odysseus Core).
 *
 * Mirrors the backend AgentEvent contract emitted by opti_oignon/agent/loop.py
 * (round_start / model_output / tool_result / done / error / verifier_output)
 * and exposes a live event-stream subscription plus run-control endpoints. The
 * tool-call approval surface is reused verbatim from the existing
 * /api/security/tool-approval/* API (toolCallApproval.ts): the agent panel
 * polls and resolves Bulbe approvals through it rather than a duplicate route.
 *
 * The live event-stream endpoint and the run-control endpoints are wired to the
 * backend during the end-to-end integration; this client defines the
 * contract the agent panel consumes.
 */

import { apiGet, apiPost, ReconnectingWebSocket, wsUrl } from './client';

// Re-export the existing approval surface so the agent panel has a single
// import for both the event stream and the Bulbe approval prompts.
export {
	getPendingApprovals,
	approveToolCall,
	denyToolCall,
	getApprovalAudit
} from './toolCallApproval';
export type { PendingApproval, PendingResponse, ApprovalResult } from './toolCallApproval';

/** The event kinds the loop emits, mirroring loop.AgentEvent.kind. */
export type AgentEventKind =
	| 'round_start'
	| 'model_output'
	| 'tool_result'
	| 'done'
	| 'error'
	| 'verifier_output';

/** The full set of event kinds, in loop order. */
export const AGENT_EVENT_KINDS: AgentEventKind[] = [
	'round_start',
	'model_output',
	'tool_result',
	'done',
	'error',
	'verifier_output'
];

/** A normalised tool result, mirroring dispatch.DispatchResult.to_dict(). */
export interface AgentToolResult {
	tool_name: string;
	executed: boolean;
	reason: string;
	observation: string;
	source: string;
	mode: string;
}

/** One observable step from the loop, mirroring loop.AgentEvent. */
export interface AgentEvent {
	kind: AgentEventKind;
	round: number;
	data: Record<string, unknown>;
}

/** Snapshot of the current agent run. */
export interface AgentRunStatus {
	running: boolean;
	rounds: number;
	stop_reason: string;
}

/** Parse a raw stream payload into an AgentEvent, or null when malformed. */
export function parseAgentEvent(raw: string): AgentEvent | null {
	try {
		const obj = JSON.parse(raw) as Partial<AgentEvent>;
		if (!obj || typeof obj.kind !== 'string') return null;
		return {
			kind: obj.kind as AgentEventKind,
			round: typeof obj.round === 'number' ? obj.round : 0,
			data: (obj.data as Record<string, unknown>) ?? {}
		};
	} catch {
		return null;
	}
}

/**
 * Subscribe to the live agent event stream over a reconnecting WebSocket.
 *
 * Each well-formed message is parsed into an AgentEvent and passed to onEvent;
 * malformed frames are ignored. Returns the socket so the caller can close it.
 */
export function connectAgentStream(
	onEvent: (event: AgentEvent) => void,
	onError?: (error: Event) => void,
	onClose?: () => void
): ReconnectingWebSocket {
	const socket = new ReconnectingWebSocket(wsUrl('/api/agent/stream'));
	socket.onmessage = (msg: MessageEvent) => {
		const event = parseAgentEvent(typeof msg.data === 'string' ? msg.data : '');
		if (event) onEvent(event);
	};
	if (onError) socket.onerror = onError;
	if (onClose) socket.onclose = onClose;
	return socket;
}

/** Get the current agent run status. */
export async function getAgentStatus(): Promise<AgentRunStatus> {
	return apiGet('/api/agent/status');
}

/** Cancel the current agent run. */
export async function cancelAgentRun(): Promise<{ cancelled: boolean }> {
	return apiPost('/api/agent/cancel');
}
