/**
 * API client for the Coding Agent.
 *
 * Provides typed functions for all coding agent endpoints:
 * start, plan, step, status, diff, approve, abort.
 * Also provides WebSocket connection for live progress.
 */

import { apiGet, apiPost } from './client';
import type {
	CodingTaskRequest,
	CodingPlanResponse,
	CodingCheckpointRequest,
	CodingStepResponse,
	CodingStatusResponse,
	CodingDiffResponse,
	CodingApplyRequest,
	CodingApplyResponse
} from '../types';

/** Start a new coding task. */
export async function startCodingTask(
	request: CodingTaskRequest
): Promise<CodingStatusResponse> {
	return apiPost('/api/coding/start', request);
}

/** Generate a plan (no body) or respond to plan checkpoint. */
export async function codingPlan(
	checkpoint?: CodingCheckpointRequest
): Promise<CodingPlanResponse> {
	return apiPost('/api/coding/plan', checkpoint ?? {});
}

/** Execute the next step in the plan. */
export async function executeNextStep(): Promise<CodingStepResponse> {
	return apiPost('/api/coding/step');
}

/** Get current coding agent status. */
export async function getCodingStatus(): Promise<CodingStatusResponse> {
	return apiGet('/api/coding/status');
}

/** Generate and return diffs of all changes. */
export async function getCodingDiff(): Promise<CodingDiffResponse> {
	return apiGet('/api/coding/diff');
}

/** Approve and apply sandbox changes to real filesystem. */
export async function approveCodingChanges(
	request?: CodingApplyRequest
): Promise<CodingApplyResponse> {
	return apiPost('/api/coding/approve', request ?? {});
}

/** Abort the current coding task. */
export async function abortCodingTask(): Promise<{
	aborted: boolean;
	cleanup_success: boolean;
	task_id: string;
}> {
	return apiPost('/api/coding/abort');
}

/** Connect to the coding agent live progress WebSocket. */
export function connectCodingWebSocket(
	onEvent: (event: Record<string, unknown>) => void,
	onError?: (error: Event) => void,
	onClose?: () => void
): WebSocket | null {
	try {
		const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
		const host = window.location.hostname;
		const port = '8001';
		const url = `${protocol}//${host}:${port}/ws/coding/live`;

		const ws = new WebSocket(url);

		ws.onmessage = (msg) => {
			try {
				const data = JSON.parse(msg.data);
				onEvent(data);
			} catch {
				// Ignore non-JSON messages
			}
		};

		ws.onerror = (err) => {
			if (onError) onError(err);
		};

		ws.onclose = () => {
			if (onClose) onClose();
		};

		return ws;
	} catch {
		return null;
	}
}
