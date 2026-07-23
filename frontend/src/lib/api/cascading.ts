/**
 * Cascading Inference API client
 *
 * Typed API for cascading inference status, config, and test endpoints.
 */

import type {
	CascadeStatus,
	CascadeConfigUpdate,
	CascadeTestResult,
} from '../types';

const BASE = '/api/cascading';

/** Get cascading inference status. */
export async function getCascadingStatus(): Promise<CascadeStatus> {
	const resp = await fetch(`${BASE}/status`);
	if (!resp.ok) throw new Error(`Failed to fetch cascading status: ${resp.status}`);
	return resp.json();
}

/** Update cascading inference configuration. */
export async function updateCascadingConfig(update: CascadeConfigUpdate): Promise<CascadeStatus> {
	const resp = await fetch(`${BASE}/config`, {
		method: 'PUT',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(update),
	});
	if (!resp.ok) throw new Error(`Failed to update cascading config: ${resp.status}`);
	return resp.json();
}

/** Run a test cascade on a sample query. */
export async function testCascade(query: string, taskType?: string): Promise<CascadeTestResult> {
	const resp = await fetch(`${BASE}/test`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ query, task_type: taskType }),
	});
	if (!resp.ok) throw new Error(`Cascade test failed: ${resp.status}`);
	return resp.json();
}
