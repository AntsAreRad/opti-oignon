/**
 * Speculative Generation API client -- S70
 *
 * Typed API for speculative generation status, config, and test endpoints.
 */

import type {
	SpeculativeStatus,
	SpeculativeConfigUpdate,
	SpeculativeTestResult,
} from '../types';

const BASE = '/api/speculative';

/** Get speculative generation status. */
export async function getSpeculativeStatus(): Promise<SpeculativeStatus> {
	const resp = await fetch(`${BASE}/status`);
	if (!resp.ok) throw new Error(`Failed to fetch speculative status: ${resp.status}`);
	return resp.json();
}

/** Update speculative generation configuration. */
export async function updateSpeculativeConfig(update: SpeculativeConfigUpdate): Promise<SpeculativeStatus> {
	const resp = await fetch(`${BASE}/config`, {
		method: 'PUT',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(update),
	});
	if (!resp.ok) throw new Error(`Failed to update speculative config: ${resp.status}`);
	return resp.json();
}

/** Run a test speculative generation on a sample query. */
export async function testSpeculative(query: string, taskType?: string): Promise<SpeculativeTestResult> {
	const resp = await fetch(`${BASE}/test`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ query, task_type: taskType }),
	});
	if (!resp.ok) throw new Error(`Speculative test failed: ${resp.status}`);
	return resp.json();
}
