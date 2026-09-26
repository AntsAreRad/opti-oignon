/**
 * Cascading Inference API client
 *
 * Typed API for cascading inference status, config, and test endpoints,
 * through the API client (its authentication, CSRF header and error
 * handling).
 */

import { apiGet, apiPost, apiPut } from './client';
import type {
	CascadeStatus,
	CascadeConfigUpdate,
	CascadeTestResult,
} from '../types';

const BASE = '/api/cascading';

/** Get cascading inference status. */
export async function getCascadingStatus(): Promise<CascadeStatus> {
	return apiGet<CascadeStatus>(`${BASE}/status`);
}

/** Update cascading inference configuration (saved by the server). */
export async function updateCascadingConfig(update: CascadeConfigUpdate): Promise<CascadeStatus> {
	return apiPut<CascadeStatus>(`${BASE}/config`, update);
}

/** Run a test cascade on a sample query. */
export async function testCascade(query: string, taskType?: string): Promise<CascadeTestResult> {
	return apiPost<CascadeTestResult>(`${BASE}/test`, { query, task_type: taskType });
}
