/**
 * Auto-Tuner API client -- S110
 *
 * Typed API for inference parameter tuning: start/cancel sessions,
 * retrieve results, apply optimal parameters.
 */

import type {
	TunerStatus,
	TunerJob,
	TunerProfile,
	TunerResultsResponse,
	TunerRecommendationsResponse,
} from '../types';

const BASE = '/api/tuner';

/** Get auto-tuner status, config, and active jobs. */
export async function getTunerStatus(): Promise<TunerStatus> {
	const resp = await fetch(`${BASE}/status`);
	if (!resp.ok) throw new Error(`Failed to fetch tuner status: ${resp.status}`);
	return resp.json();
}

/** Start a tuning session for a model. Returns job info. */
export async function startTuning(modelName: string): Promise<TunerJob> {
	const resp = await fetch(`${BASE}/run`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ model_name: modelName }),
	});
	if (!resp.ok) {
		const err = await resp.json().catch(() => ({ detail: resp.statusText }));
		throw new Error(err.detail || `Failed to start tuning: ${resp.status}`);
	}
	return resp.json();
}

/** List all tuning results (per model). */
export async function listTunerResults(): Promise<TunerResultsResponse> {
	const resp = await fetch(`${BASE}/results`);
	if (!resp.ok) throw new Error(`Failed to list tuner results: ${resp.status}`);
	return resp.json();
}

/** Get best config for a specific model. */
export async function getTunerResult(modelName: string): Promise<TunerProfile> {
	const resp = await fetch(`${BASE}/results/${encodeURIComponent(modelName)}`);
	if (!resp.ok) throw new Error(`Failed to get tuner result: ${resp.status}`);
	return resp.json();
}

/** Apply tuned parameters as defaults for a model. */
export async function applyTunerResult(modelName: string): Promise<{ applied_params: Record<string, unknown> }> {
	const resp = await fetch(`${BASE}/apply/${encodeURIComponent(modelName)}`, {
		method: 'POST',
	});
	if (!resp.ok) throw new Error(`Failed to apply tuner result: ${resp.status}`);
	return resp.json();
}

/** Delete tuning data for a model. */
export async function deleteTunerResult(modelName: string): Promise<void> {
	const resp = await fetch(`${BASE}/results/${encodeURIComponent(modelName)}`, {
		method: 'DELETE',
	});
	if (!resp.ok) throw new Error(`Failed to delete tuner result: ${resp.status}`);
}

/** Cancel an active tuning session. */
export async function cancelTuning(modelName: string): Promise<void> {
	const resp = await fetch(`${BASE}/cancel/${encodeURIComponent(modelName)}`, {
		method: 'POST',
	});
	if (!resp.ok) throw new Error(`Failed to cancel tuning: ${resp.status}`);
}

/** Poll job status for a model. */
export async function getTunerJob(modelName: string): Promise<TunerJob> {
	const resp = await fetch(`${BASE}/job/${encodeURIComponent(modelName)}`);
	if (!resp.ok) throw new Error(`Failed to get tuner job: ${resp.status}`);
	return resp.json();
}

/** Get optimization recommendations for a tuned model (S112). */
export async function getTunerRecommendations(modelName: string): Promise<TunerRecommendationsResponse> {
	const resp = await fetch(`${BASE}/recommendations/${encodeURIComponent(modelName)}`);
	if (!resp.ok) throw new Error(`Failed to get recommendations: ${resp.status}`);
	return resp.json();
}
