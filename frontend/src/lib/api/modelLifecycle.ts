/**
 * Model Lifecycle API client -- S112
 *
 * Typed API for model pull/delete/update operations,
 * alias management, and stale model detection.
 */

import type {
	PullJob,
	ModelUpdateInfo,
	ModelLifecycleStatus,
	ModelEntry,
} from '../types';

const BASE = '/api/model-lifecycle';

/** Get lifecycle manager status. */
export async function getLifecycleStatus(): Promise<ModelLifecycleStatus> {
	const resp = await fetch(`${BASE}/status`);
	if (!resp.ok) throw new Error(`Failed to fetch lifecycle status: ${resp.status}`);
	return resp.json();
}

/** Start pulling a model. Returns job info. */
export async function startModelPull(modelName: string): Promise<PullJob> {
	const resp = await fetch(`${BASE}/pull`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ model_name: modelName }),
	});
	if (!resp.ok) {
		const err = await resp.json().catch(() => ({ detail: resp.statusText }));
		throw new Error(err.detail || `Failed to start pull: ${resp.status}`);
	}
	return resp.json();
}

/** Get pull progress for a job. */
export async function getPullProgress(jobId: string): Promise<PullJob> {
	const resp = await fetch(`${BASE}/pull-progress/${encodeURIComponent(jobId)}`);
	if (!resp.ok) throw new Error(`Failed to get pull progress: ${resp.status}`);
	return resp.json();
}

/** Cancel an active pull job. */
export async function cancelPull(jobId: string): Promise<void> {
	const resp = await fetch(`${BASE}/pull-cancel/${encodeURIComponent(jobId)}`, {
		method: 'POST',
	});
	if (!resp.ok) throw new Error(`Failed to cancel pull: ${resp.status}`);
}

/** List all pull jobs. */
export async function listPullJobs(): Promise<{ jobs: PullJob[]; count: number }> {
	const resp = await fetch(`${BASE}/pull-jobs`);
	if (!resp.ok) throw new Error(`Failed to list pull jobs: ${resp.status}`);
	return resp.json();
}

/** Delete a locally stored model. */
export async function deleteModel(modelName: string): Promise<{ success: boolean; model: string; error?: string }> {
	const resp = await fetch(`${BASE}/models/${encodeURIComponent(modelName)}`, {
		method: 'DELETE',
	});
	if (!resp.ok) {
		const err = await resp.json().catch(() => ({ detail: resp.statusText }));
		throw new Error(err.detail || `Failed to delete model: ${resp.status}`);
	}
	return resp.json();
}

/** Check for model updates. Empty array = check all. */
export async function checkModelUpdates(modelNames: string[] = []): Promise<{ results: ModelUpdateInfo[] }> {
	const resp = await fetch(`${BASE}/update-check`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ model_names: modelNames }),
	});
	if (!resp.ok) throw new Error(`Failed to check updates: ${resp.status}`);
	return resp.json();
}

/** List all model aliases. */
export async function listAliases(): Promise<{ aliases: Record<string, string> }> {
	const resp = await fetch(`${BASE}/aliases`);
	if (!resp.ok) throw new Error(`Failed to list aliases: ${resp.status}`);
	return resp.json();
}

/** Create or update an alias. */
export async function setAlias(alias: string, modelName: string): Promise<void> {
	const resp = await fetch(`${BASE}/aliases`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ alias, model_name: modelName }),
	});
	if (!resp.ok) throw new Error(`Failed to set alias: ${resp.status}`);
}

/** Remove an alias. */
export async function removeAlias(alias: string): Promise<void> {
	const resp = await fetch(`${BASE}/aliases/${encodeURIComponent(alias)}`, {
		method: 'DELETE',
	});
	if (!resp.ok) throw new Error(`Failed to remove alias: ${resp.status}`);
}

/** Detect stale models. */
export async function detectStaleModels(): Promise<{ models: ModelEntry[]; threshold_days: number }> {
	const resp = await fetch(`${BASE}/stale`);
	if (!resp.ok) throw new Error(`Failed to detect stale models: ${resp.status}`);
	return resp.json();
}

/** Get detailed info about a specific model. */
export async function getModelDetail(modelName: string): Promise<Record<string, unknown>> {
	const resp = await fetch(`${BASE}/models/${encodeURIComponent(modelName)}`);
	if (!resp.ok) throw new Error(`Failed to get model detail: ${resp.status}`);
	return resp.json();
}
