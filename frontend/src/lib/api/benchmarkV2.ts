/**
 * Benchmark V2 API client.
 *
 * Typed API for autonomous quality evaluation: profiles, runs, progress,
 * results, comparison, history, LLM-as-Judge, leaderboard, head-to-head,
 * trends, recommendations, export, custom profile CRUD, question preview,
 * auto-trigger management, and test poll.
 */

import type {
	BenchmarkV2ProfilesResponse,
	BenchmarkV2RunRequest,
	BenchmarkV2RunStarted,
	BenchmarkV2Progress,
	BenchmarkV2Results,
	BenchmarkV2CompareResponse,
	BenchmarkV2HistoryResponse,
	BenchmarkV2LeaderboardResponse,
	BenchmarkV2HeadToHeadResponse,
	BenchmarkV2TrendResponse,
	BenchmarkV2RecommendationsResponse,
	BenchmarkV2ApplyResponse,
	BenchmarkV2CustomProfile,
	BenchmarkV2CustomProfileCreate,
	BenchmarkV2CustomProfileUpdate,
	BenchmarkV2CustomProfilesListResponse,
	BenchmarkV2QuestionPreview,
	BenchmarkV2AutoTriggerStatus,
	BenchmarkV2AutoTriggerConfig,
	BenchmarkV2AutoTriggerConfigUpdate,
	BenchmarkV2AutoTriggerEventsResponse,
	BenchmarkV2AutoTriggerTestPollResponse,
} from '../types';

const BASE = '/api/benchmark/v2';

/** List available benchmark profiles. */
export async function getProfiles(): Promise<BenchmarkV2ProfilesResponse> {
	const resp = await fetch(`${BASE}/profiles`);
	if (!resp.ok) throw new Error(`Failed to fetch profiles: ${resp.status}`);
	return resp.json();
}

/** Start a benchmark run (optionally with LLM-as-Judge). */
export async function startRun(request: BenchmarkV2RunRequest): Promise<BenchmarkV2RunStarted> {
	const resp = await fetch(`${BASE}/run`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(request),
	});
	if (!resp.ok) throw new Error(`Failed to start run: ${resp.status}`);
	return resp.json();
}

/** Poll run progress. */
export async function getRunStatus(runId: string): Promise<BenchmarkV2Progress> {
	const resp = await fetch(`${BASE}/status/${encodeURIComponent(runId)}`);
	if (!resp.ok) throw new Error(`Failed to fetch status: ${resp.status}`);
	return resp.json();
}

/** Cancel a running benchmark. */
export async function cancelRun(runId: string): Promise<{ run_id: string; status: string }> {
	const resp = await fetch(`${BASE}/cancel/${encodeURIComponent(runId)}`, {
		method: 'POST',
	});
	if (!resp.ok) throw new Error(`Failed to cancel run: ${resp.status}`);
	return resp.json();
}

/** Get detailed results for a completed run (includes judge scores). */
export async function getRunResults(runId: string): Promise<BenchmarkV2Results> {
	const resp = await fetch(`${BASE}/results/${encodeURIComponent(runId)}`);
	if (!resp.ok) throw new Error(`Failed to fetch results: ${resp.status}`);
	return resp.json();
}

/** Compare models across runs. */
export async function compareModels(
	models?: string[],
	profile?: string,
	limit: number = 10,
): Promise<BenchmarkV2CompareResponse> {
	const params = new URLSearchParams();
	if (models && models.length > 0) params.set('models', models.join(','));
	if (profile) params.set('profile', profile);
	params.set('limit', String(limit));
	const resp = await fetch(`${BASE}/compare?${params.toString()}`);
	if (!resp.ok) throw new Error(`Failed to compare models: ${resp.status}`);
	return resp.json();
}

/** Get historical benchmark runs. */
export async function getHistory(
	limit: number = 50,
	profile?: string,
	model?: string,
): Promise<BenchmarkV2HistoryResponse> {
	const params = new URLSearchParams();
	params.set('limit', String(limit));
	if (profile) params.set('profile', profile);
	if (model) params.set('model', model);
	const resp = await fetch(`${BASE}/history?${params.toString()}`);
	if (!resp.ok) throw new Error(`Failed to fetch history: ${resp.status}`);
	return resp.json();
}

/**
 * Poll a run until completion or failure.
 * Calls onProgress at each interval, returns final results.
 */
export async function pollUntilDone(
	runId: string,
	onProgress?: (progress: BenchmarkV2Progress) => void,
	intervalMs: number = 2000,
): Promise<BenchmarkV2Results> {
	while (true) {
		const progress = await getRunStatus(runId);
		if (onProgress) onProgress(progress);

		if (progress.status === 'completed' || progress.status === 'failed' || progress.status === 'cancelled') {
			return getRunResults(runId);
		}
		await new Promise((r) => setTimeout(r, intervalMs));
	}
}

// -- endpoints --

/** Get ranked model leaderboard. */
export async function getLeaderboard(
	profile?: string,
	limit: number = 20,
): Promise<BenchmarkV2LeaderboardResponse> {
	const params = new URLSearchParams();
	if (profile) params.set('profile', profile);
	params.set('limit', String(limit));
	const resp = await fetch(`${BASE}/leaderboard?${params.toString()}`);
	if (!resp.ok) throw new Error(`Failed to fetch leaderboard: ${resp.status}`);
	return resp.json();
}

/** Head-to-head comparison of two models. */
export async function getHeadToHead(
	modelA: string,
	modelB: string,
	profile?: string,
): Promise<BenchmarkV2HeadToHeadResponse> {
	const params = new URLSearchParams();
	params.set('model_a', modelA);
	params.set('model_b', modelB);
	if (profile) params.set('profile', profile);
	const resp = await fetch(`${BASE}/head-to-head?${params.toString()}`);
	if (!resp.ok) throw new Error(`Failed to fetch head-to-head: ${resp.status}`);
	return resp.json();
}

/** Get temporal performance data for a model. */
export async function getTrends(
	model: string,
	limit: number = 50,
	profile?: string,
): Promise<BenchmarkV2TrendResponse> {
	const params = new URLSearchParams();
	params.set('model', model);
	params.set('limit', String(limit));
	if (profile) params.set('profile', profile);
	const resp = await fetch(`${BASE}/trends?${params.toString()}`);
	if (!resp.ok) throw new Error(`Failed to fetch trends: ${resp.status}`);
	return resp.json();
}

/** Get current model recommendations. */
export async function getRecommendations(): Promise<BenchmarkV2RecommendationsResponse> {
	const resp = await fetch(`${BASE}/recommendations`);
	if (!resp.ok) throw new Error(`Failed to fetch recommendations: ${resp.status}`);
	return resp.json();
}

/** Apply recommendations to smart router. */
export async function applyRecommendations(): Promise<BenchmarkV2ApplyResponse> {
	const resp = await fetch(`${BASE}/recommendations/apply`, {
		method: 'POST',
	});
	if (!resp.ok) throw new Error(`Failed to apply recommendations: ${resp.status}`);
	return resp.json();
}

/** Export run results as JSON. Returns a Blob for download. */
export async function exportJson(runId: string): Promise<Blob> {
	const resp = await fetch(`${BASE}/export/${encodeURIComponent(runId)}?format=json`);
	if (!resp.ok) throw new Error(`Failed to export JSON: ${resp.status}`);
	return resp.blob();
}

/** Export run results as CSV. Returns a Blob for download. */
export async function exportCsv(runId: string): Promise<Blob> {
	const resp = await fetch(`${BASE}/export/${encodeURIComponent(runId)}?format=csv`);
	if (!resp.ok) throw new Error(`Failed to export CSV: ${resp.status}`);
	return resp.blob();
}

/** Helper: trigger a file download from a Blob. */
export function downloadBlob(blob: Blob, filename: string): void {
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
	URL.revokeObjectURL(url);
}

// -- — Custom Profile endpoints --

/** List all custom profiles. */
export async function getCustomProfiles(): Promise<BenchmarkV2CustomProfilesListResponse> {
	const resp = await fetch(`${BASE}/profiles/custom`);
	if (!resp.ok) throw new Error(`Failed to fetch custom profiles: ${resp.status}`);
	return resp.json();
}

/** Create a custom profile. */
export async function createCustomProfile(
	data: BenchmarkV2CustomProfileCreate,
): Promise<BenchmarkV2CustomProfile> {
	const resp = await fetch(`${BASE}/profiles/custom`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(data),
	});
	if (!resp.ok) throw new Error(`Failed to create custom profile: ${resp.status}`);
	return resp.json();
}

/** Update a custom profile. */
export async function updateCustomProfile(
	profileId: string,
	data: BenchmarkV2CustomProfileUpdate,
): Promise<BenchmarkV2CustomProfile> {
	const resp = await fetch(`${BASE}/profiles/custom/${encodeURIComponent(profileId)}`, {
		method: 'PUT',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(data),
	});
	if (!resp.ok) throw new Error(`Failed to update custom profile: ${resp.status}`);
	return resp.json();
}

/** Delete a custom profile. */
export async function deleteCustomProfile(
	profileId: string,
): Promise<{ profile_id: string; deleted: boolean }> {
	const resp = await fetch(`${BASE}/profiles/custom/${encodeURIComponent(profileId)}`, {
		method: 'DELETE',
	});
	if (!resp.ok) throw new Error(`Failed to delete custom profile: ${resp.status}`);
	return resp.json();
}

/** Preview question counts for given categories. */
export async function previewProfileQuestions(
	categories: string[],
): Promise<BenchmarkV2QuestionPreview> {
	const resp = await fetch(`${BASE}/profiles/preview`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(categories),
	});
	if (!resp.ok) throw new Error(`Failed to preview questions: ${resp.status}`);
	return resp.json();
}

// -- — Auto-Trigger endpoints --

/** Get auto-trigger status. */
export async function getAutoTriggerStatus(): Promise<BenchmarkV2AutoTriggerStatus> {
	const resp = await fetch(`${BASE}/auto-trigger/status`);
	if (!resp.ok) throw new Error(`Failed to fetch auto-trigger status: ${resp.status}`);
	return resp.json();
}

/** Get auto-trigger configuration. */
export async function getAutoTriggerConfig(): Promise<BenchmarkV2AutoTriggerConfig> {
	const resp = await fetch(`${BASE}/auto-trigger/config`);
	if (!resp.ok) throw new Error(`Failed to fetch auto-trigger config: ${resp.status}`);
	return resp.json();
}

/** Update auto-trigger configuration. */
export async function updateAutoTriggerConfig(
	data: BenchmarkV2AutoTriggerConfigUpdate,
): Promise<BenchmarkV2AutoTriggerConfig> {
	const resp = await fetch(`${BASE}/auto-trigger/config`, {
		method: 'PUT',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(data),
	});
	if (!resp.ok) throw new Error(`Failed to update auto-trigger config: ${resp.status}`);
	return resp.json();
}

/** Enable auto-trigger. */
export async function enableAutoTrigger(): Promise<{ enabled: boolean; running: boolean }> {
	const resp = await fetch(`${BASE}/auto-trigger/enable`, { method: 'POST' });
	if (!resp.ok) throw new Error(`Failed to enable auto-trigger: ${resp.status}`);
	return resp.json();
}

/** Disable auto-trigger. */
export async function disableAutoTrigger(): Promise<{ enabled: boolean; running: boolean }> {
	const resp = await fetch(`${BASE}/auto-trigger/disable`, { method: 'POST' });
	if (!resp.ok) throw new Error(`Failed to disable auto-trigger: ${resp.status}`);
	return resp.json();
}

/** Get recent auto-trigger events. */
export async function getAutoTriggerEvents(): Promise<BenchmarkV2AutoTriggerEventsResponse> {
	const resp = await fetch(`${BASE}/auto-trigger/events`);
	if (!resp.ok) throw new Error(`Failed to fetch auto-trigger events: ${resp.status}`);
	return resp.json();
}

/** Reset auto-trigger model snapshot. */
export async function resetAutoTriggerSnapshot(): Promise<{ reset: boolean }> {
	const resp = await fetch(`${BASE}/auto-trigger/reset`, { method: 'POST' });
	if (!resp.ok) throw new Error(`Failed to reset auto-trigger: ${resp.status}`);
	return resp.json();
}

// -- — Test Poll --

/** Run a single poll without triggering (test connection). */
export async function testPollAutoTrigger(): Promise<BenchmarkV2AutoTriggerTestPollResponse> {
	const resp = await fetch(`${BASE}/auto-trigger/test-poll`, { method: 'POST' });
	if (!resp.ok) throw new Error(`Failed to test poll: ${resp.status}`);
	return resp.json();
}
