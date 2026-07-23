/**
 * Benchmark Dashboard store.
 *
 * Manages benchmark run state, history, WebSocket progress,
 * model config, and suite/task metadata.
 */

import { writable, derived, get } from 'svelte/store';
import type {
	BenchmarkSuiteInfo,
	BenchmarkTaskInfo,
	BenchmarkRunSummaryInfo,
	BenchmarkRunDetailInfo,
	BenchmarkResultItem,
	BenchmarkProgressEvent,
	ModelRoleInfo,
} from '$lib/types';
import {
	listSuites,
	listTasks,
	startBenchmark,
	getBenchmarkStatus,
	cancelBenchmark,
	listRuns,
	getRunDetail,
	deleteRun,
	getConfigRoles,
	updateRoleAssignment,
	listInstalledModels,
	connectBenchmarkProgress,
} from '$lib/api/benchmark';

// -- Core state --

export const suites = writable<BenchmarkSuiteInfo[]>([]);
export const tasks = writable<BenchmarkTaskInfo[]>([]);
export const installedModels = writable<string[]>([]);

// -- Run state --

export const isRunning = writable(false);
export const currentRunId = writable<string | null>(null);
export const progress = writable<BenchmarkProgressEvent | null>(null);
export const liveResults = writable<BenchmarkResultItem[]>([]);
export const runError = writable<string | null>(null);

// -- History --

export const runs = writable<BenchmarkRunSummaryInfo[]>([]);
export const runsTotal = writable(0);
export const selectedRun = writable<BenchmarkRunDetailInfo | null>(null);

// -- Model config --

export const roles = writable<ModelRoleInfo[]>([]);

// -- UI state --

export const benchmarkLoading = writable(false);
export const benchmarkError = writable<string | null>(null);

// -- WebSocket --

let ws: WebSocket | null = null;

function connectWs() {
	if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
		return;
	}

	try {
		ws = connectBenchmarkProgress();
	} catch {
		return;
	}

	ws.onmessage = (event) => {
		try {
			const msg = JSON.parse(event.data);
			switch (msg.type) {
				case 'status':
					isRunning.set(msg.data?.running ?? false);
					currentRunId.set(msg.data?.run_id ?? null);
					break;
				case 'progress':
					progress.set(msg.data as BenchmarkProgressEvent);
					break;
				case 'result':
					liveResults.update((r) => [...r, msg.data as BenchmarkResultItem]);
					break;
				case 'completed':
				case 'cancelled':
					isRunning.set(false);
					progress.set(null);
					// Refresh history
					loadRuns();
					break;
				case 'error':
					isRunning.set(false);
					runError.set(msg.data?.message ?? 'Benchmark error');
					break;
				case 'heartbeat':
					break;
			}
		} catch {
			// Ignore parse errors
		}
	};

	ws.onclose = () => {
		ws = null;
	};

	ws.onerror = () => {
		ws = null;
	};
}

export function disconnectWs() {
	if (ws) {
		try {
			ws.close();
		} catch {
			// Ignore
		}
		ws = null;
	}
}

// -- Actions --

export async function loadSuites() {
	try {
		const data = await listSuites();
		suites.set(data.suites);
	} catch {
		// Non-critical
	}
}

export async function loadTasks() {
	try {
		const data = await listTasks();
		tasks.set(data.tasks);
	} catch {
		// Non-critical
	}
}

export async function loadInstalledModels() {
	try {
		const data = await listInstalledModels();
		installedModels.set(data.models);
	} catch {
		// Non-critical
	}
}

export async function loadRuns(limit = 20, offset = 0) {
	benchmarkLoading.set(true);
	try {
		const data = await listRuns('llm', limit, offset);
		runs.set(data.runs);
		runsTotal.set(data.total);
	} catch (e) {
		benchmarkError.set(e instanceof Error ? e.message : 'Failed to load runs');
	} finally {
		benchmarkLoading.set(false);
	}
}

export async function loadRunDetail(runId: string) {
	benchmarkLoading.set(true);
	try {
		const data = await getRunDetail(runId);
		selectedRun.set(data);
	} catch (e) {
		benchmarkError.set(e instanceof Error ? e.message : 'Failed to load run');
	} finally {
		benchmarkLoading.set(false);
	}
}

export async function removeRun(runId: string) {
	try {
		await deleteRun(runId);
		runs.update((r) => r.filter((run) => run.id !== runId));
		runsTotal.update((t) => Math.max(0, t - 1));
		if (get(selectedRun)?.id === runId) {
			selectedRun.set(null);
		}
	} catch (e) {
		benchmarkError.set(e instanceof Error ? e.message : 'Failed to delete run');
	}
}

export async function launchBenchmark(opts: {
	models: string[];
	tasks?: string[];
	suiteId?: string;
	temperature?: number;
	timeout?: number;
	maxTokens?: number;
}) {
	benchmarkError.set(null);
	runError.set(null);
	liveResults.set([]);
	progress.set(null);

	try {
		const resp = await startBenchmark({
			models: opts.models,
			tasks: opts.tasks,
			suite_id: opts.suiteId,
			temperature: opts.temperature,
			timeout: opts.timeout,
			max_tokens: opts.maxTokens,
		});
		isRunning.set(true);
		currentRunId.set(resp.run_id);
		connectWs();
	} catch (e) {
		benchmarkError.set(e instanceof Error ? e.message : 'Failed to start benchmark');
	}
}

export async function stopBenchmark() {
	try {
		await cancelBenchmark();
	} catch (e) {
		benchmarkError.set(e instanceof Error ? e.message : 'Failed to cancel');
	}
}

export async function checkStatus() {
	try {
		const data = await getBenchmarkStatus();
		isRunning.set(data.running);
		currentRunId.set(data.run_id);
		if (data.running) {
			connectWs();
		}
	} catch {
		// Non-critical
	}
}

// -- Model config actions --

export async function loadRoles() {
	try {
		const data = await getConfigRoles();
		roles.set(data.roles);
		installedModels.set(data.installed_models);
	} catch {
		// Non-critical
	}
}

export async function saveRole(role: string, assignment: { primary?: string; fast?: string; quality?: string }) {
	try {
		await updateRoleAssignment(role, assignment);
		await loadRoles();
	} catch (e) {
		benchmarkError.set(e instanceof Error ? e.message : 'Failed to save role');
	}
}

// -- Init --

export async function initBenchmarkStore() {
	await Promise.all([loadSuites(), loadTasks(), loadInstalledModels(), checkStatus(), loadRuns()]);
}
