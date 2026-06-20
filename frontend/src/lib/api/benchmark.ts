/**
 * API functions for the Benchmark Dashboard (S60).
 *
 * Covers suite listing, benchmark execution, history management,
 * model config CRUD, and WebSocket progress connection.
 */

import { apiGet, apiPost, apiPut, apiDelete, wsUrl } from './client';
import type {
	BenchmarkSuiteInfo,
	BenchmarkSuiteDetail,
	BenchmarkTaskInfo,
	BenchmarkRunSummaryInfo,
	BenchmarkRunDetailInfo,
	BenchmarkComparisonInfo,
	BenchmarkModelTrendInfo,
	BenchmarkRunRequest,
	ModelConfigInfo,
	ModelRoleInfo,
} from '$lib/types';

// -- Suites --

export function listSuites(): Promise<{ suites: BenchmarkSuiteInfo[] }> {
	return apiGet('/api/benchmark/suites');
}

export function getSuiteDetail(suiteId: string): Promise<BenchmarkSuiteDetail> {
	return apiGet(`/api/benchmark/suites/${suiteId}`);
}

export function listTasks(): Promise<{ tasks: BenchmarkTaskInfo[] }> {
	return apiGet('/api/benchmark/tasks');
}

// -- LLM Benchmark Run --

export function startBenchmark(request: BenchmarkRunRequest): Promise<{
	run_id: string;
	status: string;
	models: string[];
	tasks: string[];
	total_tests: number;
}> {
	return apiPost('/api/benchmark/llm/run', request);
}

export function getBenchmarkStatus(): Promise<{
	running: boolean;
	run_id: string | null;
	status: string;
}> {
	return apiGet('/api/benchmark/llm/status');
}

export function cancelBenchmark(): Promise<{ status: string; run_id: string }> {
	return apiPost('/api/benchmark/llm/cancel');
}

export function submitUserScore(
	runId: string,
	model: string,
	task: string,
	score: number
): Promise<{ status: string; final_score: number }> {
	return apiPost('/api/benchmark/llm/user-score', {
		run_id: runId,
		model,
		task,
		score,
	});
}

// -- History --

export function listRuns(
	runType: string = 'llm',
	limit: number = 20,
	offset: number = 0
): Promise<{ runs: BenchmarkRunSummaryInfo[]; total: number }> {
	return apiGet('/api/benchmark/runs', {
		run_type: runType,
		limit: String(limit),
		offset: String(offset),
	});
}

export function getRunDetail(runId: string): Promise<BenchmarkRunDetailInfo> {
	return apiGet(`/api/benchmark/runs/${runId}`);
}

export function deleteRun(runId: string): Promise<{ status: string }> {
	return apiDelete(`/api/benchmark/runs/${runId}`);
}

export function compareRuns(runIds: string[]): Promise<BenchmarkComparisonInfo> {
	return apiGet('/api/benchmark/compare', { runs: runIds.join(',') });
}

export function getModelTrends(model: string, lastN: number = 10): Promise<BenchmarkModelTrendInfo> {
	return apiGet(`/api/benchmark/trends/${encodeURIComponent(model)}`, {
		last_n: String(lastN),
	});
}

// -- Model Config --

export function getModelsConfig(): Promise<ModelConfigInfo> {
	return apiGet('/api/benchmark/models/config');
}

export function saveModelsConfig(config: Record<string, unknown>): Promise<{ status: string }> {
	return apiPut('/api/benchmark/models/config', { config });
}

export function getConfigRoles(): Promise<{ roles: ModelRoleInfo[]; installed_models: string[] }> {
	return apiGet('/api/benchmark/models/config/roles');
}

export function updateRoleAssignment(
	role: string,
	assignment: { primary?: string; fast?: string; quality?: string }
): Promise<{ status: string }> {
	return apiPut(`/api/benchmark/models/config/roles/${role}`, assignment);
}

export function validateConfig(config?: Record<string, unknown>): Promise<{
	valid: boolean;
	warnings: Array<{ role: string; priority: string; model: string; issue: string }>;
	installed_count: number;
}> {
	return apiPost('/api/benchmark/models/config/validate', { config: config || {} });
}

export function listInstalledModels(): Promise<{ models: string[]; count: number }> {
	return apiGet('/api/benchmark/models/installed');
}

// -- WebSocket --

export function connectBenchmarkProgress(): WebSocket {
	const url = wsUrl('/api/benchmark/llm/progress');
	return new WebSocket(url);
}
