/**
 * Typed API functions for health and benchmark endpoints.
 *
 * Provides health dashboard, module status, and benchmark operations.
 */

import { apiGet, apiPost } from './client';
import type { HealthResponse, HealthDashboard, BenchmarkResultSchema } from '$lib/types';

/** Basic health check (status, version, modules). */
export async function getHealth(): Promise<HealthResponse> {
	return apiGet<HealthResponse>('/api/health');
}

/** Full dashboard: modules, stats, cache, warmup. */
export async function getHealthDashboard(): Promise<HealthDashboard> {
	return apiGet<HealthDashboard>('/api/health/dashboard');
}

/** Run all available benchmarks. */
export async function runBenchmarks(iterations: number = 200): Promise<Record<string, BenchmarkResultSchema>> {
	return apiPost<Record<string, BenchmarkResultSchema>>('/api/health/benchmarks', undefined);
}

/** Run a specific benchmark by name. */
export async function runBenchmark(name: string): Promise<BenchmarkResultSchema> {
	return apiPost<BenchmarkResultSchema>(`/api/health/benchmarks/${encodeURIComponent(name)}`);
}
