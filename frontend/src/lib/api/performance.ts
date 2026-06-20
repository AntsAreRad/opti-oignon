/**
 * Performance Dashboard API client -- S72 Real-Time Performance Dashboard.
 *
 * Typed functions for throughput, latency, drift detection,
 * recommendations, utilization, and metric history.
 */

import { apiGet, apiPost } from './client';
import type {
	PerformanceSummary,
	PerformanceLatencyStats,
	PerformanceDriftResponse,
	PerformanceRecommendationsResponse,
	PerformanceHistoryResponse,
	PerformanceThroughput,
	PerformanceUtilization,
} from '$lib/types';

/** Get complete performance summary (throughput + latency + utilization). */
export async function getPerformanceSummary(): Promise<PerformanceSummary> {
	return apiGet('/api/performance/summary');
}

/** Get latency stats (p50/p95/p99) for a specific model or all models. */
export async function getLatencyStats(
	model?: string,
	window: number = 300
): Promise<PerformanceLatencyStats> {
	const params: Record<string, string> = { window: String(window) };
	if (model) params.model = model;
	return apiGet('/api/performance/latency', params);
}

/** Get drift detection results for all active models. */
export async function getDriftResults(): Promise<PerformanceDriftResponse> {
	return apiGet('/api/performance/drift');
}

/** Get optimization recommendations based on current metrics. */
export async function getRecommendations(): Promise<PerformanceRecommendationsResponse> {
	return apiGet('/api/performance/recommendations');
}

/** Get raw metric history records. */
export async function getPerformanceHistory(
	model?: string,
	hours: number = 24,
	limit: number = 500
): Promise<PerformanceHistoryResponse> {
	const params: Record<string, string> = {
		hours: String(hours),
		limit: String(limit),
	};
	if (model) params.model = model;
	return apiGet('/api/performance/history', params);
}

/** Get token throughput over a rolling window. */
export async function getThroughput(
	window: number = 300
): Promise<PerformanceThroughput> {
	return apiGet('/api/performance/throughput', { window: String(window) });
}

/** Get model utilization distribution. */
export async function getUtilization(
	window: number = 3600
): Promise<PerformanceUtilization> {
	return apiGet('/api/performance/utilization', { window: String(window) });
}

/** Delete metrics older than retention period. */
export async function cleanupMetrics(): Promise<{ available: boolean; deleted: number }> {
	return apiPost('/api/performance/cleanup');
}
