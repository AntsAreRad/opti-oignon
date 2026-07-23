/**
 * RAG Dashboard API client.
 *
 * Stats, usage over time, source reliability, collection health,
 * auto-refresh, external connector status.
 */

import { apiGet, apiPost } from './client';
import type {
	RAGDashboardStats,
	RAGUsageResponse,
	RAGSourcesResponse,
	RAGHealthResponse,
	RAGRefreshResult,
	RAGConnectorsResponse,
	RAGBackendsResponse,
} from '$lib/types';

const BASE = '/api/rag/dashboard';

/** Get overall dashboard statistics. */
export async function getDashboardStats(): Promise<RAGDashboardStats> {
	return (await apiGet(`${BASE}/stats`)) as RAGDashboardStats;
}

/** Get query usage over time (daily data points). */
export async function getUsageOverTime(
	days: number = 30
): Promise<RAGUsageResponse> {
	return (await apiGet(`${BASE}/usage`, { days: String(days) })) as RAGUsageResponse;
}

/** Get source reliability ranking. */
export async function getSourceReliability(
	limit: number = 50
): Promise<RAGSourcesResponse> {
	return (await apiGet(`${BASE}/sources`, { limit: String(limit) })) as RAGSourcesResponse;
}

/** Get collection health metrics. */
export async function getCollectionHealth(): Promise<RAGHealthResponse> {
	return (await apiGet(`${BASE}/health`)) as RAGHealthResponse;
}

/** Trigger auto-refresh check for stale sources. */
export async function triggerRefresh(): Promise<RAGRefreshResult> {
	return (await apiPost(`${BASE}/refresh`, {})) as RAGRefreshResult;
}

/** Get external connector statuses. */
export async function getConnectors(): Promise<RAGConnectorsResponse> {
	return (await apiGet(`${BASE}/connectors`)) as RAGConnectorsResponse;
}

/** Get available external backends. */
export async function getBackends(): Promise<RAGBackendsResponse> {
	return (await apiGet(`${BASE}/backends`)) as RAGBackendsResponse;
}
