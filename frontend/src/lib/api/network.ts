/**
 * Network Manager API client -- Offline-First Intelligence.
 *
 * Typed functions for connectivity status, queue management,
 * and pre-cache warming.
 */

import { apiGet, apiPost, apiDelete } from './client';

export interface NetworkStatusInfo {
	available: boolean;
	online: boolean;
	ollama_reachable: boolean;
	embedding_reachable: boolean;
	last_check: number;
	last_error: string;
	latency_ms: number;
	consecutive_failures: number;
	polling_active: boolean;
	queue_size: number;
	config: Record<string, unknown>;
}

export interface QueueEntryInfo {
	id: string;
	query: string;
	task_type: string;
	priority: number;
	created_at: number;
	status: string;
	error: string;
	model: string;
}

export interface QueueListInfo {
	available: boolean;
	entries: QueueEntryInfo[];
	total: number;
	pending: number;
}

export interface QueueProcessInfo {
	processed: number;
	results: Array<Record<string, unknown>>;
}

export interface PreCacheInfo {
	total: number;
	cached: number;
	skipped: number;
	failed: number;
	duration_ms: number;
	errors: string[];
}

/** Get current network connectivity status. */
export async function getNetworkStatus(): Promise<NetworkStatusInfo> {
	return apiGet('/api/network/status');
}

/** Trigger an immediate connectivity check. */
export async function pollNow(): Promise<NetworkStatusInfo> {
	return apiPost('/api/network/poll');
}

/** List queue entries, optionally filtered by status. */
export async function getQueueEntries(
	statusFilter?: string,
	limit: number = 50
): Promise<QueueListInfo> {
	const params = new URLSearchParams();
	if (statusFilter) params.set('status_filter', statusFilter);
	params.set('limit', String(limit));
	return apiGet(`/api/network/queue?${params}`);
}

/** Manually trigger queue processing. */
export async function processQueue(): Promise<QueueProcessInfo> {
	return apiPost('/api/network/queue/process');
}

/** Clear queue entries, optionally filtered by status. */
export async function clearQueue(statusFilter?: string): Promise<{ removed: number }> {
	const params = statusFilter ? `?status_filter=${statusFilter}` : '';
	return apiDelete(`/api/network/queue${params}`);
}

/** Trigger a pre-cache warming run. */
export async function runPreCache(): Promise<PreCacheInfo> {
	return apiPost('/api/network/pre-cache');
}
