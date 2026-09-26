/**
 * API client for the live inference metrics (the server's /api/metrics
 * routes): the snapshot the chat's metrics overlay polls while a reply is
 * generated.
 */

import { apiGet } from './client';

/**
 * One point-in-time snapshot. A GPU reading of -1 means the server has no
 * GPU data; the memory readings are in megabytes.
 */
export type LiveMetricsSample = {
	timestamp: number;
	tokens_per_second: number;
	prompt_eval_time_ms: number;
	eval_time_ms: number;
	total_tokens: number;
	pending_tokens: number;
	gpu_utilization_pct: number;
	gpu_memory_used_mb: number;
	gpu_memory_total_mb: number;
	gpu_temperature_c: number;
	system_memory_used_mb: number;
	system_memory_total_mb: number;
	is_generating: boolean;
	active_model: string;
};

/** The current snapshot; 503 when the server has no metrics collector. */
export async function getLiveMetrics(): Promise<LiveMetricsSample> {
	return apiGet<LiveMetricsSample>('/api/metrics/live');
}
