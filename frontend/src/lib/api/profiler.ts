/**
 * Inference Profiler API -- S113.
 *
 * Client functions for the inference profiler endpoints.
 */

import { apiGet } from './client';

export interface InferenceProfile {
	request_id: string;
	model: string;
	timestamp: number;
	total_ms: number;
	prompt_eval_ms: number;
	token_gen_ms: number;
	overhead_ms: number;
	tokens_in: number;
	tokens_out: number;
	tok_per_sec: number;
}

export interface ProfilerSummary {
	model: string;
	request_count: number;
	avg_total_ms: number;
	p50_total_ms: number;
	p95_total_ms: number;
	p99_total_ms: number;
	avg_prompt_eval_ms: number;
	avg_token_gen_ms: number;
	avg_overhead_ms: number;
	avg_tok_per_sec: number;
}

export interface ProfilerSummaryResponse {
	models: ProfilerSummary[];
	total_profiled_requests: number;
}

export interface ProfilerRecentResponse {
	profiles: InferenceProfile[];
	count: number;
}

/** Get aggregated profiling stats per model. */
export async function getProfilerSummary(): Promise<ProfilerSummaryResponse> {
	return apiGet<ProfilerSummaryResponse>('/api/profiler/summary');
}

/** Get the most recent N inference profiles. */
export async function getRecentProfiles(n: number = 20): Promise<ProfilerRecentResponse> {
	return apiGet<ProfilerRecentResponse>('/api/profiler/recent', { n: String(n) });
}
