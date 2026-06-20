/**
 * Telemetry Dashboard API -- S113.
 *
 * Client functions for the telemetry dashboard endpoints.
 */

import { apiGet, apiPost, apiDelete } from './client';

export interface TelemetryConsumerInfo {
	name: string;
	healthy: boolean;
}

export interface TelemetryStats {
	enabled: boolean;
	total_events: number;
	total_requests: number;
	total_tokens: number;
	active_requests: number;
	buffer_size: number;
	buffer_max_size: number;
	consumer_count: number;
}

export interface TelemetryConsumersResponse {
	consumers: TelemetryConsumerInfo[];
	count: number;
}

export interface TelemetryFlushResponse {
	flushed_events: number;
}

// S114: History types
export interface HistoryEvent {
	id: number;
	request_id: string;
	model: string;
	timestamp: number;
	latency_ms: number;
	tokens_in: number;
	tokens_out: number;
	tok_per_sec: number;
	prompt_eval_ms: number;
	token_gen_ms: number;
}

export interface TelemetryHistoryResponse {
	events: HistoryEvent[];
	total: number;
	limit: number;
	offset: number;
}

export interface TrendBucket {
	bucket_start: number;
	bucket_label: string;
	event_count: number;
	avg_latency_ms: number;
	avg_tok_per_sec: number;
	total_tokens_in: number;
	total_tokens_out: number;
}

export interface TelemetryTrendsResponse {
	buckets: TrendBucket[];
	hours: number;
	model: string;
}

export interface ModelBreakdown {
	model: string;
	event_count: number;
	avg_latency_ms: number;
	avg_tok_per_sec: number;
	total_tokens_in: number;
	total_tokens_out: number;
}

export interface TelemetryHistoryStats {
	available: boolean;
	total_stored: number;
	retention_days: number;
	oldest_event_ts: number;
	max_events: number;
	auto_purge_enabled: boolean;
}

export interface TelemetryPurgeResponse {
	purged_count: number;
}

/** Get telemetry collector statistics. */
export async function getTelemetryStats(): Promise<TelemetryStats> {
	return apiGet<TelemetryStats>('/api/telemetry/stats');
}

/** Get registered consumers with health status. */
export async function getTelemetryConsumers(): Promise<TelemetryConsumersResponse> {
	return apiGet<TelemetryConsumersResponse>('/api/telemetry/consumers');
}

/** Manually flush the telemetry event buffer. */
export async function flushTelemetry(): Promise<TelemetryFlushResponse> {
	return apiPost<TelemetryFlushResponse>('/api/telemetry/flush');
}

// S114: History endpoints

/** Get paginated telemetry event history. */
export async function getTelemetryHistory(
	limit: number = 50,
	offset: number = 0,
	model: string = '',
): Promise<TelemetryHistoryResponse> {
	const params: Record<string, string> = {
		limit: String(limit),
		offset: String(offset),
	};
	if (model) params.model = model;
	return apiGet<TelemetryHistoryResponse>('/api/telemetry/history', params);
}

/** Get aggregated latency/throughput trends. */
export async function getTelemetryTrends(
	hours: number = 24,
	model: string = '',
): Promise<TelemetryTrendsResponse> {
	const params: Record<string, string> = { hours: String(hours) };
	if (model) params.model = model;
	return apiGet<TelemetryTrendsResponse>('/api/telemetry/trends', params);
}

/** Get per-model breakdown from history. */
export async function getHistoryModelBreakdown(): Promise<{ models: ModelBreakdown[] }> {
	return apiGet<{ models: ModelBreakdown[] }>('/api/telemetry/history/models');
}

/** Get history store stats. */
export async function getHistoryStats(): Promise<TelemetryHistoryStats> {
	return apiGet<TelemetryHistoryStats>('/api/telemetry/history/stats');
}

/** Purge old events (0 = purge all). */
export async function purgeHistory(olderThanDays: number = 0): Promise<TelemetryPurgeResponse> {
	return apiDelete<TelemetryPurgeResponse>(
		`/api/telemetry/history?older_than_days=${olderThanDays}`,
	);
}

// S115: Settings & Export

export interface HistorySettingsRequest {
	retention_days?: number;
	auto_purge_enabled?: boolean;
}

export interface HistorySettingsResponse {
	retention_days: number;
	auto_purge_enabled: boolean;
}

/** Update history retention and auto-purge settings. */
export async function updateHistorySettings(
	settings: HistorySettingsRequest,
): Promise<HistorySettingsResponse> {
	return apiPut<HistorySettingsResponse>('/api/telemetry/history/settings', settings);
}

/** Get CSV export URL for history. */
export function getHistoryExportUrl(model: string = ''): string {
	const base = '/api/telemetry/history/export';
	return model ? `${base}?model=${encodeURIComponent(model)}` : base;
}
