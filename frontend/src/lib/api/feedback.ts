/**
 * Typed API functions for Feedback & Analytics endpoints.
 *
 * Provides access to feedback submission, stats, and analytics
 * overview / trends data for the dashboard.
 */

import { apiGet, apiPost, apiDelete } from './client';
import type {
	FeedbackSubmitRequest,
	FeedbackEntryInfo,
	FeedbackStatsInfo,
	AnalyticsOverviewInfo,
	TrendsInfo,
	RoutingAccuracyInfo,
} from '$lib/types';

// -------------------------------------------------------------------------
// Feedback
// -------------------------------------------------------------------------

/** Submit feedback for a message (thumbs or stars). */
export async function submitFeedback(
	request: FeedbackSubmitRequest
): Promise<{ feedback_id: string; status: string }> {
	return apiPost<{ feedback_id: string; status: string }>('/api/feedback', request);
}

/** Get aggregated feedback statistics. */
export async function getFeedbackStats(
	since?: number
): Promise<FeedbackStatsInfo> {
	const params: Record<string, string> = {};
	if (since !== undefined) params.since = String(since);
	return apiGet<FeedbackStatsInfo>('/api/feedback/stats', params);
}

/** Get feedback entries for a specific model. */
export async function getFeedbackByModel(
	model: string,
	limit: number = 100
): Promise<FeedbackEntryInfo[]> {
	return apiGet<FeedbackEntryInfo[]>(
		`/api/feedback/by-model/${encodeURIComponent(model)}`,
		{ limit: String(limit) }
	);
}

/** Get feedback entries for a specific pipeline. */
export async function getFeedbackByPipeline(
	pipeline: string,
	limit: number = 100
): Promise<FeedbackEntryInfo[]> {
	return apiGet<FeedbackEntryInfo[]>(
		`/api/feedback/by-pipeline/${encodeURIComponent(pipeline)}`,
		{ limit: String(limit) }
	);
}

/** List feedback entries with pagination. */
export async function listFeedback(
	limit: number = 50,
	offset: number = 0
): Promise<FeedbackEntryInfo[]> {
	return apiGet<FeedbackEntryInfo[]>('/api/feedback/list', {
		limit: String(limit),
		offset: String(offset),
	});
}

/** Delete a feedback entry. */
export async function deleteFeedback(
	feedbackId: string
): Promise<{ status: string }> {
	return apiDelete<{ status: string }>(`/api/feedback/${feedbackId}`);
}

// -------------------------------------------------------------------------
// Analytics
// -------------------------------------------------------------------------

/** Get performance analytics overview. */
export async function getAnalyticsOverview(
	since?: number
): Promise<AnalyticsOverviewInfo> {
	const params: Record<string, string> = {};
	if (since !== undefined) params.since = String(since);
	return apiGet<AnalyticsOverviewInfo>('/api/analytics/overview', params);
}

/** Get time-series performance trends. */
export async function getAnalyticsTrends(
	window: string = '24h',
	buckets: number = 24,
	model?: string,
	pipeline?: string
): Promise<TrendsInfo> {
	const params: Record<string, string> = {
		window,
		buckets: String(buckets),
	};
	if (model) params.model = model;
	if (pipeline) params.pipeline = pipeline;
	return apiGet<TrendsInfo>('/api/analytics/trends', params);
}

/** Get routing accuracy comparison. */
export async function getRoutingAccuracy(
	since?: number
): Promise<RoutingAccuracyInfo> {
	const params: Record<string, string> = {};
	if (since !== undefined) params.since = String(since);
	return apiGet<RoutingAccuracyInfo>('/api/analytics/routing-accuracy', params);
}

/** Trigger analytics cleanup of old records. */
export async function cleanupAnalytics(): Promise<{ status: string; deleted: number }> {
	return apiPost<{ status: string; deleted: number }>('/api/analytics/cleanup');
}
