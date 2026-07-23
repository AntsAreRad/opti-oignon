/**
 * API client for Humanizer endpoints --.
 *
 * Provides typed access to humanizer rewrite, config, feedback, and stats.
 */

import { apiGet, apiPost } from './client';
import type {
	HumanizerConfigResponse,
	HumanizerConfigUpdate,
	HumanizerFeedbackRequest,
	HumanizerFeedbackResponse,
	HumanizerRewriteRequest,
	HumanizerRewriteResponse,
	HumanizerStatsResponse,
} from '$lib/types';

/** Humanize a text passage. */
export async function rewriteText(
	request: HumanizerRewriteRequest
): Promise<HumanizerRewriteResponse> {
	return apiPost<HumanizerRewriteResponse>('/api/humanizer/rewrite', request);
}

/** Get current humanizer configuration. */
export async function getHumanizerConfig(): Promise<HumanizerConfigResponse> {
	return apiGet<HumanizerConfigResponse>('/api/humanizer/config');
}

/** Update humanizer configuration. */
export async function updateHumanizerConfig(
	update: HumanizerConfigUpdate
): Promise<HumanizerConfigResponse> {
	return apiPost<HumanizerConfigResponse>('/api/humanizer/config', update);
}

/** Submit A/B comparison feedback. */
export async function submitHumanizerFeedback(
	request: HumanizerFeedbackRequest
): Promise<HumanizerFeedbackResponse> {
	return apiPost<HumanizerFeedbackResponse>('/api/humanizer/feedback', request);
}

/** Get aggregated feedback statistics. */
export async function getHumanizerStats(): Promise<HumanizerStatsResponse> {
	return apiGet<HumanizerStatsResponse>('/api/humanizer/stats');
}
