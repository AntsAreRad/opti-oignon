/**
 * Typed API functions for smart routing endpoints.
 *
 * Provides access to model profile CRUD, smart model selection,
 * and router configuration.
 */

import { apiGet, apiPut, apiPost, apiDelete } from './client';
import type {
	ModelProfileInfo,
	ModelProfilesResponse,
	SmartRoutingResult,
	SmartRouterConfig,
} from '$lib/types';

// -------------------------------------------------------------------------
// Profiles
// -------------------------------------------------------------------------

/** Fetch all model profiles. */
export async function getProfiles(): Promise<ModelProfilesResponse> {
	return apiGet<ModelProfilesResponse>('/api/models/profiles');
}

/** Create or update a model profile. */
export async function saveProfile(
	modelName: string,
	profile: Partial<ModelProfileInfo>
): Promise<{ status: string; profile: ModelProfileInfo }> {
	return apiPut<{ status: string; profile: ModelProfileInfo }>(
		`/api/smart-routing/profiles/${encodeURIComponent(modelName)}`,
		profile
	);
}

/** Delete a model profile. */
export async function deleteProfile(
	modelName: string
): Promise<{ status: string; removed: string }> {
	return apiDelete<{ status: string; removed: string }>(
		`/api/smart-routing/profiles/${encodeURIComponent(modelName)}`
	);
}

/** Update task scores for a model. */
export async function updateTaskScores(
	modelName: string,
	taskScores: Record<string, number>
): Promise<{ status: string; task_scores: Record<string, number> }> {
	return apiPut<{ status: string; task_scores: Record<string, number> }>(
		`/api/smart-routing/profiles/${encodeURIComponent(modelName)}/task-scores`,
		{ task_scores: taskScores }
	);
}

/** Auto-detect model capabilities via ollama.show(). */
export async function autoDetectModel(
	modelName: string
): Promise<{ status: string; profile: ModelProfileInfo }> {
	return apiPost<{ status: string; profile: ModelProfileInfo }>(
		`/api/smart-routing/profiles/${encodeURIComponent(modelName)}/auto-detect`
	);
}

/** Save all profiles to YAML file on disk. */
export async function saveAllProfiles(): Promise<{ status: string; count: number }> {
	return apiPost<{ status: string; count: number }>('/api/smart-routing/profiles/save');
}

// -------------------------------------------------------------------------
// Smart routing
// -------------------------------------------------------------------------

/** Select the optimal model for a pipeline step type. */
export async function selectModel(
	stepType: string,
	options?: { required_context?: number; prefer_speed?: boolean }
): Promise<SmartRoutingResult> {
	const params: Record<string, string> = { step_type: stepType };
	if (options?.required_context !== undefined) {
		params.required_context = String(options.required_context);
	}
	if (options?.prefer_speed !== undefined) {
		params.prefer_speed = String(options.prefer_speed);
	}
	return apiGet<SmartRoutingResult>('/api/smart-routing/select', params);
}

/** Select models for each step in a pipeline. */
export async function selectForPipeline(
	stepTypes: string[]
): Promise<{ selections: Record<string, SmartRoutingResult>; count: number }> {
	return apiPost<{ selections: Record<string, SmartRoutingResult>; count: number }>(
		'/api/smart-routing/select-pipeline',
		stepTypes
	);
}

// -------------------------------------------------------------------------
// Configuration
// -------------------------------------------------------------------------

/** Get smart router configuration. */
export async function getRouterConfig(): Promise<SmartRouterConfig> {
	return apiGet<SmartRouterConfig>('/api/smart-routing/config');
}

/** Update smart router configuration. */
export async function updateRouterConfig(
	config: Partial<{ enabled: boolean; default_model: string; speed_preference: string }>
): Promise<{ status: string; config: SmartRouterConfig }> {
	return apiPut<{ status: string; config: SmartRouterConfig }>(
		'/api/smart-routing/config',
		config
	);
}

/** Save router configuration to YAML file. */
export async function saveRouterConfig(): Promise<{ status: string; config: SmartRouterConfig }> {
	return apiPost<{ status: string; config: SmartRouterConfig }>(
		'/api/smart-routing/config/save'
	);
}

// -------------------------------------------------------------------------
// Model Health
// -------------------------------------------------------------------------

/** Health record for a single model. */
export interface ModelHealthRecord {
	model: string;
	status: 'healthy' | 'degraded' | 'unavailable' | 'unknown';
	latency_ms: number;
	last_check: number;
	last_success: number;
	error_count: number;
	consecutive_failures: number;
	last_error: string;
	check_count: number;
}

/** Response for all model health records. */
export interface AllModelHealthResponse {
	records: Record<string, ModelHealthRecord>;
	summary: { healthy: number; degraded: number; unavailable: number };
	config: Record<string, unknown>;
}

/** Fetch all model health records. */
export async function getAllModelHealth(): Promise<AllModelHealthResponse> {
	return apiGet<AllModelHealthResponse>('/api/smart-routing/model-health');
}

/** Fetch health record for a single model. */
export async function getModelHealth(modelName: string): Promise<ModelHealthRecord> {
	return apiGet<ModelHealthRecord>(
		`/api/smart-routing/model-health/${encodeURIComponent(modelName)}`
	);
}

/** Force an immediate health check on all models. */
export async function forceHealthCheck(): Promise<{
	status: string;
	checked: number;
	records: Record<string, ModelHealthRecord>;
}> {
	return apiPost<{
		status: string;
		checked: number;
		records: Record<string, ModelHealthRecord>;
	}>('/api/smart-routing/model-health/check');
}
