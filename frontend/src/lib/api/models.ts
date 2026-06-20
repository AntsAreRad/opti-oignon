/**
 * Typed API functions for model endpoints.
 */

import { apiGet } from './client';
import type { ModelListResponse, EffectiveModelResponse, ModelProfilesResponse, ModelProfileInfo } from '$lib/types';

export async function listModels(): Promise<ModelListResponse> {
	return apiGet<ModelListResponse>('/api/models');
}

export async function getEffectiveModel(params?: {
	conversation_id?: string;
	preset?: string;
}): Promise<EffectiveModelResponse> {
	const queryParams: Record<string, string> = {};
	if (params?.conversation_id) queryParams.conversation_id = params.conversation_id;
	if (params?.preset) queryParams.preset = params.preset;
	return apiGet<EffectiveModelResponse>('/api/models/effective', queryParams);
}

// S46: Model profiles
export async function listModelProfiles(): Promise<ModelProfilesResponse> {
	return apiGet<ModelProfilesResponse>('/api/models/profiles');
}

export async function getModelProfile(modelName: string): Promise<ModelProfileInfo> {
	return apiGet<ModelProfileInfo>(`/api/models/profiles/${encodeURIComponent(modelName)}`);
}
