/**
 * Typed API functions for preset endpoints.
 *
 * Provides full CRUD, search, match, and duplicate operations for presets.
 */

import { apiGet, apiPost, apiPut, apiDelete } from './client';
import type { PresetInfo } from '$lib/types';

/** Retrieve the list of all presets. */
export async function listPresets(): Promise<PresetInfo[]> {
	return apiGet<PresetInfo[]>('/api/presets');
}

/** Retrieve a preset by its ID. */
export async function getPreset(id: string): Promise<PresetInfo> {
	return apiGet<PresetInfo>(`/api/presets/${encodeURIComponent(id)}`);
}

/** Create a new preset. */
export async function createPreset(data: Partial<PresetInfo>): Promise<PresetInfo> {
	return apiPost<PresetInfo>('/api/presets', data);
}

/** Met a jour un preset existant. */
export async function updatePreset(id: string, data: Partial<PresetInfo>): Promise<PresetInfo> {
	return apiPut<PresetInfo>(`/api/presets/${encodeURIComponent(id)}`, data);
}

/** Delete a preset. */
export async function deletePreset(id: string): Promise<void> {
	return apiDelete<void>(`/api/presets/${encodeURIComponent(id)}`);
}

/** Duplicate an existing preset. */
export async function duplicatePreset(id: string): Promise<PresetInfo> {
	return apiPost<PresetInfo>(`/api/presets/${encodeURIComponent(id)}/duplicate`);
}

/** Recherche de presets par texte libre. */
export async function searchPresets(query: string): Promise<PresetInfo[]> {
	return apiGet<PresetInfo[]>('/api/presets/search', { q: query });
}

/** Detection automatique du preset le plus adapte a un texte. */
export async function matchPreset(text: string): Promise<PresetInfo | null> {
	try {
		return await apiGet<PresetInfo>('/api/presets/match', { text });
	} catch {
		// 404 si aucun match
		return null;
	}
}
