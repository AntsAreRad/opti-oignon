/**
 * Typed API functions for cache management endpoints.
 *
 * Provides stats retrieval and cache clearing operations.
 */

import { apiGet, apiDelete } from './client';
import type { CacheCombinedStats, CacheClearResponse } from '$lib/types';

/** Retrieve combined cache statistics (response + semantic). */
export async function getCacheStats(): Promise<CacheCombinedStats> {
	return apiGet<CacheCombinedStats>('/api/cache/stats');
}

/** Clear the entire response cache. */
export async function clearAllCache(): Promise<CacheClearResponse> {
	return apiDelete<CacheClearResponse>('/api/cache');
}

/** Vide le cache pour un modele specifique. */
export async function clearModelCache(model: string): Promise<CacheClearResponse> {
	return apiDelete<CacheClearResponse>(`/api/cache/${encodeURIComponent(model)}`);
}
