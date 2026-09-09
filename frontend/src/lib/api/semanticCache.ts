/**
 * Typed API client for the semantic cache endpoints.
 *
 * Provides status, toggle, config update, clear, and expire operations.
 */

import { apiGet, apiPost, apiPut } from './client';
import type { SemCacheStatus, SemCacheConfigUpdate, CacheClearResponse } from '$lib/types';

/** Get semantic cache status, stats, and config. */
export async function getSemCacheStatus(): Promise<SemCacheStatus> {
	return apiGet<SemCacheStatus>('/api/cache/semcache/status');
}

/** Toggle semantic cache on/off. */
export async function toggleSemCache(): Promise<SemCacheStatus> {
	return apiPost<SemCacheStatus>('/api/cache/semcache/toggle');
}

/** Update semantic cache configuration (partial). */
export async function updateSemCacheConfig(
	updates: SemCacheConfigUpdate
): Promise<SemCacheStatus> {
	return apiPut<SemCacheStatus>('/api/cache/semcache/config', updates);
}

/** Clear cache entries (all or by conversation). */
export async function clearSemCache(
	conversationId?: string
): Promise<CacheClearResponse> {
	return apiPost<CacheClearResponse>('/api/cache/semcache/clear', {
		conversation_id: conversationId ?? null,
	});
}

/** Remove expired entries from cache. */
export async function expireSemCache(): Promise<CacheClearResponse> {
	return apiPost<CacheClearResponse>('/api/cache/semcache/expire');
}
