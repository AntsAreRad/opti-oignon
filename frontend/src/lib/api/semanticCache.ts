/**
 * S68: Typed API client for the semantic cache endpoints.
 *
 * Provides status, toggle, config update, clear, and expire operations.
 */

import { apiGet, apiPost, apiPut } from './client';
import type { S68CacheStatus, S68CacheConfigUpdate, CacheClearResponse } from '$lib/types';

/** Get S68 semantic cache status, stats, and config. */
export async function getS68CacheStatus(): Promise<S68CacheStatus> {
	return apiGet<S68CacheStatus>('/api/cache/s68/status');
}

/** Toggle S68 semantic cache on/off. */
export async function toggleS68Cache(): Promise<S68CacheStatus> {
	return apiPost<S68CacheStatus>('/api/cache/s68/toggle');
}

/** Update S68 semantic cache configuration (partial). */
export async function updateS68CacheConfig(
	updates: S68CacheConfigUpdate
): Promise<S68CacheStatus> {
	return apiPut<S68CacheStatus>('/api/cache/s68/config', updates);
}

/** Clear S68 cache entries (all or by conversation). */
export async function clearS68Cache(
	conversationId?: string
): Promise<CacheClearResponse> {
	return apiPost<CacheClearResponse>('/api/cache/s68/clear', {
		conversation_id: conversationId ?? null,
	});
}

/** Remove expired entries from S68 cache. */
export async function expireS68Cache(): Promise<CacheClearResponse> {
	return apiPost<CacheClearResponse>('/api/cache/s68/expire');
}
