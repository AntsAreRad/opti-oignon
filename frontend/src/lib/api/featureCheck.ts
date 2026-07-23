/**
 * Feature availability checker.
 *
 * Queries /api/health and caches the modules map for 30 seconds.
 * Used by settings sub-panels to show a graceful "feature unavailable"
 * state instead of a broken or blank panel when optional backend
 * modules are not loaded.
 */

import { apiGet } from './client';
import type { HealthResponse } from '$lib/types';

/** Cached health response with timestamp. */
let _cachedModules: Record<string, boolean> | null = null;
let _cacheTimestamp = 0;

/** Cache TTL in milliseconds (30 seconds). */
const CACHE_TTL_MS = 30_000;

/**
 * Get the full modules availability map from /api/health.
 * Results are cached for 30 seconds to avoid hammering the endpoint
 * when multiple panels mount simultaneously.
 */
export async function getFeatureMap(): Promise<Record<string, boolean>> {
	const now = Date.now();
	if (_cachedModules && now - _cacheTimestamp < CACHE_TTL_MS) {
		return _cachedModules;
	}

	try {
		const resp = await apiGet<HealthResponse>('/api/health');
		_cachedModules = resp.modules ?? {};
		_cacheTimestamp = Date.now();
		return _cachedModules;
	} catch {
		// If health check fails entirely, return empty map
		// (all features will appear unavailable).
		return _cachedModules ?? {};
	}
}

/**
 * Check whether a specific backend feature is available.
 *
 * Feature keys match the keys in the /api/health modules map,
 * for example: 'rag_store', 'plugin_registry', 'fine_tune_export',
 * 'analytics', 'telemetry', 'context_optimizer', etc.
 *
 * @param featureKey - The module key from the health endpoint.
 * @returns true if the feature is available, false otherwise.
 */
export async function checkFeatureAvailable(featureKey: string): Promise<boolean> {
	const modules = await getFeatureMap();
	return modules[featureKey] === true;
}

/**
 * Check multiple features at once. Returns a map of key -> available.
 */
export async function checkFeaturesAvailable(
	featureKeys: string[]
): Promise<Record<string, boolean>> {
	const modules = await getFeatureMap();
	const result: Record<string, boolean> = {};
	for (const key of featureKeys) {
		result[key] = modules[key] === true;
	}
	return result;
}

/**
 * Invalidate the cached health response, forcing the next
 * checkFeatureAvailable call to re-fetch from the backend.
 */
export function invalidateFeatureCache(): void {
	_cachedModules = null;
	_cacheTimestamp = 0;
}
