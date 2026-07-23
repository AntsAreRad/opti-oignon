/**
 * API client for web search endpoints.
 *
 * Functions for web search, proxy management, and PII sanitization preview.
 */

import { apiGet, apiPost } from './client';
import type {
	SearchResponse,
	SearchEngine,
	SearchHistoryEntry,
	ProxyStatusResponse,
	ProxyConfigRequest,
	ProxyConfigResponse,
	PIISanitizePreviewResponse,
	SearchConfigResponse,
} from '$lib/types';

/**
 * Launch a web search.
 * POST /api/search
 */
export async function searchWeb(params: {
	query: string;
	max_results?: number;
	engine?: string;
}): Promise<SearchResponse> {
	return apiPost<SearchResponse>('/api/search', params);
}

/**
 * List available search engines.
 * GET /api/search/engines
 */
export async function listEngines(): Promise<SearchEngine[]> {
	return apiGet<SearchEngine[]>('/api/search/engines');
}

/**
 * Get search history.
 * GET /api/search/history
 */
export async function searchHistory(): Promise<SearchHistoryEntry[]> {
	return apiGet<SearchHistoryEntry[]>('/api/search/history');
}

// -- Proxy & PII --

/**
 * Check proxy health and connectivity.
 * GET /api/search/proxy-status
 */
export async function getProxyStatus(): Promise<ProxyStatusResponse> {
	return apiGet<ProxyStatusResponse>('/api/search/proxy-status');
}

/**
 * Get current proxy configuration.
 * GET /api/search/proxy-config
 */
export async function getProxyConfig(): Promise<ProxyConfigResponse> {
	return apiGet<ProxyConfigResponse>('/api/search/proxy-config');
}

/**
 * Update proxy configuration at runtime.
 * POST /api/search/proxy-config
 */
export async function updateProxyConfig(
	config: ProxyConfigRequest,
): Promise<ProxyConfigResponse> {
	return apiPost<ProxyConfigResponse>('/api/search/proxy-config', config);
}

/**
 * Preview PII sanitization for a query.
 * POST /api/search/pii-preview
 */
export async function previewPII(query: string): Promise<PIISanitizePreviewResponse> {
	return apiPost<PIISanitizePreviewResponse>('/api/search/pii-preview', { query });
}

/**
 * Get search configuration and stats overview.
 * GET /api/search/config
 */
export async function getSearchConfig(): Promise<SearchConfigResponse> {
	return apiGet<SearchConfigResponse>('/api/search/config');
}
