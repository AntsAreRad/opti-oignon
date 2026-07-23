/**
 * API client for Conversation Compressor.
 *
 * Provides typed access to compression config, stats,
 * and full-archive search endpoints.
 */

import { apiGet, apiPut, apiPost } from './client';

// ============================================================================
// Types
// ============================================================================

export interface CompressionConfig {
	enabled: boolean;
	strategy: 'rule' | 'llm' | 'hybrid';
	recent_messages_keep: number;
	compression_threshold_ratio: number;
	llm_summary_model: string | null;
	llm_summary_max_tokens: number;
	llm_summary_temperature: number;
	llm_summary_timeout: number;
	rule_max_facts_per_message: number;
	rule_min_message_length: number;
	archive_retrieval_top_k: number;
	archive_retrieval_min_score: number;
	archive_retrieval_snippet_length: number;
	retrieval_trigger_enabled: boolean;
	retrieval_trigger_min_confidence: number;
}

export interface CompressionConfigUpdate {
	enabled?: boolean;
	strategy?: 'rule' | 'llm' | 'hybrid';
	recent_messages_keep?: number;
	compression_threshold_ratio?: number;
	llm_summary_model?: string | null;
	llm_summary_max_tokens?: number;
	llm_summary_temperature?: number;
	archive_retrieval_top_k?: number;
	archive_retrieval_min_score?: number;
	retrieval_trigger_enabled?: boolean;
	retrieval_trigger_min_confidence?: number;
}

export interface CompressionStats {
	conversation_id: string;
	last_compression_available: boolean;
	summary: string | null;
	original_count: number | null;
	compressed_count: number | null;
	strategy_used: string | null;
	tokens_saved: number | null;
	compression_ratio: number | null;
}

export interface ArchiveSearchResultItem {
	message_id: number;
	role: string;
	snippet: string;
	score: number;
	timestamp: string;
}

export interface ArchiveSearchResponse {
	conversation_id: string;
	query: string;
	results: ArchiveSearchResultItem[];
	total_found: number;
}

// ============================================================================
// Config endpoints
// ============================================================================

export function getCompressionConfig(): Promise<CompressionConfig> {
	return apiGet<CompressionConfig>('/api/compression/config');
}

export function updateCompressionConfig(updates: CompressionConfigUpdate): Promise<CompressionConfig> {
	return apiPut<CompressionConfig>('/api/compression/config', updates);
}

export function reloadCompressionConfig(): Promise<{ status: string }> {
	return apiPost('/api/compression/config/reload');
}

// ============================================================================
// Stats endpoint
// ============================================================================

export function getCompressionStats(conversationId: string): Promise<CompressionStats> {
	return apiGet<CompressionStats>(`/api/compression/stats/${conversationId}`);
}

// ============================================================================
// Archive search endpoint
// ============================================================================

export function searchArchive(
	conversationId: string,
	query: string,
	topK = 3,
	minScore = 0.05
): Promise<ArchiveSearchResponse> {
	return apiPost<ArchiveSearchResponse>(
		`/api/compression/archive/search/${conversationId}`,
		{ query, top_k: topK, min_score: minScore }
	);
}
