/**
 * Typed API functions for memory endpoints.
 *
 * Manage facts in memory: list, add, delete,
 * nettoyage complet et extraction automatique depuis une conversation.
 */

import { apiGet, apiPost, apiDelete } from './client';
import type { MemoryFact, MemoryAddRequest, MemoryExtractResponse } from '$lib/types';

/** List all facts in memory. */
export async function listFacts(params?: {
	active_only?: boolean;
	category?: string;
}): Promise<MemoryFact[]> {
	const queryParams: Record<string, string> = {};
	if (params?.active_only !== undefined) {
		queryParams.active_only = String(params.active_only);
	}
	if (params?.category) {
		queryParams.category = params.category;
	}
	return apiGet<MemoryFact[]>('/api/memory', queryParams);
}

/** Add a fact to memory. */
export async function createFact(request: MemoryAddRequest): Promise<MemoryFact> {
	return apiPost<MemoryFact>('/api/memory', {
		fact: request.fact,
		category: request.category ?? 'context',
		source_conversation_id: request.source_conversation_id ?? '',
		confidence: request.confidence ?? 1.0,
	});
}

/** Delete a specific fact. */
export async function deleteFact(factId: string): Promise<void> {
	return apiDelete<void>(`/api/memory/${factId}`);
}

/** Delete all facts in memory. */
export async function clearAllFacts(): Promise<{ cleared: boolean; count: number }> {
	return apiDelete<{ cleared: boolean; count: number }>('/api/memory');
}

/** Extrait les faits d'une conversation et les stocke. */
export async function extractFacts(conversationId: string): Promise<MemoryExtractResponse> {
	return apiPost<MemoryExtractResponse>(`/api/memory/extract/${conversationId}`);
}
