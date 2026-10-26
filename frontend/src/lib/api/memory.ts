/**
 * Typed API functions for memory endpoints.
 *
 * Manage facts in memory: list, add, delete,
 * nettoyage complet et extraction automatique depuis une conversation.
 */

import { apiGet, apiPost, apiDelete } from './client';
import type { MemoryFact, MemoryAddRequest, MemoryExtractResponse, MemoryTrash } from '$lib/types';

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

/** Move a fact to the trash: out of the window, restorable until the trash erases it. */
export async function deleteFact(factId: string): Promise<void> {
	return apiDelete<void>(`/api/memory/${factId}`);
}

/** Move every fact in memory to the trash. */
export async function clearAllFacts(): Promise<{ cleared: boolean; count: number }> {
	return apiDelete<{ cleared: boolean; count: number }>('/api/memory');
}

/** The facts in the trash, each with when it is erased for good, and the trash's days. */
export async function listTrash(): Promise<MemoryTrash> {
	return apiGet<MemoryTrash>('/api/memory/trash');
}

/** Take a fact out of the trash, back into service. */
export async function restoreFact(factId: string): Promise<MemoryFact> {
	return apiPost<MemoryFact>(`/api/memory/${factId}/restore`);
}

/** Erase one fact in the trash for good. */
export async function eraseFact(factId: string): Promise<{ erased: boolean; id: string }> {
	return apiDelete<{ erased: boolean; id: string }>(`/api/memory/trash/${factId}`);
}

/** Erase every fact in the trash for good. */
export async function emptyTrash(): Promise<{ erased: number }> {
	return apiDelete<{ erased: number }>('/api/memory/trash');
}

/** Extrait les faits d'une conversation et les stocke. */
export async function extractFacts(conversationId: string): Promise<MemoryExtractResponse> {
	return apiPost<MemoryExtractResponse>(`/api/memory/extract/${conversationId}`);
}
