/**
 * Typed API functions for context pipeline health endpoints.
 *
 * Provides context health status, budget allocation, and window stats.
 */

import { apiGet } from './client';
import type { ContextHealthResponse, ContextBudgetResponse, ContextStatsResponse } from '$lib/types';

/** Context pipeline health with conversation data. */
export async function getContextHealth(conversationId?: string, model?: string): Promise<ContextHealthResponse> {
	const params: Record<string, string> = {};
	if (conversationId) params.conversation_id = conversationId;
	if (model) params.model = model;
	return apiGet<ContextHealthResponse>('/api/context/health', params);
}

/** Allocation de budget pour un modele specifique. */
export async function getModelBudget(modelName: string): Promise<ContextBudgetResponse> {
	return apiGet<ContextBudgetResponse>(`/api/context/budget/${encodeURIComponent(modelName)}`);
}

/** Statistiques detaillees de la fenetre glissante. */
export async function getContextStats(conversationId?: string): Promise<ContextStatsResponse> {
	const params: Record<string, string> = {};
	if (conversationId) params.conversation_id = conversationId;
	return apiGet<ContextStatsResponse>('/api/context/stats', params);
}
