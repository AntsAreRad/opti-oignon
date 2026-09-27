/**
 * Typed API functions for conversation endpoints.
 */

import { apiGet, apiPost, apiDelete, apiPatch } from './client';
import type {
	ConversationSummary,
	ConversationDetail,
	ConversationCreate,
	MessageItem
} from '$lib/types';

/**
 * The conversations, newest first. With `q`, the server searches titles and
 * messages and returns at most `limit` matches (it takes no offset then);
 * without, it lists them a page at a time from `offset`. Aborting `signal`
 * stops the request.
 */
export async function listConversations(params?: {
	q?: string;
	limit?: number;
	offset?: number;
	signal?: AbortSignal;
}): Promise<ConversationSummary[]> {
	const queryParams: Record<string, string> = {};
	if (params?.q) queryParams.q = params.q;
	if (params?.limit) queryParams.limit = String(params.limit);
	if (params?.offset !== undefined && !params?.q) queryParams.offset = String(params.offset);
	return apiGet('/api/conversations', queryParams, params?.signal);
}

export async function getConversation(id: string): Promise<ConversationDetail> {
	return apiGet<ConversationDetail>(`/api/conversations/${id}`);
}

export async function createConversation(
	data?: ConversationCreate
): Promise<ConversationSummary> {
	return apiPost<ConversationSummary>('/api/conversations', data ?? {});
}

export async function renameConversation(id: string, title: string): Promise<void> {
	await apiPatch<void>(`/api/conversations/${id}`, { title });
}

export async function deleteConversation(id: string): Promise<void> {
	await apiDelete<void>(`/api/conversations/${id}`);
}

export async function getMessages(id: string): Promise<MessageItem[]> {
	return apiGet<MessageItem[]>(`/api/conversations/${id}/messages`);
}
