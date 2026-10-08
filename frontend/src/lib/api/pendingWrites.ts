/**
 * Typed API functions for the review of pending writes.
 *
 * The agent's memory and notes writes that the user's typed words did not
 * endorse, and the facts the manual extraction drew from anything else,
 * wait for the user: list them, then accept or decline a batch by id.
 */

import { apiGet, apiPost } from './client';
import type { DecisionResult, PendingStore, PendingWrite } from '$lib/pendingWrites';

/** The proposals waiting for review, oldest first; one store's when ``store`` is given. */
export async function listPendingWrites(store?: PendingStore): Promise<PendingWrite[]> {
	return apiGet<PendingWrite[]>('/api/pending-writes', store ? { store } : undefined);
}

/** Apply the given proposals in order, each once and exactly as proposed. */
export async function acceptPendingWrites(ids: string[]): Promise<{ results: DecisionResult[] }> {
	return apiPost<{ results: DecisionResult[] }>('/api/pending-writes/accept', { ids });
}

/** Decline the given proposals; nothing is written. */
export async function declinePendingWrites(ids: string[]): Promise<{ results: DecisionResult[] }> {
	return apiPost<{ results: DecisionResult[] }>('/api/pending-writes/decline', { ids });
}
