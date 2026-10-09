/**
 * Typed API functions for the review of pending writes.
 *
 * The agent's memory and notes writes that the user's typed words did not
 * endorse, the facts the manual extraction drew from anything else, and the
 * skills the agent or its teacher writes wait for the user: list them, then
 * accept a batch by id with the digest of what was shown, or decline one.
 */

import { apiGet, apiPost } from './client';
import type { DecisionResult, PendingStore, PendingWrite } from '$lib/pendingWrites';

/** The proposals waiting for review, oldest first; one store's when ``store`` is given. */
export async function listPendingWrites(store?: PendingStore): Promise<PendingWrite[]> {
	return apiGet<PendingWrite[]>('/api/pending-writes', store ? { store } : undefined);
}

/**
 * Apply the given proposals in order, each once and exactly as proposed. Each
 * digest names what the user was shown: a skill is applied only by the digest
 * of its text, any other proposal only by its own when one is given.
 */
export async function acceptPendingWrites(
	ids: string[],
	digests: Record<string, string> = {}
): Promise<{ results: DecisionResult[] }> {
	return apiPost<{ results: DecisionResult[] }>('/api/pending-writes/accept', { ids, digests });
}

/** Decline the given proposals; nothing is written. */
export async function declinePendingWrites(ids: string[]): Promise<{ results: DecisionResult[] }> {
	return apiPost<{ results: DecisionResult[] }>('/api/pending-writes/decline', { ids });
}
