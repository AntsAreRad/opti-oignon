/**
 * Typed API client for the two-tier memory store (S174).
 *
 * Distinct from the legacy api/memory.ts: these endpoints operate over the
 * coordinated MemoryStore (/api/memories) and expose list, edit, soft delete,
 * and restore, per user and encrypted at rest. The legacy client stays intact.
 */

import { apiGet, apiPatch, apiPost, apiDelete } from './client';

export type MemoryCategory =
	| 'identity'
	| 'preference'
	| 'fact'
	| 'contact'
	| 'project'
	| 'goal';

/** A fact in the two-tier memory store. */
export interface MemoryRecord {
	id: string;
	text: string;
	category: string;
	source: string;
	created_at: string;
	updated_at: string;
	active: boolean;
	use_count: number;
}

/** Fields for editing a stored memory; omit a field to leave it unchanged. */
export interface MemoryEdit {
	text?: string;
	category?: string;
}

export const MEMORY_CATEGORIES: MemoryCategory[] = [
	'identity',
	'preference',
	'fact',
	'contact',
	'project',
	'goal'
];

/** List memories in the store, optionally filtered by category. */
export async function listMemories(params?: {
	active_only?: boolean;
	category?: string;
}): Promise<MemoryRecord[]> {
	const query: Record<string, string> = {};
	if (params?.active_only !== undefined) {
		query.active_only = String(params.active_only);
	}
	if (params?.category) {
		query.category = params.category;
	}
	return apiGet<MemoryRecord[]>('/api/memories', query);
}

/** Edit a memory's text and/or category. */
export async function editMemory(id: string, edit: MemoryEdit): Promise<MemoryRecord> {
	return apiPatch<MemoryRecord>(`/api/memories/${id}`, edit);
}

/** Soft-delete a memory: clears the active flag; the row is retained for restore. */
export async function softDeleteMemory(
	id: string
): Promise<{ soft_deleted: boolean; id: string }> {
	return apiDelete<{ soft_deleted: boolean; id: string }>(`/api/memories/${id}`);
}

/** Restore a soft-deleted memory. */
export async function restoreMemory(id: string): Promise<MemoryRecord> {
	return apiPost<MemoryRecord>(`/api/memories/${id}/restore`);
}
