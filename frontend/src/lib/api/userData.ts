/**
 * API client for the user's own data: the export, the wipe's preview, the wipe.
 *
 *   GET    /api/users/{id}/export              -- everything the account holds, as JSON
 *   GET    /api/users/{id}/data/wipe-preview   -- what a wipe removes and leaves; `?remove=notes&remove=projects`
 *                                                 counts what is named and gives the phrase to repeat
 *   DELETE /api/users/{id}/data                -- the wipe; `?remove=...&confirm=<the phrase>` removes what is named
 */

import { apiDelete, apiGet } from './client';
import { wipeQuery } from '$lib/settings/yourData';

export interface WipePreview {
	user_id: string;
	single_user: boolean;
	remove: string[];
	removed: string[];
	left: string[];
	retained_by_design: string[];
	/** The content removed only when named, by its word, with the stores it covers (single-user mode only). */
	on_request: Record<string, string[]>;
	counts?: {
		notes?: { notes: number; attachments: number; disputed: number | null; deleted_notes: number; code: string };
		projects?: { projects: number; files: number; outputs: number; code: string };
		knowledge?: { collections: number; documents: number; chunks: number; code: string };
		skills?: { skills: number; published: number; drafts: number; versions: number; code: string };
	};
	phrase?: string;
	notice?: string;
}

export interface WipeResult {
	user_id: string;
	deleted_at: number;
	conversations: number;
	memories: number;
	rag_collections: number;
	plugin_configs: number;
	plugin_reviews: number;
	pending_writes: number;
	settings: boolean;
	encryption_keys: boolean;
	content_removed: Record<string, number>;
	not_covered: string[];
	retained_by_design: string[];
}

function userPath(userId: string): string {
	return `/api/users/${encodeURIComponent(userId)}`;
}

/** Everything the account holds, as the export gives it. */
export async function exportUserData(userId: string): Promise<Record<string, unknown>> {
	return apiGet<Record<string, unknown>>(`${userPath(userId)}/export`);
}

/** Download the export as a JSON file. */
export async function downloadUserData(userId: string): Promise<void> {
	const data = await exportUserData(userId);
	const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = `opti-oignon-my-data-${new Date().toISOString().slice(0, 10)}.json`;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
	URL.revokeObjectURL(url);
}

/** What a wipe would remove and leave, with the named content counted. Reads only. */
export async function previewWipe(userId: string, remove: string[]): Promise<WipePreview> {
	return apiGet<WipePreview>(`${userPath(userId)}/data/wipe-preview${wipeQuery(remove)}`);
}

/** The wipe, with the named content and the preview's phrase repeated. */
export async function wipeUserData(userId: string, remove: string[], phrase?: string): Promise<WipeResult> {
	return apiDelete<WipeResult>(`${userPath(userId)}/data${wipeQuery(remove, phrase)}`);
}
