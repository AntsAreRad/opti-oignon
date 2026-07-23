/**
 * Typed API client for the Notes update log (N.8 editor seam).
 *
 * Operates over the per-note update legs (/api/notes/{id}/updates) the
 * sibling route exposes on top of the append-only update store: append
 * one opaque Yjs update, and replay the surviving tail from a seq. Mirrors the
 * api/notes.ts shape: a thin typed wrapper over the base client, one async
 * function per endpoint, no Svelte state here.
 *
 * The update blob is an opaque, client-owned Yjs update carried base64-encoded;
 * the backend never interprets it. No author or device identity rides the
 * request (decision N9-D3): the per-(user, note) seq is minted by the store and
 * the local author's signature is attached by the sync engine at publish.
 */

import { apiGet, apiPost } from './client';

/** One appended update (mirrors the backend NoteUpdateRecordSchema). */
export interface NoteUpdateRecord {
	id: number;
	note_id: string;
	seq: number;
	/** The opaque, client-owned Yjs update, base64-encoded. */
	update_blob_b64: string;
	/** Informational local metadata; absent on a locally appended row. */
	author_device: string | null;
	created_at: string;
}

/**
 * Append one opaque update to a note's log.
 *
 * Resolves with the appended record (carrying the store-minted seq) ONLY after
 * the local backend acknowledges the append -- the confirmed posture the editor
 * renders behind (NOTES_CRDT_SPEC.md section 5): nothing renders that the
 * backend has not recorded. A refusal rejects (the caller surfaces it and
 * leaves the display at server truth); the eventual remote echo is a no-op by
 * Yjs idempotence.
 */
export async function appendNoteUpdate(
	noteId: string,
	blobB64: string
): Promise<NoteUpdateRecord> {
	return apiPost<NoteUpdateRecord>(`/api/notes/${noteId}/updates`, {
		update_blob_b64: blobB64,
	});
}

/**
 * Replay the surviving update tail (section 4) from a seq.
 *
 * A fresh or behind device bootstraps from the checkpoint body (the whole-note
 * row) and then replays this tail to converge.
 */
export async function fetchNoteUpdates(
	noteId: string,
	afterSeq = 0
): Promise<NoteUpdateRecord[]> {
	const query: Record<string, string> = { after_seq: String(afterSeq) };
	return apiGet<NoteUpdateRecord[]>(`/api/notes/${noteId}/updates`, query);
}
