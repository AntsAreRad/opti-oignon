/**
 * Typed API client for the Notes data layer (N.2).
 *
 * Operates over the per-user NotesStore surface (/api/notes): list, get,
 * create, update, and soft-delete (tombstone), per user and encrypted at rest.
 * Mirrors the api/memories.ts shape: a thin typed wrapper over the base client,
 * one async function per endpoint, no Svelte state here (the store lives in
 * $lib/stores/notes).
 *
 * The note body is an opaque, client-owned CRDT carried base64-encoded; the
 * backend never interprets it. Until the Yjs text CRDT lands (N.8), the body is
 * plain text/markdown carried opaque: encodeNoteBody / decodeNoteBody convert
 * between the editor's text and the base64 wire field. Tags are an OR-Set
 * carried directly as a string array; the backend owns the opaque JSON-array
 * encoding.
 */

import { apiGet, apiPost, apiPatch, apiDelete } from './client';

/** A note's metadata and opaque CRDT body (mirrors the backend NoteSchema). */
export interface NoteRecord {
	id: string;
	title: string;
	/** The opaque, client-owned CRDT body, base64-encoded. */
	body_crdt_b64: string;
	tags: string[];
	pinned: boolean;
	/**
	 * N.9 / S256: the per-item phone-sync opt-in (false by default, the secure
	 * default). Rides the existing PATCH leg; the backend flips it only
	 * through the store's dedicated setter (decision N9-D3).
	 */
	mobile_allowed: boolean;
	created_at: string;
	updated_at: string;
	deleted: boolean;
}

/** Fields for creating a note; the body is optional (an empty note is valid). */
export interface NoteCreate {
	title: string;
	body_crdt_b64?: string;
	tags?: string[];
	pinned?: boolean;
}

/** Fields for updating a note; omit a field to leave it unchanged. An empty
 * tags array is a deliberate clear. */
export interface NoteUpdate {
	title?: string;
	body_crdt_b64?: string;
	tags?: string[];
	pinned?: boolean;
	/**
	 * N.9 / S260: the phone-sync opt-in, riding the existing PATCH (omitted
	 * means unchanged). A human trust decision made at the desktop; the route
	 * flips it through the dedicated setter only, never the generic path.
	 */
	mobile_allowed?: boolean;
}

/**
 * Encode the editor's text as the base64 opaque body the wire carries.
 * UTF-8 safe: encodes to bytes first, then base64 (btoa is latin1-only).
 */
export function encodeNoteBody(text: string): string {
	const bytes = new TextEncoder().encode(text ?? '');
	let binary = '';
	for (let i = 0; i < bytes.length; i++) {
		binary += String.fromCharCode(bytes[i]);
	}
	return btoa(binary);
}

/** Decode the base64 opaque body back to the editor's text. Tolerant: an empty
 * or invalid value yields an empty string rather than throwing. */
export function decodeNoteBody(b64: string): string {
	if (!b64) {
		return '';
	}
	try {
		const binary = atob(b64);
		const bytes = new Uint8Array(binary.length);
		for (let i = 0; i < binary.length; i++) {
			bytes[i] = binary.charCodeAt(i);
		}
		return new TextDecoder().decode(bytes);
	} catch {
		return '';
	}
}

/** List the user's notes (most-recently-updated first). */
export async function listNotes(params?: {
	pinned_only?: boolean;
	include_deleted?: boolean;
	limit?: number;
}): Promise<NoteRecord[]> {
	const query: Record<string, string> = {};
	if (params?.pinned_only !== undefined) {
		query.pinned_only = String(params.pinned_only);
	}
	if (params?.include_deleted !== undefined) {
		query.include_deleted = String(params.include_deleted);
	}
	if (params?.limit !== undefined) {
		query.limit = String(params.limit);
	}
	return apiGet<NoteRecord[]>('/api/notes', query);
}

/** Fetch one note by id; a missing or tombstoned note is a 404. */
export async function getNote(id: string): Promise<NoteRecord> {
	return apiGet<NoteRecord>(`/api/notes/${id}`);
}

/** Create a note. The body is stored opaque; tags are an OR-Set. */
export async function createNote(input: NoteCreate): Promise<NoteRecord> {
	return apiPost<NoteRecord>('/api/notes', input);
}

/** Update title / body / tags / pinned. Omitted fields are left unchanged. */
export async function updateNote(id: string, update: NoteUpdate): Promise<NoteRecord> {
	return apiPatch<NoteRecord>(`/api/notes/${id}`, update);
}

/** Soft-delete (tombstone) a note so the deletion syncs (CRDT-safe). */
export async function deleteNote(id: string): Promise<{ deleted: boolean; id: string }> {
	return apiDelete<{ deleted: boolean; id: string }>(`/api/notes/${id}`);
}
