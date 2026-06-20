/**
 * Typed API client for the note attachments surface (N.5 / N.6, S253).
 *
 * Operates over the per-user S249 attachments route
 * (/api/notes/attachments): multipart upload (the bytes sealed server-side
 * under a per-attachment subkey), list per note, manifest read, decrypted
 * blob fetch, and delete (blob first, then manifest). Mirrors the
 * api/notes.ts shape: a thin typed wrapper over the base client, one async
 * function per endpoint, no Svelte state here (the store lives in
 * $lib/stores/attachments).
 *
 * The blob leg returns raw bytes, not JSON, so it cannot ride the JSON
 * helpers: fetchAttachmentBlob does an authed fetch (Bearer when a token is
 * set, cookies always) and returns a Blob the caller turns into a short-lived
 * object URL. The decrypted bytes exist in browser memory only; persisting
 * them anywhere is the caller's deliberate act, never this client's.
 */

import { apiGet, apiDelete, apiUpload, getAccessToken, ApiError } from './client';

/** The attachment kinds the backend accepts (notes_store.ATTACHMENT_KINDS). */
export type AttachmentKind = 'audio' | 'image' | 'drawing';

/** An attachment's manifest row (mirrors the backend AttachmentSchema).
 * transcript_text (audio) and caption_text / ocr_text (image) stay null until
 * the opt-in, sandboxed post-processing writes them back on approval. */
export interface AttachmentRecord {
	id: string;
	note_id: string;
	kind: string;
	mime: string;
	byte_size: number;
	nonce: string;
	created_at: string;
	transcript_text: string | null;
	caption_text: string | null;
	ocr_text: string | null;
}

/** The outcome of deleting one attachment (mirrors AttachmentDeleteResponse). */
export interface AttachmentDeleteResult {
	deleted: boolean;
	id: string;
}

/** Same base resolution as the base client (VITE_API_URL or same-origin). */
const API_BASE: string = (import.meta.env.VITE_API_URL as string | undefined) ?? '';

/**
 * Upload one media blob to a note as a multipart form (kind + file).
 *
 * The backend validates the kind against its allowlist BEFORE sealing (a
 * rejected kind is a 422 with no orphan blob) and refuses with a 503 when no
 * master key is loaded -- never a plaintext write.
 */
export async function uploadAttachment(
	noteId: string,
	kind: AttachmentKind,
	file: Blob,
	filename: string = 'attachment'
): Promise<AttachmentRecord> {
	const form = new FormData();
	form.append('kind', kind);
	form.append('file', file, filename);
	return apiUpload<AttachmentRecord>(
		`/api/notes/attachments/note/${encodeURIComponent(noteId)}`,
		form
	);
}

/** List one note's attachment manifests (oldest first). */
export async function listAttachments(noteId: string): Promise<AttachmentRecord[]> {
	return apiGet<AttachmentRecord[]>(
		`/api/notes/attachments/note/${encodeURIComponent(noteId)}`
	);
}

/** Fetch one attachment's manifest; a missing or cross-user id is a 404. */
export async function getAttachmentMeta(attachmentId: string): Promise<AttachmentRecord> {
	return apiGet<AttachmentRecord>(
		`/api/notes/attachments/${encodeURIComponent(attachmentId)}`
	);
}

/**
 * Fetch one attachment's decrypted bytes as a Blob (in memory only).
 *
 * The server decrypts in memory and streams; this client holds the bytes in
 * a Blob the caller wraps in URL.createObjectURL for playback / thumbnails
 * and revokes when done.
 */
export async function fetchAttachmentBlob(attachmentId: string): Promise<Blob> {
	const headers: Record<string, string> = { Accept: 'application/octet-stream' };
	const token = getAccessToken();
	if (token) {
		headers['Authorization'] = `Bearer ${token}`;
	}
	const path = `/api/notes/attachments/${encodeURIComponent(attachmentId)}/blob`;
	let response: Response;
	try {
		response = await fetch(`${API_BASE}${path}`, {
			headers,
			credentials: 'include'
		});
	} catch (err) {
		const msg = err instanceof Error ? err.message : 'Network error';
		throw new ApiError(0, `Cannot reach the backend for ${path}: ${msg}`, msg, true);
	}
	if (!response.ok) {
		let detail = '';
		try {
			detail = await response.text();
		} catch {
			detail = '';
		}
		throw new ApiError(
			response.status,
			`Attachment blob fetch failed (${response.status})`,
			detail
		);
	}
	return response.blob();
}

/** Delete an attachment (the encrypted blob first, then the manifest row). */
export async function deleteAttachment(
	attachmentId: string
): Promise<AttachmentDeleteResult> {
	return apiDelete<AttachmentDeleteResult>(
		`/api/notes/attachments/${encodeURIComponent(attachmentId)}`
	);
}
