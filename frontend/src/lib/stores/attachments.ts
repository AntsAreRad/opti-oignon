/**
 * Svelte stores for note-attachment state management (N.5 / N.6, S253).
 *
 * Reactive stores and action functions over the per-user media surfaces:
 * the S249 attachments route via $lib/api/attachments, the S250 transcription
 * trigger via $lib/api/transcription, and the S251 caption / OCR trigger via
 * $lib/api/caption. Mirrors stores/notes.ts: writable stores, derived views,
 * async actions that call the clients and update state.
 *
 * The store tracks ONE note's attachments at a time (the note open in the
 * editor). loadAttachments is idempotent per note id, so NotesVoiceCapture
 * and NotesMediaGallery can both call it reactively without a double fetch.
 * An approved transcription / caption (written_back) updates the manifest
 * row in place so the UI reflects the persisted text without a reload; a
 * preview (approve=false) is returned to the caller and NOT written into the
 * list, the safe default.
 */

import { writable, derived, get } from 'svelte/store';
import * as api from '$lib/api/attachments';
import { requestTranscription, type TranscriptionResult } from '$lib/api/transcription';
import { requestCaption, type CaptionResult } from '$lib/api/caption';

// -- Stores --

/** The loaded attachments of the note in attachmentsNoteId (oldest first). */
export const attachments = writable<api.AttachmentRecord[]>([]);

/** The note id the attachments list belongs to, or null. */
export const attachmentsNoteId = writable<string | null>(null);

/** True while the list is loading. */
export const mediaLoading = writable<boolean>(false);

/** The current media error message, or null. */
export const mediaError = writable<string | null>(null);

// -- Derived --

/** The audio attachments (the voice-capture list). */
export const audioAttachments = derived(attachments, ($attachments) =>
	$attachments.filter((a) => a.kind === 'audio')
);

/** The image attachments (the gallery list). */
export const imageAttachments = derived(attachments, ($attachments) =>
	$attachments.filter((a) => a.kind === 'image')
);

/** The drawing attachments (the canvas list, Notes feature N.7, S254). */
export const drawingAttachments = derived(attachments, ($attachments) =>
	$attachments.filter((a) => a.kind === 'drawing')
);

// -- Actions --

/**
 * Load a note's attachments. Idempotent per note id: a repeat call for the
 * already-loaded note is a no-op unless force is true, so two components can
 * both ask for the same note without a double fetch.
 */
export async function loadAttachments(
	noteId: string,
	force: boolean = false
): Promise<void> {
	if (!force && get(attachmentsNoteId) === noteId) {
		return;
	}
	attachmentsNoteId.set(noteId);
	mediaLoading.set(true);
	mediaError.set(null);
	try {
		const list = await api.listAttachments(noteId);
		// The note may have changed while the fetch was in flight; only the
		// current note's result lands.
		if (get(attachmentsNoteId) === noteId) {
			attachments.set(list);
		}
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to load attachments';
		if (get(attachmentsNoteId) === noteId) {
			mediaError.set(msg);
			attachments.set([]);
		}
	} finally {
		if (get(attachmentsNoteId) === noteId) {
			mediaLoading.set(false);
		}
	}
}

/** Clear the list (when the editor closes or the note is deselected). */
export function clearAttachments(): void {
	attachments.set([]);
	attachmentsNoteId.set(null);
	mediaError.set(null);
	mediaLoading.set(false);
}

/** Upload one media blob to the current note and append its manifest row. */
export async function uploadNoteAttachment(
	noteId: string,
	kind: api.AttachmentKind,
	file: Blob,
	filename?: string
): Promise<api.AttachmentRecord> {
	mediaError.set(null);
	try {
		const record = await api.uploadAttachment(noteId, kind, file, filename);
		if (get(attachmentsNoteId) === noteId) {
			attachments.update((list) => [...list, record]);
		}
		return record;
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to upload attachment';
		mediaError.set(msg);
		throw err;
	}
}

/** Delete an attachment (blob then manifest) and drop it from the list. */
export async function removeAttachment(attachmentId: string): Promise<void> {
	mediaError.set(null);
	try {
		await api.deleteAttachment(attachmentId);
		attachments.update((list) => list.filter((a) => a.id !== attachmentId));
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to delete attachment';
		mediaError.set(msg);
		throw err;
	}
}

/** Replace one manifest row in place (after an approved write-back). */
function patchRecord(attachmentId: string, patch: Partial<api.AttachmentRecord>): void {
	attachments.update((list) =>
		list.map((a) => (a.id === attachmentId ? { ...a, ...patch } : a))
	);
}

/**
 * Trigger the opt-in transcription of one audio attachment.
 *
 * approve=false previews; approve=true persists. On written_back the
 * transcript_text lands in the manifest row so the UI updates in place. The
 * structured result (incl. a refusal with its reason) is returned to the
 * caller; only the 503 availability guard raises, and it lands in mediaError.
 */
export async function transcribeAttachment(
	attachmentId: string,
	approve: boolean = false
): Promise<TranscriptionResult> {
	mediaError.set(null);
	try {
		const result = await requestTranscription(attachmentId, approve);
		if (result.ok && result.written_back) {
			patchRecord(attachmentId, { transcript_text: result.transcript_text });
		}
		return result;
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Transcription request failed';
		mediaError.set(msg);
		throw err;
	}
}

/**
 * Trigger the opt-in caption / OCR of one image attachment.
 *
 * approve=false previews; approve=true persists. On written_back the
 * caption_text / ocr_text land in the manifest row so the UI updates in
 * place. The structured result (incl. a refusal with its reason) is returned
 * to the caller; only the 503 availability guard raises, into mediaError.
 */
export async function captionAttachment(
	attachmentId: string,
	approve: boolean = false
): Promise<CaptionResult> {
	mediaError.set(null);
	try {
		const result = await requestCaption(attachmentId, approve);
		if (result.ok && result.written_back) {
			patchRecord(attachmentId, {
				caption_text: result.caption_text,
				ocr_text: result.ocr_text
			});
		}
		return result;
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Caption request failed';
		mediaError.set(msg);
		throw err;
	}
}
