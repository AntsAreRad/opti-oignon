/**
 * Svelte stores for notes state management (N.2).
 *
 * Reactive stores and action functions over the per-user /api/notes surface,
 * mirroring stores/conversations.ts: writable stores, derived views, and async
 * actions that call $lib/api/notes and update state. The note body is opaque on
 * the wire; this store carries it as the base64 field and leaves text encoding
 * to the editor (via encodeNoteBody / decodeNoteBody in the client).
 */

import { writable, derived, get } from 'svelte/store';
import * as api from '$lib/api/notes';

// -- Stores --

/** All loaded notes (most-recently-updated first, as the API returns them). */
export const notes = writable<api.NoteRecord[]>([]);

/** The id of the note currently open in the editor, or null. */
export const activeNoteId = writable<string | null>(null);

/** True while the list is loading. */
export const loading = writable<boolean>(false);

/** The current error message, or null. */
export const error = writable<string | null>(null);

/** The current search query (matched against title and tags). */
export const search = writable<string>('');

// -- Derived --

/** The note matching activeNoteId, or null. */
export const activeNote = derived(
	[notes, activeNoteId],
	([$notes, $activeId]) => {
		if (!$activeId) return null;
		return $notes.find((n) => n.id === $activeId) ?? null;
	}
);

/** The visible notes: non-deleted, matching the search query, pinned first. */
export const filteredNotes = derived([notes, search], ([$notes, $search]) => {
	const query = $search.trim().toLowerCase();
	const visible = $notes.filter((n) => {
		if (n.deleted) return false;
		if (!query) return true;
		const inTitle = n.title.toLowerCase().includes(query);
		const inTags = n.tags.some((t) => t.toLowerCase().includes(query));
		return inTitle || inTags;
	});
	// Stable pinned-first ordering; the API's updated-desc order is preserved
	// within each group.
	return visible
		.map((n, i) => ({ n, i }))
		.sort((a, b) => {
			if (a.n.pinned !== b.n.pinned) return a.n.pinned ? -1 : 1;
			return a.i - b.i;
		})
		.map((x) => x.n);
});

// -- Actions --

/** Load the user's notes (tombstoned notes excluded). */
export async function loadNotes(): Promise<void> {
	loading.set(true);
	error.set(null);
	try {
		const list = await api.listNotes({ include_deleted: false });
		notes.set(list);
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to load notes';
		error.set(msg);
		notes.set([]);
	} finally {
		loading.set(false);
	}
}

/** Open a note in the editor. */
export function selectNote(id: string | null): void {
	activeNoteId.set(id);
}

/** Create a note, prepend it to the list, and open it. Return the new id. */
export async function createNote(input: api.NoteCreate): Promise<string> {
	error.set(null);
	try {
		const note = await api.createNote(input);
		notes.update((list) => [note, ...list]);
		activeNoteId.set(note.id);
		return note.id;
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to create note';
		error.set(msg);
		throw err;
	}
}

/** Save changes to a note (title / body / tags / pinned). Return the updated note. */
export async function saveNote(
	id: string,
	update: api.NoteUpdate
): Promise<api.NoteRecord> {
	error.set(null);
	try {
		const updated = await api.updateNote(id, update);
		notes.update((list) => list.map((n) => (n.id === id ? updated : n)));
		return updated;
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to save note';
		error.set(msg);
		throw err;
	}
}

/** Toggle a note's pinned flag. Return the updated note. */
export async function togglePin(note: api.NoteRecord): Promise<api.NoteRecord> {
	return saveNote(note.id, { pinned: !note.pinned });
}

/**
 * Toggle a note's phone-sync opt-in (mobile_allowed). CONFIRMED posture by
 * construction: saveNote replaces the row from the PATCH response, so the
 * rendered flag is only ever the server's truth -- never an optimistic local
 * flip. The flag rides the existing PATCH leg; the backend writes it through
 * its dedicated setter only (N9-D3). Return the updated note.
 */
export async function toggleMobileAllowed(
	note: api.NoteRecord
): Promise<api.NoteRecord> {
	return saveNote(note.id, { mobile_allowed: !note.mobile_allowed });
}

/** Soft-delete a note (tombstone) and remove it from the list. Deselect it if
 * it was open. */
export async function removeNote(id: string): Promise<void> {
	error.set(null);
	try {
		await api.deleteNote(id);
		notes.update((list) => list.filter((n) => n.id !== id));
		if (get(activeNoteId) === id) {
			activeNoteId.set(null);
		}
	} catch (err: unknown) {
		const msg = err instanceof Error ? err.message : 'Failed to delete note';
		error.set(msg);
		throw err;
	}
}
