/**
 * The export dialog: one, mounted by the shell, opened from anywhere.
 *
 * The export shortcut, the chat's header and any other caller open it
 * through `openExportDialog(id, title)`; the shell's layout draws it while
 * it is open. Nothing travels as a window event.
 */

import { writable } from 'svelte/store';

export interface ExportRequest {
	open: boolean;
	/** The conversation to export. */
	id: string;
	/** Its title, for the file name and the dialog's heading. */
	title: string;
}

const CLOSED: ExportRequest = { open: false, id: '', title: '' };

export const exportDialog = writable<ExportRequest>(CLOSED);

/** Opens the dialog on conversation `id`; an empty id opens nothing. */
export function openExportDialog(id: string, title: string): void {
	if (!id) return;
	exportDialog.set({ open: true, id, title: title || 'conversation' });
}

/** Closes the dialog. */
export function closeExportDialog(): void {
	exportDialog.set(CLOSED);
}
