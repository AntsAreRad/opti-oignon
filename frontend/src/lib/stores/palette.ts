/**
 * The command palette: whether it is open, and the words it opens with.
 *
 * One palette, mounted once by the layout of both spaces
 * (lib/components/palette/CommandPalette.svelte), shows what this store
 * says. Ctrl+K opens it through the registry's command, the sidebar's Search
 * entry opens it directly, and anything else that wants it calls
 * `openPalette()`: nothing on the way reads the page's markup, and nothing
 * travels as a window event.
 */

import { writable } from 'svelte/store';

export interface PaletteState {
	open: boolean;
	/** The words the field starts with when the palette opens. */
	words: string;
}

const CLOSED: PaletteState = { open: false, words: '' };

export const palette = writable<PaletteState>(CLOSED);

/**
 * Opens the palette, its field holding `words` (none by default). A caller
 * that hands it something other than words (an event) opens it empty.
 */
export function openPalette(words: string = ''): void {
	palette.set({ open: true, words: typeof words === 'string' ? words : '' });
}

/** Closes the palette. */
export function closePalette(): void {
	palette.set(CLOSED);
}
