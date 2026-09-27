/**
 * The list of keyboard shortcuts: whether it is open.
 *
 * The shortcut handler (lib/components/ui/KeyboardShortcuts.svelte) draws
 * the list while this store says so; the registry's commands open and
 * close it (lib/palette/run.ts), from a key or from the command palette,
 * and read it to know whether a dialog is open.
 */

import { writable } from 'svelte/store';

export const shortcutsHelp = writable<boolean>(false);

/** Shows the list if it is shut, shuts it if it is shown. */
export function toggleShortcutsHelp(): void {
	shortcutsHelp.update((open) => !open);
}

/** Shuts the list. */
export function closeShortcutsHelp(): void {
	shortcutsHelp.set(false);
}
