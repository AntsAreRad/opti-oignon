/**
 * The notification history's panel: whether it is open.
 *
 * The history sits in Preferences (lib/components/ui/NotificationCenter.svelte,
 * drawn by the settings hub of the Use space). Its bell opens and shuts the
 * panel through this store, and so does the registry's "Show notifications"
 * command (lib/palette/run.ts), from a key or from the command palette, after
 * going to Preferences.
 */

import { writable } from 'svelte/store';

export const notificationCenter = writable<boolean>(false);

/** Opens the history's panel. */
export function openNotifications(): void {
	notificationCenter.set(true);
}

/** Shuts it. */
export function closeNotifications(): void {
	notificationCenter.set(false);
}

/** Shuts it if it is open, opens it if it is shut. */
export function toggleNotifications(): void {
	notificationCenter.update((open) => !open);
}
