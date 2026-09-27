/**
 * The keys that run each bound command for this reader: the registry's
 * (lib/palette/commands.ts), with the keys the reader chose over them.
 *
 * The shortcut handler (lib/components/ui/KeyboardShortcuts.svelte) loads
 * the reader's keys from the server and writes them here; the handler, the
 * command palette's entries and the sidebar's Search hint all read them from
 * here, so the keys shown are the keys that work.
 */

import { writable } from 'svelte/store';
import { COMMANDS, chosenBindings, type Binding } from '$lib/palette/commands';

export const shortcutKeys = writable<Record<string, Binding>>(chosenBindings(COMMANDS));
