/**
 * What each command of the registry does, and the one way to run one.
 *
 * The shortcut handler (lib/components/ui/KeyboardShortcuts.svelte) and the
 * command palette both run a command through `runCommand(id)`. It asks the
 * registry why the command cannot run where the reader is
 * (lib/palette/commands.ts, `reasonFor`) before it looks for a handler: a
 * command refused there does nothing, and `runCommand` returns the reason
 * in words; a command that runs returns null.
 *
 * Where the reader is (the space, whether the page is in one at all, the
 * conversation on screen, whether a reply is being written in it, whether
 * the palette or the list of shortcuts is open, and who asks) is read by one
 * rule, `contextAt`. `currentContext` reads it for a key; the palette reads
 * it for itself, draws its commands in it and runs the one chosen in it,
 * so what the palette shows disabled is exactly what the runner refuses.
 *
 * Every handler goes through a store or through navigation: the palette
 * opens through its store, the export dialog through its own, the list of
 * shortcuts through its own, the notification history through its own once
 * Preferences is on screen. Nothing here reads the page's markup; the one
 * event it sends, the composer's send, is heard by the composer.
 */

import { get } from 'svelte/store';
import { goto } from '$app/navigation';
import { COMMANDS, reasonFor, type CommandContext } from './commands';
import { DESTINATIONS, spaceHome } from '$lib/nav/destinations';
import { spaceOf } from '$lib/nav/space';
import { palette, openPalette, closePalette } from '$lib/stores/palette';
import { shortcutsHelp, toggleShortcutsHelp, closeShortcutsHelp } from '$lib/stores/shortcutsHelp';
import { conversations, createNewConversation } from '$lib/stores/conversations';
import { isStreaming, cancelCurrentGeneration } from '$lib/stores/chat';
import { openExportDialog } from '$lib/stores/exportDialog';
import { toggleDayNight } from '$lib/stores/preferences';
import { toggleSidebar } from '$lib/stores/ui';
import { toastError } from '$lib/stores/notifications';
import { openNotifications } from '$lib/stores/notificationCenter';

/** What the context is read from. */
export interface Surroundings {
	/** The path on screen. */
	pathname: string;
	/** Whether a reply is being written. */
	streaming: boolean;
	/** Whether the command palette is open. */
	palette: boolean;
	/** Whether the list of shortcuts is open. */
	help: boolean;
	/** Who asks: a key by default, or the palette. */
	from?: 'keys' | 'palette';
}

type Handler = (context: CommandContext) => void | Promise<void>;

/** The page of the chats destination, where one conversation is shown under it. */
function chatsPage(): string {
	const chats = DESTINATIONS.find((d) => d.id === 'chats');
	return chats ? chats.href.replace(/\/+$/, '') : '';
}

/** The conversation a path shows, or null: a page under the chats destination, one level down. */
export function chatOnScreen(pathname: string): string | null {
	const base = chatsPage();
	if (!base || typeof pathname !== 'string') return null;
	const prefix = `${base}/`;
	if (!pathname.startsWith(prefix)) return null;
	const rest = pathname.slice(prefix.length).replace(/\/+$/, '');
	if (!rest || rest.includes('/')) return null;
	try {
		return decodeURIComponent(rest);
	} catch {
		return null;
	}
}

/**
 * Where the reader is, as a command reads it: the space of the path (Use
 * for a page outside both), whether the path is in a space at all (the
 * shell's pages; not sign-in, registration or the component gallery), the
 * conversation it shows (never one in the Workshop), a reply being written
 * only in a conversation on screen, whether a dialog of the keyboard is
 * open, and who asks (a key unless the palette says it is asking).
 */
export function contextAt(where: Surroundings): CommandContext {
	const inSpace = spaceOf(where.pathname);
	const space = inSpace ?? 'use';
	const chatId = inSpace === 'use' ? chatOnScreen(where.pathname) : null;
	return {
		space,
		chatId,
		streaming: chatId !== null && where.streaming === true,
		palette: where.palette === true,
		help: where.help === true,
		shell: inSpace !== null,
		from: where.from === 'palette' ? 'palette' : 'keys'
	};
}

/** Where the reader is now, as a key asks. */
export function currentContext(): CommandContext {
	return contextAt({
		pathname: typeof window === 'undefined' ? '/' : window.location.pathname,
		streaming: get(isStreaming),
		palette: get(palette).open,
		help: get(shortcutsHelp)
	});
}

async function newChat(): Promise<void> {
	try {
		const id = await createNewConversation();
		await goto(`${chatsPage()}/${encodeURIComponent(id)}`);
	} catch {
		toastError('Failed to create a conversation');
	}
}

function exportChat(id: string | null): void {
	if (!id) return;
	const title = get(conversations).find((c) => c.id === id)?.title;
	openExportDialog(id, title || 'conversation');
}

function openPreferences(): Promise<void> {
	const preferences = DESTINATIONS.find((d) => d.id === 'preferences' && d.ready);
	return goto(preferences ? preferences.href : spaceHome('use', DESTINATIONS));
}

/** The notification history sits in Preferences: there first, then open. */
async function showNotifications(): Promise<void> {
	await openPreferences();
	openNotifications();
}

/** What each command does, once its context lets it run. */
export const HANDLERS: Readonly<Record<string, Handler>> = {
	new_chat: () => newChat(),
	send_message: () => {
		window.dispatchEvent(new CustomEvent('opti-send-message'));
	},
	stop_reply: (context) => {
		if (context.chatId) void cancelCurrentGeneration(context.chatId);
	},
	export_conversation: (context) => exportChat(context.chatId),
	search_conversations: () => openPalette(),
	open_settings: () => openPreferences(),
	toggle_theme: () => toggleDayNight(),
	toggle_sidebar: () => toggleSidebar(),
	show_shortcuts: () => toggleShortcutsHelp(),
	show_notifications: () => showNotifications(),
	close_dialog: (context) => {
		if (context.palette) closePalette();
		if (context.help) closeShortcutsHelp();
	}
};

/**
 * Runs the command `id` in `context` (where the reader is, as a key asks,
 * by default): null when it ran, else why it did not, in words.
 */
export function runCommand(id: string, context: CommandContext = currentContext()): string | null {
	const command = COMMANDS.find((c) => c.id === id);
	if (!command) return 'No such command';
	const reason = reasonFor(command, context);
	if (reason !== null) return reason;
	const handler = HANDLERS[id];
	if (!handler) return 'This command does nothing yet';
	const failed = () => toastError(`${command.label} failed`);
	try {
		const done = handler(context);
		if (done instanceof Promise) done.catch(failed);
	} catch {
		failed();
	}
	return null;
}
