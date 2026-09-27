/**
 * The registry of commands: every action the keyboard or the command palette
 * can run, with its default keys and the reason it cannot run in a context.
 *
 * Every default shortcut is a command here, under the action name the server
 * keeps custom bindings for, so a binding the reader changed still applies.
 * The shortcut handler starts from `defaultShortcuts()`, over the reader's
 * own keys (`chosenBindings()`), and the palette lists every command through
 * `commandOptions()`, each with the keys that run it for this reader: a
 * command that cannot run where the reader is stays listed, disabled, with
 * its reason in words, and the runner (lib/palette/run.ts) asks for that
 * reason before any handler runs. What each command does lives in the
 * runner; this file holds none of it, so it runs under Node as it is.
 *
 * Two rules come before a command's own reason. A key pressed while a
 * dialog of the keyboard is open (the palette, the list of shortcuts) runs
 * only what that dialog answers to (closing it, and ? for the list), since
 * anything else would act behind a modal dialog, where the reader cannot see
 * it: Ctrl+Enter in the palette's field does not send the draft under it.
 * The palette itself judges its commands from inside, and closes before one
 * runs. And a page outside both spaces (sign-in, registration, the component
 * gallery) has no shell and no palette: only the theme and closing a dialog
 * run there.
 *
 * Stop all is not a command: it is the one stop control, which every modal
 * dialog of the shell holds in its own head. The palette offers an entry
 * that opens that control's confirmation (lib/palette/sources.ts,
 * `stopOption`), never the stop itself.
 *
 * Pure data and pure functions, with no dependency.
 */

import type { Space } from '$lib/nav/destinations';
import type { PaletteOption } from './rank';

/** A command's default keys. Ctrl stands for Command on a Mac as well. */
export interface Binding {
	key: string;
	ctrl?: boolean;
	shift?: boolean;
	alt?: boolean;
}

/** Where a command would run. */
export interface CommandContext {
	space: Space;
	/** The conversation on screen, or null. */
	chatId: string | null;
	/** Whether a reply is being written in it. */
	streaming: boolean;
	/** Whether the command palette is open. */
	palette: boolean;
	/** Whether the list of shortcuts is open. */
	help: boolean;
	/**
	 * Whether the page is in one of the two spaces, where the shell and the
	 * palette live; false on sign-in, registration and the component gallery.
	 */
	shell?: boolean;
	/**
	 * Who asks: a key (the runner's own context says so, lib/palette/run.ts)
	 * or the palette, which judges its commands from inside and closes
	 * before one runs. A key is refused behind an open dialog of the
	 * keyboard; the palette is not.
	 */
	from?: 'keys' | 'palette';
}

/** A dialog of the keyboard: the command palette, or the list of shortcuts. */
export type KeyboardDialog = 'palette' | 'help';

export interface Command {
	/** The action name; the server keeps custom bindings under it. */
	id: string;
	label: string;
	/** Other words it is found by in the palette. */
	keywords?: readonly string[];
	binding?: Binding;
	/** The icon the palette draws beside it: a name of the icon set (lib/ds/icons.ts). */
	icon?: string;
	/**
	 * The dialogs of the keyboard a key may run it over; none by default, so
	 * a key pressed in an open palette or list runs nothing behind it.
	 */
	over?: readonly KeyboardDialog[];
	/** Whether it runs on a page outside both spaces; no by default. */
	anywhere?: boolean;
	/** Why it cannot run in `context`, in words; null when it can. */
	when?: (context: CommandContext) => string | null;
}

/** A shortcut as the handler matches it and its help lists it. */
export interface DefaultShortcut {
	key: string;
	ctrl: boolean;
	shift: boolean;
	alt: boolean;
	description: string;
	action: string;
}

function inAChat(reason: string) {
	return (context: CommandContext): string | null => (context.chatId ? null : reason);
}

export const COMMANDS: readonly Command[] = [
	{
		id: 'new_chat',
		label: 'New chat',
		icon: 'plus',
		keywords: ['new conversation', 'start'],
		binding: { key: 'n', ctrl: true }
	},
	{
		id: 'send_message',
		label: 'Send the message',
		icon: 'arrow-up',
		keywords: ['submit'],
		binding: { key: 'Enter', ctrl: true },
		when: (context) =>
			!context.chatId
				? 'Open a chat to send a message'
				: context.streaming
					? 'Wait for the reply to finish'
					: null
	},
	{
		id: 'stop_reply',
		label: 'Stop this reply',
		icon: 'stop-fill',
		keywords: ['cancel', 'interrupt'],
		when: (context) => (context.chatId && context.streaming ? null : 'No reply is being written')
	},
	{
		id: 'export_conversation',
		label: 'Export this chat',
		icon: 'download',
		keywords: ['download', 'save'],
		binding: { key: 'e', ctrl: true, shift: true },
		when: inAChat('Open a chat to export it')
	},
	{
		id: 'search_conversations',
		label: 'Search chats and commands',
		icon: 'search',
		keywords: ['command palette', 'find', 'go to'],
		binding: { key: 'k', ctrl: true },
		when: (context) => (context.palette ? 'The palette is open' : null)
	},
	{
		id: 'open_settings',
		label: 'Open Preferences',
		icon: 'sliders',
		keywords: ['settings', 'options'],
		binding: { key: ',', ctrl: true }
	},
	{
		id: 'toggle_theme',
		label: 'Switch between day and night',
		icon: 'bulb',
		keywords: ['theme', 'dark', 'light', 'colours'],
		binding: { key: 't', ctrl: true, shift: true },
		anywhere: true
	},
	{
		id: 'toggle_sidebar',
		label: 'Show or hide the sidebar',
		icon: 'panel-left',
		keywords: ['navigation', 'rail', 'drawer'],
		binding: { key: 'b', ctrl: true }
	},
	{
		id: 'show_shortcuts',
		label: 'Show keyboard shortcuts',
		icon: 'info',
		keywords: ['keys', 'help'],
		binding: { key: '?' },
		over: ['help']
	},
	{
		id: 'show_notifications',
		label: 'Show notifications',
		icon: 'info',
		keywords: ['notification history', 'alerts', 'messages']
	},
	{
		id: 'close_dialog',
		label: 'Close this dialog',
		icon: 'x',
		keywords: ['dismiss', 'escape'],
		binding: { key: 'Escape' },
		over: ['palette', 'help'],
		anywhere: true,
		when: (context) => (context.palette || context.help ? null : 'No dialog is open')
	}
];

/** The words a key is refused with while a dialog of the keyboard is open. */
const OPEN_DIALOG: Readonly<Record<KeyboardDialog, string>> = {
	palette: 'Close the palette first',
	help: 'Close the list of shortcuts first'
};

/** Why `command` cannot run in `context`, or null when it can. */
export function reasonFor(command: Command, context: CommandContext): string | null {
	if (context.shell === false && command.anywhere !== true) return 'Not available on this page';
	if (context.from === 'keys') {
		const over = command.over ?? [];
		for (const dialog of ['palette', 'help'] as const) {
			if (context[dialog] === true && !over.includes(dialog)) return OPEN_DIALOG[dialog];
		}
	}
	if (!command.when) return null;
	const reason = command.when(context);
	return typeof reason === 'string' && reason.trim() ? reason : null;
}

/**
 * Every command as the palette lists it: none left out, those that cannot
 * run disabled with their reason, each beside the keys that run it for this
 * reader (`bindings`, by action, over the registry's own).
 */
export function commandOptions(
	commands: readonly Command[],
	context: CommandContext,
	bindings: Readonly<Record<string, Binding>> = {}
): PaletteOption[] {
	return commands.map((command): PaletteOption => {
		const reason = reasonFor(command, context);
		const binding = bindings[command.id] ?? command.binding;
		return {
			id: command.id,
			group: 'commands',
			label: command.label,
			keywords: command.keywords ?? [],
			command: command.id,
			detail: binding ? bindingLabel(binding) : '',
			icon: command.icon,
			disabled: reason !== null,
			reason
		};
	});
}

/** The keys the reader chose for an action, as the server keeps them. */
export interface ChosenKeys {
	key?: string;
	ctrl?: boolean;
	shift?: boolean;
	alt?: boolean;
}

/**
 * The keys that run each bound command for this reader: the registry's,
 * with what the reader chose (`chosen`, by action) over them, field by
 * field. A choice for an action the registry does not bind is ignored.
 */
export function chosenBindings(
	commands: readonly Command[],
	chosen: Readonly<Record<string, ChosenKeys>> = {}
): Record<string, Binding> {
	const out: Record<string, Binding> = {};
	for (const command of commands) {
		if (!command.binding) continue;
		const mine = chosen && typeof chosen === 'object' ? chosen[command.id] : undefined;
		const own = mine && typeof mine === 'object' ? mine : {};
		out[command.id] = {
			key: typeof own.key === 'string' && own.key ? own.key : command.binding.key,
			ctrl: typeof own.ctrl === 'boolean' ? own.ctrl : command.binding.ctrl === true,
			shift: typeof own.shift === 'boolean' ? own.shift : command.binding.shift === true,
			alt: typeof own.alt === 'boolean' ? own.alt : command.binding.alt === true
		};
	}
	return out;
}

/**
 * The shortcuts the handler starts from: every command with keys, in the
 * registry's order, each with the keys `bindings` gives it (the registry's
 * by default).
 */
export function defaultShortcuts(
	commands: readonly Command[],
	bindings: Readonly<Record<string, Binding>> = {}
): DefaultShortcut[] {
	return commands
		.filter((command) => command.binding)
		.map((command): DefaultShortcut => {
			const binding = bindings[command.id] ?? (command.binding as Binding);
			return {
				key: binding.key,
				ctrl: binding.ctrl === true,
				shift: binding.shift === true,
				alt: binding.alt === true,
				description: command.label,
				action: command.id
			};
		});
}

// The keys named by a word, by the word in lower case (the server keeps
// keys in lower case).
const KEY_NAMES: Readonly<Record<string, string>> = {
	escape: 'Esc',
	enter: 'Enter',
	tab: 'Tab',
	space: 'Space',
	backspace: 'Backspace',
	delete: 'Delete',
	arrowup: 'Up',
	arrowdown: 'Down',
	arrowleft: 'Left',
	arrowright: 'Right'
};

/** A binding as its keys read: "Ctrl + Shift + E". */
export function bindingLabel(binding: Binding): string {
	const parts: string[] = [];
	if (binding.ctrl) parts.push('Ctrl');
	if (binding.shift) parts.push('Shift');
	if (binding.alt) parts.push('Alt');
	const key = typeof binding.key === 'string' ? binding.key : '';
	parts.push(KEY_NAMES[key.toLowerCase()] ?? (key.length === 1 ? key.toUpperCase() : key));
	return parts.join(' + ');
}
