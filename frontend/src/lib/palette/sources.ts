/**
 * What the command palette lists, built from where each kind of option lives.
 *
 * Destinations are the visible ones of the destination table
 * (lib/nav/destinations.ts, through visibleDestinations): the ready ones,
 * and the componion's only while its switch is on. Settings groups are the
 * settings search's index (lib/settings/search.ts, settingsIndex): every
 * group of the catalog that has a page, in both spaces, an embedded group
 * linked to its host's page, never a retired one; each is found by the
 * words the settings page's own search reads too: its synonyms, its
 * description, the page that holds it and the old section that listed it.
 * Commands are the registry's (lib/palette/commands.ts, commandOptions),
 * with one entry beside them that is not a command: Stop all, which opens
 * the stop control's confirmation in the palette's head and never stops
 * anything by itself. Chats are the conversations: the recent ones while
 * nothing is typed, then the server's answer to the words
 * (lib/palette/conversationSource.ts), each linked under the chats
 * destination; when the server found more than the group shows, a last
 * entry opens the chats index on the same words, where every match is
 * listed.
 *
 * Pure functions, with no dependency: the palette hands them the table's
 * entries, the index and the conversations, and they run under Node as they
 * are.
 */

import type { PaletteOption } from './rank';

/** What the palette reads of a destination. */
export interface DestinationEntry {
	id: string;
	label: string;
	href: string;
	space: 'use' | 'workshop';
	/** A name of the icon set (lib/ds/icons.ts). */
	icon?: string;
}

/** What the palette reads of a settings group the index found. */
export interface SettingEntry {
	id: string;
	title: string;
	/** The page that holds it, with the group to open there. */
	href: string;
	/** Where that page is, in words. */
	where: string;
	/** The group's description. */
	description?: string;
	/** The name of the old settings section that listed it, or ''. */
	former?: string;
}

/** What the palette reads of a conversation. */
export interface ChatEntry {
	id: string;
	title?: string | null;
}

/** The destinations to list, in the order given. */
export function destinationOptions(visible: readonly DestinationEntry[]): PaletteOption[] {
	return visible.map(
		(destination): PaletteOption => ({
			id: destination.id,
			group: 'destinations',
			label: destination.label,
			href: destination.href,
			detail: destination.space === 'workshop' ? 'Workshop' : 'Use',
			icon: destination.icon
		})
	);
}

/**
 * The settings groups to list, in the index's order, each found by its
 * synonyms (read from `groups`), where it lives, its description and the
 * old section that listed it.
 */
export function settingOptions(
	index: readonly SettingEntry[],
	groups: readonly { id: string; synonyms?: readonly string[] }[] = []
): PaletteOption[] {
	const synonyms = new Map(groups.map((group) => [group.id, group.synonyms ?? []]));
	return index.map(
		(hit): PaletteOption => ({
			id: hit.id,
			group: 'settings',
			label: hit.title,
			href: hit.href,
			detail: hit.where,
			keywords: [...(synonyms.get(hit.id) ?? []), hit.where, hit.description ?? '', hit.former ?? ''].filter(
				(words) => typeof words === 'string' && words.trim() !== ''
			),
			icon: 'sliders'
		})
	);
}

/**
 * The chats to list, linked under the chats destination (`chatsHref`);
 * `found` says the server found them for the words being searched.
 */
export function chatOptions(
	chats: readonly ChatEntry[],
	chatsHref: string,
	found = false
): PaletteOption[] {
	const base = chatsHref.replace(/\/+$/, '');
	return chats.map(
		(chat): PaletteOption => ({
			id: chat.id,
			group: 'chats',
			label: (typeof chat.title === 'string' && chat.title.trim()) || 'Untitled chat',
			href: `${base}/${encodeURIComponent(chat.id)}`,
			found,
			icon: 'chat'
		})
	);
}

/**
 * The Stop all entry: found by the words a reader types to stop things,
 * listed with the commands, and marked `stop`, so the palette opens the stop
 * control's confirmation in its head instead of running anything. Disabled,
 * with `refusal` as its reason, when the stop control says the confirmation
 * cannot be asked for now (everything is stopped already, or the server
 * cannot stop).
 */
export function stopOption(refusal: string | null = null): PaletteOption {
	const reason = typeof refusal === 'string' && refusal.trim() ? refusal : null;
	return {
		id: 'stop_all',
		group: 'commands',
		label: 'Stop all',
		keywords: ['emergency stop', 'halt', 'stop everything', 'kill'],
		detail: 'Asks first',
		icon: 'stop',
		stop: true,
		disabled: reason !== null,
		reason
	};
}

/**
 * The last entry of the chats group when the server found more chats for
 * `words` than the group shows (`found` of them, `shown` at most): it opens
 * the chats index on the same words, where every match is listed. Null
 * otherwise.
 */
export function moreChatsOption(
	words: string,
	chatsHref: string,
	found: number,
	shown: number
): PaletteOption | null {
	const trimmed = typeof words === 'string' ? words.trim() : '';
	if (!trimmed || !(found > shown)) return null;
	const base = chatsHref.replace(/\/+$/, '') || '/';
	return {
		id: 'more_chats',
		group: 'chats',
		label: `Every chat matching "${trimmed}"`,
		href: `${base}?q=${encodeURIComponent(trimmed)}`,
		detail: 'Chats index',
		icon: 'search',
		trailing: true
	};
}
