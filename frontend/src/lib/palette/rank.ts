/**
 * The command palette's options, and the order it lists them in.
 *
 * An option is a destination, a command, a settings group or a chat, each
 * in its group. A search ranks the options of a group by how the words meet
 * the option's label: the whole label (exact), its start (prefix), the start
 * of a later word (word prefix), or its letters in the same order
 * (subsequence), in that order of rank. An option is also found by its
 * keywords (a setting's synonyms and the page that holds it, a command's
 * other names), after every exact, prefix or word prefix match of a label
 * (a keyword is a hint, the label is the name the reader sees) and before
 * letters in order; a keyword counts by exact, prefix or word prefix only:
 * letters in order, spread over a phrase, would find almost anything. A chat
 * the server found for these very words (in its title or in its messages) is
 * kept, after every option whose label or keywords met them. Case, accents
 * and the spaces around the words are ignored. Options that tie keep the
 * order they were given in.
 *
 * Each group shows at most `limit` options (GROUP_LIMIT unless told), and
 * the groups follow the rank of their best option, ties in GROUP_ORDER, so
 * the first option listed is the best match. An option marked `trailing`
 * (the way to every chat that matched, when the server found more than the
 * group shows) is not ranked by the words: it closes its group, past the
 * limit, whenever that group lists something for them. With no words the
 * palette lists every destination, then the recent chats (at most `limit`
 * of them), and nothing else.
 *
 * Pure functions, with no dependency: the palette hands them its options,
 * and they run under Node as they are.
 */

export type PaletteGroupId = 'destinations' | 'commands' | 'settings' | 'chats';

/** The groups, in the order they are listed when their best options tie. */
export const GROUP_ORDER: readonly PaletteGroupId[] = ['destinations', 'commands', 'settings', 'chats'];

/** The label each group is shown under. */
export const GROUP_LABELS: Readonly<Record<PaletteGroupId, string>> = {
	destinations: 'Go to',
	commands: 'Commands',
	settings: 'Settings',
	chats: 'Chats'
};

/** How many options one group shows for a search, and how many recent chats the empty palette lists. */
export const GROUP_LIMIT = 6;

/** What the ranking reads of an option. */
export interface Rankable {
	id: string;
	group: PaletteGroupId;
	label: string;
	/** Other words the option is found by: exact, prefix or word prefix. */
	keywords?: readonly string[];
	/** Found by its source for the words being searched: kept, after every label match. */
	found?: boolean;
	/** Not ranked: listed last in its group, past the limit, when the group lists anything for the words. */
	trailing?: boolean;
}

/** An option as the palette lists it. */
export interface PaletteOption extends Rankable {
	/** Where a destination, a setting or a chat goes. */
	href?: string;
	/** The command a command option runs. */
	command?: string;
	/** The Stop all entry: it opens the stop control's confirmation, never the stop. */
	stop?: boolean;
	/** A second, quieter line: where a setting lives, a command's keys. */
	detail?: string;
	/** The icon drawn beside the label: a name of the icon set (lib/ds/icons.ts). */
	icon?: string;
	/** A command that cannot run here. */
	disabled?: boolean;
	/** Why it cannot, in words; null when it can. */
	reason?: string | null;
}

/** A group of options, as listed. */
export interface RankedGroup<T extends Rankable> {
	id: PaletteGroupId;
	label: string;
	items: T[];
}

const EXACT = 0;
const PREFIX = 1;
const WORD = 2;
// A keyword's exact, prefix and word prefix matches rank after the label's.
const KEYWORD = 3;
const SUBSEQUENCE = 6;
const FOUND = 7;

const MARKS = /\p{M}+/gu;
const WORD_CHARACTER = /[\p{L}\p{N}]/u;

/** Words as the ranking compares them: lower case, no accents, single spaces, trimmed. */
export function fold(text: string): string {
	if (typeof text !== 'string') return '';
	return text.normalize('NFD').replace(MARKS, '').toLowerCase().replace(/\s+/g, ' ').trim();
}

function isWordCharacter(character: string): boolean {
	return WORD_CHARACTER.test(character);
}

/** Whether the letters of `query` (spaces aside) appear in `text` in the same order. */
function inOrder(text: string, query: string): boolean {
	const letters = query.replace(/ /g, '');
	if (!letters) return false;
	let at = 0;
	for (const character of text) {
		if (character === letters[at]) at += 1;
		if (at === letters.length) return true;
	}
	return false;
}

/** How `query` meets `text` (both folded), or null when it does not. */
function meet(text: string, query: string, loose: boolean): number | null {
	if (!text || !query) return null;
	if (text === query) return EXACT;
	if (text.startsWith(query)) return PREFIX;
	for (let at = 1; at < text.length; at += 1) {
		if (!isWordCharacter(text[at - 1]) && isWordCharacter(text[at]) && text.startsWith(query, at)) {
			return WORD;
		}
	}
	if (loose && inOrder(text, query)) return SUBSEQUENCE;
	return null;
}

/** The rank of an option for folded words, or null when it is not listed. */
function rankOf(item: Rankable, query: string): number | null {
	let best = meet(fold(item.label), query, true);
	for (const keyword of item.keywords ?? []) {
		const met = meet(fold(keyword), query, false);
		const rank = met === null ? null : KEYWORD + met;
		if (rank !== null && (best === null || rank < best)) best = rank;
	}
	if (best === null && item.found === true) return FOUND;
	return best;
}

function capOf(limit: number): number {
	return Number.isFinite(limit) ? Math.max(0, Math.floor(limit)) : GROUP_LIMIT;
}

function listed<T extends Rankable>(groups: { id: PaletteGroupId; items: T[] }[]): RankedGroup<T>[] {
	return groups
		.filter((group) => group.items.length > 0)
		.map((group) => ({ id: group.id, label: GROUP_LABELS[group.id], items: group.items }));
}

/** The palette with no words: every destination, then the recent chats. */
function emptyListing<T extends Rankable>(items: readonly T[], cap: number): RankedGroup<T>[] {
	return listed([
		{ id: 'destinations', items: items.filter((item) => item.group === 'destinations') },
		{
			id: 'chats',
			items: items.filter((item) => item.group === 'chats' && item.trailing !== true).slice(0, cap)
		}
	]);
}

/**
 * The groups to list for `query`: each group's options in their order of
 * rank (ties in the order given), at most `limit` of them, and the groups in
 * the order of their best option (ties in GROUP_ORDER).
 */
export function rankPalette<T extends Rankable>(
	items: readonly T[],
	query: string,
	limit: number = GROUP_LIMIT
): RankedGroup<T>[] {
	const words = fold(query);
	const cap = capOf(limit);
	if (!words) return emptyListing(items, cap);
	const byGroup = new Map<PaletteGroupId, { item: T; rank: number; at: number }[]>();
	items.forEach((item, at) => {
		if (!GROUP_ORDER.includes(item.group) || item.trailing === true) return;
		const rank = rankOf(item, words);
		if (rank === null) return;
		const list = byGroup.get(item.group) ?? [];
		list.push({ item, rank, at });
		byGroup.set(item.group, list);
	});
	const groups = [...byGroup.entries()].map(([id, list]) => {
		list.sort((a, b) => a.rank - b.rank || a.at - b.at);
		const closing = items.filter((item) => item.group === id && item.trailing === true);
		return { id, best: list[0].rank, items: [...list.slice(0, cap).map((entry) => entry.item), ...closing] };
	});
	groups.sort((a, b) => a.best - b.best || GROUP_ORDER.indexOf(a.id) - GROUP_ORDER.indexOf(b.id));
	return listed(groups);
}
