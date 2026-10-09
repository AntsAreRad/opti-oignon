/**
 * The ranges of the composer's text the user pasted or dropped rather than typed.
 *
 * Pasted text is a document: whatever its author wrote, orders included, it
 * is not the user's own words, and the server saves it as a document part of
 * the turn. The composer keeps, for each character, whether it was typed or
 * pasted, through every edit, and sends the pasted ranges with the text.
 *
 * Ranges are [start, end) in UTF-16 code units of the composer's text, as the
 * textarea counts, kept sorted, apart and non-empty. An edit is placed by the
 * selection it was made in when the browser reports one that fits the text,
 * else by the text's own difference; the characters it inserts are pasted
 * unless its input type is one this module knows as typing, so an input it
 * does not know -- a paste, a drop, an undo, a type to come -- is held pasted,
 * the safe side. What is sent is the text trimmed as `String.prototype.trim`
 * trims it, with its ranges in code points, the server's measure, a range
 * that ends inside a character widened to the whole of it.
 *
 * Pure and dependency-free: the composer calls it, and it runs under Node as
 * it is.
 */

/** [start, end), in UTF-16 code units of the composer's text. */
export type Range = [number, number];

/** Where an edit took place: what it removed and what it inserted there. */
export interface Edit {
	start: number;
	removed: number;
	inserted: number;
}

/**
 * The input types that are the user typing (`insertFromComposition` is the
 * name an input method's text commits under in the first level of Input
 * Events); any other input that inserts text is held pasted.
 */
export const TYPED_INPUTS: ReadonlySet<string> = new Set([
	'insertText',
	'insertLineBreak',
	'insertParagraph',
	'insertCompositionText',
	'insertFromComposition',
	'insertReplacementText',
]);

/** Whether an input type is the user typing. */
export function isTypedInput(inputType: string | null | undefined): boolean {
	return typeof inputType === 'string' && TYPED_INPUTS.has(inputType);
}

/** Ranges held to a text of ``length`` units: clipped, sorted, empty ones dropped, touching ones merged. */
export function normalize(ranges: readonly (readonly number[])[], length: number): Range[] {
	const held: Range[] = [];
	for (const range of ranges) {
		const start = Math.max(0, Math.min(length, Math.floor(range[0])));
		const end = Math.max(0, Math.min(length, Math.floor(range[1])));
		if (start < end) held.push([start, end]);
	}
	held.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
	const merged: Range[] = [];
	for (const [start, end] of held) {
		const last = merged[merged.length - 1];
		if (last && start <= last[1]) last[1] = Math.max(last[1], end);
		else merged.push([start, end]);
	}
	return merged;
}

/**
 * The edit that turned ``before`` into ``after``: placed by ``hint``, the
 * selection the edit was made in, when the text bears it out; else by the
 * longest common start and end the two texts share.
 */
export function editBetween(before: string, after: string, hint?: { start: number; end: number }): Edit {
	if (hint) {
		const { start, end } = hint;
		const insertedEnd = after.length - (before.length - end);
		if (
			start >= 0 &&
			start <= end &&
			end <= before.length &&
			insertedEnd >= start &&
			before.slice(0, start) === after.slice(0, start) &&
			before.slice(end) === after.slice(insertedEnd)
		) {
			return { start, removed: end - start, inserted: insertedEnd - start };
		}
	}
	const shortest = Math.min(before.length, after.length);
	let prefix = 0;
	while (prefix < shortest && before[prefix] === after[prefix]) prefix++;
	let suffix = 0;
	while (suffix < shortest - prefix && before[before.length - 1 - suffix] === after[after.length - 1 - suffix]) {
		suffix++;
	}
	return { start: prefix, removed: before.length - prefix - suffix, inserted: after.length - prefix - suffix };
}

/**
 * The ranges once ``edit`` is made: what it removed leaves them, what follows
 * moves with the text, and what it inserted is a range of its own when it
 * was pasted. ``length`` is the text's length after the edit.
 */
export function applyEdit(ranges: readonly Range[], edit: Edit & { pasted: boolean }, length: number): Range[] {
	const { start, removed, inserted, pasted } = edit;
	const cut = start + removed;
	const shift = inserted - removed;
	const held: Range[] = [];
	for (const [a, b] of ranges) {
		if (a < start) held.push([a, Math.min(b, start)]);
		if (b > cut) held.push([Math.max(a, cut) + shift, b + shift]);
	}
	if (pasted && inserted > 0) held.push([start, start + inserted]);
	return normalize(held, length);
}

/**
 * What the composer sends: its text trimmed as `String.prototype.trim` trims
 * it, and the pasted ranges of the trimmed text in code points.
 */
export function sendable(text: string, ranges: readonly Range[]): { text: string; pasted: Range[] } {
	const trimmed = text.trim();
	const lead = trimmed ? text.length - text.trimStart().length : 0;
	const end = lead + trimmed.length;
	const units = normalize(
		ranges.map(([a, b]) => [Math.max(a, lead) - lead, Math.min(b, end) - lead]),
		trimmed.length
	);
	// Code points before each unit: a start inside a character's two units
	// falls back to the character, an end inside it reaches past it.
	const floor: number[] = [];
	const ceil: number[] = [];
	let count = 0;
	for (let at = 0; at < trimmed.length; at++) {
		floor[at] = count;
		ceil[at] = count;
		const code = trimmed.charCodeAt(at);
		if (code >= 0xd800 && code <= 0xdbff && at + 1 < trimmed.length) {
			const next = trimmed.charCodeAt(at + 1);
			if (next >= 0xdc00 && next <= 0xdfff) {
				floor[at + 1] = count;
				ceil[at + 1] = count + 1;
				at++;
			}
		}
		count++;
	}
	floor[trimmed.length] = count;
	ceil[trimmed.length] = count;
	return { text: trimmed, pasted: normalize(units.map(([a, b]) => [floor[a], ceil[b]]), count) };
}

/**
 * The blanks the server strips a message of (Python's `str.isspace`): those
 * of `String.prototype.trim` but U+FEFF, and U+001C-U+001F and U+0085
 * besides.
 */
const SERVER_BLANKS: ReadonlySet<number> = new Set([
	0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003,
	0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000
]);

/**
 * Whether the text starts with a command the user typed, as the server reads
 * what the composer sends: the text trimmed as the composer trims it, then
 * stripped of the server's blanks, is the command alone or followed by a
 * space, and no pasted range covers any of its characters. A command pasted
 * is the pasted text's, and starts nothing.
 */
export function typedCommand(text: string, ranges: readonly Range[], command: string): boolean {
	const sent = text.trim();
	let start = 0;
	let end = sent.length;
	while (start < end && SERVER_BLANKS.has(sent.charCodeAt(start))) start++;
	while (end > start && SERVER_BLANKS.has(sent.charCodeAt(end - 1))) end--;
	const stripped = sent.slice(start, end);
	if (stripped !== command && !stripped.startsWith(command + ' ')) return false;
	// Every blank before the command is one code unit: units and the server's code points agree up to it.
	const from = text.length - text.trimStart().length + start;
	const to = from + command.length;
	return !ranges.some(([a, b]) => a < to && b > from);
}
