/**
 * icons.ts (lib/ds) -- the interface's icons, drawn once, as path data.
 *
 * Every icon is drawn on a 24 unit square and stroked in the text colour,
 * with round caps and joins, at the width Icon.svelte gives it (1.5 unless a
 * call asks otherwise). A shape is a path's data, or a path filled with the
 * text colour. The data is compact: no space stands before or after a
 * command letter, so a command never reads as a separate word.
 *
 * Names are kebab-case. Where the older icon package names the same picture,
 * the name is that package's, so a call written for it draws this set's
 * drawing; Icon.svelte falls back to the package only for a name not drawn
 * here yet.
 *
 * Dependency-free: the contracts run it under Node.
 */

/** One shape of an icon: stroked path data, or a path filled with the text colour. */
export type IconShape = string | { readonly d: string; readonly filled: true };

export const ICONS: Readonly<Record<string, readonly IconShape[]>> = Object.freeze({
	// The design's thirty drawings.
	onion: [
		'M12 6.5c-3 2.4-7 4.3-7 8.6A6.9 6.9 0 0 0 12 22a6.9 6.9 0 0 0 7-6.9c0-4.3-4-6.2-7-8.6Z',
		'M12 6.5c-1.4 2.6-2.6 5.3-2.6 8.6 0 3.2 1.1 5.5 2.6 6.9',
		'M12 6.5V2.5',
		'M12 4.5c.9-1.3 2.2-1.9 3.6-1.8',
	],
	'panel-left': [
		'M7 4h10a4 4 0 0 1 4 4v8a4 4 0 0 1-4 4h-10a4 4 0 0 1-4-4v-8a4 4 0 0 1 4-4Z',
		'M9.5 4v16',
	],
	'panel-right': [
		'M7 4h10a4 4 0 0 1 4 4v8a4 4 0 0 1-4 4h-10a4 4 0 0 1-4-4v-8a4 4 0 0 1 4-4Z',
		'M14.5 4v16',
	],
	plus: ['M12 5v14', 'M5 12h14'],
	search: ['M4.5 11a6.5 6.5 0 1 0 13 0a6.5 6.5 0 1 0-13 0Z', 'm20 20-4.2-4.2'],
	home: ['M4 10.5 12 4l8 6.5', 'M6 9.5V19a1 1 0 0 0 1 1h3.5v-5.5h3V20H17a1 1 0 0 0 1-1V9.5'],
	chat: [
		'M20 11.5c0 4.1-3.6 7.5-8 7.5-1.2 0-2.3-.2-3.3-.6L4 20l1.3-3.8C4.5 14.9 4 13.3 4 11.5 4 7.4 7.6 4 12 4s8 3.4 8 7.5Z',
	],
	note: [
		'M7 3.5h7.5L19 8v11.5a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1v-15a1 1 0 0 1 1-1Z',
		'M14 3.5v5h5',
		'M9 13h6',
		'M9 16.5h4',
	],
	folder: [
		'M3.5 7.5a2 2 0 0 1 2-2H9l2 2.5h7.5a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2Z',
	],
	sprout: [
		'M12 20.5V11',
		'M12 12.5c0-4 2.4-6.5 7-6.5 0 4.2-2.6 6.5-7 6.5Z',
		'M12 10.5C12 7.2 10 5.5 5.5 5.5c0 3.3 2.2 5 6.5 5Z',
	],
	sliders: [
		'M4 7h9',
		'M17 7h3',
		'M4 17h3',
		'M11 17h9',
		'M13 7a2 2 0 1 0 4 0a2 2 0 1 0-4 0Z',
		'M7 17a2 2 0 1 0 4 0a2 2 0 1 0-4 0Z',
	],
	stop: [
		'M8.3 3h7.4L21 8.3v7.4L15.7 21H8.3L3 15.7V8.3Z',
		'M10.25 9.25h3.5a1 1 0 0 1 1 1v3.5a1 1 0 0 1-1 1h-3.5a1 1 0 0 1-1-1v-3.5a1 1 0 0 1 1-1Z',
	],
	'chevron-down': ['m6.5 9.5 5.5 5.5 5.5-5.5'],
	'chevron-right': ['m9.5 6.5 5.5 5.5-5.5 5.5'],
	'chevron-left': ['m14.5 6.5-5.5 5.5 5.5 5.5'],
	'arrow-up': ['M12 19V5.5', 'm6 11.5 6-6 6 6'],
	'arrow-right': ['M5 12h13.5', 'm12.5 6 6 6-6 6'],
	check: ['m5 12.5 4.5 4.5L19 7.5'],
	bulb: [
		'M9.5 18h5',
		'M10.5 21h3',
		'M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2V16h5v-.1c0-.8.4-1.5 1-2A6 6 0 0 0 12 3Z',
	],
	globe: [
		'M3.5 12a8.5 8.5 0 1 0 17 0a8.5 8.5 0 1 0-17 0Z',
		'M3.5 12h17',
		'M12 3.5c2.3 2.4 3.5 5.2 3.5 8.5s-1.2 6.1-3.5 8.5c-2.3-2.4-3.5-5.2-3.5-8.5s1.2-6.1 3.5-8.5Z',
	],
	box: ['M20.5 8 12 3.5 3.5 8v8l8.5 4.5 8.5-4.5Z', 'm3.5 8 8.5 4.5L20.5 8', 'M12 12.5v8'],
	code: ['m15.5 17.5 5.5-5.5-5.5-5.5', 'M8.5 6.5 3 12l5.5 5.5'],
	copy: [
		'M11.5 8.5h6a3 3 0 0 1 3 3v6a3 3 0 0 1-3 3h-6a3 3 0 0 1-3-3v-6a3 3 0 0 1 3-3Z',
		'M15.5 8.5v-2a3 3 0 0 0-3-3h-6a3 3 0 0 0-3 3v6a3 3 0 0 0 3 3h2',
	],
	retry: ['M4 12a8 8 0 1 0 2.6-5.9L4 8.5', 'M4 4v4.5h4.5'],
	branch: [
		'M6 3.5v11',
		'M15.5 6a2.5 2.5 0 1 0 5 0a2.5 2.5 0 1 0-5 0Z',
		'M3.5 18a2.5 2.5 0 1 0 5 0a2.5 2.5 0 1 0-5 0Z',
		'M18 8.5a9 9 0 0 1-9 9',
	],
	chip: [
		'M9 6.5h6a2.5 2.5 0 0 1 2.5 2.5v6a2.5 2.5 0 0 1-2.5 2.5h-6a2.5 2.5 0 0 1-2.5-2.5v-6a2.5 2.5 0 0 1 2.5-2.5Z',
		'M10 3.5v3',
		'M14 3.5v3',
		'M10 17.5v3',
		'M14 17.5v3',
		'M3.5 10h3',
		'M3.5 14h3',
		'M17.5 10h3',
		'M17.5 14h3',
	],
	pencil: ['M16.5 4.5a2.1 2.1 0 0 1 3 3L8 19l-4 1 1-4Z'],
	pin: ['M12 16.5V21', 'M8.5 3.5h7l-1 5.5 3 3v2h-11v-2l3-3Z'],
	'stop-fill': [
		{ d: 'M9 6.5h6a2.5 2.5 0 0 1 2.5 2.5v6a2.5 2.5 0 0 1-2.5 2.5h-6a2.5 2.5 0 0 1-2.5-2.5v-6a2.5 2.5 0 0 1 2.5-2.5Z', filled: true },
	],
	more: [
		'M4.5 12a1 1 0 1 0 2 0a1 1 0 1 0-2 0Z',
		'M11 12a1 1 0 1 0 2 0a1 1 0 1 0-2 0Z',
		'M17.5 12a1 1 0 1 0 2 0a1 1 0 1 0-2 0Z',
	],
	// Drawn in the same hand for the surfaces that need them: closing, a
	// warning, an error and a note, a straight line (the drawing tool's),
	// a reply's feedback, export, wipe and attach.
	x: ['M7 7l10 10', 'M17 7 7 17'],
	'chevron-up': ['m6.5 14.5 5.5-5.5 5.5 5.5'],
	'alert-triangle': [
		'M10.3 4.9a2 2 0 0 1 3.4 0l6.9 11.9a2 2 0 0 1-1.7 3H5.1a2 2 0 0 1-1.7-3Z',
		'M12 9.5v4',
		'M12 16.8h.01',
	],
	'alert-octagon': ['M8.3 3h7.4L21 8.3v7.4L15.7 21H8.3L3 15.7V8.3Z', 'M12 7.5v5.5', 'M12 16.5h.01'],
	info: ['M3.5 12a8.5 8.5 0 1 0 17 0a8.5 8.5 0 1 0-17 0Z', 'M12 11v5', 'M12 8h.01'],
	minus: ['M5 12h14'],
	'thumbs-up': [
		'M7 10.5 10.6 4.2a1.6 1.6 0 0 1 2.9 1.2L12.8 9H18a2 2 0 0 1 2 2.3l-1.1 6.5a2 2 0 0 1-2 1.7H7Z',
		'M7 10.5H4.5a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1H7',
	],
	'thumbs-down': [
		'M7 13.5 10.6 19.8a1.6 1.6 0 0 0 2.9-1.2L12.8 15H18a2 2 0 0 0 2-2.3l-1.1-6.5a2 2 0 0 0-2-1.7H7Z',
		'M7 13.5H4.5a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1H7',
	],
	trash: [
		'M4.5 7h15',
		'M9.5 7V5.2a1.2 1.2 0 0 1 1.2-1.2h2.6a1.2 1.2 0 0 1 1.2 1.2V7',
		'M6.5 7l.9 12a1.6 1.6 0 0 0 1.6 1.5h6a1.6 1.6 0 0 0 1.6-1.5l.9-12',
		'M10.2 11v5.5',
		'M13.8 11v5.5',
	],
	download: [
		'M12 4v11',
		'm7.5 10.5 4.5 4.5 4.5-4.5',
		'M4.5 15.5V18a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-2.5',
	],
	paperclip: [
		'M19.5 11.5l-7.1 7.1a4.6 4.6 0 0 1-6.5-6.5l7.4-7.4a3.1 3.1 0 0 1 4.4 4.4l-7.2 7.2a1.6 1.6 0 0 1-2.3-2.3l6.6-6.6',
	],
	// The Workshop's pages that had no drawing: the benchmarks, the
	// observability of the pipeline, the security page.
	'bar-chart-3': ['M4.5 20h15', 'M7.5 16.5v-4', 'M12 16.5v-9', 'M16.5 16.5v-6'],
	activity: ['M3 12h4l2.5-6 5 12 2.5-6H21'],
	'shield-check': ['M12 3.5 19 6v5.5c0 4.4-3 7.6-7 9-4-1.4-7-4.6-7-9V6Z', 'm9 12 2.2 2.2L15.5 10'],
});

/** A name as the set spells it: kebab-case, from PascalCase, camelCase or snake_case. */
export function iconKey(name: string): string {
	return name
		.replace(/([a-z0-9])([A-Z])/g, '$1-$2')
		.replace(/[_\s]+/g, '-')
		.toLowerCase();
}

/** The shapes of an icon the set draws, or undefined; never an inherited property. */
export function inlineIcon(name: string): readonly IconShape[] | undefined {
	const key = iconKey(name);
	return Object.prototype.hasOwnProperty.call(ICONS, key) ? ICONS[key] : undefined;
}
