/**
 * The theme path: the one place that decides what the root element carries.
 *
 * resolve(stored, env) reads what the browser stored (the palette choice,
 * the retired palette names and the older binary theme it migrates, the
 * density, the text size, the motion choice, the componion switch) and what
 * the system asks for (its colour scheme and its contrast), and returns the
 * state the root must carry. applyTo(root, resolved) writes that state on
 * the root, and only there: the palette attribute, the dark class, one
 * density class, the root font size, the motion classes and the componion
 * attribute. Neither function writes storage; a choice is stored by the
 * preferences store, and only when the user makes it.
 *
 * retireLegacyTheme(stored, env), run once at startup, removes the older
 * binary theme when it no longer decides anything: a palette choice is
 * stored, or it is what the system shows anyway (it then means Match
 * system, and would otherwise pin the palette the first time the system
 * changes). Unlike the system, it is a pin the user made: it stays, and the
 * choice it means is returned so the caller can hold it for the visit. It
 * removes a key and stores nothing.
 *
 * The inline pre-render in app.html makes the same decisions before the
 * first paint, from the same keys; a contract runs both over every
 * combination of stored values, system settings, storage failure and text
 * sizes and requires the same root. A change here is a change there.
 *
 * Storage may be absent or may throw (a private window, blocked site
 * data); a store that throws reads as nothing stored, so the defaults
 * apply: "Match system", comfortable density, the default text size.
 *
 * "Match system" shows the night palette unless the system asks for a
 * light scheme, and the high contrast palette whenever the system asks for
 * more contrast, whatever its scheme.
 *
 * This module imports nothing, so it runs as it is under Node.
 */

/** What the user can choose. */
export type Choice = 'system' | 'day' | 'night' | 'high-contrast';
/** What the root can show. */
export type Palette = 'day' | 'night' | 'high-contrast';
export type Density = 'compact' | 'comfortable' | 'spacious';
export type TextSize = 'small' | 'default' | 'large' | 'x-large';
export type Motion = 'system' | 'reduced' | 'full';

/** The part of Storage the theme path reads. */
export interface StorageReader {
	getItem(key: string): string | null;
}

/** The part of Storage the startup retirement writes: it removes, never stores. */
export interface StorageRemover extends StorageReader {
	removeItem(key: string): void;
}

/** The part of MediaQueryList the theme path uses. */
export interface MediaList {
	readonly matches: boolean;
	addEventListener?(type: 'change', listener: () => void): void;
	removeEventListener?(type: 'change', listener: () => void): void;
	addListener?(listener: () => void): void;
	removeListener?(listener: () => void): void;
}

/** What the theme path asks of the system; no matchMedia is no preference. */
export interface ThemeEnvironment {
	matchMedia?: (query: string) => MediaList;
}

/** The part of an element the theme path writes. */
export interface RootElement {
	setAttribute(name: string, value: string): void;
	classList: {
		add(...names: string[]): void;
		remove(...names: string[]): void;
		toggle(name: string, force?: boolean): boolean;
	};
	style: { setProperty(name: string, value: string): void };
	ownerDocument?: unknown;
}

/** The state the root carries. */
export interface Resolved {
	/** What was chosen, the retired names migrated. */
	choice: Choice;
	/** The palette shown. */
	theme: Palette;
	/** Whether the palette shown is a dark one. */
	dark: boolean;
	density: Density;
	size: TextSize;
	/** The root font size of the text size, as a percentage. */
	rootSize: string;
	motion: Motion;
	/** Whether the componion is shown; on unless it was turned off. */
	componion: boolean;
}

export const PALETTE_KEY = 'oo-palette';
/** The older binary theme ('light' or 'dark'), read to migrate, never written. */
export const LEGACY_THEME_KEY = 'oo-theme';
export const DENSITY_KEY = 'oo-density';
export const TEXT_SIZE_KEY = 'oo-type-scale';
export const MOTION_KEY = 'oo-motion';
export const COMPONION_KEY = 'oo-componion';

/** The choices, in display order. */
export const CHOICES: Choice[] = ['system', 'day', 'night', 'high-contrast'];

export const CHOICE_LABELS: Record<Choice, string> = {
	system: 'Match system',
	day: 'Day',
	night: 'Night',
	'high-contrast': 'High contrast'
};

/** The palettes a root can show. */
export const PALETTES: Palette[] = ['day', 'night', 'high-contrast'];

/** Palettes of a dark colour scheme. */
const DARK: Record<Palette, boolean> = { day: false, night: true, 'high-contrast': true };

/** Palette names of earlier versions, and the palette each one now means. */
const RETIRED: Record<string, Palette> = {
	anthracite: 'night',
	slate: 'night',
	parchment: 'day',
	linen: 'day'
};

export const DENSITIES: Density[] = ['compact', 'comfortable', 'spacious'];
export const TEXT_SIZES: TextSize[] = ['small', 'default', 'large', 'x-large'];

/** The root font size of each text size: every rem follows it. */
export const ROOT_SIZE: Record<TextSize, string> = {
	small: '92%',
	default: '100%',
	large: '109%',
	'x-large': '118%'
};

export const MOTIONS: Motion[] = ['system', 'reduced', 'full'];

/** The classes the motion choice sets; lib/motion.ts reads the same two. */
const REDUCE_MOTION_CLASS = 'oo-reduce-motion';
const FULL_MOTION_CLASS = 'oo-motion-full';

const SCHEME_LIGHT = '(prefers-color-scheme: light)';
const CONTRAST_MORE = '(prefers-contrast: more)';

function has<T extends string>(allowed: readonly T[], value: string | null): value is T {
	return value !== null && (allowed as readonly string[]).includes(value);
}

function read(stored: StorageReader | null | undefined, key: string): string | null {
	try {
		return stored ? stored.getItem(key) : null;
	} catch {
		return null;
	}
}

function asks(env: ThemeEnvironment | null | undefined, query: string): boolean {
	try {
		return !!(env && typeof env.matchMedia === 'function' && env.matchMedia(query).matches);
	} catch {
		return false;
	}
}

/** What the user chose: a choice, a retired palette's successor, the older
 * binary theme (Match system when it is what the system shows anyway),
 * or Match system when nothing readable was stored. */
function choiceOf(stored: StorageReader | null | undefined, systemLight: boolean): Choice {
	const palette = read(stored, PALETTE_KEY);
	if (has(CHOICES, palette)) return palette;
	if (palette !== null && Object.prototype.hasOwnProperty.call(RETIRED, palette)) {
		return RETIRED[palette];
	}
	const theme = read(stored, LEGACY_THEME_KEY);
	if (theme === 'light' || theme === 'dark') {
		if (theme === (systemLight ? 'light' : 'dark')) return 'system';
		return theme === 'light' ? 'day' : 'night';
	}
	return 'system';
}

/**
 * Retires the older binary theme once it decides nothing: under a stored
 * palette choice, or when it equals what the system shows. Returns the
 * choice an older theme unlike the system still means (day or night), or
 * null. Storage that throws is left as it is.
 */
export function retireLegacyTheme(
	stored: StorageRemover | null | undefined,
	env: ThemeEnvironment | null | undefined
): Choice | null {
	const theme = read(stored, LEGACY_THEME_KEY);
	if (theme !== 'light' && theme !== 'dark') return null;
	const palette = read(stored, PALETTE_KEY);
	const chosen =
		has(CHOICES, palette) || (palette !== null && Object.prototype.hasOwnProperty.call(RETIRED, palette));
	if (!chosen && theme !== (asks(env, SCHEME_LIGHT) ? 'light' : 'dark')) {
		return theme === 'light' ? 'day' : 'night';
	}
	try {
		if (stored) stored.removeItem(LEGACY_THEME_KEY);
	} catch {
		// Storage that throws keeps the key; resolve() reads it as before.
	}
	return null;
}

/** The state the root must carry, from what was stored and what the system asks. */
export function resolve(
	stored: StorageReader | null | undefined,
	env: ThemeEnvironment | null | undefined
): Resolved {
	const systemLight = asks(env, SCHEME_LIGHT);
	const choice = choiceOf(stored, systemLight);
	let theme: Palette;
	if (choice !== 'system') theme = choice;
	else if (asks(env, CONTRAST_MORE)) theme = 'high-contrast';
	else theme = systemLight ? 'day' : 'night';

	const density = read(stored, DENSITY_KEY);
	const size = read(stored, TEXT_SIZE_KEY);
	const motion = read(stored, MOTION_KEY);
	const textSize: TextSize = has(TEXT_SIZES, size) ? size : 'default';
	return {
		choice,
		theme,
		dark: DARK[theme],
		density: has(DENSITIES, density) ? density : 'comfortable',
		size: textSize,
		rootSize: ROOT_SIZE[textSize],
		motion: has(MOTIONS, motion) ? motion : 'system',
		componion: read(stored, COMPONION_KEY) !== 'off'
	};
}

/** Writes the resolved state on the root, replacing whatever an earlier
 * application set there, and nothing else. */
export function applyTo(root: RootElement, resolved: Resolved): void {
	root.setAttribute('data-oo-theme', resolved.theme);
	root.classList.toggle('dark', resolved.dark);
	for (const density of DENSITIES) root.classList.remove(`oo-density-${density}`);
	root.classList.add(`oo-density-${resolved.density}`);
	root.style.setProperty('font-size', resolved.rootSize);
	root.classList.toggle(REDUCE_MOTION_CLASS, resolved.motion === 'reduced');
	root.classList.toggle(FULL_MOTION_CLASS, resolved.motion === 'full');
	root.setAttribute('data-oo-componion', resolved.componion ? 'on' : 'off');
	themeColour(root);
}

/** The browser's theme-color follows the ground of the palette shown, read
 * from the palette file itself; where no style can be computed (before the
 * style sheets load, or outside a browser) it is left as it is. */
function themeColour(root: RootElement): void {
	const doc = root.ownerDocument as
		| {
				defaultView?: { getComputedStyle?: (el: unknown) => { getPropertyValue(name: string): string } } | null;
				querySelector?: (selector: string) => { setAttribute(name: string, value: string): void } | null;
		  }
		| null
		| undefined;
	const view = doc && doc.defaultView;
	if (!view || typeof view.getComputedStyle !== 'function' || typeof doc.querySelector !== 'function') return;
	const ground = view.getComputedStyle(root).getPropertyValue('--oo-role-bg').trim();
	const meta = doc.querySelector('meta[name="theme-color"]');
	if (ground && meta) meta.setAttribute('content', ground);
}

/**
 * Follows the system: whenever its colour scheme or its contrast changes,
 * the state is resolved again from what is stored now and applied, so a
 * choice made while following is read at the next change and Match system
 * moves with the system. Returns the function that stops following. With
 * no media queries there is nothing to follow.
 */
export function followSystem(
	stored: StorageReader | null | undefined,
	env: ThemeEnvironment | null | undefined,
	root: RootElement,
	onApply?: (resolved: Resolved) => void
): () => void {
	const matchMedia = env && env.matchMedia;
	if (typeof matchMedia !== 'function') return () => {};
	const lists: MediaList[] = [];
	for (const query of [SCHEME_LIGHT, CONTRAST_MORE]) {
		try {
			lists.push(matchMedia(query));
		} catch {
			// A query the browser cannot answer is not followed.
		}
	}
	const changed = () => {
		const resolved = resolve(stored, env);
		applyTo(root, resolved);
		if (onApply) onApply(resolved);
	};
	for (const list of lists) {
		if (typeof list.addEventListener === 'function') list.addEventListener('change', changed);
		else if (typeof list.addListener === 'function') list.addListener(changed);
	}
	return () => {
		for (const list of lists) {
			if (typeof list.removeEventListener === 'function') list.removeEventListener('change', changed);
			else if (typeof list.removeListener === 'function') list.removeListener(changed);
		}
	};
}

/** The browser's storage, or null where reaching it throws or there is none. */
export function browserStorage(): StorageRemover | null {
	try {
		return typeof localStorage === 'undefined' ? null : localStorage;
	} catch {
		return null;
	}
}

/** The browser's media queries, or none outside a browser. */
export function browserEnvironment(): ThemeEnvironment {
	if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return {};
	return { matchMedia: (query: string) => window.matchMedia(query) };
}
