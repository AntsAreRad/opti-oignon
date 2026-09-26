/**
 * Preferences store -- the appearance choices and the status footer.
 *
 * The palette choice ("Match system", day, night or high contrast), the
 * density, the text size and the motion choice are stored here when the
 * user makes them, and nowhere else. What the root element carries is
 * decided and written by the theme path, lib/theme/apply.ts: every change
 * here resolves the stored choices again and applies the result, and the
 * Svelte stores below follow what was applied. Under "Match system" the
 * theme path also follows the system's colour scheme and contrast.
 *
 * A choice the browser cannot store (blocked storage) still applies for the
 * rest of the visit: it is kept in memory and read before storage.
 */

import { writable, get } from 'svelte/store';
import { darkMode, prefersReducedMotion } from './ui';
import {
	CHOICES,
	CHOICE_LABELS,
	DENSITIES,
	TEXT_SIZES,
	MOTIONS,
	PALETTE_KEY,
	DENSITY_KEY,
	TEXT_SIZE_KEY,
	MOTION_KEY,
	applyTo,
	browserEnvironment,
	browserStorage,
	followSystem,
	resolve,
	retireLegacyTheme,
	type Choice,
	type Density,
	type Motion,
	type Palette,
	type Resolved,
	type StorageReader,
	type TextSize
} from '$lib/theme/apply';

export { CHOICES, CHOICE_LABELS, DENSITIES };
export type { Choice, Density, Palette };
/** The text size: the root font size every rem follows. */
export type TypeScale = TextSize;
/** Motion preference: follow the system, force reduced, or force full motion. */
export type MotionPref = Motion;

/** Density labels. */
export const DENSITY_LABELS: Record<Density, string> = {
	compact: 'Compact',
	comfortable: 'Comfortable',
	spacious: 'Spacious'
};

/** All text sizes, in display order. */
export const TYPE_SCALES: TypeScale[] = TEXT_SIZES;

/** Text size labels. */
export const TYPE_SCALE_LABELS: Record<TypeScale, string> = {
	small: 'Small',
	default: 'Default',
	large: 'Large',
	'x-large': 'Extra large'
};

/** All motion preferences, in display order. */
export const MOTION_PREFS: MotionPref[] = MOTIONS;

/** Motion preference labels. */
export const MOTION_LABELS: Record<MotionPref, string> = {
	system: 'Match system',
	reduced: 'Reduce motion',
	full: 'Full motion'
};

const FOOTER_KEY = 'oo-status-footer';

/** Choices made in this visit, read before the browser's storage. */
const chosen = new Map<string, string>();

const stored: StorageReader = {
	getItem(key: string): string | null {
		const made = chosen.get(key);
		if (made !== undefined) return made;
		const storage = browserStorage();
		try {
			return storage ? storage.getItem(key) : null;
		} catch {
			return null;
		}
	}
};

const initial = resolve(stored, browserEnvironment());

/** The palette choice. */
export const palette = writable<Choice>(initial.choice);
/** The palette on screen: the choice, or what "Match system" resolved to. */
export const shownPalette = writable<Palette>(initial.theme);
/** Currently selected density. */
export const density = writable<Density>(initial.density);
/** Currently selected text size. */
export const typeScale = writable<TypeScale>(initial.size);
/** Currently selected motion preference. */
export const motionPref = writable<MotionPref>(initial.motion);
/** Whether the optional status footer is shown. */
export const statusFooterVisible = writable<boolean>(initialFooter());

function initialFooter(): boolean {
	const storage = browserStorage();
	try {
		const value = storage ? storage.getItem(FOOTER_KEY) : null;
		return value === null ? true : value === 'true';
	} catch {
		return true;
	}
}

/** The stores follow what the theme path applied. */
function follow(resolved: Resolved): void {
	palette.set(resolved.choice);
	shownPalette.set(resolved.theme);
	density.set(resolved.density);
	typeScale.set(resolved.size);
	motionPref.set(resolved.motion);
	darkMode.set(resolved.dark);
	if (resolved.motion === 'reduced') prefersReducedMotion.set(true);
	else if (resolved.motion === 'full') prefersReducedMotion.set(false);
	else if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
		prefersReducedMotion.set(window.matchMedia('(prefers-reduced-motion: reduce)').matches);
	}
}

/** Resolves every stored choice again and applies it, with a short colour
 * transition when asked and motion is allowed. */
function applyNow(animate = false): void {
	const resolved = resolve(stored, browserEnvironment());
	if (typeof document !== 'undefined') {
		const root = document.documentElement;
		const transition = animate && !get(prefersReducedMotion);
		if (transition) root.classList.add('theme-transitioning');
		applyTo(root, resolved);
		if (transition) setTimeout(() => root.classList.remove('theme-transitioning'), 350);
	}
	follow(resolved);
}

/** Choose a palette: kept, stored, and applied. */
export function setPalette(choice: Choice): void {
	chosen.set(PALETTE_KEY, choice);
	try {
		localStorage.setItem(PALETTE_KEY, choice);
	} catch {
		// Kept for this visit only.
	}
	applyNow(true);
}

/** Switch between the day and the night palettes: from day to night, and
 * from night or high contrast to day. */
export function toggleDayNight(): void {
	setPalette(get(shownPalette) === 'day' ? 'night' : 'day');
}

/** Choose a density: kept, stored, and applied. */
export function setDensity(choice: Density): void {
	chosen.set(DENSITY_KEY, choice);
	try {
		localStorage.setItem(DENSITY_KEY, choice);
	} catch {
		// Kept for this visit only.
	}
	applyNow();
}

/** Choose a text size: kept, stored, and applied. */
export function setTypeScale(choice: TypeScale): void {
	chosen.set(TEXT_SIZE_KEY, choice);
	try {
		localStorage.setItem(TEXT_SIZE_KEY, choice);
	} catch {
		// Kept for this visit only.
	}
	applyNow();
}

/** Choose a motion preference: kept, stored, and applied. */
export function setMotionPref(choice: MotionPref): void {
	chosen.set(MOTION_KEY, choice);
	try {
		localStorage.setItem(MOTION_KEY, choice);
	} catch {
		// Kept for this visit only.
	}
	applyNow();
}

/** Show or hide the optional status footer, and store the choice. */
export function setStatusFooterVisible(visible: boolean): void {
	statusFooterVisible.set(visible);
	try {
		localStorage.setItem(FOOTER_KEY, String(visible));
	} catch {
		// Kept for this visit only.
	}
}

/**
 * Applies the stored choices at startup, without the transition, and
 * follows the system from then on. Returns the function that stops
 * following. Call once from the root layout.
 *
 * First the older binary theme is retired where it decides nothing; where
 * it is still a pin (unlike the system), the choice it means is held for
 * the visit, so a change of the system does not move the choice shown, and
 * choosing Match system stores it.
 */
export function initPreferences(): () => void {
	const pinned = retireLegacyTheme(browserStorage(), browserEnvironment());
	if (pinned !== null && !chosen.has(PALETTE_KEY)) chosen.set(PALETTE_KEY, pinned);
	applyNow();
	if (typeof document === 'undefined') return () => {};
	return followSystem(stored, browserEnvironment(), document.documentElement, follow);
}
