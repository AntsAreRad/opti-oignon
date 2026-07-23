/**
 * Preferences store -- palette + density + status footer.
 *
 * Source of truth for the 5 curated palettes (spec 7.3) and the 3
 * density modes (spec 7.2). Applies `data-oo-theme` and a single
 * `html.oo-density-*` class to the document root, and keeps the legacy
 * `.dark` class and the `darkMode` store in sync so every pre
 * consumer stays correct. Persisted to localStorage.
 *
 * The 5 theme files raise their `:root[data-oo-theme="..."]` selectors to
 * a higher specificity than the inline default-mode blocks in theme.css,
 * so setting `data-oo-theme` is authoritative for all five palettes.
 */

import { writable, get } from 'svelte/store';
import { darkMode, prefersReducedMotion } from './ui';

export type ThemePalette = 'anthracite' | 'parchment' | 'slate' | 'linen' | 'high-contrast';
export type Density = 'compact' | 'comfortable' | 'spacious';
/** Global typography scale (multiplier on every --oo-text-* token). */
export type TypeScale = 'small' | 'default' | 'large' | 'x-large';
/** Motion preference: follow the OS, force reduced, or force full motion. */
export type MotionPref = 'system' | 'reduced' | 'full';

/** All selectable palettes, in display order. */
export const PALETTES: ThemePalette[] = [
	'anthracite',
	'parchment',
	'slate',
	'linen',
	'high-contrast'
];

/** Human-readable labels for the palettes. */
export const PALETTE_LABELS: Record<ThemePalette, string> = {
	anthracite: 'Anthracite',
	parchment: 'Parchment',
	slate: 'Slate',
	linen: 'Linen',
	'high-contrast': 'High Contrast'
};

/** All density modes, in display order. */
export const DENSITIES: Density[] = ['compact', 'comfortable', 'spacious'];

/** Density labels. */
export const DENSITY_LABELS: Record<Density, string> = {
	compact: 'Compact',
	comfortable: 'Comfortable',
	spacious: 'Spacious'
};

/** All typography scales, in display order. */
export const TYPE_SCALES: TypeScale[] = ['small', 'default', 'large', 'x-large'];

/** Typography scale labels. */
export const TYPE_SCALE_LABELS: Record<TypeScale, string> = {
	small: 'Small',
	default: 'Default',
	large: 'Large',
	'x-large': 'Extra large'
};

/** Multiplier applied to every --oo-text-* token via --oo-type-scale. */
export const TYPE_SCALE_VALUE: Record<TypeScale, number> = {
	small: 0.92,
	default: 1,
	large: 1.09,
	'x-large': 1.18
};

/** All motion preferences, in display order. */
export const MOTION_PREFS: MotionPref[] = ['system', 'reduced', 'full'];

/** Motion preference labels. */
export const MOTION_LABELS: Record<MotionPref, string> = {
	system: 'Match system',
	reduced: 'Reduce motion',
	full: 'Full motion'
};

/**
 * Preview colors per palette, mirrored from the theme files. Kept here (a
 * TS module, not a .svelte file) so the ThemeSwitcher chips can show each
 * palette's identity without the current theme's tokens overriding them,
 * and without raw hex inside a component.
 */
export const PALETTE_SWATCH: Record<ThemePalette, { base: string; surface: string; fg: string }> = {
	anthracite: { base: '#1F1F22', surface: '#27272A', fg: '#ECE9E3' },
	parchment: { base: '#E5DECE', surface: '#DDD5C3', fg: '#2D2C2A' },
	slate: { base: '#1A1E24', surface: '#22272F', fg: '#E4E8EC' },
	linen: { base: '#EAEBE7', surface: '#E3E4E0', fg: '#22272D' },
	'high-contrast': { base: '#000000', surface: '#0A0A0A', fg: '#FFFFFF' }
};

/** Palettes that should carry the `.dark` class (logo filter + legacy dark styles). */
const DARK_PALETTES: ReadonlySet<ThemePalette> = new Set<ThemePalette>([
	'anthracite',
	'slate',
	'high-contrast'
]);

const PALETTE_KEY = 'oo-palette';
const DENSITY_KEY = 'oo-density';
const FOOTER_KEY = 'oo-status-footer';
const TYPE_SCALE_KEY = 'oo-type-scale';
const MOTION_KEY = 'oo-motion';
/** Legacy binary-theme key, kept coherent for any pre-reader. */
const LEGACY_THEME_KEY = 'oo-theme';

function readStored<T extends string>(key: string, allowed: readonly T[]): T | null {
	if (typeof localStorage === 'undefined') return null;
	const v = localStorage.getItem(key) as T | null;
	return v && allowed.includes(v) ? v : null;
}

/** Whether a palette is a dark palette. */
export function isDarkPalette(p: ThemePalette): boolean {
	return DARK_PALETTES.has(p);
}

function initialPalette(): ThemePalette {
	const stored = readStored<ThemePalette>(PALETTE_KEY, PALETTES);
	if (stored) return stored;
	// Continuity with the pre-binary preference.
	if (typeof localStorage !== 'undefined') {
		const legacy = localStorage.getItem(LEGACY_THEME_KEY);
		if (legacy === 'light') return 'parchment';
		if (legacy === 'dark') return 'anthracite';
	}
	if (
		typeof window !== 'undefined' &&
		window.matchMedia &&
		window.matchMedia('(prefers-color-scheme: light)').matches
	) {
		return 'parchment';
	}
	return 'anthracite';
}

function initialDensity(): Density {
	return readStored<Density>(DENSITY_KEY, DENSITIES) ?? 'comfortable';
}

function initialFooter(): boolean {
	if (typeof localStorage === 'undefined') return true;
	const v = localStorage.getItem(FOOTER_KEY);
	return v === null ? true : v === 'true';
}

function initialTypeScale(): TypeScale {
	return readStored<TypeScale>(TYPE_SCALE_KEY, TYPE_SCALES) ?? 'default';
}

function initialMotion(): MotionPref {
	return readStored<MotionPref>(MOTION_KEY, MOTION_PREFS) ?? 'system';
}

/** Currently selected palette. */
export const palette = writable<ThemePalette>(initialPalette());
/** Currently selected density. */
export const density = writable<Density>(initialDensity());
/** Whether the optional status footer is shown (spec 8.5). */
export const statusFooterVisible = writable<boolean>(initialFooter());
/** Currently selected typography scale. */
export const typeScale = writable<TypeScale>(initialTypeScale());
/** Currently selected motion preference. */
export const motionPref = writable<MotionPref>(initialMotion());

/** Apply the palette to <html>: data-oo-theme + sync .dark + darkMode store. */
function applyPalette(p: ThemePalette, animate = true): void {
	if (typeof document === 'undefined') return;
	const html = document.documentElement;
	const dark = isDarkPalette(p);
	const reduced = get(prefersReducedMotion);
	if (animate && !reduced) html.classList.add('theme-transitioning');
	html.setAttribute('data-oo-theme', p);
	html.classList.toggle('dark', dark);
	darkMode.set(dark);
	if (animate && !reduced) {
		setTimeout(() => html.classList.remove('theme-transitioning'), 350);
	}
}

/** Apply the density to <html>: a single oo-density-* class. */
function applyDensity(d: Density): void {
	if (typeof document === 'undefined') return;
	const html = document.documentElement;
	DENSITIES.forEach((x) => html.classList.remove(`oo-density-${x}`));
	html.classList.add(`oo-density-${d}`);
}

/** Apply the typography scale: a single --oo-type-scale multiplier on <html>. */
function applyTypeScale(t: TypeScale): void {
	if (typeof document === 'undefined') return;
	document.documentElement.style.setProperty('--oo-type-scale', String(TYPE_SCALE_VALUE[t]));
}

/**
 * Apply the motion preference. `oo-reduce-motion` forces reduction regardless
 * of the OS; `oo-motion-full` opts out of the prefers-reduced-motion media
 * query (app.css scopes it to `html:not(.oo-motion-full)`). The
 * prefersReducedMotion store is kept in sync so JS-driven animations (the
 * theme-transition flag) honor the choice too.
 */
function applyMotion(m: MotionPref): void {
	if (typeof document === 'undefined') return;
	const html = document.documentElement;
	html.classList.toggle('oo-reduce-motion', m === 'reduced');
	html.classList.toggle('oo-motion-full', m === 'full');
	if (m === 'reduced') {
		prefersReducedMotion.set(true);
	} else if (m === 'full') {
		prefersReducedMotion.set(false);
	} else if (typeof window !== 'undefined' && window.matchMedia) {
		prefersReducedMotion.set(window.matchMedia('(prefers-reduced-motion: reduce)').matches);
	}
}

/** Select a palette: update the store, the document, and localStorage. */
export function setPalette(p: ThemePalette): void {
	palette.set(p);
	applyPalette(p, true);
	if (typeof localStorage !== 'undefined') {
		localStorage.setItem(PALETTE_KEY, p);
		localStorage.setItem(LEGACY_THEME_KEY, isDarkPalette(p) ? 'dark' : 'light');
	}
}

/** Select a density: update the store, the document, and localStorage. */
export function setDensity(d: Density): void {
	density.set(d);
	applyDensity(d);
	if (typeof localStorage !== 'undefined') localStorage.setItem(DENSITY_KEY, d);
}

/** Toggle the optional status footer and persist the choice. */
export function setStatusFooterVisible(v: boolean): void {
	statusFooterVisible.set(v);
	if (typeof localStorage !== 'undefined') localStorage.setItem(FOOTER_KEY, String(v));
}

/** Select a typography scale: update the store, the document, and localStorage. */
export function setTypeScale(t: TypeScale): void {
	typeScale.set(t);
	applyTypeScale(t);
	if (typeof localStorage !== 'undefined') localStorage.setItem(TYPE_SCALE_KEY, t);
}

/** Select a motion preference: update the store, the document, and localStorage. */
export function setMotionPref(m: MotionPref): void {
	motionPref.set(m);
	applyMotion(m);
	if (typeof localStorage !== 'undefined') localStorage.setItem(MOTION_KEY, m);
}

/**
 * Initialize palette + density + typography + motion on the document at
 * startup, without the transition flash. Call once from the root layout
 * after initTheme().
 */
export function initPreferences(): void {
	applyPalette(get(palette), false);
	applyDensity(get(density));
	applyTypeScale(get(typeScale));
	applyMotion(get(motionPref));
}
