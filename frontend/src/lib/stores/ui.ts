/**
 * Svelte stores for UI state (sidebar, theme).
 * System theme detection, smooth transition on toggle,
 *      prefers-reduced-motion awareness.
 */

import { writable, get } from 'svelte/store';

/** Sidebar open/closed (desktop: always visible, mobile: overlay). */
export const sidebarOpen = writable<boolean>(true);

/**
 * Whether the user prefers reduced motion.
 * Updated on mount and when the media query changes.
 */
export const prefersReducedMotion = writable<boolean>(false);

/**
 * Resolve initial dark mode state:
 * 1. Check localStorage for explicit user preference
 * 2. Fall back to OS preference via prefers-color-scheme
 * 3. Default to dark if neither available
 */
function getInitialDarkMode(): boolean {
	if (typeof window === 'undefined') return true;
	const stored = localStorage.getItem('oo-theme');
	if (stored === 'light') return false;
	if (stored === 'dark') return true;
	if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
		return false;
	}
	return true;
}

/** Dark/light theme state. */
export const darkMode = writable<boolean>(getInitialDarkMode());

/** Toggle sidebar. */
export function toggleSidebar(): void {
	sidebarOpen.update((v) => !v);
}

/**
 * Toggle theme with smooth CSS transition.
 * Adds a transitioning class to <html> so all properties animate,
 * then removes it after the transition completes.
 * Skips the transition if the user prefers reduced motion.
 */
export function toggleTheme(): void {
	darkMode.update((v) => {
		const next = !v;
		if (typeof document !== 'undefined') {
			const html = document.documentElement;
			const reduced = get(prefersReducedMotion);

			// Add transition class unless reduced motion is preferred
			if (!reduced) {
				html.classList.add('theme-transitioning');
			}

			html.classList.toggle('dark', next);
			localStorage.setItem('oo-theme', next ? 'dark' : 'light');

			// Remove transition class after animation completes
			if (!reduced) {
				setTimeout(() => {
					html.classList.remove('theme-transitioning');
				}, 350);
			}
		}
		return next;
	});
}

/**
 * Initialize theme class on document load.
 * Also sets up listeners for system theme changes and reduced-motion.
 */
export function initTheme(): void {
	if (typeof document === 'undefined' || typeof window === 'undefined') return;

	const isDark = getInitialDarkMode();
	document.documentElement.classList.toggle('dark', isDark);
	darkMode.set(isDark);

	// Listen for OS theme changes (only applies if no explicit user preference)
	try {
		const colorSchemeQuery = window.matchMedia('(prefers-color-scheme: dark)');
		colorSchemeQuery.addEventListener('change', (e) => {
			const stored = localStorage.getItem('oo-theme');
			// Only follow system if user has no explicit preference
			if (!stored) {
				const sysDark = e.matches;
				document.documentElement.classList.toggle('dark', sysDark);
				darkMode.set(sysDark);
			}
		});
	} catch {
		// matchMedia listener not supported, ignore
	}

	// Track prefers-reduced-motion
	try {
		const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
		prefersReducedMotion.set(motionQuery.matches);
		motionQuery.addEventListener('change', (e) => {
			prefersReducedMotion.set(e.matches);
		});
	} catch {
		// matchMedia listener not supported, ignore
	}
}
