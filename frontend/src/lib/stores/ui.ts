/**
 * Svelte stores for UI state: the sidebar, whether the shell is drawn for a
 * phone, whether the palette on screen is a dark one, and the system's
 * reduced-motion setting.
 *
 * Nothing here writes the root element or storage: the palette and every
 * other appearance choice go through the preferences store and the theme
 * path (lib/theme/apply.ts), and the preferences store keeps darkMode in
 * step with what the theme path applied.
 */

import { writable } from 'svelte/store';

/**
 * The sidebar: on a desktop, expanded (true) or the 72 px rail (false), and
 * mounted either way; on a phone, the drawer open (true) or shut (false).
 */
export const sidebarOpen = writable<boolean>(true);

/**
 * Whether the shell is drawn for a phone (under 768 px): a header holding
 * the stop and the drawer's opener, the sidebar as a drawer. The shell sets
 * it from the viewport's width.
 */
export const isPhone = writable<boolean>(false);

/**
 * Whether motion should be reduced: the user's motion choice, or the
 * system's setting when the choice is to follow it.
 */
export const prefersReducedMotion = writable<boolean>(false);

/** Whether the palette on screen is a dark one, as the pre-render left it. */
export const darkMode = writable<boolean>(
	typeof document === 'undefined' ? true : document.documentElement.classList.contains('dark')
);

/** Toggle sidebar. */
export function toggleSidebar(): void {
	sidebarOpen.update((v) => !v);
}

/**
 * Tracks the system's reduced-motion setting while the motion choice is to
 * follow it. Call once at startup, after the preferences are applied.
 */
export function initReducedMotion(): void {
	if (typeof document === 'undefined' || typeof window === 'undefined') return;
	try {
		const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
		const followed = () =>
			!document.documentElement.classList.contains('oo-reduce-motion') &&
			!document.documentElement.classList.contains('oo-motion-full');
		if (followed()) prefersReducedMotion.set(motionQuery.matches);
		motionQuery.addEventListener('change', (e) => {
			if (followed()) prefersReducedMotion.set(e.matches);
		});
	} catch {
		// matchMedia listener not supported, ignore
	}
}
