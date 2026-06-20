/**
 * Svelte stores for side panel state (artifacts, code execution).
 *
 * Gere le panneau lateral droit: type actif, largeur, toggle.
 * Mobile: le panneau devient un overlay plein ecran.
 */

import { writable, derived } from 'svelte/store';
import type { PanelType } from '$lib/types';

/** Panneau actif (none = ferme). */
export const activePanel = writable<PanelType>('none');

/** Largeur du panneau en pixels (desktop). */
export const panelWidth = writable<number>(420);

/** Largeur minimale et maximale du panneau. */
export const PANEL_MIN_WIDTH = 300;
export const PANEL_MAX_WIDTH = 800;

/** Le panneau est-il ouvert? */
export const isPanelOpen = derived(activePanel, ($p) => $p !== 'none');

/**
 * Toggle un panneau: ouvre s'il est ferme ou different,
 * ferme si c'est le meme deja ouvert.
 */
export function togglePanel(panel: PanelType): void {
	activePanel.update((current) => (current === panel ? 'none' : panel));
}

/** Ferme le panneau. */
export function closePanel(): void {
	activePanel.set('none');
}

/** Met a jour la largeur du panneau (clampee aux limites). */
export function setPanelWidth(width: number): void {
	panelWidth.set(Math.max(PANEL_MIN_WIDTH, Math.min(PANEL_MAX_WIDTH, width)));
}
