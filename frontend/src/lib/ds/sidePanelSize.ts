/**
 * sidePanelSize.ts (lib/ds) -- a side panel's width, as pure functions.
 *
 * SidePanel.svelte asks resizeKey what a key on its resize handle does,
 * dragWidth what a drag of the handle does and releaseWidth what letting go
 * of it does; every other width it is given goes through clampWidth. A
 * panel on the right grows toward the left, so the left arrow widens it; a
 * panel on the left grows toward the right. Home and End take it to its
 * narrowest and widest. A move of DRAG_SLOP pixels or less is not yet a
 * drag, and a press released without a drag steps the width to the next of
 * three stops (the narrowest, the middle, the widest), past the widest back
 * to the narrowest: a pointer resizes the panel without dragging. A width
 * is a whole number of pixels between the bounds; a width that is not a
 * number is the narrowest.
 *
 * Dependency-free: the contracts run it under Node.
 */

export interface PanelBounds {
	/** The narrowest width, in px. */
	min: number;
	/** The widest width, in px. */
	max: number;
	/** How far one arrow key moves the edge, in px. */
	step: number;
	/** The side of the page the panel stands on; right when not given. */
	side?: 'left' | 'right';
}

/** `width` rounded to a whole pixel and held between the bounds. */
export function clampWidth(width: number, bounds: PanelBounds): number {
	if (!Number.isFinite(width)) return bounds.min;
	return Math.min(bounds.max, Math.max(bounds.min, Math.round(width)));
}

/** The width a key on the resize handle sets, or null for a key the handle does not take. */
export function resizeKey(key: string, width: number, bounds: PanelBounds): number | null {
	const towardPage = (bounds.side ?? 'right') === 'right' ? 'ArrowLeft' : 'ArrowRight';
	const towardEdge = towardPage === 'ArrowLeft' ? 'ArrowRight' : 'ArrowLeft';
	switch (key) {
		case towardPage:
			return clampWidth(width + bounds.step, bounds);
		case towardEdge:
			return clampWidth(width - bounds.step, bounds);
		case 'Home':
			return bounds.min;
		case 'End':
			return bounds.max;
		default:
			return null;
	}
}

/** How far, in px, the pointer may move before a press becomes a drag. */
export const DRAG_SLOP = 3;

/** The width after dragging the handle from `startX` to `x`, from `startWidth`. */
export function dragWidth(startWidth: number, startX: number, x: number, bounds: PanelBounds): number {
	if (Math.abs(x - startX) <= DRAG_SLOP) return clampWidth(startWidth, bounds);
	const moved = (bounds.side ?? 'right') === 'right' ? startX - x : x - startX;
	return clampWidth(startWidth + moved, bounds);
}

/** The three widths a press steps through: the narrowest, the middle, the widest. */
export function widthStops(bounds: PanelBounds): number[] {
	return [bounds.min, Math.round((bounds.min + bounds.max) / 2), bounds.max];
}

/**
 * The width when the pointer is let go at `x`: after a drag, where the drag
 * went; after a press with no drag, the next stop above `startWidth`, or the
 * narrowest past the widest.
 */
export function releaseWidth(startWidth: number, startX: number, x: number, bounds: PanelBounds): number {
	if (Math.abs(x - startX) > DRAG_SLOP) return dragWidth(startWidth, startX, x, bounds);
	const width = clampWidth(startWidth, bounds);
	return widthStops(bounds).find((stop) => stop > width) ?? bounds.min;
}
