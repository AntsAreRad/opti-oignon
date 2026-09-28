/**
 * The loader's onion, as pixel data: the fallback the status line shows
 * beside its words until the first token.
 *
 * A full onion of 11 x 12 art pixels, one character per pixel, drawn in the
 * brand ink: a thin sprout and neck flaring into a round shoulder, a full
 * belly, a gently narrowing base, and a small tuft of roots under it. Its
 * silhouette never changes. Four skin lines, a quarter turn apart, cross the
 * bulb in the colour of the ground it stands on, and each frame turns them
 * by a sixteenth of a turn. A line is the projection of its meridian on the
 * bulb's rows, kept two pixels inside the outline, from the shoulder down to
 * above the base: the lines draw together toward the neck as an onion's
 * skin does, and the bulb stays one piece in every frame. The positions are
 * data, computed once and frozen here, never a formula.
 *
 *   .  nothing    o  the onion    x  a skin line
 *
 * Only data: no clock, no timer, no DOM.
 */

import type { Grid, Strip } from './plantFrames';

export const ONION_W = 11;
export const ONION_H = 12;

/** One step of the turn: 2.8 steps a second, under the 3 Hz of the flicker rule. */
export const ONION_STEP_MS = 360;

/** The app token each character is drawn in. */
export const ONION_INKS: Readonly<Record<string, string>> = {
	o: '--oo-acc-mark',
	x: '--oo-mark-ground',
};

/** The grounds the onion stands on: the thread's page ground and a card's surface. */
export const ONION_GROUNDS: readonly string[] = ['--oo-bg-base', '--oo-bg-surface'];

const BASE: Grid = [
	'.....o.....',
	'.....o.....',
	'....ooo....',
	'..ooooooo..',
	'.ooooooooo.',
	'ooooooooooo',
	'ooooooooooo',
	'ooooooooooo',
	'.ooooooooo.',
	'..ooooooo..',
	'...o.o.o...',
	'...........',
];

/** The skin lines' columns on each row of the bulb (rows 3 to 8), per frame. */
const MERIDIANS: readonly Readonly<Record<number, readonly number[]>>[] = [
	{ 3: [5], 4: [5], 5: [5], 6: [5], 7: [5], 8: [5] },
	{ 3: [4, 6], 4: [3, 7], 5: [2, 7], 6: [2, 7], 7: [2, 7], 8: [3, 7] },
	{ 3: [4, 6], 4: [3, 7], 5: [2, 8], 6: [2, 8], 7: [2, 8], 8: [3, 7] },
	{ 3: [4, 6], 4: [3, 7], 5: [3, 8], 6: [3, 8], 7: [3, 8], 8: [3, 7] },
];

function turned(meridians: Readonly<Record<number, readonly number[]>>): Grid {
	return Object.freeze(
		BASE.map((line, y) => [...line].map((c, x) => ((meridians[y] ?? []).includes(x) ? 'x' : c)).join('')),
	);
}

/** The four positions of the turn; the first is the still. */
export const ONION_FRAMES: readonly Grid[] = Object.freeze(MERIDIANS.map(turned));

/**
 * The loop: ten steps of 360 ms, 3.6 s. It holds still for its first three
 * steps, 1.08 s, then turns twice through the four positions.
 */
export const ONION_SEQUENCE: readonly number[] = [0, 0, 0, 1, 2, 3, 0, 1, 2, 3];

/** What the onion plays; frame 0 is also the frame reduced motion holds. */
export const ONION_STRIP: Strip = {
	frames: ONION_SEQUENCE.map((index) => ONION_FRAMES[index]),
	motion: 'turn',
	loop: ONION_SEQUENCE.length,
	stepMs: ONION_STEP_MS,
	lead: 0,
	leadMs: 0,
};
