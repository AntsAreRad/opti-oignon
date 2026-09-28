/**
 * The step plants of the chat loader, as pixel data.
 *
 * A run the server executes shows one plant per step. Every frame is a grid
 * of 11 x 16 art pixels, one character per pixel, and every frame stands on
 * the same baseline: the soil line is row 11, shared by the whole row of
 * steps. The seed and the roots are drawn under it, with no ground behind
 * them.
 *
 * The stages follow the onion's own germination. An onion has one
 * cotyledon. After the root, it comes up bent into a crook: the bend breaks
 * the soil while its tip stays in the seed underground. It then straightens
 * and pulls the empty seed coat out of the soil at its tip: the flag. The
 * first true leaf comes out of its sheath beside it. The bulb, much later
 * in a real garden, stands for a finished step.
 *
 * The characters, each drawn in one app token (`PLANT_INKS`):
 *
 *   .  nothing           -  the soil line
 *   g  a growing plant   k  the leaves of a finished plant
 *   s  seed, coat, bulb  r  roots
 *   w  a waiting seed    m  a plant stopped by a safety mechanism
 *   y  a plant whose step failed, kept as it stood, in straw
 *   t  the machine's tin tag hanging from that plant's stem
 *   u  the dots of a plant whose outcome is unknown
 *
 * The living frames are drawn here pixel by pixel. The ends are derived
 * from them, never drawn apart: a failed step's plant, a stopped one and an
 * unknown one each keep the exact silhouette of their stage's first frame.
 *
 * Only data and pure functions: no clock, no timer, no DOM.
 */

import type { Stage, StepState } from '../chat/progress';

/** One frame: one string per row, one character per art pixel. */
export type Grid = readonly string[];

/**
 * How a strip moves. The strip component plays each motion by one CSS class
 * whose steps and durations are the table's own (a contract holds them
 * equal): `turn` the onion's loop, `sway` a running plant's, `root` the
 * root's frame once and then the sway, `bloom` the flowering once, then the
 * bulb.
 */
export type Motion = 'still' | 'turn' | 'sway' | 'root' | 'bloom';

/**
 * What one drawing plays. Frame 0 is the still: the only frame drawn under
 * reduced motion, and the one shown while nothing moves. The first `loop`
 * frames loop, `stepMs` each. The `lead` frames after them play once before
 * the loop, `leadMs` each; a strip whose `loop` is 1 rests on frame 0 once
 * its lead has played.
 */
export interface Strip {
	frames: readonly Grid[];
	motion: Motion;
	loop: number;
	stepMs: number;
	lead: number;
	leadMs: number;
}

export const PLANT_W = 11;
export const PLANT_H = 16;
/** The soil line's row. Everything at or above it is the plant's aerial part. */
export const SOIL_ROW = 11;
/** Art pixels between two plants of a row. */
export const PLANT_GAP = 3;
/** One plant and its gap. */
export const PLANT_PITCH = PLANT_W + PLANT_GAP;

/** The running plant sways between its two frames at this pace, 1.25 Hz. */
export const SWAY_MS = 800;
/** The root shows this long when a step starts running. */
export const ROOT_MS = 120;
/** Each frame of the flowering that follows `done`. */
export const BLOOM_MS = 120;

/** The app token each character is drawn in. */
export const PLANT_INKS: Readonly<Record<string, string>> = {
	'-': '--oo-bd-subtle',
	g: '--oo-acc-mark',
	k: '--oo-status-ok',
	s: '--oo-acc-mark-2',
	r: '--oo-fg-muted',
	w: '--oo-bd-strong',
	m: '--oo-fg-muted',
	y: '--oo-dried',
	t: '--oo-machine',
	u: '--oo-bd-strong',
};

/**
 * The grounds a row of plants may stand on: the page and the surface. Never
 * a tint or the sunken ground, where the finished leaves and the seeds fall
 * under 3:1.
 */
export const ROW_GROUNDS: readonly string[] = ['--oo-bg-base', '--oo-bg-surface'];

/** An art pixel in CSS pixels at an integer scale: every edge on a device pixel. */
export function artPixel(scale: number, dpr = 1): number {
	return Math.round(scale * dpr) / dpr;
}

// ---------------------------------------------------------------------------
// The living frames
// ---------------------------------------------------------------------------

/** A seed waiting under the soil: a step not started, or never run. */
const SEED: Grid = [
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'-----------',
	'......ww...',
	'.....ww....',
	'...........',
	'...........',
];

/** The root comes out of the seed: a step starts running. */
const ROOT: Grid = [
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'-----------',
	'......ss...',
	'.....ss....',
	'.....r.....',
	'...........',
];

/** The crook: the bend above the soil, both arms in it, the tip still in the seed. */
const CROOK_A: Grid = [
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'....gg.....',
	'...g..g....',
	'...g..g....',
	'...g..g....',
	'---g--g----',
	'...r..ss...',
	'...r.ss....',
	'...........',
	'...........',
];

/** The bend leans one pixel. */
const CROOK_B: Grid = [
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'.....gg....',
	'....g..g...',
	'...g..g....',
	'...g..g....',
	'---g--g----',
	'...r..ss...',
	'...r.ss....',
	'...........',
	'...........',
];

/** The flag: the cotyledon straight, the empty seed coat out of the soil at its tip. */
const FLAG_A: Grid = [
	'...........',
	'...........',
	'.....g.....',
	'....g.g....',
	'....g.ss...',
	'....g..ss..',
	'....g......',
	'....g......',
	'....g......',
	'....g......',
	'....g......',
	'----g------',
	'....r......',
	'....r......',
	'....r......',
	'...........',
];

/** The flag leans one pixel. */
const FLAG_B: Grid = [
	'...........',
	'...........',
	'......g....',
	'.....g.g...',
	'.....g.ss..',
	'.....g..ss.',
	'....g......',
	'....g......',
	'....g......',
	'....g......',
	'....g......',
	'----g------',
	'....r......',
	'....r......',
	'....r......',
	'...........',
];

/** The first true leaf, out of the sheath beside the flag, and taller. */
const LEAF_A: Grid = [
	'..g........',
	'..g........',
	'..g..g.....',
	'..g.g.g....',
	'..g.g.ss...',
	'..g.g..ss..',
	'..g.g......',
	'...gg......',
	'....g......',
	'....g......',
	'....g......',
	'----g------',
	'....r......',
	'....r......',
	'....r......',
	'....r......',
];

/** The leaf's tip leans one pixel. */
const LEAF_B: Grid = [
	'...g.......',
	'...g.......',
	'..g..g.....',
	'..g.g.g....',
	'..g.g.ss...',
	'..g.g..ss..',
	'..g.g......',
	'...gg......',
	'....g......',
	'....g......',
	'....g......',
	'----g------',
	'....r......',
	'....r......',
	'....r......',
	'....r......',
];

/** The flowering after `done`, first frame: three leaves, the bulb's base forming. */
const BLOOM_LEAVES: Grid = [
	'.....k.....',
	'...k.k.....',
	'...k.k.k...',
	'...k.k.k...',
	'....kkk....',
	'.....k.....',
	'.....k.....',
	'.....k.....',
	'.....k.....',
	'.....k.....',
	'.....k.....',
	'----sss----',
	'....rrr....',
	'.....r.....',
	'...........',
	'...........',
];

/** The flowering, second frame: the bulb swells. */
const BLOOM_SWELL: Grid = [
	'.....k.....',
	'...k.k.....',
	'...k.k.k...',
	'...k.k.k...',
	'....kkk....',
	'.....k.....',
	'.....k.....',
	'.....s.....',
	'....sss....',
	'...sssss...',
	'...sssss...',
	'---sssss---',
	'....rrr....',
	'.....r.....',
	'...........',
	'...........',
];

/** A finished step: the bulb, a teardrop whose neck runs into three upright leaves, and a tuft of roots. */
const BULB: Grid = [
	'.....k.....',
	'...k.k.....',
	'...k.k.k...',
	'...k.k.k...',
	'....kkk....',
	'.....s.....',
	'....sss....',
	'...sssss...',
	'..sssssss..',
	'..sssssss..',
	'..sssssss..',
	'---sssss---',
	'....rrr....',
	'.....r.....',
	'...........',
	'...........',
];

/** A skipped step: bare soil. */
const BARE: Grid = [
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'...........',
	'-----------',
	'...........',
	'...........',
	'...........',
	'...........',
];

type Reached = 'crook' | 'flag' | 'leaf';

const RUNNING: Readonly<Record<Reached, readonly [Grid, Grid]>> = {
	crook: [CROOK_A, CROOK_B],
	flag: [FLAG_A, FLAG_B],
	leaf: [LEAF_A, LEAF_B],
};

// ---------------------------------------------------------------------------
// The ends, derived from the living frames
// ---------------------------------------------------------------------------

const PLANT = new Set(['g', 'k', 's', 'r', 'w']);

function freeze(rows: string[][]): Grid {
	return Object.freeze(rows.map((row) => row.join('')));
}

/** Every plant pixel in one ink; the soil stays. */
function recolour(grid: Grid, ink: string): Grid {
	return freeze(grid.map((line) => [...line].map((c) => (PLANT.has(c) ? ink : c))));
}

/** One plant pixel in two, as a checkerboard; the others are not drawn. */
function dotted(grid: Grid): Grid {
	return freeze(
		grid.map((line, y) =>
			[...line].map((c, x) => {
				if (!PLANT.has(c)) return c;
				if ((x + y) % 2 === 1) return 'u';
				return y === SOIL_ROW ? '-' : '.';
			}),
		),
	);
}

/**
 * Where the tin tag hangs, per stage: the top left corner of its 3 x 3 box,
 * right of the stem at the same height for every stage, clear of the soil,
 * its point touching the stem and nothing else of the plant.
 */
const TAG_AT: Readonly<Record<Reached, readonly [number, number]>> = {
	crook: [7, 7],
	flag: [5, 7],
	leaf: [5, 7],
};

/** The tag: a small label whose point, on its left, is where it hangs. */
const TAG: Grid = ['.tt', 'ttt', '.tt'];

/**
 * A failed step's plant, kept as it stood: its stage's first frame,
 * straight and whole, every part at or above the soil line in the pale
 * straw, the seed coat too when it is out, and the machine's tin tag
 * hanging from its stem. Nothing droops, nothing falls, nothing on it is
 * red. Under the soil nothing changes.
 */
function pressed(stage: Reached): Grid {
	const rows = RUNNING[stage][0].map((line) => [...line]);
	for (let y = 0; y <= SOIL_ROW; y += 1) {
		for (let x = 0; x < PLANT_W; x += 1) if (PLANT.has(rows[y][x])) rows[y][x] = 'y';
	}
	const [left, top] = TAG_AT[stage];
	TAG.forEach((line, dy) =>
		[...line].forEach((c, dx) => {
			if (c === 't') rows[top + dy][left + dx] = 't';
		}),
	);
	return freeze(rows);
}

const PRESSED: Readonly<Record<Reached, Grid>> = {
	crook: pressed('crook'),
	flag: pressed('flag'),
	leaf: pressed('leaf'),
};

const STOPPED: Readonly<Record<'seed' | Reached, Grid>> = {
	seed: recolour(SEED, 'm'),
	crook: recolour(CROOK_A, 'm'),
	flag: recolour(FLAG_A, 'm'),
	leaf: recolour(LEAF_A, 'm'),
};

const UNKNOWN: Readonly<Record<'seed' | Reached, Grid>> = {
	seed: dotted(SEED),
	crook: dotted(CROOK_A),
	flag: dotted(FLAG_A),
	leaf: dotted(LEAF_A),
};

/** Every frame by name: what a preview or a contract reads. */
export const PLANT_FRAMES: Readonly<Record<string, Grid>> = {
	seed: SEED,
	root: ROOT,
	crook_a: CROOK_A,
	crook_b: CROOK_B,
	flag_a: FLAG_A,
	flag_b: FLAG_B,
	leaf_a: LEAF_A,
	leaf_b: LEAF_B,
	bloom_leaves: BLOOM_LEAVES,
	bloom_swell: BLOOM_SWELL,
	bulb: BULB,
	bare: BARE,
	pressed_crook: PRESSED.crook,
	pressed_flag: PRESSED.flag,
	pressed_leaf: PRESSED.leaf,
	stopped_seed: STOPPED.seed,
	stopped_crook: STOPPED.crook,
	stopped_flag: STOPPED.flag,
	stopped_leaf: STOPPED.leaf,
	unknown_seed: UNKNOWN.seed,
	unknown_crook: UNKNOWN.crook,
	unknown_flag: UNKNOWN.flag,
	unknown_leaf: UNKNOWN.leaf,
};

function still(grid: Grid): Strip {
	return { frames: [grid], motion: 'still', loop: 1, stepMs: 0, lead: 0, leadMs: 0 };
}

/** A plant that ran reached at least the crook. */
function reached(stage: Stage): Reached {
	return stage === 'flag' || stage === 'leaf' ? stage : 'crook';
}

/** What a step's plant plays, from its state and its stage. */
export function plantStrip(state: StepState, stage: Stage): Strip {
	switch (state) {
		case 'running': {
			const at = reached(stage);
			const [a, b] = RUNNING[at];
			if (at === 'crook') {
				return { frames: [a, b, ROOT], motion: 'root', loop: 2, stepMs: SWAY_MS, lead: 1, leadMs: ROOT_MS };
			}
			return { frames: [a, b], motion: 'sway', loop: 2, stepMs: SWAY_MS, lead: 0, leadMs: 0 };
		}
		case 'done':
			return {
				frames: [BULB, BLOOM_LEAVES, BLOOM_SWELL],
				motion: 'bloom',
				loop: 1,
				stepMs: 0,
				lead: 2,
				leadMs: BLOOM_MS,
			};
		case 'failed':
			return still(PRESSED[reached(stage)]);
		case 'cancelled':
			return still(stage === 'seed' ? STOPPED.seed : STOPPED[reached(stage)]);
		case 'unknown':
			return still(stage === 'seed' ? UNKNOWN.seed : UNKNOWN[reached(stage)]);
		case 'skipped':
			return still(BARE);
		default:
			return still(SEED);
	}
}

/**
 * The soil between two plants: the row's one soil line runs on under the
 * gap, so every plant of a run stands on the same line.
 */
export const SOIL_GAP: Strip = still(
	Object.freeze(Array.from({ length: PLANT_H }, (_, y) => (y === SOIL_ROW ? '-' : '.').repeat(PLANT_GAP))),
);

/** How a row of `steps` plants fits a column: its scale, and the window shown. */
export interface RowFit {
	scale: number;
	/** The first plant shown, and how many. */
	first: number;
	count: number;
	/** The plants left out before and after the window, said in words. */
	earlier: number;
	later: number;
}

/** The scales a row is drawn at, largest first. Never below 2x. */
export const ROW_SCALES: readonly number[] = [3, 2];

/** How many plants a column holds at a scale: one pitch each. */
export function rowCapacity(columnPx: number, scale: number, dpr = 1): number {
	return Math.max(0, Math.floor(columnPx / (PLANT_PITCH * artPixel(scale, dpr))));
}

/**
 * 3x when every plant fits the column, else 2x; else, at 2x, a window of
 * the 2x capacity less two plants, centred on the current step, the counts
 * left out said at both ends. Never below 2x.
 */
export function fitRow(steps: number, current: number, columnPx: number, dpr = 1): RowFit {
	for (const scale of ROW_SCALES) {
		if (steps <= rowCapacity(columnPx, scale, dpr)) {
			return { scale, first: 0, count: steps, earlier: 0, later: 0 };
		}
	}
	const scale = ROW_SCALES[ROW_SCALES.length - 1];
	const count = Math.min(steps, Math.max(1, rowCapacity(columnPx, scale, dpr) - 2));
	const at = Math.min(Math.max(current, 0), steps - 1);
	const first = Math.min(Math.max(at - Math.floor((count - 1) / 2), 0), steps - count);
	return { scale, first, count, earlier: first, later: steps - first - count };
}
