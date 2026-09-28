/**
 * The pace at which a step's plant changes its drawing.
 *
 * The reducer (`lib/chat/progress.ts`) says what a plant should look like as
 * each frame arrives; this module says what it shows, and when. Each look
 * stays at least 400 ms, so a burst of frames never flickers through the
 * stages. When more than two looks wait, the plant jumps to the last one
 * rather than falling behind. A run that ends within its first 400 ms shows
 * only its final looks: nothing of it is drawn before its end. And an end (a
 * finished, failed, stopped, skipped, unrun or unknown step) is never kept
 * waiting behind a living stage: the plant goes straight to it, so no one
 * ever watches a plant dry.
 *
 * Everything here is pure and clock-free: every time is handed in, and the
 * caller sets its own timer for `nextAt`.
 */

/** The least time a look is shown. */
export const STAGE_MS = 400;

/** More looks than this waiting, and the plant jumps to the last. */
export const MAX_WAITING = 2;

/** A look the reducer gave a plant, and when. `end`: the step's state is final. */
export interface Arrival {
	at: number;
	look: string;
	end: boolean;
}

/** A look the plant shows, from when. */
export interface Shown {
	at: number;
	look: string;
}

/**
 * The looks a plant shows, and when, from the looks it was given. `start`
 * is when its run opened, `end` when its run ended (null while it is open).
 * The past of a schedule never changes when a later look arrives.
 */
export function schedule(arrivals: readonly Arrival[], start: number, end: number | null): Shown[] {
	const given: Arrival[] = [];
	for (const arrival of [...arrivals].sort((a, b) => a.at - b.at)) {
		if (given.length === 0 || given[given.length - 1].look !== arrival.look) given.push(arrival);
	}
	if (given.length === 0) return [];

	// Nothing is drawn before the run's first 400 ms, or before its end if it ends sooner.
	const first = Math.max(given[0].at, end === null ? start + STAGE_MS : Math.min(start + STAGE_MS, end));
	let next = given.findIndex((arrival) => arrival.at > first);
	if (next === -1) next = given.length;
	const shown: Shown[] = [{ at: first, look: given[next - 1].look }];

	while (next < given.length) {
		const at = Math.max(shown[shown.length - 1].at + STAGE_MS, given[next].at);
		let last = next;
		while (last + 1 < given.length && given[last + 1].at <= at) last += 1;
		const waiting = last - next + 1;
		const pick = waiting > MAX_WAITING || given[last].end ? last : next;
		shown.push({ at, look: given[pick].look });
		next = pick + 1;
	}
	return shown;
}

/** The look shown at `now`, or null before the first. */
export function shownAt(plan: readonly Shown[], now: number): string | null {
	let look: string | null = null;
	for (const entry of plan) {
		if (entry.at > now) break;
		look = entry.look;
	}
	return look;
}

/** When the shown look next changes after `now`, or null when it will not. */
export function nextAt(plan: readonly Shown[], now: number): number | null {
	const entry = plan.find((item) => item.at > now);
	return entry ? entry.at : null;
}
