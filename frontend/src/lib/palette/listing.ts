/**
 * What the command palette's list says to the reader, beyond its order:
 * which option is active, the id each option carries in the page, and the
 * line its status region reads.
 *
 * The active option is held by its identity (its group and its id), never by
 * its place: the list is ranked again when the server answers or the chats
 * change, and an option held by its place would be swapped for another under
 * the reader's Enter. The option the reader picked (with the arrows, Home,
 * End or the pointer) stays active while it is listed; otherwise the first
 * option that can run is, so Enter never lands on a command that would do
 * nothing while another one could run; with none that can, the first option.
 * New words drop the pick, and the best match is active again.
 *
 * Each option's id in the page is built from its identity, one to one, so a
 * new best match changes the id the field names as active, and assistive
 * technology reads it out; the same option keeps its id however it moves.
 *
 * The status line says how many options are listed once the list has
 * settled ("3 results"), that nothing matches, that the chats could not be
 * searched, or that the chats are still being searched when the server is
 * slow to answer (SLOW_MS); while the answer is only a moment away it says
 * nothing. `createStatusLine` paces it: a line is spoken once it has stood
 * still for SETTLE_MS, so a burst of keys is not read out key by key; an
 * empty line and a notice are spoken at once. A disabled option the reader
 * tried to open says why it cannot run, before anything else.
 *
 * Pure functions, with no dependency: the palette hands them its list, and
 * the timer is handed in too, so they run under Node as they are.
 */

/** What these functions read of an option. */
export interface Listed {
	id: string;
	group: string;
	disabled?: boolean;
}

/** A move of the active option. */
export type Move = 'next' | 'previous' | 'first' | 'last';

/** An option's identity in the list: its group and its id. */
export function optionKey(option: Listed): string {
	return `${option.group}:${option.id}`;
}

/**
 * The key of the active option in `flat` (the list as shown, groups in
 * order): `pick` while it is listed, else the first option that can run,
 * else the first option; null for an empty list.
 */
export function activeFor(flat: readonly Listed[], pick: string | null): string | null {
	if (flat.length === 0) return null;
	if (pick !== null && flat.some((option) => optionKey(option) === pick)) return pick;
	const runnable = flat.find((option) => option.disabled !== true);
	return optionKey(runnable ?? flat[0]);
}

/**
 * The key the active option moves to: the next or the previous one, wrapping
 * at either end, or the first or the last. From no active option, next goes
 * to the first and previous to the last. Disabled options are visited too:
 * their reason is read on the way.
 */
export function stepActive(flat: readonly Listed[], active: string | null, move: Move): string | null {
	if (flat.length === 0) return null;
	const last = flat.length - 1;
	const at = active === null ? -1 : flat.findIndex((option) => optionKey(option) === active);
	let to: number;
	if (move === 'first') to = 0;
	else if (move === 'last') to = last;
	else if (move === 'next') to = at < 0 || at >= last ? 0 : at + 1;
	else to = at <= 0 ? last : at - 1;
	return optionKey(flat[to]);
}

/**
 * The id an option carries in the page, under `prefix`: letters, digits and
 * the hyphen kept, every other character written as its code between
 * underscores, so two options never share an id and one option keeps its id.
 */
export function optionDomId(prefix: string, option: Listed): string {
	let encoded = '';
	for (const character of optionKey(option)) {
		encoded += /[A-Za-z0-9-]/.test(character) ? character : `_${character.codePointAt(0)?.toString(16)}_`;
	}
	return `${prefix}-option-${encoded}`;
}

/** What the status line is read from. */
export interface StatusState {
	/** The words typed, trimmed. */
	query: string;
	/** How many options are listed. */
	count: number;
	/** Whether the chats are still being searched for these words. */
	asking: boolean;
	/** Whether that search has taken long enough to say so. */
	slow: boolean;
	/** Why the chats could not be searched, or null. */
	error: string | null;
	/** Why the option the reader tried to open cannot run, or null. */
	notice: string | null;
}

function results(count: number): string {
	return `${count} ${count === 1 ? 'result' : 'results'}`;
}

/** The line the palette's status region reads. */
export function statusFor(state: StatusState): string {
	if (state.notice) return state.notice;
	if (!state.query) return '';
	if (state.error) return `${results(state.count)}. The chats could not be searched: ${state.error}`;
	if (state.asking) return state.slow ? `${results(state.count)}, still searching the chats` : '';
	if (state.count === 0) return `Nothing matches "${state.query}".`;
	return results(state.count);
}

/** How long a search of the chats runs before the status line says so, in milliseconds. */
export const SLOW_MS = 800;

/** How long a line stands still before it is spoken, in milliseconds. */
export const SETTLE_MS = 400;

/** Runs `task` after `ms`; returns what cancels it. */
export type Timer = (task: () => void, ms: number) => () => void;

export interface StatusLine {
	/** The palette's state now; the line follows it at its own pace. */
	update(state: Omit<StatusState, 'slow'>): void;
	/** Cancels what is waiting; nothing more is spoken. */
	close(): void;
}

export interface StatusLineOptions {
	/** The timer (setTimeout by default). */
	schedule?: Timer;
	/** Speak every line at once: a page rendered on the server has no time to wait. */
	immediate?: boolean;
}

function wait(task: () => void, ms: number): () => void {
	const timer = setTimeout(task, ms);
	return () => clearTimeout(timer);
}

/**
 * The palette's status line, paced: `speak` is called with each line to
 * read out, once it has stood still for SETTLE_MS (at once when it is empty
 * or a notice); the chats' search turns slow once it has run for SLOW_MS.
 */
export function createStatusLine(speak: (line: string) => void, options: StatusLineOptions = {}): StatusLine {
	const schedule = options.schedule ?? wait;
	const immediate = options.immediate === true;
	let state: Omit<StatusState, 'slow'> | null = null;
	let slow = false;
	let askedFor: string | null = null;
	let spoken: string | null = null;
	let cancelSlow: (() => void) | null = null;
	let cancelSettle: (() => void) | null = null;
	let closed = false;

	function say(line: string) {
		if (closed || line === spoken) return;
		spoken = line;
		speak(line);
	}

	function render() {
		if (!state) return;
		const line = statusFor({ ...state, slow });
		if (cancelSettle) cancelSettle();
		cancelSettle = null;
		if (immediate || line === '' || state.notice) {
			say(line);
			return;
		}
		cancelSettle = schedule(() => {
			cancelSettle = null;
			say(line);
		}, SETTLE_MS);
	}

	return {
		update(next) {
			if (closed) return;
			state = next;
			const asking = next.asking ? next.query : null;
			if (asking !== askedFor) {
				askedFor = asking;
				slow = false;
				if (cancelSlow) cancelSlow();
				cancelSlow = null;
				if (asking !== null && !immediate) {
					cancelSlow = schedule(() => {
						cancelSlow = null;
						slow = true;
						render();
					}, SLOW_MS);
				}
			}
			render();
		},
		close() {
			closed = true;
			if (cancelSlow) cancelSlow();
			if (cancelSettle) cancelSettle();
			cancelSlow = null;
			cancelSettle = null;
		}
	};
}
