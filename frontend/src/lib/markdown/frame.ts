/**
 * At most one lex per animation frame while a reply streams, and no lex at
 * all once lexing a reply proved slow.
 *
 * A streaming reply grows by many small updates between two frames; lexing
 * each one would cost a full lex per update for a picture only the next
 * frame shows. push() records the latest source and asks for one frame; the
 * frame lexes that latest source once and hands the tokens over. A source
 * equal to the one last lexed asks for nothing. now() lexes at once and
 * drops a pending frame (a first render, or the end of a stream); cancel()
 * drops it (the component goes away).
 *
 * The frame clock is injected. By default it is the browser's; where there
 * is none (server rendering), every push lexes at once, since nothing would
 * ever run the frame.
 *
 * guardSlowLex wraps the lexer in a timer. The tree's own estimate keeps the
 * lexes it knows to be slow from running at all; a lex that runs anyway and
 * takes longer than the limit gives no tokens (the reply is shown as plain
 * text), and from then on this reply is not lexed again: not for the rest
 * of its stream, whose later sources hold the slow one, and not when it is
 * shown again, since its latest source is remembered (the last
 * SLOW_REMEMBERED replies). So a reply costs at most one slow lex.
 *
 * It imports nothing, so it runs under Node's type stripping.
 */

export interface Frames {
	request: (callback: () => void) => number;
	cancel: (handle: number) => void;
}

export interface FrameLexer<T> {
	/** Lexes the latest source at the next frame, once. */
	push(source: string): void;
	/** Lexes now, drops a pending frame, and returns the tokens. */
	now(source: string): T;
	/** Drops a pending frame. */
	cancel(): void;
}

interface FrameScope {
	requestAnimationFrame?: (callback: () => void) => number;
	cancelAnimationFrame?: (handle: number) => void;
}

/** The browser's frame clock, or null where there is none. */
export function browserFrames(): Frames | null {
	const scope = globalThis as FrameScope;
	const request = scope.requestAnimationFrame;
	const cancel = scope.cancelAnimationFrame;
	if (typeof request !== 'function' || typeof cancel !== 'function') return null;
	return {
		request: (callback) => request.call(globalThis, callback),
		cancel: (handle) => cancel.call(globalThis, handle)
	};
}

export function createFrameLexer<T>(
	lex: (source: string) => T,
	deliver: (tokens: T) => void,
	frames: Frames | null = browserFrames()
): FrameLexer<T> {
	let handle: number | null = null;
	let latest = '';
	let lexed: string | null = null;

	const run = (source: string): T => {
		lexed = source;
		const tokens = lex(source);
		deliver(tokens);
		return tokens;
	};

	const drop = (): void => {
		if (handle !== null && frames !== null) frames.cancel(handle);
		handle = null;
	};

	return {
		push(source: string): void {
			latest = source;
			if (handle !== null || source === lexed) return;
			if (frames === null) {
				run(source);
				return;
			}
			handle = frames.request(() => {
				handle = null;
				run(latest);
			});
		},
		now(source: string): T {
			drop();
			latest = source;
			return run(source);
		},
		cancel(): void {
			drop();
		}
	};
}

/** A lex slower than this, in milliseconds, is not run again for its reply. */
export const SLOW_LEX_MS = 100;

/** How many slow replies are remembered, the most recent kept. */
export const SLOW_REMEMBERED = 16;

const SLOW_SOURCES: Set<string> = new Set();

export interface SlowGuard {
	/** The clock, in milliseconds; by default the page's. */
	now?: () => number;
	/** The limit, in milliseconds. */
	slowMs?: number;
	/** The sources known to lex slowly; by default the module's own. */
	memory?: Set<string>;
}

function pageClock(): number {
	return globalThis.performance.now();
}

/**
 * The lexer, timed: its tokens, or null when this reply lexes slowly. One
 * guard per reply. A null from the lexer itself (its own refusal) passes
 * through and does not stick.
 */
export function guardSlowLex<T>(
	lex: (source: string) => T | null,
	guard: SlowGuard = {}
): (source: string) => T | null {
	const now = guard.now ?? pageClock;
	const slowMs = guard.slowMs ?? SLOW_LEX_MS;
	const memory = guard.memory ?? SLOW_SOURCES;
	let slow = false;
	let mine: string | null = null;

	const remember = (source: string): void => {
		if (mine !== null) memory.delete(mine);
		memory.delete(source);
		memory.add(source);
		mine = source;
		for (const oldest of memory) {
			if (memory.size <= SLOW_REMEMBERED) break;
			memory.delete(oldest);
		}
	};

	return (source: string): T | null => {
		if (slow || memory.has(source)) {
			slow = true;
			remember(source);
			return null;
		}
		const start = now();
		const tokens = lex(source);
		if (now() - start > slowMs) {
			slow = true;
			remember(source);
			return null;
		}
		return tokens;
	};
}
