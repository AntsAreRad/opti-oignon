/**
 * The command palette's chats, asked of the server as the reader types.
 *
 * The server searches titles and messages (GET /api/conversations with `q`);
 * the palette sends the words, trimmed, with a limit, and never searches the
 * conversations on its own. The words go once the reader pauses
 * (SOURCE_DELAY_MS), so a burst of keys asks once, for its last words. A new
 * search aborts the request before it, and an answer that is not the latest
 * one asked is dropped, in whatever order the answers come back: the chats
 * shown always answer the words they are shown for. Closing the palette
 * aborts the request in flight, cancels one still waiting for the pause,
 * and drops any answer still to come. No words, no request.
 *
 * Its state says what it shows and what it is waiting for: `query`, the
 * words the `hits` answer; `asking`, the words asked and not answered yet;
 * `error`, why the latest search failed. A request aborted because a newer
 * one replaced it, or because the palette closed, is no failure.
 *
 * `followPalette` keeps a source in step with the palette: while it is
 * open, each change of its words is asked, trimmed, once; when it closes,
 * the source is closed; opened again, it asks again.
 *
 * Pure: the fetcher is handed in (the palette hands it the conversations
 * API, which carries the signal down to fetch), and so is the timer, so it
 * runs under Node as it is.
 */

/** How many conversations one search asks for. */
export const SOURCE_LIMIT = 20;

/** How long the reader pauses before the words are sent, in milliseconds. */
export const SOURCE_DELAY_MS = 150;

/** The request a search sends. */
export interface ConversationQuery {
	q: string;
	limit: number;
}

export type ConversationFetcher<T> = (query: ConversationQuery, signal: AbortSignal) => Promise<T[]>;

/** Runs `task` after `ms`; returns what cancels it. */
export type Scheduler = (task: () => void, ms: number) => () => void;

export interface ConversationState<T> {
	/** The words the hits answer; '' before any answer. */
	query: string;
	hits: T[];
	/** The words asked and not answered yet, or null. */
	asking: string | null;
	/** Why the latest search failed, or null. */
	error: string | null;
}

export interface ConversationSource {
	/** Asks for `words` once the reader pauses; no words shows nothing. */
	search(words: string): void;
	/** Aborts what is in flight and drops what is still to come. */
	close(): void;
}

export interface SourceOptions {
	delay?: number;
	schedule?: Scheduler;
}

function later(task: () => void, ms: number): () => void {
	const timer = setTimeout(task, ms);
	return () => clearTimeout(timer);
}

function reasonOf(error: unknown): string {
	if (error instanceof Error && error.message) return error.message;
	if (typeof error === 'string' && error) return error;
	return 'The search failed';
}

/**
 * A source that asks `fetcher` for the chats matching the words it is given,
 * and tells `onChange` each state it goes through.
 */
export function createConversationSource<T>(
	fetcher: ConversationFetcher<T>,
	onChange: (state: ConversationState<T>) => void,
	options: SourceOptions = {}
): ConversationSource {
	const delay = typeof options.delay === 'number' && options.delay >= 0 ? options.delay : SOURCE_DELAY_MS;
	const schedule = options.schedule ?? later;
	let state: ConversationState<T> = { query: '', hits: [], asking: null, error: null };
	// The latest search: an answer to any other is dropped.
	let serial = 0;
	let cancelWait: (() => void) | null = null;
	let inFlight: AbortController | null = null;

	function emit(next: ConversationState<T>) {
		state = next;
		onChange(state);
	}

	function stop() {
		serial += 1;
		if (cancelWait) cancelWait();
		cancelWait = null;
		if (inFlight) inFlight.abort();
		inFlight = null;
	}

	function send(words: string, mine: number) {
		if (mine !== serial) return;
		cancelWait = null;
		const control = new AbortController();
		inFlight = control;
		let answer: Promise<T[]>;
		try {
			answer = Promise.resolve(fetcher({ q: words, limit: SOURCE_LIMIT }, control.signal));
		} catch (error) {
			answer = Promise.reject(error);
		}
		answer.then(
			(hits) => {
				if (mine !== serial) return;
				inFlight = null;
				emit({ query: words, hits: Array.isArray(hits) ? hits : [], asking: null, error: null });
			},
			(error: unknown) => {
				if (mine !== serial || control.signal.aborted) return;
				inFlight = null;
				emit({ query: words, hits: [], asking: null, error: reasonOf(error) });
			}
		);
	}

	return {
		search(words: string) {
			const trimmed = typeof words === 'string' ? words.trim() : '';
			stop();
			const mine = serial;
			if (!trimmed) {
				emit({ query: '', hits: [], asking: null, error: null });
				return;
			}
			emit({ ...state, asking: trimmed, error: null });
			cancelWait = schedule(() => send(trimmed, mine), delay);
		},
		close() {
			stop();
			emit({ query: '', hits: [], asking: null, error: null });
		}
	};
}

/**
 * What keeps a conversation source in step with the palette: call it with
 * whether the palette is open and the words in its field, each time either
 * changes. Open, a change of the words (trimmed) is handed to `ask`, once
 * per change; the opening itself asks the words it opens with. Closing
 * calls `close`, once; opened again, the words are asked again.
 */
export function followPalette(
	ask: (words: string) => void,
	close: () => void
): (open: boolean, words: string) => void {
	let isOpen = false;
	let asked: string | null = null;
	return (open, words) => {
		if (!open) {
			if (isOpen) close();
			isOpen = false;
			asked = null;
			return;
		}
		isOpen = true;
		const trimmed = typeof words === 'string' ? words.trim() : '';
		if (trimmed === asked) return;
		asked = trimmed;
		ask(trimmed);
	};
}
