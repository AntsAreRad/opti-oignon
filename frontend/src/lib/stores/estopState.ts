/**
 * The emergency stop's state, and the two rules its reads follow.
 *
 * `afterRead` takes what one read of the status brought: the server's
 * answer, or null when the status could not be read. An answer is taken as
 * the server gives it; a status that could not be read makes the stop
 * unknown again (`available` null), so the control is enabled and a request
 * that fails says so, rather than keeping a last answer that may be stale (a
 * stop the server once said it could not make is not left disabled while
 * nothing is known). What else was known (stopped, a request on its way, the
 * last error and announcement) is kept.
 *
 * `oneAtATime` wraps a read so that there is never more than one on its way:
 * whoever asks while a read is open joins it, and once it has answered or
 * failed the next ask reads again.
 *
 * Pure functions, with no dependency: the estop store (lib/stores/estop.ts)
 * reads through them, and they run under Node as they are.
 */

export interface EstopState {
	/** True when the server can stop, false when it says it cannot, null while unknown. */
	available: boolean | null;
	/** Whether the machine is stopped. */
	stopped: boolean;
	/** A request is on its way. */
	busy: boolean;
	/** What went wrong with the last request, or with its steps. */
	error: string;
	/** The last change, for the polite live region. */
	announce: string;
}

/** What one read of the stop's status answers. */
export interface EstopStatusRead {
	available?: boolean;
	stopped?: boolean;
}

/** The state after one read: the server's answer, or unknown when the status could not be read. */
export function afterRead(state: EstopState, status: EstopStatusRead | null): EstopState {
	if (status === null) return { ...state, available: null };
	return { ...state, available: !!status.available, stopped: !!status.stopped };
}

/** A read that is never on its way twice: an ask while one is open joins it. */
export function oneAtATime<T>(read: () => Promise<T>): () => Promise<T> {
	let open: Promise<T> | null = null;
	return () => {
		if (open) return open;
		open = read().finally(() => {
			open = null;
		});
		return open;
	};
}
