/**
 * A server switch: a setting the server holds, shown as the server confirmed
 * it, never as it was asked for.
 *
 * - Its state is unknown (null) until the server has been read.
 * - A toggle asks for the opposite of the confirmed state. While it is
 *   pending the shown state does not move, and a second toggle is refused
 *   without a request.
 * - A success adopts the state the server answers, which may not be the one
 *   asked for; when it is not, the answer is named.
 * - A refusal (the server answered with a client error status, 4xx) keeps
 *   the state and names the refusal: the server refused before acting.
 * - Anything else leaves the outcome unknown, and the state is read again
 *   (unknown if that read fails too): a server error (5xx), which some
 *   routes answer after the change has landed; a failure to reach the
 *   server; an unknown failure; an answer without a state.
 * - While the state is unknown, its mirrors are told to forget it.
 *
 * It is a Svelte store (subscribe), with no dependency: the API layer is
 * reached through the read and write functions it is given, so it runs
 * under Node as it is.
 */

export interface SwitchState {
	/** The state the server confirmed; null while it is unknown. */
	value: boolean | null;
	/** A read or a write is in flight. */
	pending: boolean;
	/** Why the last read or write did not give a confirmed state; null otherwise. */
	error: string | null;
}

/** What a toggle did: adopted an answer, kept the state, read it again, or nothing. */
export type ToggleOutcome = 'adopted' | 'kept' | 'reread' | 'refused';

export interface ServerSwitchOptions {
	/** The switch's name, as its messages say it. */
	label: string;
	/** Reads the server's state. */
	read: () => Promise<boolean>;
	/** Asks the server for a state and returns the state it confirms. */
	write: (next: boolean) => Promise<boolean>;
	/** Called with every state the server confirms. */
	adopt?: (value: boolean) => void;
	/** Called when the state becomes unknown, so a mirror of it forgets it. */
	forget?: () => void;
}

export interface ServerSwitch {
	subscribe(run: (state: SwitchState) => void): () => void;
	/** The state now. */
	current(): SwitchState;
	/** Reads the server's state; does nothing while a read or write is pending. */
	load(): Promise<SwitchState>;
	/** Asks for the opposite of the confirmed state. */
	toggle(): Promise<ToggleOutcome>;
}

/**
 * An HTTP error answer, when the error is one: the server answered with an
 * error status. Its detail is the server's own reason when the error carries
 * it (the API client keeps it as serverDetail), else the error's text.
 */
export function httpRefusal(error: unknown): { status: number; detail: string } | null {
	if (!error || typeof error !== 'object') return null;
	const failure = error as {
		status?: unknown;
		detail?: unknown;
		serverDetail?: unknown;
		message?: unknown;
		isNetworkError?: unknown;
	};
	if (failure.isNetworkError === true) return null;
	if (typeof failure.status !== 'number' || !Number.isInteger(failure.status) || failure.status < 400) {
		return null;
	}
	const detail =
		typeof failure.serverDetail === 'string' && failure.serverDetail
			? failure.serverDetail
			: typeof failure.detail === 'string' && failure.detail
				? failure.detail
				: typeof failure.message === 'string'
					? failure.message
					: '';
	return { status: failure.status, detail };
}

/**
 * The aria-pressed of a switch's control: the confirmed state, or absent
 * while the state is unknown, so the control never announces a state the
 * server did not confirm.
 */
export function pressed(value: boolean | null): boolean | undefined {
	return value === null ? undefined : value;
}

function word(value: boolean | null): string {
	if (value === null) return 'unknown';
	return value ? 'on' : 'off';
}

function refusalText(refusal: { status: number; detail: string }): string {
	return refusal.detail ? `${refusal.status}: ${refusal.detail}` : String(refusal.status);
}

export function createServerSwitch(options: ServerSwitchOptions): ServerSwitch {
	const { label, read, write, adopt, forget } = options;
	let state: SwitchState = { value: null, pending: false, error: null };
	const listeners = new Set<(state: SwitchState) => void>();

	function update(next: Partial<SwitchState>): void {
		state = { ...state, ...next };
		for (const listener of [...listeners]) listener(state);
	}

	function confirmed(value: boolean, error: string | null = null): void {
		update({ value, pending: false, error });
		adopt?.(value);
	}

	function unknown(error: string): void {
		update({ value: null, pending: false, error });
		forget?.();
	}

	/** Reads the state again after an outcome that is not known. */
	async function reread(why: string): Promise<void> {
		try {
			const value = await read();
			if (typeof value !== 'boolean') throw new Error('the answer holds no state');
			confirmed(value, `${label}: ${why} The server says it is ${word(value)}.`);
		} catch {
			unknown(`${label}: ${why} Its state is unknown.`);
		}
	}

	async function load(): Promise<SwitchState> {
		if (state.pending) return state;
		update({ pending: true });
		try {
			const value = await read();
			if (typeof value !== 'boolean') throw new Error('the answer holds no state');
			confirmed(value);
		} catch (error) {
			const refusal = httpRefusal(error);
			unknown(
				refusal
					? `${label}: the server ${refusal.status < 500 ? 'refused to give' : 'failed to give'} its state (${refusalText(refusal)}).`
					: `${label}: the server could not be read. Its state is unknown.`
			);
		}
		return state;
	}

	async function toggle(): Promise<ToggleOutcome> {
		if (state.pending || state.value === null) return 'refused';
		const before = state.value;
		const asked = !before;
		update({ pending: true });
		let answer: unknown;
		try {
			answer = await write(asked);
		} catch (error) {
			const refusal = httpRefusal(error);
			if (refusal && refusal.status < 500) {
				update({
					pending: false,
					error: `${label}: the server refused the change (${refusalText(refusal)}). It is still ${word(before)}.`,
				});
				return 'kept';
			}
			await reread(
				refusal
					? `the server failed on the change (${refusalText(refusal)}), which may have landed.`
					: 'the change may not have reached the server.'
			);
			return 'reread';
		}
		if (typeof answer !== 'boolean') {
			await reread("the server's answer did not say its state.");
			return 'reread';
		}
		confirmed(
			answer,
			answer === asked ? null : `${label}: the server answered ${word(answer)}, not ${word(asked)}.`
		);
		return 'adopted';
	}

	return {
		subscribe(run) {
			listeners.add(run);
			run(state);
			return () => {
				listeners.delete(run);
			};
		},
		current: () => state,
		load,
		toggle,
	};
}
