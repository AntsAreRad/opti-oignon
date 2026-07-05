/**
 * Poll scheduler for the sandbox file list.
 *
 * Decides, after every fetch outcome, whether polling continues and how
 * long to wait. A 404 is terminal for a given session id: 'absent' when
 * no listing ever succeeded (the sandbox was never created, or it lives
 * under another id), 'expired' when the session was alive earlier and
 * is now gone. Transient failures (network, 5xx) back off exponentially
 * up to a hard cap; a success resets the ladder. Switching to another
 * session id resets the whole machine and requests an immediate fetch.
 *
 * Dependency-free on purpose: the state machine is exercised directly
 * under Node without a bundler, and the Svelte component only schedules
 * timers around the decisions it returns.
 */

export type PollState = 'idle' | 'live' | 'absent' | 'expired' | 'backoff';

export interface PollDecision {
	/** The lifecycle state after this outcome. */
	state: PollState;
	/** Delay before the next fetch in ms; null means polling stops. */
	nextDelayMs: number | null;
}

export interface PollerOptions {
	/** Steady cadence between successful listings (default 5000 ms). */
	baseDelayMs?: number;
	/** Hard cap for the transient-failure backoff (default 60000 ms). */
	maxDelayMs?: number;
}

export class SandboxFilesPoller {
	private state: PollState = 'idle';
	private sawAlive = false;
	private delay: number;
	private readonly base: number;
	private readonly max: number;

	constructor(options: PollerOptions = {}) {
		this.base = options.baseDelayMs ?? 5000;
		this.max = options.maxDelayMs ?? 60000;
		this.delay = this.base;
	}

	/** The current lifecycle state. */
	get current(): PollState {
		return this.state;
	}

	/** Whether at least one listing succeeded for the current id. */
	get everListed(): boolean {
		return this.sawAlive;
	}

	/** A listing succeeded: steady cadence, backoff ladder reset. */
	onSuccess(): PollDecision {
		this.sawAlive = true;
		this.state = 'live';
		this.delay = this.base;
		return { state: this.state, nextDelayMs: this.base };
	}

	/**
	 * The files API answered 404. Terminal for this id: 'absent' before
	 * any successful listing (never born), 'expired' after one
	 * (destroyed). Polling stops; a session change re-arms the machine.
	 */
	onNotFound(): PollDecision {
		this.state = this.sawAlive ? 'expired' : 'absent';
		return { state: this.state, nextDelayMs: null };
	}

	/** A transient failure: exponential backoff capped at maxDelayMs. */
	onTransientError(): PollDecision {
		this.state = 'backoff';
		const wait = this.delay;
		this.delay = Math.min(this.delay * 2, this.max);
		return { state: this.state, nextDelayMs: wait };
	}

	/** The component now targets another session id: fresh machine. */
	onSessionChange(): PollDecision {
		this.state = 'idle';
		this.sawAlive = false;
		this.delay = this.base;
		return { state: this.state, nextDelayMs: 0 };
	}
}
