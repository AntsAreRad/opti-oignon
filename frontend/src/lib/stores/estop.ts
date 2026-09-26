/**
 * The emergency stop's state, and its one poller.
 *
 * The stop makes the machine quiet (it cancels generations and runs,
 * unloads models, destroys sandboxes and stops the sync node) and resumes
 * without a ceremony, sign-in still required. This store is the one place
 * that reads its status and the one place that engages or resumes it; the
 * stop control (components/layout/StopAllButton.svelte) draws what it holds
 * wherever the shell mounts it, so every copy on screen shows one state.
 *
 * `available` is true when the server can stop, false when it says it
 * cannot, and null while nothing has been read or the last read could not
 * reach the status: the control is enabled then, and a request that fails
 * says so (lib/stores/estopState.ts holds that rule). The status is read
 * while something shows the stop, every ten seconds, one request at a time,
 * whoever asks: an ask while a read is open joins it, and the next read is
 * scheduled when the last one has answered.
 */

import { get, writable } from 'svelte/store';
import {
	engageEmergencyStop,
	getEmergencyStopStatus,
	resumeFromEmergencyStop,
	type EmergencyActionResult
} from '$lib/api/estop';
import { refreshSecurityMode } from '$lib/stores/securityMode';
import { afterRead, oneAtATime, type EstopState } from './estopState';

export type { EstopState } from './estopState';

const READ_EVERY_MS = 10_000;

const INITIAL: EstopState = {
	available: null,
	stopped: false,
	busy: false,
	error: '',
	announce: ''
};

let timer: ReturnType<typeof setTimeout> | null = null;
let reading = false;

/** The stop's state. Subscribing in a browser starts the reads; the last unsubscribe stops them. */
export const estop = writable<EstopState>(INITIAL, () => {
	if (typeof document === 'undefined') return undefined;
	reading = true;
	void refreshEstop();
	return () => {
		reading = false;
		if (timer !== null) {
			clearTimeout(timer);
			timer = null;
		}
	};
});

function scheduleNextRead(): void {
	if (!reading) return;
	if (timer !== null) clearTimeout(timer);
	timer = setTimeout(() => {
		timer = null;
		void refreshEstop();
	}, READ_EVERY_MS);
}

/** Reads the stop's status now, or joins the read on its way; a status that cannot be read is unknown. */
export const refreshEstop = oneAtATime(async (): Promise<void> => {
	try {
		const status = await getEmergencyStopStatus();
		estop.update((state) => afterRead(state, status));
	} catch {
		estop.update((state) => afterRead(state, null));
	} finally {
		scheduleNextRead();
	}
});

function stepErrors(prefix: string, result: EmergencyActionResult): string {
	const failed = result.failed_steps ?? [];
	return failed.length > 0 ? `${prefix} with step errors: ${failed.join(', ')}` : '';
}

/**
 * Engages the stop, and with `dropToBulbe` switches the machine to Bulbe as
 * well. Returns whether the request was answered.
 */
export async function engageStop(dropToBulbe: boolean): Promise<boolean> {
	if (get(estop).busy) return false;
	estop.update((state) => ({ ...state, busy: true, error: '' }));
	try {
		const result = await engageEmergencyStop(dropToBulbe);
		estop.update((state) => ({
			...state,
			available: true,
			busy: false,
			stopped: !!result.stopped,
			announce: 'Emergency stop engaged',
			error: stepErrors('Stopped', result)
		}));
		if (dropToBulbe) void refreshSecurityMode();
		return true;
	} catch {
		estop.update((state) => ({ ...state, busy: false, error: 'The emergency stop request failed' }));
		return false;
	}
}

/** Resumes from the stop. Returns whether the request was answered. */
export async function resumeStop(): Promise<boolean> {
	if (get(estop).busy) return false;
	estop.update((state) => ({ ...state, busy: true, error: '' }));
	try {
		const result = await resumeFromEmergencyStop();
		estop.update((state) => ({
			...state,
			busy: false,
			stopped: !!result.stopped,
			announce: 'Resumed',
			error: stepErrors('Resumed', result)
		}));
		return true;
	} catch {
		estop.update((state) => ({ ...state, busy: false, error: 'The resume request failed' }));
		return false;
	}
}
