/**
 * How the agent panel tells the end of a run: the reason a loop event ends
 * the run with, and the words the panel says of it.
 *
 * The loop ends a run with 'done' (the model answered), 'error' (the model
 * call failed), or an 'aborted' event that names its reason: 'doom_loop'
 * when the model kept repeating the same call, 'proposal_cap' when the run
 * tried to propose more writes for review than one run may. A run that tries
 * that many writes is likelier to be following words it read in a page or a
 * file than its task; what it proposed before waits in the review. This
 * module holds no state and imports nothing.
 */

/** A loop event as the panel receives it: its kind, and its data. */
export interface EndingEvent {
	kind: string;
	data?: Record<string, unknown>;
}

const LABELS: Record<string, string> = {
	done: 'Done',
	error: 'Stopped: error',
	cancelled: 'Stopped: cancelled',
	doom_loop: 'Stopped: repeating',
	proposal_cap: 'Stopped: too many writes',
	max_rounds: 'Stopped: round limit',
};

const SENTENCES: Record<string, string> = {
	done: 'Done.',
	error: 'Stopped: the model call failed.',
	cancelled: 'Stopped: you cancelled the run.',
	doom_loop: 'Stopped: the run kept repeating the same call.',
	proposal_cap:
		'Stopped: the run tried to propose more writes for your review than one run may. A task rarely needs ' +
		'that many; an instruction read in a page or a file often asks for them. What it proposed waits in your ' +
		'review, where you can decline it.',
	max_rounds: 'Stopped: the run reached its round limit.',
};

/**
 * The reason this event ends the run with -- 'done', 'error', or the reason
 * an 'aborted' event names -- or null; a subtask's event (it names its task)
 * ends only the subtask.
 */
export function endingReason(event: EndingEvent): string | null {
	if (event.data?.task !== undefined) return null;
	if (event.kind === 'done') return 'done';
	if (event.kind === 'error') return 'error';
	if (event.kind === 'aborted') {
		const reason = event.data?.reason;
		return typeof reason === 'string' && reason ? reason : 'aborted';
	}
	return null;
}

/** The panel's short status for a run that ended for `reason`. */
export function stopLabel(reason: string): string {
	return LABELS[reason] ?? `Stopped (${reason})`;
}

/** What the panel says of a run that ended for `reason`, in the run's event list. */
export function stopSentence(reason: string): string {
	return SENTENCES[reason] ?? `Stopped (${reason}).`;
}
