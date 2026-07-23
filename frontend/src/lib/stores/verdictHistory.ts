/**
 * Verdict-history store (the verdict-history affordance).
 *
 * A session-local, in-memory history of the most recent verification verdicts
 * the user has run. It is fed by a verifier surface (the AnswerVerifier on
 * /verify-answer this lot; other verifiers may feed it later) and read by the
 * VerdictHistory sub-component. The history is capped to the most recent
 * MAX_VERDICT_HISTORY entries and is never persisted: it lives only for the
 * lifetime of the page, so closing or reloading clears it. There is no backend
 * coupling and no mode gate -- this is a read-only display of verdicts the
 * server already returned.
 */

import { writable } from 'svelte/store';

/** The most recent verdicts kept in the session history (oldest dropped). */
export const MAX_VERDICT_HISTORY = 20;

/**
 * One recorded verdict in the session history.
 *
 * The verdict, ok and summary are taken verbatim from a server-returned result;
 * id and at are stamped at record time. surface labels which verifier produced
 * the verdict (e.g. 'answer') so a future shared history can distinguish them.
 */
export interface VerdictEntry {
	/** A unique, monotonically increasing id stamped at record time. */
	id: string;
	/** Which verifier surface produced the verdict (e.g. 'answer'). */
	surface: string;
	/** The mapped aggregate verdict: supported / unsupported / uncertain. */
	verdict: string;
	/** Whether the verification completed cleanly (the aggregate ok flag). */
	ok: boolean;
	/** A short human-readable summary (e.g. the number of pairs checked). */
	summary: string;
	/** The epoch-millisecond timestamp the verdict was recorded. */
	at: number;
}

/** The input a caller supplies; id and at are stamped by recordVerdict. */
export type VerdictEntryInput = Pick<VerdictEntry, 'surface' | 'verdict' | 'ok' | 'summary'>;

/** The reactive session history, newest first. */
export const verdictHistory = writable<VerdictEntry[]>([]);

// A per-session counter so two verdicts recorded in the same millisecond still
// get distinct ids (used only for the keyed each-list in the sub-component).
let seq = 0;

/**
 * Record a verdict at the head of the session history.
 *
 * The entry is stamped with a unique id and the current time, prepended so the
 * newest verdict is first, and the list is truncated to MAX_VERDICT_HISTORY.
 * Returns the stamped entry.
 */
export function recordVerdict(input: VerdictEntryInput): VerdictEntry {
	const entry: VerdictEntry = {
		id: String(Date.now()) + '-' + String(seq++),
		surface: input.surface,
		verdict: input.verdict,
		ok: input.ok,
		summary: input.summary,
		at: Date.now()
	};
	verdictHistory.update((list) => [entry, ...list].slice(0, MAX_VERDICT_HISTORY));
	return entry;
}

/** Clear the session history. */
export function clearVerdicts(): void {
	verdictHistory.set([]);
}
