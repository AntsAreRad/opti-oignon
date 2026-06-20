/**
 * Typed API client for the note selection-action surface (N.3).
 *
 * From a note selection the user runs one local action -- fact-check, develop,
 * summarize, rewrite, make-checklist -- or the Daily-only fact-check-with-web,
 * over the single per-user endpoint POST /api/notes/actions/run. The selected
 * text is wrapped as untrusted context by the backend (note_actions); this
 * client never interprets it. The action values mirror the backend ACTION_*
 * string constants exactly.
 *
 * The runner never raises: every outcome (ok / refused / a clean failure)
 * crosses the wire as a structured NoteActionResult with HTTP 200. A web action
 * outside Daily returns a structured refusal (refused=true), never a silent
 * local downgrade.
 */

import { apiPost } from './client';

/** The selection actions, matching the backend ACTION_* constants. */
export type NoteActionKind =
	| 'fact_check'
	| 'fact_check_web'
	| 'develop'
	| 'summarize'
	| 'rewrite'
	| 'make_checklist';

/** A selection action's wire value, label, and whether it needs web egress. */
export interface NoteActionDef {
	kind: NoteActionKind;
	/** Button label (active voice, sentence case). */
	label: string;
	/** True for the Daily-only web action; refused outside Daily. */
	requiresWeb: boolean;
}

/** The selection actions offered in the UI, in display order. */
export const NOTE_ACTIONS: NoteActionDef[] = [
	{ kind: 'fact_check', label: 'Fact-check', requiresWeb: false },
	{ kind: 'fact_check_web', label: 'Fact-check with web', requiresWeb: true },
	{ kind: 'develop', label: 'Develop', requiresWeb: false },
	{ kind: 'summarize', label: 'Summarize', requiresWeb: false },
	{ kind: 'rewrite', label: 'Rewrite', requiresWeb: false },
	{ kind: 'make_checklist', label: 'Make checklist', requiresWeb: false }
];

/** The structured outcome of a selection action (mirrors NoteActionResultSchema).
 * ok true carries the model text; refused true marks the Daily-only web-egress
 * refusal; any other failure is ok false with a reason and refused false. */
export interface NoteActionResult {
	action: string;
	ok: boolean;
	text: string;
	refused: boolean;
	reason: string;
}

/** Run one selection action over the selected note text with the chosen model. */
export async function runNoteAction(input: {
	action: NoteActionKind | string;
	selection: string;
	model?: string;
}): Promise<NoteActionResult> {
	return apiPost<NoteActionResult>('/api/notes/actions/run', {
		action: input.action,
		selection: input.selection,
		model: input.model ?? ''
	});
}
