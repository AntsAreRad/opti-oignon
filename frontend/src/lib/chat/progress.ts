/**
 * The chat loader's reducer: what a reply's stream has said so far.
 *
 * It folds the frames of one reply into the status line, the runs the
 * server executes and their steps, and what the status region announces.
 * Everything here is pure and clock-free: a function that needs the time
 * takes it as an argument, and no plant grows with it. A plant grows only
 * on a true fraction, read from a step's `progress` or from its child run's
 * finished count over a known total, and never decreases.
 *
 * Each step keeps the looks its plant was given, and when (the time handed
 * in): the pacing (`lib/chat/pacing.ts`) schedules what the plant shows from
 * them, and a run keeps when it opened.
 *
 * It knows no word. A line is a key of the word table
 * (`lib/chat/loaderWords.ts`) with its values; the caller renders it, and
 * hands `observe` the table's status mapper so a status the table does not
 * know changes nothing visible.
 */

import type { Arrival } from './pacing';

/** Thirty seconds without any frame, pings included: the server is silent. */
export const SILENCE_MS = 30000;
/** A line shows `Sending` only when nothing has come 200 ms after the send. */
export const SENDING_AFTER_MS = 200;
/** A burst of announcements speaks its last after this much calm. */
export const COALESCE_MS = 400;
/** At most one announcement per this many milliseconds. */
export const ANNOUNCE_GAP_MS = 1000;

export type StepState =
	| 'pending'
	| 'running'
	| 'done'
	| 'failed'
	| 'skipped'
	| 'cancelled'
	| 'not_run'
	| 'unknown';

/** The states the server sends; `unknown` is the client's own. */
export const STEP_STATES: readonly StepState[] = [
	'pending',
	'running',
	'done',
	'failed',
	'skipped',
	'cancelled',
	'not_run',
];

const FINAL: readonly StepState[] = ['done', 'failed', 'skipped', 'cancelled', 'not_run', 'unknown'];

/** A plant's stage: its growth, lowest first, and bare soil for a skipped step. */
export type Stage = 'seed' | 'crook' | 'flag' | 'leaf' | 'bulb' | 'soil';

const GROWTH: readonly Stage[] = ['seed', 'crook', 'flag', 'leaf', 'bulb'];

export interface Line {
	key: string;
	values?: Record<string, string | number>;
}

export interface StepProgress {
	done: number;
	total: number;
	unit: string;
}

export interface Step {
	index: number;
	label: string;
	state: StepState;
	stage: Stage;
	/** The highest stage the step reached while running. */
	reached: Stage;
	seq: number;
	progress: StepProgress | null;
	reason: string | null;
	ranAs: string | null;
	durationMs: number | null;
	stepType: string | null;
	/** Every look the plant was given (`state:stage`), in order, each when it came. */
	given: Arrival[];
}

export interface Run {
	id: string;
	kind: string;
	name: string;
	parent: { run: string; index: number } | null;
	total: number | null;
	steps: Step[];
	/** When its first frame came. */
	since: number;
}

export type End = 'done' | 'error' | 'stopped' | 'lost';

export interface LoaderState {
	startedAt: number;
	lastFrameAt: number;
	lineSince: number;
	seen: number[];
	runs: Run[];
	line: Line | null;
	/** The line a pending tool approval covers, back when it resolves. */
	held: Line | null;
	model: string | null;
	thinking: boolean;
	writing: boolean;
	stopRequested: boolean;
	stopSaid: boolean;
	end: End | null;
	durationMs: number | null;
	/** Whether `done` carried the steps' record. */
	doneSteps: boolean;
	silentSince: number | null;
	/** What the last transition announces. */
	said: Line[];
}

export interface Frame {
	type: string;
	content?: string;
	metadata?: Record<string, unknown>;
}

export type StatusMapper = (message: string) => Line | null;

export interface Summary {
	form: 'done' | 'stopped' | 'lost';
	kind: string;
	name: string;
	steps: number;
	at: number;
	counts: Record<string, number>;
	durationMs: number | null;
}

export interface Look {
	/** `stage`: the reached stage's drawing; `seed` under the soil; `soil` alone. */
	form: 'stage' | 'seed' | 'soil';
	ink: string | null;
	mark: 'tag' | null;
	pattern: 'solid' | 'dotted';
}

const noStatus: StatusMapper = () => null;

export function start(now: number): LoaderState {
	return {
		startedAt: now,
		lastFrameAt: now,
		lineSince: now,
		seen: [],
		runs: [],
		line: null,
		held: null,
		model: null,
		thinking: false,
		writing: false,
		stopRequested: false,
		stopSaid: false,
		end: null,
		durationMs: null,
		doneSteps: false,
		silentSince: null,
		said: [],
	};
}

function copy(state: LoaderState): LoaderState {
	return {
		...state,
		seen: [...state.seen],
		runs: state.runs.map((run) => ({
			...run,
			steps: run.steps.map((step) => ({ ...step, given: [...step.given] })),
		})),
		said: [],
	};
}

function setLine(state: LoaderState, line: Line, now: number): void {
	state.line = line;
	state.lineSince = now;
}

function text(value: unknown): string | null {
	return typeof value === 'string' && value !== '' ? value : null;
}

function count(value: unknown): number | null {
	return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : null;
}

function runOpen(state: LoaderState): boolean {
	return state.runs.some((run) => run.steps.some((step) => !FINAL.includes(step.state)));
}

function topRuns(state: LoaderState): Run[] {
	return state.runs.filter((run) => run.parent === null);
}

function finished(run: Run): number {
	return run.steps.filter((step) => FINAL.includes(step.state)).length;
}

/** A running step's true fraction, or null when there is none. */
export function fractionOf(state: LoaderState, run: Run, step: Step): number | null {
	if (step.progress && step.progress.total >= 1) {
		return Math.min(1, Math.max(0, step.progress.done / step.progress.total));
	}
	const child = state.runs.find(
		(other) => other.parent !== null && other.parent.run === run.id && other.parent.index === step.index,
	);
	if (child && child.total !== null && child.total >= 1) {
		return Math.min(1, finished(child) / child.total);
	}
	return null;
}

/** The stage a state and a fraction draw, by the decision's table. */
export function stageFor(state: StepState, fraction: number | null): Stage {
	if (state === 'pending' || state === 'not_run') return 'seed';
	if (state === 'done') return 'bulb';
	if (state === 'skipped') return 'soil';
	if (fraction === null || fraction <= 0) return 'crook';
	return fraction < 0.5 ? 'flag' : 'leaf';
}

function higher(a: Stage, b: Stage): Stage {
	return GROWTH.indexOf(b) > GROWTH.indexOf(a) ? b : a;
}

function grow(state: LoaderState): void {
	for (const run of state.runs) {
		for (const step of run.steps) {
			if (step.state === 'running') {
				step.reached = higher(step.reached, stageFor('running', fractionOf(state, run, step)));
				step.stage = step.reached;
			} else if (step.state === 'failed' || step.state === 'cancelled' || step.state === 'unknown') {
				step.stage = step.reached;
			} else {
				step.stage = stageFor(step.state, null);
			}
		}
	}
}

/** The look a step's plant is given: its state and its stage. */
export function lookOf(step: Step): string {
	return `${step.state}:${step.stage}`;
}

/** Each step whose look changed keeps the new one, and when it came. */
function stamp(state: LoaderState, now: number): void {
	for (const run of state.runs) {
		for (const step of run.steps) {
			const look = lookOf(step);
			const last = step.given[step.given.length - 1];
			if (!last || last.look !== look) step.given.push({ at: now, look, end: FINAL.includes(step.state) });
		}
	}
}

function announceStep(state: LoaderState, run: Run, step: Step, before: StepState | null): void {
	if (run.parent !== null) return;
	const where: Record<string, string | number> = { i: step.index + 1, label: step.label };
	if (run.total !== null) where.n = run.total;
	if (step.state === 'running' && before !== 'running') {
		state.said.push({ key: run.total !== null ? 'step_started' : 'step_started_open', values: where });
	} else if (step.state === 'failed') {
		where.reason = step.reason ?? '';
		state.said.push({ key: run.total !== null ? 'step_failed' : 'step_failed_open', values: where });
	} else if (step.state === 'cancelled' && !state.stopSaid) {
		state.stopSaid = true;
		state.said.push(step.reason ? { key: 'stopped_because', values: { reason: step.reason } } : { key: 'stopped' });
	}
}

function applyStep(state: LoaderState, meta: Record<string, unknown>, announce: boolean, now: number): void {
	const seq = count(meta.seq);
	const stateName = meta.state as StepState;
	const id = text(meta.run);
	const index = count(meta.index);
	if (seq === null || seq < 1 || id === null || index === null) return;
	if (!STEP_STATES.includes(stateName)) return;
	if (state.seen.includes(seq)) return;
	state.seen.push(seq);

	let run = state.runs.find((r) => r.id === id);
	if (!run) {
		const parent = meta.parent as { run?: unknown; index?: unknown } | null;
		run = {
			id,
			kind: text(meta.kind) ?? '',
			name: text(meta.name) ?? '',
			parent:
				parent && text(parent.run) !== null && count(parent.index) !== null
					? { run: parent.run as string, index: parent.index as number }
					: null,
			total: count(meta.total),
			steps: [],
			since: now,
		};
		state.runs.push(run);
		if (announce && run.parent === null) {
			const values: Record<string, string | number> = { kind: run.kind, name: run.name };
			if (run.total !== null) values.n = run.total;
			const key = run.total === null ? 'run_started_open' : run.total === 1 ? 'run_started_one' : 'run_started';
			state.said.push({ key, values });
		}
	} else if (run.total === null && count(meta.total) !== null) {
		run.total = count(meta.total);
	}

	let step = run.steps.find((s) => s.index === index);
	const before = step ? step.state : null;
	if (step && (FINAL.includes(step.state) || seq < step.seq)) return;
	if (!step) {
		step = {
			index,
			label: '',
			state: 'pending',
			stage: 'seed',
			reached: 'seed',
			seq,
			progress: null,
			reason: null,
			ranAs: null,
			durationMs: null,
			stepType: null,
			given: [],
		};
		run.steps.push(step);
		run.steps.sort((a, b) => a.index - b.index);
	}
	const progress = meta.progress as StepProgress | null;
	step.seq = seq;
	step.state = stateName;
	step.label = text(meta.label) ?? step.label;
	step.stepType = text(meta.step_type);
	step.progress =
		stateName === 'running' && progress && count(progress.done) !== null && count(progress.total) !== null
			? { done: progress.done, total: progress.total, unit: String(progress.unit ?? '') }
			: null;
	step.reason = text(meta.reason);
	step.ranAs = text(meta.ran_as);
	step.durationMs = count(meta.duration_ms);
	if (stateName === 'running' && step.reached === 'seed') step.reached = 'crook';
	if (announce) announceStep(state, run, step, before);
}

function counts(state: LoaderState): { steps: number; counts: Record<string, number> } {
	const tally: Record<string, number> = { failed: 0, skipped: 0, not_run: 0, cancelled: 0 };
	let steps = 0;
	for (const run of topRuns(state)) {
		for (const step of run.steps) {
			steps += 1;
			if (step.state in tally) tally[step.state] += 1;
		}
	}
	return { steps, counts: tally };
}

/** Folds one server frame into the state; a type it does not know changes nothing but the silence. */
export function observe(state: LoaderState, frame: Frame, now: number, mapStatus: StatusMapper = noStatus): LoaderState {
	if (state.end !== null) return { ...state, said: [] };
	const next = copy(state);
	next.lastFrameAt = now;
	next.silentSince = null;
	const meta = (frame.metadata ?? {}) as Record<string, unknown>;
	switch (frame.type) {
		case 'metadata': {
			const model = text(meta.model);
			if (model !== null && next.model === null) {
				next.model = model;
				setLine(next, { key: 'sent', values: { model } }, now);
				next.said.push({ key: 'sent', values: { model } });
			}
			break;
		}
		case 'status': {
			const message = text(meta.message);
			const line = message === null ? null : mapStatus(message);
			if (line) setLine(next, line, now);
			break;
		}
		case 'vision_delegation': {
			const message = text(meta.message);
			const line = meta.status === 'analyzing' && message !== null ? mapStatus(message) : null;
			if (line) setLine(next, line, now);
			break;
		}
		case 'thinking':
			if (!next.thinking) {
				next.thinking = true;
				setLine(next, { key: 'reasoning' }, now);
				if (!runOpen(next)) next.said.push({ key: 'reasoning' });
			}
			break;
		case 'token':
			if (!next.writing) {
				next.writing = true;
				if (!runOpen(next)) next.said.push({ key: 'writing' });
			}
			break;
		case 'tool_call': {
			const tool = text(meta.tool_name);
			if (tool !== null) {
				const seconds = typeof meta.execution_time === 'number' ? meta.execution_time : 0;
				setLine(next, { key: 'tool_used', values: { tool, s: seconds < 10 ? Math.round(seconds * 10) / 10 : Math.round(seconds) } }, now);
			}
			break;
		}
		case 'tool_call_pending': {
			const tool = text(meta.tool_name);
			if (tool !== null) {
				next.held = next.line;
				setLine(next, { key: 'tool_pending', values: { tool } }, now);
			}
			break;
		}
		case 'tool_call_resolved':
			if (next.line && next.line.key === 'tool_pending') {
				next.line = next.held;
				next.lineSince = now;
			}
			next.held = null;
			break;
		case 'pipeline_step':
			applyStep(next, meta, true, now);
			break;
		case 'done': {
			if (Array.isArray(meta.steps)) {
				for (const record of meta.steps) {
					if (record && typeof record === 'object') applyStep(next, record as Record<string, unknown>, false, now);
				}
				next.doneSteps = true;
			}
			next.durationMs = count(meta.duration_ms);
			const stopped = meta.cancelled === true || next.stopRequested;
			next.end = stopped ? 'stopped' : 'done';
			if (stopped) {
				if (!next.stopSaid) {
					next.stopSaid = true;
					next.said.push({ key: 'stopped' });
				}
			} else if (next.doneSteps && next.runs.length > 0) {
				const tally = counts(next);
				next.said.push({ key: 'complete_counts', values: { n: tally.steps, ...tally.counts } });
			} else {
				next.said.push({ key: 'complete' });
			}
			break;
		}
		case 'error':
			next.end = 'error';
			if (frame.content) next.said.push({ key: 'error', values: { message: frame.content } });
			break;
		// 'ping' and every type the loader does not read: the silence alone is reset.
		default:
			break;
	}
	grow(next);
	stamp(next, now);
	return next;
}

/** A retry of the connection, before any frame has come. */
export function reconnecting(state: LoaderState, attempt: number, max: number, now: number): LoaderState {
	if (state.end !== null) return { ...state, said: [] };
	const next = copy(state);
	setLine(next, { key: 'reconnecting', values: { n: attempt, max } }, now);
	return next;
}

/** The reader's Stop: said at once; the server's closing frames still come. */
export function stop(state: LoaderState, now: number): LoaderState {
	if (state.end !== null || state.stopRequested) return { ...state, said: [] };
	const next = copy(state);
	next.stopRequested = true;
	next.stopSaid = true;
	setLine(next, { key: 'stopped' }, now);
	next.said.push({ key: 'stopped' });
	return next;
}

/** The stream closed without `done` or `error`: every open step becomes unknown. */
export function lose(state: LoaderState, now: number): LoaderState {
	if (state.end !== null) return { ...state, said: [] };
	const next = copy(state);
	next.end = 'lost';
	let open = 0;
	let steps = 0;
	for (const run of next.runs) {
		for (const step of run.steps) {
			if (run.parent === null) steps += 1;
			if (FINAL.includes(step.state)) continue;
			step.state = 'unknown';
			if (run.parent === null) open += 1;
		}
	}
	grow(next);
	stamp(next, now);
	if (open > 0) next.said.push({ key: 'lost', values: { k: open, n: steps } });
	next.lineSince = now;
	return next;
}

/** The passing of time: the only thing it can change is the silence. */
export function tick(state: LoaderState, now: number): LoaderState {
	if (state.end !== null || state.silentSince !== null || !silent(state, now)) return { ...state, said: [] };
	const next = copy(state);
	next.silentSince = now;
	next.said.push({ key: 'silence', values: { s: SILENCE_MS / 1000 } });
	return next;
}

export function silent(state: LoaderState, now: number): boolean {
	return state.end === null && now - state.lastFrameAt >= SILENCE_MS;
}

/** Whether the loader's one moving thing moves. */
export function moving(state: LoaderState, now: number): boolean {
	if (state.end !== null || state.stopRequested || silent(state, now)) return false;
	return runOpen(state) || !state.writing;
}

/** The status line's words, as a line of the table, or null when it is retired. */
export function lineOf(state: LoaderState, now: number): Line | null {
	if (state.end !== null) return null;
	if (silent(state, now)) return { key: 'silence', values: { s: SILENCE_MS / 1000 } };
	if (state.stopRequested) return { key: 'stopped' };
	if (state.writing) return null;
	if (state.line) return state.line;
	return now - state.startedAt >= SENDING_AFTER_MS ? { key: 'sending' } : null;
}

/** The summary of the reply's first run, or null when there is none to say. */
export function summary(state: LoaderState): Summary | null {
	const run = topRuns(state)[0];
	if (!run || run.steps.length === 0) return null;
	const total = run.total ?? run.steps.length;
	const base = { kind: run.kind, name: run.name, steps: total, durationMs: state.durationMs };
	const tally = counts(state).counts;
	if (state.end === 'lost') {
		const at = run.steps.find((step) => step.state === 'unknown');
		return { ...base, form: 'lost', at: at ? at.index + 1 : total, counts: tally, durationMs: null };
	}
	if ((state.end === 'done' || state.end === 'stopped') && state.doneSteps) {
		if (state.end === 'stopped') {
			const at =
				run.steps.find((step) => step.state === 'cancelled') ??
				run.steps.find((step) => step.state === 'not_run');
			return { ...base, form: 'stopped', at: at ? at.index + 1 : total, counts: tally };
		}
		return { ...base, form: 'done', at: total, counts: tally };
	}
	return null;
}

/** How an end is told apart: form, ink, mark and pattern, never colour alone. */
export function endLook(step: Step): Look {
	switch (step.state) {
		case 'failed':
			return { form: 'stage', ink: 'dried', mark: 'tag', pattern: 'solid' };
		case 'cancelled':
			return { form: 'stage', ink: 'text_muted', mark: null, pattern: 'solid' };
		case 'unknown':
			return { form: 'stage', ink: 'rule', mark: null, pattern: 'dotted' };
		case 'not_run':
			return { form: 'seed', ink: 'rule', mark: null, pattern: 'solid' };
		case 'skipped':
			return { form: 'soil', ink: 'rule_soft', mark: null, pattern: 'solid' };
		default:
			return { form: 'stage', ink: null, mark: null, pattern: 'solid' };
	}
}

/** The status region's queue: the last of a burst, after calm, one a second. */
export interface Announcer {
	text: string | null;
	at: number;
	lastSpoken: number | null;
}

export function announcer(): Announcer {
	return { text: null, at: 0, lastSpoken: null };
}

export function announce(queue: Announcer, words: string, now: number): Announcer {
	return { ...queue, text: words, at: now };
}

/** What the region speaks now, if anything, and the queue after it. */
export function due(queue: Announcer, now: number): { announcer: Announcer; text: string | null } {
	if (queue.text === null || now - queue.at < COALESCE_MS) return { announcer: queue, text: null };
	if (queue.lastSpoken !== null && now - queue.lastSpoken < ANNOUNCE_GAP_MS) return { announcer: queue, text: null };
	return { announcer: { text: null, at: queue.at, lastSpoken: now }, text: queue.text };
}
