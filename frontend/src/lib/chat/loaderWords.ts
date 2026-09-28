/**
 * Every word the chat loader shows or announces: one closed table.
 *
 * The loader's words stand beside the being, so each of them has passed the
 * garden's two ethics nets (the governed class and the blocklist). They are
 * printable ASCII, with a comma as the only separator. No word describes the
 * drawing, no `ran_as` code is shown raw, and only a safety stop is said
 * "Stopped". A status the table does not map has no words: it changes
 * nothing visible until the table learns it. No step text is parsed: a run's
 * steps come from `pipeline_step` alone.
 *
 * Templates carry their values as `{name}`; `fill` puts them in.
 */

import type { Line, Step, Summary } from './progress';

export const WORDS: Record<string, string> = {
	sending: 'Sending',
	sent: 'Sent to {model}',
	web_search: 'Searching the web',
	web_results: 'Web search: {n} results',
	web_none: 'Web search: no results',
	web_failed: 'Web search failed',
	web_off: 'Web search skipped: the search kill switch is on',
	project_reading: 'Reading project context',
	project_passages: 'Project context: {n} passages',
	fitting: 'Fitting the history into the context',
	cache: 'Found in the cache',
	queued: 'Queued: Ollama is offline',
	vision: 'Reading the image with {model}',
	self_correct: 'Correcting the reply',
	cascade: 'Trying a cascade of models',
	speculative: 'Drafting and verifying the reply',
	reasoning: 'Reasoning',
	writing: 'Writing the reply',
	tool_used: 'Used {tool}, {s} s',
	tool_pending: 'Tool approval pending: {tool}',
	reconnecting: 'Connection lost, trying again ({n} of {max})',
	silence: 'No word from the server for {s} s',
	stopped: 'Stopped',
	stopped_because: 'Stopped: {reason}',
	error: '{message}',
	complete: 'Reply complete',
	complete_counts: 'Reply complete: {counts}',
	run_started: '{run} started: {n} steps',
	run_started_one: '{run} started: 1 step',
	run_started_open: '{run} started',
	header: '{run}, step {i} of {n}',
	header_open: '{run}, step {i}',
	at_step: 'Step {i} of {n}, {label}',
	at_step_open: 'Step {i}, {label}',
	step_started: 'Step {i} of {n} started: {label}',
	step_started_open: 'Step {i} started: {label}',
	step_failed: 'Step {i} of {n} failed: {reason}',
	step_failed_open: 'Step {i} failed: {reason}',
	lost: 'Connection closed before the reply ended. Outcome unknown for {k} of {n} steps',
	state_pending: 'To come',
	state_running: 'Running',
	state_done: 'Done',
	state_failed: 'Failed',
	state_cancelled: 'Stopped',
	state_cancelled_because: 'Stopped: {reason}',
	state_skipped: 'Skipped: its condition was not met',
	state_not_run: 'Not run',
	state_not_run_because: 'Not run: {reason}',
	state_unknown: 'Outcome unknown: the connection closed before this step ended',
	count_steps: '{n} steps',
	count_step: '1 step',
	count_failed: '{n} failed',
	count_skipped: '{n} skipped',
	count_not_run: '{n} not run',
	count_cancelled: '{n} stopped',
	row_earlier: '{n} earlier',
	row_later: '{n} later',
	summary_done: '{run}, {counts}, {s} s',
	summary_done_open: '{run}, {counts}',
	summary_stopped: '{run}, stopped at step {i} of {n}, {s} s',
	summary_stopped_open: '{run}, stopped at step {i} of {n}',
	summary_lost: '{run}, connection closed at step {i} of {n}',
	seconds: ', {s} s',
	reason: ': {reason}',
	and: ', {words}',
};

/** How a pipeline step really ran, said in words (never the code). */
export const RAN_AS: Record<string, string> = {
	direct: 'Ran as a direct reply',
	tools: 'Ran as a reply with tools',
	code_verify: 'Ran as code with verification',
	think: 'Ran as a reply with reasoning',
	web_search: 'Ran as a web search',
	think_tools: 'Ran as reasoning with tools',
	reasoning: 'Ran as the step-by-step reasoning pipeline',
	consensus: 'Ran as several models compared',
	self_correct: 'Ran as a self-checked reply',
	cascading: 'Ran as a cascade of models',
	speculative: 'Ran as speculative decoding',
};

/** A run's name by its kind; an execution pipeline says its own name. */
export const RUN_NAMES: Record<string, string> = {
	reasoning: 'Reasoning',
	consensus: 'Models compared',
	self_correct: 'Self-check',
	coding: 'Coding',
	other: 'Steps',
};

/** A running step's sub-progress, by its unit. */
export const UNITS: Record<string, string> = {
	sub_step: '{done} of {total} sub-steps done',
	model: '{done} of {total} models finished',
	sample: '{done} of {total} samples done',
};

type Values = Record<string, string | number>;

/**
 * The server's statuses the table says, by their fixed openings. The runner's
 * own step line is not among them: the terminal reads it, the loader does not.
 */
const STATUSES: readonly [RegExp, string, ((match: RegExpMatchArray) => Values)?][] = [
	[/^\[>\] Web search for:/, 'web_search'],
	[/^\[OK\] (\d+) search results injected/, 'web_results', (m) => ({ n: Number(m[1]) })],
	[/^\[!\] Web search returned no results/, 'web_none'],
	[/^\[!\] Web search failed/, 'web_failed'],
	[/^\[!\] Web search skipped \(kill switch engaged\)/, 'web_off'],
	[/^\[>\] Project context:/, 'project_reading'],
	[/^\[OK\] Project context injected: (\d+) chunks/, 'project_passages', (m) => ({ n: Number(m[1]) })],
	[/^\[>\] (?:Archive retrieval|Optimizer|Multi-turn):/, 'fitting'],
	[/^\[(?:CACHE|SEMANTIC|CACHE-X)\]/, 'cache'],
	[/^\[!\] Ollama offline -- request queued/, 'queued'],
	[/^Analyzing image with (.+?)\.{3}$/, 'vision', (m) => ({ model: m[1] })],
	[/^Running self-correction/, 'self_correct'],
	[/^\[>\] Running cascading inference/, 'cascade'],
	[/^\[>\] Running speculative generation/, 'speculative'],
];

export function fill(template: string, values: Values = {}): string {
	return template.replace(/\{(\w+)\}/g, (_, name: string) => (name in values ? String(values[name]) : ''));
}

function seconds(ms: number): number {
	return Math.round(ms / 1000);
}

/** Live seconds are shown from this many seconds on the line. */
export const LIVE_FROM_S = 5;

/** A duration the server measured, said as the table says seconds: `, 12 s`. */
export function durationWords(ms: number): string {
	return fill(WORDS.seconds, { s: seconds(ms) });
}

/** Live seconds, counted in the tab: whole seconds gone, from the fifth; else null. */
export function liveSeconds(since: number, now: number): string | null {
	const s = Math.floor((now - since) / 1000);
	return s >= LIVE_FROM_S ? fill(WORDS.seconds, { s }) : null;
}

export function runName(kind: string, name: string): string {
	if (kind === 'exec_pipeline') return name || RUN_NAMES.other;
	return RUN_NAMES[kind] ?? RUN_NAMES.other;
}

/** `3 steps, 1 failed`: the step count, then each non-zero end. */
export function countsText(values: Values): string {
	const n = Number(values.n ?? 0);
	const parts = [n === 1 ? WORDS.count_step : fill(WORDS.count_steps, { n })];
	for (const end of ['failed', 'skipped', 'not_run', 'cancelled']) {
		const k = Number(values[end] ?? 0);
		if (k > 0) parts.push(fill(WORDS[`count_${end}`], { n: k }));
	}
	return parts.join(', ');
}

/** A line's words, or null for a key the table does not have. */
export function words(line: Line | null): string | null {
	if (!line) return null;
	const template = WORDS[line.key];
	if (template === undefined) return null;
	const values: Values = { ...(line.values ?? {}) };
	if ('kind' in values) values.run = runName(String(values.kind), String(values.name ?? ''));
	if (line.key === 'complete_counts') values.counts = countsText(values);
	return fill(template, values);
}

/** A line as the status region says it: a sentence. */
export function said(line: Line): string {
	const text = words(line) ?? '';
	return text.endsWith('.') ? text : `${text}.`;
}

/** The words for a server status, or null when the table does not map it. */
export function statusLine(message: string): Line | null {
	for (const [pattern, key, values] of STATUSES) {
		const match = message.match(pattern);
		if (match) return values ? { key, values: values(match) } : { key };
	}
	return null;
}

export function ranAsWords(code: string): string | null {
	return Object.prototype.hasOwnProperty.call(RAN_AS, code) ? RAN_AS[code] : null;
}

/** A running step's sub-progress in words, or null without a fraction. */
export function progressWords(step: Step): string | null {
	if (step.state !== 'running' || !step.progress) return null;
	const template = UNITS[step.progress.unit];
	return template === undefined ? null : fill(template, { done: step.progress.done, total: step.progress.total });
}

/** A step line's words for its state. */
export function stepWords(step: Step): string {
	const reason = step.reason ? { reason: step.reason } : null;
	switch (step.state) {
		case 'pending':
			return WORDS.state_pending;
		case 'running':
			return WORDS.state_running;
		case 'done':
			return WORDS.state_done;
		case 'failed':
			return WORDS.state_failed;
		case 'cancelled':
			return reason ? fill(WORDS.state_cancelled_because, reason) : WORDS.state_cancelled;
		case 'skipped':
			return WORDS.state_skipped;
		case 'not_run':
			return reason ? fill(WORDS.state_not_run_because, reason) : WORDS.state_not_run;
		default:
			return WORDS.state_unknown;
	}
}

/** The summary line after the run: its name, then its end in words. */
export function summaryText(summary: Summary): string {
	const run = runName(summary.kind, summary.name);
	const values: Values = { run, i: summary.at, n: summary.steps };
	if (summary.durationMs !== null) values.s = seconds(summary.durationMs);
	if (summary.form === 'lost') return fill(WORDS.summary_lost, values);
	if (summary.form === 'stopped') {
		return fill(summary.durationMs !== null ? WORDS.summary_stopped : WORDS.summary_stopped_open, values);
	}
	values.counts = countsText({ n: summary.steps, ...summary.counts });
	return fill(summary.durationMs !== null ? WORDS.summary_done : WORDS.summary_done_open, values);
}
