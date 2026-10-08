/**
 * Pending writes: what the review section shows, and how its selection moves.
 *
 * A memory or notes write of the agent that the user's typed words did not
 * endorse waits for the user's review (opti_oignon/pending_writes.py), and
 * so does a fact the manual extraction drew from anything but those words.
 * This module holds no state and imports nothing: it says what a proposal
 * would do, where each of its words came from, and which ids a decision
 * applies to, so the panels stay thin views over it.
 */

export type PendingStore = 'memory' | 'notes';

/** Where a proposal's words came from, as the queue records them. */
export interface PendingProvenance {
	/** 'agent' for a tool call of a run, 'extraction' for the manual extraction. */
	source?: string;
	/** 'typed' when the run's turn held typed words, 'none' otherwise. */
	turn?: string;
	/** Each endorsed argument with the typed unit (or units, for tags) that endorsed it. */
	typed?: Record<string, string | string[]>;
	/** The content arguments no typed unit endorsed. */
	untyped?: string[];
	/** The identifier an update or a delete aims at. */
	target?: string | null;
	/** The tools whose results the run had read before proposing, in order. */
	read?: string[];
}

export interface PendingWrite {
	id: string;
	store: PendingStore;
	action: string;
	arguments: Record<string, unknown>;
	provenance: PendingProvenance;
	conversation_id: string;
	run_id: string;
	created_at: string;
	/** What an update or a delete would change, as the store holds it now. */
	target: Record<string, string> | null;
}

export interface DecisionResult {
	id: string;
	applied?: boolean;
	declined?: boolean;
	reason?: string;
	outcome?: string;
}

export type ArgumentOrigin = 'typed' | 'not-typed' | 'target' | 'setting';

export interface ArgumentRow {
	name: string;
	value: string;
	origin: ArgumentOrigin;
}

const ACTIONS: Record<string, string> = {
	'memory:add': 'Remember a fact',
	'memory:update': 'Change a remembered fact',
	'memory:delete': 'Forget a remembered fact',
	'notes:make': 'Create a note',
	'notes:update': 'Change a note',
	'notes:delete': 'Delete a note',
};

const NAMES: Record<string, string> = {
	text: 'Fact',
	category: 'Category',
	title: 'Title',
	body: 'Body',
	tags: 'Tags',
	pinned: 'Pinned',
	fact_id: 'Fact',
	note_id: 'Note',
};

/** How each origin of an argument is said in the panel. */
export const ORIGIN_LABELS: Record<ArgumentOrigin, string> = {
	typed: 'typed by you',
	'not-typed': 'not typed by you',
	target: 'what it changes',
	setting: 'a setting',
};

const READ_NAMES: Record<string, string> = {
	web_search: 'web search results',
	view: 'files',
	grep: 'files',
	glob: 'files',
	ls: 'files',
	bash: 'command output',
	create_file: 'files',
	str_replace: 'files',
	manage_memory: 'your memory',
	manage_notes: 'your notes',
	manage_skills: 'skills',
	task: 'a subtask',
};

/** What accepting the proposal would do, in a few words. */
export function describeWrite(item: PendingWrite): string {
	return ACTIONS[`${item.store}:${item.action}`] ?? `${item.action} (${item.store})`;
}

/** Who proposed it: a run of the agent, or the manual extraction. */
export function sourceLabel(item: PendingWrite): string {
	return item.provenance?.source === 'extraction' ? 'Drawn from the conversation' : 'Proposed by the agent';
}

function tagList(raw: unknown): string[] {
	if (Array.isArray(raw)) return raw.map(String);
	if (typeof raw !== 'string') return [];
	try {
		const parsed: unknown = JSON.parse(raw);
		return Array.isArray(parsed) ? parsed.map(String) : [raw];
	} catch {
		return [raw];
	}
}

/** Each argument the proposal would write, with its value and where its words came from. */
export function argumentRows(item: PendingWrite): ArgumentRow[] {
	const provenance = item.provenance ?? {};
	const untyped = new Set(provenance.untyped ?? []);
	const typed = provenance.typed ?? {};
	const rows: ArgumentRow[] = [];
	for (const [name, raw] of Object.entries(item.arguments ?? {})) {
		if (raw === null || raw === undefined) continue;
		const value = name === 'tags' ? tagList(raw).join(', ') : String(raw);
		if (value === '' || (name === 'pinned' && raw === false)) continue;
		let origin: ArgumentOrigin = 'setting';
		if (name === provenance.target) origin = 'target';
		else if (untyped.has(name)) origin = 'not-typed';
		else if (name in typed) origin = 'typed';
		rows.push({ name: NAMES[name] ?? name, value, origin });
	}
	return rows;
}

/** What the agent had read before proposing, or an empty string when it had read nothing. */
export function readBefore(item: PendingWrite): string {
	const read = item.provenance?.read ?? [];
	if (read.length === 0) return '';
	const names = [...new Set(read.map((tool) => READ_NAMES[tool] ?? tool))];
	return `Read before proposing: ${names.join(', ')}`;
}

/** What an update or a delete would change, or an empty string. */
export function targetLine(item: PendingWrite): string {
	const now = item.target?.text ?? item.target?.title ?? '';
	return now ? `Now: ${now}` : '';
}

/** ``selected`` with ``id`` flipped. */
export function toggle(selected: readonly string[], id: string): string[] {
	return selected.includes(id) ? selected.filter((one) => one !== id) : [...selected, id];
}

/** Whether every proposal shown is selected. */
export function allSelected(items: readonly PendingWrite[], selected: readonly string[]): boolean {
	return items.length > 0 && items.every((item) => selected.includes(item.id));
}

/** Every proposal shown, or none when every one is already selected. */
export function toggleAll(items: readonly PendingWrite[], selected: readonly string[]): string[] {
	return allSelected(items, selected) ? [] : items.map((item) => item.id);
}

/** The ids a decision applies to: the selected proposals still shown, in the order shown. */
export function decisionIds(items: readonly PendingWrite[], selected: readonly string[]): string[] {
	const chosen = new Set(selected);
	return items.filter((item) => chosen.has(item.id)).map((item) => item.id);
}

/** The reasons that say a proposal is decided, or was never there: it leaves the list. */
const DECIDED = new Set(['not pending', 'not found', 'target not found']);

/** The proposals still waiting once a decision returned: a failed write stays, anything decided leaves. */
export function remaining(items: readonly PendingWrite[], results: readonly DecisionResult[]): PendingWrite[] {
	const gone = new Set(
		results.filter((r) => r.applied || r.declined || DECIDED.has(r.reason ?? '')).map((r) => r.id),
	);
	return items.filter((item) => !gone.has(item.id));
}
