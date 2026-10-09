/**
 * Pending writes: what the review section shows, and how its selection moves.
 *
 * A memory or notes write of the agent that the user's typed words did not
 * endorse waits for the user's review (opti_oignon/pending_writes.py), and
 * so does a fact the manual extraction drew from anything but those words,
 * and every skill the agent or its teacher writes. This module holds no
 * state and imports nothing: it says what a proposal would do, where each of
 * its words came from, how each is shown (every character a screen hides
 * written as its escape), and which ids and digests a decision applies to,
 * so the panels stay thin views over it.
 */

export type PendingStore = 'memory' | 'notes' | 'skills';

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
	/** Each value and target as the approval drawer shows it, whole, every hidden character written out. */
	shown?: { arguments?: Record<string, unknown>; target?: Record<string, unknown> | null };
	/** The digest an acceptance names: a skill's text (its target's for a delete), else the proposal's own. */
	digest?: string;
	/** 'high' for a skill: its text reaches a system prompt. */
	risk?: string;
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
	'skills:add': 'Add a skill',
	'skills:edit': 'Change a skill',
	'skills:delete': 'Delete a skill',
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

/** Who proposed it: a run of the agent, its teacher model, or the manual extraction. */
export function sourceLabel(item: PendingWrite): string {
	if (item.provenance?.source === 'extraction') return 'Drawn from the conversation';
	if (item.provenance?.source === 'teacher') return 'Proposed by the teacher model';
	return 'Proposed by the agent';
}

const BACKSLASH = String.fromCharCode(92);

/** Every character but printable ASCII and a line break written as its escape, a backslash doubled: hides nothing. */
export function escapeAll(text: string): string {
	let out = '';
	for (const ch of text) {
		const code = ch.codePointAt(0) ?? 0;
		if (ch === BACKSLASH) out += BACKSLASH + BACKSLASH;
		else if (ch === '\n' || (code >= 0x20 && code <= 0x7e)) out += ch;
		else if (code <= 0xff) out += BACKSLASH + 'x' + code.toString(16).padStart(2, '0');
		else if (code <= 0xffff) out += BACKSLASH + 'u' + code.toString(16).padStart(4, '0');
		else out += BACKSLASH + 'U' + code.toString(16).padStart(8, '0');
	}
	return out;
}

/**
 * A value as the review shows it: the server's rendering, the approval
 * drawer's, when it sent one; else every character but printable ASCII and a
 * line break written as its escape. Nothing a screen hides is drawn as itself.
 */
export function shownValue(shown: unknown, raw: unknown): string {
	if (typeof shown === 'string') return shown;
	if (Array.isArray(shown)) return shown.map((one) => (typeof one === 'string' ? one : escapeAll(String(one)))).join(', ');
	return escapeAll(raw === null || raw === undefined ? '' : String(raw));
}

/** What a skill proposal shows: where it writes, its whole text, what it replaces, its digest. */
export interface SkillView {
	where: string;
	draft: boolean;
	text: string;
	replaces: string;
	digest: string;
	tested: boolean;
	/** What it changes is no longer the text it was proposed against: accepting it would be refused. */
	changed: boolean;
}

/** A skill proposal as the review shows it, every text through ``shownValue``; null for any other proposal. */
export function skillView(item: PendingWrite): SkillView | null {
	if (item.store !== 'skills') return null;
	const args = item.arguments ?? {};
	const shownArgs = item.shown?.arguments ?? {};
	const shownTarget = item.shown?.target ?? {};
	return {
		where: escapeAll(`${String(args.category ?? '')}/${String(args.name ?? '')}`),
		draft: args.draft === true,
		text: item.action === 'delete' ? '' : shownValue(shownArgs.text, args.text),
		replaces: item.target ? shownValue(shownTarget.text, item.target.text) : '',
		digest: item.digest ?? '',
		tested: args.tested === true,
		changed: (item.target?.sha256 ?? null) !== (args.base_sha256 ?? null),
	};
}

/** The digest of each proposal a decision applies to, by id: what the user was shown. */
export function decisionDigests(items: readonly PendingWrite[], ids: readonly string[]): Record<string, string> {
	const chosen = new Set(ids);
	const digests: Record<string, string> = {};
	for (const item of items) {
		if (chosen.has(item.id) && item.digest) digests[item.id] = item.digest;
	}
	return digests;
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

/** Each argument the proposal would write, with its value as shown and where its words came from. */
export function argumentRows(item: PendingWrite): ArgumentRow[] {
	const provenance = item.provenance ?? {};
	const untyped = new Set(provenance.untyped ?? []);
	const typed = provenance.typed ?? {};
	const shown = item.shown?.arguments ?? {};
	const rows: ArgumentRow[] = [];
	for (const [name, raw] of Object.entries(item.arguments ?? {})) {
		if (raw === null || raw === undefined) continue;
		const value = name === 'tags' ? tagList(raw).map(escapeAll).join(', ') : shownValue(shown[name], raw);
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

/** What an update or a delete would change, as shown, or an empty string. */
export function targetLine(item: PendingWrite): string {
	const raw = item.target?.text ?? item.target?.title ?? '';
	const shown = item.shown?.target?.text ?? item.shown?.target?.title;
	const now = raw ? shownValue(shown, raw) : '';
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
const DECIDED = new Set(['not pending', 'not found', 'target not found', 'target changed', 'digest mismatch']);

/** The proposals still waiting once a decision returned: a failed write stays, anything decided leaves. */
export function remaining(items: readonly PendingWrite[], results: readonly DecisionResult[]): PendingWrite[] {
	const gone = new Set(
		results.filter((r) => r.applied || r.declined || DECIDED.has(r.reason ?? '')).map((r) => r.id),
	);
	return items.filter((item) => !gone.has(item.id));
}
