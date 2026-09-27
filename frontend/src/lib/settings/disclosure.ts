/**
 * The folding rules of a Workshop settings page: which group is open, what
 * the address says, and which groups keep their panel mounted.
 *
 * One group is open at a time, and the address names it (`?g=`), so the
 * palette, the settings search and a shared link open it. On arrival, the
 * group `g` names opens; a group embedded in another opens its host; with
 * no group named, the page's only group opens when it has one, and none
 * when it has several (the page then reads as a quiet table of contents).
 * The rules run on arrival only, never after the hub's own address writes,
 * so a page's only group can be closed.
 *
 * A group mounts its panel while it is open, and also once its panel was
 * used on this visit: a field typed in or changed, a choice clicked or
 * tapped (a switch, a mode, a row added), a key pressed on a control, a
 * file dropped. Closing it then hides it, and whatever it holds waits for
 * it: a draft, a queue of files, an ingest's progress. Moving through the
 * page (Tab, Escape, a modifier key alone) is no use of the panel. Leaving
 * the page drops what was not saved, as it always did.
 *
 * Pure functions, with no dependency: the settings hub and SettingsGroup
 * call them, and they run under Node as they are.
 */

/** The key under which the hub hands its groups their context. */
export const GROUPS_CONTEXT = 'oo-settings-groups';

/** A store as the context carries it: something a component can subscribe to. */
export interface Subscribable<T> {
	subscribe(run: (value: T) => void): () => void;
}

/** What the hub hands every group of its page. */
export interface GroupsContext {
	/** The level of the groups' titles: 2 on a Workshop page, 3 in Preferences. */
	level: 2 | 3;
	/** Whether the groups fold (a Workshop page) or all stand open (Preferences). */
	collapsible: boolean;
	/** The open group's id, or null. */
	open: Subscribable<string | null>;
	/** The groups whose fields were edited on this visit. */
	edited: Subscribable<ReadonlySet<string>>;
	/** Opens a group, or closes it when it is the open one. */
	toggle(id: string): void;
	/** Marks a group edited: its panel stays mounted until the page is left. */
	markEdited(id: string): void;
}

/**
 * The group to open when a page is reached: the one `g` names when it is on
 * the page, its host when `g` names an embedded group whose host is, else
 * the page's only group, else none.
 */
export function openOnArrival(
	g: string | null | undefined,
	pageGroupIds: readonly string[],
	hostOf: Readonly<Record<string, string>>
): string | null {
	if (g) {
		if (pageGroupIds.includes(g)) return g;
		const host = Object.prototype.hasOwnProperty.call(hostOf, g) ? hostOf[g] : undefined;
		if (host && pageGroupIds.includes(host)) return host;
		if (host) return null;
	}
	return pageGroupIds.length === 1 ? pageGroupIds[0] : null;
}

/** The open group after `id` is toggled: none when it was the open one, else `id`. */
export function toggled(open: string | null, id: string): string | null {
	return open === id ? null : id;
}

/**
 * The address that names `open`: `g` set to it, or removed when nothing is
 * open, every other key kept. The URL given is never changed.
 */
export function addressFor(url: URL, open: string | null): string {
	const next = new URL(url.href);
	if (open) next.searchParams.set('g', open);
	else next.searchParams.delete('g');
	return `${next.pathname}${next.search}`;
}

/** Whether a group's panel is mounted: while it is open, or once it was edited. */
export function mounted(id: string, open: string | null, edited: ReadonlySet<string>): boolean {
	return open === id || edited.has(id);
}

/** The groups edited, with `id` among them. The set given is never changed. */
export function withEdited(edited: ReadonlySet<string>, id: string): ReadonlySet<string> {
	return edited.has(id) ? edited : new Set([...edited, id]);
}

/**
 * The events a group listens for, in the capture phase, on its panel: any
 * of them may start a draft, whatever the panel does with it afterwards (a
 * switch that dispatches its own event, a handler that stops propagation).
 */
export const DRAFT_EVENTS: readonly string[] = ['input', 'change', 'click', 'keydown', 'drop'];

/** The keys that move through the page or hold a modifier, and change nothing. */
const PASSING_KEYS: readonly string[] = ['Tab', 'Escape', 'Shift', 'Control', 'Alt', 'Meta'];

/** Whether an event on a group's panel may start a draft, which the group then keeps. */
export function startsDraft(type: string, key?: string): boolean {
	if (!DRAFT_EVENTS.includes(type)) return false;
	return type !== 'keydown' || !PASSING_KEYS.includes(key ?? '');
}
