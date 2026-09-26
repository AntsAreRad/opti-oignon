/**
 * The URLs the interface used to serve, and the pages that now hold what
 * they held.
 *
 * Each old page's `load` redirects through `legacyTarget`, its query kept.
 * The old settings page's links are resolved against the settings catalog,
 * which the caller hands in (so this module imports nothing and runs alone):
 * a group named by `g` is found wherever the catalog now places it, through
 * its host when a host's panel renders it; an old section or tab goes to the
 * page that took its content; anything unknown goes to Preferences, never to
 * a missing page. This is the one file that knows the old URLs.
 *
 * Pure functions, with no dependency.
 */

/** A group of the settings catalog, with where it lives now. */
export interface PlacedGroup {
	id: string;
	space?: 'use' | 'workshop';
	section?: string;
	retired?: string;
	embeddedIn?: string;
}

/** The parts of the settings catalog the old links are resolved against. */
export interface LegacyCatalog {
	SETTINGS_SECTIONS: readonly { id: string; groups: readonly PlacedGroup[] }[];
	INLINE_GROUPS: readonly PlacedGroup[];
	LEGACY_TAB_TO_SECTION: Readonly<Record<string, string>>;
}

/** Where an old link lands: a page, or the page holding a group. */
type Landing = { page: string } | { group: string };

const PREFERENCES = '/preferences';
const WORKSHOP = '/workshop';
const SLUG = /^[a-z][a-z0-9-]*$/;

/** The old pages other than settings: their new page, and a query they set. */
const MOVED: Readonly<Record<string, { to: string; set?: Readonly<Record<string, string>> }>> = {
	'/health': { to: WORKSHOP },
	'/benchmark': { to: `${WORKSHOP}/benchmarks` },
	'/verify': { to: `${WORKSHOP}/verify` },
	'/claims': { to: `${WORKSHOP}/verify`, set: { mode: 'pairs' } },
	'/verify-answer': { to: `${WORKSHOP}/verify`, set: { mode: 'pairs' } },
	'/verify-citations': { to: `${WORKSHOP}/verify`, set: { mode: 'cited' } }
};

/** The old settings sections, each to the page that took its content. */
const OLD_SECTIONS: Readonly<Record<string, Landing>> = {
	appearance: { page: PREFERENCES },
	account: { page: `${WORKSHOP}/security` },
	conversation: { group: 'task-presets' },
	models: { page: `${WORKSHOP}/models` },
	knowledge: { page: `${WORKSHOP}/knowledge` },
	plugins: { page: `${WORKSHOP}/extensions` },
	performance: { page: `${WORKSHOP}/observability` },
	network: { page: `${WORKSHOP}/network` },
	data: { page: `${WORKSHOP}/backup` }
};

/** The old tabs whose content is now one group rather than a whole page. */
const OLD_TAB_GROUPS: Readonly<Record<string, string>> = {
	quick: 'conversation-system-preset',
	presets: 'task-presets',
	prompt: 'prompt-config',
	analytics: 'analytics',
	backup: 'backup-restore',
	'fine-tune': 'fine-tune'
};

function own<T>(record: Readonly<Record<string, T>>, key: string): T | undefined {
	return Object.prototype.hasOwnProperty.call(record, key) ? record[key] : undefined;
}

function withQuery(path: string, params: URLSearchParams): string {
	const query = params.toString();
	return query ? `${path}?${query}` : path;
}

function findGroup(id: string, catalog: LegacyCatalog): PlacedGroup | undefined {
	for (const section of catalog.SETTINGS_SECTIONS) {
		const found = section.groups.find((group) => group.id === id);
		if (found) return found;
	}
	return catalog.INLINE_GROUPS.find((group) => group.id === id);
}

/**
 * The page holding group `id`, and the group to open there: its host when
 * the group is embedded; null when the group is unknown, retired, or placed
 * nowhere a page exists.
 */
function pageOfGroup(id: string, catalog: LegacyCatalog): { page: string; g: string } | null {
	let group = findGroup(id, catalog);
	if (group?.embeddedIn) group = findGroup(group.embeddedIn, catalog);
	if (!group || group.retired || group.embeddedIn) return null;
	if (group.space === 'use') return { page: PREFERENCES, g: group.id };
	if (group.space === 'workshop' && group.section && SLUG.test(group.section)) {
		return { page: `${WORKSHOP}/${group.section}`, g: group.id };
	}
	return null;
}

function tabLanding(tab: string, catalog: LegacyCatalog): Landing | undefined {
	const group = own(OLD_TAB_GROUPS, tab);
	if (group) return { group };
	const section = own(OLD_SECTIONS, tab);
	if (section) return section;
	const mapped = own(catalog.LEGACY_TAB_TO_SECTION, tab);
	return mapped ? own(OLD_SECTIONS, mapped) : undefined;
}

function settingsTarget(search: URLSearchParams, catalog: LegacyCatalog): string {
	const params = new URLSearchParams(search);
	const g = params.get('g');
	const section = params.get('section');
	const tab = params.get('tab');
	params.delete('g');
	params.delete('section');
	params.delete('tab');

	const held = g ? pageOfGroup(g, catalog) : null;
	if (held) {
		params.set('g', held.g);
		return withQuery(held.page, params);
	}

	const landing = (section ? own(OLD_SECTIONS, section) : undefined) ?? (tab ? tabLanding(tab, catalog) : undefined);
	if (landing && 'page' in landing) return withQuery(landing.page, params);
	if (landing && 'group' in landing) {
		const place = pageOfGroup(landing.group, catalog);
		if (place) {
			params.set('g', place.g);
			return withQuery(place.page, params);
		}
	}
	return withQuery(PREFERENCES, params);
}

/**
 * The page an old URL now means, its query kept; null when the URL is not
 * one the interface used to serve.
 */
export function legacyTarget(url: URL, catalog: LegacyCatalog): string | null {
	const path = url.pathname.replace(/\/+$/, '') || '/';
	if (path === '/settings') return settingsTarget(url.searchParams, catalog);
	const moved = own(MOVED, path);
	if (!moved) return null;
	const params = new URLSearchParams(url.searchParams);
	for (const [key, value] of Object.entries(moved.set ?? {})) params.set(key, value);
	return withQuery(moved.to, params);
}
