/**
 * The destination table: every page the navigation links to, in both spaces.
 *
 * Use holds the pages a person works in; Workshop holds the operator pages,
 * all under /workshop. The sidebar, the route announcer and the command
 * palette read their links from this table and declare none of their own.
 *
 * An entry that is not ready has no page yet: it is never rendered, and no
 * route serves it. An entry marked `settings` is a page that renders groups
 * of the settings catalog (lib/settings/catalog.ts), which names the
 * Workshop page of each group by the last segment of that page's URL.
 *
 * Pure data and pure functions, with no dependency.
 */

export type Space = 'use' | 'workshop';

export interface Destination {
	id: string;
	label: string;
	href: string;
	space: Space;
	/** A name of the icon set (lib/ds/icons.ts, or the fallback set). */
	icon: string;
	/** False while the destination has no page: it is never rendered. */
	ready: boolean;
	/** True for a page that renders groups of the settings catalog. */
	settings?: boolean;
}

/** What the reader chose to see: the componion's switch. */
export interface Visibility {
	componion: boolean;
}

const WORKSHOP_PREFIX = '/workshop/';

export const DESTINATIONS: readonly Destination[] = [
	{ id: 'home', label: 'Home', href: '/', space: 'use', icon: 'home', ready: false },
	{ id: 'chats', label: 'Chats', href: '/chat', space: 'use', icon: 'chat', ready: true },
	{ id: 'notes', label: 'Notes', href: '/notes', space: 'use', icon: 'note', ready: true },
	{ id: 'projects', label: 'Projects', href: '/projects', space: 'use', icon: 'folder', ready: true },
	{ id: 'componion', label: 'Componion', href: '/garden', space: 'use', icon: 'sprout', ready: false },
	{ id: 'preferences', label: 'Preferences', href: '/preferences', space: 'use', icon: 'sliders', ready: true, settings: true },
	{ id: 'status', label: 'System status', href: '/workshop', space: 'workshop', icon: 'info', ready: true },
	{ id: 'models', label: 'Models and inference', href: '/workshop/models', space: 'workshop', icon: 'chip', ready: true, settings: true },
	{ id: 'knowledge', label: 'Knowledge', href: '/workshop/knowledge', space: 'workshop', icon: 'bulb', ready: true, settings: true },
	{ id: 'extensions', label: 'Extensions', href: '/workshop/extensions', space: 'workshop', icon: 'box', ready: true, settings: true },
	{ id: 'verify', label: 'Verify', href: '/workshop/verify', space: 'workshop', icon: 'check', ready: true },
	{ id: 'benchmarks', label: 'Benchmarks', href: '/workshop/benchmarks', space: 'workshop', icon: 'bar-chart-3', ready: true },
	{ id: 'observability', label: 'Observability', href: '/workshop/observability', space: 'workshop', icon: 'activity', ready: true, settings: true },
	{ id: 'network', label: 'Network and sync', href: '/workshop/network', space: 'workshop', icon: 'globe', ready: true, settings: true },
	{ id: 'security', label: 'Security', href: '/workshop/security', space: 'workshop', icon: 'shield-check', ready: true, settings: true },
	{ id: 'backup', label: 'Backup', href: '/workshop/backup', space: 'workshop', icon: 'download', ready: true, settings: true }
];

/**
 * The entries to render, in the table's order: the ready ones, and the
 * componion's only while its switch is on.
 */
export function visibleDestinations(
	list: readonly Destination[],
	visibility: Visibility
): Destination[] {
	return list.filter((d) => d.ready && (d.id !== 'componion' || visibility.componion));
}

/** A space's home: its first ready destination. */
export function spaceHome(space: Space, list: readonly Destination[]): string {
	const home = list.find((d) => d.space === space && d.ready);
	return home ? home.href : '/';
}

/**
 * Whether `param` names a Workshop settings page: the last segment of the
 * URL of a ready Workshop destination that renders settings groups. The
 * route parameter matcher (src/params/workshopSection.ts) is this function.
 */
export function isWorkshopSection(param: string): boolean {
	if (!param) return false;
	return DESTINATIONS.some(
		(d) =>
			d.space === 'workshop' &&
			d.settings === true &&
			d.ready &&
			d.href === WORKSHOP_PREFIX + param
	);
}
