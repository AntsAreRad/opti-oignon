/**
 * The settings search: every group of the catalog, in both spaces, found by
 * its title, its description and its synonyms, by the name of the page that
 * holds it ("Preferences" and its section, "Workshop" and its page), and by
 * the name of the section of the old settings page it sat in (a reader who
 * remembers "Plugins & Extensions" still finds them), and linked to the page
 * that holds it.
 *
 * A search from Preferences finds a Workshop group, and one from a Workshop
 * page finds a Preferences group: the index is the whole catalog, whichever
 * page asks. A group is linked to its page with its id in `g`; an embedded
 * group, to its host's page with the host's id; a retired group has no page
 * and is never found. Every word of a search must be found in a group, in
 * any case.
 *
 * Pure functions, with no dependency: the settings hub calls them with the
 * catalog (lib/settings/catalog.ts), the destination table
 * (lib/nav/destinations.ts), the Preferences sections and the old page's
 * sections, and they run under Node as they are.
 */

/** What the search reads of a group, and where the group lives. */
export interface SearchableGroup {
	id: string;
	title: string;
	description: string;
	synonyms?: readonly string[];
	space?: 'use' | 'workshop';
	section?: string;
	retired?: string;
	embeddedIn?: string;
	/** The old settings section a group rendered inline sat in. */
	sectionId?: string;
}

/** A section of the old settings page: its name, and the groups it listed. */
export interface FormerSection {
	id: string;
	label: string;
	groups?: readonly { id: string }[];
}

/** What the search reads of a destination. */
export interface SearchDestination {
	id: string;
	label: string;
	href: string;
}

/** A group the search can find. */
export interface SettingsHit {
	id: string;
	title: string;
	description: string;
	/** The page that holds the group, with the group to open there in `g`. */
	href: string;
	/** Where that page is, in words: "Preferences, Account", "Workshop, Security". */
	where: string;
	/** The name of the old settings section that listed it, or '' when none did. */
	former: string;
	/** The group's words, lower-cased: title, description, synonyms, where it sits and sat. */
	haystack: string;
}

const WORKSHOP_PREFIX = '/workshop/';

/**
 * Every group of `groups` that has a page, linked to it. The page of a
 * Preferences group is the Preferences destination; the page of a Workshop
 * group is the destination whose URL ends with its section. A group is also
 * found by that page's name, and by the name of the old section
 * (`formerSections`) that listed it or whose introduction rendered it.
 */
export function settingsIndex(
	groups: readonly SearchableGroup[],
	destinations: readonly SearchDestination[],
	preferencesSections: readonly { id: string; label: string }[],
	formerSections: readonly FormerSection[] = []
): SettingsHit[] {
	const byId = new Map(groups.map((group) => [group.id, group]));
	const preferences = destinations.find((d) => d.id === 'preferences');
	const hits: SettingsHit[] = [];
	for (const group of groups) {
		const host = group.embeddedIn ? byId.get(group.embeddedIn) : group;
		if (!host || host.retired || host.embeddedIn || !host.section) continue;
		let page: string | null = null;
		let where = '';
		if (host.space === 'use' && preferences) {
			const section = preferencesSections.find((s) => s.id === host.section);
			if (!section) continue;
			page = preferences.href;
			where = `${preferences.label}, ${section.label}`;
		} else if (host.space === 'workshop') {
			const destination = destinations.find((d) => d.href === WORKSHOP_PREFIX + host.section);
			if (!destination) continue;
			page = destination.href;
			where = `Workshop, ${destination.label}`;
		}
		if (!page) continue;
		const former = formerSections.find(
			(s) => s.id === group.sectionId || (s.groups ?? []).some((listed) => listed.id === group.id)
		);
		hits.push({
			id: group.id,
			title: group.title,
			description: group.description,
			href: `${page}?g=${encodeURIComponent(host.id)}`,
			where,
			former: former?.label ?? '',
			haystack: [group.title, group.description, ...(group.synonyms ?? []), where, former?.label ?? '']
				.join(' ')
				.toLowerCase()
		});
	}
	return hits;
}

/** The groups of `index` that hold every word of `words`, in the index's order. */
export function searchSettings(index: readonly SettingsHit[], words: string): SettingsHit[] {
	const terms = (typeof words === 'string' ? words : '').trim().toLowerCase().split(/\s+/).filter(Boolean);
	if (terms.length === 0) return [];
	return index.filter((hit) => terms.every((term) => hit.haystack.includes(term)));
}
