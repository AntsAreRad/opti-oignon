/**
 * Which navigation entry is current, for a page's path.
 *
 * An entry is `active` on its own page, and `section` on a page under it
 * that no other destination claims more closely: in a conversation the
 * chats entry is the section, and the conversation's own recent row is
 * active. Home is active on itself alone, never by prefix. The sidebar and
 * the route announcer decide with these two functions and nothing else.
 *
 * Pure functions, with no dependency.
 */

import type { Destination } from './destinations';

export type ActiveState = 'active' | 'section' | 'none';

/** A path with no trailing slash, the root kept as it is. */
function normal(path: string): string {
	const trimmed = path.replace(/\/+$/, '');
	return trimmed === '' ? '/' : trimmed;
}

/** Whether `path` is `href` or a page under it (by whole segments). */
function under(path: string, href: string): boolean {
	return path === href || (href !== '/' && path.startsWith(href + '/'));
}

/**
 * The state of the entry linking to `href` on the page at `pathname`, where
 * `hrefs` are the destinations' links (a closer one takes the section).
 */
export function activeState(
	pathname: string,
	href: string,
	hrefs: readonly string[]
): ActiveState {
	const path = normal(pathname);
	const own = normal(href);
	if (path === own) return 'active';
	if (own === '/' || !under(path, own)) return 'none';
	const closer = hrefs
		.map(normal)
		.some((other) => other !== own && other.length > own.length && under(path, other));
	return closer ? 'none' : 'section';
}

/**
 * The destination the page at `pathname` belongs to: the one active on it,
 * else the one holding it as its section, else null.
 */
export function destinationFor<T extends Pick<Destination, 'href'>>(
	pathname: string,
	list: readonly T[]
): T | null {
	const hrefs = list.map((d) => d.href);
	const states = list.map((d) => activeState(pathname, d.href, hrefs));
	const exact = states.indexOf('active');
	if (exact >= 0) return list[exact];
	const section = states.indexOf('section');
	return section >= 0 ? list[section] : null;
}
