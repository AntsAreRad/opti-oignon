/**
 * The two spaces, and where switching between them goes.
 *
 * A page belongs to Workshop when its path is /workshop or under it, to Use
 * otherwise, and to neither when it is sign-in, registration or the
 * component gallery. The shell remembers the last route of each space;
 * switching goes back to it, or to the space's home when there is none or
 * the one remembered is not a page of that space on this origin.
 *
 * Pure functions, with no dependency.
 */

import type { Space } from './destinations';

/** The last route of each space, as a path and its query. */
export type LastRoutes = Partial<Record<Space, string>>;

/** The pages outside both spaces. */
const OUTSIDE = ['/login', '/register', '/dev'];

function pathOf(route: string): string {
	const end = route.search(/[?#]/);
	return end < 0 ? route : route.slice(0, end);
}

/** A route of this origin: a path, never a scheme, a host or a backslash. */
function sameOrigin(route: string): boolean {
	return (
		typeof route === 'string' &&
		route.startsWith('/') &&
		!route.startsWith('//') &&
		!route.includes('\\') &&
		![...route].some((c) => c.charCodeAt(0) < 32)
	);
}

/** The space a path belongs to, or null for a page outside both. */
export function spaceOf(pathname: string): Space | null {
	const path = pathOf(pathname);
	if (OUTSIDE.some((outside) => path === outside || path.startsWith(outside + '/'))) return null;
	return path === '/workshop' || path.startsWith('/workshop/') ? 'workshop' : 'use';
}

/**
 * A new record with `route` kept as its space's last route; the record given
 * is left as it was. A route outside both spaces, or not of this origin,
 * changes nothing.
 */
export function rememberRoute(last: LastRoutes, route: string): LastRoutes {
	if (!sameOrigin(route)) return { ...last };
	const space = spaceOf(route);
	if (space === null) return { ...last };
	return { ...last, [space]: route };
}

/**
 * Where switching to `space` goes: its last route, when that is a page of
 * this origin in that space, else its home.
 */
export function switchTarget(
	space: Space,
	last: LastRoutes,
	homes: Record<Space, string>
): string {
	const route = last[space];
	if (route !== undefined && sameOrigin(route) && spaceOf(route) === space) return route;
	return homes[space];
}
