/**
 * An address the interface used to serve. Its content lives elsewhere now;
 * the page that holds it is decided by lib/nav/legacy.ts, which alone knows
 * the old addresses, and the reader is sent there permanently, the query
 * kept.
 */

import { redirect } from '@sveltejs/kit';
import { legacyTarget } from '$lib/nav/legacy';
import * as catalog from '$lib/settings/catalog';
import type { PageLoad } from './$types';

export const load: PageLoad = ({ url }) => {
	redirect(308, legacyTarget(url, catalog) ?? '/preferences');
};
