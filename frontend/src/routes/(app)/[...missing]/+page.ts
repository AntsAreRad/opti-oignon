/**
 * An address no page serves: a mistyped one, a Workshop page that does not
 * exist, or a stale link. Answered inside the shell, with a 404 the shell's
 * error page draws, so the sidebar and Stop all stay where they are; an
 * address SvelteKit cannot match would otherwise get its own error page,
 * outside the shell.
 */

import { error } from '@sveltejs/kit';
import type { PageLoad } from './$types';

export const load: PageLoad = () => {
	error(404, 'This page does not exist');
};
