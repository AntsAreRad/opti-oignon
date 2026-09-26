/**
 * The root sends its reader to the chats index. The redirect is temporary:
 * the root gets a page of its own later, and a permanent one would outlive
 * it in the browser's cache.
 */

import { redirect } from '@sveltejs/kit';
import type { PageLoad } from './$types';

export const load: PageLoad = () => {
	redirect(307, '/chat');
};
