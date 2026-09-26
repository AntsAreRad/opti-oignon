/**
 * The last route of each space, so switching space goes back where the
 * reader was (lib/nav/space.ts decides; the shell remembers each route it
 * shows). Kept for the tab's life, in memory.
 */

import { writable } from 'svelte/store';
import type { LastRoutes } from '$lib/nav/space';

export const lastRoutes = writable<LastRoutes>({});
