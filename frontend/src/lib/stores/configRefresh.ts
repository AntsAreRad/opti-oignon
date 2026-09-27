/**
 * What the interface reads again after the server's configuration changed:
 * a system preset applied, or the configuration reloaded from disk.
 *
 * A preset rewrites the server's configuration and its default model. The
 * interface used to reload the whole page to see it; it now reads again
 * what the change touches: the models, the presets and the effective model
 * the chat offers (chatOptions), the feature map, whose cache is dropped
 * first, and the backends. Once all of them have settled, the epoch moves,
 * and any view that reads the server's switches once (the chat's control
 * bar) reads them again. A view that is not on screen reads afresh when it
 * next mounts.
 *
 * The refreshers are imported when a refresh runs, so the root layout's
 * first load does not carry them.
 */

import { writable, type Readable } from 'svelte/store';

const epoch = writable(0);

/** Moves once each time a refresh after a configuration change has settled. */
export const configEpoch: Readable<number> = { subscribe: epoch.subscribe };

/** Reads again what a configuration change touches, then moves the epoch. */
export async function refreshAfterConfigChange(): Promise<void> {
	const [options, features, backends] = await Promise.all([
		import('$lib/stores/chatOptions'),
		import('$lib/api/featureCheck'),
		import('$lib/stores/backendStatus')
	]);
	features.invalidateFeatureCache();
	await Promise.allSettled([
		options.loadOptions(),
		features.getFeatureMap(),
		backends.refreshBackendStatus()
	]);
	epoch.update((count) => count + 1);
}
