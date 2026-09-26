// Stand-in for $app/navigation when a component is compiled for the server
// by tests/_frontend.ssr(). Nothing navigates on the server: every function
// is inert and returns what its SvelteKit namesake returns.

export async function goto() {}

export function afterNavigate() {}

export function beforeNavigate() {}

export function onNavigate() {}

export async function invalidate() {}

export async function invalidateAll() {}

export async function preloadCode() {}

export async function preloadData() {
	return { type: 'loaded', status: 200, data: {} };
}

export function pushState() {}

export function replaceState() {}

export function disableScrollHandling() {}
