// Stand-in for $app/stores when a component is compiled for the server by
// tests/_frontend.ssr(). Each store honours the Svelte store contract by hand
// (subscribe calls back at once and returns an unsubscribe), so the stub has
// no dependency to resolve.

function readable(value) {
	return {
		subscribe(run) {
			run(value);
			return () => {};
		}
	};
}

export const page = readable({
	url: new URL('http://localhost/'),
	params: {},
	route: { id: null },
	status: 200,
	error: null,
	data: {},
	form: null,
	state: {}
});

export const navigating = readable(null);

export const updated = { ...readable(false), check: async () => false };

export function getStores() {
	return { page, navigating, updated };
}
