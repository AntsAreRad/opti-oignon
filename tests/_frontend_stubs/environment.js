// Stand-in for $app/environment when a component is compiled for the server
// by tests/_frontend.ssr(): rendering happens outside a browser, at no build.

export const browser = false;

export const building = false;

export const dev = false;

export const version = 'server-rendering-contract';
