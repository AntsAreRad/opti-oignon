/**
 * The benchmarks page's tab, in its address.
 *
 * The page is one row of tabs over the quality evaluation's seven
 * sections. The tab it shows is the address's `tab`, so a reload, Back and
 * a shared link land where the reader was; any other value, or none, is
 * Run. Writing a tab keeps `run` (the run whose detail is open) and every
 * other key. A History run links with `tab=history`, so its detail opens
 * over History and closes onto it.
 *
 * Pure functions, with no dependency: the benchmarks page calls them, and
 * they run under Node as they are.
 */

/** The seven tabs, in the order the row shows them. */
export const BENCHMARK_TABS = ['run', 'leaderboard', 'h2h', 'trends', 'compare', 'history', 'profiles'] as const;

export type BenchmarkTab = (typeof BENCHMARK_TABS)[number];

/** The tab an address names: its `tab` when it is one of the seven, else Run. */
export function benchmarkTab(url: URL): BenchmarkTab {
	const named = url.searchParams.get('tab') ?? '';
	return (BENCHMARK_TABS as readonly string[]).includes(named) ? (named as BenchmarkTab) : 'run';
}

/** The address that shows `tab`: `tab` set, `run` and every other key kept. The URL given is never changed. */
export function tabAddress(url: URL, tab: BenchmarkTab): string {
	const next = new URL(url.href);
	next.searchParams.set('tab', tab);
	return `${next.pathname}${next.search}`;
}
