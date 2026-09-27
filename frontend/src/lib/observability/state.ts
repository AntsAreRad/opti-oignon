/**
 * The inference pipeline's overview, in words: whether telemetry is
 * collecting, how much the profiler has seen, how much history is kept.
 *
 * Each state is a sentence a reader can take in without a legend, never a
 * coloured mark alone. A part whose status could not be read (null) is
 * "Unavailable". The host also asks here whether a tab is shut: only when
 * its group's feature key reads false in the health feature map. Pure
 * functions, with no dependency: the Observability host
 * (components/panels/ObservabilityPanel.svelte) calls them, and they run
 * under Node as they are.
 */

/** What the overview reads of the telemetry pipeline's status. */
export interface TelemetryStatusLike {
	enabled?: boolean;
}

/** What the overview reads of the profiler's summary. */
export interface ProfilerSummaryLike {
	total_profiled_requests?: number;
}

/** What the overview reads of the history's status. */
export interface HistoryStatusLike {
	available?: boolean;
	total_stored?: number;
}

const UNAVAILABLE = 'Unavailable';

function counted(count: number, one: string, many: string): string {
	return `${count.toLocaleString('en-US')} ${count === 1 ? one : many}`;
}

/** "Collecting", "Off", or "Unavailable" when the status could not be read. */
export function telemetryState(stats: TelemetryStatusLike | null | undefined): string {
	if (!stats) return UNAVAILABLE;
	return stats.enabled ? 'Collecting' : 'Off';
}

/** "N requests profiled", "Nothing profiled yet", or "Unavailable". */
export function profilerState(summary: ProfilerSummaryLike | null | undefined): string {
	if (!summary) return UNAVAILABLE;
	const count = summary.total_profiled_requests ?? 0;
	return count > 0 ? `${counted(count, 'request', 'requests')} profiled` : 'Nothing profiled yet';
}

/** "N events stored", "History off", or "Unavailable". */
export function historyState(stats: HistoryStatusLike | null | undefined): string {
	if (!stats) return UNAVAILABLE;
	if (!stats.available) return 'History off';
	return `${counted(stats.total_stored ?? 0, 'event', 'events')} stored`;
}

/**
 * Whether a tab is shown as unavailable: its group names a feature key and
 * the feature map reads that key false. A group with no key, and a key the
 * map does not name, keep their tab open, so a panel says its own failure.
 */
export function tabUnavailable(
	feature: string | undefined,
	featureMap: Readonly<Record<string, boolean>>
): boolean {
	return !!feature && featureMap[feature] === false;
}
