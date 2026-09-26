/**
 * The chats index: the request it sends, and how it sorts what comes back.
 *
 * The server's listing (GET /api/conversations) answers two questions. With
 * words in `q` it searches titles and messages and returns at most `limit`
 * matches, newest first; it takes no offset then, so a search asks once for
 * SEARCH_LIMIT and the index says when that limit was reached. Without
 * words it lists every conversation, newest first, a page at a time with
 * `limit` and `offset`.
 *
 * The index groups what it shows by the day of its last change, the way a
 * notebook is read: today, yesterday, the previous seven days, earlier.
 *
 * Pure functions, with no dependency: the page calls them, and they run
 * under Node as they are.
 */

/** How many conversations a search asks for: the most the index shows for one. */
export const SEARCH_LIMIT = 200;

/** How many conversations one page of the plain listing holds. */
export const PAGE_SIZE = 50;

/** The query the listing endpoint is sent. */
export interface IndexParams {
	q?: string;
	limit: number;
	offset?: number;
}

/**
 * The listing request for `query` and page `page` (from 0). A search sends
 * its words and the search limit, never an offset; the plain listing sends
 * a page's size and where it starts. `removed` is how many conversations
 * the index deleted since its first page, which moved every later one up by
 * as many places.
 */
export function indexParams(query: string, page: number, removed = 0): IndexParams {
	const words = typeof query === 'string' ? query.trim() : '';
	if (words) return { q: words, limit: SEARCH_LIMIT };
	const start = Math.max(0, Math.floor(page)) * PAGE_SIZE;
	return { limit: PAGE_SIZE, offset: Math.max(0, start - Math.max(0, Math.floor(removed))) };
}

/** Whether a search for `query` that returned `count` conversations filled its limit. */
export function limitReached(query: string, count: number): boolean {
	const words = typeof query === 'string' ? query.trim() : '';
	return words !== '' && count >= SEARCH_LIMIT;
}

/** Whether a page of the plain listing that returned `count` may have one after it. */
export function morePages(query: string, count: number): boolean {
	const words = typeof query === 'string' ? query.trim() : '';
	return words === '' && count >= PAGE_SIZE;
}

/** What the index needs of a conversation to place it. */
export interface Dated {
	updated_at?: string | null;
	created_at?: string | null;
}

export type DayKey = 'today' | 'yesterday' | 'week' | 'earlier';

export interface DayGroup<T> {
	key: DayKey;
	label: string;
	items: T[];
}

const DAY_LABELS: Record<DayKey, string> = {
	today: 'Today',
	yesterday: 'Yesterday',
	week: 'Previous 7 days',
	earlier: 'Earlier'
};
const DAY_ORDER: DayKey[] = ['today', 'yesterday', 'week', 'earlier'];
const DAY_MS = 86_400_000;

/** The time a conversation last changed, in milliseconds, or null when unknown. */
function timeOf(item: Dated): number | null {
	const stamp = item.updated_at ?? item.created_at;
	if (!stamp) return null;
	const time = new Date(stamp).getTime();
	return Number.isNaN(time) ? null : time;
}

function startOfDay(now: Date): number {
	const day = new Date(now.getTime());
	day.setHours(0, 0, 0, 0);
	return day.getTime();
}

/** The day group a time falls in, seen from `now`. */
export function dayOf(time: number | null, now: Date): DayKey {
	if (time === null) return 'earlier';
	const today = startOfDay(now);
	if (time >= today) return 'today';
	if (time >= today - DAY_MS) return 'yesterday';
	if (time >= today - 7 * DAY_MS) return 'week';
	return 'earlier';
}

/** `items` in their day groups, each group in the order it was given. */
export function dayGroups<T extends Dated>(items: T[], now: Date): DayGroup<T>[] {
	const groups = new Map<DayKey, T[]>();
	for (const item of items) {
		const key = dayOf(timeOf(item), now);
		const list = groups.get(key) ?? [];
		list.push(item);
		groups.set(key, list);
	}
	return DAY_ORDER.filter((key) => groups.has(key)).map((key) => ({
		key,
		label: DAY_LABELS[key],
		items: groups.get(key) ?? []
	}));
}

/**
 * When a conversation last changed, as its row says it: the time of day
 * (on a 24-hour clock) today and yesterday, the weekday within the week,
 * the date before.
 */
export function whenOf(item: Dated, now: Date): string {
	const time = timeOf(item);
	if (time === null) return '';
	const date = new Date(time);
	const day = dayOf(time, now);
	if (day === 'today' || day === 'yesterday') {
		return date.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
	}
	if (day === 'week') return date.toLocaleDateString('en', { weekday: 'long' });
	const sameYear = date.getFullYear() === now.getFullYear();
	return date.toLocaleDateString('en', {
		day: 'numeric',
		month: 'short',
		...(sameYear ? {} : { year: 'numeric' })
	});
}

/** How many messages a conversation holds, in words. */
export function messagesOf(count: number | null | undefined): string {
	const n = typeof count === 'number' && count > 0 ? Math.floor(count) : 0;
	if (n === 0) return 'No messages yet';
	return n === 1 ? '1 message' : `${n} messages`;
}
