/**
 * The rules of the "Your data" panel, dependency-free so a contract runs them under Node.
 *
 * The panel offers the export first, then shows what a wipe removes and leaves, lets the user name the content
 * removed only on request (the notes), and asks two typed confirmations: the wipe's own words, and the phrase the
 * preview gave for what is named -- the server checks that phrase again against what it counts at that moment.
 */

/** What the user types to confirm the wipe itself. */
export const WIPE_WORDS = 'delete my data';

/** The query a wipe or its preview sends: each named word once, sorted, then the phrase when one is given. */
export function wipeQuery(remove: string[], confirm?: string): string {
	const words = Array.from(new Set(remove.map((w) => w.trim()).filter((w) => w !== ''))).sort();
	const parts = words.map((w) => `remove=${encodeURIComponent(w)}`);
	if (confirm !== undefined && words.length > 0) parts.push(`confirm=${encodeURIComponent(confirm.trim())}`);
	return parts.length > 0 ? `?${parts.join('&')}` : '';
}

/** Whether what the user typed is the expected words, exactly (spaces around them ignored). */
export function typedMatches(typed: string, expected: string | undefined): boolean {
	return expected !== undefined && expected !== '' && typed.trim() === expected;
}

export interface WipeGate {
	/** The export was downloaded, or the user chose to go on without a copy. */
	exportOffered: boolean;
	/** What the user typed for the wipe's own words. */
	typedWipe: string;
	/** The content named for removal (the notes), as the preview counted it. */
	named: string[];
	/** The phrase the preview gave for what is named. */
	phrase: string | undefined;
	/** What the user typed for that phrase. */
	typedPhrase: string;
	/** A preview or a wipe is running. */
	busy: boolean;
}

/** Whether the wipe may be asked: the export offered first, the wipe's words typed, and the preview's phrase typed
 * when content is named. */
export function canWipe(gate: WipeGate): boolean {
	if (gate.busy || !gate.exportOffered || !typedMatches(gate.typedWipe, WIPE_WORDS)) return false;
	return gate.named.length === 0 || typedMatches(gate.typedPhrase, gate.phrase);
}

export interface WipeAnswerLines {
	/** What could not be removed: every line of the answer's `not_covered` that is not a store the preview said
	 * stays. Shown first. */
	failed: string[];
	/** What stays, as the preview said. */
	left: string[];
	/** What went, counted where the answer counts it. */
	removed: string[];
}

/** The wipe's answer, failures first. */
export function answerLines(
	notCovered: string[],
	expectedLeft: string[],
	counts: Record<string, number>
): WipeAnswerLines {
	const known = new Set(expectedLeft);
	return {
		failed: notCovered.filter((line) => !known.has(line)),
		left: notCovered.filter((line) => known.has(line)),
		removed: Object.entries(counts)
			.filter(([, n]) => typeof n === 'number' && n > 0)
			.map(([what, n]) => `${what.replace(/_/g, ' ')}: ${n}`)
	};
}
