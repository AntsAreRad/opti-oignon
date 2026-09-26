/**
 * Character references in reply text, decoded through a fixed table.
 *
 * marked's lexer keeps references verbatim in its text tokens, and the reply
 * shows text through Svelte's interpolation, which escapes every character:
 * without decoding, "a &lt; b" would show its ampersand and its name. This
 * decodes once, in a single pass, so "&amp;lt;" becomes the text "&lt;" and
 * never "<".
 *
 * The named references are a fixed table; a name outside it stays as
 * written. Every numeric reference decodes (decimal up to seven digits,
 * hexadecimal up to six), and one that names no character (zero, a
 * surrogate, or beyond the last code point) becomes the replacement
 * character, as CommonMark does. Code spans, code blocks and raw HTML never
 * pass through here: tree.ts keeps them literal.
 *
 * It imports nothing, so it runs under Node's type stripping.
 */

const NAMED: ReadonlyMap<string, string> = new Map([
	['lt', '<'],
	['gt', '>'],
	['amp', '&'],
	['quot', '"'],
	['nbsp', String.fromCharCode(0xa0)],
	['mdash', String.fromCharCode(0x2014)],
	['ndash', String.fromCharCode(0x2013)],
	['hellip', String.fromCharCode(0x2026)],
	['copy', String.fromCharCode(0xa9)]
]);

const REFERENCE = /&(?:#([0-9]{1,7})|#[xX]([0-9a-fA-F]{1,6})|([A-Za-z][A-Za-z0-9]{0,31}));/g;

const REPLACEMENT = 0xfffd;
const LAST_CODE_POINT = 0x10ffff;

function names(point: number): boolean {
	return point > 0 && point <= LAST_CODE_POINT && !(point >= 0xd800 && point <= 0xdfff);
}

/** Decodes the named references of the table and every numeric reference. */
export function decodeEntities(text: string): string {
	if (!text.includes('&')) return text;
	return text.replace(
		REFERENCE,
		(
			whole: string,
			decimal: string | undefined,
			hex: string | undefined,
			name: string | undefined
		): string => {
			if (name !== undefined) return NAMED.get(name) ?? whole;
			const point = decimal !== undefined ? parseInt(decimal, 10) : parseInt(hex ?? '', 16);
			return String.fromCodePoint(names(point) ? point : REPLACEMENT);
		}
	);
}
