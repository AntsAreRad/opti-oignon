/**
 * A two-role highlighter for the reply's code blocks.
 *
 * Five grammars (Python, JavaScript and TypeScript, Rust, shell, SQL), each a
 * keyword list and one definition pattern. The code is cut into spans of
 * three roles: "keyword", "name" (the name a definition introduces) and
 * "plain" for everything else. Strings and comments stay plain, so a keyword
 * quoted or commented is not coloured, and so does a word after a dot (a
 * member such as node.type). The spans always concatenate back to the code,
 * character for character; a language outside the five is one plain span.
 *
 * It imports nothing, so it runs under Node's type stripping.
 */

export type Role = 'plain' | 'keyword' | 'name';

export const HIGHLIGHT_ROLES: readonly Role[] = ['plain', 'keyword', 'name'];

export interface Span {
	role: Role;
	text: string;
}

interface Grammar {
	keywords: ReadonlySet<string>;
	/** Keywords match in any letter case (SQL). */
	caseless: boolean;
	/** Global; each match ends with the defined name, its last capture. */
	definition: RegExp;
	lineComments: readonly string[];
	/** A line comment opens only at a line start or after white space. */
	commentAfterSpace: boolean;
	blockComment: readonly [string, string] | null;
	/** String delimiters, longest first. */
	quotes: readonly string[];
	/** Delimiters whose strings may run over a line end. */
	multiline: ReadonlySet<string>;
}

function words(list: string): ReadonlySet<string> {
	return new Set(list.split(/\s+/).filter((word) => word !== ''));
}

const PYTHON: Grammar = {
	keywords: words(`
		False None True and as assert async await break class continue def del
		elif else except finally for from global if import in is lambda nonlocal
		not or pass raise return try while with yield`),
	caseless: false,
	definition: /\b(?:def|class)\s+([A-Za-z_]\w*)/g,
	lineComments: ['#'],
	commentAfterSpace: false,
	blockComment: null,
	quotes: ['"""', "'''", '"', "'"],
	multiline: new Set(['"""', "'''"])
};

const SCRIPT: Grammar = {
	keywords: words(`
		abstract as async await break case catch class const continue debugger
		declare default delete do else enum export extends false finally for from
		function get if implements import in instanceof interface keyof let new
		null of private protected public readonly return set static super switch
		this throw true try type typeof undefined var void while with yield`),
	caseless: false,
	definition: /\b(?:function\*?|class|interface|type|enum|const|let|var)\s+([A-Za-z_$][\w$]*)/g,
	lineComments: ['//'],
	commentAfterSpace: false,
	blockComment: ['/*', '*/'],
	quotes: ['`', '"', "'"],
	multiline: new Set(['`'])
};

const RUST: Grammar = {
	keywords: words(`
		as async await break const continue crate dyn else enum extern false fn
		for if impl in let loop match mod move mut pub ref return self Self static
		struct super trait true type unsafe use where while`),
	caseless: false,
	definition: /\b(?:fn|struct|enum|trait|type|mod|const|static)\s+([A-Za-z_]\w*)/g,
	lineComments: ['//'],
	commentAfterSpace: false,
	blockComment: ['/*', '*/'],
	quotes: ['"'],
	multiline: new Set(['"'])
};

const SHELL: Grammar = {
	keywords: words(`
		case declare do done elif else esac exit export fi for function if in
		local readonly return select then time until while`),
	caseless: false,
	definition: /^[ \t]*(?:function[ \t]+([A-Za-z_]\w*)|([A-Za-z_]\w*)(?=[ \t]*\(\)))/gm,
	lineComments: ['#'],
	commentAfterSpace: true,
	blockComment: null,
	quotes: ['"', "'"],
	multiline: new Set(['"', "'"])
};

const SQL: Grammar = {
	keywords: words(`
		add all alter and as asc begin between by case check column commit
		constraint create database default delete desc distinct drop else end
		exists foreign from full group having if in index inner insert into is
		join key left like limit not null offset on or order outer primary
		references returning right rollback select set table then transaction
		union unique update values view when where with`),
	caseless: true,
	definition:
		/\bcreate\s+(?:or\s+replace\s+)?(?:(?:temp|temporary|unique)\s+)?(?:table|view|function|procedure|index|trigger|schema|database)\s+(?:if\s+not\s+exists\s+)?([A-Za-z_]\w*)/gi,
	lineComments: ['--'],
	commentAfterSpace: false,
	blockComment: ['/*', '*/'],
	quotes: ["'", '"'],
	multiline: new Set<string>()
};

const GRAMMARS: ReadonlyMap<string, Grammar> = new Map([
	['py', PYTHON],
	['python', PYTHON],
	['python3', PYTHON],
	['js', SCRIPT],
	['javascript', SCRIPT],
	['jsx', SCRIPT],
	['mjs', SCRIPT],
	['cjs', SCRIPT],
	['ts', SCRIPT],
	['typescript', SCRIPT],
	['tsx', SCRIPT],
	['rs', RUST],
	['rust', RUST],
	['sh', SHELL],
	['bash', SHELL],
	['shell', SHELL],
	['zsh', SHELL],
	['sql', SQL]
]);

/** Where each defined name starts, with its length. */
function definedNames(code: string, grammar: Grammar): Map<number, number> {
	const starts = new Map<number, number>();
	for (const match of code.matchAll(grammar.definition)) {
		const name = match.slice(1).filter((group) => group !== undefined).pop();
		if (name === undefined || match.index === undefined) continue;
		// A definition word after a dot is a member (schema.table), not one.
		if (match.index > 0 && code[match.index - 1] === '.') continue;
		starts.set(match.index + match[0].length - name.length, name.length);
	}
	return starts;
}

/** The end of the comment or string opening at ``at``, or ``at`` itself. */
function quotedEnd(code: string, at: number, grammar: Grammar): number {
	for (const marker of grammar.lineComments) {
		if (!code.startsWith(marker, at)) continue;
		if (grammar.commentAfterSpace && at > 0 && !/\s/.test(code[at - 1])) continue;
		const end = code.indexOf('\n', at);
		return end === -1 ? code.length : end;
	}
	if (grammar.blockComment !== null && code.startsWith(grammar.blockComment[0], at)) {
		const [open, close] = grammar.blockComment;
		const end = code.indexOf(close, at + open.length);
		return end === -1 ? code.length : end + close.length;
	}
	for (const quote of grammar.quotes) {
		if (!code.startsWith(quote, at)) continue;
		let index = at + quote.length;
		while (index < code.length) {
			if (code[index] === '\\') {
				index += 2;
				continue;
			}
			if (code.startsWith(quote, index)) return index + quote.length;
			if (code[index] === '\n' && !grammar.multiline.has(quote)) return index;
			index += 1;
		}
		return code.length;
	}
	return at;
}

const WORD_START = /[A-Za-z_$]/;
const WORD_PART = /[A-Za-z0-9_$]/;
const NUMBER_START = /[0-9]/;
const NUMBER_PART = /[A-Za-z0-9_.]/;

function runEnd(code: string, at: number, part: RegExp): number {
	let index = at + 1;
	while (index < code.length && part.test(code[index])) index += 1;
	return index;
}

/** The code as spans of three roles, concatenating back to the code. */
export function highlight(code: string, lang: string): Span[] {
	if (code === '') return [];
	const grammar = GRAMMARS.get(lang.trim().toLowerCase());
	if (grammar === undefined) return [{ role: 'plain', text: code }];
	const names = definedNames(code, grammar);
	const spans: Span[] = [];
	const push = (role: Role, piece: string): void => {
		const last = spans[spans.length - 1];
		if (last !== undefined && last.role === role) last.text += piece;
		else spans.push({ role, text: piece });
	};
	let at = 0;
	while (at < code.length) {
		const quoted = quotedEnd(code, at, grammar);
		if (quoted > at) {
			push('plain', code.slice(at, quoted));
			at = quoted;
			continue;
		}
		const char = code[at];
		if (NUMBER_START.test(char)) {
			const end = runEnd(code, at, NUMBER_PART);
			push('plain', code.slice(at, end));
			at = end;
			continue;
		}
		if (WORD_START.test(char)) {
			const end = runEnd(code, at, WORD_PART);
			const word = code.slice(at, end);
			const key = grammar.caseless ? word.toLowerCase() : word;
			// A word after a dot is a member (node.type, map.get), never a keyword.
			const member = at > 0 && code[at - 1] === '.';
			if (names.get(at) === word.length) push('name', word);
			else if (!member && grammar.keywords.has(key)) push('keyword', word);
			else push('plain', word);
			at = end;
			continue;
		}
		push('plain', char);
		at += 1;
	}
	return spans;
}
