/**
 * A reply's markdown as a closed tree of plain nodes.
 *
 * marked's lexer (never its parser or its renderer) turns the source into
 * tokens; toTree maps each token to a node of a closed set, NODE_KINDS, that
 * the reply's components render through text interpolation only. No token
 * reaches the page as markup:
 *
 *   - raw HTML, block or inline, is text with its exact characters (a block
 *     less the blank lines that end it, as a raw paragraph that keeps its
 *     line breaks and indentation on screen), and the text marked finds
 *     inside a raw element keeps its characters too;
 *   - a link is a link only when its destination parses as an absolute
 *     http:, https: or mailto: URL; any other link is its source text;
 *   - an image never loads: it is the text "Image: alt (url)";
 *   - references in text, link destinations and titles decode through the
 *     injected decoder (entities.ts); code spans, code blocks, escapes and
 *     raw HTML keep them literal;
 *   - a heading of depth d has level min(6, d + 2), under the reply's own
 *     heading;
 *   - a code block is open (still arriving) only while the reply streams,
 *     when it is the last block at every level and its fence is not closed;
 *     an indented block is always closed;
 *   - a token of a type the mapping does not know is its raw source as text;
 *   - a token nested deeper than MAX_DEPTH levels is its raw source as text
 *     (a raw paragraph in a block's place), so no node lies more than two
 *     levels below the cap, and a reply nested a thousand levels deep costs
 *     no deeper recursion than any other.
 *
 * Lexing is bounded too. marked's lexer takes super-linear time on some
 * inputs (a run of link openers with no white space is cubic; unclosed
 * emphasis and many reference definitions are quadratic), so a few kilobytes
 * can hold the page for seconds. lexMarkdown estimates the work first, from
 * marked's own block pass, and gives null for a reply over LEX_WORK_LIMIT,
 * or for one marked cannot lex (it throws): the reply is then shown as
 * plain text. The estimate is a model of the costs measured on marked
 * 15.0.11, not a bound on every path; frame.ts guards what it misses by
 * timing the lexes that run.
 *
 * It imports marked's Lexer and nothing else; the decoder is injected, so
 * the module runs under Node's type stripping with no bundler.
 */

import { Lexer } from 'marked';

export const NODE_KINDS = [
	'p',
	'heading',
	'code_block',
	'blockquote',
	'ol',
	'ul',
	'li',
	'table',
	'hr',
	'text',
	'strong',
	'em',
	'del',
	'code_inline',
	'br',
	'link'
] as const;

export type NodeKind = (typeof NODE_KINDS)[number];
export type Align = 'left' | 'center' | 'right' | null;

export interface TextNode {
	kind: 'text';
	text: string;
}
export interface SpanNode {
	kind: 'strong' | 'em' | 'del';
	children: InlineNode[];
}
export interface CodeInlineNode {
	kind: 'code_inline';
	text: string;
}
export interface BreakNode {
	kind: 'br';
}
export interface LinkNode {
	kind: 'link';
	href: string;
	title: string;
	children: InlineNode[];
}
export type InlineNode = TextNode | SpanNode | CodeInlineNode | BreakNode | LinkNode;

export interface ParagraphNode {
	kind: 'p';
	/** Present, and true, only for raw source shown as it is written (a block
	 * of HTML, a token the mapping does not know, a level past the depth
	 * cap): its line breaks and indentation are kept on screen. */
	raw?: true;
	children: InlineNode[];
}
export interface HeadingNode {
	kind: 'heading';
	level: 3 | 4 | 5 | 6;
	children: InlineNode[];
}
export interface CodeBlockNode {
	kind: 'code_block';
	/** The info string's first word, or '' when it is no plain label. */
	lang: string;
	/** The code, without fences or info string: what Copy writes. */
	code: string;
	/** False only while the block is still arriving. */
	closed: boolean;
}
export interface BlockquoteNode {
	kind: 'blockquote';
	children: BlockNode[];
}
export interface ListItemNode {
	kind: 'li';
	task: boolean;
	checked: boolean;
	children: MdNode[];
}
export interface OrderedListNode {
	kind: 'ol';
	start: number;
	children: ListItemNode[];
}
export interface UnorderedListNode {
	kind: 'ul';
	children: ListItemNode[];
}
export interface TableNode {
	kind: 'table';
	align: Align[];
	header: InlineNode[][];
	rows: InlineNode[][][];
}
export interface RuleNode {
	kind: 'hr';
}
export type BlockNode =
	| ParagraphNode
	| HeadingNode
	| CodeBlockNode
	| BlockquoteNode
	| OrderedListNode
	| UnorderedListNode
	| TableNode
	| RuleNode;
export type MdNode = BlockNode | ListItemNode | InlineNode;

export interface TreeOptions {
	/** True while the reply is still arriving. */
	streaming: boolean;
	/** Decodes character references in text (entities.ts). */
	decode: (text: string) => string;
}

/** A token as marked hands it over: every field is read defensively. */
interface Token {
	type?: unknown;
	raw?: unknown;
	text?: unknown;
	tokens?: unknown;
	escaped?: unknown;
	depth?: unknown;
	lang?: unknown;
	codeBlockStyle?: unknown;
	href?: unknown;
	title?: unknown;
	ordered?: unknown;
	start?: unknown;
	loose?: unknown;
	items?: unknown;
	task?: unknown;
	checked?: unknown;
	header?: unknown;
	rows?: unknown;
	align?: unknown;
}

const OPTIONS = { gfm: true, breaks: true };

/** The deepest a node nests; a token deeper than this is its source as text. */
export const MAX_DEPTH = 32;

/** The estimated work above which a reply is not lexed but shown as text.
 * The unit is about a nanosecond of lexing on the machine the costs were
 * measured on: the limit stands for some 40 ms. */
export const LEX_WORK_LIMIT = 4e7;

/**
 * Lexes a reply: GitHub-flavoured, a single newline is a line break. Gives
 * null when the estimated work is over LEX_WORK_LIMIT or marked cannot lex
 * the source (it throws); the reply is then shown as plain text. A source
 * that is not a string lexes as an empty one.
 */
export function lexMarkdown(source: string): unknown[] | null {
	const text = typeof source === 'string' ? source : '';
	try {
		if (lexWork(text) > LEX_WORK_LIMIT) return null;
		return new Lexer(OPTIONS).lex(text);
	} catch {
		return null;
	}
}

/** Maps lexed tokens to the closed node set. */
export function toTree(tokens: readonly unknown[], options: TreeOptions): BlockNode[] {
	return blocks(asTokens(tokens), options, true, 1);
}

const LINK_PROTOCOLS: ReadonlySet<string> = new Set(['http:', 'https:', 'mailto:']);

/** The destination as an absolute http:, https: or mailto: URL, or null. */
export function safeHref(href: string): string | null {
	let url: URL;
	try {
		url = new URL(href);
	} catch {
		return null;
	}
	return LINK_PROTOCOLS.has(url.protocol) ? url.href : null;
}

const LABEL = /^[A-Za-z0-9+#.-]{0,20}$/;

/** A code block's label: the info string's first word, or '' when that word
 * is longer than 20 characters or holds anything but [A-Za-z0-9+#.-]. */
export function codeLabel(info: string): string {
	const word = info.trim().split(/\s+/, 1)[0] ?? '';
	return LABEL.test(word) ? word : '';
}

function asTokens(value: unknown): Token[] {
	if (!Array.isArray(value)) return [];
	return value.filter((item): item is Token => typeof item === 'object' && item !== null);
}

function str(value: unknown): string {
	return typeof value === 'string' ? value : '';
}

function text(value: string): TextNode {
	return { kind: 'text', text: value };
}

/** The value less the newlines that end it, in time linear in their run. */
function trimEndNewlines(value: string): string {
	let end = value.length;
	while (end > 0 && value.charCodeAt(end - 1) === 10) end -= 1;
	return end === value.length ? value : value.slice(0, end);
}

/** A token shown as its source, in a raw paragraph; nothing when it has none. */
function rawBlock(token: Token): ParagraphNode | null {
	const raw = trimEndNewlines(str(token.raw));
	return raw === '' ? null : { kind: 'p', raw: true, children: [text(raw)] };
}

function blocks(list: Token[], options: TreeOptions, tail: boolean, depth: number): BlockNode[] {
	const content = list.filter((token) => token.type !== 'space');
	const out: BlockNode[] = [];
	content.forEach((token, index) => {
		const node = block(token, options, tail && index === content.length - 1, depth);
		if (node !== null) out.push(node);
	});
	return out;
}

function block(token: Token, options: TreeOptions, last: boolean, depth: number): BlockNode | null {
	if (depth > MAX_DEPTH) return rawBlock(token);
	switch (token.type) {
		case 'paragraph':
			return { kind: 'p', children: inlines(token.tokens, options, depth + 1) };
		case 'heading':
			return {
				kind: 'heading',
				level: headingLevel(token.depth),
				children: inlines(token.tokens, options, depth + 1)
			};
		case 'code':
			return codeBlock(token, options, last);
		case 'blockquote':
			return {
				kind: 'blockquote',
				children: blocks(asTokens(token.tokens), options, last, depth + 1)
			};
		case 'list':
			// Its items sit one level below it: past the cap, the list is its source.
			return depth + 1 > MAX_DEPTH ? rawBlock(token) : list(token, options, last, depth);
		case 'table':
			return table(token, options, depth);
		case 'hr':
			return { kind: 'hr' };
		case 'html':
			return rawBlock(token);
		case 'text':
			return { kind: 'p', children: textInlines(token, options, depth + 1) };
		default:
			return rawBlock(token);
	}
}

function headingLevel(depth: unknown): HeadingNode['level'] {
	const d = typeof depth === 'number' && Number.isInteger(depth) ? depth : 1;
	if (d <= 1) return 3;
	if (d === 2) return 4;
	if (d === 3) return 5;
	return 6;
}

const FENCE_OPEN = /^ {0,3}(`{3,}|~{3,})/;
const FENCE_CLOSE = /^ {0,3}(`{3,}|~{3,})[ \t]*$/;

function fenceClosed(raw: string): boolean {
	const lines = trimEndNewlines(raw).split('\n');
	const opening = FENCE_OPEN.exec(lines[0] ?? '');
	if (opening === null) return true;
	if (lines.length < 2) return false;
	const closing = FENCE_CLOSE.exec(lines[lines.length - 1] ?? '');
	return (
		closing !== null &&
		closing[1][0] === opening[1][0] &&
		closing[1].length >= opening[1].length
	);
}

function codeBlock(token: Token, options: TreeOptions, last: boolean): CodeBlockNode {
	const indented = token.codeBlockStyle === 'indented';
	const open = !indented && options.streaming && last && !fenceClosed(str(token.raw));
	return {
		kind: 'code_block',
		lang: indented ? '' : codeLabel(str(token.lang)),
		code: str(token.text),
		closed: !open
	};
}

function list(
	token: Token,
	options: TreeOptions,
	last: boolean,
	depth: number
): OrderedListNode | UnorderedListNode {
	const items = asTokens(token.items);
	const loose = token.loose === true;
	const children = items.map((item, index) =>
		listItem(item, options, last && index === items.length - 1, loose, depth + 1)
	);
	if (token.ordered === true) {
		const start = typeof token.start === 'number' && Number.isInteger(token.start) ? token.start : 1;
		return { kind: 'ol', start, children };
	}
	return { kind: 'ul', children };
}

function listItem(
	item: Token,
	options: TreeOptions,
	last: boolean,
	loose: boolean,
	depth: number
): ListItemNode {
	const content = asTokens(item.tokens).filter((token) => token.type !== 'space');
	const tight = !loose && item.loose !== true;
	const children: MdNode[] = [];
	content.forEach((token, index) => {
		if (tight && token.type === 'text') {
			children.push(...textInlines(token, options, depth + 1));
			return;
		}
		const node = block(token, options, last && index === content.length - 1, depth + 1);
		if (node !== null) children.push(node);
	});
	const task = item.task === true;
	return { kind: 'li', task, checked: task && item.checked === true, children };
}

function table(token: Token, options: TreeOptions, depth: number): TableNode {
	const cells = (row: unknown): InlineNode[][] =>
		asTokens(row).map((cell) => inlines(cell.tokens, options, depth + 1));
	const rows = Array.isArray(token.rows) ? token.rows : [];
	const align = Array.isArray(token.align) ? token.align : [];
	return {
		kind: 'table',
		align: align.map(alignment),
		header: cells(token.header),
		rows: rows.map(cells)
	};
}

function alignment(value: unknown): Align {
	return value === 'left' || value === 'center' || value === 'right' ? value : null;
}

function textInlines(token: Token, options: TreeOptions, depth: number): InlineNode[] {
	if (Array.isArray(token.tokens)) return inlines(token.tokens, options, depth);
	return [textOf(token, options)];
}

function textOf(token: Token, options: TreeOptions): TextNode {
	const value = str(token.text);
	// marked marks the text it finds inside a raw HTML element as escaped:
	// it belongs to the raw run and keeps its characters.
	return text(token.escaped === true ? value : options.decode(value));
}

function inlines(value: unknown, options: TreeOptions, depth: number): InlineNode[] {
	return asTokens(value).flatMap((token) => inline(token, options, depth));
}

function inline(token: Token, options: TreeOptions, depth: number): InlineNode[] {
	if (depth > MAX_DEPTH) {
		const raw = str(token.raw);
		return raw === '' ? [] : [text(raw)];
	}
	switch (token.type) {
		case 'text':
			return textInlines(token, options, depth + 1);
		case 'escape':
			return [text(str(token.text))];
		case 'html':
			return [text(str(token.raw))];
		case 'strong':
			return [{ kind: 'strong', children: inlines(token.tokens, options, depth + 1) }];
		case 'em':
			return [{ kind: 'em', children: inlines(token.tokens, options, depth + 1) }];
		case 'del':
			return [{ kind: 'del', children: inlines(token.tokens, options, depth + 1) }];
		case 'codespan':
			return [{ kind: 'code_inline', text: str(token.text) }];
		case 'br':
			return [{ kind: 'br' }];
		case 'link':
			return [link(token, options, depth)];
		case 'image':
			return [text(imageText(token, options))];
		default: {
			const raw = str(token.raw);
			return raw === '' ? [] : [text(raw)];
		}
	}
}

function link(token: Token, options: TreeOptions, depth: number): InlineNode {
	const href = safeHref(options.decode(str(token.href)));
	if (href === null) return text(str(token.raw));
	const children = inlines(token.tokens, options, depth + 1);
	return {
		kind: 'link',
		href,
		title: options.decode(str(token.title)),
		children: children.length > 0 ? children : [text(href)]
	};
}

function imageText(token: Token, options: TreeOptions): string {
	const alt = options.decode(str(token.text));
	const url = options.decode(str(token.href));
	return alt === '' ? `Image: (${url})` : `Image: ${alt} (${url})`;
}

// ---------------------------------------------------------------------------
// The work estimate
// ---------------------------------------------------------------------------
// Costs per unit of work, fitted on marked 15.0.11 (container, Node 22): an
// opener of '*' or '_' that is not closed reads every later run of its
// character, about 90 each, a '~' one about 4; a long run is lexed level by
// level, about 4 per character of its reach and level; a link attempt reads
// its destination and, at each '(' , '"' or "'" in it, a title on to that
// title's closer, about 1.6 per character; each reference use and each run
// of inline text looks through every definition, about 50 each.
const EMPHASIS_COST = 90;
const STRIKE_COST = 4;
const NESTING_COST = 4;
const LINK_COST = 1.6;
const REFERENCE_COST = 50;

/**
 * The work marked's inline pass would do on the source, estimated from the
 * block pass: the texts marked lexes inline (paragraphs, headings, table
 * cells, the text of list items) are read one by one for the three costs it
 * is super-linear in, unclosed emphasis, link attempts and reference
 * lookups. Linear in the source.
 */
export function lexWork(source: string): number {
	const lexer = new Lexer(OPTIONS);
	const tokens = lexer.blockTokens(source.replace(/\r\n|\r/g, '\n'));
	const definitions = Object.keys(lexer.tokens.links ?? {}).length;
	let work = 0;
	let units = 0;
	let brackets = 0;
	for (const unit of inlineSources(tokens)) {
		units += 1;
		work += emphasisWork(unit) + linkWork(unit);
		if (definitions > 0) {
			for (let i = 0; i < unit.length; i += 1) if (unit.charCodeAt(i) === 91) brackets += 1;
		}
	}
	return work + REFERENCE_COST * definitions * (units + brackets);
}

/** The texts the block pass queued for the inline pass: every token with a
 * text and an empty token list (the inline pass has not filled it yet). */
function inlineSources(tokens: unknown): string[] {
	const out: string[] = [];
	const pending: unknown[] = [tokens];
	while (pending.length > 0) {
		const value = pending.pop();
		if (Array.isArray(value)) {
			for (let i = value.length - 1; i >= 0; i -= 1) pending.push(value[i]);
			continue;
		}
		if (typeof value !== 'object' || value === null) continue;
		const token = value as Token;
		if (typeof token.text === 'string' && Array.isArray(token.tokens) && token.tokens.length === 0) {
			out.push(token.text);
		}
		pending.push(token.rows, token.header, token.items, token.tokens);
	}
	return out;
}

const PUNCTUATION = /[\p{P}\p{S}]/u;
const WHITE = /\s/;

function isSpace(code: number): boolean {
	return code === 32 || code === 9 || code === 10 || code === 13 || code === 12 || code === 11;
}

/** Each run of '*', '_' or '~' that can open emphasis costs a step for
 * each later run of the same character marked reads before the next one
 * that can close it (or the end of the text); a run longer than one also
 * costs its length times that distance, the depth its nested levels are
 * lexed to. */
function emphasisWork(unit: string): number {
	const n = unit.length;
	const runs: { code: number; start: number; length: number; open: boolean; close: boolean }[] = [];
	const counts = new Map<number, number>([
		[42, 0],
		[95, 0],
		[126, 0]
	]);
	let i = 0;
	while (i < n) {
		const code = unit.charCodeAt(i);
		if (code !== 42 && code !== 95 && code !== 126) {
			i += 1;
			continue;
		}
		let j = i + 1;
		while (j < n && unit.charCodeAt(j) === code) j += 1;
		const before = i > 0 ? unit[i - 1] : ' ';
		const after = j < n ? unit[j] : ' ';
		const spaceBefore = WHITE.test(before);
		const spaceAfter = WHITE.test(after);
		const punctBefore = PUNCTUATION.test(before);
		const punctAfter = PUNCTUATION.test(after);
		const left = !spaceAfter && (!punctAfter || spaceBefore || punctBefore);
		const right = !spaceBefore && (!punctBefore || spaceAfter || punctAfter);
		const underscore = code === 95;
		runs.push({
			code,
			start: i,
			length: j - i,
			open: underscore ? left && (!right || punctBefore) : left,
			close: underscore ? right && (!left || punctAfter) : right
		});
		counts.set(code, (counts.get(code) ?? 0) + 1);
		i = j;
	}
	let work = 0;
	const ordinal = new Map<number, number>(counts);
	const closerOrdinal = new Map<number, number>(counts);
	const closerStart = new Map<number, number>([
		[42, n],
		[95, n],
		[126, n]
	]);
	for (let k = runs.length - 1; k >= 0; k -= 1) {
		const run = runs[k];
		const own = (ordinal.get(run.code) ?? 0) - 1;
		ordinal.set(run.code, own);
		if (run.open) {
			const steps = (closerOrdinal.get(run.code) ?? own) - own;
			const distance = (closerStart.get(run.code) ?? n) - run.start;
			const step = run.code === 126 ? STRIKE_COST : EMPHASIS_COST;
			work += step * steps + NESTING_COST * (run.length - 1) * distance;
		}
		if (run.close) {
			closerOrdinal.set(run.code, own);
			closerStart.set(run.code, run.start);
		}
	}
	return work;
}

/** Each '[' costs a link attempt at the next ']('. An attempt reads the run
 * of the destination (to the next white space) and, at each '(' , '"' or "'"
 * in it or just after it, a title on to that title's closing character. */
function linkWork(unit: string): number {
	const n = unit.length;
	const weight = new Float64Array(n);
	const runEnd = new Int32Array(n + 1);
	const solidFrom = new Int32Array(n + 1);
	const linkAt = new Int32Array(n + 1);
	runEnd[n] = n;
	solidFrom[n] = n;
	linkAt[n] = -1;
	let nextParen = n;
	let nextDouble = n;
	let nextSingle = n;
	for (let i = n - 1; i >= 0; i -= 1) {
		const code = unit.charCodeAt(i);
		if (code === 40) weight[i] = nextParen - i;
		else if (code === 34) weight[i] = nextDouble - i;
		else if (code === 39) weight[i] = nextSingle - i;
		if (code === 41) nextParen = i;
		else if (code === 34) nextDouble = i;
		else if (code === 39) nextSingle = i;
		const space = isSpace(code);
		runEnd[i] = space ? i : runEnd[i + 1];
		solidFrom[i] = space ? solidFrom[i + 1] : i;
		linkAt[i] = code === 93 && unit.charCodeAt(i + 1) === 40 ? i : linkAt[i + 1];
	}
	const titles = new Float64Array(n + 1);
	for (let i = 0; i < n; i += 1) titles[i + 1] = titles[i] + weight[i];
	let work = 0;
	for (let i = 0; i < n; i += 1) {
		if (unit.charCodeAt(i) !== 91) continue;
		const at = linkAt[i];
		if (at < 0) continue;
		const start = Math.min(at + 2, n);
		const end = runEnd[start];
		const after = solidFrom[end];
		work += end - start + (titles[end] - titles[start]) + (after < n ? weight[after] : 0);
	}
	return LINK_COST * work;
}
