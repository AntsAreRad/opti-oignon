/**
 * A long finished reply, collapsed by whole top-level blocks; plain text,
 * collapsed by whole lines.
 *
 * A reply whose blocks weigh more than the threshold shows its first blocks
 * while they fit the kept budget (always at least the first one, whole),
 * and says how many it hides; the hidden blocks are not in the output at
 * all, so nothing renders them. A block is never cut: a code block or a
 * table is shown whole or not at all. A block's weight approximates the
 * lines of source it came from: a paragraph or a heading is one line and
 * one more per line break, a code block its lines and its two fences, a
 * table its rows and its header, a quote or a list the sum of what it
 * holds. The first block is always shown whole, so a reply that is one long
 * block has nothing to hide.
 *
 * Plain text (a user's message, or a reply shown as text because it was not
 * lexed) has no blocks: its units are its lines, and it is cut between them.
 *
 * It imports types only, so it runs under Node's type stripping.
 */

import type { BlockNode, InlineNode, MdNode } from './tree';

export interface CollapseOptions {
	/** Weight above which a finished reply collapses; 0 never collapses. */
	threshold: number;
	/** Weight of whole blocks kept while collapsed. */
	keep: number;
	/** True once the reader asked for the rest. */
	expanded: boolean;
}

export interface Collapsed {
	/** The blocks to render, the input's own objects, in order. */
	shown: BlockNode[];
	/** How many blocks are hidden and not rendered. */
	hidden: number;
	/** True when the reply is long enough to hide some blocks. */
	collapsible: boolean;
}

const INLINE_KINDS: ReadonlySet<string> = new Set([
	'text',
	'strong',
	'em',
	'del',
	'code_inline',
	'br',
	'link'
]);

function isInline(node: MdNode): node is InlineNode {
	return INLINE_KINDS.has(node.kind);
}

function breaks(nodes: readonly InlineNode[]): number {
	let count = 0;
	for (const node of nodes) {
		if (node.kind === 'br') count += 1;
		else if ('children' in node) count += breaks(node.children);
	}
	return count;
}

function contentWeight(children: readonly MdNode[]): number {
	const inline = children.filter(isInline);
	let weight = inline.length > 0 ? 1 + breaks(inline) : 0;
	for (const child of children) {
		if (!isInline(child)) weight += blockWeight(child);
	}
	return Math.max(1, weight);
}

/** The lines of source a block stands for, at least one. */
export function blockWeight(node: MdNode): number {
	switch (node.kind) {
		case 'code_block':
			return node.code.split('\n').length + 2;
		case 'table':
			return node.rows.length + 2;
		case 'blockquote':
		case 'ol':
		case 'ul':
		case 'li':
			return contentWeight(node.children);
		case 'p':
		case 'heading':
			return 1 + breaks(node.children);
		default:
			return 1;
	}
}

/** The blocks to show, by whole blocks within the kept budget. */
export function collapseBlocks(blocks: readonly BlockNode[], options: CollapseOptions): Collapsed {
	const all: Collapsed = { shown: [...blocks], hidden: 0, collapsible: false };
	if (!(options.threshold > 0) || blocks.length < 2) return all;
	const weights = blocks.map(blockWeight);
	const total = weights.reduce((sum, weight) => sum + weight, 0);
	if (total <= options.threshold) return all;
	let count = 1;
	let used = weights[0];
	while (count < blocks.length && used + weights[count] <= options.keep) {
		used += weights[count];
		count += 1;
	}
	if (count >= blocks.length) return all;
	if (options.expanded) return { shown: [...blocks], hidden: 0, collapsible: true };
	return { shown: blocks.slice(0, count), hidden: blocks.length - count, collapsible: true };
}

export interface CollapsedText {
	/** The lines to show, joined. */
	shown: string;
	/** How many lines are hidden and not rendered. */
	hidden: number;
	/** True when the text is long enough to hide some lines. */
	collapsible: boolean;
}

/** Plain text over the threshold, in lines, shows its first ``keep`` lines. */
export function collapseLines(text: string, options: CollapseOptions): CollapsedText {
	const all: CollapsedText = { shown: text, hidden: 0, collapsible: false };
	if (!(options.threshold > 0)) return all;
	const lines = text.split('\n');
	if (lines.length <= options.threshold) return all;
	const keep = Math.max(1, Math.floor(options.keep));
	if (keep >= lines.length) return all;
	if (options.expanded) return { shown: text, hidden: 0, collapsible: true };
	return { shown: lines.slice(0, keep).join('\n'), hidden: lines.length - keep, collapsible: true };
}
