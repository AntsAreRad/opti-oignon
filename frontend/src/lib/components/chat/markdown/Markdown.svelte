<script context="module" lang="ts">
	let rendered = 0;
</script>

<script lang="ts">
	/**
	 * Markdown.svelte -- an assistant reply rendered from its markdown.
	 *
	 * The source is lexed by marked's lexer through lib/markdown/tree.ts, at most
	 * once per animation frame while the reply streams (lib/markdown/frame.ts)
	 * and at once otherwise, behind the frame's slow-lex guard; the tokens become
	 * a closed tree of plain nodes, with references in text decoded
	 * (lib/markdown/entities.ts); a long finished reply shows its first whole
	 * blocks and a toggle for the rest (lib/markdown/collapse.ts), and the hidden
	 * blocks are not rendered. MarkdownNode renders each node through text
	 * interpolation only. A reply with no tree -- its lexing would take too long,
	 * marked could not lex it, or the tree could not be built -- is shown as
	 * plain text (PlainText), never as an error. While the reply streams, the
	 * caret is drawn after its last character, inside its last block.
	 */
	import { onDestroy } from 'svelte';
	import { lexMarkdown, toTree } from '$lib/markdown/tree';
	import type { BlockNode } from '$lib/markdown/tree';
	import { decodeEntities } from '$lib/markdown/entities';
	import { collapseBlocks } from '$lib/markdown/collapse';
	import { createFrameLexer, guardSlowLex } from '$lib/markdown/frame';
	import TextButton from '$lib/ds/TextButton.svelte';
	import MarkdownNode from './MarkdownNode.svelte';
	import PlainText from './PlainText.svelte';
	import Caret from './Caret.svelte';

	/** The reply's markdown; it grows while the reply streams. Anything but a
	 * string reads as an empty reply. */
	export let source: string;
	/** True while the reply is still arriving: nothing collapses, and an
	 * unterminated last code fence is still open. */
	export let streaming = false;
	/** Weight (about one line of source) above which a finished reply
	 * collapses; 0 never collapses. */
	export let collapseAbove = 0;
	/** Weight of the whole blocks shown while collapsed. */
	export let collapseKeep = 20;
	/** True to draw the streaming caret after the reply's last character. */
	export let caret = false;

	rendered += 1;
	const regionId = `oo-md-${rendered}`;

	let tokens: unknown[] | null = [];
	let expanded = false;
	const lexer = createFrameLexer(guardSlowLex(lexMarkdown), (next: unknown[] | null) => {
		tokens = next;
	});

	$: text = typeof source === 'string' ? source : '';

	// Streaming: one lex at the next frame for the latest source. Otherwise
	// (a first render, the end of a stream): lex now.
	$: if (streaming) lexer.push(text);
	else lexer.now(text);

	$: blocks = build(tokens, streaming);
	$: view =
		blocks === null
			? null
			: collapseBlocks(blocks, {
					threshold: streaming ? 0 : collapseAbove,
					keep: collapseKeep,
					expanded
				});
	$: toggleLabel = expanded
		? 'Show less'
		: `Show the rest (${view?.hidden ?? 0} more ${view?.hidden === 1 ? 'block' : 'blocks'})`;

	/** The tree, or null when there are no tokens or it cannot be built. */
	function build(list: unknown[] | null, open: boolean): BlockNode[] | null {
		if (list === null) return null;
		try {
			return toTree(list, { streaming: open, decode: decodeEntities });
		} catch {
			return null;
		}
	}

	onDestroy(() => lexer.cancel());
</script>

{#if blocks === null}
	<div class="oo-md">
		<PlainText {text} collapseAbove={streaming ? 0 : collapseAbove} {collapseKeep} {caret} />
	</div>
{:else if view !== null}
	<div class="oo-md" id={regionId}>
		{#each view.shown as node, index}<MarkdownNode
				{node}
				caret={caret && index === view.shown.length - 1}
			/>{/each}{#if caret && view.shown.length === 0}<Caret />{/if}
	</div>
	{#if view.collapsible}
		<TextButton {expanded} controls={regionId} on:click={() => (expanded = !expanded)}
			>{toggleLabel}</TextButton
		>
	{/if}
{/if}

<style>
	.oo-md {
		font-family: var(--oo-font-serif);
		line-height: 1.6;
		overflow-wrap: anywhere;
	}

	.oo-md > :global(:last-child) {
		margin-bottom: 0;
	}
</style>
